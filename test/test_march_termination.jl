# One termination rule for every fixed-control march (results/blp/multigpu/dag.md, MG04).
#
# A backward-traced ray (p = -k) that falls toward a hole approaches the future horizon
# asymptotically: its energy in the local normal frame, E_n = -k·n, grows without bound
# while r - r_h -> 0. Every march (CPU sampled tracers, the fused unpolarized, polarized,
# K-variant, AMR and generic-spacetime kernels, the device tracer) stops through the
# shared predicate `march_stop` (Geodesics, integrator.jl):
#   captured  <=>  d < HORIZON_EPS_SAMPLED  or  d non-finite  or  !(E_n <= CAPTURE_REDSHIFT)
#   escaped   <=>  (R > escape or R > R_em) and R > R_prev
# with d the distance above the (nearest) horizon. The redshift cut replaces the
# GPU-polarized frame-drift stop, so CPU and device marches integrate the same samples.
#
# Error bound of the cut. Beyond it the fluid-frame frequency is ν = (-k·u) ν_obs ≥
# E_c ν_obs / (γ(1+v)) ≥ E_c ν_obs / (2γ), γ = -u·n. The remaining affine length is
# Δλ_rest ≈ α_h / (κ E_c) (M = 1, camera E = 1; κ the surface gravity, α_h ≤ 1/√2 the
# KS lapse on the horizon), and the omitted invariant intensity obeys
#   ΔI_ν / ν_obs³ ≤ (L_unit / ν_obs) Σ_beyond j_inv ds ≤ (L_unit/ν_obs) Δλ_rest max_beyond j_inv.
# The tests below check the stop sample, the emitting-sample parity of CPU and device
# marches, the Δλ_rest scaling, the per-ray bound, and the flux change on a deck whose
# plasma reaches the horizon.
using Test, StaticArrays, LinearAlgebra, KernelAbstractions
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, SuperposedKerrSchild, PlaneParallelCamera,
                      camera_rays, trace_rays_sampled, horizon_radius, ks_radius,
                      rk4_step_fixed, ks_radius_hole, horizon_radius_hole
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Radiation: polarized_march, unpolarized_march, SimCoeffConfig,
                                eight_coefficients
using BlackLightPlus.Simulation: pgas_from_eint, units, primitive_four_vectors
const MT_GEO = BlackLightPlus.Geodesics

const _MT_SRC = joinpath(@__DIR__, "..", "src")
@isdefined(_gimg_thermal) || include(joinpath(_MT_SRC, "gpu_image.jl"))
@isdefined(_gpol_thermal) || include(joinpath(_MT_SRC, "gpu_pol.jl"))
@isdefined(_gsp_ctl)      || include(joinpath(_MT_SRC, "gpu_spacetime.jl"))
@isdefined(_trace_kernel!) || include(joinpath(_MT_SRC, "geodesics", "gpu_tracer.jl"))

# ------------------------------------------------------------------ deck
const MT_A = 0.9
const MT_M_BH = 6.2e9 * 1.989e33
const MT_L = BlackLightPlus.Constants.GNEWT * MT_M_BH / BlackLightPlus.Constants.CL^2
const MT_C = 80.0
const MT_NMAX = 6000
mt_cfg(; a = MT_A, res = 12, fov = 10.0) =
    RunConfig(bh_a = a, m_bh_cgs = MT_M_BH, mass_scale_cgs = 1.0e-21 * MT_L^3,
              frequency_hz = 2.3e11,
              camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = fov,
                                    resolution = res),
              plasma = PlasmaConfig(rat_high = 10.0, rat_low = 1.0, gamma = 13.0 / 9.0,
                                    use_p = true, sigma_cut = -1.0, theta_e_min = -1.0))

# Plasma that reaches the horizon: rho peaks at R = 3 with no inner cut.
function mt_prim(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    r < 0.5 && return zero(SVector{8,Float64})
    rho = exp(-((r - 3.0) / 2.5)^2)
    return SVector{8,Float64}(rho, -0.2y / r, 0.2x / r, -0.1, 0.08rho,
                              0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho))
end

function mt_grid(; n = 40, bh = 6.0)
    host = zeros(Float32, 8, n, n, n)
    dx = 2bh / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        host[:, i, j, k] .= Float32.(mt_prim(-bh + (i - 1) * dx, -bh + (j - 1) * dx,
                                             -bh + (k - 1) * dx))
    end
    return (host = host, n = n, xo = -bh, inv_dx = (n - 1) / (2bh), R_em = sqrt(3.0) * bh)
end

function mt_scc(cfg)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs); p = cfg.plasma
    return SimCoeffConfig(bh_a = cfg.bh_a, d_unit = u.dens_unit,
                          e_unit = u.dens_unit * BlackLightPlus.Constants.CL^2,
                          b_unit = u.B_unit, plasma_mu = p.mu, plasma_ne_ni = p.ne_ni,
                          plasma_use_p = p.use_p ? 1 : 0, plasma_rat_low = p.rat_low,
                          plasma_rat_high = p.rat_high, plasma_gamma = p.gamma,
                          plasma_gamma_i = p.gamma_i, plasma_gamma_e = p.gamma_e,
                          plasma_thermal_frac = p.thermal_frac,
                          cut_sigma_max = p.sigma_cut, cut_theta_e_min = p.theta_e_min), u
end

# CPU sampler over a primitive function prim(x,y,z)::SVector{8}.
mt_cpu_sampler(prim, γ) = x -> begin
    pr = prim(x[2], x[3], x[4])
    (rho = pr[1], pgas = pgas_from_eint(pr[5], γ), uu = (pr[2], pr[3], pr[4]),
     bb = (pr[6], pr[7], pr[8]), inside = pr[1] > 0.0)
end

# Device march body run on the host with a recording sampler: returns (result, samples),
# samples = the states the march sampled, camera -> source.
function mt_record(march, y0, prim)
    ys = SVector{8,Float64}[]
    out = march(y0, y -> (push!(ys, SVector{8,Float64}(y)); prim(Float64(y[2]), Float64(y[3]), Float64(y[4]))))
    return out, ys
end

corr(a, b) = (ma = sum(a) / length(a); mb = sum(b) / length(b);
              sum((a .- ma) .* (b .- mb)) / sqrt(sum(abs2, a .- ma) * sum(abs2, b .- mb)))

@testset "MG04: one march termination rule" begin

# ------------------------------------------------------------------ the predicate
@testset "shared predicate and constants" begin
    @test MT_GEO.HORIZON_EPS_SAMPLED === 1e-3
    @test MT_GEO.CAPTURE_REDSHIFT === 1.0e2
    cap = MT_GEO.march_captured
    @test !cap(0.5, 10.0)
    @test cap(0.5e-3, 10.0)                        # horizon backstop
    @test cap(0.5, 100.5)                          # redshift cut
    @test !cap(0.5, 100.0)                         # E_n = E_c still emits
    @test cap(NaN, 10.0) && cap(-Inf, 10.0) && cap(Inf, 10.0) && cap(0.5, NaN)
    @test cap(0.5f0, 101.0f0) && !cap(0.5f0, 99.0f0)            # Float32 lane
    @test !cap(0.0, 10.0, 0.0) && cap(-1e-300, 10.0, 0.0)      # explicit epsilon (ADM lapse)
    esc = MT_GEO.march_escaped
    @test esc(1200.0, 1100.0, 1100.0, Inf)
    @test !esc(1200.0, 1300.0, 1100.0, Inf)        # inbound beyond escape is not killed
    @test esc(12.0, 11.0, 1100.0, 11.5)            # R_em early-out, outbound only
    stop = MT_GEO.march_stop
    @test stop(0.5, 101.0, 5.0, 6.0, 1100.0, Inf)
    @test stop(0.5, 1.0, 1200.0, 1100.0, 1100.0, Inf)
    @test !stop(0.5, 1.0, 5.0, 6.0, 1100.0, Inf)
    @test_throws MethodError stop(0.5f0, 1.0, 5.0f0, 6.0f0, 1.1f3, Inf32)   # no silent promotion
    # the device wrapper: same answer, same radius, fp32 stays fp32
    st = KerrSchildCartesian(MT_A); r_h = horizon_radius(st)
    y = SVector(0.0, 1.2, 0.8, 0.3, 1.0, 0.4, -0.3, 0.2)
    r, En = MT_GEO.ks_radius_energy(y, MT_A)
    @test r == ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
    @test En ≈ MT_GEO.normal_frame_energy(st, y) rtol = 1e-14
    s, rr = _gimg_stop(y, Inf, r_h, 1100.0, Inf, MT_A)
    @test rr == r && s == stop(r - r_h, En, r, Inf, 1100.0, Inf)
    y32 = SVector{8,Float32}(y)
    s32, r32 = _gimg_stop(y32, Inf32, Float32(r_h), 1100.0f0, Inf32, Float32(MT_A))
    @test typeof(r32) === Float32
    @test_throws MethodError _gimg_stop(y32, Inf32, Float32(r_h), 1100.0, Inf32, Float32(MT_A))
end

# ------------------------------------------------------------------ b_sq guard
@testset "b² ≤ 0 is rejected identically on CPU and device" begin
    cfg = mt_cfg(); scc, u = mt_scc(cfg)
    st = KerrSchildCartesian(MT_A)
    x = SVector(0.0, 4.0, 1.0, 0.5)
    gcov = metric(st, x); gcon = inverse_metric(st, x)
    fv0 = primitive_four_vectors(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, gcov, gcon)
    kcov = SVector(-1.0, 0.3, 0.1, -0.2); kcon = gcon * kcov
    for b_sq in (0.0, -1e-30)
        fv = merge(fv0, (b_sq = b_sq,))
        j, α, ρ, tet, valid = eight_coefficients(scc, 1.0, 0.1, fv, gcov, gcon, kcov, kcon, 2.3e11)
        @test !valid && all(iszero, j)
    end
end

# ------------------------------------------------------------------ Kerr, near horizon
@testset "Kerr near-horizon deck: CPU and device marches stop on the same sample" begin
    cfg = mt_cfg()
    st = KerrSchildCartesian(MT_A); r_h = horizon_radius(st)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam); nr = length(rays)
    g = mt_grid()
    prim(x, y, z) = _gimg_sample(g.host, g.n, g.xo, g.inv_dx, x, y, z)
    escape = 1.1 * cfg.camera.distance
    gpu_u = _gimg_params(cfg); gpu_p = _gpol_params(cfg)
    scc, u = mt_scc(cfg)
    ν = cfg.frequency_hz

    states, dls = trace_rays_sampled(st, rays; C = MT_C, escape = escape, nmax = MT_NMAX,
                                     R_em = g.R_em)
    captured = [ks_radius(st, SVector(s[1][1], s[1][2], s[1][3], s[1][4])) < 2.0 for s in states]
    @test count(captured) > nr ÷ 2                 # the deck is dominated by plunging rays

    # (1) every captured ray is stopped by the redshift cut, one sample past E_c
    for i in findall(captured)
        s = states[i]                              # source -> camera
        @test !(MT_GEO.normal_frame_energy(st, s[1]) <= MT_GEO.CAPTURE_REDSHIFT)
        @test MT_GEO.normal_frame_energy(st, s[2]) <= MT_GEO.CAPTURE_REDSHIFT
        @test ks_radius(st, SVector(s[1][1], s[1][2], s[1][3], s[1][4])) - r_h >
              MT_GEO.HORIZON_EPS_SAMPLED
    end

    # (2) device march bodies sample exactly the CPU tracer's emitting states
    Iu = zeros(nr); Sp = Vector{SVector{4,Float64}}(undef, nr)
    same_u = 0; same_p = 0; maxdev = 0.0
    for i in 1:nr
        Iu[i], yu = mt_record((y0, f) -> _gimg_march(y0, f, Float64, gpu_u, MT_A, r_h, MT_C,
                                                     escape, g.R_em, MT_NMAX), rays[i], prim)
        Sp[i], yp = mt_record((y0, f) -> _gpol_march(y0, f, gpu_p, MT_A, r_h, MT_C,
                                                     escape, g.R_em, MT_NMAX), rays[i], prim)
        m = length(states[i])
        same_u += length(yu) == m - 1
        same_p += length(yp) == m - 1
        if length(yu) == m - 1
            for k in 1:m-1                          # device sample k == CPU state m+1-k
                maxdev = max(maxdev, maximum(abs.(yu[k] .- states[i][m+1-k]) ./
                                             max.(abs.(states[i][m+1-k]), 1.0)))
            end
        end
    end
    @test same_u == nr                             # same stop step, every ray
    @test same_p == nr                             # polarized: no extra frame-drift stop
    @test maxdev < 1e-8                            # same trajectory to RHS round-off

    # (3) the KA CPU kernels reproduce the host-run march bodies (unpolarized bit for
    # bit; polarized to round-off relative to the brightest pixel, the recording
    # closure changes its inlining and Faraday rotation amplifies ulps on dim pixels)
    backend = CPU()
    ku = zeros(nr)
    _fused_kernel!(backend)(ku, rays, g.host, gpu_u, MT_A, r_h, MT_C, escape, g.R_em,
                            MT_NMAX, g.n, g.xo, g.inv_dx, nr; ndrange = nr)
    kp = Vector{SVector{4,Float64}}(undef, nr)
    _gpol_kernel!(backend)(kp, rays, g.host, gpu_p, MT_A, r_h, MT_C, escape, g.R_em,
                           MT_NMAX, g.n, g.xo, g.inv_dx, nr; ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test ku == Iu
    @test maximum(norm(kp[i] - Sp[i]) for i in 1:nr) < 1e-9 * maximum(norm, Sp)

    # (4) image parity with the CPU marches (thresholds from the measured values)
    smp = mt_cpu_sampler(prim, cfg.plasma.gamma)
    up = BlackLightPlus._camera_up(cam)
    cpuU = [unpolarized_march(st, scc, smp, states[i], dls[i], ν, u.L_unit) for i in 1:nr]
    cpuP = [polarized_march(st, scc, smp, states[i], dls[i], ν, u.L_unit; cam_up = up)
            for i in 1:nr]
    @test corr(Iu, cpuU) > 1 - 1e-12
    @test abs(sum(Iu) / sum(cpuU) - 1) < 1e-7
    for s in 1:4
        @test corr([o[s] for o in Sp], [c[s] for c in cpuP]) > 0.99999
    end
    @test abs(sum(o[1] for o in Sp) / sum(c[1] for c in cpuP) - 1) < 5e-3
    # the device polarized I and unpolarized I now integrate the same samples
    @test corr([o[1] for o in Sp], Iu) > 0.99999

    # (5) device tracer: same stop step as the CPU tracer
    traj = Matrix{SVector{8,Float64}}(undef, nr, MT_NMAX + 1)
    hlog = Matrix{Float64}(undef, nr, MT_NMAX); cnt = zeros(Int32, nr)
    _trace_kernel!(backend)(traj, hlog, cnt, rays, MT_A, MT_C, escape, g.R_em, MT_NMAX,
                            r_h, nr; ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test all(cnt[i] == length(states[i]) - 1 for i in 1:nr)
end

# ------------------------------------------------------------------ frame fidelity
# The polarized march parallel-transports the camera frame; the invariant -k·e_u must
# stay at its camera value. Up to the redshift cut it drifts by far less than the 1%
# that ended the old GPU-only stop, in fp64 and in the fp32 (:mixed) lane, so that
# stop can never fire first (measured max drift: 5.2e-3 fp64, see the assertions).
function mt_frame_drift(y0::SVector{8,Tg}, gp, a::Tg, r_h::Tg, escape::Tg, R_em::Tg) where {Tg}
    y64 = SVector{8,Float64}(y0)
    x0 = SVector(y64[1], y64[2], y64[3], y64[4]); kcov0 = -SVector(y64[5], y64[6], y64[7], y64[8])
    gcov0, gcon0 = _gimg_metric(x0, gp.bh_a); kcon0 = gcon0 * kcov0
    e0c = _gpol_normal_observer(gcon0); e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = _gpol_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0)
    eU = SVector{4,Tg}(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])
    eUu = SVector{4,Tg}(e0c); wcam = -dot(kcov0, e0c)
    y = y0; r_prev = Tg(Inf); worst = 0.0; C = Tg(MT_C)
    for _ in 1:MT_NMAX
        stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
        stop && break
        r_prev = r; h = (r - r_h) / C
        kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])
        worst = max(worst, abs(-dot(kcov, SVector{4,Float64}(eUu)) - wcam) / wcam)
        dy, de, du = _gpol_ext_rhs2(y, eU, eUu, a); ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a); ay += 2dy; ae += 2de; au += 2du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a); ay += 2dy; ae += 2de; au += 2du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a); ay += dy; ae += de; au += du
        y += (h/6)*ay; eU += (h/6)*ae; eUu += (h/6)*au
    end
    return worst
end

@testset "frame fidelity up to the redshift cut (fp64 and fp32 lanes)" begin
    cfg = mt_cfg(res = 16)
    st = KerrSchildCartesian(MT_A); r_h = horizon_radius(st)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam); gp = _gpol_params(cfg)
    esc = 1.1 * cfg.camera.distance
    d64 = [mt_frame_drift(y, gp, MT_A, r_h, esc, Inf) for y in rays]
    d32 = [mt_frame_drift(SVector{8,Float32}(y), gp, Float32(MT_A), Float32(r_h),
                          Float32(esc), Inf32) for y in rays]
    @test maximum(d64) < 1e-2
    @test maximum(d32) < 1e-2
end

# ------------------------------------------------------------------ error bound
@testset "emission beyond the redshift cut: bound and flux change" begin
    cfg = mt_cfg()
    st = KerrSchildCartesian(MT_A); r_h = horizon_radius(st)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam); nr = length(rays)
    g = mt_grid()
    prim(x, y, z) = _gimg_sample(g.host, g.n, g.xo, g.inv_dx, x, y, z)
    smp = mt_cpu_sampler(prim, cfg.plasma.gamma)
    scc, u = mt_scc(cfg); ν = cfg.frequency_hz
    escape = 1.1 * cfg.camera.distance
    states, dls = trace_rays_sampled(st, rays; C = MT_C, escape = escape, nmax = MT_NMAX,
                                     R_em = g.R_em)
    # surface gravity; the KS lapse on the horizon is at most 1/√2 (poles, r_h²+a² = 2r_h)
    rm = 1 - sqrt(1 - MT_A^2)
    κ = (r_h - rm) / (2 * (r_h^2 + MT_A^2))
    Ec = MT_GEO.CAPTURE_REDSHIFT
    Δλ_est = 1 / (sqrt(2.0) * κ * Ec)
    F_cut = 0.0; F_ext = 0.0; worst_rest = 0.0
    for i in 1:nr
        s = states[i]
        Icut = unpolarized_march(st, scc, smp, s, dls[i], ν, u.L_unit)
        F_cut += Icut
        # continue the same ray past the cut with the horizon backstop only
        y = s[1]; ext = SVector{8,Float64}[]; hs = Float64[]
        r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
        if r < 2.0
            for _ in 1:MT_NMAX
                r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
                (r - r_h < MT_GEO.HORIZON_EPS_SAMPLED || !isfinite(r)) && break
                h = (r - r_h) / MT_C
                y = rk4_step_fixed(st, y, h); push!(ext, y); push!(hs, h)
            end
        end
        rest = sum(hs; init = 0.0)
        worst_rest = max(worst_rest, rest)
        # extended trajectory, source -> camera
        ys = vcat(reverse(ext), s)
        dl = vcat(reverse(hs), dls[i])
        Iext = unpolarized_march(st, scc, smp, ys, dl, ν, u.L_unit)
        F_ext += Iext
        # per-ray bound: ΔI ≤ ν³ (L/ν) Σ_beyond j_inv ds  (transmission ≤ 1, B ≤ j ds)
        jsum = 0.0
        for k in eachindex(ext)
            x = SVector(ext[k][1], ext[k][2], ext[k][3], ext[k][4]); sm = smp(x)
            sm.inside || continue
            gcov = metric(st, x); gcon = inverse_metric(st, x)
            kcov = -SVector(ext[k][5], ext[k][6], ext[k][7], ext[k][8])
            fv = primitive_four_vectors(sm.uu..., sm.bb..., gcov, gcon)
            j4, _, _, _, ok = eight_coefficients(scc, sm.rho, sm.pgas, fv, gcov, gcon,
                                                 kcov, gcon * kcov, ν)
            ok && (jsum += j4[1] * hs[k])
        end
        # the cut sample itself couples over the first omitted step
        x = SVector(s[1][1], s[1][2], s[1][3], s[1][4]); sm = smp(x)
        if !isempty(hs) && sm.inside
            gcov = metric(st, x); gcon = inverse_metric(st, x)
            kcov = -SVector(s[1][5], s[1][6], s[1][7], s[1][8])
            fv = primitive_four_vectors(sm.uu..., sm.bb..., gcov, gcon)
            j4, _, _, _, ok = eight_coefficients(scc, sm.rho, sm.pgas, fv, gcov, gcon,
                                                 kcov, gcon * kcov, ν)
            ok && (jsum += j4[1] * hs[1])
        end
        bound = ν^3 * (u.L_unit / ν) * jsum
        @test Iext - Icut <= bound * (1 + 1e-12) + 1e-300
    end
    @test worst_rest > 0                           # some rays did continue past the cut
    @test worst_rest <= 1.05 * Δλ_est              # Δλ_rest ≈ α_h / (κ E_c)
    @test abs(F_ext / F_cut - 1) < 1e-8            # negligible on a plasma-to-horizon deck
end

# ------------------------------------------------------------------ superposed binary
@testset "superposed binary: CPU tracer and generic kernels stop on the same sample" begin
    cfg = mt_cfg(res = 10, fov = 16.0)
    st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam); nr = length(rays)
    escape = 1.1 * cfg.camera.distance; R_em = 12.0
    states, dls = trace_rays_sampled(st, rays; C = MT_C, escape = escape, nmax = MT_NMAX,
                                     R_em = R_em)
    gpu_u = _gimg_params(cfg); gpu_p = _gpol_params(cfg)
    scc, u = mt_scc(cfg); ν = cfg.frequency_hz
    smp = mt_cpu_sampler(mt_prim, cfg.plasma.gamma)
    up = BlackLightPlus._camera_up(cam)
    Iu = zeros(nr); Sp = Vector{SVector{4,Float64}}(undef, nr)
    same_u = 0; same_p = 0; ncap = 0
    for i in 1:nr
        Iu[i], yu = mt_record((y0, f) -> _gsp_march(y0, f, gpu_u, st, MT_C, escape, R_em,
                                                    MT_NMAX), rays[i], mt_prim)
        Sp[i], yp = mt_record((y0, f) -> _gsp_pol_march(y0, f, gpu_p, st, MT_C, escape,
                                                        R_em, MT_NMAX), rays[i], mt_prim)
        m = length(states[i])
        same_u += length(yu) == m - 1
        same_p += length(yp) == m - 1
        ncap += first(superposed_control(st, states[i][1], MT_C))
    end
    @test ncap > 0
    @test same_u == nr
    @test same_p == nr
    cpuU = [unpolarized_march(st, scc, smp, states[i], dls[i], ν, u.L_unit) for i in 1:nr]
    cpuP = [polarized_march(st, scc, smp, states[i], dls[i], ν, u.L_unit; cam_up = up)
            for i in 1:nr]
    @test corr(Iu, cpuU) > 1 - 1e-12
    @test abs(sum(Iu) / sum(cpuU) - 1) < 1e-7
    for s in 1:4
        @test corr([o[s] for o in Sp], [c[s] for c in cpuP]) > 0.9999
    end
end

end
