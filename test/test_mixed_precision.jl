# Mixed-precision tests: fp32-geodesic / fp64-radiative-transfer two-type kernels
# for the fused unpolarized imager (gpu_image.jl :: _fused_kernel!), the 2-phase
# tracer (gpu_tracer.jl :: _trace_kernel!), the fused polarized imager
# (gpu_pol.jl :: _gpol_kernel!; the polarization-frame transport rides the Tgeo
# lane) and the native-AMR kernels (gpu_amr.jl), plus the critical-curve rescue
# mask. Runs the REAL kernel source on the KA CPU() backend — no CUDA device is
# touched. Check labels: G1 type-flow (a mismatched-precision scalar is a
# MethodError, never a silent promotion) · G3 rescue-mask geometry (α_cam =
# α_Bardeen convention) · G4 accumulator-from-output-type · G5 mixed-vs-fp64 accuracy.
using Test, StaticArrays, KernelAbstractions, Statistics, LinearAlgebra
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot
using BlackLightPlus.Geodesics: geodesic_rhs, geodesic_transport_rhs, rk4_step_fixed, ks_radius_energy

@isdefined(_fused_kernel!) || include(joinpath(@__DIR__, "..", "src", "gpu_image.jl"))
@isdefined(_trace_kernel!) || include(joinpath(@__DIR__, "..", "src", "geodesics", "gpu_tracer.jl"))
@isdefined(_gpol_kernel!) || include(joinpath(@__DIR__, "..", "src", "gpu_pol.jl"))
@isdefined(_gamr_sample) || include(joinpath(@__DIR__, "..", "src", "gpu_amr.jl"))

# KA CPU kernels may surface an in-kernel error wrapped in task/composite layers.
_mp_is_methoderror(e::MethodError) = true
_mp_is_methoderror(e::TaskFailedException) = _mp_is_methoderror(e.task.exception)
_mp_is_methoderror(e::CompositeException) = any(_mp_is_methoderror, e.exceptions)
_mp_is_methoderror(e::CapturedException) = _mp_is_methoderror(e.ex)
_mp_is_methoderror(e) = false
function _mp_throws_methoderror(f)
    try
        f()
        return false
    catch e
        _mp_is_methoderror(e) && return true
        rethrow()
    end
end

# G5 accuracy deck: torus with a GRMHD-like field, b = sqrt(σ_t ρ) d̂ (σ ≡ σ_t = 0.5
# wherever emission exists), so the σ=1 cut surface is never razor-grazed. A
# constant-|b| deck instead holds σ ≈ 1.0003 tangentially along the ray at the torus
# edge, where a ~1e-7 fp32 position shift flips the σ-cut over an EXTENDED segment
# (measured: two pixels at ~6e-2 rel) — a modeling-discontinuity sensitivity of that
# deck, not the Lyapunov channel the rescue targets.
function _mp_torus_smooth(n, box_half; sigma_t = 0.5)
    host = zeros(Float32, 8, n, n, n)
    dxg = 2box_half / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        x = -box_half + (i-1)*dxg; y = -box_half + (j-1)*dxg; z = -box_half + (k-1)*dxg
        r = sqrt(x^2 + y^2 + z^2)
        if 2.5 < r < 26.0
            rho = exp(-((r-8)/6)^2)
            host[1,i,j,k] = rho
            host[2,i,j,k] = -0.2*y/r; host[3,i,j,k] = 0.2*x/r; host[4,i,j,k] = 0.01
            host[5,i,j,k] = 0.08*rho
            d1 = 0.02; d2 = 0.03*x/r; d3 = 0.05
            dn = sqrt(d1^2 + d2^2 + d3^2)
            bmag = sqrt(sigma_t * rho)
            host[6,i,j,k] = bmag*d1/dn; host[7,i,j,k] = bmag*d2/dn; host[8,i,j,k] = bmag*d3/dn
        end
    end
    host
end

@testset "G1: geodesic lane runs at Tgeo; mismatched scalar is a MethodError" begin
    a64 = 0.9375
    a32 = Float32(a64)
    y32 = SVector{8,Float32}(500.0, 5.0, 5.0, 5.0, -1.0, 0.1, 0.1, 0.1)
    y64 = SVector{8,Float64}(y32)
    # state-lane types stay Float32 end to end (the Geodesics RHS / RK4 step the imager
    # and tracer kernels compile, on the lane-typed Kerr spacetime)
    st32 = KerrSchildCartesian(a32); st64 = KerrSchildCartesian(a64)
    @test eltype(geodesic_rhs(st32, y32)) === Float32
    @test typeof(ks_radius_energy(y32, a32)[1]) === Float32
    @test eltype(rk4_step_fixed(st32, y32, 0.01f0)) === Float32
    # fp64 lane types unchanged
    @test eltype(geodesic_rhs(st64, y64)) === Float64
    # G1: a Float64 scalar into an fp32 march is a MethodError (never silent fp64)
    @test_throws MethodError rk4_step_fixed(st32, y32, 0.01)
    @test_throws MethodError _gimg_stop(y32, Inf32, Float32(1.35), 550.0, 52.0f0, a32)
    @test_throws MethodError _gimg_stop(y32, Inf32, Float32(1.35), 550.0f0, 52.0f0, a64)
    @test_throws MethodError _gimg_march(y32, _ -> zero(SVector{8,Float64}), Float64,
                                         (nu_obs = 2.3e11, L_unit = 1.0), a64,
                                         Float32(1.35), 80.0f0, 550.0f0, Float32(Inf), 16)
    # G4: the march accumulator comes from the OUTPUT type, never Tgeo — an fp32
    # state with a Float64 accumulator returns Float64 (the AD twin relies on this)
    vac = _ -> zero(SVector{8,Float64})
    I = _gimg_march(y32, vac, Float64, (nu_obs = 2.3e11, L_unit = 1.0), a32,
                    Float32(1 + sqrt(1 - a64^2)), 80.0f0, 550.0f0, Float32(Inf), 16)
    @test typeof(I) === Float64
end

@testset "G1/G5: fused kernel two-type on the KA CPU backend (48² deck) + rescue" begin
    n = 32; box_half = 30.0; res = 48
    host = _mp_torus_smooth(n, box_half)
    a = 0.9375
    cfg = RunConfig(bh_a = a,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 30.0, resolution = res),
                    plasma = PlasmaConfig(thermal_frac = 1.0))
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    gp = _gimg_params(cfg)
    r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    xo = -box_half; inv_dx = (n - 1) / (2box_half); nr = length(rays)
    backend = CPU(); kern = _fused_kernel!(backend)

    img64 = Vector{Float64}(undef, nr)
    kern(img64, rays, host, gp, a, r_h, C, escape, R_em, nmax, n, xo, inv_dx, nr;
         ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test sum(img64) > 0.0

    # fp32 geodesic lane: fp64-built rays narrowed at upload, matched fp32 scalars,
    # Float64 output (accumulator from eltype(img) — G4)
    rays32 = [SVector{8,Float32}(r) for r in rays]
    img32 = Vector{Float64}(undef, nr)
    kern(img32, rays32, host, gp, Float32(a), Float32(r_h), Float32(C), Float32(escape),
         Float32(R_em), nmax, n, xo, inv_dx, nr; ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test all(isfinite, img32)

    # G1 at kernel level: fp32 rays with the fp64 spin must throw, not promote
    @test _mp_throws_methoderror() do
        bad = Vector{Float64}(undef, nr)
        kern(bad, rays32, host, gp, a, Float32(r_h), Float32(C), Float32(escape),
             Float32(R_em), nmax, n, xo, inv_dx, nr; ndrange = nr)
        KernelAbstractions.synchronize(backend)
    end

    # G3 rescue: critical curve in-frame ⇒ nonempty mask; fp64 subset re-render
    # splices EXACTLY (per-ray independence of the kernel)
    pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, res; rescue_px = 2)
    @test !isempty(pix)
    @test all(p -> 1 <= p <= nr, pix)
    @test issubset(_rescue_pixels(a, 30.0, 30.0, res; rescue_px = 1), pix)  # dilation monotone
    imgm = copy(img32)
    sub = Vector{Float64}(undef, length(pix))
    kern(sub, rays[pix], host, gp, a, r_h, C, escape, R_em, nmax, n, xo, inv_dx,
         length(pix); ndrange = length(pix))
    KernelAbstractions.synchronize(backend)
    imgm[pix] .= sub
    @test imgm[pix] == img64[pix]                       # spliced pixels == fp64 bitwise

    # G5 (measured: flux_rel 1.59e-5, corr 1 - 2e-10, NMSE 4.2e-10; the rescue
    # cuts NMSE ~17x — raw fp32 7.1e-9 — and max|Δ|/maxI 5.3e-4 -> 4.7e-5)
    flux_rel = abs(sum(imgm) - sum(img64)) / sum(img64)
    corr = cor(imgm, img64)
    nmse = sum(abs2, imgm .- img64) / sum(abs2, img64)
    @test flux_rel < 1e-4
    @test corr > 0.9999
    @test nmse < 1e-6
    @test nmse <= sum(abs2, img32 .- img64) / sum(abs2, img64)  # rescue never hurts
end

@testset "G3: rescue mask covers the capture boundary (screen-mapping convention)" begin
    # Backward-traced capture set == shadow, bounded by the critical curve. Trace the
    # SAME camera's rays (fp64 tracer kernel) and demand every captured pixel with an
    # escaped 4-neighbor lie inside the rescue mask even at rescue_px = 1 — this pins
    # the α_cam = α_Bardeen sign, the β symmetry, the pixel indexing AND the px scale
    # (a flipped α covers only ~40% of the boundary at this resolution; measured).
    a = 0.9375; res = 48; fov = 30.0; inc = 30.0
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(inc, 500.0, fov, res)
    rays = camera_rays(st, cam)
    nr = length(rays); r_h = 1 + sqrt(1 - a*a)
    nmax = 1600; C = 20.0
    traj = fill(zero(SVector{8,Float64}), nr, nmax + 1)
    hlog = zeros(Float64, nr, nmax); cnt = zeros(Int32, nr)
    _trace_kernel!(CPU())(traj, hlog, cnt, rays, a, C, 550.0, Inf, nmax, r_h, nr;
                          ndrange = nr)
    KernelAbstractions.synchronize(CPU())
    cap = falses(res, res)
    for i in 1:nr
        r, En = BlackLightPlus.Geodesics.ks_radius_energy(traj[i, Int(cnt[i]) + 1], a)
        cap[i] = BlackLightPlus.Geodesics.march_captured(r - r_h, En) || Int(cnt[i]) == nmax
    end
    @test 0 < count(cap) < nr                            # shadow in-frame, not everything
    mask1 = falses(res, res)
    mask1[_rescue_pixels(a, inc, fov, res; rescue_px = 1)] .= true
    nbnd = 0
    for iα in 1:res, iβ in 1:res
        cap[iβ, iα] || continue
        isbnd = any(((dβ, dα),) -> (1 <= iβ+dβ <= res && 1 <= iα+dα <= res &&
                                    !cap[iβ+dβ, iα+dα]),
                    ((1,0), (-1,0), (0,1), (0,-1)))
        isbnd || continue
        nbnd += 1
        @test mask1[iβ, iα]
    end
    @test nbnd > 20                                      # a real closed boundary was tested
    # a frame that misses the curve entirely rescues nothing
    @test isempty(_rescue_pixels(a, inc, 2.0, res; rescue_px = 2))
end

@testset "tracer kernel two-type (fp32 containers) — fp64 classification preserved" begin
    a = 0.9375
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(30.0, 500.0, 30.0, 16)
    rays = camera_rays(st, cam)
    nr = length(rays); nmax = 900; C = 20.0; escape = 550.0
    r_h = 1 + sqrt(1 - a*a)
    backend = CPU()
    function run(::Type{T}) where {T}
        rT = [SVector{8,T}(r) for r in rays]
        traj = fill(zero(SVector{8,T}), nr, nmax + 1)
        hlog = zeros(T, nr, nmax); cnt = zeros(Int32, nr)
        _trace_kernel!(backend)(traj, hlog, cnt, rT, T(a), T(C),
                                T(escape), T(Inf), nmax, T(r_h), nr; ndrange = nr)
        KernelAbstractions.synchronize(backend)
        traj, hlog, cnt
    end
    t64, h64, c64 = run(Float64)
    t32, h32, c32 = run(Float32)
    @test eltype(t32) === SVector{8,Float32} && eltype(h32) === Float32
    # trip counts and capture classification agree on this (non-critical) deck
    # (measured: max|Δcnt| = 0, zero flips, escaped-position rel dev median 4.9e-6)
    @test maximum(abs.(Int.(c32) .- Int.(c64))) <= 2
    cls(t, c, aT) = [ks_radius_energy(t[i, Int(c[i]) + 1], aT)[1] < (1 + sqrt(1 - a^2)) + 2e-3
                     for i in 1:nr]
    m64 = cls(t64, c64, a); m32 = cls(t32, c32, Float32(a))
    @test m64 == m32
    devs = [norm(SVector{3,Float64}(t32[i, Int(c32[i])+1][2], t32[i, Int(c32[i])+1][3],
                                    t32[i, Int(c32[i])+1][4]) -
                 SVector(t64[i, Int(c64[i])+1][2], t64[i, Int(c64[i])+1][3],
                         t64[i, Int(c64[i])+1][4])) /
            norm(SVector(t64[i, Int(c64[i])+1][2], t64[i, Int(c64[i])+1][3],
                         t64[i, Int(c64[i])+1][4]))
            for i in 1:nr if !(m64[i] || m32[i])]
    @test median(devs) < 1e-4
    # G1 at kernel level: fp32 rays + fp64 spin throws
    @test _mp_throws_methoderror() do
        r32 = [SVector{8,Float32}(r) for r in rays]
        traj = fill(zero(SVector{8,Float32}), nr, nmax + 1)
        hlog = zeros(Float32, nr, nmax); cnt = zeros(Int32, nr)
        _trace_kernel!(backend)(traj, hlog, cnt, r32, a, Float32(C), Float32(escape),
                                Float32(Inf), nmax, Float32(r_h), nr; ndrange = nr)
        KernelAbstractions.synchronize(backend)
    end
end

@testset "driver precision kwarg validation (no device touched)" begin
    a = 0.9375
    st = KerrSchildCartesian(a)
    rays = camera_rays(st, PlaneParallelCamera(30.0, 500.0, 10.0, 2))
    @test_throws ErrorException trace_rays_gpu(st, rays; devices = Int[], precision = :f16)
    fg = UniformFluidGrid(zeros(Float32, 8, 4, 4, 4), 4, 30.0, -30.0, 0.05)
    cfg = RunConfig(bh_a = a,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 30.0, resolution = 2))
    @test_throws ErrorException render_gpu_fused(cfg, fg; devices = Int[], precision = :f16)
end

# ================================================ polarized + AMR two-type kernels
# Fused polarized (gpu_pol.jl) + native-AMR (gpu_amr.jl) two-type kernels: the
# polarization-frame parallel transport rides the Tgeo lane with the geodesic
# (`geodesic_transport_rhs` runs in the state's number type);
# the Stokes/Mueller coupling and every coefficient stay Float64 from converted state.

# physical units for the polarized decks (the test_transfer.jl parity constants)
const _MP_M_BH = 6.2e9 * 1.989e33
const _MP_LU = BlackLightPlus.Constants.GNEWT * _MP_M_BH / BlackLightPlus.Constants.CL^2
_mp_pol_cfg(res) = RunConfig(bh_a = 0.9375, m_bh_cgs = _MP_M_BH,
                             mass_scale_cgs = 1.0e-21 * _MP_LU^3, frequency_hz = 2.3e11,
                             camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                                   fov = 30.0, resolution = res),
                             plasma = PlasmaConfig(thermal_frac = 1.0))

# Synthetic 2-level AMR mesh over [-32,32]^3: 7 level-0 blocks (size 32) + the
# (-x,-y,-z) octant refined into 8 level-1 blocks (size 16); finest/origin-aligned so
# the leaf table builds. Fluid = the SAME smooth torus (b = √(σ_t ρ) d̂) at each
# block's cell centres — the AMR twin of _mp_torus_smooth.
function _mp_amr_torus(; ncell = 16, box = 32.0, sigma_t = 0.5)
    blocks = Tuple{Int,Int,Int,Int}[]
    for l3 in 0:1, l2 in 0:1, l1 in 0:1
        (l1, l2, l3) == (0, 0, 0) || push!(blocks, (0, l1, l2, l3))
    end
    for l3 in 0:1, l2 in 0:1, l1 in 0:1
        push!(blocks, (1, l1, l2, l3))
    end
    nmb = length(blocks)
    data = zeros(Float32, ncell, ncell, ncell, nmb, 8)
    ext_lo = SVector{3,Float64}[]; ext_hi = SVector{3,Float64}[]
    x_left = SVector{3,Float64}[]; dxs = SVector{3,Float64}[]
    levels = Int[]; ll = zeros(Int, 3, nmb)
    for (m, (lev, l1, l2, l3)) in enumerate(blocks)
        bs = box / (1 << lev)
        lo = SVector(-box + l1*bs, -box + l2*bs, -box + l3*bs)
        dx = bs / ncell
        push!(ext_lo, lo); push!(ext_hi, lo .+ bs); push!(x_left, lo .+ dx/2)
        push!(dxs, SVector(dx, dx, dx)); push!(levels, lev)
        ll[:, m] .= (l1, l2, l3)
        for k in 1:ncell, j in 1:ncell, i in 1:ncell
            x = lo[1] + dx/2 + (i-1)*dx; y = lo[2] + dx/2 + (j-1)*dx; z = lo[3] + dx/2 + (k-1)*dx
            r = sqrt(x^2 + y^2 + z^2)
            if 2.5 < r < 26.0
                rho = exp(-((r-8)/6)^2)
                d1 = 0.02; d2 = 0.03*x/r; d3 = 0.05
                dn = sqrt(d1^2 + d2^2 + d3^2); bmag = sqrt(sigma_t * rho)
                data[i,j,k,m,1] = rho
                data[i,j,k,m,2] = -0.2*y/r; data[i,j,k,m,3] = 0.2*x/r; data[i,j,k,m,4] = 0.01
                data[i,j,k,m,5] = 0.08*rho
                data[i,j,k,m,6] = bmag*d1/dn; data[i,j,k,m,7] = bmag*d2/dn; data[i,j,k,m,8] = bmag*d3/dn
            end
        end
    end
    AthenaKSnapshot(data, ext_lo, ext_hi, x_left, dxs, levels, ll,
                    ["dens","velx","vely","velz","eint","bcc1","bcc2","bcc3"],
                    ncell, ncell, ncell, nmb)
end

# G5-pol metric helpers: per-Stokes flux rel + corr, and EVPA |Δχ| on LP pixels
function _mp_pol_g5(o64, om)
    fr = ntuple(s -> abs(sum(x[s] for x in om) - sum(x[s] for x in o64)) /
                     abs(sum(x[s] for x in o64)), 4)
    cr = ntuple(s -> cor([x[s] for x in om], [x[s] for x in o64]), 4)
    P0 = [hypot(x[2], x[3]) for x in o64]
    dchi = Float64[]
    for i in eachindex(o64)
        P0[i] > 1e-3 * maximum(P0) || continue
        push!(dchi, abs(rad2deg(rem(0.5 * atan(om[i][3], om[i][2]) -
                                    0.5 * atan(o64[i][3], o64[i][2]), pi, RoundNearest))))
    end
    return fr, cr, dchi
end

@testset "G1: polarized geodesic+frame lane runs at Tgeo" begin
    a64 = 0.9375; a32 = Float32(a64)
    y32 = SVector{8,Float32}(500.0, 5.0, 5.0, 5.0, -1.0, 0.1, 0.1, 0.1)
    y64 = SVector{8,Float64}(y32)
    e32 = SVector{4,Float32}(0.0, 0.0, 1.0, 0.0); u32 = SVector{4,Float32}(1.0, 0.0, 0.0, 0.0)
    e64 = SVector{4,Float64}(e32); u64 = SVector{4,Float64}(u32)
    st32 = KerrSchildCartesian(a32); st64 = KerrSchildCartesian(a64)
    dy, (de, du) = geodesic_transport_rhs(st32, y32, (e32, u32))
    @test eltype(dy) === Float32 && eltype(de) === Float32 && eltype(du) === Float32
    dy4, (de4, du4) = geodesic_transport_rhs(st64, y64, (e64, u64))   # fp64 lane unchanged
    @test eltype(dy4) === Float64 && eltype(de4) === Float64
    # the fp32 transport tracks the fp64 one to fp32 tolerance at the same input
    @test maximum(abs.(SVector{4,Float64}(de) .- de4)) <
          1e-4 * max(1.0, maximum(abs.(de4)))
    # G1: any mismatched lane member is a MethodError, never a silent promotion
    @test_throws MethodError geodesic_transport_rhs(st32, y32, (e64, u64))
    @test_throws MethodError geodesic_transport_rhs(st64, y64, (e32, u32))
    @test_throws MethodError _gpol_march(y32, _ -> zero(SVector{8,Float64}),
                                         _gpol_params(_mp_pol_cfg(2)), a64,
                                         Float32(1.35), 80.0f0, 550.0f0, Float32(Inf), 16)
    # the march: fp32 state + matched fp32 scalars — Stokes accumulator stays Float64
    gp = _gpol_params(_mp_pol_cfg(2))
    vac = _ -> zero(SVector{8,Float64})
    S = _gpol_march(y32, vac, gp, a32, Float32(1 + sqrt(1 - a64^2)), 80.0f0, 550.0f0,
                    Float32(Inf), 16)
    @test S isa SVector{4,Float64}
    @test _gpol_march(y64, vac, gp, a64, 1 + sqrt(1 - a64^2), 80.0, 550.0,
                      Inf, 16) isa SVector{4,Float64}
    # G1 at march level: fp32 state with the fp64 spin must throw
    @test_throws MethodError _gpol_march(y32, vac, gp, a64, Float32(1 + sqrt(1 - a64^2)),
                                         80.0f0, 550.0f0, Float32(Inf), 16)
end

@testset "G1/G5: fused polarized kernel two-type (KA CPU, 32² deck) + rescue" begin
    n = 32; box_half = 30.0; res = 32
    host = _mp_torus_smooth(n, box_half)
    a = 0.9375
    cfg = _mp_pol_cfg(res)
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(30.0, 500.0, 30.0, res)
    rays = camera_rays(st, cam)
    gp = _gpol_params(cfg)
    r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    xo = -box_half; inv_dx = (n - 1) / (2box_half); nr = length(rays)
    backend = CPU(); kern = _gpol_kernel!(backend)

    out64 = Vector{SVector{4,Float64}}(undef, nr)
    kern(out64, rays, host, gp, a, r_h, C, escape, R_em, nmax, n, xo, inv_dx, nr;
         ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test sum(o[1] for o in out64) > 0.0
    @test sum(abs(o[2]) + abs(o[3]) for o in out64) > 0.0      # genuinely polarized deck

    rays32 = [SVector{8,Float32}(r) for r in rays]
    out32 = Vector{SVector{4,Float64}}(undef, nr)
    kern(out32, rays32, host, gp, Float32(a), Float32(r_h), Float32(C), Float32(escape),
         Float32(R_em), nmax, n, xo, inv_dx, nr; ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test all(o -> all(isfinite, o), out32)

    # G1 at kernel level: fp32 rays with the fp64 spin must throw, not promote
    @test _mp_throws_methoderror() do
        bad = Vector{SVector{4,Float64}}(undef, nr)
        kern(bad, rays32, host, gp, a, Float32(r_h), Float32(C), Float32(escape),
             Float32(R_em), nmax, n, xo, inv_dx, nr; ndrange = nr)
        KernelAbstractions.synchronize(backend)
    end

    # rescue: fp64 subset re-render splices EXACTLY (per-ray independence)
    pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, res; rescue_px = 2)
    @test !isempty(pix)
    sub = Vector{SVector{4,Float64}}(undef, length(pix))
    kern(sub, rays[pix], host, gp, a, r_h, C, escape, R_em, nmax, n, xo, inv_dx,
         length(pix); ndrange = length(pix))
    KernelAbstractions.synchronize(backend)
    om = copy(out32); om[pix] .= sub
    @test om[pix] == out64[pix]                        # spliced pixels == fp64 bitwise

    # G5-pol (measured on the 48² deck: I flux 1.51e-5 corr 1-2e-10 NMSE 4.1e-10;
    # Q/U/V flux ≤ 4.4e-5 corr 1-3e-10; EVPA median 2.1e-4°, p99 2.9e-3°, max 1.9e-2°)
    fr, cr, dchi = _mp_pol_g5(out64, om)
    @test fr[1] < 1e-4                                 # I flux
    @test cr[1] > 0.9999                               # I corr
    @test all(fr[s] < 1e-3 for s in 2:4)               # Q,U,V flux
    @test all(cr[s] > 0.999 for s in 2:4)              # Q,U,V corr
    @test median(dchi) < 0.1                           # EVPA median (deg)
    @test quantile(dchi, 0.99) < 1.0                   # EVPA p99 (deg)
    I0 = [x[1] for x in out64]; Im = [x[1] for x in om]
    @test sum(abs2, Im .- I0) / sum(abs2, I0) < 1e-6   # NMSE(I)
end

@testset "AMR kernels two-type (KA CPU, 2-level mesh) + rescue splice" begin
    snap = _mp_amr_torus()
    blk = permutedims(snap.data, (5, 1, 2, 3, 4))
    xlft = reduce(hcat, snap.x_left); dxb = reduce(hcat, snap.dx)
    leaf, amrp = _build_leaf_table(snap)
    R_em = _gamr_domain_radius(amrp)
    a = 0.9375; r_h = 1 + sqrt(1 - a*a); escape = 550.0; nmax = 3000; C = 80.0
    st = KerrSchildCartesian(a)
    backend = CPU()

    # ---- unpolarized AMR kernel, 24² (_gimg_march is already two-type; this pins
    # the AMR sampler's Float64 conversion + the fp32 lane)
    res = 24
    rays = camera_rays(st, PlaneParallelCamera(30.0, 500.0, 30.0, res))
    nr = length(rays)
    gpi = _gimg_params(_mp_pol_cfg(res))
    ki = _famr_kernel!(backend)
    i64 = Vector{Float64}(undef, nr)
    ki(i64, rays, blk, xlft, dxb, leaf, amrp, gpi, a, r_h, C, escape, R_em, nmax, nr;
       ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test sum(i64) > 0.0
    rays32 = [SVector{8,Float32}(r) for r in rays]
    i32 = Vector{Float64}(undef, nr)
    ki(i32, rays32, blk, xlft, dxb, leaf, amrp, gpi, Float32(a), Float32(r_h), Float32(C),
       Float32(escape), Float32(R_em), nmax, nr; ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test all(isfinite, i32)
    pix = _rescue_pixels(a, 30.0, 30.0, res; rescue_px = 2)
    @test !isempty(pix)
    subI = Vector{Float64}(undef, length(pix))
    ki(subI, rays[pix], blk, xlft, dxb, leaf, amrp, gpi, a, r_h, C, escape, R_em, nmax,
       length(pix); ndrange = length(pix))
    KernelAbstractions.synchronize(backend)
    im = copy(i32); im[pix] .= subI
    @test im[pix] == i64[pix]
    @test abs(sum(im) - sum(i64)) / sum(i64) < 1e-4
    @test cor(im, i64) > 0.9999
    @test sum(abs2, im .- i64) / sum(abs2, i64) < 1e-6
    # G1 at kernel level
    @test _mp_throws_methoderror() do
        bad = Vector{Float64}(undef, nr)
        ki(bad, rays32, blk, xlft, dxb, leaf, amrp, gpi, a, Float32(r_h), Float32(C),
           Float32(escape), Float32(R_em), nmax, nr; ndrange = nr)
        KernelAbstractions.synchronize(backend)
    end

    # ---- polarized AMR kernel, 16² (measured on the 24² deck: I flux 9.7e-6,
    # corr 1-1e-10, NMSE 1.7e-10; Q/U/V flux ≤ 1.2e-5; EVPA median 1.6e-4°, max 6.5e-3°)
    resp = 16
    raysp = camera_rays(st, PlaneParallelCamera(30.0, 500.0, 30.0, resp))
    nrp = length(raysp)
    gpp = _gpol_params(_mp_pol_cfg(resp))
    kp = _gpol_amr_kernel!(backend)
    o64 = Vector{SVector{4,Float64}}(undef, nrp)
    kp(o64, raysp, blk, xlft, dxb, leaf, amrp, gpp, a, r_h, C, escape, R_em, nmax, nrp;
       ndrange = nrp)
    KernelAbstractions.synchronize(backend)
    @test sum(o[1] for o in o64) > 0.0
    raysp32 = [SVector{8,Float32}(r) for r in raysp]
    o32 = Vector{SVector{4,Float64}}(undef, nrp)
    kp(o32, raysp32, blk, xlft, dxb, leaf, amrp, gpp, Float32(a), Float32(r_h), Float32(C),
       Float32(escape), Float32(R_em), nmax, nrp; ndrange = nrp)
    KernelAbstractions.synchronize(backend)
    @test all(o -> all(isfinite, o), o32)
    pixp = _rescue_pixels(a, 30.0, 30.0, resp; rescue_px = 2)
    @test !isempty(pixp)
    subp = Vector{SVector{4,Float64}}(undef, length(pixp))
    kp(subp, raysp[pixp], blk, xlft, dxb, leaf, amrp, gpp, a, r_h, C, escape, R_em, nmax,
       length(pixp); ndrange = length(pixp))
    KernelAbstractions.synchronize(backend)
    omp = copy(o32); omp[pixp] .= subp
    @test omp[pixp] == o64[pixp]
    fr, cr, dchi = _mp_pol_g5(o64, omp)
    @test fr[1] < 1e-4 && cr[1] > 0.9999
    @test all(fr[s] < 1e-3 for s in 2:4) && all(cr[s] > 0.999 for s in 2:4)
    @test median(dchi) < 0.1 && quantile(dchi, 0.99) < 1.0
end

@testset "driver precision kwarg validation (no device touched)" begin
    cfg = _mp_pol_cfg(2)
    fg = UniformFluidGrid(zeros(Float32, 8, 4, 4, 4), 4, 30.0, -30.0, 0.05)
    @test_throws ErrorException render_gpu_polarized(cfg, fg; devices = Int[], precision = :f16)
    snap = _mp_amr_torus(ncell = 4)
    @test_throws ErrorException render_gpu_amr(cfg, snap; devices = Int[], precision = :f16)
    @test_throws ErrorException render_gpu_amr_polarized(cfg, snap; devices = Int[], precision = :f16)
end
