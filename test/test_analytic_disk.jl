# Polarized regression on a semi-analytic thick-disk model re-derived from the
# §3.3 equations (3.10-3.14) of arXiv:2407.10431 (Huang, Zheng, Guo & Chen):
# marginally-bound (E=1, L=Q=0) ingoing flow, n_e/T_e power laws, split-monopole
# b^mu, a=0.94, i=17 deg, nu=230 GHz. The BL closed forms are transformed
# BL -> KS-spherical -> KS-Cartesian and fed to Radiation.polarized_march. The
# paper reports images, not integrated Stokes scalars, so the checks are physical
# (finite I>0, per-pixel LP band, ordered EVPA, nonzero |V| from the sign(cos θ)
# field flip) plus in-test transform verification (u·u=-1, b·u=0, round-trips).
# GRMHD-free; the GPU stiffness block is skipped without CUDA or if BLP_SKIP_GPU=1.

module TestAnalyticDisk  # isolate consts from Main (suite-include-safe)
using Test, StaticArrays, LinearAlgebra, Printf
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays,
                    trace_rays_sampled, metric, inverse_metric, ks_radius, horizon_radius,
                    SimCoeffConfig
const Rad  = BlackLightPlus.Radiation
const Sim  = BlackLightPlus.Simulation
const Cst  = BlackLightPlus.Constants

# ---------------------------------------------------------------- physical constants (match Cst)
const MP = Cst.MP; const ME = Cst.ME; const CL = Cst.CL; const KB = Cst.KB
const GNEWT = Cst.GNEWT; const MSUN = Cst.MSUN

# ------------------------------------ thick-disk parameters (arXiv:2407.10431 §3.3)
const A_SPIN  = 0.94
const R_H     = 1 + sqrt(1 - A_SPIN^2)              # 1.34117 (Δ(r₊)=0)
const N0      = 1.0e6                                # cm⁻³
const THETA_J = π/2 - 1.0e-3
const SIG     = 0.2
const T0      = 1.0e12                               # K
const ZPAR    = 20.0
const TE_EXP  = (1 + ZPAR) / (3 * (2 + ZPAR))        # 21/66
const B0      = 10.0                                 # Gauss
const NU_OBS  = 230.0e9                              # Hz
const INC_DEG = 17.0
# The paper does not state M_BH; Sgr A* (4.14e6 Msun) is adopted, consistent with
# n0=1e6 cm^-3, B0=10 G, T0=1e12 K. M_BH only sets L_unit (optical/Faraday depth);
# the plane-parallel camera makes the image observer-distance independent far-field.
const M_BH    = 4.14e6 * MSUN
const L_UNIT  = GNEWT * M_BH / CL^2                   # cm per GM/c²
const R_OUT   = 50.0                                 # emission confinement radius (n_e ∝ 1/r³)

# ---------------------------------------------------------------- Boyer–Lindquist closed forms
"Radial potential R(r,θ)=2r(r²+a²) for the marginally-bound (E=1,L=0,Q=0) flow (Eq. 3.11)."
@inline _Rrad(r) = 2r * (r^2 + A_SPIN^2)

"BL contravariant 4-velocity (u^t,u^r,u^θ,u^φ) of the ingoing marginally-bound flow (Eq. 3.10)."
@inline function _u_bl(r, cθ)
    Σ = r^2 + A_SPIN^2 * cθ^2
    Δ = r^2 - 2r + A_SPIN^2
    ut = 1 + 2r*(r^2 + A_SPIN^2)/(Δ*Σ)
    ur = -sqrt(_Rrad(r))/Σ                            # σ_r = −1 (ingoing)
    uφ = 2*A_SPIN*r/(Δ*Σ)
    return SVector(ut, ur, 0.0, uφ), Σ, Δ
end

"BL magnetic 4-vector b^μ = −sign(cosθ)·Ψ₀/(Σu^r)·(u_t u^μ + δ^μ_t), u_t=−1 (Eq. 3.14)."
@inline function _b_bl(r, cθ, ubl::SVector{4}, Σ, ψ0)
    ur = ubl[2]
    c  = -sign(cθ) * ψ0 / (Σ * ur)
    ut_cov = -1.0
    return SVector(c*(ut_cov*ubl[1] + 1.0), c*(ut_cov*ubl[2]), 0.0, c*(ut_cov*ubl[4]))
end

"Horizon-stable |b| via the closed form |b|²=c²(u_t²+g_tt) (u_t=−1); used to fix Ψ₀ at r₊."
@inline function _bmag_bl(r, cθ, ψ0)
    Σ  = r^2 + A_SPIN^2 * cθ^2
    ur = -sqrt(_Rrad(r))/Σ
    c  = -sign(cθ) * ψ0 / (Σ * ur)
    gtt = -(1 - 2r/Σ)
    return abs(c) * sqrt(max(1.0 + gtt, 0.0))
end
const PSI0 = B0 / _bmag_bl(R_H, cos(THETA_J), 1.0)    # |b(r₊,θ_J)| = B₀

# ---------------------------------------------------------------- BL → Kerr-Schild-Cartesian
# A contravariant 4-vector V_BL=(t,r,θ,φ) → KS-spherical (dt_ks=dt_bl+2r/Δ dr, dφ_ks=dφ_bl+a/Δ dr)
# → KS-Cartesian (X+iY=(r+ia)sinθ e^{iφ}, Z=r cosθ; the BlackLightPlus _rfl convention).
@inline function _bl_to_kscart(V::SVector{4}, r, θ, φ, Δ)
    Vt_ks = V[1] + (2r/Δ) * V[2]
    Vr    = V[2]; Vθ = V[3]
    Vφ_ks = V[4] + (A_SPIN/Δ) * V[2]
    sθ, cθ = sin(θ), cos(θ); sφ, cφ = sin(φ), cos(φ)
    dXdr = sθ*cφ; dXdθ = cθ*(r*cφ - A_SPIN*sφ); dXdφ = sθ*(-r*sφ - A_SPIN*cφ)
    dYdr = sθ*sφ; dYdθ = cθ*(r*sφ + A_SPIN*cφ); dYdφ = sθ*( r*cφ - A_SPIN*sφ)
    dZdr = cθ;    dZdθ = -r*sθ;                 dZdφ = 0.0
    return SVector(Vt_ks,
                   dXdr*Vr + dXdθ*Vθ + dXdφ*Vφ_ks,
                   dYdr*Vr + dYdθ*Vθ + dYdφ*Vφ_ks,
                   dZdr*Vr + dZdθ*Vθ + dZdφ*Vφ_ks)
end

# ---------------------------------------------------------------- the fluid sampler
# Returns (rho, pgas, uu, bb, inside) in the polarized_march contract, KS-Cartesian frame.
# rho/pgas ENCODE n_e/θ_e exactly under the unit map below (verified in-test); uu is the
# normal-frame velocity and bb the lab 3-field, so primitive_four_vectors reconstructs u^μ,b^μ.
function disk_sample(st, x::SVector{4})
    r = ks_radius(st, x)
    (r <= R_H + 1.0e-4 || r > R_OUT) && return (rho=0.0, pgas=0.0, uu=(0.0,0.0,0.0),
                                                bb=(0.0,0.0,0.0), inside=false)
    cθ = clamp(x[4]/r, -1.0, 1.0)
    θ  = acos(cθ)
    φ  = atan(r*x[3] - A_SPIN*x[2], r*x[2] + A_SPIN*x[3])   # from l₂,l₃ of _rfl

    ratio  = _Rrad(R_H) / _Rrad(r)                          # R(r₊)/R(r)  (θ-independent, E=1)
    ne     = N0 * exp(-((sin(θ) - sin(THETA_J))/SIG)^2) * ratio
    Te     = T0 * ratio^TE_EXP
    thetae = KB * Te / (ME * CL^2)

    ubl, Σ, Δ = _u_bl(r, cθ)
    bbl = _b_bl(r, cθ, ubl, Σ, PSI0)
    ucart = _bl_to_kscart(ubl, r, θ, φ, Δ)
    bcart = _bl_to_kscart(bbl, r, θ, φ, Δ)

    gcov = metric(st, x); gcon = inverse_metric(st, x)
    u0   = ucart[1]
    sh1 = -gcon[1,2]/gcon[1,1]; sh2 = -gcon[1,3]/gcon[1,1]; sh3 = -gcon[1,4]/gcon[1,1]
    uu = (ucart[2] + sh1*u0, ucart[3] + sh2*u0, ucart[4] + sh3*u0)   # normal-frame velocity
    b0 = bcart[1]
    bb = (bcart[2]*u0 - b0*ucart[2], bcart[3]*u0 - b0*ucart[3], bcart[4]*u0 - b0*ucart[4])  # lab B^i

    rho  = 2 * MP * ne                                      # → n_e_cgs = ne  (unit map)
    pgas = thetae * (ME * CL^2) * rho / MP                  # → θ_e = thetae  (unit map)
    return (rho=rho, pgas=pgas, uu=uu, bb=bb, inside=true)
end

# Unit map: d_unit=e_unit=b_unit=1, μ=1, n_e/n_i=1, R_high=R_low=1, use_p ⇒ n_e_cgs=rho/(2MP),
# θ_e = MP·pgas/(rho·m_e c²); all plasma cuts disabled (thermal-only, no σ/θ_e gating).
function disk_scc()
    return SimCoeffConfig(bh_a=A_SPIN, d_unit=1.0, e_unit=1.0, b_unit=1.0,
                          plasma_mu=1.0, plasma_ne_ni=1.0, plasma_use_p=1,
                          plasma_rat_low=1.0, plasma_rat_high=1.0,
                          plasma_gamma=13.0/9.0, plasma_thermal_frac=1.0,
                          cut_sigma_max=-1.0, cut_theta_e_min=-1.0)
end

"KS-Cartesian position of BL/KS (r,θ,φ): X+iY=(r+ia)sinθ e^{iφ}, Z=r cosθ (BlackLightPlus frame)."
function _kscart_of(r, θ, φ)
    sθ = sin(θ)
    return SVector(0.0, sθ*(r*cos(φ) - A_SPIN*sin(φ)), sθ*(r*sin(φ) + A_SPIN*cos(φ)), r*cos(θ))
end

# ---------------------------------------------------------------- the image
function render_disk(; npix=64, fov=600*tan(π/96), dist=100.0)
    st  = KerrSchildCartesian(A_SPIN)
    cam = PlaneParallelCamera(INC_DEG, dist, fov, npix)
    rays = camera_rays(st, cam)
    states, dls = trace_rays_sampled(st, rays; C=80.0, escape=1.1*dist, R_em=R_OUT)
    scc = disk_scc()
    i = deg2rad(INC_DEG)
    up = SVector(0.0, -cos(i), 0.0, sin(i))                # image-β vertical (matches render.jl)
    n = length(rays)
    out = Matrix{Float64}(undef, 4, n)
    Threads.@threads for k in 1:n
        s4 = Rad.polarized_march(st, scc, x -> disk_sample(st, x), states[k], dls[k],
                                 NU_OBS, L_UNIT; cam_up=up)
        out[:, k] .= s4
    end
    shp(j) = reshape(out[j, :], npix, npix)
    return (I=shp(1), Q=shp(2), U=shp(3), V=shp(4))
end

# ============================================================================================
@testset "Analytic-disk polarized regression + carve-outs" begin

    st = KerrSchildCartesian(A_SPIN)

    @testset "model self-consistency (BL fields + BL→KS-Cart transform)" begin
        scc = disk_scc()
        # sample a handful of in-disk points spanning both hemispheres and radii
        for (r, θ, φ) in [(3.0, π/2-0.15, 0.4), (6.0, π/2+0.2, 2.1),
                          (2.0, π/2-0.05, -1.3), (12.0, π/2+0.1, 3.0)]
            x = _kscart_of(r, θ, φ)
            # coordinate round-trip: (X,Y,Z) → (r,θ,φ) recovers the inputs
            @test ks_radius(st, x) ≈ r rtol=1e-10
            @test acos(clamp(x[4]/r,-1,1)) ≈ θ rtol=1e-9
            @test mod(atan(r*x[3]-A_SPIN*x[2], r*x[2]+A_SPIN*x[3]) - φ, 2π) < 1e-8 ||
                  mod(φ - atan(r*x[3]-A_SPIN*x[2], r*x[2]+A_SPIN*x[3]), 2π) < 1e-8

            s = disk_sample(st, x)
            @test s.inside
            gcov = metric(st, x); gcon = inverse_metric(st, x)
            fv = Sim.primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3], s.bb[1], s.bb[2], s.bb[3],
                                            gcov, gcon)
            # transformed 4-velocity is a valid future-pointing normalized fluid velocity
            @test dot(fv.ucon, gcov*fv.ucon) ≈ -1.0 rtol=1e-8
            @test fv.ucon[1] > 0
            # covariant energy u_t = −E = −1 (conserved along the marginally-bound flow)
            @test (gcov*fv.ucon)[1] ≈ -1.0 rtol=1e-6
            # magnetic 4-vector is spacelike and orthogonal to u (b·u = 0)
            @test dot(fv.bcon, gcov*fv.ucon) ≈ 0.0 atol=1e-6*sqrt(fv.b_sq)
            @test fv.b_sq > 0
            # unit map reproduces the intended n_e and θ_e EXACTLY
            pd = Rad.plasma_decode(scc, s.rho, s.pgas, fv.b_sq)
            ratio = _Rrad(R_H)/_Rrad(r)
            ne_want  = N0 * exp(-((sin(θ)-sin(THETA_J))/SIG)^2) * ratio
            the_want = KB * (T0*ratio^TE_EXP) / (ME*CL^2)
            @test pd.n_e_cgs ≈ ne_want rtol=1e-10
            @test pd.theta_e ≈ the_want rtol=1e-10
        end
        # B-field normalization: Ψ₀ fixes |b(r₊,θ_J)| = B₀ = 10 G (closed-form, horizon-stable)
        @test _bmag_bl(R_H, cos(THETA_J), PSI0) ≈ B0 rtol=1e-12
        # the BL→KS-Cart transform PRESERVES the field magnitude |b| (invariant scalar):
        # sqrt(b·b) from the reconstructed KS-Cart b^μ matches the analytic BL |b| at (r,θ)
        for (r, θ, φ) in [(2.0, π/2-0.05, 0.3), (7.0, π/2+0.15, 1.7)]
            x = _kscart_of(r, θ, φ); s = disk_sample(st, x)
            gcov = metric(st, x); gcon = inverse_metric(st, x)
            fv = Sim.primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3], s.bb[1], s.bb[2], s.bb[3],
                                            gcov, gcon)
            @test sqrt(fv.b_sq) ≈ _bmag_bl(r, cos(θ), PSI0) rtol=1e-6
        end
        # sign(cosθ) flip ⇒ B reverses across the equatorial plane (V-generating structure)
        xa = _kscart_of(5.0, π/2-0.25, 0.7); xb = _kscart_of(5.0, π/2+0.25, 0.7)
        ba = disk_sample(st, xa).bb; bb_ = disk_sample(st, xb).bb
        @test sign(ba[3]) != sign(bb_[3]) || sign(ba[1]) != sign(bb_[1])
    end

    @testset "polarized image — physical gates" begin
        img = render_disk()
        Itot = sum(img.I); Qtot = sum(img.Q); Utot = sum(img.U); Vtot = sum(img.V)
        # finite, positive Stokes I everywhere (stokes_couple enforces I ≥ 0, |P| ≤ I)
        @test all(isfinite, img.I) && all(isfinite, img.Q) &&
              all(isfinite, img.U) && all(isfinite, img.V)
        @test Itot > 0
        @test all(>=(0.0), img.I)
        @test maximum(img.I) > 0
        # image-integrated linear polarization: finite and below the intrinsic synchrotron
        # ceiling. For this idealized split-monopole disk viewed near face-on (i=17°) the
        # ordered field yields a symmetric EVPA pattern whose global vector sum LARGELY
        # CANCELS, so the *net* image LP is small — the polarization lives per-pixel.
        LP = hypot(Qtot, Utot) / Itot
        @test 0.0 <= LP < 0.75
        # per-pixel emission is strongly polarized (Faraday-thin: near the intrinsic ceiling)
        thr = 0.05 * maximum(img.I)
        mask = img.I .>= thr
        lp_pix = [hypot(img.Q[i], img.U[i]) / img.I[i] for i in eachindex(img.I) if img.I[i] > thr]
        lp_med = isempty(lp_pix) ? 0.0 : sort(lp_pix)[max(1, length(lp_pix) ÷ 2)]
        lp_max = isempty(lp_pix) ? 0.0 : maximum(lp_pix)
        @test 0.02 < lp_med < 0.76
        # circular polarization IS generated (sign(cosθ) field flip). Use the UNSIGNED map
        # sum|V|/sum(I): the signed net V ~ cancels across the equator for a symmetric disk.
        CP = sum(abs, img.V) / Itot
        @test CP > 1e-3
        # per-pixel |P| ≤ I admissibility on the whole map
        Ppix = @. sqrt(img.Q^2 + img.U^2 + img.V^2)
        @test all(Ppix .<= img.I .* (1 + 1e-6))
        # EVPA smoothness: LOCAL neighbor-to-neighbor EVPA change is small over the bright
        # region (a smooth, ordered pattern — not pixel-to-pixel noise). χ = ½atan(U,Q) is
        # π-periodic; differences are wrapped into (−π/2, π/2].
        χ = @. 0.5 * atan(img.U, img.Q)
        wrap(d) = (d = mod(d + π/2, π) - π/2; d)
        difs = Float64[]
        np = size(img.I, 1)
        for c in 1:np, rr in 1:np
            mask[rr, c] || continue
            rr < np && mask[rr+1, c] && push!(difs, abs(wrap(χ[rr+1,c] - χ[rr,c])))
            c  < np && mask[rr, c+1] && push!(difs, abs(wrap(χ[rr,c+1] - χ[rr,c])))
        end
        mean_dχ = isempty(difs) ? 0.0 : sum(difs)/length(difs)
        @test rad2deg(mean_dχ) < 30.0     # ordered EVPA, not noise
        @printf("\n[analytic-disk] flux(I)=%.3e  net-LP=%.4f  sum|V|/I=%.4f  npix²=%d\n",
                Itot, LP, CP, length(img.I))
        @printf("[analytic-disk] bright pixels=%d  per-pixel LP median=%.4f max=%.4f  mean-neighbour ΔEVPA=%.1f deg\n",
                count(mask), lp_med, lp_max, rad2deg(mean_dχ))
    end

    @testset "AMR-cell step cap (RAPTOR gr_integrator.c:277-292, opt-in)" begin
        cam = PlaneParallelCamera(INC_DEG, 100.0, 20.0, 8)
        rays = camera_rays(st, cam)
        # default: bit-identical to no-cap
        s0, d0 = trace_rays_sampled(st, rays; C=40.0, escape=110.0)
        s1, d1 = trace_rays_sampled(st, rays; C=40.0, escape=110.0, max_step=Inf, cell_size_fn=nothing)
        @test all(s0[i] == s1[i] && d0[i] == d1[i] for i in eachindex(s0))
        # hard cap: every taken step ≤ max_step
        cap = 0.05
        _, d2 = trace_rays_sampled(st, rays; C=40.0, escape=110.0, max_step=cap)
        @test all(all(<=(cap + 1e-12), d2[i][1:end-1]) for i in eachindex(d2))
        # capping strictly increases (or keeps) the sample count vs uncapped
        @test all(length(d2[i]) >= length(d0[i]) for i in eachindex(d2))
        # cell_size_fn callback also caps (fraction of a local cell size)
        _, d3 = trace_rays_sampled(st, rays; C=40.0, escape=110.0, cell_size_fn=(x->0.03), cell_frac=1.0)
        @test all(all(<=(0.03 + 1e-12), d3[i][1:end-1]) for i in eachindex(d3))
    end

    # ---- stiffness detector pure functions (gpu_pol.jl); guarded on CUDA availability ------
    if get(ENV, "BLP_SKIP_GPU", "0") == "0"
        loaded = try
            REPO = dirname(@__DIR__)
            @eval Main include(joinpath($REPO, "src", "gpu_image.jl"))
            @eval Main include(joinpath($REPO, "src", "gpu_pol.jl"))
            true
        catch err
            @info "stiffness-detector tests skipped (gpu_pol.jl/CUDA unavailable)" err
            false
        end
        if loaded
            @testset "stiffness detector pure functions (RAPTOR check_stiffness port)" begin
                sm  = Main._stiffness_metric; sis = Main._stiffness_is_stiff
                slam = Main._stiffness_lambdas; srk4 = Main._stiffness_rk4_maxmag
                α = SVector(2.0, 0.6, 0.0, -0.4); ρ = SVector(0.0, 1.1, 0.0, 0.7); ds = 0.3
                l1, l2 = slam(α, ρ)
                # RAPTOR eigenvalues: sqrt(zplus)=λ1 (real), sqrt(zminus)=±i λ2 (imag)
                aQ,aV,rQ,rV = α[2],α[4],ρ[2],ρ[4]
                a2 = rQ^2+rV^2-aQ^2-aV^2; a0 = -2aV*aQ*rV*rQ-aQ^2*rQ^2-aV^2*rV^2
                disc = sqrt(Complex(a2^2-4a0)); zp=(-a2+disc)/2; zm=(-a2-disc)/2
                @test sqrt(zp) ≈ complex(l1) atol=1e-9
                @test abs(sqrt(zm)) ≈ l2 atol=1e-9
                @test abs(real(sqrt(zm))) < 1e-9
                # metric = max|λ|·ds with λ = α_I±λ1, α_I±iλ2
                mre = max(abs(α[1]+l1), abs(α[1]-l1)); mim = sqrt(α[1]^2+l2^2)
                @test sm(α, ρ, ds) ≈ max(mre, mim)*ds
                # emission-only ⇒ metric = α_I·ds
                @test sm(SVector(1.5,0.0,0.0,0.0), SVector(0.0,0.0,0.0,0.0), 0.4) ≈ 0.6
                # threshold = ln(1/0.99); flag tracks it and the RAPTOR RK4 magnitude
                @test Main._STIFF_METRIC_THRESH ≈ -log(0.99)
                @test !sis(SVector(1e-4,0.,0.,0.), SVector(0.,0.,0.,0.), 1e-3)
                @test  sis(SVector(5.0,0.,0.,0.), SVector(0.,0.,0.,0.), 1.0)
                @test srk4(SVector(5.0,0.,0.,0.), SVector(0.,0.,0.,0.), 1.0) > 0.99
                @test srk4(SVector(1e-4,0.,0.,0.), SVector(0.,0.,0.,0.), 1e-3) < 1.001
            end
        end
    end
end

# --- regression: camera-readout normal observer is a unit 4-velocity (shift sign) ------
@testset "camera normal observer is a unit 4-velocity (shift-sign)" begin
    st = KerrSchildCartesian(0.9)
    for d in (100.0, 1000.0)
        x = SVector(0.0, d * sind(60), 0.0, d * cosd(60))
        gcov = metric(st, x); gcon = inverse_metric(st, x)
        u = Rad._normal_observer(gcon)
        @test dot(u, gcov * u) ≈ -1.0 rtol = 1e-10        # exact unit norm (was ≈ -0.998)
        @test u[1] > 0
        # the previously-shipped flipped shift sign is NOT a unit observer
        lp = 1.0 / sqrt(-gcon[1, 1])
        ubad = SVector(1.0 / lp, -gcon[1, 2] / gcon[1, 1] / lp,
                       -gcon[1, 3] / gcon[1, 1] / lp, -gcon[1, 4] / gcon[1, 1] / lp)
        @test !isapprox(dot(ubad, gcov * ubad), -1.0; rtol = 1e-6)
    end
end

# --- regression: fluid tetrad e_(1) renormalization off single CKS (|det g| ≠ 1) -------
@testset "polarized march renorm_e1 off single CKS (det g ≠ 1)" begin
    # dispatch predicate threaded by polarized_march / camera readout
    @test !(SuperposedKerrSchild([1.0], [0.9], [(0.0, 0.0, 0.0)]) isa KerrSchildCartesian)
    @test KerrSchildCartesian(0.9) isa KerrSchildCartesian
    # a BINARY superposition (single-hole superposed KS reduces to exact CKS, |det g|=1);
    # sampled between the two holes where both perturbations overlap.
    st = SuperposedKerrSchild([1.0, 1.0], [0.9, 0.9], [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
    x = SVector(0.0, 0.0, 0.0, 2.0)
    gcov = metric(st, x); gcon = inverse_metric(st, x)
    @test !(abs(det(gcov)) ≈ 1.0)                          # precondition: breaks |det g| = 1
    fv = Sim.primitive_four_vectors(0.1, 0.0, 0.0, 0.2, -0.1, 0.05, gcov, gcon)
    kcov = SVector(-1.0, 0.3, 0.1, -0.2); kcon = gcon * kcov
    e1(tet) = SVector(tet[2, 1], tet[2, 2], tet[2, 3], tet[2, 4])
    tet0 = Rad.fluid_tetrad(fv.ucon, fv.ucov, kcon, kcov, fv.bcon, gcov, gcon)
    tet1 = Rad.fluid_tetrad(fv.ucon, fv.ucov, kcon, kcov, fv.bcon, gcov, gcon; renorm_e1 = true)
    @test !isapprox(dot(e1(tet0), gcov * e1(tet0)), 1.0; rtol = 1e-6)   # BUG: e_(1) not unit
    @test dot(e1(tet1), gcov * e1(tet1)) ≈ 1.0 rtol = 1e-12             # FIXED: unit e_(1)
    # eight_coefficients honors the renorm flag it now receives (returned Q/U frame)
    cfg = SimCoeffConfig(d_unit = 1.0, e_unit = 1.0, b_unit = 1.0)
    _, _, _, tetc, valid8 = Rad.eight_coefficients(cfg, 1.0, 1.0, fv, gcov, gcon,
                                                   kcov, kcon, 230.0e9, true)
    @test valid8
    @test dot(e1(tetc), gcov * e1(tetc)) ≈ 1.0 rtol = 1e-12
end

end # module TestAnalyticDisk