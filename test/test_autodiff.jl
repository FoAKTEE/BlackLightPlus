# AD-generic transfer pipeline: ForwardDiff Dual numbers must flow through the
# ENTIRE radiation+simulation chain (forward-mode parameter sensitivities).
# Each derivative is validated against central finite differences. The
# end-to-end test uses FIXED geometry (Float64 rays/metric) and a Dual-typed
# SimCoeffConfig, so the sensitivity is w.r.t. a plasma unit scale.
using Test, StaticArrays, LinearAlgebra
using ForwardDiff
using BlackLightPlus: stokes_couple, thermal_synchrotron, SimCoeffConfig,
                    KerrSchildCartesian, PlaneParallelCamera, camera_rays,
                    trace_rays_sampled
using BlackLightPlus.Radiation: unpolarized_march
using BlackLightPlus.Simulation: units
using BlackLightPlus.Constants: CL, MSUN

@testset "stokes_couple full branch: derivative through emission" begin
    # x scales the emission vector; result[1] is exactly linear in x here, so the
    # central difference is essentially exact (roundoff-limited). This exercises the
    # full absorption+rotation branch with a Dual emission vector flowing through
    # the analytic constant-coefficient Mueller solution.
    S0 = SVector(2.0, 0.5, 0.3, 0.1)
    j  = SVector(1.0, -0.3, 0.0, 0.02)
    α  = SVector(2.0, -0.6, 0.0, 0.15)
    ρ  = SVector(0.0, 0.8, 0.0, -1.2)
    ds = 0.37
    f  = x -> stokes_couple(S0, j .* x, α, ρ, ds)[1]
    d_ad = ForwardDiff.derivative(f, 1.0)
    h = 1e-6
    d_fd = (f(1.0 + h) - f(1.0 - h)) / (2h)
    @test isapprox(d_ad, d_fd; rtol = 1e-6)
    @test isfinite(d_ad) && d_ad != 0.0
end

@testset "thermal j_i: derivative wrt theta_e" begin
    n_e = 1.0e6; B = 30.0; nu = 230.0e9; sin_th = 0.7
    ji(θ) = thermal_synchrotron(1.0, n_e, θ, B, nu, sin_th)[1]
    d_ad = ForwardDiff.derivative(ji, 10.0)
    h = 1.0e-4
    d_fd = (ji(10.0 + h) - ji(10.0 - h)) / (2h)
    @test isapprox(d_ad, d_fd; rtol = 1e-6)
    @test isfinite(d_ad) && d_ad != 0.0
end

@testset "end-to-end pipeline gradient (fixed geometry, mass-scale)" begin
    st  = KerrSchildCartesian(0.9)
    cam = PlaneParallelCamera(inc_deg = 60.0, d = 100.0, fov = 10.0, npix = 2)
    rays = camera_rays(st, cam)
    # trace ONCE with fixed geometry; the rays/dls are reused for every s
    states, dls = trace_rays_sampled(st, rays; C = 20.0, escape = 110.0)

    # synthetic uniform fluid: Float64 primitives (only the config carries the Dual)
    sampler = x -> begin
        r = sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
        (rho = 1.0, pgas = 0.1, uu = SVector(0.1, 0.0, 0.0),
         bb = SVector(0.5, 0.1, 0.0), inside = r < 20.0)
    end

    # Fixed geometry length scale, chosen so the uniform slab is optically thin at
    # 230 GHz: the intensity is then emission-dominated and its mass-scale gradient
    # is nonzero and well-conditioned (a saturated/thick slab would give dF/ds ≈ 0
    # since the source function B_ν(Θ_e) is s-independent under this scaling).
    L_unit = units(1.0e3 * MSUN, 1.0).L_unit
    ν      = 230.0e9
    CL2    = CL^2
    Bbase  = CL * sqrt(4π * 1.0e-16)            # B_unit at dens_unit = 1e-16

    # F(s): mass-scale (accretion density) scaling of the plasma units. d_unit ∝ s,
    # e_unit ∝ s (so Theta_e is s-independent), b_unit ∝ sqrt(s) (B ∝ sqrt(ρ)).
    F = s -> begin
        scc = SimCoeffConfig{typeof(s)}(
            bh_a = 0.9,
            d_unit = s * 1.0e-16,
            e_unit = s * 1.0e-16 * CL2,
            b_unit = sqrt(s) * Bbase,
            plasma_mu = 0.5, plasma_ne_ni = 1.0,
            plasma_rat_low = 1.0, plasma_rat_high = 10.0,
            plasma_gamma = 5.0 / 3.0, plasma_thermal_frac = 1.0)
        acc = zero(s)
        for i in eachindex(rays)
            acc += unpolarized_march(st, scc, sampler, states[i], dls[i], ν, L_unit)
        end
        acc
    end

    d_ad = ForwardDiff.derivative(F, 1.0)
    h    = 1.0e-5
    d_fd = (F(1.0 + h) - F(1.0 - h)) / (2h)
    @test F(1.0) > 0.0                       # nonzero emission on the rays
    @test isfinite(d_ad) && d_ad != 0.0
    @test isapprox(d_ad, d_fd; rtol = 1e-5)
end

@testset "Enzyme reverse mode through the transfer chain" begin
    # Reverse-mode AD over chained analytic couplings.
    # Guarded: skip cleanly if Enzyme is absent from the environment.
    enzyme_ok = try
        @eval using Enzyme
        true
    catch
        false
    end
    if enzyme_ok
        slabflux(x) = begin
            S = SVector(0.0, 0.0, 0.0, 0.0)
            j = SVector(1.0, -0.3, 0.0, 0.02) .* x
            α = SVector(2.0, -0.6, 0.0, 0.15)
            ρ = SVector(0.0, 0.8, 0.0, -1.2)
            for _ in 1:5
                S = stokes_couple(S, j, α, ρ, 0.4)
            end
            S[1]
        end
        g = Enzyme.gradient(Enzyme.Reverse, slabflux, 1.0)[1]
        h = 1e-6
        fd = (slabflux(1 + h) - slabflux(1 - h)) / (2h)
        @test abs(g - fd) / abs(fd) < 1e-8
    else
        @test_skip "Enzyme not installed"
    end
end

# ---------------------------------------------------------------------------------------
# gpu_image_ad.jl thermal_frac regression (KA CPU backend; no device). The Dual-generic
# coefficient path silently IGNORED PlasmaConfig.thermal_frac (the fused non-AD kernel
# was fixed separately; the AD twin still rendered the thermal_frac==1 image and
# gradient for any fraction). Guarded, idempotent includes as in test_transfer.jl.
@isdefined(_gimg_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_image.jl"))
@isdefined(_giad_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_image_ad.jl"))

using KernelAbstractions
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig

@testset "gpu AD kernel honors thermal_frac (was silently dropped)" begin
    n_e = 1.0e6; θe = 12.0; bb = 30.0; ν = 2.3e11; s = 0.7
    j1, a1 = _giad_thermal(1.0, n_e, θe, bb, ν, s)
    j5, a5 = _giad_thermal(0.5, n_e, θe, bb, ν, s)
    @test j5 ≈ 0.5 * j1 rtol = 1e-12               # was: j5 == j1 (fraction ignored)
    @test a5 ≈ 0.5 * a1 rtol = 1e-12
    @test _giad_thermal(0.0, n_e, θe, bb, ν, s) == (0.0, 0.0)
    # exact twin of the (already fixed) non-AD device helper at any fraction
    for f in (1.0, 0.5, 0.25)
        @test _giad_thermal(f, n_e, θe, bb, ν, s) == _gimg_thermal(f, n_e, θe, bb, ν, s)
    end
    # Dual argument still flows: seeding theta_e propagates a finite nonzero derivative
    jd, ad = _giad_thermal(0.5, n_e, ForwardDiff.Dual{Nothing}(θe, 1.0), bb, ν, s)
    @test isfinite(ForwardDiff.partials(jd, 1)) && ForwardDiff.partials(jd, 1) != 0.0

    # kernel level: value lane reproduces the fused image at thermal_frac = 0.5 and the
    # flux scales ~linearly with the fraction (optically thin deck); partials stay finite.
    n = 32; box_half = 30.0
    host = zeros(Float32, 8, n, n, n)
    dxg = 2box_half / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        x = -box_half + (i-1)*dxg; y = -box_half + (j-1)*dxg; z = -box_half + (k-1)*dxg
        r = sqrt(x^2 + y^2 + z^2)
        if 2.5 < r < 26.0
            host[1,i,j,k] = exp(-((r-8)/6)^2)
            host[2,i,j,k] = -0.2*y/r; host[3,i,j,k] = 0.2*x/r; host[4,i,j,k] = 0.01
            host[5,i,j,k] = 0.08*host[1,i,j,k]
            host[6,i,j,k] = 0.02; host[7,i,j,k] = 0.03*x/r; host[8,i,j,k] = 0.05
        end
    end
    mkcfg(f) = RunConfig(bh_a = 0.9375,
                         camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                               fov = 30.0, resolution = 10),
                         plasma = PlasmaConfig(thermal_frac = f))
    cfg1 = mkcfg(1.0); cfg5 = mkcfg(0.5)
    st = KerrSchildCartesian(cfg1.bh_a)
    cam = PlaneParallelCamera(cfg1.camera.inclination_deg, cfg1.camera.distance,
                              cfg1.camera.fov, cfg1.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg1.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg1.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    xo = -box_half; inv_dx = (n - 1) / (2box_half); nrays = length(rays)
    backend = CPU()
    function run_ad(cfg)
        gp = _giad_params(cfg, :rat_high)
        DT = promote_type(typeof(gp.d_unit), typeof(gp.e_unit), typeof(gp.b_unit),
                          typeof(gp.rat_high), typeof(gp.rat_low))
        out = Vector{DT}(undef, nrays)
        _fused_kernel_ad!(backend)(out, rays, host, gp, a, r_h, C, escape, R_em, nmax,
                                   n, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        out
    end
    function run_plain(cfg)
        img = Vector{Float64}(undef, nrays)
        _fused_kernel!(backend)(img, rays, host, _gimg_params(cfg), a, r_h, C, escape,
                                R_em, nmax, n, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        img
    end
    ad5 = run_ad(cfg5); ad1 = run_ad(cfg1)
    val5 = ForwardDiff.value.(ad5); val1 = ForwardDiff.value.(ad1)
    @test sum(val1) > 0.0
    @test isapprox(val5, run_plain(cfg5); rtol = 1e-12)     # value lane == fused image
    @test sum(val5) < sum(val1)                             # FAILS pre-fix (bit-identical)
    @test isapprox(sum(val5), 0.5 * sum(val1); rtol = 1e-2) # ~linear (optically thin)
    par5 = ForwardDiff.partials.(ad5, 1)
    @test all(isfinite, par5) && any(!=(0.0), par5)         # dI/dR_high still flows
end
