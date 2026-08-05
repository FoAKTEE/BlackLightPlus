# Polarized radiative-transfer tests: the analytic constant-coefficient Stokes
# coupling (Landi Degl'Innocenti 1985; cf. ipole / blacklight
# StokesCoupleNoSplit) must satisfy exact physics limits and match dense matrix
# exponentials. Also covers GPU-kernel parity and consolidation regressions on
# the KernelAbstractions CPU backend (no CUDA device touched).
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus.PolarizedTransfer: stokes_couple

const S0 = SVector(2.0, 0.5, 0.3, 0.1)
const ZERO4 = SVector(0.0, 0.0, 0.0, 0.0)

@testset "identity & pure emission" begin
    # all-zero coefficients -> identity
    @test stokes_couple(S0, ZERO4, ZERO4, ZERO4, 0.7) ≈ S0
    # pure emission: S + j*ds
    j = SVector(1.0, -0.2, 0.0, 0.05)
    @test stokes_couple(S0, j, ZERO4, ZERO4, 0.7) ≈ S0 + 0.7 * j
end

@testset "total absorption only" begin
    j = SVector(1.0, -0.2, 0.0, 0.05)
    α = SVector(2.0, 0.0, 0.0, 0.0)
    ds = 0.3
    out = stokes_couple(S0, j, α, ZERO4, ds)
    expected = exp(-α[1] * ds) .* (S0 .+ j ./ α[1] .* expm1(α[1] * ds))
    @test out ≈ expected rtol = 1e-12
    # deep optically-thick saturation: S -> j/alpha_I regardless of S0
    thick = stokes_couple(S0, j, α, ZERO4, 1e5)
    @test thick ≈ j ./ α[1] rtol = 1e-10
    @test stokes_couple(10 .* S0, j, α, ZERO4, 1e5) ≈ thick rtol = 1e-10
end

@testset "pure Faraday rotation (norm-preserving)" begin
    ds = 0.4
    # rho_V only: rotation about the V axis: Q -> Q cosθ - U sinθ, U -> U cosθ + Q sinθ
    ρv = SVector(0.0, 0.0, 0.0, 1.3)
    out = stokes_couple(S0, ZERO4, ZERO4, ρv, ds)
    θ = 1.3 * ds
    @test out[1] ≈ S0[1]
    @test out[2] ≈ S0[2] * cos(θ) - S0[3] * sin(θ) rtol = 1e-12
    @test out[3] ≈ S0[3] * cos(θ) + S0[2] * sin(θ) rtol = 1e-12
    @test out[4] ≈ S0[4]
    # rho_Q only (Faraday conversion): rotation in the U-V plane; I, Q invariant
    ρq = SVector(0.0, 0.9, 0.0, 0.0)
    out2 = stokes_couple(S0, ZERO4, ZERO4, ρq, ds)
    @test out2[1] ≈ S0[1]
    @test out2[2] ≈ S0[2]
    @test hypot(out2[3], out2[4]) ≈ hypot(S0[3], S0[4]) rtol = 1e-12
    # any pure rotation preserves |P|
    ρ = SVector(0.0, 0.7, 0.0, -1.1)
    out3 = stokes_couple(S0, ZERO4, ZERO4, ρ, ds)
    @test norm(out3[2:4]) ≈ norm(S0[2:4]) rtol = 1e-12
end

@testset "polarized absorption, no rotation" begin
    j = SVector(1.0, -0.3, 0.0, 0.02)
    α = SVector(2.0, -0.8, 0.0, 0.1)
    ds = 0.5
    out = stokes_couple(S0, j, α, ZERO4, ds)
    # cross-check against dense matrix exponential of dS/ds = j - K S
    K = @SMatrix [α[1] α[2] 0.0 α[4];
                  α[2] α[1] 0.0 0.0;
                  0.0 0.0 α[1] 0.0;
                  α[4] 0.0 0.0 α[1]]
    Sinf = K \ j
    expected = exp(-Matrix(K) * ds) * (S0 - Sinf) + Sinf
    @test out ≈ SVector{4}(expected) rtol = 1e-9
end

@testset "full branch: absorption + rotation vs matrix exponential" begin
    j = SVector(1.0, -0.3, 0.0, 0.02)
    α = SVector(2.0, -0.6, 0.0, 0.15)
    ρ = SVector(0.0, 0.8, 0.0, -1.2)
    ds = 0.37
    # Mueller matrix K per Landi Degl'Innocenti / ipole convention (U comps zero)
    K = @SMatrix [α[1] α[2] 0.0  α[4];
                  α[2] α[1] ρ[4] 0.0;
                  0.0 -ρ[4] α[1] ρ[2];
                  α[4] 0.0 -ρ[2] α[1]]
    Sinf = K \ j
    expected = SVector{4}(exp(-Matrix(K) * ds) * (S0 - Sinf) + Sinf)
    out = stokes_couple(S0, j, α, ρ, ds)
    @test out ≈ expected rtol = 1e-8
    # optically-thick saturation of the full branch: independent of S0
    thickA = stokes_couple(S0, j, α, ρ, 1e4)
    thickB = stokes_couple(5 .* S0, j, α, ρ, 1e4)
    @test thickA ≈ thickB rtol = 1e-8
    @test thickA ≈ SVector{4}(K \ j) rtol = 1e-6
end

@testset "degenerate guard + admissibility" begin
    # alpha_p == rho_p aligned degeneracy must not NaN
    j = SVector(0.1, 0.05, 0.0, 0.0)
    α = SVector(1.0, 0.7, 0.0, 0.0)
    ρ = SVector(0.0, 0.7, 0.0, 0.0)
    out = stokes_couple(S0, j, α, ρ, 0.2)
    @test all(isfinite, out)
    # admissibility: I >= 0, |P| <= I always
    weird = stokes_couple(SVector(0.01, 0.5, 0.5, 0.5), ZERO4, SVector(1.0, 0.99, 0.0, 0.0),
                          ZERO4, 3.0)
    @test weird[1] >= 0
    @test norm(weird[2:4]) <= weird[1] * (1 + 1e-12)
end

# --- regression: fluid_tetrad B ∥ k degeneracy (tetrads.jl NaN-poison guard) -----------
using BlackLightPlus.PolarizedTransfer: fluid_tetrad, eight_coefficients
using BlackLightPlus: SimCoeffConfig

@testset "B ∥ k degenerate fluid tetrad stays finite (no NaN poisoning)" begin
    # Flat metric, fluid at rest, null photon along +x, magnetic field spatially ∥ k
    # (b·u = 0). The perpendicular part of b vanishes, so e2 = b_perp/|b_perp| = 0/0
    # without the guard → NaN tetrad → NaN 8-coefficients → NaN Stokes (silent).
    gflat = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))
    u = SVector(1.0, 0.0, 0.0, 0.0); ucov = gflat * u
    k = SVector(1.0, 1.0, 0.0, 0.0); kcov = gflat * k
    b = SVector(0.0, 1.0, 0.0, 0.0)                        # b·u = 0, spatially ∥ k
    tet = fluid_tetrad(u, ucov, k, kcov, b, gflat, gflat)
    @test all(isfinite, tet)
    # the fallback e2 still yields an orthonormal frame:  e_(a)·e_(b) = η_ab
    η = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))
    @test tet * gflat * transpose(tet) ≈ η atol = 1e-12
    # end-to-end: coefficients finite and (sin θ_B = 0 ⇒) zero, Stokes coupling finite
    cfg = SimCoeffConfig(d_unit = 1.0, e_unit = 1.0, b_unit = 1.0)
    fv = (ucon = u, ucov = ucov, bcon = b, bcov = gflat * b, b_sq = 1.0)
    j, α, ρ, _, valid = eight_coefficients(cfg, 1.0e-20, 1.0e-14, fv, gflat, gflat,
                                           kcov, k, 230.0e9)
    @test valid
    @test all(isfinite, j) && all(isfinite, α) && all(isfinite, ρ)
    @test maximum(abs, j) < 1e-25                          # no emission when b ∥ k
    se = stokes_couple(SVector(1.0, 0.1, 0.0, 0.0), j, α, ρ, 1.0e10)
    @test all(isfinite, se)
end

# ---------------------------------------------------------------------------------------
# Fused GPU-kernel regressions (KA CPU backend; no device). Previously the fused renderers
# silently ignored PlasmaConfig.thermal_frac and dropped power_frac/kappa_frac, producing
# the thermal_frac==1 image for any nonthermal mix. Guarded, idempotent includes (gpu_*
# `using CUDA` loads without a device; the kernels run on the KernelAbstractions CPU backend).
@isdefined(_gimg_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_image.jl"))
@isdefined(_gpol_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_pol.jl"))

using KernelAbstractions
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays
using BlackLightPlus.Radiation: thermal_synchrotron, thermal_synchrotron_polarized
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig

@testset "gpu thermal_frac honored: device coeffs == CPU coeffs, scale linearly" begin
    n_e = 1.0e6; theta_e = 12.0; bb = 30.0; nu = 2.3e11; sin_tb = 0.7; cos_tb = 0.5
    # device unpolarized thermal now matches the CPU thermal_synchrotron at ANY fraction
    for f in (1.0, 0.5, 0.25)
        jg, ag = _gimg_thermal(f, n_e, theta_e, bb, nu, sin_tb)
        jc, ac = thermal_synchrotron(f, n_e, theta_e, bb, nu, sin_tb)
        @test jg ≈ jc rtol = 1e-12
        @test ag ≈ ac rtol = 1e-12
    end
    j1, a1 = _gimg_thermal(1.0, n_e, theta_e, bb, nu, sin_tb)
    j5, a5 = _gimg_thermal(0.5, n_e, theta_e, bb, nu, sin_tb)
    @test j5 ≈ 0.5 * j1 rtol = 1e-12          # was: 0.5 == 1.0 (fraction dropped)
    @test a5 ≈ 0.5 * a1 rtol = 1e-12
    @test _gimg_thermal(0.0, n_e, theta_e, bb, nu, sin_tb) == (0.0, 0.0)
    # polarized: all 8 coeffs scale by thermal_frac and match the CPU port bit-for-bit
    pg = _gpol_thermal(0.5, n_e, theta_e, bb, nu, sin_tb, cos_tb)
    pc = thermal_synchrotron_polarized(0.5, n_e, theta_e, bb, nu, sin_tb, cos_tb)
    @test all(isapprox.(pg, pc; rtol = 1e-12))
    p1 = _gpol_thermal(1.0, n_e, theta_e, bb, nu, sin_tb, cos_tb)
    @test all(isapprox.(collect(pg), 0.5 .* collect(p1); rtol = 1e-12))
end

@testset "gpu _gpol_thermal mirrors the rho_Q blend (CPU parity in (0.005, 0.01))" begin
    # The CPU thermal rho_Q C1 smoothstep bridge (coefficients.jl :: _theta_e_blend)
    # previously had no device twin: _gpol_thermal kept the hard 0.01 switch, so CPU/GPU
    # diverged for Theta_e in (0.005, 0.01). Device coeffs now match the CPU across the
    # window (below it both trap rho_Q to 0; at/above 0.01 both are unchanged).
    n_e = 1.0e6; bb = 30.0; nu = 2.3e11; sin_tb = 0.7; cos_tb = 0.5
    for te in (0.004, 0.0055, 0.007, 0.0099, 0.0100001, 0.02)
        pg = _gpol_thermal(1.0, n_e, te, bb, nu, sin_tb, cos_tb)
        pc = thermal_synchrotron_polarized(1.0, n_e, te, bb, nu, sin_tb, cos_tb)
        @test all(isapprox.(pg, pc; rtol = 1e-12))
    end
    # mid-window rho_Q is genuinely nonzero on the device now (0.0 pre-mirror)
    @test _gpol_thermal(1.0, n_e, 0.007, bb, nu, sin_tb, cos_tb)[7] != 0.0
    # and the device switch at 0.01 is continuous (mirrors the CPU continuity check)
    lo = _gpol_thermal(1.0, n_e, 0.0099999, bb, nu, sin_tb, cos_tb)[7]
    hi = _gpol_thermal(1.0, n_e, 0.0100001, bb, nu, sin_tb, cos_tb)[7]
    @test abs(hi - lo) < 1e-6
end

@testset "gpu params: carry thermal_frac + power-law + kappa eDFs" begin
    @test _gimg_params(RunConfig()).thermal_frac == 1.0
    @test _gimg_params(RunConfig(plasma = PlasmaConfig(thermal_frac = 0.4))).thermal_frac == 0.4
    @test _gpol_params(RunConfig(plasma = PlasmaConfig(thermal_frac = 0.4))).thermal_frac == 0.4
    # thermal deck (both fracs 0): the tuple carries NO nonthermal fields (byte-identical shape)
    @test !hasproperty(_gimg_params(RunConfig()), :power_frac)
    @test !hasproperty(_gpol_params(RunConfig()), :power_frac)
    @test !hasproperty(_gimg_params(RunConfig()), :kappa_frac)
    @test !hasproperty(_gpol_params(RunConfig()), :kappa_frac)
    # power_frac>0 is SUPPORTED — the host-precomputed constants are merged in
    let cfg = RunConfig(plasma = PlasmaConfig(thermal_frac = 0.5, power_frac = 0.5))
        gpi = _gimg_params(cfg); gpp = _gpol_params(cfg)
        @test gpi.power_frac == 0.5 && gpi.power_p == 3.5
        @test gpp.power_frac == 0.5 && gpp.power_jj_v != 0.0 && gpp.power_rho_v != 0.0
        # power_jj/power_aa are the precompute_power_law! outputs (nonzero, finite)
        @test isfinite(gpi.power_jj) && gpi.power_jj != 0.0
        @test isfinite(gpi.power_aa) && gpi.power_aa != 0.0
        # a power-law-only deck carries NO kappa fields (single-eDF tuple)
        @test !hasproperty(gpi, :kappa_frac) && !hasproperty(gpp, :kappa_frac)
    end
    # kappa_frac>0 is SUPPORTED — precompute_kappa! constants are merged in (no error)
    let cfg = RunConfig(plasma = PlasmaConfig(thermal_frac = 0.5, kappa_frac = 0.5, kappa = 4.0, w = 0.5))
        gpi = _gimg_params(cfg); gpp = _gpol_params(cfg)
        @test gpi.kappa_frac == 0.5 && gpi.kappa == 4.0 && gpi.w == 0.5
        # unpol (kappa_jj/kappa_aa) + pol-only (kappa_rho_v besselk factor) constants filled
        @test isfinite(gpi.kappa_jj_low) && gpi.kappa_jj_low != 0.0
        @test isfinite(gpi.kappa_aa_low) && gpi.kappa_aa_low != 0.0
        @test isfinite(gpp.kappa_rho_v) && gpp.kappa_rho_v != 0.0
        # a kappa-only deck carries NO power-law fields
        @test !hasproperty(gpi, :power_frac) && !hasproperty(gpp, :power_frac)
    end
    # mixed thermal+power-law+kappa deck: BOTH nonthermal field sets present
    let cfg = RunConfig(plasma = PlasmaConfig(thermal_frac = 0.34, power_frac = 0.33,
                                              kappa_frac = 0.33, kappa = 4.0, w = 0.5))
        gpi = _gimg_params(cfg); gpp = _gpol_params(cfg)
        @test gpi.power_frac == 0.33 && gpi.kappa_frac == 0.33
        @test gpp.power_frac == 0.33 && gpp.kappa_frac == 0.33
        @test isfinite(gpi.power_jj) && isfinite(gpi.kappa_jj_low)
    end
end

@testset "gpu fused kernel (KA CPU backend) uses thermal_frac (was silently dropped)" begin
    n = 40; box_half = 30.0
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
    cfg1 = RunConfig(bh_a = 0.9375,
                     camera = CameraConfig(inclination_deg = 30.0, distance = 500.0, fov = 30.0, resolution = 12),
                     plasma = PlasmaConfig(thermal_frac = 1.0))
    cfg5 = RunConfig(bh_a = cfg1.bh_a, camera = cfg1.camera, plasma = PlasmaConfig(thermal_frac = 0.5))
    st = KerrSchildCartesian(cfg1.bh_a)
    cam = PlaneParallelCamera(cfg1.camera.inclination_deg, cfg1.camera.distance,
                              cfg1.camera.fov, cfg1.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg1.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg1.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    ng = n; xo = -box_half; inv_dx = (n - 1) / (2box_half); nrays = length(rays)
    backend = CPU(); kern = _fused_kernel!(backend)
    march_flux = gp -> begin
        img = Vector{Float64}(undef, nrays)
        kern(img, rays, host, gp, a, r_h, C, escape, R_em, nmax, ng, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        img
    end
    img1 = march_flux(_gimg_params(cfg1))
    img5 = march_flux(_gimg_params(cfg5))
    @test sum(img1) > 0.0                                     # the synthetic torus emits
    @test sum(img5) < sum(img1)                               # FAILS pre-fix (was bit-identical)
    @test isapprox(sum(img5), 0.5 * sum(img1); rtol = 1e-2)   # ~linear (optically thin deck)
end

@testset "upload_window: validate matching (n, xo, inv_dx)" begin
    mkgrid(m; xo = -30.0, invdx = 1.0) =
        UniformFluidGrid(zeros(Float32, 8, m, m, m), m, 30.0, xo, invdx)
    a32 = mkgrid(32); b32 = mkgrid(32)
    # matching geometry with an empty device list builds without touching CUDA
    win = upload_window(a32, b32, 0.0, 1.0, Int[])
    @test win isa DeviceFluidWindow
    # mismatched n / xo / inv_dx must error BEFORE any upload
    @test_throws ErrorException upload_window(a32, mkgrid(16), 0.0, 1.0, Int[])
    @test_throws ErrorException upload_window(a32, mkgrid(32; xo = -20.0), 0.0, 1.0, Int[])
    @test_throws ErrorException upload_window(a32, mkgrid(32; invdx = 2.0), 0.0, 1.0, Int[])
end

using BlackLightPlus   # bind the root module for the qualified references below

# --- regression: polarized_march coherency-tensor rewrite is bit-exact to the loop port -----
# The march was rewritten off mutable MMatrix temporaries (which heap-allocated ~19.6 kB per
# emitting sample) onto immutable @SMatrix contractions. Every contraction is written as an
# explicit left-to-right sum matching the reference `+=` accumulation, so the result is
# bit-for-bit identical (===), not merely close. This locks that equivalence.
@testset "polarized march helpers bit-exact vs reference loops (allocation rewrite)" begin
    Rad = BlackLightPlus.Radiation
    # verbatim-original mutable-loop references
    function ref_conn(v, Γ)
        ta = zero(MMatrix{4,4,Float64}); for μ in 1:4, β in 1:4, α in 1:4; ta[μ,β] += v[α]*Γ[μ,α,β]; end
        SMatrix(ta)
    end
    function ref_dNN(T, N)
        o = zero(MMatrix{4,4,ComplexF64})
        for μ in 1:4, ν in 1:4
            d = zero(ComplexF64); for β in 1:4; d -= T[μ,β]*N[β,ν] + T[ν,β]*N[μ,β]; end; o[μ,ν] = d
        end
        SMatrix(o)
    end
    function ref_coord(nn, gcov, tet)
        tb = zero(MMatrix{4,4,ComplexF64}); for ν in 1:4, α in 1:4, β in 1:4; tb[ν,α] += gcov[ν,β]*nn[α,β]; end
        tc = zero(MMatrix{4,4,ComplexF64}); for μ in 1:4, ν in 1:4, α in 1:4; tc[μ,ν] += gcov[μ,α]*tb[ν,α]; end
        td = zero(MMatrix{4,4,ComplexF64}); for b in 1:4, μ in 1:4, ν in 1:4; td[b,μ] += tet[b,ν]*tc[μ,ν]; end
        o  = zero(MMatrix{4,4,ComplexF64}); for a in 1:4, b in 1:4, μ in 1:4; o[a,b] += tet[a,μ]*td[b,μ]; end
        SMatrix(o)
    end
    function ref_rebuild(se, tet)
        ntc = zero(MMatrix{4,4,ComplexF64})
        ntc[2,2] = se[1]+se[2]; ntc[3,3] = se[1]-se[2]; ntc[2,3] = se[3]-im*se[4]; ntc[3,2] = se[3]+im*se[4]
        tf = zero(MMatrix{4,4,ComplexF64}); for ν in 1:4, a in 1:4, b in 1:4; tf[ν,a] += tet[b,ν]*ntc[a,b]; end
        o  = zero(MMatrix{4,4,ComplexF64})
        for μ in 1:4, ν in 1:4; acc = zero(ComplexF64); for a in 1:4; acc += tet[a,μ]*tf[ν,a]; end; o[μ,ν] = acc; end
        SMatrix(o)
    end
    for _ in 1:500
        v    = @SVector randn(4)
        Γ    = SArray{Tuple{4,4,4},Float64}(randn(4,4,4))
        N    = SMatrix{4,4,ComplexF64}(randn(ComplexF64,4,4))
        T    = SMatrix{4,4,Float64}(randn(4,4))
        gcov = SMatrix{4,4,Float64}(randn(4,4))
        tet  = SMatrix{4,4,Float64}(randn(4,4))
        se   = @SVector randn(4)
        @test Rad._connection_dot(v, Γ)            === ref_conn(v, Γ)
        @test Rad._transport_dNN(T, N)             === ref_dNN(T, N)
        @test Rad._coord_to_tetrad_cov(N, gcov, tet) === ref_coord(N, gcov, tet)
        @test Rad._tetrad_to_coord(se, tet)        === ref_rebuild(se, tet)
    end
end

# --- regression: LD85 admissibility clamp (|alpha_P| <= alpha_I) bounds the full branch -------
# Without the clamp, a fit that returns alpha_P > alpha_I drives the optically-thin full-branch
# propagator exp(-delta_tau) cosh(lambda_1 ds) with lambda_1 -> alpha_P, i.e. it grows like
# exp((alpha_P - alpha_I) ds) and amplifies I; the trailing |P| <= I clamp cannot bound I itself.
@testset "stokes_couple admissibility clamp (alpha_P > alpha_I) bounds I" begin
    S = SVector(2.0, 0.3, 0.0, 0.1)
    # inadmissible: alpha_P = sqrt(1.5^2 + 0^2) = 1.5 > alpha_I = 1.0. Pure absorption (rho=0),
    # so the segment MUST decay toward j/alpha_I, never amplify past max(S0, j/alpha_I).
    j = SVector(0.5, 0.0, 0.0, 0.0)
    α = SVector(1.0, 1.5, 0.0, 0.0)
    ceiling = max(S[1], j[1] / α[1])              # source-function / initial ceiling
    for ds in (1.0, 5.0, 20.0, 100.0)
        out = stokes_couple(S, j, α, SVector(0.0,0.0,0.0,0.0), ds)
        @test all(isfinite, out)
        @test out[1] >= 0.0
        @test out[1] <= ceiling * (1 + 1e-9)      # bounded (was exp((1.5-1.0) ds)-amplified)
        @test norm(out[2:4]) <= out[1] * (1 + 1e-9)
    end
    # deep column saturates to j/alpha_I (the clamp lands exactly on the degeneracy boundary,
    # which the guard routes to total-absorption coupling).
    @test stokes_couple(S, j, α, SVector(0.0,0.0,0.0,0.0), 1e4)[1] ≈ j[1]/α[1] rtol = 1e-6
    # admissible inputs are untouched: alpha_P = sqrt(0.6^2+0.15^2)=0.618 < 2.0 (the full-branch
    # deck above) is bit-identical whether or not the clamp exists.
    jf = SVector(1.0, -0.3, 0.0, 0.02); αf = SVector(2.0, -0.6, 0.0, 0.15); ρf = SVector(0.0, 0.8, 0.0, -1.2)
    K = @SMatrix [αf[1] αf[2] 0.0  αf[4]; αf[2] αf[1] ρf[4] 0.0; 0.0 -ρf[4] αf[1] ρf[2]; αf[4] 0.0 -ρf[2] αf[1]]
    expected = SVector{4}(exp(-Matrix(K) * 0.37) * (S0 - K \ jf) + K \ jf)
    @test stokes_couple(S0, jf, αf, ρf, 0.37) ≈ expected rtol = 1e-8
end

# ---------------------------------------------------------------------------------------
# GPU-parity mirrors of CPU fixes (KA CPU backend; no device):
#  * _gpol_tetrad B ∥ k degeneracy guard (mirror of radiation/tetrads.jl — was NaN)
#  * _gpol_normal_observer unit Eulerian observer (mirror of the camera shift-sign fix)
#  * _gpol_admissible reverse-chain Stokes clamp (mirror of the stokes_couple clamp)
#  * same-fluid polarized parity: fused GPU kernel (KA CPU) vs the CPU coherency march.
using BlackLightPlus
using BlackLightPlus.Radiation: polarized_march
using BlackLightPlus.Simulation: pgas_from_eint, units
using BlackLightPlus: trace_rays_sampled

@testset "gpu _gpol_tetrad: B ∥ k degeneracy guard (fluid_tetrad mirror)" begin
    gflat = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))
    u = SVector(1.0, 0.0, 0.0, 0.0); ucov = gflat * u
    k = SVector(1.0, 1.0, 0.0, 0.0); kcov = gflat * k
    b = SVector(0.0, 1.0, 0.0, 0.0)                      # b·u = 0, spatially ∥ k
    tet = _gpol_tetrad(u, ucov, k, kcov, b, gflat, gflat)
    @test all(isfinite, tet)                             # was e2 = 0/sqrt(0) -> NaN ray poison
    η = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))
    @test tet * gflat * transpose(tet) ≈ η atol = 1e-12  # still an orthonormal frame
    # identical fallback as the CPU guard
    @test tet ≈ fluid_tetrad(u, ucov, k, kcov, b, gflat, gflat) atol = 1e-14
    # non-degenerate inputs are untouched: still the CPU tetrad, in curved space too
    x = SVector(0.0, 3.0, 2.0, 1.5); a = 0.9375
    gcov, gcon = _gimg_metric(x, a)
    uc = _gpol_normal_observer(gcon); ucv = gcov * uc
    kc2 = SVector(-1.0, 0.4, -0.2, 0.1); kn2 = gcon * kc2
    b2 = SVector(0.0, 0.3, 0.8, -0.1)
    @test _gpol_tetrad(uc, ucv, kn2, kc2, b2, gcov, gcon) ≈
          fluid_tetrad(uc, ucv, kn2, kc2, b2, gcov, gcon) rtol = 1e-13
end

@testset "gpu camera observer: unit 4-velocity (shift-sign fix mirror)" begin
    x = SVector(0.0, 3.0, 2.0, 1.5)                      # strong field (f ~ 0.5)
    a = 0.9375
    gcov, gcon = _gimg_metric(x, a)
    n = _gpol_normal_observer(gcon)
    @test dot(n, gcov * n) ≈ -1.0 atol = 1e-14           # unit timelike
    @test n == BlackLightPlus.Radiation._normal_observer(gcon)  # exact device twin of the CPU fix
    # the old flipped-shift formula is measurably non-unit here — the regression
    lp = 1.0 / sqrt(-gcon[1, 1])
    old = SVector(1.0/lp, -gcon[1,2]/gcon[1,1]/lp, -gcon[1,3]/gcon[1,1]/lp, -gcon[1,4]/gcon[1,1]/lp)
    @test abs(dot(old, gcov * old) + 1.0) > 1e-3
end

@testset "gpu _gpol_admissible: reverse-chain Stokes clamp (stokes_couple mirror)" begin
    ok = SVector(1.0, 0.3, -0.2, 0.1)
    @test _gpol_admissible(ok) == ok                      # admissible -> bit-identical no-op
    c = _gpol_admissible(SVector(1.0, 2.0, 0.0, 0.0))     # |P| > I -> rescaled onto the cone
    @test c[1] == 1.0 && norm(c[2:4]) <= c[1] * (1 + 1e-12)
    @test c[2] ≈ 1.0 rtol = 1e-12
    @test _gpol_admissible(SVector(-0.5, 0.2, 0.1, 0.0)) == SVector(0.0, 0.0, 0.0, 0.0)
    # exactly the CPU clamp: stokes_couple over a zero-coefficient ds=0 segment is
    # identity + admissibility clamp, so the two must agree on any input
    for S0 in (SVector(0.5, 0.6, 0.3, -0.2), SVector(2.0, 0.5, 0.3, 0.1),
               SVector(0.01, 0.5, 0.5, 0.5))
        @test _gpol_admissible(S0) ≈ stokes_couple(S0, ZERO4, ZERO4, ZERO4, 0.0) rtol = 1e-15
    end
end

@testset "gpu polarized kernel (KA CPU) vs CPU coherency march: same-fluid parity" begin
    # synthetic torus on a uniform grid (same deck as the thermal_frac kernel test above)
    n = 40; box_half = 30.0
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
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    cfg = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                    frequency_hz = 2.3e11,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 30.0, resolution = 12))
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    ng = n; xo = -box_half; inv_dx = (n - 1) / (2box_half); nrays = length(rays)
    backend = CPU()
    gout = Vector{SVector{4,Float64}}(undef, nrays)
    _gpol_kernel!(backend)(gout, rays, host, _gpol_params(cfg), a, r_h, C, escape, R_em,
                           nmax, ng, xo, inv_dx, nrays; ndrange = nrays)
    KernelAbstractions.synchronize(backend)
    # CPU reference: coherency-tensor march on the IDENTICAL uniform fluid
    states, dls = trace_rays_sampled(st, rays; C = C, escape = escape, nmax = nmax)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs); p = cfg.plasma
    scc = SimCoeffConfig(bh_a = cfg.bh_a, d_unit = u.dens_unit,
                         e_unit = u.dens_unit * BlackLightPlus.Constants.CL^2, b_unit = u.B_unit,
                         plasma_mu = p.mu, plasma_ne_ni = p.ne_ni,
                         plasma_use_p = p.use_p ? 1 : 0, plasma_rat_low = p.rat_low,
                         plasma_rat_high = p.rat_high, plasma_gamma = p.gamma,
                         plasma_gamma_i = p.gamma_i, plasma_gamma_e = p.gamma_e,
                         plasma_thermal_frac = p.thermal_frac,
                         cut_sigma_max = p.sigma_cut, cut_theta_e_min = p.theta_e_min)
    γ = p.gamma
    sampler = x -> begin
        prim = _gimg_sample(host, ng, xo, inv_dx, x[2], x[3], x[4])
        (rho = prim[1], pgas = pgas_from_eint(prim[5], γ),
         uu = (prim[2], prim[3], prim[4]), bb = (prim[6], prim[7], prim[8]),
         inside = prim[1] > 0.0)
    end
    up = BlackLightPlus._camera_up(cam)
    cpu = Matrix{Float64}(undef, 4, nrays)
    Threads.@threads for i in 1:nrays
        cpu[:, i] .= polarized_march(st, scc, sampler, states[i], dls[i],
                                     cfg.frequency_hz, u.L_unit; cam_up = up)
    end
    corr(va, vb) = begin
        ma = sum(va) / length(va); mb = sum(vb) / length(vb)
        num = sum((va .- ma) .* (vb .- mb))
        den = sqrt(sum(abs2, va .- ma) * sum(abs2, vb .- mb))
        den == 0 ? 0.0 : num / den
    end
    gI = [o[1] for o in gout]; gQ = [o[2] for o in gout]
    gU = [o[3] for o in gout]; gV = [o[4] for o in gout]
    @test sum(gI) > 0.0
    @test corr(gI, cpu[1, :]) > 0.999
    @test corr(gQ, cpu[2, :]) > 0.99
    @test corr(gU, cpu[3, :]) > 0.99
    @test corr(gV, cpu[4, :]) > 0.99
    @test isapprox(sum(gI), sum(cpu[1, :]); rtol = 2e-2)
end

# ---------------------------------------------------------------------------------------
# GPU consolidation pins. The AD device twins and the polarized camera observer are
# ALIASES of the one shared implementation (hand-copies deleted), and the generic
# _fused_kernel! reproduces its own Float64 image in the Dual value lane BIT-for-bit —
# the invariant that let _giad_thermal/_giad_coeffs/_giad_ab and _fused_kernel_ad!
# be deleted.
@isdefined(_giad_params) || include(joinpath(@__DIR__, "..", "src", "gpu_image_ad.jl"))

@testset "gpu consolidation: alias identities + Dual value lane bit-exact" begin
    @test _giad_thermal === _gimg_thermal                 # one thermal body (Dual-generic)
    @test _fused_kernel_ad! === _fused_kernel!            # one unpolarized kernel
    @test _gpol_normal_observer === BlackLightPlus.Radiation._normal_observer  # one source
    # small deck: the Dual-seeded run's value lane == the plain Float64 run, bitwise
    n = 16; box_half = 20.0
    host = zeros(Float32, 8, n, n, n)
    dxg = 2box_half / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        x = -box_half + (i-1)*dxg; y = -box_half + (j-1)*dxg; z = -box_half + (k-1)*dxg
        r = sqrt(x^2 + y^2 + z^2)
        if 2.5 < r < 17.0
            host[1,i,j,k] = exp(-((r-6)/4)^2)
            host[2,i,j,k] = -0.2*y/r; host[3,i,j,k] = 0.2*x/r; host[4,i,j,k] = 0.01
            host[5,i,j,k] = 0.08*host[1,i,j,k]
            host[6,i,j,k] = 0.02; host[7,i,j,k] = 0.03*x/r; host[8,i,j,k] = 0.05
        end
    end
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    cfg = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                    frequency_hz = 2.3e11,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 24.0, resolution = 8))
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    xo = -box_half; inv_dx = (n - 1) / (2box_half); nrays = length(rays)
    backend = CPU()
    img = Vector{Float64}(undef, nrays)
    _fused_kernel!(backend)(img, rays, host, _gimg_params(cfg), a, r_h, C, escape, R_em,
                            nmax, n, xo, inv_dx, nrays; ndrange = nrays)
    KernelAbstractions.synchronize(backend)
    gpd = _giad_params(cfg, :rat_high)
    outd = Vector{typeof(gpd.rat_high)}(undef, nrays)
    _fused_kernel!(backend)(outd, rays, host, gpd, a, r_h, C, escape, R_em,
                            nmax, n, xo, inv_dx, nrays; ndrange = nrays)
    KernelAbstractions.synchronize(backend)
    @test sum(img) > 0.0
    @test ForwardDiff.value.(outd) == img                 # value lane BIT-identical
    @test any(!=(0.0), ForwardDiff.partials.(outd, 1))    # dI/dR_high still flows
end

# ---------------------------------------------------------------------------------------
# render C step-density passthrough. The fixed-control sampler density C (h=(r-r_h)/C) was
# hard-coded 80.0 in _pipeline_setup; it is now a render kwarg (default 80.0) so cross-code
# benchmarks can sweep step density. Exposing it
# MUST NOT perturb the production image: an explicit C=80.0 has to equal the default render
# byte-for-byte, and a different C must actually change the trace (guard against a no-op kwarg).
@testset "render C kwarg: explicit C=80 == default render (bitwise); C changes the trace" begin
    DATA = get(ENV, "BLP_TEST_GRMHD",
        "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
    if isfile(DATA)
        m_bh = 6.2e9 * 1.989e33
        L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
        cfg = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                        frequency_hz = 2.3e11,
                        camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                              fov = 40.0, resolution = 16))
        d = render(cfg, DATA; polarized = false)
        e = render(cfg, DATA; polarized = false, C = 80.0)
        @test e.I == d.I                                       # default C=80 preserved, bitwise
        p = render(cfg, DATA; polarized = true)
        pe = render(cfg, DATA; polarized = true, C = 80.0)
        @test pe.I == p.I && pe.Q == p.Q && pe.U == p.U && pe.V == p.V
        c2 = render(cfg, DATA; polarized = false, C = 160.0)   # denser sampling => new image
        @test c2.I != d.I
        @test !isapprox(c2.I, d.I; rtol = 1e-3)                # C genuinely reaches the tracer
    else
        @info "render C kwarg: skipping (data file not present)" DATA
    end
end

# ---------------------------------------------------------------------------------------
# GPU physics twins pinned bit-for-bit against their CPU source. Three device helpers had
# only end-to-end coverage, unlike the thermal/thermal-pol/tetrad twins pinned above: the
# A&S Bessel fits _gpol_k0/k1/k2 (feed the rho_Q/rho_V Faraday rotativities), the
# fluid-decode _gimg_pfv, and the closed-form KS inverse metric _gimg_gi. Each is a
# byte-copy of a CPU function, so a CPU-side physics fix would otherwise silently not reach
# the GPU image. NO consolidation: bessel feeds Faraday (the ~1e-4-amplified path) and
# _gimg_gi is the geodesic-RHS metric — both fmuladd-sensitive.
using Random
using BlackLightPlus.Radiation: bessel_k0, bessel_k1, bessel_k2
using BlackLightPlus.Simulation: primitive_four_vectors

@testset "gpu physics twins == CPU source (bessel / pfv / KS inverse metric), bitwise" begin
    rng = MersenneTwister(0xBE55E1)
    # (1) A&S Bessel K0/K1/K2 — both branches: the x<=2 series and the x>2 asymptotic form
    for _ in 1:4000
        x = 10.0 ^ (rand(rng) * 4 - 2)                  # 1e-2 .. 1e2, straddles the 2.0 branch
        @test _gpol_k0(x) == bessel_k0(x)
        @test _gpol_k1(x) == bessel_k1(x)
        @test _gpol_k2(x) == bessel_k2(x)
    end
    for x in (2.0, prevfloat(2.0), nextfloat(2.0), 0.5, 5.0, 50.0)   # exact branch boundary
        @test _gpol_k0(x) == bessel_k0(x)
        @test _gpol_k1(x) == bessel_k1(x)
        @test _gpol_k2(x) == bessel_k2(x)
    end
    # (2) normal-frame primitive -> GR four-vector decode. rtol=1e-12 (NOT bitwise): _gimg_pfv
    # is a verbatim source copy, but its uu0/ucon decode chain is a sum-of-products whose fma
    # contraction is inline-context-dependent, so it rounds 1-2 ulp off primitive_four_vectors
    # in some enclosing scopes (the fmuladd contraction hazard). Same tolerance the sibling
    # thermal/tetrad twin checks above use (1e-12/1e-13) — 4+ orders below any real physics drift.
    a = 0.9375; st = KerrSchildCartesian(a); r_h = 1 + sqrt(1 - a * a)
    for _ in 1:2000
        x = SVector(0.0, (rand(rng) - 0.5) * 60, (rand(rng) - 0.5) * 60, (rand(rng) - 0.5) * 60)
        ks_radius(st, x) > r_h + 0.5 || continue
        gcov = metric(st, x); gcon = inverse_metric(st, x)
        uu1, uu2, uu3 = 0.3 * randn(rng), 0.3 * randn(rng), 0.3 * randn(rng)
        bb1, bb2, bb3 = 0.1 * randn(rng), 0.1 * randn(rng), 0.1 * randn(rng)
        g = _gimg_pfv(uu1, uu2, uu3, bb1, bb2, bb3, gcov, gcon)
        c = primitive_four_vectors(uu1, uu2, uu3, bb1, bb2, bb3, gcov, gcon)
        @test g[1] ≈ c.ucon rtol = 1e-12
        @test g[2] ≈ c.ucov rtol = 1e-12
        @test g[3] ≈ c.bcon rtol = 1e-12
        @test g[4] ≈ c.b_sq rtol = 1e-12
    end
    # (3) closed-form KS inverse metric == the CPU inverse_metric over realistic trace states.
    # BITWISE by design: should this ever fail ONLY because the CPU _ks_term changed a
    # primitive (e.g. hypot -> sqrt), update the GPU copy to match — do NOT relax to rtol here.
    for _ in 1:4000
        x = SVector(0.0, (rand(rng) - 0.5) * 2.4e3, (rand(rng) - 0.5) * 2.4e3,
                    (rand(rng) - 0.5) * 2.4e3)
        ks_radius(st, x) > r_h || continue
        @test _gimg_gi(x, a) == inverse_metric(st, x)
    end
end
