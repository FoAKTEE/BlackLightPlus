# Superposed Kerr-Schild binary with backward-traced camera rays (p = -k, x^t
# decreasing). Deck of the i = 90° BBH figure: m1 = m2 = 0.5, a = 0, holes at
# ±12.5 (cos φ0, sin φ0, 0). A backward ray that falls toward a hole approaches the
# future horizon asymptotically; in the superposition that horizon sits outside the
# isolated-hole radius r_A = 2 m_A (the companion's potential f_B ≈ 2 m_B / d adds to
# f_A), so the per-hole test r_A - r_h,A < eps alone never fires there.
using Test, StaticArrays, LinearAlgebra, KernelAbstractions
using BlackLightPlus
using BlackLightPlus: SuperposedKerrSchild, PlaneParallelCamera, camera_rays_at,
                      trace_rays_sampled, trace_geodesic, ks_radius_hole,
                      horizon_radius_hole, geodesic_rhs, null_error
const GEO = BlackLightPlus.Geodesics

@isdefined(_gsp_amr_kernel!) || include(joinpath(@__DIR__, "..", "src", "gpu_spacetime.jl"))

const BB_PHI0 = -0.2233022
const BB_C1 = 12.5 .* (cos(BB_PHI0), sin(BB_PHI0), 0.0)
const BB_C2 = (-BB_C1[1], -BB_C1[2], 0.0)
bb_st() = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [BB_C1, BB_C2])
bb_cam() = PlaneParallelCamera(inc_deg = 90.0, d = 1000.0, fov = 40.0, npix = 64)
# near hole 1 (x ≈ +12.19, in front): its centre projects to α = y1, β = 0; the
# capture cross-section radius is 3√3 m = 2.6, so offsets ≤ 0.7 are all captured
const BB_AIM = [(BB_C1[2] + da, db) for da in (-0.7, 0.0, 0.7) for db in (-0.7, 0.0, 0.7)]
bb_hole1dist(y) = sqrt((y[2] - BB_C1[1])^2 + (y[3] - BB_C1[2])^2 + (y[4] - BB_C1[3])^2)

@testset "binary camera rays are backward-traced and null" begin
    st = bb_st()
    for y in camera_rays_at(st, bb_cam(), BB_AIM)
        @test y[5] == 1.0                                  # p_t = +1  (k_t = -1)
        @test geodesic_rhs(st, y)[1] < 0                   # x^t decreases
        @test abs(null_error(st, y)) < 1e-12
    end
end

@testset "binary metric: inverse and AD Christoffels" begin
    st = bb_st()
    for xyz in ((12.8, -2.5, 0.3), (-11.0, 3.5, -0.8), (0.5, 0.2, 1.0), (40.0, 7.0, -3.0))
        x = SVector(0.0, xyz...)
        g = metric(st, x)
        @test maximum(abs.(g * inverse_metric(st, x) - I)) < 1e-12
        # Γ^μ_{αβ} from AD vs central differences of the covariant metric
        h = 1e-6
        dg = zeros(4, 4, 4)
        for c in 2:4
            dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
            dg[:, :, c] = (metric(st, x + dx) - metric(st, x - dx)) / (2h)
        end
        gi = inverse_metric(st, x)
        Γfd = [sum(gi[μ, λ] * (dg[λ, β, α] + dg[λ, α, β] - dg[α, β, λ]) / 2 for λ in 1:4)
               for μ in 1:4, α in 1:4, β in 1:4]
        @test maximum(abs.(Array(GEO.christoffel(st, x)) - Γfd)) < 1e-7
    end
end

@testset "sampled tracer: backward rays falling into a hole end captured" begin
    st = bb_st()
    rays = camera_rays_at(st, bb_cam(), BB_AIM)
    states, dls = trace_rays_sampled(st, rays; C = 80.0, nmax = 2500, escape = 1100.0)
    for (ys, dl) in zip(states, dls)
        @test all(all(isfinite, y) for y in ys)
        @test length(ys) < 2500                           # ended by capture, not nmax
        @test bb_hole1dist(ys[1]) < 2.0                    # source end at hole 1
        @test maximum(bb_hole1dist(y) for y in ys[1:min(end, 5)]) < 2.0   # no blow-up jump
        @test all(sqrt(y[2]^2 + y[3]^2 + y[4]^2) < 1001.0 for y in ys)
    end
end

@testset "adaptive tracer: backward ray falling into a hole ends captured" begin
    st = bb_st()
    y0 = camera_rays_at(st, bb_cam(), [(BB_C1[2], 0.0)])[1]
    sol = trace_geodesic(st, y0; escape = 1100.0, reltol = 1e-9, abstol = 1e-11)
    u = sol.u[end]
    @test all(isfinite, u)
    @test bb_hole1dist(u) < 2.0
    @test length(sol.t) < 20_000
end

@testset "fused-kernel control: backward ray falling into a hole ends captured" begin
    st = bb_st()
    rays = camera_rays_at(st, bb_cam(), BB_AIM)
    n = length(rays)
    oy = Vector{SVector{8,Float64}}(undef, n)
    om = Vector{SVector{4,Float64}}(undef, n)
    _gsp_shadow_kernel!(CPU())(oy, om, rays, st, 80.0, 1100.0, 2500, n; ndrange = n)
    KernelAbstractions.synchronize(CPU())
    for k in 1:n
        @test om[k][1] == 1.0                              # status: captured
        @test om[k][2] < 2500
        @test all(isfinite, oy[k]) && bb_hole1dist(oy[k]) < 2.0
    end
end

# Camera Stokes basis through the superposed-metric readout (renormalized e_(1)): an
# optically thin slab on the i = 90° line of sight y = z = 500, far from both holes,
# must give the IAU/IEEE signs of the Kerr path (test_ray_orientation.jl): B → north
# gives Q < 0; B → north-west (+y = west) gives U > 0; B toward the camera gives V > 0.
const BB_L = 6.674e-8 * 6.5e9 * 1.989e33 / 2.99792458e10^2
function bb_slab_stokes(st, bb)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = 1000.0, fov = 10.0, npix = 1)
    ray = camera_rays_at(st, cam, [(500.0, 500.0)])
    states, dls = trace_rays_sampled(st, ray; C = 80.0, escape = 1.1e3)
    d_unit = 1.0e-31
    cfg = BlackLightPlus.SimCoeffConfig(d_unit = d_unit, e_unit = d_unit * 2.99792458e10^2,
                                        b_unit = 100.0 / 0.01, plasma_rat_low = 1.0,
                                        plasma_rat_high = 1.0)
    z3 = SVector(0.0, 0.0, 0.0)
    samp = x -> (400.0 < x[2] < 600.0 && 400.0 < x[3] < 600.0 && 400.0 < x[4] < 600.0) ?
        (rho = 1.0, pgas = 1.0e-3, uu = z3, bb = bb, inside = true) :
        (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
    return BlackLightPlus.Radiation.polarized_march(st, cfg, samp, states[1], dls[1], 1.0e9,
                                                    BB_L; cam_up = SVector(0.0, 0.0, 0.0, 1.0))
end

@testset "superposed readout: Stokes basis and V sign match the Kerr path" begin
    kerr = KerrSchildCartesian(0.0)
    for (bb, check) in ((SVector(0.0, 0.0, 0.01), S -> S[2] / S[1] < -0.5),
                        (SVector(0.0, 0.01, 0.01) / sqrt(2.0), S -> S[3] / S[1] > 0.5),
                        (SVector(0.01, 0.0, 0.01) / sqrt(2.0), S -> S[4] > 0))
        Sk = bb_slab_stokes(kerr, bb)
        Sb = bb_slab_stokes(bb_st(), bb)
        @test check(Sk) && check(Sb)
        # same far-field polarization fractions (I itself differs by the slab-edge
        # quadrature: the step h ≈ (r_A - r_h,A)/C ≈ 11 M lands differently per metric)
        @test maximum(abs.(Sb[2:4] / Sb[1] - Sk[2:4] / Sk[1])) < 1e-3
    end
end
