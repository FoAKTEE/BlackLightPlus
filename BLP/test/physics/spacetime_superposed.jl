# The superposed Kerr-Schild binary: exact reduction to single Kerr, the Woodbury
# inverse and its AD gradient, capture by either hole, and backward-traced camera rays
# that end at a hole (the deck of the i = 90° binary figure).
using Test, BLP, StaticArrays, LinearAlgebra
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, metric, inverse_metric,
                     inverse_metric_gradient, nullify_covariant, null_error, trace_geodesic,
                     ks_radius, horizon_radius, geodesic_rhs, PlaneParallelCamera, CameraFrame,
                     camera_ray, MarchBudget, march_control, rk4_step
const Decks = Main.Decks

const BB_PHI0 = -0.2233022
const BB_C1 = 12.5 .* (cos(BB_PHI0), sin(BB_PHI0), 0.0)
const BB_C2 = (-BB_C1[1], -BB_C1[2], 0.0)
bb_st() = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [BB_C1, BB_C2])
bb_cam() = PlaneParallelCamera(inc_deg = 90.0, d = 1000.0, fov = 40.0, npix = 64)
# near hole 1 (in front): its centre projects to α = y1, β = 0; the capture cross-section
# radius is 3√3 m = 2.6, so offsets ≤ 0.7 are all captured
const BB_AIM = [(BB_C1[2] + da, db) for da in (-0.7, 0.0, 0.7) for db in (-0.7, 0.0, 0.7)]
bb_hole1dist(y) = sqrt((y[2] - BB_C1[1])^2 + (y[3] - BB_C1[2])^2 + (y[4] - BB_C1[3])^2)
bb_rays(st) = [first(camera_ray(st, CameraFrame(bb_cam()), α, β)) for (α, β) in BB_AIM]

@testset "superposed Kerr-Schild" begin
    @testset "reduces to single Kerr" begin
        a = 0.9
        kerr = KerrSchildCartesian(a)
        bin1 = SuperposedKerrSchild([1.0], [a], [(0.0, 0.0, 0.0)])
        bin2 = SuperposedKerrSchild([1.0, 0.0], [a, 0.0], [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
        @test horizon_radius(bin1, 1) == kerr.r_h && horizon_radius(bin2) == SVector(kerr.r_h, 0.0)
        for xyz in ((6.0, 2.0, 1.0), (10.0, -4.0, 3.0), (4.0, 0.0, 2.0))
            x = SVector(0.0, xyz...)
            gk = metric(kerr, x)
            @test maximum(abs.(metric(bin1, x) - gk)) < 1e-12
            @test maximum(abs.(metric(bin2, x) - gk)) < 1e-12
            @test maximum(abs.(inverse_metric(bin1, x) - inverse_metric(kerr, x))) < 1e-9
            @test ks_radius(bin1, x, 1) == ks_radius(kerr, x)
            dg = inverse_metric_gradient(bin2, x)
            h = 1e-6
            for c in 2:4
                dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
                fd = (inverse_metric(bin2, x + dx) - inverse_metric(bin2, x - dx)) / (2h)
                @test maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)) < 1e-5
            end
        end
        x0 = SVector(0.0, 100.0, 6.0, 2.0)
        y0 = vcat(x0, nullify_covariant(kerr, x0, SVector(-1.0, 0.05, 0.02)))
        sk = trace_geodesic(kerr, y0; escape = 110.0, reltol = 1e-10, abstol = 1e-12)
        sb = trace_geodesic(bin1, y0; escape = 110.0, reltol = 1e-10, abstol = 1e-12)
        dk = normalize(SVector(sk.u[end][6], sk.u[end][7], sk.u[end][8]))
        db = normalize(SVector(sb.u[end][6], sb.u[end][7], sb.u[end][8]))
        @test maximum(abs.(dk - db)) < 1e-6
        @test abs(null_error(bin1, sb.u[end])) < 1e-7
    end

    @testset "equal-mass binary captures by both holes" begin
        bin = SuperposedKerrSchild([1.0, 1.0], [0.0, 0.0], [(-15.0, 0.0, 0.0), (15.0, 0.0, 0.0)])
        captured(aim) = begin
            x0 = SVector(0.0, 0.0, 200.0, 0.0)
            p = metric(bin, x0) * SVector(1.0, aim, -1.0, 0.0)
            p0 = nullify_covariant(bin, x0, SVector(p[2], p[3], p[4]))
            u = trace_geodesic(bin, vcat(x0, p0 / (-p0[1])); escape = 400.0,
                               reltol = 1e-9, abstol = 1e-9).u[end]
            x = SVector(u[1], u[2], u[3], u[4])
            min(ks_radius(bin, x, 1), ks_radius(bin, x, 2)) < 3.0
        end
        @test captured(0.075)
        @test captured(-0.075)
        x0 = SVector(0.0, 60.0, 200.0, 0.0)
        p0 = nullify_covariant(bin, x0, SVector(0.0, -1.0, 0.0))
        u = trace_geodesic(bin, vcat(x0, p0 / (-p0[1])); escape = 400.0, reltol = 1e-9, abstol = 1e-9).u[end]
        @test sqrt(u[2]^2 + u[3]^2 + u[4]^2) > 100
        # escape is outbound only
        bin1 = SuperposedKerrSchild([1.0], [0.9], [(0.0, 0.0, 0.0)])
        x0 = SVector(0.0, 2000.01, 0.0, 0.0)
        p = metric(bin1, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        u = trace_geodesic(bin1, vcat(x0, nullify_covariant(bin1, x0, SVector(p[2], p[3], p[4])))).u[end]
        @test ks_radius(bin1, SVector(u[1], u[2], u[3], u[4]), 1) < horizon_radius(bin1, 1) + 1e-2
    end

    @testset "binary camera rays are backward-traced and null" begin
        st = bb_st()
        for y in bb_rays(st)
            @test y[5] == 1.0
            @test geodesic_rhs(st, y)[1] < 0
            @test abs(null_error(st, y)) < 1e-12
        end
        rays = Decks.camera_rays(st, PlaneParallelCamera(inc_deg = 90.0, d = 300.0, fov = 40.0, npix = 6))
        @test length(rays) == 36 && all(abs(null_error(st, y)) < 1e-9 for y in rays)
    end

    @testset "rays falling into a hole end captured" begin
        st = bb_st()
        rays = bb_rays(st)
        states, _ = Decks.sampled_trace(st, rays; C = 80.0, nmax = 2500, escape = 1100.0)
        for ys in states
            @test all(all(isfinite, y) for y in ys)
            @test length(ys) < 2500                             # ended by capture, not nmax
            @test bb_hole1dist(ys[1]) < 2.0                     # source end at hole 1
            @test maximum(bb_hole1dist(y) for y in ys[1:min(end, 5)]) < 2.0
            @test all(sqrt(y[2]^2 + y[3]^2 + y[4]^2) < 1001.0 for y in ys)
        end
        # the stop at the source end is a capture (march_control would stop there)
        b = MarchBudget(80.0, 1100.0, Inf, 2500)
        @test all(march_control(st, s[1], Inf, b).stop for s in states)
        u = trace_geodesic(st, rays[5]; escape = 1100.0, reltol = 1e-9, abstol = 1e-11)
        @test all(isfinite, u.u[end]) && bb_hole1dist(u.u[end]) < 2.0 && length(u.t) < 20_000
    end
end
