# Kerr as a numerical grid: the ADM grid reproduces the analytic metric to interpolation
# order, AD differentiates through the cubic interpolant, geodesics agree, off-grid is
# flat, and the horizon excision gives captured rays a clean stopping surface.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
using BlackLightPlus.Spacetime: KerrSchildCartesian, ADMGrid, StationarySpacetime, ETA, adm_grid_from_metric,
                     metric, inverse_metric, inverse_metric_gradient, sample_adm, on_grid,
                     nullify_covariant, null_error, trace_geodesic, ks_radius, horizon_radius,
                     PlaneParallelCamera, MarchBudget, march_control
const Decks = Main.Decks

@testset "ADM grid" begin
    a = 0.9
    kerr = KerrSchildCartesian(a)
    xs = -20.0:0.5:20.0
    grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min = 0.1,
                                excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
    @test grid isa ADMGrid{Float64,Array{Float64,4}} && grid isa StationarySpacetime
    @test horizon_radius(grid) == 0.1

    @testset "metric to interpolation order" begin
        for pt in ((7.3, -4.1, 2.7), (5.55, 3.35, -1.15), (-6.7, 2.2, 4.4), (9.1, 5.3, -3.7))
            x = SVector(0.0, pt...)
            gg = metric(grid, x); gk = metric(kerr, x)
            @test gg ≈ gg'
            @test maximum(abs.(gg - gk)) / maximum(abs.(gk)) < 5e-3
        end
        α, β, γ = sample_adm(grid, 7.3, -4.1, 2.7)
        @test 0.5 < α ≤ 1.0
        @test det(γ) > 1.0                                    # γ = I + f l_i l_j
    end

    @testset "AD through the cubic interpolant" begin
        x = SVector(0.0, 7.3, -4.1, 2.7)
        dg = inverse_metric_gradient(grid, x)
        h = 1e-3
        for c in 2:4
            dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
            fd = (inverse_metric(grid, x + dx) - inverse_metric(grid, x - dx)) / (2h)
            @test maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)) < 1e-2
        end
    end

    @testset "gridded geodesic matches the analytic one" begin
        x0 = SVector(0.0, 15.2, 7.7, 1.3)
        p = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        y0 = vcat(x0, nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4])))
        @test abs(null_error(grid, y0)) < 1e-6
        uk = trace_geodesic(kerr, y0; escape = 19.0, reltol = 1e-10, abstol = 1e-12).u[end]
        ug = trace_geodesic(grid, y0; escape = 19.0, reltol = 1e-10, abstol = 1e-12).u[end]
        @test norm(SVector(uk[2], uk[3], uk[4])) > 15.0 && norm(SVector(ug[2], ug[3], ug[4])) > 15.0
        outdir(st, u) = begin
            v = inverse_metric(st, SVector(u[1], u[2], u[3], u[4])) * SVector(u[5], u[6], u[7], u[8])
            normalize(SVector(v[2], v[3], v[4]))
        end
        @test norm(outdir(kerr, uk) - outdir(grid, ug)) < 1e-2
        @test abs(null_error(grid, ug)) < 1e-6
    end

    @testset "off the grid and captured rays" begin
        xo = SVector(0.0, 100.0, 0.0, 0.0)
        @test metric(grid, xo) == ETA && !on_grid(grid, xo)
        @test count(isnan, grid.data) == 0
        @test all(isfinite, metric(grid, SVector(0.0, 1.49, 1.49, 0.99)))
        x0 = SVector(0.0, 15.2, 0.0, 1.3)
        p = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        y0 = vcat(x0, nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4])))
        for esc in (19.0, 10.0)                               # 10 < start radius: no inbound kill
            sg = trace_geodesic(grid, y0; escape = esc)
            @test Symbol(sg.retcode) == :Terminated
            @test all(isfinite, sg.u[end]) && norm(SVector(sg.u[end][2], sg.u[end][3], sg.u[end][4])) < 5.0
        end
        # The fixed-control march ends by capture. Below a capture lapse of 0.1 the
        # interpolated lapse near the excised core stays above the floor and a step of
        # max(R, 1)/C overshoots before the redshift cut stops the ray; a capture lapse
        # of 0.65 stops it at the hole.
        b = MarchBudget(80.0, 19.0, Inf, 6000)
        states, _ = Decks.sampled_trace(grid, [y0]; C = 80.0, escape = 19.0, nmax = 6000)
        @test all(all(isfinite, y) for y in states[1]) && march_control(grid, states[1][1], Inf, b).stop
        g65 = ADMGrid(grid; alpha_min = 0.65)
        states, _ = Decks.sampled_trace(g65, [y0]; C = 80.0, escape = 19.0, nmax = 6000)
        y = states[1][1]
        @test all(isfinite, y) && norm(SVector(y[2], y[3], y[4])) < 2.5
        @test march_control(g65, y, Inf, b).stop
        @test maximum(maximum(abs, momentum) for momentum in (s[5:8] for s in states[1])) < 2.0
    end

    @testset "excision makes the stopping surface alignment-independent" begin
        xs2 = -19.75:0.5:20.25                                # no z = 0 node plane
        g2 = adm_grid_from_metric(x -> metric(kerr, x), xs2, xs2, xs2; alpha_min = 0.1,
                                  excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
        @test count(isnan, g2.data) == 0
        @test minimum(@view g2.data[1, :, :, :]) < 0.1
        xfar = SVector(0.0, 7.3, -4.1, 2.7)
        @test maximum(abs.(metric(g2, xfar) - metric(kerr, xfar))) / maximum(abs.(metric(kerr, xfar))) < 5e-3
        x0 = SVector(0.0, 15.2, 0.0, 1.3)
        p = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        sg = trace_geodesic(g2, vcat(x0, nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4]))); escape = 19.0)
        @test Symbol(sg.retcode) == :Terminated
        @test norm(SVector(sg.u[end][2], sg.u[end][3], sg.u[end][4])) < 5.0
    end

    @testset "camera rays and trajectories on the grid" begin
        g3 = adm_grid_from_metric(x -> metric(kerr, x), -15.0:1.0:15.0, -15.0:1.0:15.0, -15.0:1.0:15.0;
                                  alpha_min = 0.1, excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
        rays = Decks.camera_rays(g3, PlaneParallelCamera(inc_deg = 60.0, d = 120.0, fov = 8.0, npix = 4))
        @test length(rays) == 16
        @test all(abs(null_error(g3, y)) < 1e-9 for y in rays) && all(y -> y[5] ≈ 1.0, rays)
        states, dls = Decks.sampled_trace(g3, rays; C = 80.0, escape = 130.0, nmax = 6000)
        @test all(all(all(isfinite, y) for y in s) for s in states)
        @test all(length(s) == length(d) && d[end] == 0 for (s, d) in zip(states, dls))
        @test any(minimum(sqrt(y[2]^2 + y[3]^2 + y[4]^2) for y in s) < 6.0 for s in states)
    end
end
