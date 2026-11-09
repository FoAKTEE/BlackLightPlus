# ADM slices in time against their exact limits: the time stencil reproduces quadratics, is
# C¹ and of third order; one slice is the frozen `ADMGrid` itself, copies of a slice are its
# metric to round-off and a series at a slice time is that slice; against an orbiting
# binary sampled into slices the metric errs as Δt³ and its time derivative as Δt² at a
# lattice node, and as Δx³ and Δx² at a slice time; rays through slices of a uniformly
# moving Kerr hole converge to the analytic rays at third order, and the hole's Killing
# charge p_t + v·p, kept to round-off by the analytic flow, drifts on the series at that
# order.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
import ForwardDiff
using BlackLightPlus.Spacetime: KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit, ADMGrid,
                     ADMSeries, Spacetime, adm_grid_from_metric, adm_series_from_metric, adm_slice,
                     metric, inverse_metric, metric_and_inverse, lower_index, raise_index, raise_and_norm,
                     inverse_metric_gradient, geodesic_rhs, geodesic_transport_rhs, rk4_step,
                     sample_adm, on_grid, proxy_distance, march_control, MarchBudget, horizon_floor,
                     horizon_radius, helical_symmetry, trace_geodesic, ks_radius, nullify_covariant,
                     null_covariant, lane
using BlackLightPlus.Numerics: hermite_weights, cr_weights
const DK = Main.Decks

function interp(ts, f, t)
    w, i = hermite_weights(ts, t)
    return sum(w[m] * f(ts[i[m]]) for m in 1:4)
end

@testset "ADM series" begin
    @testset "time stencil" begin
        quad(t) = 1.3 - 0.4t + 0.7t^2
        for ts in ([0.0, 0.7, 1.5, 2.1, 3.4, 4.0], [0.0, 1.0, 2.5], collect(0.0:0.5:3.0))
            for t in range(ts[1], ts[end]; length = 41)
                @test interp(ts, quad, t) ≈ quad(t) atol = 1e-13
            end
        end
        lin(t) = 2.0 - 3.0t
        @test all(t -> interp([1.0, 2.5], lin, t) ≈ lin(t), range(1.0, 2.5; length = 9))
        # Catmull-Rom on the interior intervals of uniform nodes
        ts = collect(0.0:0.5:4.0)
        for t in (1.1, 1.75, 2.6)
            w, i = hermite_weights(ts, t)
            k = searchsortedlast(ts, t)
            @test i == (k - 1, k, k + 1, k + 2)
            @test w ≈ cr_weights((t - ts[k]) / 0.5) atol = 1e-14
        end
        # the end nodes outside the span, constant; the first node for NaN
        @test hermite_weights(ts, -1.0) == (SVector(1.0, 0.0, 0.0, 0.0), (1, 1, 1, 1))
        @test hermite_weights(ts, 9.0) == (SVector(1.0, 0.0, 0.0, 0.0), (9, 9, 9, 9))
        @test hermite_weights(ts, NaN)[2] == (1, 1, 1, 1)
        d = ForwardDiff.derivative(t -> interp(ts, sin, t), -1.0)          # constant outside
        @test d == 0.0
        # C¹ at the nodes: the derivative from either side agrees (first, interior, last)
        tn = [0.0, 0.7, 1.5, 2.1, 3.4, 4.0]
        der(t) = ForwardDiff.derivative(s -> interp(tn, sin, s), t)
        for k in 2:5
            @test abs(der(tn[k] - 1e-9) - der(tn[k] + 1e-9)) < 1e-7
        end
        # third order in the value, second in the derivative
        function err(h)
            tg = collect(0.0:h:6.0); s = range(0.0, 6.0; length = 601)[2:end - 1]
            return (maximum(t -> abs(interp(tg, sin, t) - sin(t)), s),
                    maximum(t -> abs(ForwardDiff.derivative(u -> interp(tg, sin, u), t) - cos(t)), s))
        end
        e1 = err(0.2); e2 = err(0.1)
        @test 7.0 < e1[1] / e2[1] < 9.0
        @test 3.5 < e1[2] / e2[2] < 5.0
    end

    kerr = KerrSchildCartesian(0.9)
    xs = -12.0:0.5:12.0
    grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min = 0.1,
                                excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))

    @testset "one slice is the frozen slice; equal slices are the frozen metric" begin
        one = adm_series_from_metric(x -> metric(kerr, x), xs, xs, xs, [3.0]; alpha_min = 0.1,
                                     excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
        @test one isa ADMGrid{Float64,Array{Float64,4}} && one.data == grid.data
        @test_throws ArgumentError ADMSeries([grid], [0.0])
        # copies of the slice: the frozen metric to round-off, no time derivative
        ser = ADMSeries([grid, grid, grid, grid], [0.0, 1.0, 2.0, 3.0])
        @test ser isa Spacetime{Float64} && lane(ser, Float32) === ser && horizon_radius(ser) == 0.1
        @test horizon_floor(ser) == horizon_floor(grid) && !helical_symmetry(ser, 0.0)
        @test DK.device_isbits(ser)
        p = SVector(-1.0, 0.3, -0.5, 0.2)
        for x in (SVector(0.4, 7.3, -4.1, 2.7), SVector(2.2, 2.2, 1.9, -0.6), SVector(1.7, 1.49, 1.49, 0.99),
                  SVector(1.0, 30.0, 0.0, 0.0))
            y = vcat(x, p)
            @test metric(ser, x) ≈ metric(grid, x) rtol = 1e-14
            @test raise_and_norm(ser, x, p)[1] ≈ raise_and_norm(grid, x, p)[1] rtol = 1e-14
            @test lower_index(ser, x, p) ≈ lower_index(grid, x, p) rtol = 1e-14
            @test geodesic_rhs(ser, y) ≈ geodesic_rhs(grid, y) rtol = 1e-12 atol = 1e-14
            @test abs(geodesic_rhs(ser, y)[5]) < 1e-13
            @test proxy_distance(ser, x)[1] ≈ proxy_distance(grid, x)[1] rtol = 1e-14
            @test on_grid(ser, x) == on_grid(grid, x)
        end
        x0 = SVector(0.0, 15.2, 7.7, 1.3)
        pk = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        y0 = vcat(x0, nullify_covariant(kerr, x0, SVector(pk[2], pk[3], pk[4])))
        b = MarchBudget(80.0, 19.0, Inf, 6000)
        sa, _ = DK.sampled_trace(ser, [y0]; C = 80.0, escape = 19.0, nmax = 6000)
        sb, _ = DK.sampled_trace(grid, [y0]; C = 80.0, escape = 19.0, nmax = 6000)
        @test length(sa[1]) == length(sb[1]) && maximum(maximum(abs, u - v) for (u, v) in zip(sa[1], sb[1])) < 1e-9
        @test march_control(ser, sa[1][2], Inf, b) == march_control(grid, sb[1][2], Inf, b)
        @test trace_geodesic(ser, y0; escape = 19.0).u[end] ≈ trace_geodesic(grid, y0; escape = 19.0).u[end] rtol = 1e-7
    end

    @testset "construction and the slice times" begin
        o = CircularOrbit((0.5, 0.5), 6.0)
        bin = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.3), (0.0, 0.0, -0.2)], o)
        ys = -4.0:0.5:4.0
        ts = [0.0, 1.5, 2.5, 4.0]
        ser = adm_series_from_metric(x -> metric(bin, x), ys, ys, ys, ts)
        sl = [adm_grid_from_metric(x -> metric(bin, x), ys, ys, ys; t = t) for t in ts]
        @test ADMSeries(sl, ts).data == ser.data && ser.times == ts
        @test !helical_symmetry(ser, 0.0) && !helical_symmetry(ser, o.omega)
        for (n, t) in enumerate(ts), x in (SVector(2.3, -1.1, 0.7), SVector(-3.3, 2.9, -0.2))
            ev = SVector(t, x...)
            @test metric(ser, ev) == metric(sl[n], ev)               # a slice time is its slice
            @test metric(adm_slice(ser, n), ev) == metric(sl[n], ev)
        end
        @test metric(ser, SVector(-5.0, 1.0, 1.0, 1.0)) == metric(sl[1], SVector(0.0, 1.0, 1.0, 1.0))
        @test metric(ser, SVector(9.0, 1.0, 1.0, 1.0)) == metric(sl[4], SVector(0.0, 1.0, 1.0, 1.0))
        @test inverse_metric_gradient(ser, SVector(9.0, 1.0, 1.0, 1.0))[:, :, 1] == zeros(4, 4)
        # the time column of the series' force against the four-column Jacobian of its metric
        x = SVector(2.0, 1.3, -0.4, 2.2)
        J = ForwardDiff.jacobian(z -> inverse_metric(ser, z), x)
        @test reshape(inverse_metric_gradient(ser, x), 16, 4) ≈ J rtol = 1e-12
        y = vcat(x, SVector(-1.0, 0.3, -0.5, 0.2))
        dpt = -0.5 * sum(J[(ν - 1) * 4 + μ, 1] * y[4 + μ] * y[4 + ν] for μ in 1:4, ν in 1:4)
        @test geodesic_rhs(ser, y)[5] ≈ dpt rtol = 1e-10
        # one lattice, increasing times
        g2 = adm_grid_from_metric(x -> metric(bin, x), -4.0:0.25:4.0, ys, ys)
        @test_throws ArgumentError ADMSeries([sl[1], g2], [0.0, 1.0])
        @test_throws ArgumentError ADMSeries(sl[1:2], [1.0, 1.0])
        @test_throws ArgumentError ADMSeries(ys, ys, ys, [0.0], zeros(10, 17, 17, 17, 2))
    end

    @testset "pointwise orders against an orbiting binary" begin
        o = CircularOrbit((0.5, 0.5), 6.0)
        bin = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.3), (0.0, 0.0, -0.2)], o)
        g(x) = metric(bin, x)
        # time: at a lattice node, where the spatial stencil is exact
        node = (1.0, -0.5, 3.0)
        function terr(dt)
            ax(c) = (c - 2.0):1.0:(c + 2.0)
            ser = adm_series_from_metric(g, ax(node[1]), ax(node[2]), ax(node[3]), collect(-4.0:dt:24.0))
            ev = [SVector(t, node...) for t in range(3.1, 20.3; length = 400)]
            return (maximum(x -> maximum(abs, inverse_metric(ser, x) - inverse_metric(bin, x)), ev),
                    maximum(x -> maximum(abs, inverse_metric_gradient(ser, x)[:, :, 1] -
                                              inverse_metric_gradient(bin, x)[:, :, 1]), ev))
        end
        a = terr(0.5); b = terr(0.25)
        @test a[1] / b[1] > 6.0
        @test a[2] / b[2] > 3.2
        # space: at a slice time, where the time stencil is exact
        c = SVector(0.3, -0.2, 3.2)
        function serr(dx)
            ax(k) = (c[k] - 2.0):dx:(c[k] + 2.0)
            ser = adm_series_from_metric(g, ax(1), ax(2), ax(3), [0.0, 5.0, 10.0])
            ev = [SVector(5.0, (c .+ SVector(sin(1.3k), cos(2.1k), sin(0.7k + 1)))...) for k in 1:30]
            return (maximum(x -> maximum(abs, inverse_metric(ser, x) - inverse_metric(bin, x)), ev),
                    maximum(x -> maximum(abs, inverse_metric_gradient(ser, x)[:, :, 2:4] -
                                              inverse_metric_gradient(bin, x)[:, :, 2:4]), ev))
        end
        a = serr(0.25); b = serr(0.125)
        @test a[1] / b[1] > 6.0
        @test a[2] / b[2] > 3.2
    end

    @testset "rays through a moving Kerr hole and its Killing charge" begin
        v = SVector(0.3, 0.1, 0.0)
        hole = BoostedSuperposedKerrSchild([0.5], [(0.0, 0.0, 0.3)], t -> ((SVector(-2.0, -1.0, 0.0) + v * t, v),))
        g(x) = metric(hole, x)
        inner(x) = ks_radius(hole, x, 1) < 0.5 * horizon_radius(hole, 1)
        function series(dx)
            ax = -7.0:dx:7.0
            ts = collect(-4dx:2dx:22.0 + 4dx)
            sl = Vector{ADMGrid{Float64,Array{Float64,4}}}(undef, length(ts))
            Threads.@threads for k in eachindex(ts)
                sl[k] = adm_grid_from_metric(g, ax, ax, ax; excise = inner, t = ts[k])
            end
            return ADMSeries(sl, ts)
        end
        starts = [SVector(0.0, 6.0, β * cos(θ), β * sin(θ)) for β in (4.5, 5.5) for θ in range(0, 2π; length = 7)[1:6]]
        function march(st, x0)
            p, ok = null_covariant(st, x0, SVector(-1.0, 0.0, 0.0))
            y = vcat(x0, p)
            q0 = y[5] + dot(v, y[6:8]); dq = 0.0
            for _ in 1:500
                y = rk4_step(st, y, 0.02)
                dq = max(dq, abs(y[5] + dot(v, y[6:8]) - q0))
            end
            return y, dq
        end
        ref = [march(hole, x) for x in starts]
        @test all(r -> r[2] < 1e-12 && maximum(abs, r[1][2:4]) < 6.5, ref)   # exact Killing charge
        res = [[march(s, x) for x in starts] for s in (series(0.5), series(0.25))]
        ex = [maximum(norm(r[i][1][2:4] - ref[i][1][2:4]) for i in eachindex(ref)) for r in res]
        ep = [maximum(norm(r[i][1][5:8] - ref[i][1][5:8]) for i in eachindex(ref)) for r in res]
        eq = [maximum(r[i][2] for i in eachindex(ref)) for r in res]
        @test ex[2] < 1e-3 && ex[1] / ex[2] > 6.0
        @test ep[1] / ep[2] > 6.0
        @test eq[2] < 1e-3 && eq[1] / eq[2] > 4.0
    end
end
