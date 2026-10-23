# Metric identities, AD gradients, the null root, Kerr-Schild disk regularization and the
# user-metric path. Exact-GR ground truth; no reference data.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
using BlackLightPlus.Spacetime: Spacetime, KerrSchildCartesian, Minkowski, UserMetric, SuperposedKerrSchild,
                     ETA, metric, inverse_metric, metric_and_inverse, inverse_metric_gradient,
                     metric_gradient, christoffel, lower_index, raise_index, raise_and_norm,
                     geodesic_rhs, null_error, nullify_covariant, null_covariant,
                     PlaneParallelCamera, trace_geodesic, horizon_radius, ks_radius
const Decks = Main.Decks

# (f, l_μ) of single Kerr from the metric: g_tt = -1 + f (l_0 = 1), g_ti = f l_i.
ks_fl(g) = (f = g[1, 1] + 1; (f, SVector(1.0, g[1, 2] / f, g[1, 3] / f, g[1, 4] / f)))

@testset "spacetime metrics" begin
    @testset "metric identities" begin
        for a in (0.0, 0.5, 0.9, 0.998), xyz in ((4.0, 1.0, 2.0), (6.0, -3.0, 1.5), (-5.0, 3.0, 2.5))
            st = KerrSchildCartesian(a)
            x = SVector(0.0, xyz...)
            g = metric(st, x); gi = inverse_metric(st, x)
            @test maximum(abs.(g * gi - I)) < 1e-12
            @test g == g'
            @test metric_and_inverse(st, x) == (g, gi)
            _, l = ks_fl(g)
            @test abs(l[2]^2 + l[3]^2 + l[4]^2 - 1) < 1e-12      # l is η-null
            dg = inverse_metric_gradient(st, x)
            h = 1e-6
            for c in 2:4
                dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
                fd = (inverse_metric(st, x + dx) - inverse_metric(st, x - dx)) / (2h)
                @test maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)) < 1e-6
            end
            @test all(iszero, dg[:, :, 1])                       # stationary: exact zero t column
            v = SVector(0.3, -1.0, 0.5, 0.2); p = SVector(1.0, -0.7, 0.1, 0.4)
            @test lower_index(st, x, v) ≈ g * v rtol = 1e-13
            @test raise_index(st, x, p) ≈ gi * p rtol = 1e-13
            @test raise_and_norm(st, x, p)[2] ≈ dot(p, gi * p) rtol = 1e-12
        end
        @test metric(Minkowski(), SVector(0.0, 1.0, 2.0, 3.0)) == ETA
        @test inverse_metric_gradient(Minkowski(), SVector(0.0, 1.0, 2.0, 3.0)) == zeros(4, 4, 4)
    end

    @testset "stationary gradient == four-column Jacobian" begin
        for (a, xyz) in ((0.9, (4.0, 1.0, 2.0)), (0.5, (-5.0, 3.0, 2.5)), (0.998, (2.0, 0.5, -1.0)))
            st = KerrSchildCartesian(a)
            x = SVector(0.0, xyz...)
            dg4 = invoke(inverse_metric_gradient, Tuple{Spacetime,SVector{4,Float64}}, st, x)
            @test inverse_metric_gradient(st, x) == dg4
        end
    end

    @testset "Christoffel symbols" begin
        st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(12.0, -2.8, 0.0), (-12.0, 2.8, 0.0)])
        for xyz in ((12.8, -2.5, 0.3), (-11.0, 3.5, -0.8), (0.5, 0.2, 1.0), (40.0, 7.0, -3.0))
            x = SVector(0.0, xyz...)
            @test maximum(abs.(metric(st, x) * inverse_metric(st, x) - I)) < 1e-12
            h = 1e-6
            dg = zeros(4, 4, 4)
            for c in 2:4
                dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
                dg[:, :, c] = (metric(st, x + dx) - metric(st, x - dx)) / (2h)
            end
            @test maximum(abs.(Array(metric_gradient(st, x)) - dg)) < 1e-7
            gi = inverse_metric(st, x)
            Γfd = [sum(gi[μ, λ] * (dg[λ, β, α] + dg[λ, α, β] - dg[α, β, λ]) / 2 for λ in 1:4)
                   for μ in 1:4, α in 1:4, β in 1:4]
            @test maximum(abs.(Array(christoffel(st, x)) - Γfd)) < 1e-7
        end
    end

    @testset "null root and its contract" begin
        st = KerrSchildCartesian(0.9)
        xerg = SVector(0.0, 1.8, 0.0, 0.0)                     # inside the ergosphere: f > 1
        f, l = ks_fl(metric(st, xerg))
        @test f > 1.0
        @test_throws ErrorException nullify_covariant(st, xerg, SVector(l[2], l[3], l[4]))
        @test !null_covariant(st, xerg, SVector(l[2], l[3], l[4]))[2]
        xout = SVector(0.0, 100.0, 6.0, 2.0)
        p = nullify_covariant(st, xout, SVector(-1.0, 0.05, 0.02))
        @test -p[1] > 0
        @test abs(null_error(st, vcat(xout, p))) < 1e-12
    end

    @testset "Kerr-Schild equatorial disk is regular inside the ring" begin
        for a in (0.5, 0.9, 0.998)
            st = KerrSchildCartesian(a)
            for rho in (0.0, 0.4a, 0.9a)
                x = SVector(0.0, rho, 0.0, 0.0)
                @test metric(st, x) == ETA
                @test inverse_metric(st, x) == ETA
            end
            @test any(isnan, metric(st, SVector(0.0, a, 0.0, 0.0)))   # the ring stays singular
            x9 = SVector(0.0, 0.4a, 0.3a, 1.0e-9)
            @test all(isfinite, metric(st, x9))
            @test maximum(abs.(metric(st, x9) - ETA)) < 1e-7
            @test maximum(abs.(metric(st, SVector(0.0, 0.4a, 0.3a, 1.0e-6)) - ETA)) < 4e-5
        end
        st9 = KerrSchildCartesian(0.9)
        g0 = metric(st9, SVector(0.0, 0.36, 0.27, 0.0))
        @test maximum(abs.(metric(st9, SVector(0.0, 0.36, 0.27, 1.0e-8)) - g0)) < 1e-7
        @test any(isnan, metric(KerrSchildCartesian(0.0), SVector(0.0, 0.0, 0.0, 0.0)))
        sks = SuperposedKerrSchild([0.55, 0.45], [0.4, -0.3], [(-6.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
        @test all(isfinite, metric(sks, SVector(0.0, -6.0 + 0.1, 0.0, 0.0)))
    end

    @testset "user metric rides the general engine" begin
        a = 0.5
        kerr = KerrSchildCartesian(a)
        um = UserMetric(x -> metric(kerr, x))
        x0 = SVector(0.0, 100.0, 6.0, 2.0)
        y0 = vcat(x0, nullify_covariant(um, x0, SVector(-1.0, 0.05, 0.02)))
        @test abs(null_error(um, y0)) < 1e-12
        r_um = geodesic_rhs(um, y0)
        @test r_um[5] == 0.0                                    # ∂_t g ≡ 0 ⇒ dp_t exactly 0
        @test maximum(abs.(r_um - geodesic_rhs(kerr, y0))) < 1e-14
        su = trace_geodesic(um, y0; escape = 110.0)
        sk = trace_geodesic(kerr, y0; escape = 110.0)
        du = normalize(SVector(su.u[end][6], su.u[end][7], su.u[end][8]))
        dk = normalize(SVector(sk.u[end][6], sk.u[end][7], sk.u[end][8]))
        @test maximum(abs.(du - dk)) < 1e-6
        @test abs(null_error(um, su.u[end])) < 1e-7
        # a time-dependent closure keeps dp_t/dλ = -½ ∂_t g^{μν} p_μ p_ν
        c = 0.3
        gt = UserMetric(x -> @SMatrix [-(1 + c * x[1]) 0.0 0.0 0.0
                                        0.0            1.0 0.0 0.0
                                        0.0            0.0 1.0 0.0
                                        0.0            0.0 0.0 1.0])
        xt = SVector(0.0, 10.0, 0.0, 0.0); ps = SVector(-1.0, 0.4, 0.2)
        rt = geodesic_rhs(gt, vcat(xt, nullify_covariant(gt, xt, ps)))
        @test rt[5] ≈ -0.5 * c * dot(ps, ps) rtol = 1e-10
        # camera rays of the closure reproduce the closed-form Kerr rays
        cam = PlaneParallelCamera(inc_deg = 60.0, d = 100.0, fov = 12.0, npix = 5)
        rk = Decks.camera_rays(KerrSchildCartesian(0.9), cam)
        ru = Decks.camera_rays(UserMetric(x -> metric(KerrSchildCartesian(0.9), x)), cam)
        @test maximum(maximum(abs.(yu .- yk)) for (yu, yk) in zip(ru, rk)) < 1e-9
        @test all(y -> y[5] ≈ 1.0, ru)
    end
end
