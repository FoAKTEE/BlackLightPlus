# The moving superposed Kerr-Schild binary: at rest it is single Kerr and the static
# superposition; a hole in uniform motion keeps a null covector, is vacuum, and is the
# rest-frame Kerr metric carried by the boost; its time derivative is -v·∇g; the circular
# orbit's velocities are the derivatives of its positions; rays are captured by a moving
# hole at the hole's position at the time of passage, and the march stops there.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
import ForwardDiff
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, BoostedSuperposedKerrSchild,
                     CircularOrbit, Spacetime, StationarySpacetime, ETA, metric, inverse_metric,
                     metric_and_inverse, lower_index, raise_index, raise_and_norm,
                     inverse_metric_gradient, christoffel, geodesic_rhs, geodesic_transport_rhs,
                     nullify_covariant, null_error, trace_geodesic, ks_radius, horizon_radius, lane,
                     MarchBudget, march_control

at_rest(centers) = t -> ntuple(A -> (SVector{3,Float64}(centers[A]...), zero(SVector{3,Float64})),
                               length(centers))
uniform(s0, v) = t -> ((s0 + v * t, v),)

# Λ(v): Λ⁰₀ = Γ, Λ⁰ᵢ = Λⁱ₀ = -Γ vᵢ, Λⁱⱼ = δᵢⱼ + (Γ-1) nᵢ nⱼ (symmetric)
function boost_matrix(v)
    β = norm(v); Γ = 1 / sqrt(1 - β^2); n = v / β
    S = SMatrix{3,3}(I) + (Γ - 1) * (n * n')
    return SMatrix{4,4}(Γ, -Γ*v[1], -Γ*v[2], -Γ*v[3],
                        -Γ*v[1], S[1,1], S[2,1], S[3,1],
                        -Γ*v[2], S[1,2], S[2,2], S[3,2],
                        -Γ*v[3], S[1,3], S[2,3], S[3,3])
end

# R_μν from AD of the Christoffel symbols
function ricci(st, x)
    Γ = christoffel(st, x)
    J = ForwardDiff.jacobian(xx -> vec(christoffel(st, xx)), x)   # (64, 4): column c = ∂_c
    dΓ(α, μ, ν, c) = J[α + 4 * (μ - 1) + 16 * (ν - 1), c]
    R = zeros(4, 4)
    for μ in 1:4, ν in 1:4
        s = 0.0
        for α in 1:4
            s += dΓ(α, μ, ν, α) - dΓ(α, μ, α, ν)
            for β in 1:4
                s += Γ[α, α, β] * Γ[β, μ, ν] - Γ[α, ν, β] * Γ[β, μ, α]
            end
        end
        R[μ, ν] = s
    end
    return R
end

@testset "moving superposed Kerr-Schild" begin
    @testset "at rest it is single Kerr and the static superposition" begin
        a = 0.9
        kerr = KerrSchildCartesian(a)
        one_hole = BoostedSuperposedKerrSchild([1.0], [(0.0, 0.0, a)], at_rest([(0.0, 0.0, 0.0)]))
        @test one_hole isa Spacetime && !(one_hole isa StationarySpacetime)
        @test horizon_radius(one_hole, 1) == kerr.r_h && horizon_radius(one_hole) == SVector(kerr.r_h)
        c1 = (6.0, -2.0, 1.0); c2 = (-6.0, 2.5, -1.0)
        stat = SuperposedKerrSchild([0.5, 0.5], [0.15, -0.25], [c1, c2])
        for boost in (true, false)
            two = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.15), (0.0, 0.0, -0.25)],
                                              at_rest([c1, c2]); boost)
            @test horizon_radius(two) == horizon_radius(stat)
            for (t, xyz) in ((0.0, (6.0, 2.0, 1.0)), (37.5, (10.0, -4.0, 3.0)), (-5.0, (4.0, 0.0, 2.0)))
                x = SVector(t, xyz...)
                @test maximum(abs.(metric(one_hole, x) - metric(kerr, x))) < 1e-12
                @test maximum(abs.(inverse_metric(one_hole, x) - inverse_metric(kerr, x))) < 1e-9
                @test ks_radius(one_hole, x, 1) ≈ ks_radius(kerr, x) atol = 1e-12
                g, gi = metric_and_inverse(two, x)
                @test maximum(abs.(g - metric(stat, x))) < 1e-12
                @test maximum(abs.(gi - inverse_metric(stat, x))) < 1e-9
                p = SVector(1.0, 0.3, -0.2, 0.7)
                @test maximum(abs.(lower_index(two, x, p) - lower_index(stat, x, p))) < 1e-12
                @test maximum(abs.(raise_index(two, x, p) - raise_index(stat, x, p))) < 1e-9
                r1, n1 = raise_and_norm(two, x, p); r2, n2 = raise_and_norm(stat, x, p)
                @test maximum(abs.(r1 - r2)) < 1e-9 && abs(n1 - n2) < 1e-9
                @test ks_radius(two, x, 2) ≈ ks_radius(stat, x, 2) atol = 1e-12
                @test maximum(abs.(inverse_metric_gradient(two, x)[:, :, 1])) == 0
            end
        end
    end

    @testset "uniform motion: null, vacuum, and Kerr in the rest frame" begin
        a = SVector(0.3, -0.2, 0.5)
        v = SVector(0.3, 0.1, -0.2)
        s0 = SVector(1.0, -2.0, 0.5)
        st = BoostedSuperposedKerrSchild([1.0], [Tuple(a)], uniform(s0, v))
        rest = BoostedSuperposedKerrSchild([1.0], [Tuple(a)], at_rest([(0.0, 0.0, 0.0)]))
        Λ = boost_matrix(v)
        Λi = inv(Λ)
        for (t, xyz) in ((0.0, (5.0, 3.0, -2.0)), (12.0, (3.0, 7.0, 4.0)), (-9.0, (-4.0, -4.0, 5.0)))
            x = SVector(t, xyz...)
            g = metric(st, x)
            h = g - ETA
            @test maximum(abs.(h * ETA * h)) < 1e-12 * maximum(abs.(h))^2      # l null: (f l l')η(f l l') = 0
            @test maximum(abs.(g * inverse_metric(st, x) - I)) < 1e-12
            xp = Λi[2:4, 2:4] * (SVector(xyz...) - (s0 + v * t))                 # hole frame, Λ symmetric
            @test ks_radius(st, x, 1) ≈ ks_radius(rest, SVector(0.0, xp...), 1) atol = 1e-12
            gr = metric(rest, SVector(0.0, xp...))
            @test maximum(abs.(Λ * gr * Λ - g)) < 1e-12                        # g = Λ g_rest Λ
            @test maximum(abs.(ricci(st, x))) < 1e-7
            dg = inverse_metric_gradient(st, x)
            dt = SMatrix{4,4}(dg[:, :, 1])
            adv = -(v[1] * SMatrix{4,4}(dg[:, :, 2]) + v[2] * SMatrix{4,4}(dg[:, :, 3]) +
                    v[3] * SMatrix{4,4}(dg[:, :, 4]))
            @test maximum(abs.(dt - adv)) < 1e-10                               # ∂_t g = -v·∇g
            @test maximum(abs.(dt)) > 1e-4
            hfd = 1e-5
            fd = (inverse_metric(st, x + SVector(hfd, 0, 0, 0)) - inverse_metric(st, x - SVector(hfd, 0, 0, 0))) / (2hfd)
            @test maximum(abs.(dt - fd)) < 1e-7
            p = SVector(1.0, -0.9, 0.2, 0.1)
            y = vcat(x, p)
            rhs = geodesic_rhs(st, y)
            @test rhs[5] ≈ -dot(p, dt * p) / 2 atol = 1e-12
            @test maximum(abs.(SVector(rhs[1], rhs[2], rhs[3], rhs[4]) - inverse_metric(st, x) * p)) < 1e-14
        end
        unb = BoostedSuperposedKerrSchild([1.0], [Tuple(a)], uniform(s0, v); boost = false)
        x = SVector(3.0, 5.0, 3.0, -2.0)
        @test maximum(abs.(metric(unb, x) - metric(rest, SVector(0.0, (SVector(5.0, 3.0, -2.0) - (s0 + 3v))...)))) < 1e-12
        @test maximum(abs.(metric(unb, x) - metric(st, x))) > 1e-3
    end

    @testset "circular orbit" begin
        o = CircularOrbit((0.25, 0.75), 10.0; phase = 0.3)
        @test o.omega ≈ sqrt(1.0 / 1000.0)
        for t in (0.0, 17.3, -250.0)
            (s1, v1), (s2, v2) = o(t)
            @test 0.25 * s1 + 0.75 * s2 ≈ zero(s1) atol = 1e-12
            @test norm(s1 - s2) ≈ 10.0
            @test s1[3] == 0 && v1[3] == 0
            for A in 1:2
                ds = ForwardDiff.derivative(tt -> o(tt)[A][1], t)
                @test maximum(abs.(ds - o(t)[A][2])) < 1e-12
            end
        end
        (s1, _), _ = o(0.0)
        @test s1 ≈ SVector(7.5 * cos(0.3), 7.5 * sin(0.3), 0.0)
        o2 = CircularOrbit((0.5, 0.5), 20.0; omega = 0.01)
        @test o2.omega == 0.01 && o2.r == SVector(10.0, -10.0)
        o32 = lane(BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], o2), Float32)
        @test o32 isa BoostedSuperposedKerrSchild{2,Float32} && o32.trajectory isa CircularOrbit{Float32}
        @test eltype(metric(o32, SVector{4,Float32}(1, 3, 2, 1))) == Float32
    end

    @testset "rays captured by a moving hole, escape, and the march" begin
        v = SVector(0.12, 0.0, 0.0)
        hole = BoostedSuperposedKerrSchild([1.0], [(0.0, 0.0, 0.0)], uniform(zero(v), v))
        d = 100.0
        s = v * (-d)                               # the hole when the photon passes, t ≈ -d
        ray(st, xy) = (x0 = SVector(0.0, xy[1], xy[2], 0.0);
                       k = nullify_covariant(st, x0, SVector(0.0, 1.0, 0.0)); vcat(x0, k / k[1]))
        y0 = ray(hole, (s[1], d))                  # camera on +y, aimed at that position
        @test y0[5] == 1.0 && abs(null_error(hole, y0)) < 1e-12
        u = trace_geodesic(hole, y0; escape = 2d, reltol = 1e-10, abstol = 1e-12).u[end]
        xe = SVector(u[1], u[2], u[3], u[4])
        @test ks_radius(hole, xe, 1) < horizon_radius(hole, 1) + 0.1   # the redshift cut ends it
        @test abs(null_error(hole, u)) < 1e-6
        b = MarchBudget(80.0, 2d, Inf, 5000)
        xin = SVector(u[1], (v * u[1] + SVector(0.0, horizon_radius(hole, 1) + 5e-4, 0.0))...)   # transverse: r' = r
        @test march_control(hole, vcat(xin, SVector(1.0, 0.0, -1.0, 0.0)), Inf, b).stop
        @test !march_control(hole, y0, Inf, b).stop
        # aimed at the hole's t = 0 position (where a frozen metric keeps it), 12 M off: a miss
        u0 = trace_geodesic(hole, ray(hole, (0.0, d)); escape = 2d, reltol = 1e-10, abstol = 1e-12).u[end]
        @test sqrt(u0[2]^2 + u0[3]^2 + u0[4]^2) > 1.9d
        @test u0[5] != 1.0                         # p_t drifts on the non-stationary metric
        # the orbiting binary: a scan about hole 1's position at the passage time finds captures
        orbit = CircularOrbit((0.5, 0.5), 20.0)
        st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
        dc = 60.0
        s1 = orbit(-dc)[1][1]
        fates = map((-4.0, -2.0, 0.0, 2.0, 4.0)) do da
            ub = trace_geodesic(st, ray(st, (s1[1] + da, dc)); escape = 2dc,
                                reltol = 1e-10, abstol = 1e-12).u[end]
            xb = SVector(ub[1], ub[2], ub[3], ub[4])
            minimum(ks_radius(st, xb, A) - horizon_radius(st, A) for A in 1:2) < 0.1
        end
        @test any(fates)
        uf = trace_geodesic(st, ray(st, (60.0, dc)); escape = 2dc, reltol = 1e-10, abstol = 1e-12).u[end]
        @test sqrt(uf[2]^2 + uf[3]^2 + uf[4]^2) > 1.9dc
    end

    @testset "parallel transport on the moving metric" begin
        # the general transport equals the stationary one on Kerr
        kerr = KerrSchildCartesian(0.9)
        x0 = SVector(0.0, 9.0, 4.0, 2.0)
        k0 = nullify_covariant(kerr, x0, SVector(-1.0, 0.1, 0.05))
        y = vcat(x0, k0 / k0[1])
        vs = (SVector(0.0, 0.3, -1.0, 0.2), SVector(1.0, 0.0, 0.0, 0.0))
        dy_s, dv_s = geodesic_transport_rhs(kerr, y, vs)
        dy_g, dv_g = invoke(geodesic_transport_rhs,
                            Tuple{Spacetime,SVector{8,Float64},NTuple{2,SVector{4,Float64}}}, kerr, y, vs)
        @test maximum(abs.(dy_s - dy_g)) < 1e-12
        @test all(maximum(abs.(dv_s[i] - dv_g[i])) < 1e-12 for i in 1:2)
        # on a hole in motion the transported products g(k, V), g(V, W) and g(V, V) stay constant
        v = SVector(0.12, 0.05, 0.0)
        hole = BoostedSuperposedKerrSchild([1.0], [(0.0, 0.0, 0.2)], uniform(zero(v), v))
        xh = SVector(0.0, -8.0, 30.0, 3.0)                # passes 5.8 M from the hole
        kh = nullify_covariant(hole, xh, SVector(0.0, 1.0, 0.0))
        yh = vcat(xh, kh / kh[1])
        g0 = metric(hole, xh)
        kcon = inverse_metric(hole, xh) * SVector(yh[5], yh[6], yh[7], yh[8])
        V = SVector(0.0, 1.0, 0.0, 0.3); W = SVector(0.2, 0.0, 1.0, -0.1)
        prods(yy, vv) = (gg = metric(hole, SVector(yy[1], yy[2], yy[3], yy[4]));
                         kk = inverse_metric(hole, SVector(yy[1], yy[2], yy[3], yy[4])) * SVector(yy[5], yy[6], yy[7], yy[8]);
                         (dot(kk, gg * vv[1]), dot(vv[1], gg * vv[2]), dot(vv[1], gg * vv[1])))
        p0 = prods(yh, (V, W))
        state = (yh, (V, W)); h = 0.02
        for _ in 1:1500                                # 30 M of affine parameter past the hole
            f = (s) -> geodesic_transport_rhs(hole, s[1], s[2])
            k1 = f(state); s2 = (state[1] + (h / 2) * k1[1], state[2] .+ (h / 2) .* k1[2])
            k2 = f(s2);    s3 = (state[1] + (h / 2) * k2[1], state[2] .+ (h / 2) .* k2[2])
            k3 = f(s3);    s4 = (state[1] + h * k3[1], state[2] .+ h .* k3[2])
            k4 = f(s4)
            state = (state[1] + (h / 6) * (k1[1] + 2k2[1] + 2k3[1] + k4[1]),
                     state[2] .+ (h / 6) .* (k1[2] .+ 2 .* k2[2] .+ 2 .* k3[2] .+ k4[2]))
        end
        p1 = prods(state...)
        @test abs(null_error(hole, state[1])) < 1e-9
        @test all(abs.(p1 .- p0) .< 1e-9 .* max.(1.0, abs.(p0)))
        @test state[1][5] != yh[5]                     # p_t drifted: the metric is time dependent
    end
end
