# The march integrators: the order of the ninth-order step, the RK4 step bit for bit that
# of rk4_step and of the polarized march, the frame transport, the surface crossing, and
# the camera at infinity, whose rays carry Bardeen's conserved quantities.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, Minkowski, ClassicalRK4,
    Verner9, erk_step, march_step, transport_step, rk4_step, geodesic_rhs, geodesic_transport_rhs,
    EquatorialPlane, surface_crossing, surface_value, camera_ray, CameraFrame, PlaneParallelCamera,
    infinity_ray, raise_index, raise_index_offset, metric, inverse_metric, ks_radius,
    march_integrator, integrator_stages, null_covariant, MarchBudget, march_control, winding_factor

# Carter constant and L_z of a Kerr-Schild state (unit mass, spin a along z)
function carter(st, y)
    a = st.a
    r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
    z = y[4]; px, py, pz = y[6], y[7], y[8]; E = y[5]
    L = y[2] * py - y[3] * px
    c2 = z * z / (r * r); s2 = 1 - c2; A = y[2] * px + y[3] * py
    Q = if s2 > 0.1
        pth = sqrt(c2 / s2) * sign(z) * A - r * sqrt(s2) * pz
        pth * pth + c2 * (L * L / s2 - a * a * E * E)
    else
        c2 * ((r * r + a * a) * (px * px + py * py) - a * a * E * E) - 2 * z * A * pz + (r * r - z * z) * pz * pz
    end
    return L, Q
end

allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

@testset "march integrators" begin
    @testset "order of the steps" begin
        f(y) = SVector(y[2], -y[1])                        # z'' = -z, exact (cos, -sin)
        err(integ, n) = (h = 1.0 / n; y = SVector(1.0, 0.0);
                         for _ in 1:n; y = erk_step(integ, f, y, h); end;
                         norm(y - SVector(cos(1.0), -sin(1.0))))
        @test 12 < err(ClassicalRK4(), 8) / err(ClassicalRK4(), 16) < 20
        @test err(Verner9(), 2) / err(Verner9(), 4) > 300          # 2^9 = 512
        @test err(Verner9(), 4) < 1e-10
        @test integrator_stages(ClassicalRK4()) == 4 && integrator_stages(Verner9()) == 15
        @test march_integrator(:rk4) === ClassicalRK4() && march_integrator(:verner9) === Verner9()
        @test_throws ArgumentError march_integrator(:euler)
    end

    st = KerrSchildCartesian(0.94)
    bin = SuperposedKerrSchild([0.5, 0.5], [0.3, -0.2], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)])
    ys = (SVector(0.0, 7.0, 3.0, 2.0, 1.0, -0.9, 0.05, 0.02),
          SVector(0.0, 1.6, 0.4, -0.3, 1.0, 0.7, -1.1, 0.4))

    @testset "ClassicalRK4 is rk4_step bit for bit" begin
        for s in (st, bin, Minkowski()), y in ys, h in (0.01, 0.3)
            @test march_step(ClassicalRK4(), s, y, h) === rk4_step(s, y, h)
            # the generic tableau step compiles apart from rk4_step, so operation fusion may
            # differ by context: the same arithmetic to a few ulps
            @test isapprox(erk_step(ClassicalRK4(), z -> geodesic_rhs(s, z), y, h), rk4_step(s, y, h);
                           rtol = 8eps())
        end
    end

    @testset "frame transport" begin
        vs = (SVector(0.0, 0.3, -0.2, 1.0), SVector(1.0, 0.1, 0.0, 0.0))
        # the streaming RK4 of the polarized march, written out
        function rk4_frame(s, y, eU, eUu, h)
            dy, (de, du) = geodesic_transport_rhs(s, y, (eU, eUu))
            ay = dy; ae = de; au = du
            yq = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
            dy, (de, du) = geodesic_transport_rhs(s, yq, (es, us))
            ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
            yq = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
            dy, (de, du) = geodesic_transport_rhs(s, yq, (es, us))
            ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
            yq = y + h*dy; es = eU + h*de; us = eUu + h*du
            dy, (de, du) = geodesic_transport_rhs(s, yq, (es, us))
            ay = ay + dy; ae = ae + de; au = au + du
            return y + (h/6)*ay, (eU + (h/6)*ae, eUu + (h/6)*au)
        end
        for s in (st, bin), y in ys                        # the same operations, compiled apart
            a, (b1, b2) = transport_step(ClassicalRK4(), s, y, vs, 0.05)
            c, (c1, c2) = rk4_frame(s, y, vs..., 0.05)
            @test isapprox(a, c; rtol = 1e-15) && isapprox(b1, c1; rtol = 1e-15) &&
                  isapprox(b2, c2; rtol = 1e-15)
        end
        # the ninth-order transport keeps g(V, V), g(V, W) and g(V, k) along a ray
        y, _ = camera_ray(st, CameraFrame(PlaneParallelCamera(inc_deg = 60.0, d = 50.0)), 7.0, -3.0)
        g0 = metric(st, SVector(y[1], y[2], y[3], y[4]))
        k0 = inverse_metric(st, SVector(y[1], y[2], y[3], y[4])) * SVector(y[5], y[6], y[7], y[8])
        dots(g, k, v) = (dot(v[1], g * v[1]), dot(v[1], g * v[2]), dot(v[1], g * k))
        d0 = dots(g0, k0, vs)
        w = vs
        for _ in 1:150
            r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
            y, w = transport_step(Verner9(), st, y, w, (r - st.r_h) / 20)
        end
        @test ks_radius(st, SVector(y[1], y[2], y[3], y[4])) > 4      # past the closest approach
        x = SVector(y[1], y[2], y[3], y[4])
        k = inverse_metric(st, x) * SVector(y[5], y[6], y[7], y[8])
        @test maximum(abs.(dots(metric(st, x), k, w) .- d0)) < 1e-11
    end

    @testset "surface crossing" begin
        cam = CameraFrame(PlaneParallelCamera(inc_deg = 60.0, d = 50.0))
        y, _ = camera_ray(st, cam, 7.0, -3.0)
        H = 0.0
        for _ in 1:10_000                                          # march to the step that crosses z = 0
            r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
            H = (r - st.r_h) / 20
            yn = march_step(Verner9(), st, y, H)
            signbit(yn[4]) != signbit(y[4]) && break
            y = yn
        end
        yc, c = surface_crossing(Verner9(), st, y, H, EquatorialPlane())
        @test 0 < c < H && abs(surface_value(EquatorialPlane(), SVector(yc[1], yc[2], yc[3], yc[4]))) < 1e-14
        # the same point by 64 substeps to c
        yf = y
        for _ in 1:64
            yf = march_step(Verner9(), st, yf, c / 64)
        end
        @test maximum(abs.(yc - yf)) < 1e-13
        yr, cr = surface_crossing(ClassicalRK4(), st, y, H, EquatorialPlane())
        @test abs(yr[4]) < 1e-14 && abs(cr - c) < 1e-6 * H
    end

    @testset "camera at infinity" begin
        # flat space: the ray from infinity is the straight plane-parallel ray
        for inc in (17.0, 80.0)
            fi = CameraFrame(PlaneParallelCamera(inc_deg = inc, d = 1000.0, infinity = true))
            fp = CameraFrame(PlaneParallelCamera(inc_deg = inc, d = 1000.0))
            yi, ok = infinity_ray(Minkowski(), fi, 3.0, -4.0)
            yp, _ = camera_ray(Minkowski(), fp, 3.0, -4.0)
            @test ok && maximum(abs.(yi - yp)) < 1e-12
        end
        # Kerr: λ = -α sin i and η = (α² - a²) cos² i + β² of Bardeen's screen point
        worst_inf = 0.0; best_plane = Inf
        for a in (0.0, 0.94, 0.99), inc in (17.0, 60.0, 80.0)
            k = KerrSchildCartesian(a)
            fi = CameraFrame(PlaneParallelCamera(inc_deg = inc, d = 1000.0, infinity = true))
            fp = CameraFrame(PlaneParallelCamera(inc_deg = inc, d = 1000.0))
            for (α, β) in ((3.0, 4.0), (-5.5, 0.7), (0.2, -6.0), (20.0, -15.0))
                λb = -α * sind(inc); ηb = (α^2 - a^2) * cosd(inc)^2 + β^2
                yi, ok = camera_ray(k, fi, α, β)
                L, Q = carter(k, big.(yi))                  # the state's charges, evaluated exactly
                @test ok && yi[5] == 1.0
                worst_inf = max(worst_inf, abs(-L - λb) / hypot(λb, 1), abs(Q - ηb) / ηb)
                yp, _ = camera_ray(k, fp, α, β)
                Lp, Qp = carter(k, big.(yp))
                best_plane = min(best_plane, max(abs(-Lp - λb) / hypot(λb, 1), abs(Qp - ηb) / ηb))
            end
        end
        @test worst_inf < 1e-13
        @test best_plane > 1e-8                             # the camera at 1000 M is not at infinity
        # the Woodbury offset of a one-hole superposition gives the Kerr ray
        one = SuperposedKerrSchild([1.0], [0.94], [(0.0, 0.0, 0.0)])
        fi = CameraFrame(PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, infinity = true))
        y1, _ = infinity_ray(one, fi, 3.0, 4.0); y2, _ = infinity_ray(st, fi, 3.0, 4.0)
        @test maximum(abs.(y1[2:8] - y2[2:8])) < 1e-12
        p = SVector(1.0, -0.3, 0.2, -0.9); x = SVector(0.0, 300.0, 20.0, 900.0)
        @test raise_index_offset(st, x, p) ≈ raise_index(st, x, p) - SVector(-p[1], p[2], p[3], p[4]) atol = 1e-15
        @test raise_index_offset(one, x, p) ≈ raise_index_offset(st, x, p) rtol = 1e-12
    end

    @testset "spherical photon orbits" begin
        # orbits A (r0 = 1.8) and E (r0 = 3) of an extremal hole (Teo 2003): E = 1,
        # L_z = Φ(r0), Q(r0), started on the equator moving up with dr/dλ = 0. Unstable: an
        # offset grows by e every 1.5-2.2 M, so the radius error after 20 M measures the
        # integrator's seed.
        k = KerrSchildCartesian(1.0)
        Φ(r) = -(r^3 - 3r^2 + r + 1) / (r - 1)
        Qs(r) = -r^3 * (r^3 - 6r^2 + 9r - 4) / (r - 1)^2
        for (r0, h, tol) in ((1.8, 1 / 16, 1e-8), (3.0, 1 / 8, 1e-10))
            x = SVector(0.0, sqrt(r0^2 + 1), 0.0, 0.0)
            py = Φ(r0) / x[2]; pz = sqrt(Qs(r0)) / r0
            gi = inverse_metric(k, x)
            px = -(gi[2, 1] * -1.0 + gi[2, 3] * py + gi[2, 4] * pz) / gi[2, 2]
            y = SVector(0.0, x[2], 0.0, 0.0, -1.0, px, py, pz)
            dr = 0.0
            for _ in 1:round(Int, 20 / h)
                y = march_step(Verner9(), k, y, h)
                dr = max(dr, abs(ks_radius(k, SVector(y[1], y[2], y[3], y[4])) - r0))
            end
            @test dr < tol
        end
    end

    @testset "budget: capture, backstop, winding" begin
        b6 = MarchBudget(20.0, 1100.0, Inf, 1000; integrator = Verner9(), capture = 1e6,
                         horizon_eps = 0.0, winding = true)
        @test isbits(b6) && b6.integrator === Verner9() && b6.capture == 1e6 && b6.winding
        @test MarchBudget(80.0, 1100.0, Inf, 6000) === MarchBudget{Float64}(80.0, 1100.0, Inf, 6000)
        # a state 0.3 M above the horizon with E_n ≈ 500: Kerr marches on under a cut of 1e6,
        # a superposition captures at its cut of at most 1e2
        y = SVector(0.0, sqrt((st.r_h + 0.3)^2 + 0.94^2), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)   # r = r_+ + 0.3
        pk = null_covariant(st, SVector(y[1], y[2], y[3], y[4]), SVector(-400.0, 50.0, 0.0))[1]
        yk = SVector(y[1], y[2], y[3], y[4], pk[1], pk[2], pk[3], pk[4])
        @test !march_control(st, yk, Inf, b6).stop && march_control(st, yk, Inf,
              MarchBudget(20.0, 1100.0, Inf, 1000)).stop
        one = SuperposedKerrSchild([1.0], [0.94], [(0.0, 0.0, 0.0)])
        @test march_control(one, yk, Inf, b6).stop
        # the winding factor: 1 at a = 0 and far out, ≈ (r_+ - r_-)/(2a) at the horizon
        @test winding_factor(2.5, 2.0, 0.0) == 1.0 && winding_factor(1e3, st.r_h, 0.94) > 0.99999
        rm = 0.94^2 / st.r_h
        @test winding_factor(st.r_h, st.r_h, 0.94) ≈ (st.r_h - rm) / (2 * 0.94) rtol = 0.02
        h1 = march_control(st, yk, Inf, b6).h
        @test h1 ≈ 0.3 * winding_factor(st.r_h + 0.3, st.r_h, 0.94) / 20 rtol = 1e-3
    end

    @testset "zero allocations" begin
        y = ys[1]; vs = (SVector(0.0, 0.3, -0.2, 1.0), SVector(1.0, 0.1, 0.0, 0.0))
        fi = CameraFrame(PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, infinity = true))
        for s in (st, bin)
            @test allocs(march_step, Verner9(), s, y, 0.1) == 0
            @test allocs(transport_step, Verner9(), s, y, vs, 0.1) == 0
            @test allocs(surface_crossing, Verner9(), s, y, 0.1, EquatorialPlane()) == 0
            @test allocs(infinity_ray, s, fi, 3.0, 4.0) == 0
        end
    end
end
