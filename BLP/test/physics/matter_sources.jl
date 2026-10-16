# The analytic sources against their closed forms: unit timelike four-velocity and unit
# spacelike field orthogonal to it, the torus potential and geometry, the Page & Thorne
# (1974) flux (counter-rotating through Bardeen, Press & Teukolsky 1972), the Keplerian
# hot-spot orbit, the on-axis field, the closed-torus window, the domain radii, and
# ForwardDiff through the states with one concrete state type on every branch.
using Test, BLP, StaticArrays, ForwardDiff, Random
using LinearAlgebra: dot
using BLP.Spacetime: KerrSchildCartesian, metric, metric_and_inverse, ks_radius, isco_radius,
                     keplerian_omega, keplerian_specific_angular_momentum
using BLP.Matter: PolishDoughnut, PageThorneDisk, EquatorialHotspot, PlasmaState, AnalyticProvider,
                  plasma_state, sample, domain_radius, vacuum, torus_potential, page_thorne_flux,
                  page_thorne_temperature, hotspot_center

state(src, st, x) = plasma_state(src, st, x, metric_and_inverse(st, x), nothing)
udotu(st, x, u) = dot(u, metric(st, x) * u)
# CKS point on the equator at Boyer–Lindquist radius r.
eqpoint(a, r) = SVector(0.0, sqrt(r^2 + a^2), 0.0, 0.0)
finite_state(s) = all(isfinite, s.ucon) && all(isfinite, s.bcon) && all(isfinite, s.bcov) &&
                  isfinite(s.n_e) && isfinite(s.theta_e) && isfinite(s.B)

# The kinematic contract of a valid analytic state at x.
function kinematics_ok(st, x, s)
    g = metric(st, x)
    return isapprox(dot(s.ucon, g * s.ucon), -1.0; atol = 1e-12) &&
           isapprox(dot(s.bcon, g * s.bcon), 1.0; atol = 1e-10) &&
           isapprox(dot(s.bcon, s.ucov), 0.0; atol = 1e-10) &&
           isapprox(s.ucov, g * s.ucon; rtol = 1e-14) && isapprox(s.bcov, g * s.bcon; rtol = 1e-14) &&
           s.b_sq ≈ 1.0 && s.sigma == 0.0 && s.beta_inv == 0.0
end

@testset "analytic sources" begin
    @testset "Polish doughnut" begin
        st = KerrSchildCartesian(0.0)
        d = PolishDoughnut(0.0; r_center = 9.0, n_e_center = 1e7, theta_e_center = 40.0,
                           polytrope_n = 1.5, beta = 8.0)
        @test d isa AnalyticProvider && isbits(d)
        xc = eqpoint(0.0, d.r_center)
        pc = state(d, st, xc)
        @test pc.valid && finite_state(pc) && kinematics_ok(st, xc, pc)
        @test !state(d, st, eqpoint(0.0, d.r_out + 5.0)).valid              # bounded
        @test state(d, st, SVector(0.0, 2.0, 0.0, 0.0)) === vacuum(Float64)  # inner funnel
        @test pc.n_e > 0 && pc.theta_e > 0 && pc.B > 0

        # the potential is minimal at the pressure maximum r_center
        W0 = torus_potential(d, st, xc)
        @test W0 < torus_potential(d, st, eqpoint(0.0, d.r_center - 0.5))
        @test W0 < torus_potential(d, st, eqpoint(0.0, d.r_center + 0.5))
        rs = range(d.r_cusp + 0.2, d.r_out - 0.2; length = 400)
        Ws = [torus_potential(d, st, eqpoint(0.0, r)) for r in rs]
        @test abs(rs[argmin(Ws)] - d.r_center) < 0.2

        # w = 1 at the centre: the central density and temperature; both peak there
        @test isapprox(pc.n_e, d.n_e_center; rtol = 1e-6)
        @test isapprox(pc.theta_e, d.theta_e_center; rtol = 1e-6)
        for r in (8.0, 10.0, 15.0, 18.0)
            p = state(d, st, eqpoint(0.0, r))
            p.valid || continue
            @test p.n_e <= pc.n_e + 1e-3
            @test p.theta_e <= pc.theta_e + 1e-9
        end
        # B from β and the pressure of electrons and ions at Θₑ
        p_gas = 2 * pc.n_e * pc.theta_e * BLP.Numerics.ME * BLP.Numerics.CL^2
        @test pc.B ≈ sqrt(8π * p_gas / 8.0)
        @test d.r_cusp < isco_radius(0.0) < d.r_center < d.r_out
    end

    @testset "Page–Thorne disk" begin
        @test isapprox(PageThorneDisk(0.0).r_isco, 6.0; atol = 1e-6)
        @test isapprox(PageThorneDisk(0.5).r_isco, 4.2330; atol = 2e-3)
        @test isapprox(PageThorneDisk(0.94).r_isco, 2.0236; atol = 2e-3)
        for a in (0.0, 0.5, 0.94)
            disk = PageThorneDisk(a; r_out = 40.0)
            riso = disk.r_isco
            @test page_thorne_flux(disk, riso) == 0.0
            @test page_thorne_flux(disk, riso + 0.5) > 0.0
            f_near = page_thorne_flux(disk, riso + 1.0)
            @test 0.0 < page_thorne_flux(disk, 800.0) < f_near
            @test page_thorne_temperature(disk, riso) == 0.0
            @test page_thorne_temperature(disk, riso + 1.0) > 0.0
            # finite luminosity: r F(r) ∝ r⁻² at large r
            r = range(riso, 500.0; length = 40000); dr = step(r)
            L = sum(2π * ri * page_thorne_flux(disk, ri) for ri in r) * dr
            @test isfinite(L) && L > 0
            integrand(rr) = rr * page_thorne_flux(disk, rr)
            @test isapprox(integrand(2000.0) / integrand(1000.0), 0.25; atol = 0.02)
        end

        st = KerrSchildCartesian(0.5)
        disk = PageThorneDisk(0.5; r_out = 30.0)
        xin = eqpoint(0.5, 10.0)
        pin = state(disk, st, xin)
        @test pin.valid && finite_state(pin) && kinematics_ok(st, xin, pin)
        @test pin.n_e > 0 && pin.B > 0
        # the in-plane density and temperature follow the flux profile
        Fn = page_thorne_flux(disk, 10.0) / disk.f_peak
        @test pin.n_e ≈ disk.n_e_max * Fn
        @test pin.theta_e ≈ disk.theta_e_max * Fn^(1 / 4)
        @test !state(disk, st, SVector(0.0, 10.012, 0.0, 5.0)).valid       # above the layer
        @test !state(disk, st, eqpoint(0.5, 3.0)).valid                     # inside the ISCO
    end

    @testset "equatorial hot spot" begin
        a = 0.0
        st = KerrSchildCartesian(a)
        r0 = 8.0
        spot = EquatorialHotspot(a; r_orbit = r0, t0 = 0.0, phi0 = 0.3, R_spot = 0.6,
                                 n_e0 = 1e6, theta_e0 = 50.0, B0 = 15.0)
        Ω = keplerian_omega(a, r0)
        Rc = sqrt(r0^2 + a^2)
        for t in (0.0, 3.0, 7.5, 11.0)
            c = hotspot_center(spot, t)
            φ = 0.3 + Ω * t
            @test isapprox(atan(c.y, c.x), atan(sin(φ), cos(φ)); atol = 1e-10)
            @test isapprox(hypot(c.x, c.y), Rc; atol = 1e-12)
            @test c.z == 0.0
            xe = SVector(t, c.x, c.y, c.z)
            pc = state(spot, st, xe)
            @test pc.valid && finite_state(pc) && kinematics_ok(st, xe, pc)
            @test isapprox(pc.n_e, spot.n_e0; rtol = 1e-6)
            @test pc.theta_e ≈ spot.theta_e0 && pc.B ≈ spot.B0
        end
        c0 = hotspot_center(spot, 0.0)
        @test !state(spot, st, SVector(0.0, c0.x, c0.y + 25.0, 0.0)).valid

        # the ambient background fills the domain away from the spot
        spot2 = EquatorialHotspot(a; r_orbit = r0, R_spot = 0.6, n_e0 = 1e6,
                                  ambient_n_e = 1e3, ambient_theta_e = 5.0, ambient_B = 1.0)
        pamb = state(spot2, st, SVector(0.0, c0.x, c0.y + 25.0, 0.0))
        @test pamb.valid && isapprox(pamb.n_e, 1e3; rtol = 1e-6)
        @test pamb.theta_e ≈ 5.0 && pamb.B ≈ 1.0

        # no circular orbit inside the photon orbit: the real square root throws
        @test_throws DomainError state(spot, st, SVector(0.0, 2.5, 0.0, 0.0))
    end

    @testset "AD through the states" begin
        a = 0.0
        st = KerrSchildCartesian(a)
        xe = SVector(2.0, 8.0 - 0.4, 0.5, 0.0)
        ne_of(r) = state(EquatorialHotspot(a; r_orbit = r, t0 = 0.0, phi0 = 0.0, R_spot = 0.6,
                                           n_e0 = 1e6, theta_e0 = 50.0, B0 = 15.0), st, xe).n_e
        g = ForwardDiff.derivative(ne_of, 8.0)
        h = 1e-6
        @test isfinite(g) && abs(g) > 0
        @test isapprox(g, (ne_of(8.0 + h) - ne_of(8.0 - h)) / 2h; rtol = 1e-6)

        # a Dual position: the vacuum and the plasma branch return one concrete type
        src = PolishDoughnut(0.0; r_center = 9.0, n_e_center = 1e7, theta_e_center = 40.0,
                             polytrope_n = 1.5, beta = 8.0)
        dual_x(x) = SVector(x[1], ForwardDiff.Dual(x[2], 1.0, 0.0, 0.0),
                            ForwardDiff.Dual(x[3], 0.0, 1.0, 0.0), ForwardDiff.Dual(x[4], 0.0, 0.0, 1.0))
        x_vac = dual_x(SVector(0.0, 2.0, 0.0, 0.0))
        x_in = dual_x(SVector(0.0, src.r_center, 0.0, 0.0))
        rv = state(src, st, x_vac); ri = state(src, st, x_in)
        @test !rv.valid && ri.valid
        @test typeof(rv) == typeof(ri) && rv.n_e isa ForwardDiff.Dual
        g = metric_and_inverse(st, x_vac)
        for p in (src, PageThorneDisk(0.5), EquatorialHotspot(0.5))
            rt = Base.return_types(plasma_state, (typeof(p), typeof(st), typeof(x_vac), typeof(g), Nothing))
            @test isconcretetype(only(rt))
        end
        # a Dual source parameter with a Float64 position
        sd = EquatorialHotspot(0.0; r_orbit = ForwardDiff.Dual(8.0, 1.0))
        x64 = SVector(0.0, 7.6, 0.5, 0.0); g64 = metric_and_inverse(st, x64)
        rt = Base.return_types(plasma_state, (typeof(sd), typeof(st), typeof(x64), typeof(g64), Nothing))
        @test isconcretetype(only(rt)) && only(rt) <: PlasmaState{<:ForwardDiff.Dual}
        @test state(sd, st, SVector(0.0, 0.0, 30.0, 0.0)) isa only(rt)
        # Float64 inputs keep the Float64 vacuum
        @test state(src, st, SVector(0.0, 2.0, 0.0, 0.0)) === vacuum(Float64)
    end

    @testset "counter-rotation and edge cases" begin
        for (a, r) in ((0.9, 10.0), (0.5, 8.0), (0.94, 20.0))
            st = KerrSchildCartesian(a)
            x = eqpoint(a, r)
            g = metric(st, x)
            Ω = keplerian_omega(a, r; prograde = false)
            xi = SVector(1.0, 0.0, 0.0, 0.0); eta = SVector(0.0, -x[3], x[2], 0.0)
            ubar = xi + Ω * eta
            ucov = g * (ubar / sqrt(-dot(ubar, g * ubar)))
            ell = keplerian_specific_angular_momentum(a, r; prograde = false)
            @test ell < 0
            @test isapprox(ell, -dot(eta, ucov) / dot(xi, ucov); rtol = 1e-10)
        end

        a = 0.9
        st = KerrSchildCartesian(a)
        d = PolishDoughnut(a; lambda = 0.5, prograde = false)
        @test d.l0 < 0
        xc = eqpoint(a, d.r_center)
        pc = state(d, st, xc)
        @test pc.valid && kinematics_ok(st, xc, pc)
        @test xc[2] * pc.ucon[3] - xc[3] * pc.ucon[2] < 0                 # counter-rotating
        dp = PolishDoughnut(a; lambda = 0.5, prograde = true)
        xcp = eqpoint(a, dp.r_center)
        pcp = state(dp, st, xcp)
        @test xcp[2] * pcp.ucon[3] - xcp[3] * pcp.ucon[2] > 0

        disk = PageThorneDisk(a; prograde = false, r_out = 60.0)
        @test isapprox(disk.r_isco, 8.71735; atol = 1e-3)
        for xr in (disk.x1, disk.x2, disk.x3)
            @test isapprox(xr^3 - 3xr - 2a, 0.0; atol = 1e-10)
        end
        function pt_ref(aeff, r, r_isco)
            r <= r_isco && return 0.0
            x = sqrt(r); x0 = sqrt(r_isco); ac = acos(aeff) / 3
            x1 = 2cos(ac - π / 3); x2 = 2cos(ac + π / 3); x3 = -2cos(ac)
            c(xi, xj, xk) = 3 * (xi - aeff)^2 / (xi * (xi - xj) * (xi - xk))
            br = x - x0 - 1.5aeff * log(x / x0) -
                 c(x1, x2, x3) * log((x - x1) / (x0 - x1)) -
                 c(x2, x1, x3) * log((x - x2) / (x0 - x2)) -
                 c(x3, x1, x2) * log((x - x3) / (x0 - x3))
            return 1.5 / (x^2 * (x^3 - 3x + 2aeff)) * br / (4π * r)
        end
        for r in (disk.r_isco + 0.5, 12.0, 20.0, 50.0)
            @test page_thorne_flux(disk, r) > 0
            @test isapprox(page_thorne_flux(disk, r), pt_ref(-a, r, disk.r_isco); rtol = 1e-8)
        end
        @test page_thorne_flux(disk, disk.r_isco) == 0.0
        disk2 = PageThorneDisk(-a; prograde = true, r_out = 60.0)
        @test disk2.a == a && !disk2.prograde && disk2.r_isco == disk.r_isco
        @test all(page_thorne_flux(disk2, r) == page_thorne_flux(disk, r) for r in (9.0, 12.0, 20.0))

        # the toroidal direction on the spin axis stays a finite unit vector
        for a in (0.0, 0.5, 0.9375), z in (4.0, 8.0, 20.0)
            st = KerrSchildCartesian(a)
            spot = EquatorialHotspot(a; ambient_n_e = 1e3, ambient_theta_e = 5.0, ambient_B = 1.0)
            x = SVector(0.0, 0.0, 0.0, z)
            pl = state(spot, st, x)
            @test pl.valid && all(isfinite, pl.bcon) && kinematics_ok(st, x, pl)
        end

        a = 0.5
        l_ms = keplerian_specific_angular_momentum(a, isco_radius(a))
        @test_throws ArgumentError PolishDoughnut(a; l0 = 0.8 * l_ms)
        d = PolishDoughnut(a; lambda = 0.5)
        @test isfinite(d.r_out) && d.r_cusp < d.r_center < d.r_out
        @test PolishDoughnut(a; r_center = 25.0) isa PolishDoughnut       # overflowing cusp
    end

    @testset "domain radius and sample" begin
        a = 0.9375
        st = KerrSchildCartesian(a)
        d = PolishDoughnut(a; lambda = 0.5)
        disk = PageThorneDisk(a; r_out = 25.0)
        spot = EquatorialHotspot(a; r_orbit = 7.0, R_spot = 0.8)
        @test domain_radius(d) == floor(d.r_out) + 1 && domain_radius(d) > d.r_out
        @test domain_radius(disk) == 25.0
        @test domain_radius(spot) == floor(sqrt(7.0^2 + a^2) + 3.2) + 1
        @test domain_radius(EquatorialHotspot(a; ambient_n_e = 1.0)) == Inf
        @test domain_radius(EquatorialHotspot(a; r_orbit = ForwardDiff.Dual(7.0, 1.0), R_spot = 0.8)) ==
              domain_radius(spot)
        @test all(domain_radius(p) isa Float64 for p in (d, disk, spot))

        # every sample beyond the domain radius is vacuum, at every time for the spot
        rng = Xoshiro(7)
        for p in (d, disk, spot), k in 1:400
            v = randn(rng, SVector{3,Float64}); v /= sqrt(sum(abs2, v))
            R = domain_radius(p) * (1 + 0.5rand(rng))
            x = SVector(40 * randn(rng), R * v...)
            ks_radius(st, x) > domain_radius(p) || continue
            @test !state(p, st, x).valid
        end
        # near the spot, at the centre's time, inside the radius
        c = hotspot_center(spot, 5.0)
        @test state(spot, st, SVector(5.0, c.x + 0.3, c.y, 0.1)).valid

        # `sample` evaluates the metric pair once and returns it with the state; the
        # contractions may round differently once inlined into the caller
        x = eqpoint(a, d.r_center)
        s, g = sample(d, st, x, nothing)
        s0 = state(d, st, x)
        @test g === metric_and_inverse(st, x) && s.valid
        @test all(isapprox.((s.n_e, s.theta_e, s.B), (s0.n_e, s0.theta_e, s0.B); rtol = 1e-14))
        @test isapprox(s.ucon, s0.ucon; rtol = 1e-14) && isapprox(s.bcon, s0.bcon; rtol = 1e-14)
    end
end
