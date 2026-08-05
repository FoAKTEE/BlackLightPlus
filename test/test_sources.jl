# Analytic source-model tests (Sources: PolishDoughnut, PageThorneDisk,
# EquatorialHotspot). Self-contained: ground truth is analytic / self-consistent
# — no external data, no reference binaries. Checks include u.u = -1 and unit
# spacelike b orthogonal to u, torus potential/geometry closed forms, the Page &
# Thorne (1974) flux (retrograde via Bardeen, Press & Teukolsky 1972), Keplerian
# hotspot orbits, and ForwardDiff differentiability through source_plasma_at.
# Each source implements source_plasma_at(src, st, x) -> the grmhd plasma_at
# NamedTuple, so it plugs into the existing Radiation coefficients unchanged
# (see the integration testset).

using Test
using StaticArrays
using ForwardDiff
using LinearAlgebra: dot
using BlackLightPlus
using BlackLightPlus: metric, inverse_metric, ks_radius, KerrSchildCartesian,
                    nullify_covariant, fluid_tetrad,
                    thermal_synchrotron, thermal_synchrotron_polarized

# --- helpers ---------------------------------------------------------------

# u.u using the metric at x; must equal -1 for a physical 4-velocity.
udotu(st, x, ucon) = dot(ucon, metric(st, x) * ucon)

# A CKS point on the equator at Boyer–Lindquist radius r (KS cyl. radius sqrt(r^2+a^2)).
eqpoint(a, r) = SVector(0.0, sqrt(r^2 + a^2), 0.0, 0.0)

# valid grmhd-style tuple: all fields present, finite, right shapes.
function isvalid_tuple(pl)
    for k in (:n_e, :theta_e, :B, :ucon, :ucov, :bcon, :gcov, :gcon, :inside)
        haskey(pl, k) || return false
    end
    all(isfinite, pl.ucon) && all(isfinite, pl.gcov) &&
        isfinite(pl.n_e) && isfinite(pl.theta_e) && isfinite(pl.B)
end

# pitch angle sin/cos exactly as radiation/integrator.jl::eight_coefficients does,
# from the fluid tetrad built out of the source's own (u, b) and a photon (k).
function pitch_angle(st, x, pl, kcov)
    gcon = pl.gcon
    kcon = gcon * kcov
    bcov = pl.gcov * pl.bcon
    tet = fluid_tetrad(pl.ucon, pl.ucov, kcon, kcov, pl.bcon, pl.gcov, gcon)
    kt1 = kt2 = kt3 = 0.0
    bt1 = bt2 = bt3 = 0.0
    @inbounds for μ in 1:4
        kt1 += tet[2, μ] * kcov[μ]; kt2 += tet[3, μ] * kcov[μ]; kt3 += tet[4, μ] * kcov[μ]
        bt1 += tet[2, μ] * bcov[μ]; bt2 += tet[3, μ] * bcov[μ]; bt3 += tet[4, μ] * bcov[μ]
    end
    k2t = kt1^2 + kt2^2 + kt3^2
    b2t = bt1^2 + bt2^2 + bt3^2
    kbt = kt1 * bt1 + kt2 * bt2 + kt3 * bt3
    cos2 = min(kbt^2 / (k2t * b2t), 1.0)
    return sqrt(1.0 - cos2), sqrt(cos2) * (kbt >= 0.0 ? 1.0 : -1.0)
end

@testset "Sources — analytic plasma library" begin

    # ======================================================= PolishDoughnut
    @testset "PolishDoughnut" begin
        st = KerrSchildCartesian(0.0)
        # r_center=9 => l0 = l_K(9) ∈ (l_ms, l_mb) => a CLOSED (Roche-lobe-filling) torus.
        d = PolishDoughnut(0.0; r_center=9.0, n_e_center=1e7, theta_e_center=40.0,
                           polytrope_n=1.5, beta=8.0)

        xc = eqpoint(0.0, d.r_center)
        pc = source_plasma_at(d, st, xc)
        far = source_plasma_at(d, st, eqpoint(0.0, d.r_out + 5.0))
        funnel = source_plasma_at(d, st, SVector(0.0, 2.0, 0.0, 0.0))

        @test isvalid_tuple(pc)
        @test pc.inside                          # centre is inside
        @test !far.inside                        # bounded: far away is vacuum
        @test !funnel.inside                     # inner funnel excluded
        @test pc.n_e > 0 && pc.theta_e > 0 && pc.B > 0

        # (b) 4-velocity normalization u.u = -1.
        @test isapprox(udotu(st, xc, pc.ucon), -1.0; atol=1e-12)
        # bcon is a unit spacelike field direction orthogonal to u.
        @test isapprox(dot(pc.bcon, pc.gcov * pc.bcon), 1.0; atol=1e-10)
        @test isapprox(dot(pc.bcon, pc.ucov), 0.0; atol=1e-10)

        # (c) potential minimum sits at the pressure-max radius r_center.
        W0 = torus_potential(d, st, xc)
        @test W0 < torus_potential(d, st, eqpoint(0.0, d.r_center - 0.5))
        @test W0 < torus_potential(d, st, eqpoint(0.0, d.r_center + 0.5))
        # scan the equatorial potential: global min within the torus at r_center.
        rs = range(d.r_cusp + 0.2, d.r_out - 0.2; length=400)
        Ws = [torus_potential(d, st, eqpoint(0.0, r)) for r in rs]
        @test abs(rs[argmin(Ws)] - d.r_center) < 0.2

        # (c) density & temperature peak at the centre; w=1 there.
        @test isapprox(pc.n_e, d.n_e_center; rtol=1e-6)
        @test isapprox(pc.theta_e, d.theta_e_center; rtol=1e-6)
        for r in (8.0, 10.0, 15.0, 18.0)
            p = source_plasma_at(d, st, eqpoint(0.0, r))
            p.inside || continue
            @test p.n_e <= pc.n_e + 1e-3
            @test p.theta_e <= pc.theta_e + 1e-9
        end

        # geometry sanity: cusp inside ISCO, centre outside, torus bracketed.
        @test d.r_cusp < isco_radius(0.0) < d.r_center < d.r_out
    end

    # ======================================================= PageThorneDisk
    @testset "PageThorneDisk" begin
        # (d) ISCO matches the known Kerr values at a in {0, 0.5, 0.94}.
        @test isapprox(PageThorneDisk(0.0).r_isco, 6.0; atol=1e-6)
        @test isapprox(PageThorneDisk(0.5).r_isco, 4.2330; atol=2e-3)
        @test isapprox(PageThorneDisk(0.94).r_isco, 2.0236; atol=2e-3)

        for a in (0.0, 0.5, 0.94)
            disk = PageThorneDisk(a; r_out=40.0)
            riso = disk.r_isco
            # (d) F = 0 at (and inside) the ISCO, > 0 just outside.
            @test page_thorne_flux(disk, riso) == 0.0
            @test page_thorne_flux(disk, riso + 0.5) > 0.0
            # (d) F -> 0 at large r, and below the near-ISCO value.
            f_near = page_thorne_flux(disk, riso + 1.0)
            @test 0.0 < page_thorne_flux(disk, 800.0) < f_near
            # effective temperature tracks F: zero at ISCO, positive outside.
            @test page_thorne_temperature(disk, riso) == 0.0
            @test page_thorne_temperature(disk, riso + 1.0) > 0.0

            # (d) total luminosity integral L = ∫ 2π r F dr is finite & positive,
            #     and CONVERGES: the emitting integrand r F(r) ~ r^-2 at large r,
            #     so the tail (r >= 500) is a negligible fraction of the peak.
            r = range(riso, 500.0; length=40000); dr = step(r)
            L = sum(2π * ri * page_thorne_flux(disk, ri) for ri in r) * dr
            @test isfinite(L) && L > 0
            integrand(rr) = rr * page_thorne_flux(disk, rr)
            # r F(r) ~ r^-2 => halving r-spacing quarters the integrand: converges.
            @test isapprox(integrand(2000.0) / integrand(1000.0), 0.25; atol=0.02)
        end

        # volumetric plasma: thin equatorial band, r > ISCO, u.u = -1.
        st = KerrSchildCartesian(0.5)
        disk = PageThorneDisk(0.5; r_out=30.0)
        xin = eqpoint(0.5, 10.0)
        pin = source_plasma_at(disk, st, xin)
        @test pin.inside && isvalid_tuple(pin)
        @test pin.n_e > 0 && pin.B > 0
        @test isapprox(udotu(st, xin, pin.ucon), -1.0; atol=1e-12)
        # thin: 5 GM/c^2 above the plane is outside the disk; inside ISCO is vacuum.
        @test !source_plasma_at(disk, st, SVector(0.0, 10.012, 0.0, 5.0)).inside
        @test !source_plasma_at(disk, st, eqpoint(0.5, 3.0)).inside
    end

    # ======================================================= EquatorialHotspot
    @testset "EquatorialHotspot" begin
        a = 0.0
        st = KerrSchildCartesian(a)
        r0 = 8.0
        spot = EquatorialHotspot(a; r_orbit=r0, t0=0.0, phi0=0.3, R_spot=0.6,
                                 n_e0=1e6, theta_e0=50.0, B0=15.0)
        Ω = keplerian_omega(a, r0)
        Rc = sqrt(r0^2 + a^2)

        # (e) the spot centre follows the Keplerian orbit phi = phi0 + Ω (t - t0).
        for t in (0.0, 3.0, 7.5, 11.0)
            xc, yc, zc = hotspot_center(spot, t)
            phi_expect = 0.3 + Ω * t
            @test isapprox(atan(yc, xc), atan(sin(phi_expect), cos(phi_expect)); atol=1e-10)
            @test isapprox(hypot(xc, yc), Rc; atol=1e-12)
            @test zc == 0.0

            # peak density sits at the moving centre; u.u = -1 there.
            xeval = SVector(t, xc, yc, zc)
            pc = source_plasma_at(spot, st, xeval)
            @test pc.inside && isvalid_tuple(pc)
            @test isapprox(pc.n_e, spot.n_e0; rtol=1e-6)
            @test isapprox(udotu(st, xeval, pc.ucon), -1.0; atol=1e-12)
        end

        # (a) far from the spot => vacuum (no ambient background here).
        xc0, yc0, _ = hotspot_center(spot, 0.0)
        @test !source_plasma_at(spot, st, SVector(0.0, xc0, yc0 + 25.0, 0.0)).inside

        # ambient background fills the domain even away from the spot.
        spot2 = EquatorialHotspot(a; r_orbit=r0, R_spot=0.6, n_e0=1e6,
                                  ambient_n_e=1e3, ambient_theta_e=5.0, ambient_B=1.0)
        pamb = source_plasma_at(spot2, st, SVector(0.0, xc0, yc0 + 25.0, 0.0))
        @test pamb.inside && isapprox(pamb.n_e, 1e3; rtol=1e-6)
    end

    # ======================================================= AD through the seam
    @testset "AD: d(source_plasma_at)/d(r_orbit)" begin
        a = 0.0
        st = KerrSchildCartesian(a)
        r0 = 8.0
        # evaluate slightly OFF the spot centre so n_e depends nontrivially on r_orbit.
        Rc = sqrt(r0^2 + a^2)
        xeval = SVector(2.0, Rc - 0.4, 0.5, 0.0)

        ne_of(rorb) = source_plasma_at(
            EquatorialHotspot(a; r_orbit=rorb, t0=0.0, phi0=0.0, R_spot=0.6,
                              n_e0=1e6, theta_e0=50.0, B0=15.0),
            st, xeval).n_e

        g = ForwardDiff.derivative(ne_of, r0)
        @test isfinite(g)
        @test abs(g) > 0        # r_orbit moves the spot => nonzero density gradient
    end

    # ============================================ INTEGRATION PROOF (the payoff)
    # Feed a source straight into the EXISTING Radiation coefficients: a torus
    # sample -> fluid tetrad pitch angle -> thermal synchrotron. Finite, positive
    # emissivity inside the doughnut with NO source-specific emission code.
    @testset "integration: source -> Radiation coefficients" begin
        st = KerrSchildCartesian(0.0)
        d = PolishDoughnut(0.0; r_center=12.0, n_e_center=1e7, theta_e_center=40.0,
                           beta=8.0)
        x = eqpoint(0.0, 12.0)
        pl = source_plasma_at(d, st, x)
        @test pl.inside

        # a photon momentum at x (spatial covariant guess -> nullified), then the
        # orthonormal-frame pitch angle, exactly like eight_coefficients.
        kcov = nullify_covariant(st, x, SVector(0.3, 0.15, 0.2))
        sin_th, cos_th = pitch_angle(st, x, pl, kcov)
        @test 0.0 < sin_th <= 1.0

        ν = 230.0e9
        j_i, α_i = thermal_synchrotron(1.0, pl.n_e, pl.theta_e, pl.B, ν, sin_th)
        @test j_i > 0.0 && α_i > 0.0                    # finite positive emissivity

        # full 8 polarized coefficients also fire (j_i>0, j_q<0 as usual).
        ji, jq, jv, ai, aq, av, rq, rv =
            thermal_synchrotron_polarized(1.0, pl.n_e, pl.theta_e, pl.B, ν, sin_th, cos_th)
        @test ji > 0.0 && jq < 0.0 && ai > 0.0
        @test all(isfinite, (ji, jq, jv, ai, aq, av, rq, rv))
    end
end

# ============================================================================
# Regression tests for retrograde / edge-case fixes:
#   * kerr_orbits.jl  — retrograde specific angular momentum sign (BPT lower sign)
#   * page_thorne.jl  — retrograde flux via the a -> -a closed form (+ signed-spin)
#   * sources.jl      — on-axis toroidal direction no longer DomainError-crashes
#   * polish_doughnut.jl — closed-torus window guard on l0
# ============================================================================
@testset "retrograde kinematics & source edge-case fixes" begin

    # -- retrograde l_K carries BPT-1972's leading minus (l = -u_phi/u_t < 0) --
    @testset "retrograde l_K sign" begin
        for (a, r) in ((0.9, 10.0), (0.5, 8.0), (0.94, 20.0))
            st = KerrSchildCartesian(a)
            x = eqpoint(a, r)
            g = metric(st, x)
            Ω = keplerian_omega(a, r; prograde=false)
            xi = SVector(1.0, 0.0, 0.0, 0.0); eta = SVector(0.0, -x[3], x[2], 0.0)
            ubar = xi + Ω * eta
            ut = 1 / sqrt(-dot(ubar, g * ubar))
            ucov = g * (ut * ubar)
            ell_kin = -dot(eta, ucov) / dot(xi, ucov)        # -u_phi/u_t, retro
            ell = keplerian_specific_angular_momentum(a, r; prograde=false)
            @test ell < 0                                    # counter-rotating => < 0
            @test isapprox(ell, ell_kin; rtol=1e-10)
            @test keplerian_specific_angular_momentum(a, r; prograde=true) > 0  # unchanged
        end
    end

    # -- a retrograde Polish doughnut actually counter-rotates ---------------
    @testset "PolishDoughnut retrograde rotation" begin
        a = 0.9
        st = KerrSchildCartesian(a)
        d = PolishDoughnut(a; lambda=0.5, prograde=false)
        @test d.l0 < 0
        xc = eqpoint(a, d.r_center)
        pc = source_plasma_at(d, st, xc)
        @test pc.inside && isvalid_tuple(pc)
        @test isapprox(udotu(st, xc, pc.ucon), -1.0; atol=1e-12)
        # u^phi proxy  x u^y - y u^x < 0  => genuinely retrograde
        @test xc[2] * pc.ucon[3] - xc[3] * pc.ucon[2] < 0
        # prograde companion spins the OTHER way (the sign really flips)
        dp = PolishDoughnut(a; lambda=0.5, prograde=true)
        xcp = eqpoint(a, dp.r_center)
        pcp = source_plasma_at(dp, st, xcp)
        @test xcp[2] * pcp.ucon[3] - xcp[3] * pcp.ucon[2] > 0
    end

    # -- Page–Thorne retrograde flux uses the a -> -a 15n closed form --------
    @testset "PageThorneDisk retrograde flux" begin
        a = 0.9
        disk = PageThorneDisk(a; prograde=false, r_out=60.0)
        @test isapprox(disk.r_isco, 8.71735; atol=1e-3)      # BPT retro ISCO at a=0.9
        for xr in (disk.x1, disk.x2, disk.x3)                # solve x^3-3x+2(-a)=0
            @test isapprox(xr^3 - 3xr - 2a, 0.0; atol=1e-10)
        end
        # independent P&T 15n closed form at the effective spin -a (mdot=1)
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
            @test isapprox(page_thorne_flux(disk, r), pt_ref(-a, r, disk.r_isco); rtol=1e-8)
        end
        @test page_thorne_flux(disk, disk.r_isco) == 0.0
        @test 0 < page_thorne_flux(disk, 800.0) < page_thorne_flux(disk, disk.r_isco + 1.0)
        # signed-spin input a<0 normalizes to (|a|, flipped sense) — no DomainError
        disk2 = PageThorneDisk(-a; prograde=true, r_out=60.0)
        @test disk2.a == a && disk2.prograde == false && disk2.r_isco == disk.r_isco
        for r in (disk.r_isco + 0.5, 12.0, 20.0)
            @test page_thorne_flux(disk2, r) == page_thorne_flux(disk, r)
        end
    end

    # -- on-axis toroidal direction stays finite (was a sqrt DomainError) ----
    @testset "hotspot ambient on-axis bcon finite" begin
        for a in (0.0, 0.5, 0.9375), z in (4.0, 8.0, 20.0)
            st = KerrSchildCartesian(a)
            spot = EquatorialHotspot(a; ambient_n_e=1e3, ambient_theta_e=5.0, ambient_B=1.0)
            pl = source_plasma_at(spot, st, SVector(0.0, 0.0, 0.0, z))   # on the spin axis
            @test pl.inside
            @test all(isfinite, pl.bcon)
            @test isapprox(dot(pl.bcon, pl.gcov * pl.bcon), 1.0; atol=1e-10)  # unit spacelike
            @test isapprox(dot(pl.bcon, pl.ucov), 0.0; atol=1e-10)            # b . u = 0
        end
    end

    # -- PolishDoughnut rejects l0 below the marginally-stable value ---------
    @testset "PolishDoughnut closed-torus window guard" begin
        a = 0.5
        l_ms = keplerian_specific_angular_momentum(a, isco_radius(a))
        @test_throws ArgumentError PolishDoughnut(a; l0=0.8 * l_ms)  # sub-l_ms: no torus
        d = PolishDoughnut(a; lambda=0.5)                            # valid interior l0
        @test isfinite(d.r_out) && d.r_cusp < d.r_center < d.r_out
        @test PolishDoughnut(a; r_center=25.0) isa PolishDoughnut    # overflow side still OK
    end
end

# Regression for the eltype-generic vacuum sentinel (_vacuum): under a Dual-typed
# position the vacuum branch of source_plasma_at must return the SAME concrete
# NamedTuple type as the plasma branch. With the old Float64-hardcoded sentinel the
# two branches disagreed (n_e/ucon Float64 vs Dual) and inference of the whole call
# collapsed to a boxing Union — defeating the allocation-free/isbits seam the module
# docstring advertises ("ForwardDiff flows through source_plasma_at").
@testset "vacuum sentinel: AD type-stable (eltype-generic)" begin
    st = KerrSchildCartesian(0.0)
    src = PolishDoughnut(0.0; r_center=9.0, n_e_center=1e7, theta_e_center=40.0,
                         polytrope_n=1.5, beta=8.0)
    dual_x(x) = SVector(x[1],
                        ForwardDiff.Dual(x[2], 1.0, 0.0, 0.0),
                        ForwardDiff.Dual(x[3], 0.0, 1.0, 0.0),
                        ForwardDiff.Dual(x[4], 0.0, 0.0, 1.0))
    x_vac = dual_x(SVector(0.0, 2.0, 0.0, 0.0))          # inner funnel r<r_cusp -> vacuum
    x_in  = dual_x(SVector(0.0, src.r_center, 0.0, 0.0)) # pressure-max centre -> plasma
    rv = source_plasma_at(src, st, x_vac)
    ri = source_plasma_at(src, st, x_in)
    @test rv.inside == false && ri.inside == true
    @test typeof(rv) == typeof(ri)                        # ONE concrete NamedTuple type
    @test rv.n_e isa ForwardDiff.Dual                     # vacuum n_e now follows the Dual eltype
    @test eltype(rv.ucon) === eltype(rv.gcov)             # zeroed 4-vectors share the working eltype
    rt = only(Base.return_types(source_plasma_at, (typeof(src), typeof(st), typeof(x_vac))))
    @test isconcretetype(rt)                              # was a boxing Union before the fix
    # Float64 inputs unchanged: vacuum stays exactly the zeroed Float64 tuple.
    r64 = source_plasma_at(src, st, SVector(0.0, 2.0, 0.0, 0.0))
    @test r64.inside == false
    @test r64.n_e == 0.0 && r64.theta_e == 0.0 && r64.B == 0.0
    @test r64.ucon == SVector(0.0, 0.0, 0.0, 0.0) && eltype(r64.ucon) == Float64
end
