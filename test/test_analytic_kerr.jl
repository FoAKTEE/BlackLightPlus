# Exact-Kerr analytic tracer tests (AART/Krang-style closed forms). Checks:
#   (a) ORACLE INVERSION : analytic rs == numerical-ODE equatorial-crossing KS
#       radius to integrator tolerance (the analytic result is the ORACLE).
#   (b) SHADOW           : critical_curve(a→0) == 3√3 M to 1e-10; a=0.94 D-shaped
#       and consistent with AART's dpara_BH_shadow reference file
#       (path overridable via ENV["BLP_REF_AART"], isfile-guarded).
#   (c) BANDS            : n≥1 bands straddle the critical curve, nested inside
#       n=0, band (shell) areas decrease with n.
#   (d) AD               : ForwardDiff d rs/d{a,θo} == central finite differences.

using Test
using BlackLightPlus
using BlackLightPlus.Geodesics
using StaticArrays
using ForwardDiff

# ── coordinate map (documented) ──────────────────────────────────────────────
# The plane-parallel camera's image coordinates are Bardeen's: α_Bardeen = α_image,
# β_Bardeen = β_image. The camera ray at image-α is the received photon traced
# backward, with L_z/E = −α_image·sinθo (`angular_momentum_z`/`energy`), which is
# AART/Krang's λ = −α_Bardeen·sinθo.
cam_to_bardeen(αimg) = αimg

# All equatorial (Z=0) crossing KS radii of the numerically-traced camera ray.
# ks_radius r ≡ Boyer–Lindquist r, so the equatorial-crossing KS radius is the
# invariant to compare against the analytic BL source radius rs.
function num_equatorial_crossings(st, y0; d, reltol=1e-11, abstol=1e-13)
    sol = trace_geodesic(st, y0; λmax=3.0 * d, reltol, abstol,
                         escape=1.02 * d, horizon_eps=1e-5, save_everystep=true)
    cr = Float64[]
    for k in 1:length(sol.t)-1
        z0 = sol.u[k][4]; z1 = sol.u[k+1][4]
        (z0 == 0 || sign(z0) == sign(z1)) && continue
        lo, hi = sol.t[k], sol.t[k+1]
        for _ in 1:80
            mid = (lo + hi) / 2
            (sign(sol(mid)[4]) == sign(z0)) ? (lo = mid) : (hi = mid)
        end
        uc = sol((lo + hi) / 2)
        push!(cr, ks_radius(st, SVector(uc[1], uc[2], uc[3], uc[4])))
    end
    return cr
end

@testset "analytic Kerr (AART/Krang)" begin

    # ── engine sanity ────────────────────────────────────────────────────────
    @testset "engine primitives" begin
        # conserved quantities match the Bardeen definitions
        λ, η = conserved_quantities(2.0, 3.0, deg2rad(50.0), 0.9)
        @test λ ≈ -2.0 * sin(deg2rad(50.0))
        @test η ≈ (4.0 - 0.81) * cos(deg2rad(50.0))^2 + 9.0
        # radial roots satisfy the quartic R(r)=0
        for (a, α, β) in [(0.9, -4.0, 2.0), (0.5, 3.0, -1.0), (0.94, -6.0, 0.0)]
            λ, η = conserved_quantities(α, β, deg2rad(45.0), a)
            r1, r2, r3, r4 = radial_roots(λ, η, a)
            Rr(r) = (r^2 + a^2 - a*λ)^2 - (r^2 - 2r + a^2)*(η + (λ - a)^2)
            for r in (r1, r2, r3, r4)
                @test abs(Rr(r)) < 1e-8 * max(1.0, abs(r)^4)
            end
        end
        # Type II (outside crit curve → 4 real roots) vs Type III (inside → 2 real)
        λ2, η2 = conserved_quantities(cam_to_bardeen(-6.0), 2.0, deg2rad(60.0), 0.94)
        r2s = radial_roots(λ2, η2, 0.94)
        @test count(z -> abs(imag(z)) < 1e-8, r2s) == 4          # Type II
        λ3, η3 = conserved_quantities(cam_to_bardeen(-1.0), 0.3, deg2rad(60.0), 0.94)
        r3s = radial_roots(λ3, η3, 0.94)
        @test count(z -> abs(imag(z)) < 1e-8, r3s) == 2          # Type III
    end

    # ── CHECK (a): ORACLE INVERSION ───────────────────────────────────────────
    @testset "gate a: analytic matches numerical ODE tracer" begin
        d = 1.0e6                        # large distance ⇒ plane-parallel ≈ point-observer
        pixels = [(5.0, 2.0), (-5.0, 2.0), (4.0, -3.0), (2.0, 4.0)]  # β ≠ 0 (β=0 is degenerate)
        maxrel = 0.0
        for a in (0.001, 0.5, 0.94), inc in (17.0, 60.0)
            st = KerrSchildCartesian(a)
            θo = deg2rad(inc)
            cam = PlaneParallelCamera(inc, d, 40.0, 64)
            for (αimg, β) in pixels
                y0 = camera_rays_at(st, cam, [(αimg, β)])[1]
                r_cam = ks_radius(st, SVector(y0[1], y0[2], y0[3], y0[4]))
                cr = num_equatorial_crossings(st, y0; d)
                isempty(cr) && continue
                rs, φs, ts, νr, valid = analytic_emission(cam_to_bardeen(αimg), β, θo, a, 0; ro=r_cam)
                @test valid
                @test νr in (-1, 1)
                @test isfinite(φs) && isfinite(ts)
                @test rs > horizon_radius(st)
                rel = abs(rs - cr[1]) / cr[1]     # compare to FIRST (direct, n=0) crossing
                maxrel = max(maxrel, rel)
                @test rel < 5e-5                  # integrator-tolerance (residual ∝ 1/d)
            end
        end
        @info "gate a: max rel err (analytic rs vs ODE crossing)" maxrel
    end

    # ── CHECK (b): SHADOW ─────────────────────────────────────────────────────
    @testset "gate b: Kerr shadow / critical curve" begin
        # Schwarzschild: exact circle of radius 3√3 M to machine precision
        R0 = 3 * sqrt(3.0)
        for inc in (17.0, 60.0, 89.0)
            αc, βc = critical_curve(0.0, deg2rad(inc); n=200)
            rr = sqrt.(αc .^ 2 .+ βc .^ 2)
            @test maximum(abs.(rr .- R0)) < 1e-10
        end
        # a=0.94: D-shaped (asymmetric in α), diameter near the Kerr value
        αc, βc = critical_curve(0.94, deg2rad(60.0); n=400)
        wmax = maximum(αc); wmin = minimum(αc)
        @test wmax + wmin > 0.3                       # shifted right → D-shape asymmetry
        dperp = maximum(βc) - minimum(βc)             # extent ⟂ to spin projection
        dpara = wmax - wmin                           # extent ∥
        @test 8.0 < dpara < 11.0
        @test 9.0 < dperp < 11.0
        # AART reference: dpara_BH_shadow.txt[1,1] is the Schwarzschild diameter 2·3√3
        ref = get(ENV, "BLP_REF_AART",
                  "/data/haiyangw/claude/Mahakala/ref-code/aart/dpara_BH_shadow.txt")
        if isfile(ref)
            firstval = parse(Float64, split(readline(ref))[1])
            @test isapprox(firstval, 2 * R0; rtol=1e-3)   # ≈ 10.392 (a≈0 diameter)
        end
    end

    # ── CHECK (c): LENSING BANDS ──────────────────────────────────────────────
    @testset "gate c: lensing bands nest & thin with n" begin
        for a in (0.5, 0.94), inc in (17.0, 60.0)
            θo = deg2rad(inc)
            αc, βc = critical_curve(a, θo; n=100)
            Ac = polygon_area(αc, βc)
            i0, o0 = lensing_band(a, θo, 0; npts=80)
            i1, o1 = lensing_band(a, θo, 1; npts=80)
            i2, o2 = lensing_band(a, θo, 2; npts=80)
            A0i = polygon_area(i0)                       # apparent horizon
            A1i = polygon_area(i1); A1o = polygon_area(o1)
            A2i = polygon_area(i2); A2o = polygon_area(o2)
            # n≥1 bands straddle the critical curve
            @test A1i < Ac < A1o
            @test A2i < Ac < A2o
            # nested inside n=0 (apparent horizon ⊂ band inner edges; outer ⊂ box)
            @test A0i < A1i
            @test A1o < polygon_area(o0)                 # inside the image box
            # band (shell) areas decrease with n, and stay positive
            shell1 = A1o - A1i; shell2 = A2o - A2i
            @test shell1 > shell2 > 0
            # apparent horizon is strictly inside the critical curve
            @test A0i < Ac
        end
        # band_grid: mask marks points in the annulus, and grid is Cartesian
        i1, o1 = lensing_band(0.94, deg2rad(60.0), 1; npts=80)
        grid, mask, N = band_grid(i1, o1; dx=0.1)
        @test size(grid, 1) == N * N
        @test count(mask) > 0
        @test count(mask) < N * N                        # not everything is in the band
    end

    # ── CHECK (d): AUTODIFF ───────────────────────────────────────────────────
    @testset "gate d: ForwardDiff == finite differences" begin
        maxrel = 0.0
        for (αimg, β, inc, a0) in [(-5.0, 2.0, 60.0, 0.94), (2.0, 4.0, 17.0, 0.94),
                                   (-6.0, -1.0, 60.0, 0.5), (5.0, 2.0, 60.0, 0.3)]
            αB = cam_to_bardeen(αimg); θ0 = deg2rad(inc)
            rs_a(a) = analytic_emission(αB, β, θ0, a, 0; ro=1.0e5)[1]
            rs_θ(θ) = analytic_emission(αB, β, θ, a0, 0; ro=1.0e5)[1]
            dA = ForwardDiff.derivative(rs_a, a0)
            dθ = ForwardDiff.derivative(rs_θ, θ0)
            h = 1e-6
            fdA = (rs_a(a0 + h) - rs_a(a0 - h)) / (2h)
            fdθ = (rs_θ(θ0 + h) - rs_θ(θ0 - h)) / (2h)
            @test isfinite(dA) && isfinite(dθ)
            @test isapprox(dA, fdA; rtol=1e-6)
            @test isapprox(dθ, fdθ; rtol=1e-6)
            maxrel = max(maxrel, abs(dA - fdA) / abs(fdA), abs(dθ - fdθ) / abs(fdθ))
        end
        @info "gate d: max rel err (AD vs FD)" maxrel
    end

    # ── render seam integration ──────────────────────────────────────────────
    @testset "render_analytic seam" begin
        cfg = RunConfig(bh_a=0.94, tracer=:analytic,
                        camera=CameraConfig(inclination_deg=60.0, distance=1000.0,
                                            fov=20.0, resolution=48))
        out = render_analytic(cfg)
        @test size(out.I) == (48, 48)
        @test length(out.rs) == 3                      # bands 0,1,2
        @test any(isfinite, out.rs[1])                 # some pixels land on the disk
        @test all(x -> x >= 0, out.I)                  # r⁻² placeholder ≥ 0
        @test sum(out.I) > 0
        # redshift-sign map ∈ {NaN, -1, +1}
        s = filter(isfinite, vec(out.redshift_sign[1]))
        @test all(v -> v == 1 || v == -1, isempty(s) ? Float64[] : s)
        # dispatch through render(): :analytic returns unpolarized image
        img = render(cfg, "unused_snapshot_path")
        @test img.I == out.I
        @test all(iszero, img.Q)
        # placeholder equatorial_intensity seam (the r⁻² placeholder body)
        @test equatorial_intensity(6.0, 1, 0.94) ≈ 1 / 36
        @test equatorial_intensity(1.0, 1, 0.94) == 0.0   # inside horizon ⇒ dark
    end
end

# ── regressions: Mino budget, domain symmetries, β = 0 ───────────────────────
@testset "analytic Kerr domain & aliasing regressions" begin
    # (1) Mino-budget validity: this ray crosses the equator ONCE (ODE oracle),
    # then plunges — the n≥1 sn/cn-aliased ghost landings must be invalid.
    @testset "Mino budget rejects aliased ghost landings" begin
        θo = deg2rad(17.0)
        rs0, _, _, _, v0 = analytic_emission(-3.5, -0.1, θo, 0.94, 0; ro=1000.0)
        @test v0 && isapprox(rs0, 2.6226742150344196; rtol=1e-10)   # direct hit kept
        for n in 1:3
            @test !analytic_emission(-3.5, -0.1, θo, 0.94, n; ro=1000.0)[5]
        end
        # full-grid census (fov 24, 40²): pre-fix n=2 marked 1425 pixels 'valid'
        # across the shadow interior; the true n=2 band holds only a handful.
        edges = range(-12.0, 12.0; length=41)
        cent = (edges[1:end-1] .+ edges[2:end]) ./ 2
        nvalid(n) = count(analytic_emission(-x, y, θo, 0.94, n; ro=1000.0)[5]
                          for x in cent, y in cent)
        @test nvalid(0) > 1000                 # direct image survives the bound
        @test nvalid(2) < 100                  # ghosts gone (was 1425)
    end

    # (2) observer below the equator: exact mirror (θo → π−θo, β → −β)
    @testset "θo > π/2 equatorial mirror" begin
        for (αB, β) in ((-5.0, 2.0), (5.0, 2.0), (4.0, -3.0), (2.0, 4.0))
            up = analytic_emission(αB, -β, deg2rad(60.0), 0.94, 0; ro=1000.0)
            lo = analytic_emission(αB, β, deg2rad(120.0), 0.94, 0; ro=1000.0)
            @test lo[5] == up[5]
            if up[5]
                @test isapprox(lo[1], up[1]; rtol=1e-9)    # rs
                @test isapprox(lo[2], up[2]; rtol=1e-9)    # φs (mirror keeps φ)
                @test isapprox(lo[3], up[3]; rtol=1e-9)    # ts
                @test lo[4] == up[4]                       # νr
            end
        end
        o = analytic_emission(-5.0, 2.0, deg2rad(163.0), 0.94, 0; ro=1000.0)  # M87*-like
        @test o[5] && o[1] > kerr_horizon(0.94)
        # ODE-oracle anchor at θo = 120°: analytic n=0 == FIRST equatorial
        # crossing (pre-fix it returned the SECOND crossing or valid=false).
        st = KerrSchildCartesian(0.94); d = 1.0e6
        cam = PlaneParallelCamera(120.0, d, 40.0, 64)
        for (αimg, β) in ((5.0, 2.0), (-5.0, 2.0))
            y0 = camera_rays_at(st, cam, [(αimg, β)])[1]
            r_cam = ks_radius(st, SVector(y0[1], y0[2], y0[3], y0[4]))
            cr = num_equatorial_crossings(st, y0; d)
            @test !isempty(cr)
            isempty(cr) && continue
            rs = analytic_emission(cam_to_bardeen(αimg), β, deg2rad(120.0), 0.94, 0; ro=r_cam)
            @test rs[5]
            @test abs(rs[1] - cr[1]) / cr[1] < 5e-5
        end
    end

    # (3) a = 0 finite (tiny-spin clamp) and a < 0 via the azimuthal mirror
    @testset "a = 0 and retrograde a < 0" begin
        o0 = analytic_emission(-5.0, 2.0, deg2rad(60.0), 0.0, 0; ro=1000.0)
        oe = analytic_emission(-5.0, 2.0, deg2rad(60.0), 1e-6, 0; ro=1000.0)
        @test o0[5]
        @test all(isfinite, (o0[1], o0[2], o0[3]))
        @test isapprox(o0[1], oe[1]; rtol=1e-8)            # continuous a → 0 limit
        # 3.845428284554: the 3.8453296… value baked before the stable
        # angular_turning_points was polluted by u_p's ε/a² cancellation at the
        # a=1e-6 clamp (2.6e-5 off); the ODE oracle at exact a=0, camera 1e6,
        # sits within 7.9e-7 of the STABLE form's landing.
        @test isapprox(o0[1], 3.845428284554; rtol=1e-6)
        # a < 0 ≡ azimuthal mirror of |a|: α → −α, Δφ → −Δφ about φo = −π/2
        pro = analytic_emission(5.0, 2.0, deg2rad(60.0), 0.5, 0; ro=1000.0)
        ret = analytic_emission(-5.0, 2.0, deg2rad(60.0), -0.5, 0; ro=1000.0)
        @test pro[5] && ret[5]
        @test isapprox(ret[1], pro[1]; rtol=1e-12)         # rs  (was -12.28, invalid)
        @test isapprox(ret[3], pro[3]; rtol=1e-12)         # ts
        @test isapprox(ret[2], -π - pro[2]; atol=1e-10)    # φs
        @test ret[4] == pro[4]
    end

    # (4) β = 0 must equal the two-sided β → 0 limit (H(0)=1 docstring convention)
    @testset "β = 0 grazes the far side (H(0)=1)" begin
        lim = analytic_emission(-5.0, 1e-9, deg2rad(60.0), 0.94, 0; ro=1000.0)
        for b in (0.0, -0.0)
            o = analytic_emission(-5.0, b, deg2rad(60.0), 0.94, 0; ro=1000.0)
            @test o[5]
            @test isapprox(o[1], lim[1]; rtol=1e-6)        # rs was ~20% off pre-fix
            @test isapprox(o[2], lim[2]; rtol=1e-5)        # φs
        end
    end

    # (5) n=1 outer band edge at high inclination: the bisection bracket must
    # grow past s=4 instead of silently collapsing onto the critical curve.
    @testset "n=1 outer band edge at 85° inclination" begin
        a = 0.94; θo = deg2rad(85.0)
        αc, βc = critical_curve(a, θo; n=60)
        Ac = polygon_area(αc, βc)
        inner, outer = lensing_band(a, θo, 1; npts=60)
        Ai = polygon_area(inner); Ao = polygon_area(outer)
        @test Ai < Ac < Ao
        @test Ao > 10 * Ac      # pre-fix: 92/200 edge points collapsed → Ao ≈ 1.7·Ac
    end
end

# ── regressions: stable u_±, finite-observer νr threshold ────────────────────
@testset "angular_turning_points stable at small spin (BigFloat oracle)" begin
    # u_p = Δθ + √(Δθ²+η/a²) cancels catastrophically for a → 0, where
    # Δθ ≈ −(η+λ²)/(2a²) dwarfs the root: rel err was 3.5e-7 at a=1e-4 and
    # 1.8e-5 at a=1e-5, 1.9e-4 at the a=1e-6 clamp every |a|<1e-6 render uses
    # (AART raytracing_f.py shares the flaw). Stable form: same-sign root by
    # addition + the exact product u_p·u_m = −η/a².
    θo = deg2rad(60.0)
    for a in (1e-2, 1e-4, 1e-5, 1e-6)
        λ, η = conserved_quantities(-5.0, 2.0, θo, a)
        u_p, u_m = angular_turning_points(λ, η, a)
        Δb = (1 - (big(η) + big(λ)^2) / big(a)^2) / 2
        db = sqrt(Δb^2 + big(η) / big(a)^2)
        @test abs(u_p - Float64(Δb + db)) <= 1e-13 * Float64(Δb + db)
        @test abs(u_m - Float64(Δb - db)) <= 1e-13 * abs(Float64(Δb - db))
    end
    # Δθ > 0 branch: u_m is then the cancelling root — still stable
    λ2, η2 = conserved_quantities(0.3, 0.02, deg2rad(80.0), 0.94)
    u_p2, u_m2 = angular_turning_points(λ2, η2, 0.94)
    Δb = (1 - (big(η2) + big(λ2)^2) / big(0.94)^2) / 2
    db = sqrt(Δb^2 + big(η2) / big(0.94)^2)
    @test abs(u_p2 - Float64(Δb + db)) <= 1e-14 * Float64(Δb + db)
    @test abs(u_m2 - Float64(Δb - db)) <= 1e-13 * abs(Float64(Δb - db))
end

@testset "band_grid row-bucketed mask == brute-force winding (bit-identical)" begin
    # band_grid now buckets each polygon's row-crossing edges once per grid row
    # (O(N·(P+kN)) instead of O(N²·P), measured 59x at photon_ring_layers
    # defaults); the mask must stay BIT-IDENTICAL to the brute-force
    # `_in_polygon` winding loop it replaced.
    G = BlackLightPlus.Geodesics
    a = 0.94; θo = deg2rad(17.0)
    for n in (0, 1)
        inner, outer = lensing_band(a, θo, n; npts=60)
        grid, mask, _ = band_grid(inner, outer; dx=0.1)
        ref = [G._in_polygon(grid[i, 1], grid[i, 2], view(outer, :, 1), view(outer, :, 2)) &&
               !(size(inner, 1) >= 3 &&
                 G._in_polygon(grid[i, 1], grid[i, 2], view(inner, :, 1), view(inner, :, 2)))
               for i in 1:size(grid, 1)]
        @test mask == ref
        @test count(mask) > 0
    end
    # non-convex loop and a degenerate (< 3 vertex) inner keep the same semantics
    ts = range(0, 2π; length=41)[1:40]
    star = hcat(cos.(ts) .* (2 .+ 1.5 .* cos.(5 .* ts)),
                sin.(ts) .* (2 .+ 1.5 .* cos.(5 .* ts)))
    grid, mask, _ = band_grid(zeros(0, 2), star; dx=0.05)
    ref = [G._in_polygon(grid[i, 1], grid[i, 2], view(star, :, 1), view(star, :, 2))
           for i in 1:size(grid, 1)]
    @test mask == ref
end

@testset "lensing bands honor θo > π/2 (equatorial mirror)" begin
    # analytic_emission normalises θo > π/2 through (θo → π−θo, β → −β); the
    # host-side band setup previously assumed an upper-hemisphere observer.
    # Membership must mirror: (α,β) ∈ band(θo) ⇔ (α,−β) ∈ band(π−θo).
    a = 0.94; θo = deg2rad(120.0); θm = π - θo
    # critical curve is β-symmetric — already mirror-invariant (up to the 1-ulp
    # difference between cos²θo and cos²(π−θo))
    αc1, βc1 = critical_curve(a, θo; n=64)
    αc2, βc2 = critical_curve(a, θm; n=64)
    @test maximum(abs.(αc1 - αc2)) < 1e-10 && maximum(abs.(βc1 - βc2)) < 1e-10
    # apparent horizon (NOT β-symmetric: its m=1/m=0 legs differ) mirrors exactly
    αh, βh = apparent_horizon(a, θo; n=40)
    αm, βm = apparent_horizon(a, θm; n=40)
    @test αh == αm && βh == -βm
    @test polygon_area(αh, βh) > 0
    for n in (0, 1)
        i1, o1 = lensing_band(a, θo, n; npts=40)
        i2, o2 = lensing_band(a, θm, n; npts=40)
        @test i1[:, 1] == i2[:, 1] && i1[:, 2] == -i2[:, 2]
        @test o1[:, 1] == o2[:, 1] && o1[:, 2] == -o2[:, 2]
        @test polygon_area(i1) > 0
    end
    # and the n=1 band still straddles the (mirror-invariant) critical curve
    i1, o1 = lensing_band(a, θo, 1; npts=40)
    Ac = polygon_area(αc1, βc1)
    @test polygon_area(i1) < Ac < polygon_area(o1)
end

@testset "νr flips at the FINITE-observer turning time (not the r→∞ one)" begin
    # Pre-fix the case-II νr threshold was _tau_max_case2's r→∞ amplitude, which
    # overshoots the finite-observer turning time by ∫_ro^∞ dr/√R ≈ 1/ro, so rays
    # in that Mino window kept νr=+1 though the source lies PAST the radial turn
    # (AART's Jmax is bug-compatible). The threshold is now the I0 antiderivative
    # at ro; exactly at the flip the inverted source radius must equal r4.
    G = BlackLightPlus.Geodesics
    a = 0.94; θo = deg2rad(17.0); ro = 1000.0; n = 1; α = 5.10
    gap(β) = begin
        p = G._emission_prep(α, β, θo, a; ro=ro)
        Gθ, _, _ = G._angular_from_parts(n, p.ang)
        Gθ - p.τturn
    end
    βa, βb = -2.005, -1.995
    @test gap(βa) > 0 && gap(βb) < 0
    for _ in 1:80
        mid = (βa + βb) / 2
        (gap(mid) > 0) == (gap(βa) > 0) ? (βa = mid) : (βb = mid)
    end
    βs = (βa + βb) / 2
    p = G._emission_prep(α, βs, θo, a; ro=ro)
    rs_flip = G._emission_band(p, n)[1]
    @test isapprox(rs_flip, real(p.roots[4]); rtol=1e-9)   # flip exactly at rs = r4
    # a pixel INSIDE the (τ(ro→r4), τ(∞→r4)) window (margins ≈ 5e-4 each side):
    # the source has turned, so νr must be −1 — the pre-fix threshold said +1.
    pw = G._emission_prep(α, -2.000262457545051, θo, a; ro=ro)
    Gθw, _, _ = G._angular_from_parts(n, pw.ang)
    rsw, _, _, νw, vw = G._emission_band(pw, n)
    @test vw
    @test pw.τturn < Gθw <= G._tau_max_case2(pw.roots..., Float64)  # in the window
    @test νw == -1
    @test rsw > real(pw.roots[4])                          # past the turn
end

@testset "analytic elliptic near-critical conditioning: two-sided continuity + edge validity" begin
    # Pin the Float64 conditioning plateau of the case II/III split near the critical curve
    # (root splitting ~√δ falls below fp resolution; see the NOTE at is_case2).
    # Outside the plateau rs is oracle-linear in δ;
    # the plateau saturates at ~5-6e-4 relative — 5 orders below a 0.01-0.05 M pixel; every
    # probed argument edge stays finite/valid. Guards a future _cuberoot/tolerance tweak from
    # silently WIDENING the plateau from ~1e-7 to ~1e-4 (which WOULD matter at 0.01 M pixels).
    probe(αc, βc, i, δ) = begin
        n = length(αc); ip = i == n ? 1 : i + 1; im = i == 1 ? n : i - 1
        tα = αc[ip] - αc[im]; tβ = βc[ip] - βc[im]; s = hypot(tα, tβ)
        (αc[i] + δ * tβ / s, βc[i] - δ * tα / s)         # step ±δ along the curve normal (tβ,-tα)/|t|
    end
    a = 0.94; θo = deg2rad(60.0)
    αc, βc = critical_curve(a, θo; n = 400)
    i = 100
    Δrs(δ) = begin
        α1, β1 = probe(αc, βc, i, +δ); α2, β2 = probe(αc, βc, i, -δ)
        abs(analytic_emission(α1, β1, θo, a, 0; ro = 1000.0)[1] -
            analytic_emission(α2, β2, θo, a, 0; ro = 1000.0)[1])
    end
    # OUTSIDE the plateau: two-sided rs is continuous and shrinks ~linearly with δ (measured
    # 2.2e-3 / 2.2e-5 at δ = 1e-3 / 1e-5)
    @test Δrs(1e-3) < 5e-3
    @test Δrs(1e-5) < 5e-5
    # INSIDE the plateau (|δ| ≲ 1e-7): rs conditioning saturates at ~5.4e-4 — bounded ≪ a pixel,
    # and genuinely plateaued (no longer the linear ~δ of the healthy branch)
    @test Δrs(1e-9) < 1e-3
    @test Δrs(1e-9) > 1e-5
    # argument EDGES (θo->90, β->0, a->1) all stay finite + valid
    for (ae, ie, αe, βe) in ((0.94, 89.999, -5.0, 2.0), (0.94, 89.999999, -5.0, 2.0),
                             (0.998, 60.0, -5.0, 1e-9), (0.9999, 60.0, -5.0, 2.0),
                             (0.999999, 60.0, -5.0, 2.0), (0.94, 60.0, -5.0, 1e-12),
                             (1.0 - 1e-12, 45.0, -4.0, 1.5))
        rs, φs, ts, νr, valid = analytic_emission(αe, βe, deg2rad(ie), ae, 0; ro = 1000.0)
        @test valid
        @test isfinite(rs) && isfinite(φs) && isfinite(ts)
    end
end
