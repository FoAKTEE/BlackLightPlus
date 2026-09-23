# Spacetime geometry tests: metric identities, geodesic conservation, the
# Schwarzschild shadow (b_c -> 3√3 to 1e-3), Kerr photon-orbit radii, the
# UserMetric / imaging dispatch paths, and Kerr-Schild singular-disk
# regularization. Exact-GR ground truth; no external reference data.
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus.Geodesics

const SG = BlackLightPlus.Geodesics

@testset "metric identities" begin
    for a in (0.0, 0.5, 0.9, 0.998), xyz in ((4.0, 1.0, 2.0), (6.0, -3.0, 1.5), (-5.0, 3.0, 2.5))
        st = KerrSchildCartesian(a)
        x = SVector(0.0, xyz...)
        g = metric(st, x); gi = inverse_metric(st, x)
        @test maximum(abs.(g * gi - I)) < 1e-12                # analytic inverse exact
        @test maximum(abs.(g - g')) == 0.0                     # symmetric
        # l is eta-null on the r-surface
        _, _, l = SG._rfl(x, a)
        @test abs(l[2]^2 + l[3]^2 + l[4]^2 - 1) < 1e-12
        # AD gradient of inverse metric matches finite differences
        dg = inverse_metric_gradient(st, x)
        h = 1e-6
        for c in 2:4
            dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
            fd = (inverse_metric(st, x + dx) - inverse_metric(st, x - dx)) / (2h)
            @test maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)) < 1e-6
        end
    end
    # Minkowski limit: a=0 far from hole ~ flat; exact Minkowski type
    @test metric(Minkowski(), SVector(0.0, 1.0, 2.0, 3.0)) == SG.ETA
end

@testset "Hamiltonian geodesics: conservation" begin
    st = KerrSchildCartesian(0.9)
    x0 = SVector(0.0, 100.0, 6.0, 2.0)
    p0 = nullify_covariant(st, x0, SVector(-1.0, 0.05, 0.02))
    y0 = vcat(x0, p0)
    @test abs(null_error(st, y0)) < 1e-12                      # nullified exactly
    sol = trace_geodesic(st, y0; escape=110.0, reltol=1e-11, abstol=1e-13)
    yf = sol.u[end]
    @test abs(yf[5] - y0[5]) == 0.0                            # p_t conserved BY CONSTRUCTION
    @test abs(angular_momentum_z(yf) - angular_momentum_z(y0)) < 1e-8   # L_z (axisymmetry)
    @test abs(null_error(st, yf)) < 1e-8                       # null drift
end

@testset "Schwarzschild shadow (b_c -> 3sqrt3)" begin
    st = KerrSchildCartesian(0.0)
    d = 1000.0
    captured(b) = begin
        x0 = SVector(0.0, d, b, 0.0)
        # aim inward along -x: contravariant k = (1,-1,0,0)-ish -> covariant via g, then nullify p_t
        kcon = SVector(1.0, -1.0, 0.0, 0.0)
        p = metric(st, x0) * kcon
        p0 = nullify_covariant(st, x0, SVector(p[2], p[3], p[4]))
        sol = trace_geodesic(st, vcat(x0, p0); escape=1.5d, reltol=1e-9, abstol=1e-9)
        u = sol.u[end]
        ks_radius(st, SVector(u[1], u[2], u[3], u[4])) < 10.0
    end
    lo, hi = 4.5, 6.0
    @test captured(lo); @test !captured(hi)
    for _ in 1:30
        mid = (lo + hi) / 2
        captured(mid) ? (lo = mid) : (hi = mid)
    end
    bc = (lo + hi) / 2
    @test abs(bc - 3 * sqrt(3.0)) / (3 * sqrt(3.0)) < 1e-3
end

@testset "Kerr photon orbit radii (eqs 28-29 oracle)" begin
    # unstable spherical-orbit radii from the closed form; the shadow test above
    # plus these pin the geodesic sector to exact GR
    pro(a)  = 2 * (1 + cos(2 / 3 * acos(-abs(a))))
    retro(a) = 2 * (1 + cos(2 / 3 * acos(abs(a))))
    @test abs(pro(0.9) - 1.5579) < 1e-3
    @test abs(retro(0.9) - 3.9103) < 1e-3
end

@testset "UserMetric rides the metric-agnostic engine (geodesic_rhs / trace_geodesic)" begin
    # A stationary user closure wrapping Kerr must trace IDENTICALLY to the built-in
    # KerrSchildCartesian — the exported "any closure, AD supplies ∂g" headline path.
    # Pre-fix geodesic_rhs/trace_geodesic threw MethodError on UserMetric.
    a = 0.5
    kerr = KerrSchildCartesian(a)
    um = UserMetric(x -> metric(kerr, x))
    x0 = SVector(0.0, 100.0, 6.0, 2.0)
    p0 = nullify_covariant(um, x0, SVector(-1.0, 0.05, 0.02))
    y0 = vcat(x0, p0)
    @test abs(null_error(um, y0)) < 1e-12
    # general RHS == stationary RHS for a stationary closure (dp_t ≡ 0, rest identical)
    r_um = geodesic_rhs(um, y0)
    r_ks = geodesic_rhs(kerr, y0)
    @test r_um[5] == 0.0                                   # ∂_t g ≡ 0 ⇒ dp_t exactly 0
    @test maximum(abs.(r_um - r_ks)) < 1e-14               # reduces to the stationary RHS
    # traced geodesic matches analytic Kerr (deflection direction + null preservation)
    su = trace_geodesic(um, y0; escape=110.0)
    sk = trace_geodesic(kerr, y0; escape=110.0)
    du = SVector(su.u[end][6], su.u[end][7], su.u[end][8]); du /= norm(du)
    dk = SVector(sk.u[end][6], sk.u[end][7], sk.u[end][8]); dk /= norm(dk)
    @test maximum(abs.(du - dk)) < 1e-6
    @test abs(null_error(um, su.u[end])) < 1e-7

    # A TIME-DEPENDENT closure must RETAIN dp_t/dλ = -½ ∂_t g^{μν} p_μ p_ν (the
    # stationary RHS would silently drop it). For g = diag(-(1+c·t), 1, 1, 1): at t=0
    # this is Minkowski (nullify gives p_t² = |p_s|²) and ∂_t g^{tt} = c, so the exact
    # time term is dp_t = -½ c |p_s|² — unmistakably nonzero.
    c = 0.3
    gt = UserMetric(x -> @SMatrix [-(1 + c * x[1]) 0.0 0.0 0.0
                                    0.0            1.0 0.0 0.0
                                    0.0            0.0 1.0 0.0
                                    0.0            0.0 0.0 1.0])
    xt = SVector(0.0, 10.0, 0.0, 0.0)
    ps = SVector(-1.0, 0.4, 0.2)
    yt = vcat(xt, nullify_covariant(gt, xt, ps))
    rt = geodesic_rhs(gt, yt)
    @test rt[5] ≈ -0.5 * c * dot(ps, ps) rtol = 1e-10      # ∂_t g term retained…
    @test abs(rt[5]) > 1e-3                                 # …and unmistakably ≠ 0
end

@testset "escape callback is directional (no inbound kill)" begin
    st = KerrSchildCartesian(0.9)
    # Ray STARTS beyond the escape sphere aimed at the hole: pre-fix the bidirectional
    # escape event killed it at r=escape on the way IN (a vacuum trajectory). It must
    # instead reach the horizon.
    x0 = SVector(0.0, 2000.01, 0.0, 0.0)
    p = metric(st, x0) * SVector(1.0, -1.0, 0.0, 0.0)
    p0 = nullify_covariant(st, x0, SVector(p[2], p[3], p[4]))
    sol = trace_geodesic(st, vcat(x0, p0))                 # default escape=1.1e3 < start r
    rfin = ks_radius(st, SVector(sol.u[end][1], sol.u[end][2], sol.u[end][3], sol.u[end][4]))
    @test rfin < horizon_radius(st) + 1e-2                 # captured at the horizon, not r=escape
    # trace_rays_sampled likewise no longer returns a single-sample (vacuum) trajectory
    states, _ = trace_rays_sampled(st, [vcat(x0, p0)])
    @test length(states[1]) > 2
    # a normal ray starting INSIDE escape still terminates on the OUTBOUND crossing
    xin = SVector(0.0, 1000.0, 40.0, 0.0)
    pin = metric(st, xin) * SVector(1.0, -1.0, 0.0, 0.0)
    p0in = nullify_covariant(st, xin, SVector(pin[2], pin[3], pin[4]))
    sin_ = trace_geodesic(st, vcat(xin, p0in); escape=1100.0)
    rin = ks_radius(st, SVector(sin_.u[end][1], sin_.u[end][2], sin_.u[end][3], sin_.u[end][4]))
    @test rin > 1.0e3                                      # escaped outward (b≫b_c), not killed
end

@testset "stationary inverse_metric_gradient: 3-partial == general 4-column" begin
    # The StationarySpacetime specialization seeds only the 3 spatial duals
    # (∂_t g ≡ 0 exactly); it must be bit-identical to the general 4-column
    # jacobian, including the exact-zero t-column.
    for (a, xyz) in ((0.9, (4.0, 1.0, 2.0)), (0.5, (-5.0, 3.0, 2.5)), (0.998, (2.0, 0.5, -1.0)))
        st = KerrSchildCartesian(a)
        x = SVector(0.0, xyz...)
        dg = inverse_metric_gradient(st, x)
        dg4 = invoke(inverse_metric_gradient,
                     Tuple{AbstractSpacetime,SVector{4,Float64}}, st, x)
        @test dg == dg4
    end
end

@testset "nullify_covariant honors the E>0 contract (errors in the ergosphere)" begin
    st = KerrSchildCartesian(0.9)
    # Inside the ergosphere (KS r≈1.56, f>1) both null roots have p_t>0, so no
    # past-oriented (E>0) covector exists: pre-fix this silently returned E<0.
    xerg = SVector(0.0, 1.8, 0.0, 0.0)
    _, ferg, lerg = SG._rfl(xerg, 0.9)
    @test ferg > 1.0                                       # confirm we are in the ergosphere
    @test_throws ErrorException nullify_covariant(st, xerg, SVector(lerg[2], lerg[3], lerg[4]))
    # Outside the ergosphere the contract still holds: E = -p_t > 0, exactly null.
    xout = SVector(0.0, 100.0, 6.0, 2.0)
    p = nullify_covariant(st, xout, SVector(-1.0, 0.05, 0.02))
    @test -p[1] > 0                                        # E > 0
    @test abs(null_error(st, vcat(xout, p))) < 1e-12
end

@testset "KS closed form: one shared primitive (_rfl == _ks_term, M=1)" begin
    # Single Kerr `_rfl` is now a thin M=1 wrapper of the M-general `_ks_term` (the
    # same primitive the superposed binary uses) — pin that it stays BIT-identical, so
    # a guard added to `_ks_term` reaches single Kerr with no drift.
    for a in (0.0, 0.5, 0.9, 0.998), xyz in ((4.0, 1.0, 2.0), (1.8, 0.0, 0.0), (-5.0, 3.0, 2.5))
        x = SVector(0.0, xyz...)
        r1, f1, l1 = SG._rfl(x, a)
        r2, f2, l2 = SG._ks_term(x[2], x[3], x[4], 1.0, a)
        @test r1 === r2 && f1 === f2 && l1 === l2          # bit-identical, not merely ≈
    end
end

@testset "imaging path widened to AbstractSpacetime (UserMetric smoke)" begin
    # camera_rays / camera_rays_at / trace_rays were ::KerrSchildCartesian-only though
    # their bodies only touch the generic metric / nullify_covariant / trace_geodesic
    # API. A UserMetric wrapping Kerr must now ride the SAME imaging path and reproduce
    # the built-in Kerr rays (to inv-vs-closed-form roundoff on p_t).
    a = 0.9
    kerr = KerrSchildCartesian(a)
    um = UserMetric(x -> metric(kerr, x))
    cam = PlaneParallelCamera(inc_deg=60.0, d=100.0, fov=12.0, npix=5)

    rk = camera_rays(kerr, cam)
    ru = camera_rays(um, cam)                              # pre-fix: MethodError on UserMetric
    @test length(ru) == length(rk) == 25
    @test maximum(maximum(abs.(yu .- yk)) for (yu, yk) in zip(ru, rk)) < 1e-9
    for y in ru
        @test abs(null_error(um, y)) < 1e-10               # null
        @test y[5] ≈ 1.0                                   # p = -k with k_t = -1 (E = 1)
    end
    # camera_rays_at (AMR refinement entry point) is equally widened
    ra = camera_rays_at(um, cam, [(0.0, 0.0), (3.0, -2.0)])
    @test length(ra) == 2 && abs(null_error(um, ra[1])) < 1e-10

    # trace_rays batches through the per-spacetime trace_geodesic; source→camera contract
    sts, dls = trace_rays(um, ru[1:5]; escape=120.0)
    @test length(sts) == 5
    for (s, d) in zip(sts, dls)
        @test length(s) == length(d) && length(s) >= 1
        @test d[end] == 0.0                                # last affine span is 0
        @test all(all(isfinite, y) for y in s)
    end
end

@testset "KS equatorial disk: ring-only singularity (interior limit is exactly η)" begin
    # The 0/0 in f and l_z at r=0 is removable on the OPEN disk x²+y²<a², z=0 —
    # Kerr is truly singular only on the rim. _ks_term now returns the exact limit
    # f=0 (g=η) there and in the ~|z|≲5e-9 slab where hypot cancellation rounds
    # r²→0; the rim and the a=0 origin keep their NaN (true curvature singularity).
    # Everything regularized lies inside the horizon — production tracing stops at
    # r_h and the ADM factory excises via its horizon predicate.
    for a in (0.5, 0.9, 0.998)
        st = KerrSchildCartesian(a)
        for rho in (0.0, 0.4a, 0.9a)                       # open-disk interior, z = 0
            x = SVector(0.0, rho, 0.0, 0.0)
            @test metric(st, x) == SG.ETA                  # exact η, not merely ≈
            @test inverse_metric(st, x) == SG.ETA
        end
        @test any(isnan, metric(st, SVector(0.0, a, 0.0, 0.0)))   # rim NaN kept
        # just above the disk: finite, and → η linearly in z (measured residuals:
        # 0.0 inside the r²-rounding slab at z=1e-9; 2.9e-8 at z=1e-8; 3.8e-6 at
        # z=1e-6 for a=0.9 — pinned with 3x headroom)
        x9 = SVector(0.0, 0.4a, 0.3a, 1.0e-9)
        @test all(isfinite, metric(st, x9))
        @test maximum(abs.(metric(st, x9) - SG.ETA)) < 1e-7
        @test maximum(abs.(metric(st, SVector(0.0, 0.4a, 0.3a, 1.0e-6)) - SG.ETA)) < 4e-5
    end
    # continuity across the regularized plane (a=0.9): |g(z)−g(0)| ~ O(z)
    st9 = KerrSchildCartesian(0.9)
    g0 = metric(st9, SVector(0.0, 0.36, 0.27, 0.0))
    @test maximum(abs.(metric(st9, SVector(0.0, 0.36, 0.27, 1.0e-8)) - g0)) < 1e-7
    # Schwarzschild point singularity unchanged (a=0 has no disk to regularize)
    @test any(isnan, metric(KerrSchildCartesian(0.0), SVector(0.0, 0.0, 0.0, 0.0)))
    # superposed binary rides the same primitive: hole-local open disk → its term
    # vanishes (metric = the OTHER hole's contribution only, finite)
    sks = SuperposedKerrSchild([0.55, 0.45], [0.4, -0.3],
                               [(-6.0, 0.0, 0.0), (6.0, 0.0, 0.0)])
    @test all(isfinite, metric(sks, SVector(0.0, -6.0 + 0.1, 0.0, 0.0)))  # inside hole-1 disk
end
