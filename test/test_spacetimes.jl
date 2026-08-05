# General analytic spacetimes: the superposed KS binary must reduce EXACTLY to
# single Kerr when the second mass -> 0, and give a sensible two-BH image.
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus

@testset "SuperposedKerrSchild reduces to single Kerr (M_B -> 0)" begin
    a = 0.9
    kerr = KerrSchildCartesian(a)
    # one hole at origin == single Kerr; a degenerate second hole with M=0 must not perturb it
    bin1 = SuperposedKerrSchild([1.0], [a], [(0.0, 0.0, 0.0)])
    bin2 = SuperposedKerrSchild([1.0, 0.0], [a, 0.0], [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
    for xyz in ((6.0, 2.0, 1.0), (10.0, -4.0, 3.0), (4.0, 0.0, 2.0))
        x = SVector(0.0, xyz...)
        gk = metric(kerr, x)
        @test maximum(abs.(metric(bin1, x) - gk)) < 1e-12     # one-hole == single Kerr
        @test maximum(abs.(metric(bin2, x) - gk)) < 1e-12     # M_B=0 term vanishes
        # inverse via 4x4 inv matches the analytic KS inverse for the single hole
        @test maximum(abs.(inverse_metric(bin1, x) - inverse_metric(kerr, x))) < 1e-9
        # AD gradient of the inverse works for the (general) binary metric
        dg = inverse_metric_gradient(bin2, x)
        h = 1e-6
        for c in 2:4
            dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
            fd = (inverse_metric(bin2, x + dx) - inverse_metric(bin2, x - dx)) / (2h)
            @test maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)) < 1e-5
        end
    end
end

@testset "Single-hole binary geodesic == Kerr geodesic" begin
    a = 0.9
    kerr = KerrSchildCartesian(a)
    bin = SuperposedKerrSchild([1.0], [a], [(0.0, 0.0, 0.0)])
    x0 = SVector(0.0, 100.0, 6.0, 2.0)
    p0 = nullify_covariant(kerr, x0, SVector(-1.0, 0.05, 0.02))
    y0 = vcat(x0, p0)
    sk = trace_geodesic(kerr, y0; escape=110.0, reltol=1e-10, abstol=1e-12)
    sb = trace_geodesic(bin, y0; escape=110.0, reltol=1e-10, abstol=1e-12)
    # Same RHS (metric identical to 1e-12) => same trajectory. The two traces use
    # different escape surfaces (KS-r vs coordinate-r), so they stop at slightly
    # different points; compare the escape-invariant observable — the final photon
    # 3-momentum DIRECTION (the deflection) — plus null preservation.
    dk = SVector(sk.u[end][6], sk.u[end][7], sk.u[end][8]); dk /= norm(dk)
    db = SVector(sb.u[end][6], sb.u[end][7], sb.u[end][8]); db /= norm(db)
    @test maximum(abs.(dk - db)) < 1e-6                      # identical deflection
    @test abs(null_error(bin, sb.u[end])) < 1e-7            # null preserved on the binary
end

@testset "Equal-mass binary: photons captured by both holes" begin
    # two unit holes on the x-axis at +/-15 M; a ray aimed at each hole is captured
    bin = SuperposedKerrSchild([1.0, 1.0], [0.0, 0.0],
                               [(-15.0, 0.0, 0.0), (15.0, 0.0, 0.0)])
    captured(bx, bz, aim) = begin
        x0 = SVector(0.0, 0.0, 200.0, bz)          # start far along +y
        # aim in -y toward a hole offset in x
        kcon = SVector(1.0, aim, -1.0, 0.0)
        p = metric(bin, x0) * kcon
        p0 = nullify_covariant(bin, x0, SVector(p[2], p[3], p[4]))
        sol = trace_geodesic(bin, vcat(x0, p0 / (-p0[1])); escape=400.0,
                             reltol=1e-9, abstol=1e-9)
        u = sol.u[end]
        min(ks_radius_hole(bin, SVector(u[1], u[2], u[3], u[4]), 1),
            ks_radius_hole(bin, SVector(u[1], u[2], u[3], u[4]), 2)) < 3.0
    end
    # straight-down ray offset toward +x hole (x=15) with small aim gets captured
    @test captured(0.0, 0.0, 0.075)      # aimed at the +x hole
    @test captured(0.0, 0.0, -0.075)     # aimed at the -x hole
    # a ray far off to the side escapes
    x0 = SVector(0.0, 60.0, 200.0, 0.0)
    p0 = nullify_covariant(bin, x0, SVector(0.0, -1.0, 0.0))
    sol = trace_geodesic(bin, vcat(x0, p0 / (-p0[1])); escape=400.0, reltol=1e-9, abstol=1e-9)
    u = sol.u[end]
    @test sqrt(u[2]^2 + u[3]^2 + u[4]^2) > 100    # escaped
end

@testset "SuperposedKerrSchild escape is directional (no inbound kill)" begin
    a = 0.9
    bin = SuperposedKerrSchild([1.0], [a], [(0.0, 0.0, 0.0)])
    # Ray starting BEYOND the escape sphere aimed at the hole must reach the horizon,
    # not terminate at r=escape on the way in (pre-fix bidirectional escape callback).
    x0 = SVector(0.0, 2000.01, 0.0, 0.0)
    p = metric(bin, x0) * SVector(1.0, -1.0, 0.0, 0.0)
    p0 = nullify_covariant(bin, x0, SVector(p[2], p[3], p[4]))
    sol = trace_geodesic(bin, vcat(x0, p0))       # default escape=1.1e3 < start r
    u = sol.u[end]
    rfin = ks_radius_hole(bin, SVector(u[1], u[2], u[3], u[4]), 1)
    @test rfin < horizon_radius_hole(bin, 1) + 1e-2   # captured, not stopped at r=escape
end

@testset "binary rides the widened camera/trace_rays imaging path" begin
    # camera_rays / trace_rays were ::KerrSchildCartesian-only, forcing this file to
    # hand-roll binary rays. Both are now AbstractSpacetime-generic (the binary's own
    # trace_geodesic owns per-hole capture), so the standard imaging path images a
    # binary directly. Equal-mass holes on the x-axis, edge-on camera.
    bin = SuperposedKerrSchild([1.0, 1.0], [0.0, 0.0],
                               [(-8.0, 0.0, 0.0), (8.0, 0.0, 0.0)])
    cam = PlaneParallelCamera(inc_deg=90.0, d=300.0, fov=40.0, npix=6)
    rays = camera_rays(bin, cam)
    @test length(rays) == 36
    @test all(abs(null_error(bin, y)) < 1e-9 for y in rays)          # null on the binary metric
    @test all(y[5] ≈ -1.0 for y in rays)                             # p_t = -1 normalized

    sts, dls = trace_rays(bin, rays; escape=400.0, reltol=1e-8, abstol=1e-10)
    @test length(sts) == 36
    for (s, d) in zip(sts, dls)
        @test length(s) == length(d) && d[end] == 0.0               # source→camera contract
        @test all(all(isfinite, y) for y in s)
    end
    # A head-on ray (image centre ⇒ coasts through the +x hole) must be CAPTURED: its
    # deepest sample sits at that hole's horizon. Measure distance from the hole CENTRE
    # (the holes are offset to x=±8), not the origin.
    holedist(y) = min(sqrt((y[2] + 8)^2 + y[3]^2 + y[4]^2),
                      sqrt((y[2] - 8)^2 + y[3]^2 + y[4]^2))
    scen, _ = trace_rays(bin, camera_rays_at(bin, cam, [(0.0, 0.0)]);
                         escape=400.0, reltol=1e-8, abstol=1e-10)
    @test minimum(holedist(y) for y in scen[1]) < 3.0              # reached a hole horizon
end

# --- regression: trace_rays_sampled sizehint! preallocation is value-preserving --------------
# The fixed-control sampler grows two push! buffers (ys, hs) one element per RK4 step; these
# are sizehint!(·, 1024)'d to cut the _growend! reallocation churn. sizehint! changes ONLY
# the allocation strategy, so states/steps must be BIT-IDENTICAL. This pins that against a
# verbatim no-hint reference march (same
# pattern as test_transfer.jl's mutable-loop references), which also locks the sampling
# contract (h=(r-r_h)/C, r_h+1e-3 capture, source→camera reversal, dl[k]=step-behind).
@testset "trace_rays_sampled sizehint! preallocation is bit-identical" begin
    # verbatim pre-sizehint loop (defaults C=40, escape=ESCAPE_RADIUS=1.1e3, nmax=6000).
    # Kept under Threads.@threads exactly like the real function: @threads lowers the body
    # into a closure, and that boundary fixes how the inlined rk4_step_fixed contracts its
    # fmuladds — a plain for-loop here would round 1-2 ulp differently and fail bitwise.
    function ref_sampled(st, rays; C, escape, nmax)
        r_h = horizon_radius(st)
        states = Vector{Vector{SVector{8,Float64}}}(undef, length(rays))
        dls = Vector{Vector{Float64}}(undef, length(rays))
        Threads.@threads for i in eachindex(rays)
            ys = SVector{8,Float64}[rays[i]]; hs = Float64[]; y = rays[i]; r_prev = Inf
            for _ in 1:nmax
                x = SVector(y[1], y[2], y[3], y[4])
                r = ks_radius(st, x)
                (r < r_h + 1e-3 || !isfinite(r)) && break
                ((r > escape) && r > r_prev) && break
                r_prev = r
                h = (r - r_h) / C
                h = min(h, Inf)         # mirrors the max_step=Inf cap: a value barrier that
                                        # pins h's rounding before the inlined rk4 (fmuladd)
                y = rk4_step_fixed(st, y, h)
                push!(ys, y); push!(hs, h)
            end
            m = length(ys)
            rev = Vector{SVector{8,Float64}}(undef, m); dl = Vector{Float64}(undef, m)
            for k in 1:m; rev[k] = ys[m+1-k]; end
            for k in 1:m-1; dl[k] = hs[m-k]; end
            dl[m] = 0.0
            states[i] = rev; dls[i] = dl
        end
        return states, dls
    end
    st = KerrSchildCartesian(0.9375)
    cam = PlaneParallelCamera(60.0, 1000.0, 40.0, 32)
    rays = camera_rays(st, cam)
    s, d = trace_rays_sampled(st, rays)                            # sizehint'd path, defaults
    sref, dref = ref_sampled(st, rays; C=40.0, escape=1.1e3, nmax=6000)
    @test length(s) == length(rays) == 32 * 32
    @test all(s[i] == sref[i] for i in eachindex(rays))           # states BIT-identical
    @test all(d[i] == dref[i] for i in eachindex(rays))           # affine steps BIT-identical
    @test maximum(length, s) > 100                                # captured rays exercise the tail
end
