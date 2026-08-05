# Kerr-as-numerical-grid: sample analytic KerrSchildCartesian onto a regular
# Cartesian grid, then check that the numerical spacetime reproduces the
# analytic one — metric values, AD-through-cubic-interpolation gradients, and a
# traced geodesic — all to interpolation order (needs no external data). This
# shows a numerical spacetime rides the SAME AD-derived-∂g geodesic engine as
# the analytic metrics. Also covers the horizon-excision factory kwarg, the
# AthenaK .bin reader on synthetic dumps, and the ADM-grid imaging path.
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus

@testset "Kerr-as-numerical-grid" begin
    a = 0.9
    kerr = KerrSchildCartesian(a)
    xs = -20.0:0.5:20.0                       # 81^3 = 531441 nodes (< 1e6), inscribed radius 20 M
    # horizon-aware excision: the factory's collapsed-lapse fill inside r_h is the
    # tracer's inner stopping surface (the KS metric itself is regular — exactly η —
    # on the open equatorial disk, so no node NaNs on its own except the rim)
    grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min=0.1,
                                excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
    @test grid isa ADMGridSpacetime
    @test grid isa BlackLightPlus.Geodesics.StationarySpacetime   # rides the analytic engine

    # (1) metric(grid, x) matches metric(kerr, x) at OFF-NODE interior points to
    #     cubic-interpolation order (expect ~1e-5 for smooth Kerr away from the hole).
    @testset "metric match to interpolation order" begin
        maxrel = 0.0
        for pt in ((7.3, -4.1, 2.7), (5.55, 3.35, -1.15), (-6.7, 2.2, 4.4), (9.1, 5.3, -3.7))
            x = SVector(0.0, pt...)
            gg = metric(grid, x); gk = metric(kerr, x)
            @test gg ≈ gg'                                        # symmetric
            rel = maximum(abs.(gg - gk)) / maximum(abs.(gk))
            maxrel = max(maxrel, rel)
            @test rel < 5e-3
        end
        @info "grid metric match" max_relative = maxrel
        # sample_adm round-trips a lapse in a sane physical range (α ∈ (0,1] far from hole)
        α, β, γ = sample_adm(grid, 7.3, -4.1, 2.7)
        @test 0.5 < α ≤ 1.0
        @test det(γ) > 1.0                                        # γ = I + f l_i l_j, det = 1 + f > 1
    end

    # (2) AD inverse_metric_gradient(grid, x) matches central FD of the gridded
    #     inverse metric — the load-bearing check that ForwardDiff.Dual flows
    #     through the cubic interpolant (both are the derivative of the SAME
    #     interpolant, so agreement is to FD truncation, not interpolation error).
    @testset "AD through cubic interpolation == finite differences" begin
        x = SVector(0.0, 7.3, -4.1, 2.7)
        dg = inverse_metric_gradient(grid, x)
        h = 1e-3
        maxad = 0.0
        for c in 2:4
            dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
            fd = (inverse_metric(grid, x + dx) - inverse_metric(grid, x - dx)) / (2h)
            err = maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd))
            maxad = max(maxad, err)
            @test err < 1e-2
        end
        @info "AD-vs-FD" max_abs = maxad
    end

    # (3) A geodesic traced through the gridded Kerr matches the analytic-Kerr
    #     geodesic. Photon starts INSIDE the box (r0 ≈ 17 M, off-node, off-plane)
    #     with impact parameter b ≈ 8 M so it scatters; escape ≈ grid half-width so
    #     the ray leaves through the escape sphere while still on-grid. Compare the
    #     asymptotic propagation direction (invariant along the straight outgoing
    #     ray ⇒ insensitive to the exact termination surface).
    @testset "gridded-Kerr geodesic == analytic-Kerr geodesic" begin
        x0 = SVector(0.0, 15.2, 7.7, 1.3)                        # r0 ≈ 17.09, off every node
        kcon = SVector(1.0, -1.0, 0.0, 0.0)                      # aim inward, b ≈ 7.7
        p = metric(kerr, x0) * kcon
        p0 = nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4]))
        y0 = vcat(x0, p0)
        @test abs(null_error(grid, y0)) < 1e-6                   # interpolated metric ~ null at start
        sk = trace_geodesic(kerr, y0; escape=19.0, reltol=1e-10, abstol=1e-12)
        sg = trace_geodesic(grid, y0; escape=19.0, reltol=1e-10, abstol=1e-12)
        uk, ug = sk.u[end], sg.u[end]
        @test norm(SVector(uk[2], uk[3], uk[4])) > 15.0          # analytic ray scattered/escaped
        @test norm(SVector(ug[2], ug[3], ug[4])) > 15.0          # gridded ray scattered/escaped
        outdir(st, u) = begin
            v = inverse_metric(st, SVector(u[1], u[2], u[3], u[4])) * SVector(u[5], u[6], u[7], u[8])
            s = SVector(v[2], v[3], v[4]); s / norm(s)
        end
        dirdiff = norm(outdir(kerr, uk) - outdir(grid, ug))
        @info "geodesic match" asymptotic_direction_diff = dirdiff null_error_end = null_error(grid, ug)
        @test dirdiff < 1e-2                                     # interpolation-limited
        @test abs(null_error(grid, ug)) < 1e-6                   # null preserved along the gridded trace
    end

    # Off-grid → Minkowski + clean termination contract.
    @testset "off-grid returns Minkowski" begin
        xo = SVector(0.0, 100.0, 0.0, 0.0)                       # far outside the box
        @test metric(grid, xo) == BlackLightPlus.Geodesics.ETA
        @test !BlackLightPlus.Geodesics.on_grid(grid, xo)
    end

    # A CAPTURED ray (the shipped test above only traces a scattering one) must
    # terminate cleanly. Pre-fix the factory baked NaN at the KS singular disk
    # (r→0), which poisoned the interpolation stencil OUTSIDE the horizon and
    # aborted captured rays with retcode Unstable + an all-NaN final state; now
    # the horizon `excise` predicate supplies the sub-threshold lapse surface.
    @testset "captured ray terminates cleanly (no NaN nodes)" begin
        @test count(isnan, grid.data) == 0                       # excision fill is finite
        @test all(isfinite, metric(grid, SVector(0.0, 1.49, 1.49, 0.99)))  # KS r≈2.19 > r_h, finite
        # ray aimed INTO the hole terminates with a finite state (pre-fix: all-NaN)
        x0 = SVector(0.0, 15.2, 0.0, 1.3)
        p = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        p0 = nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4]))
        sg = trace_geodesic(grid, vcat(x0, p0); escape=19.0)
        @test Symbol(sg.retcode) == :Terminated                  # not Unstable
        @test all(isfinite, sg.u[end])
        @test norm(SVector(sg.u[end][2], sg.u[end][3], sg.u[end][4])) < 5.0   # reached the hole
        # a ray STARTING beyond the escape sphere is not killed on its way in
        # (directional escape callback): escape=10 < start r≈15.3.
        sb = trace_geodesic(grid, vcat(x0, p0); escape=10.0)
        @test all(isfinite, sb.u[end])
        @test norm(SVector(sb.u[end][2], sb.u[end][3], sb.u[end][4])) < 5.0   # captured, not killed at r=10
    end
end

# ------------------------------------------------------------------------------------
# Horizon-excision factory kwarg (regression): the non-finite trigger alone excises
# ONLY nodes where the sampled metric NaNs — for Kerr the measure-zero z=0 disk
# nodes. A grid with the SAME spacing/extent but shifted off that plane (no z=0
# node) then has zero excised nodes and NO inner stopping surface: the true KS
# lapse never drops below alpha_min at any node (min stored lapse ≈ 0.55 ≫ 0.1),
# so a captured ray aborts Unstable with dt forced below eps. Passing the horizon
# predicate via `excise` restores a grid-alignment-independent stopping surface.
@testset "horizon-excision kwarg: shifted grid (no z=0 node plane) captures cleanly" begin
    a = 0.9
    kerr = KerrSchildCartesian(a)
    r_h = horizon_radius(kerr)
    xs = -19.75:0.5:20.25                     # the grid above shifted by dx/2
    grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min=0.1,
                                excise = x -> ks_radius(kerr, x) < r_h)
    @test count(isnan, grid.data) == 0
    @test minimum(@view grid.data[1, :, :, :]) < 0.1   # a real sub-threshold lapse surface
    # far-field nodes untouched by excision: interpolated metric still matches Kerr
    xfar = SVector(0.0, 7.3, -4.1, 2.7)
    @test maximum(abs.(metric(grid, xfar) - metric(kerr, xfar))) /
          maximum(abs.(metric(kerr, xfar))) < 5e-3
    # captured ray (same setup as the captured-ray test above) terminates cleanly
    x0 = SVector(0.0, 15.2, 0.0, 1.3)
    p = metric(kerr, x0) * SVector(1.0, -1.0, 0.0, 0.0)
    p0 = nullify_covariant(kerr, x0, SVector(p[2], p[3], p[4]))
    sg = trace_geodesic(grid, vcat(x0, p0); escape=19.0)
    @test Symbol(sg.retcode) == :Terminated            # not Unstable
    @test all(isfinite, sg.u[end])
    @test norm(SVector(sg.u[end][2], sg.u[end][3], sg.u[end][4])) < 5.0   # reached the hole
end

# ------------------------------------------------------------------------------------
# AthenaK .bin reader honours the header's declared sizes (regression): a
# single-precision AthenaK build (Athena_SINGLE_PRECISION=ON) writes 4-byte block
# bounds ("size of location=4"); the reader previously hardcoded 8-byte bounds and
# returned a silently floor-truncated, byte-shifted snapshot. Synthetic v1.1 dumps,
# no external data.
function _write_bin_dump(path::AbstractString, ::Type{LocT}; var_size::Int = 4) where {LocT}
    nx = 4; cells = nx^3; nvars = 2
    params = "<par_end>\n"
    open(path, "w") do io
        print(io, "Athena binary output version=1.1\n")
        print(io, "  size of preheader=5\n")
        print(io, "  time=3.5\n")
        print(io, "  cycle=7\n")
        print(io, "  size of location=", sizeof(LocT), "\n")
        print(io, "  size of variable=", var_size, "\n")
        print(io, "  number of variables=", nvars, "\n")
        print(io, "  variables:  z4c_alpha  z4c_chi  \n")
        print(io, "  header offset=", length(params), "\n")
        print(io, params)
        for b in 0:1                                  # two blocks tiling x in [0,2]
            write(io, Int32[0, nx - 1, 0, nx - 1, 0, nx - 1, b, 0, 0, 0])
            write(io, LocT[b, b + 1, 0, 1, 0, 1])
            for v in 0:nvars-1, c in 0:cells-1        # disk order: i fastest, then j,k,var
                write(io, Float32(1000 * (b + 1) + 100 * v + c))
            end
        end
    end
    return path
end

@testset "athenak .bin reader honours declared header sizes" begin
    mktempdir() do dir
        for LocT in (Float64, Float32)                # 8- and 4-byte "size of location"
            s = read_athenak_bin(_write_bin_dump(joinpath(dir, "loc$(sizeof(LocT)).bin"), LocT))
            @test s.n_mb == 2
            @test s.time == 3.5
            @test (s.nx1, s.nx2, s.nx3) == (4, 4, 4)
            @test s.x1min == [0.0, 1.0] && s.x1max == [1.0, 2.0]
            @test s.x2min == [0.0, 0.0] && s.x3max == [1.0, 1.0]
            @test s.data[1, 1, 1, 1, 1] == 1000.0f0   # block 1, var 1, first cell
            @test s.data[2, 1, 1, 2, 2] == 2101.0f0   # block 2, var 2, second cell
        end
        # unsupported / inconsistent layouts fail loudly instead of returning garbage
        @test_throws ErrorException read_athenak_bin(
            _write_bin_dump(joinpath(dir, "var8.bin"), Float64; var_size = 8))
        p = _write_bin_dump(joinpath(dir, "trunc.bin"), Float64)
        open(p, "a") do io; write(io, 0x00); end      # stray trailing byte
        @test_throws ErrorException read_athenak_bin(p)
    end
end

@testset "ADM grid rides the widened camera/trace_rays imaging path" begin
    # camera_rays / trace_rays were ::KerrSchildCartesian-only; a numerical spacetime
    # now images through the SAME path (ADMGridSpacetime's own trace_geodesic owns the
    # lapse-based termination). Kerr-as-grid, coarse box; escape ≲ grid half-width.
    a = 0.9
    kerr = KerrSchildCartesian(a)
    xs = -15.0:1.0:15.0
    grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min=0.1,
                                excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
    cam = PlaneParallelCamera(inc_deg=60.0, d=120.0, fov=8.0, npix=4)

    rays = camera_rays(grid, cam)                       # camera is off-grid ⇒ flat-null rays
    @test length(rays) == 16
    @test all(abs(null_error(grid, y)) < 1e-9 for y in rays)
    @test all(y[5] ≈ -1.0 for y in rays)                # p_t = -1 normalized

    sts, dls = trace_rays(grid, rays; escape=13.0, reltol=1e-9, abstol=1e-11)
    @test length(sts) == 16
    for (s, d) in zip(sts, dls)
        @test length(s) == length(d) && d[end] == 0.0   # source→camera contract
        @test all(all(isfinite, y) for y in s)          # no NaN plunge (factory excision holds)
    end
    # small fov aims into the strong field ⇒ ≥1 ray reaches the numerical horizon
    deepr(s) = minimum(sqrt(y[2]^2 + y[3]^2 + y[4]^2) for y in s)
    @test any(deepr(s) < 6.0 for s in sts)
end
