# fig_capture_dyn.jl — ray fates on the image of the orbiting equal-mass binary of
# fig_dynrays.jl (boosted superposed Kerr–Schild on a CircularOrbit, metric and ∂_t g at the
# ray's own coordinate time) at the view of its frame 1 (panel (d) of fig_dynrays.pdf), with
# seven levels of adaptive image-plane refinement. Drawn by paper/scripts/plot_capture_dyn.py
# into fig_capture_dyn.pdf.
#
# Binary, camera, classes, capture test and integrator are those of the maps part of
# fig_dynrays.jl: masses 1/2 and 1/2 (M = 1), no spin, separation 8 M, Keplerian orbit
# prograde about +z with hole 1 at +x at t = −DCAM; plane-parallel camera at inclination 60°
# from the orbital axis and distance DCAM = 100 M, box |α|, |β| ≤ 8 M, observer time
# T_OBS = T/8; Vern7, reltol 1e-9, abstol 1e-11, λ ≤ 600, escape at R > 250 M. Classes:
# 0 escaped with bending ≤ 360°, 1 captured by hole 1, 2 by hole 2, 3 escaped after bending
# > 360°, 4 undetermined (affine budget); bending is the summed turning angle of the path.
# Capture: r'_A < r_h,A + 2e-3 or normal-frame energy above 0.99 of the redshift cut, by the
# nearer hole. t_pass is the median time at which the escaped (class 0) rays of level 0
# cross the sky plane d̂·x = 0, d̂ = (sin i, 0, cos i).
#
# Refinement: a BASE² grid of pixels (each traced at its centre) in BLOCK² blocks; a block of
# level l < MAXLEVEL is split into four blocks of BLOCK² pixels at twice the density when its
# pixels and its one-pixel border hold more than one class. A traced block reads its border at
# level l (or at the finest level traced there); after every pass each unsplit block below
# MAXLEVEL is tested again against the ring of finest pixels around it and split when the ring
# holds another class, so a boundary found later beside a block is followed into it. Passes
# run until nothing splits; then every unsplit block below MAXLEVEL holds one class within it
# and its ring, and every class boundary lies between blocks of MAXLEVEL (recorded as the
# count of boundary edges touching a coarser block). Level l has BASE 2^l pixels across the
# box. The plain rule (no second test), that of fig_dynrays.jl, is the check part's.
#
# Convergence, for (L − 1, L) = (MAXLEVEL − 2, MAXLEVEL − 1) and (MAXLEVEL − 1, MAXLEVEL), on
# the maps of separate runs to maximum levels L − 1 and L: the area of each class, its
# boundary length (4-neighbour pixel edges between classes), the area change in pixels of
# level L − 1 and as a mean normal shift (area change over boundary length), the pixels whose
# class changes (centres) and their distance to the level-(L − 1) boundary, and the distance
# between the two boundaries (from each edge midpoint of one to the nearest pixel edge of the
# other), each way; distances in pixels of level L − 1.
#
# check: the maps settings and plain rule of fig_dynrays.jl (BASE 128, MAXLEVEL 3) at the
# same frame, compared pixel by pixel with figures/paper/out/dynrays/map_moving_1.npy; the
# map part also runs BASE to maximum level 2 (1024², the grid of panel (d)) with the ring
# test and with the plain rule and compares both with it.
#
# Writes BLP_CAPDYN_OUT (default figures/paper/out/capture_dyn):
#   class_L<L>.npy   Int8 class plane, (BASE 2^L)², [i_β, i_α], of the run to maximum level
#                    L, for L = 2 and L = MAXLEVEL − 2 … MAXLEVEL
#   level.npy        Int8, finest level per finest block of the MAXLEVEL run,
#                    (BASE 2^MAXLEVEL / BLOCK)²
#   records.json     setup; per run and pass the blocks, rays, levels, splits (by the first
#                    test and by the ring test), class counts of the traced rays and wall
#                    time; rays per level; the ray total and its fraction of a uniform frame;
#                    the boundary edges touching coarser blocks; t_pass and the hole
#                    positions at t_pass and T_OBS (orbital plane and image); the
#                    convergence; the comparison with panel (d); code SHA, threads
#   records_check.json   the check part
# The .npy files are Fortran-ordered (numpy reads them as [i_β, i_α] without a copy).
# Run from the repository root:
#   julia -t 40 --project=. figures/paper/fig_capture_dyn.jl
# Env: BLP_CAPDYN_PARTS (check,map), BLP_CAPDYN_BASE (256), BLP_CAPDYN_MAXLEVEL (7),
#      BLP_CAPDYN_OUT.
using Dates, LinearAlgebra, Printf, Statistics, StaticArrays
using Adapt
using BlackLightPlus
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, trace_geodesic,
                                null_covariant, metric, ks_radius, horizon_radius,
                                normal_frame_energy, CAPTURE_REDSHIFT
using BlackLightPlus.Execution: ThreadedExecutor, image_coords, BlockGrid, ImageBlock,
                                stream_blocks!

const ROOT = dirname(dirname(@__DIR__))
const OUT = get(ENV, "BLP_CAPDYN_OUT", joinpath(@__DIR__, "out", "capture_dyn"))
const DYN = joinpath(@__DIR__, "out", "dynrays")
const PARTS = split(get(ENV, "BLP_CAPDYN_PARTS", "check,map"), ",")
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
envi(k, d) = parse(Int, get(ENV, k, string(d)))

const MASSES = (0.5, 0.5)
const SEP = 8.0
const OMEGA = sqrt(sum(MASSES) / SEP^3)
const PERIOD = 2π / OMEGA
const INC = 60.0
const DCAM = 100.0
const LIM = 8.0
const FOV = 2 * LIM
const T_OBS = 1 * (PERIOD / 2) / 4          # frame 1 of the four frames of fig_dynrays.jl
const LMAX = 600.0
const RESC = 250.0
const RELTOL = 1e-9
const ABSTOL = 1e-11
const BEND_CUT = 360.0
const BASE = envi("BLP_CAPDYN_BASE", 256)
const BLOCK = 8
const MAXLEVEL = envi("BLP_CAPDYN_MAXLEVEL", 7)
const CHECK_BASE = 128                       # the maps settings of fig_dynrays.jl
const CHECK_MAXLEVEL = 3
const PANEL_D_LEVEL = 2                      # BASE 2^2 = 1024, the effective size of panel (d)
const RMAX = 16                              # distance search radius, buckets of 4 pixels
const SI = sin(deg2rad(INC)); const CI = cos(deg2rad(INC))
const DHAT = SVector(SI, 0.0, CI)
const EXEC = ThreadedExecutor(chunk = BLOCK^2)

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

orbit() = CircularOrbit(MASSES, SEP; phase = OMEGA * DCAM)
moving(o) = BoostedSuperposedKerrSchild(collect(MASSES), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], o)
holes_at(o, t) = [collect(o(t)[A][1]) for A in 1:2]
"image-plane position (α, β) of a point s"
img(s) = [s[2], -s[1] * CI + s[3] * SI]

# ---------------------------------------------------------------- one ray
"fate of an end state: 1/2 captured by the nearer hole, 0 escaped, 3 budget"
function fate(st, u)
    x = SVector(u[1], u[2], u[3], u[4])
    R = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
    dA = (ks_radius(st, x, 1) - horizon_radius(st, 1), ks_radius(st, x, 2) - horizon_radius(st, 2))
    if minimum(dA) < 2e-3 || normal_frame_energy(st, u) > 0.99CAPTURE_REDSHIFT
        return dA[1] <= dA[2] ? 1 : 2
    end
    R > 0.99RESC && return 0
    return 3
end

"coordinate time at which the saved path first crosses n̂·x = 0 going away from the camera"
function plane_time(us, n̂)
    s(u) = n̂[1] * u[2] + n̂[2] * u[3] + n̂[3] * u[4]
    for k in 1:length(us)-1
        s1 = s(us[k]); s2 = s(us[k+1])
        if s1 > 0 && s2 <= 0
            w = s1 / (s1 - s2)
            return us[k][1] + w * (us[k+1][1] - us[k][1])
        end
    end
    return NaN
end

"summed turning angle (degrees) of the saved coordinate path"
function bend_deg(us)
    tot = 0.0
    prev = nothing
    for k in 1:length(us)-1
        d = SVector(us[k+1][2] - us[k][2], us[k+1][3] - us[k][3], us[k+1][4] - us[k][4])
        n = norm(d)
        n < 1e-12 && continue
        tt = d / n
        prev !== nothing && (tot += acos(clamp(dot(prev, tt), -1.0, 1.0)))
        prev = tt
    end
    return rad2deg(tot)
end

# backward ray of image point (a, b) received at observer time t along the inward plane
# normal d̂; α along y, β along (-cos i, 0, sin i)
function map_ray(st, t, a, b)
    x0 = SVector(t, -b * CI + DCAM * SI, a, b * SI + DCAM * CI)
    k = metric(st, x0) * SVector(1.0, SI, 0.0, CI)
    knull, ok = null_covariant(st, x0, SVector(k[2], k[3], k[4]))
    ok || error("no positive-energy null root at the camera")
    return vcat(x0, knull / knull[1])
end

"(class, bending in degrees, sky-plane crossing time, 0)"
function classify(st, t, a, b)
    sol = trace_geodesic(st, map_ray(st, t, a, b); λmax = LMAX, escape = RESC,
                         reltol = RELTOL, abstol = ABSTOL, save_everystep = true)
    f = fate(st, sol.u[end])
    bend = bend_deg(sol.u)
    tc = plane_time(sol.u, DHAT)
    f in (1, 2) && return SVector(Float64(f), bend, tc, 0.0)
    f == 0 && return SVector(bend > BEND_CUT ? 3.0 : 0.0, bend, tc, 0.0)
    return SVector(4.0, bend, tc, 0.0)
end

struct ClassProgram{S,G} <: Function
    st::S
    t::Float64
    gen::G
end
Adapt.@adapt_structure ClassProgram
function (p::ClassProgram)(item, i, out)
    a, b = image_coords(p.gen, item)
    @inbounds out[i] = classify(p.st, p.t, a, b)
    return nothing
end

# ---------------------------------------------------------------- the refined image
base_work(base) = (px0 = FOV / base; nb = base ÷ BLOCK;
                   vec([ImageBlock(bi, bj, 0, -FOV / 2 + (bi - 1) * BLOCK * px0,
                                   -FOV / 2 + (bj - 1) * BLOCK * px0, BLOCK * px0)
                        for bi in 1:nb, bj in 1:nb]))

"values of every pixel of `work`, block w at (w − 1) BLOCK² + (i − 1) BLOCK + j"
function trace_level(st, t, work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    S = Vector{SVector{4,Float64}}(undef, length(work) * bb)
    index = Dict(blk => w for (w, blk) in enumerate(work))
    visit = (blk, vals) -> begin
        o = (index[blk] - 1) * bb
        for q in 1:bb
            S[o + q] = vals[q]
        end
        return ImageBlock[]
    end
    stream_blocks!(EXEC, ClassProgram(st, t, BlockGrid(BLOCK)), work, BLOCK, visit;
                   outtype = SVector{4,Float64})
    return S
end

"lower-left pixel (i_α, i_β), zero-based, of block `blk` on the grid of level-l pixels"
corner(blk, base, l) = (ps = FOV / (base * 2^l);
                        (round(Int, (blk.a0 + FOV / 2) / ps), round(Int, (blk.b0 + FOV / 2) / ps)))

"paint the classes of a level-l block onto the plane P of level L ≥ l"
function paint!(P::Matrix{Int8}, blk, vals, base, l, L)
    ix0, iy0 = corner(blk, base, l)
    s = 2^(L - l)
    @inbounds for i in 1:BLOCK, j in 1:BLOCK
        v = Int8(vals[(i - 1) * BLOCK + j][1])
        a0 = (ix0 + i - 1) * s; b0 = (iy0 + j - 1) * s
        for ca in a0+1:a0+s, cb in b0+1:b0+s
            P[cb, ca] = v
        end
    end
    return nothing
end

"mark the finest blocks inside a level-l block with l"
function paint_level!(lev::Matrix{Int8}, blk, base, l, maxlevel)
    ix0, iy0 = corner(blk, base, l)
    s = 2^(maxlevel - l)                    # finest blocks per side of this block
    a0 = (ix0 ÷ BLOCK) * s; b0 = (iy0 ÷ BLOCK) * s
    @inbounds for ca in a0+1:a0+s, cb in b0+1:b0+s
        lev[cb, ca] = Int8(l)
    end
    return nothing
end

"more than one class among the block's level-l pixels and its one-pixel border"
function needs_split(C::Matrix{Int8}, blk, base, maxlevel)
    l = blk.level
    ix0, iy0 = corner(blk, base, l)
    n = base * 2^l
    s = 2^(maxlevel - l)
    @inbounds c0 = C[iy0 * s + 1, ix0 * s + 1]
    @inbounds for ix in ix0-1:ix0+BLOCK, iy in iy0-1:iy0+BLOCK
        (0 <= ix < n && 0 <= iy < n) || continue
        c = C[iy * s + 1, ix * s + 1]       # level l, or the finest level traced there
        c >= 0 && c != c0 && return true
    end
    return false
end

"another class than the block's own in the ring of finest pixels around an unsplit block"
function ring_differs(C::Matrix{Int8}, blk, base, maxlevel)
    ix0, iy0 = corner(blk, base, blk.level)
    s = 2^(maxlevel - blk.level)
    a0 = ix0 * s; b0 = iy0 * s; w = BLOCK * s; n = size(C, 1)
    @inbounds c0 = C[b0 + 1, a0 + 1]
    @inbounds for ca in a0:a0+w+1, cb in (b0, b0 + w + 1)
        (1 <= ca <= n && 1 <= cb <= n) && C[cb, ca] != c0 && return true
    end
    @inbounds for ca in (a0, a0 + w + 1), cb in b0+1:b0+w
        (1 <= ca <= n && 1 <= cb <= n) && C[cb, ca] != c0 && return true
    end
    return false
end

children(blk) = (half = blk.span / 2;
                 [ImageBlock(blk.bi, blk.bj, blk.level + 1, blk.a0 + da, blk.b0 + db, half)
                  for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))])

"""
The refined class map of observer time t, (base 2^maxlevel)², with the finest level per
finest block, the level-0 values and the per-pass record. retest = false is the plain rule.
"""
function amr_frame(st, t, base, maxlevel; retest = true)
    bb = BLOCK * BLOCK
    eff = base * 2^maxlevel
    C = fill(Int8(-1), eff, eff)
    lev = zeros(Int8, eff ÷ BLOCK, eff ÷ BLOCK)
    work = base_work(base)
    leaves = ImageBlock[]                   # unsplit blocks below maxlevel
    stats = Dict{String,Any}[]
    rays_level = zeros(Int, maxlevel + 1)
    S0 = SVector{4,Float64}[]
    pass = 0
    while !isempty(work)
        t0 = time()
        S = trace_level(st, t, work)
        tw = time() - t0
        pass == 0 && (S0 = S)
        Threads.@threads for w in eachindex(work)
            paint!(C, work[w], view(S, (w - 1) * bb + 1:w * bb), base, work[w].level, maxlevel)
            paint_level!(lev, work[w], base, work[w].level, maxlevel)
        end
        splits = Vector{Bool}(undef, length(work))
        Threads.@threads for w in eachindex(work)
            splits[w] = work[w].level < maxlevel && needs_split(C, work[w], base, maxlevel)
        end
        kids = ImageBlock[]
        for (w, blk) in enumerate(work)
            splits[w] ? append!(kids, children(blk)) : (blk.level < maxlevel && push!(leaves, blk))
        end
        nre = 0
        if retest
            again = Vector{Bool}(undef, length(leaves))
            Threads.@threads for k in eachindex(leaves)
                again[k] = ring_differs(C, leaves[k], base, maxlevel)
            end
            for k in eachindex(leaves)
                again[k] && append!(kids, children(leaves[k]))
            end
            nre = count(again)
            leaves = leaves[.!again]
        end
        lv = [count(b -> b.level == q, work) for q in 0:maxlevel]
        for q in 0:maxlevel
            rays_level[q + 1] += lv[q + 1] * bb
        end
        cls = [count(s -> s[1] == q, S) for q in 0:4]
        push!(stats, Dict{String,Any}("pass" => pass, "blocks" => length(work), "rays" => length(S),
                                      "blocks_by_level" => lv, "split" => count(splits),
                                      "split_by_ring" => nre, "classes_traced" => cls,
                                      "wall_s_trace" => tw, "wall_s" => time() - t0))
        logf("  pass %d  %8d blocks  %9d rays  levels %s  split %7d + ring %4d  classes %s  %7.1f s\n",
             pass, length(work), length(S), string(findall(>(0), lv) .- 1), count(splits), nre,
             string(cls), time() - t0)
        work = kids
        pass += 1
    end
    return C, lev, S0, stats, rays_level
end

"class boundary edges (4-neighbour pixels of different class) with a pixel in a block below maxlevel"
function coarse_boundary_edges(C::Matrix{Int8}, lev::Matrix{Int8}, maxlevel)
    n = size(C, 1)
    cnt = zeros(Int, n)
    Threads.@threads for ia in 1:n
        @inbounds for ib in 1:n
            cp = lev[(ib - 1) ÷ BLOCK + 1, (ia - 1) ÷ BLOCK + 1] < maxlevel
            ib < n && C[ib + 1, ia] != C[ib, ia] &&
                (cp || lev[ib ÷ BLOCK + 1, (ia - 1) ÷ BLOCK + 1] < maxlevel) && (cnt[ia] += 1)
            ia < n && C[ib, ia + 1] != C[ib, ia] &&
                (cp || lev[(ib - 1) ÷ BLOCK + 1, ia ÷ BLOCK + 1] < maxlevel) && (cnt[ia] += 1)
        end
    end
    return sum(cnt)
end

# ---------------------------------------------------------------- convergence
"class counts of plane P"
function class_counts(P)
    c = zeros(Int, 5)
    for v in P
        0 <= v <= 4 && (c[v + 1] += 1)
    end
    return c
end

"""
The boundary of plane P: the pixel edges between 4-neighbour pixels of different class, each
as (α, β) of its midpoint and its direction (0 along α, 1 along β), and per class the
number of its edges with other classes.
"""
function edges(P::Matrix{Int8})
    n = size(P, 1)
    px = FOV / n
    c(i) = -LIM + (i - 0.5) * px
    segs = [SVector{3,Float64}[] for _ in 1:n]
    per = [zeros(Int, 5) for _ in 1:n]
    Threads.@threads for ia in 1:n
        v = segs[ia]; e = per[ia]
        @inbounds for ib in 1:n
            p = P[ib, ia]
            if ib < n
                q = P[ib + 1, ia]
                if p != q
                    push!(v, SVector(c(ia), -LIM + ib * px, 0.0))
                    e[p + 1] += 1; e[q + 1] += 1
                end
            end
            if ia < n
                q = P[ib, ia + 1]
                if p != q
                    push!(v, SVector(-LIM + ia * px, c(ib), 1.0))
                    e[p + 1] += 1; e[q + 1] += 1
                end
            end
        end
    end
    return reduce(vcat, segs), reduce(+, per)
end

"pixel edges of half-length hl sorted into square buckets of side h by midpoint; bucket k holds off[k]:off[k+1]-1"
struct Buckets
    segs::Vector{SVector{3,Float64}}
    off::Vector{Int}
    nk::Int
    h::Float64
    hl::Float64
end
bucket_xy(nk, h, p) = (clamp(floor(Int, (p[1] + LIM) / h), 0, nk - 1),
                       clamp(floor(Int, (p[2] + LIM) / h), 0, nk - 1))
function Buckets(segs::Vector{SVector{3,Float64}}, h, hl)
    nk = ceil(Int, FOV / h)
    key = [(xy = bucket_xy(nk, h, p); xy[1] * nk + xy[2] + 1) for p in segs]
    off = zeros(Int, nk * nk + 1)
    for k in key
        off[k + 1] += 1
    end
    off[1] = 1
    for k in 1:nk * nk
        off[k + 1] += off[k]
    end
    return Buckets(segs[sortperm(key)], off, nk, h, hl)
end

"distance from p to the nearest edge of B, and whether that edge lies within RMAX buckets"
function nearest(B::Buckets, p)
    bx, by = bucket_xy(B.nk, B.h, p)
    best = Inf
    for r in 0:RMAX
        for ix in bx-r:bx+r, iy in by-r:by+r
            max(abs(ix - bx), abs(iy - by)) == r || continue
            (0 <= ix < B.nk && 0 <= iy < B.nk) || continue
            k = ix * B.nk + iy + 1
            @inbounds for q in B.off[k]:B.off[k + 1] - 1
                s = B.segs[q]
                da = abs(p[1] - s[1]); db = abs(p[2] - s[2])
                s[3] == 0.0 ? (da = max(da - B.hl, 0.0)) : (db = max(db - B.hl, 0.0))
                d = hypot(da, db)
                d < best && (best = d)
            end
        end
        best <= r * B.h - B.hl && return best, true   # every edge of a farther ring is farther
    end
    return best, false
end

function distances(B::Buckets, pts)
    d = Vector{Float64}(undef, length(pts))
    found = Vector{Bool}(undef, length(pts))
    Threads.@threads for k in eachindex(pts)
        d[k], found[k] = nearest(B, pts[k])
    end
    return d, found
end

dist_stats(d, found, unit) = isempty(d) ? Dict{String,Any}("n" => 0) : Dict{String,Any}(
    "n" => length(d), "median" => median(d) / unit, "p99" => quantile(d, 0.99) / unit,
    "p999" => quantile(d, 0.999) / unit, "max" => maximum(d) / unit,
    "beyond_one_px" => count(>(unit), d), "beyond_search" => count(!, found),
    "search_radius_px" => RMAX * 4)

"the convergence record of the coarse plane Pc (level L − 1) against the fine plane Pf (L)"
function convergence(Pc::Matrix{Int8}, Pf::Matrix{Int8})
    nc = size(Pc, 1); nf = size(Pf, 1)
    nf == 2nc || error("planes $nc and $nf are not one level apart")
    pc = FOV / nc; pf = FOV / nf
    cc = class_counts(Pc); cf = class_counts(Pf)
    ec, lc = edges(Pc)
    ef, lf = edges(Pf)
    # pixels whose class changes: the fine plane against the coarse plane at fine pixels
    changed = [SVector{2,Float64}[] for _ in 1:nf]
    Threads.@threads for ia in 1:nf
        @inbounds for ib in 1:nf
            Pf[ib, ia] != Pc[(ib + 1) ÷ 2, (ia + 1) ÷ 2] &&
                push!(changed[ia], SVector(-LIM + (ia - 0.5) * pf, -LIM + (ib - 0.5) * pf))
        end
    end
    changed = reduce(vcat, changed)
    Bc = Buckets(ec, 4pc, pc / 2)
    Bf = Buckets(ef, 4pc, pf / 2)
    dch, fch = distances(Bc, changed)
    dfc, ffc = distances(Bc, ef)
    dcf, fcf = distances(Bf, ec)
    area = Dict{String,Any}[]
    for q in 0:4
        dA = cf[q + 1] / 4 - cc[q + 1]            # in coarse pixels
        Lb = lf[q + 1] * pf                        # boundary length of class q on the fine plane
        push!(area, Dict{String,Any}("class" => q, "area_coarse" => cc[q + 1] * pc^2,
                                     "area_fine" => cf[q + 1] * pf^2, "change_coarse_px" => dA,
                                     "relative_change" => cc[q + 1] > 0 ? dA / cc[q + 1] : 0.0,
                                     "boundary_length" => Lb,
                                     "mean_shift_coarse_px" => Lb > 0 ? dA * pc / Lb : 0.0))
    end
    return Dict{String,Any}("coarse_px" => pc, "fine_px" => pf, "classes" => area,
                            "changed_fine_px" => length(changed),
                            "changed_area_coarse_px" => length(changed) / 4,
                            "boundary_edges_coarse" => length(ec), "boundary_edges_fine" => length(ef),
                            "changed_to_coarse_boundary" => dist_stats(dch, fch, pc),
                            "fine_to_coarse_boundary" => dist_stats(dfc, ffc, pc),
                            "coarse_to_fine_boundary" => dist_stats(dcf, fcf, pc))
end

# ---------------------------------------------------------------- output
json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(x::Tuple) = json_str(collect(x))
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"
save_record(name, rec) = open(io -> println(io, json_str(rec)), joinpath(OUT, "$name.json"), "w")

"an Int8 matrix as a Fortran-ordered .npy (no transposed copy)"
function write_npy_i8(path, a::Matrix{Int8})
    header = "{'descr': '|i1', 'fortran_order': True, 'shape': ($(size(a, 1)), $(size(a, 2))), }"
    pad = 64 - mod(6 + 2 + 2 + length(header) + 1, 64)
    header = header * repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93], "NUMPY", UInt8[0x01, 0x00], UInt16(length(header)), header)
        write(io, a)
    end
    return path
end

"a C-ordered little-endian Float64 .npy matrix (BlackLightPlus.Data.write_npy) as [row, column]"
function read_npy_f8(path)
    open(path) do io
        read(io, 8)
        hdr = String(read(io, Int(read(io, UInt16))))
        (occursin("'<f8'", hdr) && occursin("False", hdr)) || error("$path: not a C-ordered <f8 array")
        r, c = parse.(Int, match(r"\((\d+), (\d+)\)", hdr).captures)
        a = Vector{Float64}(undef, r * c)
        read!(io, a)
        return permutedims(reshape(a, c, r))
    end
end

common_record() = Dict{String,Any}(
    "masses" => collect(MASSES), "separation" => SEP, "omega" => OMEGA, "period" => PERIOD,
    "hole_speed" => OMEGA * SEP / 2, "r_h" => [horizon_radius(moving(orbit()), A) for A in 1:2],
    "phase" => OMEGA * DCAM, "inc_deg" => INC, "dcam" => DCAM, "lim" => LIM, "fov" => FOV,
    "t_obs" => T_OBS, "lmax" => LMAX, "resc" => RESC, "reltol" => RELTOL, "abstol" => ABSTOL,
    "bend_cut_deg" => BEND_CUT, "block" => BLOCK,
    "classes" => ["escaped, bend <= 360 deg", "captured by hole 1", "captured by hole 2",
                  "escaped, bend > 360 deg", "undetermined"],
    "code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()))

# ---------------------------------------------------------------- parts
function run_check()
    logf("check: base %d, max level %d, plain rule (the maps settings of fig_dynrays.jl)\n",
         CHECK_BASE, CHECK_MAXLEVEL)
    t0 = time()
    C, _, _, stats, rl = amr_frame(moving(orbit()), T_OBS, CHECK_BASE, CHECK_MAXLEVEL; retest = false)
    ref = Int8.(read_npy_f8(joinpath(DYN, "map_moving_1.npy")))
    size(ref) == size(C) || error("map_moving_1.npy is $(size(ref)), the check plane $(size(C))")
    diff = count(ref .!= C)
    rec = merge(common_record(), Dict{String,Any}(
        "base" => CHECK_BASE, "maxlevel" => CHECK_MAXLEVEL, "retest" => false, "passes" => stats,
        "rays_by_level" => rl, "reference" => joinpath(DYN, "map_moving_1.npy"),
        "pixels" => length(C), "pixels_differing" => diff, "classes" => class_counts(C),
        "classes_reference" => class_counts(ref), "wall_s" => time() - t0))
    save_record("records_check", rec)
    logf("check: %d of %d pixels differ from fig_dynrays panel (d)  %.0f s\n", diff, length(C), time() - t0)
end

function run_map()
    o = orbit()
    st = moving(o)
    eff = BASE * 2^MAXLEVEL
    rec = merge(common_record(), Dict{String,Any}("base" => BASE, "maxlevel" => MAXLEVEL, "eff" => eff,
                                                  "finest_px" => FOV / eff, "runs" => Dict{String,Any}[]))
    planes = Dict{Int,Matrix{Int8}}()
    for (L, retest) in ((MAXLEVEL, true), (MAXLEVEL - 1, true), (MAXLEVEL - 2, true),
                        (PANEL_D_LEVEL, true), (PANEL_D_LEVEL, false))
        n = BASE * 2^L
        logf("map: base %d, block %d, max level %d, %d² effective, %s\n", BASE, BLOCK, L, n,
             retest ? "ring test" : "plain rule")
        t0 = time()
        C, lev, S0, stats, rl = amr_frame(st, T_OBS, BASE, L; retest = retest)
        wall = time() - t0
        rays = sum(rl)
        ce = coarse_boundary_edges(C, lev, L)
        push!(rec["runs"], Dict{String,Any}("maxlevel" => L, "retest" => retest, "eff" => n,
                                            "passes" => stats, "rays_by_level" => rl, "rays" => rays,
                                            "uniform_fraction" => rays / n^2, "wall_s" => wall,
                                            "classes" => class_counts(C), "coarse_boundary_edges" => ce))
        logf("map: max level %d: %d rays (by level %s), %.3e of a uniform %d² frame, %.0f s; boundary edges on coarser blocks %d\n",
             L, rays, string(rl), rays / n^2, n, wall, ce)
        if L == MAXLEVEL
            tc = [s[3] for s in S0 if s[1] == 0.0 && isfinite(s[3])]
            t_pass = median(tc)
            merge!(rec, Dict{String,Any}(
                "rays" => rays, "rays_by_level" => rl, "uniform_fraction" => rays / n^2, "wall_s" => wall,
                "coarse_boundary_edges" => ce, "classes" => class_counts(C), "t_pass" => t_pass,
                "t_pass_quartiles" => [quantile(tc, 0.25), quantile(tc, 0.75)],
                "holes_t_pass" => holes_at(o, t_pass), "holes_image_t_pass" => [img(s) for s in holes_at(o, t_pass)],
                "holes_t_obs" => holes_at(o, T_OBS), "holes_image_t_obs" => [img(s) for s in holes_at(o, T_OBS)]))
            write_npy_i8(joinpath(OUT, "level.npy"), lev)
            logf("map: t_pass %.3f\n", t_pass)
        end
        ref = L == PANEL_D_LEVEL ? Int8.(read_npy_f8(joinpath(DYN, "map_moving_1.npy"))) : C
        if size(ref) == size(C) && ref !== C       # panel (d) of fig_dynrays: the same 1024² grid
            cmp = compare_planes(ref, C)
            rec[retest ? "panel_d" : "panel_d_plain"] = cmp
            logf("panel (d): %d of %d pixels differ\n", cmp["pixels_differing"], length(ref))
        end
        retest && (write_npy_i8(joinpath(OUT, "class_L$L.npy"), C); planes[L] = C)
        save_record("records", rec)
    end
    rec["convergence"] = Dict{String,Any}[]
    for L in (MAXLEVEL - 1, MAXLEVEL)
        t1 = time()
        cv = convergence(planes[L - 1], planes[L])
        cv["levels"] = [L - 1, L]
        cv["wall_s"] = time() - t1
        push!(rec["convergence"], cv)
        logf("convergence %d -> %d: changed %d fine px; area change (coarse px) %s; mean shift (coarse px) %s\n",
             L - 1, L, cv["changed_fine_px"], string([round(c["change_coarse_px"]; digits = 2) for c in cv["classes"]]),
             string([round(c["mean_shift_coarse_px"]; sigdigits = 3) for c in cv["classes"]]))
        for k in ("changed_to_coarse_boundary", "fine_to_coarse_boundary", "coarse_to_fine_boundary")
            s = cv[k]
            s["n"] == 0 && continue
            logf("   %-27s n %8d  median %.3f  p99 %.3f  p99.9 %.3f  max %.3f  > 1 px %d  beyond search %d\n",
                 k, s["n"], s["median"], s["p99"], s["p999"], s["max"], s["beyond_one_px"], s["beyond_search"])
        end
        save_record("records", rec)
    end
end

"pixels of two class planes of one grid that differ, and their distance to the boundary of `ref`"
function compare_planes(ref::Matrix{Int8}, P::Matrix{Int8})
    n = size(ref, 1)
    px = FOV / n
    d = SVector{2,Float64}[]
    for ia in 1:n, ib in 1:n
        ref[ib, ia] != P[ib, ia] && push!(d, SVector(-LIM + (ia - 0.5) * px, -LIM + (ib - 0.5) * px))
    end
    e, _ = edges(ref)
    dist, found = distances(Buckets(e, 4px, px / 2), d)
    return Dict{String,Any}("pixels_differing" => length(d), "fraction" => length(d) / n^2,
                            "classes" => class_counts(P), "classes_reference" => class_counts(ref),
                            "to_reference_boundary" => dist_stats(dist, found, px))
end

logf("fig_capture_dyn: parts %s, %d threads, code %s\n", join(PARTS, ","), Threads.nthreads(), GIT_SHA)
"check" in PARTS && run_check()
"map" in PARTS && run_map()
logf("wrote %s\n", OUT)
