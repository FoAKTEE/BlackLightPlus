# fig_z4c_amr.jl — image-plane adaptive refinement of the ray class for the AthenaK Z4c
# binary-black-hole slice at ι = 30°, with the spacetime, rays, integrator, capture and
# escape criteria of fig_z4c_phasespace.jl. Classes: 0 escaped with total bending ≤ 360°,
# 1 captured (lapse < 0.55), 2 escaped after bending > 360°, 3 neither (affine budget
# exhausted). Bending: summed turning angle of the saved coordinate path.
#
# Refinement: the image box |a|, |b| ≤ 7.5 M is tiled into BLOCK × BLOCK pixel blocks of a
# BASE² grid; a block is replaced by four children at twice its pixel density, down to
# MAXLEVEL, when the classes of its pixels and of the one-pixel border around it are not
# all equal. A border pixel is read from the same level where that level traced it, else
# from the finest coarser level covering it, so a class change on a block edge refines
# both neighbours. Each level is one run of `stream_blocks!` on the threaded executor with
# a per-pixel classification program over the level's blocks; the border needs the whole
# level, so the children are queued between levels. Blocks are painted into a mosaic at
# BASE · 2^MAXLEVEL per side and the level map comes from the per-level visited masks.
#
# Resumable: every chunk of CHUNK rays of a level is cached in OUT/cache/L<l>_c<k>.npy
# (rows a, b, class, bend, fate) when its last block finishes; a rerun loads the chunks
# whose coordinates match and traces the rest.
#
# Writes figures/paper/out/z4c_amr/: class.npy and bend.npy (EFF × EFF, [i_b, i_a]),
# levels.npy (finest level per base block), levels_fine.npy (finest level per finest
# BLOCK × BLOCK tile), root_class.npy (level-0 classes, BASE × BASE), records.json (setup,
# per-level rays, wall times and classes, histograms, code SHA, threads).
#
# Run from the repository root:
#   julia -t 48 --project=. figures/paper/fig_z4c_amr.jl
# Env: BLP_Z4C (dump path), BLP_Z4C_BASE (256), BLP_Z4C_BLOCK (8), BLP_Z4C_MAXLEVEL (5),
#      BLP_Z4C_OUT (output directory).
include(joinpath(@__DIR__, "fig_z4c_phasespace.jl"))
using Adapt
using BlackLightPlus.Execution: ImageBlock, BlockGrid, stream_blocks!
using BlackLightPlus.Products: Mosaic, paint_block!, mark_visited!, level_map

const BASE = parse(Int, get(ENV, "BLP_Z4C_BASE", "256"))
const BLOCK = parse(Int, get(ENV, "BLP_Z4C_BLOCK", "8"))
const MAXLEVEL = parse(Int, get(ENV, "BLP_Z4C_MAXLEVEL", "5"))
const EFF = BASE * 2^MAXLEVEL
const FOV = 2 * LIM                              # image box |a|, |b| ≤ LIM
const AOUT = get(ENV, "BLP_Z4C_OUT", joinpath(@__DIR__, "out", "z4c_amr"))
const CACHE = joinpath(AOUT, "cache")
const CHUNK = 8192
const BEND_CUT = 360.0
# one block per worklist chunk: the threads pull blocks, each block's pixels run in order
const AMR_EXEC = ThreadedExecutor(chunk = BLOCK^2)

"class, total bending (deg) and fate code (0 escape, 1 capture, 2 other, 3 off-grid)"
function classify(st, B, a, b)
    sol = trace(st, B, a, b; save=true)
    sol === nothing && return (3.0, NaN, 3.0)
    f = fate(st, sol)
    n = length(sol.u)
    X = zeros(n, 5)
    for k in 1:n
        X[k, 3] = sol.u[k][2]; X[k, 4] = sol.u[k][3]; X[k, 5] = sol.u[k][4]
    end
    bend = cumulative_bend_deg(X)
    f == :capture && return (1.0, bend, 1.0)
    f == :escape && return (bend > BEND_CUT ? 2.0 : 0.0, bend, 0.0)
    return (3.0, bend, 2.0)
end

"""
    ClassProgram(st, basis, gen)

Per-pixel program of the worklist: `(class, bend, fate, 0)` of the ray at
`image_coords(gen, item)`. `gen` is the `BlockGrid` that `stream_blocks!` fills per chunk.
"""
struct ClassProgram{S,Bs,G} <: Function
    st::S
    basis::Bs
    gen::G
end
Adapt.@adapt_structure ClassProgram

function (p::ClassProgram)(item, i, out)
    a, b = image_coords(p.gen, item)
    c, bend, f = classify(p.st, p.basis, a, b)
    @inbounds out[i] = SVector(c, bend, f, 0.0)
    return nothing
end

function read_npy_f64(path)
    open(path) do io
        read(io, 6); ver = read(io, UInt8); read(io, UInt8)
        hl = ver == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        hdr = String(read(io, hl))
        shp = Tuple(parse.(Int, filter(!isempty, strip.(split(
            match(r"'shape':\s*\(([^)]*)\)", hdr).captures[1], ",")))))
        fortran = occursin("'fortran_order': True", hdr)
        v = Vector{Float64}(undef, prod(shp)); read!(io, v)
        length(shp) == 1 && return v
        fortran ? reshape(v, shp) : permutedims(reshape(v, reverse(shp)), reverse(1:length(shp)))
    end
end

"pixel centres of a level's blocks, block-major, pixel (i, j) with j fastest (as BlockGrid)"
function level_coords(work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    coords = Vector{Tuple{Float64,Float64}}(undef, length(work) * bb)
    for (w, blk) in enumerate(work)
        ps = blk.span / BLOCK
        for i in 1:BLOCK, j in 1:BLOCK
            coords[(w - 1) * bb + (i - 1) * BLOCK + j] =
                (blk.a0 + (i - 0.5) * ps, blk.b0 + (j - 0.5) * ps)
        end
    end
    return coords
end

"the level-0 blocks tiling the image box, a-block index fastest"
base_work() = (px0 = FOV / BASE; nb = BASE ÷ BLOCK;
               vec([ImageBlock(bi, bj, 0, -FOV / 2 + (bi - 1) * BLOCK * px0,
                               -FOV / 2 + (bj - 1) * BLOCK * px0, BLOCK * px0)
                    for bi in 1:nb, bj in 1:nb]))

"""
    trace_level(st, B, l, work) -> (S, coords, ncached)

Classify every pixel of the level-`l` blocks `work`: chunks whose cache file matches are
loaded, the remaining blocks run through `stream_blocks!`, whose visits store the values
and write each chunk's cache once its last block is in. Visits queue no children.
"""
function trace_level(st, B, l, work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    per = CHUNK ÷ bb
    n = length(work) * bb
    coords = level_coords(work)
    S = Vector{SVector{4,Float64}}(undef, n)
    nch = cld(length(work), per)
    cfile(c) = joinpath(CACHE, @sprintf("L%d_c%05d.npy", l, c))
    crange(c) = ((c - 1) * CHUNK + 1):min(c * CHUNK, n)
    pending = zeros(Int, nch)
    todo = ImageBlock[]
    ncached = 0
    for c in 1:nch
        r = crange(c)
        f = cfile(c)
        if isfile(f)
            M = read_npy_f64(f)                    # rows: a, b, class, bend, fate
            if size(M, 2) == length(r) &&
               all(M[1, k] == coords[r[k]][1] && M[2, k] == coords[r[k]][2]
                   for k in eachindex(r))
                for k in eachindex(r)
                    S[r[k]] = SVector(M[3, k], M[4, k], M[5, k], 0.0)
                end
                ncached += length(r)
                continue
            end
        end
        wr = ((c - 1) * per + 1):min(c * per, length(work))
        pending[c] = length(wr)
        append!(todo, work[wr])
    end
    function write_chunk(c)
        r = crange(c)
        M = Matrix{Float64}(undef, 5, length(r))
        for (k, q) in enumerate(r)
            M[1, k], M[2, k] = coords[q]
            M[3, k], M[4, k], M[5, k] = S[q][1], S[q][2], S[q][3]
        end
        write_npy(cfile(c) * ".tmp", M)
        mv(cfile(c) * ".tmp", cfile(c); force=true)
    end
    index = Dict(blk => w for (w, blk) in enumerate(work))
    lk = ReentrantLock()
    visit = (blk, vals) -> begin
        w = index[blk]
        for q in 1:bb
            S[(w - 1) * bb + q] = vals[q]
        end
        c = (w - 1) ÷ per + 1
        last = lock(lk) do
            pending[c] -= 1
            pending[c] == 0
        end
        last && write_chunk(c)
        return ImageBlock[]
    end
    stream_blocks!(AMR_EXEC, ClassProgram(st, B, BlockGrid(BLOCK)), todo, BLOCK, visit;
                   outtype=SVector{4,Float64})
    return S, coords, ncached
end

function main_amr()
    mkpath(CACHE)
    t0 = time()
    st, snap = z4c_grid_spacetime(BIN, HALF, NGRID; alpha_min=ALPHA_TERM)
    load_s = time() - t0
    logf("[amr] grid spacetime built in %.1fs (t=%.2f), %d threads\n", load_s, snap.time,
            Threads.nthreads())
    B = basis(IOTA)
    bb = BLOCK * BLOCK

    # per-level class lookup keyed by the level's global pixel index
    D = Dict{Tuple{Int,Int},Int8}[]
    lvl_rays = Int[]; lvl_wall = Float64[]; lvl_cached = Int[]
    lvl_class = Vector{Vector{Int}}()

    "class at level l pixel (ix, iy), else the finest coarser level that covers it"
    function cls(l, ix, iy)
        while l >= 0
            v = get(D[l+1], (ix, iy), Int8(-1))
            v >= 0 && return v
            l -= 1; ix = fld(ix, 2); iy = fld(iy, 2)
        end
        return Int8(-1)
    end
    "class change within block w of level l or its one-pixel border"
    function refine(l, blk, S, w)
        ps = blk.span / BLOCK
        n = BASE * 2^l
        ix0 = round(Int, (blk.a0 + FOV / 2) / ps)
        iy0 = round(Int, (blk.b0 + FOV / 2) / ps)
        o = (w - 1) * bb
        c0 = Int8(S[o + 1][1])
        for ix in ix0-1:ix0+BLOCK, iy in iy0-1:iy0+BLOCK
            (0 <= ix < n && 0 <= iy < n) || continue
            c = (ix0 <= ix < ix0 + BLOCK && iy0 <= iy < iy0 + BLOCK) ?
                Int8(S[o + (ix - ix0) * BLOCK + (iy - iy0) + 1][1]) : cls(l, ix, iy)
            c >= 0 && c != c0 && return true
        end
        return false
    end

    m = Mosaic(BASE, BLOCK, MAXLEVEL, FOV)
    work = base_work()
    root = zeros(BASE, BASE)
    t1 = time()
    l = 0
    while !isempty(work)
        t = time()
        S, coords, ncached = trace_level(st, B, l, work)
        # paint and mark: blocks of one level own distinct mosaic cells
        Threads.@threads for w in eachindex(work)
            paint_block!(m, work[w], view(S, (w - 1) * bb + 1:w * bb))
            mark_visited!(m, work[w])
        end
        ps = FOV / (BASE * 2^l)
        d = Dict{Tuple{Int,Int},Int8}()
        sizehint!(d, length(coords))
        for (k, (a, b)) in enumerate(coords)
            d[(floor(Int, (a + FOV / 2) / ps), floor(Int, (b + FOV / 2) / ps))] = Int8(S[k][1])
        end
        push!(D, d)
        if l == 0                                  # root class image, [i_b, i_a]
            for (w, blk) in enumerate(work), i in 1:BLOCK, j in 1:BLOCK
                root[(blk.bj - 1) * BLOCK + j, (blk.bi - 1) * BLOCK + i] =
                    S[(w - 1) * bb + (i - 1) * BLOCK + j][1]
            end
        end
        kids = ImageBlock[]
        if l < MAXLEVEL
            need = Vector{Bool}(undef, length(work))
            foreach_item!(ThreadedExecutor(chunk = 64),
                          (w, i, out) -> (out[i] = refine(l, work[w], S, w); nothing),
                          eachindex(work); outs=(need,))
            for (w, blk) in enumerate(work)
                need[w] || continue
                half = blk.span / 2
                for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))
                    push!(kids, ImageBlock(blk.bi, blk.bj, l + 1, blk.a0 + da, blk.b0 + db,
                                           half))
                end
            end
        end
        push!(lvl_rays, length(coords)); push!(lvl_wall, time() - t)
        push!(lvl_cached, ncached)
        push!(lvl_class, [count(s -> s[1] == Float64(q), S) for q in 0:3])
        logf("[amr] level %d: %d rays (%d cached) in %.0fs | classes esc/cap/bent/other %s\n",
                l, length(coords), ncached, lvl_wall[end], string(lvl_class[end]))
        flush(stdout)
        work = kids
        l += 1
    end
    amr_s = time() - t1
    nrays = sum(count, m.visited) * bb
    @assert nrays == sum(lvl_rays)
    levels = level_map(m)
    logf("[amr] %d^2 eff: %d rays in %.0fs\n", EFF, nrays, amr_s)

    # per-tile refinement level on the finest-block tile grid (EFF / BLOCK per side)
    nt = EFF ÷ BLOCK
    lf = zeros(Int, nt, nt)
    for lv in 0:MAXLEVEL
        ntb = 2^(MAXLEVEL - lv)
        V = m.visited[lv + 1]
        for ia in axes(V, 2), ib in axes(V, 1)
            V[ib, ia] && (lf[(ib - 1) * ntb + 1:ib * ntb, (ia - 1) * ntb + 1:ia * ntb] .= lv)
        end
    end
    nbb = BASE ÷ BLOCK
    tpb = nt ÷ nbb
    fine_ok = all(maximum(lf[(bj-1)*tpb+1:bj*tpb, (bi-1)*tpb+1:bi*tpb]) == levels[bj, bi]
                  for bj in 1:nbb, bi in 1:nbb)

    C = m.planes[1]
    write_npy(joinpath(AOUT, "class.npy"), C)            # [i_b, i_a]
    write_npy(joinpath(AOUT, "bend.npy"), m.planes[2])
    write_npy(joinpath(AOUT, "levels.npy"), Float64.(levels))
    write_npy(joinpath(AOUT, "levels_fine.npy"), Float64.(lf))
    write_npy(joinpath(AOUT, "root_class.npy"), root)

    pix_class = [count(==(Float64(q)), C) for q in 0:3]
    meta = Dict(
        "what" => "ray-class image-plane AMR of the AthenaK Z4c BBH frozen slice",
        "package" => "BlackLightPlus", "when" => string(now()), "git_sha" => GIT_SHA,
        "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "host_cpus" => Sys.CPU_THREADS, "host" => gethostname(), "backend" => "cpu",
        "executor" => string(AMR_EXEC),
        "dataset" => BIN, "dump_time" => snap.time, "spacetime_load_s" => load_s,
        "iota_deg" => IOTA, "d_hat" => collect(B.d), "e_a" => collect(B.ea),
        "e_b" => collect(B.eb), "half" => HALF, "ngrid" => NGRID,
        "alpha_term" => ALPHA_TERM, "escape" => ESCAPE, "lim" => LIM, "fov" => FOV,
        "integrator" => Dict("solver" => "Vern7", "reltol" => RELTOL, "abstol" => ABSTOL,
                             "lambda_max" => LAMBDA_MAX),
        "classes" => ["escaped, bend <= 360 deg", "captured",
                      "escaped, bend > 360 deg", "undetermined"],
        "bend_cut_deg" => BEND_CUT,
        "amr" => Dict("base_res" => BASE, "block" => BLOCK, "max_level" => MAXLEVEL,
                      "eff_res" => EFF, "finest_px" => FOV / EFF,
                      "indicator" => "class change within the block or its one-pixel halo",
                      "worklist" => "stream_blocks! per level, children queued between levels"),
        "nrays" => nrays, "wall_s" => amr_s,
        "rays_per_s" => nrays / amr_s,
        "per_level_rays" => lvl_rays, "per_level_wall_s" => lvl_wall,
        "per_level_cached_rays" => lvl_cached,
        "per_level_classes_esc_cap_bent_other" => lvl_class,
        "levels_hist" => [count(==(lv), levels) for lv in 0:MAXLEVEL],
        "levels_fine" => Dict("tile_grid" => nt, "tile_px" => BLOCK,
                              "tile_hist" => [count(==(lv), lf) for lv in 0:MAXLEVEL],
                              "consistent_with_base_map" => fine_ok),
        "pixel_class_counts_esc_cap_bent_other" => pix_class,
        "uniform_rays" => EFF^2, "ray_fraction_vs_uniform" => nrays / EFF^2)
    open(joinpath(AOUT, "records.json"), "w") do io
        write(io, json_str(meta))
    end
    println("Z4C-AMR DONE -> ", AOUT)
end

main_amr()
