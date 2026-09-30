# fig_z4c_amr.jl — image-plane AMR of the ray class for the AthenaK Z4c BBH slice at
# ι = 30°: `render_amr` with a `stokes` closure that traces every image point with the
# adaptive Verner integrator of fig_z4c_phasespace.jl (same spacetime, camera, capture
# and escape criteria) and a `refine` class indicator. Classes: 0 escaped with total
# bending ≤ 360°, 1 captured (lapse < 0.55), 2 escaped after bending > 360°, 3 neither
# (step or affine budget exhausted). A block is refined when the classes of its pixels
# and of a one-pixel halo around it (read from the same level where traced, else from the
# finest coarser level covering the point) are not all equal, so class changes on block
# edges refine both neighbours.
#
# Resumable: every chunk of CHUNK rays is cached in OUT/cache/ keyed by level and chunk
# index with its coordinates; a rerun loads the cached chunks whose coordinates match.
# Writes OUT/{class.npy, bend.npy, levels.npy, levels_fine.npy, root_class.npy,
# records.json}. Run: julia --project=. -t 64 figures/fig_z4c_amr.jl
# Env: BLP_Z4C_BASE (256), BLP_Z4C_BLOCK (8), BLP_Z4C_MAXLEVEL (4), BLP_Z4C_OUT.
include(joinpath(@__DIR__, "fig_z4c_phasespace.jl"))
using Dates
using BlackLightPlus: render_amr

const BASE = parse(Int, get(ENV, "BLP_Z4C_BASE", "256"))
const BLOCK = parse(Int, get(ENV, "BLP_Z4C_BLOCK", "8"))
const MAXLEVEL = parse(Int, get(ENV, "BLP_Z4C_MAXLEVEL", "4"))
const EFF = BASE * 2^MAXLEVEL
const FOV = 2 * LIM                              # image box |α|, |β| ≤ LIM
const AOUT = get(ENV, "BLP_Z4C_OUT", joinpath(@__DIR__, "out", "z4c_amr"))
const CACHE = joinpath(AOUT, "cache")
const CHUNK = 8192
const BEND_CUT = 360.0

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

function main_amr()
    mkpath(CACHE)
    t0 = time()
    st, snap = z4c_grid_spacetime(BIN, HALF, NGRID; alpha_min=ALPHA_TERM)
    load_s = time() - t0
    @printf("[amr] grid spacetime built in %.1fs (t=%.2f), %d threads\n", load_s, snap.time,
            Threads.nthreads())
    B = basis(IOTA)

    # per-level class lookup keyed by the level's global pixel index
    D = Dict{Tuple{Int,Int},Int8}[]
    lvl_rays = Int[]; lvl_wall = Float64[]; lvl_cached = Int[]
    lvl_class = Vector{Vector{Int}}()
    lvl_coords = Vector{Vector{Tuple{Float64,Float64}}}()
    lvl_bendmax = Float64[]

    stokes = coords -> begin
        l = length(lvl_rays)
        ps = FOV / (BASE * 2^l)
        S = zeros(4, length(coords))
        t = time()
        ncached = 0
        nch = cld(length(coords), CHUNK)
        for c in 1:nch
            r = ((c - 1) * CHUNK + 1):min(c * CHUNK, length(coords))
            f = joinpath(CACHE, @sprintf("L%d_c%05d.npy", l, c))
            if isfile(f)
                M = read_npy_f64(f)             # rows: a, b, class, bend, fate
                if size(M, 2) == length(r) &&
                   all(M[1, k] == coords[r[k]][1] && M[2, k] == coords[r[k]][2]
                       for k in eachindex(r))
                    S[1:3, r] .= M[3:5, :]
                    ncached += length(r)
                    continue
                end
            end
            Threads.@threads :dynamic for k in r
                S[1, k], S[2, k], S[3, k] = classify(st, B, coords[k]...)
            end
            M = vcat(permutedims(first.(coords[r])), permutedims(last.(coords[r])),
                     S[1:3, r])
            write_npy(f * ".tmp", M)
            mv(f * ".tmp", f; force=true)
        end
        d = Dict{Tuple{Int,Int},Int8}()
        sizehint!(d, length(coords))
        for (k, (a, b)) in enumerate(coords)
            d[(floor(Int, (a + FOV / 2) / ps), floor(Int, (b + FOV / 2) / ps))] = Int8(S[1, k])
        end
        push!(D, d)
        push!(lvl_rays, length(coords)); push!(lvl_wall, time() - t)
        push!(lvl_cached, ncached)
        push!(lvl_class, [count(==(Float64(q)), @view S[1, :]) for q in 0:3])
        push!(lvl_coords, copy(coords))
        push!(lvl_bendmax, maximum(x -> isfinite(x) ? x : 0.0, @view S[2, :]))
        @printf("[amr] level %d: %d rays (%d cached) in %.0fs | classes esc/cap/bent/other %s\n",
                l, length(coords), ncached, lvl_wall[end], string(lvl_class[end]))
        flush(stdout)
        S
    end

    "class at level l pixel (ix, iy), else the finest coarser level that covers it"
    function cls(l, ix, iy)
        while l >= 0
            v = get(D[l+1], (ix, iy), Int8(-1))
            v >= 0 && return v
            l -= 1; ix = fld(ix, 2); iy = fld(iy, 2)
        end
        return Int8(-1)
    end
    refine = (blk, vals) -> begin
        _, _, l, a0, b0, span = blk
        ps = span / BLOCK
        n = BASE * 2^l
        ix0 = round(Int, (a0 + FOV / 2) / ps)
        iy0 = round(Int, (b0 + FOV / 2) / ps)
        c0 = Int8(vals[1, 1, 1])
        for ix in ix0-1:ix0+BLOCK, iy in iy0-1:iy0+BLOCK
            (0 <= ix < n && 0 <= iy < n) || continue
            c = (ix0 <= ix < ix0 + BLOCK && iy0 <= iy < iy0 + BLOCK) ?
                Int8(vals[1, ix - ix0 + 1, iy - iy0 + 1]) : cls(l, ix, iy)
            c >= 0 && c != c0 && return true
        end
        return false
    end

    cfg = RunConfig(bh_a=0.0, m_bh_cgs=1.0e43, mass_scale_cgs=1.0e24, frequency_hz=2.3e11,
                    camera=CameraConfig(inclination_deg=IOTA, distance=1000.0, fov=FOV,
                                        resolution=BASE))
    t1 = time()
    R = render_amr(cfg, BIN; block=BLOCK, max_level=MAXLEVEL, stokes, refine)
    amr_s = time() - t1
    @printf("[amr] %d^2 eff: %d rays in %.0fs\n", EFF, R.nrays, amr_s)

    # per-pixel refinement level on the finest-block tile grid (EFF / BLOCK per side)
    nt = EFF ÷ BLOCK
    ts = FOV / nt
    lf = zeros(Int, nt, nt)
    for (li, lc) in enumerate(lvl_coords)
        l = li - 1
        ps = FOV / (BASE * 2^l)
        ntb = 2^(MAXLEVEL - l)
        for k in 1:(length(lc) ÷ BLOCK^2)
            blk = @view lc[(k-1)*BLOCK^2+1:k*BLOCK^2]
            ta = round(Int, (minimum(first, blk) - ps / 2 + FOV / 2) / ts)
            tb = round(Int, (minimum(last, blk) - ps / 2 + FOV / 2) / ts)
            lf[tb+1:tb+ntb, ta+1:ta+ntb] .= l
        end
    end
    nbb = BASE ÷ BLOCK
    tpb = nt ÷ nbb
    fine_ok = all(maximum(lf[(bj-1)*tpb+1:bj*tpb, (bi-1)*tpb+1:bi*tpb]) == R.levels[bj, bi]
                  for bj in 1:nbb, bi in 1:nbb)

    # root (level-0) class image in the [iβ, iα] layout; level-0 worklist is bi fastest
    root = zeros(BASE, BASE)
    let w = 0, S0 = lvl_coords[1]
        for bj in 1:nbb, bi in 1:nbb, i in 1:BLOCK, j in 1:BLOCK
            a, b = S0[w * BLOCK^2 + (i - 1) * BLOCK + j]
            root[(bj - 1) * BLOCK + j, (bi - 1) * BLOCK + i] =
                D[1][(floor(Int, (a + FOV / 2) / (FOV / BASE)),
                      floor(Int, (b + FOV / 2) / (FOV / BASE)))]
            j == BLOCK && i == BLOCK && (w += 1)
        end
    end

    write_npy(joinpath(AOUT, "class.npy"), R.I)          # [iβ, iα]
    write_npy(joinpath(AOUT, "bend.npy"), R.Q)
    write_npy(joinpath(AOUT, "levels.npy"), Float64.(R.levels))
    write_npy(joinpath(AOUT, "levels_fine.npy"), Float64.(lf))
    write_npy(joinpath(AOUT, "root_class.npy"), root)

    pix_class = [count(==(Float64(q)), R.I) for q in 0:3]
    meta = Dict(
        "what" => "ray-class image-plane AMR of the AthenaK Z4c BBH frozen slice",
        "when" => string(now()), "git_sha" => GIT_SHA, "julia" => string(VERSION),
        "threads" => Threads.nthreads(), "host_cpus" => Sys.CPU_THREADS,
        "host" => gethostname(), "backend" => "cpu",
        "dataset" => BIN, "dump_time" => snap.time, "spacetime_load_s" => load_s,
        "iota_deg" => IOTA, "d_hat" => collect(B.d), "e_a" => collect(B.ea),
        "e_b" => collect(B.eb), "half" => HALF, "ngrid" => NGRID,
        "alpha_term" => ALPHA_TERM, "escape" => ESCAPE, "lim" => LIM, "fov" => FOV,
        "integrator" => Dict("solver" => "Vern7", "reltol" => 1e-10, "abstol" => 1e-12,
                             "lambda_max" => 6000.0),
        "classes" => ["escaped, bend <= 360 deg", "captured",
                      "escaped, bend > 360 deg", "undetermined"],
        "bend_cut_deg" => BEND_CUT,
        "amr" => Dict("base_res" => BASE, "block" => BLOCK, "max_level" => MAXLEVEL,
                      "eff_res" => EFF, "finest_px" => FOV / EFF,
                      "indicator" => "class change within the block or its one-pixel halo"),
        "nrays" => R.nrays, "wall_s" => amr_s,
        "rays_per_s" => R.nrays / amr_s,
        "per_level_rays" => lvl_rays, "per_level_wall_s" => lvl_wall,
        "per_level_cached_rays" => lvl_cached,
        "per_level_classes_esc_cap_bent_other" => lvl_class,
        "levels_hist" => [count(==(l), R.levels) for l in 0:MAXLEVEL],
        "levels_fine" => Dict("tile_grid" => nt, "tile_px" => BLOCK,
                              "tile_hist" => [count(==(l), lf) for l in 0:MAXLEVEL],
                              "consistent_with_base_map" => fine_ok),
        "pixel_class_counts_esc_cap_bent_other" => pix_class,
        "uniform_rays" => EFF^2, "ray_fraction_vs_uniform" => R.nrays / EFF^2)
    open(joinpath(AOUT, "records.json"), "w") do io
        write(io, json_str(meta))
    end
    println("Z4C-AMR DONE -> ", AOUT)
end

main_amr()
