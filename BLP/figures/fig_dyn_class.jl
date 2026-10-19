# fig_dyn_class.jl — ray-class and refinement-level maps of an orbiting binary in the
# boosted superposed Kerr-Schild metric over one orbital period: the left column of the
# paper's binary figure ((a) and (e), BLP/figures/fig_z4c_amr.jl) as an animation.
# Spacetime as fig_dyn_rays.jl: `BoostedSuperposedKerrSchild` on a `CircularOrbit`, masses
# q/(q+1) and 1/(q+1), no spin, separation SEP, prograde about +z. Camera: plane-parallel
# at inclination INC from the orbital axis, distance DCAM, image box |α|, |β| ≤ LIM; frame
# k is the observer time t_k = k T/NFRAME (T the orbital period): every ray starts at
# x⁰ = t_k and runs backward through the moving metric.
# Classes: 0 escaped with bending ≤ 360°, 1 captured by hole 1, 2 captured by hole 2,
# 3 escaped after bending > 360°, 4 neither (affine budget). Capture: r'_A < r_h,A + 1e-3
# or the redshift cut, the nearer hole; escape at R > RESC outbound; bending is the summed
# turning angle of the saved path. Refinement as fig_z4c_amr.jl: the box is tiled into
# BLOCK × BLOCK pixel blocks of a BASE² grid; a block is replaced by four children at
# twice its pixel density, down to MAXLEVEL, when the classes of its pixels and of the
# one-pixel border around it are not all equal (border pixels from the same level where
# traced, else from the finest coarser level); each level is one `stream_blocks!` run on
# the threaded executor, the children queued between levels.
# Writes BLP/figures/out/dyn_class_q<Q>/: class_<k>.npy (EFF × EFF, [i_β, i_α]),
# levels_fine_<k>.npy (finest level per finest BLOCK × BLOCK tile) and setup.json (setup,
# per-frame observer time, hole image positions at t_k − DCAM, rays, wall, classes).
# Run from the repository root:   julia -t 48 --project=BLP BLP/figures/fig_dyn_class.jl
# Env: BLP_DYN_Q (100), BLP_DYN_SEP (8), BLP_DYN_INC (60), BLP_DYN_DCAM (100),
#      BLP_DYN_LIM (12), BLP_DYN_NFRAME (36), BLP_DYN_BASE (128), BLP_DYN_BLOCK (8),
#      BLP_DYN_MAXLEVEL (3).
using Dates, LinearAlgebra, Printf, StaticArrays
using Adapt
using BLP
using BLP.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, trace_geodesic, null_covariant,
                     metric, ks_radius, horizon_radius, normal_frame_energy, CAPTURE_REDSHIFT
using BLP.Execution: ThreadedExecutor, foreach_item!, image_coords, BlockGrid, ImageBlock,
                     stream_blocks!
using BLP.Products: Mosaic, paint_block!, mark_visited!, level_map
using BLP.Data: write_npy

const Q = parse(Float64, get(ENV, "BLP_DYN_Q", "100"))
const SEP = parse(Float64, get(ENV, "BLP_DYN_SEP", "8"))
const INC = parse(Float64, get(ENV, "BLP_DYN_INC", "60"))       # degrees from the orbital axis
const DCAM = parse(Float64, get(ENV, "BLP_DYN_DCAM", "100"))
const LIM = parse(Float64, get(ENV, "BLP_DYN_LIM", "12"))
const NFRAME = parse(Int, get(ENV, "BLP_DYN_NFRAME", "36"))
const BASE = parse(Int, get(ENV, "BLP_DYN_BASE", "128"))
const BLOCK = parse(Int, get(ENV, "BLP_DYN_BLOCK", "8"))
const MAXLEVEL = parse(Int, get(ENV, "BLP_DYN_MAXLEVEL", "3"))
const EFF = BASE * 2^MAXLEVEL
const FOV = 2 * LIM
const OUT = joinpath(@__DIR__, "out", "dyn_class_q$(round(Int, Q))")
const ROOT = dirname(dirname(@__DIR__))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
const MASSES = (Q / (Q + 1), 1 / (Q + 1))
const OMEGA = sqrt(sum(MASSES) / SEP^3)
const PERIOD = 2π / OMEGA
const LMAX = 600.0
const RESC = 250.0
const RELTOL = 1e-9
const ABSTOL = 1e-11
const BEND_CUT = 360.0
const AMR_EXEC = ThreadedExecutor(chunk = BLOCK^2)

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

const ORBIT = CircularOrbit(MASSES, SEP)
const ST = BoostedSuperposedKerrSchild(collect(MASSES), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], ORBIT)
const SI = sin(deg2rad(INC)); const CI = cos(deg2rad(INC))

# the backward ray of image point (a, b) received at observer time t along the inward plane
# normal d̂ = (sin i, 0, cos i); image α along y, β along (-cos i, 0, sin i)
function ray(st, t, a, b)
    x0 = SVector(t, -b * CI + DCAM * SI, a, b * SI + DCAM * CI)
    k = metric(st, x0) * SVector(1.0, SI, 0.0, CI)
    knull, ok = null_covariant(st, x0, SVector(k[2], k[3], k[4]))
    ok || error("no positive-energy null root at the camera")
    return vcat(x0, knull / knull[1])
end

# fate of an end state: 1/2 captured by that hole, 0 escaped, 3 budget
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

# summed turning angle (degrees) of the saved coordinate path
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

"class, bending (deg) and fate of the ray of image point (a, b) at observer time t"
function classify(st, t, a, b)
    sol = trace_geodesic(st, ray(st, t, a, b); λmax = LMAX, escape = RESC, reltol = RELTOL,
                         abstol = ABSTOL, save_everystep = true)
    f = fate(st, sol.u[end])
    bend = bend_deg(sol.u)
    f in (1, 2) && return (Float64(f), bend, Float64(f))
    f == 0 && return (bend > BEND_CUT ? 3.0 : 0.0, bend, 0.0)
    return (4.0, bend, 3.0)
end

struct ClassProgram{S,G} <: Function
    st::S
    t::Float64
    gen::G
end
Adapt.@adapt_structure ClassProgram

function (p::ClassProgram)(item, i, out)
    a, b = image_coords(p.gen, item)
    c, bend, f = classify(p.st, p.t, a, b)
    @inbounds out[i] = SVector(c, bend, f, 0.0)
    return nothing
end

"the level-0 blocks tiling the image box, a-block index fastest"
base_work() = (px0 = FOV / BASE; nb = BASE ÷ BLOCK;
               vec([ImageBlock(bi, bj, 0, -FOV / 2 + (bi - 1) * BLOCK * px0,
                               -FOV / 2 + (bj - 1) * BLOCK * px0, BLOCK * px0)
                    for bi in 1:nb, bj in 1:nb]))

"pixel centres of a level's blocks, block-major, pixel (i, j) with j fastest"
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

"classify every pixel of the blocks `work` at observer time t: (values, coords)"
function trace_level(st, t, work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    S = Vector{SVector{4,Float64}}(undef, length(work) * bb)
    index = Dict(blk => w for (w, blk) in enumerate(work))
    visit = (blk, vals) -> begin
        w = index[blk]
        for q in 1:bb
            S[(w - 1) * bb + q] = vals[q]
        end
        return ImageBlock[]
    end
    stream_blocks!(AMR_EXEC, ClassProgram(st, t, BlockGrid(BLOCK)), work, BLOCK, visit;
                   outtype = SVector{4,Float64})
    return S, level_coords(work)
end

"one frame: the class plane (EFF × EFF), the finest level per finest tile, and the record"
function frame(st, t)
    bb = BLOCK * BLOCK
    D = Dict{Tuple{Int,Int},Int8}[]
    function cls(l, ix, iy)
        while l >= 0
            v = get(D[l+1], (ix, iy), Int8(-1))
            v >= 0 && return v
            l -= 1; ix = fld(ix, 2); iy = fld(iy, 2)
        end
        return Int8(-1)
    end
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
    lvl_rays = Int[]; lvl_wall = Float64[]; lvl_class = Vector{Vector{Int}}()
    l = 0
    while !isempty(work)
        t0 = time()
        S, coords = trace_level(st, t, work)
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
        kids = ImageBlock[]
        if l < MAXLEVEL
            need = Vector{Bool}(undef, length(work))
            foreach_item!(ThreadedExecutor(chunk = 64),
                          (w, i, out) -> (out[i] = refine(l, work[w], S, w); nothing),
                          eachindex(work); outs = (need,))
            for (w, blk) in enumerate(work)
                need[w] || continue
                half = blk.span / 2
                for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))
                    push!(kids, ImageBlock(blk.bi, blk.bj, l + 1, blk.a0 + da, blk.b0 + db, half))
                end
            end
        end
        push!(lvl_rays, length(coords)); push!(lvl_wall, time() - t0)
        push!(lvl_class, [count(s -> s[1] == Float64(q), S) for q in 0:4])
        work = kids
        l += 1
    end
    nt = EFF ÷ BLOCK
    lf = zeros(Int, nt, nt)
    for lv in 0:MAXLEVEL
        ntb = 2^(MAXLEVEL - lv)
        V = m.visited[lv + 1]
        for ia in axes(V, 2), ib in axes(V, 1)
            V[ib, ia] && (lf[(ib - 1) * ntb + 1:ib * ntb, (ia - 1) * ntb + 1:ia * ntb] .= lv)
        end
    end
    return m.planes[1], lf, (rays = lvl_rays, wall = lvl_wall, classes = lvl_class)
end

# image-plane position (α, β) = (y, -x cos i + z sin i) of hole A at time t
hole_image(A, t) = (s = ORBIT(t)[A][1]; (s[2], -s[1] * CI + s[3] * SI))

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(x::Tuple) = json_str(collect(x))
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"

setup = Dict{String,Any}(
    "q" => Q, "masses" => collect(MASSES), "separation" => SEP, "omega" => OMEGA,
    "period" => PERIOD, "r_h" => [horizon_radius(ST, A) for A in 1:2], "inc_deg" => INC,
    "dcam" => DCAM, "lim" => LIM, "fov" => FOV, "nframe" => NFRAME, "base" => BASE,
    "block" => BLOCK, "maxlevel" => MAXLEVEL, "eff" => EFF, "finest_px" => FOV / EFF,
    "lmax" => LMAX, "resc" => RESC, "reltol" => RELTOL, "abstol" => ABSTOL,
    "bend_cut_deg" => BEND_CUT,
    "classes" => ["escaped, bend <= 360 deg", "captured by hole 1", "captured by hole 2",
                  "escaped, bend > 360 deg", "undetermined"],
    "code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()))
times = Float64[]; holes = Vector{Vector{Float64}}(); rays = Int[]; walls = Float64[]
classes = Vector{Vector{Int}}(); levels_hist = Vector{Vector{Int}}()
logf("q = %.0f, masses %s, period %.1f M, %d frames, %d threads, eff %d^2\n", Q,
     string(MASSES), PERIOD, NFRAME, Threads.nthreads(), EFF)
for k in 0:NFRAME-1
    t = k * PERIOD / NFRAME
    t0 = time()
    C, lf, rec = frame(ST, t)
    write_npy(joinpath(OUT, @sprintf("class_%03d.npy", k)), C)
    write_npy(joinpath(OUT, @sprintf("levels_fine_%03d.npy", k)), Float64.(lf))
    push!(times, t)
    push!(holes, [hole_image(1, t - DCAM)..., hole_image(2, t - DCAM)...])
    push!(rays, sum(rec.rays)); push!(walls, time() - t0)
    push!(classes, [count(==(Float64(q)), C) for q in 0:4])
    push!(levels_hist, [count(==(lv), lf) for lv in 0:MAXLEVEL])
    logf("frame %3d  t = %7.2f  rays %7d  %.0f s  classes %s  levels %s\n", k, t, rays[end],
         walls[end], string(classes[end]), string(levels_hist[end]))
    setup["times"] = times; setup["holes_image"] = holes; setup["rays"] = rays
    setup["wall_s"] = walls; setup["pixel_classes"] = classes; setup["levels_hist"] = levels_hist
    open(io -> println(io, json_str(setup)), joinpath(OUT, "setup.json"), "w")
end
logf("wrote %s\n", OUT)
