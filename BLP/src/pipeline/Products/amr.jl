# Adaptive image-plane refinement: base blocks of the camera are marched, each marched
# block is painted into a mosaic at the finest resolution and tested, and a block that
# needs refinement is replaced by four children at twice its pixel density, down to
# `max_level`. Blocks stream through the executor's worklist without level barriers.
# Reference: White 2022, ApJS 262, 28, adaptive ray tracing.

# The mosaic of one adaptive render: the four Stokes planes at the finest resolution
# `eff × eff` and, per level `l`, which blocks of the level grid were marched.
struct Mosaic
    planes::NTuple{4,Matrix{Float64}}
    visited::Vector{Matrix{Bool}}
    block::Int
    max_level::Int
    fov::Float64
    eff::Int
end

function Mosaic(npix::Int, block::Int, max_level::Int, fov::Float64)
    eff = npix * 2^max_level
    nb = npix ÷ block
    return Mosaic(ntuple(_ -> Matrix{Float64}(undef, eff, eff), 4),
                  [fill(false, nb * 2^l, nb * 2^l) for l in 0:max_level], block, max_level, fov,
                  eff)
end

@inline stokes_component(v::Real, c) = c == 1 ? Float64(v) : 0.0
@inline stokes_component(v::SVector{4}, c) = @inbounds v[c]

# Paint block pixel (i, j) = vals[(i - 1) block + j] over its `cell × cell` canvas cells;
# the block's corner maps to the canvas offsets (ca, cb). Blocks visited concurrently
# never overlap: a child is queued only after its parent's visit returns.
function paint_block!(m::Mosaic, blk::ImageBlock, vals)
    px = m.fov / m.eff
    ca = round(Int, (blk.a0 + m.fov / 2) / px)
    cb = round(Int, (blk.b0 + m.fov / 2) / px)
    cell = 2^(m.max_level - blk.level)
    bs = m.block
    @inbounds for comp in 1:4
        P = m.planes[comp]
        for i in 1:bs, di in 1:cell
            c = ca + (i - 1) * cell + di
            for j in 1:bs
                v = stokes_component(vals[(i - 1) * bs + j], comp)
                for dj in 1:cell
                    P[cb + (j - 1) * cell + dj, c] = v
                end
            end
        end
    end
    return nothing
end

# Mark the block in its level grid; blocks of one level have distinct grid cells.
function mark_visited!(m::Mosaic, blk::ImageBlock)
    ia = round(Int, (blk.a0 + m.fov / 2) / blk.span) + 1
    ib = round(Int, (blk.b0 + m.fov / 2) / blk.span) + 1
    @inbounds m.visited[blk.level + 1][ib, ia] = true
    return nothing
end

# The finest level reached in each base block, `[β block, α block]`.
function level_map(m::Mosaic)
    nb = size(m.visited[1], 1)
    levels = zeros(Int, nb, nb)
    for l in 1:m.max_level, ia in axes(m.visited[l + 1], 2), ib in axes(m.visited[l + 1], 1)
        m.visited[l + 1][ib, ia] || continue
        bi = (ia - 1) ÷ 2^l + 1
        bj = (ib - 1) ÷ 2^l + 1
        levels[bj, bi] = max(levels[bj, bi], l)
    end
    return levels
end

# The visit of a marched block: paint, mark, test, and return its children.
struct MosaicVisit{R}
    mosaic::Mosaic
    refine::R
end

function (v::MosaicVisit)(blk::ImageBlock, vals)
    m = v.mosaic
    paint_block!(m, blk, vals)
    mark_visited!(m, blk)
    blk.level < m.max_level || return ImageBlock[]
    bs = m.block
    I = [stokes_component(vals[(i - 1) * bs + j], 1) for i in 1:bs, j in 1:bs]
    needs_refinement(v.refine, I) || return ImageBlock[]
    half = blk.span / 2
    return [ImageBlock(blk.bi, blk.bj, blk.level + 1, blk.a0 + da, blk.b0 + db, half)
            for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))]
end

"""
    render_amr(plan; block = 16, max_level = 2, refine = refinement_indicator(),
               backend = nothing) -> NamedTuple

Adaptive-refinement image of `plan`: the camera frame is tiled into `block × block`
pixel blocks (the resolution must be a multiple of `block`), every block is marched
with the plan's imaging program on its geodesic lane ([`ImagingProgram`](@ref) over a
`BlockGrid`), and a block whose Stokes I values `I[i, j]` (pixel `(i, j)` at
`(a0 + (i - ½) ps, b0 + (j - ½) ps)`) satisfy `needs_refinement(refine, I)` is replaced
by four children of half its side at the same pixel count, down to level `max_level`.
`refine` is a refinement indicator of `Analysis` or any callable `f(I) -> Bool`. Blocks
stream through `stream_blocks!` on the executor `resolve(backend)`: children are
marched as soon as their parent is tested. Returns `(image, levels, nrays)`: `image` an
[`Image`](@ref) mosaic at `resolution × 2^max_level` per side (each pixel the value of
the finest block covering it), `levels[β block, α block]` the finest level reached in
each base block, `nrays` the rays marched. `MixedLanes` plans are an `ArgumentError`.

Determinism: bitwise per item; the visited blocks, the mosaic, the level map and the ray
count are independent of the executor, the chunking and the completion order.
"""
function render_amr(plan::Plan; block::Integer = 16, max_level::Integer = 2,
                    refine = refinement_indicator(), backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    plan.lanes isa MixedLanes &&
        throw(ArgumentError("render_amr: MixedLanes rescues pixels of the camera grid; " *
                            "prepare the plan with Lanes()"))
    cam = plan.camera
    bs = Int(block)
    cam.npix % bs == 0 ||
        throw(ArgumentError("render_amr: resolution $(cam.npix) is not divisible by block " *
                            "($bs); the trailing pixel strip would never be rendered"))
    m = Mosaic(cam.npix, bs, Int(max_level), cam.fov)
    stream_blocks!(exec, imaging_program(plan; gen = BlockGrid(bs)), base_blocks(cam, bs), bs,
                   MosaicVisit(m, refine); outtype = output_type(plan))
    image = Image(m.planes..., plan, run_record(exec, plan, t0))
    return (image = image, levels = level_map(m), nrays = sum(count, m.visited) * bs * bs)
end

# The level-0 blocks tiling the camera frame.
function base_blocks(cam, bs::Int)
    nb = cam.npix ÷ bs
    px0 = cam.fov / cam.npix
    return vec([ImageBlock(bi, bj, 0, -cam.fov / 2 + (bi - 1) * bs * px0,
                           -cam.fov / 2 + (bj - 1) * bs * px0, bs * px0)
                for bi in 1:nb, bj in 1:nb])
end
