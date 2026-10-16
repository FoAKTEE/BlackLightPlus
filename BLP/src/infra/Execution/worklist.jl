# The image-block worklist of adaptive refinement: every block is marched (one program
# item per block pixel), visited, and the children its visit returns are queued at once,
# so refined blocks run while the rest of their parent level is still in flight. A
# block's pixel values depend only on the block, and its children only on the block and
# its values, so the visited set and every block's values are independent of the
# executor, the device count and the completion order.

"""
    ImageBlock(bi, bj, level, a0, b0, span)

A square block of the image plane in adaptive refinement: base-block indices `(bi, bj)`,
refinement `level` (0 for base blocks), lower corner `(a0, b0)` and side `span`, in the
image-plane units of the camera.
"""
struct ImageBlock
    bi::Int
    bj::Int
    level::Int
    a0::Float64
    b0::Float64
    span::Float64
end

# Block geometry of a chunk as `BlockGrid` reads it: (a0, b0, pixel size) per block.
block_geometry(blks, block::Integer) = [SVector(b.a0, b.b0, b.span / block) for b in blks]

# Adaptor that sets the blocks of every `BlockGrid` inside a program.
struct BlockSwap{V}
    blocks::V
end
Adapt.adapt_structure(to::BlockSwap, g::BlockGrid) = BlockGrid(to.blocks, g.bs)

# `prog` with the blocks of its `BlockGrid`s replaced by `blocks`.
with_blocks(prog, blocks) = Adapt.adapt(BlockSwap(blocks), prog)

block_chunks(blks, per::Integer) = [blks[r] for r in chunk_ranges(length(blks), per)]

"""
    stream_blocks!(exec, prog, work, block, visit; outtype = Float64)

Run the refinement worklist: march every [`ImageBlock`](@ref) of `work` and of the
children that visits return, until none remain. A chunk of blocks (at most
`max(1, chunk ÷ block²)` of them) is one run of the program `prog` over the chunk's
`block × block` pixels, `prog` seeing its [`BlockGrid`](@ref) filled with the chunk's
blocks (so `prog` must reach that grid through `Adapt.adapt_structure`). Each item writes
one value of type `outtype`. Then `visit(blk, vals)` is called for each block of the
chunk, `vals[(i - 1) * block + j]` holding pixel `(i, j)` at
`(a0 + (i - 0.5) ps, b0 + (j - 0.5) ps)`, `ps = span / block`; it returns the block's
children (any iterable of `ImageBlock`), which are queued at once. `visit` may run
concurrently for different blocks, may keep `vals`, and must not start executor work.

Determinism: the visited blocks and each block's values are bitwise independent of the
executor, the chunking, the device count and the completion order.
"""
function stream_blocks!(exec::Executor, prog, work::AbstractVector{ImageBlock},
                        block::Integer, visit; outtype::Type = Float64)
    bb = block * block
    per = max(1, chunk_size(exec) ÷ bb)
    stream_run!(exec, prog, block_chunks(work, per)) do worker, ctx, blks, job
        vals = block_values(exec, worker, ctx, blks, block, outtype)
        kids = ImageBlock[]
        for (w, b) in enumerate(blks)
            append!(kids, visit(b, view(vals, (w - 1) * bb + 1:w * bb)))
        end
        foreach(c -> enqueue!(job, c), block_chunks(kids, per))
    end
    return nothing
end

# Thread workers pull block chunks; each chunk's pixels run through the threaded executor.
stream_run!(f, exec::ThreadedExecutor, prog, chunks) =
    run_workers!((w, blks, job) -> f(w, prog, blks, job), Threads.nthreads(), chunks)

function block_values(exec::ThreadedExecutor, _, prog, blks, block, ::Type{T}) where {T}
    vals = Vector{T}(undef, length(blks) * block * block)
    foreach_item!(exec, with_blocks(prog, block_geometry(blks, block)), eachindex(vals);
                  outs = (vals,))
    return vals
end

# Pool lanes pull block chunks; each chunk is one launch with its geometry staged in.
function stream_run!(f, exec::PoolExecutor, prog, chunks)
    twins = ProgramTwins(exec.pool, prog)
    stream_chunks((lane, blks, job) -> f(lane, twins, blks, job), exec.pool, chunks)
end

function block_values(::PoolExecutor, lane::PoolLane, twins::ProgramTwins, blks, block,
                      ::Type{T}) where {T}
    geom = block_geometry(blks, block)
    m = length(geom)
    st = lane_stage(lane, :blocks, eltype(geom), m)
    copyto!(st, 1, geom, 1, m)
    gd = lane_buffer(lane, :blocks, eltype(geom), m)
    copyto!(gd, 1, st, 1, m)
    vals = Vector{T}(undef, m * block * block)
    run_chunk!(lane, twins.prog, with_blocks(program_twin!(twins, lane), gd),
               eachindex(vals), (vals,))
    return vals
end
