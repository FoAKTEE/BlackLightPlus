# The refinement worklist: blocks discovered while visiting are marched as they appear;
# the visited blocks and every block's pixel values equal those of a level-synchronous
# serial oracle on the threaded executor and on pools of 1 and 3 host devices.
using Test, BLP
using StaticArrays, Adapt
using BLP.Execution: ThreadedExecutor, PoolExecutor, BlockGrid, ImageBlock, image_coords,
                     stream_blocks!
const Decks = Main.Decks

# A bright ring of radius 5.2 on a weak gradient: sharp where the ring is.
@inline field(α, β) = exp(-((hypot(α - 1.0, β) - 5.2) / 0.4)^2) + 0.01 * α

struct TileProgram{G}
    gen::G
end
Adapt.@adapt_structure TileProgram
@inline function (p::TileProgram)(item, i, out)
    α, β = image_coords(p.gen, item)
    @inbounds out[i] = field(α, β)
    return nothing
end

# Two values per pixel, to exercise a non-scalar output type.
struct PairProgram{G}
    gen::G
end
Adapt.@adapt_structure PairProgram
@inline function (p::PairProgram)(item, i, out)
    α, β = image_coords(p.gen, item)
    @inbounds out[i] = SVector(field(α, β), α * β)
    return nothing
end

const BLOCK = 4
const MAX_LEVEL = 2

# Refine where a block's values vary by more than 0.2, up to MAX_LEVEL.
function children(b::ImageBlock, vals)
    (b.level < MAX_LEVEL && maximum(first, vals) - minimum(first, vals) > 0.2) ||
        return ImageBlock[]
    h = b.span / 2
    return [ImageBlock(b.bi, b.bj, b.level + 1, b.a0 + da, b.b0 + db, h)
            for (da, db) in ((0.0, 0.0), (h, 0.0), (0.0, h), (h, h))]
end

base_blocks(fov, nb) = [ImageBlock(bi, bj, 0, -fov / 2 + (bi - 1) * fov / nb,
                                   -fov / 2 + (bj - 1) * fov / nb, fov / nb)
                        for bi in 1:nb for bj in 1:nb]

# Level-synchronous serial oracle: each block's values by direct evaluation.
function oracle(mk, work, T)
    seen = Dict{ImageBlock,Vector{T}}()
    while !isempty(work)
        next = ImageBlock[]
        for b in work
            prog = mk(BlockGrid([SVector(b.a0, b.b0, b.span / BLOCK)], BLOCK))
            vals = Vector{T}(undef, BLOCK^2)
            foreach(k -> prog(k, k, vals), 1:BLOCK^2)
            seen[b] = vals
            append!(next, children(b, vals))
        end
        work = next
    end
    return seen
end

function streamed(exec, mk, work, T)
    seen = Dict{ImageBlock,Vector{T}}()
    lk = ReentrantLock()
    function visit(b, vals)
        @lock lk begin
            haskey(seen, b) && error("visited twice")
            seen[b] = collect(vals)
        end
        return children(b, vals)
    end
    stream_blocks!(exec, mk(BlockGrid(BLOCK)), work, BLOCK, visit; outtype = T)
    return seen
end

same(a, b) = keys(a) == keys(b) && all(k -> Decks.bits(a[k]) == Decks.bits(b[k]), keys(a))

@testset "block worklist" begin
    work = base_blocks(24.0, 4)
    ref = oracle(TileProgram, work, Float64)
    @test maximum(b.level for b in keys(ref)) == MAX_LEVEL
    @test count(b -> b.level > 0, keys(ref)) > 8
    execs = (ThreadedExecutor(chunk = 3), ThreadedExecutor(chunk = 48), ThreadedExecutor(),
             PoolExecutor(Decks.hosts(1); lanes = 2, chunk = 48),
             PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 48),
             PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
    for exec in execs
        @test same(streamed(exec, TileProgram, work, Float64), ref)
    end
    refp = oracle(PairProgram, work, SVector{2,Float64})
    pool2 = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 48)
    for exec in (ThreadedExecutor(chunk = 48), pool2)
        @test same(streamed(exec, PairProgram, work, SVector{2,Float64}), refp)
    end
    template = TileProgram(BlockGrid(BLOCK))
    @test stream_blocks!(ThreadedExecutor(), template, ImageBlock[], BLOCK,
                         (b, v) -> ImageBlock[]) === nothing
    failing(b, v) = b.level == 1 ? error("visit failed") : children(b, v)
    for exec in (pool2, ThreadedExecutor(chunk = 48))
        @test_throws CapturedException stream_blocks!(exec, template, work, BLOCK, failing)
    end
end
