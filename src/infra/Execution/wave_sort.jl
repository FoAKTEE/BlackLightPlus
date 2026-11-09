# The coherence sort of the wavefront (gated): a key stage writes one key per member of the
# cohort and counts the members of every key; the host turns the counts into the slot before
# the first member of each key; a placement launch writes the order in which the advance
# reads the cohort. The sort moves indices, never states, and no result depends on the order.

"Advances of a device between two sorts of its cohort, by default: every advance."
const WAVE_SORT = 1

"""
    KeyCount(A, keys, counts)
    KeyCount()

The queues of the key stage of a wavefront's sort in one launch: the cohort `A` (a
[`Cohort`](@ref)) whose members the stage keys, the key of each member `keys` (`UInt32`,
entry `j` for member `j`) and the members of each key value `counts` (`Int32`, entry
`k + 1` for key `k`). `KeyCount()` is the placeholder a key stage holds; a sort binds the
launch's queues in its place through [`WaveSwap`](@ref).
"""
struct KeyCount{A,K,C}
    A::A
    keys::K
    counts::C
end
KeyCount() = KeyCount(nothing, nothing, nothing)
Adapt.@adapt_structure KeyCount
Adapt.adapt_structure(to::WaveSwap, ::KeyCount) = to.queues

"""
    count_key!(q::KeyCount, j, k) -> nothing

Store the key `k` (in `0:wave_keys(prog) - 1`) of member `j` of the cohort and count it.
Atomic; device-safe.
"""
@inline function count_key!(q::KeyCount, j::Integer, k::Integer)
    @inbounds q.keys[j] = k % UInt32
    Atomix.@atomic q.counts[k + 1] += Int32(1)
    return nothing
end

"""
    KeyPlace(keys, cursors, order)

The placement program of a wavefront's sort, called as `prog(item, i)`: member `j = item`
of key `k = keys[j]` takes the next slot of its key from `cursors[k + 1]` (one `Int32`
atomic, starting from the slot before the key's first) and writes `order[slot] = j`. The
order of the members of one key is the order of their atomics.
"""
struct KeyPlace{K,C,O} <: Function
    keys::K
    cursors::C
    order::O
end
Adapt.@adapt_structure KeyPlace

function (p::KeyPlace)(item, i)
    k = @inbounds p.keys[item]
    s = Atomix.@atomic p.cursors[k + 1] += Int32(1)
    @inbounds p.order[s] = item % Int32
    return nothing
end

"""
    key_cursors!(counts) -> counts

The exclusive prefix sum of `counts` in place: entry `k + 1` becomes the members of the
keys below `k`, the slot before the first member of key `k`.
"""
function key_cursors!(c::AbstractVector{Int32})
    s = Int32(0)
    for k in eachindex(c)
        n = c[k]
        c[k] = s
        s += n
    end
    return c
end

"""
    wave_keys(prog) -> Int

The number of key values of the key stage `prog` of a [`Wavefront`](@ref): its keys lie in
`0:wave_keys(prog) - 1`. A key stage program defines it.
"""
wave_keys(prog) = throw(ArgumentError("$(nameof(typeof(prog))) is not the key stage of a wavefront"))

# Whether the iteration that advances a cohort for the `m`-th time (from 0) sorts it. A
# cohort of one state is sorted too: at `sort_every = 1` every advance of a call reads its
# cohort through an order, so the call compiles one advance kernel.
sorts(key, every::Int, m::Int) = key !== nothing && every > 0 && m % every == 0

# The order of the `nA` members of `states` on `lane` by the keys of the key stage `key`
# (device form `dkey`), in the lane buffer `(:wave, :order)`: the counts zeroed by a copy
# from a pinned stage of zeros, the key launch, one copy of the counts to the host (the
# host waits for it), the scan, one copy of the cursors back, the placement launch. The
# launches wait for the device unless `wait = false`; the key stage runs in workgroups of
# `wg`. The host writes the zero stage only when it is new and the count stage only after
# the wait, so no queued copy reads a stage the host writes.
function sort_cohort!(lane::PoolLane, key, dkey, states, nA::Int; wait::Bool = true,
                      wg = workgroupsize(key))
    nk = wave_keys(key)
    keys = lane_buffer(lane, (:wave, :keys), UInt32, nA)
    counts = lane_buffer(lane, (:wave, :key_counts), Int32, nk)
    order = lane_buffer(lane, (:wave, :order), Int32, nA)
    zs = lane_stage(lane, (:wave, :key_zeros), Int32, nk)
    st = lane_stage(lane, (:wave, :key_counts), Int32, nk)
    all(iszero, view(zs, 1:nk)) || fill!(view(zs, 1:nk), Int32(0))
    copyto!(counts, 1, zs, 1, nk)
    q = KeyCount(Cohort(states, nothing), keys, counts)
    launch!(lane, key, Adapt.adapt(WaveSwap(q, nothing), dkey), 1:nA, nA, (); wait, wg)
    copyto!(st, 1, counts, 1, nk)
    sum(view(st, 1:nk)) == nA ||
        error("wavefront: the key stage counted $(sum(view(st, 1:nk))) of $nA members on $(lane.dev)")
    key_cursors!(view(st, 1:nk))
    copyto!(counts, 1, st, 1, nk)
    place = KeyPlace(keys, counts, order)
    launch!(lane, place, place, 1:nA, nA, (); wait)
    return order
end
