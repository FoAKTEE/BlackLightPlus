# The device queues of the wavefront: the read view of the live states of an advance, the
# append queues of states, the child queue the branch opens, the counter block of a lane
# and the adaptor that binds a launch's queues into the device form of a stage program. A
# state is stored where its slot falls; no result depends on the order of the states in a
# queue.

"Elements of the counter block of a lane's wavefront, in this order."
const WAVE_LIVE = 1        # states the advance admitted to B
const WAVE_DONE = 2        # states the advance admitted to D
const WAVE_CHILDREN = 3    # children spawned in the advance
const WAVE_DROPPED = 4     # children beyond the depth: counted, not tracked
const WAVE_FAULT = 5       # states that broke the contract of the advance
"Elements of the counter block."
const WAVE_COUNTERS = 5

"""
    Cohort(states, order)

The read view of the live states of an advance: member `j` is `states[order[j]]`, or
`states[j]` when `order` is `nothing`. `isbits` after `Adapt.adapt` to a device.
"""
struct Cohort{S,O}
    states::S
    order::O
end
Adapt.@adapt_structure Cohort

@inline Base.getindex(c::Cohort{S,Nothing}, j::Integer) where {S} = @inbounds c.states[j]
@inline Base.getindex(c::Cohort, j::Integer) = @inbounds c.states[c.order[j]]

"""
    StateBuffer(states, count)

An append queue of states: [`admit!`](@ref) takes the next slot of `states` from the
one-element counter `count` with one `Int64` atomic; a state beyond the last slot is
counted and not stored, so the counter is the number of states admitted, which the host
compares with the slots. `q[j]` is the state in slot `j`.
"""
struct StateBuffer{S,C}
    states::S
    count::C
end
Adapt.@adapt_structure StateBuffer

@inline Base.getindex(q::StateBuffer, j::Integer) = @inbounds q.states[j]

"""
    admit!(q::StateBuffer, s) -> nothing

Append the state `s` to the queue `q`: the next slot from its counter, the state stored
there when the slot exists. Atomic; device-safe.
"""
@inline function admit!(q::StateBuffer, s)
    k = Atomix.@atomic q.count[1] += Int64(1)
    k <= length(q.states) && (@inbounds q.states[k] = s)
    return nothing
end

"""
    WaveQueues(A, B, C, D, counters, K, base, depth)
    WaveQueues()

The queues a stage program of the wavefront reads and writes in one launch on a lane: the
cohort `A` (a [`Cohort`](@ref)) that an advance runs; the append queues `B` (survivors,
opened children and started primaries, counting into element `WAVE_LIVE` of `counters`)
and `D` (stopped states, counting into `WAVE_DONE`), both [`StateBuffer`](@ref)s; the
children `C` that the advance spawned, as the [`Generation`](@ref) a branch reads; the
counter block `counters` (`WAVE_COUNTERS` `Int64`: live, done, children, dropped, fault);
the budget `K`, the loop iterations an advance runs per state at most; `base`, the slot of
`B` after which a start launch places its states ([`place!`](@ref)); and `depth`, the
last generation the call tracks ([`admit_child!`](@ref)). `WaveQueues()` is the
placeholder a stage program holds; [`WaveSwap`](@ref) replaces it at every launch.
"""
struct WaveQueues{A,B,C,D,N}
    A::A
    B::B
    C::C
    D::D
    counters::N
    K::Int
    base::Int
    depth::Int
end
WaveQueues() = WaveQueues(nothing, nothing, nothing, nothing, nothing, 0, 0, 0)
Adapt.@adapt_structure WaveQueues

"""
    place!(q::WaveQueues, i, s) -> nothing

Store the state `s` that position `i` of a start launch made in slot `q.base + i` of `q.B`.
"""
@inline function place!(q::WaveQueues, i::Integer, s)
    @inbounds q.B.states[q.base + i] = s
    return nothing
end

"""
    admit_child!(q::WaveQueues, s) -> nothing

Append the state `s` of a child that a branch opened to `q.B` when its
[`generation`](@ref) is at most `q.depth`; count it into element `WAVE_DROPPED` of the
counter block otherwise, as the cascade counts and does not track the children of its last
generation. Atomic; device-safe.
"""
@inline function admit_child!(q::WaveQueues, s)
    if generation(s) <= q.depth
        admit!(q.B, s)
    else
        Atomix.@atomic q.counters[WAVE_DROPPED] += Int64(1)
    end
    return nothing
end

"""
    fault!(q::WaveQueues) -> nothing

Count one state that broke the contract of the advance (a survivor whose loop iteration
did not grow, a stopped state without an outcome); the host fails the iteration. Atomic.
"""
@inline function fault!(q::WaveQueues)
    Atomix.@atomic q.counters[WAVE_FAULT] += Int64(1)
    return nothing
end

"""
    WaveSwap(queues, children)

The `Adapt` adaptor that binds one launch into the device form of a stage program: every
[`WaveQueues`](@ref) becomes `queues` and every [`ChildBuffer`](@ref) becomes `children`,
the child buffer of the launch; every other field stays as the device form holds it.
"""
struct WaveSwap{Q,C}
    queues::Q
    children::C
end
Adapt.adapt_structure(to::WaveSwap, ::WaveQueues) = to.queues
Adapt.adapt_structure(to::WaveSwap, b::ChildBuffer) = to.children === nothing ? b : to.children

"""
    ZeroCounters(counters)

The program that sets the counter block `counters` to zero: one item.
"""
struct ZeroCounters{C} <: Function
    counters::C
end
Adapt.@adapt_structure ZeroCounters

function (z::ZeroCounters)(item, i)
    for k in 1:WAVE_COUNTERS
        @inbounds z.counters[k] = Int64(0)
    end
    return nothing
end

# The counter block `counters` of a lane's wavefront and its two pinned host stages. A read
# queues one copy of the block on the lane's queue into the stage of its parity
# (`queue_read!`), and the host takes the values only after waiting for the queue
# (`counter_values`): the reads alternate between the stages, so a copy queued for the next
# read never targets the stage the host reads, and the host writes neither.
mutable struct CounterStages{C}
    const counters::C
    const stages::NTuple{2,Vector{Int64}}
    reads::Int
end
CounterStages(lane::PoolLane, counters) =
    CounterStages(counters, (lane_stage(lane, (:wave, :counters, 1), Int64, WAVE_COUNTERS),
                             lane_stage(lane, (:wave, :counters, 2), Int64, WAVE_COUNTERS)), 0)

# Queue the copy of the counter block of the next read; returns its stage.
function queue_read!(lane::PoolLane, cs::CounterStages)
    cs.reads += 1
    s = cs.stages[isodd(cs.reads) ? 1 : 2]
    KernelAbstractions.copyto!(lane.backend, view(s, 1:WAVE_COUNTERS),
                               view(cs.counters, 1:WAVE_COUNTERS))
    return s
end

# Wait for the lane's queue, then the counters in the stage `s`.
function counter_values(lane::PoolLane, s::Vector{Int64})
    KernelAbstractions.synchronize(lane.backend)
    return (live = s[WAVE_LIVE], done = s[WAVE_DONE], children = s[WAVE_CHILDREN],
            dropped = s[WAVE_DROPPED], fault = s[WAVE_FAULT])
end

"""
    wave_state(prog) -> Type

The type of the states the stage program `prog` of a [`Wavefront`](@ref) reads and
writes. A stage program defines it; the constructor of `Wavefront` requires one type for
all its stages.
"""
wave_state(prog) = throw(ArgumentError("$(nameof(typeof(prog))) is not a stage program of a wavefront"))

"""
    generation(state) -> Int

The scattering generation of a state of a wavefront: 0 for a primary, one more than its
parent's for a child. A stage program's state type defines it; a branch drops a child
whose generation exceeds the call's depth ([`admit_child!`](@ref)).
"""
generation(s) = throw(ArgumentError("$(nameof(typeof(s))) is not a state of a wavefront"))

"""
    wave_steps(prog) -> Int

The most loop iterations a state of the advance `prog` runs before it stops (the step
limit `M`): a state stops within `M + 1` advances of its admission, so a device whose
cohort retires nothing in `M + 2` consecutive iterations fails the call.
"""
wave_steps(prog) = throw(ArgumentError("$(nameof(typeof(prog))) is not the advance of a wavefront"))
