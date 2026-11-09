# The stage programs of the device wavefront of a Monte Carlo run: the statements of
# `MCPrimary` cut at the loop back edge of the march. `MCStart` emits a primary and opens
# its march, `MCAdvance` runs a live march for at most the launch's budget of loop
# iterations, `MCRetire` closes a stopped march: its record and its tally. Every value the
# march carries from one iteration to the next lives in the `MarchState`, so a march cut
# into any number of advances is the uncut march.

# The march of one photon of a run: the transport's fixed inputs and the sink of a stripe.
@inline mc_march(t::MCTransport, sink) =
    March(t.ctl, t.st, t.provider, t.opac, sink, t.absorb, t.scatter)

"""
    MCStart(source, transport, sink, seed)

The start stage of a Monte Carlo wavefront, called as `prog(item, i)`: primary `g = item`
draws the stream `primary_rng(seed, g)`, is emitted by `source`, adds its `w E` to the
emitted power of the ledger (stripe of `g`) and opens its march by the
[`MCTransport`](@ref) `transport` into the [`MCSink`](@ref) `sink`; the `MarchState` goes
to slot `i` after the launch's base in the cohort's queue ([`Execution.place!`](@ref)).
The statements of [`MCPrimary`](@ref) before its march.
"""
struct MCStart{G,T,K,Q} <: Function
    source::G
    transport::T
    sink::K
    seed::UInt32
    queues::Q
end
MCStart(source, transport, sink, seed) =
    MCStart(source, transport, sink, UInt32(seed), Execution.WaveQueues())
Adapt.@adapt_structure MCStart

function (p::MCStart)(item, i)
    sink = at_stripe(p.sink, item)
    sph, rng = emit(p.source, item, primary_rng(p.seed, item))
    add_ledger!(sink, EMITTED, sph.w * sph.E)
    Execution.place!(p.queues, i, march_open(mc_march(p.transport, sink), sph, rng))
    return nothing
end

"""
    MCAdvance(transport, sink)

The advance stage of a Monte Carlo wavefront, called as `prog(item, i)`: member `j = item`
of the launch's cohort runs `march_steps` of its march up to loop iteration
`min(it − 1 + min(K, M), M)` (`K` the launch's budget, `M = max_steps(ctl)`, in `Int64`)
and is admitted to the live queue while its outcome is `LIVE_MARCH`, to the stopped queue
otherwise ([`Execution.admit!`](@ref)); a live state whose iteration did not grow, or a
stopped state without an outcome, counts a fault ([`Execution.fault!`](@ref)). Its
children go to the sink. It writes no accumulator.
"""
struct MCAdvance{T,K,Q} <: Function
    transport::T
    sink::K
    queues::Q
end
MCAdvance(transport, sink) = MCAdvance(transport, sink, Execution.WaveQueues())
Adapt.@adapt_structure MCAdvance

function (p::MCAdvance)(item, i)
    q = p.queues
    s = q.A[item]
    t = p.transport
    M = max_steps(t.ctl)
    last = min(Int(s.it) - 1 + min(q.K, M), M)
    s2 = march_steps(mc_march(t, at_stripe(p.sink, item)), s, last)
    if s2.outcome == LIVE_MARCH
        s2.it > s.it || Execution.fault!(q)
        Execution.admit!(q.B, s2)
    else
        stopped_outcome(s2.outcome) || Execution.fault!(q)
        Execution.admit!(q.D, s2)
    end
    return nothing
end

@inline stopped_outcome(o) = o == ESCAPED_MARCH || o == CAPTURED_MARCH || o == DROPPED_MARCH

"""
    MCRetire(transport, sink)

The retire stage of a Monte Carlo wavefront, called as `prog(item, i)`: the stopped
`MarchState` in slot `j = item` of the stopped queue is closed by `march_close` into the
[`MCSink`](@ref) `sink` (ledger stripe of `j`): `record!` of an escape, `tally!` of
every photon.
"""
struct MCRetire{T,K,Q} <: Function
    transport::T
    sink::K
    queues::Q
end
MCRetire(transport, sink) = MCRetire(transport, sink, Execution.WaveQueues())
Adapt.@adapt_structure MCRetire

function (p::MCRetire)(item, i)
    march_close(mc_march(p.transport, at_stripe(p.sink, item)), p.queues.D[item])
    return nothing
end

const MCStage = Union{MCStart,MCAdvance,MCRetire}
Execution.wave_state(::MCStage) = MarchState
Execution.wave_steps(p::MCAdvance) = max_steps(p.transport.ctl)
workgroupsize(::MCStage) = 32

"""
    mc_wavefront(source, transport, sink, seed, primary, layer, depth) -> Wavefront

The [`Execution.Wavefront`](@ref) of a stationary Monte Carlo run: the stage programs
[`MCStart`](@ref), [`MCAdvance`](@ref), [`MCRetire`](@ref) and the history programs
`primary` and `layer` of the same `source`, `transport`, `sink` and `seed`. The run's
scattering depth `depth` must be the transport's `max_scatt`, so that a photon of the
last generation never spawns.
"""
function mc_wavefront(source, transport::MCTransport, sink::MCSink, seed::UInt32, primary,
                      layer, depth::Integer)
    depth == transport.opac.max_scatt ||
        throw(ArgumentError("mc_wavefront: the run's depth $depth is not the transport's " *
                            "max_scatt $(transport.opac.max_scatt)"))
    return Execution.Wavefront(; start = MCStart(source, transport, sink, seed),
                               advance = MCAdvance(transport, sink),
                               retire = MCRetire(transport, sink), primary, layer)
end
