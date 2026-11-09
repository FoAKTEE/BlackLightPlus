# The stage programs of the device wavefront of a Monte Carlo run: the statements of
# `MCPrimary` and `MCLayer` cut at the loop back edge of the march. `MCStart` emits a
# primary and opens its march, `MCBranch` opens the march of a child, `MCAdvance` runs a
# live march for at most the launch's budget of loop iterations (its children kicked
# inline, as `track!` kicks them), `MCRetire` closes a stopped march: its record and its
# tally; `MCKey` keys a live march for the sort of the cohort. Every value the march carries from one iteration to the next lives in the
# `MarchState`, so a march cut into any number of advances is the uncut march.

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
children are kicked inside the march by the sink's `branch!` and go to the child queue
the launch binds as the sink's `ChildBuffer`. It writes no accumulator.
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

"""
    MCBranch(transport, sink)

The branch stage of a Monte Carlo wavefront, called as `prog(item, i)`: child `j = item`
of the launch's child queue, a scattered `Superphoton` on its genealogy stream, opens its
march by `march_open` of the [`MCTransport`](@ref) `transport` and is handed to
[`Execution.admit_child!`](@ref), which admits it to the live queue or counts it as
dropped beyond the call's depth. The statements of [`MCLayer`](@ref) before its march.
"""
struct MCBranch{T,K,Q} <: Function
    transport::T
    sink::K
    queues::Q
end
MCBranch(transport, sink) = MCBranch(transport, sink, Execution.WaveQueues())
Adapt.@adapt_structure MCBranch

function (p::MCBranch)(item, i)
    q = p.queues
    @inbounds sph = q.C.photons[item]
    @inbounds rng = q.C.rngs[item]
    Execution.admit_child!(q, march_open(mc_march(p.transport, p.sink), sph, rng))
    return nothing
end

"""
    MCKey(transport)

The key stage of a Monte Carlo wavefront, called as `prog(item, i)`: member `j = item` of
the launch's cohort is keyed by its scattering generation `g` (the entry photon's `nscatt`,
at most the `max_scatt` of the [`MCTransport`](@ref) `transport`), then by
[`photon_key`](@ref) of its state on the transport's spacetime under its march control:
`g × PHOTON_KEYS + photon_key`, one of `(max_scatt + 1) × PHOTON_KEYS` values, stored and
counted by [`Execution.count_key!`](@ref). Sorted so, the warps of an advance hold photons
of one generation with nearby positions and directions. The key orders the advance's reads
of the cohort and enters no result.
"""
struct MCKey{S,C,Q} <: Function
    st::S
    ctl::C
    ngen::Int
    queues::Q
end
MCKey(transport::MCTransport) =
    MCKey(transport.st, transport.ctl, Int(transport.opac.max_scatt) + 1, Execution.KeyCount())
Adapt.@adapt_structure MCKey

# The key of the state `s` under the key stage `p`: its generation, then its photon_key.
@inline mc_key(p::MCKey, s::MarchState) =
    min(Execution.generation(s), p.ngen - 1) * PHOTON_KEYS + photon_key(p.st, p.ctl, s.y)

function (p::MCKey)(item, i)
    q = p.queues
    Execution.count_key!(q, item, mc_key(p, q.A[item]))
    return nothing
end

Execution.wave_keys(p::MCKey) = p.ngen * PHOTON_KEYS

const MCStage = Union{MCStart,MCAdvance,MCRetire,MCBranch}
Execution.wave_state(::Union{MCStage,MCKey}) = MarchState
Execution.wave_steps(p::MCAdvance) = max_steps(p.transport.ctl)
Execution.generation(s::MarchState) = Int(s.entry.nscatt)
workgroupsize(::MCStage) = 32

"""
    mc_wavefront(source, transport, sink, seed, primary, layer, depth) -> Wavefront

The [`Execution.Wavefront`](@ref) of a stationary Monte Carlo run: the stage programs
[`MCStart`](@ref), [`MCAdvance`](@ref), [`MCRetire`](@ref) and, for a scattering run
(`depth > 0`), [`MCBranch`](@ref) and the key stage [`MCKey`](@ref), and the history
programs `primary` and `layer` of the same `source`, `transport`, `sink` and `seed`. A run
without scattering has no key stage: its cohort is never sorted. The run's scattering depth `depth` must be
the transport's `max_scatt`, so that a photon of the last generation never spawns.
"""
function mc_wavefront(source, transport::MCTransport, sink::MCSink, seed::UInt32, primary,
                      layer, depth::Integer)
    depth == transport.opac.max_scatt ||
        throw(ArgumentError("mc_wavefront: the run's depth $depth is not the transport's " *
                            "max_scatt $(transport.opac.max_scatt)"))
    return Execution.Wavefront(; start = MCStart(source, transport, sink, seed),
                               advance = MCAdvance(transport, sink),
                               retire = MCRetire(transport, sink),
                               branch = depth > 0 ? MCBranch(transport, sink) : nothing,
                               key = depth > 0 ? MCKey(transport) : nothing, primary, layer)
end
