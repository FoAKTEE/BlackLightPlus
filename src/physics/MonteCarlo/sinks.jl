# The sink protocol: where a tracked superphoton's products go. `track!` hands each
# escaped photon to `record!`, each scattered child to `spawn!` on the stream that
# `child_stream` names for its event, and the photon's ledger to `tally!`. Device sinks
# write through atomics; the accumulators that implement them for both executors are
# `Execution.ExactBins`, `Execution.ExactScalars` and `Execution.ChildBuffer`.

"""
    Sink

Receiver of the products of [`track!`](@ref). A sink implements `record!(sink, sph)`
(an escaped photon), `child_stream(sink, rng, event)` (the stream of a scattering
event's child) and `spawn!(sink, child, rng)` (a scattered child on its stream), and
may implement `tally!(sink, sph, ledger)` (the [`MarchLedger`](@ref) of one tracked
photon; the default ignores it). A sink used inside a kernel is `isbits` after
`Adapt.adapt` and writes only through atomics.
"""
abstract type Sink end

"""
    record!(sink::Sink, sph::Superphoton) -> nothing

Receive the superphoton `sph` that escaped: its final position, momentum and weight.
"""
function record! end

"""
    child_stream(sink::Sink, rng::MCRng, event::Integer) -> (MCRng, Bool)

The stream of the child of the scattering event of 0-based ordinal `event` along the
march of the photon whose stream is `rng` (at any draw count; only its identity enters),
and whether the ordinal fits the sink's stream layout: `child_rng(rng, event, layout)`
and `event < event_capacity(layout)`. The child's kick and its own march draw from this
stream, so a child depends only on its genealogy.
"""
function child_stream end

"""
    spawn!(sink::Sink, child::Superphoton, rng::MCRng) -> nothing

Receive a scattered `child` with its stream `rng`, the [`child_stream`](@ref) of its
event advanced past the draws of its kick.
"""
function spawn! end

"Outcomes of a march in a [`MarchLedger`](@ref): recorded, captured by the horizon, ended by the step limit."
const ESCAPED_MARCH = 1
"See [`ESCAPED_MARCH`](@ref)."
const CAPTURED_MARCH = 2
"See [`ESCAPED_MARCH`](@ref)."
const DROPPED_MARCH = 3

"""
    MarchLedger

The weight and power account of one tracked photon, handed to [`tally!`](@ref). With
`Δτ = Δτ_a + Δτ_s` of a step and `w` the weight at its start, the step removes
`w (1 − e^{−Δτ})`: `w Δτ_a (1 − e^{−Δτ})/Δτ` absorbed and `w Δτ_s (1 − e^{−Δτ})/Δτ`
scattered, the expected weight of the step's children. Fields: `outcome`
(`ESCAPED_MARCH`, `CAPTURED_MARCH`, `DROPPED_MARCH`); `scatter_loss`, the scattered
weight of the steps that may spawn; `scatter_loss_final`, that of the steps that may
not (the order `max_scatt`); `spawned`, the weight of the children spawned;
`unspawned`, the weight of the events that spawned nothing (an event site outside the
plasma, a child weight at most `weight_min`, an ordinal beyond the stream layout);
`absorbed`, `Σ w E Δτ_a (1 − e^{−Δτ})/Δτ` (mₑc², `E = −p_t` at the step start); `w` and
`E`, the final weight and energy; `events`, the number of event ordinals used. Per
photon, `w_start = w + Σ (absorbed + scattered weights)` up to rounding, and the
expectation of `spawned + unspawned` is `scatter_loss`.
"""
struct MarchLedger
    outcome::Int
    scatter_loss::Float64
    scatter_loss_final::Float64
    spawned::Float64
    unspawned::Float64
    absorbed::Float64
    w::Float64
    E::Float64
    events::Int
end

"""
    tally!(sink::Sink, sph::Superphoton, ledger::MarchLedger) -> nothing

Receive the [`MarchLedger`](@ref) of the photon `sph` (as it entered the tracker). The
default does nothing.
"""
@inline tally!(::Sink, sph, ledger) = nothing

"""
    NullSink()

A sink that discards everything; a child's stream is its parent's.
"""
struct NullSink <: Sink end
@inline record!(::NullSink, sph) = nothing
@inline child_stream(::NullSink, rng, event) = (rng, true)
@inline spawn!(::NullSink, child, rng) = nothing

"""
    VectorSink(layout::MCStreamLayout)

A host sink for tests and serial loops: `escaped` collects the recorded photons,
`children` the spawned children with their streams, and `ledger` the `(nscatt, ledger)`
of every tally, in arrival order; child streams follow `layout`. Not thread-safe, not
device-safe.
"""
struct VectorSink <: Sink
    layout::MCStreamLayout
    escaped::Vector{Superphoton}
    children::Vector{Tuple{Superphoton,MCRng}}
    ledger::Vector{Tuple{Int32,MarchLedger}}
end
VectorSink(layout::MCStreamLayout) =
    VectorSink(layout, Superphoton[], Tuple{Superphoton,MCRng}[], Tuple{Int32,MarchLedger}[])

function record!(s::VectorSink, sph::Superphoton)
    push!(s.escaped, sph)
    return nothing
end

@inline child_stream(s::VectorSink, rng::MCRng, event::Integer) =
    (child_rng(rng, event, s.layout), event < event_capacity(s.layout))

function spawn!(s::VectorSink, child::Superphoton, rng::MCRng)
    push!(s.children, (child, rng))
    return nothing
end

function tally!(s::VectorSink, sph::Superphoton, ledger::MarchLedger)
    push!(s.ledger, (sph.nscatt, ledger))
    return nothing
end
