# The sink protocol: where a tracked superphoton's products go. `track!` hands each
# escaped photon to `record!`, each scattered child to `spawn!` and, when scattering, the
# photon's scattering ledger to `tally!`. Device sinks write through atomics; the
# accumulators that implement them for both executors are `Execution.ExactBins` and
# `Execution.ChildBuffer`.

"""
    Sink

Receiver of the products of [`track!`](@ref). A sink implements
`record!(sink, sph)` (an escaped photon) and `spawn!(sink, child, rng, event)` (a
scattered child), and may implement `tally!(sink, sph, loss, spawned)` (the scattering
ledger of one tracked photon; the default ignores it). A sink used inside a kernel is
`isbits` after `Adapt.adapt` and writes only through atomics.
"""
abstract type Sink end

"""
    record!(sink::Sink, sph::Superphoton) -> nothing

Receive the superphoton `sph` that escaped: its final position, momentum and weight.
"""
function record! end

"""
    spawn!(sink::Sink, child::Superphoton, rng::MCRng, event::Int) -> nothing

Receive a scattered `child`. `rng` is the stream of the parent at any draw count (only
its identity enters) and `event` the 0-based ordinal of this spawn along the parent's
march; the child's stream is `child_rng(rng, event, layout)` for the layout of the run,
so the children and their streams depend only on the genealogy.
"""
function spawn! end

"""
    tally!(sink::Sink, sph::Superphoton, loss, spawned) -> nothing

Receive the scattering ledger of the photon `sph` (as it entered the tracker):
`loss = Σ w (1 − e^{−dτ_s})`, the incident weight lost to scattering along the march,
and `spawned`, the total weight of the children it spawned. Their ratio checks the
unbiasedness of the exponential transform. The default does nothing.
"""
@inline tally!(::Sink, sph, loss, spawned) = nothing

"""
    NullSink()

A sink that discards everything.
"""
struct NullSink <: Sink end
@inline record!(::NullSink, sph) = nothing
@inline spawn!(::NullSink, child, rng, event) = nothing

"""
    VectorSink(layout::MCStreamLayout)

A host sink for tests and serial loops: `escaped` collects the recorded photons,
`children` the spawned children with their streams `child_rng(rng, event, layout)`,
and `ledger` the `(nscatt, loss, spawned)` of every tally, in arrival order. Not
thread-safe, not device-safe.
"""
struct VectorSink <: Sink
    layout::MCStreamLayout
    escaped::Vector{Superphoton}
    children::Vector{Tuple{Superphoton,MCRng}}
    ledger::Vector{Tuple{Int32,Float64,Float64}}
end
VectorSink(layout::MCStreamLayout) =
    VectorSink(layout, Superphoton[], Tuple{Superphoton,MCRng}[], Tuple{Int32,Float64,Float64}[])

function record!(s::VectorSink, sph::Superphoton)
    push!(s.escaped, sph)
    return nothing
end

function spawn!(s::VectorSink, child::Superphoton, rng::MCRng, event::Integer)
    push!(s.children, (child, child_rng(rng, event, s.layout)))
    return nothing
end

function tally!(s::VectorSink, sph::Superphoton, loss, spawned)
    push!(s.ledger, (sph.nscatt, Float64(loss), Float64(spawned)))
    return nothing
end
