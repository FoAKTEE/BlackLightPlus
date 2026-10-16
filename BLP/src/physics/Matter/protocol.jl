"""
    PlasmaProvider

A source of plasma for the transfer coefficients and the Monte Carlo tracker.
Every provider implements `domain_radius(p)`, the coordinate radius beyond
which it is vacuum, and is sampled by `sample(p, st, x, params)`. The two
families are [`GridProvider`](@ref), which supplies code-unit primitives, and
[`AnalyticProvider`](@ref), which supplies the cgs state directly. A provider
that is sampled inside a kernel must be `isbits` after `Adapt.adapt`.
"""
abstract type PlasmaProvider end

"""
    GridProvider <: PlasmaProvider

A provider that implements `primitives(p, x) -> PlasmaPrimitives` in code units;
its state is `decode`d with the units, electron model and cuts of the parameter
set. The metric is evaluated only where the primitives carry rest mass.
"""
abstract type GridProvider <: PlasmaProvider end

"""
    AnalyticProvider <: PlasmaProvider

A provider that implements `plasma_state(p, st, x, g, params) -> PlasmaState`
directly in cgs from the metric pair `g` at `x`.
"""
abstract type AnalyticProvider <: PlasmaProvider end

"""
    primitives(p::GridProvider, x::SVector{4}) -> PlasmaPrimitives

The fluid primitives at the spacetime point `x`, with `inside = false` and
every field zero outside the provider's domain.
"""
function primitives end

"""
    domain_radius(p::PlasmaProvider) -> Float64

Coordinate radius outside which the provider is vacuum.
"""
function domain_radius end

"""
    plasma_state(p::PlasmaProvider, st, x, g, params) -> PlasmaState

The plasma state at `x` under the spacetime `st`, whose metric pair at `x` is
`g = (gcov, gcon) = metric_and_inverse(st, x)`, decoded with the `units`,
`electrons` and `cuts` fields of `params`. Grid providers decode their
primitives; analytic providers define it directly.
"""
@inline plasma_state(p::GridProvider, st, x, g, params) =
    decode(primitives(p, x), g, params.units, params.electrons, params.cuts)

"""
    sample(p::PlasmaProvider, st, x, params) -> (state::PlasmaState, g)

The plasma state at `x` and the metric pair `g = (gcov, gcon)` at `x` that
decoded it, in one call, so that a march evaluates the metric once per emitting
sample. For a grid provider whose primitives carry no rest mass at `x` the state
is `vacuum` and `g` is a pair of zero matrices that must not be read.
"""
@inline function sample(p::GridProvider, st, x::SVector{4,Tx}, params) where {Tx}
    prim = primitives(p, x)
    if prim.rho > 0.0
        g = metric_and_inverse(st, x)
        return decode(prim, g, params.units, params.electrons, params.cuts), g
    end
    T = state_type(typeof(prim.rho), Tx, params.units, params.electrons)
    z = zero(SMatrix{4,4,Tx})
    return vacuum(T), (z, z)
end

@inline function sample(p::AnalyticProvider, st, x::SVector{4}, params)
    g = metric_and_inverse(st, x)
    return plasma_state(p, st, x, g, params), g
end
