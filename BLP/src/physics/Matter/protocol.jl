"""
    PlasmaProvider

A source of plasma for the transfer coefficients and the Monte Carlo tracker.
Grid-backed providers implement `primitives(p, x)`; analytic providers implement
`plasma_state` directly in cgs. Every provider implements `domain_radius(p)`,
the radius beyond which it is vacuum, which bounds the emission region of a
march. A provider that is sampled inside a kernel must be `isbits` after
`Adapt.adapt`.
"""
abstract type PlasmaProvider end

"""
    primitives(p::PlasmaProvider, x::SVector{4}) -> PlasmaPrimitives

The fluid primitives at spacetime point `x`, with `inside = false` outside the
provider's domain.
"""
function primitives end

"""
    domain_radius(p::PlasmaProvider) -> Float64

Coordinate radius outside which the provider is vacuum.
"""
function domain_radius end

"""
    plasma_state(p::PlasmaProvider, st, x, g, params) -> PlasmaState

The plasma state at `x` under the spacetime `st`, whose covariant and
contravariant metric at `x` is the pair `g = (gcov, gcon)`, decoded with the
`units`, `electrons` and `cuts` fields of `params`. The default decodes the
provider's primitives; analytic providers override it.
"""
@inline plasma_state(p::PlasmaProvider, st, x, g, params) =
    decode(primitives(p, x), g, params.units, params.electrons, params.cuts)
