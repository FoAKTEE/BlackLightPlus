"""
    Matter

Plasma as the transfer coefficients see it: the `PlasmaProvider` protocol, the primitive and state records, the code-unit to cgs decode with the electron model and the cuts, the analytic sources and the Monte Carlo emission zones.
"""
module Matter

using StaticArrays
using ..Numerics: MP, ME, CL

include("records.jl")
include("electrons.jl")
include("protocol.jl")

export PlasmaPrimitives, PlasmaState, vacuum, PlasmaProvider, primitives, plasma_state, domain_radius
export ElectronModel, electron_theta, Bound, legacy_bound, outside, PlasmaCuts, omitted

end # module
