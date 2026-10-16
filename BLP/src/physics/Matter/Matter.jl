"""
    Matter

Plasma as the transfer coefficients see it: the `PlasmaProvider` protocol, the primitive and state records, the code-unit to cgs decode with the electron model and the cuts, the analytic sources and the Monte Carlo emission zones.
"""
module Matter

using StaticArrays
using LinearAlgebra: dot
using ..Numerics: MP, ME, CL, Units
using ..Spacetime: metric_and_inverse

include("records.jl")
include("electrons.jl")
include("decode.jl")
include("protocol.jl")

export PlasmaPrimitives, PlasmaState, vacuum
export PlasmaProvider, GridProvider, AnalyticProvider, primitives, plasma_state, sample, domain_radius
export pgas_from_eint, primitive_four_vectors, state_type, decode
export ElectronModel, electron_theta, Bound, legacy_bound, outside, PlasmaCuts, omitted

end # module
