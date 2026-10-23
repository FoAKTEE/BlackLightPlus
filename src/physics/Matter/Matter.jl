"""
    Matter

Plasma as the transfer coefficients see it: the `PlasmaProvider` protocol, the primitive and state records, the code-unit to cgs decode with the electron model and the cuts, the analytic sources and the Monte Carlo emission zones.
"""
module Matter

using StaticArrays
using LinearAlgebra: dot
using ..Numerics: MP, ME, CL, KB, HPL, Units, primal
using ..Spacetime: metric, metric_and_inverse, ks_radius, KerrSchildCartesian, Landing,
                   kerr_horizon, kerr_delta, kerr_radial_potential, isco_radius, isco_prograde,
                   marginally_bound_radius, photon_orbit_radius, keplerian_omega,
                   keplerian_specific_angular_momentum

include("records.jl")
include("electrons.jl")
include("decode.jl")
include("protocol.jl")
include("sources/shared.jl")
include("sources/polish_doughnut.jl")
include("sources/page_thorne.jl")
include("sources/hotspot.jl")
include("sources/gdisk.jl")
include("zones.jl")

export PlasmaPrimitives, PlasmaState, vacuum
export PlasmaProvider, GridProvider, AnalyticProvider, primitives, plasma_state, sample, domain_radius
export pgas_from_eint, primitive_four_vectors, state_type, decode
export ElectronModel, electron_theta, Bound, legacy_bound, outside, PlasmaCuts, omitted
export PolishDoughnut, torus_potential, PageThorneDisk, page_thorne_flux, page_thorne_temperature,
       EquatorialHotspot, hotspot_center
export EquatorialDisk, intensity, gfactor_circular, gfactor_plunge, gfactor_disk,
       johnson_su_profile, equatorial_disk_intensity
export EmissionZone, UniformSphere, emission_zone

end # module
