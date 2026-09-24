"""
    Geodesics

Null-geodesic engine: metric type tree with AD-derived metric derivatives
(`metrics.jl`), plane-parallel camera (`camera.jl`), the Hamiltonian
covariant-momentum integrator (`integrator.jl`), general analytic and numerical
spacetimes, and the exact-Kerr analytic tracer + lensing bands (AART/Krang).
"""
module Geodesics

using StaticArrays
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq: Vern7, Tsit5
using SciMLBase: ODEProblem, solve, CallbackSet, ContinuousCallback, terminate!

export AbstractSpacetime, AnalyticSpacetime, StationarySpacetime,
       KerrSchildCartesian, Minkowski, UserMetric, SuperposedKerrSchild,
       ADMGridSpacetime, adm_grid_from_metric, sample_adm,
       ks_radius_hole, horizon_radius_hole,
       metric, inverse_metric, metric_and_inverse, inverse_metric_gradient,
       ks_radius, horizon_radius,
       geodesic_rhs, geodesic_transport_rhs, null_error, energy, angular_momentum_z,
       lower_index, raise_index, raise_and_norm, nullify_covariant, metric_gradient, christoffel,
       trace_geodesic, trace_rays, trace_rays_sampled, rk4_step_fixed,
       PlaneParallelCamera, camera_rays, camera_rays_at,
       # shared exact-Kerr orbital primitives (kerr_orbits.jl)
       kerr_horizon, kerr_delta, kerr_radial_potential,
       isco_radius, isco_prograde, photon_orbit_radius, marginally_bound_radius,
       keplerian_omega, keplerian_specific_angular_momentum,
       # analytic Kerr (AART/Krang) elliptic equatorial tracer
       conserved_quantities, radial_roots, angular_turning_points,
       analytic_angular_integrals, analytic_radial_integrals,
       analytic_emission, equatorial_intensity,
       # lensing bands (AART)
       critical_curve, apparent_horizon, lensing_band, band_grid, polygon_area

include("metrics.jl")
include("camera.jl")
include("integrator.jl")
include("spacetimes.jl")  # after integrator: uses _rhs_ode, geodesic_rhs
include("numerical_spacetime.jl")  # after integrator: uses ETA, _rhs_ode, geodesic engine
include("kerr_orbits.jl")  # shared exact-Kerr orbital primitives (before analytic/lensing)
include("analytic_kerr.jl")  # AART/Krang exact-Kerr elliptic equatorial tracer
include("lensing_bands.jl")  # AART critical curve + n-band lensing bands (host setup)

end # module
