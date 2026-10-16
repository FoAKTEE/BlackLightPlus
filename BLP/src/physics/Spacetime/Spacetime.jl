# Spacetimes and null geodesics. The module and its protocol type share the name
# `Spacetime`, and Julia keys a module's docstring to the binding of its own name, so
# the module is documented by the docstring of `abstract type Spacetime` (protocol.jl).
module Spacetime

using StaticArrays
using LinearAlgebra: dot, Diagonal
import ForwardDiff
import Adapt
import JacobiElliptic
using OrdinaryDiffEq: Vern7
using SciMLBase: ODEProblem, solve, CallbackSet, ContinuousCallback, terminate!
using ..Numerics: position, momentum, raystate, primal, cr_axis, cr_sample

include("protocol.jl")
include("kerr_schild.jl")
include("user_metric.jl")
include("superposed.jl")
include("adm_grid.jl")
include("metric_ops.jl")
include("geodesic.jl")
include("termination.jl")
include("adaptive.jl")
include("camera.jl")
include("kerr_orbits.jl")
include("analytic_kerr.jl")

# protocol and spacetimes
export Spacetime, StationarySpacetime, ETA, metric, inverse_metric, metric_and_inverse,
       lower_index, raise_index, raise_and_norm, horizon_radius, lane,
       KerrSchildCartesian, Minkowski, UserMetric, SuperposedKerrSchild, ADMGrid,
       ks_radius, sample_adm, on_grid, adm_grid_from_metric
# derivatives and geodesics
export inverse_metric_gradient, metric_gradient, christoffel,
       geodesic_rhs, geodesic_transport_rhs, rk4_step, null_error, energy,
       angular_momentum_z, null_covariant, nullify_covariant
# termination and the adaptive tracer
export ESCAPE_RADIUS, ADM_ESCAPE_RADIUS, HORIZON_EPS, HORIZON_EPS_SAMPLED, CAPTURE_REDSHIFT,
       march_captured, march_escaped, march_stop, normal_frame_energy, ks_radius_energy,
       MarchBudget, MarchStep, march_control, trace_geodesic
# camera
export PlaneParallelCamera, CameraFrame, camera_ray, npixels, pixel_centres, pixel_coords
# exact Kerr
export kerr_horizon, kerr_delta, kerr_radial_potential, isco_radius, isco_prograde,
       photon_orbit_radius, marginally_bound_radius, keplerian_omega,
       keplerian_specific_angular_momentum,
       conserved_quantities, radial_roots, angular_turning_points,
       analytic_angular_integrals, analytic_radial_integrals,
       pixel_prep, Landing, band_landing, critical_curve

end # module
