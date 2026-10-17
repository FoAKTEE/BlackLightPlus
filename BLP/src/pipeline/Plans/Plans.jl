"""
    Plans

Run configuration, precision lanes and `prepare`, which turns a configuration, a spacetime and a matter source into a backend-free `Plan`.
"""
module Plans

using ..Numerics: Units
using ..Spacetime: Spacetime, KerrSchildCartesian, Minkowski, PlaneParallelCamera, CameraFrame,
                   MarchBudget, lane, horizon_radius, critical_curve
import ..Spacetime: npixels
using ..Matter: ElectronModel, PlasmaCuts, UniformSphere, legacy_bound, domain_radius
using ..Transfer: TransferMode, StokesI, IQUV, PowerLaw, Kappa, TransferParams
import ..Transfer: output_type
using ..Execution: PixelGrid, cost_order, rescue_pixels, curve_inclination

include("config.jl")
include("lanes.jl")
include("plan.jl")
include("prepare.jl")

export CameraConfig, PlasmaConfig, RunConfig
export Lanes, MixedLanes, geodesic_type
export Plan, prepare, transfer_params

end # module
