"""
    Transfer

Radiative transfer along a ray: synchrotron coefficients of the thermal, power-law and kappa populations, fluid tetrads, segment operators and the unpolarized (`StokesI`) and polarized (`IQUV`) marches over a march control.
"""
module Transfer

using StaticArrays
using LinearAlgebra: dot
using SpecialFunctions: gamma, besselk
import Adapt
using ..Numerics: CL, EE, ME, HPL, SQRT2, Units, primal, bessel_k0, bessel_k1, harmonic_bridge,
                  hypergeometric_2f1
using ..Spacetime: KerrSchildCartesian, MarchBudget, MarchStep, metric_and_inverse, rk4_step,
                   geodesic_transport_rhs
import ..Spacetime: march_control
using ..Matter: PlasmaState, ElectronModel, PlasmaCuts, GridProvider, sample, primitives, decode

include("params.jl")
include("synchrotron.jl")
include("tetrads.jl")
include("coefficients.jl")
include("segments.jl")
include("controls.jl")
include("march.jl")

# modes and parameters
export TransferMode, StokesI, IQUV, PowerLaw, Kappa, TransferParams, has_powerlaw, has_kappa,
       output_type
# synchrotron fits
export thermal_synchrotron, thermal_synchrotron_polarized, power_law_synchrotron,
       power_law_synchrotron_polarized, kappa_synchrotron, kappa_synchrotron_polarized,
       K_THETA_E_ZERO, K_THETA_E_BLEND_LO
# tetrads and coefficients
export fluid_tetrad, fluid_tetrad_checked, normal_observer, qu_rotation, rotate_qu,
       coefficients, kinematic
# segment operators
export POL_DELTA_TAU_MAX, segment, couple, stokes_admissible
# march controls and the marches
export SpacetimeMarch, StoredPath, march_control, max_steps, geodesic_step, next_state,
       transfer_spacetime, geodesic_spacetime, renorm_e1, march

end # module
