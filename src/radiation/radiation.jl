"""
    Radiation

Radiative transfer in curved spacetime: fluid orthonormal tetrads, synchrotron
transfer coefficients (thermal, power-law and kappa; unpolarized and the 8
polarized), the analytic constant-coefficient Stokes segment operators, the reverse
transfer marches shared with the GPU kernels (Stokes I and IQUV in a
parallel-transported camera frame), and the AART equatorial g-disk source.
Follows blacklight's `radiation/` module; deviations are noted in the
individual files.
"""
module Radiation

using StaticArrays
using LinearAlgebra
using SpecialFunctions: gamma, besselk
using ..Constants
using ..Geodesics: metric, inverse_metric, metric_and_inverse, metric_gradient, christoffel,
                   KerrSchildCartesian, ks_radius_energy, march_stop, march_escaped,
                   spacetime_control, rk4_step_fixed, geodesic_transport_rhs
using ..Simulation: primitive_four_vectors, pgas_from_eint

export fluid_tetrad, fluid_tetrad_checked,
       SimCoeffConfig, coefficient_params, has_power_law, has_kappa,
       electron_theta, plasma_cut_omits, plasma_decode,
       thermal_synchrotron, thermal_synchrotron_polarized,
       power_law_synchrotron_polarized, kappa_synchrotron_polarized,
       precompute_power_law!, precompute_kappa!,
       bessel_k0, bessel_k1, bessel_k2, harmonic_bridge,
       stokes_couple, couple_unpolarized, stokes_segment, unpolarized_segment,
       stokes_admissible,
       eight_coefficients, unpolarized_coefficients, polarized_march, unpolarized_march,
       KerrMarch, SpacetimeMarch, StoredPath, reverse_unpolarized_march, reverse_polarized_march,
       # AART equatorial g-disk source
       gfactor_disk, gfactor_circular, gfactor_plunge, johnson_su_profile,
       equatorial_disk_intensity, EquatorialDisk, isco_prograde

include("tetrads.jl")
include("coefficients.jl")
include("transfer.jl")
include("integrator.jl")
include("reverse_march.jl")
include("equatorial_source.jl")

end # module
