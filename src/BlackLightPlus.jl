"""
    BlackLightPlus

General-relativistic ray tracing and polarized radiative transfer in Julia,
after blacklight (C++/Kokkos) and ipole: a two-phase driver (geodesics, then
radiation) over pure StaticArrays staged functions shared by CPU, GPU, and AD.
Conventions: signature (−,+,+,+); G = c = M = 1; Cartesian Kerr-Schild; state
y = [x^μ; p_μ] (covariant momentum, p_t conserved for stationary metrics);
invariant coefficients (j/ν², αν, ρν); Stokes transfer in a parallel-transported
camera frame."""
module BlackLightPlus

include("constants.jl")
include("inputs/parameters.jl")
include("geodesics/geodesics.jl")
include("simulation/simulation.jl")
include("radiation/radiation.jl")
include("montecarlo/montecarlo.jl")
include("sources/sources.jl")
include("outputs/outputs.jl")

using .Constants
using .Inputs
using .Geodesics
using .Simulation
using .Radiation
using .MonteCarlo
using .Sources
using .Outputs

include("render.jl")
include("amr.jl")

# Alias for the polarized coupling in Radiation (tests import `PolarizedTransfer.stokes_couple`).
const PolarizedTransfer = Radiation

# Curated public API, grouped by owning submodule; Constants stays module-private.

# Inputs — run configuration
export CameraConfig, PlasmaConfig, RunConfig

# Geodesics — spacetimes, metric, integration, camera, Kerr orbits, analytic (AART/Krang) tracer
export KerrSchildCartesian, Minkowski, UserMetric, SuperposedKerrSchild,
       ADMGridSpacetime, adm_grid_from_metric, sample_adm,
       ks_radius_hole, horizon_radius_hole,
       metric, inverse_metric, inverse_metric_gradient, ks_radius, horizon_radius,
       geodesic_rhs, null_error, energy, angular_momentum_z, nullify_covariant,
       trace_geodesic, trace_rays, trace_rays_sampled, rk4_step_fixed,
       PlaneParallelCamera, camera_rays, camera_rays_at,
       isco_radius, isco_prograde, photon_orbit_radius, marginally_bound_radius,
       keplerian_omega, keplerian_specific_angular_momentum,
       conserved_quantities, radial_roots, angular_turning_points,
       analytic_angular_integrals, analytic_radial_integrals,
       analytic_emission, equatorial_intensity,
       critical_curve, apparent_horizon, lensing_band, band_grid, polygon_area

# Simulation — AthenaK / Z4c readers + slow-light snapshot series
export read_athenak_bin, athenak_bin_snapshot, resample_z4c_adm, z4c_grid_spacetime,
       SnapshotSeries, synthetic_series, scale_density, bracketing_pair,
       get_frame, sample_primitives_t,
       LeafTable, leaf_table, meshblock_index

# Radiation — tetrads, transfer coefficients, Stokes coupling, equatorial disk source
export fluid_tetrad, SimCoeffConfig, electron_theta, plasma_cut_omits,
       thermal_synchrotron, thermal_synchrotron_polarized,
       power_law_synchrotron_polarized, kappa_synchrotron_polarized,
       precompute_power_law!, precompute_kappa!,
       bessel_k0, bessel_k1, bessel_k2, harmonic_bridge,
       stokes_couple, couple_unpolarized,
       gfactor_disk, gfactor_circular, gfactor_plunge, johnson_su_profile,
       equatorial_disk_intensity, EquatorialDisk

# MonteCarlo — full module surface (rng -> Compton -> GRMHD spectrum)
export philox4x32, philox_rand, MCRng, mc_uniform, mc_gauss,
       sample_direction, standard_gamma, chi_square,
       j_nu, alpha_nu, jmax, int_jnu_zone, emission_tetrad,
       Superphoton, photon_position, photon_momentum, photon_weight,
       photon_energy, photon_nscatt,
       kn_total, boostcross, dNdgamma_e, total_compton_cross_num,
       HotcrossTable, load_hotcross, build_hotcross_table, hotcross_lookup,
       kappa_es, alpha_inv_scatt,
       lorentz_boost, klein_nishina, sample_scattered_energy,
       sample_electron, sample_electron_gamma, sample_mu, compton_scatter,
       scatter_superphoton,
       bias_func, track_superphoton_scatter!, ScatterWorkspace,
       push_photon!, pop_photon!,
       EmissionZone, WeightTable, weight_at, build_weight_table,
       zone_nsuperphotons, sample_zone_photon,
       ZoneEmissionTable, build_zone_emission_table, zone_counts,
       track_superphoton,
       Spectrum, record!, record_energy!, merge_spectra!, nu_centers, nuLnu, nuLnu_total,
       nuLnu_solid, total_power, total_weight, mean_energy,
       SphereModel, sphere_zone, sample_sphere_photon, analytic_nuLnu, mc_spectrum,
       ScatteringSphere, scattering_tau_es, mc_spectrum_scatter, ScatterResult,
       GRMHDModel, plasma_at, emission_zones,
       track_superphoton_grmhd, track_superphoton_grmhd_scatter!, grmhd_bias,
       mc_spectrum_grmhd, GRMHDSpectrumResult, incl_nuLnu

# Sources — analytic plasma models (imaging and Monte Carlo via the source_plasma_at seam)
export AbstractSource, source_plasma_at,
       PolishDoughnut, PageThorneDisk, EquatorialHotspot,
       torus_potential, page_thorne_flux, page_thorne_temperature, hotspot_center

# Outputs — image writers + AART interferometry (visibility amplitudes)
export write_npy,
       radon_cut, visamp, visamp_minima, tukey_window, MUAS_TO_RAD

# BlackLightPlus top-level (render.jl / amr.jl) — not owned by any submodule
export render, render_amr, render_analytic, photon_ring_layers

end # module
