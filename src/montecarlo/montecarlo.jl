"""
    MonteCarlo

BlackLightPlus Monte Carlo radiative-transfer module, following grmonty
(Dolence et al. 2009) and its GPU descendants (gpumonty, igrmonty). Provides a
counter-based Philox RNG, thermal-synchrotron emission physics, superphoton
generation, tracking, Compton scattering, and spectrum binning. All kernel code
is pure, allocation-free, `isbits` and StaticArrays-typed, with no global RNG
state — one source for CPU, GPU (KernelAbstractions) and AD.
"""
module MonteCarlo

using StaticArrays
using LinearAlgebra
using ..Constants
using ..Radiation: thermal_synchrotron, bessel_k2,
                   SimCoeffConfig, plasma_decode
using ..Simulation: AthenaKSnapshot, sample_primitives, units,
                    primitive_four_vectors, pgas_from_eint
using ..Geodesics

export
    # rng.jl
    philox4x32, philox_rand, MCRng, mc_uniform, mc_gauss,
    sample_direction, standard_gamma, chi_square,
    # emissivity.jl
    j_nu, alpha_nu, jmax, int_jnu_zone, emission_tetrad,
    # superphoton.jl
    Superphoton, photon_position, photon_momentum, photon_weight,
    photon_energy, photon_nscatt,
    # hotcross.jl
    kn_total, boostcross, dNdgamma_e, total_compton_cross_num,
    HotcrossTable, load_hotcross, build_hotcross_table, hotcross_lookup,
    kappa_es, alpha_inv_scatt,
    # scatter.jl
    lorentz_boost, klein_nishina, sample_scattered_energy,
    sample_electron, sample_electron_gamma, sample_mu, compton_scatter,
    scatter_superphoton,
    # track.jl (scattering)
    bias_func, track_superphoton_scatter!, ScatterWorkspace,
    push_photon!, pop_photon!,
    # emission_tables.jl
    EmissionZone, WeightTable, weight_at, build_weight_table, zone_nsuperphotons,
    ZoneEmissionTable, build_zone_emission_table, zone_counts,
    # generate.jl
    sample_zone_photon,
    # track.jl
    track_superphoton,
    # spectrum.jl
    Spectrum, record!, record_energy!, merge_spectra!, nu_centers, nuLnu, nuLnu_total,
    nuLnu_solid, total_power, total_weight, mean_energy,
    # sphere_model.jl
    SphereModel, sphere_zone, sample_sphere_photon, analytic_nuLnu,
    ScatteringSphere, scattering_tau_es,
    # grmhd_model.jl
    GRMHDModel, plasma_at, emission_zones,
    # mc_cpu.jl
    mc_spectrum, mc_spectrum_scatter, ScatterResult,
    track_superphoton_grmhd, track_superphoton_grmhd_scatter!, grmhd_bias,
    mc_spectrum_grmhd, GRMHDSpectrumResult, incl_nuLnu

using ..Geodesics: AbstractSpacetime

include("rng.jl")
include("mc_common.jl")
include("emissivity.jl")
include("superphoton.jl")
include("hotcross.jl")
include("scatter.jl")
include("emission_tables.jl")
include("generate.jl")
include("track.jl")
include("spectrum.jl")
include("sphere_model.jl")
include("grmhd_model.jl")
include("mc_cpu.jl")

end # module
