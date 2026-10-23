"""
    MonteCarlo

Monte Carlo Comptonization: superphotons, emissivities, hot-cross-section lookup, generation, the Compton kick, opacity rates, the `Sink` protocol and the one tracker.
"""
module MonteCarlo

using StaticArrays
using LinearAlgebra: dot, norm, cross
import Adapt
using ..Numerics: CL, ME, MP, HPL, SIGMA_THOMSON, MCRng, MCStreamLayout, mc_uniform,
                  sample_direction, chi_square, child_rng, loglinear_at, bessel_k2,
                  position, momentum, raystate
using ..Spacetime: Minkowski, MarchStep, HORIZON_EPS_SAMPLED, metric_and_inverse,
                   inverse_metric, proxy_distance, horizon_floor, rk4_step
using ..Matter: PlasmaState, PlasmaProvider, EmissionZone, UniformSphere, sample, emission_zone
using ..Transfer: thermal_synchrotron, TransferParams
import ..Transfer: march_control, max_steps, geodesic_step

include("superphoton.jl")
include("emissivity.jl")
include("hotcross.jl")
include("tables.jl")
include("generate.jl")
include("scatter.jl")
include("opacity.jl")
include("sinks.jl")
include("track.jl")

# superphotons
export Superphoton, photon_position, photon_momentum, photon_weight, photon_energy,
       photon_nscatt
# emissivity
export j_nu, alpha_nu, jmax, int_jnu_zone, N_ZONE_THETA, emission_tetrad, SMALL_VECTOR
# hot cross section
export MINW, MAXW, MINT, MAXT, NW, NT, DMUE, DGAMMAE, MAXGAMMA, N_MUE, N_GAMMAE,
       kn_total, boostcross, dNdgamma_e, total_compton_cross_num, HotcrossTable,
       hotcross_lookup, kappa_es, alpha_inv_scatt
# generation tables and generation
export MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP, WEIGHT_FLOOR, INT_JNU_REFINE, WeightTable, weight_at,
       ZoneEmissionTable, IntJnuTable, int_jnu_at, zone_nsuperphotons, tabulated_dnmax,
       REJECT_MAX_ITER, GenerationTables, generate, table_zone
# the Compton kick
export lorentz_boost, sample_y, sample_electron_gamma, sample_mu, sample_electron,
       klein_nishina, sample_scattered_energy, compton_scatter, scatter_superphoton
# opacities
export MC_WEIGHT_MIN, FluidFrame, StaticFrame, rate_frame, absorption_rate, scattering_rate,
       Bias, bias, Opacities
# sinks
export Sink, record!, spawn!, tally!, NullSink, VectorSink
# the tracker
export SPHERE_STEP_FRAC, SPHERE_MAX_STEPS, CURVED_MAX_STEPS, FlatMarch, CurvedMarch, escaped,
       escape_direction, scatter_event, may_scatter, scatter_site, track!

end # module
