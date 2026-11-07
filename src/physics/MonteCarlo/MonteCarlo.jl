"""
    MonteCarlo

Monte Carlo Comptonization: superphotons, emissivities, hot-cross-section lookup, generation, the Compton kick, opacity rates, the `Sink` protocol and the one tracker.
"""
module MonteCarlo

using StaticArrays
using LinearAlgebra: dot, norm, cross
import Adapt
using ..Numerics: CL, EE, ME, MP, HPL, SQRT2, SIGMA_THOMSON, MCRng, MCStreamLayout, mc_uniform,
                  sample_direction, chi_square, child_rng, event_capacity, loglinear_at, bessel_k2,
                  position, momentum, raystate
using ..Spacetime: Minkowski, MarchStep, HORIZON_EPS_SAMPLED, metric_and_inverse,
                   inverse_metric, proxy_distance, horizon_floor, plunge_radius, rk4_step
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
export j_nu, alpha_nu, jmax, int_jnu_simpson, N_ZONE_THETA, EmissivityTable, build_emissivity_table,
       EMISSIVITY_TABLE, F_TABLE_THETA, F_TABLE_KMIN, F_TABLE_KMID, F_TABLE_KMAX, F_TABLE_DL1,
       F_TABLE_DL2, synchrotron_F, int_jnu, int_jnu_zone, emission_tetrad, SMALL_VECTOR
# hot cross section
export MINW, MAXW, MINT, MAXT, NW, NT, DMUE, DGAMMAE, MAXGAMMA, N_MUE, N_GAMMAE,
       kn_total, boostcross, dNdgamma_e, total_compton_cross_num, HotcrossTable,
       hotcross_lookup, kappa_es, alpha_inv_scatt
# generation tables and generation
export MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP, WEIGHT_FLOOR, WeightTable, weight_at, node_weight,
       mean_weight, zone_count, zone_nsuperphotons, REJECT_MAX_ITER, GenerationTables, generate,
       table_zone
# the Compton kick
export lorentz_boost, sample_y, sample_electron_gamma, sample_mu, sample_electron,
       klein_nishina, sample_scattered_energy, compton_scatter, scatter_superphoton
# opacities
export MC_WEIGHT_MIN, FluidFrame, StaticFrame, rate_frame, absorption_rate, scattering_rate,
       Bias, bias, amplification, StateBias, event_bias, Opacities
# sinks
export Sink, record!, child_stream, spawn!, tally!, MarchLedger, ESCAPED_MARCH, CAPTURED_MARCH,
       DROPPED_MARCH, NullSink, VectorSink
# the tracker
export SPHERE_STEP_FRAC, SPHERE_MAX_STEPS, CURVED_MAX_STEPS, FlatMarch, CurvedMarch, escaped,
       captured, plunging, escape_direction, event_weight, next_event, scatter_site, track!

end # module
