"""
    Products

The products: programs built from a plan and the entry points that run them on an executor chosen by the `backend` keyword.
"""
module Products

using StaticArrays
using LinearAlgebra: dot, det
import Adapt
using ForwardDiff: Dual, value, partials
using ..Numerics: position, momentum, MCRng, MCStreamLayout, mc_uniform, primary_rng, ME, CL
using ..Spacetime: KerrSchildCartesian, Minkowski, MarchBudget, march_control, rk4_step,
                   inverse_metric, trace_geodesic, pixel_prep, band_landing, npixels, ks_radius
using ..Matter: GridProvider, EquatorialDisk, UniformSphere, EmissionZone, intensity, sample
using ..Transfer: IQUV, StokesI, SpacetimeMarch, StoredPath, march, output_type
using ..MonteCarlo: Sink, Superphoton, GenerationTables, IntJnuTable, Opacities, Bias, FlatMarch,
                    CurvedMarch, escape_direction, generate, track!, table_zone,
                    zone_nsuperphotons, tabulated_dnmax, int_jnu_zone, MC_NU_MIN, MC_NU_MAX,
                    MC_N_ESAMP, SPHERE_MAX_STEPS, CURVED_MAX_STEPS
import ..MonteCarlo: record!, spawn!, tally!
using ..Data: FluidBlocks, AthenaKSnapshot, locate, build_weight_table, build_zone_emission_table,
              zone_counts, zone_int_jnu_tables, build_int_jnu_table, build_hotcross_table
import ..Execution
using ..Execution: ThreadedExecutor, resolve, describe, foreach_item!, ray, image_coords,
                   CameraRays, BlockGrid, CoordList, mixed_partition, ImageBlock, stream_blocks!,
                   ExactBins, ExactScalars, ChildBuffer, PolarBands, InclinationBands, Spectrum,
                   band, gather!, children, overflow, scalar, addends, nu_centers, nuLnu_solid
import ..Execution: workgroupsize, stack_bytes
using ..Plans: Plan, Lanes, MixedLanes, geodesic_type, transfer_params
using ..Analysis: refinement_indicator, needs_refinement, lensing_band, band_grid

"Version of the package that produced a result."
const PACKAGE_VERSION = pkgversion(@__MODULE__)

include("programs.jl")
include("image.jl")
include("trace.jl")
include("sensitivity.jl")
include("analytic.jl")
include("amr.jl")
include("montecarlo.jl")

# programs
export ImagingProgram, TraceProgram, TraceStep, TraceState, ShadowProgram, AnalyticProgram,
       StoredPathProgram, AdaptiveTraceProgram, TRACE_WINDOW, imaging_program, shadow_program,
       trace_outputs
# Monte Carlo programs
export MCSink, MCTransport, SphereSource, ZoneSource, MCPrimary, MCLayer, LEDGER_STRIPES,
       mc_ledger, ledger_sum, ledger_count
# products and results
export Image, Trajectories, render, render_amr, render_analytic, photon_ring_layers, trace,
       shadow, sensitivity, variants
export mc_spectrum, ScatterResult, GRMHDSpectrumResult, emission_zones, analytic_nuLnu,
       incl_nuLnu

end # module
