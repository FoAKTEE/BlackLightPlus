"""
    Products

The products: programs built from a plan and the entry points that run them on an executor chosen by the `backend` keyword.
"""
module Products

using StaticArrays
using LinearAlgebra: dot, det
import Adapt
using ForwardDiff: Dual, value, partials
using ..Numerics: position, momentum, MCRng, MCStreamLayout, widest_layout, mc_uniform, primary_rng,
                  ME, CL, HPL, XSUM_NSLOT, XSUM_MAX_ADDENDS, xsum_add!
using ..Spacetime: KerrSchildCartesian, Minkowski, MarchBudget, march_control, rk4_step, march_step,
                   inverse_metric, trace_geodesic, pixel_prep, band_landing, npixels, proxy_distance,
                   radius_prep, band_radius,
                   surface_prep, surface_landing, image_crossings, surface_terms,
                   ks_radius, raise_index, surface_crossing, EquatorialPlane, ClassicalRK4,
                   march_step_compensated
using ..Matter: GridProvider, EquatorialDisk, UniformSphere, EmissionZone, ZoneSite, intensity,
                sample, emission_zone, SynchrotronSurface, PolarizedDisk, surface_angle, material_spin,
                polarized_emission, EquatorialSpot, spot_amplitude, equatorial_disk_intensity
using ..Transfer: IQUV, StokesI, SpacetimeMarch, StoredPath, march, output_type
using ..MonteCarlo: Sink, Superphoton, GenerationTables, Opacities, Bias, StateBias, FlatMarch,
                    CurvedMarch, MarchLedger, CAPTURED_MARCH, DROPPED_MARCH, escape_direction,
                    generate, track!, table_zone, zone_nsuperphotons, mean_weight, int_jnu_zone,
                    EMISSIVITY_TABLE, MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP, WeightTable, ZoneNodes,
                    zone_count, int_jnu, WEIGHT_FLOOR, HotcrossTable, hotcross_nodes, juttner_norm,
                    total_compton_cross_num
import ..MonteCarlo: record!, child_stream, spawn!, tally!
using ..Data: FluidBlocks, AthenaKSnapshot, locate, build_weight_table, emission_sites, zone_tables,
              build_hotcross_table
import ..Execution
using ..Execution: ThreadedExecutor, resolve, describe, foreach_item!, ray, image_coords,
                   CameraRays, BlockGrid, CoordList, mixed_partition, ImageBlock, stream_blocks!,
                   ExactBins, ExactScalars, ChildBuffer, PolarBands, InclinationBands, Spectrum,
                   band, gather!, children, overflow, scalar, addends, nu_centers, nuLnu_solid
import ..Execution: workgroupsize, stack_bytes
using ..Plans: Plan, Lanes, MixedLanes, geodesic_type, transfer_params, hotcross_table
using ..Analysis: refinement_indicator, needs_refinement, lensing_band, band_grid, band_points

"Version of the package that produced a result."
const PACKAGE_VERSION = pkgversion(@__MODULE__)

include("programs.jl")
include("image.jl")
include("trace.jl")
include("sensitivity.jl")
include("analytic.jl")
include("analytic_surfaces.jl")
include("analytic_movie.jl")
include("crossings.jl")
include("amr.jl")
include("montecarlo.jl")
include("montecarlo_tables.jl")

# programs
export ImagingProgram, TraceProgram, TraceStep, TraceState, ShadowProgram, AnalyticProgram,
       SurfaceProgram, surface_outputs, SpotProgram, FrameProgram, RadiusProgram, CrossingProgram,
       crossing_outputs,
       StoredPathProgram, AdaptiveTraceProgram, TRACE_WINDOW, imaging_program, shadow_program,
       trace_outputs
# Monte Carlo programs
export MCSink, MCTransport, SphereSource, ZoneSource, MCPrimary, MCLayer, LEDGER_STRIPES,
       mc_ledger, ledger_sum, ledger_count
# products and results
export Image, Trajectories, render, render_amr, render_analytic, render_crossings,
       photon_ring_layers, trace, shadow, sensitivity, variants, analytic_movie
export mc_spectrum, ScatterResult, GRMHDSpectrumResult, emission_zones, analytic_nuLnu,
       incl_nuLnu

end # module
