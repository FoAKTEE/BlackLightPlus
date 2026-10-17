"""
    Products

The products: programs built from a plan and the entry points that run them on an executor chosen by the `backend` keyword.
"""
module Products

using StaticArrays
using LinearAlgebra: dot
import Adapt
using ForwardDiff: Dual, value, partials
using ..Numerics: position, momentum
using ..Spacetime: KerrSchildCartesian, MarchBudget, march_control, rk4_step, inverse_metric,
                   trace_geodesic, pixel_prep, band_landing, npixels
using ..Matter: GridProvider, EquatorialDisk, intensity
using ..Transfer: IQUV, StokesI, SpacetimeMarch, StoredPath, march, output_type
using ..Execution: ThreadedExecutor, resolve, describe, foreach_item!, ray, image_coords,
                   CameraRays, BlockGrid, CoordList, mixed_partition, ImageBlock, stream_blocks!
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

# programs
export ImagingProgram, TraceProgram, TraceStep, TraceState, ShadowProgram, AnalyticProgram,
       StoredPathProgram, AdaptiveTraceProgram, TRACE_WINDOW, imaging_program, shadow_program,
       trace_outputs
# products and results
export Image, Trajectories, render, render_amr, render_analytic, photon_ring_layers, trace,
       shadow, sensitivity, variants

end # module
