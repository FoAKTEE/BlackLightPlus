"""
    Execution

Where programs run: devices, the `Executor` protocol with its threaded and pooled
implementations, the CPU/GPU switch, the one kernel, residency of host objects on devices,
ray sources, scheduling and the image-block worklist. A program is an `isbits`-after-
`Adapt` functor called per item as `prog(item, i, outs...)`; every executor gives every
item the same result. CUDA methods of the device protocol live in the CUDA extension.
"""
module Execution

using StaticArrays: SVector
using KernelAbstractions: KernelAbstractions, @index, @Const
using Adapt: Adapt
using ..Spacetime: camera_ray

include("devices.jl")
include("pool.jl")
include("residency.jl")
include("kernel.jl")
include("executor.jl")
include("backend.jl")
include("ray_sources.jl")
include("schedule.jl")
include("worklist.jl")
include("accumulators.jl")

# executors and the CPU/GPU switch
export Executor, ThreadedExecutor, PoolExecutor, foreach_item!, describe,
       resolve, backend, backend!, BackendUnavailable
# devices and the pool
export AbstractDevice, HostDevice, CUDADevice, DevicePool, select_devices, cuda_loaded,
       activate!, functional, stage_alloc, stage_free, free!, prepare_launch!, stack_bytes,
       readonly, upload, workgroupsize
# residency
export resident, device_program
# ray sources
export RaySource, ray, HostRays, CameraRays, PixelGrid, BlockGrid, CoordList, image_coords
# scheduling and the worklist
export cost_order, rescue_pixels, mixed_partition, march_sort_order, queue_items,
       curve_inclination, ImageBlock, stream_blocks!
# Monte Carlo accumulators
export Bands, PolarBands, InclinationBands, ExactBins, record!, gather!, Spectrum, band,
       nu_centers, nuLnu, nuLnu_total, nuLnu_solid, total_power, total_weight, mean_energy,
       ExactScalars, add!, scalar, addends, ChildBuffer, spawn!, capacity, overflow, children

function __init__()
    atexit(close_pools)
end

end # module
