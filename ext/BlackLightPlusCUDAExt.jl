"""
    BlackLightPlusCUDAExt

CUDA methods of the `Execution` device protocol for `CUDADevice`: activation, the
KernelAbstractions backend, the functional probe, page-locked staging buffers, early
release of device arrays, the free device memory, the lanes' streams and the arrays they
share, the per-thread stack limit, the launch under a register cap, read-only loads
through the constant cache, and the idle-device selection behind `select_devices`.
"""
module BlackLightPlusCUDAExt

using BlackLightPlus, CUDA
using Adapt: Adapt
using KernelAbstractions: KernelAbstractions
using BlackLightPlus.Execution: CUDADevice
import BlackLightPlus.Execution: backend, activate!, functional, stage_alloc, stage_free, free!,
                      free_bytes, share_lanes!, lane_queue, sync_queue, queue_idle,
                      prepare_launch!, stack_bytes, launch_capped!, readonly, select_devices

const KA = KernelAbstractions
const CUDACore = CUDA.CUDACore

backend(::CUDADevice) = CUDA.CUDABackend()

activate!(d::CUDADevice) = (CUDA.device!(d.ordinal); nothing)

functional(::Type{CUDADevice}) = CUDA.functional()
functional(d::CUDADevice) = CUDA.functional() && 0 <= d.ordinal < visible_count()

# Number of visible devices; 0 when the driver cannot be queried.
visible_count() = try
    CUDA.ndevices()
catch
    0
end

# Page-locked memory from `cuMemHostAlloc`, released by `stage_free`: a buffer pinned by
# `CUDA.pin` is unregistered by a finalizer that fails while another thread holds the pin
# lock, which would leave freed memory registered.
function stage_alloc(::CUDADevice, ::Type{T}, n::Integer) where {T}
    mem = CUDA.alloc(CUDA.HostMemory, max(n, 1) * sizeof(T))
    return (unsafe_wrap(Array, convert(Ptr{T}, pointer(mem)), n; own = false), mem)
end
stage_free(::CUDADevice, mem::CUDA.HostMemory) = CUDA.free(mem)

free!(::CUDADevice, x::CUDA.CuArray) = CUDA.unsafe_free!(x)

# The driver's free memory of the device (`cuMemGetInfo`).
free_bytes(d::CUDADevice) = CUDA.device!(() -> Int(CUDA.memory_info()[1]), d.ordinal)

# A lane's queue is its task's stream. CUDA.jl orders a device array by the stream that
# last converted it for a launch: a conversion on another stream waits for that one and
# takes it over, and a free is ordered against it alone. A shared array waits once for the
# stream that wrote it, then no conversion waits; the stream a free is ordered against is
# then any lane's, so it is freed only after `sync_lanes`.
lane_queue(::CUDADevice) = CUDA.stream()
sync_queue(d::CUDADevice, s::CUDA.CuStream) =
    (CUDA.device!(() -> CUDA.synchronize(s), d.ordinal); nothing)
queue_idle(::CUDADevice, s::CUDA.CuStream) = CUDA.isdone(s)

function share_lanes!(::CUDADevice, a::CUDA.CuArray)
    CUDA.synchronize(a)
    CUDA.enable_synchronization!(a, false)
    return a
end

function prepare_launch!(::CUDADevice, prog)
    bytes = stack_bytes(prog)
    bytes > 0 && CUDA.limit(CUDA.LIMIT_STACK_SIZE) < bytes &&
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, bytes)
    return nothing
end

# The launch of a KernelAbstractions kernel under a register cap: the statements of the
# backend's launch, `(obj::KA.Kernel{CUDABackend})(args...; ndrange, workgroupsize)` of
# CUDACore's `CUDAKernels.jl`, with `maxregs = cap` added to the compilation, so the
# kernel's context, specialisation, block size and grid are the backend's.
# `test/infra/execution_launch.jl` compares these statements with the backend's.
function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function launch_capped!(::CUDADevice, obj::KA.Kernel{CUDA.CUDABackend}, args::Tuple, ndrange,
                        workgroupsize, cap::Integer)
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = @cuda launch=false always_inline=backend.always_inline maxthreads=maxthreads maxregs=cap obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        if backend.prefer_blocks
            # Prefer blocks over threads
            threads = min(prod(ndrange), config.threads)
            cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
            threads = cld(prod(ndrange), cu_blocks)
        else
            threads = config.threads
        end

        workgroupsize = threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if blocks == 0
        return nothing
    end

    # Launch kernel
    kernel(ctx, args...; threads, blocks)

    return nothing
end

# A resident array read through the read-only data cache: kernel argument conversion turns
# it into `CUDA.Const` of the device array, the wrapper `@Const` gives top-level arguments.
struct ReadOnlyArray{T,N,A<:CUDA.CuArray{T,N}} <: AbstractArray{T,N}
    a::A
end
Base.size(r::ReadOnlyArray) = size(r.a)
Base.getindex(r::ReadOnlyArray, i::Int...) = r.a[i...]
Base.parent(r::ReadOnlyArray) = r.a
Adapt.adapt_structure(to::CUDA.KernelAdaptor, r::ReadOnlyArray) =
    Base.Experimental.Const(Adapt.adapt(to, r.a))

readonly(::CUDADevice, a::CUDA.CuArray) = ReadOnlyArray(a)

# Idle when free memory ≥ `min_free_gb` GB and NVML compute utilisation ≤ `util_threshold`
# percent (an unknown utilisation is ignored); a failed probe is not idle.
function idle(d::Integer, util_threshold::Real, min_free_gb::Real)
    return try
        CUDA.device!(d) do
            free, _ = CUDA.memory_info()
            u = try
                Float64(CUDA.NVML.utilization_rates(
                    CUDA.NVML.Device(CUDA.uuid(CUDA.CuDevice(d)))).compute) * 100.0
            catch
                -1.0
            end
            (free / 1024^3 >= min_free_gb) && (u < 0 || u <= util_threshold)
        end
    catch
        false
    end
end

function select_devices(::Type{CUDADevice}; candidates = nothing, util_threshold::Real = 15,
                        min_free_gb::Real = 8, fallback::Bool = false)
    CUDA.functional() || return CUDADevice[]
    n = visible_count()
    cands = candidates === nothing ? (0:n-1) : candidates
    vis = filter(d -> 0 <= d < n, collect(Int, cands))
    isempty(vis) && return CUDADevice[]
    picked = filter(d -> idle(d, util_threshold, min_free_gb), vis)
    isempty(picked) && fallback && (picked = vis[1:1])
    return CUDADevice.(picked)
end

end # module
