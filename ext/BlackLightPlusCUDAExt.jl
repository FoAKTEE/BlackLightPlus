"""
    BlackLightPlusCUDAExt

CUDA methods of the `Execution` device protocol for `CUDADevice`: activation, the
KernelAbstractions backend, the functional probe, page-locked staging buffers, early
release of device arrays, the per-thread stack limit, read-only loads through the
constant cache, and the idle-device selection behind `select_devices`.
"""
module BlackLightPlusCUDAExt

using BlackLightPlus, CUDA
using Adapt: Adapt
using BlackLightPlus.Execution: CUDADevice
import BlackLightPlus.Execution: backend, activate!, functional, stage_alloc, stage_free, free!,
                      prepare_launch!, stack_bytes, readonly, select_devices

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

function prepare_launch!(::CUDADevice, prog)
    bytes = stack_bytes(prog)
    bytes > 0 && CUDA.limit(CUDA.LIMIT_STACK_SIZE) < bytes &&
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, bytes)
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
