# Device handles and the device protocol: KernelAbstractions backend, activation,
# staging buffers, release, lane queues and shared arrays, launch preparation and device
# selection. The methods for `CUDADevice` come from the package extension; the fallbacks
# here serve `HostDevice` and report a missing extension for everything else.

"""
    AbstractDevice

A place where a `PoolExecutor` launches kernels. A device answers [`backend`](@ref),
[`activate!`](@ref), [`functional`](@ref), [`stage_alloc`](@ref)/[`stage_free`](@ref),
[`free!`](@ref), [`free_bytes`](@ref), [`share_lanes!`](@ref), [`lane_queue`](@ref),
[`sync_queue`](@ref), [`queue_idle`](@ref), [`prepare_launch!`](@ref),
[`launch_capped!`](@ref), [`readonly`](@ref) and [`wavefront`](@ref). Devices compare by
value, so two handles of the same device are interchangeable.
"""
abstract type AbstractDevice end

"""
    HostDevice(id)

Logical device on the KernelAbstractions `CPU()` backend. A pool over several
`HostDevice`s runs the device code path (uploads, chunked launches, staging, gathers)
on the host, one logical device per `id`; this is how the pool is tested without a GPU.
"""
struct HostDevice <: AbstractDevice
    id::Int
end

"""
    CUDADevice(ordinal)

The CUDA device with zero-based `ordinal` as numbered under `CUDA_VISIBLE_DEVICES`.
Usable once `using CUDA` has loaded the CUDA extension. An integer is accepted wherever a
device is and means `CUDADevice(ordinal)`.
"""
struct CUDADevice <: AbstractDevice
    ordinal::Int
end

"""
    BackendUnavailable(msg)

Thrown when an explicitly requested backend or device cannot be used. `msg` names the
cause and the remedy (`using CUDA`, `CUDA_VISIBLE_DEVICES`, or `backend!(:cpu)`).
"""
struct BackendUnavailable <: Exception
    msg::String
end
Base.showerror(io::IO, e::BackendUnavailable) = print(io, "BackendUnavailable: ", e.msg)

"""
    device(x) -> AbstractDevice

The device handle of `x`: a device as given, an integer as `CUDADevice(x)`.
"""
device(d::AbstractDevice) = d
device(ordinal::Integer) = CUDADevice(ordinal)

"""
    cuda_loaded() -> Bool

Whether the CUDA extension of the package is loaded, i.e. whether `using CUDA` has run in
this session.
"""
cuda_loaded() = Base.get_extension(parentmodule(@__MODULE__), :BlackLightPlusCUDAExt) !== nothing

# The error for a device whose methods are missing or whose hardware is unusable.
function unavailable(dev::CUDADevice)
    cuda_loaded() || return BackendUnavailable(
        "$(dev) needs the CUDA extension: run `using CUDA` first, or choose the CPU " *
        "with `backend!(:cpu)`")
    return BackendUnavailable(
        "$(dev) is not functional or not visible: check the driver and " *
        "`CUDA_VISIBLE_DEVICES`, or choose the CPU with `backend!(:cpu)`")
end
unavailable(dev::AbstractDevice) = BackendUnavailable(
    "$(dev) has no device methods loaded; choose the CPU with `backend!(:cpu)`")

"""
    backend(dev) -> KernelAbstractions.Backend

The KernelAbstractions backend that runs kernels on `dev`: `CPU()` for a `HostDevice`,
`CUDABackend()` for a `CUDADevice` (CUDA extension). Throws `BackendUnavailable` when the
device's methods are not loaded. `backend()` without arguments is the session executor.
"""
backend(::HostDevice) = KernelAbstractions.CPU()
backend(dev::AbstractDevice) = throw(unavailable(dev))

"""
    activate!(dev)

Make `dev` the current device of the calling task. Pool workers call it once, before
their first job; a `HostDevice` needs nothing.
"""
activate!(::HostDevice) = nothing
activate!(dev::AbstractDevice) = throw(unavailable(dev))

"""
    functional(dev) -> Bool
    functional(::Type{<:AbstractDevice}) -> Bool

Whether `dev` can run kernels now; with a device type, whether its runtime is usable at
all (`CUDA.functional()` for `CUDADevice`). A `HostDevice` is always functional; a
`CUDADevice` is not until the CUDA extension is loaded.
"""
functional(::HostDevice) = true
functional(::AbstractDevice) = false
functional(::Type{HostDevice}) = true
functional(::Type{<:AbstractDevice}) = false

# Throw `BackendUnavailable` unless `dev` is functional.
check_available(dev::AbstractDevice) = functional(dev) ? dev : throw(unavailable(dev))

"""
    stage_alloc(dev, T, n) -> (buffer::Vector{T}, handle)

A host staging buffer of `n` elements of `T` for copies to and from `dev`, and the handle
that releases it through [`stage_free`](@ref). Page-locked memory on a `CUDADevice`, freed
explicitly rather than by a finalizer; a plain `Vector` (handle `nothing`) elsewhere.
"""
stage_alloc(::AbstractDevice, ::Type{T}, n::Integer) where {T} =
    (Vector{T}(undef, n), nothing)

"""
    stage_free(dev, handle)

Release a staging buffer obtained from [`stage_alloc`](@ref). The caller guarantees that
no copy still reads or writes it.
"""
stage_free(::AbstractDevice, ::Nothing) = nothing

"""
    free!(dev, x)

Release the device memory of the array `x` now instead of at garbage collection
(`CUDA.unsafe_free!` on a `CUDADevice`); a no-op for host arrays.
"""
free!(::AbstractDevice, x) = nothing

"""
    free_bytes(dev) -> Int

Bytes of memory free for allocation on `dev` now: the driver's free memory of a
`CUDADevice` (CUDA extension), `0` on every other device. The cascade grants child slots
from an eighth of it ([`foreach_cascade!`](@ref)); `0` turns that budget off.
"""
free_bytes(::AbstractDevice) = 0

"""
    share_lanes!(dev, a) -> a

Let every lane of `dev` read the device array `a` without waiting for the lane that last
used it: on a `CUDADevice` (CUDA extension) wait for the stream that wrote `a` and exempt
`a` from the implicit stream synchronisation of the CUDA package
(`CUDA.enable_synchronization!(a, false)`), so that launches of the lanes that read it
overlap; the identity on every other device. Only for an array that is complete
and never written afterwards: the residency cache shares each upload
([`lane_sharing!`](@ref)), and a shared array is freed only after
[`sync_lanes`](@ref).
"""
share_lanes!(::AbstractDevice, a) = a

"""
    lane_queue(dev) -> queue

The queue that the launches of the calling task on `dev` run on: the task's stream on a
`CUDADevice` (CUDA extension), `nothing` on every other device, whose launches have
finished when they return. A pool lane records it once its device is active.
"""
lane_queue(::AbstractDevice) = nothing

"""
    sync_queue(dev, q)

Wait until the work queued on the queue `q` of `dev` ([`lane_queue`](@ref)) has finished.
"""
sync_queue(::AbstractDevice, q) = nothing

"""
    queue_idle(dev, q) -> Bool

Whether the queue `q` of `dev` has no unfinished work (`CUDA.isdone` on a `CUDADevice`);
`true` on every other device.
"""
queue_idle(::AbstractDevice, q) = true

const LANE_SHARING = Threads.Atomic{Bool}(true)
const LANE_CHECK = Threads.Atomic{Bool}(false)

"""
    lane_sharing!(on::Bool) -> Bool

Whether the residency cache shares the device arrays it uploads from now on between the
lanes of their device ([`share_lanes!`](@ref)); on by default. Off, every device array
keeps the implicit stream synchronisation of the CUDA package, and the lanes of a device
do not overlap launches that read the same array (the form compared against). Arrays already uploaded
keep their setting. Returns the previous setting; `lane_sharing()` reads it.
"""
lane_sharing!(on::Bool) = Threads.atomic_xchg!(LANE_SHARING, on)
lane_sharing() = LANE_SHARING[]

"""
    lane_check!(on::Bool) -> Bool

The lane check, a debug switch, off by default: while on, every [`gather!`](@ref) of
accumulator or child twins, which frees their device arrays without waiting for the
lanes, first asserts that the queue of every lane of each twin's device is idle and
throws an `ErrorException` naming the busy lane otherwise. An eviction from a residency
cache waits for every lane of the device ([`sync_lanes`](@ref)) before it frees and is
not checked; a lane frees its own buffers while the other lanes run. Returns the
previous setting; `lane_check()` reads it.
"""
lane_check!(on::Bool) = Threads.atomic_xchg!(LANE_CHECK, on)
lane_check() = LANE_CHECK[]

"""
    wavefront(dev) -> Bool

Whether the Monte Carlo phases that the device wavefront serves run on it on `dev`
([`foreach_wavefront!`](@ref)): `true` for a `CUDADevice`, whose warps lose their lanes
to the spread of the photons' step counts; `false` for a `HostDevice`, whose pool runs
the cascade with the compiled programs of the threads. A pool runs the wavefront only
when every one of its devices answers `true`.
"""
wavefront(::AbstractDevice) = false
wavefront(::CUDADevice) = true

"""
    prepare_launch!(dev, prog)

Prepare `dev` for launching the program `prog`: on a `CUDADevice`, raise the per-thread
stack limit to at least [`stack_bytes`](@ref)`(prog)`. Called before every launch; a no-op
on other devices.
"""
prepare_launch!(::AbstractDevice, prog) = nothing

"""
    stack_bytes(prog) -> Int

Per-thread device stack, in bytes, that the program `prog` needs; `0` (the default) keeps
the device default. A program whose inlined body overflows the default stack (the
closed-form Kerr tracer needs 32 KiB) overrides it.
"""
stack_bytes(prog) = 0

"""
    readonly(dev, a)

The read-only view of the device array `a` that a program reads through: the
`KernelAbstractions` constant-memory wrapper (the one `@Const` applies to kernel arguments),
carried inside program structs. On a `HostDevice` it is `KernelAbstractions.constify(a)`;
the CUDA extension supplies the `CUDADevice` method; other devices get `a` unchanged.
"""
readonly(::AbstractDevice, a) = a
readonly(::HostDevice, a::Array) = KernelAbstractions.constify(a)

"""
    upload(dev, x::AbstractArray) -> device array

A copy of the host array `x` in the memory of `dev` (a new `Array` on a `HostDevice`).
Runs on the calling task, which must have `dev` active.
"""
function upload(dev::AbstractDevice, x::AbstractArray)
    isbitstype(eltype(x)) ||
        throw(ArgumentError("a device copy needs isbits elements; got eltype $(eltype(x))"))
    return copyto!(KernelAbstractions.allocate(backend(dev), eltype(x), size(x)), x)
end

"""
    select_devices(; candidates, util_threshold = 15, min_free_gb = 8, fallback = false)
        -> Vector{CUDADevice}

The idle CUDA devices among `candidates` (default: every visible ordinal): free memory of
at least `min_free_gb` GB and, when NVML reports it, compute utilisation of at most
`util_threshold` percent. With `fallback = true` an empty selection becomes the first
visible candidate. Without the CUDA extension, or when no device is visible, the result
is empty.
"""
select_devices(; kw...) = select_devices(CUDADevice; kw...)
select_devices(::Type{<:AbstractDevice}; kw...) = CUDADevice[]
