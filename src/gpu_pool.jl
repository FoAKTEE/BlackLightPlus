# gpu_pool.jl — the multi-device runtime behind every GPU driver (results/blp/multigpu/
# design.md §5.1). A `DevicePool` owns persistent worker tasks ("lanes", `Threads.@spawn`,
# so host work runs on separate OS threads), `lanes` per device, each with its own device
# stream (CUDA streams are task-local) and cached device / pinned-host staging buffers.
# Work is a queue of chunks that the lanes pull dynamically; outputs are written by item
# index, so results do not depend on which lane ran which chunk. Device copies of large
# host inputs (fluid grids, AMR meshes, ADM grids) are cached per pool and uploaded once,
# concurrently on every device. CUDA optional: included by the GPU driver files.

using StaticArrays
using KernelAbstractions
using CUDA

# ---------------------------------------------------------------- device handles
"""
    HostDevice(id)

Logical device on the KernelAbstractions `CPU()` backend. A `devices` list of
`HostDevice`s in place of CUDA ordinals runs a multi-device driver's pool, uploads and
launches on the host, one logical device per entry.
"""
struct HostDevice
    id::Int
end

# Device handle -> activation, KA backend, host->device copy. CUDA ordinals keep the
# CUDA.device! / CUDABackend / CuArray path; HostDevice maps to CPU() and a host copy.
_device!(dev::Integer) = CUDA.device!(dev)
_device!(::HostDevice) = nothing
_backend(::Integer) = CUDABackend()
_backend(::HostDevice) = CPU()
_upload(backend, x::AbstractArray) =
    copyto!(KernelAbstractions.allocate(backend, eltype(x), size(x)), x)

# Page-locked host memory for the per-chunk staging copies (pageable elsewhere).
_pin(backend, v::Vector) = v
_pin(::CUDABackend, v::Vector) = CUDA.pin(v)

"""
    select_devices(candidates=[0,1]; util_threshold=15, min_free_gb=8) -> Vector{Int}

Probe each visible CUDA device in `candidates` (correct under `CUDA_VISIBLE_DEVICES`
masking) and return the idle subset: free memory ≥ `min_free_gb` GB and, when NVML is
available, compute utilization ≤ `util_threshold` %. Always returns at least one device;
any probe failure degrades safely to the caller's static candidate list. The default
pool (`default_pool`) calls it once per process.
"""
function select_devices(candidates = [0, 1]; util_threshold::Real = 15, min_free_gb::Real = 8)
    vis = try
        filter(d -> 0 <= d < CUDA.ndevices(), candidates)
    catch
        return collect(candidates)
    end
    isempty(vis) && return collect(candidates)
    idle = Int[]
    for d in vis
        ok = try
            CUDA.device!(d)
            free, _ = CUDA.memory_info()
            # NVML compute util is a fraction in [0,1] (0.26 == 26%); -1 => unknown (ignore).
            u = try
                Float64(CUDA.NVML.utilization_rates(CUDA.NVML.Device(CUDA.uuid(CUDA.CuDevice(d)))).compute) * 100.0
            catch
                -1.0
            end
            (free / 1024^3 >= min_free_gb) && (u < 0 || u <= util_threshold)
        catch
            false
        end
        ok && push!(idle, d)
    end
    return isempty(idle) ? vis[1:1] : idle
end

# ---------------------------------------------------------------- pool, lanes, jobs
"""
    PoolLane

One persistent worker of a `DevicePool`: a task bound to device `dev` (backend `backend`)
that runs the jobs posted to its `inbox`. `bufs` / `stage` cache its device buffers and
pinned host staging buffers by (purpose, element type); only the lane touches them.
"""
mutable struct PoolLane
    const dev::Any
    const backend::Any
    const inbox::Channel{Any}
    const bufs::Dict{Any,Any}
    const stage::Dict{Any,Any}
    task::Task
    PoolLane(dev) = new(dev, _backend(dev), Channel{Any}(Inf), Dict{Any,Any}(), Dict{Any,Any}())
end

"""
    DevicePool(devices; lanes=2, chunk=65536)

Persistent multi-device runtime over `devices` (CUDA ordinals or `HostDevice`s): `lanes`
worker tasks per device, spawned once, and a work queue of `chunk`-item chunks.
Drivers accept a pool wherever they take `devices`; `device_pool(devices)` returns the
process-wide pool of a device list, so repeated renders reuse its workers, staging
buffers and resident uploads. Run Julia with `-t ≥ length(devices) + 1` so the workers
of different devices execute concurrently. `close(pool)` stops the workers and frees
their buffers; pools from `device_pool` are closed at exit, a pool built here is the
caller's to close.
"""
mutable struct DevicePool
    const devices::Vector{Any}
    const lanes::Vector{PoolLane}
    const chunk::Int
    const post_lock::ReentrantLock
    const resident_lock::ReentrantLock
    const resident::Dict{UInt,Tuple{WeakRef,Dict{Any,Any}}}
    @atomic open::Bool
end

# One unit of queued work: `g(lane, item, job)` for every item of `queue`, run by the lanes
# it was posted to. `pending` counts queued-but-unfinished items (the lane finishing the
# last one closes the queue); the first failure is kept and the remaining items drain.
mutable struct PoolJob
    const g::Any
    const queue::Channel{Any}
    const pending::Threads.Atomic{Int}
    const done::Channel{Nothing}
    @atomic err::Any
end

const _LANE_KEY = :blp_pool_lane

function DevicePool(devices; lanes::Integer = 2, chunk::Integer = 65536)
    devs = collect(Any, devices)
    isempty(devs) && error("DevicePool: no devices")
    allunique(devs) || error("DevicePool: duplicate devices in $devs")
    (lanes >= 1 && chunk >= 1) || error("DevicePool: lanes and chunk must be positive")
    Threads.nthreads() > length(devs) ||
        @warn "DevicePool: $(Threads.nthreads()) Julia thread(s) for $(length(devs)) " *
              "device(s); start Julia with -t $(length(devs) + 1) or more so the device " *
              "workers run concurrently" maxlog = 1
    ls = PoolLane[]
    for dev in devs, _ in 1:lanes
        lane = PoolLane(dev)
        lane.task = Threads.@spawn _lane_loop(lane)
        push!(ls, lane)
    end
    return DevicePool(devs, ls, Int(chunk), ReentrantLock(), ReentrantLock(),
                      Dict{UInt,Tuple{WeakRef,Dict{Any,Any}}}(), true)
end

Base.isopen(pool::DevicePool) = @atomic pool.open
Base.show(io::IO, pool::DevicePool) =
    print(io, "DevicePool(", pool.devices, "; lanes=", length(pool.lanes) ÷ length(pool.devices),
          ", chunk=", pool.chunk, isopen(pool) ? "" : ", closed", ")")

# A lane whose device cannot be activated still serves its jobs: it fails each one with
# the activation error (and drains its items), so no caller waits forever.
function _lane_loop(lane::PoolLane)
    task_local_storage(_LANE_KEY, lane)
    init_err = try
        _device!(lane.dev); nothing
    catch e
        CapturedException(e, catch_backtrace())
    end
    for job in lane.inbox
        try
            init_err === nothing || @atomicreplace job.err nothing => init_err
            _work!(job, lane)
        finally
            put!(job.done, nothing)
        end
    end
end

# Workers outlive the world age they started in, so job functions (defined later) run
# through `invokelatest`.
function _work!(job::PoolJob, lane::PoolLane)
    for item in job.queue
        if (@atomic job.err) === nothing
            try
                Base.invokelatest(job.g, lane, item, job)
            catch e
                @atomicreplace job.err nothing => CapturedException(e, catch_backtrace())
            end
        end
        Threads.atomic_sub!(job.pending, 1) == 1 && close(job.queue)
    end
end

"""
    enqueue!(job, item)

Add `item` to a running job from inside one of its items (streaming work such as AMR
refinement); the job ends once every queued item, including these, has finished.
"""
function enqueue!(job::PoolJob, item)
    Threads.atomic_add!(job.pending, 1)
    put!(job.queue, item)
    return job
end

_in_lane() = haskey(task_local_storage(), _LANE_KEY)
_check_caller(pool) = (isopen(pool) || error("DevicePool is closed");
    _in_lane() && error("DevicePool: a pool job cannot be started from inside a pool worker"))

# Post a job over `items` (in queue order) to `lanes`, wait for all of them, rethrow the
# first failure.
function _run!(g, pool::DevicePool, lanes, items)
    isempty(items) && return nothing
    _check_caller(pool)
    job = PoolJob(g, Channel{Any}(Inf), Threads.Atomic{Int}(length(items)),
                  Channel{Nothing}(length(lanes)), nothing)
    foreach(it -> put!(job.queue, it), items)
    @lock pool.post_lock foreach(lane -> put!(lane.inbox, job), lanes)
    foreach(_ -> take!(job.done), lanes)
    err = @atomic job.err
    err === nothing || throw(err)
    return nothing
end

"""
    foreach_chunk(f, pool, items)

Run `f(lane, item)` for every element of `items` on the pool's lanes, pulled from one
shared queue in the given order (the first items start first), and return when all have
finished. An exception in any `f` is rethrown here after the queue drains.
"""
foreach_chunk(f, pool::DevicePool, items) = _run!((l, it, _) -> f(l, it), pool, pool.lanes, items)

"""
    stream_chunks(f, pool, items)

As `foreach_chunk`, but `f(lane, item, job)` may `enqueue!(job, new_item)` while running.
"""
stream_chunks(f, pool::DevicePool, items) = _run!(f, pool, pool.lanes, items)

"""
    foreach_device(f, pool) -> Vector

Run `f(lane)` once per device, concurrently, on the first lane of each device; returns
the results in `pool.devices` order.
"""
function foreach_device(f, pool::DevicePool)
    _check_caller(pool)
    nl = length(pool.lanes) ÷ length(pool.devices)
    out = Vector{Any}(undef, length(pool.devices))
    firsts = pool.lanes[1:nl:end]
    @sync for (di, lane) in enumerate(firsts)
        Threads.@spawn _run!((l, _, _) -> (out[di] = f(l)), pool, [lane], (di,))
    end
    return out
end

function Base.close(pool::DevicePool)
    (@atomicswap pool.open = false) || return nothing
    foreach(lane -> close(lane.inbox), pool.lanes)
    foreach(lane -> wait(lane.task), pool.lanes)
    for lane in pool.lanes
        foreach(_free!, values(lane.bufs)); empty!(lane.bufs); empty!(lane.stage)
    end
    @lock pool.resident_lock empty!(pool.resident)
    return nothing
end

_free!(x) = nothing
_free!(x::CuArray) = CUDA.unsafe_free!(x)

# ---------------------------------------------------------------- pool registry
const _POOLS = Dict{Any,DevicePool}()
const _POOLS_LOCK = ReentrantLock()
const _DEFAULT_DEVICES = Ref{Any}(nothing)

"""
    device_pool(devices) -> DevicePool

The process-wide pool of the device list `devices` (created on first use, reused after);
a `DevicePool` is returned unchanged.
"""
device_pool(pool::DevicePool) = pool
function device_pool(devices)
    key = Tuple(devices)
    return @lock _POOLS_LOCK begin
        p = get(_POOLS, key, nothing)
        (p === nothing || !isopen(p)) && (p = _POOLS[key] = DevicePool(collect(Any, devices)))
        p
    end
end

"""
    default_pool() -> DevicePool

The pool the drivers use when no `devices` are given: `select_devices()`, probed once.
"""
function default_pool()
    devs = @lock _POOLS_LOCK begin
        _DEFAULT_DEVICES[] === nothing && (_DEFAULT_DEVICES[] = select_devices())
        _DEFAULT_DEVICES[]
    end
    return device_pool(devs)
end

"Close every registered pool (run at exit)."
close_pools() = foreach(close, @lock(_POOLS_LOCK, collect(values(_POOLS))))
atexit(close_pools)

# Device handles of a `devices` argument (a list or a pool).
device_list(devices) = collect(Any, devices)
device_list(pool::DevicePool) = pool.devices

# ---------------------------------------------------------------- uploads and residency
"""
    upload_per_device(x, devices) -> Dict

One device copy of host array `x` per device of `devices` (a list or a pool), keyed by
device handle; the copies are made concurrently, one per device worker.
"""
function upload_per_device(x::AbstractArray, devices)
    devs = device_list(devices)
    isempty(devs) && return Dict{Any,Any}()
    pool = device_pool(devices)
    ups = foreach_device(lane -> _upload(lane.backend, x), pool)
    return Dict(pool.devices[i] => ups[i] for i in eachindex(ups))
end

"""
    resident(build, pool, host, tag)

Device-side object built from the host array `host` (e.g. a fluid grid's data), cached
per `(host, tag)` for the life of the pool: `build()` runs on the first request only.
The cache holds `host` weakly and assumes it is not mutated in place afterwards.
"""
function resident(build, pool::DevicePool, host::AbstractArray, tag)
    return @lock pool.resident_lock begin
        for (k, (w, _)) in collect(pool.resident)            # drop entries of freed hosts
            w.value === nothing && delete!(pool.resident, k)
        end
        id = objectid(host)
        slot = get(pool.resident, id, nothing)
        if slot === nothing || slot[1].value !== host
            slot = pool.resident[id] = (WeakRef(host), Dict{Any,Any}())
        end
        get!(build, slot[2], tag)
    end
end

# ---------------------------------------------------------------- staging buffers
# Device buffer of `n` or more elements of `T` for `key`, reused across chunks.
function lane_buffer(lane::PoolLane, key, ::Type{T}, n::Integer) where {T}
    b = get(lane.bufs, (key, T), nothing)
    if b === nothing || length(b) < n
        b === nothing || _free!(b)
        b = lane.bufs[(key, T)] = KernelAbstractions.allocate(lane.backend, T, n)
    end
    return b
end

# Device array of exactly `dims` for `key` (per-lane scratch, e.g. trajectory logs).
function lane_array(lane::PoolLane, key, ::Type{T}, dims::Dims) where {T}
    b = get(lane.bufs, (key, T, length(dims)), nothing)
    if b === nothing || size(b) != dims
        b === nothing || _free!(b)
        b = lane.bufs[(key, T, length(dims))] = KernelAbstractions.allocate(lane.backend, T, dims)
    end
    return b
end

# Pinned host staging buffer of `n` or more elements of `T` for `key`.
function lane_stage(lane::PoolLane, key, ::Type{T}, n::Integer) where {T}
    s = get(lane.stage, (key, T), nothing)
    (s === nothing || length(s) < n) &&
        (s = lane.stage[(key, T)] = _pin(lane.backend, Vector{T}(undef, n)))
    return s
end

# ---------------------------------------------------------------- per-ray march executor
"""
    HostRays(rays, T=eltype(eltype(rays)))

Ray source of host-built initial states: a chunk gathers `rays[item]` (converted to
`SVector{8,T}`) into pinned staging and copies it to the device.
"""
struct HostRays{T,V}
    rays::V
end
HostRays(rays::AbstractVector{<:SVector{8}}, ::Type{T} = eltype(eltype(rays))) where {T} =
    HostRays{T,typeof(rays)}(rays)

function rays_on_device!(lane::PoolLane, src::HostRays{T}, idx) where {T}
    n = length(idx)
    st = lane_stage(lane, :rays, SVector{8,T}, n)
    @inbounds for (q, i) in enumerate(idx)
        st[q] = SVector{8,T}(src.rays[i])
    end
    d = lane_buffer(lane, :rays, SVector{8,T}, n)
    copyto!(d, 1, st, 1, n)
    return d
end

"""
    MarchPart(items, rays, launch)

One kernel's share of a `march!`: the output indices `items` (in queue order), their ray
source `rays` (`HostRays` or a device generator) and `launch(lane, outs_d, y0_d, n)`,
which launches the kernel for the first `n` rays of `y0_d` into the device outputs
`outs_d` on `lane`'s device.
"""
struct MarchPart{I<:AbstractVector{<:Integer},S,L}
    items::I
    rays::S
    launch::L
end

"""
    march!(outs::Tuple, pool, parts)

Run every `MarchPart` over the pool: each part's `items` split into chunks of
`pool.chunk`, queued part by part in item order. A chunk makes its rays on the device,
launches, copies the outputs back through pinned staging and stores output `k` of item
`i` at `outs[k][i]`. Items are independent, so the result does not depend on the device
count, the chunking or the queue order.
"""
function march!(outs::Tuple, pool::DevicePool, parts)
    chunks = [(p, r) for (p, part) in enumerate(parts)
              for r in Iterators.partition(eachindex(part.items), pool.chunk)]
    foreach_chunk(pool, chunks) do lane, (p, r)
        part = parts[p]
        idx = view(part.items, r)
        n = length(idx)
        y0d = rays_on_device!(lane, part.rays, idx)
        outd = ntuple(k -> lane_buffer(lane, (:out, k), eltype(outs[k]), n), length(outs))
        part.launch(lane, outd, y0d, n)
        KernelAbstractions.synchronize(lane.backend)
        for k in eachindex(outs)
            st = lane_stage(lane, (:out, k), eltype(outs[k]), n)
            copyto!(st, 1, outd[k], 1, n)
            o = outs[k]
            @inbounds for (q, i) in enumerate(idx)
                o[i] = st[q]
            end
        end
    end
    return outs
end

"""
    queue_items(order, n) -> AbstractVector{Int}

Queue order of `n` items: `1:n`, or the permutation `order` (validated).
"""
function queue_items(order, n::Integer)
    order === nothing && return 1:n
    (length(order) == n && isperm(order)) ||
        error("render `order` must be a permutation of 1:$n")
    return order
end
