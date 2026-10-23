# The device pool: persistent worker tasks ("lanes"), `lanes` per device, pulling queued
# work items from one shared queue; per-lane device buffers and host staging buffers;
# the process-wide registry of pools. Work items are independent and write their results
# by item index, so a result never depends on which lane, device or chunk ran it.

"""
    PoolLane

One persistent worker of a `DevicePool`: a task bound to device `dev` (KernelAbstractions
backend `backend`) that runs the jobs posted to its `inbox`. `bufs` and `stage` cache its
device buffers and host staging buffers by (purpose, element type); only the lane itself
touches them while the pool is open.
"""
mutable struct PoolLane
    const dev::AbstractDevice
    const backend::Any
    const inbox::Channel{Any}
    const bufs::Dict{Any,Any}
    const stage::Dict{Any,Any}
    task::Task
    PoolLane(dev) =
        new(dev, backend(dev), Channel{Any}(Inf), Dict{Any,Any}(), Dict{Any,Any}())
end

# Device twins of host objects on one device, keyed by `objectid` and holding the host
# weakly; every access holds `lock`.
struct ResidencyCache
    lock::ReentrantLock
    twins::Dict{UInt,Tuple{WeakRef,Any}}
end
ResidencyCache() = ResidencyCache(ReentrantLock(), Dict{UInt,Tuple{WeakRef,Any}}())

"""
    DevicePool(devices; lanes = 2, chunk = :auto)

Persistent multi-device runtime over `devices` (device handles or CUDA ordinals): `lanes`
worker tasks per device, spawned once, a shared work queue, and one residency cache per
device ([`resident`](@ref)). `chunk` is the chunk setting of the executors built on the
pool with `PoolExecutor(pool)`: a positive item count, or `:auto`, sized per run
([`chunk_for`](@ref)). Every device must be functional (`BackendUnavailable` otherwise). Run Julia with more threads than devices so the workers of different devices
run concurrently. `close(pool)` stops the workers and releases their buffers; a pool built
here is the caller's to close, a pool obtained through `PoolExecutor(devices)` or
[`resolve`](@ref) belongs to the process registry and is closed at exit.
"""
mutable struct DevicePool
    const devices::Vector{AbstractDevice}
    const lanes::Vector{PoolLane}
    const chunk::Union{Int,Symbol}
    const post_lock::ReentrantLock
    const resident::Vector{ResidencyCache}
    @atomic open::Bool
end

# One unit of queued work: `g(worker, item, job)` for every item of `queue`. `pending`
# counts queued-but-unfinished items, and the worker finishing the last one closes the
# queue; the first failure is kept and the remaining items drain without running.
mutable struct Job
    const g::Any
    const queue::Channel{Any}
    const pending::Threads.Atomic{Int}
    const done::Channel{Nothing}
    @atomic err::Any
end

function Job(g, items, nworkers::Integer)
    job = Job(g, Channel{Any}(Inf), Threads.Atomic{Int}(length(items)),
              Channel{Nothing}(nworkers), nothing)
    foreach(it -> put!(job.queue, it), items)
    return job
end

const LANE_KEY = :blp_pool_lane

"""
    chunk_setting(c) -> Union{Int,Symbol}

A validated chunk setting: a positive item count as `Int`, or `:auto`, which the executor
resolves for every run from its item count ([`chunk_for`](@ref)).
"""
function chunk_setting(c)
    c === :auto && return :auto
    c isa Integer && c >= 1 && return Int(c)
    throw(ArgumentError("chunk must be a positive integer or :auto; got $(repr(c))"))
end

function DevicePool(devices; lanes::Integer = 2, chunk = :auto)
    devs = AbstractDevice[device(d) for d in devices]
    isempty(devs) && throw(ArgumentError("DevicePool: no devices"))
    allunique(devs) || throw(ArgumentError("DevicePool: duplicate devices in $devs"))
    lanes >= 1 || throw(ArgumentError("DevicePool: lanes must be positive"))
    setting = chunk_setting(chunk)
    foreach(check_available, devs)
    Threads.nthreads() > length(devs) ||
        @warn "DevicePool: $(Threads.nthreads()) Julia thread(s) for $(length(devs)) " *
              "device(s); start Julia with -t $(length(devs) + 1) or more so the device " *
              "workers run concurrently" maxlog = 1
    ls = PoolLane[]
    for dev in devs, _ in 1:lanes
        lane = PoolLane(dev)
        lane.task = Threads.@spawn lane_loop(lane)
        push!(ls, lane)
    end
    return DevicePool(devs, ls, setting, ReentrantLock(),
                      [ResidencyCache() for _ in devs], true)
end

Base.isopen(pool::DevicePool) = @atomic pool.open
Base.show(io::IO, pool::DevicePool) =
    print(io, "DevicePool(", join(string.(pool.devices), ", "), "; lanes = ",
          lanes_per_device(pool), ", chunk = ", repr(pool.chunk),
          isopen(pool) ? "" : ", closed", ")")

"Worker tasks per device of `pool`."
lanes_per_device(pool::DevicePool) = length(pool.lanes) ÷ length(pool.devices)

# Position of `dev` in `pool.devices`; the residency caches use the same order.
function device_slot(pool::DevicePool, dev)
    d = findfirst(==(device(dev)), pool.devices)
    d === nothing && throw(ArgumentError("$(device(dev)) is not a device of $pool"))
    return d
end

# The first lane of `dev`; lanes are stored device-major.
first_lane(pool::DevicePool, dev) =
    pool.lanes[(device_slot(pool, dev) - 1) * lanes_per_device(pool) + 1]

# A lane whose device cannot be activated still serves its jobs: it fails each one with
# the activation error and drains its items, so no caller waits forever.
function lane_loop(lane::PoolLane)
    task_local_storage(LANE_KEY, lane)
    init_err = try
        activate!(lane.dev); nothing
    catch e
        CapturedException(e, catch_backtrace())
    end
    while serve_next!(lane, init_err) end
    init_err === nothing && free_stage!(lane)
end

# Take and run the next job of the lane's inbox; false once the inbox is closed and empty.
# The job lives only in this frame, so a finished job (and the host objects its function
# captures) is not kept alive by an idle lane.
@noinline function serve_next!(lane::PoolLane, init_err)
    job = try
        take!(lane.inbox)
    catch e
        e isa InvalidStateException && return false
        rethrow()
    end
    try
        init_err === nothing || @atomicreplace job.err nothing => init_err
        drain!(job, lane)
    finally
        put!(job.done, nothing)
    end
    return true
end

# Release the lane's staging buffers, on the lane, after its last job.
function free_stage!(lane::PoolLane)
    KernelAbstractions.synchronize(lane.backend)
    foreach(s -> stage_free(lane.dev, s[2]), values(lane.stage))
    empty!(lane.stage)
end

# Workers outlive the world age they started in, so job functions defined later run
# through `invokelatest`.
function drain!(job::Job, worker)
    for item in job.queue
        if (@atomic job.err) === nothing
            try
                Base.invokelatest(job.g, worker, item, job)
            catch e
                @atomicreplace job.err nothing => CapturedException(e, catch_backtrace())
            end
        end
        Threads.atomic_sub!(job.pending, 1) == 1 && close(job.queue)
    end
end

"""
    enqueue!(job, item)

Add `item` to a running job from inside one of its items (streaming work such as image
refinement); the job ends once every queued item, including these, has finished.
"""
function enqueue!(job::Job, item)
    Threads.atomic_add!(job.pending, 1)
    put!(job.queue, item)
    return job
end

"The pool lane running the calling task, or `nothing`."
current_lane() = get(task_local_storage(), LANE_KEY, nothing)

function check_caller(pool::DevicePool)
    isopen(pool) || error("DevicePool is closed")
    current_lane() === nothing ||
        error("DevicePool: a pool job cannot be started from inside a pool worker")
end

# Post a job over `items` (in queue order) to `lanes`, wait for all of them, rethrow the
# first failure.
function run_job!(g, pool::DevicePool, lanes, items)
    isempty(items) && return nothing
    check_caller(pool)
    job = Job(g, items, length(lanes))
    @lock pool.post_lock foreach(lane -> put!(lane.inbox, job), lanes)
    foreach(_ -> take!(job.done), lanes)
    err = @atomic job.err
    err === nothing || throw(err)
    return nothing
end

# The same job loop on `nworkers` freshly spawned host tasks (worker argument `nothing`):
# the items are pulled in queue order, and the first failure is rethrown once they drain.
function run_workers!(g, nworkers::Integer, items)
    isempty(items) && return nothing
    job = Job(g, items, nworkers)
    for _ in 1:nworkers
        Threads.@spawn try
            drain!(job, nothing)
        finally
            put!(job.done, nothing)
        end
    end
    foreach(_ -> take!(job.done), 1:nworkers)
    err = @atomic job.err
    err === nothing || throw(err)
    return nothing
end

"""
    foreach_chunk(f, pool, items)

Run `f(lane, item)` for every element of `items` on the pool's lanes, pulled from one
shared queue in the given order (the first items start first), and return when all have
finished. An exception in any `f` is rethrown here, as a `CapturedException`, after the
queue drains.
"""
foreach_chunk(f, pool::DevicePool, items) =
    run_job!((l, it, _) -> f(l, it), pool, pool.lanes, items)

"""
    stream_chunks(f, pool, items)

As [`foreach_chunk`](@ref), but `f(lane, item, job)` may [`enqueue!`](@ref)`(job, new)`
while running.
"""
stream_chunks(f, pool::DevicePool, items) = run_job!(f, pool, pool.lanes, items)

"""
    foreach_device(f, pool) -> Vector

Run `f(lane)` once per device, concurrently, on the first lane of each device; returns the
results in `pool.devices` order.
"""
function foreach_device(f, pool::DevicePool)
    check_caller(pool)
    out = Vector{Any}(undef, length(pool.devices))
    @sync for (d, dev) in enumerate(pool.devices)
        lane = first_lane(pool, dev)
        Threads.@spawn run_job!((l, _, _) -> (out[d] = f(l)), pool, [lane], (d,))
    end
    return out
end

# Run `f()` with `dev` active: directly on a lane of `dev`, else as a job on its first lane.
function on_device(f, pool::DevicePool, dev)
    lane = current_lane()
    lane !== nothing && lane.dev == device(dev) && any(l -> l === lane, pool.lanes) &&
        return f()
    out = Ref{Any}(nothing)
    run_job!((_, _, _) -> (out[] = f(); nothing), pool, [first_lane(pool, dev)], (1,))
    return out[]
end

function Base.close(pool::DevicePool)
    (@atomicswap pool.open = false) || return nothing
    foreach(lane -> close(lane.inbox), pool.lanes)
    foreach(lane -> wait(lane.task), pool.lanes)
    for lane in pool.lanes
        foreach(b -> free!(lane.dev, b), values(lane.bufs))
        empty!(lane.bufs); empty!(lane.stage)
    end
    for cache in pool.resident
        @lock cache.lock empty!(cache.twins)
    end
    return nothing
end

# ---------------------------------------------------------------- registry
const POOLS = Dict{Any,DevicePool}()
const POOLS_LOCK = ReentrantLock()

"""
    device_pool(devices; lanes = 2) -> DevicePool

The process-wide pool of the device list `devices` with `lanes` workers per device and
the chunk setting `:auto`, created on first use and reused after (a closed pool is
replaced), so repeated runs share its workers, staging buffers and residency caches
whatever chunk their executors use. A `DevicePool` is returned unchanged.
"""
device_pool(pool::DevicePool; kw...) = pool
function device_pool(devices; lanes::Integer = 2)
    devs = AbstractDevice[device(d) for d in devices]
    key = (Tuple(devs), Int(lanes))
    return @lock POOLS_LOCK begin
        p = get(POOLS, key, nothing)
        (p === nothing || !isopen(p)) && (p = POOLS[key] = DevicePool(devs; lanes))
        p
    end
end

"Close every registered pool (run at exit)."
close_pools() = foreach(close, @lock(POOLS_LOCK, collect(values(POOLS))))

# ---------------------------------------------------------------- lane buffers
"""
    lane_buffer(lane, key, T, n) -> device vector

Device buffer of at least `n` elements of `T` for purpose `key`, reused across chunks;
a buffer that is too small is released and replaced. Call only from the lane's task.
"""
function lane_buffer(lane::PoolLane, key, ::Type{T}, n::Integer) where {T}
    b = get(lane.bufs, (key, T), nothing)
    if b === nothing || length(b) < n
        b === nothing || free!(lane.dev, b)
        b = lane.bufs[(key, T)] = KernelAbstractions.allocate(lane.backend, T, n)
    end
    return b
end

"""
    lane_array(lane, key, T, dims) -> device array

Device array of exactly `dims` for purpose `key` (per-lane scratch such as a trajectory
log), reused while `dims` stays the same. Call only from the lane's task.
"""
function lane_array(lane::PoolLane, key, ::Type{T}, dims::Dims) where {T}
    b = get(lane.bufs, (key, T, length(dims)), nothing)
    if b === nothing || size(b) != dims
        b === nothing || free!(lane.dev, b)
        b = lane.bufs[(key, T, length(dims))] =
            KernelAbstractions.allocate(lane.backend, T, dims)
    end
    return b
end

"""
    lane_stage(lane, key, T, n) -> Vector{T}

Host staging buffer ([`stage_alloc`](@ref)) of at least `n` elements of `T` for purpose
`key`. A buffer that is too small is released once the lane's device work has drained,
so no copy still reads it. Call only from the lane's task.
"""
function lane_stage(lane::PoolLane, key, ::Type{T}, n::Integer) where {T}
    s = get(lane.stage, (key, T), nothing)
    s === nothing || length(s[1]) >= n && return s[1]::Vector{T}
    if s !== nothing
        KernelAbstractions.synchronize(lane.backend)
        stage_free(lane.dev, s[2])
    end
    s = lane.stage[(key, T)] = stage_alloc(lane.dev, T, n)
    return s[1]::Vector{T}
end
