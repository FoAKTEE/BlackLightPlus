# Residency: a host object as a device sees it, built by `Adapt` with every host array
# replaced by its copy on the device. Array copies are cached per pool and device by the
# identity of the host array, which the cache holds weakly: a dropped host array frees
# its entries once every lane of the device has synchronised. A cached host array is not
# mutated in place unless its copies are dropped first ([`invalidate!`](@ref), as the
# products do for the arrays a caller gives them to overwrite), and a copy is only read:
# the lanes of a device share it. The program twins of a call are per lane.

"""
    DeviceAdaptor(pool, dev, readonly)

The `Adapt` adaptor that turns a host object into its twin on `dev`: each `Array` leaf
becomes its cached copy on `dev` (wrapped by [`readonly`](@ref) when `readonly` is true).
A type whose device twin is not "its arrays copied" (an accumulator that must be merged
back, a buffer swapped per chunk) defines `Adapt.adapt_structure(::DeviceAdaptor, x)`.
"""
struct DeviceAdaptor{D<:AbstractDevice}
    pool::DevicePool
    dev::D
    readonly::Bool
end

function Adapt.adapt_storage(to::DeviceAdaptor, a::Array)
    t = cached_twin(to.pool, to.dev, a)
    return to.readonly ? readonly(to.dev, t) : t
end

# Evict the entries of the residency cache of `dev` whose host object has been collected.
evict!(pool::DevicePool, dev::AbstractDevice, cache::ResidencyCache) =
    drop_twins!(w -> w.value === nothing, pool, dev, cache)

# Drop the entries of the residency cache of `dev` whose weak reference to the host object
# satisfies `pred`: wait for every lane of the device, then free their device arrays. The
# wait is the guarantee, so the lane check does not apply: a lane may queue new work as
# soon as it returns. Runs under the cache's lock.
function drop_twins!(pred, pool::DevicePool, dev::AbstractDevice, cache::ResidencyCache)
    hit = [k for (k, (w, _)) in cache.twins if pred(w)]
    isempty(hit) && return cache
    sync_lanes(pool, dev)
    for k in hit
        free!(dev, cache.twins[k][2])
        delete!(cache.twins, k)
    end
    return cache
end

"""
    invalidate!(arrays) -> nothing
    invalidate!(pool, arrays) -> pool

Drop the device copies of every host array that shares memory with one of `arrays`
(`Base.mightalias`: the array itself, a `vec`, `reshape` or view of it) from the residency
caches of `pool`, or of every pool whose caches hold a copy, before the caller overwrites
`arrays` in place: on each device holding such a copy, every lane synchronised, then the
copies freed. The next upload of such an array copies its new values. Call it outside the
pools' lanes, while no run reads `arrays`.
"""
function invalidate!(pool::DevicePool, arrays)
    isopen(pool) || return pool
    stale(w) = (h = w.value; h !== nothing && any(a -> Base.mightalias(h, a), arrays))
    for (d, dev) in enumerate(pool.devices)
        cache = pool.resident[d]
        @lock cache.lock drop_twins!(stale, pool, dev, cache)
    end
    return pool
end

function invalidate!(arrays)
    foreach(p -> invalidate!(p, arrays), lock(() -> collect(keys(CACHING_POOLS)), CACHING_POOLS))
    return nothing
end

# The pools that have uploaded a host array to a residency cache, held weakly: the pools
# `invalidate!(arrays)` visits, whether registered or built by the caller.
const CACHING_POOLS = WeakKeyDict{DevicePool,Nothing}()

# The copy of `host` on `dev`, uploaded on first request and shared by the lanes of `dev`
# (`share_lanes!`, while `lane_sharing()`). Runs on a task with `dev` active.
function cached_twin(pool::DevicePool, dev::AbstractDevice, host::Array)
    cache = pool.resident[device_slot(pool, dev)]
    return @lock cache.lock begin
        id = objectid(host)
        slot = get(cache.twins, id, nothing)
        if slot === nothing || slot[1].value !== host
            evict!(pool, dev, cache)
            t = upload(dev, host)
            lane_sharing() && share_lanes!(dev, t)
            slot = cache.twins[id] = (WeakRef(host), t)
            CACHING_POOLS[pool] = nothing
        end
        slot[2]
    end
end

"""
    resident(pool, dev, obj) -> device twin

The device-resident form of the host object `obj` on device `dev` of `pool` (`pool` may
also be a `PoolExecutor`; `dev` a handle or an ordinal): an array's cached device copy,
or for any other object `Adapt.adapt(DeviceAdaptor(pool, dev, false), obj)`, i.e. the
same structure with each array replaced by its cached copy. The same host array always
yields the same copy on a given device, each device its own. Uploads run on a lane of
`dev`; calling from a lane of another device of the same pool is an error. The copies
are read-only, shared by the lanes of `dev`, and valid while their host arrays live:
once a host array has been collected, its copy is freed at the cache's next upload or
[`resident_count`](@ref).
"""
resident(pool::DevicePool, dev, obj) =
    on_device(() -> Adapt.adapt(DeviceAdaptor(pool, device(dev), false), obj), pool, dev)

"""
    device_program(pool, dev, prog) -> device program

The device form of the program `prog` on `dev`: `prog` adapted with
`DeviceAdaptor(pool, dev, true)`, so every array it carries is its cached device copy
behind [`readonly`](@ref). Programs are `isbits` after this adaptation on a GPU; a field
that the program writes must define its own `Adapt.adapt_structure` (see
[`DeviceAdaptor`](@ref)).
"""
device_program(pool::DevicePool, dev, prog) =
    on_device(() -> adapt_program(pool, device(dev), prog), pool, dev)

adapt_program(pool::DevicePool, dev::AbstractDevice, prog) =
    Adapt.adapt(DeviceAdaptor(pool, dev, true), prog)

"""
    resident_count(pool, dev) -> Int

Number of live entries in the residency cache of `dev` in `pool`, after evicting those
whose host object has been collected (every lane of `dev` synchronised, their device
arrays freed).
"""
function resident_count(pool::DevicePool, dev)
    d = device(dev)
    cache = pool.resident[device_slot(pool, d)]
    return @lock cache.lock length(evict!(pool, d, cache).twins)
end

"""
    ProgramTwins(pool, prog)

The device twins of the program `prog` for one call on `pool`, one per lane, each built by
its lane at its first chunk ([`program_twin!`](@ref)). The twins of the lanes of a device
read the same residency arrays and write their own accumulator and child twins, which
[`gather!`](@ref) folds; the call's job holds them until [`sync_lanes`](@ref).
"""
struct ProgramTwins{P}
    pool::DevicePool
    prog::P
    twins::Vector{Any}
end
ProgramTwins(pool::DevicePool, prog) = ProgramTwins(pool, prog, Any[nothing for _ in pool.lanes])

"""
    program_twin!(tw::ProgramTwins, lane) -> device program

The device form of `tw.prog` for `lane` ([`device_program`](@ref)'s adaptation), built at
the lane's first request. Call only from the lane's task.
"""
function program_twin!(tw::ProgramTwins, lane::PoolLane)
    k = lane_slot(tw.pool, lane)
    t = tw.twins[k]
    t === nothing && (t = tw.twins[k] = adapt_program(tw.pool, lane.dev, tw.prog))
    return t
end
