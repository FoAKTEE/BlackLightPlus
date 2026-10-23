# Residency: a host object as a device sees it, built by `Adapt` with every host array
# replaced by its copy on the device. Array copies are cached per pool and device by the
# identity of the host array, which the cache holds weakly: a dropped host array frees
# its entries. A cached host array must not be mutated in place afterwards.

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

# Drop the entries whose host object has been collected.
function purge!(cache::ResidencyCache)
    for (k, (w, _)) in collect(cache.twins)
        w.value === nothing && delete!(cache.twins, k)
    end
    return cache
end

# The copy of `host` on `dev`, uploaded on first request. Runs on a task with `dev` active.
function cached_twin(pool::DevicePool, dev::AbstractDevice, host::Array)
    cache = pool.resident[device_slot(pool, dev)]
    return @lock cache.lock begin
        id = objectid(host)
        slot = get(cache.twins, id, nothing)
        if slot === nothing || slot[1].value !== host
            purge!(cache)
            slot = cache.twins[id] = (WeakRef(host), upload(dev, host))
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
`dev`; calling from a lane of another device of the same pool is an error.
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

Number of live entries in the residency cache of `dev` in `pool`, after dropping those
whose host object has been collected.
"""
function resident_count(pool::DevicePool, dev)
    cache = pool.resident[device_slot(pool, dev)]
    return @lock cache.lock length(purge!(cache).twins)
end

# Per-call device twins of one program, built lazily by the first lane of each device
# that needs one (devices build concurrently, lanes of one device once).
struct ProgramTwins{P}
    pool::DevicePool
    prog::P
    twins::Vector{Any}
    locks::Vector{ReentrantLock}
end
ProgramTwins(pool::DevicePool, prog) =
    ProgramTwins(pool, prog, Any[nothing for _ in pool.devices],
                 [ReentrantLock() for _ in pool.devices])

function program_twin!(tw::ProgramTwins, lane::PoolLane)
    d = device_slot(tw.pool, lane.dev)
    return @lock tw.locks[d] begin
        t = tw.twins[d]
        t === nothing && (t = tw.twins[d] = adapt_program(tw.pool, lane.dev, tw.prog))
        t
    end
end
