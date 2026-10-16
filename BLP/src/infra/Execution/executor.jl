# Executors: run a program over a list of items and gather its per-item outputs. Both
# visit every item exactly once and leave item `item`'s outputs at `outs[k][item]`, so
# the outputs do not depend on the executor, the chunking, the device count, the lane
# that ran a chunk or the visiting order.

"""
    Executor

Runs a per-item program over a list of items ([`foreach_item!`](@ref)). Concrete
executors: [`ThreadedExecutor`](@ref) (Julia threads, host arrays) and
[`PoolExecutor`](@ref) (the device pool over KernelAbstractions backends). Products obtain
one from [`resolve`](@ref).
"""
abstract type Executor end

"""
    ThreadedExecutor(; chunk = 4096)

Runs a program on the host from Julia threads: items are cut into chunks of `chunk`,
handed to threads dynamically, and each item calls the host program directly with the
host output arrays.
"""
struct ThreadedExecutor <: Executor
    chunk::Int
    function ThreadedExecutor(chunk::Integer)
        chunk >= 1 || throw(ArgumentError("ThreadedExecutor: chunk must be positive"))
        return new(Int(chunk))
    end
end
ThreadedExecutor(; chunk::Integer = 4096) = ThreadedExecutor(chunk)

"""
    PoolExecutor(devices; lanes = 2, chunk = 65536)
    PoolExecutor(pool::DevicePool)

Runs a program on the devices of a [`DevicePool`](@ref): items are cut into chunks of
`chunk`, the chunks are pulled by the pool's lanes (`lanes` per device), and each chunk is
one launch of the program's device twin on the lane's device, its outputs copied back
into the host arrays. `devices` are device handles or CUDA ordinals; executors built with
the same devices, `lanes` and `chunk` share one registered pool and its residency caches.
"""
struct PoolExecutor <: Executor
    pool::DevicePool
    PoolExecutor(pool::DevicePool) = new(pool)
end
PoolExecutor(devices; lanes::Integer = 2, chunk::Integer = 65536) =
    PoolExecutor(device_pool(devices; lanes, chunk))

"Items per chunk of `exec`."
chunk_size(e::ThreadedExecutor) = e.chunk
chunk_size(e::PoolExecutor) = e.pool.chunk

resident(e::PoolExecutor, dev, obj) = resident(e.pool, dev, obj)
device_program(e::PoolExecutor, dev, prog) = device_program(e.pool, dev, prog)
resident_count(e::PoolExecutor, dev) = resident_count(e.pool, dev)

"""
    foreach_item!(exec, prog, items; outs::Tuple = (), order = nothing) -> outs
    foreach_item!(exec, (prog1 => items1, prog2 => items2, ...); outs::Tuple = ()) -> outs

Run the program `prog` once for every item of `items` (distinct integers) on the executor
`exec`, as `prog(item, i, outs...)`: the program writes its outputs for `item` at
`outs[k][i]` and returns `nothing`. `i` is the slot that the executor gathers into
`outs[k][item]` (the item itself under `ThreadedExecutor`, the position in the chunk on a
device), so after the call `outs[k][item]` holds the item's outputs. A program without
outputs (`outs = ()`) may write through atomics. `order`, a permutation of
`1:length(items)`, sets the visiting order to `items[order]`. The second form queues
several programs over their own items in one queue, part by part, sharing `outs`.

Determinism: bitwise per item, whatever the executor, chunk size, device count, lane
scheduling or `order`.
"""
function foreach_item!(exec::Executor, prog, items::AbstractVector{<:Integer};
                       outs::Tuple = (), order = nothing)
    run_parts!(exec, ((prog, visit_queue(items, order)),), outs)
    return outs
end

function foreach_item!(exec::Executor, parts::Tuple{Vararg{Pair}}; outs::Tuple = ())
    run_parts!(exec, map(p -> (p.first, p.second::AbstractVector{<:Integer}), parts), outs)
    return outs
end

visit_queue(items, ::Nothing) = items
visit_queue(items, order) = view(items, queue_items(order, length(items)))

# Chunk boundaries depend only on the item count and the chunk size.
chunk_ranges(n::Integer, c::Integer) =
    [((k - 1) * c + 1):min(k * c, n) for k in 1:cld(n, c)]

# (part, range) of every chunk, part by part in queue order.
part_chunks(parts::Tuple, c::Integer) =
    [(p, r) for p in eachindex(parts) for r in chunk_ranges(length(parts[p][2]), c)]

function run_parts!(exec::ThreadedExecutor, parts::Tuple, outs::Tuple)
    chunks = part_chunks(parts, exec.chunk)
    Threads.@threads :dynamic for c in chunks
        prog, q = parts[c[1]]
        run_range!(prog, q, c[2], outs)
    end
    return nothing
end

function run_range!(prog, q, r::UnitRange{Int}, outs::Tuple)
    for j in r
        item = @inbounds q[j]
        prog(item, item, outs...)
    end
    return nothing
end

function run_parts!(exec::PoolExecutor, parts::Tuple, outs::Tuple)
    pool = exec.pool
    chunks = part_chunks(parts, pool.chunk)
    twins = map(p -> ProgramTwins(pool, p[1]), parts)
    foreach_chunk(pool, chunks) do lane, (p, r)
        prog, q = parts[p]
        run_chunk!(lane, prog, program_twin!(twins[p], lane), view(q, r), outs)
    end
    return nothing
end

"""
    run_chunk!(lane, prog, dprog, idx, outs::Tuple) -> outs

One chunk on `lane`: stage the items `idx` to the device, launch the device program
`dprog` (made from `prog`, which supplies the launch traits) over them, copy the device
outputs back through host staging and store output `k` of item `idx[q]` at
`outs[k][idx[q]]`.
"""
function run_chunk!(lane::PoolLane, prog, dprog, idx::AbstractVector{<:Integer},
                    outs::Tuple)
    n = length(idx)
    st = lane_stage(lane, :items, Int, n)
    copyto!(st, 1, idx, 1, n)
    itd = lane_buffer(lane, :items, Int, n)
    copyto!(itd, 1, st, 1, n)
    outd = ntuple(k -> lane_buffer(lane, (:out, k), eltype(outs[k]), n), length(outs))
    launch!(lane, prog, dprog, itd, n, outd)
    for k in eachindex(outs)
        gather!(lane, k, outs[k], outd[k], idx)
    end
    return outs
end

function gather!(lane::PoolLane, k::Int, o, d, idx)
    n = length(idx)
    st = lane_stage(lane, (:out, k), eltype(o), n)
    copyto!(st, 1, d, 1, n)
    scatter!(o, st, idx)
    return o
end

# o[idx[q]] = st[q] for every q; a function barrier, so the loop is specialised.
function scatter!(o, st::Vector, idx)
    @inbounds for q in eachindex(idx)
        o[idx[q]] = st[q]
    end
    return o
end

"""
    describe(exec) -> NamedTuple

Where and how `exec` runs, as plain data for a result's `run` record:
`(kind, devices, lanes, chunk, threads)` with `kind` `:threads` or `:pool`, `devices` the
device names, `lanes` the workers per device (0 for threads), `chunk` the items per
chunk and `threads` the Julia thread count.
"""
describe(e::ThreadedExecutor) =
    (kind = :threads, devices = String[], lanes = 0, chunk = e.chunk,
     threads = Threads.nthreads())
describe(e::PoolExecutor) =
    (kind = :pool, devices = string.(e.pool.devices), lanes = lanes_per_device(e.pool),
     chunk = e.pool.chunk, threads = Threads.nthreads())

Base.show(io::IO, e::ThreadedExecutor) =
    print(io, "ThreadedExecutor(chunk = ", e.chunk, ") on ", Threads.nthreads(),
          " Julia threads")
Base.show(io::IO, e::PoolExecutor) =
    print(io, "PoolExecutor([", join(string.(e.pool.devices), ", "), "]; lanes = ",
          lanes_per_device(e.pool), ", chunk = ", e.pool.chunk, ") on ",
          Threads.nthreads(), " Julia threads")
