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
    ThreadedExecutor(; chunk = :auto)

Runs a program on the host from Julia threads: items are cut into chunks of `chunk` items
(`:auto`: sized per run, [`chunk_for`](@ref)), worker tasks, one per Julia thread,
pull the chunks from one queue in queue order, and each item calls the host program
directly with the host output arrays. An exception in a program is rethrown, as a
`CapturedException`, once the queue has drained. Executors compare equal when their chunk
settings do.
"""
struct ThreadedExecutor <: Executor
    chunk::Union{Int,Symbol}
    used::Threads.Atomic{Int}
    ThreadedExecutor(chunk) = new(chunk_setting(chunk), Threads.Atomic{Int}(0))
end
ThreadedExecutor(; chunk = :auto) = ThreadedExecutor(chunk)

"""
    PoolExecutor(devices; lanes = 2, chunk = :auto)
    PoolExecutor(pool::DevicePool; chunk = pool.chunk)

Runs a program on the devices of a [`DevicePool`](@ref): items are cut into chunks of
`chunk` items (`:auto`: sized per run, [`chunk_for`](@ref)), the chunks are pulled by
the pool's lanes (`lanes` per device), and each chunk is one launch of the program's device
twin on the lane's device, its outputs copied back into the host arrays. `devices` are
device handles or CUDA ordinals; executors built with the same devices and `lanes` share
one registered pool and its residency caches, whatever their chunk. Executors compare
equal when they share the pool and the chunk setting.
"""
struct PoolExecutor <: Executor
    pool::DevicePool
    chunk::Union{Int,Symbol}
    used::Threads.Atomic{Int}
    PoolExecutor(pool::DevicePool, chunk) =
        new(pool, chunk_setting(chunk), Threads.Atomic{Int}(0))
end
PoolExecutor(pool::DevicePool; chunk = pool.chunk) = PoolExecutor(pool, chunk)
PoolExecutor(devices; lanes::Integer = 2, chunk = :auto) =
    PoolExecutor(device_pool(devices; lanes), chunk)

Base.:(==)(a::ThreadedExecutor, b::ThreadedExecutor) = a.chunk == b.chunk
Base.:(==)(a::PoolExecutor, b::PoolExecutor) = a.pool === b.pool && a.chunk == b.chunk
Base.hash(e::ThreadedExecutor, h::UInt) = hash(e.chunk, hash(:ThreadedExecutor, h))
Base.hash(e::PoolExecutor, h::UInt) = hash(e.chunk, hash(objectid(e.pool), h))

# ---------------------------------------------------------------- automatic chunk
const POOL_CHUNK_MAX = 2^20      # items of one launch
const POOL_CHUNK_MIN = 2^16      # smallest chunk of a run shared by several devices
const CHUNKS_PER_LANE = 2        # on several devices
const CHUNK_OUT_BYTES = 2^24     # outputs of one launch, copied back and scattered by its lane
const THREAD_CHUNK_MAX = 4096
const THREAD_CHUNK_MIN = 16
const CHUNKS_PER_THREAD = 16

# The rule of `:auto` for a run of `n` items with `bytes` of outputs per item.
function auto_chunk(e::PoolExecutor, n::Integer, bytes::Integer)
    nd = length(e.pool.devices)
    c = nd == 1 ? POOL_CHUNK_MAX :
        clamp(cld(n, CHUNKS_PER_LANE * lanes_per_device(e.pool) * nd), POOL_CHUNK_MIN,
              POOL_CHUNK_MAX)
    return bytes > 0 ? min(c, max(POOL_CHUNK_MIN, CHUNK_OUT_BYTES ÷ bytes)) : c
end
auto_chunk(::ThreadedExecutor, n::Integer, bytes::Integer) =
    clamp(cld(n, CHUNKS_PER_THREAD * Threads.nthreads()), THREAD_CHUNK_MIN, THREAD_CHUNK_MAX)

"""
    chunk_for(exec, n; hint = nothing, bytes = 0) -> Int

The chunk of a run of `n` items on `exec` whose program writes `bytes` of outputs per item
and has the [`chunk_hint`](@ref) `hint`: the executor's chunk setting when it is an
integer, else the rule of `:auto`:

- a pool on one device: `2^20`, one launch per run of up to `2^20` items;
- a pool on `d > 1` devices of `l` lanes: `clamp(cld(n, 2 l d), 2^16, 2^20)`, two chunks
  per lane where the run is large enough;
- on a pool, at most `max(2^16, 2^24 ÷ bytes)`, so that the outputs of one launch (at most
  16 MiB), copied back and scattered by its lane, overlap the launches of the others;
- threads: `clamp(cld(n, 16 t), 16, 4096)` on `t` Julia threads;

then at most `hint`. Results never depend on the chunk.
"""
chunk_for(e::Executor, n::Integer; hint = nothing, bytes::Integer = 0) =
    e.chunk isa Int ? e.chunk : min(auto_chunk(e, n, bytes), something(hint, typemax(Int)))

"""
    resolve_chunk!(exec, n; hint = nothing, bytes = 0) -> Int

The chunk of a run of `n` items on `exec` ([`chunk_for`](@ref)), recorded as the chunk of
the executor's last run, which [`describe`](@ref) reports.
"""
function resolve_chunk!(e::Executor, n::Integer; hint = nothing, bytes::Integer = 0)
    c = chunk_for(e, n; hint, bytes)
    e.used[] = c
    return c
end

# The smallest chunk hint of the programs of `parts`; `nothing` when none has one.
parts_hint(parts::Tuple) = foldl((h, p) -> min_hint(h, chunk_hint(p[1])), parts; init = nothing)
min_hint(a, b) = a === nothing ? b : b === nothing ? a : min(a, b)

# Bytes of output per item.
item_bytes(outs::Tuple) = sum(o -> sizeof(eltype(o)), outs; init = 0)

# The chunk of the executor's last run, or its setting before the first run.
last_chunk(e::Executor) = (c = e.used[]; c > 0 ? c : e.chunk)

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

# A run of the parts in one queue, cut at the run's chunk.
function run_parts!(exec::Executor, parts::Tuple, outs::Tuple)
    n = sum(p -> length(p[2]), parts; init = 0)
    c = resolve_chunk!(exec, n; hint = parts_hint(parts), bytes = item_bytes(outs))
    run_parts!(exec, parts, outs, c)
end

visit_queue(items, ::Nothing) = items
visit_queue(items, order) = view(items, queue_items(order, length(items)))

# Chunk boundaries depend only on the item count and the chunk size.
chunk_ranges(n::Integer, c::Integer) =
    [((k - 1) * c + 1):min(k * c, n) for k in 1:cld(n, c)]

# (part, range) of every chunk, part by part in queue order.
part_chunks(parts::Tuple, c::Integer) =
    [(p, r) for p in eachindex(parts) for r in chunk_ranges(length(parts[p][2]), c)]

# Worker tasks, one per thread (at most one per chunk), pull the chunks in queue order.
function run_parts!(::ThreadedExecutor, parts::Tuple, outs::Tuple, c::Int)
    chunks = part_chunks(parts, c)
    run_workers!(min(Threads.nthreads(), length(chunks)), chunks) do _, (p, r), _
        prog, q = parts[p]
        run_range!(prog, q, r, outs)
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

function run_parts!(exec::PoolExecutor, parts::Tuple, outs::Tuple, c::Int)
    pool = exec.pool
    chunks = part_chunks(parts, c)
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
device names, `lanes` the workers per device (0 for threads), `chunk` the items per chunk
of the executor's last run (its setting, an integer or `:auto`, before the first run) and
`threads` the Julia thread count.
"""
describe(e::ThreadedExecutor) =
    (kind = :threads, devices = String[], lanes = 0, chunk = last_chunk(e),
     threads = Threads.nthreads())
describe(e::PoolExecutor) =
    (kind = :pool, devices = string.(e.pool.devices), lanes = lanes_per_device(e.pool),
     chunk = last_chunk(e), threads = Threads.nthreads())

Base.show(io::IO, e::ThreadedExecutor) =
    print(io, "ThreadedExecutor(chunk = ", repr(e.chunk), ") on ", Threads.nthreads(),
          " Julia threads")
Base.show(io::IO, e::PoolExecutor) =
    print(io, "PoolExecutor([", join(string.(e.pool.devices), ", "), "]; lanes = ",
          lanes_per_device(e.pool), ", chunk = ", repr(e.chunk), ") on ",
          Threads.nthreads(), " Julia threads")
