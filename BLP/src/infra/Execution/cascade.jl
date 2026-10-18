# The cascade: a primary program over chunks of items, and inside each chunk a layer
# program over the children every generation spawned, generation after generation. The
# children of a chunk never leave its worker: a pool lane keeps them in two device
# buffers it swaps per generation and reads back only their count; a thread keeps them
# in two task-local buffers. The children's streams are keyed by their genealogy and the
# accumulators are exact integer sums, so the totals and the set of children do not
# depend on the executor, the chunking, the device count, the lane or the arrival order.

"Most primaries in one device chunk of a cascade."
const CASCADE_CHUNK = 2^20

"Fewest child slots a chunk's generation starts with."
const MIN_CHILD_SLOTS = 4096

"""
    cascade_chunk(exec, n) -> Int

Primaries per chunk of a cascade over `n` primaries on `exec`. On a pool: a quarter of a
device's share, at most `CASCADE_CHUNK` and at least the pool's chunk. On threads: a
sixty-fourth of a thread's share, at least the executor's chunk.
"""
cascade_chunk(e::PoolExecutor, n::Integer) =
    max(e.pool.chunk, min(cld(n, 4 * length(e.pool.devices)), CASCADE_CHUNK))
cascade_chunk(e::ThreadedExecutor, n::Integer) =
    max(e.chunk, cld(n, 64 * Threads.nthreads()))

"""
    foreach_cascade!(exec, primary, layer, n, depth; capacity = nothing,
                     chunk = cascade_chunk(exec, n)) -> Int

Run a cascade of at most `depth + 1` generations on the executor `exec`: generation 0 is
the program `primary` over the items `1:n`, generation `g ≥ 1` the program `layer` over
the children that generation `g − 1` spawned. The items are cut into chunks of `chunk`
consecutive items, and one worker runs every generation of a chunk before it takes the
next chunk. Both programs are called as `prog(item, i)` without outputs: the primary's
`item` is its global index, the layer's `item` its member of the [`Generation`](@ref)
the layer program holds, which the cascade replaces by the chunk's children of the
previous generation. A program spawns into the [`ChildBuffer`](@ref) it holds, a slot
whose element type and layout the cascade reads (pass one of capacity 0) and which it
replaces by the chunk's child buffer. A primary that holds no `ChildBuffer` spawns
nothing, and only generation 0 runs. Returns the number of children spawned in all
generations; the children of generation `depth` are counted, not tracked.

The programs' [`ExactBins`](@ref) and [`ExactScalars`](@ref) receive every addition
exactly once; call [`gather!`](@ref) on them afterwards, as after `foreach_item!`.
Children that overflow a buffer are recovered by running their parents again, after the
buffer grows, into zeroed scratch accumulators of the same types, so a rerun neither
counts twice nor changes the compiled program. On a `PoolExecutor` each lane passes the
chunk's range to the device without staging, keeps the chunk's children in two
lane-resident buffers swapped per generation and reads back only their count; a
generation that can spawn records into lane scratch, added into the device's totals once
its children fit, and an overflowing generation runs again in full. A device generation
starts with `capacity` child slots (`nothing`: at least `MIN_CHILD_SLOTS`, half the
chunk). On a `ThreadedExecutor` worker tasks pull the chunks and run their items in
order into the shared totals, with two task-local buffers of `capacity` slots
(`nothing`: `MIN_CHILD_SLOTS`) that grow; only an item whose children overflow runs
again.

Determinism: the accumulated totals, the set of children and the returned count are
bitwise independent of the executor, the chunking, the device count, the lane and the
arrival order.
"""
function foreach_cascade!(exec::Executor, primary, layer, n::Integer, depth::Integer;
                          capacity = nothing, chunk::Integer = cascade_chunk(exec, n))
    n >= 0 || throw(ArgumentError("foreach_cascade!: n must be non-negative"))
    depth >= 0 || throw(ArgumentError("foreach_cascade!: depth must be non-negative"))
    chunk >= 1 || throw(ArgumentError("foreach_cascade!: chunk must be positive"))
    capacity === nothing || capacity >= 1 ||
        throw(ArgumentError("foreach_cascade!: capacity must be positive"))
    n == 0 && return 0
    slot = child_slot(primary)
    c = Cascade(primary, layer, slot, slot === nothing ? 0 : Int(depth), capacity)
    return run_cascade!(exec, c, chunk_ranges(n, chunk))
end

# One cascade call: the programs, the child slot they spawn into (or `nothing`), the last
# generation and the initial child slots of a device generation.
struct Cascade{P,L,S}
    primary::P
    layer::L
    slot::S
    depth::Int
    capacity::Any
end

# The `ChildBuffer` that a program spawns into, or `nothing`.
struct SlotProbe
    found::Vector{Any}
end
Adapt.adapt_structure(to::SlotProbe, b::ChildBuffer) = (push!(to.found, b); b)

function child_slot(prog)
    probe = SlotProbe(Any[])
    Adapt.adapt(probe, prog)
    return isempty(probe.found) ? nothing : first(probe.found)
end

"""
    CascadeSwap(lane, children, parents, scratch)

The `Adapt` adaptor that places one generation of a cascade into a program: every
`ChildBuffer` becomes `children` (the chunk's child buffer), every `Generation` becomes
`parents`, and with `scratch` every `ExactBins` and `ExactScalars` writes zeroed scratch
arrays of the same types instead of its own: lane-resident arrays of the pool lane
`lane`, or fresh host arrays when `lane` is `nothing`. `pairs` lists each (own, scratch)
array pair in the order met.
"""
struct CascadeSwap{L,C,G}
    lane::L
    children::C
    parents::G
    scratch::Bool
    pairs::Vector{Any}
end
CascadeSwap(lane, children, parents, scratch::Bool) =
    CascadeSwap(lane, children, parents, scratch, Any[])

Adapt.adapt_structure(to::CascadeSwap, ::ChildBuffer) = to.children
Adapt.adapt_structure(to::CascadeSwap, ::Generation) = to.parents
Adapt.adapt_structure(to::CascadeSwap, b::ExactBins) =
    to.scratch ? with_arrays(b, scratch!(to, b.count), scratch!(to, b.sumw),
                             scratch!(to, b.sumwE), fresh_twins(b.twins)) : b
Adapt.adapt_structure(to::CascadeSwap, s::ExactScalars) =
    to.scratch ? ExactScalars(scratch!(to, s.acc), scratch!(to, s.n), fresh_twins(s.twins)) : s

# The twin list of a scratch accumulator: of the same type as the accumulator's own.
fresh_twins(::Twins) = Twins()
fresh_twins(::Nothing) = nothing

# The scratch array of the next pair. On a lane it is keyed by position, so the same
# accumulator of every generation and call finds the same array.
function scratch!(to::CascadeSwap, a::AbstractArray{T}) where {T}
    s = to.lane === nothing ? zeros(T, size(a)) :
        lane_array(to.lane, (:cascade_scratch, length(to.pairs) + 1), T, size(a))
    push!(to.pairs, (a, s))
    return s
end

"""
    Drain(totals, scratch, keep)

The program that empties scratch accumulators: item `k` adds element `k` of every array
of the tuple `scratch` into the same element of the matching array of `totals` (atomic,
when `keep`) and sets it to zero. Run over `1:maximum(length, scratch)`.
"""
struct Drain{T,S} <: Function
    totals::T
    scratch::S
    keep::Bool
end
Adapt.@adapt_structure Drain

function (d::Drain)(k, i)
    drain_element!(d.totals, d.scratch, k, d.keep)
    return nothing
end

@inline drain_element!(::Tuple{}, ::Tuple{}, k, keep) = nothing
@inline function drain_element!(t::Tuple, s::Tuple, k, keep)
    a = first(t)
    b = first(s)
    if k <= length(b)
        @inbounds v = b[k]
        if v != zero(v)
            keep && (Atomix.@atomic a[k] += v)
            @inbounds b[k] = zero(v)
        end
    end
    drain_element!(Base.tail(t), Base.tail(s), k, keep)
end

# ---------------------------------------------------------------- threads

function run_cascade!(exec::ThreadedExecutor, c::Cascade, chunks)
    spawned = Threads.Atomic{Int}(0)
    run_workers!(Threads.nthreads(), chunks) do _, r, _
        bufs = c.slot === nothing ? nothing :
               get!(() -> task_buffers(c.slot, something(c.capacity, MIN_CHILD_SLOTS)),
                    task_local_storage(), c)
        Threads.atomic_add!(spawned, chunk_generations!(c, r, bufs))
    end
    return spawned[]
end

# Two child buffers of the slot's type, of `cap` slots each, for one task.
task_buffers(slot::ChildBuffer, cap::Integer) =
    (ChildBuffer(eltype(slot.photons), cap, slot.layout),
     ChildBuffer(eltype(slot.photons), cap, slot.layout))

# Every generation of the primaries `r` in the calling task; returns the children spawned.
function chunk_generations!(c::Cascade, r::UnitRange{Int}, bufs)
    if bufs === nothing
        run_items!(c.primary, r)
        return 0
    end
    spawned = 0
    m = 0
    for g in 0:c.depth
        out = bufs[isodd(g) ? 2 : 1]
        parents = bufs[isodd(g) ? 1 : 2]
        out.count[1] = 0
        gen = g == 0 ? nothing : Generation(parents.photons, parents.rngs)
        prog = g == 0 ? c.primary : c.layer
        items = g == 0 ? r : (1:m)
        run = Adapt.adapt(CascadeSwap(nothing, out, gen, false), prog)
        if g == c.depth
            run_items!(run, items)
        else
            rerun = () -> Adapt.adapt(CascadeSwap(nothing, out, gen, true), prog)
            run_growing!(run, rerun, out, items)
        end
        n_out = Int(out.count[1])
        spawned += n_out
        m = g == c.depth ? 0 : n_out
        m == 0 && break
    end
    return spawned
end

function run_items!(prog, items)
    for item in items
        prog(item, item)
    end
    return nothing
end

# `prog` over `items` in order, spawning into `out`. An item whose children overflow `out`
# runs again once `out` has grown, as the program `rerun()` writing scratch accumulators,
# from the slot where its children began: its additions count once and all its children
# are stored.
function run_growing!(prog, rerun, out::ChildBuffer, items)
    for item in items
        before = out.count[1]
        prog(item, item)
        if out.count[1] > length(out.photons)
            n = max(Int(out.count[1]), 2 * length(out.photons))
            resize!(out.photons, n)
            resize!(out.rngs, n)
            out.count[1] = before
            again = rerun()
            typeof(again) === typeof(prog) ||
                error("foreach_cascade!: the scratch program differs in type from the program")
            again(item, item)
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the pool

function run_cascade!(exec::PoolExecutor, c::Cascade, chunks)
    pool = exec.pool
    twins = (ProgramTwins(pool, c.primary), ProgramTwins(pool, c.layer))
    spawned = Threads.Atomic{Int}(0)
    fresh = fill(true, length(pool.lanes))
    try
        foreach_chunk(pool, chunks) do lane, r
            Threads.atomic_add!(spawned, chunk_generations!(lane, c, twins, r, fresh))
        end
    finally
        foreach(s -> s === nothing || empty!(s), (c.slot, child_slot(c.layer)))
    end
    return spawned[]
end

# Every generation of the primaries `r` on `lane`; returns the children spawned. `fresh`
# marks the lanes whose scratch arrays have not been zeroed in this call.
function chunk_generations!(lane::PoolLane, c::Cascade, twins, r::UnitRange{Int}, fresh)
    if c.slot === nothing
        launch!(lane, c.primary, program_twin!(twins[1], lane), r, length(r), ())
        return 0
    end
    T = eltype(c.slot.photons)
    cap = c.capacity === nothing ? max(MIN_CHILD_SLOTS, cld(length(r), 2)) : Int(c.capacity)
    k = findfirst(l -> l === lane, twins[1].pool.lanes)
    ctr = lane_buffer(lane, (:cascade, :count), Int64, 1)
    spawned = 0
    m = 0
    for g in 0:c.depth
        final = g == c.depth
        prog = g == 0 ? c.primary : c.layer
        twin = program_twin!(twins[g == 0 ? 1 : 2], lane)
        items = g == 0 ? r : (1:m)
        parents = g == 0 ? nothing :
                  Generation(map(a -> readonly(lane.dev, a),
                                 lane_children(lane, T, isodd(g) ? :a : :b, m))...)
        slots = final ? 0 : cap
        while true
            out = ChildBuffer(lane_children(lane, T, isodd(g) ? :b : :a, slots)..., ctr,
                              c.slot.layout, nothing)
            swap = CascadeSwap(lane, out, parents, !final)
            dprog = Adapt.adapt(swap, twin)
            if !final && fresh[k]
                drain_lane!(lane, swap.pairs, false)
                fresh[k] = false
            end
            set_count!(lane, ctr, 0)
            launch!(lane, prog, dprog, items, length(items), ())
            n_out = get_count(lane, ctr)
            if !final && n_out > slots
                drain_lane!(lane, swap.pairs, false)
                slots = cap = n_out
                continue
            end
            final || drain_lane!(lane, swap.pairs, true)
            m = final ? 0 : n_out
            spawned += n_out
            break
        end
        m == 0 && break
    end
    return spawned
end

# The first `n` slots of the lane's child buffer `side` (`:a` or `:b`): photons and streams.
lane_children(lane::PoolLane, ::Type{T}, side::Symbol, n::Integer) where {T} =
    (view(lane_buffer(lane, (:cascade, side), T, max(n, 1)), 1:n),
     view(lane_buffer(lane, (:cascade, side), MCRng, max(n, 1)), 1:n))

function set_count!(lane::PoolLane, d, v::Integer)
    st = lane_stage(lane, (:cascade, :count_in), Int64, 1)
    st[1] = v
    copyto!(d, 1, st, 1, 1)
    return nothing
end

function get_count(lane::PoolLane, d)
    st = lane_stage(lane, (:cascade, :count_out), Int64, 1)
    copyto!(st, 1, d, 1, 1)
    return Int(st[1])
end

# Empty the scratch arrays of `pairs` on `lane`, adding them into their totals when `keep`.
function drain_lane!(lane::PoolLane, pairs::Vector{Any}, keep::Bool)
    isempty(pairs) && return nothing
    d = Drain(Tuple(p[1] for p in pairs), Tuple(p[2] for p in pairs), keep)
    n = maximum(p -> length(p[2]), pairs)
    launch!(lane, d, d, 1:n, n, ())
    return nothing
end
