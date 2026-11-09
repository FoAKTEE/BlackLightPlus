# The device wavefront: a phase of the Monte Carlo cut into launches of stage programs
# over queues of the first lane of every device. A start launch emits a block of
# primaries into the cohort, an advance runs every live state for at most K loop
# iterations and spawns children into the child queue, a retire launch records and
# tallies the states that stopped, a branch launch opens the children into the cohort; the
# host reads the counter block after the advance and after the branch and refills the
# cohort from a block counter that the devices share. A state's march is cut only between
# loop iterations, every value it carries lives in the state and the advance writes
# nothing but its queues, so an advance whose children overflowed runs again unchanged;
# the totals do not depend on K, the cohort, the refill, the child slots, the device count
# or the order of the states, which a key stage may sort (wave_sort.jl).

"Loop iterations per state of an advance launch, by default."
const WAVE_BUDGET = 32
"Live states per device, by default."
const WAVE_COHORT = 2^18
"Consecutive primaries per block of the refill, by default."
const WAVE_REFILL = 2^14
"The most loop iterations the budget doubles to in a call whose states spawn children."
const WAVE_SPAWN_CAP = 128
"The most seconds a device of a call waits for the other devices before its first refill."
const WAVE_ARRIVAL = 1.0

"""
    WaveSettings(budget, cohort, refill, doubling, wavefront, sort_every)

The settings of the wavefront that a call reads ([`with_wavefront_settings`](@ref)):
the budget `K` of an advance, the cohort `P`, the refill block `R`, the budget doubling
(`true`: up to the cap of the phase, `false`: off, an integer: up to that many loop
iterations), whether a phase that the wavefront can serve runs on it (`false` runs
the cascade of the same programs) and the advances between two sorts of the cohort by
the key stage (0: never).
"""
struct WaveSettings
    budget::Int
    cohort::Int
    refill::Int
    doubling::Union{Bool,Int}
    wavefront::Bool
    sort_every::Int
end

const DEFAULT_WAVE = WaveSettings(WAVE_BUDGET, WAVE_COHORT, WAVE_REFILL, true, true, WAVE_SORT)
const WAVE_SETTINGS = Base.ScopedValues.ScopedValue(DEFAULT_WAVE)

"""
    with_wavefront_settings(f; budget = WAVE_BUDGET, cohort = WAVE_COHORT,
                            refill = WAVE_REFILL, doubling = true, wavefront = true,
                            sort_every = WAVE_SORT)

Run `f()` with these settings of the wavefront, read on the host by every wavefront call
inside `f` on the calling task; outside such a scope every call uses the defaults. `budget`
is the loop iterations per state of an advance (`typemax(Int)` runs every state to its
end in one advance), `cohort` the live states per device the refill fills up to, `refill`
the primaries per block a device takes from the shared block counter, `doubling` the
budget doubling once no block remains and the cohort is below a quarter of `cohort`
(`true` up to the cap of the phase: the `Wavefront`'s, and at most `WAVE_SPAWN_CAP` in a
call with children; `false` off; an integer `k ≥ 1` up to `k`), `wavefront = false`
runs the cascade of the same programs, and `sort_every = m ≥ 1` sorts the cohort by the
keys of the `Wavefront`'s key stage before every `m`-th advance of a device, from its first
(`WAVE_SORT = 1` by default; 0 never; a `Wavefront` without a key stage never sorts).
Results do not depend on any of them. Returns `f()`.
"""
function with_wavefront_settings(f; budget::Integer = WAVE_BUDGET, cohort::Integer = WAVE_COHORT,
                                 refill::Integer = WAVE_REFILL, doubling::Union{Bool,Integer} = true,
                                 wavefront::Bool = true, sort_every::Integer = WAVE_SORT)
    budget >= 1 && cohort >= 1 && refill >= 1 ||
        throw(ArgumentError("with_wavefront_settings: budget, cohort and refill must be positive"))
    doubling isa Bool || doubling >= 1 ||
        throw(ArgumentError("with_wavefront_settings: doubling must be true, false or a positive cap"))
    sort_every >= 0 || throw(ArgumentError("with_wavefront_settings: sort_every must be non-negative"))
    s = WaveSettings(Int(budget), Int(cohort), Int(refill),
                     doubling isa Bool ? doubling : Int(doubling), wavefront, Int(sort_every))
    return Base.ScopedValues.with(f, WAVE_SETTINGS => s)
end

"The wavefront settings of the calling task ([`with_wavefront_settings`](@ref))."
wave_settings() = WAVE_SETTINGS[]

settings_record(s::WaveSettings) = (budget = s.budget, cohort = s.cohort, refill = s.refill,
                                    doubling = s.doubling, wavefront = s.wavefront,
                                    sort_every = s.sort_every)

"""
    Wavefront(; start, advance, retire, primary, layer, branch = nothing, key = nothing,
              doubling_cap = typemax(Int))

The programs of one Monte Carlo transport for [`foreach_wavefront!`](@ref): the stage
programs `start` (item `g`, a primary, placed into the cohort by [`place!`](@ref)),
`advance` (member `j` of the cohort: at most `K` loop iterations, then [`admit!`](@ref)
into `B` while live or into `D` once stopped, [`fault!`](@ref) for a state that broke
the contract; its children go to the [`ChildBuffer`](@ref) it holds), `retire` (slot `j`
of `D`: its record and tally) and `branch` (child `j` of `C`: its state opened and
handed to [`admit_child!`](@ref); `nothing` when the wavefront serves only calls without
children), each called as `prog(item, i)` with a [`WaveQueues`](@ref) placeholder that
every launch binds; the key stage `key` (member `j` of the cohort: its key stored and
counted by [`count_key!`](@ref) into a [`KeyCount`](@ref) placeholder; `nothing` when the
cohort is never sorted); and the history programs `primary` and `layer` of the same
transport, which the cascade runs where the wavefront does not serve. `doubling_cap` is
the most loop iterations the budget doubles to; a call with children caps it further at
`WAVE_SPAWN_CAP`. The stage programs must agree on [`wave_state`](@ref); the advance
answers [`wave_steps`](@ref), the key stage [`wave_keys`](@ref).
"""
struct Wavefront{S,A,R,B,K,P,L}
    start::S
    advance::A
    retire::R
    branch::B
    key::K
    primary::P
    layer::L
    doubling_cap::Int
end

function Wavefront(; start, advance, retire, primary, layer, branch = nothing, key = nothing,
                   doubling_cap::Integer = typemax(Int))
    T = wave_state(start)
    wave_state(advance) === T && wave_state(retire) === T &&
        (branch === nothing || wave_state(branch) === T) && (key === nothing || wave_state(key) === T) ||
        throw(ArgumentError("Wavefront: the stage programs disagree on the state type"))
    wave_steps(advance) >= 1 || throw(ArgumentError("Wavefront: the advance's step limit must be positive"))
    key === nothing || wave_keys(key) >= 1 ||
        throw(ArgumentError("Wavefront: the key stage must have at least one key value"))
    doubling_cap >= 1 || throw(ArgumentError("Wavefront: doubling_cap must be positive"))
    return Wavefront(start, advance, retire, branch, key, primary, layer, Int(doubling_cap))
end

"""
    foreach_wavefront!(exec, wf::Wavefront, n, depth; capacity = nothing,
                       chunk = nothing) -> Int

Run the transport of `wf` over the primaries `1:n` with at most `depth` scattering
generations, as [`foreach_cascade!`](@ref)`(exec, wf.primary, wf.layer, n, depth)` would,
and return the children spawned. The device wavefront serves the call when `exec` is a
`PoolExecutor` whose every device answers [`wavefront`](@ref)`(dev) = true`, the settings
allow it ([`with_wavefront_settings`](@ref)) and the call is a phase the wavefront has
stages for: `depth = 0` (the children of the primaries counted and not tracked), or any
depth when `wf` has a branch. Any other call runs the cascade with `capacity` and, when
given, `chunk` (the cascade's default otherwise).

On the wavefront each device runs its own cohort on its first lane until no primary
block remains and its cohort is empty. Per iteration: with a key stage and `sort_every =
m ≥ 1`, before every `m`-th advance of the device (from its first) the cohort's read order
sorted by key: the key launch, one copy of the counts to the host, their exclusive scan,
one copy back and a [`KeyPlace`](@ref) launch; the counter block zeroed; the
advance over the cohort in that order, at most `K` loop iterations per state, its children into the
child queue `C` of `cap_C` slots; one read of the counter block, each of a fault,
`live + done ≠ nA` and a queue beyond its slots an error; with more children than
`cap_C`, `C` grows to the children and an eighth more, the executor counts a rerun and the
advance runs again on the unchanged cohort; the retire launch over the stopped states; the
branch over the children, each opened into the cohort or, beyond generation `depth`,
counted as dropped, then a second read of the counter block; the refill: when fewer than
`P` states are live and blocks remain, the device takes
`max(1, min(cld(P − nB, R), cld(r, ndev)))` of the `r` remaining blocks of `R`
consecutive primaries in one claim and runs the start over them; then the queues swap.
Before its first refill a device waits, at most `WAVE_ARRIVAL` seconds, until every device
of the call has reached its own, so the blocks of a call spread over every device. `cap_C`
is `capacity` at every iteration when given; otherwise it starts at the larger of `P` and
the child slots that earlier calls left on the device's first lane, and keeps every
growth. A call with `depth = 0` stores no child: the advance counts the children, all
dropped. With doubling on, `K` doubles every iteration once no block remains and fewer
than `P/4` states are live, up to the phase's cap (at most `WAVE_SPAWN_CAP` when
`depth > 0`). A device whose cohort retires nothing in `M + 2` consecutive iterations
(`M` = [`wave_steps`](@ref) of the advance) is an error. At the end the started primaries
must equal `n`, their ranges must partition `1:n` and the retired states must equal `n`
plus the children spawned less those dropped. In `finally` the child slot of every program
is emptied.

Determinism: the accumulated totals and the returned count are those of the cascade,
bitwise when the stage programs give the history programs' bits, whatever the settings,
the child slots, the device count and the arrival order.
"""
function foreach_wavefront!(exec::Executor, wf::Wavefront, n::Integer, depth::Integer;
                            capacity = nothing, chunk = nothing)
    n >= 0 || throw(ArgumentError("foreach_wavefront!: n must be non-negative"))
    depth >= 0 || throw(ArgumentError("foreach_wavefront!: depth must be non-negative"))
    capacity === nothing || capacity >= 1 ||
        throw(ArgumentError("foreach_wavefront!: capacity must be positive"))
    set = wave_settings()
    serves(exec, wf, depth, set) &&
        return run_wavefront!(exec, wf, Int(n), Int(depth), set; capacity)
    kw = chunk === nothing ? (; capacity) : (; capacity, chunk)
    return foreach_cascade!(exec, wf.primary, wf.layer, n, depth; kw...)
end

# Whether the wavefront serves a call: a pool whose devices answer `wavefront`, the
# setting, and a phase with stages (the primaries alone, their children counted, or every
# generation with a branch). Threads and every other executor run the cascade.
serves(::Executor, wf, depth, set) = false
serves(e::PoolExecutor, wf::Wavefront, depth, set::WaveSettings) =
    set.wavefront && (depth == 0 || wf.branch !== nothing) && all(wavefront, e.pool.devices)

"""
    run_wavefront!(exec::PoolExecutor, wf::Wavefront, n, depth, set::WaveSettings;
                   capacity = nothing) -> Int

The wavefront of [`foreach_wavefront!`](@ref) over the primaries `1:n` and their
children up to generation `depth`, with the settings `set` and the child slots
`capacity`, on every device of the pool; returns the children spawned. Records the
settings, the advances run again for their children and the primaries admitted and
children spawned per device for [`describe`](@ref).
"""
function run_wavefront!(exec::PoolExecutor, wf::Wavefront, n::Int, depth::Int, set::WaveSettings;
                        capacity = nothing)
    pool = exec.pool
    exec.reruns[] = 0
    slots = map(child_slot, (wf.start, wf.advance, wf.retire, wf.branch, wf.primary, wf.layer))
    blocks = BlockCounter(n, set.refill, length(pool.devices))
    progs = depth > 0 ? (wf.start, wf.advance, wf.retire, wf.branch) : (wf.start, wf.advance, wf.retire)
    twins = map(p -> ProgramTwins(pool, p), progs)
    ktwin = wf.key !== nothing && set.sort_every > 0 ? ProgramTwins(pool, wf.key) : nothing
    phase = depth > 0 ? min(wf.doubling_cap, WAVE_SPAWN_CAP) : wf.doubling_cap
    cap = set.doubling === true ? phase : set.doubling === false ? set.budget : set.doubling
    call = (depth = depth, capacity = capacity === nothing ? nothing : Int(capacity),
            reruns = exec.reruns)
    res = try
        n == 0 ? [empty_device() for _ in pool.devices] :
        foreach_device(pool) do lane
            device_wavefront!(lane, wf, twins, ktwin, slots[2], blocks, set, cap, call)
        end
    finally
        foreach(s -> s === nothing || empty!(s), slots)
    end
    admitted = [r.admitted for r in res]
    spawned = [r.spawned for r in res]
    record_run!(exec, :wavefront, settings_record(set), admitted, spawned;
                iterations = [r.iterations for r in res], log = [r.log for r in res])
    check_partition(n, reduce(vcat, (r.ranges for r in res); init = UnitRange{Int}[]))
    retired = sum(r -> r.retired, res)
    dropped = sum(r -> r.dropped, res)
    retired == n + sum(spawned) - dropped ||
        error("wavefront: $retired states retired, expected $n primaries + $(sum(spawned)) " *
              "children - $dropped dropped")
    return sum(spawned)
end

empty_device() = (admitted = 0, spawned = 0, dropped = 0, retired = 0, ranges = UnitRange{Int}[],
                  iterations = 0, log = NamedTuple[])

# The started ranges of all devices partition `1:n`.
function check_partition(n::Int, ranges::Vector{UnitRange{Int}})
    sum(length, ranges; init = 0) == n ||
        error("wavefront: $(sum(length, ranges; init = 0)) primaries started, expected $n")
    next = 1
    for r in sort(ranges; by = first)
        first(r) == next || error("wavefront: the started ranges do not partition 1:$n at $next")
        next = last(r) + 1
    end
    next == n + 1 || error("wavefront: the started ranges end at $(next - 1), expected $n")
    return nothing
end

"""
    BlockCounter(n, R, ndev)

The primaries `1:n` in blocks of `R` consecutive primaries (the last one clipped), with
one atomic counter of the blocks taken and one of the devices ready to take them, which the
`ndev` devices of a call share.
"""
struct BlockCounter
    n::Int
    R::Int
    ndev::Int
    blocks::Int
    taken::Threads.Atomic{Int}
    arrived::Threads.Atomic{Int}
end
BlockCounter(n::Int, R::Int, ndev::Int) =
    BlockCounter(n, R, ndev, cld(n, R), Threads.Atomic{Int}(0), Threads.Atomic{Int}(0))

"""
    arrive!(b::BlockCounter, wait = WAVE_ARRIVAL)

Count the calling device as ready to take blocks, then yield until every device of the call
is, or `wait` seconds have passed: a device whose set-up ends first does not claim the
blocks of a small call before the others can. The blocks a device takes change no result.
"""
function arrive!(b::BlockCounter, wait::Real = WAVE_ARRIVAL)
    Threads.atomic_add!(b.arrived, 1)
    t0 = time()
    while b.arrived[] < b.ndev && time() - t0 < wait
        yield()
    end
    return nothing
end

"Whether every block has been taken."
exhausted(b::BlockCounter) = b.taken[] >= b.blocks

"Primaries not yet taken (an upper bound while other devices take blocks)."
remaining(b::BlockCounter) = max(0, b.n - b.taken[] * b.R)

"""
    take_blocks!(b::BlockCounter, room) -> UnitRange{Int}

The primaries of the blocks a device with room for `room` more live states takes at once:
`max(1, min(cld(room, R), cld(r, ndev)))` of the `r` remaining blocks, consecutive; an
empty range once no block remains.
"""
function take_blocks!(b::BlockCounter, room::Int)
    while true
        t = b.taken[]
        t >= b.blocks && return 1:0
        r = b.blocks - t
        k = min(r, max(1, min(cld(room, b.R), cld(r, b.ndev))))
        Threads.atomic_cas!(b.taken, t, t + k) == t || continue
        return (t * b.R + 1):min((t + k) * b.R, b.n)
    end
end

# n + k without overflow.
sat_add(a::Int, b::Int) = a > typemax(Int) - b ? typemax(Int) : a + b

# The slots of B for an iteration of `nA` live states with `capC` child slots: the
# survivors and the opened children (at most `nA + capC`), then a refill that stops once
# `P` states are live, by at most a block past `P` and the primaries remaining.
queue_slots(nA::Int, capC::Int, P::Int, b::BlockCounter) =
    max(1, sat_add(nA, capC), min(sat_add(P, b.R - 1), sat_add(sat_add(nA, capC), remaining(b))))

# The child slots of type `E` that the first lane `lane` holds from earlier calls.
held_slots(lane::PoolLane, ::Type{E}) where {E} =
    (c = get(lane.bufs, ((:wave, :child), E), nothing); c === nothing ? 0 : length(c))

# The child queue of `n` slots of type `E` on `lane`, counting into the counter block.
child_queue(lane::PoolLane, ::Type{E}, n::Int, counters, slot) where {E} =
    ChildBuffer(view(lane_buffer(lane, (:wave, :child), E, max(n, 1)), 1:n),
                view(lane_buffer(lane, (:wave, :child), MCRng, max(n, 1)), 1:n),
                view(counters, WAVE_CHILDREN:WAVE_CHILDREN), slot.layout, nothing)

# The wavefront of one device on its first lane `lane`; returns its counts, the primary
# ranges it started and one record per iteration (cohort, budget, counts, advance reruns,
# whether the cohort was sorted, host seconds of the sort, of the advances with their
# counter reads, of the retire launch, of the branch with its counter read and of the
# start launches). `ktwin` holds the key stage's twins when the call sorts. `call` holds
# the depth, the explicit child slots or `nothing`, and the executor's rerun counter.
function device_wavefront!(lane::PoolLane, wf::Wavefront, twins, ktwin, slot, blocks::BlockCounter,
                           set::WaveSettings, cap::Int, call)
    T = wave_state(wf.advance)
    M = wave_steps(wf.advance)
    P = set.cohort
    dstart, dadv, dret = map(t -> program_twin!(t, lane), twins[1:3])
    dbr = length(twins) == 4 ? program_twin!(twins[4], lane) : nothing
    dkey = ktwin === nothing ? nothing : program_twin!(ktwin, lane)
    counters = lane_buffer(lane, (:wave, :counters), Int64, WAVE_COUNTERS)
    stage = lane_stage(lane, (:wave, :counters), Int64, WAVE_COUNTERS)
    zc = ZeroCounters(counters)
    kids = slot !== nothing && call.depth > 0
    E = slot === nothing ? Nothing : eltype(slot.photons)
    capC = !kids ? 0 : something(call.capacity, max(P, held_slots(lane, E)))
    C = slot === nothing ? nothing : child_queue(lane, E, 0, counters, slot)
    live_ctr = view(counters, WAVE_LIVE:WAVE_LIVE)
    done_ctr = view(counters, WAVE_DONE:WAVE_DONE)
    a, b = 1, 2
    nA = 0
    nadv = 0
    K = set.budget
    admitted = 0; spawned = 0; dropped = 0; retired = 0; idle = 0
    ranges = UnitRange{Int}[]
    iters = NamedTuple[]
    arrive!(blocks)
    while true
        kids && call.capacity !== nothing && (capC = call.capacity)
        live = done = children = drop = reruns = 0
        ts = t0 = t1 = t2 = t3 = time_ns()
        sorted = false
        local B, qa
        if nA > 0
            exhausted(blocks) && 4 * nA < P && K < cap && (K = K > cap ÷ 2 ? cap : 2K)
            D = lane_buffer(lane, (:wave, :done), T, nA)
            states = lane_buffer(lane, (:wave, a), T, nA)
            sorted = dkey !== nothing && sorts(wf.key, set.sort_every, nadv)
            A = sorted ? Cohort(states, sort_cohort!(lane, wf.key, dkey, states, nA)) : Cohort(states, nothing)
            sorted && (t0 = time_ns())
            nadv += 1
            while true
                # B takes the survivors, the opened children and the refill, D the stopped states
                B = lane_buffer(lane, (:wave, b), T, queue_slots(nA, capC, P, blocks))
                C = slot === nothing ? nothing : child_queue(lane, E, capC, counters, slot)
                qa = WaveQueues(A, StateBuffer(B, live_ctr), nothing, StateBuffer(D, done_ctr),
                                counters, K, 0, call.depth)
                launch!(lane, zc, zc, 1:1, 1, ())
                launch!(lane, wf.advance, Adapt.adapt(WaveSwap(qa, C), dadv), 1:nA, nA, ())
                copyto!(stage, 1, counters, 1, WAVE_COUNTERS)
                live, done, children, fault = stage[WAVE_LIVE], stage[WAVE_DONE],
                                              stage[WAVE_CHILDREN], stage[WAVE_FAULT]
                fault == 0 || error("wavefront: $fault states broke the advance on $(lane.dev)")
                live + done == nA ||
                    error("wavefront: the advance of $nA states admitted $live live and $done stopped on $(lane.dev)")
                live <= length(B) && done <= length(D) ||
                    error("wavefront: a queue overflowed on $(lane.dev): $live live in $(length(B)) slots, " *
                          "$done stopped in $(length(D))")
                kids && children > capC || break
                capC = children + children ÷ 8
                reruns += 1
                Threads.atomic_add!(call.reruns, 1)
            end
            t1 = time_ns()
            done > 0 && launch!(lane, wf.retire, Adapt.adapt(WaveSwap(qa, C), dret), 1:done, done, ())
            t2 = time_ns()
            drop = children
            if kids && children > 0
                kc = Generation(view(C.photons, 1:children), view(C.rngs, 1:children))
                qb = WaveQueues(nothing, StateBuffer(B, live_ctr), kc, nothing, counters, K, 0, call.depth)
                launch!(lane, wf.branch, Adapt.adapt(WaveSwap(qb, C), dbr), 1:children, children, ())
                copyto!(stage, 1, counters, 1, WAVE_COUNTERS)
                drop = stage[WAVE_DROPPED]
                stage[WAVE_LIVE] + drop == live + children && stage[WAVE_LIVE] <= length(B) ||
                    error("wavefront: the branch of $children children left $(stage[WAVE_LIVE]) live " *
                          "and $drop dropped after $live survivors on $(lane.dev)")
                live = stage[WAVE_LIVE]
            end
            t3 = time_ns()
            spawned += children
            dropped += drop
            retired += done
            idle = done > 0 ? 0 : idle + 1
            idle - 1 > M &&
                error("wavefront: $nA live states on $(lane.dev) retired nothing in $(M + 2) iterations")
        else
            B = lane_buffer(lane, (:wave, b), T, queue_slots(0, 0, P, blocks))
        end
        nB = live
        t4 = time_ns()
        r = nB < P ? take_blocks!(blocks, P - nB) : 1:0
        if !isempty(r)
            q = WaveQueues(nothing, StateBuffer(B, live_ctr), nothing, nothing, counters, K, nB, call.depth)
            launch!(lane, wf.start, Adapt.adapt(WaveSwap(q, C), dstart), r, length(r), ())
            push!(ranges, r)
            admitted += length(r)
            nB += length(r)
        end
        push!(iters, (nA = nA, K = K, live = live, done = done, children = children,
                      dropped = drop, reruns = reruns, started = nB - live, sorted = sorted,
                      sort = (t0 - ts) / 1e9, advance = (t1 - t0) / 1e9, retire = (t2 - t1) / 1e9,
                      branch = (t3 - t2) / 1e9, start = (time_ns() - t4) / 1e9))
        nB == 0 && break
        a, b = b, a
        nA = nB
    end
    return (admitted = admitted, spawned = spawned, dropped = dropped, retired = retired,
            ranges = ranges, iterations = length(iters), log = iters)
end
