# The device wavefront: a phase of the Monte Carlo cut into launches of stage programs
# over queues of the first lane of every device. A start launch emits a block of
# primaries into the cohort, an advance runs every live state for at most K loop
# iterations, a retire launch records and tallies the states that stopped; the host reads
# one counter block per iteration and refills the cohort from a block counter that the
# devices share. A state's march is cut only between loop iterations and every value it
# carries lives in the state, so the totals do not depend on K, the cohort, the refill,
# the device count or the order of the states.

"Loop iterations per state of an advance launch, by default."
const WAVE_BUDGET = 32
"Live states per device, by default."
const WAVE_COHORT = 2^18
"Consecutive primaries per block of the refill, by default."
const WAVE_REFILL = 2^14

"""
    WaveSettings(budget, cohort, refill, doubling, wavefront)

The settings of the wavefront that a call reads ([`with_wavefront_settings`](@ref)):
the budget `K` of an advance, the cohort `P`, the refill block `R`, the budget doubling
(`true`: up to the cap of the phase, `false`: off, an integer: up to that many loop
iterations) and whether a phase that the wavefront can serve runs on it (`false` runs
the cascade of the same programs).
"""
struct WaveSettings
    budget::Int
    cohort::Int
    refill::Int
    doubling::Union{Bool,Int}
    wavefront::Bool
end

const DEFAULT_WAVE = WaveSettings(WAVE_BUDGET, WAVE_COHORT, WAVE_REFILL, true, true)
const WAVE_SETTINGS = Base.ScopedValues.ScopedValue(DEFAULT_WAVE)

"""
    with_wavefront_settings(f; budget = WAVE_BUDGET, cohort = WAVE_COHORT,
                            refill = WAVE_REFILL, doubling = true, wavefront = true)

Run `f()` with these settings of the wavefront, read on the host by every wavefront call
inside `f` on the calling task; outside such a scope every call uses the defaults. `budget`
is the loop iterations per state of an advance (`typemax(Int)` runs every state to its
end in one advance), `cohort` the live states per device the refill fills up to, `refill`
the primaries per block a device takes from the shared block counter, `doubling` the
budget doubling once no block remains and the cohort is below a quarter of `cohort`
(`true` up to the cap of the phase, `false` off, an integer `k ≥ 1` up to `k`), and
`wavefront = false` runs the cascade of the same programs. Results do not depend on any
of them. Returns `f()`.
"""
function with_wavefront_settings(f; budget::Integer = WAVE_BUDGET, cohort::Integer = WAVE_COHORT,
                                 refill::Integer = WAVE_REFILL, doubling::Union{Bool,Integer} = true,
                                 wavefront::Bool = true)
    budget >= 1 && cohort >= 1 && refill >= 1 ||
        throw(ArgumentError("with_wavefront_settings: budget, cohort and refill must be positive"))
    doubling isa Bool || doubling >= 1 ||
        throw(ArgumentError("with_wavefront_settings: doubling must be true, false or a positive cap"))
    s = WaveSettings(Int(budget), Int(cohort), Int(refill),
                     doubling isa Bool ? doubling : Int(doubling), wavefront)
    return Base.ScopedValues.with(f, WAVE_SETTINGS => s)
end

"The wavefront settings of the calling task ([`with_wavefront_settings`](@ref))."
wave_settings() = WAVE_SETTINGS[]

settings_record(s::WaveSettings) = (budget = s.budget, cohort = s.cohort, refill = s.refill,
                                    doubling = s.doubling, wavefront = s.wavefront)

"""
    Wavefront(; start, advance, retire, primary, layer, doubling_cap = typemax(Int))

The programs of one Monte Carlo transport for [`foreach_wavefront!`](@ref): the stage
programs `start` (item `g`, a primary, placed into the cohort by [`place!`](@ref)),
`advance` (member `j` of the cohort: at most `K` loop iterations, then [`admit!`](@ref)
into `B` while live or into `D` once stopped, [`fault!`](@ref) for a state that broke
the contract) and `retire` (slot `j` of `D`: its record and tally), each called as
`prog(item, i)` with a [`WaveQueues`](@ref) placeholder that every launch binds; and the
history programs `primary` and `layer` of the same transport, which the cascade runs
where the wavefront does not serve. `doubling_cap` is the most loop iterations the
budget doubles to in this phase. The stage programs must agree on
[`wave_state`](@ref); the advance answers [`wave_steps`](@ref).
"""
struct Wavefront{S,A,R,P,L}
    start::S
    advance::A
    retire::R
    primary::P
    layer::L
    doubling_cap::Int
end

function Wavefront(; start, advance, retire, primary, layer, doubling_cap::Integer = typemax(Int))
    T = wave_state(start)
    wave_state(advance) === T && wave_state(retire) === T ||
        throw(ArgumentError("Wavefront: the stage programs disagree on the state type"))
    wave_steps(advance) >= 1 || throw(ArgumentError("Wavefront: the advance's step limit must be positive"))
    doubling_cap >= 1 || throw(ArgumentError("Wavefront: doubling_cap must be positive"))
    return Wavefront(start, advance, retire, primary, layer, Int(doubling_cap))
end

"""
    foreach_wavefront!(exec, wf::Wavefront, n, depth; capacity = nothing,
                       chunk = nothing) -> Int

Run the transport of `wf` over the primaries `1:n` with at most `depth` scattering
generations, as [`foreach_cascade!`](@ref)`(exec, wf.primary, wf.layer, n, depth)` would,
and return the children spawned. The device wavefront serves the call when `exec` is a
`PoolExecutor` whose every device answers [`wavefront`](@ref)`(dev) = true`, the settings
allow it ([`with_wavefront_settings`](@ref)) and the call is a phase the wavefront has
stages for: `depth = 0`, the children of the primaries counted and not tracked. Any
other call runs the cascade with `capacity` and, when given, `chunk` (the cascade's
default otherwise).

On the wavefront each device runs its own cohort on its first lane until no primary
block remains and its cohort is empty. Per iteration: the counter block zeroed; the
advance over the cohort, at most `K` loop iterations per state; one read of the counter
block, each of a fault, `live + done ≠ nA` and a queue beyond its slots an error; the
retire launch over the stopped states; the refill: while fewer than `P` states are live
and blocks remain, the device takes `max(1, min(cld(P − nB, R), cld(r, ndev)))` of the
`r` remaining blocks of `R` consecutive primaries at once and runs the start over them;
then the queues swap. With doubling on, `K` doubles every iteration once no block
remains and fewer than `P/4` states are live. A device whose cohort retires nothing in
`M + 2` consecutive iterations (`M` = [`wave_steps`](@ref) of the advance) is an error.
At the end the started primaries must equal `n`, their ranges must partition `1:n` and
the retired states must equal `n` plus the children spawned less those dropped. In
`finally` the child slot of every program is emptied.

Determinism: the accumulated totals and the returned count are those of the cascade,
bitwise when the stage programs give the history programs' bits, whatever the settings,
the device count and the arrival order.
"""
function foreach_wavefront!(exec::Executor, wf::Wavefront, n::Integer, depth::Integer;
                            capacity = nothing, chunk = nothing)
    n >= 0 || throw(ArgumentError("foreach_wavefront!: n must be non-negative"))
    depth >= 0 || throw(ArgumentError("foreach_wavefront!: depth must be non-negative"))
    set = wave_settings()
    serves(exec, wf, depth, set) && return run_wavefront!(exec, wf, Int(n), set)
    kw = chunk === nothing ? (; capacity) : (; capacity, chunk)
    return foreach_cascade!(exec, wf.primary, wf.layer, n, depth; kw...)
end

# Whether the wavefront serves a call: a pool whose devices answer `wavefront`, the
# setting, and a phase with stages (the primaries alone; their children counted). Threads
# and every other executor run the cascade.
serves(::Executor, wf, depth, set) = false
serves(e::PoolExecutor, wf::Wavefront, depth, set::WaveSettings) =
    set.wavefront && depth == 0 && all(wavefront, e.pool.devices)

"""
    run_wavefront!(exec::PoolExecutor, wf::Wavefront, n, set::WaveSettings) -> Int

The wavefront of [`foreach_wavefront!`](@ref) over the primaries `1:n` with the
settings `set`, on every device of the pool; returns the children spawned. Records the
settings and the primaries admitted and children spawned per device for
[`describe`](@ref).
"""
function run_wavefront!(exec::PoolExecutor, wf::Wavefront, n::Int, set::WaveSettings)
    pool = exec.pool
    exec.reruns[] = 0
    slots = map(child_slot, (wf.start, wf.advance, wf.retire, wf.primary, wf.layer))
    blocks = BlockCounter(n, set.refill, length(pool.devices))
    twins = map(p -> ProgramTwins(pool, p), (wf.start, wf.advance, wf.retire))
    cap = set.doubling === true ? wf.doubling_cap : set.doubling === false ? set.budget :
          set.doubling
    res = try
        n == 0 ? [empty_device() for _ in pool.devices] :
        foreach_device(pool) do lane
            device_wavefront!(lane, wf, twins, slots[2], blocks, set, cap)
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
one atomic counter of the blocks taken that the `ndev` devices of a call share.
"""
struct BlockCounter
    n::Int
    R::Int
    ndev::Int
    blocks::Int
    taken::Threads.Atomic{Int}
end
BlockCounter(n::Int, R::Int, ndev::Int) = BlockCounter(n, R, ndev, cld(n, R), Threads.Atomic{Int}(0))

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

# The wavefront of one device on its first lane `lane`; returns its counts, the primary
# ranges it started and one record per iteration (cohort, budget, counts, host seconds of
# the advance with its counter read, of the retire launch and of the start launches).
function device_wavefront!(lane::PoolLane, wf::Wavefront, twins, slot, blocks::BlockCounter,
                           set::WaveSettings, cap::Int)
    T = wave_state(wf.advance)
    M = wave_steps(wf.advance)
    P = set.cohort
    dstart, dadv, dret = map(t -> program_twin!(t, lane), twins)
    counters = lane_buffer(lane, (:wave, :counters), Int64, WAVE_COUNTERS)
    stage = lane_stage(lane, (:wave, :counters), Int64, WAVE_COUNTERS)
    zc = ZeroCounters(counters)
    drop = slot === nothing ? nothing :
           ChildBuffer(lane_children(lane, eltype(slot.photons), :a, 0)...,
                       view(counters, WAVE_CHILDREN:WAVE_CHILDREN), slot.layout, nothing)
    live_ctr = view(counters, WAVE_LIVE:WAVE_LIVE)
    done_ctr = view(counters, WAVE_DONE:WAVE_DONE)
    a, b = 1, 2
    nA = 0
    K = set.budget
    admitted = 0; spawned = 0; dropped = 0; retired = 0; idle = 0
    ranges = UnitRange{Int}[]
    iters = NamedTuple[]
    while true
        # B takes the survivors and the refill, D the stopped states
        B = lane_buffer(lane, (:wave, b), T,
                        max(nA, 1, min(sat_add(P, blocks.R - 1), sat_add(nA, remaining(blocks)))))
        live = done = children = 0
        t0 = t1 = t2 = time_ns()
        if nA > 0
            exhausted(blocks) && 4 * nA < P && K < cap && (K = K > cap ÷ 2 ? cap : 2K)
            D = lane_buffer(lane, (:wave, :done), T, nA)
            q = WaveQueues(Cohort(lane_buffer(lane, (:wave, a), T, nA), nothing),
                           StateBuffer(B, live_ctr), StateBuffer(D, done_ctr), counters, K, 0)
            launch!(lane, zc, zc, 1:1, 1, ())
            launch!(lane, wf.advance, Adapt.adapt(WaveSwap(q, drop), dadv), 1:nA, nA, ())
            copyto!(stage, 1, counters, 1, WAVE_COUNTERS)
            live, done, children, fault = stage[WAVE_LIVE], stage[WAVE_DONE],
                                          stage[WAVE_CHILDREN], stage[WAVE_FAULT]
            fault == 0 || error("wavefront: $fault states broke the advance on $(lane.dev)")
            live + done == nA ||
                error("wavefront: the advance of $nA states admitted $live live and $done stopped on $(lane.dev)")
            live <= length(B) && done <= length(D) ||
                error("wavefront: a queue overflowed on $(lane.dev): $live live in $(length(B)) slots, " *
                      "$done stopped in $(length(D))")
            spawned += children
            dropped += children
            t1 = time_ns()
            done > 0 && launch!(lane, wf.retire, Adapt.adapt(WaveSwap(q, drop), dret), 1:done, done, ())
            t2 = time_ns()
            retired += done
            idle = done > 0 ? 0 : idle + 1
            idle - 1 > M &&
                error("wavefront: $nA live states on $(lane.dev) retired nothing in $(M + 2) iterations")
        end
        nB = live
        while nB < P
            r = take_blocks!(blocks, P - nB)
            isempty(r) && break
            q = WaveQueues(nothing, StateBuffer(B, live_ctr), nothing, counters, K, nB)
            launch!(lane, wf.start, Adapt.adapt(WaveSwap(q, drop), dstart), r, length(r), ())
            push!(ranges, r)
            admitted += length(r)
            nB += length(r)
        end
        push!(iters, (nA = nA, K = K, live = live, done = done, children = children,
                      started = nB - live, advance = (t1 - t0) / 1e9, retire = (t2 - t1) / 1e9,
                      start = (time_ns() - t2) / 1e9))
        nB == 0 && break
        a, b = b, a
        nA = nB
    end
    return (admitted = admitted, spawned = spawned, dropped = dropped, retired = retired,
            ranges = ranges, iterations = length(iters), log = iters)
end
