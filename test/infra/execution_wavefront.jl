# The device wavefront on pools of `WaveHost`s: a toy transport whose photon of key `g`
# lives `life(g)` loop iterations, folds every iteration into a checksum and spawns a child
# of the next generation at some of them, as history programs (the cascade) and as stage
# programs (the wavefront). Its totals, member checksum and child count are bitwise the
# cascade's on threads and host pools, at depths 0 to 3, on 1, 2, 3 and 5 devices, at every
# budget with the doubling on, capped and off, at every cohort and refill block, with `n`
# not a multiple of the block, at every child capacity (the advance runs again for its
# children); the children of the last generation are counted and dropped; the started
# primaries, the retirements and the per-device counts of `describe` close; every device
# takes blocks, also of a call smaller than one cohort; a call with
# children doubles its budget to at most `WAVE_SPAWN_CAP`; the child slots carry over
# calls; a state that cannot advance and a state that reports a fault are errors;
# accumulator twins and child slots are released and the residency cache does not grow
# over calls; calls outside `with_wavefront_settings` use the defaults; threads, host
# pools, executors of other types and phases without stages take the cascade. The
# coherence sort: its order is a permutation of the cohort with non-decreasing keys on
# every device kind, and the toy's bits and per-iteration counts do not depend on it.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Numerics: MCRng, MCStreamLayout, primary_rng
using BlackLightPlus.MonteCarlo: Superphoton
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, ExactScalars, ChildBuffer, Generation,
                     foreach_cascade!, foreach_wavefront!, with_wavefront_settings, Wavefront,
                     WaveQueues, admit!, admit_child!, place!, fault!, gather!, add!, spawn!,
                     describe, wave_log, HostDevice, resident_count, KeyCount, count_key!,
                     Cohort
import BlackLightPlus.Execution
const Adapt = BlackLightPlus.Execution.Adapt

const DK = Main.Decks
const SEED = 0x7a5e
const LIFE = 23

"The loop iterations of the photon of key `g`, read from the table `lives` (a residency array)."
@inline life(lives, g) = @inbounds lives[(g - 1) % length(lives) + 1]
const LIVES = Int32[1 + (g * 7919) % LIFE for g in 1:97]
"Longer lives, for the budget doubling."
const LONG = Int32[1 + (g * 7919) % 400 for g in 1:97]

# One loop iteration: the checksum after iteration `it` of the photon of key `g`.
@inline fold(acc::UInt64, g::Int, it::Int) = acc * 0x9e3779b97f4a7c15 + UInt64(g) * UInt64(it) + 0x1
# Whether iteration `it` of the photon of key `g` spawns a child, and the child's key.
@inline spawns(g::Int, it::Int) = (g + it) % 5 == 0
@inline child_key(g::Int, it::Int) = 32g + it
# The child of iteration `it` of the photon of key `g` and generation `gen - 1`.
@inline function spawn_child!(buf, g, it, gen)
    buf === nothing && return nothing
    k = child_key(g, it)
    spawn!(buf, Superphoton(SVector(0.0, Float64(k), 0.0, 0.0), SVector(-1.0, 0.0, 0.0, 1.0), 1.0;
                            nscatt = gen), primary_rng(SEED, k))
    return nothing
end
# The totals of a photon: its key, its checksum (exact in Float64) and one count per path.
@inline function tally!(sc, g::Int, acc::UInt64)
    add!(sc, 1, g / 3.0)
    add!(sc, 2, Float64(acc % (UInt64(1) << 40)))
    return nothing
end
# The whole life of the photon of key `g` and generation `gen`.
@inline function whole_life!(lives, buf, sc, g::Int, gen::Integer)
    acc = UInt64(0)
    for it in 1:life(lives, g)
        acc = fold(acc, g, it)
        spawns(g, it) && spawn_child!(buf, g, it, gen + 1)
    end
    tally!(sc, g, acc)
    return nothing
end

"The state of a toy photon: its key, the next loop iteration, the checksum so far and its generation."
struct Toy
    g::Int
    it::Int32
    acc::UInt64
    gen::Int32
end
Execution.generation(s::Toy) = Int(s.gen)

# History programs: a primary, and a child of a generation the cascade fills.
struct ToyPrimary{L,S,P,C} <: Function
    lives::L
    sc::S
    path::P
    buf::C
end
Adapt.@adapt_structure ToyPrimary
function (p::ToyPrimary)(g, i)
    whole_life!(p.lives, p.buf, p.sc, g, 0)
    add!(p.path, 1, 1.0)
    return nothing
end
struct ToyLayer{G,L,S,P,C} <: Function
    generation::G
    lives::L
    sc::S
    path::P
    buf::C
end
Adapt.@adapt_structure ToyLayer
function (p::ToyLayer)(item, i)
    sph = @inbounds p.generation.photons[item]
    whole_life!(p.lives, p.buf, p.sc, Int(sph.x[2]), sph.nscatt)
    add!(p.path, 1, 1.0)
    return nothing
end

# Stage programs of the same transport.
struct ToyStart{P,Q} <: Function
    path::P
    queues::Q
end
Adapt.@adapt_structure ToyStart
function (p::ToyStart)(g, i)
    place!(p.queues, i, Toy(g, Int32(1), UInt64(0), Int32(0)))
    add!(p.path, 2, 1.0)
    return nothing
end

# `mode`: 0 the transport, 1 a state that never advances, 2 a state that reports a fault.
struct ToyAdvance{L,C,Q} <: Function
    lives::L
    buf::C
    mode::Int
    M::Int
    queues::Q
end
Adapt.@adapt_structure ToyAdvance
function (p::ToyAdvance)(item, i)
    q = p.queues
    s = q.A[item]
    L = Int(life(p.lives, s.g))
    acc = s.acc
    last = p.mode == 1 ? Int(s.it) - 1 : min(Int(s.it) - 1 + min(q.K, L), L)
    for it in Int(s.it):last
        acc = fold(acc, s.g, it)
        spawns(s.g, it) && spawn_child!(p.buf, s.g, it, s.gen + 1)
    end
    s2 = Toy(s.g, Int32(last + 1), acc, s.gen)
    p.mode == 2 && s.g == 17 && fault!(q)
    last + 1 > L ? admit!(q.D, s2) : admit!(q.B, s2)
    return nothing
end

struct ToyRetire{S,P,Q} <: Function
    sc::S
    path::P
    queues::Q
end
Adapt.@adapt_structure ToyRetire
function (p::ToyRetire)(item, i)
    s = p.queues.D[item]
    tally!(p.sc, s.g, s.acc)
    add!(p.path, 3, 1.0)
    return nothing
end

struct ToyBranch{P,Q} <: Function
    path::P
    queues::Q
end
Adapt.@adapt_structure ToyBranch
function (p::ToyBranch)(item, i)
    sph = @inbounds p.queues.C.photons[item]
    admit_child!(p.queues, Toy(Int(sph.x[2]), Int32(1), UInt64(0), sph.nscatt))
    add!(p.path, 4, 1.0)
    return nothing
end

# The key stage: `nk` key values of a photon's key and loop iteration.
struct ToyKey{Q} <: Function
    nk::Int
    queues::Q
end
Adapt.@adapt_structure ToyKey
@inline toy_key(s::Toy, nk) = (s.g * 31 + Int(s.it)) % nk
function (p::ToyKey)(item, i)
    count_key!(p.queues, item, toy_key(p.queues.A[item], p.nk))
    return nothing
end
Execution.wave_keys(p::ToyKey) = p.nk

Execution.wave_state(::Union{ToyStart,ToyAdvance,ToyRetire,ToyBranch,ToyKey}) = Toy
Execution.wave_steps(p::ToyAdvance) = p.M

"""
The toy transport over `1:n`, spawning when `spawning`, with a branch stage when `branch`;
`mode` of the advance, the life table `lives` and its step limit `M`, `nk` key values of
the key stage.
"""
function toy(n; spawning = true, branch = true, mode = 0, lives = LIVES, M = LIFE, nk = 13)
    sc = ExactScalars(2)
    path = ExactScalars(4)
    buf = spawning ? ChildBuffer(Superphoton, 0, MCStreamLayout(max(n, 1), 64, 1)) : nothing
    lives = copy(lives)
    wf = Wavefront(; start = ToyStart(path, WaveQueues()),
                   advance = ToyAdvance(lives, buf, mode, M, WaveQueues()),
                   retire = ToyRetire(sc, path, WaveQueues()),
                   branch = branch ? ToyBranch(path, WaveQueues()) : nothing,
                   key = ToyKey(nk, KeyCount()),
                   primary = ToyPrimary(lives, sc, path, buf),
                   layer = ToyLayer(Generation(Superphoton[], MCRng[]), lives, sc, path, buf))
    return (wf = wf, sc = sc, path = path, buf = buf, lives = lives)
end

"Run the toy on `exec` with `depth`; the totals, the child count and the counts per path."
function run_toy(exec, n; depth = 0, capacity = nothing, kw...)
    t = toy(n; kw...)
    c = foreach_wavefront!(exec, t.wf, n, depth; capacity)
    gather!(t.sc); gather!(t.path)
    return (totals = (copy(t.sc.acc), copy(t.sc.n)), children = c, path = copy(t.path.n),
            pending = length(t.sc.twins.list) + length(t.path.twins.list),
            slot = t.buf === nothing ? 0 : length(t.buf.twins.list))
end

"The message of the exception `f()` throws, `nothing` when it returns."
failure(f) = try
    f()
    nothing
catch e
    sprint(showerror, e)
end

"An executor of another type that forwards the cascade to threads and records its keywords."
struct Forwarding <: Execution.Executor
    inner::ThreadedExecutor
    seen::Vector{Any}
end
Execution.foreach_cascade!(e::Forwarding, p, l, n::Integer, d::Integer; kw...) =
    (push!(e.seen, values(kw)); foreach_cascade!(e.inner, p, l, n, d; kw...))

const N = 203
const DEFAULTS = (budget = 32, cohort = 2^18, refill = 2^14, doubling = true, wavefront = true,
                  sort_every = 1)

@testset "device wavefront" begin
    refs = Dict(d => run_toy(ThreadedExecutor(chunk = 7), N; depth = d) for d in 0:3)
    ref = refs[0]
    @test ref.children > 100 && ref.path == [N, 0, 0, 0]
    # the photons the cascade tracks: the primaries and the children of generations below the depth
    tracked = Dict(d => Int(refs[d].path[1]) for d in 0:3)
    @test tracked[0] == N && issorted([tracked[d] for d in 0:3]) && tracked[3] > tracked[2] > 2N
    @test all(d -> refs[d].children >= tracked[d] - N, 0:3)
    expect(d) = (totals = refs[d].totals, children = refs[d].children)
    same(r, d = 0) = (totals = r.totals, children = r.children) == expect(d)
    # on the wavefront: every primary started, every tracked photon retired, every child branched
    waved(r, d) = r.path == [0, N, tracked[d], d == 0 ? 0 : refs[d].children]

    @testset "threads, host pools, other executors and phases without stages: the cascade" begin
        for exec in (ThreadedExecutor(), PoolExecutor(DK.hosts(1)), PoolExecutor(DK.hosts(2); chunk = 11)),
            d in (0, 2)
            r = run_toy(exec, N; depth = d)
            @test same(r, d) && r.path == [tracked[d], 0, 0, 0]
        end
        ex = PoolExecutor(DK.wavehosts(2); chunk = 13)
        r = with_wavefront_settings(wavefront = false) do
            run_toy(ex, N; depth = 2)
        end
        @test same(r, 2) && r.path == [tracked[2], 0, 0, 0] && describe(ex).schedule == :cascade
        @test describe(ex).settings === nothing && sum(describe(ex).admitted) == N
        @test sum(describe(ex).spawned) == refs[2].children
        # any other executor: the cascade it forwards, with the caller's keywords only
        fx = Forwarding(ThreadedExecutor(), Any[])
        r = run_toy(fx, N)
        @test same(r) && r.path == [N, 0, 0, 0] && fx.seen == [(capacity = nothing,)]
        t = toy(N)
        foreach_wavefront!(fx, t.wf, N, 0; capacity = 5, chunk = 9)
        @test fx.seen[end] == (capacity = 5, chunk = 9)
        # a transport without a branch stage: a scattering phase takes the cascade
        r1 = run_toy(ex, N; depth = 1, branch = false)
        @test same(r1, 1) && r1.path == [tracked[1], 0, 0, 0] && describe(ex).schedule == :cascade
        # with a branch, every depth runs on the wavefront
        r1 = run_toy(ex, N; depth = 1)
        @test same(r1, 1) && waved(r1, 1) && describe(ex).schedule == :wavefront
    end

    @testset "every device count, budget, doubling, cohort and refill: the cascade's bits" begin
        for nd in DK.NDEV, d in (0, 1, 3)
            ex = PoolExecutor(DK.wavehosts(nd))
            for (K, dbl) in ((1, false), (3, true), (3, false), (64, 4), (64, true),
                             (typemax(Int), false)),
                (P, R) in ((1, 1), (7, 5), (2^10, 5), (2^10, 1))
                r = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R) do
                    run_toy(ex, N; depth = d)
                end
                dsc = describe(ex)
                ok = same(r, d) && waved(r, d) && dsc.schedule == :wavefront &&
                     sum(dsc.admitted) == N && sum(dsc.spawned) == refs[d].children &&
                     dsc.settings == (budget = K, cohort = P, refill = R, doubling = dbl, wavefront = true,
                                      sort_every = Execution.WAVE_SORT)
                ok || @info "wavefront differs" nd d K dbl P R r.path dsc
                @test ok
            end
        end
    end

    @testset "child slots: explicit capacities rerun the advance, the default carries over calls" begin
        for nd in (1, 2, 3), cap in (1, 3, 64, nothing), d in (1, 3)
            ex = PoolExecutor(DK.wavehosts(nd))
            r = with_wavefront_settings(cohort = 16, refill = 5) do
                run_toy(ex, N; depth = d, capacity = cap)
            end
            dsc = describe(ex)
            @test same(r, d) && waved(r, d) && dsc.schedule == :wavefront
            cap in (1, 3) && @test dsc.reruns > 0
            log = reduce(vcat, wave_log(ex))
            @test sum(it -> it.reruns, log) == dsc.reruns && sum(it -> it.children, log) == refs[d].children
        end
        # an explicit capacity returns at every iteration: every iteration with more children reruns
        ex = PoolExecutor(DK.wavehosts(1))
        with_wavefront_settings(cohort = 16, refill = 5) do
            run_toy(ex, N; depth = 2, capacity = 1)
        end
        log = only(wave_log(ex))
        @test all(it -> it.reruns == (it.children > 1 ? 1 : 0), log)
        # without one, the slots start at the cohort and keep every growth, over calls too
        ex = PoolExecutor([DK.WaveHost(41)])
        runs = map(1:2) do _
            with_wavefront_settings(cohort = 4, refill = 2) do
                r = run_toy(ex, N; depth = 3)
                (r, describe(ex).reruns, reduce(vcat, wave_log(ex)))
            end
        end
        @test same(runs[1][1], 3) && same(runs[2][1], 3)
        @test runs[1][2] > 0 && runs[2][2] == 0
        @test sum(it -> it.reruns, runs[1][3]) == runs[1][2]
    end

    @testset "the devices of the comparisons ($(DK.DEVICE_KIND)): the cascade's bits" begin
        for nd in (1, 2)
            devs = DK.devices(nd)
            ex = PoolExecutor(devs)
            on = all(Execution.wavefront, devs)
            for (K, dbl, P, R, cap) in ((1, false, 7, 5, 1), (32, true, 2^10, 1, nothing),
                                        (typemax(Int), false, 64, 3, 3)),
                d in (0, 2)
                r = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R) do
                    run_toy(ex, N; depth = d, capacity = cap)
                end
                @test same(r, d) && (on ? waved(r, d) : r.path == [tracked[d], 0, 0, 0])
                @test describe(ex).schedule == (on ? :wavefront : :cascade)
            end
        end
    end

    @testset "budget doubling: on, capped, off; at most WAVE_SPAWN_CAP with children" begin
        ex = PoolExecutor(DK.wavehosts(1))
        Ks(dbl; d = 0, K = 1, P = 64, kw...) = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = 8) do
            r = run_toy(ex, N; depth = d, kw...)
            isempty(kw) && @test same(r, d)
            [it.K for it in only(wave_log(ex)) if it.nA > 0]
        end
        on, capped, off = Ks(true), Ks(4), Ks(false)
        @test all(==(1), off) && maximum(capped) == 4 && maximum(on) > 4
        @test issorted(on) && issorted(capped)
        # doubling starts once no block remains and the cohort is below a quarter of P
        log = only(wave_log(ex))
        @test sum(it -> it.started, log) == N && sum(it -> it.done, log) == N
        # long lives, every block taken at once: past the cap without children, at the cap
        # with them, an explicit cap honoured
        long = (P = 2^20, lives = LONG, M = 400)
        thin = Ks(true; K = 32, long...)
        scat = Ks(true; d = 2, K = 32, long...)
        wide = Ks(1024; d = 2, K = 32, long...)
        @test maximum(thin) > Execution.WAVE_SPAWN_CAP
        @test maximum(scat) == Execution.WAVE_SPAWN_CAP && issorted(scat)
        @test maximum(wide) > Execution.WAVE_SPAWN_CAP
    end

    @testset "every device takes blocks at small settings, and of a call smaller than a cohort" begin
        for nd in (2, 3), d in (0, 2), (P, R) in ((7, 1), (2^10, 5))
            ex = PoolExecutor(DK.wavehosts(nd))
            r = with_wavefront_settings(cohort = P, refill = R) do
                run_toy(ex, N; depth = d)
            end
            dsc = describe(ex)
            @test same(r, d) && all(>(0), dsc.admitted) && sum(dsc.admitted) == N
            @test all(>(0), dsc.iterations) && length(wave_log(ex)) == nd
            # one claim per iteration: at most a share of the blocks remaining
            @test all(lg -> all(it -> it.started <= R * cld(cld(N, R), nd), lg), wave_log(ex))
        end
        # a device waits for the others of its call at its first refill, at most WAVE_ARRIVAL seconds
        b = Execution.BlockCounter(10, 2, 2)
        t = @elapsed Execution.arrive!(b, 0.2)
        @test b.arrived[] == 1 && 0.2 <= t < 2.0
        t = @elapsed Execution.arrive!(b, 5.0)
        @test b.arrived[] == 2 && t < 1.0
    end

    @testset "children of the last generation: counted and dropped; none without a slot" begin
        ex = PoolExecutor(DK.wavehosts(2))
        r = run_toy(ex, N)
        @test same(r) && r.slot == 0 && r.pending == 0
        log = reduce(vcat, wave_log(ex))
        @test sum(it -> it.dropped, log) == ref.children
        for d in 1:3
            r = run_toy(ex, N; depth = d)
            log = reduce(vcat, wave_log(ex))
            @test same(r, d) && r.slot == 0 && r.pending == 0
            # dropped: the children of generation d, the children the cascade counts and does not track
            @test sum(it -> it.dropped, log) == refs[d].children - (tracked[d] - N)
        end
        r0 = run_toy(ex, N; spawning = false, depth = 2)
        @test r0.children == 0 && r0.totals == ref.totals && r0.path == [0, N, N, 0]
    end

    @testset "the coherence sort: the cascade's bits at every sort_every, the counts of the unsorted call" begin
        for nd in (1, 2, 3), d in (0, 2), every in (1, 2, 4),
            (K, dbl, P, R, cap) in ((1, false, 7, 5, nothing), (3, true, 64, 5, 1), (32, true, 2^10, 1, nothing))
            ex = PoolExecutor(DK.wavehosts(nd))
            r = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R, sort_every = every) do
                run_toy(ex, N; depth = d, capacity = cap)
            end
            dsc = describe(ex)
            @test same(r, d) && waved(r, d) && dsc.settings.sort_every == every
            cap == 1 && d > 0 && @test dsc.reruns > 0
            # every `every`-th advance of a device sorts, from its first
            for lg in wave_log(ex)
                adv = [it for it in lg if it.nA > 0]
                @test [it.sorted for it in adv] == [(m - 1) % every == 0 for m in eachindex(adv)]
                @test all(it -> it.sorted || it.sort == 0, lg)
            end
            if nd == 1
                # one device: the same iterations with the sort off
                off = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R, sort_every = 0) do
                    run_toy(ex, N; depth = d, capacity = cap)
                    only(wave_log(ex))
                end
                cut(lg) = [(it.nA, it.K, it.live, it.done, it.children, it.dropped, it.started) for it in lg]
                @test cut(only(wave_log(ex))) == cut(off) && !any(it -> it.sorted, off)
            end
        end
        # a wavefront without a key stage never sorts
        t = toy(N)
        bare = Wavefront(; start = t.wf.start, advance = t.wf.advance, retire = t.wf.retire,
                         branch = t.wf.branch, primary = t.wf.primary, layer = t.wf.layer)
        ex = PoolExecutor(DK.wavehosts(2))
        with_wavefront_settings(sort_every = 1) do
            foreach_wavefront!(ex, bare, N, 2)
        end
        @test !any(it -> it.sorted, reduce(vcat, wave_log(ex)))
    end

    @testset "the sort's order on $(kind): a permutation of the cohort with non-decreasing keys" for
            (kind, devs) in (("WaveHost", DK.wavehosts(1)), (DK.DEVICE_KIND, DK.devices(1)))
        pool = PoolExecutor(devs).pool
        for nk in (1, 7, 8192), nA in (1, 2, 37, 1000)
            states = [Toy(g, Int32(1 + g % 11), UInt64(g), Int32(0)) for g in 1:nA]
            key = ToyKey(nk, KeyCount())
            keys = [toy_key(s, nk) for s in states]
            tw = Execution.ProgramTwins(pool, key)
            order = only(Execution.foreach_device(pool) do lane
                d = Execution.lane_buffer(lane, (:test, :states), Toy, nA)
                copyto!(d, 1, states, 1, nA)
                o = Execution.sort_cohort!(lane, key, Execution.program_twin!(tw, lane), d, nA)
                h = Vector{Int32}(undef, nA)
                copyto!(h, 1, o, 1, nA)
                h
            end)
            @test sort(order) == 1:nA
            @test issorted(keys[order])
            @test [Cohort(states, order)[j] for j in 1:nA] == states[order]
        end
        @test Execution.key_cursors!(Int32[3, 0, 2, 1]) == Int32[0, 3, 3, 5]
    end

    @testset "contract errors: a state that cannot advance, a fault" begin
        ex = PoolExecutor(DK.wavehosts(1))
        m = failure(() -> with_wavefront_settings(cohort = 4, refill = 2) do
            run_toy(ex, 40; mode = 1)
        end)
        @test m !== nothing && occursin("retired nothing in $(LIFE + 2) iterations", m)
        m = failure(() -> run_toy(ex, 40; mode = 2, depth = 2))
        @test m !== nothing && occursin("broke the advance", m)
        @test_throws ArgumentError foreach_wavefront!(ex, toy(10).wf, 10, 1; capacity = 0)
    end

    @testset "twins and residency after a call" begin
        ex = PoolExecutor(DK.wavehosts(2))
        t = toy(N)
        n0 = [resident_count(ex, d) for d in DK.wavehosts(2)]
        for _ in 1:3
            with_wavefront_settings(cohort = 7, refill = 3) do
                foreach_wavefront!(ex, t.wf, N, 2)
            end
            @test length(t.buf.twins.list) == 0 && t.buf.count[1] == 0
            @test length(t.sc.twins.list) > 0
            gather!(t.sc); gather!(t.path)
            @test length(t.sc.twins.list) == 0 && length(t.path.twins.list) == 0
            @test [resident_count(ex, d) for d in DK.wavehosts(2)] == n0 .+ 1    # the life table
        end
        @test t.path.n == [0, 3N, 3tracked[2], 3refs[2].children]
    end

    @testset "settings: defaults outside a scope, checked arguments" begin
        ex = PoolExecutor(DK.wavehosts(1))
        run_toy(ex, N)
        @test describe(ex).settings == DEFAULTS
        with_wavefront_settings(budget = 5) do
            run_toy(ex, N)
            @test describe(ex).settings == merge(DEFAULTS, (budget = 5,))
            with_wavefront_settings(cohort = 9) do
                run_toy(ex, N)
                @test describe(ex).settings == merge(DEFAULTS, (cohort = 9,))
            end
        end
        run_toy(ex, N)
        @test describe(ex).settings == DEFAULTS
        @test with_wavefront_settings(() -> 42; budget = typemax(Int)) == 42
        for kw in ((budget = 0,), (cohort = 0,), (refill = -1,), (doubling = 0,), (sort_every = -1,))
            @test_throws ArgumentError with_wavefront_settings(() -> nothing; kw...)
        end
        t = toy(10)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.layer, primary = t.wf.primary,
                                             layer = t.wf.layer)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.retire, branch = t.wf.layer,
                                             primary = t.wf.primary, layer = t.wf.layer)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.retire, primary = t.wf.primary,
                                             layer = t.wf.layer, doubling_cap = 0)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.retire, key = t.wf.layer,
                                             primary = t.wf.primary, layer = t.wf.layer)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.retire, key = ToyKey(0, KeyCount()),
                                             primary = t.wf.primary, layer = t.wf.layer)
        @test_throws ArgumentError Execution.wave_keys(t.wf.advance)
        @test run_toy(ex, 0).children == 0 && describe(ex).admitted == [0]
        @test_throws ArgumentError foreach_wavefront!(ex, t.wf, -1, 0)
        @test DK.device_isbits(t.wf.advance) && DK.device_isbits(t.wf.start) &&
              DK.device_isbits(t.wf.branch) && DK.device_isbits(t.wf.key)
        @test_throws ArgumentError Execution.generation(1.0)
    end
end
