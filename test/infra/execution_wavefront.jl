# The device wavefront on pools of `WaveHost`s: a toy transport whose primary `g` lives
# `life(g)` loop iterations, folds every iteration into a checksum and spawns a child at
# some of them, as history programs (the cascade) and as stage programs (the wavefront).
# Its totals, member checksum and child count are bitwise the cascade's on threads and
# host pools, on 1, 2, 3 and 5 devices, at every budget with the doubling on, capped and
# off, at every cohort and refill block, with `n` not a multiple of the block; the
# children of the primaries (the last generation) are counted and dropped; the started
# primaries, the retirements and the per-device counts of `describe` close; a state that
# cannot advance and a state that reports a fault are errors; accumulator twins and child
# slots are released and the residency cache does not grow over calls; calls outside
# `with_wavefront_settings` use the defaults; threads, host pools, executors of other
# types and phases without stages take the cascade.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Numerics: MCRng, MCStreamLayout, primary_rng
using BlackLightPlus.MonteCarlo: Superphoton
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, ExactScalars, ChildBuffer, Generation,
                     foreach_cascade!, foreach_wavefront!, with_wavefront_settings, Wavefront,
                     WaveQueues, admit!, place!, fault!, gather!, add!, spawn!, child_stream,
                     describe, wave_log, HostDevice, resident_count
import BlackLightPlus.Execution
const Adapt = BlackLightPlus.Execution.Adapt

const DK = Main.Decks
const SEED = 0x7a5e
const LIFE = 23

"The loop iterations of primary `g`, read from the table `lives` (a residency array)."
@inline life(lives, g) = @inbounds lives[(g - 1) % length(lives) + 1]
const LIVES = Int32[1 + (g * 7919) % LIFE for g in 1:97]

# One loop iteration: the checksum after iteration `it` of primary `g`.
@inline fold(acc::UInt64, g::Int, it::Int) = acc * 0x9e3779b97f4a7c15 + UInt64(g) * UInt64(it) + 0x1
# Whether iteration `it` of primary `g` spawns a child.
@inline spawns(g::Int, it::Int) = (g + it) % 5 == 0
@inline function spawn_child!(buf, g, it)
    buf === nothing && return nothing
    spawn!(buf, Superphoton(SVector(0.0, g, it, 0.0), SVector(-1.0, 0.0, 0.0, 1.0), 1.0; nscatt = 1),
           first(child_stream(buf, primary_rng(SEED, g), it - 1)))
    return nothing
end
# The totals of a primary: its weight, its checksum (exact in Float64) and one count per path.
@inline function tally!(sc, g::Int, acc::UInt64)
    add!(sc, 1, g / 3.0)
    add!(sc, 2, Float64(acc % (UInt64(1) << 40)))
    return nothing
end

"The state of a toy primary: its index, the next loop iteration and the checksum so far."
struct Toy
    g::Int
    it::Int32
    acc::UInt64
end

# History programs: the whole life of primary `g`; the layer runs no generation (depth 0).
struct ToyPrimary{L,S,P,C} <: Function
    lives::L
    sc::S
    path::P
    buf::C
end
Adapt.@adapt_structure ToyPrimary
function (p::ToyPrimary)(g, i)
    acc = UInt64(0)
    for it in 1:life(p.lives, g)
        acc = fold(acc, g, it)
        spawns(g, it) && spawn_child!(p.buf, g, it)
    end
    tally!(p.sc, g, acc)
    add!(p.path, 1, 1.0)
    return nothing
end
struct ToyLayer{G} <: Function
    generation::G
end
Adapt.@adapt_structure ToyLayer
(p::ToyLayer)(item, i) = nothing

# Stage programs of the same transport.
struct ToyStart{P,Q} <: Function
    path::P
    queues::Q
end
Adapt.@adapt_structure ToyStart
function (p::ToyStart)(g, i)
    place!(p.queues, i, Toy(g, Int32(1), UInt64(0)))
    add!(p.path, 2, 1.0)
    return nothing
end

# `mode`: 0 the transport, 1 a state that never advances, 2 a state that reports a fault.
struct ToyAdvance{L,C,Q} <: Function
    lives::L
    buf::C
    mode::Int
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
        spawns(s.g, it) && spawn_child!(p.buf, s.g, it)
    end
    s2 = Toy(s.g, Int32(last + 1), acc)
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

Execution.wave_state(::Union{ToyStart,ToyAdvance,ToyRetire}) = Toy
Execution.wave_steps(::ToyAdvance) = LIFE

"The toy transport over `1:n`, spawning when `spawning`; `mode` of the advance."
function toy(n; spawning = true, mode = 0)
    sc = ExactScalars(2)
    path = ExactScalars(3)
    buf = spawning ? ChildBuffer(Superphoton, 0, MCStreamLayout(max(n, 1), 64, 1)) : nothing
    lives = copy(LIVES)
    wf = Wavefront(; start = ToyStart(path, WaveQueues()), advance = ToyAdvance(lives, buf, mode, WaveQueues()),
                   retire = ToyRetire(sc, path, WaveQueues()),
                   primary = ToyPrimary(lives, sc, path, buf),
                   layer = ToyLayer(Generation(Superphoton[], MCRng[])))
    return (wf = wf, sc = sc, path = path, buf = buf, lives = lives)
end

"Run the toy on `exec` with `depth`; the totals, the child count and the counts per path."
function run_toy(exec, n; depth = 0, kw...)
    t = toy(n; kw...)
    c = foreach_wavefront!(exec, t.wf, n, depth)
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
const DEFAULTS = (budget = 32, cohort = 2^18, refill = 2^14, doubling = true, wavefront = true)

@testset "device wavefront" begin
    ref = run_toy(ThreadedExecutor(chunk = 7), N)
    @test ref.children > 100 && ref.path == [N, 0, 0]
    expect = (totals = ref.totals, children = ref.children)
    same(r) = (totals = r.totals, children = r.children) == expect

    @testset "threads, host pools, other executors and phases without stages: the cascade" begin
        for exec in (ThreadedExecutor(), PoolExecutor(DK.hosts(1)), PoolExecutor(DK.hosts(2); chunk = 11))
            r = run_toy(exec, N)
            @test same(r) && r.path == [N, 0, 0]
        end
        ex = PoolExecutor(DK.wavehosts(2); chunk = 13)
        r = with_wavefront_settings(wavefront = false) do
            run_toy(ex, N)
        end
        @test same(r) && r.path == [N, 0, 0] && describe(ex).schedule == :cascade
        @test describe(ex).settings === nothing && sum(describe(ex).admitted) == N
        @test sum(describe(ex).spawned) == ref.children
        # any other executor: the cascade it forwards, with the caller's keywords only
        fx = Forwarding(ThreadedExecutor(), Any[])
        r = run_toy(fx, N)
        @test same(r) && r.path == [N, 0, 0] && fx.seen == [(capacity = nothing,)]
        t = toy(N)
        foreach_wavefront!(fx, t.wf, N, 0; capacity = 5, chunk = 9)
        @test fx.seen[end] == (capacity = 5, chunk = 9)
        # a scattering phase has no stages here: the cascade, which counts generation 1's children
        r1 = run_toy(ex, N; depth = 1)
        r1t = run_toy(ThreadedExecutor(), N; depth = 1)
        @test r1.path == [N, 0, 0] && describe(ex).schedule == :cascade
        @test (r1.totals, r1.children) == (r1t.totals, r1t.children)
    end

    @testset "every device count, budget, doubling, cohort and refill: the cascade's bits" begin
        for nd in DK.NDEV
            ex = PoolExecutor(DK.wavehosts(nd))
            for (K, dbl) in ((1, false), (3, true), (3, false), (64, 4), (64, true),
                             (typemax(Int), false)),
                (P, R) in ((1, 1), (7, 5), (2^10, 5), (2^10, 1))
                r = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R) do
                    run_toy(ex, N)
                end
                d = describe(ex)
                ok = same(r) && r.path == [0, N, N] && d.schedule == :wavefront &&
                     sum(d.admitted) == N && sum(d.spawned) == ref.children &&
                     d.settings == (budget = K, cohort = P, refill = R, doubling = dbl, wavefront = true)
                ok || @info "wavefront differs" nd K dbl P R r.path d
                @test ok
            end
        end
    end

    @testset "the devices of the comparisons ($(DK.DEVICE_KIND)): the cascade's bits" begin
        for nd in (1, 2)
            devs = DK.devices(nd)
            ex = PoolExecutor(devs)
            on = all(Execution.wavefront, devs)
            for (K, dbl, P, R) in ((1, false, 7, 5), (32, true, 2^10, 1), (typemax(Int), false, 64, 3))
                r = with_wavefront_settings(budget = K, doubling = dbl, cohort = P, refill = R) do
                    run_toy(ex, N)
                end
                @test same(r) && r.path == (on ? [0, N, N] : [N, 0, 0])
                @test describe(ex).schedule == (on ? :wavefront : :cascade)
            end
        end
    end

    @testset "budget doubling: on, capped, off" begin
        ex = PoolExecutor(DK.wavehosts(1))
        Ks(dbl) = with_wavefront_settings(budget = 1, doubling = dbl, cohort = 64, refill = 8) do
            run_toy(ex, N)
            [it.K for it in only(wave_log(ex)) if it.nA > 0]
        end
        on, capped, off = Ks(true), Ks(4), Ks(false)
        @test all(==(1), off) && maximum(capped) == 4 && maximum(on) > 4
        @test issorted(on) && issorted(capped)
        # doubling starts once no block remains and the cohort is below a quarter of P
        log = only(wave_log(ex))
        @test sum(it -> it.started, log) == N && sum(it -> it.done, log) == N
    end

    @testset "every device takes blocks at small settings" begin
        for nd in (2, 3)
            ex = PoolExecutor(DK.wavehosts(nd))
            r = with_wavefront_settings(cohort = 7, refill = 1) do
                run_toy(ex, N)
            end
            d = describe(ex)
            @test same(r) && all(>(0), d.admitted) && sum(d.admitted) == N
            @test all(>(0), d.iterations) && length(wave_log(ex)) == nd
        end
    end

    @testset "children of the last generation: counted and dropped; none without a slot" begin
        ex = PoolExecutor(DK.wavehosts(2))
        r = run_toy(ex, N)
        @test same(r) && r.slot == 0 && r.pending == 0
        r0 = run_toy(ex, N; spawning = false)
        @test r0.children == 0 && r0.totals == ref.totals
    end

    @testset "contract errors: a state that cannot advance, a fault" begin
        ex = PoolExecutor(DK.wavehosts(1))
        m = failure(() -> with_wavefront_settings(cohort = 4, refill = 2) do
            run_toy(ex, 40; mode = 1)
        end)
        @test m !== nothing && occursin("retired nothing in $(LIFE + 2) iterations", m)
        m = failure(() -> run_toy(ex, 40; mode = 2))
        @test m !== nothing && occursin("broke the advance", m)
    end

    @testset "twins and residency after a call" begin
        ex = PoolExecutor(DK.wavehosts(2))
        t = toy(N)
        n0 = [resident_count(ex, d) for d in DK.wavehosts(2)]
        for _ in 1:3
            with_wavefront_settings(cohort = 7, refill = 3) do
                foreach_wavefront!(ex, t.wf, N, 0)
            end
            @test length(t.buf.twins.list) == 0 && t.buf.count[1] == 0
            @test length(t.sc.twins.list) > 0
            gather!(t.sc); gather!(t.path)
            @test length(t.sc.twins.list) == 0 && length(t.path.twins.list) == 0
            @test [resident_count(ex, d) for d in DK.wavehosts(2)] == n0 .+ 1    # the life table
        end
        @test t.path.n == [0, 3N, 3N]
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
        for kw in ((budget = 0,), (cohort = 0,), (refill = -1,), (doubling = 0,))
            @test_throws ArgumentError with_wavefront_settings(() -> nothing; kw...)
        end
        t = toy(10)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.layer, primary = t.wf.primary,
                                             layer = t.wf.layer)
        @test_throws ArgumentError Wavefront(; start = t.wf.start, advance = t.wf.advance,
                                             retire = t.wf.retire, primary = t.wf.primary,
                                             layer = t.wf.layer, doubling_cap = 0)
        @test run_toy(ex, 0).children == 0 && describe(ex).admitted == [0]
        @test_throws ArgumentError foreach_wavefront!(ex, t.wf, -1, 0)
        @test DK.device_isbits(t.wf.advance) && DK.device_isbits(t.wf.start)
    end
end
