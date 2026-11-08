# The cascade: a primary program over chunks of items and a layer program over each
# generation's children, generation after generation within the chunk. The accumulated
# totals, the set of members of every generation (an exact checksum of their stream
# identities) and the returned child count agree with a serial breadth-first count and
# are bitwise the same under threads of any chunking and on pools of 1, 2, 3 and 5
# devices, with chunks that cut the items finely and child buffers that overflow; the
# children of the last generation are counted and not tracked; a program without a child
# buffer runs generation 0 only; lane buffers and scratch survive between calls. Child
# slots: an explicit capacity starts every generation of every chunk and reruns the
# overflowing ones, which `describe` counts; without one a generation starts with the slot
# rule, whose learned term is shared by the lanes of a device and whose memory budget is
# off on a device without free memory and capped at 16 slots per primary on one with it.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Numerics: MCRng, MCStreamLayout, child_rng, stream_id, primary_rng, philox_rand,
                    ME, CL, HPL, xsum_round
using BlackLightPlus.MonteCarlo: Superphoton
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, ExactBins, ExactScalars, ChildBuffer,
                     Generation, InclinationBands, PolarBands, foreach_cascade!,
                     cascade_chunk, gather!, add!, scalar, addends, spawn!, child_stream, children,
                     describe, DevicePool, HostDevice, AbstractDevice, free_bytes
import BlackLightPlus.Execution

const DK = Main.Decks
const SEED = 77
const MOD = 1_000_003

# Generation 0: item `g` records one photon, adds to the scalars and spawns `g % 4` children
# when it holds a child buffer.
struct Gen0{B,S,C} <: Function
    bins::B
    scalars::S
    buf::C
end
BlackLightPlus.Execution.Adapt.@adapt_structure Gen0
function (p::Gen0)(item, i)
    r = primary_rng(SEED, item)
    u = philox_rand(5, item, 0)
    E = exp(log(1.0e9) + 12.0 * u) * HPL / (ME * CL * CL)
    d = SVector(sin(3.0 * item), cos(5.0 * item), sin(item) * cos(7.0 * item))
    Execution.record!(p.bins, E, 1.0 + item / 7.0, d)
    add!(p.scalars, 1, exp(-item / 100.0))
    p.buf === nothing && return nothing
    for e in 0:(item % 4 - 1)
        spawn!(p.buf, Superphoton(SVector(0.0, item, e, 0.0), SVector(-E, 0.0, 0.0, E), 1.0 + e;
                                  nscatt = 1), first(child_stream(p.buf, r, e)))
    end
    return nothing
end

# Generation n ≥ 1: member `j` adds its stream identity (mod MOD) and weight to the scalars,
# records itself and spawns `id % 3` children.
struct GenN{G,B,S,C} <: Function
    generation::G
    bins::B
    scalars::S
    buf::C
end
BlackLightPlus.Execution.Adapt.@adapt_structure GenN
function (p::GenN)(item, i)
    sph = @inbounds p.generation.photons[item]
    rng = @inbounds p.generation.rngs[item]
    id = stream_id(rng)
    add!(p.scalars, 2, Float64(id % MOD))
    add!(p.scalars, 3, sph.w)
    Execution.record!(p.bins, -sph.p[1] * Int(sph.nscatt), sph.w, SVector(0.3, -0.2, sph.x[3] - 1.0))
    for e in 0:(Int(id % 3) - 1)
        spawn!(p.buf, Superphoton(sph.x, sph.p, sph.w / 2; nscatt = sph.nscatt + 1),
               first(child_stream(p.buf, rng, e)))
    end
    return nothing
end

# A fan: generation 0 spawns `p.k` children per item, generation n ≥ 1 one per member, so a
# chunk of m primaries spawns k m children in every generation.
struct Fan0{S,C} <: Function
    scalars::S
    buf::C
    k::Int
end
BlackLightPlus.Execution.Adapt.@adapt_structure Fan0
function (p::Fan0)(item, i)
    r = primary_rng(SEED, item)
    add!(p.scalars, 1, Float64(item))
    for e in 0:(p.k - 1)
        spawn!(p.buf, Superphoton(SVector(0.0, item, e, 0.0), SVector(-1.0, 0.0, 0.0, 1.0), 1.0 + e;
                                  nscatt = 1), first(child_stream(p.buf, r, e)))
    end
    return nothing
end

struct FanN{G,S,C} <: Function
    generation::G
    scalars::S
    buf::C
end
BlackLightPlus.Execution.Adapt.@adapt_structure FanN
function (p::FanN)(item, i)
    sph = @inbounds p.generation.photons[item]
    rng = @inbounds p.generation.rngs[item]
    add!(p.scalars, 2, Float64(stream_id(rng) % MOD))
    add!(p.scalars, 3, sph.w)
    spawn!(p.buf, Superphoton(sph.x, sph.p, sph.w / 2; nscatt = sph.nscatt + 1),
           first(child_stream(p.buf, rng, 0)))
    return nothing
end

function run_fan(exec, n, depth, k; kw...)
    sc = ExactScalars(3)
    slot = ChildBuffer(Superphoton, 0, MCStreamLayout(n, 16, depth + 1))
    c = foreach_cascade!(exec, Fan0(sc, slot, k), FanN(Generation(Superphoton[], MCRng[]), sc, slot),
                         n, depth; kw...)
    gather!(sc)
    return (count = c, acc = copy(sc.acc), n = copy(sc.n))
end

# A host device with `bytes` of free memory, so that the memory budget of the child slots
# runs without a GPU.
struct BudgetHost <: AbstractDevice
    id::Int
    bytes::Int
end
Execution.backend(::BudgetHost) = Execution.KernelAbstractions.CPU()
Execution.activate!(::BudgetHost) = nothing
Execution.functional(::BudgetHost) = true
Execution.free_bytes(d::BudgetHost) = d.bytes

fresh_acc() = (ExactBins(InclinationBands((30.0, 90.0, 100.0), 20.0); nu_min = 1.0e8,
                         nu_max = 1.0e24, n_ebins = 50), ExactScalars(3))

function run_cascade(exec, n, depth; spawn = true, kw...)
    bins, sc = fresh_acc()
    slot = spawn ? ChildBuffer(Superphoton, 0, MCStreamLayout(max(n, 1), 16, depth + 1)) : nothing
    p0 = Gen0(bins, sc, slot)
    pl = GenN(Generation(Superphoton[], MCRng[]), bins, sc, slot)
    k = foreach_cascade!(exec, p0, pl, n, depth; kw...)
    gather!(bins); gather!(sc)
    return (count = k, bins = bins, scalars = sc)
end

same(a, b) = a.count == b.count && a.bins.count == b.bins.count && a.bins.sumw == b.bins.sumw &&
             a.bins.sumwE == b.bins.sumwE && a.scalars.acc == b.scalars.acc && a.scalars.n == b.scalars.n

# Serial breadth-first count: the members of generations 1..depth, their checksum, and the
# children the last generation spawns.
function expected(n, depth)
    L = MCStreamLayout(n, 16, depth + 1)
    gen = [child_rng(primary_rng(SEED, g), e, L) for g in 1:n for e in 0:(g % 4 - 1)]
    members = 0; sum_id = 0; last_spawn = 0
    for d in 1:depth
        members += length(gen)
        sum_id += sum((Int(stream_id(r) % MOD) for r in gen); init = 0)
        nxt = [child_rng(r, e, L) for r in gen for e in 0:(Int(stream_id(r) % 3) - 1)]
        d == depth && (last_spawn = length(nxt))
        gen = nxt
    end
    return (members = members, sum_id = sum_id, last_spawn = last_spawn)
end

@testset "Execution: the cascade" begin
    n = 2_000
    depth = 3
    ex = expected(n, depth)
    ref = run_cascade(ThreadedExecutor(chunk = 1), n, depth)

    @testset "against the serial count" begin
        @test ex.members > 2_000 && ex.last_spawn > 0
        @test ref.count == ex.members + ex.last_spawn
        @test addends(ref.scalars, 2) == ex.members && scalar(ref.scalars, 2) == Float64(ex.sum_id)
        @test addends(ref.scalars, 1) == n && sum(ref.bins.count) > 0
    end

    @testset "bitwise across executors, chunks and capacities" begin
        execs = (ThreadedExecutor(chunk = 7), ThreadedExecutor(chunk = 4096),
                 (PoolExecutor(DK.hosts(nd); chunk = c) for (nd, c) in zip(DK.NDEV, (64, 97, 300, 4096)))...)
        for exec in execs
            @test same(run_cascade(exec, n, depth), ref)
        end
        # fine chunks, and child buffers that overflow (an item reruns on threads, a
        # generation on a pool)
        for exec in (ThreadedExecutor(chunk = 64), PoolExecutor(DK.hosts(2); chunk = 16),
                     PoolExecutor(DK.hosts(3); chunk = 50))
            @test same(run_cascade(exec, n, depth; chunk = 37), ref)
            @test same(run_cascade(exec, n, depth; chunk = 211, capacity = 1), ref)
            @test same(run_cascade(exec, n, depth; capacity = 2), ref)
        end
        # the same pool again (lane-resident buffers and scratch), and after other shapes
        pool = PoolExecutor(DK.hosts(2); chunk = 128)
        @test same(run_cascade(pool, n, depth; capacity = 5), ref)
        other = run_cascade(pool, 500, 1)
        @test other.count == expected(500, 1).members + expected(500, 1).last_spawn
        @test same(run_cascade(pool, n, depth; capacity = 5), ref)
    end

    @testset "child slots" begin
        T = Superphoton
        # an explicit capacity starts every generation: each overflow reruns, results unchanged
        for exec in (PoolExecutor(DK.hosts(2); chunk = 16), PoolExecutor(DK.hosts(3); chunk = 300),
                     PoolExecutor(DK.hosts(1)))
            for cap in (1, 2, 5)
                @test same(run_cascade(exec, n, depth; capacity = cap), ref)
                @test describe(exec).reruns > 0
            end
            run_cascade(exec, n, depth; capacity = 4n)
            @test describe(exec).reruns == 0
        end
        # verbatim for every generation of every chunk: 4 chunks × 3 spawning generations
        fan = run_fan(ThreadedExecutor(chunk = 64), 6_000, 3, 3)
        @test fan.count == 3 * 6_000 * (3 + 1)
        fx = PoolExecutor(DK.hosts(2); chunk = 1_500)
        @test run_fan(fx, 6_000, 3, 3; capacity = 1) == fan && describe(fx).reruns == 4 * 3
        @test run_fan(fx, 6_000, 3, 3; capacity = 3 * 1_500) == fan && describe(fx).reruns == 0
        @test run_fan(fx, 6_000, 3, 3; capacity = 3 * 1_500 - 1) == fan && describe(fx).reruns == 4 * 3
        @test run_cascade(fx, 0, depth).count == 0 && describe(fx).reruns == 0

        # no memory budget on a host device: the rule is the floor and the learned slots
        dev = HostDevice(1)
        @test free_bytes(dev) == 0
        p = DevicePool([dev]; lanes = 2)
        @test Execution.slot_budget(p, dev, T) == 0 && Execution.learned_slots(p, dev) == 0
        @test Execution.child_slots(p, dev, 6_000, T) == Execution.MIN_CHILD_SLOTS
        @test Execution.child_slots(p, dev, 10_001, T) == 5_001
        e1 = PoolExecutor(p; chunk = 6_000)
        @test run_fan(e1, 6_000, 3, 3) == fan
        @test describe(e1).reruns == 1              # generation 0 only: 18,000 > 4,096
        @test Execution.learned_slots(p, dev) == 18_000
        @test Execution.child_slots(p, dev, 6_000, T) == 18_000
        # the learned slots serve every lane of the device, in later calls and executors
        e2 = PoolExecutor(p; chunk = 6_000)
        f2 = run_fan(e2, 12_000, 3, 3)
        @test f2 == run_fan(ThreadedExecutor(chunk = 64), 12_000, 3, 3) && describe(e2).reruns == 0
        # an explicit capacity bypasses them
        @test run_fan(e2, 12_000, 3, 3; capacity = 1) == f2 && describe(e2).reruns == 2 * 3
        close(p)
        # learned per device: one chunk teaches the device that ran it
        p2 = DevicePool(DK.hosts(2); lanes = 1)
        run_fan(PoolExecutor(p2; chunk = 6_000), 6_000, 3, 3)
        @test sort([Execution.learned_slots(p2, d) for d in p2.devices]) == [0, 18_000]
        close(p2)

        # a memory budget: an eighth of the free memory over two sides and the lanes, at most
        # 16 slots per primary
        per = 8 * 2 * 2 * (sizeof(T) + sizeof(MCRng))
        bh = BudgetHost(1, 20_000 * per + per - 1)
        pb = DevicePool([bh]; lanes = 2)
        @test Execution.slot_budget(pb, bh, T) == 20_000
        @test Execution.child_slots(pb, bh, 6_000, T) == 20_000
        @test Execution.child_slots(pb, bh, 1_000, T) == 16_000
        @test Execution.child_slots(pb, bh, 100, T) == Execution.MIN_CHILD_SLOTS
        eb = PoolExecutor(pb; chunk = 6_000)
        @test run_fan(eb, 6_000, 3, 3) == fan
        @test describe(eb).reruns == 0 && Execution.learned_slots(pb, bh) == 18_000
        @test same(run_cascade(eb, n, depth), ref) && describe(eb).reruns == 0
        close(pb)

        # a GPU: free memory, a budget, and no rerun without a capacity
        if DK.DEVICE_KIND == "cuda"
            gdev = only(DK.devices(1))
            @test free_bytes(gdev) > 0
            pg = DevicePool([gdev]; lanes = 2)
            @test Execution.slot_budget(pg, gdev, T) >= 16 * 6_000
            @test Execution.child_slots(pg, gdev, 6_000, T) == 16 * 6_000
            eg = PoolExecutor(pg; chunk = 6_000)
            g0 = run_fan(eg, 12_000, 3, 3)
            @test describe(eg).reruns == 0
            @test run_fan(eg, 12_000, 3, 3; capacity = 1) == g0 && describe(eg).reruns == 2 * 3
            @test g0 == run_fan(ThreadedExecutor(chunk = 64), 12_000, 3, 3)
            close(pg)
        end
    end

    @testset "depth 0 and no child buffer" begin
        r0 = run_cascade(ThreadedExecutor(chunk = 64), n, 0)
        @test r0.count == sum(g % 4 for g in 1:n) && addends(r0.scalars, 2) == 0
        none = run_cascade(ThreadedExecutor(chunk = 64), n, depth; spawn = false)
        @test none.count == 0 && addends(none.scalars, 2) == 0
        @test none.scalars.acc[:, 1] == ref.scalars.acc[:, 1]
        for exec in (PoolExecutor(DK.hosts(1); chunk = 64), PoolExecutor(DK.hosts(3); chunk = 300))
            @test same(run_cascade(exec, n, 0), r0)
            @test same(run_cascade(exec, n, depth; spawn = false), none)
        end
        @test run_cascade(PoolExecutor(DK.hosts(2); chunk = 64), 0, depth).count == 0
    end

    @testset "generations" begin
        sp = Superphoton(SVector(0.0, 1.0, 0.0, 0.0), SVector(-1.0, 1.0, 0.0, 0.0), 2.0)
        g = Generation([sp, sp], [primary_rng(1, 1), primary_rng(1, 2)])
        @test length(g) == 2 && DK.device_isbits(g)
    end

    @testset "chunk sizes and argument checks" begin
        @test cascade_chunk(PoolExecutor(DK.hosts(1); chunk = 64), 10^7) == 2^20
        @test cascade_chunk(PoolExecutor(DK.hosts(2); chunk = 64), 4_000) == 500
        @test cascade_chunk(PoolExecutor(DK.hosts(2); chunk = 4096), 4_000) == 4096
        @test cascade_chunk(ThreadedExecutor(chunk = 4096), 10^7) ==
              max(4096, cld(10^7, 64 * Threads.nthreads()))
        # `:auto` leaves the rule; the run records the cascade's chunk
        @test cascade_chunk(PoolExecutor(DK.hosts(2)), 4_000) == 500
        @test cascade_chunk(PoolExecutor(DK.hosts(1)), 10^7) == 2^20
        @test cascade_chunk(ThreadedExecutor(), 10^7) == cld(10^7, 64 * Threads.nthreads())
        @test cascade_chunk(ThreadedExecutor(), 10) == 1
        for ex in (ThreadedExecutor(), PoolExecutor(DK.hosts(2)))
            @test describe(ex).chunk === :auto
            run_cascade(ex, 100, 1; chunk = 37)
            @test describe(ex).chunk == 37
            run_cascade(ex, 100, 1)
            @test describe(ex).chunk == cascade_chunk(ex, 100)
        end
        e = ThreadedExecutor(chunk = 64)
        p0 = Gen0(fresh_acc()..., nothing)
        @test_throws ArgumentError foreach_cascade!(e, p0, p0, -1, 1)
        @test_throws ArgumentError foreach_cascade!(e, p0, p0, 10, -1)
        @test_throws ArgumentError foreach_cascade!(e, p0, p0, 10, 1; chunk = 0)
        @test_throws ArgumentError foreach_cascade!(e, p0, p0, 10, 1; capacity = 0)
    end

    @testset "a layer with accumulators of its own" begin
        # the layer's scratch arrays differ in size from the primary's at the same positions,
        # so a lane reallocates them every generation: each allocation starts at zero
        function run_split(exec; kw...)
            bins, sc = fresh_acc()
            bins2 = ExactBins(InclinationBands((30.0, 90.0, 100.0), 20.0); nu_min = 1.0e8,
                              nu_max = 1.0e24, n_ebins = 30)
            sc2 = ExactScalars(5)
            slot = ChildBuffer(Superphoton, 0, MCStreamLayout(300, 16, 3))
            k = foreach_cascade!(exec, Gen0(bins, sc, slot),
                                 GenN(Generation(Superphoton[], MCRng[]), bins2, sc2, slot), 300, 2; kw...)
            foreach(gather!, (bins, sc, bins2, sc2))
            return (k, bins.count, bins.sumw, scalar(sc, 1), bins2.count, bins2.sumw,
                    scalar(sc2, 2), scalar(sc2, 3))
        end
        ref = run_split(ThreadedExecutor(chunk = 1))
        @test ref[1] > 0 && ref[8] > 0
        for exec in (ThreadedExecutor(chunk = 64), PoolExecutor(DK.hosts(2); chunk = 16),
                     PoolExecutor(DK.hosts(3)))
            @test run_split(exec; chunk = 37) == ref
            @test run_split(exec; chunk = 37, capacity = 1) == ref
            @test run_split(exec) == ref
        end
    end
end
