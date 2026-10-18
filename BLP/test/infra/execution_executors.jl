# Executor equivalence: one program run through the threaded executor and through device
# pools of 1, 2, 3 and 5 host devices (2 lanes, 7-item chunks) gives bitwise identical
# per-item outputs for any chunking and visiting order; programs without outputs run every
# item exactly once, seen as `Int` whatever the item list; several programs share one
# queue, part by part.
using Test, BLP
using StaticArrays, KernelAbstractions, Adapt, Random
using Atomix: Atomix
using BLP.Execution: ThreadedExecutor, PoolExecutor, HostDevice, HostRays, ray,
                     foreach_item!, workgroupsize, chunk_hint, describe, mixed_partition
const Ex = BLP.Execution
const Decks = Main.Decks

# A nonlinear oscillator pair, integrated by fixed-step RK4.
@inline rhs(y::SVector{4,Float64}, c) =
    SVector(y[2], -c * sin(y[1]) - 0.1 * y[2], y[4], -y[3] * (1.0 + 0.5 * y[1] * y[1]))

struct ODEProgram{R,P}
    rays::R
    coef::P
end
Adapt.@adapt_structure ODEProgram

# Per item: an item-dependent number of RK4 steps from the item's ray; two outputs.
@inline function (p::ODEProgram)(item, i, o64, o32)
    y8, ok = ray(p.rays, item)
    y = y8[SVector(1, 2, 3, 4)]
    c = @inbounds p.coef[item]
    h = 0.01
    for _ in 1:(10 + item % 37)
        k1 = rhs(y, c)
        k2 = rhs(y + (h / 2) * k1, c)
        k3 = rhs(y + (h / 2) * k2, c)
        k4 = rhs(y + h * k3, c)
        y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    @inbounds o64[i] = ok ? y : zero(y)
    @inbounds o32[i] = SVector{4,Float32}(y * y8[5])
    return nothing
end

# The same program with a fixed workgroup size.
struct WGProgram{P}
    inner::P
end
Adapt.@adapt_structure WGProgram
BLP.Execution.workgroupsize(::WGProgram) = 4
@inline (p::WGProgram)(item, i, o64, o32) = p.inner(item, i, o64, o32)

# The same program with a chunk hint.
struct HintProgram{P}
    inner::P
end
Adapt.@adapt_structure HintProgram
BLP.Execution.chunk_hint(::HintProgram) = 10
@inline (p::HintProgram)(item, i, o64, o32) = p.inner(item, i, o64, o32)

# Counts items and sums their values through atomics on a host array, which it shares
# with every host device instead of taking a device copy.
struct CountProgram{A}
    acc::A
end
Adapt.adapt_structure(to, p::CountProgram) = p
@inline function (p::CountProgram)(item, i)
    Atomix.@atomic p.acc[1] += 1
    Atomix.@atomic p.acc[2] += item
    return nothing
end

# Counts the items it sees as `Int` and sums their offsets from `base`.
struct IntItemProgram{A}
    acc::A
    base::Int
end
Adapt.adapt_structure(to, p::IntItemProgram) = p
@inline function (p::IntItemProgram)(item, i)
    Atomix.@atomic p.acc[1] += (item isa Int)
    Atomix.@atomic p.acc[2] += item - p.base
    return nothing
end

# Writes `tag + item` and the order in which items started (an atomic ticket).
struct TicketProgram{A}
    tag::Float64
    ticket::A
end
Adapt.adapt_structure(to, p::TicketProgram) = p
@inline function (p::TicketProgram)(item, i, val, seq)
    @inbounds val[i] = p.tag + item
    @inbounds seq[i] = Atomix.@atomic p.ticket[1] += 1
    return nothing
end

const N = 103
const RNG = Xoshiro(20261016)
const RAYS = [SVector{8,Float64}(randn(RNG, 8)) for _ in 1:N]
const COEF = 0.5 .+ rand(RNG, N)
const PROG = ODEProgram(HostRays(RAYS), COEF)

function serial_reference(prog, items)
    o64 = fill(SVector{4,Float64}(NaN, NaN, NaN, NaN), N)
    o32 = fill(SVector{4,Float32}(NaN, NaN, NaN, NaN), N)
    foreach(item -> prog(item, item, o64, o32), items)
    return (o64, o32)
end

function run_outputs(exec, prog, items; order = nothing)
    o64 = fill(SVector{4,Float64}(NaN, NaN, NaN, NaN), N)
    o32 = fill(SVector{4,Float32}(NaN, NaN, NaN, NaN), N)
    foreach_item!(exec, prog, items; outs = (o64, o32), order)
    return (o64, o32)
end

executors() = [ThreadedExecutor(chunk = 3); ThreadedExecutor();
               [PoolExecutor(Decks.hosts(nd); lanes = 2, chunk = 7) for nd in Decks.NDEV];
               PoolExecutor(Decks.hosts(1)); PoolExecutor(Decks.hosts(3); lanes = 1)]

@testset "executors" begin

@testset "bitwise per item for every executor, chunking and order" begin
    queue = randperm(Xoshiro(3), N)
    ref = Decks.bits(serial_reference(PROG, 1:N))
    @test all(isfinite, ref)
    orders = (nothing, randperm(Xoshiro(5), N), collect(N:-1:1))
    for exec in executors(), items in (1:N, queue), order in orders
        @test Decks.bits(run_outputs(exec, PROG, items; order)) == ref
    end
    pool = PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7)
    @test Decks.bits(run_outputs(pool, WGProgram(PROG), 1:N)) == ref     # workgroup size 4
    @test workgroupsize(PROG) === nothing && workgroupsize(WGProgram(PROG)) == 4
end

@testset "automatic chunk: the rule, the hint, an explicit chunk, the record" begin
    t = ThreadedExecutor()
    nt = Threads.nthreads()
    @test t.chunk === :auto && describe(t).chunk === :auto && t == ThreadedExecutor(chunk = :auto)
    @test t != ThreadedExecutor(chunk = 64) && ThreadedExecutor(chunk = 64) == ThreadedExecutor(64)
    @test Ex.chunk_for(t, 10^8) == 4096 && Ex.chunk_for(t, 48 * 16 * nt) == 48
    @test Ex.chunk_for(t, 10) == 16 && Ex.chunk_for(t, 10^5; hint = 7) == 7
    @test Ex.chunk_for(t, 10^8; bytes = 10^6) == 4096                 # outputs: pools only
    p1 = PoolExecutor(Decks.hosts(1))
    @test Ex.chunk_for(p1, 10^8) == 2^20 && Ex.chunk_for(p1, 1000) == 2^20
    @test Ex.chunk_for(p1, 10^8; hint = 1000) == 1000
    @test Ex.chunk_for(p1, 10^8; bytes = 8) == 2^20                   # 16 MiB of outputs
    @test Ex.chunk_for(p1, 10^8; bytes = 120) == 2^24 ÷ 120
    @test Ex.chunk_for(p1, 10^8; bytes = 1600) == 2^16
    p3 = PoolExecutor(Decks.hosts(3); lanes = 2)
    @test Ex.chunk_for(p3, 10^7) == cld(10^7, 12) && Ex.chunk_for(p3, 10^8) == 2^20
    @test Ex.chunk_for(p3, 300_000) == 2^16 && Ex.chunk_for(p3, 1000) == 2^16
    @test Ex.chunk_for(PoolExecutor(Decks.hosts(3); lanes = 1), 10^6) == cld(10^6, 6)
    for e in (PoolExecutor(Decks.hosts(1); chunk = 7), ThreadedExecutor(chunk = 7))
        @test Ex.chunk_for(e, 10^8; hint = 3, bytes = 10^6) == 7     # explicit wins
    end
    ref = Decks.bits(serial_reference(PROG, 1:N))
    bytes = sizeof(SVector{4,Float64}) + sizeof(SVector{4,Float32})
    for e in (ThreadedExecutor(), p1, p3)
        @test Decks.bits(run_outputs(e, PROG, 1:N)) == ref
        @test describe(e).chunk == Ex.chunk_for(e, N; bytes)            # the chunk used
        @test Decks.bits(run_outputs(e, HintProgram(PROG), collect(1:N))) == ref
        @test describe(e).chunk == 10
        @test e.chunk === :auto
    end
    e = PoolExecutor(Decks.hosts(2); chunk = 9)
    run_outputs(e, PROG, 1:N)
    @test describe(e).chunk == 9 && chunk_hint(PROG) === nothing
end

@testset "a subset of items leaves the other slots untouched" begin
    sub = collect(2:3:N)
    for exec in executors()[[1, 3, 5]]
        o64, o32 = run_outputs(exec, PROG, sub)
        @test all(k -> isnan(o64[k][1]) == !(k in sub), 1:N)
    end
end

@testset "programs without outputs: every item exactly once" begin
    for exec in executors()
        acc = zeros(Int, 2)
        @test foreach_item!(exec, CountProgram(acc), 1:N) == ()
        @test acc == [N, N * (N + 1) ÷ 2]
    end
end

@testset "item lists: ranges as they are, vectors staged as Int32 or Int" begin
    pool = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    kinds = Channel{Any}(Inf)
    Ex.foreach_chunk(pool.pool, 1:1) do lane, _
        put!(kinds, Ex.device_items(lane, 3:9))
        put!(kinds, eltype(Ex.device_items(lane, view([5, 1, 9], 1:3))))
        put!(kinds, eltype(Ex.device_items(lane, [1, 2^40])))
    end
    @test take!(kinds) === 3:9 && take!(kinds) === Int32 && take!(kinds) === Int
    wide = 2^40
    for exec in (pool, PoolExecutor(Decks.hosts(1)), ThreadedExecutor()),
        (items, base) in ((1:N, 0), (Base.OneTo(N), 0), (collect(N:-1:1), 0),
                          (collect(wide .+ (1:N)), wide), (wide .+ (1:N), wide))
        acc = zeros(Int, 2)
        foreach_item!(exec, IntItemProgram(acc, base), items)
        @test acc == [N, N * (N + 1) ÷ 2]
    end
    o = zeros(N)
    foreach_item!(pool, (item, i, o) -> (@inbounds o[i] = 2item; nothing), 4:N; outs = (o,))
    @test o[4:N] == 2 .* (4:N) && all(iszero, o[1:3])            # contiguous gather
end

@testset "several programs in one queue, part by part" begin
    queue = randperm(Xoshiro(7), N)
    parts = mixed_partition(queue, sort(queue[1:17]))
    pool = PoolExecutor([HostDevice(1)]; lanes = 1, chunk = 5)
    pool3 = PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7)
    for exec in (pool, ThreadedExecutor(chunk = 4), pool3)
        ticket = zeros(Int, 1)
        val = zeros(N); seq = zeros(Int, N)
        foreach_item!(exec, (TicketProgram(1000.0, ticket) => parts.rescue,
                             TicketProgram(2000.0, ticket) => parts.bulk);
                      outs = (val, seq))
        @test val[parts.rescue] == 1000.0 .+ parts.rescue
        @test val[parts.bulk] == 2000.0 .+ parts.bulk
        @test sort(seq) == 1:N
        # one lane: every chunk of the first part starts before any chunk of the second
        exec === pool && @test maximum(seq[parts.rescue]) < minimum(seq[parts.bulk])
    end
end

@testset "programs defined after the pool started" begin
    exec = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    foreach_item!(exec, CountProgram(zeros(Int, 2)), 1:3)
    @eval struct LateProgram end
    @eval (::LateProgram)(item, i, o) = (@inbounds o[i] = 2item; nothing)
    o = zeros(Int, N)
    foreach_item!(exec, Base.invokelatest(LateProgram), 1:N; outs = (o,))
    @test o == 2 .* (1:N)
end

@testset "argument checks and empty work" begin
    exec = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    o = zeros(N)
    @test foreach_item!(exec, PROG, Int[]; outs = (o,)) === (o,)
    @test foreach_item!(ThreadedExecutor(), PROG, Int[]; outs = (o,)) === (o,)
    @test all(iszero, o)
    for e in (exec, ThreadedExecutor())
        @test_throws ArgumentError foreach_item!(e, PROG, 1:N; order = [1, 1, 2])
        @test_throws ArgumentError foreach_item!(e, PROG, 1:N; order = collect(1:N - 1))
    end
    failing(item, i, o) = (item == 50 && error("item 50 failed"); o[i] = item; nothing)
    for e in (ThreadedExecutor(chunk = 4), ThreadedExecutor())
        err = try
            foreach_item!(e, failing, 1:N; outs = (zeros(N),)); nothing
        catch x
            x
        end
        @test err isa CapturedException && occursin("item 50", sprint(showerror, err.ex))
        @test foreach_item!(e, CountProgram(zeros(Int, 2)), 1:N) == ()   # still usable
    end
    @test_throws ArgumentError ThreadedExecutor(chunk = 0)
    @test describe(exec).chunk == 7 && describe(ThreadedExecutor(chunk = 3)).chunk == 3
end

end
