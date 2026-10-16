# Executor equivalence: one program run through the threaded executor and through device
# pools of 1, 2, 3 and 5 host devices (2 lanes, 7-item chunks) gives bitwise identical
# per-item outputs for any chunking and visiting order; programs without outputs run every
# item exactly once; several programs share one queue, part by part.
using Test, BLP
using StaticArrays, KernelAbstractions, Adapt, Random
using Atomix: Atomix
using BLP.Execution: ThreadedExecutor, PoolExecutor, HostDevice, HostRays, ray,
                     foreach_item!, workgroupsize, describe, mixed_partition
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
               [PoolExecutor(Decks.hosts(nd); lanes = 2, chunk = 7) for nd in Decks.NDEV]]

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
    @test_throws ArgumentError ThreadedExecutor(chunk = 0)
    @test describe(exec).chunk == 7 && describe(ThreadedExecutor(chunk = 3)).chunk == 3
end

end
