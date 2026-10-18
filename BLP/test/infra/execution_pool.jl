# The device pool on host devices: the queue runs every item exactly once in queue order
# across persistent lanes, rethrows a worker's exception and stays usable, streams items
# added while running, refuses nested jobs, reuses one registered pool per device list,
# whatever the chunk, shuts its workers down, and releases replaced and dropped staging
# buffers.
using Test, BLP
using KernelAbstractions
using BLP.Execution: DevicePool, PoolExecutor, HostDevice, CUDADevice, resident, describe
const Ex = BLP.Execution
const Decks = Main.Decks

@testset "device pool" begin

@testset "construction and registry" begin
    @test_throws ArgumentError DevicePool(HostDevice[])
    @test_throws ArgumentError DevicePool([HostDevice(1), HostDevice(1)])
    @test_throws ArgumentError DevicePool(Decks.hosts(1); chunk = 0)
    @test_throws ArgumentError DevicePool(Decks.hosts(1); lanes = 0)
    p = DevicePool(Decks.hosts(3); lanes = 2, chunk = 5)
    @test length(p.lanes) == 6 && [l.dev for l in p.lanes[1:2:end]] == Decks.hosts(3)
    @test all(l -> l.backend isa CPU, p.lanes) && Ex.lanes_per_device(p) == 2
    @test occursin("lanes = 2", sprint(show, p))
    close(p)
    @test occursin("closed", sprint(show, p))
    q = Ex.device_pool(Decks.hosts(2))
    @test Ex.device_pool(Decks.hosts(2)) === q && Ex.device_pool(q) === q
    @test Ex.device_pool(Decks.hosts(3)) !== q
    @test Ex.device_pool(Decks.hosts(2); lanes = 3) !== q
    @test PoolExecutor(Decks.hosts(2)).pool === q
    @test PoolExecutor(Decks.hosts(2); chunk = 9).pool === q    # any chunk shares the pool
    @test PoolExecutor(q).pool === q && q.chunk === :auto
    @test PoolExecutor(q).chunk === :auto && PoolExecutor(q; chunk = 9).chunk == 9
    @test PoolExecutor(DevicePool(Decks.hosts(1); chunk = 5)).chunk == 5
    @test_throws ArgumentError DevicePool(Decks.hosts(1); chunk = :fast)
end

@testset "queue: each item once, in order, on every lane" begin
    p = DevicePool(Decks.hosts(3); lanes = 2)
    seen = Channel{Tuple{Int,Any}}(Inf)
    Ex.foreach_chunk(p, 1:500) do lane, i
        i % 10 == 0 && sleep(0.001)                       # uneven item durations
        put!(seen, (i, lane.dev))
    end
    close(seen)
    got = collect(seen)
    @test sort(first.(got)) == 1:500
    @test length(unique(last.(got))) == 3
    p1 = DevicePool(Decks.hosts(1); lanes = 1)
    order = Int[]
    Ex.foreach_chunk((_, i) -> push!(order, i), p1, [5, 3, 9, 1])
    @test order == [5, 3, 9, 1]                           # one lane runs in queue order
    Ex.foreach_chunk((_, _) -> error("never runs"), p1, Int[])
    close(p); close(p1)
end

@testset "errors: rethrown with the worker's exception, pool reusable" begin
    p = DevicePool(Decks.hosts(2); lanes = 2)
    ran = Threads.Atomic{Int}(0)
    err = try
        Ex.foreach_chunk(p, 1:200) do _, i
            Threads.atomic_add!(ran, 1)
            i == 7 && throw(DomainError(i, "item 7 failed"))
            sleep(0.0005)
        end
        nothing
    catch e
        e
    end
    @test err isa CapturedException && err.ex isa DomainError && err.ex.val == 7
    @test ran[] < 200                                     # the rest drained, not run
    s = Threads.Atomic{Int}(0)
    Ex.foreach_chunk((_, i) -> Threads.atomic_add!(s, i), p, 1:100)
    @test s[] == 5050
    nested = try
        Ex.foreach_chunk((_, _) -> Ex.foreach_chunk((_, _) -> nothing, p, 1:2), p, 1:1)
        nothing
    catch e
        e
    end
    @test nested isa CapturedException &&
          occursin("inside a pool worker", sprint(showerror, nested.ex))
    close(p)
end

@testset "stream_chunks: items enqueued while running" begin
    p = DevicePool(Decks.hosts(3); lanes = 2)
    visited = Channel{Int}(Inf)
    Ex.stream_chunks(p, [1]) do _, k, job              # binary tree of depth 6 rooted at 1
        put!(visited, k)
        k < 64 && (Ex.enqueue!(job, 2k); Ex.enqueue!(job, 2k + 1))
    end
    close(visited)
    @test sort(collect(visited)) == 1:127
    close(p)
end

@testset "foreach_device and per-device copies" begin
    p = DevicePool(Decks.hosts(4); lanes = 3)
    ids = Ex.foreach_device(lane -> (lane.dev.id, Threads.threadid()), p)
    @test first.(ids) == 1:4
    x = rand(Float32, 3, 4, 5)
    twins = [resident(p, d, x) for d in p.devices]
    @test all(t -> t == x && t !== x, twins)
    @test allunique(objectid.(twins))
    close(p)
    @test all(c -> isempty(c.twins), p.resident)          # dropped at close
end

@testset "shutdown" begin
    p = DevicePool(Decks.hosts(2); lanes = 2)
    Ex.foreach_chunk((_, _) -> nothing, p, 1:10)
    close(p)
    @test !isopen(p) && all(l -> istaskdone(l.task), p.lanes)
    @test_throws ErrorException Ex.foreach_chunk((_, _) -> nothing, p, 1:1)
    close(p)                                              # idempotent
    q = Ex.device_pool(Decks.hosts(5))
    close(q)
    r = Ex.device_pool(Decks.hosts(5))
    @test r !== q && isopen(r)                            # reopened on demand
end

@testset "host staging: replaced and dropped buffers are released" begin
    for trial in 1:3
        pool = DevicePool(Decks.hosts(2); lanes = 8, chunk = 1)
        Ex.foreach_chunk(pool, 1:64) do lane, i
            for n in (1 << 10, 1 << 14, 1 << 17) .+ i     # grow: drop the smaller buffer
                st = Ex.lane_stage(lane, :stress, Float64, n)
                d = Ex.lane_buffer(lane, :stress, Float64, n)
                st[1:n] .= i
                copyto!(d, 1, st, 1, n)
                copyto!(st, 1, d, 1, n)
                @assert st[n] == i
                i % 4 == 0 && GC.gc(false)
            end
        end
        close(pool)
        GC.gc()
        @test all(lane -> isempty(lane.stage) && isempty(lane.bufs), pool.lanes)
    end
    pool = DevicePool(Decks.hosts(2); lanes = 2, chunk = 4)
    sizes = Channel{Int}(Inf)
    Ex.foreach_chunk(pool, 1:8) do lane, i
        Ex.lane_stage(lane, :a, Float64, 100)
        put!(sizes, length(Ex.lane_stage(lane, :a, Float64, 10_000)))   # replaces the 100
        a = Ex.lane_array(lane, :log, Float64, (3, 4))
        put!(sizes, length(a))
    end
    close(sizes)
    @test all(s -> s in (10_000, 12), collect(sizes))
    close(pool)
    @test all(lane -> isempty(lane.stage), pool.lanes)
end

end
