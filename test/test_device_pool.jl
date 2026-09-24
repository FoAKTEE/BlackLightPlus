# DevicePool runtime (src/gpu_pool.jl; results/blp/multigpu/dag.md, MG07): the chunk
# queue runs every item exactly once in queue order across persistent per-device workers,
# rethrows worker exceptions and stays usable, streams items added while running,
# refuses nested jobs, caches resident uploads per host object, reuses one pool per
# device list, and shuts its workers down. KA CPU backend (HostDevice), no GPU needed.
using Test, KernelAbstractions
using BlackLightPlus
@isdefined(DevicePool) || include(joinpath(@__DIR__, "..", "src", "gpu_pool.jl"))

hosts(nd) = [HostDevice(k) for k in 1:nd]

@testset "DevicePool runtime (KA CPU backend)" begin

@testset "construction and registry" begin
    @test_throws ErrorException DevicePool(HostDevice[])
    @test_throws ErrorException DevicePool([HostDevice(1), HostDevice(1)])
    @test_throws ErrorException DevicePool(hosts(1); chunk = 0)
    p = DevicePool(hosts(3); lanes = 2, chunk = 5)
    @test length(p.lanes) == 6 && [l.dev for l in p.lanes[1:2:end]] == hosts(3)
    @test all(l -> l.backend isa CPU, p.lanes) && device_list(p) == p.devices
    close(p)
    q = device_pool(hosts(2))
    @test device_pool(hosts(2)) === q && device_pool(q) === q     # one pool per device list
    @test device_pool(hosts(3)) !== q
end

@testset "queue: each item once, in order, on every lane" begin
    p = DevicePool(hosts(3); lanes = 2)
    seen = Channel{Tuple{Int,Any}}(Inf)
    foreach_chunk(p, 1:500) do lane, i
        rand() < 0.1 && sleep(0.001)                     # uneven chunk durations
        put!(seen, (i, lane.dev))
    end
    close(seen)
    got = collect(seen)
    @test sort(first.(got)) == 1:500                     # every item exactly once
    @test length(unique(last.(got))) == 3                # all devices pulled work
    # one lane: the queue order is the execution order
    p1 = DevicePool(hosts(1); lanes = 1)
    order = Int[]
    foreach_chunk((_, i) -> push!(order, i), p1, [5, 3, 9, 1])
    @test order == [5, 3, 9, 1]
    foreach_chunk((_, _) -> error("never runs"), p1, Int[])  # empty queue: no-op
    close(p); close(p1)
end

@testset "errors: rethrown with the worker's exception, pool reusable" begin
    p = DevicePool(hosts(2); lanes = 2)
    ran = Threads.Atomic{Int}(0)
    err = try
        foreach_chunk(p, 1:200) do _, i
            Threads.atomic_add!(ran, 1)
            i == 7 && throw(DomainError(i, "chunk 7 failed"))
            sleep(0.0005)
        end
        nothing
    catch e
        e
    end
    @test err isa CapturedException && err.ex isa DomainError && err.ex.val == 7
    @test ran[] < 200                                     # remaining items drained, not run
    s = Threads.Atomic{Int}(0)
    foreach_chunk((_, i) -> Threads.atomic_add!(s, i), p, 1:100)
    @test s[] == 5050                                     # the pool still works
    # a job may not be started from inside a worker (it would wait on itself)
    nested = try
        foreach_chunk((_, _) -> foreach_chunk((_, _) -> nothing, p, 1:2), p, 1:1); nothing
    catch e
        e
    end
    @test nested isa CapturedException && occursin("inside a pool worker", sprint(showerror, nested.ex))
    close(p)
end

@testset "stream_chunks: items enqueued while running" begin
    p = DevicePool(hosts(3); lanes = 2)
    visited = Channel{Int}(Inf)
    # a binary tree of depth 6 rooted at 1: node k spawns 2k, 2k+1 until k >= 64
    stream_chunks(p, [1]) do _, k, job
        put!(visited, k)
        k < 64 && (enqueue!(job, 2k); enqueue!(job, 2k + 1))
    end
    close(visited)
    @test sort(collect(visited)) == 1:127
    close(p)
end

@testset "foreach_device and uploads" begin
    p = DevicePool(hosts(4); lanes = 3)
    ids = foreach_device(lane -> (lane.dev.id, Threads.threadid()), p)
    @test first.(ids) == 1:4
    x = rand(Float32, 3, 4, 5)
    d = upload_per_device(x, p)
    @test sort(collect(keys(d)), by = h -> h.id) == hosts(4)
    @test all(v -> v == x && v !== x, values(d))           # a device copy per device
    @test isempty(upload_per_device(x, HostDevice[]))       # no devices: nothing uploaded
    close(p)
end

@testset "residency: built once per (host object, tag)" begin
    p = DevicePool(hosts(2))
    builds = Ref(0)
    a = zeros(4); b = zeros(4)
    mk() = (builds[] += 1; builds[])
    @test resident(mk, p, a, :grid) == 1
    @test resident(mk, p, a, :grid) == 1                  # cached
    @test resident(mk, p, a, :amr) == 2                   # another tag of the same host
    @test resident(mk, p, b, :grid) == 3                  # an equal but distinct host array
    close(p)
    @test isempty(p.resident)                             # dropped at close
end

@testset "shutdown" begin
    p = DevicePool(hosts(2); lanes = 2)
    foreach_chunk((_, _) -> nothing, p, 1:10)
    close(p)
    @test !isopen(p) && all(l -> istaskdone(l.task), p.lanes)
    @test_throws ErrorException foreach_chunk((_, _) -> nothing, p, 1:1)
    close(p)                                              # idempotent
    q = device_pool(hosts(5))
    close(q)
    @test device_pool(hosts(5)) !== q && isopen(device_pool(hosts(5)))   # reopened on demand
end

end
