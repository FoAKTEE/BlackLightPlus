# Residency: a host object's device twin is cached per pool and device by identity and
# held weakly; program twins read their arrays through the read-only wrapper, which a
# struct field carries on the KernelAbstractions CPU backend; ray sources and programs
# built from them are `isbits` once adapted to a device.
using Test, BLP
using StaticArrays, KernelAbstractions, Adapt
using BLP.Execution: PoolExecutor, HostDevice, resident, device_program, readonly, upload,
                     HostRays, CameraRays, PixelGrid, BlockGrid, CoordList, foreach_item!
const Ex = BLP.Execution
const Decks = Main.Decks

struct Holder{A,B}
    a::A
    b::B
    scale::Float64
end
Adapt.@adapt_structure Holder

struct Opaque{A}
    a::A
end

# Reads a 3-d array and a vector through nested struct fields.
struct NestedRead{H}
    h::H
end
Adapt.@adapt_structure NestedRead
@inline function (p::NestedRead)(item, i, out)
    @inbounds out[i] =
        p.h.scale * p.h.a[1 + item % 2, 1 + item % 3, 1 + item % 4] + p.h.b[item]
    return nothing
end

struct DummySpacetime
    a::Float64
end
struct DummyFrame
    sin_i::Float64
    cos_i::Float64
    d::Float64
end

@noinline upload_temporary(exec, dev) = (resident(exec, dev, rand(1000)); nothing)

@testset "residency" begin

@testset "twins: one per host array and device" begin
    exec = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    a = rand(10)
    t1 = resident(exec, HostDevice(1), a)
    @test t1 == a && t1 !== a
    @test resident(exec, HostDevice(1), a) === t1
    @test resident(exec, HostDevice(1), copy(a)) !== t1          # equal content, new object
    t2 = resident(exec, HostDevice(2), a)
    @test t2 == a && t2 !== t1
    h = resident(exec, HostDevice(2), Holder(a, [1, 2, 3], 2.0))
    @test h.a === t2 && h.b == [1, 2, 3] && h.scale == 2.0
    @test resident(exec, HostDevice(2), 3.5) === 3.5
    @test_throws ArgumentError resident(exec, HostDevice(7), a)
    @test_throws ArgumentError upload(HostDevice(1), [[1.0], [2.0]])
end

@testset "weak: a collected host array leaves the cache" begin
    exec = PoolExecutor(Decks.hosts(2); lanes = 1, chunk = 11)
    dev = HostDevice(2)
    keep = rand(5)
    resident(exec, dev, keep)
    n0 = Ex.resident_count(exec, dev)
    upload_temporary(exec, dev)
    @test Ex.resident_count(exec, dev) == n0 + 1
    for _ in 1:3
        GC.gc(true)
    end
    @test Ex.resident_count(exec, dev) == n0
    @test resident(exec, dev, keep) == keep                       # live entries stay
end

@testset "program twins read through the read-only wrapper" begin
    @test readonly(HostDevice(1), [1.0]) isa Base.Experimental.Const
    exec = PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7)
    a = rand(2, 3, 4); b = rand(40)
    prog = NestedRead(Holder(a, b, 0.5))
    dp = device_program(exec, HostDevice(2), prog)
    @test dp.h.a isa Base.Experimental.Const && dp.h.b isa Base.Experimental.Const
    @test dp.h.a == a && dp.h.scale == 0.5
    # a Const array inside a struct field compiles and runs in the CPU kernel
    ref = zeros(40); foreach(k -> prog(k, k, ref), 1:40)
    out = zeros(40)
    foreach_item!(exec, prog, 1:40; outs = (out,))
    @test out == ref
end

@testset "device-likeness: isbits after adaptation" begin
    rays = [SVector{8,Float64}(randn(8)) for _ in 1:5]
    @test Decks.device_isbits(HostRays(rays))
    st = DummySpacetime(0.9); fr = DummyFrame(0.8, 0.6, 1000.0)
    @test isbits(st) && isbits(fr)
    @test Decks.device_isbits(CameraRays(st, fr, PixelGrid(24.0, 10)))
    @test Decks.device_isbits(CameraRays(st, fr, BlockGrid([SVector(0.0, 0.0, 0.5)], 4)))
    @test Decks.device_isbits(CameraRays(st, fr, BlockGrid(4)))
    @test Decks.device_isbits(CameraRays(st, fr, CoordList([1.0, 2.0], [0.0, 3.0])))
    @test Decks.device_isbits(Holder(rand(2, 2), rand(Float32, 3), 1.0))
    @test Decks.device_isbits(NestedRead(Holder(rand(2, 2), [1, 2], 1.0)))
    @test !Decks.device_isbits(Opaque(rand(3)))                   # no Adapt structure
    adapted = Adapt.adapt(Decks.DeviceLike(), HostRays(rays))
    @test adapted.rays[3] == rays[3]
end

end
