# The CPU/GPU switch: precedence keyword > session default > BLP_BACKEND > :auto, the
# environment grammar and its errors, strict explicit choices, the executor description
# and printing, and the one-time log of the :auto choice.
using Test, BlackLightPlus
using BlackLightPlus.Execution: resolve, backend, backend!, describe, ThreadedExecutor, PoolExecutor,
                     DevicePool, HostDevice, CUDADevice, BackendUnavailable, cuda_loaded
const Ex = BlackLightPlus.Execution
const Decks = Main.Decks

hostpool(e, nd) = e isa PoolExecutor && e.pool.devices == Decks.hosts(nd)
message(f) = try
    f(); ""
catch e
    sprint(showerror, e)
end

@testset "backend switch" begin

@testset "values accepted by resolve" begin
    t = ThreadedExecutor(chunk = 5)
    @test resolve(t) === t
    @test resolve(:cpu) == ThreadedExecutor()
    @test hostpool(resolve(:host_pool), 1)
    @test hostpool(resolve([HostDevice(1)]), 1)
    @test hostpool(resolve(HostDevice(1)), 1)
    p = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    @test resolve(p) === p
    @test resolve(p.pool).pool === p.pool
    @test_throws ArgumentError resolve(:tpu)
    @test_throws ArgumentError resolve(HostDevice[])
end

@testset "BLP_BACKEND grammar" begin
    @test resolve("cpu") == ThreadedExecutor()
    @test resolve(" CPU ") == ThreadedExecutor()
    @test hostpool(resolve("host_pool"), 1)
    @test hostpool(resolve("host_pool:2"), 2)
    @test hostpool(resolve("host_pool:3"), 3)
    for s in ("", "tpu", "gpu:", "gpu:a", "gpu:0,,1", "gpu:0,0", "gpu:-1", "host_pool:0",
              "host_pool:x", "host_pool:1,2", "cpu:1", "auto:2", "gpu0")
        @test_throws ArgumentError resolve(s)
        @test occursin("host_pool:<count>", message(() -> resolve(s)))
    end
    if !cuda_loaded()
        @test_throws BackendUnavailable resolve("gpu:0,1")
        @test_throws BackendUnavailable resolve("gpu")
    end
end

@testset "explicit GPU choices are strict" begin
    if cuda_loaded()
        @test_skip "CUDA extension already loaded; covered by the extension test"
    else
        @test_throws BackendUnavailable resolve(:gpu)
        @test occursin("using CUDA", message(() -> resolve(:gpu)))
        @test occursin("backend!(:cpu)", message(() -> resolve(:gpu)))
        @test_throws BackendUnavailable resolve([0, 1])
        @test_throws BackendUnavailable resolve(CUDADevice(0))
        @test_throws BackendUnavailable PoolExecutor([0])
        @test_throws BackendUnavailable DevicePool([CUDADevice(0)])
        @test occursin("using CUDA", message(() -> PoolExecutor([0])))
        Ex.reset_backend!()
        @test_throws BackendUnavailable backend!(:gpu)
        withenv("BLP_BACKEND" => "cpu") do
            @test backend() == ThreadedExecutor()             # the failed call set nothing
        end
    end
    Ex.reset_backend!()
end

@testset "precedence: keyword > backend! > BLP_BACKEND > :auto" begin
    withenv("BLP_BACKEND" => "host_pool:2") do
        Ex.reset_backend!()
        b = backend()
        @test hostpool(b, 2)                                  # environment beats :auto
        @test resolve(nothing) === b
        @test backend!(:cpu) == ThreadedExecutor()
        @test backend() == ThreadedExecutor()                 # backend! beats environment
        @test hostpool(resolve(:host_pool), 1)                # keyword beats backend!
        @test resolve(nothing) == ThreadedExecutor()
        @test backend!(nothing) === nothing
        @test backend() == b                                  # environment again
        e = backend!(PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
        @test backend() === e && hostpool(e, 3)
        @test hostpool(backend!([HostDevice(1), HostDevice(2)]), 2)
    end
    withenv("BLP_BACKEND" => "cpu") do                        # read once, at first use
        Ex.reset_backend!()
        @test backend() == ThreadedExecutor()
    end
    withenv("BLP_BACKEND" => "host_pool") do
        @test backend() == ThreadedExecutor()
        backend!(nothing)
        @test backend() == ThreadedExecutor()
    end
    withenv("BLP_BACKEND" => "tpu") do
        Ex.reset_backend!()
        @test_throws ArgumentError backend()
        @test_throws ArgumentError backend()                  # still malformed at next use
    end
    withenv("BLP_BACKEND" => "") do                           # empty means unset
        Ex.reset_backend!()
        b = @test_logs (:info, r":auto chose") backend()
        cuda_loaded() || @test b == ThreadedExecutor()
    end
    Ex.reset_backend!()
end

@testset ":auto logs its choice once per session" begin
    withenv("BLP_BACKEND" => nothing) do
        Ex.reset_backend!()
        reason = cuda_loaded() ? r"no functional device|no idle device|devices" :
                                 r"CUDA not loaded"
        a = @test_logs (:info, reason) begin
            x = resolve(:auto)
            @test resolve(:auto) === x
            @test backend() === x
            x
        end
        cuda_loaded() || @test a == ThreadedExecutor()
        Ex.reset_backend!()
        @test_logs (:info, reason) resolve(:auto)
    end
    Ex.reset_backend!()
end

@testset "describe and show" begin
    d = describe(ThreadedExecutor(chunk = 5))
    @test keys(d) == (:kind, :devices, :lanes, :chunk, :threads)
    @test d == (kind = :threads, devices = String[], lanes = 0, chunk = 5,
                threads = Threads.nthreads())
    p = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)
    @test describe(p) == (kind = :pool, devices = ["HostDevice(1)", "HostDevice(2)"],
                          lanes = 2, chunk = 7, threads = Threads.nthreads())
    for (e, kind) in ((ThreadedExecutor(), "ThreadedExecutor"), (p, "PoolExecutor"))
        s = sprint(show, e)
        @test !isempty(s) && occursin(kind, s) && occursin("Julia threads", s)
        @test occursin(kind, repr("text/plain", e))
    end
    s = sprint(show, p)
    @test all(w -> occursin(w, s), ("HostDevice(2)", "lanes = 2", "chunk = 7"))
end

end
