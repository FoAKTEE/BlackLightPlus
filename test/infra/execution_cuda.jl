# The CUDA extension: it loads with `CUDA` and supplies the `CUDADevice` methods; without
# a functional device an explicit `:gpu` throws and `:auto` falls back to threads with the
# reason logged. Skipped when the CUDA package is not installed in the environment. On a
# functional device the two lanes of a device share the residency arrays: two lanes give
# the sums of one, each lane writes its own accumulators, and in the stress run one lane
# reads shared arrays in a long kernel while the other zeroes, uploads, launches, frees
# and evicts; the long kernel's sums are exact, the shared arrays intact, an eviction
# waits for the busy lane, and with the lane check on a gather under a busy lane throws.
using Test, BlackLightPlus
using BlackLightPlus.Execution: resolve, ThreadedExecutor, PoolExecutor, CUDADevice, DevicePool,
                     BackendUnavailable, cuda_loaded, functional, select_devices,
                     stage_alloc, backend, ExactScalars, add!, gather!, foreach_item!,
                     resident, device_program
const Ex = BlackLightPlus.Execution

const CUDA_ID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const HAS_CUDA = Base.locate_package(CUDA_ID) !== nothing
const CU = HAS_CUDA ? Base.require(CUDA_ID) : nothing

# Item `k` adds `a[k]` and its square to exact scalars (a residency array, an accumulator
# twin per lane).
struct SharedSum{A,S} <: Function
    a::A
    scalars::S
end
Ex.Adapt.@adapt_structure SharedSum
function (p::SharedSum)(item, i)
    v = @inbounds p.a[item]
    add!(p.scalars, 1, v)
    add!(p.scalars, 2, v * v)
    return nothing
end

function shared_sums(exec, a)
    s = ExactScalars(2)
    foreach_item!(exec, SharedSum(a, s), 1:length(a))
    lanes = length(s.twins.list)
    gather!(s)
    return (acc = s.acc, n = s.n, lanes = lanes)
end

@noinline temporary_call(exec) = (shared_sums(exec, rand(30_000)); nothing)

# A long read of two shared arrays: item `i` sums `a[j] + b[j]` over `j ≡ i (mod m)`,
# `reps` times, in wrapping `UInt64` arithmetic, into `out[i]`.
struct SpinRead{A,B} <: Function
    a::A
    b::B
    m::Int
    reps::Int
end
Ex.Adapt.@adapt_structure SpinRead
function (p::SpinRead)(item, i, out)
    n = length(p.a)
    acc = zero(UInt64)
    for _ in 1:p.reps
        j = item
        while j <= n
            acc += @inbounds p.a[j] + p.b[j]
            j += p.m
        end
    end
    @inbounds out[i] = acc
    return nothing
end

# What `SpinRead(a, b, m, reps)` writes for items `1:m`.
function spin_expected(a, b, m, reps)
    s = zeros(UInt64, m)
    for j in eachindex(a)
        s[mod1(j, m)] += a[j] + b[j]
    end
    return UInt64(reps) .* s
end

# One launch of the device program `dp` over `1:m` on `lane`; the output and the time it
# returned.
function spin_on!(lane, dp, m)
    out = zeros(UInt64, m)
    Ex.run_chunk!(lane, dp, dp, 1:m, (out,))
    return (out = out, t = time())
end

# Run `f(lane)` on `lane` as a pool job; its value.
function on_lane(f, pool, lane)
    r = Ref{Any}(nothing)
    Ex.run_job!((l, _, _) -> (r[] = f(l)), pool, [lane], (1,))
    return r[]
end

busy(lane) = !Ex.queue_idle(lane.dev, @atomic lane.queue)

# Wait until `lane` has unfinished work queued; false after `timeout` seconds.
function wait_busy(lane; timeout = 60.0)
    t0 = time()
    while !busy(lane)
        time() - t0 > timeout && return false
        sleep(0.001)
    end
    return true
end

# Lane A runs a long kernel over two shared arrays while lane B zeroes accumulators,
# uploads, launches a short kernel over the same arrays (its stream then owns them) and
# grows (frees) a lane buffer; then the cache entry of one shared array is marked
# collected, and lane B evicts it (waiting for lane A) and fills a new buffer of its size,
# which a free that did not wait would let overwrite the array lane A still reads.
function stress_run(dev; n = 2^23, m = 2^15, seconds = 4.0)
    pool = DevicePool([dev]; lanes = 2)
    la, lb = pool.lanes
    a_host, c_host = rand(UInt64, n), rand(UInt64, n)
    spin(mi, reps) = device_program(pool, dev, SpinRead(a_host, c_host, mi, reps))
    # calibrate the long kernel to about `seconds`: grow the passes until a launch takes
    # half a second, then scale
    on_lane(l -> spin_on!(l, spin(m, 1), m), pool, la)
    reps, dt = 16, 0.0
    while true
        t0 = time()
        on_lane(l -> spin_on!(l, spin(m, reps), m), pool, la)
        dt = time() - t0
        (dt >= 0.5 || reps >= 10^6) && break
        reps *= 4
    end
    reps = clamp(round(Int, reps * seconds / dt), 16, 10^7)
    mb = 2^12
    dpa, dpb = spin(m, reps), spin(mb, 1)
    sb = ExactScalars(2)
    u_host = rand(5_000)
    ready = Channel{Any}(1); go = Channel{Nothing}(1)
    ta = Threads.@spawn on_lane(l -> spin_on!(l, dpa, m), pool, la)
    started = wait_busy(la)
    tb = Threads.@spawn on_lane(pool, lb) do l
        acc = Ex.program_twin!(Ex.ProgramTwins(pool, SharedSum(u_host, sb)), l)
        Ex.run_chunk!(l, acc, acc, 1:length(u_host), ())
        short = spin_on!(l, dpb, mb)
        g1 = Ex.lane_buffer(l, :stress_grow, UInt64, 2^20)
        g2 = Ex.lane_buffer(l, :stress_grow, UInt64, 2^21)
        put!(ready, (short = short, a_busy = busy(la), grown = g1 !== g2))
        take!(go)
        a_busy_evict = busy(la)
        t_start = time()
        resident(pool, dev, rand(10))
        t_evict = time()
        post = Ex.lane_buffer(l, :stress_post, UInt64, n)
        fill!(post, 0xa5a5a5a5a5a5a5a5)
        Ex.KernelAbstractions.synchronize(l.backend)
        (a_busy_evict = a_busy_evict, t_start = t_start, t_evict = t_evict,
         post_ok = all(==(0xa5a5a5a5a5a5a5a5), Array(post)))
    end
    b1 = take!(ready)
    a_busy_gather = busy(la)
    old = Ex.lane_check!(true)
    gather_err = try
        gather!(sb); nothing
    catch e
        e
    finally
        Ex.lane_check!(old)
    end
    # the host of `a` counts as collected: its cache entry keeps the device array only
    cache = pool.resident[1]
    @lock cache.lock begin
        id = objectid(a_host)
        cache.twins[id] = (WeakRef(), cache.twins[id][2])
    end
    put!(go, nothing)
    b2 = fetch(tb)
    ra = fetch(ta)
    old = Ex.lane_check!(true)
    try
        gather!(sb)
    finally
        Ex.lane_check!(old)
    end
    c_twin = parent(dpa.b)
    res = (started = started, reps = reps, a_ok = ra.out == spin_expected(a_host, c_host, m, reps),
           b_ok = b1.short.out == spin_expected(a_host, c_host, mb, 1), b1 = b1,
           a_busy_gather = a_busy_gather, gather_err = gather_err, b2 = b2, ta_done = ra.t,
           c_ok = Array(c_twin) == c_host, c_shared = !c_twin.data[].synchronizing,
           a_freed = parent(dpa.a).data.freed,
           u_sums = sb.acc == shared_sums(ThreadedExecutor(chunk = 64), u_host).acc)
    close(pool)
    return res
end

@testset "CUDA extension" begin
    if !HAS_CUDA
        @test_skip "the CUDA package is not installed in this environment"
    else
        ext = Base.get_extension(BlackLightPlus, :BlackLightPlusCUDAExt)
        @test ext !== nothing && cuda_loaded()
        @test which(stage_alloc, (CUDADevice, Type{Float64}, Int)).module === ext
        @test which(backend, (CUDADevice,)).module === ext
        if functional(CUDADevice)
            @test resolve(:gpu) isa PoolExecutor
            @test !isempty(select_devices(; fallback = true))
            dev = CUDADevice(0)
            # two lanes reading one shared array into their own accumulators equal one lane
            a = rand(2^18)
            p1 = DevicePool([dev]; lanes = 1)
            p2 = DevicePool([dev]; lanes = 2)
            one = shared_sums(PoolExecutor(p1; chunk = 4096), a)
            two = shared_sums(PoolExecutor(p2; chunk = 4096), a)
            @test two.acc == one.acc && two.n == one.n
            @test one.acc == shared_sums(ThreadedExecutor(chunk = 64), a).acc
            @test 1 <= two.lanes <= 2
            # the lanes' streams; each lane's device form shares the residency array
            # (implicit synchronisation off) and owns its accumulator
            q1, q2 = (@atomic p2.lanes[1].queue), (@atomic p2.lanes[2].queue)
            @test q1 isa CU.CuStream && q2 isa CU.CuStream && q1 != q2
            s = ExactScalars(2)
            tw = Ex.ProgramTwins(p2, SharedSum(a, s))
            ts = [on_lane(l -> Ex.program_twin!(tw, l), p2, l) for l in p2.lanes]
            @test parent(ts[1].a) === parent(ts[2].a) === resident(p2, dev, a)
            @test !parent(ts[1].a).data[].synchronizing
            @test ts[1].scalars.acc !== ts[2].scalars.acc && length(s.twins.list) == 2
            gather!(s)
            @test isempty(s.twins.list) && s.n == [0, 0]
            # the residency cache returns to its size once a call's host arrays are collected
            n0 = Ex.resident_count(p2, dev)
            temporary_call(PoolExecutor(p2; chunk = 4096))
            @test Ex.resident_count(p2, dev) == n0 + 1
            for _ in 1:3
                GC.gc(true)
            end
            @test Ex.resident_count(p2, dev) == n0
            # sharing off: an upload keeps CUDA.jl's implicit synchronisation
            old = Ex.lane_sharing!(false)
            try
                b = rand(1_000)
                @test resident(p2, dev, b).data[].synchronizing
                @test shared_sums(PoolExecutor(p2; chunk = 128), b).acc ==
                      shared_sums(PoolExecutor(p1; chunk = 128), b).acc
            finally
                Ex.lane_sharing!(old)
            end
            foreach(close, (p1, p2))
            # the stress run
            r = stress_run(dev)
            @test r.started && r.reps >= 16
            @test r.a_ok && r.b_ok && r.c_ok && r.c_shared && r.u_sums
            @test r.b1.a_busy && r.b1.grown            # lane B's work overlapped lane A's kernel
            @test r.a_busy_gather && r.gather_err isa ErrorException &&
                  occursin("lane check", r.gather_err.msg)
            # the eviction began under a busy lane, waited for it, then freed the array
            @test r.b2.a_busy_evict && r.a_freed && r.b2.post_ok
            @test r.b2.t_evict >= r.ta_done - 0.25
        else
            @test_throws BackendUnavailable resolve(:gpu)
            err = try resolve(:gpu); nothing catch e; e end
            @test occursin("CUDA_VISIBLE_DEVICES", sprint(showerror, err))
            @test_throws BackendUnavailable PoolExecutor([0])
            @test !functional(CUDADevice(0)) && isempty(select_devices())
            withenv("BLP_BACKEND" => nothing) do
                Ex.reset_backend!()
                a = @test_logs (:info, r"no functional device") resolve(:auto)
                @test a == ThreadedExecutor()
            end
        end
        Ex.reset_backend!()
    end
end
