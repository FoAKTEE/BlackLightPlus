# Launches under a register cap. A program's `register_cap` reaches `launch_capped!` of
# its lane's device; every device but a `CUDADevice` ignores it, so capped and uncapped
# programs give the same results on threads and host pools. The CUDA extension's capped
# launch must be the statements of the KernelAbstractions launch of the CUDA backend with
# `maxregs` added: compared statement by statement with the backend's source whenever CUDA
# is installed, so a change of those internals fails here. On a functional device the
# capped kernel's PTX is the uncapped PTX with a `.maxnreg` directive, and capped and
# uncapped launches are bitwise equal.
using Test, BlackLightPlus
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, DevicePool, HostDevice, CUDADevice,
                     AbstractDevice, foreach_item!, functional, register_cap, launch_capped!
const Ex = BlackLightPlus.Execution
const KA = Ex.KernelAbstractions

const CUDA_ID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const HAS_CUDA = Base.locate_package(CUDA_ID) !== nothing
const CU = HAS_CUDA ? Base.require(CUDA_ID) : nothing

# Item `k` iterates a coupled recurrence on eight `Float64` values and writes their sum:
# about 38 registers uncapped on sm_80, so a cap of 32 binds. `C` is its register cap.
struct Recur{C} <: Function
    iters::Int
end
Ex.register_cap(::Recur{C}) where {C} = C

@inline recur_step(x::NTuple{N,Float64}) where {N} =
    ntuple(j -> x[j] * 0.5 + x[mod1(j + 1, N)] * 0.25 - x[mod1(j + 3, N)] * 0.125, Val(N))
function (p::Recur)(item, i, out)
    x = ntuple(j -> Float64(item + j), Val(8))
    for _ in 1:p.iters
        x = recur_step(x)
    end
    @inbounds out[i] = sum(x)
    return nothing
end

recur(exec, p, n) = foreach_item!(exec, p, 1:n; outs = (zeros(n),))[1]

# A host device that records the caps its launches receive.
struct RecordingDevice <: AbstractDevice end
const CAPS = Any[]
Ex.backend(::RecordingDevice) = KA.CPU()
Ex.activate!(::RecordingDevice) = nothing
Ex.functional(::RecordingDevice) = true
Ex.readonly(::RecordingDevice, a::Array) = KA.constify(a)
function Ex.launch_capped!(d::RecordingDevice, kernel, args::Tuple, n, wg, cap)
    push!(CAPS, cap)
    return invoke(launch_capped!, Tuple{AbstractDevice,Any,Tuple,Any,Any,Any}, d, kernel, args,
                  n, wg, cap)
end

# The parsed source of the method `m`, without line numbers.
function method_expr(m::Method)
    file, line = functionloc(m)
    text = read(file, String)
    i = 1
    for _ in 1:line-1
        i = findnext('\n', text, i) + 1
    end
    return strip_lines(Meta.parse(text, i)[1])
end
strip_lines(x) = x
function strip_lines(ex::Expr)
    args = Any[a isa LineNumberNode ? nothing : strip_lines(a) for a in ex.args]
    ex.head === :macrocall || filter!(!isnothing, args)
    return Expr(ex.head, args...)
end
# The `@cuda` call of a body without its `maxregs = cap` argument.
drop_maxregs(x) = x
drop_maxregs(ex::Expr) =
    Expr(ex.head, (drop_maxregs(a) for a in ex.args if a != :(maxregs = cap))...)

# PTX for comparison: comments, the `.maxnreg` line and the session counters of Julia's
# function names removed.
ptx_canon(s) = replace(replace(replace(replace(s, r"//.*$"m => ""), r"^[ \t]*\.maxnreg \d+[ \t]*\n"m => ""),
                               r"julia_\w+" => m -> replace(m, r"\d+" => "N")), r"[ \t]+$"m => "")
# The PTX of every kernel compiled while `f()` runs.
function compiled_ptx(f)
    io = IOBuffer()
    Core.eval(@__MODULE__, :(CU.@device_code_ptx io = $io $f()))
    return String(take!(io))
end

@testset "register cap" begin
    @test register_cap(identity) === nothing
    @test register_cap(Recur{32}(1)) == 32 && register_cap(Recur{nothing}(1)) === nothing
    n = 1_000
    ref = recur(ThreadedExecutor(chunk = 64), Recur{nothing}(40), n)
    @test recur(ThreadedExecutor(chunk = 64), Recur{32}(40), n) == ref
    # host pools ignore the cap
    hp = DevicePool([HostDevice(1), HostDevice(2)]; lanes = 2, chunk = 128)
    @test recur(PoolExecutor(hp), Recur{32}(40), n) == ref
    @test recur(PoolExecutor(hp), Recur{nothing}(40), n) == ref
    close(hp)
    # a capped program reaches the device's launch_capped! with its cap, an uncapped one never
    rp = DevicePool([RecordingDevice()]; lanes = 1, chunk = 250)
    empty!(CAPS)
    @test recur(PoolExecutor(rp), Recur{nothing}(40), n) == ref
    @test isempty(CAPS)
    @test recur(PoolExecutor(rp), Recur{32}(40), n) == ref
    @test CAPS == fill(32, 4)
    close(rp)
end

@testset "capped CUDA launch" begin
    if !HAS_CUDA
        @test_skip "the CUDA package is not installed in this environment"
    else
        ext = Base.get_extension(BlackLightPlus, :BlackLightPlusCUDAExt)
        @test ext !== nothing
        # the capped launch is the backend's launch with `maxregs = cap` added
        kern = Ex._foreach!(CU.CUDABackend())
        backend_launch = which(kern, (Int,))
        capped = which(launch_capped!, (CUDADevice, typeof(kern), Tuple, Int, Nothing, Int))
        @test capped.module === ext
        a, b = method_expr(backend_launch), method_expr(capped)
        @test a.head === b.head === :function
        @test b.args[2] != drop_maxregs(b.args[2])            # the cap is passed
        @test a.args[2] == drop_maxregs(b.args[2])
        @test method_expr(which(CU.CUDACore.CUDAKernels.threads_to_workgroupsize, (Int, Tuple{Int}))) ==
              method_expr(which(ext.threads_to_workgroupsize, (Int, Tuple{Int})))
        if functional(CUDADevice)
            dev = CUDADevice(0)
            CU.device!(0)
            # the same program uncapped and capped: PTX equal but for `.maxnreg`, bits equal
            m = 4_096
            p = Recur{nothing}(64)
            o1, o2 = CU.zeros(Float64, m), CU.zeros(Float64, m)
            ptx1 = compiled_ptx(() -> (kern(p, 1:m, m, o1; ndrange = m, workgroupsize = 32);
                                       CU.synchronize()))
            ptx2 = compiled_ptx(() -> (launch_capped!(dev, kern, (p, 1:m, m, o2), m, 32, 32);
                                       CU.synchronize()))
            @test !occursin(".maxnreg", ptx1) && occursin(r"^\.maxnreg 32\b"m, ptx2)
            @test ptx_canon(ptx1) == ptx_canon(ptx2)
            @test Array(o1) == Array(o2) == recur(ThreadedExecutor(chunk = 64), p, m)
            # through a pool: the program's cap reaches the compilation, results bitwise
            pool = DevicePool([dev]; lanes = 2, chunk = 512)
            r = Ref{Any}()
            ptx3 = compiled_ptx(() -> (r[] = recur(PoolExecutor(pool), Recur{32}(64), m)))
            @test occursin(r"^\.maxnreg 32\b"m, ptx3)
            @test r[] == Array(o1) == recur(PoolExecutor(pool), Recur{nothing}(64), m)
            close(pool)
        end
    end
end
