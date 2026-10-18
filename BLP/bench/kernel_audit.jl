# kernel_audit.jl — offline register, stack and spill audit of BLP's device programs, no GPU.
#
# BLP has one kernel, `_foreach!`, specialised on the program it runs. Every program kind
# a product launches is captured from the product itself: each case below calls a product
# on small synthetic decks (`BLP/test/decks.jl`) with `backend = AuditExecutor()`, which
# runs it on threads and records every program passed to `foreach_item!` or
# `stream_blocks!` with the element types of its outputs. Each distinct program type is
# then made into its device twin by BLP's own `device_program` on an `AuditDevice`, a
# device whose uploads and allocations are null-pointer `CuDeviceArray`s and whose
# read-only view is CUDA's `Const` (the type `ReadOnlyArray` of the CUDA extension turns
# into at launch), so accumulators, block swaps and locate strategies get the twins a GPU
# launch builds. The kernel is compiled for sm_80 as a `CUDABackend` launch compiles it
# (`always_inline = false`, the KernelAbstractions launch context, the twin after
# `cudaconvert`, the item indices as `run_chunk!` stages them, one device vector per
# output), and `ptxas --verbose` reports the resources.
#
# Output, one line per kernel (parse with grep '^KRES'):
#   KRES <name> regs=<n> stack=<bytes> spill_st=<bytes> spill_ld=<bytes> callee_spill=<bytes>
# `regs` and `stack` (cumulative per-thread local memory) belong to the entry function;
# `spill_st` and `spill_ld` are its spill stores and loads; `callee_spill` sums the spill
# stores and loads of the non-inlined device functions it calls (the camera ray, …).
# Names: `<product>_<mode>_<matter>_<spacetime>[_<lane>][_<variant>]`; `img` the imaging
# program (`render`), `amr` its block-grid form (`render_amr`), `coords` its coordinate-
# list form (`render(plan, coords)`), `dmass`/`drathigh` the `sensitivity` derivatives,
# `ladder4` a four-rung `variants` ladder; `analytic`, `analytic_layer`
# (`photon_ring_layers`), `trace` (first window and `_resume`), `shadow`, and the Monte
# Carlo `mc_*` programs: `primary` and `layer` of the scattering runs. `MixedLanes` runs
# the `f64` and `f32` imaging kernels.
#
# Usage, from the repository root, in the CUDA environment (no GPU needed for the default
# mode; the runtime pin of its LocalPreferences.toml lets CUDA.jl compile without a driver):
#   julia -t 4 --project=results/blp/gpu/env BLP/bench/kernel_audit.jl          # print the table
#   julia -t 4 --project=results/blp/gpu/env BLP/bench/kernel_audit.jl --write  # record the baseline
#   julia -t 4 --project=results/blp/gpu/env BLP/bench/kernel_audit.jl --check  # exit 1 on growth
#   julia -t 4 --project=results/blp/gpu/env BLP/bench/kernel_audit.jl --device # on a GPU
# Options: --baseline <path> (default BLP/bench/kernel_audit_baseline.json), --json <path>
# (also write this run's table as JSON), --only <name,name,...>, --tol-regs <n> (default
# 0), --tol-stack <bytes> (default 0).
# --check fails when a kernel's registers exceed the baseline by more than --tol-regs, its
# stack by more than --tol-stack bytes, when any of its spill counts grows at all, when a
# kernel fails to compile, or when a baseline kernel is no longer audited; kernels absent
# from the baseline are reported as NEW. --device compiles each kernel for CUDA device 0
# through `cufunction`, the path of a launch, and prints
#   KDEV <name> regs=<device> offline=<offline> local=<device local bytes> stack=<offline> match=<bool>
# with `DEVICE PASS` / `DEVICE FAIL` (exit 1) on whether every register count matches.
using BLP, CUDA, KernelAbstractions, StaticArrays, Printf
using Adapt: Adapt
using BLP.Plans: CameraConfig, PlasmaConfig, RunConfig, Lanes, prepare
using BLP.Products: render, render_amr, render_analytic, photon_ring_layers, trace, shadow,
                    sensitivity, variants, mc_spectrum
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, adm_grid_from_metric, metric
using BLP.Matter: PolishDoughnut, EquatorialDisk, UniformSphere
using BLP.Transfer: StokesI, IQUV
using BLP.Data: FrameWindow, scale_density
using BLP.Numerics: GNEWT, CL, SIGMA_THOMSON
using BLP.MonteCarlo: HotcrossTable, NW, NT
using BLP.Execution: Executor, ThreadedExecutor, DevicePool, AbstractDevice, ImageBlock,
                     foreach_item!, stream_blocks!, device_program
import BLP.Execution: backend, functional, activate!, upload, readonly, describe

const BLP_DIR = dirname(@__DIR__)
const REPO = dirname(BLP_DIR)
include(joinpath(BLP_DIR, "test", "decks.jl"))     # module Decks: the synthetic decks

const CC  = CUDA.CUDACore
const GPC = CC.GPUCompiler
const SM  = v"8.0"
const CONFIG = CC._compiler_config(nothing; arch = SM, always_inline = false)
const GLOBAL = CUDA.AS.Global
DV(T) = CuDeviceArray{T,1,GLOBAL}
"The item indices of a launch as `run_chunk!` stages them."
const ITEMS = DV(Int)

# ---------------------------------------------------------------- the audit device
"A device without memory: arrays become null-pointer device arrays of the right type."
struct AuditDevice <: AbstractDevice end
struct AuditBackend <: KernelAbstractions.GPU end

null_array(::Type{T}, dims::Dims{N}) where {T,N} =
    CuDeviceArray{T,N,GLOBAL}(reinterpret(Core.LLVMPtr{T,GLOBAL}, C_NULL), dims)

backend(::AuditDevice) = AuditBackend()
functional(::AuditDevice) = true
activate!(::AuditDevice) = nothing
function upload(::AuditDevice, x::AbstractArray)
    isbitstype(eltype(x)) || throw(ArgumentError("device copy of non-isbits $(eltype(x))"))
    return null_array(eltype(x), size(x))
end
readonly(::AuditDevice, a::CuDeviceArray) = Base.Experimental.Const(a)
KernelAbstractions.allocate(::AuditBackend, ::Type{T}, dims::Tuple; unified = nothing) where {T} =
    null_array(T, dims)
KernelAbstractions.synchronize(::AuditBackend) = nothing

const POOL = DevicePool([AuditDevice()]; lanes = 1, chunk = 1)

# ---------------------------------------------------------------- the recording executor
"Runs on threads and records each program it is given with its output element types."
struct AuditExecutor <: Executor
    inner::ThreadedExecutor
    seen::Vector{Any}
end
AuditExecutor() = AuditExecutor(ThreadedExecutor(), Any[])
describe(e::AuditExecutor) = describe(e.inner)

function BLP.Execution.foreach_item!(e::AuditExecutor, prog, items::AbstractVector{<:Integer};
                                     outs::Tuple = (), order = nothing)
    push!(e.seen, (prog = prog, outs = map(eltype, outs), blocks = false))
    return foreach_item!(e.inner, prog, items; outs, order)
end
function BLP.Execution.foreach_item!(e::AuditExecutor, parts::Tuple{Vararg{Pair}};
                                     outs::Tuple = ())
    foreach(p -> push!(e.seen, (prog = p.first, outs = map(eltype, outs), blocks = false)), parts)
    return foreach_item!(e.inner, parts; outs)
end
function BLP.Execution.stream_blocks!(e::AuditExecutor, prog, work::AbstractVector{ImageBlock},
                                      block::Integer, visit; outtype::Type = Float64)
    push!(e.seen, (prog = prog, outs = (outtype,), blocks = true))
    return stream_blocks!(e.inner, prog, work, block, visit; outtype)
end

# ---------------------------------------------------------------- the decks
cfg(; res = 4, kw...) = RunConfig(; camera = CameraConfig(fov = 24.0, resolution = res), kw...)
const KERR = KerrSchildCartesian(0.9)
const BINARY = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
const ADM = let xs = -14.0:2.0:14.0
    adm_grid_from_metric(x -> metric(KERR, x), xs, xs, xs; alpha_min = 0.1)
end
const UNIFORM = Decks.uniform_deck()
const BLOCKS = Decks.amr_deck()
const WINDOW = FrameWindow(UNIFORM, scale_density(UNIFORM, 1.3), 0.0, 30.0; t_obs = 15.0)
const WINDOW_BLOCKS = FrameWindow(BLOCKS.blocks, Decks.amr_deck(scale = 1.3f0).blocks, 0.0, 30.0;
                                  t_obs = 15.0)
const MATTER = (uniform = UNIFORM, blocks = BLOCKS, window = WINDOW, windowblocks = WINDOW_BLOCKS,
                doughnut = PolishDoughnut(0.9))
const SPACETIME = (kerr = KERR, binary = BINARY, adm = ADM)
const LANE = (f64 = Lanes(), f32 = Lanes{Float32,Float64,Float64}())
const MODE = (I = StokesI(), IQUV = IQUV())
const THOMSON = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
const L_TORUS = GNEWT * Decks.TORUS_MBH / CL^2
const TORUS_PLAN = prepare(RunConfig(m_bh_cgs = Decks.TORUS_MBH,
                                     mass_scale_cgs = Decks.TORUS_DENS * L_TORUS^3,
                                     frequency_hz = 2.3e11);
                           spacetime = KerrSchildCartesian(Decks.TORUS_A), matter = BLOCKS)

plan(m, s, mode, lane; kw...) =
    prepare(cfg(; kw...); spacetime = SPACETIME[s], matter = MATTER[m], mode = MODE[mode],
            lanes = LANE[lane])

# name => (product call on an executor, suffixes of its distinct program types in order)
imaging(m, s, mode, lane) = "img_$(mode)_$(m)_$(s)_$(lane)" =>
    (e -> render(plan(m, s, mode, lane); backend = e), [""])

const CASES = [
    [imaging(m, :kerr, mode, lane) for m in (:uniform, :blocks), mode in (:I, :IQUV),
                                        lane in (:f64, :f32)]...,
    [imaging(m, :kerr, mode, :f64) for m in (:window, :windowblocks, :doughnut),
                                       mode in (:I, :IQUV)]...,
    [imaging(:blocks, s, mode, :f64) for s in (:binary, :adm), mode in (:I, :IQUV)]...,
    imaging(:blocks, :binary, :I, :f32),
    "amr_IQUV_blocks_kerr_f64" =>
        (e -> render_amr(plan(:blocks, :kerr, :IQUV, :f64; res = 8); block = 4, max_level = 1,
                         backend = e), [""]),
    "amr_I_blocks_kerr_f64" =>
        (e -> render_amr(plan(:blocks, :kerr, :I, :f64; res = 8); block = 4, max_level = 1,
                         backend = e), [""]),
    "coords_IQUV_blocks_kerr_f64" =>
        (e -> render(plan(:blocks, :kerr, :IQUV, :f64), [(1.0, 2.0), (-3.0, 4.0)]; backend = e), [""]),
    "dmass_I_uniform_kerr_f64" =>
        (e -> sensitivity(plan(:uniform, :kerr, :I, :f64), :mass_scale; backend = e), [""]),
    "drathigh_I_uniform_kerr_f64" =>
        (e -> sensitivity(plan(:uniform, :kerr, :I, :f64), :rat_high; backend = e), [""]),
    "ladder4_I_uniform_kerr_f64" =>
        (e -> variants(plan(:uniform, :kerr, :I, :f64), [1.0e18, 2.0e18, 3.0e18, 4.0e18];
                       backend = e), [""]),
    "analytic" =>
        (e -> render_analytic(prepare(cfg(); spacetime = KERR, matter = nothing),
                              EquatorialDisk(0.9); backend = e), [""]),
    "analytic_layer" =>
        (e -> photon_ring_layers(prepare(cfg(); spacetime = KERR, matter = nothing),
                                 EquatorialDisk(0.9); bands = (0,), dx = 1.0, backend = e), [""]),
    "trace_kerr_f64" =>
        (e -> trace(prepare(cfg(); spacetime = KERR, matter = UNIFORM); backend = e),
         ["", "_resume"]),
    "trace_kerr_f32" =>
        (e -> trace(prepare(cfg(); spacetime = KERR, matter = UNIFORM, lanes = LANE.f32);
                    backend = e), ["", "_resume"]),
    [("shadow_$(s)_$(lane)" => (e -> shadow(prepare(cfg(); spacetime = SPACETIME[s], matter = nothing,
                                                    lanes = LANE[lane]); backend = e), [""]))
     for (s, lane) in ((:kerr, :f64), (:kerr, :f32), (:binary, :f64), (:adm, :f64))]...,
    "mc_sphere_thin" =>
        (e -> mc_spectrum(prepare(UniformSphere()), 2_000; backend = e), ["_primary"]),
    "mc_sphere_scatter" =>
        (e -> mc_spectrum(prepare(UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)),
                          1_500; scattering = true, table = THOMSON, nu_max = 1.0e13, backend = e),
         ["_primary", "_layer"]),
    "mc_torus_thin" =>
        (e -> mc_spectrum(TORUS_PLAN, 300; C = 32.0, stride = 2, backend = e), ["_primary"]),
    "mc_torus_scatter" =>
        (e -> mc_spectrum(TORUS_PLAN, 300; scattering = true, table = THOMSON, bias = 1.0e5,
                          C = 32.0, stride = 2, backend = e), ["_primary", "_layer"]),
]

# The kernel argument types (after the launch context) of every distinct program a case
# launches: the device twin, the staged items, the item count and one device vector per
# output.
function kernel_args(call, suffixes)
    e = AuditExecutor()
    call(e)
    seen = Any[]
    for s in e.seen
        any(t -> typeof(t.prog) === typeof(s.prog) && t.blocks == s.blocks, seen) || push!(seen, s)
    end
    length(seen) == length(suffixes) ||
        error("$(length(seen)) distinct programs launched, $(length(suffixes)) named: " *
              join(string.(nameof.(typeof.(getfield.(seen, :prog)))), ", "))
    return map(seen) do s
        twin = device_program(POOL, AuditDevice(), s.prog)
        s.blocks && (twin = BLP.Execution.with_blocks(twin, null_array(SVector{3,Float64}, (1,))))
        dt = CUDA.cudaconvert(twin)
        isbits(dt) || error("the device twin of $(nameof(typeof(s.prog))) is not isbits")
        (typeof(dt), ITEMS, Int, map(DV, s.outs)...)
    end
end

# ---------------------------------------------------------------- compile + ptxas
function ptx_of(argT)
    k = BLP.Execution._foreach!(CUDABackend())
    _, _, isp, _ = KernelAbstractions.launch_config(k, 1024, nothing)
    ctx = KernelAbstractions.mkcontext(k, (1024,), isp)
    tt = Tuple{typeof(ctx), argT...}
    all(isconcretetype, tt.parameters) || error("non-concrete argument type in $tt")
    job = GPC.CompilerJob(GPC.methodinstance(typeof(k.f), tt), CONFIG)
    return GPC.JuliaContext() do _
        GPC.compile(:asm, job)[1]
    end
end

function ptxas_resources(asm)
    ptx = tempname() * ".ptx"
    write(ptx, asm)
    buf = IOBuffer()
    try
        run(pipeline(`$(CC.CUDA_Compiler_jll.ptxas()) --generate-line-info --verbose
                      --gpu-name sm_$(SM.major)$(SM.minor) --output-file $ptx.cubin $ptx`;
                     stdout = buf, stderr = buf))
    finally
        rm(ptx; force = true); rm(ptx * ".cubin"; force = true)
    end
    log = String(take!(buf))
    entry = match(r"Compiling entry function '([^']+)'", log)[1]
    regs = parse(Int, match(r"Used (\d+) registers", log)[1])
    props = Dict(m[1] => parse.(Int, (m[2], m[3], m[4])) for m in eachmatch(
        r"Function properties for (\S+)\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        log))
    frame, st, ld = props[entry]
    cum = match(r"(\d+) bytes cumulative stack size", log)
    stack = cum === nothing ? frame : parse(Int, cum[1])
    callee = sum((v[2] + v[3] for (k, v) in props if k != entry); init = 0)
    return (regs = regs, stack = stack, spill_st = st, spill_ld = ld, callee_spill = callee)
end

# Registers and local memory of the kernel as compiled for the current CUDA device.
function device_resources(argT)
    k = BLP.Execution._foreach!(CUDABackend())
    _, _, isp, _ = KernelAbstractions.launch_config(k, 1024, nothing)
    ctx = KernelAbstractions.mkcontext(k, (1024,), isp)
    hk = CUDA.cufunction(k.f, Tuple{typeof(ctx), argT...})
    return (regs = CUDA.registers(hk), local_bytes = CUDA.memory(hk).local)
end

# ---------------------------------------------------------------- baseline JSON (flat)
const FIELDS = (:regs, :stack, :spill_st, :spill_ld, :callee_spill)

function toolchain()
    ptxas_v = match(r"V(\d+\.\d+\.\d+)", read(`$(CC.CUDA_Compiler_jll.ptxas()) --version`, String))[1]
    commit = try readchomp(`git -C $REPO rev-parse --short HEAD`) catch; "unknown" end
    return (julia = string(VERSION), ptxas = ptxas_v, sm = "sm_$(SM.major)$(SM.minor)",
            ptx = string(CONFIG.params.ptx), commit = commit)
end

function write_table(path, rows, tc)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "{")
        for k in (:julia, :ptxas, :sm, :ptx, :commit)
            println(io, "  \"", k, "\": \"", getfield(tc, k), "\",")
        end
        println(io, "  \"kernels\": {")
        ok = filter(p -> p.second !== nothing, rows)
        for (i, (name, r)) in enumerate(ok)
            @printf(io, "    \"%s\": {\"regs\": %d, \"stack\": %d, \"spill_st\": %d, \"spill_ld\": %d, \"callee_spill\": %d}%s\n",
                    name, r.regs, r.stack, r.spill_st, r.spill_ld, r.callee_spill,
                    i == length(ok) ? "" : ",")
        end
        println(io, "  }\n}")
    end
end

function read_table(path)
    txt = read(path, String)
    tc = Dict(m[1] => m[2] for m in eachmatch(r"\"(\w+)\": \"([^\"]*)\"", txt))
    rows = Dict{String,NamedTuple}()
    for m in eachmatch(r"\"(\w+)\": \{\"regs\": (\d+), \"stack\": (\d+), \"spill_st\": (\d+), \"spill_ld\": (\d+), \"callee_spill\": (\d+)\}", txt)
        rows[m[1]] = NamedTuple{FIELDS}(parse.(Int, (m[2], m[3], m[4], m[5], m[6])))
    end
    return tc, rows
end

# ---------------------------------------------------------------- main
function main(args)
    opt(flag, default) = (i = findfirst(==(flag), args); i === nothing ? default : args[i + 1])
    mode = "--write" in args ? :write : "--check" in args ? :check :
           "--device" in args ? :device : :print
    path = opt("--baseline", joinpath(BLP_DIR, "bench", "kernel_audit_baseline.json"))
    json = opt("--json", nothing)
    only = opt("--only", nothing)
    tol_regs = parse(Int, opt("--tol-regs", "0"))
    tol_stack = parse(Int, opt("--tol-stack", "0"))
    mode === :write && only !== nothing && error("--write records every kernel; drop --only")
    wanted = only === nothing ? nothing : Set(split(only, ','))

    tc = toolchain()
    println("# kernel_audit ", join(("$k=$(getfield(tc, k))" for k in keys(tc)), " "),
            " always_inline=false threads=", Threads.nthreads())
    kernels = Pair{String,Any}[]
    for (prefix, (call, suffixes)) in CASES
        names = prefix .* suffixes
        wanted === nothing || any(in(wanted), names) || continue
        args_of = try
            kernel_args(call, suffixes)
        catch err
            println("KERR ", prefix, " capture: ", first(sprint(showerror, err), 300))
            fill(nothing, length(names))
        end
        for (n, a) in zip(names, args_of)
            wanted === nothing || n in wanted || continue
            push!(kernels, n => a)
        end
    end
    wanted === nothing || length(kernels) == length(wanted) ||
        error("unknown kernel in --only $only")

    rows = Pair{String,Any}[]
    for (name, argT) in kernels
        t = time()
        r = argT === nothing ? nothing : try
            ptxas_resources(ptx_of(argT))
        catch err
            println("KERR ", name, " ", first(sprint(showerror, err), 300))
            nothing
        end
        push!(rows, name => r)
        r === nothing ||
            @printf("KRES %-36s regs=%d stack=%d spill_st=%d spill_ld=%d callee_spill=%d (%.0f s)\n",
                    name, r.regs, r.stack, r.spill_st, r.spill_ld, r.callee_spill, time() - t)
        flush(stdout)
    end
    failed = any(p -> p.second === nothing, rows)
    json === nothing || write_table(json, rows, tc)

    if mode === :device
        CUDA.functional() || (println("DEVICE SKIP: CUDA not functional"); return 1)
        println("# device ", CUDA.name(CUDA.device()), " capability ", CUDA.capability(CUDA.device()))
        for ((name, argT), (_, r)) in zip(kernels, rows)
            argT === nothing && continue
            d = try
                device_resources(argT)
            catch err
                println("KDEV ", name, " ERROR ", first(sprint(showerror, err), 300))
                failed = true; continue
            end
            ok = r !== nothing && d.regs == r.regs
            ok || (failed = true)
            @printf("KDEV %-36s regs=%d offline=%s local=%d stack=%s match=%s\n", name, d.regs,
                    r === nothing ? "?" : string(r.regs), d.local_bytes,
                    r === nothing ? "?" : string(r.stack), ok)
        end
        println(failed ? "DEVICE FAIL" : "DEVICE PASS")
    elseif mode === :write
        failed && (println("NOT WRITTEN: a kernel failed to compile"); return 1)
        write_table(path, rows, tc)
        println("# baseline written to ", path)
    elseif mode === :check
        btc, base = read_table(path)
        for k in ("julia", "ptxas", "ptx")
            get(btc, k, "") == string(getfield(tc, Symbol(k))) ||
                println("WARN toolchain $k: baseline $(get(btc, k, "?")) vs now $(getfield(tc, Symbol(k)))")
        end
        if only === nothing
            for name in sort!(collect(setdiff(keys(base), first.(rows))))
                println("CHECK ", name, " STALE (in baseline, not audited)"); failed = true
            end
        end
        for (name, r) in rows
            r === nothing && (println("CHECK ", name, " FAIL (compile error)"); continue)
            b = get(base, name, nothing)
            b === nothing && (println("CHECK ", name, " NEW"); continue)
            bad = String[]
            r.regs > b.regs + tol_regs && push!(bad, "regs $(b.regs)->$(r.regs)")
            r.stack > b.stack + tol_stack && push!(bad, "stack $(b.stack)->$(r.stack)")
            for f in (:spill_st, :spill_ld, :callee_spill)
                getfield(r, f) > getfield(b, f) && push!(bad, "$f $(getfield(b, f))->$(getfield(r, f))")
            end
            if isempty(bad)
                better = any(getfield(r, f) < getfield(b, f) for f in FIELDS)
                println("CHECK ", name, better ? " IMPROVED" : " OK")
            else
                println("CHECK ", name, " REGRESSION ", join(bad, ", ")); failed = true
            end
        end
        println(failed ? "CHECK FAIL" : "CHECK PASS")
    end
    return failed ? 1 : 0
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main(ARGS))
