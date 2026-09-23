# kernel_audit.jl — offline register / stack / spill audit of every GPU kernel, no device.
#
# Each KernelAbstractions kernel in src/ is compiled for sm_80 exactly as a CUDABackend
# launch would compile it (CUDACore's compiler config for an explicit arch: default PTX
# ISA, always_inline=false, the KA launch context type), then `ptxas --verbose` reports
# the resources. Argument types mirror the production launch sites (host arrays become
# CuDeviceArray; isbits parameter bundles are built by the production builders).
#
# Output, one line per kernel (parse with grep '^KRES'):
#   KRES <name> regs=<n> stack=<bytes> spill_st=<bytes> spill_ld=<bytes> callee_spill=<bytes>
# `regs` and `stack` (cumulative per-thread local memory) belong to the entry; spill_st /
# spill_ld are the entry's spill stores / loads; callee_spill sums stores + loads over the
# non-inlined device functions it calls.
#
# Usage (from the repo root):
#   julia --project=. bench/kernel_audit.jl                   # print the table
#   julia --project=. bench/kernel_audit.jl --write           # record the baseline
#   julia --project=. bench/kernel_audit.jl --check           # exit 1 on a regression
# Options: --baseline <path> (default bench/kernel_audit_baseline.json)
#          --only <name,name,...>   --tol-regs <n> (default 0)   --tol-stack <bytes> (default 0)
# --check fails when a kernel's registers or stack exceed baseline + tolerance, when any
# of its spill counts grows at all, when a kernel fails to compile, or when a baseline
# kernel is no longer audited. Kernels absent from the baseline are reported as NEW.
using BlackLightPlus, CUDA, KernelAbstractions, StaticArrays, Printf
using ForwardDiff: Dual
using BlackLightPlus: EquatorialDisk, SuperposedKerrSchild, SphereModel, Spectrum, MCRng
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig

const REPO = dirname(@__DIR__)
for f in ("gpu_image.jl", "gpu_pol.jl", "gpu_amr.jl", "gpu_spacetime.jl", "gpu_image_ad.jl",
          "gpu_analytic.jl", joinpath("geodesics", "gpu_tracer.jl"),
          joinpath("montecarlo", "mc_gpu.jl"))
    include(joinpath(REPO, "src", f))
end

const CC  = CUDA.CUDACore
const GPC = CC.GPUCompiler
const SM  = v"8.0"
const CONFIG = CC._compiler_config(nothing; arch = SM, always_inline = false)

# ---------------------------------------------------------------- argument types
const F   = Float64
const R8  = SVector{8,Float64}
const R8f = SVector{8,Float32}
const S4  = SVector{4,Float64}
DV(T) = CuDeviceArray{T,1,CUDA.AS.Global}
DA(T, N) = CuDeviceArray{T,N,CUDA.AS.Global}
const GRID = DA(Float32, 4)                         # DeviceFluidGrid.dev_data
const BLK, XL, LEAF = DA(Float32, 5), DA(F, 2), DA(Int32, 3)   # DeviceAMRGrid fields

# The benchmark deck (bench/bench_gpu.jl): a = 0.9, 230 GHz, thermal eDF.
const CFG = RunConfig(bh_a = 0.9, m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1.0e24,
                      frequency_hz = 230.0e9,
                      camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                            fov = 40.0, resolution = 16),
                      plasma = PlasmaConfig(rat_high = 40.0, rat_low = 1.0, gamma = 13.0 / 9.0,
                                            use_p = false, sigma_cut = 100.0,
                                            theta_e_min = 0.3))
const GP  = typeof(_gimg_params(CFG))
const GPP = typeof(_gpol_params(CFG))
const AMRP = typeof((ni = 1, nj = 1, nk = 1, nmb = 1, x1min = 0.0, x2min = 0.0, x3min = 0.0,
                     inv_fdx1 = 1.0, inv_fdx2 = 1.0, inv_fdx3 = 1.0, N1 = 1, N2 = 1, N3 = 1))
const BBH = SuperposedKerrSchild{2,Float64}
const ADM = DevADMParams{DA(F, 4)}

# the per-lane march scalars (a, r_h, C, escape, R_em) at the geodesic-lane type T
march(T) = (T, T, T, T, T)
uni(out, y, gp, T) = (DV(out), DV(y), GRID, gp, march(T)..., Int, Int, F, F, Int)
uni_sl(out, gp) = (DV(out), DV(R8), GRID, GRID, F, F, gp, march(F)..., Int, Int, F, F, Int)
amr(out, y, gp, T) = (DV(out), DV(y), BLK, XL, XL, LEAF, AMRP, gp, march(T)..., Int, Int)
amr_sl(out, gp) = (DV(out), DV(R8), BLK, BLK, XL, XL, LEAF, AMRP, F, F, gp, march(F)..., Int, Int)
gsp(out, sp, gp) = (DV(out), DV(R8), BLK, XL, XL, XL, XL, sp, gp, F, F, F, Int,
                    Int, Int, Int, Int, Int)
trace(T) = (DA(SVector{8,T}, 2), DA(T, 2), DV(Int32), DV(SVector{8,T}), T, T, T, T, Int, T, Int)

# MC parameter bundles from the mc_gpu.jl host builders, on the default sphere.
function mc_params()
    model = SphereModel()
    tabs = _mc_sphere_tables(model, 1.0e3, 1.0e8, 1.0e13, 64, false)
    thin = _mc_thin_params(model, tabs, Spectrum(; nu_min = 1.0e8, nu_max = 1.0e16,
                                                 n_ebins = 200, n_thbins = 6), 1)
    layout = BlackLightPlus.MonteCarlo.MCStreamLayout(tabs.N_gen, 100_000, 3)
    scat = _mc_scatter_params(model, tabs, Spectrum(; nu_min = 1.0e8, nu_max = 1.0e17,
                                                     n_ebins = 140, n_thbins = 6), 1, 3, layout)
    return typeof(thin), typeof(scat)
end
const MCP_THIN, MCP_SCAT = mc_params()
const XBINS = (DA(Int64, 3), DA(Int64, 3), DA(Int64, 2))     # exact limbs w, wE; count
const CHILD = NamedTuple{(:x, :p, :w, :n, :r),
                         Tuple{DV(S4), DV(S4), DV(F), DV(Int32), DV(MCRng)}}

const DUAL = Dual{_GIADTag,Float64,1}
const GP_AD_RH = typeof(_giad_params(CFG, :rat_high))
const GP_AD_MS = typeof(_giad_params(CFG, :mass_scale))

# name => (kernel constructor, argument types after the KA context)
const KERNELS = [
    "fused"            => (_fused_kernel!,    uni(F, R8, GP, F)),
    "fused_f32"        => (_fused_kernel!,    uni(F, R8f, GP, Float32)),
    "fused_kv4"        => (_fused_kernel_kv!, uni(SVector{4,F}, R8, NTuple{4,GP}, F)),
    "fused_sl"         => (_fused_kernel_sl!, uni_sl(F, GP)),
    "fused_ad_rathigh" => (_fused_kernel!,    uni(DUAL, R8, GP_AD_RH, F)),
    "fused_ad_mass"    => (_fused_kernel!,    uni(DUAL, R8, GP_AD_MS, F)),
    "gpol"             => (_gpol_kernel!,     uni(S4, R8, GPP, F)),
    "gpol_f32"         => (_gpol_kernel!,     uni(S4, R8f, GPP, Float32)),
    "gpol_sl"          => (_gpol_kernel_sl!,  uni_sl(S4, GPP)),
    "gpol_stiff"       => (_gpol_stiff_kernel!, (DV(S4), DV(R8), GRID, GPP, march(F)..., Int,
                                                 Int, F, F, F, Int)),
    "famr"             => (_famr_kernel!,     amr(F, R8, GP, F)),
    "famr_f32"         => (_famr_kernel!,     amr(F, R8f, GP, Float32)),
    "famr_sl"          => (_famr_kernel_sl!,  amr_sl(F, GP)),
    "gpol_amr"         => (_gpol_amr_kernel!, amr(S4, R8, GPP, F)),
    "gpol_amr_f32"     => (_gpol_amr_kernel!, amr(S4, R8f, GPP, Float32)),
    "gpol_amr_sl"      => (_gpol_amr_kernel_sl!, amr_sl(S4, GPP)),
    "amr_probe"        => (_amr_probe_kernel!, (DV(R8), DV(SVector{3,F}), BLK, XL, XL, LEAF,
                                                AMRP, Int)),
    "amr_locate_probe" => (_amr_locate_probe_kernel!, (DV(Int32), DV(Int32), DV(SVector{3,F}),
                                                       XL, XL, LEAF, AMRP, Int)),
    "gsp_bbh"          => (_gsp_amr_kernel!,     gsp(F, BBH, GP)),
    "gsp_adm"          => (_gsp_amr_kernel!,     gsp(F, ADM, GP)),
    "gsp_pol_bbh"      => (_gsp_pol_amr_kernel!, gsp(S4, BBH, GPP)),
    "gsp_pol_adm"      => (_gsp_pol_amr_kernel!, gsp(S4, ADM, GPP)),
    "gsp_shadow_bbh"   => (_gsp_shadow_kernel!, (DV(R8), DV(S4), DV(R8), BBH, F, F, Int, Int)),
    "gsp_shadow_adm"   => (_gsp_shadow_kernel!, (DV(R8), DV(S4), DV(R8), ADM, F, F, Int, Int)),
    "analytic"         => (_ganalytic_kernel!, (DV(F), DV(F), DV(F), DV(F), DV(F), DV(F), F, F,
                                                NTuple{3,Int}, F, EquatorialDisk{F}, Bool,
                                                Int, Int, Int)),
    "trace"            => (_trace_kernel!,    trace(F)),
    "trace_f32"        => (_trace_kernel!,    trace(Float32)),
    "mc_thin"          => (_mc_thin_kernel!,  (XBINS..., DV(F), Nothing, MCP_THIN, Int, Int)),
    "mc_scatter_primary" => (_mc_scatter_primary_kernel!,
                             (XBINS..., CHILD, DV(Int64), Int, DA(Int64, 3), DV(F), Nothing,
                              DA(F, 2), MCP_SCAT, Int, Int)),
    "mc_scatter_layer" => (_mc_scatter_layer_kernel!,
                           (XBINS..., CHILD, CHILD, DV(Int64), Int, DA(Int64, 3), DA(F, 2),
                            MCP_SCAT, Int)),
]

# ---------------------------------------------------------------- compile + ptxas
function ptx_of(kfun, argT)
    k = kfun(CUDABackend())
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

# ---------------------------------------------------------------- baseline JSON (flat)
const FIELDS = (:regs, :stack, :spill_st, :spill_ld, :callee_spill)

function toolchain()
    ptxas_v = match(r"V(\d+\.\d+\.\d+)", read(`$(CC.CUDA_Compiler_jll.ptxas()) --version`, String))[1]
    commit = try readchomp(`git -C $REPO rev-parse --short HEAD`) catch; "unknown" end
    return (julia = string(VERSION), ptxas = ptxas_v, sm = "sm_$(SM.major)$(SM.minor)",
            ptx = string(CONFIG.params.ptx), commit = commit)
end

function write_baseline(path, rows, tc)
    open(path, "w") do io
        println(io, "{")
        for k in (:julia, :ptxas, :sm, :ptx, :commit)
            println(io, "  \"", k, "\": \"", getfield(tc, k), "\",")
        end
        println(io, "  \"kernels\": {")
        for (i, (name, r)) in enumerate(rows)
            @printf(io, "    \"%s\": {\"regs\": %d, \"stack\": %d, \"spill_st\": %d, \"spill_ld\": %d, \"callee_spill\": %d}%s\n",
                    name, r.regs, r.stack, r.spill_st, r.spill_ld, r.callee_spill,
                    i == length(rows) ? "" : ",")
        end
        println(io, "  }\n}")
    end
end

function read_baseline(path)
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
    mode = "--write" in args ? :write : "--check" in args ? :check : :print
    path = opt("--baseline", joinpath(REPO, "bench", "kernel_audit_baseline.json"))
    only = opt("--only", nothing)
    tol_regs = parse(Int, opt("--tol-regs", "0"))
    tol_stack = parse(Int, opt("--tol-stack", "0"))
    sel = only === nothing ? KERNELS : filter(p -> p.first in split(only, ','), KERNELS)
    only === nothing || length(sel) == length(split(only, ',')) || error("unknown kernel in --only $only")
    mode === :write && only !== nothing && error("--write records every kernel; drop --only")

    tc = toolchain()
    println("# kernel_audit ", join(("$k=$(getfield(tc, k))" for k in keys(tc)), " "),
            " always_inline=false")
    rows = Pair{String,Any}[]
    for (name, (kfun, argT)) in sel
        r = try
            ptxas_resources(ptx_of(kfun, argT))
        catch err
            println("KERR ", name, " ", sprint(showerror, err)[1:min(end, 300)])
            nothing
        end
        push!(rows, name => r)
        r === nothing ||
            @printf("KRES %-18s regs=%d stack=%d spill_st=%d spill_ld=%d callee_spill=%d\n",
                    name, r.regs, r.stack, r.spill_st, r.spill_ld, r.callee_spill)
    end
    failed = any(p -> p.second === nothing, rows)

    if mode === :write
        failed && (println("NOT WRITTEN: a kernel failed to compile"); return 1)
        write_baseline(path, rows, tc)
        println("# baseline written to ", path)
    elseif mode === :check
        btc, base = read_baseline(path)
        for k in ("julia", "ptxas", "ptx")
            get(btc, k, "") == string(getfield(tc, Symbol(k))) ||
                println("WARN toolchain $k: baseline $(get(btc, k, "?")) vs now $(getfield(tc, Symbol(k)))")
        end
        audited = Set(first.(rows))
        if only === nothing
            for name in sort!(collect(setdiff(keys(base), audited)))
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
