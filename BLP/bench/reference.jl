# reference.jl — wall time and throughput of the reference package BlackLightPlus (the
# repository's src/) on the decks of bench/bench_multigpu.jl and on bench.jl's torus deck,
# written as bench.jl records so that `bench.jl merge-reference` pairs every leg with the
# BLP leg of the same deck (bench.jl with BLP_DECK=reference or BLP_COMPARE=1).
#
# Run from the repository root in results/blp/gpu/env_old (the repository developed,
# CUDA, the pinned CUDA runtime); bench_multigpu.jl loads the device files from the `src`
# beside the active project, which in env_old is a link to the repository's src/:
#   BLP_BACKEND=gpu BLP_DEVICES=0 julia -t 4 --project=results/blp/gpu/env_old BLP/bench/reference.jl
#   BLP_BACKEND=cpu julia -t 16 --project=results/blp/gpu/env_old BLP/bench/reference.jl
#   BLP/bench/slurm/run_reference.sh ...     (both packages, one Slurm job per item; README.md)
#
# Decks (BLP_DECK) and GPU legs (BLP_LEGS, comma list; default every leg below except the
# binary ones and amr_driver_pol_closure):
#   reference (default): bench/bench_multigpu.jl, included unchanged: its decks, uploads and
#     calls. Imaging: a = 0.9, M = 6.2e9 Msun, mass scale 1e24 g, 230 GHz, i = 60°,
#     d = 1000, fov 40, R_high 40, R_low 1, γ = 13/9, use_p = false, σ_cut 100,
#     Θe_min 0.3 on torus.mhd_w_bcc.00999.athdf (BLP_TEST_GRMHD).
#       fused_384, fused_768           render_gpu_fused on resample_fluid_uniform(snap, 40, 256)
#       fused_mixed_768                the same with precision = :mixed
#       fused_pol_256, fused_pol_384   render_gpu_polarized on the resample
#       amr_384, amr_pol_384           render_gpu_amr, render_gpu_amr_polarized on the blocks
#       amr_driver_pol                 render_amr(cfg, path; block 8, max_level 2, polarized,
#                                      stokes = AMRStokes(cfg, blocks)), base 384
#       amr_driver_pol_closure         the same through the per-level closure
#       analytic_1024                  render_analytic_gpu, g-disk a = 0.94, i = 17°, fov 20
#       mc_thin                        mc_spectrum_gpu(SphereModel(), Ns)
#       mc_scatter                     mc_spectrum_gpu(ScatteringSphere(), table, Ns), the
#                                      table read from BLP_REF_GPUMONTY (gpumonty hotcross.dat;
#                                      by default the Mahakala checkout beside this repository)
#       bbh455_480, bbh455_pol_480, bbh122_pol_480, bbh_i90_amr_pol   the binary legs
#   torus: bench.jl's deck (a = 0.9, mass scale 1e-21 L³, fov 30, R_high 10, R_low 1,
#     σ_cut 1, use_p): img_192, img_384, img_768 (render_gpu_amr on the blocks) and
#     amr_384 (render_amr, block 16, max_level 2, polarized, AMRStokes on the blocks).
# CPU legs (BLP_BACKEND=cpu), the reference's CPU entry points on the same decks:
#   amr_384, amr_pol_384, img_<res>   render(cfg, snapshot; polarized): the two-phase render
#                                     (trace_rays_sampled, then the march along the stored
#                                     states), given the loaded snapshot
#   amr_driver_pol (reference, block 8), amr_384 (torus, block 16)
#                                     render_amr(cfg, path; max_level 2, polarized) with its
#                                     own two-phase closure, which reads the snapshot per call
#   analytic_1024                     render_analytic(cfg; source = EquatorialDisk(0.94))
#   mc_thin, mc_scatter               mc_spectrum(SphereModel(), Ns),
#                                     mc_spectrum_scatter(ScatteringSphere(), table, Ns)
#
# Protocol per leg, as bench.jl: the setup (the harness's uploads; timed as `setup_s`, not
# part of `first_s`; the harness's setup of amr_driver_pol, amr_driver_pol_closure and
# bbh_i90_amr_pol runs the driver once on one device to count its rays, `setup_runs`), one
# first call (`first_s`), then BLP_REPEAT (3) timed calls; `wall_s` is the best. GPU
# configurations, one record each: `default`, the device list as the harness passes it
# (the cached DevicePool of lanes 2, chunk 65536; the Monte Carlo drivers cut the
# primaries into chunks of their own `chunk` keyword, 2^20); then, for each chunk C of
# BLP_REF_CHUNKS (131072), a fresh DevicePool(devices; lanes = 2, chunk = C), and for the
# Monte Carlo legs the drivers' `chunk` keyword at 65536 and at each C. With BLP_PHASES=1
# (the default on a GPU) one more call of the default configuration runs under the CUPTI
# profiler and is written as a `phase` record (common.jl `cuda_phases`). Every record
# carries the threads, devices, lanes, chunk, the 1-minute load before and after the timed
# calls, the cores other processes kept busy during them, the nvidia-smi state, the commit,
# the tree hash of src/ and the Slurm job.
#
# Output: one JSON object per line appended to BLP_OUT (default
# BLP/bench/results/reference/raw/ref.jsonl); the first configuration's output arrays go to
# BLP_SAVE_DIR (BLP/bench/results/reference/out/<deck>) as raw Float64 files
# `<leg>.ref-<cpu|gpuN>.<field>.<n1>x<n2>.f64`; a file already there is compared bitwise
# instead (`same_as_saved`). BLP_SKIP_DONE=1 skips a configuration already recorded for the
# same src/ tree, deck, backend and thread or device count. Other variables: BLP_DEVICES
# (0), BLP_REPEAT (3), BLP_MC_NS (1e7), BLP_TEST_GRMHD, BLP_REF_GPUMONTY, BLP_BBH455,
# BLP_BBH122.

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REQ_BACKEND = lowercase(get(ENV, "BLP_BACKEND", "gpu"))
REQ_BACKEND in ("cpu", "gpu") || error("BLP_BACKEND must be cpu or gpu, got $REQ_BACKEND")
const GPU = REQ_BACKEND == "gpu"
const DECK = lowercase(get(ENV, "BLP_DECK", "reference"))
DECK in ("reference", "torus") || error("BLP_DECK must be reference or torus, got $DECK")
get!(ENV, "BLP_REPEAT", "3")
get!(ENV, "BLP_REF_GPUMONTY",
     normpath(joinpath(ROOT, "..", "Mahakala", "ref-code", "gpumonty", "table", "hotcross.dat")))
isfile(ENV["BLP_REF_GPUMONTY"]) || error("BLP_REF_GPUMONTY: no hot-cross table at $(ENV["BLP_REF_GPUMONTY"])")
ENV["BLP_BACKEND"] = "cuda"          # the harness's production decks (its `cpu` is a 24² smoke deck)
include(joinpath(ROOT, "bench", "bench_multigpu.jl"))
include(joinpath(@__DIR__, "common.jl"))
using BlackLightPlus.Constants: MSUN, GNEWT, CL

const NREP = parse(Int, ENV["BLP_REPEAT"])
const DEVS = envlist("BLP_DEVICES", "0")
const REF_CHUNKS = envlist("BLP_REF_CHUNKS", "131072")
const PHASES = GPU && get(ENV, "BLP_PHASES", "1") == "1"
const SKIP = get(ENV, "BLP_SKIP_DONE", "0") == "1"
const CMP = joinpath(ROOT, "BLP", "bench", "results", "reference")
const REF_OUT = get(ENV, "BLP_OUT", joinpath(CMP, "raw", "ref.jsonl"))
const REF_SAVE = get(ENV, "BLP_SAVE_DIR", joinpath(CMP, "out", DECK))
const TAG = "ref-" * (GPU ? "gpu$(length(DEVS))" : "cpu")
const SRC_TREE = try
    t = readchomp(`git -C $ROOT rev-parse --short HEAD:src`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? t : t * "-dirty"
catch
    "unknown"
end
const HOST = gethostname()
const CPU_MODEL = try
    strip(split(first(filter(l -> startswith(l, "model name"), readlines("/proc/cpuinfo"))), ':')[2])
catch
    "unknown"
end

# ---------------------------------------------------------------- decks
# The snapshot alone (the torus deck and the CPU legs), or the harness's (resample, blocks).
const BLOCKS = Ref{Any}(nothing)
blocks() = FLUID[] !== nothing ? FLUID[][2] :
    (BLOCKS[] === nothing && (BLOCKS[] = AthenaKSnapshot(SNAP)); BLOCKS[])

const TORUS_M = 6.2e9 * MSUN
const TORUS_L = GNEWT * TORUS_M / CL^2
torus_cfg(res) = RunConfig(bh_a = 0.9, m_bh_cgs = TORUS_M, mass_scale_cgs = 1.0e-21 * TORUS_L^3,
    frequency_hz = 230.0e9, max_steps = 2500,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                          theta_e_min = -1.0))

torus_img_leg(r) = Leg("img_$r", "rays/s", devs -> begin
    dg = upload_amr(blocks(), devs; upload_bounds = false); cfg = torus_cfg(r)
    (run = d -> render_gpu_amr(cfg, dg; devices = d), items = r^2, variant = "")
end)
# items = 0: the ray count comes from the result
torus_amr_leg() = Leg("amr_384", "rays/s", devs -> begin
    dg = upload_amr(blocks(), devs; upload_bounds = false); cfg = torus_cfg(384)
    (run = d -> render_amr(cfg, SNAP; block = 16, max_level = 2, polarized = true,
                           stokes = AMRStokes(cfg, dg; polarized = true, devices = d)),
     items = 0, variant = "streamed")
end)

const DEFAULT_LEGS = DECK == "torus" ? ["img_192", "img_384", "img_768", "amr_384"] :
    GPU ? ["fused_384", "fused_768", "fused_mixed_768", "fused_pol_256", "fused_pol_384",
           "amr_384", "amr_pol_384", "amr_driver_pol", "analytic_1024", "mc_thin", "mc_scatter"] :
          ["amr_384", "amr_pol_384", "amr_driver_pol", "analytic_1024", "mc_thin", "mc_scatter"]
const LEG_NAMES = let s = strip(get(ENV, "BLP_LEGS", ""))
    isempty(s) ? DEFAULT_LEGS : String.(strip.(split(s, ',')))
end

function gpu_legs()
    legs = Dict{String,Leg}()
    if DECK == "reference"
        for l in [imaging_legs(); mixed_leg(); driver_closure_leg(); analytic_leg(); mc_legs();
                  binary_legs()]
            legs[l.name] = l
        end
    else
        for r in (192, 384, 768)
            legs["img_$r"] = torus_img_leg(r)
        end
        legs["amr_384"] = torus_amr_leg()
    end
    return legs
end

# CPU legs: name => (unit, setup() -> call)
function cpu_legs()
    two_phase(cfg, pol) = () -> (snap = blocks(); () -> BlackLightPlus.render(cfg, snap; polarized = pol))
    legs = Dict{String,Tuple{String,Function}}()
    if DECK == "reference"
        legs["amr_384"] = ("rays/s", two_phase(cfg_img(384), false))
        legs["amr_pol_384"] = ("rays/s", two_phase(cfg_img(384), true))
        legs["amr_driver_pol"] = ("rays/s", () -> (cfg = cfg_img(384);
            () -> render_amr(cfg, SNAP; block = 8, max_level = 2, polarized = true)))
        legs["analytic_1024"] = ("rays/s", () -> (cfg = cfg_analytic(1024); disk = EquatorialDisk(0.94);
            () -> BlackLightPlus.render_analytic(cfg; source = disk)))
        legs["mc_thin"] = ("photons/s", () -> () -> BlackLightPlus.mc_spectrum(SphereModel(), MC_NS))
        legs["mc_scatter"] = ("photons/s", () -> (tbl = hotcross_table();
            () -> BlackLightPlus.mc_spectrum_scatter(ScatteringSphere(), tbl, MC_NS)))
    else
        for r in (192, 384, 768)
            legs["img_$r"] = ("rays/s", two_phase(torus_cfg(r), false))
        end
        legs["amr_384"] = ("rays/s", () -> (cfg = torus_cfg(384);
            () -> render_amr(cfg, SNAP; block = 16, max_level = 2, polarized = true)))
    end
    return legs
end

const SETUP_RUNS = ("amr_driver_pol", "amr_driver_pol_closure", "bbh_i90_amr_pol")
leg_res(name) = name in ("amr_driver_pol", "amr_driver_pol_closure") ? 384 :
    name == "bbh_i90_amr_pol" ? 256 :
    (m = match(r"_(\d+)$", name); m === nothing ? 0 : parse(Int, m[1]))
# legs of one program family share its compilation
program_kind(name) = replace(name, r"_\d+$" => "")

# ---------------------------------------------------------------- results
const ARR = AbstractMatrix{<:Real}
function ref_fields(r)
    r isa ARR && return Pair{String,Any}["I" => r]
    if r isa NamedTuple
        haskey(r, :Ib) && return vcat(Pair{String,Any}["I" => r.I],
                                      Pair{String,Any}["I$(b - 1)" => r.Ib[b] for b in eachindex(r.Ib)])
        out = Pair{String,Any}[String(k) => getfield(r, k) for k in (:I, :Q, :U, :V) if haskey(r, k)]
        haskey(r, :levels) && push!(out, "levels" => Float64.(r.levels))
        return out
    end
    spec(tag, s) = Pair{String,Any}[tag * "count" => Float64.(s.count), tag * "sumw" => s.sumw,
                                    tag * "sumwE" => s.sumwE]
    hasproperty(r, :orders) &&
        return vcat(spec("total_", r.total), [spec("o$(k - 1)_", o) for (k, o) in enumerate(r.orders)]...)
    hasproperty(r, :sumw) && return spec("", r)
    return Pair{String,Any}[]
end

# Items marched and leg facts, counted as bench.jl counts them.
function ref_facts(r, res)
    if r isa ARR
        return res^2, Dict{String,Any}("flux_I" => sum(filter(isfinite, r)))
    elseif r isa NamedTuple && haskey(r, :nrays)
        return r.nrays, Dict{String,Any}("nrays" => r.nrays, "flux_I" => sum(r.I),
                                         "effective" => size(r.I, 1),
                                         "levels_hist" => [count(==(l), r.levels) for l in 0:maximum(r.levels)])
    elseif r isa NamedTuple
        return res^2, Dict{String,Any}("flux_I" => sum(filter(isfinite, r.I)))
    elseif hasproperty(r, :orders)
        return r.n_primary + r.n_children,
               Dict{String,Any}("n_primary" => r.n_primary, "n_children" => r.n_children,
                                "recorded" => sum(r.total.count))
    else
        n = sum(r.count)
        return n, Dict{String,Any}("n_primary" => n, "n_children" => 0, "recorded" => n)
    end
end

function device_state()
    GPU || return Dict{String,Any}()
    used = Float64[]; stack = Int[]
    for d in DEVS
        CUDA.device!(d) do
            free, total = CUDA.memory_info()
            push!(used, (total - free) / 1e9)
            push!(stack, Int(CUDA.limit(CUDA.LIMIT_STACK_SIZE)))
        end
    end
    return Dict{String,Any}("dev_used_gb" => used, "stack_bytes" => stack,
                            "gpu_name" => CUDA.device!(() -> CUDA.name(CUDA.device()), DEVS[1]))
end

# ---------------------------------------------------------------- driver
base_record(kind, name, unit) = Dict{String,Any}("kind" => kind, "package" => "reference",
    "deck" => DECK, "leg" => name, "key" => name, "res" => leg_res(name), "unit" => unit,
    "backend" => GPU ? "gpu" : "cpu", "threads" => Threads.nthreads(),
    "ndev" => GPU ? length(DEVS) : 0, "devices" => GPU ? DEVS : Int[], "commit" => COMMIT,
    "src_tree" => SRC_TREE, "job" => JOB, "host" => HOST, "cpu_model" => CPU_MODEL,
    "julia" => string(VERSION), "date" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())))

done(recs, name, config) = any(recs) do r
    get(r, "kind", "") == "bench" && get(r, "package", "") == "reference" &&
        get(r, "deck", "") == DECK && r["key"] == name && r["config"] == config &&
        r["backend"] == (GPU ? "gpu" : "cpu") && get(r, "src_tree", "") == SRC_TREE &&
        (GPU ? r["ndev"] == length(DEVS) : r["threads"] == Threads.nthreads())
end

timed(f) = (t = time_ns(); r = f(); ((time_ns() - t) / 1e9, r))
const KINDS_RUN = Set{String}()

# One configuration: the first call, then NREP timed calls.
function measure(call)
    l0 = load1(); smi = GPU ? smi_state() : Dict{String,Any}()
    first_s, out = timed(call)
    walls = Float64[]
    s0 = cpu_sample()
    for _ in 1:NREP
        t, out = timed(call)
        push!(walls, t)
    end
    s1 = cpu_sample(); l1 = load1()
    return (; first_s, walls, out, l0, l1, oc = other_cores(s0, s1), smi)
end

function record!(name, unit, config, lanes, chunk, m, res, setup_s, compiled; extra...)
    items, facts = ref_facts(m.out, res)
    rec = merge(base_record("bench", name, unit), facts, device_state(), m.smi,
        Dict{String,Any}("config" => config, "lanes" => lanes, "chunk" => chunk, "items" => items,
            "wall_s" => minimum(m.walls), "walls" => m.walls, "first_s" => m.first_s,
            "first_compiles" => !compiled, "setup_s" => setup_s,
            "setup_runs" => GPU && name in SETUP_RUNS, "rate" => items / minimum(m.walls),
            "nrep" => NREP, "load1_start" => m.l0, "load1_end" => m.l1, "other_cores" => m.oc),
        Dict{String,Any}(String(k) => v for (k, v) in extra))
    rec["same_as_saved"] = save_or_compare(REF_SAVE, name, TAG, ref_fields(m.out))
    append_record(REF_OUT, rec)
    @printf("BENCH package=reference deck=%s leg=%s backend=%s threads=%d ndev=%d config=%s lanes=%d chunk=%d items=%d wall_s=%.4g first_s=%.4g setup_s=%.4g rate=%.4g unit=%s load1=%.1f/%.1f other_cores=%.1f same_as_saved=%s\n",
            DECK, name, rec["backend"], rec["threads"], rec["ndev"], config, lanes, chunk, items,
            rec["wall_s"], m.first_s, setup_s, rec["rate"], unit, m.l0, m.l1, m.oc,
            string(rec["same_as_saved"]))
    flush(stdout)
    return rec
end

# The configurations of a GPU leg: (config, make() -> devices argument, run).
function gpu_configs(leg, s)
    mc = startswith(leg.name, "mc_")
    cs = Any[(config = "default", make = () -> DEVS, run = s.run, mc_chunk = mc ? 2^20 : 0)]
    if mc
        tbl = leg.name == "mc_scatter" ? hotcross_table() : nothing
        for c in unique([65536; REF_CHUNKS])
            run = leg.name == "mc_thin" ?
                (d -> mc_spectrum_gpu(SphereModel(), MC_NS; devices = d, chunk = c)) :
                (d -> mc_spectrum_gpu(ScatteringSphere(), tbl, MC_NS; devices = d, chunk = c))
            push!(cs, (config = "chunk$c", make = () -> DEVS, run = run, mc_chunk = c))
        end
    else
        for c in REF_CHUNKS
            push!(cs, (config = "chunk$c", make = () -> DevicePool(DEVS; lanes = 2, chunk = c),
                       run = s.run, mc_chunk = 0))
        end
    end
    return cs
end

function run_gpu_leg(leg::Leg)
    recs = SKIP ? read_records(REF_OUT) : Dict{String,Any}[]
    names = ["default"; (startswith(leg.name, "mc_") ? ["chunk$c" for c in unique([65536; REF_CHUNKS])] :
                         ["chunk$c" for c in REF_CHUNKS])]
    if SKIP && all(c -> done(recs, leg.name, c), names)
        println("# skip $(leg.name): recorded"); return
    end
    setup_s, s = timed(() -> leg.setup(DEVS))
    res = leg_res(leg.name)
    for c in gpu_configs(leg, s)
        SKIP && done(recs, leg.name, c.config) && continue
        arg = c.make()
        p = device_pool(arg)
        lanes = length(p.lanes) ÷ length(p.devices)
        chunk = c.mc_chunk > 0 ? c.mc_chunk : p.chunk
        compiled = program_kind(leg.name) in KINDS_RUN || leg.name in SETUP_RUNS
        m = measure(() -> c.run(arg))
        push!(KINDS_RUN, program_kind(leg.name))
        record!(leg.name, leg.unit, c.config, lanes, chunk, m, res, setup_s, compiled;
                variant = s.variant, pool_chunk = p.chunk)
        if PHASES && c.config == "default"
            ph = try
                cuda_phases(() -> c.run(arg))
            catch err
                Dict{String,Any}("error" => sprint(showerror, err))
            end
            append_record(REF_OUT, merge(base_record("phase", leg.name, leg.unit), ph,
                Dict{String,Any}("config" => c.config, "lanes" => lanes, "chunk" => chunk)))
        end
        arg isa DevicePool && close(arg)
    end
    s = nothing
    GC.gc(true); CUDA.reclaim()
end

function run_cpu_leg(name, unit, setup)
    if SKIP && done(read_records(REF_OUT), name, "threads")
        println("# skip $name: recorded"); return
    end
    setup_s, call = timed(setup)
    compiled = program_kind(name) in KINDS_RUN
    m = measure(call)
    push!(KINDS_RUN, program_kind(name))
    record!(name, unit, "threads", 0, 0, m, leg_res(name), setup_s, compiled)
    GC.gc(true)
end

function main_reference()
    println("# reference.jl deck=$DECK backend=$(GPU ? "gpu" : "cpu") commit=$COMMIT src_tree=$SRC_TREE ",
            "threads=$(Threads.nthreads()) ", GPU ? "devices=$DEVS " : "", "job=$JOB ",
            "legs=$(join(LEG_NAMES, ',')) load1=$(load1()) out=$REF_OUT")
    if GPU
        CUDA.functional() || error("BLP_BACKEND=gpu: CUDA is not functional here")
        length(CUDA.devices()) >= maximum(DEVS) + 1 ||
            error("BLP_DEVICES=$DEVS but $(length(CUDA.devices())) device(s) visible")
        legs = gpu_legs()
        for n in LEG_NAMES
            haskey(legs, n) || error("unknown GPU leg $n of deck $DECK")
            run_gpu_leg(legs[n])
        end
    else
        legs = cpu_legs()
        for n in LEG_NAMES
            haskey(legs, n) || error("unknown CPU leg $n of deck $DECK")
            run_cpu_leg(n, legs[n]...)
        end
    end
    return 0
end

exit(main_reference())
