# bench.jl — wall time and throughput of the BLP products on CPU threads or CUDA devices.
#
# Decks (BLP_DECK): `torus` (the default) and `reference`, the decks of the reference
# harness bench/bench_multigpu.jl, timed against the reference package by reference.jl.
#
# Torus legs (BLP_LEGS, comma list of legs or case keys such as img_384; default all):
#   img         render, StokesI, of the AthenaK torus at every BLP_RESES side (192,384,768)
#   pol         render, IQUV, at every BLP_POL_RESES side (256,384)
#   amr         render_amr(plan; block = 16, max_level = 2), IQUV, base BLP_AMR_RES (384)
#   analytic    render_analytic(plan, EquatorialDisk(0.94); bands = (0, 1, 2)) at BLP_AN_RES
#               (1024): a = 0.94, inclination 17°, distance 1e4, fov 20
#   mc_thin     mc_spectrum(prepare(UniformSphere()), BLP_MC_NS (1e7))
#   mc_scatter  mc_spectrum(prepare(UniformSphere(n_e = 1e18, theta_e = 3, B = 100,
#               R = 3e5)), Ns; scattering = true): Thomson depth n_e σ_T R ≈ 0.2
#   mc_scatter0 mc_spectrum(prepare(UniformSphere()), Ns; scattering = true): Thomson
#               depth 7e-12, no children; the scattering transport on the thin sphere
# Torus deck (img, pol, amr): `BLP_TEST_GRMHD` (torus.mhd_w_bcc.00999.athdf) on
# KerrSchildCartesian(0.9); M = 6.2e9 Msun, mass scale 1e-21 L³ (L = GM/c²), 230 GHz,
# inclination 60°, distance 1000, fov 30, C = 80, 2500 steps, escape 1.1 × distance;
# PlasmaConfig(rat_low = 1, rat_high = 10, gamma = 13/9, sigma_cut = 1, theta_e_min = -1).
# The matter is the snapshot's blocks, or with BLP_MATTER=uniform its resample
# UniformGrid(snapshot, BLP_UNIFORM_HALF (40), BLP_UNIFORM_N (256)) (case keys `<leg>_<res>_uniform`).
# The scattering legs pass one hot-cross table, built before the timing.
#
# Reference legs (BLP_DECK=reference; BLP_LEGS names them; default all), the BLP calls of
# the legs of bench/bench_multigpu.jl: M = 6.2e9 × 1.989e33 g, mass scale 1e24 g, 230 GHz,
# inclination 60°, distance 1000, fov 40, KerrSchildCartesian(0.9), RunConfig defaults
# (C = 80, 2500 steps, escape 1.1 × distance), PlasmaConfig(rat_high = 40, rat_low = 1,
# gamma = 13/9, use_p = false, sigma_cut = 100, theta_e_min = 0.3):
#   fused_384, fused_768   render, StokesI, matter UniformGrid(snapshot, 40, 256)
#   fused_mixed_768        the same with lanes = MixedLanes()
#   fused_pol_256, fused_pol_384   render, IQUV, on the same resample
#   amr_384, amr_pol_384   render, StokesI and IQUV, on the snapshot's blocks
#   amr_driver_pol         render_amr(plan; block = 8, max_level = 2), IQUV, blocks, base 384
#   analytic_1024          the torus deck's `analytic` leg
#   mc_thin                the torus deck's `mc_thin` leg (the defaults of SphereModel()
#                          and of mc_spectrum_gpu)
#   mc_scatter             mc_spectrum(prepare(UniformSphere(n_e = 1e18, theta_e = 3,
#                          B = 100, R = 3e5)), Ns; scattering = true, table, seed = 0x5ca77e5,
#                          nu_max = 1e13, spec_nu_max = 1e17, n_ebins = 140): ScatteringSphere()
#                          and the defaults of the scattering mc_spectrum_gpu; the table is
#                          load_hotcross(BLP_REF_GPUMONTY), the gpumonty hotcross.dat (by
#                          default that of the Mahakala checkout beside this repository)
#   bbh455_480, bbh455_pol_480, bbh122_pol_480 (named only)   render, I or IQUV, 480², on
#                          SuperposedKerrSchild at the dump's phase, matter the dump
#                          (BLP_BBH455 .bin, BLP_BBH122 .athdf, gamma 13/9): the EHT deck of
#                          figures/fig_eht_bbh.jl (i = 17°, fov 56, 4000 steps, R_high 40,
#                          mu 0.5, sigma_cut 1, the calibrated mass scale)
#   bbh_i90_amr_pol (named only)   render_amr(plan; block = 8, max_level = 4, refine =
#                          AbsoluteGradient(5.282132294e-13; count = 5)), IQUV, base 256,
#                          i = 90°, fov 40, the deck of figures/fig_bbh_i90_pol_amr.jl
#
# Comparison runs (BLP_DECK=reference, or BLP_COMPARE=1 on the torus deck): records go to
# BLP/bench/results/reference/raw/blp.jsonl and arrays to
# BLP/bench/results/reference/out/<deck> with the tags `blp-cpu`, `blp-gpuN`; each case runs on every GPU pool of BLP_POOLS
# (lanes:chunk list, default 2:auto,2:65536,2:131072, each on a fresh pool; a chunk `auto`
# is the executor's own rule) or with every CPU chunk of BLP_CPU_CHUNKS (default
# auto,4096,64); BLP_PHASES=1 (the default there on a GPU) adds one profiled call of the first
# pool, written as a `phase` record (common.jl `cuda_phases`).
#
# Protocol per leg: `prepare` (not in `first_s`; recorded as `setup_s`); one first call
# (timed as `first_s`: the JIT compilation when the leg's program has not run in this
# process, and on a GPU the uploads to the fresh pool); then BLP_REPEAT (3) timed calls;
# `wall_s` is the best.
# Throughput: rays (pixels; the ray count of render_amr) or superphotons (primaries plus
# children) per wall second. Every record carries the executor (`describe` of the result's
# `run`), threads, devices, lanes, chunk, the 1-minute load before and after the timed
# calls, `other_cores` (the cores that other processes kept busy during the timed calls,
# from /proc/stat less this process's CPU time), on a GPU the nvidia-smi state, the
# commit, the tree hashes of BLP/src and BLP/ext, and the Slurm job.
#
# Backends: BLP_BACKEND=cpu (default; `ThreadedExecutor(chunk = BLP_CPU_CHUNK)`, `auto`, the
# executor's rule, unless set, under `julia -t N`: one worker per thread pulls the
# chunks, so a chunk count below the thread count idles threads) or gpu (a DevicePool
# over the ordinals in BLP_DEVICES, default 0; needs CUDA in the environment). On a GPU
# the pool's lanes and chunk are those of BLP_POOLS when set, else BLP_LANES and
# BLP_CHUNK when both are set, else the best warm configuration of the `tune` records of
# the same leg and device count (in BLP_TUNE_FROM, default BLP_OUT), else 2 and `auto`.
# BLP_TUNE=1 runs the sweep instead:
# lanes 1–4 × chunk 2^14–2^17, each on a fresh pool (one first call, one warm call),
# appending one `tune` record per configuration;
# BLP_TUNE_LANES and BLP_TUNE_CHUNKS (comma lists) change the grid and BLP_TUNE_KIND the
# record kind (records of another kind do not choose the pool of the timed legs).
#
# Output: one JSON object per line appended to BLP_OUT (kind "bench" or "tune"; a
# preempted run loses only the leg in flight), one BENCH or TUNE line per record on
# stdout. BLP_SKIP_DONE=1 skips a leg (or a tuning configuration, or a pool or CPU chunk
# of a comparison run) whose record for the same BLP/src and BLP/ext trees, deck, backend,
# thread or device count is already in BLP_OUT.
# Each leg's output arrays go to BLP_SAVE_DIR (BLP/bench/results/out) as raw Float64
# files `<leg>_<res>.<cpu|gpuN>.<field>.<n1>x<n2>.f64`; a file already there is compared
# bitwise instead of rewritten (`same_as_saved`).
#
# Merge: `julia --project=BLP BLP/bench/bench.jl merge [BLP_DATE]` reads
# BLP/bench/results/raw/*.jsonl, adds to each GPU record its agreement with the CPU output
# of the same leg (bitwise; else the largest difference over the peak, the largest
# per-pixel relative difference above 1e-3 of the peak, and the flux ratio; Monte Carlo
# bins bitwise), marks the latest record per leg, resolution, backend and thread or device
# count as `final`, and writes BLP/bench/results/anta_<date>.json and .md.
# `bench.jl merge-reference [date]` reads BLP/bench/results/reference/raw/*.jsonl (the
# records of both packages) and logs/runner.log, pairs each reference leg with the BLP
# leg of the same deck, backend and device or thread count, compares the saved outputs of
# the two packages on the same hardware, and writes
# BLP/bench/results/anta_<date>_reference.json and .md.
#
# Usage, from the repository root:
#   BLP_OUT=BLP/bench/results/raw/cpu16.jsonl julia -t 16 --project=BLP BLP/bench/bench.jl
#   BLP_BACKEND=gpu BLP_DEVICES=0 BLP_OUT=BLP/bench/results/raw/gpu.jsonl \
#       julia -t 4 --project=results/blp/gpu/env BLP/bench/bench.jl
#   BLP_DECK=reference BLP_BACKEND=gpu julia -t 4 --project=results/blp/gpu/env BLP/bench/bench.jl
#   BLP/bench/slurm/run_gpu.sh             (the GPU legs through Slurm, see README.md)
#   BLP/bench/slurm/run_reference.sh       (both packages through Slurm, see README.md)
#   julia --project=BLP BLP/bench/bench.jl merge
#   julia --project=BLP BLP/bench/bench.jl merge-reference
# Other variables: BLP_REPEAT (3), BLP_CPU_CHUNK (64), BLP_RESES, BLP_POL_RESES,
# BLP_AMR_RES, BLP_AN_RES, BLP_MC_NS, BLP_TEST_GRMHD, BLP_REF_GPUMONTY.

using Printf
using BLP
using BLP.Numerics: GNEWT, CL, MSUN
using BLP.Execution: describe, DevicePool, CUDADevice
using BLP.Data: build_hotcross_table, load_hotcross, athenak_bin_snapshot
using BLP.Analysis: refinement_indicator

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MODE = isempty(ARGS) ? "bench" : ARGS[1]
const BACKEND = lowercase(get(ENV, "BLP_BACKEND", "cpu"))
BACKEND in ("cpu", "gpu") || error("BLP_BACKEND must be cpu or gpu, got $BACKEND")
const GPU = MODE == "bench" && BACKEND == "gpu"
if GPU
    using CUDA
end
include(joinpath(@__DIR__, "common.jl"))

const DECK = lowercase(get(ENV, "BLP_DECK", "torus"))
DECK in ("torus", "reference") || error("BLP_DECK must be torus or reference, got $DECK")
const COMPARE = DECK == "reference" || get(ENV, "BLP_COMPARE", "0") == "1"
const LEGS_TORUS = ["img", "pol", "amr", "analytic", "mc_thin", "mc_scatter", "mc_scatter0"]
const LEGS_REFERENCE = ["fused_384", "fused_768", "fused_mixed_768", "fused_pol_256",
    "fused_pol_384", "amr_384", "amr_pol_384", "amr_driver_pol", "analytic_1024", "mc_thin",
    "mc_scatter"]
const LEGS_BINARY = ["bbh455_480", "bbh455_pol_480", "bbh122_pol_480", "bbh_i90_amr_pol"]
const LEGS_ALL = DECK == "torus" ? LEGS_TORUS : LEGS_REFERENCE
const LEGS = let s = strip(get(ENV, "BLP_LEGS", ""))
    isempty(s) ? LEGS_ALL : String.(strip.(split(s, ',')))
end
const RESES = envlist("BLP_RESES", "192,384,768")
const POL_RESES = envlist("BLP_POL_RESES", "256,384")
const AMR_RES = parse(Int, get(ENV, "BLP_AMR_RES", "384"))
const AN_RES = parse(Int, get(ENV, "BLP_AN_RES", "1024"))
const MC_NS = parse(Float64, get(ENV, "BLP_MC_NS", "1e7"))
const REPEAT = parse(Int, get(ENV, "BLP_REPEAT", "3"))
# A chunk setting of the environment: an item count, or `auto` (the executor's own rule).
chunk_arg(s) = strip(s) == "auto" ? :auto : parse(Int, s)
const CPU_CHUNK = chunk_arg(get(ENV, "BLP_CPU_CHUNK", "auto"))
const DEVICES = envlist("BLP_DEVICES", "0")
const TUNE = get(ENV, "BLP_TUNE", "0") == "1"
const SKIP_DONE = get(ENV, "BLP_SKIP_DONE", "0") == "1"
const RESULTS = joinpath(ROOT, "BLP", "bench", "results")
const CMP_DIR = joinpath(RESULTS, "reference")
const OUT = get(ENV, "BLP_OUT", COMPARE ? joinpath(CMP_DIR, "raw", "blp.jsonl") :
                                          joinpath(RESULTS, "raw", "records.jsonl"))
const TUNE_FROM = get(ENV, "BLP_TUNE_FROM", OUT)
const SAVE_DIR = get(ENV, "BLP_SAVE_DIR", COMPARE ? joinpath(CMP_DIR, "out", DECK) :
                                                    joinpath(RESULTS, "out"))
const SNAP = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const REF_GPUMONTY = get(ENV, "BLP_REF_GPUMONTY",
    normpath(joinpath(ROOT, "..", "Mahakala", "ref-code", "gpumonty", "table", "hotcross.dat")))
const BBH455 = get(ENV, "BLP_BBH455", "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/" *
                   "SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const BBH122 = get(ENV, "BLP_BBH122", "/data2/haiyangw/aurora/cbd_rel/torus.mhd_w_bcc.00122.athdf")
const MATTER = lowercase(get(ENV, "BLP_MATTER", "blocks"))
MATTER in ("blocks", "uniform") || error("BLP_MATTER must be blocks or uniform, got $MATTER")
const UNIFORM_HALF = parse(Float64, get(ENV, "BLP_UNIFORM_HALF", "40"))
const UNIFORM_N = parse(Int, get(ENV, "BLP_UNIFORM_N", "256"))
parse_pools(s) = [(lanes = parse(Int, a), chunk = chunk_arg(b))
                  for (a, b) in (split(strip(x), ':') for x in split(s, ',') if !isempty(strip(x)))]
const POOLS = haskey(ENV, "BLP_POOLS") ? parse_pools(ENV["BLP_POOLS"]) :
    COMPARE ? [(lanes = 2, chunk = :auto), (lanes = 2, chunk = 65536), (lanes = 2, chunk = 131072)] :
    nothing
const CPU_CHUNKS = haskey(ENV, "BLP_CPU_CHUNKS") ?
    [chunk_arg(x) for x in split(ENV["BLP_CPU_CHUNKS"], ',') if !isempty(strip(x))] :
    COMPARE ? [:auto, 4096, 64] : [CPU_CHUNK]
const PHASES = GPU && get(ENV, "BLP_PHASES", COMPARE ? "1" : "0") == "1"
const JOB = get(ENV, "SLURM_JOB_ID", "none")
const COMMIT = try
    sha = readchomp(`git -C $ROOT rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`git -C $ROOT status --porcelain -- BLP/src BLP/ext`))
    dirty ? sha * "-dirty" : sha
catch
    "unknown"
end
# The code under test: the tree hashes of BLP/src and BLP/ext at a commit ("-dirty" with
# uncommitted changes there). Records of commits with the same trees count as done.
tree_of(commit) = try
    join((readchomp(`git -C $ROOT rev-parse --short $(commit):BLP/$d`) for d in ("src", "ext")),
         '+')
catch
    "unknown"
end
const SRC_TREE = tree_of("HEAD") * (endswith(COMMIT, "-dirty") ? "-dirty" : "")
const TREES = Dict{String,String}()
same_code(r) = get!(() -> get(r, "src_tree", tree_of(r["commit"])), TREES,
                    r["commit"] * get(r, "src_tree", "")) == SRC_TREE
const HOST = gethostname()
const CPU_MODEL = try
    line = first(filter(l -> startswith(l, "model name"), readlines("/proc/cpuinfo")))
    strip(split(line, ':')[2])
catch
    "unknown"
end

append_record(rec::AbstractDict) = append_record(OUT, rec)

function device_state()
    GPU || return Dict{String,Any}()
    used = Float64[]; stack = Int[]
    for d in DEVICES
        CUDA.device!(d) do
            free, total = CUDA.memory_info()
            push!(used, (total - free) / 1e9)
            push!(stack, Int(CUDA.limit(CUDA.LIMIT_STACK_SIZE)))
        end
    end
    return Dict{String,Any}("dev_used_gb" => used, "stack_bytes" => stack,
                            "gpu_name" => CUDA.device!(() -> CUDA.name(CUDA.device()), DEVICES[1]))
end

# ---------------------------------------------------------------- decks
const M_BH = 6.2e9 * MSUN
const L_UNIT = GNEWT * M_BH / CL^2

torus_cfg(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = 1.0e-21 * L_UNIT^3,
    frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                          theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)

# The imaging deck of bench/bench_multigpu.jl (its cfg_img).
ref_cfg(res) = RunConfig(m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1.0e24,
    frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 40.0, resolution = res),
    plasma = PlasmaConfig(rat_high = 40.0, rat_low = 1.0, gamma = 13.0 / 9.0, use_p = false,
                          sigma_cut = 100.0, theta_e_min = 0.3))

const SNAPSHOT = Ref{Any}(nothing)
snapshot() = (SNAPSHOT[] === nothing && (SNAPSHOT[] = AthenaKSnapshot(SNAP)); SNAPSHOT[])

# The uniform resample of the snapshot (bench_multigpu.jl's resample_fluid_uniform(snap, 40, 256)).
const UNIFORM = Ref{Any}(nothing)
uniform() = (UNIFORM[] === nothing &&
             (UNIFORM[] = UniformGrid(snapshot(), UNIFORM_HALF, UNIFORM_N)); UNIFORM[])
torus_matter() = MATTER == "uniform" ? uniform() : snapshot()

const HOTCROSS = Ref{Any}(nothing)
hotcross() = (HOTCROSS[] === nothing && (HOTCROSS[] = build_hotcross_table()); HOTCROSS[])
const REF_HOTCROSS = Ref{Any}(nothing)
ref_hotcross() = (REF_HOTCROSS[] === nothing && (REF_HOTCROSS[] = load_hotcross(REF_GPUMONTY));
                  REF_HOTCROSS[])

const SCATTER_SPHERE = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)

# The binary decks of bench/bench_multigpu.jl: the EHT deck of figures/fig_eht_bbh.jl (each
# hole 6.5e9 Msun, the code mass the binary's, 230 GHz, i = 17°, fov 56, 480², 4000 steps,
# R_high 40, mu 0.5, sigma_cut 1, the calibrated mass scale; the metric at the dump's
# orbital phase) and the i = 90° AMR deck of figures/fig_bbh_i90_pol_amr.jl. One dump in
# memory at a time; BLP locates on the snapshot's own index.
const BBH_SNAP = Ref{Any}(nothing)
function bbh_snapshot(path)
    if BBH_SNAP[] === nothing || BBH_SNAP[][1] != path
        BBH_SNAP[] = nothing; GC.gc()
        snap = endswith(path, ".bin") ? athenak_bin_snapshot(path; gamma = 13.0 / 9.0) :
               AthenaKSnapshot(path; gamma = 13.0 / 9.0)
        BBH_SNAP[] = (path, snap)
    end
    return BBH_SNAP[][2]
end
const BBH_DECKS = Dict(
    "s455" => (path = BBH455, a = [0.0, 0.0], ms = 2.274019497e28,
               c = [(12.1912068, -2.761245515, 0.0), (-12.1912068, 2.761245515, 0.0)]),
    "s122" => (path = BBH122, a = [0.4, 0.4], ms = 1.000498515e21,
               c = [(0.321593038, 7.993533506, 0.0), (-0.321593038, -7.993533506, 0.0)]))
bbh_cfg(ms, res) = RunConfig(m_bh_cgs = 2 * 6.5e9 * 1.98847e33, mass_scale_cgs = ms,
    frequency_hz = 230.0e9, max_steps = 4000,
    camera = CameraConfig(inclination_deg = 17.0, distance = 1000.0, fov = 56.0, resolution = res),
    plasma = PlasmaConfig(gamma = 13.0 / 9.0, rat_low = 1.0, rat_high = 40.0, mu = 0.5,
                          sigma_cut = 1.0))

# A case: its leg, resolution (0 for Monte Carlo), item unit, program kind (the first case
# of a kind in a process compiles it), `setup() -> (call(exec) -> result)`, and its key.
struct Case
    leg::String
    res::Int
    unit::String
    kind::String
    setup::Function
    name::String
end
Case(leg, res, unit, kind, setup) = Case(leg, res, unit, kind, setup,
    (res > 0 ? "$(leg)_$(res)" : leg) * (MATTER == "uniform" && leg in ("img", "pol", "amr") ? "_uniform" : ""))
key(c::Case) = c.name

torus_plan(res, mode) =
    prepare(torus_cfg(res); spacetime = KerrSchildCartesian(0.9), matter = torus_matter(), mode = mode)
ref_plan(res, matter, mode; kw...) =
    prepare(ref_cfg(res); spacetime = KerrSchildCartesian(0.9), matter = matter, mode = mode, kw...)

function analytic_case(name, res = AN_RES)
    Case(name, res, "rays/s", "analytic", () -> begin
        cfg = RunConfig(camera = CameraConfig(inclination_deg = 17.0, distance = 1.0e4,
                                              fov = 20.0, resolution = res))
        p = prepare(cfg; spacetime = KerrSchildCartesian(0.94), matter = nothing,
                    mode = StokesI())
        disk = EquatorialDisk(0.94)
        exec -> render_analytic(p, disk; bands = (0, 1, 2), backend = exec)
    end, name)
end

# A torus entry of BLP_LEGS: a leg (every resolution of its list) or a case key such as
# `img_384`, `amr_384` or `analytic_1024` (that leg at that resolution).
function torus_entry(e)
    m = match(r"^(img|pol|amr|analytic)_(\d+)$", e)
    m === nothing ? (e, nothing) : (String(m[1]), parse(Int, m[2]))
end

function cases()
    DECK == "reference" && return reference_cases()
    cs = Case[]
    for e in LEGS
        leg, rsel = torus_entry(e)
        if leg == "img"
            for r in (rsel === nothing ? RESES : [rsel])
                push!(cs, Case("img", r, "rays/s", "img",
                               () -> (p = torus_plan(r, StokesI()); exec -> render(p; backend = exec))))
            end
        elseif leg == "pol"
            for r in (rsel === nothing ? POL_RESES : [rsel])
                push!(cs, Case("pol", r, "rays/s", "pol",
                               () -> (p = torus_plan(r, IQUV()); exec -> render(p; backend = exec))))
            end
        elseif leg == "amr"
            ra = rsel === nothing ? AMR_RES : rsel
            push!(cs, Case("amr", ra, "rays/s", "amr", () -> begin
                p = torus_plan(ra, IQUV())
                exec -> render_amr(p; block = 16, max_level = 2, backend = exec)
            end))
        elseif leg == "analytic"
            c = analytic_case("analytic", rsel === nothing ? AN_RES : rsel)
            push!(cs, Case(c.leg, c.res, c.unit, c.kind, c.setup))
        elseif leg == "mc_thin"
            push!(cs, Case("mc_thin", 0, "photons/s", "mc_thin", () -> begin
                p = prepare(UniformSphere())
                exec -> mc_spectrum(p, MC_NS; backend = exec)
            end))
        elseif leg == "mc_scatter"
            push!(cs, Case("mc_scatter", 0, "photons/s", "mc_scatter", () -> begin
                p = prepare(SCATTER_SPHERE); tbl = hotcross()
                exec -> mc_spectrum(p, MC_NS; scattering = true, table = tbl, backend = exec)
            end))
        elseif leg == "mc_scatter0"
            push!(cs, Case("mc_scatter0", 0, "photons/s", "mc_scatter", () -> begin
                p = prepare(UniformSphere()); tbl = hotcross()
                exec -> mc_spectrum(p, MC_NS; scattering = true, table = tbl, backend = exec)
            end))
        else
            error("unknown leg $e; expected one of $(join(LEGS_ALL, ", ")) or a case key")
        end
    end
    return cs
end

# The legs of bench/bench_multigpu.jl as BLP calls; the case key is the harness's leg name.
function reference_cases()
    img(name, res, matter, mode, kind; kw...) = Case(name, res, "rays/s", kind,
        () -> (p = ref_plan(res, matter(), mode; kw...); exec -> render(p; backend = exec)), name)
    all = Dict(
        "fused_384" => img("fused_384", 384, uniform, StokesI(), "fused"),
        "fused_768" => img("fused_768", 768, uniform, StokesI(), "fused"),
        "fused_mixed_768" => img("fused_mixed_768", 768, uniform, StokesI(), "fused_mixed";
                                 lanes = MixedLanes()),
        "fused_pol_256" => img("fused_pol_256", 256, uniform, IQUV(), "fused_pol"),
        "fused_pol_384" => img("fused_pol_384", 384, uniform, IQUV(), "fused_pol"),
        "amr_384" => img("amr_384", 384, snapshot, StokesI(), "amr"),
        "amr_pol_384" => img("amr_pol_384", 384, snapshot, IQUV(), "amr_pol"),
        "amr_driver_pol" => Case("amr_driver_pol", 384, "rays/s", "amr_driver", () -> begin
            p = ref_plan(384, snapshot(), IQUV())
            exec -> render_amr(p; block = 8, max_level = 2, backend = exec)
        end, "amr_driver_pol"),
        "analytic_1024" => analytic_case("analytic_1024"),
        "mc_thin" => Case("mc_thin", 0, "photons/s", "mc_thin", () -> begin
            p = prepare(UniformSphere())
            exec -> mc_spectrum(p, MC_NS; backend = exec)
        end, "mc_thin"),
        "mc_scatter" => Case("mc_scatter", 0, "photons/s", "mc_scatter", () -> begin
            p = prepare(SCATTER_SPHERE); tbl = ref_hotcross()
            exec -> mc_spectrum(p, MC_NS; scattering = true, table = tbl, seed = 0x5ca77e5,
                                nu_max = 1.0e13, spec_nu_max = 1.0e17, n_ebins = 140,
                                backend = exec)
        end, "mc_scatter"))
    bbh(name, tag, mode) = Case(name, 480, "rays/s", mode isa IQUV ? "bbh_pol" : "bbh", () -> begin
        d = BBH_DECKS[tag]
        p = prepare(bbh_cfg(d.ms, 480); spacetime = SuperposedKerrSchild([0.5, 0.5], d.a, d.c),
                    matter = bbh_snapshot(d.path), mode = mode)
        exec -> render(p; backend = exec)
    end, name)
    all["bbh455_480"] = bbh("bbh455_480", "s455", StokesI())
    all["bbh455_pol_480"] = bbh("bbh455_pol_480", "s455", IQUV())
    all["bbh122_pol_480"] = bbh("bbh122_pol_480", "s122", IQUV())
    all["bbh_i90_amr_pol"] = Case("bbh_i90_amr_pol", 256, "rays/s", "bbh_i90", () -> begin
        φ0 = -0.2233022; c1 = 12.5 .* (cos(φ0), sin(φ0), 0.0)
        st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [c1, (-c1[1], -c1[2], 0.0)])
        cfg = RunConfig(m_bh_cgs = 6.5e9 * 1.989e33, mass_scale_cgs = 1.0e24,
                        frequency_hz = 2.3e11, max_steps = 2500,
                        camera = CameraConfig(inclination_deg = 90.0, distance = 1000.0,
                                              fov = 40.0, resolution = 256),
                        plasma = PlasmaConfig(gamma = 13.0 / 9.0))
        p = prepare(cfg; spacetime = st, matter = bbh_snapshot(BBH455), mode = IQUV())
        ind = refinement_indicator(abs_grad_cut = 5.282132294e-13, abs_grad_count = 5)
        exec -> render_amr(p; block = 8, max_level = 4, refine = ind, backend = exec)
    end, "bbh_i90_amr_pol")
    for l in LEGS
        haskey(all, l) || error("unknown reference leg $l; expected one of " *
                                join([LEGS_REFERENCE; LEGS_BINARY], ", "))
    end
    return [all[l] for l in LEGS]
end

# ---------------------------------------------------------------- results
run_of(r::Image) = r.run
run_of(r::NamedTuple) = haskey(r, :image) ? r.image.run : r.run
run_of(r) = r.run

# Items marched and leg-specific facts of a result.
function facts(c::Case, r)
    if r isa Image
        return c.res^2, Dict{String,Any}("flux_I" => sum(filter(isfinite, r.I)))
    elseif r isa NamedTuple && haskey(r, :nrays)
        return r.nrays, Dict{String,Any}("nrays" => r.nrays, "flux_I" => sum(r.image.I),
                                         "effective" => size(r.image.I, 1),
                                         "levels_hist" => [count(==(l), r.levels) for l in 0:maximum(r.levels)])
    elseif r isa NamedTuple
        return c.res^2, Dict{String,Any}("flux_I" => sum(filter(isfinite, r.I)))
    elseif r isa ScatterResult
        return r.n_primary + r.n_children,
               Dict{String,Any}("n_primary" => r.n_primary, "n_children" => r.n_children,
                                "recorded" => sum(r.total.count))
    else                                    # Spectrum: every primary is recorded once
        n = sum(r.count)
        return n, Dict{String,Any}("n_primary" => n, "n_children" => 0, "recorded" => n)
    end
end

# The arrays of a result that the GPU/CPU agreement compares.
function fields(r)
    r isa Image && return ["I" => r.I, "Q" => r.Q, "U" => r.U, "V" => r.V]
    r isa NamedTuple && haskey(r, :nrays) &&
        return ["I" => r.image.I, "Q" => r.image.Q, "U" => r.image.U, "V" => r.image.V,
                "levels" => Float64.(r.levels)]
    r isa NamedTuple &&
        return vcat(["I" => r.I], ["I$(b - 1)" => r.Ib[b] for b in eachindex(r.Ib)])
    spec(tag, s) = [tag * "count" => Float64.(s.count), tag * "sumw" => s.sumw,
                    tag * "sumwE" => s.sumwE]
    r isa ScatterResult &&
        return vcat(spec("total_", r.total),
                    [spec("o$(k - 1)_", o) for (k, o) in enumerate(r.orders)]...)
    return spec("", r)
end

backend_tag(ndev) = (COMPARE ? "blp-" : "") * (GPU ? "gpu$ndev" : "cpu")

# ---------------------------------------------------------------- driver
function base_record(kind, c::Case)
    return Dict{String,Any}("kind" => kind, "package" => "BLP", "deck" => DECK,
        "leg" => c.leg, "res" => c.res, "key" => key(c),
        "unit" => c.unit, "backend" => BACKEND, "threads" => Threads.nthreads(),
        "ndev" => GPU ? length(DEVICES) : 0, "devices" => GPU ? DEVICES : Int[],
        "commit" => COMMIT, "src_tree" => SRC_TREE, "job" => JOB, "host" => HOST,
        "cpu_model" => CPU_MODEL, "julia" => string(VERSION),
        "date" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())))
end

config_name(cfg) = GPU ? "lanes$(cfg.lanes)_chunk$(cfg.chunk)" : "chunk$(cfg.chunk)"

# Whether `recs` hold a `kind` record of case `c` for this code, deck, backend and device
# or thread count; with `cfg`, of that pool (explicit pools only) or CPU chunk.
done(recs, kind, c, cfg = nothing, extra = Dict()) = any(recs) do r
    get(r, "kind", "") == kind && r["key"] == key(c) && r["backend"] == BACKEND &&
        get(r, "deck", "torus") == DECK && same_code(r) &&
        (GPU ? r["ndev"] == length(DEVICES) &&
               (cfg === nothing || POOLS === nothing ||
                (r["lanes"] == cfg.lanes && string(r["chunk"]) == string(cfg.chunk))) :
               r["threads"] == Threads.nthreads() &&
               string(r["chunk"]) == string(cfg === nothing ? CPU_CHUNK : cfg.chunk)) &&
        all(get(r, k, nothing) == v for (k, v) in extra)
end

timed(f) = (t = time_ns(); r = f(); ((time_ns() - t) / 1e9, r))

const POOL_DEFAULT = (lanes = 2, chunk = :auto)
const TUNE_LANES = envlist("BLP_TUNE_LANES", "1,2,3,4")
const TUNE_CHUNKS = envlist("BLP_TUNE_CHUNKS", "16384,32768,65536,131072")
const TUNE_KIND = get(ENV, "BLP_TUNE_KIND", "tune")

gpu_devices() = [CUDADevice(d) for d in DEVICES]

function release(pool)
    close(pool)
    GC.gc(true)
    CUDA.reclaim()
end

# The pool configuration of a GPU case: the environment, else the best tune record.
function pool_config(c::Case)
    haskey(ENV, "BLP_LANES") && haskey(ENV, "BLP_CHUNK") &&
        return (lanes = parse(Int, ENV["BLP_LANES"]), chunk = chunk_arg(ENV["BLP_CHUNK"]),
                source = "env")
    tunes = filter(r -> get(r, "kind", "") == "tune" && r["key"] == key(c) &&
                        r["ndev"] == length(DEVICES) && same_code(r), read_records(TUNE_FROM))
    isempty(tunes) && return (POOL_DEFAULT..., source = "default")
    best = tunes[argmin([r["warm_s"] for r in tunes])]
    return (lanes = best["lanes"], chunk = best["chunk"], source = "tune")
end

# The executor configurations a case runs on, in order.
function configs(c::Case)
    if GPU
        POOLS === nothing && return [pool_config(c)]
        return [(lanes = p.lanes, chunk = p.chunk, source = "pools") for p in POOLS]
    end
    return [(lanes = 0, chunk = k, source = "none") for k in CPU_CHUNKS]
end

const KINDS_RUN = Set{String}()

function bench_case(c::Case)
    call = nothing; setup_s = NaN
    for (ci, cfg) in enumerate(configs(c))
        recs = SKIP_DONE ? read_records(OUT) : Dict{String,Any}[]
        if SKIP_DONE && done(recs, "bench", c, cfg)
            println("# skip $(key(c)) $(config_name(cfg)): recorded"); continue
        end
        call === nothing && ((setup_s, call) = timed(c.setup))
        pool = GPU ? DevicePool(gpu_devices(); lanes = cfg.lanes, chunk = cfg.chunk) : nothing
        exec = GPU ? PoolExecutor(pool) : ThreadedExecutor(chunk = cfg.chunk)
        compiled = c.kind in KINDS_RUN
        smi = GPU ? smi_state() : Dict{String,Any}()
        l0 = load1()
        first_s, r = timed(() -> call(exec))
        push!(KINDS_RUN, c.kind)
        walls = Float64[]
        s0 = cpu_sample()
        for _ in 1:REPEAT
            t, r = timed(() -> call(exec))
            push!(walls, t)
        end
        s1 = cpu_sample()
        l1 = load1()
        items, extra = facts(c, r)
        run = run_of(r)
        d = describe_record(run.executor)
        rec = merge(base_record("bench", c), d, extra, device_state(), smi, Dict{String,Any}(
            "items" => items, "wall_s" => minimum(walls), "walls" => walls, "first_s" => first_s,
            "first_compiles" => !compiled, "rate" => items / minimum(walls), "nrep" => REPEAT,
            "load1_start" => l0, "load1_end" => l1, "other_cores" => other_cores(s0, s1),
            "run_seconds" => run.seconds, "setup_s" => setup_s, "config" => config_name(cfg),
            "pool_source" => GPU ? cfg.source : "none"))
        DECK == "torus" && (rec["matter"] = MATTER)
        rec["same_as_saved"] = save_or_compare(SAVE_DIR, key(c), backend_tag(rec["ndev"]), fields(r))
        append_record(rec)
        if PHASES && ci == 1
            ph = try
                cuda_phases(() -> call(exec))
            catch err
                Dict{String,Any}("error" => sprint(showerror, err))
            end
            append_record(merge(base_record("phase", c), ph, Dict{String,Any}(
                "config" => config_name(cfg), "lanes" => cfg.lanes, "chunk" => cfg.chunk)))
        end
        GPU && release(pool)
        @printf("BENCH deck=%s leg=%s res=%d backend=%s threads=%d ndev=%d lanes=%d chunk=%d items=%d \
                 wall_s=%.4g first_s=%.4g rate=%.4g unit=%s load1=%.1f/%.1f other_cores=%.1f \
                 same_as_saved=%s\n",
                DECK, key(c), c.res, BACKEND, rec["threads"], rec["ndev"], rec["lanes"], rec["chunk"],
                items, rec["wall_s"], first_s, rec["rate"], c.unit, l0, l1, rec["other_cores"],
                string(rec["same_as_saved"]))
        flush(stdout)
    end
end

describe_record(e) = Dict{String,Any}("executor_kind" => string(e.kind),
    "executor_devices" => collect(String, e.devices), "lanes" => e.lanes, "chunk" => e.chunk,
    "executor_threads" => e.threads)

function tune_case(c::Case)
    call = c.setup()
    for lanes in TUNE_LANES, chunk in TUNE_CHUNKS
        recs = read_records(OUT)
        if SKIP_DONE && done(recs, TUNE_KIND, c, nothing, Dict("lanes" => lanes, "chunk" => chunk))
            continue
        end
        pool = DevicePool(gpu_devices(); lanes, chunk)
        exec = PoolExecutor(pool)
        compiled = c.kind in KINDS_RUN
        l0 = load1()
        first_s, _ = timed(() -> call(exec))
        push!(KINDS_RUN, c.kind)
        warm_s, r = timed(() -> call(exec))
        items, _ = facts(c, r)
        rec = merge(base_record(TUNE_KIND, c), device_state(), Dict{String,Any}(
            "lanes" => lanes, "chunk" => chunk, "items" => items, "first_s" => first_s,
            "first_compiles" => !compiled, "warm_s" => warm_s, "rate" => items / warm_s,
            "load1_start" => l0))
        append_record(rec)
        release(pool)
        @printf("TUNE kind=%s leg=%s res=%d ndev=%d lanes=%d chunk=%d first_s=%.4g \
                 warm_s=%.4g rate=%.4g\n",
                TUNE_KIND, c.leg, c.res, length(DEVICES), lanes, chunk, first_s, warm_s, items / warm_s)
        flush(stdout)
    end
end

function main_bench()
    println("# bench.jl deck=$DECK backend=$BACKEND commit=$COMMIT threads=$(Threads.nthreads()) ",
            GPU ? "devices=$(DEVICES) " : "", "job=$JOB tune=$TUNE legs=$(join(LEGS, ',')) ",
            "load1=$(load1()) out=$OUT")
    if GPU
        CUDA.functional() || error("BLP_BACKEND=gpu: CUDA is not functional here")
        length(CUDA.devices()) >= maximum(DEVICES) + 1 ||
            error("BLP_DEVICES=$(DEVICES) but $(length(CUDA.devices())) device(s) visible")
    end
    TUNE && !GPU && error("BLP_TUNE=1 needs BLP_BACKEND=gpu")
    for c in cases()
        TUNE ? tune_case(c) : bench_case(c)
    end
    return 0
end

# ---------------------------------------------------------------- merge
function load_saved(dir, k, tag)
    out = Dict{String,Vector{Float64}}()
    pre = "$k.$tag."
    for f in readdir(dir)
        startswith(f, pre) && endswith(f, ".f64") || continue
        name = split(f[length(pre)+1:end], '.')[1]
        out[name] = collect(reinterpret(Float64, read(joinpath(dir, f))))
    end
    return out
end

bitwise(a, b) = length(a) == length(b) && reinterpret(UInt64, a) == reinterpret(UInt64, b)

# Agreement of an output `g` with the baseline output `c` of the same case.
function agreement(leg, g, c)
    isempty(g) && return Dict{String,Any}("agreement" => "no GPU arrays")
    isempty(c) && return Dict{String,Any}("agreement" => "no CPU arrays")
    names = sort!(collect(intersect(keys(g), keys(c))))
    a = Dict{String,Any}("bitwise" => all(bitwise(g[n], c[n]) for n in names),
                         "fields" => names)
    if startswith(leg, "mc")
        relmax(n) = (m = c[n] .!= 0;
                     any(m) ? maximum(abs.(g[n][m] .- c[n][m]) ./ abs.(c[n][m])) : 0.0)
        cnt = filter(n -> endswith(n, "count"), names)
        a["counts_bitwise"] = all(bitwise(g[n], c[n]) for n in cnt)
        for q in ("sumw", "sumwE")
            ns = filter(n -> endswith(n, q) && !(q == "sumw" && endswith(n, "sumwE")), names)
            a["max_rel_$q"] = maximum(relmax, ns)
            same(n) = count(reinterpret(UInt64, g[n]) .== reinterpret(UInt64, c[n]))
            a["bins_bitwise_$q"] = sum(same, ns) / sum(n -> length(c[n]), ns)
        end
        return a
    end
    gI = g["I"]; cI = c["I"]
    length(gI) == length(cI) || return merge(a, Dict{String,Any}("agreement" => "size differs"))
    ok = isfinite.(gI) .& isfinite.(cI)
    peak = maximum(abs, cI[ok])
    a["flux_ratio"] = sum(gI[ok]) / sum(cI[ok])
    a["max_diff_over_peak"] = maximum(abs.(gI[ok] .- cI[ok])) / peak
    big = ok .& (abs.(cI) .> 1e-3 * peak)
    a["max_rel_above_1e-3_peak"] = maximum(abs.(gI[big] .- cI[big]) ./ abs.(cI[big]))
    a["frac_pixels_bitwise"] = count(reinterpret(UInt64, gI) .== reinterpret(UInt64, cI)) / length(cI)
    a["nan_pixels"] = count(!, ok)
    for n in ("Q", "U", "V")
        haskey(g, n) && haskey(c, n) && any(!iszero, c[n]) &&
            (a["max_diff_over_peak_$n"] = maximum(abs.(g[n][ok] .- c[n][ok])) / peak)
    end
    if haskey(g, "levels") && haskey(c, "levels")
        a["levels_equal"] = g["levels"] == c["levels"]
    end
    return a
end

fmt(x; d = 3) = x === nothing ? "–" : x isa AbstractFloat ? (abs(x) >= 1e4 || abs(x) < 1e-2 ?
    @sprintf("%.*e", d - 1, x) : abs(x) >= 100 ? @sprintf("%.0f", x) :
    string(round(x; sigdigits = d))) : string(x)

ratekey(r) = (r["key"], r["backend"],
              r["backend"] == "cpu" ? (r["threads"], r["chunk"]) : r["ndev"])

const LEG_LABEL = Dict("img" => "unpolarized render (I)", "pol" => "polarized render (IQUV)",
    "amr" => "adaptive polarized image", "analytic" => "exact-Kerr g-disk",
    "mc_thin" => "MC thin sphere", "mc_scatter" => "MC scattering sphere (τ≈0.2)",
    "mc_scatter0" => "MC thin sphere, scattering on")

function main_merge()
    raw = joinpath(ROOT, "BLP", "bench", "results", "raw")
    recs = Dict{String,Any}[]
    for f in sort(readdir(raw))
        endswith(f, ".jsonl") && append!(recs, read_records(joinpath(raw, f)))
    end
    bench = filter(r -> r["kind"] == "bench", recs)
    tune = filter(r -> r["kind"] == "tune", recs)
    tune_ext = filter(r -> r["kind"] == "tune_ext", recs)
    sort!(bench; by = r -> r["date"])
    final = Dict{Any,Dict{String,Any}}()
    for r in bench
        final[ratekey(r)] = r
    end
    for r in bench
        r["final"] = final[ratekey(r)] === r
    end
    outdir = joinpath(ROOT, "BLP", "bench", "results", "out")
    for r in values(final)
        r["backend"] == "gpu" || continue
        k = r["key"]
        cpu = load_saved(outdir, k, "cpu")
        gpu = load_saved(outdir, k, "gpu$(r["ndev"])")
        r["agreement"] = agreement(r["leg"], gpu, cpu)
        if r["ndev"] > 1
            g1 = load_saved(outdir, k, "gpu1")
            r["agreement"]["bitwise_vs_gpu1"] = !isempty(g1) &&
                all(bitwise(gpu[n], g1[n]) for n in intersect(keys(gpu), keys(g1)))
        end
    end
    date = length(ARGS) >= 2 ? ARGS[2] : isempty(bench) ? "none" : bench[1]["date"][1:10]
    base = joinpath(ROOT, "BLP", "bench", "results", "anta_$date")
    doc = Dict{String,Any}("host" => isempty(bench) ? HOST : bench[1]["host"],
        "cpu_model" => isempty(bench) ? CPU_MODEL : bench[1]["cpu_model"],
        "records" => bench, "tune" => tune, "tune_ext" => tune_ext)
    open(io -> (json(io, doc); println(io)), base * ".json", "w")
    write_table(base * ".md", collect(values(final)), tune, tune_ext, date)
    println("wrote $base.json and $base.md: $(length(bench)) bench, $(length(tune)) tune records")
    return 0
end

function write_table(path, finals, tune, tune_ext, date)
    order = Dict(l => i for (i, l) in enumerate(LEGS_TORUS))
    sort!(finals; by = r -> (order[r["leg"]], r["res"], r["backend"] == "gpu",
                             r["backend"] == "cpu" ? r["threads"] : r["ndev"]))
    best_cpu = Dict{String,Any}()
    for r in finals
        r["backend"] == "cpu" || continue
        b = get(best_cpu, r["key"], nothing)
        (b === nothing || r["rate"] > b["rate"]) && (best_cpu[r["key"]] = r)
    end
    commits = join(sort!(unique(r["commit"] for r in finals)), ", ")
    trees = join(sort!(unique(get(r, "src_tree", tree_of(r["commit"])) for r in finals)), ", ")
    open(path, "w") do io
        println(io, "# BLP benchmarks on anta, $date\n")
        println(io, "Host: 2× $(isempty(finals) ? CPU_MODEL : finals[1]["cpu_model"]) ",
                "(72 cores, 144 hardware threads), 2× NVIDIA A100 80 GB PCIe; Julia ",
                isempty(finals) ? string(VERSION) : finals[1]["julia"], "; commits ", commits,
                ", all with the BLP/src+BLP/ext trees ", trees, ".")
        println(io, "The host was shared: `load` is the 1-minute load average before and ",
                "after the timed calls (it includes the run's own threads), `other cores` the ",
                "cores that other processes kept busy during them. GPU rows ran inside ",
                "Slurm jobs (4 Julia threads). ",
                "`wall` is the best of $(isempty(finals) ? REPEAT : finals[1]["nrep"]) warm ",
                "calls after a first call; `first` is that first call (`*`: it compiled the ",
                "leg's program). Speed-up: GPU rate over the best CPU rate of the same leg ",
                "(its thread count in brackets).\n")
        println(io, "| leg | res | backend | threads / devices | lanes | chunk | items | ",
                "wall [s] | first [s] | rate | unit | load | other cores | speed-up | ",
                "agreement with CPU |")
        println(io, "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|---|---|--:|--:|---|")
        for r in finals
            gpu = r["backend"] == "gpu"
            sp = "–"
            if gpu && haskey(best_cpu, r["key"])
                b = best_cpu[r["key"]]
                sp = @sprintf("%.1f× (%d)", r["rate"] / b["rate"], b["threads"])
            end
            ag = gpu ? agreement_text(get(r, "agreement", nothing)) :
                 (r["same_as_saved"] === nothing ? "reference" :
                  r["same_as_saved"] ? "bitwise = reference" : "differs from reference")
            first = fmt(r["first_s"]) * (r["first_compiles"] ? "*" : "")
            println(io, "| ", LEG_LABEL[r["leg"]], " | ", r["res"] > 0 ? "$(r["res"])²" : "–",
                    " | ", r["backend"], " | ", gpu ? "$(r["ndev"]) GPU" : "$(r["threads"]) thr",
                    " | ", gpu ? r["lanes"] : "–", " | ", r["chunk"], " | ", r["items"], " | ",
                    fmt(r["wall_s"]), " | ", first, " | ", fmt(r["rate"]), " | ", r["unit"],
                    " | ", @sprintf("%.0f/%.0f", r["load1_start"], r["load1_end"]), " | ",
                    fmt(get(r, "other_cores", nothing); d = 2), " | ", sp,
                    " | ", ag, " |")
        end
        isempty(tune) && return
        println(io, "\n## GPU pool sweep (warm rate of one call per configuration)\n")
        println(io, "Lanes 1–4 per device × chunk 2^14–2^17, each on a fresh pool; the timed ",
                "legs above use the best configuration.\n")
        keys_ = sort!(unique([(t["key"], t["ndev"]) for t in tune]))
        println(io, "| case | devices | best lanes | best chunk | best rate | worst rate | ",
                "rate at lanes 2, chunk 65536 |")
        println(io, "|---|--:|--:|--:|--:|--:|--:|")
        for (k, nd) in keys_
            ts = filter(t -> t["key"] == k && t["ndev"] == nd, tune)
            b = ts[argmax([t["rate"] for t in ts])]; w = ts[argmin([t["rate"] for t in ts])]
            dflt = filter(t -> t["lanes"] == 2 && t["chunk"] == 65536, ts)
            println(io, "| $k | $nd | $(b["lanes"]) | $(b["chunk"]) | $(fmt(b["rate"])) | ",
                    "$(fmt(w["rate"])) | $(isempty(dflt) ? "–" : fmt(dflt[1]["rate"])) |")
        end
        isempty(tune_ext) && return
        println(io, "\n## Chunks above the sweep (warm rate of one call)\n")
        println(io, "| case | devices | lanes | chunk | rate |")
        println(io, "|---|--:|--:|--:|--:|")
        for t in sort(tune_ext; by = t -> (t["key"], t["ndev"], t["lanes"], t["chunk"]))
            println(io, "| $(t["key"]) | $(t["ndev"]) | $(t["lanes"]) | $(t["chunk"]) | ",
                    "$(fmt(t["rate"])) |")
        end
    end
end

function agreement_text(a)
    a === nothing && return "–"
    haskey(a, "agreement") && return a["agreement"]
    s = a["bitwise"] ? "bitwise" : ""
    if haskey(a, "flux_ratio") && !a["bitwise"]
        s = @sprintf("flux ratio %.6f, max ΔI/peak %.1e, max rel ΔI (I > 1e-3 peak) %.1e",
                     a["flux_ratio"], a["max_diff_over_peak"], a["max_rel_above_1e-3_peak"])
        haskey(a, "levels_equal") && (s *= a["levels_equal"] ? ", same levels" : ", levels differ")
    elseif haskey(a, "counts_bitwise") && !a["bitwise"]
        s = @sprintf("counts %s; Σw, ΣwE max rel %.1e, %.1e",
                     a["counts_bitwise"] ? "bitwise" : "differ", a["max_rel_sumw"],
                     a["max_rel_sumwE"])
    elseif !a["bitwise"]
        s = "not bitwise"
    end
    haskey(a, "bitwise_vs_gpu1") && (s *= a["bitwise_vs_gpu1"] ? "; = 1 GPU bitwise" : "; ≠ 1 GPU")
    return s
end

# ---------------------------------------------------------------- merge-reference
# The records of both packages (reference.jl and the comparison runs of this script) in
# BLP/bench/results/reference/raw, paired per deck, leg, backend and device or thread count.
hwkey(r) = r["backend"] == "gpu" ? r["ndev"] : r["threads"]
rcfg(r) = get(r, "config", r["backend"] == "gpu" ? "lanes$(r["lanes"])_chunk$(r["chunk"])" :
                                                 "chunk$(r["chunk"])")
# BLP's configurations: the defaults (PoolExecutor lanes 2 and ThreadedExecutor, both at
# chunk `auto`) and the tuned ones (chunk 2^17 on the pool; chunk 64 on threads).
blp_default_cfg(be) = be == "gpu" ? "lanes2_chunkauto" : "chunkauto"
blp_tuned_cfg(be) = be == "gpu" ? "lanes2_chunk131072" : "chunk64"

const CMP_ORDER = Dict(("reference", l) => i for (i, l) in enumerate([LEGS_REFERENCE; LEGS_BINARY]))
for (i, l) in enumerate(["img_192", "img_384", "img_768", "amr_384"])
    CMP_ORDER[("torus", l)] = 100 + i
end
const CMP_LABEL = Dict(
    ("reference", "fused_384") => "fused I 384², uniform resample",
    ("reference", "fused_768") => "fused I 768², uniform resample",
    ("reference", "fused_mixed_768") => "fused I 768², mixed precision",
    ("reference", "fused_pol_256") => "fused IQUV 256², uniform resample",
    ("reference", "fused_pol_384") => "fused IQUV 384², uniform resample",
    ("reference", "amr_384") => "native blocks I 384²",
    ("reference", "amr_pol_384") => "native blocks IQUV 384²",
    ("reference", "amr_driver_pol") => "adaptive IQUV, base 384, block 8",
    ("reference", "analytic_1024") => "exact-Kerr g-disk 1024²",
    ("reference", "mc_thin") => "MC thin sphere, 10⁷",
    ("reference", "mc_scatter") => "MC scattering sphere, 10⁷",
    ("reference", "bbh455_480") => "binary 00455 I 480²",
    ("reference", "bbh455_pol_480") => "binary 00455 IQUV 480²",
    ("reference", "bbh122_pol_480") => "binary 00122 IQUV 480²",
    ("reference", "bbh_i90_amr_pol") => "binary 00455 i = 90° adaptive IQUV",
    ("torus", "img_192") => "torus: blocks I 192²",
    ("torus", "img_384") => "torus: blocks I 384²",
    ("torus", "img_768") => "torus: blocks I 768²",
    ("torus", "amr_384") => "torus: adaptive IQUV, base 384, block 16")
cmp_label(deck, k) = get(CMP_LABEL, (deck, k), "$deck: $k")

# The fields of a record that the comparison keeps.
brief(r) = r === nothing ? nothing : Dict{String,Any}(k => get(r, k, nothing) for k in
    ("backend", "config", "lanes", "chunk", "items", "wall_s", "walls", "rate", "first_s", "first_compiles",
     "setup_s", "setup_runs", "load1_start", "load1_end", "other_cores", "job", "date",
     "same_as_saved", "nrays", "levels_hist", "n_primary", "n_children", "smi_gpus", "smi_apps",
     "dev_used_gb", "commit", "src_tree"))

# Slurm submissions and final states from the runner log of run_reference.sh.
function runner_jobs(path)
    jobs = Dict{String,Dict{String,Any}}()
    isfile(path) || return Dict{String,Any}[]
    for l in eachline(path)
        m = match(r"^(\S+ \S+) (\S+) attempt (\d+) job (\d+) (submitted|adopted)", l)
        if m !== nothing
            haskey(jobs, m[4]) && continue
            jobs[m[4]] = Dict{String,Any}("job" => m[4], "item" => m[2], "attempt" => parse(Int, m[3]),
                                          "submitted" => m[1], "state" => "PENDING")
            continue
        end
        m = match(r"^(\S+ \S+) (\S+) job (\d+) (\S+)$", l)
        (m === nothing || !haskey(jobs, m[3])) && continue
        if m[4] == "running"
            jobs[m[3]]["started"] = m[1]
        else
            jobs[m[3]]["state"] = m[4]; jobs[m[3]]["ended"] = m[1]
        end
    end
    dir = dirname(path)
    for j in values(jobs)            # the exit status at the end of the job's log
        logs = filter(isfile, [joinpath(d, f) for d in (dir, joinpath(dir, "superseded")) if isdir(d)
                               for f in readdir(d) if endswith(f, "_$(j["job"]).log")])
        isempty(logs) && continue
        ex = filter(l -> startswith(l, "exit "), readlines(first(logs)))
        j["log"] = relpath(first(logs), dir)
        j["exit"] = isempty(ex) ? "none" : split(last(ex))[2]
    end
    return sort!(collect(values(jobs)); by = j -> parse(Int, j["job"]))
end

function main_merge_reference()
    raw = joinpath(CMP_DIR, "raw")
    recs = Dict{String,Any}[]
    if isdir(raw)
        for f in sort(readdir(raw))
            endswith(f, ".jsonl") && append!(recs, read_records(joinpath(raw, f)))
        end
    end
    bench = sort!(filter(r -> r["kind"] == "bench", recs); by = r -> r["date"])
    phases = sort!(filter(r -> r["kind"] == "phase", recs); by = r -> r["date"])
    fk(r) = (r["package"], r["deck"], r["key"], r["backend"], hwkey(r), rcfg(r))
    final = Dict{Any,Dict{String,Any}}()
    for r in bench
        final[fk(r)] = r
    end
    for r in bench
        r["final"] = final[fk(r)] === r
    end
    pfinal = Dict{Any,Dict{String,Any}}()
    for r in phases
        pfinal[(r["package"], r["deck"], r["key"], hwkey(r))] = r
    end
    groups = unique([(r["deck"], r["key"], r["backend"], hwkey(r)) for r in values(final)])
    sort!(groups; by = g -> (g[3] == "cpu", g[4], get(CMP_ORDER, (g[1], g[2]), 999)))
    pairs = Dict{String,Any}[]
    for (deck, k, be, hw) in groups
        mine(pkg) = sort!([r for r in values(final) if r["package"] == pkg && r["deck"] == deck &&
                           r["key"] == k && r["backend"] == be && hwkey(r) == hw]; by = rcfg)
        ref = mine("reference"); blp = mine("BLP")
        pick(rs, c) = (i = findfirst(r -> rcfg(r) == c, rs); i === nothing ? nothing : rs[i])
        best(rs) = isempty(rs) ? nothing : rs[argmax([r["rate"] for r in rs])]
        rdef = pick(ref, be == "gpu" ? "default" : "threads")
        rbest = best(ref)
        bdef = pick(blp, blp_default_cfg(be)); btun = pick(blp, blp_tuned_cfg(be)); bbest = best(blp)
        # the first call that compiled the leg's program, else the default's
        firstrec(rs, d) = (i = findfirst(r -> r["first_compiles"] === true, rs); i === nothing ? d : rs[i])
        ratio(a, b) = a === nothing || b === nothing ? nothing : a["rate"] / b["rate"]
        tag = be == "gpu" ? "gpu$hw" : "cpu"
        dir = joinpath(CMP_DIR, "out", deck)
        ag = isdir(dir) ? agreement(k, load_saved(dir, k, "blp-$tag"), load_saved(dir, k, "ref-$tag")) :
             Dict{String,Any}("agreement" => "no arrays")
        push!(pairs, Dict{String,Any}("deck" => deck, "key" => k, "label" => cmp_label(deck, k),
            "backend" => be, "ndev" => be == "gpu" ? hw : 0, "threads" => be == "cpu" ? hw : 4,
            "unit" => (isempty(ref) ? blp : ref)[1]["unit"],
            "ref_default" => brief(rdef), "ref_best" => brief(rbest),
            "ref_configs" => [brief(r) for r in ref],
            "blp_default" => brief(bdef), "blp_tuned" => brief(btun), "blp_best" => brief(bbest),
            "blp_configs" => [brief(r) for r in blp],
            "ratio_default" => ratio(bdef, rdef), "ratio_tuned" => ratio(btun, rdef),
            "ratio_best" => ratio(bbest, rbest), "agreement" => ag,
            "ref_first" => brief(firstrec(ref, rdef)), "blp_first" => brief(firstrec(blp, bdef)),
            "phases_ref" => get(pfinal, ("reference", deck, k, hw), nothing),
            "phases_blp" => get(pfinal, ("BLP", deck, k, hw), nothing)))
    end
    jobs = runner_jobs(joinpath(CMP_DIR, "logs", "runner.log"))
    date = length(ARGS) >= 2 ? ARGS[2] : isempty(bench) ? "none" : bench[1]["date"][1:10]
    base = joinpath(RESULTS, "anta_$(date)_reference")
    doc = Dict{String,Any}("host" => isempty(bench) ? HOST : bench[1]["host"],
        "cpu_model" => isempty(bench) ? CPU_MODEL : bench[1]["cpu_model"],
        "pairs" => pairs, "records" => bench, "phases" => phases, "jobs" => jobs)
    open(io -> (json(io, doc); println(io)), base * ".json", "w")
    write_reference_table(base * ".md", pairs, bench, jobs, date)
    println("wrote $base.json and $base.md: $(length(pairs)) pairs, $(length(bench)) records, ",
            "$(length(phases)) phase records, $(length(jobs)) jobs")
    return 0
end

fmtw(r) = r === nothing ? "–" : fmt(r["wall_s"])
fmtr(r) = r === nothing ? "–" : fmt(r["rate"])
fmtx(x) = x === nothing ? "–" : @sprintf("%.2f", x)
fmtload(r) = r === nothing ? "–" : @sprintf("%.0f/%.0f", r["load1_start"], r["load1_end"])
first_text(r) = r === nothing ? "–" :
    (get(r, "setup_runs", false) === true && r["backend"] == "gpu" ? "setup " * fmt(r["setup_s"]) * ", " : "") *
    fmt(r["first_s"]) * (r["first_compiles"] === true ? "*" : "")

function cmp_agreement_text(a)
    a === nothing && return "–"
    haskey(a, "agreement") && return a["agreement"]
    a["bitwise"] && return "bitwise"
    if haskey(a, "flux_ratio")
        s = @sprintf("flux %.9f, max ΔI/peak %.1e", a["flux_ratio"], a["max_diff_over_peak"])
        q = [get(a, "max_diff_over_peak_$n", 0.0) for n in ("Q", "U", "V")]
        any(>(0), q) && (s *= @sprintf(", QUV %.1e", maximum(q)))
        haskey(a, "levels_equal") && (s *= a["levels_equal"] ? ", same levels" : ", levels differ")
        return s
    elseif haskey(a, "counts_bitwise")
        return @sprintf("counts %s; Σw, ΣwE max rel %.1e, %.1e",
                        a["counts_bitwise"] ? "bitwise" : "differ", a["max_rel_sumw"], a["max_rel_sumwE"])
    end
    return "not bitwise"
end

ref_cfg_text(r) = r === nothing ? "–" :
    r["config"] == "default" ? "default" : replace(r["config"], "chunk" => "chunk ")

function geomean_text(xs)
    xs = filter(x -> x !== nothing, xs)
    isempty(xs) && return "–"
    return @sprintf("%.2f (%.2f–%.2f, %d legs)", exp(sum(log, xs) / length(xs)), minimum(xs),
                    maximum(xs), length(xs))
end

function write_reference_table(path, pairs, bench, jobs, date)
    refs = filter(r -> r["package"] == "reference", bench)
    blps = filter(r -> r["package"] == "BLP", bench)
    trees(rs) = join(sort!(unique(string(r["src_tree"]) for r in rs)), ", ")
    commits = join(sort!(unique(string(r["commit"]) for r in bench)), ", ")
    open(path, "w") do io
        println(io, "# BLP against the reference package on anta, $date\n")
        println(io, "Host: 2× $(isempty(bench) ? CPU_MODEL : bench[1]["cpu_model"]) (72 cores, ",
                "144 hardware threads), 2× NVIDIA A100 80 GB PCIe; Julia ",
                isempty(bench) ? string(VERSION) : bench[1]["julia"], "; commits ", commits,
                "; reference `src/` trees ", trees(refs), "; BLP `BLP/src`+`BLP/ext` trees ",
                trees(blps), ". Both GPU environments (`results/blp/gpu/env_old` for the ",
                "reference, `results/blp/gpu/env` for BLP) hold the same CUDA.jl, GPUCompiler, ",
                "LLVM.jl and KernelAbstractions versions and the CUDA runtime 13.3.\n")
        println(io, "Every GPU row ran inside a Slurm job on partition `preempt` with 4 Julia ",
                "threads; the reference leg and the BLP leg of a row ran in the same job, one ",
                "process each, back to back, on the same devices. CPU rows ran on the login ",
                "shell, started only when the 1-minute load was below 16. `wall` is the best of ",
                "3 calls after a first call; `first` is that first call (`*`: it compiled the ",
                "leg's program; `setup`: the reference harness runs the AMR driver once on one ",
                "device in its setup to count the rays, which then holds the compilation). ",
                "The uploads of the reference harness's setup are outside `first`; BLP uploads ",
                "inside its first call. `load` is the 1-minute load average before/after the ",
                "timed calls. Ratio: BLP rate over reference rate (above 1: BLP faster); ",
                "`default` pairs the defaults (reference: the harness's call; BLP: ",
                "PoolExecutor lanes 2 or ThreadedExecutor, both at chunk `auto`), `best` the ",
                "best configuration of each package. Agreement: the BLP output against the ",
                "reference output of the same hardware.\n")
        println(io, "The reference's GPU configurations are its harness default (the cached ",
                "DevicePool of lanes 2, chunk 65536; the Monte Carlo drivers cut their own ",
                "chunks of 2^20 primaries) and a DevicePool of lanes 2, chunk 131072 (Monte ",
                "Carlo: the drivers' `chunk` keyword at 65536 and 131072). BLP's: lanes 2 at ",
                "chunk `auto` (default: one launch of up to 2^20 items on one device, at most ",
                "16 MiB of outputs per launch, the Monte Carlo cascade at 2^20 primaries), ",
                "131072 (tuned) and 65536, and where recorded 1048576 (the Monte Carlo and ",
                "binary legs, which then count in BLP's best). On the CPU the reference runs its ",
                "CPU entry points: the two-phase render (trace, then march along the stored ",
                "states), `render_amr` with its two-phase closure (which reads the snapshot in ",
                "each call), `render_analytic`, `mc_spectrum`, `mc_spectrum_scatter`; BLP runs ",
                "the same products as on the GPU with ThreadedExecutor chunk `auto` (default), ",
                "64 (tuned) and 4096.\n")
        for (title, sel) in (("GPU", p -> p["backend"] == "gpu"), ("CPU", p -> p["backend"] == "cpu"))
            ps = filter(sel, pairs)
            isempty(ps) && continue
            gpu = title == "GPU"
            println(io, "\n## $title\n")
            println(io, "| leg | ", gpu ? "devices" : "threads", " | reference wall [s] | ",
                    "reference rate | reference best (config) | BLP wall [s] default | ",
                    "BLP rate default | BLP wall [s] tuned | BLP rate tuned | BLP/ref default | ",
                    "BLP/ref best | agreement | first [s] ref / BLP | load ref / BLP | jobs |")
            println(io, "|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|---|---|---|---|")
            for p in ps
                rd = p["ref_default"]; rb = p["ref_best"]; bd = p["blp_default"]; bt = p["blp_tuned"]
                jobs_ = join(unique(string(r["job"]) for r in (rd, bd, bt) if r !== nothing), ", ")
                println(io, "| ", p["label"], " | ", gpu ? p["ndev"] : p["threads"], " | ",
                        fmtw(rd), " | ", fmtr(rd), " | ",
                        rb === nothing ? "–" : "$(fmtr(rb)) ($(ref_cfg_text(rb)))", " | ",
                        fmtw(bd), " | ", fmtr(bd), " | ", fmtw(bt), " | ", fmtr(bt), " | ",
                        fmtx(p["ratio_default"]), " | ", fmtx(p["ratio_best"]), " | ",
                        cmp_agreement_text(p["agreement"]), " | ", first_text(p["ref_first"]), " / ",
                        first_text(p["blp_first"]), " | ", fmtload(rd), " / ", fmtload(bd), " | ", jobs_, " |")
            end
        end
        gpu1 = [(p["deck"], p["key"]) for p in pairs if p["backend"] == "gpu" && p["ndev"] == 1]
        gpu2 = Set((p["deck"], p["key"]) for p in pairs if p["backend"] == "gpu" && p["ndev"] == 2)
        missing = [cmp_label(d, k) for (d, k) in gpu1 if !((d, k) in gpu2)]
        if !isempty(missing)
            println(io, "\n## Not measured\n")
            println(io, "On two GPUs (no record of either package): ", join(missing, "; "), ". ",
                    "No two-device run of these legs had completed when this table was ",
                    "written; the Slurm jobs below show the attempts.")
        end
        println(io, "\n## BLP/reference ratio per hardware (reference deck)\n")
        println(io, "Geometric mean over the legs (smallest–largest), rate ratio BLP/reference.\n")
        println(io, "| hardware | default vs default | best vs best |")
        println(io, "|---|---|---|")
        hws = unique([(p["backend"], p["backend"] == "gpu" ? p["ndev"] : p["threads"])
                      for p in pairs if p["deck"] == "reference"])
        for (be, hw) in hws
            ps = filter(p -> p["deck"] == "reference" && p["backend"] == be &&
                             (be == "gpu" ? p["ndev"] : p["threads"]) == hw, pairs)
            println(io, "| ", be == "gpu" ? "$hw GPU" : "CPU, $hw threads", " | ",
                    geomean_text([p["ratio_default"] for p in ps]), " | ",
                    geomean_text([p["ratio_best"] for p in ps]), " |")
        end
        ph = filter(p -> p["phases_ref"] !== nothing || p["phases_blp"] !== nothing, pairs)
        if !isempty(ph)
            println(io, "\n## Device phases of one profiled call (default configuration)\n")
            println(io, "CUPTI trace of one warm call: the call's window, the host time before the ",
                    "first device activity (prologue) and after the last (epilogue), and per device ",
                    "the union of kernel intervals, the kernel launches and the idle time.\n")
            println(io, "| leg | devices | package | window [ms] | prologue [ms] | epilogue [ms] | ",
                    "kernel [ms] per device | launches per device | idle [ms] per device |")
            println(io, "|---|--:|---|--:|--:|--:|---|---|---|")
            for p in ph, (pkg, r) in (("reference", p["phases_ref"]), ("BLP", p["phases_blp"]))
                r === nothing && continue
                ms(x) = x === nothing ? "–" : @sprintf("%.1f", 1e3 * x)
                if haskey(r, "error")
                    println(io, "| ", p["label"], " | ", p["ndev"], " | ", pkg, " | profiler error: ",
                            first(r["error"], 80), " | | | | | |")
                    continue
                end
                devs = sort!(unique([parse(Int, m[1]) for m in (match(r"^d(\d+)_kernel_s$", k)
                                     for k in keys(r)) if m !== nothing]))
                per(q, f) = join((f(get(r, "d$(d)_$q", nothing)) for d in devs), " / ")
                println(io, "| ", p["label"], " | ", p["ndev"], " | ", pkg, " | ",
                        ms(get(r, "window_s", nothing)), " | ", ms(get(r, "host_prologue_s", nothing)),
                        " | ", ms(get(r, "host_epilogue_s", nothing)), " | ", per("kernel_s", ms),
                        " | ", per("launches", x -> x === nothing ? "–" : string(x)), " | ",
                        per("idle_s", ms), " |")
            end
        end
        println(io, "\n## Every configuration\n")
        println(io, "| leg | backend | devices / threads | package | config | wall [s] | rate | ",
                "first [s] | load | other cores | same as first config | job |")
        println(io, "|---|---|--:|---|---|--:|--:|--:|---|--:|---|---|")
        for p in pairs, (pkg, rs) in (("reference", p["ref_configs"]), ("BLP", p["blp_configs"])), r in rs
            println(io, "| ", p["label"], " | ", p["backend"], " | ",
                    p["backend"] == "gpu" ? p["ndev"] : p["threads"], " | ", pkg, " | ",
                    r["config"], " | ", fmtw(r), " | ", fmtr(r), " | ", first_text(r), " | ",
                    fmtload(r), " | ", fmt(r["other_cores"]; d = 2), " | ",
                    r["same_as_saved"] === nothing ? "first" : r["same_as_saved"] ? "bitwise" : "differs",
                    " | ", r["job"], " |")
        end
        if !isempty(jobs)
            println(io, "\n## Slurm jobs\n")
            println(io, "Every submission of `slurm/run_reference.sh` (partition `preempt`): a job ",
                    "that ended PREEMPTED or CANCELLED was resubmitted and its successor skipped ",
                    "the legs already recorded; `exit` is the status at the end of the job's log ",
                    "(`none`: the job was cut off). A job whose log ends in a nonzero exit is not ",
                    "retried; its log is under `logs/superseded/`, and the BLP records of its legs ",
                    "whose reference half did not run are under `raw/superseded/`, outside the ",
                    "tables.\n")
            println(io, "| job | item | attempt | submitted | started | ended | final state | exit |")
            println(io, "|--:|---|--:|---|---|---|---|---|")
            for j in jobs
                println(io, "| ", j["job"], " | ", j["item"], " | ", j["attempt"], " | ",
                        j["submitted"], " | ", get(j, "started", "–"), " | ", get(j, "ended", "–"),
                        " | ", j["state"], " | ", get(j, "exit", "–"), " |")
            end
            items = unique([(j["item"]) for j in jobs])
            println(io, "\nItems (`tag`: `<devices>@<deck>=<legs>/…`, see `slurm/run_reference.sh`):\n")
            for l in eachline(joinpath(CMP_DIR, "logs", "runner.log"))
                m = match(r"^\S+ \S+ (\S+) is (\S+)$", l)
                (m === nothing || !(m[1] in items)) && continue
                println(io, "- `", m[1], "`: `", m[2], "`")
                filter!(!=(m[1]), items)
            end
        end
    end
end

exit(MODE == "merge" ? main_merge() : MODE == "bench" ? main_bench() :
     MODE == "merge-reference" ? main_merge_reference() :
     error("mode must be bench, merge or merge-reference, got $MODE"))
