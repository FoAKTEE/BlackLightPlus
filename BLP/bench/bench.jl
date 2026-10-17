# bench.jl — wall time and throughput of the BLP products on CPU threads or CUDA devices.
#
# Legs (BLP_LEGS, comma list; default all):
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
# The scattering legs pass one hot-cross table, built before the timing.
#
# Protocol per leg: `prepare` (not timed); one first call (timed as `first_s`: the JIT
# compilation when the leg's program has not run in this process, and on a GPU the
# uploads to the fresh pool); then BLP_REPEAT (3) timed calls; `wall_s` is the best.
# Throughput: rays (pixels; the ray count of render_amr) or superphotons (primaries plus
# children) per wall second. Every record carries the executor (`describe` of the result's
# `run`), threads, devices, lanes, chunk, the 1-minute load before and after the timed
# calls, `other_cores` (the cores that other processes kept busy during the timed calls,
# from /proc/stat less this process's CPU time), the commit, the tree hashes of BLP/src
# and BLP/ext, and the Slurm job.
#
# Backends: BLP_BACKEND=cpu (default; `ThreadedExecutor(chunk = BLP_CPU_CHUNK)`, 64 unless
# set, under `julia -t N`: `Threads.@threads` gives each thread one contiguous run of
# chunks, so a chunk count below the thread count idles threads) or gpu (a DevicePool
# over the ordinals in BLP_DEVICES, default 0; needs CUDA in the environment). On a GPU
# the pool's lanes and chunk are BLP_LANES and BLP_CHUNK when both are set, else the best
# warm configuration of the `tune` records of the same leg and device count (in
# BLP_TUNE_FROM, default BLP_OUT), else 2 and 65536. BLP_TUNE=1 runs the sweep instead:
# lanes 1–4 × chunk 2^14–2^17, each on a fresh pool (one first call, one warm call),
# appending one `tune` record per configuration;
# BLP_TUNE_LANES and BLP_TUNE_CHUNKS (comma lists) change the grid and BLP_TUNE_KIND the
# record kind (records of another kind do not choose the pool of the timed legs).
#
# Output: one JSON object per line appended to BLP_OUT (kind "bench" or "tune"; a
# preempted run loses only the leg in flight), one BENCH or TUNE line per record on
# stdout. BLP_SKIP_DONE=1 skips a leg (or a tuning configuration) whose record for the
# same BLP/src and BLP/ext trees, backend, thread or device count is already in BLP_OUT.
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
#
# Usage, from the repository root:
#   BLP_OUT=BLP/bench/results/raw/cpu16.jsonl julia -t 16 --project=BLP BLP/bench/bench.jl
#   BLP_BACKEND=gpu BLP_DEVICES=0 BLP_OUT=BLP/bench/results/raw/gpu.jsonl \
#       julia -t 4 --project=results/blp/gpu/env BLP/bench/bench.jl
#   BLP/bench/slurm/run_gpu.sh             (the GPU legs through Slurm, see README.md)
#   julia --project=BLP BLP/bench/bench.jl merge
# Other variables: BLP_REPEAT (3), BLP_CPU_CHUNK (64), BLP_RESES, BLP_POL_RESES,
# BLP_AMR_RES, BLP_AN_RES, BLP_MC_NS, BLP_TEST_GRMHD.

using Printf
using BLP
using BLP.Numerics: GNEWT, CL, MSUN
using BLP.Execution: describe, DevicePool, CUDADevice
using BLP.Data: build_hotcross_table

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const MODE = isempty(ARGS) ? "bench" : ARGS[1]
const BACKEND = lowercase(get(ENV, "BLP_BACKEND", "cpu"))
BACKEND in ("cpu", "gpu") || error("BLP_BACKEND must be cpu or gpu, got $BACKEND")
const GPU = MODE == "bench" && BACKEND == "gpu"
if GPU
    using CUDA
end

envlist(k, d) = [parse(Int, s) for s in split(get(ENV, k, d), ',') if !isempty(strip(s))]
const LEGS_ALL = ["img", "pol", "amr", "analytic", "mc_thin", "mc_scatter", "mc_scatter0"]
const LEGS = let s = strip(get(ENV, "BLP_LEGS", ""))
    isempty(s) ? LEGS_ALL : String.(strip.(split(s, ',')))
end
const RESES = envlist("BLP_RESES", "192,384,768")
const POL_RESES = envlist("BLP_POL_RESES", "256,384")
const AMR_RES = parse(Int, get(ENV, "BLP_AMR_RES", "384"))
const AN_RES = parse(Int, get(ENV, "BLP_AN_RES", "1024"))
const MC_NS = parse(Float64, get(ENV, "BLP_MC_NS", "1e7"))
const REPEAT = parse(Int, get(ENV, "BLP_REPEAT", "3"))
const CPU_CHUNK = parse(Int, get(ENV, "BLP_CPU_CHUNK", "64"))
const DEVICES = envlist("BLP_DEVICES", "0")
const TUNE = get(ENV, "BLP_TUNE", "0") == "1"
const SKIP_DONE = get(ENV, "BLP_SKIP_DONE", "0") == "1"
const RESULTS = joinpath(ROOT, "BLP", "bench", "results")
const OUT = get(ENV, "BLP_OUT", joinpath(RESULTS, "raw", "records.jsonl"))
const TUNE_FROM = get(ENV, "BLP_TUNE_FROM", OUT)
const SAVE_DIR = get(ENV, "BLP_SAVE_DIR", joinpath(RESULTS, "out"))
const SNAP = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
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

# ---------------------------------------------------------------- JSON (flat records)
json(io, v::AbstractString) = print(io, '"', escape_string(v), '"')
json(io, v::Symbol) = json(io, String(v))
json(io, v::Bool) = print(io, v ? "true" : "false")
json(io, v::Integer) = print(io, v)
json(io, v::AbstractFloat) = isfinite(v) ? print(io, repr(Float64(v))) : print(io, "null")
json(io, ::Nothing) = print(io, "null")
function json(io, v::AbstractVector)
    print(io, '[')
    for (i, x) in enumerate(v)
        i > 1 && print(io, ", ")
        json(io, x)
    end
    print(io, ']')
end
function json(io, d::AbstractDict)
    print(io, '{')
    for (i, k) in enumerate(sort!(collect(keys(d)); by = string))
        i > 1 && print(io, ", ")
        json(io, string(k)); print(io, ": "); json(io, d[k])
    end
    print(io, '}')
end
json(io, nt::NamedTuple) = json(io, Dict(pairs(nt)))

# A JSON reader for the records written above: objects, arrays, strings, numbers,
# true, false, null.
function parse_json(s::AbstractString)
    v, i = _pj(s, _ws(s, firstindex(s)))
    return v
end
_ws(s, i) = (while i <= lastindex(s) && isspace(s[i]); i = nextind(s, i); end; i)
function _pj(s, i)
    c = s[i]
    if c == '{'
        d = Dict{String,Any}(); i = _ws(s, i + 1)
        s[i] == '}' && return d, i + 1
        while true
            k, i = _pj(s, _ws(s, i)); i = _ws(s, i)
            s[i] == ':' || error("JSON: expected ':' at $i")
            v, i = _pj(s, _ws(s, i + 1)); d[k] = v; i = _ws(s, i)
            s[i] == '}' && return d, i + 1
            s[i] == ',' || error("JSON: expected ',' at $i"); i += 1
        end
    elseif c == '['
        a = Any[]; i = _ws(s, i + 1)
        s[i] == ']' && return a, i + 1
        while true
            v, i = _pj(s, _ws(s, i)); push!(a, v); i = _ws(s, i)
            s[i] == ']' && return a, i + 1
            s[i] == ',' || error("JSON: expected ',' at $i"); i += 1
        end
    elseif c == '"'
        j = i + 1; buf = IOBuffer()
        while s[j] != '"'
            if s[j] == '\\'
                e = s[j + 1]
                if e == 'u'
                    print(buf, Char(parse(UInt16, s[j+2:j+5]; base = 16))); j += 6
                else
                    print(buf, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e); j += 2
                end
            else
                print(buf, s[j]); j = nextind(s, j)
            end
        end
        return String(take!(buf)), j + 1
    elseif startswith(SubString(s, i), "true")
        return true, i + 4
    elseif startswith(SubString(s, i), "false")
        return false, i + 5
    elseif startswith(SubString(s, i), "null")
        return nothing, i + 4
    else
        m = match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?", SubString(s, i))
        m === nothing && error("JSON: unexpected character at $i")
        v = m[1] === nothing && m[2] === nothing ? parse(Int, m.match) : parse(Float64, m.match)
        return v, i + ncodeunits(m.match)
    end
end

read_records(path) = isfile(path) ?
    [parse_json(l) for l in eachline(path) if !isempty(strip(l))] : Dict{String,Any}[]

function append_record(rec::AbstractDict)
    mkpath(dirname(OUT))
    line = sprint(json, rec)
    open(OUT, "a") do io
        println(io, line)
    end
    return rec
end

# ---------------------------------------------------------------- host and device state
load1() = try parse(Float64, split(read("/proc/loadavg", String))[1]) catch; NaN end

# (busy jiffies of the host, CPU jiffies of this process, time): the cores that other
# processes kept busy between two samples a, b are `other_cores(a, b)`.
function cpu_sample()
    v = parse.(Int, split(readline("/proc/stat"))[2:end])
    s = split(read("/proc/self/stat", String))
    return (v[1] + v[2] + v[3] + v[6] + v[7] + v[8], parse(Int, s[14]) + parse(Int, s[15]), time())
end
const CLK_TCK = try parse(Int, readchomp(`getconf CLK_TCK`)) catch; 100 end
other_cores(a, b) = ((b[1] - a[1]) - (b[2] - a[2])) / (CLK_TCK * (b[3] - a[3]))

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

const SNAPSHOT = Ref{Any}(nothing)
snapshot() = (SNAPSHOT[] === nothing && (SNAPSHOT[] = AthenaKSnapshot(SNAP)); SNAPSHOT[])

const HOTCROSS = Ref{Any}(nothing)
hotcross() = (HOTCROSS[] === nothing && (HOTCROSS[] = build_hotcross_table()); HOTCROSS[])

const SCATTER_SPHERE = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)

# A case: its leg, resolution (0 for Monte Carlo), item unit, program kind (the first case
# of a kind in a process compiles it), and `setup() -> (call(exec) -> result)`.
struct Case
    leg::String
    res::Int
    unit::String
    kind::String
    setup::Function
end
key(c::Case) = c.res > 0 ? "$(c.leg)_$(c.res)" : c.leg

torus_plan(res, mode) =
    prepare(torus_cfg(res); spacetime = KerrSchildCartesian(0.9), matter = snapshot(), mode = mode)

function cases()
    cs = Case[]
    for leg in LEGS
        if leg == "img"
            for r in RESES
                push!(cs, Case("img", r, "rays/s", "img",
                               () -> (p = torus_plan(r, StokesI()); exec -> render(p; backend = exec))))
            end
        elseif leg == "pol"
            for r in POL_RESES
                push!(cs, Case("pol", r, "rays/s", "pol",
                               () -> (p = torus_plan(r, IQUV()); exec -> render(p; backend = exec))))
            end
        elseif leg == "amr"
            push!(cs, Case("amr", AMR_RES, "rays/s", "amr", () -> begin
                p = torus_plan(AMR_RES, IQUV())
                exec -> render_amr(p; block = 16, max_level = 2, backend = exec)
            end))
        elseif leg == "analytic"
            push!(cs, Case("analytic", AN_RES, "rays/s", "analytic", () -> begin
                cfg = RunConfig(camera = CameraConfig(inclination_deg = 17.0, distance = 1.0e4,
                                                      fov = 20.0, resolution = AN_RES))
                p = prepare(cfg; spacetime = KerrSchildCartesian(0.94), matter = nothing,
                            mode = StokesI())
                disk = EquatorialDisk(0.94)
                exec -> render_analytic(p, disk; bands = (0, 1, 2), backend = exec)
            end))
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
            error("unknown leg $leg; expected one of $(join(LEGS_ALL, ", "))")
        end
    end
    return cs
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
                                         "levels_hist" => [count(==(l), r.levels) for l in 0:2])
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

backend_tag(ndev) = GPU ? "gpu$ndev" : "cpu"

# Save the result's arrays, or compare them bitwise with the files already saved.
function save_or_compare(c::Case, tag, r)
    mkpath(SAVE_DIR)
    same = true; any_old = false
    for (name, a) in fields(r)
        f = joinpath(SAVE_DIR, "$(key(c)).$tag.$name.$(join(size(a), 'x')).f64")
        v = Vector{Float64}(vec(a))
        if isfile(f)
            any_old = true
            old = reinterpret(Float64, read(f))
            same &= length(old) == length(v) &&
                    reinterpret(UInt64, old) == reinterpret(UInt64, v)
        else
            write(f, v)
        end
    end
    return any_old ? same : nothing
end

# ---------------------------------------------------------------- driver
function base_record(kind, c::Case)
    return Dict{String,Any}("kind" => kind, "leg" => c.leg, "res" => c.res, "key" => key(c),
        "unit" => c.unit, "backend" => BACKEND, "threads" => Threads.nthreads(),
        "ndev" => GPU ? length(DEVICES) : 0, "devices" => GPU ? DEVICES : Int[],
        "commit" => COMMIT, "src_tree" => SRC_TREE, "job" => JOB, "host" => HOST,
        "cpu_model" => CPU_MODEL, "julia" => string(VERSION),
        "date" => string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())))
end

done(recs, kind, c, extra = Dict()) = any(recs) do r
    get(r, "kind", "") == kind && r["key"] == key(c) && r["backend"] == BACKEND &&
        same_code(r) && (GPU ? r["ndev"] == length(DEVICES) :
                                  r["threads"] == Threads.nthreads() && r["chunk"] == CPU_CHUNK) &&
        all(get(r, k, nothing) == v for (k, v) in extra)
end

timed(f) = (t = time_ns(); r = f(); ((time_ns() - t) / 1e9, r))

const POOL_DEFAULT = (lanes = 2, chunk = 65536)
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
        return (lanes = parse(Int, ENV["BLP_LANES"]), chunk = parse(Int, ENV["BLP_CHUNK"]),
                source = "env")
    tunes = filter(r -> get(r, "kind", "") == "tune" && r["key"] == key(c) &&
                        r["ndev"] == length(DEVICES) && same_code(r), read_records(TUNE_FROM))
    isempty(tunes) && return (POOL_DEFAULT..., source = "default")
    best = tunes[argmin([r["warm_s"] for r in tunes])]
    return (lanes = best["lanes"], chunk = best["chunk"], source = "tune")
end

const KINDS_RUN = Set{String}()

function bench_case(c::Case)
    recs = SKIP_DONE ? read_records(OUT) : Dict{String,Any}[]
    if SKIP_DONE && done(recs, "bench", c)
        println("# skip $(key(c)): recorded"); return
    end
    call = c.setup()
    cfg = GPU ? pool_config(c) : nothing
    pool = GPU ? DevicePool(gpu_devices(); lanes = cfg.lanes, chunk = cfg.chunk) : nothing
    exec = GPU ? PoolExecutor(pool) : ThreadedExecutor(chunk = CPU_CHUNK)
    compiled = c.kind in KINDS_RUN
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
    rec = merge(base_record("bench", c), d, extra, device_state(), Dict{String,Any}(
        "items" => items, "wall_s" => minimum(walls), "walls" => walls, "first_s" => first_s,
        "first_compiles" => !compiled, "rate" => items / minimum(walls), "nrep" => REPEAT,
        "load1_start" => l0, "load1_end" => l1, "other_cores" => other_cores(s0, s1),
        "run_seconds" => run.seconds,
        "pool_source" => GPU ? cfg.source : "none"))
    rec["same_as_saved"] = save_or_compare(c, backend_tag(rec["ndev"]), r)
    append_record(rec)
    GPU && release(pool)
    @printf("BENCH leg=%s res=%d backend=%s threads=%d ndev=%d lanes=%d chunk=%d items=%d \
             wall_s=%.4g first_s=%.4g rate=%.4g unit=%s load1=%.1f/%.1f other_cores=%.1f \
             same_as_saved=%s\n",
            c.leg, c.res, BACKEND, rec["threads"], rec["ndev"], rec["lanes"], rec["chunk"],
            items, rec["wall_s"], first_s, rec["rate"], c.unit, l0, l1, rec["other_cores"],
            string(rec["same_as_saved"]))
    flush(stdout)
end

describe_record(e) = Dict{String,Any}("executor_kind" => string(e.kind),
    "executor_devices" => collect(String, e.devices), "lanes" => e.lanes, "chunk" => e.chunk,
    "executor_threads" => e.threads)

function tune_case(c::Case)
    call = c.setup()
    for lanes in TUNE_LANES, chunk in TUNE_CHUNKS
        recs = read_records(OUT)
        if SKIP_DONE && done(recs, TUNE_KIND, c, Dict("lanes" => lanes, "chunk" => chunk))
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
    println("# bench.jl backend=$BACKEND commit=$COMMIT threads=$(Threads.nthreads()) ",
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

# Agreement of a GPU output `g` with the CPU output `c` of the same case.
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
    order = Dict(l => i for (i, l) in enumerate(LEGS_ALL))
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

exit(MODE == "merge" ? main_merge() : MODE == "bench" ? main_bench() :
     error("mode must be bench or merge, got $MODE"))
