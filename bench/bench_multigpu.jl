# bench_multigpu.jl — phase-timed multi-device benchmark (results/blp/multigpu/design.md §7).
#
# Every leg runs on 1 and on n devices (CUDA [0] vs [0, 1], or HostDevices on the KA CPU
# backend) and prints bench_lines.jl records:
#   BENCH  warm best-of-BLP_REPEAT wall time, first-call (JIT-inclusive) time, rate,
#          speedup T1/Tn and bitwise equality of the n-device output with the 1-device one;
#   PHASE  (CUDA) from one CUPTI-profiled warm run: host prologue / epilogue, and per device
#          kernel busy, host-to-device, device-to-host and idle time;
#   SKIP   one line per leg that cannot run (no functional CUDA, missing snapshot).
# Legs: fused unpolarized 384² / 768², fused polarized 256² / 384², native AMR I / IQUV at
# 384², the AMR driver on the production deck (a = 0.9, 230 GHz, i = 60°, base 384,
# 2 levels, polarized), Monte Carlo thin / scatter at Ns = 1e7.
#
# Usage (from the checkout under test; the harness loads that checkout's src/):
#   julia --project=. -t 8 bench/bench_multigpu.jl                     # CUDA, 2 GPUs
#   BLP_BACKEND=cpu julia --project=. -t 8 bench/bench_multigpu.jl     # KA CPU smoke test
# To time another commit, run this file from a worktree of that commit:
#   git worktree add /tmp/base 3736b32 && cd /tmp/base &&
#   BLP_OUT=<repo>/results/blp/multigpu/baseline_3736b32.json julia --project=. -t 8 <repo>/bench/bench_multigpu.jl
# Environment: BLP_BACKEND (cuda|cpu), BLP_LEGS (comma list; default all), BLP_REPEAT (5),
# BLP_TEST_GRMHD (AthenaK snapshot for the imaging legs), BLP_REF_GPUMONTY (hot-cross
# table; Thomson limit otherwise), BLP_MC_NS (1e7), BLP_OUT (JSON copy of the records).
# The CPU backend uses a synthetic torus deck at 24² and Ns = 2e4: it checks that every
# leg runs and is device-count invariant; its times are not GPU predictions.
using BlackLightPlus, StaticArrays, Printf, KernelAbstractions, CUDA
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays_at, SphereModel,
                      ScatteringSphere, HotcrossTable, load_hotcross, render_amr
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot
using BlackLightPlus.Constants: SIGMA_THOMSON
using BlackLightPlus.MonteCarlo: NW, NT

const SRC = joinpath(dirname(Base.active_project()), "src")
include(joinpath(@__DIR__, "bench_lines.jl"))
for f in ("gpu_image.jl", "gpu_pol.jl", "gpu_amr.jl", joinpath("montecarlo", "mc_gpu.jl"))
    include(joinpath(SRC, f))
end

const BACKEND = get(ENV, "BLP_BACKEND", "cuda")
BACKEND in ("cuda", "cpu") || error("BLP_BACKEND must be cuda or cpu, got $BACKEND")
const CPU_SMOKE = BACKEND == "cpu"
const REPEAT = parse(Int, get(ENV, "BLP_REPEAT", CPU_SMOKE ? "2" : "5"))
const MC_NS = parse(Float64, get(ENV, "BLP_MC_NS", CPU_SMOKE ? "2e4" : "1e7"))
const SNAP = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const HOTCROSS = get(ENV, "BLP_REF_GPUMONTY", "")
const COMMIT = try readchomp(`git -C $(dirname(SRC)) rev-parse --short HEAD`) catch; "unknown" end

# every record goes to stdout and to LOG (the BLP_OUT JSON source)
const LOG = IOBuffer()
emit(kind, pairs...) = (l = bench_line(kind, pairs...); println(l); println(LOG, l); flush(stdout))

# device sets: 1 device, then all (up to 2)
function device_sets()
    CPU_SMOKE && return [[HostDevice(1)], [HostDevice(1), HostDevice(2)]]
    n = min(length(CUDA.devices()), 2)
    return n >= 2 ? [[0], [0, 1]] : [[0]]
end

# ---------------------------------------------------------------- decks
cfg_img(res) = RunConfig(bh_a = 0.9, m_bh_cgs = 6.2e9 * 1.989e33,
    mass_scale_cgs = 1.0e24, frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                          fov = CPU_SMOKE ? 24.0 : 40.0, resolution = res),
    plasma = PlasmaConfig(rat_high = 40.0, rat_low = 1.0, gamma = 13.0 / 9.0, use_p = false,
                          sigma_cut = 100.0, theta_e_min = 0.3))

# synthetic torus primitives (the CPU smoke deck; test/test_device_invariance.jl's shape)
function torus_prim(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < 2.2 || r > 11.0) && return ntuple(_ -> 0.0f0, 8)
    rho = exp(-((r - 5.5) / 2.5)^2)
    return Float32.((rho, -0.2y / r, 0.2x / r, 0.01, 0.08rho,
                     0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho)))
end
function synthetic_uniform(n = 24, bh = 12.0)
    host = zeros(Float32, 8, n, n, n); dx = 2bh / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        host[:, i, j, k] .= torus_prim(-bh + (i - 1) * dx, -bh + (j - 1) * dx, -bh + (k - 1) * dx)
    end
    return UniformFluidGrid(host, n, bh, -bh, (n - 1) / (2bh))
end
function synthetic_amr(nc = 6, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, nc, nc, nc, nmb, 8)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; xl = SVector{3,Float64}[]
    dxs = SVector{3,Float64}[]; lloc = zeros(Int, 3, nmb); m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        push!(lo, l); push!(hi, l .+ w); push!(xl, l .+ dx / 2); push!(dxs, SVector(dx, dx, dx))
        lloc[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc, q in 1:8
            data[i, j, k, m, q] = torus_prim(xl[m][1] + (i - 1) * dx, xl[m][2] + (j - 1) * dx,
                                            xl[m][3] + (k - 1) * dx)[q]
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                           ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                           nc, nc, nc, nmb)
end

hotcross_table() = isfile(HOTCROSS) ? load_hotcross(HOTCROSS) :
    HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))

# ---------------------------------------------------------------- legs
# A leg: its name, rate unit and a setup that uploads its inputs to every device once.
struct Leg
    name::String
    unit::String
    setup::Function            # devs_all -> (run = devs -> output, items = Int, variant)
end

res_of(r) = CPU_SMOKE ? 24 : r

# (uniform resample, AMR snapshot) of the imaging deck, loaded once per process
const FLUID = Ref{Any}(nothing)
function fluid()
    FLUID[] === nothing && (FLUID[] = CPU_SMOKE ? (synthetic_uniform(), synthetic_amr()) :
        (snap = AthenaKSnapshot(SNAP); (resample_fluid_uniform(snap, 40.0, 256), snap)))
    return FLUID[]
end

function imaging_legs()
    uni(name, res, f) = Leg(name, "rays/s", devs -> begin
        grid, _ = fluid(); dg = upload_grid(grid, devs); cfg = cfg_img(res_of(res))
        (run = d -> f(cfg, dg; devices = d), items = res_of(res)^2, variant = "")
    end)
    amr(name, res, f) = Leg(name, "rays/s", devs -> begin
        _, snap = fluid(); dg = upload_amr(snap, devs; upload_bounds = false)
        cfg = cfg_img(res_of(res))
        (run = d -> f(cfg, dg; devices = d), items = res_of(res)^2, variant = "")
    end)
    driver = Leg("amr_driver_pol", "rays/s", devs -> begin
        _, snap = fluid(); dg = upload_amr(snap, devs; upload_bounds = false)
        cfg = cfg_img(CPU_SMOKE ? 16 : 384)
        st = KerrSchildCartesian(cfg.bh_a)
        cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                                  cfg.camera.fov, cfg.camera.resolution)
        # the streamed device marcher where the checkout has it, else the level closure
        streamed = @isdefined(AMRStokes)
        stokes(d) = streamed ? AMRStokes(cfg, dg; polarized = true, devices = d) :
            coords -> march_rays_gpu_amr(cfg, dg, camera_rays_at(st, cam, coords);
                                         polarized = true, devices = d)
        run(d) = render_amr(cfg, SNAP; block = 8, max_level = 2, polarized = true,
                            stokes = stokes(d))
        (run = run, items = run(devs[1:1]).nrays, variant = streamed ? "streamed" : "closure")
    end)
    return [uni("fused_384", 384, render_gpu_fused), uni("fused_768", 768, render_gpu_fused),
            uni("fused_pol_256", 256, render_gpu_polarized),
            uni("fused_pol_384", 384, render_gpu_polarized),
            amr("amr_384", 384, render_gpu_amr), amr("amr_pol_384", 384, render_gpu_amr_polarized),
            driver]
end

function mc_legs()
    thin = Leg("mc_thin", "photons/s", devs -> (
        run = d -> mc_spectrum_gpu(SphereModel(), MC_NS; devices = d),
        items = round(Int, MC_NS), variant = ""))
    scat = Leg("mc_scatter", "photons/s", devs -> begin
        tbl = hotcross_table()
        (run = d -> mc_spectrum_gpu(ScatteringSphere(), tbl, MC_NS; devices = d),
         items = round(Int, MC_NS), variant = isfile(HOTCROSS) ? "gpumonty" : "thomson")
    end)
    return [thin, scat]
end

# ---------------------------------------------------------------- output equality
out_bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
out_bits(x::AbstractArray{<:SVector}) = reduce(vcat, (out_bits(collect(v)) for v in x))
out_bits(x::NamedTuple) = reduce(vcat, (out_bits(v) for v in values(x) if v isa AbstractArray))
out_bits(x) = hasproperty(x, :total) ? out_bits(x.total) :
              hasproperty(x, :sumw) ? out_bits(x.sumw) : error("no bit view of $(typeof(x))")

# ---------------------------------------------------------------- CUPTI phases
# Union length of [start, stop] intervals.
function union_length(iv)
    isempty(iv) && return 0.0
    iv = sort(iv; by = first); tot = 0.0; s, e = iv[1]
    for (a, b) in iv[2:end]
        if a > e
            tot += e - s; s, e = a, b
        else
            e = max(e, b)
        end
    end
    return tot + (e - s)
end

# Activity class of a CUPTI device-trace row name.
function row_class(name)
    m = match(r"^\[copy (\S+) to (\S+) memory\]", name)
    m === nothing || return occursin("DEVICE", m[1]) ?
        (occursin("DEVICE", m[2]) ? :d2d : :d2h) : (occursin("DEVICE", m[2]) ? :h2d : :h2h)
    return startswith(name, "[") ? :other : :kernel
end

cupti_now() = (t = Ref{UInt64}(0); CUDA.CUPTI.cuptiGetTimestamp(t); t[] / 1e9)

# One profiled warm run -> PHASE records (seconds on the CUPTI clock).
function emit_cuda_phases(leg, run, devs)
    t0 = Ref(0.0); t1 = Ref(0.0)
    prof = CUDA.@profile (t0[] = cupti_now(); run(devs); t1[] = cupti_now())
    d = prof.device
    rows = [(start = d.start[i], stop = d.stop[i], dev = d.device[i], cls = row_class(d.name[i]))
            for i in eachindex(d.start)]
    base = (:leg => leg, :backend => BACKEND, :ndev => length(devs))
    isempty(rows) && return emit("PHASE", base..., :device => "all", :phase => "no_device_activity",
                                 :seconds => t1[] - t0[])
    emit("PHASE", base..., :device => "all", :phase => "host_prologue",
         :seconds => minimum(r.start for r in rows) - t0[])
    emit("PHASE", base..., :device => "all", :phase => "host_epilogue",
         :seconds => t1[] - maximum(r.stop for r in rows))
    for dev in sort!(unique(r.dev for r in rows))
        mine = filter(r -> r.dev == dev, rows)
        iv(c) = [(r.start, r.stop) for r in mine if r.cls === c]
        for (ph, c) in (("kernel", :kernel), ("h2d", :h2d), ("d2h", :d2h))
            emit("PHASE", base..., :device => dev, :phase => ph, :seconds => union_length(iv(c)))
        end
        busy = union_length([(r.start, r.stop) for r in mine])
        emit("PHASE", base..., :device => dev, :phase => "idle", :seconds => (t1[] - t0[]) - busy)
    end
end

# ---------------------------------------------------------------- driver
function run_leg(leg::Leg, sets)
    s = leg.setup(sets[end])
    ref = nothing; t_one = NaN
    for devs in sets
        t = time_ns(); out = s.run(devs); first_s = (time_ns() - t) / 1e9
        best = Inf
        for _ in 1:REPEAT
            t = time_ns(); out = s.run(devs); best = min(best, (time_ns() - t) / 1e9)
        end
        nd = length(devs)
        nd == 1 && (ref = out_bits(out); t_one = best)
        extra = nd == 1 ? () : (:speedup => t_one / best, :equal => out_bits(out) == ref)
        emit("BENCH", :leg => leg.name, :backend => BACKEND, :ndev => nd, :items => s.items,
             :wall_s => best, :first_s => first_s, :rate => s.items / best, :unit => leg.unit,
             extra..., (isempty(s.variant) ? () : (:variant => s.variant,))...,
             :commit => COMMIT)
        CPU_SMOKE || emit_cuda_phases(leg.name, s.run, devs)
    end
end

# JSON array of the parsed records (flat objects of strings, numbers and booleans).
function write_json(path, recs)
    js(v) = v isa AbstractString ? repr(v) : v isa Bool ? string(v) :
            isfinite(v) ? string(v) : "null"
    open(path, "w") do io
        println(io, "[")
        for (i, r) in enumerate(recs)
            print(io, "  {", join(("\"$k\": $(js(v))" for (k, v) in pairs(r)), ", "), "}")
            println(io, i == length(recs) ? "" : ",")
        end
        println(io, "]")
    end
end

function main()
    println("# bench_multigpu backend=$BACKEND commit=$COMMIT threads=$(Threads.nthreads())")
    legs = [imaging_legs(); mc_legs()]
    sel = get(ENV, "BLP_LEGS", "")
    isempty(sel) || (legs = filter(l -> l.name in split(sel, ','), legs))
    if !CPU_SMOKE && !CUDA.functional()
        foreach(l -> emit("SKIP", :leg => l.name, :reason => "CUDA not functional"), legs)
    else
        sets = device_sets()
        for leg in legs
            if !CPU_SMOKE && startswith(leg.name, r"fused|amr") && !isfile(SNAP)
                emit("SKIP", :leg => leg.name, :reason => "missing snapshot $SNAP")
            else
                run_leg(leg, sets)
            end
        end
    end
    path = get(ENV, "BLP_OUT", "")
    isempty(path) || write_json(path, parse_bench(String(take!(LOG))))
    return 0
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main())
