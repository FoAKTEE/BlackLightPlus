# bench_multigpu.jl — phase-timed multi-device benchmark (results/blp/multigpu/design.md §7).
#
# Every leg runs on 1 and on n devices (CUDA [0] vs [0, 1], or HostDevices on the KA CPU
# backend) and prints bench_lines.jl records:
#   BENCH  warm best-of-BLP_REPEAT wall time, first-call (JIT-inclusive) time, rate,
#          speedup T1/Tn and bitwise equality of the n-device output with the 1-device one;
#   PHASE  (CUDA) from one CUPTI-profiled warm run: host prologue / epilogue, and per device
#          kernel busy, host-to-device, device-to-host and idle time;
#   SKIP   one line per leg that cannot run (no functional CUDA, missing snapshot).
# Legs, Kerr: fused unpolarized 384² / 768² (fp64) and 768² (`precision = :mixed`), fused
# polarized 256² / 384², native AMR I / IQUV at 384², the AMR driver on the production deck
# (a = 0.9, 230 GHz, i = 60°, base 384, 2 levels, polarized; streamed device marcher where
# the checkout has one, and the per-level closure), the analytic exact-Kerr g-disk at 1024²,
# Monte Carlo thin / scatter at Ns = 1e7. Binary (SuperposedKerrSchild, generic-spacetime
# kernels, scan-locate AMR fluid): the figures/fig_eht_bbh.jl deck (i = 17°, 480², FOV 56 M)
# on dump 00455 unpolarized and polarized and on dump 00122 (a/m = 0.8) polarized, and the
# figures/fig_bbh_i90_pol_amr.jl deck (i = 90°, base 256, block 8, 4 levels) through
# render_amr with a polarized device closure.
# Every BENCH record also carries the Slurm job id, the 1-minute host load and the unix
# start / end times of the leg (for matching an external nvidia-smi memory log).
#
# Usage (from the checkout under test; the harness loads that checkout's src/):
#   julia --project=. -t 8 bench/bench_multigpu.jl                     # CUDA, 2 GPUs
#   BLP_BACKEND=cpu julia --project=. -t 8 bench/bench_multigpu.jl     # KA CPU smoke test
# To time another commit, run this file from a worktree of that commit:
#   git worktree add /tmp/base 3736b32 && cd /tmp/base &&
#   BLP_OUT=<repo>/results/blp/multigpu/baseline_3736b32.json julia --project=. -t 8 <repo>/bench/bench_multigpu.jl
# Environment: BLP_BACKEND (cuda|cpu), BLP_LEGS (comma list; default all), BLP_REPEAT (5),
# BLP_TEST_GRMHD (AthenaK snapshot for the imaging legs), BLP_BBH455 / BLP_BBH122 (binary
# dumps), BLP_REF_GPUMONTY (hot-cross table; Thomson limit otherwise), BLP_MC_NS (1e7),
# BLP_OUT (JSON copy of the records), BLP_SAVE_DIR (write each leg's 1-device output there
# as raw Float64 arrays `<leg>.<field>.<n1>x<n2>.f64`, for cross-commit image comparison).
# Legs whose first warm run exceeds 5 s are timed best-of-3 instead of best-of-BLP_REPEAT.
# The CPU backend uses a synthetic torus deck at 24² and Ns = 2e4: it checks that every
# leg runs and is device-count invariant; its times are not GPU predictions.
using BlackLightPlus, StaticArrays, Printf, KernelAbstractions, CUDA, JacobiElliptic, HDF5
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays_at, SphereModel,
                      ScatteringSphere, HotcrossTable, load_hotcross, render_amr,
                      SuperposedKerrSchild, EquatorialDisk, athenak_bin_snapshot
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot
using BlackLightPlus.Constants: SIGMA_THOMSON
using BlackLightPlus.MonteCarlo: NW, NT

const SRC = joinpath(dirname(Base.active_project()), "src")
include(joinpath(@__DIR__, "bench_lines.jl"))
for f in ("gpu_image.jl", "gpu_pol.jl", "gpu_amr.jl", "gpu_spacetime.jl", "gpu_analytic.jl",
          joinpath("montecarlo", "mc_gpu.jl"))
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
const BBH455 = get(ENV, "BLP_BBH455", "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/" *
                   "SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const BBH122 = get(ENV, "BLP_BBH122", "/data2/haiyangw/aurora/cbd_rel/torus.mhd_w_bcc.00122.athdf")
const SAVE_DIR = get(ENV, "BLP_SAVE_DIR", "")
const JOB = get(ENV, "SLURM_JOB_ID", "none")
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

# The production AMR deck through the level-by-level closure (host rays per level), on any
# checkout: with `amr_driver_pol` it separates the streamed driver from the kernel time.
function driver_closure_leg()
    Leg("amr_driver_pol_closure", "rays/s", devs -> begin
        _, snap = fluid(); dg = upload_amr(snap, devs; upload_bounds = false)
        cfg = cfg_img(CPU_SMOKE ? 16 : 384)
        st = KerrSchildCartesian(cfg.bh_a)
        cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                                  cfg.camera.fov, cfg.camera.resolution)
        stokes(d) = coords -> march_rays_gpu_amr(cfg, dg, camera_rays_at(st, cam, coords);
                                                 polarized = true, devices = d)
        run(d) = render_amr(cfg, SNAP; block = 8, max_level = 2, polarized = true,
                            stokes = stokes(d))
        (run = run, items = run(devs[1:1]).nrays, variant = "closure")
    end)
end

# fp32 geodesic lane with fp64 rescue of the near-ring pixels (dag.md MG09).
mixed_leg() = Leg("fused_mixed_768", "rays/s", devs -> begin
    grid, _ = fluid(); dg = upload_grid(grid, devs); cfg = cfg_img(res_of(768))
    (run = d -> render_gpu_fused(cfg, dg; devices = d, precision = :mixed),
     items = res_of(768)^2, variant = "")
end)

# Exact-Kerr equatorial g-disk (bench_gpu.jl deck: a = 0.94, i = 17°, FOV 20 M), layered I.
cfg_analytic(res) = RunConfig(bh_a = 0.94, tracer = :analytic,
    camera = CameraConfig(inclination_deg = 17.0, distance = 1.0e4, fov = 20.0, resolution = res))
analytic_leg() = Leg("analytic_1024", "rays/s", devs -> begin
    cfg = cfg_analytic(CPU_SMOKE ? 16 : 1024); disk = EquatorialDisk(0.94)
    (run = d -> render_analytic_gpu(cfg; devices = d, source = disk).I,
     items = cfg.camera.resolution^2, variant = "")
end)

# ---------------------------------------------------------------- binary decks
# AthenaK snapshot of an athdf dump, with or without the press layout: checkouts whose
# reader predates the `gamma` keyword (c2f615b) read `dens velx vely velz press
# [temperature]` here with the same conversion (eint = press/(gamma-1) in Float64, one
# Float32 rounding, non-finite cells zeroed); the result is tagged `adapter_reader`.
function load_snapshot(path, gamma)
    endswith(path, ".bin") && return athenak_bin_snapshot(path; gamma = gamma), ""
    try
        return AthenaKSnapshot(path; gamma = gamma), ""
    catch err
        err isa MethodError || rethrow()
    end
    uov, B, x1v, x2v, x3v, x1f, x2f, x3f, lev, loc, vn = h5open(path, "r") do h
        (read(h["uov"]), read(h["B"]), read(h["x1v"]), read(h["x2v"]), read(h["x3v"]),
         read(h["x1f"]), read(h["x2f"]), read(h["x3f"]), read(h["Levels"]),
         read(h["LogicalLocations"]), String.(read_attribute(h, "VariableNames")))
    end
    vn[1:5] == ["dens", "velx", "vely", "velz", "press"] || error("unexpected layout $vn")
    ni, nj, nk, nmb, _ = size(uov); gm1 = Float64(gamma) - 1.0
    data = Array{Float32,5}(undef, ni, nj, nk, nmb, 8)
    Threads.@threads for m in 1:nmb
        @inbounds for q in 1:8, k in 1:nk, j in 1:nj, i in 1:ni
            v = q <= 5 ? uov[i, j, k, m, q] : B[i, j, k, m, q - 5]
            isfinite(v) || (v = 0.0f0)
            data[i, j, k, m, q] = q == 5 ? Float32(Float64(v) / gm1) : v
        end
    end
    snap = AthenaKSnapshot(data, [SVector(x1f[1, m], x2f[1, m], x3f[1, m]) for m in 1:nmb],
        [SVector(x1f[end, m], x2f[end, m], x3f[end, m]) for m in 1:nmb],
        [SVector(x1v[1, m], x2v[1, m], x3v[1, m]) for m in 1:nmb],
        [SVector(x1v[2, m] - x1v[1, m], x2v[2, m] - x2v[1, m], x3v[2, m] - x3v[1, m])
         for m in 1:nmb], Int.(lev), Int.(loc),
        ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"], ni, nj, nk, nmb)
    return snap, "adapter_reader"
end

# The EHT deck of figures/fig_eht_bbh.jl: each hole 6.5e9 Msun (code mass = the binary),
# 230 GHz, i = 17°, FOV 56 M, 480², D = 1000 M, C = 80, 4000 steps, R_high = 40, mu = 0.5,
# sigma_cut = 1, the calibrated 1 Jy mass scale; the metric at the dump's orbital phase.
# Centres, spins and mass scales are those recorded by that script in
# figures/out/eht_bbh/<dump>_i17_Rh40/records.json.
const BBH_DECKS = Dict(
    "s455" => (path = () -> BBH455, a = (0.0, 0.0), ms = 2.274019497e28,
               c = ((12.1912068, -2.761245515, 0.0), (-12.1912068, 2.761245515, 0.0))),
    "s122" => (path = () -> BBH122, a = (0.4, 0.4), ms = 1.000498515e21,
               c = ((0.321593038, 7.993533506, 0.0), (-0.321593038, -7.993533506, 0.0))))
cfg_bbh(ms, res; inc = 17.0, fov = 56.0) = RunConfig(bh_a = 0.0,
    m_bh_cgs = 2 * 6.5e9 * 1.98847e33, mass_scale_cgs = ms, frequency_hz = 230.0e9,
    polarized = true, max_steps = 4000,
    camera = CameraConfig(inclination_deg = inc, distance = 1000.0, fov = fov, resolution = res),
    plasma = PlasmaConfig(gamma = 13.0 / 9.0, rat_low = 1.0, rat_high = 40.0, mu = 0.5,
                          sigma_cut = 1.0))

# (snapshot, reader tag) of a binary dump, one at a time (each is ~5 GB on the host)
const BBH_SNAP = Ref{Any}(nothing)
function bbh_snapshot(tag)
    CPU_SMOKE && return fluid()[2], ""
    path = BBH_DECKS[tag].path()
    (BBH_SNAP[] === nothing || BBH_SNAP[][1] != path) &&
        (BBH_SNAP[] = nothing; GC.gc(); BBH_SNAP[] = (path, load_snapshot(path, 13.0 / 9.0)...))
    return BBH_SNAP[][2], BBH_SNAP[][3]
end
bbh_metric(tag) = CPU_SMOKE ?
    SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)]) :
    SuperposedKerrSchild([0.5, 0.5], collect(BBH_DECKS[tag].a), collect(BBH_DECKS[tag].c))

bbh_leg(name, tag, polarized) = Leg(name, "rays/s", devs -> begin
    snap, reader = bbh_snapshot(tag)
    sg = upload_amr_scan(snap, devs)
    st = bbh_metric(tag)
    cfg = CPU_SMOKE ? cfg_bbh(BBH_DECKS[tag].ms, 12; fov = 16.0) : cfg_bbh(BBH_DECKS[tag].ms, 480)
    (run = d -> render_gpu_spacetime(cfg, st, sg; polarized = polarized, devices = d),
     items = cfg.camera.resolution^2, variant = reader)
end)

# The i = 90° AMR deck of figures/fig_bbh_i90_pol_amr.jl on dump 00455: equal-mass
# non-spinning holes at ±12.5 (cos φ0, sin φ0, 0), φ0 = -0.2233022; 230 GHz, FOV 40 M,
# 2500 steps, mass scale 1e24 g, base 256, block 8, 4 levels, abs_grad_cut from that
# run's level-0 probe (figures/out/bbh_i90_pol_amr/records.json), abs_grad_count = 5.
i90_leg() = Leg("bbh_i90_amr_pol", "rays/s", devs -> begin
    snap, reader = bbh_snapshot("s455")
    sg = upload_amr_scan(snap, devs)
    φ0 = -0.2233022; c1 = 12.5 .* (cos(φ0), sin(φ0), 0.0)
    st = CPU_SMOKE ? bbh_metric("s455") :
         SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [c1, (-c1[1], -c1[2], 0.0)])
    base = CPU_SMOKE ? 16 : 256; lmax = CPU_SMOKE ? 1 : 4
    cfg = RunConfig(bh_a = 0.0, m_bh_cgs = 6.5e9 * 1.989e33, mass_scale_cgs = 1.0e24,
                    frequency_hz = 2.3e11, polarized = true, max_steps = 2500,
                    camera = CameraConfig(inclination_deg = 90.0, distance = 1000.0,
                                          fov = CPU_SMOKE ? 16.0 : 40.0, resolution = base),
                    plasma = PlasmaConfig(gamma = 13.0 / 9.0))
    cam = PlaneParallelCamera(90.0, 1000.0, cfg.camera.fov, base)
    stokes(d) = coords -> march_rays_gpu_spacetime(cfg, st, sg, camera_rays_at(st, cam, coords);
                                                   polarized = true, devices = d)
    run(d) = render_amr(cfg, ""; block = 8, max_level = lmax, polarized = true,
                        stokes = stokes(d), abs_grad_cut = CPU_SMOKE ? nothing : 5.282132294e-13,
                        abs_grad_count = 5)
    (run = run, items = run(devs[1:1]).nrays, variant = isempty(reader) ? "closure" : reader)
end)

binary_legs() = [bbh_leg("bbh455_480", "s455", false), bbh_leg("bbh455_pol_480", "s455", true),
                 i90_leg(), bbh_leg("bbh122_pol_480", "s122", true)]

# ---------------------------------------------------------------- saved outputs
# Raw Float64 arrays `<leg>.<field>.<n1>x<n2>.f64` (column-major) for image comparison.
function save_output(dir, leg, out)
    mkpath(dir)
    fields = out isa NamedTuple ? pairs(out) : (:out => out,)
    for (k, v) in fields
        v isa AbstractArray{<:Real} || continue
        write(joinpath(dir, "$leg.$k.$(join(size(v), 'x')).f64"), Float64.(v))
    end
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
load1() = try parse(Float64, split(read("/proc/loadavg", String))[1]) catch; NaN end

function run_leg(leg::Leg, sets)
    s = leg.setup(sets[end])
    ref = nothing; t_one = NaN
    for devs in sets
        t_start = time(); l1 = load1()
        t = time_ns(); out = s.run(devs); first_s = (time_ns() - t) / 1e9
        best = Inf; nrep = REPEAT; k = 0
        while k < nrep
            t = time_ns(); out = s.run(devs); best = min(best, (time_ns() - t) / 1e9)
            k += 1
            k == 1 && best > 5.0 && (nrep = min(nrep, 3))
        end
        t_end = time()
        nd = length(devs)
        nd == 1 && (ref = out_bits(out); t_one = best)
        nd == 1 && !isempty(SAVE_DIR) && save_output(SAVE_DIR, leg.name, out)
        extra = nd == 1 ? () : (:speedup => t_one / best, :equal => out_bits(out) == ref)
        emit("BENCH", :leg => leg.name, :backend => BACKEND, :ndev => nd, :items => s.items,
             :wall_s => best, :first_s => first_s, :rate => s.items / best, :unit => leg.unit,
             extra..., (isempty(s.variant) ? () : (:variant => s.variant,))...,
             :commit => COMMIT, :nrep => nrep, :job => JOB, :load1 => l1,
             :t_start => floor(Int, t_start), :t_end => ceil(Int, t_end))
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
    println("# bench_multigpu backend=$BACKEND commit=$COMMIT threads=$(Threads.nthreads()) job=$JOB")
    legs = [imaging_legs(); mixed_leg(); driver_closure_leg(); analytic_leg(); mc_legs();
            binary_legs()]
    sel = get(ENV, "BLP_LEGS", "")
    isempty(sel) || (legs = filter(l -> l.name in split(sel, ','), legs))
    if !CPU_SMOKE && !CUDA.functional()
        foreach(l -> emit("SKIP", :leg => l.name, :reason => "CUDA not functional"), legs)
    else
        sets = device_sets()
        for leg in legs
            need = CPU_SMOKE ? "" : startswith(leg.name, r"fused|amr") ? SNAP :
                   startswith(leg.name, r"bbh455|bbh_i90") ? BBH455 :
                   startswith(leg.name, "bbh122") ? BBH122 : ""
            if !isempty(need) && !isfile(need)
                emit("SKIP", :leg => leg.name, :reason => "missing snapshot $need")
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
