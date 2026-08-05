# bench_gpu.jl — BlackLightPlus GPU benchmark: imaging + analytic legs on 1 or 2 CUDA devices.
#   imaging  (rays/s) : render_gpu_fused of a GRMHD torus deck (a=0.9, 230 GHz), res in
#                       BLP_RESES; one-time fluid resample+upload excluded from timing.
#   analytic (rays/s) : render_analytic_gpu, exact-Kerr g-disk (a=0.94, θo=17°), res
#                       BLP_AN_RES; validated against the CPU render_analytic before timing.
# Emits one machine-parseable line per timed leg (parse with grep '^BENCH'):
#   BENCH <arena> BlackLightPlus gpu <ngpu> <problem> <res> <wall_s> <rate> rays/s
# Usage: CUDA_VISIBLE_DEVICES=0 BLP_NGPU=1 julia --project=. bench/bench_gpu.jl
#   (BLP_NGPU=2 uses devices [0,1]; set CUDA_VISIBLE_DEVICES accordingly)
using StaticArrays, Printf, Statistics
using BlackLightPlus
using BlackLightPlus: AthenaKSnapshot, EquatorialDisk, render_analytic
using BlackLightPlus.Simulation: AthenaKSnapshot
using CUDA, KernelAbstractions, JacobiElliptic

const REPO = dirname(@__DIR__)
include(joinpath(REPO, "src", "gpu_image.jl"))
include(joinpath(REPO, "src", "gpu_analytic.jl"))

const DATA  = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const RESES = [parse(Int, r) for r in split(strip(get(ENV, "BLP_RESES", "192 384 768")))]
const NGPU  = parse(Int, get(ENV, "BLP_NGPU", "1"))
const AN_RES = parse(Int, get(ENV, "BLP_AN_RES", "1024"))
const DEVS  = NGPU == 2 ? [0, 1] : [0]

cfg_img(res) = RunConfig(bh_a=0.9, m_bh_cgs=6.2e9 * 1.989e33, mass_scale_cgs=1.0e24,
                         frequency_hz=230.0e9,
                         camera=CameraConfig(inclination_deg=60.0, distance=1000.0,
                                             fov=40.0, resolution=res),
                         plasma=PlasmaConfig(rat_high=40.0, rat_low=1.0, gamma=13.0 / 9.0,
                                             use_p=false, sigma_cut=100.0, theta_e_min=0.3))
cfg_analytic(res) = RunConfig(bh_a=0.94, tracer=:analytic,
                              camera=CameraConfig(inclination_deg=17.0, distance=1.0e4,
                                                  fov=20.0, resolution=res))
timed(f) = (t0 = time(); r = f(); (time() - t0, r))

maxrel(g, c) = begin
    m = 0.0
    for i in eachindex(c)
        (isfinite(c[i]) && isfinite(g[i]) && (c[i] != 0 || g[i] != 0)) &&
            (m = max(m, abs(g[i] - c[i]) / max(abs(c[i]), 1e-300)))
    end; m
end

function arena_imaging()
    isfile(DATA) || (@printf("# imaging skip: missing GRMHD snapshot %s (set BLP_TEST_GRMHD)\n", DATA); return)
    snap = AthenaKSnapshot(DATA)
    grid = resample_fluid_uniform(snap, 40.0, 256)          # one-time preprocess
    dgrid = upload_grid(grid, DEVS)
    render_gpu_fused(cfg_img(16), dgrid; devices=DEVS)       # JIT warmup
    for res in RESES
        cfg = cfg_img(res); nrays = res^2
        best = Inf; local img
        for _ in 1:3; (dt, img) = timed(() -> render_gpu_fused(cfg, dgrid; devices=DEVS)); best = min(best, dt); end
        @printf("BENCH imaging BlackLightPlus gpu %d torus_a0.9_230GHz %d %.5f %.1f rays/s\n",
                NGPU, res, best, nrays / best); flush(stdout)
    end
end

function arena_analytic()
    disk = EquatorialDisk(0.94)
    # --- validate GPU vs CPU (256^2, worst relative pixel diff) ---
    cfgv = cfg_analytic(256)
    cpu = render_analytic(cfgv; source=disk)
    gpu = render_analytic_gpu(cfgv; devices=DEVS, source=disk)
    mI = maxrel(gpu.I, cpu.I)
    @printf("# analytic GPU-vs-CPU validation (256^2): worst rel image diff = %.2e  %s\n",
            mI, mI < 1e-8 ? "MATCH" : "WARN")
    # --- timed leg at AN_RES^2 ---
    render_analytic_gpu(cfg_analytic(AN_RES); devices=DEVS, source=disk)   # warm
    cfg = cfg_analytic(AN_RES); nrays = AN_RES^2
    best = Inf
    for _ in 1:5; (dt, _) = timed(() -> render_analytic_gpu(cfg; devices=DEVS, source=disk)); best = min(best, dt); end
    @printf("BENCH analytic BlackLightPlus gpu %d gdisk_a0.94_i17 %d %.5f %.1f rays/s\n",
            NGPU, AN_RES, best, nrays / best); flush(stdout)
end

@printf("# bench_gpu.jl  NGPU=%d devices=%s  CUDA.ndevices=%d  RESES=%s\n",
        NGPU, string(DEVS), CUDA.ndevices(), string(RESES))
for d in DEVS; CUDA.device!(d); @printf("#   dev[%d] = %s\n", d, CUDA.name(CUDA.device())); end
arena_imaging()
arena_analytic()
@printf("# bench_gpu.jl DONE (NGPU=%d)\n", NGPU)
