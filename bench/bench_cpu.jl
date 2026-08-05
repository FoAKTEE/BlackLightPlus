# bench_cpu.jl — BlackLightPlus CPU benchmark: three legs in one julia process.
#   imaging    (rays/s)    : render() of a GRMHD torus deck (a=0.9, 230 GHz), res in BLP_RESES
#   montecarlo (superph/s) : mc_spectrum_scatter on ScatteringSphere, Ns = BLP_MC_NS
#   analytic   (rays/s)    : render_analytic, exact-Kerr g-disk (a=0.94, θo=17°), res BLP_AN_RES
# Each timed leg emits one machine-parseable line (parse with grep '^BENCH'):
#   BENCH <arena> <code> <hw> <threads> <problem> <res_or_Ns> <wall_s> <rate> <units>
# Usage (thread count fixed at startup): OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
#   julia --project=. -t <N> bench/bench_cpu.jl
# Data: BLP_TEST_GRMHD (.athdf snapshot), BLP_REF_GPUMONTY (hotcross.dat); missing => leg skipped.
using StaticArrays, Printf, Statistics
using BlackLightPlus
using BlackLightPlus: ScatteringSphere, mc_spectrum_scatter, load_hotcross,
                    EquatorialDisk, render_analytic

const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const HOTDAT = get(ENV, "BLP_REF_GPUMONTY",
    "/data/haiyangw/claude/Mahakala/ref-code/gpumonty/table/hotcross.dat")
const RESES  = [parse(Int, r) for r in split(strip(get(ENV, "BLP_RESES", "192 384 768")))]
const MC_NS  = round(Int, parse(Float64, get(ENV, "BLP_MC_NS", "1e6")))
const AN_RES = parse(Int, get(ENV, "BLP_AN_RES", "1024"))
const SEED   = 0xC0FFEE
const NCORE  = Threads.nthreads()

timed(f) = (GC.gc(); t0 = time(); r = f(); (time() - t0, r))
bestof(f, reps) = (b = Inf; local r; for _ in 1:reps; (dt, r) = timed(f); b = min(b, dt); end; (b, r))

# ------------------------------------------------------------------ imaging config (torus, 230 GHz)
cfg_img(res) = RunConfig(bh_a=0.9, m_bh_cgs=6.2e9 * 1.989e33, mass_scale_cgs=1.0e24,
                         frequency_hz=230.0e9,
                         camera=CameraConfig(inclination_deg=60.0, distance=1000.0,
                                             fov=40.0, resolution=res),
                         plasma=PlasmaConfig(rat_high=40.0, rat_low=1.0, gamma=13.0 / 9.0,
                                             use_p=false, sigma_cut=100.0, theta_e_min=0.3))

# ------------------------------------------------------------------ analytic config (exact-Kerr g-disk)
cfg_analytic(res) = RunConfig(bh_a=0.94, tracer=:analytic,
                              camera=CameraConfig(inclination_deg=17.0, distance=1.0e4,
                                                  fov=20.0, resolution=res))

function arena_imaging()
    isfile(DATA) || (@printf("# imaging skip: missing GRMHD snapshot %s (set BLP_TEST_GRMHD)\n", DATA); return)
    render(cfg_img(16), DATA; polarized=false)                        # JIT warmup
    for res in RESES
        cfg = cfg_img(res); nrays = res^2
        reps = (res >= 768 || (NCORE <= 2 && res >= 384)) ? 1 : 2
        best, img = bestof(() -> render(cfg, DATA; polarized=false), reps)
        @printf("BENCH imaging BlackLightPlus cpu %d torus_a0.9_230GHz %d %.4f %.1f rays/s\n",
                NCORE, res, best, nrays / best); flush(stdout)
    end
end

function arena_mc()
    isfile(HOTDAT) || (@printf("# MC skip: missing hotcross table %s (set BLP_REF_GPUMONTY)\n", HOTDAT); return)
    tbl = load_hotcross(HOTDAT)
    sph = ScatteringSphere(); K = 3
    mc_spectrum_scatter(sph, tbl, 20_000; seed=SEED, max_scatt=K)     # JIT warmup
    reps = NCORE <= 4 ? 1 : 2
    best, r = bestof(() -> mc_spectrum_scatter(sph, tbl, MC_NS; seed=SEED, max_scatt=K), reps)
    ntr = r.n_primary + r.n_children
    @printf("BENCH montecarlo BlackLightPlus cpu %d scatter_sphere Ns%d %.4f %.1f superph/s\n",
            NCORE, MC_NS, best, ntr / best); flush(stdout)
end

function arena_analytic()
    disk = EquatorialDisk(0.94)
    render_analytic(cfg_analytic(64); source=disk)                    # JIT warmup
    cfg = cfg_analytic(AN_RES); nrays = AN_RES^2
    reps = NCORE <= 2 ? 1 : 2
    best, img = bestof(() -> render_analytic(cfg; source=disk), reps)
    @printf("BENCH analytic BlackLightPlus cpu %d gdisk_a0.94_i17 %d %.4f %.1f rays/s\n",
            NCORE, AN_RES, best, nrays / best); flush(stdout)
end

@printf("# bench_cpu.jl  threads=%d  MC_NS=%d  AN_RES=%d  RESES=%s\n",
        NCORE, MC_NS, AN_RES, string(RESES))
arena_imaging()
arena_mc()
arena_analytic()
@printf("# bench_cpu.jl DONE (threads=%d)\n", NCORE)
