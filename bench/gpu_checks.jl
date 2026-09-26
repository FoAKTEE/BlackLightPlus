# gpu_checks.jl — device-side correctness checks of the GPU acceptance runs
# (results/blp/multigpu/dag.md, MG08 and the GPU-vs-CPU parity of MG12/MG13).
#
#   rays    device camera rays (`CameraRays`, `_camray_kernel!`) vs the host `camera_rays`,
#           per state component: largest distance in units in the last place (ulp) and the
#           fraction of components within 2 ulp; Kerr 768² (a = 0.9, i = 60°) and the binary
#           00455 deck (480², i = 17°).
#   parity  GPU image vs the CPU pipeline on the same deck at 64²: CPU = camera_rays ->
#           trace_rays_sampled -> Radiation.(un)polarized_march over the snapshot sampler;
#           GPU = render_gpu_amr(_polarized) (Kerr, native AMR fluid) and
#           render_gpu_spacetime (binary, scan-locate fluid). Reported per Stokes component:
#           max |GPU - CPU| / max I_CPU, total-flux ratio and pixel correlation.
# One line per result:  GCHECK kind=<rays|parity> case=<name> key=value ...
#
# Usage (GPU node): julia --project=. -t 4 bench/gpu_checks.jl
# Environment: BLP_TEST_GRMHD, BLP_BBH455 (as bench_multigpu.jl), BLP_PARITY_RES (64).
using BlackLightPlus, StaticArrays, Printf, KernelAbstractions, CUDA, Statistics
using BlackLightPlus: KerrSchildCartesian, SuperposedKerrSchild, PlaneParallelCamera,
                      camera_rays, trace_rays_sampled, athenak_bin_snapshot
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot, sample_primitives, pgas_from_eint, units
const RAD = BlackLightPlus.Radiation

const SRC = joinpath(dirname(Base.active_project()), "src")
for f in ("gpu_image.jl", "gpu_pol.jl", "gpu_amr.jl", "gpu_spacetime.jl")
    include(joinpath(SRC, f))
end

const SNAP = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const BBH455 = get(ENV, "BLP_BBH455", "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/" *
                   "SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const PRES = parse(Int, get(ENV, "BLP_PARITY_RES", "64"))
const GAMMA = 13.0 / 9.0

gline(pairs...) = (println("GCHECK ", join(("$k=$(v isa AbstractFloat ? @sprintf("%.6g", v) : v)"
                                            for (k, v) in pairs), ' ')); flush(stdout))

# decks: bench_multigpu.jl's Kerr imaging deck and the fig_eht_bbh.jl deck on dump 00455
cfg_kerr(res) = RunConfig(bh_a = 0.9, m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1.0e24,
    frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 40.0, resolution = res),
    plasma = PlasmaConfig(rat_high = 40.0, rat_low = 1.0, gamma = GAMMA, use_p = false,
                          sigma_cut = 100.0, theta_e_min = 0.3))
cfg_bbh(res) = RunConfig(bh_a = 0.0, m_bh_cgs = 2 * 6.5e9 * 1.98847e33,
    mass_scale_cgs = 2.274019497e28, frequency_hz = 230.0e9, polarized = true, max_steps = 4000,
    camera = CameraConfig(inclination_deg = 17.0, distance = 1000.0, fov = 56.0, resolution = res),
    plasma = PlasmaConfig(gamma = GAMMA, rat_low = 1.0, rat_high = 40.0, mu = 0.5, sigma_cut = 1.0))
const ST_BBH = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0],
    [(12.1912068, -2.761245515, 0.0), (-12.1912068, 2.761245515, 0.0)])
camera_of(cfg) = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                                     cfg.camera.fov, cfg.camera.resolution)

# ---------------------------------------------------------------- device rays
ulp(a::Float64, b::Float64) = a == b ? 0 : signbit(a) != signbit(b) ?
    typemax(Int) : abs(reinterpret(Int64, a) - reinterpret(Int64, b))

function check_rays(name, st, cam)
    host = camera_rays(st, cam); n = length(host)
    pool = device_pool([0])
    dev = foreach_device(pool) do lane
        Array(view(rays_on_device!(lane, CameraRays(st, cam), 1:n), 1:n))
    end[1]
    u = [ulp(host[i][c], dev[i][c]) for i in 1:n for c in 1:8]
    gline(:kind => "rays", :case => name, :n => n, :max_ulp => maximum(u),
          :frac_le2ulp => count(<=(2), u) / length(u), :frac_equal => count(==(0), u) / length(u))
end

# ---------------------------------------------------------------- GPU vs CPU
function cpu_image(st, cfg, snap, polarized; R_em = Inf)
    cam = camera_of(cfg)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    scc = BlackLightPlus._prepared_scc(cfg, u)
    γ = cfg.plasma.gamma
    smp = x -> (s = sample_primitives(snap, x);
                (rho = s.rho, pgas = pgas_from_eint(s.eint, γ), uu = s.uu, bb = s.bb,
                 inside = s.inside))
    rays = camera_rays(st, cam)
    states, dls = trace_rays_sampled(st, rays; C = 80.0, nmax = cfg.max_steps,
                                     escape = 1.1 * cfg.camera.distance, R_em = R_em)
    S = zeros(4, length(rays)); up = BlackLightPlus._camera_up(cam)
    Threads.@threads for i in eachindex(rays)
        if polarized
            S[:, i] .= RAD.polarized_march(st, scc, smp, states[i], dls[i], cfg.frequency_hz,
                                           u.L_unit; cam_up = up)
        else
            S[1, i] = RAD.unpolarized_march(st, scc, smp, states[i], dls[i], cfg.frequency_hz,
                                            u.L_unit)
        end
    end
    np = cam.npix
    return polarized ? (I = reshape(S[1, :], np, np), Q = reshape(S[2, :], np, np),
                        U = reshape(S[3, :], np, np), V = reshape(S[4, :], np, np)) :
                       reshape(S[1, :], np, np)
end

function report_parity(name, gpu, cpu)
    g = gpu isa NamedTuple ? gpu : (I = gpu,)
    c = cpu isa NamedTuple ? cpu : (I = cpu,)
    scale = maximum(abs, c.I)
    for k in keys(c)
        a = vec(g[k]); b = vec(c[k])
        gline(:kind => "parity", :case => name, :stokes => k,
              :maxdiff_over_maxI => maximum(abs, a .- b) / scale,
              :flux_ratio => sum(b) == 0 ? NaN : sum(a) / sum(b), :corr => cor(a, b),
              :finite => all(isfinite, a))
    end
end

function main()
    CUDA.functional() || (println("GCHECK kind=skip reason=CUDA-not-functional"); return 0)
    println("# gpu_checks threads=$(Threads.nthreads()) device=$(CUDA.name(CUDA.device()))")
    check_rays("kerr_768", KerrSchildCartesian(0.9), camera_of(cfg_kerr(768)))
    check_rays("bbh455_480", ST_BBH, camera_of(cfg_bbh(480)))

    if isfile(SNAP)
        snap = AthenaKSnapshot(SNAP); cfg = cfg_kerr(PRES)
        dg = upload_amr(snap, [0]; upload_bounds = false)
        R_em = _gamr_domain_radius(dg.amrp); st = KerrSchildCartesian(0.9)
        report_parity("kerr_amr_I_$PRES", render_gpu_amr(cfg, dg; devices = [0]),
                      cpu_image(st, cfg, snap, false; R_em = R_em))
        report_parity("kerr_amr_IQUV_$PRES", render_gpu_amr_polarized(cfg, dg; devices = [0]),
                      cpu_image(st, cfg, snap, true; R_em = R_em))
        snap = dg = nothing; GC.gc()
    end
    if isfile(BBH455)
        snap = athenak_bin_snapshot(BBH455; gamma = GAMMA); cfg = cfg_bbh(PRES)
        sg = upload_amr_scan(snap, [0])
        report_parity("bbh455_I_$PRES", render_gpu_spacetime(cfg, ST_BBH, sg; devices = [0]),
                      cpu_image(ST_BBH, cfg, snap, false))
        report_parity("bbh455_IQUV_$PRES",
                      render_gpu_spacetime(cfg, ST_BBH, sg; polarized = true, devices = [0]),
                      cpu_image(ST_BBH, cfg, snap, true))
    end
    return 0
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main())
