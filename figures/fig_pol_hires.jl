# fig_pol_hires.jl — high-resolution polarized GRMHD render (Stokes I, Q, U, V).
# Physics: a = 0.9, M = 6.2e9 Msun, mass scale 1e-21 L^3, 230 GHz, i = 60 deg,
# FOV = 30 GM/c^2, R_high = 10. Backend: fused GPU-AMR polarized kernel when a
# CUDA context is available, else the CPU native-AMR polarized march (render).
# Writes I/Q/U/V .npy and meta.txt to figures/out/pol_hires/ (plot with
# plot_pol_hires.py). Env: BLP_FIG_RES (default 1024), BLP_TEST_GRMHD (snapshot).
# Run: julia --project=. -t auto figures/fig_pol_hires.jl

using Printf
using BlackLightPlus
using BlackLightPlus.Constants: GNEWT, CL
using BlackLightPlus.Simulation: AthenaKSnapshot
using CUDA

const RES = parse(Int, get(ENV, "BLP_FIG_RES", "1024"))
const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const OUT = joinpath(@__DIR__, "out", "pol_hires")
const SRC = joinpath(dirname(@__DIR__), "src")
# code version recorded in meta.txt: HEAD SHA, "-dirty" if src/ has uncommitted edits
const GIT_SHA = try
    root = dirname(@__DIR__)
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# GPU only if a context can actually be created (fails under exclusive-mode use).
const GPU_OK = try
    CUDA.functional() || error("CUDA not functional")
    CUDA.zeros(Float64, 1)
    true
catch
    false
end

if GPU_OK
    include(joinpath(SRC, "gpu_image.jl"))
    include(joinpath(SRC, "gpu_pol.jl"))
    include(joinpath(SRC, "gpu_amr.jl"))
end

function main()
    mkpath(OUT)
    m_bh = 6.2e9 * 1.989e33
    L = GNEWT * m_bh / CL^2
    cfg = RunConfig(bh_a=0.9, m_bh_cgs=m_bh, mass_scale_cgs=1.0e-21 * L^3,
                    frequency_hz=2.3e11,
                    camera=CameraConfig(inclination_deg=60.0, distance=1000.0,
                                        fov=30.0, resolution=RES),
                    plasma=PlasmaConfig(rat_low=1.0, rat_high=10.0, gamma=13.0 / 9.0,
                                        use_p=true, sigma_cut=1.0, theta_e_min=-1.0))
    snap = AthenaKSnapshot(DATA)
    backend = GPU_OK ? "gpu-amr-polarized" : "cpu-polarized-march"
    t0 = time()
    img = GPU_OK ? Base.invokelatest(render_gpu_amr_polarized, cfg, snap) :
          render(cfg, snap; polarized=true)
    dt = time() - t0
    I, Q, U, V = img.I, img.Q, img.U, img.V
    lp = sum(sqrt.(Q .^ 2 .+ U .^ 2)) / sum(I)
    cp = sum(abs.(V)) / sum(I)
    for (name, A) in (("I", I), ("Q", Q), ("U", U), ("V", V))
        write_npy(joinpath(OUT, "$name.npy"), A)
    end
    open(joinpath(OUT, "meta.txt"), "w") do io
        println(io, 60)
        println(io, "res=$RES backend=$backend LP=$(round(lp, digits=4))")
        println(io, "git=$GIT_SHA")
    end
    @printf("pol_hires: %d^2  %s  %.1f s  sum(I)=%.4e  LP=%.3f  |V|/I=%.4f\n",
            RES, backend, dt, sum(I), lp, cp)
end

main()
