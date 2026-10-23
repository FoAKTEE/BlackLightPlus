# fig_pol_hires.jl — polarized GRMHD image (Stokes I, Q, U, V) of an AthenaK torus snapshot.
# Physics: Kerr spin a = 0.9, hole mass M = 6.2e9 Msun, density mass scale 1e-21 L³ with
# L = GM/c², 230 GHz, inclination 60°, camera distance 1000 GM/c², full field of view
# 30 GM/c²; electrons: R_low = 1, R_high = 10, adiabatic index 13/9, total-pressure
# matching, sigma cut 1, no Theta_e floor; step control C = 80, at most 2500 steps, escape
# at 1.1 × the camera distance. The image is the fused IQUV march of `render` on the
# snapshot's native blocks, run on CPU threads (backend = :cpu).
# Writes to figures/paper/out/pol_hires/:
#   {I,Q,U,V}.npy   RES × RES, laid out [iβ, iα] (numpy axis 0 = β, axis 1 = α)
#   meta.txt        line 1 inclination; line 2 res, backend and the intensity-weighted
#                   linear polarization LP = Σ√(Q²+U²)/ΣI; line 3 git=<code SHA>;
#                   line 4 threads and the wall time of the render
# Env: BLP_FIG_RES (default 384), BLP_TEST_GRMHD (snapshot path; default the torus dump
# torus.mhd_w_bcc.00999.athdf below).
# Run from the repository root: julia -t 32 --project=. figures/paper/fig_pol_hires.jl

using Printf
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN

const RES = parse(Int, get(ENV, "BLP_FIG_RES", "384"))
const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const OUT = joinpath(@__DIR__, "out", "pol_hires")
# code version recorded in meta.txt: HEAD SHA, "-dirty" if src has uncommitted edits
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

function main()
    mkpath(OUT)
    m_bh = 6.2e9 * MSUN
    L = GNEWT * m_bh / CL^2
    cfg = RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = 2.3e11,
                    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                          fov = 30.0, resolution = RES),
                    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                          use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
                    max_steps = 2500, C = 80.0, escape_factor = 1.1)
    snap = AthenaKSnapshot(DATA)
    plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
    img = render(plan; backend = :cpu)
    dt = img.run.seconds
    I, Q, U, V = img.I, img.Q, img.U, img.V
    lp = sum(sqrt.(Q .^ 2 .+ U .^ 2)) / sum(I)
    cp = sum(abs.(V)) / sum(I)
    for (name, A) in (("I", I), ("Q", Q), ("U", U), ("V", V))
        write_npy(joinpath(OUT, "$name.npy"), A)
    end
    open(joinpath(OUT, "meta.txt"), "w") do io
        println(io, 60)
        println(io, "res=$RES backend=cpu LP=$(round(lp, digits=4))")
        println(io, "git=$GIT_SHA")
        @printf(io, "threads=%d seconds=%.1f\n", Threads.nthreads(), dt)
    end
    @printf("pol_hires: %d^2  cpu  %.1f s on %d threads  sum(I)=%.4e  LP=%.3f  |V|/I=%.4f\n",
            RES, dt, Threads.nthreads(), sum(I), lp, cp)
end

main()
