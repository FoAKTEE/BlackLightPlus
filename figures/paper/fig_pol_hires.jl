# fig_pol_hires.jl — polarized GRMHD image (Stokes I, Q, U, V) of an AthenaK torus snapshot.
# Physics: Kerr spin a = 0.9, the M87* scaling of m87_scaling.jl (hole mass 6.5e9 Msun,
# 16.8 Mpc, the density scale of run 00999 at which the unpolarized 230 GHz image seen from
# 17° holds 0.5 Jy), 230 GHz, inclination 60°, camera distance 1000 GM/c², full field of
# view 30 GM/c²; electrons: R_low = 1, R_high = 10, adiabatic index 13/9, total-pressure
# matching, sigma cut 1, no Theta_e floor; step control C = 80, at most 2500 C/80 steps,
# escape at 1.1 × the camera distance. The image is the fused IQUV march of `render` on
# the snapshot's native blocks, run on CPU threads (backend = :cpu). `pol_config` is the
# configuration that m87_calibrate.jl calibrates at 17° (including this file runs nothing).
# Writes to figures/paper/out/pol_hires/:
#   {I,Q,U,V}.npy   RES × RES, laid out [iβ, iα] (numpy axis 0 = β, axis 1 = α)
#   meta.txt        line 1 inclination; line 2 res, backend and the intensity-weighted
#                   linear polarization LP = Σ√(Q²+U²)/ΣI; line 3 git=<code SHA>;
#                   line 4 threads and the wall time of the render; line 5 the density
#                   scale, the step divisor and the Stokes I flux density in Jy
# Env: BLP_FIG_RES (default 384), BLP_FIG_C (default 80), BLP_TEST_GRMHD (snapshot path;
# default the torus dump torus.mhd_w_bcc.00999.athdf below).
# Run from the repository root: julia -t 32 --project=. figures/paper/fig_pol_hires.jl

using Printf
using BlackLightPlus
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

const RES = parse(Int, get(ENV, "BLP_FIG_RES", "384"))
const CVAL = parse(Float64, get(ENV, "BLP_FIG_C", "80"))
const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const OUT = joinpath(@__DIR__, "out", "pol_hires")
const SPIN = 0.9
const INC = 60.0
const FOV = 30.0
# code version recorded in meta.txt: HEAD SHA, "-dirty" if src has uncommitted edits
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

"The run configuration of the torus image at inclination `inc`, `res`² pixels over `fov`, step divisor `C`, density scale `ms`."
pol_config(; inc = INC, fov = FOV, res = RES, C = CVAL, ms = m87_density_scale("00999")) =
    RunConfig(m_bh_cgs = M87_MASS_G, mass_scale_cgs = ms, frequency_hz = M87_FREQUENCY_HZ,
              camera = CameraConfig(inclination_deg = inc, distance = 1000.0, fov = fov,
                                    resolution = res),
              plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                    use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
              max_steps = round(Int, 2500 * C / 80), C = C, escape_factor = 1.1)

function main()
    mkpath(OUT)
    ms = m87_density_scale("00999")
    snap = AthenaKSnapshot(DATA)
    plan = prepare(pol_config(; ms); spacetime = KerrSchildCartesian(SPIN), matter = snap,
                   mode = IQUV())
    img = render(plan; backend = :cpu)
    dt = img.run.seconds
    I, Q, U, V = img.I, img.Q, img.U, img.V
    lp = sum(sqrt.(Q .^ 2 .+ U .^ 2)) / sum(I)
    cp = sum(abs.(V)) / sum(I)
    F = m87_flux_jy(I, FOV, M87_MASS_G)
    for (name, A) in (("I", I), ("Q", Q), ("U", U), ("V", V))
        write_npy(joinpath(OUT, "$name.npy"), A)
    end
    open(joinpath(OUT, "meta.txt"), "w") do io
        println(io, round(Int, INC))
        println(io, "res=$RES backend=cpu LP=$(round(lp, digits=4))")
        println(io, "git=$GIT_SHA")
        @printf(io, "threads=%d seconds=%.1f\n", Threads.nthreads(), dt)
        @printf(io, "mass_scale_g=%.10g C=%g flux_Jy=%.6g\n", ms, CVAL, F)
    end
    @printf("pol_hires: %d^2  cpu  %.1f s on %d threads  sum(I)=%.4e  F=%.4f Jy  LP=%.3f  |V|/I=%.4f\n",
            RES, dt, Threads.nthreads(), sum(I), F, lp, cp)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
