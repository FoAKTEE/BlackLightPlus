# fig_examples.jl — two demonstration renders of BlackLightPlus on a Kerr hole of spin a = 0.94.
#   (a) Exact-Kerr image of the AART-style equatorial disk EquatorialDisk(0.94):
#       render_analytic on a plan with no matter, inclination 17°, camera distance 1e4,
#       full field of view 20 GM/c², photon-ring orders n = 0, 1, 2 summed -> Stokes I.
#   (b) Polarized image (IQUV) of the analytic PolishDoughnut torus (r_center = 9,
#       n_e_center = 5e5 cm⁻³, theta_e_center = 40, beta = 8): the fused march of
#       render on the torus as an analytic plasma provider, inclination 60°, distance
#       1000, full field of view 2.2 r_out, 230 GHz, length unit GM/c² = 9.6e14 cm (an
#       M87*-mass hole), step control C = 80, at most 2500 steps, escape at 1.1 × 1000.
# Writes to figures/paper/out/:
#   analytic_disk_I.npy          (a), RES × RES
#   torus_{I,Q,U,V}.npy          (b), min(128, RES)²
#   examples_meta.txt            code SHA, parameters of both panels, threads, wall times
# Every array is laid out [iβ, iα] (numpy axis 0 = β, axis 1 = α); every render runs on
# CPU threads (backend = :cpu).
# Env: BLP_FIG_RES (default 256; the polarized panel is capped at 128).
# Run from the repository root: julia -t 32 --project=. figures/paper/fig_examples.jl

using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL
using Printf

const RES = parse(Int, get(ENV, "BLP_FIG_RES", "256"))
const OUT = joinpath(@__DIR__, "out")
mkpath(OUT)
# code version recorded in examples_meta.txt: HEAD SHA, "-dirty" if src has uncommitted edits
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const A = 0.94                                   # black-hole spin for both panels
const st = KerrSchildCartesian(A)

# ------------------------------------------------------- (a) analytic Kerr disk
cfg_a = RunConfig(camera = CameraConfig(inclination_deg = 17.0, distance = 1.0e4,
                                        fov = 20.0, resolution = RES))
plan_a = prepare(cfg_a; spacetime = st, matter = nothing)
img = render_analytic(plan_a, EquatorialDisk(A); bands = (0, 1, 2), backend = :cpu)
ta = img.run.seconds
write_npy(joinpath(OUT, "analytic_disk_I.npy"), img.I)
@printf("analytic disk : %4d^2  bands (0,1,2)  %6.1f s  sum(I) = %.4e\n",
        RES, ta, sum(img.I))

# --------------------------------------------------- (b) polarized torus (IQUV)
const PRES = min(128, RES)
torus = PolishDoughnut(A; r_center = 9.0, n_e_center = 5.0e5, theta_e_center = 40.0,
                       beta = 8.0)
inc_deg = 60.0
dist = 1000.0
fovp = 2.2 * torus.r_out
nu_obs = 230.0e9
L_unit = 9.6e14                                  # GM/c^2 of an M87*-mass hole [cm]
m_bh = L_unit * CL^2 / GNEWT                     # the hole mass whose GM/c^2 is L_unit
GNEWT * m_bh / CL^2 == L_unit || error("m_bh does not reproduce L_unit = $L_unit")

cfg_b = RunConfig(m_bh_cgs = m_bh, frequency_hz = nu_obs,
                  camera = CameraConfig(inclination_deg = inc_deg, distance = dist,
                                        fov = fovp, resolution = PRES),
                  max_steps = 2500, C = 80.0, escape_factor = 1.1)
plan_b = prepare(cfg_b; spacetime = st, matter = torus, mode = IQUV())
pol = render(plan_b; backend = :cpu)
tb = pol.run.seconds
for (name, S) in (("I", pol.I), ("Q", pol.Q), ("U", pol.U), ("V", pol.V))
    write_npy(joinpath(OUT, "torus_$(name).npy"), S)
end
@printf("polarized torus: %4d^2  r_out = %.1f M  %6.1f s  sum(I) = %.4e  max|P|/max I = %.3f\n",
        PRES, torus.r_out, tb, sum(pol.I), maximum(hypot.(pol.Q, pol.U)) / maximum(pol.I))

open(joinpath(OUT, "examples_meta.txt"), "w") do io
    println(io, "git=$GIT_SHA")
    println(io, "analytic_disk: a=$A tracer=analytic inc=17 dist=1e4 fov=20 res=$RES bands=0,1,2 source=EquatorialDisk")
    println(io, "analytic_disk_I.npy axes: [beta, alpha] (axis 0 = beta, axis 1 = alpha)")
    println(io, "torus: PolishDoughnut a=$A r_center=9 n_e=5e5 theta_e=40 beta=8 inc=$inc_deg dist=$dist fov=$(round(fovp, digits=3)) res=$PRES nu=$nu_obs")
    @printf(io, "run: backend=cpu threads=%d analytic_seconds=%.2f torus_seconds=%.2f\n",
            Threads.nthreads(), ta, tb)
end
