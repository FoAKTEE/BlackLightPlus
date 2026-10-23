# fig_accuracy_cost.jl — integration time against image error (KPolaris figure 2): the
# fiducial torus image (fig_pol_hires.jl deck: a = 0.9, M = 6.2e9 Msun, mass scale 1e-21 L³,
# 230 GHz, i = 60°, d = 1000 M, fov 30 M, R_high 10, σ cut 1) in full Stokes at 256² for a
# ladder of the step divisor C of the fixed-control march (step = (r − r_h)/C, step limit
# 40 C), timed as the median of three `render` calls after one warm-up call, and the error
# as the largest NMSE over I, Q, U, V against the image at C_REF. The adopted setting is
# C = 80. The backend is BLP_BACKEND (cpu | gpu); BLP_NDEV (1 | 2) selects the devices.
# Writes figures/paper/out/kpolaris/accuracy_cost/<backend>.json.
# Run: julia -t 32 --project=. figures/kpolaris/fig_accuracy_cost.jl (CPU) or inside a
# Slurm GPU job with --project=results/blp/gpu/env and BLP_BACKEND=gpu.
using Printf, Dates, Statistics
const BACKEND = get(ENV, "BLP_BACKEND", "cpu")
const NDEV = parse(Int, get(ENV, "BLP_NDEV", "1"))
BACKEND == "gpu" && (using CUDA)
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN

const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const OUT = joinpath(@__DIR__, "..", "out", "kpolaris", "accuracy_cost")
const RES = 256
const CS = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
const C_REF = 2560.0
const NREP = 3
mkpath(OUT)
backend_arg() = BACKEND == "gpu" ? (NDEV == 2 ? [0, 1] : :gpu) : :cpu

function config(C)
    m_bh = 6.2e9 * MSUN
    L = GNEWT * m_bh / CL^2
    return RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = 2.3e11,
                     camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0,
                                           resolution = RES),
                     plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                           use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
                     max_steps = round(Int, 40C), C = C, escape_factor = 1.1)
end
nmse(a, r) = sum((a .- r) .^ 2) / sum(r .^ 2)

snap = AthenaKSnapshot(DATA)
function image(C)
    plan = prepare(config(C); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
    img = render(plan; backend = backend_arg())                 # warm-up (JIT, uploads)
    ts = Float64[]
    for _ in 1:NREP
        img = render(plan; backend = backend_arg())
        push!(ts, img.run.seconds)
    end
    return img, ts
end
ref, tref = image(C_REF)
@printf("reference C = %.0f  median %.2f s\n", C_REF, median(tref))
rows = Dict{String,Any}[]
for C in CS
    img, ts = image(C)
    r = Dict("I" => nmse(img.I, ref.I), "Q" => nmse(img.Q, ref.Q), "U" => nmse(img.U, ref.U),
             "V" => nmse(img.V, ref.V))
    push!(rows, Dict("C" => C, "max_steps" => round(Int, 40C), "seconds" => ts,
                     "median" => median(ts), "nmse" => r, "max_nmse" => maximum(values(r)),
                     "sum_I" => sum(img.I)))
    @printf("C = %5.0f  median %7.2f s  NMSE I %.2e  max %.2e\n", C, median(ts), r["I"],
            maximum(values(r)))
end
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::Integer) = string(x)
json(x::AbstractString) = "\"$x\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ", ") * "}"
rec = Dict("backend" => BACKEND, "ndev" => NDEV, "threads" => Threads.nthreads(), "res" => RES,
           "C_ref" => C_REF, "ref_seconds" => tref, "rows" => rows, "date" => string(now()),
           "adopted_C" => 80.0)
open(io -> println(io, json(rec)), joinpath(OUT, "$(BACKEND)$(BACKEND == "gpu" ? NDEV : "").json"), "w")
println("wrote ", OUT)
