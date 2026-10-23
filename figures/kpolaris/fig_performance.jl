# fig_performance.jl — complete image time against resolution (KPolaris figure 3): the
# fiducial torus image (fig_accuracy_cost.jl deck, C = 80) in full Stokes at 256², 512²,
# 1024² and 2048², the median and range of three `render` calls after one warm-up call,
# on the backend BLP_BACKEND (cpu | gpu) with BLP_NDEV devices. The snapshot load and the
# plan preparation are timed once and recorded apart. Writes
# figures/paper/out/kpolaris/performance/<backend>.json.
# Run: julia -t 32 --project=. figures/kpolaris/fig_performance.jl, or inside a Slurm
# GPU job with --project=results/blp/gpu/env and BLP_BACKEND=gpu BLP_NDEV=1|2.
using Printf, Dates, Statistics
const BACKEND = get(ENV, "BLP_BACKEND", "cpu")
const NDEV = parse(Int, get(ENV, "BLP_NDEV", "1"))
BACKEND == "gpu" && (using CUDA)
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN

const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const OUT = joinpath(@__DIR__, "..", "out", "kpolaris", "performance")
const RESES = parse.(Int, split(get(ENV, "BLP_RESES", "256,512,1024,2048"), ","))
const NREP = 3
mkpath(OUT)
backend_arg() = BACKEND == "gpu" ? (NDEV == 2 ? [0, 1] : :gpu) : :cpu

function config(res)
    m_bh = 6.2e9 * MSUN
    L = GNEWT * m_bh / CL^2
    return RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = 2.3e11,
                     camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0,
                                           resolution = res),
                     plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                           use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
                     max_steps = 2500, C = 80.0, escape_factor = 1.1)
end

t0 = time()
snap = AthenaKSnapshot(DATA)
load_s = time() - t0
rows = Dict{String,Any}[]
for res in RESES
    t1 = time()
    plan = prepare(config(res); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
    prep_s = time() - t1
    img = render(plan; backend = backend_arg())
    ts = Float64[]
    for _ in 1:NREP
        img = render(plan; backend = backend_arg())
        push!(ts, img.run.seconds)
    end
    push!(rows, Dict("res" => res, "seconds" => ts, "median" => median(ts), "min" => minimum(ts),
                     "max" => maximum(ts), "prepare_s" => prep_s, "sum_I" => sum(img.I)))
    @printf("%5d^2  median %8.2f s  range %.2f–%.2f  prepare %.2f s\n", res, median(ts),
            minimum(ts), maximum(ts), prep_s)
end
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::Integer) = string(x)
json(x::AbstractString) = "\"$x\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ", ") * "}"
rec = Dict("backend" => BACKEND, "ndev" => NDEV, "threads" => Threads.nthreads(),
           "load_s" => load_s, "rows" => rows, "date" => string(now()), "host" => gethostname())
open(io -> println(io, json(rec)), joinpath(OUT, "$(BACKEND)$(BACKEND == "gpu" ? NDEV : "").json"), "w")
println("wrote ", OUT)
