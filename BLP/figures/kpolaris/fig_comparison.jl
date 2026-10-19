# fig_comparison.jl — BLP images for the code comparison of the KPolaris paper (figure 1,
# table of NMSE): the torus snapshot torus.mhd_w_bcc.00999 (a = 0.9) imaged in full Stokes
# at 230 and 345 GHz, fast light, and in slow light over the ten dumps 00990–00999
# (Δt = 30 M, t_obs = 78570 the last dump's time), with the fiducial plasma deck of
# fig_pol_hires.jl (M = 6.2e9 Msun, mass scale 1e-21 L³, R_low 1, R_high 10, γ = 13/9,
# σ cut 1) and a plane-parallel camera at i = 60°, distance 100 M, field of view 30 M,
# 256². March: C = 80, 2500 steps, escape 1.1 d. The reference images of Blacklight on the
# same data come from run_blacklight.sh.
# Writes BLP/figures/out/kpolaris/comparison/blp_<fast|slow>_<230|345>_{I,Q,U,V}.npy
# ([iβ, iα]) and blp_record.json. Run: julia -t 32 --project=BLP BLP/figures/kpolaris/fig_comparison.jl
using Printf, Dates
using BLP
using BLP.Numerics: GNEWT, CL, MSUN
using BLP.Data: SnapshotSeries, SeriesProvider

const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no"
const DUMPS = 990:999
const OUT = joinpath(@__DIR__, "..", "out", "kpolaris", "comparison")
const RES = 256
const DCAM = 100.0
const INC = 60.0
const FOV = 30.0
mkpath(OUT)

function config(nu)
    m_bh = 6.2e9 * MSUN
    L = GNEWT * m_bh / CL^2
    return RunConfig(m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = nu,
                     camera = CameraConfig(inclination_deg = INC, distance = DCAM, fov = FOV,
                                           resolution = RES),
                     plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0,
                                           use_p = true, sigma_cut = 1.0, theta_e_min = -1.0),
                     max_steps = 2500, C = 80.0, escape_factor = 1.1)
end

function save(tag, img)
    for (n, A) in (("I", img.I), ("Q", img.Q), ("U", img.U), ("V", img.V))
        write_npy(joinpath(OUT, "blp_$(tag)_$n.npy"), A)
    end
    lp = sqrt(sum(img.Q)^2 + sum(img.U)^2) / sum(img.I)
    @printf("%-10s sum I = %.4e  m_L,net = %.4f  |V|/I = %.4e  %.1f s\n", tag, sum(img.I), lp,
            sum(img.V) / sum(img.I), img.run.seconds)
    return Dict("sumI" => sum(img.I), "mLnet" => lp, "V_over_I" => sum(img.V) / sum(img.I),
                "seconds" => img.run.seconds)
end

rec = Dict{String,Any}("res" => RES, "dcam" => DCAM, "inc" => INC, "fov" => FOV,
                       "dumps" => collect(DUMPS), "threads" => Threads.nthreads(),
                       "date" => string(now()))
snap = AthenaKSnapshot(joinpath(DATA, @sprintf("torus.mhd_w_bcc.%05d.athdf", last(DUMPS))))
paths = [joinpath(DATA, @sprintf("torus.mhd_w_bcc.%05d.athdf", n)) for n in DUMPS]
times = [snap.time - 30.0 * (last(DUMPS) - n) for n in DUMPS]
rec["times"] = times
series = SnapshotSeries(paths, times; maxslots = length(paths), t_obs = times[end])
for nu in (2.3e11, 3.45e11)
    tag = nu == 2.3e11 ? "230" : "345"
    plan = prepare(config(nu); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
    rec["fast_$tag"] = save("fast_$tag", render(plan; backend = :cpu))
    plan_s = prepare(config(nu); spacetime = KerrSchildCartesian(0.9), matter = SeriesProvider(series),
                     mode = IQUV())
    rec["slow_$tag"] = save("slow_$tag", render(plan_s; backend = :cpu))
end
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::Integer) = string(x)
json(x::AbstractString) = "\"$x\""
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ", ") * "}"
open(io -> println(io, json(rec)), joinpath(OUT, "blp_record.json"), "w")
println("wrote ", OUT)
