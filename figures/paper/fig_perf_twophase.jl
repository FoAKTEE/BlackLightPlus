# fig_perf_twophase.jl — the single pass against store-then-transfer on one backend, every
# call kept (fig_perf_strong, panel b). The package images a frame in a single pass
# (`render(plan)`: geodesic, plasma sample and transfer in one march, no ray stored) or in two
# phases (`trace(plan)` stores every ray's states on the host, then
# `render(plan; path = trace)` marches each pixel along its stored trajectory on host threads).
#
# Deck (TP_DECK=paper): the Kerr accretion flow of the appendix table, as fig_scaling_cpu.jl:
# AthenaK snapshot 00999 on KerrSchildCartesian(0.9), black-hole mass 6.5e9 x 1.98847e33 g,
# density scale 2.272651712e23 g, 230 GHz, inclination 60 deg, distance 1000, field of view
# 30, C = 80, 2500 steps, escape at 1.1 times the distance, R_low 1, R_high 10, gamma 13/9,
# sigma cut 1, no Theta_e floor. The default (TP_DECK unset) is the deck of bench/bench.jl
# (6.2e9 Msun, density scale 1e-21 (GM/c^2)^3).
# Rows, for each mode (IQUV, Stokes I) at res^2 pixels:
#   sp_<m>   render(plan; backend)                               (single pass)
#   tr_<m>   trace(plan; backend)                                (trace and store)
#   st_<m>   render(plan; path = trace, backend = :cpu)          (transfer along stored rays)
#   tp_<m>   trace(plan; backend) then render(plan; path, :cpu)  (store-then-transfer call)
# One untimed call per row, then n timed calls (@timed: wall, GC, bytes); the stored states
# and their bytes; the largest relative difference of each plane between the stored-ray and
# the single-pass image, and whether they are bitwise equal. Output: <outdir>/<label>.bin
# (Serialization: meta, every call, facts) and the printed lines.
# Usage: julia -t T --project=<env> fig_perf_twophase.jl <label> <backend> <outdir> [res] [n] [modes] [rows]
#   backend cpu | gpu:0 | gpu:0,1; rows a comma list of sp, tr, st, tp (default all).
using Serialization, Dates, Statistics, CRC32c, Printf
using BlackLightPlus, StaticArrays
using BlackLightPlus.Numerics: GNEWT, CL, MSUN
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Execution: resolve, describe
const LABEL, BKS, OUT = ARGS[1], ARGS[2], ARGS[3]
const RES = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 384
const NCALL = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 5
const MODES = length(ARGS) >= 6 ? split(ARGS[6], ',') : ["IQUV", "I"]
const ROWS = length(ARGS) >= 7 ? split(ARGS[7], ',') : ["sp", "tr", "st", "tp"]
startswith(BKS, "gpu") && @eval using CUDA
const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const GIT = let root = dirname(dirname(pathof(BlackLightPlus)))
    sha = readchomp(`git -C $root rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`git -C $root status --porcelain -- src ext Project.toml`))
    dirty ? sha * "-dirty" : sha
end
line(s) = (println(s); flush(stdout))
line(@sprintf("twophase_calls %s: deck %s, commit %s, backend %s, %d Julia threads, cpus %s, %d^2, %d calls, modes %s, %s",
              LABEL, get(ENV, "TP_DECK", "bench"), GIT, BKS, Threads.nthreads(), match(r"Cpus_allowed_list:\s*(\S+)",
              read("/proc/self/status", String))[1], RES, NCALL, join(MODES, ","), now()))
const DECK = get(ENV, "TP_DECK", "bench")
const M_BH = DECK == "paper" ? 6.5e9 * 1.98847e33 : 6.2e9 * MSUN
const L_UNIT = GNEWT * M_BH / CL^2
const MSCALE = DECK == "paper" ? 2.272651712e23 : 1.0e-21 * L_UNIT^3
torus_cfg(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MSCALE,
    frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                          theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)
const EXEC = resolve(BKS)
const HOST = resolve(:cpu)
line("executor: " * sprint(show, describe(EXEC)) * "; host: " * sprint(show, describe(HOST)))
snap = AthenaKSnapshot(DATA)
planes(x::Image) = (x.I, x.Q, x.U, x.V)
const CALLS = Dict{String,Any}()
const FACTS = Dict{String,Any}()
function timed_row(name, f, n)
    t_first = @elapsed r = f()
    GC.gc()
    calls = NamedTuple[]
    for _ in 1:n
        s = @timed f()
        r = s.value
        push!(calls, (time = s.time, gctime = s.gctime, bytes = s.bytes))
    end
    CALLS[name] = (calls = calls, first = t_first)
    ts = [c.time for c in calls]
    line(@sprintf("%-10s median %.4f s best %.4f s first %.2f s gc median %.4f s %.1f MiB/call calls %s", name,
                  median(ts), minimum(ts), t_first, median([c.gctime for c in calls]), calls[end].bytes / 2^20,
                  join((@sprintf("%.4f/%.4f", c.time, c.gctime) for c in calls), " ")))
    return r
end
for m in MODES
    mode = m == "IQUV" ? IQUV() : StokesI()
    p = prepare(torus_cfg(RES); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = mode)
    a = "sp" in ROWS ? timed_row("sp_$m", () -> render(p; backend = EXEC), NCALL) : render(p; backend = EXEC)
    t = timed_row("tr_$m", () -> trace(p; backend = EXEC), NCALL)
    b = "st" in ROWS ? timed_row("st_$m", () -> render(p; path = t, backend = HOST), NCALL) : nothing
    "tp" in ROWS && timed_row("tp_$m", () -> render(p; path = trace(p; backend = EXEC), backend = HOST), NCALL)
    ns = sum(length, t.states)
    nbytes = ns * sizeof(SVector{8,Float64}) + sum(length, t.dls) * 8
    rel = b === nothing ? Float64[] :
          [maximum(abs.(x .- y)) / max(maximum(abs, x), floatmin()) for (x, y) in zip(planes(a), planes(b))]
    same = b === nothing ? missing : all(x == y for (x, y) in zip(planes(a), planes(b)))
    FACTS[m] = (rays = length(t.states), states = ns, bytes = nbytes, rel = rel, bitwise = same)
    line(@sprintf("facts %-4s %d rays, %d stored states (%.1f per ray), %.2f GiB; two-phase against single pass: bitwise %s, largest relative difference per plane %s",
                  m, length(t.states), ns, ns / length(t.states), nbytes / 2^30, same,
                  join((@sprintf("%.2e", x) for x in rel), " ")))
    t = nothing; GC.gc()
end
mkpath(OUT)
serialize(joinpath(OUT, LABEL * ".bin"), Dict{String,Any}("meta" => (label = LABEL, backend = BKS, commit = GIT,
    threads = Threads.nthreads(), res = RES, date = string(now()), job = get(ENV, "SLURM_JOB_ID", ""),
    executor = sprint(show, describe(EXEC))), "calls" => CALLS, "facts" => FACTS))
line("twophase_calls $LABEL written")
line("twophase_calls done")
