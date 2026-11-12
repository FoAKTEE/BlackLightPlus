# fig_scaling_gpu_img.jl — strong and weak scaling of the images on one backend, every call
# kept (the GPU half of fig_perf_strong (a) and fig_perf_weak (b)). Run inside a GPU job, one
# fresh process per backend and repetition: one GPU is backend gpu:0, two GPUs gpu:0,1.
#
# Decks (one AthenaK snapshot, torus.mhd_w_bcc.00999, Kerr a = 0.9, 230 GHz):
#   p  the Kerr accretion flow of the appendix table (fig_pol_hires.jl, fig_scaling_cpu.jl):
#      M = 6.5e9 x 1.98847e33 g, density scale 2.272651712e23 g, inclination 60 deg,
#      distance 1000, field of view 30, C = 80, 2500 steps, escape 1.1,
#      PlasmaConfig(rat_low 1, rat_high 10, gamma 13/9, sigma_cut 1, theta_e_min -1)
#   f  the fiducial strong-scaling deck of figures/blacklight/fig_scaling.jl: M = 4.152e6 Msun,
#      density scale 7.709296641905617e15 g, camera at r = 100, inclination 45 deg, field of
#      view 24, 4000 steps, C = 80, escape 1.1, the same plasma with use_p = true
# Rows <deck>_<kind><res>: kind `amr` = render_amr(plan(res, IQUV); block = 16,
# max_level = 2) (res a multiple of 16; the production image at res 384), `IQUV` / `I` =
# render of a uniform res^2 frame, polarized / Stokes I. The weak-scaling rows double the
# pixels: round(sqrt(2) res) per side (the adaptive image: the next multiple of 16).
# Per row: one untimed call (compilation, uploads), then n timed calls with @timed (wall, GC,
# bytes), the load average and the foreign CPU fraction of this process's CPUs over the timed
# calls, the CRC-32C of the I, Q, U, V planes of the last call, the ray count. Output:
# <outdir>/<label>.bin (Serialization) and the printed lines.
# Usage: julia -t T --project=<env> fig_scaling_gpu_img.jl <label> <backend> <outdir> <rows,...> [n] [n_slow]
#   backend cpu | gpu:0 | gpu:0,1 ; n (default 7) calls per row, n_slow (default 3) for rows
#   whose frame has at least 1536^2 polarized pixels or whose adaptive base is at least 544.
using Serialization, Dates, Statistics, CRC32c, Printf
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Execution: resolve, describe
const LABEL, BKS, OUT = ARGS[1], ARGS[2], ARGS[3]
const ROWS = String.(split(ARGS[4], ','))
const NCALL = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 7
const NSLOW = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
const ISGPU = startswith(BKS, "gpu")
ISGPU && @eval using CUDA
const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const GIT = let root = dirname(dirname(pathof(BlackLightPlus)))
    sha = readchomp(`git -C $root rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`git -C $root status --porcelain -- src ext Project.toml`))
    dirty ? sha * "-dirty" : sha
end
line(s) = (println(s); flush(stdout))

function cpu_list(s)
    out = Int[]
    for r in split(s, ',')
        a, b = occursin('-', r) ? split(r, '-') : (r, r)
        append!(out, parse(Int, a):parse(Int, b))
    end
    return out
end
const CPUSTR = match(r"Cpus_allowed_list:\s*(\S+)", read("/proc/self/status", String))[1]
const MYCPUS = cpu_list(CPUSTR)
function busy_ticks()
    t = 0
    for l in eachline("/proc/stat")
        m = match(r"^cpu(\d+) (.*)$", l)
        (m === nothing || !(parse(Int, m[1]) in MYCPUS)) && continue
        f = parse.(Int, split(m[2]))
        t += f[1] + f[2] + f[3] + f[6] + f[7] + f[8]
    end
    return t
end
own_ticks() = (s = read("/proc/self/stat", String); f = split(s[findlast(')', s) + 2:end]);
               parse(Int, f[12]) + parse(Int, f[13]))
loadavg() = parse(Float64, split(read("/proc/loadavg", String))[1])

line(@sprintf("img_scale %s: commit %s, backend %s, %d Julia threads, cpus %s (%d), rows %s, %d (%d) calls, %s",
              LABEL, GIT, BKS, Threads.nthreads(), CPUSTR, length(MYCPUS), join(ROWS, ","), NCALL, NSLOW, now()))

const PLASMA = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                            theta_e_min = -1.0)
function deck_cfg(deck, res)
    if deck == "p"
        m = 6.5e9 * 1.98847e33
        return RunConfig(m_bh_cgs = m, mass_scale_cgs = 2.272651712e23, frequency_hz = 230.0e9,
            camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
            plasma = PLASMA, max_steps = 2500, C = 80.0, escape_factor = 1.1)
    elseif deck == "f"
        return RunConfig(m_bh_cgs = 4.152e6 * MSUN, mass_scale_cgs = 7.709296641905617e15, frequency_hz = 2.3e11,
            camera = CameraConfig(inclination_deg = 45.0, distance = 100.0, fov = 24.0, resolution = res),
            plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                                  theta_e_min = -1.0),
            max_steps = 4000, C = 80.0, escape_factor = 1.1)
    end
    error("unknown deck $deck")
end

function parse_row(r)
    m = match(r"^(p|f)_(amr|IQUV|I)(\d+)$", r)
    m === nothing && error("row $r is not <p|f>_<amr|IQUV|I><res>")
    return (deck = String(m[1]), kind = String(m[2]), res = parse(Int, m[3]))
end
foreach(parse_row, ROWS)
slow(s) = (s.kind == "IQUV" && s.res >= 1536) || (s.kind == "amr" && s.res >= 544)

const EXEC = resolve(BKS)
line("executor: " * sprint(show, describe(EXEC)))
t_load = @elapsed snap = AthenaKSnapshot(DATA)
line(@sprintf("snapshot loaded in %.2f s", t_load))

planes(x::Image) = (x.I, x.Q, x.U, x.V)
planes(x::NamedTuple) = planes(x.image)
bits_hash(x) = [crc32c(reinterpret(UInt8, vec(collect(m)))) for m in planes(x)]

const CALLS = Dict{String,Any}()
const HASH = Dict{String,Any}()
const NRAYS = Dict{String,Int}()
for name in ROWS
    s = parse_row(name)
    p = prepare(deck_cfg(s.deck, s.res); spacetime = KerrSchildCartesian(0.9), matter = snap,
                mode = s.kind == "I" ? StokesI() : IQUV())
    f = s.kind == "amr" ? (() -> render_amr(p; block = 16, max_level = 2, backend = EXEC)) :
                          (() -> render(p; backend = EXEC))
    n = slow(s) ? NSLOW : NCALL
    t_first = @elapsed r = f()                    # compilation and first uploads
    NRAYS[name] = r isa NamedTuple ? r.nrays : s.res^2
    GC.gc()
    b0, o0, l0, t0 = busy_ticks(), own_ticks(), loadavg(), time()
    calls = NamedTuple[]
    for _ in 1:n
        st = @timed f()
        r = st.value
        push!(calls, (time = st.time, gctime = st.gctime, bytes = st.bytes))
    end
    b1, o1, l1, t1 = busy_ticks(), own_ticks(), loadavg(), time()
    foreign = ((b1 - b0) - (o1 - o0)) / (100 * (t1 - t0) * length(MYCPUS))
    CALLS[name] = (calls = calls, first = t_first, load0 = l0, load1 = l1, foreign = foreign,
                   deck = s.deck, kind = s.kind, res = s.res, t_start = t0, t_end = t1)
    HASH[name] = bits_hash(r)
    ts = [c.time for c in calls]
    line(@sprintf("%-10s median %.4f s best %.4f s first %.2f s gc median %.4f s %.1f MiB/call load %.1f->%.1f foreign %.3f rays %d calls %s",
                  name, median(ts), minimum(ts), t_first, median([c.gctime for c in calls]),
                  calls[end].bytes / 2^20, l0, l1, foreign, NRAYS[name],
                  join((@sprintf("%.4f/%.4f", c.time, c.gctime) for c in calls), " ")))
    line(@sprintf("%-10s crc %s", name, join(string.(HASH[name]; base = 16), " ")))
end
mkpath(OUT)
out = joinpath(OUT, LABEL * ".bin")
serialize(out, Dict{String,Any}("meta" => (label = LABEL, backend = BKS, commit = GIT,
                                            threads = Threads.nthreads(), date = string(now()),
                                            job = get(ENV, "SLURM_JOB_ID", ""), cpus = MYCPUS,
                                            executor = sprint(show, describe(EXEC)),
                                            host = gethostname(), cvd = get(ENV, "CUDA_VISIBLE_DEVICES", "")),
                                "calls" => CALLS, "hash" => HASH, "nrays" => NRAYS))
line("img_scale $LABEL written: $out")
line("img_scale done")
