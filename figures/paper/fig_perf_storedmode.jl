# fig_perf_storedmode.jl — the single pass against the stored mode, the stored mode's two passes
# timed inside each of its calls (fig_perf_strong, panel b). The package images a frame in a
# single pass (`render(plan)`: geodesic, plasma sample and transfer in one march, no ray stored)
# or in the stored mode: a first pass `trace(plan)` stores every ray's states on the host, and a
# second pass `render(plan; path = stored)` integrates the transfer along each stored ray on host
# threads. One stored-mode call runs both passes; the wall, GC time and bytes of each pass are
# read inside the call, so the two passes add up to the call. Nothing outlives a call, and a full
# collection runs before every timed call of every row, so each call starts from the same heap.
# Deck: the Kerr accretion flow of the appendix table, as fig_scaling_cpu.jl: AthenaK snapshot
# 00999 on KerrSchildCartesian(0.9), black-hole mass 6.5e9 x 1.98847e33 g, density scale
# 2.272651712e23 g, 230 GHz, inclination 60 deg, distance 1000, field of view 30, C = 80, 2500
# steps, escape at 1.1 times the distance, R_low 1, R_high 10, gamma 13/9, sigma cut 1, no
# Theta_e floor; uniform IQUV frame of res^2 pixels.
# Rows:
#   sp   render(plan; backend)                                                (single pass)
#   sm   stored = trace(plan; backend); render(plan; path = stored, backend = :cpu)
#        (stored mode; first pass on the backend, second pass on the host's threads)
# One untimed call per row, then n timed calls. Every call: wall, GC time, bytes, its start and
# end (epoch seconds), the share of this process's CPUs that other processes took over it
# (/proc/stat busy ticks less this process's ticks), the CRC-32C of the I, Q, U, V planes; an sm
# call also the wall, GC time and bytes of each pass. Facts: rays, stored states and bytes; the
# largest relative difference of each plane between the stored-mode and the single-pass image.
# Output: <outdir>/<label>.bin (Serialization: meta, calls, facts) and the printed lines.
# Usage: julia -t T --project=<env> fig_perf_storedmode.jl <label> <backend> <outdir> [res] [n]
#   backend cpu | gpu:0 | gpu:0,1
using Serialization, Dates, Statistics, CRC32c, Printf
using BlackLightPlus, StaticArrays
using BlackLightPlus.Transfer: IQUV
using BlackLightPlus.Execution: resolve, describe
const LABEL, BKS, OUT = ARGS[1], ARGS[2], ARGS[3]
const RES = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 384
const NCALL = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 5
startswith(BKS, "gpu") && @eval using CUDA
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
const MYCPUS = Set(cpu_list(CPUSTR))
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

line(@sprintf("storedmode %s: commit %s, backend %s, %d Julia threads, cpus %s (%d), %d^2, %d calls, load %s, %s",
              LABEL, GIT, BKS, Threads.nthreads(), CPUSTR, length(MYCPUS), RES, NCALL,
              join(split(read("/proc/loadavg", String))[1:3], " "), now()))
const M_BH = 6.5e9 * 1.98847e33
const MSCALE = 2.272651712e23
cfg(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MSCALE, frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                          theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)
const EXEC = resolve(BKS)
const HOST = resolve(:cpu)
line("executor: " * sprint(show, describe(EXEC)) * "; host: " * sprint(show, describe(HOST)))
snap = AthenaKSnapshot(DATA)
planes(x::Image) = (x.I, x.Q, x.U, x.V)
crcs(x::Image) = Tuple(crc32c(reinterpret(UInt8, vec(collect(p)))) for p in planes(x))
pass(s) = (time = s.time, gctime = s.gctime, bytes = s.bytes)

function stored_call(p)
    a = @timed trace(p; backend = EXEC)
    b = @timed render(p; path = a.value, backend = HOST)
    return (image = b.value, trace = pass(a), transfer = pass(b))
end

const CALLS = Dict{String,Any}()
function timed_row(name, f, n)
    t_first = @elapsed r0 = f()
    ref = crcs(r0 isa Image ? r0 : r0.image)
    calls = NamedTuple[]
    for _ in 1:n
        GC.gc(true)
        b0, o0, e0 = busy_ticks(), own_ticks(), time()
        s = @timed f()
        b1, o1, e1 = busy_ticks(), own_ticks(), time()
        img = s.value isa Image ? s.value : s.value.image
        c = (time = s.time, gctime = s.gctime, bytes = s.bytes, t_start = e0, t_end = e1,
             foreign = ((b1 - b0) - (o1 - o0)) / (100 * (e1 - e0) * length(MYCPUS)),
             crc = crcs(img), same = crcs(img) == ref)
        s.value isa Image || (c = merge(c, (trace = s.value.trace, transfer = s.value.transfer)))
        push!(calls, c)
        s = nothing; img = nothing
    end
    CALLS[name] = (calls = calls, first = t_first, crc_first = ref)
    ts = [c.time for c in calls]
    extra = haskey(calls[1], :trace) ?
        @sprintf(" first pass median %.4f s, second pass median %.4f s",
                 median(c.trace.time for c in calls), median(c.transfer.time for c in calls)) : ""
    line(@sprintf("%-4s median %.4f s best %.4f s first %.2f s gc median %.4f s %.1f MiB/call foreign max %.4f same bits %s%s calls %s",
                  name, median(ts), minimum(ts), t_first, median(c.gctime for c in calls),
                  calls[end].bytes / 2^20, maximum(c.foreign for c in calls),
                  all(c.same for c in calls), extra,
                  join((haskey(c, :trace) ? @sprintf("%.4f=%.4f+%.4f", c.time, c.trace.time, c.transfer.time) :
                        @sprintf("%.4f", c.time) for c in calls), " ")))
    return r0
end

p = prepare(cfg(RES); spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
a = timed_row("sp", () -> render(p; backend = EXEC), NCALL)
t = trace(p; backend = EXEC)
ns = sum(length, t.states)
facts = (rays = length(t.states), states = ns,
         bytes = ns * sizeof(SVector{8,Float64}) + sum(length, t.dls) * 8)
t = nothing; GC.gc(true)
b = timed_row("sm", () -> stored_call(p), NCALL).image
rel = [maximum(abs.(x .- y)) / max(maximum(abs, x), floatmin()) for (x, y) in zip(planes(a), planes(b))]
facts = merge(facts, (rel = rel, bitwise = all(x == y for (x, y) in zip(planes(a), planes(b)))))
line(@sprintf("facts %d rays, %d stored states (%.1f per ray), %.2f GiB; stored mode against single pass: bitwise %s, largest relative difference per plane %s",
              facts.rays, ns, ns / facts.rays, facts.bytes / 2^30, facts.bitwise,
              join((@sprintf("%.2e", x) for x in rel), " ")))
mkpath(OUT)
serialize(joinpath(OUT, LABEL * ".bin"), Dict{String,Any}("meta" => (label = LABEL, backend = BKS, commit = GIT,
    threads = Threads.nthreads(), cpus = CPUSTR, res = RES, date = string(now()), job = get(ENV, "SLURM_JOB_ID", ""),
    executor = sprint(show, describe(EXEC)), host = sprint(show, describe(HOST))), "calls" => CALLS, "facts" => facts))
line("storedmode $LABEL written")
line("storedmode done")
