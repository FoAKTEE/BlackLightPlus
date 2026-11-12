# fig_scaling_gpu_mc.jl — the Monte Carlo legs of the weak-scaling ladder of
# fig_scaling_cpu.jl (mcthin, mccompton) on one backend: warm mc_spectrum calls on the Kerr
# accretion deck, every call kept (the GPU half of fig_perf_weak (b)). Run inside a GPU job.
#
# Deck (= fig_scaling_cpu.jl `deck`, `MCKW`): AthenaK snapshot 00999 on KerrSchildCartesian(0.9),
# M = 6.5e9 x 1.98847e33 g, density scale 2.272651712e23 g, 230 GHz, R_low 1, R_high 10,
# gamma 13/9 with total-pressure matching, sigma cut 1, no Theta_e floor, C = 80; stride 1,
# seed 0x5eed0700, 160 bins over 1e8-1e24 Hz; absorption only (thin), or Compton scattering
# with bias 1e4 and at most three scatterings (compton).
# The process's first call (thin, on the first run's backend) builds the zones and the
# generation tables of its Ns; a leg's untimed call rebuilds the tables when the leg's Ns
# differs from the last call's, and its timed calls reuse them through the package's memo of
# the last grid run (Products.ZONE_MEMO), the warm call of a repeated run: the cost of the
# photon phase. The memo must hold the same tables before and after the timed calls of a
# leg (no rebuild), else the process exits 3. Every timed call of a leg must return the bins
# of its untimed call (CRC-32C of the bins, counts and powers, as fig_scaling_cpu.jl's
# crc_mc), else the process exits 3. Output: <outdir>/<label>.bin and the printed lines.
# Usage: julia -t T --project=<env> fig_scaling_gpu_mc.jl <label> <outdir> <Ns thin> <Ns compton>
#          <calls thin> <calls compton> <run>=<backend> [<run>=<backend> ...]
#   backend cpu | gpu:0 | gpu:0,1; the runs in the given order, each both legs.
using Serialization, Dates, Statistics, CRC32c, Printf
using BlackLightPlus
import BlackLightPlus.Products as PR
using BlackLightPlus.Execution: resolve, describe
const LABEL, OUT = ARGS[1], ARGS[2]
const NS_THIN, NS_COMPTON = parse(Float64, ARGS[3]), parse(Float64, ARGS[4])
const N_THIN, N_COMPTON = parse(Int, ARGS[5]), parse(Int, ARGS[6])
const RUNS = [Pair(String.(split(a, "="; limit = 2))...) for a in ARGS[7:end]]
any(r -> startswith(r[2], "gpu"), RUNS) && @eval using CUDA
const DATA = "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"
const GIT = let root = dirname(dirname(pathof(BlackLightPlus)))
    sha = readchomp(`git -C $root rev-parse --short HEAD`)
    dirty = !isempty(readchomp(`git -C $root status --porcelain -- src ext Project.toml`))
    dirty ? sha * "-dirty" : sha
end
line(s) = (println(s); flush(stdout))
const CPUSTR = String(match(r"Cpus_allowed_list:\s*(\S+)", read("/proc/self/status", String))[1])
loadavg() = parse(Float64, split(read("/proc/loadavg", String))[1])

const M_BH = 6.5e9 * 1.98847e33
const MASS_SCALE = 2.272651712e23
const MCKW = (stride = 1, seed = 0x5eed0700, spec_nu_min = 1.0e8, spec_nu_max = 1.0e24, n_ebins = 160)
deck(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                          sigma_cut = 1.0, theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)
mc_call(plan, leg, Ns, exec) = leg == "thin" ? mc_spectrum(plan, Ns; MCKW..., backend = exec) :
    mc_spectrum(plan, Ns; MCKW..., scattering = true, bias = 1.0e4, backend = exec)
function crc_mc(r)
    c = UInt32(0)
    for s in r.orders, a in (s.count, s.sumw, s.sumwE, s.sumwE2)
        c = crc32c(reinterpret(UInt8, vec(a)), c)
    end
    for v in (r.n_gen, r.n_children, r.n_recorded)
        c = crc32c(reinterpret(UInt8, [Int64(v)]), c)
    end
    for v in (r.emitted_power, r.escaped_power, r.absorbed_power, r.captured_power, r.dropped_power)
        c = crc32c(reinterpret(UInt8, [Float64(v)]), c)
    end
    return c
end

line(@sprintf("mc_scale %s: commit %s, %d Julia threads, cpus %s, Ns thin %.4g compton %.4g, calls %d %d, runs %s, %s",
              LABEL, GIT, Threads.nthreads(), CPUSTR, NS_THIN, NS_COMPTON, N_THIN, N_COMPTON,
              join(("$(a)=$(b)" for (a, b) in RUNS), " "), now()))
t_load = @elapsed snap = AthenaKSnapshot(DATA)
line(@sprintf("snapshot loaded in %.2f s", t_load))
plan = prepare(deck(64); spacetime = KerrSchildCartesian(0.9), matter = snap)
execs = Dict(l => (b == "cpu" ? resolve(:cpu) : resolve(b)) for (l, b) in RUNS)
t_cold = @elapsed r0 = mc_call(plan, "thin", NS_THIN, execs[RUNS[1][1]])
line(@sprintf("cold thin call on %s: %.2f s, %d zones, %d primaries", RUNS[1][2], t_cold, r0.n_zones, r0.n_gen))
ok = true
res = Dict{String,Any}()
for (lab, bk) in RUNS
    exec = execs[lab]
    line("== $lab: $bk, " * sprint(show, describe(exec)))
    legs = Dict{String,Any}()
    for (leg, Ns, n) in (("thin", NS_THIN, N_THIN), ("compton", NS_COMPTON, N_COMPTON))
        t_first = @elapsed r1 = mc_call(plan, leg, Ns, exec)
        tab = PR.ZONE_MEMO.tables
        GC.gc()
        l0 = loadavg()
        calls = NamedTuple[]
        crcs = UInt32[]
        r = r1
        for _ in 1:n
            ts = time()
            s = @timed mc_call(plan, leg, Ns, exec)
            r = s.value
            push!(calls, (time = s.time, gctime = s.gctime, bytes = s.bytes, t_start = ts, t_end = time()))
            push!(crcs, crc_mc(r))
        end
        l1 = loadavg()
        hit = PR.ZONE_MEMO.tables === tab
        same = all(==(crc_mc(r1)), crcs)
        global ok &= hit & same
        ts = [c.time for c in calls]
        line(@sprintf("%s %-7s Ns %.4g: median %.4f s min %.4f max %.4f (%d calls) untimed %.2f s, %d primaries %d children, memo %s, bins %s, load %.1f->%.1f, crc %08x, calls %s",
                      lab, leg, Ns, median(ts), minimum(ts), maximum(ts), n, t_first, r.n_gen, r.n_children,
                      hit ? "held" : "REBUILT", same ? "same" : "DIFFER", l0, l1, crc_mc(r1),
                      join((@sprintf("%.4f", t) for t in ts), " ")))
        legs[leg] = (Ns = Ns, calls = calls, first = t_first, primaries = r.n_gen, children = r.n_children,
                     zones = r.n_zones, crc = crc_mc(r1), memo_held = hit, bins_same = same, load0 = l0, load1 = l1,
                     escaped_power = r.escaped_power)
    end
    res[lab] = (backend = bk, executor = sprint(show, describe(exec)), legs = legs)
end
mkpath(OUT)
out = joinpath(OUT, LABEL * ".bin")
serialize(out, Dict{String,Any}("meta" => (label = LABEL, commit = GIT, threads = Threads.nthreads(), cpus = CPUSTR,
                                            date = string(now()), job = get(ENV, "SLURM_JOB_ID", ""),
                                            host = gethostname(), ns_thin = NS_THIN, ns_compton = NS_COMPTON,
                                            cold_s = t_cold, cvd = get(ENV, "CUDA_VISIBLE_DEVICES", "")),
                                "runs" => res))
line("mc_scale $LABEL written: $out")
line(ok ? "mc_scale done" : "mc_scale: memo rebuilt or bins differ")
exit(ok ? 0 : 3)
