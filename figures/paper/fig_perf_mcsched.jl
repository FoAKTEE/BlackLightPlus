# fig_perf_mcsched.jl — the two schedules of the Monte Carlo transport on GPUs: the device
# wavefront (the default) against the generation-by-generation cascade of the same code
# (`Execution.with_wavefront_settings(; wavefront = false)`, one launch per generation of
# children), on one and two GPUs, timed around the transport of the superphotons on the deck
# of the Monte Carlo legs of fig_scaling_cpu.jl and fig_scaling_gpu_mc.jl. Run inside a GPU job.
#
# Deck (= fig_scaling_gpu_mc.jl `deck`, `MCKW`; mc_spectrum's defaults otherwise): AthenaK
# snapshot 00999 on KerrSchildCartesian(0.9), M = 6.5e9 x 1.98847e33 g, density scale
# 2.272651712e23 g (M87_DENSITY_SCALE["00999"] of m87_scaling.jl), 230 GHz, R_low 1, R_high 10,
# gamma 13/9 with total-pressure matching, sigma cut 1, no Theta_e floor, C = 80; stride 1
# (every owned cell above the horizon emits), seed 0x5eed0700, generation over 1e8-1e16 Hz on
# 400 intervals, 160 bins over 1e8-1e24 Hz in the default inclination bands (17, 50, 90 deg,
# half width 10); escape at r = 1e3, emission radius the larger of 100 and 1.2 times the
# outermost zone; absorption always; no scattering (thin), or Compton scattering (compton)
# with the bias of mc_spectrum(bias = 1e4): the state bias of b0 = 1e4, order factor 1, no
# splitting, cap Products.MC_BIAS_MAX, at most three scatterings.
#
# A process loads the snapshot and runs two cold mc_spectrum calls of --ns primaries (thin,
# then compton) on --ref (default gpu:0): they build the zones, the generation tables and the
# hot-cross table, and their bins, ledger totals and counts are the reference. Then each run
# <name>=<backend>[@wavefront=false] in the order given: the generation tables of the zones on
# the run's own executor, twice (the first compiles), which must equal bitwise those of the
# reference calls; per leg one untimed call of the first 2000 primaries, then --calls timed
# calls of the transport, built as mc_spectrum's grid run builds it after its tables:
# zone_offsets, the march, the opacities, the bins and the ledger, Products.run_layers and the
# spectra of its bins (a repeated mc_spectrum call adds only its memo lookups and its result).
# Every timed call must return bitwise the bins, ledger totals, primaries, escaped count and
# children of the reference call of its leg, and every call of a leg the ledger of its first
# call, else the process exits 3.
#
# Per timed call: wall, GC time, bytes, start and end (Unix time), process CPU time per Julia
# thread, the load average before and after, the share of the process's CPUs that other
# processes took, the primaries and children, the CRC-32C of the bins and the ledger, and the
# leaves (plain arrays and numbers) for the comparison across processes.
# Output: <outdir>/<label>.bin (Serialization of plain data: meta, reference, runs) and the
# printed lines.
#
# Usage:
#   julia -t T --project=<env> fig_perf_mcsched.jl run <label> <outdir> [--calls N] [--ns X]
#         [--ref <backend>] <name>=<backend>[@wavefront=false] ...
#   backend gpu:0 | gpu:0,1 | cpu (defaults: 5 calls, Ns 1e6, reference on gpu:0)
#   julia --project=<env> fig_perf_mcsched.jl compare <outdir>
# compare: every call of every run of every <outdir>/*.bin against the reference of the first
# file (bins, totals, counts) and the ledger of its first run; exits 1 on any difference.
using Serialization, Dates, Statistics, CRC32c, Printf

# ---------------------------------------------------------------- leaves and their bits

same_bits(a::AbstractArray, b::AbstractArray) =
    typeof(a) == typeof(b) && size(a) == size(b) && reinterpret(UInt8, vec(a)) == reinterpret(UInt8, vec(b))
same_bits(a::Number, b::Number) = typeof(a) == typeof(b) && reinterpret(UInt8, [a]) == reinterpret(UInt8, [b])
same_bits(a, b) = false
"Keys of `ref` whose leaves differ in `d` (missing keys included)."
differing(d, ref) = [k for k in sort(collect(keys(ref))) if !(haskey(d, k) && same_bits(d[k], ref[k]))]
function crc_leaves(d)
    c = UInt32(0)
    for k in sort(collect(keys(d)))
        v = d[k]
        c = crc32c(reinterpret(UInt8, v isa AbstractArray ? vec(v) : [v]), c)
    end
    return c
end

function compare_main(dir)
    files = sort(filter(f -> endswith(f, ".bin"), readdir(dir; join = true)))
    isempty(files) && (println("compare: no .bin in $dir"); return 1)
    F = [deserialize(f) for f in files]
    ref = F[1]["reference"]; led = Dict{String,Any}()
    bad = 0; ncmp = 0
    for (f, D) in zip(files, F)
        for leg in ("thin", "compton")
            nd = differing(D["reference"][leg].leaves, ref[leg].leaves)
            ncmp += 1
            isempty(nd) || (bad += 1; println("DIFF $(basename(f)) reference $leg: $(join(nd, " "))"))
        end
        for r in D["runs"], leg in ("thin", "compton")
            L = r.legs[leg]
            haskey(led, leg) || (led[leg] = L.calls[1].ledger)
            for (k, c) in enumerate(L.calls)
                nd = vcat(differing(c.leaves, ref[leg].leaves), differing(c.ledger, led[leg]))
                ncmp += 1
                isempty(nd) || (bad += 1; println("DIFF $(basename(f)) $(r.name) $leg call $k: $(join(nd, " "))"))
            end
            @printf("%-8s %-6s %-8s %d calls, %d primaries %d children, crc %08x\n", basename(f), r.name, leg,
                    length(L.calls), L.calls[1].leaves["n_primary"], L.calls[1].leaves["n_children"],
                    L.calls[1].crc)
        end
    end
    println("compare: $(length(files)) files, $ncmp results against the reference of $(basename(files[1])) " *
            "($(length(ref["thin"].leaves)) thin and $(length(ref["compton"].leaves)) compton leaves) and the " *
            "ledger of its first run: " * (bad == 0 ? "all bitwise" : "$bad differ"))
    return bad == 0 ? 0 : 1
end
ARGS[1] == "compare" && exit(compare_main(ARGS[2]))

# ---------------------------------------------------------------- run

using BlackLightPlus
import BlackLightPlus.Products as PR
using BlackLightPlus.Execution: resolve, describe, with_wavefront_settings, ExactBins, InclinationBands,
                                Spectrum
using BlackLightPlus.MonteCarlo: CurvedMarch, Opacities, StateBias, EMISSIVITY_TABLE, mean_weight,
                                 MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP
using BlackLightPlus.Numerics: widest_layout

ARGS[1] == "run" || error("usage: fig_perf_mcsched.jl run <label> <outdir> ... | compare <outdir>")
const LABEL, OUT = ARGS[2], ARGS[3]
function parse_rest(args)
    pos = String[]; opt = Dict{String,String}(); i = 1
    while i <= length(args)
        args[i] in ("--calls", "--ns", "--ref") ? (opt[args[i]] = args[i + 1]; i += 2) : (push!(pos, args[i]); i += 1)
    end
    return pos, opt
end
const POS, OPT = parse_rest(ARGS[4:end])
const NCALLS = parse(Int, get(OPT, "--calls", "5"))
const NS = parse(Float64, get(OPT, "--ns", "1e6"))
const REF_BACKEND = get(OPT, "--ref", "gpu:0")
# A run's backend and its wavefront settings: `<backend>[@k=v,...]`.
function split_spec(spec)
    i = findfirst(==('@'), spec)
    i === nothing && return String(spec), (;)
    kv = map(split(spec[nextind(spec, i):end], ',')) do a
        k, v = split(a, '='; limit = 2)
        Symbol(k) => (v == "true" ? true : v == "false" ? false : parse(Int, v))
    end
    return String(spec[1:prevind(spec, i)]), NamedTuple(kv)
end
const RUNS = [(name = String(a[1]), spec = String(a[2])) for a in (split(p, "="; limit = 2) for p in POS)]
isempty(RUNS) && error("fig_perf_mcsched: no run")
allunique(r.name for r in RUNS) || error("fig_perf_mcsched: a run name twice")
(startswith(REF_BACKEND, "gpu") || any(r -> startswith(r.spec, "gpu"), RUNS)) && @eval using CUDA
executor(b) = b == "cpu" ? resolve(:cpu) : resolve(b)
in_settings(f, set) = isempty(set) ? f() : with_wavefront_settings(f; set...)

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
const CPUSTR = String(match(r"Cpus_allowed_list:\s*(\S+)", read("/proc/self/status", String))[1])
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
loadavg() = parse(Float64, split(read("/proc/loadavg", String))[1])

const M_BH = 6.5e9 * 1.98847e33
const MASS_SCALE = 2.272651712e23
const SEED = 0x5eed0700
const MCKW = (stride = 1, seed = SEED, spec_nu_min = 1.0e8, spec_nu_max = 1.0e24, n_ebins = 160)
const B0 = 1.0e4                     # mc_spectrum(bias = B0) with scattering
const MAX_SCATT = 3                  # mc_spectrum's max_scatt
const ESCAPE, R_EM = 1.0e3, 100.0    # mc_spectrum's escape and R_em
const BANDS = InclinationBands((17.0, 50.0, 90.0), 10.0)
deck(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                          sigma_cut = 1.0, theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)

# The leaves of a phase: bins of every order, primaries, children, escaped count and the
# eleven ledger totals of the result (GRMHDSpectrumResult's order).
function phase_leaves(orders, n_primary, n_children, n_recorded, totals)
    d = Dict{String,Any}()
    for (k, s) in enumerate(orders), f in (:count, :sumw, :sumwE, :sumwE2, :dOmega)
        d["orders[$k].$f"] = copy(getfield(s, f))
    end
    d["n_primary"] = Int(n_primary); d["n_children"] = Int(n_children); d["n_recorded"] = Int(n_recorded)
    d["totals"] = collect(Float64, totals)
    return d
end
result_leaves(r) = phase_leaves(r.orders, r.n_gen, r.n_children, r.n_recorded,
                          (r.w_primary, r.w_children, r.w_first_order, r.scatter_loss, r.scatter_loss_final,
                           r.w_unspawned, r.emitted_power, r.escaped_power, r.absorbed_power,
                           r.captured_power, r.dropped_power))

line(@sprintf("mcsched %s: commit %s, tree %s, %d Julia threads, cpus %s, Ns %.4g, calls %d, runs %s, load %s, %s",
              LABEL, GIT, readchomp(`git rev-parse --short HEAD`), Threads.nthreads(), CPUSTR, NS, NCALLS,
              join(("$(r.name)=$(r.spec)" for r in RUNS), " "), join(split(read("/proc/loadavg", String))[1:3], " "),
              now()))
t_load = @elapsed snap = AthenaKSnapshot(DATA)
const PLAN = prepare(deck(64); spacetime = KerrSchildCartesian(0.9), matter = snap)
const ST = PLAN.spacetime
line(@sprintf("snapshot loaded in %.2f s", t_load))

# Reference: cold mc_spectrum calls on --ref (the zones, tables and hot-cross table of the run).
const EXEC_REF = executor(REF_BACKEND)
reference = Dict{String,Any}()
for leg in ("thin", "compton")
    t = @elapsed r = leg == "thin" ? mc_spectrum(PLAN, NS; MCKW..., backend = EXEC_REF) :
                                     mc_spectrum(PLAN, NS; MCKW..., scattering = true, bias = B0, backend = EXEC_REF)
    d = result_leaves(r)
    reference[leg] = (time = t, leaves = d, crc = crc_leaves(d), zones = r.n_zones)
    line(@sprintf("reference %-7s mc_spectrum on %s, cold: %.2f s, %d zones, %d primaries, %d children, %d escaped, crc %08x",
                  leg, REF_BACKEND, t, r.n_zones, r.n_gen, r.n_children, r.n_recorded, crc_leaves(d)))
end
const MEMO = PR.ZONE_MEMO
isequal(MEMO.tkey, (NS, MC_NU_MIN, MC_NU_MAX, MC_N_ESAMP)) || error("memo holds the tables of $(MEMO.tkey)")
const SITES, R_ZONES = MEMO.zones
const TABLES_REF = MEMO.tables
const HC = PLAN.hotcross[]
HC === nothing && error("no hot-cross table after the reference Compton call")
nsp() = NS / (log(MC_NU_MAX) - log(MC_NU_MIN))
tables_same(a, b) = all(f -> same_bits(getfield(a[1], f), getfield(b[1], f)), fieldnames(typeof(a[1]))) &&
                    same_bits(a[2], b[2]) && same_bits(a[3], b[3])

# One transport of leg `leg` on `exec` with the tables `tabs` (of the first `n` primaries).
function transport_call(exec, tabs, leg, n = nothing)
    wgt, Nsz, dnmax = tabs
    sc = leg == "compton"
    offset = PR.zone_offsets(Nsz, UInt32(SEED))
    N = n === nothing ? offset[end] : n
    ctl = CurvedMarch(ST; C = PLAN.budget.C, escape = ESCAPE, R_em = max(R_EM, R_ZONES))
    K = sc ? MAX_SCATT : 0
    opac = Opacities(PLAN.params; table = sc ? HC : nothing,
                     bias = StateBias(sc ? B0 : 1.0, mean_weight(wgt), PR.MC_BIAS_MAX, 1, 1.0), max_scatt = K)
    tr = PR.MCTransport(ctl, ST, PLAN.provider, opac, Val(true), Val(sc))
    orders = ntuple(_ -> ExactBins(BANDS; nu_min = 1.0e8, nu_max = 1.0e24, n_ebins = 160), K + 1)
    ledger = PR.mc_ledger()
    source = PR.ZoneSource(SITES, PLAN.provider, ST, PLAN.params, wgt, EMISSIVITY_TABLE, dnmax, offset)
    nch = PR.run_layers(exec, source, tr, UInt32(SEED), N, orders, ledger, widest_layout(max(1, N), K), sc, nothing)
    d = phase_leaves([Spectrum(b) for b in orders], N, nch, PR.ledger_count(ledger, PR.ESCAPED), PR.ledger_totals(ledger))
    led = Dict{String,Any}("ledger_sum" => [PR.ledger_sum(ledger, k) for k in 1:11],
                           "ledger_count" => [PR.ledger_count(ledger, k) for k in 1:11])
    return d, led
end

function timed_call(f)
    l0 = loadavg(); b0, o0, e0 = busy_ticks(), own_ticks(), time()
    s = @timed f()
    b1, o1, e1 = busy_ticks(), own_ticks(), time(); l1 = loadavg()
    return s.value, (time = s.time, gctime = s.gctime, bytes = s.bytes, t_start = e0, t_end = e1,
                     cpu_per_thread = (o1 - o0) / 100 / Threads.nthreads(), load0 = l0, load1 = l1,
                     foreign = ((b1 - b0) - (o1 - o0)) / (100 * (e1 - e0) * length(MYCPUS)))
end

ok = true
runs = NamedTuple[]
for r in RUNS
    bk, set = split_spec(r.spec)
    exec = executor(bk)
    line("== $(r.name): $(r.spec), " * sprint(show, describe(exec)))
    t1 = @elapsed in_settings(() -> PR.grid_tables(exec, SITES, nsp(); n_esamp = MC_N_ESAMP, nu_min = MC_NU_MIN,
                                                   nu_max = MC_NU_MAX), set)
    t2 = @elapsed tabs = in_settings(() -> PR.grid_tables(exec, SITES, nsp(); n_esamp = MC_N_ESAMP,
                                                          nu_min = MC_NU_MIN, nu_max = MC_NU_MAX), set)
    tsame = tables_same(tabs, TABLES_REF)
    global ok &= tsame
    line(@sprintf("%s tables on its executor: %.3f s (first), %.3f s; bitwise those of the reference: %s",
                  r.name, t1, t2, tsame))
    legs = Dict{String,Any}()
    for leg in ("thin", "compton")
        t_c = @elapsed in_settings(() -> transport_call(exec, tabs, leg, 2000), set)
        calls = NamedTuple[]
        for k in 1:NCALLS
            (d, led), rec = timed_call(() -> in_settings(() -> transport_call(exec, tabs, leg), set))
            nd = differing(d, reference[leg].leaves)
            ld = isempty(calls) ? String[] : differing(led, calls[1].ledger)
            push!(calls, merge(rec, (leaves = d, ledger = led, crc = crc_leaves(merge(d, led)),
                                     same_reference = isempty(nd), same_ledger = isempty(ld))))
            isempty(nd) || line("  $(r.name) $leg call $k differs from the reference in $(join(nd, " "))")
            isempty(ld) || line("  $(r.name) $leg call $k ledger differs from call 1 in $(join(ld, " "))")
            global ok &= isempty(nd) & isempty(ld)
            line(@sprintf("%s %-7s call %d %9.3f s  gc %.3f s  cpu/thread %7.3f s  foreign %.3f  load %.1f -> %.1f  %d primaries %d children  crc %08x",
                          r.name, leg, k, rec.time, rec.gctime, rec.cpu_per_thread, rec.foreign, rec.load0,
                          rec.load1, d["n_primary"], d["n_children"], calls[end].crc))
        end
        ts = [c.time for c in calls]
        line(@sprintf("%s %-7s median %.4f s (calls 2-%d: %.4f s), compile call %.2f s, bits: reference %s, ledger %s",
                      r.name, leg, median(ts), NCALLS, NCALLS > 1 ? median(ts[2:end]) : NaN, t_c,
                      all(c -> c.same_reference, calls), all(c -> c.same_ledger, calls)))
        legs[leg] = (compile = t_c, calls = calls)
    end
    push!(runs, (name = r.name, spec = r.spec, backend = bk, settings = string(set),
                 executor = sprint(show, describe(exec)), tables = (first = t1, again = t2, same = tsame),
                 legs = legs))
end
mkpath(OUT)
out = joinpath(OUT, LABEL * ".bin")
serialize(out, Dict{String,Any}("meta" => (label = LABEL, commit = GIT, tree = readchomp(`git rev-parse --short HEAD`),
                                            threads = Threads.nthreads(), cpus = CPUSTR, ns = NS, ncalls = NCALLS,
                                            ref = REF_BACKEND, zones = length(SITES), r_zones = R_ZONES, date = string(now()),
                                            job = get(ENV, "SLURM_JOB_ID", ""), host = gethostname(),
                                            cvd = get(ENV, "CUDA_VISIBLE_DEVICES", ""), load_s = t_load),
                                "reference" => reference, "runs" => runs))
line("mcsched $LABEL written: $out")
line(ok ? "mcsched done" : "mcsched: tables or bits differ")
exit(ok ? 0 : 3)
