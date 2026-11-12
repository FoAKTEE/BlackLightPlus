# fig_scaling_cpu.jl — strong and weak scaling of BlackLight+ on CPU threads, every call kept
# (the CPU half of the performance figures). One process per thread count; the driver
# run_scaling_cpu.sh pins each process to a CPU set with taskset and repeats the ladder.
#
# Deck: the Kerr accretion flow of the appendix table (fig_pol_hires.jl, fig_mc_kerr_common.jl):
# AthenaK snapshot 00999 (BLP_TEST_GRMHD) on KerrSchildCartesian(0.9), black-hole mass
# 6.5e9 x 1.98847e33 g, density scale 2.272651712e23 g (M87_DENSITY_SCALE["00999"] of
# m87_scaling.jl), 230 GHz, inclination 60 deg, distance 1000, field of view 30, C = 80,
# 2500 steps, escape at 1.1 times the distance, R_low 1, R_high 10, gamma 13/9 with
# total-pressure matching, sigma cut 1, no Theta_e floor.
#
# Legs (each process runs the legs it is given on its T = Threads.nthreads() threads):
#   amr        strong: the adaptive polarized production image, render_amr(plan(384, IQUV());
#              block = 16, max_level = 2), 1536^2 effective
#   i768       strong: the uniform Stokes I image, render(plan(768, StokesI()))
#   wiquv      weak: the uniform polarized image of round(NW_IQUV sqrt(T)) pixels per side
#   wi         weak: the uniform Stokes I image of round(NW_I sqrt(T)) pixels per side
#   mcthin     weak: mc_spectrum without scattering (absorption only) from NS_THIN T primaries
#   mccompton  weak: mc_spectrum with Compton scattering (bias 1e4, at most three
#              scatterings) from NS_COMPTON T primaries
# The Monte Carlo legs: stride 1 (every owned cell above the horizon emits), seed 0x5eed0700,
# generation 1e8-1e16 Hz, 160 bins over 1e8-1e24 Hz, the default inclination bands.
# An imaging leg compiles on a small plan of its product (32^2; the adaptive leg 64^2), then
# times its calls. A Monte Carlo leg reads the zones and the generation tables that mode
# `tables` wrote for its primary count and puts them in the package's memo of the last
# grid run (Products.ZONE_MEMO, which a call on the same snapshot and settings reuses
# instead of building them); it compiles with one call on the tables of one thread's
# primaries and times its calls: the warm call of a repeated run, whose cost is the photon
# phase. The bins of the compiling call and of the last timed call must equal those that
# mode `tables` recorded for the same primary count, and the memo must still hold the
# tables read (no rebuild); the process exits 3 otherwise.
#
# Per timed call: wall time, GC time, bytes allocated, and its start and end (Unix time).
# Per leg: the load average before and after, the share of the process's CPUs that other
# processes used over the timed calls, the rays (or primaries, children), the CRC-32C of the
# image planes or of the bins. One JSON line per leg is appended to BLP_SCALING_OUT
# (default figures/paper/out/scaling_cpu/records.jsonl).
#
# Usage (from any directory; the package from the active project):
#   julia -t T --project=<env> fig_scaling_cpu.jl run <label> <leg>:<calls>[,<leg>:<calls>...]
#   julia -t T --project=<env> fig_scaling_cpu.jl tables <T1,T2,...>
# Environment: BLP_SCALING_OUT (records), BLP_SCALING_TABLES (directory of the Monte Carlo
# tables, default figures/paper/out/scaling_cpu/tables), BLP_SCALING_ROUND and
# BLP_SCALING_CPUSET (labels written into every record), BLP_TEST_GRMHD (the snapshot).
using Dates, Printf, Serialization, Statistics, CRC32c
using BlackLightPlus
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Execution: resolve, describe
import BlackLightPlus.Products as PR

const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const OUTDIR = joinpath(dirname(dirname(@__DIR__)), "figures", "paper", "out", "scaling_cpu")
const OUT = get(ENV, "BLP_SCALING_OUT", joinpath(OUTDIR, "records.jsonl"))
const TABLES = get(ENV, "BLP_SCALING_TABLES", joinpath(OUTDIR, "tables"))

const M_BH = 6.5e9 * 1.98847e33                 # g (M87_MASS_G)
const MASS_SCALE = 2.272651712e23               # g (M87_DENSITY_SCALE["00999"].mass_scale_g)
const NW_IQUV = 96                              # pixels per side of the weak polarized image at T = 1
const NW_I = 128                                # the same, Stokes I
const NS_THIN = 3.0e4                           # primaries per thread, absorption only
const NS_COMPTON = 2.5e3                        # primaries per thread, Compton scattering
const SEED = 0x5eed0700
const MCKW = (stride = 1, seed = SEED, spec_nu_min = 1.0e8, spec_nu_max = 1.0e24, n_ebins = 160)

deck(res) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = 230.0e9,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                          sigma_cut = 1.0, theta_e_min = -1.0),
    max_steps = 2500, C = 80.0, escape_factor = 1.1)

line(s) = (println(s); flush(stdout))

# ---------------------------------------------------------------- process facts

const PKGROOT = dirname(dirname(pathof(BlackLightPlus)))
const COMMIT = try
    sha = readchomp(`git -C $PKGROOT rev-parse --short HEAD`)
    isempty(readchomp(`git -C $PKGROOT status --porcelain -- src ext Project.toml`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
function cpu_list(s)
    out = Int[]
    for r in split(s, ',')
        a, b = occursin('-', r) ? split(r, '-') : (r, r)
        append!(out, parse(Int, a):parse(Int, b))
    end
    return out
end
const CPUSTR = String(match(r"Cpus_allowed_list:\s*(\S+)", read("/proc/self/status", String))[1])
const MYCPUS = cpu_list(CPUSTR)
const T = Threads.nthreads()

# Busy ticks of this process's CPUs (all processes) and of this process, for the share of
# the CPUs that other processes took.
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

# ---------------------------------------------------------------- JSON (one line per record)

js(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
js(x::Bool) = x ? "true" : "false"
js(x::Unsigned) = js(string(x; base = 16, pad = 2 * sizeof(x)))
js(x::Integer) = string(x)
js(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
js(::Nothing) = "null"
js(x::Symbol) = js(string(x))
js(x::Union{AbstractVector,Tuple}) = "[" * join(js.(collect(x)), ",") * "]"
js(x::NamedTuple) = js(Dict{String,Any}(string(k) => v for (k, v) in pairs(x)))
js(d::AbstractDict) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in sort(collect(d); by = first)), ",") * "}"

function emit(rec)
    mkpath(dirname(OUT))
    open(io -> println(io, js(rec)), OUT, "a")
end

base_record(leg) = Dict{String,Any}(
    "leg" => leg, "threads" => T, "cpus" => CPUSTR, "ncpus" => length(MYCPUS),
    "cpuset" => get(ENV, "BLP_SCALING_CPUSET", ""), "round" => get(ENV, "BLP_SCALING_ROUND", ""),
    "label" => length(ARGS) >= 2 ? ARGS[2] : "", "commit" => COMMIT, "host" => gethostname(),
    "julia" => string(VERSION), "cpu_model" => Sys.cpu_info()[1].model, "date" => string(now()),
    "pid" => getpid(), "executor" => sprint(show, describe(resolve(:cpu))))

# ---------------------------------------------------------------- hashes

planes(x::Image) = (x.I, x.Q, x.U, x.V)
planes(x::NamedTuple) = planes(x.image)
crc_planes(x) = [crc32c(reinterpret(UInt8, vec(collect(m)))) for m in planes(x)]
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

# ---------------------------------------------------------------- the timed calls

# `ncall` timed calls of `f` with the load, the foreign CPU share and each call's window.
function timed_calls(f, ncall)
    GC.gc()
    b0, o0, l0, t0 = busy_ticks(), own_ticks(), loadavg(), time()
    calls = Dict{String,Any}[]
    r = nothing
    for _ in 1:ncall
        ts = time()
        s = @timed f()
        r = s.value
        push!(calls, Dict{String,Any}("time" => s.time, "gctime" => s.gctime, "bytes" => s.bytes,
                                      "t_start" => ts, "t_end" => time()))
    end
    b1, o1, l1, t1 = busy_ticks(), own_ticks(), loadavg(), time()
    foreign = ((b1 - b0) - (o1 - o0)) / (100 * (t1 - t0) * length(MYCPUS))
    return r, calls, Dict{String,Any}("load0" => l0, "load1" => l1, "foreign" => foreign,
                                      "t_start" => t0, "t_end" => t1)
end

function report(leg, rec)
    ts = [c["time"] for c in rec["calls"]]
    line(@sprintf("%-10s T=%-3d %s median %.4f s min %.4f max %.4f (%d calls) compile %.2f s load %.1f->%.1f foreign %.3f %s",
                  leg, T, haskey(rec, "res") ? "res $(rec["res"])" : "Ns $(get(rec, "Ns", 0))",
                  median(ts), minimum(ts), maximum(ts), length(ts), rec["first"], rec["load0"],
                  rec["load1"], rec["foreign"],
                  haskey(rec, "crc") ? "crc " * join(string.(rec["crc"]; base = 16), " ") : ""))
end

weak_res(n0) = round(Int, n0 * sqrt(T))

function image_leg(snap, leg, ncall)
    st = KerrSchildCartesian(0.9)
    mk(res, mode) = prepare(deck(res); spacetime = st, matter = snap, mode = mode)
    if leg == "amr"
        small, big = mk(64, IQUV()), mk(384, IQUV())
        f = p -> render_amr(p; block = 16, max_level = 2, backend = :cpu)
        res, kind = 384, "amr_IQUV"
    elseif leg == "i768"
        small, big = mk(32, StokesI()), mk(768, StokesI())
        f = p -> render(p; backend = :cpu)
        res, kind = 768, "I"
    elseif leg == "wiquv"
        res = weak_res(NW_IQUV)
        small, big = mk(32, IQUV()), mk(res, IQUV())
        f = p -> render(p; backend = :cpu)
        kind = "IQUV"
    elseif leg == "wi"
        res = weak_res(NW_I)
        small, big = mk(32, StokesI()), mk(res, StokesI())
        f = p -> render(p; backend = :cpu)
        kind = "I"
    else
        error("unknown leg $leg")
    end
    t_first = @elapsed f(small)
    r, calls, w = timed_calls(() -> f(big), ncall)
    rec = merge(base_record(leg), w)
    rec["scaling"] = leg in ("amr", "i768") ? "strong" : "weak"
    rec["product"] = kind
    rec["res"] = res
    rec["rays"] = r isa NamedTuple ? r.nrays : res^2
    rec["first"] = t_first
    rec["calls"] = calls
    rec["crc"] = crc_planes(r)
    return rec
end

tables_file(leg, Ns) = joinpath(TABLES, @sprintf("%s_%d.bin", leg, round(Int, Ns)))
const ZONES_FILE = joinpath(TABLES, "zones.bin")

mc_call(plan, leg, Ns) = leg == "mcthin" ? mc_spectrum(plan, Ns; MCKW..., backend = :cpu) :
    mc_spectrum(plan, Ns; MCKW..., scattering = true, bias = 1.0e4, backend = :cpu)
mc_tkey(Ns) = (Float64(Ns), Float64(PR.MC_NU_MIN), Float64(PR.MC_NU_MAX), Int(PR.MC_N_ESAMP))
mc_zkey(plan) = (plan.spacetime, plan.params,
                 (stride = 1, r_emit_max = Inf, max_zones = typemax(Int)), nothing)
mc_plan(snap) = prepare(deck(64); spacetime = KerrSchildCartesian(0.9), matter = snap)

# Put the zones and the tables of `Ns` in the memo of the last grid run.
function inject!(plan, zones, tables, Ns)
    blocks = PR.zone_blocks(plan.provider)
    m = PR.ZONE_MEMO
    lock(PR.ZONE_MEMO_LOCK) do
        m.data = WeakRef(blocks.data); m.blocks = blocks; m.zkey = mc_zkey(plan)
        m.zones = zones; m.tkey = mc_tkey(Ns); m.tables = tables
    end
end

function mc_leg(snap, plan, zones, leg, ncall)
    per = leg == "mcthin" ? NS_THIN : NS_COMPTON
    Ns = per * T
    # compile on the tables of one thread's primaries, then time the run of T threads'
    ref1 = deserialize(tables_file(leg, per))
    inject!(plan, zones, ref1["tables"], per)
    t_first = @elapsed r1 = mc_call(plan, leg, per)
    ok1 = crc_mc(r1) == ref1["crc"]
    ref = deserialize(tables_file(leg, Ns))
    ref["tkey"] == mc_tkey(Ns) || error("$leg: tables of $(ref["tkey"]), not $(mc_tkey(Ns))")
    tables = ref["tables"]
    inject!(plan, zones, tables, Ns)
    r, calls, w = timed_calls(() -> mc_call(plan, leg, Ns), ncall)
    hit = PR.ZONE_MEMO.tables === tables && PR.ZONE_MEMO.zones === zones
    same = crc_mc(r) == ref["crc"]
    rec = merge(base_record(leg), w)
    rec["scaling"] = "weak"
    rec["product"] = leg == "mcthin" ? "mc_absorption" : "mc_compton"
    rec["Ns"] = Ns
    rec["primaries"] = r.n_gen
    rec["children"] = r.n_children
    rec["zones"] = r.n_zones
    rec["first"] = t_first
    rec["first_Ns"] = per
    rec["calls"] = calls
    rec["crc"] = [crc_mc(r)]
    rec["crc_ref"] = ref["crc"]
    rec["bits_equal_ref"] = same && ok1
    rec["memo_hit"] = hit
    return rec, same && ok1 && hit
end

# ---------------------------------------------------------------- modes

function run_mode(label, spec)
    legs = [(String(a), parse(Int, b)) for (a, b) in (split(s, ':') for s in split(spec, ','))]
    line(@sprintf("fig_scaling_cpu run %s: commit %s, %d threads, cpus %s (%d), legs %s, %s",
                  label, COMMIT, T, CPUSTR, length(MYCPUS), spec, now()))
    t_load = @elapsed snap = AthenaKSnapshot(DATA)
    line(@sprintf("snapshot loaded in %.2f s", t_load))
    ok = true
    zones = nothing
    plan = nothing
    for (leg, n) in legs
        if startswith(leg, "mc")
            if zones === nothing
                t_z = @elapsed zones = deserialize(ZONES_FILE)
                plan = mc_plan(snap)
                line(@sprintf("zones read in %.2f s: %d zones", t_z, length(zones[1])))
            end
            rec, good = mc_leg(snap, plan, zones, leg, n)
            ok &= good
            good || line("  $leg: bits equal to the reference $(rec["bits_equal_ref"]), memo kept $(rec["memo_hit"])")
        else
            rec = image_leg(snap, leg, n)
        end
        rec["snapshot_load"] = t_load
        emit(rec)
        report(leg, rec)
    end
    line("fig_scaling_cpu run $label done $(now())")
    ok || exit(3)
end

# The zones of the deck, and for every thread count of the ladder the generation tables
# of both Monte Carlo legs with the bins of a cold call (the reference of mc_leg).
function tables_mode(ladder)
    mkpath(TABLES)
    snap = AthenaKSnapshot(DATA)
    plan = mc_plan(snap)
    runs = Dict{String,Any}[]
    for n in ladder, leg in ("mcthin", "mccompton")
        Ns = (leg == "mcthin" ? NS_THIN : NS_COMPTON) * n
        f = tables_file(leg, Ns)
        isfile(f) && (line("$f exists"); continue)
        t = @elapsed r = mc_call(plan, leg, Ns)
        m = PR.ZONE_MEMO
        m.tkey == mc_tkey(Ns) || error("memo holds $(m.tkey)")
        isfile(ZONES_FILE) || serialize(ZONES_FILE, m.zones)
        serialize(f, Dict{String,Any}("tkey" => m.tkey, "tables" => m.tables, "crc" => crc_mc(r),
                                      "n_gen" => r.n_gen, "n_children" => r.n_children,
                                      "commit" => COMMIT, "threads" => T))
        push!(runs, Dict{String,Any}("leg" => leg, "threads_of_ladder" => n, "Ns" => Ns, "seconds" => t,
                                    "primaries" => r.n_gen, "children" => r.n_children,
                                    "zones" => r.n_zones, "crc" => crc_mc(r)))
        line(@sprintf("tables %-9s T=%-3d Ns %.4g: %.2f s (cold call), %d primaries %d children %d zones crc %08x",
                      leg, n, Ns, t, r.n_gen, r.n_children, r.n_zones, crc_mc(r)))
    end
    open(io -> println(io, js(Dict{String,Any}("mode" => "tables", "commit" => COMMIT, "threads" => T,
                                               "cpus" => CPUSTR, "date" => string(now()), "runs" => runs))),
         joinpath(TABLES, "tables_log.jsonl"), "a")
end

if ARGS[1] == "run"
    run_mode(ARGS[2], ARGS[3])
elseif ARGS[1] == "tables"
    tables_mode(parse.(Int, split(ARGS[2], ',')))
else
    error("usage: fig_scaling_cpu.jl run <label> <legs> | tables <ladder>")
end
