# fig_mc_binary.jl — Monte Carlo spectra of the circumbinary GRMHD snapshot 00455 on the
# orbiting binary (Fig. mc_binary and Fig. mc_binary_conv): without and with Compton
# scattering; the same plasma on the binary held at rest and on the orbit without the
# boost of each black hole's term; the edge-on spectrum by viewing azimuth (orbital phase);
# the gas inside the excision radius removed; convergence in the superphoton number.
#
# Deck: AthenaK snapshot torus.mhd_w_bcc.00455.bin of the binary black hole accreting from
# a torus (equal masses 0.5, separation 25 M, non-spinning, Omega = 25^(-3/2), period
# 785.4 M, adiabatic index 13/9, excision radius 4 M), with the binary phase
# phi0 = rem2pi(Omega t_dump) from the dump's own header. Scaling of Fig. binary (a):
# 6.5e9 Msun per black hole (code mass 1.3e10 Msun) at 16.8 Mpc, the mass scale MASS_SCALE at
# which the 230 GHz flux density seen from i = 17 deg is 0.5 Jy (static metric), thermal
# electrons R_low = 1, R_high = 40, mu = 0.5, sigma cut 1.
#
# Spacetimes (all `BoostedSuperposedKerrSchild`, one code path):
#   moving   the boosted superposition on the circular orbit (CircularOrbit with phase phi0
#            at t = 0), the metric of the evolution, read at each photon's own time;
#   unboost  the same orbit with unboosted terms (boost = false): at t = 0 bitwise the
#            metric of `rest`, so its emission zones and primaries are those of `rest`;
#   rest     the black holes frozen at their t = 0 positions with zero velocity: the static
#            superposed metric of the images.
# The plasma is the one snapshot, held fixed; every photon leaves its zone at t = 0.
#
# Emission zones: every owned cell centre of the snapshot above the horizon proxy and within
# r < BLP_MCB_REMIT = 1000 M of the origin (`Data.emission_sites` at stride 1; the cells beyond
# carry 1e-3 of the emission at 1 GHz and less above), with the zone tables of
# `Data.zone_tables` built once per metric at Ns = BLP_MCB_NREF and rescaled for every photon
# count (the weights scale as 1/Ns, the counts and envelopes as Ns).
#
# Monte Carlo: the package's generation, transport and scattering (`BlackLightPlus.Products`
# internals, the body of `mc_spectrum` for a plasma grid) run by `mc_sky` below, which bins
# the escaping photons in `SkyBands`: the inclination bands of `mc_spectrum`, then NAZ
# azimuth sectors of each band, then the whole sky. The `check` leg shows the band rows and
# the sky row bitwise equal to `mc_spectrum` on the same zones. March: C = 80, escape at
# 1000 M or outbound beyond max(100, 1.2 r_zone,max). Generation 1e8-1e16 Hz; spectra on 200
# log bins over 1e8-1e24 Hz. Scattering: the package's hot-cross table, the state bias
# b0 = BLP_MCB_B0 with the order factor BLP_MCB_BIAS_ORDER, capped at BLP_MCB_BIAS_MAX, at most 3
# scatterings. The gas next to the black holes is thick to scattering: with the order factor
# of the Kerr accretion flow (300) the later generations multiply there and their children fill the memory
# (more than 100 GB within minutes), so the binary runs without it and with the cap 1e4.
#
# Legs (BLP_MCB_LEGS, comma-separated; default all, in this order):
#   check    mc_spectrum(plan, NCHECK; stride = 8) on the moving binary, thin and Compton,
#            against mc_sky on the same zones (rows, counts, ledger bitwise);
#   zones    the emission of the zones by Kerr-Schild radius about the nearer black hole and
#            by distance from the origin, at seven frequencies;
#   prod     moving binary, thin and Compton (BLP_MCB_PROD_SCAT), NPROD primaries, seed
#            BLP_MCB_PROD_SEED (runs of another seed than 1 carry the suffix _s<seed>);
#   time     rest and unboost, thin and Compton (BLP_MCB_TIME_SCAT), NTIME primaries, and the
#            moving binary at the same NTIME and seed (`moving3_*`), for NSEED_TIME seeds, so
#            the ratios to `rest` carry the seed-to-seed scatter of paired runs;
#   mask     moving binary with the gas within R_EXC of either t = 0 puncture removed from
#            emission, absorption and scattering, thin and Compton (BLP_MCB_MASK_N primaries);
#   conv     moving binary, thin, N in NCONV, NSEED seeds each.
#
# Writes to BLP_OUT (default figures/paper/out/mc_binary):
#   nu.npy                    bin centres (Hz), shape (1, 200)
#   rows.json                 the row layout of SkyBands (inclinations, sectors, solid angles)
#   S_<run>.npy               (4 rows, bins): for row j and quantity q (nu L_nu in erg/s,
#                             isotropic equivalent; its standard error; the effective photon
#                             number; the photon count) the matrix row q + 4 (j - 1)
#   O_<run>.npy               the same for each scattering order n: matrix row
#                             q + 4 (j - 1) + 4 rows n
#   rec_<leg>.json, run_<tag>.json   setup, checks, emission shares; every run's ledger
#
# Env: BLP_MCB_MASS_SCALE (the mass scale, g; default the 00455 scale of m87_scaling.jl), BLP_BBH_BIN, BLP_MCB_REMIT (1000), BLP_MCB_N (3e6), BLP_MCB_NTIME (5e6),
# BLP_MCB_NCONV (1e5,3e5,1e6,3e6), BLP_MCB_NSEED (4), BLP_MCB_NCHECK (1e5),
# BLP_MCB_NSEED_TIME (4), BLP_MCB_TIME_SCAT (thin), BLP_MCB_PROD_SCAT (compton),
# BLP_MCB_PROD_SEED (1),
# BLP_MCB_MASK_N (1e7,1e7; target counts of the whole snapshot, of which the zones outside
# the excision radius hold about a twelfth), BLP_MCB_NREF (1e7),
# BLP_MCB_B0 (1e4), BLP_MCB_BIAS_ORDER (1), BLP_MCB_BIAS_MAX (1e4), BLP_MCB_MASK_SCAT
# (thin,compton), BLP_MCB_NAZ (12), BLP_MCB_LEGS, BLP_OUT.
# Run, from the repository root:
#   julia -t 64 --project=. figures/paper/fig_mc_binary.jl
# (plots: paper/scripts/plot_mc_binary.py, paper/scripts/plot_mc_binary_conv.py)

using StaticArrays, Printf, Dates, Statistics, LinearAlgebra
using BlackLightPlus
using BlackLightPlus.Data: athenak_bin_snapshot, emission_sites, zone_tables, write_npy
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, proxy_distance,
                                metric, ks_radius
using BlackLightPlus.Matter: GridProvider, PlasmaPrimitives
import BlackLightPlus.Matter: primitives, domain_radius
using BlackLightPlus.MonteCarlo: CurvedMarch, WeightTable, mean_weight, EMISSIVITY_TABLE, int_jnu
using BlackLightPlus.Numerics: widest_layout, ME, CL
using BlackLightPlus.Execution: Bands, InclinationBands, ExactBins, Spectrum, band, nu_centers,
                                resolve
using BlackLightPlus.Plans: hotcross_table
import BlackLightPlus.Execution: nrows, solid_angles, total_row, record_rows!
const PR = BlackLightPlus.Products
const EX = BlackLightPlus.Execution
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

const ROOT = dirname(dirname(@__DIR__))
const BIN = get(ENV, "BLP_BBH_BIN",
    "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
_int(s) = round(Int, parse(Float64, s))
const REMIT = parse(Float64, get(ENV, "BLP_MCB_REMIT", "1000"))
const NPROD = _int(get(ENV, "BLP_MCB_N", "3e6"))
const NTIME = _int(get(ENV, "BLP_MCB_NTIME", "5e6"))
const NCONV = _int.(_list(get(ENV, "BLP_MCB_NCONV", "1e5,3e5,1e6,3e6")))
const NSEED = _int(get(ENV, "BLP_MCB_NSEED", "4"))
const NSEED_TIME = _int(get(ENV, "BLP_MCB_NSEED_TIME", "4"))
const TIME_SCAT = Bool[x == "compton" for x in _list(get(ENV, "BLP_MCB_TIME_SCAT", "thin"))]
const PROD_SCAT = Bool[x == "compton" for x in _list(get(ENV, "BLP_MCB_PROD_SCAT", "compton"))]
const PROD_SEED = _int(get(ENV, "BLP_MCB_PROD_SEED", "1"))     # seed 1: run moving_*; else moving_*_s<seed>
const MASK_N = _int.(_list(get(ENV, "BLP_MCB_MASK_N", "1e7,1e7")))      # thin, Compton
const MASK_SCAT = Bool[x == "compton" for x in _list(get(ENV, "BLP_MCB_MASK_SCAT", "thin,compton"))]
const NCHECK = _int(get(ENV, "BLP_MCB_NCHECK", "1e5"))
const NREF = parse(Float64, get(ENV, "BLP_MCB_NREF", "1e7"))
const B0 = parse(Float64, get(ENV, "BLP_MCB_B0", "1e4"))
const BIAS_ORDER = parse(Float64, get(ENV, "BLP_MCB_BIAS_ORDER", "1"))
const BIAS_MAX = parse(Float64, get(ENV, "BLP_MCB_BIAS_MAX", "1e4"))   # caps the events of a step in dense gas
const NAZ = _int(get(ENV, "BLP_MCB_NAZ", "12"))
const LEGS = _list(get(ENV, "BLP_MCB_LEGS", "check,zones,prod,time,mask,conv"))
const OUT = get(ENV, "BLP_OUT", joinpath(ROOT, "figures", "paper", "out", "mc_binary"))

const M_BH = 2 * M87_MASS_G              # code mass unit: the total binary mass
# g: the mass scale of Fig. binary (a), at which the 230 GHz flux density seen from i = 17 deg at
# 16.8 Mpc is 0.5 Jy (m87_scaling.jl, run 00455)
const MASS_SCALE = haskey(ENV, "BLP_MCB_MASS_SCALE") ? parse(Float64, ENV["BLP_MCB_MASS_SCALE"]) :
                   m87_density_scale("00455")
const INCS = (17.0, 50.0, 90.0)
const HALF_WIDTH = 10.0
const CVAL = 80.0
const NU_GEN = (1.0e8, 1.0e16)
const GRID = (nu_min = 1.0e8, nu_max = 1.0e24, n_ebins = 200)
const MAX_SCATT = 3
const ESCAPE = 1.0e3
const R_EM = 100.0
const R_EXC = 4.0
const CHECK_STRIDE = 8
const MEC2 = ME * CL * CL

const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# ---------------------------------------------------------------- JSON
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = isfinite(x) ? @sprintf("%.10g", x) : "null"
_json(x::Nothing) = "null"
_json(x::Symbol) = _json(string(x))
_json(x::Union{AbstractVector,Tuple}) = "[" * join(map(_json, collect(x)), ", ") * "]"
_json(x::NamedTuple) = _json(Dict(string(k) => v for (k, v) in pairs(x)))
_json(d::AbstractDict) = "{" * join([_json(string(k)) * ": " * _json(v) for (k, v) in sort(collect(d); by = first)], ", ") * "}"
writejson(path, d) = open(io -> println(io, _json(d)), path, "w")
lg(a...) = (println(a...); flush(stdout))

# ---------------------------------------------------------------- binary from the dump header
"Dump time, separation, mass ratio, adiabatic index and excision radii from the .bin header."
function dump_binary(path)
    tt = NaN; text = ""
    open(path, "r") do io
        H = -1
        while H < 0
            l = readline(io)
            startswith(l, "  time=") && (tt = parse(Float64, strip(l[8:end])))
            startswith(l, "  header offset=") && (H = parse(Int, strip(l[17:end])))
        end
        text = String(read(io, H))
    end
    par = Dict{String,Dict{String,String}}()
    blk = ""
    for l in split(text, '\n')
        s = strip(l)
        if startswith(s, "<") && endswith(s, ">")
            blk = s[2:end-1]; par[blk] = get(par, blk, Dict{String,String}())
        elseif occursin('=', s) && !isempty(blk)
            k, v = split(s, '='; limit = 2)
            par[blk][strip(k)] = strip(split(v, '#')[1])
        end
    end
    num(b, k, d) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : d
    tlim = num("time", "tlim", NaN)
    t = isfinite(tlim) && abs(tlim - tt) <= 0.5 ? tlim : tt
    sep = num("problem", "sep", NaN); q = num("problem", "q", 1.0)
    (num("problem", "a1", 0.0) == 0.0 && num("problem", "a2", 0.0) == 0.0 && q == 1.0) ||
        error("$path: this generator takes the equal-mass non-spinning binary")
    Om = sep^-1.5
    return (t = t, t_header = tt, sep = sep, q = q, Omega = Om, period = 2π / Om,
            phi0 = rem2pi(Om * t, RoundNearest), gamma = num("mhd", "gamma", 13 / 9),
            excise_rad = (num("coord", "excise_1_rad", NaN), num("coord", "excise_2_rad", NaN)))
end

# ---------------------------------------------------------------- binning: bands and sectors
"""
    SkyBands(incl_deg, half_width_deg, naz)

The rows of `InclinationBands(incl_deg, half_width_deg)` (rows `1:N`, the polar angle of
the escape direction about the orbital axis, azimuth averaged), then every band split
into `naz` equal sectors of the escape azimuth `φ = atan(d², d¹) ∈ [0, 2π)` (row
`N + (k − 1) naz + j` is sector `j` of band `k`, solid angle `ΔΩ_k / naz`), then the
whole sky.
"""
struct SkyBands{N} <: Bands
    incl_deg::NTuple{N,Float64}
    half_width_deg::Float64
    lo::NTuple{N,Float64}
    hi::NTuple{N,Float64}
    naz::Int
end
function SkyBands(incl, hw, naz)
    ib = InclinationBands(incl, hw)
    return SkyBands(ib.incl_deg, ib.half_width_deg, ib.lo, ib.hi, Int(naz))
end
nrows(b::SkyBands{N}) where {N} = N * (b.naz + 1) + 1
function solid_angles(b::SkyBands{N}) where {N}
    dO = [2.0 * π * (cos(b.lo[k]) - cos(b.hi[k])) for k in 1:N]
    return [dO; vec([dO[k] / b.naz for j in 1:b.naz, k in 1:N]); 4.0 * π]
end
total_row(b::SkyBands) = nrows(b)
@inline function record_rows!(b::ExactBins, r::SkyBands{N}, off::Int, ie::Int, E, w, d) where {N}
    ksp = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    ksp <= 0.0 && return nothing
    th = acos(clamp(d[3] / ksp, -1.0, 1.0))
    ph = atan(d[2], d[1])
    ph < 0.0 && (ph += 2.0 * π)
    j = clamp(unsafe_trunc(Int, floor(ph * r.naz / (2.0 * π))) + 1, 1, r.naz)
    EX.add_cell!(b, off + N * (r.naz + 1) + 1, ie, E, w)
    for k in 1:N
        if th >= r.lo[k] && th <= r.hi[k]
            EX.add_cell!(b, off + k, ie, E, w)
            EX.add_cell!(b, off + N + (k - 1) * r.naz + j, ie, E, w)
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the excised gas removed
"The snapshot with the gas within coordinate distance `r` of either t = 0 puncture removed."
struct ExciseMask{P} <: GridProvider
    snap::P
    s1::SVector{3,Float64}
    s2::SVector{3,Float64}
    r::Float64
end
domain_radius(m::ExciseMask) = domain_radius(m.snap)
@inline function primitives(m::ExciseMask, x::SVector{4})
    p = SVector(Float64(x[2]), Float64(x[3]), Float64(x[4]))
    (norm(p - m.s1) < m.r || norm(p - m.s2) < m.r) && return PlasmaPrimitives{Float64}()
    return primitives(m.snap, x)
end

# ---------------------------------------------------------------- zones and the driver
"Emission sites and their generation tables at Ns = Ns_ref photons."
struct Zones{S}
    sites::S
    lwgt::Vector{Float64}
    lnu_min::Float64
    lnu_max::Float64
    dlnu::Float64
    Nsz::Vector{Float64}
    dnmax::Vector{Float64}
    r_max::Float64
    Ns_ref::Float64
    seconds::Float64
end
function Zones(st, sites, Ns_ref)
    t0 = time()
    wgt, Nsz, dnmax = zone_tables(sites, Ns_ref / (log(NU_GEN[2]) - log(NU_GEN[1]));
                                  nu_min = NU_GEN[1], nu_max = NU_GEN[2])
    r_max = maximum(z -> proxy_distance(st, z.x)[2], sites)
    return Zones(sites, wgt.lwgt, wgt.lnu_min, wgt.lnu_max, wgt.dlnu, Nsz, dnmax, r_max,
                 Float64(Ns_ref), time() - t0)
end
"The zones `keep` of `z`, on the weights of `z`."
subset(z::Zones, keep) = Zones(z.sites[keep], z.lwgt, z.lnu_min, z.lnu_max, z.dlnu, z.Nsz[keep],
                               z.dnmax[keep], z.r_max, z.Ns_ref, 0.0)
"The tables of `z` for `Ns` photons: weights over f = Ns/Ns_ref, counts and envelopes times f."
function scaled(z::Zones, Ns)
    f = Ns / z.Ns_ref
    lw = f == 1.0 ? z.lwgt : z.lwgt .- log(f)
    return WeightTable(z.lnu_min, z.lnu_max, z.dlnu, lw), (f == 1.0 ? z.Nsz : z.Nsz .* f),
           (f == 1.0 ? z.dnmax : z.dnmax .* f)
end

"""
    mc_sky(plan, zs, Ns; seed, scattering, b0, provider, bands, exec)

The body of `mc_spectrum(plan, Ns; …)` for a plasma grid on the prebuilt zones `zs`, with
any `Bands` and the plasma `provider` of the transport (the plan's by default). Returns the
per-order spectra, their sum, the ledger totals, the primary and child counts and the wall
time.
"""
function mc_sky(plan, zs::Zones, Ns; seed = 1, scattering = false, b0 = B0,
                bands = SkyBands(INCS, HALF_WIDTH, NAZ), exec = resolve(:cpu),
                provider = plan.provider)
    t0 = time()
    K = scattering ? MAX_SCATT : 0
    useed = UInt32(seed & 0xffffffff)
    wgt, Nsz, dnmax = scaled(zs, Ns)
    offset = PR.zone_offsets(Nsz, useed)
    N = offset[end]
    st = plan.spacetime
    ctl = CurvedMarch(st; C = CVAL, escape = ESCAPE, R_em = max(R_EM, 1.2 * zs.r_max))
    o = (table = scattering ? hotcross_table(plan) : nothing,
         bias = (b = b0, b_max = BIAS_MAX, split = 1, order = BIAS_ORDER), K = K)
    transport = PR.MCTransport(ctl, st, provider, PR.mc_opacities(plan.params, o, mean_weight(wgt)),
                               Val(true), Val(scattering))
    orders = ntuple(_ -> ExactBins(bands; GRID...), K + 1)
    ledger = PR.mc_ledger()
    layout = widest_layout(max(1, N), K)
    source = PR.ZoneSource(zs.sites, provider, st, plan.params, wgt, EMISSIVITY_TABLE, dnmax, offset)
    nch = PR.run_layers(exec, source, transport, useed, N, orders, ledger, layout, scattering, nothing)
    total = similar(orders[1])
    foreach(b -> merge!(total, b), orders)
    lt = PR.ledger_totals(ledger)
    return (spec = Spectrum(total), orders = [Spectrum(b) for b in orders], N = N, n_children = nch,
            n_escaped = PR.ledger_count(ledger, PR.ESCAPED),
            w_primary = lt[1], w_children = lt[2], scatter_loss = lt[4], w_unspawned = lt[6],
            emitted = lt[7], escaped = lt[8], absorbed = lt[9], captured = lt[10], dropped = lt[11],
            wall = time() - t0)
end

"[L, sigma, neff, count] of every row and bin of a spectrum, shape (4, rows, bins)."
function spec_stats(s::Spectrum)
    nr, nb = size(s.count)
    A = zeros(4, nr, nb)
    for j in 1:nr
        f = MEC2 / s.dlnu * (4π / s.dOmega[j])
        for i in 1:nb
            a = s.sumwE[j, i]; a2 = s.sumwE2[j, i]
            A[1, j, i] = f * a; A[2, j, i] = f * sqrt(a2)
            A[3, j, i] = a2 > 0 ? a * a / a2 : 0.0; A[4, j, i] = s.count[j, i]
        end
    end
    return A
end

bitsof(s::Spectrum) = (copy(s.count), reinterpret(UInt64, vec(s.sumw)), reinterpret(UInt64, vec(s.sumwE)),
                       reinterpret(UInt64, vec(s.sumwE2)))
samebits(a::Spectrum, b::Spectrum) = bitsof(a) == bitsof(b)
writerec(name, d) = writejson(joinpath(OUT, "rec_$name.json"), d)

function save_run(tag, r; extra = Dict{String,Any}())
    A = spec_stats(r.spec)
    write_npy(joinpath(OUT, "S_$tag.npy"), reshape(A, :, size(A, 3)))
    if length(r.orders) > 1
        O = cat([spec_stats(s) for s in r.orders]...; dims = 4)
        write_npy(joinpath(OUT, "O_$tag.npy"), reshape(permutedims(O, (1, 2, 4, 3)), :, size(O, 3)))
    end
    rec = merge(Dict{String,Any}("tag" => tag, "N_primaries" => r.N, "n_children" => r.n_children,
                     "n_escaped" => r.n_escaped, "emitted_erg_s" => r.emitted,
                     "escaped_erg_s" => r.escaped, "absorbed_erg_s" => r.absorbed,
                     "captured_erg_s" => r.captured, "dropped_erg_s" => r.dropped,
                     "w_primary" => r.w_primary, "w_children" => r.w_children,
                     "w_unspawned" => r.w_unspawned, "scatter_loss" => r.scatter_loss,
                     "wall_s" => r.wall, "git_sha" => GIT_SHA, "threads" => Threads.nthreads(),
                     "when" => string(now())), extra)
    writejson(joinpath(OUT, "run_$tag.json"), rec)
    lg(@sprintf("  %-22s N=%d children=%d escaped/emitted=%.4f  %.1f s", tag, r.N, r.n_children,
                r.escaped / r.emitted, r.wall))
    return nothing
end

# ---------------------------------------------------------------- setup
mkpath(OUT)
lg("BlackLightPlus ", GIT_SHA, " | threads=", Threads.nthreads(), " | legs ", join(LEGS, ","))
const BB = dump_binary(BIN)
t0 = time()
const SNAP = athenak_bin_snapshot(BIN; gamma = BB.gamma)
const LOAD_S = time() - t0
lg(@sprintf("snapshot loaded in %.1f s (nmb = %d); t = %.6g M, phi0 = %.10g rad", LOAD_S,
            SNAP.blocks.nmb, BB.t, BB.phi0))

const ORBIT = CircularOrbit((0.5, 0.5), BB.sep; phase = BB.phi0)
const S0 = ORBIT(0.0)
const FROZEN = let p1 = S0[1][1], p2 = S0[2][1], z = zero(SVector{3,Float64})
    t -> ((p1, z), (p2, z))
end
const SPINS = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
const ST = Dict(
    "moving" => BoostedSuperposedKerrSchild([0.5, 0.5], SPINS, ORBIT),
    "unboost" => BoostedSuperposedKerrSchild([0.5, 0.5], SPINS, ORBIT; boost = false),
    "rest" => BoostedSuperposedKerrSchild([0.5, 0.5], SPINS, FROZEN; boost = false))
const CFG = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = 230.0e9,
                      plasma = PlasmaConfig(gamma = BB.gamma, rat_low = 1.0, rat_high = 40.0,
                                            mu = 0.5, sigma_cut = 1.0),
                      C = CVAL)
const PLAN = Dict(k => prepare(CFG; spacetime = st, matter = SNAP) for (k, st) in ST)

# the metric of `rest` against the static SuperposedKerrSchild of the images, and of
# `unboost` against `rest` at t = 0, on a lattice of points
const STATIC = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [Tuple(S0[1][1]), Tuple(S0[2][1])])
const METRIC_CHECK = let mx = 0.0, same_u = true, dmov = 0.0
    for i in -20:20, j in -20:20
        x = SVector(0.0, 1.0 * i + 0.37, 1.0 * j - 0.21, 0.6)
        g = metric(ST["rest"], x)
        mx = max(mx, maximum(abs.(g .- metric(STATIC, x))))
        same_u &= g == metric(ST["unboost"], x)
        dmov = max(dmov, maximum(abs.(metric(ST["moving"], x) .- g)))
    end
    (max_abs_rest_vs_static = mx, unboost_equals_rest_at_t0 = same_u, max_abs_moving_vs_rest_t0 = dmov)
end
METRIC_CHECK.unboost_equals_rest_at_t0 || error("unboost differs from rest at t = 0")

const BANDS = SkyBands(INCS, HALF_WIDTH, NAZ)
const NU = nu_centers(Spectrum(ExactBins(BANDS; GRID...)))
write_npy(joinpath(OUT, "nu.npy"), reshape(NU, 1, :))
writejson(joinpath(OUT, "rows.json"),
          Dict("incl_deg" => collect(INCS), "half_width_deg" => HALF_WIDTH, "naz" => NAZ,
               "rows" => "1:N inclination bands (azimuth averaged); N + (k-1) naz + j: sector j of band k, escape azimuth in [(j-1), j) 2pi/naz; last: all sky",
               "dOmega_sr" => solid_angles(BANDS)))
writerec("setup", Dict(
    "git_sha" => GIT_SHA, "julia" => string(VERSION), "threads" => Threads.nthreads(),
    "when" => string(now()), "legs" => LEGS, "snapshot" => BIN, "snapshot_load_s" => LOAD_S,
    "nmb" => SNAP.blocks.nmb, "time_M" => BB.t, "time_header_M" => BB.t_header,
    "separation_M" => BB.sep, "q" => BB.q, "Omega" => BB.Omega, "period_M" => BB.period,
    "phi0_rad" => BB.phi0, "gamma" => BB.gamma, "excise_rad_M" => collect(BB.excise_rad),
    "hole_positions_t0" => [collect(S0[1][1]), collect(S0[2][1])],
    "hole_velocities_t0" => [collect(S0[1][2]), collect(S0[2][2])],
    "m_bh_g" => M_BH, "mass_scale_g" => MASS_SCALE, "R_low" => 1.0, "R_high" => 40.0, "mu" => 0.5,
    "sigma_cut" => 1.0, "L_unit_cm" => PLAN["moving"].params.units.L_unit,
    "zones" => "stride 1, r < $(REMIT) M", "C" => CVAL, "escape_M" => ESCAPE, "R_em_min_M" => R_EM,
    "nu_gen_hz" => collect(NU_GEN), "grid" => GRID, "bias" => "state, b0 = $(B0), order factor $(BIAS_ORDER), cap $(BIAS_MAX)",
    "max_scatt" => MAX_SCATT, "incl_deg" => collect(INCS), "half_width_deg" => HALF_WIDTH,
    "naz" => NAZ, "N" => Dict("prod" => NPROD, "time" => NTIME, "conv" => NCONV, "nseed" => NSEED,
                              "check" => NCHECK, "ref" => NREF),
    "metric_check" => METRIC_CHECK))

# ---------------------------------------------------------------- check: mc_sky = mc_spectrum
if "check" in LEGS
    st = ST["moving"]; plan = PLAN["moving"]
    for scat in (false, true)
        t1 = time()
        ref = mc_spectrum(plan, NCHECK; scattering = scat, bias = B0, bias_order = BIAS_ORDER,
                          bias_max = BIAS_MAX,
                          max_scatt = MAX_SCATT, seed = 7,
                          nu_min = NU_GEN[1], nu_max = NU_GEN[2], spec_nu_min = GRID.nu_min,
                          spec_nu_max = GRID.nu_max, n_ebins = GRID.n_ebins,
                          bands = InclinationBands(INCS, HALF_WIDTH), C = CVAL, escape = ESCAPE,
                          R_em = R_EM, stride = CHECK_STRIDE, r_emit_max = REMIT, backend = :cpu)
        tref = time() - t1
        sites = emission_sites(SNAP.blocks, st, plan.params; stride = CHECK_STRIDE, seed = 7,
                               r_emit_max = REMIT)
        mine = mc_sky(plan, Zones(st, sites, NCHECK), NCHECK; seed = 7, scattering = scat)
        n = length(INCS)
        rows_ok = all(samebits(ref.spectra[k], band(mine.spec, k)) for k in 1:n) &&
                  samebits(ref.total, band(mine.spec, nrows(BANDS)))
        ledger_ok = ref.n_gen == mine.N && ref.n_recorded == mine.n_escaped &&
                    ref.emitted_power === mine.emitted && ref.escaped_power === mine.escaped &&
                    ref.n_children == mine.n_children
        sectors_ok = let s = mine.spec
            all(k -> sum(s.count[n + (k - 1) * NAZ + j, :] for j in 1:NAZ) == s.count[k, :], 1:n)
        end
        writerec("check_$(scat ? "compton" : "thin")", Dict(
            "N" => NCHECK, "seed" => 7, "stride" => CHECK_STRIDE, "n_zones" => ref.n_zones,
            "rows_bitwise" => rows_ok, "ledger_bitwise" => ledger_ok,
            "sector_counts_sum_to_band" => sectors_ok, "mc_spectrum_wall_s" => tref,
            "mc_sky_wall_s" => mine.wall))
        lg(@sprintf("check %s: rows %s ledger %s sectors %s (mc_spectrum %.1f s, mc_sky %.1f s)",
                    scat ? "compton" : "thin", rows_ok, ledger_ok, sectors_ok, tref, mine.wall))
    end
end

# ---------------------------------------------------------------- zones
const ZS = Dict{String,Any}()
function zoneset(k)
    haskey(ZS, k) && return ZS[k]
    t1 = time()
    st = ST[k]
    sites = emission_sites(SNAP.blocks, st, PLAN[k].params; stride = 1, r_emit_max = REMIT)
    ts = time() - t1
    zs = Zones(st, sites, NREF)
    lg(@sprintf("zones %-7s %d (r_max %.1f M), sites %.1f s, tables %.1f s", k, length(sites),
                zs.r_max, ts, zs.seconds))
    writerec("zones_$k", Dict("n_zones" => length(sites), "r_max_M" => zs.r_max, "sites_s" => ts,
                              "tables_s" => zs.seconds, "Ns_ref" => NREF))
    ZS[k] = zs
    return zs
end

near_radius(st, x) = min(ks_radius(st, x, 1), ks_radius(st, x, 2))
const SHARE_NU = (1.0e9, 1.0e10, 1.0e11, 2.3e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15)
if "zones" in LEGS
    st = ST["moving"]
    zs = zoneset("moving")
    sites = zs.sites
    rp = [near_radius(st, s.x) for s in sites]
    ro = [sqrt(s.x[2]^2 + s.x[3]^2 + s.x[4]^2) for s in sites]
    jdv = zeros(length(sites), length(SHARE_NU))
    Threads.@threads for i in eachindex(sites)
        s = sites[i]
        for (m, f) in enumerate(SHARE_NU)
            jdv[i, m] = int_jnu(EMISSIVITY_TABLE, s.n_e, s.theta_e, s.B, f) * s.dV
        end
    end
    tot = vec(sum(jdv; dims = 1))
    share = Dict{String,Any}("nu_hz" => collect(SHARE_NU), "total_jdV" => tot)
    for (lab, sel) in (("rp<2", rp .< 2), ("2<rp<4", (rp .>= 2) .& (rp .< R_EXC)),
                       ("4<rp<10", (rp .>= R_EXC) .& (rp .< 10)), ("r<30", ro .< 30),
                       ("r<100", ro .< 100), ("r<300", ro .< 300), ("r<1000", ro .< 1000))
        share[lab] = vec(sum(jdv[sel, :]; dims = 1)) ./ tot
        share[lab * "_n"] = count(sel)
    end
    writerec("emission_share", share)
    for (k, v) in sort(collect(share); by = first)
        v isa AbstractVector && length(v) == length(SHARE_NU) && k != "nu_hz" &&
            lg(@sprintf("share %-9s %s", k, join([@sprintf("%.4f", x) for x in v], " ")))
    end
end

# ---------------------------------------------------------------- production and the metric
if "prod" in LEGS
    for scat in PROD_SCAT
        r = mc_sky(PLAN["moving"], zoneset("moving"), NPROD; seed = PROD_SEED, scattering = scat)
        save_run("moving_$(scat ? "compton" : "thin")" * (PROD_SEED == 1 ? "" : "_s$(PROD_SEED)"), r)
    end
end
if "time" in LEGS
    for scat in TIME_SCAT, s in 1:NSEED_TIME
        tag = (scat ? "compton" : "thin") * "_s$s"
        save_run("moving3_$tag", mc_sky(PLAN["moving"], zoneset("moving"), NTIME; seed = 10 + s, scattering = scat))
        save_run("rest_$tag", mc_sky(PLAN["rest"], zoneset("rest"), NTIME; seed = 10 + s, scattering = scat))
        save_run("unboost_$tag", mc_sky(PLAN["unboost"], zoneset("rest"), NTIME; seed = 10 + s, scattering = scat))
    end
end
if "mask" in LEGS
    zs = zoneset("moving")
    keep = [i for (i, s) in enumerate(zs.sites)
            if norm(s.x[SVector(2, 3, 4)] - S0[1][1]) >= R_EXC && norm(s.x[SVector(2, 3, 4)] - S0[2][1]) >= R_EXC]
    zm = subset(zs, keep)
    writerec("zones_mask", Dict("n_zones" => length(keep), "r_exc_M" => R_EXC))
    masked = ExciseMask(SNAP, S0[1][1], S0[2][1], R_EXC)
    for (scat, N) in zip((false, true), MASK_N)
        scat in MASK_SCAT || continue
        save_run("mask_$(scat ? "compton" : "thin")",
                 mc_sky(PLAN["moving"], zm, N; seed = 1, scattering = scat, provider = masked))
    end
end

# ---------------------------------------------------------------- convergence
if "conv" in LEGS
    for N in NCONV, s in 1:NSEED
        save_run("conv_N$(N)_s$(s)", mc_sky(PLAN["moving"], zoneset("moving"), N; seed = 100 + s))
    end
end
lg("done; records in ", OUT)
