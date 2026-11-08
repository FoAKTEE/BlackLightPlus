# fig_mc_dyn_binary.jl — light curves and time-resolved spectra of the circumbinary GRMHD
# snapshot of fig_binary (a), observed through arrival time, with the metric of the orbiting
# binary and the plasma carried along the orbit. CPU threads.
#
# Deck: AthenaK snapshot torus.mhd_w_bcc.00455.bin (equal masses, no spin, separation from
# its header, Ω = sep^-3/2, gamma from its header), the scaling and plasma of fig_eht_bbh.jl
# (the M87* scaling of m87_scaling.jl: two holes of 6.5e9 Msun, the density scale of run
# 00455 at which its 230 GHz image seen from 17 degrees holds 0.5 Jy; R_low = 1,
# R_high = 40, mu = 0.5, sigma cut 1). The binary phase at coordinate time 0 is the dump's
# φ0 = rem2pi(Ω t_dump); the metric is BoostedSuperposedKerrSchild on that CircularOrbit and
# the plasma Corotating(snapshot, Ω): the dump at time 0, carried by R_z(Ω t), so metric and
# plasma together have the helical symmetry of the orbit. One dump is on disk; the run's dump
# cadence is one orbital period.
#
# Monte Carlo: emission zones of stride BLP_MCB_STRIDE within r < BLP_MCB_REMIT, window
# [0, BLP_MCB_PERIODS P], BLP_MCB_N primaries, Compton scattering (the gpumonty hot-cross table
# when found, else built), the state bias b = clamp(b0 A(Θe) (w/w_ref) f^n, 1, 1e6) with
# b0 = BLP_MCB_BIAS and f = BLP_MCB_BIAS_ORDER (BLP_MCB_BIASMODE = constant: b = b0 (n + 1)²),
# at most 3 scatterings,
# C = 80, escape at 1000 M; spectra on 200 bins over 1e8-1e24 Hz; time-resolved bins: arrival
# bins of P/BLP_MCB_SECTORS over the window, 32 frequency bins, the inclination bands 17, 50
# and 90 ± 10 degrees with BLP_MCB_SECTORS azimuth sectors each.
#
# Writes BLP_MCB_OUT (figures/paper/out/mc_dyn/binary): spec_<order>.npy (window-mean νLν of
# each band, rows 17, 50, 90, sky), timed_<what>.npy (the time-resolved sums: count, sumwE,
# sumwE2 per order and the total, [arrival bin, row, frequency bin]), axes.npy and
# records.json (setup, run diagnostics, the helical-symmetry statistics).
# Env: BLP_BBH_BIN (the 00455 dump), BLP_MCB_N (5e6), BLP_MCB_STRIDE (2), BLP_MCB_REMIT (100),
#      BLP_MCB_PERIODS (2), BLP_MCB_SECTORS (16), BLP_MCB_BIASMODE (state), BLP_MCB_BIAS (1e4),
#      BLP_MCB_BIAS_ORDER (100), BLP_MCB_MS (mass scale, g; default M87_DENSITY_SCALE of run 00455),
#      BLP_REF_GPUMONTY (hotcross.dat), BLP_MCB_OUT.
# Run, from the repository root:
#   julia -t 48 --project=. figures/paper/fig_mc_dyn_binary.jl
using Dates, LinearAlgebra, Printf, Statistics, StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: ME, CL
using BlackLightPlus.MonteCarlo: Bias
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, helical_symmetry
using BlackLightPlus.Data: athenak_bin_snapshot, Corotating, load_hotcross, build_hotcross_table, write_npy
using BlackLightPlus.Execution: InclinationBands, Spectrum, sector_row, nrows
using BlackLightPlus.Products: light_curve, arrival_times, time_spectrum
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

const ROOT = dirname(dirname(@__DIR__))
const SNAP = get(ENV, "BLP_BBH_BIN",
                 "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
envnum(k, d) = parse(Float64, get(ENV, k, string(d)))
const N = round(Int, envnum("BLP_MCB_N", 5e6))
const STRIDE = round(Int, envnum("BLP_MCB_STRIDE", 2))
const REMIT = envnum("BLP_MCB_REMIT", 100)
const PERIODS = envnum("BLP_MCB_PERIODS", 2)
const NAZ = round(Int, envnum("BLP_MCB_SECTORS", 16))
const B0 = envnum("BLP_MCB_BIAS", 1e4)
const BIASMODE = get(ENV, "BLP_MCB_BIASMODE", "state")        # state: StateBias(b0, order); constant: Bias(b0)
const BORDER = envnum("BLP_MCB_BIAS_ORDER", 100)
const MS = envnum("BLP_MCB_MS", m87_density_scale("00455"))
const HOTCROSS = get(ENV, "BLP_REF_GPUMONTY",
                     joinpath(dirname(ROOT), "Mahakala", "ref-code", "gpumonty", "table", "hotcross.dat"))
const OUT = get(ENV, "BLP_MCB_OUT", joinpath(@__DIR__, "out", "mc_dyn", "binary"))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
const M_BH = 2 * M87_MASS_G
const NU = M87_FREQUENCY_HZ
const INCL = (17.0, 50.0, 90.0)

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))
json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::Tuple) = json(collect(x))
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"

# ---------------------------------------------------------------- dump header (as fig_eht_bbh.jl)
function read_header(path)
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
    return tt, par
end
_num(par, b, k, dflt) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : dflt

function main()
    rec = Dict{String,Any}("code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()),
                           "snapshot" => SNAP, "N" => N, "stride" => STRIDE, "r_emit_max" => REMIT,
                           "periods" => PERIODS, "sectors" => NAZ, "bias_b0" => B0, "bias_mode" => BIASMODE,
                           "bias_order" => BORDER, "mass_scale_g" => MS,
                           "m_bh_g" => M_BH, "incl_deg" => collect(INCL), "half_width_deg" => 10.0)
    t_hdr, par = read_header(SNAP)
    sep = _num(par, "problem", "sep", NaN); q = _num(par, "problem", "q", 1.0)
    a1 = _num(par, "problem", "a1", 0.0); a2 = _num(par, "problem", "a2", 0.0)
    gam = _num(par, "mhd", "gamma", 13 / 9)
    tlim = _num(par, "time", "tlim", NaN)
    q == 1 && a1 == 0 && a2 == 0 || error("this generator assumes q = 1 and no spin")
    t_s = (isfinite(tlim) && abs(tlim - t_hdr) <= 0.5) ? tlim : t_hdr
    Ω = sep^-1.5; P = 2π / Ω
    phi0 = rem2pi(Ω * t_s, RoundNearest)
    orbit = CircularOrbit((0.5, 0.5), sep; phase = phi0)
    st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
    helical_symmetry(st, Ω) || error("the orbit is not a helical symmetry")
    merge!(rec, Dict{String,Any}("t_dump" => t_s, "sep" => sep, "omega" => Ω, "period" => P, "phi0" => phi0,
                                 "gamma" => gam, "hole_speed" => Ω * sep / 2,
                                 "holes_t0" => [collect(orbit(0.0)[A][1]) for A in 1:2]))
    logf("dump t = %.2f, sep %.3g, Ω %.6g, P %.2f M, φ0 %.5f; %d threads\n", t_s, sep, Ω, P, phi0,
         Threads.nthreads())
    t0 = time()
    snap = athenak_bin_snapshot(SNAP; gamma = gam)
    rec["load_s"] = time() - t0
    logf("loaded in %.1f s\n", rec["load_s"])
    table = isfile(HOTCROSS) ? load_hotcross(HOTCROSS) : build_hotcross_table()
    rec["hotcross"] = isfile(HOTCROSS) ? HOTCROSS : "built"
    cfg = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
                    plasma = PlasmaConfig(gamma = gam, rat_low = 1.0, rat_high = 40.0, mu = 0.5, sigma_cut = 1.0),
                    C = 80.0)
    plan = prepare(cfg; spacetime = st, matter = Corotating(snap, Ω))
    rec["L_unit_cm"] = plan.params.units.L_unit; rec["T_unit_s"] = plan.params.units.L_unit / CL
    win = (0.0, PERIODS * P)
    nt = round(Int, PERIODS * NAZ)
    t0 = time()
    bias = BIASMODE == "state" ? B0 : Bias(B0)
    r = mc_spectrum(plan, N; scattering = true, table = table, bias = bias, bias_order = BORDER, max_scatt = 3, seed = 455,
                    stride = STRIDE, r_emit_max = REMIT, spec_nu_max = 1.0e24, n_ebins = 200,
                    bands = InclinationBands(INCL, 10.0), window = win, time_bins = (win[1], win[2], nt),
                    time_ebins = 32, sectors = NAZ, backend = :cpu)
    rec["wall_s"] = time() - t0
    g = r.result
    merge!(rec, Dict{String,Any}("n_zones" => g.n_zones, "n_gen" => g.n_gen, "n_recorded" => g.n_recorded,
                                 "n_children" => g.n_children, "emitted_erg_s" => g.emitted_power,
                                 "escaped_erg_s" => g.escaped_power, "absorbed_erg_s" => g.absorbed_power,
                                 "captured_erg_s" => g.captured_power, "dropped_erg_s" => g.dropped_power,
                                 "w_children" => g.w_children, "scatter_loss" => g.scatter_loss,
                                 "w_unspawned" => g.w_unspawned, "interior" => collect(r.interior),
                                 "window" => collect(win), "dOmega" => g.dOmega))
    logf("run: %d zones, %d primaries, %d children, %d escaped, escaped %.4e of %.4e erg/s in %.1f s\n",
         g.n_zones, g.n_gen, g.n_children, g.n_recorded, g.escaped_power, g.emitted_power, rec["wall_s"])

    # window-mean spectra of every order: rows 17, 50, 90 and the sky
    nu = BlackLightPlus.Execution.nu_centers(g.total)
    vec2(a) = reshape(collect(Float64, a), :, 1)
    write_npy(joinpath(OUT, "nu.npy"), vec2(nu))
    for (k, o) in enumerate(g.orders)
        write_npy(joinpath(OUT, "spec_order$(k - 1).npy"), o.sumwE)
        write_npy(joinpath(OUT, "spec2_order$(k - 1).npy"), o.sumwE2)
    end
    # time-resolved sums
    tc, edges = arrival_times(r)
    nr = nrows(r.rule.bands)
    # rows of a time-resolved Spectrum: arrival bin k (0 underflow ... n_t + 1 overflow) times
    # the nr rows of the sector rule, frequency fastest in the file: [k * nr + row, ν bin]
    for (k, o) in enumerate(r.orders)
        for (name, a) in (("count", Float64.(o.count)), ("sumwE", o.sumwE), ("sumwE2", o.sumwE2))
            write_npy(joinpath(OUT, "timed_order$(k - 1)_$(name).npy"), collect(Float64, a))
        end
    end
    write_npy(joinpath(OUT, "timed_nu.npy"), vec2(BlackLightPlus.Execution.nu_centers(r.total)))
    write_npy(joinpath(OUT, "timed_edges.npy"), vec2(edges))
    write_npy(joinpath(OUT, "timed_dOmega.npy"), vec2(BlackLightPlus.Execution.solid_angles(r.rule.bands)))
    rec["timed_nt"] = r.rule.n_t
    rec["timed_rows"] = nr; rec["timed_scale"] = ME * CL^2 * (win[2] - win[1]) / r.rule.dt

    # helical symmetry: sector j + 1 at bin k + 1 against sector j at bin k, every band; the
    # band light curves against their mean; frequency-integrated, all orders
    ib = 1:nt
    kint = findall(k -> edges[k] >= r.interior[1] && edges[k + 1] <= r.interior[2], ib)
    for (b, inc) in enumerate(INCL)
        S = zeros(nt, NAZ); V = zeros(nt, NAZ)
        for j in 1:NAZ
            _, L, s, _ = light_curve(r; row = sector_row(r.rule.bands, b, j))
            S[:, j] = L; V[:, j] = s
        end
        χ = [(S[i, j] - S[i + 1, mod1(j + 1, NAZ)])^2 / (V[i, j]^2 + V[i + 1, mod1(j + 1, NAZ)]^2)
             for i in kint[1:end-1] for j in 1:NAZ if V[i, j] > 0 && V[i + 1, mod1(j + 1, NAZ)] > 0]
        χ0 = [(S[i, j] - S[i + 1, j])^2 / (V[i, j]^2 + V[i + 1, j]^2)
              for i in kint[1:end-1] for j in 1:NAZ if V[i, j] > 0 && V[i + 1, j] > 0]
        _, Lb, sb, _ = light_curve(r; row = b)
        w = 1 ./ sb[kint] .^ 2
        m = sum(w .* Lb[kint]) / sum(w)
        rec["helical_chi2_$(Int(inc))"] = mean(χ); rec["helical_n_$(Int(inc))"] = length(χ)
        rec["noshift_chi2_$(Int(inc))"] = mean(χ0)
        rec["band_flat_chi2_$(Int(inc))"] = sum(((Lb[kint] .- m) ./ sb[kint]) .^ 2) / (length(kint) - 1)
        logf("  %2d deg: helical χ²/n %.3f (n %d), unshifted %.3f, band flatness χ²/n %.3f\n", inc, mean(χ),
             length(χ), mean(χ0), rec["band_flat_chi2_$(Int(inc))"])
    end
    open(io -> println(io, json(rec)), joinpath(OUT, "records.json"), "w")
end

main()
