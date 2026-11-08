# fig_mc_kerr.jl — Monte Carlo spectra of the Kerr accretion flow of Fig. polarized (a = 0.9)
# (fig_mc_kerr_common.jl), without and with Compton scattering, the ray-traced spectrum of
# the same plasma, and the convergence of the Monte Carlo with the superphoton number.
#
# The plasma of every run here is `CellCentred(snapshot)` (BLP_MC_PLASMA=cell, default):
# each point takes the primitives of its cell, decoded with the metric at the point. The Monte
# Carlo emits every primary at a point drawn uniformly within its cell, weighted by the
# emissivity there over that of the cell centre (BLP_MC_EMIT=incell, default; `centre`: at
# the cell centre, as `mc_spectrum`), and the cells next to the horizon whose centre is cut but
# whose other points are not emit as well (`partial_sites`), so it and the ray tracer see one
# and the same plasma, in emission, absorption and scattering. Emission from the centre falls below the ray-traced
# spectrum above 1e13 Hz (by 8 % at 1.1e14 Hz in the 50 deg band), where the light comes from
# the cells next to the horizon and its escape and redshift change across a cell.
# BLP_MC_PLASMA=trilinear runs the Monte Carlo on the interpolated snapshot instead.
#
# Monte Carlo (`mc_incell` of fig_mc_kerr_common.jl, the body of `mc_spectrum` with the in-cell
# source; CPU threads, stride 1: every owned cell above the horizon):
# generation 1e8-1e16 Hz, spectra on 160 log bins over 1e8-1e24 Hz, inclination bands
# 17, 50, 90 deg +- 10 deg about the spin axis; with scattering the state bias b0 = B0 with
# the order factor BIAS_ORDER, and at most three scatterings.
# Ray tracing: at the centres of the MC bins 11 + 5 j (10^(9.05 + 0.5 j) Hz, j = 0 ... 11),
# the band mean of `rt_band` (Gauss-Legendre in cos i with RT_NTH[b] nodes in band b, RT_NPHI
# uniform azimuths) for the cell-centred plasma, and at the j in RT_TRI for the trilinear plasma
# (NaN elsewhere). The edge-on band takes four nodes: its light is beamed toward the equatorial
# plane, and two nodes overestimate its mean by 2 % at 1.1e13 Hz (four nodes: 17 and 50 deg
# within 0.1 % of two). A run over a subset RT_BANDS of the bands leaves the others NaN.
# Convergence: thin runs of N in CONV_THIN with CONV_SEEDS seeds, kept at the ray-tracing
# frequencies (L and sigma per band); Compton runs of N in CONV_COMP with CONV_SEEDS seeds,
# kept whole (the sum over orders, every band and the sky).
#
# Legs (BLP_MC_LEGS, default "rt,thin,compton,conv_thin,conv_comp"). Writes to BLP_OUT
# (default figures/paper/out/mc_kerr):
#   torus_rt.npy          [nu, MC bin, cell-centred band 1-3, trilinear band 1-3, cell-centred
#                         band 1-3 at the base resolution of NEST]; from j = RT_PX2 on, the
#                         cell-centred means use fields of twice the pixels per side (the light
#                         there comes from the cells next to the horizon); a run over a subset
#                         RT_JS of the j writes torus_rt_j<j_j_...>.npy, its rows only, and
#                         one over a subset RT_BANDS of the bands torus_rt_b<bands>[_j...].npy
#   torus_thin_<plasma>.npy     stats_table of the thin run (fig_mc_kerr_common.jl)
#   torus_compton_<plasma>.npy  stats_table of the Compton run (orders 0-3, then their sum)
#   conv_thin_<plasma>.npy      [N, seed, band, j, L, sigma] rows
#   conv_comp_<plasma>_N<N>_s<s>.npy   stats_table of each Compton convergence run
#   torus_record_<plasma>_<legs>.json   parameters, ledgers, wall times, code SHA
# Env: BLP_MC_MASS_SCALE (the mass scale, g; default the 00999 scale of m87_scaling.jl),
# BLP_MC_EMIT (incell), BLP_MC_NTHIN (6e7),
# BLP_MC_NCOMP (3e6), BLP_MC_B0 (1e4),
# BLP_MC_BIAS_ORDER (300), BLP_MC_PLASMA,
# BLP_MC_CONV_THIN, BLP_MC_CONV_COMP, BLP_MC_CONV_SEEDS (4), BLP_MC_RT_NTH (2,2,4), BLP_MC_RT_BANDS (1,2,3),
# BLP_MC_RT_NPHI (8), BLP_MC_RT_TRI (0,4,6,8,10), BLP_MC_RT_PX2 (8), BLP_MC_RT_JS (0,...,11),
# BLP_MC_LEGS, BLP_OUT.
# Run from the repository root: julia -t 48 --project=. figures/paper/fig_mc_kerr.jl
# (plot: paper/scripts/plot_mc_kerr.py)
include(joinpath(@__DIR__, "fig_mc_kerr_common.jl"))

_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
const LEGS = _list(get(ENV, "BLP_MC_LEGS", "rt,thin,compton,conv_thin,conv_comp"))
const N_THIN = parse(Float64, get(ENV, "BLP_MC_NTHIN", "6e7"))
const N_COMP = parse(Float64, get(ENV, "BLP_MC_NCOMP", "3e6"))
const B0 = parse(Float64, get(ENV, "BLP_MC_B0", "1e4"))
const BIAS_ORDER = parse(Float64, get(ENV, "BLP_MC_BIAS_ORDER", "300"))
const PLASMA = get(ENV, "BLP_MC_PLASMA", "cell")
const INCELL = get(ENV, "BLP_MC_EMIT", "incell") == "incell"   # emission uniform within each cell
const TAG = PLASMA * (INCELL ? "_incell" : "")
const CONV_THIN = parse.(Float64, _list(get(ENV, "BLP_MC_CONV_THIN", "1e5,3e5,1e6,3e6,1e7")))
const CONV_COMP = parse.(Float64, _list(get(ENV, "BLP_MC_CONV_COMP", "1e4,3e4,1e5,3e5")))
const CONV_SEEDS = parse(Int, get(ENV, "BLP_MC_CONV_SEEDS", "4"))
const RT_NTH = let v = parse.(Int, _list(get(ENV, "BLP_MC_RT_NTH", "2,2,4")))   # nodes per band
    length(v) == 1 ? fill(v[1], length(BANDS)) : v
end
const RT_BANDS = parse.(Int, _list(get(ENV, "BLP_MC_RT_BANDS", "1,2,3")))         # bands traced
const RT_NPHI = parse(Int, get(ENV, "BLP_MC_RT_NPHI", "8"))
const SEED = 0x5eed0003
const RT_BINS = [11 + 5j for j in 0:11]
const RT_TRI = parse.(Int, _list(get(ENV, "BLP_MC_RT_TRI", "0,4,6,8,10")))   # j with a trilinear band mean
const RT_PX2 = parse(Int, get(ENV, "BLP_MC_RT_PX2", "8"))    # j from which the fields have twice the pixels
const RT_JS = parse.(Int, _list(get(ENV, "BLP_MC_RT_JS", join(0:length(RT_BINS) - 1, ","))))   # j traced by this run

# One Monte Carlo run of the deck: `mc_spectrum` (emission from the cell centres), or every
# primary emitted uniformly within its cell (`mc_incell`, fig_mc_kerr_common.jl).
function mc_run_kerr(plan, snap, N; seed, scattering = false)
    INCELL && return mc_incell(plan, snap, N; seed = seed, scattering = scattering, bias = B0,
                               bias_order = BIAS_ORDER, max_scatt = 3)
    return scattering ?
           mc_spectrum(plan, N; seed = seed, SPEC..., scattering = true, bias = B0,
                       bias_order = BIAS_ORDER, max_scatt = 3, backend = :cpu) :
           mc_spectrum(plan, N; seed = seed, SPEC..., backend = :cpu)
end

function main()
    t0 = time()
    mkpath(OUT)
    snap = AthenaKSnapshot(DATA)
    matter = PLASMA == "cell" ? CellCentred(snap) : snap
    plan = prepare(kerr_config(); spacetime = KerrSchildCartesian(SPIN), matter = matter)
    nu = [exp(log(SPEC.spec_nu_min) + (k - 0.5) * log(SPEC.spec_nu_max / SPEC.spec_nu_min) / SPEC.n_ebins)
          for k in 1:SPEC.n_ebins]
    jtag = (RT_BANDS == collect(1:length(BANDS)) ? "" : "_b$(join(RT_BANDS))") *
           (RT_JS == collect(0:length(RT_BINS) - 1) ? "" : "_j$(join(RT_JS, '_'))")
    recpath = joinpath(OUT, "torus_record_$(TAG)_$(join(LEGS, '+'))$(jtag).json")
    rec = Dict{String,Any}(
        "snapshot" => DATA, "spin" => SPIN, "m_bh_msun" => 6.5e9, "mass_scale_g" => MASS_SCALE,
        "electrons" => Dict("R_low" => 1.0, "R_high" => 10.0, "gamma" => 13 / 9, "use_p" => true,
                            "sigma_cut" => 1.0), "C" => 80.0, "plasma" => PLASMA, "stride" => 1,
        "bands_deg" => BANDS, "half_width_deg" => HALF, "spec" => SPEC, "bias_b0" => B0,
        "bias_order" => BIAS_ORDER, "max_scatt" => 3, "emission_in_cell" => INCELL, "git_sha" => GIT_SHA, "threads" => Threads.nthreads(), "legs" => LEGS)
    if "rt" in LEGS
        t = time()
        rows = fill(NaN, length(RT_JS), 11)
        cc = CellCentred(snap)
        nest2 = Tuple((f, 2n) for (f, n) in NEST)
        btag = RT_BANDS == collect(1:length(BANDS)) ? "" : "_b$(join(RT_BANDS))"
        rtname = "torus_rt" * btag *
                 (RT_JS == collect(0:length(RT_BINS) - 1) ? "" : "_j$(join(RT_JS, '_'))") * ".npy"
        for (r, j) in enumerate(RT_JS)
            k = RT_BINS[j + 1]
            rows[r, 1] = nu[k]; rows[r, 2] = k
            for (b, inc) in enumerate(BANDS)
                b in RT_BANDS || continue
                nth = RT_NTH[b]
                base = rt_band(cc, nu[k], inc; nth = nth, nphi = RT_NPHI)[1]
                rows[r, 8 + b] = base
                rows[r, 2 + b] = j >= RT_PX2 ?
                                 rt_band(cc, nu[k], inc; nth = nth, nphi = RT_NPHI, nest = nest2)[1] : base
                rows[r, 5 + b] = j in RT_TRI ?
                                 rt_band(snap, nu[k], inc; nth = nth, nphi = RT_NPHI)[1] : NaN
            end
            lg(@sprintf("rt nu %.3e: cell %s | trilinear %s (%.0f s)", nu[k],
                        join([@sprintf("%.4e", x) for x in rows[r, 3:5]], " "),
                        join([@sprintf("%.4e", x) for x in rows[r, 6:8]], " "), time() - t))
            write_npy(joinpath(OUT, rtname), rows[1:r, :])
        end
        rec["rt"] = Dict("bins" => RT_BINS[RT_JS .+ 1], "bands" => RT_BANDS, "n_theta" => RT_NTH,
                         "n_phi" => RT_NPHI, "px2_from_j" => RT_PX2,
                         "nest" => NEST, "seconds" => time() - t)
    end
    if "thin" in LEGS
        r = mc_run_kerr(plan, snap, N_THIN; seed = SEED)
        write_npy(joinpath(OUT, "torus_thin_$(TAG).npy"), stats_table(r))
        rec["thin"] = merge(ledger(r), Dict("Ns" => N_THIN))
        lg(@sprintf("thin: %d zones, %d primaries, %d recorded, emitted %.4e, escaped %.4e erg/s (%.1f s)",
                    r.n_zones, r.n_gen, r.n_recorded, r.emitted_power, r.escaped_power, r.run.seconds))
    end
    if "compton" in LEGS
        r = mc_run_kerr(plan, snap, N_COMP; seed = SEED, scattering = true)
        write_npy(joinpath(OUT, "torus_compton_$(TAG).npy"), stats_table(r))
        rec["compton"] = merge(ledger(r), Dict("Ns" => N_COMP))
        lg(@sprintf("compton: %d primaries, %d children, escaped %.4e erg/s, ledger %.5f (%.1f s)",
                    r.n_gen, r.n_children, r.escaped_power,
                    (r.w_children + r.w_unspawned) / r.scatter_loss, r.run.seconds))
    end
    if "conv_thin" in LEGS
        rows = Float64[]
        for N in CONV_THIN, s in 1:CONV_SEEDS
            r = mc_run_kerr(plan, snap, N; seed = 1000 + s)
            for b in eachindex(BANDS)
                L, sg, _, _ = row_stats(r.spectra[b], 1)
                for (j, k) in enumerate(RT_BINS)
                    append!(rows, (N, s, b, j, L[k], sg[k]))
                end
            end
            write_npy(joinpath(OUT, "conv_thin_$(TAG).npy"), permutedims(reshape(rows, 6, :)))
            lg(@sprintf("conv thin N %.0e seed %d: %.1f s", N, s, r.run.seconds))
        end
    end
    if "conv_comp" in LEGS
        for N in CONV_COMP, s in 1:CONV_SEEDS
            r = mc_run_kerr(plan, snap, N; seed = 2000 + s, scattering = true)
            write_npy(joinpath(OUT, @sprintf("conv_comp_%s_N%.0e_s%d.npy", TAG, N, s)), stats_table(r))
            lg(@sprintf("conv compton N %.0e seed %d: %d children, %.1f s", N, s, r.n_children, r.run.seconds))
        end
    end
    rec["wall_seconds"] = time() - t0
    write_json(recpath, rec)
    lg(@sprintf("wrote %s (%.1f s on %d threads)", OUT, time() - t0, Threads.nthreads()))
end

main()
