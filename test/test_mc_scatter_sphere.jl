# Monte Carlo scattering-sphere tests: successive Compton up-scattering. The
# idealized Comptonization sphere (n_e=1e18, Theta_e=3, B=100 G, R=3e5 cm ->
# tau_es ~ 0.20) emits thermal-synchrotron seeds and Compton-scatters them.
# Self-absorption is neglected (validated separately by the thin-sphere test; see
# the ScatteringSphere docstring). Comptonization has no closed-form spectrum, so
# the checks are physical: (a) the nscatt=0 spectrum equals the analytic direct
# emission up to a flat grey survival exp(-tau) (<10% shape deviation); (b)
# photons are up-scattered; (c) scattered weight fraction ~ 1-exp(-<tau>); (d)
# spawned child weight equals the incident scatter loss; (e) distinct bumps.
using Test
import BlackLightPlus
using BlackLightPlus: ScatteringSphere, mc_spectrum_scatter, load_hotcross,
                    nuLnu_total, analytic_nuLnu, mean_energy, total_weight,
                    scattering_tau_es, nu_centers

# ------------------------------------------------------------ run the simulation
# gpumonty reference table (override with BLP_REF_GPUMONTY).
const HOTCROSS_DAT = get(ENV, "BLP_REF_GPUMONTY",
    "/data/haiyangw/claude/Mahakala/ref-code/gpumonty/table/hotcross.dat")
const SC_TBL   = load_hotcross(HOTCROSS_DAT)
const SC_SPH   = ScatteringSphere()               # tau_es ~ 0.20
const SC_MAXSC = 3                                 # cap layers (documented, unit-test)
const SC_NS    = 200_000
const SEED_BAND_LO = 1.0e9                         # seed-emission band for check (a)
const SEED_BAND_HI = 1.0e11
const SC_MINCOUNT  = 50

const SC_RES = mc_spectrum_scatter(SC_SPH, SC_TBL, SC_NS; seed = 0xBEEF,
                 gen_nu_min = 1.0e8, gen_nu_max = 1.0e13,
                 spec_nu_min = 1.0e8, spec_nu_max = 1.0e17,
                 n_ebins = 140, max_scatt = SC_MAXSC)

# ------------------------------------------------------------ helpers

"Median of `v` (no Statistics dependency)."
function _median(v::AbstractVector{<:Real})
    s = sort(v)
    n = length(s)
    n == 0 && return 0.0
    return isodd(n) ? Float64(s[(n + 1) ÷ 2]) : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1])
end

"""
    survival_shape(res, model) -> (S, worst, nbins)

Grey scattering survival `S = median(nscatt0/analytic)` over well-populated seed-band
bins, and the worst fractional deviation of `nscatt0/analytic` from that flat `S`
(the SHAPE-corruption metric). The nscatt=0 direct emission must equal the analytic
thin emissivity up to the (frequency-independent) survival factor.
"""
function survival_shape(res, model; nu_lo = SEED_BAND_LO, nu_hi = SEED_BAND_HI,
                        min_count = SC_MINCOUNT)
    nu, L0, cnt = nuLnu_total(res.orders[1])
    ratios = Float64[]
    for i in eachindex(nu)
        (nu[i] < nu_lo || nu[i] > nu_hi) && continue
        cnt[i] < min_count && continue
        a = analytic_nuLnu(model, nu[i])
        a > 0.0 || continue
        push!(ratios, L0[i] / a)
    end
    S = _median(ratios)
    worst = isempty(ratios) ? Inf : maximum(abs.(ratios ./ S .- 1.0))
    return S, worst, length(ratios)
end

"Energy-weighted mean energy of the scattered (nscatt>=1) population [m_e c^2 units]."
function scattered_mean_energy(res)
    sw = 0.0
    swe = 0.0
    for n in 1:res.max_scatt
        s = res.orders[n + 1]
        sw += total_weight(s)
        swe += sum(s.sumwE)
    end
    return sw > 0.0 ? swe / sw : 0.0
end

"Per-order (recorded weight, mean energy, peak frequency) for orders 0..max_scatt."
function order_stats(res)
    W = Float64[]
    E = Float64[]
    NUPK = Float64[]
    for n in 0:res.max_scatt
        s = res.orders[n + 1]
        push!(W, total_weight(s))
        push!(E, mean_energy(s))
        nu, L, _ = nuLnu_total(s)
        # peak of nu L_nu smoothed over +-3 bins (+-0.2 dex): the raw argmax of the sparse
        # order-2/3 bumps near the grid top moves between neighbouring bins with the
        # Monte Carlo noise (the raw-argmax ordering failed 2 of 32 seeds of the
        # slot-keyed implementation; the smoothed peak failed 0 of 64 runs)
        Ls = [sum(view(L, max(1, i - 3):min(length(L), i + 3))) for i in eachindex(L)]
        push!(NUPK, nu[argmax(Ls)])
    end
    return W, E, NUPK
end

# ------------------------------------------------------------ (a) direct emission uncorrupted

@testset "scattering sphere: nscatt=0 direct emission uncorrupted" begin
    S, worst, nbins = survival_shape(SC_RES, SC_SPH)
    @info "scatter sphere gate (a) direct emission" survival = round(S, digits = 4) shape_worst = round(worst, digits = 4) nbins
    @test nbins >= 20                                 # seed band densely populated
    # SHAPE preserved: nscatt=0 / analytic is a FLAT survival factor to <10%.
    @test worst < 0.10
    # survival is the grey scattering attenuation: exp(-tau_es)=0.82 <= S <= ~0.92.
    @test 0.82 <= S <= 0.93
end

# ------------------------------------------------------------ (b) Compton up-scattering

@testset "scattering sphere: photons are Compton up-scattered" begin
    E0 = mean_energy(SC_RES.orders[1])
    Es = scattered_mean_energy(SC_RES)
    E1 = mean_energy(SC_RES.orders[2])
    boost1 = E1 / E0
    @info "scatter sphere gate (b) up-scatter" E0 first_order_boost = round(boost1, digits = 2) scatt_over_direct = round(Es / E0, digits = 1)
    @test E0 > 0.0
    # scattered population is far more energetic than the direct seed
    @test Es / E0 > 10.0
    # single-scatter boost ~ 1 + 4 Theta_e + 16 Theta_e^2 = 157 for Theta_e = 3
    # (energy-weighted, KN-suppressed at the top): wide physically-justified band.
    @test 30.0 < boost1 < 400.0
end

# ------------------------------------------------------------ (c) scattered fraction ~ tau

@testset "scattering sphere: scattered fraction vs optical depth" begin
    frac = SC_RES.w_first_order / SC_RES.w_primary
    tau_es = scattering_tau_es(SC_SPH)
    @info "scatter sphere gate (c) scattered fraction" tau_es = round(tau_es, digits = 4) frac = round(frac, digits = 4)
    # geometry-averaged escape depth for uniform isotropic emission is <tau> ~ 0.75
    # tau_es, so <1-exp(-tau)> ~ 0.135 for tau_es ~ 0.20 (independent MC quadrature).
    @test 0.11 <= frac <= 0.16
end

# ------------------------------------------------------------ (d) unbiasedness

@testset "scattering sphere: unbiased estimator (weight conservation)" begin
    ratio = SC_RES.w_children / SC_RES.scatter_loss
    @info "scatter sphere gate (d) unbiasedness" w_children = SC_RES.w_children scatter_loss = SC_RES.scatter_loss ratio = round(ratio, digits = 4)
    # total spawned child weight == incident weight lost to scattering, to MC error
    # (the bias b cancels: E[spawned]=b*dtau*(w/b)=w*dtau=physical loss).
    @test 0.95 <= ratio <= 1.05
end

# ------------------------------------------------------------ (e) distinct Compton bumps

@testset "scattering sphere: distinct successive Compton bumps" begin
    W, E, NUPK = order_stats(SC_RES)
    @info "scatter sphere gate (e) bumps" recorded_weight = round.(W, sigdigits = 3) mean_energy = round.(E, sigdigits = 3) peak_nu = round.(NUPK, sigdigits = 3)
    # every scattering order is populated (bumps 0..max_scatt all present)
    @test all(W .> 0.0)
    # distinct populations: recorded weight falls off with order (thinning by ~tau)
    @test all(W[n] > W[n + 1] for n in 1:length(W)-1)
    # mean energy rises monotonically -> successive up-scattering
    @test all(E[n + 1] > E[n] for n in 1:length(E)-1)
    # each scatter boosts the mean energy by a clear factor (> 4 Theta_e / 3 ~ 4)
    @test all(E[n + 1] / E[n] > 4.0 for n in 1:length(E)-1)
    # bumps peak at successively higher frequency (distinct, separated in nu)
    @test all(NUPK[n + 1] > NUPK[n] for n in 1:length(NUPK)-1)
    # finite total output power
    @test 0.0 < BlackLightPlus.MonteCarlo.total_power(SC_RES.total) < Inf
end

# ------------------------------------------------------------ (f) estimator does not saturate
# The biased exponential-transform child weight must equal the incident scatter loss in
# EXPECTATION for any bias b -- including b*dtau_s ~ 1, where the bare `w/b` scheme silently
# saturates (spawn prob 1-exp(-b*dtau_s) tops out at 1 while w/b is fixed, under-spawning by
# [(1-e^{-b dtau})/b]/(1-e^{-dtau})). Track first-order spawns off primaries at increasing
# tau_es and check the child/loss ledger ratio stays ~1 (b = max(1, tau_es) for primaries).
using StaticArrays
using BlackLightPlus: SphereModel, sphere_zone, build_weight_table, zone_nsuperphotons,
                    sample_sphere_photon, track_superphoton_scatter!, ScatterWorkspace, MCRng
using BlackLightPlus.Constants: SIGMA_THOMSON

function first_order_ledger(tau_es; N = 20_000, seed = 0xBEEF)
    R = 3.0e5
    n_e = tau_es / (SIGMA_THOMSON * R)
    model = SphereModel(n_e = n_e, theta_e = 3.0, B = 100.0, R = R)
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], 1.0e5 / log(1.0e13 / 1.0e8); nu_min = 1.0e8, nu_max = 1.0e13)
    _, dnmax = zone_nsuperphotons(zone, wgt)
    ws = ScatterWorkspace()
    loss = 0.0
    for i in 1:N
        rng = MCRng(Int(seed), i)
        sp, rng = sample_sphere_photon(model, wgt, dnmax, rng)
        # track ONLY the primary (do not drain children): first_order_weight(ws) collects
        # the weight it spawns, `l` its physical scatter loss -- their ratio is the estimator.
        _, _, l, _ = track_superphoton_scatter!(ws, sp, model, SC_TBL, rng; max_scatt = SC_MAXSC)
        loss += l
    end
    return BlackLightPlus.MonteCarlo.first_order_weight(ws) / loss
end

@testset "scattering sphere: exp-transform estimator does not saturate at large b*dtau" begin
    # b = max(1, tau_es); at tau_es = 1 (b = 1) the scheme was always exact, but at tau_es =
    # 3, 10 (b = 3, 10) the OLD `w/b` weight under-spawned to 0.87 / 0.26. The reweighted
    # w(1-e^{-dtau})/(1-e^{-b dtau}) restores conservation for every bias.
    for tau in (1.0, 3.0, 10.0)
        r = first_order_ledger(tau)
        @info "scatter saturation gate (f)" tau_es = tau ledger_ratio = round(r, digits = 4)
        @test isapprox(r, 1.0; atol = 0.04)
    end
end

# ------------------------------------------------------------ (g) tabulated generation opt-in
# The tabulated frequency rejection (emission_tables.jl `IntJnuTable`) must leave
# the scattering estimator untouched: the unbiasedness ledger (w_children ==
# scatter_loss), the created seed power and the direct-emission survival shape all
# match the exact run (the chord-clamped table flipped no streams at this deck;
# the w_primary tolerance budgets for the rare interpolation-error accept flip).
const SC_RES_TAB = mc_spectrum_scatter(SC_SPH, SC_TBL, SC_NS; seed = 0xBEEF,
                     gen_nu_min = 1.0e8, gen_nu_max = 1.0e13,
                     spec_nu_min = 1.0e8, spec_nu_max = 1.0e17,
                     n_ebins = 140, max_scatt = SC_MAXSC, tabulated_gen = true)

@testset "scattering sphere: tabulated_gen preserves the estimator" begin
    ratio = SC_RES_TAB.w_children / SC_RES_TAB.scatter_loss
    wprel = abs(SC_RES_TAB.w_primary - SC_RES.w_primary) / SC_RES.w_primary
    @info "scatter sphere gate (g) tabulated" unbias = round(ratio, digits = 4) w_primary_rel = wprel
    @test 0.95 <= ratio <= 1.05          # check (d) unbiasedness, tabulated run
    @test wprel < 1.0e-3                 # seed power parity with the exact run
    S, worstf, nbins = survival_shape(SC_RES_TAB, SC_SPH)
    @test nbins >= 20
    @test worstf < 0.10                  # check (a) shape, tabulated run
    @test 0.82 <= S <= 0.93
end

# ------------------------------------------------------------ (h) agreement with slot-keyed streams
# Genealogy-keyed child streams (results/blp/multigpu/dag.md MG05) and exact accumulation
# (MG06) change which random numbers a child of order >= 2 draws, not the estimator. The
# pinned numbers are the mean and seed-to-seed standard deviation of 32 runs of the
# previous implementation (per-chunk child seeds, float sums; seeds 0xBEEF + 1:32, this
# deck): nu L_nu power per decade band [10^d, 10^(d+1)) Hz of each scattering order
# (bands with a mean count >= 20), and the child ledger. This run must agree band by band
# within 4.5 sd, and in aggregate (chi^2 over the bands).
const SC_OLD_BANDS = [   # (order, decade d, mean Sum wE, sd)
    (0, 8, 1.265946e+30, 8.3347e+27), (0, 9, 1.446833e+31, 8.3712e+28),
    (0, 10, 7.325630e+31, 3.6373e+29), (0, 11, 4.814777e+31, 2.8290e+29),
    (0, 12, 1.430858e+30, 1.1927e+28),
    (1, 8, 1.451611e+28, 8.3568e+26), (1, 9, 6.494493e+29, 1.9957e+28),
    (1, 10, 1.583397e+31, 2.5934e+29), (1, 11, 1.466211e+32, 2.5097e+30),
    (1, 12, 7.685633e+32, 1.3205e+31), (1, 13, 1.240176e+33, 3.6397e+31),
    (1, 14, 4.788721e+32, 3.0484e+31), (1, 15, 1.826814e+31, 4.9709e+30),
    (2, 9, 9.625652e+27, 1.9405e+27), (2, 10, 5.348658e+29, 5.0572e+28),
    (2, 11, 1.251500e+31, 6.9997e+29), (2, 12, 2.167657e+32, 1.1423e+31),
    (2, 13, 1.805904e+33, 8.2653e+31), (2, 14, 9.761087e+33, 5.5153e+32),
    (2, 15, 2.276690e+34, 1.6924e+33), (2, 16, 2.236291e+34, 3.9369e+33),
    (3, 11, 3.738538e+29, 9.7192e+28), (3, 12, 1.197925e+31, 1.7427e+30),
    (3, 13, 2.167107e+32, 2.3417e+31), (3, 14, 2.962615e+33, 2.7431e+32),
    (3, 15, 2.311128e+34, 1.9049e+33), (3, 16, 1.256385e+35, 1.0138e+34)]
const SC_OLD_LEDGER = (n_children = (3.649075e+04, 2.4133e+02),
                       w_children = (2.057956e+41, 1.8989e+39),
                       scatter_loss = (2.062531e+41, 5.7502e+38))

"Sum w E of order `k` over the bins whose centre lies in decade `d`."
function band_power(res, k, d)
    o = res.orders[k + 1]
    s = 0.0
    for (i, nu) in enumerate(nu_centers(o))
        floor(Int, log10(nu)) == d && (s += sum(view(o.sumwE, :, i)))
    end
    return s
end

@testset "scattering sphere: agrees with the slot-keyed implementation within MC error" begin
    z = [(band_power(SC_RES, k, d) - m) / sd for (k, d, m, sd) in SC_OLD_BANDS]
    @info "scatter sphere gate (h)" max_abs_z = round(maximum(abs, z), digits = 2) chi2 = round(sum(abs2, z), digits = 1) nbands = length(z)
    @test maximum(abs, z) <= 4.5
    # chi^2 with 27 dof: mean 27, sd sqrt(54); the bound is ~ +5.5 sd
    @test sum(abs2, z) <= length(z) + 5.5 * sqrt(2 * length(z))
    for (f, (m, sd)) in pairs(SC_OLD_LEDGER)
        @test abs(getfield(SC_RES, f) - m) <= 4.5 * sd
    end
end
