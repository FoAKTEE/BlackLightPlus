# Monte Carlo thin-sphere spectrum test. A uniform, optically-thin thermal-
# synchrotron sphere (n_e = 1e13, Theta_e = 100, B = 1 G, R = 1 cm) is run through
# the full generate -> track -> record pipeline; its escaping angle-summed spectrum
# nu L_nu(nu) is compared to the closed-form thin-sphere emissivity
#     nu L_nu = nu * V * int_jnu_zone(nu),   V = (4/3) pi R^3,
# the same angle-integrated comparison as the gpumonty python sphere checker. Both
# the MC weights and the analytic curve derive from `int_jnu_zone`, so there is no
# free normalisation. Pass: per-bin |Delta L / L| < 0.10 (gpumonty's band) over
# [1e9, 1e14] Hz bins holding >= 50 packets; bolometric power matches to 3%.
using Test
import BlackLightPlus
using BlackLightPlus: SphereModel, mc_spectrum, nuLnu_total, analytic_nuLnu,
                    total_power, int_jnu_zone

# ------------------------------------------------------------ run the simulation
# Fixed seed => deterministic, reproducible. Ns is the target packet count;
# packets are uniform in ln nu over the generation band, so in-band bins each hold
# ~ Ns * dln nu_bin / band superphotons (thousands here).
const SPHERE   = SphereModel()                 # n_e=1e13, Theta_e=100, B=1, R=1
const NS_GATE  = 2_000_000
const NU_LO    = 1.0e9                          # test band [Hz]
const NU_HI    = 1.0e14
const WT_NUMIN = 1.0e8                          # generation / spectrum band [Hz]
const WT_NUMAX = 1.0e16
const MINCOUNT = 50                             # sparse-bin cut (superphotons)

const SPEC = mc_spectrum(SPHERE, NS_GATE; seed = 0xC0FFEE,
                         nu_min = WT_NUMIN, nu_max = WT_NUMAX, n_ebins = 200)

# ------------------------------------------------------------ helpers

"""
    sphere_gate(spec, model) -> (worst_rel, worst_nu, n_tested, n_pass)

Worst per-bin |MC - analytic| / analytic over well-populated in-band bins, the
frequency at which it occurs, and the (tested, passing<0.10) bin counts. The MC
spectrum is the angle-summed `nuLnu_total`; the analytic is `analytic_nuLnu` at the
bin centre.
"""
function sphere_gate(spec, model; nu_lo = NU_LO, nu_hi = NU_HI, min_count = MINCOUNT)
    nu, Lmc, cnt = nuLnu_total(spec)
    worst = 0.0
    worst_nu = 0.0
    n_tested = 0
    n_pass = 0
    for i in eachindex(nu)
        (nu[i] < nu_lo || nu[i] > nu_hi) && continue
        cnt[i] < min_count && continue
        a = analytic_nuLnu(model, nu[i])
        a > 0.0 || continue
        rel = abs(Lmc[i] - a) / a
        n_tested += 1
        rel < 0.10 && (n_pass += 1)
        if rel > worst
            worst = rel
            worst_nu = nu[i]
        end
    end
    return worst, worst_nu, n_tested, n_pass
end

"""
    analytic_bolometric(model) -> Float64

Frequency-integrated analytic luminosity `Int nu L_nu dln nu = Int int_jnu_zone V d
nu` [erg s^-1] over the generation band, by composite Simpson in ln nu.
"""
function analytic_bolometric(model; nu_min = WT_NUMIN, nu_max = WT_NUMAX, n = 4000)
    lo = log(nu_min)
    hi = log(nu_max)
    h = (hi - lo) / n
    acc = 0.0
    for i in 0:n
        nu = exp(lo + i * h)
        wgt = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        acc += wgt * analytic_nuLnu(model, nu)
    end
    return acc * h / 3.0
end

# ------------------------------------------------------------ the tests

@testset "MC sphere: spectrum vs analytic (<10% per bin)" begin
    worst, worst_nu, n_tested, n_pass = sphere_gate(SPEC, SPHERE)
    @info "MC sphere spectrum gate" n_tested worst_rel = round(worst, digits = 4) worst_nu n_pass
    @test n_tested >= 50                    # the [1e9,1e14] band is densely populated
    @test n_pass == n_tested                # EVERY well-populated bin within 10%
    @test worst < 0.10                      # worst-bin relative error
end

@testset "MC sphere: energy conservation (bolometric power)" begin
    Lmc = total_power(SPEC)
    Lan = analytic_bolometric(SPHERE)
    ratio = Lmc / Lan
    @info "MC sphere energy conservation" Lmc Lan ratio = round(ratio, digits = 4)
    @test isapprox(ratio, 1.0; atol = 0.03)   # within Monte Carlo error (few %)
end

# ------------------------------------------------------------ tabulated generation (opt-in)
# The `tabulated_gen` opt-in (emission_tables.jl `IntJnuTable`) must be a drop-in
# statistical twin of the exact generator: the same analytic + energy checks AND
# near-identical streams — the chord-clamped table deviates from the exact
# integrand only where the exact path clips its acceptance at 1, so at this deck
# the two runs agree bin-for-bin (measured 0 accept flips; the tolerances below
# budget for the rare interpolation-error flip).
const SPEC_TAB = mc_spectrum(SPHERE, NS_GATE; seed = 0xC0FFEE,
                             nu_min = WT_NUMIN, nu_max = WT_NUMAX, n_ebins = 200,
                             tabulated_gen = true)

"Worst per-bin count z-score |c_b - c_a| / sqrt(max(c_a, 1)) between two runs."
function count_zscore(sa, sb)
    ca = vec(sum(sa.count, dims = 1))
    cb = vec(sum(sb.count, dims = 1))
    worst = 0.0
    for i in eachindex(ca)
        worst = max(worst, abs(cb[i] - ca[i]) / sqrt(max(ca[i], 1)))
    end
    return worst
end

@testset "MC sphere (tabulated_gen): spectrum gates + exact-path agreement" begin
    worst, worst_nu, n_tested, n_pass = sphere_gate(SPEC_TAB, SPHERE)
    @info "MC sphere tabulated gate" n_tested worst_rel = round(worst, digits = 4) n_pass
    @test n_tested >= 50
    @test n_pass == n_tested
    @test worst < 0.10                          # same analytic tolerance as the exact run
    ratio = total_power(SPEC_TAB) / analytic_bolometric(SPHERE)
    @test isapprox(ratio, 1.0; atol = 0.03)     # energy conservation
    # exact-path agreement (stream-parity checks)
    rel = abs(total_power(SPEC_TAB) - total_power(SPEC)) / total_power(SPEC)
    z = count_zscore(SPEC, SPEC_TAB)
    @info "MC sphere tabulated vs exact" total_power_rel = rel worst_count_z = z
    @test rel < 1.0e-4                          # total nu L nu parity (measured 0)
    @test z < 5.0                               # per-bin counts within Poisson error
end
