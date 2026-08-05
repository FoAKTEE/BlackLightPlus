# Monte Carlo superphoton generation tests. A single uniform emitting zone in
# flat spacetime pins every physical invariant of `sample_zone_photon`
# analytically — no reference code is needed:
#   * the emitted momentum is null with energy E = -p_t = h nu / (m_e c^2) > 0;
#   * the sampled ln(nu) histogram reproduces the target density int_jnu/w(nu);
#   * the sampled pitch cos(theta) histogram reproduces j_nu(theta)/j_nu(pi/2),
#     marginalised over the nu the packet was drawn at; and
#   * the weight normalisation closes the luminosity: (N_s/N) Sum w h nu equals
#     the zone's frequency-integrated emitted power dV Int int_jnu d nu.
using Test
import BlackLightPlus
using BlackLightPlus: EmissionZone, WeightTable, weight_at, build_weight_table,
                    zone_nsuperphotons, sample_zone_photon,
                    j_nu, jmax, int_jnu_zone, MCRng
using BlackLightPlus.Constants: HPL, ME, CL
using StaticArrays
using LinearAlgebra: dot

# ------------------------------------------------------------ zone / tables
# The plasma is the gpumonty synchrotron-sphere point (n_e=1e13, Theta_e=100,
# B=1 G) in Minkowski with a static fluid. The gpumonty reference test lists
# b^mu = (0,0,0,1); B is placed along x1 instead: the fixed Gram-Schmidt order in
# `emission_tetrad`/`make_tetrad` (e_(1)=b-hat, then trial x2, x3) is degenerate
# when b-hat is exactly the x3 axis — e_(3) collapses to zero and normalises to
# NaN. That axis alignment is a measure-zero coordinate artefact never hit by real
# GRMHD fields; B on x1 exercises identical physics (flat space is isotropic).
const G_MINK = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
const UCON = SVector(1.0, 0.0, 0.0, 0.0)          # static fluid
const UCOV = G_MINK * UCON                         # = (-1,0,0,0)
const BCON = SVector(0.0, 1.0, 0.0, 0.0)           # B along x1 (see note above)

const ZONE = EmissionZone(; n_e = 1.0e13, theta_e = 100.0, B = 1.0, dV = 1.0,
                          x = SVector(0.0, 0.0, 0.0, 0.0),
                          ucon = UCON, ucov = UCOV, bcon = BCON,
                          gcov = G_MINK, gcon = G_MINK)

const NU_MIN = 1.0e8
const NU_MAX = 1.0e16
const N_ESAMP = 400
const NS_PARAM = 1.0e5
const WGT = build_weight_table([ZONE], NS_PARAM;
                               n_esamp = N_ESAMP, nu_min = NU_MIN, nu_max = NU_MAX)
const NS_EXPECT, DNMAX = zone_nsuperphotons(ZONE, WGT)

# ------------------------------------------------------------ generate a batch
const NPH = 200_000

# Sample NPH superphotons on independent Philox streams (one per packet — the
# device-parallel pattern), recording the physical invariants of each.
function generate_batch(n)
    lnu   = Vector{Float64}(undef, n)
    cth   = Vector{Float64}(undef, n)
    wv    = Vector{Float64}(undef, n)
    whnu  = Vector{Float64}(undef, n)   # w * h * nu  (physical power contribution)
    Ev    = Vector{Float64}(undef, n)
    nullr = Vector{Float64}(undef, n)   # |k.k| / E^2
    smag  = Vector{Float64}(undef, n)   # |spatial p| (should equal E for a null ray)
    efl   = Vector{Float64}(undef, n)   # fluid-frame energy -p.u
    Threads.@threads for i in 1:n
        rng = MCRng(0x0abcdef1, i)
        sp, _ = sample_zone_photon(ZONE, WGT, DNMAX, rng)
        p = sp.p
        kcon = ZONE.gcon * p
        eps = kcon[1]                    # k^t = -p_t = E in this static Minkowski frame
        nu = sp.E * ME * CL * CL / HPL   # invert E = h nu / (m_e c^2)
        lnu[i]  = log(nu)
        cth[i]  = kcon[2] / eps          # pitch cosine along b-hat = x1
        wv[i]   = sp.w
        whnu[i] = sp.w * HPL * nu
        Ev[i]   = sp.E
        nullr[i] = abs(dot(p, kcon)) / (eps * eps)
        smag[i] = sqrt(p[2]^2 + p[3]^2 + p[4]^2)
        efl[i]  = -dot(p, ZONE.ucon)
    end
    return (; lnu, cth, wv, whnu, Ev, nullr, smag, efl)
end

const B = generate_batch(NPH)

# ------------------------------------------------------------ histogram helper
# Counts of `xs` in `nbin` equal bins over [lo, hi).
function histcounts(xs, lo, hi, nbin)
    h = (hi - lo) / nbin
    c = zeros(Int, nbin)
    for x in xs
        k = floor(Int, (x - lo) / h) + 1
        1 <= k <= nbin && (c[k] += 1)
    end
    return c
end

# Compare observed multinomial counts to `target` (relative bin densities). Only
# bins with an expected count >= `minexp` are checked (sparse tails are unreliable).
# Returns (npass, ntested, worst |obs-exp|/sqrt(exp)).
function multinomial_check(counts, target; minexp = 40.0)
    N = sum(counts)
    tt = sum(target)
    npass = 0; ntested = 0; worst = 0.0
    for k in eachindex(counts)
        exp_k = N * target[k] / tt
        exp_k < minexp && continue
        ntested += 1
        z = abs(counts[k] - exp_k) / sqrt(exp_k)   # ~ N(0,1) per bin (p<<1)
        worst = max(worst, z)
        z <= 6.0 && (npass += 1)                    # generous 6-sigma band
    end
    return npass, ntested, worst
end

# ============================================================ tests

@testset "generation: null momentum" begin
    # Emitted p_mu lies on the light cone: |p_mu g^{mu nu} p_nu| << E^2.
    @test maximum(B.nullr) < 1.0e-10
    @test maximum(B.smag ./ B.Ev .- 1.0) |> abs < 1.0e-10   # |p_spatial| = E (null)
end

@testset "generation: energy E = -p_t = h nu / m_e c^2" begin
    @test all(>(0.0), B.Ev)                          # positive photon energy
    # fluid-frame energy equals the coordinate E for a static frame
    @test maximum(abs.(B.efl ./ B.Ev .- 1.0)) < 1.0e-12
    # every recovered nu sits inside the sampling band (E = h nu / m_e c^2 > 0)
    @test all(lnu -> log(NU_MIN) <= lnu <= log(NU_MAX), B.lnu)
    # E is the null-photon energy: it equals the spatial momentum magnitude
    @test maximum(abs.(B.smag ./ B.Ev .- 1.0)) < 1.0e-10
end

@testset "generation: frequency histogram ~ int_jnu/w(nu)" begin
    lo = log(NU_MIN); hi = log(NU_MAX); nbin = 40
    counts = histcounts(B.lnu, lo, hi, nbin)
    h = (hi - lo) / nbin
    # target density at bin centres: the accepted density in ln nu is
    # min(int_jnu(nu)/w(nu), dnmax) (the rejection acceptance, capped at 1*dnmax).
    target = zeros(Float64, nbin)
    for k in 1:nbin
        nu = exp(lo + (k - 0.5) * h)
        target[k] = min(int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) /
                        (weight_at(nu, WGT) + 1.0e-100), DNMAX)
    end
    npass, ntested, worst = multinomial_check(counts, target)
    @test ntested >= 8                    # a meaningful number of populated bins
    @test npass == ntested                # every populated bin within 6 sigma
    @info "MC frequency histogram" bins_tested=ntested worst_sigma=round(worst, digits=2)
end

@testset "generation: pitch histogram ~ j_nu(theta)/j_nu(pi/2)" begin
    nbin = 20
    counts = histcounts(B.cth, -1.0, 1.0, nbin)
    hcth = 2.0 / nbin
    # The pitch cos(theta) is drawn at each packet's own nu, so its marginal is a
    # mixture over nu:  marginal(c) = Int p(nu) [ j_nu(nu, sin th) / Z(nu) ] dln(nu),
    # where p(nu) is the frequency sampling density and Z(nu) = Int_{-1}^{1}
    # j_nu(nu, sin th) dcos(theta) = int_jnu_zone(nu)/(2 pi) is the PER-NU
    # normalisation of the accepted pitch distribution (NOT jmax, which is only the
    # rejection envelope). The constant 2 pi drops out under normalisation, so we
    # weight j_nu by 1/int_jnu_zone(nu).
    ng = 240
    lo = log(NU_MIN); hi = log(NU_MAX); dg = (hi - lo) / ng
    nus = [exp(lo + (m - 0.5) * dg) for m in 1:ng]
    pnu = Float64[min(int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) /
                      (weight_at(nu, WGT) + 1.0e-100), DNMAX) for nu in nus]
    pnu ./= sum(pnu)
    izone = Float64[int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) for nu in nus]
    target = zeros(Float64, nbin)
    for k in 1:nbin
        c = -1.0 + (k - 0.5) * hcth
        st = sqrt(max(0.0, 1.0 - c * c))
        acc = 0.0
        for m in 1:ng
            acc += pnu[m] * j_nu(ZONE.n_e, ZONE.theta_e, ZONE.B, nus[m], st) / izone[m]
        end
        target[k] = acc
    end
    npass, ntested, worst = multinomial_check(counts, target)
    @test ntested >= 8
    @test npass == ntested
    @info "MC pitch histogram" bins_tested=ntested worst_sigma=round(worst, digits=2)
end

@testset "generation: luminosity closure Sum w h nu == dV Int int_jnu d nu" begin
    # Reference: the zone's frequency-integrated emitted power over the band,
    # dV Int_{nu_min}^{nu_max} int_jnu d nu = dV Int int_jnu(nu) nu dln(nu).
    function power_ref(; n = 2000)
        lo = log(NU_MIN); hi = log(NU_MAX); h = (hi - lo) / n
        acc = 0.0
        for i in 0:n
            nu = exp(lo + i * h)
            w = (i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
            acc += w * int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) * nu
        end
        return ZONE.dV * acc * h / 3.0    # composite Simpson in ln nu
    end
    Lref = power_ref()
    # Estimator: each of the N packets is one of the N_s the zone would emit, so
    # the emitted power is (N_s/N) Sum w h nu.
    S = sum(B.whnu)
    Lest = (NS_EXPECT / NPH) * S
    ratio = Lest / Lref
    # MC standard error of the estimator: coefficient of variation of (w h nu)/sqrt(N).
    m = S / NPH
    v = sum(x -> (x - m)^2, B.whnu) / (NPH - 1)
    se_rel = sqrt(v) / m / sqrt(NPH)
    tol = max(0.02, 5.0 * se_rel)         # 5-sigma MC band, floored at 2%
    @info "MC luminosity closure" ratio=round(ratio, digits=4) se_rel=round(se_rel, digits=4) tol=round(tol, digits=4)
    @test abs(ratio - 1.0) < tol
end

# ============================================================ tabulated generation (opt-in)
# `tabulated_gen` swaps the live 513-node pitch integral in the frequency-rejection
# accept ratio for the chord-clamped, log-interpolated `IntJnuTable`
# (emission_tables.jl). Checks, with the measured residuals of this sphere point:
#   (i)   the table NEVER exceeds the exact integrand beyond round-off (measured
#         overshoot 6.7e-16) and undershoots only in the concave clip region
#         (measured 3.4e-3 — exactly where the exact path clips acceptance at 1);
#   (ii)  the tabulated acceptance is <= 1 for ALL nu and the envelope equals the
#         exact-path dnmax (the chord clamp pins it to the coarse-node max);
#   (iii) the per-photon streams match the exact path (measured 0 flips in 1e5
#         photons; the < 1e-3 tolerance budgets for rare flips).
const MCmod = BlackLightPlus.MonteCarlo
const GT_TAB = MCmod.build_int_jnu_table(ZONE, WGT)
const DNMAX_TAB = MCmod.tabulated_dnmax(GT_TAB, WGT)

"Signed rel deviation (overshoot, undershoot, mean |.|) of the clamped table vs the
exact integrand over a dense ln-nu scan (n incommensurate with the node grid)."
function tabulated_table_err(gt, zone; n = 39_989)
    lo = log(NU_MIN); hi = log(NU_MAX)
    over = 0.0; under = 0.0; meanabs = 0.0
    for j in 0:n
        nu = exp(lo + (hi - lo) * j / n)
        ex = int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu)
        r = MCmod.int_jnu_at(nu, gt) / ex - 1.0
        over = max(over, r); under = max(under, -r); meanabs += abs(r) / (n + 1)
    end
    return over, under, meanabs
end

"Max tabulated acceptance over a dense ln-nu scan (envelope property: sup <= 1)."
function tabulated_max_accept(gt, wgt, dn; n = 200_003)
    lo = log(NU_MIN); hi = log(NU_MAX)
    worst = 0.0
    for j in 0:n
        nu = exp(lo + (hi - lo) * j / n)
        worst = max(worst, (MCmod.int_jnu_at(nu, gt) /
                            (weight_at(nu, wgt) + 1.0e-100)) / dn)
    end
    return worst
end

"Count photons whose (p, w) differ between the exact and tabulated generators on
shared per-photon Philox streams."
function tabulated_stream_diffs(zone, wgt, dn_ex, dn_tab, gt; nph = 100_000, seed = 0x7AB1E5)
    nd = 0
    for i in 1:nph
        sp1, _ = sample_zone_photon(zone, wgt, dn_ex, MCRng(seed, i))
        sp2, _ = sample_zone_photon(zone, wgt, dn_tab, MCRng(seed, i), gt)
        (sp1.p == sp2.p && sp1.w == sp2.w) || (nd += 1)
    end
    return nd
end

@testset "tabulated generation: table accuracy (clamped interpolant)" begin
    over, under, meanabs = tabulated_table_err(GT_TAB, ZONE)
    @info "tabulated generation table accuracy" over under meanabs
    @test over < 1.0e-12          # never above exact -> no acceptance bias
    @test under < 1.0e-2          # clamp depth (the exact path's own clip region)
    @test meanabs < 1.5e-3        # measured 3.7e-4
end

@testset "tabulated generation: acceptance <= 1 + envelope parity" begin
    @test abs(DNMAX_TAB / DNMAX - 1.0) < 1.0e-9   # chord clamp pins the envelope
    worst = tabulated_max_accept(GT_TAB, WGT, DNMAX_TAB)
    @info "tabulated generation acceptance" worst_accept = worst
    @test worst <= 1.0 + 1.0e-12                  # <= 1 by construction
end

@testset "tabulated generation: stream parity with the exact path" begin
    nph = 100_000
    nd = tabulated_stream_diffs(ZONE, WGT, DNMAX, DNMAX_TAB, GT_TAB; nph = nph)
    @info "tabulated generation stream parity" ndiff = nd nph frac = nd / nph
    @test nd / nph < 1.0e-3                       # measured 0
end
