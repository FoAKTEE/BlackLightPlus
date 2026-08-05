# Monte Carlo GRMHD tests: the Monte Carlo radiative-transfer machinery run on a
# real AthenaK GRMHD snapshot (a=0.9 torus, M_bh=6.2e9 Msun, mass_scale from
# d_unit=1e-21 — the same deck as bench/bench_gpu_pol.jl) with curved Kerr-Schild
# geodesic tracking and the real fluid sampled every step. Physical-sanity checks
# on the emergent unpolarized spectrum, on a subsampled grid for speed:
#   (a) generated superphotons are null (|k.k|/E^2 < 1e-8) and start inside
#       emitting zones; (b) with scattering off, escaped/emitted power is in
#       (0,1]; (c) a thermal-synchrotron peak lies within the physical band;
#   (d) scattering adds a Compton tail above the peak. Skipped if no snapshot.
using Test
import BlackLightPlus
using BlackLightPlus: GRMHDModel, emission_zones, plasma_at, sample_zone_photon,
                    build_weight_table, zone_nsuperphotons,
                    mc_spectrum_grmhd, nuLnu_total, nu_centers, MCRng
using StaticArrays
using LinearAlgebra: dot

const _GRMHD_DECK = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")

# Deck constants (imaging-figure parity).
const _MSUN = 1.989e33
const _MBH = 6.2e9 * _MSUN
const _L = BlackLightPlus.GNEWT * _MBH / BlackLightPlus.CL^2
const _MASS_SCALE = 1.0e-21 * _L^3

"Build the imaging-deck GRMHD model."
_grmhd_model() = GRMHDModel(_GRMHD_DECK; bh_a = 0.9, m_bh_cgs = _MBH,
                            mass_scale_cgs = _MASS_SCALE, rat_high = 10.0, rat_low = 1.0,
                            use_p = true, gamma = 13.0 / 9.0, sigma_cut = 1.0,
                            theta_e_min = -1.0)

"|k.k| / E^2 for a superphoton with covariant momentum `sp.p` in metric `gcon`."
function _null_residual(sp, gcon)
    kk = dot(sp.p, gcon * sp.p)          # p_mu g^{mu nu} p_nu (== 0 on the light cone)
    return abs(kk) / (sp.E * sp.E)
end

"Peak-frequency index of an angle-summed nu L_nu (max over bins with counts)."
function _peak_bin(nu, L, cnt)
    ip = 0
    best = 0.0
    for i in eachindex(L)
        (cnt[i] > 0 && L[i] > best) && (best = L[i]; ip = i)
    end
    return ip
end

"Escaped power [erg s^-1] carried by bins with centre frequency > `nu_thresh`."
function _power_above(nu, L, nu_thresh)
    s = 0.0
    for i in eachindex(L)
        nu[i] > nu_thresh && (s += L[i])
    end
    return s
end

if !isfile(_GRMHD_DECK)
    @info "GRMHD MC gate skipped: snapshot not found" path = _GRMHD_DECK
else
    @testset "Monte Carlo GRMHD grid" begin
        model = _grmhd_model()

        # ---- subsampled emitting grid (fast) ----
        # stride=20 + r_emit_max=40 subsamples the 11.3M-cell grid to ~700 emitting
        # zones of the luminous torus. Ns is kept modest — the curved per-photon
        # geodesic tracking (rk4 + per-step fluid sampling) is far heavier than the
        # flat sphere push, so this test trades statistics for speed (physical
        # sanity, not a precision match).
        zones = emission_zones(model; stride = 20, r_emit_max = 40.0)
        @test !isempty(zones)
        @test all(z -> z.n_e > 0.0 && z.theta_e > 0.0 && z.B > 0.0 && z.dV > 0.0, zones)

        # ---- (a): generated superphotons null + start inside emitting zones ----
        @testset "(a) null + origin" begin
            wgt = build_weight_table(zones, 1000.0; nu_min = 1.0e8, nu_max = 1.0e16)
            rng = MCRng(0xBEEF, 1)
            nzt = min(length(zones), 200)
            worst = 0.0
            all_inside = true
            for zi in 1:nzt
                zone = zones[zi]
                _, dn = zone_nsuperphotons(zone, wgt)
                dn <= 0.0 && continue
                sp, rng = sample_zone_photon(zone, wgt, dn, rng)
                worst = max(worst, _null_residual(sp, zone.gcon))
                all_inside &= plasma_at(model, sp.x).inside
            end
            @test worst < 1.0e-8               # emitted momentum on the light cone
            @test all_inside                    # every seed starts in an emitting zone
        end

        # ---- run both cases: no scatter (energy reference) + scatter (Compton) ----
        # C=32 (coarse but adequate stepping for a sanity check), bias0 boosts the
        # rare scattering so the Compton tail is well-sampled at Ns=1500.
        r0 = mc_spectrum_grmhd(model, 1_500; scattering = false, stride = 20,
                               r_emit_max = 40.0, C = 32.0, seed = 0xC0FFEE, n_ebins = 120)
        r1 = mc_spectrum_grmhd(model, 1_500; scattering = true, stride = 20,
                               r_emit_max = 40.0, C = 32.0, seed = 0xC0FFEE, n_ebins = 120,
                               bias0 = 300.0, max_scatt = 3)

        @test r0.n_zones > 0
        @test r0.n_recorded > 0
        @test r0.emitted_power > 0.0

        # ---- (b): energy conservation (absorption + capture only) ----
        @testset "(b) energy conservation" begin
            ratio = r0.escaped_power / r0.emitted_power
            @test 0.0 < ratio <= 1.0 + 1.0e-9
        end

        # ---- (c): synchrotron bump present in a physical band ----
        @testset "(c) synchrotron bump" begin
            nu, L, cnt = nuLnu_total(r0.total)
            ip = _peak_bin(nu, L, cnt)
            @test ip > 0
            @test L[ip] > 0.0
            @test 1.0e9 <= nu[ip] <= 1.0e18     # broad thermal-synchrotron band
        end

        # ---- (d): Compton tail with scattering ----
        @testset "(d) Compton tail" begin
            nu0, L0, c0 = nuLnu_total(r0.total)
            nu1, L1, c1 = nuLnu_total(r1.total)
            ip0 = _peak_bin(nu0, L0, c0)
            nu_peak = nu0[ip0]
            # escaped power well above the synchrotron peak (x30) appears with
            # scattering, and exceeds the no-scatter run there.
            hi = 30.0 * nu_peak
            p_scat = _power_above(nu1, L1, hi)
            p_dir = _power_above(nu0, L0, hi)
            @test p_scat > p_dir
            @test p_scat > 0.0
        end
    end
end

# ------------------------------------------------------------ tabulated generation (opt-in)
# GRMHD twin of the tabulated-generation tests (test_mc_generate): the opt-in reuses
# the already-cached per-(zone, node) `ZoneEmissionTable` integrals as the frequency-
# rejection numerator, with a per-zone envelope over the same interpolant — so the
# acceptance must be <= 1 for every zone and every nu (verified on a zone sample over
# a dense scan), and a tabulated run must pass the same physical-sanity checks as the
# exact one (a stats run at this deck measured zero stream flips over 1502 photons).
# Skipped when the snapshot is unavailable, like the main testset.
if isfile(_GRMHD_DECK)
    using BlackLightPlus: weight_at
    const MCmod = BlackLightPlus.MonteCarlo

    "Max tabulated acceptance of zone `z` over a dense ln-nu scan of the band."
    function grmhd_max_accept(lint, dnv, wgt, z; n = 20_000)
        lo = log(1.0e8); hi = log(1.0e16)
        gz = MCmod.IntJnuTable(wgt.lnu_min, wgt.dlnu, view(lint, :, z))
        worst = 0.0
        for j in 0:n
            nu = exp(lo + (hi - lo) * j / n)
            worst = max(worst, (MCmod.int_jnu_at(nu, gz) /
                                (weight_at(nu, wgt) + 1.0e-100)) / dnv[z])
        end
        return worst
    end

    @testset "GRMHD tabulated generation (opt-in)" begin
        model = _grmhd_model()
        zones = emission_zones(model; stride = 20, r_emit_max = 40.0)
        jt = MCmod.build_zone_emission_table(zones; nu_min = 1.0e8, nu_max = 1.0e16)
        wgt = MCmod.build_weight_table(jt, 1000.0)
        lint, dnv = MCmod.zone_int_jnu_tables(jt, wgt)
        @test size(lint) == (jt.n_esamp + 1, length(zones))
        @test all(>(0.0), dnv)
        # acceptance <= 1 for a zone sample (first / middle / last)
        worst = 0.0
        for z in (1, length(zones) ÷ 2, length(zones))
            worst = max(worst, grmhd_max_accept(lint, dnv, wgt, z))
        end
        @info "GRMHD tabulated acceptance" worst_accept = worst
        @test worst <= 1.0 + 1.0e-12

        # tabulated no-scatter run passes the same physical-sanity checks
        rt = mc_spectrum_grmhd(model, 800; scattering = false, stride = 20,
                               r_emit_max = 40.0, C = 32.0, seed = 0xC0FFEE,
                               n_ebins = 120, tabulated_gen = true)
        @test rt.n_gen > 0 && rt.n_recorded > 0
        ratio = rt.escaped_power / rt.emitted_power
        @test 0.0 < ratio <= 1.0 + 1.0e-9        # energy conservation (check (b))
        nu, L, cnt = nuLnu_total(rt.total)
        ip = _peak_bin(nu, L, cnt)
        @test ip > 0 && 1.0e9 <= nu[ip] <= 1.0e18  # synchrotron bump (check (c))
    end
end
