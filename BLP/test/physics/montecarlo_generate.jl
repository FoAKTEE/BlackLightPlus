# Superphoton generation from one uniform zone in flat spacetime, where every invariant
# is analytic: the momentum is null with E = −p_t = hν/mₑc² > 0, the ln ν histogram
# follows min(∫j dΩ/w, dnmax), the pitch histogram follows j_ν(θ) marginalised over the
# drawn ν, and Σ w hν closes the zone's emitted power. Tabulated generation keeps the
# acceptance ≤ 1, stays within the clip region of the exact integrand, and draws the
# same packets as the exact path. The sphere generator draws its points in the ball.
using Test, BLP
using StaticArrays
using LinearAlgebra: dot
using BLP.Numerics: HPL, ME, CL, MCRng, primary_rng
using BLP.Matter: EmissionZone, UniformSphere
using BLP.MonteCarlo: GenerationTables, generate, weight_at, zone_nsuperphotons, j_nu,
                      int_jnu_zone, int_jnu_at, tabulated_dnmax, table_zone, WEIGHT_FLOOR
using BLP.Data: build_weight_table, build_int_jnu_table

const G_MINK = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
const UCON = SVector(1.0, 0.0, 0.0, 0.0)
const ZONE = EmissionZone(; n_e = 1.0e13, theta_e = 100.0, B = 1.0, dV = 1.0, x = zeros(4),
                          ucon = UCON, ucov = G_MINK * UCON, bcon = SVector(0.0, 1.0, 0.0, 0.0),
                          gcov = G_MINK, gcon = G_MINK)
const NU_MIN = 1.0e8
const NU_MAX = 1.0e16
const WGT = build_weight_table([ZONE], 1.0e5; n_esamp = 400, nu_min = NU_MIN, nu_max = NU_MAX)
const NS_EXPECT, DNMAX = zone_nsuperphotons(ZONE, WGT)
const TABS = GenerationTables(WGT, DNMAX)
const NPH = 200_000

function generate_batch(n)
    lnu = zeros(n); cth = zeros(n); whnu = zeros(n); E = zeros(n); nullr = zeros(n)
    smag = zeros(n); efl = zeros(n)
    Threads.@threads for i in 1:n
        sp, _ = generate(ZONE, TABS, MCRng(0x0abcdef1, i))
        p = sp.p
        kcon = ZONE.gcon * p
        nu = sp.E * ME * CL * CL / HPL
        lnu[i] = log(nu); cth[i] = kcon[2] / kcon[1]; whnu[i] = sp.w * HPL * nu
        E[i] = sp.E; nullr[i] = abs(dot(p, kcon)) / (kcon[1] * kcon[1])
        smag[i] = sqrt(p[2]^2 + p[3]^2 + p[4]^2); efl[i] = -dot(p, ZONE.ucon)
    end
    return (; lnu, cth, whnu, E, nullr, smag, efl)
end

const B = generate_batch(NPH)

function histcounts(xs, lo, hi, nbin)
    h = (hi - lo) / nbin
    c = zeros(Int, nbin)
    for x in xs
        k = floor(Int, (x - lo) / h) + 1
        1 <= k <= nbin && (c[k] += 1)
    end
    return c
end

# (bins within 6σ, bins tested) of multinomial counts against relative densities.
function multinomial_check(counts, target; minexp = 40.0)
    N = sum(counts); tt = sum(target)
    npass = 0; ntested = 0
    for k in eachindex(counts)
        e = N * target[k] / tt
        e < minexp && continue
        ntested += 1
        abs(counts[k] - e) / sqrt(e) <= 6.0 && (npass += 1)
    end
    return npass, ntested
end

accepted_density(nu) = min(int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) /
                           (weight_at(nu, WGT) + WEIGHT_FLOOR), DNMAX)

@testset "Monte Carlo generation" begin
    @testset "null momentum, E = hν/mₑc²" begin
        @test maximum(B.nullr) < 1.0e-10
        @test maximum(abs.(B.smag ./ B.E .- 1.0)) < 1.0e-10
        @test all(>(0.0), B.E)
        @test maximum(abs.(B.efl ./ B.E .- 1.0)) < 1.0e-12
        @test all(l -> log(NU_MIN) <= l <= log(NU_MAX), B.lnu)
    end

    @testset "frequency histogram" begin
        lo = log(NU_MIN); hi = log(NU_MAX); nbin = 40; h = (hi - lo) / nbin
        target = [accepted_density(exp(lo + (k - 0.5) * h)) for k in 1:nbin]
        npass, ntested = multinomial_check(histcounts(B.lnu, lo, hi, nbin), target)
        @test ntested >= 8
        @test npass == ntested
    end

    @testset "pitch histogram" begin
        nbin = 20; hc = 2.0 / nbin; ng = 240
        lo = log(NU_MIN); hi = log(NU_MAX); dg = (hi - lo) / ng
        nus = [exp(lo + (m - 0.5) * dg) for m in 1:ng]
        pnu = accepted_density.(nus); pnu ./= sum(pnu)
        iz = [int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) for nu in nus]
        target = map(1:nbin) do k
            c = -1.0 + (k - 0.5) * hc
            st = sqrt(max(0.0, 1.0 - c * c))
            sum(pnu[m] * j_nu(ZONE.n_e, ZONE.theta_e, ZONE.B, nus[m], st) / iz[m] for m in 1:ng)
        end
        npass, ntested = multinomial_check(histcounts(B.cth, -1.0, 1.0, nbin), target)
        @test ntested >= 8
        @test npass == ntested
    end

    @testset "luminosity closure" begin
        n = 2000; lo = log(NU_MIN); hi = log(NU_MAX); h = (hi - lo) / n
        acc = sum(((i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)) *
                  int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, exp(lo + i * h)) * exp(lo + i * h)
                  for i in 0:n)
        Lref = ZONE.dV * acc * h / 3.0
        S = sum(B.whnu)
        m = S / NPH
        se = sqrt(sum(x -> (x - m)^2, B.whnu) / (NPH - 1)) / m / sqrt(NPH)
        @test abs((NS_EXPECT / NPH) * S / Lref - 1.0) < max(0.02, 5.0 * se)
    end

    @testset "tabulated generation" begin
        gt = build_int_jnu_table(ZONE, WGT)
        dn = tabulated_dnmax(gt, WGT)
        @test abs(dn / DNMAX - 1.0) < 1.0e-9
        over = 0.0; under = 0.0; worst = 0.0
        lo = log(NU_MIN); hi = log(NU_MAX)
        for j in 0:39_989
            nu = exp(lo + (hi - lo) * j / 39_989)
            r = int_jnu_at(nu, gt) / int_jnu_zone(ZONE.n_e, ZONE.theta_e, ZONE.B, nu) - 1.0
            over = max(over, r); under = max(under, -r)
            worst = max(worst, (int_jnu_at(nu, gt) / (weight_at(nu, WGT) + WEIGHT_FLOOR)) / dn)
        end
        @test over < 1.0e-12
        @test under < 1.0e-2
        @test worst <= 1.0 + 1.0e-12
        tt = GenerationTables(WGT, dn, gt)
        nd = count(1:20_000) do i
            a, _ = generate(ZONE, TABS, MCRng(0x7AB1E5, i))
            b, _ = generate(ZONE, tt, MCRng(0x7AB1E5, i))
            !(a.p == b.p && a.w == b.w)
        end
        @test nd / 20_000 < 1.0e-3
    end

    @testset "sphere: emission points fill the ball" begin
        s = UniformSphere(R = 2.0)
        z = table_zone(s)
        @test z.dV ≈ (4 / 3) * π * 8.0 && z.x == zeros(4)
        wgt = build_weight_table([z], 1.0e4)
        tabs = GenerationTables(wgt, zone_nsuperphotons(z, wgt)[2])
        r = map(1:20_000) do g
            sp, _ = generate(s, tabs, primary_rng(3, g))
            sqrt(sp.x[2]^2 + sp.x[3]^2 + sp.x[4]^2)
        end
        @test maximum(r) <= 2.0
        @test abs(count(<(1.0), r) / length(r) - 1 / 8) < 0.01   # (1/2)³ of the volume
    end
end
