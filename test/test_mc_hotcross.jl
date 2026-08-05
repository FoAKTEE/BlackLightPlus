# Monte Carlo hot-cross-section tests. sigma_h(eps, Theta_e) is the thermally
# averaged Compton opacity: the Klein-Nishina total cross section Doppler-boosted
# into every electron rest frame and averaged over the Maxwell-Juttner electron
# distribution. Ground truth is (i) the analytic Klein-Nishina total in an
# independent closed form, (ii) the cold Thomson / KN limits, and (iii) the
# gpumonty `hotcross.dat` lookup table (the same quadrature, so agreement is to
# machine precision at the grid nodes and to interpolation error off them).
using Test
using BlackLightPlus: kn_total, boostcross, dNdgamma_e, total_compton_cross_num,
                    HotcrossTable, load_hotcross, hotcross_lookup,
                    kappa_es, alpha_inv_scatt
using BlackLightPlus.Constants: SIGMA_THOMSON, HPL, ME, CL, MP
const _MC = BlackLightPlus.MonteCarlo

# ------------------------------------------------------------ reference / helpers

# Independent closed form of the Klein-Nishina total cross section (over sigma_T),
# algebraically distinct from `kn_total`'s grouping (Rybicki & Lightman 7.5): a
# genuine cross-check, not a restatement.
function kn_total_textbook(x)
    return 0.75 * ((1.0 + x) / x^3 * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - log(1.0 + 2.0 * x)) +
                   log(1.0 + 2.0 * x) / (2.0 * x) - (1.0 + 3.0 * x) / (1.0 + 2.0 * x)^2)
end

# (eps, Theta_e) at integer grid node (i, j) of the hotcross table.
function node_eps_theta(i, j)
    dlw = log10(_MC.MAXW / _MC.MINW) / _MC.NW
    dlT = log10(_MC.MAXT / _MC.MINT) / _MC.NT
    return 10.0^(log10(_MC.MINW) + i * dlw), 10.0^(log10(_MC.MINT) + j * dlT)
end

# Worst relative error between the table lookup and the direct quadrature.
function max_lookup_vs_num(tbl, pts)
    m = 0.0
    for (eps, th) in pts
        lk = hotcross_lookup(tbl, eps, th)
        num = total_compton_cross_num(eps, th)
        m = max(m, abs(lk - num) / num)
    end
    return m
end

# Worst relative error between the loaded table node and the direct quadrature.
function max_node_vs_num(tbl, nodes)
    m = 0.0
    for (i, j) in nodes
        eps, th = node_eps_theta(i, j)
        tabval = 10.0^tbl.logsigma[i + 1, j + 1]
        m = max(m, abs(tabval - total_compton_cross_num(eps, th)) / tabval)
    end
    return m
end

# gpumonty reference table (override with BLP_REF_GPUMONTY).
const _HOTCROSS_DAT = get(ENV, "BLP_REF_GPUMONTY",
    "/data/haiyangw/claude/Mahakala/ref-code/gpumonty/table/hotcross.dat")

# ------------------------------------------------------------ Klein-Nishina total

@testset "kn_total: Thomson limit + KN closed form" begin
    # -> 1 (Thomson) as eps_e -> 0
    @test kn_total(1e-10) ≈ 1.0 atol = 1e-8
    @test kn_total(1e-6)  ≈ 1.0 atol = 1e-5
    # low-energy expansion 1 - 2 eps_e joins the grouped form at the switch point
    # eps_e = 1e-3 to the design tolerance (~5e-6, the O(eps^2) truncation error)
    @test kn_total(1e-3) ≈ 1.0 - 2e-3 rtol = 1e-5

    # matches an independent closed form across the KN regime
    for x in (0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0)
        @test kn_total(x) ≈ kn_total_textbook(x) rtol = 1e-9
    end

    # spot literature values (sigma_KN / sigma_T)
    @test kn_total(1.0)  ≈ 0.4307278 rtol = 1e-5
    @test kn_total(10.0) ≈ 0.1227598 rtol = 1e-5

    # strictly decreasing in eps (KN suppression)
    for (a, b) in ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0))
        @test kn_total(a) > kn_total(b)
    end
end

# ------------------------------------------------------------ Maxwell-Juttner dN/dgamma

@testset "dNdgamma_e normalisation" begin
    # dN/dgamma_e integrates to 1 over gamma_e (midpoint rule, hot regime).
    function mj_integral(theta_e)
        dg = theta_e * 1e-3
        acc = 0.0
        g = 1.0 + 0.5 * dg
        while g < 1.0 + 40.0 * theta_e
            acc += dNdgamma_e(theta_e, g) * dg
            g += dg
        end
        return acc
    end
    for th in (0.1, 1.0, 5.0)
        @test mj_integral(th) ≈ 1.0 rtol = 2e-3
    end
    # boostcross is bounded by the Thomson value in the flux-weighted sense
    @test boostcross(1e-8, 0.0, 1.0) ≈ 1.0 atol = 1e-6
end

@testset "dNdgamma_e: continuous across the Theta_e = 1e-2 K2f branch switch" begin
    # The Maxwell-Juttner normalisation K2f switches from the exact
    # bessel_k2(1/Theta_e) exp(1/Theta_e) (Theta_e > 1e-2) to a large-argument
    # asymptotic series below (the exact product overflows -> NaN by ~1.4e-3). With
    # only the leading sqrt(pi Theta_e/2) term the dropped 15 Theta_e/8 correction
    # (1.9% at the switch) put a knife-edge discontinuity into the normalisation, so
    # sigma_h was ~1.85% high across MINT < Theta_e < 1e-2. Carrying the series
    # through the Theta_e^2 term (105 Theta_e^2/128) removes it.
    g = 1.005
    below = dNdgamma_e(0.0099999, g)   # non-relativistic asymptotic-series branch
    above = dNdgamma_e(0.0100001, g)   # exact-Bessel branch
    # pre-fix this jump was 1.885e-2; what remains is the PHYSICAL variation across
    # the 2e-7-wide switch gap (dN varies smoothly), not a normalisation artefact.
    @test abs(below - above) / above < 1e-4

    # the discontinuity propagated into the hot cross section; a soft photon
    # (kn_total ~ 1) isolates the normalisation. Pre-fix jump ~1.85e-2, now ~2e-7.
    eps = 1e-6
    lo = total_compton_cross_num(eps, 0.0099999)
    hi = total_compton_cross_num(eps, 0.0100001)
    @test abs(lo - hi) / hi < 1e-6
end

# ------------------------------------------------------------ numeric quadrature limits

@testset "total_compton_cross_num: cold limits" begin
    # cold, soft -> Thomson (quadrature branch, Theta_e > MINT)
    @test total_compton_cross_num(1e-6, 1e-3) ≈ SIGMA_THOMSON rtol = 1e-2
    # cold hard photon short-circuit (Theta_e < MINT) -> kn_total(eps) sigma_T
    @test total_compton_cross_num(10.0, 1e-5) ≈ kn_total(10.0) * SIGMA_THOMSON rtol = 1e-12
    @test total_compton_cross_num(1e3, 1e-6) ≈ kn_total(1e3) * SIGMA_THOMSON rtol = 1e-12
    # both cold corners -> Thomson exactly
    @test total_compton_cross_num(1e-13, 1e-6) == SIGMA_THOMSON

    # monotonic decrease with eps at fixed Theta_e in the KN regime
    prev = Inf
    for eps in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
        v = total_compton_cross_num(eps, 3.0)
        @test v < prev
        prev = v
    end
end

# ------------------------------------------------------------ table lookup

@testset "hotcross_lookup: table vs quadrature vs reference dat" begin
    @test isfile(_HOTCROSS_DAT)
    tbl = load_hotcross(_HOTCROSS_DAT)
    @test size(tbl.logsigma) == (_MC.NW + 1, _MC.NT + 1)

    # loaded table nodes reproduce our quadrature to machine precision (identical
    # Bessel fits + identical midpoint rule as gpumonty). Nodes are kept away from
    # the Theta_e = 1e-2 K2f branch (j=20), a ~2% knife-edge in the piecewise
    # Maxwell-Juttner normalisation that the bilinear table smooths over anyway.
    nodes = ((50, 40), (100, 50), (150, 60), (120, 45), (80, 55), (60, 30), (200, 70))
    @test max_node_vs_num(tbl, nodes) < 1e-6

    # bilinear interpolation matches the direct quadrature to a few %.
    pts = ((0.5, 2.0), (2.0, 5.0), (0.1, 0.5), (5.0, 10.0), (0.01, 0.3),
           (50.0, 20.0), (0.3, 1.0), (1.0, 3.0))
    @test max_lookup_vs_num(tbl, pts) < 0.03

    # lookup branch order: soft/cold -> sigma_T; cold hard -> kn_total sigma_T
    @test hotcross_lookup(tbl, 1e-7, 1e-3) == SIGMA_THOMSON       # eps*Theta < 1e-6
    @test hotcross_lookup(tbl, 5.0, 1e-6) ≈ kn_total(5.0) * SIGMA_THOMSON rtol = 1e-12

    # opacity wrappers: alpha_inv,s = nu n_e sigma_h, kappa_es = sigma_h/m_p
    nu = 1e18; th = 2.0; ne = 1e6
    eps = HPL * nu / (ME * CL * CL)
    sig = hotcross_lookup(tbl, eps, th)
    @test kappa_es(tbl, nu, th) ≈ sig / MP rtol = 1e-12
    @test alpha_inv_scatt(tbl, nu, th, ne) ≈ nu * ne * sig rtol = 1e-12
end
