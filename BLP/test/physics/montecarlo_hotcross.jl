# The hot Compton cross section: the Klein–Nishina total against an independent closed
# form and the Thomson limit, the Maxwell–Jüttner normalisation and its continuity at the
# Bessel/series switch, the cold limits of the quadrature, and the table lookup against
# the quadrature and the gpumonty `hotcross.dat` (skipped when the file is absent).
using Test, BLP
using BLP.Numerics: SIGMA_THOMSON, HPL, ME, CL, MP
using BLP.MonteCarlo: kn_total, boostcross, dNdgamma_e, total_compton_cross_num, HotcrossTable,
                      hotcross_lookup, kappa_es, alpha_inv_scatt, MINW, MAXW, MINT, MAXT, NW, NT
using BLP.Data: load_hotcross

# The Klein–Nishina total over σ_T in the grouping of Rybicki & Lightman (7.5).
kn_textbook(x) = 0.75 * ((1.0 + x) / x^3 * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - log(1.0 + 2.0 * x)) +
                         log(1.0 + 2.0 * x) / (2.0 * x) - (1.0 + 3.0 * x) / (1.0 + 2.0 * x)^2)

# (ε, Θₑ) of table node (i, j).
function node(i, j)
    dlw = log10(MAXW / MINW) / NW
    dlT = log10(MAXT / MINT) / NT
    return 10.0^(log10(MINW) + i * dlw), 10.0^(log10(MINT) + j * dlT)
end

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

const HOTCROSS_DAT = Main.Decks.HOTCROSS_DAT

@testset "Monte Carlo hot cross section" begin
    @testset "Klein–Nishina total" begin
        @test kn_total(1e-10) ≈ 1.0 atol = 1e-8
        @test kn_total(1e-6) ≈ 1.0 atol = 1e-5
        @test kn_total(1e-3) ≈ 1.0 - 2e-3 rtol = 1e-5
        for x in (0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0)
            @test kn_total(x) ≈ kn_textbook(x) rtol = 1e-9
        end
        @test kn_total(1.0) ≈ 0.4307278 rtol = 1e-5
        @test kn_total(10.0) ≈ 0.1227598 rtol = 1e-5
        for (a, b) in ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0))
            @test kn_total(a) > kn_total(b)
        end
        @test boostcross(1e-8, 0.0, 1.0) ≈ 1.0 atol = 1e-6
    end

    @testset "Maxwell–Jüttner normalisation and branch continuity" begin
        for th in (0.1, 1.0, 5.0)
            @test mj_integral(th) ≈ 1.0 rtol = 2e-3
        end
        below = dNdgamma_e(0.0099999, 1.005)
        above = dNdgamma_e(0.0100001, 1.005)
        @test abs(below - above) / above < 1e-4
        lo = total_compton_cross_num(1e-6, 0.0099999)
        hi = total_compton_cross_num(1e-6, 0.0100001)
        @test abs(lo - hi) / hi < 1e-6
    end

    @testset "quadrature: cold limits and Klein–Nishina decline" begin
        @test total_compton_cross_num(1e-6, 1e-3) ≈ SIGMA_THOMSON rtol = 1e-2
        @test total_compton_cross_num(10.0, 1e-5) === kn_total(10.0) * SIGMA_THOMSON
        @test total_compton_cross_num(1e3, 1e-6) === kn_total(1e3) * SIGMA_THOMSON
        @test total_compton_cross_num(1e-13, 1e-6) === SIGMA_THOMSON
        prev = Inf
        for eps in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
            v = total_compton_cross_num(eps, 3.0)
            @test v < prev
            prev = v
        end
    end

    @testset "lookup branches on a uniform table" begin
        tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
        @test hotcross_lookup(tbl, 1e-7, 1e-3) === SIGMA_THOMSON
        @test hotcross_lookup(tbl, 5.0, 1e-6) === kn_total(5.0) * SIGMA_THOMSON
        @test hotcross_lookup(tbl, 0.5, 2.0) ≈ SIGMA_THOMSON rtol = 1e-14
        @test hotcross_lookup(tbl, 2.0e15, 2.0) === total_compton_cross_num(2.0e15, 2.0)
        nu = 1e18; th = 2.0; ne = 1e6
        sig = hotcross_lookup(tbl, HPL * nu / (ME * CL * CL), th)
        @test kappa_es(tbl, nu, th) === sig / MP
        @test alpha_inv_scatt(tbl, nu, th, ne) === nu * ne * sig
    end

    if isfile(HOTCROSS_DAT)
        @testset "gpumonty table: nodes and interpolation" begin
            tbl = load_hotcross(HOTCROSS_DAT)
            @test size(tbl.logsigma) == (NW + 1, NT + 1)
            worst = 0.0
            for (i, j) in ((50, 40), (100, 50), (150, 60), (120, 45), (80, 55), (60, 30), (200, 70))
                eps, th = node(i, j)
                tv = 10.0^tbl.logsigma[i + 1, j + 1]
                worst = max(worst, abs(tv - total_compton_cross_num(eps, th)) / tv)
            end
            @test worst < 1e-6
            worst = 0.0
            for (eps, th) in ((0.5, 2.0), (2.0, 5.0), (0.1, 0.5), (5.0, 10.0), (0.01, 0.3),
                              (50.0, 20.0), (0.3, 1.0), (1.0, 3.0))
                num = total_compton_cross_num(eps, th)
                worst = max(worst, abs(hotcross_lookup(tbl, eps, th) - num) / num)
            end
            @test worst < 0.03
        end
    else
        @info "gpumonty hotcross.dat not found: table checks skipped" path = HOTCROSS_DAT
    end
end
