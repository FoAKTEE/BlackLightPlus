# The Monte Carlo table builders: the hot-cross table holds the quadrature at its nodes
# and the gpumonty file (when present) agrees with it; the serial and exact weight tables
# agree, the exact one bitwise for every thread count; per-zone counts equal the
# per-zone formula; the tabulated emissivities keep every weight node, clip interior
# nodes to the chord and keep the acceptance at most 1.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Spacetime: KerrSchildCartesian
using BlackLightPlus.MonteCarlo: total_compton_cross_num, int_jnu_zone, zone_nsuperphotons, int_jnu_at,
                      weight_at, tabulated_dnmax, IntJnuTable, WEIGHT_FLOOR, MINW, MAXW, MINT,
                      MAXT, NW, NT, MC_N_ESAMP, INT_JNU_REFINE
using BlackLightPlus.Data: load_hotcross, build_hotcross_table, build_weight_table,
                build_zone_emission_table, zone_counts, build_int_jnu_table, zone_int_jnu_tables

const DK = Main.Decks
const HOTCROSS_DAT = DK.HOTCROSS_DAT

@testset "Data: Monte Carlo tables" begin
    @testset "hot-cross table" begin
        tbl = build_hotcross_table()
        @test size(tbl.logsigma) == (NW + 1, NT + 1)
        dlw = log10(MAXW / MINW) / NW; dlT = log10(MAXT / MINT) / NT
        for (i, j) in ((0, 0), (17, 80), (110, 40), (220, 3), (64, 61))
            eps = 10.0^(log10(MINW) + i * dlw); th = 10.0^(log10(MINT) + j * dlT)
            @test tbl.logsigma[i + 1, j + 1] === log10(total_compton_cross_num(eps, th))
        end
        if isfile(HOTCROSS_DAT)
            ref = load_hotcross(HOTCROSS_DAT)
            @test size(ref.logsigma) == size(tbl.logsigma)
            # above Θₑ = 1e-2 (column 21) both evaluate the Bessel normalisation; below it
            # gpumonty keeps only the leading term of the large-argument series
            cols = 22:(NT + 1)
            @test maximum(abs.(10.0 .^ ref.logsigma[:, cols] ./ 10.0 .^ tbl.logsigma[:, cols] .- 1)) < 1e-8
        else
            @info "gpumonty hotcross.dat not found: file comparison skipped" path = HOTCROSS_DAT
        end
    end

    zones = DK.torus_zones(DK.amr_deck().blocks, KerrSchildCartesian(DK.TORUS_A), DK.torus_params())

    @testset "weight tables and zone counts" begin
        ws = build_weight_table(zones, 1000.0)
        jt = build_zone_emission_table(zones)
        @test size(jt.jnu) == (MC_N_ESAMP + 1, length(zones)) && jt.dV == [z.dV for z in zones]
        @test jt.jnu[17, 5] === int_jnu_zone(zones[5].n_e, zones[5].theta_e, zones[5].B,
                                             exp(16 * jt.dlnu + jt.lnu_min))
        we = build_weight_table(jt, 1000.0; nthreads = 1)
        @test (we.lnu_min, we.lnu_max, we.dlnu) == (ws.lnu_min, ws.lnu_max, ws.dlnu)
        @test maximum(abs.(exp.(we.lwgt) ./ exp.(ws.lwgt) .- 1)) < 1e-12
        for nt in (2, 3, 8, 64)
            @test build_weight_table(jt, 1000.0; nthreads = nt).lwgt == we.lwgt
        end
        Ns, dn = zone_counts(jt, we)
        for z in (1, 7, length(zones))
            @test (Ns[z], dn[z]) === zone_nsuperphotons(zones[z], we)
        end
        @test zone_counts(jt, we; nthreads = 3) == (Ns, dn)
        @test sum(Ns) ≈ 1000.0 * log(1.0e16 / 1.0e8) rtol = 0.01   # Ns per unit ln ν
    end

    @testset "tabulated emissivities" begin
        zone = zones[length(zones) ÷ 2]
        wgt = build_weight_table([zone], 1000.0)
        gt = build_int_jnu_table(zone, wgt)
        @test length(gt.ljnu) == INT_JNU_REFINE * (length(wgt.lwgt) - 1) + 1
        for m in (0, 1, 200, 400)
            nu = exp(m * wgt.dlnu + wgt.lnu_min)
            @test gt.ljnu[INT_JNU_REFINE * m + 1] ===
                  log(int_jnu_zone(zone.n_e, zone.theta_e, zone.B, nu) + WEIGHT_FLOOR)
        end
        dn = tabulated_dnmax(gt, wgt)
        worst = maximum(1:50_001) do j
            nu = exp(wgt.lnu_min + (wgt.lnu_max - wgt.lnu_min) * j / 50_001)
            int_jnu_at(nu, gt) / (weight_at(nu, wgt) + WEIGHT_FLOOR) / dn
        end
        @test worst <= 1.0 + 1e-12
        jt = build_zone_emission_table(zones)
        we = build_weight_table(jt, 1000.0)
        lint, dnv = zone_int_jnu_tables(jt, we)
        @test lint == log.(jt.jnu .+ WEIGHT_FLOOR)
        @test dnv[3] === tabulated_dnmax(IntJnuTable(jt.lnu_min, jt.dlnu, lint[:, 3]), we)
        @test_throws ErrorException zone_counts(jt, build_weight_table([zone], 1.0; n_esamp = 10))
    end
end
