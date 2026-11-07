# The Monte Carlo table builders: the hot-cross table holds the quadrature at its nodes
# and the gpumonty file (when present) agrees with it; the streaming zone tables equal the
# serial weight table and the per-zone formula, bitwise for every thread count, and their
# trapezoid counts sum to the target exactly; the emission sites are the owned cells
# above the horizon at stride 1, and at any stride an unbiased quadrature of the stride-1
# sum on a two-level mesh with a steep emissivity.
using Test, BlackLightPlus
using StaticArrays
using Statistics: mean, std
using BlackLightPlus.Spacetime: KerrSchildCartesian, Minkowski, proxy_distance
using BlackLightPlus.MonteCarlo: total_compton_cross_num, int_jnu_zone, zone_nsuperphotons, weight_at,
                      WEIGHT_FLOOR, MINW, MAXW, MINT, MAXT, NW, NT, MC_N_ESAMP
using BlackLightPlus.Data: load_hotcross, build_hotcross_table, build_weight_table, zone_tables,
                emission_sites, AthenaKSnapshot, locate

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
        we, Ns, dn = zone_tables(zones, 1000.0; nthreads = 1)
        @test (we.lnu_min, we.lnu_max, we.dlnu) == (ws.lnu_min, ws.lnu_max, ws.dlnu)
        @test length(we.lwgt) == MC_N_ESAMP + 1
        @test maximum(abs.(exp.(we.lwgt) ./ exp.(ws.lwgt) .- 1)) < 1e-12
        for nt in (2, 3, 8, 64)
            wn, Nn, dnn = zone_tables(zones, 1000.0; nthreads = nt)
            @test wn.lwgt == we.lwgt && Nn == Ns && dnn == dn
        end
        for z in (1, 7, length(zones))
            @test (Ns[z], dn[z]) === zone_nsuperphotons(zones[z], we)
        end
        # trapezoid counts: Σ N_z = Ns' ln(ν_max/ν_min), the target, to rounding, when every
        # node emits (the deck's emission is below the weight floor near 1e16 Hz)
        for n_esamp in (400, 8)
            wt, Nt, _ = zone_tables(zones, 50.0; n_esamp = n_esamp, nu_max = 1.0e14)
            @test length(wt.lwgt) == n_esamp + 1 && all(wt.lwgt .> log(1.0e-60))
            @test sum(Nt) ≈ 50.0 * log(1.0e14 / 1.0e8) rtol = 1e-12
        end
    end

    @testset "emission sites: stride 1 and the deck" begin
        blocks = DK.amr_deck().blocks
        st = KerrSchildCartesian(DK.TORUS_A)
        sites = emission_sites(blocks, st, DK.torus_params())
        @test [(s.x, s.dV) for s in sites] == [(z.x, z.dV) for z in zones]
        # the plasma of an independent serial decode, to the last bits of the contraction
        @test all(isapprox(s.n_e, z.n_e; rtol = 1e-14) && isapprox(s.theta_e, z.theta_e; rtol = 1e-14) &&
                  isapprox(s.B, z.B; rtol = 1e-14) for (s, z) in zip(sites, zones))
        @test emission_sites(blocks, st, DK.torus_params(); seed = 99) == sites
        @test emission_sites(blocks, st, DK.torus_params(); max_zones = 7) == sites[1:7]
        @test_throws ArgumentError emission_sites(blocks, st, DK.torus_params(); stride = 0)
        # the horizon proxy: no site below it on a hole that swallows the deck's centre
        big = KerrSchildCartesian(0.0)
        @test all(s -> proxy_distance(big, s.x)[1] >= 1.0e-3,
                  emission_sites(blocks, big, DK.torus_params()))
    end

    @testset "emission sites: unbiased stride quadrature on a two-level mesh" begin
        # Level 0: 2³ blocks of 4³ cells over [-8, 8]³; level 1: one block of 4³ cells over
        # [0, 4]³ stored after them, so 8 cells of the coarse block above it are not owned.
        nc = 4
        lo = Float64[]; hi = Float64[]; xl = Float64[]; dxs = Float64[]; lv = Int[]; ll = Int[]
        for bk in 0:1, bj in 0:1, bi in 0:1
            l = (-8.0 + 8bi, -8.0 + 8bj, -8.0 + 8bk)
            append!(lo, l); append!(hi, l .+ 8.0); append!(xl, l .+ 1.0); append!(dxs, (2.0, 2.0, 2.0))
            push!(lv, 0); append!(ll, (bi, bj, bk))
        end
        append!(lo, (0.0, 0.0, 0.0)); append!(hi, (4.0, 4.0, 4.0)); append!(xl, (0.5, 0.5, 0.5))
        append!(dxs, (1.0, 1.0, 1.0)); push!(lv, 1); append!(ll, (2, 2, 2))
        nmb = 9
        data = zeros(Float32, 8, nc, nc, nc, nmb)
        data[1, :, :, :, :] .= 1.0f0; data[5, :, :, :, :] .= 0.1f0; data[6, :, :, :, :] .= 0.3f0
        R = m -> reshape(m, 3, nmb)
        snap = AthenaKSnapshot(data, R(lo), R(hi), R(xl), R(dxs), lv, reshape(ll, 3, nmb))
        b = snap.blocks
        params = DK.torus_params()
        Lu3 = params.units.L_unit^3
        f(x) = (x[2]^2 + x[3]^2 + x[4]^2 + 0.5)^-3                 # an r⁻⁶ emissivity
        quad(sites) = sum(s -> f(s.x) * s.dV, sites) / Lu3
        one = emission_sites(b, Minkowski(), params)
        @test length(one) == 8 * 64 - 8 + 64                        # every owned cell
        @test all(s -> locate(b, s.x[2], s.x[3], s.x[4]) > 0, one)
        S1 = quad(one)
        for stride in (2, 3)
            sites = emission_sites(b, Minkowski(), params; stride = stride, seed = 1)
            # every zone sits on an owned cell; its volume is its group's owned cells
            @test all(sites) do s
                m = locate(b, s.x[2], s.x[3], s.x[4])
                dV = b.dx[1, m]^3 * Lu3
                n_own = s.dV / dV
                m > 0 && isinteger(round(n_own; digits = 9)) && 1 <= n_own <= stride^3
            end
            est = [quad(emission_sites(b, Minkowski(), params; stride = stride, seed = k)) for k in 1:3000]
            @test abs(mean(est) - S1) < 4.0 * std(est) / sqrt(length(est))
            @test std(est) > 0.0
        end
    end
end
