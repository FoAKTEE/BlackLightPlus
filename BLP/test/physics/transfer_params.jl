# The parameter set of the marches: the nonthermal fit constants computed once by the
# constructors, the population switches, the number type of each sub-record and the
# output type of each mode and lane.
using Test, BLP, StaticArrays, ForwardDiff
using SpecialFunctions: besselk
using BLP.Transfer: StokesI, IQUV, PowerLaw, Kappa, TransferParams, has_powerlaw, has_kappa,
                    output_type
using BLP.Numerics: Units, MSUN
using BLP.Matter: ElectronModel, PlasmaCuts, Bound
const Decks = Main.Decks
dual(x) = ForwardDiff.Dual{Nothing}(x, 1.0)

@testset "transfer parameters" begin
    u = Units(6.2e9 * MSUN, 1.0e24)
    em = ElectronModel()
    cuts = PlasmaCuts(sigma = Bound(-Inf, 1.0))

    @testset "power-law constants" begin
        p, gmin, gmax = 3.5, 25.0, 1.0e7
        pl = PowerLaw(0.3, p, gmin, gmax)
        var_c = gmin^(1 - p) - gmax^(1 - p)
        @test (pl.frac, pl.p, pl.gamma_min, pl.gamma_max) === (0.3, p, gmin, gmax)
        @test pl.rho == (p - 1.0) / var_c
        @test pl.jj_q == -(p + 1.0) / (p + 7.0 / 3.0)
        @test pl.rho_q == -gmin^(2.0 - p) / (p / 2.0 - 1.0)
        @test pl.rho_v == 2.0 * (p + 2.0) / (p + 1.0) * gmin^(-(p + 1.0)) * log(gmin)
        @test pl.jj > 0 && pl.aa > 0 && pl.aa_v > 0
        @test PowerLaw(0.3, 2.0, gmin, gmax).rho_q == -Inf          # the removable pole p = 2
        @test PowerLaw(0.3f0, 3.5, 25, 1.0e7) isa PowerLaw{Float64}
        pl32 = PowerLaw{Float32}(pl)
        @test pl32 isa PowerLaw{Float32} && pl32.jj === Float32(pl.jj)
        @test isbits(pl)
    end

    @testset "kappa constants" begin
        for (k, frac) in ((3.8, 0.6), (4.2, 0.4), (4.75, 0.5))
            kp = Kappa(0.3, k, 0.5)
            @test kp.rho_frac ≈ frac
            @test kp.jj_x_v === kp.jj_x_i
            @test kp.jj_x_i == 3.0 * k^(-1.5)
            @test all(isfinite, ntuple(i -> getfield(kp, i), fieldcount(Kappa)))
        end
        kp = Kappa(0.3, 4.0, 0.5)
        @test kp.rho_v == besselk(0, 2.0) / besselk(2, 2.0)
        @test (kp.frac, kp.kappa, kp.w) === (0.3, 4.0, 0.5)
        # the rotativity fits of the bracketing indices meet at the tabulated kappa = 4
        lo = Kappa(0.3, 4.0 - 1e-12, 0.5); hi = Kappa(0.3, 4.0, 0.5)
        @test lo.rho_q_high_a ≈ hi.rho_q_low_a && lo.rho_v_high_a ≈ hi.rho_v_low_a
        @test Kappa{Float32}(kp).aa_low === Float32(kp.aa_low)
        @test isbits(kp)
    end

    @testset "parameter set" begin
        p0 = TransferParams(u, em, cuts, 230.0e9)
        pp = TransferParams(u, em, cuts, 230.0e9; powerlaw = PowerLaw(0.3, 3.5, 25.0, 1.0e7))
        pk = TransferParams(u, em, cuts, 230.0e9; kappa = Kappa(0.3, 4.0, 0.5))
        pz = TransferParams(u, em, cuts, 230.0e9; powerlaw = PowerLaw(0.0, 3.5, 25.0, 1.0e7))
        @test !has_powerlaw(p0) && !has_kappa(p0)
        @test has_powerlaw(pp) && !has_kappa(pp)
        @test has_kappa(pk) && !has_powerlaw(pk)
        @test !has_powerlaw(pz)                                     # a zero fraction is inactive
        @test (@inferred has_powerlaw(p0)) === false
        @test p0.powerlaw === nothing && p0.kappa === nothing
        for p in (p0, pp, pk, pz)
            @test isbits(p) && Decks.device_isbits(p)
            @test output_type(StokesI(), p) === Float64
            @test output_type(IQUV(), p) === SVector{4,Float64}
        end
        @test output_type(StokesI(), (p0, p0, p0)) === SVector{3,Float64}
    end

    @testset "derivative lanes" begin
        D = typeof(dual(1.0))
        ud = Units(6.2e9 * MSUN, dual(1.0e24))
        pd = TransferParams(ud, em, cuts, 230.0e9; powerlaw = PowerLaw(0.3, 3.5, 25.0, 1.0e7))
        # the mass scale makes the units Dual; the electron model, the populations and
        # the frequency keep their own number type
        @test pd.units isa Units{D} && pd.electrons isa ElectronModel{Float64}
        @test pd.powerlaw isa PowerLaw{Float64} && pd.nu_obs isa Float64
        @test output_type(StokesI(), pd) === D
        er = ElectronModel{D}(rat_high = dual(10.0))
        pr = TransferParams(u, er, cuts, 230.0e9)
        @test output_type(StokesI(), pr) === D
        @test isbits(pd) && isbits(pr)
    end
end
