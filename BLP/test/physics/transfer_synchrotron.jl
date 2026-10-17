# Synchrotron fits: the special functions they use against tabulated values, the thermal
# coefficients' signs and Kirchhoff's law, pinned reference values of every population,
# the p = 2 power-law pole, the thermal ρ_Q blend across the cold-plasma trap, and
# derivatives through the fits.
using Test, BLP, StaticArrays, ForwardDiff
using BLP.Numerics: bessel_k0, bessel_k1, bessel_k2, harmonic_bridge, HPL, CL, ME, EE
using BLP.Transfer: thermal_synchrotron, thermal_synchrotron_polarized, power_law_synchrotron,
                    power_law_synchrotron_polarized, kappa_synchrotron, kappa_synchrotron_polarized,
                    PowerLaw, Kappa, K_THETA_E_ZERO, K_THETA_E_BLEND_LO

@testset "synchrotron fits" begin
    @testset "special functions" begin
        @test bessel_k0(1.0) ≈ 0.421024 atol = 1e-6
        @test bessel_k1(1.0) ≈ 0.601907 atol = 1e-6
        @test bessel_k2(1.0) ≈ 1.624839 atol = 1e-6
        @test bessel_k2(1.3) ≈ bessel_k0(1.3) + (2.0 / 1.3) * bessel_k1(1.3)
        @test harmonic_bridge(2.0, 8.0, 3.0) ≈ harmonic_bridge(8.0, 2.0, 3.0)
        @test harmonic_bridge(2.0, 8.0, 3.0) < 2.0
        @test harmonic_bridge(0.0, 5.0, 2.0) == 0.0
    end

    @testset "thermal: signs and Kirchhoff's law" begin
        theta_e = 10.0; B = 30.0; nu = 230.0e9; sin_th = 0.7; n_e = 1.0e6
        j_i, alpha_i = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_th)
        @test j_i > 0 && alpha_i > 0
        b_nu_nu_3 = 2.0 * HPL / (CL * CL) / expm1(HPL * nu / (theta_e * ME * CL * CL))
        @test alpha_i ≈ j_i / b_nu_nu_3 rtol = 1e-12
        ji, jq, jv, ai, aq, av, rq, rv = thermal_synchrotron_polarized(1.0, n_e, theta_e, B, nu, sin_th, 0.5)
        @test ji > 0 && jq < 0
        @test ji ≈ j_i rtol = 1e-12
        @test ai ≈ alpha_i rtol = 1e-12
        @test aq ≈ jq / b_nu_nu_3 rtol = 1e-12
        @test thermal_synchrotron(1.0, 0.0, theta_e, B, nu, sin_th) == (0.0, 0.0)
        @test thermal_synchrotron(0.0, n_e, theta_e, B, nu, sin_th) == (0.0, 0.0)
        j5, a5 = thermal_synchrotron(0.5, n_e, 12.0, B, nu, sin_th)
        j1, a1 = thermal_synchrotron(1.0, n_e, 12.0, B, nu, sin_th)
        @test j5 ≈ 0.5 * j1 rtol = 1e-12
        @test a5 ≈ 0.5 * a1 rtol = 1e-12
    end

    # Reference values of the fits (Marszewski et al. 2021 thermal; precomputed power-law
    # and kappa constants), relative tolerance 1e-14.
    @testset "pinned reference values" begin
        n_e = 1.0e6; bb = 30.0; nu = 2.3e11; s = 0.7; c = 0.5
        th_unpol = [(1.8037425164911681e-59, 7.59034221207488e-21),
                    (1.349755370014262e-41, 0.0005679915530470241),
                    (2.2913473952407873e-39, 0.024105589690606073),
                    (5.2532325823844156e-39, 0.008289812542281)]
        for (te, ref) in zip((0.3, 3.0, 12.0, 80.0), th_unpol)
            @test all(isapprox.(thermal_synchrotron(1.0, n_e, te, bb, nu, s), ref; rtol = 1e-14))
        end
        th_pol = [(0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -3.0844988371583857),
                  (0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0002001910318425074, -3.0844988371583857),
                  (2.855876997400009e-185, -2.8397216436240363e-185, -5.801651797701869e-186,
                   1.8026756426593645e-145, -1.7924781226761843e-145, -3.6620962290846177e-146,
                   -0.0006003702825492607, -3.0844988371583857),
                  (1.8037425164911681e-59, -1.7447065475227951e-59, -1.475648729926927e-60,
                   7.59034221207488e-21, -7.341912514823477e-21, -6.209688324444328e-22,
                   -0.001337544262341298, -1.8261289794457995),
                  (2.2913473952407873e-39, -1.819388036437651e-39, -5.146467354296298e-41,
                   0.024105589690606073, -0.019140450542530975, -0.0005414221809248125,
                   -0.015765164606405622, -0.0278032601912245)]
        for (te, ref) in zip((0.004, 0.007, 0.02, 0.3, 12.0), th_pol)
            @test all(isapprox.(thermal_synchrotron_polarized(1.0, n_e, te, bb, nu, s, -c), ref;
                                rtol = 1e-14))
        end
        pw = Dict(3.5 => (7.46027059890174e-40, -5.7550658905813415e-40, 1.8646722524550488e-41,
                          0.011768761595628839, -0.009516661377446336, 0.00047920198039646785,
                          -0.011224194634741605, 0.014561974150186173),
                  2.0 => (1.132351228122282e-39, -7.83935465623118e-40, 2.1514954867878364e-41,
                          0.00866832925309803, -0.00647441564487873, 0.0002842820138707917,
                          -0.009269626113669174, 0.006354331878638145))
        for (p, ref) in pw
            pl = PowerLaw(0.3, p, 25.0, 1.0e7)
            @test all(isapprox.(power_law_synchrotron_polarized(pl, n_e, bb, nu, s, c), ref; rtol = 1e-14))
            @test all(isapprox.(power_law_synchrotron(pl, n_e, bb, nu, s), (ref[1], ref[4]); rtol = 1e-14))
        end
        kp = Dict(3.6 => (5.819808847945793e-41, -4.2240291989113396e-41, -1.7720310205247482e-42,
                          0.0005095095510620858, -0.0004189503555023219, -1.544511590627413e-5,
                          -0.001124330396693988, -0.25169892111415154),
                  4.2 => (1.8984084331842757e-41, -1.40308643400436e-41, -6.31898092792001e-43,
                          0.00020764836724335929, -0.00017777736666517584, -6.789495224045051e-6,
                          -0.001081901421851477, -0.27999170750502694),
                  5.0 => (4.7003409674580456e-42, -3.542533270867788e-42, -1.7324857428288726e-43,
                          6.57137302282038e-5, -5.9090261752622003e-5, -2.3068905758187354e-6,
                          -0.0009656854151227171, -0.30699469517718214))
        for (k, ref) in kp
            kk = Kappa(0.3, k, 0.5)
            @test all(isapprox.(kappa_synchrotron_polarized(kk, n_e, bb, nu, s, -c), ref; rtol = 1e-14))
            @test all(isapprox.(kappa_synchrotron(kk, n_e, bb, nu, s), (ref[1], ref[4]); rtol = 1e-14))
        end
        # a zero fraction is no emission, of the input type
        @test power_law_synchrotron(PowerLaw(0.0, 3.5, 25.0, 1.0e7), n_e, bb, nu, s) == (0.0, 0.0)
        @test kappa_synchrotron_polarized(Kappa(0.0, 4.0, 0.5), n_e, bb, nu, s, c) == ntuple(_ -> 0.0, 8)
    end

    @testset "power-law ρ_Q at the removable pole p = 2" begin
        rq(p) = power_law_synchrotron_polarized(PowerLaw(0.5, p, 10.0, 1.0e4), 1.0e6, 30.0, 230.0e9, 0.7, 0.5)
        t2 = rq(2.0); tm = rq(2.0 - 1.0e-6); tp = rq(2.0 + 1.0e-6)
        @test all(isfinite, t2) && all(isfinite, tm) && all(isfinite, tp)
        @test t2[7] ≈ 0.5 * (tm[7] + tp[7]) rtol = 1e-4              # the ln X limit
        @test t2[1] ≈ 0.5 * (tm[1] + tp[1]) rtol = 1e-4
    end

    @testset "thermal ρ_Q blend across the cold-plasma trap" begin
        n_e = 1.0e6; B = 30.0; nu = 230.0e9; s_th = 0.7; c_th = 0.5
        rq(te) = thermal_synchrotron_polarized(1.0, n_e, te, B, nu, s_th, c_th)[7]
        @test K_THETA_E_ZERO === 0.01 && K_THETA_E_BLEND_LO === 0.005
        r_lo = rq(0.0099999); r_hi = rq(0.0100001)
        @test abs(r_hi - r_lo) < 1.0e-6
        @test abs(r_hi - r_lo) / abs(r_hi) < 1.0e-3
        @test rq(0.005 - 1.0e-6) == 0.0
        @test abs(rq(0.005 + 1.0e-6)) < abs(r_hi) * 1.0e-3
        @test 0.0 < abs(rq(0.006)) < abs(rq(0.008)) < abs(rq(0.0099)) < abs(r_hi)
        # above K_THETA_E_ZERO the blend weight is one: the unblended fit
        function ref_rhoq(te)
            nu_c = EE * B / (2.0 * π * ME * CL)
            nu_s = 2.0 / 9.0 * nu_c * te * te * s_th
            xx = nu / nu_s; xn = 1.0 / sqrt(xx)
            va = 2.011 * exp(-19.78 * xx^(-0.5175))
            vb = cos(39.89 * xn) * exp(-70.16 * xx^(-0.6))
            vc = 0.011 * exp(-1.69 * xn)
            vd = 0.003135 * xx^(4.0 / 3.0)
            ve = 0.5 * (1.0 + tanh(10.0 * log(0.6648 * xn)))
            f_m = (va - vb - vc) + (vc - vd) * ve
            inv = 1.0 / te; k0 = bessel_k0(inv); k1 = bessel_k1(inv); k2 = k0 + (2.0 / inv) * k1
            coeff_q = -1.0 * n_e * EE * EE * nu_c * nu_c * (s_th * s_th) / (ME * CL * nu * nu)
            return coeff_q * f_m * (k1 / k2 + 6.0 * te)
        end
        for te in (0.0100001, 0.02, 0.1, 10.0)
            @test rq(te) ≈ ref_rhoq(te) rtol = 1.0e-12
        end
    end

    @testset "derivatives through the fits" begin
        n_e = 1.0e6; B = 30.0; nu = 230.0e9; sin_th = 0.7
        ji(θ) = thermal_synchrotron(1.0, n_e, θ, B, nu, sin_th)[1]
        d_ad = ForwardDiff.derivative(ji, 10.0)
        @test d_ad ≈ (ji(10.0 + 1.0e-4) - ji(10.0 - 1.0e-4)) / 2.0e-4 rtol = 1e-6
        @test isfinite(d_ad) && d_ad != 0.0
        D = ForwardDiff.Dual{Nothing}(12.0, 1.0)
        jd, ad = thermal_synchrotron(0.5, n_e, D, B, nu, sin_th)
        @test ForwardDiff.partials(jd, 1) != 0.0 && isfinite(ForwardDiff.partials(ad, 1))
        @test thermal_synchrotron(0.0, n_e, D, B, nu, sin_th) isa NTuple{2,typeof(D)}
        f(m) = power_law_synchrotron(PowerLaw(0.3, 3.5, 25.0, 1.0e7), m * n_e, B, nu, sin_th)[1]
        @test ForwardDiff.derivative(f, 1.0) ≈ f(1.0) rtol = 1e-12  # linear in n_e
    end
end
