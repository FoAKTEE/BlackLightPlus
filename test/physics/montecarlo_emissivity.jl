# Emission-frame physics of the Monte Carlo: j_ν equals the invariant thermal synchrotron
# emissivity times ν², jmax envelopes the pitch-angle distribution, the angle integral
# F(K) of the table matches the Simpson rule over its whole range and through it the zone
# integral matches independent quadratures, the emission tetrad is orthonormal with
# e₍₀₎ = u and e₍₁₎ along b, for a field along any coordinate axis too, and the weight
# interpolant clamps to the last node at the band top.
using Test, BlackLightPlus
using StaticArrays
using LinearAlgebra: dot, Diagonal
using BlackLightPlus.Spacetime: KerrSchildCartesian, metric_and_inverse
using BlackLightPlus.Transfer: thermal_synchrotron
using BlackLightPlus.MonteCarlo: j_nu, alpha_nu, jmax, int_jnu_zone, int_jnu, int_jnu_simpson,
                      EMISSIVITY_TABLE, build_emissivity_table, synchrotron_F, F_TABLE_KMIN,
                      F_TABLE_KMAX, F_TABLE_THETA, emission_tetrad, SMALL_VECTOR, TETRAD_MIN_RESIDUAL, weight_at,
                      table_zone
using BlackLightPlus.Numerics: EE, ME, CL, SQRT2
using BlackLightPlus.Data: build_weight_table
using BlackLightPlus.Matter: UniformSphere

const NE = 1.0e6
const TE = 40.0
const BF = 30.0

# Independent trapezoid over the half sphere: ∫ j dΩ = 4π ∫₀^{π/2} j(sin θ) sin θ dθ.
function ref_int(nu; n = 20000)
    h = (0.5π) / n
    acc = 0.0
    for i in 0:n
        th = i * h
        w = (i == 0 || i == n) ? 0.5 : 1.0
        acc += w * j_nu(NE, TE, BF, nu, sin(th)) * sin(th)
    end
    return 4π * acc * h
end

@testset "Monte Carlo emissivity" begin
    @testset "j_nu and alpha_nu from the invariant fits" begin
        for nu in (3.0e10, 1.0e11, 2.3e11, 1.0e12), s in (0.2, 0.5, 0.9, 1.0)
            j_inv, a_inv = thermal_synchrotron(1.0, NE, TE, BF, nu, s)
            @test j_nu(NE, TE, BF, nu, s) === j_inv * nu^2
            @test alpha_nu(NE, TE, BF, nu, s) === a_inv / nu
            @test j_nu(NE, TE, BF, nu, s) > 0.0
        end
    end

    @testset "jmax envelopes the pitch-angle distribution" begin
        for nu in (5.0e10, 2.3e11, 1.0e12)
            jm = jmax(NE, TE, BF, nu)
            @test jm === j_nu(NE, TE, BF, nu, 1.0)
            for s in (0.05, 0.3, 0.6, 0.85, 1.0)
                @test j_nu(NE, TE, BF, nu, s) <= jm * (1 + 1e-12)
            end
        end
    end

    @testset "F(K) table against the Simpson rule" begin
        t = EMISSIVITY_TABLE
        @test t.lnF == build_emissivity_table().lnF            # the compiled table is the build
        # F(K) = ∫₀^{π/2} sin²θ G(K/sin θ) dθ by the 512- and 8192-interval Simpson rules
        G(X) = (sqrt(X) + 2.0^(11 / 12) * X^(1 / 6))^2 * exp(-cbrt(X))
        function F_simpson(K, n)
            h = 0.5π / n
            return sum((i == n ? 1.0 : (isodd(i) ? 4.0 : 2.0)) * sin(i * h)^2 * G(K / sin(i * h))
                       for i in 1:n) * h / 3
        end
        worst512 = 0.0; worst8192 = 0.0
        for k in 0:400
            K = exp(log(1.0e-35) + (log(F_TABLE_KMAX) - log(1.0e-35)) * (k + 0.37) / 401)
            F = synchrotron_F(t, K)
            worst512 = max(worst512, abs(F / F_simpson(K, 512) - 1))
            worst8192 = max(worst8192, abs(F / F_simpson(K, 8192) - 1))
        end
        @test worst512 < 1.0e-8
        @test worst8192 < 2.0e-10
        @test synchrotron_F(t, 2 * F_TABLE_KMAX) == 0.0 && synchrotron_F(t, 0.0) == 0.0
        @test synchrotron_F(t, -1.0) == 0.0 && synchrotron_F(t, NaN) == 0.0
        # K → 0: F ∝ K^{1/3}
        @test synchrotron_F(t, 1.0e-45) / synchrotron_F(t, 1.0e-42) ≈ 0.1 rtol = 1e-8
        # the prefactor: ∫ j dΩ = 4π n_e e² ν_c/c (√2π/27) F(K)
        nu_c = EE * BF / (2π * ME * CL)
        K = 2.3e11 / (2 / 9 * nu_c * TE^2)
        @test int_jnu(t, NE, TE, BF, 2.3e11) ≈ 4π * NE * EE^2 * nu_c / CL * (SQRT2 * π / 27) *
                                               F_simpson(K, 8192) rtol = 1e-10
        @test int_jnu(t, 0.0, TE, BF, 1e11) == int_jnu(t, NE, -1.0, BF, 1e11) ==
              int_jnu(t, NE, TE, 0.0, 1e11) == int_jnu(t, NE, TE, BF, 0.0) == 0.0
    end

    @testset "int_jnu_zone against independent quadratures" begin
        for nu in (5.0e10, 2.3e11, 1.0e12)
            iz = int_jnu_zone(NE, TE, BF, nu)
            @test iz > 0.0
            @test iz ≈ ref_int(nu) rtol = 1e-6
            @test iz ≈ int_jnu_simpson(NE, TE, BF, nu) rtol = 1e-8
            @test iz <= 4π * jmax(NE, TE, BF, nu)
        end
        for (ne, te, b, nu) in ((1e5, 0.05, 3.0, 1e9), (1e13, 100.0, 1.0, 1e16), (1e7, 300.0, 1e3, 1e8),
                                (1e4, 1.0, 0.1, 3e12))
            @test int_jnu_zone(ne, te, b, nu) ≈ int_jnu_simpson(ne, te, b, nu) rtol = 1e-8
        end
        @test j_nu(NE, TE, BF, 1.0e15, 1.0) < j_nu(NE, TE, BF, 1.0e11, 1.0)
    end

    @testset "emission tetrad: orthonormal, e(0) = u, e(1) along b" begin
        g = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
        γ = 1.5; β = sqrt(1 - 1 / γ^2)
        ucon = SVector(γ, γ * β, 0.0, 0.0)
        bcon = SVector(0.0, 0.3, 1.0, 0.5)
        E = emission_tetrad(ucon, g * ucon, bcon, g, g)
        η = SVector(-1.0, 1.0, 1.0, 1.0)
        row(a) = SVector(E[a, 1], E[a, 2], E[a, 3], E[a, 4])
        for a in 1:4, b in 1:4
            @test dot(row(a), g * row(b)) ≈ (a == b ? η[a] : 0.0) atol = 1e-12
        end
        @test row(1) ≈ ucon atol = 1e-12
        # e(1) lies in the plane of u and b
        e1 = row(2)
        bperp = bcon + dot(bcon, g * ucon) * ucon
        @test e1 ≈ bperp / sqrt(dot(bperp, g * bperp)) atol = 1e-12
        # a null field falls back to the x¹ axis
        Ez = emission_tetrad(SVector(1.0, 0.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0, 0.0),
                             SVector(0.0, 0.0, 0.0, 0.0), g, g)
        @test SVector(Ez[2, 1], Ez[2, 2], Ez[2, 3], Ez[2, 4]) == SVector(0.0, 1.0, 0.0, 0.0)
        @test SMALL_VECTOR == 1.0e-30
        # a field along any coordinate axis, at rest and in a moving fluid on Kerr: the frame
        # stays orthonormal (along x³ the x³ residual vanishes and another axis takes its place)
        st = KerrSchildCartesian(0.9)
        x = SVector(0.0, 5.0, -2.0, 1.5)
        gk, _ = metric_and_inverse(st, x)
        for v in (SVector(0.0, 0.0, 0.0), SVector(0.0, 0.3, 0.0), SVector(0.2, -0.1, 0.05))
            u = SVector(1.0, v...)
            u = u / sqrt(-dot(u, gk * u))
            for b in (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0),
                      SVector(0.0, 1.0e-9, 1.0))
                bc = SVector(0.0, b...)
                bc = bc + dot(bc, gk * u) * u                    # b·u = 0
                Ek = emission_tetrad(u, gk * u, bc, gk, gk)
                @test all(isfinite, Ek)
                @test maximum(abs.(Ek * gk * Ek' - Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))) < 1e-12
            end
        end
        @test TETRAD_MIN_RESIDUAL == 1.0e-2
    end

    @testset "weight_at clamps to the last node at the band top" begin
        zone = table_zone(UniformSphere())
        wgt = build_weight_table([zone], 1000.0; n_esamp = 10, nu_min = 1.0e8, nu_max = 1.0e16)
        @test !isapprox(exp(wgt.lwgt[end]), exp(wgt.lwgt[end - 1]); rtol = 1e-3)
        @test weight_at(1.0e16, wgt) ≈ exp(wgt.lwgt[end]) rtol = 1e-12
        @test weight_at(2.0e16, wgt) ≈ exp(wgt.lwgt[end]) rtol = 1e-12
        @test weight_at(1.0e16, wgt) ≈ weight_at(1.0e16 * (1 - 1e-12), wgt) rtol = 1e-6
        @test weight_at(1.0e7, wgt) ≈ exp(wgt.lwgt[1]) rtol = 1e-12
    end
end
