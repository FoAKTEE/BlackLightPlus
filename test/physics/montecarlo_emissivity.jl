# Emission-frame physics of the Monte Carlo: j_ν equals the invariant thermal synchrotron
# emissivity times ν², jmax envelopes the pitch-angle distribution, the zone integral
# matches an independent quadrature, the emission tetrad is orthonormal with e₍₀₎ = u
# and e₍₁₎ along b, and the weight interpolant clamps to the last node at the band top.
using Test, BlackLightPlus
using StaticArrays
using LinearAlgebra: dot
using BlackLightPlus.Transfer: thermal_synchrotron
using BlackLightPlus.MonteCarlo: j_nu, alpha_nu, jmax, int_jnu_zone, emission_tetrad, SMALL_VECTOR,
                      weight_at, table_zone
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

    @testset "int_jnu_zone against an independent quadrature" begin
        for nu in (5.0e10, 2.3e11, 1.0e12)
            iz = int_jnu_zone(NE, TE, BF, nu)
            @test iz > 0.0
            @test iz ≈ ref_int(nu) rtol = 1e-6
            @test iz <= 4π * jmax(NE, TE, BF, nu)
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
