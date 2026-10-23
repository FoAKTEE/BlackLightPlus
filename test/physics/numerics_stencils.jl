# Interpolation stencils, units, ray-state accessors and the special functions.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using BlackLightPlus.Numerics: lerp8, corner_weights, cr_weights, cr_axis, cr_sample, loglinear_at, primal,
                    Units, RayState, position, momentum, raystate, CL, GNEWT, MP, ME, MSUN,
                    bessel_i0, bessel_i1, bessel_k0, bessel_k1, bessel_k2, harmonic_bridge,
                    hypergeometric_2f1
import SpecialFunctions

@testset "stencils, units, special functions" begin
    @testset "trilinear blend" begin
        f(x, y, z) = 1.5 + 2x - 3y + 0.5z + 0.25x * y - x * y * z
        c = (f(0, 0, 0), f(1, 0, 0), f(0, 1, 0), f(1, 1, 0), f(0, 0, 1), f(1, 0, 1), f(0, 1, 1), f(1, 1, 1))
        for (d1, d2, d3) in ((0.3, 0.7, 0.1), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.25, 0.9))
            @test lerp8(c, d1, d2, d3) ≈ f(d1, d2, d3) atol = 1e-14
        end
        @test lerp8(c, 0.0, 0.0, 0.0) === c[1] && lerp8(c, 1.0, 1.0, 1.0) == c[8]
        v = ntuple(k -> SVector(c[k], -c[k], 2c[k]), 8)
        @test lerp8(v, 0.3, 0.7, 0.1) == SVector(1, -1, 2) .* lerp8(c, 0.3, 0.7, 0.1)
        @test lerp8(Float32.(c), 0.3f0, 0.7f0, 0.1f0) isa Float32
        @test corner_weights(2.25, 10) == (2, 0.25)
        @test corner_weights(-0.4, 10) == (0, 0.0)           # one-sided below the first node
        @test corner_weights(9.5, 10) == (8, 1.0)            # upper corner stays in range
        @test corner_weights(3.5f0, 10) === (3, 0.5f0)
    end

    @testset "Catmull-Rom" begin
        for t in (0.0, 0.3, 0.75, 1.0)
            @test sum(cr_weights(t)) ≈ 1 atol = 1e-15
        end
        @test cr_weights(0.0) == SVector(0.0, 1.0, 0.0, 0.0)
        # exact for a polynomial of degree two per axis on a uniform grid
        p(x, y, z) = 0.5 + x - 0.2x^2 + 0.3y^2 * z^2 - 0.1z^2 + x * y * z
        xs = 0.0:0.5:4.0
        d = zeros(1, 9, 9, 9)
        for k in 1:9, j in 1:9, i in 1:9
            d[1, i, j, k] = p(xs[i], xs[j], xs[k])
        end
        for pt in ((1.3, 2.1, 1.7), (2.0, 1.0, 3.2), (0.6, 3.4, 2.2))
            ax = map(c -> cr_axis(c, 0.0, 0.5, 9), pt)
            @test cr_sample(d, 1, ax[1][2], ax[2][2], ax[3][2], ax[1][1], ax[2][1], ax[3][1]) ≈ p(pt...) rtol = 1e-12
        end
        w, idx = cr_axis(0.1, 0.0, 0.5, 9)
        @test idx == (1, 1, 2, 3)                             # clamped below
        # a Dual coordinate differentiates through the weights only
        D = ForwardDiff.Dual{Nothing}(1.3, 1.0)
        wd, id = cr_axis(D, 0.0, 0.5, 9)
        @test id == cr_axis(1.3, 0.0, 0.5, 9)[2]
        @test ForwardDiff.value.(wd) == cr_axis(1.3, 0.0, 0.5, 9)[1]
        @test primal(ForwardDiff.Dual{:a}(ForwardDiff.Dual{:b}(2.5, 1.0), 1.0)) === 2.5
    end

    @testset "log-linear lookup" begin
        lw = log.([1.0, 4.0, 2.0, 8.0])                       # f at ln x = 0, 1, 2, 3
        @test loglinear_at(exp(1.0), lw, 0.0, 1.0, 4) ≈ 4.0
        @test loglinear_at(exp(1.5), lw, 0.0, 1.0, 4) ≈ sqrt(8.0)
        @test loglinear_at(exp(-3.0), lw, 0.0, 1.0, 4) == 1.0   # clamped to the first node
        @test loglinear_at(exp(9.0), lw, 0.0, 1.0, 4) ≈ 8.0     # clamped to the last node
    end

    @testset "units and ray state" begin
        m = 6.2e9 * MSUN; s = 1.0e24
        u = Units(m, s)
        @test u.L_unit == GNEWT * m / CL^2
        @test u.T_unit == u.L_unit / CL
        @test u.Ne_unit ≈ u.dens_unit / (MP + ME)
        @test u.B_unit ≈ CL * sqrt(4π * u.dens_unit)
        @test u.e_unit ≈ u.dens_unit * CL^2
        @test isbits(u) && u isa Units{Float64}
        ud = Units(m, ForwardDiff.Dual{Nothing}(s, 1.0))
        @test ud isa Units{<:ForwardDiff.Dual}
        @test ForwardDiff.partials(ud.dens_unit)[1] ≈ u.dens_unit / s
        @test ForwardDiff.partials(ud.L_unit)[1] == 0
        y = SVector{8}(1.0:8.0)
        @test y isa RayState{Float64}
        @test position(y) == SVector(1.0, 2.0, 3.0, 4.0) && momentum(y) == SVector(5.0, 6.0, 7.0, 8.0)
        @test raystate(position(y), momentum(y)) === y
        @test raystate(SVector(1.0f0, 2, 3, 4), SVector(5.0, 6, 7, 8)) isa RayState{Float64}
    end

    @testset "special functions" begin
        for x in (0.05, 0.5, 1.9, 2.0, 3.0, 3.75, 10.0, 40.0)
            @test bessel_i0(x) ≈ SpecialFunctions.besseli(0, x) rtol = 2e-7
            @test bessel_i1(x) ≈ SpecialFunctions.besseli(1, x) rtol = 2e-7
            @test bessel_k0(x) ≈ SpecialFunctions.besselk(0, x) rtol = 2e-7
            @test bessel_k1(x) ≈ SpecialFunctions.besselk(1, x) rtol = 2e-7
            @test bessel_k2(x) ≈ SpecialFunctions.besselk(2, x) rtol = 1e-6
        end
        @test bessel_i1(-1.5) == -bessel_i1(1.5)
        @test harmonic_bridge(2.0, 2.0, 1.0) ≈ 1.0
        @test harmonic_bridge(1.0e-3, 1.0e3, 2.0) ≈ 1.0e-3 rtol = 1e-10
        @test harmonic_bridge(1.0e-320, 3.0, 2.0) == 0.0
        @test harmonic_bridge(5.0, 0.0, 2.0) == 0.0
        # ₂F₁(1, 1; 2; z) = -ln(1 - z)/z
        # (ten series terms in x = z/(z - 1): truncation error of order x^11)
        for z in (-0.1, -0.5)
            @test hypergeometric_2f1(1.0, 1.0, 2.0, z) ≈ -log(1 - z) / z rtol = 1e-5
        end
    end
end
