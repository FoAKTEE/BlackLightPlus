# Segment operators of constant-coefficient transfer: the physical limits of the
# polarized coupling (Landi Degl'Innocenti 1985; ipole, blacklight StokesCoupleNoSplit),
# agreement with dense matrix exponentials, the degenerate and saturated branches,
# Stokes admissibility and the order independence it buys, the unpolarized operator,
# and derivatives through the coupling.
using Test, BLP, StaticArrays, LinearAlgebra, ForwardDiff
using BLP.Transfer: StokesI, IQUV, segment, couple, stokes_admissible, POL_DELTA_TAU_MAX,
                    thermal_synchrotron_polarized

const S0 = SVector(2.0, 0.5, 0.3, 0.1)
const ZERO4 = SVector(0.0, 0.0, 0.0, 0.0)
stokes_couple(S, j, α, ρ, ds) = couple(IQUV(), S, j, α, ρ, ds)
mueller(α, ρ) = @SMatrix [α[1] α[2] 0.0 α[4];
                          α[2] α[1] ρ[4] 0.0;
                          0.0 -ρ[4] α[1] ρ[2];
                          α[4] 0.0 -ρ[2] α[1]]

# invariant coefficients of one thermal sample (U components zero)
function thermal(n_e, theta_e, B, nu, sin_th)
    c = thermal_synchrotron_polarized(1.0, n_e, theta_e, B, nu, sin_th, sqrt(1 - sin_th^2))
    return SVector(c[1], c[2], 0.0, c[3]), SVector(c[4], c[5], 0.0, c[6]), SVector(0.0, c[7], 0.0, c[8])
end
forward(segs) = foldl((S, s) -> stokes_couple(S, s...), segs; init = ZERO4)
function reverse_chain(segs)
    S = ZERO4; G = one(SMatrix{4,4,Float64})
    for s in Iterators.reverse(segs)
        M, b = segment(IQUV(), s...)
        S = stokes_admissible(S + G * b)
        G = G * M
    end
    return S
end

@testset "segment operators" begin
    @testset "identity and pure emission" begin
        @test POL_DELTA_TAU_MAX === 100.0
        @test stokes_couple(S0, ZERO4, ZERO4, ZERO4, 0.7) ≈ S0
        j = SVector(1.0, -0.2, 0.0, 0.05)
        @test stokes_couple(S0, j, ZERO4, ZERO4, 0.7) ≈ S0 + 0.7 * j
    end

    @testset "total absorption" begin
        j = SVector(1.0, -0.2, 0.0, 0.05); α = SVector(2.0, 0.0, 0.0, 0.0); ds = 0.3
        @test stokes_couple(S0, j, α, ZERO4, ds) ≈ exp(-α[1] * ds) .* (S0 .+ j ./ α[1] .* expm1(α[1] * ds)) rtol = 1e-12
        thick = stokes_couple(S0, j, α, ZERO4, 1e5)
        @test thick ≈ j ./ α[1] rtol = 1e-10
        @test stokes_couple(10 .* S0, j, α, ZERO4, 1e5) ≈ thick rtol = 1e-10
    end

    @testset "pure Faraday rotation preserves |P|" begin
        ds = 0.4
        out = stokes_couple(S0, ZERO4, ZERO4, SVector(0.0, 0.0, 0.0, 1.3), ds)
        θ = 1.3 * ds
        @test out[1] ≈ S0[1] && out[4] ≈ S0[4]
        @test out[2] ≈ S0[2] * cos(θ) - S0[3] * sin(θ) rtol = 1e-12
        @test out[3] ≈ S0[3] * cos(θ) + S0[2] * sin(θ) rtol = 1e-12
        out2 = stokes_couple(S0, ZERO4, ZERO4, SVector(0.0, 0.9, 0.0, 0.0), ds)
        @test out2[1] ≈ S0[1] && out2[2] ≈ S0[2]
        @test hypot(out2[3], out2[4]) ≈ hypot(S0[3], S0[4]) rtol = 1e-12
        out3 = stokes_couple(S0, ZERO4, ZERO4, SVector(0.0, 0.7, 0.0, -1.1), ds)
        @test norm(out3[2:4]) ≈ norm(S0[2:4]) rtol = 1e-12
    end

    @testset "against the matrix exponential" begin
        j = SVector(1.0, -0.3, 0.0, 0.02)
        α = SVector(2.0, -0.8, 0.0, 0.1); ds = 0.5
        K = mueller(α, ZERO4); Sinf = K \ j
        @test stokes_couple(S0, j, α, ZERO4, ds) ≈ SVector{4}(exp(-Matrix(K) * ds) * (S0 - Sinf) + Sinf) rtol = 1e-9
        α = SVector(2.0, -0.6, 0.0, 0.15); ρ = SVector(0.0, 0.8, 0.0, -1.2); ds = 0.37
        K = mueller(α, ρ); Sinf = K \ j
        @test stokes_couple(S0, j, α, ρ, ds) ≈ SVector{4}(exp(-Matrix(K) * ds) * (S0 - Sinf) + Sinf) rtol = 1e-8
        thickA = stokes_couple(S0, j, α, ρ, 1e4); thickB = stokes_couple(5 .* S0, j, α, ρ, 1e4)
        @test thickA ≈ thickB rtol = 1e-8
        @test thickA ≈ SVector{4}(K \ j) rtol = 1e-6
        M, b = segment(IQUV(), j, α, ρ, ds)
        @test M ≈ exp(-Matrix(K) * ds) rtol = 1e-10
        # saturated above dtau_max: M = 0, b = K⁻¹ j
        M, b = segment(IQUV(), j, α, ρ, 30.0; dtau_max = 100.0)
        Ms, bs = segment(IQUV(), j, α, ρ, 30.0; dtau_max = 0.1)
        @test norm(M) > 0 && Ms == zero(Ms)
        @test bs ≈ SVector{4}(K \ j) rtol = 1e-8
    end

    @testset "affine chaining" begin
        j = SVector(1.0, -0.3, 0.0, 0.02); α = SVector(2.0, -0.6, 0.0, 0.15); ρ = SVector(0.0, 0.8, 0.0, -1.2)
        M1, b1 = segment(IQUV(), j, α, ρ, 0.37); M2, b2 = segment(IQUV(), 0.5j, α, 2ρ, 0.21)
        S = SVector(0.4, 0.1, -0.05, 0.02)
        @test M2 * (M1 * S + b1) + b2 ≈ (M2 * M1) * S + (M2 * b1 + b2) rtol = 1e-14
    end

    @testset "degenerate guard and admissibility" begin
        out = stokes_couple(S0, SVector(0.1, 0.05, 0.0, 0.0), SVector(1.0, 0.7, 0.0, 0.0),
                            SVector(0.0, 0.7, 0.0, 0.0), 0.2)
        @test all(isfinite, out)
        weird = stokes_couple(SVector(0.01, 0.5, 0.5, 0.5), ZERO4, SVector(1.0, 0.99, 0.0, 0.0), ZERO4, 3.0)
        @test weird[1] >= 0 && norm(weird[2:4]) <= weird[1] * (1 + 1e-12)
        ok = SVector(1.0, 0.3, -0.2, 0.1)
        @test stokes_admissible(ok) == ok
        c = stokes_admissible(SVector(1.0, 2.0, 0.0, 0.0))
        @test c[1] == 1.0
        @test c[2] ≈ 1.0 rtol = 1e-12
        @test stokes_admissible(SVector(-0.5, 0.2, 0.1, 0.0)) == ZERO4
        for S in (SVector(0.5, 0.6, 0.3, -0.2), S0, SVector(0.01, 0.5, 0.5, 0.5))
            @test stokes_admissible(S) == stokes_couple(S, ZERO4, ZERO4, ZERO4, 0.0)
        end
    end

    @testset "α_P > α_I is rescaled onto the cone" begin
        S = SVector(2.0, 0.3, 0.0, 0.1); j = SVector(0.5, 0.0, 0.0, 0.0); α = SVector(1.0, 1.5, 0.0, 0.0)
        ceiling = max(S[1], j[1] / α[1])
        for ds in (1.0, 5.0, 20.0, 100.0)
            out = stokes_couple(S, j, α, ZERO4, ds)
            @test all(isfinite, out) && 0.0 <= out[1] <= ceiling * (1 + 1e-9)
            @test norm(out[2:4]) <= out[1] * (1 + 1e-9)
        end
        @test stokes_couple(S, j, α, ZERO4, 1e4)[1] ≈ j[1] / α[1] rtol = 1e-6
    end

    @testset "emission at small pitch angle is admissible" begin
        j, α, ρ = thermal(1.2e-5, 0.918, 100.0, 1.0e9, 0.02)
        @test hypot(j[2], j[4]) > j[1]                           # the fit leaves the cone
        for ds in (4.8e5, 5.8e6, 4.8e7)
            M, b = segment(IQUV(), j, α, ρ, ds)
            @test norm(b[2:4]) <= b[1] * (1 + 1e-12)
        end
        screen = thermal(1.2e-5, 0.918, 100.0, 1.0e9, 0.02)
        slab = thermal(6.0e-8, 0.918, 100.0, 1.0e9, 1.0)
        for front in (true, false)
            segs = front ? vcat(fill((slab..., 1.9e8 / 8), 8), fill((screen..., 4.8e7 / 8), 8)) :
                           vcat(fill((screen..., 4.8e7 / 8), 8), fill((slab..., 1.9e8 / 8), 8))
            Sf = forward(segs); Sr = reverse_chain(segs)
            @test Sr ≈ Sf rtol = 1e-6
            @test norm(Sf[2:4]) <= Sf[1] * (1 + 1e-12)
        end
    end

    @testset "unpolarized operator" begin
        A, B = segment(StokesI(), 2.0, 0.5, 0.3)
        @test A ≈ exp(-0.15) rtol = 1e-14
        @test B ≈ 4.0 * (1 - exp(-0.15)) rtol = 1e-14
        @test segment(StokesI(), 2.0, 0.5, 1000.0) == (0.0, 4.0)               # saturated
        @test segment(StokesI(), 2.0, 0.0, 0.3) == (1.0, 0.6)                  # emission only
        @test segment(StokesI(), 2.0, -1.0, 0.3) == (1.0, 0.6)
        @test segment(StokesI(), 2.0, 0.5, 30.0; dtau_max = 10.0) == (0.0, 4.0)
        @test couple(StokesI(), 1.5, 2.0, 0.5, 0.3) ≈ A * 1.5 + B
        D = ForwardDiff.Dual{Nothing}(2.0, 1.0)
        Ad, Bd = segment(StokesI(), D, 0.5, 0.3)
        @test ForwardDiff.partials(Bd, 1) ≈ B / 2.0 && Ad isa Real
    end

    @testset "derivatives through the coupling" begin
        j = SVector(1.0, -0.3, 0.0, 0.02); α = SVector(2.0, -0.6, 0.0, 0.15); ρ = SVector(0.0, 0.8, 0.0, -1.2)
        f = x -> stokes_couple(S0, j .* x, α, ρ, 0.37)[1]
        d_ad = ForwardDiff.derivative(f, 1.0)
        @test d_ad ≈ (f(1.0 + 1e-6) - f(1.0 - 1e-6)) / 2e-6 rtol = 1e-6
        @test isfinite(d_ad) && d_ad != 0.0
    end
end
