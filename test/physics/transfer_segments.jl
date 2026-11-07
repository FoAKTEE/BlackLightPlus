# Segment operators of constant-coefficient transfer: the physical limits of the
# polarized coupling (Landi Degl'Innocenti 1985; ipole, blacklight StokesCoupleNoSplit),
# agreement with dense matrix exponentials, the exact limits (dichroism-cone boundary,
# nilpotent coupling, rotation without absorption, thick segments with a shallow mode,
# small absorption down to the underflow range, any coefficient scale) against 256-bit
# solutions, the unprojected solver on the problems of Prather et al.
# (2023), Stokes admissibility and the order independence it buys, the unpolarized
# operator, and derivatives through the coupling.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra, ForwardDiff, Random
using BlackLightPlus.Transfer: StokesI, IQUV, segment, couple, stokes_admissible, POL_DELTA_TAU_MAX,
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

# exp(A) in BigFloat by scaling and squaring of the Taylor series
function bigexp(A::Matrix{BigFloat})
    sq = max(0, ceil(Int, log2(Float64(maximum(sum(abs.(A), dims = 2))))) + 2)
    B = A ./ BigFloat(2)^sq
    E = Matrix{BigFloat}(I, size(A)...); t = copy(E)
    for k in 1:300
        t = t * B ./ k; E += t
        maximum(abs.(t)) < BigFloat(10)^-75 && break
    end
    for _ in 1:sq
        E = E * E
    end
    return E
end
# exact (M, b) = (e^{-K ds}, ∫₀^ds e^{-K s} ds j) from exp(ds [-K j; 0 0]) at 256 bits
function exact_segment(j, α, ρ, ds)
    setprecision(BigFloat, 256) do
        a = BigFloat.(collect(α)); r = BigFloat.(collect(ρ))
        K = [a[1] a[2] a[3] a[4]; a[2] a[1] r[4] -r[3]; a[3] -r[4] a[1] r[2]; a[4] r[3] -r[2] a[1]]
        A = zeros(BigFloat, 5, 5); A[1:4, 1:4] = -K; A[1:4, 5] = BigFloat.(collect(j))
        E = bigexp(A .* BigFloat(ds))
        SMatrix{4,4}(Float64.(E[1:4, 1:4])), SVector{4}(Float64.(E[1:4, 5]))
    end
end
close_to(x, y; rtol) = maximum(abs.(x .- y)) <= rtol * max(maximum(abs.(y)), 1e-300)
# S(L) of n equal unprojected segments, S(0) = 0, without the admissibility clamp
function chain_unprojected(j, α, ρ, L, n)
    M, b = segment(IQUV(), j, α, ρ, L / n; project = false)
    S = ZERO4
    for _ in 1:n
        S = M * S + b
    end
    return S
end
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

    @testset "α_P > α_I is projected onto the cone boundary" begin
        # the mode polarized against α̂ is unabsorbed at |α_P| = α_I: I - Q carries its
        # initial value and gathers j_I - j_Q linearly, I + Q relaxes to (j_I + j_Q)/2α_I
        S = SVector(2.0, 0.3, 0.0, 0.1); j = SVector(0.5, 0.0, 0.0, 0.0); α = SVector(1.0, 1.5, 0.0, 0.0)
        αb = SVector(1.0, 1.0, 0.0, 0.0)
        for ds in (1.0, 5.0, 20.0, 100.0, 1e4)
            M, b = segment(IQUV(), j, α, ZERO4, ds)
            @test (M, b) == segment(IQUV(), j, αb, ZERO4, ds; project = false)
            out = stokes_couple(S, j, α, ZERO4, ds)
            @test all(isfinite, out) && norm(out[2:4]) <= out[1] * (1 + 1e-9)
            @test out[1] - out[2] ≈ (S[1] - S[2]) + j[1] * ds rtol = 1e-12
            @test isapprox(out[1] + out[2], (S[1] + S[2]) * exp(-2ds) + j[1] / 2 * (1 - exp(-2ds));
                           atol = 1e-12 * out[1])
        end
    end

    @testset "exact limits against the 256-bit solution" begin
        z = ZERO4
        cases = [  # (j, α, ρ): name
            (SVector(2.0, 1.0, 0.0, 0.0), SVector(1.0, 1.0, 0.0, 0.0), z),          # |α_P| = α_I, ρ = 0
            (SVector(2.0, 0.3, 0.0, -0.5), SVector(1.0, 0.6, 0.0, -0.8), z),        # same, along Q and V
            (SVector(1.0, 0.5, 0.0, 0.2), SVector(1.0, 1.0, 0.0, 0.0), SVector(0.0, 3.0, 0.0, 0.0)),  # α ∥ ρ
            (SVector(1.0, 0.5, 0.0, 0.2), SVector(1.0, 0.5, 0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.5)),  # nilpotent N
            (SVector(0.2, 0.1, 0.1, 0.1), z, SVector(0.0, 10.0, 0.0, -4.0)),        # rotation, no absorption
            (SVector(0.2, 0.1, 0.1, 0.1), z, SVector(0.0, 1e-3, 0.0, -4e-4)),       # slow rotation
            (SVector(1.0, 0.3, 0.0, 0.0), SVector(0.0, 0.5, 0.0, 0.0), z),          # α_I = 0 < |α_P| (unprojected)
            (SVector(2.0, 1.0, 0.0, 0.0), SVector(1.0, 1.2, 0.0, 0.0), z),          # |α_P| > α_I (unprojected)
        ]
        for (j, α, ρ) in cases, ds in (1e-3, 0.3, 3.0, 30.0)
            M, b = segment(IQUV(), j, α, ρ, ds; project = false)
            Me, be = exact_segment(j, α, ρ, ds)
            @test close_to(b, be; rtol = 1e-12)
            @test maximum(abs.(M .- Me)) <= 1e-12 * max(maximum(abs.(Me)), 1.0)
        end
        # the boundary is never saturated: the unabsorbed mode keeps its emission
        j = SVector(2.0, 1.0, 0.0, 0.0); α = SVector(1.0, 1.0, 0.0, 0.0)
        M, b = segment(IQUV(), j, α, z, 300.0)
        Me, be = exact_segment(j, α, z, 300.0)
        @test close_to(b, be; rtol = 1e-14)
        @test b[1] - b[2] ≈ 300.0 rtol = 1e-14
        @test M[1, 1] ≈ 0.5 && M[1, 2] ≈ -0.5
        # a thick segment with a shallow slow mode is integrated, not saturated
        α = SVector(1.0, 0.999, 0.0, 0.0); ρ = SVector(0.0, 0.0, 0.0, 0.02)
        M, b = segment(IQUV(), j, α, ρ, 200.0)
        Me, be = exact_segment(j, α, ρ, 200.0)
        @test close_to(b, be; rtol = 1e-12)
        @test maximum(abs.(M .- Me)) <= 1e-12 * maximum(abs.(Me))
        # rotation without absorption: one segment equals any chain of segments
        j = SVector(0.2, 0.1, 0.1, 0.1); ρ = SVector(0.0, 10.0, 0.0, -4.0)
        @test chain_unprojected(j, z, ρ, 3.0, 1) ≈ chain_unprojected(j, z, ρ, 3.0, 300) rtol = 1e-12
    end

    @testset "small absorption: the regular limit, to rounding at every scale" begin
        # the absorptivity of a power-law or kappa fit far out: α_I² underflows, and the
        # closed form's 1/(α_I² − λ₁²) is 0/0 (the GRMHD comparison's reproducer)
        j = SVector(1.29e-198, -1.28e-198, 0.0, 1.03e-201)
        α = SVector(1.98e-160, -1.97e-160, 0.0, 1.58e-163)
        ρ = SVector(0.0, -1.46e-11, 0.0, 1.54e-6); ds = 0.394
        M, b = segment(IQUV(), j, α, ρ, ds)
        Me, be = exact_segment(j, α, ρ, ds)
        @test all(isfinite, M) && all(isfinite, b)
        @test close_to(b, be; rtol = 1e-14) && maximum(abs.(M .- Me)) <= 1e-14
        @test all(isfinite, stokes_couple(SVector(1.0, 0.1, 0.0, 0.0), j, α, ρ, ds))
        # α → 0 at fixed ρ: through the ladder to the rotation segment, every rung exact
        j1 = SVector(1.29, -1.28, 0.0, 0.0103); a1 = SVector(1.98, -1.97, 0.0, 0.00158)
        Mr, br = segment(IQUV(), j1, ZERO4, ρ, ds)
        for k in (4, 8, 12, 16, 50, 154, 156, 160, 300)
            M, b = segment(IQUV(), j1, a1 .* 10.0^(-k), ρ, ds)
            Me, be = exact_segment(j1, a1 .* 10.0^(-k), ρ, ds)
            @test close_to(b, be; rtol = 1e-14) && maximum(abs.(M .- Me)) <= 1e-14
            k >= 50 && @test close_to(b, br; rtol = 1e-14) && maximum(abs.(M .- Mr)) <= 1e-14
        end
        # thin physical segments: b exact where the closed form kept ε/(α_I ds) of it
        rng = Xoshiro(20261107)
        for dtau in (1e-6, 1e-10, 1e-14), _ in 1:10
            ap = 0.9 * rand(rng); th = 2π * rand(rng); r = 10.0^(3 * rand(rng)); ph = 2π * rand(rng)
            α = SVector(1.0, ap * cos(th), 0.0, ap * sin(th))
            ρ = SVector(0.0, r * cos(ph), 0.0, r * sin(ph))
            j = SVector(1.0, 0.5 * cos(ph + 1), 0.0, 0.5 * sin(ph + 1))
            M, b = segment(IQUV(), j, α, ρ, dtau)
            Me, be = exact_segment(j, α, ρ, dtau)
            @test close_to(b, be; rtol = 1e-14) && maximum(abs.(M .- Me)) <= 1e-14
        end
        # every coefficient times c and ds over c: the same operator at any scale
        j = SVector(1.0, 0.3, 0.0, -0.2); α = SVector(1.0, 0.5, 0.0, 0.3); ρ = SVector(0.0, 2.0, 0.0, -1.0)
        M1, b1 = segment(IQUV(), j, α, ρ, 0.7)
        for c in (1e-150, 1e-160, 1e-200, 1e-300, 1e150)
            Mc, bc = segment(IQUV(), j .* c, α .* c, ρ .* c, 0.7 / c)
            @test maximum(abs.(Mc .- M1)) <= 1e-14 && close_to(bc, b1; rtol = 1e-14)
        end
        Mc, bc = segment(IQUV(), j .* 1e-160, ZERO4, ρ .* 1e-160, 0.7e160)   # rotation alone
        @test all(isfinite, Mc) && close_to(bc, segment(IQUV(), j, ZERO4, ρ, 0.7)[2]; rtol = 1e-14)
        # Faraday rotation 1e10 times the absorption: λ₁ = |α·ρ|/ρ_P = 0.0702 to rounding (as
        # the difference of two terms of 3.6e19 it is lost, and the slow mode decays at α_I:
        # M₁₁ = 2.1e-9 for 4.45e-9 at ds = 20); the rotation angle ρ_P ds itself carries
        # ε ρ_P ds, so the decaying I row is compared
        α = SVector(1.0, 0.6, 0.0, 0.3); ρ = SVector(0.0, 3e9, 0.0, -8e9)
        for ds in (1.0, 20.0)
            M, b = segment(IQUV(), j, α, ρ, ds)
            Me, be = exact_segment(j, α, ρ, ds)
            @test M[1, 1] ≈ Me[1, 1] rtol = 1e-12
            @test b[1] ≈ be[1] rtol = 1e-12
            @test maximum(abs, M) <= exp(-(1.0 - 0.0706) * ds) * (1 + 1e-6)
        end
    end

    @testset "unprojected solver on the problems of Prather et al. (2023)" begin
        # S(3) from the closed forms: I ± Q decay at α_I ± α_Q in the first problem;
        # P = λ j_∥ + sin(ρλ)/ρ j_⊥ + (1 - cos ρλ)/ρ ρ̂ × j_⊥ in the second
        jE = SVector(2.0, 1.0, 0.0, 0.0); αE = SVector(1.0, 1.2, 0.0, 0.0)
        plus = 3.0 / 2.2 * -expm1(-6.6); minus = 1.0 / -0.2 * -expm1(0.6)
        exactE = SVector((plus + minus) / 2, (plus - minus) / 2, 0.0, 0.0)
        jR = SVector(0.0, 0.1, 0.1, 0.1); ρR = SVector(0.0, 10.0, 0.0, -4.0)
        r = SVector(10.0, 0.0, -4.0); ρn = sqrt(116.0); rh = r ./ ρn; jp = SVector(0.1, 0.1, 0.1)
        jpar = dot(rh, jp) .* rh; jperp = jp .- jpar
        P = 3.0 .* jpar .+ sin(3ρn) / ρn .* jperp .+ (1 - cos(3ρn)) / ρn .* cross(rh, jperp)
        exactR = SVector(0.0, P...)
        for n in (1, 30, 300)
            @test chain_unprojected(jE, αE, ZERO4, 3.0, n) ≈ exactE rtol = 1e-12
            @test chain_unprojected(jR, ZERO4, ρR, 3.0, n) ≈ exactR rtol = 1e-12
        end
        # production projects both onto the physical cone
        @test segment(IQUV(), jE, αE, ZERO4, 0.01) ==
              segment(IQUV(), jE, SVector(1.0, 1.0, 0.0, 0.0), ZERO4, 0.01; project = false)
        @test couple(IQUV(), ZERO4, jR, ZERO4, ρR, 0.01) == ZERO4
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
