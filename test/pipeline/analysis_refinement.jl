# Block refinement indicators: the documented formulas and thresholds on blocks whose
# indicator values are known in closed form, the counting and disabling conventions,
# non-finite pixels, the protocol (built-in indicators, callables, disjunction, the
# keyword form), and a class indicator on a disk deck: a block refines exactly when the
# disk edge crosses its pixel centres.
using Test, BlackLightPlus
using BlackLightPlus.Analysis: RefinementIndicator, RelativeGradient, RelativeLaplacian, AbsoluteGradient,
                    AnyOf, refinement_indicator, needs_refinement

# Two-valued 8×8 step: columns 1–4 hold 1, columns 5–8 hold 2.
step_block() = [j <= 4 ? 1.0 : 2.0 for i in 1:8, j in 1:8]

@testset "refinement indicators" begin
    @testset "constructors and the keyword form" begin
        @test RelativeGradient() === RelativeGradient(0.15, 0.03)
        @test RelativeLaplacian() === RelativeLaplacian(1.0, 0.03)
        @test AbsoluteGradient(0.1) === AbsoluteGradient(0.1, 5)
        @test RelativeGradient(1; frac = 0) === RelativeGradient(1.0, 0.0)
        @test AbsoluteGradient(2; count = Int32(3)) === AbsoluteGradient(2.0, 3)
        @test refinement_indicator() === AnyOf(RelativeGradient(), RelativeLaplacian())
        @test refinement_indicator(; rel_grad_cut = 0.2, rel_lapl_frac = -1.0) ===
              AnyOf(RelativeGradient(0.2, 0.03), RelativeLaplacian(1.0, -1.0))
        @test refinement_indicator(; abs_grad_cut = 0.5, abs_grad_count = 2) === AbsoluteGradient(0.5, 2)
        @test all(x -> x isa RefinementIndicator,
                  (RelativeGradient(), RelativeLaplacian(), AbsoluteGradient(1.0), AnyOf()))
        @test isbits(refinement_indicator())
    end

    @testset "uniform and linear blocks" begin
        U = fill(3.0, 8, 8)
        @test !needs_refinement(RelativeGradient(0.0; frac = 0.0), U)
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), U)
        @test !needs_refinement(AbsoluteGradient(0.0; count = 1), U)
        # I = 1 + j/4: absolute gradient exactly 1/4 at every pixel (edges and interior)
        R = [1 + 0.25j for i in 1:8, j in 1:8]
        @test needs_refinement(AbsoluteGradient(prevfloat(0.25); count = 64), R)
        @test !needs_refinement(AbsoluteGradient(0.25; count = 1), R)
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), R)    # L = 0
        # relative gradient of the ramp: 1/(4 + j) inside, 2/11 and 2/23 at the edges
        @test !needs_refinement(RelativeGradient(0.2; frac = 0.0), R)
        @test needs_refinement(RelativeGradient(0.1; frac = 0.5), R)
    end

    @testset "relative gradient: fraction above the cut" begin
        S = step_block()
        # q = 2/5 in column 4, 2/7 in column 5, 0 elsewhere
        @test needs_refinement(RelativeGradient(0.15; frac = 0.2), S)       # 16/64 > 0.2
        @test !needs_refinement(RelativeGradient(0.15; frac = 0.25), S)     # strict
        @test needs_refinement(RelativeGradient(0.3; frac = 0.1), S)        # 8/64 > 0.1
        @test !needs_refinement(RelativeGradient(0.4; frac = 0.0), S)       # strict cut
        @test needs_refinement(RelativeGradient(prevfloat(0.4); frac = 0.0), S)
        @test !needs_refinement(RelativeGradient(0.0; frac = -1.0), S)      # disabled
        @test !needs_refinement(RelativeGradient(0.0; frac = NaN), S)
        # the decision is symmetric under transposition and works on rectangles
        @test needs_refinement(RelativeGradient(0.15; frac = 0.2), permutedims(S))
        @test needs_refinement(RelativeGradient(0.15; frac = 0.2), S[1:3, :])
        @test !needs_refinement(RelativeGradient(0.15; frac = 0.2), S[:, 5:8])
    end

    @testset "relative Laplacian: interior pixels" begin
        # I = j²: L = 2 and m = j² + 1/4, so q = 2/(j² + 1/4); above 0.3 only at j = 2
        Q = [Float64(j^2) for i in 1:8, j in 1:8]
        @test needs_refinement(RelativeLaplacian(0.3; frac = 0.1), Q)        # 6/36
        @test !needs_refinement(RelativeLaplacian(0.3; frac = 1 / 6), Q)     # strict
        @test !needs_refinement(RelativeLaplacian(0.3; frac = -0.5), Q)      # disabled
        # no interior pixel: never fires
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), [1.0 5.0; 2.0 9.0])
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), rand(1, 7))
    end

    @testset "non-finite pixels are not counted" begin
        Z = zeros(8, 8)                                   # 0/0 everywhere
        @test !needs_refinement(RelativeGradient(0.0; frac = 0.0), Z)
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), Z)
        N = fill(2.0, 8, 8); N[4, 4] = NaN               # poisons its stencil neighbours only
        @test !needs_refinement(RelativeGradient(0.0; frac = 0.0), N)
        @test !needs_refinement(RelativeLaplacian(0.0; frac = 0.0), N)
        @test !needs_refinement(AbsoluteGradient(0.0; count = 1), N)
        N[1, 8] = 3.0
        @test needs_refinement(RelativeGradient(0.0; frac = 0.0), N)
        @test needs_refinement(AbsoluteGradient(0.4; count = 3), N)       # √2, 1/2, 1/2
        @test !needs_refinement(AbsoluteGradient(0.4; count = 4), N)
    end

    @testset "absolute gradient: count" begin
        S = step_block()                   # q = 1/2 in columns 4 and 5: 16 pixels
        @test needs_refinement(AbsoluteGradient(0.4; count = 16), S)
        @test !needs_refinement(AbsoluteGradient(0.4; count = 17), S)
        @test !needs_refinement(AbsoluteGradient(0.5; count = 1), S)
        # count ≤ 0 fires at the first finite pixel, and never on an all-NaN block
        @test needs_refinement(AbsoluteGradient(1e9; count = 0), fill(1.0, 4, 4))
        @test !needs_refinement(AbsoluteGradient(1e9; count = 0), fill(NaN, 4, 4))
    end

    @testset "protocol: callables, disjunction, block size" begin
        S = step_block()
        @test needs_refinement(I -> maximum(I) > 1.5, S)
        @test !needs_refinement(I -> maximum(I) > 2.5, S)
        @test_throws TypeError needs_refinement(I -> 1, S)
        @test !needs_refinement(AnyOf(), S)
        @test needs_refinement(AnyOf(I -> true, I -> error("not evaluated")), S)
        @test needs_refinement(AnyOf(RelativeGradient(0.4; frac = 0.0), AbsoluteGradient(0.4)), S)
        @test !needs_refinement(AnyOf(RelativeGradient(0.4; frac = 0.0), AbsoluteGradient(0.5)), S)
        @test needs_refinement(AnyOf(AnyOf(I -> false), RelativeLaplacian(0.0; frac = 0.0)), S)
        @test_throws ArgumentError needs_refinement(RelativeGradient(), rand(1, 8))
        @test_throws ArgumentError needs_refinement(AbsoluteGradient(0.1), rand(8, 1))
        @test needs_refinement(RelativeGradient(0.0; frac = 0.0), [1.0 2.0; 1.0 2.0])
        # Float32 blocks and views are accepted
        @test needs_refinement(RelativeGradient(0.15; frac = 0.2), Float32.(S))
        @test needs_refinement(RelativeGradient(0.15; frac = 0.2), view(S, :, 1:8))
    end

    @testset "class indicator on a disk deck" begin
        # level-0 image of a 32² camera over a 16-wide field; class 1 inside ρ < 3.3
        npix, block, fov, R = 32, 8, 16.0, 3.3
        ps = fov / npix
        c = [-fov / 2 + (k - 0.5) * ps for k in 1:npix]
        img = [hypot(c[i], c[j]) < R ? 1.0 : 0.0 for i in 1:npix, j in 1:npix]
        class_change = I -> any(!=(I[1, 1]), I)
        nb = npix ÷ block
        blocks = [img[(bi-1)*block+1:bi*block, (bj-1)*block+1:bj*block] for bi in 1:nb, bj in 1:nb]
        mixed = [any(==(1.0), b) && !all(==(1.0), b) for b in blocks]
        @test any(mixed) && !all(mixed)
        @test map(b -> needs_refinement(class_change, b), blocks) == mixed
        @test map(b -> needs_refinement(AbsoluteGradient(0.1; count = 1), b), blocks) == mixed
        # pure: concurrent evaluation agrees with the serial one
        ind = refinement_indicator()
        serial = map(b -> needs_refinement(ind, b), blocks)
        par = similar(serial)
        Threads.@threads for k in eachindex(blocks)
            par[k] = needs_refinement(ind, blocks[k])
        end
        @test par == serial
    end
end
