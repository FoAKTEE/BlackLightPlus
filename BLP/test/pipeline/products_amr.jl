# Adaptive refinement: the mosaic, the level map and the ray count independent of the
# executor, the device count and the chunking; refinement driven by built-in and callable
# indicators; the argument checks.
using Test, BLP, StaticArrays
using BLP.Plans: CameraConfig, RunConfig, MixedLanes, prepare
using BLP.Products: Image, render, render_amr
using BLP.Spacetime: KerrSchildCartesian
using BLP.Transfer: StokesI, IQUV
using BLP.Analysis: refinement_indicator, AbsoluteGradient
using BLP.Execution: PoolExecutor
const Decks = Main.Decks

plan(; mode = StokesI(), res = 16, kw...) =
    prepare(RunConfig(camera = CameraConfig(fov = 24.0, resolution = res));
            spacetime = KerrSchildCartesian(0.9), matter = Decks.amr_deck(), mode, kw...)
planes(img) = (img.I, img.Q, img.U, img.V)
result_bits(r) = (Decks.bits(planes(r.image)), r.levels, r.nrays)

@testset "adaptive refinement" begin
    @testset "mosaic independent of executor and chunking ($(nameof(typeof(mode))))" for
            mode in (StokesI(), IQUV())
        p = plan(; mode)
        ref = render_amr(p; block = 4, max_level = 2, backend = :cpu)
        @test ref.image isa Image && size(ref.image.I) == (64, 64) && size(ref.levels) == (4, 4)
        @test maximum(ref.levels) >= 1 && ref.nrays > 16^2
        @test all(isfinite, ref.image.I) && count(>(0), ref.image.I) > 64^2 ÷ 4
        mode isa StokesI && @test all(iszero, ref.image.Q)
        for exec in (:host_pool, PoolExecutor(Decks.hosts(1); lanes = 2, chunk = 7),
                     PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7),
                     PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 50),
                     PoolExecutor(Decks.hosts(5); lanes = 3, chunk = 16))
            @test result_bits(render_amr(p; block = 4, max_level = 2, backend = exec)) == result_bits(ref)
        end
    end

    @testset "indicators decide the refinement" begin
        p = plan()
        none = render_amr(p; block = 4, max_level = 2, refine = I -> false, backend = :cpu)
        @test all(iszero, none.levels) && none.nrays == 16^2
        all4 = render_amr(p; block = 4, max_level = 2, refine = I -> true, backend = :cpu)
        @test all(==(2), all4.levels) && all4.nrays == 16^2 * (1 + 4 + 16)
        # unrefined blocks paint their pixels over 4 × 4 canvas cells
        @test none.image.I[1:4, 1:4] == fill(none.image.I[1, 1], 4, 4)
        shallow = render_amr(p; block = 4, max_level = 1, backend = :cpu)
        @test size(shallow.image.I) == (32, 32) && maximum(shallow.levels) == 1
        steep = render_amr(p; block = 8, max_level = 1, refine = AbsoluteGradient(-1.0; count = 1),
                           backend = :cpu)
        @test all(==(1), steep.levels) && steep.nrays == 16^2 * 5
        def = render_amr(p; block = 4, max_level = 2, refine = refinement_indicator(), backend = :cpu)
        @test result_bits(def) == result_bits(render_amr(p; block = 4, max_level = 2, backend = :cpu))
    end

    @testset "argument checks" begin
        @test_throws ArgumentError render_amr(plan(); block = 5, backend = :cpu)
        @test_throws ArgumentError render_amr(plan(; lanes = MixedLanes()); block = 4, backend = :cpu)
    end
end
