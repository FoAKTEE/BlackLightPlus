# Full-frame and coordinate renders: the image record and its run metadata, a grid
# provider and an analytic source on both executors, coordinate lists against the pixel
# grid, the mixed-lane partition and the rescue of the critical curve, the two-phase
# render along stored trajectories, the fast-light limit of a frame window, and the
# polarized image of an analytic torus.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, PlasmaConfig, RunConfig, Lanes, MixedLanes, prepare
using BlackLightPlus.Products: Image, render, trace
using BlackLightPlus.Spacetime: KerrSchildCartesian, pixel_centres
using BlackLightPlus.Matter: PolishDoughnut
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Data: FrameWindow, scale_density
using BlackLightPlus.Execution: PoolExecutor, CoordList
const Decks = Main.Decks

cfg(; res = 12, kw...) = RunConfig(; camera = CameraConfig(fov = 24.0, resolution = res), kw...)
plan(matter; mode = IQUV(), res = 12, kw...) =
    prepare(cfg(; res); spacetime = KerrSchildCartesian(0.9), matter, mode, kw...)
planes(img) = (img.I, img.Q, img.U, img.V)
same(a, b) = Decks.bits(planes(a)) == Decks.bits(planes(b))

@testset "render" begin
    grid = Decks.uniform_deck()

    @testset "the image and its run record" begin
        p = plan(grid; mode = StokesI())
        img = render(p; backend = :cpu)
        @test img isa Image && img.plan === p
        @test all(size(m) == (12, 12) for m in planes(img))
        @test all(iszero, img.Q) && all(iszero, img.U) && all(iszero, img.V)
        @test count(>(0), img.I) > 40 && all(isfinite, img.I)
        @test keys(img.run) == (:executor, :lanes, :version, :seconds)
        @test keys(img.run.executor) == (:kind, :devices, :lanes, :chunk, :threads)
        @test img.run.executor.kind === :threads && img.run.lanes === p.lanes
        @test img.run.version == pkgversion(BlackLightPlus) && img.run.seconds >= 0
        pool = render(p; backend = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7)).run.executor
        @test pool.kind === :pool && pool.devices == ["HostDevice(1)", "HostDevice(2)"] &&
              pool.chunk == 7 && pool.lanes == 2
        q = render(plan(grid); backend = :cpu)
        @test isapprox(sum(q.I), sum(img.I); rtol = 1e-2)
        @test count(!iszero, q.Q) > 40 && count(!iszero, q.V) > 40
        @test occursin("12×12", sprint(show, img))
        v = rand(144)                            # a Float64 output vector is the I plane
        w = Image(p, v, img.run)
        @test w.I isa Matrix{Float64} && size(w.I) == (12, 12) && pointer(w.I) == pointer(v)
        @test all(iszero, w.Q) && w.Q !== w.U && w.U !== w.V
        @test Image(p, Float32.(v), img.run).I == reshape(Float64.(Float32.(v)), 12, 12)
    end

    @testset "$name on both executors" for (name, matter) in (("torus grid provider", Decks.TorusDeck()),
                                                               ("Polish doughnut", PolishDoughnut(0.9)))
        for mode in (StokesI(), IQUV())
            p = plan(matter; mode)
            img = render(p; backend = :cpu)
            @test same(img, render(p; backend = :host_pool))
            @test same(img, render(p; backend = PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7)))
            @test count(>(0), img.I) > 20 && all(isfinite, img.I)
        end
    end

    @testset "polarized torus: admissible, ordered, circularly polarized" begin
        img = render(plan(PolishDoughnut(0.9); res = 16); backend = :cpu)
        @test all(isfinite, img.I) && all(>=(0.0), img.I) && sum(img.I) > 0
        P = @. sqrt(img.Q^2 + img.U^2 + img.V^2)
        @test all(P .<= img.I .+ 1e-9 * maximum(img.I))
        @test 0 <= hypot(sum(img.Q), sum(img.U)) / sum(img.I) < 0.75
        @test sum(abs, img.V) / sum(img.I) > 1e-4
    end

    @testset "coordinate lists are the pixel grid's rays" begin
        for mode in (StokesI(), IQUV())
            p = plan(Decks.amr_deck(); mode)
            img = render(p; backend = :cpu)
            c = pixel_centres(p.camera)
            ks = [3, 50, 77, 100, 144]
            coords = [(c[(k - 1) ÷ 12 + 1], c[(k - 1) % 12 + 1]) for k in ks]
            S = render(p, coords; backend = :host_pool)
            @test size(S) == (4, length(ks))
            @test Decks.bits(S) == Decks.bits([getindex(m, k) for m in planes(img), k in ks])
            @test S == render(p, CoordList(coords); backend = :cpu)
        end
        @test_throws ArgumentError render(plan(grid; lanes = MixedLanes()), [(1.0, 2.0)]; backend = :cpu)
    end

    @testset "mixed lanes: the rescue set marched in Float64, the rest in Float32" begin
        for mode in (StokesI(), IQUV())
            p64 = plan(grid; mode, res = 16)
            p32 = plan(grid; mode, res = 16, lanes = Lanes{Float32,Float64,Float64}())
            pm = plan(grid; mode, res = 16, lanes = MixedLanes())
            i64 = planes(render(p64; backend = :cpu)); i32 = planes(render(p32; backend = :cpu))
            im = planes(render(pm; backend = :host_pool))
            rescued = falses(256); rescued[pm.rescue] .= true
            @test all(Decks.bits(m[rescued]) == Decks.bits(r[rescued]) for (m, r) in zip(im, i64))
            @test all(Decks.bits(m[.!rescued]) == Decks.bits(r[.!rescued]) for (m, r) in zip(im, i32))
            @test same(render(pm; backend = :cpu), render(pm; backend = :host_pool))
            # the fp32 lane's large errors lie in the rescue set
            bad = findall(abs.(i32[1] .- i64[1]) .> 1e-2 .* maximum(i64[1]))
            @test all(in(pm.rescue), bad)
        end
    end

    @testset "two-phase render along stored trajectories" begin
        for mode in (StokesI(), IQUV())
            p = plan(grid; mode)
            t = trace(p; backend = :cpu)
            two = render(p; path = t, backend = :cpu)
            fused = render(p; backend = :cpu)
            @test all(isapprox(a, b; atol = 1e-6 * maximum(fused.I)) for (a, b) in zip(planes(two), planes(fused)))
            @test_throws ArgumentError render(p; path = t, backend = :host_pool)
        end
        @test_throws ArgumentError render(plan(grid; res = 8); path = trace(plan(grid); backend = :cpu),
                                          backend = :cpu)
    end

    @testset "frame window: the fast-light limit is the lower frame" begin
        for mode in (StokesI(), IQUV())
            fast = FrameWindow(grid, scale_density(grid, 3.0), 0.0, Inf)
            @test same(render(plan(fast; mode); backend = :cpu), render(plan(grid; mode); backend = :cpu))
            # emission times t_obs + X⁰ ≈ 1030 - 1018 fall inside the window [0, 30]
            slow = FrameWindow(grid, scale_density(grid, 3.0), 0.0, 30.0; t_obs = 1030.0)
            s = render(plan(slow; mode); backend = :cpu)
            a = render(plan(grid; mode); backend = :cpu)
            @test maximum(abs.(s.I .- a.I)) > 1e-3 * maximum(a.I)
            @test same(s, render(plan(slow; mode); backend = :host_pool))
        end
    end
end
