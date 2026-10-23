# The backend switch through the products: an explicit keyword wins over the session
# default, which wins over BLP_BACKEND; `nothing` is the session default; an unusable
# :gpu is an error; every product records where it ran.
using Test, BlackLightPlus
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare
using BlackLightPlus.Products: render, render_amr, render_analytic, trace, shadow, sensitivity, variants
using BlackLightPlus.Spacetime: KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit
using BlackLightPlus.Matter: EquatorialDisk
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Execution: BackendUnavailable, backend!, reset_backend!, cuda_loaded, functional, CUDADevice
const Decks = Main.Decks

@testset "backend switch" begin
    p = prepare(RunConfig(camera = CameraConfig(fov = 24.0, resolution = 4)); spacetime = KerrSchildCartesian(0.9),
                matter = Decks.uniform_deck(), mode = StokesI())
    if !(cuda_loaded() && functional(CUDADevice))
        @test_throws BackendUnavailable render(p; backend = :gpu)
        @test_throws BackendUnavailable trace(p; backend = :gpu)
    end
    reset_backend!()
    try
        withenv("BLP_BACKEND" => "host_pool:2") do
            r = render(p)
            @test r.run.executor.kind === :pool && length(r.run.executor.devices) == 2
            backend!(:cpu)
            @test render(p).run.executor.kind === :threads
            @test render(p; backend = :host_pool).run.executor.kind === :pool
            backend!(nothing)
            @test render(p).run.executor.kind === :pool
        end
        backend!(:host_pool)
        runs = (render(p).run, trace(p).run, shadow(p).run, sensitivity(p, :mass_scale).image.run,
                variants(p, [1.0e24]).run, render_analytic(p, EquatorialDisk(0.9)).run,
                render_amr(p; block = 2, max_level = 1).image.run)
        @test all(r -> r.executor.kind === :pool && r.executor.devices == ["HostDevice(1)"], runs)
        @test all(r -> r.version == pkgversion(BlackLightPlus) && r.lanes === p.lanes, runs)
    finally
        reset_backend!()
    end
end

# The moving binary through every product on the pool backend (the GPU under
# `BLP_TEST_DEVICES=cuda`): bitwise against the threads on one hardware, to 1e-9 of the
# largest value across hardware.
agree(a, b) = Decks.CROSS_HARDWARE ? maximum(abs.(a .- b)) <= 1.0e-9 * maximum(abs.(b)) : Decks.bits(a) == Decks.bits(b)

@testset "moving binary on the pool backend" begin
    orbit = CircularOrbit((0.5, 0.5), 8.0)
    mov = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
    @test Decks.device_isbits(mov)
    cfg = RunConfig(camera = CameraConfig(fov = 24.0, resolution = 6, distance = 60.0), max_steps = 4000)
    pool = Decks.POOL_BACKEND
    for mode in (StokesI(), IQUV())
        p = prepare(cfg; spacetime = mov, matter = Decks.uniform_deck(), mode = mode)
        img = render(p; backend = :cpu)
        @test count(>(0), img.I) > 0 && all(isfinite, img.I)
        @test agree(render(p; backend = pool).I, img.I)
        t = trace(p; backend = :cpu)
        tp = trace(p; backend = pool)
        @test map(length, tp.states) == map(length, t.states)
        @test all(agree(reduce(hcat, tp.states[k]), reduce(hcat, t.states[k])) for k in eachindex(t.states))
        @test agree(render(p; path = tp, backend = :cpu).I, render(p; path = t, backend = :cpu).I)   # the two-phase render is host-only
    end
    ps = prepare(cfg; spacetime = mov, matter = nothing)
    s = shadow(ps; backend = :cpu)
    sp = shadow(ps; backend = pool)
    @test s.fate == sp.fate && s.steps == sp.steps
    @test count(==(2), s.fate) > 0 && count(==(1), s.fate) > 0        # captured and escaped rays
end
