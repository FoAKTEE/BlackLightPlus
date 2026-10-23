# The backend switch through the products: an explicit keyword wins over the session
# default, which wins over BLP_BACKEND; `nothing` is the session default; an unusable
# :gpu is an error; every product records where it ran.
using Test, BlackLightPlus
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare
using BlackLightPlus.Products: render, render_amr, render_analytic, trace, shadow, sensitivity, variants
using BlackLightPlus.Spacetime: KerrSchildCartesian
using BlackLightPlus.Matter: EquatorialDisk
using BlackLightPlus.Transfer: StokesI
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
