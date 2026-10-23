# Trajectories and shadows: the windowed fixed-control trace against a single march and
# on every executor, the step limit, the Float32 lane, the host-only adaptive trace, the
# Schwarzschild shadow, and a shadow on an ADM grid.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, RunConfig, Lanes, MixedLanes, prepare
using BlackLightPlus.Products: Trajectories, trace, shadow, TRACE_WINDOW
using BlackLightPlus.Spacetime: KerrSchildCartesian, adm_grid_from_metric, metric, ks_radius, horizon_radius,
                     trace_geodesic, camera_ray, lane
using BlackLightPlus.Execution: PoolExecutor
const Decks = Main.Decks

cfg(; res = 10, kw...) = RunConfig(; camera = CameraConfig(fov = 24.0, resolution = res), kw...)
plan(; st = KerrSchildCartesian(0.9), matter = Decks.uniform_deck(), res = 10, lanes = Lanes(), kw...) =
    prepare(cfg(; res, kw...); spacetime = st, matter, lanes)

"Largest component difference of two trajectory sets, relative to 1 + |state|."
pathdev(s1, s2) = maximum(maximum(maximum(abs.(a .- b) ./ (1 .+ abs.(b))) for (a, b) in zip(x, y))
                          for (x, y) in zip(s1, s2))

@testset "trajectories and shadows" begin
    @testset "fixed-control trace" begin
        p = plan()
        t = trace(p; backend = :cpu)
        @test t isa Trajectories && length(t.states) == 100 && t.plan === p
        @test all(length(s) == length(d) && d[end] == 0.0 && all(>(0), d[1:end-1])
                  for (s, d) in zip(t.states, t.dls))
        rays = Decks.camera_rays(p.spacetime, p.camera)
        @test all(s[end] == y for (s, y) in zip(t.states, rays))
        ref = Decks.sampled_trace(p.spacetime, rays; C = 80.0, escape = p.escape, nmax = 2500,
                                  R_em = p.R_em)
        @test length.(t.states) == length.(ref[1])
        @test pathdev(t.states, ref[1]) < 1e-7 && pathdev(t.dls, ref[2]) < 1e-7
        @test maximum(length, t.states) > 4 * TRACE_WINDOW
        for exec in (:host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
            u = trace(p; backend = exec)
            @test Decks.bits(u.states) == Decks.bits(t.states) && Decks.bits(u.dls) == Decks.bits(t.dls)
        end
        @test t.run.executor.kind === :threads && t.run.seconds >= 0
    end

    @testset "step limit and the Float32 lane" begin
        for nmax in (TRACE_WINDOW - 3, 2 * TRACE_WINDOW, 37)
            t = trace(plan(max_steps = nmax); backend = :cpu)
            @test maximum(length, t.states) == nmax + 1
        end
        p32 = plan(lanes = Lanes{Float32,Float64,Float64}())
        t = trace(p32; backend = :host_pool)
        @test all(all(y -> SVector{8,Float64}(SVector{8,Float32}(y)) == y, s) for s in t.states)
        rays = [SVector{8,Float32}(y) for y in Decks.camera_rays(p32.spacetime, p32.camera)]
        ref = Decks.sampled_trace(lane(p32.spacetime, Float32), rays; C = 80.0, escape = p32.escape,
                                  nmax = 2500, R_em = p32.R_em)
        # a Float32 march resolves the critical curve poorly: compare the step counts
        @test count(length.(t.states) .== length.(ref[1])) >= 90
        u = trace(p32; backend = :cpu)
        @test Decks.bits(u.states) == Decks.bits(t.states) && Decks.bits(u.dls) == Decks.bits(t.dls)
        @test_throws ArgumentError trace(plan(lanes = MixedLanes()); backend = :cpu)
    end

    @testset "adaptive trace (host only)" begin
        p = plan(res = 4)
        t = trace(p; backend = :cpu, adaptive = true)
        y0, ok = camera_ray(p.spacetime, p.frame, p.pixels.centres[2], p.pixels.centres[2])
        sol = trace_geodesic(p.spacetime, y0; escape = p.escape, reltol = 1e-9, abstol = 1e-11,
                             save_everystep = true)
        @test t.states[6] == reverse(sol.u) && t.dls[6][1:end-1] == -diff(reverse(sol.t))
        @test all(d[end] == 0.0 for d in t.dls)
        @test_throws ArgumentError trace(p; backend = :host_pool, adaptive = true)
    end

    @testset "Schwarzschild shadow" begin
        c = RunConfig(camera = CameraConfig(inclination_deg = 90.0, fov = 16.0, resolution = 16))
        p = prepare(c; spacetime = KerrSchildCartesian(0.0), matter = nothing)
        sh = shadow(p; backend = :cpu)
        c0 = p.pixels.centres
        b = [hypot(c0[iα], c0[iβ]) for iβ in 1:16, iα in 1:16]
        bc = 3 * sqrt(3.0)
        @test all(sh.fate[b .< bc - 0.3] .== 1) && all(sh.fate[b .> bc + 0.3] .== 2)
        @test maximum(sh.null_error[sh.fate .== 2]) < 1e-8 && all(sh.steps .> 0)
        @test all(sh.radius[sh.fate .== 2] .> p.escape)
        u = shadow(p; backend = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7))
        @test u.fate == sh.fate && u.steps == sh.steps &&
              Decks.bits((sh.null_error, sh.radius, sh.final)) == Decks.bits((u.null_error, u.radius, u.final))
        @test_throws ArgumentError shadow(prepare(c; spacetime = KerrSchildCartesian(0.5), matter = nothing,
                                                  lanes = MixedLanes()); backend = :cpu)
    end

    @testset "shadow on an ADM grid" begin
        kerr = KerrSchildCartesian(0.9)
        xs = -16.0:1.0:16.0
        grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min = 0.1,
                                    excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
        c = RunConfig(camera = CameraConfig(inclination_deg = 60.0, distance = 12.0, fov = 12.0,
                                            resolution = 6))
        p = prepare(c; spacetime = grid, matter = nothing)
        sh = shadow(p; backend = :cpu)
        @test 0 < count(==(1), sh.fate) < 36 && count(==(2), sh.fate) > 0
        @test all(isfinite, sh.null_error) && all(all(isfinite, y) for y in sh.final)
        u = shadow(p; backend = :host_pool)
        @test u.fate == sh.fate && Decks.bits(u.final) == Decks.bits(sh.final)
    end
end
