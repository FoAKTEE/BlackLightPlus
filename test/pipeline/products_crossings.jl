# The marched image of the equatorial disk: the precise march against the closed form on
# the two comparison configurations, the convergence of the production integrator with
# the camera at infinity, executor independence, the program layout and the checks.
using Test, BlackLightPlus, StaticArrays, Statistics
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare, precise, PRECISE_C, PRECISE_CAPTURE
using BlackLightPlus.Products: render_crossings, render_analytic, CrossingProgram, crossing_outputs
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, BoostedSuperposedKerrSchild,
                                CircularOrbit, Verner9, ClassicalRK4, isco_prograde
using BlackLightPlus.Matter: EquatorialDisk, PolishDoughnut
using BlackLightPlus.Transfer: IQUV
using BlackLightPlus.Products: render, trace, shadow
using BlackLightPlus.Execution: PoolExecutor, CameraRays
const Decks = Main.Decks

# |marched - closed form| relative to the closed-form maximum of each order
relerr(m, c) = [abs.(m.Ib[b] .- c.Ib[b]) ./ maximum(c.Ib[b]) for b in 1:3]
agree(a, b) = Decks.CROSS_HARDWARE ? maximum(abs.(a .- b)) <= 1.0e-9 * maximum(abs.(b)) :
              Decks.bits(a) == Decks.bits(b)

@testset "marched disk" begin
    @testset "the precise march is the closed form" begin
        # configuration A of the code comparison (a = 0.94, i = 17°) and B (0.99, 80°)
        for (a, inc) in ((0.94, 17.0), (0.99, 80.0))
            cfg = precise(RunConfig(camera = CameraConfig(inclination_deg = inc, fov = 14.0,
                                                          resolution = 32)))
            p = prepare(cfg; spacetime = KerrSchildCartesian(a), matter = nothing)
            d = EquatorialDisk(a)
            m = render_crossings(p, d; backend = :cpu)
            c = render_analytic(p, d; ro = 1.0e15, backend = :cpu)
            for (b, e) in enumerate(relerr(m, c))
                lit = c.Ib[b] .> 0
                # the plunge velocity is the root of a difference that vanishes at the ISCO,
                # so there the disk's intensity amplifies the rounding of its radius
                far = lit .& (abs.(c.rs[b] .- isco_prograde(a)) .> 1e-3)
                @test count(lit) > 0 && maximum(e) < 1e-9 && median(e[lit]) < 1e-12
                @test maximum(e[far]) < 5e-11
                @test count(lit .⊻ (m.Ib[b] .> 0)) == 0
            end
            @test all(isfinite, m.steps) && maximum(m.steps) < cfg.max_steps
        end
        cfg = precise()
        @test cfg.integrator === :verner9 && cfg.C == PRECISE_C && cfg.capture_redshift == PRECISE_CAPTURE
        @test cfg.horizon_eps == 0 && cfg.winding && cfg.camera.infinity
        @test !precise(; infinity = false).camera.infinity
        b = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = nothing).budget
        @test b.integrator === Verner9() && b.winding && b.capture == PRECISE_CAPTURE
    end

    @testset "the production integrator converges with the camera at infinity" begin
        a = 0.94
        d = EquatorialDisk(a)
        pts = [(4.0, 3.0), (-5.5, 1.0), (2.0, -6.5), (6.0, 6.0)]
        cam = CameraConfig(inclination_deg = 17.0, infinity = true)
        err(C) = begin
            p = prepare(RunConfig(camera = cam, C = C, capture_redshift = 1e6, max_steps = 10^5);
                        spacetime = KerrSchildCartesian(a), matter = nothing)
            r = render_crossings(p, d; orders = (0,), coords = pts, backend = :cpu)
            q = render_crossings(prepare(precise(RunConfig(camera = cam)); spacetime = p.spacetime,
                                         matter = nothing), d; orders = (0,), coords = pts, backend = :cpu)
            maximum(abs.(r.Ib[1] .- q.Ib[1]))
        end
        e40, e80 = err(40.0), err(80.0)
        @test 10 < e40 / e80 < 25                          # fourth order
    end

    @testset "executors and layout" begin
        p = prepare(precise(RunConfig(camera = CameraConfig(inclination_deg = 60.0, fov = 16.0,
                                                            resolution = 8)));
                    spacetime = KerrSchildCartesian(0.9), matter = nothing)
        d = EquatorialDisk(0.9)
        r = render_crossings(p, d; backend = :cpu)
        @test r.I == r.Ib[1] + r.Ib[2] + r.Ib[3] && r.orders == [0, 1, 2] && size(r.steps) == (8, 8)
        for exec in (PoolExecutor(Decks.devices(2); chunk = 5),)
            q = render_crossings(p, d; backend = exec)
            @test all(agree(q.Ib[b], r.Ib[b]) && agree(q.steps, r.steps) for b in 1:3)
        end
        one = render_crossings(p, d; orders = (1,), backend = :cpu)
        @test Decks.bits(one.Ib[1]) == Decks.bits(r.Ib[2])
        @test length(crossing_outputs(3, 4)) == 16
        prog = CrossingProgram(p.geodesic_spacetime, p.geodesic_budget,
                               CameraRays(p.spacetime, p.frame, p.pixels), d, (0, 1, 2))
        @test Decks.device_isbits(prog)
        outs = crossing_outputs(3, 64)
        call(prog, outs) = prog(20, 20, outs...)
        call(prog, outs)
        @test (@allocated call(prog, outs)) == 0
        @test_throws ArgumentError render_crossings(p, EquatorialDisk(0.5); backend = :cpu)
        @test_throws ArgumentError render_crossings(p, d; orders = (2, 1), backend = :cpu)
    end

    @testset "the ninth-order budget through render, trace and shadow" begin
        st = KerrSchildCartesian(0.9)
        torus = PolishDoughnut(0.9; r_center = 9.0, n_e_center = 1e7, theta_e_center = 40.0,
                               polytrope_n = 1.5, beta = 8.0)
        cam = CameraConfig(inclination_deg = 60.0, fov = 30.0, resolution = 6)
        fine = render(prepare(RunConfig(camera = cam, C = 640.0, max_steps = 10^5); spacetime = st,
                              matter = torus, mode = IQUV()); backend = :cpu)
        p9 = prepare(precise(RunConfig(camera = cam); infinity = false, C = 160.0); spacetime = st,
                     matter = torus, mode = IQUV())
        r9 = render(p9; backend = :cpu)
        @test all(isfinite, r9.I) && all(isfinite, r9.Q) && all(isfinite, r9.V)
        # one plasma sample per step: the two agree to the transfer quadrature at C = 160
        @test abs(sum(r9.I) - sum(fine.I)) < 2e-2 * sum(fine.I)
        tr = trace(prepare(precise(RunConfig(camera = cam)); spacetime = st, matter = nothing);
                   backend = :cpu)
        @test all(s -> all(y -> all(isfinite, y), s), tr.states)
        sh = shadow(prepare(precise(RunConfig(camera = cam)); spacetime = st, matter = nothing);
                    backend = :cpu)
        @test all(in((1, 2)), sh.fate)
    end

    @testset "the camera at infinity needs a stationary spacetime" begin
        mov = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                                          CircularOrbit((0.5, 0.5), 8.0))
        @test_throws ArgumentError prepare(precise(); spacetime = mov, matter = nothing)
        @test prepare(precise(; infinity = false); spacetime = mov, matter = nothing).budget.winding
        bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        @test prepare(precise(); spacetime = bin, matter = nothing).frame.infinity
    end
end
