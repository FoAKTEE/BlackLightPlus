# The per-item programs: valid kernel arguments once adapted to a device, no allocation
# per call on every lane and parameter type, their launch traits, the invalid value, and
# a camera without positive-energy null roots.
using Test, BLP, StaticArrays
using ForwardDiff: Dual
using BLP.Plans: CameraConfig, RunConfig, Lanes, MixedLanes, prepare, transfer_params
using BLP.Products: ImagingProgram, TraceProgram, TraceStep, TraceState, ShadowProgram,
                    AnalyticProgram, imaging_program, shadow_program, trace_outputs, render,
                    trace, shadow
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, adm_grid_from_metric, metric
using BLP.Matter: PolishDoughnut, EquatorialDisk
using BLP.Transfer: StokesI, IQUV, output_type
using BLP.Data: FrameWindow, scale_density
using BLP.Execution: CameraRays, BlockGrid, CoordList, workgroupsize, stack_bytes
import BLP.Products: invalid, analytic_outputs
const Decks = Main.Decks

allocs(f, args...) = (f(args...); @allocated f(args...))
cfg(; res = 8) = RunConfig(camera = CameraConfig(fov = 24.0, resolution = res))
kerr() = KerrSchildCartesian(0.9)
plan(matter; st = kerr(), kw...) = prepare(cfg(); spacetime = st, matter, kw...)
bits_equal(a, b) = Decks.bits((a.I, a.Q, a.U, a.V)) == Decks.bits((b.I, b.Q, b.U, b.V))

@testset "programs" begin
    grid = Decks.uniform_deck()
    snap = Decks.amr_deck()
    bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    xs = -14.0:1.0:14.0
    adm = adm_grid_from_metric(x -> metric(kerr(), x), xs, xs, xs; alpha_min = 0.1)
    win = FrameWindow(grid, scale_density(grid, 1.3), 0.0, 30.0; t_obs = 15.0)
    cases = (("uniform grid", plan(grid)), ("blocks", plan(snap)), ("torus", plan(Decks.TorusDeck())),
             ("Polish doughnut", plan(PolishDoughnut(0.9))), ("frame window", plan(win)),
             ("binary", plan(grid; st = bin)), ("ADM grid", plan(grid; st = adm)),
             ("fp32 lane", plan(grid; lanes = Lanes{Float32,Float64,Float64}())),
             ("Stokes I", plan(snap; mode = StokesI())))

    @testset "imaging: $name" for (name, p) in cases
        prog = imaging_program(p)
        @test prog isa ImagingProgram && Decks.device_isbits(prog)
        out = Vector{output_type(p)}(undef, 64)
        @test allocs(prog, 28, 3, out) == 0
        @test Decks.device_isbits(imaging_program(p; gen = BlockGrid(4)))
        @test Decks.device_isbits(imaging_program(p; gen = CoordList([(1.0, 2.0), (3.0, -4.0)])))
    end

    @testset "imaging: mixed lanes, derivatives, ladders" begin
        p = plan(grid; lanes = MixedLanes(), mode = StokesI())
        rescue = imaging_program(p, p.spacetime, p.budget)
        bulk = imaging_program(p)
        out = Vector{Float64}(undef, 64)
        @test allocs(rescue, 28, 1, out) == 0 && allocs(bulk, 28, 2, out) == 0
        @test Decks.device_isbits(rescue) && Decks.device_isbits(bulk)
        pd = transfer_params(p.config; mass_scale = Dual(p.config.mass_scale_cgs, 1.0))
        dprog = imaging_program(plan(grid); mode = StokesI(), params = pd)
        dout = Vector{typeof(Dual(1.0, 1.0))}(undef, 64)
        @test allocs(dprog, 28, 1, dout) == 0 && Decks.device_isbits(dprog)
        lad = ntuple(k -> transfer_params(p.config; mass_scale = k * p.config.mass_scale_cgs), 3)
        lprog = imaging_program(plan(grid); mode = StokesI(), params = lad)
        lout = Vector{SVector{3,Float64}}(undef, 64)
        @test allocs(lprog, 28, 1, lout) == 0 && Decks.device_isbits(lprog)
    end

    @testset "trace, shadow and analytic programs" begin
        p = plan(grid)
        rays = CameraRays(p.spacetime, p.frame, p.pixels)
        outs = trace_outputs(Float64, 64)
        tp = TraceProgram(p.geodesic_spacetime, p.geodesic_budget, rays, nothing)
        @test allocs(tp, 28, 5, outs...) == 0 && Decks.device_isbits(tp)
        st = outs[end][5]
        @test st isa TraceState{Float64} && st.ok && st.live && st.done == BLP.Products.TRACE_WINDOW
        @test outs[1][5] isa TraceStep{Float64} && outs[1][5].h == 0.0
        tr = TraceProgram(p.geodesic_spacetime, p.geodesic_budget, rays, [st])
        @test allocs(tr, 1, 6, outs...) == 0 && Decks.device_isbits(tr)
        @test outs[1][6].y == st.y && outs[end][6].done == 2 * BLP.Products.TRACE_WINDOW
        p32 = plan(grid; lanes = Lanes{Float32,Float64,Float64}())
        t32 = TraceProgram(p32.geodesic_spacetime, p32.geodesic_budget, rays, nothing)
        @test allocs(t32, 28, 5, trace_outputs(Float32, 64)...) == 0
        sp = shadow_program(p)
        @test sp isa ShadowProgram && Decks.device_isbits(sp)
        ys = Vector{SVector{8,Float64}}(undef, 64); ms = Vector{SVector{4,Float64}}(undef, 64)
        @test allocs(sp, 28, 1, ys, ms) == 0
        ap = AnalyticProgram(p.pixels, p.theta_o, 0.9, 1000.0, EquatorialDisk(0.9), (0, 1, 2))
        aouts = analytic_outputs(3, 64)
        @test Decks.device_isbits(ap) && aouts isa NTuple{15,Vector{Float64}}
        # JacobiElliptic's sn and cn allocate when bounds checks are forced on (Pkg.test)
        Base.JLOptions().check_bounds == 1 || @test allocs(ap, 28, 1, aouts...) == 0
        @test Decks.device_isbits(AnalyticProgram(CoordList([(1.0, 2.0)]), 1.0, 0.9, 1000.0,
                                                  EquatorialDisk(0.9), (1,)))
    end

    @testset "launch traits and the invalid value" begin
        @test workgroupsize(imaging_program(plan(grid))) == 64
        @test workgroupsize(imaging_program(plan(grid; mode = StokesI()))) === nothing
        p = plan(grid)
        @test stack_bytes(AnalyticProgram(p.pixels, p.theta_o, 0.9, 1000.0, EquatorialDisk(0.9), (0,))) ==
              32 * 1024
        @test stack_bytes(imaging_program(p)) == 0
        @test isnan(invalid(Float64)) && all(isnan, invalid(SVector{4,Float64}))
        d = invalid(typeof(Dual(1.0, 1.0)))
        @test d isa Dual && isnan(d.value)
        @test_throws ArgumentError imaging_program(plan(nothing))
    end

    @testset "a camera without positive-energy null roots" begin
        # the camera inside the ergosphere: no ray has a past-oriented null root
        c = RunConfig(camera = CameraConfig(inclination_deg = 90.0, distance = 1.2, fov = 0.1,
                                            resolution = 2))
        p = prepare(c; spacetime = KerrSchildCartesian(0.99), matter = grid)
        img = render(p; backend = :cpu)
        @test all(isnan, img.I) && all(isnan, img.V)
        @test bits_equal(img, render(p; backend = :host_pool))
        @test_throws ErrorException trace(p; backend = :cpu)
        @test_throws ErrorException shadow(p; backend = :cpu)
    end
end
