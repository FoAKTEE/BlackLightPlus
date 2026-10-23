# March controls: the fixed-control march defers to the spacetime's rule and RK4 step in
# its own lane, a stored path replays a traced trajectory camera → source with its steps,
# and both report their step limits, spacetimes and tetrad normalisation.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Transfer: SpacetimeMarch, StoredPath, march_control, max_steps, geodesic_step,
                    next_state, transfer_spacetime, geodesic_spacetime, renorm_e1
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, MarchBudget, MarchStep, lane,
                     rk4_step, PlaneParallelCamera
const Decks = Main.Decks

@testset "march controls" begin
    st = KerrSchildCartesian(0.9)
    bin = SuperposedKerrSchild([0.5, 0.5], [0.3, -0.2], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)])
    b = MarchBudget(80.0, 1100.0, 12.0, 2500)
    rays = Decks.camera_rays(st, PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, fov = 24.0, npix = 4))

    @testset "fixed-control march" begin
        for (s, T) in ((st, Float64), (lane(st, Float32), Float32), (bin, Float64))
            ctl = SpacetimeMarch(s, lane(b, T))
            @test isbits(ctl) && max_steps(ctl) == 2500
            @test geodesic_spacetime(ctl) === s && renorm_e1(ctl) == (s isa SuperposedKerrSchild)
            for y64 in rays
                y = SVector{8,T}(y64)
                @test march_control(ctl, y, T(Inf), 7) === march_control(s, y, T(Inf), lane(b, T))
                @test geodesic_step(ctl, y, T(0.25), 7) === rk4_step(s, y, T(0.25))
                @test next_state(ctl, y, 7) === y
            end
        end
        # the lane is part of the control: a state of another lane is a MethodError
        ctl32 = SpacetimeMarch(lane(st, Float32), lane(b, Float32))
        @test_throws MethodError march_control(ctl32, rays[1], Inf, 1)
    end

    @testset "stored path" begin
        states, dls = Decks.sampled_trace(st, rays; C = 80.0, escape = 1100.0, nmax = 2500, R_em = 12.0)
        ys, hs = states[5], dls[5]
        n = length(ys)
        ctl = StoredPath(st, ys, hs)
        @test max_steps(ctl) == n && renorm_e1(ctl) == false && geodesic_spacetime(ctl) === st
        @test renorm_e1(StoredPath(bin, ys, hs))
        # the replay visits the stored states camera → source with the step behind each
        y = ys[end]; r_prev = 3.0
        for it in 1:n
            m = march_control(ctl, y, r_prev, it)
            @test m isa MarchStep{Float64} && !m.stop && m.R === r_prev
            @test m.h === (it < n ? hs[n-it] : 0.0)
            @test y === ys[n+1-it]
            y = geodesic_step(ctl, y, m.h, it)
        end
        @test march_control(ctl, y, r_prev, n + 1).stop
        # the stored steps are the fixed-control march's own steps
        ctlf = SpacetimeMarch(st, b)
        y = ys[end]; r_prev = Inf
        for it in 1:n-1
            m = march_control(ctlf, y, r_prev, it)
            @test m.h === hs[n-it]
            r_prev = m.R
            y = geodesic_step(ctlf, y, m.h, it)
            @test y === ys[n-it]
        end
        @test next_state(ctl, ys[1], n) === ys[1]
        @test !isbits(ctl) && Decks.device_isbits(ctl)
    end

    @testset "program spacetimes" begin
        prog = Decks.MarchProgram(st, SpacetimeMarch(lane(st, Float32), lane(b, Float32)), Decks.TorusDeck(),
                                  nothing, nothing)
        @test transfer_spacetime(prog) === st
        @test geodesic_spacetime(prog.control) === lane(st, Float32)
    end
end
