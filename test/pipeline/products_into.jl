# Caller storage: every image product given `into`, an earlier result of the same shape,
# writes its maps into that result's arrays and returns them, bitwise the result without
# `into`, on threads, host pools and the test's device backend; a series of calls
# allocates no maps; a host array that a pool uploaded and a call then overwrote is
# uploaded again with its new values; `into` of another shape, type or with shared memory
# is an error.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare, precise
using BlackLightPlus.Products: Image, render, render_analytic, render_crossings, analytic_movie,
                               trace, StokesPlanes
using BlackLightPlus.Spacetime: KerrSchildCartesian
using BlackLightPlus.Matter: EquatorialDisk, EquatorialSpot, SynchrotronSurface
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Execution: PoolExecutor, DevicePool, HostDevice, resolve, resident
const Decks = Main.Decks

kerr(; a = 0.94, res = 16, inc = 17.0, fov = 20.0) =
    prepare(RunConfig(camera = CameraConfig(inclination_deg = inc, fov = fov, resolution = res));
            spacetime = KerrSchildCartesian(a), matter = nothing)
cone(a) = SynchrotronSurface(a; theta_s = deg2rad(75.0), b = (0.3, 0.4, 0.8), beta = 0.6,
                             phi_v = -1.7, sigma = 0.73)

# The Float64 arrays of a result, field by field, and their bits.
arrays(r::Image) = Any[r.I, r.Q, r.U, r.V]
arrays(r::NamedTuple) = Any[x for v in values(r) for x in (v isa Array{Float64} ? (v,) :
                            v isa AbstractVector{<:Array{Float64}} ? v : ())]
bits(r) = [Decks.bits(a) for a in arrays(r)]
# Whether `r` returns the arrays of `buf` themselves.
shares(r, buf) = all(x === y for (x, y) in zip(arrays(r), arrays(buf))) &&
                 length(arrays(r)) == length(arrays(buf))

const EXECS = (:cpu, :host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7),
               resolve(Decks.POOL_BACKEND))

# `f(exec, into)` with `into` the result of `g` (other values, the same shape): bitwise the
# result without `into` on every executor, sharing the arrays of `into`.
function check_into(f, g)
    for exec in EXECS
        ref = f(exec, nothing)
        buf = g()
        @test bits(buf) != bits(ref)
        r = f(exec, buf)
        @test bits(r) == bits(ref)
        @test shares(r, buf)
    end
end

@testset "caller storage" begin
    p = kerr(); p2 = kerr(inc = 40.0)
    disk = EquatorialDisk(0.94)

    @testset "render_analytic, disk" begin
        for obs in (true, false), bands in ((0, 1, 2), (1,))
            check_into((e, i) -> render_analytic(p, disk; bands, observables = obs, backend = e, into = i),
                       () -> render_analytic(p2, disk; bands, observables = !obs, backend = :cpu))
        end
    end

    @testset "render_analytic, surfaces" begin
        mats = (cone(0.94), EquatorialDisk(0.94))
        check_into((e, i) -> render_analytic(p, mats; bands = (0, 1), ro = Inf, backend = e, into = i),
                   () -> render_analytic(p2, mats; bands = (0, 1), ro = Inf, backend = :cpu))
    end

    @testset "render_crossings" begin
        cam(inc) = CameraConfig(inclination_deg = inc, fov = 16.0, resolution = 8)
        pc = prepare(precise(RunConfig(camera = cam(60.0))); spacetime = KerrSchildCartesian(0.9),
                     matter = nothing)
        pc2 = prepare(precise(RunConfig(camera = cam(30.0))); spacetime = KerrSchildCartesian(0.9),
                      matter = nothing)
        d = EquatorialDisk(0.9)
        check_into((e, i) -> render_crossings(pc, d; backend = e, into = i),
                   () -> render_crossings(pc2, d; backend = :cpu))
        pts = [(4.0, 3.0), (-5.5, 1.0), (2.0, -6.5)]
        check_into((e, i) -> render_crossings(pc, d; orders = (0, 1), coords = pts, backend = e, into = i),
                   () -> render_crossings(pc2, d; orders = (0, 1), coords = reverse(pts), backend = :cpu))
    end

    @testset "render: Stokes I, IQUV and stored paths" begin
        grid = Decks.uniform_deck()
        cfg(inc) = RunConfig(camera = CameraConfig(inclination_deg = inc, fov = 24.0, resolution = 12))
        pl(mode, inc) = prepare(cfg(inc); spacetime = KerrSchildCartesian(0.9), matter = grid, mode)
        for mode in (StokesI(), IQUV())
            check_into((e, i) -> render(pl(mode, 60.0); backend = e, into = i),
                       () -> render(pl(IQUV(), 30.0); backend = :cpu))
            q = pl(mode, 60.0)
            t = trace(q; backend = :cpu)
            ref = render(q; path = t, backend = :cpu)
            buf = render(pl(IQUV(), 30.0); backend = :cpu)
            r = render(q; path = t, backend = :cpu, into = buf)
            @test bits(r) == bits(ref) && shares(r, buf)
        end
        # the planes of an IQUV image as one vector of Stokes vectors
        P = StokesPlanes(zeros(3), zeros(3), zeros(3), zeros(3))
        P[2] = SVector(1.0, -2.0, 3.0, -0.0)
        @test P[2] === SVector(1.0, -2.0, 3.0, -0.0) && P.Q == [0.0, -2.0, 0.0] && size(P) == (3,)
        @test_throws BoundsError P[4] = SVector(1.0, 1.0, 1.0, 1.0)
        @test_throws DimensionMismatch StokesPlanes(zeros(3), zeros(3), zeros(2), zeros(3))
    end

    @testset "analytic_movie" begin
        spot = EquatorialSpot(EquatorialDisk(0.94); r_c = 7.0, omega = 0.06, width = 0.8)
        spot2 = EquatorialSpot(EquatorialDisk(0.94); r_c = 9.0, omega = 0.04, width = 1.2)
        check_into((e, i) -> analytic_movie(p, spot; times = (0.0, 20.0, 40.0), bands = (0, 1),
                                            ro = 1e4, backend = e, into = i),
                   () -> analytic_movie(p2, spot2; times = [5.0, 7.0, 9.0], bands = (0, 1), ro = 1e4,
                                        backend = :cpu))
    end

    @testset "no maps allocated" begin
        big = kerr(res = 256)
        radius(e, i) = render_analytic(big, disk; observables = false, backend = e, into = i)
        for e in (:cpu, :host_pool, resolve(Decks.POOL_BACKEND))
            r = radius(e, nothing)
            radius(e, r)
            fresh = @allocated radius(e, nothing)
            into = @allocated radius(e, r)
            @test fresh > 16 * 256^2 * 8 && into < 2^20
        end
        # an IQUV image on threads keeps its vector of Stokes vectors (32 bytes per pixel)
        for (mode, e, bound) in ((StokesI(), :cpu, 0), (StokesI(), :host_pool, 0),
                                 (IQUV(), :host_pool, 0), (IQUV(), :cpu, 32 * 96^2))
            q = prepare(RunConfig(camera = CameraConfig(fov = 24.0, resolution = 96));
                        spacetime = KerrSchildCartesian(0.9), matter = Decks.uniform_deck(), mode)
            img = render(q; backend = e)
            render(q; backend = e, into = img)
            fresh = @allocated render(q; backend = e)
            into = @allocated render(q; backend = e, into = img)
            @test fresh > 4 * 96^2 * 8 && into < bound + 2^20
        end
        spot = EquatorialSpot(EquatorialDisk(0.94); r_c = 7.0, omega = 0.06, width = 0.8)
        mv = analytic_movie(big, spot; times = (0.0, 10.0), bands = (0, 1), backend = :cpu)
        analytic_movie(big, spot; times = (0.0, 10.0), bands = (0, 1), backend = :cpu, into = mv)
        @test (@allocated analytic_movie(big, spot; times = (0.0, 10.0), bands = (0, 1), backend = :cpu,
                                         into = mv)) < 2^20
    end

    @testset "an uploaded host array overwritten through into is uploaded again" begin
        for exec in (resolve(:host_pool), PoolExecutor(DevicePool(Decks.hosts(2); lanes = 2)))
            dev = exec.pool.devices[end]
            for run_on in (exec, :cpu)
                r = render_analytic(p, disk; backend = exec)
                stale = copy(r.Ib[1])
                @test resident(exec, dev, r.Ib[1]) == stale                  # a run's upload
                render_analytic(p2, disk; backend = run_on, into = r)
                @test r.Ib[1] != stale
                @test resident(exec, dev, r.Ib[1]) == r.Ib[1]                # uploaded again
            end
            exec.pool === resolve(:host_pool).pool || close(exec.pool)
        end
    end

    @testset "into of another shape, type or with shared memory" begin
        r = render_analytic(p, disk; backend = :cpu)
        @test_throws ArgumentError render_analytic(p, disk; bands = (0, 1), backend = :cpu, into = r)
        @test_throws ArgumentError render_analytic(kerr(res = 12), disk; backend = :cpu, into = r)
        @test_throws ArgumentError render_analytic(p, disk; backend = :cpu, into = (I = r.I,))
        @test_throws ArgumentError render_analytic(p, disk; backend = :cpu,
                                                   into = merge(r, (I = Float32.(r.I),)))
        @test_throws ArgumentError render_analytic(p, disk; backend = :cpu,
                                                   into = merge(r, (I = r.Ib[1],)))
        @test_throws ArgumentError render_analytic(p, disk; backend = :cpu,
                                                   into = merge(r, (rs = [r.rs[1], vec(r.rs[1]), r.rs[3]],)))
        grid(res) = prepare(RunConfig(camera = CameraConfig(fov = 24.0, resolution = res));
                            spacetime = KerrSchildCartesian(0.9), matter = Decks.uniform_deck())
        img = render(grid(12); backend = :cpu)
        @test_throws ArgumentError render(grid(16); backend = :cpu, into = img)
        @test_throws ArgumentError render(grid(12); backend = :cpu, into = r)
        spot = EquatorialSpot(EquatorialDisk(0.94); r_c = 7.0, omega = 0.06, width = 0.8)
        mv = analytic_movie(p, spot; times = (0.0, 1.0), bands = (0, 1), backend = :cpu)
        @test_throws ArgumentError analytic_movie(p, spot; times = (0.0, 1.0, 2.0), bands = (0, 1),
                                                  backend = :cpu, into = mv)
        @test_throws ArgumentError render_crossings(p, disk; backend = :cpu, into = r)
    end
end
