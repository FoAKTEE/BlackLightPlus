# Slow light: the fast-light limit of an identity series, the exact linear blend of two
# frames, the emission-time axis t_obs + X⁰ and its causal bracket order along a backward
# ray, read-only sampling of a held-in-memory series, and the two-frame window (end
# short-circuits, default epoch, advance, mesh guard, device form); the frame sequence at
# the sample's own time and the co-rotating plasma; the frame stack (the frame sequence on
# a uniform cadence, its time origin, its frames carried along a rotation, its radius and
# its device form).
using Test, BlackLightPlus, StaticArrays
using Adapt: Adapt
using BlackLightPlus.Data: AthenaKSnapshot, SnapshotSeries, synthetic_series, scale_density,
                bracketing_pair, get_frame, blend, FrameWindow, window, advance, SeriesProvider,
                UniformGrid, resample, with_locate, level_hash, BlockScan, FluidBlocks, FrameSequence,
                Corotating, FrameStack, frame_time, frame_blocks, retimed
using BlackLightPlus.Matter: PlasmaPrimitives, primitives, domain_radius
using BlackLightPlus.Spacetime: KerrSchildCartesian, PlaneParallelCamera
const Decks = Main.Decks

# One block on [-4, 4]³ of 4³ cells with spatially varying primitives.
function tiny_snap(; rho0 = 2.0)
    data = Array{Float32}(undef, 8, 4, 4, 4, 1)
    for k in 1:4, j in 1:4, i in 1:4
        data[:, i, j, k, 1] .= (rho0 + 0.1 * (i + j + k), 0.01i, 0.02j, 0.03k, 0.5 + 0.05i,
                                0.10, 0.20, 0.30)
    end
    return AthenaKSnapshot(data, [-4.0, -4.0, -4.0][:, :], [4.0, 4.0, 4.0][:, :],
                           [-3.0, -3.0, -3.0][:, :], [2.0, 2.0, 2.0][:, :], [0], zeros(Int, 3, 1))
end

const X = SVector(0.5, -0.3, 1.1)
sample_t(ser, t) = primitives(SeriesProvider(ser), SVector(t, X...))

@testset "slow light" begin

@testset "a: fast-light limit (identity series, bit-identical)" begin
    snap = tiny_snap()
    ser = synthetic_series(snap; times = [0.0, 10.0, 25.0])
    base = primitives(snap, SVector(0.0, X...))
    for t in (-5.0, 0.0, 4.2, 10.0, 17.3, 25.0, 99.0)
        @test sample_t(ser, t) === base
    end
end

@testset "b: exact two-frame blend and an analytic sine" begin
    snap = tiny_snap()
    eps = 0.2; omega = 2π / 40; ts = [0.0, 10.0]
    ser = synthetic_series(snap; times = ts,
                           evolve = (s, t) -> scale_density(s, 1 + eps * sin(omega * t)))
    A = get_frame(ser, 1); B = get_frame(ser, 2)
    pA = primitives(A, SVector(0.0, X...)); pB = primitives(B, SVector(0.0, X...))
    rho0 = primitives(snap, SVector(0.0, X...)).rho
    fA = 1 + eps * sin(omega * ts[1]); fB = 1 + eps * sin(omega * ts[2])
    maxrel = 0.0
    for t in range(0.0, -10.0; length = 9)
        ia, w = bracketing_pair(ser, ser.t_obs + t)
        @test ia == 1
        s = sample_t(ser, t)
        @test s.rho == (1.0 - w) * pA.rho + w * pB.rho
        @test s === (w == 0.0 ? pA : w == 1.0 ? pB : blend(pA, pB, w))
        @test s.eint == primitives(snap, SVector(0.0, X...)).eint
        approx = ((1.0 - w) * fA + w * fB) * rho0
        maxrel = max(maxrel, abs(s.rho - approx) / approx)
    end
    @test maxrel < 1e-6                                       # Float32 storage
    @test bracketing_pair(ser, -100.0) == (1, 0.0)
    @test bracketing_pair(ser, 100.0) == (1, 1.0)
    @test bracketing_pair(synthetic_series(snap; times = [3.0]), 7.0) == (1, 0.0)
end

@testset "c: brackets never advance along a backward ray" begin
    st = KerrSchildCartesian(0.9)
    cam = PlaneParallelCamera(60.0, 1000.0, 30.0, 8)
    rays = Decks.camera_rays(st, cam)
    states, _ = Decks.sampled_trace(st, rays[30:30]; C = 80.0, escape = 1100.0, nmax = 6000)
    ys = states[1]                                            # source to camera
    x0 = [ys[i][1] for i in length(ys):-1:1]                  # march order, camera first
    @test x0[1] == 0.0 && all(<=(0.0), diff(x0)) && x0[end] < -100.0
    times = collect(range(x0[end] - 10.0, 10.0; length = 6))
    ser = synthetic_series(tiny_snap(); times = times, t_obs = 0.0)
    ias = [bracketing_pair(ser, ser.t_obs + t)[1] for t in x0]
    @test issorted(ias; rev = true)
end

@testset "f: physical dump-time axis" begin
    snap = tiny_snap()
    ts = [12000.0, 12500.0, 13000.0]
    fac(t) = 1.0 + (t - ts[1]) / 1000.0
    ser = synthetic_series(snap; times = ts, evolve = (s, t) -> scale_density(s, fac(t)))
    @test ser.t_obs == ts[end]
    rho(i) = primitives(get_frame(ser, i), SVector(0.0, X...)).rho
    @test sample_t(ser, 0.0).rho == rho(3)
    @test sample_t(ser, -500.0).rho == rho(2)
    @test sample_t(ser, -750.0).rho == 0.5 * rho(1) + 0.5 * rho(2)
    @test sample_t(ser, -2145.0).rho == rho(1)
    @test issorted([sample_t(ser, t).rho for t in 0.0:-100.0:-1200.0]; rev = true)
    @test_throws ErrorException synthetic_series(snap; times = ts, t_obs = 0.0)
    @test_throws ErrorException synthetic_series(snap; times = reverse(ts))
    @test_throws ErrorException get_frame(ser, 4)
end

@testset "g: held-in-memory sampling is read only" begin
    snap = tiny_snap()
    ser = synthetic_series(snap; times = [0.0, 10.0, 25.0],
                           evolve = (s, t) -> scale_density(s, 1.0 + 0.01 * t))
    ord0 = copy(ser.order)
    for t in (0.0, 4.2, 12.0, 20.0, 24.9)
        sample_t(ser, t)
    end
    @test ser.order == ord0
    tsamp = collect(range(0.0, 25.0; length = 512))
    serial = [sample_t(ser, t).rho for t in tsamp]
    threaded = similar(serial)
    Threads.@threads for i in eachindex(tsamp)
        threaded[i] = sample_t(ser, tsamp[i]).rho
    end
    @test threaded == serial
    alloc(p, x) = (primitives(p, x); @allocated primitives(p, x))
    @test alloc(SeriesProvider(ser), SVector(-3.0, X...)) == 0
    @test domain_radius(SeriesProvider(ser)) == domain_radius(snap)
end

@testset "i: window ends short-circuit to one frame" begin
    sA = Decks.amr_deck(); sB = Decks.amr_deck(scale = 1.5f0)
    gA = Decks.uniform_deck(); gB = UniformGrid(gA.data .* Float32[1.5, 1, 1, 1, 1, 1, 1, 1], 12.0)
    for (a, b) in ((sA.blocks, sB.blocks), (gA, gB))
        win = FrameWindow(a, b, 0.0, 10.0)                    # t_obs = 10: X⁰ = -10 is t_a
        @test win.t_obs == 10.0 && win.inv_dt == 0.1
        for p in (SVector(0.5, -0.3, 1.1), SVector(-3.9, 3.9, 0.0), SVector(5.2, 2.3, -2.4))
            pa = primitives(a, SVector(0.0, p...)); pb = primitives(b, SVector(0.0, p...))
            @test primitives(win, SVector(-10.0, p...)) === pa
            @test primitives(win, SVector(-30.0, p...)) === pa
            @test primitives(win, SVector(0.0, p...)) === pb
            @test primitives(win, SVector(5.0, p...)) === pb
            @test primitives(win, SVector(-7.5, p...)) === blend(pa, pb, 0.25)
            @test blend(pa, pb, 1.0) === PlasmaPrimitives(0.0 * pa.rho + 1.0 * pb.rho,
                0.0 .* pa.uu .+ 1.0 .* pb.uu, 0.0 * pa.eint + 1.0 * pb.eint,
                0.0 .* pa.bb .+ 1.0 .* pb.bb, pa.inside || pb.inside)
        end
        # an explicit epoch, and the fast-light limit of an open window
        @test primitives(FrameWindow(a, b, 0.0, 10.0; t_obs = 7.5), SVector(0.0, X...)) ===
              blend(primitives(a, SVector(0.0, X...)), primitives(b, SVector(0.0, X...)), 0.75)
        win0 = FrameWindow(a, b, 4.0, Inf)
        @test win0.inv_dt == 0.0 && win0.t_obs == 4.0
        @test all(primitives(win0, SVector(t, X...)) === primitives(a, SVector(0.0, X...))
                  for t in (-1e4, -3.0, 0.0, 50.0))
        @test domain_radius(FrameWindow(a, b, 0.0, 10.0)) == domain_radius(a)
    end
    @test_throws ErrorException FrameWindow(gA, resample(sA, 12.0, 16), 0.0, 1.0)
    @test_throws ErrorException FrameWindow(sA.blocks, gA, 0.0, 1.0)
    @test_throws ErrorException FrameWindow(sA.blocks, with_locate(sB.blocks, BlockScan()), 0.0, 1.0)
end

@testset "k: windows over a series and their advance" begin
    snap = Decks.amr_deck()
    ser = synthetic_series(snap; times = [0.0, 10.0, 20.0, 30.0],
                           evolve = (s, t) -> scale_density(s, 1 + t / 100))
    win = window(ser, 1)
    @test win.a === get_frame(ser, 1).blocks && win.b === get_frame(ser, 2).blocks
    @test win.t_a == 0.0 && win.inv_dt == 0.1 && win.t_obs == 10.0
    @test window(ser, 1; t_obs = 4.0).t_obs == 4.0
    win2 = advance(win, ser, 2)
    @test win2.a === win.b && win2.b === get_frame(ser, 3).blocks
    @test win2.t_a == 10.0 && win2.inv_dt == 1.0 / (20.0 - 10.0) && win2.t_obs == 20.0
    endwin = advance(advance(win2, ser, 3), ser, 4)
    @test endwin.inv_dt == 0.0 && endwin.a === endwin.b
    # resampled windows: one resample per frame, the geometry of the promoted frame
    winF = window(ser, 1, 12.0, 16)
    @test winF.a isa UniformGrid && winF.a.n == 16 && winF.a.box_half == 12.0
    @test winF.b.data == resample(get_frame(ser, 2), 12.0, 16).data
    winF2 = advance(winF, ser, 2)
    @test winF2.a === winF.b && (winF2.b.n, winF2.b.xo, winF2.b.inv_dx) == (winF.b.n, winF.b.xo, winF.b.inv_dx)
    @test winF2.b.data == resample(get_frame(ser, 3), 12.0, 16).data
    @test advance(winF, ser, 4).inv_dt == 0.0
    ident = synthetic_series(snap; times = [0.0, 10.0])
    wI = window(ident, 1, 12.0, 8)
    @test wI.a === wI.b
end

@testset "device form and hot path" begin
    snap = Decks.amr_deck()
    ser = synthetic_series(snap; times = [0.0, 10.0], evolve = (s, t) -> scale_density(s, 1 + t))
    hb = with_locate(snap.blocks, level_hash(snap.blocks.lo, snap.blocks.hi, snap.levels,
                                             snap.logical_locations))
    hwin = FrameWindow(hb, hb, 0.0, 1.0)
    @test Adapt.adapt(Decks.DeviceLike(), hwin).a.locate === BlockScan()
    alloc(p, x) = (primitives(p, x); @allocated primitives(p, x))
    for w in (window(ser, 1), window(ser, 1, 12.0, 16), hwin)
        @test Decks.device_isbits(w)
        for t in (-20.0, -5.0, 0.0)
            @test alloc(w, SVector(t, 5.0, 1.0, 0.3)) == 0
        end
    end
    @test !Decks.device_isbits(SeriesProvider(ser))
end


@testset "frame sequence and co-rotating plasma" begin
    a = tiny_snap(); b = tiny_snap(rho0 = 5.0)
    fs = FrameSequence([0.0, 10.0, 20.0], [a.blocks, b.blocks, b.blocks])
    @test domain_radius(fs) == domain_radius(a.blocks)
    p(t) = primitives(fs, SVector(t, X...))
    @test p(0.0) === primitives(a.blocks, SVector(0.0, X...)) && p(-3.0) === p(0.0)
    @test p(10.0) === primitives(b.blocks, SVector(10.0, X...))
    @test p(15.0) === primitives(b.blocks, SVector(15.0, X...))   # the same frame twice
    @test p(25.0) === p(20.0)
    sa = primitives(a.blocks, SVector(0.0, X...)); sb = primitives(b.blocks, SVector(0.0, X...))
    @test p(2.5).rho === blend(sa, sb, 0.25).rho
    # the series form is the imager's series at epoch t_obs = 0 (here the first frame time)
    ser = synthetic_series(a; times = [0.0, 10.0], evolve = (s, t) -> t == 0 ? s : b, t_obs = 0.0)
    fq = FrameSequence(ser)
    @test fq.frames[1] === a.blocks && fq.frames[2] === b.blocks
    @test all(primitives(fq, SVector(t, X...)) == sample_t(ser, t) for t in (0.0, 3.7, 9.9))
    @test_throws ArgumentError FrameSequence([0.0, 0.0], [a.blocks, a.blocks])
    @test_throws ArgumentError FrameSequence(Float64[], typeof(a.blocks)[])

    # co-rotation: at x⁰ the plasma of time 0 at R(−ωx⁰)x with its vectors rotated by R(ωx⁰)
    snap = Decks.amr_deck()
    ω = 0.03
    c = Corotating(snap, ω)
    @test domain_radius(c) == domain_radius(snap) && Decks.device_isbits(Corotating(snap.blocks, ω))
    x = SVector(0.0, 4.2, -2.5, 1.3)
    @test primitives(c, x) === primitives(snap, x)                 # θ = 0
    for t in (7.0, -40.0, 123.4)
        θ = ω * t
        R = @SMatrix [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
        xr = R * SVector(x[2], x[3], x[4])
        q = primitives(c, SVector(t, xr...))
        ref = primitives(snap, SVector(t, (R' * xr)...))
        @test q.rho ≈ ref.rho rtol = 1e-12
        @test q.eint ≈ ref.eint rtol = 1e-12
        @test q.uu ≈ R * ref.uu rtol = 1e-12
        @test q.bb ≈ R * ref.bb rtol = 1e-12
    end
    @test !primitives(c, SVector(5.0, 30.0, 0.0, 0.0)).inside
end

@testset "frame stack" begin
    a = tiny_snap(); b = tiny_snap(rho0 = 5.0); c = tiny_snap(rho0 = 7.0)
    fq = FrameSequence([0.0, 10.0, 20.0], [a.blocks, b.blocks, c.blocks])
    st = FrameStack([a, b, c], 0.0, 10.0)
    @test st.n == 3 && st.nmb == 1 && frame_time(st, 3) == 20.0
    @test domain_radius(st) == domain_radius(a.blocks)
    for t in (-4.0, 0.0, 2.5, 9.99, 10.0, 13.0, 20.0, 31.0)
        q = primitives(st, SVector(t, X...)); r = primitives(fq, SVector(t, X...))
        @test q.rho ≈ r.rho rtol = 1e-14
        @test q.uu ≈ r.uu rtol = 1e-14
        @test q.eint ≈ r.eint rtol = 1e-14
    end
    @test primitives(st, SVector(10.0, X...)) === primitives(b.blocks, SVector(0.0, X...))
    @test primitives(st, SVector(20.0, X...)) === primitives(c.blocks, SVector(0.0, X...))
    @test all(primitives(frame_blocks(st, k), SVector(0.0, X...)) ===
              primitives((a, b, c)[k].blocks, SVector(0.0, X...)) for k in 1:3)
    # the time origin: x⁰ = 0 is series time t_shift
    s5 = retimed(st; t_shift = 5.0)
    @test all(primitives(s5, SVector(t - 5.0, X...)) === primitives(st, SVector(t, X...))
              for t in (1.0, 12.5, 17.0))
    # the radius
    @test !primitives(retimed(st; r_max = 1.0), SVector(0.0, X...)).inside
    @test domain_radius(retimed(st; r_max = 1.0)) == 1.0
    # a loader builds the same stack
    sl = FrameStack(k -> (a, b, c)[k], 3, 0.0, 10.0)
    @test sl.blocks.data == st.blocks.data
    @test_throws ArgumentError FrameStack([a, Decks.amr_deck()], 0.0, 1.0)
    @test_throws ArgumentError FrameStack([a, b], 0.0, 0.0)

    # carried frames: each frame is the co-rotating plasma of its own time, blended
    snap = Decks.amr_deck()
    ω = 0.03
    cs = FrameStack([snap, snap], 4.0, 10.0; omega = ω)
    x = SVector(0.0, 4.2, -2.5, 1.3)
    @test primitives(cs, SVector(4.0, x[2], x[3], x[4])) === primitives(snap, x)   # a frame time
    for t in (6.5, 9.0, 13.0)
        y = SVector(t, x[2], x[3], x[4])
        w = (t - 4.0) / 10.0
        ra = primitives(Corotating(snap, ω), SVector(t - 4.0, x[2], x[3], x[4]))
        rb = primitives(Corotating(snap, ω), SVector(t - 14.0, x[2], x[3], x[4]))
        q = primitives(cs, y); r = blend(ra, rb, w)
        @test q.rho ≈ r.rho rtol = 1e-13
        @test q.uu ≈ r.uu rtol = 1e-13
        @test q.bb ≈ r.bb rtol = 1e-13
    end
    # the device form: isbits, no allocation
    @test Decks.device_isbits(cs)
    alloc(p, y) = (primitives(p, y); @allocated primitives(p, y))
    @test alloc(cs, SVector(7.0, 5.0, 1.0, 0.3)) == 0
    @test alloc(st, SVector(7.0, 1.0, 1.0, 0.3)) == 0
end

end
