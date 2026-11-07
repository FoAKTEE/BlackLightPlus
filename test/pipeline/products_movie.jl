# Closed-form movies: frames equal to the landing-wise intensity, the light curve, fast and
# slow light, the observer at infinity, executor independence and the argument checks.
using Test, BlackLightPlus
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare
using BlackLightPlus.Products: analytic_movie
using BlackLightPlus.Spacetime: KerrSchildCartesian, pixel_prep, band_landing
using BlackLightPlus.Matter: EquatorialDisk, EquatorialSpot, intensity
using BlackLightPlus.Execution: PoolExecutor
const Decks = Main.Decks

@testset "exact-Kerr movies" begin
    a = 0.94
    p = prepare(RunConfig(camera = CameraConfig(inclination_deg = 17.0, fov = 24.0, resolution = 24));
                spacetime = KerrSchildCartesian(a), matter = nothing)
    spot = EquatorialSpot(EquatorialDisk(a); r_c = 7.0, omega = 0.06, width = 0.8)
    times = (0.0, 20.0, 40.0)
    m = analytic_movie(p, spot; times = times, bands = (0, 1), ro = 1e4, backend = :cpu)
    @test size(m.frames) == (24, 24, 3) && m.times == collect(times) && m.bands == [0, 1]
    xs = p.pixels.centres
    k = 7 * 24 + 15
    α, β = xs[(k - 1) ÷ 24 + 1], xs[(k - 1) % 24 + 1]
    pp = pixel_prep(α, β, p.theta_o, a; ro = 1e4)
    for (j, t) in enumerate(times)
        v = 0.0
        for n in (0, 1)
            l = band_landing(pp, n)
            i = intensity(spot, l, t)
            i > 0 && (v += i)
        end
        @test m.frames[k + (j - 1) * 576] == v
    end
    @test m.lightcurve ≈ [sum(m.frames[:, :, j]) * (24.0 / 24)^2 for j in 1:3]
    @test count(>(0), m.frames) > 50
    @test length(unique(round.(m.lightcurve; sigdigits = 6))) == 3       # the spot moves
    fast = analytic_movie(p, spot; times = times, bands = (0, 1), ro = 1e4, slow = false, backend = :cpu)
    @test all(iszero, fast.ts[1]) && fast.frames != m.frames
    inf = analytic_movie(p, spot; times = times, bands = (0,), ro = Inf, backend = :cpu)
    @test all(isfinite, inf.frames) && count(>(0), inf.frames) > 50
    for exec in (:host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
        q = analytic_movie(p, spot; times = times, bands = (0, 1), ro = 1e4, backend = exec)
        @test Decks.bits(q.frames) == Decks.bits(m.frames)
    end
    mp = analytic_movie(p, spot; times = times, bands = (0, 1), ro = Inf, backend = Decks.POOL_BACKEND)
    mc = analytic_movie(p, spot; times = times, bands = (0, 1), ro = Inf, backend = :cpu)
    @test Decks.CROSS_HARDWARE ? maximum(abs.(mp.frames .- mc.frames)) <= 1e-9 * maximum(mc.frames) :
          Decks.bits(mp.frames) == Decks.bits(mc.frames)
    @test_throws ArgumentError analytic_movie(p, EquatorialSpot(EquatorialDisk(0.5)); times = (0.0,),
                                              backend = :cpu)
end
