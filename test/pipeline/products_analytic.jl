# The exact-Kerr disk products: the per-band maps and their sum, landings outside the
# horizon, executor independence, the lensing-band layers, the host epilogues (band sums,
# constant maps, Stokes planes) bitwise their broadcast forms, and the argument checks.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare
using BlackLightPlus.Products: render_analytic, photon_ring_layers, AnalyticProgram, BandSum, Fill,
                               Split
import BlackLightPlus.Products: analytic_outputs, band_sums, filled_maps
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, kerr_horizon
using BlackLightPlus.Matter: EquatorialDisk
using BlackLightPlus.Execution: PoolExecutor, ThreadedExecutor, foreach_item!, resolve
const Decks = Main.Decks

plan(; a = 0.9, res = 12, inc = 60.0, st = KerrSchildCartesian(a)) =
    prepare(RunConfig(camera = CameraConfig(inclination_deg = inc, fov = 24.0, resolution = res));
            spacetime = st, matter = nothing)
const FIELDS = (:I, :Ib, :rs, :ts, :phis, :redshift_sign)
# the band sum as a broadcast, band after band from zero
bsum(maps) = (I = zero(maps[1]); for m in maps; I .+= m; end; I)

@testset "exact-Kerr disk" begin
    @testset "render_analytic" begin
        p = plan()
        disk = EquatorialDisk(0.9)
        r = render_analytic(p, disk; backend = :cpu)
        @test r.bands == [0, 1, 2] && length(r.Ib) == 3 && r.plan === p
        @test all(size(m) == (12, 12) for f in FIELDS[2:end] for m in getfield(r, f))
        @test r.I == r.Ib[1] + r.Ib[2] + r.Ib[3] && Decks.bits(r.I) == Decks.bits(bsum(r.Ib))
        for b in 1:3
            off = isnan.(r.rs[b])
            @test all(iszero, r.Ib[b][off]) && all(isnan, r.ts[b][off]) && all(isnan, r.phis[b][off])
            @test all(>(kerr_horizon(0.9)), r.rs[b][.!off])
            @test all(in((-1.0, 1.0)), r.redshift_sign[b][.!off])
        end
        @test count(>(0), r.Ib[1]) > 30 && count(>(0), r.Ib[2]) > 0
        @test count(!isnan, r.rs[3]) <= count(!isnan, r.rs[2]) <= count(!isnan, r.rs[1])
        # the program writes quantity q of band b to output (q - 1) K + b, which the maps share
        prog = AnalyticProgram(p.pixels, p.theta_o, 0.9, p.camera.d, disk, (0, 1, 2))
        outs = analytic_outputs(3, 144)
        foreach(k -> prog(k, k, outs...), 1:144)
        @test all(Decks.bits(reshape(outs[(q - 1) * 3 + b], 12, 12)) == Decks.bits(getfield(r, f)[b])
                  for (q, f) in enumerate(FIELDS[2:end]) for b in 1:3)
        @test all(m isa Matrix{Float64} for f in FIELDS[2:end] for m in getfield(r, f))
        @test_throws ArgumentError prog(1, 1, outs[1:14]...)
        for exec in (:host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
            q = render_analytic(p, disk; backend = exec)
            @test all(Decks.bits(getfield(q, f)) == Decks.bits(getfield(r, f)) for f in FIELDS)
        end
        fast = render_analytic(p, disk; observables = false, backend = :cpu)
        @test all(Decks.bits(getfield(fast, f)) == Decks.bits(getfield(r, f)) for f in (:I, :Ib, :rs, :redshift_sign))
        @test all(all(isnan, m) for m in fast.ts) && all(all(isnan, m) for m in fast.phis)
        @test all(Decks.bits(m) == Decks.bits(fill(NaN, 12, 12)) for m in vcat(fast.ts, fast.phis))
        @test length(fast.ts) == 3 && all(m isa Matrix{Float64} for m in vcat(fast.ts, fast.phis))
        @test length(unique(objectid, vcat(fast.ts, fast.phis))) == 6
        fastp = render_analytic(p, disk; observables = false, backend = Decks.POOL_BACKEND)
        @test Decks.CROSS_HARDWARE ? maximum(abs.(fastp.I .- fast.I)) <= 1e-9 * maximum(fast.I) :
              Decks.bits(fastp.I) == Decks.bits(fast.I)
        one = render_analytic(p, disk; bands = (1,), backend = :cpu)
        @test Decks.bits(one.Ib[1]) == Decks.bits(r.Ib[2]) && one.bands == [1]
        far = render_analytic(p, disk; ro = 2000.0, backend = :cpu)
        @test isapprox(sum(far.I), sum(r.I); rtol = 1e-2)
        @test_throws ArgumentError render_analytic(p, EquatorialDisk(0.5); backend = :cpu)
        bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        @test_throws ArgumentError render_analytic(plan(st = bin), disk; backend = :cpu)
    end

    @testset "photon_ring_layers" begin
        p = plan()
        disk = EquatorialDisk(0.9)
        l = photon_ring_layers(p, disk; dx = 0.25, backend = :cpu)
        @test size(l.I) == (l.N, l.N) && size(l.grid) == (l.N^2, 2) && l.N == 96
        @test l.bands == [0, 1, 2] && l.dx == 0.25 && l.box == 12.0
        @test all(l.masks[b] == (l.Ib[b] .> 0) for b in 1:3)
        @test count(l.masks[1]) > count(l.masks[2]) > 0 && count(l.masks[3]) <= count(l.masks[2])
        @test l.I == l.Ib[1] + l.Ib[2] + l.Ib[3]
        f = photon_ring_layers(p, disk; dx = 0.25, fudge = 2.0, backend = :host_pool)
        @test Decks.bits(f.Ib) == Decks.bits(l.Ib)
        @test f.I == l.Ib[1] + 2.0 .* l.Ib[2] + 2.0 .* l.Ib[3]
        @test l.dxs == [0.25, 0.25, 0.25] && all(isempty(pt.α) for pt in l.points)
        # per-order resolution: finer grids in the thin bands, deposited on the common grid
        q = photon_ring_layers(p, disk; dx = (0.25, 0.125, 0.0625), backend = :cpu)
        @test Decks.bits(q.Ib[1]) == Decks.bits(l.Ib[1]) && q.dxs == [0.25, 0.125, 0.0625]
        @test length(q.points[2].α) > 3 * count(l.masks[2]) && length(q.points[3].α) > 3 * count(l.masks[3])
        h(N) = 24.0 / (N - 1)
        Nf = 2 * round(Int, 12.0 / 0.125)
        @test sum(q.Ib[2]) * h(q.N)^2 ≈ sum(filter(isfinite, q.points[2].I)) * h(Nf)^2 rtol = 1e-12
        @test isapprox(sum(q.Ib[2]), sum(l.Ib[2]); rtol = 0.05)            # the same ring flux
        @test_throws ArgumentError photon_ring_layers(p, disk; dx = (0.25, 0.1), backend = :cpu)
    end

    @testset "host epilogues: bitwise the broadcasts" begin
        # signed zeros, NaN and infinities among the band values: 0 + (-0) is +0
        vals = [0.0, -0.0, 1.5, -2.25, NaN, Inf, -Inf, 1e-310, 3.0e300, -3.0e300]
        Ib = [reshape([vals[mod1(7k + 3b, 10)] for k in 1:63], 7, 9) for b in 1:3]
        Qb = [-m for m in Ib]
        I, Q = band_sums((Ib, Qb), (7, 9))
        @test I isa Matrix{Float64} && Decks.bits(I) == Decks.bits(bsum(Ib)) &&
              Decks.bits(Q) == Decks.bits(bsum(Qb))
        @test any(i -> signbit(bsum(Ib[1:1])[i]) != signbit(Ib[1][i]), eachindex(I))  # a -0 became +0
        @test Decks.bits(only(band_sums((Matrix{Float64}[],), (2, 2)))) == Decks.bits(zeros(2, 2))
        # maps of several epilogue chunks (2^14 items each)
        big = [reshape([vals[mod1(k * b + 5b, 10)] for k in 1:201 * 199], 201, 199) for b in 1:3]
        @test Decks.bits(only(band_sums((big,), (201, 199)))) == Decks.bits(bsum(big))
        @test all(Decks.bits(m) == Decks.bits(fill(NaN, 201, 199)) for m in filled_maps(NaN, 2, (201, 199)))
        nan = filled_maps(NaN, 4, (5, 3))
        @test length(nan) == 4 && all(Decks.bits(m) == Decks.bits(fill(NaN, 5, 3)) for m in nan)
        @test isempty(filled_maps(0.0, 0, (5, 3)))
        S = [SVector(vals[k], -vals[k], vals[mod1(k + 1, 10)], 2.0) for k in 1:10]
        planes = ntuple(_ -> Vector{Float64}(undef, 10), 4)
        foreach_item!(ThreadedExecutor(chunk = 3), Split(S), 1:10; outs = planes)
        @test all(Decks.bits(planes[c]) == Decks.bits([s[c] for s in S]) for c in 1:4)
        # the programs on a pool and on the test's device backend: one result (a GPU's sum
        # may carry another NaN payload; every other value is the correctly rounded sum)
        nanbits(a) = (isnan.(a), Decks.bits(filter(!isnan, a)))
        for (exec, eq) in ((PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7), Decks.bits),
                           (resolve(Decks.POOL_BACKEND), Decks.CROSS_HARDWARE ? nanbits : Decks.bits))
            o = ntuple(_ -> Vector{Float64}(undef, 63), 2)
            foreach_item!(exec, BandSum(map(Tuple, (vec.(Ib), vec.(Qb)))), 1:63; outs = o)
            @test eq(o[1]) == eq(vec(I)) && eq(o[2]) == eq(vec(Q))
            f = ntuple(_ -> Vector{Float64}(undef, 63), 2)
            foreach_item!(exec, Fill(NaN), 1:63; outs = f)
            @test all(Decks.bits(v) == Decks.bits(fill(NaN, 63)) for v in f)
            s = ntuple(_ -> Vector{Float64}(undef, 10), 4)
            foreach_item!(exec, Split(S), 1:10; outs = s)
            @test all(Decks.bits(s[c]) == Decks.bits(planes[c]) for c in 1:4)
        end
        @test Decks.device_isbits(BandSum(((Ib[1], Ib[2]),))) && Decks.device_isbits(Split(S)) &&
              isbits(Fill(NaN))
        @test_throws ArgumentError BandSum(((Ib[1],),))(1, 1, I, Q)
        @test_throws ArgumentError Split(S)(1, 1, planes[1:3]...)
        # sizes: as the broadcast forms, a band map of another size is an error, and an item
        # outside the maps is a BoundsError, not a read past them
        @test_throws DimensionMismatch BandSum(((Ib[1], zeros(9, 7)),))
        @test_throws DimensionMismatch band_sums((Ib,), (9, 7))
        @test_throws DimensionMismatch band_sums((Ib, [Ib[1], zeros(7, 8)]), (7, 9))
        @test_throws BoundsError BandSum(((vec(Ib[1]), vec(Ib[2])),))(64, 1, zeros(64))
        @test_throws BoundsError Split(S)(11, 1, planes...)
    end
end
