# The exact accumulators of the Monte Carlo: binning rules and the spectrum accessors,
# exact merging, the child buffer's slots, overflow and genealogy streams, and the
# executor contract: a program that writes through atomics gives the same limbs, the
# same scalars and the same child set under threads of any chunking and on a pool of
# 1, 2, 3 or 5 devices; the thin and scattering sphere runs are identical everywhere.
using Test, BLP, Random
using StaticArrays
using BLP.Numerics: MCRng, MCStreamLayout, child_rng, stream_id, primary_rng, ME, CL, HPL,
                    SIGMA_THOMSON, xsum_round
using BLP.Matter: UniformSphere
using BLP.MonteCarlo: Superphoton, HotcrossTable, NW, NT
using BLP.Execution: ThreadedExecutor, PoolExecutor, foreach_item!, ExactBins, ExactScalars,
                     ChildBuffer, PolarBands, InclinationBands, Spectrum, band, gather!,
                     nu_centers, nuLnu, nuLnu_total, nuLnu_solid, total_power, total_weight,
                     mean_energy, add!, scalar, addends, capacity, overflow, children
import BLP.Execution

const DK = Main.Decks
const MEC2 = ME * CL * CL

energy_at(nu) = nu * HPL / MEC2

# Writes item-keyed photons, scalars and children through the accumulators.
struct AccProgram{B,S,C}
    bins::B
    scalars::S
    buf::C
end
BLP.Execution.Adapt.@adapt_structure AccProgram
function (p::AccProgram)(item, i)
    r = primary_rng(77, item)
    u = BLP.Numerics.philox_rand(5, item, 0)
    E = energy_at(exp(log(1.0e9) + 12.0 * u))
    d = SVector(sin(3.0 * item), cos(5.0 * item), sin(item) * cos(7.0 * item))
    Execution.record!(p.bins, E, 1.0 + item / 7.0, d)
    add!(p.scalars, 1, exp(-item / 100.0))
    add!(p.scalars, 2, sin(Float64(item)))
    item % 3 == 0 && Execution.spawn!(p.buf, Superphoton(SVector(0.0, item, 0.0, 0.0),
                                                         SVector(-E, 0.0, 0.0, E), 1.0), r, item % 5)
    return nothing
end

function run_acc(exec, n; cap = n)
    p = AccProgram(ExactBins(InclinationBands((30.0, 90.0, 100.0), 20.0); nu_min = 1.0e8,
                             nu_max = 1.0e24, n_ebins = 50),
                   ExactScalars(2), ChildBuffer(Superphoton, cap, MCStreamLayout(n, 10, 2)))
    foreach_item!(exec, p, collect(1:n))
    gather!(p.bins); gather!(p.scalars); gather!(p.buf)
    return p
end

childset(buf) = (c = children(buf); sort(collect(zip(stream_id.(c[2]), getfield.(c[1], :x)))))

@testset "Execution: Monte Carlo accumulators" begin
    @testset "binning rules" begin
        b = ExactBins(PolarBands(6); nu_min = 1.0e8, nu_max = 1.0e16, n_ebins = 80)
        E = energy_at(3.0e12)
        ie = floor(Int, (log(3.0e12) - log(1.0e8)) / ((log(1.0e16) - log(1.0e8)) / 80)) + 1
        Execution.record!(b, E, 2.0, SVector(0.0, 1.0, -0.2))     # |cos θ| ≈ 0.2: band 6
        Execution.record!(b, E, 3.0, SVector(0.1, 0.0, -5.0))     # near the pole: band 1
        Execution.record!(b, E, 1.0, SVector(0.0, 0.0, 0.0))      # no direction: band 6
        Execution.record!(b, -E, 1.0, SVector(1.0, 0.0, 0.0))     # E ≤ 0: dropped
        Execution.record!(b, energy_at(2.0e16), 1.0, SVector(1.0, 0.0, 0.0))   # off the grid
        @test b.count[6, ie] == 2 && b.count[1, ie] == 1 && sum(b.count) == 3
        s = Spectrum(b)
        @test s.sumw[6, ie] === 3.0 && s.sumw[1, ie] === 3.0 && s.sumwE[1, ie] === 3.0 * E
        @test sum(s.dOmega) ≈ 4π && s.total_row == 0
        ib = ExactBins(InclinationBands((17.0, 50.0, 60.0), 10.0))
        th = deg2rad(55.0)
        Execution.record!(ib, E, 1.0, SVector(sin(th), 0.0, cos(th)))  # bands 2 and 3
        Execution.record!(ib, E, 1.0, SVector(0.0, 0.0, 0.0))          # no direction: dropped
        @test vec(sum(ib.count; dims = 2)) == [0, 1, 1, 1]
        si = Spectrum(ib)
        @test si.total_row == 4 && si.dOmega[4] ≈ 4π
        @test si.dOmega[1] ≈ 2π * (cos(deg2rad(7.0)) - cos(deg2rad(27.0)))
    end

    @testset "spectrum accessors" begin
        b = ExactBins(PolarBands(3); nu_min = 1.0e8, nu_max = 1.0e16, n_ebins = 40)
        rng = Xoshiro(3)
        for _ in 1:5000
            Execution.record!(b, energy_at(10.0^(8 + 8 * rand(rng))), rand(rng),
                              SVector(randn(rng), randn(rng), randn(rng)))
        end
        s = Spectrum(b)
        nu, L = nuLnu(s)
        @test nu ≈ [exp(s.lnu_min + (i - 0.5) * s.dlnu) for i in 1:40] == nu_centers(s)
        @test L[2, 7] ≈ MEC2 / s.dlnu * 4π / s.dOmega[2] * s.sumwE[2, 7]
        nut, Lt, ct = nuLnu_total(s)
        @test Lt ≈ MEC2 / s.dlnu .* vec(sum(s.sumwE; dims = 1)) && ct == vec(sum(s.count; dims = 1))
        @test nuLnu_solid(s, π)[2] ≈ 4.0 .* Lt
        @test total_power(s) ≈ MEC2 * sum(s.sumwE) && total_weight(s) ≈ sum(s.sumw)
        @test mean_energy(s) ≈ sum(s.sumwE) / sum(s.sumw)
        @test band(s, 2).sumwE == s.sumwE[2:2, :] && band(s, 2).dOmega == [s.dOmega[2]]
        @test mean_energy(Spectrum(ExactBins(PolarBands(2)))) == 0.0
    end

    @testset "exact merge and scalars" begin
        rng = Xoshiro(11)
        w = exp.(30.0 .* randn(rng, 20_000))
        all1 = ExactScalars(1); foreach(x -> add!(all1, 1, x), w)
        parts = [ExactScalars(1) for _ in 1:7]
        foreach(((k, x),) -> add!(parts[1 + k % 7], 1, x), enumerate(w))
        m = ExactScalars(1); foreach(p -> merge!(m, p), parts)
        ref = setprecision(BigFloat, 2400) do; Float64(sum(BigFloat.(w))); end
        @test scalar(all1, 1) === ref === scalar(m, 1) && addends(m, 1) == 20_000
        b1 = ExactBins(PolarBands(2)); b2 = similar(b1); b3 = similar(b1)
        for (k, x) in enumerate(w[1:2000])
            d = SVector(1.0, 0.0, iseven(k) ? 0.1 : 3.0)
            Execution.record!(b1, energy_at(1.0e10), x, d)
            Execution.record!(isodd(k) ? b2 : b3, energy_at(1.0e10), x, d)
        end
        merge!(b2, b3)
        @test b2.count == b1.count && b2.sumw == b1.sumw && b2.sumwE == b1.sumwE
        @test_throws ArgumentError merge!(b1, ExactBins(PolarBands(3)))
    end

    @testset "child buffer: slots, overflow, streams" begin
        L = MCStreamLayout(100, 10, 2)
        buf = ChildBuffer(Superphoton, 3, L)
        sp = Superphoton(SVector(0.0, 1.0, 0.0, 0.0), SVector(-1.0, 1.0, 0.0, 0.0), 2.0)
        parent = primary_rng(9, 4)
        for e in 0:4
            Execution.spawn!(buf, sp, parent, e)
        end
        @test capacity(buf) == 3 && length(buf) == 3 && overflow(buf) == 2
        ph, rg = children(buf)
        @test rg == [child_rng(parent, e, L) for e in 0:2] && all(==(sp), ph)
        empty!(buf)
        @test length(buf) == 0 && overflow(buf) == 0
    end

    @testset "executor contract" begin
        n = 3001
        ref = run_acc(ThreadedExecutor(chunk = 1), n)
        @test sum(ref.bins.count) > 0 && length(ref.buf) == n ÷ 3 && overflow(ref.buf) == 0
        for exec in (ThreadedExecutor(chunk = 7), ThreadedExecutor(chunk = 4096),
                     (PoolExecutor(DK.hosts(nd); chunk = 97) for nd in DK.NDEV)...)
            p = run_acc(exec, n)
            @test p.bins.count == ref.bins.count && p.bins.sumw == ref.bins.sumw &&
                  p.bins.sumwE == ref.bins.sumwE
            @test p.scalars.acc == ref.scalars.acc && p.scalars.n == ref.scalars.n
            @test childset(p.buf) == childset(ref.buf)
        end
        # an undersized buffer counts every child; the rerun at the counted size keeps all
        for exec in (ThreadedExecutor(chunk = 64), PoolExecutor(DK.hosts(3); chunk = 97))
            small = run_acc(exec, n; cap = 50)
            @test overflow(small.buf) > 0 && Int(small.buf.count[1]) == n ÷ 3
            @test childset(run_acc(exec, n; cap = Int(small.buf.count[1])).buf) == childset(ref.buf)
        end
        # a pool run's twins must be gathered before reading
        p = AccProgram(ExactBins(PolarBands(2)), ExactScalars(2),
                       ChildBuffer(Superphoton, 10, MCStreamLayout(10, 10, 2)))
        foreach_item!(PoolExecutor(DK.hosts(2); chunk = 3), p, collect(1:10))
        @test_throws ErrorException Spectrum(p.bins)
        @test_throws ErrorException scalar(p.scalars, 1)
        gather!(p.bins); gather!(p.scalars)
        @test sum(Spectrum(p.bins).count) == 10 && addends(p.scalars, 1) == 10
    end

    @testset "sphere runs: threads and pools agree bitwise" begin
        s = UniformSphere()
        tabs, N = DK.sphere_tables(s, 20_000)
        ref = DK.sphere_thin_bins(ThreadedExecutor(chunk = 4096), s, tabs, N, 0xC0FFEE; n_ebins = 60)
        @test sum(ref.count) > 1000
        for exec in (ThreadedExecutor(chunk = 33), PoolExecutor(DK.hosts(2); chunk = 997),
                     PoolExecutor(DK.hosts(5); chunk = 4096))
            b = DK.sphere_thin_bins(exec, s, tabs, N, 0xC0FFEE; n_ebins = 60)
            @test b.count == ref.count && b.sumw == ref.sumw && b.sumwE == ref.sumwE
        end
        table = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        st, Ns = DK.sphere_tables(ss, 2_000; nu_min = 1.0e8, nu_max = 1.0e13)
        o1, l1, k1 = DK.sphere_scatter_run(ThreadedExecutor(chunk = 64), ss, table, st, Ns, 0xBEEF;
                                           n_ebins = 60)
        @test sum(length(first(k)) for k in k1) > 2_000 && all(!isempty(first(k)) for k in k1[1:3])
        ids(k) = [sort(stream_id.(last(layer))) for layer in k]
        for (exec, cap) in ((ThreadedExecutor(chunk = 7), 10), (PoolExecutor(DK.hosts(2); chunk = 300), 100),
                            (PoolExecutor(DK.hosts(5); chunk = 4096), 1))
            o, l, k = DK.sphere_scatter_run(exec, ss, table, st, Ns, 0xBEEF; n_ebins = 60,
                                            capacity = cap)
            @test all(o[n].sumwE == o1[n].sumwE && o[n].count == o1[n].count for n in 1:4)
            @test l.acc == l1.acc && ids(k) == ids(k1)
        end
    end
end
