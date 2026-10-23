# The hot-path contract of the Monte Carlo: generation, both controls of the tracker
# (with and without scattering, into sinks over the exact accumulators), the binning and
# the child buffer allocate nothing, and every record a per-item program carries is
# `isbits` once adapted to a device.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Numerics: MCRng, MCStreamLayout, primary_rng, SIGMA_THOMSON
using BlackLightPlus.Spacetime: Minkowski, KerrSchildCartesian
using BlackLightPlus.Matter: UniformSphere
using BlackLightPlus.MonteCarlo: Superphoton, GenerationTables, Opacities, Bias, FlatMarch, CurvedMarch,
                      HotcrossTable, NullSink, generate, track!, zone_nsuperphotons, NW, NT,
                      CURVED_MAX_STEPS, SPHERE_MAX_STEPS
using BlackLightPlus.Execution: ExactBins, ExactScalars, ChildBuffer, PolarBands, InclinationBands

const DK = Main.Decks
import BlackLightPlus.Execution

allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))
trk(ctl, st, prov, opac, sph, rng, sink, a, s) =
    track!(ctl, st, prov, opac, sph, rng, sink; absorb = a, scatter = s)

@testset "Monte Carlo hot path" begin
    table = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
    s = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
    tabs, _ = DK.sphere_tables(s, 1000; nu_min = 1.0e8, nu_max = 1.0e13)
    sph, rng = generate(s, tabs, primary_rng(1, 1))
    flat = FlatMarch(s)
    layout = MCStreamLayout(1000, SPHERE_MAX_STEPS, 3)
    bins = ntuple(_ -> ExactBins(PolarBands(6); nu_min = 1.0e8, nu_max = 1.0e17), 4)
    buf = ChildBuffer(Superphoton, 4096, layout)
    sink = DK.BinsSink(flat, bins, buf, DK.mc_ledger())
    opac = Opacities(; table = table, bias = Bias(s), max_scatt = 3)

    @testset "no allocation" begin
        @test allocs(generate, s, tabs, primary_rng(1, 2)) == 0
        @test allocs(generate, DK.torus_zones(DK.amr_deck().blocks, KerrSchildCartesian(0.9),
                                              DK.torus_params())[1],
                     tabs, primary_rng(1, 3)) == 0
        @test allocs(trk, flat, Minkowski(), s, Opacities(), sph, rng, NullSink(), Val(true),
                     Val(false)) == 0
        @test allocs(trk, flat, Minkowski(), s, opac, sph, rng, sink, Val(false), Val(true)) == 0
        for g in 2:200                                 # until the scattering march spawns
            buf.count[1] > 0 && break
            p, r = generate(s, tabs, primary_rng(1, g))
            trk(flat, Minkowski(), s, opac, p, r, sink, Val(false), Val(true))
        end
        @test buf.count[1] > 0
        st = KerrSchildCartesian(0.9)
        blocks = DK.amr_deck().blocks
        params = DK.torus_params()
        zone = DK.torus_zones(blocks, st, params)[100]
        zt = GenerationTables(tabs.weights, zone_nsuperphotons(zone, tabs.weights)[2])
        zs, zrng = generate(zone, zt, primary_rng(2, 1))
        curved = CurvedMarch(st; C = 32.0, R_em = 30.0)
        incl = (ExactBins(InclinationBands((17.0, 50.0), 10.0)),)
        csink = DK.BinsSink(curved, incl, ChildBuffer(Superphoton, 64, MCStreamLayout(10, CURVED_MAX_STEPS, 3)),
                            DK.mc_ledger())
        copac = Opacities(params; table = table, bias = Bias(1.0e6), max_scatt = 3)
        @test allocs(trk, curved, st, blocks, Opacities(params), zs, zrng, NullSink(), Val(true),
                     Val(false)) == 0
        @test allocs(trk, curved, st, blocks, copac, zs, zrng, csink, Val(true), Val(true)) == 0
        @test allocs(Execution.record!, bins[1], 1.0e-6, 2.0, SVector(0.1, 0.2, 0.3)) == 0
        @test allocs(Execution.record!, incl[1], 1.0e-6, 2.0, SVector(0.1, 0.2, 0.3)) == 0
        @test allocs(Execution.spawn!, buf, sph, rng, 3) == 0
        @test allocs(Execution.add!, DK.mc_ledger(), 2, 1.5) == 0
    end

    @testset "isbits on a device" begin
        st = KerrSchildCartesian(0.9)
        prog = DK.PrimaryProgram(s, tabs, flat, Minkowski(), s, opac, sink, UInt32(1),
                                 Val(false), Val(true))
        for x in (flat, CurvedMarch(st), opac, Opacities(DK.torus_params(); table = table),
                  tabs, table, bins[1], buf, DK.mc_ledger(), sink, prog, sph, rng)
            @test DK.device_isbits(x)
        end
        @test isbits(flat) && isbits(CurvedMarch(st)) && isbits(sph) && isbits(Bias(2.0))
    end
end
