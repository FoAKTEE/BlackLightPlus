# The Monte Carlo spectrum product: the sphere plan, the thin sphere against its analytic
# spectrum, the scattering sphere's orders and weight ledger, the torus spectrum on the
# block deck against the transport-tier template, bitwise agreement across executors,
# device counts, chunks and child-buffer capacities (the overflow rerun), the emission
# zones, the programs' device contract, the hot-cross table the plan builds once, the
# backend keyword and the run record.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: RunConfig, Plan, prepare, hotcross_table
using BlackLightPlus.Products: mc_spectrum, emission_zones, analytic_nuLnu, incl_nuLnu, ScatterResult,
                    GRMHDSpectrumResult, MCSink, MCTransport, SphereSource, ZoneSource, MCPrimary,
                    MCLayer, mc_ledger, ledger_sum, ledger_count, LEDGER_STRIPES, render
using BlackLightPlus.Spacetime: Minkowski, KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit,
                     inverse_metric
using BlackLightPlus.Matter: UniformSphere
using BlackLightPlus.MonteCarlo: HotcrossTable, NW, NT, Superphoton, GenerationTables, Opacities, Bias,
                      FlatMarch, CurvedMarch, generate
using BlackLightPlus.Numerics: SIGMA_THOMSON, GNEWT, CL, ME, MCStreamLayout, primary_rng
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, Spectrum, ExactBins, ExactScalars, ChildBuffer,
                     PolarBands, InclinationBands, band, nuLnu_total, nuLnu_solid, total_power,
                     total_weight, mean_energy, scalar, add!, describe, backend!, reset_backend!
using BlackLightPlus.Data: build_weight_table, zone_counts, build_zone_emission_table
const DK = Main.Decks

allocs(f, args...) = (f(args...); @allocated f(args...))
const ME_C2 = ME * CL * CL
const THOMSON = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
const L_TORUS = GNEWT * DK.TORUS_MBH / CL^2
torus_plan(snap) =
    prepare(RunConfig(m_bh_cgs = DK.TORUS_MBH, mass_scale_cgs = DK.TORUS_DENS * L_TORUS^3,
                      frequency_hz = 2.3e11);
            spacetime = KerrSchildCartesian(DK.TORUS_A), matter = snap)
"Every number of a spectrum."
sbits(s::Spectrum) = (s.count, DK.bits(s.sumw), DK.bits(s.sumwE))
same(a::Spectrum, b::Spectrum) = sbits(a) == sbits(b)
same(a::ScatterResult, b::ScatterResult) =
    all(same.(a.orders, b.orders)) && same(a.total, b.total) &&
    (a.n_primary, a.n_children) == (b.n_primary, b.n_children) &&
    DK.bits((a.w_primary, a.w_children, a.w_first_order, a.scatter_loss)) ==
    DK.bits((b.w_primary, b.w_children, b.w_first_order, b.scatter_loss))
same(a::GRMHDSpectrumResult, b::GRMHDSpectrumResult) =
    all(same.(a.spectra, b.spectra)) && same(a.total, b.total) &&
    (a.n_zones, a.n_gen, a.n_recorded) == (b.n_zones, b.n_gen, b.n_recorded) &&
    DK.bits((a.emitted_power, a.escaped_power)) == DK.bits((b.emitted_power, b.escaped_power))

const POOLS = [PoolExecutor(DK.hosts(nd); chunk = c) for (nd, c) in zip(DK.NDEV, (64, 997, 300, 4096))]

@testset "Monte Carlo spectra" begin
    s = UniformSphere()
    p = prepare(s)

    @testset "the sphere plan" begin
        @test p isa Plan && p.provider === s && p.spacetime === Minkowski() && p.params === nothing
        @test p.R_em === s.R && p.config === RunConfig()
        @test_throws ArgumentError render(p; backend = :cpu)
        @test analytic_nuLnu(p, 1.0e11) === analytic_nuLnu(s, 1.0e11) > 0.0
    end

    @testset "thin sphere: analytic spectrum and power" begin
        spec = mc_spectrum(p, 1_000_000; seed = 0xC0FFEE, backend = :cpu)
        @test spec isa Spectrum && spec.plan === p && spec.n_bands == 6
        nu, L, cnt = nuLnu_total(spec)
        rel = [abs(L[i] / analytic_nuLnu(s, nu[i]) - 1.0) for i in eachindex(nu)
               if 1.0e9 <= nu[i] <= 1.0e14 && cnt[i] >= 200]
        @test length(rel) >= 100 && maximum(rel) < 0.10
        lo = log(1.0e8); h = (log(1.0e16) - lo) / 4000
        bol = sum(((i == 0 || i == 4000) ? 1.0 : (isodd(i) ? 4.0 : 2.0)) *
                  analytic_nuLnu(s, exp(lo + i * h)) for i in 0:4000) * h / 3.0
        @test isapprox(total_power(spec) / bol, 1.0; atol = 0.03)
        tab = mc_spectrum(p, 1_000_000; seed = 0xC0FFEE, tabulated = true, backend = :cpu)
        @test abs(total_power(tab) / total_power(spec) - 1.0) < 1.0e-3
        @test sum(tab.count) == sum(spec.count)
    end

    @testset "scattering sphere: orders and weight conservation" begin
        # τ_es = n_e σ_T R ≈ 0.2: the top order loses little weight to scatterings it may
        # not spawn, so the spawned weight matches the scattering loss
        ss = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        ps = prepare(ss)
        r = mc_spectrum(ps, 40_000; scattering = true, table = THOMSON, seed = 0xBEEF,
                        nu_max = 1.0e13, spec_nu_max = 1.0e18, n_ebins = 100, backend = :cpu)
        @test r isa ScatterResult && length(r.orders) == 4 && r.max_scatt == 3
        @test r.plan === ps && r.n_primary > 30_000 && r.n_children > 0.1 * r.n_primary
        @test all(total_weight(o) > 0.0 for o in r.orders)
        @test all(mean_energy(r.orders[n + 1]) > 4.0 * mean_energy(r.orders[n]) for n in 1:3)
        @test r.total.count == sum(o.count for o in r.orders)
        @test isapprox(r.w_children / r.scatter_loss, 1.0; atol = 0.05)
        @test 0.0 < r.w_first_order < r.w_children && r.w_primary > 0.0
        τ = ss.n_e * SIGMA_THOMSON * ss.R
        @test 0.5τ < r.w_first_order / r.w_primary < τ
        # absorption may be switched on; it only removes weight
        ra = mc_spectrum(ps, 40_000; scattering = true, table = THOMSON, seed = 0xBEEF,
                         nu_max = 1.0e13, spec_nu_max = 1.0e18, n_ebins = 100, absorption = true,
                         backend = :cpu)
        @test total_weight(ra.orders[1]) < total_weight(r.orders[1])
        # with no scattering allowed the run is the transparent sphere
        r0 = mc_spectrum(ps, 2_000; scattering = true, max_scatt = 0, table = THOMSON,
                         seed = 0xBEEF, nu_max = 1.0e13, backend = :cpu)
        @test length(r0.orders) == 1 && r0.n_children == 0 && r0.w_children == 0.0
    end

    snap = DK.amr_deck()
    pt = torus_plan(snap)
    zones = emission_zones(pt)

    @testset "emission zones" begin
        @test pt.params == DK.torus_params()
        @test zones == DK.torus_zones(snap.blocks, pt.spacetime, pt.params)
        @test emission_zones(snap.blocks, pt.spacetime, pt.params) == zones
        @test length(zones) > 600 && all(z -> z.dV > 0.0 && z.n_e > 0.0 && z.theta_e > 0.0, zones)
        z2 = emission_zones(pt; stride = 2)
        @test 0 < length(z2) < length(zones) / 4 && z2[1].dV ≈ 8.0 * zones[1].dV
        z8 = emission_zones(pt; r_emit_max = 8.0)
        @test 0 < length(z8) < length(zones) && issubset(z8, zones)
        @test emission_zones(pt; max_zones = 10) == zones[1:10]
        @test_throws ArgumentError emission_zones(p)
        grid = prepare(RunConfig(); spacetime = KerrSchildCartesian(0.9), matter = DK.uniform_deck())
        @test_throws ArgumentError emission_zones(grid)
    end

    @testset "torus: the template run, energy, Compton tail" begin
        g = mc_spectrum(pt, 1_500; seed = 0xC0FFEE, C = 32.0, spec_nu_max = 1.0e24, n_ebins = 120,
                        backend = :cpu)
        b0, l0, n0, _ = DK.torus_run(ThreadedExecutor(chunk = 64), snap.blocks, zones, 1_500, 0xC0FFEE)
        s0 = Spectrum(b0)
        @test g isa GRMHDSpectrumResult && g.plan === pt && !g.scattering
        @test g.n_zones == length(zones) && g.n_gen == n0 > 1000
        @test same(g.total, band(s0, 4)) && all(same(g.spectra[k], band(s0, k)) for k in 1:3)
        @test g.incl_deg == [17.0, 50.0, 90.0] && g.half_width_deg == 10.0
        @test g.dOmega == s0.dOmega[1:3]
        @test g.emitted_power == ME_C2 * scalar(l0, 6) && g.escaped_power == ME_C2 * scalar(l0, 5)
        @test 0.0 < g.escaped_power < g.emitted_power && 0 < g.n_recorded <= g.n_gen
        nu, L = incl_nuLnu(g)
        @test size(L) == (3, 120) && L[2, :] == nuLnu_solid(g.spectra[2], g.dOmega[2])[2]
        gs = mc_spectrum(pt, 1_500; scattering = true, table = THOMSON, bias = 1.0e5, seed = 0xC0FFEE,
                         C = 32.0, n_ebins = 120, backend = :cpu)
        @test gs.scattering && gs.n_gen == g.n_gen
        nut, Lt, _ = nuLnu_total(g.total); nus, Ls, _ = nuLnu_total(gs.total)
        hi = 30.0 * nut[argmax(Lt)]
        @test sum(Ls[nus .> hi]) > sum(Lt[nut .> hi]) > 0.0
        @test_throws ArgumentError mc_spectrum(pt, 100; bands = PolarBands(3), backend = :cpu)
    end

    @testset "bitwise across executors, devices, chunks and capacities" begin
        thin = mc_spectrum(p, 6_000; seed = 7, n_ebins = 60, backend = :cpu)
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        kw = (scattering = true, table = THOMSON, seed = 0xBEEF, nu_max = 1.0e13, n_ebins = 60)
        sc = mc_spectrum(prepare(ss), 1_500; kw..., backend = ThreadedExecutor(chunk = 1))
        tk = (scattering = true, table = THOMSON, bias = 1.0e5, seed = 0xC0FFEE, C = 32.0,
              n_ebins = 80, stride = 2)
        tor = mc_spectrum(pt, 800; tk..., backend = :cpu)
        @test sc.n_children > 1_000 && tor.n_gen > 500
        for exec in (ThreadedExecutor(chunk = 7), POOLS...)
            @test same(mc_spectrum(p, 6_000; seed = 7, n_ebins = 60, backend = exec), thin)
            @test same(mc_spectrum(prepare(ss), 1_500; kw..., backend = exec), sc)
            @test same(mc_spectrum(pt, 800; tk..., backend = exec), tor)
        end
        # an undersized child buffer overflows; the rerun at the counted size is identical
        for exec in (ThreadedExecutor(chunk = 64), POOLS[3])
            @test same(mc_spectrum(prepare(ss), 1_500; kw..., capacity = 1, backend = exec), sc)
            @test same(mc_spectrum(pt, 800; tk..., capacity = 3, backend = exec), tor)
        end
    end

    @testset "programs: device contract, no allocation" begin
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        tabs, _ = DK.sphere_tables(ss, 1000; nu_min = 1.0e8, nu_max = 1.0e13)
        ctl = FlatMarch(ss)
        layout = MCStreamLayout(1000, 100_000, 3)
        bins = ntuple(_ -> ExactBins(PolarBands(6); nu_min = 1.0e8, nu_max = 1.0e17), 4)
        sink = MCSink(ctl, bins, ChildBuffer(Superphoton, 4096, layout), mc_ledger())
        tr = MCTransport(ctl, Minkowski(), ss, Opacities(; table = THOMSON, bias = Bias(ss), max_scatt = 3),
                         Val(false), Val(true))
        prim = MCPrimary(SphereSource(ss, tabs), tr, sink, UInt32(3))
        @test DK.device_isbits(prim) && allocs(prim, 5, 5) == 0
        items = [5, 5]
        for g in 6:400                                  # until a march spawns
            length(sink.children) > 0 && break
            prim(g, g)
            push!(items, g)
        end
        @test length(sink.children) > 0
        emitted = ExactScalars(1)
        for g in items
            sph = generate(ss, tabs, primary_rng(3, g))[1]
            add!(emitted, 1, sph.w * sph.E)
        end
        @test ledger_count(sink.ledger, 6) == length(items)
        @test ledger_sum(sink.ledger, 6) === scalar(emitted, 1)
        ph = [generate(ss, tabs, primary_rng(3, g))[1] for g in 1:4]
        lay = MCLayer(ph, [primary_rng(3, g) for g in 1:4], tr, sink)
        @test DK.device_isbits(lay) && allocs(lay, 2, 2) == 0
        blocks = snap.blocks
        jt = build_zone_emission_table(zones[1:50])
        wgt = build_weight_table(jt, 100.0)
        _, dn = zone_counts(jt, wgt)
        off = collect(0:2:100)
        ctr = CurvedMarch(pt.spacetime; C = 32.0, R_em = 30.0)
        ztr = MCTransport(ctr, pt.spacetime, blocks, Opacities(pt.params; table = THOMSON, bias = Bias(1.0e5),
                                                                max_scatt = 3), Val(true), Val(true))
        zsink = MCSink(ctr, (ExactBins(InclinationBands((17.0, 50.0), 10.0)),),
                       ChildBuffer(Superphoton, 4096, layout), mc_ledger())
        zp = MCPrimary(ZoneSource(zones[1:50], wgt, dn, off, nothing), ztr, zsink, UInt32(1))
        @test DK.device_isbits(zp) && allocs(zp, 17, 17) == 0
        @test ledger_count(zsink.ledger, 6) == 2
        @test length(mc_ledger().n) == 6 * LEDGER_STRIPES
    end

    @testset "the plan's hot-cross table" begin
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        ps = prepare(ss)
        kw = (scattering = true, seed = 0xBEEF, nu_max = 1.0e13, n_ebins = 40, backend = :cpu)
        @test ps.hotcross[] === nothing
        r1 = mc_spectrum(ps, 300; kw...)
        t = ps.hotcross[]
        @test t isa HotcrossTable && hotcross_table(ps) === t
        @test same(mc_spectrum(ps, 300; kw...), r1) && ps.hotcross[] === t
        @test same(mc_spectrum(ps, 300; kw..., table = t), r1)
        @test prepare(ss).hotcross[] === nothing
        @test mc_spectrum(prepare(ss), 300; kw..., table = THOMSON).plan.hotcross[] === nothing
    end

    @testset "backend keyword and run record" begin
        ref = mc_spectrum(p, 3_000; seed = 11, n_ebins = 40, backend = :cpu)
        @test ref.run.executor.kind === :threads && ref.run.version == pkgversion(BlackLightPlus)
        @test ref.run.lanes === p.lanes && ref.run.seconds > 0.0 && ref.plan === p
        hp = mc_spectrum(p, 3_000; seed = 11, n_ebins = 40, backend = :host_pool)
        @test hp.run.executor.kind === :pool && hp.run.executor.devices == ["HostDevice(1)"]
        ex = PoolExecutor(DK.hosts(2); chunk = 333)
        pe = mc_spectrum(p, 3_000; seed = 11, n_ebins = 40, backend = ex)
        @test pe.run.executor == describe(ex)
        @test same(hp, ref) && same(pe, ref)
        reset_backend!()
        try
            backend!(:host_pool)
            d = mc_spectrum(p, 3_000; seed = 11, n_ebins = 40)
            @test d.run.executor.kind === :pool && same(d, ref)
            @test mc_spectrum(p, 3_000; seed = 11, n_ebins = 40, backend = :cpu).run.executor.kind === :threads
        finally
            reset_backend!()
        end
        g = mc_spectrum(pt, 300; stride = 3, seed = 2, n_ebins = 40, backend = ex)
        @test g.run.executor.kind === :pool && g.run.lanes === pt.lanes && g.plan === pt
        @test occursin("GRMHDSpectrumResult", sprint(show, g))
    end

    @testset "argument checks" begin
        none = prepare(RunConfig(); spacetime = KerrSchildCartesian(0.9), matter = nothing)
        @test_throws ArgumentError mc_spectrum(none, 100; backend = :cpu)
        kerr_sphere = prepare(RunConfig(); spacetime = KerrSchildCartesian(0.9), matter = s)
        @test_throws ArgumentError mc_spectrum(kerr_sphere, 100; backend = :cpu)
        @test_throws ArgumentError mc_spectrum(p, 0; backend = :cpu)
        @test_throws ArgumentError mc_spectrum(p, 100; scattering = true, table = THOMSON,
                                               capacity = 0, backend = :cpu)
        @test_throws ArgumentError analytic_nuLnu(pt, 1.0e11)
    end

    @testset "moving binary: zones with a normal observer, the march at the holes' times" begin
        orbit = CircularOrbit((0.5, 0.5), 8.0)
        mov = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
        pb = prepare(RunConfig(m_bh_cgs = DK.TORUS_MBH, mass_scale_cgs = DK.TORUS_DENS * L_TORUS^3,
                               frequency_hz = 2.3e11); spacetime = mov, matter = snap)
        zb = emission_zones(pb)
        @test length(zb) > 100 && all(z -> inverse_metric(mov, z.x)[1, 1] < 0.0, zb)   # no cell without a normal observer
        gb = mc_spectrum(pb, 1_500; seed = 0xC0FFEE, C = 32.0, backend = :cpu)
        @test gb.n_zones == length(zb) && gb.n_gen > 1000 && gb.n_recorded > 0
        @test 0.0 < gb.escaped_power < gb.emitted_power
        @test same(gb, mc_spectrum(pb, 1_500; seed = 0xC0FFEE, C = 32.0, backend = :host_pool))
        gs = mc_spectrum(pb, 1_000; scattering = true, table = THOMSON, seed = 0xC0FFEE, C = 32.0, backend = :cpu)
        @test gs.scattering && gs.n_recorded > 0 && all(isfinite, gs.total.sumw)
    end
end
