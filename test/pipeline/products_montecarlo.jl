# The Monte Carlo spectrum product: the sphere plan, the thin sphere against its analytic
# spectrum with Σ N_z = Ns, the scattering sphere's orders, weight ledger and power
# budget, the scattered power independent of the bias with absorption on, the torus
# spectrum on the block deck against the transport-tier template with its per-order split
# and power budget, bitwise agreement across executors, device counts, chunks and
# child-buffer capacities (the overflow rerun), thin and scattering runs on the device
# wavefront of `WaveHost` pools (bitwise among device counts, budgets and child slots, within
# 1e-11 of threads; the straight-line march of the sphere bitwise the threads; the same bits
# with the cohort sorted by the key stage), the emission zones, the programs' device contract, the process's hot-cross table, the backend
# keyword, the run record and the photon order key.
using Test, BlackLightPlus, StaticArrays
using Adapt: Adapt
using BlackLightPlus.Plans: RunConfig, Plan, prepare, hotcross_table
using BlackLightPlus.Products: mc_spectrum, emission_zones, analytic_nuLnu, incl_nuLnu, ScatterResult,
                    GRMHDSpectrumResult, MCSink, MCTransport, SphereSource, ZoneSource, MCPrimary,
                    MCLayer, mc_ledger, ledger_sum, ledger_count, LEDGER_STRIPES, render, grid_tables,
                    grid_zones, ZoneSums, ZoneCounts, hotcross_build, photon_key, PHOTON_KEYS,
                    MCStart, MCAdvance, MCRetire, MCBranch, MCKey
using BlackLightPlus.Spacetime: Minkowski, KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit,
                     inverse_metric, proxy_distance, SuperposedKerrSchild, horizon_floor
using BlackLightPlus.Matter: UniformSphere, ZoneSite
using BlackLightPlus.MonteCarlo: HotcrossTable, NW, NT, Superphoton, GenerationTables, Opacities, Bias,
                      StateBias, FlatMarch, CurvedMarch, generate, EMISSIVITY_TABLE, ZoneNodes,
                      MarchState
using BlackLightPlus.Numerics: SIGMA_THOMSON, GNEWT, CL, ME, MCStreamLayout, primary_rng, mc_uniform
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, Spectrum, ExactBins, ExactScalars, ChildBuffer,
                     PolarBands, InclinationBands, band, nuLnu_total, nuLnu_solid, total_power,
                     total_weight, mean_energy, scalar, add!, describe, backend!, reset_backend!,
                     nuLnu_sigma, foreach_item!, with_wavefront_settings, WaveQueues, Cohort,
                     StateBuffer, WaveSwap, wave_state, wave_steps, generation, Generation, wave_log,
                     KeyCount, wave_keys
import BlackLightPlus.Execution, BlackLightPlus.Products
using BlackLightPlus.Data: zone_tables, emission_sites
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
sbits(s::Spectrum) = (s.count, DK.bits(s.sumw), DK.bits(s.sumwE), DK.bits(s.sumwE2))
same(a::Spectrum, b::Spectrum) = sbits(a) == sbits(b)
"The ledger of a result, weights and powers."
ledger(r) = (r.w_primary, r.w_children, r.w_first_order, r.scatter_loss, r.scatter_loss_final,
             r.w_unspawned, r.emitted_power, r.escaped_power, r.absorbed_power, r.captured_power,
             r.dropped_power)
same(a::ScatterResult, b::ScatterResult) =
    all(same.(a.orders, b.orders)) && same(a.total, b.total) &&
    (a.n_primary, a.n_children) == (b.n_primary, b.n_children) &&
    DK.bits(ledger(a)) == DK.bits(ledger(b))
same(a::GRMHDSpectrumResult, b::GRMHDSpectrumResult) =
    all(same.(a.spectra, b.spectra)) && same(a.total, b.total) && all(same.(a.orders, b.orders)) &&
    (a.n_zones, a.n_gen, a.n_recorded, a.n_children) == (b.n_zones, b.n_gen, b.n_recorded, b.n_children) &&
    DK.bits(ledger(a)) == DK.bits(ledger(b))

const POOLS = [PoolExecutor(DK.hosts(nd); chunk = c) for (nd, c) in zip(DK.NDEV, (64, 997, 300, 4096))]

"Equal counts, and every sum of every cell and the ledger within `tol` of `b`'s, relatively."
function close_to(a::GRMHDSpectrumResult, b::GRMHDSpectrumResult, tol)
    rel(x, y) = all(i -> x[i] == y[i] || abs(x[i] - y[i]) <= tol * abs(y[i]), eachindex(x, y))
    cells(s, t) = s.count == t.count && rel(s.sumw, t.sumw) && rel(s.sumwE, t.sumwE) &&
                  rel(s.sumwE2, t.sumwE2)
    return cells(a.total, b.total) && all(cells.(a.spectra, b.spectra)) &&
           all(cells.(a.orders, b.orders)) && rel(collect(ledger(a)), collect(ledger(b))) &&
           (a.n_gen, a.n_recorded, a.n_children) == (b.n_gen, b.n_recorded, b.n_children)
end

"Writes the order key of the state `ys[g]` on `st` under `ctl` to `out[i]`."
struct KeyProgram{S,C,V}
    st::S
    ctl::C
    ys::V
end
Adapt.@adapt_structure KeyProgram
(k::KeyProgram)(g, i, out) = (@inbounds out[i] = photon_key(k.st, k.ctl, k.ys[g]); nothing)

"The state at `x` with covariant momentum `p`."
state(x, p) = SVector{8,Float64}(x..., p...)

"""
`n` states of the stream `seed`: radii log-uniform in `[10⁻³, 10⁵]`, isotropic
directions, momenta with every spatial sign.
"""
function key_cloud(n; seed = 0x5eed)
    ys = Vector{SVector{8,Float64}}(undef, n)
    for g in 1:n
        rng = primary_rng(seed, g)
        u = zeros(6)
        for j in 1:6
            u[j], rng = mc_uniform(rng)
        end
        r = 10.0^(8.0 * u[1] - 3.0)
        c = 2.0 * u[2] - 1.0
        sn = sqrt(1.0 - c * c)
        ys[g] = state((0.0, r * sn * cospi(2u[3]), r * sn * sinpi(2u[3]), r * c),
                      (-1.0, 2.0 * u[4] - 1.0, 2.0 * u[5] - 1.0, 2.0 * u[6] - 1.0))
    end
    return ys
end

"The key of `y` with the shell `s`, the octant of the position and that of `g^{iν} p_ν` by the StaticArrays product."
function expected_key(st, s, y)
    oct(v) = 4 * signbit(v[1]) + 2 * signbit(v[2]) + signbit(v[3])
    k = inverse_metric(st, y[SVector(1, 2, 3, 4)]) * y[SVector(5, 6, 7, 8)]
    return (s * 8 + oct(y[SVector(2, 3, 4)])) * 8 + oct(k[SVector(2, 3, 4)])
end

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
        @test sum(spec.count) == 1_000_000                      # trapezoid counts: N = Ns
        # per-bin standard errors: the residuals against the closed form are N(0, 1)
        _, sig, neff = nuLnu_sigma(spec)
        z = [(L[i] - analytic_nuLnu(s, nu[i])) / sig[i] for i in eachindex(nu)
             if 1.0e9 <= nu[i] <= 1.0e14 && cnt[i] >= 200]
        @test length(z) >= 100 && abs(sum(z) / length(z)) < 4.0 / sqrt(length(z))
        @test 0.8 < sum(abs2, z) / length(z) < 1.2
        @test all(i -> neff[i] <= cnt[i] * (1 + 1e-12), eachindex(cnt))
        @test all(i -> 0.9 * cnt[i] <= neff[i], [i for i in eachindex(nu) if 1.0e9 <= nu[i] <= 1.0e14 && cnt[i] >= 200])
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
        @test r.total.sumwE2 ≈ sum(o.sumwE2 for o in r.orders) rtol = 1e-12
        @test isapprox((r.w_children + r.w_unspawned) / r.scatter_loss, 1.0; atol = 0.01)
        @test r.w_unspawned == 0.0 && r.scatter_loss_final > 0.0 && r.captured_power == 0.0
        @test r.absorbed_power == 0.0 && r.dropped_power == 0.0 && r.emitted_power > 0.0
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

    @testset "absorbing scattering sphere: the scattered power does not depend on the bias" begin
        # τ_T = 0.2 and absorption in the steps that scatter: the escaped power of every
        # order is the same for biases whose events per step differ a hundredfold (b dτ_s
        # per step about 1 for b = 100, where the one-event-per-step rule was 14 % high)
        ss = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 30.0, R = 3.0e5)
        ps = prepare(ss)
        kw = (scattering = true, table = THOMSON, nu_max = 1.0e13, spec_nu_max = 1.0e18,
              n_ebins = 50, absorption = true, max_scatt = 2, backend = :cpu)
        ra = mc_spectrum(ps, 20_000; kw..., seed = 0xAB01, bias = Bias(1.0))
        rb = mc_spectrum(ps, 20_000; kw..., seed = 0xAB02, bias = Bias(100.0))
        @test rb.n_children > 30 * ra.n_children
        @test abs((rb.w_children + rb.w_unspawned) / rb.scatter_loss - 1.0) < 0.05
        @test ra.absorbed_power > 0.0 && rb.absorbed_power > 0.0
        for n in 1:3
            Pa = total_power(ra.orders[n]); Pb = total_power(rb.orders[n])
            sa = ME_C2 * sqrt(sum(ra.orders[n].sumwE2)); sb = ME_C2 * sqrt(sum(rb.orders[n].sumwE2))
            @test abs(Pa - Pb) < 5.0 * sqrt(sa^2 + sb^2)
        end
    end

    @testset "order factor of the state bias: every order unchanged, with and without absorption" begin
        # τ_T = 0.2, three orders: the state bias with the order factor 30 (each later
        # generation biased 30 times more) against the plain state bias. Without absorption
        # (B = 30 G, absorption off) the escaped power of every order agrees and the factor
        # samples orders 2 and 3 several times better; with absorption (B = 100 G, self-
        # absorbed seeds) every order agrees within its error as well.
        for (B, absorb, b0, N) in ((30.0, false, 1.0, 20_000), (100.0, true, 10.0, 200_000))
            ps = prepare(UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = B, R = 3.0e5))
            kw = (scattering = true, table = THOMSON, nu_max = 1.0e13, spec_nu_max = 1.0e18,
                  n_ebins = 50, absorption = absorb, max_scatt = 3, backend = :cpu, bias = b0)
            ra = mc_spectrum(ps, N; kw..., seed = 0xAC01)
            rb = mc_spectrum(ps, N; kw..., seed = 0xAC02, bias_order = 30.0)
            absorb || @test rb.n_children > 5 * ra.n_children
            @test abs((rb.w_children + rb.w_unspawned) / rb.scatter_loss - 1.0) < (absorb ? 0.1 : 0.01)
            rel = zeros(2, 4)
            for n in 1:4
                Pa = total_power(ra.orders[n]); Pb = total_power(rb.orders[n])
                sa = ME_C2 * sqrt(sum(ra.orders[n].sumwE2)); sb = ME_C2 * sqrt(sum(rb.orders[n].sumwE2))
                @test Pa > 0.0 && Pb > 0.0
                @test abs(Pa - Pb) < 5.0 * sqrt(sa^2 + sb^2)
                rel[1, n] = sa / Pa; rel[2, n] = sb / Pb
            end
            @test rel[2, 4] < rel[1, 4]
            absorb || @test rel[2, 3] < 0.5 * rel[1, 3] && rel[2, 4] < 0.2 * rel[1, 4]
        end
        @test_throws ArgumentError mc_spectrum(prepare(UniformSphere()), 100; scattering = true,
                                               bias_order = 0.5)
    end

    snap = DK.amr_deck()
    pt = torus_plan(snap)
    zones = emission_zones(pt)

    @testset "emission zones" begin
        @test pt.params == DK.torus_params()
        ref = DK.torus_zones(snap.blocks, pt.spacetime, pt.params)    # an independent serial decode
        @test length(zones) == length(ref) && all(z.x == r.x && z.dV == r.dV for (z, r) in zip(zones, ref))
        @test all(isapprox(z.n_e, r.n_e; rtol = 1e-14) && isapprox(z.theta_e, r.theta_e; rtol = 1e-14) &&
                  isapprox(z.B, r.B; rtol = 1e-14) for (z, r) in zip(zones, ref))
        @test emission_zones(snap.blocks, pt.spacetime, pt.params) == zones
        @test length(zones) > 600 && all(z -> z.dV > 0.0 && z.n_e > 0.0 && z.theta_e > 0.0, zones)
        z2 = emission_zones(pt; stride = 2)
        @test 0 < length(z2) < length(zones) / 4
        at = Dict(z.x => z for z in zones)
        @test all(z -> haskey(at, z.x) && round(z.dV / at[z.x].dV) in 1:8, z2)   # an owned cell, n_own of them
        @test emission_zones(pt; stride = 2, seed = 3) != z2 && emission_zones(pt; stride = 2) == z2
        z8 = emission_zones(pt; r_emit_max = 8.0)
        @test 0 < length(z8) < length(zones) && issubset(z8, zones)
        @test emission_zones(pt; max_zones = 10) == zones[1:10]
        @test_throws ArgumentError emission_zones(p)
        grid = prepare(RunConfig(); spacetime = KerrSchildCartesian(0.9), matter = DK.uniform_deck())
        @test_throws ArgumentError emission_zones(grid)
    end

    @testset "zone tables on every executor, and kept across runs" begin
        sites = emission_sites(snap.blocks, pt.spacetime, pt.params)
        ref = zone_tables(sites, 100.0)
        tbits(t) = (DK.bits(t[1].lwgt), DK.bits(t[2]), DK.bits(t[3]))
        for exec in (ThreadedExecutor(chunk = 5), POOLS..., PoolExecutor(DK.devices(1)))
            @test tbits(grid_tables(exec, sites, 100.0)) == tbits(ref)
        end
        @test tbits(grid_tables(PoolExecutor(DK.devices(1)), sites, 7.0e3; n_esamp = 37,
                                nu_max = 1.0e14)) ==
              tbits(zone_tables(sites, 7.0e3; n_esamp = 37, nu_max = 1.0e14))
        nodes = ZoneNodes(ref[1])
        @test DK.device_isbits(ZoneSums(sites, nodes.nu, EMISSIVITY_TABLE, 256, ExactScalars(401)))
        @test DK.device_isbits(ZoneCounts(sites, nodes, EMISSIVITY_TABLE))
        o = (seed = UInt32(5), zones = (stride = 1, r_emit_max = Inf, max_zones = typemax(Int)),
             gen = (nu_min = 1.0e8, nu_max = 1.0e16, n_esamp = 400, Ns = 1.0e3))
        b = snap.blocks
        a = grid_zones(b, pt.spacetime, pt.params, ThreadedExecutor(), o)
        @test a[1] == sites && a[2] == maximum(z -> 1.2 * proxy_distance(pt.spacetime, z.x)[2], sites)
        nsp(Ns) = Ns / (log(1.0e16) - log(1.0e8))
        @test tbits(a[3:5]) == tbits(zone_tables(sites, nsp(1.0e3)))
        h = grid_zones(b, pt.spacetime, pt.params, PoolExecutor(DK.devices(1)), (o..., seed = UInt32(6)))
        @test h[1] === a[1] && h[3] === a[3]            # the seed draws nothing at stride 1
        c = grid_zones(b, pt.spacetime, pt.params, ThreadedExecutor(), (o..., gen = (o.gen..., Ns = 2.0e3)))
        @test c[1] === a[1] && c[3] !== a[3] && tbits(c[3:5]) == tbits(zone_tables(sites, nsp(2.0e3)))
        s2 = (o..., zones = (o.zones..., stride = 2))
        d = grid_zones(b, pt.spacetime, pt.params, ThreadedExecutor(), s2)
        @test d[1] == emission_sites(b, pt.spacetime, pt.params; stride = 2, seed = 5)
        @test grid_zones(b, pt.spacetime, pt.params, ThreadedExecutor(), (s2..., seed = UInt32(6)))[1] !=
              d[1]
        k5 = KerrSchildCartesian(0.5)
        @test grid_zones(b, k5, pt.params, ThreadedExecutor(), o)[1] == emission_sites(b, k5, pt.params)
        e = grid_zones(b, pt.spacetime, pt.params, ThreadedExecutor(), o)
        @test e[1] == sites && e[1] !== a[1]
    end

    @testset "torus: the template run, energy, Compton tail" begin
        g = mc_spectrum(pt, 1_500; seed = 0xC0FFEE, C = 32.0, spec_nu_max = 1.0e24, n_ebins = 120,
                        backend = :cpu)
        b0, l0, n0, _ = DK.torus_run(ThreadedExecutor(chunk = 64), snap.blocks,
                                     emission_sites(snap.blocks, pt.spacetime, pt.params), 1_500, 0xC0FFEE)
        s0 = Spectrum(b0)
        @test g isa GRMHDSpectrumResult && g.plan === pt && !g.scattering
        @test g.n_zones == length(zones) && g.n_gen == n0 > 1000
        @test same(g.total, band(s0, 4)) && all(same(g.spectra[k], band(s0, k)) for k in 1:3)
        @test g.incl_deg == [17.0, 50.0, 90.0] && g.half_width_deg == 10.0
        @test g.dOmega == s0.dOmega[1:3]
        @test g.emitted_power == ME_C2 * scalar(l0, 6) && g.escaped_power == ME_C2 * scalar(l0, 5)
        @test 0.0 < g.escaped_power < g.emitted_power && 0 < g.n_recorded <= g.n_gen
        @test g.absorbed_power == ME_C2 * scalar(l0, 8) && g.captured_power == ME_C2 * scalar(l0, 9)
        # power budget of a thin run on a stationary spacetime: emitted = escaped + absorbed +
        # captured + dropped, to rounding
        @test g.escaped_power + g.absorbed_power + g.captured_power + g.dropped_power ≈
              g.emitted_power rtol = 1e-10
        @test length(g.orders) == 1 && same(band(g.orders[1], 4), g.total) && g.n_children == 0
        @test g.w_children == g.scatter_loss == g.w_primary == 0.0
        nu, L = incl_nuLnu(g)
        @test size(L) == (3, 120) && L[2, :] == nuLnu_solid(g.spectra[2], g.dOmega[2])[2]
        gs = mc_spectrum(pt, 1_500; scattering = true, table = THOMSON, bias = 1.0e5, seed = 0xC0FFEE,
                         C = 32.0, n_ebins = 120, backend = :cpu)
        @test gs.scattering && gs.n_gen == g.n_gen && length(gs.orders) == 4 && gs.max_scatt == 3
        @test gs.total.count == sum(o.count[4:4, :] for o in gs.orders)
        @test abs((gs.w_children + gs.w_unspawned) / gs.scatter_loss - 1.0) < 0.2
        @test gs.n_children > 0 && gs.w_first_order <= gs.w_children
        nut, Lt, _ = nuLnu_total(g.total); nus, Ls, _ = nuLnu_total(gs.total)
        hi = 30.0 * nut[argmax(Lt)]
        @test sum(Ls[nus .> hi]) > sum(Lt[nut .> hi]) > 0.0
        @test_throws ArgumentError mc_spectrum(pt, 100; bands = PolarBands(3), backend = :cpu)
    end

    @testset "bitwise across executors, devices, chunks and capacities" begin
        thin = mc_spectrum(p, 6_000; seed = 7, n_ebins = 60, backend = :cpu)
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        kw = (scattering = true, table = THOMSON, seed = 0xBEEF, nu_max = 1.0e13, n_ebins = 60,
              bias = Bias(ss))
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

    @testset "the device wavefront: thin runs on WaveHost pools" begin
        tw = (seed = 0xC0FFEE, C = 32.0, n_ebins = 80, stride = 2)
        thr = mc_spectrum(pt, 800; tw..., backend = :cpu)
        hw = map(1:3) do nd
            ex = PoolExecutor(DK.wavehosts(nd))
            r = mc_spectrum(pt, 800; tw..., backend = ex)
            @test describe(ex).schedule == :wavefront && sum(describe(ex).admitted) == r.n_gen
            r
        end
        @test hw[1].n_gen > 500 && all(r -> same(r, hw[1]), hw)
        ex = PoolExecutor(DK.wavehosts(2))
        for K in (1, 7, 32, typemax(Int))
            r = with_wavefront_settings(budget = K, doubling = false, cohort = 64, refill = 37) do
                mc_spectrum(pt, 800; tw..., backend = ex)
            end
            @test same(r, hw[1]) && all(>(0), describe(ex).admitted)
        end
        for every in (1, 4)
            r = with_wavefront_settings(sort_every = every, cohort = 64, refill = 37) do
                mc_spectrum(pt, 800; tw..., backend = ex)
            end
            # a run without scattering has no key stage: never sorted
            @test same(r, hw[1]) && !any(it -> it.sorted, reduce(vcat, wave_log(ex)))
        end
        # the refill in the zone radii's cost order or in index order, launches waiting or not,
        # every workgroup size: the same bits
        for co in (true, false), wt in (true, false), wg in (0, 64)
            r = with_wavefront_settings(cost_order = co, wait = wt, workgroup = wg, cohort = 64, refill = 37) do
                mc_spectrum(pt, 800; tw..., backend = ex)
            end
            set = describe(ex).settings
            @test same(r, hw[1]) && set.cost_order == co && set.wait == wt && set.workgroup == wg
        end
        # threads run track! in another inlining context: equal counts, sums within 1e-11
        @test close_to(hw[1], thr, 1.0e-11) && hw[1].n_recorded == thr.n_recorded
        # the straight-line march of the sphere: bitwise the threads
        @test same(mc_spectrum(p, 6_000; seed = 7, n_ebins = 60, backend = ex),
                   mc_spectrum(p, 6_000; seed = 7, n_ebins = 60, backend = :cpu))
        # the cascade of the same pool compiles the programs of threads: their bits
        @test same(with_wavefront_settings(() -> mc_spectrum(pt, 800; tw..., backend = ex);
                                           wavefront = false), thr)
        @test describe(ex).schedule == :cascade
    end

    @testset "the device wavefront: scattering runs on WaveHost pools" begin
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        kw = (scattering = true, table = THOMSON, seed = 0xBEEF, nu_max = 1.0e13, n_ebins = 60,
              bias = Bias(ss))
        tk = (scattering = true, table = THOMSON, bias = 1.0e5, seed = 0xC0FFEE, C = 32.0,
              n_ebins = 80, stride = 2)
        sph = mc_spectrum(prepare(ss), 1_500; kw..., backend = :cpu)
        tor = mc_spectrum(pt, 800; tk..., backend = :cpu)
        @test sph.n_children > 10_000 && tor.n_children > 100
        hw = map(1:3) do nd
            ex = PoolExecutor(DK.wavehosts(nd))
            a = mc_spectrum(prepare(ss), 1_500; kw..., backend = ex)
            d = describe(ex)
            @test d.schedule == :wavefront && sum(d.admitted) == a.n_primary &&
                  sum(d.spawned) == a.n_children
            b = mc_spectrum(pt, 800; tk..., backend = ex)
            @test describe(ex).schedule == :wavefront && sum(describe(ex).spawned) == b.n_children
            (a, b)
        end
        @test all(r -> same(r[1], hw[1][1]) && same(r[2], hw[1][2]), hw)
        # every budget with the doubling off and every child capacity; capacity 1 at a cohort of
        # 64 and a budget of 1 overflows the child queue in every iteration with two children
        ex = PoolExecutor(DK.wavehosts(2))
        for cap in (nothing, 1, 3)
            torus_reruns = 0
            for K in (1, 7, 32, typemax(Int))
                a, ra, b, rb = with_wavefront_settings(budget = K, doubling = false, cohort = 64, refill = 37) do
                    a = mc_spectrum(prepare(ss), 1_500; kw..., capacity = cap, backend = ex)
                    ra = describe(ex).reruns
                    b = mc_spectrum(pt, 800; tk..., capacity = cap, backend = ex)
                    (a, ra, b, describe(ex).reruns)
                end
                @test same(a, hw[1][1]) && same(b, hw[1][2]) && all(>(0), describe(ex).admitted)
                cap === nothing || @test ra > 0
                torus_reruns += rb
            end
            cap === nothing || @test torus_reruns > 0
        end
        # the cohort sorted by the key stage before every advance (the default) and every fourth,
        # and never: the same bits
        @test describe(ex).settings.sort_every == 1 && count(it -> it.sorted, reduce(vcat, wave_log(ex))) > 2
        for every in (0, 1, 4), cap in (nothing, 1)
            a, b, ns = with_wavefront_settings(sort_every = every, cohort = 64, refill = 37) do
                a = mc_spectrum(prepare(ss), 1_500; kw..., capacity = cap, backend = ex)
                ns = count(it -> it.sorted, reduce(vcat, wave_log(ex)))
                (a, mc_spectrum(pt, 800; tk..., capacity = cap, backend = ex), ns)
            end
            @test same(a, hw[1][1]) && same(b, hw[1][2]) && (every == 0 ? ns == 0 : ns > 2)
            @test describe(ex).settings.sort_every == every
        end
        # the straight-line march of the sphere: bitwise the threads; the curved march of the
        # torus runs track! in another inlining context on threads: equal counts, sums within 1e-11
        @test same(hw[1][1], sph)
        @test close_to(hw[1][2], tor, 1.0e-11) && hw[1][2].n_children == tor.n_children
        # the cascade of the same pool compiles the programs of threads: their bits
        @test same(with_wavefront_settings(() -> mc_spectrum(pt, 800; tk..., backend = ex);
                                           wavefront = false), tor)
        @test describe(ex).schedule == :cascade
    end

    @testset "block costs of a source: the mean inverse zone radius of a block's primaries" begin
        site(r) = ZoneSite(SVector(0.0, 0.0, r, 0.0), 1.0, 1.0, 1.0, 1.0)
        zs = ZoneSource([site(2.0), site(4.0), site(8.0)], nothing, nothing, nothing, nothing,
                        nothing, nothing, [0, 10, 30, 31])
        @test Products.source_costs(zs, 31, 10) == [0.5, 0.25, 0.25, 0.125]
        @test Products.source_costs(zs, 31, 31) ≈ [(10 / 2 + 20 / 4 + 1 / 8) / 31] rtol = 0.05
        @test Execution.block_order(MCStart(zs, nothing, nothing, 0), 31, 10) == [1, 2, 3, 4]
        ss = UniformSphere(n_e = 5.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        tabs, _ = DK.sphere_tables(ss, 1000; nu_min = 1.0e8, nu_max = 1.0e13)
        @test Products.source_costs(SphereSource(ss, tabs), 1000, 100) === nothing
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
        sites = emission_sites(blocks, pt.spacetime, pt.params)[1:50]
        wgt, _, dn = zone_tables(sites, 100.0)
        off = collect(0:2:100)
        ctr = CurvedMarch(pt.spacetime; C = 32.0, R_em = 30.0)
        ztr = MCTransport(ctr, pt.spacetime, blocks, Opacities(pt.params; table = THOMSON, bias = Bias(1.0e5),
                                                                max_scatt = 3), Val(true), Val(true))
        zsink = MCSink(ctr, (ExactBins(InclinationBands((17.0, 50.0), 10.0)),),
                       ChildBuffer(Superphoton, 4096, layout), mc_ledger())
        src = ZoneSource(sites, blocks, pt.spacetime, pt.params, wgt, EMISSIVITY_TABLE, dn, off)
        zp = MCPrimary(src, ztr, zsink, UInt32(1))
        @test DK.device_isbits(zp) && allocs(zp, 17, 17) == 0
        @test ledger_count(zsink.ledger, 6) == 2
        # the stage programs of the wavefront: isbits on a device, with and without queues
        states = Vector{MarchState}(undef, 8); done = Vector{MarchState}(undef, 8)
        cnt = zeros(Int64, 5)
        kids = Generation([generate(ss, tabs, primary_rng(3, g))[1] for g in 1:2],
                          [primary_rng(3, g) for g in 1:2])
        q = WaveQueues(Cohort(states, nothing), StateBuffer(states, view(cnt, 1:1)), kids,
                       StateBuffer(done, view(cnt, 2:2)), cnt, 32, 0, 3)
        stages = (MCStart(src, ztr, zsink, UInt32(1)), MCAdvance(ztr, zsink), MCRetire(ztr, zsink),
                  MCBranch(ztr, zsink))
        @test all(DK.device_isbits, stages)
        st, adv, ret, br = map(x -> Adapt.adapt(WaveSwap(q, nothing), x), stages)
        @test all(DK.device_isbits, (st, adv, ret, br))
        @test allocs(st, 17, 1) == 0 && ledger_count(zsink.ledger, 6) == 4
        @test allocs(adv, 1, 1) == 0 && cnt[1] + cnt[2] == 2 && cnt[5] == 0
        done[1] = states[1]
        @test allocs(ret, 1, 1) == 0
        @test wave_state(adv) === wave_state(br) === MarchState && wave_steps(adv) == ctr.nmax
        # the key stage stores and counts a member's generation, then its photon_key
        @test wave_keys(MCKey(ztr)) == 4 * PHOTON_KEYS && wave_state(MCKey(ztr)) === MarchState
        kc = KeyCount(Cohort(states, nothing), zeros(UInt32, 8), zeros(Int32, wave_keys(MCKey(ztr))))
        key = Adapt.adapt(WaveSwap(kc, nothing), MCKey(ztr))
        @test DK.device_isbits(MCKey(ztr)) && DK.device_isbits(key)
        @test allocs(key, 1, 1) == 0 && sum(kc.counts) == 2
        @test kc.keys[1] == generation(states[1]) * PHOTON_KEYS + photon_key(pt.spacetime, ctr, states[1].y)
        @test kc.counts[kc.keys[1] + 1] == 2
        # the branch opens a child into the live queue; beyond the depth it counts it dropped
        cnt .= 0
        @test allocs(br, 2, 1) == 0 && cnt[1] == 2 && cnt[4] == 0
        @test generation(states[1]) == kids.photons[2].nscatt == 0
        k1 = kids.photons[1]
        once = Generation([Superphoton(k1.x, k1.p, k1.w; nscatt = Int32(1))], kids.rngs[1:1])
        for (depth, live, dropped) in ((1, 3, 0), (0, 3, 1))
            b = Adapt.adapt(WaveSwap(WaveQueues(nothing, q.B, once, nothing, cnt, 32, 0, depth), nothing),
                            MCBranch(ztr, zsink))
            b(1, 1)
            @test cnt[1] == live && cnt[4] == dropped
        end
        @test generation(states[3]) == 1
        @test length(mc_ledger().n) == 11 * LEDGER_STRIPES
        # the zone a primary is generated from is the site's, decoded again
        z9 = BlackLightPlus.Products.emit(src, 17, primary_rng(1, 17))[1]
        @test z9.x == sites[9].x && z9.ne0 == sites[9].n_e && z9.thetae0 == sites[9].theta_e
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
        @test hotcross_table(prepare(ss)) === t                  # one table per process
        # the table built on any executor is the host's
        for exec in (ThreadedExecutor(), POOLS[2], PoolExecutor(DK.devices(1)))
            @test hotcross_build(exec).logsigma == t.logsigma
        end
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

@testset "photon order key" begin
    kerr = KerrSchildCartesian(0.9)
    bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.3], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    flat = FlatMarch(10.0, 0.05, 100)
    mink = CurvedMarch(Minkowski(); R_em = 30.0)
    # the shell floor: the horizon floor, else the march radius times e^-6.4
    cases = ((Minkowski(), flat, 10.0 * exp(-6.4)), (Minkowski(), mink, 30.0 * exp(-6.4)),
             (kerr, CurvedMarch(kerr), horizon_floor(kerr)), (bin, CurvedMarch(bin), 0.9))
    p = (-1.0, 0.3, -0.2, 0.1)
    n = SVector(1.0, -2.0, 2.0) / 3.0                       # octant 2
    for (st, ctl, r_min) in cases
        # the centre of shell j is shell j; past the last shell, the last
        for j in 0:127
            y = state((0.0, (r_min * exp(0.05 * (j + 0.5)) * n)...), p)
            @test photon_key(st, ctl, y) == expected_key(st, j, y)
        end
        far = state((0.0, (r_min * exp(9.0) * n)...), p)
        @test photon_key(st, ctl, far) == expected_key(st, 127, far) >= 127 * 64
        # r = 0, r = r_min and below: shell 0, octants still from the sign bits
        for y in (state((0.0, 0.0, 0.0, 0.0), p), state((0.0, -0.0, 0.0, -0.0), p),
                  state((0.0, 0.0, 0.0, r_min), p), state((0.0, (0.5 * r_min * n)...), p))
            @test photon_key(st, ctl, y) == expected_key(st, 0, y) < 64
        end
        @test photon_key(st, ctl, state((0.0, -0.0, 0.0, -0.0), p)) ==
              5 * 8 + photon_key(st, ctl, state((0.0, 0.0, 0.0, 0.0), p))
        # a non-finite entry of the position or the momentum: key 0
        for bad in (NaN, Inf, -Inf), j in 1:8
            y = setindex(state((0.0, (2.0 * n)...), p), bad, j)
            @test photon_key(st, ctl, y) == 0
        end
        ys = key_cloud(20_000)
        ks = [photon_key(st, ctl, y) for y in ys]
        @test all(k -> 0 <= k < PHOTON_KEYS, ks) && PHOTON_KEYS == 8192
        @test length(unique(ks .% 64)) == 64 && extrema(ks .÷ 64) == (0, 127)
        @test allocs(photon_key, st, ctl, ys[1]) == 0
        # the same keys on threads and on host pools of every device count
        prog = KeyProgram(st, ctl, ys)
        @test DK.device_isbits(prog)
        for exec in (ThreadedExecutor(), POOLS..., PoolExecutor(DK.hosts(1)),
                     PoolExecutor(DK.wavehosts(2)))
            out = fill(-1, length(ys))
            foreach_item!(exec, prog, 1:length(ys); outs = (out,))
            @test out == ks
        end
    end
    # the octant bits in order x¹, x², x³ (4, 2, 1): a position and a wave vector (−, +, +)
    y = state((0.0, -1.0, 2.0, 3.0), (-1.0, -1.0, 0.5, 0.5))
    @test photon_key(Minkowski(), flat, y) % 64 == 4 * 8 + 4
    @test photon_key(Minkowski(), flat, state((0.0, 1.0, 2.0, -3.0), (-1.0, 1.0, 0.5, -0.5))) % 64 == 1 * 8 + 1
    # a radius that overflows has shell 0; an inverse metric that is not finite, key 0
    @test photon_key(Minkowski(), flat, state((0.0, 1.0e200, -1.0e200, 1.0), p)) == 2 * 8 + 2
    @test !all(isfinite, inverse_metric(bin, SVector(0.0, 3.0, 0.0, 0.0)))
    @test photon_key(bin, CurvedMarch(bin), state((0.0, 3.0, 0.0, 0.0), p)) == 0
    @test photon_key(kerr, CurvedMarch(kerr), state((0.0, 1.0e200, 0.0, 0.0), p)) == 0
end
