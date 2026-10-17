# The tracker and its controls: the straight-line march steps and stops as specified and
# its absorption depth is α_ν times the path; the curved march captures infalling and
# releases outgoing photons; the torus deck over Kerr gives null seeds inside emitting
# zones, conserves energy without scattering, peaks in the synchrotron band and grows a
# Compton tail with scattering, identically under every executor.
using Test, BLP
using StaticArrays
using LinearAlgebra: dot
using BLP.Numerics: MCRng, primary_rng, MCStreamLayout, HPL, ME, CL
using BLP.Spacetime: Minkowski, KerrSchildCartesian, ETA, inverse_metric, null_covariant,
                     horizon_radius, ks_radius
using BLP.Matter: UniformSphere, sample
using BLP.MonteCarlo: Superphoton, FlatMarch, CurvedMarch, Opacities, Bias, NullSink, VectorSink,
                      track!, escaped, escape_direction, alpha_nu, generate, GenerationTables,
                      zone_nsuperphotons, scatter_event, bias, SPHERE_STEP_FRAC
using BLP.Transfer: march_control, geodesic_step, max_steps
using BLP.Execution: ThreadedExecutor, PoolExecutor, Spectrum, nuLnu_total, scalar, addends
using BLP.Data: build_weight_table, load_hotcross

const DK = Main.Decks
const SPH = UniformSphere()
const HOTCROSS_DAT = DK.HOTCROSS_DAT

# A photon of energy E at x travelling along the unit direction n, flat coordinates.
flat_photon(x, n, E; w = 1.0) = Superphoton(SVector(0.0, x...), ETA * SVector(E, (E .* n)...), w)

peak_bin(L, cnt) = argmax([c > 0 ? l : -Inf for (l, c) in zip(L, cnt)])

@testset "Monte Carlo tracker" begin
    @testset "straight-line control" begin
        ctl = FlatMarch(SPH)
        @test max_steps(ctl) == ctl.nmax + 1 && ctl.step_frac == SPHERE_STEP_FRAC
        sp = flat_photon((0.0, 0.0, 0.0), (0.0, 0.6, 0.8), 2.0)
        y = vcat(sp.x, sp.p)
        d = march_control(ctl, y, Inf, 1)
        @test !d.stop && d.h ≈ SPHERE_STEP_FRAC * SPH.R / 2.0
        y2 = geodesic_step(ctl, y, d.h, 1)
        @test y2[2:4] ≈ [0.0, 0.6, 0.8] * SPHERE_STEP_FRAC atol = 1e-15
        @test SVector(y2[5], y2[6], y2[7], y2[8]) === sp.p
        out = vcat(SVector(0.0, 1.1, 0.0, 0.0), sp.p)
        @test !march_control(ctl, out, Inf, 1).stop          # the emission point is not tested
        dout = march_control(ctl, out, Inf, 2)
        @test dout.stop && escaped(ctl, dout)
        dcap = march_control(ctl, y, Inf, ctl.nmax + 1)
        @test dcap.stop && !escaped(ctl, dcap)
        @test escape_direction(ctl, sp) === SVector(sp.p[2], sp.p[3], sp.p[4])
    end

    @testset "sphere: absorption depth along a radius" begin
        ctl = FlatMarch(SPH)
        E = 1.0e-9
        nu = E * ME * CL * CL / HPL
        sp = flat_photon((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), E; w = 3.0)
        r = track!(ctl, Minkowski(), SPH, Opacities(), sp, MCRng(1, 1), NullSink())
        @test r.recorded
        @test sqrt(sum(abs2, r.sph.x[2:4])) > SPH.R
        a = alpha_nu(SPH.n_e, SPH.theta_e, SPH.B, nu, 1.0)
        @test r.sph.tau_abs ≈ a * SPH.R rtol = SPHERE_STEP_FRAC
        @test r.sph.w ≈ 3.0 * exp(-r.sph.tau_abs) rtol = 1e-12
        @test r.rng === MCRng(1, 1)                              # absorption draws nothing
        sink = VectorSink(MCStreamLayout(10, 10, 1))
        r2 = track!(ctl, Minkowski(), SPH, Opacities(), sp, MCRng(1, 1), sink)
        @test length(sink.escaped) == 1 && sink.escaped[1] === r2.sph && isempty(sink.ledger)
        r0 = track!(ctl, Minkowski(), SPH, Opacities(), sp, MCRng(1, 1), NullSink();
                    absorb = Val(false))
        @test r0.sph.w === 3.0 && r0.sph.tau_abs === 0.0 && r0.sph.x === r.sph.x
    end

    @testset "scattering decision" begin
        @test bias(Bias(0.5), 0) == 1.0 && bias(Bias(2.0), 1) == 8.0 && bias(Bias(NaN), 0) == 1.0
        @test Bias(SPH).b0 == SPH.n_e * BLP.Numerics.SIGMA_THOMSON * SPH.R
        n = 0; wsum = 0.0; fracs_ok = true
        rng = MCRng(4, 4)
        for _ in 1:100_000
            ev = scatter_event(1.0, 3.0, 0.01, 0.0, rng)
            rng = ev.rng
            ev.spawn && (n += 1; wsum += ev.weight; fracs_ok &= 0.0 <= ev.frac < 1.0)
        end
        @test fracs_ok
        @test n / 100_000 ≈ -expm1(-0.03) rtol = 0.05            # spawn probability
        @test wsum / 100_000 ≈ -expm1(-0.01) rtol = 0.05         # expected spawned weight
    end

    @testset "curved control: capture and escape" begin
        st = KerrSchildCartesian(0.9)
        ctl = CurvedMarch(st; C = 40.0, escape = 200.0, R_em = 30.0)
        @test ctl.r_h == horizon_radius(st) && max_steps(ctl) == ctl.nmax
        x = SVector(0.0, 12.0, 0.0, 1.0)
        pin, _ = null_covariant(st, x, SVector(-1.0, 0.0, 0.0))
        pout, _ = null_covariant(st, x, SVector(1.0, 0.0, 0.0))
        plasma = Opacities()
        rin = track!(ctl, st, UniformSphere(R = 0.5), plasma, Superphoton(x, pin, 1.0),
                     MCRng(1, 1), NullSink(); absorb = Val(false))
        rout = track!(ctl, st, UniformSphere(R = 0.5), plasma, Superphoton(x, pout, 1.0),
                      MCRng(1, 1), NullSink(); absorb = Val(false))
        @test rin.sph.E > 0 && rout.sph.E > 0
        @test !rin.recorded && ks_radius(st, rin.sph.x) < ctl.r_h + 1e-2
        @test rout.recorded && 30.0 < ks_radius(st, rout.sph.x) < 32.0
        k = inverse_metric(st, rout.sph.x) * rout.sph.p
        @test escape_direction(ctl, rout.sph) ≈ SVector(k[2], k[3], k[4]) rtol = 1e-14
        @test rout.sph.p[1] ≈ pout[1] rtol = 1e-8                # conserved energy
    end

    snap = DK.amr_deck()
    blocks = snap.blocks
    st = KerrSchildCartesian(DK.TORUS_A)
    params = DK.torus_params()
    zones = DK.torus_zones(blocks, st, params)

    @testset "torus: seeds are null and start inside emitting zones" begin
        @test length(zones) > 100
        @test all(z -> z.n_e > 0.0 && z.theta_e > 0.0 && z.B > 0.0 && z.dV > 0.0, zones)
        wgt = build_weight_table(zones, 1000.0)
        worst = 0.0; inside = true
        for (i, zone) in enumerate(zones[1:min(end, 200)])
            _, dn = zone_nsuperphotons(zone, wgt)
            dn > 0.0 || continue
            sp, _ = generate(zone, GenerationTables(wgt, dn), primary_rng(0xBEEF, i))
            worst = max(worst, abs(dot(sp.p, zone.gcon * sp.p)) / sp.E^2)
            inside &= sample(blocks, st, sp.x, params)[1].valid
        end
        @test worst < 1.0e-8
        @test inside
    end

    @testset "torus: energy, synchrotron peak, Compton tail" begin
        exec = ThreadedExecutor(chunk = 64)
        b0, l0, n0, _ = DK.torus_run(exec, blocks, zones, 1_500, 0xC0FFEE)
        @test n0 > 0 && addends(l0, 5) > 0
        ratio = scalar(l0, 5) / scalar(l0, 6)
        @test 0.0 < ratio <= 1.0 + 1.0e-9
        s0 = Spectrum(b0)
        nu, L, cnt = nuLnu_total(s0)
        ip = peak_bin(L, cnt)
        @test L[ip] > 0.0 && 1.0e9 <= nu[ip] <= 1.0e18
        if isfile(HOTCROSS_DAT)
            table = load_hotcross(HOTCROSS_DAT)
            b1, l1, n1, layers = DK.torus_run(exec, blocks, zones, 1_500, 0xC0FFEE;
                                              scattering = true, table = table, bias0 = 1.0e5)
            @test n1 == n0 && sum(length(first(l)) for l in layers) > 0
            nu1, L1, _ = nuLnu_total(Spectrum(b1))
            hi = 30.0 * nu[ip]
            p_scat = sum(L1[i] for i in eachindex(L1) if nu1[i] > hi)
            p_dir = sum(L[i] for i in eachindex(L) if nu[i] > hi)
            @test p_scat > p_dir && p_scat > 0.0
            # the same run on the device path and with other chunkings: identical bits
            for ex in (ThreadedExecutor(chunk = 7), PoolExecutor(DK.hosts(3); chunk = 50))
                b, l, n, lay = DK.torus_run(ex, blocks, zones, 1_500, 0xC0FFEE; scattering = true,
                                            table = table, bias0 = 1.0e5)
                @test b.count == b1.count && b.sumw == b1.sumw && b.sumwE == b1.sumwE
                @test l.acc == l1.acc && l.n == l1.n
                @test sort(map(r -> BLP.Numerics.stream_id(r), reduce(vcat, last.(lay)))) ==
                      sort(map(r -> BLP.Numerics.stream_id(r), reduce(vcat, last.(layers))))
            end
        else
            @info "gpumonty hotcross.dat not found: torus scattering checks skipped" path = HOTCROSS_DAT
        end
    end
end
