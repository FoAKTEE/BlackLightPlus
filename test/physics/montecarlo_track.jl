# The tracker and its controls: the straight-line march steps and stops as specified and
# its absorption depth is α_ν times the path; the scattering events of one step deposit
# the physical child weight at the physical positions for every depth and bias; the
# biases follow their rules; every march closes its weight ledger and spawns at every
# valid event site; the curved march captures infalling and releases outgoing photons
# and stops plunging photons without changing the bins; the torus deck over Kerr gives
# null seeds inside emitting zones, conserves energy without scattering, peaks in the
# synchrotron band and grows a Compton tail with scattering, identically under every
# executor.
using Test, BlackLightPlus
using StaticArrays
using LinearAlgebra: dot, norm
using BlackLightPlus.Numerics: MCRng, mc_uniform, primary_rng, MCStreamLayout, HPL, ME, CL, SIGMA_THOMSON
using BlackLightPlus.Spacetime: Minkowski, KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit,
                     ETA, inverse_metric, null_covariant, proxy_distance, horizon_floor,
                     horizon_radius, ks_radius, photon_orbit_radius, plunge_radius, PLUNGE_MARGIN
using BlackLightPlus.Matter: UniformSphere, sample
using BlackLightPlus.MonteCarlo: Superphoton, FlatMarch, CurvedMarch, Opacities, Bias, StateBias, NullSink,
                      VectorSink, track!, escaped, captured, plunging, escape_direction,
                      alpha_nu, generate, GenerationTables, zone_nsuperphotons, event_weight,
                      next_event, bias, event_bias, amplification, HotcrossTable, NW, NT,
                      ESCAPED_MARCH, CAPTURED_MARCH, DROPPED_MARCH, SPHERE_STEP_FRAC
using BlackLightPlus.Transfer: march_control, geodesic_step, max_steps
using BlackLightPlus.Execution: ThreadedExecutor, PoolExecutor, Spectrum, nuLnu_total, scalar, addends
using BlackLightPlus.Data: build_weight_table, load_hotcross, emission_sites

const DK = Main.Decks
const SPH = UniformSphere()
const HOTCROSS_DAT = DK.HOTCROSS_DAT

# A photon of energy E at x travelling along the unit direction n, flat coordinates.
flat_photon(x, n, E; w = 1.0) = Superphoton(SVector(0.0, x...), ETA * SVector(E, (E .* n)...), w)

peak_bin(L, cnt) = argmax([c > 0 ? l : -Inf for (l, c) in zip(L, cnt)])

# A sink that sums the children's weight and its first two moments in the fraction of a
# unit step along x¹ (the photon of `E = 1e-6` from the origin), the child's stream its
# parent's.
struct StepSink <: BlackLightPlus.MonteCarlo.Sink
    acc::Vector{Float64}
end
BlackLightPlus.MonteCarlo.child_stream(::StepSink, rng, event) = (rng, true)
function BlackLightPlus.MonteCarlo.spawn!(s::StepSink, child, rng)
    f = child.x[2] / 1.0e-6
    s.acc[1] += child.w; s.acc[2] += child.w * f; s.acc[3] += child.w * f * f
    return nothing
end
BlackLightPlus.MonteCarlo.record!(::StepSink, sph) = nothing

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
        @test length(sink.escaped) == 1 && sink.escaped[1] === r2.sph && length(sink.ledger) == 1
        l = sink.ledger[1][2]
        @test l.outcome == ESCAPED_MARCH && l.w === r2.sph.w && l.E === r2.sph.E
        @test l.scatter_loss == l.spawned == l.unspawned == 0.0 && l.events == 0
        @test l.absorbed + l.w * l.E ≈ sp.w * sp.E rtol = 1e-12          # absorbed power closes
        r0 = track!(ctl, Minkowski(), SPH, Opacities(), sp, MCRng(1, 1), NullSink();
                    absorb = Val(false))
        @test r0.sph.w === 3.0 && r0.sph.tau_abs === 0.0 && r0.sph.x === r.sph.x
    end

    @testset "scattering events: one step, exact in expectation" begin
        # Per step of depths (s, a), Δτ = s + a, the physical child weight density is
        # w s e^{−Δτ f} df: total w s (1 − e^{−Δτ})/Δτ, first moment
        # w s (1 − (1 + Δτ) e^{−Δτ})/Δτ². The events, Poisson of mean b s at fractions of
        # density Δτ e^{−Δτ f}/(1 − e^{−Δτ}), each of weight W, reproduce both for every b.
        grid = [(ds, da, b) for ds in (0.01, 0.3, 2.0) for da in (0.0, 0.5, 5.0)
                for b in (10.0, 100.0, 1000.0)]
        res = Vector{NTuple{3,Float64}}(undef, length(grid))
        Threads.@threads for k in eachindex(grid)
            ds, da, b = grid[k]
            dtau = ds + da
            W = event_weight(1.0, b, dtau)
            M = ceil(Int, 4.0e7 / (b * ds))
            sw = 0.0; swf = 0.0; fmin = 1.0; fmax = 0.0
            for i in 1:M
                rng = MCRng(0x5A5A, k * 10^9 + i)
                u = 0.0
                while true
                    u, f, rng = next_event(u, b * ds, dtau, rng)
                    u < 1.0 || break
                    sw += W; swf += W * f
                    fmin = min(fmin, f); fmax = max(fmax, f)
                end
            end
            q = -expm1(-dtau)
            res[k] = ((sw / M) / (ds * q / dtau), (swf / M) / (ds * (q - dtau * exp(-dtau)) / dtau^2),
                      (fmin >= 0.0 && fmax < 1.0) ? 1.0 : 0.0)
        end
        @test all(r -> abs(r[1] - 1.0) < 1.0e-3, res)
        @test all(r -> abs(r[2] - 1.0) < 1.0e-3, res)
        @test all(r -> r[3] == 1.0, res)
        for (ds, da, b) in grid
            dtau = ds + da
            @test event_weight(2.0, b, dtau) <= 2.0 / b
            @test event_weight(2.0, b, dtau) ≈ 2.0 * (1 - exp(-dtau)) / (b * dtau) rtol = 1e-14
        end
        # the fraction inverts the CDF u = (1 − e^{−Δτ f})/(1 − e^{−Δτ}); one draw ends a quiet step
        for dtau in (1.0e-12, 0.1, 3.0, 40.0)
            u, f, r = next_event(0.0, 1.0e-300, dtau, MCRng(1, 2))
            @test u >= 1.0 && f == 1.0 && r == mc_uniform(MCRng(1, 2))[2]
            u, f, _ = next_event(0.0, 1.0e3, dtau, MCRng(1, 3))
            @test u < 1.0
            @test expm1(-dtau * f) / expm1(-dtau) ≈ u rtol = 1e-9
        end
    end

    @testset "state bias: one step at b of order 1" begin
        # The bias b = event_bias(StateBias(0.3, …), Θₑ, w) of a step (1 ≤ b ≤ 63 on this grid)
        # is chosen before the step's draws, so its events deposit the physical child weight
        # w Δτ_s (1 − e^{−Δτ})/Δτ with the density ∝ e^{−Δτ f}: the deposit and the first two
        # moments of the event fraction to 1e-3 over a grid of (Δτ_s, Δτ_a, w/w_ref, Θₑ).
        sb = StateBias(0.3, 1.0, 1.0e6, 1)
        grid = [(ds, da, wr, th) for ds in (0.1, 1.0) for da in (0.0, 1.0, 10.0)
                for wr in (0.03, 1.0, 10.0) for th in (0.05, 1.0)]
        mom(n, ds, dt) = ds * (n == 0 ? -expm1(-dt) / dt :
                              n == 1 ? (-expm1(-dt) - dt * exp(-dt)) / dt^2 :
                                       (2 - exp(-dt) * (dt^2 + 2dt + 2)) / dt^3)
        res = Vector{NTuple{4,Float64}}(undef, length(grid))
        Threads.@threads for k in eachindex(grid)
            ds, da, wr, th = grid[k]
            b, copies = event_bias(sb, 0, th, wr)
            dt = ds + da
            W = event_weight(wr, b, dt) / copies
            M = ceil(Int, 3.0e7 / (b * ds))
            m0 = 0.0; m1 = 0.0; m2 = 0.0
            for i in 1:M
                rng = MCRng(0x5B5B, k * 10^9 + i)
                u = 0.0
                while true
                    u, f, rng = next_event(u, b * ds, dt, rng)
                    u < 1.0 || break
                    m0 += copies * W; m1 += copies * W * f; m2 += copies * W * f * f
                end
            end
            res[k] = (b, m0 / (M * wr * mom(0, ds, dt)), m1 / (M * wr * mom(1, ds, dt)),
                      m2 / (M * wr * mom(2, ds, dt)))
        end
        bs = first.(res)
        @test minimum(bs) == 1.0 && count(<(10.0), bs) >= 24 && maximum(bs) > 50.0
        @test all(r -> abs(r[2] - 1) < 1.0e-3 && abs(r[3] - 1) < 1.0e-3 && abs(r[4] - 1) < 1.5e-3, res)
        # the same for a photon scattered once and twice under the order factor 4 (b from the
        # photon's order as well as its weight and gas): the expectation holds for each b
        so = StateBias(0.3, 1.0, 1.0e6, 1, 4.0)
        go = [(ds, da, wr, th, n) for ds in (0.1, 1.0) for da in (0.0, 1.0, 10.0)
              for wr in (0.03, 1.0) for th in (0.05, 1.0) for n in (1, 2)]
        ro = Vector{NTuple{4,Float64}}(undef, length(go))
        Threads.@threads for k in eachindex(go)
            ds, da, wr, th, n = go[k]
            b, copies = event_bias(so, n, th, wr)
            dt = ds + da
            W = event_weight(wr, b, dt) / copies
            M = ceil(Int, 3.0e7 / (b * ds))
            m0 = 0.0; m1 = 0.0; m2 = 0.0
            for i in 1:M
                rng = MCRng(0x5D5D, k * 10^9 + i)
                u = 0.0
                while true
                    u, f, rng = next_event(u, b * ds, dt, rng)
                    u < 1.0 || break
                    m0 += copies * W; m1 += copies * W * f; m2 += copies * W * f * f
                end
            end
            ro[k] = (b, m0 / (M * wr * mom(0, ds, dt)), m1 / (M * wr * mom(1, ds, dt)),
                     m2 / (M * wr * mom(2, ds, dt)))
        end
        @test minimum(first.(ro)) == 1.0 && maximum(first.(ro)) > 30.0
        @test all(r -> abs(r[2] - 1) < 1.0e-3 && abs(r[3] - 1) < 1.0e-3 && abs(r[4] - 1) < 1.5e-3, ro)
        # the same through the tracker's step: kick, child streams and sink, the event fraction
        # read back from the child's position on a straight line
        thom = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
        s = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        E = 1.0e-6
        for (ds, da, wr, th) in ((0.1, 1.0, 1.0, 0.05), (1.0, 0.0, 1.0, 1.0), (0.3, 10.0, 10.0, 0.05))
            b, copies = event_bias(sb, 0, th, wr)
            sink = StepSink(zeros(3))
            m = BlackLightPlus.MonteCarlo.March(FlatMarch(s), Minkowski(), s,
                                                Opacities(; table = thom, bias = sb, max_scatt = 1), sink)
            y = vcat(SVector(0.0, 0.0, 0.0, 0.0), SVector(-E, E, 0.0, 0.0))
            sph = Superphoton(SVector(0.0, 0.0, 0.0, 0.0), SVector(-E, E, 0.0, 0.0), wr)
            stp = (it = 1, y = y, h = 1.0, tau_a = 0.0, tau_s = 0.0)
            M = ceil(Int, 1.0e6 / (b * ds))
            for i in 1:M
                BlackLightPlus.MonteCarlo.step_events(m, sph, stp, wr, ds, ds + da, b, copies, 0,
                                                      MCRng(0x5C5C, i))
            end
            dt = ds + da
            @test abs(sink.acc[1] / (M * wr * mom(0, ds, dt)) - 1) < 4.0e-3
            @test abs(sink.acc[2] / (M * wr * mom(1, ds, dt)) - 1) < 5.0e-3
            @test abs(sink.acc[3] / (M * wr * mom(2, ds, dt)) - 1) < 6.0e-3
        end
    end

    @testset "biases" begin
        @test bias(Bias(0.5), 0) == 1.0 && bias(Bias(2.0), 1) == 8.0 && bias(Bias(NaN), 0) == 1.0
        @test Bias(SPH).b0 == SPH.n_e * SIGMA_THOMSON * SPH.R
        @test event_bias(Bias(2.0), 1, 50.0, 1.0e40) == (8.0, 1)
        @test amplification(0.0) == 1.0 && amplification(3.0) == 1.0 + 12.0 + 144.0
        sb = StateBias(2.0, 1.0e10, 1.0e5, 1)
        @test event_bias(sb, 0, 3.0, 1.0e10) == (2.0 * 157.0, 1)
        @test event_bias(sb, 2, 3.0, 2.0e10) == (4.0 * 157.0, 1)          # no order factor
        so = StateBias(2.0, 1.0e10, 1.0e7, 1, 3.0)                            # order factor 3
        @test event_bias(so, 0, 3.0, 1.0e10) == (2.0 * 157.0, 1)
        @test event_bias(so, 2, 3.0, 2.0e10) == (4.0 * 157.0 * 9.0, 1)
        @test event_bias(StateBias(2.0, 1.0e10, 1.0e3, 8, 3.0), 2, 3.0, 2.0e10) == (1.0e3, 6)
        # an order factor of 1 is the bias without one, bit for bit
        for n in 0:3, th in (0.1, 3.0, 50.0), w in (1.0e8, 2.0e10, 3.0e12)
            @test event_bias(StateBias(2.0, 1.0e10, 1.0e5, 4, 1.0), n, th, w) ===
                  event_bias(StateBias(2.0, 1.0e10, 1.0e5, 4), n, th, w)
        end
        @test event_bias(sb, 0, 0.0, 1.0e8) == (1.0, 1)                    # floor at 1
        @test event_bias(sb, 0, 100.0, 1.0e12) == (1.0e5, 1)               # cap, no split
        @test event_bias(StateBias(2.0, 1.0e10, 1.0e5, 8), 0, 100.0, 1.0e12)[2] == 8
        @test event_bias(StateBias(2.0, 1.0e10, 1.0e7, 8), 0, 100.0, 1.0e12) == (1.0e7, 4)
        @test event_bias(StateBias(2.0, 1.0e10, 1.0e5, 8), 0, NaN, 1.0e12) == (1.0, 1)
        @test isbits(sb) && isbits(Opacities(; bias = sb))
    end

    @testset "ledger closure, edge events, layout overflow" begin
        thom = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
        s = UniformSphere(n_e = 3.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        tabs, _ = DK.sphere_tables(s, 1000; nu_min = 1.0e8, nu_max = 1.0e13)
        ctl = FlatMarch(s)
        sink = VectorSink(MCStreamLayout(10_000, 2^20, 2))
        opac = Opacities(; table = thom, bias = Bias(30.0), max_scatt = 2)
        loss = 0.0; got = 0.0; worst = 0.0
        for g in 1:1000, (A, k) in ((Val(true), 0), (Val(false), 1000))
            sp, rng = generate(s, tabs, primary_rng(9, g + k))
            r = track!(ctl, Minkowski(), s, opac, sp, rng, sink; absorb = A, scatter = Val(true))
            l = sink.ledger[end][2]
            # every photon: w₀E = absorbed + E (scattered weight) + w E (flat: E fixed)
            worst = max(worst, abs((l.absorbed + sp.E * (l.scatter_loss + l.scatter_loss_final) +
                                    l.w * l.E) / (sp.w * sp.E) - 1.0))
            loss += l.scatter_loss; got += l.spawned + l.unspawned
            @test l.unspawned == 0.0 && l.scatter_loss_final == 0.0
        end
        @test worst < 1.0e-12
        @test length(sink.children) > 1000
        @test abs(got / loss - 1.0) < 0.1                    # heavy-tailed weights: a loose check
        ids = [BlackLightPlus.Numerics.stream_id(c[2]) for c in sink.children]
        @test all(c -> c[1].nscatt == 1 && BlackLightPlus.Numerics.stream_layer(c[2]) == 1, sink.children)
        @test allunique(ids)
        # a photon of the last order scatters out of the beam but spawns nothing
        sp, rng = generate(s, tabs, primary_rng(9, 1))
        last = Superphoton(sp.x, sp.p, sp.w; nscatt = 2)
        n0 = length(sink.children)
        track!(ctl, Minkowski(), s, opac, last, rng, sink; absorb = Val(false), scatter = Val(true))
        l = sink.ledger[end][2]
        @test length(sink.children) == n0 && l.scatter_loss == 0.0 && l.scatter_loss_final > 0.0
        @test l.events == 0
        # ordinals beyond the layout's event field spawn nothing and are counted
        small = VectorSink(MCStreamLayout(100, 2, 2))
        track!(ctl, Minkowski(), s, Opacities(; table = thom, bias = Bias(1.0e4), max_scatt = 2), sp,
               rng, small; absorb = Val(false), scatter = Val(true))
        l = small.ledger[end][2]
        @test length(small.children) == 2 && l.events > 2 && l.unspawned > 0.0 && l.spawned > 0.0
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
        @test !rin.recorded && ks_radius(st, rin.sph.x) < ctl.r_plunge     # stopped plunging
        lin = VectorSink(MCStreamLayout(10, 10, 1))
        nop = CurvedMarch(st; C = 40.0, escape = 200.0, R_em = 30.0, plunge = false)
        rnp = track!(nop, st, UniformSphere(R = 0.5), plasma, Superphoton(x, pin, 1.0),
                     MCRng(1, 1), lin; absorb = Val(false))
        @test !rnp.recorded && ks_radius(st, rnp.sph.x) < ctl.r_h + 1e-2
        @test lin.ledger[1][2].outcome == CAPTURED_MARCH
        @test rout.recorded && 30.0 < ks_radius(st, rout.sph.x) < 32.0
        k = inverse_metric(st, rout.sph.x) * rout.sph.p
        @test escape_direction(ctl, rout.sph) ≈ SVector(k[2], k[3], k[4]) rtol = 1e-14
        @test rout.sph.p[1] ≈ pout[1] rtol = 1e-8                # conserved energy
    end

    @testset "plunge radius" begin
        for a in (0.0, 0.5, 0.9, -0.9, 0.998)
            st = KerrSchildCartesian(a)
            @test plunge_radius(st) == photon_orbit_radius(abs(a); prograde = true) - PLUNGE_MARGIN
            @test plunge_radius(st) < 3.0
            abs(a) <= 0.9 && @test plunge_radius(st) > horizon_radius(st)
        end
        @test plunge_radius(KerrSchildCartesian(0.9)) ≈ 1.5578 - 0.02 atol = 1e-4
        @test plunge_radius(Minkowski()) == -Inf
        st = KerrSchildCartesian(0.9)
        c = CurvedMarch(st); cn = CurvedMarch(st; plunge = false)
        @test c.r_plunge == plunge_radius(st) && cn.r_plunge == -Inf
        @test plunging(c, 1.5, 1.52) && !plunging(c, 1.5, 1.49) && !plunging(c, 1.5, Inf)
        @test !plunging(c, 1.55, 1.6) && !plunging(cn, 1.5, 1.52) && !plunging(FlatMarch(SPH), 0.0, 1.0)
    end

    snap = DK.amr_deck()
    blocks = snap.blocks
    st = KerrSchildCartesian(DK.TORUS_A)
    params = DK.torus_params()
    zones = DK.torus_zones(blocks, st, params)
    sites = emission_sites(blocks, st, params)

    @testset "curved control on the moving binary" begin
        orbit = CircularOrbit((0.5, 0.5), 8.0)
        st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
        ctl = CurvedMarch(st; C = 40.0, escape = 200.0, R_em = 30.0)
        @test ctl.r_h == horizon_floor(st) == minimum(st.r_h)
        s1 = orbit(0.0)[1][1]
        x = SVector(0.0, s1[1], s1[2] + 3.0, s1[3])               # 3 M beside hole 1, at its t = 0 position
        y = vcat(x, null_covariant(st, x, SVector(0.0, -1.0, 0.0))[1])
        p0 = SVector(y[5], y[6], y[7], y[8])
        d = march_control(ctl, y, Inf, 1)
        kx = inverse_metric(st, x) * p0
        @test !d.stop && d.R == norm(x[SVector(2, 3, 4)]) && d.h ≈ proxy_distance(st, x)[1] / (40.0 * norm(kx[SVector(2, 3, 4)]))
        # a point deep inside hole 2 is a capture, not an escape
        s2 = orbit(0.0)[2][1]
        yc = vcat(SVector(0.0, s2[1], s2[2], s2[3] + 0.5), p0)
        dc = march_control(ctl, yc, Inf, 1)
        @test dc.stop && !escaped(ctl, dc)
        # beyond R_em and outbound: an escape
        yo = vcat(SVector(0.0, 31.0, 0.0, 0.0), p0)
        do_ = march_control(ctl, yo, 30.5, 1)
        @test do_.stop && escaped(ctl, do_)
        # inbound beyond R_em: the march goes on
        @test !march_control(ctl, yo, 31.5, 1).stop
        pin, _ = null_covariant(st, x, SVector(0.0, -1.0, 0.0))
        pout, _ = null_covariant(st, x, SVector(0.0, 1.0, 0.0))
        plasma = Opacities()
        rin = track!(ctl, st, UniformSphere(R = 0.5), plasma, Superphoton(x, pin, 1.0),
                     MCRng(1, 1), NullSink(); absorb = Val(false))
        rout = track!(ctl, st, UniformSphere(R = 0.5), plasma, Superphoton(x, pout, 1.0),
                      MCRng(1, 1), NullSink(); absorb = Val(false))
        @test !rin.recorded && proxy_distance(st, rin.sph.x)[1] < 1e-2
        @test rin.sph.x[1] > 0.0                                    # forward in time: the hole is read where it is then
        @test rout.recorded && 30.0 < norm(rout.sph.x[SVector(2, 3, 4)]) < 32.0
    end

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
        b0, l0, n0, _ = DK.torus_run(exec, blocks, sites, 1_500, 0xC0FFEE)
        @test n0 > 0 && addends(l0, 5) > 0
        ratio = scalar(l0, 5) / scalar(l0, 6)
        @test 0.0 < ratio <= 1.0 + 1.0e-9
        s0 = Spectrum(b0)
        nu, L, cnt = nuLnu_total(s0)
        ip = peak_bin(L, cnt)
        @test L[ip] > 0.0 && 1.0e9 <= nu[ip] <= 1.0e18
        # stopping the plunging photons changes no bin and no emitted or escaped power
        bp, lp, _, _ = DK.torus_run(exec, blocks, sites, 1_500, 0xC0FFEE; plunge = false)
        @test bp.count == b0.count && bp.sumw == b0.sumw && bp.sumwE == b0.sumwE && bp.sumwE2 == b0.sumwE2
        @test scalar(lp, 5) === scalar(l0, 5) && scalar(lp, 6) === scalar(l0, 6)
        if isfile(HOTCROSS_DAT)
            table = load_hotcross(HOTCROSS_DAT)
            b1, l1, n1, layers = DK.torus_run(exec, blocks, sites, 1_500, 0xC0FFEE;
                                              scattering = true, table = table, bias0 = 1.0e5)
            @test n1 == n0 && sum(length(first(l)) for l in layers) > 0
            nu1, L1, _ = nuLnu_total(Spectrum(b1))
            hi = 30.0 * nu[ip]
            p_scat = sum(L1[i] for i in eachindex(L1) if nu1[i] > hi)
            p_dir = sum(L[i] for i in eachindex(L) if nu[i] > hi)
            @test p_scat > p_dir && p_scat > 0.0
            bq, lq, _, layq = DK.torus_run(exec, blocks, sites, 1_500, 0xC0FFEE; scattering = true,
                                           table = table, bias0 = 1.0e5, plunge = false)
            @test bq.count == b1.count && bq.sumwE == b1.sumwE && bq.sumwE2 == b1.sumwE2
            @test [length(first(l)) for l in layq] == [length(first(l)) for l in layers]
            # the same run on the device path and with other chunkings: identical bits
            for ex in (ThreadedExecutor(chunk = 7), PoolExecutor(DK.hosts(3); chunk = 50))
                b, l, n, lay = DK.torus_run(ex, blocks, sites, 1_500, 0xC0FFEE; scattering = true,
                                            table = table, bias0 = 1.0e5)
                @test b.count == b1.count && b.sumw == b1.sumw && b.sumwE == b1.sumwE
                @test l.acc == l1.acc && l.n == l1.n
                @test sort(map(r -> BlackLightPlus.Numerics.stream_id(r), reduce(vcat, last.(lay)))) ==
                      sort(map(r -> BlackLightPlus.Numerics.stream_id(r), reduce(vcat, last.(layers))))
            end
        else
            @info "gpumonty hotcross.dat not found: torus scattering checks skipped" path = HOTCROSS_DAT
        end
    end
end
