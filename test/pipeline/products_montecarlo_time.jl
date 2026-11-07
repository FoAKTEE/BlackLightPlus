# The Monte Carlo over an emission window, against its exact tests: the emission-time
# profiles and their exact sampling, the placement of a photon at its emission time and
# its arrival time; T1, a frozen sphere over a window is the stationary sphere bit for bit,
# thin and scattering, with a flat light curve; T5, a snapshot or a series of one frame on
# Kerr is the stationary torus bit for bit, and copies of the frame agree within errors;
# T2, the variable sphere's light curve has the amplitude A sinc²(π/n) 3 j₁(ωR)/(ωR) and
# no phase shift; T3, the light echo of a thin Thomson shell is the closed form; T4, a
# co-rotating plasma on the orbiting binary is helically symmetric, with a flat band light
# curve, and a plasma symmetric under the rotation by π repeats every half period; T6,
# light-curve errors fall as N^-1/2 and match the scatter of independent seeds; and the
# argument checks.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra, Statistics
using BlackLightPlus.Plans: RunConfig, prepare
using BlackLightPlus.Products: mc_spectrum, light_curve, time_spectrum, arrival_times,
                    TimedSpectrumResult, TimeProfile, clip_profile, sample_time, time_stream, place,
                    arrival_time, profile_area
using BlackLightPlus.Matter: UniformSphere, VariableSphere, SphericalShell
using BlackLightPlus.MonteCarlo: HotcrossTable, NW, NT, Bias, Superphoton
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, BoostedSuperposedKerrSchild,
                     CircularOrbit, metric_and_inverse, raise_index
using BlackLightPlus.Data: FrameSequence, Corotating, synthetic_series, SeriesProvider, AthenaKSnapshot
using BlackLightPlus.Execution: Spectrum, InclinationBands, sector_row, ThreadedExecutor
using BlackLightPlus.Numerics: SIGMA_THOMSON, GNEWT, CL, MCRng, mc_uniform, primary_rng
const DK = Main.Decks

const THOMSON = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
const L_TORUS = GNEWT * DK.TORUS_MBH / CL^2
const TORUS_CFG = RunConfig(m_bh_cgs = DK.TORUS_MBH, mass_scale_cgs = DK.TORUS_DENS * L_TORUS^3,
                            frequency_hz = 2.3e11)
sbits(s::Spectrum) = (s.count, DK.bits(s.sumw), DK.bits(s.sumwE), DK.bits(s.sumwE2))
same(a::Spectrum, b::Spectrum) = sbits(a) == sbits(b)
same(a, b) = all(same(getfield(a, f), getfield(b, f)) for f in (:total,)) &&
             all(same.(a.orders, b.orders))
P̂(x) = 3 * (sin(x) - x * cos(x)) / x^3
sinc2(x) = (sin(x) / x)^2

# The interior arrival bins of a result (both edges inside `res.interior`), optionally
# restricted to whole periods P.
function interior_bins(res; P = nothing)
    _, e = arrival_times(res)
    lo, hi = res.interior
    P === nothing || (lo = ceil(lo / P) * P; hi = floor(hi / P) * P)
    return findall(k -> e[k] >= lo - 1e-9 && e[k + 1] <= hi + 1e-9, 1:length(e) - 1)
end

# χ²/n of bins against their weighted mean.
function flatness(L, s)
    w = 1 ./ s .^ 2
    m = sum(w .* L) / sum(w)
    return sum(((L .- m) ./ s) .^ 2) / (length(L) - 1), m
end

# The weighted least-squares fit a + b ⟨sin ωt⟩ + c ⟨cos ωt⟩ over whole periods of the
# interior, with the exact bin averages: (b/a, σ, c/a, σ).
function sine_fit(res, ω)
    t, L, s, _ = light_curve(res)
    _, e = arrival_times(res)
    k = interior_bins(res; P = 2π / ω)
    e0 = e[k]; e1 = e[k .+ 1]
    X = hcat(ones(length(k)), (cos.(ω .* e0) .- cos.(ω .* e1)) ./ (ω .* (e1 .- e0)),
             (sin.(ω .* e1) .- sin.(ω .* e0)) ./ (ω .* (e1 .- e0)))
    W = Diagonal(1 ./ s[k] .^ 2)
    C = inv(X' * W * X)
    c = C * (X' * W * L[k])
    return c[2] / c[1], sqrt(C[2, 2]) / c[1], c[3] / c[1], sqrt(C[3, 3]) / c[1]
end

# The first-order echo fraction per unit delay of the sin² shell, averaged over [a, b].
function echo_fraction(a, b; r_in, Δ, nr = 200, nt = 200)
    Ψ(τ, r) = 0.0 <= τ <= 2r ? 3 / (8r) * (1 + (1 - τ / r)^2) : 0.0
    s = 0.0
    for i in 1:nr
        r = r_in + (i - 0.5) / nr * Δ
        p = 2 * sin(π * (i - 0.5) / nr)^2
        for j in 1:nt
            s += p * Ψ(a + (j - 0.5) / nt * (b - a), r)
        end
    end
    return s / (nr * nt)
end

# The torus deck's mesh with primitives symmetric under the rotation by π about z: the
# deck's density and heat, azimuthal velocity, vertical field (`Decks.amr_deck` layout).
function pi_symmetric_deck(; nc = 6, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, 8, nc, nc, nc, nmb)
    lo = zeros(3, nmb); hi = zeros(3, nmb); xl = zeros(3, nmb); dxs = fill(dx, 3, nmb)
    ll = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        lo[:, m] .= l; hi[:, m] .= l .+ w; xl[:, m] .= l .+ dx / 2
        ll[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            x = xl[1, m] + (i - 1) * dx; y = xl[2, m] + (j - 1) * dx; z = xl[3, m] + (k - 1) * dx
            t = DK.torus_primitives(x, y, z)
            ρ = Float64(t[1])
            ρ > 0 || continue
            r = sqrt(x^2 + y^2 + z^2)
            data[:, i, j, k, m] .= Float32.((ρ, -0.2y / r, 0.2x / r, 0.0, 0.08ρ, 0.0, 0.0, 0.3sqrt(ρ)))
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), ll)
end

@testset "Monte Carlo over an emission window" begin
    @testset "emission-time profiles, placement and arrival time" begin
        hat = TimeProfile([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        @test profile_area(hat) == 1.0
        ramp = TimeProfile([2.0, 5.0], [1.0, 3.0])
        @test profile_area(ramp) == 6.0
        @test clip_profile([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 0.5, 1.5).t == [0.5, 1.0, 1.5]
        @test clip_profile([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 0.5, 1.5).v == [0.5, 1.0, 0.5]
        @test clip_profile([0.0, 4.0], [1.0, 1.0], 1.0, 3.0).t == [1.0, 3.0]
        # exact sampling: the hat (mean 1, variance 1/6), the ramp ∝ 1 + 2(t − 2)/3 on [2, 5]
        # (mean 15/4, variance 11/16) and the flat profile on [0, 3] (mean 3/2, variance 3/4)
        n = 200_000
        for (p, m1, m2) in ((hat, 1.0, 1 / 6), (ramp, 3.75, 0.6875),
                            (TimeProfile([0.0, 3.0], [1.0, 1.0]), 1.5, 0.75))
            x = [sample_time(p, MCRng(9, k))[1] for k in 1:n]
            @test all(t -> p.t[1] <= t <= p.t[end], x)
            @test abs(mean(x) - m1) < 5 * std(x) / sqrt(n)
            @test abs(var(x) / m2 - 1) < 0.02
        end
        r = primary_rng(3, 17)
        ts = time_stream(r)
        @test ts.key == r.key && ts.sid == r.sid && ts.ctr == 0x80000000
        @test first(mc_uniform(ts)) != first(mc_uniform(r))

        # placement: a time translation, and the helical map onto a null photon of the orbit
        o = CircularOrbit((0.5, 0.5), 8.0)
        st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], o)
        x0 = SVector(0.0, 9.0, 3.0, 1.0)
        gcov, gcon = metric_and_inverse(st, x0)
        k = SVector(0.0, 0.3, -0.7, 0.2)                      # a null k with k^t from g(k, k) = 0
        a = gcov[1, 1]; b = 2 * dot(gcov[1, 2:4], k[2:4]); c = dot(k[2:4], gcov[2:4, 2:4] * k[2:4])
        kt = (-b - sqrt(b^2 - 4a * c)) / (2a)
        kt < 0 && (kt = (-b + sqrt(b^2 - 4a * c)) / (2a))
        p = gcov * SVector(kt, k[2], k[3], k[4])
        sph = Superphoton(x0, p, 2.5)
        s0 = place(sph, 37.0, 0.0)
        @test s0.x == SVector(37.0, 9.0, 3.0, 1.0) && s0.p === sph.p && s0.w === sph.w && s0.E === sph.E
        s1 = place(sph, 37.0, o.omega)
        @test s1.E === sph.E && norm(s1.x[2:4]) ≈ norm(x0[2:4]) && s1.x1i == s1.x[2]
        @test abs(dot(s1.p, raise_index(st, s1.x, s1.p))) < 1e-12 * sph.E^2   # null at its new event
        # the arrival time is constant along a straight line
        d = SVector(0.3, -0.5, 0.8)
        xs = [SVector(1.0 + λ * norm(d), 2.0 + λ * d[1], -1.0 + λ * d[2], 0.5 + λ * d[3]) for λ in (0.0, 3.0, 50.0)]
        @test all(xx -> arrival_time(xx, d) ≈ arrival_time(xs[1], d), xs)
        @test arrival_time(xs[1], 2.0 * d) ≈ arrival_time(xs[1], d) && arrival_time(xs[1], zero(d)) == 1.0
    end

    @testset "T1: a frozen sphere over a window is the stationary sphere" begin
        p = prepare(UniformSphere())
        kw = (seed = 7, n_ebins = 60, backend = :cpu)
        a = mc_spectrum(p, 100_000; kw...)
        r = mc_spectrum(p, 100_000; kw..., window = (0.0, 30.0), time_bins = 32, time_ebins = 60)
        @test r isa TimedSpectrumResult && same(a, r.result) && r.shares == [1.0]
        @test r.beyond_series == 0.0 && r.interior == (1.0, 29.0) && r.rule.n_t == 32
        # every recorded photon has an arrival bin: the time rows add up to the stationary bins
        nr = 6
        @test sum(r.total.count[nr * k .+ (1:nr), :] for k in 0:33) == a.count
        @test sum(r.total.sumwE[nr * k .+ (1:nr), :] for k in 0:33) ≈ a.sumwE rtol = 1e-12
        # the light curve is flat inside, at the stationary luminosity
        t, L, s, _ = light_curve(r)
        k = interior_bins(r)
        q, m = flatness(L[k], s[k])
        @test 0.5 < q < 1.6
        Ls = sum(a.sumwE) * BlackLightPlus.Numerics.ME * CL^2
        @test abs(m / Ls - 1) < 4 * sqrt(sum(abs2, s[k] / L[k]) / length(k)^2)
        # the window mean of the time-resolved spectrum is the stationary spectrum
        nu, Lt, st, ct = time_spectrum(r, k[5])
        @test length(nu) == 60 && all(st .>= 0) && sum(ct) > 0
        # scattering: the orders, children and ledger
        ss = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        ks = (scattering = true, table = THOMSON, seed = 0xBEEF, nu_max = 1.0e13, spec_nu_max = 1.0e18,
              n_ebins = 50, backend = :cpu)
        b = mc_spectrum(prepare(ss), 10_000; ks...)
        rb = mc_spectrum(prepare(ss), 10_000; ks..., window = (0.0, 1.0e7), time_bins = 10)
        @test same(b, rb.result) && b.n_children == rb.result.n_children
        @test DK.bits((b.w_children, b.scatter_loss, b.escaped_power)) ==
              DK.bits((rb.result.w_children, rb.result.scatter_loss, rb.result.escaped_power))
        @test sum(rb.orders[2].count) > 0 && sum(rb.total.count) >= sum(rb.orders[1].count)
        # the time-resolved bins do not depend on the chunking
        rc = mc_spectrum(prepare(ss), 10_000; ks..., window = (0.0, 1.0e7), time_bins = 10,
                         backend = ThreadedExecutor(chunk = 7))
        @test all(same.(rc.orders, rb.orders)) && same(rc.result, rb.result)
    end

    @testset "T5: a snapshot or a one-frame series on Kerr is the stationary torus" begin
        snap = DK.amr_deck()
        st = KerrSchildCartesian(DK.TORUS_A)
        pt = prepare(TORUS_CFG; spacetime = st, matter = snap)
        kw = (seed = 0xC0FFEE, C = 32.0, n_ebins = 60, backend = :cpu)
        a = mc_spectrum(pt, 12_000; kw...)
        win = (kw..., window = (0.0, 600.0), time_bins = 24)
        for matter in (snap, FrameSequence(collect(0.0:150.0:600.0), fill(snap.blocks, 5)),
                       SeriesProvider(synthetic_series(snap; times = [0.0, 300.0, 600.0])))
            r = mc_spectrum(prepare(TORUS_CFG; spacetime = st, matter = matter), 12_000; win...)
            @test same(a.total, r.result.total) && all(same.(a.spectra, r.result.spectra))
            @test (a.n_gen, a.n_recorded) == (r.result.n_gen, r.result.n_recorded)
            @test DK.bits(a.escaped_power) == DK.bits(r.result.escaped_power) && r.shares == [1.0]
        end
        # copies of the frame are five nodes, the hats of the frames, with the stationary mean
        cp = FrameSequence(collect(0.0:150.0:600.0), [deepcopy(snap.blocks) for _ in 1:5])
        r = mc_spectrum(prepare(TORUS_CFG; spacetime = st, matter = cp), 12_000; win...)
        @test r.shares == [0.125, 0.25, 0.25, 0.25, 0.125]
        @test 0.0 < r.beyond_series < 1.0
        sa = sqrt(sum(a.total.sumwE2)); sr = sqrt(sum(r.result.total.sumwE2))
        @test abs(sum(r.result.total.sumwE) - sum(a.total.sumwE)) < 4 * hypot(sa, sr)
        t, L, s, _ = light_curve(r; row = 4)
        k = interior_bins(r)
        @test length(k) >= 10 && 0.4 < flatness(L[k], s[k])[1] < 1.8
    end

    @testset "T2: the variable sphere damps the light curve by 3 j₁(x)/x" begin
        R = 1.0; A = 0.5; n = 24
        for x in (1.5, 4.4934, 6.0)
            ω = x / R; P = 2π / ω
            τ = collect(range(0.0, 6P; length = 6n + 1))
            vs = VariableSphere(UniformSphere(R = R), τ, 1 .+ A .* sin.(ω .* τ))
            r = mc_spectrum(prepare(vs), 400_000; seed = 21, backend = :cpu, window = (0.0, 6P),
                            time_bins = (0.0, 6P, 72))
            @test length(r.shares) == 6n + 1 && sum(r.shares) ≈ 1.0
            b, sb, c, sc = sine_fit(r, ω)
            @test abs(b / A - sinc2(π / n) * P̂(x)) < 4 * sb / A
            @test abs(c / A) < 4 * sc / A                          # no phase shift
        end
    end

    @testset "T3: the light echo of a thin Thomson shell" begin
        r_in = 1.0; Δ = 0.04; τr = 1.0e-4
        shell = SphericalShell(n_e = 2τr / (SIGMA_THOMSON * Δ), theta_e = 1.0e-4, r_in = r_in,
                               r_out = r_in + Δ)
        src = UniformSphere(n_e = 1.0e13, theta_e = 10.0, B = 1.0, R = 1.0e-3)
        nb = 21
        r = mc_spectrum(prepare(src), 100_000; seed = 3, backend = :cpu, scattering = true,
                        max_scatt = 1, table = THOMSON, bias = Bias(1 / τr), medium = shell,
                        step_frac = Δ / 8 / (r_in + Δ), window = (0.0, 1.0e-3),
                        time_bins = (-0.05, 2.05, nb), nu_max = 1.0e12)
        @test abs(r.result.n_children / r.result.n_primary - 1) < 0.02     # b τ = 1 event per photon
        @test r.result.w_unspawned < 0.005 * r.result.scatter_loss
        _, L1, s1, _ = light_curve(r; order = 1)
        _, L0, _, _ = light_curve(r; order = 0)
        _, e = arrival_times(r)
        z = Float64[]
        for k in 2:nb
            f = τr * echo_fraction(e[k], e[k + 1]; r_in = r_in, Δ = Δ) * (e[k + 1] - e[k])
            push!(z, (L1[k] / sum(L0) - f) / (s1[k] / sum(L0)))
        end
        @test maximum(abs, z) < 4.5 && 0.4 < sum(abs2, z) / length(z) < 1.8
    end

    @testset "T4: the co-rotating plasma on the orbiting binary" begin
        orbit = CircularOrbit((0.5, 0.5), 8.0)
        Ω = orbit.omega; P = 2π / Ω
        mov = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
        naz = 8; nt = 32
        kw = (C = 32.0, backend = :cpu, window = (0.0, 4P), bands = InclinationBands((90.0,), 90.0),
              sectors = naz, time_bins = (0.0, 4P, nt))
        sector_curves(r) = [light_curve(r; row = sector_row(r.rule.bands, 1, j))[2:3] for j in 1:naz]
        # helical symmetry: sector j + 1 at bin k + 1 is sector j at bin k; the sky is flat
        r = mc_spectrum(prepare(TORUS_CFG; spacetime = mov, matter = Corotating(DK.amr_deck(), Ω)),
                        50_000; seed = 5, kw...)
        @test r.omega == Ω && r.shares == [1.0]
        S = sector_curves(r)
        k = interior_bins(r)
        χ = [(S[j][1][i] - S[mod1(j + 1, naz)][1][i + 1])^2 /
             (S[j][2][i]^2 + S[mod1(j + 1, naz)][2][i + 1]^2) for i in k[1:end-1] for j in 1:naz]
        @test 0.6 < mean(χ) < 1.45
        t, L, s, _ = light_curve(r)
        @test 0.5 < flatness(L[k], s[k])[1] < 1.6
        # a plasma symmetric under R_z(π) on the equal-mass binary: period P/2 at fixed azimuth
        rp = mc_spectrum(prepare(TORUS_CFG; spacetime = mov, matter = Corotating(pi_symmetric_deck(), Ω)),
                         50_000; seed = 6, kw...)
        Sp = sector_curves(rp)
        kp = interior_bins(rp)
        χp = [(Sp[j][1][i] - Sp[j][1][i + 4])^2 / (Sp[j][2][i]^2 + Sp[j][2][i + 4]^2)   # bins of P/8
              for i in kp if i + 4 in kp for j in 1:naz]
        @test 0.6 < mean(χp) < 1.45
        # the helical map needs the symmetry; a frozen snapshot needs a stationary spacetime
        @test_throws ArgumentError mc_spectrum(prepare(TORUS_CFG; spacetime = mov,
                                                       matter = Corotating(DK.amr_deck(), 1.1Ω)), 100; kw...)
        @test_throws ArgumentError mc_spectrum(prepare(TORUS_CFG; spacetime = mov, matter = DK.amr_deck()),
                                               100; kw...)
        stat = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)])
        @test_throws ArgumentError mc_spectrum(prepare(TORUS_CFG; spacetime = stat,
                                                       matter = Corotating(DK.amr_deck(), Ω)), 100; kw...)
    end

    @testset "T6: errors fall as N^-1/2 and match the seed scatter" begin
        p = prepare(UniformSphere())
        lc(N, seed) = light_curve(mc_spectrum(p, N; seed = seed, backend = :cpu, window = (0.0, 20.0),
                                              time_bins = (0.0, 20.0, 40)))
        k = 3:38                                                # inside [1, 19]
        small = [lc(20_000, s) for s in 1:12]
        large = [lc(80_000, s) for s in 101:112]
        ratio = mean(mean(c[3][k]) for c in small) / mean(mean(c[3][k]) for c in large)
        @test abs(ratio - 2.0) < 0.05
        for runs in (small, large)
            Ls = reduce(hcat, [c[2][k] for c in runs]); σ = reduce(hcat, [c[3][k] for c in runs])
            q = mean(var(Ls; dims = 2) ./ mean(σ .^ 2; dims = 2))   # seed variance over the reported one
            @test 0.7 < q < 1.3
        end
    end

    @testset "argument checks" begin
        p = prepare(UniformSphere())
        vs = VariableSphere(UniformSphere(), [0.0, 1.0], [1.0, 2.0])
        @test_throws ArgumentError mc_spectrum(p, 100; window = (1.0, 1.0), backend = :cpu)
        @test_throws ArgumentError mc_spectrum(p, 100; window = (0.0, Inf), backend = :cpu)
        @test_throws ArgumentError mc_spectrum(p, 100; window = (0.0, 1.0), backend = :host_pool)
        @test_throws ArgumentError mc_spectrum(prepare(vs), 100; backend = :cpu)
        @test_throws ArgumentError mc_spectrum(prepare(vs), 100; window = (0.0, 2.0), backend = :cpu)
        snap = DK.amr_deck()
        ps = prepare(TORUS_CFG; spacetime = KerrSchildCartesian(0.9),
                     matter = FrameSequence([0.0, 10.0], [snap.blocks, snap.blocks]))
        @test_throws ArgumentError mc_spectrum(ps, 100; backend = :cpu)
        @test_throws ArgumentError mc_spectrum(ps, 100; window = (-1.0, 5.0), backend = :cpu)
        pc = prepare(TORUS_CFG; spacetime = KerrSchildCartesian(0.9), matter = Corotating(snap, 0.01))
        @test_throws ArgumentError mc_spectrum(pc, 100; backend = :cpu)
        @test_throws ArgumentError mc_spectrum(pc, 100; window = (0.0, 5.0), bands = BlackLightPlus.Execution.PolarBands(3),
                                               backend = :cpu)
    end
end
