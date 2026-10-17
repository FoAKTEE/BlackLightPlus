# End-to-end Monte Carlo of the uniform sphere through generate → track! → ExactBins on
# the threaded executor. Thin sphere (n_e = 1e13, Θₑ = 100, B = 1 G, R = 1 cm): the
# angle-summed ν L_ν matches ν V ∫j_ν dΩ within 10 % per well-populated bin over
# [1e9, 1e14] Hz and the bolometric power within 3 %, for exact and tabulated generation
# (gpumonty sphere test). Scattering sphere (n_e = 1e18, Θₑ = 3, B = 100 G, R = 3e5 cm,
# τ_es ≈ 0.2, needs the gpumonty hot-cross table): the unscattered spectrum is the direct
# emission times a grey survival, photons are up-scattered in distinct bumps, the
# scattered fraction matches the optical depth, the spawned weight equals the scattering
# loss for every bias, and the result agrees with the pinned statistics of 32 runs of
# the slot-keyed reference implementation.
using Test, BLP
using StaticArrays
using BLP.Numerics: SIGMA_THOMSON, MCStreamLayout, primary_rng
using BLP.Spacetime: Minkowski
using BLP.Matter: UniformSphere
using BLP.MonteCarlo: int_jnu_zone, GenerationTables, tabulated_dnmax, table_zone, FlatMarch,
                      Opacities, Bias, VectorSink, generate, track!, SPHERE_MAX_STEPS
using BLP.Execution: ThreadedExecutor, Spectrum, nuLnu_total, total_power, total_weight,
                     mean_energy, nu_centers, scalar
using BLP.Data: build_int_jnu_table, load_hotcross

const DK = Main.Decks
const EXEC = ThreadedExecutor(chunk = 4096)
const HOTCROSS_DAT = DK.HOTCROSS_DAT

analytic_nuLnu(s, nu) = nu * s.dV * int_jnu_zone(s.n_e, s.theta_e, s.B, nu)

# Worst per-bin |MC − analytic|/analytic over [1e9, 1e14] Hz bins with ≥ 50 photons.
function sphere_gate(spec, s)
    nu, L, cnt = nuLnu_total(spec)
    rel = [abs(L[i] - analytic_nuLnu(s, nu[i])) / analytic_nuLnu(s, nu[i])
           for i in eachindex(nu) if 1.0e9 <= nu[i] <= 1.0e14 && cnt[i] >= 50]
    return maximum(rel), length(rel)
end

function analytic_bolometric(s; n = 4000)
    lo = log(1.0e8); h = (log(1.0e16) - lo) / n
    return sum(((i == 0 || i == n) ? 1.0 : (isodd(i) ? 4.0 : 2.0)) * analytic_nuLnu(s, exp(lo + i * h))
               for i in 0:n) * h / 3.0
end

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1]))

# Grey survival S = median(order-0/analytic) over seed-band bins and the worst deviation.
function survival_shape(order0, s)
    nu, L0, cnt = nuLnu_total(order0)
    r = [L0[i] / analytic_nuLnu(s, nu[i]) for i in eachindex(nu)
         if 1.0e9 <= nu[i] <= 1.0e11 && cnt[i] >= 50]
    S = median(r)
    return S, maximum(abs.(r ./ S .- 1.0)), length(r)
end

"Σ w E of order `k` over the bins centred in decade `d`."
band_power(spec, d) = sum(sum(view(spec.sumwE, :, i)) for (i, nu) in enumerate(nu_centers(spec))
                          if floor(Int, log10(nu)) == d; init = 0.0)

# Mean and seed-to-seed standard deviation of 32 runs of the slot-keyed implementation
# (seeds 0xBEEF + 1:32): Σ w E per decade of each order, and the child ledger.
const OLD_BANDS = [
    (0, 8, 1.265946e+30, 8.3347e+27), (0, 9, 1.446833e+31, 8.3712e+28),
    (0, 10, 7.325630e+31, 3.6373e+29), (0, 11, 4.814777e+31, 2.8290e+29),
    (0, 12, 1.430858e+30, 1.1927e+28),
    (1, 8, 1.451611e+28, 8.3568e+26), (1, 9, 6.494493e+29, 1.9957e+28),
    (1, 10, 1.583397e+31, 2.5934e+29), (1, 11, 1.466211e+32, 2.5097e+30),
    (1, 12, 7.685633e+32, 1.3205e+31), (1, 13, 1.240176e+33, 3.6397e+31),
    (1, 14, 4.788721e+32, 3.0484e+31), (1, 15, 1.826814e+31, 4.9709e+30),
    (2, 9, 9.625652e+27, 1.9405e+27), (2, 10, 5.348658e+29, 5.0572e+28),
    (2, 11, 1.251500e+31, 6.9997e+29), (2, 12, 2.167657e+32, 1.1423e+31),
    (2, 13, 1.805904e+33, 8.2653e+31), (2, 14, 9.761087e+33, 5.5153e+32),
    (2, 15, 2.276690e+34, 1.6924e+33), (2, 16, 2.236291e+34, 3.9369e+33),
    (3, 11, 3.738538e+29, 9.7192e+28), (3, 12, 1.197925e+31, 1.7427e+30),
    (3, 13, 2.167107e+32, 2.3417e+31), (3, 14, 2.962615e+33, 2.7431e+32),
    (3, 15, 2.311128e+34, 1.9049e+33), (3, 16, 1.256385e+35, 1.0138e+34)]
const OLD_LEDGER = (n_children = (3.649075e+04, 2.4133e+02), w_children = (2.057956e+41, 1.8989e+39),
                    scatter_loss = (2.062531e+41, 5.7502e+38))

# Weight spawned by primaries over their scattering loss, primaries only (no children).
function first_order_ledger(tau_es, table; N = 20_000)
    R = 3.0e5
    s = UniformSphere(n_e = tau_es / (SIGMA_THOMSON * R), theta_e = 3.0, B = 100.0, R = R)
    tabs, _ = DK.sphere_tables(s, 1.0e5; nu_min = 1.0e8, nu_max = 1.0e13)
    opac = Opacities(; table = table, bias = Bias(s), max_scatt = 3)
    sink = VectorSink(MCStreamLayout(N, SPHERE_MAX_STEPS, 3))
    for i in 1:N
        sp, rng = generate(s, tabs, primary_rng(0xBEEF, i))
        track!(FlatMarch(s), Minkowski(), s, opac, sp, rng, sink; absorb = Val(false),
               scatter = Val(true))
    end
    return sum(l[3] for l in sink.ledger) / sum(l[2] for l in sink.ledger)
end

@testset "Monte Carlo sphere" begin
    @testset "thin sphere: spectrum and power, exact and tabulated generation" begin
        s = UniformSphere()
        tabs, N = DK.sphere_tables(s, 2_000_000)
        spec = Spectrum(DK.sphere_thin_bins(EXEC, s, tabs, N, 0xC0FFEE))
        worst, ntested = sphere_gate(spec, s)
        @test ntested >= 50 && worst < 0.10
        @test isapprox(total_power(spec) / analytic_bolometric(s), 1.0; atol = 0.03)
        gt = build_int_jnu_table(table_zone(s), tabs.weights)
        tt = GenerationTables(tabs.weights, tabulated_dnmax(gt, tabs.weights), gt)
        spec_t = Spectrum(DK.sphere_thin_bins(EXEC, s, tt, N, 0xC0FFEE))
        worst_t, ntested_t = sphere_gate(spec_t, s)
        @test ntested_t >= 50 && worst_t < 0.10
        @test abs(total_power(spec_t) - total_power(spec)) / total_power(spec) < 1.0e-4
        ca = vec(sum(spec.count; dims = 1)); cb = vec(sum(spec_t.count; dims = 1))
        @test maximum(abs.(cb .- ca) ./ sqrt.(max.(ca, 1))) < 5.0
    end

    if isfile(HOTCROSS_DAT)
        table = load_hotcross(HOTCROSS_DAT)
        s = UniformSphere(n_e = 1.0e18, theta_e = 3.0, B = 100.0, R = 3.0e5)
        tabs, N = DK.sphere_tables(s, 200_000; nu_min = 1.0e8, nu_max = 1.0e13)
        orders, ledger, layers = DK.sphere_scatter_run(EXEC, s, table, tabs, N, 0xBEEF;
                                                       capacity = N)
        spec = Spectrum.(orders)

        @testset "scattering sphere: direct emission is a grey survival of the analytic" begin
            S, worst, nbins = survival_shape(spec[1], s)
            @test nbins >= 20 && worst < 0.10
            @test 0.82 <= S <= 0.93
        end

        @testset "scattering sphere: up-scattering in distinct bumps" begin
            E0 = mean_energy(spec[1])
            @test 30.0 < mean_energy(spec[2]) / E0 < 400.0
            W = total_weight.(spec); E = mean_energy.(spec)
            @test all(W .> 0.0)
            @test all(W[n] > W[n + 1] for n in 1:3)
            @test all(E[n + 1] / E[n] > 4.0 for n in 1:3)
            @test 0.0 < total_power(spec[4]) < Inf
        end

        @testset "scattering sphere: scattered fraction and unbiasedness" begin
            @test 0.11 <= scalar(ledger, 2) / scalar(ledger, 4) <= 0.16
            @test 0.95 <= scalar(ledger, 1) / scalar(ledger, 3) <= 1.05
            for tau in (1.0, 3.0, 10.0)
                @test isapprox(first_order_ledger(tau, table), 1.0; atol = 0.04)
            end
        end

        @testset "scattering sphere: agrees with the slot-keyed implementation" begin
            z = [(band_power(spec[k + 1], d) - m) / sd for (k, d, m, sd) in OLD_BANDS]
            @test maximum(abs, z) <= 4.5
            @test sum(abs2, z) <= length(z) + 5.5 * sqrt(2 * length(z))
            got = (n_children = sum(length(first(l)) for l in layers),
                   w_children = scalar(ledger, 1), scatter_loss = scalar(ledger, 3))
            for (f, (m, sd)) in pairs(OLD_LEDGER)
                @test abs(getfield(got, f) - m) <= 4.5 * sd
            end
        end
    else
        @info "gpumonty hotcross.dat not found: scattering-sphere checks skipped" path = HOTCROSS_DAT
    end
end
