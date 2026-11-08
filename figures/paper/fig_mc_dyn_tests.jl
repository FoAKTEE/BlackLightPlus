# fig_mc_dyn_tests.jl — the exact tests of the Monte Carlo over an emission window, at
# converged photon counts. CPU threads.
#
# Legs (BLP_MCD_LEGS, comma-separated; default all, in this order):
#   sphere  T2: a VariableSphere of radius 1 cm (n_e = 1e13, Θe = 100, B = 1 G, thin), density
#           factor 1 + 0.5 sin ωt through 32 nodes per period, window of 8 periods, 16 arrival
#           bins per period, BLP_MCD_NSPH primaries at each ωR of OMEGAS: the amplitude and the
#           cosine term of the light curve's fundamental over the whole periods of the
#           interior (weighted least squares on the exact bin averages of sin and cos), and
#           for three ωR the light curve with its exact bin averages (the interpolated emission,
#           zero outside the window, convolved with the slab delay distribution
#           P(τ) = 3(R² − τ²)/4R³, by a midpoint rule of 400 × 400 points per bin);
#   echo    T3: a flash (a ball of radius 1e-3 cm, n_e = 1e13, Θe = 10, B = 1 G, emitting over
#           [0, 1e-3] cm) inside a SphericalShell of radii 1 and 1.04 cm, Θe = 1e-4, radial Thomson
#           depth 1e-4, Thomson cross section, one scattering, bias 1e4 (one event per photon),
#           step 1/32 of the shell width, BLP_MCD_NECHO primaries, 0.02 cm arrival bins: the
#           singly scattered light over the emitted, with the first-order closed form averaged
#           over the shell's sin² depth profile, the flash's duration and each bin, and the
#           thin-shell Ψ at r = 1.02;
#   frozen  T1 and T5: the uniform sphere over [0, 30] cm (BLP_MCD_NSPH) and the Kerr torus
#           (AthenaK snapshot 00999 on a = 0.9, the plasma of fig_pol_hires.jl, emission zones of
#           stride BLP_MCD_STRIDE within r < 40, thin, BLP_MCD_NTOR) over [0, 1000] M, each against
#           the stationary run (bitwise flags of every bin and of the ledger), the torus also as a
#           FrameSequence of one frame at five times (bitwise) and of five copies of it (five
#           nodes, statistical); the light curves of the 17, 50 and 90 degree bands;
#   conv    T6: the variable sphere at ωR = 3 for N in BLP_MCD_NCONV, BLP_MCD_NSEED seeds each:
#           the per-bin errors of the interior and the scatter of the seeds;
#   pisym   T4 with and without the symmetry under the rotation by π: the torus deck of the
#           tests with azimuthal velocity and vertical field (symmetric under R_z(π)), and the same
#           with the density times 1 + 0.6 x/ϖ (lopsided), co-rotating on the orbiting equal-mass
#           binary of separation 8, 16 sectors of the whole sky, arrival bins of P/16,
#           BLP_MCD_NPI primaries, C = 32: the sector light curves and the helical, half-period
#           and unshifted statistics.
#
# Writes BLP_MCD_OUT (figures/paper/out/mc_dyn/tests): <leg>_*.npy and <leg>.json.
# Env: BLP_MCD_NSPH (2e6), BLP_MCD_NECHO (2e6), BLP_MCD_NTOR (2e6), BLP_MCD_STRIDE (4),
#      BLP_MCD_NCONV (1e4,3e4,1e5,3e5,1e6), BLP_MCD_NSEED (8), BLP_MCD_NPI (1e6), BLP_TEST_GRMHD.
# Run, from the repository root:
#   julia -t 48 --project=. figures/paper/fig_mc_dyn_tests.jl
using Dates, LinearAlgebra, Printf, Statistics, StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: GNEWT, CL, MSUN, ME, SIGMA_THOMSON
using BlackLightPlus.Matter: UniformSphere, VariableSphere, SphericalShell
using BlackLightPlus.MonteCarlo: HotcrossTable, NW, NT, Bias
using BlackLightPlus.Spacetime: KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit
using BlackLightPlus.Data: AthenaKSnapshot, FrameSequence, Corotating, write_npy
using BlackLightPlus.Execution: InclinationBands, Spectrum, sector_row
using BlackLightPlus.Products: light_curve, arrival_times, TimedSpectrumResult

const ROOT = dirname(dirname(@__DIR__))
const OUT = get(ENV, "BLP_MCD_OUT", joinpath(@__DIR__, "out", "mc_dyn", "tests"))
const LEGS = split(get(ENV, "BLP_MCD_LEGS", "sphere,echo,frozen,conv,pisym"), ",")
envnum(k, d) = parse(Float64, get(ENV, k, string(d)))
const NSPH = round(Int, envnum("BLP_MCD_NSPH", 2e6))
const NECHO = round(Int, envnum("BLP_MCD_NECHO", 2e6))
const NTOR = round(Int, envnum("BLP_MCD_NTOR", 2e6))
const STRIDE = round(Int, envnum("BLP_MCD_STRIDE", 4))
const NCONV = [round(Int, parse(Float64, s)) for s in split(get(ENV, "BLP_MCD_NCONV", "1e4,3e4,1e5,3e5,1e6"), ",")]
const NSEED = round(Int, envnum("BLP_MCD_NSEED", 8))
const NPI = round(Int, envnum("BLP_MCD_NPI", 1e6))
const TORUS = get(ENV, "BLP_TEST_GRMHD",
                  "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
const THOMSON = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
const OMEGAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.4934, 5.0, 6.0, 7.0, 7.7253, 8.5, 9.5, 10.9041, 12.0]
const LC_OMEGAS = [1.5, 4.4934, 6.0]

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))
json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::Tuple) = json(collect(x))
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"
record(leg, d) = open(io -> println(io, json(merge(d, Dict("code" => GIT_SHA, "threads" => Threads.nthreads(),
                                                           "date" => string(now()))))),
                      joinpath(OUT, leg * ".json"), "w")
npy(name, a::AbstractMatrix) = write_npy(joinpath(OUT, name * ".npy"), collect(Float64, a))
npy(name, a::AbstractVector) = npy(name, reshape(collect(Float64, a), :, 1))
bits(x) = reinterpret(UInt64, x)
same(a::Spectrum, b::Spectrum) = a.count == b.count && bits.(a.sumw) == bits.(b.sumw) &&
                                 bits.(a.sumwE) == bits.(b.sumwE) && bits.(a.sumwE2) == bits.(b.sumwE2)

P̂(x) = 3 * (sin(x) - x * cos(x)) / x^3
sinc2(x) = (sin(x) / x)^2

# The interior arrival bins (both edges inside), on whole periods P when given.
function interior_bins(res; P = nothing)
    _, e = arrival_times(res)
    lo, hi = res.interior
    P === nothing || (lo = ceil(lo / P - 1e-9) * P; hi = floor(hi / P + 1e-9) * P)
    return findall(k -> e[k] >= lo - 1e-9 && e[k + 1] <= hi + 1e-9, 1:length(e) - 1)
end

# (b/a, σ_b/a, c/a, σ_c/a) of a + b⟨sin ωt⟩ + c⟨cos ωt⟩ over the whole periods of the interior.
function sine_fit(res, ω)
    _, L, s, _ = light_curve(res)
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

# The exact bin averages of (f ∗ P)(t) for the emission f, piecewise linear through (τ, fv)
# inside the window [τ[1], τ[end]] and zero outside it.
function sphere_exact(edges, τ, fv, R; n = 400)
    f(t) = (t < τ[1] || t > τ[end] ? 0.0 :
            (k = clamp(searchsortedlast(τ, t), 1, length(τ) - 1); w = (t - τ[k]) / (τ[k + 1] - τ[k]);
             (1 - w) * fv[k] + w * fv[k + 1]))
    out = zeros(length(edges) - 1)
    Threads.@threads for b in 1:length(edges) - 1
        s = 0.0
        for i in 1:n, j in 1:n
            t = edges[b] + (i - 0.5) / n * (edges[b + 1] - edges[b])
            d = -R + (j - 0.5) / n * 2R
            s += 3 * (R^2 - d^2) / (4R^3) * f(t - d)
        end
        out[b] = s * 2R / n^2
    end
    return out
end

function leg_sphere()
    R = 1.0; A = 0.5; n = 32; np = 8; nb = 16
    rows = Dict{String,Any}[]
    for x in OMEGAS
        ω = x / R; P = 2π / ω
        τ = collect(range(0.0, np * P; length = np * n + 1))
        fv = 1 .+ A .* sin.(ω .* τ)
        vs = VariableSphere(UniformSphere(R = R), τ, fv)
        t0 = time()
        r = mc_spectrum(prepare(vs), NSPH; seed = 1000 + findfirst(==(x), OMEGAS), backend = :cpu, window = (0.0, np * P),
                        time_bins = (0.0, np * P, np * nb), time_ebins = 10)
        b, sb, c, sc = sine_fit(r, ω)
        push!(rows, Dict{String,Any}("x" => x, "amp" => b / A, "amp_sigma" => sb / A, "cos" => c / A,
                                     "cos_sigma" => sc / A, "expect" => sinc2(π / n) * P̂(x),
                                     "continuous" => P̂(x), "wall_s" => time() - t0,
                                     "n_primary" => sum(r.result.count)))
        logf("  ωR = %.4f: amplitude/A %.5f ± %.5f (expect %.5f), cos %.5f ± %.5f  [%.1f s]\n", x, b / A, sb / A,
             sinc2(π / n) * P̂(x), c / A, sc / A, time() - t0)
        if x in LC_OMEGAS
            t, L, s, cnt = light_curve(r)
            _, e = arrival_times(r)
            Lm = sum(r.result.sumwE) * ME * CL^2                 # the window mean
            tag = replace(@sprintf("%.4f", x), "." => "p")
            npy("sphere_lc_$(tag)", hcat(t, L ./ Lm, s ./ Lm, sphere_exact(e, τ, fv, R)))
            npy("sphere_lc_$(tag)_edges", e)
        end
    end
    record("sphere", Dict{String,Any}("R_cm" => R, "A" => A, "nodes_per_period" => n, "periods" => np,
                                      "bins_per_period" => nb, "N" => NSPH, "rows" => rows,
                                      "lc_omegas" => LC_OMEGAS))
end

function leg_echo()
    r_in = 1.0; Δ = 0.04; τr = 1.0e-4; δt = 1.0e-3
    shell = SphericalShell(n_e = 2τr / (SIGMA_THOMSON * Δ), theta_e = 1.0e-4, r_in = r_in, r_out = r_in + Δ)
    src = UniformSphere(n_e = 1.0e13, theta_e = 10.0, B = 1.0, R = 1.0e-3)
    lo, hi, nb = -0.05, 2.11, 108
    t0 = time()
    r = mc_spectrum(prepare(src), NECHO; seed = 2002, backend = :cpu, scattering = true, max_scatt = 1,
                    table = THOMSON, bias = Bias(1 / τr), medium = shell, step_frac = Δ / 32 / (r_in + Δ),
                    window = (0.0, δt), time_bins = (lo, hi, nb), nu_max = 1.0e12, time_ebins = 4)
    wall = time() - t0
    t, L1, s1, c1 = light_curve(r; order = 1)
    _, L0, _, _ = light_curve(r; order = 0)
    _, e = arrival_times(r)
    S0 = sum(L0)
    Ψ(τ, rr) = 0.0 <= τ <= 2rr ? 3 / (8rr) * (1 + (1 - τ / rr)^2) : 0.0
    # the bin average over [a, b] of the echo of a flash uniform over [0, δt]
    function avg(a, b; nr = 400, nt = 200, ns = 16)
        s = 0.0
        for i in 1:nr
            rr = r_in + (i - 0.5) / nr * Δ
            p = 2 * sin(π * (i - 0.5) / nr)^2
            for j in 1:nt, k in 1:ns
                s += p * Ψ(a + (j - 0.5) / nt * (b - a) - (k - 0.5) / ns * δt, rr)
            end
        end
        return s / (nr * nt * ns)
    end
    ex = [avg(e[k], e[k + 1]) for k in 1:nb]
    dt = e[2] - e[1]
    npy("echo", hcat(t, L1 ./ S0 ./ (τr * dt), s1 ./ S0 ./ (τr * dt), ex, Ψ.(t, r_in + Δ / 2), c1))
    z = [(L1[k] / S0 - τr * ex[k] * dt) / (s1[k] / S0) for k in 1:nb if s1[k] > 0]
    logf("  echo: %d primaries, %d children, unspawned %.2e of the loss, chi2/n %.3f over %d bins [%.1f s]\n",
         r.result.n_primary, r.result.n_children, r.result.w_unspawned / r.result.scatter_loss,
         sum(abs2, z) / length(z), length(z), wall)
    record("echo", Dict{String,Any}("r_in" => r_in, "width" => Δ, "tau_r" => τr, "flash_cm" => δt,
                                    "N" => NECHO, "n_children" => r.result.n_children,
                                    "unspawned_over_loss" => r.result.w_unspawned / r.result.scatter_loss,
                                    "chi2_per_bin" => sum(abs2, z) / length(z), "nbins" => length(z),
                                    "max_abs_z" => maximum(abs, z), "wall_s" => wall))
end

function torus_plan(matter)
    MB = 6.2e9 * MSUN
    L = GNEWT * MB / CL^2
    cfg = RunConfig(m_bh_cgs = MB, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = 2.3e11,
                    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, gamma = 13.0 / 9.0, use_p = true,
                                          sigma_cut = 1.0, theta_e_min = -1.0), C = 80.0)
    return prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = matter)
end

function leg_frozen()
    rec = Dict{String,Any}()
    # the sphere
    p = prepare(UniformSphere())
    a = mc_spectrum(p, NSPH; seed = 3003, backend = :cpu)
    r = mc_spectrum(p, NSPH; seed = 3003, backend = :cpu, window = (0.0, 30.0), time_bins = (-2.0, 32.0, 68))
    rec["sphere_bitwise"] = same(a, r.result)
    t, L, s, _ = light_curve(r)
    Ls = sum(a.sumwE) * ME * CL^2
    npy("frozen_sphere", hcat(t, L ./ Ls, s ./ Ls))
    logf("  frozen sphere: bitwise %s\n", rec["sphere_bitwise"])
    # the torus
    t0 = time()
    snap = AthenaKSnapshot(TORUS; gamma = 13.0 / 9.0)
    rec["torus_load_s"] = time() - t0
    kw = (seed = 3004, stride = STRIDE, r_emit_max = 40.0, n_ebins = 120, spec_nu_max = 1.0e16, backend = :cpu)
    win = (window = (0.0, 1000.0), time_bins = (-100.0, 1100.0, 60), time_ebins = 12)
    t0 = time()
    st = mc_spectrum(torus_plan(snap), NTOR; kw...)
    rec["torus_stationary_s"] = time() - t0
    runs = Dict("snapshot" => snap, "frames" => FrameSequence(collect(0.0:250.0:1000.0), fill(snap.blocks, 5)),
                "copies" => FrameSequence(collect(0.0:250.0:1000.0), [deepcopy(snap.blocks) for _ in 1:5]))
    for (name, m) in sort(collect(runs); by = first)
        t0 = time()
        r = mc_spectrum(torus_plan(m), NTOR; kw..., win...)
        rec["torus_$(name)_s"] = time() - t0
        rec["torus_$(name)_bitwise"] = same(st.total, r.result.total) && all(same.(st.spectra, r.result.spectra)) &&
                                       bits(st.escaped_power) == bits(r.result.escaped_power) &&
                                       bits(st.absorbed_power) == bits(r.result.absorbed_power)
        rec["torus_$(name)_shares"] = r.shares
        rec["torus_$(name)_escaped_ratio"] = r.result.escaped_power / st.escaped_power
        rec["torus_$(name)_escaped_sigma"] = sqrt(sum(r.result.total.sumwE2) + sum(st.total.sumwE2)) /
                                             sum(st.total.sumwE)
        cols = Any[]
        for k in 1:3
            tt, L, s, _ = light_curve(r; row = k)
            Lk = sum(st.spectra[k].sumwE) * ME * CL^2 * (4π / st.dOmega[k])
            k == 1 && push!(cols, tt)
            push!(cols, L ./ Lk, s ./ Lk)
        end
        npy("frozen_torus_$(name)", reduce(hcat, cols))
        logf("  torus %s: bitwise %s, shares %s, escaped ratio %.5f ± %.5f [%.1f s]\n", name,
             rec["torus_$(name)_bitwise"], string(r.shares), rec["torus_$(name)_escaped_ratio"],
             rec["torus_$(name)_escaped_sigma"], rec["torus_$(name)_s"])
        rec["interior"] = r.interior
    end
    rec["N_torus"] = NTOR; rec["N_sphere"] = NSPH; rec["stride"] = STRIDE; rec["torus"] = TORUS
    rec["n_zones"] = st.n_zones; rec["n_gen"] = st.n_gen
    record("frozen", rec)
end

function leg_conv()
    ω = 3.0; P = 2π / ω; A = 0.5; n = 32; np = 8; nb = 16
    τ = collect(range(0.0, np * P; length = np * n + 1))
    vs = VariableSphere(UniformSphere(R = 1.0), τ, 1 .+ A .* sin.(ω .* τ))
    out = zeros(length(NCONV), 4)
    for (i, N) in enumerate(NCONV)
        Ls = Vector{Float64}[]; σs = Vector{Float64}[]
        for s in 1:NSEED
            r = mc_spectrum(prepare(vs), N; seed = 4000 + 100i + s, backend = :cpu, window = (0.0, np * P),
                            time_bins = (0.0, np * P, np * nb), time_ebins = 10)
            k = interior_bins(r)
            _, L, σ, _ = light_curve(r)
            push!(Ls, L[k] ./ mean(L[k])); push!(σs, σ[k] ./ mean(L[k]))
        end
        Lm = reduce(hcat, Ls); σm = reduce(hcat, σs)
        out[i, :] = [N, mean(σm), mean(std(Lm; dims = 2)), mean(var(Lm; dims = 2) ./ mean(σm .^ 2; dims = 2))]
        logf("  N = %d: mean relative error %.4e, seed scatter %.4e, variance ratio %.3f\n", N, out[i, 2:4]...)
    end
    npy("conv", out)
    record("conv", Dict{String,Any}("omegaR" => ω, "N" => NCONV, "seeds" => NSEED,
                                    "columns" => ["N", "mean_rel_sigma", "mean_seed_std", "var_ratio"]))
end

# The torus deck of the tests on its mesh with azimuthal velocity and vertical field, symmetric
# under R_z(π), or with the density made lopsided by 1 + 0.6 x/ϖ, ϖ = √(x² + y²).
function deck(; lopsided = false, nc = 6, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, 8, nc, nc, nc, nmb)
    lo = zeros(3, nmb); hi = zeros(3, nmb); xl = zeros(3, nmb); dxs = fill(dx, 3, nmb); ll = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        lo[:, m] .= l; hi[:, m] .= l .+ w; xl[:, m] .= l .+ dx / 2; ll[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            x = xl[1, m] + (i - 1) * dx; y = xl[2, m] + (j - 1) * dx; z = xl[3, m] + (k - 1) * dx
            r = sqrt(x^2 + y^2 + z^2)
            (r < 2.2 || r > 11.0) && continue
            ρ = exp(-((r - 5.5) / 2.5)^2)
            lopsided && (ρ *= 1 + 0.6x / hypot(x, y))
            data[:, i, j, k, m] .= Float32.((ρ, -0.2y / r, 0.2x / r, 0.0, 0.08ρ, 0.0, 0.0, 0.3sqrt(ρ)))
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), ll)
end

function leg_pisym()
    MB = 6.2e9 * MSUN
    L = GNEWT * MB / CL^2
    cfg = RunConfig(m_bh_cgs = MB, mass_scale_cgs = 1.0e-21 * L^3, frequency_hz = 2.3e11)
    orbit = CircularOrbit((0.5, 0.5), 8.0)
    Ω = orbit.omega; P = 2π / Ω
    mov = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
    naz = 16; per = 4
    rec = Dict{String,Any}("omega" => Ω, "period" => P, "sectors" => naz, "periods" => per, "N" => NPI)
    for (name, lop) in (("lopsided", true), ("pisym", false))
        t0 = time()
        r = mc_spectrum(prepare(cfg; spacetime = mov, matter = Corotating(deck(lopsided = lop), Ω)), NPI;
                        seed = 5005, C = 32.0, backend = :cpu, window = (0.0, per * P),
                        bands = InclinationBands((90.0,), 90.0), sectors = naz, time_bins = (0.0, per * P, per * naz))
        t, _ = arrival_times(r)
        S = zeros(length(t), naz); V = zeros(length(t), naz)
        for j in 1:naz
            _, Lj, sj, _ = light_curve(r; row = sector_row(r.rule.bands, 1, j))
            S[:, j] = Lj; V[:, j] = sj
        end
        _, Lb, sb, _ = light_curve(r)
        k = interior_bins(r)
        χh = [(S[i, j] - S[i + 1, mod1(j + 1, naz)])^2 / (V[i, j]^2 + V[i + 1, mod1(j + 1, naz)]^2)
              for i in k[1:end-1] for j in 1:naz]
        χ2 = [(S[i, j] - S[i + naz ÷ 2, j])^2 / (V[i, j]^2 + V[i + naz ÷ 2, j]^2) for i in k if i + naz ÷ 2 in k
              for j in 1:naz]
        χ0 = [(S[i, j] - S[i + 1, j])^2 / (V[i, j]^2 + V[i + 1, j]^2) for i in k[1:end-1] for j in 1:naz]
        rec["$(name)_helical_chi2"] = mean(χh); rec["$(name)_half_period_chi2"] = mean(χ2)
        rec["$(name)_noshift_chi2"] = mean(χ0); rec["$(name)_n"] = length(χh)
        rec["$(name)_wall_s"] = time() - t0
        rec["interior"] = r.interior
        npy("pisym_$(name)_S", S); npy("pisym_$(name)_V", V); npy("pisym_$(name)_band", hcat(t, Lb, sb))
        logf("  %s: helical χ²/n %.3f (n %d), half-period χ²/n %.3f, unshifted χ²/n %.3f [%.1f s]\n", name,
             mean(χh), length(χh), mean(χ2), mean(χ0), time() - t0)
    end
    record("pisym", rec)
end

logf("fig_mc_dyn_tests: legs %s on %d threads, code %s\n", join(LEGS, ","), Threads.nthreads(), GIT_SHA)
for leg in LEGS
    logf("leg %s\n", leg)
    t0 = time()
    getfield(Main, Symbol("leg_" * leg))()
    logf("leg %s done in %.1f s\n", leg, time() - t0)
end
