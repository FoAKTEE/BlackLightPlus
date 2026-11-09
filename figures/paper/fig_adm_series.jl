# fig_adm_series.jl — convergence of ADM slices in time (`ADMSeries`) to the analytic
# time-dependent metric they sample, for rays and for Monte Carlo superphotons.
#
# Spacetimes (G = c = M = 1):
#   binary: two holes of mass 1/2, spins (0.3, -0.2) along z, on the Keplerian circular orbit
#           of separation 6 (ω = 6^{-3/2}), boosted terms (`BoostedSuperposedKerrSchild`);
#           helical Killing vector ∂_t + ω ∂_φ, invariant J = p_t + ω (x p_y - y p_x).
#   hole:   one hole of mass 1/2, spin 0.3 along z, in uniform motion v = (0.3, 0.1, 0) from
#           (-2, -1, 0) at t = 0; Killing vector ∂_t + v·∇, invariant E' = p_t + v·p, linear
#           in the state, so RK4 keeps it to round-off on the analytic metric.
# Each spacetime is sampled by `adm_grid_from_metric` at the slice times on a lattice of
# spacing Δx (nodes inside 0.5 r_h of a hole take the collapsed-lapse fill) and stacked into
# an `ADMSeries`; Δt is the slice spacing.
# Legs (BLP_ADMSERIES_LEGS, comma-separated; default all):
#   points  binary metric at a lattice node over t ∈ [3.1, 20.3] (2000 times) for Δt = 2 … 0.125
#           (value and ∂_t g^{μν} errors, max over the times), and at a slice time over 400
#           points within 1 of (0.3, -0.2, 3.2) for Δx = 0.5 … 0.0625 (value and ∂_i g^{μν});
#           the stencil is exact at nodes and slice times, so each scan isolates one order.
#   rays    the hole: 24 rays from (6, b cos θ, b sin θ), b = 3, 4.5, 6, along -x, RK4 at the
#           fixed affine step 0.02 for 600 steps on the analytic metric and on the series of
#           (Δx, Δt) = (1, 2), (0.5, 1), (0.25, 0.5), (0.125, 0.25) (lattice [-8, 8]³, slices
#           -4Δt … 24 + 4Δt): end-state errors |Δx|, |Δp| and the largest |ΔE'| along the ray,
#           for the rays that stay outside 1.5 r_h; and the time-only self-convergence at
#           Δx = 0.25 against Δt = 0.0625.
#   helical the binary: 24 rays from (7, b cos θ, b sin θ), b = 3.5, 5, 6.5, along -x, the same
#           march on the lattice [-9, 9]³ with slices -4Δt … 40 + 4Δt: the largest |ΔJ| along
#           the ray on the analytic metric (the integrator's drift) and on the series.
#   mc      paired superphotons on the binary: the axisymmetric torus plasma (Decks layout,
#           12 cells per block side, half-width 12), a `FrameStack` of its frames at 0, 2, …,
#           12 emitting over the window [2, 10], zones within r = 5; each primary drawn once
#           and generated at its event in each metric from the same zone and stream
#           (`Products.EventSource`), transported with absorption in the stack to R_em = 7
#           (`MonteCarlo.track!`, C = 64) through the analytic binary under the series'
#           capture and step rule (lapse above 0.3) and through the series of
#           (Δx, Δt) = (0.5, 1), (0.25, 0.5), (0.125, 0.25) on
#           [-8, 8]³, slices -2 … 56. Per photon escaping in both with its march ended before
#           t = 54: relative errors of the emitted and escaping energy and the weight, the
#           arrival-time and direction errors; the fates that differ; the all-sky νL_ν of each
#           run on 40 bins over the escaped energies, and its L1 difference.
#   e2e     `mc_spectrum` over the window [2, 18] of a `FrameStack` of frames 0, 2, …, 20 on
#           the series of (0.25, 0.5) and on the analytic binary under the lapse rule (neither
#           declares a symmetry, so each photon is generated at its event), 40 000 primaries:
#           the band powers and their z-scores.
# Writes to figures/paper/out/adm_series/ (BLP_ADMSERIES_OUT):
#   points_time.npy  rows [Δt, value err, ∂_t err]; points_space.npy rows [Δx, value err, ∂_x err]
#   rays.npy         rows [Δx, Δt, max|Δx|, max|Δp|, max|ΔE'|, median|Δx|, median|ΔE'|]
#   rays_time.npy    rows [Δt, max|Δx|, max|Δp|, max|ΔE'|] (Δx = 0.25, against Δt = 0.0625)
#   helical.npy      rows [Δx, Δt, max|ΔJ|, median|ΔJ|]; row Δx = Δt = 0: the analytic metric
#   mc.npy           rows [Δx, Δt, flips, paired, then median and 99th percentile of
#                    |ΔE_emit|/E, |ΔE|/E, |Δw|/w, |Δt_arr|, |Δn̂|, then spectrum L1]
#   mc_spectra.npy   rows [E bin centre (mₑc²), νL_ν analytic, νL_ν per resolution]
#   record_<legs>.json  setups, the tables above as numbers, git SHA, threads, wall time
# Run from the repository root: julia -t 40 --project=. figures/paper/fig_adm_series.jl
# (the finest series hold up to 40 GB).
using LinearAlgebra
using Printf
using Statistics
using StaticArrays
using BlackLightPlus
using BlackLightPlus.Spacetime: Spacetime, BoostedSuperposedKerrSchild, CircularOrbit, ADMSeries,
    adm_grid_from_metric, metric, inverse_metric, metric_and_inverse, inverse_metric_gradient, rk4_step,
    null_covariant, ks_radius, horizon_radius
import BlackLightPlus.Spacetime: proxy_distance, horizon_floor, lane
using BlackLightPlus.Plans: RunConfig, prepare
using BlackLightPlus.Data: AthenaKSnapshot, FrameStack, emission_sites, zone_tables
using BlackLightPlus.Matter: ZoneSite
using BlackLightPlus.MonteCarlo: track!, CurvedMarch, Opacities, NullSink, EMISSIVITY_TABLE, escape_direction
using BlackLightPlus.Products: mc_spectrum, ZoneSource, EventSource, emit, clip_profile, profile_area,
    arrival_time
using BlackLightPlus.Execution: InclinationBands
using BlackLightPlus.Numerics: GNEWT, CL, ME, primary_rng

const OUT = get(ENV, "BLP_ADMSERIES_OUT", joinpath(@__DIR__, "out", "adm_series"))
const LEGS = split(get(ENV, "BLP_ADMSERIES_LEGS", "points,rays,helical,mc,e2e"), ",")
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# ---- spacetimes
const ORBIT = CircularOrbit((0.5, 0.5), 6.0)
const BIN = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.3), (0.0, 0.0, -0.2)], ORBIT)
const V = SVector(0.3, 0.1, 0.0)
const S0 = SVector(-2.0, -1.0, 0.0)
const HOLE = BoostedSuperposedKerrSchild([0.5], [(0.0, 0.0, 0.3)], t -> ((S0 + V * t, V),))
inner(st, N) = x -> any(A -> ks_radius(st, x, A) < 0.5 * horizon_radius(st, A), 1:N)

"The ADM series of `st` (N holes) on [-L, L]³ at spacing dx, slices `ts`, capture lapse `amin`."
function series(st, N, dx, ts; L, amin = 0.1)
    ax = -L:dx:L
    n = length(ax)
    data = Array{Float64,5}(undef, 10, n, n, n, length(ts))
    exc = inner(st, N)
    Threads.@threads for k in eachindex(ts)
        g = adm_grid_from_metric(x -> metric(st, x), ax, ax, ax; alpha_min = amin, excise = exc, t = ts[k])
        data[:, :, :, :, k] .= g.data
    end
    return ADMSeries(ax, ax, ax, ts, data; alpha_min = amin)
end

# The analytic metric under the series' capture and step rule: its lapse above `alpha_min`.
struct LapseControlled{S} <: Spacetime{Float64}
    st::S
    alpha_min::Float64
end
BlackLightPlus.Spacetime.metric(w::LapseControlled, x::SVector{4}) = metric(w.st, x)
BlackLightPlus.Spacetime.inverse_metric(w::LapseControlled, x::SVector{4}) = inverse_metric(w.st, x)
BlackLightPlus.Spacetime.metric_and_inverse(w::LapseControlled, x::SVector{4}) = metric_and_inverse(w.st, x)
function proxy_distance(w::LapseControlled, x::SVector{4})
    gtt = inverse_metric(w.st, x)[1, 1]
    return (gtt < 0 ? 1 / sqrt(-gtt) : one(gtt)) - w.alpha_min, sqrt(x[2]^2 + x[3]^2 + x[4]^2)
end
horizon_floor(w::LapseControlled) = w.alpha_min
lane(w::LapseControlled, ::Type) = w

maxabs(A) = maximum(abs, A)

# ---- leg points
function leg_points(rec)
    node = (1.0, -0.5, 3.0)
    tev = [SVector(t, node...) for t in range(3.1, 20.3; length = 2000)]
    rows_t = Vector{Vector{Float64}}()
    for dt in (2.0, 1.0, 0.5, 0.25, 0.125)
        ax(c) = (c - 2.0):1.0:(c + 2.0)
        sl = [adm_grid_from_metric(x -> metric(BIN, x), ax(node[1]), ax(node[2]), ax(node[3]); t = t)
              for t in -4.0:dt:24.0]
        ser = ADMSeries(sl, collect(-4.0:dt:24.0))
        ev = maximum(x -> maxabs(inverse_metric(ser, x) - inverse_metric(BIN, x)), tev)
        ed = maximum(x -> maxabs(inverse_metric_gradient(ser, x)[:, :, 1] - inverse_metric_gradient(BIN, x)[:, :, 1]), tev)
        push!(rows_t, [dt, ev, ed])
        @printf("points time  dt %.4f  value %.3e  d_t %.3e\n", dt, ev, ed)
    end
    c = SVector(0.3, -0.2, 3.2)
    xev = [SVector(5.0, (c .+ SVector(sin(1.3k), cos(2.1k), sin(0.7k + 1)))...) for k in 1:400]
    rows_x = Vector{Vector{Float64}}()
    for dx in (0.5, 0.25, 0.125, 0.0625)
        ax(k) = (c[k] - 2.0):dx:(c[k] + 2.0)
        sl = [adm_grid_from_metric(x -> metric(BIN, x), ax(1), ax(2), ax(3); t = t) for t in (0.0, 5.0, 10.0)]
        ser = ADMSeries(sl, [0.0, 5.0, 10.0])
        ev = maximum(x -> maxabs(inverse_metric(ser, x) - inverse_metric(BIN, x)), xev)
        ed = maximum(x -> maxabs(inverse_metric_gradient(ser, x)[:, :, 2:4] - inverse_metric_gradient(BIN, x)[:, :, 2:4]), xev)
        push!(rows_x, [dx, ev, ed])
        @printf("points space dx %.4f  value %.3e  d_x %.3e\n", dx, ev, ed)
    end
    write_npy(joinpath(OUT, "points_time.npy"), permutedims(reduce(hcat, rows_t)))
    write_npy(joinpath(OUT, "points_space.npy"), permutedims(reduce(hcat, rows_x)))
    rec["points"] = Dict("time" => rows_t, "space" => rows_x, "node" => collect(node), "centre" => collect(c))
end

# ---- fixed-step rays and the largest drift of a linear charge q(y)
function march(st, x0, q; h = 0.02, n = 600)
    p, _ = null_covariant(st, x0, SVector(-1.0, 0.0, 0.0))
    y = vcat(x0, p)
    q0 = q(y); dq = 0.0
    for _ in 1:n
        y = rk4_step(st, y, h)
        dq = max(dq, abs(q(y) - q0))
    end
    return y, dq
end
# The closest approach of the analytic march to any hole, in units of its horizon radius.
function rmin(st, x0, N; h = 0.02, n = 600)
    p, _ = null_covariant(st, x0, SVector(-1.0, 0.0, 0.0))
    y = vcat(x0, p); r = Inf
    for _ in 1:n
        y = rk4_step(st, y, h)
        r = min(r, minimum(A -> ks_radius(st, SVector(y[1], y[2], y[3], y[4]), A) / horizon_radius(st, A), 1:N))
    end
    return r
end

function leg_rays(rec)
    qE(y) = y[5] + V[1] * y[6] + V[2] * y[7] + V[3] * y[8]
    starts = [SVector(0.0, 6.0, b * cos(θ), b * sin(θ)) for b in (3.0, 4.5, 6.0) for θ in range(0, 2π; length = 9)[1:8]]
    keep = [i for i in eachindex(starts) if rmin(HOLE, starts[i], 1) > 1.5]
    ref = [march(HOLE, starts[i], qE) for i in keep]
    inside = [all(isfinite, r[1]) && maximum(abs, r[1][2:4]) < 7.5 for r in ref]
    keep = keep[inside]; ref = ref[inside]
    @printf("rays: %d of %d rays outside 1.5 r_h; analytic max |ΔE'| %.2e\n", length(keep), length(starts),
            maximum(r[2] for r in ref))
    rows = Vector{Vector{Float64}}()
    for (dx, dt) in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.25))
        ser = series(HOLE, 1, dx, collect(-4dt:dt:24.0 + 4dt); L = 8.0)
        res = Vector{Any}(undef, length(keep))
        Threads.@threads for j in eachindex(keep)
            res[j] = march(ser, starts[keep[j]], qE)
        end
        ex = [norm(res[j][1][2:4] - ref[j][1][2:4]) for j in eachindex(keep)]
        ep = [norm(res[j][1][5:8] - ref[j][1][5:8]) for j in eachindex(keep)]
        eq = [res[j][2] for j in eachindex(keep)]
        push!(rows, [dx, dt, maximum(ex), maximum(ep), maximum(eq), median(ex), median(eq)])
        @printf("rays dx %.3f dt %.3f  |dx| %.3e  |dp| %.3e  |dE'| %.3e\n", dx, dt, maximum(ex), maximum(ep), maximum(eq))
        ser = nothing; GC.gc()
    end
    write_npy(joinpath(OUT, "rays.npy"), permutedims(reduce(hcat, rows)))
    # time only: Δx = 0.25, against Δt = 0.0625 on the same lattice
    sref = series(HOLE, 1, 0.25, collect(-0.25:0.0625:24.25); L = 8.0)
    rr = Vector{Any}(undef, length(keep))
    Threads.@threads for j in eachindex(keep)
        rr[j] = march(sref, starts[keep[j]], qE)
    end
    sref = nothing; GC.gc()
    rows_t = Vector{Vector{Float64}}()
    for dt in (2.0, 1.0, 0.5, 0.25, 0.125)
        ser = series(HOLE, 1, 0.25, collect(-4dt:dt:24.0 + 4dt); L = 8.0)
        res = Vector{Any}(undef, length(keep))
        Threads.@threads for j in eachindex(keep)
            res[j] = march(ser, starts[keep[j]], qE)
        end
        ex = maximum(norm(res[j][1][2:4] - rr[j][1][2:4]) for j in eachindex(keep))
        ep = maximum(norm(res[j][1][5:8] - rr[j][1][5:8]) for j in eachindex(keep))
        eq = maximum(abs(res[j][2] - rr[j][2]) for j in eachindex(keep))
        push!(rows_t, [dt, ex, ep, eq])
        @printf("rays time-only dt %.4f  |dx| %.3e  |dp| %.3e  |d dE'| %.3e\n", dt, ex, ep, eq)
        ser = nothing; GC.gc()
    end
    write_npy(joinpath(OUT, "rays_time.npy"), permutedims(reduce(hcat, rows_t)))
    rec["rays"] = Dict("diagonal" => rows, "time_only" => rows_t, "n_rays" => length(keep),
                       "analytic_max_dE" => maximum(r[2] for r in ref), "step" => 0.02, "steps" => 600)
end

function leg_helical(rec)
    ω = ORBIT.omega
    qJ(y) = y[5] + ω * (y[2] * y[7] - y[3] * y[6])
    starts = [SVector(0.0, 7.0, b * cos(θ), b * sin(θ)) for b in (3.5, 5.0, 6.5) for θ in range(0, 2π; length = 9)[1:8]]
    keep = [i for i in eachindex(starts) if rmin(BIN, starts[i], 2; n = 700) > 1.5]
    ref = [march(BIN, starts[i], qJ; n = 700) for i in keep]
    inside = [all(isfinite, r[1]) && maximum(abs, r[1][2:4]) < 8.5 for r in ref]
    keep = keep[inside]; ref = ref[inside]
    rows = [[0.0, 0.0, maximum(r[2] for r in ref), median([r[2] for r in ref])]]
    @printf("helical: %d rays; analytic max |ΔJ| %.2e\n", length(keep), rows[1][3])
    for (dx, dt) in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.125, 0.25))
        ser = series(BIN, 2, dx, collect(-4dt:dt:40.0 + 4dt); L = 9.0)
        res = Vector{Any}(undef, length(keep))
        Threads.@threads for j in eachindex(keep)
            res[j] = march(ser, starts[keep[j]], qJ; n = 700)
        end
        eq = [res[j][2] for j in eachindex(keep)]
        push!(rows, [dx, dt, maximum(eq), median(eq)])
        @printf("helical dx %.3f dt %.3f  max|dJ| %.3e  median %.3e\n", dx, dt, maximum(eq), median(eq))
        ser = nothing; GC.gc()
    end
    write_npy(joinpath(OUT, "helical.npy"), permutedims(reduce(hcat, rows)))
    rec["helical"] = Dict("rows" => rows, "n_rays" => length(keep), "omega" => ω)
end

# ---- the plasma
torus(x, y, z) = (r = sqrt(x^2 + y^2 + z^2); (r < 2.2 || r > 11.0) ? 0.0 : exp(-((r - 5.5) / 2.5)^2))
function axisymmetric_deck(; nc = 12, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, 8, nc, nc, nc, nmb)
    lo = zeros(3, nmb); hi = zeros(3, nmb); xl = zeros(3, nmb); dxs = fill(dx, 3, nmb)
    ll = zeros(Int, 3, nmb); m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        lo[:, m] .= l; hi[:, m] .= l .+ w; xl[:, m] .= l .+ dx / 2; ll[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            x = xl[1, m] + (i - 1) * dx; y = xl[2, m] + (j - 1) * dx; z = xl[3, m] + (k - 1) * dx
            ρ = torus(x, y, z); ρ > 0 || continue
            r = sqrt(x^2 + y^2 + z^2)
            data[:, i, j, k, m] .= Float32.((ρ, -0.2y / r, 0.2x / r, 0.0, 0.08ρ, 0.0, 0.0, 0.3sqrt(ρ)))
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), ll)
end
const MBH = 6.2e9 * 1.989e33
const LU = GNEWT * MBH / CL^2
const CFG = RunConfig(m_bh_cgs = MBH, mass_scale_cgs = 1.0e-21 * LU^3, frequency_hz = 2.3e11)
const AMIN = 0.3

function leg_mc(rec)
    snap = axisymmetric_deck()
    params = prepare(CFG; spacetime = BIN, matter = snap).params
    τ = collect(0.0:2.0:12.0)
    frames = FrameStack(fill(snap, length(τ)), 0.0, 2.0)
    win = (2.0, 10.0)
    prof = [clip_profile([t - 2, t, t + 2], [0.0, 1.0, 0.0], win...) for t in τ]
    nodes = findall(p -> profile_area(p) > 0, prof)
    coarse = series(BIN, 2, 0.5, collect(-2.0:1.0:56.0); L = 8.0, amin = AMIN)
    per = [filter(s -> proxy_distance(coarse, s.x)[1] > 0.05,
                  emission_sites(snap.blocks, BIN, params; t0 = τ[m], r_emit_max = 5.0)) for m in nodes]
    coarse = nothing
    W = [profile_area(prof[m]) / (win[2] - win[1]) for m in nodes]
    sites = reduce(vcat, [[ZoneSite{Float64}(z.x, z.dV * W[i], z.n_e, z.theta_e, z.B) for z in p]
                          for (i, p) in enumerate(per)])
    wgt, Nsz, dnmax = zone_tables(sites, 20_000 / log(1e16 / 1e8))
    offset = [0; cumsum(round.(Int, Nsz))]
    zf = [0; cumsum(length.(per))]
    source(st) = EventSource([ZoneSource(sites[zf[i] + 1:zf[i + 1]], snap.blocks, st, params, wgt,
                                         EMISSIVITY_TABLE, dnmax[zf[i] + 1:zf[i + 1]],
                                         offset[zf[i] + 1:zf[i + 1] + 1]) for i in eachindex(per)],
                             offset[zf .+ 1], prof[nodes])
    N = offset[end]
    opac = Opacities(params)
    function run(st)
        src = source(st)
        ctl = CurvedMarch(st; C = 64.0, escape = 7.5, R_em = 7.0, plunge = false)
        out = Vector{NTuple{9,Float64}}(undef, N)
        Threads.@threads for i in 1:N
            sph, rng = emit(src, i, primary_rng(0x11, i))
            r = track!(ctl, st, frames, opac, sph, rng, NullSink())
            d = escape_direction(ctl, r.sph)
            n = d / norm(d)
            out[i] = (r.recorded ? 1.0 : 0.0, sph.E, r.sph.E, r.sph.w, arrival_time(r.sph.x, d), n[1], n[2], n[3],
                      r.sph.x[1])
        end
        return out
    end
    t0 = time()
    A = run(LapseControlled(BIN, AMIN))
    @printf("mc: %d zones, %d primaries, analytic run %.0f s, %d escaped\n", length(sites), N, time() - t0,
            count(a -> a[1] == 1.0, A))
    esc(R) = [i for i in 1:N if R[i][1] == 1.0 && R[i][9] < 54.0]
    Ebins = exp.(range(log(1e-12), log(1e-5); length = 41))
    function spectrum(R)
        s = zeros(40)
        for i in esc(R)
            k = searchsortedlast(Ebins, R[i][3])
            1 <= k <= 40 && (s[k] += R[i][3] * R[i][4])
        end
        return s
    end
    SA = spectrum(A)
    rows = Vector{Vector{Float64}}(); specs = [SA]
    for (dx, dt) in ((0.5, 1.0), (0.25, 0.5), (0.125, 0.25))
        t1 = time()
        ser = series(BIN, 2, dx, collect(-2.0:dt:56.0); L = 8.0, amin = AMIN)
        B = run(ser)
        ser = nothing; GC.gc()
        both = [i for i in 1:N if A[i][1] == B[i][1] == 1.0 && A[i][9] < 54.0 && A[i][4] > 0 && A[i][2] > 0]
        flips = count(i -> A[i][1] != B[i][1], 1:N)
        q(v) = (median(v), quantile(v, 0.99))
        e0 = [abs(B[i][2] / A[i][2] - 1) for i in both]
        eE = [abs(B[i][3] / A[i][3] - 1) for i in both]
        ew = [abs(B[i][4] / A[i][4] - 1) for i in both]
        et = [abs(B[i][5] - A[i][5]) for i in both]
        en = [norm(SVector(B[i][6] - A[i][6], B[i][7] - A[i][7], B[i][8] - A[i][8])) for i in both]
        SB = spectrum(B); push!(specs, SB)
        L1 = sum(abs, SB - SA) / sum(SA)
        push!(rows, [dx, dt, flips, length(both), q(e0)..., q(eE)..., q(ew)..., q(et)..., q(en)..., L1])
        @printf("mc dx %.3f dt %.3f flips %d paired %d  E0 %.2e  E %.2e/%.2e  w %.2e/%.2e  tarr %.2e/%.2e  n %.2e  L1 %.2e  (%.0f s)\n",
                dx, dt, flips, length(both), q(e0)[1], q(eE)..., q(ew)..., q(et)..., q(en)[1], L1, time() - t1)
    end
    write_npy(joinpath(OUT, "mc.npy"), permutedims(reduce(hcat, rows)))
    Ec = sqrt.(Ebins[1:end-1] .* Ebins[2:end])
    write_npy(joinpath(OUT, "mc_spectra.npy"), hcat(Ec, reduce(hcat, specs)))
    rec["mc"] = Dict("rows" => rows, "zones" => length(sites), "primaries" => N, "window" => collect(win),
                     "frames" => τ, "alpha_min" => AMIN, "R_em" => 7.0, "C" => 64.0,
                     "escaped_analytic" => count(a -> a[1] == 1.0, A),
                     "columns" => "dx dt flips paired med/p99: dE_emit dE dw dtarr dn, L1")
end

function leg_e2e(rec)
    snap = axisymmetric_deck()
    frames = FrameStack(fill(snap, 11), 0.0, 2.0)
    kw = (seed = 5, C = 32.0, backend = :cpu, window = (2.0, 18.0), time_bins = 8, r_emit_max = 5.0,
          R_em = 7.0, escape = 7.5, bands = InclinationBands((30.0, 60.0, 90.0), 15.0))
    ser = series(BIN, 2, 0.25, collect(-2.0:0.5:60.0); L = 8.0, amin = AMIN)
    rs = mc_spectrum(prepare(CFG; spacetime = ser, matter = frames), 40_000; kw...)
    ra = mc_spectrum(prepare(CFG; spacetime = LapseControlled(BIN, AMIN), matter = frames), 40_000; kw...)
    z = Float64[]; P = Vector{Vector{Float64}}()
    for (a, b) in zip(rs.result.spectra, ra.result.spectra)
        pa, pb = sum(a.sumwE), sum(b.sumwE)
        ea, eb = sqrt(sum(a.sumwE2)), sqrt(sum(b.sumwE2))
        push!(z, (pa - pb) / hypot(ea, eb)); push!(P, [pa, ea, pb, eb])
    end
    @printf("e2e: band z-scores %s, beyond_series %.3f\n", string(round.(z; digits = 2)), rs.beyond_series)
    rec["e2e"] = Dict("z" => z, "powers_series_analytic" => P, "shares" => rs.shares,
                      "beyond_series" => rs.beyond_series, "primaries" => 40_000)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
json_str(x::Bool) = x ? "true" : "false"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ", ") * "]"
json_str(x::Tuple) = json_str(collect(x))
function json_str(d::AbstractDict)
    items = [json_str(string(k)) * ": " * json_str(d[k]) for k in sort!(collect(keys(d)))]
    return "{" * join(items, ", ") * "}"
end

function main()
    t0 = time()
    mkpath(OUT)
    rec = Dict{String,Any}("git_sha" => GIT_SHA, "threads" => Threads.nthreads(), "legs" => collect(String, LEGS),
                           "binary" => Dict("masses" => [0.5, 0.5], "spins_z" => [0.3, -0.2], "separation" => 6.0,
                                            "omega" => ORBIT.omega),
                           "hole" => Dict("mass" => 0.5, "spin_z" => 0.3, "velocity" => collect(V), "s0" => collect(S0)))
    "points" in LEGS && leg_points(rec)
    "rays" in LEGS && leg_rays(rec)
    "helical" in LEGS && leg_helical(rec)
    "mc" in LEGS && leg_mc(rec)
    "e2e" in LEGS && leg_e2e(rec)
    rec["wall_seconds"] = time() - t0
    open(joinpath(OUT, "record_" * join(LEGS, "_") * ".json"), "w") do io
        write(io, json_str(rec), "\n")
    end
    @printf("wrote %s (%.0f s on %d threads)\n", OUT, time() - t0, Threads.nthreads())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
