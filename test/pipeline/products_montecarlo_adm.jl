# The Monte Carlo over an emission window on a spacetime that changes in time without a
# helical symmetry, where the frames of a `FrameStack` held in place emit each photon at its
# own event: on a stationary spacetime the photon generated at its event is the placed photon
# bit for bit; on the orbiting binary stripped of its declared symmetry a stack of an
# axisymmetric plasma (each frame decoded at its time, each photon generated at its event)
# agrees with the same plasma co-rotating under the exact helical map on the binary itself,
# and one frame, a carried stack, a snapshot or a frame sequence is refused there, as a stack
# in place is on the binary; a user metric of t emits frame by frame; through an ADM series
# of the binary every paired photon (same zone, same draws) converges to its twin through the
# analytic binary at third order in the slice and grid spacing (the same lapse capture and
# step rule on both); and the product runs end to end through the series, without and over a
# window, in agreement with the analytic binary within errors.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra, Statistics
using BlackLightPlus.Plans: RunConfig, prepare
using BlackLightPlus.Products: mc_spectrum, TimedSpectrumResult, ZoneSource, TimedSource, EventSource, emit,
                    TimeProfile, clip_profile, profile_area, arrival_time, GRMHDSpectrumResult
using BlackLightPlus.Spacetime: Spacetime, KerrSchildCartesian, BoostedSuperposedKerrSchild, CircularOrbit, UserMetric,
                     ADMGrid, ADMSeries, adm_grid_from_metric, metric, inverse_metric, metric_and_inverse,
                     ks_radius, horizon_radius, helical_symmetry
import BlackLightPlus.Spacetime: proxy_distance, horizon_floor, lane
using BlackLightPlus.Data: AthenaKSnapshot, FrameStack, FrameSequence, Corotating, emission_sites,
                zone_tables
using BlackLightPlus.Matter: ZoneSite
using BlackLightPlus.MonteCarlo: track!, CurvedMarch, Opacities, NullSink, EMISSIVITY_TABLE, escape_direction
using BlackLightPlus.Execution: InclinationBands, Spectrum
using BlackLightPlus.Numerics: GNEWT, CL, primary_rng, MCRng, mc_uniform
const DK = Main.Decks

const L_TORUS = GNEWT * DK.TORUS_MBH / CL^2
const CFG = RunConfig(m_bh_cgs = DK.TORUS_MBH, mass_scale_cgs = DK.TORUS_DENS * L_TORUS^3,
                      frequency_hz = 2.3e11)

# The torus deck's mesh with an axisymmetric plasma: density and heat of the torus, azimuthal
# velocity, vertical field (`Decks.amr_deck` layout, `nc` cells per block side).
function axisymmetric_deck(; nc = 12, half = 12.0)
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
            ρ = Float64(DK.torus_primitives(x, y, z)[1])
            ρ > 0 || continue
            r = sqrt(x^2 + y^2 + z^2)
            data[:, i, j, k, m] .= Float32.((ρ, -0.2y / r, 0.2x / r, 0.0, 0.08ρ, 0.0, 0.0, 0.3sqrt(ρ)))
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), ll)
end

# The analytic metric under the series' capture and step rule: the lapse α = (-g^tt)^(-1/2)
# above `alpha_min`, as `proxy_distance` of an `ADMSeries` reads it.
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

# A spacetime that declares no symmetry: the metric, capture surface and step rule of `st`.
struct Undeclared{S} <: Spacetime{Float64}
    st::S
end
BlackLightPlus.Spacetime.metric(w::Undeclared, x::SVector{4}) = metric(w.st, x)
BlackLightPlus.Spacetime.inverse_metric(w::Undeclared, x::SVector{4}) = inverse_metric(w.st, x)
BlackLightPlus.Spacetime.metric_and_inverse(w::Undeclared, x::SVector{4}) = metric_and_inverse(w.st, x)
proxy_distance(w::Undeclared, x::SVector{4}) = proxy_distance(w.st, x)
horizon_floor(w::Undeclared) = horizon_floor(w.st)
lane(w::Undeclared, ::Type) = w

@testset "Monte Carlo through a metric that changes in time" begin
    snap = axisymmetric_deck()
    orbit = CircularOrbit((0.5, 0.5), 6.0)
    Ω = orbit.omega
    bin = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.3), (0.0, 0.0, -0.2)], orbit)

    @testset "on a stationary spacetime the photon at its event is the placed photon" begin
        kerr = KerrSchildCartesian(DK.TORUS_A)
        params = prepare(CFG; spacetime = kerr, matter = snap).params
        sites = emission_sites(snap.blocks, kerr, params; r_emit_max = 8.0)
        wgt, Nsz, dnmax = zone_tables(sites, 3000 / log(1e16 / 1e8))
        offset = [0; cumsum(ceil.(Int, Nsz))]
        zs = ZoneSource(sites, snap.blocks, kerr, params, wgt, EMISSIVITY_TABLE, dnmax, offset)
        prof = [TimeProfile([0.0, 4.0, 8.0], [0.0, 1.0, 0.5])]
        ev = EventSource([zs], [0], prof)
        pl = TimedSource([zs], [0], prof, 0.0)
        same = 0
        for g in 1:offset[end]
            a, ra = emit(ev, g, primary_rng(0x5eed, g)); b, rb = emit(pl, g, primary_rng(0x5eed, g))
            same += (a.x === b.x && a.p === b.p && a.w === b.w && a.x1i === b.x1i && ra == rb)
        end
        @test same == offset[end] > 1000
    end

    @testset "a frame stack without a symmetry: each photon at its event" begin
        free = Undeclared(bin)
        @test helical_symmetry(bin) && helical_symmetry(KerrSchildCartesian(0.5)) && !helical_symmetry(free)
        frames = FrameStack(fill(snap, 11), 0.0, 2.0)
        kw = (seed = 5, C = 32.0, backend = :cpu, window = (2.0, 18.0), time_bins = 8, r_emit_max = 5.0,
              R_em = 7.0, escape = 7.5, bands = InclinationBands((30.0, 60.0, 90.0), 15.0))
        rf = mc_spectrum(prepare(CFG; spacetime = free, matter = frames), 30_000; kw...)
        rc = mc_spectrum(prepare(CFG; spacetime = bin, matter = Corotating(snap, Ω)), 30_000; kw...)
        @test rf isa TimedSpectrumResult && rf.omega == 0.0 && rc.omega == Ω
        @test rf.shares == [0.0625; fill(0.125, 7); 0.0625]               # equal frames, nine nodes
        z = [(sum(a.sumwE) - sum(b.sumwE)) / hypot(sqrt(sum(a.sumwE2)), sqrt(sum(b.sumwE2)))
             for (a, b) in zip(rf.result.spectra, rc.result.spectra)]
        @test maximum(abs, z) < 4.0 && sum(abs2, z) / 3 < 4.0
        @test sum(rf.result.spectra[2].count) > 1000
        # a snapshot alone, one frame, a carried stack or a frame sequence cannot follow the
        # metric; on the binary, which has the helical symmetry, a stack in place is refused
        refused(st, m) = @test_throws ArgumentError mc_spectrum(prepare(CFG; spacetime = st, matter = m), 100; kw...)
        refused(free, snap)
        refused(free, FrameStack([snap], 10.0, 1.0))
        refused(free, FrameStack(fill(snap, 11), 0.0, 2.0; omega = Ω))
        refused(free, FrameSequence(collect(0.0:2.0:20.0), fill(snap.blocks, 11)))
        refused(bin, frames)
    end

    @testset "a user metric of t" begin
        # a softened weak-field potential whose mass oscillates: no horizon, ∂_t g ≠ 0
        Φ(x) = -0.3 * (1 + 0.5 * sin(0.2 * x[1])) / sqrt(x[2]^2 + x[3]^2 + x[4]^2 + 4)
        um = UserMetric(x -> (f = 2 * Φ(x); SMatrix{4,4}(-(1 + f), 0, 0, 0, 0, 1 - f, 0, 0,
                                                       0, 0, 1 - f, 0, 0, 0, 0, 1 - f)))
        frames = FrameStack(fill(snap, 7), 0.0, 2.0)
        r = mc_spectrum(prepare(CFG; spacetime = um, matter = frames), 3000; seed = 3, C = 32.0,
                        backend = :cpu, window = (2.0, 10.0), time_bins = 4, r_emit_max = 6.0,
                        R_em = 8.0, escape = 8.5, bands = InclinationBands((30.0, 60.0, 90.0), 15.0))
        @test r.shares == [0.125, 0.25, 0.25, 0.25, 0.125] && r.omega == 0.0
        @test r.result.n_recorded > 1000 && all(isfinite, r.total.sumwE) && sum(r.total.count) > 1000
    end

    # The binary's ADM slices: lattice [-L, L]³ at spacing dx, slices every dt from -2 to T.
    g(x) = metric(bin, x)
    inner(x) = any(A -> ks_radius(bin, x, A) < 0.5 * horizon_radius(bin, A), 1:2)
    function series(dx, dt; L = 7.0, T = 40.0)
        ax = -L:dx:L
        ts = collect(-2.0:dt:T)
        sl = Vector{ADMGrid{Float64,Array{Float64,4}}}(undef, length(ts))
        Threads.@threads for k in eachindex(ts)
            sl[k] = adm_grid_from_metric(g, ax, ax, ax; alpha_min = 0.3, excise = inner, t = ts[k])
        end
        return ADMSeries(sl, ts)
    end
    coarse = series(0.5, 1.0)
    fine = series(0.25, 0.5)
    lapse = LapseControlled(bin, 0.3)

    @testset "paired photons through the series converge to the analytic binary" begin
        params = prepare(CFG; spacetime = bin, matter = snap).params
        τ = collect(0.0:2.0:8.0)
        frames = FrameStack(fill(snap, length(τ)), 0.0, 2.0)
        win = (2.0, 6.0)
        prof = [clip_profile([t - 2, t, t + 2], [0.0, 1.0, 0.0], win...) for t in τ]
        keep = findall(p -> profile_area(p) > 0, prof)
        per = [filter(s -> proxy_distance(coarse, s.x)[1] > 0.05,
                      emission_sites(snap.blocks, bin, params; t0 = τ[m], r_emit_max = 4.5)) for m in keep]
        W = [profile_area(prof[m]) / (win[2] - win[1]) for m in keep]
        sites = reduce(vcat, [[ZoneSite{Float64}(z.x, z.dV * W[i], z.n_e, z.theta_e, z.B) for z in p]
                              for (i, p) in enumerate(per)])
        wgt, Nsz, dnmax = zone_tables(sites, 3000 / log(1e16 / 1e8))
        offset = [0; cumsum(round.(Int, Nsz))]
        zf = [0; cumsum(length.(per))]
        source(st) = EventSource([ZoneSource(sites[zf[i] + 1:zf[i + 1]], snap.blocks, st, params, wgt,
                                             EMISSIVITY_TABLE, dnmax[zf[i] + 1:zf[i + 1]],
                                             offset[zf[i] + 1:zf[i + 1] + 1]) for i in eachindex(per)],
                                 offset[zf .+ 1], prof[keep])
        N = offset[end]
        opac = Opacities(params)
        function run(st)
            src = source(st)
            ctl = CurvedMarch(st; C = 64.0, escape = 6.5, R_em = 6.0, plunge = false)
            out = Vector{NTuple{5,Float64}}(undef, N)
            Threads.@threads for i in 1:N
                sph, rng = emit(src, i, primary_rng(0x11, i))
                r = track!(ctl, st, frames, opac, sph, rng, NullSink())
                d = escape_direction(ctl, r.sph)
                out[i] = (r.recorded ? 1.0 : 0.0, sph.E, r.sph.E, arrival_time(r.sph.x, d), r.sph.x[1])
            end
            return out
        end
        A = run(lapse)
        errs = map((coarse, fine)) do st
            B = run(st)
            both = findall(i -> A[i][1] == B[i][1] == 1.0 && A[i][5] < 38.0, 1:N)
            (count(i -> A[i][1] != B[i][1], 1:N), length(both),
             median(abs(B[i][2] / A[i][2] - 1) for i in both), median(abs(B[i][3] / A[i][3] - 1) for i in both),
             median(abs(B[i][4] - A[i][4]) for i in both))
        end
        @test N > 2000 && errs[2][2] > 0.5N
        @test errs[2][1] <= errs[1][1] && errs[2][1] < 0.01N             # fates that differ
        for k in 3:5                                                      # emission E, escape E, arrival
            @test errs[1][k] / errs[2][k] > 5.0
        end
        @test errs[2][4] < 1e-5
    end

    @testset "the product through the series" begin
        frames = FrameStack(fill(snap, 8), 0.0, 2.0)
        kw = (seed = 9, C = 32.0, backend = :cpu, r_emit_max = 4.5, R_em = 6.0, escape = 6.5,
              bands = InclinationBands((30.0, 60.0, 90.0), 15.0))
        st = mc_spectrum(prepare(CFG; spacetime = fine, matter = snap), 4000; kw...)
        @test st isa GRMHDSpectrumResult && st.n_recorded > 1000 && all(isfinite, st.total.sumwE)
        rs = mc_spectrum(prepare(CFG; spacetime = fine, matter = frames), 12_000; kw..., window = (2.0, 12.0),
                         time_bins = 6)
        ra = mc_spectrum(prepare(CFG; spacetime = lapse, matter = frames), 12_000; kw..., window = (2.0, 12.0),
                         time_bins = 6)
        @test rs.shares == ra.shares && length(rs.shares) == 6
        z = [(sum(a.sumwE) - sum(b.sumwE)) / hypot(sqrt(sum(a.sumwE2)), sqrt(sum(b.sumwE2)))
             for (a, b) in zip(rs.result.spectra, ra.result.spectra)]
        @test maximum(abs, z) < 4.0
        # the march past the last slice: the metric frozen there, the time reported
        short = ADMSeries([adm_grid_from_metric(g, -7.0:0.5:7.0, -7.0:0.5:7.0, -7.0:0.5:7.0;
                                                alpha_min = 0.3, excise = inner, t = t) for t in (-2.0, 4.0, 10.0)],
                          [-2.0, 4.0, 10.0])
        rshort = mc_spectrum(prepare(CFG; spacetime = short, matter = frames), 3000; kw..., window = (2.0, 12.0),
                             time_bins = 6)
        @test rshort.beyond_series > rs.beyond_series > 0.0
    end
end
