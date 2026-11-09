# Monte Carlo spectra in time: emission spread over a window [t_a, t_b] of coordinate time,
# transport through plasma and metric at each superphoton's own time, and binning in the
# arrival time at a distant observer. The emission is a sum of nodes: node m emits the
# stationary photons of one plasma state (a frame of a series, a variable sphere at a node,
# a snapshot), at times drawn from its profile φ_m(t) (the hat of a frame, a uniform window),
# carried to that time by the time translation and, for a co-rotating plasma, the rotation
# R_z(ω t) of the helical symmetry; on a metric without any helical symmetry the frames of a
# stack held in place generate each photon at its own event (`EventSource`). The node's
# zones enter the generation tables with the measure W_m dV, W_m = ∫φ_m dt/(t_b − t_a), so
# the weight table is the window mean of the emission and the expected number of primaries
# is the target; with one node and a uniform profile, W = 1 and every table, count and
# photon is the stationary module's, bit for bit, but for its emission time. The emission
# time draws from the primary's identity at draw 2^31 (`time_stream`), so the stationary
# draws are untouched.
#
# A packet of weight w stands for w (t_b − t_a) T_unit physical photons, T_unit = L_unit/c
# per unit of coordinate time; a photon escaping at (t, x) with spatial direction n̂
# reaches a distant observer along n̂ at t_arr = t − n̂·x up to a constant common to all
# photons. Arrival bin k of width Δt then has the isotropic-equivalent
#   ν L_ν = (mₑc²/Δln ν)(4π/ΔΩ)((t_b − t_a)/Δt) Σ_k w E,
# whose sum over the bins, weighted by Δt/(t_b − t_a), is the window mean of the
# stationary estimator.

"""
    TimeProfile(t, v)

A piecewise-linear density of emission time through the knots `(t[k], v[k])` (at least
two, ascending times, non-negative values), zero outside `[t[1], t[end]]`; `cum[k]` is
`∫ φ dt` up to knot `k` by the trapezoid rule, exact for the profile, and
[`profile_area`](@ref) the whole integral. [`sample_time`](@ref) draws from it exactly.
"""
struct TimeProfile
    t::Vector{Float64}
    v::Vector{Float64}
    cum::Vector{Float64}
end

function TimeProfile(t::AbstractVector{<:Real}, v::AbstractVector{<:Real})
    (length(t) == length(v) >= 2 && issorted(t) && all(>=(0), v)) ||
        throw(ArgumentError("TimeProfile: at least two ascending knots with non-negative values"))
    cum = zeros(Float64, length(t))
    for k in 1:(length(t) - 1)
        cum[k + 1] = cum[k] + 0.5 * (v[k] + v[k + 1]) * (t[k + 1] - t[k])
    end
    return TimeProfile(collect(Float64, t), collect(Float64, v), cum)
end

"The integral `∫ φ dt` of a profile."
profile_area(p::TimeProfile) = p.cum[end]

"""
    clip_profile(t, v, t_a, t_b) -> TimeProfile

The profile through the knots `(t, v)`, zero outside them, restricted to `[t_a, t_b]`:
knots at `t_a`, at every knot strictly inside and at `t_b`.
"""
function clip_profile(t::AbstractVector, v::AbstractVector, t_a::Real, t_b::Real)
    val(s) = s < t[1] || s > t[end] ? 0.0 :
             (k = clamp(searchsortedlast(t, s), 1, length(t) - 1);
              w = (s - t[k]) / (t[k + 1] - t[k]); (1.0 - w) * v[k] + w * v[k + 1])
    ts = [Float64(t_a)]; vs = [val(t_a)]
    for k in eachindex(t)
        t_a < t[k] < t_b && (push!(ts, t[k]); push!(vs, v[k]))
    end
    push!(ts, Float64(t_b)); push!(vs, val(t_b))
    return TimeProfile(ts, vs)
end

"""
    sample_time(p::TimeProfile, rng) -> (t, rng)

A time drawn from the profile `p`: a segment by its share of the area (no draw for a
single segment), then the linear density `(1 − s) v₀ + s v₁` on it as the mixture of
`2(1 − s)` and `2s` with the weights `v₀` and `v₁` (one uniform when `v₀ = v₁`).
"""
@inline function sample_time(p::TimeProfile, rng::MCRng)
    n = length(p.t)
    k = 1
    if n > 2
        u, rng = mc_uniform(rng)
        k = clamp(searchsortedlast(p.cum, u * p.cum[n]), 1, n - 1)
    end
    v0 = p.v[k]; v1 = p.v[k + 1]
    u, rng = mc_uniform(rng)
    if v0 == v1
        s = u
    else
        u2, rng = mc_uniform(rng)
        s = u * (v0 + v1) < v0 ? 1.0 - sqrt(1.0 - u2) : sqrt(u2)
    end
    return p.t[k] + s * (p.t[k + 1] - p.t[k]), rng
end

"""
    time_stream(rng::MCRng) -> MCRng

The stream of a primary's emission time: the primary's own identity from draw `2^31`, so
it never meets the draws of the photon's generation and march (fewer than `2^31`).
"""
@inline time_stream(rng::MCRng) = MCRng(rng.key, 0x80000000, rng.sid)

"""
    place(sph::Superphoton, t, omega) -> Superphoton

`sph`, emitted at its coordinate time `t_g = sph.x⁰` (0 for every node but the frames of a
[`FrameStack`](@ref), generated at their own times), carried to time `t`: the time
translation, and for `omega ≠ 0` the rotation `R_z(ω (t − t_g))` of its position and of
the spatial components of its covariant momentum (`E = −p_t` unchanged); the emission
coordinates follow the position. On a spacetime with the helical symmetry of `omega` the
carried momentum stays null.
"""
@inline function place(sph::Superphoton, t::Float64, omega::Float64)
    x = sph.x; p = sph.p
    if omega == 0.0
        xn = SVector(t, x[2], x[3], x[4])
        pn = p
    else
        θ = omega * (t - x[1])
        c = cos(θ); s = sin(θ)
        xn = SVector(t, c * x[2] - s * x[3], s * x[2] + c * x[3], x[4])
        pn = SVector(p[1], c * p[2] - s * p[3], s * p[2] + c * p[3], p[4])
    end
    return Superphoton(xn, pn, sph.w; tau_abs = sph.tau_abs, tau_scatt = sph.tau_scatt,
                       nscatt = sph.nscatt, x1i = xn[2], x2i = xn[3], ne0 = sph.ne0,
                       thetae0 = sph.thetae0, b0 = sph.b0)
end

"""
    TimedSource(sources, first, profiles, omega)

The primaries of a run over a window: primary `g` of node `m`, `first[m] < g ≤ first[m + 1]`,
is `emit(sources[m], g, rng)` ([`SphereSource`](@ref), [`ZoneSource`](@ref)), carried by
[`place`](@ref) to a time drawn from `profiles[m]` on its [`time_stream`](@ref) and
rotated at `omega`.
"""
struct TimedSource{S,P}
    sources::Vector{S}
    first::Vector{Int}
    profiles::Vector{P}
    omega::Float64
end
Adapt.@adapt_structure TimedSource

@inline function emit(s::TimedSource, item, rng)
    m = searchsortedlast(s.first, Int(item) - 1)
    t_e, _ = sample_time(@inbounds(s.profiles[m]), time_stream(rng))
    sph, rng = emit(@inbounds(s.sources[m]), item, rng)
    return place(sph, t_e, s.omega), rng
end

"""
    arrival_time(x, d) -> Float64

The arrival time `x⁰ − n̂·x` at a distant observer along `n̂ = d/|d|` of a photon at the
event `x` moving along the spatial direction `d`, up to a constant common to all photons
(`x⁰` for `d = 0`).
"""
@inline function arrival_time(x::SVector{4}, d::SVector{3})
    n = sqrt(d[1] * d[1] + d[2] * d[2] + d[3] * d[3])
    n > 0.0 || return Float64(x[1])
    return Float64(x[1]) - (d[1] * x[2] + d[2] * x[3] + d[3] * x[4]) / n
end

"""
    record_arrival!(timed, ctl, sph::Superphoton) -> nothing

Record the escaped `sph` of order `n` into `timed[min(n, K) + 1]` (a tuple of `ExactBins`
of `TimedBands`) at its [`arrival_time`](@ref) along `escape_direction(ctl, sph)`.
"""
@inline function record_arrival!(timed, ctl, sph::Superphoton)
    k = min(Int(sph.nscatt), length(timed) - 1) + 1
    d = escape_direction(ctl, sph)
    Execution.record!(timed[k], sph.E, sph.w, d, arrival_time(sph.x, d))
    return nothing
end

"""
    TimedRecord(timed, clock, t_last)

What a march of a run over a window records beside the stationary sink: the escape into
the time-resolved bins `timed` ([`record_arrival!`](@ref)), and into the two exact scalars
`clock` the march time after the plasma's last frame `t_last` (`Inf`: none) and the march
time, both weighted by the photon's starting weight.
"""
struct TimedRecord{B,C}
    timed::B
    clock::C
    t_last::Float64
end
Adapt.@adapt_structure TimedRecord

@inline function record_march!(r::TimedRecord, ctl, sph::Superphoton, out)
    out.recorded && record_arrival!(r.timed, ctl, out.sph)
    t0 = Float64(sph.x[1]); t1 = Float64(out.sph.x[1])
    Execution.add!(r.clock, 1, sph.w * max(0.0, t1 - max(r.t_last, t0)))
    Execution.add!(r.clock, 2, sph.w * (t1 - t0))
    return nothing
end

"""
    TimedPrimary(source, transport, sink, seed, rec::TimedRecord)

One primary of a run over a window: [`MCPrimary`](@ref) with the [`TimedSource`](@ref)
`source`, whose march is also recorded by `rec`. The march runs in the [`MCSink`](@ref)
`sink` of the stationary programs, so its `track!` is theirs.
"""
struct TimedPrimary{G,T,K,R} <: Function
    source::G
    transport::T
    sink::K
    seed::UInt32
    rec::R
end
Adapt.@adapt_structure TimedPrimary

function (p::TimedPrimary)(item, i)
    sink = at_stripe(p.sink, item)
    sph, rng = emit(p.source, item, primary_rng(p.seed, item))
    add_ledger!(sink, EMITTED, sph.w * sph.E)
    out = transport!(p.transport, sph, rng, sink)
    record_march!(p.rec, p.transport.ctl, sph, out)
    return nothing
end

"""
    TimedLayer(generation, transport, sink, rec::TimedRecord)

One scattered child of a run over a window: [`MCLayer`](@ref), whose march is also
recorded by `rec`.
"""
struct TimedLayer{G,T,K,R} <: Function
    generation::G
    transport::T
    sink::K
    rec::R
end
Adapt.@adapt_structure TimedLayer

function (p::TimedLayer)(item, i)
    g = p.generation
    sph = @inbounds g.photons[item]
    out = transport!(p.transport, sph, @inbounds(g.rngs[item]), at_stripe(p.sink, item))
    record_march!(p.rec, p.transport.ctl, sph, out)
    return nothing
end

"""
    TimedSpectrumResult

The Monte Carlo of a plasma over an emission window ([`mc_spectrum`](@ref) with
`window`). `result` is what the stationary module returns for the same matter (a
`Spectrum`, a `ScatterResult` or a `GRMHDSpectrumResult`) with every spectrum and power
the mean over the window `(t_a, t_b)`; `rule` the `TimedBands` of the time-resolved bins
and `orders[n + 1]` their `Spectrum` for the photons of scattering order `n` (`total` the
sum), at the arrival time `t − n̂·x` of a distant observer. `interior` is the span of
arrival times reached by every emitting point over the whole window; outside it a light
curve sees part of the source. `shares` are the nodes' shares `W_m` of the emission,
`omega` the rotation rate of the emission and `beyond_series` the fraction of the march
time, weighted by the photons' starting weights, spent after the last frame of a series
or the last slice of an `ADMSeries` (0 for neither), where the last frame stands in for the
plasma and the last slice for the metric;
[`arrival_times`](@ref), [`light_curve`](@ref) and [`time_spectrum`](@ref) read the bins.
"""
struct TimedSpectrumResult{R,B,S}
    result::R
    window::NTuple{2,Float64}
    rule::B
    orders::Vector{S}
    total::S
    interior::NTuple{2,Float64}
    shares::Vector{Float64}
    omega::Float64
    beyond_series::Float64
end

Base.show(io::IO, r::TimedSpectrumResult) =
    print(io, "TimedSpectrumResult(window ", r.window, ", ", r.rule.n_t, " arrival bins, ",
          length(r.shares), " nodes, ", r.result, ")")

"""
    arrival_times(res::TimedSpectrumResult) -> (centres, edges)

The centres and the `n_t + 1` edges of the arrival bins `1:n_t` (code time units).
"""
function arrival_times(res::TimedSpectrumResult)
    r = res.rule
    edges = [r.t_lo + k * r.dt for k in 0:r.n_t]
    return [r.t_lo + (k - 0.5) * r.dt for k in 1:r.n_t], edges
end

# The rows of arrival bin k for the angular row `row` of the inner rule (0: the sky, the
# all-sky row of a rule that has one or every row of a partition) and their solid angle.
function timed_rows(res::TimedSpectrumResult, k::Integer, row::Integer)
    inner = res.rule.bands
    nr = Execution.nrows(inner)
    if row == 0
        tr = Execution.total_row(inner)
        return tr == 0 ? (k * nr .+ (1:nr), 4.0 * π) : (k * nr .+ (tr:tr), 4.0 * π)
    end
    1 <= row <= nr || throw(ArgumentError("row $row outside 1:$nr"))
    return (k * nr .+ (row:row), Execution.solid_angles(inner)[row])
end

timed_spectrum(res::TimedSpectrumResult, order::Nothing) = res.total
timed_spectrum(res::TimedSpectrumResult, order::Integer) = res.orders[order + 1]

"""
    light_curve(res::TimedSpectrumResult; row = 0, nu_min = 0, nu_max = Inf, order = nothing)
        -> (t, L, sigma, count)

The isotropic-equivalent luminosity (erg s⁻¹) of the frequency bins with centres in
`[nu_min, nu_max]` in every arrival bin `1:n_t` (centres `t`), seen in the angular row
`row` of the time-resolved rule (0: the whole sky), `L_k = mₑc² (4π/ΔΩ)((t_b − t_a)/Δt)
Σ w E`, with its standard error `sigma` from `Σ (w E)²` and the photon count; `order`
selects one scattering order (`nothing`: all).
"""
function light_curve(res::TimedSpectrumResult; row::Integer = 0, nu_min::Real = 0.0,
                     nu_max::Real = Inf, order = nothing)
    spec = timed_spectrum(res, order)
    nu = nu_centers(spec)
    cols = findall(v -> nu_min <= v <= nu_max, nu)
    t, _ = arrival_times(res)
    L = zeros(length(t)); sig = zeros(length(t)); cnt = zeros(Int, length(t))
    span = res.window[2] - res.window[1]
    for k in 1:res.rule.n_t
        rows, dO = timed_rows(res, k, row)
        f = ME * CL * CL * (4.0 * π / dO) * (span / res.rule.dt)
        L[k] = f * sum(@view spec.sumwE[rows, cols])
        sig[k] = f * sqrt(sum(@view spec.sumwE2[rows, cols]))
        cnt[k] = sum(@view spec.count[rows, cols])
    end
    return t, L, sig, cnt
end

"""
    time_spectrum(res::TimedSpectrumResult, k; row = 0, order = nothing) -> (nu, L, sigma, count)

The isotropic-equivalent `ν L_ν` (erg s⁻¹) of arrival bin `k ∈ 0:n_t + 1` (0 and `n_t + 1`
the underflow and overflow bins) in the angular row `row` (0: the whole sky), at the bin
centres `nu` of the time-resolved frequency grid, with its standard error and counts.
"""
function time_spectrum(res::TimedSpectrumResult, k::Integer; row::Integer = 0, order = nothing)
    0 <= k <= res.rule.n_t + 1 || throw(ArgumentError("arrival bin $k outside 0:$(res.rule.n_t + 1)"))
    spec = timed_spectrum(res, order)
    rows, dO = timed_rows(res, k, row)
    f = ME * CL * CL / spec.dlnu * (4.0 * π / dO) * ((res.window[2] - res.window[1]) / res.rule.dt)
    L = f .* vec(sum(spec.sumwE[rows, :]; dims = 1))
    sig = f .* sqrt.(vec(sum(spec.sumwE2[rows, :]; dims = 1)))
    return nu_centers(spec), L, sig, vec(sum(spec.count[rows, :]; dims = 1))
end

# ---------------------------------------------------------------- emission nodes

# A node of the emission: the stationary matter it emits (a sphere, or a grid provider
# whose blocks give the zones), its time profile on the window and the coordinate time t0
# at which its zones are generated (0, or a frame's own time on a time-dependent metric).
struct EmissionNode{M}
    matter::M
    profile::TimeProfile
    t0::Float64
end
EmissionNode(m, profile::TimeProfile) = EmissionNode(m, profile, 0.0)

# The window of the run, checked.
function run_window(o)
    w = o.time.window
    (length(w) == 2 && isfinite(w[1]) && isfinite(w[2]) && w[2] > w[1]) ||
        throw(ArgumentError("mc_spectrum: window must be (t_a, t_b) with finite t_a < t_b"))
    return (Float64(w[1]), Float64(w[2]))
end

# The nodes of frames `frames` at times `ts` on the window `(ta, tb)`: the runs of equal
# consecutive frames (`same(a, b)`), each with the sum of its frames' hats; `anchored`
# generates each node's zones at its first frame's time.
function series_nodes(ts::AbstractVector, frames::AbstractVector, ta, tb, same; anchored = false)
    (ts[1] <= ta && tb <= ts[end]) ||
        throw(ArgumentError("mc_spectrum: the window ($ta, $tb) leaves the series span " *
                            "[$(ts[1]), $(ts[end])]"))
    M = length(ts)
    nodes = EmissionNode[]
    i = 1
    while i <= M
        j = i
        while j < M && same(frames[j], frames[j + 1])
            j += 1
        end
        kt = Float64[]; kv = Float64[]
        i > 1 && (push!(kt, ts[i - 1]); push!(kv, 0.0))
        push!(kt, ts[i]); push!(kv, 1.0)
        j > i && (push!(kt, ts[j]); push!(kv, 1.0))
        j < M && (push!(kt, ts[j + 1]); push!(kv, 0.0))
        M == 1 && (kt = [ta, tb]; kv = [1.0, 1.0])
        prof = clip_profile(kt, kv, ta, tb)
        profile_area(prof) > 0.0 &&
            push!(nodes, EmissionNode(frames[i], prof, anchored ? Float64(ts[i]) : 0.0))
        i = j + 1
    end
    return [n for n in nodes]
end

# The nodes, the transport provider and the rotation rate of the matter of a run.
uniform_node(m, ta, tb) = [EmissionNode(m, TimeProfile([ta, tb], [1.0, 1.0]))]

timed_matter(s::UniformSphere, st, ta, tb) = (uniform_node(s, ta, tb), s, 0.0)
timed_matter(s::VariableSphere, st, ta, tb) =
    (series_nodes(s.times, [node_sphere(s, m) for m in eachindex(s.times)], ta, tb,
                  (a, b) -> a.n_e == b.n_e), s, 0.0)
function timed_matter(c::Corotating, st, ta, tb)
    helical_symmetry(st, c.omega) ||
        throw(ArgumentError("mc_spectrum: a plasma co-rotating at ω = $(c.omega) needs a spacetime " *
                            "with that helical symmetry; $(nameof(typeof(st))) has none"))
    return uniform_node(c.inner, ta, tb), c, c.omega
end
function timed_matter(fs::FrameSequence, st, ta, tb)
    stationary_emission(st)
    return series_nodes(fs.times, fs.frames, ta, tb, ===), fs, 0.0
end
timed_matter(sp::SeriesProvider, st, ta, tb) = timed_matter(FrameSequence(sp.series), st, ta, tb)
# The frames of a stack, each emitting from its own time over its hat and carried by the
# rotation of the stack: on a time-dependent metric the rotation must be its helical symmetry.
# On a metric with no helical symmetry at all, frames held in place (two or more) emit each
# photon at its own event (`EventSource`, chosen by `mc_run_timed`).
function timed_matter(fs::FrameStack, st, ta, tb)
    if !(fs.omega == 0.0 && fs.n > 1 && !helical_symmetry(st))
        fs.omega == 0.0 ? stationary_emission(st) :
        helical_symmetry(st, fs.omega) ||
            throw(ArgumentError("mc_spectrum: a frame stack carried at ω = $(fs.omega) needs a " *
                                "spacetime with that helical symmetry; " *
                                "$(nameof(typeof(st))) has none"))
    end
    ts = [frame_time(fs, k) - fs.t_shift for k in 1:fs.n]
    frames = [frame_blocks(fs, k) for k in 1:fs.n]
    fs.n == 1 &&           # one frame emits over any window, carried from its own time
        return [EmissionNode(frames[1], TimeProfile([ta, tb], [1.0, 1.0]), ts[1])], fs, fs.omega
    return series_nodes(ts, frames, ta, tb, (a, b) -> false; anchored = true), fs, fs.omega
end
function timed_matter(m, st, ta, tb)
    stationary_emission(st)
    return uniform_node(m, ta, tb), m, 0.0
end

stationary_emission(st) =
    helical_symmetry(st, 0.0) ||
    throw(ArgumentError("mc_spectrum: a snapshot emits over a window only on a stationary " *
                        "spacetime; on $(nameof(typeof(st))) carry it with Corotating"))

# The node shares W_m = ∫φ_m dt/(t_b − t_a).
node_shares(nodes, ta, tb) = [profile_area(n.profile) / (tb - ta) for n in nodes]

# The rule and bins of the time-resolved accumulator of a run whose emitting points lie
# within `R` of the origin, and its interior.
function timed_bins(rows, o, ta, tb, R)
    t = o.time
    inner = t.sectors > 0 ? SectorBands(rows, t.sectors) : rows
    lo, hi, n = t.bins isa Integer ? (ta - R, tb + R, Int(t.bins)) : t.bins
    rule = TimedBands(inner, lo, hi, n)
    bins = ntuple(_ -> ExactBins(rule; nu_min = o.grid.nu_min, nu_max = o.grid.nu_max,
                                 n_ebins = t.ebins), o.K + 1)
    return rule, bins, (ta + R, tb - R)
end

timed_record(bins, matter, st) =
    TimedRecord(bins, ExactScalars(2), Float64(min(last_frame(matter), last_slice(st))))

# The time-resolved result of the record `rec`.
function timed_result(result, rec::TimedRecord, rule, window, interior, shares, omega)
    total = similar(rec.timed[1])
    foreach(b -> merge!(total, b), rec.timed)
    tm = scalar(rec.clock, 2)
    return TimedSpectrumResult(result, window, rule, [Spectrum(b) for b in rec.timed],
                               Spectrum(total), interior, shares, omega,
                               tm > 0.0 ? scalar(rec.clock, 1) / tm : 0.0)
end

# The last frame time of the matter of a run (Inf without a series).
last_frame(::Any) = Inf
last_frame(s::VariableSphere) = s.times[end]
last_frame(fs::FrameSequence) = fs.times[end]
last_frame(sp::SeriesProvider) = sp.series.times[end]
last_frame(fs::FrameStack) = frame_time(fs, fs.n) - fs.t_shift

# ---------------------------------------------------------------- the runs

mc_run_timed(::Nothing, plan, exec, bands, o, t0) = mc_run(nothing, plan, exec, bands, o, t0)

# A sphere (uniform or variable) over the window.
function mc_run_timed(matter::Union{UniformSphere,VariableSphere}, plan, exec, bands, o, t0)
    st = plan.spacetime
    st isa Minkowski ||
        throw(ArgumentError("the Monte Carlo sphere runs on Minkowski(); the plan's spacetime " *
                            "is $(nameof(typeof(st)))"))
    ta, tb = run_window(o)
    nodes, provider, omega = timed_matter(matter, st, ta, tb)
    W = node_shares(nodes, ta, tb)
    g = o.gen
    zs = [scaled_zone(table_zone(n.matter), W[m]) for (m, n) in enumerate(nodes)]
    wgt = build_weight_table(zs, g.Ns / (log(g.nu_max) - log(g.nu_min)); n_esamp = g.n_esamp,
                             nu_min = g.nu_min, nu_max = g.nu_max)
    counts = [zone_nsuperphotons(z, wgt) for z in zs]
    N_m = sphere_counts(first.(counts), o.seed)
    first_items = [0; cumsum(N_m)]
    sources = [SphereSource(n.matter, GenerationTables(wgt, counts[m][2])) for (m, n) in enumerate(nodes)]
    src = TimedSource(sources, first_items, [n.profile for n in nodes], omega)
    medium = o.medium === nothing ? provider : o.medium
    R = max(Float64(nodes[1].matter.R), Float64(domain_radius(medium)))
    ctl = FlatMarch(R, o.step_frac, SPHERE_MAX_STEPS)
    transport = MCTransport(ctl, st, medium, mc_opacities(plan.params, o, mean_weight(wgt)),
                            Val(o.absorption), Val(o.scattering))
    rows = bands === nothing ? PolarBands(6) : bands
    orders = ntuple(_ -> ExactBins(rows; o.grid...), o.K + 1)
    rule, timed, interior = timed_bins(rows, o, ta, tb, R)
    trec = timed_record(timed, matter, st)
    ledger = mc_ledger()
    N = first_items[end]
    layout = widest_layout(max(1, N), o.K)
    nchildren = run_layers(exec, src, transport, o.seed, N, orders, ledger, layout,
                           o.scattering, o.capacity; timed = trec)
    rec = run_record(exec, plan, t0)
    result = sphere_result(orders, o, N, nchildren, ledger, plan, rec)
    return timed_result(result, trec, rule, (ta, tb), interior, W, omega)
end

# The zone `z` with its volume scaled by the share `W` (the identity at W = 1).
scaled_zone(z::EmissionZone, W) =
    EmissionZone{Float64}(z.n_e, z.theta_e, z.B, z.dV * W, z.x, z.ucon, z.ucov, z.bcon, z.gcov, z.gcon)
scaled_site(z::ZoneSite, W) = ZoneSite{Float64}(z.x, z.dV * W, z.n_e, z.theta_e, z.B)

# Primaries of the sphere nodes: one node keeps the stationary sphere's rounding,
# max(1, round(N)); several are rounded stochastically in node order on `MCRng(seed, 0)`.
function sphere_counts(Ns::AbstractVector{Float64}, seed::UInt32)
    length(Ns) == 1 && return [max(1, round(Int, Ns[1]))]
    return diff(zone_offsets(Ns, seed))
end

# A plasma grid over the window: a snapshot, a co-rotating plasma or a series of frames.
function mc_run_timed(matter, plan, exec, bands, o, t0)
    rows = bands === nothing ? InclinationBands((17.0, 50.0, 90.0), 10.0) : bands
    rows isa InclinationBands ||
        throw(ArgumentError("the spectrum of a plasma grid is binned by InclinationBands; " *
                            "got $(nameof(typeof(rows)))"))
    st = plan.spacetime
    ta, tb = run_window(o)
    nodes, provider, omega = timed_matter(matter, st, ta, tb)
    W = node_shares(nodes, ta, tb)
    per = [emission_sites(zone_blocks(n.matter), st, plan.params; seed = o.seed, t0 = n.t0,
                          o.zones...) for n in nodes]
    sites = reduce(vcat, [scaled_site.(s, W[m]) for (m, s) in enumerate(per)])
    isempty(sites) &&
        throw(ArgumentError("mc_spectrum: no emitting zones (check the matter and the cuts)"))
    g = o.gen
    wgt, Nsz, dnmax = zone_tables(sites, g.Ns / (log(g.nu_max) - log(g.nu_min));
                                  n_esamp = g.n_esamp, nu_min = g.nu_min, nu_max = g.nu_max)
    offset = zone_offsets(Nsz, o.seed)
    zfirst = [0; cumsum(length.(per))]
    sources = [ZoneSource(sites[zfirst[m] + 1:zfirst[m + 1]], n.matter, st, plan.params,
                          wgt, EMISSIVITY_TABLE, dnmax[zfirst[m] + 1:zfirst[m + 1]],
                          offset[zfirst[m] + 1:zfirst[m + 1] + 1]) for (m, n) in enumerate(nodes)]
    profiles = [n.profile for n in nodes]
    src = helical_symmetry(st, omega) ? TimedSource(sources, offset[zfirst .+ 1], profiles, omega) :
          EventSource(sources, offset[zfirst .+ 1], profiles)      # no map carries a photon
    R_em = max(o.march.R_em, maximum(z -> 1.2 * proxy_distance(st, z.x)[2], sites))
    check_stop_sphere(st, o, R_em)
    ctl = CurvedMarch(st; C = o.march.C, escape = o.march.escape, R_em = R_em)
    transport = MCTransport(ctl, st, provider, mc_opacities(plan.params, o, mean_weight(wgt)),
                            Val(o.absorption), Val(o.scattering))
    orders = ntuple(_ -> ExactBins(rows; o.grid...), o.K + 1)
    rule, timed, interior = timed_bins(rows, o, ta, tb, R_em)
    trec = timed_record(timed, matter, st)
    ledger = mc_ledger()
    N = offset[end]
    layout = widest_layout(max(1, N), o.K)
    nchildren = run_layers(exec, src, transport, o.seed, N, orders, ledger, layout,
                           o.scattering, o.capacity; timed = trec)
    rec = run_record(exec, plan, t0)
    result = grid_result(orders, rows, o, length(sites), N, ledger, nchildren, plan, rec)
    return timed_result(result, trec, rule, (ta, tb), interior, W, omega)
end
