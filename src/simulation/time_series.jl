# Simulation — time_series.jl
#
# Slow light: time-dependent GRMHD frames along the geodesic (Bronzwaer et al.
# 2018, A&A 613 A2; window/blend conventions cross-checked against ipole). Both
# bracketing dumps are sampled spatially, the primitives blended linearly in
# coordinate time, and derived quantities reconstructed downstream. The backward
# trace starts at X⁰ = 0 at the camera with X⁰ increasing along the ray, so the
# emission time is t_emit = t_obs - X⁰, clamped at the dump-window ends.
# Δt_dump → ∞ gives w == 0 exactly, recovering fast light bit-for-bit.

"""
    SnapshotSeries

Time-ordered GRMHD snapshot sequence for slow light: ascending frame coordinate
times plus either on-disk `.athdf` paths (lazily loaded through a `maxslots`-slot
LRU cache) or pre-materialised in-memory frames (from `synthetic_series`).
`t_obs` is the observation epoch at the camera (X⁰ = 0): a ray sample at X⁰ was
emitted at `t_obs - X⁰`. `bracketing_pair(series, t)` maps a coordinate time to
`(ia, w)`; `sample_primitives_t(series, x)` blends the bracketing dumps; `lk`
serialises load/LRU/evict on a path-backed series.
"""
struct SnapshotSeries
    times::Vector{Float64}
    paths::Vector{String}
    frames::Dict{Int,AthenaKSnapshot}
    order::Vector{Int}
    maxslots::Int
    t_obs::Float64
    lk::ReentrantLock
end

"""
    SnapshotSeries(paths::Vector{String}, times::Vector{Float64};
                   maxslots=2, t_obs=times[end])

Path-based series: `paths[i]` is loaded lazily on first access and cached,
evicting the least-recently-used loadable frame once more than `maxslots` are
resident. `times` must be ascending and the same length as `paths`. `t_obs`
(default: the last dump time) must lie inside the dump window so ray emission
times t_obs - X⁰ can actually bracket dumps.
"""
function SnapshotSeries(paths::Vector{String}, times::Vector{Float64};
                        maxslots::Integer = 2, t_obs::Real = times[end])
    length(paths) == length(times) ||
        error("paths ($(length(paths))) and times ($(length(times))) length mismatch")
    issorted(times) || error("SnapshotSeries times must be ascending")
    maxslots >= 2 || error("maxslots must be >= 2 for a bracketing window")
    times[1] <= t_obs <= times[end] ||
        error("t_obs = $t_obs outside the dump window [$(times[1]), $(times[end])]: " *
              "every sample would clamp to one window end (silent fast light)")
    return SnapshotSeries(collect(Float64.(times)), copy(paths),
                          Dict{Int,AthenaKSnapshot}(), Int[], Int(maxslots),
                          Float64(t_obs), ReentrantLock())
end

Base.length(series::SnapshotSeries) = length(series.times)

"""
    scale_density(snap::AthenaKSnapshot, factor) -> AthenaKSnapshot

Return a snapshot sharing `snap`'s geometry but with the density primitive (axis 1)
scaled by `factor` (a global density modulation; `factor == 1` returns `snap`
unchanged so an identity evolve is object-identical). Only the primitive block is
copied; the per-block geometry vectors are shared.
"""
function scale_density(snap::AthenaKSnapshot, factor::Real)
    factor == 1.0 && return snap
    data = copy(snap.data)
    @inbounds @views data[:, :, :, :, 1] .*= Float32(factor)
    # Geometry is identical, so the leaf-block index grid is identical: share the
    # parent's cache so a density-scaled frame reuses one table.
    return AthenaKSnapshot(data, snap.ext_lo, snap.ext_hi, snap.x_left, snap.dx,
                           snap.levels, snap.logical_locations, snap.variable_names,
                           snap.ni, snap.nj, snap.nk, snap.nmb, snap.leafcache)
end

"""
    synthetic_series(snap; times, evolve=(s,t)->s) -> SnapshotSeries

Build a controlled test series from a single snapshot. `evolve(snap, t)` returns
the frame at time `t`; the default identity gives object-identical frames and
hence the fast-light limit. E.g. a global density modulation is
`evolve = (s,t) -> scale_density(s, 1 + eps*sin(omega*t))`.
"""
function synthetic_series(snap::AthenaKSnapshot; times::AbstractVector{<:Real},
                          evolve = (s, t) -> s, t_obs::Real = last(times))
    ts = collect(Float64.(times))
    issorted(ts) || error("synthetic_series times must be ascending")
    length(ts) >= 1 || error("synthetic_series needs >= 1 time")
    ts[1] <= t_obs <= ts[end] ||
        error("t_obs = $t_obs outside the dump window [$(ts[1]), $(ts[end])]: " *
              "every sample would clamp to one window end (silent fast light)")
    frames = Dict{Int,AthenaKSnapshot}()
    order = Int[]
    for (i, t) in enumerate(ts)
        frames[i] = evolve(snap, t)
        push!(order, i)
    end
    return SnapshotSeries(ts, String[], frames, order, max(2, length(ts)),
                          Float64(t_obs), ReentrantLock())
end

"""
    bracketing_pair(series, t) -> (ia, w)

Locate the dump bracket for coordinate time `t`: returns the lower frame index
`ia ∈ 1:N-1` and the blend weight `w = (t - t_a)/(t_b - t_a) ∈ [0,1]` on the UPPER
frame `ia+1` (so the blend is `(1-w)*frame[ia] + w*frame[ia+1]`). Clamps at the
ends — `t <= times[1] -> (1, 0)`, `t >= times[end] -> (N-1, 1)` — matching ipole's
window clamp. A single-frame series returns `(1, 0.0)`.
"""
@inline function bracketing_pair(series::SnapshotSeries, t::Real)
    ts = series.times
    N = length(ts)
    N >= 1 || error("empty SnapshotSeries")
    N == 1 && return (1, 0.0)
    t <= ts[1]   && return (1, 0.0)
    t >= ts[N]   && return (N - 1, 1.0)
    ia = clamp(searchsortedlast(ts, t), 1, N - 1)   # last frame with ts[ia] <= t
    w = (t - ts[ia]) / (ts[ia + 1] - ts[ia])
    return (ia, w)
end

_touch!(order::Vector{Int}, ia::Int) = (filter!(!=(ia), order); push!(order, ia))

# Give a freshly-loaded frame the leaf table of an already-resident, same-mesh frame
# so a series builds the O(1) index grid once. A frame whose mesh differs (e.g.
# across an AMR regrid) falls back to building its own table lazily.
function _adopt_shared_leaf!(series::SnapshotSeries, snap::AthenaKSnapshot)
    for (_, f) in series.frames
        f === snap && continue
        _mesh_matches(f, snap) || continue
        t = leaf_table(f)
        t === nothing && continue
        c = snap.leafcache
        c.tbl = t
        @atomic :release c.state = Int8(1)   # publish the shared table lock-free
        return
    end
    return
end

"""
    get_frame(series, ia) -> AthenaKSnapshot

Return frame `ia`, loading it from `series.paths[ia]` (and caching it, evicting the
least-recently-used loadable frame past `maxslots`) if it is not already resident.
Materialised (synthetic) frames are always resident and never evicted, so that hot
path is read-only (thread-safe, allocation-free); loads/LRU touches/evictions on a
path-backed series are serialised by `series.lk`.
"""
function get_frame(series::SnapshotSeries, ia::Integer)
    ia = Int(ia)
    (1 <= ia <= length(series.times)) ||
        error("frame index $ia out of range 1:$(length(series.times))")
    frames = series.frames
    if isempty(series.paths)
        # fully materialised: nothing is ever loaded or evicted, so skip the LRU
        # bookkeeping — per-sample calls must not mutate shared series state
        f = get(frames, ia, nothing)
        f === nothing &&
            error("frame $ia is not materialised and the series has no paths to load")
        return f
    end
    lock(series.lk)
    try
        if haskey(frames, ia)
            _touch!(series.order, ia)
            return frames[ia]
        end
        snap = AthenaKSnapshot(series.paths[ia])
        _adopt_shared_leaf!(series, snap)   # reuse one leaf table across same-mesh frames
        frames[ia] = snap
        _touch!(series.order, ia)
        while length(frames) > series.maxslots
            victim = 0
            for idx in series.order
                if idx != ia && haskey(frames, idx)
                    victim = idx
                    break
                end
            end
            victim == 0 && break
            delete!(frames, victim)
            filter!(!=(victim), series.order)
        end
        return snap
    finally
        unlock(series.lk)
    end
end

"""
    sample_primitives_t(series, x::SVector{4}) -> NamedTuple

Slow-light primitive sample at CKS position+time `x = (t, X, Y, Z)`: pick the dump
bracket at the emission time `series.t_obs - x[1]` (X⁰ = x[1] is the light-travel
time from the camera), spatially sample both dumps with `sample_primitives`, and
blend the primitives linearly in time. Returns the same NamedTuple shape as
`sample_primitives`. `w == 0` / `w == 1` (the window-end clamps) and
object-identical dumps short-circuit to a single bit-identical spatial sample.
"""
@inline function sample_primitives_t(series::SnapshotSeries, x::SVector{4})
    ia, w = bracketing_pair(series, series.t_obs - x[1])   # emission time t_obs - X⁰
    A = get_frame(series, ia)
    (w == 0.0) && return sample_primitives(A, x)
    B = get_frame(series, ia + 1)
    (A === B || w == 1.0) && return sample_primitives(B, x)
    sA = sample_primitives(A, x)
    sB = sample_primitives(B, x)
    wa = 1.0 - w
    return (rho  = wa * sA.rho  + w * sB.rho,
            eint = wa * sA.eint + w * sB.eint,
            uu   = wa .* sA.uu .+ w .* sB.uu,
            bb   = wa .* sA.bb .+ w .* sB.bb,
            inside = sA.inside || sB.inside)
end
