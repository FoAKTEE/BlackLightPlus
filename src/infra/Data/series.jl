# Slow light: a time-ordered series of snapshots, the two-frame window that a device
# samples, and the host provider that brackets any emission time through the series.
# The ray starts at X⁰ = 0 on the camera and X⁰ decreases along the backward march, so a
# sample at X⁰ was emitted at t_obs + X⁰. Both bracketing frames are sampled in space and
# the two samples blended linearly in time by `blend`. After Bronzwaer et al. (2018),
# A&A 613, A2.

"""
    SnapshotSeries
    SnapshotSeries(paths, times; maxslots = 2, t_obs = times[end])

Snapshots at ascending coordinate times `times` (code units), either loaded from the
`.athdf` files `paths` on first use and kept in a least-recently-used cache of
`maxslots ≥ 2` frames, or held in memory (built by [`synthetic_series`](@ref)). `t_obs`
is the observation epoch at the camera and must lie in `[times[1], times[end]]`. A frame
loaded while a frame of the same mesh is resident shares that frame's locate strategy and
geometry arrays. Loads, cache updates and evictions of a path-backed series hold the
series lock; a held-in-memory series is read only. Fields: `times`, `paths`, `frames`
(index → snapshot), `order` (least recent first), `maxslots`, `t_obs`, `lk`.
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

`snap` with the density (feature 1) multiplied by `Float32(factor)` in a copy of the
fluid array; the geometry, the locate strategy and the metadata are shared.
`factor == 1` returns `snap` itself.
"""
function scale_density(snap::AthenaKSnapshot, factor::Real)
    factor == 1.0 && return snap
    b = snap.blocks
    data = copy(b.data)
    @inbounds @views data[1, :, :, :, :] .*= Float32(factor)
    blocks = FluidBlocks(data, b.x_left, b.dx, b.lo, b.hi, b.ni, b.nj, b.nk, b.nmb, b.locate)
    return AthenaKSnapshot(blocks, snap.levels, snap.logical_locations, snap.names, snap.time)
end

"""
    synthetic_series(snap; times, evolve = (s, t) -> s, t_obs = last(times)) -> SnapshotSeries

A held-in-memory series whose frame at `times[i]` is `evolve(snap, times[i])`. The
default `evolve` gives the same object at every time, the fast-light limit; a global
density modulation is `evolve = (s, t) -> scale_density(s, 1 + ε sin(ω t))`. `times`
must be ascending and non-empty and `t_obs` inside them.
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

The frame bracket of coordinate time `t`: the lower frame `ia ∈ 1:N-1` (the last with
`times[ia] ≤ t`) and the weight `w = (t - t_a)/(t_b - t_a) ∈ [0, 1]` of the upper frame
`ia + 1`. Clamped at the ends: `t ≤ times[1]` gives `(1, 0.0)`, `t ≥ times[N]` gives
`(N - 1, 1.0)`; a one-frame series gives `(1, 0.0)`.
"""
@inline function bracketing_pair(series::SnapshotSeries, t::Real)
    ts = series.times
    N = length(ts)
    N >= 1 || error("empty SnapshotSeries")
    N == 1 && return (1, 0.0)
    t <= ts[1] && return (1, 0.0)
    t >= ts[N] && return (N - 1, 1.0)
    ia = clamp(searchsortedlast(ts, t), 1, N - 1)
    w = (t - ts[ia]) / (ts[ia + 1] - ts[ia])
    return (ia, w)
end

_touch!(order::Vector{Int}, ia::Int) = (filter!(!=(ia), order); push!(order, ia))

# `snap` sharing the mesh of a resident frame with the same mesh, if any.
function _adopt_mesh(series::SnapshotSeries, snap::AthenaKSnapshot)
    for (_, f) in series.frames
        same_mesh(f, snap) && return share_mesh(snap, f)
    end
    return snap
end

"""
    get_frame(series, ia) -> AthenaKSnapshot

Frame `ia`. A held-in-memory series returns it without changing any state. A
path-backed series loads `paths[ia]` with `AthenaKSnapshot(path)` if it is not resident,
marks it most recently used, and evicts least recently used frames other than `ia` while
more than `maxslots` are resident, all under the series lock.
"""
function get_frame(series::SnapshotSeries, ia::Integer)
    ia = Int(ia)
    (1 <= ia <= length(series.times)) ||
        error("frame index $ia out of range 1:$(length(series.times))")
    frames = series.frames
    if isempty(series.paths)
        f = get(frames, ia, nothing)
        f === nothing &&
            error("frame $ia is not materialised and the series has no paths to load")
        return f
    end
    return @lock series.lk _load_frame!(series, ia)
end

function _load_frame!(series::SnapshotSeries, ia::Int)
    frames = series.frames
    if haskey(frames, ia)
        _touch!(series.order, ia)
        return frames[ia]
    end
    snap = _adopt_mesh(series, AthenaKSnapshot(series.paths[ia]))
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
end

"""
    blend(sa::PlasmaPrimitives, sb::PlasmaPrimitives, w) -> PlasmaPrimitives

The linear time blend `(1 - w) sa + w sb` of two samples, field by field (`1 - w` is
formed once), with `inside = sa.inside || sb.inside`.
"""
@inline function blend(sa::PlasmaPrimitives{T}, sb::PlasmaPrimitives{T}, w) where {T}
    wa = one(w) - w
    return PlasmaPrimitives{T}(wa * sa.rho + w * sb.rho,
                               wa .* sa.uu .+ w .* sb.uu,
                               wa * sa.eint + w * sb.eint,
                               wa .* sa.bb .+ w .* sb.bb,
                               sa.inside || sb.inside)
end

"""
    FrameWindow{P} <: GridProvider
    FrameWindow(a::P, b::P, t_a, t_b; t_obs = <upper end>)

Two frames of one provider type `P` on the same mesh, `a` at coordinate time `t_a` and
`b` at `t_b`, blended at the emission time `t_obs + X⁰` of each sample. Fields: `a`, `b`,
`t_a`, `inv_dt = 1/(t_b - t_a)` (0 when `t_b` is not finite or not after `t_a`, the
fast-light limit) and `t_obs`, the observation epoch at the camera, by default the upper
end `t_a + 1/inv_dt` (`t_a` when `inv_dt = 0`). `isbits` after `Adapt.adapt` when `P` is.

`primitives(win, x)` forms `w = clamp((t_obs - t_a + X⁰) inv_dt, 0, 1)` with
`X⁰ = x[1]` and returns frame `b`'s sample when `w == 1`, frame `a`'s when `w == 0`, and
[`blend`](@ref) of the two samples otherwise.
"""
struct FrameWindow{P} <: GridProvider
    a::P
    b::P
    t_a::Float64
    inv_dt::Float64
    t_obs::Float64
end
Adapt.@adapt_structure FrameWindow

_win_inv_dt(t_a, t_b) = (isfinite(t_b) && t_b > t_a) ? 1.0 / (t_b - t_a) : 0.0
_win_t_obs(t_a, inv_dt) = inv_dt > 0.0 ? t_a + 1.0 / inv_dt : t_a

function FrameWindow(a::P, b::P, t_a::Real, t_b::Real;
                     t_obs::Union{Nothing,Real} = nothing) where {P<:GridProvider}
    same_mesh(a, b) ||
        error("FrameWindow: the two frames must share the mesh (same nodes or blocks)")
    ta = Float64(t_a)
    inv_dt = _win_inv_dt(ta, Float64(t_b))
    epoch = t_obs === nothing ? _win_t_obs(ta, inv_dt) : Float64(t_obs)
    return FrameWindow{P}(a, b, ta, inv_dt, epoch)
end

FrameWindow(a::GridProvider, b::GridProvider, t_a::Real, t_b::Real; kw...) =
    error("FrameWindow: the two frames must be of one provider type on the same mesh; " *
          "got $(typeof(a)) and $(typeof(b))")

@inline function primitives(win::FrameWindow, x::SVector{4})
    w = clamp((win.t_obs - win.t_a + Float64(x[1])) * win.inv_dt, 0.0, 1.0)
    w == 1.0 && return primitives(win.b, x)
    sa = primitives(win.a, x)
    w == 0.0 && return sa
    sb = primitives(win.b, x)
    return blend(sa, sb, w)
end

"""
    domain_radius(win::FrameWindow) -> Float64

The larger domain radius of the two frames.
"""
domain_radius(win::FrameWindow) = max(domain_radius(win.a), domain_radius(win.b))

# Frames ia and min(ia + 1, N) of `series` and their times (t_b = Inf at the series end).
function _bracket(series::SnapshotSeries, ia::Integer)
    N = length(series)
    ia = Int(ia)
    ib = min(ia + 1, N)
    t_b = ib > ia ? series.times[ib] : Inf
    return ia, ib, series.times[ia], t_b
end

"""
    window(series, ia; t_obs = <upper end>) -> FrameWindow{<:FluidBlocks}
    window(series, ia, box_half, n; t_obs = <upper end>) -> FrameWindow{<:UniformGrid}

The window over frames `ia` and `ia + 1` of `series` (the frame `ia` twice, with
`inv_dt = 0`, at the series end): their fluid blocks, or their [`resample`](@ref)s onto
`n³` nodes of `[-box_half, box_half]³` (one resample when the two frames are the same
object). `t_obs` as in [`FrameWindow`](@ref). The frames must share the mesh.
"""
function window(series::SnapshotSeries, ia::Integer; t_obs::Union{Nothing,Real} = nothing)
    ia, ib, t_a, t_b = _bracket(series, ia)
    A = get_frame(series, ia)
    B = get_frame(series, ib)
    return FrameWindow(A.blocks, B.blocks, t_a, t_b; t_obs = t_obs)
end

function window(series::SnapshotSeries, ia::Integer, box_half::Real, n::Integer;
                t_obs::Union{Nothing,Real} = nothing)
    ia, ib, t_a, t_b = _bracket(series, ia)
    A = get_frame(series, ia)
    B = get_frame(series, ib)
    ga = resample(A, box_half, n)
    gb = A === B ? ga : resample(B, box_half, n)
    return FrameWindow(ga, gb, t_a, t_b; t_obs = t_obs)
end

"""
    advance(win::FrameWindow, series, ia; t_obs = <upper end>) -> FrameWindow

The window over frames `ia` and `ia + 1`, where `win.b` is frame `ia`: `win.b` becomes
the lower frame unchanged (same object) and only the upper frame is built, as the frame's
blocks or as a resample with the geometry of `win.b`. At the series end the window
holds `win.b` twice with `inv_dt = 0`. The new frame must share the mesh.
"""
function advance(win::FrameWindow, series::SnapshotSeries, ia::Integer;
                 t_obs::Union{Nothing,Real} = nothing)
    ia, ib, t_a, t_b = _bracket(series, ia)
    a = win.b
    b = ib > ia ? _frame_like(a, get_frame(series, ib)) : a
    return FrameWindow(a, b, t_a, t_b; t_obs = t_obs)
end

_frame_like(::FluidBlocks, snap::AthenaKSnapshot) = snap.blocks
_frame_like(g::UniformGrid, snap::AthenaKSnapshot) = resample(snap, g.box_half, g.n)

"""
    SeriesProvider <: GridProvider

A host provider that samples `series` at any emission time: `primitives(p, x)` finds the
bracket of `t_obs + X⁰` with [`bracketing_pair`](@ref), returns the lower frame's sample
when `w == 0`, the upper frame's when the two frames are the same object or `w == 1`,
and [`blend`](@ref) of the two samples otherwise. Frames are fetched with
[`get_frame`](@ref), so a path-backed series may load and evict under its lock; not for
devices.
"""
struct SeriesProvider <: GridProvider
    series::SnapshotSeries
end

function primitives(p::SeriesProvider, x::SVector{4})
    s = p.series
    ia, w = bracketing_pair(s, s.t_obs + x[1])
    A = get_frame(s, ia)
    w == 0.0 && return primitives(A, x)
    B = get_frame(s, ia + 1)
    (A === B || w == 1.0) && return primitives(B, x)
    return blend(primitives(A, x), primitives(B, x), w)
end

"""
    domain_radius(p::SeriesProvider) -> Float64

The largest domain radius of the frames of a held-in-memory series; `Inf` for a
path-backed series, whose frames are not all resident.
"""
function domain_radius(p::SeriesProvider)
    s = p.series
    isempty(s.paths) || return Inf
    return maximum(domain_radius(f) for f in values(s.frames))
end
