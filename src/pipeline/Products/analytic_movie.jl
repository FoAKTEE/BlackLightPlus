# Movies and light curves of time-dependent equatorial sources in closed form: the
# landings of every pixel and order once (amplitude, position, emission time), then each
# frame as one cheap pass over them at its observation time.

"""
    SpotProgram(gen, theta_o, a, ro, spot, bands, slow)

The time-independent part of a movie of `spot` (an `EquatorialSpot`): per screen point
`(α, β) = image_coords(gen, item)` and order `n` of `bands` the equatorial landing and
[`spot_amplitude`](@ref), written quantity-major into the `4K` outputs of
`analytic_outputs`-like vectors: the amplitudes `A` of bands `1:K`, then `x`, `y` and the
emission time `t_s` (zero for every landing when `slow` is false: fast light).
"""
struct SpotProgram{G,S,B} <: Function
    gen::G
    theta_o::Float64
    a::Float64
    ro::Float64
    spot::S
    bands::B
    slow::Bool
end
Adapt.@adapt_structure SpotProgram

@inline function spot_band(prep, spot, n, slow)
    A, x, y, ts = spot_amplitude(spot, band_landing(prep, n))
    return (Float64(A), Float64(x), Float64(y), slow ? Float64(ts) : 0.0)
end

function (p::SpotProgram)(item, i, outs::Vararg{Any,N}) where {N}
    N == 4 * length(p.bands) ||
        throw(ArgumentError("SpotProgram: $N outputs for $(length(p.bands)) bands"))
    α, β = image_coords(p.gen, item)
    prep = pixel_prep(α, β, p.theta_o, p.a; ro = p.ro)
    maps = map(n -> spot_band(prep, p.spot, n, p.slow), p.bands)
    store_all!(outs, i, (map(m -> m[1], maps)..., map(m -> m[2], maps)...,
                         map(m -> m[3], maps)..., map(m -> m[4], maps)...))
    return nothing
end

stack_bytes(::SpotProgram) = 32 * 1024

"""
    FrameProgram(A, x, y, ts, profile, t_obs)

One frame of a movie: item `i` writes `Σ_b A[b][i] profile(ts[b][i] + t_obs, x[b][i],
y[b][i])` over the bands (tuples of vectors from a [`SpotProgram`](@ref)), the band sum in
band order, terms with `A = 0` skipped.
"""
struct FrameProgram{V,F} <: Function
    A::V
    x::V
    y::V
    ts::V
    profile::F
    t_obs::Float64
end
Adapt.@adapt_structure FrameProgram

@inline _frame_sum(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, f, t, i, acc) = acc
@inline function _frame_sum(A::Tuple, x::Tuple, y::Tuple, ts::Tuple, f, t, i, acc)
    a = @inbounds first(A)[i]
    if a > 0
        acc += a * f(@inbounds(first(ts)[i]) + t, @inbounds(first(x)[i]), @inbounds(first(y)[i]))
    end
    return _frame_sum(Base.tail(A), Base.tail(x), Base.tail(y), Base.tail(ts), f, t, i, acc)
end

function (p::FrameProgram)(item, i, out)
    @inbounds out[i] = _frame_sum(p.A, p.x, p.y, p.ts, p.profile, p.t_obs, item, 0.0)
    return nothing
end

"""
    analytic_movie(plan, spot::EquatorialSpot; times, bands = (0, 1, 2),
                   ro = plan.camera.d, slow = true, backend = nothing) -> NamedTuple

Frames of the time-dependent equatorial source `spot` on the plan's pixels at the
observation times `times`: the closed-form landings of every pixel and order once
([`SpotProgram`](@ref); `ro = Inf` for the observer at infinity, whose times are retarded),
then one [`FrameProgram`](@ref) per time. `slow = false` paints every landing at its
observation time (fast light). Returns `(frames, lightcurve, times, A, x, y, ts, bands,
plan, run)`: `frames[:, :, k]` (rows β) the image at `times[k]`, the light curve
`lightcurve[k] = Σ frames[:, :, k] × pixel area` (M²), and the per-band landing maps.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function analytic_movie(plan::Plan, spot::EquatorialSpot; times, bands = (0, 1, 2),
                        ro::Real = plan.camera.d, slow::Bool = true, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    bt = map(Int, Tuple(bands))
    a = analytic_spin(plan, (spot,))
    N = npixels(plan)
    K = length(bt)
    outs = ntuple(_ -> Vector{Float64}(undef, N), 4K)
    foreach_item!(exec, SpotProgram(plan.pixels, plan.theta_o, a, Float64(ro), spot, bt, slow),
                  1:N; outs)
    A = outs[1:K]; X = outs[K+1:2K]; Y = outs[2K+1:3K]; TS = outs[3K+1:4K]
    n = plan.camera.npix
    ts = collect(Float64, times)
    frames = Array{Float64}(undef, n, n, length(ts))
    for (k, t) in enumerate(ts)
        foreach_item!(exec, FrameProgram(A, X, Y, TS, spot.profile, t), 1:N;
                      outs = (view(frames, :, :, k),))
    end
    dpix = plan.camera.fov / n
    lc = [sum(view(frames, :, :, k)) * dpix^2 for k in eachindex(ts)]
    maps(v) = [reshape(v[b], n, n) for b in 1:K]
    return (frames = frames, lightcurve = lc, times = ts, A = maps(A), x = maps(X), y = maps(Y),
            ts = maps(TS), bands = collect(bt), plan = plan, run = run_record(exec, plan, t0))
end
