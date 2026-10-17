# March controls: how a march advances the ray state and when it stops. A fixed-control
# march steps the geodesic itself with RK4 under the one termination rule of
# `Spacetime.march_control`; a stored path replays a traced trajectory camera → source.
# The march runs the geodesic (and, polarized, the transported camera frame) in the
# control's lane and the transfer in Float64 on the program's spacetime.

"""
    SpacetimeMarch{S,T}(st, budget::MarchBudget{T})

Fixed-control march on the geodesic-lane spacetime `st` (scalars of type `T`, e.g.
`lane(KerrSchildCartesian(a), Float32)`): at every step `march_control(st, y, r_prev,
budget)` decides the stop and the affine step, and `rk4_step` advances the state.
The step limit is `budget.nmax`.
"""
struct SpacetimeMarch{S,T}
    st::S
    budget::MarchBudget{T}
end

"""
    StoredPath{S,V,D}(st, ys, dls)

March along a stored trajectory of the spacetime `st`: the states `ys` ordered source
to camera and `dls[n]` the affine step from state `n` to `n + 1` (as
`Decks.sampled_trace` and the trace products return them). The march visits the
states camera → source, pairs the sample at state `n` with the step `dls[n-1]` behind
it, and transports the polarized camera frame over each stored step with RK4. The step
limit is `length(ys)`. `isbits` after `Adapt.adapt` to a device.
"""
struct StoredPath{S,V,D}
    st::S
    ys::V
    dls::D
end
Adapt.@adapt_structure StoredPath

"""
    max_steps(ctl) -> Int

The step limit of a march under the control `ctl`: `budget.nmax` for a
[`SpacetimeMarch`](@ref), the number of stored states for a [`StoredPath`](@ref).
"""
@inline max_steps(c::SpacetimeMarch) = c.budget.nmax
@inline max_steps(c::StoredPath) = length(c.ys)

"""
    march_control(ctl, y, r_prev, it) -> MarchStep

The decision at step `it` of a march in state `y` (`r_prev` the `R` of the previous
decision): for a [`SpacetimeMarch`](@ref) the spacetime's fixed-control rule
`march_control(ctl.st, y, r_prev, ctl.budget)`; for a [`StoredPath`](@ref) a stop
once the stored states are exhausted, the stored step behind the current state
(zero at the source end) and `R = r_prev`.
"""
@inline march_control(c::SpacetimeMarch{S,T}, y::SVector{8,T}, r_prev::T, it) where {S,T} =
    march_control(c.st, y, r_prev, c.budget)
@inline function march_control(c::StoredPath, y, r_prev, it)
    n = length(c.ys) + 1 - it                            # stored index of the state y
    return MarchStep(n < 1, n > 1 ? c.dls[n-1] : zero(eltype(c.dls)), r_prev)
end

"""
    next_state(ctl, y_rk4, it) -> RayState

The state after step `it`: the RK4 result `y_rk4` on a fixed-control march, the next
stored state on a stored path (where the RK4 result only carries the transported
frame), or `y_rk4` past the source end.
"""
@inline next_state(::SpacetimeMarch, y_rk4, it) = y_rk4
@inline function next_state(c::StoredPath, y_rk4, it)
    n = length(c.ys) + 1 - it
    return n > 1 ? c.ys[n-1] : y_rk4
end

"""
    geodesic_step(ctl, y, h, it) -> RayState

Advance the ray state over step `it` of affine length `h`: `rk4_step` on the control's
spacetime for a [`SpacetimeMarch`](@ref), the next stored state for a
[`StoredPath`](@ref).
"""
@inline geodesic_step(c::SpacetimeMarch, y, h, it) = rk4_step(geodesic_spacetime(c), y, h)
@inline geodesic_step(c::StoredPath, y, h, it) = next_state(c, y, it)

"""
    geodesic_spacetime(ctl)

The spacetime the geodesic and the polarized camera frame are advanced in, on the
control's lane.
"""
@inline geodesic_spacetime(c::Union{SpacetimeMarch,StoredPath}) = c.st

"""
    transfer_spacetime(prog)

The spacetime of the transfer lane of the program `prog` (its `st` field, scalars
`Float64`): the metric of the plasma decode, of the photon wave vector and of the
tetrads, evaluated at the `Float64` image of the geodesic-lane state.
"""
@inline transfer_spacetime(prog) = prog.st

"""
    renorm_e1(ctl) -> Bool

Whether the tetrads of a march renormalise `e_(1)`: false on single Kerr-Schild,
where `|det g| = 1` makes the metric cross product unit, true on every other
spacetime.
"""
@inline renorm_e1(c::Union{SpacetimeMarch,StoredPath}) = !(c.st isa KerrSchildCartesian)
