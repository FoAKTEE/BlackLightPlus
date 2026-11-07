# Fixed-control integrators of the march. The step h is a function of the state alone
# (`march_control`), never of an accept/reject decision, so the step sequence of a ray is a
# continuous function of its initial state: two backends whose arithmetic differs in the
# last bit take the same number of steps and end within that rounding times the ray's
# condition number. `ClassicalRK4` is the production integrator (`rk4_step`); `Verner9` is
# the ninth-order member of Verner's 9(8) pair (J. H. Verner, Numer. Algor. 53, 383 (2010),
# the "most efficient" tableau, 15 stages for the ninth-order solution; the embedded
# eighth-order weights are not used). At a step divisor C its local error is O(C^-10) per
# step against RK4's O(C^-5), at 3.75 times the right-hand sides per step.

"""
    MarchIntegrator

The integrator of a fixed-control march: [`ClassicalRK4`](@ref) (the default) or
[`Verner9`](@ref). A singleton carried in the [`MarchBudget`](@ref), so each integrator
compiles to its own march.
"""
abstract type MarchIntegrator end

"""
    ClassicalRK4()

The classical fourth-order Runge–Kutta step in the streaming form of [`rk4_step`](@ref):
four right-hand sides per step, global error O(C^-4) in the step divisor `C`.
"""
struct ClassicalRK4 <: MarchIntegrator end

"""
    Verner9()

The ninth-order explicit Runge–Kutta step of Verner's 9(8) pair (the 2010 "most efficient"
tableau): fifteen right-hand sides per step, global error O(C^-9) in the step divisor
`C`. Its coefficients are the tableau's rationals rounded to `Float64` and converted to the
state's number type.
"""
struct Verner9 <: MarchIntegrator end

"""
    integrator_stages(integrator) -> Int

Right-hand-side evaluations per step: 4 for [`ClassicalRK4`](@ref), 15 for
[`Verner9`](@ref).
"""
integrator_stages(::ClassicalRK4) = 4
integrator_stages(::Verner9) = 15

"""
    march_integrator(name::Symbol) -> MarchIntegrator

The integrator called `name`: `:rk4` ([`ClassicalRK4`](@ref)) or `:verner9`
([`Verner9`](@ref)); `ArgumentError` for any other name.
"""
function march_integrator(name::Symbol)
    name === :rk4 && return ClassicalRK4()
    name === :verner9 && return Verner9()
    throw(ArgumentError("march_integrator: :rk4 or :verner9, got :$name"))
end

"""
    erk_step(integrator, f, y::SVector{N,T}, h::T) -> SVector{N,T}

One step of the explicit Runge–Kutta `integrator` of the autonomous system `dy/dλ = f(y)`
from `y` over `h`. `ClassicalRK4` accumulates its stages in the streaming order of
[`rk4_step`](@ref), so the two agree bit for bit on the geodesic flow.
"""
@inline function erk_step(::ClassicalRK4, f::F, y::SVector{N,T}, h::T) where {F,N,T}
    d = f(y);  acc = d;         ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + h * d
    d = f(ys); acc = acc + d
    return y + (h / 6) * acc
end

@inline function erk_increment(::Verner9, f::F, y::SVector{N,T}, h::T) where {F,N,T}
    k1 = f(y)
    k2 = f(y + h * (T(0.03462) * k1))
    k3 = f(y + h * (T(-0.03893354388572872) * k1 + T(0.13595789452450915) * k2))
    k4 = f(y + h * (T(0.03638413148954266) * k1 + T(0.109152394468628) * k3))
    k5 = f(y + h * (T(2.02576391439397) * k1 + T(-7.638023836496292) * k3 + T(6.173259922102322) * k4))
    k6 = f(y + h * (T(0.05112275589406061) * k1 + T(0.17708237945550215) * k4 + T(0.0008027762409222465) * k5))
    k7 = f(y + h * (T(0.13160063579752163) * k1 + T(-0.29572762526696367) * k4 + T(0.0878137803564295) * k5 + T(0.6213052975225275) * k6))
    k8 = f(y + h * (T(0.07166666666666667) * k1 + T(0.33055335789153195) * k6 + T(0.24277997544180138) * k7))
    k9 = f(y + h * (T(0.071806640625) * k1 + T(0.3294380283228177) * k6 + T(0.11651900292718227) * k7 + T(-0.034013671875) * k8))
    k10 = f(y + h * (T(0.04836757646340647) * k1 + T(0.039289899256761636) * k6 + T(0.10547409458903446) * k7 + T(-0.021438652846483126) * k8 + T(-0.10412291746271944) * k9))
    k11 = f(y + h * (T(-0.026645614872014788) * k1 + T(0.03333333333333333) * k6 + T(-0.16310722448724674) * k7 + T(0.033960816841277615) * k8 + T(0.15723194138146263) * k9 + T(0.21522674780318796) * k10))
    k12 = f(y + h * (T(0.036890092487086225) * k1 + T(-0.1465181576725543) * k6 + T(0.22425777681720246) * k7 + T(0.022944057170660722) * k8 + T(-0.0035850052905728935) * k9 + T(0.08669223316444385) * k10 + T(0.43838406519683376) * k11))
    k13 = f(y + h * (T(-0.48660122151133406) * k1 + T(-6.304602650282853) * k6 + T(-0.2812456182894722) * k7 + T(-2.6790192362198493) * k8 + T(0.5188156639241572) * k9 + T(1.3653531876033416) * k10 + T(5.8850910885039465) * k11 + T(2.8028087862720628) * k12))
    k14 = f(y + h * (T(0.4185367457753471) * k1 + T(6.724547581906459) * k6 + T(-0.4254442801646122) * k7 + T(3.3432791530012658) * k8 + T(0.6170816631175381) * k9 + T(-0.9299661239399327) * k10 + T(-6.099948804751011) * k11 + T(-3.002206187889399) * k12 + T(0.2553202529443446) * k13))
    k15 = f(y + h * (T(-0.7793740861228845) * k1 + T(-13.937342538107776) * k6 + T(1.2520488533793583) * k7 + T(-14.69150040801687) * k8 + T(-0.4947050585331424) * k9 + T(2.2429749091462363) * k10 + T(13.367893803828643) * k11 + T(14.396650486650687) * k12 + T(-0.79758133317768) * k13 + T(0.4409353709534278) * k14))
    return h * (T(0.014611976858423152) * k1 + T(-0.3915211862331339) * k8 + T(0.23109325002895065) * k9 + T(0.12747667699928525) * k10 + T(0.2246434176204158) * k11 + T(0.5684352689748513) * k12 + T(0.058258715572158275) * k13 + T(0.13643174034822156) * k14 + T(0.030570139830827976) * k15)
end
@inline erk_step(integ::Verner9, f::F, y::SVector{N,T}, h::T) where {F,N,T} =
    y + erk_increment(integ, f, y, h)

"""
    erk_increment(integrator, f, y, h) -> SVector

The increment `h Σ_i b_i k_i` of one step of [`erk_step`](@ref), before it is added to
`y`; [`march_step_compensated`](@ref) adds it with the carried rounding error.
"""
erk_increment

@inline function erk_increment(::ClassicalRK4, f::F, y::SVector{N,T}, h::T) where {F,N,T}
    d = f(y);  acc = d;         ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + h * d
    d = f(ys); acc = acc + d
    return (h / 6) * acc
end

"""
    march_step(integrator, st, y::RayState{T}, h::T) -> RayState{T}

One step of the geodesic flow on `st`: [`rk4_step`](@ref) for `ClassicalRK4`, the
[`erk_step`](@ref) of `geodesic_rhs` for `Verner9`.
"""
@inline march_step(::ClassicalRK4, st, y::SVector{8,T}, h::T) where {T} = rk4_step(st, y, h)
@inline march_step(integ::Verner9, st, y::SVector{8,T}, h::T) where {T} =
    erk_step(integ, z -> (@inline; geodesic_rhs(st, z)), y, h)

"""
    march_step_compensated(integrator, st, y::RayState{T}, lo::RayState{T}, h::T)
        -> (y, lo)

[`march_step`](@ref) with the state carried as the unevaluated sum `y + lo`: the step's
increment and the carried `lo` are added to `y` by the two-sum of Knuth, whose rounding
error becomes the new `lo`. The stages see `y` alone; the state update then rounds once
per ray instead of once per step, which removes the rounding of the position and
momentum that accumulates over the steps of a long march.
"""
@inline function march_step_compensated(integ::MarchIntegrator, st, y::SVector{8,T},
                                        lo::SVector{8,T}, h::T) where {T}
    Δ = erk_increment(integ, z -> (@inline; geodesic_rhs(st, z)), y, h) + lo
    s = y + Δ
    bb = s - y
    return s, (y - (s - bb)) + (Δ - bb)
end

# (y, v_1, …, v_M) as one vector and back.
@inline _pack(y::SVector{8,T}, vs::NTuple{M,SVector{4,T}}) where {T,M} =
    SVector{8 + 4M,T}(ntuple(i -> (@inline; i <= 8 ? y[i] : vs[(i - 9) ÷ 4 + 1][(i - 9) % 4 + 1]),
                             Val(8 + 4M)))
@inline _unpack(Y::SVector{L,T}, ::Val{M}) where {L,T,M} =
    (SVector{8,T}(ntuple(i -> Y[i], Val(8))),
     ntuple(m -> SVector{4,T}(ntuple(k -> Y[8 + 4(m - 1) + k], Val(4))), Val(M)))

"""
    transport_step(integrator, st, y::RayState{T}, vs::NTuple{M,SVector{4,T}}, h::T)
        -> (y, vs)

One step of the geodesic flow with the parallel transport of the `M` vectors `vs`
([`geodesic_transport_rhs`](@ref)) by `integrator`. With `ClassicalRK4` every component
takes the operations of `rk4_step`; the polarized march keeps its own written-out RK4 of
the same operations, whose compiled contractions this one need not share.
"""
@inline function transport_step(integ::MarchIntegrator, st, y::SVector{8,T},
                                vs::NTuple{M,SVector{4,T}}, h::T) where {T,M}
    f(Y) = (@inline; (yy, ww) = _unpack(Y, Val(M)); (dy, dw) = geodesic_transport_rhs(st, yy, ww);
            _pack(dy, dw))
    return _unpack(erk_step(integ, f, _pack(y, vs), h), Val(M))
end

# ---------------------------------------------------------------- surface crossings

"""
    EquatorialPlane()

The emission surface `z = 0` of Cartesian Kerr–Schild coordinates (the equator
`θ = π/2` of every Kerr–Schild hole whose spin lies along z).
"""
struct EquatorialPlane end

"""
    surface_value(surface, x::SVector{4}) -> Real

The signed level function of `surface` at the event `x`, zero on the surface: `z` for
[`EquatorialPlane`](@ref).
"""
@inline surface_value(::EquatorialPlane, x::SVector{4}) = x[4]

"""
    surface_crossing(integrator, st, y::RayState{T}, h::T, surface) -> (y_c, c)

The state on `surface` inside the step of length `h` from `y`, whose end lies on the
other side: Newton's iteration on `c ↦ s(c)`, the level function of the state after the
partial step `march_step(integrator, st, y, c)`, with `ds/dc` the level function's
rate `∇s · dx/dλ` at that state, started from the secant of the step's end values and
kept inside the bracket `[0, h]` by a bisection fallback. It stops once `|Δc| ≤ 4 eps h`
(at most 32 iterations), so the crossing carries the integrator's own order: the partial
step is the integrator applied over `c ≤ h`. Deterministic: the iterations depend on the
state alone.
"""
@inline function surface_crossing(integ::MarchIntegrator, st, y::SVector{8,T}, h::T,
                                  surface) where {T}
    s0 = surface_value(surface, position(y))
    yh = march_step(integ, st, y, h)
    s1 = surface_value(surface, position(yh))
    lo = zero(T); hi = h; slo = s0
    c = h * s0 / (s0 - s1)
    (c > zero(T) && c < h) || (c = h / 2)
    yc = yh
    for _ in 1:32
        yc = march_step(integ, st, y, c)
        sc = surface_value(surface, position(yc))
        iszero(sc) && break
        (signbit(sc) == signbit(slo)) ? (lo = c; slo = sc) : (hi = c)
        rate = _surface_rate(surface, st, yc)
        cn = c - sc / rate
        (cn > lo && cn < hi && isfinite(cn)) || (cn = (lo + hi) / 2)
        dc = abs(cn - c)
        c = cn
        dc <= 4 * eps(T) * h && (yc = march_step(integ, st, y, c); break)
    end
    return yc, c
end

# d s/dλ of the level function along the flow at y: ∇s · g^{μν} p_ν.
@inline _surface_rate(::EquatorialPlane, st, y::SVector{8}) =
    raise_index(st, position(y), momentum(y))[4]
