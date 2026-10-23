# The fixed-control march: the step size and the one termination rule.
#
# A backward-traced ray (p = -k, x^t decreasing) that falls toward a hole approaches the
# future horizon asymptotically: x^t → -∞ while its normal-frame energy
# E_n = -k·n = α |dx^t/dλ| (camera E = 1) grows without bound and r - r_h → 0. The march
# stops when the distance d above the horizon proxy drops below an epsilon or turns
# non-finite, or when E_n exceeds CAPTURE_REDSHIFT = E_c (the redshift cut). Beyond the
# cut the fluid-frame frequency is ν ≥ E_c ν_obs / (2γ), γ = -u·n, and the affine length
# left is Δλ ≈ α_h / (κ E_c) (κ the surface gravity, α_h ≤ 1/√2 the Kerr-Schild lapse
# on the horizon), so with transmission ≤ 1 the omitted intensity obeys
#   ΔI_ν ≤ ν_obs³ (L_unit / ν_obs) Δλ max_beyond j_ν/ν²,
# with j_ν/ν² evaluated at ν ≥ E_c ν_obs / (2γ). In a superposed binary the companion's
# potential moves the null surface outward, so r_A - r_h,A stalls above the epsilon and
# the redshift cut ends the approach. A ray escapes once it is outbound beyond `escape`
# or the emission radius `R_em`.

"Default escape radius of the adaptive tracer on analytic spacetimes."
const ESCAPE_RADIUS = 1.1e3
"Default escape radius of the adaptive tracer on an ADM grid (bounded by the grid half-width)."
const ADM_ESCAPE_RADIUS = 18.0
"Horizon offset of the adaptive Kerr tracer: it stops at `ks_radius = r_h + HORIZON_EPS`."
const HORIZON_EPS = 1e-4
"Horizon backstop of the fixed-control march: captured at `r - r_h < HORIZON_EPS_SAMPLED`."
const HORIZON_EPS_SAMPLED = 1e-3
"Normal-frame photon energy (camera energy 1) above which the march captures."
const CAPTURE_REDSHIFT = 1.0e2

"""
    march_captured(d, E_n, eps = HORIZON_EPS_SAMPLED) -> Bool

Capture half of the termination rule: `d` is the distance above the horizon
proxy, `E_n` the normal-frame photon energy. True when `d < eps`, `d` is
non-finite, or `E_n` is not `≤ CAPTURE_REDSHIFT` (NaN captures). The constants
are converted to the type of `d`, so a `Float32` lane stays `Float32`.
"""
@inline march_captured(d::T, En::T, eps::Real = HORIZON_EPS_SAMPLED) where {T<:Real} =
    d < T(eps) || !isfinite(d) || !(En <= T(CAPTURE_REDSHIFT))

"""
    march_escaped(R, R_prev, escape, R_em) -> Bool

Escape half of the termination rule: `R` beyond `escape` or beyond the emission
radius `R_em`, and outbound (`R > R_prev`), so a ray started beyond `escape` is
not stopped on its way in. `R_em = Inf` disables the emission early-out.
"""
@inline march_escaped(R::T, R_prev::T, escape::T, R_em::T) where {T<:Real} =
    (R > escape || R > R_em) && R > R_prev

"""
    march_stop(d, E_n, R, R_prev, escape, R_em) -> Bool

The termination rule `march_captured(d, E_n) || march_escaped(R, R_prev, escape, R_em)`.
All arguments share one type: a `Float64` scalar in a `Float32` march is a
`MethodError`, never a silent promotion.
"""
@inline march_stop(d::T, En::T, R::T, R_prev::T, escape::T, R_em::T) where {T<:Real} =
    march_captured(d, En) || march_escaped(R, R_prev, escape, R_em)

"""
    normal_frame_energy(gcon::SMatrix{4,4}, y::RayState)
    normal_frame_energy(st, y::RayState)

Photon energy in the local normal (Eulerian) frame, `|g^{tμ} p_μ| / √(-g^{tt})`,
in units of the camera energy.
"""
@inline function normal_frame_energy(gi::SMatrix{4,4}, y::SVector{8})
    tdot = gi[1, 1] * y[5] + gi[1, 2] * y[6] + gi[1, 3] * y[7] + gi[1, 4] * y[8]
    return abs(tdot) / sqrt(-gi[1, 1])
end
@inline normal_frame_energy(st, y::SVector{8}) = normal_frame_energy(inverse_metric(st, position(y)), y)

"""
    ks_radius_energy(y::RayState, a) -> (r, E_n)

Kerr-Schild radius and normal-frame photon energy of `y` on Kerr of spin `a`
(unit mass) from one closed-form evaluation: with `g^{μν} = η^{μν} - f l^μ l^ν`,
`g^{tt} = -(1 + f)` and `g^{ti} = f l_i`.
"""
@inline function ks_radius_energy(y::SVector{8,T}, a) where {T}
    r, f, l = _ks_term(y[2], y[3], y[4], one(T), a)
    tdot = f * (l[2] * y[6] + l[3] * y[7] + l[4] * y[8]) - (1 + f) * y[5]
    return r, abs(tdot) / sqrt(1 + f)
end

"""
    MarchBudget{T}(C, escape, R_em, nmax)

Control of a fixed-control march: step divisor `C` (the step is the distance to
the horizon proxy over `C`), escape radius `escape`, emission radius `R_em`
(`Inf` disables the early-out) and step limit `nmax`. `lane(b, T)` converts the
scalars.
"""
struct MarchBudget{T}
    C::T
    escape::T
    R_em::T
    nmax::Int
end
MarchBudget(C, escape, R_em, nmax::Integer) = MarchBudget(promote(C, escape, R_em)..., Int(nmax))

lane(b::MarchBudget, ::Type{T}) where {T} = MarchBudget{T}(T(b.C), T(b.escape), T(b.R_em), b.nmax)

"""
    MarchStep{T}

One decision of [`march_control`](@ref): `stop`, whether the march ends before
sampling the current state; `h`, the affine step to take; `R`, the radius the
next escape test compares against (`R_prev`).
"""
struct MarchStep{T}
    stop::Bool
    h::T
    R::T
end

"""
    march_control(st, y::RayState{T}, r_prev::T, b::MarchBudget{T}) -> MarchStep{T}

The fixed-control step at state `y` (`r_prev` the radius of the previous
decision, `Inf` at the first):

- `KerrSchildCartesian`: `d = r - r_h` on the Kerr-Schild radius `r`,
  `stop = march_stop(d, E_n, r, r_prev, escape, R_em)`, `h = d / C`, `R = r`;
- `SuperposedKerrSchild` and `BoostedSuperposedKerrSchild`: `d = min_A(r_A - r_h,A)`
  (`-Inf` on a non-finite `r_A`; the moving holes' `r_A` at the ray's time), the
  same rule on the coordinate radius `R`, `h = d / C`;
- `ADMGrid`: `d` is the lapse `1/√(-g^{tt})` minus `alpha_min` (lapse 1 and no
  redshift cut where `g^{tt} ≥ 0`), capture at `d < 0`, `h = max(R, 1)/C` on the
  coordinate radius `R`; a non-finite `R` captures;
- any other spacetime: no capture surface, `h = max(R, 1)/C`, stop on escape or a
  non-finite `R`.
"""
@inline function march_control(st::KerrSchildCartesian{T}, y::SVector{8,T}, r_prev::T,
                               b::MarchBudget{T}) where {T}
    r, En = ks_radius_energy(y, st.a)
    return MarchStep{T}(march_stop(r - st.r_h, En, r, r_prev, b.escape, b.R_em),
                        (r - st.r_h) / b.C, r)
end

@inline function march_control(st::Union{SuperposedKerrSchild{N,T},BoostedSuperposedKerrSchild{N,T}},
                               y::SVector{8,T}, r_prev::T, b::MarchBudget{T}) where {N,T}
    x = position(y)
    dmin = T(Inf)
    @inbounds for A in 1:N
        dA = ks_radius(st, x, A) - horizon_radius(st, A)
        isfinite(dA) || (dmin = T(-Inf); break)
        dmin = min(dmin, dA)
    end
    R = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    cap = march_captured(dmin, normal_frame_energy(st, y))
    return MarchStep{T}(cap || march_escaped(R, r_prev, b.escape, b.R_em), dmin / b.C, R)
end

@inline function march_control(st::ADMGrid{T}, y::SVector{8,T}, r_prev::T,
                               b::MarchBudget{T}) where {T}
    x = position(y)
    r = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    gi = inverse_metric(st, x)
    gtt = gi[1, 1]
    alpha = gtt < zero(T) ? one(T) / sqrt(-gtt) : one(T)
    En = gtt < zero(T) ? normal_frame_energy(gi, y) : zero(T)
    d = isfinite(r) ? alpha - st.alpha_min : r
    cap = march_captured(d, En, zero(T))
    return MarchStep{T}(cap || march_escaped(r, r_prev, b.escape, b.R_em),
                        max(r, one(T)) / b.C, r)
end

@inline function march_control(st::Spacetime{T}, y::SVector{8,T}, r_prev::T,
                               b::MarchBudget{T}) where {T}
    R = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    stop = !isfinite(R) || march_escaped(R, r_prev, b.escape, b.R_em)
    return MarchStep{T}(stop, max(R, one(T)) / b.C, R)
end

"""
    proxy_distance(st, x::SVector{4}) -> (d, r)

The distance `d` above the spacetime's horizon proxy at the event `x` and the radius
`r` that escape rules compare against, the pair every march that is not the
fixed-control march reads (the Monte Carlo march): `KerrSchildCartesian`
`(r_ks - r_h, r_ks)`; `SuperposedKerrSchild` and `BoostedSuperposedKerrSchild`
`(min_A (r_A - r_h,A), |x|)` with `r_A` the per-hole radius at the ray's time (`-Inf`
when one is not finite); `ADMGrid` `(α - alpha_min, |x|)` with the lapse from `g^{tt}`
where it is negative and 1 elsewhere; `Minkowski` and any other spacetime `(|x|, |x|)`,
no horizon.
"""
@inline function proxy_distance(st::KerrSchildCartesian, x::SVector{4})
    r = ks_radius(st, x)
    return r - st.r_h, r
end
@inline function proxy_distance(st::Union{SuperposedKerrSchild{N},BoostedSuperposedKerrSchild{N}},
                                x::SVector{4}) where {N}
    T = eltype(x)
    dmin = T(Inf)
    @inbounds for A in 1:N
        dA = ks_radius(st, x, A) - horizon_radius(st, A)
        isfinite(dA) || (dmin = T(-Inf); break)
        dmin = min(dmin, dA)
    end
    return dmin, sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
end
@inline function proxy_distance(st::ADMGrid, x::SVector{4})
    R = sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
    gtt = inverse_metric(st, x)[1, 1]
    α = gtt < zero(gtt) ? one(gtt) / sqrt(-gtt) : one(gtt)
    return α - st.alpha_min, R
end
@inline function proxy_distance(st::Spacetime, x::SVector{4})
    R = sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
    return R, R
end

"""
    horizon_floor(st) -> Real

The scalar horizon proxy of a march on `st`: the Kerr outer horizon radius, the
smallest per-hole horizon radius of a binary, the lapse floor of an ADM grid, and 0
for `Minkowski` or any spacetime without a horizon.
"""
@inline horizon_floor(st::KerrSchildCartesian) = st.r_h
@inline horizon_floor(st::Union{SuperposedKerrSchild,BoostedSuperposedKerrSchild}) = minimum(st.r_h)
@inline horizon_floor(st::ADMGrid) = st.alpha_min
@inline horizon_floor(::Spacetime) = 0.0
