"""
    Geodesics — metrics.jl

Metric type tree + AD metric derivatives. Signature (−,+,+,+), G = c = M = 1,
Cartesian Kerr-Schild coordinates. Christoffels and metric derivatives are
DERIVED (ForwardDiff over the metric closure), never hand-supplied; the shipped
Kerr-Schild metric additionally has the exact closed-form inverse
`g^{ab} = η^{ab} − f l^a l^b` (validated to machine precision against AD).
"""
# ---------------------------------------------------------------- type tree
abstract type AbstractSpacetime{T} end
abstract type AnalyticSpacetime{T} <: AbstractSpacetime{T} end
abstract type StationarySpacetime{T} <: AnalyticSpacetime{T} end

"Kerr in Cartesian Kerr-Schild coordinates (blacklight-compatible default)."
struct KerrSchildCartesian{T} <: StationarySpacetime{T}
    a::T
end

"Flat spacetime."
struct Minkowski{T} <: StationarySpacetime{T} end
Minkowski() = Minkowski{Float64}()

"""
Any user closure `x::SVector{4} -> SMatrix{4,4}` (covariant); AD supplies ∂g. Traced
through the general `AnalyticSpacetime` `geodesic_rhs`/`trace_geodesic` path, which
retains the `∂_t g` term — a TIME-DEPENDENT closure integrates correctly. The engine
knows no horizon for an arbitrary metric, so `trace_geodesic` stops only on the escape
sphere or `λmax`.
"""
struct UserMetric{F,T} <: AnalyticSpacetime{T}
    g::F
end
UserMetric(f::F) where {F} = UserMetric{F,Float64}(f)

const ETA = @SMatrix Float64[-1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

# ---------------------------------------------------------------- Kerr-Schild
# (r, f, l_a) for ONE Kerr-Schild hole of mass M, spin a, at the local Cartesian
# offset (X,Y,Z) from that hole (hypot-form root, robust near the ring). THE single
# closed-form primitive — single Kerr (`_rfl`, M=1), the superposed binary
# (spacetimes.jl), and every `ks_radius*` route through it, so a guard added here
# reaches every KS entry point at once. (The GPU kernels, gpu_metric.jl, re-seed
# the same closed form for in-kernel duals — agreement is fp-roundoff, not
# bit-for-bit; keep the two in sync.)
@inline function _ks_term(X::T, Y::T, Z::T, M, a) where {T}
    a2 = a * a
    rr2 = X * X + Y * Y + Z * Z
    r2 = (rr2 - a2 + hypot(rr2 - a2, 2 * a * Z)) / 2
    r = sqrt(r2)
    # r = 0 on the OPEN equatorial disk x²+y²<a², z=0 (including the ~|z| ≲ 5e-9
    # slab where hypot cancellation rounds r² to 0): the 0/0 in f and l_z is
    # removable and the limit is exactly g = η, so return f = 0 with the finite
    # part of l (f·l l' → 0 regardless of l_z). The RIM x²+y²=a² (and the a=0
    # origin) falls through and keeps its NaN — a true curvature singularity.
    # All of this sits inside the horizon.
    if iszero(r) && rr2 < a2
        return r, zero(M * r), SVector(one(T), Y / a, -X / a, zero(T))
    end
    f = 2 * M * r2 * r / (r2 * r2 + a2 * Z * Z)          # 2 M r^3 / (r^4 + a^2 z^2)
    l = SVector(one(T), (r * X + a * Y) / (r2 + a2),
                (r * Y - a * X) / (r2 + a2), Z / r)             # l_a, l_0 = +1
    return r, f, l
end

# Single Kerr (M = 1) from the shared primitive.
@inline _rfl(x::SVector{4,T}, a) where {T} = _ks_term(x[2], x[3], x[4], one(T), a)

@inline ks_radius(st::KerrSchildCartesian, x::SVector{4}) = _rfl(x, st.a)[1]
@inline horizon_radius(st::KerrSchildCartesian) = 1 + sqrt(1 - st.a * st.a)

@inline function metric(st::KerrSchildCartesian, x::SVector{4})
    _, f, l = _rfl(x, st.a)
    return ETA + f * (l * l')
end

"Exact Kerr-Schild inverse: g^{ab} = η^{ab} − f l^a l^b with l^a = η^{ab} l_b."
@inline function inverse_metric(st::KerrSchildCartesian, x::SVector{4})
    _, f, l = _rfl(x, st.a)
    lu = SVector(-l[1], l[2], l[3], l[4])
    return ETA - f * (lu * lu')
end

@inline metric(::Minkowski, ::SVector{4}) = ETA
@inline inverse_metric(::Minkowski, ::SVector{4}) = ETA

@inline metric(st::UserMetric, x::SVector{4}) = st.g(x)
@inline inverse_metric(st::UserMetric, x::SVector{4}) = inv(metric(st, x))

"""
    inverse_metric_gradient(st, x) -> SArray{Tuple{4,4,4}}

`dg[a,b,c] = ∂ g^{ab} / ∂ x^c`, by forward-mode AD of `inverse_metric`
(compile-time specialized; exact through the KS closed form).
"""
@inline function inverse_metric_gradient(st::AbstractSpacetime, x::SVector{4,T}) where {T}
    J = ForwardDiff.jacobian(xx -> inverse_metric(st, xx), x)    # (16, 4), column-major
    return SArray{Tuple{4,4,4},T}(J)
end
# Stationary metrics have ∂_t g^{ab} ≡ 0 exactly, so differentiate over the three
# SPATIAL coordinates only (3 hand-seeded dual partials — the same seeding the
# in-kernel GPU RHS uses) and write the exact-zero t-column. Spatial columns are
# bit-identical to the 4-column jacobian (per-partial dual arithmetic is independent).
@inline function inverse_metric_gradient(st::StationarySpacetime, x::SVector{4,T}) where {T}
    # A REAL tag (not Nothing), parameterized by T: under nested AD (Dual st.a or x)
    # T embeds the outer tag, so this tag is instantiated later ⇒ ranks outermost
    # (ForwardDiff tagcount ordering) and `partials` below extracts OUR seeds.
    Tg = typeof(ForwardDiff.Tag(inverse_metric_gradient, T))
    z = zero(T); o = one(T)
    xd = SVector(ForwardDiff.Dual{Tg}(x[1], (z, z, z)), ForwardDiff.Dual{Tg}(x[2], (o, z, z)),
                 ForwardDiff.Dual{Tg}(x[3], (z, o, z)), ForwardDiff.Dual{Tg}(x[4], (z, z, o)))
    gd = inverse_metric(st, xd)                                  # values + ∂/∂(x,y,z)
    return SArray{Tuple{4,4,4},T}((ntuple(_ -> z, Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[1], Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[2], Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[3], Val(16))...))
end
@inline inverse_metric_gradient(::Minkowski, x::SVector{4,T}) where {T} =
    zero(SArray{Tuple{4,4,4},T})

# ---------------------------------------------------------------- index utils
@inline lower_index(st, x, v::SVector{4}) = metric(st, x) * v
@inline raise_index(st, x, v::SVector{4}) = inverse_metric(st, x) * v


"""
    metric_gradient(st, x) -> SArray{Tuple{4,4,4}}

`dg[a,b,c] = ∂ g_{ab} / ∂ x^c` (covariant), by forward-mode AD of `metric`.
"""
@inline function metric_gradient(st::AbstractSpacetime, x::SVector{4,T}) where {T}
    J = ForwardDiff.jacobian(xx -> metric(st, xx), x)
    return SArray{Tuple{4,4,4},T}(J)
end

"""
    christoffel(st, x)           -> SArray{Tuple{4,4,4}}
    christoffel(st, x, gcon, dg)  -> SArray{Tuple{4,4,4}}

Γ^μ_{αβ} = ½ g^{μλ}(∂_α g_{λβ} + ∂_β g_{λα} − ∂_λ g_{αβ}), derived via AD;
index order Γ[μ, α, β]. The four-argument form contracts a PRECOMPUTED inverse
metric `gcon = g^{μλ}` and covariant gradient `dg[a,b,c] = ∂_c g_{ab}`
(`metric_gradient(st, x)`) so hot-loop callers that already hold `g^{μλ}` skip a
second closed-form metric evaluation; the two-argument form derives both and
delegates. Allocation-free when inlined (`@SArray` comprehensions).
"""
@inline function christoffel(st::AbstractSpacetime, x::SVector{4,T}, gcon, dg) where {T}
    # Γl[λ,α,β] = ½(∂_α g_{λβ} + ∂_β g_{λα} − ∂_λ g_{αβ})
    Γl = @inbounds @SArray T[
        (dg[λ, β, α] + dg[λ, α, β] - dg[α, β, λ]) / 2
        for λ in 1:4, α in 1:4, β in 1:4]
    # Γ[μ,α,β] = Σ_λ g^{μλ} Γl[λ,α,β]  (explicit zero-seeded left-to-right chain)
    return @inbounds @SArray T[
        (((zero(T) + gcon[μ, 1] * Γl[1, α, β]) + gcon[μ, 2] * Γl[2, α, β]) +
         gcon[μ, 3] * Γl[3, α, β]) + gcon[μ, 4] * Γl[4, α, β]
        for μ in 1:4, α in 1:4, β in 1:4]
end

christoffel(st::AbstractSpacetime, x::SVector{4}) =
    christoffel(st, x, inverse_metric(st, x), metric_gradient(st, x))
