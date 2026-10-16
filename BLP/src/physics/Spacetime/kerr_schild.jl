# Kerr in Cartesian Kerr-Schild coordinates, g_μν = η_μν + f l_μ l_ν, and flat space.
# The closed-form inverse g^μν = η^μν - f l^μ l^ν with l^μ = η^μν l_ν holds because l is
# null; every Kerr-Schild route (single hole, superposed binary, march control) evaluates
# the one primitive `_ks_term`.

# (r, f, l_μ) of one Kerr-Schild hole of mass M and spin a at the Cartesian offset
# (X, Y, Z) from the hole, with the hypot form of the radius. r = 0 on the open
# equatorial disk X² + Y² < a², Z = 0 (and in the |Z| ≲ 5e-9 slab where the hypot
# cancellation rounds r² to 0): the 0/0 in f and l_z is removable with limit g = η, so
# f = 0 is returned with the finite part of l. The rim X² + Y² = a² (and the a = 0
# origin) keeps its NaN, a curvature singularity. All of this is inside the horizon.
@inline function _ks_term(X::T, Y::T, Z::T, M, a) where {T}
    a2 = a * a
    rr2 = X * X + Y * Y + Z * Z
    r2 = (rr2 - a2 + hypot(rr2 - a2, 2 * a * Z)) / 2
    r = sqrt(r2)
    if iszero(r) && rr2 < a2
        return r, zero(M * r), SVector(one(T), Y / a, -X / a, zero(T))
    end
    f = 2 * M * r2 * r / (r2 * r2 + a2 * Z * Z)          # 2 M r^3 / (r^4 + a^2 z^2)
    l = SVector(one(T), (r * X + a * Y) / (r2 + a2),
                (r * Y - a * X) / (r2 + a2), Z / r)             # l_μ, l_0 = +1
    return r, f, l
end

# Single Kerr (M = 1). The unit mass takes the spin's type, so a Dual position does not
# carry a derivative-free mass through the products.
@inline _rfl(x::SVector{4}, a) = _ks_term(x[2], x[3], x[4], one(a), a)

"""
    KerrSchildCartesian{T} <: StationarySpacetime{T}
    KerrSchildCartesian(a)

Kerr of unit mass and spin `a` (along +z) in Cartesian Kerr-Schild coordinates.
Fields: `a`, the spin, and `r_h = 1 + √(1 - a²)`, the outer horizon radius,
computed once by the constructor in the precision of `a`. The `Float32` lane is
`lane(KerrSchildCartesian(a), Float32)`, which narrows both from `Float64`.
"""
struct KerrSchildCartesian{T} <: StationarySpacetime{T}
    a::T
    r_h::T
end
KerrSchildCartesian(a::Real) = (af = float(a); KerrSchildCartesian(af, 1 + sqrt(1 - af * af)))

lane(st::KerrSchildCartesian, ::Type{T}) where {T} = KerrSchildCartesian{T}(T(st.a), T(st.r_h))

"""
    ks_radius(st::KerrSchildCartesian, x::SVector{4})

Kerr-Schild radius `r` at `x`, the spheroidal coordinate with
`(x² + y²)/(r² + a²) + z²/r² = 1` (equal to the Boyer-Lindquist `r`). It
coincides with the Euclidean radius only as `r → ∞` or for `a = 0`.
"""
@inline ks_radius(st::KerrSchildCartesian, x::SVector{4}) = _rfl(x, st.a)[1]

@inline horizon_radius(st::KerrSchildCartesian) = st.r_h

@inline function metric(st::KerrSchildCartesian, x::SVector{4})
    _, f, l = _rfl(x, st.a)
    return _eta(typeof(f)) + f * (l * l')
end

@inline function inverse_metric(st::KerrSchildCartesian, x::SVector{4})
    _, f, l = _rfl(x, st.a)
    lu = SVector(-l[1], l[2], l[3], l[4])
    return _eta(typeof(f)) - f * (lu * lu')
end

@inline function metric_and_inverse(st::KerrSchildCartesian, x::SVector{4})
    _, f, l = _rfl(x, st.a)
    lu = SVector(-l[1], l[2], l[3], l[4])
    eta = _eta(typeof(f))
    return eta + f * (l * l'), eta - f * (lu * lu')
end

@inline function lower_index(st::KerrSchildCartesian, x::SVector{4}, v::SVector{4})
    _, f, l = _rfl(x, st.a)
    return _eta_apply(v) + (f * dot(l, v)) * l
end

@inline function raise_index(st::KerrSchildCartesian, x::SVector{4}, p::SVector{4})
    _, f, l = _rfl(x, st.a)
    lu = _eta_apply(l)
    return _eta_apply(p) - (f * dot(lu, p)) * lu
end

# Norm by the closed form η(p, p) - f (l^μ p_μ)².
@inline function raise_and_norm(st::KerrSchildCartesian, x::SVector{4}, p::SVector{4})
    _, f, l = _rfl(x, st.a)
    lu = _eta_apply(l)
    lp = dot(lu, p)
    ηp = _eta_apply(p)
    return ηp - (f * lp) * lu, dot(p, ηp) - f * (lp * lp)
end

"""
    Minkowski{T} <: StationarySpacetime{T}
    Minkowski()

Flat spacetime, `g = η` everywhere (`Minkowski()` is the `Float64` lane).
"""
struct Minkowski{T} <: StationarySpacetime{T} end
Minkowski() = Minkowski{Float64}()

lane(::Minkowski, ::Type{T}) where {T} = Minkowski{T}()

@inline metric(::Minkowski, x::SVector{4,T}) where {T} = _eta(T)
@inline inverse_metric(::Minkowski, x::SVector{4,T}) where {T} = _eta(T)
@inline lower_index(::Minkowski, x::SVector{4}, v::SVector{4}) = _eta_apply(v)
@inline raise_index(::Minkowski, x::SVector{4}, p::SVector{4}) = _eta_apply(p)
