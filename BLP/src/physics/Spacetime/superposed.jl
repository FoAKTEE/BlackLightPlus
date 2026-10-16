# Superposed Kerr-Schild holes, g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν: exact for one hole,
# the standard large-separation binary initial-data ansatz for two. The single-hole
# inverse does not extend to the sum (cross terms f_A f_B l^A·l^B), so the inverse is
# the Woodbury identity with an N×N solve.

"""
    SuperposedKerrSchild{N,T} <: StationarySpacetime{T}
    SuperposedKerrSchild(masses, spins, centers)

`N` static Kerr-Schild holes with spins along ±z, superposed as
`g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν` (covariant `l^A_0 = 1`). Fields: `mass`,
`spin` and `center` (Cartesian, `SVector{3}`) per hole, and `r_h`, the per-hole
horizon radii `M_A + √(M_A² - a_A²)` (zero discriminant floor), computed once by
the constructor. The vector constructor builds the `Float64` lane. A second
hole of zero mass leaves single Kerr unchanged.
"""
struct SuperposedKerrSchild{N,T} <: StationarySpacetime{T}
    mass::SVector{N,T}
    spin::SVector{N,T}
    center::SVector{N,SVector{3,T}}
    r_h::SVector{N,T}
end

function SuperposedKerrSchild(mass::SVector{N,T}, spin::SVector{N,T},
                              center::SVector{N,SVector{3,T}}) where {N,T}
    r_h = SVector{N,T}(ntuple(A -> _hole_horizon(mass[A], spin[A]), Val(N)))
    return SuperposedKerrSchild{N,T}(mass, spin, center, r_h)
end

function SuperposedKerrSchild(masses::AbstractVector, spins::AbstractVector,
                              centers::AbstractVector)
    n = length(masses)
    @assert length(spins) == n == length(centers)
    ctrs = SVector{n}(SVector{3,Float64}(c...) for c in centers)
    return SuperposedKerrSchild(SVector{n,Float64}(masses...), SVector{n,Float64}(spins...), ctrs)
end

@inline function _hole_horizon(m, a)
    d = m * m - a * a
    return m + (d > 0 ? sqrt(d) : zero(m))
end

lane(st::SuperposedKerrSchild{N}, ::Type{T}) where {N,T} =
    SuperposedKerrSchild{N,T}(SVector{N,T}(st.mass), SVector{N,T}(st.spin),
                              SVector{N,SVector{3,T}}(map(SVector{3,T}, st.center)),
                              SVector{N,T}(st.r_h))

"""
    ks_radius(st::SuperposedKerrSchild, x::SVector{4}, A::Int)

Kerr-Schild radius of hole `A` at `x`, measured from that hole's centre.
"""
@inline function ks_radius(st::SuperposedKerrSchild, x::SVector{4}, A::Int)
    c = st.center[A]
    return _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])[1]
end

@inline horizon_radius(st::SuperposedKerrSchild, A::Int) = st.r_h[A]
@inline horizon_radius(st::SuperposedKerrSchild) = st.r_h

# (f_A, l^A_μ) of hole A at x.
@inline function _hole_term(st::SuperposedKerrSchild, x::SVector{4}, A::Int)
    c = st.center[A]
    _, f, l = _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])
    return (f, l)
end
# The `@inline` inside the closure keeps the per-hole body inlined in kernels.
@inline _hole_terms(st::SuperposedKerrSchild{N}, x::SVector{4}) where {N} =
    ntuple(A -> (@inline; _hole_term(st, x, A)), Val(N))

# Left fold over a tuple, unrolled at compile time (type-stable when the accumulator's
# type changes on the first step, e.g. a Float64 η plus a Dual term).
@inline _tfold(op, acc, ::Tuple{}) = acc
@inline _tfold(op, acc, t::Tuple) = _tfold(op, op(acc, first(t)), Base.tail(t))

# g = η + Σ_A f_A l_A l_Aᵀ, summed hole by hole.
@inline _add_hole(g, t) = g + t[1] * (t[2] * t[2]')
@inline _superposed_gcov(terms::Tuple) = _tfold(_add_hole, _eta(typeof(terms[1][1])), terms)

# Woodbury: g⁻¹ = η - U K⁻¹ F Uᵀ with U = η L (the raised l^A), F = diag(f_A) and the
# N×N K = I + F P, P = Lᵀ η L (P_AA = l_A·η·l_A vanishes up to round-off: each l_A is
# null). One hole gives η - f l^μ l^ν; two holes need a 2×2 solve.
@inline function _superposed_gcon(terms::NTuple{N}) where {N}
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    return _eta(eltype(F)) - U * ((inv(K) * F) * U')
end

# g⁻¹ p by the same Woodbury form, contracted without the 4×4 matrix.
@inline function _superposed_raise(terms::NTuple{N}, p::SVector{4}) where {N}
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    return _eta_apply(p) - U * ((inv(K) * F) * (U' * p))
end

@inline function raise_and_norm(st::SuperposedKerrSchild{N}, x::SVector{4},
                                p::SVector{4}) where {N}
    terms = _hole_terms(st, x)
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    w = U' * p                                            # l^A p (N components)
    Mw = (inv(K) * F) * w
    ηp = _eta_apply(p)
    return ηp - U * Mw, dot(p, ηp) - dot(w, Mw)
end

@inline lower_index(st::SuperposedKerrSchild, x::SVector{4}, v::SVector{4}) =
    _tfold((g, t) -> (@inline; g + (t[1] * dot(t[2], v)) * t[2]), _eta_apply(v),
           _hole_terms(st, x))
@inline raise_index(st::SuperposedKerrSchild, x::SVector{4}, p::SVector{4}) =
    _superposed_raise(_hole_terms(st, x), p)

@inline metric(st::SuperposedKerrSchild, x::SVector{4}) = _superposed_gcov(_hole_terms(st, x))
@inline inverse_metric(st::SuperposedKerrSchild, x::SVector{4}) =
    _superposed_gcon(_hole_terms(st, x))
@inline function metric_and_inverse(st::SuperposedKerrSchild, x::SVector{4})
    terms = _hole_terms(st, x)
    return _superposed_gcov(terms), _superposed_gcon(terms)
end
