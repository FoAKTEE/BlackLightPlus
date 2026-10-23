# Kerr-Schild holes in motion: the superposition g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν with each
# hole's term taken in the rest frame of its instantaneous velocity v_A(t), at the spatial
# offset x'_A = Λ(v_A)(x - s_A(t)) of the hole, and its null covector boosted back to the
# lab frame, l^A = Λ(v_A) l'^A (arXiv:2403.13308, eqs. sks and covectorBH, in the t̄ = 0
# form of arXiv:2103.15707). The term is exact Kerr for a hole in uniform motion, and a
# boosted null covector is null, so the Woodbury inverse of the static superposition
# serves the moving one unchanged. The metric depends on t through s_A and v_A: the
# spacetime is not stationary, the general geodesic right-hand side keeps ∂_t g and p_t is
# not conserved along a ray.

"""
    CircularOrbit{T}
    CircularOrbit(masses, separation; omega = √(M/d³), phase = 0.0)

The trajectory of two holes on a circular orbit in the x-y plane about their
centre of mass: hole 1 at `r₁ = +M₂ d/M`, hole 2 at `r₂ = -M₁ d/M` along
`(cos(ωt + φ), sin(ωt + φ), 0)`, with `d` the separation, `M = M₁ + M₂`, `ω`
the angular frequency (Keplerian, `√(M/d³)`, by default) and `φ` the phase at
`t = 0`; the velocities are the time derivatives. `orbit(t)` returns the
`NTuple{2}` of `(s_A, v_A)` pairs (`SVector{3}` each) and is generic in `t`.
"""
struct CircularOrbit{T}
    r::SVector{2,T}
    omega::T
    phase::T
end

function CircularOrbit(masses, separation; omega = nothing, phase = 0.0)
    m1, m2 = float(masses[1]), float(masses[2])
    M = m1 + m2
    d = float(separation)
    ω = omega === nothing ? sqrt(M / (d * d * d)) : float(omega)
    return CircularOrbit(SVector(m2 / M * d, -m1 / M * d), ω, float(phase))
end

@inline function (o::CircularOrbit)(t)
    φ = o.omega * t + o.phase
    c = cos(φ); s = sin(φ)
    return ntuple(A -> (@inline; r = o.r[A];
                        (SVector(r * c, r * s, zero(c)),
                         SVector(-r * o.omega * s, r * o.omega * c, zero(c)))), Val(2))
end

_lane_trajectory(o::CircularOrbit, ::Type{T}) where {T} =
    CircularOrbit{T}(SVector{2,T}(o.r), T(o.omega), T(o.phase))
_lane_trajectory(f, ::Type) = f

"""
    BoostedSuperposedKerrSchild{N,T,F} <: Spacetime{T}
    BoostedSuperposedKerrSchild(masses, spins, trajectory; boost = true)

`N` Kerr-Schild holes in prescribed motion, `g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν`
with each hole's term evaluated in the rest frame of its velocity: at the
offset `x'_A = Λ(v_A)(x - s_A(t))` (the spatial Lorentz contraction) and with
the covector boosted back, `l^A = Λ(v_A) l'^A`. Fields: `mass` and `spin` (a
spin vector `a_A`, `|a_A| ≤ M_A`) per hole, `trajectory`, the callable
`trajectory(t) -> NTuple{N}` of `(s_A, v_A)` pairs (a [`CircularOrbit`](@ref)
or any closure generic in `t`), `boost`, and `r_h`, the per-hole horizon
radii `M_A + √(M_A² - |a_A|²)` measured in `x'_A`. With `boost = false` the
holes move without the boost of their terms (the static superposition carried
along the trajectory). The vector constructor builds the `Float64` lane;
`lane(st, T)` converts the scalars and a `CircularOrbit`.
"""
struct BoostedSuperposedKerrSchild{N,T,F} <: Spacetime{T}
    mass::SVector{N,T}
    spin::SVector{N,SVector{3,T}}
    trajectory::F
    boost::Bool
    r_h::SVector{N,T}
end

function BoostedSuperposedKerrSchild(masses::AbstractVector, spins::AbstractVector, trajectory;
                                     boost::Bool = true)
    n = length(masses)
    @assert length(spins) == n
    m = SVector{n,Float64}(masses...)
    a = SVector{n}(SVector{3,Float64}(s...) for s in spins)
    r_h = SVector{n,Float64}(ntuple(A -> _hole_horizon(m[A], sqrt(dot(a[A], a[A]))), Val(n)))
    return BoostedSuperposedKerrSchild{n,Float64,typeof(trajectory)}(m, a, trajectory, boost, r_h)
end

function lane(st::BoostedSuperposedKerrSchild{N}, ::Type{T}) where {N,T}
    tr = _lane_trajectory(st.trajectory, T)
    return BoostedSuperposedKerrSchild{N,T,typeof(tr)}(
        SVector{N,T}(st.mass), SVector{N,SVector{3,T}}(map(SVector{3,T}, st.spin)), tr,
        st.boost, SVector{N,T}(st.r_h))
end

# (r, f, l_μ) of a hole of mass M and spin vector a moving at v, at the lab offset d from
# its centre: the rest-frame Kerr-Schild term at x' = d + (Γ-1)(n·d) n (arbitrary spin
# direction: r² = (R² - a² + √((R² - a²)² + 4 (a·x')²))/2, f = 2 M r³/(r⁴ + (a·x')²),
# l'_i = (r x' - a × x' + (a·x') a/r)/(r² + a²), l'_0 = 1), then l = Λ(v) l' with
# Λ⁰₀ = Γ, Λ⁰ᵢ = Λⁱ₀ = -Γ vᵢ, Λⁱⱼ = δᵢⱼ + (Γ-1) nᵢ nⱼ, written with
# (Γ-1)/v² = Γ²/(Γ+1) so that v = 0 needs no direction. Unboosted: x' = d, l = l'.
@inline function _moving_ks_term(M, a::SVector{3}, d::SVector{3}, v::SVector{3}, boost::Bool)
    v2 = dot(v, v)
    Γ = 1 / sqrt(1 - v2)
    c = boost ? Γ * Γ / (Γ + 1) : zero(Γ)
    xp = d + (c * dot(v, d)) * v
    a2 = dot(a, a)
    R2 = dot(xp, xp)
    adx = dot(a, xp)
    r2 = (R2 - a2 + hypot(R2 - a2, 2 * adx)) / 2
    r = sqrt(r2)
    f = 2 * M * r2 * r / (r2 * r2 + adx * adx)
    ls = (r * xp - cross(a, xp) + (adx / r) * a) / (r2 + a2)   # l'_i, with l'_0 = +1
    if boost
        vl = dot(v, ls)
        l0 = Γ - Γ * vl
        li = ls - Γ * v + (c * vl) * v
        return r, f, SVector(l0, li[1], li[2], li[3])
    end
    return r, f, SVector(one(r), ls[1], ls[2], ls[3])
end

@inline function _moving_hole(st::BoostedSuperposedKerrSchild, x::SVector{4}, sv, A::Int)
    s, v = sv
    d = SVector(x[2], x[3], x[4]) - s
    return _moving_ks_term(st.mass[A], st.spin[A], d, v, st.boost)
end

@inline function _hole_terms(st::BoostedSuperposedKerrSchild{N}, x::SVector{4}) where {N}
    traj = st.trajectory(x[1])
    return ntuple(A -> (@inline; t = _moving_hole(st, x, traj[A], A); (t[2], t[3])), Val(N))
end

"""
    ks_radius(st::BoostedSuperposedKerrSchild, x::SVector{4}, A::Int)

Kerr-Schild radius of hole `A` in its own rest frame at the event `x`: the
spheroidal radius of `x'_A = Λ(v_A)(x - s_A(x⁰))`.
"""
@inline function ks_radius(st::BoostedSuperposedKerrSchild, x::SVector{4}, A::Int)
    return _moving_hole(st, x, st.trajectory(x[1])[A], A)[1]
end

@inline horizon_radius(st::BoostedSuperposedKerrSchild, A::Int) = st.r_h[A]
@inline horizon_radius(st::BoostedSuperposedKerrSchild) = st.r_h

@inline metric(st::BoostedSuperposedKerrSchild, x::SVector{4}) = _superposed_gcov(_hole_terms(st, x))
@inline inverse_metric(st::BoostedSuperposedKerrSchild, x::SVector{4}) =
    _superposed_gcon(_hole_terms(st, x))
@inline function metric_and_inverse(st::BoostedSuperposedKerrSchild, x::SVector{4})
    terms = _hole_terms(st, x)
    return _superposed_gcov(terms), _superposed_gcon(terms)
end
@inline lower_index(st::BoostedSuperposedKerrSchild, x::SVector{4}, v::SVector{4}) =
    _superposed_lower(_hole_terms(st, x), v)
@inline raise_index(st::BoostedSuperposedKerrSchild, x::SVector{4}, p::SVector{4}) =
    _superposed_raise(_hole_terms(st, x), p)
@inline raise_and_norm(st::BoostedSuperposedKerrSchild, x::SVector{4}, p::SVector{4}) =
    _superposed_raise_and_norm(_hole_terms(st, x), p)
