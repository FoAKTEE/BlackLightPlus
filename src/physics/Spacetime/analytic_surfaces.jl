# Closed-form landings on a surface of constant Boyer–Lindquist polar angle θs (a cone; the
# equator is one), for ordinary rays (η > 0) and vortical rays (η < 0, confined to one
# hemisphere). With sβ = sign β (sign 0 = +1), a ray traced back from the observer moves
# first toward the pole (increasing Ψ below), and after m polar turning points it meets
# the surface at the Mino time
#   G(m) = m Ĝ − sβ Ψ(θo) + νθ Ψ(θs),   νθ = sβ (−1)^m,
# with νθ the sign of p^θ of the photon at emission (P1 Eqs. 18–23 generalised to the
# source term). Ordinary rays: Ψθ = F(arcsin(cos θ/√u₊)|u₊/u₋)/(a√−u₋) between the
# turning values ±Ĝ/2, Ĝ = 2K/(a√−u₋). Vortical rays: Ψθ = F(ψ|1 − u₊/u₋)/(a√u₋) − H/2
# with sin²ψ = (cos²θ − u₋)/(u₊ − u₋) and Ĝ = H = K(1 − u₊/u₋)/(a√u₋), the half
# oscillation between cos²θ = u₋ and u₊. Gφ and Gt follow with the same coefficients
# (their integrands are 1/sin²θ and cos²θ times dθ/√Θ).

# The equator exactly: cos(π/2) of the Float64 π/2 is 6e-17, so the equator is matched
# by value and every source term of the equatorial surface is an exact zero.
@inline _surface_cos(θs::T) where {T} = θs == T(π) / 2 ? zero(T) : cos(θs)

# The vortical polar elliptic terms of the observer: half oscillations H and the centred
# observer antiderivatives Ψ(θo) − H/2 of θ, φ and t, with the parameters of the source term.
@inline function _vortical_parts(u_p, u_m, co, a)
    kv = 1 - u_p / u_m
    nv = (u_p - u_m) / (1 - u_m)
    s = sqrt(u_m)
    cθ = 1 / (a * s); cφ = 1 / ((1 - u_m) * a * s); ct = s / a
    Hθ = cθ * JacobiElliptic.K(kv)
    Hφ = cφ * JacobiElliptic.Pi(nv, oftype(kv, π) / 2, kv)
    Ht = ct * JacobiElliptic.E(kv)
    ψo = asin(sqrt(clamp((co * co - u_m) / (u_p - u_m), zero(kv), one(kv))))
    Ψθ = cθ * JacobiElliptic.F(ψo, kv) - Hθ / 2
    Ψφ = cφ * JacobiElliptic.Pi(nv, ψo, kv) - Hφ / 2
    Ψt = ct * JacobiElliptic.E(ψo, kv) - Ht / 2
    return (kv = kv, nv = nv, cθ = cθ, cφ = cφ, ct = ct, Hθ = Hθ, Hφ = Hφ, Ht = Ht,
            Ψθ = Ψθ, Ψφ = Ψφ, Ψt = Ψt)
end

"""
    surface_prep(α, β, θo, a; ro = 1000) -> NamedTuple

The state of the closed-form inversion of the screen point `(α, β)` (Bardeen) for
surfaces at any polar angle: the fields of [`pixel_prep`](@ref) with every ray prepared,
the vortical ones (`η < 0`; `vortical` then marks the ray class) included, the radial
case `rcase ∈ (2, 3, 4)` (four, two or no real roots; case IV occurs for vortical rays
only), `vang` (the vortical polar terms), `co = cos θo` and `pm_o = sign β` of the
normalised frame. A ray with `η = 0` reaches no surface. Consumed by
[`surface_landing`](@ref) and [`image_crossings`](@ref).
"""
@inline function surface_prep(α::T, β, θo, a; ro = T(1000)) where {T}
    sθ = cos(θo) < 0 ? -one(T) : one(T)
    if sθ < 0
        θo = oftype(θo, π) - θo
        β = -β
    end
    sφ = a < 0 ? -one(T) : one(T)
    α = sφ * α
    a = float(abs(a))
    a < 1e-6 && (a = oftype(a, 1e-6))
    rph = kerr_horizon(a)
    λ, η = conserved_quantities(α, β, θo, a)
    roots = radial_roots(λ, η, a)
    u_p, u_m = angular_turning_points(λ, η, a)
    co = cos(θo)
    pm_o = β >= 0 ? one(T) : -one(T)
    vort = η < 0
    z = zero(λ + η + a + β)
    zang = (u_p = z, K = z, Ep = z, PI = z, F_o = z, Ep_o = z, PI_o = z, Hβ = 0, pm_o = z,
            den = z, k = z, u_m = z)
    zv = (kv = z, nv = z, cθ = z, cφ = z, ct = z, Hθ = z, Hφ = z, Ht = z, Ψθ = z, Ψφ = z,
          Ψt = z)
    ang = vort || !(η > 0) ? zang : _angular_parts(β, u_p, u_m, θo, a)
    vang = vort ? _vortical_parts(u_p, u_m, co, a) : zv
    rs = _radial_state(ro, roots, a, λ, η, rph, T, true)
    return (vortical = vort, sφ = sφ, sθ = sθ, a = a, rph = rph, λ = λ, η = η,
            is_case2 = rs.rcase == 2, rcase = rs.rcase, roots = roots, ang = ang, vang = vang,
            co = co, pm_o = pm_o, u_p = u_p, u_m = u_m, kr = rs.kr, Fr = rs.Fr,
            τturn = rs.τturn, τtot = rs.τtot, po = rs.po, pt = rs.pt)
end

# The source terms of the surface of normalised cos θs = c: (Ψθ, Ψφ, Ψt) in the units of
# the observer terms (numerators over `den` for ordinary rays, centred values for vortical
# ones), and whether the ray reaches it.
@inline function _source_terms(prep, c)
    T = typeof(prep.u_p)
    if prep.vortical
        v = prep.vang
        ok = c > 0 && prep.u_m <= c * c <= prep.u_p
        ψ = asin(sqrt(clamp((c * c - prep.u_m) / (prep.u_p - prep.u_m), zero(T), one(T))))
        return (v.cθ * JacobiElliptic.F(ψ, v.kv) - v.Hθ / 2,
                v.cφ * JacobiElliptic.Pi(v.nv, ψ, v.kv) - v.Hφ / 2,
                v.ct * JacobiElliptic.E(ψ, v.kv) - v.Ht / 2), ok
    end
    ok = prep.η > 0 && c * c < prep.u_p
    iszero(c) && return (zero(T), zero(T), zero(T)), ok
    g = prep.ang
    arg = asin(clamp(c / sqrt(g.u_p), -one(T), one(T)))
    Fs = JacobiElliptic.F(arg, g.k)
    return (Fs, JacobiElliptic.Pi(g.u_p, arg, g.k), (JacobiElliptic.E(arg, g.k) - Fs) / (2 * g.k)), ok
end

# (Gθ, Gφ, Gt) after m turning points with the source terms s and the emission sign νθ
# of the normalised frame. For an ordinary ray at the equator (s = 0) these are the
# operations of `_angular_from_parts` plus an exact zero.
@inline function _surface_angular(prep, s, m, νθ)
    if prep.vortical
        v = prep.vang
        return (m * v.Hθ - prep.pm_o * v.Ψθ + νθ * s[1],
                m * v.Hφ - prep.pm_o * v.Ψφ + νθ * s[2],
                m * v.Ht - prep.pm_o * v.Ψt + νθ * s[3])
    end
    g = prep.ang
    return ((2 * m * g.K - g.pm_o * g.F_o + νθ * s[1]) / g.den,
            (2 * m * g.PI - g.pm_o * g.PI_o + νθ * s[2]) / g.den,
            -2 * g.u_p * (2 * m * g.Ep - g.pm_o * g.Ep_o + νθ * s[3]) / g.den)
end

"""
    surface_landing(prep, θs, m::Integer) -> Landing

The landing of the screen point of `prep = surface_prep(…)` on the surface of polar
angle `θs` (rad, `π/2` the equator) after `m` polar turning points: the Mino time
`G(m)` of the polar motion, its radial inversion (sn, cn or sc for cases II, III, IV),
the azimuth and the emission time. `valid` requires a ray that reaches the surface's
latitude (ordinary rays `cos²θs < u₊`; vortical rays the observer's hemisphere and
`u₋ ≤ cos²θs ≤ u₊`), `G(m) > 0`, `G(m)` within the Mino budget and a landing outside
the horizon. `θ` is `θs` and `νθ` the sign of `p^θ` at emission, both of the original
frame. ForwardDiff-generic.
"""
@inline function surface_landing(prep, θs, m::Integer)
    T = typeof(prep.sφ)
    θs = float(θs)
    c = prep.sθ * _surface_cos(θs)
    s, reach = _source_terms(prep, c)
    νθn = (prep.pm_o > 0) == iseven(m) ? 1 : -1
    Gθ, Gφ, Gt = _surface_angular(prep, s, m, νθn)
    νθ = prep.sθ < 0 ? -νθn : νθn
    λo = prep.sφ * prep.λ
    if !(reach && Gθ > 0 && Gθ <= prep.τtot)
        return _landing(prep.rph, T(NaN), T(NaN), 1, false, λo, prep.η, θs, νθ)
    end
    rs, νr = _radius_at(prep, Gθ)
    valid = isfinite(rs) && rs > prep.rph
    I_r, I_φ, I_t = _radial_from_parts(rs, prep.roots, prep.a, prep.λ, prep.η, prep.rcase, νr,
                                       (prep.po, prep.pt))
    φs = -prep.sφ * (I_φ + prep.λ * Gφ) - T(π) / 2     # P3 Eq. 11 (sφ: a < 0 mirror)
    ts = -(I_t + prep.a * prep.a * Gt)                 # P3 Eq. 12
    return _landing(rs, φs, ts, νr, valid, λo, prep.η, θs, νθ)
end

"""
    image_crossings(prep, θs, n::Integer) -> (m₁, m₂, k)

The polar turning-point counts of the `k ∈ (0, 1, 2)` crossings of the surface `θs` that
form the image of order `n` of the screen point of `prep` (in the labelling of Krang, with
every crossing in exactly one order). Let `m₀ = 0` when the first leg of the ray, traced
back from the observer, reaches the surface and `m₀ = 1` otherwise. Ordinary rays and a
surface at most as inclined as the observer's line of sight (`|cos θs| ≤ |cos θo|`, the
equator among them): one crossing `m = n + m₀`. Ordinary rays and a surface closer to
the pole: the pair `{n, n + 1}` when `n ≡ m₀ (mod 2)`, none otherwise. Vortical rays:
the pair `{2n + m₀, 2n + m₀ + 1}`. Crossings whose landing is not valid are kept; the
caller tests `valid`.
"""
@inline function image_crossings(prep, θs, n::Integer)
    c = prep.sθ * _surface_cos(float(θs))
    s, reach = _source_terms(prep, c)
    reach || return (0, 0, 0)
    m0 = first(_surface_angular(prep, s, 0, prep.pm_o > 0 ? 1 : -1)) > 0 ? 0 : 1
    prep.vortical && return (2n + m0, 2n + m0 + 1, 2)
    abs(c) <= abs(prep.co) && return (n + m0, 0, 1)
    return iseven(n - m0) ? (n, n + 1, 2) : (0, 0, 0)
end
