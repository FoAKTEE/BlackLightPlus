# Constant-angular-momentum equilibrium torus ("Polish doughnut"): Abramowicz,
# Jaroszyński & Sikora 1978 (A&A 63, 221) with the polytropic thermodynamics of
# Komissarov 2006 (MNRAS 368, 993), conventions as in Gyoto's PolishDoughnut. A
# barotropic flow of constant ℓ = −u_φ/u_t has isobars on the equipotentials of
# W = ln|u_t|, minimal at the pressure-maximum centre with a saddle at the cusp, both
# where ℓ = ℓ_K(r); the torus is the closed region W < W_surface between r_cusp and
# r_out. A polytrope p = κ ρ^Γ (Γ = 1 + 1/n) with p ≪ ρc² gives, at the depth
# w = (W_surface − W)/(W_surface − W_centre) ∈ (0, 1]: n_e ∝ wⁿ, Θₑ ∝ w, |B| from β.

"""
    PolishDoughnut{T} <: AnalyticProvider
    PolishDoughnut(a; l0 = nothing, lambda = nothing, r_center = nothing,
                   n_e_center = 1e7, theta_e_center = 50.0, polytrope_n = 1.5,
                   beta = 10.0, W_surface = nothing, fill_factor = 0.9, prograde = true)

Constant-ℓ equilibrium torus around a Kerr hole of spin `a` (M = 1).

The specific angular momentum comes from the first given of `r_center`
(`ℓ = ℓ_K(r_center)`), `l0`, or `lambda ∈ (0, 1)` (`ℓ = λ (ℓ_mb − ℓ_ms) + ℓ_ms`),
default `lambda = 0.5`; a torus exists only for `|ℓ|` beyond the marginally stable
`|ℓ_ms|` toward `ℓ_mb`, otherwise `ArgumentError`. `n_e_center` (cm⁻³) and
`theta_e_center` are the central density and temperature, `polytrope_n` the
polytropic index, `beta` the plasma beta fixing |B|. The surface is `W_surface` if
given, else the cusp potential (Roche-lobe filling) when the cusp is bound, else
`fill_factor` times the central potential so that the outer edge stays finite.
`prograde` selects the rotation sense.

Fields: `a`, `l0`, `polytrope_n`, `beta`, `n_e_center`, `theta_e_center`, the
Boyer–Lindquist radii `r_cusp`, `r_center`, `r_out` of the cusp, the centre and the
outer equatorial edge, the potentials `W_surface` and `W_center`, and `prograde`.

The analytic state ignores the units, electron model and cuts of the parameter set.

# References
Abramowicz, Jaroszyński & Sikora (1978); Komissarov (2006).
"""
struct PolishDoughnut{T} <: AnalyticProvider
    a::T
    l0::T
    polytrope_n::T
    beta::T
    n_e_center::T
    theta_e_center::T
    r_cusp::T
    r_center::T
    r_out::T
    W_surface::T
    W_center::T
    prograde::Bool
end

param_type(::PolishDoughnut{T}) where {T} = T

function PolishDoughnut(a::Real; l0 = nothing, lambda = nothing, r_center = nothing,
                        n_e_center::Real = 1e7, theta_e_center::Real = 50.0,
                        polytrope_n::Real = 1.5, beta::Real = 10.0,
                        W_surface = nothing, fill_factor::Real = 0.9, prograde::Bool = true)
    af = Float64(a)
    st = KerrSchildCartesian(af)
    r_isco = isco_radius(af; prograde = prograde)
    r_ph = photon_orbit_radius(af; prograde = prograde)
    lK(r) = keplerian_specific_angular_momentum(af, r; prograde = prograde)
    l_ms = lK(r_isco)
    l_mb = lK(marginally_bound_radius(af; prograde = prograde))

    local l0f, rc
    if r_center !== nothing
        rc = Float64(r_center)
        l0f = lK(rc)
    elseif l0 !== nothing
        l0f = Float64(l0)
        rc = bisect(r -> lK(r) - l0f, r_isco, 1.0e5)          # outer (stable) root
    else
        lam = lambda === nothing ? 0.5 : Float64(lambda)
        l0f = lam * (l_mb - l_ms) + l_ms
        rc = bisect(r -> lK(r) - l0f, r_isco, 1.0e5)
    end

    # ℓ_K(r) is extremal (ℓ_ms) at the ISCO, so ℓ_K(r) = ℓ has the cusp and centre
    # roots only for |ℓ| at or beyond |ℓ_ms| toward ℓ_mb; overflow (|ℓ| ≥ |ℓ_mb|) is
    # closed by `fill_factor` below.
    toward_mb = sign(l_mb - l_ms)
    (l0f - l_ms) * toward_mb > 0 || throw(ArgumentError(
        "PolishDoughnut: l0=$(l0f) admits no equilibrium torus — need |l0| beyond the " *
        "marginally-stable l_ms=$(l_ms) toward l_mb=$(l_mb) " *
        "($(prograde ? "prograde" : "retrograde") branch)."))

    r_cusp = bisect(r -> lK(r) - l0f, r_ph + 1e-6, r_isco)

    function Weq(r)
        x = equatorial_point(af, r)
        X = tphi_scalars(metric(st, x), x)
        return constant_l_potential(X.tt, X.tp, X.pp, l0f).W
    end
    W_center = Weq(rc)
    W_cusp = Weq(r_cusp)
    # A bound cusp (W_cusp < 0, ℓ < ℓ_mb) is the surface; an unbound one never closes,
    # and the surface is taken `fill_factor` down the central well instead.
    Wsurf = W_surface !== nothing ? Float64(W_surface) :
            (W_cusp < 0 ? W_cusp : fill_factor * W_center)

    # W → 0⁻ as r → ∞ and W_surface < 0, so the outer edge lies in (r_center, ∞).
    r_out = bisect(r -> Weq(r) - Wsurf, rc, 1.0e5)

    T = promote_type(typeof(af), Float64)
    return PolishDoughnut{T}(af, l0f, Float64(polytrope_n), Float64(beta),
                             Float64(n_e_center), Float64(theta_e_center),
                             r_cusp, rc, r_out, Wsurf, W_center, prograde)
end

"""
    torus_potential(src::PolishDoughnut, st, x::SVector{4}) -> W

Effective potential `W = ln|u_t|` of the torus flow at the CKS position `x` in the
spacetime `st` (publication sign: minimal at the pressure-maximum centre); `+Inf`
where the constant-ℓ flow is undefined.
"""
@inline function torus_potential(src::PolishDoughnut, st, x::SVector{4})
    gcov = metric(st, x)
    X = tphi_scalars(gcov, x)
    p = constant_l_potential(X.tt, X.tp, X.pp, src.l0)
    return p.den < 0 ? p.W : convert(typeof(p.W), Inf)
end

"""
    plasma_state(src::PolishDoughnut, st, x, g, params) -> PlasmaState

The torus plasma at `x`, with `g = (gcov, gcon)` the metric pair of `st` at `x`:
`n_e = n_e_center wⁿ`, `Θₑ = theta_e_center w`, `B` from `beta` and the pressure of
electrons and ions at Θₑ, the circular constant-ℓ flow and its unit toroidal field
direction. `vacuum` inside the cusp radius, where the flow is undefined, and outside
the surface (`w ≤ 0`). `sigma` and `beta_inv` are zero; `params` is not read.
"""
@inline function plasma_state(src::PolishDoughnut, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(src, x, gcov)
    r = ks_radius(st, x)
    r < src.r_cusp && return vacuum(T)

    X = tphi_scalars(gcov, x)
    p = constant_l_potential(X.tt, X.tp, X.pp, src.l0)
    p.den < 0 || return vacuum(T)

    w = (src.W_surface - p.W) / (src.W_surface - src.W_center)
    w > 0 || return vacuum(T)
    w = min(w, one(w))

    n_e = src.n_e_center * w^src.polytrope_n
    theta_e = src.theta_e_center * w
    p_gas = gas_pressure(n_e, theta_e)
    B = bfield_from_beta(p_gas, src.beta)

    u = circular_velocity(gcov, x, p.Omega)
    bcon = toroidal_direction(gcov, x, src.l0, u.ucon, u.ucov)
    return analytic_state(T, n_e, theta_e, B, u.ucon, u.ucov, bcon, gcov)
end

"""
    domain_radius(src::PolishDoughnut) -> Float64

The smallest integer above the outer equatorial edge `r_out`, the largest radius of
the closed surface `W = W_surface`; the margin covers the bisection tolerance.
"""
domain_radius(src::PolishDoughnut) = radius_above(src.r_out)
