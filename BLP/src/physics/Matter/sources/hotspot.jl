# Orbiting Gaussian hot spot of Schnittman & Bertschinger 2004 (ApJ 606, 1098): an
# overdensity on a circular equatorial Keplerian orbit, conventions as in Gyoto's
# EquatorialHotSpot (σ = R_spot, cut at 4 R_spot). At coordinate time t = x⁰ the centre
# sits at azimuth φ_c = φ₀ + Ω_K (t − t₀), CKS position (R_c cos φ_c, R_c sin φ_c, 0)
# with R_c = √(r_orbit² + a²). The overdensity is Gaussian in the proper separation
# ds² = (g_μν + u_μ u_ν) dx^μ dx^ν of the local fluid rest frame, so the spot is round
# in the rest space of the flow.

"""
    EquatorialHotspot{T} <: AnalyticProvider
    EquatorialHotspot(a; r_orbit = 6.0, t0 = 0.0, phi0 = 0.0, R_spot = 0.5,
                      n_e0 = 1e6, theta_e0 = 50.0, B0 = 10.0,
                      ambient_n_e = 0.0, ambient_theta_e = 0.0, ambient_B = 0.0,
                      prograde = true)

Gaussian hot spot around a Kerr hole of spin `a` (M = 1), on the circular equatorial
Keplerian orbit of Boyer–Lindquist radius `r_orbit`, at azimuth `phi0` at coordinate
time `t0`. `R_spot` is the Gaussian width in proper length; `n_e0` (cm⁻³), `theta_e0`
and `B0` (G) are the peak density, temperature and field. A uniform background
`ambient_n_e`, `ambient_theta_e`, `ambient_B` is superposed when `ambient_n_e > 0`.

Every field shares the number type `T`, the promotion of the arguments with
`Float64`, so a `ForwardDiff.Dual` parameter (`r_orbit`, …) flows through the state.

The analytic state ignores the units, electron model and cuts of the parameter set.

# References
Schnittman & Bertschinger (2004).
"""
struct EquatorialHotspot{T} <: AnalyticProvider
    a::T
    r_orbit::T
    t0::T
    phi0::T
    R_spot::T
    n_e0::T
    theta_e0::T
    B0::T
    ambient_n_e::T
    ambient_theta_e::T
    ambient_B::T
    prograde::Bool
end

param_type(::EquatorialHotspot{T}) where {T} = T

function EquatorialHotspot(a; r_orbit = 6.0, t0 = 0.0, phi0 = 0.0, R_spot = 0.5,
                           n_e0 = 1e6, theta_e0 = 50.0, B0 = 10.0,
                           ambient_n_e = 0.0, ambient_theta_e = 0.0, ambient_B = 0.0,
                           prograde::Bool = true)
    T = promote_type(typeof(a), typeof(r_orbit), typeof(t0), typeof(phi0),
                     typeof(R_spot), typeof(n_e0), typeof(theta_e0), typeof(B0),
                     typeof(ambient_n_e), typeof(ambient_theta_e), typeof(ambient_B),
                     Float64)
    return EquatorialHotspot{T}(T(a), T(r_orbit), T(t0), T(phi0), T(R_spot),
                                T(n_e0), T(theta_e0), T(B0), T(ambient_n_e),
                                T(ambient_theta_e), T(ambient_B), prograde)
end

"""
    hotspot_center(src::EquatorialHotspot, t) -> (x, y, z)

CKS position of the spot centre at coordinate time `t`: azimuth
`φ_c = phi0 + Ω_K (t − t0)` on the equatorial circle of cylindrical radius
`√(r_orbit² + a²)`.
"""
@inline function hotspot_center(src::EquatorialHotspot, t)
    Omega = keplerian_omega(src.a, src.r_orbit; prograde = src.prograde)
    phi = src.phi0 + Omega * (t - src.t0)
    Rc = sqrt(src.r_orbit^2 + src.a^2)
    return (x = Rc * cos(phi), y = Rc * sin(phi), z = zero(t))
end

"""
    plasma_state(src::EquatorialHotspot, st, x, g, params) -> PlasmaState

The spot plasma at `x`, with `g = (gcov, gcon)` the metric pair of `st` at `x`. The
fluid moves on the Keplerian circular orbit through `x`, with the unit toroidal field
direction; the spot weight is `G = exp(−ds²/2R_spot²)` for `ds² < (4 R_spot)²`, else
zero, with `ds²` the rest-frame proper separation from the centre at `t = x⁰`. Density
`ambient_n_e + n_e0 G`; temperature and field the density-weighted blends of the
ambient and spot values. `vacuum` outside the cut without ambient plasma or at zero
density. Where the circular flow at the Keplerian Ω of the Kerr-Schild radius of `x`
is not timelike (`den > 0` in `circular_velocity`, as inside the photon orbit), the
square root of the four-velocity throws a `DomainError`. `sigma` and `beta_inv` are
zero; `params` is not read.
"""
@inline function plasma_state(src::EquatorialHotspot, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(src, x, gcov)
    r = ks_radius(st, x)

    Omega = keplerian_omega(src.a, r; prograde = src.prograde)
    u = circular_velocity(gcov, x, Omega)
    u.den < 0 || return vacuum(T)

    c = hotspot_center(src, x[1])
    dx = SVector(zero(eltype(x)), x[2] - c.x, x[3] - c.y, x[4])

    # rest-frame proper separation: h_μν dx^μ dx^ν with h = g + u u
    udx = dot(u.ucov, dx)
    ds2 = dot(dx, gcov * dx) + udx^2
    gauss = exp(-ds2 / (2 * src.R_spot^2))

    has_ambient = src.ambient_n_e > 0
    inside_spot = ds2 < (4 * src.R_spot)^2
    (inside_spot || has_ambient) || return vacuum(T)

    spotw = inside_spot ? gauss : zero(gauss)
    n_e = src.ambient_n_e + src.n_e0 * spotw
    n_e > 0 || return vacuum(T)
    wsum = src.ambient_n_e + src.n_e0 * spotw
    theta_e = (src.ambient_n_e * src.ambient_theta_e + src.n_e0 * spotw * src.theta_e0) / wsum
    B = (src.ambient_n_e * src.ambient_B + src.n_e0 * spotw * src.B0) / wsum

    ell = keplerian_specific_angular_momentum(src.a, r; prograde = src.prograde)
    bcon = toroidal_direction(gcov, x, ell, u.ucon, u.ucov)
    return analytic_state(T, n_e, theta_e, B, u.ucon, u.ucov, bcon, gcov)
end

"""
    domain_radius(src::EquatorialHotspot) -> Float64

`Inf` when ambient plasma fills the domain. Otherwise the smallest integer above
`√(r_orbit² + a²) + 4 R_spot`: the Kerr-Schild spatial metric `δ_ij + f l_i l_j`
(`f ≥ 0`) dominates the Euclidean one and `(u·dx)² ≥ 0`, so the cut `ds < 4 R_spot`
confines the spot to a Euclidean distance below `4 R_spot` from a centre on the
circle of radius `√(r_orbit² + a²)`, and the Kerr-Schild radius never exceeds the
Euclidean one.
"""
function domain_radius(src::EquatorialHotspot)
    src.ambient_n_e > 0 && return Inf
    return radius_above(sqrt(src.r_orbit^2 + src.a^2) + 4 * src.R_spot)
end
