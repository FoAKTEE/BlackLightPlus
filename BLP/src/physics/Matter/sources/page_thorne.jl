# Page & Thorne 1974 (ApJ 191, 499) radiative flux of the Novikov–Thorne thin disk on
# Keplerian circular equatorial geodesics, and the effective temperature it implies;
# closed form and ISCO roots as in Gyoto's PageThorneDisk. With M = 1, x = √r,
# x₀ = √r_ISCO and xᵢ the roots of x³ − 3x + 2a = 0 (Page & Thorne eq. 15n):
#   f(x) = 3/(2x²(x³ − 3x + 2a)) [x − x₀ − (3/2) a ln(x/x₀)
#          − Σᵢ 3(xᵢ − a)²/(xᵢ(xᵢ − xⱼ)(xᵢ − xₖ)) ln((x − xᵢ)/(x₀ − xᵢ))],
#   F(r) = ṁ f(x)/(4π r),
# zero at the ISCO and ∝ r⁻³ at large r; T_eff = (F/σ_SB)^¼.

"Stefan–Boltzmann constant (erg cm⁻² s⁻¹ K⁻⁴)."
const SIGMA_SB = 2 * π^5 * KB^4 / (15 * HPL^3 * CL^2)

"""
    PageThorneDisk{T} <: AnalyticProvider
    PageThorneDisk(a; mdot = 1.0, r_out = 50.0, n_e_max = 1e6, theta_e_max = 1.0,
                   beta = 10.0, scale_height = 0.05, prograde = true)

Page–Thorne thin accretion disk around a Kerr hole of spin `a` (M = 1), as a
volumetric source: a Gaussian layer of relative half-thickness `scale_height` (H/r)
about the equator between the ISCO and `r_out`, with density `n_e_max F/F_peak`
(cm⁻³) and temperature `theta_e_max (F/F_peak)^¼` in the plane, |B| from the plasma
`beta`, on Keplerian circular orbits. `mdot` scales the flux.

A negative spin is the disk of spin `|a|` with the rotation sense flipped: the record
stores `a = |a|` and the flipped `prograde`. The flux closed form is written for a
co-rotating disk; a counter-rotating disk uses it at the effective spin
`a_flux = −a` (Bardeen, Press & Teukolsky 1972, lower sign).

Fields: `a`, `a_flux`, `mdot`, `r_isco`, `r_out`, the roots `x0` (= √r_ISCO), `x1`,
`x2`, `x3`, `f_peak`, the largest flux on 4001 uniform nodes of `[r_ISCO, r_out]` (so
`F/F_peak` may exceed 1 between nodes), `n_e_max`, `theta_e_max`, `beta`,
`scale_height`, `prograde`.

The analytic state ignores the units, electron model and cuts of the parameter set.

# References
Page & Thorne (1974); Bardeen, Press & Teukolsky (1972).
"""
struct PageThorneDisk{T} <: AnalyticProvider
    a::T
    a_flux::T
    mdot::T
    r_isco::T
    r_out::T
    x0::T
    x1::T
    x2::T
    x3::T
    f_peak::T
    n_e_max::T
    theta_e_max::T
    beta::T
    scale_height::T
    prograde::Bool
end

param_type(::PageThorneDisk{T}) where {T} = T

function PageThorneDisk(a::Real; mdot::Real = 1.0, r_out::Real = 50.0,
                        n_e_max::Real = 1e6, theta_e_max::Real = 1.0,
                        beta::Real = 10.0, scale_height::Real = 0.05,
                        prograde::Bool = true)
    a0 = Float64(a)
    pro = a0 >= 0 ? prograde : !prograde
    af = abs(a0)
    r_isco = isco_radius(af; prograde = pro)
    a_flux = pro ? af : -af
    aco3 = acos(a_flux) / 3
    x0 = sqrt(r_isco)
    x1 = 2 * cos(aco3 - π / 3)
    x2 = 2 * cos(aco3 + π / 3)
    x3 = -2 * cos(aco3)

    # flux maximum over [r_ISCO, r_out] on 4001 nodes, the profile normalisation
    f_peak = 0.0
    r = r_isco
    r_hi = Float64(r_out)
    dr = (r_hi - r_isco) / 4000
    for _ in 1:4001
        f = page_thorne_f(a_flux, Float64(mdot), r, r_isco, x0, x1, x2, x3)
        f > f_peak && (f_peak = f)
        r += dr
    end
    f_peak <= 0 && (f_peak = 1.0)

    T = promote_type(typeof(af), Float64)
    return PageThorneDisk{T}(af, a_flux, Float64(mdot), r_isco, r_hi, x0, x1, x2, x3,
                             f_peak, Float64(n_e_max), Float64(theta_e_max),
                             Float64(beta), Float64(scale_height), pro)
end

"""
    page_thorne_coefficient(xi, xj, xk, a)

The coefficient `3(xᵢ − a)²/(xᵢ(xᵢ − xⱼ)(xᵢ − xₖ))` of eq. 15n; zero for `|xᵢ| < 1e-8`,
its limit in the degenerate root `x₂ → 0` of `a = 0`.
"""
@inline function page_thorne_coefficient(xi, xj, xk, a)
    abs(xi) < 1e-8 && return zero(xi)
    return 3 * (xi - a)^2 / (xi * (xi - xj) * (xi - xk))
end

"""
    page_thorne_f(a, mdot, r, r_isco, x0, x1, x2, x3)

The flux `F(r) = ṁ f(x)/(4π r)` of eq. 15n from the effective spin `a` and the roots;
zero at and inside the ISCO and wherever the closed form is not positive.
"""
@inline function page_thorne_f(a, mdot, r, r_isco, x0, x1, x2, x3)
    r <= r_isco && return zero(r)
    x = sqrt(r)
    denom = x^3 - 3 * x + 2 * a
    denom <= 0 && return zero(r)
    c1 = page_thorne_coefficient(x1, x2, x3, a)
    c2 = page_thorne_coefficient(x2, x1, x3, a)
    c3 = page_thorne_coefficient(x3, x1, x2, a)
    bracket = x - x0 - 1.5 * a * log(x / x0) -
              c1 * log((x - x1) / (x0 - x1)) -
              c2 * log((x - x2) / (x0 - x2)) -
              c3 * log((x - x3) / (x0 - x3))
    f = 1.5 / (x^2 * denom) * bracket
    f <= 0 && return zero(r)
    return mdot * f / (4 * π * r)
end

"""
    page_thorne_flux(disk::PageThorneDisk, r) -> F

Time-averaged radiative flux `F(r)` of the disk at Boyer–Lindquist radius `r`, in
G = c = M = 1 units scaled by `mdot`: zero at and inside the ISCO, positive outside,
∝ r⁻³ at large r.

# References
Page & Thorne (1974), eq. 15n.
"""
@inline page_thorne_flux(disk::PageThorneDisk, r) =
    page_thorne_f(disk.a_flux, disk.mdot, r, disk.r_isco, disk.x0, disk.x1, disk.x2, disk.x3)

"""
    page_thorne_temperature(disk::PageThorneDisk, r; flux_scale = 1.0) -> T_eff

Effective blackbody temperature `(flux_scale F(r)/σ_SB)^¼` at Boyer–Lindquist radius
`r`. `flux_scale` converts the geometrized flux to cgs (`c⁶ Ṁ/(G² M²)` for a mass and
accretion rate); the default gives the temperature profile in geometrized flux.
"""
@inline function page_thorne_temperature(disk::PageThorneDisk, r; flux_scale::Real = 1.0)
    F = page_thorne_flux(disk, r)
    F <= 0 && return zero(F)
    return (flux_scale * F / SIGMA_SB)^(1 / 4)
end

"""
    plasma_state(disk::PageThorneDisk, st, x, g, params) -> PlasmaState

The disk plasma at `x`, with `g = (gcov, gcon)` the metric pair of `st` at `x`:
`n_e = n_e_max (F/F_peak) exp(−z²/2H²)` with `H = scale_height r`,
`Θₑ = theta_e_max (F/F_peak)^¼`, `B` from `beta` and the pressure of electrons and
ions at Θₑ, the Keplerian circular flow and its unit toroidal field direction.
`vacuum` for `r ≤ r_ISCO`, `r > r_out`, `|z| > 4H` or zero flux. `sigma` and
`beta_inv` are zero; `params` is not read.
"""
@inline function plasma_state(disk::PageThorneDisk, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(disk, x, gcov)
    r = ks_radius(st, x)
    (r <= disk.r_isco || r > disk.r_out) && return vacuum(T)

    z = x[4]
    sh = disk.scale_height * r
    abs(z) > 4 * sh && return vacuum(T)
    vprof = exp(-z^2 / (2 * sh^2))

    Fn = page_thorne_flux(disk, r) / disk.f_peak
    Fn <= 0 && return vacuum(T)

    n_e = disk.n_e_max * Fn * vprof
    theta_e = disk.theta_e_max * Fn^(1 / 4)
    n_e > 0 || return vacuum(T)
    p_gas = gas_pressure(n_e, theta_e)
    B = bfield_from_beta(p_gas, disk.beta)

    Omega = keplerian_omega(disk.a, r; prograde = disk.prograde)
    u = circular_velocity(gcov, x, Omega)
    u.den < 0 || return vacuum(T)
    ell = keplerian_specific_angular_momentum(disk.a, r; prograde = disk.prograde)
    bcon = toroidal_direction(gcov, x, ell, u.ucon, u.ucov)
    return analytic_state(T, n_e, theta_e, B, u.ucon, u.ucov, bcon, gcov)
end

"""
    domain_radius(disk::PageThorneDisk) -> Float64

The outer edge `r_out`: the disk is vacuum at every larger Kerr-Schild radius.
"""
domain_radius(disk::PageThorneDisk) = Float64(primal(disk.r_out))
