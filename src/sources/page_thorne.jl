# Page & Thorne 1974 (ApJ 191, 499) relativistic thin-disk radiative flux and the
# effective blackbody temperature it implies (Novikov–Thorne geometrically-thin,
# optically-thick disk on Keplerian circular equatorial geodesics). Gyoto
# lib/PageThorneDisk.C pins the closed form and ISCO x-roots (equation-first;
# no code ported). Flux (Page & Thorne eq. 15n, M=1, x = sqrt(r)):
#   f(x) = 3/(2 x^2 (x^3 - 3 x + 2 a)) * [ x - x0 - (3/2) a ln(x/x0)
#          - sum_i 3 (x_i - a)^2 / (x_i (x_i - x_j)(x_i - x_k)) ln((x-x_i)/(x0-x_i)) ]
#   F(r) = mdot * f(x) / (4 pi r);   x0 = sqrt(r_ISCO), x_i roots of x^3 - 3x + 2a = 0.
# F=0 at the ISCO, ~r^-3 at large r (finite luminosity); T_eff = (F/sigma_SB)^{1/4}.

const _SIGMA_SB = 2 * π^5 * KB^4 / (15 * HPL^3 * CL^2)   # Stefan–Boltzmann [cgs]

"""
    PageThorneDisk(a; mdot=1.0, r_out=50.0, n_e_max=1e6, theta_e_max=1.0,
                   beta=10.0, scale_height=0.05, prograde=true)

Page–Thorne thin accretion disk around a Kerr hole of spin `a`. `mdot` scales
the flux; `r_out` is the outer edge of the volumetric model. The volumetric
plasma is a thin Gaussian layer of relative half-thickness `scale_height` (H/r)
about the equator, with peak density `n_e_max` [cm^-3] and peak temperature
`theta_e_max` located at the flux maximum, and |B| set by plasma-`beta`. The
ISCO x-roots and the flux peak are precomputed; the struct is `isbits`.
"""
struct PageThorneDisk{T} <: AbstractSource
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

function PageThorneDisk(a::Real; mdot::Real=1.0, r_out::Real=50.0,
                        n_e_max::Real=1e6, theta_e_max::Real=1.0,
                        beta::Real=10.0, scale_height::Real=0.05,
                        prograde::Bool=true)
    # Package convention: nonnegative spin `a` + a `prograde` flag. A signed spin
    # a<0 is the SAME disk as |a| with the rotation sense flipped, so normalize to
    # (|a|, flipped flag) — this also keeps the flux closed form below well-posed
    # (its x-roots must be consistent with the ISCO).
    a0 = Float64(a)
    pro = a0 >= 0 ? prograde : !prograde
    af = abs(a0)
    r_isco = isco_radius(af; prograde=pro)
    # The Page–Thorne 15n closed form (roots of x^3-3x+2a=0, the 1.5·a·ln term and
    # the (x_i-a)^2 coefficients) is written for a CO-rotating disk. A counter-
    # rotating disk maps onto it by a -> -a throughout (Bardeen–Press–Teukolsky
    # 1972 lower sign): use this effective spin for the roots AND the flux.
    a_flux = pro ? af : -af
    aco3 = acos(a_flux) / 3
    x0 = sqrt(r_isco)
    x1 = 2 * cos(aco3 - π / 3)
    x2 = 2 * cos(aco3 + π / 3)
    x3 = -2 * cos(aco3)

    # peak flux over the disk (F=0 at the ISCO, rises then falls) for normalizing
    # the density/temperature profiles.
    f_peak = 0.0
    r = r_isco
    r_hi = Float64(r_out)
    dr = (r_hi - r_isco) / 4000
    for _ in 1:4001
        f = _pt_flux(a_flux, Float64(mdot), r, r_isco, x0, x1, x2, x3)
        f > f_peak && (f_peak = f)
        r += dr
    end
    f_peak <= 0 && (f_peak = 1.0)

    T = promote_type(typeof(af), Float64)
    return PageThorneDisk{T}(af, a_flux, Float64(mdot), r_isco, r_hi, x0, x1, x2, x3,
                             f_peak, Float64(n_e_max), Float64(theta_e_max),
                             Float64(beta), Float64(scale_height), pro)
end

# Coefficient c_i = 3 (x_i - a)^2 / (x_i (x_i - x_j)(x_i - x_k)); guarded so the
# a=0 degeneracy x2 -> 0 gives its correct vanishing limit instead of 0/0.
@inline function _pt_coef(xi, xj, xk, a)
    abs(xi) < 1e-8 && return zero(xi)
    return 3 * (xi - a)^2 / (xi * (xi - xj) * (xi - xk))
end

"Page–Thorne flux F(r) (natural units) from the precomputed roots."
@inline function _pt_flux(a, mdot, r, r_isco, x0, x1, x2, x3)
    r <= r_isco && return zero(r)
    x = sqrt(r)
    denom = x^3 - 3 * x + 2 * a
    denom <= 0 && return zero(r)
    c1 = _pt_coef(x1, x2, x3, a)
    c2 = _pt_coef(x2, x1, x3, a)
    c3 = _pt_coef(x3, x1, x2, a)
    bracket = x - x0 - 1.5 * a * log(x / x0) -
              c1 * log((x - x1) / (x0 - x1)) -
              c2 * log((x - x2) / (x0 - x2)) -
              c3 * log((x - x3) / (x0 - x3))
    f = 1.5 / (x^2 * denom) * bracket
    f <= 0 && return zero(r)
    return mdot * f / (4 * π * r)
end

"""
    page_thorne_flux(disk::PageThorneDisk, r) -> F(r)

Time-averaged radiative flux F(r) of the Page–Thorne disk at Boyer–Lindquist
radius `r` (natural G=c=M=1 units, scaled by `mdot`). Zero at and inside the
ISCO, positive outside, ~r^-3 at large r.
"""
@inline page_thorne_flux(disk::PageThorneDisk, r) =
    _pt_flux(disk.a_flux, disk.mdot, r, disk.r_isco, disk.x0, disk.x1, disk.x2, disk.x3)

"""
    page_thorne_temperature(disk::PageThorneDisk, r; flux_scale=1.0) -> T_eff

Effective blackbody temperature T = (flux_scale * F(r) / sigma_SB)^{1/4}.
`flux_scale` converts the natural-unit flux to cgs (= c^6/(G^2 M^2) * Mdot_cgs
for a chosen mass and accretion rate); the default returns the natural-unit
temperature shape (0 at the ISCO, peaked, -> 0 at large r).
"""
@inline function page_thorne_temperature(disk::PageThorneDisk, r; flux_scale::Real=1.0)
    F = page_thorne_flux(disk, r)
    F <= 0 && return zero(F)
    return (flux_scale * F / _SIGMA_SB)^(1 / 4)
end

function source_plasma_at(disk::PageThorneDisk, st, x::SVector{4})
    gcov = metric(st, x)
    gcon = inverse_metric(st, x)
    r = ks_radius(st, x)
    (r <= disk.r_isco || r > disk.r_out) && return _vacuum(gcov, gcon)

    # thin Gaussian layer about the equator (CKS z), half-thickness H = h*r.
    z = x[4]
    sh = disk.scale_height * r
    abs(z) > 4 * sh && return _vacuum(gcov, gcon)
    vprof = exp(-z^2 / (2 * sh^2))

    Fn = page_thorne_flux(disk, r) / disk.f_peak                 # in (0,1]
    Fn <= 0 && return _vacuum(gcov, gcon)

    n_e = disk.n_e_max * Fn * vprof
    theta_e = disk.theta_e_max * Fn^(1 / 4)                      # T ∝ F^{1/4}
    n_e > 0 || return _vacuum(gcov, gcon)
    p_gas = _gas_pressure(n_e, theta_e)
    B = _bfield_from_beta(p_gas, disk.beta)

    Omega = keplerian_omega(disk.a, r; prograde=disk.prograde)
    ucon, ucov, den = _circular_velocity(gcov, x, Omega)
    den < 0 || return _vacuum(gcov, gcon)
    ell = keplerian_specific_angular_momentum(disk.a, r; prograde=disk.prograde)
    bcon = _toroidal_direction(gcov, x, ell, ucon, ucov)
    return _plasma(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon)
end
