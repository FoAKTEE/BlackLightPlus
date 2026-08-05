# Schnittman & Bertschinger 2004 (ApJ 606, 1098) orbiting Gaussian hot spot: a
# localized overdensity on a circular equatorial Keplerian orbit (the standard
# Sgr A* / M87* flare model). Gyoto lib/EquatorialHotSpot.C pins the conventions
# (Gaussian in separation with sigma = R_spot, 4 R_spot cutoff; equation-first,
# no code ported). The centre orbits at Omega_K = ±1/(r^{3/2} ± a): at coordinate
# time t (= x[1]) its azimuth is phi_c = phi0 + Omega_K (t - t0), CKS position
# (R_c cos phi_c, R_c sin phi_c, 0) with R_c = sqrt(r_orbit^2+a^2). The overdensity
# is Gaussian in the PROPER separation ds^2 = (g_{mu nu} + u_mu u_nu) dx^mu dx^nu
# in the local fluid rest frame, so the spot is round in the physical rest space.

"""
    EquatorialHotspot(a; r_orbit=6.0, t0=0.0, phi0=0.0, R_spot=0.5,
                      n_e0=1e6, theta_e0=50.0, B0=10.0,
                      ambient_n_e=0.0, ambient_theta_e=0.0, ambient_B=0.0,
                      prograde=true)

Gaussian hot spot on a circular equatorial Keplerian orbit at Boyer–Lindquist
radius `r_orbit`, phase `phi0` at time `t0`. `R_spot` is the Gaussian sigma
(proper units); `n_e0`/`theta_e0`/`B0` are the peak spot density [cm^-3],
temperature and field [Gauss]. A uniform `ambient_*` background (default off)
can be superposed. The struct is `isbits` and AD-generic: `source_plasma_at`
differentiates cleanly w.r.t. `r_orbit` (and the other parameters).
"""
struct EquatorialHotspot{T} <: AbstractSource
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

function EquatorialHotspot(a; r_orbit=6.0, t0=0.0, phi0=0.0, R_spot=0.5,
                           n_e0=1e6, theta_e0=50.0, B0=10.0,
                           ambient_n_e=0.0, ambient_theta_e=0.0, ambient_B=0.0,
                           prograde::Bool=true)
    T = promote_type(typeof(a), typeof(r_orbit), typeof(t0), typeof(phi0),
                     typeof(R_spot), typeof(n_e0), typeof(theta_e0), typeof(B0),
                     typeof(ambient_n_e), typeof(ambient_theta_e), typeof(ambient_B),
                     Float64)
    return EquatorialHotspot{T}(T(a), T(r_orbit), T(t0), T(phi0), T(R_spot),
                                T(n_e0), T(theta_e0), T(B0), T(ambient_n_e),
                                T(ambient_theta_e), T(ambient_B), prograde)
end

"""
    hotspot_center(src::EquatorialHotspot, t) -> (x_c, y_c, z_c)

CKS position of the spot centre at coordinate time `t`
(phi_c = phi0 + Omega_K (t - t0), R_c = sqrt(r_orbit^2 + a^2), equatorial).
"""
@inline function hotspot_center(src::EquatorialHotspot, t)
    Omega = keplerian_omega(src.a, src.r_orbit; prograde=src.prograde)
    phi = src.phi0 + Omega * (t - src.t0)
    Rc = sqrt(src.r_orbit^2 + src.a^2)
    return Rc * cos(phi), Rc * sin(phi), zero(t)
end

function source_plasma_at(src::EquatorialHotspot, st, x::SVector{4})
    gcov = metric(st, x)
    gcon = inverse_metric(st, x)
    r = ks_radius(st, x)

    # local Keplerian rest frame at the field point (normalized u.u = -1).
    Omega = keplerian_omega(src.a, r; prograde=src.prograde)
    ucon, ucov, den = _circular_velocity(gcov, x, Omega)
    den < 0 || return _vacuum(gcov, gcon)                        # no circular orbit here

    xc, yc, _ = hotspot_center(src, x[1])
    dx = SVector(zero(eltype(x)), x[2] - xc, x[3] - yc, x[4])    # spatial separation

    # proper separation^2 in the rest frame: h_{mu nu} dx dx, h = g + u u.
    udx = dot(ucov, dx)
    ds2 = dot(dx, gcov * dx) + udx^2
    g = exp(-ds2 / (2 * src.R_spot^2))

    has_ambient = src.ambient_n_e > 0
    inside_spot = ds2 < (4 * src.R_spot)^2                       # S&B 2004 cutoff
    (inside_spot || has_ambient) || return _vacuum(gcov, gcon)

    spotw = inside_spot ? g : zero(g)
    n_e = src.ambient_n_e + src.n_e0 * spotw
    n_e > 0 || return _vacuum(gcov, gcon)
    # density-weighted blend of spot and ambient temperature / field.
    wsum = src.ambient_n_e + src.n_e0 * spotw
    theta_e = (src.ambient_n_e * src.ambient_theta_e + src.n_e0 * spotw * src.theta_e0) / wsum
    B = (src.ambient_n_e * src.ambient_B + src.n_e0 * spotw * src.B0) / wsum

    ell = keplerian_specific_angular_momentum(src.a, r; prograde=src.prograde)
    bcon = _toroidal_direction(gcov, x, ell, ucon, ucov)
    return _plasma(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon)
end
