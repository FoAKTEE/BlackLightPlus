"""
    Sources

Semi-analytic plasma source models: Polish doughnut, Page–Thorne thin disk, and
orbiting hot spot, derived equation-first from the published papers. Each model
is a pure `isbits` struct implementing `source_plasma_at(src, st, x)`, returning
the decoded-plasma NamedTuple consumed by the `Radiation` coefficients.
Signature (-,+,+,+), G = c = M = 1, Cartesian Kerr–Schild; axisymmetric
quantities use the Killing vectors xi = d_t and eta = d_phi = (0,-y,x,0), so the
circular-orbit machinery works directly in the CKS frame. AD-generic.
"""
module Sources

using StaticArrays
using LinearAlgebra: dot
using ..Geodesics: metric, inverse_metric, ks_radius, horizon_radius,
                   KerrSchildCartesian,
                   isco_radius, marginally_bound_radius, photon_orbit_radius,
                   keplerian_omega, keplerian_specific_angular_momentum
using ..Constants: CL, ME, KB, HPL

export AbstractSource, source_plasma_at,
       PolishDoughnut, PageThorneDisk, EquatorialHotspot,
       torus_potential, page_thorne_flux, page_thorne_temperature, hotspot_center,
       isco_radius, marginally_bound_radius, photon_orbit_radius,
       keplerian_omega, keplerian_specific_angular_momentum

# ============================================================ interface

"""
    AbstractSource

Base type for every analytic plasma model. A concrete `src <: AbstractSource`
is a pure parameter struct and must implement

    source_plasma_at(src, st, x::SVector{4}) -> NamedTuple

(the grmhd `plasma_at` tuple, see the module docstring).
"""
abstract type AbstractSource end

"""
    source_plasma_at(src::AbstractSource, st, x::SVector{4}) -> NamedTuple

Emitting-plasma state at CKS 4-position `x` for analytic model `src` in spacetime
`st`: `(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon, inside)` with `n_e` [cm^-3],
`theta_e = k T_e/(m_e c^2)` dimensionless, `B` [Gauss], `ucon/ucov` normalized
u.u = -1, and `bcon` a UNIT spacelike field direction (b.b = +1, b.u = 0) — the
synchrotron coefficients take the pitch angle from its direction and the field
strength from the scalar `B`. `inside == false` (zeroed plasma, valid metric)
marks vacuum. Each concrete source provides a method.
"""
function source_plasma_at end

# ============================================================ shared kinematics

# The two axisymmetric Killing vectors in the CKS frame, as contravariant
# SVectors: xi = d_t = (1,0,0,0), eta = d_phi = x d_y - y d_x = (0,-y,x,0).
@inline function _killing_vectors(x::SVector{4,T}) where {T}
    xi  = SVector(one(T), zero(T), zero(T), zero(T))
    eta = SVector(zero(T), -x[3], x[2], zero(T))
    return xi, eta
end

# Killing-projected metric scalars  X_tt = g(xi,xi), X_tp = g(xi,eta),
# X_pp = g(eta,eta). These equal the Boyer–Lindquist g_tt, g_tphi, g_phiphi
# (same geometric Killing vectors) but are evaluated in whatever frame `gcov`
# lives in — here CKS.
@inline function _tphi_scalars(gcov::SMatrix{4,4}, x::SVector{4})
    xi, eta = _killing_vectors(x)
    g_xi  = gcov * xi
    g_eta = gcov * eta
    return dot(xi, g_xi), dot(xi, g_eta), dot(eta, g_eta)
end

# keplerian_omega, keplerian_specific_angular_momentum, isco_radius,
# marginally_bound_radius and photon_orbit_radius are the shared exact-Kerr
# orbital primitives from Geodesics/kerr_orbits.jl, imported above and re-exported.

# Circular 4-velocity u^mu = u^t (xi + Omega eta), normalized u.u = -1.
# `den` = X_tt + 2 Omega X_tp + Omega^2 X_pp = -1/(u^t)^2 must be < 0 for the
# orbit to exist; returns it so callers can gate.
@inline function _circular_velocity(gcov::SMatrix{4,4}, x::SVector{4}, Omega)
    xi, eta = _killing_vectors(x)
    ubar = xi + Omega * eta
    den = dot(ubar, gcov * ubar)
    ut = 1 / sqrt(-den)
    ucon = ut * ubar
    ucov = gcov * ucon
    return ucon, ucov, den
end

# Relativistic effective potential of a constant-l flow, PUBLICATION sign
# (Abramowicz, Jaroszyński & Sikora 1978): W = ln|u_t|, a MINIMUM at the
# pressure-max centre and increasing outward. (Gyoto's getPotential returns the
# opposite sign, -ln|u_t|; we use the publication convention throughout so the
# torus interior is W < W_surface with the centre at the potential minimum.)
# Returns (W, Omega, den) with Omega the flow angular velocity and
# den = X_tt + 2 Omega X_tp + Omega^2 X_pp (< 0 where the flow is defined).
@inline function _constant_l_potential(Xtt, Xtp, Xpp, l0)
    Omega = -(Xtp + l0 * Xtt) / (Xpp + l0 * Xtp)
    num = Xtt + Omega * Xtp
    den = Xtt + 2 * Omega * Xtp + Omega^2 * Xpp
    W = log(abs(num)) - 0.5 * log(abs(den))
    return W, Omega, den
end

# CKS point on the equator (z=0) at Boyer–Lindquist radius r: the KS cylindrical
# radius there is sqrt(r^2 + a^2) (from (x^2+y^2)/(r^2+a^2) + z^2/r^2 = 1).
@inline _equatorial_point(a, r) = SVector(zero(r), sqrt(r^2 + a^2), zero(r), zero(r))

# Toroidal magnetic field DIRECTION for a flow of specific angular momentum l:
# b ∝ l xi + eta (Gyoto's B ∝ l d_t + d_phi), Gram–Schmidt-projected orthogonal
# to u (u.u = -1) and unit-normalized (b.b = +1, spacelike). Only its direction
# feeds the pitch angle; the strength is carried by the scalar B.
@inline function _toroidal_direction(gcov::SMatrix{4,4}, x::SVector{4},
                                     ell, ucon::SVector{4}, ucov::SVector{4})
    xi, eta = _killing_vectors(x)
    bdir = ell * xi + eta
    bu = dot(gcov * bdir, ucon)          # b_dir . u
    bp = bdir + bu * ucon                # project orthogonal to u  (u.u = -1)
    n2 = dot(bp, gcov * bp)
    if !(n2 > 1e-10)
        # On the rotation axis eta -> 0, so bdir ∝ u and the projection collapses
        # to roundoff (n2 ≈ ±0 -> sqrt DomainError / NaN). The toroidal direction
        # is genuinely degenerate there (no azimuth on-axis; pitch angle ill-defined
        # — the scalar |B| carries the weight). Fall back to ANY unit spacelike
        # vector orthogonal to u so bcon stays finite: project a spatial coordinate
        # direction (u is purely timelike here, so it never collapses too).
        T = eltype(bp)
        v = SVector(zero(T), one(T), zero(T), zero(T))
        bp = v + dot(gcov * v, ucon) * ucon
        n2 = dot(bp, gcov * bp)
    end
    return bp / sqrt(n2)
end

# Bracketed bisection for a monotone-on-the-bracket scalar root (host-side torus
# geometry; not on any hot path). Assumes f(lo) and f(hi) straddle zero.
function _bisect(f, lo::Float64, hi::Float64; tol::Float64=1e-12, itmax::Int=200)
    flo = f(lo)
    for _ in 1:itmax
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0
            hi = mid
        else
            lo = mid
            flo = fm
        end
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

# The grmhd-parity plasma tuple, and its vacuum sentinel.
@inline _plasma(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon) =
    (n_e = n_e, theta_e = theta_e, B = B, ucon = ucon, ucov = ucov,
     bcon = bcon, gcov = gcov, gcon = gcon, inside = true)

# Type the zeroed scalars/vectors on the eltype of `gcov` so the vacuum sentinel
# returns the same concrete NamedTuple type as `_plasma` under AD (Dual-typed
# positions) and `source_plasma_at` stays type-stable.
@inline function _vacuum(gcov::SMatrix{4,4,T}, gcon::SMatrix{4,4}) where {T}
    o = zero(T)
    z = zero(SVector{4,T})
    return (n_e = o, theta_e = o, B = o, ucon = z, ucov = z, bcon = z,
            gcov = gcov, gcon = gcon, inside = false)
end

# Gas pressure [erg/cm^3] from a two-temperature-agnostic estimate: the thermal
# pressure of the electrons plus an equal-density ion population at the same
# temperature, p = (1 + f_ion) n_e (Theta_e m_e c^2). f_ion = 1 -> electrons+ions.
@inline _gas_pressure(n_e, theta_e; f_ion::Float64=1.0) =
    (1.0 + f_ion) * n_e * theta_e * (ME * CL * CL)

# Magnetic field magnitude [Gauss] from a plasma-beta parametrization,
# p_mag = B^2/(8 pi) = p_gas / beta  =>  B = sqrt(8 pi p_gas / beta).
@inline _bfield_from_beta(p_gas, beta) = sqrt(8.0 * π * p_gas / beta)

include("polish_doughnut.jl")
include("page_thorne.jl")
include("hotspot.jl")

end # module
