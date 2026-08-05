# AART equatorial g-disk source: the redshift factor g = ν_obs/ν_emit for a
# sub-Keplerian / mixed accretion flow with an inside-ISCO plunging (gGas)
# branch, the Johnson-SU radial emissivity profile, and the observed intensity
# I_obs = profile(r_s) · g^σ for an optically-thin equatorial disk.
# Ported from AART (MIT, © 2024 Cárdenas-Avendaño, Zhu & Lupsasca;
# github.com/iAART/aart, aart_func/{intensity_f.py, rprofs_f.py}); MIT notice
# preserved per its terms. Physics: Gralla & Lupsasca PRD 101 044032 (P1) — the
# sub-Keplerian 4-velocity (Eqs. B32–B52), the redshift factor (Eq. B13), the
# Johnson-SU envelope (Eq. 58; cf. 2008.03879 Eq. 12). Equation tags follow AART.

# Shared exact-Kerr orbital primitives (Geodesics/kerr_orbits.jl); isco_prograde
# is re-exported by Radiation.
using ..Geodesics: kerr_horizon, kerr_delta, kerr_radial_potential,
                   isco_prograde, keplerian_specific_angular_momentum

# ── Kerr equatorial kinematic helpers (M = 1, Boyer–Lindquist) ───────────────

"Π(r) = (r²+a²)² − a²Δ (AART `PIF`, P1 Eq. B6)."
@inline _kerr_pif(r, a) = (r * r + a * a)^2 - a * a * kerr_delta(r, a)

"Contravariant uʳ of pure radial infall (AART `urbar`, P1 Eq. B34b)."
@inline _u_r_infall(r, a) = -sqrt(2 * r * (r * r + a * a)) / (r * r)

"Angular velocity Ω of pure radial infall (AART `Omegabar`, P1 Eq. B32a)."
@inline _omega_infall(r, a) = (2 * a * r) / _kerr_pif(r, a)

"Angular velocity Ω̂ of the sub-Keplerian orbit with specific ℓ (AART `Omegahat`, P1 Eq. B39)."
@inline _omega_subkep(r, a, ℓ) = (a + (1 - 2 / r) * (ℓ - a)) / (_kerr_pif(r, a) / (r * r) - (2 * a * ℓ) / r)

"Contravariant uᵗ of the general (uʳ, Ω) 4-velocity (AART `uttilde`, P1 Eq. B52)."
@inline function _u_t_general(r, a, ur, Ω)
    Δ = kerr_delta(r, a)
    return sqrt((1 + ur * ur * r * r / Δ) / (1 - (r * r + a * a) * Ω * Ω - (2 / r) * (1 - a * Ω)^2))
end

"Orbital energy Ê of the sub-Keplerian flow (AART `Ehat`, P1 Eq. B44a)."
@inline function _E_subkep(r, a, ℓ)
    Δ = kerr_delta(r, a)
    return sqrt(Δ / (_kerr_pif(r, a) / (r * r) - (4 * a * ℓ) / r - (1 - 2 / r) * ℓ * ℓ))
end

"Radial velocity ν̂ of the sub-Keplerian flow (AART `nuhat`, P1 Eq. B45)."
@inline function _nu_subkep(r, a, ℓ, E)
    Δ = kerr_delta(r, a)
    return r / Δ * sqrt(abs(_kerr_pif(r, a) / (r * r) - (4 * a * ℓ) / r - (1 - 2 / r) * ℓ * ℓ - Δ / (E * E)))
end

# Specific angular momentum ℓ̂ of the sub-Keplerian flow (AART `lhat`, P1 Eq. B44b):
# `sub_kep` scales the prograde Keplerian ℓ_K (shared `keplerian_specific_angular_momentum`).
@inline _l_subkep(r, a, sub_kep) = sub_kep * keplerian_specific_angular_momentum(a, r)

# `kerr_radial_potential(r,a,λ,η)` is the source-frame R(r) (AART `Rint`, P2 Eq. 5)
# that sets the p_r sign weight; `isco_prograde` is the prograde ISCO.

# ── redshift factor g = ν_obs / ν_emit ───────────────────────────────────────

# Assemble g from the emitter 4-velocity (uᵗ, uʳ, Ω) and the photon (λ, η, sign
# b of p^r). Common tail of gDisk / gGas (AART, P1 Eq. B13). The p^r term is
# written b·uʳ·√|R| (≡ AART's b·sign(uʳ)·√|R·uʳ²| for real uʳ): pulling uʳ OUT of
# the sqrt keeps the value identical but AD-safe — a Keplerian flow has uʳ≡0, and
# √(R·uʳ²) at uʳ=0 has an ill-defined (0/0) derivative, whereas b·uʳ·√|R| → 0.
@inline function _g_from_velocity(r, a, b, λ, η, ur, Ω)
    Δ = kerr_delta(r, a)
    ut = _u_t_general(r, a, ur, Ω)
    uφ = ut * Ω
    return 1 / (ut * (1 - b * ur * sqrt(abs(kerr_radial_potential(r, a, λ, η))) / Δ / ut - λ * uφ / ut))
end

"""
    gfactor_circular(rs, a, b, λ, η; sub_kep=1, betar=1, betaphi=1) -> g

Redshift factor for a photon emitted OUTSIDE the ISCO by the (generally
sub-Keplerian, mixed radial/azimuthal) circular-ish flow — AART `gDisk`
(P1 Eq. B13). `b` is the sign of the emitted p^r (the `redshift_sign` / νr from
`analytic_emission`). The velocity knobs interpolate between profiles:
`sub_kep` scales the specific angular momentum (1 = Keplerian ℓ), `betaphi`
blends Ω̂ (1) toward the frame-dragging Ω̄ (0), `betar` blends uʳ from 0 (1)
toward pure radial infall (0). Defaults (all 1) = the equatorial Keplerian disk.
"""
@inline function gfactor_circular(rs, a, b, λ, η; sub_kep=1.0, betar=1.0, betaphi=1.0)
    ℓ = _l_subkep(rs, a, sub_kep)
    OH = _omega_subkep(rs, a, ℓ)
    OT = OH + (1 - betaphi) * (_omega_infall(rs, a) - OH)
    ur = (1 - betar) * _u_r_infall(rs, a)
    return _g_from_velocity(rs, a, b, λ, η, ur, OT)
end

"""
    gfactor_plunge(rs, a, b, λ, η; sub_kep=1, betar=1, betaphi=1) -> g

Redshift factor for a photon emitted INSIDE the ISCO by the plunging flow — AART
`gGas` (P1 Eq. B13). The plunging material conserves the ISCO angular momentum
ℓ_ms and energy Ê_ms; its radial velocity ûʳ(r) is set by the geodesic radial
potential (ν̂), continuously → 0 at the ISCO so gGas ≡ gDisk there.
"""
@inline function gfactor_plunge(rs, a, b, λ, η; sub_kep=1.0, betar=1.0, betaphi=1.0)
    isco = isco_prograde(a)
    ℓms = _l_subkep(isco, a, sub_kep)
    OH = _omega_subkep(rs, a, ℓms)
    OT = OH + (1 - betaphi) * (_omega_infall(rs, a) - OH)
    Ems = _E_subkep(isco, a, ℓms)
    urhat = -kerr_delta(rs, a) / (rs * rs) * _nu_subkep(rs, a, ℓms, Ems) * Ems
    ur = urhat + (1 - betar) * (_u_r_infall(rs, a) - urhat)
    return _g_from_velocity(rs, a, b, λ, η, ur, OT)
end

"""
    gfactor_disk(rs, a, b, λ, η; sub_kep=1, betar=1, betaphi=1) -> g

The mixed-flow equatorial redshift factor: circular (`gfactor_circular`) for
`rs ≥ r_ISCO`, plunging (`gfactor_plunge`) for `rs < r_ISCO` — AART's
`bright_radial` gDisk/gGas selector (P1 Eq. B13). Continuous across the ISCO by
construction; `b` is the emitted-p^r sign (νr). In the a→0, all-knobs=1 limit this
is the Schwarzschild circular-orbit redshift g = 1/(uᵗ(1 − λΩ_K)); at large r, g → 1.
"""
@inline function gfactor_disk(rs, a, b, λ, η; sub_kep=1.0, betar=1.0, betaphi=1.0)
    isco = isco_prograde(a)
    return rs >= isco ?
           gfactor_circular(rs, a, b, λ, η; sub_kep=sub_kep, betar=betar, betaphi=betaphi) :
           gfactor_plunge(rs, a, b, λ, η; sub_kep=sub_kep, betar=betar, betaphi=betaphi)
end

# ── Johnson-SU radial emissivity ─────────────────────────────────────────────

"""
    johnson_su_profile(r; mu=0.0, sigma=0.5, gamma=-1.5) -> emissivity

The AART Johnson-SU radial emissivity envelope (P1 Eq. 58; cf. 2008.03879 Eq. 12):
`f(r) = exp(−½(γ + asinh((r−μ)/σ))²) / √((r−μ)² + σ²)`. `mu` sets the location,
`sigma` the width, `gamma` the skewness; strictly positive and integrable. AART's
canonical disk uses γ = −3/2, μ = 1 − √(1−a²), σ = 1/2.
"""
@inline function johnson_su_profile(r; mu=0.0, sigma=0.5, gamma=-1.5)
    z = (r - mu) / sigma
    return exp(-(gamma + asinh(z))^2 / 2) / sqrt((r - mu)^2 + sigma * sigma)
end

# ── observed equatorial intensity  I_obs = profile · g^σ ─────────────────────

"""
    equatorial_disk_intensity(rs, νr, a, λ, η;
        sub_kep=1, betar=1, betaphi=1,
        gamma_p=-1.5, mu_p=1-√(1-a²), sigma_p=0.5,
        redshift_power=3, rin=0.0) -> I_obs

Observed specific intensity of the AART optically-thin equatorial g-disk at
source radius `rs` for a ray with emitted-p^r sign `νr` and conserved
quantities `(λ, η)` (AART `bright_radial`, P1 Eq. 50):

    I_obs = johnson_su_profile(rs) · gfactor_disk(rs)^redshift_power.

`redshift_power` is AART's `gfactor` exponent (σ = 3 for a frequency-integrated
optically-thin disk: g³ combines the g² solid-angle boost with the g¹ intensity
Lorentz-invariance I_ν/ν³). Zero inside the horizon or `rs < rin`.
"""
@inline function equatorial_disk_intensity(rs, νr, a, λ, η;
                                           sub_kep=1.0, betar=1.0, betaphi=1.0,
                                           gamma_p=-1.5, mu_p=1 - sqrt(1 - a * a),
                                           sigma_p=0.5, redshift_power=3.0, rin=0.0)
    rph = kerr_horizon(a)
    (rs <= rph || rs < rin) && return zero(promote_type(typeof(rs), typeof(a), typeof(λ), typeof(η)))
    g = gfactor_disk(rs, a, νr, λ, η; sub_kep=sub_kep, betar=betar, betaphi=betaphi)
    prof = johnson_su_profile(rs; mu=mu_p, sigma=sigma_p, gamma=gamma_p)
    return prof * g^redshift_power
end

# ── isbits carrier for the render seam ───────────────────────────────────────

"""
    EquatorialDisk(a; sub_kep=1, betar=1, betaphi=1,
                   gamma_p=-1.5, mu_p=1-√(1-a²), sigma_p=0.5,
                   redshift_power=3, rin=0.0)

`isbits` bundle of the AART g-disk knobs, threaded through `render_analytic`
(and the `equatorial_intensity` seam) so the analytic tracer renders the g-disk
photon-ring image. `equatorial_disk_intensity(d, rs, νr, a, λ, η)` evaluates the
observed intensity for this disk. AART's canonical M87* case is
`EquatorialDisk(0.94)`.
"""
struct EquatorialDisk{T}
    sub_kep::T
    betar::T
    betaphi::T
    gamma_p::T
    mu_p::T
    sigma_p::T
    redshift_power::T
    rin::T
end

function EquatorialDisk(a; sub_kep=1.0, betar=1.0, betaphi=1.0,
                        gamma_p=-1.5, mu_p=1 - sqrt(1 - a * a), sigma_p=0.5,
                        redshift_power=3.0, rin=0.0)
    T = promote_type(typeof(sub_kep), typeof(betar), typeof(betaphi),
                     typeof(gamma_p), typeof(mu_p), typeof(sigma_p),
                     typeof(redshift_power), typeof(rin), Float64)
    return EquatorialDisk{T}(T(sub_kep), T(betar), T(betaphi), T(gamma_p),
                             T(mu_p), T(sigma_p), T(redshift_power), T(rin))
end

@inline function equatorial_disk_intensity(d::EquatorialDisk, rs, νr, a, λ, η)
    return equatorial_disk_intensity(rs, νr, a, λ, η;
                                     sub_kep=d.sub_kep, betar=d.betar, betaphi=d.betaphi,
                                     gamma_p=d.gamma_p, mu_p=d.mu_p, sigma_p=d.sigma_p,
                                     redshift_power=d.redshift_power, rin=d.rin)
end
