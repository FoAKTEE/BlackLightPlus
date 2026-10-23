# The equatorial g-disk: redshift factor g = ν_obs/ν_emit of a sub-Keplerian, mixed
# radial/azimuthal flow with a plunging branch inside the ISCO, the Johnson-SU radial
# emissivity, and the observed intensity I_obs = profile(r_s) g^σ of an optically thin
# equatorial disk; Kerr, M = 1, Boyer–Lindquist. Follows AART (MIT, © 2024
# Cárdenas-Avendaño, Zhu & Lupsasca); see NOTICE. Physics: Gralla & Lupsasca PRD 101
# 044032 (P1), flow four-velocity Eqs. B32–B52, redshift Eq. B13, Johnson-SU envelope
# Eq. 58; equation tags and the AART names in the helper docstrings follow AART.

"Π(r) = (r² + a²)² − a²Δ (AART `PIF`, P1 Eq. B6)."
@inline kerr_pif(r, a) = (r * r + a * a)^2 - a * a * kerr_delta(r, a)

"Contravariant uʳ of pure radial infall (AART `urbar`, P1 Eq. B34b)."
@inline u_r_infall(r, a) = -sqrt(2 * r * (r * r + a * a)) / (r * r)

"Angular velocity Ω̄ of pure radial infall (AART `Omegabar`, P1 Eq. B32a)."
@inline omega_infall(r, a) = (2 * a * r) / kerr_pif(r, a)

"Angular velocity Ω̂ of the sub-Keplerian orbit of specific angular momentum ℓ (AART `Omegahat`, P1 Eq. B39)."
@inline omega_subkep(r, a, ℓ) = (a + (1 - 2 / r) * (ℓ - a)) / (kerr_pif(r, a) / (r * r) - (2 * a * ℓ) / r)

"Contravariant uᵗ of the four-velocity with radial component `ur` and angular velocity `Ω` (AART `uttilde`, P1 Eq. B52)."
@inline function u_t_general(r, a, ur, Ω)
    Δ = kerr_delta(r, a)
    return sqrt((1 + ur * ur * r * r / Δ) / (1 - (r * r + a * a) * Ω * Ω - (2 / r) * (1 - a * Ω)^2))
end

"Orbital energy Ê of the sub-Keplerian flow (AART `Ehat`, P1 Eq. B44a)."
@inline function E_subkep(r, a, ℓ)
    Δ = kerr_delta(r, a)
    return sqrt(Δ / (kerr_pif(r, a) / (r * r) - (4 * a * ℓ) / r - (1 - 2 / r) * ℓ * ℓ))
end

"Radial velocity ν̂ of the sub-Keplerian flow of energy `E` (AART `nuhat`, P1 Eq. B45)."
@inline function nu_subkep(r, a, ℓ, E)
    Δ = kerr_delta(r, a)
    return r / Δ * sqrt(abs(kerr_pif(r, a) / (r * r) - (4 * a * ℓ) / r - (1 - 2 / r) * ℓ * ℓ - Δ / (E * E)))
end

"Specific angular momentum ℓ̂ = `sub_kep` ℓ_K of the sub-Keplerian flow (AART `lhat`, P1 Eq. B44b)."
@inline l_subkep(r, a, sub_kep) = sub_kep * keplerian_specific_angular_momentum(a, r)

"""
    g_from_velocity(r, a, b, λ, η, ur, Ω) -> g

Redshift factor (P1 Eq. B13) of a photon of conserved `(λ, η)` and emitted radial
direction `b = sign(pʳ)` from an emitter with radial velocity `ur` and angular velocity
`Ω` at radius `r`. The pʳ term is written `b uʳ √|R(r)|`, equal to AART's
`b sign(uʳ) √|R uʳ²|` for real uʳ and differentiable at uʳ = 0.
"""
@inline function g_from_velocity(r, a, b, λ, η, ur, Ω)
    Δ = kerr_delta(r, a)
    ut = u_t_general(r, a, ur, Ω)
    uφ = ut * Ω
    return 1 / (ut * (1 - b * ur * sqrt(abs(kerr_radial_potential(r, a, λ, η))) / Δ / ut - λ * uφ / ut))
end

"""
    gfactor_circular(rs, a, b, λ, η; sub_kep = 1, betar = 1, betaphi = 1) -> g

Redshift factor `g = ν_obs/ν_emit` of a photon emitted at Boyer–Lindquist radius `rs`
outside the ISCO by the sub-Keplerian, mixed radial/azimuthal flow (AART `gDisk`,
P1 Eq. B13). `b` is the sign of the emitted pʳ, `(λ, η)` the photon's conserved
quantities. `sub_kep` scales the specific angular momentum (1 is Keplerian), `betaphi`
blends Ω̂ (1) toward the infall Ω̄ (0), `betar` blends uʳ from 0 (1) toward pure radial
infall (0); the defaults give the Keplerian equatorial disk.

# References
Gralla & Lupsasca (2020), Eq. B13.
"""
@inline function gfactor_circular(rs, a, b, λ, η; sub_kep = 1.0, betar = 1.0, betaphi = 1.0)
    ℓ = l_subkep(rs, a, sub_kep)
    OH = omega_subkep(rs, a, ℓ)
    OT = OH + (1 - betaphi) * (omega_infall(rs, a) - OH)
    ur = (1 - betar) * u_r_infall(rs, a)
    return g_from_velocity(rs, a, b, λ, η, ur, OT)
end

"""
    gfactor_plunge(rs, a, b, λ, η; sub_kep = 1, betar = 1, betaphi = 1) -> g

Redshift factor of a photon emitted at `rs` inside the ISCO by the plunging flow
(AART `gGas`, P1 Eq. B13): the gas keeps the ISCO angular momentum ℓ_ms and energy
Ê_ms, and its radial velocity follows the geodesic radial potential, vanishing at the
ISCO so that it matches `gfactor_circular` there. Arguments as in `gfactor_circular`.

# References
Gralla & Lupsasca (2020), Eq. B13.
"""
@inline function gfactor_plunge(rs, a, b, λ, η; sub_kep = 1.0, betar = 1.0, betaphi = 1.0)
    isco = isco_prograde(a)
    ℓms = l_subkep(isco, a, sub_kep)
    OH = omega_subkep(rs, a, ℓms)
    OT = OH + (1 - betaphi) * (omega_infall(rs, a) - OH)
    Ems = E_subkep(isco, a, ℓms)
    urhat = -kerr_delta(rs, a) / (rs * rs) * nu_subkep(rs, a, ℓms, Ems) * Ems
    ur = urhat + (1 - betar) * (u_r_infall(rs, a) - urhat)
    return g_from_velocity(rs, a, b, λ, η, ur, OT)
end

"""
    gfactor_disk(rs, a, b, λ, η; sub_kep = 1, betar = 1, betaphi = 1) -> g

Redshift factor of the mixed equatorial flow: `gfactor_circular` for `rs ≥ r_ISCO`,
`gfactor_plunge` below (AART `bright_radial` selector), continuous across the ISCO.
For `a → 0` and unit knobs it is the circular-orbit redshift `1/(uᵗ(1 − λΩ_K))`;
`g → 1` at large `rs`.
"""
@inline function gfactor_disk(rs, a, b, λ, η; sub_kep = 1.0, betar = 1.0, betaphi = 1.0)
    isco = isco_prograde(a)
    return rs >= isco ?
           gfactor_circular(rs, a, b, λ, η; sub_kep = sub_kep, betar = betar, betaphi = betaphi) :
           gfactor_plunge(rs, a, b, λ, η; sub_kep = sub_kep, betar = betar, betaphi = betaphi)
end

"""
    johnson_su_profile(r; mu = 0.0, sigma = 0.5, gamma = -1.5) -> emissivity

The Johnson-SU radial emissivity
`f(r) = exp(−½(γ + asinh((r − μ)/σ))²)/√((r − μ)² + σ²)` with location `mu`, width
`sigma` and skewness `gamma`: positive and integrable. AART's canonical disk has
`γ = −3/2`, `μ = 1 − √(1 − a²)`, `σ = 1/2`.

# References
Gralla & Lupsasca (2020), Eq. 58.
"""
@inline function johnson_su_profile(r; mu = 0.0, sigma = 0.5, gamma = -1.5)
    z = (r - mu) / sigma
    return exp(-(gamma + asinh(z))^2 / 2) / sqrt((r - mu)^2 + sigma * sigma)
end

"""
    equatorial_disk_intensity(rs, νr, a, λ, η; sub_kep = 1, betar = 1, betaphi = 1,
                              gamma_p = -1.5, mu_p = 1 - √(1 - a²), sigma_p = 0.5,
                              redshift_power = 3, rin = 0.0) -> I_obs

Observed intensity of the optically thin equatorial g-disk for a ray landing at
Boyer–Lindquist radius `rs` with emitted-pʳ sign `νr` and conserved `(λ, η)` around a
hole of spin `a` (AART `bright_radial`, P1 Eq. 50):
`I_obs = johnson_su_profile(rs) gfactor_disk(rs)^redshift_power`, zero at or inside
the horizon or for `rs < rin`. `redshift_power = 3` is the frequency-integrated
optically thin disk: g² from the solid angle and g from `I_ν/ν³` invariance.
"""
@inline function equatorial_disk_intensity(rs, νr, a, λ, η;
                                           sub_kep = 1.0, betar = 1.0, betaphi = 1.0,
                                           gamma_p = -1.5, mu_p = 1 - sqrt(1 - a * a),
                                           sigma_p = 0.5, redshift_power = 3.0, rin = 0.0)
    rph = kerr_horizon(a)
    (rs <= rph || rs < rin) && return zero(promote_type(typeof(rs), typeof(a), typeof(λ), typeof(η)))
    g = gfactor_disk(rs, a, νr, λ, η; sub_kep = sub_kep, betar = betar, betaphi = betaphi)
    prof = johnson_su_profile(rs; mu = mu_p, sigma = sigma_p, gamma = gamma_p)
    return prof * g^redshift_power
end

"""
    EquatorialDisk{T}
    EquatorialDisk(a; sub_kep = 1, betar = 1, betaphi = 1, gamma_p = -1.5,
                   mu_p = 1 - √(1 - a²), sigma_p = 0.5, redshift_power = 3, rin = 0.0)

The AART g-disk around a hole of spin `a`: the flow knobs `sub_kep`, `betar`,
`betaphi` of `gfactor_disk`, the Johnson-SU profile `gamma_p`, `mu_p`, `sigma_p`, the
exponent `redshift_power` of g, and the inner cut `rin`. Every field has the type `T`,
the promotion of the arguments with `Float64`. `intensity(disk, landing)` paints a
landing of the exact-Kerr tracer. AART's canonical M87* disk is `EquatorialDisk(0.94)`.
"""
struct EquatorialDisk{T}
    a::T
    sub_kep::T
    betar::T
    betaphi::T
    gamma_p::T
    mu_p::T
    sigma_p::T
    redshift_power::T
    rin::T
end

function EquatorialDisk(a; sub_kep = 1.0, betar = 1.0, betaphi = 1.0,
                        gamma_p = -1.5, mu_p = 1 - sqrt(1 - a * a), sigma_p = 0.5,
                        redshift_power = 3.0, rin = 0.0)
    T = promote_type(typeof(a), typeof(sub_kep), typeof(betar), typeof(betaphi),
                     typeof(gamma_p), typeof(mu_p), typeof(sigma_p),
                     typeof(redshift_power), typeof(rin), Float64)
    return EquatorialDisk{T}(T(a), T(sub_kep), T(betar), T(betaphi), T(gamma_p),
                             T(mu_p), T(sigma_p), T(redshift_power), T(rin))
end

"""
    intensity(disk::EquatorialDisk, landing::Landing) -> I_obs

Observed intensity of `disk` along the ray of `landing`, a photon-ring layer from
`band_landing`: `equatorial_disk_intensity` at the landing radius `landing.r` with
its `νr`, `λ`, `η` and the disk spin; zero when the landing is not `valid`. The
number type is the promotion of the disk's and the landing's.
"""
@inline function intensity(disk::EquatorialDisk{T}, landing::Landing{Tl}) where {T,Tl}
    R = promote_type(T, Tl)
    landing.valid || return zero(R)
    return R(equatorial_disk_intensity(landing.r, landing.νr, disk.a, landing.λ, landing.η;
                                       sub_kep = disk.sub_kep, betar = disk.betar,
                                       betaphi = disk.betaphi, gamma_p = disk.gamma_p,
                                       mu_p = disk.mu_p, sigma_p = disk.sigma_p,
                                       redshift_power = disk.redshift_power, rin = disk.rin))
end
