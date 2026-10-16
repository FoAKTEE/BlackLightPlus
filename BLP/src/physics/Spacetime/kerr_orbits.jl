# Exact-Kerr orbital primitives (M = 1, Boyer–Lindquist): the outer horizon r₊, the
# metric function Δ(r), the photon radial potential R(r), the special circular-orbit
# radii (ISCO, photon, marginally bound) and the Keplerian kinematics (Ω_K, ℓ_K).
# Pure, isbits and AD-generic (ForwardDiff flows through `a`, `r`).

# ── metric primitives ────────────────────────────────────────────────────────

"Outer event-horizon radius r₊ = 1 + √(1 − a²) (M = 1, Boyer–Lindquist)."
@inline kerr_horizon(a) = 1 + sqrt(1 - a * a)

"Kerr Δ(r) = r² − 2r + a² (M = 1, Boyer–Lindquist)."
@inline kerr_delta(r, a) = r * r - 2 * r + a * a

"""
    kerr_radial_potential(r, a, λ, η) -> R(r)

The Kerr photon radial potential R(r) = (r²+a²−aλ)² − Δ(r)(η+(λ−a)²) (Gralla &
Lupsasca PRD 101, P2 Eq. 5); its reciprocal square root ∫dr/√R is the Mino-time
integrand. M = 1, Boyer–Lindquist; AD-generic in every argument.
"""
@inline kerr_radial_potential(r, a, λ, η) =
    (r * r + a * a - a * λ)^2 - kerr_delta(r, a) * (η + (λ - a)^2)

# ── special circular-orbit radii ─────────────────────────────────────────────

"""
    isco_radius(a; prograde=true) -> r_ISCO

Kerr innermost stable circular orbit (Bardeen, Press & Teukolsky 1972), M=1.
Returns 6 at a=0, 4.233 at a=0.5, 2.024 at a=0.94 (prograde).
"""
@inline function isco_radius(a; prograde::Bool=true)
    s = prograde ? 1.0 : -1.0
    z1 = 1 + cbrt(1 - a^2) * (cbrt(1 + a) + cbrt(1 - a))
    z2 = sqrt(3 * a^2 + z1^2)
    return 3 + z2 - s * sqrt((3 - z1) * (3 + z1 + 2 * z2))
end

"""
    isco_prograde(a) -> r_ISCO

Prograde ISCO radius (M = 1) — AART `rms`, P1 Eq. B16: `isco_radius(a;
prograde=true)`, the inner edge of the g-disk source; AD-generic in `a`.
"""
@inline isco_prograde(a) = isco_radius(a; prograde=true)

"Marginally bound (E=1) circular-orbit radius (M=1): 2 - a + 2 sqrt(1-a) prograde."
@inline marginally_bound_radius(a; prograde::Bool=true) =
    prograde ? (2 - a + 2 * sqrt(1 - a)) : (2 + a + 2 * sqrt(1 + a))

"Prograde photon (marginally bound null) circular orbit radius, M=1."
@inline photon_orbit_radius(a; prograde::Bool=true) =
    2 * (1 + cos((2 / 3) * acos(prograde ? -a : a)))

# ── Keplerian circular-orbit kinematics ──────────────────────────────────────

"""
    keplerian_omega(a, r; prograde=true) -> Omega_K

Coordinate angular velocity of a circular equatorial (Keplerian) orbit at
Boyer–Lindquist radius `r` in Kerr (M=1): Omega_K = ±1 / (r^{3/2} ± a).
"""
@inline keplerian_omega(a, r; prograde::Bool=true) =
    prograde ? one(r) / (r^1.5 + a) : -one(r) / (r^1.5 - a)

"""
    keplerian_specific_angular_momentum(a, r; prograde=true) -> l_K

Specific angular momentum l = -u_phi/u_t of a circular equatorial geodesic
(Bardeen, Press & Teukolsky 1972, sign ±), M=1:
l_K = ±(r^2 ∓ 2 a sqrt(r) + a^2) / (r^{3/2} - 2 sqrt(r) ± a) — upper sign
prograde, lower sign retrograde. At a=0 this is ±r^{3/2}/(r-2) (Schwarzschild).
The prograde branch is > 0 with its minimum l_ms at the ISCO; the retrograde
branch carries BPT's overall LEADING minus, so it is NEGATIVE (a counter-rotating
orbit has u_phi > 0, hence l = -u_phi/u_t < 0).
"""
@inline function keplerian_specific_angular_momentum(a, r; prograde::Bool=true)
    s = prograde ? one(r) : -one(r)
    return s * (r^2 - s * 2 * a * sqrt(r) + a^2) / (r^1.5 - 2 * sqrt(r) + s * a)
end
