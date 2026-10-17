# The Compton kick (Canfield, Howard & Liang 1987; grmonty, Dolence et al. 2009): draw the
# electron the photon meets from the flux-weighted, Klein–Nishina-accepted relativistic
# Maxwellian, boost the photon into its rest frame, draw the scattered energy and angle
# (Thomson or Klein–Nishina), and boost back. Every function threads an `MCRng`.

"Cap of the electron rejection loops (acceptance about 72 % at Θₑ ≈ 1)."
const MAX_ELECTRON_ITER = 100_000
"Cap of the Thomson and Klein–Nishina energy rejection loops."
const MAX_KN_ITER = 1_000_000
"Guard of the boost direction normalisation 1/(γV) as β → 0."
const BOOST_SMALL = 1.0e-40

# The Canfield χ² mixture weights √π/4 and 1/√2.
const SQRT_PI_OVER4 = 0.25 * sqrt(π)
const INV_SQRT2 = 1.0 / sqrt(2.0)

"""
    lorentz_boost(v, u) -> SVector{4,Float64}

The four-vector `v` in the frame moving with four-velocity `u = (γ, γβ n̂)` relative to
the current one, signature (−,+,+,+). The inverse is the boost by `(u[1], −u[2], −u[3],
−u[4])`. `BOOST_SMALL` regularises `n̂` as `β → 0`, where the boost is the identity.
"""
@inline function lorentz_boost(v::SVector{4}, u::SVector{4})
    g = u[1]
    V = sqrt(abs(1.0 - 1.0 / (g * g)))
    gV_inv = 1.0 / (g * V + BOOST_SMALL)
    n1 = u[2] * gV_inv
    n2 = u[3] * gV_inv
    n3 = u[4] * gV_inv
    gm1 = g - 1.0
    n1g = n1 * gm1
    n2g = n2 * gm1
    n3g = n3 * gm1

    vp0 = g * v[1] - u[2] * v[2] - u[3] * v[3] - u[4] * v[4]
    vp1 = -u[2] * v[1] + (1.0 + n1 * n1g) * v[2] + n1g * n2 * v[3] + n1g * n3 * v[4]
    vp2 = -u[3] * v[1] + n2g * n1 * v[2] + (1.0 + n2 * n2g) * v[3] + n2g * n3 * v[4]
    vp3 = -u[4] * v[1] + n3g * n1 * v[2] + n3g * n2 * v[3] + (1.0 + n3 * n3g) * v[4]
    return SVector(vp0, vp1, vp2, vp3)
end

"""
    sample_y(theta_e, rng) -> (y, rng)

Canfield's variable `y = √((γ − 1)/Θₑ)` of a Maxwell–Jüttner electron: a χ² deviate
with 3, 4, 5 or 6 degrees of freedom chosen by the mixture weights, `y = √(χ²/2)`,
accepted with probability `√(1 + Θₑy²/2)/(1 + y√(Θₑ/2))`.
"""
@inline function sample_y(theta_e, rng::MCRng)
    sqrt_te = sqrt(theta_e)
    pi3 = SQRT_PI_OVER4
    pi4 = INV_SQRT2 * sqrt_te / 2.0
    pi5 = 1.5 * SQRT_PI_OVER4 * theta_e
    pi6 = theta_e * INV_SQRT2 * sqrt_te
    S3 = pi3 + pi4 + pi5 + pi6
    c3 = pi3 / S3
    c4 = c3 + pi4 / S3
    c5 = c4 + pi5 / S3
    inv = INV_SQRT2 * sqrt_te

    y = 0.0
    @inbounds for _ in 1:MAX_ELECTRON_ITER
        x1, rng = mc_uniform(rng)
        local chi
        if x1 < c3
            chi, rng = chi_square(3, rng)
        elseif x1 < c4
            chi, rng = chi_square(4, rng)
        elseif x1 < c5
            chi, rng = chi_square(5, rng)
        else
            chi, rng = chi_square(6, rng)
        end
        y = sqrt(chi / 2.0)
        prob = sqrt(1.0 + 0.5 * theta_e * y * y) / (1.0 + y * inv)
        u, rng = mc_uniform(rng)
        if u < prob
            break
        end
    end
    return y, rng
end

"""
    sample_electron_gamma(theta_e, rng) -> (gamma_e, beta_e, rng)

Lorentz factor `γ = y²Θₑ + 1` and speed `β = √(1 − 1/γ²)` of an electron of the bare
Maxwell–Jüttner distribution at temperature `theta_e` (no flux or Klein–Nishina
weighting), whose mean is `3Θₑ + K₁(1/Θₑ)/K₂(1/Θₑ)`.
"""
@inline function sample_electron_gamma(theta_e, rng::MCRng)
    y, rng = sample_y(theta_e, rng)
    gamma_e = y * y * theta_e + 1.0
    beta_e = sqrt(1.0 - 1.0 / (gamma_e * gamma_e))
    return gamma_e, beta_e, rng
end

"""
    sample_mu(beta_e, r) -> Float64

The cosine `μ` between photon and electron momenta from the collision-flux density
`∝ (1 − β_e μ)`, by inverting its distribution at the uniform variate `r ∈ [0, 1)`,
clamped to `[−1, 1]`.
"""
@inline function sample_mu(beta_e, r)
    det = 1.0 + 2.0 * beta_e + beta_e * beta_e - 4.0 * beta_e * r
    mu = (1.0 - sqrt(abs(det))) / beta_e
    return clamp(mu, -1.0, 1.0)
end

"""
    sample_electron(theta_e, k_tet, rng) -> (p_tet::SVector{4}, rng)

Four-velocity `(γ, γβ n̂)` of the electron that the photon of tetrad-frame wave vector
`k_tet` scatters off: `(γ, β)` from the Maxwell–Jüttner distribution at `theta_e`, the
collision cosine from the flux density, accepted with probability
`kn_total(γ(1 − βμ) k⁰)`, and the direction at that cosine to the photon about a random
transverse axis.
"""
@inline function sample_electron(theta_e, k_tet::SVector{4}, rng::MCRng)
    gamma_e = 1.0
    beta_e = 0.0
    mu = 0.0
    @inbounds for _ in 1:MAX_ELECTRON_ITER
        gamma_e, beta_e, rng = sample_electron_gamma(theta_e, rng)
        r, rng = mc_uniform(rng)
        mu = sample_mu(beta_e, r)
        K = gamma_e * (1.0 - beta_e * mu) * k_tet[1]
        accept = kn_total(K)
        u, rng = mc_uniform(rng)
        if u < accept
            break
        end
    end

    v0, v1, v2, rng = photon_triad(SVector(k_tet[2], k_tet[3], k_tet[4]), rng)
    phiu, rng = mc_uniform(rng)
    phi = 2.0 * π * phiu
    sphi = sin(phi)
    cphi = cos(phi)
    sth = sqrt(max(0.0, 1.0 - mu * mu))
    gb = gamma_e * beta_e
    p1 = gb * (mu * v0[1] + sth * (cphi * v1[1] + sphi * v2[1]))
    p2 = gb * (mu * v0[2] + sth * (cphi * v1[2] + sphi * v2[2]))
    p3 = gb * (mu * v0[3] + sth * (cphi * v1[3] + sphi * v2[3]))
    return SVector(gamma_e, p1, p2, p3), rng
end

# Orthonormal triad (v0, v1, v2): v0 along `dir`, v1 a random unit vector normal to it,
# v2 = v0 × v1.
@inline function photon_triad(dir::SVector{3}, rng::MCRng)
    v0 = dir ./ norm(dir)
    n0, rng = sample_direction(rng)
    v1u = n0 .- dot(n0, v0) .* v0
    v1 = v1u ./ norm(v1u)
    v2 = cross(v0, v1)
    return v0, v1, v2, rng
end

"""
    klein_nishina(a, ap) -> Float64

The Klein–Nishina differential cross section, up to a constant, for rest-frame photon
energies `a` before and `ap` after the scattering: `(a/ap + ap/a − 1 + χ²)/a²` with
`χ = 1 + 1/a − 1/ap`.
"""
@inline function klein_nishina(a, ap)
    ch = 1.0 + 1.0 / a - 1.0 / ap
    return (a / ap + ap / a - 1.0 + ch * ch) / (a * a)
end

# Thomson scattering cosine from the dipole (3/8)(1 + cos² θ), by rejection.
@inline function sample_thomson_cos(rng::MCRng)
    cth = 0.0
    @inbounds for _ in 1:MAX_KN_ITER
        u1, rng = mc_uniform(rng)
        u2, rng = mc_uniform(rng)
        cth = 2.0 * u1 - 1.0
        if 0.75 * u2 < 0.375 * (1.0 + cth * cth)
            break
        end
    end
    return cth, rng
end

# Scattered rest-frame energy from the Klein–Nishina spectrum, by rejection in the box of
# height kn(k0, k0pmin) over [k0/(1 + 2k0), k0].
@inline function sample_klein_nishina_energy(k0, rng::MCRng)
    k0pmin = k0 / (1.0 + 2.0 * k0)
    height = 2.0 * (1.0 + 2.0 * k0 + 2.0 * k0 * k0) / (k0 * k0 * (1.0 + 2.0 * k0))
    k0p = k0pmin
    @inbounds for _ in 1:MAX_KN_ITER
        u1, rng = mc_uniform(rng)
        k0p = k0pmin + (k0 - k0pmin) * u1
        u2, rng = mc_uniform(rng)
        if u2 * height < klein_nishina(k0, k0p)
            break
        end
    end
    return k0p, rng
end

"""
    sample_scattered_energy(k0, rng) -> (k0p, cos_theta, rng)

Scattered energy `k0p` and scattering cosine in the electron rest frame for incident
energy `k0` (mₑc²): Thomson for `k0 ≤ 10⁻⁴` (`k0p = k0`, cosine from the dipole), else
Klein–Nishina (`k0p` from its spectrum, `cos θ = 1 − 1/k0p + 1/k0` by the Compton
relation).
"""
@inline function sample_scattered_energy(k0, rng::MCRng)
    if k0 <= 1.0e-4
        cth, rng = sample_thomson_cos(rng)
        return k0, cth, rng
    end
    k0p, rng = sample_klein_nishina_energy(k0, rng)
    cth = 1.0 - 1.0 / k0p + 1.0 / k0
    return k0p, cth, rng
end

"""
    compton_scatter(k_tet, theta_e, rng) -> (k_scattered::SVector{4}, rng)

The photon of tetrad-frame wave vector `k_tet` scattered off a thermal electron gas of
temperature `theta_e`, in the same frame: the electron is drawn by `sample_electron`,
the photon boosted into its rest frame, the scattered energy and angle drawn by
`sample_scattered_energy` about the incident direction with a random azimuth, and the
result boosted back. Null to round-off.
"""
@inline function compton_scatter(k_tet::SVector{4}, theta_e, rng::MCRng)
    p, rng = sample_electron(theta_e, k_tet, rng)

    ke = lorentz_boost(k_tet, p)
    k0p, cth, rng = sample_scattered_energy(ke[1], rng)
    sth = sqrt(abs(1.0 - cth * cth))

    ke_sp = SVector(ke[2], ke[3], ke[4])
    v0, v1, v2, rng = photon_triad(ke_sp, rng)
    phiu, rng = mc_uniform(rng)
    phi = 2.0 * π * phiu
    sphi = sin(phi)
    cphi = cos(phi)
    dir1 = cth * v0[1] + sth * (cphi * v1[1] + sphi * v2[1])
    dir2 = cth * v0[2] + sth * (cphi * v1[2] + sphi * v2[2])
    dir3 = cth * v0[3] + sth * (cphi * v1[3] + sphi * v2[3])
    kpe = SVector(k0p, k0p * dir1, k0p * dir2, k0p * dir3)

    p_rev = SVector(p[1], -p[2], -p[3], -p[4])
    kp = lorentz_boost(kpe, p_rev)
    return kp, rng
end

"""
    scatter_superphoton(p_cov, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
        -> (p_new::SVector{4}, E_new, rng)

The covariant momentum `p_cov` after one Compton scattering off the thermal plasma of
temperature `theta_e` moving with `ucon`: `p_cov` is projected on the emission tetrad
of `(ucon, ucov, bcon, gcov, gcon)` (`k^(a) = η^{ab} e_(b)^μ p_μ`), scattered by
`compton_scatter`, mapped back and lowered with `gcov`. `E_new = −p_new[1]`. Only
`e_(0) = u` matters physically; the spatial axes drop out of the isotropised kick.
"""
@inline function scatter_superphoton(p_cov::SVector{4}, ucon::SVector{4}, ucov::SVector{4},
                                     bcon::SVector{4}, theta_e,
                                     gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, rng::MCRng)
    Econ = emission_tetrad(ucon, ucov, bcon, gcov, gcon)

    Kd = Econ * p_cov
    k_tet = SVector(-Kd[1], Kd[2], Kd[3], Kd[4])

    kp_tet, rng = compton_scatter(k_tet, theta_e, rng)

    Kcon = transpose(Econ) * kp_tet
    p_new = gcov * Kcon
    E_new = -p_new[1]
    return p_new, E_new, rng
end

"""
    scatter_superphoton(sp::Superphoton, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
        -> (Superphoton, rng)

`sp` after one Compton scattering: the momentum of the method above, `nscatt + 1`, and
every other field unchanged (the scattering is local).
"""
@inline function scatter_superphoton(sp::Superphoton, ucon::SVector{4}, ucov::SVector{4},
                                     bcon::SVector{4}, theta_e,
                                     gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, rng::MCRng)
    p_new, _, rng = scatter_superphoton(sp.p, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
    new_sp = Superphoton(sp.x, p_new, sp.w;
                         tau_abs = sp.tau_abs, tau_scatt = sp.tau_scatt,
                         nscatt = sp.nscatt + Int32(1),
                         x1i = sp.x1i, x2i = sp.x2i,
                         ne0 = sp.ne0, thetae0 = sp.thetae0, b0 = sp.b0)
    return new_sp, rng
end

"""
    scatter_superphoton(p_cov, zone::EmissionZone, rng) -> (p_new, E_new, rng)

The momentum method with the plasma and metric of `zone`.
"""
@inline scatter_superphoton(p_cov::SVector{4}, z::EmissionZone, rng::MCRng) =
    scatter_superphoton(p_cov, z.ucon, z.ucov, z.bcon, z.theta_e, z.gcov, z.gcon, rng)
