# Monte Carlo — scatter.jl
#
# The Compton kick: pick the electron a superphoton collides with, transform
# into its rest frame, apply the Klein-Nishina (or Thomson) scattering
# kinematics, and transform back (Canfield, Howard & Liang 1987; cf. grmonty,
# Dolence et al. 2009). `sample_electron` draws from the relativistic
# Maxwellian, flux-weighted by (1 - beta mu) and Klein-Nishina accepted;
# `compton_scatter` chains the stages in the tetrad frame; `scatter_superphoton`
# drives in coordinates. All leaf functions are pure, isbits, and thread an `MCRng`.

# ============================================================ caps / constants

const _MAX_ELECTRON_ITER = 100_000   # electron rejection cap (eff ~72% at Theta_e~1)
const _MAX_KN_ITER       = 1_000_000 # KN/Thomson energy rejection cap (low eff, hard photons)
const _BOOST_SMALL       = 1.0e-40   # guards the 1/(gamma*V) direction normalisation

# sqrt(pi)/4 and 1/sqrt(2): the Canfield chi^2-mixture weights.
const _SQRT_MPI_OVER4 = 0.25 * sqrt(π)
const _INV_SQRT2      = 1.0 / sqrt(2.0)

# ============================================================ Lorentz boost

"""
    lorentz_boost(v, u) -> SVector{4,Float64}

General special-relativistic boost of the 4-vector `v` into the frame whose
4-velocity (in the current frame) is `u = (gamma, gamma*beta*n_hat)`. The inverse
boost is `lorentz_boost(v, (u[1], -u[2], -u[3], -u[4]))` (reversed spatial
velocity). Pure / allocation-free; `_BOOST_SMALL` guards the `beta -> 0` direction
normalisation (the boost then reduces to the identity). Signature (-,+,+,+).
"""
@inline function lorentz_boost(v::SVector{4}, u::SVector{4})
    g = u[1]
    V = sqrt(abs(1.0 - 1.0 / (g * g)))
    gV_inv = 1.0 / (g * V + _BOOST_SMALL)
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

# ============================================================ electron sampling

# Canfield-Howard-Liang chi^2 mixture: draw y with dof selected by cumulative
# weights, accept the reweight to the Maxwell-Juttner beta distribution.
@inline function sample_y(theta_e, rng::MCRng)
    sqrt_te = sqrt(theta_e)
    pi3 = _SQRT_MPI_OVER4
    pi4 = _INV_SQRT2 * sqrt_te / 2.0
    pi5 = 1.5 * _SQRT_MPI_OVER4 * theta_e
    pi6 = theta_e * _INV_SQRT2 * sqrt_te
    S3 = pi3 + pi4 + pi5 + pi6
    c3 = pi3 / S3
    c4 = c3 + pi4 / S3
    c5 = c4 + pi5 / S3
    inv = _INV_SQRT2 * sqrt_te

    y = 0.0
    @inbounds for _ in 1:_MAX_ELECTRON_ITER
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

Draw an electron Lorentz factor / speed from the relativistic Maxwell-Juttner
distribution at temperature `Theta_e` (Canfield-Howard-Liang `y` sampler:
`gamma_e = y^2 Theta_e + 1`, `beta_e = sqrt(1 - 1/gamma_e^2)`). This is the BARE
thermal distribution (no flux / Klein-Nishina weighting); its mean is
`<gamma_e> = 3 Theta_e + K1(1/Theta_e)/K2(1/Theta_e)`.
"""
@inline function sample_electron_gamma(theta_e, rng::MCRng)
    y, rng = sample_y(theta_e, rng)
    gamma_e = y * y * theta_e + 1.0
    beta_e = sqrt(1.0 - 1.0 / (gamma_e * gamma_e))
    return gamma_e, beta_e, rng
end

"""
    sample_mu(beta_e, r) -> Float64

Sample the cosine ``\\mu`` of the angle between the photon and electron momenta
from the ``(1 - \\beta_e \\mu)`` collision-flux distribution.

`beta_e` is the electron speed in units of ``c`` and `r` is a uniform variate on
``[0,1)``; the result is drawn by inverse CDF and clamped to ``[-1, 1]``. The
discriminant satisfies ``(1-\\beta_e)^2 \\le \\mathrm{det} \\le (1+\\beta_e)^2``,
so the `abs` guarding the square root is defensive only.

See also [`sample_electron`](@ref), [`compton_scatter`](@ref).
"""
@inline function sample_mu(beta_e, r)
    det = 1.0 + 2.0 * beta_e + beta_e * beta_e - 4.0 * beta_e * r
    mu = (1.0 - sqrt(abs(det))) / beta_e
    return clamp(mu, -1.0, 1.0)
end

"""
    sample_electron(theta_e, k_tet, rng) -> (p_tet::SVector{4}, rng)

Sample the 4-velocity `p_tet = (gamma_e, gamma_e beta_e n_hat)` of the electron
a photon of tetrad-frame wavevector `k_tet` scatters off: `(gamma_e, beta_e)`
from the Maxwell-Juttner distribution, the collision cosine `mu` from the
`(1 - beta_e mu)` flux distribution, accepted with the Klein-Nishina
probability `kn_total(gamma_e (1 - beta_e mu) k_tet[1])`. The direction is
built in a random triad about the photon axis. Rejection-capped for device safety.
"""
@inline function sample_electron(theta_e, k_tet::SVector{4}, rng::MCRng)
    gamma_e = 1.0
    beta_e = 0.0
    mu = 0.0
    @inbounds for _ in 1:_MAX_ELECTRON_ITER
        gamma_e, beta_e, rng = sample_electron_gamma(theta_e, rng)
        r, rng = mc_uniform(rng)
        mu = sample_mu(beta_e, r)
        K = gamma_e * (1.0 - beta_e * mu) * k_tet[1]
        accept = kn_total(K)          # = 1 - 2K for K < 1e-3 (kn_total's own branch)
        u, rng = mc_uniform(rng)
        if u < accept
            break
        end
    end

    v0, v1, v2, rng = _photon_triad(SVector(k_tet[2], k_tet[3], k_tet[4]), rng)
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

# Orthonormal spatial triad (v0, v1, v2) with v0 along `dir`, v1 a random
# perpendicular unit vector, v2 = v0 x v1.
@inline function _photon_triad(dir::SVector{3}, rng::MCRng)
    v0 = dir ./ norm(dir)
    n0, rng = sample_direction(rng)
    v1u = n0 .- dot(n0, v0) .* v0
    v1 = v1u ./ norm(v1u)
    v2 = cross(v0, v1)
    return v0, v1, v2, rng
end

# ============================================================ scattering kinematics

"""
    klein_nishina(a, ap) -> Float64

The (unnormalised) Klein-Nishina differential cross section factor for incoming /
outgoing rest-frame photon energies `a` / `ap`,
`(a/ap + ap/a - 1 + ch^2) / a^2` with `ch = 1 + 1/a - 1/ap`. Used as the rejection
envelope target in `sample_klein_nishina_energy`.
"""
@inline function klein_nishina(a, ap)
    ch = 1.0 + 1.0 / a - 1.0 / ap
    return (a / ap + ap / a - 1.0 + ch * ch) / (a * a)
end

# Thomson polar cosine from the (3/8)(1 + cos^2 theta) dipole (rejection).
@inline function sample_thomson_cos(rng::MCRng)
    cth = 0.0
    @inbounds for _ in 1:_MAX_KN_ITER
        u1, rng = mc_uniform(rng)
        u2, rng = mc_uniform(rng)
        cth = 2.0 * u1 - 1.0
        if 0.75 * u2 < 0.375 * (1.0 + cth * cth)
            break
        end
    end
    return cth, rng
end

# Scattered rest-frame energy from the Klein-Nishina spectrum (rejection in the
# box of height kn(k0, k0pmin)); efficiency ~ ln(2 k0)/(2 k0), poor for hard photons.
@inline function sample_klein_nishina_energy(k0, rng::MCRng)
    k0pmin = k0 / (1.0 + 2.0 * k0)
    height = 2.0 * (1.0 + 2.0 * k0 + 2.0 * k0 * k0) / (k0 * k0 * (1.0 + 2.0 * k0))
    k0p = k0pmin
    @inbounds for _ in 1:_MAX_KN_ITER
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

Scattered photon energy `k0p` and scattering cosine `cos_theta` in the ELECTRON
REST frame for an incident rest-frame energy `k0`. Elastic Thomson for
`k0 <= 1e-4` (`k0p = k0`, `cos_theta` from the dipole), else Klein-Nishina
(`k0p` from the KN spectrum, `cos_theta = 1 - 1/k0p + 1/k0` — the Compton relation
`1/k0p - 1/k0 = 1 - cos_theta`).
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

# ============================================================ Compton scatter

"""
    compton_scatter(k_tet, theta_e, rng) -> (k_scattered_tet::SVector{4}, rng)

Scatter a photon of tetrad-frame wavevector `k_tet` off a thermal electron gas at
temperature `Theta_e`, returning the scattered wavevector in the SAME tetrad frame.
Samples the electron 4-velocity, boosts the photon into its rest frame, draws the
scattered energy + angle (Thomson or Klein-Nishina), rebuilds the outgoing
direction in a random triad about the incident rest-frame direction, and boosts
back. The output is null (`k'.k' = 0`) to round-off.
"""
@inline function compton_scatter(k_tet::SVector{4}, theta_e, rng::MCRng)
    p, rng = sample_electron(theta_e, k_tet, rng)

    ke = lorentz_boost(k_tet, p)                       # photon in the electron rest frame
    k0p, cth, rng = sample_scattered_energy(ke[1], rng)
    sth = sqrt(abs(1.0 - cth * cth))

    # Explicit spatial magnitude (do not inherit light-cone error from ke[1]).
    ke_sp = SVector(ke[2], ke[3], ke[4])
    v0, v1, v2, rng = _photon_triad(ke_sp, rng)
    phiu, rng = mc_uniform(rng)
    phi = 2.0 * π * phiu
    sphi = sin(phi)
    cphi = cos(phi)
    dir1 = cth * v0[1] + sth * (cphi * v1[1] + sphi * v2[1])
    dir2 = cth * v0[2] + sth * (cphi * v1[2] + sphi * v2[2])
    dir3 = cth * v0[3] + sth * (cphi * v1[3] + sphi * v2[3])
    kpe = SVector(k0p, k0p * dir1, k0p * dir2, k0p * dir3)

    # Boost back to the tetrad frame with the electron velocity reversed.
    p_rev = SVector(p[1], -p[2], -p[3], -p[4])
    kp = lorentz_boost(kpe, p_rev)
    return kp, rng
end

# ============================================================ coordinate-frame driver

"""
    scatter_superphoton(p_cov, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
        -> (p_new_cov::SVector{4}, E_new, rng)

Coordinate-frame Compton scatter driver: builds the fluid (`e_(0) = u`)
emission tetrad, transforms the incident covariant `p_cov` into it, scatters
(`compton_scatter`), transforms back to the covariant `p'_mu`, and returns it
with the new energy `E' = -p'_t`. Only `e_(0) = u` matters physically; the
spatial tetrad axes cancel under the randomised scatter.
"""
@inline function scatter_superphoton(p_cov::SVector{4}, ucon::SVector{4}, ucov::SVector{4},
                                     bcon::SVector{4}, theta_e,
                                     gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, rng::MCRng)
    Econ = emission_tetrad(ucon, ucov, bcon, gcov, gcon)   # rows e_(a)^mu

    # coord (covariant p_mu) -> tetrad contravariant k^(a): K_(a) = e_(a)^mu p_mu,
    # then raise with eta = diag(-1,1,1,1) (flip the time component).
    Kd = Econ * p_cov
    k_tet = SVector(-Kd[1], Kd[2], Kd[3], Kd[4])

    kp_tet, rng = compton_scatter(k_tet, theta_e, rng)

    # tetrad -> coord: K'^mu = e_(a)^mu k'^(a); lower to covariant with gcov.
    Kcon = transpose(Econ) * kp_tet
    p_new = gcov * Kcon
    E_new = -p_new[1]
    return p_new, E_new, rng
end

"""
    scatter_superphoton(sp::Superphoton, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
        -> (Superphoton, rng)

`Superphoton` overload of the driver: scatters `sp` in place of its momentum,
increments `nscatt`, and rebuilds the packet with the new covariant momentum
(`E = -p'_t` recomputed by the constructor). Position and weight are unchanged
(scattering is local).
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
