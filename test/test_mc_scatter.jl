# Monte Carlo Compton-scatter tests. The scatter kick is exact special-relativistic
# kinematics wrapped in rejection sampling, so most checks are deterministic to
# round-off: the Lorentz boost is invertible and norm-preserving, Thomson is elastic
# in the electron rest frame, Klein-Nishina obeys the Compton relation, and the
# scattered photon is null. Two checks are statistical (mean energy gain, mean
# electron gamma) and use physically-justified few-sigma bands.
using Test
using BlackLightPlus: lorentz_boost, klein_nishina, sample_scattered_energy,
                    sample_electron, sample_electron_gamma, compton_scatter,
                    scatter_superphoton, MCRng, bessel_k1, bessel_k2
using StaticArrays
using LinearAlgebra: dot, norm

# Minkowski inner product (-,+,+,+).
mink4(a) = -a[1] * a[1] + a[2] * a[2] + a[3] * a[3] + a[4] * a[4]

# ------------------------------------------------------------ boost helpers

# Worst boost round-trip error: boost into frame u, then back with u's velocity
# reversed, over many random 4-vectors and electron 4-velocities.
function boost_roundtrip_err(n, seed)
    m = 0.0
    for t in 1:n
        g = 1.0 + 5.0 * (((seed * 2654435761 + t * 40503) % 1000) / 1000.0)
        be = sqrt(1.0 - 1.0 / (g * g))
        d = SVector(cos(t * 0.7), sin(t * 0.7) * cos(t * 0.3), sin(t * 0.7) * sin(t * 0.3))
        u = SVector(g, g * be * d[1], g * be * d[2], g * be * d[3])
        # O(1) test vectors (absolute round-trip error scales with |v|)
        v = SVector(1.0 + 0.5 * sin(t), 0.5 * cos(t), -0.3 * sin(2t), 0.4 * cos(3t))
        vp = lorentz_boost(v, u)
        vback = lorentz_boost(vp, SVector(u[1], -u[2], -u[3], -u[4]))
        m = max(m, norm(vback - v))
    end
    return m
end

# Boost preserves the Minkowski inner product (isometry).
function boost_isometry_err(n, seed)
    m = 0.0
    for t in 1:n
        g = 1.5 + 3.0 * (((seed + 7t) % 997) / 997.0)
        be = sqrt(1.0 - 1.0 / (g * g))
        u = SVector(g, g * be, 0.0, 0.0)
        v = SVector(2.0, 0.5 * sin(t) - 1.0, 0.3, -0.4 * cos(t))
        vp = lorentz_boost(v, u)
        m = max(m, abs(mink4(vp) - mink4(v)))
    end
    return m
end

# ------------------------------------------------------------ kinematics helpers

# Worst Compton-relation residual |1/eps' - 1/eps - (1 - cos)| in the KN regime.
function compton_relation_residual(n, seed)
    rng = MCRng(seed, 2)
    m = 0.0
    for i in 1:n
        k0 = 0.01 + 5.0 * (i / n)
        k0p, cth, rng = sample_scattered_energy(k0, rng)
        m = max(m, abs((1.0 / k0p - 1.0 / k0) - (1.0 - cth)))
    end
    return m
end

# Thomson is elastic: k0' == k0 exactly for k0 <= 1e-4.
function thomson_elastic_err(n, seed)
    rng = MCRng(seed, 3)
    m = 0.0
    for _ in 1:n
        k0p, cth, rng = sample_scattered_energy(1e-5, rng)
        m = max(m, abs(k0p - 1e-5))
        # cosine stays physical
        (abs(cth) <= 1.0) || (m = Inf)
    end
    return m
end

# Worst relative null residual |k'.k'| / (k'^0)^2 of the scattered tetrad photon.
function compton_null_residual(n, theta_e, k0, seed)
    rng = MCRng(seed, 4)
    k_tet = SVector(k0, k0, 0.0, 0.0)
    m = 0.0
    for _ in 1:n
        kp, rng = compton_scatter(k_tet, theta_e, rng)
        m = max(m, abs(mink4(kp)) / (kp[1] * kp[1]))
    end
    return m
end

# ------------------------------------------------------------ statistical helpers

# Mean fractional fluid-frame energy gain <E'/E> - 1 over many scatters.
function mean_energy_gain(theta_e, k0, n, seed)
    rng = MCRng(seed, 1)
    k_tet = SVector(k0, k0, 0.0, 0.0)
    s = 0.0
    for _ in 1:n
        kp, rng = compton_scatter(k_tet, theta_e, rng)
        s += kp[1] / k0
    end
    return s / n - 1.0
end

# Mean electron Lorentz factor from the bare Maxwell-Juttner sampler.
function mean_gamma_mj(theta_e, n, seed)
    rng = MCRng(seed, 5)
    s = 0.0
    maxbeta = 0.0
    minbeta = 1.0
    for _ in 1:n
        g, be, rng = sample_electron_gamma(theta_e, rng)
        (g >= 1.0) || (s = NaN)
        s += g
        maxbeta = max(maxbeta, be)
        minbeta = min(minbeta, be)
    end
    return s / n, minbeta, maxbeta
end

# Mean electron gamma from the FULL flux/KN-weighted sampler + worst 4-velocity
# normalisation residual |p.p + 1| (a valid unit timelike 4-velocity).
function mean_gamma_full(theta_e, k0, n, seed)
    rng = MCRng(seed, 9)
    k_tet = SVector(k0, k0, 0.0, 0.0)
    s = 0.0
    normerr = 0.0
    for _ in 1:n
        p, rng = sample_electron(theta_e, k_tet, rng)
        s += p[1]
        normerr = max(normerr, abs(mink4(p) + 1.0))
    end
    return s / n, normerr
end

# Coordinate-frame driver: worst null residual + smallest post-scatter energy for a
# boosted fluid in Minkowski.
function driver_null_residual(n, theta_e, seed)
    gcov = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    gcon = gcov
    vf = 0.3
    gf = 1.0 / sqrt(1.0 - vf * vf)
    ucon = SVector(gf, gf * vf, 0.0, 0.0)
    ucov = gcov * ucon
    bcon = SVector(0.0, 1.0, 0.0, 0.0)
    rng = MCRng(seed, 5)
    m = 0.0
    minE = Inf
    for _ in 1:n
        E = 1e-4
        kcon = SVector(E, 0.0, 0.0, E)          # null photon along +z
        pcov = gcov * kcon
        pnew, Enew, rng = scatter_superphoton(pcov, ucon, ucov, bcon, theta_e, gcov, gcon, rng)
        knew = gcon * pnew
        m = max(m, abs(dot(pnew, knew)) / (Enew * Enew))
        minE = min(minE, Enew)
    end
    return m, minE
end

# ------------------------------------------------------------ (a) boost identities

@testset "lorentz_boost: round-trip + isometry" begin
    @test boost_roundtrip_err(2000, 12345) < 1e-11
    @test boost_isometry_err(2000, 6789)   < 1e-11
end

# ------------------------------------------------------------ (b) rest-frame kinematics

@testset "scattered energy: Thomson elastic + KN Compton relation" begin
    @test thomson_elastic_err(20000, 3) == 0.0
    @test compton_relation_residual(100000, 2) < 1e-10
    # KN differential factor is positive and symmetric under a<->ap at back-scatter
    @test klein_nishina(1.0, 1.0 / 3.0) > 0.0
end

# ------------------------------------------------------------ (c) null scattered photon

@testset "compton_scatter: scattered photon is null" begin
    @test compton_null_residual(20000, 1.0, 1e-3, 4) < 1e-9
    @test compton_null_residual(20000, 4.0, 1.0, 41) < 1e-9   # KN regime
end

# ------------------------------------------------------------ (d) mean energy gain

@testset "compton up-scattering: positive mean energy gain" begin
    # <E'/E> - 1 ~ 4 Theta_e (Compton up-scattering off a hot thermal gas). Wide
    # physically-justified band [2 Theta_e, 8 Theta_e] (MC error << band width).
    for th in (0.1, 0.3)
        g = mean_energy_gain(th, 1e-5, 200_000, 11)
        @test g > 0.0
        @test 2.0 * th < g < 8.0 * th
    end
end

# ------------------------------------------------------------ (e) electron sampling

@testset "sample_electron: relativistic-Maxwellian gamma" begin
    # bare Maxwell-Juttner mean: <gamma> = 3 Theta_e + K1(1/Theta_e)/K2(1/Theta_e)
    for th in (1.0, 3.0)
        m, minb, maxb = mean_gamma_mj(th, 200_000, 7)
        exact = 3.0 * th + bessel_k1(1.0 / th) / bessel_k2(1.0 / th)
        @test m ≈ exact rtol = 1e-2
        @test 0.0 <= minb < maxb < 1.0             # betas physical
        # order 1 + 3 Theta_e (loose sanity band)
        @test 0.7 * (1.0 + 3.0 * th) < m < 1.1 * (1.0 + 3.0 * th)
    end

    # full flux/KN-weighted sampler: valid 4-velocities; soft photon -> ~ Maxwell-Juttner
    m, normerr = mean_gamma_full(1.0, 1e-6, 200_000, 13)
    @test normerr < 1e-10
    exact = 3.0 + bessel_k1(1.0) / bessel_k2(1.0)
    @test m ≈ exact rtol = 3e-2
end

# ------------------------------------------------------------ coordinate-frame driver

@testset "scatter_superphoton: coord round-trip keeps the photon null" begin
    m, minE = driver_null_residual(20000, 1.0, 5)
    @test m < 1e-9
    @test minE > 0.0
end
