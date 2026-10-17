# The Compton kick: the Lorentz boost is invertible and an isometry, Thomson is elastic
# in the electron rest frame, Klein–Nishina obeys the Compton relation, the scattered
# photon is null, and the statistics of the up-scattering and of the electron sampler
# match their closed forms within physically justified bands.
using Test, BLP
using StaticArrays
using LinearAlgebra: dot, norm
using BLP.Numerics: MCRng, bessel_k1, bessel_k2
using BLP.MonteCarlo: lorentz_boost, klein_nishina, sample_scattered_energy, sample_electron,
                      sample_electron_gamma, sample_mu, compton_scatter, scatter_superphoton,
                      Superphoton
using BLP.Matter: EmissionZone

mink4(a) = -a[1] * a[1] + a[2] * a[2] + a[3] * a[3] + a[4] * a[4]
const G4 = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

function boost_errors(n)
    rt = 0.0; iso = 0.0
    for t in 1:n
        g = 1.0 + 5.0 * (((12345 * 2654435761 + t * 40503) % 1000) / 1000.0)
        be = sqrt(1.0 - 1.0 / (g * g))
        d = SVector(cos(t * 0.7), sin(t * 0.7) * cos(t * 0.3), sin(t * 0.7) * sin(t * 0.3))
        u = SVector(g, g * be * d[1], g * be * d[2], g * be * d[3])
        v = SVector(1.0 + 0.5 * sin(t), 0.5 * cos(t), -0.3 * sin(2t), 0.4 * cos(3t))
        vp = lorentz_boost(v, u)
        rt = max(rt, norm(lorentz_boost(vp, SVector(u[1], -u[2], -u[3], -u[4])) - v))
        iso = max(iso, abs(mink4(vp) - mink4(v)))
    end
    return rt, iso
end

function kinematics(n)
    rng = MCRng(2, 2)
    rel = 0.0
    for i in 1:n
        k0 = 0.01 + 5.0 * (i / n)
        k0p, cth, rng = sample_scattered_energy(k0, rng)
        rel = max(rel, abs((1.0 / k0p - 1.0 / k0) - (1.0 - cth)))
    end
    rng = MCRng(3, 3)
    elastic = 0.0
    for _ in 1:n
        k0p, cth, rng = sample_scattered_energy(1e-5, rng)
        elastic = max(elastic, abs(k0p - 1e-5))
        abs(cth) <= 1.0 || (elastic = Inf)
    end
    return rel, elastic
end

function null_residual(n, theta_e, k0, seed)
    rng = MCRng(seed, 4)
    m = 0.0
    for _ in 1:n
        kp, rng = compton_scatter(SVector(k0, k0, 0.0, 0.0), theta_e, rng)
        m = max(m, abs(mink4(kp)) / (kp[1] * kp[1]))
    end
    return m
end

function mean_gain(theta_e, k0, n)
    rng = MCRng(11, 1)
    s = 0.0
    for _ in 1:n
        kp, rng = compton_scatter(SVector(k0, k0, 0.0, 0.0), theta_e, rng)
        s += kp[1] / k0
    end
    return s / n - 1.0
end

function mj_stats(theta_e, n)
    rng = MCRng(7, 5)
    s = 0.0; lo = 1.0; hi = 0.0
    for _ in 1:n
        g, be, rng = sample_electron_gamma(theta_e, rng)
        s += g; lo = min(lo, be); hi = max(hi, be)
    end
    return s / n, lo, hi
end

function full_electron(theta_e, k0, n)
    rng = MCRng(13, 9)
    s = 0.0; normerr = 0.0
    for _ in 1:n
        p, rng = sample_electron(theta_e, SVector(k0, k0, 0.0, 0.0), rng)
        s += p[1]
        normerr = max(normerr, abs(mink4(p) + 1.0))
    end
    return s / n, normerr
end

function driver_residual(n, theta_e)
    vf = 0.3; gf = 1.0 / sqrt(1.0 - vf * vf)
    ucon = SVector(gf, gf * vf, 0.0, 0.0)
    zone = EmissionZone(n_e = 1.0, theta_e = theta_e, B = 1.0, dV = 1.0, x = zeros(4),
                        ucon = ucon, ucov = G4 * ucon, bcon = SVector(0.0, 1.0, 0.0, 0.0),
                        gcov = G4, gcon = G4)
    rng = MCRng(5, 5)
    m = 0.0; minE = Inf
    for _ in 1:n
        pcov = G4 * SVector(1e-4, 0.0, 0.0, 1e-4)
        pnew, Enew, rng = scatter_superphoton(pcov, zone, rng)
        m = max(m, abs(dot(pnew, G4 * pnew)) / (Enew * Enew))
        minE = min(minE, Enew)
    end
    return m, minE
end

@testset "Monte Carlo Compton kick" begin
    @testset "Lorentz boost: round trip and isometry" begin
        rt, iso = boost_errors(2000)
        @test rt < 1e-11
        @test iso < 1e-11
    end

    @testset "rest-frame kinematics" begin
        rel, elastic = kinematics(100_000)
        @test rel < 1e-10
        @test elastic == 0.0
        @test klein_nishina(1.0, 1.0 / 3.0) > 0.0
        @test sample_mu(0.5, 0.0) == -1.0 && sample_mu(0.5, 1.0) ≈ 1.0
    end

    @testset "the scattered photon is null" begin
        @test null_residual(20000, 1.0, 1e-3, 4) < 1e-9
        @test null_residual(20000, 4.0, 1.0, 41) < 1e-9
    end

    @testset "up-scattering: mean energy gain about 4Θₑ" begin
        for th in (0.1, 0.3)
            g = mean_gain(th, 1e-5, 200_000)
            @test 2.0 * th < g < 8.0 * th
        end
    end

    @testset "electron sampling: Maxwell–Jüttner mean" begin
        for th in (1.0, 3.0)
            m, lo, hi = mj_stats(th, 200_000)
            @test m ≈ 3.0 * th + bessel_k1(1.0 / th) / bessel_k2(1.0 / th) rtol = 1e-2
            @test 0.0 <= lo < hi < 1.0
        end
        m, normerr = full_electron(1.0, 1e-6, 200_000)
        @test normerr < 1e-10
        @test m ≈ 3.0 + bessel_k1(1.0) / bessel_k2(1.0) rtol = 3e-2
    end

    @testset "coordinate-frame driver" begin
        m, minE = driver_residual(20000, 1.0)
        @test m < 1e-9
        @test minE > 0.0
        sp = Superphoton(SVector(0.0, 1.0, 2.0, 3.0), G4 * SVector(1e-4, 0.0, 0.0, 1e-4), 2.0;
                         tau_abs = 0.5, nscatt = 1)
        vf = 0.3; gf = 1.0 / sqrt(1.0 - vf * vf); ucon = SVector(gf, gf * vf, 0.0, 0.0)
        args = (ucon, G4 * ucon, SVector(0.0, 1.0, 0.0, 0.0), 1.0, G4, G4, MCRng(1, 2))
        s2, r2 = scatter_superphoton(sp, args...)
        p2, E2, r3 = scatter_superphoton(sp.p, args...)
        @test s2.p === p2 && s2.E === E2 && r2 === r3
        @test s2.nscatt == 2 && s2.w === 2.0 && s2.tau_abs === 0.5 && s2.x === sp.x
    end
end
