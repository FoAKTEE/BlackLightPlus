# Monte Carlo RNG tests. The counter-based Philox stream must be uniform,
# reproducible and independent across superphoton streams, and the derived
# samplers (isotropic direction, chi-square, Gamma) must hit their analytic
# moments. Ground truth is analytic — no reference code needed.
using Test
using BlackLightPlus: philox4x32, philox_rand, MCRng, mc_uniform, mc_gauss,
                    sample_direction, standard_gamma, chi_square
using LinearAlgebra: norm

# ---- streaming-moment helpers (kept out of the @testset soft scope) ----
function philox_moments(n)
    key = (0x1234abcd, 0x0000002a)
    s = 0.0; s2 = 0.0
    for i in 1:n
        u = philox_rand(key, (UInt32(i), 0x00000000, 0x00000000, 0x00000000))
        s += u; s2 += u * u
    end
    m = s / n
    return m, s2 / n - m * m
end

# Pearson correlation between two Philox streams (different stream ids) drawn at
# the same counters — should be ~0 for independent streams.
function stream_correlation(n)
    ka = (0xdeadbeef, 0x00000001); kb = (0xdeadbeef, 0x00000002)
    sx = 0.0; sy = 0.0; sxy = 0.0; sx2 = 0.0; sy2 = 0.0
    for i in 1:n
        c = (UInt32(i), 0x00000000, 0x00000000, 0x00000000)
        x = philox_rand(ka, c); y = philox_rand(kb, c)
        sx += x; sy += y; sxy += x * y; sx2 += x * x; sy2 += y * y
    end
    cov = sxy / n - (sx / n) * (sy / n)
    vx = sx2 / n - (sx / n)^2; vy = sy2 / n - (sy / n)^2
    return cov / sqrt(vx * vy)
end

function sampler_moments(sampler, id, n)
    rng = MCRng(2024, id)
    s = 0.0; s2 = 0.0
    for _ in 1:n
        x, rng = sampler(rng)
        s += x; s2 += x * x
    end
    m = s / n
    return m, s2 / n - m * m
end

function direction_moments(n)
    rng = MCRng(99, 7)
    mx = 0.0; my = 0.0; mz = 0.0; maxdev = 0.0
    for _ in 1:n
        v, rng = sample_direction(rng)
        mx += v[1]; my += v[2]; mz += v[3]
        maxdev = max(maxdev, abs(norm(v) - 1.0))
    end
    return mx / n, my / n, mz / n, maxdev
end

@testset "Philox core + philox_rand" begin
    key = (0x0badf00d, 0x00000005)
    ctr = (0x00000011, 0x00000000, 0x00000000, 0x00000000)
    # bijection is deterministic and 4x UInt32
    out = philox4x32(ctr, key)
    @test out isa NTuple{4,UInt32}
    @test philox4x32(ctr, key) === out
    # philox_rand is a pure function of (key, ctr): same inputs -> same output
    @test philox_rand(key, ctr) === philox_rand(key, ctr)
    # a different counter gives a different draw
    @test philox_rand(key, ctr) != philox_rand(key, (0x00000012, 0x00000000, 0x00000000, 0x00000000))
    # strictly in [0, 1)
    for i in 1:10000
        u = philox_rand(key, (UInt32(i), 0x00000000, 0x00000000, 0x00000000))
        @test 0.0 <= u < 1.0
    end
    # convenience seeding agrees with the explicit (key, ctr) form
    @test philox_rand(7, 3, 11) === philox_rand((UInt32(7), UInt32(3)),
                                                (UInt32(11), 0x00000000, 0x00000000, 0x00000000))
end

@testset "Philox uniformity" begin
    m, v = philox_moments(1_000_000)
    @test m ≈ 0.5 atol = 5e-3          # E[U] = 1/2
    @test v ≈ 1 / 12 atol = 5e-3       # Var[U] = 1/12
end

@testset "stream reproducibility + independence" begin
    # two MCRng streams with identical seed/id reproduce the same sequence
    a = MCRng(42, 1); b = MCRng(42, 1)
    for _ in 1:1000
        ua, a = mc_uniform(a)
        ub, b = mc_uniform(b)
        @test ua === ub
    end
    # a different stream id decorrelates (SE ~ 1/sqrt(n) ~ 2.2e-3)
    @test abs(stream_correlation(200_000)) < 1.5e-2
end

@testset "sample_direction isotropy" begin
    mx, my, mz, maxdev = direction_moments(200_000)
    @test maxdev < 1e-12               # exactly on the unit sphere
    @test abs(mx) < 1e-2               # <n> = 0 (isotropic)
    @test abs(my) < 1e-2
    @test abs(mz) < 1e-2
end

@testset "chi_square moments" begin
    # chi^2_k has mean k, variance 2k
    for k in (3, 4, 6)
        m, v = sampler_moments(rng -> chi_square(k, rng), k, 200_000)
        @test m ≈ k atol = 0.1
        @test v ≈ 2k atol = 0.6
    end
end

@testset "standard_gamma moments" begin
    # Gamma(a,1) has mean a, variance a (covers shape>=1 and the shape<1 boost)
    for a in (0.5, 1.5, 2.5, 5.0)
        m, v = sampler_moments(rng -> standard_gamma(a, rng), round(Int, 10a), 200_000)
        @test m ≈ a atol = 0.05
        @test v ≈ a atol = 0.15
    end
end
