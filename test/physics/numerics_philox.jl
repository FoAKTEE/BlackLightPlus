# The Philox stream is uniform, reproducible and independent across streams; the
# samplers hit their analytic moments; stream identities are injective over genealogies.
using Test, BlackLightPlus
using BlackLightPlus.Numerics: philox4x32, philox_rand, MCRng, mc_uniform, mc_gauss, sample_direction,
                    standard_gamma, chi_square, MCStreamLayout, primary_rng, child_rng,
                    stream_id, stream_layer
using LinearAlgebra: norm

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

# Pearson correlation of two streams drawn at the same counters.
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

@testset "Philox and samplers" begin
    @testset "core" begin
        key = (0x0badf00d, 0x00000005)
        ctr = (0x00000011, 0x00000000, 0x00000000, 0x00000000)
        out = philox4x32(ctr, key)
        @test out isa NTuple{4,UInt32}
        @test philox4x32(ctr, key) === out
        @test philox_rand(key, ctr) === philox_rand(key, ctr)
        @test philox_rand(key, ctr) != philox_rand(key, (0x00000012, 0x00000000, 0x00000000, 0x00000000))
        @test all(0.0 <= philox_rand(key, (UInt32(i), 0x00000000, 0x00000000, 0x00000000)) < 1.0
                  for i in 1:10000)
        @test philox_rand(7, 3, 11) === philox_rand((UInt32(7), UInt32(3)),
                                                    (UInt32(11), 0x00000000, 0x00000000, 0x00000000))
        # Random123 known-answer vector of Philox4x32-10 (zero counter and key)
        @test philox4x32((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000)) ===
              (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
    end

    @testset "uniformity and independence" begin
        m, v = philox_moments(1_000_000)
        @test m ≈ 0.5 atol = 5e-3
        @test v ≈ 1 / 12 atol = 5e-3
        a = MCRng(42, 1); b = MCRng(42, 1)
        same = true
        for _ in 1:1000
            ua, a = mc_uniform(a); ub, b = mc_uniform(b)
            same &= ua === ub
        end
        @test same
        @test abs(stream_correlation(200_000)) < 1.5e-2
    end

    @testset "sampler moments" begin
        mx, my, mz, maxdev = direction_moments(200_000)
        @test maxdev < 1e-12
        @test abs(mx) < 1e-2 && abs(my) < 1e-2 && abs(mz) < 1e-2
        for k in (3, 4, 6)
            m, v = sampler_moments(rng -> chi_square(k, rng), k, 200_000)
            @test m ≈ k atol = 0.1
            @test v ≈ 2k atol = 0.6
        end
        for a in (0.5, 1.5, 2.5, 5.0)
            m, v = sampler_moments(rng -> standard_gamma(a, rng), round(Int, 10a), 200_000)
            @test m ≈ a atol = 0.05
            @test v ≈ a atol = 0.15
        end
        m, v = sampler_moments(mc_gauss, 5, 200_000)
        @test abs(m) < 1e-2
        @test v ≈ 1 atol = 2e-2
    end

    @testset "stream identity" begin
        L = MCStreamLayout(5, 4, 3)
        @test (L.gid_bits, L.event_bits) == (3, 2)
        ids = Set{UInt128}()
        n = 0
        for gid in 1:5
            r0 = primary_rng(7, gid)
            push!(ids, stream_id(r0)); n += 1
            @test stream_layer(r0) == 0
            for e1 in 0:3
                r1 = child_rng(r0, e1, L)
                push!(ids, stream_id(r1)); n += 1
                @test stream_layer(r1) == 1
                for e2 in 0:3
                    r2 = child_rng(r1, e2, L)
                    push!(ids, stream_id(r2)); n += 1
                    for e3 in 0:3
                        r3 = child_rng(r2, e3, L)
                        push!(ids, stream_id(r3)); n += 1
                        @test stream_layer(r3) == 3
                    end
                end
            end
        end
        @test length(ids) == n == 5 * (1 + 4 + 16 + 64)
        # a child's stream does not depend on the parent's draw count
        r = primary_rng(7, 3)
        _, radv = mc_uniform(r)
        @test child_rng(radv, 2, L) == child_rng(r, 2, L)
        # a primary of gid < 2^32 is the stream MCRng(seed, gid)
        @test primary_rng(0xBEEF, 12345) == MCRng(0xBEEF, 12345)
        a = child_rng(primary_rng(1, 1), 0, MCStreamLayout(10^6, 10^5, 3))
        @test mc_uniform(a)[1] != mc_uniform(primary_rng(1, 1))[1]
        @test_throws ArgumentError MCStreamLayout(10^7, 100_000, 7)
        @test_throws ArgumentError MCStreamLayout(10, 10, 16)
        @test isbits(primary_rng(1, 1))
    end
end
