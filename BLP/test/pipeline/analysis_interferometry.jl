# Interferometry: the Tukey window's closed forms, the Radon cut's axis-aligned sums and
# symmetry, the baseline axis and normalisation of the visibility amplitude, the minima
# finder's conventions, and the ring ruler: a thin ring of diameter d μas shows
# visibility minima spaced by 1/(d MUAS_TO_RAD 10⁹) Gλ to a few per cent.
using Test, BLP
using BLP.Analysis: MUAS_TO_RAD, tukey_window, radon_cut, visamp, visamp_minima

# Thin Gaussian ring of angular diameter d and width wsig (μas) on an N² grid of field fov.
function ring_image(N, fov, d, wsig)
    Δ = fov / N; c = (N + 1) / 2
    return [exp(-0.5 * ((hypot((iα - c) * Δ, (iβ - c) * Δ) - d / 2) / wsig)^2)
            for iβ in 1:N, iα in 1:N]
end

@testset "interferometry" begin
    @testset "constants and Tukey window" begin
        @test MUAS_TO_RAD ≈ deg2rad(1 / 3600) * 1e-6 rtol = 1e-15
        w = tukey_window(101, 0.5)
        @test length(w) == 101 && w[51] == 1.0
        @test w[1] < 1e-12 && w[end] < 1e-12
        @test w == reverse(w)
        @test all(==(1.0), w[26:76])                              # flat centre
        @test tukey_window(101) == w
        @test tukey_window(1, 0.5) == [1.0]
        @test tukey_window(9, 0.0) == ones(9) == tukey_window(9, -1.0)
        n = 33
        hann = [0.5 * (1 - cos(2π * k / (n - 1))) for k in 0:n-1]
        @test tukey_window(n, 1.0) ≈ hann atol = 1e-15
        @test tukey_window(n, 2.5) == tukey_window(n, 1.0)
    end

    @testset "Radon cut" begin
        img = ring_image(256, 100.0, 30.0, 1.0)
        proj, sax = radon_cut(img, 0.0; fov_muas = 100.0)
        @test length(proj) == length(sax) == 256
        @test all(>=(0), proj)
        @test isapprox(proj, reverse(proj); rtol = 0, atol = 1e-9 * maximum(proj))
        @test sax ≈ [(k - 128.5) * 100.0 / 256 for k in 1:256] && sax == -reverse(sax)
        # axis-aligned: column sums at 0°, row sums at 90° (to the cos 90° round-off)
        A = rand(16, 16)
        @test radon_cut(A, 0.0; fov_muas = 1.0)[1] ≈ vec(sum(A; dims = 1))
        @test radon_cut(A, 90.0; fov_muas = 1.0)[1] ≈ vec(sum(A; dims = 2))
        @test radon_cut(A, 180.0; fov_muas = 1.0)[1] ≈ reverse(vec(sum(A; dims = 1)))
        # the projection keeps the total flux while the cut stays inside the field
        B = zeros(32, 32); B[12:20, 10:22] .= rand(9, 13)
        for ang in (17.0, 45.0, 123.0)
            @test sum(radon_cut(B, ang; fov_muas = 1.0)[1]) ≈ sum(B)
        end
        @test_throws AssertionError radon_cut(rand(4, 5), 0.0; fov_muas = 1.0)
    end

    @testset "baselines, normalisation, minima" begin
        N = 128; fov = 100.0; pad = 8
        img = ring_image(N, fov, 40.0, 2.0)
        res = visamp(img; angles = (0, 45.0), fov_muas = fov, pad = pad)
        @test length(res) == 2 && res[1].angle === 0.0 && res[2].angle === 45.0
        r = res[1]
        dbl = 1 / (pad * N * fov / N) / MUAS_TO_RAD / 1e9          # Gλ per FFT bin
        @test r.baselines[1] == 0.0 && length(r.baselines) == pad * N ÷ 2
        @test r.baselines ≈ dbl .* (0:pad*N÷2-1)
        @test r.amplitude[1] == 1.0 && all(<=(1.0 + 1e-12), r.amplitude)
        rc = visamp(img; angles = (0.0,), fov_muas = fov, pad = pad, maxbaseline = 5.0)[1]
        @test all(<(5.0), rc.baselines) && rc.baselines == r.baselines[1:length(rc.baselines)]
        rn = visamp(img; angles = (0.0,), fov_muas = fov, pad = pad, window = :none)[1]
        @test rn.baselines == r.baselines && rn.amplitude != r.amplitude
        # minima: strict on the left, weak on the right, above bl_min, skipping the first
        bl = collect(0.0:1.0:9.0)
        amp = [5.0, 3.0, 4.0, 2.0, 2.0, 3.0, 1.0, 4.0, 0.5, 0.7]
        @test visamp_minima(bl, amp; skip = 0) == [1.0, 3.0, 6.0, 8.0]
        @test visamp_minima(bl, amp) == [3.0, 6.0, 8.0]
        @test visamp_minima(bl, amp; skip = 0, bl_min = 3.0) == [6.0, 8.0]
        @test visamp_minima(bl, amp; skip = 4) == [1.0, 3.0, 6.0, 8.0]   # too few: all kept
    end

    @testset "ring ruler" begin
        N = 512; fov = 160.0; wsig = 1.5
        for d in (30.0, 45.0)
            img = ring_image(N, fov, d, wsig)
            pred = 1 / (d * MUAS_TO_RAD * 1e9)
            for r in visamp(img; angles = (0.0, 90.0), fov_muas = fov, pad = 16, maxbaseline = 120.0)
                mins = visamp_minima(r.baselines, r.amplitude; skip = 1, bl_min = 2.0)
                @test length(mins) >= 5
                sp = diff(mins)
                @test abs(sum(sp) / length(sp) - pred) / pred < 0.03
            end
        end
    end
end
