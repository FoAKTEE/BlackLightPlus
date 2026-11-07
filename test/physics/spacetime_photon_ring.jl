# Photon-ring orders of the equatorial disk: the successive screen offsets of one source
# radius from the critical curve shrink by e^{−γ}, and the landing azimuth and emission
# time step by 2π − δ and τ, the photon-shell exponents of the critical point; the
# lensing bands of orders up to 6 are found and narrow by about e^{−γ}.
using Test, BlackLightPlus
using BlackLightPlus.Spacetime: pixel_prep, band_landing, critical_point, photon_shell_exponents
using BlackLightPlus.Analysis: lensing_band

landing(a, θo, α0, β0, s, n) = band_landing(pixel_prep(s * α0, s * β0, θo, a; ro = Inf), n)

# s > 1 with r_n(s (α0, β0)) = rs: logarithmic scan in s − 1, then bisection
function offset_scale(a, θo, α0, β0, n, rs)
    f(s) = (l = landing(a, θo, α0, β0, s, n); l.valid ? l.r - rs : NaN)
    dprev = 0.5; fprev = f(1 + dprev)
    for k in 1:600
        d = 0.5 * 10.0^(-15 * k / 600)
        fd = f(1 + d)
        if isfinite(fd) && isfinite(fprev) && fd * fprev <= 0
            lo, hi, flo = 1 + d, 1 + dprev, fd
            for _ in 1:200
                m = (lo + hi) / 2
                (m == lo || m == hi) && break
                fm = f(m)
                isfinite(fm) || break
                (sign(fm) == sign(flo)) ? (lo = m; flo = fm) : (hi = m)
            end
            return (lo + hi) / 2
        end
        dprev, fprev = d, fd
    end
    return NaN
end

@testset "photon-ring orders" begin
    for (a, inc, rt) in ((0.94, 17.0, 2.3), (0.94, 17.0, 2.8), (0.99, 80.0, 3.2))
        θo = deg2rad(inc)
        α0, β0 = critical_point(a, θo, rt)
        @test isfinite(β0) && β0 > 0
        γ, δ, τ = photon_shell_exponents(a, rt)
        s = [offset_scale(a, θo, α0, β0, n, 6.0) for n in 4:6]
        @test all(isfinite, s)
        ls = [landing(a, θo, α0, β0, s[k], k + 3) for k in 1:3]
        @test all(l -> l.valid && isapprox(l.r, 6.0; rtol = 1e-6), ls)
        @test log((s[2] - 1) / (s[3] - 1)) ≈ γ rtol = 1e-4                 # demagnification
        @test mod(ls[3].φ - ls[2].φ, 2π) ≈ mod(-δ, 2π) atol = 1e-4         # rotation
        @test ls[2].t - ls[3].t ≈ τ rtol = 1e-5                            # delay
    end
    a, θo = 0.94, deg2rad(17.0)
    w = map(1:6) do n
        inner, outer = lensing_band(a, θo, n; npts = 24)
        sum(hypot(outer[i, 1], outer[i, 2]) - hypot(inner[i, 1], inner[i, 2]) for i in 1:24) / 24
    end
    @test all(>(0), w)
    γ̄ = sum(first(photon_shell_exponents(a, r)) for r in range(2.1, 2.9; length = 9)) / 9
    @test all(k -> 0.5 < log(w[k] / w[k + 1]) / γ̄ < 1.5, 3:5)
end
