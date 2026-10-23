# The equatorial g-disk: the redshift factor in its circular-orbit and large-radius
# limits and across the ISCO, the Johnson-SU profile, the intensity painted on the
# exact-Kerr landings (zero off the disk, demagnified from one photon-ring layer to the
# next), and ForwardDiff through the intensity.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using BlackLightPlus.Spacetime: conserved_quantities, isco_prograde, kerr_horizon, pixel_prep, band_landing, Landing
using BlackLightPlus.Matter: EquatorialDisk, intensity, gfactor_disk, gfactor_circular, gfactor_plunge,
                  johnson_su_profile, equatorial_disk_intensity

landing(α, β, θo, a, n; ro = 1.0e4) = band_landing(pixel_prep(α, β, θo, a; ro = ro), n)

@testset "equatorial g-disk" begin
    @testset "redshift factor" begin
        # Schwarzschild Keplerian orbit, λ = 0 photon: g = 1/uᵗ = √(1 − 3/r)
        for r in (6.0, 10.0, 30.0, 100.0)
            @test isapprox(gfactor_disk(r, 0.0, 1, 0.0, 0.0), sqrt(1 - 3 / r); rtol = 1e-10)
        end
        for a in (0.0, 0.5, 0.94)
            λ, η = conserved_quantities(0.0, 1.0, deg2rad(60.0), a)
            @test isapprox(gfactor_disk(2000.0, a, 1, λ, η), 1.0; atol = 2e-3)
        end
        # the plunging branch meets the circular one at the ISCO
        for a in (0.3, 0.6, 0.94)
            isco = isco_prograde(a)
            λ, η = conserved_quantities(-5.0, 2.0, deg2rad(17.0), a)
            @test isapprox(gfactor_disk(isco + 1e-6, a, 1, λ, η), gfactor_disk(isco - 1e-6, a, 1, λ, η); rtol = 1e-4)
            @test isapprox(gfactor_circular(isco, a, 1, λ, η), gfactor_plunge(isco, a, 1, λ, η); rtol = 1e-6)
        end
        a = 0.94
        λ, η = conserved_quantities(-4.0, 3.0, deg2rad(17.0), a)
        for r in range(isco_prograde(a) + 0.01, 20.0; length = 50)
            g = gfactor_disk(r, a, 1, λ, η)
            @test isfinite(g) && g > 0
        end
    end

    @testset "Johnson-SU profile" begin
        rr = collect(range(0.05, 40.0; length = 40000))
        for mu in (2.0, 3.5, 6.0)
            p = [johnson_su_profile(r; mu = mu, sigma = 0.5, gamma = 0.0) for r in rr]
            @test all(>(0), p)
            @test isapprox(rr[argmax(p)], mu; atol = 2e-3)
        end
        p = [johnson_su_profile(r; mu = 0.66, sigma = 0.5, gamma = -1.5) for r in rr]
        @test all(x -> x > 0 && isfinite(x), p)
        integral = sum((p[1:end-1] .+ p[2:end]) ./ 2 .* diff(rr))
        @test isfinite(integral) && integral > 0
        @test johnson_su_profile(40.0; mu = 0.66, sigma = 0.5, gamma = -1.5) <
              johnson_su_profile(2.0; mu = 0.66, sigma = 0.5, gamma = -1.5)
    end

    @testset "disk record and intensity of a landing" begin
        d = EquatorialDisk(0.94)
        @test isbits(d) && d isa EquatorialDisk{Float64}
        @test d.a == 0.94 && d.mu_p == 1 - sqrt(1 - 0.94^2) && d.redshift_power == 3.0
        @test EquatorialDisk(0.5f0; sub_kep = 1) isa EquatorialDisk{Float64}

        θo = deg2rad(17.0)
        knobs = (sub_kep = 0.8, betar = 0.6, betaphi = 0.9, gamma_p = -1.0, sigma_p = 0.7,
                 redshift_power = 4.0, rin = 2.5)
        dk = EquatorialDisk(0.94; knobs...)
        nvalid = 0; ninvalid = 0
        for α in range(-8.0, 8.0; length = 9), β in range(-8.0, 8.0; length = 9), n in 0:2
            l = landing(α, β, θo, 0.94, n)
            iv = intensity(dk, l)
            if l.valid
                nvalid += 1
                @test iv == equatorial_disk_intensity(l.r, l.νr, 0.94, l.λ, l.η; knobs...)
                @test (l.r < 2.5) == (iv == 0)
            else
                ninvalid += 1
                @test iv === 0.0
            end
        end
        @test nvalid > 0 && ninvalid > 0
        bad = Landing{Float64}(NaN, NaN, NaN, 1, false, 1.0, 2.0)
        @test intensity(d, bad) === 0.0
        @test equatorial_disk_intensity(kerr_horizon(0.94), 1, 0.94, 1.0, 2.0) == 0.0
    end

    @testset "photon-ring layers demagnify" begin
        a = 0.94
        θo = deg2rad(17.0)
        disk = EquatorialDisk(a)
        xs = range(-8.0, 8.0; length = 121)
        F = zeros(3); npx = zeros(Int, 3)
        for α in xs, β in xs
            prep = pixel_prep(α, β, θo, a; ro = 1.0e4)
            for n in 0:2
                iv = intensity(disk, band_landing(prep, n))
                F[n + 1] += iv; npx[n + 1] += iv > 0
            end
        end
        @test all(>(0), F)
        @test F[3] < F[2] < F[1]
        @test npx[3] < npx[2] < npx[1]
    end

    @testset "ForwardDiff" begin
        a = 0.94
        λ, η = conserved_quantities(-4.0, 3.0, deg2rad(17.0), a)
        rs = 4.0; h = 1e-6
        f_sk(sk) = equatorial_disk_intensity(rs, 1, a, λ, η; sub_kep = sk)
        f_mu(mu) = equatorial_disk_intensity(rs, 1, a, λ, η; mu_p = mu)
        f_a(aa) = equatorial_disk_intensity(rs, 1, aa, λ, η)
        # through the record and the landing: a spin derivative of a painted pixel
        f_l(aa) = intensity(EquatorialDisk(aa), landing(-4.0, 3.0, deg2rad(17.0), aa, 1))
        f_b(b) = intensity(EquatorialDisk(a; betaphi = b), landing(-4.0, 3.0, deg2rad(17.0), a, 0))
        for (f, x0) in ((f_sk, 1.0), (f_mu, 0.66), (f_a, a), (f_l, a), (f_b, 0.8))
            d = ForwardDiff.derivative(f, x0)
            fd = (f(x0 + h) - f(x0 - h)) / (2h)
            @test isfinite(d) && d != 0
            @test isapprox(d, fd; rtol = 1e-5)
        end
        ld = landing(-4.0, 3.0, deg2rad(17.0), a, 0)
        dd = EquatorialDisk(a; sub_kep = ForwardDiff.Dual(1.0, 1.0))
        rt = Base.return_types(intensity, (typeof(dd), typeof(ld)))
        @test isconcretetype(only(rt)) && only(rt) <: ForwardDiff.Dual
    end
end
