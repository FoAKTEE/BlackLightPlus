# Time-dependent equatorial sources: the orbiting Gaussian spot against AART's flare model
# and the emission coordinates against Krang's slow-light ones at stored screen points; the
# amplitude and intensity of a landing; vacuum at invalid landings; the hot-path contract.
using Test, BlackLightPlus
using BlackLightPlus.Spacetime: pixel_prep, band_landing, Landing
using BlackLightPlus.Matter: EquatorialDisk, EquatorialSpot, GaussianSpot, spot_amplitude, intensity

const Decks = Main.Decks
allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

@testset "equatorial spot" begin
    a = 0.94; θo = deg2rad(17.0)
    spot = EquatorialSpot(EquatorialDisk(a), GaussianSpot(; r_c = 8.0, omega = 0.05, width = 0.5))
    @testset "AART flare model" begin
        # (n, α, β, t_obs, I, t_s) of AART flare_model / calculate_observables at D_obs = 1e4
        ref = ((0, -8.0836820083682, 4.1673640167364, 15.707963267948966, 0.7191677115517912, -10016.467249598069),
               (0, 1.8577405857740583, -8.0836820083682, 57.59586531581288, 0.4891256149892551, -10012.655889977395),
               (1, 2.8619246861924683, -5.071129707112971, 15.707963267948966, 0.39958332402363234, -10036.912471366342),
               (1, 2.6610878661087867, 4.87029288702929, 57.59586531581288, 0.20506461547323607, -10039.159547749996))
        for (n, α, β, t, I, ts) in ref
            l = band_landing(pixel_prep(α, β, θo, a; ro = 1e4), n)
            # n = 1: the ring's conditioning lifts the rounding to ~1e-12 of t_s
            @test l.t ≈ ts rtol = (n == 0 ? 1e-13 : 1e-11)
            @test intensity(spot, l, t) ≈ I rtol = (n == 0 ? 1e-10 : 1e-7)
        end
    end
    @testset "Krang slow-light coordinates (observer at infinity)" begin
        # (α, β, n, r_s, t_s, φ_s) from Krang emission_coordinates: t = −t_Krang, φ = −ϕ_Krang − π/2
        ref = ((3.0, -4.0, 0, 4.4369361530535105, 2.7601895293304244, 5.2437812215435375),
               (-6.5, 2.0, 0, 5.74304889890143, 1.7066677370928491, 2.769792731785488),
               (5.2, 2.1, 1, 4.135659358523722, -16.473439226016577, 2.774856055750834),
               (-2.0, 9.0, 0, 7.964546131371041, 0.5676243004359977, 1.7428730851957166))
        for (α, β, n, r, t, φ) in ref
            l = band_landing(pixel_prep(α, β, θo, a; ro = Inf), n)
            @test l.r ≈ r rtol = 1e-12
            @test l.t ≈ t atol = (n == 0 ? 1e-9 : 1e-7)
            @test mod(l.φ, 2π) ≈ φ atol = (n == 0 ? 1e-10 : 1e-8)
        end
    end
    @testset "amplitude, intensity, vacuum" begin
        l = band_landing(pixel_prep(-6.5, 2.0, θo, a; ro = Inf), 0)
        A, x, y, ts = spot_amplitude(spot, l)
        @test A > 0 && hypot(x, y) ≈ l.r && ts == l.t
        @test intensity(spot, l, 3.0) == A * spot.profile(ts + 3.0, x, y)
        @test spot.profile(0.0, 8.0, 0.0) == 1.0
        @test spot.profile(2π / 0.05, 8.0, 0.0) ≈ 1.0                  # one period later
        bad = Landing{Float64}(NaN, NaN, NaN, 1, false, 1.0, 2.0, π / 2, 1)
        @test spot_amplitude(spot, bad) === (0.0, 0.0, 0.0, 0.0)
        @test intensity(spot, bad, 1.0) === 0.0
        k = EquatorialSpot(EquatorialDisk(a); r_c = 6.0)
        @test k.profile.omega ≈ 1 / (6.0^1.5 + a)                       # Keplerian by default
        @test allocs(intensity, spot, l, 1.0) == 0
        @test isbits(spot) && Decks.device_isbits(spot)
    end
end
