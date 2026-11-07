# Closed-form landings on surfaces at any polar angle: the equator reproduces the equatorial
# tracer bit for bit; cone landings, ordinary and vortical (cases II, III and IV), match the
# crossings of numerically traced rays in radius, order and polar sign; the image orders
# hold every crossing exactly once; the observer at infinity is the limit of a far
# observer; the photon momentum, the Walker–Penrose constant and the screen polarization
# obey their identities; ForwardDiff passes through a cone landing.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using LinearAlgebra: dot
using BlackLightPlus.Spacetime: KerrSchildCartesian, PlaneParallelCamera, CameraFrame, camera_ray,
                                ks_radius, trace_geodesic, pixel_prep, band_landing, surface_prep,
                                surface_landing, image_crossings, kerr_bl_metric, photon_momentum,
                                zamo_components, zamo_vector, rest_components, penrose_walker,
                                screen_polarization, photon_shell_exponents, kerr_horizon

# (radius, sign of d cos θ along the trace) of every crossing of the cone cos θ = c by the
# camera ray y0, in the order of the trace (backward in time: the sign is νθ of the photon)
function cone_crossings(st, y0, c; d)
    sol = trace_geodesic(st, y0; λmax = 3.0 * d, reltol = 1e-12, abstol = 1e-14,
                         escape = 1.02 * d, horizon_eps = 1e-5, save_everystep = true)
    f(u) = u[4] - ks_radius(st, SVector(u[1], u[2], u[3], u[4])) * c
    out = Tuple{Float64,Int}[]
    for k in 1:length(sol.t)-1
        z0 = f(sol.u[k]); z1 = f(sol.u[k+1])
        (z0 == 0 || sign(z0) == sign(z1)) && continue
        lo, hi = sol.t[k], sol.t[k+1]
        for _ in 1:80
            mid = (lo + hi) / 2
            (sign(f(sol(mid))) == sign(z0)) ? (lo = mid) : (hi = mid)
        end
        uc = sol((lo + hi) / 2)
        push!(out, (ks_radius(st, SVector(uc[1], uc[2], uc[3], uc[4])), Int(sign(z1 - z0))))
    end
    return out
end

valid_landings(q, θs; mmax = 8) = [l for l in (surface_landing(q, θs, m) for m in 0:mmax) if l.valid]

@testset "analytic surfaces" begin
    @testset "the equator is the equatorial tracer, bit for bit" begin
        nb = 0
        for a in (0.94, -0.5), inc in (17.0, 120.0), ro in (1e4, Inf)
            θo = deg2rad(inc)
            for α in range(-9.7, 9.3; length = 15), β in range(-9.9, 9.1; length = 15)
                p = pixel_prep(α, β, θo, a; ro = ro); q = surface_prep(α, β, θo, a; ro = ro)
                for n in 0:3
                    l = band_landing(p, n)
                    m1, _, k = image_crossings(q, π / 2, n)
                    k == 0 && (@test !l.valid; continue)
                    @test k == 1
                    s = surface_landing(q, π / 2, m1)
                    @test s.valid == l.valid
                    if l.valid
                        @test (s.r, s.φ, s.t, s.νr, s.νθ) === (l.r, l.φ, l.t, l.νr, l.νθ)
                        nb += 1
                    end
                end
            end
        end
        @test nb > 1000
    end

    @testset "cone landings == crossings of the traced ray" begin
        d = 1.0e6
        stats = Dict{Tuple{Bool,Int},Int}()
        for (a, inc, θsd) in ((0.94, 17.0, 75.0), (0.94, 17.0, 105.0), (0.9, 60.0, 30.0),
                              (0.9, 60.0, 150.0), (0.94, 17.0, 12.0), (0.94, 17.0, 22.0),
                              (0.94, 163.0, 168.0))
            st = KerrSchildCartesian(a); θo = deg2rad(inc); θs = deg2rad(θsd)
            fr = CameraFrame(PlaneParallelCamera(inc, d, 40.0, 64))
            for α in (-5.0, -0.6, 0.0, 0.5, 4.0), β in (-6.0, -0.4, 0.3, 2.0, 5.0)
                y0 = first(camera_ray(st, fr, α, β))
                ro = ks_radius(st, SVector(y0[1], y0[2], y0[3], y0[4]))
                q = surface_prep(α, β, θo, a; ro = ro)
                cr = cone_crossings(st, y0, cos(θs); d)
                lv = valid_landings(q, θs)
                @test length(lv) == length(cr)
                for (l, c) in zip(lv, cr)
                    # the plane-parallel camera's O(1/d) offset grows along far, grazing crossings
                    @test abs(l.r - c[1]) < 1e-4 * c[1] * max(1.0, c[1] / 30)
                    @test l.νθ == c[2]
                end
                isempty(cr) || (stats[(q.vortical, q.rcase)] = get(stats, (q.vortical, q.rcase), 0) + length(cr))
            end
        end
        @test get(stats, (true, 3), 0) > 0 && get(stats, (true, 4), 0) > 0     # vortical, cases III and IV
        @test get(stats, (false, 2), 0) > 0 && get(stats, (false, 3), 0) > 0
    end

    @testset "image orders hold every crossing once" begin
        for (a, inc, θsd) in ((0.94, 17.0, 75.0), (0.9, 60.0, 30.0), (0.9, 60.0, 150.0),
                              (0.94, 17.0, 10.0), (0.94, 17.0, 25.0), (0.99, 80.0, 90.0))
            θo = deg2rad(inc); θs = deg2rad(θsd)
            for α in range(-6.0, 6.0; length = 9), β in range(-6.0, 6.0; length = 9)
                q = surface_prep(α, β, θo, a; ro = Inf)
                seen = Int[]
                for n in 0:9
                    m1, m2, k = image_crossings(q, θs, n)
                    k >= 1 && push!(seen, m1)
                    k == 2 && push!(seen, m2)
                end
                @test allunique(seen)
                valid = [m for m in 0:8 if surface_landing(q, θs, m).valid]
                @test issubset(valid, seen)
            end
        end
        # an out-of-cone surface: orders of one parity carry two crossings
        q = surface_prep(2.0, 4.0, deg2rad(60.0), 0.9; ro = Inf)
        @test image_crossings(q, deg2rad(30.0), 0) == (0, 1, 2)
        @test image_crossings(q, deg2rad(30.0), 1) == (0, 0, 0)
    end

    @testset "observer at infinity: the limit of a far observer" begin
        for (a, inc, θsd, α, β, m) in ((0.94, 17.0, 90.0, -6.0, -3.0, 0),   # case II
                                       (0.94, 17.0, 90.0, 5.0, 2.0, 1),     # case III
                                       (0.94, 17.0, 12.0, -0.25, 0.25, 0))  # vortical, case IV
            θo = deg2rad(inc); θs = deg2rad(θsd)
            li = surface_landing(surface_prep(α, β, θo, a; ro = Inf), θs, m)
            @test li.valid
            for ro in (1e4, 1e5)
                l = surface_landing(surface_prep(α, β, θo, a; ro = ro), θs, m)
                @test abs(l.r - li.r) < 50 / ro
                @test abs(l.φ - li.φ) < 50 / ro
                @test abs(l.t + ro + 2log(ro) - li.t) < 50 / ro      # t̃ = lim t + ro + 2 ln ro
            end
        end
    end

    @testset "photon momentum, Walker–Penrose constant, screen polarization" begin
        a = 0.9; r = 5.3; θ = 1.1; λ = -2.1; η = 14.0
        gcov, gcon = kerr_bl_metric(r, θ, a)
        @test gcov * gcon ≈ one(gcov) atol = 1e-12
        pcov, pcon = photon_momentum(r, θ, λ, η, a, -1, 1)
        @test abs(dot(pcov, pcon)) < 1e-12 && pcov[1] == -1 && pcov[4] == λ
        v = SVector(1.3, 0.2, -0.05, 0.11)
        @test zamo_vector(zamo_components(v, r, θ, a), r, θ, a) ≈ v
        vz = zamo_components(v, r, θ, a)
        @test dot(v, gcov * v) ≈ -vz[1]^2 + vz[2]^2 + vz[3]^2 + vz[4]^2
        uz = SVector(1.4, 0.3, 0.8, -0.5); uz = uz / sqrt(uz[1]^2 - uz[2]^2 - uz[3]^2 - uz[4]^2)
        uz = SVector(sqrt(1 + uz[2]^2 + uz[3]^2 + uz[4]^2), uz[2], uz[3], uz[4])
        @test rest_components(uz, uz) ≈ SVector(1.0, 0.0, 0.0, 0.0) atol = 1e-12
        @test rest_components(rest_components(vz, uz), uz; inverse = true) ≈ vz
        tv = SVector(1.0, 0.0, 0.0, 0.0)
        f0 = SVector(0.3, -0.7, 0.4, 0.2)
        f = f0 - dot(f0, gcov * pcon) / dot(tv, gcov * pcon) * tv
        @test abs(dot(f, gcov * pcon)) < 1e-12                       # f ⊥ p
        κ = penrose_walker(r, θ, a, pcon, f)
        @test abs2(κ) ≈ dot(f, gcov * f) * (η + (λ - a)^2) rtol = 1e-10
        @test penrose_walker(r, θ, a, pcon, f + 2.7 * pcon) ≈ κ      # gauge f → f + c p
        θo = deg2rad(40.0); α = -λ / sin(θo); β = sqrt(η + a^2 * cos(θo)^2 - λ^2 * cot(θo)^2)
        Eα, Eβ = screen_polarization(κ, α, β, θo, a)
        @test Eα^2 + Eβ^2 ≈ dot(f, gcov * f) rtol = 1e-10
    end

    @testset "photon-shell exponents: the Schwarzschild limit" begin
        γ, δ, τ = photon_shell_exponents(1e-4, 3.0 + 1e-4)          # inside the shell 3 ± 1.15e-4
        @test γ ≈ π rtol = 1e-6
        @test δ ≈ π rtol = 1e-3                                         # frame dragging: π + O(a)
        @test τ ≈ 3 * sqrt(3) * π rtol = 1e-6
    end

    @testset "ForwardDiff through a cone landing" begin
        α, β, θo, θs = 4.0, -3.0, deg2rad(17.0), deg2rad(75.0)
        r_a(a) = surface_landing(surface_prep(α, β, θo, a; ro = Inf), θs, 0).r
        r_s(s) = surface_landing(surface_prep(α, β, θo, 0.94; ro = Inf), s, 0).r
        h = 1e-6
        @test ForwardDiff.derivative(r_a, 0.94) ≈ (r_a(0.94 + h) - r_a(0.94 - h)) / 2h rtol = 1e-6
        @test ForwardDiff.derivative(r_s, θs) ≈ (r_s(θs + h) - r_s(θs - h)) / 2h rtol = 1e-6
    end
end
