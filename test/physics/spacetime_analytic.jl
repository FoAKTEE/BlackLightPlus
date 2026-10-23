# Exact-Kerr equatorial landings (AART/Krang closed forms): the engine primitives, the
# landing against the numerically traced first equatorial crossing, the critical curve,
# AD against finite differences, the Mino budget, the domain symmetries, the stable
# turning points and the finite-observer νr flip.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using BlackLightPlus.Spacetime: KerrSchildCartesian, PlaneParallelCamera, CameraFrame, camera_ray, ks_radius,
                     horizon_radius, trace_geodesic, conserved_quantities, radial_roots,
                     angular_turning_points, analytic_angular_integrals, pixel_prep, band_landing,
                     Landing, critical_curve, kerr_horizon

landing(α, β, θo, a, n; ro = 1000.0) = band_landing(pixel_prep(α, β, θo, a; ro = ro), n)

# Kerr-Schild radii (= Boyer-Lindquist r) of every equatorial crossing of a traced ray.
function equatorial_crossings(st, y0; d)
    sol = trace_geodesic(st, y0; λmax = 3.0 * d, reltol = 1e-11, abstol = 1e-13,
                         escape = 1.02 * d, horizon_eps = 1e-5, save_everystep = true)
    cr = Float64[]
    for k in 1:length(sol.t)-1
        z0 = sol.u[k][4]; z1 = sol.u[k+1][4]
        (z0 == 0 || sign(z0) == sign(z1)) && continue
        lo, hi = sol.t[k], sol.t[k+1]
        for _ in 1:80
            mid = (lo + hi) / 2
            (sign(sol(mid)[4]) == sign(z0)) ? (lo = mid) : (hi = mid)
        end
        uc = sol((lo + hi) / 2)
        push!(cr, ks_radius(st, SVector(uc[1], uc[2], uc[3], uc[4])))
    end
    return cr
end

@testset "analytic Kerr" begin
    @testset "engine primitives" begin
        λ, η = conserved_quantities(2.0, 3.0, deg2rad(50.0), 0.9)
        @test λ ≈ -2.0 * sin(deg2rad(50.0))
        @test η ≈ (4.0 - 0.81) * cos(deg2rad(50.0))^2 + 9.0
        for (a, α, β) in ((0.9, -4.0, 2.0), (0.5, 3.0, -1.0), (0.94, -6.0, 0.0))
            λ, η = conserved_quantities(α, β, deg2rad(45.0), a)
            Rr(r) = (r^2 + a^2 - a * λ)^2 - (r^2 - 2r + a^2) * (η + (λ - a)^2)
            @test all(abs(Rr(r)) < 1e-8 * max(1.0, abs(r)^4) for r in radial_roots(λ, η, a))
        end
        λ2, η2 = conserved_quantities(-6.0, 2.0, deg2rad(60.0), 0.94)
        @test count(z -> abs(imag(z)) < 1e-8, radial_roots(λ2, η2, 0.94)) == 4   # case II
        λ3, η3 = conserved_quantities(-1.0, 0.3, deg2rad(60.0), 0.94)
        @test count(z -> abs(imag(z)) < 1e-8, radial_roots(λ3, η3, 0.94)) == 2   # case III
        l = landing(-5.0, 2.0, deg2rad(60.0), 0.94, 0)
        @test l isa Landing{Float64} && l.valid && l.νr in (-1, 1)
        @test l.λ == conserved_quantities(-5.0, 2.0, deg2rad(60.0), 0.94)[1]
    end

    @testset "landing == first equatorial crossing of the traced ray" begin
        d = 1.0e6
        for a in (0.001, 0.5, 0.94), inc in (17.0, 60.0)
            st = KerrSchildCartesian(a); θo = deg2rad(inc)
            fr = CameraFrame(PlaneParallelCamera(inc, d, 40.0, 64))
            for (α, β) in ((5.0, 2.0), (-5.0, 2.0), (4.0, -3.0), (2.0, 4.0))
                y0 = first(camera_ray(st, fr, α, β))
                cr = equatorial_crossings(st, y0; d)
                isempty(cr) && continue
                l = landing(α, β, θo, a, 0; ro = ks_radius(st, SVector(y0[1], y0[2], y0[3], y0[4])))
                @test l.valid && isfinite(l.φ) && isfinite(l.t) && l.r > horizon_radius(st)
                @test abs(l.r - cr[1]) / cr[1] < 5e-5
            end
        end
    end

    @testset "critical curve" begin
        R0 = 3 * sqrt(3.0)
        for inc in (17.0, 60.0, 89.0)
            αc, βc = critical_curve(0.0, deg2rad(inc); n = 200)
            @test maximum(abs.(sqrt.(αc .^ 2 .+ βc .^ 2) .- R0)) < 1e-10
        end
        αc, βc = critical_curve(0.94, deg2rad(60.0); n = 400)
        @test maximum(αc) + minimum(αc) > 0.3                  # D-shaped, shifted right
        @test 8.0 < maximum(αc) - minimum(αc) < 11.0
        @test 9.0 < maximum(βc) - minimum(βc) < 11.0
        α1, β1 = critical_curve(0.94, deg2rad(120.0); n = 64)
        α2, β2 = critical_curve(0.94, deg2rad(60.0); n = 64)
        @test maximum(abs.(α1 - α2)) < 1e-10 && maximum(abs.(β1 - β2)) < 1e-10
    end

    @testset "ForwardDiff == finite differences" begin
        for (α, β, inc, a0) in ((-5.0, 2.0, 60.0, 0.94), (2.0, 4.0, 17.0, 0.94),
                                (-6.0, -1.0, 60.0, 0.5), (5.0, 2.0, 60.0, 0.3))
            θ0 = deg2rad(inc)
            r_a(a) = landing(α, β, θ0, a, 0; ro = 1.0e5).r
            r_θ(θ) = landing(α, β, θ, a0, 0; ro = 1.0e5).r
            h = 1e-6
            @test ForwardDiff.derivative(r_a, a0) ≈ (r_a(a0 + h) - r_a(a0 - h)) / 2h rtol = 1e-6
            @test ForwardDiff.derivative(r_θ, θ0) ≈ (r_θ(θ0 + h) - r_θ(θ0 - h)) / 2h rtol = 1e-6
        end
    end

    @testset "Mino budget rejects aliased landings" begin
        θo = deg2rad(17.0)
        l0 = landing(-3.5, -0.1, θo, 0.94, 0)
        @test l0.valid && isapprox(l0.r, 2.6226742150344196; rtol = 1e-10)
        @test all(!landing(-3.5, -0.1, θo, 0.94, n).valid for n in 1:3)
        edges = range(-12.0, 12.0; length = 41)
        cent = (edges[1:end-1] .+ edges[2:end]) ./ 2
        nvalid(n) = count(landing(-x, y, θo, 0.94, n).valid for x in cent, y in cent)
        @test nvalid(0) > 1000
        @test nvalid(2) < 100
    end

    @testset "domain symmetries" begin
        for (α, β) in ((-5.0, 2.0), (5.0, 2.0), (4.0, -3.0), (2.0, 4.0))
            up = landing(α, -β, deg2rad(60.0), 0.94, 0)
            lo = landing(α, β, deg2rad(120.0), 0.94, 0)
            @test lo.valid == up.valid
            if up.valid
                @test isapprox(lo.r, up.r; rtol = 1e-9) && isapprox(lo.φ, up.φ; rtol = 1e-9)
                @test isapprox(lo.t, up.t; rtol = 1e-9) && lo.νr == up.νr
            end
        end
        o = landing(-5.0, 2.0, deg2rad(163.0), 0.94, 0)
        @test o.valid && o.r > kerr_horizon(0.94)
        o0 = landing(-5.0, 2.0, deg2rad(60.0), 0.0, 0)
        oe = landing(-5.0, 2.0, deg2rad(60.0), 1e-6, 0)
        @test o0.valid && all(isfinite, (o0.r, o0.φ, o0.t))
        @test isapprox(o0.r, oe.r; rtol = 1e-8) && isapprox(o0.r, 3.845428284554; rtol = 1e-6)
        pro = landing(5.0, 2.0, deg2rad(60.0), 0.5, 0)
        ret = landing(-5.0, 2.0, deg2rad(60.0), -0.5, 0)
        @test pro.valid && ret.valid
        @test isapprox(ret.r, pro.r; rtol = 1e-12) && isapprox(ret.t, pro.t; rtol = 1e-12)
        @test isapprox(ret.φ, -π - pro.φ; atol = 1e-10) && ret.νr == pro.νr
        lim = landing(-5.0, 1e-9, deg2rad(60.0), 0.94, 0)
        for b in (0.0, -0.0)                                    # β = 0 takes H(0) = 1
            l = landing(-5.0, b, deg2rad(60.0), 0.94, 0)
            @test l.valid && isapprox(l.r, lim.r; rtol = 1e-6) && isapprox(l.φ, lim.φ; rtol = 1e-5)
        end
    end

    @testset "turning points stable at small spin" begin
        θo = deg2rad(60.0)
        for a in (1e-2, 1e-4, 1e-5, 1e-6)
            λ, η = conserved_quantities(-5.0, 2.0, θo, a)
            u_p, u_m = angular_turning_points(λ, η, a)
            Δb = (1 - (big(η) + big(λ)^2) / big(a)^2) / 2
            db = sqrt(Δb^2 + big(η) / big(a)^2)
            @test abs(u_p - Float64(Δb + db)) <= 1e-13 * Float64(Δb + db)
            @test abs(u_m - Float64(Δb - db)) <= 1e-13 * abs(Float64(Δb - db))
        end
    end

    @testset "νr flips at the finite-observer turning time" begin
        a = 0.94; θo = deg2rad(17.0); ro = 1000.0; n = 1; α = 5.10
        gap(β) = (p = pixel_prep(α, β, θo, a; ro = ro);
                  first(analytic_angular_integrals(n, β, angular_turning_points(p.λ, p.η, a)..., θo, a)) - p.τturn)
        βa, βb = -2.005, -1.995
        @test gap(βa) > 0 && gap(βb) < 0
        for _ in 1:80
            mid = (βa + βb) / 2
            (gap(mid) > 0) == (gap(βa) > 0) ? (βa = mid) : (βb = mid)
        end
        p = pixel_prep(α, (βa + βb) / 2, θo, a; ro = ro)
        @test isapprox(band_landing(p, n).r, real(p.roots[4]); rtol = 1e-9)   # flip at r4
        pw = pixel_prep(α, -2.000262457545051, θo, a; ro = ro)
        Gθ = first(analytic_angular_integrals(n, -2.000262457545051,
                                              angular_turning_points(pw.λ, pw.η, a)..., θo, a))
        lw = band_landing(pw, n)
        @test lw.valid
        @test pw.τturn < Gθ <= pw.τtot - pw.τturn + 1e-9 * pw.τtot   # inside the window
        @test lw.νr == -1 && lw.r > real(pw.roots[4])
    end

    @testset "near-critical conditioning" begin
        a = 0.94; θo = deg2rad(60.0)
        αc, βc = critical_curve(a, θo; n = 400)
        i = 100; nc = length(αc)
        tα = αc[i+1] - αc[i-1]; tβ = βc[i+1] - βc[i-1]; s = hypot(tα, tβ)
        probe(δ) = (αc[i] + δ * tβ / s, βc[i] - δ * tα / s)
        Δr(δ) = abs(landing(probe(δ)..., θo, a, 0).r - landing(probe(-δ)..., θo, a, 0).r)
        @test Δr(1e-3) < 5e-3
        @test Δr(1e-5) < 5e-5
        @test 1e-5 < Δr(1e-9) < 1e-3
        for (ae, ie, αe, βe) in ((0.94, 89.999, -5.0, 2.0), (0.94, 89.999999, -5.0, 2.0),
                                 (0.998, 60.0, -5.0, 1e-9), (0.9999, 60.0, -5.0, 2.0),
                                 (0.999999, 60.0, -5.0, 2.0), (0.94, 60.0, -5.0, 1e-12),
                                 (1.0 - 1e-12, 45.0, -4.0, 1.5))
            l = landing(αe, βe, deg2rad(ie), ae, 0)
            @test l.valid && isfinite(l.r) && isfinite(l.φ) && isfinite(l.t)
        end
    end
end
