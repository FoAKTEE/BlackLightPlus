# Geodesic engine: conservation, the Schwarzschild shadow, photon-orbit radii, directional
# escape, and the time orientation of camera rays.
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
using BlackLightPlus.Spacetime: KerrSchildCartesian, metric, nullify_covariant, null_error, trace_geodesic,
                     angular_momentum_z, energy, geodesic_rhs, ks_radius, horizon_radius,
                     photon_orbit_radius, PlaneParallelCamera, CameraFrame, camera_ray,
                     pixel_centres, pixel_coords, npixels
const Decks = Main.Decks

@testset "geodesics" begin
    @testset "conservation" begin
        st = KerrSchildCartesian(0.9)
        x0 = SVector(0.0, 100.0, 6.0, 2.0)
        y0 = vcat(x0, nullify_covariant(st, x0, SVector(-1.0, 0.05, 0.02)))
        @test abs(null_error(st, y0)) < 1e-12
        sol = trace_geodesic(st, y0; escape = 110.0, reltol = 1e-11, abstol = 1e-13)
        yf = sol.u[end]
        @test yf[5] == y0[5]                                    # p_t conserved by construction
        @test abs(angular_momentum_z(yf) - angular_momentum_z(y0)) < 1e-8
        @test abs(null_error(st, yf)) < 1e-8
    end

    @testset "Schwarzschild shadow b_c = 3√3" begin
        st = KerrSchildCartesian(0.0)
        d = 1000.0
        captured(b) = begin
            x0 = SVector(0.0, d, b, 0.0)
            p = metric(st, x0) * SVector(1.0, -1.0, 0.0, 0.0)
            p0 = nullify_covariant(st, x0, SVector(p[2], p[3], p[4]))
            u = trace_geodesic(st, vcat(x0, p0); escape = 1.5d, reltol = 1e-9, abstol = 1e-9).u[end]
            ks_radius(st, SVector(u[1], u[2], u[3], u[4])) < 10.0
        end
        lo, hi = 4.5, 6.0
        @test captured(lo)
        @test !captured(hi)
        for _ in 1:30
            mid = (lo + hi) / 2
            captured(mid) ? (lo = mid) : (hi = mid)
        end
        @test abs((lo + hi) / 2 - 3sqrt(3.0)) / (3sqrt(3.0)) < 1e-3
    end

    @testset "Kerr photon-orbit radii" begin
        @test abs(photon_orbit_radius(0.9) - 1.5579) < 1e-3
        @test abs(photon_orbit_radius(0.9; prograde = false) - 3.9103) < 1e-3
        @test photon_orbit_radius(0.0) ≈ 3.0
    end

    @testset "escape is outbound only" begin
        st = KerrSchildCartesian(0.9)
        x0 = SVector(0.0, 2000.01, 0.0, 0.0)                   # starts beyond the escape sphere
        p = metric(st, x0) * SVector(1.0, -1.0, 0.0, 0.0)
        y0 = vcat(x0, nullify_covariant(st, x0, SVector(p[2], p[3], p[4])))
        u = trace_geodesic(st, y0).u[end]
        @test ks_radius(st, SVector(u[1], u[2], u[3], u[4])) < horizon_radius(st) + 1e-2
        states, _ = Decks.sampled_trace(st, [y0]; C = 40.0, escape = 1.1e3, nmax = 6000)
        @test length(states[1]) > 2
        xin = SVector(0.0, 1000.0, 40.0, 0.0)
        pin = metric(st, xin) * SVector(1.0, -1.0, 0.0, 0.0)
        yin = vcat(xin, nullify_covariant(st, xin, SVector(pin[2], pin[3], pin[4])))
        u = trace_geodesic(st, yin; escape = 1100.0).u[end]
        @test ks_radius(st, SVector(u[1], u[2], u[3], u[4])) > 1.0e3
    end

    @testset "a camera ray is the backward-traced photon" begin
        for a in (0.0, 0.9), inc in (30.0, 60.0, 90.0)
            st = KerrSchildCartesian(a)
            fr = CameraFrame(PlaneParallelCamera(inc_deg = inc, d = 1.0e7, fov = 10.0, npix = 1))
            for (α, β) in ((4.0, 1.0), (-3.0, 2.5), (0.0, -5.0))
                y, ok = camera_ray(st, fr, α, β)
                @test ok && y[5] == 1.0                         # p = -k, k_t = -1
                @test geodesic_rhs(st, y)[1] < 0                # x^t decreases along the trace
                @test angular_momentum_z(y) / energy(y) ≈ -α * sind(inc) atol = 1e-5   # Bardeen λ
            end
        end
        cam = PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, fov = 20.0, npix = 4)
        c = pixel_centres(cam)
        @test c ≈ [-7.5, -2.5, 2.5, 7.5] && npixels(cam) == 16
        @test pixel_coords(c, 4, 1) == (c[1], c[1]) && pixel_coords(c, 4, 2) == (c[1], c[2])
        @test pixel_coords(c, 4, 5) == (c[2], c[1])
        fr = CameraFrame(cam)
        @test fr.up ≈ SVector(0.0, -cosd(60.0), 0.0, sind(60.0))
        @test (fr.sin_i, fr.cos_i, fr.d) == (sin(deg2rad(60.0)), cos(deg2rad(60.0)), 1000.0)
    end
end
