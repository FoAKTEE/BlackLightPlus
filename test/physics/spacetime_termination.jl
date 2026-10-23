# The one termination rule of the fixed-control march: the predicates, the Kerr,
# binary, ADM and horizonless controls, the redshift cut on a near-horizon deck and the
# affine length it omits, Δλ ≈ α_h / (κ E_c).
using Test, BlackLightPlus, StaticArrays, LinearAlgebra
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, Minkowski, MarchBudget, MarchStep,
                     march_control, march_captured, march_escaped, march_stop, normal_frame_energy,
                     ks_radius_energy, ks_radius, horizon_radius, rk4_step, PlaneParallelCamera,
                     HORIZON_EPS_SAMPLED, CAPTURE_REDSHIFT, lane
const Decks = Main.Decks

@testset "march termination" begin
    @testset "predicates" begin
        @test HORIZON_EPS_SAMPLED === 1e-3 && CAPTURE_REDSHIFT === 1.0e2
        @test !march_captured(0.5, 10.0)
        @test march_captured(0.5e-3, 10.0)                     # horizon backstop
        @test march_captured(0.5, 100.5)                       # redshift cut
        @test !march_captured(0.5, 100.0)                      # E_n = E_c still emits
        @test march_captured(NaN, 10.0) && march_captured(-Inf, 10.0) && march_captured(Inf, 10.0)
        @test march_captured(0.5, NaN)
        @test march_captured(0.5f0, 101.0f0) && !march_captured(0.5f0, 99.0f0)
        @test !march_captured(0.0, 10.0, 0.0) && march_captured(-1e-300, 10.0, 0.0)
        @test march_escaped(1200.0, 1100.0, 1100.0, Inf)
        @test !march_escaped(1200.0, 1300.0, 1100.0, Inf)      # inbound beyond escape: kept
        @test march_escaped(12.0, 11.0, 1100.0, 11.5)          # R_em early-out, outbound only
        @test march_stop(0.5, 101.0, 5.0, 6.0, 1100.0, Inf)
        @test march_stop(0.5, 1.0, 1200.0, 1100.0, 1100.0, Inf)
        @test !march_stop(0.5, 1.0, 5.0, 6.0, 1100.0, Inf)
        @test_throws MethodError march_stop(0.5f0, 1.0, 5.0f0, 6.0f0, 1.1f3, Inf32)
    end

    @testset "controls" begin
        a = 0.9; st = KerrSchildCartesian(a); b = MarchBudget(80.0, 1100.0, Inf, 6000)
        y = SVector(0.0, 1.2, 0.8, 0.3, 1.0, 0.4, -0.3, 0.2)
        r, En = ks_radius_energy(y, a)
        @test r == ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
        @test En ≈ normal_frame_energy(st, y) rtol = 1e-14
        m = march_control(st, y, Inf, b)
        @test m === MarchStep{Float64}(march_stop(r - st.r_h, En, r, Inf, 1100.0, Inf), (r - st.r_h) / 80.0, r)
        m32 = march_control(lane(st, Float32), SVector{8,Float32}(y), Inf32, lane(b, Float32))
        @test m32 isa MarchStep{Float32} && m32.stop == m.stop
        bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        yb = SVector(0.0, 3.5, 1.0, 0.2, 1.0, -0.4, 0.3, 0.2)
        mb = march_control(bin, yb, Inf, b)
        d = min(ks_radius(bin, SVector(yb[1], yb[2], yb[3], yb[4]), 1) - horizon_radius(bin, 1),
                ks_radius(bin, SVector(yb[1], yb[2], yb[3], yb[4]), 2) - horizon_radius(bin, 2))
        @test mb.h == d / 80.0 && mb.R == sqrt(3.5^2 + 1.0 + 0.04)
        # no horizon: steps on the coordinate radius, stops on escape only
        mf = march_control(Minkowski(), SVector(0.0, 1200.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0), 1100.0, b)
        @test mf.stop && mf.h == 1200.0 / 80.0
        @test !march_control(Minkowski(), SVector(0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0), Inf, b).stop
        @test march_control(Minkowski(), SVector(0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0), Inf, b).h == 1 / 80
    end

    @testset "near-horizon deck: the redshift cut stops every plunging ray" begin
        a = 0.9; st = KerrSchildCartesian(a); r_h = horizon_radius(st)
        cam = PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, fov = 10.0, npix = 12)
        rays = Decks.camera_rays(st, cam)
        states, dls = Decks.sampled_trace(st, rays; C = 80.0, escape = 1100.0, nmax = 6000,
                                          R_em = sqrt(3.0) * 6.0)
        captured = [ks_radius(st, SVector(s[1][1], s[1][2], s[1][3], s[1][4])) < 2.0 for s in states]
        @test count(captured) > length(rays) ÷ 2
        for i in findall(captured)
            s = states[i]                                      # source → camera
            @test !(normal_frame_energy(st, s[1]) <= CAPTURE_REDSHIFT)
            @test normal_frame_energy(st, s[2]) <= CAPTURE_REDSHIFT
            @test ks_radius(st, SVector(s[1][1], s[1][2], s[1][3], s[1][4])) - r_h > HORIZON_EPS_SAMPLED
        end
        # the affine length beyond the cut, continued to the horizon backstop alone
        rm = 1 - sqrt(1 - a^2)
        κ = (r_h - rm) / (2 * (r_h^2 + a^2))
        Δλ_est = 1 / (sqrt(2.0) * κ * CAPTURE_REDSHIFT)
        worst = 0.0
        for i in findall(captured)
            y = states[i][1]; rest = 0.0
            for _ in 1:6000
                r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
                (r - r_h < HORIZON_EPS_SAMPLED || !isfinite(r)) && break
                h = (r - r_h) / 80.0
                y = rk4_step(st, y, h); rest += h
            end
            worst = max(worst, rest)
        end
        @test 0 < worst <= 1.05 * Δλ_est
    end

    @testset "binary: rays end on a hole through the same rule" begin
        st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        rays = Decks.camera_rays(st, PlaneParallelCamera(inc_deg = 60.0, d = 1000.0, fov = 16.0, npix = 10))
        states, _ = Decks.sampled_trace(st, rays; C = 80.0, escape = 1100.0, nmax = 6000, R_em = 12.0)
        b = MarchBudget(80.0, 1100.0, 12.0, 6000)
        ncap = count(s -> march_control(st, s[1], Inf, b).stop, states)
        @test ncap > 0
        @test all(length(s) < 6000 for s in states)
    end
end
