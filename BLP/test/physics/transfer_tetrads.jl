# Frames of the polarized transfer: the fluid tetrad is orthonormal on Kerr and on a
# binary (with e_(1) renormalised there), finite when the field lies along the photon,
# the camera's normal observer is a unit timelike vector, and the Stokes rotation between
# two frames is a rotation of (Q, U) by 2χ.
using Test, BLP, StaticArrays, LinearAlgebra, Random
using BLP.Transfer: fluid_tetrad, fluid_tetrad_checked, normal_observer, qu_rotation, rotate_qu
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, metric_and_inverse, ks_radius,
                     null_covariant
using BLP.Matter: primitive_four_vectors

const ETA = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))

# A fluid sample at a random point: four-vectors and a null wave covector with k_t = -1.
function fluid_sample(rng, st)
    while true
        x = SVector(0.0, 12.0 .* (2.0 .* rand(rng, SVector{3,Float64}) .- 1.0)...)
        outside_horizons(st, x) || continue
        gcov, gcon = metric_and_inverse(st, x)
        k, ok = null_covariant(st, x, randn(rng, SVector{3,Float64}))
        ok || continue
        kcov = k / (-k[1])
        fv = primitive_four_vectors(0.4randn(rng, SVector{3,Float64}), randn(rng, SVector{3,Float64}),
                                    gcov, gcon)
        return (; gcov, gcon, kcov, kcon = gcon * kcov, fv)
    end
end
outside_horizons(st::KerrSchildCartesian, x) = ks_radius(st, x) > 1.5
outside_horizons(st::SuperposedKerrSchild, x) = ks_radius(st, x, 1) > 1.0 && ks_radius(st, x, 2) > 1.0

@testset "tetrads" begin
    rng = Xoshiro(7)
    @testset "orthonormal frames ($(nameof(typeof(st))))" for (st, renorm) in
            ((KerrSchildCartesian(0.9), false),
             (SuperposedKerrSchild([0.5, 0.5], [0.3, -0.2], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)]), true))
        worst = 0.0
        for _ in 1:500
            s = fluid_sample(rng, st)
            tet, ok = fluid_tetrad_checked(s.fv.ucon, s.fv.ucov, s.kcon, s.kcov, s.fv.bcon, s.gcov, s.gcon;
                                           renorm_e1 = renorm)
            @test ok
            worst = max(worst, maximum(abs.(tet * s.gcov * transpose(tet) - ETA)))
            # e_(0) is the fluid velocity
            @test tet[1, :] == s.fv.ucon
        end
        @test worst < 1e-8
    end

    @testset "field along the photon" begin
        g = ETA
        u = SVector(1.0, 0.0, 0.0, 0.0); k = SVector(1.0, 1.0, 0.0, 0.0); b = SVector(0.0, 1.0, 0.0, 0.0)
        tet = fluid_tetrad(u, g * u, k, g * k, b, g, g)
        @test all(isfinite, tet)
        @test tet * g * transpose(tet) ≈ ETA atol = 1e-12
        # a wave vector orthogonal to u admits no spacelike screen leg: e_(2) is zero, not NaN
        kx = SVector(0.0, 1.0, 0.0, 0.0)
        tet2, ok = fluid_tetrad_checked(u, g * u, kx, g * kx, b, g, g)
        @test !ok && tet2[3, :] == zero(b)
    end

    @testset "normal observer" begin
        st = KerrSchildCartesian(0.9375)
        gcov, gcon = metric_and_inverse(st, SVector(0.0, 3.0, 2.0, 1.5))
        n = normal_observer(gcon)
        @test dot(n, gcov * n) ≈ -1.0 atol = 1e-14
        @test gcov * n ≈ SVector(-1 / sqrt(-gcon[1, 1]), 0.0, 0.0, 0.0) atol = 1e-14
    end

    @testset "Stokes rotation between frames" begin
        st = KerrSchildCartesian(0.9)
        for _ in 1:200
            s = fluid_sample(rng, st)
            tf = fluid_tetrad(s.fv.ucon, s.fv.ucov, s.kcon, s.kcov, s.fv.bcon, s.gcov, s.gcon)
            te = fluid_tetrad(s.fv.ucon, s.fv.ucov, s.kcon, s.kcov, randn(rng, SVector{4,Float64}), s.gcov, s.gcon)
            c, sn = qu_rotation(tf, te, s.gcov)
            @test c^2 + sn^2 ≈ 1.0 atol = 1e-12
            b = randn(rng, SVector{4,Float64}); M = randn(rng, SMatrix{4,4,Float64})
            rb = rotate_qu(c, sn, b)
            @test rb[1] == b[1] && rb[4] == b[4] && hypot(rb[2], rb[3]) ≈ hypot(b[2], b[3])
            R = @SMatrix [1.0 0.0 0.0 0.0; 0.0 c sn 0.0; 0.0 -sn c 0.0; 0.0 0.0 0.0 1.0]
            @test rotate_qu(c, sn, M) ≈ R * M * transpose(R) rtol = 1e-12
            # the rotation to the frame itself is the identity
            ci, si = qu_rotation(tf, tf, s.gcov)
            @test ci ≈ 1.0 && abs(si) < 1e-10
        end
    end
end
