# The Monte Carlo emission zones: the `EmissionZone` constructors (keyword widening, a
# plasma state, the number type), and the uniform sphere as a provider (the closed
# ball, the static fluid and the unit field along x¹ in flat spacetime, its volume
# and domain radius) and as the zone a superphoton is drawn from.
using Test, BlackLightPlus, StaticArrays, ForwardDiff
using LinearAlgebra: dot
using BlackLightPlus.Spacetime: Minkowski, KerrSchildCartesian, metric_and_inverse
using BlackLightPlus.Matter: EmissionZone, UniformSphere, PlasmaState, AnalyticProvider, emission_zone,
                  plasma_state, sample, domain_radius, vacuum

const G_MINK = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

@testset "emission zones" begin
    @testset "EmissionZone" begin
        ucon = SVector(1.0, 0.0, 0.0, 0.0)
        z = EmissionZone(; n_e = 1.0e13, theta_e = 100, B = 1.0f0, dV = 1, x = SVector(0, 0, 0, 0),
                         ucon = ucon, ucov = G_MINK * ucon, bcon = SVector(0.0, 1.0, 0.0, 0.0),
                         gcov = G_MINK, gcon = G_MINK)
        @test z isa EmissionZone{Float64} && isbits(z)
        @test z.theta_e === 100.0 && z.B === 1.0 && z.dV === 1.0 && z.x === SVector(0.0, 0.0, 0.0, 0.0)
        @test z.ucov == SVector(-1.0, 0.0, 0.0, 0.0) && z.gcov === G_MINK
        z32 = EmissionZone{Float32}(; n_e = 1.0e13, theta_e = 100, B = 1, dV = 1, x = z.x, ucon = z.ucon,
                                    ucov = z.ucov, bcon = z.bcon, gcov = z.gcov, gcon = z.gcon)
        @test z32 isa EmissionZone{Float32} && z32.gcov isa SMatrix{4,4,Float32,16}

        st = KerrSchildCartesian(0.9)
        x = SVector(1.0, 6.0, 2.0, 1.0)
        g = metric_and_inverse(st, x)
        s = PlasmaState{Float64}(2.0, 3.0, 4.0, SVector(1.1, 0.1, 0.2, 0.0), SVector(-1.0, 0.1, 0.2, 0.0),
                                 SVector(0.0, 1.0, 0.0, 0.0), SVector(0.1, 1.0, 0.0, 0.0), 1.0, 0.0, 0.0, true)
        zs = EmissionZone(s, 5.0, x, g)
        @test (zs.n_e, zs.theta_e, zs.B, zs.dV) === (2.0, 3.0, 4.0, 5.0)
        @test zs.x === x && zs.ucon === s.ucon && zs.ucov === s.ucov && zs.bcon === s.bcon
        @test (zs.gcov, zs.gcon) === g
        xd = SVector(1.0, ForwardDiff.Dual(6.0, 1.0), 2.0, 1.0)
        @test EmissionZone(s, 5.0, xd, metric_and_inverse(st, xd)) isa EmissionZone{<:ForwardDiff.Dual}
    end

    @testset "UniformSphere" begin
        sph = UniformSphere()
        @test sph isa AnalyticProvider && isbits(sph)
        @test (sph.n_e, sph.theta_e, sph.B, sph.R) === (1.0e13, 100.0, 1.0, 1.0)
        @test sph.dV == (4.0 / 3.0) * π
        @test UniformSphere(; R = 2).dV ≈ (4 / 3) * π * 8
        @test domain_radius(UniformSphere(; R = 3.0e5)) === 3.0e5

        st = Minkowski()
        for x in (SVector(0.0, 0.1, -0.3, 0.5), SVector(7.0, 1.0, 0.0, 0.0), SVector(0.0, 0.0, -0.6, 0.8))
            g = metric_and_inverse(st, x)
            s = plasma_state(sph, st, x, g, nothing)
            @test s.valid                                               # closed ball, boundary included
            @test (s.n_e, s.theta_e, s.B) === (sph.n_e, sph.theta_e, sph.B)
            @test s.ucon == SVector(1.0, 0.0, 0.0, 0.0) && s.ucov == SVector(-1.0, 0.0, 0.0, 0.0)
            @test s.bcon == SVector(0.0, 1.0, 0.0, 0.0) && s.bcov == s.bcon && s.b_sq == 1.0
            @test dot(s.ucon, s.ucov) == -1.0 && dot(s.bcon, s.ucov) == 0.0
            @test s.sigma == 0.0 && s.beta_inv == 0.0
            @test sample(sph, st, x, nothing) === (s, g)

            z = emission_zone(sph, x, g)
            @test (z.n_e, z.theta_e, z.B, z.dV) === (sph.n_e, sph.theta_e, sph.B, sph.dV)
            @test z.x === x && z.ucon === s.ucon && z.ucov === s.ucov && z.bcon === s.bcon
            @test z.gcov === G_MINK && z.gcon === G_MINK
        end
        for x in (SVector(0.0, 1.0, 1.0e-7, 0.0), SVector(0.0, 0.0, 0.0, -2.0), SVector(0.0, 0.8, 0.8, 0.0))
            @test plasma_state(sph, st, x, metric_and_inverse(st, x), nothing) === vacuum(Float64)
        end

        # a Dual position or radius flows through the state and the zone
        xd = SVector(0.0, ForwardDiff.Dual(0.2, 1.0), 0.1, 0.0)
        gd = metric_and_inverse(st, xd)
        @test plasma_state(sph, st, xd, gd, nothing) isa PlasmaState{<:ForwardDiff.Dual}
        @test emission_zone(sph, xd, gd) isa EmissionZone{<:ForwardDiff.Dual}
        xo = SVector(0.0, ForwardDiff.Dual(3.0, 1.0), 0.1, 0.0)
        @test typeof(plasma_state(sph, st, xo, gd, nothing)) == typeof(plasma_state(sph, st, xd, gd, nothing))
        V(R) = UniformSphere(; R = R).dV
        @test ForwardDiff.derivative(V, 2.0) ≈ 4π * 4
    end
end
