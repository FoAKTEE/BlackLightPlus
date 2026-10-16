# The hot-path contract of the analytic providers, the g-disk and the emission zone:
# no allocation per sample in Float64, isbits records that remain isbits after
# adaptation to a device, and one concrete return type for Float64, Float32 and Dual
# positions.
using Test, BLP, StaticArrays, ForwardDiff
using BLP.Spacetime: KerrSchildCartesian, Minkowski, metric_and_inverse, lane, pixel_prep, band_landing
using BLP.Matter: PolishDoughnut, PageThorneDisk, EquatorialHotspot, UniformSphere, EquatorialDisk,
                  EmissionZone, plasma_state, sample, intensity, emission_zone

const Decks = Main.Decks

allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))

@testset "Matter hot path" begin
    a = 0.9375
    st = KerrSchildCartesian(a)
    provs = (PolishDoughnut(a; lambda = 0.5), PageThorneDisk(a), EquatorialHotspot(a),
             EquatorialHotspot(a; ambient_n_e = 1.0), UniformSphere(; R = 30.0))
    xs = (SVector(0.0, 8.0, 1.0, 0.2), SVector(0.0, 0.0, 0.0, 9.0), SVector(0.0, 30.0, -20.0, 1.0),
          SVector(3.0, 5.0, 3.0, 0.0))
    disk = EquatorialDisk(a)
    l = band_landing(pixel_prep(-4.0, 3.0, deg2rad(17.0), a), 1)
    sph = UniformSphere()
    x0 = SVector(0.0, 0.1, 0.2, 0.3)
    g0 = metric_and_inverse(Minkowski(), x0)

    @testset "zero allocations" begin
        for p in provs, x in xs
            g = metric_and_inverse(st, x)
            @test allocs(plasma_state, p, st, x, g, nothing) == 0
            @test allocs(sample, p, st, x, nothing) == 0
        end
        @test allocs(intensity, disk, l) == 0
        @test allocs(emission_zone, sph, x0, g0) == 0
    end

    @testset "isbits records" begin
        for p in (provs..., disk, emission_zone(sph, x0, g0))
            @test isbits(p)
            @test Decks.device_isbits(p)
        end
    end

    @testset "concrete return types" begin
        st32 = lane(st, Float32)
        x32 = SVector{4,Float32}(xs[1])
        xd = SVector(0.0, ForwardDiff.Dual(8.0, 1.0), 1.0, 0.2)
        for p in provs
            for (s, x) in ((st, xs[1]), (st32, x32), (st, xd))
                g = metric_and_inverse(s, x)
                rt = Base.return_types(plasma_state, (typeof(p), typeof(s), typeof(x), typeof(g), Nothing))
                @test isconcretetype(only(rt))
            end
        end
        @test isconcretetype(only(Base.return_types(intensity, (typeof(disk), typeof(l)))))
        @test isconcretetype(only(Base.return_types(emission_zone, (typeof(sph), typeof(x0), typeof(g0)))))
    end
end
