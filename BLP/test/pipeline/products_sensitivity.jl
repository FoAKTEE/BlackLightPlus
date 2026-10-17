# Parameter derivatives and the mass-scale ladder: the derivative of the image and the
# flux against central differences for both parameters, the thermal fraction carried by
# the derivative lane, the ladder against separate renders, executor independence, and
# the argument errors.
using Test, BLP, StaticArrays
using BLP.Plans: CameraConfig, PlasmaConfig, RunConfig, MixedLanes, prepare
using BLP.Products: Image, render, sensitivity, variants
using BLP.Spacetime: KerrSchildCartesian
using BLP.Matter: PolishDoughnut
using BLP.Transfer: StokesI, IQUV
using BLP.Execution: PoolExecutor
const Decks = Main.Decks

const MS = 3.0e-18 * (6.6743e-8 * 6.2e9 * 1.989e33 / 2.99792458e10^2)^3
cfg(; ms = MS, rat_high = 10.0, thermal_frac = 1.0) =
    RunConfig(mass_scale_cgs = ms, camera = CameraConfig(fov = 24.0, resolution = 10),
              plasma = PlasmaConfig(rat_high = rat_high, thermal_frac = thermal_frac))
plan(c = cfg(); matter = Decks.uniform_deck(), kw...) =
    prepare(c; spacetime = KerrSchildCartesian(0.9), matter, mode = StokesI(), kw...)
flux(c) = sum(render(plan(c); backend = :cpu).I)

@testset "sensitivity and variants" begin
    @testset "derivatives against central differences" begin
        p = plan()
        img = render(p; backend = :cpu)
        h = 1e-4
        s = sensitivity(p, :mass_scale; backend = :cpu)
        @test s.param === :mass_scale && s.image isa Image && size(s.derivative) == (10, 10)
        @test isapprox(s.image.I, img.I; rtol = 1e-9) && isapprox(s.flux, sum(img.I); rtol = 1e-9)
        fd = (flux(cfg(ms = MS * (1 + h))) - flux(cfg(ms = MS * (1 - h)))) / (2h * MS)
        @test s.dflux != 0 && isapprox(s.dflux, fd; rtol = 1e-5)
        @test isapprox(s.dflux, sum(s.derivative); rtol = 1e-9)
        r = sensitivity(p, :rat_high; backend = :host_pool)
        fd = (flux(cfg(rat_high = 10.0 + 1e-3)) - flux(cfg(rat_high = 10.0 - 1e-3))) / 2e-3
        @test r.dflux != 0 && isapprox(r.dflux, fd; rtol = 1e-4)
        @test Decks.bits(r.derivative) == Decks.bits(sensitivity(p, :rat_high; backend = :cpu).derivative)
        @test_throws ArgumentError sensitivity(p, :spin; backend = :cpu)
    end

    @testset "the derivative lane carries the thermal fraction" begin
        half = sensitivity(plan(cfg(thermal_frac = 0.5)), :rat_high; backend = :cpu)
        full = sensitivity(plan(cfg()), :rat_high; backend = :cpu)
        @test 0 < half.flux < full.flux
        @test all(isfinite, half.derivative) && any(!iszero, half.derivative)
    end

    @testset "mass-scale ladder" begin
        p = plan()
        ms = [MS, 2.5 * MS, 0.4 * MS]
        v = variants(p, ms; backend = :cpu)
        @test length(v.images) == 3 && v.mass_scales == ms && v.plan === p
        for k in 1:3
            @test isapprox(v.images[k], render(plan(cfg(ms = ms[k])); backend = :cpu).I; rtol = 1e-9)
        end
        @test sum(v.images[2]) > sum(v.images[1]) > sum(v.images[3])
        w = variants(p, ms; backend = PoolExecutor(Decks.hosts(2); lanes = 2, chunk = 7))
        @test Decks.bits(w.images) == Decks.bits(v.images)
        @test Decks.bits(variants(plan(; lanes = MixedLanes()), ms; backend = :cpu).images[1]) ==
              Decks.bits(variants(plan(; lanes = MixedLanes()), ms; backend = :host_pool).images[1])
        @test_throws ArgumentError variants(p, Float64[]; backend = :cpu)
        @test_throws ArgumentError variants(plan(; matter = PolishDoughnut(0.9)), ms; backend = :cpu)
    end
end
