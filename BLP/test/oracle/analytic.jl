# Oracle: the exact-Kerr landings, per-pixel state, critical curve and orbit primitives of
# BLP.Spacetime against the old analytic tracer, bit for bit (values and ForwardDiff
# partials).
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))

# The result, or the exception type (both packages throw alike).
tryf(f) = try f() catch err; typeof(err) end
const thrown = Any[]

landing_tuple(l) = (l.r, l.φ, l.t, l.νr, l.valid, l.λ, l.η)

@testset "oracle: analytic Kerr" begin
    rng = Xoshiro(20261017)

    @testset "landings" begin
        nbad = 0; nvalid = 0; ntot = 0
        for a in (-0.5, 0.0, 1e-7, 0.3, 0.5, 0.94, 0.998), inc in (17.0, 60.0, 89.999, 120.0, 163.0),
            _ in 1:60
            θo = deg2rad(inc)
            α = 12 * (2rand(rng) - 1); β = rand(rng) < 0.05 ? 0.0 : 12 * (2rand(rng) - 1)
            ro = rand(rng, (1000.0, 1.0e5, 250.0))
            pn = tryf(() -> NS.pixel_prep(α, β, θo, a; ro = ro))
            po = tryf(() -> OG._emission_prep(α, β, θo, a; ro = ro))
            nbad += !(pn === po)
            if pn isa Type
                push!(thrown, (α, β, inc, a, ro, pn))
                continue
            end
            for n in 0:3
                ln = NS.band_landing(pn, n)
                lo = OG._analytic_emission_full(α, β, θo, a, n; ro = ro)
                nbad += !same(landing_tuple(ln), lo)
                nvalid += ln.valid; ntot += 1
            end
        end
        @test nbad == 0
        @test nvalid > ntot ÷ 4
        isempty(thrown) || @info "inputs (α, β, inc°, a, ro) where both tracers throw" thrown
    end

    @testset "partials" begin
        nbad = 0
        for (α, β, inc, a0) in [(-5.0, 2.0, 60.0, 0.94), (2.0, 4.0, 17.0, 0.94),
                                (-6.0, -1.0, 60.0, 0.5), (5.0, 2.0, 60.0, 0.3), (-3.5, -0.1, 17.0, 0.94)]
            θ0 = deg2rad(inc)
            for n in 0:1
                fa_n(a) = NS.band_landing(NS.pixel_prep(α, β, θ0, a; ro = 1.0e5), n).r
                fa_o(a) = OG._analytic_emission_full(α, β, θ0, a, n; ro = 1.0e5)[1]
                ft_n(θ) = NS.band_landing(NS.pixel_prep(α, β, θ, a0; ro = 1.0e5), n).φ
                ft_o(θ) = OG._analytic_emission_full(α, β, θ, a0, n; ro = 1.0e5)[2]
                nbad += !(ForwardDiff.derivative(fa_n, a0) === ForwardDiff.derivative(fa_o, a0))
                nbad += !(ForwardDiff.derivative(ft_n, θ0) === ForwardDiff.derivative(ft_o, θ0))
            end
        end
        @test nbad == 0
    end

    @testset "conserved quantities, roots, turning points, integrals" begin
        nbad = 0
        for _ in 1:2000
            a = rand(rng) * 0.999; θo = rand(rng) * π / 2; α = 10randn(rng); β = 10randn(rng)
            λ, η = NS.conserved_quantities(α, β, θo, a)
            nbad += !same((λ, η), OG.conserved_quantities(α, β, θo, a))
            nbad += !same(NS.radial_roots(λ, η, a), OG.radial_roots(λ, η, a))
            η > 0 || continue
            up, um = NS.angular_turning_points(λ, η, a)
            nbad += !same((up, um), OG.angular_turning_points(λ, η, a))
            nbad += !same(NS.analytic_angular_integrals(1, β, up, um, θo, a),
                          OG.analytic_angular_integrals(1, β, up, um, θo, a))
        end
        @test nbad == 0
    end

    @testset "critical curve and orbit primitives" begin
        for a in (0.0, 1e-9, 0.5, 0.94, 0.998), inc in (17.0, 60.0, 85.0, 120.0)
            an, bn = NS.critical_curve(a, deg2rad(inc); n = 64)
            ao, bo = OG.critical_curve(a, deg2rad(inc); n = 64)
            @test reinterpret(UInt64, collect(an)) == reinterpret(UInt64, collect(ao))
            @test reinterpret(UInt64, collect(bn)) == reinterpret(UInt64, collect(bo))
        end
        as = rand(rng, 500) .* 0.999; rs = 1.5 .+ 20 .* rand(rng, 500)
        @test all(NS.kerr_horizon(a) === OG.kerr_horizon(a) && NS.isco_radius(a) === OG.isco_radius(a) &&
                  NS.isco_radius(a; prograde = false) === OG.isco_radius(a; prograde = false) &&
                  NS.photon_orbit_radius(a) === OG.photon_orbit_radius(a) &&
                  NS.marginally_bound_radius(a) === OG.marginally_bound_radius(a) &&
                  NS.keplerian_omega(a, r) === OG.keplerian_omega(a, r) &&
                  NS.keplerian_specific_angular_momentum(a, r) === OG.keplerian_specific_angular_momentum(a, r) &&
                  NS.kerr_radial_potential(r, a, 2.0, 20.0) === OG.kerr_radial_potential(r, a, 2.0, 20.0)
                  for (a, r) in zip(as, rs))
    end
end
