# `prepare` and its records: the configuration defaults, every derived field of a plan
# on Kerr, a binary, an ADM grid and a user metric, the lanes and their validation, the
# rescue set of mixed lanes, and the transfer parameter set with nonthermal populations
# and seeded derivatives.
using Test, BLP, StaticArrays
using ForwardDiff: Dual, value, partials
using BLP.Plans: CameraConfig, PlasmaConfig, RunConfig, Lanes, MixedLanes, Plan, prepare,
                 transfer_params, geodesic_type
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, UserMetric, Minkowski, ADMGrid,
                     adm_grid_from_metric, metric, PlaneParallelCamera, CameraFrame, MarchBudget,
                     pixel_centres, npixels, critical_curve, horizon_radius
using BLP.Matter: ElectronModel, domain_radius
using BLP.Transfer: StokesI, IQUV, PowerLaw, Kappa, TransferParams, output_type
using BLP.Numerics: Units
using BLP.Execution: cost_order, rescue_pixels, curve_inclination
const Decks = Main.Decks

cfg(; res = 12, kw...) = RunConfig(; camera = CameraConfig(fov = 24.0, resolution = res), kw...)

@testset "plans" begin
    @testset "configuration defaults" begin
        c = RunConfig()
        @test (c.max_steps, c.C, c.escape_factor) == (2500, 80.0, 1.1)
        @test c.camera == CameraConfig(60.0, 1000.0, 40.0, 64)
        p = PlasmaConfig()
        @test (p.gamma, p.rat_high, p.sigma_cut, p.theta_e_min, p.power_frac, p.kappa_frac) ==
              (13.0 / 9.0, 10.0, 1.0, -1.0, 0.0, 0.0)
        @test !hasfield(RunConfig, :bh_a) && !hasfield(RunConfig, :polarized) &&
              !hasfield(RunConfig, :tracer)
    end

    @testset "Kerr plan" begin
        grid = Decks.uniform_deck()
        c = cfg()
        p = prepare(c; spacetime = KerrSchildCartesian(0.9), matter = grid)
        @test p isa Plan && p.config === c && p.provider === grid && p.mode === IQUV()
        @test p.horizon === 1 + sqrt(1 - 0.9 * 0.9) && p.horizon === horizon_radius(p.spacetime)
        @test p.escape === 1.1 * 1000.0
        @test p.R_em === sqrt(3.0) * 12.0 === domain_radius(grid)
        @test p.budget === MarchBudget{Float64}(80.0, 1.1 * 1000.0, sqrt(3.0) * 12.0, 2500)
        @test p.geodesic_budget === p.budget && p.geodesic_spacetime === p.spacetime
        @test p.camera === PlaneParallelCamera(60.0, 1000.0, 24.0, 12)
        @test p.frame === CameraFrame(p.camera)
        @test p.pixels.centres == pixel_centres(p.camera) && p.pixels.npix == 12
        @test p.theta_o === deg2rad(60.0)
        @test p.curve == critical_curve(0.9, curve_inclination(60.0))
        @test p.queue == cost_order(p.curve, 24.0, 12; tile = 16)
        @test sort(p.queue) == 1:144 && isempty(p.rescue)
        @test npixels(p) == 144 && output_type(p) === SVector{4,Float64}
        @test output_type(prepare(c; spacetime = KerrSchildCartesian(0.9), matter = grid,
                                  mode = StokesI())) === Float64
        @test prepare(c; spacetime = KerrSchildCartesian(0.9), matter = grid, tile = 4).queue ==
              cost_order(p.curve, 24.0, 12; tile = 4)
        @test occursin("KerrSchildCartesian", sprint(show, p))
        pn = prepare(c; spacetime = KerrSchildCartesian(0.9), matter = nothing)
        @test pn.R_em === Inf && pn.provider === nothing
    end

    @testset "lanes" begin
        grid = Decks.uniform_deck()
        st = KerrSchildCartesian(0.9)
        p32 = prepare(cfg(); spacetime = st, matter = grid, lanes = Lanes{Float32,Float64,Float64}())
        @test p32.geodesic_spacetime === KerrSchildCartesian{Float32}(Float32(0.9), Float32(st.r_h))
        @test p32.geodesic_budget === MarchBudget{Float32}(80.0f0, Float32(1.1 * 1000.0),
                                                          Float32(sqrt(3.0) * 12.0), 2500)
        @test p32.spacetime === st && isempty(p32.rescue) && geodesic_type(p32.lanes) === Float32
        pm = prepare(cfg(); spacetime = st, matter = grid, lanes = MixedLanes())
        @test pm.geodesic_spacetime === p32.geodesic_spacetime && pm.budget === p32.budget
        @test pm.rescue == rescue_pixels(pm.curve, 24.0, 12; rescue_px = 2)
        @test issorted(pm.rescue) && 0 < length(pm.rescue) < 144
        @test length(prepare(cfg(); spacetime = st, matter = grid, lanes = MixedLanes(0)).rescue) <
              length(pm.rescue)
        @test geodesic_type(MixedLanes()) === Float32 && geodesic_type(Lanes()) === Float64
        @test_throws ArgumentError MixedLanes(-1)
        @test_throws ArgumentError prepare(cfg(); spacetime = st, matter = grid,
                                           lanes = Lanes{Float64,Float32,Float64}())
        @test_throws ArgumentError prepare(cfg(); spacetime = KerrSchildCartesian{Float32}(0.9f0, 1.4f0),
                                           matter = grid)
    end

    @testset "other spacetimes" begin
        grid = Decks.uniform_deck()
        bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        p = prepare(cfg(); spacetime = bin, matter = grid)
        @test p.horizon == bin.r_h && p.queue == 1:144 && isempty(p.curve[1])
        @test p.geodesic_spacetime === p.spacetime
        @test_throws ArgumentError prepare(cfg(); spacetime = bin, matter = grid, lanes = MixedLanes())
        @test prepare(cfg(); spacetime = bin, matter = grid,
                      lanes = Lanes{Float32,Float64,Float64}()).geodesic_spacetime isa
              SuperposedKerrSchild{2,Float32}
        xs = -6.0:1.0:6.0
        adm = adm_grid_from_metric(x -> metric(KerrSchildCartesian(0.5), x), xs, xs, xs)
        pa = prepare(cfg(); spacetime = adm, matter = nothing)
        @test pa.horizon === adm.alpha_min && pa.geodesic_spacetime === adm
        @test_throws ArgumentError prepare(cfg(); spacetime = adm, matter = nothing,
                                           lanes = Lanes{Float32,Float64,Float64}())
        @test isnan(prepare(cfg(); spacetime = Minkowski(), matter = nothing).horizon)
        um = UserMetric(x -> metric(KerrSchildCartesian(0.3), x))
        @test isnan(prepare(cfg(); spacetime = um, matter = grid).horizon)
    end

    @testset "transfer parameters" begin
        c = cfg(plasma = PlasmaConfig(rat_high = 20.0, sigma_cut = 0.5, theta_e_min = 2.0,
                                      use_p = false))
        tp = transfer_params(c)
        @test tp isa TransferParams{Float64,Nothing,Nothing}
        @test tp.units === Units(c.m_bh_cgs, c.mass_scale_cgs) && tp.nu_obs === c.frequency_hz
        @test tp.electrons === ElectronModel(rat_high = 20.0, gamma = 13.0 / 9.0, use_p = false)
        @test (tp.cuts.sigma.lo, tp.cuts.sigma.hi) === (-Inf, 0.5)
        @test (tp.cuts.theta_e.lo, tp.cuts.theta_e.hi) === (2.0, Inf)
        @test tp.cuts.rho.lo === -Inf && tp.cuts.b.hi === Inf
        off = transfer_params(cfg(plasma = PlasmaConfig(sigma_cut = -1.0)))
        @test (off.cuts.sigma.lo, off.cuts.sigma.hi, off.cuts.theta_e.lo) === (-Inf, Inf, -Inf)
        nt = transfer_params(cfg(plasma = PlasmaConfig(thermal_frac = 0.4, power_frac = 0.3,
                                                       kappa_frac = 0.3, kappa = 4.5, w = 0.7)))
        @test nt.powerlaw === PowerLaw(0.3, 3.5, 25.0, 1.0e7)
        @test nt.kappa === Kappa(0.3, 4.5, 0.7)
        p = prepare(c; spacetime = KerrSchildCartesian(0.9), matter = Decks.uniform_deck())
        @test p.params === tp
        dm = transfer_params(c; mass_scale = Dual(c.mass_scale_cgs, 1.0))
        @test value(dm.units.dens_unit) === tp.units.dens_unit && partials(dm.units.dens_unit, 1) > 0
        @test output_type(StokesI(), dm) <: Dual
        dr = transfer_params(c; rat_high = Dual(20.0, 1.0))
        @test dr.electrons isa ElectronModel{<:Dual} && partials(dr.electrons.rat_high, 1) == 1.0
        @test transfer_params(c; mass_scale = 2c.mass_scale_cgs).units ===
              Units(c.m_bh_cgs, 2c.mass_scale_cgs)
    end
end
