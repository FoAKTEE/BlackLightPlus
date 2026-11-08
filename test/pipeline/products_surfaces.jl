# The exact-Kerr surface products: polarized band maps and their sums, several surfaces in
# one image, the g-disk through the surface path, the observer at infinity, executor
# independence, the band sums bitwise their broadcast forms and the argument checks.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare
using BlackLightPlus.Products: render_analytic, SurfaceProgram, surface_outputs
using BlackLightPlus.Spacetime: KerrSchildCartesian, surface_prep, surface_landing, image_crossings
using BlackLightPlus.Matter: EquatorialDisk, PolarizedDisk, SynchrotronSurface, polarized_emission
using BlackLightPlus.Execution: PoolExecutor
const Decks = Main.Decks

plan(; a = 0.94, res = 16, inc = 17.0) =
    prepare(RunConfig(camera = CameraConfig(inclination_deg = inc, fov = 20.0, resolution = res));
            spacetime = KerrSchildCartesian(a), matter = nothing)
cone(a, θs) = SynchrotronSurface(a; theta_s = θs, b = (0.3, 0.4, 0.8), beta = 0.6, phi_v = -1.7,
                                 sigma = 0.73)
const SFIELDS = (:I, :Q, :U, :Ib, :Qb, :Ub, :rs, :ts, :phis, :redshift_sign)
# the pool backend (the GPU under `BLP_TEST_DEVICES=cuda`): bitwise on one hardware, to
# 1e-9 of the largest value across hardware
agree(a, b) = Decks.CROSS_HARDWARE ?
              all(isnan.(a) .== isnan.(b)) && maximum(abs.(filter(!isnan, a .- b)); init = 0.0) <=
              1.0e-9 * maximum(abs.(filter(!isnan, b)); init = 0.0) : Decks.bits(a) == Decks.bits(b)

@testset "exact-Kerr surfaces" begin
    a = 0.94
    p = plan()
    mats = (cone(a, deg2rad(75.0)), cone(a, deg2rad(105.0)))
    r = render_analytic(p, mats; bands = (0, 1), ro = Inf, backend = :cpu)
    @test r.bands == [0, 1] && length(r.Ib) == 2 && r.plan === p
    @test all(size(m) == (16, 16) for f in SFIELDS[4:end] for m in getfield(r, f))
    @test r.I == r.Ib[1] + r.Ib[2] && r.Q == r.Qb[1] + r.Qb[2] && r.U == r.Ub[1] + r.Ub[2]
    bsum(maps) = (I = zero(maps[1]); for m in maps; I .+= m; end; I)
    @test all(Decks.bits(getfield(r, f)) == Decks.bits(bsum(getfield(r, g)))
              for (f, g) in ((:I, :Ib), (:Q, :Qb), (:U, :Ub)))
    @test r.I !== r.Q && r.Q !== r.U && all(getfield(r, f) isa Matrix{Float64} for f in (:I, :Q, :U))
    @test count(>(0), r.Ib[1]) > 100
    @test all(sqrt.(r.Qb[1] .^ 2 .+ r.Ub[1] .^ 2) .<= r.Ib[1] .* (1 + 1e-12) .+ 1e-300)
    # a pixel: the band sum of both cones' crossings
    xs = p.pixels.centres
    k = 9 * 16 + 5                                  # item k: α = xs[10], β = xs[5]
    α, β = xs[(k - 1) ÷ 16 + 1], xs[(k - 1) % 16 + 1]
    q = surface_prep(α, β, p.theta_o, a; ro = Inf)
    S = sum(mats) do m
        m1, m2, kk = image_crossings(q, m.theta_s, 0)
        s = kk >= 1 ? collect(polarized_emission(m, surface_landing(q, m.theta_s, m1), α, β, p.theta_o)) : zeros(3)
        kk == 2 && (s .+= collect(polarized_emission(m, surface_landing(q, m.theta_s, m2), α, β, p.theta_o)))
        s
    end
    @test (r.Ib[1][k], r.Qb[1][k], r.Ub[1][k]) == Tuple(S)
    # executors and devices: bitwise
    for exec in (:host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
        rr = render_analytic(p, mats; bands = (0, 1), ro = Inf, backend = exec)
        @test all(Decks.bits(getfield(rr, f)) == Decks.bits(getfield(r, f)) for f in SFIELDS)
    end
    rp = render_analytic(p, mats; bands = (0, 1), ro = Inf, backend = Decks.POOL_BACKEND)
    @test all(agree(getfield(rp, f)[b], getfield(r, f)[b]) for f in SFIELDS[4:end] for b in 1:2)
    # the program writes quantity q of band b to output (q - 1) K + b
    prog = SurfaceProgram(p.pixels, p.theta_o, a, Inf, mats, (0, 1))
    outs = surface_outputs(2, 256)
    foreach(i -> prog(i, i, outs...), 1:256)
    @test Decks.bits(reshape(outs[(1 - 1) * 2 + 2], 16, 16)) == Decks.bits(r.Ib[2])
    @test Decks.bits(reshape(outs[(4 - 1) * 2 + 1], 16, 16)) == Decks.bits(r.rs[1])
    @test_throws ArgumentError prog(1, 1, outs[1:13]...)
    # the polarized g-disk carries the g-disk's Stokes I bit for bit
    pd = render_analytic(p, PolarizedDisk(EquatorialDisk(a)); bands = (0, 1, 2), backend = :cpu)
    gd = render_analytic(p, EquatorialDisk(a); bands = (0, 1, 2), backend = :cpu)
    @test Decks.bits(pd.Ib) == Decks.bits(gd.Ib) && Decks.bits(pd.rs) == Decks.bits(gd.rs)
    @test Decks.bits(pd.ts) == Decks.bits(gd.ts) && Decks.bits(pd.phis) == Decks.bits(gd.phis)
    @test count(!iszero, pd.Q) > 50
    mix = render_analytic(p, (EquatorialDisk(a), mats[1]); bands = (0,), backend = :cpu)
    @test all(mix.I .>= gd.Ib[1] .- 1e-15)
    @test_throws ArgumentError render_analytic(p, cone(0.5, 1.0); backend = :cpu)
end
