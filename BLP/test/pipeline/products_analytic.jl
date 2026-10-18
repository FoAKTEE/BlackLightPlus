# The exact-Kerr disk products: the per-band maps and their sum, landings outside the
# horizon, executor independence, the lensing-band layers, and the argument checks.
using Test, BLP, StaticArrays
using BLP.Plans: CameraConfig, RunConfig, prepare
using BLP.Products: render_analytic, photon_ring_layers, AnalyticProgram
import BLP.Products: analytic_outputs
using BLP.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, kerr_horizon
using BLP.Matter: EquatorialDisk
using BLP.Execution: PoolExecutor
const Decks = Main.Decks

plan(; a = 0.9, res = 12, inc = 60.0, st = KerrSchildCartesian(a)) =
    prepare(RunConfig(camera = CameraConfig(inclination_deg = inc, fov = 24.0, resolution = res));
            spacetime = st, matter = nothing)
const FIELDS = (:I, :Ib, :rs, :ts, :phis, :redshift_sign)

@testset "exact-Kerr disk" begin
    @testset "render_analytic" begin
        p = plan()
        disk = EquatorialDisk(0.9)
        r = render_analytic(p, disk; backend = :cpu)
        @test r.bands == [0, 1, 2] && length(r.Ib) == 3 && r.plan === p
        @test all(size(m) == (12, 12) for f in FIELDS[2:end] for m in getfield(r, f))
        @test r.I == r.Ib[1] + r.Ib[2] + r.Ib[3]
        for b in 1:3
            off = isnan.(r.rs[b])
            @test all(iszero, r.Ib[b][off]) && all(isnan, r.ts[b][off]) && all(isnan, r.phis[b][off])
            @test all(>(kerr_horizon(0.9)), r.rs[b][.!off])
            @test all(in((-1.0, 1.0)), r.redshift_sign[b][.!off])
        end
        @test count(>(0), r.Ib[1]) > 30 && count(>(0), r.Ib[2]) > 0
        @test count(!isnan, r.rs[3]) <= count(!isnan, r.rs[2]) <= count(!isnan, r.rs[1])
        # the program writes quantity q of band b to output (q - 1) K + b, which the maps share
        prog = AnalyticProgram(p.pixels, p.theta_o, 0.9, p.camera.d, disk, (0, 1, 2))
        outs = analytic_outputs(3, 144)
        foreach(k -> prog(k, k, outs...), 1:144)
        @test all(Decks.bits(reshape(outs[(q - 1) * 3 + b], 12, 12)) == Decks.bits(getfield(r, f)[b])
                  for (q, f) in enumerate(FIELDS[2:end]) for b in 1:3)
        @test all(m isa Matrix{Float64} for f in FIELDS[2:end] for m in getfield(r, f))
        @test_throws ArgumentError prog(1, 1, outs[1:14]...)
        for exec in (:host_pool, PoolExecutor(Decks.hosts(3); lanes = 2, chunk = 7))
            q = render_analytic(p, disk; backend = exec)
            @test all(Decks.bits(getfield(q, f)) == Decks.bits(getfield(r, f)) for f in FIELDS)
        end
        one = render_analytic(p, disk; bands = (1,), backend = :cpu)
        @test Decks.bits(one.Ib[1]) == Decks.bits(r.Ib[2]) && one.bands == [1]
        far = render_analytic(p, disk; ro = 2000.0, backend = :cpu)
        @test isapprox(sum(far.I), sum(r.I); rtol = 1e-2)
        @test_throws ArgumentError render_analytic(p, EquatorialDisk(0.5); backend = :cpu)
        bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        @test_throws ArgumentError render_analytic(plan(st = bin), disk; backend = :cpu)
    end

    @testset "photon_ring_layers" begin
        p = plan()
        disk = EquatorialDisk(0.9)
        l = photon_ring_layers(p, disk; dx = 0.25, backend = :cpu)
        @test size(l.I) == (l.N, l.N) && size(l.grid) == (l.N^2, 2) && l.N == 96
        @test l.bands == [0, 1, 2] && l.dx == 0.25 && l.box == 12.0
        @test all(l.masks[b] == (l.Ib[b] .> 0) for b in 1:3)
        @test count(l.masks[1]) > count(l.masks[2]) > 0 && count(l.masks[3]) <= count(l.masks[2])
        @test l.I == l.Ib[1] + l.Ib[2] + l.Ib[3]
        f = photon_ring_layers(p, disk; dx = 0.25, fudge = 2.0, backend = :host_pool)
        @test Decks.bits(f.Ib) == Decks.bits(l.Ib)
        @test f.I == l.Ib[1] + 2.0 .* l.Ib[2] + 2.0 .* l.Ib[3]
    end
end
