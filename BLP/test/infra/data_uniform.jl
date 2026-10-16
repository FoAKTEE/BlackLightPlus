# The uniform grid: node values, the box test and its `inside` flag, the unclamped
# fraction on the upper face, `resample` of a block provider, density scaling, the domain
# radius, and the device form.
using Test, BLP, StaticArrays, Random
using BLP.Data: UniformGrid, resample, scale_density, grid_sample, same_mesh
using BLP.Matter: PlasmaPrimitives, primitives, domain_radius
const Decks = Main.Decks

@testset "uniform grid" begin

@testset "nodes, box and flag" begin
    g = Decks.uniform_deck()
    @test (g.n, g.xo, g.box_half) == (24, -12.0, 12.0) && g.inv_dx == 23 / 24
    node(i) = g.xo + (i - 1) / g.inv_dx
    for (i, j, k) in ((1, 1, 1), (5, 17, 12), (23, 23, 23), (22, 2, 23))
        s = primitives(g, SVector(0.0, node(i), node(j), node(k)))
        v = Float64.(g.data[:, i, j, k])
        @test s.inside
        @test isapprox(s.rho, v[1]; atol = 1e-12) && isapprox(s.eint, v[5]; atol = 1e-12)
        @test isapprox(s.uu, SVector(v[2], v[3], v[4]); atol = 1e-12)
    end
    # the box faces belong to the box; anything beyond is outside with zero fields
    @test primitives(g, SVector(0.0, 12.0, 0.0, 0.0)).inside
    @test primitives(g, SVector(0.0, -12.0, -12.0, -12.0)).inside
    for p in ((12.01, 0.0, 0.0), (0.0, -12.5, 0.0), (0.0, 0.0, 100.0))
        s = primitives(g, SVector(0.0, p...))
        @test !s.inside && s === PlasmaPrimitives{Float64}()
    end
    p, inside = grid_sample(g, 1.0, 2.0, 3.0)
    s = primitives(g, SVector(7.0, 1.0, 2.0, 3.0))
    @test inside && p === SVector(s.rho, s.uu..., s.eint, s.bb...)
    @test domain_radius(g) == sqrt(3.0) * 12.0
    @test UniformGrid(g.data, 12.0) == g
    @test_throws ArgumentError UniformGrid(zeros(Float32, 8, 4, 4, 5), 1.0)
end

@testset "resample of a block provider" begin
    snap = Decks.amr_deck()
    for (bh, n) in ((12.0, 24), (14.0, 15))
        r = resample(snap, bh, n)
        @test (r.n, r.xo, r.inv_dx, r.box_half) == (n, -bh, (n - 1) / (2bh), bh)
        dx = 2bh / (n - 1)
        for (i, j, k) in ((1, 1, 1), (3, 7, 9), (n, n ÷ 2, 2))
            s = primitives(snap, SVector(0.0, -bh + (i - 1) * dx, -bh + (j - 1) * dx, -bh + (k - 1) * dx))
            @test r.data[:, i, j, k] == Float32[s.rho, s.uu..., s.eint, s.bb...]
        end
        @test UniformGrid(snap, bh, n).data == r.data
    end
    # nodes beyond the mesh hold zeros
    r = resample(snap, 14.0, 15)
    @test all(iszero, r.data[:, 1, :, :]) && all(iszero, r.data[:, :, :, 15])
    @test resample(snap.blocks, 12.0, 24).data == resample(snap, 12.0, 24).data
end

@testset "density scaling" begin
    g = Decks.uniform_deck()
    @test scale_density(g, 1.0) === g
    s = scale_density(g, 1.5)
    @test s.data[1, :, :, :] == g.data[1, :, :, :] .* 1.5f0
    @test s.data[2:8, :, :, :] == g.data[2:8, :, :, :] && s.data !== g.data
    @test same_mesh(s, g) && !same_mesh(g, resample(Decks.amr_deck(), 12.0, 16))
end

@testset "device form and hot path" begin
    g = Decks.uniform_deck()
    @test Decks.device_isbits(g)
    alloc(p, x) = (primitives(p, x); @allocated primitives(p, x))
    @test alloc(g, SVector(0.0, 1.0, 2.0, 3.0)) == 0
    @test alloc(g, SVector(0.0, 100.0, 2.0, 3.0)) == 0
end

end
