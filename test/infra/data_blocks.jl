# Fluid blocks: the clamped in-block trilinear sample, the three locate strategies and
# their face rules, the eager choice of strategy (leaf table, level hash past the cell
# cap, scan for a mesh with neither), mesh sharing, the domain radius, and the device form.
using Test, BlackLightPlus, StaticArrays, Random
using Adapt: Adapt
using BlackLightPlus.Data: FluidBlocks, LeafTable, LevelHash, BlockScan, AthenaKSnapshot, leaf_table,
                level_hash, build_locate, with_locate, locate, block_sample, same_mesh,
                share_mesh, PRIM_NAMES
using BlackLightPlus.Matter: PlasmaPrimitives, primitives, domain_radius
const Decks = Main.Decks

# Two level-0 blocks tiling [0,2]×[0,1]×[0,1] with densities 10 (x < 1) and 20 (x > 1);
# `swap` permutes the storage order only.
function two_block_snap(swap::Bool)
    n = 4
    ord = swap ? [2, 1] : [1, 2]
    lo = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)][ord]
    hi = [SVector(1.0, 1.0, 1.0), SVector(2.0, 1.0, 1.0)][ord]
    ll = [0 1; 0 0; 0 0][:, ord]
    data = zeros(Float32, 8, n, n, n, 2)
    for (slot, blk) in enumerate(ord)
        data[1, :, :, :, slot] .= 10.0f0 * blk
        data[5, :, :, :, slot] .= 1.0f0
    end
    return AthenaKSnapshot(data, lo, hi, [l .+ 0.125 for l in lo],
                           [SVector(0.25, 0.25, 0.25) for _ in 1:2], [0, 0], ll)
end

# Leaf-only nested mesh of the binary runs: domain ±4096, per level L a box ±4096/2^L of
# 4³ blocks whose central 2³ are refined to level L+1; block m holds density m.
function nested_snap(nlev; n = 2)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; lv = Int[]
    for L in 0:nlev
        w = 2048.0 / 2^L
        b0 = -4096.0 / 2^L
        for k in 0:3, j in 0:3, i in 0:3
            L < nlev && all(in((1, 2)), (i, j, k)) && continue
            l = SVector(b0 + i * w, b0 + j * w, b0 + k * w)
            push!(lo, l); push!(hi, l .+ w); push!(lv, L)
        end
    end
    nmb = length(lv)
    ll = Matrix{Int}(undef, 3, nmb)
    for m in 1:nmb, ax in 1:3
        ll[ax, m] = round(Int, (lo[m][ax] + 4096.0) / (hi[m][ax] - lo[m][ax]))
    end
    data = zeros(Float32, 8, n, n, n, nmb)
    for m in 1:nmb
        data[1, :, :, :, m] .= m
    end
    dx = [(hi[m] - lo[m]) ./ n for m in 1:nmb]
    return AthenaKSnapshot(data, lo, hi, [lo[m] + dx[m] / 2 for m in 1:nmb], dx, lv, ll)
end

isoutside(s::PlasmaPrimitives) = !s.inside && s.rho === 0.0 && s.eint === 0.0 &&
                                 s.uu === zero(SVector{3,Float64}) && s.bb === zero(SVector{3,Float64})

@testset "fluid blocks" begin

@testset "sample: cell values, clamped shell, exterior" begin
    snap = Decks.amr_deck()
    b = snap.blocks
    @test b.locate isa LeafTable                                  # built in the constructor
    @test (b.ni, b.nj, b.nk, b.nmb) == (6, 6, 6, 8) && size(b.data) == (8, 6, 6, 6, 8)
    # at a cell centre the sample is the cell's eight values
    for (m, i, j, k) in ((1, 3, 4, 5), (8, 1, 6, 2), (5, 6, 6, 6))
        c = SVector(b.x_left[1, m] + (i - 1) * b.dx[1, m], b.x_left[2, m] + (j - 1) * b.dx[2, m],
                    b.x_left[3, m] + (k - 1) * b.dx[3, m])
        s = primitives(snap, SVector(0.0, c...))
        v = Float64.(b.data[:, i, j, k, m])
        @test s.inside && s.rho === v[1] && s.uu === SVector(v[2], v[3], v[4]) &&
              s.eint === v[5] && s.bb === SVector(v[6], v[7], v[8])
    end
    # inside the outer half cell the stencil is one-sided: constant along the face normal
    m = 2; j = 3; k = 4
    y = b.x_left[2, m] + (j - 1) * 2.0; z = b.x_left[3, m] + (k - 1) * 2.0
    s1 = primitives(b, SVector(0.0, b.hi[1, m] - 0.2, y, z))
    s2 = primitives(b, SVector(0.0, b.hi[1, m] - 0.9, y, z))
    @test s1.rho === s2.rho === Float64(b.data[1, 6, j, k, m])
    # the time coordinate is ignored and a Float32 position samples as its Float64 value
    x = SVector(0.0, 4.3, -2.2, 1.7)
    @test primitives(b, x) === primitives(b, SVector(123.0, 4.3, -2.2, 1.7))
    xf = SVector{4,Float32}(0, 4.3, -2.2, 1.7)
    @test primitives(b, xf) === primitives(b, SVector{4,Float64}(xf))
    # off the mesh: the outside record
    for p in ((13.0, 0.0, 0.0), (0.0, -12.5, 0.0), (0.0, 0.0, 30.0), (12.0, 1.0, 1.0))
        @test isoutside(primitives(snap, SVector(0.0, p...)))
    end
    p = primitives(b, SVector(0.0, -6.1, -5.3, -4.2))
    @test block_sample(b, 1, -6.1, -5.3, -4.2) === SVector(p.rho, p.uu..., p.eint, p.bb...)
end

@testset "locate strategies agree off faces; face rules differ" begin
    snap = Decks.amr_deck()
    b = snap.blocks
    h = level_hash(b.lo, b.hi, snap.levels, snap.logical_locations)
    @test h isa LevelHash
    rng = Xoshiro(7)
    nbad = 0
    for _ in 1:20_000
        p = 14.0 .* (2 .* rand(rng, 3) .- 1)
        m = locate(BlockScan(), b, p...)
        nbad += locate(b.locate, b, p...) != m || locate(h, b, p...) != m ||
                (m == 0) != any(abs.(p) .> 12.0)
    end
    @test nbad == 0
    # the shared face x = 0 of blocks 1 (x < 0) and 2 (x > 0)
    @test locate(b, 0.0, -5.0, -5.0) == 2                      # leaf: high side
    @test locate(h, b, 0.0, -5.0, -5.0) == 2                   # hash: high side
    @test locate(BlockScan(), b, 0.0, -5.0, -5.0) == 1         # scan: lo < x ≤ hi
    # the outer faces: leaf and hash take [min, max), the scan (min, max]
    @test locate(b, -12.0, -5.0, -5.0) == 1 && locate(BlockScan(), b, -12.0, -5.0, -5.0) == 0
    @test locate(b, 12.0, -5.0, -5.0) == 0 && locate(BlockScan(), b, 12.0, -5.0, -5.0) == 2
    @test locate(snap, 1.0, 1.0, 1.0) == locate(b, 1.0, 1.0, 1.0) == 8
end

@testset "overlapping blocks: the later block wins" begin
    # block 2 is a finer block covering the upper half of block 1 (a non-leaf layout)
    lo = [SVector(0.0, 0.0, 0.0), SVector(0.5, 0.0, 0.0)]
    hi = [SVector(1.0, 1.0, 1.0), SVector(1.0, 0.5, 0.5)]
    t = leaf_table(reduce(hcat, lo), reduce(hcat, hi), [0, 1], [0 1; 0 0; 0 0])
    data = zeros(Float32, 8, 2, 2, 2, 2)
    b = FluidBlocks(data, reduce(hcat, lo), fill(0.25, 3, 2), reduce(hcat, lo), reduce(hcat, hi),
                    2, 2, 2, 2, t)
    @test locate(b, 0.75, 0.25, 0.25) == 2 && locate(BlockScan(), b, 0.75, 0.25, 0.25) == 2
    @test locate(b, 0.25, 0.25, 0.25) == 1 && locate(BlockScan(), b, 0.25, 0.25, 0.25) == 1
end

@testset "strategy choice: leaf table, level hash, scan" begin
    # an origin-aligned mesh builds its table (positive control)
    good = two_block_snap(false)
    t = good.blocks.locate
    @test t isa LeafTable && (t.N1, t.N2, t.N3) == (2, 1, 1) && vec(t.leaf) == Int32[1, 2]
    # a partial-domain dump (logical x locations 2 and 3) does not fit the finest grid
    lo = [0.0 1.0; 0.0 0.0; 0.0 0.0] .+ [2.0, 0.0, 0.0]; hi = lo .+ 1.0
    @test leaf_table(lo, hi, [0, 0], [2 3; 0 0; 0 0]) === nothing
    @test !(build_locate(lo, hi, [0, 0], [2 3; 0 0; 0 0]) isa LeafTable)
    # unequal root widths and no leaf table: the scan
    lo = [0.0 1.0; 0.0 0.0; 0.0 0.0]; hi = [1.0 3.0; 1.0 1.0; 1.0 1.0]
    @test level_hash(lo, hi, [0, 0], [0 3; 0 0; 0 0]) === nothing
    @test build_locate(lo, hi, [0, 0], [0 3; 0 0; 0 0]) isa BlockScan
    @test leaf_table(zeros(3, 0), zeros(3, 0), Int[], zeros(Int, 3, 0)) === nothing
end

@testset "deep nested mesh: level hash == scan" begin
    snap = nested_snap(9)                         # finest block width 4: a 2048³ finest grid
    b = snap.blocks
    @test b.locate isa LevelHash
    @test leaf_table(b.lo, b.hi, snap.levels, snap.logical_locations) === nothing
    rng = Xoshiro(20260924)
    agree = 0
    npt = 40_000
    for _ in 1:npt
        r = exp(rand(rng) * log(6000.0))          # log-uniform radius: every level is hit
        v = randn(rng, 3); v = v / sqrt(sum(abs2, v))
        x, y, z = r .* v
        agree += locate(b, x, y, z) == locate(BlockScan(), b, x, y, z)
    end
    @test agree == npt
    @test all(m -> locate(b, ((b.lo[:, m] .+ b.hi[:, m]) ./ 2)...) == m, 1:b.nmb)
    s = primitives(snap, SVector(0.0, 0.3, -0.2, 0.1))
    @test s.inside && s.rho == Float64(locate(b, 0.3, -0.2, 0.1))
    @test locate(b, 5000.0, 0.0, 0.0) == 0 && locate(b, 0.0, -4097.0, 0.0) == 0
    @test isoutside(primitives(snap, SVector(0.0, 0.0, 0.0, 9000.0)))
    # a shallow nesting fits a leaf table, which agrees with the hash off faces
    s3 = nested_snap(3; n = 4)
    @test s3.blocks.locate isa LeafTable
    h3 = level_hash(s3.blocks.lo, s3.blocks.hi, s3.levels, s3.logical_locations)
    @test all(1:5000) do _
        p = 4000.0 .* (2 .* rand(rng, 3) .- 1) .* rand(rng)
        locate(s3.blocks, p...) == locate(h3, s3.blocks, p...)
    end
end

@testset "domain radius" begin
    b = Decks.amr_deck().blocks
    t = b.locate
    hi_leaf = (t.x1min + t.N1 / t.inv_fdx1, t.x2min + t.N2 / t.inv_fdx2, t.x3min + t.N3 / t.inv_fdx3)
    @test domain_radius(b) == sqrt(max(t.x1min^2, hi_leaf[1]^2) + max(t.x2min^2, hi_leaf[2]^2) +
                                   max(t.x3min^2, hi_leaf[3]^2))
    @test domain_radius(b) ≈ sqrt(3 * 12.0^2)
    @test domain_radius(with_locate(b, BlockScan())) == sqrt(3 * 12.0^2)
    h = level_hash(b.lo, b.hi, zeros(Int, 8), Decks.amr_deck().logical_locations)
    @test domain_radius(with_locate(b, h)) == sqrt(3 * 12.0^2)
    nested = nested_snap(9)
    @test domain_radius(nested) == sqrt(3 * 4096.0^2)
    # off-centre domain: the farthest corner
    snap = two_block_snap(false)
    @test domain_radius(snap) == sqrt(2.0^2 + 1.0 + 1.0)
end

@testset "mesh sharing" begin
    snapA = two_block_snap(false)
    snapB = two_block_snap(true)                  # permuted storage order, same bounding box
    @test !same_mesh(snapA, snapB)
    @test share_mesh(snapB, snapA) === snapB
    for s in (snapA, snapB), (x, y, z) in ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5))
        @test locate(s, x, y, z) == locate(BlockScan(), s.blocks, x, y, z)
    end
    @test primitives(snapB, SVector(0.0, 1.5, 0.5, 0.5)).rho == 20.0
    snapA2 = two_block_snap(false)
    @test same_mesh(snapA, snapA2) && same_mesh(snapA.blocks, snapA2.blocks)
    shared = share_mesh(snapA2, snapA)
    @test shared.blocks.locate === snapA.blocks.locate && shared.blocks.lo === snapA.blocks.lo
    @test shared.blocks.x_left === snapA.blocks.x_left && shared.blocks.data === snapA2.blocks.data
    @test primitives(shared, SVector(0.0, 1.5, 0.5, 0.5)) === primitives(snapA2, SVector(0.0, 1.5, 0.5, 0.5))
end

@testset "array constructor checks" begin
    snap = Decks.amr_deck()
    b = snap.blocks
    @test snap.names == collect(PRIM_NAMES) && snap.time == 0.0
    @test_throws ArgumentError AthenaKSnapshot(zeros(Float32, 7, 2, 2, 2, 1), zeros(3, 1),
                                               ones(3, 1), zeros(3, 1), ones(3, 1), [0], zeros(Int, 3, 1))
    @test_throws ArgumentError AthenaKSnapshot(zeros(Float32, 8, 2, 2, 2, 2), zeros(3, 1),
                                               ones(3, 1), zeros(3, 1), ones(3, 1), [0], zeros(Int, 3, 1))
    w = with_locate(b, BlockScan())
    @test w.data === b.data && w.lo === b.lo && w.locate === BlockScan()
end

@testset "device form and hot path" begin
    snap = Decks.amr_deck()
    b = snap.blocks
    hb = with_locate(b, level_hash(b.lo, b.hi, snap.levels, snap.logical_locations))
    @test Adapt.adapt(Decks.DeviceLike(), hb).locate === BlockScan()
    @test Adapt.adapt(Decks.DeviceLike(), snap) isa FluidBlocks
    for p in (b, with_locate(b, BlockScan()), hb, snap)
        @test Decks.device_isbits(p)
    end
    @test !isbits(hb) && !isbits(snap)
    alloc(p, x) = (primitives(p, x); @allocated primitives(p, x))
    for p in (b, with_locate(b, BlockScan()), hb, snap, nested_snap(9)),
        x in (SVector(0.0, 3.0, -4.0, 5.0), SVector(0.0, 300.0, 0.0, 0.0))
        @test alloc(p, x) == 0
    end
    # the adapted provider samples as the host one
    d = Adapt.adapt(Decks.DeviceLike(), hb)
    rng = Xoshiro(3)
    @test all(1:2000) do _
        x = SVector(0.0, (13.0 .* (2 .* rand(rng, 3) .- 1))...)
        primitives(d, x) === primitives(with_locate(b, BlockScan()), x)
    end
end

end
