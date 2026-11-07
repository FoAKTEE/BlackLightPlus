# The ghost layers of a multi-level block mesh: copies from blocks of the same level,
# restriction from finer and limited prolongation from coarser blocks across faces, edges
# and corners, extrapolation at the outer boundary. A linear field is reproduced exactly
# everywhere, a quadratic one to second order; the sample is continuous across faces
# between blocks of one level; positive data stay positive inside the mesh; the device
# form, the pool executor, the uniform resample, the frame window and the columns of a
# `.bin` dump on the same mesh sample the same field.
using Test, BlackLightPlus, StaticArrays, Random
using Adapt: Adapt
using BlackLightPlus.Data: AthenaKSnapshot, FluidBlocks, FrameWindow, LeafTable, block_sample,
                           corner, locate, resample, scale_density, MeshBox, mesh_value,
                           AthenaKBin, BinColumns, ghost_value, sample_var
using BlackLightPlus.Matter: primitives
const Decks = Main.Decks

# The leaves of a mesh of nroot³ root blocks: a block (L, lx, ly, lz) with refine(L, l)
# true and L < Lmax is replaced by its eight children.
function mesh_leaves(refine; nroot = 4, Lmax = 2)
    out = NTuple{4,Int}[]
    function visit(L, l)
        if L < Lmax && refine(L, l)
            for o in Iterators.product(0:1, 0:1, 0:1)
                visit(L + 1, 2 .* l .+ o)
            end
        else
            push!(out, (L, l...))
        end
    end
    for l in Iterators.product(0:(nroot - 1), 0:(nroot - 1), 0:(nroot - 1))
        visit(0, l)
    end
    return out
end

# Two levels: one refined root block, whose children meet level-0 blocks across faces,
# edges and corners and each other across faces.
const LEAVES2 = mesh_leaves((L, l) -> L == 0 && l == (1, 2, 1))
# Three levels, 2:1 balanced: the central 2³ roots refined, and an L-shaped set of three of
# their children refined again, so level-2 blocks meet level-1 blocks across concave and
# convex edges and corners.
const LEAVES3 = mesh_leaves((L, l) -> (L == 0 && all(in((1, 2)), l)) ||
                                      (L == 1 && l in ((3, 3, 3), (4, 3, 3), (3, 4, 3))))

# The snapshot of `leaves` on [-D, D]³ with n³ cells per block, every cell holding f at its
# centre (f returns the eight primitives).
function mesh_snap(leaves, n, f; D = 8.0, nroot = 4)
    w0 = 2D / nroot
    nmb = length(leaves)
    lo = zeros(3, nmb); hi = zeros(3, nmb); xl = zeros(3, nmb); dx = zeros(3, nmb)
    lv = zeros(Int, nmb); ll = zeros(Int, 3, nmb)
    data = zeros(Float32, 8, n, n, n, nmb)
    for (m, (L, l1, l2, l3)) in enumerate(leaves)
        w = w0 / 2^L
        for (a, l) in enumerate((l1, l2, l3))
            lo[a, m] = -D + l * w; hi[a, m] = lo[a, m] + w
            dx[a, m] = w / n; xl[a, m] = lo[a, m] + dx[a, m] / 2
        end
        lv[m] = L; ll[:, m] .= (l1, l2, l3)
        for k in 1:n, j in 1:n, i in 1:n
            x = SVector(xl[1, m] + (i - 1) * dx[1, m], xl[2, m] + (j - 1) * dx[2, m],
                        xl[3, m] + (k - 1) * dx[3, m])
            data[:, i, j, k, m] .= f(x)
        end
    end
    return AthenaKSnapshot(data, lo, hi, xl, dx, lv, ll)
end

vec8(p) = SVector(p.rho, p.uu..., p.eint, p.bb...)
sample(s, x) = vec8(primitives(s, SVector(0.0, x...)))

# A linear field with dyadic coefficients, exact in Float32 at every cell centre and ghost.
const LIN = [SVector(3.0, 0.5, -0.25, 0.125) .* q for q in (1.0, 2.0, -1.0, 0.5, 4.0, -2.0, 1.5, 0.25)]
linear(x) = SVector{8,Float64}(ntuple(q -> LIN[q][1] + LIN[q][2] * x[1] + LIN[q][3] * x[2] +
                                           LIN[q][4] * x[3], Val(8)))
quadratic(x) = (v = 1.0 + 0.03 * x[1]^2 - 0.02 * x[1] * x[2] + 0.01 * x[3]^2 + 0.015 * x[2] * x[3] +
                    0.2 * x[2];
                SVector{8,Float64}(ntuple(q -> q * v, Val(8))))

# Points of the mesh: uniform in the box, and on every block's faces, edges and corners.
function mesh_points(s, rng; nrand = 4000, perblock = 6)
    b = s.blocks
    D = maximum(b.hi)
    pts = [SVector{3}((2 .* rand(rng, 3) .- 1) .* (D * (1 - 1e-9))) for _ in 1:nrand]
    for m in 1:b.nmb, _ in 1:perblock
        lo = SVector(b.lo[1, m], b.lo[2, m], b.lo[3, m]); hi = SVector(b.hi[1, m], b.hi[2, m], b.hi[3, m])
        x = lo .+ rand(rng, 3) .* (hi .- lo)
        for nfix in 1:3                                   # face, edge, corner
            axes = randperm(rng, 3)[1:nfix]
            y = SVector(ntuple(a -> a in axes ? (rand(rng, Bool) ? lo[a] : hi[a]) : x[a], Val(3)))
            all(abs.(y) .< D) && push!(pts, y)
        end
    end
    return pts
end

maxerr(s, f, pts) = maximum(maximum(abs.(sample(s, x) .- f(x))) for x in pts)

@testset "ghost layers" begin

@testset "layout: the ghost layer around the interior, copies across same-level faces" begin
    s = mesh_snap(LEAVES3, 4, linear)
    b = s.blocks
    @test size(b.data) == (8, 6, 6, 6, length(LEAVES3)) && (b.ni, b.nj, b.nk) == (4, 4, 4)
    @test b.locate isa LeafTable && Set(s.levels) == Set(0:2)
    key = Dict((s.levels[m], s.logical_locations[:, m]...) => m for m in 1:b.nmb)
    ncopy = 0
    for m in 1:b.nmb
        L = s.levels[m]; l = s.logical_locations[:, m]
        r = get(key, (L, l[1] + 1, l[2], l[3]), 0)          # same-level neighbour along +x
        r == 0 && continue
        @test b.data[:, 6, 2:5, 2:5, m] == b.data[:, 2, 2:5, 2:5, r]
        @test b.data[:, 1, 2:5, 2:5, r] == b.data[:, 5, 2:5, 2:5, m]
        ncopy += 1
    end
    @test ncopy > 50
    # corner(b, i, j, k, m) reads interior cells 1:n and the ghost layer 0 and n + 1
    @test corner(b, 1, 1, 1, 7) == Float64.(b.data[:, 2, 2, 2, 7])
    @test corner(b, 0, 5, 2, 7) == Float64.(b.data[:, 1, 6, 3, 7])
end

@testset "a linear field is exact everywhere: faces, edges, corners, every level pair" begin
    rng = Xoshiro(20261107)
    for leaves in (LEAVES2, LEAVES3), n in (2, 4)
        s = mesh_snap(leaves, n, linear)
        b = s.blocks
        box = MeshBox(b)
        # every ghost cell holds the field at its centre
        bad = 0
        for m in 1:b.nmb, k in 0:(n + 1), j in 0:(n + 1), i in 0:(n + 1)
            x = SVector(b.x_left[1, m] + (i - 1) * b.dx[1, m], b.x_left[2, m] + (j - 1) * b.dx[2, m],
                        b.x_left[3, m] + (k - 1) * b.dx[3, m])
            bad += corner(b, i, j, k, m) != linear(x)
        end
        @test bad == 0
        @test maxerr(s, linear, mesh_points(s, rng)) <= 1e-12
        @test mesh_value(b, box, SVector(-7.5, 0.5, 1.5), SVector(1.0, 1.0, 1.0)) == linear(SVector(-7.5, 0.5, 1.5))
    end
end

@testset "a quadratic field converges at second order, the coarse-fine shells included" begin
    rng = Xoshiro(11)
    snaps = [mesh_snap(LEAVES3, n, quadratic) for n in (4, 8, 16)]
    pts = mesh_points(snaps[1], rng; nrand = 6000, perblock = 8)
    e = [maxerr(s, quadratic, pts) for s in snaps]
    @test all(r -> 3.5 <= r <= 5.0, e[1:2] ./ e[2:3])     # 4.6, 4.1: second order
    # points within half a finest cell of a level-2 block face, where the ghosts are prolonged
    # from or restricted to level 1
    b = snaps[1].blocks
    near = filter(pts) do x
        m = locate(b, x...)
        m != 0 && snaps[1].levels[m] == 2 &&
            any(a -> min(x[a] - b.lo[a, m], b.hi[a, m] - x[a]) <= b.dx[a, m] / 2, 1:3)
    end
    @test length(near) > 200
    en = [maxerr(s, quadratic, near) for s in snaps]
    @test all(r -> 3.5 <= r <= 5.0, en[1:2] ./ en[2:3])   # 4.0, 4.1
end

@testset "continuous across faces between blocks of one level, to rounding" begin
    rng = Xoshiro(5)
    for D in (8.0, 3π)                            # a dyadic and a non-representable geometry
    s = mesh_snap(LEAVES3, 4, x -> SVector{8,Float64}(rand(rng, 8)); D)
    b = s.blocks
    key = Dict((s.levels[m], s.logical_locations[:, m]...) => m for m in 1:b.nmb)
    worst = 0.0
    nface = 0
    for m in 1:b.nmb, a in 1:3
        l = collect(s.logical_locations[:, m]); l[a] += 1
        r = get(key, (s.levels[m], l...), 0)
        r == 0 && continue
        nface += 1
        lo = b.lo[:, m]; hi = b.hi[:, m]
        for t in 1:40
            u = t <= 8 ? Float64.(digits(t - 1; base = 2, pad = 3)) : rand(rng, 3)   # face corners first
            x = lo .+ u .* (hi .- lo); x[a] = hi[a]
            va = block_sample(b, m, x...); vb = block_sample(b, r, x...)
            worst = max(worst, maximum(abs.(va .- vb)))
        end
    end
    @test nface == 280
    @test worst <= 1e-14
    end
end

@testset "positive data: ghosts inside the range of the cells, samples positive" begin
    rng = Xoshiro(17)
    f(x) = SVector{8,Float64}(10.0^(-6 * rand(rng)), 0, 0, 0, 10.0^(-6 * rand(rng)), 0, 0, 0)
    s = mesh_snap(LEAVES3, 4, f)
    b = s.blocks
    inner = b.data[:, 2:5, 2:5, 2:5, :]
    lo, hi = extrema(inner[1, :, :, :, :])
    box = MeshBox(b)
    bad = 0
    for m in 1:b.nmb, k in 0:5, j in 0:5, i in 0:5
        x = SVector(b.x_left[1, m] + (i - 1) * b.dx[1, m], b.x_left[2, m] + (j - 1) * b.dx[2, m],
                    b.x_left[3, m] + (k - 1) * b.dx[3, m])
        all(box.lo .< x .< box.hi) || continue            # extrapolated at the outer boundary
        v = corner(b, i, j, k, m)
        bad += !(lo <= v[1] <= hi && v[5] > 0)
    end
    @test bad == 0
    pts = [SVector{3}((2 .* rand(rng, 3) .- 1) .* 7.5) for _ in 1:20_000]
    @test all(x -> (v = sample(s, x); v[1] > 0 && v[5] > 0), pts)
end

@testset "outer boundary: one-sided extrapolation, held on a mesh one cell thick" begin
    s = mesh_snap(LEAVES2, 4, linear)
    @test sample(s, SVector(7.99, -7.99, 3.3)) ≈ linear(SVector(7.99, -7.99, 3.3)) atol = 1e-12
    # a one-block, one-cell-thick slab in z: the field is held along z
    data = zeros(Float32, 8, 4, 4, 1, 1)
    for j in 1:4, i in 1:4
        data[:, i, j, 1, 1] .= linear(SVector(i - 0.5, j - 0.5, 0.5))
    end
    col(v...) = reshape(collect(Float64, v), 3, 1)
    slab = AthenaKSnapshot(data, col(0, 0, 0), col(4, 4, 1), col(0.5, 0.5, 0.5), col(1, 1, 1),
                           [0], zeros(Int, 3, 1))
    @test sample(slab, SVector(1.7, 2.2, 0.1)) ≈ linear(SVector(1.7, 2.2, 0.5)) atol = 1e-12
    @test sample(slab, SVector(3.9, 0.05, 0.9)) ≈ linear(SVector(3.9, 0.05, 0.5)) atol = 1e-12
end

@testset "every consumer samples the ghost-extended field" begin
    rng = Xoshiro(9)
    s = mesh_snap(LEAVES3, 4, linear)
    # the device form samples as the host
    d = Adapt.adapt(Decks.DeviceLike(), s)
    @test Decks.device_isbits(s)
    pts = mesh_points(s, rng; nrand = 2000, perblock = 2)
    @test all(x -> primitives(d, SVector(0.0, x...)) === primitives(s, SVector(0.0, x...)), pts)
    # the uniform resample holds the field at every node inside the mesh
    g = resample(s, 7.5, 31)
    node(i) = -7.5 + (i - 1) * 0.5
    @test all(Float64.(g.data[:, i, j, k]) ≈ linear(SVector(node(i), node(j), node(k)))
              for i in (1, 7, 16, 31), j in (2, 15, 30), k in (1, 16, 31))
    # a frame window of two density scalings blends the two extended fields
    s2 = scale_density(s, 3.0)
    @test s2.blocks.data[1, :, :, :, :] == s.blocks.data[1, :, :, :, :] .* 3.0f0
    w = FrameWindow(s.blocks, s2.blocks, 0.0, 10.0; t_obs = 10.0)
    x = SVector(-5.0, 3.2, 1.0)
    @test primitives(w, SVector(-5.0, x...)).rho ≈ 2.0 * linear(x)[1] rtol = 1e-12
end

@testset "the columns of a .bin dump on the same mesh: the same ghosts and samples" begin
    rng = Xoshiro(23)
    s = mesh_snap(LEAVES3, 4, x -> SVector{8,Float64}(rand(rng, 8)))
    b = s.blocks
    n = b.ni
    raw = permutedims(b.data[:, 2:(n + 1), 2:(n + 1), 2:(n + 1), :], (2, 3, 4, 1, 5))
    row(M, a) = M[a, :]
    bin = AthenaKBin(b.nmb, n, n, n, ["v$q" for q in 1:8], s.levels, row(b.lo, 1), row(b.hi, 1),
                     row(b.lo, 2), row(b.hi, 2), row(b.lo, 3), row(b.hi, 3), raw, 0.0)
    cols = BinColumns(bin, ntuple(identity, 8))
    box = MeshBox(cols)
    # every ghost evaluated on the dump is the stored ghost of the snapshot, to its rounding
    bad = 0
    for m in 1:b.nmb, k in 0:(n + 1), j in 0:(n + 1), i in 0:(n + 1)
        bad += Float32.(ghost_value(cols, box, m, SVector(i, j, k))) != Float32.(corner(b, i, j, k, m))
    end
    @test bad == 0
    pts = [SVector{3}((2 .* rand(rng, 3) .- 1) .* 7.999) for _ in 1:3000]
    @test maximum(abs(sample_var(bin, v, x...) - sample(s, x)[v]) for x in pts, v in 1:8) <= 1e-6
end

@testset "a multi-level torus on the pool executor" begin
    s = mesh_snap(LEAVES3, 4, x -> SVector{8,Float64}(Decks.torus_primitives(x...)); D = 12.0)
    p = prepare(RunConfig(; camera = CameraConfig(fov = 24.0, resolution = 12));
                spacetime = KerrSchildCartesian(0.9), matter = s, mode = IQUV())
    a = render(p; backend = :cpu)
    c = render(p; backend = Decks.POOL_BACKEND)
    planes(img) = (img.I, img.Q, img.U, img.V)
    @test count(>(0), a.I) > 20 && all(isfinite, a.I)
    if Decks.CROSS_HARDWARE
        @test all(maximum(abs.(x .- y)) <= 1e-11 * maximum(abs.(x)) for (x, y) in zip(planes(a), planes(c)))
    else
        @test Decks.bits(planes(a)) == Decks.bits(planes(c))
    end
end

end
