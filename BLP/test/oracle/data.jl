# Oracle: the Data module against BlackLightPlus.Simulation and the old device samplers,
# bit for bit. Fluid blocks (leaf table, level hash, block scan) against
# `sample_primitives` and the old locates; the uniform grid against `_gimg_sample`; the
# two-frame window and the series provider against `_gimg_sample_t`, `_gamr_sample_t`
# and `sample_primitives_t`; `resample` against `resample_fluid_uniform`; `.athdf` and
# `.bin` files read by both packages; the Z4c resample. Points include the mesh exterior,
# block faces, edges and corners and their neighbouring floats.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
# HDF5, and Adapt for the new decks of `test/decks.jl`, load from the BLP environment.
const BLP_DIR = normpath(joinpath(@__DIR__, "..", ".."))
BLP_DIR in LOAD_PATH || push!(LOAD_PATH, BLP_DIR)
using HDF5: h5open, write_attribute
using BLP.Data
using BLP.Matter: PlasmaPrimitives, primitives, domain_radius

const Decks = Module(:Decks)
Base.include(Decks, joinpath(BLP_DIR, "test", "decks.jl"))
const DK = Decks.Decks

const OS = BlackLightPlus.Simulation

# The old device samplers, evaluated from their source files.
const GD = scratch_module(:OldData)
Core.eval(GD, :(using BlackLightPlus.Simulation: AthenaKSnapshot, sample_primitives, leaf_table))
load_old!(GD, "gpu_image.jl", (:UniformFluidGrid, :resample_fluid_uniform, :_gimg_corner,
                               :_gimg_sample, :_gimg_sample_t, :scale_grid_density,
                               :_win_inv_dt, :_win_t_obs))
load_old!(GD, "gpu_amr.jl", (:_build_leaf_table, :_gamr_locate_scan, :_gamr_locate_fast,
                             :_gamr_corner, :_gamr_lerp, :_gamr_sample, :_gamr_sample_t,
                             :_gamr_domain_radius))
load_old!(GD, "gpu_spacetime.jl", (:_gamr_sample_scan,))

@noinline callf(f::F, args...) where {F} = f(args...)

# Bitwise equality of a new record and an old NamedTuple or SVector{8} sample.
same_prim(n::PlasmaPrimitives, o::NamedTuple) =
    n.rho === o.rho && n.eint === o.eint && n.uu === o.uu && n.bb === o.bb && n.inside === o.inside
same_prim(n::PlasmaPrimitives, o::SVector{8,Float64}) =
    n.rho === o[1] && n.uu === SVector(o[2], o[3], o[4]) && n.eint === o[5] &&
    n.bb === SVector(o[6], o[7], o[8])

# ------------------------------------------------------------------ decks

# The old AMR deck: `di_amr` of `test/device_decks.jl`.
function old_amr(; nc = 6, half = 12.0, scale = 1.0f0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, nc, nc, nc, nmb, 8)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; xl = SVector{3,Float64}[]
    dxs = SVector{3,Float64}[]; lloc = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        push!(lo, l); push!(hi, l .+ w); push!(xl, l .+ dx / 2); push!(dxs, SVector(dx, dx, dx))
        lloc[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            p = DK.torus_primitives(xl[m][1] + (i - 1) * dx, xl[m][2] + (j - 1) * dx,
                                    xl[m][3] + (k - 1) * dx)
            for q in 1:8
                data[i, j, k, m, q] = q == 1 ? scale * p[q] : p[q]
            end
        end
    end
    return OS.AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                              collect(String, BLP.Data.PRIM_NAMES), nc, nc, nc, nmb)
end

# The old uniform deck: `di_uniform` of `test/device_decks.jl`.
function old_uniform(; n = 24, bh = 12.0)
    host = zeros(Float32, 8, n, n, n)
    dx = 2bh / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        host[:, i, j, k] .= DK.torus_primitives(-bh + (i - 1) * dx, -bh + (j - 1) * dx,
                                                -bh + (k - 1) * dx)
    end
    return GD.UniformFluidGrid(host, n, bh, -bh, (n - 1) / (2bh))
end

# One mesh in both packages from per-block geometry and random primitives.
function twin_snaps(rng, lo, hi, levels, ll, n)
    nmb = length(lo)
    data = rand(rng, Float32, n, n, n, nmb, 8) .- 0.25f0
    dx = [(hi[m] - lo[m]) ./ n for m in 1:nmb]
    xl = [lo[m] + dx[m] / 2 for m in 1:nmb]
    old = OS.AthenaKSnapshot(data, lo, hi, xl, dx, levels, ll,
                             collect(String, BLP.Data.PRIM_NAMES), n, n, n, nmb)
    new = AthenaKSnapshot(permutedims(data, (5, 1, 2, 3, 4)), lo, hi, xl, dx, levels, ll)
    return new, old
end

# Nested boxes of the binary runs: per level L a box ±4096/2^L of 4³ blocks whose central
# 2³ are refined to level L+1. Nine levels exceed the dense leaf table.
function nested_mesh(nlev)
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
    ll = Matrix{Int}(undef, 3, length(lv))
    for m in eachindex(lv), ax in 1:3
        ll[ax, m] = round(Int, (lo[m][ax] + 4096.0) / (hi[m][ax] - lo[m][ax]))
    end
    return lo, hi, lv, ll
end

# Two level-0 blocks of unequal widths, the second off the finest grid: neither a leaf
# table nor a level hash exists, so both packages scan.
scan_mesh() = ([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)],
               [SVector(1.0, 1.0, 1.0), SVector(3.0, 1.0, 1.0)], [0, 0], [0 3; 0 0; 0 0])

# ------------------------------------------------------------------ points

# `n` points in the box `±(half + margin)`, a quarter free, the rest with some
# coordinates on the given faces, on cell centres, or one float off a face.
function deck_points(rng, n; half, faces, centres, margin = 2.0)
    pts = Vector{SVector{3,Float64}}(undef, n)
    snap_face(f) = rand(rng, (f, prevfloat(f), nextfloat(f)))
    for k in 1:n
        p = MVector{3,Float64}((half + margin) .* (2 .* rand(rng, 3) .- 1))
        mode = k % 4
        for a in 1:3
            r = rand(rng)
            if mode == 1 && r < 0.6
                p[a] = rand(rng, faces)
            elseif mode == 2 && r < 0.6
                p[a] = r < 0.3 ? rand(rng, centres) : rand(rng, faces)
            elseif mode == 3 && r < 0.6
                p[a] = snap_face(rand(rng, faces))
            end
        end
        pts[k] = SVector(p)
    end
    return pts
end

# Log-uniform radii through every level of the nested mesh, plus face points.
function nested_points(rng, n, lo)
    pts = Vector{SVector{3,Float64}}(undef, n)
    for k in 1:n
        if isodd(k)
            r = exp(rand(rng) * log(6000.0))
            pts[k] = r * normalize(randn(rng, SVector{3,Float64}))
        else
            c = lo[rand(rng, 1:length(lo))]
            jit = SVector(ntuple(_ -> rand(rng, (0.0, 1e-9, -1e-9, 3.7)), 3))
            pts[k] = c + jit
        end
    end
    return pts
end

const NPTS = 20_000
const rng = MersenneTwister(0xDA7A)

const amr_faces = [-12.0, -0.0, 0.0, 12.0, -10.0, -2.0, 2.0, 4.0, 10.0]
const amr_centres = collect(-11.0:2.0:11.0)
const AMR_PTS = deck_points(rng, NPTS; half = 12.0, faces = amr_faces, centres = amr_centres)

@testset "Data oracle" begin

@testset "fluid blocks: leaf-table deck" begin
    new = DK.amr_deck(); old = old_amr()
    b = new.blocks
    @test b.locate isa LeafTable
    @test b.data == permutedims(old.data, (5, 1, 2, 3, 4))
    t = OS.leaf_table(old)
    h_old = OS._build_level_hash(old.ext_lo, old.ext_hi, old.levels, old.logical_locations, old.nmb)
    h_new = level_hash(b.lo, b.hi, new.levels, new.logical_locations)
    @test b.locate.leaf == t.leaf && (b.locate.N1, b.locate.N2, b.locate.N3) == (t.N1, t.N2, t.N3)
    @test b.locate.inv_fdx1 === t.inv_fdx1 && b.locate.x1min === t.x1min
    blk = permutedims(old.data, (5, 1, 2, 3, 4))
    xlft = reduce(hcat, old.x_left); dxb = reduce(hcat, old.dx)
    lo = reduce(hcat, old.ext_lo); hi = reduce(hcat, old.ext_hi)
    leaf, amrp = GD._build_leaf_table(old)
    scan = with_locate(b, BlockScan()); hashed = with_locate(b, h_new)
    nbad = zeros(Int, 9)
    for p in AMR_PTS
        x, y, z = p
        X = SVector(0.0, x, y, z)
        nbad[1] += !same_prim(callf(primitives, new, X), callf(OS.sample_primitives, old, X))
        nbad[2] += callf(locate, b, x, y, z) != callf(OS.meshblock_index, old, x, y, z)
        nbad[3] += callf(locate, b.locate, b, x, y, z) != callf(OS._leaf_locate, t, x, y, z)
        nbad[4] += callf(locate, b.locate, b, x, y, z) != Int(callf(GD._gamr_locate_fast, leaf, amrp, x, y, z))
        nbad[5] += callf(locate, h_new, b, x, y, z) != callf(OS._hash_locate, h_old, x, y, z)
        nbad[6] += callf(locate, BlockScan(), b, x, y, z) != callf(OS._find_meshblock_scan, old, x, y, z)
        nbad[7] += callf(locate, BlockScan(), b, x, y, z) != callf(GD._gamr_locate_scan, lo, hi, old.nmb, x, y, z)
        nbad[8] += !same_prim(callf(primitives, scan, X),
                              callf(GD._gamr_sample_scan, blk, xlft, dxb, lo, hi, old.ni, old.nj,
                                    old.nk, old.nmb, x, y, z))
        nbad[9] += !same_prim(callf(primitives, b, X), callf(GD._gamr_sample, blk, xlft, dxb, leaf, amrp, x, y, z))
        hashed_ok = same_prim(callf(primitives, hashed, X),
                              callf(OS.sample_primitives, old, X)) ||
                    callf(locate, h_new, b, x, y, z) != callf(OS.meshblock_index, old, x, y, z)
        nbad[1] += !hashed_ok
    end
    @test nbad == zeros(Int, 9)
    @test domain_radius(new) === GD._gamr_domain_radius(amrp)
    @test count(p -> locate(b, p...) == 0, AMR_PTS) > NPTS ÷ 10        # exterior covered
    @test count(p -> locate(b, p...) != locate(BlockScan(), b, p...), AMR_PTS) > 100   # faces hit
end

@testset "fluid blocks: nested deck, leaf table and level hash" begin
    for (nlev, n, strategy) in ((3, 4, LeafTable), (9, 2, LevelHash))
        lo, hi, lv, ll = nested_mesh(nlev)
        new, old = twin_snaps(rng, lo, hi, lv, ll, n)
        b = new.blocks
        @test b.locate isa strategy
        @test (OS.leaf_table(old) === nothing) == (strategy === LevelHash)
        h_old = OS._build_level_hash(old.ext_lo, old.ext_hi, old.levels, old.logical_locations, old.nmb)
        h_new = level_hash(b.lo, b.hi, new.levels, new.logical_locations)
        pts = nested_points(rng, NPTS, lo)
        nbad = zeros(Int, 4)
        for p in pts
            x, y, z = p
            X = SVector(0.0, x, y, z)
            nbad[1] += !same_prim(callf(primitives, new, X), callf(OS.sample_primitives, old, X))
            nbad[2] += callf(locate, b, x, y, z) != callf(OS.meshblock_index, old, x, y, z)
            nbad[3] += callf(locate, h_new, b, x, y, z) != callf(OS._hash_locate, h_old, x, y, z)
            nbad[4] += callf(locate, BlockScan(), b, x, y, z) != callf(OS._find_meshblock_scan, old, x, y, z)
        end
        @test nbad == zeros(Int, 4)
    end
end

@testset "fluid blocks: scan deck" begin
    new, old = twin_snaps(rng, scan_mesh()..., 4)
    @test new.blocks.locate isa BlockScan
    @test OS.leaf_table(old) === nothing && OS._leaf_hash(old) === nothing
    pts = deck_points(rng, NPTS; half = 1.5, faces = [0.0, 1.0, 2.0, 3.0], centres = [0.125, 1.25],
                      margin = 2.0)
    pts = [p + SVector(1.5, 0.0, 0.0) for p in pts]
    nbad = 0
    for p in pts
        X = SVector(0.0, p...)
        nbad += !same_prim(callf(primitives, new, X), callf(OS.sample_primitives, old, X))
    end
    @test nbad == 0
end

@testset "uniform grid and resample" begin
    g = DK.uniform_deck(); og = old_uniform()
    @test g.data == og.host && (g.n, g.xo, g.inv_dx, g.box_half) === (og.n, og.xo, og.inv_dx, og.box_half)
    nodes = collect(range(-12.0, 12.0; length = 24))
    pts = deck_points(rng, NPTS; half = 12.0, faces = [-12.0, 12.0, 0.0, nodes[5], nodes[17]],
                      centres = nodes[2:end-1] .+ 0.5 * step(range(-12.0, 12.0; length = 24)))
    nbad = 0
    for p in pts
        X = SVector(0.0, p...)
        n = callf(primitives, g, X)
        o = callf(GD._gimg_sample, og.host, og.n, og.xo, og.inv_dx, p...)
        nbad += !same_prim(n, o)
        nbad += n.inside != !(any(f -> f < 0.0 || f > g.n - 1, (p .- g.xo) .* g.inv_dx))
    end
    @test nbad == 0
    @test domain_radius(g) === sqrt(3.0) * abs(og.xo)
    new = DK.amr_deck(); old = old_amr()
    for (bh, n) in ((12.0, 24), (14.0, 33), (9.5, 16))
        r = resample(new, bh, n); o = GD.resample_fluid_uniform(old, bh, n)
        @test reinterpret(UInt32, vec(r.data)) == reinterpret(UInt32, vec(o.host))
        @test (r.n, r.xo, r.inv_dx, r.box_half) === (o.n, o.xo, o.inv_dx, o.box_half)
    end
    s = scale_density(g, 1.5); os_ = GD.scale_grid_density(og, 1.5)
    @test reinterpret(UInt32, vec(s.data)) == reinterpret(UInt32, vec(os_.host))
end

@testset "two-frame window and series provider" begin
    ts = [12000.0, 13200.0]
    X0s = vcat([0.0, -1.0, -100.0, -600.0, -1199.9, -1200.0, -1200.5, -1500.0, 50.0],
               -1300.0 .+ 1400.0 .* rand(rng, 40))
    pts = AMR_PTS[1:1000]
    # uniform frames
    gA = DK.uniform_deck(); gB = scale_density(gA, 1.5)
    ogA = old_uniform(); ogB = GD.scale_grid_density(ogA, 1.5)
    winU = FrameWindow(gA, gB, ts[1], ts[2])
    @test winU.t_obs === GD._win_t_obs((t_a = ts[1], inv_dt = GD._win_inv_dt(ts[1], ts[2])))
    # block frames
    sA = DK.amr_deck(); sB = DK.amr_deck(scale = 1.5f0)
    oA = old_amr(); oB = old_amr(scale = 1.5f0)
    serN = synthetic_series(sA; times = ts, evolve = (s, t) -> t == ts[1] ? s : sB)
    serO = OS.synthetic_series(oA; times = ts, evolve = (s, t) -> t == ts[1] ? s : oB)
    winB = window(serN, 1)
    blkA = permutedims(oA.data, (5, 1, 2, 3, 4)); blkB = permutedims(oB.data, (5, 1, 2, 3, 4))
    xlft = reduce(hcat, oA.x_left); dxb = reduce(hcat, oA.dx)
    leaf, amrp = GD._build_leaf_table(oA)
    sp = SeriesProvider(serN)
    nbad = zeros(Int, 3); nwin_series = 0; ntot = 0
    for X0 in X0s, p in pts
        x, y, z = p
        X = SVector(X0, x, y, z)
        t0 = winU.t_obs + 0.0 - winU.t_a
        w = clamp((t0 + X0) * winU.inv_dt, 0.0, 1.0)
        nbad[1] += !same_prim(callf(primitives, winU, X),
                              callf(GD._gimg_sample_t, ogA.host, ogB.host, ogA.n, ogA.xo, ogA.inv_dx, w, x, y, z))
        nbad[2] += !same_prim(callf(primitives, winB, X),
                              callf(GD._gamr_sample_t, blkA, blkB, xlft, dxb, leaf, amrp, w, x, y, z))
        o = callf(OS.sample_primitives_t, serO, X)
        nbad[3] += !same_prim(callf(primitives, sp, X), o)
        nwin_series += same_prim(callf(primitives, winB, X), o)
        ntot += 1
    end
    @test nbad == zeros(Int, 3)
    @info "FrameWindow vs sample_primitives_t: $nwin_series of $ntot samples bitwise equal"
    # three frames, and an identity series (the same object at every time)
    ts3 = [0.0, 10.0, 25.0]
    fac(t) = 1.0 + t / 40
    ser3N = synthetic_series(sA; times = ts3, evolve = (s, t) -> scale_density(s, fac(t)), t_obs = 20.0)
    ser3O = OS.synthetic_series(oA; times = ts3, evolve = (s, t) -> OS.scale_density(s, fac(t)), t_obs = 20.0)
    serIN = synthetic_series(sA; times = ts3); serIO = OS.synthetic_series(oA; times = ts3)
    nbad = 0
    for X0 in (-30.0, -20.0, -15.0, -12.5, -10.0, -3.3, 0.0, 4.9, 5.0, 9.0), p in pts
        X = SVector(X0, p...)
        nbad += !same_prim(callf(primitives, SeriesProvider(ser3N), X), callf(OS.sample_primitives_t, ser3O, X))
        nbad += !same_prim(callf(primitives, SeriesProvider(serIN), X), callf(OS.sample_primitives_t, serIO, X))
    end
    @test nbad == 0
    # fast-light limit: an infinite upper time samples frame a alone
    win0 = FrameWindow(gA, gB, ts[1], Inf)
    @test win0.inv_dt == 0.0
    @test all(same_prim(primitives(win0, SVector(-500.0, p...)), GD._gimg_sample(ogA.host, ogA.n, ogA.xo, ogA.inv_dx, p...)) for p in pts)
    @test domain_radius(winB) === GD._gamr_domain_radius(amrp)
    @test domain_radius(winU) === sqrt(3.0) * abs(ogA.xo)
end

# ------------------------------------------------------------------ files

# An `.athdf` in AthenaK's on-disk layout from per-block arrays (`(ni, nj, nk, nmb, 8)`).
function write_athdf(path, data, lo, hi, levels, ll; uovnames = BLP.Data.PRIM_NAMES[1:5],
                     extra = nothing)
    ni, nj, nk, nmb, _ = size(data)
    uov = extra === nothing ? data[:, :, :, :, 1:5] : extra
    B = data[:, :, :, :, 6:8]
    xf(a, n) = reduce(hcat, [collect(range(lo[m][a], hi[m][a]; length = n + 1)) for m in 1:nmb])
    xv(f) = (f[1:end-1, :] .+ f[2:end, :]) ./ 2
    x1f = xf(1, ni); x2f = xf(2, nj); x3f = xf(3, nk)
    h5open(path, "w") do h
        h["uov"] = uov; h["B"] = B
        h["x1f"] = x1f; h["x2f"] = x2f; h["x3f"] = x3f
        h["x1v"] = xv(x1f); h["x2v"] = xv(x2f); h["x3v"] = xv(x3f)
        h["Levels"] = Int32.(levels)
        h["LogicalLocations"] = Int64.(ll)
        write_attribute(h, "VariableNames", [uovnames...; "bcc1"; "bcc2"; "bcc3"])
        write_attribute(h, "Time", 2.5)
    end
    return path
end

same_snapshot(n::AthenaKSnapshot, o) =
    n.blocks.data == permutedims(o.data, (5, 1, 2, 3, 4)) &&
    n.blocks.lo == reduce(hcat, o.ext_lo) && n.blocks.hi == reduce(hcat, o.ext_hi) &&
    n.blocks.x_left == reduce(hcat, o.x_left) && n.blocks.dx == reduce(hcat, o.dx) &&
    n.levels == o.levels && n.logical_locations == o.logical_locations &&
    (n.blocks.ni, n.blocks.nj, n.blocks.nk, n.blocks.nmb) == (o.ni, o.nj, o.nk, o.nmb)

function count_bad(new, old, pts)
    nbad = 0
    for p in pts
        X = SVector(0.0, p...)
        nbad += !same_prim(callf(primitives, new, X), callf(OS.sample_primitives, old, X))
    end
    return nbad
end

@testset "athdf files" begin
    mktempdir() do dir
        o = old_amr()
        lo = o.ext_lo; hi = o.ext_hi
        path = write_athdf(joinpath(dir, "eint.athdf"), o.data, lo, hi, o.levels, o.logical_locations)
        n = AthenaKSnapshot(path); od = OS.AthenaKSnapshot(path)
        @test same_snapshot(n, od) && n.time == 2.5
        @test count_bad(n, od, AMR_PTS) == 0
        # the press layout: uov = dens velx vely velz press temperature, one poisoned cell
        uov = cat(o.data[:, :, :, :, 1:4], 3.0f0 .* o.data[:, :, :, :, 5:5],
                  rand(rng, Float32, size(o.data)[1:4]..., 1); dims = 5)
        uov[2, 3, 1, 4, 5] = NaN32
        pp = write_athdf(joinpath(dir, "press.athdf"), o.data, lo, hi, o.levels, o.logical_locations;
                         uovnames = ("dens", "velx", "vely", "velz", "press", "temperature"),
                         extra = uov)
        γ = 13 / 9
        n = AthenaKSnapshot(pp; gamma = γ); od = OS.AthenaKSnapshot(pp; gamma = γ)
        @test same_snapshot(n, od)
        @test count_bad(n, od, AMR_PTS) == 0
        # the nested mesh past the leaf-table cap
        lo9, hi9, lv9, ll9 = nested_mesh(9)
        _, old9 = twin_snaps(rng, lo9, hi9, lv9, ll9, 2)
        p9 = write_athdf(joinpath(dir, "nested.athdf"), old9.data, lo9, hi9, lv9, ll9)
        n = AthenaKSnapshot(p9); od = OS.AthenaKSnapshot(p9)
        @test n.blocks.locate isa LevelHash
        @test same_snapshot(n, od)
        @test count_bad(n, od, nested_points(rng, NPTS, lo9)) == 0
    end
end

# An "Athena binary output version=1.1" dump of `(nx, nx, nx, nvar, nmb)` data.
function write_bin(path, vars, data, x1, x2, x3, levels; time = 1.5)
    nx = size(data, 1); nmb = size(data, 5)
    params = "<par_end>\n"
    open(path, "w") do io
        print(io, "Athena binary output version=1.1\n  size of preheader=5\n  time=$time\n",
              "  cycle=1\n  size of location=8\n  size of variable=4\n",
              "  number of variables=", length(vars), "\n",
              "  variables:  ", join(vars, "  "), "  \n",
              "  header offset=", length(params), "\n", params)
        for b in 1:nmb
            write(io, Int32[0, nx - 1, 0, nx - 1, 0, nx - 1, 0, 0, 0, levels[b]])
            write(io, Float64[x1[b]..., x2[b]..., x3[b]...])
            write(io, data[:, :, :, :, b])
        end
    end
    return path
end

@testset "bin files: Z4c resample and GRMHD adapter" begin
    mktempdir() do dir
        zvars = ["z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
                 "z4c_gzz", "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz"]
        nx = 5; nv = length(zvars)
        bounds = ((-4.0, 4.0), (-1.0, 1.0), (1.0, 3.0))
        x1 = [bounds[1], bounds[2], bounds[3]]; x2 = [bounds[1], bounds[2], (-1.0, 1.0)]
        x3 = [bounds[1], bounds[2], (-1.0, 1.0)]
        level = [0, 1, 1]
        data = Array{Float32,5}(undef, nx, nx, nx, nv, 3)
        for b in 1:3, v in 1:nv, k in 1:nx, j in 1:nx, i in 1:nx
            cx = x1[b][1] + (i - 0.5) * (x1[b][2] - x1[b][1]) / nx
            cy = x2[b][1] + (j - 0.5) * (x2[b][2] - x2[b][1]) / nx
            cz = x3[b][1] + (k - 0.5) * (x3[b][2] - x3[b][1]) / nx
            val = v == 1 ? 0.4 + 0.3 * sin(0.3cx) * cos(0.2cy) :
                           0.2 * sin(0.17cx * v) + 0.1 * cos(cy + v) + 0.03cz * v
            data[i, j, k, v, b] = Float32(val)
        end
        data[2, 2, 2, 1, 1] = NaN32
        path = write_bin(joinpath(dir, "z4c.bin"), zvars, data, x1, x2, x3, level)
        n = read_athenak_bin(path); o = OS.read_athenak_bin(path)
        @test all(getfield(n, f) == getfield(o, f) for f in fieldnames(AthenaKBin) if f != :data)
        @test isequal(n.data, o.data)
        nbad = 0
        for _ in 1:NPTS, v in (1, 8)
            p = 5.0 .* (2 .* rand(rng, 3) .- 1)
            rand(rng) < 0.3 && (p[1] = rand(rng, (-4.0, -1.0, 1.0, 3.0, 4.0)))
            nbad += callf(sample_var, n, v, p...) !== callf(OS.sample_var, o, v, p...)
        end
        @test nbad == 0
        for (half, ng) in ((6.0, 20), (3.0, 17))
            grid = resample_z4c_adm(n, half, ng)
            axis, α, β, γ = OS.resample_z4c_adm(o, half, ng)
            @test grid.data[1, :, :, :] == α && grid.data[2:4, :, :, :] == β &&
                  grid.data[5:10, :, :, :] == γ
            @test (grid.x0, grid.dx, grid.nx) === (first(axis), step(axis), length(axis))
        end
        g, s = z4c_grid_spacetime(path, 6.0, 20; alpha_min = 0.2)
        og, os_ = OS.z4c_grid_spacetime(path, 6.0, 20; alpha_min = 0.2)
        @test all(getfield(g, f) === getfield(og, f) for f in fieldnames(typeof(og)) if f != :data)
        @test g.data == og.data && s.time == os_.time
        # the GRMHD adapter
        fvars = ["dens", "velx", "vely", "velz", "press", "bcc1", "bcc2", "bcc3", "temperature"]
        fb = rand(rng, Float32, 4, 4, 4, length(fvars), 2) .+ 0.1f0
        fb[1, 2, 3, 5, 2] = NaN32
        pf = write_bin(joinpath(dir, "fluid.bin"), fvars, fb, [(0.0, 1.0), (1.0, 2.0)],
                       [(0.0, 1.0), (0.0, 1.0)], [(0.0, 1.0), (0.0, 1.0)], [0, 0])
        n = athenak_bin_snapshot(pf; gamma = 13 / 9); o = OS.athenak_bin_snapshot(pf; gamma = 13 / 9)
        @test same_snapshot(n, o) && n.time == 1.5
        pts = [SVector(2.4 * rand(rng) - 0.2, 1.2 * rand(rng) - 0.1, 1.2 * rand(rng) - 0.1) for _ in 1:NPTS]
        @test count_bad(n, o, pts) == 0
    end
end

@testset "device form and allocations" begin
    snap = DK.amr_deck()
    lo9, hi9, lv9, ll9 = nested_mesh(9)
    deep, _ = twin_snaps(rng, lo9, hi9, lv9, ll9, 2)
    g = DK.uniform_deck()
    ser = synthetic_series(snap; times = [0.0, 10.0],
                           evolve = (s, t) -> scale_density(s, 1 + t / 10))
    providers = (snap.blocks, with_locate(snap.blocks, BlockScan()), deep.blocks, snap, g,
                 FrameWindow(g, scale_density(g, 1.5), 0.0, 10.0), window(ser, 1),
                 window(ser, 1, 12.0, 16), SeriesProvider(ser))
    @test deep.blocks.locate isa LevelHash
    for p in providers[1:end-1]
        @test DK.device_isbits(p)
    end
    @test !DK.device_isbits(SeriesProvider(ser))
    alloc(p, x) = (primitives(p, x); @allocated primitives(p, x))
    for p in providers, x in (SVector(-3.0, 5.0, 1.0, 0.3), SVector(-3.0, 50.0, 1.0, 0.3))
        @test alloc(p, x) == 0
    end
end

end # testset
