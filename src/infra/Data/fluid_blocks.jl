# AthenaK meshblocks as a grid provider: the primitives of every block in one
# feature-fastest array, the per-block geometry, and a locate strategy that maps a point
# to its owning block. The sampler is clamped trilinear inside the owning block (no
# ghost cells): exact deeper than half a cell from a block face, one-sided in the outer
# half-cell shell. One struct serves host and device; `Adapt.adapt` replaces the arrays
# and turns the host-only `LevelHash` into the `BlockScan` that every device runs.
# AthenaK: Stone et al. (2024), ApJS 271, 9.

"Primitive order of the fluid arrays, the AthenaK `VariableNames` of the `eint` layout."
const PRIM_NAMES = ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3")

# Cap on the cells of a dense leaf table; a larger finest-block grid uses `LevelHash`.
const _LEAF_MAX_CELLS = 8_000_000

"""
    LeafTable{A<:AbstractArray{Int32,3}}

Dense finest-block index grid: `leaf[i, j, k]` is the owning block (1-based, 0 for
vacuum) of finest-block cell `(i, j, k)`, whose low corner is
`(x1min + (i-1)/inv_fdx1, …)`. Fields: `leaf`, the mesh bounding box `x1min … x3max`,
the inverse finest-block widths `inv_fdx1 … inv_fdx3`, the grid size `N1, N2, N3` and
the block count `nmb`. A point is assigned to the cell `⌊(x - x1min) inv_fdx1⌋` (and
likewise per axis), so a shared face belongs to the block on its high side; outside
`[0, N)` on any axis the point is vacuum. Built by [`leaf_table`](@ref); `isbits` after
`Adapt.adapt`.
"""
struct LeafTable{A<:AbstractArray{Int32,3}}
    leaf::A
    x1min::Float64
    x2min::Float64
    x3min::Float64
    x1max::Float64
    x2max::Float64
    x3max::Float64
    inv_fdx1::Float64
    inv_fdx2::Float64
    inv_fdx3::Float64
    N1::Int
    N2::Int
    N3::Int
    nmb::Int
end
Adapt.@adapt_structure LeafTable

"""
    LevelHash

Per-level hash of the leaf blocks, for meshes without a dense leaf table (over
8 000 000 finest-block cells, or not aligned to the finest grid): `dict[(L, lx1, lx2, lx3)]` is the block at level `L` and logical
location `(lx1, lx2, lx3)`; `levels` lists the levels present in ascending order;
`x1min … x3max` is the mesh bounding box and `inv_w0` the inverse root-block width per
axis. A point inside the half-open box `[min, max)` is located by probing the levels
coarse to fine at logical index `⌊(x - xmin) 2^L / w0⌋`; the first hit wins. Host only:
`Adapt.adapt` returns `BlockScan()`. Built by [`level_hash`](@ref).
"""
struct LevelHash
    dict::Dict{NTuple{4,Int},Int32}
    levels::Vector{Int}
    x1min::Float64
    x2min::Float64
    x3min::Float64
    x1max::Float64
    x2max::Float64
    x3max::Float64
    inv_w0::SVector{3,Float64}
end
Adapt.adapt_structure(to, ::LevelHash) = BlockScan()

"""
    BlockScan()

Linear scan over the block bounds: a point belongs to block `m` when
`lo[a, m] < x_a ≤ hi[a, m]` on every axis, and the last such block in storage order
wins (AthenaK stores finer blocks later). `O(nmb)` per point; valid for any mesh and on
any device.
"""
struct BlockScan end

"""
    FluidBlocks{A5,A2,L} <: GridProvider

The fluid of an AthenaK snapshot as a sampling struct. Fields:

- `data`: `(8, ni, nj, nk, nmb)` `Float32` cell-centred primitives, feature fastest, in
  the order `PRIM_NAMES = (dens, velx, vely, velz, eint, bcc1, bcc2, bcc3)`: rest-mass
  density, normal-frame velocity `W v^i`, internal energy density, cell-centred field;
  code units.
- `x_left`, `dx`: `(3, nmb)` `Float64` first cell centre and cell width of each block.
- `lo`, `hi`: `(3, nmb)` `Float64` face bounds of each block.
- `ni, nj, nk`: cells per block along each axis; `nmb`: number of blocks.
- `locate`: the strategy that finds a point's block, a [`LeafTable`](@ref),
  [`LevelHash`](@ref) or [`BlockScan`](@ref).

`primitives(b, x)` is the clamped trilinear sample in the owning block, in `Float64`.
`Adapt.adapt(to, b)` moves every array and turns a `LevelHash` into a `BlockScan`; the
result is `isbits` on a device. Built by the [`AthenaKSnapshot`](@ref) constructors.
"""
struct FluidBlocks{A5,A2,L} <: GridProvider
    data::A5
    x_left::A2
    dx::A2
    lo::A2
    hi::A2
    ni::Int
    nj::Int
    nk::Int
    nmb::Int
    locate::L
end
Adapt.@adapt_structure FluidBlocks

"""
    with_locate(b::FluidBlocks, strategy) -> FluidBlocks

`b` with its locate strategy replaced by `strategy`, sharing every array.
"""
with_locate(b::FluidBlocks, strategy) =
    FluidBlocks(b.data, b.x_left, b.dx, b.lo, b.hi, b.ni, b.nj, b.nk, b.nmb, strategy)

# ---------------------------------------------------------------- locate tables (host)

# Bounding box of the blocks: (x1min, x2min, x3min, x1max, x2max, x3max).
function _bounding_box(lo::AbstractMatrix, hi::AbstractMatrix, nmb::Int)
    x1min = minimum(lo[1, m] for m in 1:nmb)
    x2min = minimum(lo[2, m] for m in 1:nmb)
    x3min = minimum(lo[3, m] for m in 1:nmb)
    x1max = maximum(hi[1, m] for m in 1:nmb)
    x2max = maximum(hi[2, m] for m in 1:nmb)
    x3max = maximum(hi[3, m] for m in 1:nmb)
    return x1min, x2min, x3min, x1max, x2max, x3max
end

_block_width(lo, hi, m) = SVector(hi[1, m] - lo[1, m], hi[2, m] - lo[2, m], hi[3, m] - lo[3, m])

"""
    leaf_table(lo, hi, levels, logical_locations) -> Union{LeafTable, Nothing}

The dense finest-block index grid of a mesh with `(3, nmb)` face bounds `lo`, `hi`,
per-block AMR `levels` and `(3, nmb)` `logical_locations`. The finest width per axis is
the smallest block width; a level-`L` block at logical `lx` covers finest cells
`[lx 2^s, (lx+1) 2^s)` with `s = Lmax - L`. Blocks are written in storage order, so
where two overlap the later one owns the cell. Returns `nothing` for an empty or
degenerate mesh, a grid over 8 000 000 cells, or blocks that fall outside the grid (a
mesh not aligned to its finest level from the domain origin).
"""
function leaf_table(lo::AbstractMatrix, hi::AbstractMatrix, levels::AbstractVector,
                    logical_locations::AbstractMatrix)
    nmb = size(lo, 2)
    nmb >= 1 || return nothing
    x1min, x2min, x3min, x1max, x2max, x3max = _bounding_box(lo, hi, nmb)
    fdx1 = minimum(hi[1, m] - lo[1, m] for m in 1:nmb)
    fdx2 = minimum(hi[2, m] - lo[2, m] for m in 1:nmb)
    fdx3 = minimum(hi[3, m] - lo[3, m] for m in 1:nmb)
    (fdx1 > 0.0 && fdx2 > 0.0 && fdx3 > 0.0) || return nothing
    N1 = round(Int, (x1max - x1min) / fdx1)
    N2 = round(Int, (x2max - x2min) / fdx2)
    N3 = round(Int, (x3max - x3min) / fdx3)
    (N1 >= 1 && N2 >= 1 && N3 >= 1) || return nothing
    (N1 * N2 * N3 <= _LEAF_MAX_CELLS) || return nothing
    leaf = _fill_leaf(N1, N2, N3, levels, logical_locations, nmb)
    leaf === nothing && return nothing
    return LeafTable(leaf, x1min, x2min, x3min, x1max, x2max, x3max,
                     1.0 / fdx1, 1.0 / fdx2, 1.0 / fdx3, N1, N2, N3, nmb)
end

# The leaf grid of `leaf_table`, or `nothing` when a block lies past the grid.
function _fill_leaf(N1, N2, N3, levels, logical_locations, nmb)
    Lmax = maximum(levels)
    leaf = zeros(Int32, N1, N2, N3)
    @inbounds for m in 1:nmb
        s = Lmax - levels[m]
        lx1 = logical_locations[1, m]; lx2 = logical_locations[2, m]; lx3 = logical_locations[3, m]
        i0 = lx1 << s; i1 = (lx1 + 1) << s
        j0 = lx2 << s; j1 = (lx2 + 1) << s
        k0 = lx3 << s; k1 = (lx3 + 1) << s
        (i1 <= N1 && j1 <= N2 && k1 <= N3) || return nothing
        for kk in k0:(k1 - 1), jj in j0:(j1 - 1), ii in i0:(i1 - 1)
            leaf[ii + 1, jj + 1, kk + 1] = Int32(m)
        end
    end
    return leaf
end

"""
    level_hash(lo, hi, levels, logical_locations) -> Union{LevelHash, Nothing}

The per-level hash of a mesh (arguments as in [`leaf_table`](@ref)). The root-block
width `w0` is block 1's width times `2^level`; returns `nothing` for an empty mesh or
when any block's width times `2^level` differs from `w0` by more than `1e-9 |w0|` on an
axis.
"""
function level_hash(lo::AbstractMatrix, hi::AbstractMatrix, levels::AbstractVector,
                    logical_locations::AbstractMatrix)
    nmb = size(lo, 2)
    nmb >= 1 || return nothing
    w0 = _block_width(lo, hi, 1) .* 2.0^levels[1]
    dict = Dict{NTuple{4,Int},Int32}()
    sizehint!(dict, nmb)
    for m in 1:nmb
        L = Int(levels[m])
        wm = _block_width(lo, hi, m) .* 2.0^L
        all(abs.(wm .- w0) .<= 1e-9 .* abs.(w0)) || return nothing
        dict[(L, Int(logical_locations[1, m]), Int(logical_locations[2, m]),
              Int(logical_locations[3, m]))] = Int32(m)
    end
    x1min, x2min, x3min, x1max, x2max, x3max = _bounding_box(lo, hi, nmb)
    return LevelHash(dict, sort(unique(Int.(levels))), x1min, x2min, x3min,
                     x1max, x2max, x3max, 1.0 ./ w0)
end

"""
    build_locate(lo, hi, levels, logical_locations) -> LeafTable | LevelHash | BlockScan

The locate strategy of a mesh: its [`leaf_table`](@ref) when one can be built, else its
[`level_hash`](@ref), else [`BlockScan`](@ref).
"""
function build_locate(lo::AbstractMatrix, hi::AbstractMatrix, levels::AbstractVector,
                      logical_locations::AbstractMatrix)
    t = leaf_table(lo, hi, levels, logical_locations)
    t === nothing || return t
    h = level_hash(lo, hi, levels, logical_locations)
    h === nothing || return h
    return BlockScan()
end

# ---------------------------------------------------------------- locate (hot path)

"""
    locate(strategy, b::FluidBlocks, x, y, z) -> Int
    locate(b::FluidBlocks, x, y, z) -> Int

The block (1-based) that owns the Cartesian Kerr–Schild point `(x, y, z)`, or 0 when
no block does, under `strategy` (default `b.locate`). The strategies agree everywhere
off block faces; on a face each keeps its own rule (see [`LeafTable`](@ref),
[`LevelHash`](@ref), [`BlockScan`](@ref)).
"""
@inline locate(b::FluidBlocks, x, y, z) = locate(b.locate, b, x, y, z)

# The test `f >= 0` precedes the truncation, so `unsafe_trunc` is the floor.
@inline function locate(t::LeafTable, b, x, y, z)
    f1 = (x - t.x1min) * t.inv_fdx1
    f2 = (y - t.x2min) * t.inv_fdx2
    f3 = (z - t.x3min) * t.inv_fdx3
    (f1 < 0.0 || f1 >= t.N1 || f2 < 0.0 || f2 >= t.N2 || f3 < 0.0 || f3 >= t.N3) && return 0
    i1 = unsafe_trunc(Int, f1); i2 = unsafe_trunc(Int, f2); i3 = unsafe_trunc(Int, f3)
    return Int(@inbounds t.leaf[i1 + 1, i2 + 1, i3 + 1])
end

@inline function locate(h::LevelHash, b, x, y, z)
    (h.x1min <= x < h.x1max && h.x2min <= y < h.x2max && h.x3min <= z < h.x3max) || return 0
    f1 = (x - h.x1min) * h.inv_w0[1]
    f2 = (y - h.x2min) * h.inv_w0[2]
    f3 = (z - h.x3min) * h.inv_w0[3]
    @inbounds for L in h.levels
        s = ldexp(1.0, L)
        m = get(h.dict, (L, unsafe_trunc(Int, f1 * s), unsafe_trunc(Int, f2 * s),
                         unsafe_trunc(Int, f3 * s)), Int32(0))
        m != 0 && return Int(m)
    end
    return 0
end

@inline function locate(::BlockScan, b, x, y, z)
    lo = b.lo
    hi = b.hi
    mb = 0
    @inbounds for m in 1:b.nmb
        if lo[1, m] < x <= hi[1, m] && lo[2, m] < y <= hi[2, m] && lo[3, m] < z <= hi[3, m]
            mb = m
        end
    end
    return mb
end

# ---------------------------------------------------------------- sample (hot path)

"""
    corner(b::FluidBlocks, i, j, k, mb) -> SVector{8,Float64}

The eight primitives of cell `(i, j, k)` (1-based) of block `mb`, widened to `Float64`.
"""
@inline corner(b::FluidBlocks, i, j, k, mb) =
    SVector{8,Float64}(ntuple(f -> Float64(@inbounds b.data[f, i, j, k, mb]), Val(8)))

"""
    block_sample(b::FluidBlocks, mb, x, y, z) -> SVector{8,Float64}

Clamped trilinear sample of the eight primitives inside block `mb` at `(x, y, z)`: the
fractional cell index is `(x - x_left)/dx` per axis (a division, so `dx` is never
inverted), the lower corner and the blend fraction are clamped by `corner_weights`, and
the corners are blended by `lerp8` along `i`, then `j`, then `k`.
"""
@inline function block_sample(b::FluidBlocks, mb::Int, x, y, z)
    xl = b.x_left
    dx = b.dx
    g1 = (x - @inbounds(xl[1, mb])) / @inbounds(dx[1, mb])
    g2 = (y - @inbounds(xl[2, mb])) / @inbounds(dx[2, mb])
    g3 = (z - @inbounds(xl[3, mb])) / @inbounds(dx[3, mb])
    i0, d1 = corner_weights(g1, b.ni)
    j0, d2 = corner_weights(g2, b.nj)
    k0, d3 = corner_weights(g3, b.nk)
    i1 = i0 + 1; j1 = j0 + 1; k1 = k0 + 1
    i2 = i1 + 1; j2 = j1 + 1; k2 = k1 + 1
    c = (corner(b, i1, j1, k1, mb), corner(b, i2, j1, k1, mb),
         corner(b, i1, j2, k1, mb), corner(b, i2, j2, k1, mb),
         corner(b, i1, j1, k2, mb), corner(b, i2, j1, k2, mb),
         corner(b, i1, j2, k2, mb), corner(b, i2, j2, k2, mb))
    return lerp8(c, d1, d2, d3)
end

# The record of an eight-primitive sample `p` in `PRIM_NAMES` order.
@inline _record(p::SVector{8,Float64}, inside::Bool) =
    PlasmaPrimitives{Float64}(p[1], SVector(p[2], p[3], p[4]), p[5],
                              SVector(p[6], p[7], p[8]), inside)

"""
    primitives(b::FluidBlocks, x::SVector{4}) -> PlasmaPrimitives{Float64}

The primitives at the spacetime point `x = (t, x, y, z)` (Cartesian Kerr–Schild; the
time is ignored, the position is converted to `Float64`): the [`block_sample`](@ref) in
the block found by `b.locate`, or the outside record when no block owns the point.
"""
@inline function primitives(b::FluidBlocks, x::SVector{4})
    px = Float64(x[2]); py = Float64(x[3]); pz = Float64(x[4])
    mb = locate(b, px, py, pz)
    mb == 0 && return PlasmaPrimitives{Float64}()
    return _record(block_sample(b, mb, px, py, pz), true)
end

# ---------------------------------------------------------------- domain radius (host)

"""
    domain_radius(b::FluidBlocks) -> Float64

Radius of the sphere about the origin that circumscribes the located domain; every point
beyond it is vacuum. With a `LeafTable` the box is the leaf grid,
`[x1min, x1min + N1/inv_fdx1] × …`; with a `LevelHash` or `BlockScan` it is the bounding
box of the block faces. The radius is
`√(max(x1lo², x1hi²) + max(x2lo², x2hi²) + max(x3lo², x3hi²))`. Host only.
"""
domain_radius(b::FluidBlocks) = _domain_radius(b.locate, b)

function _domain_radius(t::LeafTable, b)
    x1hi = t.x1min + t.N1 / t.inv_fdx1
    x2hi = t.x2min + t.N2 / t.inv_fdx2
    x3hi = t.x3min + t.N3 / t.inv_fdx3
    return sqrt(max(t.x1min^2, x1hi^2) + max(t.x2min^2, x2hi^2) + max(t.x3min^2, x3hi^2))
end

_domain_radius(h::LevelHash, b) =
    sqrt(max(h.x1min^2, h.x1max^2) + max(h.x2min^2, h.x2max^2) + max(h.x3min^2, h.x3max^2))

function _domain_radius(::BlockScan, b)
    x1min, x2min, x3min, x1max, x2max, x3max = _bounding_box(b.lo, b.hi, b.nmb)
    return sqrt(max(x1min^2, x1max^2) + max(x2min^2, x2max^2) + max(x3min^2, x3max^2))
end

"""
    same_mesh(a::FluidBlocks, b::FluidBlocks) -> Bool

Whether `a` and `b` have the same block shape and count and identical block geometry
(`x_left`, `dx`, `lo`, `hi`), so a point has the same block and cell in both.
"""
same_mesh(a::FluidBlocks, b::FluidBlocks) =
    (a.ni, a.nj, a.nk, a.nmb) == (b.ni, b.nj, b.nk, b.nmb) &&
    (a.x_left === b.x_left || a.x_left == b.x_left) && (a.dx === b.dx || a.dx == b.dx) &&
    (a.lo === b.lo || a.lo == b.lo) && (a.hi === b.hi || a.hi == b.hi)
