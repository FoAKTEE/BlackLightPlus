"""
athenak_reader.jl — AthenaK `.athdf` (HDF5) GRMHD snapshot reader and trilinear
primitive sampler (AthenaK: Stone et al. 2024). Ghost-free: the stencil is
clamped to the owning meshblock interior — exact deeper than half a cell from
block faces, one-sided first-order in the outer shell; seam/clamp error is well
below the ray step-quadrature error. HDF5.jl reverses C-ordered axes: on-disk
`uov (5,nmb,nk,nj,ni)` becomes `(ni,nj,nk,nmb,5)`, indexed `[i,j,k,mb,prim]` in
`VariableNames` order; Float32 cell data, Float64 interpolation.
"""

# Primitive axis order == VariableNames == concat([uov; B]); pins the slicing below.
const _PRIM_NAMES = ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3")

# ------------------------------------------------------------ O(1) leaf-block locate
#
# Every AthenaK block face is an exact multiple of the finest block extent and the
# domain origin is aligned, so a dense index grid at finest-block spacing maps any
# point to its owning leaf block in one floor + one indexed read. Validated
# bit-identical to the linear scan for every point off a block face (faces are
# measure-zero for Float64 geodesic samples).

"""
    LeafTable

Dense finest-block index grid for O(1) meshblock location: `leaf[i,j,k]` is the
owning meshblock (1-based; 0 = vacuum) of finest-block cell `(i,j,k)`. Carries
the mesh bounds and inverse finest spacing used by the locate.
"""
struct LeafTable
    leaf::Array{Int32,3}
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

# Lazy, thread-safe holder for a snapshot's leaf table. `state` (0 = not attempted,
# 1 = attempted) uses release/acquire ordering to publish the `tbl` write made under
# `lk` to the lock-free fast path. `tbl === nothing` after `state == 1` means the
# grid was too large / degenerate and the scan fallback is used permanently.
mutable struct LeafCache
    @atomic state::Int8
    tbl::Union{Nothing,LeafTable}
    lk::ReentrantLock
end
LeafCache() = LeafCache(Int8(0), nothing, ReentrantLock())

# Max finest-index cells; larger meshes fall back to the linear scan.
const _LEAF_MAX_CELLS = 8_000_000

# Build the dense finest-block index grid from raw per-block geometry. Returns
# `nothing` for a degenerate / oversized mesh (caller falls back to the linear scan).
# A level-L block at logical (lx1,lx2,lx3) covers finest indices [lx<<s, (lx+1)<<s)
# per axis (s = Lmax - L); iterate m ascending and overwrite so a shared face
# resolves to the later (finer) block — the scan's last-wins tiebreak.
function _build_leaf_table(ext_lo::Vector{SVector{3,Float64}}, ext_hi::Vector{SVector{3,Float64}},
                           levels::Vector{Int}, logical_locations::Matrix{Int}, nmb::Int)
    nmb >= 1 || return nothing
    x1min = minimum(e[1] for e in ext_lo); x2min = minimum(e[2] for e in ext_lo); x3min = minimum(e[3] for e in ext_lo)
    x1max = maximum(e[1] for e in ext_hi); x2max = maximum(e[2] for e in ext_hi); x3max = maximum(e[3] for e in ext_hi)
    fdx1 = minimum(ext_hi[m][1] - ext_lo[m][1] for m in 1:nmb)
    fdx2 = minimum(ext_hi[m][2] - ext_lo[m][2] for m in 1:nmb)
    fdx3 = minimum(ext_hi[m][3] - ext_lo[m][3] for m in 1:nmb)
    (fdx1 > 0.0 && fdx2 > 0.0 && fdx3 > 0.0) || return nothing
    N1 = round(Int, (x1max - x1min) / fdx1)
    N2 = round(Int, (x2max - x2min) / fdx2)
    N3 = round(Int, (x3max - x3min) / fdx3)
    (N1 >= 1 && N2 >= 1 && N3 >= 1) || return nothing
    (N1 * N2 * N3 <= _LEAF_MAX_CELLS) || return nothing   # too large -> scan fallback
    Lmax = maximum(levels)
    leaf = zeros(Int32, N1, N2, N3)
    @inbounds for m in 1:nmb
        s = Lmax - levels[m]
        lx1 = logical_locations[1, m]; lx2 = logical_locations[2, m]; lx3 = logical_locations[3, m]
        i0 = lx1 << s; i1 = (lx1 + 1) << s
        j0 = lx2 << s; j1 = (lx2 + 1) << s
        k0 = lx3 << s; k1 = (lx3 + 1) << s
        (i1 <= N1 && j1 <= N2 && k1 <= N3) || return nothing   # geometry not finest-aligned
        for kk in k0:(k1 - 1), jj in j0:(j1 - 1), ii in i0:(i1 - 1)
            leaf[ii + 1, jj + 1, kk + 1] = Int32(m)
        end
    end
    return LeafTable(leaf, x1min, x2min, x3min, x1max, x2max, x3max,
                     1.0 / fdx1, 1.0 / fdx2, 1.0 / fdx3, N1, N2, N3, nmb)
end

# O(1) locate: one floor + one indexed read. f >= 0 is guaranteed before the trunc,
# so unsafe_trunc == floor. Out of range -> 0 (vacuum), matching the scan.
@inline function _leaf_locate(t::LeafTable, x, y, z)
    f1 = (x - t.x1min) * t.inv_fdx1
    f2 = (y - t.x2min) * t.inv_fdx2
    f3 = (z - t.x3min) * t.inv_fdx3
    (f1 < 0.0 || f1 >= t.N1 || f2 < 0.0 || f2 >= t.N2 || f3 < 0.0 || f3 >= t.N3) && return 0
    i1 = unsafe_trunc(Int, f1); i2 = unsafe_trunc(Int, f2); i3 = unsafe_trunc(Int, f3)
    return Int(@inbounds t.leaf[i1 + 1, i2 + 1, i3 + 1])
end

"""
    AthenaKSnapshot(filename)

Load an AthenaK `.athdf` snapshot: reads `uov`/`B` primitives, cell centres
(`x1v`,`x2v`,`x3v`), cell faces (`x1f`,`x2f`,`x3f`), `Levels`, and
`LogicalLocations` via HDF5.jl. Stores the `(ni,nj,nk,nmb,8)` primitive block,
per-meshblock face-extent bounding boxes, and per-block first cell centre +
spacing for the clamped-trilinear sampler.
"""
struct AthenaKSnapshot
    data::Array{Float32,5}                 # (ni, nj, nk, nmb, 8) == [i,j,k,mb,prim]
    ext_lo::Vector{SVector{3,Float64}}     # per-block low  face (x1,x2,x3)
    ext_hi::Vector{SVector{3,Float64}}     # per-block high face
    x_left::Vector{SVector{3,Float64}}     # per-block first cell centre
    dx::Vector{SVector{3,Float64}}         # per-block cell spacing
    levels::Vector{Int}                    # per-block AMR level
    logical_locations::Matrix{Int}         # (3, nmb) per-block logical index
    variable_names::Vector{String}
    ni::Int
    nj::Int
    nk::Int
    nmb::Int
    leafcache::LeafCache                   # lazy O(1) leaf-block index grid
end

# 12-arg constructor (no leaf cache supplied): attach a fresh cache, built lazily
# on first locate.
AthenaKSnapshot(data, ext_lo, ext_hi, x_left, dx, levels, logical_locations,
                variable_names, ni, nj, nk, nmb) =
    AthenaKSnapshot(data, ext_lo, ext_hi, x_left, dx, levels, logical_locations,
                    variable_names, ni, nj, nk, nmb, LeafCache())

function AthenaKSnapshot(filename::AbstractString)
    uov, B, x1v, x2v, x3v, x1f, x2f, x3f, levels, loclocs, varnames =
        h5open(filename, "r") do h
            (read(h["uov"]), read(h["B"]),
             read(h["x1v"]), read(h["x2v"]), read(h["x3v"]),
             read(h["x1f"]), read(h["x2f"]), read(h["x3f"]),
             read(h["Levels"]), read(h["LogicalLocations"]),
             read_attribute(h, "VariableNames"))
        end

    data = cat(uov, B; dims=5)             # (ni,nj,nk,nmb,8), prim axis == VariableNames
    ni, nj, nk, nmb, _ = size(data)

    names = String.(varnames)
    Tuple(names) == _PRIM_NAMES ||
        error("unexpected AthenaK primitive order $names (expected $(collect(_PRIM_NAMES)))")

    ext_lo = [SVector(x1f[1, m],   x2f[1, m],   x3f[1, m])   for m in 1:nmb]
    ext_hi = [SVector(x1f[end, m], x2f[end, m], x3f[end, m]) for m in 1:nmb]
    x_left = [SVector(x1v[1, m], x2v[1, m], x3v[1, m])       for m in 1:nmb]
    dx     = [SVector(x1v[2, m] - x1v[1, m],
                      x2v[2, m] - x2v[1, m],
                      x3v[2, m] - x3v[1, m])                 for m in 1:nmb]

    return AthenaKSnapshot(data, ext_lo, ext_hi, x_left, dx,
                           Int.(levels), Int.(loclocs), names, ni, nj, nk, nmb)
end

# Owning meshblock for CKS position (x,y,z), linear-scan reference: strict `<` low
# face, `<=` high face; later block in file order wins (== finest level). Returns 0
# outside every block. The leaf table reproduces this off block faces.
@inline function _find_meshblock_scan(snap::AthenaKSnapshot, x, y, z)
    mb = 0
    @inbounds for m in 1:snap.nmb
        lo = snap.ext_lo[m]
        hi = snap.ext_hi[m]
        if lo[1] < x <= hi[1] && lo[2] < y <= hi[2] && lo[3] < z <= hi[3]
            mb = m
        end
    end
    return mb
end

"""
    leaf_table(snap) -> Union{Nothing,LeafTable}

The snapshot's O(1) leaf-block index grid, built lazily and cached (a lock guards
the one-time build; an atomic flag publishes it lock-free afterwards). Returns
`nothing` for a mesh too large / not finest-aligned (callers then use the scan).
"""
@inline function leaf_table(snap::AthenaKSnapshot)
    c = snap.leafcache
    (@atomic :acquire c.state) == Int8(1) && return c.tbl
    return _leaf_build_locked!(c, snap)
end

@noinline function _leaf_build_locked!(c::LeafCache, snap::AthenaKSnapshot)
    lock(c.lk)
    try
        (@atomic :acquire c.state) == Int8(1) && return c.tbl
        t = _build_leaf_table(snap.ext_lo, snap.ext_hi, snap.levels,
                              snap.logical_locations, snap.nmb)
        c.tbl = t
        @atomic :release c.state = Int8(1)
        return t
    finally
        unlock(c.lk)
    end
end

# Do snapshots `a` and `b` share the same mesh (for SnapshotSeries table sharing)?
# Requires exact equality of every leaf-table input: an AMR regrid between dumps can
# permute/relocate blocks at constant nmb inside the fixed domain, so nmb + global
# bounding box alone is not enough.
function _mesh_matches(a::AthenaKSnapshot, b::AthenaKSnapshot)
    a.nmb == b.nmb || return false
    a.levels == b.levels || return false
    a.logical_locations == b.logical_locations || return false
    return a.ext_lo == b.ext_lo && a.ext_hi == b.ext_hi
end

"""
    meshblock_index(snap, x, y, z) -> Int

Finest owning meshblock (1-based; 0 = vacuum) of CKS position `(x,y,z)`, via the
O(1) leaf table when available and the linear scan otherwise. The single locate
entry point for sampling.
"""
@inline function meshblock_index(snap::AthenaKSnapshot, x, y, z)
    t = leaf_table(snap)
    t === nothing && return _find_meshblock_scan(snap, x, y, z)
    return _leaf_locate(t, x, y, z)
end

# Internal alias for meshblock_index.
@inline _find_meshblock(snap::AthenaKSnapshot, x, y, z) = meshblock_index(snap, x, y, z)

# The 8 primitives at one cell centre, promoted to Float64 (SVector{8}).
@inline _corner(data, i, j, k, mb) =
    SVector{8,Float64}(ntuple(p -> Float64(@inbounds data[i, j, k, mb, p]), Val(8)))

"""
    sample_primitives(snap, pos::SVector{4}) -> NamedTuple

Clamped trilinear sample of the 8 primitives at CKS position `pos = (t,x,y,z)`.
Returns `(rho, eint, uu::SVector{3}, bb::SVector{3}, inside::Bool)`; outside every
meshblock, `inside = false` and all fields zero. Cell indices are clamped so the
`i0+1` upper corner stays in range (ghost-free edge behaviour); clamp-shell and
refinement-seam errors are an order below the ray step-quadrature error.
"""
function sample_primitives(snap::AthenaKSnapshot, pos::SVector{4})
    x, y, z = pos[2], pos[3], pos[4]
    mb = _find_meshblock(snap, x, y, z)
    z3 = zero(SVector{3,Float64})
    mb == 0 && return (rho=0.0, eint=0.0, uu=z3, bb=z3, inside=false)

    xyz = SVector(x, y, z)
    fi  = (xyz - snap.x_left[mb]) ./ snap.dx[mb]          # cell-unit coordinate (x1,x2,x3)
    nax = SVector(snap.ni, snap.nj, snap.nk)
    i0  = clamp.(floor.(Int, fi), 0, nax .- 2)            # 0-based low, i0+1 in range
    d   = clamp.(fi .- i0, 0.0, 1.0)                       # one-sided blend at faces

    il1, il2, il3 = i0[1] + 1, i0[2] + 1, i0[3] + 1        # 1-based low corner (i,j,k)
    ih1, ih2, ih3 = il1 + 1, il2 + 1, il3 + 1
    d1, d2, d3 = d[1], d[2], d[3]

    data = snap.data
    c000 = _corner(data, il1, il2, il3, mb)
    c100 = _corner(data, ih1, il2, il3, mb)
    c010 = _corner(data, il1, ih2, il3, mb)
    c110 = _corner(data, ih1, ih2, il3, mb)
    c001 = _corner(data, il1, il2, ih3, mb)
    c101 = _corner(data, ih1, il2, ih3, mb)
    c011 = _corner(data, il1, ih2, ih3, mb)
    c111 = _corner(data, ih1, ih2, ih3, mb)

    c00 = c000 + (c100 - c000) * d1                        # lerp along x1 (i)
    c10 = c010 + (c110 - c010) * d1
    c01 = c001 + (c101 - c001) * d1
    c11 = c011 + (c111 - c011) * d1
    c0  = c00 + (c10 - c00) * d2                           # lerp along x2 (j)
    c1  = c01 + (c11 - c01) * d2
    p   = c0 + (c1 - c0) * d3                              # lerp along x3 (k)

    return (rho=p[1], eint=p[5],
            uu=SVector(p[2], p[3], p[4]),
            bb=SVector(p[6], p[7], p[8]),
            inside=true)
end

"""
    units(m_bh_cgs, mass_scale_cgs) -> NamedTuple

GRMHD code → CGS unit conversions: `m_bh_cgs` is the black-hole mass [g] and
`mass_scale_cgs` the density mass scale [g]. Returns
`(L_unit, T_unit, dens_unit, Ne_unit, B_unit)` with L = GM/c², T = L/c,
dens = mass_scale / L³, Ne = dens / (m_p + m_e), B = c·sqrt(4π dens).
"""
@inline function units(m_bh_cgs, mass_scale_cgs)
    L_unit    = GNEWT * m_bh_cgs / CL^2
    T_unit    = L_unit / CL
    dens_unit = mass_scale_cgs / L_unit^3
    Ne_unit   = dens_unit / (MP + ME)
    B_unit    = CL * sqrt(4π * dens_unit)
    return (L_unit=L_unit, T_unit=T_unit, dens_unit=dens_unit,
            Ne_unit=Ne_unit, B_unit=B_unit)
end
