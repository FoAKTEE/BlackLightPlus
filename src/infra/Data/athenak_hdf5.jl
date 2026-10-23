# AthenaK `.athdf` (HDF5) snapshots: the host record that pairs the fluid blocks with
# the mesh metadata, its constructors from a file and from arrays, and the mesh
# comparison used to share geometry between the frames of a series. HDF5.jl reverses the
# C-ordered axes: on disk `uov (nvar, nmb, nk, nj, ni)` reads as `(ni, nj, nk, nmb, nvar)`.
# AthenaK: Stone et al. (2024), ApJS 271, 9.

# Fluid blocks of a host snapshot: `Array` storage and one of the three locate strategies.
const _HostBlocks = Union{FluidBlocks{Array{Float32,5},Matrix{Float64},LeafTable{Array{Int32,3}}},
                          FluidBlocks{Array{Float32,5},Matrix{Float64},LevelHash},
                          FluidBlocks{Array{Float32,5},Matrix{Float64},BlockScan}}

"""
    AthenaKSnapshot <: GridProvider
    AthenaKSnapshot(filename; gamma = nothing)
    AthenaKSnapshot(data, lo, hi, x_left, dx, levels, logical_locations;
                    names = PRIM_NAMES, time = 0.0)

One AthenaK snapshot on the host. Fields: `blocks`, the [`FluidBlocks`](@ref) sampled by
`primitives`; `levels`, the AMR level of each block; `logical_locations`, the `(3, nmb)`
logical index of each block at its level; `names`, the primitive names in storage order;
`time`, the simulation time of the dump (code units; `NaN` when an `.athdf` file carries
no `Time` attribute).

The file constructor reads `uov`, `B`, the cell centres `x1v, x2v, x3v`, the faces
`x1f, x2f, x3f`, `Levels`, `LogicalLocations` and `VariableNames`. Two `uov` layouts
are accepted: `dens velx vely velz eint` (stored as is) and `dens velx vely velz press
[temperature]` (the `mhd_w_bcc` output with `use_e = true`), which needs `gamma`:
`eint = press/(gamma - 1)` is computed in `Float64` and rounded once to `Float32`, the
temperature columns are dropped and non-finite cells are zeroed with one warning. Any
other layout is an error. The block geometry is `lo = x?f[1]`, `hi = x?f[end]`,
`x_left = x?v[1]`, `dx = x?v[2] - x?v[1]` per block.

The array constructor takes `data` as `(8, ni, nj, nk, nmb)` in `PRIM_NAMES` order,
`lo`, `hi`, `x_left`, `dx` as `(3, nmb)` matrices (or vectors of `SVector{3}`), and
`levels`, `logical_locations` as above; arrays already of type `Array{Float32,5}` and
`Matrix{Float64}` are stored without a copy. Both constructors build the locate strategy
eagerly with [`build_locate`](@ref). `primitives(snap, x)` and `domain_radius(snap)`
are those of `snap.blocks`; `Adapt.adapt(to, snap)` is `Adapt.adapt(to, snap.blocks)`.
"""
struct AthenaKSnapshot <: GridProvider
    blocks::_HostBlocks
    levels::Vector{Int}
    logical_locations::Matrix{Int}
    names::Vector{String}
    time::Float64
end

Adapt.adapt_structure(to, s::AthenaKSnapshot) = Adapt.adapt(to, s.blocks)

@inline primitives(s::AthenaKSnapshot, x::SVector{4}) = primitives(s.blocks, x)
@inline locate(s::AthenaKSnapshot, x, y, z) = locate(s.blocks, x, y, z)
domain_radius(s::AthenaKSnapshot) = domain_radius(s.blocks)

_as_columns(a::AbstractMatrix) = convert(Matrix{Float64}, a)
_as_columns(v::AbstractVector{<:StaticVector{3}}) =
    Float64[v[m][a] for a in 1:3, m in eachindex(v)]

function AthenaKSnapshot(data::AbstractArray{<:Real,5}, lo, hi, x_left, dx,
                         levels::AbstractVector{<:Integer},
                         logical_locations::AbstractMatrix{<:Integer};
                         names = collect(String, PRIM_NAMES), time::Real = 0.0)
    nf, ni, nj, nk, nmb = size(data)
    nf == 8 || throw(ArgumentError("data must be (8, ni, nj, nk, nmb); got $(size(data))"))
    geometry = map(_as_columns, (lo, hi, x_left, dx))
    all(g -> size(g) == (3, nmb), geometry) ||
        throw(ArgumentError("lo, hi, x_left, dx must be (3, $nmb)"))
    (length(levels) == nmb && size(logical_locations) == (3, nmb)) ||
        throw(ArgumentError("levels must have $nmb entries and logical_locations be (3, $nmb)"))
    lo_, hi_, xl_, dx_ = geometry
    lv = collect(Int, levels)
    ll = convert(Matrix{Int}, logical_locations)
    blocks = FluidBlocks(convert(Array{Float32,5}, data), xl_, dx_, lo_, hi_, ni, nj, nk, nmb,
                         build_locate(lo_, hi_, lv, ll))
    return AthenaKSnapshot(blocks, lv, ll, collect(String, names), Float64(time))
end

function AthenaKSnapshot(filename::AbstractString; gamma::Union{Nothing,Real} = nothing)
    f = h5open(filename, "r") do h
        (uov = read(h["uov"]), B = read(h["B"]),
         x1v = read(h["x1v"]), x2v = read(h["x2v"]), x3v = read(h["x3v"]),
         x1f = read(h["x1f"]), x2f = read(h["x2f"]), x3f = read(h["x3f"]),
         levels = read(h["Levels"]), loclocs = read(h["LogicalLocations"]),
         names = String.(read_attribute(h, "VariableNames")),
         time = haskey(attributes(h), "Time") ? _scalar(read_attribute(h, "Time")) : NaN)
    end
    if Tuple(f.names) == PRIM_NAMES
        data = _athdf_eint_layout(f.uov, f.B)
    else
        data = _athdf_press_layout(f.uov, f.B, f.names, gamma, filename)
    end
    nmb = size(data, 5)
    lo = Float64[x[1, m] for x in (f.x1f, f.x2f, f.x3f), m in 1:nmb]
    hi = Float64[x[end, m] for x in (f.x1f, f.x2f, f.x3f), m in 1:nmb]
    xl = Float64[x[1, m] for x in (f.x1v, f.x2v, f.x3v), m in 1:nmb]
    dx = Float64[x[2, m] - x[1, m] for x in (f.x1v, f.x2v, f.x3v), m in 1:nmb]
    return AthenaKSnapshot(data, lo, hi, xl, dx, Int.(f.levels), Int.(f.loclocs);
                           time = f.time)
end

_scalar(t::Real) = Float64(t)
_scalar(t::AbstractArray) = Float64(only(t))

# The `eint` layout: uov = dens velx vely velz eint, B = bcc1..3, copied feature fastest.
function _athdf_eint_layout(uov, B)
    ni, nj, nk, nmb, _ = size(uov)
    data = Array{Float32,5}(undef, 8, ni, nj, nk, nmb)
    @inbounds for m in 1:nmb, k in 1:nk, j in 1:nj, i in 1:ni, p in 1:8
        data[p, i, j, k, m] = p <= 5 ? uov[i, j, k, m, p] : B[i, j, k, m, p - 5]
    end
    return data
end

# The `mhd_w_bcc` layout with `use_e = true`: uov = dens velx vely velz press
# [temperature], B = bcc1..3. Returns the feature-fastest `PRIM_NAMES` block with
# eint = press/(gamma-1) (Float64, one Float32 rounding) and non-finite cells zeroed.
function _athdf_press_layout(uov, B, names, gamma, filename)
    want = ("dens", "velx", "vely", "velz", "press")
    (length(names) >= 8 && Tuple(names[1:5]) == want &&
     Tuple(names[end-2:end]) == ("bcc1", "bcc2", "bcc3") &&
     all(==("temperature"), names[6:end-3])) ||
        error("unexpected AthenaK primitive order $names (expected " *
              "$(collect(PRIM_NAMES)) or dens velx vely velz press [temperature] bcc1 bcc2 bcc3)")
    gamma === nothing &&
        error("AthenaKSnapshot($filename): uov stores press, so the adiabatic index is " *
              "needed for eint = press/(gamma-1); pass gamma=...")
    gm1 = Float64(gamma) - 1.0
    ni, nj, nk, nmb, _ = size(uov)
    data = Array{Float32,5}(undef, 8, ni, nj, nk, nmb)
    nbad = Threads.Atomic{Int}(0)
    Threads.@threads for m in 1:nmb
        bad = 0
        @inbounds for k in 1:nk, j in 1:nj, i in 1:ni, prim in 1:8
            val = prim <= 5 ? uov[i, j, k, m, prim] : B[i, j, k, m, prim - 5]
            isfinite(val) || (val = 0.0f0; bad += 1)
            data[prim, i, j, k, m] = prim == 5 ? Float32(Float64(val) / gm1) : val
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    nbad[] > 0 && @warn "AthenaKSnapshot: zeroed $(nbad[]) non-finite cell values in " *
                        "$filename (zeroed cells emit nothing)" maxlog=1
    return data
end

"""
    same_mesh(a::AthenaKSnapshot, b::AthenaKSnapshot) -> Bool

Whether two snapshots have the same mesh: equal block count, levels, logical locations
and face bounds, which fix the locate strategy and its tables. A regrid can permute or
move blocks at a constant block count, so the bounding box alone does not suffice.
"""
function same_mesh(a::AthenaKSnapshot, b::AthenaKSnapshot)
    a.blocks.nmb == b.blocks.nmb || return false
    a.levels == b.levels || return false
    a.logical_locations == b.logical_locations || return false
    return a.blocks.lo == b.blocks.lo && a.blocks.hi == b.blocks.hi
end

"""
    share_mesh(snap::AthenaKSnapshot, ref::AthenaKSnapshot) -> AthenaKSnapshot

`snap` with its locate strategy, and its block geometry where it is equal, replaced by
those of `ref` when the two have the [`same_mesh`](@ref); otherwise `snap` itself. The
arrays read by `primitives` hold the same values either way, so the samples are
unchanged; the shared objects are uploaded to a device once.
"""
function share_mesh(snap::AthenaKSnapshot, ref::AthenaKSnapshot)
    (snap === ref || !same_mesh(snap, ref)) && return snap
    a = snap.blocks
    r = ref.blocks
    geometry_equal = same_mesh(a, r)
    b = FluidBlocks(a.data, geometry_equal ? r.x_left : a.x_left, geometry_equal ? r.dx : a.dx,
                    r.lo, r.hi, a.ni, a.nj, a.nk, a.nmb, r.locate)
    return AthenaKSnapshot(b, ref.levels, ref.logical_locations, snap.names, snap.time)
end
