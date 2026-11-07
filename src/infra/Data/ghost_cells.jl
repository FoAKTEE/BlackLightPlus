# The ghost layer of every meshblock, built once from the mesh at load time. The ghost at
# centre `p` of a block whose cells have widths `h` takes the field of the mesh on the
# cell of widths `h` about `p` (`mesh_value`): a copy where a block of the same level
# covers `p`, the mean of the finer cells inside the ghost cell where a finer block does
# (restriction), the minmod-limited piecewise-linear prolongation of the coarse cell that
# contains `p` where a coarser block does (AthenaK `ProlongCC`), and the one-sided linear
# extrapolation along each axis on which `p` lies outside the mesh. A ghost value depends
# on `p` and `h` alone, never on the block that asks, so blocks of one level that share a
# face see the same nodes along it. The method page states the scheme and its order.
# AthenaK: Stone et al. (2024), ApJS 271, 9.

# Relative tolerance of a cell-width comparison (widths of one level agree to rounding).
const _WIDTH_RTOL = 1.0e-6
# Recursion bound of `mesh_value`: a 2:1-balanced mesh needs three levels; a mesh that is
# not leaf-aligned falls back to the containing cell's value past this depth.
const _MAX_DEPTH = 16

"""
    MeshBox

The bounding box of a block mesh, `lo` and `hi` its low and high corners: a point is
outside the mesh along axis `a` when `x[a] < lo[a]` or `x[a] > hi[a]`.
"""
struct MeshBox
    lo::SVector{3,Float64}
    hi::SVector{3,Float64}
end

MeshBox(b::FluidBlocks) = (bb = _bounding_box(b.lo, b.hi, b.nmb);
                           MeshBox(SVector(bb[1], bb[2], bb[3]), SVector(bb[4], bb[5], bb[6])))

_cells(b::FluidBlocks) = SVector(b.ni, b.nj, b.nk)
_width(b::FluidBlocks, m) = SVector(b.dx[1, m], b.dx[2, m], b.dx[3, m])
_left(b::FluidBlocks, m) = SVector(b.x_left[1, m], b.x_left[2, m], b.x_left[3, m])

# The interior cell of block `m` that contains `p` (1-based, clamped to the block).
function _containing_cell(b::FluidBlocks, m, p)
    n = _cells(b)
    return SVector(ntuple(a -> clamp(floor(Int, (p[a] - b.lo[a, m]) / b.dx[a, m]) + 1, 1, n[a]),
                          Val(3)))
end

_cell_value(b::FluidBlocks, m, c) = corner(b, c[1], c[2], c[3], m)

"""
    mesh_value(b::FluidBlocks, box::MeshBox, p, h; depth = 0)
        -> Union{SVector{8,Float64}, Nothing}

The eight primitives of the mesh `b` on the cell of widths `h` centred at `p` (the
interior cells of every block are read; ghost layers never are), or `nothing` where no
block covers `p` inside `box` (a mesh with a hole). With `m` the block that covers `p`
and `H` its cell widths:

- outside `box` along an axis `a` (the first such axis): `2 V(p − s h_a e_a) − V(p −
  2 s h_a e_a)`, `s` the outward sign, the one-sided linear extrapolation (the value
  `V(p − s h_a e_a)` held when the box is less than two cells of width `h_a` thick);
- `H = h`: the cell of `m` at `p`;
- `H ≤ h` (finer on some axes): the mean of `V` over the children of the cell, two at
  `p_a ± h_a/4` with width `h_a/2` on every axis where `H_a < h_a` (restriction, the
  volume average of the finer cells);
- `H ≥ h` (coarser on some axes): `u_c + Σ_a σ_a (p_a − x_{c,a})/H_a` over the coarser
  axes, `u_c` the cell of `m` that contains `p`, centre `x_c`, and `σ_a = minmod(u_c −
  V(x_c − H_a e_a), V(x_c + H_a e_a) − u_c)` per primitive, each side the value at width
  `H`, one side alone where the other is `nothing` (prolongation); the offset
  `(p_a − x_{c,a})/H_a` is taken on its leaf-aligned value `(k + ½)/r − ½`,
  `r = H_a/h_a`;
- otherwise, or past a recursion depth of 16: `u_c`.
"""
function mesh_value(b::FluidBlocks, box::MeshBox, p::SVector{3,Float64},
                    h::SVector{3,Float64}; depth::Int = 0)
    for a in 1:3
        s = p[a] > box.hi[a] ? 1 : p[a] < box.lo[a] ? -1 : 0
        s == 0 && continue
        e = SVector(ntuple(k -> k == a ? s * h[a] : 0.0, Val(3)))
        v1 = mesh_value(b, box, p - e, h; depth = depth + 1)
        (box.hi[a] - box.lo[a]) < 2.0 * h[a] * (1.0 - _WIDTH_RTOL) && return v1
        v2 = mesh_value(b, box, p - 2.0 * e, h; depth = depth + 1)
        (v1 === nothing || v2 === nothing) && return v1
        return 2.0 * v1 - v2
    end
    m = locate(b, p[1], p[2], p[3])
    m == 0 && return nothing
    H = _width(b, m)
    c = _containing_cell(b, m, p)
    depth >= _MAX_DEPTH && return _cell_value(b, m, c)
    finer = map((Ha, ha) -> Ha < ha * (1.0 - _WIDTH_RTOL), H, h)
    coarser = map((Ha, ha) -> Ha > ha * (1.0 + _WIDTH_RTOL), H, h)
    if !any(finer) && !any(coarser)
        return _cell_value(b, m, c)
    elseif !any(coarser)
        return _restrict(b, box, p, h, finer, depth)
    elseif !any(finer)
        return _prolong(b, box, m, c, p, h, coarser, depth)
    end
    return _cell_value(b, m, c)
end

# Restriction: the mean of the children of the cell (h, p) along the `finer` axes.
function _restrict(b, box, p, h, finer, depth)
    hc = SVector(ntuple(a -> finer[a] ? h[a] / 2 : h[a], Val(3)))
    acc = zero(SVector{8,Float64})
    n = 0
    for o3 in (finer[3] ? (-1, 1) : (0,)), o2 in (finer[2] ? (-1, 1) : (0,)),
        o1 in (finer[1] ? (-1, 1) : (0,))
        q = p + SVector(o1 * h[1], o2 * h[2], o3 * h[3]) / 4
        v = mesh_value(b, box, q, hc; depth = depth + 1)
        v === nothing && continue
        acc += v
        n += 1
    end
    n == 0 && return nothing
    return acc / n
end

@inline _minmod(l, r) = l * r <= 0.0 ? 0.0 : (abs(l) < abs(r) ? l : r)

# Prolongation from cell `c` of block `m` to the centre `p` of a cell of widths `h` along
# the `coarser` axes. On a leaf-aligned mesh the offset of `p` in the coarse cell is
# exactly `(k + ½)/r − ½` in units of `H_a`, `r = H_a/h_a`; it is taken at that value, so
# every block that asks for the ghost at `p` gets the same bits.
function _prolong(b, box, m, c, p, h, coarser, depth)
    H = _width(b, m)
    u = _cell_value(b, m, c)
    xc = _left(b, m) + (c .- 1) .* H
    out = u
    for a in 1:3
        coarser[a] || continue
        r = round(H[a] / h[a])
        t = (round((p[a] - xc[a]) / H[a] * r + 0.5 * r - 0.5) + 0.5) / r - 0.5
        e = SVector(ntuple(k -> k == a ? H[k] : 0.0, Val(3)))
        vl = _neighbour_value(b, box, m, c, a, -1, xc - e, H, depth)
        vr = _neighbour_value(b, box, m, c, a, 1, xc + e, H, depth)
        slope = if vl === nothing && vr === nothing
            zero(u)
        elseif vl === nothing
            vr - u
        elseif vr === nothing
            u - vl
        else
            map(_minmod, u - vl, vr - u)
        end
        out += slope * t
    end
    return out
end

# The cell next to `c` along axis `a` on side `s`: read in block `m` when it is interior
# there, else the mesh value at width `H`.
function _neighbour_value(b, box, m, c, a, s, q, H, depth)
    ca = c[a] + s
    if 1 <= ca <= _cells(b)[a]
        return _cell_value(b, m, setindex(c, ca, a))
    end
    return mesh_value(b, box, q, H; depth = depth + 1)
end

"""
    fill_ghosts!(b::FluidBlocks, box::MeshBox, m) -> b

Write the ghost layer of block `m` of `b` (the cells with an index `0` or `n + 1` on some
axis of `b.data`) from [`mesh_value`](@ref) at each ghost centre `x_left + (i − 1) dx`
with the block's widths `dx`, rounded once to `Float32`; where `mesh_value` is `nothing`
(a hole in the mesh) the nearest interior cell of `m` is held. Reads only interior cells,
so the blocks may be filled concurrently.
"""
function fill_ghosts!(b::FluidBlocks, box::MeshBox, m::Integer)
    n = _cells(b)
    h = _width(b, m)
    xl = _left(b, m)
    data = b.data
    for k in 0:(n[3] + 1), j in 0:(n[2] + 1), i in 0:(n[1] + 1)
        (1 <= i <= n[1] && 1 <= j <= n[2] && 1 <= k <= n[3]) && continue
        p = xl + SVector(i - 1, j - 1, k - 1) .* h
        v = mesh_value(b, box, p, h)
        v === nothing && (v = _cell_value(b, m, SVector(clamp(i, 1, n[1]), clamp(j, 1, n[2]),
                                                          clamp(k, 1, n[3]))))
        @inbounds for f in 1:8
            data[f, i + 1, j + 1, k + 1, m] = Float32(v[f])
        end
    end
    return b
end
