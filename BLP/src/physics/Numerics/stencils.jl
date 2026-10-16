# Interpolation stencils: the 8-corner trilinear blend, the Catmull-Rom (C¹ cubic)
# tensor stencil of the gridded metric, and the log-linear table lookup.

"""
    primal(x) -> Real

The value of `x` with every (nested) `ForwardDiff.Dual` layer removed; the
identity on plain reals. For derivative-free decisions only (stencil indices,
domain tests), never for the differentiated value.
"""
@inline primal(x::Real) = x
@inline primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))

"""
    corner_weights(f, n) -> (i0, d)

Lower-corner index and blend fraction of the clamped trilinear stencil along one
axis: `f` is the 0-based fractional cell coordinate, `n` the number of nodes.
`i0 = clamp(⌊f⌋, 0, n - 2)` (so the upper corner `i0 + 1` exists) and
`d = clamp(f - i0, 0, 1)` (a one-sided blend outside the node range). The
1-based corner indices are `i0 + 1` and `i0 + 2`.
"""
@inline function corner_weights(f, n::Int)
    i0 = clamp(unsafe_trunc(Int, floor(f)), 0, n - 2)
    return i0, clamp(f - i0, zero(f), one(f))
end

"""
    lerp8(c, d1, d2, d3)

Trilinear blend of the eight corner values
`c = (c000, c100, c010, c110, c001, c101, c011, c111)` (digit k is the corner's
offset along axis k) with fractions `d1, d2, d3` along axes 1, 2, 3. The corner
values may be scalars or vectors. Association order, which fixes the bits:

    c00 = c000 + (c100 - c000) * d1
    c10 = c010 + (c110 - c010) * d1
    c01 = c001 + (c101 - c001) * d1
    c11 = c011 + (c111 - c011) * d1
    c0  = c00 + (c10 - c00) * d2
    c1  = c01 + (c11 - c01) * d2
    return c0 + (c1 - c0) * d3
"""
@inline function lerp8(c::NTuple{8,V}, d1, d2, d3) where {V}
    c000, c100, c010, c110, c001, c101, c011, c111 = c
    c00 = c000 + (c100 - c000) * d1
    c10 = c010 + (c110 - c010) * d1
    c01 = c001 + (c101 - c001) * d1
    c11 = c011 + (c111 - c011) * d1
    c0  = c00 + (c10 - c00) * d2
    c1  = c01 + (c11 - c01) * d2
    return c0 + (c1 - c0) * d3
end

"""
    cr_weights(t) -> SVector{4}

Catmull-Rom weights of the four nodes bracketing the fractional position
`t ∈ [0, 1]` between the middle two: tangents are centred differences, so the
interpolant is C¹ across every node and exact for quadratics. The weights sum to
1 and take the type of `t` (a `Dual` under AD).
"""
@inline function cr_weights(t)
    t2 = t * t
    t3 = t2 * t
    return SVector((-t3 + 2t2 - t) / 2,
                   (3t3 - 5t2 + 2) / 2,
                   (-3t3 + 4t2 + t) / 2,
                   (t3 - t2) / 2)
end

"""
    cr_axis(c, c0, dc, n) -> (w::SVector{4}, idx::NTuple{4,Int})

Catmull-Rom stencil along one axis of a uniform grid with first node `c0`,
spacing `dc` and `n` nodes, at coordinate `c`: the weights `cr_weights(t)` and
the four 1-based node indices, clamped to `1:n`. The lower node is chosen from
the derivative-free `primal` of the continuous index, so a `Dual` coordinate
differentiates through the weights only.
"""
@inline function cr_axis(c, c0, dc, n::Int)
    u = (c - c0) / dc
    iu = floor(Int, primal(u))
    t = u - iu
    w = cr_weights(t)
    idx = (clamp(iu, 1, n), clamp(iu + 1, 1, n), clamp(iu + 2, 1, n), clamp(iu + 3, 1, n))
    return w, idx
end

"""
    cr_sample(d, comp, ix, iy, iz, wx, wy, wz)

Tensor-product Catmull-Rom interpolation of field `comp` of the
component-fastest array `d[comp, i, j, k]` over the 4×4×4 stencil with node
indices `ix, iy, iz` and weights `wx, wy, wz` from [`cr_axis`](@ref). The sums
run x innermost, then y, then z, each from a zero of the weight type.
"""
@inline function cr_sample(d, comp::Int, ix, iy, iz, wx, wy, wz)
    T = eltype(wx)
    s = zero(T)
    @inbounds for c in 1:4
        sc = zero(T)
        for b in 1:4
            sb = zero(T)
            for a in 1:4
                sb += wx[a] * d[comp, ix[a], iy[b], iz[c]]
            end
            sc += wy[b] * sb
        end
        s += wz[c] * sc
    end
    return s
end

"""
    loglinear_at(x, table, x0, dx, n) -> value

Piecewise log-linear lookup: `table[k]` holds `log f` at `ln x = x0 + (k-1) dx`,
`k = 1:n`; returns `f(x) = exp` of the linear interpolant in `ln x`, clamped to
the first and last nodes. `table` may be any indexable (host or device array).
"""
@inline function loglinear_at(x, table, x0, dx, n::Int)
    di = (log(x) - x0) / dx
    i = unsafe_trunc(Int, floor(di))
    if i < 0
        i = 0
        di = zero(di)
    elseif i > n - 2
        i = n - 2
        di = oftype(di, n - 1)
    end
    f = di - i
    @inbounds return exp((one(f) - f) * table[i + 1] + f * table[i + 2])
end
