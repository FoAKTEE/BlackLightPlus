# Interpolation stencils: the 8-corner trilinear blend, the Catmull-Rom (C¹ cubic)
# tensor stencil of the gridded metric, the C¹ cubic Hermite in time, and the log-linear
# table lookup.

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
    hermite_weights(ts, t) -> (w::SVector{4}, idx::NTuple{4,Int})

Weights and node indices of the C¹ cubic Hermite through values at the strictly
increasing nodes `ts` (at least two), `f(t) ≈ Σ_m w[m] f[idx[m]]`, whose tangent at node
`k` is the derivative of the parabola through nodes `k - 1, k, k + 1` (`1, 2, 3` at the
first node, `n - 2, n - 1, n` at the last, the chord when `n = 2`). The tangents are
single-valued at every node, so the interpolant and its derivative are continuous; it is
exact for quadratics, with errors `O(Δt³)` in the value and `O(Δt²)` in the derivative,
and on uniform nodes the interior intervals are Catmull–Rom ([`cr_weights`](@ref)). The
interval follows the `primal` of `t`, so a `Dual` `t` differentiates through the weights;
outside `[ts[1], ts[end]]` (and for a NaN `t`, the first node) the weights select the end
node, a constant.
"""
@inline function hermite_weights(ts, t)
    n = length(ts)
    z = zero(t); o = one(t)
    tp = primal(t)
    !(tp > ts[1]) && return SVector(o, z, z, z), (1, 1, 1, 1)
    !(tp < ts[n]) && return SVector(o, z, z, z), (n, n, n, n)
    k = clamp(searchsortedlast(ts, tp), 1, n - 1)
    lo = n < 4 ? 1 : clamp(k - 1, 1, n - 3)
    h = ts[k + 1] - ts[k]
    s = (t - ts[k]) / h
    s2 = s * s; s3 = s2 * s
    a0 = 2s3 - 3s2 + 1; a1 = 3s2 - 2s3                       # values at k, k + 1
    b0 = (s3 - 2s2 + s) * h; b1 = (s3 - s2) * h              # tangents at k, k + 1
    q0, c = _parabola_tangent(ts, n, k)
    r0, e = _parabola_tangent(ts, n, k + 1)
    wj(j) = (j == k ? a0 : z) + (j == k + 1 ? a1 : z) + b0 * _tangent_coef(c, j - q0) +
            b1 * _tangent_coef(e, j - r0)
    return SVector(wj(lo), wj(lo + 1), wj(lo + 2), wj(lo + 3)),
           (lo, min(lo + 1, n), min(lo + 2, n), min(lo + 3, n))
end

# The tangent at node j as (q0, (c0, c1, c2)): f'(t_j) ≈ c0 f[q0] + c1 f[q0 + 1] + c2 f[q0 + 2].
@inline function _parabola_tangent(ts, n::Int, j::Int)
    if n == 2
        h = ts[2] - ts[1]
        return 1, (-1 / h, 1 / h, zero(h))
    elseif j == 1
        h1 = ts[2] - ts[1]; h2 = ts[3] - ts[2]
        return 1, (-(2h1 + h2) / (h1 * (h1 + h2)), (h1 + h2) / (h1 * h2), -h1 / (h2 * (h1 + h2)))
    elseif j == n
        h1 = ts[n - 1] - ts[n - 2]; h2 = ts[n] - ts[n - 1]
        return n - 2, (h2 / (h1 * (h1 + h2)), -(h1 + h2) / (h1 * h2), (2h2 + h1) / (h2 * (h1 + h2)))
    end
    hl = ts[j] - ts[j - 1]; hr = ts[j + 1] - ts[j]
    return j - 1, (-hr / (hl * (hl + hr)), (hr - hl) / (hl * hr), hl / (hr * (hl + hr)))
end
@inline _tangent_coef(c, i::Int) = 0 <= i <= 2 ? c[i + 1] : zero(c[1])

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
