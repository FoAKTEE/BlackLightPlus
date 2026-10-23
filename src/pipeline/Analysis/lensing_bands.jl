# Screen geometry of photon-ring imaging on the Bardeen screen (M = 1): the apparent
# horizon (inner edge of the direct image n = 0), the inner and outer edges of lensing
# band n ≥ 1, and per-band Cartesian grids with membership masks. Host only. Follows
# AART (MIT, © 2024 Cárdenas-Avendaño, Zhu & Lupsasca); see NOTICE. Physics: Gralla &
# Lupsasca PRD 101 044032 (P1), 044033 (P2); Cárdenas-Avendaño, Lupsasca & Zhu PRD 107
# 043030 (P3); "Pn Eq. m" cites these.

"""
    polygon_area(xs, ys) -> area
    polygon_area(poly::AbstractMatrix) -> area

Area enclosed by the closed polygon with vertices `(xs[k], ys[k])`, the last joined to
the first, by the shoelace formula `|Σₖ (xₖ yₖ₊₁ − xₖ₊₁ yₖ)| / 2`: the enclosed area of a
simple loop of either orientation, the absolute net signed area of a self-intersecting
one. The matrix form takes the vertices as rows `(x, y)`. Units are those of the
coordinates squared (M² on the Bardeen screen).
"""
function polygon_area(xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    s = zero(eltype(xs))
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        s += xs[i] * ys[j] - xs[j] * ys[i]
    end
    return abs(s) / 2
end
polygon_area(poly::AbstractMatrix) = polygon_area(view(poly, :, 1), view(poly, :, 2))

"""
    in_polygon(px, py, xs, ys) -> Bool

Whether the point `(px, py)` lies inside the closed polygon `(xs, ys)` (last vertex
joined to the first), by a nonzero winding number. The edge from vertex `k` to `k + 1`
adds `+1` when it crosses the line `y = py` upward (`ys[k] ≤ py < ys[k+1]`) with the
point strictly to its left, and `−1` when it crosses downward (`ys[k+1] ≤ py < ys[k]`)
with the point strictly to its right. The half-open rule decides points on the
horizontal line of a vertex; an edge through the point itself contributes nothing, so a
point on the boundary is inside or outside according to the remaining edges.
"""
function in_polygon(px, py, xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    wn = 0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        yi = ys[i]; yj = ys[j]
        if yi <= py
            if yj > py && ((xs[j] - xs[i]) * (py - yi) - (px - xs[i]) * (yj - yi)) > 0
                wn += 1
            end
        else
            if yj <= py && ((xs[j] - xs[i]) * (py - yi) - (px - xs[i]) * (yj - yi)) < 0
                wn -= 1
            end
        end
    end
    return wn != 0
end

# The edges (xᵢ, yᵢ, xⱼ, yⱼ) of the closed loop (xs, ys) that cross the line y = py under
# the half-open rule of `in_polygon`. Only these contribute to the winding number of a
# point on the line, so `_winding_on_row` over them equals `in_polygon` bit for bit.
function _row_edges!(buf::Vector{NTuple{4,Float64}}, xs::AbstractVector,
                     ys::AbstractVector, py)
    empty!(buf)
    n = length(xs)
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        yi = ys[i]; yj = ys[j]
        if (yi <= py) != (yj <= py)
            push!(buf, (xs[i], yi, xs[j], yj))
        end
    end
    return buf
end

# `in_polygon` of (px, py) over the crossing edges of its row: the same cross product
# and the same ±1 accumulation.
@inline function _winding_on_row(px, py, edges::Vector{NTuple{4,Float64}})
    wn = 0
    @inbounds for (xi, yi, xj, yj) in edges
        if yi <= py
            ((xj - xi) * (py - yi) - (px - xi) * (yj - yi)) > 0 && (wn += 1)
        else
            ((xj - xi) * (py - yi) - (px - xi) * (yj - yi)) < 0 && (wn -= 1)
        end
    end
    return wn != 0
end

# Root of `f` on [lo, hi]: the first of `nscan` equal steps from `lo` whose end values
# are finite with f(a) f(x) ≤ 0 brackets the root, then `iters` bisections, a
# non-finite midpoint value counting as beyond the root. NaN when no step brackets.
function _bisect(f, lo::T, hi::T; iters = 80, nscan = 64) where {T}
    flo = f(lo)
    a = lo; fa = flo
    b = hi
    found = false
    for i in 1:nscan
        x = lo + (hi - lo) * i / nscan
        fx = f(x)
        if isfinite(fx) && isfinite(fa) && fa * fx <= 0
            b = x; found = true; break
        end
        a = x; fa = fx
    end
    found || return T(NaN)
    for _ in 1:iters
        m = (a + b) / 2
        fm = f(m)
        if !isfinite(fm)
            b = m; continue
        end
        (fa * fm <= 0) ? (b = m) : (a = m; fa = fm)
    end
    return (a + b) / 2
end

# Mino time ∫_{rh}^{ro} dr/√R(r) (P1 Eq. 8a) with R the radial potential, by the
# substitution r = 1/x (dr = dx/x², which tames the 1/r² tail) and composite Simpson on
# N intervals; points where R ≤ 0 contribute zero.
function _mino_horizon_to_observer(a, λ, η, rh, ro; N = 4000)
    xlo = 1 / ro; xhi = 1 / rh
    h = (xhi - xlo) / N
    acc = 0.0
    @inbounds for i in 0:N
        x = xlo + i * h
        r = 1 / x
        Rr = kerr_radial_potential(r, a, λ, η)
        val = Rr > 0 ? (1 / (x * x)) / sqrt(Rr) : 0.0
        w = (i == 0 || i == N) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        acc += w * val
    end
    return acc * h / 3
end

# Residual of the AART apparent-horizon condition: zero when the ray at the screen point
# s (α0, β0) left the horizon, i.e. its radial Mino time from the horizon to the
# observer equals its angular Mino time with m turning points and sign sgn (P1 Eqs. 8a
# and 20). NaN for rays that never cross the equator.
function _apparent_horizon_residual(s, a, θo, α0, β0, m, sgn, ro)
    α = s * α0; β = s * β0
    λ, η = conserved_quantities(α, β, θo, a)
    uP, uM = angular_turning_points(λ, η, a)
    rP = kerr_horizon(a)
    (uP <= 0 || uM >= 0) && return NaN
    I = _mino_horizon_to_observer(a, λ, η, rP, ro)
    return sqrt(-uM * a^2) * I +
           sgn * JacobiElliptic.F(asin(clamp(cos(θo) / sqrt(uP), -1.0, 1.0)), uP / uM) -
           2 * m * JacobiElliptic.K(uP / uM)
end

"""
    apparent_horizon(a, θo; n = 100, ro = 1e5) -> (α::Vector, β::Vector)

The apparent horizon on the Bardeen screen (M = 1) for spin `a` and observer
inclination `θo` (rad): the image of the event horizon, the inner edge of the direct
image (lensing band n = 0). Each point of [`critical_curve`](@ref)`(a, θo; n)` is scaled
by the `s ∈ (0.05, 0.999)` at which the ray from the observer at radius `ro` reaches the
horizon exactly (AART `ApparentBH`; P1 Eqs. 8a, 20), with `m = 1` angular turning point
on the upper half (`β ≥ 0`) and `m = 0` on the lower; a point whose root is not
bracketed takes `s = 0.99`. For `θo > π/2` the curve is that of `π − θo` with `β → −β`.
Empty when the critical curve is.

# References
Gralla & Lupsasca (2020a), Eqs. 8a, 20; Cárdenas-Avendaño, Lupsasca & Zhu (2023).
"""
function apparent_horizon(a::Float64, θo::Float64; n::Int = 100, ro::Float64 = 1e5)
    if cos(θo) < 0
        αm, βm = apparent_horizon(a, π - θo; n = n, ro = ro)
        return (αm, -βm)
    end
    αc, βc = critical_curve(a, θo; n = n)
    isempty(αc) && return (Float64[], Float64[])
    αo = similar(αc); βo = similar(βc)
    for i in eachindex(αc)
        α0 = αc[i]; β0 = βc[i]
        m, sgn = β0 >= 0 ? (1.0, 1.0) : (0.0, -1.0)
        f = s -> _apparent_horizon_residual(s, a, θo, α0, β0, m, sgn, ro)
        s = _bisect(f, 0.05, 0.999)
        s = isnan(s) ? 0.99 : s
        αo[i] = s * α0; βo[i] = s * β0
    end
    return (αo, βo)
end

# Residual of the AART `nlayers` condition for an equatorial source (the source term
# vanishes at θs = π/2): radial Mino time minus the angular Mino time Gθ of a ray at the
# screen point s (α0, β0) with m = mbar + H(β0) turning points, H(0) = 1. `outer` selects
# the s > 1 branch (case II, P2 Eq. A10), else the s < 1 branch (case III, P2 Eq. A11,
# observer at infinity, inward to the horizon). NaN off the branch.
function _nlayers_residual(s, a, θo, α0, β0, mbar, outer::Bool)
    α = s * α0; β = s * β0
    m = mbar + (β0 >= 0 ? 1 : 0)
    λ, η = conserved_quantities(α, β, θo, a)
    η < 0 && return NaN
    νθ = β0 >= 0 ? 1.0 : -1.0
    r1, r2, r3, r4 = radial_roots(λ, η, a)
    uP, uM = angular_turning_points(λ, η, a)
    (uP <= 0 || uM >= 0) && return NaN
    kang = uP / uM
    Gθ = 1 / (sqrt(-uM) * a) *
         (2 * m * JacobiElliptic.K(kang) -
          νθ * JacobiElliptic.F(asin(clamp(cos(θo) / sqrt(uP), -1.0, 1.0)), kang))
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    if outer
        kk = real(r32 * r41 / (r31 * r42))
        Q1 = 4 / real(sqrt(r31 * r42)) *
             JacobiElliptic.F(asin(clamp(real(sqrt(r31 / r41)), -1.0, 1.0)), kk)
        return Q1 - Gθ
    else
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        k3 = ((A + B)^2 - real(r21)^2) / (4 * A * B)
        k3 >= 1 && return NaN
        rP = kerr_horizon(a)
        xc = clamp((A - B) / (A + B), -1.0, 1.0)
        xh = clamp(real((A * (rP - r1) - B * (rP - r2)) / (A * (rP - r1) + B * (rP - r2))), -1.0, 1.0)
        Q2 = 1 / sqrt(A * B) * (JacobiElliptic.F(acos(xc), k3) - JacobiElliptic.F(acos(xh), k3))
        return Q2 - Gθ
    end
end

# Inner (s < 1) and outer (s > 1) edge scales of band n ≥ 1 along the critical-curve
# point (α0, β0). The outer bracket [1.00001, 4] doubles up to [128, 256] when it holds
# no root; unbracketed edges fall back to 0.999 and 1.001, with a warning.
function _band_edge_scales(a, θo, α0, β0, n)
    s_in = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, false), 0.2, 0.99999)
    s_out = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, true), 1.00001, 4.0)
    lo = 4.0
    while isnan(s_out) && lo < 256.0
        s_out = _bisect(s -> _nlayers_residual(s, a, θo, α0, β0, n, true), lo, 2 * lo)
        lo *= 2
    end
    isnan(s_in) &&
        @warn "lensing_band: inner-edge root not bracketed; using fallback s=0.999" a θo n maxlog = 1
    isnan(s_out) &&
        @warn "lensing_band: outer-edge root not bracketed; using fallback s=1.001" a θo n maxlog = 1
    s_in = isnan(s_in) ? 0.999 : s_in
    s_out = isnan(s_out) ? 1.001 : s_out
    return s_in, s_out
end

"""
    lensing_band(a, θo, n; npts = 100, box = 25.0) -> (inner, outer)

Inner and outer edges of lensing band `n` on the Bardeen screen (M = 1) for spin `a`
and observer inclination `θo` (rad), each an `N × 2` matrix of vertices `(α, β)`. Band
`n = 0` (the direct image) is bounded inside by [`apparent_horizon`](@ref)`(a, θo; n =
npts)` and outside by the square of half-width `box`. A band `n ≥ 1` is a thin annulus
about the critical curve: each point of [`critical_curve`](@ref)`(a, θo; n = npts)` is
scaled inward to the case-III root and outward to the case-II root of the AART
`nlayers` condition (P2 Eqs. A10, A11; P3), found by bisection; an unbracketed root
falls back to `s = 0.999` (inner) or `1.001` (outer) with a warning. For `θo > π/2` the
edges are those of `π − θo` with `β → −β`, so `(α, β)` lies in band `n` at `θo` exactly
when `(α, −β)` does at `π − θo`.

# References
Gralla & Lupsasca (2020b), Eqs. A10, A11; Cárdenas-Avendaño, Lupsasca & Zhu (2023).
"""
function lensing_band(a::Float64, θo::Float64, n::Int; npts::Int = 100, box::Float64 = 25.0)
    if cos(θo) < 0
        inner, outer = lensing_band(a, π - θo, n; npts = npts, box = box)
        inner[:, 2] .= .-inner[:, 2]
        outer[:, 2] .= .-outer[:, 2]
        return inner, outer
    end
    αc, βc = critical_curve(a, θo; n = npts)
    if n == 0
        αi, βi = apparent_horizon(a, θo; n = npts)
        inner = hcat(αi, βi)
        outer = [ box  box; -box  box; -box -box;  box -box]
        return inner, outer
    end
    αi = similar(αc); βi = similar(βc); αo = similar(αc); βo = similar(βc)
    for i in eachindex(αc)
        α0 = αc[i]; β0 = βc[i]
        s_in, s_out = _band_edge_scales(a, θo, α0, β0, n)
        αi[i] = s_in * α0; βi[i] = s_in * β0
        αo[i] = s_out * α0; βo[i] = s_out * β0
    end
    return hcat(αi, βi), hcat(αo, βo)
end

"""
    band_grid(inner, outer; dx = 0.02, box = nothing) -> (grid, mask, N)

A Cartesian grid on the Bardeen screen and the mask of its points inside the band
between the polygons `inner` and `outer` (`N × 2` vertex matrices, as returned by
[`lensing_band`](@ref)); AART `grid_mask`. The axis is `range(-lim, lim; length = N)`
with `lim = box`, or `maximum(abs, outer) + 5dx` when `box` is `nothing`, and
`N = 2 round(Int, lim / dx)`. `grid` is the `N² × 2` matrix of points `(α, β)` with `α`
fastest. `mask[k]` holds when the point is inside `outer` and, if `inner` has at least
three vertices, not inside `inner`, both by [`in_polygon`](@ref). Returned as a
`NamedTuple`.
"""
function band_grid(inner::AbstractMatrix, outer::AbstractMatrix; dx::Float64 = 0.02, box = nothing)
    oxs = view(outer, :, 1); oys = view(outer, :, 2)
    lim = box === nothing ? maximum(abs, outer) + 5dx : box
    N = 2 * cld(round(Int, lim / dx), 1)
    xs = range(-lim, lim; length = N)
    ixs = view(inner, :, 1); iys = view(inner, :, 2)
    test_inner = size(inner, 1) >= 3
    npts = N * N
    grid = Matrix{Float64}(undef, npts, 2)
    mask = Vector{Bool}(undef, npts)
    oedges = NTuple{4,Float64}[]
    iedges = NTuple{4,Float64}[]
    idx = 0
    for yy in xs
        _row_edges!(oedges, oxs, oys, yy)
        test_inner && _row_edges!(iedges, ixs, iys, yy)
        for xx in xs
            idx += 1
            grid[idx, 1] = xx; grid[idx, 2] = yy
            inside_out = _winding_on_row(xx, yy, oedges)
            mask[idx] = inside_out && !(test_inner && _winding_on_row(xx, yy, iedges))
        end
    end
    return (grid = grid, mask = mask, N = N)
end
