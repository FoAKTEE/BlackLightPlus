# A frozen-slice numerical spacetime: lapse α, shift β^i and spatial metric γ_ij on a
# regular Cartesian grid, assembled into g_μν. The metric derivatives come from AD
# through the interpolant, which therefore must be C¹: the tensor-product Catmull-Rom
# cubic (trilinear interpolation has a discontinuous derivative at every cell face).
# One struct serves host and device: `data` is any 4-d array (a device array after
# `Adapt.adapt`), and every method below reads it through the same body.

"""
    ADMGrid{T,A<:AbstractArray{T,4}} <: StationarySpacetime{T}
    ADMGrid(xs, ys, zs, data; alpha_min = 0.1, offgrid_eta = true)
    ADMGrid(st::ADMGrid; alpha_min = st.alpha_min)

Frozen-slice ADM data on a regular Cartesian grid. `data[c, i, j, k]` holds the
ten fields per node in the order `(α, β^x, β^y, β^z, γ_xx, γ_xy, γ_xz, γ_yy,
γ_yz, γ_zz)`; the node coordinates are `x0 + (i-1) dx` (likewise `y`, `z`), with
`nx, ny, nz` nodes per axis. The metric is assembled from the Catmull-Rom
interpolants: `g_00 = -α² + β_i β^i`, `g_0i = β_i = γ_ij β^j`, `g_ij = γ_ij`.
Off the grid the metric is Minkowski when `offgrid_eta`, else the clamped
extrapolation. The march captures where the interpolated lapse drops below
`alpha_min`. `Adapt.adapt(backend, st)` moves `data` to a device; the second
constructor copies the grid with another capture lapse, sharing `data`. The
scalars share the type of the data, so `lane(st, T)` returns `st`.
"""
struct ADMGrid{T,A<:AbstractArray{T,4}} <: StationarySpacetime{T}
    x0::T; y0::T; z0::T
    dx::T; dy::T; dz::T
    nx::Int; ny::Int; nz::Int
    data::A
    alpha_min::T
    offgrid_eta::Bool
end
Adapt.@adapt_structure ADMGrid

function ADMGrid(xs::AbstractRange, ys::AbstractRange, zs::AbstractRange,
                 data::Array{Float64,4}; alpha_min::Real = 0.1, offgrid_eta::Bool = true)
    @assert size(data) == (10, length(xs), length(ys), length(zs)) "data must be (10, nx, ny, nz)"
    return ADMGrid{Float64,Array{Float64,4}}(first(xs), first(ys), first(zs),
                                             step(xs), step(ys), step(zs),
                                             length(xs), length(ys), length(zs),
                                             data, Float64(alpha_min), offgrid_eta)
end

ADMGrid(st::ADMGrid{T}; alpha_min::Real = st.alpha_min) where {T} =
    ADMGrid(st.x0, st.y0, st.z0, st.dx, st.dy, st.dz, st.nx, st.ny, st.nz, st.data,
            T(alpha_min), st.offgrid_eta)

# The grid's scalars are tied to its data type, so the lane of an ADM grid is the grid.
lane(st::ADMGrid, ::Type) = st

@inline horizon_radius(st::ADMGrid) = st.alpha_min

# Floor of the extracted α² = β^i β_i - g_00, which turns negative inside a horizon.
const _ADM_ALPHA2_FLOOR = 1.0e-16

"""
    sample_adm(st::ADMGrid, x, y, z) -> (α, β::SVector{3}, γ::SMatrix{3,3})

The Catmull-Rom interpolated lapse, contravariant shift `β^i` and spatial
metric `γ_ij` at Cartesian `(x, y, z)`; the coordinates may be `Dual`s (the
stencil follows their primal values, the weights carry the partials).
"""
@inline function sample_adm(st::ADMGrid, x, y, z)
    wx, ix = cr_axis(x, st.x0, st.dx, st.nx)
    wy, iy = cr_axis(y, st.y0, st.dy, st.ny)
    wz, iz = cr_axis(z, st.z0, st.dz, st.nz)
    d = st.data
    α   = cr_sample(d, 1,  ix, iy, iz, wx, wy, wz)
    βx  = cr_sample(d, 2,  ix, iy, iz, wx, wy, wz)
    βy  = cr_sample(d, 3,  ix, iy, iz, wx, wy, wz)
    βz  = cr_sample(d, 4,  ix, iy, iz, wx, wy, wz)
    gxx = cr_sample(d, 5,  ix, iy, iz, wx, wy, wz)
    gxy = cr_sample(d, 6,  ix, iy, iz, wx, wy, wz)
    gxz = cr_sample(d, 7,  ix, iy, iz, wx, wy, wz)
    gyy = cr_sample(d, 8,  ix, iy, iz, wx, wy, wz)
    gyz = cr_sample(d, 9,  ix, iy, iz, wx, wy, wz)
    gzz = cr_sample(d, 10, ix, iy, iz, wx, wy, wz)
    γ = SMatrix{3,3}(gxx, gxy, gxz, gxy, gyy, gyz, gxz, gyz, gzz)
    return α, SVector(βx, βy, βz), γ
end

"""
    on_grid(st::ADMGrid, x::SVector{4}) -> Bool

Whether the spatial position of `x` lies inside the node range on every axis
(tested on the primal coordinates).
"""
@inline function on_grid(st::ADMGrid, X::SVector{4})
    _inaxis(primal(X[2]), st.x0, st.dx, st.nx) &&
    _inaxis(primal(X[3]), st.y0, st.dy, st.ny) &&
    _inaxis(primal(X[4]), st.z0, st.dz, st.nz)
end
@inline _inaxis(c, c0, dc, n::Int) = (u = (c - c0) / dc; 0.0 <= u <= float(n - 1))

# g_00 = -α² + β_i β^i, g_0i = β_i = γ_ij β^j, g_ij = γ_ij.
@inline function _adm_lower(α, β, γ)
    βd = γ * β
    b2 = dot(βd, β)
    g00 = -α * α + b2
    gxx = γ[1,1]; gxy = γ[1,2]; gxz = γ[1,3]
    gyy = γ[2,2]; gyz = γ[2,3]; gzz = γ[3,3]
    return SMatrix{4,4}(g00,   βd[1], βd[2], βd[3],        # symmetric: column = row
                        βd[1], gxx,   gxy,   gxz,
                        βd[2], gxy,   gyy,   gyz,
                        βd[3], gxz,   gyz,   gzz)
end

# The 3+1 inverse g^00 = -1/α², g^0i = β^i/α², g^ij = γ^ij - β^i β^j/α²: a 3×3 inverse
# of γ in place of a 4×4 inverse of g.
@inline function _adm_raise(α, β, γ)
    γi = inv(γ)
    ia2 = 1 / (α * α)
    b1 = β[1] * ia2; b2 = β[2] * ia2; b3 = β[3] * ia2
    gxx = γi[1,1] - β[1] * b1; gxy = γi[1,2] - β[1] * b2; gxz = γi[1,3] - β[1] * b3
    gyy = γi[2,2] - β[2] * b2; gyz = γi[2,3] - β[2] * b3; gzz = γi[3,3] - β[3] * b3
    return SMatrix{4,4}(-ia2, b1,  b2,  b3,
                        b1,   gxx, gxy, gxz,
                        b2,   gxy, gyy, gyz,
                        b3,   gxz, gyz, gzz)
end

# The metric methods are thin wrappers over the bodies below (the wrapper shape is part
# of how the inlined arithmetic compiles, so it is kept for every method).
@inline function _adm_gcov(st, X::SVector{4,T}) where {T}
    if !on_grid(st, X)
        st.offgrid_eta && return SMatrix{4,4,T,16}(ETA)
    end
    return _adm_lower(sample_adm(st, X[2], X[3], X[4])...)
end

@inline function _adm_gcon(st, X::SVector{4,T}) where {T}
    if !on_grid(st, X)
        st.offgrid_eta && return SMatrix{4,4,T,16}(ETA)
    end
    return _adm_raise(sample_adm(st, X[2], X[3], X[4])...)
end

@inline function _adm_pair(st, X::SVector{4,T}) where {T}
    if !on_grid(st, X)
        st.offgrid_eta && return SMatrix{4,4,T,16}(ETA), SMatrix{4,4,T,16}(ETA)
    end
    α, β, γ = sample_adm(st, X[2], X[3], X[4])
    return _adm_lower(α, β, γ), _adm_raise(α, β, γ)
end

# (g v)_0 = (β_iβ^i - α²) v^0 + β_i v^i, (g v)_i = β_i v^0 + γ_ij v^j;
# (g⁻¹p)^0 = -q/α², (g⁻¹p)^i = γ^ij p_j + β^i q/α², q = p_0 - β^i p_i.
@inline function _adm_lower_index(st, X::SVector{4,T}, v::SVector{4}) where {T}
    R = promote_type(T, eltype(v), eltype(st.data))
    if !on_grid(st, X)
        st.offgrid_eta && return SVector{4,R}(_eta_apply(v))
    end
    α, β, γ = sample_adm(st, X[2], X[3], X[4])
    βd = γ * β
    vs = SVector(v[2], v[3], v[4])
    gs = βd * v[1] + γ * vs
    return SVector{4,R}((dot(βd, β) - α * α) * v[1] + dot(βd, vs), gs[1], gs[2], gs[3])
end
@inline function _adm_raise_index(st, X::SVector{4,T}, p::SVector{4}) where {T}
    R = promote_type(T, eltype(p), eltype(st.data))
    if !on_grid(st, X)
        st.offgrid_eta && return SVector{4,R}(_eta_apply(p))
    end
    α, β, γ = sample_adm(st, X[2], X[3], X[4])
    ps = SVector(p[2], p[3], p[4])
    qa = (p[1] - dot(β, ps)) / (α * α)
    gs = inv(γ) * ps + β * qa
    return SVector{4,R}(-qa, gs[1], gs[2], gs[3])
end
# (g⁻¹p, g⁻¹(p, p)) with g⁻¹(p, p) = γ^ij p_i p_j - q²/α².
@inline function _adm_raise_and_norm(st, X::SVector{4,T}, p::SVector{4}) where {T}
    R = promote_type(T, eltype(p), eltype(st.data))
    if !on_grid(st, X)
        st.offgrid_eta && (ηp = _eta_apply(p); return SVector{4,R}(ηp), R(dot(p, ηp)))
    end
    α, β, γ = sample_adm(st, X[2], X[3], X[4])
    ps = SVector(p[2], p[3], p[4])
    q = p[1] - dot(β, ps)
    qa = q / (α * α)
    gps = inv(γ) * ps
    gs = gps + β * qa
    return SVector{4,R}(-qa, gs[1], gs[2], gs[3]), R(dot(ps, gps) - q * qa)
end

@inline raise_and_norm(st::ADMGrid, x::SVector{4}, p::SVector{4}) = _adm_raise_and_norm(st, x, p)
@inline lower_index(st::ADMGrid, x::SVector{4}, v::SVector{4}) = _adm_lower_index(st, x, v)
@inline raise_index(st::ADMGrid, x::SVector{4}, p::SVector{4}) = _adm_raise_index(st, x, p)
@inline metric(st::ADMGrid, X::SVector{4}) = _adm_gcov(st, X)
@inline inverse_metric(st::ADMGrid, x::SVector{4}) = _adm_gcon(st, x)
@inline metric_and_inverse(st::ADMGrid, x::SVector{4}) = _adm_pair(st, x)

# (α, β^i, γ_ij) of a covariant 4-metric, α² floored at _ADM_ALPHA2_FLOOR.
@inline function _adm_from_gcov(g::SMatrix{4,4})
    sp = SVector(2, 3, 4)
    γ = g[sp, sp]
    βd = SVector(g[1,2], g[1,3], g[1,4])                   # β_i = g_0i
    βu = inv(γ) * βd                                       # β^i = γ^ij β_j
    α2 = dot(βu, βd) - g[1,1]                              # α² = β^i β_i - g_00
    return sqrt(max(α2, _ADM_ALPHA2_FLOOR)), βu, γ
end

"""
    adm_grid_from_metric(metric_fn, xs, ys, zs; alpha_min = 0.1, offgrid_eta = true,
                         excise = nothing) -> ADMGrid

Sample a covariant metric closure `metric_fn(x::SVector{4}) -> SMatrix{4,4}` on
the `t = 0` slice at the nodes of the ranges `xs, ys, zs` and store its ADM
data. Nodes where `excise(x)` is true, or where the metric is non-finite, take
the collapsed-lapse fill `α = alpha_min/2`, `β = 0`, `γ = I`: invertible, and
below the capture lapse, so a ray falling into the excised set stops on the
lapse. Pass the horizon predicate (e.g.
`x -> ks_radius(st, x) < horizon_radius(st)`) for an inner stopping surface
independent of the grid alignment. Serial; the values do not depend on the
traversal order.
"""
function adm_grid_from_metric(metric_fn, xs::AbstractRange, ys::AbstractRange,
                              zs::AbstractRange; alpha_min::Real = 0.1, offgrid_eta::Bool = true,
                              excise = nothing)
    nx, ny, nz = length(xs), length(ys), length(zs)
    data = Array{Float64,4}(undef, 10, nx, ny, nz)
    α_fill = alpha_min > 0 ? Float64(alpha_min) / 2 : 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        x = SVector(0.0, Float64(xs[i]), Float64(ys[j]), Float64(zs[k]))
        if excise === nothing || !excise(x)
            g = metric_fn(x)
            α, βu, γ = _adm_from_gcov(g)
            if isfinite(α) && all(isfinite, βu) && all(isfinite, γ)
                data[1, i, j, k] = α
                data[2, i, j, k] = βu[1]; data[3, i, j, k] = βu[2]; data[4, i, j, k] = βu[3]
                data[5, i, j, k] = γ[1,1]; data[6, i, j, k] = γ[1,2]; data[7, i, j, k] = γ[1,3]
                data[8, i, j, k] = γ[2,2]; data[9, i, j, k] = γ[2,3]; data[10, i, j, k] = γ[3,3]
                continue
            end
        end
        data[1, i, j, k] = α_fill
        data[2, i, j, k] = 0.0; data[3, i, j, k] = 0.0; data[4, i, j, k] = 0.0
        data[5, i, j, k] = 1.0; data[6, i, j, k] = 0.0; data[7, i, j, k] = 0.0
        data[8, i, j, k] = 1.0; data[9, i, j, k] = 0.0; data[10, i, j, k] = 1.0
    end
    return ADMGrid(xs, ys, zs, data; alpha_min, offgrid_eta)
end
