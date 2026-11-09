# A numerical spacetime in time: ADM slices (α, β^i, γ_ij) at coordinate times
# t_1 < … < t_N on one Cartesian lattice. In space each slice is the tensor-product
# Catmull-Rom cubic of `ADMGrid`; in time the fields at a fixed point follow the cubic
# Hermite through the slices with three-point tangents (`hermite_weights`). Both are C¹,
# so the AD force, its time column ∂_t g included, is continuous, as the fixed-step march
# needs. One slice is no series: the builders return the frozen `ADMGrid` itself.

"""
    ADMSeries{T,A<:AbstractArray{T,5},V<:AbstractVector{T}} <: Spacetime{T}
    ADMSeries(slices::AbstractVector{<:ADMGrid}, times; alpha_min = slices[1].alpha_min)
    ADMSeries(xs, ys, zs, times, data; alpha_min = 0.1, offgrid_eta = true)

ADM slices at the strictly increasing coordinate times `times` on one lattice:
`data[c, i, j, k, n]` holds the ten fields `(α, β^x, β^y, β^z, γ_xx, γ_xy, γ_xz, γ_yy,
γ_yz, γ_zz)` of slice `n` at node `(x0 + (i-1) dx, y0 + (j-1) dy, z0 + (k-1) dz)`. At
`(t, x)` each field is the Catmull-Rom interpolant of [`ADMGrid`](@ref) in space and the
cubic Hermite of [`hermite_weights`](@ref) in time (Catmull-Rom on uniform times): C¹ in
`(t, x)`, exact for quadratics, errors `O(Δt³ + Δx³)` in the metric and `O(Δt² + Δx²)` in
its first derivatives. Before the first slice and after the last the metric is the end
slice (`∂_t g = 0` there; `∂_t g` jumps at the end slices). The 4-metric is assembled from
the interpolated fields as on `ADMGrid`; off the lattice it is Minkowski when
`offgrid_eta`; the march captures below the lapse `alpha_min`. The geodesic flow keeps
`∂_t g`. A series has at least two slices; [`adm_series_from_metric`](@ref) and
`Data.z4c_series_spacetime` return the frozen `ADMGrid` for one, so every product through
it is the frozen slice's bit for bit. The first constructor stacks frozen slices on one
lattice (same nodes and off-grid rule); the second takes the ranges of the nodes and the
`(10, nx, ny, nz, N)` data of any ADM source.
"""
struct ADMSeries{T,A<:AbstractArray{T,5},V<:AbstractVector{T}} <: Spacetime{T}
    x0::T; y0::T; z0::T
    dx::T; dy::T; dz::T
    nx::Int; ny::Int; nz::Int
    times::V
    data::A
    alpha_min::T
    offgrid_eta::Bool
end
Adapt.@adapt_structure ADMSeries

function ADMSeries(xs::AbstractRange, ys::AbstractRange, zs::AbstractRange,
                   times::AbstractVector{<:Real}, data::Array{Float64,5};
                   alpha_min::Real = 0.1, offgrid_eta::Bool = true)
    N = length(times)
    size(data) == (10, length(xs), length(ys), length(zs), N) ||
        throw(ArgumentError("ADMSeries: data must be (10, nx, ny, nz, length(times))"))
    _check_times(times)
    return ADMSeries{Float64,Array{Float64,5},Vector{Float64}}(
        first(xs), first(ys), first(zs), step(xs), step(ys), step(zs),
        length(xs), length(ys), length(zs), collect(Float64, times), data,
        Float64(alpha_min), offgrid_eta)
end

function ADMSeries(slices::AbstractVector{<:ADMGrid{Float64}}, times::AbstractVector{<:Real};
                   alpha_min::Real = first(slices).alpha_min)
    length(slices) == length(times) ||
        throw(ArgumentError("ADMSeries: one time per slice"))
    g = first(slices)
    lattice(s) = (s.x0, s.y0, s.z0, s.dx, s.dy, s.dz, s.nx, s.ny, s.nz, s.offgrid_eta)
    all(s -> lattice(s) == lattice(g), slices) ||
        throw(ArgumentError("ADMSeries: the slices must share one lattice and off-grid rule"))
    N = length(times)
    _check_times(times)
    data = Array{Float64,5}(undef, 10, g.nx, g.ny, g.nz, N)
    for (n, s) in enumerate(slices)
        data[:, :, :, :, n] .= s.data
    end
    return ADMSeries{Float64,Array{Float64,5},Vector{Float64}}(
        g.x0, g.y0, g.z0, g.dx, g.dy, g.dz, g.nx, g.ny, g.nz, collect(Float64, times), data,
        Float64(alpha_min), g.offgrid_eta)
end

function _check_times(times)
    N = length(times)
    N >= 2 || throw(ArgumentError("ADMSeries: at least two slices; one slice is its ADMGrid"))
    (all(isfinite, times) && all(k -> times[k + 1] > times[k], 1:N - 1)) ||
        throw(ArgumentError("ADMSeries: the slice times must be finite and strictly increasing"))
    return nothing
end

# The lattice's scalars are tied to its data type, as on `ADMGrid`.
lane(st::ADMSeries, ::Type) = st

@inline horizon_radius(st::ADMSeries) = st.alpha_min

"""
    adm_slice(st::ADMSeries, n = 1) -> ADMGrid

Slice `n` of the series as the frozen `ADMGrid` (a view of its data, the capture lapse
and the off-grid rule of the series).
"""
@inline adm_slice(st::ADMSeries, n::Integer = 1) =
    ADMGrid(st.x0, st.y0, st.z0, st.dx, st.dy, st.dz, st.nx, st.ny, st.nz,
            view(st.data, :, :, :, :, Int(n)), st.alpha_min, st.offgrid_eta)

"""
    sample_adm(st::ADMSeries, t, x, y, z) -> (α, β::SVector{3}, γ::SMatrix{3,3})

The lapse, shift and spatial metric of the series at `(t, x, y, z)`: per slice the
Catmull-Rom sample of [`ADMGrid`](@ref), combined by the time weights of
[`hermite_weights`](@ref); every coordinate may be a `Dual`.
"""
@inline function sample_adm(st::ADMSeries, t, x, y, z)
    wx, ix = cr_axis(x, st.x0, st.dx, st.nx)
    wy, iy = cr_axis(y, st.y0, st.dy, st.ny)
    wz, iz = cr_axis(z, st.z0, st.dz, st.nz)
    wt, it = hermite_weights(st.times, t)
    f(c) = _time_sample(st.data, c, ix, iy, iz, it, wx, wy, wz, wt)
    α = f(1)
    γ = SMatrix{3,3}(f(5), f(6), f(7), f(6), f(8), f(9), f(7), f(9), f(10))
    return α, SVector(f(2), f(3), f(4)), γ
end

# Σ_m w_t[m] (Catmull-Rom sample of field c on slice it[m]).
@inline function _time_sample(d, c::Int, ix, iy, iz, it, wx, wy, wz, wt)
    s = wt[1] * cr_sample(view(d, :, :, :, :, it[1]), c, ix, iy, iz, wx, wy, wz)
    @inbounds for m in 2:4
        s += wt[m] * cr_sample(view(d, :, :, :, :, it[m]), c, ix, iy, iz, wx, wy, wz)
    end
    return s
end

"""
    on_grid(st::ADMSeries, x::SVector{4}) -> Bool

Whether the spatial position of `x` lies inside the node range of the lattice.
"""
@inline on_grid(st::ADMSeries, X::SVector{4}) =
    _inaxis(primal(X[2]), st.x0, st.dx, st.nx) &&
    _inaxis(primal(X[3]), st.y0, st.dy, st.ny) &&
    _inaxis(primal(X[4]), st.z0, st.dz, st.nz)

@inline _series_fields(st::ADMSeries, X::SVector{4}) = sample_adm(st, X[1], X[2], X[3], X[4])

@inline function metric(st::ADMSeries, X::SVector{4,T}) where {T}
    on_grid(st, X) || (st.offgrid_eta && return SMatrix{4,4,T,16}(ETA))
    return _adm_lower(_series_fields(st, X)...)
end

@inline function inverse_metric(st::ADMSeries, X::SVector{4,T}) where {T}
    on_grid(st, X) || (st.offgrid_eta && return SMatrix{4,4,T,16}(ETA))
    return _adm_raise(_series_fields(st, X)...)
end

@inline function metric_and_inverse(st::ADMSeries, X::SVector{4,T}) where {T}
    on_grid(st, X) || (st.offgrid_eta && return SMatrix{4,4,T,16}(ETA), SMatrix{4,4,T,16}(ETA))
    α, β, γ = _series_fields(st, X)
    return _adm_lower(α, β, γ), _adm_raise(α, β, γ)
end

# The 3+1 contractions of `ADMGrid` at the series' fields:
# (g v)_0 = (β_iβ^i - α²) v^0 + β_i v^i, (g v)_i = β_i v^0 + γ_ij v^j;
# (g⁻¹p)^0 = -q/α², (g⁻¹p)^i = γ^ij p_j + β^i q/α², g⁻¹(p, p) = γ^ij p_i p_j - q²/α²,
# q = p_0 - β^i p_i.
@inline function lower_index(st::ADMSeries, X::SVector{4,T}, v::SVector{4}) where {T}
    R = promote_type(T, eltype(v), eltype(st.data))
    on_grid(st, X) || (st.offgrid_eta && return SVector{4,R}(_eta_apply(v)))
    α, β, γ = _series_fields(st, X)
    βd = γ * β
    vs = SVector(v[2], v[3], v[4])
    gs = βd * v[1] + γ * vs
    return SVector{4,R}((dot(βd, β) - α * α) * v[1] + dot(βd, vs), gs[1], gs[2], gs[3])
end

@inline raise_index(st::ADMSeries, X::SVector{4}, p::SVector{4}) = raise_and_norm(st, X, p)[1]

@inline function raise_and_norm(st::ADMSeries, X::SVector{4,T}, p::SVector{4}) where {T}
    R = promote_type(T, eltype(p), eltype(st.data))
    if !on_grid(st, X)
        st.offgrid_eta && (ηp = _eta_apply(p); return SVector{4,R}(ηp), R(dot(p, ηp)))
    end
    α, β, γ = _series_fields(st, X)
    ps = SVector(p[2], p[3], p[4])
    q = p[1] - dot(β, ps)
    qa = q / (α * α)
    gps = inv(γ) * ps
    gs = gps + β * qa
    return SVector{4,R}(-qa, gs[1], gs[2], gs[3]), R(dot(ps, gps) - q * qa)
end

"""
    adm_series_from_metric(metric_fn, xs, ys, zs, times; alpha_min = 0.1,
                           offgrid_eta = true, excise = nothing) -> ADMSeries | ADMGrid

The ADM slices of a covariant metric closure `metric_fn(x::SVector{4})` at the times
`times`, each sampled at the nodes of `xs, ys, zs` as by [`adm_grid_from_metric`](@ref)
on the slice `x⁰ = times[n]` (`excise` sees the event with its time); one time gives that
slice's frozen `ADMGrid`.
"""
function adm_series_from_metric(metric_fn, xs::AbstractRange, ys::AbstractRange,
                                zs::AbstractRange, times::AbstractVector{<:Real};
                                alpha_min::Real = 0.1, offgrid_eta::Bool = true, excise = nothing)
    length(times) == 1 &&
        return adm_grid_from_metric(metric_fn, xs, ys, zs; alpha_min, offgrid_eta, excise,
                                    t = only(times))
    data = Array{Float64,5}(undef, 10, length(xs), length(ys), length(zs), length(times))
    for (n, t) in enumerate(times)
        g = adm_grid_from_metric(metric_fn, xs, ys, zs; alpha_min, offgrid_eta, excise, t = t)
        data[:, :, :, :, n] .= g.data
    end
    return ADMSeries(xs, ys, zs, times, data; alpha_min, offgrid_eta)
end
