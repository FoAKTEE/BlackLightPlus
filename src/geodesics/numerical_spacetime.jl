# Geodesics — numerical_spacetime.jl
#
# NUMERICAL spacetime: a frozen-slice ADM snapshot — lapse α, shift β^i, spatial
# metric γ_ij on a REGULAR Cartesian grid — assembled into the 4-metric g_μν.
# Metric derivatives still come from AD, not finite-difference probing, so the
# generic geodesic engine composes unchanged. That requires a C¹ interpolant:
# trilinear interpolation has a DISCONTINUOUS derivative that poisons the AD
# Christoffels at every cell face, so tensor-product CUBIC Catmull-Rom is used (C¹,
# exact for cubics, Dual-transparent). AMR (AthenaK) ingestion is not yet implemented.

using StaticArrays
using LinearAlgebra: dot, norm

# ---------------------------------------------------------------- Dual-safe value
# Strip a (possibly nested) ForwardDiff.Dual down to its Float64 value. Used ONLY
# to pick the integer stencil index by the floor of the coordinate — a piecewise
# constant that carries no derivative. The fractional weights keep the Dual, so
# the AD gradient of the interpolant is exact (and C¹, matching the analytic path).
@inline _rawval(x::Real) = x
@inline _rawval(x::ForwardDiff.Dual) = _rawval(ForwardDiff.value(x))

# ---------------------------------------------------------------- ADM grid spacetime
"""
    ADMGridSpacetime(xs, ys, zs, data; alpha_min=0.1, offgrid_eta=true)

Frozen-slice numerical spacetime on a REGULAR Cartesian grid. `data` is
`(10, nx, ny, nz)` holding the ten ADM fields per node in the column order
`(α, β^x, β^y, β^z, γ_xx, γ_xy, γ_xz, γ_yy, γ_yz, γ_zz)`; `xs,ys,zs` are the
uniform per-axis node ranges. `metric` samples the ten fields with a C¹ cubic
(Catmull-Rom) interpolant and assembles `g_μν` via the ADM formula:
`g_00 = -α² + β_i β^i`, `g_0i = β_i = γ_ij β^j`, `g_ij = γ_ij`. Off-grid returns
Minkowski (`offgrid_eta`); a ray terminates when the interpolated lapse falls
below `alpha_min` or on escape. Build one from any analytic metric with
[`adm_grid_from_metric`](@ref).
"""
struct ADMGridSpacetime{T} <: StationarySpacetime{T}
    x0::T; y0::T; z0::T                 # first-node coordinate per axis
    dx::T; dy::T; dz::T                 # uniform spacing per axis
    nx::Int; ny::Int; nz::Int
    data::Array{T,4}                    # (10, nx, ny, nz), ADM column order (see docstring)
    alpha_min::T                        # inner-terminate when interpolated α < this
    offgrid_eta::Bool                   # off-grid → Minkowski (true) vs clamped extrapolation
end

const _ADM_ALPHA2_FLOOR = 1.0e-16       # guards α² inside a horizon (`_adm_from_gcov`)

# ADM termination DIFFERS from the analytic `ESCAPE_RADIUS` (integrator.jl) for two
# slice-local reasons, so it gets its own constant: the grid is finite (escape must sit
# ≲ the inscribed half-width to stop while still on-grid), and the inner surface is the
# INTERPOLATED lapse dropping below `alpha_min`, not an analytic horizon.
const ADM_ESCAPE_RADIUS = 18.0          # default escape sphere (bounded by grid half-width)

function ADMGridSpacetime(xs::AbstractRange, ys::AbstractRange, zs::AbstractRange,
                          data::Array{Float64,4}; alpha_min::Real=0.1,
                          offgrid_eta::Bool=true)
    @assert size(data) == (10, length(xs), length(ys), length(zs)) "data must be (10, nx, ny, nz)"
    return ADMGridSpacetime{Float64}(first(xs), first(ys), first(zs),
                                     step(xs), step(ys), step(zs),
                                     length(xs), length(ys), length(zs),
                                     data, Float64(alpha_min), offgrid_eta)
end

# ---------------------------------------------------------------- cubic interpolant
# Catmull-Rom basis weights for the four nodes bracketing fractional position t∈[0,1]
# (tangent = centered difference ⇒ C¹ across every node). Weights sum to 1; the
# element type follows t (Dual under AD, Float64 otherwise).
@inline function _cr_weights(t)
    t2 = t * t
    t3 = t2 * t
    return SVector((-t3 + 2t2 - t) / 2,
                   (3t3 - 5t2 + 2) / 2,
                   (-3t3 + 4t2 + t) / 2,
                   (t3 - t2) / 2)
end

# Per-axis stencil: (four 1-based node indices, clamped to [1,n]) + Catmull-Rom
# weights. `iu` (0-based lower node) is picked from the DERIVATIVE-FREE floor of
# the coordinate; `t` keeps the Dual so ∂(interp)/∂c is exact.
@inline function _axis(c, c0, dc, n::Int)
    u = (c - c0) / dc                   # 0-based continuous index
    iu = floor(Int, _rawval(u))
    t = u - iu
    w = _cr_weights(t)
    idx = (clamp(iu, 1, n), clamp(iu + 1, 1, n), clamp(iu + 2, 1, n), clamp(iu + 3, 1, n))
    return w, idx
end

# Tensor-product cubic interpolation of field `comp` over the 4×4×4 stencil.
@inline function _interp(d, comp::Int, ix, iy, iz, wx, wy, wz)
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
    sample_adm(st, x, y, z) -> (α, β::SVector{3}, γ::SMatrix{3,3})

Cubically interpolate the ten ADM fields at Cartesian `(x, y, z)` (indices may be
`ForwardDiff.Dual`). `β` is the contravariant shift β^i; `γ` the spatial metric.

Duck-typed on the grid fields (`x0…dz`, `nx…nz`, `data`) rather than
`::ADMGridSpacetime`, so the GPU twin param struct (gpu_spacetime.jl `DevADMParams`,
same field names, `data::CuDeviceArray`) runs this SAME body in-kernel — the CPU
interpolant IS the device interpolant.
"""
@inline function sample_adm(st, x, y, z)
    wx, ix = _axis(x, st.x0, st.dx, st.nx)
    wy, iy = _axis(y, st.y0, st.dy, st.ny)
    wz, iz = _axis(z, st.z0, st.dz, st.nz)
    d = st.data
    α   = _interp(d, 1,  ix, iy, iz, wx, wy, wz)
    βx  = _interp(d, 2,  ix, iy, iz, wx, wy, wz)
    βy  = _interp(d, 3,  ix, iy, iz, wx, wy, wz)
    βz  = _interp(d, 4,  ix, iy, iz, wx, wy, wz)
    gxx = _interp(d, 5,  ix, iy, iz, wx, wy, wz)
    gxy = _interp(d, 6,  ix, iy, iz, wx, wy, wz)
    gxz = _interp(d, 7,  ix, iy, iz, wx, wy, wz)
    gyy = _interp(d, 8,  ix, iy, iz, wx, wy, wz)
    gyz = _interp(d, 9,  ix, iy, iz, wx, wy, wz)
    gzz = _interp(d, 10, ix, iy, iz, wx, wy, wz)
    γ = SMatrix{3,3}(gxx, gxy, gxz, gxy, gyy, gyz, gxz, gyz, gzz)
    return α, SVector(βx, βy, βz), γ
end

# On-grid test on the RAW coordinate (bounds are a Boolean predicate, not part of
# the differentiated value). Duck-typed like sample_adm (device twin runs it too).
@inline function on_grid(st, X::SVector{4})
    _inaxis(_rawval(X[2]), st.x0, st.dx, st.nx) &&
    _inaxis(_rawval(X[3]), st.y0, st.dy, st.ny) &&
    _inaxis(_rawval(X[4]), st.z0, st.dz, st.nz)
end
@inline _inaxis(c, c0, dc, n::Int) = (u = (c - c0) / dc; 0.0 <= u <= float(n - 1))

# ---------------------------------------------------------------- metric assembly
# The ONE ADM 4-metric assembly body, duck-typed on the grid fields so the GPU twin
# param struct (gpu_spacetime.jl `DevADMParams`) evaluates this SAME code in-kernel;
# `metric(::ADMGridSpacetime, ·)` below is a thin typed dispatch shell over it.
@inline function _adm_gcov(st, X::SVector{4,T}) where {T}
    if !on_grid(st, X)
        st.offgrid_eta && return SMatrix{4,4,T,16}(ETA)   # off-grid → Minkowski (ray terminates)
    end
    α, β, γ = sample_adm(st, X[2], X[3], X[4])
    βd = γ * β                                            # β_i = γ_ij β^j
    b2 = dot(βd, β)                                        # β_i β^i
    g00 = -α * α + b2
    gxx = γ[1,1]; gxy = γ[1,2]; gxz = γ[1,3]
    gyy = γ[2,2]; gyz = γ[2,3]; gzz = γ[3,3]
    return SMatrix{4,4,T,16}(g00,   βd[1], βd[2], βd[3],   # (symmetric ⇒ column = row)
                             βd[1], gxx,   gxy,   gxz,
                             βd[2], gxy,   gyy,   gyz,
                             βd[3], gxz,   gyz,   gzz)
end

"ADM 4-metric g_μν from the cubically interpolated (α, β^i, γ_ij)."
@inline metric(st::ADMGridSpacetime, X::SVector{4}) = _adm_gcov(st, X)

"Contravariant metric by direct 4×4 inverse; the generic AD `inverse_metric_gradient` then supplies ∂g^{μν}."
@inline inverse_metric(st::ADMGridSpacetime, x::SVector{4}) = inv(metric(st, x))

# ---------------------------------------------------------------- ADM-from-metric builder
"Extract ADM data (α, β^i, γ_ij) from a covariant 4-metric."
@inline function _adm_from_gcov(g::SMatrix{4,4})
    sp = SVector(2, 3, 4)
    γ = g[sp, sp]                                          # γ_ij = g_ij
    βd = SVector(g[1,2], g[1,3], g[1,4])                   # β_i = g_0i
    βu = inv(γ) * βd                                       # β^i = γ^ij β_j
    α2 = dot(βu, βd) - g[1,1]                              # α² = β^i β_i − g_00
    return sqrt(max(α2, _ADM_ALPHA2_FLOOR)), βu, γ
end

"""
    adm_grid_from_metric(metric_fn, xs, ys, zs; alpha_min=0.1, offgrid_eta=true,
                         excise=nothing)

Sample any analytic covariant metric closure `x::SVector{4} -> SMatrix{4,4}` onto
a regular Cartesian grid, storing the extracted ADM data — the "Kerr-as-grid"
factory used to validate the numerical path against the analytic tracer.
`metric_fn` is evaluated on the t=0 slice.
`excise` (optional `x::SVector{4} -> Bool`) marks nodes to fill with the
collapsed-lapse excision data instead of sampling `metric_fn` — pass the
horizon predicate, e.g. `x -> ks_radius(st, x) < horizon_radius(st)`, to give
the tracer a grid-alignment-independent inner stopping surface (without it the
only excised nodes are those where `metric_fn` happens to be non-finite — for
Kerr the measure-zero z=0 disk nodes, which a shifted/even-count grid misses).
"""
function adm_grid_from_metric(metric_fn, xs::AbstractRange, ys::AbstractRange,
                              zs::AbstractRange; alpha_min::Real=0.1, offgrid_eta::Bool=true,
                              excise=nothing)
    nx, ny, nz = length(xs), length(ys), length(zs)
    data = Array{Float64,4}(undef, 10, nx, ny, nz)
    # Collapsed-lapse EXCISION fill for nodes flagged by `excise` and for nodes where
    # the analytic metric is non-finite (KS ring r→0: f = 0/0 ⇒ NaN). Storing NaN
    # would poison the 4×4×4 cubic stencil OUTSIDE the horizon and abort captured
    # rays with retcode Unstable + an all-NaN state. Fill instead with a
    # below-threshold lapse (α_fill < alpha_min, β = 0, γ = I): non-degenerate
    # (g stays invertible) yet pulls the interpolated lapse under `alpha_min` near
    # the excised set, so a captured ray terminates cleanly on the lapse callback.
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
    return ADMGridSpacetime(xs, ys, zs, data; alpha_min, offgrid_eta)
end

# ---------------------------------------------------------------- geodesic tracing
# Interpolated lapse for the termination callback; off-grid returns a large value
# so the lapse root-finder stays quiet (escape/off-grid handled by the box radius).
@inline function _lapse_cb(st::ADMGridSpacetime, X::SVector{4})
    on_grid(st, X) || return one(eltype(X))
    return sample_adm(st, X[2], X[3], X[4])[1]
end

"""
    trace_geodesic(st::ADMGridSpacetime, y0; escape, ...) -> ODE solution

Adaptive geodesic solve terminating when the interpolated lapse `α < alpha_min`
(numerical horizon), on escape, or off-grid. `escape` should be ≲ the grid's
inscribed half-width so a ray leaves through the escape sphere while still on-grid.
"""
function trace_geodesic(st::ADMGridSpacetime, y0::SVector{8}; λmax=1.0e5,
                        reltol=1e-10, abstol=1e-12, escape=ADM_ESCAPE_RADIUS,
                        solver=Vern7(), save_everystep=false)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    pos(u) = SVector(u[1], u[2], u[3], u[4])
    lapse(u, λ, i) = _lapse_cb(st, pos(u)) - st.alpha_min
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = CallbackSet(ContinuousCallback(lapse, terminate!),
                     # escape only on the OUTBOUND upcrossing (see integrator.jl)
                     ContinuousCallback(esc, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback=cb,
                 save_everystep, save_start=save_everystep)
end
