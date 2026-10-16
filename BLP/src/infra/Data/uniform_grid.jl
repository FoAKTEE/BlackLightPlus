# The fluid resampled onto a uniform Cartesian node grid: a grid provider with one
# index map `(x - xo) * inv_dx` per axis and vacuum outside the node box. One struct
# serves host and device; `Adapt.adapt` replaces the array.

"""
    UniformGrid{A4} <: GridProvider
    UniformGrid(data, box_half)

The fluid on the `n³` nodes of the box `[-box_half, box_half]³`. Fields: `data`, the
`(8, n, n, n)` `Float32` primitives in `PRIM_NAMES` order (feature fastest), node
`(i, j, k)` at `(xo + (i-1)/inv_dx, xo + (j-1)/inv_dx, xo + (k-1)/inv_dx)`; `n`;
`xo = -box_half`; `inv_dx = (n - 1)/(2 box_half)`; `box_half`. Nodes outside every
block hold zeros. The two-argument constructor derives `n`, `xo` and `inv_dx` from
`data` and `box_half`. Built from a provider by [`resample`](@ref); `isbits` after
`Adapt.adapt`.
"""
struct UniformGrid{A4} <: GridProvider
    data::A4
    n::Int
    xo::Float64
    inv_dx::Float64
    box_half::Float64
end
Adapt.@adapt_structure UniformGrid

function UniformGrid(data::AbstractArray{<:Real,4}, box_half::Real)
    n = size(data, 2)
    size(data) == (8, n, n, n) || throw(ArgumentError("data must be (8, n, n, n); got $(size(data))"))
    bh = Float64(box_half)
    return UniformGrid(data, n, -bh, (n - 1) / (2 * bh), bh)
end

"""
    UniformGrid(p::GridProvider, box_half, n) -> UniformGrid

The same as [`resample`](@ref)`(p, box_half, n)`.
"""
UniformGrid(p::GridProvider, box_half::Real, n::Integer) = resample(p, box_half, n)

"""
    corner(g::UniformGrid, i, j, k) -> SVector{8,Float64}

The eight primitives of node `(i, j, k)` (1-based), widened to `Float64`.
"""
@inline corner(g::UniformGrid, i, j, k) =
    SVector{8,Float64}(ntuple(f -> Float64(@inbounds g.data[f, i, j, k]), Val(8)))

"""
    grid_sample(g::UniformGrid, x, y, z) -> (p::SVector{8,Float64}, inside::Bool)

Trilinear sample of the eight primitives at `(x, y, z)`. The fractional node index is
`f = (x - xo) * inv_dx` per axis; a point with any `f < 0` or `f > n - 1` is outside
(zeros, `false`). Inside, the lower node is `min(⌊f⌋, n - 2)` and the fraction
`f - ⌊f⌋` is not clamped; the corners are blended by `lerp8` along `i`, `j`, `k`.
"""
@inline function grid_sample(g::UniformGrid, x, y, z)
    ng = g.n
    xo = g.xo
    inv_dx = g.inv_dx
    fx = (x - xo) * inv_dx; fy = (y - xo) * inv_dx; fz = (z - xo) * inv_dx
    (fx < 0.0 || fy < 0.0 || fz < 0.0 ||
     fx > ng - 1 || fy > ng - 1 || fz > ng - 1) && return zero(SVector{8,Float64}), false
    i0 = min(unsafe_trunc(Int, fx), ng - 2)
    j0 = min(unsafe_trunc(Int, fy), ng - 2)
    k0 = min(unsafe_trunc(Int, fz), ng - 2)
    dxf = fx - i0; dyf = fy - j0; dzf = fz - k0
    il = i0 + 1; jl = j0 + 1; kl = k0 + 1; ih = il + 1; jh = jl + 1; kh = kl + 1
    c = (corner(g, il, jl, kl), corner(g, ih, jl, kl), corner(g, il, jh, kl), corner(g, ih, jh, kl),
         corner(g, il, jl, kh), corner(g, ih, jl, kh), corner(g, il, jh, kh), corner(g, ih, jh, kh))
    return lerp8(c, dxf, dyf, dzf), true
end

"""
    primitives(g::UniformGrid, x::SVector{4}) -> PlasmaPrimitives{Float64}

The primitives at the spacetime point `x = (t, x, y, z)` (time ignored, position
converted to `Float64`): the [`grid_sample`](@ref), with `inside` true within the node
box.
"""
@inline function primitives(g::UniformGrid, x::SVector{4})
    p, inside = grid_sample(g, Float64(x[2]), Float64(x[3]), Float64(x[4]))
    return _record(p, inside)
end

"""
    domain_radius(g::UniformGrid) -> Float64

`√3 |xo|`, the radius of the sphere that circumscribes the node box.
"""
domain_radius(g::UniformGrid) = sqrt(3.0) * abs(g.xo)

"""
    resample(p::GridProvider, box_half, n) -> UniformGrid

Sample the host provider `p` (an [`AthenaKSnapshot`](@ref) or [`FluidBlocks`](@ref))
at the `n³` nodes `-box_half + (i - 1) dx`, `dx = 2 box_half/(n - 1)`, of the box
`[-box_half, box_half]³` at time 0, rounding each primitive to `Float32`; nodes outside
the provider's domain hold zeros. Threaded over `z` slabs.
"""
function resample(p::GridProvider, box_half::Real, n::Integer)
    n = Int(n); box_half = Float64(box_half)
    data = Array{Float32}(undef, 8, n, n, n)
    dx = 2 * box_half / (n - 1)
    Threads.@threads for k in 1:n
        z = -box_half + (k - 1) * dx
        @inbounds for j in 1:n
            y = -box_half + (j - 1) * dx
            for i in 1:n
                x = -box_half + (i - 1) * dx
                s = primitives(p, SVector(0.0, x, y, z))
                data[1, i, j, k] = s.rho
                data[2, i, j, k] = s.uu[1]
                data[3, i, j, k] = s.uu[2]
                data[4, i, j, k] = s.uu[3]
                data[5, i, j, k] = s.eint
                data[6, i, j, k] = s.bb[1]
                data[7, i, j, k] = s.bb[2]
                data[8, i, j, k] = s.bb[3]
            end
        end
    end
    return UniformGrid(data, n, -box_half, (n - 1) / (2 * box_half), box_half)
end

"""
    scale_density(g::UniformGrid, factor) -> UniformGrid

`g` with the density (feature 1) multiplied by `Float32(factor)` in a copy of `data`;
`factor == 1` returns `g` itself.
"""
function scale_density(g::UniformGrid, factor::Real)
    factor == 1.0 && return g
    data = copy(g.data)
    @inbounds @views data[1, :, :, :] .*= Float32(factor)
    return UniformGrid(data, g.n, g.xo, g.inv_dx, g.box_half)
end

"""
    same_mesh(a::UniformGrid, b::UniformGrid) -> Bool

Whether the two grids have the same nodes (`n`, `xo`, `inv_dx`).
"""
same_mesh(a::UniformGrid, b::UniformGrid) = a.n == b.n && a.xo == b.xo && a.inv_dx == b.inv_dx
