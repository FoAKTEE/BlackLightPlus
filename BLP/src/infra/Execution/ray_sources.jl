# Ray sources: the initial state of an item's ray, made inside the program. A source is a
# program field and reaches each device with the program (its arrays become resident).
# `HostRays` reads host-built states; `CameraRays` maps the item to image-plane
# coordinates through a generator and calls the spacetime's `camera_ray`, so device rays
# equal host rays made by the same function.

"""
    RaySource

Where a program's rays come from. A source answers [`ray`](@ref)`(src, item)` and is
`isbits` after `Adapt.adapt` to a device. Concrete sources: [`HostRays`](@ref),
[`CameraRays`](@ref).
"""
abstract type RaySource end

"""
    ray(src, item) -> (y::SVector{8}, ok::Bool)

Initial ray state `y = [x^μ; p_μ]` of `item` from the source `src`, and whether it is
valid (`false` when the camera position admits no past-oriented null root). Pure and
allocation-free; runs inside kernels.
"""
function ray end

"""
    HostRays(rays)

Ray source of host-built initial states: item `k` is `rays[k]` (an `SVector{8}`), always
valid. The array is uploaded once per device with the program.
"""
struct HostRays{V} <: RaySource
    rays::V
end
Adapt.@adapt_structure HostRays

@inline ray(src::HostRays, item) = (@inbounds(src.rays[item]), true)

"""
    image_coords(gen, item) -> (α, β)

Image-plane coordinates of `item` under the generator `gen` (a [`PixelGrid`](@ref),
[`BlockGrid`](@ref) or [`CoordList`](@ref)), in the units and orientation of the camera
that `camera_ray` uses.
"""
function image_coords end

"""
    PixelGrid(centres, npix)
    PixelGrid(fov, npix)

The `npix × npix` pixel grid of a square camera with field of view `fov`: item `k` is
the pixel `α = centres[(k - 1) ÷ npix + 1]`, `β = centres[(k - 1) % npix + 1]` (β fastest,
the column-major image layout). `centres` are the pixel centres along either axis; the
second form computes them as the midpoints of `npix + 1` equal cells spanning
`[-fov/2, fov/2]`.
"""
struct PixelGrid{V}
    centres::V
    npix::Int
    PixelGrid(centres::V, npix::Integer) where {V} = new{V}(centres, Int(npix))
end
Adapt.@adapt_structure PixelGrid

function PixelGrid(fov::Real, npix::Integer)
    f = Float64(fov)
    edges = range(-f / 2, f / 2; length = npix + 1)
    return PixelGrid(collect((edges[1:end-1] .+ edges[2:end]) ./ 2), npix)
end

@inline image_coords(g::PixelGrid, k) =
    (@inbounds(g.centres[(k - 1) ÷ g.npix + 1]), @inbounds(g.centres[(k - 1) % g.npix + 1]))

"""
    BlockGrid(blocks, bs)
    BlockGrid(bs)

Pixels of square image blocks of `bs × bs` pixels: `blocks[w] = SVector(a0, b0, ps)` is
block `w`'s lower corner and pixel size, and item `k` is pixel `(i, j)` (j fastest) of
block `w = (k - 1) ÷ bs² + 1`, at `α = a0 + (i - 0.5) ps`, `β = b0 + (j - 0.5) ps`. The
one-argument form has no blocks; [`stream_blocks!`](@ref) supplies them per chunk.
"""
struct BlockGrid{V}
    blocks::V
    bs::Int
    BlockGrid(blocks::V, bs::Integer) where {V} = new{V}(blocks, Int(bs))
end
Adapt.@adapt_structure BlockGrid
BlockGrid(bs::Integer) = BlockGrid(SVector{3,Float64}[], bs)

@inline function image_coords(g::BlockGrid, k)
    w, l = divrem(k - 1, g.bs * g.bs)
    i0, j0 = divrem(l, g.bs)
    b = @inbounds g.blocks[w + 1]
    i = i0 + 1; j = j0 + 1
    return (b[1] + (i - 0.5) * b[3], b[2] + (j - 0.5) * b[3])
end

"""
    CoordList(α, β)
    CoordList(coords)

An explicit list of image-plane coordinates: item `k` is `(α[k], β[k])`. The second form
takes a vector of `(α, β)` tuples.
"""
struct CoordList{A,B}
    α::A
    β::B
    CoordList(α::A, β::B) where {A,B} = new{A,B}(α, β)
end
Adapt.@adapt_structure CoordList
CoordList(coords::AbstractVector{<:Tuple{Real,Real}}) =
    CoordList(Float64[c[1] for c in coords], Float64[c[2] for c in coords])

@inline image_coords(g::CoordList, k) = (@inbounds(g.α[k]), @inbounds(g.β[k]))

"""
    CameraRays(st, frame, gen)

Ray source of camera rays made in the program: item `k` is mapped to `(α, β)` by the
generator `gen` ([`PixelGrid`](@ref), [`BlockGrid`](@ref) or [`CoordList`](@ref)) and its
ray is `camera_ray(st, frame, α, β)` for the spacetime `st` and camera frame `frame`.
A program on a narrower geodesic lane converts the returned state itself.
"""
struct CameraRays{S,F,G} <: RaySource
    st::S
    frame::F
    gen::G
end
Adapt.@adapt_structure CameraRays

@inline function ray(src::CameraRays, item)
    α, β = image_coords(src.gen, item)
    y, ok = Spacetime.camera_ray(src.st, src.frame, α, β)
    return (y, ok::Bool)
end
