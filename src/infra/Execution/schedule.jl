# Scheduling geometry of full-frame renders: the queue order of the pixels (longest
# predicted march first, in image tiles), the pixels near the critical curve that the
# fp32 lane cannot march, and the mixed-lane split of a queue. Pure functions of the
# camera geometry and the critical curve; every queue order gives the same image.
#
# Image layout: an `npix × npix` frame in the column-major [iβ, iα] layout, flat pixel
# index `(iα - 1) npix + iβ`. A critical curve is a tuple `(α, β)` of vectors tracing the
# closed curve on the camera's image plane, as `critical_curve(a, θo)` returns it.

"""
    queue_items(order, n) -> AbstractVector{Int}

Queue order of `n` items: `1:n` when `order === nothing`, else `order` after checking
that it is a permutation of `1:n` (`ArgumentError` otherwise).
"""
function queue_items(order, n::Integer)
    order === nothing && return 1:n
    (length(order) == n && isperm(order)) ||
        throw(ArgumentError("`order` must be a permutation of 1:$n"))
    return order
end

"""
    march_sort_order(counts) -> Vector{Int}

Permutation sorting items by descending march trip count (`counts[i]` = steps item `i`
marched, e.g. in the previous frame of a movie): the longest items are queued first.
"""
march_sort_order(counts::AbstractVector) = sortperm(counts; rev = true)

"""
    curve_inclination(inc_deg) -> Float64

Observer inclination, in radians, at which the critical curve for the scheduling
functions is evaluated: `inc_deg` in degrees, kept 0.02 rad away from either pole.
"""
curve_inclination(inc_deg::Real) = clamp(deg2rad(Float64(inc_deg)), 0.02, π - 0.02)

# Walk the closed critical curve at ≤ px/2 spacing on the image plane of an `npix`-pixel
# frame with field of view `fov`, calling `f(iα, iβ)` with the pixel of every sample
# (indices may fall outside 1:npix). Returns false when the curve is empty.
function walk_curve(f, curve, fov::Real, npix::Integer)
    αB, βB = curve
    isempty(αB) && return false
    px = Float64(fov) / npix
    half = Float64(fov) / 2
    m = length(αB)
    for s in 1:m
        s2 = s == m ? 1 : s + 1
        x1 = αB[s];  y1 = βB[s]
        x2 = αB[s2]; y2 = βB[s2]
        nsub = max(1, ceil(Int, hypot(x2 - x1, y2 - y1) / (px / 2)))
        for q in 0:nsub
            t = q / nsub
            f(floor(Int, (x1 + t * (x2 - x1) + half) / px) + 1,
              floor(Int, (y1 + t * (y2 - y1) + half) / px) + 1)
        end
    end
    return true
end

"""
    rescue_pixels(curve, fov, npix; rescue_px = 2) -> Vector{Int}

Flat indices, ascending, of every pixel of the `npix × npix` frame (field of view `fov`)
within `rescue_px` pixels (Chebyshev dilation) of the critical curve `curve`: the
Lyapunov-unstable set where an fp32 geodesic march fails. Empty when the curve misses
the frame.
"""
function rescue_pixels(curve, fov::Real, npix::Integer; rescue_px::Int = 2)
    npix = Int(npix)
    mask = falses(npix, npix)
    walk_curve(curve, fov, npix) do iα, iβ
        for dα in -rescue_px:rescue_px, dβ in -rescue_px:rescue_px
            jα = iα + dα; jβ = iβ + dβ
            (1 <= jα <= npix && 1 <= jβ <= npix) && (mask[jβ, jα] = true)
        end
    end || return Int[]
    return findall(vec(mask))
end

# Distance, in cells, from every `cell × cell`-pixel cell [iβ, iα] of the frame to the
# nearest cell the curve crosses: two raster passes propagating the nearest crossed cell
# over the 8-neighbourhood. Cells are infinitely far when the curve misses the frame.
function curve_cell_distance(curve, fov, npix::Int, cell::Int)
    n = cld(npix, cell)
    near = fill((0, 0), n, n)
    walk_curve(curve, fov, npix) do iα, iβ
        if 1 <= iα <= npix && 1 <= iβ <= npix
            i = (iβ - 1) ÷ cell + 1; j = (iα - 1) ÷ cell + 1
            near[i, j] = (i, j)
        end
    end
    for j in 1:n, i in 1:n
        relax!(near, i, j, i - 1, j); relax!(near, i, j, i, j - 1)
        relax!(near, i, j, i - 1, j - 1); relax!(near, i, j, i + 1, j - 1)
    end
    for j in n:-1:1, i in n:-1:1
        relax!(near, i, j, i + 1, j); relax!(near, i, j, i, j + 1)
        relax!(near, i, j, i + 1, j + 1); relax!(near, i, j, i - 1, j + 1)
    end
    return [near[i, j][1] == 0 ? Inf : sqrt(Float64(dist2(near[i, j], i, j)))
            for i in 1:n, j in 1:n]
end

dist2(c, i, j) = c[1] == 0 ? typemax(Int) : (i - c[1])^2 + (j - c[2])^2

# Adopt the nearest crossed cell of neighbour (k, l) at cell (i, j) when it is closer.
@inline function relax!(near, i, j, k, l)
    n = size(near, 1)
    (1 <= k <= n && 1 <= l <= n) || return nothing
    @inbounds c = near[k, l]
    @inbounds dist2(c, i, j) < dist2(near[i, j], i, j) && (near[i, j] = c)
    return nothing
end

"""
    cost_order(curve, fov, npix; tile = 16) -> Vector{Int}

Queue order of a full `npix × npix` frame, longest predicted work first: the pixels
grouped in `tile × tile` image tiles (column-major order inside a tile, which keeps the
sampling locality of neighbouring rays), the tiles in descending predicted cost. A ray
marches longest near the critical curve, where its orbit count grows as the logarithm of
the inverse impact-parameter offset, so a pixel's predicted cost is `log(n / (d + 1/4))`
with `d` its distance to `curve` on a grid of `max(1, tile ÷ 4)`-pixel cells and `n` the
grid size. Tiles of equal cost keep their index order.
"""
function cost_order(curve, fov::Real, npix::Integer; tile::Int = 16)
    npix = Int(npix)
    cell = max(1, tile ÷ 4)
    D = curve_cell_distance(curve, fov, npix, cell)
    nc = size(D, 1)
    nt = cld(npix, tile)
    tcost = zeros(nt, nt)                     # [β tile, α tile]: Σ over its pixels
    for j in 1:nc, i in 1:nc                  # a cell's pixels all lie in one tile
        npx = (min(i * cell, npix) - (i - 1) * cell) *
              (min(j * cell, npix) - (j - 1) * cell)
        tcost[(i - 1) * cell ÷ tile + 1, (j - 1) * cell ÷ tile + 1] +=
            npx * log(nc / (D[i, j] + 0.25))
    end
    order = Vector{Int}(undef, npix * npix)
    k = 0
    for t in sortperm(vec(tcost); rev = true)
        ta, tb = divrem(t - 1, nt)            # t: linear index of tcost[β tile, α tile]
        for iα in ta * tile + 1:min((ta + 1) * tile, npix),
            iβ in tb * tile + 1:min((tb + 1) * tile, npix)
            order[k += 1] = (iα - 1) * npix + iβ
        end
    end
    return order
end

"""
    mixed_partition(items, rescue) -> (rescue, bulk)

Split the queue `items` (a permutation of `1:n`) for a mixed-lane render: `rescue`, the
items marched on the fp64 lane and queued first, and `bulk`, the other items in `items`
order, marched on the fp32 lane. Queued as one job
(`foreach_item!(exec, (prog64 => rescue, prog32 => bulk); outs)`), the image equals an
fp32 sweep of every item overwritten by an fp64 sweep of the rescued ones.
"""
function mixed_partition(items::AbstractVector{<:Integer},
                         rescue::AbstractVector{<:Integer})
    rescued = falses(length(items))
    rescued[rescue] .= true
    return (rescue = rescue, bulk = filter(i -> !rescued[i], items))
end
