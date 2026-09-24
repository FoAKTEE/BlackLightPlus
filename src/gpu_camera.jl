# gpu_camera.jl — camera rays made on the device (results/blp/multigpu/design.md §5.2).
# A chunk uploads only its item indices; `_camray_kernel!` maps each index to image-plane
# coordinates (a pixel of the camera grid, or a pixel of an AMR block) and calls the host's
# `camera_ray`, so the device rays equal the host `camera_rays` / `camera_rays_at` bit for
# bit on the KA CPU backend (on CUDA, up to fused multiply-add contraction). Included by
# gpu_image.jl.

using StaticArrays
using KernelAbstractions
using Adapt
using BlackLightPlus: PlaneParallelCamera
using BlackLightPlus.Geodesics: CameraFrame, camera_ray, pixel_centres

"""
    PixelGrid(centres, npix)

Item `k` ↦ pixel `k` of the npix×npix camera grid in `camera_rays` order:
α = centres[(k-1) ÷ npix + 1], β = centres[(k-1) % npix + 1].
"""
struct PixelGrid{V}
    centres::V
    npix::Int
end
Adapt.@adapt_structure PixelGrid

@inline image_coords(g::PixelGrid, k) =
    (@inbounds(g.centres[(k - 1) ÷ g.npix + 1]), @inbounds(g.centres[(k - 1) % g.npix + 1]))

"""
    BlockGrid(blocks, bs)

Item `k` ↦ pixel (i, j) of block w = (k-1) ÷ bs² + 1, `blocks[w] = (a0, b0, ps)`:
α = a0 + (i - 0.5) ps, β = b0 + (j - 0.5) ps with j fastest — the `render_amr`
coordinate order.
"""
struct BlockGrid{V}
    blocks::V
    bs::Int
end
Adapt.@adapt_structure BlockGrid

@inline function image_coords(g::BlockGrid, k)
    w, l = divrem(k - 1, g.bs * g.bs)
    i0, j0 = divrem(l, g.bs)
    b = @inbounds g.blocks[w + 1]
    i = i0 + 1; j = j0 + 1
    return (b[1] + (i - 0.5) * b[3], b[2] + (j - 0.5) * b[3])
end

_upload_gen(backend, g::PixelGrid) = PixelGrid(_upload(backend, g.centres), g.npix)
_upload_gen(backend, g::BlockGrid) = BlockGrid(_upload(backend, g.blocks), g.bs)

# One ray per thread: item -> image-plane coordinates -> camera_ray, stored at the
# ray eltype (the fp32 lane narrows exactly as the host conversion). A ray without a
# positive-energy null root at the camera sets bad[1].
@kernel function _camray_kernel!(y0s, bad, @Const(items), gen, st, fr, n)
    i = @index(Global)
    @inbounds if i <= n
        α, β = image_coords(gen, items[i])
        y, ok = camera_ray(st, fr, α, β)
        y0s[i] = eltype(y0s)(y)
        ok || (bad[1] = Int32(1))
    end
end

"""
    CameraRays(st, cam, T=Float64)
    CameraRays(st, fr::CameraFrame, gen, T=Float64)

Ray source of `march!` that makes a chunk's rays on the device from its item indices:
the camera's pixel grid (`cam`), or any `PixelGrid` / `BlockGrid` generator `gen`, as
`SVector{8,T}` states. `st` is the spacetime (isbits), or a Dict of per-device
spacetimes (`upload_adm`).
"""
struct CameraRays{T,S,G}
    st::S
    fr::CameraFrame
    gen::G
    id::Int                    # tags `gen` in the workers' upload cache
end
const _CAMRAYS_ID = Threads.Atomic{Int}(0)
CameraRays(st, fr::CameraFrame, gen, ::Type{T} = Float64) where {T} =
    CameraRays{T,typeof(st),typeof(gen)}(st, fr, gen, Threads.atomic_add!(_CAMRAYS_ID, 1) + 1)
CameraRays(st, cam::PlaneParallelCamera, ::Type{T} = Float64) where {T} =
    CameraRays(st, CameraFrame(cam), PixelGrid(pixel_centres(cam), cam.npix), T)

_device_st(st, dev) = st
_device_st(st::AbstractDict, dev) = st[dev]

# The generator's device arrays on this lane, uploaded once per source.
function _device_gen!(lane::PoolLane, src::CameraRays)
    key = (:camgen, typeof(src.gen))
    c = get(lane.bufs, key, nothing)
    c !== nothing && c[1] == src.id && return c[2]
    g = _upload_gen(lane.backend, src.gen)
    lane.bufs[key] = (src.id, g)
    return g
end

function rays_on_device!(lane::PoolLane, src::CameraRays{T}, idx) where {T}
    n = length(idx)
    st = lane_stage(lane, :items, Int32, n)
    @inbounds for (q, i) in enumerate(idx)
        st[q] = i
    end
    itd = lane_buffer(lane, :items, Int32, n)
    copyto!(itd, 1, st, 1, n)
    y0d = lane_buffer(lane, :rays, SVector{8,T}, n)
    bad = lane_buffer(lane, :camray_bad, Int32, 1)
    fill!(bad, 0)
    _camray_kernel!(lane.backend)(y0d, bad, itd, _device_gen!(lane, src),
                                  _device_st(src.st, lane.dev), src.fr, n; ndrange = n)
    KernelAbstractions.synchronize(lane.backend)
    Array(view(bad, 1:1))[1] == 0 ||
        error("camera rays: no past-oriented (p_t<0 ⇒ E>0) null root at the camera " *
              "position (inside the ergosphere)")
    return y0d
end
