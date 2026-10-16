# Plane-parallel camera (image-plane geometry of arXiv:2304.03804, Eqs. 5-7): image plane
# at distance d and inclination i from the spin axis, one parallel ray per pixel along
# the inward plane normal.
#
# Time orientation: the received photon has the future-directed wave covector k_μ,
# normalized to k_t = -1 (E = 1). The ray state carries p_μ = -k_μ (p_t = +1), so the
# unchanged Hamiltonian flow to increasing λ follows the same null geodesic from the
# camera back toward the source, with x^t decreasing. Consumers recover k = -p, and the
# fluid-frame frequency is ν = -k·u ν_obs.

"""
    PlaneParallelCamera(; inc_deg = 60.0, d = 1000.0, fov = 40.0, npix = 64)

Square plane-parallel camera at inclination `inc_deg` (degrees from the spin
axis) and distance `d` (M), field of view `fov` (M) on each side, `npix` pixels
per side.
"""
Base.@kwdef struct PlaneParallelCamera
    inc_deg::Float64 = 60.0
    d::Float64       = 1000.0
    fov::Float64     = 40.0
    npix::Int        = 64
end

"""
    CameraFrame{T}
    CameraFrame(cam::PlaneParallelCamera)

The `isbits` image-plane geometry of a camera: `sin_i`, `cos_i` of the
inclination, the distance `d`, and `up`, the contravariant image vertical
`(0, -cos i, 0, sin i)` (image `+β`, north) that the polarized march transports
as the camera polarization frame.
"""
struct CameraFrame{T}
    sin_i::T
    cos_i::T
    d::T
    up::SVector{4,T}
end

function CameraFrame(cam::PlaneParallelCamera)
    i = deg2rad(cam.inc_deg)
    return CameraFrame(sin(i), cos(i), cam.d, SVector(0.0, -cos(i), 0.0, sin(i)))
end

"""
    camera_ray(st, frame::CameraFrame, α, β) -> (y::RayState, ok::Bool)

The initial state `[x^μ; p_μ]` of the ray at image-plane coordinates `(α, β)`:
position from arXiv:2304.03804 Eqs. 5–7, momentum `p = -k/E` with `k` the null
covector received along the inward plane normal, so `p_t = +1` exactly. `ok` is
false when no positive-energy null root exists at the camera (see
[`null_covariant`](@ref)).
"""
@inline function camera_ray(st, fr::CameraFrame{T}, α, β) where {T}
    x = SVector(zero(T), -β * fr.cos_i + fr.d * fr.sin_i, α, β * fr.sin_i + fr.d * fr.cos_i)
    kcon = SVector(one(T), fr.sin_i, zero(T), fr.cos_i)    # provisional received k^μ
    k = metric(st, x) * kcon
    knull, ok = null_covariant(st, x, SVector(k[2], k[3], k[4]))   # k_t < 0 (E > 0)
    return raystate(x, knull / knull[1]), ok
end

"""
    npixels(cam::PlaneParallelCamera) -> Int

Number of pixels, `npix²`.
"""
npixels(cam::PlaneParallelCamera) = cam.npix * cam.npix

"""
    pixel_centres(cam::PlaneParallelCamera) -> Vector{Float64}

Pixel-centre image coordinates along one axis (`α` and `β` share them), from
`-fov/2 + fov/(2 npix)` to `fov/2 - fov/(2 npix)`.
"""
function pixel_centres(cam::PlaneParallelCamera)
    edges = range(-cam.fov / 2, cam.fov / 2; length = cam.npix + 1)
    return collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
end

"""
    pixel_coords(centres, npix, k) -> (α, β)

Image coordinates of pixel `k` of an `npix × npix` grid in camera order (`β`
fastest): `α = centres[(k-1) ÷ npix + 1]`, `β = centres[(k-1) % npix + 1]`.
`centres` may be a device array.
"""
@inline pixel_coords(centres, npix::Int, k::Integer) =
    (@inbounds(centres[(k - 1) ÷ npix + 1]), @inbounds(centres[(k - 1) % npix + 1]))
