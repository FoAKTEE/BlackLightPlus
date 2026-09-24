# Geodesics — camera.jl
#
# Plane-parallel camera (image-plane geometry of arXiv:2304.03804, Eqs. 5-7):
# image plane at distance d and inclination i from the spin axis, one parallel
# ray per pixel along the inward plane normal.
#
# Time orientation: the received photon has wave covector k_μ (future-directed,
# travelling from the source to the camera), normalized to k_t = -1 (E = 1). The ray
# state carries p_μ = -k_μ (p_t = +1): integrating the unchanged Hamiltonian flow to
# increasing λ follows the same null geodesic from the camera back toward the source,
# with x^t decreasing. Consumers recover k = -p, so the fluid-frame frequency is
# ν = -k·u × ν_obs with no extra per-pixel momentum factor.

"Plane-parallel camera at inclination `inc_deg` [deg], distance `d`, square fov."
Base.@kwdef struct PlaneParallelCamera
    inc_deg::Float64 = 60.0
    d::Float64       = 1000.0
    fov::Float64     = 40.0
    npix::Int        = 64
end

"""
    CameraFrame(cam::PlaneParallelCamera)

Isbits image-plane geometry of `cam` (sin and cos of the inclination, distance): the
argument of `camera_ray`, shared by the host ray loop and the device ray kernels.
"""
struct CameraFrame
    sin_i::Float64
    cos_i::Float64
    d::Float64
end
CameraFrame(cam::PlaneParallelCamera) =
    (i = deg2rad(cam.inc_deg); CameraFrame(sin(i), cos(i), cam.d))

"""
    camera_ray(st, fr::CameraFrame, α, β) -> (y::SVector{8}, ok::Bool)

Initial state `[x^μ; p_μ]` of the ray at image-plane coordinates (α, β): position from
arXiv:2304.03804 Eqs. 5-7, `p = -k/E` with `k` the null covector received along the
inward plane normal (`p_t = +1`). `ok == false` when no positive-energy null root exists
at the camera (see `nullify_covariant`). The single source of `camera_rays_at` and the
device ray kernels.
"""
@inline function camera_ray(st, fr::CameraFrame, α, β)
    x = SVector(0.0, -β * fr.cos_i + fr.d * fr.sin_i, α, β * fr.sin_i + fr.d * fr.cos_i)
    kcon = SVector(1.0, fr.sin_i, 0.0, fr.cos_i)           # provisional received k^μ
    k = metric(st, x) * kcon                                # lower index
    knull, ok = null_covariant(st, x, SVector(k[2], k[3], k[4]))   # k_t < 0 (E > 0)
    return vcat(x, knull / knull[1]), ok                    # p = -k/E: p_t = +1 exactly
end

"Pixel-centre image-plane coordinates of `cam` along one axis (α and β share them)."
function pixel_centres(cam::PlaneParallelCamera)
    edges = range(-cam.fov / 2, cam.fov / 2; length=cam.npix + 1)
    return collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
end

"""
    camera_rays(st, cam) -> Vector{SVector{8,Float64}}

Initial states `[x^μ; p_μ]` for every pixel (row-major over (α, β) pixel
centres): `p = -k` with `k` the null wave covector of the photon received at the
camera, normalized to `k_t = -1` (so `p_t = +1`). Forward-λ integration traces the
received photon backward, camera → source (past-directed, spatially inward). Works for ANY `AbstractSpacetime`: only `metric` and
`nullify_covariant` at the camera location are required.
"""
function camera_rays(st::AbstractSpacetime, cam::PlaneParallelCamera)
    centres = pixel_centres(cam)
    return camera_rays_at(st, cam, [(α, β) for α in centres for β in centres])
end

"Rays at arbitrary image-plane coordinates [(α, β), ...] (AMR refinement path)."
function camera_rays_at(st::AbstractSpacetime, cam::PlaneParallelCamera,
                        coords::Vector{Tuple{Float64,Float64}})
    fr = CameraFrame(cam)
    rays = Vector{SVector{8,Float64}}(undef, length(coords))
    for (idx, (α, β)) in enumerate(coords)
        y, ok = camera_ray(st, fr, α, β)
        ok || error("camera_rays_at: no past-oriented (p_t<0 ⇒ E>0) null root at the " *
                    "camera position x=$(y[SVector(1, 2, 3, 4)]) (inside the ergosphere)")
        rays[idx] = y
    end
    return rays
end
