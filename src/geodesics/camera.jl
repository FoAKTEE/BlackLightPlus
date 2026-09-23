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

"Map image-plane coords (α, β) to Cartesian-KS position (arXiv:2304.03804 Eqs. 5-7)."
@inline function _image_to_bh(α, β, inc_deg, d)
    i = deg2rad(inc_deg)
    return SVector(0.0,
                   -β * cos(i) + d * sin(i),
                   α,
                   β * sin(i) + d * cos(i))
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
    edges = range(-cam.fov / 2, cam.fov / 2; length=cam.npix + 1)
    centres = (edges[1:end-1] .+ edges[2:end]) ./ 2
    coords = [(α, β) for α in centres for β in centres]
    return camera_rays_at(st, cam, coords)
end

"Rays at arbitrary image-plane coordinates [(α, β), ...] (AMR refinement path)."
function camera_rays_at(st::AbstractSpacetime, cam::PlaneParallelCamera,
                        coords::Vector{Tuple{Float64,Float64}})
    i = deg2rad(cam.inc_deg)
    n_out = SVector(sin(i), 0.0, cos(i))             # outward plane normal (toward camera)
    rays = Vector{SVector{8,Float64}}(undef, length(coords))
    for (idx, (α, β)) in enumerate(coords)
        x = _image_to_bh(α, β, cam.inc_deg, cam.d)
        kcon = SVector(1.0, n_out[1], n_out[2], n_out[3])      # provisional received k^μ
        k = metric(st, x) * kcon                                # lower index
        knull = nullify_covariant(st, x, SVector(k[2], k[3], k[4]))   # k_t < 0 (E > 0)
        rays[idx] = vcat(x, knull / knull[1])                   # p = -k/E: p_t = +1 exactly
    end
    return rays
end
