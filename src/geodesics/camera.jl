# Geodesics — camera.jl
#
# Plane-parallel camera (image-plane geometry of arXiv:2304.03804, Eqs. 5-7):
# image plane at distance d and inclination i from the spin axis, one parallel
# ray per pixel along the inward plane normal.
#
# Momentum normalization (blacklight convention): covariant p_μ rescaled so
# p_t = -1 exactly (E = 1), making the local frequency ν = -k·u × ν_obs with no
# extra per-pixel momentum factor.

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
centres), null and normalized to `p_t = -1`. The spatial momentum points INWARD,
so forward-λ integration traces camera → strong field (the standard backward-
ray-tracing convention). Works for ANY `AbstractSpacetime`: only `metric` and
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
    n_in = SVector(-sin(i), 0.0, -cos(i))            # inward plane normal
    rays = Vector{SVector{8,Float64}}(undef, length(coords))
    for (idx, (α, β)) in enumerate(coords)
        x = _image_to_bh(α, β, cam.inc_deg, cam.d)
        kcon = SVector(1.0, n_in[1], n_in[2], n_in[3])         # provisional k^μ
        p = metric(st, x) * kcon                                # lower index
        pnull = nullify_covariant(st, x, SVector(p[2], p[3], p[4]))
        rays[idx] = vcat(x, pnull / (-pnull[1]))                # p_t = -1 exactly
    end
    return rays
end
