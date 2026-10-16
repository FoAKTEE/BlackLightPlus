"""
    Decks

Shared synthetic inputs of the test suite: a torus-like primitive field on a
uniform grid and on a one-level block mesh, the reference run configuration,
`bits`, the bit view of a result used for bitwise comparisons, the camera rays of
every pixel, and the fixed-control sampled trajectories of a ray set.
"""
module Decks

using StaticArrays
using BLP.Spacetime: CameraFrame, pixel_centres, npixels, pixel_coords, camera_ray,
                     MarchBudget, march_control, rk4_step

"Bit pattern of every number in `x`; equal patterns mean bitwise-identical output, NaN and -0.0 included."
bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
bits(x::AbstractArray{<:SVector}) = reduce(vcat, (bits(collect(v)) for v in x))
bits(x::NamedTuple) = reduce(vcat, (bits(v) for v in values(x)))
bits(x::Tuple) = reduce(vcat, (bits(v) for v in x))
bits(x::Real) = [reinterpret(UInt64, Float64(x))]
bits(x::AbstractVector{<:AbstractArray}) = reduce(vcat, (bits(m) for m in x))

"Initial states of every pixel of `cam` in camera order (β fastest); errors when a ray has no positive-energy null root."
function camera_rays(st, cam)
    fr = CameraFrame(cam)
    centres = pixel_centres(cam)
    return map(1:npixels(cam)) do k
        y, ok = camera_ray(st, fr, pixel_coords(centres, cam.npix, k)...)
        ok || error("no positive-energy null root at the camera")
        y
    end
end

"""
Fixed-control trajectories of `rays` under `march_control` and `rk4_step`, ordered source
to camera: `(states, dls)` with `dls[i][k]` the affine step from state k to k+1 (last 0).
"""
function sampled_trace(st, rays::Vector{SVector{8,T}}; C, escape, nmax, R_em = T(Inf)) where {T}
    b = MarchBudget{T}(T(C), T(escape), T(R_em), nmax)
    states = Vector{Vector{SVector{8,T}}}(undef, length(rays))
    dls = Vector{Vector{T}}(undef, length(rays))
    for i in eachindex(rays)
        ys = [rays[i]]; hs = T[]
        y = rays[i]; r_prev = T(Inf)
        for _ in 1:nmax
            m = march_control(st, y, r_prev, b)
            m.stop && break
            r_prev = m.R
            y = rk4_step(st, y, m.h)
            push!(ys, y); push!(hs, m.h)
        end
        states[i] = reverse(ys)
        dls[i] = vcat(reverse(hs), zero(T))
    end
    return states, dls
end

"Torus-like primitives (rho, uu1..3, eint, bb1..3) at (x, y, z); zero outside 2.2 < r < 11."
function torus_primitives(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < 2.2 || r > 11.0) && return ntuple(_ -> 0.0f0, 8)
    rho = exp(-((r - 5.5) / 2.5)^2)
    return Float32.((rho, -0.2y / r, 0.2x / r, 0.01, 0.08rho,
                     0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho)))
end

end # module
