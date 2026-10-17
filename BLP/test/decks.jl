"""
    Decks

Shared synthetic inputs of the test suite: a torus-like primitive field on a
uniform grid and on a one-level block mesh, the reference run configuration,
`bits`, the bit view of a result used for bitwise comparisons, the camera rays of
every pixel, the fixed-control sampled trajectories of a ray set, the torus as a grid
provider and the program record the transfer marches read, the host-device lists of
the multi-device tests, and `device_isbits`, the check that a value is a valid kernel
argument once adapted to a device.
"""
module Decks

using StaticArrays
using BLP.Execution: HostDevice
using BLP.Spacetime: CameraFrame, pixel_centres, npixels, pixel_coords, camera_ray,
                     MarchBudget, march_control, rk4_step
using BLP.Matter: GridProvider, PlasmaPrimitives
import BLP.Matter: primitives, domain_radius

# BLP dependencies loaded by identity, so the module also loads in environments (the
# oracle's) where they are not direct dependencies.
const Adapt = Base.require(Base.PkgId(Base.UUID("79e6a3ab-5dfb-504d-930d-738a2a938a0e"), "Adapt"))
const ForwardDiff = Base.require(Base.PkgId(Base.UUID("f6369f11-7733-5829-9624-2563aa707210"),
                                            "ForwardDiff"))

"Bit pattern of every number in `x`; equal patterns mean bitwise-identical output, NaN and -0.0 included. A `ForwardDiff.Dual` contributes its value and then its partials."
bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
bits(x::ForwardDiff.Dual) = vcat(bits(ForwardDiff.value(x)), bits(collect(ForwardDiff.partials(x))))
bits(x::AbstractArray{<:ForwardDiff.Dual}) = reduce(vcat, (bits(v) for v in x))
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

"The torus of [`torus_primitives`](@ref) as a grid provider, sampled at the Float64 position."
struct TorusDeck <: GridProvider end
function primitives(::TorusDeck, x::SVector{4})
    t = torus_primitives(x[2], x[3], x[4])
    return PlasmaPrimitives{Float64}(Float64(t[1]), SVector{3,Float64}(t[2], t[3], t[4]), Float64(t[5]),
                                     SVector{3,Float64}(t[6], t[7], t[8]), t[1] > 0)
end
domain_radius(::TorusDeck) = 11.0

"A transfer program: the fields `BLP.Transfer.march` reads; `isbits` after `Adapt.adapt`."
struct MarchProgram{S,C,P,T,F}
    st::S
    control::C
    provider::P
    params::T
    frame::F
end
Adapt.@adapt_structure MarchProgram

"Device counts of the multi-device tests."
const NDEV = (1, 2, 3, 5)

"The logical devices `HostDevice(1)`, …, `HostDevice(nd)`."
hosts(nd) = [HostDevice(k) for k in 1:nd]

"""
    DeviceLike()

`Adapt` adaptor standing in for a GPU: every `Array` becomes a [`DeviceArrayLike`](@ref),
an `isbits` pointer-and-dims view of the same memory, as device arrays are inside a
kernel.
"""
struct DeviceLike end

"An `isbits` read-only array view of host memory: element pointer and dimensions."
struct DeviceArrayLike{T,N} <: AbstractArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Int}
end
Base.size(a::DeviceArrayLike) = a.dims
Base.IndexStyle(::Type{<:DeviceArrayLike}) = IndexLinear()
Base.getindex(a::DeviceArrayLike, i::Int) = unsafe_load(a.ptr, i)

Adapt.adapt_storage(::DeviceLike, a::Array{T,N}) where {T,N} =
    DeviceArrayLike{T,N}(pointer(a), size(a))

"Whether `x` is `isbits` after `Adapt.adapt` to a device, i.e. a valid kernel argument."
device_isbits(x) = isbits(Adapt.adapt(DeviceLike(), x))

end # module
