"""
    Decks

Shared synthetic inputs of the test suite: a torus-like primitive field on a
uniform grid and on a one-level block mesh, the reference run configuration,
`bits`, the bit view of a result used for bitwise comparisons, the host-device
lists of the multi-device tests, and `device_isbits`, the check that a value is a
valid kernel argument once adapted to a device.
"""
module Decks

using StaticArrays
using Adapt: Adapt
using BLP.Execution: HostDevice

"Bit pattern of every number in `x`; equal patterns mean bitwise-identical output, NaN and -0.0 included."
bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
bits(x::AbstractArray{<:SVector}) = reduce(vcat, (bits(collect(v)) for v in x))
bits(x::NamedTuple) = reduce(vcat, (bits(v) for v in values(x)))
bits(x::Tuple) = reduce(vcat, (bits(v) for v in x))
bits(x::Real) = [reinterpret(UInt64, Float64(x))]
bits(x::AbstractVector{<:AbstractArray}) = reduce(vcat, (bits(m) for m in x))

"Torus-like primitives (rho, uu1..3, eint, bb1..3) at (x, y, z); zero outside 2.2 < r < 11."
function torus_primitives(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < 2.2 || r > 11.0) && return ntuple(_ -> 0.0f0, 8)
    rho = exp(-((r - 5.5) / 2.5)^2)
    return Float32.((rho, -0.2y / r, 0.2x / r, 0.01, 0.08rho,
                     0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho)))
end

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
