# The host epilogues of the image products: the band sums, the NaN and zero maps and the
# planes of a Stokes-vector image, as programs over the items of host arrays. They run on
# Julia threads whatever executor ran the product, since their inputs and outputs are on
# the host; every item reads and writes its own elements only, so the result is the same
# on any executor and in any order.

"""
    BandSum(sums)

Band sums of per-band maps: `sums` holds, per output `k`, a tuple of band maps
`(m₁, …, m_K)`, and item `item` writes `outs[k][i] = ((0 + m₁[item]) + m₂[item]) + … +
m_K[item]`, the bands added in order starting from zero, which gives the bits of
`I = zeros(…)` followed by `I .+= m_b` band after band. Called as `prog(item, i, outs...)`
with one output per entry of `sums`.
"""
struct BandSum{S<:Tuple} <: Function
    sums::S
end
Adapt.@adapt_structure BandSum

@inline add_bands(::Tuple{}, item, acc) = acc
@inline add_bands(maps::Tuple, item, acc) =
    add_bands(Base.tail(maps), item, acc + @inbounds(first(maps)[item]))

@inline store_sums!(::Tuple{}, item, i, ::Tuple{}) = nothing
@inline function store_sums!(sums::Tuple, item, i, outs::Tuple)
    o = first(outs)
    @inbounds o[i] = add_bands(first(sums), item, zero(eltype(o)))
    return store_sums!(Base.tail(sums), item, i, Base.tail(outs))
end

function (p::BandSum)(item, i, outs::Vararg{Any,N}) where {N}
    N == length(p.sums) ||
        throw(ArgumentError("BandSum: $N outputs for $(length(p.sums)) sums"))
    store_sums!(p.sums, item, i, outs)
    return nothing
end

"""
    Fill(value)

A constant map: item `item` writes `value` to every output, `outs[k][i] = value`. Called
as `prog(item, i, outs...)`.
"""
struct Fill{T} <: Function
    value::T
end

@inline fill_all!(::Tuple{}, i, v) = nothing
@inline function fill_all!(outs::Tuple, i, v)
    @inbounds first(outs)[i] = v
    return fill_all!(Base.tail(outs), i, v)
end

function (p::Fill)(item, i, outs::Vararg{Any,N}) where {N}
    fill_all!(outs, i, p.value)
    return nothing
end

"""
    Split(src)

The components of vector-valued items as planes: item `item` writes component `k` of
`src[item]` (an `SVector`) to output `k`, `outs[k][i] = src[item][k]`. Called as
`prog(item, i, outs...)` with one output per component.
"""
struct Split{V} <: Function
    src::V
end
Adapt.@adapt_structure Split

function (p::Split)(item, i, outs::Vararg{Any,N}) where {N}
    v = @inbounds p.src[item]
    N == length(v) || throw(ArgumentError("Split: $N outputs for $(length(v)) components"))
    store_all!(outs, i, Tuple(v))
    return nothing
end

# Items per chunk of an epilogue: each costs a few nanoseconds, so a chunk of 2^14 is
# tens of microseconds of work or more, and an image of up to 2^14 pixels runs as one task.
const EPILOGUE_CHUNK = 2^14

# `prog` over the items `1:n` of the host arrays `outs` on Julia threads.
function host_epilogue!(prog, outs::Tuple, n::Integer)
    isempty(outs) && return outs
    return foreach_item!(ThreadedExecutor(chunk = EPILOGUE_CHUNK), prog, 1:n; outs)
end

"""
    band_sums(groups, dims) -> Tuple of arrays

Per group of band maps (`groups`, a tuple of vectors of maps) the band sum in band order
from zero into a new `Float64` array of size `dims` ([`BandSum`](@ref), all groups in one
run on Julia threads).
"""
function band_sums(groups::Tuple, dims::Dims)
    outs = map(_ -> Array{Float64}(undef, dims), groups)
    host_epilogue!(BandSum(map(Tuple, groups)), outs, prod(dims))
    return outs
end

"""
    filled_maps(value, k, dims) -> Vector of arrays

`k` new `Float64` arrays of size `dims` holding `value` everywhere ([`Fill`](@ref), one
run on Julia threads).
"""
function filled_maps(value::Float64, k::Integer, dims::Dims)
    maps = [Array{Float64}(undef, dims) for _ in 1:k]
    host_epilogue!(Fill(value), Tuple(maps), prod(dims))
    return maps
end
