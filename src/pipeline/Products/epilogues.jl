# The host epilogues of the image products: the band sums, the NaN and zero maps and the
# planes of a Stokes-vector image, as programs over the items of host arrays. They run on
# Julia threads whatever executor ran the product, since their inputs and outputs are on
# the host; every item reads and writes its own elements only, so the result is the same
# on any executor and in any order. A product given `into`, an earlier result, writes its
# maps and these epilogues into that result's arrays instead of new ones.

"""
    BandSum(sums)

Band sums of per-band maps: `sums` holds, per output `k`, a tuple of band maps
`(m₁, …, m_K)`, and item `item` writes `outs[k][i] = ((0 + m₁[item]) + m₂[item]) + … +
m_K[item]`, the bands added in order starting from zero, which gives the bits of
`I = zeros(…)` followed by `I .+= m_b` band after band. Called as `prog(item, i, outs...)`
with one output per entry of `sums`. The maps of one sum share their size (else a
`DimensionMismatch`), and an item outside them is a `BoundsError`.
"""
struct BandSum{S<:Tuple} <: Function
    sums::S
    function BandSum(sums::S) where {S<:Tuple}
        for maps in sums, m in maps
            axes(m) == axes(first(maps)) || throw(DimensionMismatch(
                "BandSum: band maps of sizes $(size(first(maps))) and $(size(m)) in one sum"))
        end
        return new{S}(sums)
    end
end
Adapt.@adapt_structure BandSum

@inline add_bands(::Tuple{}, item, acc) = acc
@inline add_bands(maps::Tuple, item, acc) =
    add_bands(Base.tail(maps), item, acc + first(maps)[item])

@inline store_sums!(::Tuple{}, item, i, ::Tuple{}) = nothing
@inline function store_sums!(sums::Tuple, item, i, outs::Tuple)
    o = first(outs)
    s = add_bands(first(sums), item, zero(eltype(o)))
    @inbounds o[i] = s
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
`prog(item, i, outs...)` with one output per component; an item outside `src` is a
`BoundsError`.
"""
struct Split{V} <: Function
    src::V
end
Adapt.@adapt_structure Split

function (p::Split)(item, i, outs::Vararg{Any,N}) where {N}
    v = p.src[item]
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
run on Julia threads). A band map of another size is a `DimensionMismatch`.
"""
function band_sums(groups::Tuple, dims::Dims)
    check_band_sizes(groups, dims)
    return band_sums!(map(_ -> Array{Float64}(undef, dims), groups), groups)
end

"""
    band_sums!(outs, groups) -> outs

[`band_sums`](@ref) into the existing arrays `outs`, one per group, all of one size.
"""
function band_sums!(outs::Tuple, groups::Tuple)
    length(outs) == length(groups) ||
        throw(ArgumentError("band_sums!: $(length(outs)) outputs for $(length(groups)) groups"))
    isempty(outs) && return outs
    dims = size(first(outs))
    all(o -> size(o) == dims, outs) ||
        throw(DimensionMismatch("band_sums!: outputs of sizes $(map(size, outs))"))
    check_band_sizes(groups, dims)
    host_epilogue!(BandSum(map(Tuple, groups)), outs, prod(dims))
    return outs
end

# Every band map of `groups` has the size `dims`, else a `DimensionMismatch`.
function check_band_sizes(groups::Tuple, dims::Dims)
    for g in groups, m in g
        size(m) == dims ||
            throw(DimensionMismatch("band_sums: a band map of size $(size(m)) for size $dims"))
    end
end

"""
    filled_maps(value, k, dims) -> Vector of arrays

`k` new `Float64` arrays of size `dims` holding `value` everywhere ([`Fill`](@ref), one
run on Julia threads).
"""
filled_maps(value::Float64, k::Integer, dims::Dims) =
    fill_maps!(value, [Array{Float64}(undef, dims) for _ in 1:k])

"""
    fill_maps!(value, maps) -> maps

`value` into every element of the existing arrays `maps`, all of one size ([`Fill`](@ref),
one run on Julia threads).
"""
function fill_maps!(value::Float64, maps)
    isempty(maps) && return maps
    n = length(first(maps))
    all(m -> length(m) == n, maps) ||
        throw(DimensionMismatch("fill_maps!: maps of sizes $(map(size, maps))"))
    host_epilogue!(Fill(value), Tuple(maps), n)
    return maps
end

"""
    claim_into(into, product, single, banded, K, dims; extra = ()) -> NamedTuple

The arrays of `into`, a result of `product` that a call overwrites (its keyword `into`):
the fields `single` (one array each) and `banded` (a vector of `K` arrays each) of `into`,
each an `Array{Float64}` of size `dims`, and the fields of `extra` (`field => size` pairs),
no two sharing memory, else an `ArgumentError` naming the field. Their device copies are
dropped from the pools' residency caches ([`Execution.invalidate!`](@ref)), so an array that
an earlier run uploaded is uploaded again with its new values.
"""
function claim_into(into, product::Symbol, single::Tuple, banded::Tuple, K::Integer, dims::Dims;
                    extra::Tuple = ())
    field(f) = hasproperty(into, f) ? getproperty(into, f) :
        throw(ArgumentError("$product: `into` has no field `$f`; pass a result of $product"))
    function check(m, what, d)
        m isa Array{Float64} && size(m) == d ||
            throw(ArgumentError("$product: into.$what is $(summary(m)), not a " *
                                "$(join(d, " × ")) Array{Float64}"))
        return m
    end
    function check_bands(f)
        ms = field(f)
        ms isa AbstractVector && length(ms) == K ||
            throw(ArgumentError("$product: into.$f is $(summary(ms)); the call writes $K maps"))
        foreach(b -> check(ms[b], "$f[$b]", dims), eachindex(ms))
        return ms
    end
    got = merge(NamedTuple{single}(map(f -> check(field(f), f, dims), single)),
                NamedTuple{banded}(map(check_bands, banded)),
                NamedTuple{map(first, extra)}(map(e -> check(field(e[1]), e[1], e[2]), extra)))
    arrays = Any[(got[f] for f in single)..., (m for f in banded for m in got[f])...,
                 (got[first(e)] for e in extra)...]
    for i in eachindex(arrays), j in 1:i-1
        Base.mightalias(arrays[i], arrays[j]) &&
            throw(ArgumentError("$product: two arrays of `into` share memory"))
    end
    Execution.invalidate!(arrays)
    return got
end

# The maps of `fields` of `m` (a `claim_into` result), band after band, as vectors sharing
# their memory: the output tuple of a program writing those quantities quantity-major.
flat_outputs(m, fields::Tuple) = Tuple(vec(x) for f in fields for x in getproperty(m, f))
