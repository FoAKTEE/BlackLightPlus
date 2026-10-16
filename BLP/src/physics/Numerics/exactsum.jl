# Order-independent exact summation of Float64 values in integer limbs.
#
# Every finite double is x = m 2^q with an integer |m| < 2^53 and q >= -1074, an integer
# multiple of 2^-1074 below 2^1024. An accumulator cell stores that integer in
# XSUM_NLIMBS = 68 signed Int64 limbs of 31 bits, limb j (1-based) weighing
# 2^(31(j-1) - 1074); 68 * 31 = 2108 bits cover the 2098-bit range. One addend shifted
# to its bit position spans at most 83 bits, so it touches three consecutive limbs, each
# by less than 2^31 in magnitude. Integer addition is associative, so the accumulated
# value does not depend on order, grouping or the interleaving of atomic updates; it is
# the exact sum, rounded once to nearest-even Float64 by `xsum_round`. A limb changes by
# less than 2^31 per addend, so a cell holds at least XSUM_MAX_ADDENDS = 2^32 - 1
# addends before a limb can overflow. Non-finite addends are counted in slot XSUM_NSLOT.

"Bits per limb of the exact accumulator."
const XSUM_LIMB_BITS = 31
"Number of value limbs per accumulator cell."
const XSUM_NLIMBS = 68
"Slots per accumulator cell: the value limbs plus the non-finite counter."
const XSUM_NSLOT = XSUM_NLIMBS + 1
"Addends per cell below which no limb can overflow."
const XSUM_MAX_ADDENDS = (Int64(1) << 32) - 1
const _XSUM_MASK = UInt64(0x7fffffff)

"""
    xsum_parts(x::Float64) -> (j, a, b, c)

Split `x` into the 1-based limb index `j` and the three signed increments
`a, b, c` of limbs `j, j+1, j+2`. `j == 0` for zero (nothing to add);
`j == XSUM_NSLOT` for a non-finite value, with `a == 1` its count. Pure integer
arithmetic; device-safe.
"""
@inline function xsum_parts(x::Float64)
    u = reinterpret(UInt64, x)
    e = Int((u >> 52) & 0x7ff)
    f = u & 0x000fffffffffffff
    e == 0x7ff && return (XSUM_NSLOT, Int64(1), Int64(0), Int64(0))
    m = e == 0 ? f : (f | 0x0010000000000000)
    m == 0 && return (0, Int64(0), Int64(0), Int64(0))
    s = e == 0 ? 0 : e - 1                   # bit offset of m's LSB above 2^-1074
    j = s ÷ XSUM_LIMB_BITS
    r = s - j * XSUM_LIMB_BITS               # 0 <= r <= 30
    a = (m << r) & _XSUM_MASK
    rest = m >> (XSUM_LIMB_BITS - r)         # < 2^52
    b = rest & _XSUM_MASK
    c = rest >> XSUM_LIMB_BITS               # < 2^21
    neg = (u >> 63) != 0
    ia = reinterpret(Int64, a); ib = reinterpret(Int64, b); ic = reinterpret(Int64, c)
    return neg ? (j + 1, -ia, -ib, -ic) : (j + 1, ia, ib, ic)
end

"""
    xsum_add!(acc, x::Float64, I...) -> acc

Add `x` into the exact accumulator cell `acc[:, I...]` (`XSUM_NSLOT` limbs along
the first axis), without atomics. See [`xsum_atomic_add!`](@ref) for concurrent
writers.
"""
@inline function xsum_add!(acc::AbstractArray{Int64}, x::Float64, I::Vararg{Int})
    j, a, b, c = xsum_parts(x)
    j == 0 && return acc
    @inbounds if j == XSUM_NSLOT
        acc[XSUM_NSLOT, I...] += a
    else
        acc[j, I...] += a
        acc[j + 1, I...] += b
        acc[j + 2, I...] += c
    end
    return acc
end

"""
    xsum_atomic_add!(acc, x::Float64, I...) -> nothing

[`xsum_add!`](@ref) for concurrent writers: at most three atomic Int64 additions
into the limbs of cell `acc[:, I...]` (zero increments are skipped). Integer
atomics commute, so the cell value does not depend on the arrival order. Runs on
host arrays and, inside a kernel, on device arrays.
"""
@inline function xsum_atomic_add!(acc, x::Float64, I::Vararg{Int})
    j, p1, p2, p3 = xsum_parts(x)
    j == 0 && return nothing
    if j == XSUM_NSLOT
        Atomix.@atomic acc[j, I...] += p1
    else
        p1 == 0 || (Atomix.@atomic acc[j, I...] += p1)
        p2 == 0 || (Atomix.@atomic acc[j + 1, I...] += p2)
        p3 == 0 || (Atomix.@atomic acc[j + 2, I...] += p3)
    end
    return nothing
end

"""
    xsum_round(limbs::AbstractVector{Int64}) -> Float64
    xsum_round(acc, I...) -> Float64

Round the exact value held in the `XSUM_NSLOT` limbs (or in cell `acc[:, I...]`)
to nearest-even Float64: NaN if a non-finite value was added, ±Inf beyond
`floatmax`. Host only (arbitrary-precision integers).
"""
function xsum_round(limbs::AbstractVector{Int64})
    limbs[XSUM_NSLOT] == 0 || return NaN
    hi = findlast(!iszero, view(limbs, 1:XSUM_NLIMBS))
    hi === nothing && return 0.0
    lo = findfirst(!iszero, view(limbs, 1:XSUM_NLIMBS))
    v = big(0)
    for j in hi:-1:lo
        v = (v << XSUM_LIMB_BITS) + limbs[j]
    end
    return _xsum_round_int(v, XSUM_LIMB_BITS * (lo - 1) - 1074)
end
xsum_round(acc::AbstractArray{Int64}, I::Vararg{Int}) = xsum_round(view(acc, :, I...))

# Round v 2^sh (v a BigInt) to nearest-even Float64. The last kept bit is the 53rd
# significant bit, never below 2^-1074 (subnormal range).
function _xsum_round_int(v::BigInt, sh::Int)
    v == 0 && return 0.0
    neg = v < 0
    v = abs(v)
    n = ndigits(v; base = 2)
    drop = max(n - 53, -1074 - sh)
    if drop > 0
        q = v >> drop
        rem = v - (q << drop)
        half = big(1) << (drop - 1)
        (rem > half || (rem == half && isodd(q))) && (q += 1)
        v = q
        sh += drop
    end
    r = ldexp(Float64(v), sh)                # v <= 2^53: exact unless overflow
    return neg ? -r : r
end

"""
    xsum_zeros(dims...) -> Array{Int64}

A zeroed exact accumulator: `XSUM_NSLOT` limbs along the first axis for every
cell of the trailing shape `dims`.
"""
xsum_zeros(dims::Vararg{Int}) = zeros(Int64, XSUM_NSLOT, dims...)

"""
    xsum_round_all(acc) -> Array{Float64}

Every cell of `acc` (limbs along the first axis) rounded by
[`xsum_round`](@ref); the result has the trailing shape of `acc`.
"""
function xsum_round_all(acc::AbstractArray{Int64})
    out = zeros(Float64, size(acc)[2:end]...)
    for I in CartesianIndices(out)
        out[I] = xsum_round(view(acc, :, Tuple(I)...))
    end
    return out
end
