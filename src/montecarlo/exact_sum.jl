# Monte Carlo — exact_sum.jl
#
# Order-independent exact summation of Float64 values (a sparse long
# accumulator). Every finite double is x = m 2^q with an integer |m| < 2^53 and
# q >= -1074, i.e. an integer multiple of 2^-1074 below 2^1024. The accumulator
# stores that integer in XSUM_NLIMBS = 68 signed Int64 limbs of 31 bits,
# limb j (1-based) weighing 2^(31(j-1) - 1074); 68 * 31 = 2108 bits cover the
# 2098-bit range. One addend shifted to its bit position spans at most 83 bits,
# so it touches three consecutive limbs, each by less than 2^31 in magnitude.
# Adding integers is associative, so the accumulated value is independent of
# the order, the grouping (threads, chunks, devices, stripes) and the
# interleaving of atomic updates. The result is the exact sum of the addends,
# rounded once to nearest-even Float64 by `xsum_round`.
#
# Capacity: every limb changes by less than 2^31 per addend, so a cell holds
# at least 2^32 - 1 addends before a limb can overflow Int64; `xsum_round`
# checks the carry-free bound through the caller's addend count. Non-finite
# addends are counted in slot XSUM_NSLOT and round to NaN.
#
# Cost: three integer adds (atomic on a device) per addend against one float
# add. There is no scale to choose: the synchrotron weight table spans about
# 2^152 between its largest and smallest nodes, beyond any fixed-point window
# of two or three Int64 words.

const XSUM_LIMB_BITS = 31
const XSUM_NLIMBS = 68
const XSUM_NSLOT = XSUM_NLIMBS + 1          # + non-finite counter
const XSUM_MAX_ADDENDS = (Int64(1) << 32) - 1
const _XSUM_MASK = UInt64(0x7fffffff)

"""
    xsum_parts(x::Float64) -> (j, a, b, c)

Split `x` into limb index `j` (1-based; `j == 0` for zero, `j == XSUM_NSLOT`
for a non-finite value) and the three signed limb increments `a, b, c` for limbs
`j, j+1, j+2`. Pure, branch-light integer arithmetic; device-safe.
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
    xsum_add!(acc, x, I...) -> acc

Add `x` into the exact accumulator stored along the first axis of `acc` at the
trailing index `I` (`acc[:, I...]` has `XSUM_NSLOT` limbs). Host version (no
atomics); `_xsum_atomic_add!` in mc_gpu.jl is the device twin.
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
    xsum_round(limbs) -> Float64

Round the exact accumulated value in the `XSUM_NSLOT` limbs to nearest-even
Float64 (NaN if any non-finite value was added, ±Inf past floatmax).
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

# Round v 2^sh (v a BigInt) to nearest-even Float64.
function _xsum_round_int(v::BigInt, sh::Int)
    v == 0 && return 0.0
    neg = v < 0
    v = abs(v)
    n = ndigits(v; base = 2)
    # position of the result's last kept bit: 53 significant bits, but never
    # below 2^-1074 (subnormal range)
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

"`xsum_round(acc, I...)`: round the accumulator cell `acc[:, I...]`."
xsum_round(acc::AbstractArray{Int64}, I::Vararg{Int}) = xsum_round(view(acc, :, I...))

"Exact zero-initialised accumulator array: `XSUM_NSLOT` limbs per cell of `dims`."
xsum_zeros(dims::Vararg{Int}) = zeros(Int64, XSUM_NSLOT, dims...)

"""
    xsum_round_all(acc) -> Array{Float64}

Round every cell of `acc` (limbs along axis 1) to Float64; the result has the
trailing shape of `acc`.
"""
function xsum_round_all(acc::AbstractArray{Int64})
    out = zeros(Float64, size(acc)[2:end]...)
    for I in CartesianIndices(out)
        out[I] = xsum_round(view(acc, :, Tuple(I)...))
    end
    return out
end

# ============================================================ exact spectrum

"""
    ExactSpectrum

Exact, order-independent twin of `Spectrum`: the raw count plus `Sum w` and
`Sum w E` held as `xsum` limbs per (angle, energy) cell. Merging adds limbs,
so any partition of the photons into chunks or devices gives the same bits;
`Spectrum(xs)` rounds once per cell.
"""
struct ExactSpectrum
    nu_min::Float64
    nu_max::Float64
    lnu_min::Float64
    dlnu::Float64
    n_ebins::Int
    n_thbins::Int
    count::Matrix{Int}
    sumw::Array{Int64,3}
    sumwE::Array{Int64,3}
end

function ExactSpectrum(; nu_min::Real = _MC_NU_MIN_DEFAULT, nu_max::Real = _MC_NU_MAX_DEFAULT,
                       n_ebins::Integer = 200, n_thbins::Integer = 6)
    g = Spectrum(; nu_min = nu_min, nu_max = nu_max, n_ebins = n_ebins, n_thbins = n_thbins)
    return ExactSpectrum(g.nu_min, g.nu_max, g.lnu_min, g.dlnu, g.n_ebins, g.n_thbins,
                         zeros(Int, n_thbins, n_ebins),
                         xsum_zeros(n_thbins, n_ebins), xsum_zeros(n_thbins, n_ebins))
end

"Add one escaped packet (weight `w`, energy `E`) to cell `(ith, ie)`."
@inline function _xs_add!(xs::ExactSpectrum, ith::Int, ie::Int, w::Float64, E::Float64)
    @inbounds xs.count[ith, ie] += 1
    xsum_add!(xs.sumw, w, ith, ie)
    xsum_add!(xs.sumwE, w * E, ith, ie)
    return xs
end

"`record!` into an `ExactSpectrum`: the same (angle, energy) binning as `Spectrum`."
@inline function record!(xs::ExactSpectrum, sph::Superphoton)
    E = sph.E
    E > 0.0 || return xs
    nu = E * ME * CL * CL / HPL
    ie = floor(Int, (log(nu) - xs.lnu_min) / xs.dlnu) + 1
    (ie < 1 || ie > xs.n_ebins) && return xs
    p = sph.p
    ps = sqrt(p[2] * p[2] + p[3] * p[3] + p[4] * p[4])
    mu = ps > 0.0 ? abs(p[4]) / ps : 0.0
    thf = acos(clamp(mu, 0.0, 1.0))
    ith = clamp(floor(Int, thf / ((0.5 * π) / xs.n_thbins)) + 1, 1, xs.n_thbins)
    return _xs_add!(xs, ith, ie, sph.w, E)
end

"`record_energy!` into an `ExactSpectrum` (row 1, energy only)."
@inline function record_energy!(xs::ExactSpectrum, sph::Superphoton)
    E = sph.E
    E > 0.0 || return xs
    nu = E * ME * CL * CL / HPL
    ie = floor(Int, (log(nu) - xs.lnu_min) / xs.dlnu) + 1
    (ie < 1 || ie > xs.n_ebins) && return xs
    return _xs_add!(xs, 1, ie, sph.w, E)
end

"Add the limbs and counts of `src` into `dst` (exact, associative)."
function merge_spectra!(dst::ExactSpectrum, src::ExactSpectrum)
    dst.count .+= src.count
    dst.sumw .+= src.sumw
    dst.sumwE .+= src.sumwE
    return dst
end

"Round an `ExactSpectrum` to a `Spectrum` (one rounding per cell)."
function Spectrum(xs::ExactSpectrum)
    maximum(xs.count; init = 0) <= XSUM_MAX_ADDENDS ||
        error("ExactSpectrum cell exceeds $(XSUM_MAX_ADDENDS) addends")
    return Spectrum(xs.nu_min, xs.nu_max, xs.lnu_min, xs.dlnu, xs.n_ebins, xs.n_thbins,
                    copy(xs.count), xsum_round_all(xs.sumw), xsum_round_all(xs.sumwE))
end

"""
    ExactScalars(n)

`n` exact scalar accumulators (diagnostic totals), with an addend counter for
the capacity check.
"""
struct ExactScalars
    acc::Matrix{Int64}
    n::Vector{Int}
end
ExactScalars(n::Integer) = ExactScalars(xsum_zeros(Int(n)), zeros(Int, n))

@inline function xsum_add!(s::ExactScalars, k::Int, x::Float64)
    xsum_add!(s.acc, x, k)
    @inbounds s.n[k] += 1
    return s
end

function merge_scalars!(dst::ExactScalars, src::ExactScalars)
    dst.acc .+= src.acc
    dst.n .+= src.n
    return dst
end

function xsum_round(s::ExactScalars, k::Int)
    s.n[k] <= XSUM_MAX_ADDENDS || error("exact scalar $k exceeds $(XSUM_MAX_ADDENDS) addends")
    return xsum_round(view(s.acc, :, k))
end
