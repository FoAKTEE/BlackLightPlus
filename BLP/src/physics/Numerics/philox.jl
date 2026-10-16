# Counter-based random numbers: Philox4x32-10 (Salmon et al. 2011) and the samplers of
# the Monte Carlo transport. Every draw is a pure function of a (key, counter) pair, so a
# stream is reproducible on any thread or device. Samplers thread the stream: each
# returns `(value, rng')`. The Gamma and chi-square algorithms are those of gpumonty
# `curand.cu` (Marsaglia & Tsang 2000), written independently of its GPL source.

const _PHILOX_M0 = 0xD2511F53
const _PHILOX_M1 = 0xCD9E8D57
const _PHILOX_W0 = 0x9E3779B9   # golden-ratio Weyl increment of key word 1
const _PHILOX_W1 = 0xBB67AE85   # sqrt(3) Weyl increment of key word 2

# 32x32 -> 64 multiply split into high and low 32-bit halves (plain UInt64 arithmetic).
@inline function _mulhilo(a::UInt32, b::UInt32)
    prod = UInt64(a) * UInt64(b)
    return UInt32(prod >> 32), UInt32(prod & 0x00000000ffffffff)
end

@inline function _philox_round(c::NTuple{4,UInt32}, k::NTuple{2,UInt32})
    hi0, lo0 = _mulhilo(_PHILOX_M0, c[1])
    hi1, lo1 = _mulhilo(_PHILOX_M1, c[3])
    return (hi1 ⊻ c[2] ⊻ k[1], lo1, hi0 ⊻ c[4] ⊻ k[2], lo0)
end

"""
    philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}) -> NTuple{4,UInt32}

The Philox4x32-10 bijection: ten rounds mapping the 128-bit counter `ctr` under
the 64-bit `key` to four uniformly distributed 32-bit words.
"""
@inline function philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    c = ctr
    k = key
    @inbounds for i in 1:10
        if i > 1
            k = (k[1] + _PHILOX_W0, k[2] + _PHILOX_W1)
        end
        c = _philox_round(c, k)
    end
    return c
end

"""
    philox_rand(key::NTuple{2,UInt32}, ctr::NTuple{4,UInt32}) -> Float64
    philox_rand(global_seed, stream_id, draw) -> Float64

One uniform double in `[0, 1)` from the first two output words of
`philox4x32(ctr, key)` (the first word the leading 32 bits; largest value
`1 - 2^-64`). The three-integer form uses `key = (seed, stream_id)` (low 32 bits
of each) and `ctr = (draw low word, draw high word, 0, 0)`.
"""
@inline function philox_rand(key::NTuple{2,UInt32}, ctr::NTuple{4,UInt32})
    c = philox4x32(ctr, key)
    return (Float64(c[1]) + Float64(c[2]) * 0x1p-32) * 0x1p-32
end

@inline function philox_rand(global_seed::Integer, stream_id::Integer, draw::Integer)
    key = (UInt32(global_seed & 0xffffffff), UInt32(stream_id & 0xffffffff))
    d = UInt64(draw)
    ctr = (UInt32(d & 0xffffffff), UInt32(d >> 32), 0x00000000, 0x00000000)
    return philox_rand(key, ctr)
end

"""
    MCRng(key::NTuple{2,UInt32}, ctr::UInt32[, sid::NTuple{3,UInt32}])
    MCRng(global_seed, stream_id[, ctr = 0])

Immutable `isbits` handle of one random stream. Draw `n` is
`philox4x32((n, sid...), key)`: `key = (seed, identity word 1)`, the counter
words 2–4 `sid` carry the rest of the stream identity (see
[`MCStreamLayout`](@ref)), and `ctr` is the next draw index. The integer form
starts the stream `key = (seed, stream_id)` (low 32 bits) with `sid = 0`.
"""
struct MCRng
    key::NTuple{2,UInt32}
    ctr::UInt32
    sid::NTuple{3,UInt32}
end

@inline MCRng(key::NTuple{2,UInt32}, ctr::UInt32) =
    MCRng(key, ctr, (0x00000000, 0x00000000, 0x00000000))

@inline MCRng(global_seed::Integer, stream_id::Integer, ctr::Integer = 0) =
    MCRng((UInt32(global_seed & 0xffffffff), UInt32(stream_id & 0xffffffff)), UInt32(ctr))

"""
    mc_uniform(rng::MCRng) -> (Float64, MCRng)

One uniform in `[0, 1)` and the stream advanced by one draw.
"""
@inline function mc_uniform(rng::MCRng)
    ctr = (rng.ctr, rng.sid[1], rng.sid[2], rng.sid[3])
    u = philox_rand(rng.key, ctr)
    return u, MCRng(rng.key, rng.ctr + 0x00000001, rng.sid)
end

# Stream identity. A superphoton stream is named by a 128-bit identity that depends only
# on its genealogy, never on the device, chunk, thread or buffer slot:
#
#   I = gid + sum_{j=1..k} e_j 2^(G + (j-1) E) + k 2^124
#
# with `gid` in 1:N the global primary index, `k` the scattering layer (0 for a
# primary) and `e_j` the 0-based spawn ordinal of the generation-j ancestor along its
# parent's march. Bits 0-31 of I are key word 2, bits 32-127 counter words 2-4; counter
# word 1 is the draw index. With gid < 2^G, e_j < 2^E, G + K E <= 124 and K <= 15 the
# fields are disjoint, so distinct genealogies give distinct Philox inputs for every
# draw. Different seeds differ in key word 1. A primary with gid < 2^32 has I = gid,
# the stream `MCRng(seed, gid)`.

"""
    MCStreamLayout(n_primary, track_max, max_layer)

Bit widths of the stream-identity fields for a run of `n_primary` primaries, at
most `track_max` march steps (hence spawns) per photon and at most `max_layer`
scattering layers: `gid_bits` for the primary index, `event_bits` per spawn
ordinal, `max_layer`. Throws `ArgumentError` when the fields do not fit the 124
identity bits below the layer tag or `max_layer ∉ 0:15`.
"""
struct MCStreamLayout
    gid_bits::Int
    event_bits::Int
    max_layer::Int
    function MCStreamLayout(n_primary::Integer, track_max::Integer, max_layer::Integer)
        n_primary >= 1 || throw(ArgumentError("n_primary must be >= 1"))
        track_max >= 1 || throw(ArgumentError("track_max must be >= 1"))
        G = _bitwidth(n_primary)
        E = max(1, _bitwidth(track_max - 1))
        K = Int(max_layer)
        0 <= K <= 15 || throw(ArgumentError("max_layer = $K outside 0:15"))
        G + K * E <= 124 || throw(ArgumentError(
            "stream identity does not fit: $G gid bits + $K layers x $E event bits > 124"))
        return new(G, E, K)
    end
end

@inline _bitwidth(n::Integer) = 64 - leading_zeros(UInt64(n))

"""
    stream_id(rng::MCRng) -> UInt128

The 128-bit stream identity of `rng`: key word 2 and counter words 2–4.
"""
@inline stream_id(rng::MCRng) =
    UInt128(rng.key[2]) | (UInt128(rng.sid[1]) << 32) |
    (UInt128(rng.sid[2]) << 64) | (UInt128(rng.sid[3]) << 96)

"""
    stream_layer(rng::MCRng) -> Int

Scattering layer of the stream (0 for a primary), the top four identity bits.
"""
@inline stream_layer(rng::MCRng) = Int(rng.sid[3] >> 28)

@inline function _rng_from_id(key1::UInt32, id::UInt128)
    return MCRng((key1, UInt32(id & 0xffffffff)), 0x00000000,
                 (UInt32((id >> 32) & 0xffffffff), UInt32((id >> 64) & 0xffffffff),
                  UInt32((id >> 96) & 0xffffffff)))
end

"""
    primary_rng(seed, gid) -> MCRng

The stream of global primary `gid >= 1`: identity `gid`, draw counter 0.
"""
@inline primary_rng(seed::Integer, gid::Integer) =
    _rng_from_id(UInt32(seed & 0xffffffff), UInt128(gid))

"""
    child_rng(parent::MCRng, event, layout::MCStreamLayout) -> MCRng

The stream of the child spawned at 0-based ordinal `event` along the march of
the photon owning `parent` (at any draw count): `event` goes into the next
event field of the parent's identity and the layer tag is incremented. Depends
only on the genealogy.
"""
@inline function child_rng(parent::MCRng, event::Integer, L::MCStreamLayout)
    id = stream_id(parent)
    k = id >> 124
    pos = L.gid_bits + Int(k) * L.event_bits
    cid = (id & ((UInt128(1) << 124) - UInt128(1))) | (UInt128(event) << pos) |
          ((k + UInt128(1)) << 124)
    return _rng_from_id(parent.key[1], cid)
end

"""
    mc_gauss(rng::MCRng) -> (Float64, MCRng)

One standard-normal deviate by the Box–Muller transform of two uniforms;
`1 - u1` keeps the logarithm finite as `u1 → 0`.
"""
@inline function mc_gauss(rng::MCRng)
    u1, rng = mc_uniform(rng)
    u2, rng = mc_uniform(rng)
    g = sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * π * u2)
    return g, rng
end

"""
    sample_direction(rng::MCRng) -> (SVector{3,Float64}, MCRng)

An isotropic unit vector by direct inversion, `z = 2u₁ - 1`, `φ = 2π u₂`
(rejection-free); unit length up to round-off.
"""
@inline function sample_direction(rng::MCRng)
    u1, rng = mc_uniform(rng)
    u2, rng = mc_uniform(rng)
    z = 2.0 * u1 - 1.0
    r = sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * π * u2
    return SVector(r * cos(phi), r * sin(phi), z), rng
end

# Iteration cap of the Marsaglia-Tsang rejection loop; its acceptance exceeds 95% per
# trial, so the cap only guards against a stall.
const _GAMMA_MAX_ITER = 1000

"""
    standard_gamma(shape, rng::MCRng) -> (Float64, MCRng)

A Gamma(`shape`, 1) deviate: Marsaglia & Tsang (2000) for `shape ≥ 1`, and for
`0 < shape < 1` the identity `Gamma(a) = Gamma(a + 1) U^{1/a}`.
"""
@inline function standard_gamma(shape::Real, rng::MCRng)
    a = Float64(shape)
    if a < 1.0
        g, rng = _gamma_ge1(a + 1.0, rng)
        u, rng = mc_uniform(rng)
        return g * u^(1.0 / a), rng
    end
    return _gamma_ge1(a, rng)
end

# Marsaglia-Tsang squeeze for shape >= 1; returns d if the iteration cap is reached.
@inline function _gamma_ge1(a::Float64, rng::MCRng)
    d = a - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    result = d
    @inbounds for _ in 1:_GAMMA_MAX_ITER
        x, rng = mc_gauss(rng)
        v = 1.0 + c * x
        if v > 0.0
            v = v * v * v
            u, rng = mc_uniform(rng)
            x2 = x * x
            if u < 1.0 - 0.0331 * x2 * x2
                result = d * v
                break
            end
            if log(u) < 0.5 * x2 + d * (1.0 - v + log(v))
                result = d * v
                break
            end
        end
    end
    return result, rng
end

"""
    chi_square(df::Integer, rng::MCRng) -> (Float64, MCRng)

A chi-square deviate with `df` degrees of freedom, `2 Gamma(df/2, 1)`.
"""
@inline function chi_square(df::Integer, rng::MCRng)
    g, rng = standard_gamma(0.5 * df, rng)
    return 2.0 * g, rng
end
