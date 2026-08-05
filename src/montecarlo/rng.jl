# Monte Carlo — rng.jl
# Counter-based, stateless, device-safe random-number generation for the Monte
# Carlo kernels: every number is a pure function of a (key, counter) pair, so
# the same code runs bit-identically on the CPU and inside a KernelAbstractions
# kernel, and each superphoton owns an independent, reproducible stream. The
# generator is Philox4x32-10 (Salmon et al. 2011), implemented inline. The
# higher-level samplers (isotropic direction, chi^2, Gamma) are clean-room
# reimplementations of the algorithms gpumonty's curand.cu uses (no GPL source
# copied), in a pure state-threading style: each returns `(value, rng')`.

# ============================================================ Philox4x32 core

const _PHILOX_M0 = 0xD2511F53
const _PHILOX_M1 = 0xCD9E8D57
const _PHILOX_W0 = 0x9E3779B9   # golden-ratio Weyl increment for key word 0
const _PHILOX_W1 = 0xBB67AE85   # sqrt(3)-based Weyl increment for key word 1

# 32x32 -> 64 multiply split into high/low 32-bit halves (device-safe: plain
# UInt64 arithmetic, no intrinsics).
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
    philox4x32(ctr, key) -> NTuple{4,UInt32}

Philox4x32-10 counter-based bijection: maps the 128-bit counter `ctr`
(`NTuple{4,UInt32}`) and 64-bit `key` (`NTuple{2,UInt32}`) to four uniformly
distributed 32-bit words. Pure, allocation-free, `isbits`-in/`isbits`-out.
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
    philox_rand(key, ctr) -> Float64 in [0, 1)

One uniform double from the Philox stream `key` (`NTuple{2,UInt32}`) at counter
`ctr` (`NTuple{4,UInt32}`). Two output words are combined into a ~53-bit mantissa
so the result is strictly in `[0, 1)` (max value `1 - 2^-64`). Pure/device-safe.
"""
@inline function philox_rand(key::NTuple{2,UInt32}, ctr::NTuple{4,UInt32})
    c = philox4x32(ctr, key)
    # top word gives the leading 32 bits, second word refines the tail
    return (Float64(c[1]) + Float64(c[2]) * 0x1p-32) * 0x1p-32
end

"Convenience seeding: `philox_rand(global_seed, stream_id, draw)`."
@inline function philox_rand(global_seed::Integer, stream_id::Integer, draw::Integer)
    key = (UInt32(global_seed & 0xffffffff), UInt32(stream_id & 0xffffffff))
    d = UInt64(draw)
    ctr = (UInt32(d & 0xffffffff), UInt32(d >> 32), 0x00000000, 0x00000000)
    return philox_rand(key, ctr)
end

# ============================================================ RNG stream handle

"""
    MCRng(key::NTuple{2,UInt32}, ctr::UInt32)

Immutable, `isbits` handle for a single superphoton's random stream. `key`
identifies the stream (typically `(global_seed, superphoton_id)`); `ctr` is the
next draw index. All samplers are pure functions of an `MCRng` and return the
advanced handle, so no mutable/global state is ever touched — the same call runs
on the CPU and inside a KA kernel.
"""
struct MCRng
    key::NTuple{2,UInt32}
    ctr::UInt32
end

"`MCRng(global_seed, stream_id [, ctr])` — start a stream at counter `ctr` (default 0)."
@inline MCRng(global_seed::Integer, stream_id::Integer, ctr::Integer = 0) =
    MCRng((UInt32(global_seed & 0xffffffff), UInt32(stream_id & 0xffffffff)), UInt32(ctr))

"""
    mc_uniform(rng) -> (Float64 in [0,1), MCRng)

Draw one uniform and return it together with the counter-advanced stream.
"""
@inline function mc_uniform(rng::MCRng)
    ctr = (rng.ctr, 0x00000000, 0x00000000, 0x00000000)
    u = philox_rand(rng.key, ctr)
    return u, MCRng(rng.key, rng.ctr + 0x00000001)
end

"""
    mc_gauss(rng) -> (Float64, MCRng)

One standard-normal deviate via the Box–Muller transform (rejection-free, so no
warp divergence). `1 - u1` keeps the logarithm finite for `u1 -> 0`.
"""
@inline function mc_gauss(rng::MCRng)
    u1, rng = mc_uniform(rng)
    u2, rng = mc_uniform(rng)
    g = sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * π * u2)
    return g, rng
end

# ============================================================ direction sampling

"""
    sample_direction(rng) -> (SVector{3,Float64}, MCRng)

Isotropic unit vector on S^2. Uses the direct inversion `z = 2u1 - 1`,
`phi = 2 pi u2` (rejection-free, unlike gpumonty's polar method) so it never
diverges across a warp. `|v| = 1` exactly up to round-off.
"""
@inline function sample_direction(rng::MCRng)
    u1, rng = mc_uniform(rng)
    u2, rng = mc_uniform(rng)
    z = 2.0 * u1 - 1.0
    r = sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * π * u2
    return SVector(r * cos(phi), r * sin(phi), z), rng
end

# ============================================================ Gamma / chi-square

# Guard so the (extremely rare) rejection stall cannot hang a warp; Marsaglia–Tsang
# acceptance is > 95% per iteration, so the cap is never reached in practice.
const _GAMMA_MAX_ITER = 1000

"""
    standard_gamma(shape, rng) -> (Float64, MCRng)

A Gamma(shape, 1) deviate. `shape >= 1` uses Marsaglia & Tsang (2000) — one
normal + one uniform per trial, ~96%+ acceptance. `0 < shape < 1` uses the exact
boosting identity `Gamma(a) = Gamma(a+1) * U^{1/a}` (rejection-free), which is
friendlier to SIMT than gpumonty's nested Ahrens–Dieter branch. Clean-room,
device-safe counterpart of curand.cu `legacy_standard_gamma` (algorithm per
Marsaglia & Tsang 2000; no GPL source copied).
"""
@inline function standard_gamma(shape::Real, rng::MCRng)
    a = Float64(shape)
    if a < 1.0
        # boost: draw Gamma(a+1) then scale by U^{1/a}
        g, rng = _gamma_ge1(a + 1.0, rng)
        u, rng = mc_uniform(rng)
        return g * u^(1.0 / a), rng
    end
    return _gamma_ge1(a, rng)
end

# Marsaglia–Tsang squeeze for shape >= 1.
@inline function _gamma_ge1(a::Float64, rng::MCRng)
    d = a - 1.0 / 3.0
    c = 1.0 / sqrt(9.0 * d)
    result = d                     # sensible fallback if the cap is ever hit
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
    chi_square(df, rng) -> (Float64, MCRng)

A chi-square deviate with `df` degrees of freedom via `2 * Gamma(df/2, 1)`
(the same mathematical identity curand.cu's `chi_square` uses); used by the
relativistic-Maxwellian electron sampler.
"""
@inline function chi_square(df::Integer, rng::MCRng)
    g, rng = standard_gamma(0.5 * df, rng)
    return 2.0 * g, rng
end
