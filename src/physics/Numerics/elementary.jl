# The Float64 exponential and logarithm with the same bits on every device: Base's
# table-driven algorithms (Tang 1989, 1990) written out with an explicit fused multiply-add
# wherever Base writes `muladd`, and compiled into the caller. Every operation is an
# IEEE-754 addition, multiplication, division or fma rounded to nearest, which a CPU and a
# GPU compute alike, and the tables are Base's: on a host with hardware fma (where Base's
# `muladd` fuses) `tang_exp(x) === exp(x)` and `tang_log(x) === log(x)`, and a device
# kernel, whose `exp` and `log` are the vendor library's, gets the host's bits.

const _EXP_TABLE = Base.Math.J_TABLE
const _LOG_TABLE = Base.Math.t_log_Float64

"""
    tang_exp(x::Float64) -> Float64

`exp(x)` by Base's algorithm (reduction `x = (k + j/256) ln 2 + r`, a cubic for
`e^r − 1`, the table of `2^{j/256}`), every `muladd` an explicit `fma`: bitwise
`Base.exp` on an fma host, and the same bits in a device kernel. Device-safe.
"""
@inline function tang_exp(x::Float64)
    N_float = fma(x, 369.3299304675746, 6.755399441055744e15)
    N = reinterpret(UInt64, N_float) % Int32
    N_float -= 6.755399441055744e15
    r = fma(N_float, -0.002707606173999011, x)
    r = fma(N_float, -6.327543041662719e-14, r)
    k = N >> 8
    j = @inbounds _EXP_TABLE[(N & 255) + 1]
    jU = reinterpret(Float64, 0x3ff0000000000000 | (j & (typemax(UInt64) >> 12)))
    jL = reinterpret(Float64, 0x3c00000000000000 | (j >> 8))
    p = r * fma(r, fma(r, fma(r, 0.04166666857598777, 0.1666666857598779),
                       0.4999999999999997), 0.9999999999999912)
    small_part = fma(jU, p, jL) + jU
    if !(abs(x) <= 708.3964185322641)
        isnan(x) && return x
        x >= 709.7827128933841 && return Inf
        x <= -745.1332191019412 && return 0.0
        if k <= -53
            twopk = (k + UInt64(53)) << 52
            return reinterpret(Float64, twopk + reinterpret(UInt64, small_part)) * 0x1p-53
        end
    end
    twopk = Int64(k) << 52
    return reinterpret(Float64, twopk + reinterpret(Int64, small_part))
end

"""
    tang_log(x::Float64) -> Float64

`log(x)` by Base's algorithm (Tang's procedure 2 within about 6 % of 1, procedure 1
with the 129-node table of `log(1 + j/128)` elsewhere), every `muladd` an explicit
`fma`: bitwise `Base.log` on an fma host for `x ≥ 0`, and the same bits in a device
kernel. `NaN` for `x < 0` (where Base throws). Device-safe.
"""
@inline function tang_log(x::Float64)
    if x > 0.0
        x == Inf && return x
        0.9394130628134757 < x < 1.0644944589178595 && return _tang_log2(x - 1.0)
        xu = reinterpret(UInt64, x)
        m = Int(xu >> 52) & 0x07ff
        if m == 0
            x *= 1.8014398509481984e16
            xu = reinterpret(UInt64, x)
            m = Int(xu >> 52) & 0x07ff - 54
        end
        m -= 1023
        y = reinterpret(Float64, (xu & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000)
        mf = Float64(m)
        F = (y + 3.5184372088832e13) - 3.5184372088832e13
        return _tang_log1(y, mf, F, y - F)
    end
    return x == 0.0 ? -Inf : NaN
end

# Procedure 1: log(2^m F (1 + f/F)) = m ln 2 + log F + log(1 + f/F), F = 1 + j/128.
@inline function _tang_log1(y::Float64, mf::Float64, F::Float64, f::Float64)
    jp = unsafe_trunc(Int, 128.0 * F) - 127
    hi, lo = @inbounds _LOG_TABLE[jp]
    l_hi = mf * 0.6931471805601177 + hi
    l_lo = mf * -1.7239444525614835e-13 + lo
    u = (2.0 * f) / (y + F)
    v = u * u
    q = u * v * fma(v, 0.012500053168098584, 0.08333333333303913)
    return fma(1.0, l_hi, fma(1.0, (u + (q + l_lo)), 0.0 * l_hi))
end

# Procedure 2: log(1 + f) for |f| below about 0.06.
@inline function _tang_log2(f::Float64)
    g = 1.0 / (2.0 + f)
    u = 2.0 * f * g
    v = u * u
    q = u * v * fma(v, fma(v, fma(v, 0.0004348877777076146, 0.0022321399879194482),
                           0.012500000003771751), 0.08333333333333179)
    return fma(1.0, u, fma(0.0, u, 1.0 * fma(fma(-u, f, 2.0 * (f - u)), g, q)))
end
