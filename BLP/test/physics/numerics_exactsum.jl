# The exact accumulator gives the correctly rounded sum, bitwise independent of order,
# of chunked merging and of atomic versus plain addition.
using Test, BLP, Random
using BLP.Numerics: xsum_zeros, xsum_add!, xsum_atomic_add!, xsum_round, xsum_round_all,
                    xsum_parts, XSUM_NSLOT, XSUM_NLIMBS

xe_bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
xe_bits(x::Real) = reinterpret(UInt64, Float64(x))

# BigFloat wide enough to hold any sum of Float64s exactly.
xe_ref(v) = setprecision(BigFloat, 2400) do
    Float64(sum(BigFloat.(v)))
end

function xe_sum(v)
    acc = xsum_zeros(1)
    for x in v
        xsum_add!(acc, x, 1)
    end
    return acc
end

@testset "exact accumulator" begin
    rng = Xoshiro(20260923)
    # 10^6 positive weights over ~2^±150 (the span of the synchrotron weight table)
    w = exp.(50.0 .* randn(rng, 10^6))
    ref = xe_ref(w)
    b0 = xsum_round(xe_sum(w), 1)
    @test b0 == ref
    same = true
    for _ in 1:20
        same &= xe_bits(xsum_round(xe_sum(view(w, randperm(rng, length(w)))), 1)) == xe_bits(b0)
    end
    @test same                                       # random orders: identical bits
    for nch in (2, 7, 64)                            # chunks merged by integer addition
        acc = xsum_zeros(1)
        for c in 1:nch
            lo = div((c - 1) * length(w), nch) + 1; hi = div(c * length(w), nch)
            acc .+= xe_sum(view(w, lo:hi))
        end
        @test xe_bits(xsum_round(acc, 1)) == xe_bits(b0)
    end
    # the atomic add fills the same limbs, from concurrent writers
    acc = xsum_zeros(3)
    Threads.@threads for i in eachindex(w)
        xsum_atomic_add!(acc, w[i], 1 + i % 3)
    end
    plain = xsum_zeros(3)
    for i in eachindex(w)
        xsum_add!(plain, w[i], 1 + i % 3)
    end
    @test acc == plain
    @test xsum_round_all(acc) == [xsum_round(plain, k) for k in 1:3]

    # signed, subnormal, extreme and cancelling addends
    edge = [floatmax(), -floatmax(), 5.0e-324, 2.5e-308, -1.0, 1.0 + eps(), 1.0e300,
            -1.0e-300, 3.0, -3.0, nextfloat(0.0), prevfloat(floatmax()) / 3]
    mix = vcat(edge, ldexp.(randn(rng, 1000), rand(rng, -1000:1000)))
    @test xsum_round(xe_sum(mix), 1) == xe_ref(mix)
    @test xsum_round(xe_sum([1.0, 2.0^-80, -1.0]), 1) == 2.0^-80        # no absorption
    @test xsum_round(xe_sum([5.0e-324, 5.0e-324]), 1) == 1.0e-323        # subnormal
    @test xsum_round(xe_sum(Float64[]), 1) == 0.0
    @test isnan(xsum_round(xe_sum([1.0, Inf]), 1))                       # non-finite flagged
    @test xsum_round(xe_sum([floatmax(), floatmax()]), 1) == Inf         # overflow rounds up
    @test xsum_round(xe_sum([2.0^53, 1.0]), 1) == 2.0^53                 # half to even
    @test xsum_round(xe_sum([2.0^53 + 2, 1.0]), 1) == 2.0^53 + 4
    @test xsum_parts(0.0) == (0, 0, 0, 0)
    @test first(xsum_parts(NaN)) == XSUM_NSLOT && size(xsum_zeros(2, 5)) == (XSUM_NSLOT, 2, 5)
    @test XSUM_NSLOT == XSUM_NLIMBS + 1
end
