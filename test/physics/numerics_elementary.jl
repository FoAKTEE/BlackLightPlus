# tang_exp and tang_log are Base's exp and log bit for bit on the host, over random
# arguments across the whole range, the edges of every branch, the subnormals and the
# special values; a device kernel (a pool of the comparison devices) returns the same bits;
# both are allocation-free.
using Test, BlackLightPlus, Random
using BlackLightPlus.Numerics: tang_exp, tang_log
using BlackLightPlus.Execution: PoolExecutor, foreach_item!
import Adapt
const DK = Main.Decks

ubits(x) = reinterpret(UInt64, x)
agree(a, b) = ubits(a) == ubits(b) || (isnan(a) && isnan(b))
mismatches(f, g, xs) = count(x -> !agree(f(x), g(x)), xs)

struct ElementaryProgram{V} <: Function
    x::V
end
Adapt.@adapt_structure ElementaryProgram
function (p::ElementaryProgram)(item, i, e, l)
    x = @inbounds p.x[item]
    @inbounds e[i] = tang_exp(x)
    @inbounds l[i] = tang_log(abs(x))
    return nothing
end

@testset "tang_exp and tang_log" begin
    rng = Xoshiro(20261107)
    xe = [-760.0 + 1480.0 * rand(rng) for _ in 1:10^6]
    xs = [(2.0 * rand(rng) - 1.0) * 10.0^(-15.0 + 17.0 * rand(rng)) for _ in 1:10^6]
    edges = [708.3964185322641, 709.7827128933841, -708.3964185322641, -745.1332191019412,
             -744.4400719213812]
    special = [0.0, -0.0, 1.0, -1.0, Inf, -Inf, NaN, floatmin(Float64), nextfloat(0.0),
               floatmax(Float64), -floatmax(Float64)]
    near = [f(e) for e in edges for f in (prevfloat, identity, nextfloat)]
    @test mismatches(tang_exp, exp, xe) == 0
    @test mismatches(tang_exp, exp, xs) == 0
    @test mismatches(tang_exp, exp, vcat(near, special)) == 0
    xl = filter(isfinite, [reinterpret(Float64, rand(rng, UInt64) >> 1) for _ in 1:10^6])
    xsub = [reinterpret(Float64, rand(rng, UInt64) & 0x000fffffffffffff) for _ in 1:10^4]
    xn = [1.0 + 0.14 * (rand(rng) - 0.5) for _ in 1:10^6]
    lims = [f(e) for e in (0.9394130628134757, 1.0644944589178595, 1.0, 2.0, 0.5)
            for f in (prevfloat, identity, nextfloat)]
    @test mismatches(tang_log, log, xl) == 0
    @test mismatches(tang_log, log, xsub) == 0
    @test mismatches(tang_log, log, xn) == 0
    @test mismatches(tang_log, log, vcat(lims, [0.0, -0.0, Inf, NaN, nextfloat(0.0),
                                                 floatmax(Float64)])) == 0
    @test isnan(tang_log(-1.0)) && isnan(tang_log(-Inf))
    @test (@allocated tang_exp(1.5)) == 0 && (@allocated tang_log(1.5)) == 0
    # a device computes the host's bits
    x = vcat(xe[1:20_000], xs[1:20_000], near, special)
    e = zeros(length(x)); l = zeros(length(x))
    foreach_item!(PoolExecutor(DK.devices(1)), ElementaryProgram(x), 1:length(x); outs = (e, l))
    @test all(agree.(e, exp.(x)))
    @test all(agree.(l, tang_log.(abs.(x)))) && all(agree.(l[1:40_000], log.(abs.(x[1:40_000]))))
end
