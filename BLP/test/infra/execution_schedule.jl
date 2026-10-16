# Scheduling geometry on a synthetic critical curve: `cost_order` queues contiguous tiles,
# those crossed by the curve first; `rescue_pixels` is the dilated curve; the mixed-lane
# partition keeps the queue order; a mixed queue runs its rescued part first.
using Test, BLP
using Statistics: mean
using BLP.Execution: cost_order, rescue_pixels, mixed_partition, march_sort_order,
                     queue_items, curve_inclination

# A closed curve sampled at m points: a circle of radius r centred at (cx, cy).
function circle(cx, cy, r; m = 400)
    θ = range(0, 2π; length = m + 1)[1:end-1]
    return (cx .+ r .* cos.(θ), cy .+ r .* sin.(θ))
end

# Tile (T × T) of flat pixel k of an npix² frame, numbered [β tile, α tile] column-major.
tile_of(k, npix; T = 16) = (((k - 1) ÷ npix) ÷ T) * cld(npix, T) + ((k - 1) % npix) ÷ T + 1

const CURVE = circle(1.5, 0.0, 5.2)

@testset "scheduling" begin

@testset "queue_items, march_sort_order, curve_inclination" begin
    @test queue_items(nothing, 5) == 1:5
    @test queue_items([3, 1, 2], 3) == [3, 1, 2]
    @test_throws ArgumentError queue_items([1, 1, 2], 3)
    @test_throws ArgumentError queue_items([1, 2], 3)
    @test march_sort_order([3, 9, 1, 9]) == [2, 4, 1, 3]
    @test curve_inclination(60) == deg2rad(60.0)
    @test curve_inclination(0.0) == 0.02 && curve_inclination(180.0) == π - 0.02
end

@testset "cost_order: longest-first, contiguous tiles" begin
    o = cost_order(CURVE, 24.0, 40)
    @test isperm(o) && length(o) == 1600
    @test cost_order(CURVE, 24.0, 40) == o                       # a pure function
    tiles = [tile_of(k, 40) for k in o]
    @test all(t -> diff(findall(==(t), tiles)) == ones(Int, count(==(t), tiles) - 1),
              unique(tiles))                                     # each tile contiguous
    ring = Set(rescue_pixels(CURVE, 24.0, 40; rescue_px = 0))
    @test !isempty(ring)
    first_tile = findall(k -> tile_of(k, 40) == tile_of(o[1], 40), 1:1600)
    @test any(in(ring), first_tile)                              # a ring tile goes first
    @test Set(o[1:length(first_tile)]) == Set(first_tile)
    rank = Dict(t => q for (q, t) in enumerate(unique(tiles)))
    ringtiles = unique(tile_of(k, 40) for k in ring)
    others = [t for t in keys(rank) if !(t in ringtiles)]
    @test mean(rank[t] for t in ringtiles) < mean(rank[t] for t in others)
    o8 = cost_order(CURVE, 24.0, 40; tile = 8)
    @test isperm(o8) && o8 != o
end

@testset "a curve outside the frame: no preference" begin
    far = cost_order(CURVE, 2.0, 40)
    @test isempty(rescue_pixels(CURVE, 2.0, 40))
    @test unique(tile_of(k, 40) for k in far) == 1:9
    empty_curve = (Float64[], Float64[])
    @test isempty(rescue_pixels(empty_curve, 24.0, 40))
    @test cost_order(empty_curve, 24.0, 40) == far
end

@testset "rescue_pixels: Chebyshev dilation of the curve" begin
    p0 = rescue_pixels(CURVE, 24.0, 40; rescue_px = 0)
    p2 = rescue_pixels(CURVE, 24.0, 40)
    @test issorted(p0) && issorted(p2) && issubset(p0, p2)
    ij(k) = (mod1(k, 40), (k - 1) ÷ 40 + 1)
    near(k) = any(c -> max(abs(ij(k)[1] - ij(c)[1]), abs(ij(k)[2] - ij(c)[2])) <= 2, p0)
    @test all(near, p2)
    @test all(k -> !near(k) || k in p2, 1:1600)
end

@testset "mixed_partition keeps the queue order" begin
    o = cost_order(CURVE, 24.0, 40)
    pix = rescue_pixels(CURVE, 24.0, 40)
    p = mixed_partition(o, pix)
    @test p.rescue == pix
    @test p.bulk == filter(k -> !(k in pix), o)
    @test sort(vcat(p.rescue, p.bulk)) == 1:1600
end

end
