# Ray sources and image-plane generators: the item-to-coordinate maps of the pixel grid
# (camera ray order), image blocks (refinement pixel order) and coordinate lists, and the
# host ray list.
using Test, BlackLightPlus
using StaticArrays
using BlackLightPlus.Execution: HostRays, PixelGrid, BlockGrid, CoordList, ray, image_coords

@testset "ray sources" begin

@testset "pixel grid: camera order, β fastest" begin
    fov = 24.0; n = 10
    g = PixelGrid(fov, n)
    edges = range(-fov / 2, fov / 2; length = n + 1)
    @test g.centres == collect((edges[1:end-1] .+ edges[2:end]) ./ 2) && g.npix == n
    camera_order = [(α, β) for α in g.centres for β in g.centres]
    @test [image_coords(g, k) for k in 1:n^2] == camera_order
    @test PixelGrid(24, n).centres == g.centres
    @test PixelGrid(g.centres, n).centres === g.centres
end

@testset "block grid: block-major, j fastest" begin
    bs = 4
    blocks = [SVector(-3.0, 1.5, 0.25), SVector(2.0, -6.0, 1.0 / 3)]
    g = BlockGrid(blocks, bs)
    expect = [(b[1] + (i - 0.5) * b[3], b[2] + (j - 0.5) * b[3])
              for b in blocks for i in 1:bs for j in 1:bs]
    @test [image_coords(g, k) for k in 1:length(blocks) * bs^2] == expect
    @test isempty(BlockGrid(bs).blocks) && BlockGrid(bs).bs == bs
end

@testset "coordinate list and host rays" begin
    coords = [(0.5, -1.0), (3.0, 2.5), (-4.0, 0.0)]
    g = CoordList(coords)
    @test [image_coords(g, k) for k in 1:3] == coords
    @test image_coords(CoordList([1.0, 2.0], [3.0, 4.0]), 2) == (2.0, 4.0)
    rays = [SVector{8,Float64}(1:8) .* k for k in 1:4]
    @test ray(HostRays(rays), 3) == (rays[3], true)
end

end
