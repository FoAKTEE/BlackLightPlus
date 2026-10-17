# Lensing-band geometry on the Bardeen screen: polygon area and the winding test on
# closed forms; the bands nest and thin with n and n ≥ 1 straddles the critical curve;
# the apparent horizon lies inside the critical curve; band grids equal the per-point
# winding test; the θo > π/2 mirror; the outer edge at high inclination.
using Test, BLP
using BLP.Analysis: polygon_area, in_polygon, apparent_horizon, lensing_band, band_grid
using BLP.Spacetime: critical_curve

# Membership of every grid point in the band, by the per-point winding test.
function brute_mask(grid, inner, outer)
    return [in_polygon(grid[k, 1], grid[k, 2], view(outer, :, 1), view(outer, :, 2)) &&
            !(size(inner, 1) >= 3 &&
              in_polygon(grid[k, 1], grid[k, 2], view(inner, :, 1), view(inner, :, 2)))
            for k in 1:size(grid, 1)]
end

@testset "lensing bands" begin
    @testset "polygon area and winding" begin
        sx = [0.0, 1.0, 1.0, 0.0]; sy = [0.0, 0.0, 1.0, 1.0]
        @test polygon_area(sx, sy) == 1.0
        @test polygon_area(reverse(sx), reverse(sy)) == 1.0           # either orientation
        @test polygon_area(hcat(3 .* sx, 2 .* sy)) == 6.0
        @test polygon_area([0.0, 4.0, 0.0], [0.0, 0.0, 3.0]) == 6.0
        @test polygon_area([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]) == 0.0   # bow tie: net zero
        t = range(0, 2π; length = 2001)[1:end-1]
        @test isapprox(polygon_area(2 .* cos.(t), 2 .* sin.(t)), 4π; rtol = 1e-5)
        @test in_polygon(0.5, 0.5, sx, sy)
        @test !in_polygon(1.5, 0.5, sx, sy) && !in_polygon(0.5, -0.1, sx, sy)
        @test in_polygon(0.5, 0.5, reverse(sx), reverse(sy))          # winding −1 is inside
        # half-open rule on the boundary: bottom edge in, top edge out
        @test in_polygon(0.5, 0.0, sx, sy) && !in_polygon(0.5, 1.0, sx, sy)
        # nonzero winding: a doubly wound square is inside, the bow tie lobes are inside
        @test in_polygon(0.5, 0.5, vcat(sx, sx), vcat(sy, sy))
        @test in_polygon(0.2, 0.5, [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])
        @test in_polygon(0.8, 0.5, [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])
    end

    @testset "bands nest and thin with n" begin
        for a in (0.5, 0.94), inc in (17.0, 60.0)
            θo = deg2rad(inc)
            αc, βc = critical_curve(a, θo; n = 100)
            Ac = polygon_area(αc, βc)
            i0, o0 = lensing_band(a, θo, 0; npts = 80)
            i1, o1 = lensing_band(a, θo, 1; npts = 80)
            i2, o2 = lensing_band(a, θo, 2; npts = 80)
            @test size(i1) == size(o1) == (160, 2) && size(o0) == (4, 2)
            A0i = polygon_area(i0)
            A1i = polygon_area(i1); A1o = polygon_area(o1)
            A2i = polygon_area(i2); A2o = polygon_area(o2)
            @test A1i < Ac < A1o                       # n ≥ 1 straddles the critical curve
            @test A2i < Ac < A2o
            @test A0i < A1i                            # apparent horizon inside band 1
            @test A1o < polygon_area(o0) == (2 * 25.0)^2
            @test A1o - A1i > A2o - A2i > 0            # bands thin with n
            @test A0i < Ac
        end
    end

    @testset "apparent horizon" begin
        a = 0.94; θo = deg2rad(60.0)
        αh, βh = apparent_horizon(a, θo; n = 60)
        αc, βc = critical_curve(a, θo; n = 60)
        @test length(αh) == length(αc) == 120
        s = hypot.(αh, βh) ./ hypot.(αc, βc)          # radial scale along each direction
        @test all(0.05 .< s .< 0.999)
        @test αh ≈ s .* αc && βh ≈ s .* βc
        # the upper (one turning point) and lower (none) legs differ
        @test maximum(abs, s[1:60] .- reverse(s[61:120])) > 1e-3
        @test apparent_horizon(a, θo; n = 60) == (αh, βh)              # deterministic
    end

    @testset "band grid equals the per-point winding test" begin
        a = 0.94; θo = deg2rad(17.0)
        for n in (0, 1)
            inner, outer = lensing_band(a, θo, n; npts = 60)
            grid, mask, N = band_grid(inner, outer; dx = 0.1)
            @test size(grid) == (N * N, 2) && length(mask) == N * N && iseven(N)
            @test mask == brute_mask(grid, inner, outer)
            @test 0 < count(mask) < N * N
        end
        i1, o1 = lensing_band(0.94, deg2rad(60.0), 1; npts = 80)
        g = band_grid(i1, o1; dx = 0.1)
        lim = maximum(abs, o1) + 0.5
        @test g.N == 2 * round(Int, lim / 0.1)
        xs = range(-lim, lim; length = g.N)
        @test g.grid[1:g.N, 1] == collect(xs) && all(==(xs[1]), g.grid[1:g.N, 2])  # α fastest
        @test g.grid[g.N+1, 2] == xs[2]
        gb = band_grid(i1, o1; dx = 0.1, box = 9.0)
        @test gb.grid[1, 1] == -9.0 && gb.grid[end, 2] == 9.0 && gb.N == 180
        # a non-convex outer loop and an inner loop of fewer than three vertices
        ts = range(0, 2π; length = 41)[1:40]
        star = hcat(cos.(ts) .* (2 .+ 1.5 .* cos.(5 .* ts)), sin.(ts) .* (2 .+ 1.5 .* cos.(5 .* ts)))
        grid, mask, _ = band_grid(zeros(0, 2), star; dx = 0.05)
        @test mask == brute_mask(grid, zeros(0, 2), star)
        grid, mask, _ = band_grid(star[1:2, :], star; dx = 0.05)
        @test mask == brute_mask(grid, zeros(0, 2), star)
    end

    @testset "θo > π/2 is the equatorial mirror" begin
        a = 0.94; θo = deg2rad(120.0); θm = π - θo
        αh, βh = apparent_horizon(a, θo; n = 40)
        αm, βm = apparent_horizon(a, θm; n = 40)
        @test αh == αm && βh == -βm
        @test polygon_area(αh, βh) > 0
        for n in (0, 1)
            i1, o1 = lensing_band(a, θo, n; npts = 40)
            i2, o2 = lensing_band(a, θm, n; npts = 40)
            @test i1[:, 1] == i2[:, 1] && i1[:, 2] == -i2[:, 2]
            @test o1[:, 1] == o2[:, 1] && o1[:, 2] == -o2[:, 2]
            @test polygon_area(i1) > 0
        end
        αc, βc = critical_curve(a, θo; n = 64)
        i1, o1 = lensing_band(a, θo, 1; npts = 40)
        @test polygon_area(i1) < polygon_area(αc, βc) < polygon_area(o1)
    end

    @testset "outer edge at high inclination" begin
        # the outer bracket grows past s = 4 instead of collapsing onto the critical curve
        a = 0.94; θo = deg2rad(85.0)
        Ac = polygon_area(critical_curve(a, θo; n = 60)...)
        inner, outer = lensing_band(a, θo, 1; npts = 60)
        @test polygon_area(inner) < Ac < polygon_area(outer)
        @test polygon_area(outer) > 10 * Ac
    end
end
