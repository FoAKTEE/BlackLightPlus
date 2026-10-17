# Oracle: BLP.Analysis against the old refinement indicator, lensing-band geometry and
# interferometry, bit for bit. Indicators on random 16×16 blocks over a threshold sweep
# (each indicator alone and the keyword-selected default); the apparent horizon, the
# band edges (n = 0, 1, 2) and the band grids at four (a, θo) and a mirrored case;
# polygon areas and the winding test on random polygons, with points on vertex rows and
# on edges; the Tukey window, Radon cuts, visibility amplitudes and their minima on a
# synthetic ring image.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

using BLP.Analysis

const OB = BlackLightPlus                # the old indicator lives at the package root
const OO = BlackLightPlus.Outputs
const bits = Decks.bits

# Decision of the old indicator for one keyword set.
old_refine(I; rgc = 0.15, rgf = 0.03, rlc = 1.0, rlf = 0.03, agc = nothing, agn = 5) =
    OB._block_needs_refinement(I; rel_grad_cut = rgc, rel_grad_frac = rgf, rel_lapl_cut = rlc,
                               rel_lapl_frac = rlf, abs_grad_cut = agc, abs_grad_count = agn)

# Random 16×16 blocks: smooth blobs, sharp edges, noise, exact zeros (0/0 stencils),
# negative values, NaN and Inf pixels, and constant blocks.
function random_block(rng, k)
    bs = 16
    xs = range(-1, 1; length = bs)
    kind = k % 8
    I = if kind == 0
        c = 2rand(rng, 2) .- 1; w = 0.1 + rand(rng)
        [exp(-((x - c[1])^2 + (y - c[2])^2) / w^2) for x in xs, y in xs]
    elseif kind == 1
        r0 = 0.3 + 0.6rand(rng)
        [hypot(x, y) < r0 ? 1.0 + 0.1rand(rng) : 1e-3rand(rng) for x in xs, y in xs]
    elseif kind == 2
        1 .+ 0.2 .* randn(rng, bs, bs)
    elseif kind == 3
        M = rand(rng, bs, bs); M[rand(rng, bs, bs) .< 0.4] .= 0.0; M
    elseif kind == 4
        randn(rng, bs, bs)
    elseif kind == 5
        M = rand(rng, bs, bs) .+ 0.5
        M[rand(rng, 1:bs), rand(rng, 1:bs)] = NaN
        M[rand(rng, 1:bs), rand(rng, 1:bs)] = rand(rng) < 0.5 ? Inf : -Inf
        M
    elseif kind == 6
        fill(rand(rng), bs, bs)
    else
        s = 10.0^(-3 * rand(rng))
        [s * (1 + 0.5sin(4x + 3y)) + 1e-4 * randn(rng) for x in xs, y in xs]
    end
    return Matrix{Float64}(I)
end

@testset "oracle: analysis" begin
    rng = Xoshiro(20261016)

    @testset "refinement indicators" begin
        blocks = [random_block(rng, k) for k in 1:200]
        new_d = Bool[]; old_d = Bool[]
        for I in blocks
            for rgc in (0.0, 0.01, 0.05, 0.15, 0.4, 1.0, 3.0), rgf in (-1.0, 0.0, 0.03, 0.1, 0.5)
                push!(new_d, needs_refinement(RelativeGradient(rgc; frac = rgf), I))
                push!(old_d, old_refine(I; rgc, rgf, rlf = -1.0))
            end
            for rlc in (0.0, 0.05, 0.3, 1.0, 4.0), rlf in (-1.0, 0.0, 0.03, 0.1, 0.5)
                push!(new_d, needs_refinement(RelativeLaplacian(rlc; frac = rlf), I))
                push!(old_d, old_refine(I; rgf = -1.0, rlc, rlf))
            end
            for agc in (0.0, 1e-4, 1e-2, 0.05, 0.2, 1.0), agn in (-1, 0, 1, 5, 40, 256)
                push!(new_d, needs_refinement(AbsoluteGradient(agc; count = agn), I))
                push!(old_d, old_refine(I; agc, agn))
            end
            for rgc in (0.05, 0.15, 0.5), rgf in (-1.0, 0.03), rlc in (0.3, 1.0), rlf in (-1.0, 0.03)
                ind = refinement_indicator(; rel_grad_cut = rgc, rel_grad_frac = rgf,
                                           rel_lapl_cut = rlc, rel_lapl_frac = rlf)
                push!(new_d, needs_refinement(ind, I))
                push!(old_d, old_refine(I; rgc, rgf, rlc, rlf))
            end
            push!(new_d, needs_refinement(refinement_indicator(), I))
            push!(old_d, old_refine(I))
            push!(new_d, needs_refinement(refinement_indicator(; abs_grad_cut = 0.05), I))
            push!(old_d, old_refine(I; agc = 0.05))
        end
        @test bits(new_d) == bits(old_d)
        # the sweep exercises both outcomes of every indicator
        @test 0.1 < count(new_d) / length(new_d) < 0.9
    end

    @testset "apparent horizon, lensing bands, band grids" begin
        cases = ((0.0, 17.0), (0.5, 60.0), (0.94, 17.0), (0.998, 80.0))
        for (a, inc) in cases
            θo = deg2rad(inc)
            @test bits(apparent_horizon(a, θo)) == bits(OG.apparent_horizon(a, θo))
            for n in 0:2
                bn = lensing_band(a, θo, n)
                bo = OG.lensing_band(a, θo, n)
                @test bits(bn) == bits(bo)
                gn = band_grid(bn...)
                go = OG.band_grid(bo...)
                @test bits(gn) == bits(go)
                @test count(gn.mask) > 0
            end
        end
        # keyword paths and the θo > π/2 mirror
        a = 0.94; θo = deg2rad(120.0)
        @test bits(apparent_horizon(a, θo; n = 40, ro = 1e4)) ==
              bits(OG.apparent_horizon(a, θo; n = 40, ro = 1e4))
        for n in 0:2
            bn = lensing_band(a, θo, n; npts = 40, box = 9.0)
            bo = OG.lensing_band(a, θo, n; npts = 40, box = 9.0)
            @test bits(bn) == bits(bo)
            @test bits(band_grid(bn...; dx = 0.07, box = 10.0)) ==
                  bits(OG.band_grid(bo...; dx = 0.07, box = 10.0))
        end
    end

    @testset "polygons" begin
        nbad = 0
        for k in 1:300
            nv = rand(rng, 3:40)
            xs, ys = if isodd(k)          # star-shaped about the origin
                t = sort(2π .* rand(rng, nv))
                r = 1 .+ 2rand(rng, nv)
                r .* cos.(t), r .* sin.(t)
            else                          # arbitrary, usually self-intersecting
                4 .* randn(rng, nv), 4 .* randn(rng, nv)
            end
            k % 5 == 0 && (xs = reverse(xs); ys = reverse(ys))
            nbad += !(polygon_area(xs, ys) === OG.polygon_area(xs, ys))
            P = hcat(xs, ys)
            nbad += !(polygon_area(P) === OG.polygon_area(P))
            for _ in 1:200
                u = rand(rng)
                px, py = if u < 0.25           # on a vertex's horizontal line
                    v = rand(rng, 1:nv); (6randn(rng), ys[v])
                elseif u < 0.4                 # a vertex
                    v = rand(rng, 1:nv); (xs[v], ys[v])
                elseif u < 0.55                # on an edge
                    v = rand(rng, 1:nv); w = v == nv ? 1 : v + 1; s = rand(rng)
                    (xs[v] + s * (xs[w] - xs[v]), ys[v] + s * (ys[w] - ys[v]))
                else
                    (6randn(rng), 6randn(rng))
                end
                nbad += in_polygon(px, py, xs, ys) != OG._in_polygon(px, py, xs, ys)
            end
        end
        @test nbad == 0
    end

    @testset "interferometry" begin
        # a thin Gaussian ring with a brightness asymmetry and a hot spot, N = 256
        N = 256; fov = 160.0; d = 42.0
        Δ = fov / N; c = (N + 1) / 2
        img = zeros(N, N)
        for iβ in 1:N, iα in 1:N
            x = (iα - c) * Δ; y = (iβ - c) * Δ
            ρ = hypot(x, y)
            img[iβ, iα] = exp(-0.5 * ((ρ - d / 2) / 1.5)^2) * (1 + 0.6 * cos(atan(y, x) - 0.7)) +
                          0.3 * exp(-((x - 8)^2 + (y + 5)^2) / 8)
        end
        img[img .< 1e-6] .= 0.0
        for n in (0, 1, 2, 3, 7, 64, 101, 256), α in (-0.1, 0.0, 0.25, 0.5, 1, 1.0, 1.5)
            @test bits(tukey_window(n, α)) == bits(OO.tukey_window(n, α))
        end
        @test bits(tukey_window(129)) == bits(OO.tukey_window(129))
        @test MUAS_TO_RAD === OO.MUAS_TO_RAD
        for ang in (0.0, 30.0, 45, 90.0, 137.5, -20.0, 200.0)
            @test bits(radon_cut(img, ang; fov_muas = fov)) == bits(OO.radon_cut(img, ang; fov_muas = fov))
        end
        for angles in ((0.0, 90.0), (0, 45.0, 137.5)), pad in (4, 16),
            maxbl in (Inf, 20.0), win in (:tukey, :none)
            rn = visamp(img; angles = angles, fov_muas = fov, pad = pad, maxbaseline = maxbl, window = win)
            ro = OO.visamp(img; angles = angles, fov_muas = fov, pad = pad, maxbaseline = maxbl, window = win)
            @test length(rn) == length(ro)
            @test all(bits(x) == bits(y) for (x, y) in zip(rn, ro))
            for r in rn, skip in (0, 1, 3, 1000), blmin in (0.0, 2.0)
                @test bits(visamp_minima(r.baselines, r.amplitude; skip = skip, bl_min = blmin)) ==
                      bits(OO.visamp_minima(r.baselines, r.amplitude; skip = skip, bl_min = blmin))
            end
        end
        r = visamp(img; fov_muas = fov)
        @test length(visamp_minima(r[1].baselines, r[1].amplitude)) >= 3
    end
end
