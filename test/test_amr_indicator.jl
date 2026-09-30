# User-supplied refinement indicator of `render_amr` (src/amr.jl, keyword `refine`).
# `refine(blk, vals) -> Bool` replaces the built-in value tests: `blk` is the
# `AMRBlock` (bi, bj, level, a0, b0, span) and `vals[c, i, j]` the Stokes row c of block
# pixel (i, j) at (a0 + (i - 0.5) ps, b0 + (j - 0.5) ps). Synthetic decks, no data files.
# Run standalone: julia --project=. test/test_amr_indicator.jl
using Test
using BlackLightPlus
using BlackLightPlus: render_amr, AMRBlock

const AMR_IND_CFG = RunConfig(bh_a = 0.0, m_bh_cgs = 1.0e43, mass_scale_cgs = 1.0e24,
                              frequency_hz = 2.3e11,
                              camera = CameraConfig(inclination_deg = 60.0,
                                                    distance = 1000.0, fov = 16.0,
                                                    resolution = 32))

# class deck: row 1 = 1 inside the disk ρ < R, 0 outside; rows 2, 3 = pixel (α, β)
const AMR_IND_R = 3.3
disk_stokes(coords) = begin
    S = zeros(4, length(coords))
    for (k, (a, b)) in enumerate(coords)
        S[1, k] = hypot(a, b) < AMR_IND_R ? 1.0 : 0.0
        S[2, k] = a
        S[3, k] = b
    end
    S
end
class_change(blk, vals) = any(!=(vals[1, 1, 1]), @view vals[1, :, :])

@testset "render_amr: user-supplied refinement indicator" begin
    block, ml = 8, 3
    nb = 32 ÷ block
    ps0 = 16.0 / 32

    # 1. `refine` receives the block and its pixel values in the documented layout
    bad = Threads.Atomic{Int}(0)
    calls = [Threads.Atomic{Int}(0) for _ in 0:ml]
    fired = [Threads.Atomic{Int}(0) for _ in 0:ml]
    check = (blk, vals) -> begin
        _, _, l, a0, b0, span = blk
        ps = span / block
        size(vals) == (4, block, block) || Threads.atomic_add!(bad, 1)
        for i in 1:block, j in 1:block
            (vals[2, i, j] ≈ a0 + (i - 0.5) * ps && vals[3, i, j] ≈ b0 + (j - 0.5) * ps) ||
                Threads.atomic_add!(bad, 1)
        end
        Threads.atomic_add!(calls[l + 1], 1)
        f = class_change(blk, vals)
        f && Threads.atomic_add!(fired[l + 1], 1)
        f
    end
    r = render_amr(AMR_IND_CFG, "/nonexistent"; block, max_level = ml,
                   stokes = disk_stokes, refine = check)
    @test bad[] == 0
    # every block below max_level is tested once; each firing spawns 4 children
    @test calls[1][] == nb^2
    @test all(calls[l + 1][] == 4 * fired[l][] for l in 1:ml-1)
    @test calls[ml + 1][] == 0
    @test r.nrays == block^2 * (nb^2 + 4 * sum(f[] for f in fired))

    # 2. levels follow the indicator: a base block is refined iff the disk edge crosses
    #    its level-0 pixel centres; uniform blocks stay at level 0
    for bi in 1:nb, bj in 1:nb
        a = [-8.0 + ((bi - 1) * block + i - 0.5) * ps0 for i in 1:block]
        b = [-8.0 + ((bj - 1) * block + j - 0.5) * ps0 for j in 1:block]
        inside = [hypot(x, y) < AMR_IND_R for x in a, y in b]
        mixed = any(inside) && !all(inside)
        @test (r.levels[bj, bi] > 0) == mixed
    end

    # 3. an indicator that never fires leaves the base image, whatever the value tests
    #    would have done (abs_grad_cut would refine every edge block here)
    r0 = render_amr(AMR_IND_CFG, "/nonexistent"; block, max_level = ml,
                    stokes = disk_stokes, refine = (blk, vals) -> false)
    @test all(==(0), r0.levels)
    @test r0.nrays == 32^2
    # refinement on the class change sharpens the painted disk edge
    eff = 32 * 2^ml
    pse = 16.0 / eff
    miss(img) = count(img[q, c] != (hypot(-8.0 + (c - 0.5) * pse, -8.0 + (q - 0.5) * pse) <
                                    AMR_IND_R ? 1.0 : 0.0) for c in 1:eff, q in 1:eff)
    @test miss(r.I) < miss(r0.I) ÷ 4
    rv = render_amr(AMR_IND_CFG, "/nonexistent"; block, max_level = ml,
                    stokes = disk_stokes, abs_grad_cut = 0.1, abs_grad_count = 1)
    @test any(>(0), rv.levels)

    # 4. both indicators at once is ambiguous and rejected
    @test_throws "refine" render_amr(AMR_IND_CFG, "/nonexistent"; block, max_level = ml,
                                     stokes = disk_stokes, refine = class_change,
                                     abs_grad_cut = 0.1)

    # 5. deterministic: same images, levels and ray count on a repeat
    r2 = render_amr(AMR_IND_CFG, "/nonexistent"; block, max_level = ml,
                    stokes = disk_stokes, refine = class_change)
    @test r2.I == r.I && r2.levels == r.levels && r2.nrays == r.nrays
end
