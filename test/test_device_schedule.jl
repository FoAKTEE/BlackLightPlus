# Cost-aware scheduling of full-frame renders (results/blp/multigpu/dag.md, MG09):
# `cost_order` queues 16×16 pixel tiles longest-predicted-first; its tile ranking tracks
# the measured march trip counts (Spearman ρ ≥ 0.7); images are bitwise invariant under
# any queue order; `:mixed` queues the critical-curve pixels first in fp64 in the same
# queue and equals the two-sweep splice (fp32 image, fp64 rescued pixels). KA CPU backend.
using Statistics
@isdefined(DI_NDEV) || include(joinpath(@__DIR__, "device_decks.jl"))

# Trip count of the kernel march loop: the `_gimg_stop` rule and the `rk4_step_fixed`
# step that `_gimg_march` iterates (sampling does not change the step sequence).
function ds_trips(y, a, r_h, C, escape, R_em, nmax)
    r_prev = Inf; n = 0
    for _ in 1:nmax
        stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
        stop && break
        r_prev = r
        y = BlackLightPlus.rk4_step_fixed(KerrSchildCartesian(a), y, (r - r_h) / C)
        n += 1
    end
    return n
end

ds_spearman(x, y) = cor(Float64.(invperm(sortperm(x))), Float64.(invperm(sortperm(y))))

# tile (16×16) of pixel k of an npix² frame, numbered [β tile, α tile] column-major
ds_tile(k, npix; T = 16) = (((k - 1) ÷ npix) ÷ T) * cld(npix, T) + ((k - 1) % npix) ÷ T + 1

@testset "cost-aware scheduling (KA CPU backend)" begin

@testset "cost_order: longest-first tiles" begin
    o = cost_order(0.9, 60.0, 24.0, 40)
    @test isperm(o) && length(o) == 1600
    @test cost_order(0.9, 60.0, 24.0, 40) === o                 # the latest camera is reused
    tiles = [ds_tile(k, 40) for k in o]
    @test all(t -> issorted(findall(==(t), tiles)) &&           # each tile's pixels are
                   diff(findall(==(t), tiles)) == ones(Int, count(==(t), tiles) - 1),
              unique(tiles))                                    # contiguous in the queue
    ring = Set(_rescue_pixels(0.9, 60.0, 24.0, 40; rescue_px = 0))
    first_tile = findall(k -> ds_tile(k, 40) == ds_tile(o[1], 40), 1:1600)
    @test any(in(ring), first_tile)                             # a ring tile goes first
    @test Set(o[1:length(first_tile)]) == Set(first_tile)
    # the curve outside the frame: no preference, tiles in index order
    far = cost_order(0.9, 60.0, 2.0, 40)
    @test isempty(_rescue_pixels(0.9, 60.0, 2.0, 40))
    @test unique(ds_tile(k, 40) for k in far) == 1:9
end

@testset "tile ranking vs measured trip counts: Spearman ρ ≥ 0.7" begin
    for (a, inc, fov) in ((0.9, 60.0, 24.0), (0.94, 60.0, 16.0))
        npix = 128
        cam = PlaneParallelCamera(inc, 1000.0, fov, npix)
        r_h = 1 + sqrt(1 - a^2)
        trips = [ds_trips(y, a, r_h, 80.0, 1100.0, sqrt(3.0) * 12, 3000)
                 for y in camera_rays(KerrSchildCartesian(a), cam)]
        nt = cld(npix, 16)^2
        measured = zeros(nt)
        for k in eachindex(trips)
            measured[ds_tile(k, npix)] += trips[k]
        end
        queue_rank = zeros(nt)                                  # 1 = queued first
        for (q, t) in enumerate(unique(ds_tile(k, npix) for k in cost_order(a, inc, fov, npix)))
            queue_rank[t] = q
        end
        ρ = ds_spearman(-queue_rank, measured)
        @info "cost_order tile ranking" a inc fov ρ
        @test ρ >= 0.7
    end
end

@testset "images are bitwise invariant under the queue order" begin
    cfg = di_cfg()
    fg = di_uniform()
    snap = di_amr()
    n = cfg.camera.resolution^2
    perms = (nothing, collect(1:n), collect(n:-1:1), Random.randperm(Random.Xoshiro(9), n))
    ref = di_bits(render_gpu_fused(cfg, fg; devices = di_hosts(1), order = collect(1:n)))
    @test all(di_bits(render_gpu_fused(cfg, fg; devices = DI_POOLS[3], order = o)) == ref
              for o in perms)
    refp = di_bits(render_gpu_polarized(cfg, fg; devices = di_hosts(1), order = collect(1:n)))
    @test all(di_bits(render_gpu_polarized(cfg, fg; devices = DI_POOLS[2], order = o)) == refp
              for o in perms)
    refa = di_bits(render_gpu_amr_polarized(cfg, snap; devices = di_hosts(1), order = collect(1:n)))
    @test all(di_bits(render_gpu_amr_polarized(cfg, snap; devices = DI_POOLS[5], order = o)) == refa
              for o in perms)
end

@testset ":mixed: one queue, rescued pixels first, equal to the two-sweep splice" begin
    cfg = di_cfg(res = 24)
    fg = di_uniform()
    snap = di_amr()
    pix = _rescue_pixels(cfg.bh_a, cfg.camera.inclination_deg, cfg.camera.fov, 24)
    @test !isempty(pix)
    splice(f32, f64) = (s = copy(f32); s[pix] = f64[pix]; s)
    splice(f32::NamedTuple, f64::NamedTuple) = map(splice, f32, f64)
    for (render, grid) in ((render_gpu_fused, fg), (render_gpu_polarized, fg),
                           (render_gpu_amr, snap), (render_gpu_amr_polarized, snap))
        fr = Ref(0.0)
        mixed = render(cfg, grid; devices = DI_POOLS[3], precision = :mixed, rescued_frac = fr)
        two = splice(render(cfg, grid; devices = DI_POOLS[3], precision = :f32_raw),
                     render(cfg, grid; devices = DI_POOLS[3], precision = :f64))
        @test di_bits(mixed) == di_bits(two)
        @test fr[] == length(pix) / 24^2
    end
    # queue order: every fp64 (rescued) chunk is taken before any fp32 chunk
    pool = DevicePool(di_hosts(1); lanes = 1, chunk = 5)
    taken = Symbol[]
    rec(tag) = (lane, outs, y0d, n) -> (push!(taken, tag); fill!(view(outs[1], 1:n), 0.0))
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 24)
    items = cost_order(cfg.bh_a, 60.0, 24.0, 24)
    img = Vector{Float64}(undef, 24^2)
    march!((img,), pool, _mixed_parts(items, pix, CameraRays(st, cam),
                                      CameraRays(st, cam, Float32), rec(:f64), rec(:f32)))
    @test taken == [fill(:f64, cld(length(pix), 5)); fill(:f32, cld(24^2 - length(pix), 5))]
    close(pool)
end

end
