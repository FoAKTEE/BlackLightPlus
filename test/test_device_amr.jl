# AMR image driver (src/amr.jl; results/blp/multigpu/dag.md, MG10): `render_amr` paints
# and tests blocks in parallel threads, and the native-AMR device marcher `AMRStokes`
# streams refinement blocks through the device pool without level barriers (rays made on
# the device from each block's origin and pixel size). The oracle is the previous
# level-synchronous serial driver, kept below verbatim: images, level map and ray count
# are bitwise equal to it on a synthetic deck, a CPU-Stokes deck (trace_rays_sampled +
# the CPU marches) and the device marcher on 1, 2, 3 and 5 HostDevices.
using BlackLightPlus: render_amr, camera_rays_at, trace_rays_sampled, Radiation,
                      _block_needs_refinement
using BlackLightPlus.Simulation: sample_primitives, pgas_from_eint, units
@isdefined(DI_NDEV) || include(joinpath(@__DIR__, "device_decks.jl"))

# The level-synchronous serial driver before MG10 (src/amr.jl at 41743a1), minus the
# default closure: one `stokes_at(coords)` call per level, then serial paint and test.
function level_sync_amr(cfg, stokes_at; block = 8, max_level = 2,
                        rel_grad_cut = 0.15, rel_grad_frac = 0.03, rel_lapl_cut = 1.0,
                        rel_lapl_frac = 0.03, abs_grad_cut = nothing, abs_grad_count = 5)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    npix0 = cam.npix
    nb = npix0 ÷ block                       # base blocks per side
    fov = cam.fov
    px0 = fov / npix0                        # level-0 pixel size
    eff = npix0 * 2^max_level                # effective final resolution
    imgs = zeros(4, eff, eff)
    levels = zeros(Int, nb, nb)
    nrays = 0

    # blacklight scheme: a refined block spawns 4 children, each `block`x`block`
    # pixels over a quarter of the parent area (pixel size halves per level)
    struct_area(bi, bj) = (-fov / 2 + (bi - 1) * block * px0,
                           -fov / 2 + (bj - 1) * block * px0)

    # paint helper: block at level l, origin (a0, b0), pixel size ps, block px I
    function paint!(S::Array{Float64,3}, a0, b0, vals::Matrix{Float64}, comp::Int,
                    cell::Int)
        # cell = canvas cells per block pixel = 2^(max_level - l)
        # canvas index of block origin
        ca = round(Int, (a0 + fov / 2) / (fov / eff))
        cb = round(Int, (b0 + fov / 2) / (fov / eff))
        bs = size(vals, 1)
        # canvas uses render()'s I[iβ, iα] layout: β (j / cb) is the FIRST axis
        @inbounds for i in 1:bs, j in 1:bs, di in 1:cell, dj in 1:cell
            S[comp, cb + (j-1)*cell + dj, ca + (i-1)*cell + di] = vals[i, j]
        end
    end

    # worklist of blocks to render at the current level
    work = [(bi, bj, 0, struct_area(bi, bj)..., block * px0) for bi in 1:nb, bj in 1:nb][:]
    while !isempty(work)
        # coords for ALL blocks at this level in one batch
        coords = Tuple{Float64,Float64}[]
        for (bi, bj, l, a0, b0, span) in work
            ps = span / block
            for i in 1:block, j in 1:block
                push!(coords, (a0 + (i - 0.5) * ps, b0 + (j - 0.5) * ps))
            end
        end
        S = stokes_at(coords)
        nrays += length(coords)
        next = Tuple{Int,Int,Int,Float64,Float64,Float64}[]
        for (w, (bi, bj, l, a0, b0, span)) in enumerate(work)
            base = (w - 1) * block * block
            vals = [S[1, base + (i-1)*block + j] for i in 1:block, j in 1:block]
            cellsz = Int(2^(max_level - l))
            for comp in 1:4
                vc = [S[comp, base + (i-1)*block + j] for i in 1:block, j in 1:block]
                paint!(imgs, a0, b0, vc, comp, cellsz)
            end
            if l < max_level && _block_needs_refinement(vals; rel_grad_cut, rel_grad_frac,
                                                        rel_lapl_cut, rel_lapl_frac,
                                                        abs_grad_cut, abs_grad_count)
                levels[bj, bi] = max(levels[bj, bi], l + 1)   # [β-block, α-block], like the images
                half = span / 2
                for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))
                    push!(next, (bi, bj, l + 1, a0 + da, b0 + db, half))
                end
            end
        end
        work = next
    end
    return (I=Matrix(imgs[1, :, :]), Q=Matrix(imgs[2, :, :]), U=Matrix(imgs[3, :, :]),
            V=Matrix(imgs[4, :, :]), levels=levels, nrays=nrays)
end

amr_bits(r) = (di_bits((r.I, r.Q, r.U, r.V)), r.levels, r.nrays)

# a sharp synthetic ring with polarization, so refinement fires along it
ring_stokes(coords) = (S = Matrix{Float64}(undef, 4, length(coords));
    for (k, (α, β)) in enumerate(coords)
        I = exp(-((hypot(α, 0.8β) - 5.0) / 0.6)^2) + 0.01
        S[1, k] = I; S[2, k] = 0.3I * cos(α); S[3, k] = 0.2I * sin(β); S[4, k] = 0.01I
    end; S)

@testset "AMR driver pipeline (KA CPU backend)" begin

@testset "synthetic deck: bitwise equal to the level-synchronous driver" begin
    cfg = di_cfg(res = 64)
    for kw in ((;), (abs_grad_cut = 0.02, abs_grad_count = 3), (rel_grad_cut = 0.05,))
        for (block, max_level) in ((8, 2), (4, 3), (16, 1))
            new = render_amr(cfg, ""; block, max_level, stokes = ring_stokes, kw...)
            @test amr_bits(new) == amr_bits(level_sync_amr(cfg, ring_stokes; block, max_level, kw...))
            @test maximum(new.levels) >= 1                  # refinement fired
        end
    end
end

@testset "CPU-Stokes deck: bitwise equal to the level-synchronous driver" begin
    cfg = di_cfg(res = 16)
    snap = di_amr()
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 16)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    scc = BlackLightPlus._prepared_scc(cfg, u)
    sampler = x -> (s = sample_primitives(snap, x);
                    (rho = s.rho, pgas = pgas_from_eint(s.eint, cfg.plasma.gamma), uu = s.uu,
                     bb = s.bb, inside = s.inside))
    cpu(coords) = begin
        rays = camera_rays_at(st, cam, coords)
        states, dls = trace_rays_sampled(st, rays; C = 80.0, nmax = cfg.max_steps,
                                         escape = 1100.0)
        S = zeros(4, length(rays))
        Threads.@threads for i in eachindex(rays)
            S[:, i] .= Radiation.polarized_march(st, scc, sampler, states[i], dls[i],
                                                 cfg.frequency_hz, u.L_unit;
                                                 cam_up = BlackLightPlus._camera_up(cam))
        end
        S
    end
    new = render_amr(cfg, ""; block = 4, max_level = 2, stokes = cpu)
    @test amr_bits(new) == amr_bits(level_sync_amr(cfg, cpu; block = 4, max_level = 2))
    @test maximum(new.levels) >= 1 && new.nrays > 16^2
end

@testset "device marcher: streamed blocks on 1, 2, 3, 5 devices, bitwise equal" begin
    cfg = di_cfg(res = 16)
    snap = di_amr()
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 16)
    for polarized in (false, true)
        ref_stokes(coords) = march_rays_gpu_amr(cfg, snap, camera_rays_at(st, cam, coords);
                                                polarized, devices = di_hosts(1))
        ref = amr_bits(level_sync_amr(cfg, ref_stokes; block = 4, max_level = 2))
        @test maximum(ref[2]) >= 1
        for nd in DI_NDEV
            # small pools: 3-block chunks, so children are queued while parents march
            pool = DevicePool(di_hosts(nd); lanes = 2, chunk = 48)
            m = AMRStokes(cfg, snap; polarized, devices = pool)
            @test amr_bits(render_amr(cfg, ""; block = 4, max_level = 2, stokes = m)) == ref
            @test m([(0.5, -1.0), (3.0, 2.5)]) == ref_stokes([(0.5, -1.0), (3.0, 2.5)])  # closure form
            close(pool)
        end
    end
end

end
