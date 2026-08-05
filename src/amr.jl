# amr.jl — adaptive image-plane refinement (blacklight adaptive_refinement.hpp;
# White 2022, "Adaptive Ray Tracing"). The image is tiled into blocks; after each
# level's render every block is tested with refinement indicators: the relative
# gradient (averaged-denominator stencils) and relative Laplacian, each firing
# when the fraction of pixels above the cut exceeds `frac`, or — opt-in via
# `abs_grad_cut` — the value-based absolute gradient, firing on a pixel count.
# A refined block spawns 4 children at twice the pixel density over the same
# area, re-tested up to `max_level`; the final image is a mosaic at
# base·2^max_level per side, returned with a per-block level map.

"One block's refinement indicator (blacklight EvaluateBlock: abs-grad XOR rel tests)."
function _block_needs_refinement(I::Matrix{Float64};
                                 rel_grad_cut::Float64, rel_grad_frac::Float64,
                                 rel_lapl_cut::Float64, rel_lapl_frac::Float64,
                                 abs_grad_cut::Union{Nothing,Float64}=nothing,
                                 abs_grad_count::Int=5)
    bs = size(I, 1)
    # Value-based indicator (White 2022): |∇_abs I| = hypot(Δi, Δj), centered
    # differences (one-sided at block edges); refine when >= `abs_grad_count`
    # pixels exceed the cut. When `abs_grad_cut` is set it IS the indicator —
    # the relative tests below are not evaluated (no mixing; the scale-free
    # relative pair saturates on dim backgrounds). The cut is in intensity units
    # per pixel step, so smooth features fall below it under refinement while
    # unresolved sharp edges (rings) stay above.
    if abs_grad_cut !== nothing
        nex = 0
        @inbounds for i in 1:bs, j in 1:bs
            qx = if j == 1
                I[i, 2] - I[i, 1]
            elseif j == bs
                I[i, bs] - I[i, bs-1]
            else
                (I[i, j+1] - I[i, j-1]) / 2
            end
            qy = if i == 1
                I[2, j] - I[1, j]
            elseif i == bs
                I[bs, j] - I[bs-1, j]
            else
                (I[i+1, j] - I[i-1, j]) / 2
            end
            q = hypot(qx, qy)
            isfinite(q) || continue
            q > abs_grad_cut && (nex += 1)
            nex >= abs_grad_count && return true
        end
        return false
    end
    # relative gradient (averaged denominators; one-sided at edges)
    if rel_grad_frac >= 0
        nex = 0; nch = 0
        @inbounds for i in 1:bs, j in 1:bs
            qx = if j == 1
                2 * (I[i, 2] - I[i, 1]) / (I[i, 1] + I[i, 2])
            elseif j == bs
                2 * (I[i, bs] - I[i, bs-1]) / (I[i, bs-1] + I[i, bs])
            else
                2 * (I[i, j+1] - I[i, j-1]) / (I[i, j-1] + 2 * I[i, j] + I[i, j+1])
            end
            qy = if i == 1
                2 * (I[2, j] - I[1, j]) / (I[1, j] + I[2, j])
            elseif i == bs
                2 * (I[bs, j] - I[bs-1, j]) / (I[bs-1, j] + I[bs, j])
            else
                2 * (I[i+1, j] - I[i-1, j]) / (I[i-1, j] + 2 * I[i, j] + I[i+1, j])
            end
            q = hypot(qx, qy)
            isfinite(q) || continue
            nch += 1
            q > rel_grad_cut && (nex += 1)
        end
        nch > 0 && nex / nch > rel_grad_frac && return true
    end
    # relative Laplacian (5-point, interior; normalized by the local 4-neighbour mean)
    if rel_lapl_frac >= 0
        nex = 0; nch = 0
        @inbounds for i in 2:bs-1, j in 2:bs-1
            lap = I[i, j-1] + I[i, j+1] + I[i-1, j] + I[i+1, j] - 4 * I[i, j]
            den = (I[i, j-1] + I[i, j+1] + I[i-1, j] + I[i+1, j] + 4 * I[i, j]) / 8
            q = abs(lap) / den
            isfinite(q) || continue
            nch += 1
            q > rel_lapl_cut && (nex += 1)
        end
        nch > 0 && nex / nch > rel_lapl_frac && return true
    end
    return false
end

"""
    render_amr(cfg, snapshot_path; block=8, max_level=2, polarized=false,
               tracer=trace_rays_sampled, stokes=nothing,
               rel_grad_cut=0.15, rel_grad_frac=0.03,
               rel_lapl_cut=1.0, rel_lapl_frac=0.03,
               abs_grad_cut=nothing, abs_grad_count=5)
        -> (I, Q, U, V, levels, nrays)

Adaptive-refinement render: mosaic images at `cfg.camera.resolution * 2^max_level`
per side (layout as `render`), the per-base-block finest-level map, and the total
ray count; the resolution must be a multiple of `block`. Default indicator:
relative gradient + Laplacian; setting `abs_grad_cut` selects White 2022's
value-based absolute gradient instead and disables the relative pair (see
`_block_needs_refinement`) — use it on dim-background decks where the relative
pair saturates. `tracer` is the phase-1 backend of the default Stokes closure
(e.g. a GPU tracer). `stokes`, when given, replaces that closure wholesale:

    stokes(coords::Vector{Tuple{Float64,Float64}}) -> 4 × length(coords) Matrix

rows `I,Q,U,V`; `coords[k] = (α, β)` in the `camera_rays_at` image-plane
convention. This is the seam for a fused single-pass GPU imager (built over
`march_rays_gpu_amr`); the snapshot/sampler/coefficient/`tracer` stack is then
not built, so `snapshot_path` and the plasma config go unused.
"""
function render_amr(cfg::RunConfig, snapshot_path::AbstractString;
                    block::Int=8, max_level::Int=2, polarized::Bool=false,
                    tracer=trace_rays_sampled, stokes=nothing,
                    rel_grad_cut::Float64=0.15, rel_grad_frac::Float64=0.03,
                    rel_lapl_cut::Float64=1.0, rel_lapl_frac::Float64=0.03,
                    abs_grad_cut::Union{Nothing,Float64}=nothing,
                    abs_grad_count::Int=5)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    cam.npix % block == 0 ||
        error("render_amr: resolution $(cam.npix) is not divisible by block ($block); " *
              "the trailing pixel strip would never be rendered")

    # a caller-supplied `stokes` replaces the default two-phase closure and skips
    # building the snapshot/sampler/coefficient/tracer stack
    stokes_at = if stokes === nothing
        st = KerrSchildCartesian(cfg.bh_a)
        snap = AthenaKSnapshot(snapshot_path)
        u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
        p = cfg.plasma
        scc = _prepared_scc(cfg, u)
        γ = p.gamma
        sampler = x -> begin
            s = sample_primitives(snap, x)
            (rho=s.rho, pgas=pgas_from_eint(s.eint, γ), uu=s.uu, bb=s.bb, inside=s.inside)
        end
        up = _camera_up(cam)
        ν = cfg.frequency_hz
        Lu = u.L_unit
        # march a coordinate list -> per-coord Stokes (threaded; shared by all levels)
        (coords::Vector{Tuple{Float64,Float64}}) -> begin
            rays = camera_rays_at(st, cam, coords)
            states, dls = tracer(st, rays; C=80.0, nmax=cfg.max_steps,
                                 escape=1.1 * cfg.camera.distance)
            out = Matrix{Float64}(undef, 4, length(rays))
            Threads.@threads for i in 1:length(rays)
                if polarized
                    out[:, i] .= Radiation.polarized_march(st, scc, sampler, states[i],
                                                           dls[i], ν, Lu; cam_up=up)
                else
                    out[1, i] = Radiation.unpolarized_march(st, scc, sampler, states[i],
                                                            dls[i], ν, Lu)
                    out[2, i] = out[3, i] = out[4, i] = 0.0
                end
            end
            return out
        end
    else
        stokes
    end

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
