# gpu_image.jl — fused single-pass GPU unpolarized imaging: one thread per ray marches
# the Hamiltonian null geodesic (in-kernel ForwardDiff-Dual RHS), samples the fluid, and
# accumulates the invariant intensity, returning only the per-pixel image. The CPU march
# runs source->camera with the affine couple I_n = A_n I_{n-1} + B_n; here we march
# camera->source and accumulate the same sum in reverse, I_obs = Σ_n B_n Π_{k>n} A_k,
# via a running transmission (identical up to fp round-off). The per-sample physics is
# the CPU march's own: `metric_and_inverse`, `primitive_four_vectors`,
# `unpolarized_coefficients` and `unpolarized_segment`, called in-kernel with the isbits
# `coefficient_params` set. Approximation: the AMR fluid is resampled once onto a
# uniform box and trilinearly sampled in-kernel (validated against the CPU path). CUDA
# optional: NOT included from BlackLightPlus.jl.

using StaticArrays
using ForwardDiff: Dual, partials
using KernelAbstractions
using CUDA
using LinearAlgebra: dot
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, critical_curve
using BlackLightPlus.Simulation: AthenaKSnapshot, sample_primitives, pgas_from_eint, units,
      SnapshotSeries, get_frame, bracketing_pair, primitive_four_vectors
using BlackLightPlus.Geodesics: metric_and_inverse, rk4_step_fixed
using BlackLightPlus.Radiation: coefficient_params, unpolarized_coefficients,
      unpolarized_segment

# ---------------------------------------------------------------- uniform fluid grid
"""
    UniformFluidGrid

Host-resident dense Cartesian resample of the AMR fluid: `host[f,i,j,k]` holds the 8
primitives (dens, uu1, uu2, uu3, eint, bb1, bb2, bb3) at grid node
`x = xo + (i-1)/inv_dx` (same for y,j / z,k), on `[-box_half, box_half]^3`. Built once
on the host; the drivers upload it once per device pool (`resident_grid`).
"""
struct UniformFluidGrid
    host::Array{Float32,4}    # (8, n, n, n)
    n::Int
    box_half::Float64
    xo::Float64               # = -box_half
    inv_dx::Float64           # = (n-1) / (2 box_half)
end

# Device handles, the DevicePool runtime, uploads and the march executor; device rays.
isdefined(@__MODULE__, :DevicePool) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_pool.jl"))
isdefined(@__MODULE__, :CameraRays) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_camera.jl"))

"""
    DeviceFluidGrid

Per-device pre-uploaded copies of a `UniformFluidGrid` (device => Float32 (8,n,n,n)
array), so repeated renders / both GPUs reuse one upload. Built by `upload_grid`, or
cached per pool by the drivers when they are passed a host grid or a snapshot.
"""
struct DeviceFluidGrid{D<:AbstractDict}
    dev_data::D
    n::Int
    xo::Float64
    inv_dx::Float64
end

"""
    resample_fluid_uniform(snap::AthenaKSnapshot, box_half, n) -> UniformFluidGrid

Resample the AMR primitives onto a uniform `n^3` Cartesian box spanning
`[-box_half, box_half]^3` (host, threaded over z-slabs). Nodes outside every meshblock
stay zero (treated as vacuum in-kernel).
"""
function resample_fluid_uniform(snap::AthenaKSnapshot, box_half::Real, n::Integer)
    n = Int(n); box_half = Float64(box_half)
    host = Array{Float32}(undef, 8, n, n, n)
    dx = 2 * box_half / (n - 1)
    Threads.@threads for k in 1:n
        z = -box_half + (k - 1) * dx
        @inbounds for j in 1:n
            y = -box_half + (j - 1) * dx
            for i in 1:n
                x = -box_half + (i - 1) * dx
                s = sample_primitives(snap, SVector(0.0, x, y, z))
                host[1, i, j, k] = s.rho
                host[2, i, j, k] = s.uu[1]
                host[3, i, j, k] = s.uu[2]
                host[4, i, j, k] = s.uu[3]
                host[5, i, j, k] = s.eint
                host[6, i, j, k] = s.bb[1]
                host[7, i, j, k] = s.bb[2]
                host[8, i, j, k] = s.bb[3]
            end
        end
    end
    return UniformFluidGrid(host, n, box_half, -box_half, (n - 1) / (2 * box_half))
end

"""
    upload_grid(fg::UniformFluidGrid, devices) -> DeviceFluidGrid

Upload `fg.host` to each device of `devices` (a list or a `DevicePool`), concurrently.
"""
function upload_grid(fg::UniformFluidGrid, devices)
    return DeviceFluidGrid(upload_per_device(fg.host, devices), fg.n, fg.xo, fg.inv_dx)
end

"""
    resident_grid(pool, snap_or_grid; box_half=40.0, n=256) -> DeviceFluidGrid

The device fluid grid of a driver call: a `DeviceFluidGrid` as given; a `UniformFluidGrid`
uploaded, or an `AthenaKSnapshot` resampled (`box_half`, `n`) and uploaded, once per pool
and reused by later calls with the same host object.
"""
resident_grid(pool, g::DeviceFluidGrid; kw...) = g
resident_grid(pool, g::UniformFluidGrid; kw...) =
    resident(() -> upload_grid(g, pool), pool, g.host, :grid)
resident_grid(pool, s::AthenaKSnapshot; box_half::Real = 40.0, n::Integer = 256) =
    resident(() -> upload_grid(resample_fluid_uniform(s, box_half, n), pool), pool, s.data,
             (:uniform, Float64(box_half), Int(n)))

# ---------------------------------------------------------------- device physics
# The metric, geodesic RHS and RK4 step are the Geodesics functions on
# `KerrSchildCartesian`; the fluid decode, plasma cuts, coefficients and segment operator
# are the Radiation functions (the CPU marches call the same ones). gpu_metric.jl adds
# the Kerr stop rule.
isdefined(@__MODULE__, :_gimg_stop) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_metric.jl"))

# One fluid sample -> invariant (j, α) at KS position x on the RT lane (Float64, or Dual
# for the AD render through the parameter set `gp`): normal-frame primitive decode, then
# `unpolarized_coefficients`. A vacuum or cut sample gives (0, 0), a no-op segment.
@inline function _gimg_coeffs(prim::SVector{8,Float64}, x::SVector{4,Float64},
                              kcov::SVector{4,Float64}, gp)
    gcov, gcon = metric_and_inverse(KerrSchildCartesian(gp.bh_a), x)
    return _prim_coeffs(prim, kcov, gcov, gcon, gp)
end
@inline function _prim_coeffs(prim, kcov, gcov, gcon, gp)
    fv = primitive_four_vectors(prim[2], prim[3], prim[4], prim[6], prim[7], prim[8],
                                gcov, gcon)
    return unpolarized_coefficients(gp, prim[1], pgas_from_eint(prim[5], gp.plasma_gamma),
                                    fv, kcov, gp.nu_obs)
end

# ---------------------------------------------------------------- device: grid sample
@inline _gimg_corner(grid, i, j, k) =
    SVector{8,Float64}(ntuple(f -> Float64(@inbounds grid[f, i, j, k]), Val(8)))

# Trilinear sample of the uniform grid at CKS (x,y,z). Outside the box -> vacuum (zeros).
@inline function _gimg_sample(grid, ng, xo, inv_dx, x, y, z)
    fx = (x - xo) * inv_dx; fy = (y - xo) * inv_dx; fz = (z - xo) * inv_dx
    (fx < 0.0 || fy < 0.0 || fz < 0.0 ||
     fx > ng - 1 || fy > ng - 1 || fz > ng - 1) && return zero(SVector{8,Float64})
    i0 = min(unsafe_trunc(Int, fx), ng - 2)
    j0 = min(unsafe_trunc(Int, fy), ng - 2)
    k0 = min(unsafe_trunc(Int, fz), ng - 2)
    dxf = fx - i0; dyf = fy - j0; dzf = fz - k0
    il = i0 + 1; jl = j0 + 1; kl = k0 + 1; ih = il + 1; jh = jl + 1; kh = kl + 1
    c000 = _gimg_corner(grid, il, jl, kl); c100 = _gimg_corner(grid, ih, jl, kl)
    c010 = _gimg_corner(grid, il, jh, kl); c110 = _gimg_corner(grid, ih, jh, kl)
    c001 = _gimg_corner(grid, il, jl, kh); c101 = _gimg_corner(grid, ih, jl, kh)
    c011 = _gimg_corner(grid, il, jh, kh); c111 = _gimg_corner(grid, ih, jh, kh)
    c00 = c000 + (c100 - c000) * dxf; c10 = c010 + (c110 - c010) * dxf
    c01 = c001 + (c101 - c001) * dxf; c11 = c011 + (c111 - c011) * dxf
    c0 = c00 + (c10 - c00) * dyf; c1 = c01 + (c11 - c01) * dyf
    return c0 + (c1 - c0) * dzf
end

# ---------------------------------------------------------------- shared march body
# Termination: `_gimg_stop` (gpu_metric.jl), the shared `march_stop` rule of the CPU
# sampled tracer — horizon backstop, redshift cut, outbound-only escape.

# The entire unpolarized fused march for one ray: camera->source fixed-control RK4
# (step h=(r-r_h)/C), reverse-accumulated invariant intensity (see file header). The one
# body behind every unpolarized kernel variant, parameterized by
#   * `sample(y)::SVector{8,Float64}` — inlined isbits-capture closure supplying the
#     fluid at state y (uniform grid / time-blended window / native AMR),
#   * `T` — the accumulator eltype (Float64, or a ForwardDiff Dual for the AD render),
#   * `Tg` — the geodesic-lane eltype: the state y, step h and every march scalar run
#     in Tg ∈ {Float32, Float64} (matched ::Tg, no silent promotion); the state is
#     converted to Float64 ONCE per fluid-sample/RT step (identity when Tg == Float64),
#     so all RT/coefficient arithmetic stays Float64 (or Dual) in every mode.
@inline function _gimg_march(y0::SVector{8,Tg}, sample::F, ::Type{T},
                             gp, a::Tg, r_h::Tg, C::Tg, escape::Tg, R_em::Tg,
                             nmax) where {F,T,Tg}
    y = y0
    Iinv = zero(T)
    trans = one(T)
    r_prev = Tg(Inf)
    for _ in 1:nmax
        stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
        stop && break
        r_prev = r
        h = (r - r_h) / C
        prim = @inline sample(y)   # force-inline the sampler (see the _gpol_march note)
        if prim[1] > 0.0
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
            j_inv, a_inv = _gimg_coeffs(prim, x, kcov, gp)
            ds = Float64(h) * gp.L_unit / gp.nu_obs
            A, B = unpolarized_segment(j_inv, a_inv, ds)
            Iinv += trans * B
            trans *= A
        end
        y = rk4_step_fixed(KerrSchildCartesian(a), y, h)
    end
    return Iinv * gp.nu_obs^3
end

# ---------------------------------------------------------------- fused kernel
# One ray per thread on the uniform grid. The eltype(img) accumulator parameter makes
# the same kernel serve the plain Float64 render and the Dual-seeded AD render
# (gpu_image_ad.jl). The geodesic lane runs at eltype(eltype(y0s)) — the host passes
# the march scalars at that type — while the sampler position is converted to Float64
# (identity for fp64 rays), so the trilinear sample + RT stay Float64 in every mode.
@kernel function _fused_kernel!(img, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                                R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gimg_march(y0s[i],
                             y -> _gimg_sample(grid, ng, xo, inv_dx, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             eltype(img), gp, a, r_h, C, escape, R_em, nmax)
    end
end

# ---------------------------------------------------------------- host: queue order + launch
# `order` (a permutation of 1:nrays, see `queue_items`) is the order in which the pool
# queues the rays; each ray's output lands at its own index, so any order is bit-exact.

"""
    march_sort_order(counts) -> Vector{Int}

Permutation sorting rays by descending march trip count (`counts[i]` = steps ray `i`
marched, e.g. in the previous frame of a movie). Pass it as the `order` kwarg of the
`render_gpu_*` drivers: the longest rays are queued first, so they do not end the render
on one device while the others idle (longest-processing-time-first makespan), and each
warp's lanes march similar-length rays. Note: on the fused native-AMR path, camera-order
rays are spatially coherent and trip-count sorting scatters that sampling locality — the
default `cost_order` keeps pixels in tiles; the queue order is bit-exact per pixel
either way.
"""
march_sort_order(counts::AbstractVector) = sortperm(counts; rev = true)

# KA launch with an OPTIONAL workgroupsize. Results are invariant to the block size (one
# independent ray per thread, no cross-lane reductions), so this tunes only occupancy;
# `wg === nothing` reproduces the backend-default launch.
@inline _kern_launch!(kern, wg, cnt, args...) =
    wg === nothing ? kern(args...; ndrange = cnt) :
                     kern(args...; ndrange = cnt, workgroupsize = min(wg, cnt))

# ---------------------------------------------------------------- host driver
# The isbits device parameter set: `coefficient_params` of the same prepared
# SimCoeffConfig the CPU `render` builds (units, plasma model, cuts, and the nonthermal
# constants of the active populations only), plus the observing frequency and length
# unit. The coefficient functions read it by the SimCoeffConfig field names.
function _gimg_params(cfg)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    return merge(coefficient_params(BlackLightPlus._prepared_scc(cfg, u)),
                 (nu_obs = cfg.frequency_hz, L_unit = u.L_unit))
end

# The mass_scale-dependent unit fields of a parameter set (`_scc_config` convention).
_unit_fields(m_bh, mass_scale) = (u = units(m_bh, mass_scale);
    (d_unit = u.dens_unit, e_unit = u.dens_unit * BlackLightPlus.Constants.CL^2,
     b_unit = u.B_unit))

# Launch of the uniform-grid fused kernel for one chunk: `n` rays of `y0d` into `outd` on
# the lane's device. The geodesic-lane type is eltype(eltype(y0d)); the caller passes the
# march scalars at that type. The accumulator stays Float64 (from the output eltype).
_fused_launch(dgrid, gp, a, r_h, C, escape, R_em, nmax, wg) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_fused_kernel!(lane.backend), wg, n, outd, y0d, dgrid.dev_data[lane.dev],
                  gp, a, r_h, C, escape, R_em, Int(nmax), dgrid.n, dgrid.xo, dgrid.inv_dx, n)

# `critical_curve(a, θo)` of the latest (a, θo) (a 2×10⁵-radius scan), kept for reuse.
const _CURVE_LAST = Ref{Any}(nothing)
const _CURVE_LOCK = ReentrantLock()
function _critical_curve_cached(a::Float64, θo::Float64)
    @lock _CURVE_LOCK begin
        last = _CURVE_LAST[]
        (last !== nothing && last[1] == (a, θo)) || (_CURVE_LAST[] = last = ((a, θo), critical_curve(a, θo)))
        last[2]
    end
end

# Walk the Kerr critical curve at ≤ px/2 spacing on this camera's image plane (see
# `_rescue_pixels`), calling `f(iα, iβ)` with the image-matrix pixel of every sample
# (indices may fall outside 1:npix). Returns false when the curve is empty.
function _walk_critical_curve(f, a::Real, inc_deg::Real, fov::Real, npix::Integer)
    θo = clamp(deg2rad(Float64(inc_deg)), 0.02, π - 0.02)
    αB, βB = _critical_curve_cached(Float64(a), θo)
    isempty(αB) && return false
    px = Float64(fov) / npix
    half = Float64(fov) / 2
    m = length(αB)
    for s in 1:m
        s2 = s == m ? 1 : s + 1               # closed loop (upper + mirrored lower branch)
        x1 = αB[s];  y1 = βB[s]               # camera screen coords (see _rescue_pixels)
        x2 = αB[s2]; y2 = βB[s2]
        nsub = max(1, ceil(Int, hypot(x2 - x1, y2 - y1) / (px / 2)))
        for q in 0:nsub                       # walk the segment at ≤ px/2 spacing
            t = q / nsub
            f(floor(Int, (x1 + t * (x2 - x1) + half) / px) + 1,
              floor(Int, (y1 + t * (y2 - y1) + half) / px) + 1)
        end
    end
    return true
end

"""
    _rescue_pixels(a, inc_deg, fov, npix; rescue_px=2) -> Vector{Int}

Flat ray indices (linear indices of the npix×npix image matrix) of every pixel within
`rescue_px` pixels (Chebyshev dilation) of the Kerr critical curve — the Lyapunov-
unstable set where an fp32 geodesic march fails. The `critical_curve(a, θo)` Bardeen
screen is this camera's image plane, α_cam = α_Bardeen, β_cam = β_Bardeen (a camera
ray at (α, β) carries λ = L_z/E = -α sin θo, Bardeen's λ; the curve is β-symmetric). Finite camera distance (O(M²/d) shift) and the 0.02 rad near-polar clamp
are both absorbed by the ≥2 px dilation. Returns Int[] when the curve misses the frame.
"""
function _rescue_pixels(a::Real, inc_deg::Real, fov::Real, npix::Integer; rescue_px::Int = 2)
    npix = Int(npix)
    mask = falses(npix, npix)                 # [iβ, iα] — the image-matrix layout
    _walk_critical_curve(a, inc_deg, fov, npix) do iα, iβ
        for dα in -rescue_px:rescue_px, dβ in -rescue_px:rescue_px
            jα = iα + dα; jβ = iβ + dβ
            (1 <= jα <= npix && 1 <= jβ <= npix) && (mask[jβ, jα] = true)
        end
    end || return Int[]
    return findall(vec(mask))
end

# Distance, in cells, from every `cell`×`cell`-pixel cell [iβ, iα] of the image to the
# nearest cell the critical curve crosses: two raster passes propagating the nearest
# crossed cell over the 8-neighbourhood. Cells are infinitely far when the curve misses.
function _curve_cell_distance(a, inc_deg, fov, npix::Int, cell::Int)
    n = cld(npix, cell)
    near = fill((0, 0), n, n)                 # nearest crossed cell; (0, 0): none yet
    _walk_critical_curve(a, inc_deg, fov, npix) do iα, iβ
        if 1 <= iα <= npix && 1 <= iβ <= npix
            i = (iβ - 1) ÷ cell + 1; j = (iα - 1) ÷ cell + 1
            near[i, j] = (i, j)
        end
    end
    for j in 1:n, i in 1:n
        _relax!(near, i, j, i - 1, j); _relax!(near, i, j, i, j - 1)
        _relax!(near, i, j, i - 1, j - 1); _relax!(near, i, j, i + 1, j - 1)
    end
    for j in n:-1:1, i in n:-1:1
        _relax!(near, i, j, i + 1, j); _relax!(near, i, j, i, j + 1)
        _relax!(near, i, j, i + 1, j + 1); _relax!(near, i, j, i - 1, j + 1)
    end
    return [near[i, j][1] == 0 ? Inf : sqrt(Float64(_dist2(near[i, j], i, j))) for i in 1:n, j in 1:n]
end

_dist2(c, i, j) = c[1] == 0 ? typemax(Int) : (i - c[1])^2 + (j - c[2])^2

# Adopt the nearest crossed cell of neighbour (k, l) at cell (i, j) if it is closer.
@inline function _relax!(near, i, j, k, l)
    n = size(near, 1)
    (1 <= k <= n && 1 <= l <= n) || return nothing
    @inbounds c = near[k, l]
    @inbounds _dist2(c, i, j) < _dist2(near[i, j], i, j) && (near[i, j] = c)
    return nothing
end

const _COST_ORDER_LAST = Ref{Any}(nothing)     # (key, order) of the latest camera
const _COST_ORDER_LOCK = ReentrantLock()

"""
    cost_order(a, inc_deg, fov, npix; tile=16) -> Vector{Int}

Default queue order of a full-frame Kerr render, longest predicted work first: the
pixels grouped in `tile`×`tile` image tiles (camera order inside a tile, keeping the
fluid-sampling locality of neighbouring rays), the tiles in descending predicted cost. A
ray marches longest near the critical curve, where its orbit count grows as the
logarithm of the inverse impact-parameter offset, so a pixel's predicted cost is
log(n / (d + 1/4)) with d its distance to the curve on a grid of `max(1, tile ÷ 4)`-pixel
cells and n the grid size. The order of the latest camera is kept for reuse. Any order
gives the same image; this one lets the longest rays start first.
"""
function cost_order(a::Real, inc_deg::Real, fov::Real, npix::Integer; tile::Int = 16)
    key = (Float64(a), Float64(inc_deg), Float64(fov), Int(npix), tile)
    @lock _COST_ORDER_LOCK begin
        last = _COST_ORDER_LAST[]
        last !== nothing && last[1] == key && return last[2]
    end
    npix = Int(npix)
    cell = max(1, tile ÷ 4)
    D = _curve_cell_distance(a, inc_deg, fov, npix, cell)
    nc = size(D, 1)
    nt = cld(npix, tile)
    tcost = zeros(nt, nt)                     # [β tile, α tile]: Σ over its pixels
    for j in 1:nc, i in 1:nc                  # a cell's pixels all lie in one tile
        npx = (min(i * cell, npix) - (i - 1) * cell) * (min(j * cell, npix) - (j - 1) * cell)
        tcost[(i - 1) * cell ÷ tile + 1, (j - 1) * cell ÷ tile + 1] += npx * log(nc / (D[i, j] + 0.25))
    end
    order = Vector{Int}(undef, npix * npix)
    k = 0
    for t in sortperm(vec(tcost); rev = true)
        ta, tb = divrem(t - 1, nt)            # t: linear index of tcost[β tile, α tile]
        for iα in ta * tile + 1:min((ta + 1) * tile, npix), iβ in tb * tile + 1:min((tb + 1) * tile, npix)
            order[k += 1] = (iα - 1) * npix + iβ
        end
    end
    @lock _COST_ORDER_LOCK (_COST_ORDER_LAST[] = (key, order))
    return order
end

# Queue order of a full-frame Kerr render: the caller's `order`, else `cost_order`.
_frame_items(order, cfg, npix) = order === nothing ?
    cost_order(cfg.bh_a, cfg.camera.inclination_deg, cfg.camera.fov, npix) :
    queue_items(order, npix^2)

# The `:mixed` queue: the critical-curve pixels first, marched in fp64, then the other
# items in `items` order in fp32 — one queue, the same image as an fp32 sweep of every
# pixel overwritten by an fp64 sweep of the rescued ones.
function _mixed_parts(items, pix, src64, src32, launch64, launch32)
    rescued = falses(length(items)); rescued[pix] .= true
    return [MarchPart(pix, src64, launch64),
            MarchPart(filter(i -> !rescued[i], items), src32, launch32)]
end

"""
    render_gpu_fused(cfg, snap_or_grid; devices=default_pool(), nmax=cfg.max_steps,
                     C=80, box_half=40.0, n=256, precision=:f64, rescue_px=2)
        -> Matrix{Float64}

Fused single-pass GPU render of the npix×npix unpolarized image, in the layout of
`render(...;polarized=false).I`. `snap_or_grid`: `AthenaKSnapshot` (resampled + uploaded
once per device pool), `UniformFluidGrid` (uploaded once per pool), or a pre-uploaded
`DeviceFluidGrid`. `devices` is a device list or a `DevicePool`; rays are queued in
chunks that the devices pull as they free up. `order` sets the queue order (default
`cost_order`: longest predicted tiles first; any order is bit-exact per pixel); `wg`
overrides the block size (occupancy only). `precision`: `:f64` (default, bitwise
unchanged) · `:mixed` (fp32 geodesic lane, Float64 fluid-sample/RT; the near-critical-
curve pixels are marched in fp64 instead, queued first — `rescued_frac::Ref{Float64}`
receives their fraction) · `:f32_raw` (no rescue, DIAGNOSTIC only — near-critical pixels
carry O(1) errors).
"""
function render_gpu_fused(cfg, snap_or_grid; devices = default_pool(),
                          nmax::Int = cfg.max_steps,   # step limit wired from RunConfig, as render()
                          C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                          order = nothing, wg = nothing,
                          precision::Symbol = :f64, rescue_px::Int = 2,
                          rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_fused: precision must be :f64, :mixed or :f32_raw, got :$precision")
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    dgrid = resident_grid(pool, snap_or_grid; box_half = box_half, n = n)

    gp = _gimg_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = cam.npix^2
    items = _frame_items(order, cfg, cam.npix)
    img = Vector{Float64}(undef, nrays)
    f64 = _fused_launch(dgrid, gp, a, r_h, Cf, escape, R_em, nmax, wg)
    # fp32 geodesic lane: the fp64 rays are narrowed as the device makes them; the host
    # constructs every geodesic-lane scalar as Float32 (no in-kernel widening), while gp
    # (units, frequency, fp64 spin gp.bh_a for the metric/redshift) stays Float64.
    f32() = _fused_launch(dgrid, gp, Float32(a), Float32(r_h), Float32(Cf),
                          Float32(escape), Float32(R_em), nmax, wg)
    frac = 0.0
    if precision === :f64
        march!((img,), pool, [MarchPart(items, CameraRays(st, cam), f64)])
    elseif precision === :f32_raw
        march!((img,), pool, [MarchPart(items, CameraRays(st, cam, Float32), f32())])
    else
        pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                             rescue_px = rescue_px)
        march!((img,), pool, _mixed_parts(items, pix, CameraRays(st, cam),
                                          CameraRays(st, cam, Float32), f64, f32()))
        frac = length(pix) / nrays
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    return reshape(img, cam.npix, cam.npix)
end

# ================================================================ K-VARIANT batched RT
# mass_scale-ladder batched unpolarized render: one ray/thread marches the geodesic once
# and samples the fluid once per step (both mass_scale-independent) while accumulating K
# intensity variants. mass_scale enters the physics ONLY through units(m_bh, mass_scale)
# -> (d_unit, e_unit, b_unit); every other device scalar is shared across all K.
# Unlike the Dual-seeded AD render, an SVector-valued param lane cannot ride the generic
# kernel (SVector is not a Number: scalar guards, elementwise ratios and one(T) all
# break), so only the ACCUMULATOR is vectorized (SVector{K,Float64}) and each variant
# runs the unpolarized coefficients and segment operator with its own units tuple, so
# lane k is the standalone render of that mass_scale.
@inline function _gimg_march_kv(y0::SVector{8,Tg}, sample::F, gps::NTuple{K,GP},
                                a::Tg, r_h::Tg, C::Tg, escape::Tg, R_em::Tg,
                                nmax) where {F,K,GP,Tg}
    y = y0
    Iinv  = zero(SVector{K,Float64})
    trans = SVector{K,Float64}(ntuple(_ -> 1.0, Val(K)))
    r_prev = Tg(Inf)
    nu_obs = gps[1].nu_obs; L_unit = gps[1].L_unit    # mass_scale-independent (shared lane)
    for _ in 1:nmax
        stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
        stop && break
        r_prev = r
        h = (r - r_h) / C
        prim = @inline sample(y)
        if prim[1] > 0.0
            x    = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
            ds   = Float64(h) * L_unit / nu_obs
            # per-variant (A,B) with mass_scale k's (d,e,b)_unit — the ops of a standalone
            # render of variant k. The metric is re-evaluated per variant: holding it live
            # across the K lanes costs more registers than recomputing it.
            ab = ntuple(Val(K)) do k
                j_inv, a_inv = _gimg_coeffs(prim, x, kcov, gps[k])
                unpolarized_segment(j_inv, a_inv, ds)
            end
            A = SVector{K,Float64}(ntuple(k -> ab[k][1], Val(K)))
            B = SVector{K,Float64}(ntuple(k -> ab[k][2], Val(K)))
            Iinv  = Iinv + trans .* B          # per lane: Iinv[k] += trans[k]*B[k]
            trans = trans .* A                 # per lane: trans[k] *= A[k]
        end
        y = rk4_step_fixed(KerrSchildCartesian(a), y, h)
    end
    return Iinv .* nu_obs^3
end

# K-variant fused kernel: the SAME geodesic march + uniform-grid sampler as _fused_kernel!,
# but the per-thread output is an SVector{K,Float64} of intensity variants (img::CuArray of
# SVector{K,Float64}). The K-length param tuple `gps` carries the mass_scale ladder.
@kernel function _fused_kernel_kv!(img, @Const(y0s), @Const(grid), gps, a, r_h, C, escape,
                                   R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gimg_march_kv(y0s[i],
                                y -> _gimg_sample(grid, ng, xo, inv_dx, Float64(y[2]),
                                                  Float64(y[3]), Float64(y[4])),
                                gps, a, r_h, C, escape, R_em, nmax)
    end
end

# Launch of the K-variant fused kernel for one chunk; SVector{K,Float64} output per pixel.
_fused_launch_kv(dgrid, gps, a, r_h, C, escape, R_em, nmax) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_fused_kernel_kv!(lane.backend), nothing, n, outd, y0d,
                  dgrid.dev_data[lane.dev], gps, a, r_h, C, escape, R_em, Int(nmax),
                  dgrid.n, dgrid.xo, dgrid.inv_dx, n)

"""
    render_gpu_fused_kvariant(cfg, snap_or_grid, mass_scales::Vector{Float64};
                              devices=default_pool(), nmax=cfg.max_steps,
                              C=80, box_half=40.0, n=256) -> Vector{Matrix{Float64}}

K-variant batched unpolarized fused render: one GPU pass returns K images, one per
`mass_scales` value, sharing the geodesic march and fluid sample across all variants.
Image k is bitwise-identical to `render_gpu_fused` with `mass_scale_cgs = mass_scales[k]`
(mass_scale enters only via `units` -> d/e/b_unit). Stokes-I, uniform-resample fluid,
`:f64` only. `snap_or_grid` as in `render_gpu_fused`; the same device fluid buffer feeds
all K variants. Returns a length-K `Vector{Matrix{Float64}}` in `render_gpu_fused` layout.
"""
function render_gpu_fused_kvariant(cfg, snap_or_grid, mass_scales::Vector{Float64};
                                   devices = default_pool(), nmax::Int = cfg.max_steps,
                                   C::Real = 80, box_half::Real = 40.0, n::Integer = 256)
    K = length(mass_scales)
    K >= 1 || error("render_gpu_fused_kvariant: mass_scales must be nonempty")
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    dgrid = resident_grid(pool, snap_or_grid; box_half = box_half, n = n)

    gp0 = _gimg_params(cfg)
    # per-variant param tuples differ ONLY in (d_unit,e_unit,b_unit) — exactly what
    # _gimg_params(cfg_k) with mass_scale_cgs = mass_scales[k] produces (L_unit is
    # mass_scale-independent, so gp0.L_unit is shared); the merge preserves gp0's type.
    gps = ntuple(k -> merge(gp0, _unit_fields(cfg.m_bh_cgs, mass_scales[k])), K)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = cam.npix^2
    img = Vector{SVector{K,Float64}}(undef, nrays)
    march!((img,), pool, [MarchPart(_frame_items(nothing, cfg, cam.npix), CameraRays(st, cam),
                                    _fused_launch_kv(dgrid, gps, a, r_h, Cf, escape, R_em, nmax))])
    imgmat = reshape(img, cam.npix, cam.npix)
    return [getindex.(imgmat, k) for k in 1:K]
end

# ================================================================ SLOW LIGHT (opt-in)
# Time-dependent GRMHD frames along the geodesic: a separate kernel entry point sharing
# the one _gimg_march body with the fast kernel (only the sampler differs), so the
# fast-light path stays bit-identical. The device reads two uploaded grids and lerps
# their samples on the physical dump-time axis (time_series.jl :: sample_primitives_t):
# the ray starts at X⁰ = y[1] = 0 on the camera and X⁰ decreases along the backward
# trace (X⁰ = minus the light-travel time), so the emission time is t_emit = t_obs + X⁰
# and the blend weight on frame B is
#     w = clamp((t_emit - t_a) * inv_dt, 0, 1) = clamp((t0 + X⁰) * inv_dt, 0, 1)
# with t0 = t_obs - t_a precomputed on the host. Fast-light limit: inv_dt = 0 forces
# w == 0 exactly, so the kernel reproduces render_gpu_fused on grid A bit-for-bit.

"""
    DeviceFluidWindow

A bracketing pair of pre-uploaded `DeviceFluidGrid` buffers (`a` at coordinate time
`t_a`, `b` at `t_b`) plus `inv_dt = 1/(t_b - t_a)` for the in-kernel time blend.
`inv_dt == 0.0` is the fast-light limit (Δt_dump -> ∞ or a degenerate window). The
two grids must share `(n, xo, inv_dx)`.
"""
struct DeviceFluidWindow
    a::DeviceFluidGrid
    b::DeviceFluidGrid
    t_a::Float64
    inv_dt::Float64
end

# t_b == Inf (or <= t_a) => inv_dt = 0 (fast-light limit); else 1/(t_b - t_a).
_win_inv_dt(t_a, t_b) = (isfinite(t_b) && t_b > t_a) ? 1.0 / (t_b - t_a) : 0.0

# Default observation epoch for a window (fluid or AMR): its UPPER end t_b — the
# SnapshotSeries default `t_obs = times[end]` (the dump window trails the observation).
# For inv_dt == 0 (fast light) w ≡ 0 whatever the epoch, so t_a serves.
_win_t_obs(win) = win.inv_dt > 0.0 ? win.t_a + 1.0 / win.inv_dt : win.t_a

"""
    scale_grid_density(fg::UniformFluidGrid, factor) -> UniformFluidGrid

Copy `fg` with the density channel (feature 1) scaled by `factor` — the uniform-grid
image of `scale_density` (a global density modulation), for building synthetic
slow-light windows cheaply (resample once, scale per frame). `factor == 1` returns
`fg` unchanged.
"""
function scale_grid_density(fg::UniformFluidGrid, factor::Real)
    factor == 1.0 && return fg
    host = copy(fg.host)
    @inbounds @views host[1, :, :, :] .*= Float32(factor)
    return UniformFluidGrid(host, fg.n, fg.box_half, fg.xo, fg.inv_dx)
end

"""
    upload_window(fgA, fgB, t_a, t_b, devices) -> DeviceFluidWindow

Upload two `UniformFluidGrid`s (frames at `t_a < t_b`) to `devices` as a bracketing
window. If `fgA === fgB` the single upload is shared. Pass `t_b = Inf` for the
fast-light limit (inv_dt = 0).
"""
function upload_window(fgA::UniformFluidGrid, fgB::UniformFluidGrid, t_a::Real, t_b::Real, devices)
    # The slow-light kernels index BOTH grids with frame A's (n, xo, inv_dx) (drivers pass
    # ga.* for both blk arrays), so a geometry mismatch is an out-of-bounds device read or a
    # silently misplaced frame B. Enforce the DeviceFluidWindow precondition (docstring above).
    (fgA.n == fgB.n && fgA.xo == fgB.xo && fgA.inv_dx == fgB.inv_dx) ||
        error("upload_window: the two frames must share (n, xo, inv_dx); got " *
              "($(fgA.n), $(fgA.xo), $(fgA.inv_dx)) vs ($(fgB.n), $(fgB.xo), $(fgB.inv_dx))")
    da = upload_grid(fgA, devices)
    db = fgA === fgB ? da : upload_grid(fgB, devices)
    return DeviceFluidWindow(da, db, Float64(t_a), _win_inv_dt(Float64(t_a), Float64(t_b)))
end

"""
    upload_series_window(series::SnapshotSeries, ia, devices; box_half=40.0, n=256)
        -> DeviceFluidWindow

Resample+upload the bracketing dump pair `(ia, ia+1)` of `series` onto two uniform
`DeviceFluidGrid` buffers. Identical/object-shared frames reuse one upload. A
single-frame series yields inv_dt = 0 (fast light).
"""
function upload_series_window(series::SnapshotSeries, ia::Integer, devices;
                              box_half::Real = 40.0, n::Integer = 256)
    N = length(series)
    ia = Int(ia); ib = min(ia + 1, N)
    A = get_frame(series, ia); B = get_frame(series, ib)
    t_a = series.times[ia]; t_b = ib > ia ? series.times[ib] : Inf
    fgA = resample_fluid_uniform(A, box_half, n)
    fgB = (A === B) ? fgA : resample_fluid_uniform(B, box_half, n)
    return upload_window(fgA, fgB, t_a, t_b, devices)
end

"""
    advance_series_window(win::DeviceFluidWindow, series, ia, devices; box_half=40.0, n=256)
        -> DeviceFluidWindow

Slide a slow-light window forward by one dump: build the bracket `(ia, ia+1)` reusing
`win.b` (the already-uploaded upper frame) as the new lower frame, so only the new upper
frame `ia+1` is resampled + uploaded; the reused grid is the same device buffer, so the
rendered frame is bit-identical to a fresh upload. Advance monotonically, passing the
`win` whose upper frame IS dump `ia`; `box_half`/`n` must match how `win` was built.
At the series end the window degenerates to inv_dt = 0 (fast light).
"""
function advance_series_window(win::DeviceFluidWindow, series::SnapshotSeries, ia::Integer,
                               devices; box_half::Real = 40.0, n::Integer = 256)
    N = length(series); ia = Int(ia); ib = min(ia + 1, N)
    newA = win.b                              # promote the uploaded upper frame — no re-resample
    if ib > ia
        newB = upload_grid(resample_fluid_uniform(get_frame(series, ib), box_half, n), devices)
        (newB.n == newA.n && newB.xo == newA.xo && newB.inv_dx == newA.inv_dx) ||
            error("advance_series_window: resampled frame $(ib) geometry must match the promoted " *
                  "frame ($(newB.n),$(newB.xo),$(newB.inv_dx)) vs ($(newA.n),$(newA.xo)," *
                  "$(newA.inv_dx)) — check box_half/n match how win was built")
    else
        newB = newA
    end
    t_a = series.times[ia]; t_b = ib > ia ? series.times[ib] : Inf
    return DeviceFluidWindow(newA, newB, Float64(t_a), _win_inv_dt(Float64(t_a), Float64(t_b)))
end

# Time-blended trilinear sample: lerp grid-A and grid-B samples with weight w on B.
# The clamped window ends short-circuit to a single grid read: w == 0 returns
# grid-A's sample EXACTLY (fast-light limit) and w == 1 grid-B's — both weights are
# attained exactly by the kernels' clamp, so the branches are bit-identical to the
# blend while halving the global-memory traffic on clamped segments.
@inline function _gimg_sample_t(gA, gB, ng, xo, inv_dx, w, x, y, z)
    w == 1.0 && return _gimg_sample(gB, ng, xo, inv_dx, x, y, z)
    sA = _gimg_sample(gA, ng, xo, inv_dx, x, y, z)
    w == 0.0 && return sA
    sB = _gimg_sample(gB, ng, xo, inv_dx, x, y, z)
    return (1.0 - w) .* sA .+ w .* sB
end

# SLOW-LIGHT fused kernel: the SAME _gimg_march body with the sampler swapped for the
# time-blended _gimg_sample_t (w at emission time t_obs + X⁰; t0 = t_obs - t_a — see
# the SLOW LIGHT header), so inv_dt = 0 reproduces _fused_kernel! on grid-A bit-for-bit.
@kernel function _fused_kernel_sl!(img, @Const(y0s), @Const(gridA), @Const(gridB),
                                   t0, inv_dt, gp, a, r_h, C, escape, R_em, nmax,
                                   ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gimg_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gimg_sample_t(gridA, gridB, ng, xo, inv_dx, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             eltype(img), gp, a, r_h, C, escape, R_em, nmax)
    end
end

# Launch of the slow-light fused kernel for one chunk (window grids resident per device).
_fused_launch_sl(win, t0, gp, a, r_h, C, escape, R_em, nmax, wg) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_fused_kernel_sl!(lane.backend), wg, n, outd, y0d, win.a.dev_data[lane.dev],
                  win.b.dev_data[lane.dev], t0, win.inv_dt, gp, a, r_h, C, escape, R_em,
                  Int(nmax), win.a.n, win.a.xo, win.a.inv_dx, n)

"""
    render_gpu_fused_slow(cfg, win::DeviceFluidWindow; devices=default_pool(),
                          nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                          time_offset=0.0) -> Matrix{Float64}

Fused single-pass SLOW-LIGHT render: each ray blends the two `DeviceFluidWindow`
buffers at its EMISSION time t_emit = t_obs + X⁰ with
w = clamp((t_emit - t_a) * inv_dt, 0, 1) — the physical dump-time axis of
`sample_primitives_t`. `t_obs` is the observation epoch at the camera (default:
the window's upper end, matching the `SnapshotSeries` default); `time_offset`
advances it (movie frame time). `win.inv_dt == 0` (fast-light limit) reproduces
`render_gpu_fused(cfg, win.a)` bit-for-bit. Accepts `order`/`wg` as `render_gpu_fused`.
"""
function render_gpu_fused_slow(cfg, win::DeviceFluidWindow; devices = default_pool(),
                               nmax::Int = cfg.max_steps, C::Real = 80,
                               t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                               order = nothing, wg = nothing)
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    gp = _gimg_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(win.a.xo)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    nrays = cam.npix^2
    img = Vector{Float64}(undef, nrays)
    march!((img,), pool, [MarchPart(_frame_items(order, cfg, cam.npix), CameraRays(st, cam),
                                    _fused_launch_sl(win, t0, gp, a, r_h, Cf, escape, R_em,
                                                     nmax, wg))])
    return reshape(img, cam.npix, cam.npix)
end
