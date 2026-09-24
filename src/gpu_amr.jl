# gpu_amr.jl — fused single-pass GPU imager sampling the AMR meshblock hierarchy
# DIRECTLY on device (no uniform resample; removes the resample fluid-representation
# delta, which a Faraday-thick deck amplifies into a global EVPA rotation). Only the
# fluid sample differs from gpu_image.jl / gpu_pol.jl; all other device helpers are
# shared by include order. Locate: an O(1) dense finest-block index grid (every AthenaK
# block face is an exact multiple of the finest block size, so each index cell lies in
# exactly one leaf), cross-checked against an exact linear-scan port of `_find_meshblock`.
# The in-block trilinear reproduces CPU `sample_primitives` bit-for-bit. CUDA optional:
# NOT included from BlackLightPlus.jl; include AFTER gpu_image.jl and gpu_pol.jl.

using StaticArrays
using KernelAbstractions
using CUDA
using LinearAlgebra: dot
using BlackLightPlus.Simulation: AthenaKSnapshot, SnapshotSeries, get_frame, leaf_table
using BlackLightPlus: AMRBlock, camera_rays_at
import BlackLightPlus: stream_blocks!

# ---------------------------------------------------------------- device AMR grid
"""
    DeviceAMRGrid

Per-device pre-uploaded AMR fluid + geometry (built by `upload_amr`; each Dict is keyed
by device id so repeated renders reuse one upload). Device layout: `dev_blk`
(8,ni,nj,nk,nmb) Float32 primitives, feature-fastest for coalesced corner reads;
`dev_xlft`/`dev_dxb` (3,nmb) per-block first cell centre / cell spacing; `dev_lo/hi`
(3,nmb) face extents (scan oracle only); `dev_leaf` (N1,N2,N3) Int32 finest-block index
grid (owning leaf mb, 0 = vacuum); `amrp` isbits NamedTuple of locate scalars.
"""
struct DeviceAMRGrid{DB<:AbstractDict,DX<:AbstractDict,DL<:AbstractDict}
    dev_blk::DB
    dev_xlft::DX
    dev_dxb::DX
    dev_lo::DX
    dev_hi::DX
    dev_leaf::DL
    amrp::NamedTuple
end

# Device leaf table = the one host builder (athenak_reader.jl :: leaf_table) repackaged
# as (leaf array, isbits amrp scalars) for upload. The host builder signals a degenerate
# / oversized / non-finest-aligned mesh by returning `nothing` and falls back to its
# linear scan — the device sampler has NO scan fallback, so that case must error loudly.
function _build_leaf_table(snap::AthenaKSnapshot)
    t = leaf_table(snap)
    t === nothing &&
        error("AMR mesh unsuitable for the device leaf table (degenerate, oversized, or not " *
              "origin/finest-aligned dump); the device sampler has no scan fallback (use the CPU render)")
    amrp = (ni = snap.ni, nj = snap.nj, nk = snap.nk, nmb = t.nmb,
            x1min = t.x1min, x2min = t.x2min, x3min = t.x3min,
            inv_fdx1 = t.inv_fdx1, inv_fdx2 = t.inv_fdx2, inv_fdx3 = t.inv_fdx3,
            N1 = t.N1, N2 = t.N2, N3 = t.N3)
    return t.leaf, amrp
end

"""
    upload_amr(snap::AthenaKSnapshot, devices; upload_bounds=true) -> DeviceAMRGrid

Permute the reader's `(ni,nj,nk,nmb,8)` primitives to the coalesced `(8,ni,nj,nk,nmb)`
layout, build the finest-block index grid, and upload fluid + per-block geometry to
each device of `devices` (a list or a `DevicePool`; one copy per device, made
concurrently; reusable across renders). The render kernels use only the leaf table +
`amrp`; `dev_lo/dev_hi` feed the probe/scan-oracle path, so pass `upload_bounds=false`
for a render-only grid.
"""
function upload_amr(snap::AthenaKSnapshot, devices; upload_bounds::Bool = true)
    blk_host  = permutedims(snap.data, (5, 1, 2, 3, 4))   # (8, ni, nj, nk, nmb)
    xlft_host = reduce(hcat, snap.x_left)                  # (3, nmb)
    dxb_host  = reduce(hcat, snap.dx)
    leaf_host, amrp = _build_leaf_table(snap)

    dblk = upload_per_device(blk_host, devices)
    dxl  = upload_per_device(xlft_host, devices)
    ddx  = upload_per_device(dxb_host, devices)
    dlf  = upload_per_device(leaf_host, devices)
    dlo  = upload_bounds ? upload_per_device(reduce(hcat, snap.ext_lo), devices) : empty(dxl)
    dhi  = upload_bounds ? upload_per_device(reduce(hcat, snap.ext_hi), devices) : empty(dxl)
    return DeviceAMRGrid(dblk, dxl, ddx, dlo, dhi, dlf, amrp)
end

"""
    resident_amr(pool, snap_or_amrgrid) -> DeviceAMRGrid

The render grid of a driver call: a `DeviceAMRGrid` as given, or the snapshot uploaded
(`upload_bounds=false`: the render kernels never read the face bounds) once per pool.
"""
resident_amr(pool, dg::DeviceAMRGrid) = dg
resident_amr(pool, snap::AthenaKSnapshot) =
    resident(() -> upload_amr(snap, pool; upload_bounds = false), pool, snap.data, :amr)

# ---------------------------------------------------------------- device: locate
# (a) EXACT port of _find_meshblock: linear scan, lo < xyz <= hi, later block wins,
#     0 outside every block. Bit-for-bit correctness oracle.
@inline function _gamr_locate_scan(lo, hi, nmb, x, y, z)
    mb = 0
    @inbounds for m in 1:nmb
        if lo[1, m] < x <= hi[1, m] && lo[2, m] < y <= hi[2, m] && lo[3, m] < z <= hi[3, m]
            mb = m
        end
    end
    return mb
end

# (b) O(1) dense finest-block index grid. f>=0 is guaranteed before the trunc, so
#     unsafe_trunc == floor. Out of range -> 0 (vacuum), matching _find_meshblock==0.
@inline function _gamr_locate_fast(leaf, amrp, x, y, z)
    f1 = (x - amrp.x1min) * amrp.inv_fdx1
    f2 = (y - amrp.x2min) * amrp.inv_fdx2
    f3 = (z - amrp.x3min) * amrp.inv_fdx3
    (f1 < 0.0 || f1 >= amrp.N1 || f2 < 0.0 || f2 >= amrp.N2 || f3 < 0.0 || f3 >= amrp.N3) &&
        return Int32(0)
    i1 = unsafe_trunc(Int, f1); i2 = unsafe_trunc(Int, f2); i3 = unsafe_trunc(Int, f3)
    return @inbounds leaf[i1 + 1, i2 + 1, i3 + 1]
end

# ---------------------------------------------------------------- device: sample
@inline _gamr_corner(blk, i, j, k, mb) =
    SVector{8,Float64}(ntuple(f -> Float64(@inbounds blk[f, i, j, k, mb]), Val(8)))

# In-block clamped trilinear of the 8 primitives at CKS (x,y,z) inside meshblock `mb`,
# byte-for-byte the sample_primitives interior: DIVIDE by dx (not inv_dx multiply),
# FLOOR (fi can be slightly negative in the outer low half-cell), and BOTH clamps.
# Shared by the leaf-locate `_gamr_sample` (Kerr AMR kernels) and the scan-locate
# `_gamr_sample_scan` (generic-spacetime kernels, gpu_spacetime.jl).
@inline function _gamr_lerp(blk, xlft, dxb, ni, nj, nk, mb::Int, x, y, z)
    gx = (x - @inbounds(xlft[1, mb])) / @inbounds(dxb[1, mb])
    gy = (y - @inbounds(xlft[2, mb])) / @inbounds(dxb[2, mb])
    gz = (z - @inbounds(xlft[3, mb])) / @inbounds(dxb[3, mb])
    il = clamp(unsafe_trunc(Int, floor(gx)), 0, ni - 2); d1 = clamp(gx - il, 0.0, 1.0)
    jl = clamp(unsafe_trunc(Int, floor(gy)), 0, nj - 2); d2 = clamp(gy - jl, 0.0, 1.0)
    kl = clamp(unsafe_trunc(Int, floor(gz)), 0, nk - 2); d3 = clamp(gz - kl, 0.0, 1.0)
    i1 = il + 1; j1 = jl + 1; k1 = kl + 1; i2 = i1 + 1; j2 = j1 + 1; k2 = k1 + 1
    c000 = _gamr_corner(blk, i1, j1, k1, mb); c100 = _gamr_corner(blk, i2, j1, k1, mb)
    c010 = _gamr_corner(blk, i1, j2, k1, mb); c110 = _gamr_corner(blk, i2, j2, k1, mb)
    c001 = _gamr_corner(blk, i1, j1, k2, mb); c101 = _gamr_corner(blk, i2, j1, k2, mb)
    c011 = _gamr_corner(blk, i1, j2, k2, mb); c111 = _gamr_corner(blk, i2, j2, k2, mb)
    c00 = c000 + (c100 - c000) * d1                        # lerp along x1 (i)
    c10 = c010 + (c110 - c010) * d1
    c01 = c001 + (c101 - c001) * d1
    c11 = c011 + (c111 - c011) * d1
    c0  = c00 + (c10 - c00) * d2                           # lerp along x2 (j)
    c1  = c01 + (c11 - c01) * d2
    return c0 + (c1 - c0) * d3                             # lerp along x3 (k)
end

# Native AMR trilinear sample at CKS (x,y,z) -> SVector{8,Float64}
# (dens,uu1,uu2,uu3,eint,bb1,bb2,bb3), byte-for-byte sample_primitives. Uses the fast
# locate; vacuum (locate 0) returns zeros.
@inline function _gamr_sample(blk, xlft, dxb, leaf, amrp, x, y, z)
    mbi = _gamr_locate_fast(leaf, amrp, x, y, z)
    mbi == Int32(0) && return zero(SVector{8,Float64})
    return @inline _gamr_lerp(blk, xlft, dxb, amrp.ni, amrp.nj, amrp.nk, Int(mbi), x, y, z)
end

# ---------------------------------------------------------------- probe kernels (validation)
# Device sample at arbitrary CKS points (bit-for-bit check vs CPU sample_primitives).
@kernel function _amr_probe_kernel!(out, @Const(pts), @Const(blk), @Const(xlft),
                                    @Const(dxb), @Const(leaf), amrp, npts)
    i = @index(Global)
    @inbounds if i <= npts
        p = pts[i]
        out[i] = _gamr_sample(blk, xlft, dxb, leaf, amrp, p[1], p[2], p[3])
    end
end

# Device locate cross-check: fast (index grid) vs scan (oracle) at arbitrary points.
@kernel function _amr_locate_probe_kernel!(outf, outs, @Const(pts), @Const(lo),
                                           @Const(hi), @Const(leaf), amrp, npts)
    i = @index(Global)
    @inbounds if i <= npts
        p = pts[i]
        outf[i] = _gamr_locate_fast(leaf, amrp, p[1], p[2], p[3])
        outs[i] = Int32(_gamr_locate_scan(lo, hi, amrp.nmb, p[1], p[2], p[3]))
    end
end

"""
    amr_sample_probe(dgrid::DeviceAMRGrid, pts::Vector{SVector{3,Float64}}; device=0)
        -> Vector{SVector{8,Float64}}

Run the device AMR sampler at `pts` (host CKS points) on `device`, for the bit-for-bit
check against CPU `sample_primitives`.
"""
function amr_sample_probe(dgrid::DeviceAMRGrid, pts::Vector{SVector{3,Float64}}; device::Int = 0)
    CUDA.device!(device)
    backend = CUDABackend()
    n = length(pts)
    ptsd = CuArray(pts)
    outd = CuArray{SVector{8,Float64}}(undef, n)
    _amr_probe_kernel!(backend)(outd, ptsd, dgrid.dev_blk[device], dgrid.dev_xlft[device],
                                dgrid.dev_dxb[device], dgrid.dev_leaf[device], dgrid.amrp, n;
                                ndrange = n)
    KernelAbstractions.synchronize(backend)
    return Array(outd)
end

"""
    amr_locate_probe(dgrid, pts; device=0) -> (fast::Vector{Int32}, scan::Vector{Int32})

Device locate cross-check: fast index-grid vs linear-scan oracle at `pts`.
"""
function amr_locate_probe(dgrid::DeviceAMRGrid, pts::Vector{SVector{3,Float64}}; device::Int = 0)
    CUDA.device!(device)
    backend = CUDABackend()
    n = length(pts)
    ptsd = CuArray(pts)
    fd = CuArray{Int32}(undef, n); sd = CuArray{Int32}(undef, n)
    _amr_locate_probe_kernel!(backend)(fd, sd, ptsd, dgrid.dev_lo[device], dgrid.dev_hi[device],
                                       dgrid.dev_leaf[device], dgrid.amrp, n; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return Array(fd), Array(sd)
end

# ---------------------------------------------------------------- fused AMR kernel (Stokes I)
# _gimg_march (gpu_image.jl) with the sampler swapped for the native-AMR _gamr_sample.
# Two-type: the geodesic lane runs at eltype(eltype(y0s)) — the host passes the march
# scalars at that type — while the sampler position is converted to Float64 (identity
# for fp64 rays), so the locate/trilinear sample + RT stay Float64 in every mode.
@kernel function _famr_kernel!(img, @Const(y0s), @Const(blk), @Const(xlft), @Const(dxb),
                               @Const(leaf), amrp, gp, a, r_h, C, escape, R_em, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gimg_march(y0s[i],
                             y -> _gamr_sample(blk, xlft, dxb, leaf, amrp, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             eltype(img), gp, a, r_h, C, escape, R_em, nmax)
    end
end

# ---------------------------------------------------------------- fused AMR kernel (I,Q,U,V)
# _gpol_march (gpu_pol.jl) with the same single sampler swap (same two-type contract).
@kernel function _gpol_amr_kernel!(out, @Const(y0s), @Const(blk), @Const(xlft), @Const(dxb),
                                   @Const(leaf), amrp, gp, a, r_h, C, escape, R_em, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gpol_march(y0s[i],
                             y -> _gamr_sample(blk, xlft, dxb, leaf, amrp, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             gp, a, r_h, C, escape, R_em, nmax)
    end
end

# ---------------------------------------------------------------- host drivers
# Circumscribed radius of the covered AMR domain: beyond it every point is outside the
# leaf index grid, so `_gamr_locate_fast` returns 0 (vacuum). An outbound ray past this
# radius only samples vacuum, so gating on it leaves results bit-identical.
function _gamr_domain_radius(p)
    x1hi = p.x1min + p.N1 / p.inv_fdx1
    x2hi = p.x2min + p.N2 / p.inv_fdx2
    x3hi = p.x3min + p.N3 / p.inv_fdx3
    return sqrt(max(p.x1min^2, x1hi^2) + max(p.x2min^2, x2hi^2) + max(p.x3min^2, x3hi^2))
end

# Launches of the native-AMR kernels (Stokes I / IQUV) for one chunk on the lane's device.
# The geodesic(/frame)-lane type is eltype(eltype(y0d)); the caller passes the march
# scalars at that type.
_famr_launch(dg, gp, a, r_h, C, escape, R_em, nmax, wg) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_famr_kernel!(lane.backend), wg, n, outd, y0d, dg.dev_blk[lane.dev],
                  dg.dev_xlft[lane.dev], dg.dev_dxb[lane.dev], dg.dev_leaf[lane.dev], dg.amrp,
                  gp, a, r_h, C, escape, R_em, Int(nmax), n)
_gpol_amr_launch(dg, gp, a, r_h, C, escape, R_em, nmax, wg) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_gpol_amr_kernel!(lane.backend), wg, n, outd, y0d, dg.dev_blk[lane.dev],
                  dg.dev_xlft[lane.dev], dg.dev_dxb[lane.dev], dg.dev_leaf[lane.dev], dg.amrp,
                  gp, a, r_h, C, escape, R_em, Int(nmax), n)

"""
    march_rays_gpu_amr(cfg, snap_or_amrgrid, rays::Vector{SVector{8,Float64}};
                       polarized=false, devices=default_pool(), nmax=cfg.max_steps,
                       C=80, order=nothing, wg=<pol _GPOL_WG / unpol nothing>)
        -> Matrix{Float64}(4, length(rays))

Fused single-pass GPU march of an arbitrary camera-ray list against the native-AMR
meshblock hierarchy, returning a dense `4 × length(rays)` Stokes matrix (row 1 = I;
rows 2-4 = Q,U,V, zero when `polarized=false`); column `j` corresponds to input ray `j`
(`order` sets the queue order only). The full-frame `render_gpu_amr[_polarized]` run
the same `:f64` march on camera rays made on the device. `snap_or_amrgrid`:
`AthenaKSnapshot` (uploaded once per device pool) or pre-uploaded `DeviceAMRGrid`.
`:f64`-only by design — the critical-curve rescue behind `:mixed`/`:f32_raw` is
pixel-grid-coupled and lives only at the full-frame wrapper level.
"""
function march_rays_gpu_amr(cfg, snap_or_amrgrid, rays::Vector{SVector{8,Float64}};
                            polarized::Bool = false, devices = default_pool(),
                            nmax::Int = cfg.max_steps, C::Real = 80,
                            order = nothing, wg = :default, precision::Symbol = :f64)
    precision === :f64 ||
        error("march_rays_gpu_amr is :f64-only by design: the critical-curve rescue " *
              "(`_rescue_pixels` in gpu_image.jl) is pixel-GRID-coupled and lives only " *
              "at the full-frame wrapper level; got precision=:$precision. Use " *
              "render_gpu_amr[_polarized] for :mixed / :f32_raw.")
    wg === :default && (wg = polarized ? _GPOL_WG : nothing)
    pool = device_pool(devices)
    nrays = length(rays)
    out = _amr_march(cfg, pool, resident_amr(pool, snap_or_amrgrid), HostRays(rays),
                     queue_items(order, nrays), nrays, polarized, nmax, C, wg)
    M = zeros(Float64, 4, nrays)
    @inbounds for j in 1:nrays
        o = out[j]
        for k in eachindex(o)
            M[k, j] = o[k]
        end
    end
    return M
end

# The :f64 native-AMR march of `nrays` items from the ray source `src`, queued in
# `items` order: Stokes I (Float64) or IQUV (SVector{4,Float64}) per item.
function _amr_march(cfg, pool, dg, src, items, nrays, polarized, nmax, C, wg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = _gamr_domain_radius(dg.amrp)       # vacuum beyond the covered domain -> exact gate
    if polarized
        out = Vector{SVector{4,Float64}}(undef, nrays)
        launch = _gpol_amr_launch(dg, _gpol_params(cfg), a, r_h, Cf, escape, R_em, nmax, wg)
    else
        out = Vector{Float64}(undef, nrays)
        launch = _famr_launch(dg, _gimg_params(cfg), a, r_h, Cf, escape, R_em, nmax, wg)
    end
    march!((out,), pool, [MarchPart(items, src, launch)])
    return out
end

# :mixed / :f32_raw full-frame native-AMR render (the rescue is pixel-GRID-coupled, so it
# lives only at this level): the fp32 lane over every pixel except (:mixed) those within
# `rescue_px` of the critical curve, which are queued first in fp64. `launch(a, r_h,
# escape, R_em)` builds the kernel launch at the scalars' type; writes `out`, returns the
# rescued fraction.
function _amr_mixed!(out, pool, cfg, cam, items, launch, R_em, precision, rescue_px)
    st = KerrSchildCartesian(cfg.bh_a)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    f32 = launch(Float32(a), Float32(r_h), Float32(escape), Float32(R_em))
    if precision === :f32_raw
        march!((out,), pool, [MarchPart(items, CameraRays(st, cam, Float32), f32)])
        return 0.0
    end
    pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                         rescue_px = rescue_px)
    march!((out,), pool, _mixed_parts(items, pix, CameraRays(st, cam),
                                      CameraRays(st, cam, Float32),
                                      launch(a, r_h, escape, R_em), f32))
    return length(pix) / cam.npix^2
end

"""
    render_gpu_amr(cfg, snap_or_amrgrid; devices=default_pool(), nmax=cfg.max_steps,
                   C=80, precision=:f64, rescue_px=2) -> Matrix{Float64}

Fused single-pass GPU render of the unpolarized (Stokes I) npix×npix image, sampling
the AMR meshblock hierarchy directly on device (nothing is resampled — no `box_half`/
`n`). `snap_or_amrgrid`: `AthenaKSnapshot` (uploaded once per device pool) or
pre-uploaded `DeviceAMRGrid`. Layout as `render(...;polarized=false).I`; `order` sets
the queue order (default `cost_order`; bit-exact per pixel); `wg` overrides the block
size. `precision`:
`:f64` (default, bitwise unchanged) · `:mixed` (fp32 march + fp64 RT, the pixels within
`rescue_px` px of the critical curve marched in fp64 and queued first;
`rescued_frac::Ref{Float64}` receives their fraction) · `:f32_raw` (no rescue —
DIAGNOSTIC only). See `render_gpu_fused`.
"""
function render_gpu_amr(cfg, snap_or_amrgrid; devices = default_pool(),
                        nmax::Int = cfg.max_steps, C::Real = 80,
                        order = nothing, wg = nothing,
                        precision::Symbol = :f64, rescue_px::Int = 2,
                        rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_amr: precision must be :f64, :mixed or :f32_raw, got :$precision")
    img = _amr_frame(cfg, snap_or_amrgrid, false, devices, nmax, C, order, wg, precision,
                     rescue_px, rescued_frac)
    return reshape(img, cfg.camera.resolution, cfg.camera.resolution)
end

"""
    render_gpu_amr_polarized(cfg, snap_or_amrgrid; devices=default_pool(),
                             nmax=cfg.max_steps, C=80, precision=:f64, rescue_px=2)
        -> NamedTuple(I,Q,U,V)

Full-Stokes fused GPU render sampling the AMR fluid directly on device (closes the
uniform-resample EVPA caveat). Mirrors `render_gpu_polarized` but with the AMR sampler.
`order` sets the queue order (default `cost_order`; bit-exact per pixel); `wg`
overrides the register-heavy kernel's block size (defaults to `_GPOL_WG`). `precision`:
`:f64` (default, bitwise unchanged) · `:mixed` (fp32 geodesic + frame transport, fp64
Stokes/coefficients from converted state; the pixels within `rescue_px` px of the
critical curve marched in fp64 and queued first; `rescued_frac::Ref{Float64}` receives
their fraction) · `:f32_raw` (no rescue — DIAGNOSTIC only). See `render_gpu_polarized`
for the full contract.
"""
function render_gpu_amr_polarized(cfg, snap_or_amrgrid; devices = default_pool(),
                                  nmax::Int = cfg.max_steps, C::Real = 80,
                                  order = nothing, wg = _GPOL_WG,
                                  precision::Symbol = :f64, rescue_px::Int = 2,
                                  rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_amr_polarized: precision must be :f64, :mixed or :f32_raw, got :$precision")
    out = _amr_frame(cfg, snap_or_amrgrid, true, devices, nmax, C, order, wg, precision,
                     rescue_px, rescued_frac)
    return _stokes_planes(out, cfg.camera.resolution)
end

# Full-frame native-AMR render on camera rays made on the device: per-pixel Stokes I
# (Float64) or IQUV (SVector{4,Float64}) in `camera_rays` order.
function _amr_frame(cfg, snap_or_amrgrid, polarized, devices, nmax, C, order, wg, precision,
                    rescue_px, rescued_frac)
    pool = device_pool(devices)
    dg = resident_amr(pool, snap_or_amrgrid)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    nrays = cam.npix^2
    items = _frame_items(order, cfg, cam.npix)
    if precision === :f64
        out = _amr_march(cfg, pool, dg, CameraRays(KerrSchildCartesian(cfg.bh_a), cam),
                         items, nrays, polarized, nmax, C, wg)
        frac = 0.0
    else
        Cf = Float64(C)
        out = Vector{polarized ? SVector{4,Float64} : Float64}(undef, nrays)
        gp = polarized ? _gpol_params(cfg) : _gimg_params(cfg)
        mk = polarized ? _gpol_amr_launch : _famr_launch
        frac = _amr_mixed!(out, pool, cfg, cam, items,
                           (a, r_h, escape, R_em) -> mk(dg, gp, a, r_h, typeof(a)(Cf), escape,
                                                        R_em, nmax, wg),
                           _gamr_domain_radius(dg.amrp), precision, rescue_px)
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    return out
end

# ================================================================ AMR DRIVER (streamed)
"""
    AMRStokes(cfg, snap_or_amrgrid; polarized=false, devices=default_pool(),
              nmax=cfg.max_steps, C=80, wg=<pol _GPOL_WG / unpol nothing>)

The native-AMR fused march as the `stokes` of `render_amr`, streamed: every refinement
block is a work item of the device pool; the device makes a block's rays from its origin
and pixel size (`BlockGrid`), and the worker that copies a chunk back paints and tests
its blocks and queues their children at once, so refined blocks march while the rest of
their parent level is still in flight (no level barrier). Images, level map and ray
count equal the level-by-level driver's. Also callable as `stokes(coords)` (host rays
through `march_rays_gpu_amr`).
"""
struct AMRStokes{C,G}
    cfg::C
    dg::G
    pool::DevicePool
    polarized::Bool
    nmax::Int
    C::Float64
    wg::Any
end
AMRStokes(cfg, snap_or_amrgrid; polarized::Bool = false, devices = default_pool(),
          nmax::Int = cfg.max_steps, C::Real = 80, wg = polarized ? _GPOL_WG : nothing) =
    (pool = device_pool(devices);
     AMRStokes(cfg, resident_amr(pool, snap_or_amrgrid), pool, polarized, nmax, Float64(C), wg))

_amr_camera(cfg) = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                                       cfg.camera.fov, cfg.camera.resolution)

(m::AMRStokes)(coords::Vector{Tuple{Float64,Float64}}) =
    march_rays_gpu_amr(m.cfg, m.dg, camera_rays_at(KerrSchildCartesian(m.cfg.bh_a),
                                                   _amr_camera(m.cfg), coords);
                       polarized = m.polarized, devices = m.pool, nmax = m.nmax, C = m.C,
                       wg = m.wg)

function stream_blocks!(m::AMRStokes, work::Vector{AMRBlock}, block::Int, visit)
    cfg = m.cfg
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    R_em = _gamr_domain_radius(m.dg.amrp)
    launch = m.polarized ?
        _gpol_amr_launch(m.dg, _gpol_params(cfg), a, r_h, m.C, escape, R_em, m.nmax, m.wg) :
        _famr_launch(m.dg, _gimg_params(cfg), a, r_h, m.C, escape, R_em, m.nmax, m.wg)
    st = KerrSchildCartesian(a)
    fr = CameraFrame(_amr_camera(cfg))
    bb = block * block
    per = max(1, m.pool.chunk ÷ bb)            # blocks per chunk
    stream_chunks(m.pool, _block_chunks(work, per)) do lane, blks, job
        rays = CameraRays(st, fr, BlockGrid([SVector(b[4], b[5], b[6] / block) for b in blks],
                                            block))
        S = _amr_stokes_rows(lane, rays, launch, length(blks) * bb, m.polarized)
        kids = AMRBlock[]
        for (w, b) in enumerate(blks)
            append!(kids, visit(b, S, (w - 1) * bb))
        end
        foreach(c -> enqueue!(job, c), _block_chunks(kids, per))
    end
    return nothing
end

_block_chunks(blks, per) = [blks[r] for r in Iterators.partition(eachindex(blks), per)]

# One chunk's Stokes, 4 rows × n block pixels (Q, U, V rows zero when unpolarized).
function _amr_stokes_rows(lane, rays, launch, n, polarized)
    if polarized
        out = Vector{SVector{4,Float64}}(undef, n)
        march_chunk!(lane, (out,), rays, launch, 1:n)
        return reinterpret(reshape, Float64, out)
    end
    out = Vector{Float64}(undef, n)
    march_chunk!(lane, (out,), rays, launch, 1:n)
    S = zeros(4, n)
    S[1, :] .= out
    return S
end

# ================================================================ SLOW LIGHT (opt-in, direct AMR)
# Native-AMR slow light: two DeviceAMRGrid frames blended on the physical dump-time
# axis (emission time t_emit = t_obs + X⁰; see the gpu_image.jl SLOW LIGHT header).
# The fast kernels are untouched; the slow kernels below swap only the sample line for
# the time-blended _gamr_sample_t. The two frames must SHARE the AMR mesh (same
# amrp/xlft/dxb/leaf — true for a static-mesh run); the kernel uses frame A's geometry
# for both blk arrays. w = clamp((t0 + X⁰) * inv_dt, 0, 1) with t0 = t_obs - t_a;
# inv_dt=0 (fast-light limit) makes _gamr_sample_t return frame A bit-for-bit.

"""
    DeviceAMRWindow

Bracketing pair of pre-uploaded `DeviceAMRGrid` frames (`a` at `t_a`, `b` at `t_b`)
plus `inv_dt = 1/(t_b - t_a)` for the in-kernel time blend (0.0 == fast-light
limit). The two frames must share the AMR mesh (enforced: same `amrp.nmb`).
"""
struct DeviceAMRWindow
    a::DeviceAMRGrid
    b::DeviceAMRGrid
    t_a::Float64
    inv_dt::Float64
end

"""
    upload_amr_window(series::SnapshotSeries, ia, devices; upload_bounds=false)
        -> DeviceAMRWindow

Upload the bracketing dump pair `(ia, ia+1)` of `series` as two `DeviceAMRGrid`
frames. Object-shared/identical frames reuse one upload; a single-frame series yields
inv_dt = 0 (fast light). The frames must share the AMR mesh.
"""
function upload_amr_window(series::SnapshotSeries, ia::Integer, devices; upload_bounds::Bool = false)
    N = length(series)
    ia = Int(ia); ib = min(ia + 1, N)
    A = get_frame(series, ia); B = get_frame(series, ib)
    t_a = series.times[ia]; t_b = ib > ia ? series.times[ib] : Inf
    dgA = upload_amr(A, devices; upload_bounds = upload_bounds)
    dgB = (A === B) ? dgA : upload_amr(B, devices; upload_bounds = upload_bounds)
    # The slow-light kernels locate BOTH frames with frame A's geometry (amrp/xlft/dxb/leaf),
    # so nmb-equality is not enough — a differing locate grid silently misplaces frame B.
    # Require the full leaf/index geometry (amrp) to match (identical for a static mesh).
    dgA.amrp == dgB.amrp ||
        error("slow-light AMR frames must share the mesh geometry (amrp mismatch: $(dgA.amrp) vs $(dgB.amrp))")
    return DeviceAMRWindow(dgA, dgB, Float64(t_a), _win_inv_dt(Float64(t_a), Float64(t_b)))
end

"""
    advance_amr_window(win::DeviceAMRWindow, series, ia, devices; upload_bounds=false)
        -> DeviceAMRWindow

Native-AMR twin of `advance_series_window`: slide the window to the bracket `(ia, ia+1)`
reusing `win.b` (the already-uploaded upper frame) as the new lower frame, so only the
new upper frame `ia+1` is uploaded; the reused frame is the same device buffer, so the
rendered frame is bit-identical to a fresh upload. Advance monotonically (`win`'s upper
frame IS dump `ia`); frames must share the AMR mesh (amrp checked). At the series end
the window degenerates to inv_dt = 0 (fast light).
"""
function advance_amr_window(win::DeviceAMRWindow, series::SnapshotSeries, ia::Integer,
                            devices; upload_bounds::Bool = false)
    N = length(series); ia = Int(ia); ib = min(ia + 1, N)
    newA = win.b                              # promote the uploaded upper frame — no re-upload
    if ib > ia
        newB = upload_amr(get_frame(series, ib), devices; upload_bounds = upload_bounds)
        newB.amrp == newA.amrp ||
            error("advance_amr_window: new frame $(ib) mesh geometry (amrp) must match the " *
                  "promoted frame ($(newB.amrp) vs $(newA.amrp))")
    else
        newB = newA
    end
    t_a = series.times[ia]; t_b = ib > ia ? series.times[ib] : Inf
    return DeviceAMRWindow(newA, newB, Float64(t_a), _win_inv_dt(Float64(t_a), Float64(t_b)))
end

# Time-blended native-AMR sample (shared geometry, two blk arrays). The clamped
# window ends short-circuit to a single frame's locate + corner reads: w == 0
# returns frame A EXACTLY (fast-light limit) and w == 1 frame B — both weights are
# attained exactly by the kernels' clamp, so the branches are bit-identical to the
# blend while halving the global-memory traffic on clamped segments.
@inline function _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, w, x, y, z)
    w == 1.0 && return _gamr_sample(blkB, xlft, dxb, leaf, amrp, x, y, z)
    sA = _gamr_sample(blkA, xlft, dxb, leaf, amrp, x, y, z)
    w == 0.0 && return sA
    sB = _gamr_sample(blkB, xlft, dxb, leaf, amrp, x, y, z)
    return (1.0 - w) .* sA .+ w .* sB
end

# SLOW-LIGHT Stokes-I AMR kernel: _gimg_march with the time-blended _gamr_sample_t.
@kernel function _famr_kernel_sl!(img, @Const(y0s), @Const(blkA), @Const(blkB), @Const(xlft),
                                  @Const(dxb), @Const(leaf), amrp, t0, inv_dt,
                                  gp, a, r_h, C, escape, R_em, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gimg_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             eltype(img), gp, a, r_h, C, escape, R_em, nmax)
    end
end

# SLOW-LIGHT full-Stokes AMR kernel: _gpol_march with the time-blended _gamr_sample_t.
@kernel function _gpol_amr_kernel_sl!(out, @Const(y0s), @Const(blkA), @Const(blkB), @Const(xlft),
                                      @Const(dxb), @Const(leaf), amrp, t0, inv_dt,
                                      gp, a, r_h, C, escape, R_em, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gpol_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             gp, a, r_h, C, escape, R_em, nmax)
    end
end

# Launches of the slow-light native-AMR kernels for one chunk (frame A's mesh geometry).
function _amr_launch_sl(kern, win::DeviceAMRWindow, t0, gp, a, r_h, C, escape, R_em, nmax, wg)
    ga = win.a; gb = win.b
    return (lane, (outd,), y0d, n) ->
        _kern_launch!(kern(lane.backend), wg, n, outd, y0d, ga.dev_blk[lane.dev],
                      gb.dev_blk[lane.dev], ga.dev_xlft[lane.dev], ga.dev_dxb[lane.dev],
                      ga.dev_leaf[lane.dev], ga.amrp, t0, win.inv_dt, gp, a, r_h, C, escape,
                      R_em, Int(nmax), n)
end

"""
    render_gpu_amr_slow(cfg, win::DeviceAMRWindow; devices=default_pool(),
                        nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                        time_offset=0.0) -> Matrix{Float64}

Fused single-pass SLOW-LIGHT Stokes-I render sampling the AMR hierarchy DIRECTLY on
device, blending the two `DeviceAMRWindow` frames at the EMISSION time
t_emit = t_obs + X⁰ (physical dump-time axis; `t_obs` defaults to the window's
upper end, `time_offset` advances it — movie frame time). `win.inv_dt == 0`
reproduces `render_gpu_amr(cfg, win.a)` bit-for-bit. Accepts `order`/`wg` as `render_gpu_amr`.
"""
function render_gpu_amr_slow(cfg, win::DeviceAMRWindow; devices = default_pool(),
                             nmax::Int = cfg.max_steps, C::Real = 80,
                             t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                             order = nothing, wg = nothing)
    img = _amr_slow(cfg, win, devices, _famr_kernel_sl!, _gimg_params(cfg), Float64,
                    nmax, C, t_obs, time_offset, order, wg)
    return reshape(img, cfg.camera.resolution, cfg.camera.resolution)
end

"""
    render_gpu_amr_polarized_slow(cfg, win::DeviceAMRWindow; devices=default_pool(),
                                  nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                                  time_offset=0.0) -> NamedTuple(I,Q,U,V)

Full-Stokes SLOW-LIGHT render sampling the AMR fluid directly on device (frames
blended at the emission time t_obs + X⁰, as `render_gpu_amr_slow`).
`win.inv_dt == 0` reproduces `render_gpu_amr_polarized(cfg, win.a)`.
Accepts `order`/`wg` as `render_gpu_amr_polarized`.
"""
function render_gpu_amr_polarized_slow(cfg, win::DeviceAMRWindow; devices = default_pool(),
                                       nmax::Int = cfg.max_steps, C::Real = 80,
                                       t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                                       order = nothing, wg = _GPOL_WG)
    out = _amr_slow(cfg, win, devices, _gpol_amr_kernel_sl!, _gpol_params(cfg),
                    SVector{4,Float64}, nmax, C, t_obs, time_offset, order, wg)
    return _stokes_planes(out, cfg.camera.resolution)
end

# Shared body of the two slow-light native-AMR drivers: per-pixel outputs of eltype `T`.
function _amr_slow(cfg, win, devices, kern, gp, ::Type{T}, nmax, C, t_obs, time_offset,
                   order, wg) where {T}
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    R_em = _gamr_domain_radius(win.a.amrp)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    out = Vector{T}(undef, cam.npix^2)
    march!((out,), pool, [MarchPart(_frame_items(order, cfg, cam.npix), CameraRays(st, cam),
        _amr_launch_sl(kern, win, t0, gp, a, r_h, Float64(C), escape, R_em, nmax, wg))])
    return out
end
