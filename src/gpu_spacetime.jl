# gpu_spacetime.jl — generic-spacetime fused GPU imaging: unpolarized Stokes I, full
# IQUV, and a no-fluid shadow/capture tracer. The device metric layer is the CPU
# geodesics layer compiled in-kernel: isbits spacetimes (`SuperposedKerrSchild`) pass as
# kernel parameters and the kernels call the module originals `metric` / `inv` /
# `geodesic_rhs` / `rk4_step_fixed`; `ADMGridSpacetime` is mirrored by `DevADMParams`
# (same field names, device data array) through the same `_adm_gcov` body. The fluid is
# sampled via the O(nmb) scan locate (`_gamr_locate_scan` + `_gamr_lerp`) for meshes
# beyond the dense leaf-table cap. Float64 only. CUDA optional: NOT included from
# BlackLightPlus.jl; include AFTER gpu_image/gpu_pol/gpu_amr (auto-included if absent).

using StaticArrays
using ForwardDiff: Dual, partials, value
using KernelAbstractions
using CUDA
using Adapt
using LinearAlgebra: dot
using BlackLightPlus: SuperposedKerrSchild, PlaneParallelCamera
using BlackLightPlus.Geodesics: metric, inverse_metric, metric_and_inverse, geodesic_rhs,
      geodesic_transport_rhs,
      rk4_step_fixed, ks_radius_hole, horizon_radius_hole, HORIZON_EPS_SAMPLED,
      ADMGridSpacetime, superposed_control, march_captured, march_escaped,
      normal_frame_energy, StationarySpacetime, _adm_gcov, _adm_gcon, _adm_pair,
      _adm_lower_index, _adm_raise_index, _adm_raise_and_norm, spacetime_control,
      adm_control
using BlackLightPlus.Radiation: SpacetimeMarch, reverse_unpolarized_march,
      reverse_polarized_march
using BlackLightPlus.Simulation: AthenaKSnapshot

isdefined(@__MODULE__, :_fused_kernel!) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_image.jl"))
isdefined(@__MODULE__, :_gpol_kernel!)  || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_pol.jl"))
isdefined(@__MODULE__, :_gamr_lerp)   || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_amr.jl"))

# ---------------------------------------------------------------- device ADM params
"""
    DevADMParams(st::ADMGridSpacetime, data) <: StationarySpacetime

GPU twin of `ADMGridSpacetime`: the same grid-geometry fields (same names, so the
duck-typed `sample_adm`/`on_grid`/`_adm_gcov` bodies run unchanged) with `data` swapped
for a device array. Subtyping `StationarySpacetime` makes the CPU engine methods apply
verbatim, so the device ADM derivative semantics are exactly the CPU's: 3-lane
hand-seeded duals through `inv(_adm_gcov(·))`, C¹ Catmull-Rom weights carrying the Dual.
"""
struct DevADMParams{A} <: StationarySpacetime{Float64}
    x0::Float64; y0::Float64; z0::Float64
    dx::Float64; dy::Float64; dz::Float64
    nx::Int; ny::Int; nz::Int
    data::A
    alpha_min::Float64
    offgrid_eta::Bool
end
DevADMParams(st::ADMGridSpacetime, data; alpha_min::Real = st.alpha_min) =
    DevADMParams(st.x0, st.y0, st.z0, st.dx, st.dy, st.dz, st.nx, st.ny, st.nz,
                 data, Float64(alpha_min), st.offgrid_eta)
Adapt.adapt_structure(to, p::DevADMParams) =
    DevADMParams(p.x0, p.y0, p.z0, p.dx, p.dy, p.dz, p.nx, p.ny, p.nz,
                 Adapt.adapt(to, p.data), p.alpha_min, p.offgrid_eta)

# The SAME ADM assembly bodies as the CPU spacetime (single source, numerical_spacetime.jl).
@inline BlackLightPlus.Geodesics.metric(sp::DevADMParams, X::SVector{4}) = _adm_gcov(sp, X)
@inline BlackLightPlus.Geodesics.inverse_metric(sp::DevADMParams, X::SVector{4}) =
    _adm_gcon(sp, X)
@inline BlackLightPlus.Geodesics.metric_and_inverse(sp::DevADMParams, X::SVector{4}) =
    _adm_pair(sp, X)
@inline BlackLightPlus.Geodesics.lower_index(sp::DevADMParams, X::SVector{4}, v::SVector{4}) =
    _adm_lower_index(sp, X, v)
@inline BlackLightPlus.Geodesics.raise_index(sp::DevADMParams, X::SVector{4}, p::SVector{4}) =
    _adm_raise_index(sp, X, p)
@inline BlackLightPlus.Geodesics.raise_and_norm(sp::DevADMParams, X::SVector{4},
                                                p::SVector{4}) = _adm_raise_and_norm(sp, X, p)

"""
    upload_adm(st::ADMGridSpacetime, devices; alpha_min=st.alpha_min)
        -> Dict{device,DevADMParams}

One `DevADMParams` per device of `devices` (a list or a `DevicePool`; the (10,nx,ny,nz)
field array uploaded concurrently, once per device). `alpha_min` overrides the capture
lapse (e.g. the z4c shadow figure uses 0.65 on a grid built with 0.15).
"""
upload_adm(st::ADMGridSpacetime, devices; alpha_min::Real = st.alpha_min) =
    Dict{Any,Any}(dev => DevADMParams(st, d; alpha_min = alpha_min)
                  for (dev, d) in upload_per_device(st.data, devices))

# ---------------------------------------------------------------- march control
# The kernels run the reverse marches under a `SpacetimeMarch(sp, …)` control:
# `spacetime_control` (Geodesics) steps, captures and escapes per spacetime — the CPU
# `superposed_control` for the binary, `adm_control` (lapse horizon proxy) for the ADM
# grid, whose device parameter struct runs the same duck-typed body.
@inline BlackLightPlus.Geodesics.spacetime_control(sp::DevADMParams, y::SVector{8}, C) =
    adm_control(sp, y, C)

# ---------------------------------------------------------------- device: scan sample
# Scan-locate AMR sample: the exact `_find_meshblock_scan` port (`_gamr_locate_scan`,
# later block wins, 0 outside) + the shared in-block `_gamr_lerp` — the CPU
# `sample_primitives` route on meshes whose finest-level grid exceeds the leaf-table
# cap. O(nmb) per sample.
@inline function _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj, nk, nmb, x, y, z)
    mb = _gamr_locate_scan(lo, hi, nmb, x, y, z)
    mb == 0 && return zero(SVector{8,Float64})
    return @inline _gamr_lerp(blk, xlft, dxb, ni, nj, nk, mb, x, y, z)
end

# ---------------------------------------------------------------- device scan grid
"""
    DeviceAMRScanGrid

Per-device uploaded AMR fluid + per-block face bounds for the SCAN-locate sampler
(no leaf table — for meshes beyond the dense-grid cap where `upload_amr` errors).
Built by `upload_amr_scan`; consumed by the generic-spacetime kernels.
"""
struct DeviceAMRScanGrid{DB<:AbstractDict,DX<:AbstractDict}
    dev_blk::DB
    dev_xlft::DX
    dev_dxb::DX
    dev_lo::DX
    dev_hi::DX
    ni::Int; nj::Int; nk::Int; nmb::Int
end

"""
    upload_amr_scan(snap::AthenaKSnapshot, devices) -> DeviceAMRScanGrid

Permute the reader's primitives to the coalesced `(8,ni,nj,nk,nmb)` layout and upload
fluid + per-block geometry (face bounds for the scan locate) to each device of `devices`
(a list or a `DevicePool`). Works for ANY mesh (no leaf-table requirement); reuse across
renders.
"""
function upload_amr_scan(snap::AthenaKSnapshot, devices)
    blk_host  = permutedims(snap.data, (5, 1, 2, 3, 4))    # (8, ni, nj, nk, nmb)
    xlft_host = reduce(hcat, snap.x_left)                   # (3, nmb)
    dxb_host  = reduce(hcat, snap.dx)
    lo_host   = reduce(hcat, snap.ext_lo)
    hi_host   = reduce(hcat, snap.ext_hi)
    dblk = upload_per_device(blk_host, devices)
    dxl  = upload_per_device(xlft_host, devices)
    ddx  = upload_per_device(dxb_host, devices)
    dlo  = upload_per_device(lo_host, devices)
    dhi  = upload_per_device(hi_host, devices)
    return DeviceAMRScanGrid(dblk, dxl, ddx, dlo, dhi, snap.ni, snap.nj, snap.nk, snap.nmb)
end

# ---------------------------------------------------------------- kernels
# Generic-spacetime fused unpol kernel over the scan-locate AMR fluid. One ray/thread;
# `sp` is the isbits spacetime parameter (the CPU struct itself, or DevADMParams).
@kernel function _gsp_amr_kernel!(img, @Const(y0s), @Const(blk), @Const(xlft), @Const(dxb),
                                  @Const(lo), @Const(hi), sp, gp, C, escape, R_em,
                                  nmax, ni, nj, nk, nmb, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = reverse_unpolarized_march(y0s[i],
                            y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj, nk,
                                                   nmb, y[2], y[3], y[4]),
                            eltype(img), gp, SpacetimeMarch(sp, C, escape, R_em), nmax)
    end
end

# Generic-spacetime fused POLARIZED kernel over the scan-locate AMR fluid (the
# _gsp_amr_kernel! twin with the polarized march; out[i] is the camera-frame Stokes 4-vector).
@kernel function _gsp_pol_amr_kernel!(out, @Const(y0s), @Const(blk), @Const(xlft),
                                      @Const(dxb), @Const(lo), @Const(hi), sp, gp, C,
                                      escape, R_em, nmax, ni, nj, nk, nmb, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = reverse_polarized_march(y0s[i],
                                y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj,
                                                       nk, nmb, y[2], y[3], y[4]),
                                gp, SpacetimeMarch(sp, C, escape, R_em), nmax)
    end
end

# No-fluid shadow/capture tracer (z4c lensing masks; puncture-capture + null gates).
# Mirrors the CPU fixed-control loop: per step — capture test, PLAIN escape (the camera
# sits inside the escape sphere on these decks), fixed-control step, non-finite state
# ⇒ capture. Tracks max |g^{μν} p_μ p_ν| (the CPU `null_error`) along the ray.
# out_m[i] = (status: 1 capture / 2 escape / 0 nmax, steps, max|k·k|, r_final).
@kernel function _gsp_shadow_kernel!(out_y, out_m, @Const(y0s), sp, C, escape, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        y = y0s[i]
        status = 0.0
        maxnull = 0.0
        n = 0
        for _ in 1:nmax
            cap, h, re = spacetime_control(sp, y, C)
            x = SVector(y[1], y[2], y[3], y[4])
            p = SVector(y[5], y[6], y[7], y[8])
            gi = inverse_metric(sp, x)
            nl = abs(dot(p, gi * p))
            isfinite(nl) && (maxnull = max(maxnull, nl))
            if cap
                status = 1.0; break
            end
            if re > escape
                status = 2.0; break
            end
            y = rk4_step_fixed(sp, y, h)
            n += 1
            if !(isfinite(y[1]) && isfinite(y[2]) && isfinite(y[3]) && isfinite(y[4]) &&
                 isfinite(y[5]) && isfinite(y[6]) && isfinite(y[7]) && isfinite(y[8]))
                status = 1.0; break      # blew up near a puncture => capture (trace_bbh)
            end
        end
        out_y[i] = y
        out_m[i] = SVector(status, Float64(n), maxnull,
                           sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4]))
    end
end

# ---------------------------------------------------------------- host drivers
# The per-device spacetime parameter: isbits spacetimes are shared as-is; an
# ADMGridSpacetime is uploaded once per pool (or pass the `upload_adm` Dict directly).
_gsp_per_device(st, pool) = Dict{Any,Any}(dev => st for dev in pool.devices)
_gsp_per_device(st::ADMGridSpacetime, pool) =
    resident(() -> upload_adm(st, pool), pool, st.data, :adm)
_gsp_per_device(d::Dict, pool) = d

# The scan-locate fluid of a driver call: as given, or uploaded once per pool.
_resident_scan(pool, sg::DeviceAMRScanGrid) = sg
_resident_scan(pool, snap::AthenaKSnapshot) =
    resident(() -> upload_amr_scan(snap, pool), pool, snap.data, :scan)

# Launch of a generic-spacetime scan-locate kernel (unpolarized or polarized) for a chunk.
_gsp_launch(kern, sg, sps, gp, C, escape, R_em, nmax, wg) = (lane, (outd,), y0d, n) ->
    _kern_launch!(kern(lane.backend), wg, n, outd, y0d, sg.dev_blk[lane.dev],
                  sg.dev_xlft[lane.dev], sg.dev_dxb[lane.dev], sg.dev_lo[lane.dev],
                  sg.dev_hi[lane.dev], sps[lane.dev], gp, C, escape, R_em, Int(nmax),
                  sg.ni, sg.nj, sg.nk, sg.nmb, n)

"""
    march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays; polarized=false,
                             devices=default_pool(), nmax=cfg.max_steps, C=80,
                             R_em=Inf, wg=<pol _GPOL_WG / unpol nothing>)
        -> Vector{Float64} (unpol)  |  Matrix{Float64}(4, length(rays)) (polarized)

Fused single-pass GPU march of an arbitrary camera-ray list in an arbitrary stationary
spacetime `st` (`SuperposedKerrSchild`, `ADMGridSpacetime`, or a pre-uploaded
`upload_adm` Dict), sampling the AMR fluid on device through the scan locate. The
geodesic step is the CPU engine in-kernel (bit-compatible with `trace_rays_sampled`).
`polarized=false` returns Stokes I as a flat `Vector{Float64}`; `polarized=true` runs
the full-IQUV reverse-Mueller march and returns the `4 × length(rays)` Stokes matrix
(rows I,Q,U,V). `snap_or_scangrid`: `AthenaKSnapshot` (uploaded once per device pool)
or `DeviceAMRScanGrid` (reused). Float64 only.
"""
function march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays::Vector{SVector{8,Float64}};
                                  polarized::Bool = false,
                                  devices = default_pool(), nmax::Int = cfg.max_steps,
                                  C::Real = 80, R_em::Real = Inf, wg = :default)
    pool = device_pool(devices)
    out = _gsp_march(cfg, pool, _gsp_per_device(st, pool), snap_or_scangrid, HostRays(rays),
                     length(rays), polarized, nmax, C, R_em, wg)
    polarized || return out
    M = Matrix{Float64}(undef, 4, length(rays))
    @inbounds for j in eachindex(out)
        o = out[j]
        M[1, j] = o[1]; M[2, j] = o[2]; M[3, j] = o[3]; M[4, j] = o[4]
    end
    return M
end

# The generic-spacetime march of `nrays` items from the ray source `src` over the
# per-device spacetimes `sps`: Stokes I (Float64) or IQUV (SVector{4,Float64}) per item.
function _gsp_march(cfg, pool, sps, snap_or_scangrid, src, nrays, polarized, nmax, C, R_em, wg)
    wg === :default && (wg = polarized ? _GPOL_WG : nothing)
    sg = _resident_scan(pool, snap_or_scangrid)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    if polarized
        out = Vector{SVector{4,Float64}}(undef, nrays)
        launch = _gsp_launch(_gsp_pol_amr_kernel!, sg, sps, _gpol_params(cfg), Cf, escape,
                             Float64(R_em), nmax, wg)
    else
        out = Vector{Float64}(undef, nrays)
        launch = _gsp_launch(_gsp_amr_kernel!, sg, sps, _gimg_params(cfg), Cf, escape,
                             Float64(R_em), nmax, wg)
    end
    march!((out,), pool, [MarchPart(1:nrays, src, launch)])
    return out
end

"""
    render_gpu_spacetime(cfg, st, snap_or_scangrid; polarized=false,
                         devices=default_pool(), nmax=cfg.max_steps, C=80,
                         R_em=Inf, wg=<pol _GPOL_WG / unpol nothing>)
        -> Matrix{Float64} (unpol)  |  NamedTuple (I,Q,U,V) (polarized)

Full-frame generic-spacetime fused GPU render: the device twin of
`render(cfg, snap; spacetime=st, polarized=...)` — camera rays made on the device by
the same `camera_ray` as `camera_rays(st, cam)` (on each device's copy of `st`),
marched/sampled/accumulated entirely on device. `polarized=false` returns the Stokes-I
`Matrix{Float64}` in `render().I` layout; `polarized=true` returns the `(I, Q, U, V)`
NamedTuple of npix×npix matrices (full-IQUV reverse-Mueller march, parallel-transported
camera frame).
"""
function render_gpu_spacetime(cfg, st, snap_or_scangrid; polarized::Bool = false,
                              devices = default_pool(),
                              nmax::Int = cfg.max_steps, C::Real = 80, R_em::Real = Inf,
                              wg = :default)
    st isa Dict && error("pass the ADMGridSpacetime itself to render_gpu_spacetime")
    pool = device_pool(devices)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    sps = _gsp_per_device(st, pool)
    out = _gsp_march(cfg, pool, sps, snap_or_scangrid, CameraRays(sps, cam), cam.npix^2,
                     polarized, nmax, C, R_em, wg)
    return polarized ? _stokes_planes(out, cam.npix) : reshape(out, cam.npix, cam.npix)
end

"""
    trace_shadow_gpu(st, rays; C=60.0, escape=95.0, nmax=6000,
                     devices=default_pool(), wg=nothing)
        -> (status::Vector{Float64}, nsteps, maxnull, rfinal, yfinal)

No-fluid GPU shadow/capture tracer for an arbitrary stationary spacetime (device twin
of the z4c shadow tracer and the capture/null gates of the superposed tracer).
`status`: 1 = captured, 2 = escaped, 0 = step-limited. `maxnull` is the per-ray
max |g^{μν}p_μp_ν| (CPU `null_error`) over the march.
"""
function trace_shadow_gpu(st, rays::Vector{SVector{8,Float64}};
                          C::Real = 60.0, escape::Real = 95.0, nmax::Int = 6000,
                          devices = default_pool(), wg = nothing)
    pool = device_pool(devices)
    sps = _gsp_per_device(st, pool)
    n = length(rays)
    ys = Vector{SVector{8,Float64}}(undef, n)
    ms = Vector{SVector{4,Float64}}(undef, n)
    launch = (lane, (oyd, omd), y0d, cnt) ->
        _kern_launch!(_gsp_shadow_kernel!(lane.backend), wg, cnt, oyd, omd, y0d,
                      sps[lane.dev], Float64(C), Float64(escape), Int(nmax), cnt)
    march!((ys, ms), pool, [MarchPart(1:n, HostRays(rays), launch)])
    return ([m[1] for m in ms], [m[2] for m in ms], [m[3] for m in ms],
            [m[4] for m in ms], ys)
end
