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
      _adm_lower_index, _adm_raise_index, _adm_raise_and_norm
using BlackLightPlus.Simulation: AthenaKSnapshot

isdefined(@__MODULE__, :_gimg_coeffs) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_image.jl"))
isdefined(@__MODULE__, :_gpol_march)  || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_pol.jl"))
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
# Per-spacetime fixed-control march control: (captured, step h, escape-test radius);
# capture is the shared `march_captured` rule on the spacetime's horizon proxy.
# SuperposedKerrSchild calls the CPU tracer's own `superposed_control` (spacetimes.jl):
# step by the distance to the NEAREST horizon h = min_A(r_A − r_h_A)/C, capture (either
# hole) at horizon + HORIZON_EPS_SAMPLED or above CAPTURE_REDSHIFT in the normal frame,
# escape on the global coordinate radius. Non-finite r_A ⇒ capture.
@inline _gsp_ctl(st::SuperposedKerrSchild{N,Float64}, y::SVector{8,Float64},
                  C::Float64) where {N} = superposed_control(st, y, C)

# ADM grid mirrors the z4c shadow tracer: the horizon proxy is the local lapse
# α = 1/√(−g^{tt}) against alpha_min (off-grid the metric is Minkowski ⇒ α = 1, never
# captures), with the shared redshift cut on top; coordinate-radius step
# h = max(r,1)/C, escape on the coordinate radius. A non-finite state (puncture
# blow-up) is a capture.
@inline function _gsp_ctl(sp::DevADMParams, y::SVector{8,Float64}, C::Float64)
    x = SVector(y[1], y[2], y[3], y[4])
    r = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    gi = inverse_metric(sp, x)
    gtt = gi[1, 1]
    alpha = gtt < 0.0 ? 1.0 / sqrt(-gtt) : 1.0
    En = gtt < 0.0 ? normal_frame_energy(gi, y) : 0.0
    d = isfinite(r) ? alpha - sp.alpha_min : r
    return march_captured(d, En, 0.0), max(r, 1.0) / C, r
end

# ---------------------------------------------------------------- generic unpol march
# The generic-spacetime fused unpolarized march for one ray, camera->source: the
# `_gimg_march` reverse-accumulation scheme (gpu_image.jl header) with the geodesic
# step, step control and termination supplied by the spacetime (`_gsp_ctl` +
# `rk4_step_fixed`/`geodesic_rhs`, the CPU engine in-kernel). Float64-only (the fp32
# critical-curve rescue is a Kerr-specific construct).
@inline function _gsp_march(y0::SVector{8,Float64}, sample::F, gp, sp,
                            C::Float64, escape::Float64, R_em::Float64, nmax) where {F}
    y = y0
    Iinv = 0.0
    trans = 1.0
    r_prev = Inf
    for _ in 1:nmax
        cap, h, re = _gsp_ctl(sp, y, C)
        (cap || march_escaped(re, r_prev, escape, R_em)) && break   # shared rule, as CPU
        r_prev = re
        prim = @inline sample(y)
        if prim[1] > 0.0
            kcov = -SVector(y[5], y[6], y[7], y[8])     # photon k = -p
            gcov, gcon = metric_and_inverse(sp, SVector(y[1], y[2], y[3], y[4]))
            j_inv, a_inv = _prim_coeffs(prim, kcov, gcov, gcon, gp)
            ds = h * gp.L_unit / gp.nu_obs
            A, B = unpolarized_segment(j_inv, a_inv, ds)
            Iinv += trans * B
            trans *= A
        end
        y = rk4_step_fixed(sp, y, h)
    end
    return Iinv * gp.nu_obs^3
end

# ---------------------------------------------------------------- generic pol march
# The generic-spacetime fused POLARIZED march for one ray, camera->source: the Kerr
# reverse-Mueller scheme `_gpol_march` (gpu_pol.jl) with (i) march control/termination
# from the spacetime (`_gsp_ctl`), (ii) the RT-lane metric from `metric_and_inverse(sp,·)`,
# (iii) geodesic + two-leg frame transport stepped by the fused low-storage RK4 over
# `geodesic_transport_rhs` (same combine ops as `rk4_step_fixed`, so the trajectory is the
# unpol march's / CPU tracer's), and (iv) the e_(1)-renormalizing `fluid_tetrad` in both
# the fluid tetrad and the readout frame (|det g| ≠ 1 off single CKS).
@inline function _gsp_pol_march(y0::SVector{8,Float64}, sample::F, gp, sp,
                                C::Float64, escape::Float64, R_em::Float64,
                                nmax) where {F}
    y = y0
    x0 = SVector(y0[1], y0[2], y0[3], y0[4]); kcov0 = -SVector(y0[5], y0[6], y0[7], y0[8])
    gcov0, gcon0 = metric_and_inverse(sp, x0); kcon0 = gcon0 * kcov0
    e0c = _gpol_normal_observer(gcon0)
    e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = fluid_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0; renorm_e1 = true)
    eU  = SVector(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])   # camera e_(2) (up)
    eUu = e0c                    # camera 4-velocity, parallel-transported alongside eU
    S = _gpz4; G = _gpI4
    r_prev = Inf
    for _ in 1:nmax
        cap, h, re = _gsp_ctl(sp, y, C)
        (cap || march_escaped(re, r_prev, escape, R_em)) && break   # shared rule, as CPU
        r_prev = re
        prim = @inline sample(y)
        if prim[1] > 0.0
            x = SVector(y[1], y[2], y[3], y[4])
            gcov, gcon = metric_and_inverse(sp, x)
            kcov = -SVector(y[5], y[6], y[7], y[8]); kcon = gcon * kcov   # photon k = -p
            j, al, ro, tet_fl, valid = _gpol_coeffs(prim, kcov, kcon, gcov, gcon, gp, true)
            if valid
                E = fluid_tetrad(eUu, gcov * eUu, kcon, kcov, eU, gcov, gcon; renorm_e1 = true)
                ds = h * gp.L_unit / gp.nu_obs
                M, b = stokes_segment(j, al, ro, ds)
                c2, s2 = _gpol_R(tet_fl, E, gcov)
                S = stokes_admissible(S + G * _gpol_rot(c2, s2, b))   # the stokes_couple clamp
                G = G * _gpol_rot(c2, s2, M)
            end
        end
        # low-storage streaming RK4 of {y, eU, eUu} (the _gpol_march scheme; same
        # ops/associativity as rk4_step_fixed's 4-stage combine)
        dy, (de, du) = geodesic_transport_rhs(sp, y, (eU, eUu))
        ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(sp, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(sp, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, (de, du) = geodesic_transport_rhs(sp, ys, (es, us))
        ay = ay + dy; ae = ae + de; au = au + du
        y   = y   + (h/6)*ay
        eU  = eU  + (h/6)*ae
        eUu = eUu + (h/6)*au
    end
    return S * gp.nu_obs^3
end

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
        img[i] = _gsp_march(y0s[i],
                            y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj, nk,
                                                   nmb, y[2], y[3], y[4]),
                            gp, sp, C, escape, R_em, nmax)
    end
end

# Generic-spacetime fused POLARIZED kernel over the scan-locate AMR fluid (the
# _gsp_amr_kernel! twin with _gsp_pol_march; out[i] is the camera-frame Stokes 4-vector).
@kernel function _gsp_pol_amr_kernel!(out, @Const(y0s), @Const(blk), @Const(xlft),
                                      @Const(dxb), @Const(lo), @Const(hi), sp, gp, C,
                                      escape, R_em, nmax, ni, nj, nk, nmb, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gsp_pol_march(y0s[i],
                                y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj,
                                                       nk, nmb, y[2], y[3], y[4]),
                                gp, sp, C, escape, R_em, nmax)
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
            cap, h, re = _gsp_ctl(sp, y, C)
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
