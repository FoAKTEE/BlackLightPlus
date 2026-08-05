# gpu_image_ad.jl — GPU forward-mode-AD unpolarized imaging: the whole image and the
# total flux carry their exact derivative w.r.t. one scalar physical parameter, computed
# in-kernel. Key identity (Jipole, arXiv:2509.07065): the per-step radiative transfer is
# affine in intensity (I -> A·I + B), so forward-mode AD through the transfer is exact.
# Only the RT coefficients depend on the parameter — the geodesic march stays pure
# Float64 and only the coefficient/affine-couple accumulation carries a single-partial
# ForwardDiff.Dual seeded on param ∈ {:mass_scale (d_unit, e_unit ∝ m; b_unit ∝ √m),
# :rat_high (enters Θ_e)}. The device physics and kernel live in gpu_image.jl (number-
# type-generic). CUDA optional: NOT included from BlackLightPlus.jl — include explicitly.

using StaticArrays
using CUDA
using KernelAbstractions
using LinearAlgebra: dot
using ForwardDiff
using ForwardDiff: Dual, partials, value
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays
using BlackLightPlus.Simulation: units

# Bring in render_gpu_fused + the device helpers we reuse (geodesic RHS, metric, fluid
# decode, trilinear sample, unit tuple, grid up/resample) if not already loaded.
isdefined(@__MODULE__, :_gimg_coeffs) ||
    Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_image.jl"))

# Distinct AD tag: the intensity Dual never meets the geodesic RHS's own in-kernel Duals
# (the march is pure Float64 here), but a dedicated tag keeps ForwardDiff bookkeeping clean.
struct _GIADTag end

# Aliases: the generic gpu_image.jl bodies ARE the AD path (Dual-safe by construction).
const _giad_thermal = _gimg_thermal
const _fused_kernel_ad! = _fused_kernel!

# ---------------------------------------------------------------- host driver
# Seed one scalar of the isbits param tuple as a Dual{_GIADTag,Float64,1} (value + 1.0
# partial). Only the seeded field (and, for :mass_scale, the three unit fields it feeds)
# becomes Dual; the rest stay Float64/Int — the whole tuple remains isbits.
function _giad_params(cfg, param::Symbol)
    gp = _gimg_params(cfg)
    if param === :rat_high
        return merge(gp, (rat_high = Dual{_GIADTag}(gp.rat_high, 1.0),))
    elseif param === :mass_scale
        ms = Dual{_GIADTag}(cfg.mass_scale_cgs, 1.0)
        u = units(cfg.m_bh_cgs, ms)
        return merge(gp, (d_unit = u.dens_unit, e_unit = u.dens_unit * _CL^2,
                          b_unit = u.B_unit))
    else
        error("render_gpu_fused_grad: param must be :mass_scale or :rat_high, got :$param")
    end
end

# Render ray range `rng` on the CURRENTLY active device, output Dual per pixel.
function _giad_part!(out, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C, escape, R_em, nmax, DT)
    backend = CUDABackend()
    kern = _fused_kernel_ad!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{DT}(undef, cnt)
    kern(outd, y0d, grid_d, gp, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx, cnt;
         ndrange = cnt)
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

"""
    render_gpu_fused_grad(cfg, snap_or_grid, param::Symbol; devices=select_devices(),
                          nmax=cfg.max_steps, C=80, box_half=40.0, n=256)
        -> (image::Matrix{Float64}, dimage_dparam::Matrix{Float64},
            flux::Float64, dflux_dparam::Float64)

GPU + forward-mode-AD fused render: one forward pass returns the npix×npix unpolarized
`image` (identical to `render_gpu_fused`) and its exact per-pixel derivative w.r.t. the
scalar `param ∈ {:mass_scale, :rat_high}`, plus the total `flux` and `dflux_dparam`.
The geodesic march is Float64; only the RT accumulation carries a single-partial Dual.
`snap_or_grid` accepts an `AthenaKSnapshot`, `UniformFluidGrid`, or a pre-uploaded
`DeviceFluidGrid`; pixels split across `devices`.
"""
function render_gpu_fused_grad(cfg, snap_or_grid, param::Symbol; devices = select_devices(),
                               nmax::Int = cfg.max_steps, C::Real = 80, box_half::Real = 40.0,
                               n::Integer = 256)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)

    gp = _giad_params(cfg, param)
    DT = promote_type(typeof(gp.d_unit), typeof(gp.e_unit), typeof(gp.b_unit),
                      typeof(gp.rat_high), typeof(gp.rat_low))
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = length(rays)
    out = Vector{DT}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]     # interleaved: even cheap/expensive-ray mix per GPU
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _giad_part!(out, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                        dgrid.inv_dx, gp, a, r_h, Cf, escape, R_em, nmax, DT)
        end
    end
    outmat = reshape(out, cam.npix, cam.npix)
    image = value.(outmat)
    dimage = partials.(outmat, 1)
    fdual = sum(out)
    return (image, dimage, value(fdual), partials(fdual, 1))
end
