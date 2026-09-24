# gpu_pol.jl — fused single-pass GPU polarized imager (Stokes I,Q,U,V): one thread per
# ray runs `reverse_polarized_march` (radiation/reverse_march.jl), the march the CPU
# `render` runs: camera->source fixed-control RK4 with the parallel-transported camera
# polarization frame, and a reverse Mueller chain S += G(R b_seg); G <- G(R M_seg Rᵀ) of
# the analytic segment operators `stokes_segment` in the fluid tetrad (R the Q-U rotation
# into the transported frame; real arithmetic only). This file adds the kernels, the
# host drivers and the stiffness telemetry. CUDA optional: NOT included from
# BlackLightPlus.jl — include AFTER gpu_image.jl.

using StaticArrays
using ForwardDiff: Dual, partials, value
using KernelAbstractions
using CUDA
using LinearAlgebra: dot

using BlackLightPlus: KerrSchildCartesian
using BlackLightPlus.Geodesics: metric_and_inverse, rk4_step_fixed, geodesic_transport_rhs
using BlackLightPlus.Simulation: primitive_four_vectors, pgas_from_eint
using BlackLightPlus.Radiation: KerrMarch, reverse_polarized_march, _sample_coefficients

# ---------------------------------------------------------------- device: stiffness detector
# DIAGNOSTIC-ONLY transfer-matrix stiffness detector (RAPTOR check_stiffness; RAPTOR is
# GPLv3, so the algorithm is re-derived, not copied). The 4x4 polarized transfer matrix
# (LD85 / transfer.jl convention) has eigenvalues λ = α_I ± λ1 (real) and α_I ± i·λ2,
# with (λ1, λ2) the same combinations `stokes_segment` forms for its analytic solution.
# `_stiffness_metric` returns the dominant eigenvalue scale max(|λ|)·ds; a segment is
# 'stiff' above -log(0.99) (RAPTOR's |e^{τλ}| > 0.99 criterion — an explicit RK4 step
# would be poorly conditioned). The analytic coupling handles stiffness exactly
# regardless; this is telemetry only and does NOT touch the transport numerics.
const _STIFF_METRIC_THRESH = -log(0.99)   # ≈ 0.010050

@inline function _stiffness_lambdas(α::SVector{4}, ρ::SVector{4})
    alpha_sq = α[2]*α[2] + α[4]*α[4]
    rho_sq   = ρ[2]*ρ[2] + ρ[4]*ρ[4]
    ar  = α[2]*ρ[2] + α[4]*ρ[4]
    asr = alpha_sq - rho_sq
    la  = sqrt(asr*asr/4.0 + ar*ar)
    l1  = sqrt(max(la + asr/2.0, 0.0))    # real eigenvalue offset  (α_I ± λ1)
    l2  = sqrt(max(la - asr/2.0, 0.0))    # imag eigenvalue offset  (α_I ± i λ2)
    return l1, l2
end

"Dominant transfer-matrix eigenvalue scale max(|λ|)·ds (device-safe, real arithmetic)."
@inline function _stiffness_metric(α::SVector{4}, ρ::SVector{4}, ds)
    l1, l2 = _stiffness_lambdas(α, ρ)
    aI = α[1]
    mre = max(abs(aI + l1), abs(aI - l1))    # real eigenvalues
    mim = sqrt(aI*aI + l2*l2)                # |α_I ± i λ2|
    return max(mre, mim) * ds
end

"Stiff-segment flag: max(|λ|)·ds exceeds the RAPTOR |e^{τλ}|>0.99 threshold."
@inline _stiffness_is_stiff(α::SVector{4}, ρ::SVector{4}, ds, thresh = _STIFF_METRIC_THRESH) =
    _stiffness_metric(α, ρ, ds) > thresh

# Reference implementation (host-only; ComplexF64, NOT used on-device): the RK4
# stability polynomial |1+τλ+τ²λ²/2+τ³λ³/6+τ⁴λ⁴/24| maximized over the four
# eigenvalues. Used in tests to show _stiffness_lambdas matches and the metric gate
# tracks the RK4 criterion.
function _stiffness_rk4_maxmag(α::SVector{4}, ρ::SVector{4}, ds)
    aI = α[1]; aQ = α[2]; aV = α[4]; rQ = ρ[2]; rV = ρ[4]
    a2 = rQ*rQ + rV*rV - aQ*aQ - aV*aV
    a0 = -2.0*aV*aQ*rV*rQ - aQ*aQ*rQ*rQ - aV*aV*rV*rV
    disc = sqrt(Complex(a2*a2 - 4.0*a0))
    zplus  = (-a2 + disc) / 2
    zminus = (-a2 - disc) / 2
    ls = (aI + sqrt(zplus), aI - sqrt(zplus), aI + sqrt(zminus), aI - sqrt(zminus))
    mm = 0.0
    for l in ls
        τ = ds * l
        mag = abs(1 + τ + τ*τ/2 + τ*τ*τ/6 + τ*τ*τ*τ/24)
        mm = max(mm, mag)
    end
    return mm
end

# Unit normal (Eulerian) observer n^μ = -α g^{0μ} at the camera — the CPU
# Radiation._normal_observer itself (pure @inline SVector math, hence device-safe).
# Kept under the _gpol_ name for the kernels/tests; parallel-transported here as the
# readout screen's timelike leg.
using BlackLightPlus.Radiation: _normal_observer
const _gpol_normal_observer = _normal_observer

# ---------------------------------------------------------------- march
# Every polarized kernel runs `reverse_polarized_march` (radiation/reverse_march.jl, the
# CPU `render` march) under a `KerrMarch` control; the kernels differ only in the fluid
# sampler. The geodesic and the transported frame run in the ray eltype (fp32 or fp64),
# the Stokes/Mueller transfer and every coefficient in Float64 from the converted state.

# ---------------------------------------------------------------- fused polarized kernel
# One ray/thread on the uniform grid; writes out[i]. Two-type: the geodesic/frame lane
# runs at eltype(eltype(y0s)) — the host passes (a, r_h, C, escape, R_em) at that type —
# while the sampler position is converted to Float64 (identity for fp64 rays), so the
# trilinear sample + all Stokes/coefficient arithmetic stay Float64 in every mode.
@kernel function _gpol_kernel!(out, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                               R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = reverse_polarized_march(y0s[i],
                             y -> _gimg_sample(grid, ng, xo, inv_dx, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             gp, KerrMarch(a, r_h, C, escape, R_em), nmax)
    end
end

# ---------------------------------------------------------------- host driver
# The unpolarized parameter set plus the camera vertical (image β direction).
function _gpol_params(cfg)
    i = deg2rad(cfg.camera.inclination_deg)
    return merge(_gimg_params(cfg), (cu1 = 0.0, cu2 = -cos(i), cu3 = 0.0, cu4 = sin(i)))
end

# The polarized kernel is register-heavy (parallel transport + Mueller chain) and
# register-capped at 255 in both precisions, so only ~256 threads fit per SM; wg=64
# yields 4 resident blocks (best latency hiding) and measured fastest. Block size is
# occupancy-only (results are bit-identical across wg).
const _GPOL_WG = 64

# Launch of the uniform-grid polarized kernel for one chunk. The geodesic/frame-lane type
# is eltype(eltype(y0d)); the caller passes (a, r_h, C, escape, R_em) at that type.
_gpol_launch(dgrid, gp, a, r_h, C, escape, R_em, nmax) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_gpol_kernel!(lane.backend), _GPOL_WG, n, outd, y0d, dgrid.dev_data[lane.dev],
                  gp, a, r_h, C, escape, R_em, Int(nmax), dgrid.n, dgrid.xo, dgrid.inv_dx, n)

# Per-ray Stokes 4-vectors -> the (I, Q, U, V) npix×npix planes of `render(...; polarized=true)`.
_stokes_planes(out::AbstractVector{<:SVector{4}}, npx) =
    NamedTuple{(:I, :Q, :U, :V)}(ntuple(k -> reshape([o[k] for o in out], npx, npx), 4))

"""
    render_gpu_polarized(cfg, snap_or_grid; devices=default_pool(), nmax=cfg.max_steps,
                         C=80, box_half=40.0, n=256, precision=:f64, rescue_px=2)
        -> NamedTuple(I,Q,U,V)

Fused single-pass GPU render of the full-Stokes npix×npix polarized image: one thread
per ray marches the null geodesic camera->source, parallel-transports the polarization
frame, and accumulates the observed Stokes with a running Mueller transmission (reverse
RT). `snap_or_grid`: `AthenaKSnapshot`, `UniformFluidGrid` (each uploaded once per device
pool), or pre-uploaded `DeviceFluidGrid`. Each of I,Q,U,V is a `Matrix{Float64}` in
`render(...;polarized=true)` layout; `order` sets the queue order (default
`cost_order`; bit-exact per pixel). `precision`: `:f64` (default, bitwise unchanged) ·
`:mixed` (fp32 geodesic march + frame transport, Float64 Stokes/Mueller/coefficients from
converted state; the near-critical-curve pixels are marched in fp64 instead, queued first
— `rescued_frac::Ref{Float64}` receives their fraction) · `:f32_raw` (no rescue,
DIAGNOSTIC only — near-critical pixels carry O(1) errors).
"""
function render_gpu_polarized(cfg, snap_or_grid; devices = default_pool(),
                              nmax::Int = cfg.max_steps,   # step limit wired from RunConfig, as render()
                              C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                              order = nothing,
                              precision::Symbol = :f64, rescue_px::Int = 2,
                              rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_polarized: precision must be :f64, :mixed or :f32_raw, got :$precision")
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    dgrid = resident_grid(pool, snap_or_grid; box_half = box_half, n = n)

    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = cam.npix^2
    items = _frame_items(order, cfg, cam.npix)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    f64 = _gpol_launch(dgrid, gp, a, r_h, Cf, escape, R_em, nmax)
    # fp32 geodesic/frame lane: the fp64 rays are narrowed as the device makes them; the
    # host constructs every march scalar as Float32 (no in-kernel widening), while gp
    # (units, frequency, fp64 spin gp.bh_a for the RT lane) stays Float64.
    f32() = _gpol_launch(dgrid, gp, Float32(a), Float32(r_h), Float32(Cf),
                         Float32(escape), Float32(R_em), nmax)
    frac = 0.0
    if precision === :f64
        march!((out,), pool, [MarchPart(items, CameraRays(st, cam), f64)])
    elseif precision === :f32_raw
        march!((out,), pool, [MarchPart(items, CameraRays(st, cam, Float32), f32())])
    else
        pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                             rescue_px = rescue_px)
        march!((out,), pool, _mixed_parts(items, pix, CameraRays(st, cam),
                                          CameraRays(st, cam, Float32), f64, f32()))
        frac = length(pix) / nrays
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    return _stokes_planes(out, cam.npix)
end

# ================================================================ SLOW LIGHT (opt-in, polarized)
# The SAME march with the sampler swapped for the time-blended
# _gimg_sample_t (gpu_image.jl); w = clamp((t0 + X⁰) * inv_dt, 0, 1) at the emission
# time t_emit = t_obs + X⁰, t0 = t_obs - t_a (the physical dump-time axis — see the
# gpu_image.jl SLOW LIGHT header). inv_dt=0 (fast-light limit) makes _gimg_sample_t
# return grid-A's sample bit-for-bit -> render_gpu_polarized_slow reproduces
# render_gpu_polarized(win.a).
@kernel function _gpol_kernel_sl!(out, @Const(y0s), @Const(gridA), @Const(gridB),
                                  t0, inv_dt, gp, a, r_h, C, escape,
                                  R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = reverse_polarized_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gimg_sample_t(gridA, gridB, ng, xo, inv_dx, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             gp, KerrMarch(a, r_h, C, escape, R_em), nmax)
    end
end

# Launch of the slow-light polarized kernel for one chunk.
_gpol_launch_sl(win, t0, gp, a, r_h, C, escape, R_em, nmax) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_gpol_kernel_sl!(lane.backend), _GPOL_WG, n, outd, y0d,
                  win.a.dev_data[lane.dev], win.b.dev_data[lane.dev], t0, win.inv_dt, gp,
                  a, r_h, C, escape, R_em, Int(nmax), win.a.n, win.a.xo, win.a.inv_dx, n)

"""
    render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices=default_pool(),
                              nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                              time_offset=0.0) -> NamedTuple(I,Q,U,V)

Fused single-pass SLOW-LIGHT polarized render on a `DeviceFluidWindow`: identical
transport to `render_gpu_polarized`, but each ray blends the two window buffers at
its EMISSION time t_emit = t_obs + X⁰ (physical dump-time axis; `t_obs` defaults to
the window's upper end, `time_offset` advances it — movie frame time).
`win.inv_dt == 0` reproduces `render_gpu_polarized(cfg, win.a)` bit-for-bit
(fast-light limit). Accepts `order` as `render_gpu_polarized`.
"""
function render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices = default_pool(),
                                   nmax::Int = cfg.max_steps, C::Real = 80,
                                   t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                                   order = nothing)
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(win.a.xo)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    nrays = cam.npix^2
    out = Vector{SVector{4,Float64}}(undef, nrays)
    march!((out,), pool, [MarchPart(_frame_items(order, cfg, cam.npix), CameraRays(st, cam),
                                    _gpol_launch_sl(win, t0, gp, a, r_h, Cf, escape, R_em, nmax))])
    return _stokes_planes(out, cam.npix)
end

# ---------------------------------------------------------------- stiffness telemetry kernel
# DIAGNOSTIC marcher: same geodesic stepping + fluid decode as _gpol_kernel! (the
# geodesic step is `rk4_step_fixed`, whose RHS is the geodesic part of the polarized
# march's `geodesic_transport_rhs`,
# and the same per-sample coefficients, so the sampled segments are the polarized kernel's), but
# instead of accumulating Stokes it tallies per ray the emitting segments, how many are
# 'stiff' (max(|λ|)·ds > thresh), and the max/sum metric. Kept entirely separate from
# _gpol_kernel! so the production kernel stays bit-identical. Output per ray:
# SVector(n_valid, n_stiff, max_metric, sum_metric).
@kernel function _gpol_stiff_kernel!(out, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                                     R_em, nmax, ng, xo, inv_dx, thresh, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        y = y0s[i]
        n_valid = 0.0; n_stiff = 0.0; max_m = 0.0; sum_m = 0.0
        r_prev = Inf
        for _ in 1:nmax
            stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
            stop && break
            r_prev = r
            h = (r - r_h) / C
            x = SVector(y[1], y[2], y[3], y[4])
            prim = _gimg_sample(grid, ng, xo, inv_dx, y[2], y[3], y[4])
            if prim[1] > 0.0
                gcov, gcon = metric_and_inverse(KerrSchildCartesian(gp.bh_a), x)
                kcov = -SVector(y[5], y[6], y[7], y[8])     # photon k = -p
                j, al, ro, tet_fl, valid = _sample_coefficients(prim, kcov, gcon * kcov, gcov,
                                                                gcon, gp, false)
                if valid
                    ds = h * gp.L_unit / gp.nu_obs
                    m = _stiffness_metric(al, ro, ds)
                    n_valid += 1.0
                    sum_m += m
                    max_m = max(max_m, m)
                    (m > thresh) && (n_stiff += 1.0)
                end
            end
            # geodesic-only RK4: no polarization state is marched here (the transports
            # of two zero vectors would be dead work).
            y = rk4_step_fixed(KerrSchildCartesian(a), y, h)
        end
        out[i] = SVector(n_valid, n_stiff, max_m, sum_m)
    end
end

# Launch of the stiffness telemetry kernel for one chunk.
_gpol_launch_stiff(dgrid, gp, a, r_h, C, escape, R_em, nmax, thresh) = (lane, (outd,), y0d, n) ->
    _kern_launch!(_gpol_stiff_kernel!(lane.backend), _GPOL_WG, n, outd, y0d,
                  dgrid.dev_data[lane.dev], gp, a, r_h, C, escape, R_em, Int(nmax), dgrid.n,
                  dgrid.xo, dgrid.inv_dx, Float64(thresh), n)

"""
    stiffness_telemetry(cfg, snap_or_grid; devices, nmax, C, box_half, n,
                        thresh=_STIFF_METRIC_THRESH) -> NamedTuple

DIAGNOSTIC pass over the SAME rays/segments the fused polarized kernel marches, reporting
the transfer-matrix stiffness distribution (RAPTOR check_stiffness telemetry). Returns
`(n_valid, n_stiff, frac_stiff, max_metric, mean_metric, per_ray_valid, per_ray_stiff)`:
`frac_stiff` is the fraction of emitting segments whose dominant eigenvalue scale
max(|λ|)·ds exceeds `thresh` (default the RAPTOR |e^{τλ}|>0.99 gate). Telemetry only — the
polarized transport is unchanged (our analytic coupling is exact in stiff cells).
"""
function stiffness_telemetry(cfg, snap_or_grid; devices = default_pool(),
                             nmax::Int = cfg.max_steps,
                             C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                             thresh::Real = _STIFF_METRIC_THRESH)
    pool = device_pool(devices)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    dgrid = resident_grid(pool, snap_or_grid; box_half = box_half, n = n)
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)
    nrays = cam.npix^2
    out = Vector{SVector{4,Float64}}(undef, nrays)
    march!((out,), pool, [MarchPart(_frame_items(nothing, cfg, cam.npix), CameraRays(st, cam),
                                    _gpol_launch_stiff(dgrid, gp, a, r_h, Cf, escape, R_em,
                                                       nmax, thresh))])
    nv = sum(o[1] for o in out)
    ns = sum(o[2] for o in out)
    mx = maximum(o[3] for o in out)
    sm = sum(o[4] for o in out)
    return (n_valid = nv, n_stiff = ns, frac_stiff = nv > 0 ? ns / nv : 0.0,
            max_metric = mx, mean_metric = nv > 0 ? sm / nv : 0.0,
            per_ray_valid = [o[1] for o in out], per_ray_stiff = [o[2] for o in out])
end
