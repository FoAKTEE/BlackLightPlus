# gpu_pol.jl — fused single-pass GPU polarized imager (Stokes I,Q,U,V). The CPU
# coherency march (polarized_march) runs source->camera over a stored trajectory; the
# fused tracer integrates camera->source and stores nothing, and RT is order-dependent,
# so the transfer is accumulated in reverse: the Stokes accumulator S plus a running
# REAL 4x4 Mueller transmission G (no ComplexF64 in-kernel) in a parallel-transported
# camera polarization frame, S += G(R b_seg); G <- G(R M_seg Rᵀ), with (M,b) the
# analytic segment operator (transfer.jl::stokes_segment) in the fluid tetrad and R the
# Q-U rotation into the transported frame. Exact reverse RT, validated against the CPU
# march. Coefficients (`eight_coefficients`), fluid tetrad, segment operator, clamp and
# frame transport (`geodesic_transport_rhs`) are the CPU functions called in-kernel.
# CUDA optional: NOT included from BlackLightPlus.jl — include AFTER gpu_image.jl.

using StaticArrays
using ForwardDiff: Dual, partials, value
using KernelAbstractions
using CUDA
using LinearAlgebra: dot

using BlackLightPlus: KerrSchildCartesian
using BlackLightPlus.Geodesics: metric_and_inverse, rk4_step_fixed, geodesic_transport_rhs
using BlackLightPlus.Simulation: primitive_four_vectors, pgas_from_eint
using BlackLightPlus.Radiation: fluid_tetrad, eight_coefficients, stokes_segment,
      stokes_admissible

const _gpz4  = SVector(0.0, 0.0, 0.0, 0.0)
const _gpI4  = one(SMatrix{4,4,Float64})

# One fluid sample -> the eight invariant coefficients and the fluid tetrad on the RT
# lane (Float64): normal-frame primitive decode, then `eight_coefficients`. The fluid
# tetrad renormalizes e_(1) off single CKS (|det g| ≠ 1), as the CPU march does.
@inline function _gpol_coeffs(prim::SVector{8,Float64}, kcov::SVector{4,Float64},
                              kcon::SVector{4,Float64}, gcov::SMatrix{4,4},
                              gcon::SMatrix{4,4}, gp, renorm_e1::Bool)
    fv = primitive_four_vectors(prim[2], prim[3], prim[4], prim[6], prim[7], prim[8],
                                gcov, gcon)
    return eight_coefficients(gp, prim[1], pgas_from_eint(prim[5], gp.plasma_gamma), fv,
                              gcov, gcon, kcov, kcon, gp.nu_obs, renorm_e1)
end

# ---------------------------------------------------------------- device: analytic R
# Stokes rotation fluid-tetrad -> readout-tetrad E: pure Q-U rotation by 2χ (I,V fixed),
# χ = angle of E's e_(1) in the fluid (e_(1),e_(2)) basis. Reproduces the coherency
# round-trip `_coord_to_tetrad_cov` (validated against the CPU path). Returns
# (cos 2χ, sin 2χ); R = [1 0 0 0; 0 c s 0; 0 -s c 0; 0 0 0 1] is applied by
# `_gpol_rot` without forming the 4×4 matrix.
@inline function _gpol_R(tet_fl::SMatrix{4,4}, E::SMatrix{4,4}, gcov::SMatrix{4,4})
    e1f = SVector(tet_fl[2,1], tet_fl[2,2], tet_fl[2,3], tet_fl[2,4])
    e2f = SVector(tet_fl[3,1], tet_fl[3,2], tet_fl[3,3], tet_fl[3,4])
    E1  = SVector(E[2,1], E[2,2], E[2,3], E[2,4])
    c = dot(E1, gcov * e1f); s = dot(E1, gcov * e2f)
    nrm = sqrt(c*c + s*s); c /= nrm; s /= nrm
    return c*c - s*s, 2.0*s*c
end

# R b and R M Rᵀ for the Q-U rotation R of `_gpol_R`: only the Q, U rows (and columns)
# mix.
@inline _gpol_rot(c, s, b::SVector{4}) =
    SVector(b[1], c * b[2] + s * b[3], c * b[3] - s * b[2], b[4])
@inline function _gpol_rot(c, s, M::SMatrix{4,4})
    A = SMatrix{4,4}(M[1,1], c*M[2,1] + s*M[3,1], c*M[3,1] - s*M[2,1], M[4,1],
                     M[1,2], c*M[2,2] + s*M[3,2], c*M[3,2] - s*M[2,2], M[4,2],
                     M[1,3], c*M[2,3] + s*M[3,3], c*M[3,3] - s*M[2,3], M[4,3],
                     M[1,4], c*M[2,4] + s*M[3,4], c*M[3,4] - s*M[2,4], M[4,4])   # R M
    return SMatrix{4,4}(A[1,1], A[2,1], A[3,1], A[4,1],
                        c*A[1,2] + s*A[1,3], c*A[2,2] + s*A[2,3], c*A[3,2] + s*A[3,3], c*A[4,2] + s*A[4,3],
                        c*A[1,3] - s*A[1,2], c*A[2,3] - s*A[2,2], c*A[3,3] - s*A[3,2], c*A[4,3] - s*A[4,2],
                        A[1,4], A[2,4], A[3,4], A[4,4])                        # (R M) Rᵀ
end

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

# ---------------------------------------------------------------- fused polarized march
# The entire polarized fused march for one ray, camera->source. Per-thread state:
# geodesic y(8), transported up-vector eU(4) + timelike leg eUu(4), accumulated Stokes
# S(4), running Mueller transmission G(16) — all real. The one body behind every
# polarized kernel variant, parameterized by the inlined fluid sampler `sample(y)`.
# Two-type: the geodesic state and transported frame legs run in Tg ∈ {Float32, Float64}
# with matched-::Tg march scalars; the state is converted to Float64 ONCE per fluid-
# sample/RT step (identity when Tg == Float64), so the Stokes/Mueller coupling and every
# coefficient stay Float64 in every mode. The RT-lane metric uses the fp64 spin gp.bh_a.
@inline function _gpol_march(y0::SVector{8,Tg}, sample::F,
                             gp, a::Tg, r_h::Tg, C::Tg, escape::Tg, R_em::Tg,
                             nmax) where {F,Tg}
    y = y0
    # camera readout frame: built in Float64 from the CONVERTED initial state (identity
    # at Tg == Float64), then NARROWED onto the Tg transport lane
    y64 = SVector{8,Float64}(y0)
    x0 = SVector(y64[1], y64[2], y64[3], y64[4]); kcov0 = -SVector(y64[5], y64[6], y64[7], y64[8])
    st = KerrSchildCartesian(gp.bh_a)            # RT-lane metric (fp64 spin)
    gcov0, gcon0 = metric_and_inverse(st, x0); kcon0 = gcon0 * kcov0
    # camera readout tetrad -> initial transported up-vector eU (= its e_(2))
    e0c = _gpol_normal_observer(gcon0)   # unit Eulerian observer (CPU shift-sign fix mirror)
    e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = fluid_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0)
    eU  = SVector{4,Tg}(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])   # camera e_(2) (up)
    eUu = SVector{4,Tg}(e0c)     # camera 4-velocity — parallel-transported alongside eU

    S = _gpz4; G = _gpI4
    r_prev = Tg(Inf)
    for _ in 1:nmax
        stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
        stop && break
        r_prev = r
        h = (r - r_h) / C
        # callsite @inline: the inliner otherwise leaves the sampler closure as a call
        # in this register-heavy body, which perturbs downstream fmuladd contraction
        # (~1 ulp, Faraday-amplified on dim pixels); forced inlining keeps the kernel
        # bit-for-bit stable.
        prim = @inline sample(y)
        if prim[1] > 0.0
            # RT lane: Float64 from the CONVERTED state (identity at Tg == Float64)
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            gcov, gcon = metric_and_inverse(st, x)
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8]); kcon = gcon * kcov   # photon k = -p
            j, al, ro, tet_fl, valid = _gpol_coeffs(prim, kcov, kcon, gcov, gcon, gp, false)
            if valid
                # readout screen = the parallel-transported camera tetrad: its
                # timelike leg is PT(u_cam)=eUu (not the per-sample local ZAMO,
                # which drifts and rotates EVPA per ray). fluid_tetrad uses only
                # parallel-transport-invariant operations, so E == PT(Ecam) and S
                # ends up in the camera frame with no final rotation (cf. ipole/Krang).
                # The march ends at the shared redshift cut (march_stop), before the
                # transported frame's KS components (∝ 1/(r - r_h)) degrade the
                # invariant -k·eUu (drift < 1e-2 in fp64 and fp32 up to the cut).
                eU64 = SVector{4,Float64}(eU); eUu64 = SVector{4,Float64}(eUu)
                E = fluid_tetrad(eUu64, gcov * eUu64, kcon, kcov, eU64, gcov, gcon)
                ds = Float64(h) * gp.L_unit / gp.nu_obs
                M, b = stokes_segment(j, al, ro, ds)
                c2, s2 = _gpol_R(tet_fl, E, gcov)
                S = stokes_admissible(S + G * _gpol_rot(c2, s2, b))   # the stokes_couple clamp
                G = G * _gpol_rot(c2, s2, M)
            end
        end
        # low-storage streaming RK4: same ops/associativity as the 4-stage combine but
        # the four stages of {y,eU,eUu} are not all held live -> fewer registers/thread.
        stg = KerrSchildCartesian(a)             # geodesic-lane metric (lane spin)
        dy, (de, du) = geodesic_transport_rhs(stg, y, (eU, eUu))
        ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + dy; ae = ae + de; au = au + du
        y   = y   + (h/6)*ay
        eU  = eU  + (h/6)*ae
        eUu = eUu + (h/6)*au
    end
    return S * gp.nu_obs^3
end

# ---------------------------------------------------------------- fused polarized kernel
# One ray/thread on the uniform grid; writes out[i]. Two-type: the geodesic/frame lane
# runs at eltype(eltype(y0s)) — the host passes (a, r_h, C, escape, R_em) at that type —
# while the sampler position is converted to Float64 (identity for fp64 rays), so the
# trilinear sample + all Stokes/coefficient arithmetic stay Float64 in every mode.
@kernel function _gpol_kernel!(out, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                               R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gpol_march(y0s[i],
                             y -> _gimg_sample(grid, ng, xo, inv_dx, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             gp, a, r_h, C, escape, R_em, nmax)
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

function _gpol_part!(out, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C, escape, R_em, nmax)
    backend = KernelAbstractions.get_backend(grid_d)
    kern = _gpol_kernel!(backend)
    cnt = length(rng)
    y0d = _upload(backend, rays[rng])
    outd = KernelAbstractions.allocate(backend, SVector{4,Float64}, cnt)
    kern(outd, y0d, grid_d, gp, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx, cnt;
         ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

# Launch the polarized kernel over the ray-index `pool` (any index collection), pixels
# INTERLEAVED across `devices` (the _fused_sweep! pattern — pool == 1:nrays reproduces
# the pre-mixed-precision device split exactly). The geodesic/frame-lane type is
# eltype(eltype(rays)); the caller passes (a, r_h, C, escape, R_em) at that type.
function _gpol_sweep!(out, rays, pool, devices, dgrid, gp, a, r_h, C, escape, R_em, nmax)
    nd = length(devices)
    @sync for di in 1:nd
        idx = pool[di:nd:end]; dev = devices[di]
        isempty(idx) && continue
        @async begin
            _device!(dev)
            _gpol_part!(out, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                        dgrid.inv_dx, gp, a, r_h, C, escape, R_em, nmax)
        end
    end
    return nothing
end

"""
    render_gpu_polarized(cfg, snap_or_grid; devices=select_devices(), nmax=cfg.max_steps,
                         C=80, box_half=40.0, n=256, precision=:f64, rescue_px=2)
        -> NamedTuple(I,Q,U,V)

Fused single-pass GPU render of the full-Stokes npix×npix polarized image: one thread
per ray marches the null geodesic camera->source, parallel-transports the polarization
frame, and accumulates the observed Stokes with a running Mueller transmission (reverse
RT). `snap_or_grid`: `AthenaKSnapshot`, `UniformFluidGrid`, or pre-uploaded
`DeviceFluidGrid`. Each of I,Q,U,V is a `Matrix{Float64}` in `render(...;polarized=true)`
layout; `order` marches rays in trip-count order (bit-exact per pixel). `precision`:
`:f64` (default, bitwise unchanged) · `:mixed` (fp32 geodesic march + frame transport,
Float64 Stokes/Mueller/coefficients from converted state; near-critical-curve pixels
re-rendered in fp64 and spliced — `rescued_frac::Ref{Float64}` receives the fraction) ·
`:f32_raw` (no rescue, DIAGNOSTIC only — near-critical pixels carry O(1) errors).
"""
function render_gpu_polarized(cfg, snap_or_grid; devices = select_devices(),
                              nmax::Int = cfg.max_steps,   # step limit wired from RunConfig, as render()
                              C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                              order = nothing,
                              precision::Symbol = :f64, rescue_px::Int = 2,
                              rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_polarized: precision must be :f64, :mixed or :f32_raw, got :$precision")
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)

    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    frac = 0.0
    if precision === :f64
        _gpol_sweep!(out, rays, 1:nrays, devices, dgrid, gp, a, r_h, Cf, escape, R_em, nmax)
    else
        # fp32 geodesic/frame lane: narrow the fp64-built rays at upload; the host
        # constructs every march scalar as Float32 (no in-kernel widening), while gp
        # (units, frequency, fp64 spin gp.bh_a for the RT lane) stays Float64.
        rays32 = [SVector{8,Float32}(r) for r in rays]
        _gpol_sweep!(out, rays32, 1:nrays, devices, dgrid, gp, Float32(a), Float32(r_h),
                     Float32(Cf), Float32(escape), Float32(R_em), nmax)
        if precision === :mixed
            pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                                 rescue_px = rescue_px)
            # pixel -> ordered-ray position (out/rays live in `order`ed space until _undo_order)
            pos = order === nothing ? pix : invperm(order)[pix]
            isempty(pos) ||
                _gpol_sweep!(out, rays, pos, devices, dgrid, gp, a, r_h, Cf, escape,
                             R_em, nmax)
            frac = length(pos) / nrays
        end
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    out = _undo_order(out, order)
    npx = cam.npix
    I = reshape([o[1] for o in out], npx, npx)
    Q = reshape([o[2] for o in out], npx, npx)
    U = reshape([o[3] for o in out], npx, npx)
    V = reshape([o[4] for o in out], npx, npx)
    return (I = I, Q = Q, U = U, V = V)
end

# ================================================================ SLOW LIGHT (opt-in, polarized)
# The SAME _gpol_march body with the sampler swapped for the time-blended
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
        out[i] = _gpol_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gimg_sample_t(gridA, gridB, ng, xo, inv_dx, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             gp, a, r_h, C, escape, R_em, nmax)
    end
end

function _gpol_part_sl!(out, rays, rng, gA, gB, ng, xo, inv_dx, t0, inv_dt,
                        gp, a, r_h, C, escape, R_em, nmax)
    backend = KernelAbstractions.get_backend(gA)
    kern = _gpol_kernel_sl!(backend)
    cnt = length(rng)
    y0d = _upload(backend, rays[rng])
    outd = KernelAbstractions.allocate(backend, SVector{4,Float64}, cnt)
    kern(outd, y0d, gA, gB, t0, inv_dt, gp, a, r_h, C, escape, R_em, Int(nmax),
         ng, xo, inv_dx, cnt; ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

"""
    render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices=select_devices(),
                              nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                              time_offset=0.0) -> NamedTuple(I,Q,U,V)

Fused single-pass SLOW-LIGHT polarized render on a `DeviceFluidWindow`: identical
transport to `render_gpu_polarized`, but each ray blends the two window buffers at
its EMISSION time t_emit = t_obs + X⁰ (physical dump-time axis; `t_obs` defaults to
the window's upper end, `time_offset` advances it — movie frame time).
`win.inv_dt == 0` reproduces `render_gpu_polarized(cfg, win.a)` bit-for-bit
(fast-light limit). Accepts `order` as `render_gpu_polarized`.
"""
function render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices = select_devices(),
                                   nmax::Int = cfg.max_steps, C::Real = 80,
                                   t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                                   order = nothing)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    ga = win.a; gb = win.b
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(ga.xo)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            _device!(dev)
            _gpol_part_sl!(out, rays, idx, ga.dev_data[dev], gb.dev_data[dev], ga.n,
                           ga.xo, ga.inv_dx, t0, win.inv_dt, gp, a, r_h, Cf,
                           escape, R_em, nmax)
        end
    end
    out = _undo_order(out, order)
    npx = cam.npix
    I = reshape([o[1] for o in out], npx, npx)
    Q = reshape([o[2] for o in out], npx, npx)
    U = reshape([o[3] for o in out], npx, npx)
    V = reshape([o[4] for o in out], npx, npx)
    return (I = I, Q = Q, U = U, V = V)
end

# ---------------------------------------------------------------- stiffness telemetry kernel
# DIAGNOSTIC marcher: same geodesic stepping + fluid decode as _gpol_kernel! (the
# geodesic step is `rk4_step_fixed`, whose RHS is the geodesic part of the polarized
# march's `geodesic_transport_rhs`,
# and _gpol_coeffs, so the sampled segments are the polarized kernel's), but
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
                j, al, ro, tet_fl, valid = _gpol_coeffs(prim, kcov, gcon * kcov, gcov, gcon,
                                                        gp, false)
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

function _gpol_stiff_part!(out, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C,
                           escape, R_em, nmax, thresh)
    backend = KernelAbstractions.get_backend(grid_d)
    kern = _gpol_stiff_kernel!(backend)
    cnt = length(rng)
    y0d = _upload(backend, rays[rng])
    outd = KernelAbstractions.allocate(backend, SVector{4,Float64}, cnt)
    kern(outd, y0d, grid_d, gp, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx,
         Float64(thresh), cnt; ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

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
function stiffness_telemetry(cfg, snap_or_grid; devices = select_devices(),
                             nmax::Int = cfg.max_steps,
                             C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                             thresh::Real = _STIFF_METRIC_THRESH)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            _device!(dev)
            _gpol_stiff_part!(out, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                              dgrid.inv_dx, gp, a, r_h, Cf, escape, R_em, nmax, thresh)
        end
    end
    nv = sum(o[1] for o in out)
    ns = sum(o[2] for o in out)
    mx = maximum(o[3] for o in out)
    sm = sum(o[4] for o in out)
    return (n_valid = nv, n_stiff = ns, frac_stiff = nv > 0 ? ns / nv : 0.0,
            max_metric = mx, mean_metric = nv > 0 ? sm / nv : 0.0,
            per_ray_valid = [o[1] for o in out], per_ray_stiff = [o[2] for o in out])
end
