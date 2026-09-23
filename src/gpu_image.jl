# gpu_image.jl — fused single-pass GPU unpolarized imaging: one thread per ray marches
# the Hamiltonian null geodesic (in-kernel ForwardDiff-Dual RHS), samples the fluid, and
# accumulates the invariant intensity, returning only the per-pixel image. The CPU march
# runs source->camera with the affine couple I_n = A_n I_{n-1} + B_n; here we march
# camera->source and accumulate the same sum in reverse, I_obs = Σ_n B_n Π_{k>n} A_k,
# via a running transmission (identical up to fp round-off). Pitch angle: the invariant
# identity cos²θ_B = (k·b)²/((k·u)² b²) replaces the CPU fluid tetrad. Approximation:
# the AMR fluid is resampled once onto a uniform box and trilinearly sampled in-kernel
# (validated against the CPU path). CUDA optional: NOT included from BlackLightPlus.jl.

using StaticArrays
using ForwardDiff: Dual, partials
using KernelAbstractions
using CUDA
using LinearAlgebra: dot
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays, critical_curve
using BlackLightPlus.Simulation: AthenaKSnapshot, sample_primitives, pgas_from_eint, units,
      SnapshotSeries, get_frame, bracketing_pair
# host precompute of the nonthermal eDF constants (_powerlaw_gp / _kappa_gp); explicit
# here so the device-params builder resolves them in any include scope.
using BlackLightPlus.Radiation: SimCoeffConfig, precompute_power_law!, precompute_kappa!

# ---------------------------------------------------------------- CGS constants
# Literals (BlackLightPlus.Constants values); usable as compile-time consts in a kernel.
const _CL  = 2.99792458e10
const _EE  = 4.8032e-10
const _ME  = 9.1094e-28
const _MP  = 1.6726e-24
const _HPL = 6.6261e-27
const _SQRT2 = sqrt(2.0)

# ---------------------------------------------------------------- uniform fluid grid
"""
    UniformFluidGrid

Host-resident dense Cartesian resample of the AMR fluid: `host[f,i,j,k]` holds the 8
primitives (dens, uu1, uu2, uu3, eint, bb1, bb2, bb3) at grid node
`x = xo + (i-1)/inv_dx` (same for y,j / z,k), on `[-box_half, box_half]^3`. Built once
on the host; `render_gpu_fused` uploads it per device.
"""
struct UniformFluidGrid
    host::Array{Float32,4}    # (8, n, n, n)
    n::Int
    box_half::Float64
    xo::Float64               # = -box_half
    inv_dx::Float64           # = (n-1) / (2 box_half)
end

"""
    DeviceFluidGrid

Per-device pre-uploaded copies of a `UniformFluidGrid` (id => CuArray{Float32,4}),
so repeated renders / both GPUs reuse one upload. Built by `upload_grid`.
"""
struct DeviceFluidGrid
    dev_data::Dict{Int,CuArray{Float32,4}}
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

Upload `fg.host` to each CUDA device in `devices` (one CuArray per device).
"""
function upload_grid(fg::UniformFluidGrid, devices)
    d = Dict{Int,CuArray{Float32,4}}()
    for dev in devices
        CUDA.device!(dev)
        d[dev] = CuArray(fg.host)
    end
    return DeviceFluidGrid(d, fg.n, fg.xo, fg.inv_dx)
end

# ---------------------------------------------------------------- device: geodesic RHS
# The closed-form KS device metric + in-kernel 3-lane-AD RHS (_gimg_gi/_gimg_rhs/
# _gimg_rks/_gimg_metric) live in the shared device-metric layer gpu_metric.jl.
isdefined(@__MODULE__, :_gimg_gi) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_metric.jl"))

# ---------------------------------------------------------------- device: fluid decode
# Normal-frame primitive -> (u^μ, u_μ, b^μ, b^μ b_μ), verbatim from primitives.jl.
@inline function _gimg_pfv(uu1, uu2, uu3, bb1, bb2, bb3,
                           gcov::SMatrix{4,4}, gcon::SMatrix{4,4})
    uu0 = sqrt(1 +
        gcov[2, 2] * uu1 * uu1 + 2 * gcov[2, 3] * uu1 * uu2 + 2 * gcov[2, 4] * uu1 * uu3 +
        gcov[3, 3] * uu2 * uu2 + 2 * gcov[3, 4] * uu2 * uu3 +
        gcov[4, 4] * uu3 * uu3)
    lapse  = 1 / sqrt(-gcon[1, 1])
    shift1 = -gcon[1, 2] / gcon[1, 1]
    shift2 = -gcon[1, 3] / gcon[1, 1]
    shift3 = -gcon[1, 4] / gcon[1, 1]
    ucon = SVector(uu0 / lapse, uu1 - shift1 * uu0 / lapse,
                   uu2 - shift2 * uu0 / lapse, uu3 - shift3 * uu0 / lapse)
    ucov = gcov * ucon
    b0 = ucov[2] * bb1 + ucov[3] * bb2 + ucov[4] * bb3
    bcon = SVector(b0, (bb1 + b0 * ucon[2]) / ucon[1],
                   (bb2 + b0 * ucon[3]) / ucon[1], (bb3 + b0 * ucon[4]) / ucon[1])
    bcov = gcov * bcon
    return ucon, ucov, bcon, dot(bcon, bcov)
end

# ---------------------------------------------------------------- device: coefficients
# Electron temperature Θ_e (electron_theta, ti_te_beta). The thermal_frac==0 corner is
# short-circuited in _gimg_thermal (no NaN reaches output); both use_p branches kept.
@inline function _gimg_theta_e(rho_cgs, pgas_cgs, beta_inv, gp)
    tti_tte = (gp.rat_high + gp.rat_low * beta_inv * beta_inv) / (1.0 + beta_inv * beta_inv)
    kbtt = gp.mu * _MP * pgas_cgs / rho_cgs
    if gp.use_p != 0
        kbtte = (1.0 + gp.ne_ni) / (tti_tte + gp.ne_ni) * kbtt
    else
        kbtte = (1.0 + gp.ne_ni) * kbtt / (gp.gamma - 1.0)
        kbtte /= tti_tte / (gp.gamma_i - 1.0) + gp.ne_ni / (gp.gamma_e - 1.0)
    end
    return kbtte / (_ME * _CL * _CL)
end

# Unpolarized thermal synchrotron (thermal_synchrotron) -> (j_inv, α_inv). `thermal_frac`
# scales the coefficient exactly as the CPU `thermal_synchrotron` (thermal_frac==0 -> no emission).
# Number-type-generic (typed zero on the early exits), so the SAME body serves the plain
# Float64 kernels bit-for-bit AND the Dual accumulation of gpu_image_ad.jl.
@inline function _gimg_thermal(thermal_frac, n_e, theta_e, bb, nu, sin_tb)
    z = zero(promote_type(typeof(n_e), typeof(theta_e), typeof(bb)))
    (thermal_frac == 0.0 || n_e <= 0.0 || theta_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (z, z)
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    nu_s = 2.0 / 9.0 * nu_c * theta_e * theta_e * sin_tb
    nu_2 = nu * nu
    xx = nu / nu_s
    xx12 = sqrt(xx); xx13 = cbrt(xx); xx16 = sqrt(xx13)
    coeff = thermal_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2) * exp(-xx13)
    var_a = _SQRT2 * π / 27.0 * sin_tb
    var_b = 2.0^(11.0 / 12.0)
    var_c = xx12 + var_b * xx16
    j_inv = coeff * var_a * var_c * var_c
    kbtte = theta_e * _ME * _CL * _CL
    bnn3 = 2.0 * _HPL / (_CL * _CL) / expm1(_HPL * nu / kbtte)
    alpha_inv = j_inv / bnn3
    isfinite(1.0 / (alpha_inv * alpha_inv)) || (alpha_inv = z)
    return (j_inv, alpha_inv)
end

# Unpolarized power-law synchrotron (power_law_synchrotron) -> (j_inv, α_inv), the
# additive non-thermal term when power_frac != 0. Reads the host-precomputed constants
# from the params tuple (_powerlaw_gp), so every per-step formula is bit-for-bit the CPU
# body. Number-type-generic (typed zero on the early exit), hence Dual-safe.
@inline function _gimg_powerlaw(gp, n_e, bb, nu, sin_tb)
    z = zero(promote_type(typeof(n_e), typeof(bb), typeof(nu), typeof(sin_tb)))
    (gp.power_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (z, z)
    p = gp.power_p
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_tb)
    j_inv = gp.power_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2) *
            gp.power_jj * sin_tb * xx^(-(p - 1.0) / 2.0)
    alpha_inv = gp.power_frac * n_e * _EE * _EE / (_ME * _CL) * gp.power_aa *
                xx^(-(p + 2.0) / 2.0)
    return (j_inv, alpha_inv)
end

# Harmonic-mean synchrotron bridge (coefficients.jl :: harmonic_bridge), device mirror.
# (a^-x + b^-x)^(-1/x); IEEE-safe (an underflowed branch -> pow(tiny,-x)=inf -> the other
# asymptote). Same arithmetic tree as the CPU form so the kappa mixers are bit-for-bit.
@inline _kappa_bridge(a, b, x) = (a^(-x) + b^(-x))^(-1.0 / x)

# Unpolarized kappa synchrotron (kappa_synchrotron) -> (j_inv, α_inv), the additive
# non-thermal term when kappa_frac != 0. Reads the host-precomputed constants from the
# params tuple (_kappa_gp; the kappa-index fit branch is resolved host-side in
# precompute_kappa!), so every per-step formula is bit-for-bit the CPU body.
# Number-type-generic (typed zero on the early exit), hence Dual-safe.
@inline function _gimg_kappa(gp, n_e, bb, nu, sin_tb)
    z = zero(promote_type(typeof(n_e), typeof(bb), typeof(nu), typeof(sin_tb)))
    (gp.kappa_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (z, z)
    k = gp.kappa
    w = gp.w
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    nu_2 = nu * nu
    nu_kappa = nu_c * w * w * k * k * sin_tb
    xx = nu / nu_kappa

    aa = gp.kappa_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2)
    j_low = gp.kappa_jj_low * aa * cbrt(xx) * sin_tb
    j_high = gp.kappa_jj_high * aa * xx^(-(k - 2.0) / 2.0) * sin_tb
    j_inv = _kappa_bridge(j_low, j_high, gp.kappa_jj_x_i)

    bb_ = gp.kappa_frac * n_e * _EE * _EE / (_ME * _CL)
    a_low = gp.kappa_aa_low * bb_ * xx^(-2.0 / 3.0)
    a_high = gp.kappa_aa_high * bb_ * xx^(-(1.0 + k) / 2.0) * gp.kappa_aa_high_i
    alpha_inv = _kappa_bridge(a_low, a_high, gp.kappa_aa_x_i)
    return (j_inv, alpha_inv)
end

# One sample -> (j_inv, α_inv). Mirrors the unpolarized branch of eight_coefficients:
# cuts (σ, Θ_e), invariant thermal coefficients. Invalid / cut / vacuum samples return
# (0,0), which the propagator treats as a no-op (A=1,B=0) — exactly the CPU march's
# `continue`. The zero is typed by the (possibly Dual-seeded) unit/plasma scalars, so
# the body is Dual-type-stable for the AD render.
# NOTE(bit-exactness): the fluid-decode/cut front half is deliberately NOT hoisted into
# a helper shared with _gpol_coeffs — restructuring this body changes LLVM's fmuladd
# contraction choices (~1 ulp, amplified by exp(-xx^{1/3}) on dim samples), breaking the
# pinned regression outputs; this exact body reproduces the CPU path sample-for-sample.
@inline function _gimg_coeffs(prim::SVector{8,Float64}, x::SVector{4,Float64},
                              kcov::SVector{4,Float64}, gp)
    z = zero(promote_type(typeof(gp.d_unit), typeof(gp.e_unit), typeof(gp.b_unit),
                          typeof(gp.rat_high), typeof(gp.rat_low)))
    rho = prim[1]; eint = prim[5]
    (rho <= 0.0 || eint <= 0.0) && return (z, z)
    pgas = (gp.gamma - 1.0) * eint
    gcov, gcon = _gimg_metric(x, gp.bh_a)
    ucon, ucov, bcon, b_sq = _gimg_pfv(prim[2], prim[3], prim[4],
                                       prim[6], prim[7], prim[8], gcov, gcon)
    b_sq <= 0.0 && return (z, z)
    rho_cgs = rho * gp.d_unit
    pgas_cgs = pgas * gp.e_unit
    n_cgs = rho_cgs / (gp.mu * _MP)
    n_e = n_cgs / (1.0 + 1.0 / gp.ne_ni)
    bb_cgs = sqrt(b_sq) * gp.b_unit
    sigma = b_sq / rho
    beta_inv = b_sq / (2.0 * pgas)
    theta_e = _gimg_theta_e(rho_cgs, pgas_cgs, beta_inv, gp)
    (gp.sigma_cut >= 0.0 && sigma > gp.sigma_cut) && return (z, z)
    (gp.theta_e_min >= 0.0 && theta_e < gp.theta_e_min) && return (z, z)
    minus_ku = -dot(kcov, ucon)                 # = ν_local / ν_obs (k_t = -1)
    kb = dot(kcov, bcon)
    cos2 = min(kb * kb / (minus_ku * minus_ku * b_sq), 1.0)
    sin_tb = sqrt(1.0 - cos2)
    nu_local = minus_ku * gp.nu_obs
    th = _gimg_thermal(gp.thermal_frac, n_e, theta_e, bb_cgs, nu_local, sin_tb)
    # Additive nonthermal eDF mixing in the CPU eight_coefficients order: thermal, then
    # power-law, then kappa. Each branch is keyed on the params tuple carrying that eDF's
    # constants (`hasfield` is compile-time), so a thermal-only deck compiles to the
    # untouched thermal path — byte-identical output and register count.
    j = th[1]; a = th[2]
    if hasfield(typeof(gp), :power_frac)
        jp, ap = _gimg_powerlaw(gp, n_e, bb_cgs, nu_local, sin_tb)
        j += jp; a += ap
    end
    if hasfield(typeof(gp), :kappa_frac)
        jk, ak = _gimg_kappa(gp, n_e, bb_cgs, nu_local, sin_tb)
        j += jk; a += ak
    end
    return (j, a)
end

# Affine propagator factors (A, B) for one segment (couple_unpolarized): I -> A*I + B.
# Type-generic (Dual-safe) like _gimg_thermal.
@inline function _gimg_ab(j, alpha, ds)
    T = promote_type(typeof(j), typeof(alpha))
    if alpha > 0.0
        ss = j / alpha
        dtau = alpha * ds
        if dtau <= 100.0                        # POL_DELTA_TAU_MAX
            e = exp(-dtau)
            return (e, e * ss * expm1(dtau))
        end
        return (zero(T), ss)                    # saturated: I -> j/α
    end
    return (one(T), j * ds)                     # α ≤ 0: emission only
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
# Shared termination rule (trace_rays_sampled mirror): capture/NaN stop always; the
# escape/fluid-box stop fires only when OUTBOUND (r > r_prev) — an inbound ray beyond
# `escape` is not killed. Type-pure in T (T(1e-3) === 1e-3 for Float64, so the fp64
# path is bit-identical); matched-type scalars guard against silent promotion.
@inline _gimg_stop(r::T, r_prev::T, r_h::T, escape::T, R_em::T) where {T} =
    (r < r_h + T(1e-3) || !isfinite(r)) || ((r > escape || r > R_em) && r > r_prev)

# One fixed-control geodesic step: low-storage streaming RK4 — same ops/associativity
# as (k1+2k2+2k3+k4)/6 but the four stage derivatives are not all held live -> fewer
# registers/thread. Shared by every unpolarized march and the stiffness telemetry kernel.
# Runs entirely in T = eltype(y), the geodesic lane.
@inline function _gimg_rk4_step(y::SVector{8,T}, h::T, a::T) where {T}
    d = _gimg_rhs(y, a);  acc = d;         ys = y + (h / 2) * d
    d = _gimg_rhs(ys, a); acc = acc + 2*d; ys = y + (h / 2) * d
    d = _gimg_rhs(ys, a); acc = acc + 2*d; ys = y + h * d
    d = _gimg_rhs(ys, a); acc = acc + d
    return y + (h / 6) * acc
end

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
        r = _gimg_rks(y, a)
        _gimg_stop(r, r_prev, r_h, escape, R_em) && break
        r_prev = r
        h = (r - r_h) / C
        prim = @inline sample(y)   # force-inline the sampler (see the _gpol_march note)
        if prim[1] > 0.0
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
            j_inv, a_inv = _gimg_coeffs(prim, x, kcov, gp)
            ds = Float64(h) * gp.L_unit / gp.nu_obs
            A, B = _gimg_ab(j_inv, a_inv, ds)
            Iinv += trans * B
            trans *= A
        end
        y = _gimg_rk4_step(y, h, a)
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

# ---------------------------------------------------------------- device selection
if !@isdefined(select_devices)
"""
    select_devices(candidates=[0,1]; util_threshold=15, min_free_gb=8) -> Vector{Int}

Probe each visible CUDA device in `candidates` (correct under `CUDA_VISIBLE_DEVICES`
masking) and return the idle subset: free memory ≥ `min_free_gb` GB and, when NVML is
available, compute utilization ≤ `util_threshold` %. Always returns at least one device;
any probe failure degrades safely to the caller's static candidate list.
"""
function select_devices(candidates = [0, 1]; util_threshold::Real = 15, min_free_gb::Real = 8)
    vis = try
        filter(d -> 0 <= d < CUDA.ndevices(), candidates)
    catch
        return collect(candidates)
    end
    isempty(vis) && return collect(candidates)
    idle = Int[]
    for d in vis
        ok = try
            CUDA.device!(d)
            free, _ = CUDA.memory_info()
            # NVML compute util is a fraction in [0,1] (0.26 == 26%); -1 => unknown (ignore).
            u = try
                Float64(CUDA.NVML.utilization_rates(CUDA.NVML.Device(CUDA.uuid(CUDA.CuDevice(d)))).compute) * 100.0
            catch
                -1.0
            end
            (free / 1024^3 >= min_free_gb) && (u < 0 || u <= util_threshold)
        catch
            false
        end
        ok && push!(idle, d)
    end
    return isempty(idle) ? vis[1:1] : idle
end
end # @isdefined select_devices

# ---------------------------------------------------------------- host: ray reorder + launch
# Optional host-side ray reordering: `order` (a permutation of 1:nrays) groups rays by
# predicted march trip count so a warp's lanes march similar-length rays (warp-tail
# recovery). Each ray is independent and deterministic, so the gather-in / scatter-out
# is bit-exact per ray; `order === nothing` is the untouched default.
function _apply_order(v, order)
    order === nothing && return v
    (length(order) == length(v) && isperm(order)) ||
        error("render `order` must be a permutation of 1:$(length(v))")
    return @inbounds v[order]
end
function _undo_order(v, order)
    order === nothing && return v
    w = similar(v); @inbounds w[order] = v; return w
end

"""
    march_sort_order(counts) -> Vector{Int}

Permutation sorting rays by ascending march trip count (`counts[i]` = steps ray `i`
marched). Pass it as the `order` kwarg of the `render_gpu_*` drivers to group similar-
length rays so each warp's lanes finish together. Note: on the fused native-AMR path,
camera-order rays are spatially coherent and trip-count sorting scatters that sampling
locality — leave `order = nothing` there unless a fixed-camera movie loop amortizes a
logged-frame permutation (the reorder is bit-exact per pixel either way).
"""
march_sort_order(counts::AbstractVector) = sortperm(counts)

# KA launch with an OPTIONAL workgroupsize. Results are invariant to the block size (one
# independent ray per thread, no cross-lane reductions), so this tunes only occupancy;
# `wg === nothing` reproduces the backend-default launch.
@inline _kern_launch!(kern, wg, cnt, args...) =
    wg === nothing ? kern(args...; ndrange = cnt) :
                     kern(args...; ndrange = cnt, workgroupsize = min(wg, cnt))

# ---------------------------------------------------------------- host driver
# Host precompute of the power-law eDF constants: runs precompute_power_law! once (the
# gamma/₂F₁ host math) and returns the isbits Float64 scalars the device _gimg_powerlaw /
# _gpol_powerlaw mixers consume — bit-identical to the CPU path because both read the
# same precomputed scalars. Polarized-only fields are carried but unused unpolarized.
function _powerlaw_gp(p)
    scc = SimCoeffConfig(; plasma_p = p.power_p, plasma_power_frac = p.power_frac)
    precompute_power_law!(scc, p.gamma_min, p.gamma_max)
    return (power_frac = p.power_frac, power_p = p.power_p, gamma_min = scc.plasma_gamma_min,
            power_jj = scc.power_jj, power_aa = scc.power_aa,
            power_jj_q = scc.power_jj_q, power_jj_v = scc.power_jj_v,
            power_aa_q = scc.power_aa_q, power_aa_v = scc.power_aa_v,
            power_rho = scc.power_rho, power_rho_q = scc.power_rho_q,
            power_rho_v = scc.power_rho_v)
end

# Host precompute of the kappa eDF constants: runs precompute_kappa! once (gamma/besselk/
# ₂F₁ host math AND the kappa-index three-way fit branch, so the device only reads the
# resolved scalars) and returns the isbits Float64 set the device _gimg_kappa /
# _gpol_kappa mixers consume — bit-identical to the CPU path.
function _kappa_gp(p)
    scc = SimCoeffConfig(; plasma_kappa = p.kappa, plasma_w = p.w, plasma_kappa_frac = p.kappa_frac)
    precompute_kappa!(scc)
    return (kappa_frac = p.kappa_frac, kappa = p.kappa, w = p.w,
            kappa_jj_low = scc.kappa_jj_low, kappa_jj_high = scc.kappa_jj_high,
            kappa_jj_x_i = scc.kappa_jj_x_i,
            kappa_aa_low = scc.kappa_aa_low, kappa_aa_high = scc.kappa_aa_high,
            kappa_aa_high_i = scc.kappa_aa_high_i, kappa_aa_x_i = scc.kappa_aa_x_i,
            kappa_jj_low_q = scc.kappa_jj_low_q, kappa_jj_low_v = scc.kappa_jj_low_v,
            kappa_jj_high_q = scc.kappa_jj_high_q, kappa_jj_high_v = scc.kappa_jj_high_v,
            kappa_jj_x_q = scc.kappa_jj_x_q, kappa_jj_x_v = scc.kappa_jj_x_v,
            kappa_aa_low_q = scc.kappa_aa_low_q, kappa_aa_low_v = scc.kappa_aa_low_v,
            kappa_aa_high_q = scc.kappa_aa_high_q, kappa_aa_high_v = scc.kappa_aa_high_v,
            kappa_aa_x_q = scc.kappa_aa_x_q, kappa_aa_x_v = scc.kappa_aa_x_v,
            kappa_rho_v = scc.kappa_rho_v, kappa_rho_frac = scc.kappa_rho_frac,
            kappa_rho_q_low_a = scc.kappa_rho_q_low_a, kappa_rho_q_low_b = scc.kappa_rho_q_low_b,
            kappa_rho_q_low_c = scc.kappa_rho_q_low_c, kappa_rho_q_low_d = scc.kappa_rho_q_low_d,
            kappa_rho_q_low_e = scc.kappa_rho_q_low_e,
            kappa_rho_q_high_a = scc.kappa_rho_q_high_a, kappa_rho_q_high_b = scc.kappa_rho_q_high_b,
            kappa_rho_q_high_c = scc.kappa_rho_q_high_c, kappa_rho_q_high_d = scc.kappa_rho_q_high_d,
            kappa_rho_q_high_e = scc.kappa_rho_q_high_e,
            kappa_rho_v_low_a = scc.kappa_rho_v_low_a, kappa_rho_v_low_b = scc.kappa_rho_v_low_b,
            kappa_rho_v_high_a = scc.kappa_rho_v_high_a, kappa_rho_v_high_b = scc.kappa_rho_v_high_b)
end

# Merge the host-precomputed nonthermal constants into a device params `base` tuple. A
# pure-thermal deck (both fracs 0) returns `base` unchanged (byte-identical tuple shape
# -> untouched thermal kernel); a single-eDF deck carries only that eDF's fields, keyed
# in the coeff functions via compile-time `hasfield`.
_merge_nonthermal(base, p) = begin
    gp = p.power_frac == 0.0 ? base : merge(base, _powerlaw_gp(p))
    p.kappa_frac == 0.0 ? gp : merge(gp, _kappa_gp(p))
end

# Named tuple of the isbits scalars the kernel needs (units + plasma + geometry), with
# the host-precomputed nonthermal constants merged in when active (`_merge_nonthermal`)
# so the device coefficient functions mix eDFs exactly as CPU eight_coefficients.
function _gimg_params(cfg)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    p = cfg.plasma
    base = (d_unit = u.dens_unit, e_unit = u.dens_unit * _CL^2, b_unit = u.B_unit,
            mu = p.mu, ne_ni = p.ne_ni, use_p = (p.use_p ? 1 : 0),
            rat_low = p.rat_low, rat_high = p.rat_high, gamma = p.gamma,
            gamma_i = p.gamma_i, gamma_e = p.gamma_e, thermal_frac = p.thermal_frac,
            sigma_cut = p.sigma_cut, theta_e_min = p.theta_e_min,
            nu_obs = cfg.frequency_hz, L_unit = u.L_unit, bh_a = cfg.bh_a)
    return _merge_nonthermal(base, p)
end

# Render the ray subset `rng` (any index collection) on the CURRENTLY active device with
# device-resident grid. rays[rng] gathers, img[rng] .= scatters — exact for a strided rng.
# The geodesic-lane type is eltype(eltype(rays)); the caller passes the march scalars at
# that type. The accumulator stays Float64 (from the output eltype).
function _fused_part!(img, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C, escape, R_em, nmax, wg)
    backend = CUDABackend()
    kern = _fused_kernel!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{Float64}(undef, cnt)
    _kern_launch!(kern, wg, cnt, outd, y0d, grid_d, gp, a, r_h, C, escape, R_em,
                  Int(nmax), ng, xo, inv_dx, cnt)
    KernelAbstractions.synchronize(backend)
    img[rng] .= Array(outd)
    return nothing
end

# Launch the fused kernel over the ray-index `pool` (any index collection), pixels
# INTERLEAVED across `devices` (device di gets pool[di], pool[di+nd], …) so each GPU gets
# a statistically even mix of cheap-escape and expensive-torus rays; each part runs
# concurrently in an `@async`/`CUDA.device!` task. pool == 1:nrays reproduces the
# pre-mixed-precision device split exactly (pool[di:nd:end] === di:nd:nrays).
function _fused_sweep!(img, rays, pool, devices, dgrid, gp, a, r_h, C, escape, R_em, nmax, wg)
    nd = length(devices)
    @sync for di in 1:nd
        idx = pool[di:nd:end]; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _fused_part!(img, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                         dgrid.inv_dx, gp, a, r_h, C, escape, R_em, nmax, wg)
        end
    end
    return nothing
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
    θo = clamp(deg2rad(Float64(inc_deg)), 0.02, π - 0.02)
    αB, βB = critical_curve(Float64(a), θo)
    isempty(αB) && return Int[]
    npix = Int(npix)
    px = Float64(fov) / npix
    half = Float64(fov) / 2
    mask = falses(npix, npix)                 # [iβ, iα] — the image-matrix layout
    m = length(αB)
    for s in 1:m
        s2 = s == m ? 1 : s + 1               # closed loop (upper + mirrored lower branch)
        x1 = αB[s];  y1 = βB[s]               # camera screen coords (see docstring)
        x2 = αB[s2]; y2 = βB[s2]
        nsub = max(1, ceil(Int, hypot(x2 - x1, y2 - y1) / (px / 2)))
        for q in 0:nsub                       # walk the segment at ≤ px/2 spacing
            t = q / nsub
            iα = floor(Int, (x1 + t * (x2 - x1) + half) / px) + 1
            iβ = floor(Int, (y1 + t * (y2 - y1) + half) / px) + 1
            for dα in -rescue_px:rescue_px, dβ in -rescue_px:rescue_px
                jα = iα + dα; jβ = iβ + dβ
                (1 <= jα <= npix && 1 <= jβ <= npix) && (mask[jβ, jα] = true)
            end
        end
    end
    return findall(vec(mask))
end

"""
    render_gpu_fused(cfg, snap_or_grid; devices=select_devices(), nmax=cfg.max_steps,
                     C=80, box_half=40.0, n=256, precision=:f64, rescue_px=2)
        -> Matrix{Float64}

Fused single-pass GPU render of the npix×npix unpolarized image, in the layout of
`render(...;polarized=false).I`. `snap_or_grid`: `AthenaKSnapshot` (resampled + uploaded
internally), `UniformFluidGrid` (uploaded per call), or a pre-uploaded `DeviceFluidGrid`
(reused). Pixels are interleaved across `devices` for an even cheap/expensive-ray mix.
`order` marches rays in trip-count order (bit-exact per pixel); `wg` overrides the block
size (occupancy only). `precision`: `:f64` (default, bitwise unchanged) · `:mixed`
(fp32 geodesic lane, Float64 fluid-sample/RT; near-critical-curve pixels re-rendered in
fp64 and spliced — `rescued_frac::Ref{Float64}` receives the fraction) · `:f32_raw`
(no rescue, DIAGNOSTIC only — near-critical pixels carry O(1) errors).
"""
function render_gpu_fused(cfg, snap_or_grid; devices = select_devices(),
                          nmax::Int = cfg.max_steps,   # step limit wired from RunConfig, as render()
                          C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                          order = nothing, wg = nothing,
                          precision::Symbol = :f64, rescue_px::Int = 2,
                          rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_fused: precision must be :f64, :mixed or :f32_raw, got :$precision")
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)

    gp = _gimg_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = length(rays)
    img = Vector{Float64}(undef, nrays)
    frac = 0.0
    if precision === :f64
        _fused_sweep!(img, rays, 1:nrays, devices, dgrid, gp, a, r_h, Cf, escape, R_em,
                      nmax, wg)
    else
        # fp32 geodesic lane: narrow the fp64-built rays at upload; the host constructs
        # every geodesic-lane scalar as Float32 (no in-kernel widening), while gp
        # (units, frequency, fp64 spin gp.bh_a for the metric/redshift) stays Float64.
        rays32 = [SVector{8,Float32}(r) for r in rays]
        _fused_sweep!(img, rays32, 1:nrays, devices, dgrid, gp, Float32(a), Float32(r_h),
                      Float32(Cf), Float32(escape), Float32(R_em), nmax, wg)
        if precision === :mixed
            pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                                 rescue_px = rescue_px)
            # pixel -> ordered-ray position (img/rays live in `order`ed space until _undo_order)
            pos = order === nothing ? pix : invperm(order)[pix]
            isempty(pos) ||
                _fused_sweep!(img, rays, pos, devices, dgrid, gp, a, r_h, Cf, escape,
                              R_em, nmax, wg)
            frac = length(pos) / nrays
        end
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    return reshape(_undo_order(img, order), cam.npix, cam.npix)
end

# ================================================================ K-VARIANT batched RT
# mass_scale-ladder batched unpolarized render: one ray/thread marches the geodesic once
# and samples the fluid once per step (both mass_scale-independent) while accumulating K
# intensity variants. mass_scale enters the physics ONLY through units(m_bh, mass_scale)
# -> (d_unit, e_unit, b_unit); every other device scalar is shared across all K.
# Unlike the Dual-seeded AD render, an SVector-valued param lane cannot ride the generic
# kernel (SVector is not a Number: scalar guards, elementwise ratios and one(T) all
# break), so only the ACCUMULATOR is vectorized (SVector{K,Float64}) and each variant
# calls the pinned bit-exact _gimg_coeffs/_gimg_ab verbatim with its own units tuple —
# lane k is bitwise-identical to a standalone render of that mass_scale. The redundant
# per-variant geometry inside _gimg_coeffs is deliberately NOT hoisted (fmuladd pin);
# the expensive shared work (RK4 + trilinear sample) is what the march amortizes.
@inline function _gimg_march_kv(y0::SVector{8,Tg}, sample::F, gps::NTuple{K,GP},
                                a::Tg, r_h::Tg, C::Tg, escape::Tg, R_em::Tg,
                                nmax) where {F,K,GP,Tg}
    y = y0
    Iinv  = zero(SVector{K,Float64})
    trans = SVector{K,Float64}(ntuple(_ -> 1.0, Val(K)))
    r_prev = Tg(Inf)
    nu_obs = gps[1].nu_obs; L_unit = gps[1].L_unit    # mass_scale-independent (shared lane)
    for _ in 1:nmax
        r = _gimg_rks(y, a)
        _gimg_stop(r, r_prev, r_h, escape, R_em) && break
        r_prev = r
        h = (r - r_h) / C
        prim = @inline sample(y)
        if prim[1] > 0.0
            x    = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
            ds   = Float64(h) * L_unit / nu_obs
            # per-variant (A,B): the exact _gimg_coeffs/_gimg_ab ops with mass_scale k's
            # (d,e,b)_unit — bit-identical to a standalone render of variant k.
            ab = ntuple(Val(K)) do k
                j_inv, a_inv = _gimg_coeffs(prim, x, kcov, gps[k])
                _gimg_ab(j_inv, a_inv, ds)
            end
            A = SVector{K,Float64}(ntuple(k -> ab[k][1], Val(K)))
            B = SVector{K,Float64}(ntuple(k -> ab[k][2], Val(K)))
            Iinv  = Iinv + trans .* B          # per lane: Iinv[k] += trans[k]*B[k]
            trans = trans .* A                 # per lane: trans[k] *= A[k]
        end
        y = _gimg_rk4_step(y, h, a)
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

# Render ray subset `rng` on the CURRENTLY active device; SVector{K,Float64} output per pixel.
function _fused_part_kv!(img::Vector{SVector{K,Float64}}, rays, rng, grid_d, ng, xo, inv_dx,
                         gps, a, r_h, C, escape, R_em, nmax) where {K}
    backend = CUDABackend()
    kern = _fused_kernel_kv!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{SVector{K,Float64}}(undef, cnt)
    kern(outd, y0d, grid_d, gps, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx, cnt;
         ndrange = cnt)
    KernelAbstractions.synchronize(backend)
    img[rng] .= Array(outd)
    return nothing
end

# Interleaved multi-device sweep (device di gets rays di, di+nd, …), mirroring _fused_sweep!.
function _fused_sweep_kv!(img, rays, devices, dgrid, gps, a, r_h, C, escape, R_em, nmax)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:length(rays); dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _fused_part_kv!(img, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                            dgrid.inv_dx, gps, a, r_h, C, escape, R_em, nmax)
        end
    end
    return nothing
end

"""
    render_gpu_fused_kvariant(cfg, snap_or_grid, mass_scales::Vector{Float64};
                              devices=select_devices(), nmax=cfg.max_steps,
                              C=80, box_half=40.0, n=256) -> Vector{Matrix{Float64}}

K-variant batched unpolarized fused render: one GPU pass returns K images, one per
`mass_scales` value, sharing the geodesic march and fluid sample across all variants.
Image k is bitwise-identical to `render_gpu_fused` with `mass_scale_cgs = mass_scales[k]`
(mass_scale enters only via `units` -> d/e/b_unit). Stokes-I, uniform-resample fluid,
`:f64` only. `snap_or_grid` as in `render_gpu_fused`; the same device fluid buffer feeds
all K variants. Returns a length-K `Vector{Matrix{Float64}}` in `render_gpu_fused` layout.
"""
function render_gpu_fused_kvariant(cfg, snap_or_grid, mass_scales::Vector{Float64};
                                   devices = select_devices(), nmax::Int = cfg.max_steps,
                                   C::Real = 80, box_half::Real = 40.0, n::Integer = 256)
    K = length(mass_scales)
    K >= 1 || error("render_gpu_fused_kvariant: mass_scales must be nonempty")
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)

    gp0 = _gimg_params(cfg)
    # per-variant param tuples differ ONLY in (d_unit,e_unit,b_unit) — exactly what
    # _gimg_params(cfg_k) with mass_scale_cgs = mass_scales[k] produces (L_unit is
    # mass_scale-independent, so gp0.L_unit is shared); the merge preserves gp0's type.
    gps = ntuple(K) do k
        u = units(cfg.m_bh_cgs, mass_scales[k])
        merge(gp0, (d_unit = u.dens_unit, e_unit = u.dens_unit * _CL^2, b_unit = u.B_unit))
    end
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = length(rays)
    img = Vector{SVector{K,Float64}}(undef, nrays)
    _fused_sweep_kv!(img, rays, devices, dgrid, gps, a, r_h, Cf, escape, R_em, nmax)
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

function _fused_part_sl!(img, rays, rng, gA, gB, ng, xo, inv_dx, t0, inv_dt,
                         gp, a, r_h, C, escape, R_em, nmax, wg)
    backend = CUDABackend()
    kern = _fused_kernel_sl!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{Float64}(undef, cnt)
    _kern_launch!(kern, wg, cnt, outd, y0d, gA, gB, t0, inv_dt, gp, a, r_h, C, escape,
                  R_em, Int(nmax), ng, xo, inv_dx, cnt)
    KernelAbstractions.synchronize(backend)
    img[rng] .= Array(outd)
    return nothing
end

"""
    render_gpu_fused_slow(cfg, win::DeviceFluidWindow; devices=select_devices(),
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
function render_gpu_fused_slow(cfg, win::DeviceFluidWindow; devices = select_devices(),
                               nmax::Int = cfg.max_steps, C::Real = 80,
                               t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                               order = nothing, wg = nothing)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    ga = win.a; gb = win.b
    gp = _gimg_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(ga.xo)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    nrays = length(rays)
    img = Vector{Float64}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _fused_part_sl!(img, rays, idx, ga.dev_data[dev], gb.dev_data[dev], ga.n,
                            ga.xo, ga.inv_dx, t0, win.inv_dt, gp, a, r_h, Cf,
                            escape, R_em, nmax, wg)
        end
    end
    return reshape(_undo_order(img, order), cam.npix, cam.npix)
end
