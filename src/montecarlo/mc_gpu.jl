# mc_gpu.jl — device Monte Carlo radiative transfer (KernelAbstractions).
#
# Device port of the sphere Monte Carlo: superphoton `gid` draws from the Philox
# stream `mc_primary_rng(seed, gid)` (the CPU path's stream), generates a
# synchrotron packet, tracks it through the flat sphere, and bins it into an
# exact (theta, E) spectrum; a layered host loop adds Compton scattering (cf.
# gpumonty scattering.cu), each child drawing from the stream of its genealogy
# (`mc_child_rng`). Bins and diagnostics are exact integer-limb accumulators
# (exact_sum.jl) updated with KernelAbstractions (Atomix) atomics, so the
# output is bitwise independent of device count, launch geometry and atomic
# ordering, and the kernels run on any KA backend: CUDA ordinals, or
# `HostDevice`s on the KA CPU backend. Leaf samplers are the same pure/isbits
# CPU functions; the weight lookup and scatter march use the shared
# mc_common.jl bodies, so CPU and device cannot drift. Include after
# `using BlackLightPlus`.

using StaticArrays
using KernelAbstractions
using CUDA
using LinearAlgebra: transpose

using BlackLightPlus: MCRng, mc_uniform, sample_direction,
                    int_jnu_zone, j_nu, jmax, alpha_nu, emission_tetrad,
                    SphereModel, sphere_zone, WeightTable, build_weight_table,
                    zone_nsuperphotons, Spectrum, merge_spectra!,
                    HotcrossTable, scatter_superphoton, bias_func, ScatterResult
using BlackLightPlus.Constants: ME, CL, HPL

# Device handle seam (HostDevice, _device!, _backend, _upload) lives in gpu_image.jl.
@isdefined(_gimg_thermal) || include(joinpath(@__DIR__, "..", "gpu_image.jl"))

# CPU-side constants, referenced (not re-typed) so the kernels can never drift
# from mc_cpu.jl / track.jl / generate.jl / emission_tables.jl.
const _MCM = BlackLightPlus.MonteCarlo
const _XSLOT = _MCM.XSUM_NSLOT

# ---------------------------------------------------------------- multi-device split
# Device `di` of `ndev` launches the contiguous global primary ids
# `first:first+n-1`; the ranges tile `1:N_gen`. A photon's stream depends on its
# global id only, so the union of streams is the same for every `ndev`.
_mc_device_shares(N_gen::Integer, ndev::Integer) =
    [(first = div((di - 1) * N_gen, ndev) + 1,
      n = div(di * N_gen, ndev) - div((di - 1) * N_gen, ndev)) for di in 1:ndev]

# Device handles of a driver call: `devices` (CUDA ordinals or HostDevices) or `[device]`.
_mc_devlist(device, devices) = devices === nothing ? Any[device] : collect(Any, devices)

# Launch `kern` over `n` items with an optional workgroup size.
function _mc_launch(kern, wgs, n::Int, args...)
    if wgs === nothing
        kern(args...; ndrange = n)
    else
        kern(args...; ndrange = n, workgroupsize = min(Int(wgs), n))
    end
    return nothing
end

# ---------------------------------------------------------------- exact device accumulation
# Device twin of `xsum_add!`: at most three Int64 atomic adds into the limbs of
# cell (a, b) of `acc` (limbs along axis 1). Integer atomics commute, so the
# cell value does not depend on the order in which threads arrive.
@inline function _xsum_atomic_add!(acc, x::Float64, a::Int, b::Int)
    j, p1, p2, p3 = _MCM.xsum_parts(x)
    j == 0 && return nothing
    if j == _XSLOT
        KernelAbstractions.@atomic acc[j, a, b] += p1
    else
        p1 == 0 || (KernelAbstractions.@atomic acc[j, a, b] += p1)
        p2 == 0 || (KernelAbstractions.@atomic acc[j + 1, a, b] += p2)
        p3 == 0 || (KernelAbstractions.@atomic acc[j + 2, a, b] += p3)
    end
    return nothing
end

# ---------------------------------------------------------------- device selection
if !@isdefined(select_devices)
"""
    select_devices(candidates=[0,1]; util_threshold=15, min_free_gb=8) -> Vector{Int}

Probe each visible CUDA device in `candidates` and return the subset that is
idle enough: free memory >= `min_free_gb` GB and (when NVML is available)
compute utilization <= `util_threshold` %. Always returns at least one device.
Definition is guarded so this file stays self-contained alongside gpu_image.jl.
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

# ---------------------------------------------------------------- device absorption rate
# Device twin of track.jl `_nu_alpha` for the uniform flat sphere (static fluid,
# B along x1): the invariant absorption rate nu*alpha_nu at `x` for wavevector
# `kcon`, zero outside the sphere.
@inline function _mc_nu_alpha_dev(x::SVector{4}, kcon::SVector{4}, ne, te, B, R)
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    r2 <= R * R || return 0.0
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ks <= 0.0 && return 0.0
    cth = kcon[2] / ks
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    nu = kcon[1] * ME * CL * CL / HPL
    return nu * alpha_nu(ne, te, B, nu, sth)
end

# ---------------------------------------------------------------- device straight-line track
# Mirror of track.jl `track_superphoton` for the thin sphere: k^mu is null and
# constant (flat metric), so the geodesic is a straight line and the weight
# decays by the trapezoid-integrated absorption depth each step. Returns only
# the final weight (position/momentum are unchanged for the record).
@inline function _mc_track_thin_w(x0::SVector{4}, pcov::SVector{4}, w0,
                                  gcon::SMatrix{4,4}, ne, te, B, R,
                                  step_frac, track_max::Int)
    kcon = gcon * pcov
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    dlam = ks > 0.0 ? step_frac * R / ks : R
    Cabs = HPL / (ME * CL * CL)
    na0 = _mc_nu_alpha_dev(x0, kcon, ne, te, B, R)
    w = w0
    x = x0
    @inbounds for _ in 1:track_max
        xn = x .+ kcon .* dlam
        na1 = _mc_nu_alpha_dev(xn, kcon, ne, te, B, R)
        dtau = 0.5 * (na0 + na1) * Cabs * dlam
        w *= exp(-dtau)
        x = xn
        na0 = na1
        r2 = xn[2] * xn[2] + xn[3] * xn[3] + xn[4] * xn[4]
        r2 <= R * R || break
    end
    return w
end

# ---------------------------------------------------------------- generation (shared recipe)
# Frequency-rejection accept numerator, the device mirror of generate.jl
# `_gen_int_jnu`: exact `int_jnu_zone` when `lint === nothing` (default,
# selected by dispatch), or the uploaded `log int_jnu` table read through the
# shared interpolation body when `tabulated_gen` is opted in (`p` then carries
# the grid scalars `lint_dlnu`/`lint_n`; `lnu_min` is shared with the weight grid).
@inline _mc_gen_int_jnu(::Nothing, p, nu) = int_jnu_zone(p.ne, p.te, p.B, nu)
@inline _mc_gen_int_jnu(lint, p, nu) =
    _MCM._mc_weight_at(nu, lint, p.lnu_min, p.lint_dlnu, p.lint_n)

# Device generator, called by the thin and scatter kernels: uniform-ball
# emission point + frequency & pitch rejection -> covariant seed momentum,
# mirroring sphere_model.jl `sample_sphere_photon` + generate.jl
# `sample_zone_photon`. The emission tetrad `p.E` is host-precomputed once per
# run (constant for the uniform flat sphere). `lint` selects the opt-in
# tabulated frequency rejection (`nothing` = exact default; `p.dnmax` must then
# be the matching `tabulated_dnmax` envelope).
@inline function _mc_gen_sphere_photon(rng::MCRng, p, lwgt, lint)
    u, rng = mc_uniform(rng)
    r = p.R * cbrt(u)
    dir, rng = sample_direction(rng)
    x = SVector(0.0, r * dir[1], r * dir[2], r * dir[3])

    nu = 0.0
    w = 0.0
    for _ in 1:p.reject_max
        u1, rng = mc_uniform(rng)
        nu = exp(u1 * (p.lnu_max - p.lnu_min) + p.lnu_min)
        w = _MCM._mc_weight_at(nu, lwgt, p.lnu_min, p.dlnu, p.n_nodes)
        u2, rng = mc_uniform(rng)
        accept = (_mc_gen_int_jnu(lint, p, nu) / (w + p.weight_floor)) / p.dnmax
        u2 <= accept && break
    end

    jm = jmax(p.ne, p.te, p.B, nu)
    cth = 0.0
    for _ in 1:p.reject_max
        u3, rng = mc_uniform(rng)
        cth = 2.0 * u3 - 1.0
        sth0 = sqrt(max(0.0, 1.0 - cth * cth))
        u4, rng = mc_uniform(rng)
        u4 <= j_nu(p.ne, p.te, p.B, nu, sth0) / jm && break
    end
    u5, rng = mc_uniform(rng)
    phi = 2.0 * π * u5
    sth = sqrt(max(0.0, 1.0 - cth * cth))

    eps = nu * HPL / (ME * CL * CL)
    k_tet = SVector(eps, eps * cth, eps * sth * cos(phi), eps * sth * sin(phi))
    kcon = transpose(p.E) * k_tet
    pcov = p.gcov * kcon
    return x, pcov, w, rng
end

# ---------------------------------------------------------------- record one escapee
# Bin (theta, E) of an escaped photon with covariant momentum `pcov` and weight
# `w` into exact device bins — the device twin of spectrum.jl `record!` on the
# spectrum grid in `p`.
@inline function _mc_record_dev!(xw, xwE, cnt, pcov::SVector{4}, w, p)
    Ee = -pcov[1]
    if Ee > 0.0
        nu_out = Ee * ME * CL * CL / HPL
        ie = unsafe_trunc(Int, floor((log(nu_out) - p.spec_lnu_min) / p.spec_dlnu)) + 1
        if 1 <= ie <= p.n_ebins
            ps = sqrt(pcov[2] * pcov[2] + pcov[3] * pcov[3] + pcov[4] * pcov[4])
            mu = ps > 0.0 ? abs(pcov[4]) / ps : 0.0     # |cos theta| about z, folded
            thf = acos(clamp(mu, 0.0, 1.0))
            ith = clamp(unsafe_trunc(Int, floor(thf / p.dth)) + 1, 1, p.n_thbins)
            KernelAbstractions.@atomic cnt[ith, ie] += Int64(1)
            _xsum_atomic_add!(xw, w, ith, ie)
            _xsum_atomic_add!(xwE, w * Ee, ith, ie)
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the per-superphoton kernel
"""
    _mc_thin_kernel!(xw, xwE, cnt, lwgt, lint, p, gid0, N)

One superphoton per index `i in 1:N`, global id `gid = gid0 + i`: seeds
`mc_primary_rng(p.seed, gid)` (the CPU path's stream for photon `gid`),
generates a synchrotron packet (`_mc_gen_sphere_photon`), tracks it straight to
escape with absorption-weight decay, and bins `(count, w, w*E)` exactly by exit
angle + energy. `p` is the isbits parameter bundle; `lwgt` the uploaded weight
table; `lint` the optional tabulated-generation table (`nothing` = exact default).
"""
@kernel function _mc_thin_kernel!(xw, xwE, cnt, @Const(lwgt), lint, p, @Const(gid0), @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        rng = _MCM.mc_primary_rng(p.seed, gid0 + i)
        x, pcov, w, _ = _mc_gen_sphere_photon(rng, p, lwgt, lint)
        w = _mc_track_thin_w(x, pcov, w, p.gcon, p.ne, p.te, p.B, p.R,
                             p.step_frac, p.track_max)
        _mc_record_dev!(xw, xwE, cnt, pcov, w, p)
    end
end

# ---------------------------------------------------------------- host tables (built once)
# One-zone weight table, launch count and generation envelope of the sphere
# run — identical to `mc_spectrum` / `mc_spectrum_scatter` — plus the
# position-independent emission tetrad. Built once per driver call and shared
# read-only by every device.
function _mc_sphere_tables(model::SphereModel, Ns::Real, nu_min::Real, nu_max::Real,
                           n_esamp::Integer, tabulated_gen::Bool)
    band = log(nu_max) - log(nu_min)
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Float64(Ns) / band;
                             n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    N_gen = max(1, round(Int, N_target))
    gt = tabulated_gen ? _MCM.build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = _MCM.tabulated_dnmax(gt, wgt))
    Etet = emission_tetrad(model.ucon, model.ucov, model.bcon, model.gcov, model.gcon)
    return (wgt = wgt, N_gen = N_gen, dnmax = dnmax, gt = gt, Etet = Etet)
end

# isbits kernel parameters of the thin sphere run; `grid` carries the spectrum grid.
function _mc_thin_params(model::SphereModel, tabs, grid, seed::Integer)
    wgt = tabs.wgt
    p = (seed = _mc_seed_key(seed),
         ne = model.n_e, te = model.theta_e, B = model.B, R = model.R,
         lnu_min = wgt.lnu_min, lnu_max = wgt.lnu_max, dlnu = wgt.dlnu,
         n_nodes = length(wgt.lwgt), dnmax = tabs.dnmax,
         weight_floor = _MCM._WEIGHT_FLOOR, reject_max = _MCM._MC_REJECT_MAX_ITER,
         spec_lnu_min = grid.lnu_min, spec_dlnu = grid.dlnu,
         n_ebins = Int(grid.n_ebins), n_thbins = Int(grid.n_thbins),
         dth = (0.5 * π) / grid.n_thbins,
         step_frac = _MCM._MC_STEP_FRAC, track_max = _MCM._MC_TRACK_MAX_STEPS,
         E = tabs.Etet, gcov = model.gcov, gcon = model.gcon)
    gt = tabs.gt
    return gt === nothing ? p : merge(p, (lint_dlnu = gt.dlnu, lint_n = length(gt.ljnu)))
end

# 32-bit Philox key word of a global seed.
_mc_seed_key(seed::Integer) = UInt32(Int(seed) & 0xffffffff)

# Exact device bins of one spectrum: count + limbs of Sum w and Sum w E.
_mc_dev_bins(backend, nth::Int, neb::Int) =
    (cnt = KernelAbstractions.zeros(backend, Int64, nth, neb),
     w = KernelAbstractions.zeros(backend, Int64, _XSLOT, nth, neb),
     wE = KernelAbstractions.zeros(backend, Int64, _XSLOT, nth, neb))

# Download device bins and add them into the host `ExactSpectrum` (integer adds).
function _mc_add_bins!(xs, bins)
    xs.count .+= Array(bins.cnt)
    xs.sumw .+= Array(bins.w)
    xs.sumwE .+= Array(bins.wE)
    return xs
end

# ---------------------------------------------------------------- host driver
"""
    mc_spectrum_gpu(model::SphereModel, Ns; device=0, devices=nothing, workgroupsize=nothing,
                    seed=0x5eed5eed, nu_min=1e8, nu_max=1e16, n_esamp=400,
                    n_ebins=200, n_thbins=6) -> Spectrum

Device thin-sphere Monte Carlo, the twin of `mc_spectrum` (mc_cpu.jl): builds
the same one-zone weight table and target count `N_gen` once on the host, so
packet count and Philox streams match the CPU path photon-for-photon, launches
`_mc_thin_kernel!` and rounds the exact bins into the returned `Spectrum`.
`devices` (CUDA ordinals or `HostDevice`s) splits the global ids `1:N_gen` into
contiguous ranges, one per device; the output is bitwise independent of the
device count and equals the single-device run.
"""
function mc_spectrum_gpu(model::SphereModel, Ns::Real;
                         device = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5eed5eed,
                         nu_min::Real = _MCM._MC_NU_MIN_DEFAULT,
                         nu_max::Real = _MCM._MC_NU_MAX_DEFAULT,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 200, n_thbins::Integer = 6,
                         tabulated_gen::Bool = false)
    devs = _mc_devlist(device, devices)
    tabs = _mc_sphere_tables(model, Ns, nu_min, nu_max, n_esamp, tabulated_gen)
    xs = _MCM.ExactSpectrum(; nu_min = nu_min, nu_max = nu_max,
                            n_ebins = n_ebins, n_thbins = n_thbins)
    p = _mc_thin_params(model, tabs, xs, seed)
    shares = _mc_device_shares(tabs.N_gen, length(devs))
    parts = Vector{Any}(undef, length(devs))
    @sync for di in eachindex(devs)
        @async parts[di] = _mc_thin_part(devs[di], tabs, p, shares[di],
                                         Int(n_thbins), Int(n_ebins), workgroupsize)
    end
    for part in parts
        part === nothing || _mc_add_bins!(xs, part)
    end
    return Spectrum(xs)
end

# One device's share of the thin run; returns its device bins (or `nothing`).
function _mc_thin_part(dev, tabs, p, share, nth::Int, neb::Int, wgs)
    share.n > 0 || return nothing
    _device!(dev)
    backend = _backend(dev)
    lwgt_d = _upload(backend, tabs.wgt.lwgt)
    lint_d = tabs.gt === nothing ? nothing : _upload(backend, tabs.gt.ljnu)
    bins = _mc_dev_bins(backend, nth, neb)
    _mc_launch(_mc_thin_kernel!(backend), wgs, share.n,
               bins.w, bins.wE, bins.cnt, lwgt_d, lint_d, p, share.first - 1, share.n)
    KernelAbstractions.synchronize(backend)
    return bins
end

# ======================================================================================
# DEVICE COMPTON SCATTERING (idealized Comptonization sphere)
#
# Layered, not recursive (after gpumonty scattering.cu `scattering_flow_control`):
# a host loop over scattering layers. The primary kernel marches the seed
# photons with the biased exponential-transform scatter trigger; on a trigger it
# appends the Compton-scattered child, with its genealogy stream
# `mc_child_rng(parent, spawn ordinal)`, to a device SoA buffer at a slot taken
# from an atomic counter, and continues the depleted incident beam; escaped
# photons are binned into the per-order spectrum. The host then launches the
# same march kernel on each layer's buffered children for orders 1..max_scatt.
# The slot only places a child in the buffer: its stream, and hence everything
# it does, is fixed by its genealogy. A launch whose children overflow the
# buffer is rerun with a buffer sized to the exact child count (the rerun is
# identical, the streams being genealogy-keyed), so no child is dropped.
# Unbiasedness: each child carries w(1-e^{-dtau_s})/(1-e^{-b dtau_s}) at spawn
# probability 1-exp(-b*dtau_s), so E[spawned weight] equals the physical scatter
# loss; the device accumulates both totals exactly and their ratio must be ~1
# independent of the bias. The electron / Klein-Nishina rejection loops run
# in-kernel (pure, isbits, capped); the scatter sampler sits behind a
# `@noinline` boundary so it does not inflate the march's register footprint.

# Diagnostic-accumulator stripe count: photon `i` adds its totals to stripe
# 1 + (i-1) % _MC_DIAG_STRIPE, which spreads the atomics over 64 addresses; the
# stripes are summed exactly on the host.
const _MC_DIAG_STRIPE = 64

# ---------------------------------------------------------------- device scatter opacity
# Device twin of track.jl `_nu_alpha_scatt` reading the uploaded hot-cross table `logsig`
# (a plain `CuArray`): the invariant scattering rate nu*alpha_s = nu*n_e*sigma_h(eps,Te) at
# `x` for wavevector `kcon`, zero outside the sphere. eps = HPL*nu/(ME CL^2) = kcon[1].
@inline function _mc_nu_alpha_scatt_dev(x::SVector{4}, kcon::SVector{4}, logsig, ne, te, R)
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    r2 <= R * R || return 0.0
    nu = kcon[1] * ME * CL * CL / HPL
    eps = HPL * nu / (ME * CL * CL)                      # == kcon[1]
    return nu * ne * _MCM._hotcross_lookup(logsig, eps, te)
end

# ---------------------------------------------------------------- the scatter kick (split)
# The full Compton sampler — electron draw + KN/Thomson rejection loops — behind
# a `@noinline` boundary so its registers/divergence stay out of the lean march.
# Reuses `scatter_superphoton` verbatim (emission tetrad rebuilt in-kernel; the
# spatial axes cancel).
@noinline function _mc_scatter_mom(pcov::SVector{4}, ucon::SVector{4}, ucov::SVector{4},
                                   bcon::SVector{4}, te,
                                   gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, rng::MCRng)
    p_new, _, rng = scatter_superphoton(pcov, ucon, ucov, bcon, te, gcov, gcon, rng)
    return p_new, rng
end

# ---------------------------------------------------------------- the biased scatter march
# Transport one superphoton straight through the sphere with the biased
# exponential-transform scatter decision: the march is the shared
# `_MCM._mc_march_scatter_core` (mc_common.jl); this wrapper supplies the device
# opacity lookup (uploaded hot-cross table) and the SoA child sink
# (`buf = (x, p, w, n, r)` + slot counter `ctr`, capacity `cap`). A child's
# stream is `mc_child_rng(srng, event, p.layout)`: `srng` is this photon's
# stream at any draw count, and only its identity words enter.
# Absorption is neglected (idealized Comptonization sphere), matching the CPU
# path. Returns (recorded, w_final, scatter_loss, child_w).
@inline function _mc_march_scatter!(x0::SVector{4}, pcov::SVector{4}, w0, nscatt::Int32,
                                    rng::MCRng, p, logsig, buf, ctr, cap::Int)
    nu_alpha_s = (x, kcon) -> _mc_nu_alpha_scatt_dev(x, kcon, logsig, p.ne, p.te, p.R)
    spawn! = (xev, weight_scat, tau_s, srng, event) -> begin
        p_new, srng = _mc_scatter_mom(pcov, p.ucon, p.ucov, p.bcon, p.te,
                                      p.gcov, p.gcon, srng)
        slot = (KernelAbstractions.@atomic ctr[1] += Int64(1)) - Int64(1)   # 0-based
        if slot < cap
            j = Int(slot) + 1
            @inbounds begin
                buf.x[j] = xev
                buf.p[j] = p_new
                buf.w[j] = weight_scat
                buf.n[j] = nscatt + Int32(1)
                buf.r[j] = _MCM.mc_child_rng(srng, event, p.layout)
            end
        end
        return weight_scat, srng     # an overflowing launch is rerun, never kept
    end
    recorded, _, w, _, sloss, child_w, _ =
        _MCM._mc_march_scatter_core(x0, pcov, w0, 0.0, nscatt, rng, p,
                                    nu_alpha_s, spawn!)
    return (recorded, w, sloss, child_w)
end

# ---------------------------------------------------------------- layer-0 (primary) kernel
"""
    _mc_scatter_primary_kernel!(xw, xwE, cnt, buf, ctr, cap, diag, lwgt, lint, logsig,
                                p, gid0, N)

One seed superphoton per index `i in 1:N`, global id `gid = gid0 + i`: generate
from the sphere (stream `mc_primary_rng(p.seed, gid)`), march with the biased
scatter trigger (appending order-1 children to `buf`), record the escaped
photon into the order-0 spectrum. Per-photon totals (order-1 child weight /
scatter loss / primary weight) go exactly into the striped `diag`.
"""
@kernel function _mc_scatter_primary_kernel!(xw, xwE, cnt, buf, ctr, @Const(cap), diag,
                                             @Const(lwgt), lint, @Const(logsig), p,
                                             @Const(gid0), @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        rng = _MCM.mc_primary_rng(p.seed, gid0 + i)
        x, pcov, w0, rng = _mc_gen_sphere_photon(rng, p, lwgt, lint)
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(x, pcov, w0, Int32(0), rng, p, logsig, buf, ctr, cap)
        s = (i - 1) % _MC_DIAG_STRIPE + 1
        _xsum_atomic_add!(diag, child_w, 1, s)
        _xsum_atomic_add!(diag, sloss, 2, s)
        _xsum_atomic_add!(diag, w0, 3, s)
        recorded && _mc_record_dev!(xw, xwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- layer-n (children) kernel
"""
    _mc_scatter_layer_kernel!(xw, xwE, cnt, bin, bout, ctr, cap, diag, logsig, p, M)

Process `M` buffered scattered photons (order `n`) of the previous layer: each
marches on its own genealogy stream `bin.r[j]` with the biased trigger
(appending order-(n+1) children to `bout`), and the escaped photon is recorded
into the order-`n` spectrum. Diagnostic totals go exactly into the striped
`diag`. The top layer (n == max_scatt) spawns nothing but still depletes +
records, mirroring the CPU path.
"""
@kernel function _mc_scatter_layer_kernel!(xw, xwE, cnt, bin, bout, ctr, @Const(cap), diag,
                                           @Const(logsig), p, @Const(M))
    j = @index(Global)
    @inbounds if j <= M
        rng = bin.r[j]
        pcov = bin.p[j]
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(bin.x[j], pcov, bin.w[j], bin.n[j], rng, p, logsig,
                               bout, ctr, cap)
        s = (j - 1) % _MC_DIAG_STRIPE + 1
        _xsum_atomic_add!(diag, child_w, 1, s)
        _xsum_atomic_add!(diag, sloss, 2, s)
        recorded && _mc_record_dev!(xw, xwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- scatter host driver
# Child SoA buffer of capacity `cap` on `backend`: position, momentum, weight,
# scattering order, genealogy stream.
_mc_child_buf(backend, cap::Int) =
    (x = KernelAbstractions.allocate(backend, SVector{4,Float64}, cap),
     p = KernelAbstractions.allocate(backend, SVector{4,Float64}, cap),
     w = KernelAbstractions.allocate(backend, Float64, cap),
     n = KernelAbstractions.allocate(backend, Int32, cap),
     r = KernelAbstractions.allocate(backend, MCRng, cap))
_mc_buf_cap(buf) = length(buf.w)

# isbits kernel parameters of the scattering run (thin bundle + scatter fields).
function _mc_scatter_params(model::SphereModel, tabs, grid, seed::Integer, K::Int,
                            layout::_MCM.MCStreamLayout)
    return merge(_mc_thin_params(model, tabs, grid, seed),
                 (weight_min = _MCM._MC_WEIGHT_MIN, max_scatt = K, layout = layout,
                  ucon = model.ucon, ucov = model.ucov, bcon = model.bcon))
end

"""
    mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns; device=0, devices=nothing,
                    workgroupsize=nothing, seed=0x5ca77e5,
                    gen_nu_min=1e8, gen_nu_max=1e13, spec_nu_min=1e8, spec_nu_max=1e17,
                    n_esamp=400, n_ebins=140, n_thbins=6, max_scatt=3,
                    capacity_factor=0.5) -> ScatterResult

Device Compton-scattering Monte Carlo, the twin of `mc_spectrum_scatter`
(mc_cpu.jl), selected by the extra positional hot-cross `table`: builds the
same synchrotron weight table + seed count `N_gen` once, then runs the layered
scatter loop (primary kernel over the seeds, `_mc_scatter_layer_kernel!` over
each layer's buffered children for orders 1..max_scatt), returning a
`ScatterResult` directly comparable to the CPU result. `capacity_factor` sizes
the initial child buffer (at least `min_capacity`); a launch that overflows it
is rerun with the exact size. `devices` (CUDA ordinals or `HostDevice`s) splits the seeds into
contiguous global-id ranges; spectra, children and diagnostics are bitwise
independent of the device count.
"""
function mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns::Real; kw...)
    res, _ = _mc_scatter_gpu(model, table, Ns; kw...)
    return res
end

"""
    _mc_scatter_gpu(model, table, Ns; collect_children=false, kw...) -> (ScatterResult, children)

Driver body of the scattering `mc_spectrum_gpu`. With `collect_children`,
`children[n]` lists every order-`n` child as `(id, x, p, w, nscatt)` sorted by
stream identity `id` (for invariance tests); otherwise `children === nothing`.
"""
function _mc_scatter_gpu(model::SphereModel, table::HotcrossTable, Ns::Real;
                         device = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5ca77e5,
                         gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e13,
                         spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e17,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 140, n_thbins::Integer = 6,
                         max_scatt::Integer = 3, capacity_factor::Real = 0.5,
                         min_capacity::Integer = 4096,
                         tabulated_gen::Bool = false, collect_children::Bool = false)
    devs = _mc_devlist(device, devices)
    tabs = _mc_sphere_tables(model, Ns, gen_nu_min, gen_nu_max, n_esamp, tabulated_gen)
    K = Int(max_scatt)
    layout = _MCM.MCStreamLayout(tabs.N_gen, _MCM._MC_TRACK_MAX_STEPS, K)
    mkx() = _MCM.ExactSpectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                               n_ebins = n_ebins, n_thbins = n_thbins)
    xorders = [mkx() for _ in 0:K]
    p = _mc_scatter_params(model, tabs, xorders[1], seed, K, layout)
    shares = _mc_device_shares(tabs.N_gen, length(devs))
    parts = Vector{Any}(undef, length(devs))
    @sync for di in eachindex(devs)
        @async parts[di] = _mc_scatter_part(devs[di], tabs, table, p, shares[di], K,
                                            Int(n_thbins), Int(n_ebins), capacity_factor,
                                            Int(min_capacity), workgroupsize, collect_children)
    end

    # exact merge over devices: per-order bins, layer diagnostics, child counts
    diag = _MCM.xsum_zeros(3, K + 1)         # rows: child w, scatter loss, primary w
    nphot = zeros(Int, K + 1)                # photons marched per layer (addend counts)
    children = collect_children ? [Any[] for _ in 1:K] : nothing
    for part in parts
        part === nothing && continue
        for n in 1:(K + 1)
            _mc_add_bins!(xorders[n], part.bins[n])
        end
        diag .+= part.diag
        nphot .+= part.nphot
        if collect_children
            for n in 1:K
                append!(children[n], part.children[n])
            end
        end
    end
    collect_children && foreach(c -> sort!(c; by = first), children)
    maximum(nphot) <= _MCM.XSUM_MAX_ADDENDS || error("exact diagnostic capacity exceeded")
    rnd(r, n) = _MCM.xsum_round(view(diag, :, r, n))
    rsum(r) = _MCM.xsum_round(vec(sum(view(diag, :, r, :); dims = 2)))
    res = _MCM._scatter_result(xorders, K, tabs.N_gen, sum(nphot) - nphot[1],
                               rnd(3, 1), rsum(1), rnd(1, 1), rsum(2))
    return res, children
end

# One device's share of the scattering run: all layers of the seeds
# `share.first : share.first + share.n - 1`. Returns per-order exact device bins,
# per-layer exact diagnostics (limbs x 3 x layers, stripes summed), per-layer
# photon counts and (optionally) the children of every layer.
function _mc_scatter_part(dev, tabs, table::HotcrossTable, p, share, K::Int,
                          nth::Int, neb::Int, capacity_factor, min_capacity::Int, wgs,
                          collect_children::Bool)
    share.n > 0 || return nothing
    _device!(dev)
    backend = _backend(dev)
    lwgt_d = _upload(backend, tabs.wgt.lwgt)
    lint_d = tabs.gt === nothing ? nothing : _upload(backend, tabs.gt.ljnu)
    logsig_d = _upload(backend, table.logsigma)
    bins = [_mc_dev_bins(backend, nth, neb) for _ in 0:K]
    ddiag = [KernelAbstractions.zeros(backend, Int64, _XSLOT, 3, _MC_DIAG_STRIPE) for _ in 0:K]
    ctr = KernelAbstractions.zeros(backend, Int64, 1)
    cap = max(min_capacity, 1, ceil(Int, Float64(capacity_factor) * share.n))
    bufs = [_mc_child_buf(backend, cap), _mc_child_buf(backend, cap)]
    kern0 = _mc_scatter_primary_kernel!(backend)
    kernL = _mc_scatter_layer_kernel!(backend)
    nphot = zeros(Int, K + 1)
    nphot[1] = share.n
    children = collect_children ? [Any[] for _ in 1:K] : nothing
    M = share.n                               # photons in the current layer
    for layer in 0:K
        M > 0 || break
        bin = bufs[1]                         # this layer's input (layer >= 1)
        out = bufs[2]
        while true
            fill!(ctr, 0)
            fill!(bins[layer + 1].cnt, 0); fill!(bins[layer + 1].w, 0)
            fill!(bins[layer + 1].wE, 0); fill!(ddiag[layer + 1], 0)
            b = bins[layer + 1]
            if layer == 0
                _mc_launch(kern0, wgs, M, b.w, b.wE, b.cnt, out, ctr, _mc_buf_cap(out),
                           ddiag[1], lwgt_d, lint_d, logsig_d, p, share.first - 1, M)
            else
                _mc_launch(kernL, wgs, M, b.w, b.wE, b.cnt, bin, out, ctr, _mc_buf_cap(out),
                           ddiag[layer + 1], logsig_d, p, M)
            end
            KernelAbstractions.synchronize(backend)
            n_out = Int(Array(ctr)[1])
            n_out <= _mc_buf_cap(out) && break
            out = _mc_child_buf(backend, n_out)   # exact size; the rerun is identical
            bufs[2] = out
        end
        n_out = Int(Array(ctr)[1])
        layer < K && (nphot[layer + 2] = n_out)
        if collect_children && n_out > 0
            hx = Array(out.x)[1:n_out]; hp = Array(out.p)[1:n_out]
            hw = Array(out.w)[1:n_out]; hn = Array(out.n)[1:n_out]
            hr = Array(out.r)[1:n_out]
            for q in 1:n_out
                push!(children[layer + 1],
                      (_MCM.mc_stream_id(hr[q]), hx[q], hp[q], hw[q], hn[q]))
            end
        end
        M = n_out
        bufs[1], bufs[2] = bufs[2], bufs[1]
    end

    diag = _MCM.xsum_zeros(3, K + 1)
    for n in 1:(K + 1)
        diag[:, :, n] .= dropdims(sum(Array(ddiag[n]); dims = 3); dims = 3)
    end
    return (bins = bins, diag = diag, nphot = nphot, children = children)
end
