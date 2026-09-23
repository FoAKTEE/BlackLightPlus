# mc_gpu.jl — GPU Monte Carlo radiative transfer (KernelAbstractions + CUDA).
#
# Device port of the sphere Monte Carlo: per superphoton, seed the Philox
# stream `MCRng(seed, i)` (same key as the CPU path), generate a synchrotron
# packet, track it through the flat sphere, and bin it atomically into a
# device (theta, E) spectrum; a layered host loop adds Compton scattering
# (cf. gpumonty scattering.cu). Leaf samplers are the same pure/isbits CPU
# functions; the weight lookup and scatter march use the shared mc_common.jl
# bodies, so CPU and GPU cannot drift. Include after `using BlackLightPlus`.

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

# CPU-side constants, referenced (not re-typed) so the kernels can never drift
# from mc_cpu.jl / track.jl / generate.jl / emission_tables.jl.
const _MCM = BlackLightPlus.MonteCarlo

# ---------------------------------------------------------------- multi-GPU seed stride
# Multi-device split: device `di` (1-based) uses global seed
# `seed + (di-1)*_MC_MULTI_SEED_STRIDE`, so per-device Philox streams never
# overlap; `di == 1` leaves the seed unchanged, so `devices=[d]` reproduces the
# single-device path bit-for-bit. Stride 2^30 keeps the masked 32-bit seeds far apart.
const _MC_MULTI_SEED_STRIDE = 1_073_741_824   # 2^30

# Per-device share of a multi-device run of `N_gen_full` superphotons over `ndev`
# devices: device `di` launches ids `1:shares[di].n` with global seed `shares[di].seed`.
_mc_device_shares(seed::Integer, N_gen_full::Integer, ndev::Integer) =
    [(seed = Int(seed) + (di - 1) * _MC_MULTI_SEED_STRIDE,
      n = div(N_gen_full, ndev) + (di <= mod(N_gen_full, ndev) ? 1 : 0)) for di in 1:ndev]

# 32-bit Philox key word of a global seed: superphoton `i` draws from `MCRng(key, i)`.
_mc_seed_key(seed::Integer) = UInt32(Int(seed) & 0xffffffff)

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

# ---------------------------------------------------------------- the per-superphoton kernel
"""
    _mc_thin_kernel!(sumw, sumwE, cnt, lwgt, lint, p, N)

One superphoton per global index `i in 1:N`: seeds `MCRng(p.seed, i)` (the same
stream the CPU path uses for photon `i`), generates a synchrotron packet
(`_mc_gen_sphere_photon`), tracks it straight to escape with absorption-weight
decay, and atomically bins `(count, w, w*E)` by exit angle + energy. `p` is the
isbits parameter bundle; `lwgt` the uploaded weight table; `lint` the optional
tabulated-generation table (`nothing` = exact default).
"""
@kernel function _mc_thin_kernel!(sumw, sumwE, cnt, @Const(lwgt), lint, p, @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        # ---- generate (stream keyed (seed, photon_id)) ----
        x, pcov, w, _ = _mc_gen_sphere_photon(MCRng(p.seed, i), p, lwgt, lint)

        # ---- track straight to escape with absorption-weight decay ----
        w = _mc_track_thin_w(x, pcov, w, p.gcon, p.ne, p.te, p.B, p.R,
                             p.step_frac, p.track_max)

        # ---- record: bin by exit (theta, E) (spectrum.jl record!) ----
        Ee = -pcov[1]                                    # E = -p_t
        if Ee > 0.0
            nu_out = Ee * ME * CL * CL / HPL
            ie = unsafe_trunc(Int, floor((log(nu_out) - p.spec_lnu_min) / p.spec_dlnu)) + 1
            if 1 <= ie <= p.n_ebins
                ps = sqrt(pcov[2] * pcov[2] + pcov[3] * pcov[3] + pcov[4] * pcov[4])
                mu = ps > 0.0 ? abs(pcov[4]) / ps : 0.0  # |cos theta| about z, folded
                thf = acos(clamp(mu, 0.0, 1.0))
                ith = clamp(unsafe_trunc(Int, floor(thf / p.dth)) + 1, 1, p.n_thbins)
                CUDA.@atomic cnt[ith, ie]   += Int64(1)
                CUDA.@atomic sumw[ith, ie]  += w
                CUDA.@atomic sumwE[ith, ie] += w * Ee
            end
        end
    end
end

# ---------------------------------------------------------------- host driver
"""
    mc_spectrum_gpu(model::SphereModel, Ns; device=0, workgroupsize=nothing,
                    seed=0x5eed5eed, nu_min=1e8, nu_max=1e16, n_esamp=400,
                    n_ebins=200, n_thbins=6) -> Spectrum

GPU thin-sphere Monte Carlo, the device twin of `mc_spectrum` (mc_cpu.jl):
builds the same one-zone weight table and target count `N_gen` on the host (so
packet count and Philox streams match the CPU path photon-for-photon), launches
`_mc_thin_kernel!` on `device`, and reduces the atomic device bins into the
returned host `Spectrum`. Pass `devices` for a multi-GPU split: each device
builds the full-`Ns` weight table and runs its `n_gen_override` share with a
distinct Philox seed (splitting `Ns` instead would double every weight — a 2x
total-power bug); `devices=[d]` reproduces the single-device run bit-for-bit.
"""
function mc_spectrum_gpu(model::SphereModel, Ns::Real;
                         device::Integer = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5eed5eed,
                         nu_min::Real = _MCM._MC_NU_MIN_DEFAULT,
                         nu_max::Real = _MCM._MC_NU_MAX_DEFAULT,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 200, n_thbins::Integer = 6,
                         tabulated_gen::Bool = false,
                         n_gen_override::Union{Nothing,Integer} = nothing)
    # ---- multi-GPU split ----
    # The thin spectrum is additive, so the combined result is the per-device
    # sum — but each device must build the full-`Ns` weight table and run only
    # its N_gen/ndev share (splitting `Ns` would double the weights). Devices
    # run concurrently; distinct seeds give disjoint streams.
    if devices !== nothing
        devlist = collect(Int, devices)
        ndev = length(devlist)
        # full-ensemble N_gen (from the FULL Ns) so per-device weights stay full-normalised.
        band = log(nu_max) - log(nu_min)
        zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
        wgt = build_weight_table([zone], Float64(Ns) / band;
                                 n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
        N_target, _ = zone_nsuperphotons(zone, wgt)
        N_gen_full = max(1, round(Int, N_target))
        parts = Vector{Spectrum}(undef, ndev)
        shares = _mc_device_shares(seed, N_gen_full, ndev)
        @sync for di in 1:ndev
            dev = devlist[di]
            sdev = shares[di].seed
            n_this = shares[di].n
            @async begin
                CUDA.device!(dev)
                parts[di] = mc_spectrum_gpu(model, Ns; device = dev,
                                            workgroupsize = workgroupsize, seed = sdev,
                                            nu_min = nu_min, nu_max = nu_max,
                                            n_esamp = n_esamp, n_ebins = n_ebins,
                                            n_thbins = n_thbins,
                                            tabulated_gen = tabulated_gen,
                                            n_gen_override = n_this)
            end
        end
        combined = Spectrum(; nu_min = nu_min, nu_max = nu_max,
                            n_ebins = n_ebins, n_thbins = n_thbins)
        for pr in parts
            merge_spectra!(combined, pr)
        end
        return combined
    end

    # ---- host generation tables (identical to mc_spectrum) ----
    band = log(nu_max) - log(nu_min)
    Ns_param = Float64(Ns) / band                        # so N_target ~ Ns
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    # full-Ns weight table always; the split overrides ONLY the launched superphoton count.
    N_gen = n_gen_override === nothing ? max(1, round(Int, N_target)) : max(1, Int(n_gen_override))
    # opt-in tabulated generation (host tables; the CPU opt-in's exact mirror).
    gt = tabulated_gen ? _MCM.build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = _MCM.tabulated_dnmax(gt, wgt))

    # emission tetrad is position-independent for the flat, uniform sphere -> host-constant.
    Etet = emission_tetrad(model.ucon, model.ucov, model.bcon, model.gcov, model.gcon)

    # host spectrum: carries the grid scalars AND receives the reduced device bins.
    spec = Spectrum(; nu_min = nu_min, nu_max = nu_max,
                    n_ebins = n_ebins, n_thbins = n_thbins)

    # ---- device launch ----
    CUDA.device!(Int(device))
    backend = CUDABackend()
    lwgt_d = CuArray(wgt.lwgt)
    lint_d = gt === nothing ? nothing : CuArray(gt.ljnu)
    d_sumw  = CUDA.zeros(Float64, n_thbins, n_ebins)
    d_sumwE = CUDA.zeros(Float64, n_thbins, n_ebins)
    d_cnt   = CUDA.zeros(Int64,   n_thbins, n_ebins)

    p = (seed = _mc_seed_key(seed),
         ne = model.n_e, te = model.theta_e, B = model.B, R = model.R,
         lnu_min = wgt.lnu_min, lnu_max = wgt.lnu_max, dlnu = wgt.dlnu,
         n_nodes = length(wgt.lwgt), dnmax = dnmax,
         weight_floor = _MCM._WEIGHT_FLOOR, reject_max = _MCM._MC_REJECT_MAX_ITER,
         spec_lnu_min = spec.lnu_min, spec_dlnu = spec.dlnu,
         n_ebins = Int(n_ebins), n_thbins = Int(n_thbins),
         dth = (0.5 * π) / n_thbins,
         step_frac = _MCM._MC_STEP_FRAC, track_max = _MCM._MC_TRACK_MAX_STEPS,
         E = Etet, gcov = model.gcov, gcon = model.gcon)
    gt === nothing || (p = merge(p, (lint_dlnu = gt.dlnu, lint_n = length(gt.ljnu))))

    kern = _mc_thin_kernel!(backend)
    if workgroupsize === nothing
        kern(d_sumw, d_sumwE, d_cnt, lwgt_d, lint_d, p, N_gen; ndrange = N_gen)
    else
        kern(d_sumw, d_sumwE, d_cnt, lwgt_d, lint_d, p, N_gen;
             ndrange = N_gen, workgroupsize = min(Int(workgroupsize), N_gen))
    end
    KernelAbstractions.synchronize(backend)

    # ---- reduce device bins into the host Spectrum ----
    spec.sumw  .= Array(d_sumw)
    spec.sumwE .= Array(d_sumwE)
    spec.count .= Array(d_cnt)
    return spec
end

# ======================================================================================
# GPU COMPTON SCATTERING (idealized Comptonization sphere)
#
# Layered, not recursive (after gpumonty scattering.cu `scattering_flow_control`):
# a host loop over scattering layers. The primary kernel marches the seed
# photons with the biased exponential-transform scatter trigger; on a trigger it
# atomically appends the Compton-scattered child to a preallocated device SoA
# buffer via a global slot counter and continues the depleted incident beam;
# escaped photons are binned into the per-order spectrum. The host then launches
# the same march kernel on each layer's buffered children (ping-pong buffers)
# for orders 1..max_scatt. Unbiasedness: each child carries
# w(1-e^{-dtau_s})/(1-e^{-b dtau_s}) at spawn probability 1-exp(-b*dtau_s), so
# E[spawned weight] equals the physical scatter loss; the device accumulates
# both totals (striped atomics) and their ratio must be ~1 independent of the
# bias. The electron / Klein-Nishina rejection loops run in-kernel (pure,
# isbits, capped); the scatter sampler sits behind a `@noinline` boundary so it
# does not inflate the march's register footprint.

# Diagnostic-accumulator stripe count: threads add their per-photon totals to
# diag[:, 1 + (id-1)%_MC_DIAG_STRIPE], summed on the host — avoids single-address atomic
# serialisation across ~N_gen threads.
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

# ---------------------------------------------------------------- record one escapee
# Atomic bin (theta, E) of an escaped photon with covariant momentum `pcov` and weight `w`
# into the order's device spectrum — a device twin of spectrum.jl `record!` on the SPEC grid.
@inline function _mc_record_dev!(sumw, sumwE, cnt, pcov::SVector{4}, w, p)
    Ee = -pcov[1]
    if Ee > 0.0
        nu_out = Ee * ME * CL * CL / HPL
        ie = unsafe_trunc(Int, floor((log(nu_out) - p.spec_lnu_min) / p.spec_dlnu)) + 1
        if 1 <= ie <= p.n_ebins
            ps = sqrt(pcov[2] * pcov[2] + pcov[3] * pcov[3] + pcov[4] * pcov[4])
            mu = ps > 0.0 ? abs(pcov[4]) / ps : 0.0
            thf = acos(clamp(mu, 0.0, 1.0))
            ith = clamp(unsafe_trunc(Int, floor(thf / p.dth)) + 1, 1, p.n_thbins)
            CUDA.@atomic cnt[ith, ie]   += Int64(1)
            CUDA.@atomic sumw[ith, ie]  += w
            CUDA.@atomic sumwE[ith, ie] += w * Ee
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the biased scatter march
# Transport one superphoton straight through the sphere with the biased
# exponential-transform scatter decision: the march is the shared
# `_MCM._mc_march_scatter_core` (mc_common.jl); this wrapper supplies the device
# opacity lookup (uploaded hot-cross table) and the atomic SoA child sink
# (`xb,pb,wb,nb` + slot counter `ctr`). Absorption is neglected (idealized
# Comptonization sphere), matching the CPU path. Returns
# (recorded, w_final, scatter_loss, child_w_local).
@inline function _mc_march_scatter!(x0::SVector{4}, pcov::SVector{4}, w0, nscatt::Int32,
                                    rng::MCRng, p, logsig, xb, pb, wb, nb, ctr)
    nu_alpha_s = (x, kcon) -> _mc_nu_alpha_scatt_dev(x, kcon, logsig, p.ne, p.te, p.R)
    spawn! = (xev, weight_scat, tau_s, srng) -> begin
        p_new, srng = _mc_scatter_mom(pcov, p.ucon, p.ucov, p.bcon, p.te,
                                      p.gcov, p.gcon, srng)
        slot = CUDA.atomic_add!(pointer(ctr, 1), Int64(1))   # 0-based old index
        kept = 0.0
        if slot < p.cap
            j = Int(slot) + 1
            @inbounds begin
                xb[j] = xev
                pb[j] = p_new
                wb[j] = weight_scat
                nb[j] = nscatt + Int32(1)
            end
            kept = weight_scat   # ledger only TRACKED children (overflow drops excluded)
        end
        return kept, srng
    end
    recorded, _, w, _, sloss, child_w, _ =
        _MCM._mc_march_scatter_core(x0, pcov, w0, 0.0, nscatt, rng, p,
                                    nu_alpha_s, spawn!)
    return (recorded, w, sloss, child_w)
end

# ---------------------------------------------------------------- layer-0 (primary) kernel
"""
    _mc_scatter_primary_kernel!(sumw, sumwE, cnt, xb, pb, wb, nb, ctr, diag,
                                lwgt, lint, logsig, p, N)

One seed superphoton per index `i in 1:N`: generate from the sphere (stream
`MCRng(p.seed, i)`), march with the biased scatter trigger (appending order-1
children to the SoA buffer), record the escaped photon into the order-0
spectrum. Per-thread totals (primary weight / scatter loss / order-1 child
weight) reduce into the striped `diag`.
"""
@kernel function _mc_scatter_primary_kernel!(sumw, sumwE, cnt, xb, pb, wb, nb, ctr, diag,
                                             @Const(lwgt), lint, @Const(logsig), p, @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        rng = MCRng(p.seed, i)
        x, pcov, w0, rng = _mc_gen_sphere_photon(rng, p, lwgt, lint)
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(x, pcov, w0, Int32(0), rng, p, logsig, xb, pb, wb, nb, ctr)
        s = (i - 1) % _MC_DIAG_STRIPE + 1
        CUDA.@atomic diag[1, s] += child_w
        CUDA.@atomic diag[2, s] += sloss
        CUDA.@atomic diag[3, s] += w0
        recorded && _mc_record_dev!(sumw, sumwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- layer-n (children) kernel
"""
    _mc_scatter_layer_kernel!(sumw, sumwE, cnt, xin, pin, win, nin,
                              xout, pout, wout, nout, ctr, diag, logsig, p, child_seed, M)

Process `M` buffered scattered photons (order `n`) from the previous layer:
give each a fresh stream `MCRng(child_seed, j)`, march with the biased trigger
(appending order-(n+1) children to the OUT buffer), record the escaped photon
into the order-`n` spectrum. Diagnostic totals reduce into the striped `diag`.
The top layer (n == max_scatt) spawns nothing but still depletes + records,
mirroring the CPU path.
"""
@kernel function _mc_scatter_layer_kernel!(sumw, sumwE, cnt,
                                           @Const(xin), @Const(pin), @Const(win), @Const(nin),
                                           xout, pout, wout, nout, ctr, diag,
                                           @Const(logsig), p, @Const(child_seed), @Const(M))
    j = @index(Global)
    @inbounds if j <= M
        rng = MCRng(child_seed, j)
        x = xin[j]
        pcov = pin[j]
        w0 = win[j]
        nscatt = nin[j]
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(x, pcov, w0, nscatt, rng, p, logsig, xout, pout, wout, nout, ctr)
        s = (j - 1) % _MC_DIAG_STRIPE + 1
        CUDA.@atomic diag[1, s] += child_w
        CUDA.@atomic diag[2, s] += sloss
        recorded && _mc_record_dev!(sumw, sumwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- multi-device reduction
# Sum per-device `ScatterResult`s into one: per-order spectra add, the total
# re-sums the merged orders, and every scalar diagnostic is the device sum.
# Unbiasedness (w_children ~ scatter_loss) is preserved because both numerator
# and denominator add. Mirrors `mc_spectrum_scatter`'s per-chunk reduction.
function _merge_scatter_results(parts::Vector{ScatterResult};
                                spec_nu_min, spec_nu_max, n_ebins, n_thbins)
    K = parts[1].max_scatt
    orders = [Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                       n_ebins = n_ebins, n_thbins = n_thbins) for _ in 0:K]
    for r in parts, n in 1:(K + 1)
        merge_spectra!(orders[n], r.orders[n])
    end
    total = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
    for n in 1:(K + 1)
        merge_spectra!(total, orders[n])
    end
    return ScatterResult(orders, total, K,
                         sum(r.n_primary for r in parts),
                         sum(r.n_children for r in parts),
                         sum(r.w_primary for r in parts),
                         sum(r.w_children for r in parts),
                         sum(r.w_first_order for r in parts),
                         sum(r.scatter_loss for r in parts))
end

# ---------------------------------------------------------------- host driver (scatter)
"""
    mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns; device=0,
                    workgroupsize=nothing, seed=0x5ca77e5,
                    gen_nu_min=1e8, gen_nu_max=1e13, spec_nu_min=1e8, spec_nu_max=1e17,
                    n_esamp=400, n_ebins=140, n_thbins=6, max_scatt=3,
                    capacity_factor=0.5) -> ScatterResult

GPU Compton-scattering Monte Carlo, the device twin of `mc_spectrum_scatter`
(mc_cpu.jl), selected by the extra positional hot-cross `table`: builds the
same synchrotron weight table + seed count `N_gen`, then runs the layered
scatter loop (primary kernel over the seeds, `_mc_scatter_layer_kernel!` over
each layer's buffered children for orders 1..max_scatt), returning a
`ScatterResult` directly comparable to the CPU result. `capacity_factor` sizes
the per-layer child buffer (overflow guarded + warned). Pass `devices` for a
multi-GPU split: each device builds the full-`Ns` weight table and runs its
seed share with a distinct global (hence child) seed — the split fixes the seed
count, never `Ns`, which would double every weight (a 2x total-power bug).
"""
function mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns::Real;
                         device::Integer = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5ca77e5,
                         gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e13,
                         spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e17,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 140, n_thbins::Integer = 6,
                         max_scatt::Integer = 3, capacity_factor::Real = 0.5,
                         tabulated_gen::Bool = false,
                         n_gen_override::Union{Nothing,Integer} = nothing)
    # ---- multi-GPU split ----
    # Each device runs the full layered scatter loop on its share of the N_gen
    # seeds, with the full-`Ns` weight table and a distinct global seed (hence
    # distinct child seeds); `_merge_scatter_results` sums per-order spectra +
    # diagnostics. Only the seed count is split, never `Ns` (weight doubling).
    if devices !== nothing
        devlist = collect(Int, devices)
        ndev = length(devlist)
        # full-ensemble N_gen (from the FULL Ns) so per-device weights stay full-normalised.
        zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
        wgt = build_weight_table([zone], Float64(Ns) / (log(gen_nu_max) - log(gen_nu_min));
                                 n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
        N_target, _ = zone_nsuperphotons(zone, wgt)
        N_gen_full = max(1, round(Int, N_target))
        parts = Vector{ScatterResult}(undef, ndev)
        shares = _mc_device_shares(seed, N_gen_full, ndev)
        @sync for di in 1:ndev
            dev = devlist[di]
            sdev = shares[di].seed
            n_this = shares[di].n
            @async begin
                CUDA.device!(dev)
                parts[di] = mc_spectrum_gpu(model, table, Ns; device = dev,
                                            workgroupsize = workgroupsize, seed = sdev,
                                            gen_nu_min = gen_nu_min, gen_nu_max = gen_nu_max,
                                            spec_nu_min = spec_nu_min, spec_nu_max = spec_nu_max,
                                            n_esamp = n_esamp, n_ebins = n_ebins,
                                            n_thbins = n_thbins, max_scatt = max_scatt,
                                            capacity_factor = capacity_factor,
                                            tabulated_gen = tabulated_gen,
                                            n_gen_override = n_this)
            end
        end
        return _merge_scatter_results(collect(parts); spec_nu_min = spec_nu_min,
                                      spec_nu_max = spec_nu_max, n_ebins = n_ebins,
                                      n_thbins = n_thbins)
    end

    # ---- host generation tables (identical to mc_spectrum_scatter) ----
    band = log(gen_nu_max) - log(gen_nu_min)
    Ns_param = Float64(Ns) / band
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    # full-Ns weight table always; the split overrides ONLY the launched SEED count.
    N_gen = n_gen_override === nothing ? max(1, round(Int, N_target)) : max(1, Int(n_gen_override))
    # opt-in tabulated generation (host tables; the CPU opt-in's exact mirror).
    gt = tabulated_gen ? _MCM.build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = _MCM.tabulated_dnmax(gt, wgt))
    K = Int(max_scatt)
    Etet = emission_tetrad(model.ucon, model.ucov, model.bcon, model.gcov, model.gcon)

    # spec-grid scalars (a host Spectrum both carries the grid and receives the bins)
    specg = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)

    CUDA.device!(Int(device))
    backend = CUDABackend()
    lwgt_d  = CuArray(wgt.lwgt)
    lint_d  = gt === nothing ? nothing : CuArray(gt.ljnu)
    logsig_d = CuArray(table.logsigma)

    # per-order device spectra (orders 0..K)
    d_sumw  = [CUDA.zeros(Float64, n_thbins, n_ebins) for _ in 0:K]
    d_sumwE = [CUDA.zeros(Float64, n_thbins, n_ebins) for _ in 0:K]
    d_cnt   = [CUDA.zeros(Int64,   n_thbins, n_ebins) for _ in 0:K]

    # ping-pong SoA child buffers + slot counter + striped diagnostics
    cap = max(4096, ceil(Int, Float64(capacity_factor) * N_gen))
    xA = CuArray{SVector{4,Float64}}(undef, cap); pA = CuArray{SVector{4,Float64}}(undef, cap)
    wA = CuArray{Float64}(undef, cap);            nA = CuArray{Int32}(undef, cap)
    xB = CuArray{SVector{4,Float64}}(undef, cap); pB = CuArray{SVector{4,Float64}}(undef, cap)
    wB = CuArray{Float64}(undef, cap);            nB = CuArray{Int32}(undef, cap)
    ctr  = CUDA.zeros(Int64, 1)
    diag = CUDA.zeros(Float64, 3, _MC_DIAG_STRIPE)   # [child_w; scatter_loss; primary_w]

    p = (seed = _mc_seed_key(seed),
         ne = model.n_e, te = model.theta_e, B = model.B, R = model.R,
         lnu_min = wgt.lnu_min, lnu_max = wgt.lnu_max, dlnu = wgt.dlnu,
         n_nodes = length(wgt.lwgt), dnmax = dnmax,
         weight_floor = _MCM._WEIGHT_FLOOR, reject_max = _MCM._MC_REJECT_MAX_ITER,
         spec_lnu_min = specg.lnu_min, spec_dlnu = specg.dlnu,
         n_ebins = Int(n_ebins), n_thbins = Int(n_thbins), dth = (0.5 * π) / n_thbins,
         step_frac = _MCM._MC_STEP_FRAC, track_max = _MCM._MC_TRACK_MAX_STEPS,
         weight_min = _MCM._MC_WEIGHT_MIN, max_scatt = K, cap = cap,
         E = Etet, gcov = model.gcov, gcon = model.gcon,
         ucon = model.ucon, ucov = model.ucov, bcon = model.bcon)
    gt === nothing || (p = merge(p, (lint_dlnu = gt.dlnu, lint_n = length(gt.ljnu))))

    base_child_seed = (Int(seed) ⊻ 0x0C0FFEE5) & 0xffffffff

    # ---- layer 0: primaries -> order-0 spectrum, order-1 children into (xA,...) ----
    kern0 = _mc_scatter_primary_kernel!(backend)
    if workgroupsize === nothing
        kern0(d_sumw[1], d_sumwE[1], d_cnt[1], xA, pA, wA, nA, ctr, diag,
              lwgt_d, lint_d, logsig_d, p, N_gen; ndrange = N_gen)
    else
        kern0(d_sumw[1], d_sumwE[1], d_cnt[1], xA, pA, wA, nA, ctr, diag,
              lwgt_d, lint_d, logsig_d, p, N_gen; ndrange = N_gen,
              workgroupsize = min(Int(workgroupsize), N_gen))
    end
    KernelAbstractions.synchronize(backend)
    n_cur = Int(Array(ctr)[1])
    n_overflow = max(0, n_cur - cap)
    n_cur = min(n_cur, cap)
    w_first_order = sum(Array(diag)[1, :])   # order-1 child weight (spawned by primaries)
    n_children_total = n_cur

    # ---- layers 1..K: drain the children (ping-pong) ----
    curx, curp, curw, curn = xA, pA, wA, nA
    nxtx, nxtp, nxtw, nxtn = xB, pB, wB, nB
    kernL = _mc_scatter_layer_kernel!(backend)
    for layer in 1:K
        n_cur > 0 || break
        M = n_cur
        CUDA.fill!(ctr, 0)
        cseed = UInt32((base_child_seed ⊻ (0x51ED9E37 * layer)) & 0xffffffff)
        if workgroupsize === nothing
            kernL(d_sumw[layer + 1], d_sumwE[layer + 1], d_cnt[layer + 1],
                  curx, curp, curw, curn, nxtx, nxtp, nxtw, nxtn, ctr, diag,
                  logsig_d, p, cseed, M; ndrange = M)
        else
            kernL(d_sumw[layer + 1], d_sumwE[layer + 1], d_cnt[layer + 1],
                  curx, curp, curw, curn, nxtx, nxtp, nxtw, nxtn, ctr, diag,
                  logsig_d, p, cseed, M; ndrange = M,
                  workgroupsize = min(Int(workgroupsize), M))
        end
        KernelAbstractions.synchronize(backend)
        n_next = Int(Array(ctr)[1])
        n_overflow += max(0, n_next - cap)
        n_next = min(n_next, cap)
        n_children_total += n_next
        n_cur = n_next
        curx, nxtx = nxtx, curx; curp, nxtp = nxtp, curp
        curw, nxtw = nxtw, curw; curn, nxtn = nxtn, curn
    end
    n_overflow > 0 && @warn "mc_spectrum_gpu scatter buffer overflow: $n_overflow children dropped; raise capacity_factor" cap

    # ---- reduce device bins -> host per-order Spectra + total ----
    function _host_spec(idx)
        s = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
        s.sumw  .= Array(d_sumw[idx])
        s.sumwE .= Array(d_sumwE[idx])
        s.count .= Array(d_cnt[idx])
        return s
    end
    orders = [_host_spec(n + 1) for n in 0:K]
    total = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
    for o in orders
        merge_spectra!(total, o)
    end

    dh = Array(diag)
    w_children  = sum(dh[1, :])
    scatter_loss = sum(dh[2, :])
    w_primary   = sum(dh[3, :])
    return ScatterResult(orders, total, K, N_gen, n_children_total,
                         w_primary, w_children, w_first_order, scatter_loss)
end
