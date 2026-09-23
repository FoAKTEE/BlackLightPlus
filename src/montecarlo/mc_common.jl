# Monte Carlo — mc_common.jl
#
# Shared CPU/GPU kernel bodies. The MC engine runs the same physics on two
# surfaces — the threaded CPU path (generate.jl / track.jl / mc_cpu.jl) and the
# KernelAbstractions device kernels (mc_gpu.jl); each function here is the one
# body both sides call, so the two surfaces cannot drift. Everything is pure,
# allocation-free, isbits-safe and free of CUDA/KA types; RNG plumbing, atomics
# and container flavors (host `Vector`/`Matrix` vs uploaded device arrays) stay
# in the callers, entering through plain arguments or functors that inline.

"""
    _mc_weight_at(nu, lwgt, lnu_min, dlnu, n_nodes) -> Float64

Superphoton weight `w(nu)` [photons/packet] by piecewise log-linear interpolation
in `ln nu` over the `log w` nodes `lwgt` (host `Vector` or uploaded device
array), clamped to the grid endpoints. Pure / allocation-free / device-safe.
"""
@inline function _mc_weight_at(nu, lwgt, lnu_min, dlnu, n_nodes::Int)
    di = (log(nu) - lnu_min) / dlnu   # fractional node index (0-based)
    i = unsafe_trunc(Int, floor(di))
    if i < 0
        i = 0
        di = 0.0
    elseif i > n_nodes - 2
        i = n_nodes - 2
        di = Float64(n_nodes - 1)   # clamp to the LAST node (f = 1), not node N-1
    end
    f = di - i
    @inbounds return exp((1.0 - f) * lwgt[i + 1] + f * lwgt[i + 2])
end

"""
    _mc_scatter_event(w, b, dtau_s, weight_min, rng)
        -> (spawn::Bool, weight_scat::Float64, frac::Float64, rng::MCRng)

Biased exponential-transform scatter decision, shared by all scatter marches.
The child carries `w (1-e^{-dtau_s}) / (1-e^{-b dtau_s})` at spawn probability
`1-exp(-b dtau_s)`, so `E[spawned weight] = w (1-e^{-dtau_s})` — the physical
per-step scatter loss — for any bias `b >= 1` (`-> w/b` as `b*dtau_s -> 0`).
Draws one uniform; on `spawn` the event sits `frac = x1/(b*dtau_s)` into the
step. Callers compute `b`, guard on `dtau_s > 0 && nscatt < max_scatt`, and
build/deplete the photons.
"""
@inline function _mc_scatter_event(w, b, dtau_s, weight_min, rng::MCRng)
    weight_scat = w * (-expm1(-dtau_s)) / (-expm1(-b * dtau_s))
    u, rng = mc_uniform(rng)
    x1 = -log(u > 0.0 ? u : 1.0e-300)   # -log(U); U->0 gives huge x1 (no scatter)
    spawn = weight_scat > weight_min && b * dtau_s > x1
    return spawn, weight_scat, spawn ? x1 / (b * dtau_s) : 0.0, rng
end

"""
    _mc_march_scatter_core(x0, pcov, w0, tau_s0, nscatt, rng, mp, nu_alpha_s, spawn!)
        -> (recorded, x, w, tau_s, scatter_loss, child_w, rng)

Flat-sphere biased scatter march shared by the CPU and device trackers: `k^mu`
is null and constant, so each straight-line step accumulates the trapezoid
scattering depth, runs `_mc_scatter_event`, and depletes the incident beam by
the physical `exp(-dtau_s)` (absorption neglected — idealized Comptonization
sphere). `nu_alpha_s(x, kcon)` supplies the invariant scattering rate and
`spawn!(xev, weight_scat, tau_s, rng, event) -> (kept_w, rng)` the child sink,
where `event` is the 0-based spawn ordinal along this march (< `track_max`, the
child's stream-identity field, see rng.jl); `mp`
bundles `gcon, R, ne, te, weight_min, max_scatt, step_frac, track_max`.
Returns the escape flag, final state, accumulated scattering depth, the
incident weight lost to scattering, and the total spawned child weight.
"""
@inline function _mc_march_scatter_core(x0::SVector{4}, pcov::SVector{4}, w0, tau_s0,
                                        nscatt::Int32, rng::MCRng, mp,
                                        nu_alpha_s::FA, spawn!::FS) where {FA,FS}
    kcon = mp.gcon * pcov                       # null + constant (flat sphere)
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ks > 0.0 || return (false, x0, w0, tau_s0, 0.0, 0.0, rng)
    dlam = mp.step_frac * mp.R / ks
    Cfac = HPL / (ME * CL * CL)                 # h / m_e c^2  [s]

    ns0 = nu_alpha_s(x0, kcon)
    x = x0
    w = w0
    tau_s = tau_s0
    scatter_loss = 0.0
    child_w = 0.0
    nspawn = 0
    recorded = false
    @inbounds for _ in 1:mp.track_max
        xn = x .+ kcon .* dlam
        ns1 = nu_alpha_s(xn, kcon)
        dtau_s = 0.5 * (ns0 + ns1) * Cfac * dlam

        # ---- biased scatter decision (shared exponential transform) ----------
        if dtau_s > 0.0 && Int(nscatt) < mp.max_scatt
            b = bias_func(mp.ne, mp.R, mp.te, w, nscatt)
            spawn, weight_scat, frac, rng =
                _mc_scatter_event(w, b, dtau_s, mp.weight_min, rng)
            if spawn
                xev = x .+ kcon .* (dlam * frac)
                kept, rng = spawn!(xev, weight_scat, tau_s, rng, nspawn)
                child_w += kept
                nspawn += 1
            end
        end

        # ---- physical depletion of the incident beam (unbiased) --------------
        scatter_loss += w * (-expm1(-dtau_s))   # w (1 - exp(-dtau_s))
        w *= exp(-dtau_s)
        tau_s += dtau_s

        x = xn
        ns0 = ns1
        r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
        if r2 > mp.R * mp.R                     # escaped -> record
            recorded = true
            break
        end
    end
    return (recorded, x, w, tau_s, scatter_loss, child_w, rng)
end
