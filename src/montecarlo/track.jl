# Monte Carlo — track.jl
#
# Superphoton tracking through the uniform flat sphere. The coordinate
# wavevector k^mu = g^{mu nu} p_nu is null and constant (Minkowski), so the
# geodesic is the straight line x(lambda) = x0 + k^mu lambda. Along the ray the
# packet weight decays deterministically, w *= exp(-dtau_a), with the trapezoid
# dtau_a = 1/2 [(nu alpha_nu)_n + (nu alpha_nu)_{n+1}] (h / m_e c^2) dlam,
# where nu = -(p.u) m_e c^2 / h is the fluid-frame frequency; the invariant
# grouping (nu alpha_nu) = alpha_inv makes the march frame-independent.

# Step control: advance ~ STEP_FRAC * R in space per step (a few tens of steps to
# cross the sphere; the trapezoid is exact for the uniform medium regardless).
const _MC_STEP_FRAC = 0.05
# Safety cap so a pathological ray can never spin forever (never reached: a null
# ray started inside a convex sphere always exits in ~1/STEP_FRAC steps).
const _MC_TRACK_MAX_STEPS = 100_000

"True iff the spatial position `x` lies inside (or on) the sphere of radius `R`."
@inline function _inside_sphere(model, x::SVector{4})
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    return r2 <= model.R * model.R
end

"""
    _nu_alpha(model, x, kcon) -> Float64

The invariant absorption rate `nu * alpha_nu` [cm^-1 s^-1] at coordinate `x` for a
photon with contravariant wavevector `kcon`. Returns `0` in the vacuum outside the
sphere. Sphere-specific (flat, static fluid, uniform B along x1): the fluid-frame
frequency is `nu = k^t m_e c^2 / h` and the pitch cosine to b-hat is `k^1/|k_spatial|`.
"""
@inline function _nu_alpha(model, x::SVector{4}, kcon::SVector{4})
    _inside_sphere(model, x) || return 0.0
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ks <= 0.0 && return 0.0
    cth = kcon[2] / ks                          # pitch cosine to b-hat = x1
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    nu = kcon[1] * ME * CL * CL / HPL           # k^t = -p_t = E ; nu = E m_e c^2 / h
    return nu * alpha_nu(model.n_e, model.theta_e, model.B, nu, sth)
end

"""
    track_superphoton(sph::Superphoton, model, rng) -> (Superphoton, recorded::Bool)

Transport `sph` through `model` until it escapes the sphere, decaying its
weight by the accumulated absorption optical depth. Returns the escaped packet
and `recorded = true`; a packet that hits the step cap (never, in practice)
returns `recorded = false`. Absorption-only: `rng` is accepted but unused.
"""
function track_superphoton(sph::Superphoton, model, rng::MCRng)
    x = sph.x
    p = sph.p
    w = sph.w
    tau = sph.tau_abs

    # k^mu = g^{mu nu} p_nu — null and constant for the flat sphere metric.
    kcon = model.gcon * p
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    dlam = ks > 0.0 ? _MC_STEP_FRAC * model.R / ks : model.R
    Cabs = HPL / (ME * CL * CL)                 # h / m_e c^2  [s]

    na0 = _nu_alpha(model, x, kcon)
    recorded = false
    @inbounds for _ in 1:_MC_TRACK_MAX_STEPS
        xn = x .+ kcon .* dlam                  # straight-line push (flat metric)
        na1 = _nu_alpha(model, xn, kcon)        # k^mu constant -> reuse kcon
        dtau = 0.5 * (na0 + na1) * Cabs * dlam
        tau += dtau
        w *= exp(-dtau)

        x = xn
        na0 = na1

        if !_inside_sphere(model, x)            # escaped -> record
            recorded = true
            break
        end
    end

    sp = Superphoton(x, p, w; tau_abs = tau, tau_scatt = sph.tau_scatt,
                     nscatt = sph.nscatt, x1i = sph.x1i, x2i = sph.x2i,
                     ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    return sp, recorded
end

# =========================================================================
# SCATTERING TRANSPORT (biased exponential transform, Kahn 1950; gpumonty track.cu)
# =========================================================================
#
# Each step accumulates the scattering optical depth dtau_scatt =
# 1/2 [(nu alpha_s)_n + (nu alpha_s)_{n+1}] (h / m_e c^2) dlam, with
# nu alpha_s = nu n_e sigma_h from the hot-cross table, draws x1 = -log(U) and
# an importance bias b, and if b*dtau_scatt > x1 (with w/b > WEIGHT_MIN and
# nscatt < MAX_SCATT) spawns a Compton-scattered child at
# frac = x1/(b*dtau_scatt) into the step with weight
# w(1-e^{-dtau_s})/(1-e^{-b dtau_s}). The incident photon is depleted by the
# physical exp(-dtau_scatt) every step and continues (the unscattered beam), so
# E[spawned weight] = w(1-exp(-dtau_scatt)) — the physical scatter loss — for
# any b >= 1: the bias only reduces variance (-> w/b as b*dtau -> 0).
#
# Absorption is neglected here (idealized Comptonization sphere): synchrotron
# self-absorption exceeds electron scattering by ~1e9-1e12 at the emission
# peak, so a moderate-tau_es sphere has no closed-form seed; neglecting it
# isolates the Compton up-scattering, and the absorption march is validated
# separately against the closed-form thin-sphere spectrum.

# WEIGHT_MIN for the scatter roulette / spawn cut (gpumonty sphere_model: 0).
const _MC_WEIGHT_MIN = 0.0

"""
    bias_func(n_e, R, theta_e, w, nscatt) -> Float64

Importance bias for the exponential-transform scatter decision, sphere form
(arXiv:2602.13198 §3.2; cf. gpumonty): `b = max(1, n_e sigma_T R (n_sc+1)^2)`,
growing with scattering generation `n_sc` so successively rarer higher-order
scatterings stay well-sampled. `theta_e`/`w` are accepted for signature parity
with gpumonty `bias_func` (unused in the sphere form).
"""
@inline function bias_func(n_e, R, theta_e, w, nscatt::Integer)
    b = n_e * SIGMA_THOMSON * R * (Float64(nscatt) + 1.0)^2
    return b > 1.0 ? b : 1.0
end

"""
    _nu_alpha_scatt(model, table, x, kcon) -> Float64

Invariant scattering rate `nu * alpha_scatt = nu n_e sigma_h(eps, Theta_e)`
[cm^-1 s^-1] at coordinate `x` for wavevector `kcon` — the scattering analogue of
`_nu_alpha`. Zero in the vacuum outside the sphere; `sigma_h` from the hot-cross
lookup `table` (Thomson for the soft seed photons, Klein-Nishina for hard
up-scattered photons). `nu = k^t m_e c^2 / h` (static fluid).
"""
@inline function _nu_alpha_scatt(model, table::HotcrossTable,
                                 x::SVector{4}, kcon::SVector{4})
    _inside_sphere(model, x) || return 0.0
    nu = kcon[1] * ME * CL * CL / HPL
    return alpha_inv_scatt(table, nu, model.theta_e, model.n_e)
end

# ------------------------------------------------------------ work queue

"""
    ScatterWorkspace([layout::MCStreamLayout])

Per-thread iterative work queue for the layered (non-recursive) multi-scatter
loop: a LIFO stack of pending superphotons and their RNG streams, plus exact
running diagnostics — total spawned child weight and first-order child weight
(spawned by primaries), one addend per marched photon — and the child count.
A child's stream is `mc_child_rng(parent stream, spawn ordinal, layout)`, a
function of its genealogy only, so the children and their streams do not
depend on which workspace (thread, chunk) processes the primary. The default
layout admits primaries with 32-bit ids and five scattering layers.
"""
mutable struct ScatterWorkspace
    sp::Vector{Superphoton}
    rngs::Vector{MCRng}
    layout::MCStreamLayout
    diag::ExactScalars           # 1: child weight, 2: first-order child weight
    n_children::Int
end

ScatterWorkspace(layout::MCStreamLayout) =
    ScatterWorkspace(Superphoton[], MCRng[], layout, ExactScalars(2), 0)
ScatterWorkspace() = ScatterWorkspace(MCStreamLayout(typemax(UInt32), _MC_TRACK_MAX_STEPS, 5))

"Total weight spawned into children so far (exact sum, rounded once)."
child_weight(ws::ScatterWorkspace) = xsum_round(ws.diag, 1)
"Weight spawned by primaries (first-order children) so far (exact sum, rounded once)."
first_order_weight(ws::ScatterWorkspace) = xsum_round(ws.diag, 2)

"Push a photon (typically a freshly generated primary) and its stream onto the queue."
@inline function push_photon!(ws::ScatterWorkspace, sp::Superphoton, rng::MCRng)
    push!(ws.sp, sp)
    push!(ws.rngs, rng)
    return ws
end

"Pop the most recently pushed (photon, rng) (depth-first: bounded stack depth)."
@inline function pop_photon!(ws::ScatterWorkspace)
    return pop!(ws.sp), pop!(ws.rngs)
end

# Queue a scattered child with its genealogy-keyed stream `crng`.
@inline function _spawn_child!(ws::ScatterWorkspace, child::Superphoton, crng::MCRng)
    mc_stream_layer(crng) <= ws.layout.max_layer ||
        error("scatter layer $(mc_stream_layer(crng)) exceeds the stream layout")
    push!(ws.sp, child)
    push!(ws.rngs, crng)
    ws.n_children += 1
    return ws
end

# Record one marched photon's spawned weight in the exact diagnostics.
@inline function _ws_child_weight!(ws::ScatterWorkspace, child_w::Float64, nscatt::Integer)
    xsum_add!(ws.diag, 1, child_w)
    nscatt == 0 && xsum_add!(ws.diag, 2, child_w)
    return ws
end

"""
    track_superphoton_scatter!(ws, sph, model, table, rng; max_scatt)
        -> (escaped::Superphoton, recorded::Bool, scatter_loss::Float64, rng::MCRng)

Transport one superphoton through the uniform flat sphere with the biased
exponential-transform scatter decision (see the section header), spawning
Compton-scattered children onto `ws`. The march is the shared
`_mc_march_scatter_core` (mc_common.jl); this wrapper supplies the host opacity
lookup and the `ScatterWorkspace` child sink. Returns the escaped incident
photon (`recorded=true` if it left the sphere), the incident weight lost to
scattering (`scatter_loss`, checks unbiasedness) and the advanced RNG.
Absorption is neglected (idealized Comptonization sphere).
"""
function track_superphoton_scatter!(ws::ScatterWorkspace, sph::Superphoton, model,
                                    table::HotcrossTable, rng::MCRng; max_scatt::Int)
    nu_alpha_s = (x, kcon) -> _nu_alpha_scatt(model, table, x, kcon)
    rng0 = rng                                   # this photon's stream identity
    spawn! = (xev, weight_scat, tau_s, srng, event) -> begin
        p_new, _, srng = scatter_superphoton(sph.p, model.ucon, model.ucov, model.bcon,
                                             model.theta_e, model.gcov, model.gcon, srng)
        child = Superphoton(xev, p_new, weight_scat;
                            tau_abs = sph.tau_abs, tau_scatt = tau_s,
                            nscatt = sph.nscatt + Int32(1),
                            x1i = sph.x1i, x2i = sph.x2i,
                            ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
        _spawn_child!(ws, child, mc_child_rng(rng0, event, ws.layout))
        return weight_scat, srng
    end
    mp = (gcon = model.gcon, R = model.R, ne = model.n_e, te = model.theta_e,
          weight_min = _MC_WEIGHT_MIN, max_scatt = max_scatt,
          step_frac = _MC_STEP_FRAC, track_max = _MC_TRACK_MAX_STEPS)
    recorded, x, w, tau_s, scatter_loss, child_w, rng =
        _mc_march_scatter_core(sph.x, sph.p, sph.w, sph.tau_scatt, sph.nscatt, rng,
                               mp, nu_alpha_s, spawn!)
    _ws_child_weight!(ws, child_w, sph.nscatt)
    sp = Superphoton(x, sph.p, w; tau_abs = sph.tau_abs, tau_scatt = tau_s,
                     nscatt = sph.nscatt, x1i = sph.x1i, x2i = sph.x2i,
                     ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    return sp, recorded, scatter_loss, rng
end
