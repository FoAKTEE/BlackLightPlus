# Radiation pipeline: per-ray fluid sampling -> invariant coefficients -> transfer
# march -> Stokes images. Both modes run the reverse (camera -> source) marches of
# reverse_march.jl over the stored ray states (a `StoredPath`): Stokes I with the exact
# segment operators of `unpolarized_segment`, IQUV with the Mueller chain of
# `stokes_segment` in the fluid tetrad, rotated into the camera polarization frame
# parallel-transported along the ray. Ray states carry p = -k (camera.jl), so every
# sample reads the photon wave covector as k_μ = -p_μ; k_t = -1 at the camera ⇒
# delta_lambda_cgs = dλ_code · L_unit / ν_obs; nonthermal electrons (power-law + kappa)
# enter additively in the 8 coefficients.

"""
    eight_coefficients(cfg, rho, pgas, fv, gcov, gcon, kcov, kcon, ν_obs, renorm_e1=false)
        -> (j::SVector{4}, α::SVector{4}, ρ::SVector{4}, tet, valid::Bool)

Invariant polarized coefficients at one sample: thermal + power-law + kappa
additive mixing (blacklight CalculatePolarizedCoefficients). `cfg` is a
`SimCoeffConfig` or its isbits `coefficient_params` set (the device kernels call this
same function). `kcov`/`kcon` are the
photon wave vector (future-directed, source → camera; `-p` of a ray state), so the
fluid-frame frequency is ν = -k·u ν_obs and cos θ_B is signed along the propagation. `rho`/`pgas` are
code-unit scalars; `fv` is the NamedTuple from `primitive_four_vectors`. Pass
`renorm_e1 = true` off single CKS so the returned fluid tetrad's e_(1) (hence the
Q/U frame) is unit.
"""
@inline function eight_coefficients(cfg, rho, pgas,
                                    fv, gcov, gcon, kcov::SVector{4}, kcon::SVector{4},
                                    ν_obs, renorm_e1::Bool = false)
    (rho <= 0.0 || pgas <= 0.0) && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false
    b_sq = fv.b_sq
    b_sq <= 0.0 && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false   # b² < 0 is round-off

    pd = plasma_decode(cfg, rho, pgas, b_sq)
    pd.omit && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false
    n_e_cgs = pd.n_e_cgs; bb_cgs = pd.bb_cgs; theta_e = pd.theta_e

    # Orthonormal-frame pitch angle with SIGNED cos(theta_B). The fluid tetrad is
    # returned so polarized_march can reuse it for the plasma coupling instead of
    # rebuilding it per sample. renorm_e1 = true off single CKS (|det g| != 1):
    # without it e_(1) carries norm 1/|det g| and the Q/U Stokes frame is wrong;
    # the pitch angle is unaffected (b lies in span{e0,e2,e3}).
    tet = fluid_tetrad(fv.ucon, fv.ucov, kcon, kcov, fv.bcon, gcov, gcon;
                       renorm_e1 = renorm_e1)
    kt1 = kt2 = kt3 = 0.0
    bt1 = bt2 = bt3 = 0.0
    @inbounds for μ in 1:4
        kt1 += tet[2, μ] * kcov[μ]; kt2 += tet[3, μ] * kcov[μ]; kt3 += tet[4, μ] * kcov[μ]
        bt1 += tet[2, μ] * fv.bcov[μ]; bt2 += tet[3, μ] * fv.bcov[μ]; bt3 += tet[4, μ] * fv.bcov[μ]
    end
    k2t = kt1^2 + kt2^2 + kt3^2
    b2t = bt1^2 + bt2^2 + bt3^2
    kbt = kt1 * bt1 + kt2 * bt2 + kt3 * bt3
    cos2 = min(kbt * kbt / (k2t * b2t), 1.0)
    sin_th = sqrt(1.0 - cos2)
    cos_th = sqrt(cos2) * (kbt >= 0.0 ? 1.0 : -1.0)

    minus_ku = 0.0
    @inbounds for μ in 1:4
        minus_ku -= kcov[μ] * fv.ucon[μ]
    end
    nu_cgs = minus_ku * ν_obs                    # momentum_factor = 1 (k_t = -1)

    ji, jq, jv, ai, aq, av, rq, rv =
        thermal_synchrotron_polarized(cfg.plasma_thermal_frac, n_e_cgs, theta_e,
                                      bb_cgs, nu_cgs, sin_th, cos_th)
    if has_power_law(cfg)
        t = power_law_synchrotron_polarized(cfg, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    if has_kappa(cfg)
        t = kappa_synchrotron_polarized(cfg, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    return SVector(ji, jq, 0.0, jv), SVector(ai, aq, 0.0, av), SVector(0.0, rq, 0.0, rv), tet, true
end

const _ZERO4S = SVector(0.0, 0.0, 0.0, 0.0)
const _ZERO4x4 = zero(SMatrix{4,4,Float64})

"""
    unpolarized_coefficients(cfg, rho, pgas, fv, kcov, ν_obs) -> (j, α)

Invariant Stokes-I emissivity and absorptivity at one sample: the plasma decode and
cuts of `eight_coefficients`, the pitch angle from the invariant identity
cos²θ_B = (k·b)² / ((k·u)² b²) (b·u = 0, k null: the fluid-frame spatial momentum has
norm -k·u; the traced k's null-constraint drift enters at relative order
k·k/(k·u)², below 1e-7 on the C = 80 march), and the thermal + power-law + kappa
unpolarized fits. No tetrad is built. A vacuum, unmagnetized or cut sample returns zeros, which the
segment operator treats as a no-op. `cfg` is a `SimCoeffConfig` (AD `Dual` configs
included) or its `coefficient_params` set; the zeros take the unit/plasma type so a
`Dual` accumulation is type-stable.
"""
@inline function unpolarized_coefficients(cfg, rho, pgas, fv, kcov::SVector{4}, ν_obs)
    z = zero(promote_type(typeof(cfg.d_unit), typeof(cfg.e_unit), typeof(cfg.b_unit),
                          typeof(cfg.plasma_rat_high), typeof(cfg.plasma_rat_low)))
    (rho <= 0.0 || pgas <= 0.0) && return (z, z)
    b_sq = fv.b_sq
    b_sq <= 0.0 && return (z, z)                        # b² < 0 is round-off
    pd = plasma_decode(cfg, rho, pgas, b_sq)
    pd.omit && return (z, z)
    minus_ku = -dot(kcov, fv.ucon)                      # = ν_local / ν_obs (k_t = -1)
    kb = dot(kcov, fv.bcon)
    cos2 = min(kb * kb / (minus_ku * minus_ku * b_sq), 1.0)
    sin_th = sqrt(1.0 - cos2)
    nu = minus_ku * ν_obs
    j, a = thermal_synchrotron(cfg.plasma_thermal_frac, pd.n_e_cgs, pd.theta_e, pd.bb_cgs,
                               nu, sin_th)
    if has_power_law(cfg)
        jp, ap = power_law_synchrotron(cfg, pd.n_e_cgs, pd.bb_cgs, nu, sin_th)
        j += jp; a += ap
    end
    if has_kappa(cfg)
        jk, ak = kappa_synchrotron(cfg, pd.n_e_cgs, pd.bb_cgs, nu, sin_th)
        j += jk; a += ak
    end
    return (j, a)
end

# Unit "normal"/Eulerian observer n^μ = -α g^{0μ} at the camera (α = lapse):
# n·n = -1 exactly; the spatial part is +g^{0i}/g^{00}/α = -β^i/α (β^i = -g^{0i}/g^{00}).
@inline function _normal_observer(gcon::SMatrix{4,4})
    lapse = 1.0 / sqrt(-gcon[1, 1])
    return SVector(1.0 / lapse, gcon[1, 2] / gcon[1, 1] / lapse,
                   gcon[1, 3] / gcon[1, 1] / lapse, gcon[1, 4] / gcon[1, 1] / lapse)
end

# The per-ray march parameter set of a SimCoeffConfig (field names as coefficient_params)
# plus the observing frequency, length unit and camera vertical.
_march_params(cfg::SimCoeffConfig, ν_obs, L_unit, cam_up) =
    merge(coefficient_params(cfg), (nu_obs = ν_obs, L_unit = L_unit, cu1 = cam_up[1],
                                    cu2 = cam_up[2], cu3 = cam_up[3], cu4 = cam_up[4]))

# `sampler(x)` NamedTuple (rho, pgas, uu, bb, inside) -> the marches' primitive vector
# (rho, u1, u2, u3, eint, b1, b2, b3); outside the fluid or with zero field it is vacuum.
@inline function _stored_sample(sampler, gamma)
    return y -> begin
        s = sampler(SVector(y[1], y[2], y[3], y[4]))
        if s.inside && !(s.bb[1] == 0.0 && s.bb[2] == 0.0 && s.bb[3] == 0.0)
            SVector(s.rho, s.uu[1], s.uu[2], s.uu[3], s.pgas / (gamma - 1),
                    s.bb[1], s.bb[2], s.bb[3])
        else
            zero(SVector{8,promote_type(typeof(s.rho), typeof(s.pgas / (gamma - 1)))})
        end
    end
end

"""
    polarized_march(st, cfg, sampler, ys, dls, ν_obs, L_unit;
                    cam_up, intensity_only=false) -> SVector{4}

Stokes (I, Q, U, V) of ONE ray at ν_obs (physical, scaled by ν_obs³) from its stored
trajectory `ys`/`dls` (the `trace_rays`/`trace_rays_sampled` outputs, source → camera):
`reverse_polarized_march` along the `StoredPath`, the march of the fused kernels.
`sampler(x)` returns the NamedTuple (rho, pgas, uu, bb, inside) at position x (code
units); `cam_up` is the camera vertical. `intensity_only` keeps only the Stokes-I
coefficients (diagnostic).
"""
function polarized_march(st, cfg::SimCoeffConfig, sampler, ys::Vector{SVector{8,Float64}},
                         dls::Vector{Float64}, ν_obs::Float64, L_unit::Float64;
                         cam_up::SVector{4,Float64}, intensity_only::Bool=false)
    gp = merge(_march_params(cfg, ν_obs, L_unit, cam_up), (intensity_only = intensity_only,))
    return reverse_polarized_march(ys[end], _stored_sample(sampler, cfg.plasma_gamma), gp,
                                   StoredPath(st, ys, dls), length(ys))
end

"""
    unpolarized_march(st, cfg, sampler, ys, dls, ν_obs, L_unit) -> I_ν

Stokes I of ONE ray from its stored trajectory: `reverse_unpolarized_march` along the
`StoredPath` (exact segment operators, blacklight unpolarized_transfer). The
accumulator takes the config's scalar type, so ForwardDiff Dual sensitivities flow.
"""
function unpolarized_march(st, cfg::SimCoeffConfig{T}, sampler, ys, dls,
                           ν_obs, L_unit) where {T}
    gp = _march_params(cfg, ν_obs, L_unit, SVector(0.0, 0.0, 0.0, 0.0))
    return reverse_unpolarized_march(ys[end], _stored_sample(sampler, cfg.plasma_gamma), T,
                                     gp, StoredPath(st, ys, dls), length(ys))
end
