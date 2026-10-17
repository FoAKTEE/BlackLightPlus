# Invariant transfer coefficients of one decoded sample: the pitch angle and the
# fluid-frame frequency of the photon, then the thermal, power-law and kappa fits added
# in that order (blacklight CalculatePolarizedCoefficients). The photon wave covector is
# k = -p of the ray state, future-directed and normalised to k_t = -1 at the camera, so
# the fluid-frame frequency is ν = -k·u ν_obs and cos θ_B is signed along the propagation.

const _ZERO4S = SVector(0.0, 0.0, 0.0, 0.0)
const _ZERO4x4 = zero(SMatrix{4,4,Float64})

"""
    kinematic(::Type{V}, x)

`x` (a scalar or an `SVector`) in the number type `V` of the photon wave vector and
the metric. The kinematic fields of a `PlasmaState` (`ucon`, `ucov`, `bcon`, `bcov`,
`b_sq`) are functions of the primitives and the metric only; when the state's number
type is promoted by a derivative that the parameters alone carry (a `Dual` mass
scale or electron temperature ratio), their partials are zero by construction and
dropping them is exact. A derivative carried by the primitives or the metric must
therefore be carried by `V` too. The identity when `x` already has type `V`.
"""
@inline kinematic(::Type{V}, x::V) where {V} = x
@inline kinematic(::Type{V}, x::Real) where {V} = V(primal(x))
@inline kinematic(::Type{V}, v::SVector{4,V}) where {V} = v
@inline kinematic(::Type{V}, v::SVector{4}) where {V} =
    SVector{4,V}(primal(v[1]), primal(v[2]), primal(v[3]), primal(v[4]))

"""
    coefficients(::StokesI, p::TransferParams, state::PlasmaState, kcov, ν_obs) -> (j, α)

Invariant Stokes-I emissivity `j = j_ν/ν²` and absorptivity `α = α_ν ν` (cgs) of the
decoded sample `state` for the photon wave covector `kcov` at observing frequency
`ν_obs`. The pitch angle comes from the invariant identity
`cos²θ_B = (k·b)² / ((k·u)² b²)` (b·u = 0, k null), clipped at 1, with no tetrad;
the fluid-frame frequency is `-k·u ν_obs`. The thermal fit (`p.electrons.thermal_frac`)
is followed by the power-law and kappa fits when present. An invalid state returns
zeros of the promoted unit and electron-model type, which [`segment`](@ref) treats as a
no-op. The kinematics are read through [`kinematic`](@ref) in the type of `kcov`.
"""
@inline function coefficients(::StokesI, p::TransferParams, state::PlasmaState,
                              kcov::SVector{4}, ν_obs)
    z = zero(_zero_type(p))
    state.valid || return (z, z)
    V = eltype(kcov)
    ucon = kinematic(V, state.ucon)
    bcon = kinematic(V, state.bcon)
    b_sq = kinematic(V, state.b_sq)
    minus_ku = -dot(kcov, ucon)
    kb = dot(kcov, bcon)
    cos2 = min(kb * kb / (minus_ku * minus_ku * b_sq), 1.0)
    sin_th = sqrt(1.0 - cos2)
    nu = minus_ku * ν_obs
    j, a = thermal_synchrotron(p.electrons.thermal_frac, state.n_e, state.theta_e, state.B,
                               nu, sin_th)
    if has_powerlaw(p)
        jp, ap = power_law_synchrotron(p.powerlaw, state.n_e, state.B, nu, sin_th)
        j += jp; a += ap
    end
    if has_kappa(p)
        jk, ak = kappa_synchrotron(p.kappa, state.n_e, state.B, nu, sin_th)
        j += jk; a += ak
    end
    return (j, a)
end

"""
    coefficients(::IQUV, p::TransferParams, state::PlasmaState, kcov, kcon, g, ν_obs,
                 renorm = false) -> (j::SVector{4}, α::SVector{4}, ρ::SVector{4}, tet, valid)

Invariant polarized coefficients of the decoded sample `state` in the fluid tetrad
`tet` (`fluid_tetrad` with `e_(2)` along `b`, returned for the Stokes rotation into
the readout frame), for the photon wave vector `kcov`/`kcon` and the metric pair
`g = (gcov, gcon)` at the sample. `j`, `α`, `ρ` are ordered `(I, Q, U, V)` with the U
components zero by the tetrad choice (ρ_I is zero too). The pitch angle is taken in
the tetrad with `cos θ_B` signed; the thermal, power-law and kappa contributions are
added in that order. `renorm = true` renormalises `e_(1)` off single Kerr-Schild
(`|det g| ≠ 1`); the pitch angle does not depend on it. An invalid state returns
`Float64` zeros, a zero tetrad and `valid = false`.
"""
@inline function coefficients(::IQUV, p::TransferParams, state::PlasmaState,
                              kcov::SVector{4}, kcon::SVector{4}, g, ν_obs,
                              renorm::Bool = false)
    state.valid || return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false
    gcov, gcon = g
    V = promote_type(eltype(kcov), eltype(kcon), eltype(gcov), eltype(gcon))
    ucon = kinematic(V, state.ucon); ucov = kinematic(V, state.ucov)
    bcon = kinematic(V, state.bcon); bcov = kinematic(V, state.bcov)
    n_e_cgs = state.n_e; bb_cgs = state.B; theta_e = state.theta_e

    tet = fluid_tetrad(ucon, ucov, kcon, kcov, bcon, gcov, gcon; renorm_e1 = renorm)
    kt1 = kt2 = kt3 = 0.0
    bt1 = bt2 = bt3 = 0.0
    @inbounds for μ in 1:4
        kt1 += tet[2, μ] * kcov[μ]; kt2 += tet[3, μ] * kcov[μ]; kt3 += tet[4, μ] * kcov[μ]
        bt1 += tet[2, μ] * bcov[μ]; bt2 += tet[3, μ] * bcov[μ]; bt3 += tet[4, μ] * bcov[μ]
    end
    k2t = kt1^2 + kt2^2 + kt3^2
    b2t = bt1^2 + bt2^2 + bt3^2
    kbt = kt1 * bt1 + kt2 * bt2 + kt3 * bt3
    cos2 = min(kbt * kbt / (k2t * b2t), 1.0)
    sin_th = sqrt(1.0 - cos2)
    cos_th = sqrt(cos2) * (kbt >= 0.0 ? 1.0 : -1.0)

    minus_ku = 0.0
    @inbounds for μ in 1:4
        minus_ku -= kcov[μ] * ucon[μ]
    end
    nu_cgs = minus_ku * ν_obs

    ji, jq, jv, ai, aq, av, rq, rv =
        thermal_synchrotron_polarized(p.electrons.thermal_frac, n_e_cgs, theta_e,
                                      bb_cgs, nu_cgs, sin_th, cos_th)
    if has_powerlaw(p)
        t = power_law_synchrotron_polarized(p.powerlaw, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    if has_kappa(p)
        t = kappa_synchrotron_polarized(p.kappa, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    return SVector(ji, jq, 0.0, jv), SVector(ai, aq, 0.0, av), SVector(0.0, rq, 0.0, rv), tet, true
end
