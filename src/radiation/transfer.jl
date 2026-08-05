# Constant-coefficient polarized (Stokes I,Q,U,V) plasma coupling over one segment
# (blacklight polarized_transfer.hpp StokesCoupleNoSplit) plus the scalar
# exponential-propagator step of unpolarized_transfer.hpp. Coefficients are the
# Lorentz invariants j (=j_nu/nu^2), alpha (=alpha_nu*nu), rho (=rho_nu*nu); U
# components are zero by the tetrad choice, so only the Q (index 2) and V (index 4)
# entries enter. stokes_couple evaluates exp(-K ds)(S0 - K^{-1} j) + K^{-1} j
# analytically for the Mueller matrix (LD85 / ipole convention, U rows/cols zero)
#   K = [ alpha_I alpha_Q 0 alpha_V ; alpha_Q alpha_I rho_V 0 ;
#         0 -rho_V alpha_I rho_Q ; alpha_V 0 -rho_Q alpha_I ].

"Optical-depth cap above which a segment is treated as saturated (BlackLight delta_tau_max)."
const POL_DELTA_TAU_MAX = 100.0

# ------------------------------------------------------------------ detail helpers

# Total absorptivity only (no polarized absorptivity, no rotativity): exponential decay
# toward the source function on every Stokes component. Also the last-resort fallback
# of the degenerate full branch.
@inline function _couple_total_absorption(S::SVector{4}, j::SVector{4},
                                          alpha_i, delta_tau, optically_thin::Bool)
    if optically_thin
        exp_neg = exp(-delta_tau)
        expm1_val = expm1(delta_tau)
        return exp_neg .* (S .+ j ./ alpha_i .* expm1_val)
    else
        return j ./ alpha_i
    end
end

# No absorptivity but nonzero rotativity (I A2-A5). Includes the +j*ds emission term.
@inline function _couple_pure_rotation(S::SVector{4}, j::SVector{4},
                                       ρ::SVector{4}, rho_sq, rho_p, ds)
    cos_rho = cos(rho_p * ds)
    sin_rho = sin(rho_p * ds)
    s2 = sin(rho_p * ds / 2.0)
    sin_sq_rho = s2 * s2
    rho_ss = ρ[2] * S[2] + ρ[4] * S[4]
    se1 = S[1]
    se2 = S[2] * cos_rho + 2.0 * ρ[2] * rho_ss / rho_sq * sin_sq_rho - ρ[4] * S[3] / rho_p * sin_rho
    se3 = S[3] * cos_rho + (ρ[4] * S[2] - ρ[2] * S[4]) / rho_p * sin_rho
    se4 = S[4] * cos_rho + 2.0 * ρ[4] * rho_ss / rho_sq * sin_sq_rho + ρ[2] * S[3] / rho_p * sin_rho
    return SVector(se1, se2, se3, se4) .+ j .* ds
end

# Nonzero absorptivity AND rotativity (I 24 / L 10): the full 4x4 analytic solution.
# Known index typos in blacklight's mm_2/mm_3 ([0][2]/[1][3]) and mm_4 signs are
# corrected; with alpha_U = rho_U = 0 those mm_2/mm_3 entries are genuinely zero.
@inline function _couple_full(S::SVector{4}, j::SVector{4}, α::SVector{4},
                              ρ::SVector{4}, alpha_sq, rho_sq,
                              alpha_p, delta_tau, ds, optically_thin::Bool)
    aI = α[1]
    aQ = α[2]; aV = α[4]
    rQ = ρ[2]; rV = ρ[4]
    alpha_rho = aQ * rQ + aV * rV
    alpha_sq_rho_sq = alpha_sq - rho_sq
    lambda_a = sqrt(alpha_sq_rho_sq * alpha_sq_rho_sq / 4.0 + alpha_rho * alpha_rho)
    lambda_b = alpha_sq_rho_sq / 2.0
    lambda_1 = sqrt(lambda_a + lambda_b)
    lambda_2 = sqrt(lambda_a - lambda_b)
    coefficient_theta = lambda_1 * lambda_1 + lambda_2 * lambda_2
    s = alpha_rho >= 0.0 ? 1.0 : -1.0

    # Deviation from blacklight: guard the degenerate eigenstructure so a single
    # singular sample cannot NaN the whole ray — fall back to total-absorptivity
    # coupling (the references' CoupleAbsorptionNoRotation fallback disagrees with
    # matrix-exponential reference solutions on emitting slabs and is not used);
    # the caller's admissibility clamp bounds the error.
    k_rel_eps = 1.0e-12
    a0_sq = aI * aI
    degenerate = coefficient_theta <= k_rel_eps * (alpha_sq + rho_sq) ||
                 abs(a0_sq - lambda_1 * lambda_1) <= k_rel_eps * (a0_sq + lambda_1 * lambda_1)
    if degenerate
        return _couple_total_absorption(S, j, aI, delta_tau, optically_thin)
    end

    inv_theta = 1.0 / coefficient_theta

    m2_12 = lambda_2 * aQ - s * lambda_1 * rQ
    m2_14 = lambda_2 * aV - s * lambda_1 * rV
    m2_23 = s * lambda_1 * aV + lambda_2 * rV
    m2_34 = s * lambda_1 * aQ + lambda_2 * rQ
    mm2 = inv_theta .* @SMatrix [0.0    m2_12  0.0    m2_14;
                                 m2_12  0.0    m2_23  0.0;
                                 0.0   -m2_23  0.0    m2_34;
                                 m2_14  0.0   -m2_34  0.0]

    m3_12 = lambda_1 * aQ + s * lambda_2 * rQ
    m3_14 = lambda_1 * aV + s * lambda_2 * rV
    m3_23 = -(s * lambda_2 * aV - lambda_1 * rV)
    m3_34 = -(s * lambda_2 * aQ - lambda_1 * rQ)
    mm3 = inv_theta .* @SMatrix [0.0    m3_12  0.0    m3_14;
                                 m3_12  0.0    m3_23  0.0;
                                 0.0   -m3_23  0.0    m3_34;
                                 m3_14  0.0   -m3_34  0.0]

    d0 = (alpha_sq + rho_sq) / 2.0
    m4_11 = d0
    m4_22 = aQ * aQ + rQ * rQ - d0
    m4_33 = -d0
    m4_44 = aV * aV + rV * rV - d0
    m4_13 = aQ * rV - aV * rQ
    m4_24 = aV * aQ + rV * rQ
    mm4 = (2.0 * inv_theta) .* @SMatrix [m4_11  0.0    m4_13  0.0;
                                         0.0    m4_22  0.0    m4_24;
                                        -m4_13  0.0    m4_33  0.0;
                                         0.0    m4_24  0.0    m4_44]

    mm1 = one(SMatrix{4,4,Float64})
    mm1p4 = mm1 .+ mm4
    mm1m4 = mm1 .- mm4

    f_1 = 1.0 / (a0_sq - lambda_1 * lambda_1)
    f_2 = 1.0 / (a0_sq + lambda_2 * lambda_2)
    cosh_term = (-lambda_1 * f_1) .* mm3 .+ (0.5 * aI * f_1) .* mm1p4
    cos_term = (-lambda_2 * f_2) .* mm2 .+ (0.5 * aI * f_2) .* mm1m4
    pp = cosh_term .+ cos_term

    if optically_thin
        exp_v = exp(-delta_tau)
        sin_v = sin(lambda_2 * ds)
        cos_v = cos(lambda_2 * ds)
        sinh_v = sinh(lambda_1 * ds)
        cosh_v = cosh(lambda_1 * ds)
        sin_term = (-aI * f_2) .* mm2 .+ (-0.5 * lambda_2 * f_2) .* mm1m4
        sinh_term = (-aI * f_1) .* mm3 .+ (0.5 * lambda_1 * f_1) .* mm1p4
        pp = pp .- exp_v .* (cosh_v .* cosh_term .+ cos_v .* cos_term .+
                             sin_v .* sin_term .+ sinh_v .* sinh_term)
        oo = exp_v .* (0.5 .* cosh_v .* mm1p4 .+ 0.5 .* cos_v .* mm1m4 .-
                       sin_v .* mm2 .- sinh_v .* mm3)
        return pp * j .+ oo * S
    else
        return pp * j
    end
end

# ------------------------------------------------------------------ public API

"""
    stokes_couple(S, j, α, ρ, ds; dtau_max=100.0) -> SVector{4}

Advance the orthonormal-frame Stokes 4-vector `S` (I,Q,U,V) over one segment of
invariant path length `ds` through plasma with invariant coefficients `j`, `α`, `ρ`
(each an `SVector{4}`; the U component is zero). The analytic constant-coefficient
solution (StokesCoupleNoSplit): emission-only, total absorption, pure rotation,
polarized absorption without rotation, or the full absorption+rotation branch with
the degeneracy guard. The result is clamped to be physically admissible (I >= 0,
|P| <= I).
"""
function stokes_couple(S::SVector{4}, j::SVector{4}, α::SVector{4},
                       ρ::SVector{4}, ds; dtau_max = POL_DELTA_TAU_MAX)
    alpha_sq = α[2] * α[2] + α[4] * α[4]
    alpha_p = sqrt(alpha_sq)
    # Mueller-operator admissibility (LD85): a physical K needs alpha_I >= alpha_P =
    # sqrt(alpha_Q^2 + alpha_V^2). A synchrotron fit violating this at a domain edge
    # makes the full branch carry exp((alpha_P - alpha_I) ds) — unbounded amplification
    # the trailing |P| <= I clamp cannot rein in. Rescale the polarized absorptivities
    # onto the physical cone (preserving alpha_I and the polarization direction) so
    # lambda_1 <= alpha_I and the segment strictly decays.
    if α[1] > 0.0 && alpha_p > α[1]
        sc = α[1] / alpha_p
        α = SVector(α[1], α[2] * sc, α[3] * sc, α[4] * sc)
        alpha_sq = α[1] * α[1]
        alpha_p = α[1]
    end
    rho_sq = ρ[2] * ρ[2] + ρ[4] * ρ[4]
    rho_p = sqrt(rho_sq)
    delta_tau = α[1] * ds
    optically_thin = delta_tau <= dtau_max

    # Deviation from blacklight/ipole: their dedicated rho_p == 0 branch
    # (CoupleAbsorptionNoRotation, I A14-A17) disagrees with matrix-exponential/RK4
    # reference solutions on emitting slabs; with rho = 0 the full branch reduces
    # analytically to the correct no-rotation solution (lambda_1 = alpha_p,
    # lambda_2 = 0, coefficient_theta = alpha_p^2 != 0), so it is routed there.
    se = if α[1] == 0.0 && rho_p == 0.0
        S .+ j .* ds
    elseif alpha_p == 0.0 && rho_p == 0.0
        _couple_total_absorption(S, j, α[1], delta_tau, optically_thin)
    elseif α[1] == 0.0
        _couple_pure_rotation(S, j, ρ, rho_sq, rho_p, ds)
    else
        _couple_full(S, j, α, ρ, alpha_sq, rho_sq, alpha_p, delta_tau, ds, optically_thin)
    end

    # Physical admissibility: I >= 0 and |P| <= I.
    ii = max(se[1], 0.0)
    pol = se[2] * se[2] + se[3] * se[3] + se[4] * se[4]
    if pol > ii * ii
        factor = sqrt(ii * ii / pol)
        return SVector(ii, se[2] * factor, se[3] * factor, se[4] * factor)
    end
    return SVector(ii, se[2], se[3], se[4])
end

"""
    couple_unpolarized(I, j, α, ds; dtau_max=100.0) -> Float64

One exact-segment update of the invariant intensity through plasma with emissivity
`j` and absorptivity `α` over invariant path length `ds` (UnpolarizedTransferStep):
`I e^{-dτ} + (j/α)(1 - e^{-dτ})` with `dτ = α ds`, capped at `dtau_max` (saturate to
`j/α`), and the `α <= 0` emission-only limit `I + j ds`.
"""
@inline function couple_unpolarized(intensity, j, α, ds;
                                    dtau_max = POL_DELTA_TAU_MAX)
    if α > 0.0
        ss = j / α
        delta_tau = α * ds
        if delta_tau < 1.0e-3
            # Thin-segment Taylor form (ipole approximate_solve): the exact
            # I e^{-dτ} + ss(1 - e^{-dτ}) expanded to O(dτ³). Avoids two transcendentals
            # and the catastrophic cancellation of I e^{-dτ} - I at tiny dτ.
            return intensity + (ss - intensity) * delta_tau *
                   (1.0 - delta_tau / 2.0 * (1.0 - delta_tau / 3.0))
        end
        if delta_tau <= dtau_max
            return exp(-delta_tau) * (intensity + ss * expm1(delta_tau))
        end
        return ss
    end
    return intensity + j * ds
end
