# Synchrotron emission / absorption / rotativity coefficients for the simulation
# (GRMHD) model, following blacklight radiation/simulation_coefficients.{hpp,cpp}
# and radiation/polarized_coefficients.hpp. Fits from Marszewski et al. 2021
# (ApJ 921, 17; "M" equation tags below); see also Dexter 2016 and Pandya et al.
# 2016. Everything is in the Lorentz-invariant forms consumed by the transfer
# march:
#   j_inv = j_nu / nu^2,   alpha_inv = alpha_nu * nu,   rho_inv = rho_nu * nu.
# All physics arithmetic is Float64, using the CGS constants from the Constants
# module (CL, EE, ME, MP, KB, HPL).

# ================================================================ Bessel functions
# Abramowitz & Stegun 9.8 polynomial fits (device-safe). Used by the thermal
# polarized Faraday rotativities (argument 1/theta_e).

"Modified Bessel function I_0(x) (A&S 9.8.1/9.8.2)."
@inline function bessel_i0(x)
    ax = abs(x)
    if ax < 3.75
        t2 = (x / 3.75)^2
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 +
                 t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    end
    t = 3.75 / ax
    return (exp(ax) / sqrt(ax)) *
        (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 +
          t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 +
          t * (-0.01647633 + t * 0.00392377))))))))
end

"Modified Bessel function I_1(x) (A&S 9.8.3/9.8.4)."
@inline function bessel_i1(x)
    ax = abs(x)
    local ans
    if ax < 3.75
        t2 = (x / 3.75)^2
        ans = ax * (0.5 + t2 * (0.87890594 + t2 * (0.51498869 + t2 * (0.15084934 +
                t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
    else
        t = 3.75 / ax
        ans = 0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))
        ans = 0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 +
                t * (-0.01031555 + t * ans))))
        ans *= exp(ax) / sqrt(ax)
    end
    return x < 0.0 ? -ans : ans
end

"Modified Bessel function K_0(x), x > 0 (A&S 9.8.5/9.8.6)."
@inline function bessel_k0(x)
    if x <= 2.0
        t = x * x / 4.0
        return -log(x / 2.0) * bessel_i0(x) +
            (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590 +
              t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))))
    end
    t = 2.0 / x
    return (exp(-x) / sqrt(x)) *
        (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 +
          t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))))
end

"Modified Bessel function K_1(x), x > 0 (A&S 9.8.7/9.8.8)."
@inline function bessel_k1(x)
    if x <= 2.0
        t = x * x / 4.0
        return log(x / 2.0) * bessel_i1(x) +
            (1.0 / x) * (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897 +
              t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
    end
    t = 2.0 / x
    return (exp(-x) / sqrt(x)) *
        (1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268 +
          t * (-0.00780353 + t * (0.00325614 + t * (-0.00068245)))))))
end

"Modified Bessel function K_2(x), x > 0, via K_2 = K_0 + (2/x) K_1."
@inline bessel_k2(x) = bessel_k0(x) + (2.0 / x) * bessel_k1(x)

# ================================================================ configuration

"""
    SimCoeffConfig{T<:Real}

Scalar configuration for the simulation synchrotron model (blacklight defaults).
Mutable so `precompute_power_law!` / `precompute_kappa!` can fill the derived
Gamma/2F1/Bessel constants in place; plasma-value cuts use the "-1 disables"
convention. Parametric over the scalar type `T` (default `Float64`): a ForwardDiff
`Dual` element type lets AD sensitivities w.r.t. any config scalar (e.g. `d_unit`,
`e_unit`, `b_unit`) flow through the transfer pipeline.
"""
Base.@kwdef mutable struct SimCoeffConfig{T<:Real}
    bh_a::T = 0.0
    d_unit::T = 0.0
    e_unit::T = 0.0
    b_unit::T = 0.0
    plasma_mu::T = 0.5
    plasma_ne_ni::T = 1.0
    plasma_use_p::Int = 1
    plasma_rat_low::T = 1.0
    plasma_rat_high::T = 10.0
    plasma_gamma::T = 5.0 / 3.0
    plasma_gamma_i::T = 5.0 / 3.0
    plasma_gamma_e::T = 4.0 / 3.0
    plasma_thermal_frac::T = 1.0

    plasma_power_frac::T = 0.0
    plasma_p::T = 0.0
    plasma_kappa_frac::T = 0.0
    plasma_kappa::T = 0.0
    plasma_w::T = 0.0

    power_jj::T = 0.0
    power_aa::T = 0.0

    kappa_jj_low::T = 0.0
    kappa_jj_high::T = 0.0
    kappa_jj_x_i::T = 1.0
    kappa_aa_low::T = 0.0
    kappa_aa_high::T = 0.0
    kappa_aa_high_i::T = 0.0
    kappa_aa_x_i::T = 1.0

    plasma_gamma_min::T = 0.0

    power_jj_q::T = 0.0
    power_jj_v::T = 0.0
    power_aa_q::T = 0.0
    power_aa_v::T = 0.0
    power_rho::T = 0.0
    power_rho_q::T = 0.0
    power_rho_v::T = 0.0

    kappa_jj_low_q::T = 0.0
    kappa_jj_low_v::T = 0.0
    kappa_jj_high_q::T = 0.0
    kappa_jj_high_v::T = 0.0
    kappa_jj_x_q::T = 1.0
    kappa_jj_x_v::T = 1.0
    kappa_aa_low_q::T = 0.0
    kappa_aa_low_v::T = 0.0
    kappa_aa_high_q::T = 0.0
    kappa_aa_high_v::T = 0.0
    kappa_aa_x_q::T = 1.0
    kappa_aa_x_v::T = 1.0

    kappa_rho_v::T = 0.0
    kappa_rho_frac::T = 0.0
    kappa_rho_q_low_a::T = 0.0
    kappa_rho_q_low_b::T = 0.0
    kappa_rho_q_low_c::T = 0.0
    kappa_rho_q_low_d::T = 0.0
    kappa_rho_q_low_e::T = 0.0
    kappa_rho_q_high_a::T = 0.0
    kappa_rho_q_high_b::T = 0.0
    kappa_rho_q_high_c::T = 0.0
    kappa_rho_q_high_d::T = 0.0
    kappa_rho_q_high_e::T = 0.0
    kappa_rho_v_low_a::T = 0.0
    kappa_rho_v_low_b::T = 0.0
    kappa_rho_v_high_a::T = 0.0
    kappa_rho_v_high_b::T = 0.0

    cut_rho_min::T = -1.0
    cut_rho_max::T = -1.0
    cut_n_e_min::T = -1.0
    cut_n_e_max::T = -1.0
    cut_p_gas_min::T = -1.0
    cut_p_gas_max::T = -1.0
    cut_theta_e_min::T = -1.0
    cut_theta_e_max::T = -1.0
    cut_b_min::T = -1.0
    cut_b_max::T = -1.0
    cut_sigma_min::T = -1.0
    cut_sigma_max::T = -1.0
    cut_beta_inverse_min::T = -1.0
    cut_beta_inverse_max::T = -1.0
end

# ================================================================ plasma cuts / theta_e

"""
    plasma_cut_omits(c, rho_cgs, n_e_cgs, pgas_cgs, theta_e, bb_cgs, sigma, beta_inv) -> Bool

Per-sample plasma-value cut test (blacklight PlasmaCutOmits). Returns
`true` when the sample should be OMITTED (treated as vacuum). Each bound is guarded by
`>= 0.0` so a negative threshold (canonically -1.0) disables it; 0.0 is active. All
comparisons are false for NaN, so a NaN `theta_e` never trips the theta_e cut.
"""
@inline function plasma_cut_omits(c::SimCoeffConfig, rho_cgs, n_e_cgs,
                                  pgas_cgs, theta_e, bb_cgs,
                                  sigma, beta_inv)
    return (c.cut_rho_min >= 0.0 && rho_cgs < c.cut_rho_min) ||
           (c.cut_rho_max >= 0.0 && rho_cgs > c.cut_rho_max) ||
           (c.cut_n_e_min >= 0.0 && n_e_cgs < c.cut_n_e_min) ||
           (c.cut_n_e_max >= 0.0 && n_e_cgs > c.cut_n_e_max) ||
           (c.cut_p_gas_min >= 0.0 && pgas_cgs < c.cut_p_gas_min) ||
           (c.cut_p_gas_max >= 0.0 && pgas_cgs > c.cut_p_gas_max) ||
           (c.cut_theta_e_min >= 0.0 && theta_e < c.cut_theta_e_min) ||
           (c.cut_theta_e_max >= 0.0 && theta_e > c.cut_theta_e_max) ||
           (c.cut_b_min >= 0.0 && bb_cgs < c.cut_b_min) ||
           (c.cut_b_max >= 0.0 && bb_cgs > c.cut_b_max) ||
           (c.cut_sigma_min >= 0.0 && sigma < c.cut_sigma_min) ||
           (c.cut_sigma_max >= 0.0 && sigma > c.cut_sigma_max) ||
           (c.cut_beta_inverse_min >= 0.0 && beta_inv < c.cut_beta_inverse_min) ||
           (c.cut_beta_inverse_max >= 0.0 && beta_inv > c.cut_beta_inverse_max)
end

"""
    electron_theta(c, rho_cgs, pgas_cgs, beta_inv) -> Float64

Electron temperature Theta_e (dimensionless, k T_e / m_e c^2) from the ti_te_beta
model. Returns NaN for the unsupported `plasma_thermal_frac == 0` corner (blacklight
parity — the NaN makes the theta_e plasma cuts never trip).
"""
@inline function electron_theta(c::SimCoeffConfig, rho_cgs, pgas_cgs, beta_inv)
    c.plasma_thermal_frac == 0.0 && return NaN
    tti_tte = (c.plasma_rat_high + c.plasma_rat_low * beta_inv * beta_inv) /
              (1.0 + beta_inv * beta_inv)
    kb_tt_tot_cgs = c.plasma_mu * MP * pgas_cgs / rho_cgs
    local kb_tt_e_cgs
    if c.plasma_use_p != 0
        kb_tt_e_cgs = (1.0 + c.plasma_ne_ni) / (tti_tte + c.plasma_ne_ni) * kb_tt_tot_cgs
    else
        kb_tt_e_cgs = (1.0 + c.plasma_ne_ni) * kb_tt_tot_cgs / (c.plasma_gamma - 1.0)
        kb_tt_e_cgs /= tti_tte / (c.plasma_gamma_i - 1.0) + c.plasma_ne_ni / (c.plasma_gamma_e - 1.0)
    end
    return kb_tt_e_cgs / (ME * CL * CL)
end

"""
    plasma_decode(c, rho, pgas, b_sq) -> NamedTuple

Shared code-unit -> cgs plasma decode for both the ray-traced imaging
(`eight_coefficients`) and the GRMHD Monte Carlo (`plasma_at`): cgs conversion ->
`electron_theta` (ti_te_beta) -> `plasma_cut_omits` gate. Returns
`(rho_cgs, pgas_cgs, n_e_cgs, bb_cgs, sigma, beta_inv, theta_e, omit::Bool)`;
`omit == true` means the sample is cut (treat as vacuum). Allocation-free and typed
on the bare `SimCoeffConfig` UnionAll so AD `Dual` configs match. The GPU kernels
re-implement this arithmetic against a flat isbits params struct (a mutable config
cannot live on-device) and must stay bit-exact with it.
"""
@inline function plasma_decode(c::SimCoeffConfig, rho, pgas, b_sq)
    rho_cgs = rho * c.d_unit
    pgas_cgs = pgas * c.e_unit
    n_cgs = rho_cgs / (c.plasma_mu * MP)
    n_e_cgs = n_cgs / (1.0 + 1.0 / c.plasma_ne_ni)
    bb_cgs = sqrt(b_sq) * c.b_unit
    sigma = b_sq / rho
    beta_inv = b_sq / (2.0 * pgas)
    theta_e = electron_theta(c, rho_cgs, pgas_cgs, beta_inv)
    omit = plasma_cut_omits(c, rho_cgs, n_e_cgs, pgas_cgs, theta_e, bb_cgs, sigma, beta_inv)
    return (rho_cgs = rho_cgs, pgas_cgs = pgas_cgs, n_e_cgs = n_e_cgs, bb_cgs = bb_cgs,
            sigma = sigma, beta_inv = beta_inv, theta_e = theta_e, omit = omit)
end

# ================================================================ thermal synchrotron

"""
    thermal_synchrotron(thermal_frac, n_e, theta_e, B, nu, sin_th) -> (j_inv, alpha_inv)

Unpolarized thermal synchrotron coefficients in Lorentz-invariant form (M 28/30/31),
with alpha from Kirchhoff's law. Returns `(0.0, 0.0)` on any degenerate input.
Follows blacklight ThermalSynchrotronInvariant.
"""
@inline function thermal_synchrotron(thermal_frac, n_e, theta_e, bb, nu, sin_theta_b)
    (thermal_frac == 0.0 || n_e <= 0.0 || theta_e <= 0.0 || bb <= 0.0 || nu <= 0.0 ||
     sin_theta_b <= 0.0) && return (0.0, 0.0)

    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_s = 2.0 / 9.0 * nu_c * theta_e * theta_e * sin_theta_b
    nu_2 = nu * nu

    xx = nu / nu_s
    xx_1_2 = sqrt(xx)
    xx_1_3 = cbrt(xx)
    xx_1_6 = sqrt(xx_1_3)
    coefficient = thermal_frac * n_e * EE * EE * nu_c / (CL * nu_2) * exp(-xx_1_3)
    var_a = SQRT2 * π / 27.0 * sin_theta_b
    var_b = 2.0^(11.0 / 12.0)
    var_c = xx_1_2 + var_b * xx_1_6
    j_inv = coefficient * var_a * var_c * var_c

    kb_tt_e = theta_e * ME * CL * CL
    b_nu_nu_3 = 2.0 * HPL / (CL * CL) / expm1(HPL * nu / kb_tt_e)
    alpha_inv = j_inv / b_nu_nu_3
    if !isfinite(1.0 / (alpha_inv * alpha_inv))
        alpha_inv = 0.0
    end
    return (j_inv, alpha_inv)
end

"Electron temperature below which the thermal Faraday factors take their cold-plasma limits."
const K_THETA_E_ZERO = 0.01

"Lower edge of the rho_Q blend window; below it the Faraday-conversion factor is fully trapped to 0."
const K_THETA_E_BLEND_LO = 0.005

# C1 smoothstep (3t^2 - 2t^3) ramping 0 -> 1 over [K_THETA_E_BLEND_LO, K_THETA_E_ZERO],
# bridging the thermal rho_Q Faraday-conversion factor smoothly out of its cold-plasma
# trap. Exactly 1.0 for Theta_e >= K_THETA_E_ZERO, so those coefficients are unchanged.
@inline function _theta_e_blend(theta_e)
    t = clamp((theta_e - K_THETA_E_BLEND_LO) / (K_THETA_E_ZERO - K_THETA_E_BLEND_LO), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
end

"""
    thermal_synchrotron_polarized(thermal_frac, n_e, theta_e, B, nu, sin_th, cos_th) -> NTuple{8}

Thermal synchrotron polarized coefficients (Marszewski et al. 2021), returned in the
order `(j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)`. `cos_th` is SIGNED
(sets the sign of the circular quantities). Includes the `theta_e ~ 0` cold-plasma
trap and the Kirchhoff / underflow guards. Follows blacklight
ThermalSynchrotronPolarized.
"""
@inline function thermal_synchrotron_polarized(thermal_frac, n_e, theta_e, bb, nu,
                                               sin_theta_b, cos_theta_b)
    (thermal_frac == 0.0 || n_e <= 0.0 || theta_e <= 0.0 || bb <= 0.0 || nu <= 0.0 ||
     sin_theta_b <= 0.0) && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    nu_c = EE * bb / (2.0 * π * ME * CL)
    sin2 = sin_theta_b * sin_theta_b
    nu_s = 2.0 / 9.0 * nu_c * theta_e * theta_e * sin_theta_b
    nu_2 = nu * nu

    xx = nu / nu_s
    xx_1_2 = sqrt(xx)
    xx_1_3 = cbrt(xx)
    xx_1_6 = sqrt(xx_1_3)
    coefficient = thermal_frac * n_e * EE * EE * nu_c / (CL * nu_2) * exp(-xx_1_3)
    var_a = SQRT2 * π / 27.0 * sin_theta_b
    var_b = 2.0^(11.0 / 12.0)
    var_c = xx_1_2 + var_b * xx_1_6

    # Emissivities (M 28,30).
    j_i = coefficient * var_a * var_c * var_c
    te96 = theta_e^0.96
    var_d = (7.0 * te96 + 35.0) / (10.0 * te96 + 75.0) * var_b
    var_e = xx_1_2 + var_d * xx_1_6
    var_f = cos_theta_b / theta_e
    var_g = π / 3.0 + π / 3.0 * xx_1_3 + 2.0 / 300.0 * xx_1_2 + 2.0 / 19.0 * π * xx_1_3 * xx_1_3
    j_q = -coefficient * var_a * var_e * var_e
    j_v = coefficient * var_f * var_g

    # Absorptivities from Kirchhoff's law (M 31).
    kb_tt_e = theta_e * ME * CL * CL
    b_nu_nu_3 = 2.0 * HPL / (CL * CL) / expm1(HPL * nu / kb_tt_e)
    alpha_i = j_i / b_nu_nu_3
    alpha_q = j_q / b_nu_nu_3
    alpha_v = j_v / b_nu_nu_3
    if !isfinite(1.0 / (alpha_i * alpha_i))
        alpha_i = 0.0
        alpha_q = 0.0
        alpha_v = 0.0
    end

    # Rotativities (M 33-37). rho_V takes its cold-plasma limit (factor 1) below
    # Theta_e = K_THETA_E_ZERO and is continuous there. Deviation from blacklight:
    # its rho_Q Faraday-conversion factor switches on discontinuously at K_THETA_E_ZERO;
    # here the fit is evaluated down to K_THETA_E_BLEND_LO and ramped with a C1
    # smoothstep, leaving every Theta_e >= K_THETA_E_ZERO coefficient unchanged.
    coefficient_q = -thermal_frac * n_e * EE * EE * nu_c * nu_c * sin2 / (ME * CL * nu_2)
    coefficient_v = thermal_frac * 2.0 * n_e * EE * EE * nu_c * cos_theta_b / (ME * CL * nu)
    factor_q = 0.0
    factor_v = 1.0
    if theta_e >= K_THETA_E_BLEND_LO
        inv_theta_e = 1.0 / theta_e
        kk_0 = bessel_k0(inv_theta_e)
        kk_1 = bessel_k1(inv_theta_e)
        # k_2 = k_0 + (2/x) k_1 inlined so the two Bessel fits are not re-evaluated
        # at the same argument (identical to bessel_k2(1/theta_e)).
        kk_2 = kk_0 + (2.0 / inv_theta_e) * kk_1
        xx_neg_1_2 = 1.0 / xx_1_2
        va = 2.011 * exp(-19.78 * xx^(-0.5175))
        vb = cos(39.89 * xx_neg_1_2) * exp(-70.16 * xx^(-0.6))
        vc = 0.011 * exp(-1.69 * xx_neg_1_2)
        vd = 0.003135 * xx^(4.0 / 3.0)
        ve = 0.5 * (1.0 + tanh(10.0 * log(0.6648 * xx_neg_1_2)))
        f_0 = va - vb - vc
        f_m = f_0 + (vc - vd) * ve
        factor_q = _theta_e_blend(theta_e) * f_m * (kk_1 / kk_2 + 6.0 * theta_e)
        if theta_e >= K_THETA_E_ZERO       # rho_V is already continuous; keep its cold trap
            delta_jj_5 = 0.4379 * log(1.0 + 1.3414 * xx^(-0.7515))
            factor_v = (kk_0 - delta_jj_5) / kk_2
            factor_v = (factor_v < 0.0 || factor_v > 1.0) ? 1.0 : factor_v
        end
    end
    rho_q = coefficient_q * factor_q
    rho_v = coefficient_v * factor_v

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end

# ================================================================ non-thermal bridge

"""
    harmonic_bridge(a, b, x) = (a^-x + b^-x)^(-1/x)

Harmonic-mean bridge between low- and high-frequency synchrotron asymptotes with
exponent `x` (blacklight HarmonicBridge). IEEE-safe: an underflowed branch gives
`pow(tiny, -x) -> inf` so the bridge returns the other asymptote; both underflowing
gives 0, the correct limit.
"""
@inline harmonic_bridge(a, b, x) = (a^(-x) + b^(-x))^(-1.0 / x)

# ================================================================ power-law synchrotron

"""
    power_law_synchrotron(c, n_e, B, nu, sin_th) -> (j_inv, alpha_inv)

Unpolarized power-law synchrotron coefficients (accumulation terms carrying
`plasma_power_frac`). Follows blacklight PowerLawSynchrotronInvariant.
"""
@inline function power_law_synchrotron(c::SimCoeffConfig, n_e, bb, nu, sin_theta_b)
    (c.plasma_power_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0)
    p = c.plasma_p
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_theta_b)
    j_inv = c.plasma_power_frac * n_e * EE * EE * nu_c / (CL * nu_2) *
            c.power_jj * sin_theta_b * xx^(-(p - 1.0) / 2.0)
    alpha_inv = c.plasma_power_frac * n_e * EE * EE / (ME * CL) * c.power_aa *
                xx^(-(p + 2.0) / 2.0)
    return (j_inv, alpha_inv)
end

"""
    power_law_synchrotron_polarized(c, n_e, B, nu, sin_th, cos_th) -> NTuple{8}

Power-law synchrotron polarized coefficients in the order
`(j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)` (accumulation terms carrying
`plasma_power_frac`). Follows blacklight PowerLawSynchrotronPolarized.
"""
@inline function power_law_synchrotron_polarized(c::SimCoeffConfig, n_e, bb, nu,
                                                 sin_theta_b, cos_theta_b)
    (c.plasma_power_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p = c.plasma_p
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_theta_b)

    coeff = c.plasma_power_frac * n_e * EE * EE * nu_c / (CL * nu_2) *
            c.power_jj * sin_theta_b * xx^(-(p - 1.0) / 2.0)
    j_i = coeff
    j_q = coeff * c.power_jj_q
    j_v = coeff * c.power_jj_v * (cos_theta_b / sin_theta_b) *
          (1.0 / sqrt(nu / (3.0 * nu_c * sin_theta_b)))

    acoeff = c.plasma_power_frac * n_e * EE * EE / (ME * CL) * c.power_aa * xx^(-(p + 2.0) / 2.0)
    alpha_i = acoeff
    alpha_q = acoeff * c.power_aa_q
    alpha_v = acoeff * c.power_aa_v * (3.1 * sin_theta_b^(-1.92) - 3.1)^0.512 *
              (1.0 / sqrt(xx)) * (cos_theta_b >= 0.0 ? 1.0 : -1.0)

    rcoeff = c.plasma_power_frac * c.power_rho * n_e * EE * EE * nu / (ME * CL * nu_c * sin_theta_b)
    var_b = nu_c * sin_theta_b / nu
    # X ≡ 2 ν_c γ_min² sinθ / (3ν): the low-energy cutoff ratio in the ρ_Q fit.
    xg = 2.0 * nu_c * c.plasma_gamma_min * c.plasma_gamma_min * sin_theta_b / (3.0 * nu)
    if p == 2.0
        # Removable pole: power_rho_q = -γ_min^(2-p)/(p/2-1) is -Inf and var_e = 1-X^0 = 0,
        # so the factored product is Inf*0 = NaN. The exact p→2 limit of
        # -γ_min^(2-p)(1-X^(p/2-1))/(p/2-1) is ln(X); use it instead of the factored form.
        rho_q = rcoeff * log(xg) * var_b * var_b * var_b
    else
        var_e = 1.0 - xg^(p / 2.0 - 1.0)
        rho_q = rcoeff * c.power_rho_q * var_b * var_b * var_b * var_e
    end
    rho_v = rcoeff * c.power_rho_v * var_b * var_b * (cos_theta_b / sin_theta_b)

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end

# ================================================================ kappa synchrotron

"""
    kappa_synchrotron(c, n_e, B, nu, sin_th) -> (j_inv, alpha_inv)

Unpolarized kappa-distribution synchrotron coefficients (accumulation terms carrying
`plasma_kappa_frac`). Follows blacklight KappaSynchrotronInvariant.
"""
@inline function kappa_synchrotron(c::SimCoeffConfig, n_e, bb, nu, sin_theta_b)
    (c.plasma_kappa_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0)
    k = c.plasma_kappa
    w = c.plasma_w
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    nu_kappa = nu_c * w * w * k * k * sin_theta_b
    xx = nu / nu_kappa

    aa = c.plasma_kappa_frac * n_e * EE * EE * nu_c / (CL * nu_2)
    j_low = c.kappa_jj_low * aa * cbrt(xx) * sin_theta_b
    j_high = c.kappa_jj_high * aa * xx^(-(k - 2.0) / 2.0) * sin_theta_b
    j_inv = harmonic_bridge(j_low, j_high, c.kappa_jj_x_i)

    bb_ = c.plasma_kappa_frac * n_e * EE * EE / (ME * CL)
    a_low = c.kappa_aa_low * bb_ * xx^(-2.0 / 3.0)
    a_high = c.kappa_aa_high * bb_ * xx^(-(1.0 + k) / 2.0) * c.kappa_aa_high_i
    alpha_inv = harmonic_bridge(a_low, a_high, c.kappa_aa_x_i)
    return (j_inv, alpha_inv)
end

"""
    kappa_synchrotron_polarized(c, n_e, B, nu, sin_th, cos_th) -> NTuple{8}

Kappa-distribution synchrotron polarized coefficients in the order
`(j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)` (accumulation terms carrying
`plasma_kappa_frac`). Follows blacklight KappaSynchrotronPolarized.
"""
@inline function kappa_synchrotron_polarized(c::SimCoeffConfig, n_e, bb, nu,
                                             sin_theta_b, cos_theta_b)
    (c.plasma_kappa_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k = c.plasma_kappa
    w = c.plasma_w
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    sin2 = sin_theta_b * sin_theta_b
    sgn = cos_theta_b >= 0.0 ? 1.0 : -1.0
    nu_kappa = nu_c * w * w * k * k * sin_theta_b
    xx = nu / nu_kappa
    s = 1.0 / sqrt(xx)

    # Emissivity.
    va = c.plasma_kappa_frac * n_e * EE * EE * nu_c / (CL * nu_2)
    c_low = c.kappa_jj_low * va * cbrt(xx) * sin_theta_b
    c_high = c.kappa_jj_high * va * xx^(-(k - 2.0) / 2.0) * sin_theta_b
    j_i = harmonic_bridge(c_low, c_high, c.kappa_jj_x_i)
    j_q = -harmonic_bridge(c_low * c.kappa_jj_low_q, c_high * c.kappa_jj_high_q, c.kappa_jj_x_q)
    jv_low = c_low * c.kappa_jj_low_v * (sin_theta_b^(-2.4) - 1.0)^0.48 * xx^(-0.35)
    jv_high = c_high * c.kappa_jj_high_v * (sin_theta_b^(-2.5) - 1.0)^0.44 * s
    j_v = harmonic_bridge(jv_low, jv_high, c.kappa_jj_x_v) * sgn

    # Absorptivity.
    vb = c.plasma_kappa_frac * n_e * EE * EE / (ME * CL)
    a_low = c.kappa_aa_low * vb * xx^(-2.0 / 3.0)
    a_high = c.kappa_aa_high * vb * xx^(-(1.0 + k) / 2.0)
    alpha_i = harmonic_bridge(a_low, a_high * c.kappa_aa_high_i, c.kappa_aa_x_i)
    alpha_q = -harmonic_bridge(a_low * c.kappa_aa_low_q, a_high * c.kappa_aa_high_q, c.kappa_aa_x_q)
    av_low = a_low * c.kappa_aa_low_v * (sin_theta_b^(-2.28) - 1.0)^0.446 * xx^(-0.35)
    av_high = a_high * c.kappa_aa_high_v * sqrt(sin_theta_b^(-2.05) - 1.0) * s
    alpha_v = harmonic_bridge(av_low, av_high, c.kappa_aa_x_v) * sgn

    # Rotativities.
    ra = -c.plasma_kappa_frac * n_e * EE * EE * nu_c * nu_c * sin2 / (ME * CL * nu_2)
    rb = c.plasma_kappa_frac * 2.0 * n_e * EE * EE * nu_c * cos_theta_b / (ME * CL * nu)
    rho_q_low = ra * c.kappa_rho_q_low_a * (1.0 -
        exp(c.kappa_rho_q_low_b * xx^0.84) -
        sin(c.kappa_rho_q_low_c * xx) * exp(c.kappa_rho_q_low_d * xx^c.kappa_rho_q_low_e))
    rho_q_high = ra * c.kappa_rho_q_high_a * (1.0 -
        exp(c.kappa_rho_q_high_b * xx^0.84) -
        sin(c.kappa_rho_q_high_c * xx) * exp(c.kappa_rho_q_high_d * xx^c.kappa_rho_q_high_e))
    rho_v_low = c.kappa_rho_v * rb * c.kappa_rho_v_low_a * (1.0 - 0.17 * log(1.0 + c.kappa_rho_v_low_b * s))
    rho_v_high = c.kappa_rho_v * rb * c.kappa_rho_v_high_a * (1.0 - 0.17 * log(1.0 + c.kappa_rho_v_high_b * s))
    rho_q = (1.0 - c.kappa_rho_frac) * rho_q_low + c.kappa_rho_frac * rho_q_high
    rho_v = (1.0 - c.kappa_rho_frac) * rho_v_low + c.kappa_rho_frac * rho_v_high

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end

# ================================================================ host precompute

# Host-only 2F1 hypergeometric for z < 0 (blacklight Hypergeometric): the
# Pfaff-transform equivalent via a 10-term rising-factorial series.
function _hypergeometric_2f1(alpha::Float64, beta::Float64, gam::Float64, z::Float64)
    k_max = 10
    a = alpha
    b = gam - beta
    c = gam
    x = z / (z - 1.0)
    result = 1.0
    a_k = 1.0; b_k = 1.0; c_k = 1.0; xk = 1.0; k_factorial = 1.0
    for kk in 1:k_max
        a_k *= a + kk - 1.0
        b_k *= b + kk - 1.0
        c_k *= c + kk - 1.0
        xk *= x
        k_factorial *= kk
        result += a_k * b_k * xk / (c_k * k_factorial)
    end
    result *= (1.0 - z)^(-alpha)
    return result
end

"""
    precompute_power_law!(c, gamma_min, gamma_max) -> c

Fill the p-dependent power-law constants (unpolarized `power_jj`/`power_aa` plus the
polarized set) in place. Follows blacklight PrecomputePowerLawCoefficients. Call once
when `plasma_power_frac != 0`.
"""
function precompute_power_law!(c::SimCoeffConfig, gamma_min::Float64, gamma_max::Float64)
    p = c.plasma_p
    var_a = 3.0^(p / 2.0) * (p - 1.0)
    var_b = 2.0 * (p + 1.0)
    var_c = gamma_min^(1.0 - p) - gamma_max^(1.0 - p)
    var_d = gamma((3.0 * p - 1.0) / 12.0)
    var_e = gamma((3.0 * p + 19.0) / 12.0)
    var_f = 3.0^((p + 1.0) / 2.0) * (p - 1.0) / 4.0
    var_g = gamma((3.0 * p + 2.0) / 12.0)
    var_h = gamma((3.0 * p + 22.0) / 12.0)
    c.power_jj = var_a / var_b / var_c * var_d * var_e
    c.power_aa = var_f / var_c * var_g * var_h

    c.plasma_gamma_min = gamma_min

    var_i = 2.0 * (p + 2.0) / (p + 1.0)
    var_j = gamma_min^(-(p + 1.0))
    var_k = log(gamma_min)
    c.power_jj_q = -(p + 1.0) / (p + 7.0 / 3.0)
    c.power_jj_v = 0.684 * p^0.49
    c.power_aa_q = -(0.034 * p - 0.0344)^0.086
    c.power_aa_v = (0.71 * p + 0.0352)^0.394
    c.power_rho = (p - 1.0) / var_c
    # -Inf at p==2 (removable pole); power_law_synchrotron_polarized evaluates the
    # analytic ln(X) limit there instead of multiplying this by var_e = 0.
    c.power_rho_q = -gamma_min^(2.0 - p) / (p / 2.0 - 1.0)
    c.power_rho_v = var_i * var_j * var_k
    return c
end

"""
    precompute_kappa!(c) -> c

Fill the kappa/w-dependent constants (unpolarized + polarized subsets, including the
three-way branch on kappa and the Bessel rho_V factor) in place, using `gamma()`,
`besselk()`, and the host 2F1. Follows blacklight PrecomputeKappaCoefficients. Call
once when `plasma_kappa_frac != 0`.
"""
function precompute_kappa!(c::SimCoeffConfig)
    k = c.plasma_kappa
    w = c.plasma_w
    var_a = 4.0 * π * gamma(k - 4.0 / 3.0)
    var_b = 3.0^(7.0 / 3.0) * gamma(k - 2.0)
    var_c = 3.0^((k - 1.0) / 2.0)
    var_d = (k - 2.0) * (k - 1.0) / 4.0
    var_e = gamma(k / 4.0 - 1.0 / 3.0)
    var_f = gamma(k / 4.0 + 4.0 / 3.0)
    var_g = 3.0^(1.0 / 6.0) * 10.0 / 41.0
    var_h = w * k
    var_i = 2.0 * π * var_h^(k - 10.0 / 3.0)
    var_j = (k - 2.0) * (k - 1.0) * k
    var_k = 3.0 * k - 1.0
    var_l = gamma(5.0 / 3.0)
    var_m = _hypergeometric_2f1(k - 1.0 / 3.0, k + 1.0, k + 2.0 / 3.0, -var_h)
    var_n = π^1.5 / 3.0
    var_o = var_j / (var_h * var_h * var_h)
    var_p = 2.0 * gamma(2.0 + k / 2.0) / (2.0 + k) - 1.0
    c.kappa_jj_low = var_a / var_b
    c.kappa_jj_high = var_c * var_d * var_e * var_f
    c.kappa_jj_x_i = 3.0 * k^(-1.5)
    c.kappa_aa_low = var_g * var_i * var_j / var_k * var_l * var_m
    c.kappa_aa_high = var_n * var_o * var_p
    c.kappa_aa_x_i = (-1.75 + 1.6 * k)^(-0.86)
    c.kappa_aa_high_i = (3.0 / k)^4.75 + 0.6

    var_q = 14.3 * w^(-0.928)
    var_r = 169.0 * k^(-8.0) + 0.0052 * k - 0.0526 + 47.0 / (200.0 * k)
    c.kappa_jj_low_q = 0.5
    c.kappa_jj_low_v = 0.5625 * k^(-0.528) / w
    c.kappa_jj_high_q = 0.64 + 0.02 * k
    c.kappa_jj_high_v = 0.765625 * k^(-0.44) / w
    c.kappa_jj_x_q = 3.7 * k^(-1.6)
    c.kappa_jj_x_v = c.kappa_jj_x_i
    c.kappa_aa_low_q = 25.0 / 48.0
    c.kappa_aa_low_v = 77.0 / (100.0 * w) * k^(-0.7)
    c.kappa_aa_high_q = 441.0 * k^(-5.76) + 0.55
    c.kappa_aa_high_v = var_q * var_r
    c.kappa_aa_x_q = 1.4 * k^(-1.15)
    c.kappa_aa_x_v = 1.22 * k^(-1.136) + 0.007

    c.kappa_rho_v = besselk(0, 1.0 / w) / besselk(2, 1.0 / w)

    if k < 4.0
        c.kappa_rho_frac = (k - 3.5) / (4.0 - 3.5)
        c.kappa_rho_q_low_a = 17.0 * w + sqrt(w) * (-3.0 + 7.0 * exp(-5.0 * w))
        c.kappa_rho_q_low_b = -1.0 / 30.0
        c.kappa_rho_q_low_c = 0.1
        c.kappa_rho_q_low_d = -1.5
        c.kappa_rho_q_low_e = 0.471
        c.kappa_rho_q_high_a = 46.0 / 3.0 * w + sqrt(w) * (-5.0 / 3.0 + 17.0 / 3.0 * exp(-5.0 * w))
        c.kappa_rho_q_high_b = -1.0 / 18.0
        c.kappa_rho_q_high_c = 1.0 / 6.0
        c.kappa_rho_q_high_d = -1.75
        c.kappa_rho_q_high_e = 0.5
        c.kappa_rho_v_low_a = (w * w + 2.0 * w + 1.0) / (3.125 * w * w + 4.0 * w + 1.0)
        c.kappa_rho_v_low_b = 0.447
        c.kappa_rho_v_high_a = (w * w + 54.0 * w + 50.0) / (30.0 / 11.0 * w * w + 134.0 * w + 50.0)
        c.kappa_rho_v_high_b = 0.391
    elseif k < 4.5
        c.kappa_rho_frac = (k - 4.0) / (4.5 - 4.0)
        c.kappa_rho_q_low_a = 46.0 / 3.0 * w + sqrt(w) * (-5.0 / 3.0 + 17.0 / 3.0 * exp(-5.0 * w))
        c.kappa_rho_q_low_b = -1.0 / 18.0
        c.kappa_rho_q_low_c = 1.0 / 6.0
        c.kappa_rho_q_low_d = -1.75
        c.kappa_rho_q_low_e = 0.5
        c.kappa_rho_q_high_a = 14.0 * w + sqrt(w) * (-1.625 + 4.5 * exp(-5.0 * w))
        c.kappa_rho_q_high_b = -1.0 / 12.0
        c.kappa_rho_q_high_c = 0.25
        c.kappa_rho_q_high_d = -2.0
        c.kappa_rho_q_high_e = 0.525
        c.kappa_rho_v_low_a = (w * w + 54.0 * w + 50.0) / (30.0 / 11.0 * w * w + 134.0 * w + 50.0)
        c.kappa_rho_v_low_b = 0.391
        c.kappa_rho_v_high_a = (w * w + 43.0 * w + 38.0) / (7.0 / 3.0 * w * w + 92.5 * w + 38.0)
        c.kappa_rho_v_high_b = 0.348
    else
        c.kappa_rho_frac = (k - 4.5) / (5.0 - 4.5)
        c.kappa_rho_q_low_a = 14.0 * w + sqrt(w) * (-1.625 + 4.5 * exp(-5.0 * w))
        c.kappa_rho_q_low_b = -1.0 / 12.0
        c.kappa_rho_q_low_c = 0.25
        c.kappa_rho_q_low_d = -2.0
        c.kappa_rho_q_low_e = 0.525
        c.kappa_rho_q_high_a = 12.5 * w + sqrt(w) * (-1.0 + 5.0 * exp(-5.0 * w))
        c.kappa_rho_q_high_b = -0.125
        c.kappa_rho_q_high_c = 0.375
        c.kappa_rho_q_high_d = -2.25
        c.kappa_rho_q_high_e = 0.541
        c.kappa_rho_v_low_a = (w * w + 43.0 * w + 38.0) / (7.0 / 3.0 * w * w + 92.5 * w + 38.0)
        c.kappa_rho_v_low_b = 0.348
        c.kappa_rho_v_high_a = (w + 13.0 / 14.0) / (2.0 * w + 13.0 / 14.0)
        c.kappa_rho_v_high_b = 0.313
    end
    return c
end
