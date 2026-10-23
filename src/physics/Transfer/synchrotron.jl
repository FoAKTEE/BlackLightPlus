# Synchrotron emission, absorption and rotation coefficients of the thermal, power-law
# and kappa electron populations (blacklight radiation/simulation_coefficients.cpp and
# polarized_coefficients.hpp; fits of Marszewski et al. 2021, "M" equation tags, Dexter
# 2016 and Pandya et al. 2016). Every coefficient is a Lorentz invariant:
#   j = j_ν / ν²,   α = α_ν ν,   ρ = ρ_ν ν,
# in cgs, with ν the fluid-frame frequency and θ_B the fluid-frame pitch angle.
# Polarized tuples are ordered (j_I, j_Q, j_V, α_I, α_Q, α_V, ρ_Q, ρ_V).

"""
    thermal_synchrotron(thermal_frac, n_e, theta_e, B, nu, sin_th) -> (j, α)

Unpolarized thermal synchrotron emissivity and absorptivity in invariant form
(M 28, 30, 31) for electron density `n_e` (cm⁻³), temperature `theta_e`, field `B`
(G), fluid-frame frequency `nu` (Hz) and pitch angle `sin_th`; `α` from Kirchhoff's
law, zero where `1/α²` is not finite. Zeros of the promoted type of `n_e`,
`theta_e`, `B` for any non-positive input or a zero `thermal_frac`. After blacklight
`ThermalSynchrotronInvariant`.
"""
@inline function thermal_synchrotron(thermal_frac, n_e, theta_e, bb, nu, sin_theta_b)
    z = zero(promote_type(typeof(n_e), typeof(theta_e), typeof(bb)))
    (thermal_frac == 0.0 || n_e <= 0.0 || theta_e <= 0.0 || bb <= 0.0 || nu <= 0.0 ||
     sin_theta_b <= 0.0) && return (z, z)

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
    isfinite(1.0 / (alpha_inv * alpha_inv)) || (alpha_inv = z)
    return (j_inv, alpha_inv)
end

"Electron temperature below which the thermal Faraday factors take their cold-plasma limits."
const K_THETA_E_ZERO = 0.01

"Lower edge of the ρ_Q blend window; below it the Faraday-conversion factor is zero."
const K_THETA_E_BLEND_LO = 0.005

# C¹ smoothstep 3t² - 2t³ rising from 0 to 1 over [K_THETA_E_BLEND_LO, K_THETA_E_ZERO]
# and exactly 1 above it: the weight of the thermal ρ_Q Faraday-conversion factor.
@inline function _theta_e_blend(theta_e)
    t = clamp((theta_e - K_THETA_E_BLEND_LO) / (K_THETA_E_ZERO - K_THETA_E_BLEND_LO), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
end

"""
    thermal_synchrotron_polarized(thermal_frac, n_e, theta_e, B, nu, sin_th, cos_th) -> NTuple{8}

Thermal synchrotron polarized coefficients `(j_I, j_Q, j_V, α_I, α_Q, α_V, ρ_Q, ρ_V)`
(M 28, 30, 31, 33–37); `cos_th` is signed along the propagation and sets the sign
of the circular terms. The absorptivities follow from Kirchhoff's law and are zero
where `1/α_I²` is not finite. `ρ_V` takes its cold-plasma factor 1 below
`K_THETA_E_ZERO`; the `ρ_Q` conversion factor is weighted by a C¹ smoothstep over
[`K_THETA_E_BLEND_LO`, `K_THETA_E_ZERO`] and is zero below it. Eight `Float64`
zeros for any non-positive input or a zero `thermal_frac`. After blacklight
`ThermalSynchrotronPolarized`.
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

    # emissivities (M 28, 30)
    j_i = coefficient * var_a * var_c * var_c
    te96 = theta_e^0.96
    var_d = (7.0 * te96 + 35.0) / (10.0 * te96 + 75.0) * var_b
    var_e = xx_1_2 + var_d * xx_1_6
    var_f = cos_theta_b / theta_e
    var_g = π / 3.0 + π / 3.0 * xx_1_3 + 2.0 / 300.0 * xx_1_2 + 2.0 / 19.0 * π * xx_1_3 * xx_1_3
    j_q = -coefficient * var_a * var_e * var_e
    j_v = coefficient * var_f * var_g

    # absorptivities from Kirchhoff's law (M 31)
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

    # rotativities (M 33-37)
    coefficient_q = -thermal_frac * n_e * EE * EE * nu_c * nu_c * sin2 / (ME * CL * nu_2)
    coefficient_v = thermal_frac * 2.0 * n_e * EE * EE * nu_c * cos_theta_b / (ME * CL * nu)
    factor_q = 0.0
    factor_v = 1.0
    if theta_e >= K_THETA_E_BLEND_LO
        inv_theta_e = 1.0 / theta_e
        kk_0 = bessel_k0(inv_theta_e)
        kk_1 = bessel_k1(inv_theta_e)
        # K₂ = K₀ + (2/x) K₁ at x = 1/Θ_e, from the two fits already evaluated
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
        if theta_e >= K_THETA_E_ZERO
            delta_jj_5 = 0.4379 * log(1.0 + 1.3414 * xx^(-0.7515))
            factor_v = (kk_0 - delta_jj_5) / kk_2
            factor_v = (factor_v < 0.0 || factor_v > 1.0) ? 1.0 : factor_v
        end
    end
    rho_q = coefficient_q * factor_q
    rho_v = coefficient_v * factor_v

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end

"""
    power_law_synchrotron(pl::PowerLaw, n_e, B, nu, sin_th) -> (j, α)

Unpolarized power-law synchrotron emissivity and absorptivity in invariant form,
scaled by `pl.frac`; zeros of the promoted input type for a zero fraction or any
non-positive input. After blacklight `PowerLawSynchrotronInvariant`.
"""
@inline function power_law_synchrotron(pl::PowerLaw, n_e, bb, nu, sin_theta_b)
    z = zero(promote_type(typeof(n_e), typeof(bb), typeof(nu), typeof(sin_theta_b)))
    (pl.frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (z, z)
    p = pl.p
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_theta_b)
    j_inv = pl.frac * n_e * EE * EE * nu_c / (CL * nu_2) *
            pl.jj * sin_theta_b * xx^(-(p - 1.0) / 2.0)
    alpha_inv = pl.frac * n_e * EE * EE / (ME * CL) * pl.aa *
                xx^(-(p + 2.0) / 2.0)
    return (j_inv, alpha_inv)
end

"""
    power_law_synchrotron_polarized(pl::PowerLaw, n_e, B, nu, sin_th, cos_th) -> NTuple{8}

Power-law synchrotron polarized coefficients `(j_I, j_Q, j_V, α_I, α_Q, α_V, ρ_Q, ρ_V)`,
scaled by `pl.frac`. `ρ_Q` uses the low-energy cutoff ratio
`X = 2 ν_c γ_min² sin θ / (3ν)` and its exact `ln X` limit at `p = 2`. Eight
`Float64` zeros for a zero fraction or any non-positive input. After blacklight
`PowerLawSynchrotronPolarized`.
"""
@inline function power_law_synchrotron_polarized(pl::PowerLaw, n_e, bb, nu,
                                                 sin_theta_b, cos_theta_b)
    (pl.frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p = pl.p
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_theta_b)

    coeff = pl.frac * n_e * EE * EE * nu_c / (CL * nu_2) *
            pl.jj * sin_theta_b * xx^(-(p - 1.0) / 2.0)
    j_i = coeff
    j_q = coeff * pl.jj_q
    j_v = coeff * pl.jj_v * (cos_theta_b / sin_theta_b) *
          (1.0 / sqrt(nu / (3.0 * nu_c * sin_theta_b)))

    acoeff = pl.frac * n_e * EE * EE / (ME * CL) * pl.aa * xx^(-(p + 2.0) / 2.0)
    alpha_i = acoeff
    alpha_q = acoeff * pl.aa_q
    alpha_v = acoeff * pl.aa_v * (3.1 * sin_theta_b^(-1.92) - 3.1)^0.512 *
              (1.0 / sqrt(xx)) * (cos_theta_b >= 0.0 ? 1.0 : -1.0)

    rcoeff = pl.frac * pl.rho * n_e * EE * EE * nu / (ME * CL * nu_c * sin_theta_b)
    var_b = nu_c * sin_theta_b / nu
    xg = 2.0 * nu_c * pl.gamma_min * pl.gamma_min * sin_theta_b / (3.0 * nu)
    if p == 2.0
        # p → 2 limit of -γ_min^(2-p) (1 - X^(p/2-1)) / (p/2 - 1): ln X, where the
        # factored form is -Inf × 0
        rho_q = rcoeff * log(xg) * var_b * var_b * var_b
    else
        var_e = 1.0 - xg^(p / 2.0 - 1.0)
        rho_q = rcoeff * pl.rho_q * var_b * var_b * var_b * var_e
    end
    rho_v = rcoeff * pl.rho_v * var_b * var_b * (cos_theta_b / sin_theta_b)

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end

"""
    kappa_synchrotron(kp::Kappa, n_e, B, nu, sin_th) -> (j, α)

Unpolarized kappa-distribution synchrotron emissivity and absorptivity in invariant
form, the low- and high-frequency asymptotes joined by `harmonic_bridge` and scaled
by `kp.frac`; zeros of the promoted input type for a zero fraction or any
non-positive input. After blacklight `KappaSynchrotronInvariant`.
"""
@inline function kappa_synchrotron(kp::Kappa, n_e, bb, nu, sin_theta_b)
    z = zero(promote_type(typeof(n_e), typeof(bb), typeof(nu), typeof(sin_theta_b)))
    (kp.frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (z, z)
    k = kp.kappa
    w = kp.w
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    nu_kappa = nu_c * w * w * k * k * sin_theta_b
    xx = nu / nu_kappa

    aa = kp.frac * n_e * EE * EE * nu_c / (CL * nu_2)
    j_low = kp.jj_low * aa * cbrt(xx) * sin_theta_b
    j_high = kp.jj_high * aa * xx^(-(k - 2.0) / 2.0) * sin_theta_b
    j_inv = harmonic_bridge(j_low, j_high, kp.jj_x_i)

    bb_ = kp.frac * n_e * EE * EE / (ME * CL)
    a_low = kp.aa_low * bb_ * xx^(-2.0 / 3.0)
    a_high = kp.aa_high * bb_ * xx^(-(1.0 + k) / 2.0) * kp.aa_high_i
    alpha_inv = harmonic_bridge(a_low, a_high, kp.aa_x_i)
    return (j_inv, alpha_inv)
end

"""
    kappa_synchrotron_polarized(kp::Kappa, n_e, B, nu, sin_th, cos_th) -> NTuple{8}

Kappa-distribution synchrotron polarized coefficients
`(j_I, j_Q, j_V, α_I, α_Q, α_V, ρ_Q, ρ_V)`, scaled by `kp.frac`; the rotativities
blend the fits of the two tabulated indices bracketing `kp.kappa`. Eight `Float64`
zeros for a zero fraction or any non-positive input. After blacklight
`KappaSynchrotronPolarized`.
"""
@inline function kappa_synchrotron_polarized(kp::Kappa, n_e, bb, nu,
                                             sin_theta_b, cos_theta_b)
    (kp.frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_theta_b <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k = kp.kappa
    w = kp.w
    nu_c = EE * bb / (2.0 * π * ME * CL)
    nu_2 = nu * nu
    sin2 = sin_theta_b * sin_theta_b
    sgn = cos_theta_b >= 0.0 ? 1.0 : -1.0
    nu_kappa = nu_c * w * w * k * k * sin_theta_b
    xx = nu / nu_kappa
    s = 1.0 / sqrt(xx)

    # emissivity
    va = kp.frac * n_e * EE * EE * nu_c / (CL * nu_2)
    c_low = kp.jj_low * va * cbrt(xx) * sin_theta_b
    c_high = kp.jj_high * va * xx^(-(k - 2.0) / 2.0) * sin_theta_b
    j_i = harmonic_bridge(c_low, c_high, kp.jj_x_i)
    j_q = -harmonic_bridge(c_low * kp.jj_low_q, c_high * kp.jj_high_q, kp.jj_x_q)
    jv_low = c_low * kp.jj_low_v * (sin_theta_b^(-2.4) - 1.0)^0.48 * xx^(-0.35)
    jv_high = c_high * kp.jj_high_v * (sin_theta_b^(-2.5) - 1.0)^0.44 * s
    j_v = harmonic_bridge(jv_low, jv_high, kp.jj_x_v) * sgn

    # absorptivity
    vb = kp.frac * n_e * EE * EE / (ME * CL)
    a_low = kp.aa_low * vb * xx^(-2.0 / 3.0)
    a_high = kp.aa_high * vb * xx^(-(1.0 + k) / 2.0)
    alpha_i = harmonic_bridge(a_low, a_high * kp.aa_high_i, kp.aa_x_i)
    alpha_q = -harmonic_bridge(a_low * kp.aa_low_q, a_high * kp.aa_high_q, kp.aa_x_q)
    av_low = a_low * kp.aa_low_v * (sin_theta_b^(-2.28) - 1.0)^0.446 * xx^(-0.35)
    av_high = a_high * kp.aa_high_v * sqrt(sin_theta_b^(-2.05) - 1.0) * s
    alpha_v = harmonic_bridge(av_low, av_high, kp.aa_x_v) * sgn

    # rotativities
    ra = -kp.frac * n_e * EE * EE * nu_c * nu_c * sin2 / (ME * CL * nu_2)
    rb = kp.frac * 2.0 * n_e * EE * EE * nu_c * cos_theta_b / (ME * CL * nu)
    rho_q_low = ra * kp.rho_q_low_a * (1.0 -
        exp(kp.rho_q_low_b * xx^0.84) -
        sin(kp.rho_q_low_c * xx) * exp(kp.rho_q_low_d * xx^kp.rho_q_low_e))
    rho_q_high = ra * kp.rho_q_high_a * (1.0 -
        exp(kp.rho_q_high_b * xx^0.84) -
        sin(kp.rho_q_high_c * xx) * exp(kp.rho_q_high_d * xx^kp.rho_q_high_e))
    rho_v_low = kp.rho_v * rb * kp.rho_v_low_a * (1.0 - 0.17 * log(1.0 + kp.rho_v_low_b * s))
    rho_v_high = kp.rho_v * rb * kp.rho_v_high_a * (1.0 - 0.17 * log(1.0 + kp.rho_v_high_b * s))
    rho_q = (1.0 - kp.rho_frac) * rho_q_low + kp.rho_frac * rho_q_high
    rho_v = (1.0 - kp.rho_frac) * rho_v_low + kp.rho_frac * rho_v_high

    return (j_i, j_q, j_v, alpha_i, alpha_q, alpha_v, rho_q, rho_v)
end
