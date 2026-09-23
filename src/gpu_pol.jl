# gpu_pol.jl — fused single-pass GPU polarized imager (Stokes I,Q,U,V). The CPU
# coherency march (polarized_march) runs source->camera over a stored trajectory; the
# fused tracer integrates camera->source and stores nothing, and RT is order-dependent,
# so the transfer is accumulated in reverse: the Stokes accumulator S plus a running
# REAL 4x4 Mueller transmission G (no ComplexF64 in-kernel) in a parallel-transported
# camera polarization frame, S += G(R b_seg); G <- G(R M_seg Rᵀ), with (M,b) the
# analytic segment solution (transfer.jl::stokes_couple) in the fluid tetrad and R the
# Q-U rotation into the transported frame. Exact reverse RT, validated against the CPU
# march. CUDA optional: NOT included from BlackLightPlus.jl — include AFTER gpu_image.jl.

using StaticArrays
using ForwardDiff: Dual, partials, value
using KernelAbstractions
using CUDA
using LinearAlgebra: dot

# ---------------------------------------------------------------- device: Bessel (A&S 9.8)
# Device-safe modified Bessel functions (polynomial fits; no std::cyl_bessel_k), ported from
# radiation/coefficients.jl. Used by the thermal Faraday rotativities (argument 1/theta_e).
@inline function _gpol_i0(x)
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
@inline function _gpol_i1(x)
    ax = abs(x); local ans
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
@inline function _gpol_k0(x)
    if x <= 2.0
        t = x * x / 4.0
        return -log(x / 2.0) * _gpol_i0(x) +
            (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590 +
              t * (0.00262698 + t * (0.00010750 + t * 0.00000740))))))
    end
    t = 2.0 / x
    return (exp(-x) / sqrt(x)) *
        (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 +
          t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))))
end
@inline function _gpol_k1(x)
    if x <= 2.0
        t = x * x / 4.0
        return log(x / 2.0) * _gpol_i1(x) +
            (1.0 / x) * (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897 +
              t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
    end
    t = 2.0 / x
    return (exp(-x) / sqrt(x)) *
        (1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268 +
          t * (-0.00780353 + t * (0.00325614 + t * (-0.00068245)))))))
end
@inline _gpol_k2(x) = _gpol_k0(x) + (2.0 / x) * _gpol_k1(x)

const _K_THETA_E_ZERO = 0.01
const _K_THETA_E_BLEND_LO = 0.005
# C1 smoothstep (3t^2 - 2t^3) ramping 0 -> 1 over [_K_THETA_E_BLEND_LO, _K_THETA_E_ZERO]:
# device twin of coefficients.jl :: _theta_e_blend (rho_Q bridge out of the cold-plasma
# trap). Exactly 1.0 for theta_e >= _K_THETA_E_ZERO, so those outputs are unchanged.
@inline function _gpol_theta_e_blend(theta_e)
    t = clamp((theta_e - _K_THETA_E_BLEND_LO) / (_K_THETA_E_ZERO - _K_THETA_E_BLEND_LO), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
end
const _gpz4  = SVector(0.0, 0.0, 0.0, 0.0)
const _gpI4  = one(SMatrix{4,4,Float64})
const _gpz44 = zero(SMatrix{4,4,Float64})

# ---------------------------------------------------------------- device: thermal polarized coeffs
# Port of coefficients.jl :: thermal_synchrotron_polarized. `thermal_frac` scales every
# coefficient exactly as the CPU function (thermal_frac==0 -> no emission).
# Returns invariant (j_i,j_q,j_v, a_i,a_q,a_v, rho_q,rho_v); cos_th is SIGNED.
@inline function _gpol_thermal(thermal_frac, n_e, theta_e, bb, nu, sin_tb, cos_tb)
    (thermal_frac == 0.0 || n_e <= 0.0 || theta_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    sin2 = sin_tb * sin_tb
    nu_s = 2.0 / 9.0 * nu_c * theta_e * theta_e * sin_tb
    nu_2 = nu * nu
    xx = nu / nu_s
    xx12 = sqrt(xx); xx13 = cbrt(xx); xx16 = sqrt(xx13)
    coeff = thermal_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2) * exp(-xx13)
    var_a = _SQRT2 * π / 27.0 * sin_tb
    var_b = 2.0^(11.0 / 12.0)
    var_c = xx12 + var_b * xx16
    j_i = coeff * var_a * var_c * var_c
    var_d = (7.0 * theta_e^0.96 + 35.0) / (10.0 * theta_e^0.96 + 75.0) * var_b
    var_e = xx12 + var_d * xx16
    var_f = cos_tb / theta_e
    var_g = π / 3.0 + π / 3.0 * xx13 + 2.0 / 300.0 * xx12 + 2.0 / 19.0 * π * xx13 * xx13
    j_q = -coeff * var_a * var_e * var_e
    j_v = coeff * var_f * var_g
    kb_tt_e = theta_e * _ME * _CL * _CL
    b_nu_nu_3 = 2.0 * _HPL / (_CL * _CL) / expm1(_HPL * nu / kb_tt_e)
    a_i = j_i / b_nu_nu_3; a_q = j_q / b_nu_nu_3; a_v = j_v / b_nu_nu_3
    if !isfinite(1.0 / (a_i * a_i))
        a_i = 0.0; a_q = 0.0; a_v = 0.0
    end
    coeff_q = -thermal_frac * n_e * _EE * _EE * nu_c * nu_c * sin2 / (_ME * _CL * nu_2)
    coeff_v = thermal_frac * 2.0 * n_e * _EE * _EE * nu_c * cos_tb / (_ME * _CL * nu)
    factor_q = 0.0; factor_v = 1.0
    # rho_Q blend mirror (coefficients.jl): evaluate the fit down to _K_THETA_E_BLEND_LO and
    # ramp factor_q with the C1 smoothstep; the weight is exactly 1 for theta_e >= _K_THETA_E_ZERO
    # (1.0 * x == x bitwise), so all previously-nonzero outputs are unchanged. rho_V keeps its
    # original cold trap at _K_THETA_E_ZERO (already continuous there).
    if theta_e >= _K_THETA_E_BLEND_LO
        inv_te = 1.0 / theta_e
        # k2 = k0 + (2/x) k1 with x = 1/theta_e; reuse kk_0,kk_1 (drop the _gpol_k2
        # call that re-evaluated k0,k1). Keeping 2/inv_te (not 2*theta_e) is bit-identical.
        kk_0 = _gpol_k0(inv_te); kk_1 = _gpol_k1(inv_te); kk_2 = kk_0 + (2.0 / inv_te) * kk_1
        xx_n12 = 1.0 / xx12
        va = 2.011 * exp(-19.78 * xx^(-0.5175))
        vb = cos(39.89 * xx_n12) * exp(-70.16 * xx^(-0.6))
        vc = 0.011 * exp(-1.69 * xx_n12)
        vd = 0.003135 * xx^(4.0 / 3.0)
        ve = 0.5 * (1.0 + tanh(10.0 * log(0.6648 * xx_n12)))
        f_0 = va - vb - vc
        f_m = f_0 + (vc - vd) * ve
        factor_q = _gpol_theta_e_blend(theta_e) * f_m * (kk_1 / kk_2 + 6.0 * theta_e)
        if theta_e >= _K_THETA_E_ZERO
            delta_jj_5 = 0.4379 * log(1.0 + 1.3414 * xx^(-0.7515))
            factor_v = (kk_0 - delta_jj_5) / kk_2
            factor_v = (factor_v < 0.0 || factor_v > 1.0) ? 1.0 : factor_v
        end
    end
    rho_q = coeff_q * factor_q
    rho_v = coeff_v * factor_v
    return (j_i, j_q, j_v, a_i, a_q, a_v, rho_q, rho_v)
end

# ---------------------------------------------------------------- device: power-law polarized coeffs
# Port of coefficients.jl :: power_law_synchrotron_polarized. Returns the additive non-thermal
# term (j_i,j_q,j_v, a_i,a_q,a_v, rho_q,rho_v) mixed into _gpol_coeffs when power_frac != 0.
# Reads the host-precomputed constants (gp.power_jj/_aa/_jj_q/_jj_v/_aa_q/_aa_v/_rho/_rho_q/
# _rho_v, gp.gamma_min) from the params tuple, so every per-step formula is bit-for-bit the
# CPU body — including the removable p==2 pole (analytic ln(X) limit). cos_tb is SIGNED.
@inline function _gpol_powerlaw(gp, n_e, bb, nu, sin_tb, cos_tb)
    (gp.power_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    p = gp.power_p
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    nu_2 = nu * nu
    xx = nu / (nu_c * sin_tb)

    coeff = gp.power_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2) *
            gp.power_jj * sin_tb * xx^(-(p - 1.0) / 2.0)
    j_i = coeff
    j_q = coeff * gp.power_jj_q
    j_v = coeff * gp.power_jj_v * (cos_tb / sin_tb) *
          (1.0 / sqrt(nu / (3.0 * nu_c * sin_tb)))

    acoeff = gp.power_frac * n_e * _EE * _EE / (_ME * _CL) * gp.power_aa * xx^(-(p + 2.0) / 2.0)
    a_i = acoeff
    a_q = acoeff * gp.power_aa_q
    a_v = acoeff * gp.power_aa_v * (3.1 * sin_tb^(-1.92) - 3.1)^0.512 *
          (1.0 / sqrt(xx)) * (cos_tb >= 0.0 ? 1.0 : -1.0)

    rcoeff = gp.power_frac * gp.power_rho * n_e * _EE * _EE * nu / (_ME * _CL * nu_c * sin_tb)
    var_b = nu_c * sin_tb / nu
    xg = 2.0 * nu_c * gp.gamma_min * gp.gamma_min * sin_tb / (3.0 * nu)
    local rho_q
    if p == 2.0
        rho_q = rcoeff * log(xg) * var_b * var_b * var_b
    else
        var_e = 1.0 - xg^(p / 2.0 - 1.0)
        rho_q = rcoeff * gp.power_rho_q * var_b * var_b * var_b * var_e
    end
    rho_v = rcoeff * gp.power_rho_v * var_b * var_b * (cos_tb / sin_tb)

    return (j_i, j_q, j_v, a_i, a_q, a_v, rho_q, rho_v)
end

# ---------------------------------------------------------------- device: kappa polarized coeffs
# Port of coefficients.jl :: kappa_synchrotron_polarized. Returns the additive non-thermal
# term (j_i,j_q,j_v, a_i,a_q,a_v, rho_q,rho_v) mixed into _gpol_coeffs when kappa_frac != 0.
# Reads the host-precomputed constants (gp.kappa_jj_*/kappa_aa_*/kappa_rho_* + gp.kappa/gp.w)
# from the params tuple — the kappa-index three-way fit branch is resolved HOST-side in
# precompute_kappa!, so the device sees only the branch-selected scalars — so every per-step
# formula is bit-for-bit the CPU body. cos_tb is SIGNED (sets sgn). `_kappa_bridge` is the
# shared device harmonic bridge (gpu_image.jl).
@inline function _gpol_kappa(gp, n_e, bb, nu, sin_tb, cos_tb)
    (gp.kappa_frac == 0.0 || n_e <= 0.0 || bb <= 0.0 || nu <= 0.0 || sin_tb <= 0.0) &&
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    k = gp.kappa
    w = gp.w
    nu_c = _EE * bb / (2.0 * π * _ME * _CL)
    nu_2 = nu * nu
    sin2 = sin_tb * sin_tb
    sgn = cos_tb >= 0.0 ? 1.0 : -1.0
    nu_kappa = nu_c * w * w * k * k * sin_tb
    xx = nu / nu_kappa
    s = 1.0 / sqrt(xx)

    # Emissivity.
    va = gp.kappa_frac * n_e * _EE * _EE * nu_c / (_CL * nu_2)
    c_low = gp.kappa_jj_low * va * cbrt(xx) * sin_tb
    c_high = gp.kappa_jj_high * va * xx^(-(k - 2.0) / 2.0) * sin_tb
    j_i = _kappa_bridge(c_low, c_high, gp.kappa_jj_x_i)
    j_q = -_kappa_bridge(c_low * gp.kappa_jj_low_q, c_high * gp.kappa_jj_high_q, gp.kappa_jj_x_q)
    jv_low = c_low * gp.kappa_jj_low_v * (sin_tb^(-2.4) - 1.0)^0.48 * xx^(-0.35)
    jv_high = c_high * gp.kappa_jj_high_v * (sin_tb^(-2.5) - 1.0)^0.44 * s
    j_v = _kappa_bridge(jv_low, jv_high, gp.kappa_jj_x_v) * sgn

    # Absorptivity.
    vb = gp.kappa_frac * n_e * _EE * _EE / (_ME * _CL)
    a_low = gp.kappa_aa_low * vb * xx^(-2.0 / 3.0)
    a_high = gp.kappa_aa_high * vb * xx^(-(1.0 + k) / 2.0)
    a_i = _kappa_bridge(a_low, a_high * gp.kappa_aa_high_i, gp.kappa_aa_x_i)
    a_q = -_kappa_bridge(a_low * gp.kappa_aa_low_q, a_high * gp.kappa_aa_high_q, gp.kappa_aa_x_q)
    av_low = a_low * gp.kappa_aa_low_v * (sin_tb^(-2.28) - 1.0)^0.446 * xx^(-0.35)
    av_high = a_high * gp.kappa_aa_high_v * sqrt(sin_tb^(-2.05) - 1.0) * s
    a_v = _kappa_bridge(av_low, av_high, gp.kappa_aa_x_v) * sgn

    # Rotativities.
    ra = -gp.kappa_frac * n_e * _EE * _EE * nu_c * nu_c * sin2 / (_ME * _CL * nu_2)
    rb = gp.kappa_frac * 2.0 * n_e * _EE * _EE * nu_c * cos_tb / (_ME * _CL * nu)
    rho_q_low = ra * gp.kappa_rho_q_low_a * (1.0 -
        exp(gp.kappa_rho_q_low_b * xx^0.84) -
        sin(gp.kappa_rho_q_low_c * xx) * exp(gp.kappa_rho_q_low_d * xx^gp.kappa_rho_q_low_e))
    rho_q_high = ra * gp.kappa_rho_q_high_a * (1.0 -
        exp(gp.kappa_rho_q_high_b * xx^0.84) -
        sin(gp.kappa_rho_q_high_c * xx) * exp(gp.kappa_rho_q_high_d * xx^gp.kappa_rho_q_high_e))
    rho_v_low = gp.kappa_rho_v * rb * gp.kappa_rho_v_low_a * (1.0 - 0.17 * log(1.0 + gp.kappa_rho_v_low_b * s))
    rho_v_high = gp.kappa_rho_v * rb * gp.kappa_rho_v_high_a * (1.0 - 0.17 * log(1.0 + gp.kappa_rho_v_high_b * s))
    rho_q = (1.0 - gp.kappa_rho_frac) * rho_q_low + gp.kappa_rho_frac * rho_q_high
    rho_v = (1.0 - gp.kappa_rho_frac) * rho_v_low + gp.kappa_rho_frac * rho_v_high

    return (j_i, j_q, j_v, a_i, a_q, a_v, rho_q, rho_v)
end

# ---------------------------------------------------------------- device: fluid tetrad
# Port of tetrads.jl :: fluid_tetrad. Returns the 4x4 whose row a is e_(a)^μ.
@inline function _gpol_tetrad(ucon::SVector{4}, ucov::SVector{4}, kcon::SVector{4},
                              kcov::SVector{4}, up::SVector{4},
                              gcov::SMatrix{4,4}, gcon::SMatrix{4,4})
    omega = -dot(kcov, ucon)
    k_up = dot(kcov, up) / omega
    u_up = dot(ucov, up) / omega
    e0 = ucon
    e3 = kcon ./ omega .- ucon
    e2u = up .- k_up .* e3 .+ u_up .* kcon
    e2u_n2 = dot(e2u, gcov * e2u)
    # GUARD(DEVIATION): mirror of the CPU fluid_tetrad B ∥ k degeneracy guard
    # (radiation/tetrads.jl) — when `up` lies in span(u, k) the perpendicular part
    # e2u vanishes and e2u / sqrt(0) NaN-poisons the whole ray. Substitute the
    # coordinate axis most orthogonal to e3, projected onto span(u,k)^⊥ (ipole
    # set_Econ_from_trial fallback); non-degenerate samples take the identical
    # sqrt(e2u_n2) normalization as before.
    if !(e2u_n2 > 1.0e-12 * dot(up, gcov * up))
        s2 = e3[2] * e3[2]; s3 = e3[3] * e3[3]; s4 = e3[4] * e3[4]
        trial = (s2 <= s3 && s2 <= s4) ? SVector(0.0, 1.0, 0.0, 0.0) :
                (s3 <= s4)             ? SVector(0.0, 0.0, 1.0, 0.0) :
                                         SVector(0.0, 0.0, 0.0, 1.0)
        e2u = trial .- (dot(kcov, trial) / omega) .* e3 .+ (dot(ucov, trial) / omega) .* kcon
        e2u_n2 = dot(e2u, gcov * e2u)
    end
    e2 = e2u ./ sqrt(e2u_n2)
    e1c1 = e0[2]*(e2[4]*e3[3]-e2[3]*e3[4]) + e0[3]*(e2[2]*e3[4]-e2[4]*e3[2]) + e0[4]*(e2[3]*e3[2]-e2[2]*e3[3])
    e1c2 = e0[1]*(e2[3]*e3[4]-e2[4]*e3[3]) + e0[3]*(e2[4]*e3[1]-e2[1]*e3[4]) + e0[4]*(e2[1]*e3[3]-e2[3]*e3[1])
    e1c3 = e0[1]*(e2[4]*e3[2]-e2[2]*e3[4]) + e0[2]*(e2[1]*e3[4]-e2[4]*e3[1]) + e0[4]*(e2[2]*e3[1]-e2[1]*e3[2])
    e1c4 = e0[1]*(e2[2]*e3[3]-e2[3]*e3[2]) + e0[2]*(e2[3]*e3[1]-e2[1]*e3[3]) + e0[3]*(e2[1]*e3[2]-e2[2]*e3[1])
    e1 = gcon * SVector(e1c1, e1c2, e1c3, e1c4)
    return @inbounds SMatrix{4,4}(e0[1], e1[1], e2[1], e3[1],
                                  e0[2], e1[2], e2[2], e3[2],
                                  e0[3], e1[3], e2[3], e3[3],
                                  e0[4], e1[4], e2[4], e3[4])
end

# ---------------------------------------------------------------- device: extended RK4 RHS
# One AD pass of the closed-form KS metric gives ∂g^{ab} (geodesic dp) AND ∂g_{ab}
# (parallel transport of eU). Returns (dy::SVector{8}, deU::SVector{4}).
@inline function _gpol_ext_rhs(y::SVector{8,Float64}, eU::SVector{4,Float64}, a)
    x = SVector(y[1], y[2], y[3], y[4]); p = SVector(y[5], y[6], y[7], y[8])
    z = 0.0; o = 1.0
    # 3 spatial AD lanes (∂_2,∂_3,∂_4); the time lane is dead work (∂_t g = 0). Single-
    # vector reference variant — the march transports two legs via _gpol_ext_rhs2 to
    # share one metric evaluation.
    xd = SVector(Dual(x[1], (z, z, z)), Dual(x[2], (o, z, z)),
                 Dual(x[3], (z, o, z)), Dual(x[4], (z, z, o)))
    gcovd, gcond = _gimg_metric(xd, a)
    gcon = SMatrix{4,4,Float64}(ntuple(k -> gcond[k].value, Val(16)))
    kcon = gcon * p                                   # dx^μ/dλ
    dp1 = z; dp2 = z; dp3 = z
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]; prt = partials(gcond[μ, ν])
        dp1 -= pp * prt[1]; dp2 -= pp * prt[2]; dp3 -= pp * prt[3]
    end
    dy = SVector(kcon[1], kcon[2], kcon[3], kcon[4], z, dp1 / 2, dp2 / 2, dp3 / 2)
    # ∂_c g_{ab} (c = 2,3,4; ∂_1 = 0)
    dg2 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[1], Val(16)))
    dg3 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[2], Val(16)))
    dg4 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[3], Val(16)))
    Dk  = kcon[2] .* dg2 .+ kcon[3] .* dg3 .+ kcon[4] .* dg4       # k^c ∂_c g_{ab}
    De  = eU[2]  .* dg2 .+ eU[3]  .* dg3 .+ eU[4]  .* dg4          # eU^c ∂_c g_{ab}
    t1 = Dk * eU                                                   # (k·∂ g) eU
    t2 = De * kcon                                                 # (eU·∂ g) k
    t3 = SVector(z, dot(kcon, dg2 * eU), dot(kcon, dg3 * eU), dot(kcon, dg4 * eU))
    T = 0.5 .* (gcon * (t1 .+ t2 .- t3))                          # Γ^μ_{αβ} k^α eU^β
    return dy, -T
end

# Two-vector variant: parallel-transport BOTH eU (camera up) AND eV (camera
# 4-velocity) along the geodesic, sharing the single AD metric evaluation. The readout
# screen must be the PARALLEL-TRANSPORTED camera tetrad, so its timelike leg is
# PT(u_cam) — not the per-sample local ZAMO, which drifts uncorrelated with parallel
# transport and rotates EVPA per ray (cf. ipole/Krang). The frame transport rides the
# geodesic lane T ∈ {Float32, Float64} (a duplicated fp64 connection eval would eat the
# fp32 saving); `a::T` guards against silent promotion. zero(T)/T(0.5) are 0.0/0.5
# bit-for-bit at Float64, so the fp64 path is unchanged.
@inline function _gpol_ext_rhs2(y::SVector{8,T}, eU::SVector{4,T},
                                eV::SVector{4,T}, a::T) where {T}
    x = SVector(y[1], y[2], y[3], y[4]); p = SVector(y[5], y[6], y[7], y[8])
    z = zero(T); o = one(T)
    # 3 spatial AD lanes (∂_2,∂_3,∂_4): the time lane is dead work (stationary KS,
    # ∂_t g = 0); dropping it is byte-identical on the CUDA/NVPTX backend and trims
    # local-memory spill.
    xd = SVector(Dual(x[1], (z, z, z)), Dual(x[2], (o, z, z)),
                 Dual(x[3], (z, o, z)), Dual(x[4], (z, z, o)))
    gcovd, gcond = _gimg_metric(xd, a)
    gcon = SMatrix{4,4,T}(ntuple(k -> gcond[k].value, Val(16)))
    kcon = gcon * p
    dp1 = z; dp2 = z; dp3 = z
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]; prt = partials(gcond[μ, ν])
        dp1 -= pp * prt[1]; dp2 -= pp * prt[2]; dp3 -= pp * prt[3]
    end
    dy = SVector(kcon[1], kcon[2], kcon[3], kcon[4], z, dp1 / 2, dp2 / 2, dp3 / 2)
    dg2 = SMatrix{4,4,T}(ntuple(k -> partials(gcovd[k])[1], Val(16)))
    dg3 = SMatrix{4,4,T}(ntuple(k -> partials(gcovd[k])[2], Val(16)))
    dg4 = SMatrix{4,4,T}(ntuple(k -> partials(gcovd[k])[3], Val(16)))
    Dk  = kcon[2] .* dg2 .+ kcon[3] .* dg3 .+ kcon[4] .* dg4
    @inline function _transport(v)
        Dv = v[2] .* dg2 .+ v[3] .* dg3 .+ v[4] .* dg4
        t1 = Dk * v; t2 = Dv * kcon
        t3 = SVector(z, dot(kcon, dg2 * v), dot(kcon, dg3 * v), dot(kcon, dg4 * v))
        return -(T(0.5) .* (gcon * (t1 .+ t2 .- t3)))
    end
    return dy, _transport(eU), _transport(eV)
end

# ---------------------------------------------------------------- device: pitch + 8 coeffs
# Fluid decode + plasma cuts + SIGNED pitch angle (fluid-tetrad projection, per
# eight_coefficients) + thermal polarized coeffs. Returns (j,α,ρ,tet_fl,valid).
# NOTE(bit-exactness): the decode/cut front half deliberately stays a sibling of
# _gimg_coeffs' (not a shared helper) — see the fmuladd-contraction note there.
# `tetfn` is the fluid-tetrad builder: the default `_gpol_tetrad` reproduces the Kerr
# kernels bit-for-bit (|det g| = 1 on single CKS, no e_(1) renorm needed); the generic-
# spacetime pol march passes the renormalizing `_gsp_tetrad` (gpu_spacetime.jl),
# mirroring the CPU `fluid_tetrad(...; renorm_e1 = !(st isa KerrSchildCartesian))` key.
@inline function _gpol_coeffs(prim::SVector{8,Float64}, x::SVector{4,Float64},
                              kcov::SVector{4,Float64}, gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, gp,
                              tetfn::F = _gpol_tetrad) where {F}
    rho = prim[1]; eint = prim[5]
    (rho <= 0.0 || eint <= 0.0) && return (_gpz4, _gpz4, _gpz4, _gpI4, false)
    pgas = (gp.gamma - 1.0) * eint
    ucon, ucov, bcon, b_sq = _gimg_pfv(prim[2], prim[3], prim[4], prim[6], prim[7], prim[8], gcov, gcon)
    b_sq <= 0.0 && return (_gpz4, _gpz4, _gpz4, _gpI4, false)
    kcon = gcon * kcov
    rho_cgs = rho * gp.d_unit; pgas_cgs = pgas * gp.e_unit
    n_cgs = rho_cgs / (gp.mu * _MP); n_e = n_cgs / (1.0 + 1.0 / gp.ne_ni)
    bb_cgs = sqrt(b_sq) * gp.b_unit
    sigma = b_sq / rho; beta_inv = b_sq / (2.0 * pgas)
    theta_e = _gimg_theta_e(rho_cgs, pgas_cgs, beta_inv, gp)
    (gp.sigma_cut >= 0.0 && sigma > gp.sigma_cut) && return (_gpz4, _gpz4, _gpz4, _gpI4, false)
    (gp.theta_e_min >= 0.0 && theta_e < gp.theta_e_min) && return (_gpz4, _gpz4, _gpz4, _gpI4, false)
    bcov = gcov * bcon
    tet = tetfn(ucon, ucov, kcon, kcov, bcon, gcov, gcon)
    kt1 = kt2 = kt3 = 0.0; bt1 = bt2 = bt3 = 0.0
    @inbounds for μ in 1:4
        kt1 += tet[2, μ] * kcov[μ]; kt2 += tet[3, μ] * kcov[μ]; kt3 += tet[4, μ] * kcov[μ]
        bt1 += tet[2, μ] * bcov[μ]; bt2 += tet[3, μ] * bcov[μ]; bt3 += tet[4, μ] * bcov[μ]
    end
    k2t = kt1^2 + kt2^2 + kt3^2; b2t = bt1^2 + bt2^2 + bt3^2
    kbt = kt1 * bt1 + kt2 * bt2 + kt3 * bt3
    cos2 = min(kbt * kbt / (k2t * b2t), 1.0)
    sin_tb = sqrt(1.0 - cos2); cos_tb = sqrt(cos2) * (kbt >= 0.0 ? 1.0 : -1.0)
    minus_ku = -dot(kcov, ucon)
    nu_cgs = minus_ku * gp.nu_obs
    ji, jq, jv, ai, aq, av, rq, rv = _gpol_thermal(gp.thermal_frac, n_e, theta_e, bb_cgs, nu_cgs, sin_tb, cos_tb)
    # Additive nonthermal eDF mixing in the CPU eight_coefficients order (thermal, then
    # power-law, then kappa, on all eight coefficients incl. rotativities). Each branch
    # is keyed on the params tuple carrying that eDF's constants (compile-time
    # `hasfield`), so a thermal-only deck compiles to the untouched thermal path.
    if hasfield(typeof(gp), :power_frac)
        tj_i, tj_q, tj_v, ta_i, ta_q, ta_v, tr_q, tr_v =
            _gpol_powerlaw(gp, n_e, bb_cgs, nu_cgs, sin_tb, cos_tb)
        ji += tj_i; jq += tj_q; jv += tj_v
        ai += ta_i; aq += ta_q; av += ta_v
        rq += tr_q; rv += tr_v
    end
    if hasfield(typeof(gp), :kappa_frac)
        kj_i, kj_q, kj_v, ka_i, ka_q, ka_v, kr_q, kr_v =
            _gpol_kappa(gp, n_e, bb_cgs, nu_cgs, sin_tb, cos_tb)
        ji += kj_i; jq += kj_q; jv += kj_v
        ai += ka_i; aq += ka_q; av += ka_v
        rq += kr_q; rv += kr_v
    end
    return (SVector(ji, jq, 0.0, jv), SVector(ai, aq, 0.0, av), SVector(0.0, rq, 0.0, rv), tet, true)
end

# ---------------------------------------------------------------- device: segment (M, b)
# Analytic constant-coefficient transmission M_seg and emission b_seg (S -> M S + b) over
# one segment, matching transfer.jl :: stokes_couple (U component zero). The CPU
# per-segment admissibility clamp is a safeguard on the marching TOTAL, not on this
# linear segment operator — the kernels apply its mirror (_gpol_admissible) to the
# accumulated Stokes after every segment instead.
@inline function _gpol_segMB(j::SVector{4}, α::SVector{4}, ρ::SVector{4}, ds)
    alpha_sq = α[2]*α[2] + α[4]*α[4]; alpha_p = sqrt(alpha_sq)
    rho_sq = ρ[2]*ρ[2] + ρ[4]*ρ[4]; rho_p = sqrt(rho_sq)
    aI = α[1]; delta_tau = aI * ds; thin = delta_tau <= 100.0
    if aI == 0.0 && rho_p == 0.0
        return _gpI4, j .* ds
    elseif alpha_p == 0.0 && rho_p == 0.0
        if thin
            e = exp(-delta_tau)
            return e .* _gpI4, (j ./ aI) .* (-expm1(-delta_tau))   # (1-e^{-dτ}) j/aI
        else
            return _gpz44, j ./ aI
        end
    elseif aI == 0.0
        cr = cos(rho_p*ds); sr = sin(rho_p*ds); s2 = sin(rho_p*ds/2.0); ss = s2*s2
        rQ = ρ[2]; rV = ρ[4]
        # linear map S -> M S (I unchanged), b = j*ds
        M = @SMatrix [1.0 0.0 0.0 0.0;
                      0.0 (cr+2.0*rQ*rQ/rho_sq*ss) (-rV/rho_p*sr) (2.0*rQ*rV/rho_sq*ss);
                      0.0 (rV/rho_p*sr) cr (-rQ/rho_p*sr);
                      0.0 (2.0*rV*rQ/rho_sq*ss) (rQ/rho_p*sr) (cr+2.0*rV*rV/rho_sq*ss)]
        return M, j .* ds
    else
        aQ = α[2]; aV = α[4]; rQ = ρ[2]; rV = ρ[4]
        ar = aQ*rQ + aV*rV; asr = alpha_sq - rho_sq
        la = sqrt(asr*asr/4.0 + ar*ar); lb = asr/2.0
        l1 = sqrt(la + lb); l2 = sqrt(la - lb)
        ct = l1*l1 + l2*l2; s = ar >= 0.0 ? 1.0 : -1.0; a0 = aI*aI
        if ct <= 1e-12*(alpha_sq+rho_sq) || abs(a0 - l1*l1) <= 1e-12*(a0 + l1*l1)
            if thin
                e = exp(-delta_tau)
                return e .* _gpI4, (j ./ aI) .* (-expm1(-delta_tau))
            else
                return _gpz44, j ./ aI
            end
        end
        it = 1.0 / ct
        m2 = it .* @SMatrix [0.0 (l2*aQ-s*l1*rQ) 0.0 (l2*aV-s*l1*rV);
                             (l2*aQ-s*l1*rQ) 0.0 (s*l1*aV+l2*rV) 0.0;
                             0.0 -(s*l1*aV+l2*rV) 0.0 (s*l1*aQ+l2*rQ);
                             (l2*aV-s*l1*rV) 0.0 -(s*l1*aQ+l2*rQ) 0.0]
        m3 = it .* @SMatrix [0.0 (l1*aQ+s*l2*rQ) 0.0 (l1*aV+s*l2*rV);
                             (l1*aQ+s*l2*rQ) 0.0 (-(s*l2*aV-l1*rV)) 0.0;
                             0.0 (s*l2*aV-l1*rV) 0.0 (-(s*l2*aQ-l1*rQ));
                             (l1*aV+s*l2*rV) 0.0 (s*l2*aQ-l1*rQ) 0.0]
        d0 = (alpha_sq + rho_sq) / 2.0
        m4 = (2.0*it) .* @SMatrix [d0 0.0 (aQ*rV-aV*rQ) 0.0;
                                   0.0 (aQ*aQ+rQ*rQ-d0) 0.0 (aV*aQ+rV*rQ);
                                   -(aQ*rV-aV*rQ) 0.0 -d0 0.0;
                                   0.0 (aV*aQ+rV*rQ) 0.0 (aV*aV+rV*rV-d0)]
        p4 = _gpI4 .+ m4; m4m = _gpI4 .- m4
        f1 = 1.0 / (a0 - l1*l1); f2 = 1.0 / (a0 + l2*l2)
        cosh_t = (-l1*f1) .* m3 .+ (0.5*aI*f1) .* p4
        cos_t  = (-l2*f2) .* m2 .+ (0.5*aI*f2) .* m4m
        pp = cosh_t .+ cos_t
        if thin
            ev = exp(-delta_tau); sv = sin(l2*ds); cv = cos(l2*ds); shv = sinh(l1*ds); chv = cosh(l1*ds)
            sin_t  = (-aI*f2) .* m2 .+ (-0.5*l2*f2) .* m4m
            sinh_t = (-aI*f1) .* m3 .+ (0.5*l1*f1) .* p4
            pp = pp .- ev .* (chv .* cosh_t .+ cv .* cos_t .+ sv .* sin_t .+ shv .* sinh_t)
            oo = ev .* (0.5 .* chv .* p4 .+ 0.5 .* cv .* m4m .- sv .* m2 .- shv .* m3)
            return oo, pp * j
        else
            return _gpz44, pp * j
        end
    end
end

# Physical admissibility of the accumulated camera-frame Stokes: I >= 0 and |P| <= I —
# the mirror of the CPU per-segment clamp (transfer.jl :: stokes_couple output). The CPU
# march clamps the running physical S source->camera; in the reverse Mueller chain the
# accumulator IS the observed Stokes of the camera-side truncated problem, so the same
# invariant applies after every segment accumulation. A no-op on admissible values
# (bit-identical outputs); it bounds numerical excursions exactly as the CPU clamp does.
@inline function _gpol_admissible(S::SVector{4})
    ii = max(S[1], 0.0)
    pol = S[2] * S[2] + S[3] * S[3] + S[4] * S[4]
    if pol > ii * ii
        f = sqrt(ii * ii / pol)
        return SVector(ii, S[2] * f, S[3] * f, S[4] * f)
    end
    return SVector(ii, S[2], S[3], S[4])
end

# ---------------------------------------------------------------- device: analytic R
# Stokes rotation fluid-tetrad -> readout-tetrad E: pure Q-U rotation by 2χ (I,V fixed),
# χ = angle of E's e_(1) in the fluid (e_(1),e_(2)) basis. Reproduces the coherency
# round-trip _coord_to_tetrad_cov (validated against the CPU path).
@inline function _gpol_R(tet_fl::SMatrix{4,4}, E::SMatrix{4,4}, gcov::SMatrix{4,4})
    e1f = SVector(tet_fl[2,1], tet_fl[2,2], tet_fl[2,3], tet_fl[2,4])
    e2f = SVector(tet_fl[3,1], tet_fl[3,2], tet_fl[3,3], tet_fl[3,4])
    E1  = SVector(E[2,1], E[2,2], E[2,3], E[2,4])
    c = dot(E1, gcov * e1f); s = dot(E1, gcov * e2f)
    nrm = sqrt(c*c + s*s); c /= nrm; s /= nrm
    c2 = c*c - s*s; s2 = 2.0*s*c
    return @SMatrix [1.0 0.0 0.0 0.0; 0.0 c2 s2 0.0; 0.0 -s2 c2 0.0; 0.0 0.0 0.0 1.0]
end

# ---------------------------------------------------------------- device: stiffness detector
# DIAGNOSTIC-ONLY transfer-matrix stiffness detector (RAPTOR check_stiffness; RAPTOR is
# GPLv3, so the algorithm is re-derived, not copied). The 4x4 polarized transfer matrix
# (LD85 / transfer.jl convention) has eigenvalues λ = α_I ± λ1 (real) and α_I ± i·λ2,
# with (λ1, λ2) the same combinations _gpol_segMB forms for its analytic solution.
# `_stiffness_metric` returns the dominant eigenvalue scale max(|λ|)·ds; a segment is
# 'stiff' above -log(0.99) (RAPTOR's |e^{τλ}| > 0.99 criterion — an explicit RK4 step
# would be poorly conditioned). The analytic coupling handles stiffness exactly
# regardless; this is telemetry only and does NOT touch the transport numerics.
const _STIFF_METRIC_THRESH = -log(0.99)   # ≈ 0.010050

@inline function _stiffness_lambdas(α::SVector{4}, ρ::SVector{4})
    alpha_sq = α[2]*α[2] + α[4]*α[4]
    rho_sq   = ρ[2]*ρ[2] + ρ[4]*ρ[4]
    ar  = α[2]*ρ[2] + α[4]*ρ[4]
    asr = alpha_sq - rho_sq
    la  = sqrt(asr*asr/4.0 + ar*ar)
    l1  = sqrt(max(la + asr/2.0, 0.0))    # real eigenvalue offset  (α_I ± λ1)
    l2  = sqrt(max(la - asr/2.0, 0.0))    # imag eigenvalue offset  (α_I ± i λ2)
    return l1, l2
end

"Dominant transfer-matrix eigenvalue scale max(|λ|)·ds (device-safe, real arithmetic)."
@inline function _stiffness_metric(α::SVector{4}, ρ::SVector{4}, ds)
    l1, l2 = _stiffness_lambdas(α, ρ)
    aI = α[1]
    mre = max(abs(aI + l1), abs(aI - l1))    # real eigenvalues
    mim = sqrt(aI*aI + l2*l2)                # |α_I ± i λ2|
    return max(mre, mim) * ds
end

"Stiff-segment flag: max(|λ|)·ds exceeds the RAPTOR |e^{τλ}|>0.99 threshold."
@inline _stiffness_is_stiff(α::SVector{4}, ρ::SVector{4}, ds, thresh = _STIFF_METRIC_THRESH) =
    _stiffness_metric(α, ρ, ds) > thresh

# Reference implementation (host-only; ComplexF64, NOT used on-device): the RK4
# stability polynomial |1+τλ+τ²λ²/2+τ³λ³/6+τ⁴λ⁴/24| maximized over the four
# eigenvalues. Used in tests to show _stiffness_lambdas matches and the metric gate
# tracks the RK4 criterion.
function _stiffness_rk4_maxmag(α::SVector{4}, ρ::SVector{4}, ds)
    aI = α[1]; aQ = α[2]; aV = α[4]; rQ = ρ[2]; rV = ρ[4]
    a2 = rQ*rQ + rV*rV - aQ*aQ - aV*aV
    a0 = -2.0*aV*aQ*rV*rQ - aQ*aQ*rQ*rQ - aV*aV*rV*rV
    disc = sqrt(Complex(a2*a2 - 4.0*a0))
    zplus  = (-a2 + disc) / 2
    zminus = (-a2 - disc) / 2
    ls = (aI + sqrt(zplus), aI - sqrt(zplus), aI + sqrt(zminus), aI - sqrt(zminus))
    mm = 0.0
    for l in ls
        τ = ds * l
        mag = abs(1 + τ + τ*τ/2 + τ*τ*τ/6 + τ*τ*τ*τ/24)
        mm = max(mm, mag)
    end
    return mm
end

# Unit normal (Eulerian) observer n^μ = -α g^{0μ} at the camera — the CPU
# Radiation._normal_observer itself (pure @inline SVector math, hence device-safe).
# Kept under the _gpol_ name for the kernels/tests; parallel-transported here as the
# readout screen's timelike leg.
using BlackLightPlus.Radiation: _normal_observer
const _gpol_normal_observer = _normal_observer
# Relative drift of the transported invariant -k·eUu that ends a polarized march.
const _GPOL_FRAME_TOL = 1.0e-2

# ---------------------------------------------------------------- fused polarized march
# The entire polarized fused march for one ray, camera->source. Per-thread state:
# geodesic y(8), transported up-vector eU(4) + timelike leg eUu(4), accumulated Stokes
# S(4), running Mueller transmission G(16) — all real. The one body behind every
# polarized kernel variant, parameterized by the inlined fluid sampler `sample(y)`.
# Two-type: the geodesic state and transported frame legs run in Tg ∈ {Float32, Float64}
# with matched-::Tg march scalars; the state is converted to Float64 ONCE per fluid-
# sample/RT step (identity when Tg == Float64), so the Stokes/Mueller coupling and every
# coefficient stay Float64 in every mode. The RT-lane metric uses the fp64 spin gp.bh_a.
@inline function _gpol_march(y0::SVector{8,Tg}, sample::F,
                             gp, a::Tg, r_h::Tg, C::Tg, escape::Tg, R_em::Tg,
                             nmax) where {F,Tg}
    y = y0
    # camera readout frame: built in Float64 from the CONVERTED initial state (identity
    # at Tg == Float64), then NARROWED onto the Tg transport lane
    y64 = SVector{8,Float64}(y0)
    x0 = SVector(y64[1], y64[2], y64[3], y64[4]); kcov0 = -SVector(y64[5], y64[6], y64[7], y64[8])
    gcov0, gcon0 = _gimg_metric(x0, gp.bh_a); kcon0 = gcon0 * kcov0
    # camera readout tetrad -> initial transported up-vector eU (= its e_(2))
    e0c = _gpol_normal_observer(gcon0)   # unit Eulerian observer (CPU shift-sign fix mirror)
    e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = _gpol_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0)
    eU  = SVector{4,Tg}(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])   # camera e_(2) (up)
    eUu = SVector{4,Tg}(e0c)     # camera 4-velocity — parallel-transported alongside eU
    wcam = -dot(kcov0, e0c)      # -k·u_cam: conserved by exact parallel transport

    S = _gpz4; G = _gpI4
    r_prev = Tg(Inf)
    for _ in 1:nmax
        r = _gimg_rks(y, a)
        _gimg_stop(r, r_prev, r_h, escape, R_em) && break
        r_prev = r
        h = (r - r_h) / C
        # callsite @inline: the inliner otherwise leaves the sampler closure as a call
        # in this register-heavy body, which perturbs downstream fmuladd contraction
        # (~1 ulp, Faraday-amplified on dim pixels); forced inlining keeps the kernel
        # bit-for-bit stable.
        prim = @inline sample(y)
        if prim[1] > 0.0
            # RT lane: Float64 from the CONVERTED state (identity at Tg == Float64)
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            gcov, gcon = _gimg_metric(x, gp.bh_a)
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8]); kcon = gcon * kcov   # photon k = -p
            j, al, ro, tet_fl, valid = _gpol_coeffs(prim, x, kcov, gcov, gcon, gp)
            if valid
                # readout screen = the parallel-transported camera tetrad: its
                # timelike leg is PT(u_cam)=eUu (not the per-sample local ZAMO,
                # which drifts and rotates EVPA per ray). _gpol_tetrad uses only
                # parallel-transport-invariant operations, so E == PT(Ecam) and S
                # ends up in the camera frame with no final rotation (cf. ipole/Krang).
                eU64 = SVector{4,Float64}(eU); eUu64 = SVector{4,Float64}(eUu)
                # The backward ray approaches the future horizon asymptotically, where
                # the transported frame's KS components grow like 1/(r - r_h) and the
                # RK4 transport loses the invariant -k·eUu. Stop once it drifts by 1%:
                # there ν/ν_obs = -k·u ≳ 10² and the emission is negligible.
                abs(-dot(kcov, eUu64) - wcam) > _GPOL_FRAME_TOL * wcam && break
                E = _gpol_tetrad(eUu64, gcov * eUu64, kcon, kcov, eU64, gcov, gcon)
                ds = Float64(h) * gp.L_unit / gp.nu_obs
                M, b = _gpol_segMB(j, al, ro, ds)
                R = _gpol_R(tet_fl, E, gcov)
                S = _gpol_admissible(S + G * (R * b))   # CPU stokes_couple clamp mirror
                G = G * (R * M * transpose(R))
            end
        end
        # low-storage streaming RK4: same ops/associativity as the 4-stage combine but
        # the four stages of {y,eU,eUu} are not all held live -> fewer registers/thread.
        dy, de, du = _gpol_ext_rhs2(y, eU, eUu, a)
        ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a)
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a)
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, de, du = _gpol_ext_rhs2(ys, es, us, a)
        ay = ay + dy; ae = ae + de; au = au + du
        y   = y   + (h/6)*ay
        eU  = eU  + (h/6)*ae
        eUu = eUu + (h/6)*au
    end
    return S * gp.nu_obs^3
end

# ---------------------------------------------------------------- fused polarized kernel
# One ray/thread on the uniform grid; writes out[i]. Two-type: the geodesic/frame lane
# runs at eltype(eltype(y0s)) — the host passes (a, r_h, C, escape, R_em) at that type —
# while the sampler position is converted to Float64 (identity for fp64 rays), so the
# trilinear sample + all Stokes/coefficient arithmetic stay Float64 in every mode.
@kernel function _gpol_kernel!(out, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                               R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gpol_march(y0s[i],
                             y -> _gimg_sample(grid, ng, xo, inv_dx, Float64(y[2]),
                                               Float64(y[3]), Float64(y[4])),
                             gp, a, r_h, C, escape, R_em, nmax)
    end
end

# ---------------------------------------------------------------- host driver
function _gpol_params(cfg)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs); p = cfg.plasma
    i = deg2rad(cfg.camera.inclination_deg)
    base = (d_unit = u.dens_unit, e_unit = u.dens_unit * _CL^2, b_unit = u.B_unit,
            mu = p.mu, ne_ni = p.ne_ni, use_p = (p.use_p ? 1 : 0),
            rat_low = p.rat_low, rat_high = p.rat_high, gamma = p.gamma,
            gamma_i = p.gamma_i, gamma_e = p.gamma_e, thermal_frac = p.thermal_frac,
            sigma_cut = p.sigma_cut, theta_e_min = p.theta_e_min,
            nu_obs = cfg.frequency_hz, L_unit = u.L_unit, bh_a = cfg.bh_a,
            cu1 = 0.0, cu2 = -cos(i), cu3 = 0.0, cu4 = sin(i))   # camera vertical (image β)
    # Merge the host-precomputed nonthermal constants when active (shares _powerlaw_gp /
    # _kappa_gp with the unpol path); a thermal deck keeps the byte-identical tuple.
    return _merge_nonthermal(base, p)
end

# The polarized kernel is register-heavy (parallel transport + Mueller chain) and
# register-capped at 255 in both precisions, so only ~256 threads fit per SM; wg=64
# yields 4 resident blocks (best latency hiding) and measured fastest. Block size is
# occupancy-only (results are bit-identical across wg).
const _GPOL_WG = 64

function _gpol_part!(out, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C, escape, R_em, nmax)
    backend = CUDABackend()
    kern = _gpol_kernel!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{SVector{4,Float64}}(undef, cnt)
    kern(outd, y0d, grid_d, gp, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx, cnt;
         ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

# Launch the polarized kernel over the ray-index `pool` (any index collection), pixels
# INTERLEAVED across `devices` (the _fused_sweep! pattern — pool == 1:nrays reproduces
# the pre-mixed-precision device split exactly). The geodesic/frame-lane type is
# eltype(eltype(rays)); the caller passes (a, r_h, C, escape, R_em) at that type.
function _gpol_sweep!(out, rays, pool, devices, dgrid, gp, a, r_h, C, escape, R_em, nmax)
    nd = length(devices)
    @sync for di in 1:nd
        idx = pool[di:nd:end]; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _gpol_part!(out, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                        dgrid.inv_dx, gp, a, r_h, C, escape, R_em, nmax)
        end
    end
    return nothing
end

"""
    render_gpu_polarized(cfg, snap_or_grid; devices=select_devices(), nmax=cfg.max_steps,
                         C=80, box_half=40.0, n=256, precision=:f64, rescue_px=2)
        -> NamedTuple(I,Q,U,V)

Fused single-pass GPU render of the full-Stokes npix×npix polarized image: one thread
per ray marches the null geodesic camera->source, parallel-transports the polarization
frame, and accumulates the observed Stokes with a running Mueller transmission (reverse
RT). `snap_or_grid`: `AthenaKSnapshot`, `UniformFluidGrid`, or pre-uploaded
`DeviceFluidGrid`. Each of I,Q,U,V is a `Matrix{Float64}` in `render(...;polarized=true)`
layout; `order` marches rays in trip-count order (bit-exact per pixel). `precision`:
`:f64` (default, bitwise unchanged) · `:mixed` (fp32 geodesic march + frame transport,
Float64 Stokes/Mueller/coefficients from converted state; near-critical-curve pixels
re-rendered in fp64 and spliced — `rescued_frac::Ref{Float64}` receives the fraction) ·
`:f32_raw` (no rescue, DIAGNOSTIC only — near-critical pixels carry O(1) errors).
"""
function render_gpu_polarized(cfg, snap_or_grid; devices = select_devices(),
                              nmax::Int = cfg.max_steps,   # step limit wired from RunConfig, as render()
                              C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                              order = nothing,
                              precision::Symbol = :f64, rescue_px::Int = 2,
                              rescued_frac = nothing)
    precision in (:f64, :mixed, :f32_raw) ||
        error("render_gpu_polarized: precision must be :f64, :mixed or :f32_raw, got :$precision")
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)

    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)          # fluid box [-|xo|,|xo|]^3 -> vacuum for r > R_em
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    frac = 0.0
    if precision === :f64
        _gpol_sweep!(out, rays, 1:nrays, devices, dgrid, gp, a, r_h, Cf, escape, R_em, nmax)
    else
        # fp32 geodesic/frame lane: narrow the fp64-built rays at upload; the host
        # constructs every march scalar as Float32 (no in-kernel widening), while gp
        # (units, frequency, fp64 spin gp.bh_a for the RT lane) stays Float64.
        rays32 = [SVector{8,Float32}(r) for r in rays]
        _gpol_sweep!(out, rays32, 1:nrays, devices, dgrid, gp, Float32(a), Float32(r_h),
                     Float32(Cf), Float32(escape), Float32(R_em), nmax)
        if precision === :mixed
            pix = _rescue_pixels(a, cfg.camera.inclination_deg, cfg.camera.fov, cam.npix;
                                 rescue_px = rescue_px)
            # pixel -> ordered-ray position (out/rays live in `order`ed space until _undo_order)
            pos = order === nothing ? pix : invperm(order)[pix]
            isempty(pos) ||
                _gpol_sweep!(out, rays, pos, devices, dgrid, gp, a, r_h, Cf, escape,
                             R_em, nmax)
            frac = length(pos) / nrays
        end
    end
    rescued_frac === nothing || (rescued_frac[] = frac)
    out = _undo_order(out, order)
    npx = cam.npix
    I = reshape([o[1] for o in out], npx, npx)
    Q = reshape([o[2] for o in out], npx, npx)
    U = reshape([o[3] for o in out], npx, npx)
    V = reshape([o[4] for o in out], npx, npx)
    return (I = I, Q = Q, U = U, V = V)
end

# ================================================================ SLOW LIGHT (opt-in, polarized)
# The SAME _gpol_march body with the sampler swapped for the time-blended
# _gimg_sample_t (gpu_image.jl); w = clamp((t0 + X⁰) * inv_dt, 0, 1) at the emission
# time t_emit = t_obs + X⁰, t0 = t_obs - t_a (the physical dump-time axis — see the
# gpu_image.jl SLOW LIGHT header). inv_dt=0 (fast-light limit) makes _gimg_sample_t
# return grid-A's sample bit-for-bit -> render_gpu_polarized_slow reproduces
# render_gpu_polarized(win.a).
@kernel function _gpol_kernel_sl!(out, @Const(y0s), @Const(gridA), @Const(gridB),
                                  t0, inv_dt, gp, a, r_h, C, escape,
                                  R_em, nmax, ng, xo, inv_dx, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gpol_march(y0s[i],
                             y -> begin
                                 # w on frame B at emission time t_obs + X⁰ (Float64
                                 # RT lane; the conversion is identity for fp64 rays)
                                 w = clamp((t0 + Float64(y[1])) * inv_dt, 0.0, 1.0)
                                 _gimg_sample_t(gridA, gridB, ng, xo, inv_dx, w,
                                                Float64(y[2]), Float64(y[3]),
                                                Float64(y[4]))
                             end,
                             gp, a, r_h, C, escape, R_em, nmax)
    end
end

function _gpol_part_sl!(out, rays, rng, gA, gB, ng, xo, inv_dx, t0, inv_dt,
                        gp, a, r_h, C, escape, R_em, nmax)
    backend = CUDABackend()
    kern = _gpol_kernel_sl!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{SVector{4,Float64}}(undef, cnt)
    kern(outd, y0d, gA, gB, t0, inv_dt, gp, a, r_h, C, escape, R_em, Int(nmax),
         ng, xo, inv_dx, cnt; ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

"""
    render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices=select_devices(),
                              nmax=cfg.max_steps, C=80, t_obs=_win_t_obs(win),
                              time_offset=0.0) -> NamedTuple(I,Q,U,V)

Fused single-pass SLOW-LIGHT polarized render on a `DeviceFluidWindow`: identical
transport to `render_gpu_polarized`, but each ray blends the two window buffers at
its EMISSION time t_emit = t_obs + X⁰ (physical dump-time axis; `t_obs` defaults to
the window's upper end, `time_offset` advances it — movie frame time).
`win.inv_dt == 0` reproduces `render_gpu_polarized(cfg, win.a)` bit-for-bit
(fast-light limit). Accepts `order` as `render_gpu_polarized`.
"""
function render_gpu_polarized_slow(cfg, win::DeviceFluidWindow; devices = select_devices(),
                                   nmax::Int = cfg.max_steps, C::Real = 80,
                                   t_obs::Real = _win_t_obs(win), time_offset::Real = 0.0,
                                   order = nothing)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = _apply_order(camera_rays(st, cam), order)
    ga = win.a; gb = win.b
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(ga.xo)
    t0 = Float64(t_obs) + Float64(time_offset) - win.t_a   # X⁰ at which t_emit reaches t_a
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _gpol_part_sl!(out, rays, idx, ga.dev_data[dev], gb.dev_data[dev], ga.n,
                           ga.xo, ga.inv_dx, t0, win.inv_dt, gp, a, r_h, Cf,
                           escape, R_em, nmax)
        end
    end
    out = _undo_order(out, order)
    npx = cam.npix
    I = reshape([o[1] for o in out], npx, npx)
    Q = reshape([o[2] for o in out], npx, npx)
    U = reshape([o[3] for o in out], npx, npx)
    V = reshape([o[4] for o in out], npx, npx)
    return (I = I, Q = Q, U = U, V = V)
end

# ---------------------------------------------------------------- stiffness telemetry kernel
# DIAGNOSTIC marcher: same geodesic stepping + fluid decode as _gpol_kernel! (reuses
# _gimg_rhs — whose dy is bit-identical to _gpol_ext_rhs2's geodesic part — and
# _gpol_coeffs, so the sampled segments are identical to the polarized kernel's), but
# instead of accumulating Stokes it tallies per ray the emitting segments, how many are
# 'stiff' (max(|λ|)·ds > thresh), and the max/sum metric. Kept entirely separate from
# _gpol_kernel! so the production kernel stays bit-identical. Output per ray:
# SVector(n_valid, n_stiff, max_metric, sum_metric).
@kernel function _gpol_stiff_kernel!(out, @Const(y0s), @Const(grid), gp, a, r_h, C, escape,
                                     R_em, nmax, ng, xo, inv_dx, thresh, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        y = y0s[i]
        n_valid = 0.0; n_stiff = 0.0; max_m = 0.0; sum_m = 0.0
        r_prev = Inf
        for _ in 1:nmax
            r = _gimg_rks(y, a)
            _gimg_stop(r, r_prev, r_h, escape, R_em) && break
            r_prev = r
            h = (r - r_h) / C
            x = SVector(y[1], y[2], y[3], y[4])
            prim = _gimg_sample(grid, ng, xo, inv_dx, y[2], y[3], y[4])
            if prim[1] > 0.0
                gcov, gcon = _gimg_metric(x, a)
                kcov = -SVector(y[5], y[6], y[7], y[8])     # photon k = -p
                j, al, ro, tet_fl, valid = _gpol_coeffs(prim, x, kcov, gcov, gcon, gp)
                if valid
                    ds = h * gp.L_unit / gp.nu_obs
                    m = _stiffness_metric(al, ro, ds)
                    n_valid += 1.0
                    sum_m += m
                    max_m = max(max_m, m)
                    (m > thresh) && (n_stiff += 1.0)
                end
            end
            # geodesic-only RK4: no polarization state is marched here, and _gimg_rhs's
            # dy is bit-identical to _gpol_ext_rhs2's (same closed-form gcon, same AD) at
            # roughly half the cost — the transports of two zero vectors were dead work.
            y = _gimg_rk4_step(y, h, a)
        end
        out[i] = SVector(n_valid, n_stiff, max_m, sum_m)
    end
end

function _gpol_stiff_part!(out, rays, rng, grid_d, ng, xo, inv_dx, gp, a, r_h, C,
                           escape, R_em, nmax, thresh)
    backend = CUDABackend()
    kern = _gpol_stiff_kernel!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{SVector{4,Float64}}(undef, cnt)
    kern(outd, y0d, grid_d, gp, a, r_h, C, escape, R_em, Int(nmax), ng, xo, inv_dx,
         Float64(thresh), cnt; ndrange = cnt, workgroupsize = min(_GPOL_WG, cnt))
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

"""
    stiffness_telemetry(cfg, snap_or_grid; devices, nmax, C, box_half, n,
                        thresh=_STIFF_METRIC_THRESH) -> NamedTuple

DIAGNOSTIC pass over the SAME rays/segments the fused polarized kernel marches, reporting
the transfer-matrix stiffness distribution (RAPTOR check_stiffness telemetry). Returns
`(n_valid, n_stiff, frac_stiff, max_metric, mean_metric, per_ray_valid, per_ray_stiff)`:
`frac_stiff` is the fraction of emitting segments whose dominant eigenvalue scale
max(|λ|)·ds exceeds `thresh` (default the RAPTOR |e^{τλ}|>0.99 gate). Telemetry only — the
polarized transport is unchanged (our analytic coupling is exact in stiff cells).
"""
function stiffness_telemetry(cfg, snap_or_grid; devices = select_devices(),
                             nmax::Int = cfg.max_steps,
                             C::Real = 80, box_half::Real = 40.0, n::Integer = 256,
                             thresh::Real = _STIFF_METRIC_THRESH)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    dgrid = snap_or_grid isa DeviceFluidGrid ? snap_or_grid :
            upload_grid(snap_or_grid isa UniformFluidGrid ? snap_or_grid :
                        resample_fluid_uniform(snap_or_grid, box_half, n), devices)
    gp = _gpol_params(cfg)
    a = cfg.bh_a
    r_h = 1 + sqrt(1 - a * a)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    R_em = sqrt(3.0) * abs(dgrid.xo)
    nrays = length(rays)
    out = Vector{SVector{4,Float64}}(undef, nrays)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _gpol_stiff_part!(out, rays, idx, dgrid.dev_data[dev], dgrid.n, dgrid.xo,
                              dgrid.inv_dx, gp, a, r_h, Cf, escape, R_em, nmax, thresh)
        end
    end
    nv = sum(o[1] for o in out)
    ns = sum(o[2] for o in out)
    mx = maximum(o[3] for o in out)
    sm = sum(o[4] for o in out)
    return (n_valid = nv, n_stiff = ns, frac_stiff = nv > 0 ? ns / nv : 0.0,
            max_metric = mx, mean_metric = nv > 0 ? sm / nv : 0.0,
            per_ray_valid = [o[1] for o in out], per_ray_stiff = [o[2] for o in out])
end
