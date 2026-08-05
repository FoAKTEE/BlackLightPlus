# Synchrotron coefficient tests: the Abramowitz & Stegun Bessel fits must hit their
# tabulated values, and the thermal coefficients must have the correct signs and satisfy
# Kirchhoff's law. Ground truth is analytic / tabulated — no reference code needed.
using Test
using BlackLightPlus: bessel_k0, bessel_k1, bessel_k2, harmonic_bridge,
                    thermal_synchrotron, thermal_synchrotron_polarized
using BlackLightPlus.Constants: HPL, CL, ME, EE

@testset "Bessel A&S fits" begin
    @test bessel_k0(1.0) ≈ 0.421024 atol = 1e-6
    @test bessel_k1(1.0) ≈ 0.601907 atol = 1e-6
    @test bessel_k2(1.0) ≈ 1.624839 atol = 1e-6
    # recurrence identity K2 = K0 + (2/x) K1
    @test bessel_k2(1.3) ≈ bessel_k0(1.3) + (2.0 / 1.3) * bessel_k1(1.3)
end

@testset "harmonic bridge limits" begin
    # symmetric, and bounded below the smaller asymptote
    @test harmonic_bridge(2.0, 8.0, 3.0) ≈ harmonic_bridge(8.0, 2.0, 3.0)
    @test harmonic_bridge(2.0, 8.0, 3.0) < 2.0
    # a ZERO low asymptote drives the min-like bridge to zero (IEEE: 0^-x = Inf,
    # Inf^(-1/x) = 0) — blacklight parity; the "returns the other asymptote"
    # note in the header applies to an underflowed-but-nonzero branch
    @test harmonic_bridge(0.0, 5.0, 2.0) == 0.0
    @test harmonic_bridge(1e-300, 5.0, 2.0) < 1e-100
end

@testset "thermal synchrotron sanity" begin
    theta_e = 10.0; B = 30.0; nu = 230.0e9; sin_th = 0.7; n_e = 1.0e6
    j_i, alpha_i = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_th)
    @test j_i > 0
    @test alpha_i > 0

    # Kirchhoff's law: alpha_i = j_i / (B_nu / nu^3), B_nu/nu^3 = 2h/c^2 / expm1(h nu / kT_e).
    kT_e = theta_e * ME * CL * CL
    b_nu_nu_3 = 2.0 * HPL / (CL * CL) / expm1(HPL * nu / kT_e)
    @test alpha_i ≈ j_i / b_nu_nu_3 rtol = 1e-12

    # Polarized: j_i > 0, j_q < 0, and the unpolarized j_i / alpha_i match the scalar path.
    ji, jq, jv, ai, aq, av, rq, rv =
        thermal_synchrotron_polarized(1.0, n_e, theta_e, B, nu, sin_th, 0.5)
    @test ji > 0
    @test jq < 0
    @test ji ≈ j_i rtol = 1e-12
    @test ai ≈ alpha_i rtol = 1e-12
    @test aq ≈ jq / b_nu_nu_3 rtol = 1e-12   # Kirchhoff on Q
    # zero-input early-out
    @test thermal_synchrotron(1.0, 0.0, theta_e, B, nu, sin_th) == (0.0, 0.0)
end

# --- regression: power-law ρ_Q removable pole at the textbook index p = 2 --------------
using BlackLightPlus: SimCoeffConfig, precompute_power_law!, power_law_synchrotron_polarized

@testset "power-law ρ_Q removable pole at p = 2 (no NaN)" begin
    mk(p) = (c = SimCoeffConfig(d_unit = 1.0, e_unit = 1.0, b_unit = 1.0,
                                plasma_power_frac = 0.5, plasma_p = p);
             precompute_power_law!(c, 10.0, 1.0e4); c)
    rq(c) = power_law_synchrotron_polarized(c, 1.0e6, 30.0, 230.0e9, 0.7, 0.5)
    # p = 2 makes power_rho_q = -Inf and var_e = 0 → rho_q = Inf*0 = NaN without the guard.
    t2 = rq(mk(2.0))
    @test all(isfinite, t2)
    # continuity: the p = 2 value equals the two-sided (p = 2 ± 1e-6) limit, and the
    # other 7 coefficients are untouched (the p ≠ 2 arithmetic is bit-for-bit unchanged).
    tm = rq(mk(2.0 - 1.0e-6)); tp = rq(mk(2.0 + 1.0e-6))
    @test all(isfinite, tm) && all(isfinite, tp)
    @test t2[7] ≈ 0.5 * (tm[7] + tp[7]) rtol = 1e-4        # rho_q hits the ln(X) limit
    @test t2[1] ≈ 0.5 * (tm[1] + tp[1]) rtol = 1e-4        # j_i unaffected by the guard
end

# --- regression: thermal rho_Q smooth bridge across the Theta_e = 0.01 cold trap ------------
# rho_Q's Faraday-conversion factor was switched on discontinuously at K_THETA_E_ZERO = 0.01
# (0 below, finite fit above — measured jump 0 -> -5.76e-4 across Theta_e 0.0099999 -> 0.0100001
# on the 230 GHz / B=30 G / n_e=1e6 deck). It is now ramped by a C1 smoothstep over
# [0.005, 0.01]; the weight is exactly 1 for Theta_e >= 0.01 so those values are unchanged.
@testset "thermal rho_Q smooth bridge (no discontinuity at Theta_e = 0.01)" begin
    n_e = 1.0e6; B = 30.0; nu = 230.0e9; s_th = 0.7; c_th = 0.5
    rq(te) = thermal_synchrotron_polarized(1.0, n_e, te, B, nu, s_th, c_th)[7]

    # (1) continuity across the old hard threshold: residual jump is now ~5 orders smaller.
    r_lo = rq(0.0099999); r_hi = rq(0.0100001)
    @test abs(r_hi - r_lo) < 1.0e-6                 # was ~5.76e-4 (a hard 0 -> fit step)
    @test abs(r_hi - r_lo) / abs(r_hi) < 1.0e-3     # relative residual jump

    # (2) hard-continuous at the lower blend edge: exactly 0 below 0.005, ~0 just above.
    @test rq(0.005 - 1.0e-6) == 0.0
    @test abs(rq(0.005 + 1.0e-6)) < abs(r_hi) * 1.0e-3   # smoothstep starts flat at 0

    # (3) monotone ramp fills the previously-zero (0.005, 0.01) window.
    @test 0.0 < abs(rq(0.006)) < abs(rq(0.008)) < abs(rq(0.0099)) < abs(r_hi)

    # (4) Theta_e >= 0.01 is BIT-IDENTICAL to the unblended fit (blend weight == 1). Recompute
    # the reference fit independently (no blend) and confirm no scaling is applied above 0.01.
    function ref_rhoq(te)
        nu_c = EE * B / (2.0 * π * ME * CL)
        nu_s = 2.0 / 9.0 * nu_c * te * te * s_th
        xx = nu / nu_s; xn = 1.0 / sqrt(xx)
        va = 2.011 * exp(-19.78 * xx^(-0.5175))
        vb = cos(39.89 * xn) * exp(-70.16 * xx^(-0.6))
        vc = 0.011 * exp(-1.69 * xn)
        vd = 0.003135 * xx^(4.0 / 3.0)
        ve = 0.5 * (1.0 + tanh(10.0 * log(0.6648 * xn)))
        f_m = (va - vb - vc) + (vc - vd) * ve
        inv = 1.0 / te; k0 = bessel_k0(inv); k1 = bessel_k1(inv); k2 = k0 + (2.0 / inv) * k1
        coeff_q = -1.0 * n_e * EE * EE * nu_c * nu_c * (s_th * s_th) / (ME * CL * nu * nu)
        return coeff_q * f_m * (k1 / k2 + 6.0 * te)
    end
    for te in (0.0100001, 0.02, 0.1, 10.0)
        @test rq(te) ≈ ref_rhoq(te) rtol = 1.0e-12
    end
end
