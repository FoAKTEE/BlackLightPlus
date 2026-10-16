# Special functions of the synchrotron fits: the Abramowitz & Stegun 9.8 polynomial fits
# of the modified Bessel functions (device-safe), the harmonic bridge of the nonthermal
# asymptotes, and the host-side Gauss hypergeometric function of the kappa precompute.

"""
    bessel_i0(x)

Modified Bessel function I₀(x), polynomial fits A&S 9.8.1 (|x| < 3.75) and
9.8.2 (|x| ≥ 3.75).
"""
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

"""
    bessel_i1(x)

Modified Bessel function I₁(x), polynomial fits A&S 9.8.3 (|x| < 3.75) and
9.8.4 (|x| ≥ 3.75); odd in `x`.
"""
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

"""
    bessel_k0(x)

Modified Bessel function K₀(x) for `x > 0`, polynomial fits A&S 9.8.5
(x ≤ 2) and 9.8.6 (x > 2).
"""
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

"""
    bessel_k1(x)

Modified Bessel function K₁(x) for `x > 0`, polynomial fits A&S 9.8.7
(x ≤ 2) and 9.8.8 (x > 2).
"""
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

"""
    bessel_k2(x)

Modified Bessel function K₂(x) for `x > 0` by the recurrence
K₂ = K₀ + (2/x) K₁ on the A&S 9.8 fits.
"""
@inline bessel_k2(x) = bessel_k0(x) + (2.0 / x) * bessel_k1(x)

"""
    harmonic_bridge(a, b, x) = (a^-x + b^-x)^(-1/x)

Harmonic-mean bridge of a low-frequency asymptote `a` and a high-frequency
asymptote `b` with exponent `x > 0` (blacklight `HarmonicBridge`); the smaller
asymptote dominates. A zero or underflowed asymptote makes its power `Inf`, and
the bridge returns 0, the limit of a vanishing asymptote.
"""
@inline harmonic_bridge(a, b, x) = (a^(-x) + b^(-x))^(-1.0 / x)

"""
    hypergeometric_2f1(alpha, beta, gam, z) -> Float64

Gauss hypergeometric function ₂F₁(α, β; γ; z) for `z < 0` (blacklight
`Hypergeometric`): the Pfaff transform
`(1 - z)^(-α) ₂F₁(α, γ - β; γ; z/(z - 1))` with the series truncated after ten
rising-factorial terms. Host only.
"""
function hypergeometric_2f1(alpha::Float64, beta::Float64, gam::Float64, z::Float64)
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
