# Radial pieces of the exact-Kerr inversion beyond cases II and III of the equatorial
# tracer: case IV (two complex-conjugate pairs of roots, met only by vortical rays, η < 0,
# since R(0) = −a²η < 0 forces two real roots when η > 0), the observer at infinity with
# the regularized time of Krang (lim I_t(ro) − ro − 2 ln ro), and the radial state of one
# ray shared by `pixel_prep` and `surface_prep`. Formulas: Gralla & Lupsasca PRD 101
# 044032 (P1) App. B; the regularizations by the connection formula DLMF 19.7.9.

# F(φ|k) − Π(k/n; φ|k): the part of Π(n; φ|k) that stays finite where n sin²φ → 1.
@inline _pi_regular(n, φ, k) = JacobiElliptic.F(φ, k) - JacobiElliptic.Pi(k / n, φ, k)

# ---------------------------------------------------------------------- Jacobi functions
"""
    jacobi_amplitude(u, m) -> φ

The Jacobi amplitude `φ = am(u|m)` for `0 ≤ m ≤ 1`: A&S 16.13.4 for `m < √ε`; A&S 16.15.4,
`gd u + m₁ (sinh u cosh u − u) sech u / 4` with `m₁ = 1 − m`, only where `m₁ cosh²u < √ε`
or `m₁ = 0`; the arithmetic–geometric mean (A&S 16.4) otherwise. The next term of the
m → 1 expansion is of order `(m₁ cosh²u)²`, so it fails as `u` approaches
`K(m) ≈ ln(4/√m₁)`, where `m₁ cosh²u ≈ 4`, whatever the size of `m₁`. Generic in the
number type (ForwardDiff), with an immutable fixed-size buffer and no heap allocation
(GPU).
"""
@inline function jacobi_amplitude(u::A, m::B) where {A,B}
    T = promote_type(A, B)
    u == 0 && return zero(T)
    tol = eps(T)
    sqrt_tol = sqrt(tol)
    if m < sqrt_tol
        return u - m * (u - sin(2 * u) / 2) / 4                  # A&S 16.13.4
    end
    m1 = 1 - m
    if m1 < sqrt_tol && (iszero(m1) || m1 * cosh(u)^2 < sqrt_tol)
        t = tanh(u)
        return asin(t) + m1 * (t - u * (1 - t^2)) * cosh(u) / 4   # A&S 16.15.4
    end
    buf = ntuple(_ -> zero(T), Val(16))                           # c_n / a_n
    a, b, c, n = 1, sqrt(m1), sqrt(m), 0
    while abs(c) > tol && n < 16
        a, b, c, n = ((a + b) / 2, sqrt(a * b), (a - b) / 2, n + 1)
        buf = Base.setindex(buf, c / a, n)
    end
    phi = a * u * (2^n)
    for i in n:-1:1
        @inbounds phi = (phi + asin(buf[i] * sin(phi))) / 2
    end
    return phi
end

"""
    jacobi_sncndn(u, m) -> (sn, cn, dn)

The Jacobi elliptic functions `sn(u|m)`, `cn(u|m)`, `dn(u|m)` from
[`jacobi_amplitude`](@ref): `sn = sin φ`, `cn = cos φ` and `dn = √(m₁ + m cos²φ)`, the form
of `√(1 − m sin²φ)` free of cancellation as `m → 1`, for `0 ≤ m ≤ 1`; the
negative-parameter transformation (A&S 16.10) for `m < 0` and the reciprocal-parameter
transformation (A&S 16.11) for `m > 1`. Every sn, cn and sc of the closed-form radial
inversion goes through this function.
"""
@inline function jacobi_sncndn(u, m)
    if m < 0
        mu1 = inv(1 - m)
        mu = -m * mu1
        smu1 = sqrt(mu1)
        φ = jacobi_amplitude(u / smu1, mu)
        s = sin(φ)
        d = sqrt(1 - mu * s^2)
        return smu1 * s / d, cos(φ) / d, inv(d)
    elseif m > 1
        mu = inv(m)
        φ = jacobi_amplitude(u * sqrt(m), mu)
        s = sin(φ)
        return sqrt(mu) * s, sqrt(max(1 - mu * s^2, zero(s))), cos(φ)
    end
    φ = jacobi_amplitude(u, m)
    c = cos(φ)
    return sin(φ), c, sqrt((1 - m) + m * c^2)
end

# ---------------------------------------------------------------------- case IV
# Root data of case IV (P1 Eqs. B83–B87): r₁,₂ = b₂ ∓ i a₂, r₃,₄ = b₁ ∓ i a₁.
@inline function _case4_data(roots)
    _, r2, _, r4 = roots
    a1 = abs(imag(r4)); a2 = abs(imag(r2)); b1 = real(r4); b2 = real(r2)
    C = sqrt((a1 - a2)^2 + (b1 - b2)^2)
    D = sqrt((a1 + a2)^2 + (b1 - b2)^2)
    k4 = 4 * C * D / (C + D)^2
    g0 = sqrt(max((4 * a2^2 - (C - D)^2) / ((C + D)^2 - 4 * a2^2), zero(C)))
    return (a2 = a2, b1 = b1, C = C, D = D, k4 = k4, g0 = g0)
end

# f₂ and S₁, S₂ of P1 Eqs. B90–B92 (the principal value of Π(1 + α²; φ|j) past its pole
# cancels against the logarithm of f₂).
@inline function _f2_case4(α, φ, j)
    p2 = sqrt((1 + α^2) / (1 - j + α^2))
    t = sqrt(1 - j * sin(φ)^2)
    return p2 / 2 * log(abs((1 - p2) / (1 + p2) * (1 + p2 * t) / (1 - p2 * t)))
end
@inline _S1_case4(α, φ, j) =
    (JacobiElliptic.F(φ, j) + α^2 * JacobiElliptic.Pi(1 + α^2, φ, j) - α * _f2_case4(α, φ, j)) /
    (1 + α^2)
@inline function _S2_case4(α, φ, j)
    t = sqrt(1 - j * sin(φ)^2)
    return -((1 - j) * JacobiElliptic.F(φ, j) + α^2 * JacobiElliptic.E(φ, j) +
             α^2 * t * (α - tan(φ)) / (1 + α * tan(φ)) - α^3) / ((1 + α^2) * (1 - j + α^2)) +
           (1 / (1 + α^2) + (1 - j) / (1 - j + α^2)) * _S1_case4(α, φ, j)
end

# Antiderivatives (I0, I1, I2, Ip, Im) of case IV at radius r (P1 Eqs. B93–B94).
@inline function _radial_case4_antideriv(r::T, roots, a) where {T}
    d = _case4_data(roots)
    rph = kerr_horizon(a); rmh = 2 - rph
    φ = atan((r + d.b1) / d.a2) + atan(d.g0)
    c = 2 / (d.C + d.D)
    w = d.a2 / d.g0 * (1 + d.g0^2)
    F4 = c * JacobiElliptic.F(φ, d.k4)
    P1 = c * w * _S1_case4(d.g0, φ, d.k4)
    P2 = c * w^2 * _S2_case4(d.g0, φ, d.k4)
    h = d.a2 / d.g0 - d.b1
    I0 = F4
    I1 = h * F4 - P1
    I2 = h^2 * F4 - 2 * h * P1 + P2
    Ip = _case4_ipm(F4, rph, φ, d, c)
    Im = _case4_ipm(F4, rmh, φ, d, c)
    return I0, I1, I2, Ip, Im
end

# I± of case IV at the amplitude φ (P1 Eq. B94).
@inline function _case4_ipm(F4, rpm, φ, d, c)
    x = (rpm + d.b1) / d.a2
    gpm = (d.g0 * x - 1) / (d.g0 + x)
    return d.g0 / (d.a2 * (1 - d.g0 * x)) *
           (F4 - c * (1 + d.g0^2) / (d.g0 * (d.g0 + x)) * _S1_case4(gpm, φ, d.k4))
end

# Case IV at infinity, φ = π/2 + arctan g₀, where Π(1 + g₀²; φ|k₄) and the tan φ term of
# S₂ diverge: the regularized I1 − ln r and I2 − r.
@inline function _radial_inf_case4(roots, a, ::Type{T}) where {T}
    d = _case4_data(roots)
    rph = kerr_horizon(a); rmh = 2 - rph
    g0 = d.g0; k4 = d.k4; a2 = d.a2; b1 = d.b1
    φ = T(π) / 2 + atan(g0)
    c = 2 / (d.C + d.D)
    w = a2 / g0 * (1 + g0^2)
    Fφ = JacobiElliptic.F(φ, k4)
    S1r = (Fφ + g0^2 * _pi_regular(1 + g0^2, φ, k4)) / (1 + g0^2)
    S2r = -((1 - k4) * Fφ + g0^2 * JacobiElliptic.E(φ, k4)) / ((1 + g0^2) * (1 - k4 + g0^2)) +
          (1 / (1 + g0^2) + (1 - k4) / (1 - k4 + g0^2)) * S1r
    F4 = c * Fφ
    P1 = c * w * S1r
    P2 = c * w^2 * S2r
    h = a2 / g0 - b1
    q = 1 + g0^2
    lnum = 16 * (q - sqrt(q * (q - k4))) * (q - k4)
    lden = (d.C + d.D)^2 * (q^2 - k4) * k4 * (1 + sqrt(1 - k4 / q))
    I0 = F4
    I1 = h * F4 - P1 + log(lnum / lden) / 2
    I2 = h^2 * F4 - 2 * h * P1 + P2 -
         (16 * a2^4 + (d.C^2 - d.D^2)^2 - 8 * a2^2 * (d.C^2 + d.D^2) +
          8 * a2^3 * (d.C + d.D - 2 * b1 * g0) +
          2 * a2 * (d.C + d.D) * (-(d.C - d.D)^2 + 2 * b1 * (d.C + d.D) * g0)) /
         (4 * a2 * (4 * a2^2 - (d.C + d.D)^2) * g0)
    Ip = _case4_ipm(F4, rph, φ, d, c)
    Im = _case4_ipm(F4, rmh, φ, d, c)
    return I0, I1, I2, Ip, Im
end

# ---------------------------------------------------------------------- infinity, II and III
# Case II at infinity, x₂ = √(r₃₁/r₄₁), where Π(r₄₁/r₃₁; ·|k) and √R/(r − r₃) diverge.
@inline function _radial_inf_case2(r1, r2, r3, r4, a, ::Type{T}) where {T}
    rph = kerr_horizon(a); rmh = 2 - rph
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r43 = r4 - r3
    rp3 = rph - r3; rm3 = rmh - r3; rp4 = rph - r4; rm4 = rmh - r4
    k = real(r32 * r41 / (r31 * r42))
    arg = asin(clamp(real(sqrt(r31 / r41)), -one(T), one(T)))
    sr = real(sqrt(r31 * r42))
    F2 = 2 / sr * JacobiElliptic.F(arg, k)
    E2 = sr * JacobiElliptic.E(arg, k)
    PI21 = 2 / sr * _pi_regular(real(r41 / r31), arg, k)
    PI2p = 2 / sr * real(r43 / (rp3 * rp4)) * JacobiElliptic.Pi(real(rp3 * r41 / (rp4 * r31)), arg, k)
    PI2m = 2 / sr * real(r43 / (rm3 * rm4)) * JacobiElliptic.Pi(real(rm3 * r41 / (rm4 * r31)), arg, k)
    r3r = real(r3)
    I0 = F2
    I1 = r3r * F2 + real(r43) * PI21 + log(16 / real(r31 + r42)^2) / 2
    I2 = r3r - real(r1 * r4 + r2 * r3) / 2 * F2 - E2
    Ip = -PI2p - F2 / real(rp3)
    Im = -PI2m - F2 / real(rm3)
    return I0, I1, I2, Ip, Im
end

# Case III at infinity, cos φ = (A − B)/(A + B), where 1 + α₀ cos φ = 0: R₁ diverges
# logarithmically and the last term of R₂ linearly; the R₁ terms of I2 cancel identically.
@inline function _radial_inf_case3(r1, r2, r3, r4, a, ::Type{T}) where {T}
    rph = kerr_horizon(a); rmh = 2 - rph
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    rp2 = rph - r2; rm2 = rmh - r2; rp1 = rph - r1; rm1 = rmh - r1
    A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
    k = ((A + B)^2 - real(r21)^2) / (4 * A * B)
    alpha0 = (B + A) / (B - A)
    arg = acos(clamp((A - B) / (A + B), -one(T), one(T)))
    alphap = real((B * rp2 + A * rp1) / (B * rp2 - A * rp1))
    alscm = real((B * rm2 + A * rm1) / (B * rm2 - A * rm1))
    F = JacobiElliptic.F(arg, k); E = JacobiElliptic.E(arg, k)
    r1r = real(r1); r2r = real(r2); r21r = real(r21)
    BrAr = (B * r2r + A * r1r) / (B + A)
    F3 = F / sqrt(A * B)
    c1 = 2 * r21r * sqrt(A * B) / (B * B - A * A)
    n0 = alpha0^2 / (alpha0^2 - 1)
    R1r = _pi_regular(n0, arg, k) / (1 - alpha0^2)
    R2r = (F - alpha0^2 / (k + (1 - k) * alpha0^2) * E) / (alpha0^2 - 1)
    I0 = F3
    I1 = BrAr * F3 + c1 * R1r + log(16 * r21r^2 / ((A * A - B * B)^2 + 4 * A * B * r21r^2)) / 2
    I2 = BrAr^2 * F3 + sqrt(A * B) * c1^2 * R2r + BrAr
    Ip = -1 / real(B * rp2 + A * rp1) *
         ((B + A) * F3 + 2 * r21r * sqrt(A * B) / real(B * rp2 - A * rp1) * _R1_case3(alphap, arg, k))
    Im = -1 / real(B * rm2 + A * rm1) *
         ((B + A) * F3 + 2 * r21r * sqrt(A * B) / real(B * rm2 - A * rm1) * _R1_case3(alscm, arg, k))
    return I0, I1, I2, Ip, Im
end

# ---------------------------------------------------------------------- one ray
# The radial case of the roots: 2 (r₄ real), 3 (r₁, r₂ real), 4 (no real root).
@inline function _radial_case(roots, ::Type{T}) where {T}
    r1, r2, r3, r4 = roots
    _isreal_c(r4, T(1e-10) * max(abs(r4), one(T))) && return 2
    _isreal_c(r2, T(1e-10) * max(abs(r2), one(T))) && return 3
    return 4
end

# The observer-side antiderivatives of case `rcase` at `ro` (the regularized limits at
# `ro = Inf`): (po, pt), pt the turning-point leg of case II and zeros otherwise.
@inline function _radial_obs_parts(ro, roots, a, λ, η, rcase::Int)
    r1, r2, r3, r4 = roots
    T = typeof(real(r1) + a + λ + η)
    z = zero(T)
    if rcase == 2
        po = isinf(ro) ? _radial_inf_case2(r1, r2, r3, r4, a, T) :
             _radial_case2_antideriv(ro, r1, r2, r3, r4, a, λ, η)
        pt = _radial_case2_antideriv(real(r4), r1, r2, r3, r4, a, λ, η)
        return po, pt
    elseif rcase == 3
        po = isinf(ro) ? _radial_inf_case3(r1, r2, r3, r4, a, T) :
             _radial_case3_antideriv(ro, r1, r2, r3, r4, a)
        return po, (z, z, z, z, z)
    else
        po = isinf(ro) ? _radial_inf_case4(roots, a, T) : _radial_case4_antideriv(T(ro), roots, a)
        return po, (z, z, z, z, z)
    end
end

# The source-side antiderivatives at rs combined with the observer parts into the definite
# integrals, then (I_r, I_φ, I_t) (P1 Eqs. B1–B3).
@inline function _radial_from_parts(rs::T, roots, a, λ, η, rcase::Int, νr, obs) where {T}
    rcase == 4 || return _radial_from_parts(rs, roots, a, λ, η, rcase == 2, νr, obs)
    po, _ = obs
    I0s, I1s, I2s, Ips, Ims = _radial_case4_antideriv(rs, roots, a)
    I0o, I1o, I2o, Ipo, Imo = po
    return _radial_combine(I0o - I0s, I1o - I1s, I2o - I2s, Ipo - Ips, Imo - Ims, a, λ)
end

# The radial state of one ray: case, modulus `kr`, observer amplitude `Fr`, the Mino time
# to the case-II turning point `τturn` and the ray's budget `τtot`, the observer parts.
# `four = false` classifies into cases II and III only (η ≥ 0 has two real roots).
@inline function _radial_state(ro, roots, a, λ, η, rph, ::Type{T}, four::Bool) where {T}
    r1, r2, r3, r4 = roots
    rcase = four ? _radial_case(roots, T) : (_isreal_c(r4, T(1e-10) * max(abs(r4), one(T))) ? 2 : 3)
    po, pt = _radial_obs_parts(ro, roots, a, λ, η, rcase)
    if rcase == 2
        r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        kr = real(r32 * r41 / (r31 * r42))
        x2 = isinf(ro) ? real(sqrt(r31 / r41)) : real(sqrt((ro - r4) * r31 / ((ro - r3) * r41)))
        Fr = JacobiElliptic.F(asin(clamp(x2, -one(T), one(T))), kr)
        τturn = po[1]                                  # observer → turn
        τtot = po[1] + _tau_max_case2(r1, r2, r3, r4, T)   # observer → turn → infinity
        if real(r4) <= rph
            # four real roots, all inside the horizon (P2 region II): the ray reaches the
            # horizon before r4, so its budget is observer → horizon
            x2h = real(sqrt((rph - r4) * r31 / ((rph - r3) * r41)))
            τtot = τturn - 2 / real(sqrt(r31 * r42)) *
                   JacobiElliptic.F(asin(clamp(x2h, -one(T), one(T))), kr)
        end
    elseif rcase == 3
        r21 = r2 - r1; r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        kr = ((A + B)^2 - real(r21)^2) / (4 * A * B)
        x3o = _x3_obs(ro, A, B, r1, r2)
        Fr = JacobiElliptic.F(acos(clamp(x3o, -one(T), one(T))), kr)
        τturn = zero(kr)                               # no case-III turn
        τtot = _mino_to_horizon_case3(ro, r1, r2, r3, r4, rph)  # observer → horizon
    else
        d = _case4_data(roots)
        kr = d.k4
        Fr = zero(kr)
        τturn = zero(kr)
        τtot = po[1] - first(_radial_case4_antideriv(rph, roots, a))   # observer → horizon
    end
    return (rcase = rcase, kr = kr, Fr = Fr, τturn = τturn, τtot = τtot, po = po, pt = pt)
end

# x₃ at the observer: (A(ro − r₁) − B(ro − r₂))/(A(ro − r₁) + B(ro − r₂)), (A − B)/(A + B) at ∞.
@inline _x3_obs(ro, A, B, r1, r2) =
    isinf(ro) ? (A - B) / (A + B) :
    real((A * (ro - r1) - B * (ro - r2)) / (A * (ro - r1) + B * (ro - r2)))

# Source radius at Mino time τ from the observer (P1 Eqs. B44, B71, B105): the sn
# inversion of case II, the cn inversion of case III, the sc inversion of case IV.
@inline function _radius_at(prep, τ::T) where {T}
    r1, r2, r3, r4 = prep.roots
    if prep.rcase == 2
        r31 = r3 - r1; r41 = r4 - r1; r42 = r4 - r2
        sn2 = jacobi_sncndn(real(sqrt(r31 * r42)) * τ / 2 - prep.Fr, prep.kr)[1]^2
        return real((r4 * r31 - r3 * r41 * sn2) / (r31 - r41 * sn2)), τ > prep.τturn ? -1 : 1
    elseif prep.rcase == 3
        r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        cn3 = jacobi_sncndn(sqrt(A * B) * τ - prep.Fr, prep.kr)[2]
        r1r = real(r1); r2r = real(r2)
        return ((A * r1r - B * r2r) - (A * r1r + B * r2r) * cn3) / ((A - B) - (A + B) * cn3), 1
    else
        d = _case4_data(prep.roots)
        sn4, cn4, _ = jacobi_sncndn((d.C + d.D) / 2 * (prep.po[1] - τ), d.k4)
        sc = sn4 / cn4
        return -d.a2 * (d.g0 - sc) / (1 + d.g0 * sc) - d.b1, 1
    end
end
