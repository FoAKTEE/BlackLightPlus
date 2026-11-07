# Exact-Kerr equatorial-emission ray tracing in Boyer–Lindquist coordinates (M = 1):
# per pixel, the conserved (λ, η), the closed-form quartic radial roots, the elliptic
# integrals and Jacobi sn/cn invert directly to the equatorial source point of
# photon-ring layer n, with no ODE. Isbits and ForwardDiff-transparent. Follows Krang.jl
# (MIT, © 2023 Dominic Chang et al.) and AART (MIT, © 2024 Cárdenas-Avendaño, Zhu &
# Lupsasca); see NOTICE. Physics: Gralla & Lupsasca PRD 101 044032 (P1), 044033 (P2);
# Cárdenas-Avendaño, Lupsasca & Zhu PRD 107 043030 (P3); "Pn Eq. m" cites these.

# Loose test that a complex number is effectively real (as in Krang).
@inline _isreal_c(z, tol) = abs(imag(z)) < tol

# Complex-safe cube root (as in AART): the real cbrt for an effectively real argument
# (decided on the primal imaginary part), else the principal branch.
@inline function _cuberoot(z::Complex{T}) where {T}
    if abs(imag(z)) < 1e-12 * max(abs(real(z)), one(real(T)))
        return Complex(cbrt(real(z)), zero(T))
    else
        return z^(one(T) / 3)
    end
end

"""
    conserved_quantities(α, β, θo, a) -> (λ, η)

Energy-reduced axial angular momentum `λ = -α sin θo` and Carter constant
`η = (α² - a²) cos² θo + β²` of the photon landing at Bardeen screen coordinate
`(α, β)` for an observer at inclination `θo` (rad), P2 Eqs. 58–59. The camera
image coordinates are Bardeen's: a camera ray at `(α, β)` has `L_z/E = -α sin θo`.
"""
@inline function conserved_quantities(α::T, β, θo, a) where {T}
    λ = -α * sin(θo)
    η = (α * α - a * a) * cos(θo)^2 + β * β
    return λ, η
end

"""
    radial_roots(λ, η, a) -> NTuple{4,Complex}

The four roots of the radial potential
`R(r) = r⁴ + (a² - η - λ²) r² + 2(η + (a - λ)²) r - a² η` (P2 Eqs. A1–A8),
closed form and complex-safe. Four real roots (case II) are ordered
`r1 ≤ r2 ≤ r3 ≤ r4`; with two real roots (case III) `r1, r2` are the real pair
and `r3, r4` a complex-conjugate pair.
"""
@inline function radial_roots(λ, η, a)
    T = promote_type(typeof(λ), typeof(η), typeof(a))
    z0 = zero(T)
    A = T(a * a - η - λ * λ)
    B = T(2 * (η + (λ - a)^2))
    C = T(-a * a * η)

    P = -A * A / 12 - C
    Q = -A / 3 * ((A / 6)^2 - C) - B * B / 8

    disc = Complex((P / 3)^3 + (Q / 2)^2, z0)
    sq = sqrt(disc)
    # the larger of -Q/2 ± sq, then w_m from w_p w_m = -P/3: no cancellation where P → 0
    sq = real(Complex(-Q / 2, z0) * conj(sq)) >= 0 ? sq : -sq
    w_p = _cuberoot(Complex(-Q / 2, z0) + sq)
    w_m = abs2(w_p) > 0 ? -Complex(P / 3, z0) / w_p : w_p    # value test: a Dual zero may carry partials

    z = sqrt((w_p + w_m) / 2 - Complex(A / 6, z0))               # P2 Eq. A6
    u = sqrt(Complex(-A / 2, z0) - z * z + Complex(B / 4, z0) / z)
    v = sqrt(Complex(-A / 2, z0) - z * z - Complex(B / 4, z0) / z)

    return (-z - u, -z + u, z - v, z + v)
end

"""
    angular_turning_points(λ, η, a) -> (u_p, u_m)

The roots `u_± = cos² θ_±` of the angular potential (P2 Eq. 11); real for
`η ≥ 0`, with `u_p ≥ 0 ≥ u_m` for equator-crossing rays, and for every vortical ray that
reaches an observer (the discriminant is clamped at zero against rounding). The root of the sign
of `Δθ = (1 - (η + λ²)/a²)/2` is formed by addition and the other from the exact
product `u_p u_m = -η/a²`, which avoids cancellation at small spin.
"""
@inline function angular_turning_points(λ::T, η, a) where {T}
    Δθ = (1 - (η + λ * λ) / (a * a)) / 2
    # a vortical ray with the observer at its single polar turning point has a double
    # root, Δθ² + η/a² = 0, which rounding may make negative
    disc = sqrt(max(Δθ * Δθ + η / (a * a), zero(Δθ)))
    if Δθ < 0
        u_m = Δθ - disc
        return -η / (a * a * u_m), u_m
    else
        u_p = Δθ + disc
        return u_p, -η / (a * a * u_p)
    end
end

"""
    analytic_angular_integrals(n, β, u_p, u_m, θo, a) -> (Gθ, Gφ, Gt)

Angular path integrals for equatorial emission (`θs = π/2`, so the source terms
vanish), P2 Eqs. 12–22. `n` is the photon-ring layer; the turning-point count is
`m = n + H(β)` with `H(0) = 1` (the disk's far side). `Gθ` is the Mino time the
radial inversion is evaluated at.
"""
@inline analytic_angular_integrals(n::Integer, β, u_p, u_m, θo, a) =
    _angular_from_parts(n, _angular_parts(β, u_p, u_m, θo, a))

# The layer-independent elliptic terms of the angular integrals, once per pixel.
@inline function _angular_parts(β::T, u_p, u_m, θo, a) where {T}
    k = u_p / u_m                                   # parameter m (< 0 here)
    K = JacobiElliptic.K(k)
    Ec = JacobiElliptic.E(k)
    E_prime = (Ec - K) / (2 * k)
    PI = JacobiElliptic.Pi(u_p, T(π) / 2, k)

    arg = asin(clamp(cos(θo) / sqrt(u_p), -one(T), one(T)))
    F_o = JacobiElliptic.F(arg, k)
    E_o = JacobiElliptic.E(arg, k)
    E_prime_o = (E_o - F_o) / (2 * k)
    PI_o = JacobiElliptic.Pi(u_p, arg, k)

    H_β = β >= 0 ? 1 : 0
    pm_o = β >= 0 ? one(T) : -one(T)   # H(0) = 1 (sign(0) = 0 would alias β = 0)
    den = abs(a) * sqrt(-u_m)          # √(-u_m a²)
    return (u_p=u_p, K=K, Ep=E_prime, PI=PI, F_o=F_o, Ep_o=E_prime_o, PI_o=PI_o,
            Hβ=H_β, pm_o=pm_o, den=den, k=k, u_m=u_m)
end

@inline function _angular_from_parts(n::Integer, ang)
    m = n + ang.Hβ
    Gθ = (2 * m * ang.K - ang.pm_o * ang.F_o) / ang.den
    Gφ = (2 * m * ang.PI - ang.pm_o * ang.PI_o) / ang.den
    Gt = -2 * ang.u_p * (2 * m * ang.Ep - ang.pm_o * ang.Ep_o) / ang.den
    return Gθ, Gφ, Gt
end

# Mino time to the outer radial turning point r4 from infinity (case II), P2 Eq. A10.
@inline function _tau_max_case2(r1, r2, r3, r4, ::Type{T}) where {T}
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
    k2 = real(r32 * r41 / (r31 * r42))
    return 2 / real(sqrt(r31 * r42)) *
           JacobiElliptic.F(asin(clamp(real(sqrt(r31 / r41)), -one(T), one(T))), k2)
end

# Mino time from radius ro inward to the outer horizon (case III, plunging), P2 Eq. A11
# with a finite-observer leg: the ray's whole Mino budget.
@inline function _mino_to_horizon_case3(ro::T, r1, r2, r3, r4, rph) where {T}
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
    k3 = ((A + B)^2 - real(r21)^2) / (4 * A * B)
    x3o = _x3_obs(ro, A, B, r1, r2)
    x3h = real((A * (rph - r1) - B * (rph - r2)) / (A * (rph - r1) + B * (rph - r2)))
    return (JacobiElliptic.F(acos(clamp(x3o, -one(T), one(T))), k3) -
            JacobiElliptic.F(acos(clamp(x3h, -one(T), one(T))), k3)) / sqrt(A * B)
end

# Radial antiderivatives (I0, I1, I2, Ip, Im) of case II (P3 App. B, AART
# radial_case2_antiderivative); elliptic arguments reduced to reals as in AART.
@inline function _radial_case2_antideriv(r::T, r1, r2, r3, r4, a, λ, η) where {T}
    rmh = 1 - sqrt(1 - a * a); rph = 1 + sqrt(1 - a * a)
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r43 = r4 - r3
    rp3 = rph - r3; rm3 = rmh - r3; rp4 = rph - r4; rm4 = rmh - r4
    k = real(r32 * r41 / (r31 * r42))
    q = (r - r4) * r31 / ((r - r3) * r41)
    # at the turning point r = r4 the argument is exactly zero (every leg vanishes there);
    # the value test keeps the complex square root away from a dual zero with partials
    x2 = abs2(q) > 0 ? real(sqrt(q)) : zero(T)
    arg = asin(clamp(x2, -one(T), one(T)))
    sr = real(sqrt(r31 * r42))
    F2 = 2 / sr * JacobiElliptic.F(arg, k)
    E2 = sr * JacobiElliptic.E(arg, k)
    PI21 = 2 / sr * JacobiElliptic.Pi(real(r41 / r31), arg, k)
    PI2p = 2 / sr * real(r43 / (rp3 * rp4)) * JacobiElliptic.Pi(real(rp3 * r41 / (rp4 * r31)), arg, k)
    PI2m = 2 / sr * real(r43 / (rm3 * rm4)) * JacobiElliptic.Pi(real(rm3 * r41 / (rm4 * r31)), arg, k)
    r3r = real(r3)
    I0 = F2
    I1 = r3r * F2 + real(r43) * PI21
    # √R/(r − r₃) from the factored potential, exactly zero at the turning point r = r₄:
    # the expanded polynomial leaves a rounding residue there that √ lifts to ~√ε, which
    # 1/(r₄ − r₃) amplifies near the critical curve
    Rf = real((r - r1) * (r - r2) * (r - r3) * (r - r4))
    sR = abs2(q) > 0 ? sqrt(max(Rf, zero(T))) : zero(T)
    I2 = sR / (r - r3r) - real(r1 * r4 + r2 * r3) / 2 * F2 - E2
    Ip = -PI2p - F2 / real(rp3)
    Im = -PI2m - F2 / real(rm3)
    return I0, I1, I2, Ip, Im
end

# R1, R2 auxiliaries of case III (P3 Eqs. B61–65); `arg` is the amplitude φ.
@inline function _R1_case3(alpha, arg, k)
    p1 = sqrt((alpha^2 - 1) / (k + (1 - k) * alpha^2))
    s = sin(arg); t = sqrt(1 - k * s * s)
    f1 = p1 / 2 * log(abs((p1 * t + s) / (p1 * t - s)))
    return 1 / (1 - alpha^2) * (JacobiElliptic.Pi(alpha^2 / (alpha^2 - 1), arg, k) - alpha * f1)
end
@inline function _R2_case3(alpha, arg, k)
    s = sin(arg); c = cos(arg); t = sqrt(1 - k * s * s)
    F = JacobiElliptic.F(arg, k); E = JacobiElliptic.E(arg, k)
    return 1 / (alpha^2 - 1) *
           (F - alpha^2 / (k + (1 - k) * alpha^2) * (E - alpha * s * t / (1 + alpha * c))) +
           1 / (k + (1 - k) * alpha^2) * (2 * k - alpha^2 / (alpha^2 - 1)) * _R1_case3(alpha, arg, k)
end

# Radial antiderivatives (I0, I1, I2, Ip, Im) of case III (AART
# radial_case3_antiderivative).
@inline function _radial_case3_antideriv(r::T, r1, r2, r3, r4, a) where {T}
    rmh = 1 - sqrt(1 - a * a); rph = 1 + sqrt(1 - a * a)
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    rp2 = rph - r2; rm2 = rmh - r2; rp1 = rph - r1; rm1 = rmh - r1
    A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
    k = ((A + B)^2 - real(r21)^2) / (4 * A * B)
    alpha0 = (B + A) / (B - A)
    x3 = real((A * (r - r1) - B * (r - r2)) / (A * (r - r1) + B * (r - r2)))
    arg = acos(clamp(x3, -one(T), one(T)))
    alphap = real((B * rp2 + A * rp1) / (B * rp2 - A * rp1))
    alscm = real((B * rm2 + A * rm1) / (B * rm2 - A * rm1))
    F = JacobiElliptic.F(arg, k)
    r1r = real(r1); r2r = real(r2); r21r = real(r21)
    BrAr = (B * r2r + A * r1r) / (B + A)
    F3 = F / sqrt(A * B)
    PI31 = (2 * r21r * sqrt(A * B) / (B * B - A * A)) * _R1_case3(alpha0, arg, k)
    PI32 = (2 * r21r * sqrt(A * B) / (B * B - A * A))^2 * _R2_case3(alpha0, arg, k)
    I0 = F3
    I1 = BrAr * F3 + PI31
    I2 = BrAr^2 * F3 + 2 * BrAr * PI31 + sqrt(A * B) * PI32
    Ip = -1 / real(B * rp2 + A * rp1) *
         ((B + A) * F3 + 2 * r21r * sqrt(A * B) / real(B * rp2 - A * rp1) * _R1_case3(alphap, arg, k))
    Im = -1 / real(B * rm2 + A * rm1) *
         ((B + A) * F3 + 2 * r21r * sqrt(A * B) / real(B * rm2 - A * rm1) * _R1_case3(alscm, arg, k))
    return I0, I1, I2, Ip, Im
end

"""
    analytic_radial_integrals(rs, ro, roots, a, λ, η, is_case2, νr) -> (I_r, I_φ, I_t)

Definite radial path integrals from the source radius `rs` to the observer
radius `ro` (P3 Eqs. B1–3). In case II the ray turns at `r4`, and the integral
is split with the weight `w = (νr == -1)`; case III is the plain
observer-minus-source difference.
"""
@inline analytic_radial_integrals(rs, ro, roots, a, λ, η, is_case2::Bool, νr) =
    _radial_from_parts(rs, roots, a, λ, η, is_case2, νr,
                       _radial_obs_parts(ro, roots, a, λ, η, is_case2))

# The layer-independent observer-side antiderivatives: the (ro, r4) legs in case II,
# the ro leg in case III (second slot unused); `ro = Inf` gives the regularized limits.
@inline _radial_obs_parts(ro, roots, a, λ, η, is_case2::Bool) =
    _radial_obs_parts(ro, roots, a, λ, η, is_case2 ? 2 : 3)

# Definite integrals from the observer parts `obs = (po, pt)` and the source antiderivative;
# with `τ`, the radial Mino time I0 is replaced by the angular one (the geodesic equation
# I_r = G_θ, exact) in every term proportional to it (`_radial_tau`).
@inline function _radial_from_parts(rs::T, roots, a, λ, η, is_case2::Bool, νr, obs;
                                    τ = nothing) where {T}
    po, pt = obs
    r1, r2, r3, r4 = roots
    if is_case2
        w = νr < 0 ? one(T) : zero(T)
        I0s, I1s, I2s, Ips, Ims = _radial_case2_antideriv(rs, r1, r2, r3, r4, a, λ, η)
        I0o, I1o, I2o, Ipo, Imo = po
        I0t, I1t, I2t, Ipt, Imt = pt
        I0 = I0o - I0s + 2 * w * (I0s - I0t)
        I1 = I1o - I1s + 2 * w * (I1s - I1t)
        I2 = I2o - I2s + 2 * w * (I2s - I2t)
        Ip = Ipo - Ips + 2 * w * (Ips - Ipt)
        Im = Imo - Ims + 2 * w * (Ims - Imt)
    else
        I0s, I1s, I2s, Ips, Ims = _radial_case3_antideriv(rs, r1, r2, r3, r4, a)
        I0o, I1o, I2o, Ipo, Imo = po
        I0 = I0o - I0s; I1 = I1o - I1s; I2 = I2o - I2s
        Ip = Ipo - Ips; Im = Imo - Ims
    end
    τ === nothing ||
        ((I0, I1, I2, Ip, Im) = _radial_tau((I0, I1, I2, Ip, Im), is_case2 ? 2 : 3, roots, a, τ))
    return _radial_combine(I0, I1, I2, Ip, Im, a, λ)
end

# (I_r, I_φ, I_t) from the definite integrals (P1 Eqs. B1–B3).
@inline function _radial_combine(I0, I1, I2, Ip, Im, a, λ)
    rph = 1 + sqrt(1 - a * a); rmh = 1 - sqrt(1 - a * a)
    I_φ = 2 * a / (rph - rmh) * ((rph - a * λ / 2) * Ip - (rmh - a * λ / 2) * Im)
    I_t = 4 / (rph - rmh) * (rph * (rph - a * λ / 2) * Ip - rmh * (rmh - a * λ / 2) * Im) +
          4 * I0 + 2 * I1 + I2
    return I0, I_φ, I_t
end

"""
    pixel_prep(α, β, θo, a; ro = 1000) -> NamedTuple

The layer-independent state of the exact-Kerr equatorial inversion for the
screen point `(α, β)` (Bardeen), observer inclination `θo` (rad) at radius `ro`,
spin `a` (`ro = Inf`: the observer at infinity, with the time regularized as
`lim (I_t − ro − 2 ln ro)`). The domain is normalised by exact Kerr symmetries: `θo > π/2` mirrors
through the equator (`θo → π - θo`, `β → -β`), `a < 0` through the azimuth
(`α → -α`, recorded in `sφ`), and `|a| < 1e-6` is clamped to `1e-6` (the angular
integrals are 0/0 at `a = 0`). Fields: `vortical` (`η < 0`, never crosses the
equator; the other fields are then placeholders), `sφ`, `sθ` (−1 when the equator
mirror was applied), the normalised `a`, the
horizon `rph`, `λ`, `η`, `is_case2` (four real radial roots), `rcase` (2 or 3, the
radial case), `roots`, `ang` (the
angular elliptic terms and the parameter `k = u₊/u₋`), the radial modulus `kr` and observer amplitude `Fr`,
`τturn` (Mino time from the observer to the case-II turning point, the I0
antiderivative at `ro`), `τtot` (the ray's Mino budget: observer → turn →
infinity in case II with r4 outside the horizon, observer → horizon in case II
with all four roots inside it and in case III), and the observer
antiderivatives `po`, `pt`. Consumed by [`band_landing`](@ref).
"""
@inline function pixel_prep(α::T, β, θo, a; ro = T(1000)) where {T}
    sθ = cos(θo) < 0 ? -one(T) : one(T)
    if sθ < 0
        θo = oftype(θo, π) - θo
        β = -β
    end
    sφ = a < 0 ? -one(T) : one(T)
    α = sφ * α
    a = float(abs(a))
    a < 1e-6 && (a = oftype(a, 1e-6))

    rph = kerr_horizon(a)
    λ, η = conserved_quantities(α, β, θo, a)

    if η < 0
        z = zero(λ + η + a + β + ro)
        c = Complex(z)
        parts = (z, z, z, z, z)
        return (vortical=true, sφ=sφ, sθ=sθ, a=a, rph=rph, λ=λ, η=η, is_case2=true,
                rcase=2, roots=(c, c, c, c),
                ang=(u_p=z, K=z, Ep=z, PI=z, F_o=z, Ep_o=z, PI_o=z, Hβ=0, pm_o=z, den=z,
                     k=z, u_m=z),
                kr=z, Fr=z, τturn=z, τtot=z, po=parts, pt=parts)
    end

    roots = radial_roots(λ, η, a)
    u_p, u_m = angular_turning_points(λ, η, a)
    ang = _angular_parts(β, u_p, u_m, θo, a)
    # Case II (real r4) or III (η ≥ 0 has two real roots). Near the critical curve the root
    # splitting falls below Float64 resolution; the split then has a conditioning plateau
    # far below a pixel.
    rs = _radial_state(ro, roots, a, λ, η, rph, T, false)
    return (vortical=false, sφ=sφ, sθ=sθ, a=a, rph=rph, λ=λ, η=η, is_case2=rs.rcase == 2,
            rcase=rs.rcase, roots=roots, ang=ang, kr=rs.kr, Fr=rs.Fr, τturn=rs.τturn,
            τtot=rs.τtot, po=rs.po, pt=rs.pt)
end

"""
    Landing{T}

The source point of one image of one screen point on an emitting surface: the
Boyer–Lindquist radius `r`, azimuth `φ` and emission time `t` (P3 Eqs. 11–12; the
time of the observation is zero), the sign `νr ∈ {-1, +1}` of the radial velocity at
emission, `valid` (false for a ray that does not reach the surface, a landing inside
the horizon, or a Mino time beyond the ray's budget, i.e. an sn/cn alias), the ray's
conserved `λ` and `η` (with the sign of the original spin restored in `λ`), the polar
angle `θ` of the surface and the sign `νθ ∈ {-1, +1}` of `p^θ` at emission. Both
signs are those of the photon travelling from the source to the observer.
"""
struct Landing{T}
    r::T
    φ::T
    t::T
    νr::Int
    valid::Bool
    λ::T
    η::T
    θ::T
    νθ::Int
end

@inline function _landing(r, φ, t, νr, valid, λ, η, θ, νθ)
    T = promote_type(typeof(r), typeof(φ), typeof(t), typeof(λ), typeof(η), typeof(θ))
    return Landing{T}(r, φ, t, νr, valid, λ, η, θ, νθ)
end

"""
    band_landing(prep, n::Integer) -> Landing

The equatorial landing of photon-ring layer `n` (0 direct, 1, 2, …) from the
per-pixel state `prep = pixel_prep(…)`: one sn/cn inversion (P1 Eqs. 30–32 in case
II, P2 Eqs. B57–B75 in case III) and the source-side radial antiderivative. The ray
reaches the equator after `m = n + H(β)` polar turning points and leaves it with
`νθ = sign(β)(−1)^m` (`H(0) = 1`, `sign(0) = 1`), mapped back through the equator
mirror. ForwardDiff-generic.
"""
@inline function band_landing(prep, n::Integer)
    T = typeof(prep.sφ)
    θeq = T(π) / 2
    prep.vortical &&
        return _landing(prep.rph, T(NaN), T(NaN), 1, false, prep.sφ * prep.λ, prep.η, θeq, 1)
    Gθ, Gφ, Gt = _angular_from_parts(n, prep.ang)
    # νθ = sign(β)(−1)^m of the normalised frame, flipped back through the equator mirror
    νθ = (prep.ang.pm_o > 0) == iseven(n + prep.ang.Hβ) ? 1 : -1
    νθ = prep.sθ < 0 ? -νθ : νθ
    r1, r2, r3, r4 = prep.roots
    local rs, νr
    if prep.is_case2
        r31 = r3 - r1; r41 = r4 - r1; r42 = r4 - r2
        sn2 = JacobiElliptic.sn(real(sqrt(r31 * r42)) * Gθ / 2 - prep.Fr, prep.kr)^2
        rs = real((r4 * r31 - r3 * r41 * sn2) / (r31 - r41 * sn2))
        νr = Gθ > prep.τturn ? -1 : 1
    else
        r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        cn3 = JacobiElliptic.cn(sqrt(A * B) * Gθ - prep.Fr, prep.kr)
        r1r = real(r1); r2r = real(r2)
        rs = ((A * r1r - B * r2r) - (A * r1r + B * r2r) * cn3) / ((A - B) - (A + B) * cn3)
        νr = 1
    end
    # sn/cn are periodic: a Gθ beyond the Mino budget aliases to a ghost radius.
    valid = isfinite(rs) && rs > prep.rph && Gθ <= prep.τtot
    I_r, I_φ, I_t = _radial_from_parts(rs, prep.roots, prep.a, prep.λ, prep.η,
                                       prep.is_case2, νr, (prep.po, prep.pt); τ = Gθ)
    φs = -prep.sφ * (I_φ + prep.λ * Gφ) - T(π) / 2     # P3 Eq. 11 (sφ: a < 0 mirror)
    ts = -(I_t + prep.a * prep.a * Gt)                 # P3 Eq. 12
    return _landing(rs, φs, ts, νr, valid, prep.sφ * prep.λ, prep.η, θeq, νθ)
end

# Photon-shell (λ, η) at shell radius r (P2 Eqs. 38–39), with Δ(r) = `kerr_delta`.
@inline function _shell_lambda_eta(r, a)
    Δ = kerr_delta(r, a)
    λ = a + r / a * (r - 2 * Δ / (r - 1))
    η = r^3 / a^2 * (4 * Δ / (r - 1)^2 - r)
    return λ, η
end

"""
    critical_curve(a, θo; n = 100) -> (α::Vector, β::Vector)

The Kerr critical curve (shadow boundary) on the Bardeen screen for spin `a`
and observer inclination `θo` (rad), P2 Eqs. 38–39 and 55: a closed loop of `2n`
points (the upper branch, then the mirrored lower branch), sampled over the
photon-shell radii where `β² > 0` (located on a 2×10⁵-point scan). For
`|a| < 1e-8` the Schwarzschild circle of radius 3√3. Host only.
"""
function critical_curve(a::Float64, θo::Float64; n::Int=100)
    R = 3 * sqrt(3.0)
    if abs(a) < 1e-8
        ψ = range(0, 2π; length=2n)
        return (R .* cos.(ψ), R .* sin.(ψ))
    end
    rM = photon_orbit_radius(a; prograde=true)         # P2 Eq. 40 (photon-shell bounds)
    rP = photon_orbit_radius(a; prograde=false)
    st = sin(θo); ct = cos(θo)
    cot2 = (ct / st)^2
    β2(r) = begin
        λ, η = _shell_lambda_eta(r, a)
        η + a^2 * ct^2 - λ^2 * cot2
    end
    rs = range(rM, rP; length=200000)
    lo = rM; hi = rP; seen = false
    for r in rs
        if β2(r) > 0
            seen || (lo = r; seen = true)
            hi = r
        end
    end
    seen || return (Float64[], Float64[])
    lo += 1e-12; hi -= 1e-12
    αu = Float64[]; βu = Float64[]
    for r in range(lo, hi; length=n)
        λ, η = _shell_lambda_eta(r, a)
        b2 = η + a^2 * ct^2 - λ^2 * cot2
        push!(αu, -λ / st); push!(βu, sqrt(max(b2, 0.0)))
    end
    α = vcat(αu, reverse(αu))
    β = vcat(βu, -reverse(βu))
    return (α, β)
end
