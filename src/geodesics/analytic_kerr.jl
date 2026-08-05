# Geodesics — analytic_kerr.jl
# Exact-Kerr Boyer–Lindquist EQUATORIAL-emission analytic ray tracer (M = 1): per
# pixel, conserved (λ,η) → closed-form quartic radial roots → elliptic integrals
# + Jacobi sn/cn → DIRECT inversion to the equatorial source point of photon-ring
# layer n. No ODE; isbits and ForwardDiff-transparent (CPU-thread/GPU-kernel ready).
# Templated off Krang.jl (MIT, © 2023 Dominic Chang et al.) and AART (MIT, © 2024
# Cárdenas-Avendaño, Zhu & Lupsasca); MIT notices preserved per their terms.
# Physics: Gralla & Lupsasca PRD 101 044032 (P1), 044033 (P2); Cárdenas-Avendaño,
# Lupsasca & Zhu PRD 107 043030 (P3); equation tags (Pn Eq. m) cite these papers.

using JacobiElliptic

export conserved_quantities, radial_roots, angular_turning_points,
       analytic_angular_integrals, analytic_radial_integrals,
       analytic_emission, equatorial_intensity

# Loose "is this complex number effectively real?" test (as in Krang).
@inline _isreal_c(z, tol) = abs(imag(z)) < tol

# Complex-safe cube root (as in AART): principal branch for truly complex arguments.
@inline function _cuberoot(z::Complex{T}) where {T}
    # branch decided on the PRIMAL imaginary part (ForwardDiff `<` compares values);
    # a real argument takes the real cbrt so negative reals map to their real root.
    if abs(imag(z)) < 1e-12 * max(abs(real(z)), one(real(T)))
        return Complex(cbrt(real(z)), zero(T))
    else
        return z^(one(T) / 3)
    end
end

"""
    conserved_quantities(α, β, θo, a) -> (λ, η)

Energy-reduced axial angular momentum λ and Carter constant η for a photon
landing at Bardeen screen coordinate (α, β) for an observer at inclination θo
(rad). P2 Eqs. (58–59). `α` here is the *Bardeen* horizontal coordinate; the
render seam maps the plane-parallel camera's image-α to this via `_cam_to_bardeen`.
"""
@inline function conserved_quantities(α::T, β, θo, a) where {T}
    λ = -α * sin(θo)
    η = (α * α - a * a) * cos(θo)^2 + β * β
    return λ, η
end

"""
    radial_roots(λ, η, a) -> (r1, r2, r3, r4)::NTuple{4,Complex}

The four roots of the radial potential `R(r) = r⁴ + (a²−η−λ²)r² + 2(η+(a−λ)²)r
− a²η` (P2 Eqs. A1–A8), closed form and complex-safe. Ordered r1 ≤ r2 ≤ r3 ≤ r4
when all real (case II / Type II, four real roots); when only two are real
(case III / Type III) r1,r2 are the real pair and r3,r4 a complex-conjugate pair.
"""
@inline function radial_roots(λ, η, a)
    T = promote_type(typeof(λ), typeof(η), typeof(a))    # AD-generic (Dual flows in via η, a)
    z0 = zero(T)
    A = T(a * a - η - λ * λ)
    B = T(2 * (η + (λ - a)^2))
    C = T(-a * a * η)

    P = -A * A / 12 - C
    Q = -A / 3 * ((A / 6)^2 - C) - B * B / 8

    disc = Complex((P / 3)^3 + (Q / 2)^2, z0)
    sq = sqrt(disc)
    w_p = _cuberoot(Complex(-Q / 2, z0) + sq)
    w_m = _cuberoot(Complex(-Q / 2, z0) - sq)

    z = sqrt((w_p + w_m) / 2 - Complex(A / 6, z0))               # P2 Eq. A6
    u = sqrt(Complex(-A / 2, z0) - z * z + Complex(B / 4, z0) / z)
    v = sqrt(Complex(-A / 2, z0) - z * z - Complex(B / 4, z0) / z)

    return (-z - u, -z + u, z - v, z + v)
end

"""
    angular_turning_points(λ, η, a) -> (u_p, u_m)

Roots u_± = cos²θ_± of the angular potential (P2 Eq. 11): for equatorial-crossing
(non-vortical) rays η ≥ 0 ⇒ u_p ≥ 0, u_m ≤ 0. Real for η ≥ 0.
"""
@inline function angular_turning_points(λ::T, η, a) where {T}
    Δθ = (1 - (η + λ * λ) / (a * a)) / 2
    disc = sqrt(Δθ * Δθ + η / (a * a))
    # Take the same-sign root by ADDITION and recover the other from the exact
    # product u_p·u_m = −η/a²: the subtracted root cancels catastrophically at
    # small spin, where Δθ ≈ −(η+λ²)/(2a²) dwarfs the root (u_p rel err 3.5e-7
    # at a = 1e-4, 1.8e-5 at 1e-5 in the naive Δθ + disc form; AART shares this).
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

Angular path integrals for EQUATORIAL emission (θs = π/2 ⇒ source terms vanish),
P2 Eqs. (12–22). `n` is the photon-ring layer (mbar); the effective turning-point
count is `m = n + H(β)` (Heaviside — the disk's far side for β ≥ 0). `Gθ` is the
Mino time τ the radial inversion is evaluated at.
"""
@inline analytic_angular_integrals(n::Integer, β, u_p, u_m, θo, a) =
    _angular_from_parts(n, _angular_parts(β, u_p, u_m, θo, a))

# The n-INDEPENDENT elliptic machinery of `analytic_angular_integrals` (only the
# 2m· multipliers below depend on the layer): computed once per pixel by
# `_emission_prep`, shared by every band.
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
    pm_o = β >= 0 ? one(T) : -one(T)   # H(0)=1 convention (sign(0)=0 would alias β=0)
    den = abs(a) * sqrt(-u_m)          # GL √(−u_m a²)
    return (u_p=u_p, K=K, Ep=E_prime, PI=PI, F_o=F_o, Ep_o=E_prime_o, PI_o=PI_o,
            Hβ=H_β, pm_o=pm_o, den=den)
end

@inline function _angular_from_parts(n::Integer, ang)
    m = n + ang.Hβ                     # effective turning-point count
    Gθ = (2 * m * ang.K - ang.pm_o * ang.F_o) / ang.den
    Gφ = (2 * m * ang.PI - ang.pm_o * ang.PI_o) / ang.den
    Gt = -2 * ang.u_p * (2 * m * ang.Ep - ang.pm_o * ang.Ep_o) / ang.den
    return Gθ, Gφ, Gt
end

# ── radial inversion: equatorial source radius from Mino time τ = Gθ ──────────
# The elliptic-modulus/observer-amplitude machinery (k, F at ro) is n-independent
# and lives in `_emission_prep`; the per-band sn/cn inversion (P1 Eqs. 30–32 for
# Type II; P2 Eqs. B57–B75 for Type III) is inlined in `_emission_band`. The Mino
# time from the observer to the Type-II turning point is the I0 antiderivative at
# ro (`_radial_case2_antideriv`), so no separate function is needed for it.

# Mino time to the outer radial turning point r4 (case II only): P2 Eq. (A10).
@inline function _tau_max_case2(r1, r2, r3, r4, ::Type{T}) where {T}
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
    k2 = real(r32 * r41 / (r31 * r42))
    return 2 / real(sqrt(r31 * r42)) *
           JacobiElliptic.F(asin(clamp(real(sqrt(r31 / r41)), -one(T), one(T))), k2)
end

# Mino time from radius ro inward to the outer horizon (case III, plunging):
# P2 Eq. (A11) with a finite-observer leg — the ray's whole physical Mino budget.
@inline function _mino_to_horizon_case3(ro::T, r1, r2, r3, r4, rph) where {T}
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r21 = r2 - r1
    A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
    k3 = ((A + B)^2 - real(r21)^2) / (4 * A * B)
    x3o = real((A * (ro - r1) - B * (ro - r2)) / (A * (ro - r1) + B * (ro - r2)))
    x3h = real((A * (rph - r1) - B * (rph - r2)) / (A * (rph - r1) + B * (rph - r2)))
    return (JacobiElliptic.F(acos(clamp(x3o, -one(T), one(T))), k3) -
            JacobiElliptic.F(acos(clamp(x3h, -one(T), one(T))), k3)) / sqrt(A * B)
end

# ── radial path integrals I0..Im (finite-observer antiderivatives) ───────────
# AART radial_case{2,3}_antiderivative (P3 App. B). All elliptic arguments are
# reduced to reals via `real(...)` exactly as in the AART source. The source-frame
# radial potential R(r) is `kerr_radial_potential` (kerr_orbits.jl).

# returns (I0, I1, I2, Ip, Im)
@inline function _radial_case2_antideriv(r::T, r1, r2, r3, r4, a, λ, η) where {T}
    rmh = 1 - sqrt(1 - a * a); rph = 1 + sqrt(1 - a * a)
    r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2; r43 = r4 - r3
    rp3 = rph - r3; rm3 = rmh - r3; rp4 = rph - r4; rm4 = rmh - r4
    k = real(r32 * r41 / (r31 * r42))
    x2 = real(sqrt((r - r4) * r31 / ((r - r3) * r41)))
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
    I2 = sqrt(max(kerr_radial_potential(r, a, λ, η), zero(T))) / (r - r3r) -
         real(r1 * r4 + r2 * r3) / 2 * F2 - E2
    Ip = -PI2p - F2 / real(rp3)
    Im = -PI2m - F2 / real(rm3)
    return I0, I1, I2, Ip, Im
end

# R1, R2 auxiliaries (P3 Eqs. B61–65); `arg` is the amplitude φ.
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

# returns (I0, I1, I2, Ip, Im)
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

Definite radial path integrals from source rs to observer ro (P3 Eqs. B1–3). For
Type II the ray turns at r4, so the integral is split with the redshift-sign
weighting w = (νr == −1); Type III is a plain observer−source difference.
"""
@inline analytic_radial_integrals(rs, ro, roots, a, λ, η, is_case2::Bool, νr) =
    _radial_from_parts(rs, roots, a, λ, η, is_case2, νr,
                       _radial_obs_parts(ro, roots, a, λ, η, is_case2)...)

# The n-INDEPENDENT observer-side antiderivatives: case II needs the (ro, r4)
# legs, case III the ro leg (its second slot is unused). Computed once per pixel
# by `_emission_prep`, shared by every band.
@inline function _radial_obs_parts(ro, roots, a, λ, η, is_case2::Bool)
    r1, r2, r3, r4 = roots
    if is_case2
        po = _radial_case2_antideriv(ro, r1, r2, r3, r4, a, λ, η)
        pt = _radial_case2_antideriv(real(r4), r1, r2, r3, r4, a, λ, η)
        return po, pt
    else
        po = _radial_case3_antideriv(ro, r1, r2, r3, r4, a)
        z = zero(po[1])
        return po, (z, z, z, z, z)
    end
end

# Definite integrals given the observer parts `po`/`pt` plus the per-band source
# antiderivative (the only rs-dependent evaluation).
@inline function _radial_from_parts(rs::T, roots, a, λ, η, is_case2::Bool, νr, po, pt) where {T}
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
    rph = 1 + sqrt(1 - a * a); rmh = 1 - sqrt(1 - a * a)
    I_r = I0
    I_φ = 2 * a / (rph - rmh) * ((rph - a * λ / 2) * Ip - (rmh - a * λ / 2) * Im)
    I_t = 4 / (rph - rmh) * (rph * (rph - a * λ / 2) * Ip - rmh * (rmh - a * λ / 2) * Im) +
          4 * I0 + 2 * I1 + I2
    return I_r, I_φ, I_t
end

"""
    analytic_emission(α, β, θo, a, n; ro=1000.0, tol=1e-12) -> (rs, φs, ts, νr, valid)

Exact-Kerr direct-inversion equatorial ray trace: for a photon landing at the
Bardeen screen coordinate (α, β) (observer inclination θo [rad], spin a, M = 1),
return the equatorial source point of photon-ring layer `n` (0 = direct, 1, 2, …)
— BL radius `rs`, source azimuth `φs`, coordinate time-of-flight `ts`, the
radial-velocity sign `νr` at emission (∈ {−1, +1}; the redshift sign), and a
`valid` flag (false if the ray is vortical, lands inside the outer horizon, or
needs more Mino time than the ray physically has — case II: observer → turn →
escape, case III: observer → horizon plunge — so aliased sn/cn ghost landings
are rejected and each layer n is confined to its lensing band).
ForwardDiff-generic in (α, β, θo, a).

DOMAIN. Exact Kerr symmetries normalise the inputs: θo > π/2 mirrors through the
equator (θo → π−θo, β → −β), a < 0 through the azimuth (α → −α; φs, λ mapped
back), and |a| < 1e-6 is clamped to a = 1e-6 (the GL angular integrals are 0/0
at a = 0 exactly). `α` is the *Bardeen* screen coordinate; the render seam maps
the camera image coordinate to it via `_cam_to_bardeen`. The degenerate β = 0 ray
uses the Heaviside convention H(0) = 1 (grazes the disk's far side), shared with
`lensing_bands.jl`.
"""
@inline function analytic_emission(α::T, β, θo, a, n::Integer;
                                   ro=T(1000), tol=T(1e-12)) where {T}
    rs, φs, ts, νr, valid, _, _ = _analytic_emission_full(α, β, θo, a, n; ro=ro, tol=tol)
    return (rs, φs, ts, νr, valid)
end

# Core direct-inversion, returning the same five outputs as `analytic_emission`
# PLUS the ray's conserved quantities `(λ, η)` — so the render seam (which needs
# `(λ, η)` for the g-disk source) gets them for free instead of recomputing
# `conserved_quantities`. `analytic_emission` is the five-tuple public face.
#
# It is split PER PIXEL / PER BAND: `_emission_prep` computes everything that
# does not depend on the ring order n (quartic roots, angular turning points,
# the observer-side elliptic integrals, the Mino budget) ONCE; `_emission_band`
# finishes one layer n from that state (one sn/cn inversion + the source-side
# antiderivative). A multi-band consumer calls prep once and band per n —
# measured ~2x cheaper for 3 bands than three full inversions. Everything is
# isbits, so the pair is GPU-kernel-ready.
@inline function _analytic_emission_full(α::T, β, θo, a, n::Integer;
                                         ro=T(1000), tol=T(1e-12)) where {T}
    return _emission_band(_emission_prep(α, β, θo, a; ro=ro, tol=tol), n)
end

# n-independent per-pixel state (see `_analytic_emission_full`).
@inline function _emission_prep(α::T, β, θo, a; ro=T(1000), tol=T(1e-12)) where {T}
    # Domain normalisation via exact Kerr symmetries (the GL bookkeeping below
    # assumes an upper-hemisphere observer and prograde spin):
    #  * θo > π/2 : equatorial mirror θ → π−θ ⇒ β → −β (rs/φs/ts/νr invariant);
    #  * a < 0    : azimuthal mirror φ → −φ ⇒ α → −α, λ → −λ, and the swept
    #    azimuth Δφ = I_φ + λGφ flips sign (sφ below);
    #  * |a| < 1e-6 : the angular integrals are 0/0 at a = 0 (den = a√(−u_m)),
    #    so clamp to the numerically clean a = 1e-6 (negligible O(a) error).
    if cos(θo) < 0
        θo = oftype(θo, π) - θo
        β = -β
    end
    sφ = a < 0 ? -one(T) : one(T)
    α = sφ * α
    a = float(abs(a))
    a < 1e-6 && (a = oftype(a, 1e-6))

    rph = kerr_horizon(a)
    λ, η = conserved_quantities(α, β, θo, a)

    # Vortical geodesics (η < 0) never cross the equatorial plane. Dummy fields
    # (never read: `_emission_band` returns invalid first) keep the NamedTuple
    # concrete for the Float64/GPU path.
    if η < 0
        z = zero(λ + η + a + β + ro)
        c = Complex(z)
        parts = (z, z, z, z, z)
        return (vortical=true, sφ=sφ, a=a, rph=rph, λ=λ, η=η, is_case2=true,
                roots=(c, c, c, c),
                ang=(u_p=z, K=z, Ep=z, PI=z, F_o=z, Ep_o=z, PI_o=z, Hβ=0, pm_o=z, den=z),
                kr=z, Fr=z, τturn=z, τtot=z, po=parts, pt=parts)
    end

    roots = radial_roots(λ, η, a)
    r1, r2, r3, r4 = roots
    u_p, u_m = angular_turning_points(λ, η, a)
    ang = _angular_parts(β, u_p, u_m, θo, a)

    # NOTE (near-critical conditioning, validated against a Vern7 reltol=1e-11/abstol=1e-13
    # ODE solve): this real/complex-r4 split (case II vs III) is accurate at image scales
    # arbitrarily close to the critical curve. The only degradation is a Float64 conditioning
    # plateau: at critical-curve distance δ→0 the root splitting ~√δ falls below fp
    # resolution, giving Δrs ≈ 5e-4 relative inside a band |δ| ≲ 1e-7 Bardeen units wide —
    # ~5 orders below any pixel (0.01–0.05 M) — with all outputs staying finite/valid. A
    # series-expansion branch for the degenerate quartic would add a hard-to-validate second
    # code path for a band no image samples, so it is deliberately NOT added.
    is_case2 = _isreal_c(r4, T(1e-10) * max(abs(r4), one(T)))
    po, pt = _radial_obs_parts(ro, roots, a, λ, η, is_case2)
    if is_case2
        r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        kr = real(r32 * r41 / (r31 * r42))
        x2 = real(sqrt((ro - r4) * r31 / ((ro - r3) * r41)))
        Fr = JacobiElliptic.F(asin(clamp(x2, -one(T), one(T))), kr)
        # νr threshold = Mino time from the FINITE observer to the turn — exactly
        # the I0 antiderivative at ro. (AART's Jmax uses the r→∞ amplitude of P2
        # Eq. A10 instead, so rays in the ~1/ro-wide window (τ(ro), τ(∞)) get the
        # wrong radial-velocity sign there — inherited weakness, fixed here.)
        τturn = po[1]                                  # obs → turn
        τtot = po[1] + _tau_max_case2(r1, r2, r3, r4, T)   # obs → turn → escape
    else
        r21 = r2 - r1; r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        kr = ((A + B)^2 - real(r21)^2) / (4 * A * B)
        x3o = real((A * (ro - r1) - B * (ro - r2)) / (A * (ro - r1) + B * (ro - r2)))
        Fr = JacobiElliptic.F(acos(clamp(x3o, -one(T), one(T))), kr)
        τturn = zero(kr)                               # unused (no Type-III turn)
        τtot = _mino_to_horizon_case3(ro, r1, r2, r3, r4, rph)  # obs → horizon plunge
    end
    return (vortical=false, sφ=sφ, a=a, rph=rph, λ=λ, η=η, is_case2=is_case2,
            roots=roots, ang=ang, kr=kr, Fr=Fr, τturn=τturn, τtot=τtot, po=po, pt=pt)
end

# One photon-ring layer n from the shared per-pixel state (see
# `_analytic_emission_full` for the output contract).
@inline function _emission_band(prep, n::Integer)
    T = typeof(prep.sφ)
    prep.vortical && return (prep.rph, T(NaN), T(NaN), 1, false, prep.sφ * prep.λ, prep.η)
    Gθ, Gφ, Gt = _angular_from_parts(n, prep.ang)
    r1, r2, r3, r4 = prep.roots
    local rs, νr
    if prep.is_case2                                   # P1 Eqs. (30–32)
        r31 = r3 - r1; r41 = r4 - r1; r42 = r4 - r2
        sn2 = JacobiElliptic.sn(real(sqrt(r31 * r42)) * Gθ / 2 - prep.Fr, prep.kr)^2
        rs = real((r4 * r31 - r3 * r41 * sn2) / (r31 - r41 * sn2))
        νr = Gθ > prep.τturn ? -1 : 1
    else                                               # P2 Eqs. (B57–B75)
        r31 = r3 - r1; r32 = r3 - r2; r41 = r4 - r1; r42 = r4 - r2
        A = real(sqrt(r32 * r42)); B = real(sqrt(r31 * r41))
        cn3 = JacobiElliptic.cn(sqrt(A * B) * Gθ - prep.Fr, prep.kr)
        r1r = real(r1); r2r = real(r2)
        rs = ((A * r1r - B * r2r) - (A * r1r + B * r2r) * cn3) / ((A - B) - (A + B) * cn3)
        νr = 1
    end

    # sn/cn are periodic, so a Gθ beyond the ray's physical Mino budget τtot
    # aliases to a ghost radius that can still pass rs > rph — flag it invalid.
    valid = isfinite(rs) && rs > prep.rph && Gθ <= prep.τtot
    I_r, I_φ, I_t = _radial_from_parts(rs, prep.roots, prep.a, prep.λ, prep.η,
                                       prep.is_case2, νr, prep.po, prep.pt)
    φs = -prep.sφ * (I_φ + prep.λ * Gφ) - T(π) / 2     # P3 Eq. 11 (sφ: a<0 mirror)
    ts = -(I_t + prep.a * prep.a * Gt)                 # P3 Eq. 12
    return (rs, φs, ts, νr, valid, prep.sφ * prep.λ, prep.η)
end

"""
    equatorial_intensity(rs, νr, a; rin=0.0, source=nothing, λ=0, η=0) -> Float64

The equatorial-source SEAM. Given the landing radius `rs`, the emitted-p^r sign
`νr` and (for `source`) the ray's conserved quantities `(λ, η)`, return the
observed specific intensity, zero off the disk (inside the horizon or `rs < rin`):

  * `source === nothing` (default) — a placeholder: an optically thin
    r⁻² surface-brightness falloff (kept so the bare tracer stays self-contained).
  * `source` a callable `(rs, νr, a, λ, η) -> I` — the injected AART g-disk /
    g-gas redshift source (`Radiation.equatorial_disk_intensity`, wired at the
    render/root scope where Radiation is visible; Geodesics stays dependency-free
    of Radiation). This is how `render_analytic(cfg; source=EquatorialDisk(a))`
    produces the layered photon-ring images.

The `(rs, νr, a; …)` positional call shape is preserved for the render seam.
"""
@inline function equatorial_intensity(rs::T, νr, a; rin=T(0), source=nothing,
                                      λ=zero(T), η=zero(T)) where {T}
    rph = 1 + sqrt(1 - a * a)
    (rs <= rph || rs < rin) && return zero(T)
    source === nothing && return one(T) / (rs * rs)
    return source(rs, νr, a, λ, η)
end
