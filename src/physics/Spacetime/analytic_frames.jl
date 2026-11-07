# Frames and polarization at a closed-form landing (Boyer–Lindquist, M = 1, signature
# −+++, photon energy at infinity 1): the Kerr metric, the photon momentum, the
# zero-angular-momentum observer (ZAMO) frame and the rest frame of an emitter, the
# Walker–Penrose constant of a polarization vector and its screen polarization at the
# observer (Himwich et al. 2020, PRD 101 084020, Eqs. 6, 9; Gelles et al. 2021, PRD 104
# 044060, Eq. 31), and the photon-shell exponents of the critical curve (Gralla &
# Lupsasca 2020, PRD 101 044032, Eqs. 68, 70, 72; Johnson et al. 2020).

"""
    kerr_bl_metric(r, θ, a) -> (gcov, gcon)

The Kerr metric and its inverse in Boyer–Lindquist coordinates `(t, r, θ, φ)` at
`(r, θ)`: with `Σ = r² + a² cos²θ`, `Δ = r² − 2r + a²`, `A = (r² + a²)² − a²Δ sin²θ`,
`g_tt = −(1 − 2r/Σ)`, `g_tφ = −2ar sin²θ/Σ`, `g_φφ = A sin²θ/Σ`, `g_rr = Σ/Δ`,
`g_θθ = Σ`; `g^tt = −A/(ΣΔ)`, `g^tφ = −2ar/(ΣΔ)`, `g^φφ = (Δ − a² sin²θ)/(ΣΔ sin²θ)`.
"""
@inline function kerr_bl_metric(r, θ, a)
    s, c = sincos(θ)
    Σ = r * r + a * a * c * c
    Δ = kerr_delta(r, a)
    A = (r * r + a * a)^2 - a * a * Δ * s * s
    z = zero(Σ)
    gcov = SMatrix{4,4}(-(1 - 2r / Σ), z, z, -2a * r * s * s / Σ,
                        z, Σ / Δ, z, z,
                        z, z, Σ, z,
                        -2a * r * s * s / Σ, z, z, A * s * s / Σ)
    gcon = SMatrix{4,4}(-A / (Σ * Δ), z, z, -2a * r / (Σ * Δ),
                        z, Δ / Σ, z, z,
                        z, z, 1 / Σ, z,
                        -2a * r / (Σ * Δ), z, z, (Δ - a * a * s * s) / (Σ * Δ * s * s))
    return gcov, gcon
end

"""
    photon_momentum(r, θ, λ, η, a, νr, νθ) -> (pcov, pcon)

The momentum of the photon of conserved `(λ, η)` at `(r, θ)` with radial and polar
signs `νr`, `νθ`: `p_μ = (−1, νr √R/Δ, νθ √Θ, λ)` (`R`, `Θ` clamped at zero) and
`p^μ = g^{μν} p_ν`.
"""
@inline function photon_momentum(r, θ, λ, η, a, νr, νθ)
    _, gcon = kerr_bl_metric(r, θ, a)
    Θ = η + a * a * cos(θ)^2 - λ * λ * cot(θ)^2
    R = kerr_radial_potential(r, a, λ, η)
    z = zero(R + Θ)
    pcov = SVector(-one(z), νr * sqrt(max(R, z)) / kerr_delta(r, a), νθ * sqrt(max(Θ, z)), λ + z)
    return pcov, gcon * pcov
end

"""
    zamo_components(v, r, θ, a) -> SVector{4}

The components `(v^(t), v^(x), v^(y), v^(z))` of the Boyer–Lindquist vector `v^μ` in the
ZAMO frame at `(r, θ)`, with `x` along `e_(r)`, `y` along `e_(φ)` and `z = −e_(θ)`
(toward the north pole at the equator): `v^(t) = α_L v^t`, `v^(x) = √(Σ/Δ) v^r`,
`v^(y) = √(A/Σ) sin θ (v^φ − ω v^t)`, `v^(z) = −√Σ v^θ`, with the lapse
`α_L = √(ΣΔ/A)` and `ω = 2ar/A`.
"""
@inline function zamo_components(v::SVector{4}, r, θ, a)
    s, c = sincos(θ)
    Σ = r * r + a * a * c * c
    Δ = kerr_delta(r, a)
    A = (r * r + a * a)^2 - a * a * Δ * s * s
    ω = 2a * r / A
    return SVector(sqrt(Σ * Δ / A) * v[1], sqrt(Σ / Δ) * v[2], sqrt(A / Σ) * s * (v[4] - ω * v[1]),
                   -sqrt(Σ) * v[3])
end

"""
    zamo_vector(vz, r, θ, a) -> SVector{4}

The Boyer–Lindquist components of the vector of ZAMO components `vz` (the inverse of
[`zamo_components`](@ref)).
"""
@inline function zamo_vector(vz::SVector{4}, r, θ, a)
    s, c = sincos(θ)
    Σ = r * r + a * a * c * c
    Δ = kerr_delta(r, a)
    A = (r * r + a * a)^2 - a * a * Δ * s * s
    ω = 2a * r / A
    vt = vz[1] / sqrt(Σ * Δ / A)
    return SVector(vt, vz[2] / sqrt(Σ / Δ), -vz[4] / sqrt(Σ), vz[3] / (sqrt(A / Σ) * s) + ω * vt)
end

"""
    rest_components(vz, uz; inverse = false) -> SVector{4}

The components in the rest frame of an emitter of ZAMO four-velocity components
`uz = (γ, γv)` of the vector of ZAMO components `vz`, by the pure boost
`Λ⁰₀ = γ`, `Λ⁰ᵢ = −uᵢ`, `Λⁱⱼ = δᵢⱼ + uᵢuⱼ/(1 + γ)`; `inverse = true` maps rest-frame
components back to the ZAMO frame (the boost by `−v`).
"""
@inline function rest_components(vz::SVector{4}, uz::SVector{4}; inverse::Bool = false)
    γ = uz[1]
    sg = inverse ? -one(γ) : one(γ)
    ux = sg * uz[2]; uy = sg * uz[3]; uw = sg * uz[4]
    ud = ux * vz[2] + uy * vz[3] + uw * vz[4]
    q = ud / (1 + γ) - vz[1]
    return SVector(γ * vz[1] - ud, vz[2] + q * ux, vz[3] + q * uy, vz[4] + q * uw)
end

"""
    penrose_walker(r, θ, a, p, f) -> Complex

The Walker–Penrose constant `κ = κ₁ + iκ₂ = (r − ia cos θ)(A − iB)` of the polarization
vector `f^μ` carried by the photon of momentum `p^μ` (both Boyer–Lindquist,
contravariant) at `(r, θ)`, with `A = (p^t f^r − p^r f^t) + a sin²θ (p^r f^φ − p^φ f^r)`
and `B = [(r² + a²)(p^φ f^θ − p^θ f^φ) − a(p^t f^θ − p^θ f^t)] sin θ`; conserved along the
null geodesic, invariant under `f → f + c p`, and `|κ|² = (f·f)(η + (λ − a)²)`.
"""
@inline function penrose_walker(r, θ, a, p::SVector{4}, f::SVector{4})
    s, c = sincos(θ)
    pt, pr, pθ, pφ = p
    ft, fr, fθ, fφ = f
    A = (pt * fr - pr * ft) + a * s * s * (pr * fφ - pφ * fr)
    B = ((r * r + a * a) * (pφ * fθ - pθ * fφ) - a * (pt * fθ - pθ * ft)) * s
    return Complex(r * A - a * c * B, -(a * c * A + r * B))
end

"""
    screen_polarization(κ, α, β, θo, a) -> (E_α, E_β)

The polarization vector on the Bardeen screen of an observer at infinity at inclination
`θo` of the photon of Walker–Penrose constant `κ` arriving at `(α, β)`: with
`ν = −(α + a sin θo)`, `E_α = (βκ₂ − νκ₁)/(ν² + β²)` and `E_β = (βκ₁ + νκ₂)/(ν² + β²)`.
`ν² + β² = η + (λ − a)²`, so `E_α² + E_β² = f·f`, the transported norm. The EVPA east of
north (north `+β`, east `−α`) is `atan(−E_α, E_β)`; `Q = E_β² − E_α²`, `U = −2E_αE_β`.
"""
@inline function screen_polarization(κ::Complex, α, β, θo, a)
    ν = -(α + a * sin(θo))
    K = ν * ν + β * β
    k1 = real(κ); k2 = imag(κ)
    return (β * k2 - ν * k1) / K, (β * k1 + ν * k2) / K
end

"""
    photon_shell_exponents(a, r) -> (γ, δ, τ)

The Lyapunov exponent `γ`, azimuthal lapse `δ` and time lapse `τ` per half orbit of the
bound photon orbit of Boyer–Lindquist radius `r` (the point of the critical curve of shell
radius `r`): with the critical `(λ̃, η̃)`, `ũ±`, `k = ũ₊/ũ₋`, `χ = 1 − Δ/(r (r − 1)²)`,
`γ = 4r√χ K(k)/(a√−ũ₋)`, `δ = 2/√−ũ₋ [(r + 1)/(r − 1) K(k) + λ̃ Π(ũ₊|k)/a] + 2π H(r − r₀)`
and `τ = 2/(a√−ũ₋) [r² (r + 3)/(r − 1) K(k) − 2a² ũ₊ (E(k) − K(k))/(2k)]`, `r₀` the radius
of the pole-crossing orbit (`λ̃ = 0`). Successive images demagnify by `e^{−γ}`, rotate by
`δ` and are delayed by `τ`; `a → 0` gives `(π, π, 3√3π)`.
"""
function photon_shell_exponents(a::Real, r::Real)
    a = max(abs(Float64(a)), 1e-6)
    r = Float64(r)
    λ, η = _shell_lambda_eta(r, a)
    u_p, u_m = angular_turning_points(λ, η, a)
    k = u_p / u_m
    K = JacobiElliptic.K(k)
    χ = 1 - kerr_delta(r, a) / (r * (r - 1)^2)
    sq = sqrt(-u_m)
    γ = 4 * r * sqrt(χ) * K / (a * sq)
    r0 = 1 + 2 * sqrt(1 - a^2 / 3) * cos(acos((1 - a^2) / (1 - a^2 / 3)^1.5) / 3)
    δ = 2 / sq * ((r + 1) / (r - 1) * K + λ * JacobiElliptic.Pi(u_p, π / 2, k) / a) +
        (r > r0 ? 2π : 0.0)
    τ = 2 / (a * sq) * (r^2 * (r + 3) / (r - 1) * K - 2 * a^2 * u_p * (JacobiElliptic.E(k) - K) / (2k))
    return γ, δ, τ
end
