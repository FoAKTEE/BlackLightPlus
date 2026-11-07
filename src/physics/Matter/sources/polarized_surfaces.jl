# Polarized emission from thin surfaces painted at the closed-form landings: the emitter
# rest frame (the ZAMO frame boosted by the emitter's velocity), the synchrotron electric
# vector f = k̂ × b of the rest frame (k̂ the photon direction, b the field there; the
# conditions f·p = f·u = f·B = 0 of AART select the same direction for a field given as
# a spacetime vector), its Walker–Penrose constant κ and the observed Stokes Q, U with
# north = +β and east = −α. Models: the power-law synchrotron surface of Narayan et al.
# (2021, ApJ 912, 35) and Gelles et al. (2021, PRD 104 044060) as in Krang's
# `ElectronSynchrotronPowerLawPolarization`, and the AART polarimetric g-disk
# (Cárdenas-Avendaño, Lupsasca & Zhu 2023, Sec. VI): see NOTICE.

"""
    SynchrotronSurface{T}
    SynchrotronSurface(a; theta_s = π/2, b = (0, 0, 1), beta = 0, theta_v = π/2,
                       phi_v = 0, sigma = 1, R = 4, p1 = 1, p2 = 4)

A thin surface of constant polar angle `theta_s` (rad; `π/2` the equator) around a Kerr
hole of spin `a`, emitting power-law synchrotron radiation (Narayan et al. 2021): the
emitter moves with speed `beta` relative to the ZAMO in the direction of polar angle
`theta_v` from `e_(z)` and azimuth `phi_v` from `e_(r)` toward `e_(φ)` (the ZAMO triad
`e_(r)`, `e_(φ)`, `e_(z) = −e_(θ)`), and carries the field `b = (b_r, b_φ, b_z)` in its
rest frame along the boosted triad. With `g` the redshift, `k̂` the photon direction and
`ζ` its angle to the surface normal in the rest frame, the observed intensity is
`|k̂ × b|^(1+σ) min(1/|cos ζ|, 100) (r/R)^{p₁}/(1 + (r/R)^{p₁+p₂}) g^(3+σ)`, fully
linearly polarized along `k̂ × b` (Krang's `ElectronSynchrotronPowerLawPolarization`).
Fields: `a`, `theta_s`, `b`, `uz` (the ZAMO components `(γ, γv)` of the four-velocity),
`sigma`, `R`, `p1`, `p2`, all of the promoted type `T`.
"""
struct SynchrotronSurface{T}
    a::T
    theta_s::T
    b::SVector{3,T}
    uz::SVector{4,T}
    sigma::T
    R::T
    p1::T
    p2::T
end

function SynchrotronSurface(a; theta_s = π / 2, b = (0.0, 0.0, 1.0), beta = 0.0,
                            theta_v = π / 2, phi_v = 0.0, sigma = 1.0, R = 4.0, p1 = 1.0,
                            p2 = 4.0)
    T = promote_type(typeof(a), typeof(theta_s), eltype(b), typeof(beta), typeof(theta_v),
                     typeof(phi_v), typeof(sigma), typeof(R), typeof(p1), typeof(p2), Float64)
    γ = 1 / sqrt(1 - T(beta)^2)
    st, ct = sincos(T(theta_v)); sp, cp = sincos(T(phi_v))
    uz = SVector(γ, γ * beta * st * cp, γ * beta * st * sp, γ * beta * ct)
    return SynchrotronSurface{T}(T(a), T(theta_s), SVector{3,T}(b...), SVector{4,T}(uz),
                                 T(sigma), T(R), T(p1), T(p2))
end

"""
    PolarizedDisk{T}
    PolarizedDisk(disk::EquatorialDisk; b = (1, 0, 0))

The AART polarimetric disk: the g-disk `disk` (its intensity `profile(r) g^p` and its
flow) with the electric vector perpendicular to the photon, to the flow and to the field
`B = b_r e_(r) + b_φ e_(φ) + b_z e_(z)` given in the ZAMO frame (AART's `B^r = √Δ/r` at
the equator is `b = (1, 0, 0)`), and polarized intensity equal to the intensity.
"""
struct PolarizedDisk{T}
    disk::EquatorialDisk{T}
    b::SVector{3,T}
end

PolarizedDisk(disk::EquatorialDisk{T}; b = (1.0, 0.0, 0.0)) where {T} =
    PolarizedDisk{T}(disk, SVector{3,T}(b...))

"""
    surface_angle(material) -> θs

The polar angle of the surface a closed-form material emits from: `theta_s` of a
`SynchrotronSurface`, `π/2` for the equatorial disks.
"""
surface_angle(m::SynchrotronSurface) = m.theta_s
surface_angle(::Union{EquatorialDisk{T},PolarizedDisk{T}}) where {T} = T(π) / 2

"""
    material_spin(material) -> a

The Kerr spin a closed-form material is defined around.
"""
material_spin(m::SynchrotronSurface) = m.a
material_spin(m::EquatorialDisk) = m.a
material_spin(m::PolarizedDisk) = m.disk.a

"""
    rest_photon(landing, a, uz) -> (pcon, pr)

The Boyer–Lindquist momentum `p^μ` of the photon of `landing` (`photon_momentum` at its
radius, polar angle and signs) and its components `pr` in the rest frame of the emitter
of ZAMO four-velocity components `uz`; the redshift is `g = 1/pr[1]`.
"""
@inline function rest_photon(l::Landing, a, uz)
    _, pcon = photon_momentum(l.r, l.θ, l.λ, l.η, a, l.νr, l.νθ)
    return pcon, rest_components(zamo_components(pcon, l.r, l.θ, a), uz)
end

# The Walker–Penrose constant of the rest-frame vector (0, f) and the Stokes (I, Q, U) of
# intensity I fully polarized along it on the screen point (α, β).
@inline function screen_stokes(Iv, f::SVector{3}, pcon, l::Landing, a, uz, α, β, θo)
    fz = rest_components(SVector(zero(eltype(f)), f[1], f[2], f[3]), uz; inverse = true)
    κ = penrose_walker(l.r, l.θ, a, pcon, zamo_vector(fz, l.r, l.θ, a))
    Eα, Eβ = screen_polarization(κ, α, β, θo, a)
    n2 = Eα * Eα + Eβ * Eβ
    n2 > 0 || return Iv, zero(Iv), zero(Iv)
    return Iv, Iv * (Eβ * Eβ - Eα * Eα) / n2, -2 * Iv * Eα * Eβ / n2
end

@inline _cross(k, b) = SVector(k[2] * b[3] - k[3] * b[2], k[3] * b[1] - k[1] * b[3],
                               k[1] * b[2] - k[2] * b[1])

# |k̂ × b|, the path factor, profile and redshift of the synchrotron surface: the observed
# intensity and the electric vector in the rest frame.
@inline function _synchrotron_rest(m::SynchrotronSurface, pr, r)
    khat = SVector(pr[2], pr[3], pr[4]) / pr[1]
    f = _cross(khat, m.b)
    nf = sqrt(f[1]^2 + f[2]^2 + f[3]^2)
    g = 1 / pr[1]
    lp = min(abs(pr[1] / pr[4]), oftype(g, 100))
    x = r / m.R
    prof = x^m.p1 / (1 + x^(m.p1 + m.p2)) * g^(3 + m.sigma)
    return nf^(1 + m.sigma) * lp * prof, f
end

"""
    intensity(m::SynchrotronSurface, landing) -> I_obs

The observed intensity of the synchrotron surface at `landing` (Krang's
`ElectronSynchrotronPowerLawIntensity`); zero when the landing is not valid.
"""
@inline function intensity(m::SynchrotronSurface{T}, l::Landing{Tl}) where {T,Tl}
    R = promote_type(T, Tl)
    l.valid || return zero(R)
    _, pr = rest_photon(l, m.a, m.uz)
    return R(first(_synchrotron_rest(m, pr, l.r)))
end

"""
    polarized_emission(material, landing, α, β, θo) -> (I, Q, U)

The observed Stokes `I`, `Q`, `U` of `material` (`SynchrotronSurface`, `PolarizedDisk`, or
an `EquatorialDisk` with `Q = U = 0`)
along the ray of `landing` arriving at the Bardeen screen point `(α, β)` of an observer at
inclination `θo` (the original, unmirrored coordinates): the electric vector `k̂ × b` of
the emitter rest frame carried to the screen by its Walker–Penrose constant, with
`Q > 0` along north (`+β`) and `U > 0` north-east (east `−α`). Zeros when the landing is
not valid.
"""
@inline function polarized_emission(m::SynchrotronSurface{T}, l::Landing{Tl}, α, β, θo) where {T,Tl}
    R = promote_type(T, Tl)
    z = zero(R)
    l.valid || return z, z, z
    pcon, pr = rest_photon(l, m.a, m.uz)
    Iv, f = _synchrotron_rest(m, pr, l.r)
    I, Q, U = screen_stokes(Iv, f, pcon, l, m.a, m.uz, α, β, θo)
    return R(I), R(Q), R(U)
end

@inline polarized_emission(m::EquatorialDisk, l::Landing, α, β, θo) =
    (z = zero(intensity(m, l)); (intensity(m, l), z, z))

@inline function polarized_emission(m::PolarizedDisk{T}, l::Landing{Tl}, α, β, θo) where {T,Tl}
    R = promote_type(T, Tl)
    z = zero(R)
    Iv = intensity(m.disk, l)
    (l.valid && Iv > 0) || return R(Iv), z, z
    d = m.disk
    a = d.a
    u = disk_velocity(l.r, a; sub_kep = d.sub_kep, betar = d.betar, betaphi = d.betaphi)
    uz = zamo_components(u, l.r, l.θ, a)
    pcon, pr = rest_photon(l, a, uz)
    Bz = SVector(zero(T), m.b[1], m.b[2], m.b[3])
    Br = rest_components(Bz, uz)
    khat = SVector(pr[2], pr[3], pr[4]) / pr[1]
    f = _cross(khat, SVector(Br[2], Br[3], Br[4]))
    I, Q, U = screen_stokes(Iv, f, pcon, l, a, uz, α, β, θo)
    return R(I), R(Q), R(U)
end
