# Time-dependent equatorial emission painted at the closed-form landings: a profile
# f(t, x, y) of the Boyer–Lindquist equatorial Cartesian coordinates x = r cos φ,
# y = r sin φ, evaluated at the emission time t_s + t_obs of each landing, times g^p of the
# g-disk flow. The orbiting Gaussian spot is the flare model of AART (Cárdenas-Avendaño,
# Lupsasca & Zhu 2023, `flare_model`).

"""
    GaussianSpot{T}
    GaussianSpot(; r_c, omega, phi0 = 0, width = 0.5)

The profile `f(t, x, y) = exp(−((x − x_c)² + (y − y_c)²)/w²)` of a spot whose centre
`(x_c, y_c) = r_c (cos(Ωt + φ₀), sin(Ωt + φ₀))` circles the equatorial plane at radius
`r_c` with angular velocity `omega` (AART's `flare_model` with `radhs = r_c`,
`velhs = Ω`, `rwidth = w`). Called as `spot(t, x, y)`.
"""
struct GaussianSpot{T}
    r_c::T
    omega::T
    phi0::T
    width::T
end

function GaussianSpot(; r_c, omega, phi0 = 0.0, width = 0.5)
    T = promote_type(typeof(r_c), typeof(omega), typeof(phi0), typeof(width), Float64)
    return GaussianSpot{T}(T(r_c), T(omega), T(phi0), T(width))
end

@inline function (s::GaussianSpot)(t, x, y)
    sp, cp = sincos(s.omega * t + s.phi0)
    return exp(-((x - s.r_c * cp)^2 + (y - s.r_c * sp)^2) / (s.width * s.width))
end

"""
    EquatorialSpot{T,F}
    EquatorialSpot(disk::EquatorialDisk, profile)
    EquatorialSpot(disk::EquatorialDisk; r_c = 6, omega = Keplerian, phi0 = 0, width = 0.5)

A time-dependent equatorial source: the profile `f(t, x, y)` (an isbits callable, a
[`GaussianSpot`](@ref) by default, on a circular orbit of radius `r_c` with the prograde
Keplerian `Ω = 1/(r_c^{3/2} + a)` unless `omega` is given) times `g^p` of the flow of the
g-disk `disk` (its `redshift_power`, its knobs and its inner cut `rin`). The observed
intensity at observation time `t_obs` along a landing of emission time `t_s` (the
landing's `t`; zero is the observation at the observer's radius, or the retarded time for
an observer at infinity) is `f(t_s + t_obs, r cos φ, r sin φ) g^p`.
"""
struct EquatorialSpot{T,F}
    disk::EquatorialDisk{T}
    profile::F
end

function EquatorialSpot(disk::EquatorialDisk{T}; r_c = 6.0, omega = nothing, phi0 = 0.0,
                        width = 0.5) where {T}
    Ω = omega === nothing ? 1 / (r_c^1.5 + disk.a) : omega
    return EquatorialSpot(disk, GaussianSpot(; r_c = r_c, omega = Ω, phi0 = phi0, width = width))
end

surface_angle(::EquatorialSpot{T}) where {T} = T(π) / 2
material_spin(s::EquatorialSpot) = s.disk.a

"""
    spot_amplitude(spot, landing) -> (A, x, y, t_s)

The time-independent factors of `spot` at `landing`: `A = g^p` of the flow (zero for an
invalid landing, inside the horizon or the inner cut), the equatorial Cartesian position
`(x, y) = r (cos φ, sin φ)` and the emission time `t_s`. The intensity at observation
time `t_obs` is `A f(t_s + t_obs, x, y)`.
"""
@inline function spot_amplitude(s::EquatorialSpot{T}, l::Landing{Tl}) where {T,Tl}
    R = promote_type(T, Tl)
    d = s.disk
    if !l.valid || l.r <= kerr_horizon(d.a) || l.r < d.rin
        return zero(R), zero(R), zero(R), zero(R)
    end
    g = gfactor_disk(l.r, d.a, l.νr, l.λ, l.η; sub_kep = d.sub_kep, betar = d.betar,
                     betaphi = d.betaphi)
    sφ, cφ = sincos(l.φ)
    return R(g^d.redshift_power), R(l.r * cφ), R(l.r * sφ), R(l.t)
end

"""
    intensity(spot::EquatorialSpot, landing, t_obs) -> I

The observed intensity of `spot` along `landing` at observation time `t_obs`.
"""
@inline function intensity(s::EquatorialSpot, l::Landing, t_obs)
    A, x, y, ts = spot_amplitude(s, l)
    return A > 0 ? A * s.profile(ts + t_obs, x, y) : A
end
