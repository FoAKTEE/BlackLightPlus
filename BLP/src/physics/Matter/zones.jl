# Monte Carlo emission zones: the `EmissionZone` record a superphoton is generated
# from, and the uniform synchrotron sphere, the flat-spacetime validation model of
# the Monte Carlo transport (the gpumonty sphere test: n_e = 1e13 cm⁻³, Θₑ = 100,
# B = 1 G, R = 1 cm).

"""
    EmissionZone{T}
    EmissionZone(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon)
    EmissionZone{T}(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon)
    EmissionZone(state::PlasmaState, dV, x, g)

One emitting fluid element, the input of superphoton generation, in signature
(−,+,+,+) with time first.

Fields: electron density `n_e` (cm⁻³), temperature `theta_e = k Tₑ/mₑc²`, field
strength `B` (G), proper emitting volume `dV = √−g d³x L³` (cm³), the single factor
that turns the angle-integrated emissivity into a rate, the emission point `x`, the
fluid four-velocity `ucon`/`ucov`, the field four-vector `bcon` (its spatial
direction fixes the field axis), and the metric `gcov` and inverse `gcon` at `x`.

The keyword form converts every argument to `T` (`Float64` without a parameter). The
state form takes the plasma of a valid `PlasmaState`, the volume, the point and the
metric pair `g = (gcov, gcon)` at it, in the promoted number type.
"""
struct EmissionZone{T}
    n_e::T
    theta_e::T
    B::T
    dV::T
    x::SVector{4,T}
    ucon::SVector{4,T}
    ucov::SVector{4,T}
    bcon::SVector{4,T}
    gcov::SMatrix{4,4,T,16}
    gcon::SMatrix{4,4,T,16}
end

function EmissionZone{T}(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon) where {T}
    return EmissionZone{T}(T(n_e), T(theta_e), T(B), T(dV), SVector{4,T}(x),
                           SVector{4,T}(ucon), SVector{4,T}(ucov), SVector{4,T}(bcon),
                           SMatrix{4,4,T}(gcov), SMatrix{4,4,T}(gcon))
end

EmissionZone(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon) =
    EmissionZone{Float64}(; n_e = n_e, theta_e = theta_e, B = B, dV = dV, x = x,
                          ucon = ucon, ucov = ucov, bcon = bcon, gcov = gcov, gcon = gcon)

@inline function EmissionZone(s::PlasmaState{Ts}, dV, x::SVector{4,Tx}, g) where {Ts,Tx}
    gcov, gcon = g
    T = promote_type(Ts, typeof(dV), Tx, eltype(gcov), eltype(gcon))
    return EmissionZone{T}(s.n_e, s.theta_e, s.B, dV, x, s.ucon, s.ucov, s.bcon, gcov, gcon)
end

"""
    UniformSphere{T} <: AnalyticProvider
    UniformSphere(; n_e = 1e13, theta_e = 100.0, B = 1.0, R = 1.0)

A ball of radius `R` of static, uniform, thermal synchrotron plasma in flat
spacetime: density `n_e` (cm⁻³), temperature `theta_e`, field strength `B` (G), and
the emitting volume `dV = (4/3)π R³`. Lengths are in the units of the spacetime
coordinates (cm for the Monte Carlo sphere).

The fluid is at rest, `u^μ = (1, 0, 0, 0)`, and the unit field direction is
`b^μ = (0, 1, 0, 0)` along x¹, which keeps the field off the x³ axis of the emission
tetrad's Gram–Schmidt order. These are the kinematics of `Minkowski()`; covariant
components follow from the metric the caller supplies.

The analytic state ignores the units, electron model and cuts of the parameter set.
"""
struct UniformSphere{T} <: AnalyticProvider
    n_e::T
    theta_e::T
    B::T
    R::T
    dV::T
end

param_type(::UniformSphere{T}) where {T} = T

function UniformSphere(; n_e::Real = 1.0e13, theta_e::Real = 100.0, B::Real = 1.0, R::Real = 1.0)
    T = promote_type(typeof(n_e), typeof(theta_e), typeof(B), typeof(R), Float64)
    dV = (4.0 / 3.0) * π * T(R)^3
    return UniformSphere{T}(T(n_e), T(theta_e), T(B), T(R), dV)
end

"""
    inside_sphere(s::UniformSphere, x) -> Bool

Whether the spatial point of `x` lies in the closed ball, `x₁² + x₂² + x₃² ≤ R²`.
"""
@inline function inside_sphere(s::UniformSphere, x::SVector{4})
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    return r2 <= s.R * s.R
end

"""
    sphere_state(T, s::UniformSphere, gcov) -> PlasmaState{T}

The sphere's plasma with the static fluid and the field along x¹, lowered by `gcov`.
"""
@inline function sphere_state(::Type{T}, s::UniformSphere, gcov::SMatrix{4,4}) where {T}
    ucon = SVector(one(T), zero(T), zero(T), zero(T))
    bcon = SVector(zero(T), one(T), zero(T), zero(T))
    return analytic_state(T, s.n_e, s.theta_e, s.B, ucon, gcov * ucon, bcon, gcov)
end

"""
    plasma_state(s::UniformSphere, st, x, g, params) -> PlasmaState

The sphere's uniform plasma inside the closed ball of radius `R`, `vacuum` outside,
with the covariant kinematics lowered by `gcov` of `g = (gcov, gcon)`. `sigma` and
`beta_inv` are zero; `st` and `params` are not read.
"""
@inline function plasma_state(s::UniformSphere, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(s, x, gcov)
    inside_sphere(s, x) || return vacuum(T)
    return sphere_state(T, s, gcov)
end

"""
    emission_zone(s::UniformSphere, x, g) -> EmissionZone

The emitting zone of the sphere at the point `x`, with `g = (gcov, gcon)` the metric
pair at `x`: the uniform plasma and kinematics of `plasma_state` and the volume of the
whole sphere, so that one zone stands for the sphere in the weight tables and only
`x` changes from photon to photon. `x` is not tested against the ball.
"""
@inline function emission_zone(s::UniformSphere, x::SVector{4}, g)
    gcov = first(g)
    return EmissionZone(sphere_state(analytic_type(s, x, gcov), s, gcov), s.dV, x, g)
end

"""
    domain_radius(s::UniformSphere) -> Float64

The sphere radius `R`.
"""
domain_radius(s::UniformSphere) = Float64(primal(s.R))
