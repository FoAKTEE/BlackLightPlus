# Monte Carlo emission zones: the `EmissionZone` record a superphoton is generated
# from, and the flat-spacetime validation models of the Monte Carlo transport: the
# uniform synchrotron sphere (the gpumonty sphere test: n_e = 1e13 cm⁻³, Θₑ = 100,
# B = 1 G, R = 1 cm), the sphere whose density varies in time, and the scattering shell.

"""
    EmissionZone{T}
    EmissionZone(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon)
    EmissionZone{T}(; n_e, theta_e, B, dV, x, ucon, ucov, bcon, gcov, gcon)
    EmissionZone(state::PlasmaState, dV, x, g)

One emitting fluid element, the input of superphoton generation, in signature
(−,+,+,+) with time first.

Fields: electron density `n_e` (cm⁻³), temperature `theta_e = k Tₑ/mₑc²`, field
strength `B` (G), the emitting measure `dV = √−g d³x L³` (cm³, the 4-volume per unit
coordinate time), the single factor that turns the angle-integrated emissivity into a
rate, the emission point `x`, the fluid four-velocity `ucon`/`ucov`, the field
four-vector `bcon` (its spatial direction fixes the field axis), and the metric `gcov`
and inverse `gcon` at `x`.

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
    ZoneSite{T}(x, dV, n_e, theta_e, B)

The compact record of an emission zone of a plasma grid: the emitting point `x`, the
zone's measure `dV = √−g d³x L³` (cm³) and the plasma `n_e` (cm⁻³), `theta_e`, `B` (G) decoded
there, everything the generation tables read. [`emission_zone`](@ref) decodes the full
[`EmissionZone`](@ref) at `x` again when a photon is emitted.
"""
struct ZoneSite{T}
    x::SVector{4,T}
    dV::T
    n_e::T
    theta_e::T
    B::T
end

"""
    emission_zone(provider::PlasmaProvider, st, site::ZoneSite, params) -> EmissionZone{Float64}

The zone of `site`: its plasma `n_e`, `Θₑ`, `B` and volume, with the kinematics and the
metric pair of the provider's [`sample`](@ref) at `site.x` on `st` with `params`. On the
executor that built the site the sample's plasma is the site's, bit for bit; elsewhere
the generation tables built from the site stay the zone's own.
"""
@inline function emission_zone(p::PlasmaProvider, st, site::ZoneSite, params)
    s, g = sample(p, st, site.x, params)
    return EmissionZone{Float64}(site.n_e, site.theta_e, site.B, site.dV, site.x, s.ucon,
                                 s.ucov, s.bcon, g[1], g[2])
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

"""
    VariableSphere{T} <: AnalyticProvider
    VariableSphere(sphere::UniformSphere, times, factors)

The uniform sphere whose electron density varies in time: `n_e(t) = sphere.n_e f(t)`, with
`f` the piecewise-linear function through the nodes `(times[m], factors[m])` (ascending
times, non-negative factors), held at its end values outside `[times[1], times[end]]`
([`density_factor`](@ref)). Temperature, field, radius and kinematics are the sphere's, so
the thermal synchrotron emissivity and absorptivity, both proportional to `n_e`, are
exactly piecewise linear in time between the nodes. Its emission nodes are the spheres
[`node_sphere`](@ref)`(s, m)`. The time is the coordinate `x⁰` of a sample, in cm.
"""
struct VariableSphere{T} <: AnalyticProvider
    sphere::UniformSphere{T}
    times::Vector{Float64}
    factors::Vector{Float64}
    function VariableSphere{T}(sphere::UniformSphere{T}, times::AbstractVector{<:Real},
                               factors::AbstractVector{<:Real}) where {T}
        length(times) == length(factors) >= 1 ||
            throw(ArgumentError("VariableSphere: times and factors must be non-empty and of one length"))
        issorted(times; lt = <=) ||
            throw(ArgumentError("VariableSphere: the node times must be strictly ascending"))
        all(f -> f >= 0, factors) ||
            throw(ArgumentError("VariableSphere: the density factors must be non-negative"))
        return new{T}(sphere, collect(Float64, times), collect(Float64, factors))
    end
end
VariableSphere(sphere::UniformSphere{T}, times::AbstractVector{<:Real},
               factors::AbstractVector{<:Real}) where {T} = VariableSphere{T}(sphere, times, factors)

param_type(::VariableSphere{T}) where {T} = T

"""
    density_factor(s::VariableSphere, t) -> Float64

The density factor `f(t)`: linear between the bracketing nodes, the first (last) factor
before the first (after the last) node.
"""
@inline function density_factor(s::VariableSphere, t::Real)
    ts = s.times
    n = length(ts)
    t <= ts[1] && return s.factors[1]
    t >= ts[n] && return s.factors[n]
    k = searchsortedlast(ts, t)
    w = (t - ts[k]) / (ts[k + 1] - ts[k])
    return (1.0 - w) * s.factors[k] + w * s.factors[k + 1]
end

"""
    node_sphere(s::VariableSphere, m) -> UniformSphere

The uniform sphere of node `m`: the density `sphere.n_e · factors[m]`, everything else the
sphere's.
"""
node_sphere(s::VariableSphere, m::Integer) =
    UniformSphere(n_e = s.sphere.n_e * s.factors[m], theta_e = s.sphere.theta_e, B = s.sphere.B,
                  R = s.sphere.R)

"""
    plasma_state(s::VariableSphere, st, x, g, params) -> PlasmaState

The sphere's plasma at `x` with the density of the time `x⁰`, `vacuum` outside the ball.
"""
@inline function plasma_state(s::VariableSphere, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(s, x, gcov)
    inside_sphere(s.sphere, x) || return vacuum(T)
    u = SVector(one(T), zero(T), zero(T), zero(T))
    b = SVector(zero(T), one(T), zero(T), zero(T))
    return analytic_state(T, s.sphere.n_e * density_factor(s, Float64(primal(x[1]))),
                          s.sphere.theta_e, s.sphere.B, u, gcov * u, b, gcov)
end

"The sphere radius `R`."
domain_radius(s::VariableSphere) = domain_radius(s.sphere)

"""
    SphericalShell{T} <: AnalyticProvider
    SphericalShell(; n_e, theta_e, B = 0.0, r_in, r_out)

A static spherical shell of thermal plasma in flat spacetime, the scattering medium of a
light echo: between `r_in` and `r_out` the electron density is
`n_e sin²(π (r − r_in)/(r_out − r_in))`, zero with its first derivative at both edges, so a
march of step `h` resolves its optical depth and its scattering sites to `O(h²)`; the
temperature `theta_e` and the field strength `B` (G, along x¹) are uniform and the fluid
is at rest. Lengths in the units of the spacetime coordinates (cm in a Monte Carlo sphere
run). Its radial Thomson depth is `n_e σ_T (r_out − r_in)/2`, and the fraction of it
between `r` and `r + dr` is `(2/Δ) sin²(π (r − r_in)/Δ) dr`, `Δ = r_out − r_in`.
"""
struct SphericalShell{T} <: AnalyticProvider
    n_e::T
    theta_e::T
    B::T
    r_in::T
    r_out::T
end

function SphericalShell(; n_e::Real, theta_e::Real, B::Real = 0.0, r_in::Real, r_out::Real)
    0 <= r_in < r_out || throw(ArgumentError("SphericalShell: need 0 ≤ r_in < r_out"))
    T = promote_type(typeof(n_e), typeof(theta_e), typeof(B), typeof(r_in), typeof(r_out), Float64)
    return SphericalShell{T}(T(n_e), T(theta_e), T(B), T(r_in), T(r_out))
end

param_type(::SphericalShell{T}) where {T} = T

"""
    plasma_state(s::SphericalShell, st, x, g, params) -> PlasmaState

The shell's plasma where `r_in < |x| < r_out`, `vacuum` elsewhere.
"""
@inline function plasma_state(s::SphericalShell, st, x::SVector{4}, g, params)
    gcov = first(g)
    T = analytic_type(s, x, gcov)
    r = sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4])
    (r > s.r_in && r < s.r_out) || return vacuum(T)
    sp = sin(π * (r - s.r_in) / (s.r_out - s.r_in))
    u = SVector(one(T), zero(T), zero(T), zero(T))
    b = SVector(zero(T), one(T), zero(T), zero(T))
    return analytic_state(T, s.n_e * sp * sp, s.theta_e, s.B, u, gcov * u, b, gcov)
end

"The outer radius `r_out`."
domain_radius(s::SphericalShell) = Float64(primal(s.r_out))
