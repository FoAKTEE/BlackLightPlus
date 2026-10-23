"""
    PlasmaPrimitives{T}

Fluid primitives at a sample point, in code units and AthenaK layout: rest-mass
density `rho`, normal-frame velocity `uu = W v^i` measured by the Eulerian
observer, internal energy density `eint`, cell-centred lab-frame magnetic field
`bb`, and `inside`, false when the point lies outside the provider's domain, in
which case every other field is zero.
"""
struct PlasmaPrimitives{T}
    rho::T
    uu::SVector{3,T}
    eint::T
    bb::SVector{3,T}
    inside::Bool
end

"The primitives of a point outside the domain."
@inline PlasmaPrimitives{T}() where {T} =
    PlasmaPrimitives{T}(zero(T), zero(SVector{3,T}), zero(T), zero(SVector{3,T}), false)

"""
    PlasmaState{T}

The plasma as the transfer coefficients see it: electron number density `n_e`
(cm⁻³), dimensionless electron temperature `theta_e = k Tₑ / mₑ c²`, field
strength `B` (G), the fluid four-velocity and magnetic four-vector in both index
positions (index 1 is time), `b_sq = b^μ b_μ` and magnetisation `sigma = b²/ρ`
in code units, inverse plasma beta `beta_inv`, and `valid`, false for vacuum or
for a sample removed by the cuts. Kinematics of an invalid state are zero and
must not be read. `vacuum(T)` is the one invalid state.
"""
struct PlasmaState{T}
    n_e::T
    theta_e::T
    B::T
    ucon::SVector{4,T}
    ucov::SVector{4,T}
    bcon::SVector{4,T}
    bcov::SVector{4,T}
    b_sq::T
    sigma::T
    beta_inv::T
    valid::Bool
end

"""
    vacuum(::Type{T}) -> PlasmaState{T}

The invalid plasma state: every scalar and vector zero, `valid = false`.
"""
@inline function vacuum(::Type{T}) where {T}
    o = zero(T)
    z = zero(SVector{4,T})
    return PlasmaState{T}(o, o, o, z, z, z, z, o, o, o, false)
end
