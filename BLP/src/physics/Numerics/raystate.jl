# The eight-component ray state y = [x^μ; p_μ]: contravariant position, covariant momentum.

"""
    RayState{T}

`SVector{8,T}` holding a ray state `y = [x^μ; p_μ]`: the contravariant position
`(t, x, y, z)` in slots 1–4 and the covariant momentum `(p_t, p_x, p_y, p_z)` in
slots 5–8. Read it with [`position`](@ref) and [`momentum`](@ref), build it with
[`raystate`](@ref).
"""
const RayState{T} = SVector{8,T}

"""
    position(y::RayState) -> SVector{4}

The contravariant position `x^μ` of the ray state `y`.
"""
@inline position(y::SVector{8}) = SVector(y[1], y[2], y[3], y[4])

"""
    momentum(y::RayState) -> SVector{4}

The covariant momentum `p_μ` of the ray state `y`.
"""
@inline momentum(y::SVector{8}) = SVector(y[5], y[6], y[7], y[8])

"""
    raystate(x::SVector{4}, p::SVector{4}) -> RayState

The ray state `[x^μ; p_μ]`, element type promoted over `x` and `p`.
"""
@inline raystate(x::SVector{4}, p::SVector{4}) =
    SVector(x[1], x[2], x[3], x[4], p[1], p[2], p[3], p[4])
