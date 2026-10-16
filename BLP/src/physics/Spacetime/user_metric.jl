# A spacetime from a user closure; every derivative comes from forward-mode AD.

"""
    UserMetric{F,T} <: Spacetime{T}
    UserMetric(g)

The spacetime of a closure `g(x::SVector{4}) -> SMatrix{4,4}` returning the
covariant metric at `x`; the closure must be generic in the number type of `x`
(AD supplies `∂g`). It may depend on `t`: the general geodesic right-hand side
keeps the `∂_t g` term. No horizon is known for an arbitrary metric, so
[`trace_geodesic`](@ref) stops only on the escape sphere or at `λmax`.
`UserMetric(g)` is the `Float64` lane.
"""
struct UserMetric{F,T} <: Spacetime{T}
    g::F
end
UserMetric(f::F) where {F} = UserMetric{F,Float64}(f)

lane(st::UserMetric{F}, ::Type{T}) where {F,T} = UserMetric{F,T}(st.g)

@inline metric(st::UserMetric, x::SVector{4}) = st.g(x)
@inline inverse_metric(st::UserMetric, x::SVector{4}) = inv(metric(st, x))
