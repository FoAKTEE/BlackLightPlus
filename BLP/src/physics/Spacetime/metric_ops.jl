# Metric derivatives by forward-mode AD of the metric methods, never hand-supplied.
# Stationary spacetimes seed the three spatial coordinates only (∂_t g ≡ 0 exactly).

# Position with three hand-seeded spatial dual partials. The tag is parameterized by
# T: under nested AD (a Dual spin or position) T embeds the outer tag, so this tag
# ranks outermost (ForwardDiff tag-count order) and `partials` extracts these seeds.
@inline function _seed_spatial(x::SVector{4,T}) where {T}
    Tg = typeof(ForwardDiff.Tag(inverse_metric_gradient, T))
    z = zero(T); o = one(T)
    return SVector(ForwardDiff.Dual{Tg}(x[1], (z, z, z)), ForwardDiff.Dual{Tg}(x[2], (o, z, z)),
                   ForwardDiff.Dual{Tg}(x[3], (z, o, z)), ForwardDiff.Dual{Tg}(x[4], (z, z, o)))
end

"""
    inverse_metric_gradient(st::Spacetime, x::SVector{4}) -> SArray{Tuple{4,4,4}}

`dg[a, b, c] = ∂g^{ab}/∂x^c` by forward-mode AD of `inverse_metric`. Stationary
spacetimes differentiate the three spatial coordinates with hand-seeded duals
and return an exact-zero `t` column; the spatial columns equal those of the
four-column Jacobian bit for bit.
"""
@inline function inverse_metric_gradient(st::Spacetime, x::SVector{4,T}) where {T}
    J = ForwardDiff.jacobian(xx -> inverse_metric(st, xx), x)    # (16, 4), column-major
    return SArray{Tuple{4,4,4},T}(J)
end
@inline function inverse_metric_gradient(st::StationarySpacetime, x::SVector{4,T}) where {T}
    z = zero(T)
    gd = inverse_metric(st, _seed_spatial(x))
    return SArray{Tuple{4,4,4},T}((ntuple(_ -> z, Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[1], Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[2], Val(16))...,
                                   ntuple(k -> ForwardDiff.partials(gd[k])[3], Val(16))...))
end
@inline inverse_metric_gradient(::Minkowski, x::SVector{4,T}) where {T} =
    zero(SArray{Tuple{4,4,4},T})

"""
    metric_gradient(st::Spacetime, x::SVector{4}) -> SArray{Tuple{4,4,4}}

`dg[a, b, c] = ∂g_{ab}/∂x^c` (covariant) by forward-mode AD of `metric`.
"""
@inline function metric_gradient(st::Spacetime, x::SVector{4,T}) where {T}
    J = ForwardDiff.jacobian(xx -> metric(st, xx), x)
    return SArray{Tuple{4,4,4},T}(J)
end

"""
    christoffel(st, x)            -> SArray{Tuple{4,4,4}}
    christoffel(st, x, gcon, dg)  -> SArray{Tuple{4,4,4}}

`Γ[μ, α, β] = Γ^μ_{αβ} = ½ g^{μλ}(∂_α g_{λβ} + ∂_β g_{λα} - ∂_λ g_{αβ})`. The
four-argument form contracts a given inverse metric `gcon` and covariant
gradient `dg[a, b, c] = ∂_c g_{ab}` (from [`metric_gradient`](@ref)); the
two-argument form computes both. The λ sum runs left to right from zero.
"""
@inline function christoffel(st::Spacetime, x::SVector{4,T}, gcon, dg) where {T}
    Γl = @inbounds @SArray T[
        (dg[λ, β, α] + dg[λ, α, β] - dg[α, β, λ]) / 2
        for λ in 1:4, α in 1:4, β in 1:4]
    return @inbounds @SArray T[
        (((zero(T) + gcon[μ, 1] * Γl[1, α, β]) + gcon[μ, 2] * Γl[2, α, β]) +
         gcon[μ, 3] * Γl[3, α, β]) + gcon[μ, 4] * Γl[4, α, β]
        for μ in 1:4, α in 1:4, β in 1:4]
end

christoffel(st::Spacetime, x::SVector{4}) =
    christoffel(st, x, inverse_metric(st, x), metric_gradient(st, x))
