# Hamiltonian null geodesics in covariant-momentum form (as in blacklight):
#   y = [x^μ; p_μ],   dx^μ/dλ = g^{μν} p_ν,   dp_μ/dλ = -½ (∂_μ g^{αβ}) p_α p_β.
# On a stationary metric dp_t/dλ ≡ 0, so p_t is conserved by construction.

"""
    null_error(st, y::RayState)

The null constraint `g^{μν} p_μ p_ν` at the state `y` (zero on an exact null
geodesic).
"""
@inline function null_error(st, y::SVector{8})
    x = position(y)
    p = momentum(y)
    return dot(p, inverse_metric(st, x) * p)
end

"""
    energy(y::RayState)

The conserved energy `E = -p_t` (exact on stationary metrics).
"""
@inline energy(y::SVector{8}) = -y[5]

"""
    angular_momentum_z(y::RayState)

The axial angular momentum `L_z = x p_y - y p_x` of the Killing vector
`∂_φ = x∂_y - y∂_x`, conserved on axisymmetric metrics about z.
"""
@inline angular_momentum_z(y::SVector{8}) = y[2] * y[7] - y[3] * y[6]

"""
    null_covariant(st, x::SVector{4}, p_spatial::SVector{3}) -> (p::SVector{4}, ok::Bool)

The null covector `p` with the given spatial components, solving
`g^{μν} p_μ p_ν = 0` for `p_t` and taking the root `p_t < 0` (`E > 0`). The
quadratic's first root `(-B - √disc)/2A` is taken when negative, else the
second; `ok` is false when neither is negative (inside an ergosphere both roots
may be positive). Allocation- and exception-free.
"""
@inline function null_covariant(st, x::SVector{4,T}, ps::SVector{3,T}) where {T}
    gi = inverse_metric(st, x)
    A = gi[1, 1]
    B = 2 * (gi[1, 2] * ps[1] + gi[1, 3] * ps[2] + gi[1, 4] * ps[3])
    C = zero(T)
    @inbounds for i in 1:3, j in 1:3
        C += gi[i+1, j+1] * ps[i] * ps[j]
    end
    disc = B * B - 4 * A * C
    pt = (-B - sqrt(disc)) / (2 * A)
    ok = true
    if !(pt < 0)
        pt = (-B + sqrt(disc)) / (2 * A)
        ok = pt < 0
    end
    return SVector(pt, ps[1], ps[2], ps[3]), ok
end

"""
    nullify_covariant(st, x::SVector{4}, p_spatial::SVector{3}) -> SVector{4}

[`null_covariant`](@ref) that throws when no `p_t < 0` root exists. Host only.
"""
function nullify_covariant(st, x::SVector{4,T}, ps::SVector{3,T}) where {T}
    p, ok = null_covariant(st, x, ps)
    ok || error("nullify_covariant: no null root with p_t < 0 (E > 0) at x = $x; " *
                "inside the ergosphere the positive-energy covector does not exist")
    return p
end

"""
    geodesic_rhs(st::Spacetime, y::RayState) -> RayState

`dy/dλ` of the Hamiltonian flow. On a `StationarySpacetime` both halves come from
one `raise_and_norm` at a position carrying three spatial dual partials: the
value of the raised covector is `dx^μ/dλ`, the partials of the norm are
`∂_i(2H)`, and `dp_t/dλ` is the exact zero. On a general spacetime the four
`∂_μ g^{αβ}` columns come from AD and `dp_t/dλ` is kept.
"""
@inline function geodesic_rhs(st::StationarySpacetime, y::SVector{8,T}) where {T}
    x = position(y)
    p = momentum(y)
    gp, h2 = raise_and_norm(st, _seed_spatial(x), p)     # g^{μν}p_ν, 2H and ∂_{x,y,z}
    dh = ForwardDiff.partials(h2)
    v = ForwardDiff.value
    return SVector(v(gp[1]), v(gp[2]), v(gp[3]), v(gp[4]), zero(T),
                   -dh[1] / 2, -dh[2] / 2, -dh[3] / 2)
end
@inline geodesic_rhs(::Minkowski, y::SVector{8,T}) where {T} =
    SVector(-y[5], y[6], y[7], y[8], zero(T), zero(T), zero(T), zero(T))
@inline function geodesic_rhs(st::Spacetime, y::SVector{8,T}) where {T}
    x = position(y)
    p = momentum(y)
    gi = inverse_metric(st, x)
    dg = inverse_metric_gradient(st, x)                 # dg[a,b,c] = ∂_c g^{ab}
    dx = gi * p
    dp0 = zero(T); dp1 = zero(T); dp2 = zero(T); dp3 = zero(T)
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        dp0 -= pp * dg[μ, ν, 1]
        dp1 -= pp * dg[μ, ν, 2]
        dp2 -= pp * dg[μ, ν, 3]
        dp3 -= pp * dg[μ, ν, 4]
    end
    return SVector(dx[1], dx[2], dx[3], dx[4], dp0 / 2, dp1 / 2, dp2 / 2, dp3 / 2)
end

"""
    geodesic_transport_rhs(st::Spacetime, y::RayState, vs::NTuple{M,SVector{4}})
        -> (dy, dvs::NTuple{M,SVector{4}})

The geodesic right-hand side with the parallel transport
`dV^μ/dλ = -Γ^μ_{αβ} k^α V^β` of `M` vectors along the ray (`k^μ = dx^μ/dλ`),
from `2Γ^μ_{αβ} k^α V^β = g^{μλ}[(k·∂)g_{λβ} V^β + (V·∂)g_{λα} k^α - k^α V^β ∂_λ g_{αβ}]`;
no Christoffel symbol is formed. On a `StationarySpacetime` (`∂_t g ≡ 0`) one
`metric_and_inverse` at the spatially seeded position gives `g^{ab}` with
`∂_c g^{ab}` (the geodesic, equal to `geodesic_rhs` up to round-off) and
`∂_c g_{ab}` (the transport). On a general spacetime the four columns of both
gradients come from [`inverse_metric_gradient`](@ref) and
[`metric_gradient`](@ref), the time column included, so the transported frame
follows a time-dependent metric. Runs in the number type of the state.
"""
@inline function geodesic_transport_rhs(st::StationarySpacetime, y::SVector{8,T},
                                        vs::NTuple{M,SVector{4,T}}) where {T,M}
    x = position(y)
    p = momentum(y)
    gcovd, gcond = metric_and_inverse(st, _seed_spatial(x))
    gcon = SMatrix{4,4,T}(ntuple(i -> ForwardDiff.value(gcond[i]), Val(16)))
    k = gcon * p
    dp1 = zero(T); dp2 = zero(T); dp3 = zero(T)
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        prt = ForwardDiff.partials(gcond[μ, ν])
        dp1 -= pp * prt[1]; dp2 -= pp * prt[2]; dp3 -= pp * prt[3]
    end
    dy = SVector(k[1], k[2], k[3], k[4], zero(T), dp1 / 2, dp2 / 2, dp3 / 2)
    dg2 = SMatrix{4,4,T}(ntuple(i -> ForwardDiff.partials(gcovd[i], 1), Val(16)))
    dg3 = SMatrix{4,4,T}(ntuple(i -> ForwardDiff.partials(gcovd[i], 2), Val(16)))
    dg4 = SMatrix{4,4,T}(ntuple(i -> ForwardDiff.partials(gcovd[i], 3), Val(16)))
    q2 = dg2 * k; q3 = dg3 * k; q4 = dg4 * k              # ∂_c g_{ab} k^b
    # ∂_c g is symmetric, so (V·∂)g k = Σ_c V^c q_c and k^α V^β ∂_c g_{αβ} = q_c·V
    @inline function _transport(v)
        t1 = k[2] .* (dg2 * v) .+ k[3] .* (dg3 * v) .+ k[4] .* (dg4 * v)   # (k·∂)g V
        t2 = v[2] .* q2 .+ v[3] .* q3 .+ v[4] .* q4
        t3 = SVector(zero(T), dot(q2, v), dot(q3, v), dot(q4, v))
        return -(T(0.5) .* (gcon * (t1 .+ t2 .- t3)))
    end
    return dy, map(_transport, vs)
end

@inline function geodesic_transport_rhs(st::Spacetime, y::SVector{8,T},
                                        vs::NTuple{M,SVector{4,T}}) where {T,M}
    x = position(y)
    p = momentum(y)
    gcon = inverse_metric(st, x)
    dgcon = inverse_metric_gradient(st, x)              # dgcon[a, b, c] = ∂_c g^{ab}
    dgcov = metric_gradient(st, x)                      # dgcov[a, b, c] = ∂_c g_{ab}
    k = gcon * p
    dp1 = zero(T); dp2 = zero(T); dp3 = zero(T); dp4 = zero(T)
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        dp1 -= pp * dgcon[μ, ν, 1]; dp2 -= pp * dgcon[μ, ν, 2]
        dp3 -= pp * dgcon[μ, ν, 3]; dp4 -= pp * dgcon[μ, ν, 4]
    end
    dy = SVector(k[1], k[2], k[3], k[4], dp1 / 2, dp2 / 2, dp3 / 2, dp4 / 2)
    dg1 = SMatrix{4,4,T}(ntuple(i -> dgcov[i], Val(16)))
    dg2 = SMatrix{4,4,T}(ntuple(i -> dgcov[16 + i], Val(16)))
    dg3 = SMatrix{4,4,T}(ntuple(i -> dgcov[32 + i], Val(16)))
    dg4 = SMatrix{4,4,T}(ntuple(i -> dgcov[48 + i], Val(16)))
    q1 = dg1 * k; q2 = dg2 * k; q3 = dg3 * k; q4 = dg4 * k        # ∂_c g_{ab} k^b
    @inline function _transport(v)
        t1 = k[1] .* (dg1 * v) .+ k[2] .* (dg2 * v) .+ k[3] .* (dg3 * v) .+ k[4] .* (dg4 * v)
        t2 = v[1] .* q1 .+ v[2] .* q2 .+ v[3] .* q3 .+ v[4] .* q4
        t3 = SVector(dot(q1, v), dot(q2, v), dot(q3, v), dot(q4, v))
        return -(T(0.5) .* (gcon * (t1 .+ t2 .- t3)))
    end
    return dy, map(_transport, vs)
end

"""
    rk4_step(st, y::RayState{T}, h::T) -> RayState{T}

One classical fixed-step RK4 step of the geodesic flow, in the low-storage
streaming form: the stage sum `((k1 + 2k2) + 2k3) + k4` accumulates as each stage
is formed, with the operations and association of `y + (h/6)(k1 + 2k2 + 2k3 + k4)`.
`h` must have the state's number type (a stray `Float64` in a `Float32` lane is a
`MethodError`); `Dual` states flow through.
"""
@inline function rk4_step(st, y::SVector{8,T}, h::T) where {T}
    d = geodesic_rhs(st, y);  acc = d;         ys = y + (h / 2) * d
    d = geodesic_rhs(st, ys); acc = acc + 2*d; ys = y + (h / 2) * d
    d = geodesic_rhs(st, ys); acc = acc + 2*d; ys = y + h * d
    d = geodesic_rhs(st, ys); acc = acc + d
    return y + (h / 6) * acc
end
