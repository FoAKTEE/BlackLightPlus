# gpu_metric.jl — the one shared device-metric layer for the fused GPU kernels: the
# closed-form Cartesian Kerr-Schild metric/inverse and the Hamiltonian geodesic RHS with
# in-kernel 3-lane ForwardDiff Duals (spatial lanes only; the metric is stationary).
# Depends only on StaticArrays + ForwardDiff, but is loaded solely by the CUDA-including
# GPU driver files (not from BlackLightPlus.jl), each via
#   isdefined(@__MODULE__, :_gimg_gi) || include(<this file>)
# so any include order of the GPU files yields exactly one shared copy. Bodies are kept
# verbatim: the pinned regression outputs are fmuladd-contraction-sensitive, so no
# arithmetic reordering (see the _gimg_coeffs note in gpu_image.jl).

using StaticArrays
using ForwardDiff: Dual, partials
using BlackLightPlus.Geodesics: ks_radius_energy, march_stop

# ---------------------------------------------------------------- Kerr closed form
# GPU-safe Kerr-Schild inverse metric on Duals (the closed form CPU `inverse_metric`
# also uses; seeded manually in _gimg_rhs).
@inline function _gimg_gi(x::SVector{4,T}, a) where {T}
    X, Y, Z = x[2], x[3], x[4]
    a2 = a * a
    rr2 = X * X + Y * Y + Z * Z
    r2 = (rr2 - a2 + hypot(rr2 - a2, 2 * a * Z)) / 2
    r = sqrt(r2)
    f = 2 * r2 * r / (r2 * r2 + a2 * Z * Z)
    l1 = (r * X + a * Y) / (r2 + a2); l2 = (r * Y - a * X) / (r2 + a2); l3 = Z / r
    lu = SVector(-one(T), l1, l2, l3)
    eta = SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    return eta - f * (lu * lu')
end

# Hamiltonian RHS with in-kernel 3-partial duals for ∂_c g^{ab} (spatial c only;
# ∂_t g = 0, so a 4th time lane would be dead work in the hottest device function).
# Same closed form as CPU geodesic_rhs, agreeing to fp roundoff (~1 ulp: the manual
# seeding rounds differently from ForwardDiff.jacobian). `a::T` is the promotion guard —
# a Float64 spin fed into a Float32 march is a MethodError, never a silent fp64 promotion.
@inline function _gimg_rhs(y::SVector{8,T}, a::T) where {T}
    x = SVector(y[1], y[2], y[3], y[4])
    p = SVector(y[5], y[6], y[7], y[8])
    z = zero(T); o = one(T)
    xd = SVector(Dual(x[1], (z, z, z)), Dual(x[2], (o, z, z)),
                 Dual(x[3], (z, o, z)), Dual(x[4], (z, z, o)))
    gid = _gimg_gi(xd, a)
    gi = SMatrix{4,4,T}(ntuple(k -> gid[k].value, Val(16)))
    dx = gi * p
    dp1 = z; dp2 = z; dp3 = z
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        prt = partials(gid[μ, ν])
        dp1 -= pp * prt[1]; dp2 -= pp * prt[2]; dp3 -= pp * prt[3]
    end
    return SVector(dx[1], dx[2], dx[3], dx[4], z, dp1 / 2, dp2 / 2, dp3 / 2)
end

# Kerr-Schild radius from the state (matches ks_radius); `a::T` promotion guard as _gimg_rhs.
@inline function _gimg_rks(y::SVector{8,T}, a::T) where {T}
    rr2 = y[2] * y[2] + y[3] * y[3] + y[4] * y[4]
    a2 = a * a
    return sqrt((rr2 - a2 + hypot(rr2 - a2, 2 * a * y[4])) / 2)
end

# Closed-form KS (g_ab, g^ab) values at x (matches metrics.jl metric / inverse_metric).
@inline function _gimg_metric(x::SVector{4,T}, a) where {T}
    X, Y, Z = x[2], x[3], x[4]
    a2 = a * a
    rr2 = X * X + Y * Y + Z * Z
    r2 = (rr2 - a2 + hypot(rr2 - a2, 2 * a * Z)) / 2
    r = sqrt(r2)
    f = 2 * r2 * r / (r2 * r2 + a2 * Z * Z)
    l1 = (r * X + a * Y) / (r2 + a2); l2 = (r * Y - a * X) / (r2 + a2); l3 = Z / r
    ld = SVector(one(T), l1, l2, l3)         # l_a  (covariant, l_0 = +1)
    lu = SVector(-one(T), l1, l2, l3)        # l^a  (contravariant)
    eta = SMatrix{4,4,T}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    return eta + f * (ld * ld'), eta - f * (lu * lu')
end

# ---------------------------------------------------------------- march termination
# The Kerr march control of every fused kernel and the device tracer: the shared
# `march_stop` rule (Geodesics, integrator.jl) on the KS radius and normal-frame energy
# from `ks_radius_energy` — the same functions the CPU sampled tracer calls, so CPU and
# device stop on the same sample. Returns (stop, r); r feeds the step h = (r - r_h)/C.
# Runs in T = eltype(y); the matched ::T scalars make a stray Float64 a MethodError.
@inline function _gimg_stop(y::SVector{8,T}, r_prev::T, r_h::T, escape::T, R_em::T,
                            a::T) where {T}
    r, En = ks_radius_energy(y, a)
    return march_stop(r - r_h, En, r, r_prev, escape, R_em), r
end
