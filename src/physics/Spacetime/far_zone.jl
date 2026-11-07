# The camera at infinity: the ray of Bardeen screen point (α, β) of an observer at
# infinity, carried in from infinity to the plane s = d of the camera through the far
# zone of a stationary, asymptotically flat metric.
#
# Write the position as x = n̂ s + w with n̂ the unit line of sight, s = x·n̂ and w ⊥ n̂,
# and the momentum as p = (1, -n̂ + δp). With u = 1/s the far-zone flow is
#   dw/du = -v⊥/(u² v_s),   dδp/du = -F/(u² v_s),   dτ/du = -(v^t - v_s)/(u² v_s),
# v = g^{μν}p_ν, v_s = v·n̂, v⊥ = v - n̂ v_s, F = dp/dλ, τ = t - s. A metric with
# g - η = O(M/r) gives v⊥ = O(u²), F⊥ = O(u³) and F∥ = O(u²): w and δp are regular at
# u = 0 (the ray from infinity has w(0) = (−β cos i, α, β sin i), δp(0) = 0), and τ has
# the Shapiro term A/u, common to all rays, which is integrated apart. Writing v⊥ =
# (δp + v_c)⊥ with the metric offset v_c = (g^{μν} - η^{μν})p_ν keeps every term small
# against its own scale, so the Float64 flow carries w and δp to rounding of themselves,
# never of s; a metric that forms v_c by subtracting η from g^{-1} (the protocol default)
# leaves an absolute error eps in v_c and a floor eps·s0 in w.
#
# The general solution also holds the singular mode w ∝ 1/u (a ray at another angle),
# which an error in δp⊥ excites with weight 1/u_end; the start at u0 therefore takes δp⊥
# from the trapezoid of δp⊥' ∝ u and w from the midpoint rule, with corrector passes,
# leaving an error O(u0²). Steps u → 2u by `Verner9` follow to u_end = 1/d. Each step
# [u, 2u] has the same relative structure, so the truncation is a fixed fraction of the
# far-zone bending per step, below 1e-15 M at ratio 2.

"Starting inverse distance `u0 = 1/s0` of the far-zone march of the camera at infinity."
const FAR_U0 = 1.0e-8
"Step ratio `u_{k+1}/u_k` of the far-zone march of the camera at infinity."
const FAR_RATIO = 2.0

# (dw, 0, dδp, dτ) per unit u at the far-zone state Z = (w, u, δp, τ).
@inline function _far_rhs(st, nh::SVector{3,T}, nn::T, Z::SVector{8,T}) where {T}
    u = Z[4]
    x = SVector(zero(T), nh[1] / u + Z[1], nh[2] / u + Z[2], nh[3] / u + Z[3])
    p = SVector(one(T), -nh[1] + Z[5], -nh[2] + Z[6], -nh[3] + Z[7])
    vc = raise_index_offset(st, x, p)
    F = geodesic_rhs(st, raystate(x, p))
    q = SVector(Z[5] + vc[2], Z[6] + vc[3], Z[7] + vc[4])
    qn = dot(q, nh) / nn
    vp = q - nh * qn
    dudl = u * u * (one(T) - qn)                          # du/dλ = -u² v_s, v_s = qn - 1
    # dτ/dλ = v^t - |n̂| ds/dλ with τ = t - |n̂| s, from the small terms alone: the flat
    # part (|n̂| - 1) - |n̂| ε of the rounded unit vector cancels with the null condition
    rn = sqrt(nn)
    ε = (Z[5] * nh[1] + Z[6] * nh[2] + Z[7] * nh[3]) / nn
    dτ = (_nn_minus_one(nh) / (rn + 1) - rn * ε) + vc[1] - (vc[2] * nh[1] + vc[3] * nh[2] + vc[4] * nh[3]) / rn
    return SVector(vp[1] / dudl, vp[2] / dudl, vp[3] / dudl, one(T),
                   F[6] / dudl, F[7] / dudl, F[8] / dudl, dτ / dudl)
end

# n̂·n̂ - 1 of the rounded unit vector, to rounding of itself.
@inline _nn_minus_one(nh) = fma(nh[1], nh[1], fma(nh[3], nh[3], fma(nh[2], nh[2], -one(eltype(nh)))))

# δp = ε n̂ + δp⊥ on the null cone at (u, w): Newton on ε of
# H = η(p, p) + p·v_c, dH/dε = 2 (p_sp + v_c,sp)·n̂.
@inline function _far_null(st, nh::SVector{3,T}, w, dpp, u::T) where {T}
    x = SVector(zero(T), nh[1] / u + w[1], nh[2] / u + w[2], nh[3] / u + w[3])
    ε = zero(T)
    nn = dot(nh, nh); dn = _nn_minus_one(nh)
    for _ in 1:4
        ps = -nh + ε * nh + dpp
        p = SVector(one(T), ps[1], ps[2], ps[3])
        vc = raise_index_offset(st, x, p)
        # |p_sp|² - 1 with n̂·δp⊥ = 0, from its small terms
        H = (dn - nn * ε * (2 - ε) + dot(dpp, dpp)) + dot(p, vc)
        dH = 2 * (dot(ps, nh) + vc[2] * nh[1] + vc[3] * nh[2] + vc[4] * nh[3])
        ε -= H / dH
    end
    return ε * nh + dpp
end

@inline _far_state(w, dp, u, τ) = SVector(w[1], w[2], w[3], u, dp[1], dp[2], dp[3], τ)

"""
    infinity_ray(st, frame::CameraFrame{T}, α, β) -> (y::RayState{T}, ok::Bool)

The state, on the plane `x·n̂ = frame.d` (`n̂ = (sin i, 0, cos i)`), of the ray that an
observer at infinity receives at Bardeen screen point `(α, β)`: the far-zone flow in
`u = 1/s` from `u0 = FAR_U0` to `1/d` (see the file header), `p_t = 1`, and `x^t` the
ray's Shapiro delay from the plane relative to the common logarithm, so arrival times
differ between screen points as they do at infinity. `st` must be stationary and
asymptotically flat; on a Kerr–Schild spacetime ([`raise_index_offset`](@ref) without
cancellation) the state carries the ray from infinity to rounding of its components.
`ok` is false when a non-finite state results.
"""
@inline function infinity_ray(st, fr::CameraFrame{T}, α, β) where {T}
    nh = SVector(fr.sin_i, zero(T), fr.cos_i); nn = dot(nh, nh)
    a = T(α); b = T(β)
    w0 = SVector(-b * fr.cos_i, a, b * fr.sin_i)
    proj(v) = v - nh * (dot(v, nh) / nn)
    f(Z) = (@inline; _far_rhs(st, nh, nn, Z))
    u0 = T(FAR_U0); uh = u0 / 2
    # the start at u0: δp⊥ by the trapezoid (δp⊥' ∝ u), w by the midpoint rule
    dp = _far_null(st, nh, w0, zero(w0), u0)
    d0 = f(_far_state(w0, dp, u0, zero(T)))
    dpp = u0 * proj(SVector(d0[5], d0[6], d0[7])) / 2
    w = w0
    for _ in 1:3
        wh = (w0 + w) / 2
        dph = _far_null(st, nh, wh, dpp / 4, uh)
        dm = f(_far_state(wh, dph, uh, zero(T)))
        w = w0 + u0 * proj(SVector(dm[1], dm[2], dm[3]))
        dp = _far_null(st, nh, w, dpp, u0)
        d0 = f(_far_state(w, dp, u0, zero(T)))
        dpp = u0 * proj(SVector(d0[5], d0[6], d0[7])) / 2
    end
    dp = _far_null(st, nh, w, dpp, u0)
    Z = _far_state(w, dp, u0, zero(T))
    A = u0 * f(Z)[8]                                     # dτ/du = A/u + regular
    g(Z) = (@inline; d = f(Z); SVector(d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8] - A / Z[4]))
    uend = one(T) / fr.d
    while Z[4] < uend
        h = min((T(FAR_RATIO) - 1) * Z[4], uend - Z[4])
        Zn = erk_step(Verner9(), g, Z, h)
        Z = SVector(Zn[1], Zn[2], Zn[3], Z[4] + h, Zn[5], Zn[6], Zn[7], Zn[8])
        uend - Z[4] <= 4 * eps(T) * uend && break
    end
    s = fr.d
    y = SVector(Z[8], nh[1] * s + Z[1], nh[2] * s + Z[2], nh[3] * s + Z[3],
                one(T), -nh[1] + Z[5], -nh[2] + Z[6], -nh[3] + Z[7])
    return y, all(isfinite, y)
end
