# Constant-coefficient polarized (Stokes I,Q,U,V) plasma coupling over one segment
# (blacklight polarized_transfer.hpp StokesCoupleNoSplit) plus the scalar
# exponential-propagator step of unpolarized_transfer.hpp. Coefficients are the
# Lorentz invariants j (=j_nu/nu^2), alpha (=alpha_nu*nu), rho (=rho_nu*nu); U
# components are zero by the tetrad choice, so only the Q (index 2) and V (index 4)
# entries enter. The segment solution exp(-K ds)(S0 - K^{-1} j) + K^{-1} j is affine
# in S0, S -> M S + b, for the Mueller matrix (LD85 / ipole convention, U rows/cols zero)
#   K = [ alpha_I alpha_Q 0 alpha_V ; alpha_Q alpha_I rho_V 0 ;
#         0 -rho_V alpha_I rho_Q ; alpha_V 0 -rho_Q alpha_I ].
# `stokes_segment` / `unpolarized_segment` return the operator (M, b) / (A, B) in closed
# form. The CPU march applies it source -> camera (`stokes_couple`, `couple_unpolarized`);
# the fused kernels chain the same operators camera -> source.

"Optical-depth cap above which a segment is treated as saturated (BlackLight delta_tau_max)."
const POL_DELTA_TAU_MAX = 100.0

"""
    stokes_segment(j, α, ρ, ds; dtau_max=100.0) -> (M::SMatrix{4,4}, b::SVector{4})

Affine operator S -> M S + b of one constant-coefficient segment of invariant length
`ds` with invariant coefficients `j`, `α`, `ρ` (each an `SVector{4}`, U component zero):
emission only, total absorption, pure rotation, or the full absorption + rotation
solution (I 24 / L 10) with the degeneracy guard. A segment with δτ = α_I ds above
`dtau_max` is saturated: M = 0, b = K⁻¹ j.

Deviations from blacklight: (i) polarized absorptivities with α_P > α_I (a fit
violating Mueller admissibility at a domain edge) are rescaled onto α_P = α_I,
preserving α_I and the polarization direction, so the segment decays; (ii) ρ_P = 0
takes the full solution (λ₁ = α_P, λ₂ = 0), which matches matrix-exponential
references where CoupleAbsorptionNoRotation does not; (iii) a degenerate eigenstructure
falls back to total-absorption coupling instead of dividing by zero; (iv) blacklight's
mm_2/mm_3 index typos and mm_4 signs are corrected.
"""
@inline function stokes_segment(j::SVector{4}, α::SVector{4}, ρ::SVector{4}, ds;
                                dtau_max = POL_DELTA_TAU_MAX)
    T = promote_type(eltype(α), eltype(ρ), typeof(ds))
    Tb = promote_type(eltype(j), T)
    I4 = one(SMatrix{4,4,T})
    alpha_sq = α[2] * α[2] + α[4] * α[4]
    alpha_p = sqrt(alpha_sq)
    if α[1] > 0.0 && alpha_p > α[1]
        sc = α[1] / alpha_p
        α = SVector(α[1], α[2] * sc, α[3] * sc, α[4] * sc)
        alpha_sq = α[1] * α[1]
        alpha_p = α[1]
    end
    rho_sq = ρ[2] * ρ[2] + ρ[4] * ρ[4]
    rho_p = sqrt(rho_sq)
    aI = α[1]
    delta_tau = aI * ds
    thin = delta_tau <= dtau_max
    if aI == 0.0 && rho_p == 0.0                       # emission only
        return I4, SVector{4,Tb}(j .* ds)
    elseif alpha_p == 0.0 && rho_p == 0.0              # total absorption
        return _segment_total_absorption(j, aI, delta_tau, thin, T, Tb)
    elseif aI == 0.0                                   # pure rotation (I A2-A5)
        cr = cos(rho_p * ds); sr = sin(rho_p * ds); s2 = sin(rho_p * ds / 2.0); ss = s2 * s2
        rQ = ρ[2]; rV = ρ[4]
        M = @SMatrix [1.0 0.0 0.0 0.0;
                      0.0 (cr + 2.0 * rQ * rQ / rho_sq * ss) (-rV / rho_p * sr) (2.0 * rQ * rV / rho_sq * ss);
                      0.0 (rV / rho_p * sr) cr (-rQ / rho_p * sr);
                      0.0 (2.0 * rV * rQ / rho_sq * ss) (rQ / rho_p * sr) (cr + 2.0 * rV * rV / rho_sq * ss)]
        return SMatrix{4,4,T}(M), SVector{4,Tb}(j .* ds)
    end
    # full absorption + rotation
    aQ = α[2]; aV = α[4]; rQ = ρ[2]; rV = ρ[4]
    ar = aQ * rQ + aV * rV
    asr = alpha_sq - rho_sq
    la = sqrt(asr * asr / 4.0 + ar * ar)
    lb = asr / 2.0
    l1 = sqrt(la + lb)
    l2 = sqrt(la - lb)
    ct = l1 * l1 + l2 * l2
    s = ar >= 0.0 ? 1.0 : -1.0
    a0 = aI * aI
    if ct <= 1.0e-12 * (alpha_sq + rho_sq) || abs(a0 - l1 * l1) <= 1.0e-12 * (a0 + l1 * l1)
        return _segment_total_absorption(j, aI, delta_tau, thin, T, Tb)
    end
    it = 1.0 / ct
    m2 = it .* @SMatrix [0.0 (l2 * aQ - s * l1 * rQ) 0.0 (l2 * aV - s * l1 * rV);
                         (l2 * aQ - s * l1 * rQ) 0.0 (s * l1 * aV + l2 * rV) 0.0;
                         0.0 -(s * l1 * aV + l2 * rV) 0.0 (s * l1 * aQ + l2 * rQ);
                         (l2 * aV - s * l1 * rV) 0.0 -(s * l1 * aQ + l2 * rQ) 0.0]
    m3 = it .* @SMatrix [0.0 (l1 * aQ + s * l2 * rQ) 0.0 (l1 * aV + s * l2 * rV);
                         (l1 * aQ + s * l2 * rQ) 0.0 (-(s * l2 * aV - l1 * rV)) 0.0;
                         0.0 (s * l2 * aV - l1 * rV) 0.0 (-(s * l2 * aQ - l1 * rQ));
                         (l1 * aV + s * l2 * rV) 0.0 (s * l2 * aQ - l1 * rQ) 0.0]
    d0 = (alpha_sq + rho_sq) / 2.0
    m4 = (2.0 * it) .* @SMatrix [d0 0.0 (aQ * rV - aV * rQ) 0.0;
                                 0.0 (aQ * aQ + rQ * rQ - d0) 0.0 (aV * aQ + rV * rQ);
                                 -(aQ * rV - aV * rQ) 0.0 -d0 0.0;
                                 0.0 (aV * aQ + rV * rQ) 0.0 (aV * aV + rV * rV - d0)]
    p4 = I4 .+ m4
    m4m = I4 .- m4
    f1 = 1.0 / (a0 - l1 * l1)
    f2 = 1.0 / (a0 + l2 * l2)
    cosh_t = (-l1 * f1) .* m3 .+ (0.5 * aI * f1) .* p4
    cos_t = (-l2 * f2) .* m2 .+ (0.5 * aI * f2) .* m4m
    pp = cosh_t .+ cos_t
    thin || return zero(SMatrix{4,4,T}), SVector{4,Tb}(pp * j)
    ev = exp(-delta_tau)
    sv = sin(l2 * ds); cv = cos(l2 * ds); shv = sinh(l1 * ds); chv = cosh(l1 * ds)
    sin_t = (-aI * f2) .* m2 .+ (-0.5 * l2 * f2) .* m4m
    sinh_t = (-aI * f1) .* m3 .+ (0.5 * l1 * f1) .* p4
    pp = pp .- ev .* (chv .* cosh_t .+ cv .* cos_t .+ sv .* sin_t .+ shv .* sinh_t)
    oo = ev .* (0.5 .* chv .* p4 .+ 0.5 .* cv .* m4m .- sv .* m2 .- shv .* m3)
    return SMatrix{4,4,T}(oo), SVector{4,Tb}(pp * j)
end

# Total absorptivity only (and the degenerate-branch fallback): exponential decay of every
# Stokes component toward the source function j/α_I; saturated above dtau_max.
@inline function _segment_total_absorption(j, aI, delta_tau, thin, ::Type{T},
                                           ::Type{Tb}) where {T,Tb}
    thin || return zero(SMatrix{4,4,T}), SVector{4,Tb}(j ./ aI)
    e = exp(-delta_tau)
    return SMatrix{4,4,T}(e .* one(SMatrix{4,4,T})),
           SVector{4,Tb}((j ./ aI) .* (-expm1(-delta_tau)))
end

"""
    stokes_admissible(S) -> SVector{4}

Physical admissibility of a Stokes vector: I >= 0 and |P| <= I (the polarization is
rescaled onto the cone). A no-op on admissible input.
"""
@inline function stokes_admissible(S::SVector{4})
    ii = max(S[1], 0.0)
    pol = S[2] * S[2] + S[3] * S[3] + S[4] * S[4]
    if pol > ii * ii
        f = sqrt(ii * ii / pol)
        return SVector(ii, S[2] * f, S[3] * f, S[4] * f)
    end
    return SVector(ii, S[2], S[3], S[4])
end

"""
    stokes_couple(S, j, α, ρ, ds; dtau_max=100.0) -> SVector{4}

Advance the orthonormal-frame Stokes 4-vector `S` (I,Q,U,V) over one segment of
invariant path length `ds` through plasma with invariant coefficients `j`, `α`, `ρ`:
`stokes_admissible(M S + b)` with `(M, b) = stokes_segment(j, α, ρ, ds)`.
"""
@inline function stokes_couple(S::SVector{4}, j::SVector{4}, α::SVector{4},
                               ρ::SVector{4}, ds; dtau_max = POL_DELTA_TAU_MAX)
    M, b = stokes_segment(j, α, ρ, ds; dtau_max = dtau_max)
    return stokes_admissible(M * S + b)
end

"""
    unpolarized_segment(j, α, ds; dtau_max=100.0) -> (A, B)

Affine operator I -> A I + B of one segment of invariant length `ds` with invariant
emissivity `j` and absorptivity `α` (UnpolarizedTransferStep): A = e^{-δτ},
B = (j/α)(1 - e^{-δτ}) with δτ = α ds; saturated (A = 0, B = j/α) above `dtau_max`;
emission only (A = 1, B = j ds) for α <= 0. Generic in the number type (ForwardDiff
`Dual` coefficients flow through).
"""
@inline function unpolarized_segment(j, α, ds; dtau_max = POL_DELTA_TAU_MAX)
    T = promote_type(typeof(j), typeof(α))
    if α > 0.0
        ss = j / α
        delta_tau = α * ds
        if delta_tau <= dtau_max
            e = exp(-delta_tau)
            return (e, e * ss * expm1(delta_tau))
        end
        return (zero(T), ss)
    end
    return (one(T), j * ds)
end

"""
    couple_unpolarized(I, j, α, ds; dtau_max=100.0)

One exact-segment update of the invariant intensity, `A I + B` with
`(A, B) = unpolarized_segment(j, α, ds)`.
"""
@inline function couple_unpolarized(intensity, j, α, ds; dtau_max = POL_DELTA_TAU_MAX)
    A, B = unpolarized_segment(j, α, ds; dtau_max = dtau_max)
    return A * intensity + B
end
