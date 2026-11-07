# Segment operators of constant-coefficient transfer (blacklight polarized_transfer.hpp
# StokesCoupleNoSplit and unpolarized_transfer.hpp). Over an invariant length ds the
# solution S(ds) = exp(-K ds) S₀ + ∫₀^ds exp(-K s) ds j is affine in S₀, S → M S + b, with
# the Mueller matrix (Landi Degl'Innocenti 1985 / ipole convention, U rows and columns zero)
#   K = [ α_I α_Q 0 α_V ; α_Q α_I ρ_V 0 ; 0 -ρ_V α_I ρ_Q ; α_V 0 -ρ_Q α_I ].
# The marches chain the operators camera → source; `couple` applies one source → camera.

"Optical depth above which a segment is saturated (blacklight `delta_tau_max`): α_I ds for the
scalar operator, the depth (α_I − λ₁) ds of the slowest polarized mode for `IQUV`."
const POL_DELTA_TAU_MAX = 100.0

"""
    segment(::StokesI, j, α, ds; dtau_max = POL_DELTA_TAU_MAX) -> (A, B)

Affine operator `I → A I + B` of one segment of invariant length `ds` with invariant
emissivity `j` and absorptivity `α` (blacklight `UnpolarizedTransferStep`):
`A = e^{-δτ}`, `B = (j/α) e^{-δτ} expm1(δτ)` with `δτ = α ds`; saturated
(`A = 0`, `B = j/α`) above `dtau_max`; emission only (`A = 1`, `B = j ds`) for
`α ≤ 0`. Generic in the number type.
"""
@inline function segment(::StokesI, j, α, ds; dtau_max = POL_DELTA_TAU_MAX)
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

"Relative gap `|α_I² − λ₁²|/(α_I² + λ₁²)` below which `segment(::IQUV)` integrates the eigenmodes directly."
const POL_LANDI_GAP = 1.0e-6

"""
    segment(::IQUV, j, α, ρ, ds; dtau_max = POL_DELTA_TAU_MAX, project = true)
        -> (M::SMatrix{4,4}, b::SVector{4})

Affine operator `S → M S + b`, `M = e^{-K ds}`, `b = ∫₀^ds e^{-K s} ds j`, of one
constant-coefficient segment of invariant length `ds` with invariant coefficients `j`,
`α`, `ρ` (`SVector{4}`, U components of `α`, `ρ` unused). Write `K = α_I 1 + N`; the
eigenvalues of `N` are `±λ₁`, `±iλ₂` (Landi Degl'Innocenti 1985) with
`λ₁,₂² = [(α_P² − ρ_P²)² / 4 + (α·ρ)²]^{1/2} ± (α_P² − ρ_P²)/2`, and every case below is
exact for constant coefficients:

- emission only (`α_I = α_P = ρ_P = 0`): `M = 1`, `b = j ds`;
- total absorption (`α_P = ρ_P = 0`): `M = e^{-α_I ds} 1`, `b = (1 − e^{-α_I ds}) j/α_I`;
- rotation without absorption (`α_I = α_P = 0`): `N` rotates `(Q,U,V)` about `ρ̂` at the
  rate `ρ_P`, `e^{-N s} = cos θ 1 + (1 − cos θ) ρ̂ρ̂ᵀ + sin θ [ρ̂×]` with `θ = ρ_P s`, and
  integrating each term gives `b_I = j_I ds`,
  `b_P = (sin θ/ρ_P) j_P + ((θ − sin θ)/ρ_P) ρ̂(ρ̂·j_P) + ((1 − cos θ)/ρ_P) ρ̂ × j_P` at
  `θ = ρ_P ds`;
- general (Landi Degl'Innocenti 1985): `e^{-K s} = e^{-α_I s}[½(cosh λ₁s + cos λ₂s) M₁ −
  sin λ₂s M₂ − sinh λ₁s M₃ + ½(cosh λ₁s − cos λ₂s) M₄]`, so `b = P j` with
  `P = ½(C₁ + C₂) M₁ − S₂ M₂ − S₁ M₃ + ½(C₁ − C₂) M₄`, where `C₁, S₁` and `C₂, S₂` are the
  integrals of `e^{-α_I s}(cosh, sinh) λ₁s` and `e^{-α_I s}(cos, sin) λ₂s` over the
  segment. Their closed forms carry `1/(α_I² − λ₁²)` and `1/(α_I² + λ₂²)`; within
  `POL_LANDI_GAP` of those poles, `C₁ ± S₁ = ds φ((α_I ∓ λ₁) ds)` and
  `C₂ + iS₂ = ds φ((α_I − iλ₂) ds)` with `φ(z) = (1 − e^{-z})/z`, regular at `z = 0`, give
  the exact limit. With `ρ = 0` this is the dichroism-cone boundary `|α_P| = α_I`: the
  mode polarized against `α̂` decays at `α_I − α_P = 0` and gathers its emission linearly,
  so the step is never saturated there;
- nilpotent `N` (`λ₁ = λ₂ = 0`, i.e. `|α_P| = |ρ_P|` and `α ⊥ ρ`): `N⁴ = 0`, so
  `e^{-K s} = e^{-α_I s}(1 − sN + s²N²/2 − s³N³/6)` and `P = Σₖ (−N)ᵏ gₖ` with
  `gₖ = ∫₀^ds sᵏ/k! e^{-α_I s} ds`.

A segment is saturated, `M = 0` and `b = K⁻¹ j`, when its slowest mode is deeper than
`dtau_max`, `(α_I − λ₁) ds > dtau_max`; a segment with `α_I ds > dtau_max` and a shallower
slow mode takes the eigenmode integrals, whose decay factors stay finite at any depth. With `project = true` (production) the coefficients are first projected onto
the physical cone: `|j_P| > j_I > 0` is rescaled onto `|j_P| = j_I` and `|α_P| > α_I` onto
`|α_P| = α_I` (onto `α_P = 0` when `α_I ≤ 0`), keeping `j_I`, `α_I` and the polarization
direction; the thermal fits leave the cone at small pitch angles. `project = false`
solves the transfer equation for the coefficients as given, outside the cone too (the
tests of Prather et al. 2023). Generic in the number type.
"""
@inline function segment(::IQUV, j::SVector{4}, α::SVector{4}, ρ::SVector{4}, ds;
                         dtau_max = POL_DELTA_TAU_MAX, project::Bool = true)
    T = promote_type(eltype(α), eltype(ρ), typeof(ds))
    Tb = promote_type(eltype(j), T)
    I4 = one(SMatrix{4,4,T})
    if project
        j_p = sqrt(j[2] * j[2] + j[3] * j[3] + j[4] * j[4])
        if j[1] > 0.0 && j_p > j[1]
            sj = j[1] / j_p
            j = SVector(j[1], j[2] * sj, j[3] * sj, j[4] * sj)
        end
    end
    alpha_sq = α[2] * α[2] + α[4] * α[4]
    alpha_p = sqrt(alpha_sq)
    if project && alpha_p > α[1]
        if α[1] > 0.0
            sc = α[1] / alpha_p
            α = SVector(α[1], α[2] * sc, α[3] * sc, α[4] * sc)
            alpha_sq = α[1] * α[1]
            alpha_p = α[1]
        else
            α = SVector(α[1], zero(alpha_p), zero(alpha_p), zero(alpha_p))
            alpha_sq = zero(alpha_sq)
            alpha_p = zero(alpha_p)
        end
    end
    rho_sq = ρ[2] * ρ[2] + ρ[4] * ρ[4]
    rho_p = sqrt(rho_sq)
    aI = α[1]
    delta_tau = aI * ds
    thin = delta_tau <= dtau_max
    if aI == 0.0 && alpha_p == 0.0 && rho_p == 0.0     # emission only
        return I4, SVector{4,Tb}(j .* ds)
    elseif alpha_p == 0.0 && rho_p == 0.0              # total absorption
        return _segment_total_absorption(j, aI, delta_tau, thin, T, Tb)
    elseif aI == 0.0 && alpha_p == 0.0                 # rotation without absorption
        return _segment_rotation(j, ρ[2], ρ[4], rho_p, ds, T, Tb)
    end
    # full absorption and rotation
    aQ = α[2]; aV = α[4]; rQ = ρ[2]; rV = ρ[4]
    ar = aQ * rQ + aV * rV
    asr = alpha_sq - rho_sq
    la = sqrt(asr * asr / 4.0 + ar * ar)
    lb = asr / 2.0
    l1 = sqrt(la + lb)
    l2 = sqrt(max(la - lb, zero(la)))
    ct = l1 * l1 + l2 * l2
    if ct <= 1.0e-12 * (alpha_sq + rho_sq)               # nilpotent N
        return _segment_nilpotent(j, aI, aQ, aV, rQ, rV, ds, T, Tb)
    end
    s = ar >= 0.0 ? 1.0 : -1.0
    a0 = aI * aI
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
    if abs(a0 - l1 * l1) <= POL_LANDI_GAP * (a0 + l1 * l1) ||
       a0 + l2 * l2 <= 1.0e-12 * (alpha_sq + rho_sq) || (!thin && (aI - l1) * ds <= dtau_max)
        return _segment_mode_integrals(j, aI, l1, l2, m2, m3, p4, m4m, ds, T, Tb)
    end
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

# φ(x) = (1 − e^{-x})/x, regular at x = 0.
@inline _phi1(x) = abs(x) < 1.0e-8 ? 1.0 - x / 2.0 + x * x / 6.0 : -expm1(-x) / x

# (C, S) = ∫₀^h e^{-a s} (cos b s, sin b s) ds = h φ(z), z = (a − i b) h, through
# 1 − e^{-z} = [2 sin²(y/2) − expm1(−x) cos y] − i e^{-x} sin y (x = a h, y = b h), and
# the Taylor series of φ for |z| < 1e-3.
@inline function _damped_trig_integrals(a, b, h)
    x = a * h; y = b * h
    d = x * x + y * y
    if d < 1.0e-6
        A = x * x - y * y; B = 2.0 * x * y
        re = 1.0 - x / 2.0 + A / 6.0 - (x * x * x - 3.0 * x * y * y) / 24.0 + (A * A - B * B) / 120.0
        im = y / 2.0 - x * y / 3.0 + (3.0 * x * x * y - y * y * y) / 24.0 - A * B / 60.0
        return h * re, h * im
    end
    s2 = sin(y / 2.0)
    nr = 2.0 * s2 * s2 - expm1(-x) * cos(y)
    ni = -exp(-x) * sin(y)
    return h * (nr * x - ni * y) / d, h * (ni * x + nr * y) / d
end

# The general solution with the eigenmode integrals C₁, S₁, C₂, S₂ evaluated through φ
# (exact at α_I = λ₁ and at α_I = λ₂ = 0, and at any depth) and the transmission formed
# from the decay factors of the two hyperbolic modes, e^{-α_I s} cosh λ₁s = (e^{-(α_I−λ₁)s} +
# e^{-(α_I+λ₁)s})/2, which stays finite at any depth.
@inline function _segment_mode_integrals(j, aI, l1, l2, m2, m3, p4, m4m, ds, ::Type{T},
                                         ::Type{Tb}) where {T,Tb}
    xm = (aI - l1) * ds; xp = (aI + l1) * ds
    em = exp(-xm); ep = exp(-xp)
    ech = (em + ep) / 2.0; esh = (em - ep) / 2.0
    ea = exp(-aI * ds)
    ecos = ea * cos(l2 * ds); esin = ea * sin(l2 * ds)
    oo = 0.5 .* ech .* p4 .+ 0.5 .* ecos .* m4m .- esin .* m2 .- esh .* m3
    fm = _phi1(xm); fp = _phi1(xp)
    c1 = ds * (fm + fp) / 2.0; s1 = ds * (fm - fp) / 2.0
    c2, s2 = _damped_trig_integrals(aI, l2, ds)
    pp = (0.5 * c1) .* p4 .+ (0.5 * c2) .* m4m .- s2 .* m2 .- s1 .* m3
    return SMatrix{4,4,T}(oo), SVector{4,Tb}(pp * j)
end

# Rotation without absorption: M rotates (Q, U, V) about ρ̂ = (ρ_Q, 0, ρ_V)/ρ_P through
# θ = ρ_P ds; b integrates the rotating emission, θ − sin θ by its series below 0.1.
@inline function _segment_rotation(j, rQ, rV, rho_p, ds, ::Type{T}, ::Type{Tb}) where {T,Tb}
    th = rho_p * ds
    cr = cos(th); sr = sin(th); s2 = sin(th / 2.0); ss = s2 * s2
    rho_sq = rho_p * rho_p
    M = @SMatrix [1.0 0.0 0.0 0.0;
                  0.0 (cr + 2.0 * rQ * rQ / rho_sq * ss) (-rV / rho_p * sr) (2.0 * rQ * rV / rho_sq * ss);
                  0.0 (rV / rho_p * sr) cr (-rQ / rho_p * sr);
                  0.0 (2.0 * rV * rQ / rho_sq * ss) (rQ / rho_p * sr) (cr + 2.0 * rV * rV / rho_sq * ss)]
    t2 = th * th
    tms = abs(th) < 0.1 ?
          th * t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))) : th - sr
    sc = sr / rho_p; cc = 2.0 * ss / rho_p; hm = tms / rho_p
    q = rQ / rho_p; v = rV / rho_p
    jQ = j[2]; jU = j[3]; jV = j[4]
    dt = q * jQ + v * jV
    b = SVector(j[1] * ds,
                sc * jQ + hm * q * dt - cc * v * jU,
                sc * jU + cc * (v * jQ - q * jV),
                sc * jV + hm * v * dt + cc * q * jU)
    return SMatrix{4,4,T}(M), SVector{4,Tb}(b)
end

# gₖ = ∫₀^h sᵏ/k! e^{-a s} ds for k = 0…3: the power series in x = a h for |x| < 1, else
# (1 − e^{-x} Σ_{m≤k} xᵐ/m!)/a^{k+1}.
@inline function _poly_exp_integrals(a, h)
    x = a * h
    if abs(x) < 1.0
        g0 = g1 = g2 = g3 = zero(x)
        t = one(x)                                   # (−x)^m / m!
        for m in 0:20
            g0 += t / (m + 1); g1 += t / (m + 2); g2 += t / (m + 3); g3 += t / (m + 4)
            t = -t * x / (m + 1)
        end
        return h * g0, h * h * g1, h * h * h * g2 / 2.0, h * h * h * h * g3 / 6.0
    end
    e = exp(-x)
    p1 = 1.0 + x; p2 = p1 + x * x / 2.0; p3 = p2 + x * x * x / 6.0
    return -expm1(-x) / a, (1.0 - e * p1) / (a * a), (1.0 - e * p2) / (a * a * a),
           (1.0 - e * p3) / (a * a * a * a)
end

# Nilpotent N (λ₁ = λ₂ = 0): the cubic exponential and its integral.
@inline function _segment_nilpotent(j, aI, aQ, aV, rQ, rV, ds, ::Type{T}, ::Type{Tb}) where {T,Tb}
    I4 = one(SMatrix{4,4,T})
    N = @SMatrix [0.0 aQ 0.0 aV; aQ 0.0 rV 0.0; 0.0 -rV 0.0 rQ; aV 0.0 -rQ 0.0]
    N2 = N * N; N3 = N2 * N
    oo = exp(-aI * ds) .* (I4 .- ds .* N .+ (ds * ds / 2.0) .* N2 .- (ds * ds * ds / 6.0) .* N3)
    g0, g1, g2, g3 = _poly_exp_integrals(aI, ds)
    pp = g0 .* I4 .- g1 .* N .+ g2 .* N2 .- g3 .* N3
    return SMatrix{4,4,T}(oo), SVector{4,Tb}(pp * j)
end

# Total absorption only: every Stokes component decays toward j/α_I; saturated above
# dtau_max.
@inline function _segment_total_absorption(j, aI, delta_tau, thin, ::Type{T},
                                           ::Type{Tb}) where {T,Tb}
    thin || return zero(SMatrix{4,4,T}), SVector{4,Tb}(j ./ aI)
    e = exp(-delta_tau)
    return SMatrix{4,4,T}(e .* one(SMatrix{4,4,T})),
           SVector{4,Tb}((j ./ aI) .* (-expm1(-delta_tau)))
end

"""
    stokes_admissible(S::SVector{4}) -> SVector{4}

The Stokes vector with `I` clamped to `I ≥ 0` and the polarization `(Q, U, V)`
rescaled onto the cone `|P| ≤ I` when outside it; the identity on admissible input.
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
    couple(::IQUV, S, j, α, ρ, ds; dtau_max = POL_DELTA_TAU_MAX) -> SVector{4}
    couple(::StokesI, I, j, α, ds; dtau_max = POL_DELTA_TAU_MAX)

Advance the Stokes vector `S` (or the intensity `I`) over one segment, source to
camera: `stokes_admissible(M S + b)` with `(M, b) = segment(IQUV(), j, α, ρ, ds)`, or
`A I + B` with `(A, B) = segment(StokesI(), j, α, ds)`.
"""
@inline function couple(::IQUV, S::SVector{4}, j::SVector{4}, α::SVector{4}, ρ::SVector{4}, ds;
                        dtau_max = POL_DELTA_TAU_MAX)
    M, b = segment(IQUV(), j, α, ρ, ds; dtau_max = dtau_max)
    return stokes_admissible(M * S + b)
end

@inline function couple(::StokesI, intensity, j, α, ds; dtau_max = POL_DELTA_TAU_MAX)
    A, B = segment(StokesI(), j, α, ds; dtau_max = dtau_max)
    return A * intensity + B
end
