# BLP two-hole SuperposedKerrSchild: inverse and AD gradient with two massive holes, and the
# size of the omitted boost at the near-horizon radii of an equal-mass sep = 20 binary.
using BLP, StaticArrays, LinearAlgebra, ForwardDiff, Printf
using BLP.Spacetime: SuperposedKerrSchild, metric, inverse_metric, inverse_metric_gradient,
                     metric_and_inverse, ks_radius, raise_and_norm, lower_index, raise_index

sep = 20.0; q = 1.0
m1 = 1 / (1 + q); m2 = q / (1 + q)
c1 = SVector(q / (1 + q) * sep, 0.0, 0.0); c2 = SVector(-sep / (1 + q), 0.0, 0.0)
st = SuperposedKerrSchild([m1, m2], [0.3 * m1, -0.5 * m2], [c1, c2])

println("Two massive holes: Woodbury inverse, index ops and AD gradient")
worst_inv = 0.0; worst_grad = 0.0; worst_ops = 0.0
for xyz in ((c1 + SVector(1.5, 1.0, 0.7))..., (c2 + SVector(-0.8, 1.6, -0.5))..., 0.0, 0.0, 3.0, 2.0, 1.0, -1.0)
end
pts = [SVector(0.0, (c1 + SVector(1.5, 1.0, 0.7))...), SVector(0.0, (c2 + SVector(-0.8, 1.6, -0.5))...),
       SVector(0.0, 0.0, 3.0, 2.0), SVector(0.0, 1.0, -1.0, 0.5)]
for x in pts
    g, gi = metric_and_inverse(st, x)
    global worst_inv = max(worst_inv, maximum(abs.(g * gi - I)))
    @assert maximum(abs.(gi - inverse_metric(st, x))) < 1e-13
    p = SVector(1.0, 0.3, -0.2, 0.1)
    gp, n = raise_and_norm(st, x, p)
    global worst_ops = max(worst_ops, maximum(abs.(gp - gi * p)), abs(n - dot(p, gi * p)),
                           maximum(abs.(lower_index(st, x, p) - g * p)), maximum(abs.(raise_index(st, x, p) - gi * p)))
    dg = inverse_metric_gradient(st, x)
    h = 1e-6
    for c in 2:4
        dx = SVector{4}(ntuple(i -> i == c ? h : 0.0, 4))
        fd = (inverse_metric(st, x + dx) - inverse_metric(st, x - dx)) / (2h)
        global worst_grad = max(worst_grad, maximum(abs.(SMatrix{4,4}(dg[:, :, c]) - fd)))
    end
end
@printf("  max|g g⁻¹ - I| = %.1e   max index-op mismatch = %.1e   max|AD - FD| gradient = %.1e\n", worst_inv, worst_ops, worst_grad)

# ---- size of the omitted boost -------------------------------------------------------
# Hole A moving with v_A: the SKS term is f (Λl)(Λl) instead of f l l. Compare at points
# at KS radius r' = 3, 2 (near the hole) for the orbital speed of an equal-mass sep = 20
# binary, v = (1/2)·sep^{-1/2} ≈ 0.112, and for sep = 10 (v = 0.158).
function boosted_ll(f, l, v)
    vsq = dot(v, v); β = sqrt(vsq); Γ = 1 / sqrt(1 - vsq); n = v / β
    Λ = @SMatrix [Γ -Γ*v[1] -Γ*v[2] -Γ*v[3];
                  -Γ*v[1] 1+(Γ-1)*n[1]^2 (Γ-1)*n[1]*n[2] (Γ-1)*n[1]*n[3];
                  -Γ*v[2] (Γ-1)*n[1]*n[2] 1+(Γ-1)*n[2]^2 (Γ-1)*n[2]*n[3];
                  -Γ*v[3] (Γ-1)*n[1]*n[3] (Γ-1)*n[2]*n[3] 1+(Γ-1)*n[3]^2]
    lb = Λ * l
    return f * (lb * lb')
end
println("Omitted boost of one hole (mass 1/2), metric perturbation h = g - η at the hole's own radius r':")
for (sepx, lab) in ((20.0, "sep 20"), (10.0, "sep 10"))
    v = SVector(0.0, 0.5 / sqrt(sepx), 0.0)        # hole on +x moving along +y
    for rr in (3.0, 2.0)
        maxrel = 0.0; maxgtt = 0.0
        for dir in (SVector(1.0, 0, 0), SVector(0, 1.0, 0), SVector(0, 0, 1.0), normalize(SVector(1.0, 1.0, 0.3)), normalize(SVector(-1.0, 0.5, -0.2)))
            X = rr * dir     # Schwarzschild-like hole, mass 1/2: f = 2 M r³/(r⁴) = 1/r
            f = 2 * 0.5 / rr
            l = SVector(1.0, dir...)
            hs = f * (l * l'); hb = boosted_ll(f, l, v)
            maxrel = max(maxrel, maximum(abs.(hb - hs)) / maximum(abs.(hs)))
            maxgtt = max(maxgtt, abs(hb[1, 1] - hs[1, 1]) / abs(hs[1, 1]))
        end
        @printf("  %s (v = %.3f), r' = %.0f: max|Δh|/max|h| = %.2f, |Δg_tt|/|h_tt| = %.2f\n", lab, norm(v), rr, maxrel, maxgtt)
    end
end
