# Vacuum (Ricci) check of the boosted, arbitrarily-spinning single-hole metric as coded in
# smressle-blacklight_binaries geodesic_geometry.cpp (get_prime_coords, GetBoyerLindquist
# Coordinates, boosted_BH_metric_addition), transcribed line by line, with s(t) = s0 + v t.
# A uniformly moving Kerr hole is exact, so R_μν must vanish if the formulas are right.
using ForwardDiff, StaticArrays, LinearAlgebra, Printf

const SMALL = 1e-15

# --- transcription of the fork -------------------------------------------------------
# get_prime_coords: spatial Lorentz contraction of x - s(t) (the t̄ = 0 approximation)
function prime_coords(x, y, z, sbh, v)
    vsq = dot(v, v); β = sqrt(vsq); Γ = sqrt(1 / (1 - vsq))
    n = v ./ (β + 1e-15)
    d = SVector(x, y, z) - sbh
    xp = (1 + (Γ - 1) * n[1] * n[1]) * d[1] + (Γ - 1) * n[1] * n[2] * d[2] + (Γ - 1) * n[1] * n[3] * d[3]
    yp = (Γ - 1) * n[2] * n[1] * d[1] + (1 + (Γ - 1) * n[2] * n[2]) * d[2] + (Γ - 1) * n[2] * n[3] * d[3]
    zp = (Γ - 1) * n[3] * n[1] * d[1] + (Γ - 1) * n[3] * n[2] * d[2] + (1 + (Γ - 1) * n[3] * n[3]) * d[3]
    return SVector(xp, yp, zp)
end

function bl_coords(xp, a)
    amag = norm(a)
    adx = dot(a, xp)
    acx = cross(a, xp)
    R = norm(xp)
    r = sqrt(R^2 - amag^2 + sqrt((R^2 - amag^2)^2 + 4adx^2)) / sqrt(2)
    rsq_p_asq = r^2 + amag^2
    l = (r .* xp .- acx .+ adx .* a ./ r) ./ rsq_p_asq
    return r, acos(clamp(l[3], -1, 1)), atan(l[2], l[1])
end

sph_to_cart(r, th, ph, a) = SVector(
    r * sin(th) * cos(ph) + a[2] * cos(th) - a[3] * sin(th) * sin(ph),
    r * sin(th) * sin(ph) + a[3] * sin(th) * cos(ph) - a[1] * cos(th),
    r * cos(th) + a[1] * sin(th) * sin(ph) - a[2] * sin(th) * cos(ph))

# boosted_BH_metric_addition: returns the 4x4 perturbation f l_μ l_ν (lab frame)
function boosted_term(M, xp, v, a; clamp_h = true)
    amag = norm(a)
    adx = dot(a, xp)
    r, th, ph = bl_coords(xp, a)
    if clamp_h
        rh = M + sqrt(M^2 - amag^2)
        if r < 0.8rh
            r = 0.8rh
            xp = sph_to_cart(r, th, ph, a)
        end
    end
    adx = dot(a, xp)
    acx = cross(a, xp)
    rsq_p_asq = r^2 + amag^2
    f = M * 2 * r^3 / (r^4 + adx^2)
    lp = SVector(one(r), ((r .* xp .- acx .+ adx .* a ./ r) ./ rsq_p_asq)...)   # l'_μ, l'_0 = +1
    vsq = dot(v, v); β = sqrt(vsq); Γ = sqrt(1 / (1 - vsq))
    n = v ./ (β + 1e-15)
    Λ = @SMatrix [Γ          -Γ*v[1]                 -Γ*v[2]                 -Γ*v[3];
                  -Γ*v[1]    1+(Γ-1)*n[1]*n[1]        (Γ-1)*n[1]*n[2]         (Γ-1)*n[1]*n[3];
                  -Γ*v[2]    (Γ-1)*n[2]*n[1]          1+(Γ-1)*n[2]*n[2]       (Γ-1)*n[2]*n[3];
                  -Γ*v[3]    (Γ-1)*n[3]*n[1]          (Γ-1)*n[3]*n[2]         1+(Γ-1)*n[3]*n[3]]
    l = Λ * lp
    return f * (l * l')
end

const η = @SMatrix [-1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]

# full metric at event X = (t, x, y, z); hole at s0 + v t
function g_fork(X, M, s0, v, a; boost = true)
    t = X[1]
    sbh = s0 + v * t
    xp = boost ? prime_coords(X[2], X[3], X[4], sbh, v) : SVector(X[2], X[3], X[4]) - sbh
    return η + boosted_term(M, xp, boost ? v : zero(v), a)
end

# --- curvature by nested AD ---------------------------------------------------------
function christoffel(gfun, X)
    g = gfun(X)
    gi = inv(g)
    J = ForwardDiff.jacobian(gfun, X)            # (16, 4): ∂_c g_{ab} at column c
    dg = reshape(J, 4, 4, 4)
    Γ = zeros(eltype(g), 4, 4, 4)
    for μ in 1:4, α in 1:4, β in 1:4
        s = zero(eltype(g))
        for λ in 1:4
            s += gi[μ, λ] * (dg[λ, β, α] + dg[λ, α, β] - dg[α, β, λ])
        end
        Γ[μ, α, β] = s / 2
    end
    return Γ
end

function ricci(gfun, X)
    Γ = christoffel(gfun, X)
    JΓ = ForwardDiff.jacobian(Y -> vec(christoffel(gfun, Y)), X)   # (64, 4)
    dΓ = reshape(JΓ, 4, 4, 4, 4)                                    # dΓ[μ,α,β,c] = ∂_c Γ^μ_{αβ}
    R = zeros(4, 4)
    for μ in 1:4, ν in 1:4
        s = 0.0
        for α in 1:4
            s += dΓ[α, μ, ν, α] - dΓ[α, μ, α, ν]
            for β in 1:4
                s += Γ[α, α, β] * Γ[β, μ, ν] - Γ[α, ν, β] * Γ[β, μ, α]
            end
        end
        R[μ, ν] = s
    end
    return R
end

# --- cases ---------------------------------------------------------------------------
M = 1.0
a = SVector(0.3, -0.5, 0.6)          # |a| = 0.837, arbitrary direction
v = SVector(0.25, 0.15, -0.1)        # |v| = 0.31
s0 = SVector(4.0, -2.0, 1.0)
pts = [SVector(0.7, 8.0, 1.0, 2.5), SVector(-1.3, 2.0, -5.0, 3.0), SVector(2.0, 7.0, 0.5, -0.2),
       SVector(0.0, 1.0, 2.0, 6.0)]

println("Fork formulas, uniformly boosted spinning hole (exact Kerr expected):")
for X in pts
    xp = prime_coords(X[2], X[3], X[4], s0 + v * X[1], v)
    r = bl_coords(xp, a)[1]
    g = g_fork(X, M, s0, v, a)
    @printf("  r'=%6.3f  max|R_μν| = %.2e   det g = %.6f  (η: -1)\n", r, maximum(abs.(ricci(Y -> g_fork(Y, M, s0, v, a), X))), det(g))
end

println("Control 1: same hole, no boost of l but moving centre (Galilean, wrong):")
for X in pts[1:2]
    xp = SVector(X[2], X[3], X[4]) - (s0 + v * X[1])
    @printf("  r'=%6.3f  max|R_μν| = %.2e\n", bl_coords(xp, a)[1], maximum(abs.(ricci(Y -> g_fork(Y, M, s0, v, a; boost = false), X))))
end

println("Control 2: boost with the opposite velocity sign in Λ (wrong sign convention):")
g_wrong(X) = (sbh = s0 + v * X[1]; xp = prime_coords(X[2], X[3], X[4], sbh, v); η + boosted_term(M, xp, -v, a))
for X in pts[1:2]
    @printf("  max|R_μν| = %.2e\n", maximum(abs.(ricci(g_wrong, X))))
end

println("Control 3: static spinning hole with arbitrary spin direction (v = 0):")
for X in pts[1:2]
    @printf("  max|R_μν| = %.2e\n", maximum(abs.(ricci(Y -> g_fork(Y, M, s0, zero(v), a), X))))
end

println("Round trip x' -> (r,θ,φ) -> x' (convert_spherical_to_cartesian_ks):")
for xp in (SVector(3.0, 1.0, -2.0), SVector(-1.0, 4.0, 0.5), SVector(0.2, -0.3, 2.0))
    r, th, ph = bl_coords(xp, a)
    @printf("  max|Δx'| = %.2e\n", maximum(abs.(sph_to_cart(r, th, ph, a) - xp)))
end

println("Horizon clamp: metric continuity across r' = 0.8 r_h (jump of the frozen interior):")
let rh = M + sqrt(M^2 - dot(a, a)), nhat = normalize(SVector(0.3, 0.9, 0.4))
    for s in (0.81, 0.80, 0.79, 0.5)
        xp = (s * rh) .* nhat
        g = η + boosted_term(M, xp, v, a)
        @printf("  r'/r_h = %.2f  g_tt = %.5f  det = %.4f\n", s, g[1, 1], det(g))
    end
end
