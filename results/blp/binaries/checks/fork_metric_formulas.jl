using ForwardDiff, StaticArrays, LinearAlgebra
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
