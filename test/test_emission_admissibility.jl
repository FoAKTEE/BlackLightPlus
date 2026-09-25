# Admissible polarized emission. The thermal fits (Marszewski+2021) give |j_P| > j_I at
# small pitch angles (a nearly field-aligned ray: sin θ_B = 0.02 below), an emission
# vector outside the Stokes cone. The segment operator must emit an admissible vector,
# so that the observed Stokes of a ray does not depend on the order in which segments are
# accumulated: the reverse (camera -> source) Mueller chain of the marches and the
# forward (source -> camera) coupling must agree whether the field-aligned emitter lies
# in front of or behind brighter gas.
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus.Radiation: stokes_segment, stokes_couple, stokes_admissible,
                                thermal_synchrotron_polarized

# invariant coefficients of one thermal sample (BLP basis; U components zero)
function coeffs(n_e, theta_e, B, nu, sin_th)
    c = thermal_synchrotron_polarized(1.0, n_e, theta_e, B, nu, sin_th, sqrt(1 - sin_th^2))
    return SVector(c[1], c[2], 0.0, c[3]), SVector(c[4], c[5], 0.0, c[6]),
           SVector(0.0, c[7], 0.0, c[8])
end

# Forward: source -> camera with the per-segment clamp.
forward(segs) = foldl((S, s) -> stokes_couple(S, s...), segs; init = SVector(0.0, 0.0, 0.0, 0.0))

# Reverse: camera -> source, S += G b, G <- G M, clamp after every addition
# (reverse_polarized_march with the fluid and camera frames aligned).
function reverse(segs)
    S = SVector(0.0, 0.0, 0.0, 0.0); G = one(SMatrix{4,4,Float64})
    for s in Iterators.reverse(segs)
        M, b = stokes_segment(s...)
        S = stokes_admissible(S + G * b)
        G = G * M
    end
    return S
end

@testset "thermal emission at small pitch angle is admissible" begin
    j, α, ρ = coeffs(1.2e-5, 0.918, 100.0, 1.0e9, 0.02)
    @test hypot(j[2], j[4]) > j[1]                   # the fit itself leaves the cone
    for ds in (4.8e5, 5.8e6, 4.8e7)
        M, b = stokes_segment(j, α, ρ, ds)
        @test norm(b[2:4]) <= b[1] * (1 + 1e-12)     # emitted Stokes vector is admissible
    end
end

@testset "observed Stokes independent of the accumulation order" begin
    screen = coeffs(1.2e-5, 0.918, 100.0, 1.0e9, 0.02)        # field-aligned, |j_P| > j_I
    slab = coeffs(6.0e-8, 0.918, 100.0, 1.0e9, 1.0)           # brighter, perpendicular field
    ds_screen = 4.8e7 / 8; ds_slab = 1.9e8 / 8
    for order in (:screen_in_front, :screen_behind)
        # segments in source -> camera order
        segs = order == :screen_in_front ?
               vcat(fill((slab..., ds_slab), 8), fill((screen..., ds_screen), 8)) :
               vcat(fill((screen..., ds_screen), 8), fill((slab..., ds_slab), 8))
        Sf = forward(segs); Sr = reverse(segs)
        @test Sr ≈ Sf rtol = 1e-6
        @test norm(Sf[2:4]) <= Sf[1] * (1 + 1e-12)
    end
end
