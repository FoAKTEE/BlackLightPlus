# Fluid orthonormal tetrad, following blacklight radiation/radiation_geometry.hpp
# (Tetrad). Convention (rows e_(a)^mu, contravariant):
#   e_(0) = u                                       (aligned with the fluid velocity)
#   e_(3) = k / omega - u   (omega = -k_mu u^mu)    (photon momentum orthogonal to u)
#   e_(2) = `up` projected orthogonal to e_(0), e_(3), then normalized with gcov
#   e_(1) = metric cross product of e_(0), e_(2), e_(3), raised with gcon
# Assumes |det g| = 1 (exact for single CKS); any other metric (provider /
# superposed / interpolated) needs `renorm_e1 = true` to renormalize e_(1) with
# gcov afterward.

"""
    fluid_tetrad(ucon, ucov, kcon, kcov, up_con, gcov, gcon; renorm_e1=false) -> SMatrix{4,4}

Orthonormal tetrad aligned with the fluid 4-velocity. Returns the 4x4 matrix whose
row `a` holds the contravariant components `e_(a)^mu`. `up_con` is the spacelike
up-direction (e.g. the magnetic field `b^mu`). With `renorm_e1 = true` the spatial
basis vector `e_(1)` is renormalized against `gcov` (needed off single CKS, where
|det g| != 1).
"""
@inline function fluid_tetrad(ucon::SVector{4}, ucov::SVector{4},
                              kcon::SVector{4}, kcov::SVector{4},
                              up_con::SVector{4}, gcov::SMatrix{4,4},
                              gcon::SMatrix{4,4}; renorm_e1::Bool = false)
    omega = -dot(kcov, ucon)                         # omega = -k_mu u^mu
    k_up_over_omega = dot(kcov, up_con) / omega
    u_up_over_omega = dot(ucov, up_con) / omega

    e0 = ucon
    e3 = kcon ./ omega .- ucon
    e2u = up_con .- k_up_over_omega .* e3 .+ u_up_over_omega .* kcon
    e2u_n2 = dot(e2u, gcov * e2u)
    # Deviation from blacklight: when up_con lies in span(u, k) — e.g. B ∥ k in the
    # fluid frame — e2u vanishes and `e2u / sqrt(0)` NaN-poisons the whole ray. The
    # pitch angle is then sin(theta_B) = 0, so the radiation is frame-independent;
    # substitute a finite spacelike e_(2) by projecting the coordinate axis most
    # orthogonal to e3 onto span(u,k)^⊥ (ipole set_Econ_from_trial fallback).
    if !(e2u_n2 > 1.0e-12 * dot(up_con, gcov * up_con))
        s2 = e3[2] * e3[2]; s3 = e3[3] * e3[3]; s4 = e3[4] * e3[4]
        trial = (s2 <= s3 && s2 <= s4) ? SVector(0.0, 1.0, 0.0, 0.0) :
                (s3 <= s4)             ? SVector(0.0, 0.0, 1.0, 0.0) :
                                         SVector(0.0, 0.0, 0.0, 1.0)
        e2u = trial .- (dot(kcov, trial) / omega) .* e3 .+ (dot(ucov, trial) / omega) .* kcon
        e2u_n2 = dot(e2u, gcov * e2u)
    end
    e2 = e2u ./ sqrt(e2u_n2)                          # normalize e_(2) with gcov

    # e_(1) covariant = metric cross product of e_(0), e_(2), e_(3).
    e1c1 = e0[2] * (e2[4] * e3[3] - e2[3] * e3[4]) + e0[3] * (e2[2] * e3[4] - e2[4] * e3[2]) +
           e0[4] * (e2[3] * e3[2] - e2[2] * e3[3])
    e1c2 = e0[1] * (e2[3] * e3[4] - e2[4] * e3[3]) + e0[3] * (e2[4] * e3[1] - e2[1] * e3[4]) +
           e0[4] * (e2[1] * e3[3] - e2[3] * e3[1])
    e1c3 = e0[1] * (e2[4] * e3[2] - e2[2] * e3[4]) + e0[2] * (e2[1] * e3[4] - e2[4] * e3[1]) +
           e0[4] * (e2[2] * e3[1] - e2[1] * e3[2])
    e1c4 = e0[1] * (e2[2] * e3[3] - e2[3] * e3[2]) + e0[2] * (e2[3] * e3[1] - e2[1] * e3[3]) +
           e0[3] * (e2[1] * e3[2] - e2[2] * e3[1])
    e1_cov = SVector(e1c1, e1c2, e1c3, e1c4)
    e1 = gcon * e1_cov                               # raise with gcon

    if renorm_e1
        e1_norm = sqrt(dot(e1, gcov * e1))
        if e1_norm > 0.0
            e1 = e1 ./ e1_norm
        end
    end

    return @inbounds SMatrix{4,4}(
        e0[1], e1[1], e2[1], e3[1],
        e0[2], e1[2], e2[2], e3[2],
        e0[3], e1[3], e2[3], e3[3],
        e0[4], e1[4], e2[4], e3[4])
end
