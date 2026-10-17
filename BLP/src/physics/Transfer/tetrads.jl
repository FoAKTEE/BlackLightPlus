# Orthonormal frames of the polarized transfer (blacklight radiation_geometry.hpp
# `Tetrad`). Rows of a tetrad are the contravariant legs e_(a)^μ:
#   e_(0) = u,  e_(3) = k/ω - u (ω = -k·u),
#   e_(2) = `up` projected orthogonal to e_(0), e_(3) and normalised with g_μν,
#   e_(1) = the metric cross product of e_(0), e_(2), e_(3), raised with g^μν,
# which is unit where |det g| = 1 (single Kerr-Schild) and is renormalised elsewhere.

"""
    fluid_tetrad(ucon, ucov, kcon, kcov, up_con, gcov, gcon; renorm_e1 = false) -> SMatrix{4,4}

Orthonormal tetrad of the observer `u` (both index positions) with `e_(3)` along the
photon wave vector `k` and `e_(2)` along the spacelike direction `up_con` (the
magnetic field `b^μ` for the fluid frame, the image vertical for the camera
frame); row `a` holds `e_(a)^μ`. `renorm_e1 = true` renormalises `e_(1)` with
`gcov`, required wherever `|det g| ≠ 1`. See [`fluid_tetrad_checked`](@ref).
"""
@inline fluid_tetrad(ucon::SVector{4}, ucov::SVector{4}, kcon::SVector{4},
                     kcov::SVector{4}, up_con::SVector{4}, gcov::SMatrix{4,4},
                     gcon::SMatrix{4,4}; renorm_e1::Bool = false) =
    first(fluid_tetrad_checked(ucon, ucov, kcon, kcov, up_con, gcov, gcon; renorm_e1))

"""
    fluid_tetrad_checked(ucon, ucov, kcon, kcov, up_con, gcov, gcon; renorm_e1 = false)
        -> (tet::SMatrix{4,4}, ok::Bool)

[`fluid_tetrad`](@ref) with a validity flag. When `up_con` lies in span(u, k) to
working precision (squared norm of its projection at most `1e-12` of `up·up`), the
coordinate axis least aligned with `e_(3)` is projected instead (ipole
`set_Econ_from_trial`); the pitch angle is then zero and the frame choice does not
affect the coefficients. `ok` is false when no spacelike `e_(2)` exists (u not unit
timelike or k not null to working precision); `e_(2)` is then zero, never NaN.
"""
@inline function fluid_tetrad_checked(ucon::SVector{4}, ucov::SVector{4},
                                      kcon::SVector{4}, kcov::SVector{4},
                                      up_con::SVector{4}, gcov::SMatrix{4,4},
                                      gcon::SMatrix{4,4}; renorm_e1::Bool = false)
    omega = -dot(kcov, ucon)
    k_up_over_omega = dot(kcov, up_con) / omega
    u_up_over_omega = dot(ucov, up_con) / omega

    e0 = ucon
    e3 = kcon ./ omega .- ucon
    e2u = up_con .- k_up_over_omega .* e3 .+ u_up_over_omega .* kcon
    e2u_n2 = dot(e2u, gcov * e2u)
    if !(e2u_n2 > 1.0e-12 * dot(up_con, gcov * up_con))
        s2 = e3[2] * e3[2]; s3 = e3[3] * e3[3]; s4 = e3[4] * e3[4]
        trial = (s2 <= s3 && s2 <= s4) ? SVector(0.0, 1.0, 0.0, 0.0) :
                (s3 <= s4)             ? SVector(0.0, 0.0, 1.0, 0.0) :
                                         SVector(0.0, 0.0, 0.0, 1.0)
        e2u = trial .- (dot(kcov, trial) / omega) .* e3 .+ (dot(ucov, trial) / omega) .* kcon
        e2u_n2 = dot(e2u, gcov * e2u)
    end
    ok = e2u_n2 > 0.0
    e2 = ok ? e2u ./ sqrt(e2u_n2) : zero(e2u)

    # e_(1) covariant: the metric cross product of e_(0), e_(2), e_(3)
    e1c1 = e0[2] * (e2[4] * e3[3] - e2[3] * e3[4]) + e0[3] * (e2[2] * e3[4] - e2[4] * e3[2]) +
           e0[4] * (e2[3] * e3[2] - e2[2] * e3[3])
    e1c2 = e0[1] * (e2[3] * e3[4] - e2[4] * e3[3]) + e0[3] * (e2[4] * e3[1] - e2[1] * e3[4]) +
           e0[4] * (e2[1] * e3[3] - e2[3] * e3[1])
    e1c3 = e0[1] * (e2[4] * e3[2] - e2[2] * e3[4]) + e0[2] * (e2[1] * e3[4] - e2[4] * e3[1]) +
           e0[4] * (e2[2] * e3[1] - e2[1] * e3[2])
    e1c4 = e0[1] * (e2[2] * e3[3] - e2[3] * e3[2]) + e0[2] * (e2[3] * e3[1] - e2[1] * e3[3]) +
           e0[3] * (e2[1] * e3[2] - e2[2] * e3[1])
    e1_cov = SVector(e1c1, e1c2, e1c3, e1c4)
    e1 = gcon * e1_cov

    if renorm_e1
        e1_norm = sqrt(dot(e1, gcov * e1))
        if e1_norm > 0.0
            e1 = e1 ./ e1_norm
        end
    end

    return (@inbounds SMatrix{4,4}(
        e0[1], e1[1], e2[1], e3[1],
        e0[2], e1[2], e2[2], e3[2],
        e0[3], e1[3], e2[3], e3[3],
        e0[4], e1[4], e2[4], e3[4])), ok
end

"""
    normal_observer(gcon::SMatrix{4,4}) -> SVector{4}

Four-velocity `n^μ = -α g^{0μ}` of the normal (Eulerian) observer, `α = 1/√(-g^{00})`
the lapse: `n·n = -1`, spatial part `-β^i/α` with shift `β^i = -g^{0i}/g^{00}`. The
camera frame of the polarized march is built on it.
"""
@inline function normal_observer(gcon::SMatrix{4,4})
    lapse = 1.0 / sqrt(-gcon[1, 1])
    return SVector(1.0 / lapse, gcon[1, 2] / gcon[1, 1] / lapse,
                   gcon[1, 3] / gcon[1, 1] / lapse, gcon[1, 4] / gcon[1, 1] / lapse)
end

"""
    qu_rotation(tet_fl, E, gcov) -> (cos 2χ, sin 2χ)

The Stokes rotation from the fluid tetrad `tet_fl` to the readout tetrad `E` sharing
its `e_(0)`–`e_(3)` plane: χ is the angle of `E`'s `e_(1)` in the fluid
`(e_(1), e_(2))` basis, and the Stokes vector transforms by a rotation of `(Q, U)`
through 2χ with I and V fixed.
"""
@inline function qu_rotation(tet_fl::SMatrix{4,4}, E::SMatrix{4,4}, gcov::SMatrix{4,4})
    e1f = SVector(tet_fl[2,1], tet_fl[2,2], tet_fl[2,3], tet_fl[2,4])
    e2f = SVector(tet_fl[3,1], tet_fl[3,2], tet_fl[3,3], tet_fl[3,4])
    E1  = SVector(E[2,1], E[2,2], E[2,3], E[2,4])
    c = dot(E1, gcov * e1f); s = dot(E1, gcov * e2f)
    nrm = sqrt(c*c + s*s); c /= nrm; s /= nrm
    return c*c - s*s, 2.0*s*c
end

"""
    rotate_qu(c, s, b::SVector{4}) -> R b
    rotate_qu(c, s, M::SMatrix{4,4}) -> R M Rᵀ

The Stokes rotation `R = [1 0 0 0; 0 c s 0; 0 -s c 0; 0 0 0 1]` (`c = cos 2χ`,
`s = sin 2χ` from [`qu_rotation`](@ref)) applied to a Stokes vector or conjugating
a Mueller matrix; only the Q and U rows and columns mix.
"""
@inline rotate_qu(c, s, b::SVector{4}) =
    SVector(b[1], c * b[2] + s * b[3], c * b[3] - s * b[2], b[4])
@inline function rotate_qu(c, s, M::SMatrix{4,4})
    A = SMatrix{4,4}(M[1,1], c*M[2,1] + s*M[3,1], c*M[3,1] - s*M[2,1], M[4,1],
                     M[1,2], c*M[2,2] + s*M[3,2], c*M[3,2] - s*M[2,2], M[4,2],
                     M[1,3], c*M[2,3] + s*M[3,3], c*M[3,3] - s*M[2,3], M[4,3],
                     M[1,4], c*M[2,4] + s*M[3,4], c*M[3,4] - s*M[2,4], M[4,4])   # R M
    return SMatrix{4,4}(A[1,1], A[2,1], A[3,1], A[4,1],
                        c*A[1,2] + s*A[1,3], c*A[2,2] + s*A[2,3], c*A[3,2] + s*A[3,3], c*A[4,2] + s*A[4,3],
                        c*A[1,3] - s*A[1,2], c*A[2,3] - s*A[2,2], c*A[3,3] - s*A[3,2], c*A[4,3] - s*A[4,2],
                        A[1,4], A[2,4], A[3,4], A[4,4])                        # (R M) Rᵀ
end
