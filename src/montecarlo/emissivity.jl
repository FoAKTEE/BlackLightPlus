# Monte Carlo — emissivity.jl
#
# Emission-frame physics for superphoton generation: the non-invariant thermal-
# synchrotron emissivity j_nu (cgs), its pitch-angle envelope and zone-
# integrated rate, the matching absorptivity, and the (u, b-hat)-aligned
# emission tetrad. Reuses `Radiation.thermal_synchrotron` (Leung et al. 2011
# form), undoing the invariant normalisation: j_nu = j_inv nu^2, alpha_nu =
# alpha_inv / nu. The tetrad has e_(0) = u, e_(1) = b-hat — the frame for
# emitting at a prescribed pitch angle to B (gpumonty tetrads.cu make_tetrad).

# ============================================================ emissivity

"""
    j_nu(n_e, theta_e, B, nu, sin_theta) -> Float64

Non-invariant thermal-synchrotron emissivity j_nu [erg s^-1 cm^-3 Hz^-1 sr^-1]
in cgs, from `Radiation.thermal_synchrotron` (`j_inv = j_nu/nu^2`) times `nu^2`.
`sin_theta` is the pitch angle between the photon and the magnetic field.
"""
@inline function j_nu(n_e, theta_e, B, nu, sin_theta)
    j_inv, _ = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_theta)
    return j_inv * nu^2
end

"""
    alpha_nu(n_e, theta_e, B, nu, sin_theta) -> Float64

Non-invariant thermal-synchrotron absorptivity alpha_nu [cm^-1] from Kirchhoff's
law, via `Radiation.thermal_synchrotron` (`alpha_inv = alpha_nu*nu`), so
`alpha_nu = alpha_inv / nu`.
"""
@inline function alpha_nu(n_e, theta_e, B, nu, sin_theta)
    _, alpha_inv = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_theta)
    return alpha_inv / nu
end

"""
    jmax(n_e, theta_e, B, nu) -> Float64

Rejection-sampling envelope for the pitch-angle distribution: `j_nu` at
`sin_theta = 1`, which maximises the thermal-synchrotron emissivity over pitch
angle for nu >~ nu_s. The direction is accepted with probability
`j_nu(theta)/j_nu(pi/2)` (as in grmonty/gpumonty).
"""
@inline jmax(n_e, theta_e, B, nu) = j_nu(n_e, theta_e, B, nu, 1.0)

# Fixed Simpson node count for the pitch-angle integral. j_nu(sin theta) * sin theta
# vanishes (with all derivatives) at the endpoints and is smooth, so a modest
# composite Simpson rule is far below the 1e-6 tolerance the generator needs.
const _N_ZONE_THETA = 512

"""
    int_jnu_zone(n_e, theta_e, B, nu) -> Float64

Pitch-angle-integrated emissivity `∫ j_nu dΩ` [erg s^-1 cm^-3 Hz^-1] — the
per-zone emission rate at frequency `nu`. Since `j_nu` depends on direction only
through `sin(theta)` and is symmetric about the equator,

    ∫ j_nu dΩ = 4 pi ∫_0^{pi/2} j_nu(sin theta) sin theta d theta,

evaluated with a fixed composite Simpson rule (device-safe: constant loop
bound, no allocation); validated against an independent fine quadrature to < 1e-6.
"""
@inline function int_jnu_zone(n_e, theta_e, B, nu)
    a = 0.0
    b = 0.5 * π
    h = (b - a) / _N_ZONE_THETA
    acc = 0.0
    @inbounds for i in 0:_N_ZONE_THETA
        th = a + i * h
        s = sin(th)
        w = (i == 0 || i == _N_ZONE_THETA) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        acc += w * j_nu(n_e, theta_e, B, nu, s) * s
    end
    return 4.0 * π * acc * h / 3.0
end

# ============================================================ emission tetrad

# Below this metric-norm a trial vector is treated as degenerate (gpumonty
# tetrads.cu `SMALL_VECTOR`); the emission tetrad then falls back to x1.
const SMALL_VECTOR = 1.0e-30

"""
    emission_tetrad(ucon, ucov, bcon, gcov, gcon) -> SMatrix{4,4}

(u, b-hat)-aligned orthonormal emission tetrad; row `a` holds `e_(a)^mu`
(the layout of `Radiation.fluid_tetrad`): e_(0) = u_hat, e_(1) = b_hat (sets
the pitch angle), e_(2)/e_(3) completed by Gram–Schmidt on x2, x3 with the
metric inner product (after gpumonty tetrads.cu `make_tetrad`). `ucov` is
accepted for signature parity with the imaging tetrad. A numerically null
`bcon` falls back to the x1 direction.
"""
@inline function emission_tetrad(ucon::SVector{4}, ucov::SVector{4},
                                 bcon::SVector{4}, gcov::SMatrix{4,4},
                                 gcon::SMatrix{4,4})
    # e_(0) = u / sqrt(|u.u|)  (a proper 4-velocity has u.u = -1, so this is a no-op)
    n0 = dot(ucon, gcov * ucon)
    e0 = ucon ./ sqrt(abs(n0))

    # e_(1): b-hat, projected orthogonal to e_(0), then normalised. Guard a null
    # trial vector by defaulting to the x1 coordinate direction (reference parity).
    trial = dot(bcon, gcov * bcon) <= SMALL_VECTOR ? SVector(0.0, 1.0, 0.0, 0.0) : bcon
    e1u = _project_out(trial, e0, gcov)
    e1 = e1u ./ sqrt(abs(dot(e1u, gcov * e1u)))

    # e_(2): x2 direction, projected orthogonal to e_(0), e_(1).
    t2 = SVector(0.0, 0.0, 1.0, 0.0)
    e2u = _project_out(_project_out(t2, e0, gcov), e1, gcov)
    e2 = e2u ./ sqrt(abs(dot(e2u, gcov * e2u)))

    # e_(3): x3 direction, projected orthogonal to e_(0), e_(1), e_(2).
    t3 = SVector(0.0, 0.0, 0.0, 1.0)
    e3u = _project_out(_project_out(_project_out(t3, e0, gcov), e1, gcov), e2, gcov)
    e3 = e3u ./ sqrt(abs(dot(e3u, gcov * e3u)))

    return @inbounds SMatrix{4,4}(
        e0[1], e1[1], e2[1], e3[1],
        e0[2], e1[2], e2[2], e3[2],
        e0[3], e1[3], e2[3], e3[3],
        e0[4], e1[4], e2[4], e3[4])
end

# Remove the component of `a` along `b` under the metric inner product:
# a - b * <a,b>/<b,b>. Uses the SIGNED norm of `b` (matching make_tetrad's
# project_out), which is correct for the timelike e_(0) as well.
@inline function _project_out(a::SVector{4}, b::SVector{4}, gcov::SMatrix{4,4})
    gb = gcov * b
    return a .- b .* (dot(a, gb) / dot(b, gb))
end
