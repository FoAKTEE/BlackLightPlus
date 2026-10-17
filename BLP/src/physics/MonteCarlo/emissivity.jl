# Emission-frame physics of superphoton generation: the non-invariant thermal
# synchrotron emissivity and absorptivity (cgs), the pitch-angle envelope, the
# angle-integrated zone emissivity, and the (u, b̂)-aligned emission tetrad
# (gpumonty `tetrads.cu` `make_tetrad`). The fits are `Transfer.thermal_synchrotron`
# (Leung et al. 2011), whose invariants are j/ν² and αν.

"""
    j_nu(n_e, theta_e, B, nu, sin_theta) -> Float64

Thermal synchrotron emissivity `j_ν` (erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹) of electrons of density
`n_e` (cm⁻³) and temperature `theta_e` in a field `B` (G) at frequency `nu` (Hz) and
pitch angle `θ` to the field: the invariant `j/ν²` of `thermal_synchrotron` times `ν²`.
"""
@inline function j_nu(n_e, theta_e, B, nu, sin_theta)
    j_inv, _ = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_theta)
    return j_inv * nu^2
end

"""
    alpha_nu(n_e, theta_e, B, nu, sin_theta) -> Float64

Thermal synchrotron absorptivity `α_ν` (cm⁻¹) from Kirchhoff's law: the invariant
`αν` of `thermal_synchrotron` divided by `ν`.
"""
@inline function alpha_nu(n_e, theta_e, B, nu, sin_theta)
    _, alpha_inv = thermal_synchrotron(1.0, n_e, theta_e, B, nu, sin_theta)
    return alpha_inv / nu
end

"""
    jmax(n_e, theta_e, B, nu) -> Float64

Envelope of the pitch-angle rejection: `j_nu` at `sin θ = 1`, the maximum over pitch
angle for `ν ≳ ν_s`. A direction is accepted with probability `j_ν(θ)/jmax`
(grmonty, gpumonty).
"""
@inline jmax(n_e, theta_e, B, nu) = j_nu(n_e, theta_e, B, nu, 1.0)

"Simpson intervals of the pitch-angle integral of `int_jnu_zone` (even)."
const N_ZONE_THETA = 512

"""
    int_jnu_zone(n_e, theta_e, B, nu) -> Float64

Angle-integrated emissivity `∫ j_ν dΩ` (erg s⁻¹ cm⁻³ Hz⁻¹), the per-volume emission
rate of a zone at frequency `nu`. `j_ν` depends on direction through `sin θ` only and
is symmetric about the equator, so `∫ j_ν dΩ = 4π ∫₀^{π/2} j_ν(sin θ) sin θ dθ`,
evaluated by the composite Simpson rule on `N_ZONE_THETA` intervals (fixed trip
count, no allocation).
"""
@inline function int_jnu_zone(n_e, theta_e, B, nu)
    a = 0.0
    b = 0.5 * π
    h = (b - a) / N_ZONE_THETA
    acc = 0.0
    @inbounds for i in 0:N_ZONE_THETA
        th = a + i * h
        s = sin(th)
        w = (i == 0 || i == N_ZONE_THETA) ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        acc += w * j_nu(n_e, theta_e, B, nu, s) * s
    end
    return 4.0 * π * acc * h / 3.0
end

"Metric norm below which a trial field vector is degenerate (gpumonty `SMALL_VECTOR`); the emission tetrad then takes e₍₁₎ along x¹."
const SMALL_VECTOR = 1.0e-30

"""
    emission_tetrad(ucon, ucov, bcon, gcov, gcon) -> SMatrix{4,4}

The orthonormal emission tetrad, row `a` holding `e_(a)^μ`: `e_(0) = u`, `e_(1)` along
the field `b` projected orthogonal to `u` (the pitch-angle axis), and `e_(2)`, `e_(3)`
by Gram–Schmidt on the x² and x³ directions under the metric `gcov`. A field of metric
norm at most `SMALL_VECTOR` is replaced by the x¹ direction. `ucov` and `gcon` are not
read. Not the imaging tetrad, whose `e_(3)` is along the photon.
"""
@inline function emission_tetrad(ucon::SVector{4}, ucov::SVector{4},
                                 bcon::SVector{4}, gcov::SMatrix{4,4},
                                 gcon::SMatrix{4,4})
    n0 = dot(ucon, gcov * ucon)
    e0 = ucon ./ sqrt(abs(n0))

    trial = dot(bcon, gcov * bcon) <= SMALL_VECTOR ? SVector(0.0, 1.0, 0.0, 0.0) : bcon
    e1u = project_out(trial, e0, gcov)
    e1 = e1u ./ sqrt(abs(dot(e1u, gcov * e1u)))

    t2 = SVector(0.0, 0.0, 1.0, 0.0)
    e2u = project_out(project_out(t2, e0, gcov), e1, gcov)
    e2 = e2u ./ sqrt(abs(dot(e2u, gcov * e2u)))

    t3 = SVector(0.0, 0.0, 0.0, 1.0)
    e3u = project_out(project_out(project_out(t3, e0, gcov), e1, gcov), e2, gcov)
    e3 = e3u ./ sqrt(abs(dot(e3u, gcov * e3u)))

    return @inbounds SMatrix{4,4}(
        e0[1], e1[1], e2[1], e3[1],
        e0[2], e1[2], e2[2], e3[2],
        e0[3], e1[3], e2[3], e3[3],
        e0[4], e1[4], e2[4], e3[4])
end

# a − b ⟨a,b⟩/⟨b,b⟩ under the metric, with the signed norm of b (valid for timelike b).
@inline function project_out(a::SVector{4}, b::SVector{4}, gcov::SMatrix{4,4})
    gb = gcov * b
    return a .- b .* (dot(a, gb) / dot(b, gb))
end
