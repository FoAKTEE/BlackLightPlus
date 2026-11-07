# Emission-frame physics of superphoton generation: the non-invariant thermal
# synchrotron emissivity and absorptivity (cgs), the pitch-angle envelope, the
# angle-integrated zone emissivity through the one-variable table F(K), and the
# (u, b̂)-aligned emission tetrad
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

"Simpson intervals of the reference pitch-angle integral [`int_jnu_simpson`](@ref) (even)."
const N_ZONE_THETA = 512

"""
    int_jnu_simpson(n_e, theta_e, B, nu) -> Float64

Angle-integrated emissivity `∫ j_ν dΩ` (erg s⁻¹ cm⁻³ Hz⁻¹) by direct quadrature, the
reference of [`int_jnu`](@ref): `j_ν` depends on direction through `sin θ` only and is
symmetric about the equator, so `∫ j_ν dΩ = 4π ∫₀^{π/2} j_ν(sin θ) sin θ dθ`, evaluated
by the composite Simpson rule on `N_ZONE_THETA` intervals (fixed trip count, no
allocation).
"""
@inline function int_jnu_simpson(n_e, theta_e, B, nu)
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

# The angle integral of the synchrotron fit. With X = ν/ν_s, ν_s = (2/9) ν_c Θₑ² sin θ,
# the fit of `thermal_synchrotron` is
#   j_ν = n_e e² ν_c/c · (√2 π/27) sin θ · G(X),   G(X) = (X^{1/2} + 2^{11/12} X^{1/6})² e^{−X^{1/3}},
# and depends on ν and Θₑ only through K = ν/((2/9) ν_c Θₑ²) = X sin θ. Hence
#   ∫ j_ν dΩ = 4π ∫₀^{π/2} j_ν sin θ dθ = 4π n_e e² ν_c/c · (√2 π/27) · F(K),
#   F(K) = ∫₀^{π/2} sin²θ G(K/sin θ) dθ,
# exactly: one function of one variable. Its logarithmic slope is
#   d ln F/d ln K = (1/F) ∫₀^{π/2} sin²θ X G'(X) dθ,
#   X G'(X)/G(X) = (X^{1/2} + (b/3) X^{1/6})/(X^{1/2} + b X^{1/6}) − X^{1/3}/3,  b = 2^{11/12}.
# F ∝ K^{1/3} as K → 0 and F falls as e^{−K^{1/3}} as K → ∞.

const _G_B = 2.0^(11.0 / 12.0)

# (F, dF/dlnK) of `K` by the composite Simpson rule on `n` intervals of θ ∈ [0, π/2]
# (the θ = 0 node contributes nothing: sin²θ G(K/sin θ) → 0 there).
function _synchrotron_F(K::Float64, n::Int)
    h = 0.5 * π / n
    F = 0.0
    dF = 0.0
    @inbounds for i in 1:n
        s = sin(i * h)
        X = K / s
        x13 = cbrt(X)
        x16 = sqrt(x13)
        x12 = sqrt(X)
        c = x12 + _G_B * x16
        g = s * s * c * c * exp(-x13)
        w = i == n ? 1.0 : (isodd(i) ? 4.0 : 2.0)
        F += w * g
        dF += w * g * ((x12 + _G_B / 3.0 * x16) / c - x13 / 3.0)
    end
    return F * h / 3.0, dF * h / 3.0
end

"Simpson intervals of the quadrature that builds [`EmissivityTable`](@ref) (even)."
const F_TABLE_THETA = 2048
"Smallest `K` of the [`EmissivityTable`](@ref) nodes; below it `ln F` continues linearly in `ln K`."
const F_TABLE_KMIN = 1.0e-30
"`K` where the node spacing of the [`EmissivityTable`](@ref) changes from `F_TABLE_DL1` to `F_TABLE_DL2`."
const F_TABLE_KMID = 1.0e3
"Largest `K` of the [`EmissivityTable`](@ref); `F` is taken as 0 above it (`F < 1e-280`)."
const F_TABLE_KMAX = 3.0e8
"Target spacing in `ln K` of the [`EmissivityTable`](@ref) nodes below `F_TABLE_KMID`."
const F_TABLE_DL1 = 0.02
"Target spacing in `ln K` of the [`EmissivityTable`](@ref) nodes above `F_TABLE_KMID`."
const F_TABLE_DL2 = 0.005

"""
    EmissivityTable{V}(l0, l1, l2, h1, h2, n1, n2, lnF, slope)

The angle integral `F(K) = ∫₀^{π/2} sin²θ G(K/sin θ) dθ` of the thermal synchrotron fit,
`G(X) = (X^{1/2} + 2^{11/12} X^{1/6})² e^{−X^{1/3}}`, through which
`∫ j_ν dΩ = 4π n_e e² ν_c/c · (√2 π/27) · F(K)`, `K = ν/((2/9) ν_c Θₑ²)`,
`ν_c = eB/(2π mₑ c)` ([`int_jnu`](@ref)). Nodes in `l = ln K`: `n1` intervals of width
`h1` from `l0` to `l1`, then `n2` of width `h2` from `l1` to `l2`; `lnF[k]` and
`slope[k] = d ln F/d ln K` at node `k`, both from the Simpson rule on `F_TABLE_THETA`
intervals. [`synchrotron_F`](@ref) interpolates `ln F` by cubic Hermite polynomials.
`isbits` after `Adapt.adapt` to a device. The process-wide table is
[`EMISSIVITY_TABLE`](@ref).
"""
struct EmissivityTable{V<:AbstractVector{Float64}}
    l0::Float64
    l1::Float64
    l2::Float64
    h1::Float64
    h2::Float64
    n1::Int
    n2::Int
    lnF::V
    slope::V
end
Adapt.@adapt_structure EmissivityTable

"""
    build_emissivity_table() -> EmissivityTable

The [`EmissivityTable`](@ref) over `[F_TABLE_KMIN, F_TABLE_KMAX]`: spacing at most
`F_TABLE_DL1` in `ln K` up to `F_TABLE_KMID` and at most `F_TABLE_DL2` above it (where the
curvature of `ln F` grows as `K^{1/3}`), `ln F` and its slope at every node from the
Simpson rule on `F_TABLE_THETA` intervals. Serial; about a second.
"""
function build_emissivity_table()
    l0 = log(F_TABLE_KMIN); l1 = log(F_TABLE_KMID); l2 = log(F_TABLE_KMAX)
    n1 = ceil(Int, (l1 - l0) / F_TABLE_DL1)
    n2 = ceil(Int, (l2 - l1) / F_TABLE_DL2)
    h1 = (l1 - l0) / n1
    h2 = (l2 - l1) / n2
    lnF = Vector{Float64}(undef, n1 + n2 + 1)
    slope = similar(lnF)
    for k in 1:(n1 + n2 + 1)
        l = k <= n1 + 1 ? l0 + (k - 1) * h1 : l1 + (k - 1 - n1) * h2
        F, dF = _synchrotron_F(exp(l), F_TABLE_THETA)
        lnF[k] = log(F)
        slope[k] = dF / F
    end
    return EmissivityTable(l0, l1, l2, h1, h2, n1, n2, lnF, slope)
end

"""
    EMISSIVITY_TABLE

The process-wide [`EmissivityTable`](@ref), `build_emissivity_table()` evaluated once when
the package is compiled. Uploaded once per device by the residency cache.
"""
const EMISSIVITY_TABLE = build_emissivity_table()

"""
    synchrotron_F(t::EmissivityTable, K) -> Float64

`F(K)` of the table `t`: cubic Hermite interpolation of `ln F` in `ln K` between the
two nodes that bracket `ln K`, with the node slopes; below the first node `ln F`
continues along the first node's slope (the `K^{1/3}` limit), above the last node
`F = 0`, and `F = 0` for `K ≤ 0`. The logarithm and exponential are `tang_log` and `tang_exp`, so a
device kernel computes the host's bits.
"""
@inline function synchrotron_F(t::EmissivityTable, K)
    K > 0.0 || return 0.0
    l = tang_log(K)
    l < t.l2 || return 0.0
    @inbounds if l < t.l0
        return tang_exp(t.lnF[1] + t.slope[1] * (l - t.l0))
    end
    if l < t.l1
        f = (l - t.l0) / t.h1
        n = t.n1; base = 0; h = t.h1
    else
        f = (l - t.l1) / t.h2
        n = t.n2; base = t.n1; h = t.h2
    end
    i = min(unsafe_trunc(Int, f), n - 1)
    u = f - i
    k = base + i + 1
    @inbounds begin
        y0 = t.lnF[k]; y1 = t.lnF[k + 1]
        m0 = t.slope[k] * h; m1 = t.slope[k + 1] * h
    end
    u2 = u * u
    u3 = u2 * u
    return tang_exp((2.0 * u3 - 3.0 * u2 + 1.0) * y0 + (u3 - 2.0 * u2 + u) * m0 +
                    (3.0 * u2 - 2.0 * u3) * y1 + (u3 - u2) * m1)
end

"""
    int_jnu(t::EmissivityTable, n_e, theta_e, B, nu) -> Float64

Angle-integrated thermal synchrotron emissivity `∫ j_ν dΩ` (erg s⁻¹ cm⁻³ Hz⁻¹) of the
fit of `thermal_synchrotron`, exactly `4π n_e e² ν_c/c · (√2 π/27) · F(K)` with
`ν_c = eB/(2π mₑ c)` and `K = ν/((2/9) ν_c Θₑ²)` (derivation in the source), `F` from
the table `t` ([`synchrotron_F`](@ref)). Zero for a non-positive input, as the fit.
Within `3e-9` of [`int_jnu_simpson`](@ref) wherever that quadrature does not underflow.
"""
@inline function int_jnu(t::EmissivityTable, n_e, theta_e, B, nu)
    (n_e <= 0.0 || theta_e <= 0.0 || B <= 0.0 || nu <= 0.0) && return 0.0
    nu_c = EE * B / (2.0 * π * ME * CL)
    K = nu / (2.0 / 9.0 * nu_c * theta_e * theta_e)
    return 4.0 * π * n_e * EE * EE * nu_c / CL * (SQRT2 * π / 27.0) * synchrotron_F(t, K)
end

"""
    int_jnu_zone(n_e, theta_e, B, nu) -> Float64

`∫ j_ν dΩ` (erg s⁻¹ cm⁻³ Hz⁻¹) from the process-wide [`EMISSIVITY_TABLE`](@ref):
`int_jnu(EMISSIVITY_TABLE, n_e, theta_e, B, nu)`. Host only (a kernel reads the table it
is handed).
"""
@inline int_jnu_zone(n_e, theta_e, B, nu) = int_jnu(EMISSIVITY_TABLE, n_e, theta_e, B, nu)

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
