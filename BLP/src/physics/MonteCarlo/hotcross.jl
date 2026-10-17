# The thermally averaged (hot) Compton cross section σ_h(ε, Θₑ), the scattering opacity:
# the Klein–Nishina total cross section boosted into each electron's rest frame and
# averaged over the Maxwell–Jüttner distribution and the collision angle,
#   σ_h = (1/n_e) ∫ d³p (dn/d³p) (1 − μ_e β_e) σ_KN(ε_e),  ε_e = ε γ_e (1 − μ_e β_e)
# (Wienke 1985; Dolence et al. 2009; gpumonty `hotcross.cu`). Evaluated by a 2-D midpoint
# rule with fixed trip counts; production interpolates the log-log table.

"Lowest dimensionless photon energy ε = hν/mₑc² of the table."
const MINW = 1.0e-12
"Highest dimensionless photon energy of the table."
const MAXW = 1.0e15
"Lowest electron temperature of the table; below it the cold Klein–Nishina limit applies."
const MINT = 1.0e-4
"Highest electron temperature of the table."
const MAXT = 1.0e4
"Energy intervals of the table (NW + 1 nodes)."
const NW = 220
"Temperature intervals of the table (NT + 1 nodes)."
const NT = 80
"Step of the collision-cosine quadrature."
const DMUE = 0.05
"Step of the Lorentz-factor quadrature, in units of Θₑ."
const DGAMMAE = 0.05
"Upper end of the Lorentz-factor quadrature: γ_e ≤ 1 + MAXGAMMA Θₑ."
const MAXGAMMA = 12.0
"Midpoint nodes of the collision cosine over [−1, 1]: 2/DMUE."
const N_MUE = 40
"Midpoint nodes of the Lorentz factor: MAXGAMMA/DGAMMAE, independent of Θₑ."
const N_GAMMAE = 240

"""
    kn_total(eps_e) -> Float64

Klein–Nishina total cross section over the Thomson value, `σ_KN(ε_e)/σ_T`, for a photon
of rest-frame energy `eps_e` (mₑc²): `1 − 2ε_e` below `ε_e = 10⁻³` (the expansion that
avoids the cancellation of the closed form), the closed form above.
"""
@inline function kn_total(eps_e)
    w = eps_e
    if w < 1.0e-3
        return 1.0 - 2.0 * w
    end
    return 0.75 * (2.0 / (w * w) +
                   (1.0 / (2.0 * w) - (1.0 + w) / (w * w * w)) * log(1.0 + 2.0 * w) +
                   (1.0 + w) / ((1.0 + 2.0 * w) * (1.0 + 2.0 * w)))
end

"""
    boostcross(eps, mu_e, gamma_e) -> Float64

The flux-weighted boosted cross section over `σ_T` for a lab-frame photon of energy
`eps` and an electron of Lorentz factor `gamma_e` at collision cosine `mu_e`:
`(1 − μ_e β_e) σ_KN(ε γ_e (1 − μ_e β_e))/σ_T` with `β_e = √(γ_e² − 1)/γ_e`.
"""
@inline function boostcross(eps, mu_e, gamma_e)
    v = sqrt(gamma_e * gamma_e - 1.0) / gamma_e
    we = eps * gamma_e * (1.0 - mu_e * v)
    return kn_total(we) * (1.0 - mu_e * v)
end

"""
    dNdgamma_e(theta_e, gamma_e) -> Float64

Maxwell–Jüttner distribution `dN/dγ_e` per electron (unit integral over `γ_e ≥ 1`),
`γ √(γ² − 1) e^{−(γ−1)/Θ} / (Θ K₂f)` with `K₂f = K₂(1/Θ) e^{1/Θ}` above `Θ = 10⁻²` and
the large-argument series `√(πΘ/2)(1 + 15Θ/8 + 105Θ²/128)` (A&S 9.7.2) below, where the
product of Bessel function and exponential overflows.
"""
@inline function dNdgamma_e(theta_e, gamma_e)
    K2f = theta_e > 1.0e-2 ? bessel_k2(1.0 / theta_e) * exp(1.0 / theta_e) :
          sqrt(π * theta_e / 2.0) *
              (1.0 + theta_e * (15.0 / 8.0 + theta_e * (105.0 / 128.0)))
    return (gamma_e * sqrt(gamma_e * gamma_e - 1.0) / (theta_e * K2f)) *
           exp(-(gamma_e - 1.0) / theta_e)
end

"""
    total_compton_cross_num(eps, theta_e) -> Float64

Hot Compton cross section `σ_h(ε, Θₑ)` (cm²) by the midpoint rule over `μ_e ∈ [−1, 1]`
(`N_MUE` nodes) and `γ_e ∈ [1, 1 + MAXGAMMA Θₑ]` (`N_GAMMAE` nodes) of
`boostcross × dN/dγ_e / 2`. Cold corners: `σ_T` for `Θₑ < MINT` and `ε < MINW`,
`kn_total(ε) σ_T` for `Θₑ < MINT`.
"""
@inline function total_compton_cross_num(eps, theta_e)
    if theta_e < MINT && eps < MINW
        return SIGMA_THOMSON
    end
    if theta_e < MINT
        return kn_total(eps) * SIGMA_THOMSON
    end

    dgammae = theta_e * DGAMMAE
    cross = 0.0
    @inbounds for im in 0:(N_MUE - 1)
        mue = -1.0 + (im + 0.5) * DMUE
        for ig in 0:(N_GAMMAE - 1)
            gammae = 1.0 + (ig + 0.5) * dgammae
            f = 0.5 * dNdgamma_e(theta_e, gammae)
            cross += DMUE * dgammae * boostcross(eps, mue, gammae) * f
        end
    end
    return cross * SIGMA_THOMSON
end

"""
    HotcrossTable{M}(logsigma)

The hot cross section on the log-log grid: `logsigma[i+1, j+1] = log₁₀ σ_h` (cm²) at
`ε = 10^(log₁₀ MINW + i Δw)`, `Θₑ = 10^(log₁₀ MINT + j ΔT)`, `i ∈ 0:NW`, `j ∈ 0:NT`,
`Δw = log₁₀(MAXW/MINW)/NW`, `ΔT = log₁₀(MAXT/MINT)/NT`. `isbits` after `Adapt.adapt`
to a device. Built by `Data.build_hotcross_table` or read by `Data.load_hotcross`.
"""
struct HotcrossTable{M<:AbstractMatrix{Float64}}
    logsigma::M
end
Adapt.@adapt_structure HotcrossTable

"""
    hotcross_lookup(table, eps, theta_e) -> Float64

`σ_h(ε, Θₑ)` (cm²), with the branches in the order of gpumonty
`total_compton_cross_lkup`: `σ_T` for `ε Θₑ < 10⁻⁶`; `kn_total(ε) σ_T` for
`Θₑ < MINT`; bilinear interpolation in `log₁₀ ε`, `log₁₀ Θₑ` strictly inside the
table; `total_compton_cross_num` elsewhere.
"""
@inline hotcross_lookup(table::HotcrossTable, eps, theta_e) =
    hotcross_lookup(table.logsigma, eps, theta_e)

@inline function hotcross_lookup(logsigma::AbstractMatrix, eps, theta_e)
    if eps * theta_e < 1.0e-6
        return SIGMA_THOMSON
    end
    if theta_e < MINT
        return kn_total(eps) * SIGMA_THOMSON
    end
    if (eps > MINW && eps < MAXW) && (theta_e > MINT && theta_e < MAXT)
        dlw = log10(MAXW / MINW) / NW
        dlT = log10(MAXT / MINT) / NT
        lminw = log10(MINW)
        lmint = log10(MINT)
        fi = (log10(eps) - lminw) / dlw
        fj = (log10(theta_e) - lmint) / dlT
        i = floor(Int, fi)
        j = floor(Int, fj)
        di = fi - i
        dj = fj - j
        @inbounds lcross =
            (1.0 - di) * (1.0 - dj) * logsigma[i + 1, j + 1] +
            di * (1.0 - dj)         * logsigma[i + 2, j + 1] +
            (1.0 - di) * dj         * logsigma[i + 1, j + 2] +
            di * dj                 * logsigma[i + 2, j + 2]
        return 10.0^lcross
    end
    return total_compton_cross_num(eps, theta_e)
end

"""
    kappa_es(table, nu, theta_e) -> Float64

Electron-scattering mass opacity `σ_h/m_p` (cm² g⁻¹) at frequency `nu` (Hz) and
temperature `theta_e`, with `ε = hν/mₑc²` (grmonty `kappa_es`).
"""
@inline function kappa_es(table::HotcrossTable, nu, theta_e)
    eps = HPL * nu / (ME * CL * CL)
    return hotcross_lookup(table, eps, theta_e) / MP
end

"""
    alpha_inv_scatt(table, nu, theta_e, n_e) -> Float64

Invariant scattering coefficient `ν n_e σ_h(hν/mₑc², Θₑ)` (Hz cm⁻¹) at frequency `nu`
(Hz), temperature `theta_e` and electron density `n_e` (cm⁻³).
"""
@inline function alpha_inv_scatt(table::HotcrossTable, nu, theta_e, n_e)
    eps = HPL * nu / (ME * CL * CL)
    return nu * n_e * hotcross_lookup(table, eps, theta_e)
end
