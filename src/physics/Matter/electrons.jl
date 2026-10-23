"""
    ElectronModel{T}

Electron thermodynamics of the simulation model: mean molecular weight `mu`
(mₚ), electrons per ion `ne_ni`, whether the fluid pressure is the total
pressure (`use_p`) or the internal energy fixes the temperature, the
`rat_low`/`rat_high` ion-to-electron temperature ratios at β → 0 and β → ∞,
the fluid, ion and electron adiabatic indices `gamma`, `gamma_i`, `gamma_e`,
and `thermal_frac`, the thermal population fraction. A zero `thermal_frac`
makes `electron_theta` NaN so that no temperature cut trips.
"""
struct ElectronModel{T}
    mu::T
    ne_ni::T
    use_p::Bool
    rat_low::T
    rat_high::T
    gamma::T
    gamma_i::T
    gamma_e::T
    thermal_frac::T
end

function ElectronModel{T}(; mu = 0.5, ne_ni = 1.0, use_p = true, rat_low = 1.0, rat_high = 10.0,
                          gamma = 5.0 / 3.0, gamma_i = 5.0 / 3.0, gamma_e = 4.0 / 3.0,
                          thermal_frac = 1.0) where {T}
    return ElectronModel{T}(mu, ne_ni, use_p, rat_low, rat_high, gamma, gamma_i, gamma_e, thermal_frac)
end
ElectronModel(; kw...) = ElectronModel{Float64}(; kw...)

"""
    electron_theta(m::ElectronModel, rho_cgs, pgas_cgs, beta_inv)

Dimensionless electron temperature Θₑ = k Tₑ / mₑ c² of the R_low/R_high
prescription, Tᵢ/Tₑ = (R_high + R_low β⁻²)/(1 + β⁻²), from the cgs rest-mass
density and gas pressure. NaN when `m.thermal_frac` is zero.
"""
@inline function electron_theta(m::ElectronModel, rho_cgs, pgas_cgs, beta_inv)
    tti_tte = (m.rat_high + m.rat_low * beta_inv * beta_inv) / (1.0 + beta_inv * beta_inv)
    kb_tt_tot_cgs = m.mu * MP * pgas_cgs / rho_cgs
    local kb_tt_e_cgs
    if m.use_p
        kb_tt_e_cgs = (1.0 + m.ne_ni) / (tti_tte + m.ne_ni) * kb_tt_tot_cgs
    else
        kb_tt_e_cgs = (1.0 + m.ne_ni) * kb_tt_tot_cgs / (m.gamma - 1.0)
        kb_tt_e_cgs /= tti_tte / (m.gamma_i - 1.0) + m.ne_ni / (m.gamma_e - 1.0)
    end
    theta_e = kb_tt_e_cgs / (ME * CL * CL)
    return m.thermal_frac == 0.0 ? oftype(theta_e, NaN) : theta_e
end

"""
    Bound{T}

The closed interval `[lo, hi]` of admissible values of one plasma quantity; an
infinite end is inactive. `outside(b, v)` is true for `v < lo` or `v > hi` and
therefore false for NaN.
"""
struct Bound{T}
    lo::T
    hi::T
end
Bound{T}() where {T} = Bound{T}(T(-Inf), T(Inf))
Bound(lo, hi) = Bound(promote(lo, hi)...)

"""
    Bound{T}(lo, hi; disabled = -1)

A bound in the legacy convention, where a value below zero (`disabled`) turns an
end off: `lo < 0` becomes `-Inf`, `hi < 0` becomes `Inf`.
"""
legacy_bound(::Type{T}, lo, hi) where {T} = Bound{T}(lo >= 0 ? T(lo) : T(-Inf), hi >= 0 ? T(hi) : T(Inf))

"Whether `v` lies outside the bound `b`; false for NaN."
@inline outside(b::Bound, v) = v < b.lo || v > b.hi

"""
    PlasmaCuts{T}

Bounds below or above which a decoded sample is omitted and treated as vacuum:
rest-mass density `rho` and electron density `n_e` (cgs), gas pressure `pgas`
(cgs), electron temperature `theta_e`, field strength `b` (G), magnetisation
`sigma` and inverse plasma beta `beta_inv`. Every bound defaults to inactive.
"""
struct PlasmaCuts{T}
    rho::Bound{T}
    n_e::Bound{T}
    pgas::Bound{T}
    theta_e::Bound{T}
    b::Bound{T}
    sigma::Bound{T}
    beta_inv::Bound{T}
end

function PlasmaCuts{T}(; rho = Bound{T}(), n_e = Bound{T}(), pgas = Bound{T}(), theta_e = Bound{T}(),
                       b = Bound{T}(), sigma = Bound{T}(), beta_inv = Bound{T}()) where {T}
    return PlasmaCuts{T}(rho, n_e, pgas, theta_e, b, sigma, beta_inv)
end
PlasmaCuts(; kw...) = PlasmaCuts{Float64}(; kw...)

"""
    omitted(c::PlasmaCuts, rho_cgs, n_e_cgs, pgas_cgs, theta_e, b_cgs, sigma, beta_inv) -> Bool

True when any of the seven decoded quantities falls outside its bound. A NaN
quantity never trips a cut.
"""
@inline function omitted(c::PlasmaCuts, rho_cgs, n_e_cgs, pgas_cgs, theta_e, b_cgs, sigma, beta_inv)
    return outside(c.rho, rho_cgs) || outside(c.n_e, n_e_cgs) || outside(c.pgas, pgas_cgs) ||
           outside(c.theta_e, theta_e) || outside(c.b, b_cgs) || outside(c.sigma, sigma) ||
           outside(c.beta_inv, beta_inv)
end
