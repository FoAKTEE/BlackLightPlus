# Code-unit primitives to the cgs plasma state: the normal-frame four-vector transform
# of blacklight `simulation_coefficients.cpp`, the unit conversion, the electron
# temperature and the cuts, in one sequence shared by imaging and Monte Carlo.

"Ideal-gas pressure from the internal-energy density: p_gas = (γ − 1) e_int."
@inline pgas_from_eint(eint, gamma) = (gamma - 1) * eint

"""
    primitive_four_vectors(uu, bb, gcov, gcon) -> (ucon, ucov, bcon, bcov, b_sq)

Fluid four-velocity and magnetic four-vector from the AthenaK normal-frame velocity
`uu = W v^i` and the cell-centred lab 3-field `bb`, given the covariant and
contravariant metric at the sample (index 1 is time): `W = √(1 + γ_ij uu^i uu^j)`,
lapse `1/√(−g^{00})`, shift `−g^{0i}/g^{00}`, `u^0 = W/lapse`,
`u^i = uu^i − shift^i W/lapse`, `b^0 = u_i B^i`, `b^i = (B^i + b^0 u^i)/u^0`, so that
`u·u = −1` and `b·u = 0`; `b_sq = b^μ b_μ`. After blacklight
`PrimitiveToFourVectorsG`.
"""
@inline function primitive_four_vectors(uu::SVector{3}, bb::SVector{3},
                                        gcov::SMatrix{4,4}, gcon::SMatrix{4,4})
    uu1, uu2, uu3 = uu[1], uu[2], uu[3]
    bb1, bb2, bb3 = bb[1], bb[2], bb[3]
    uu0 = sqrt(1 +
        gcov[2, 2] * uu1 * uu1 + 2 * gcov[2, 3] * uu1 * uu2 + 2 * gcov[2, 4] * uu1 * uu3 +
        gcov[3, 3] * uu2 * uu2 + 2 * gcov[3, 4] * uu2 * uu3 +
        gcov[4, 4] * uu3 * uu3)
    lapse  = 1 / sqrt(-gcon[1, 1])
    shift1 = -gcon[1, 2] / gcon[1, 1]
    shift2 = -gcon[1, 3] / gcon[1, 1]
    shift3 = -gcon[1, 4] / gcon[1, 1]

    ucon = SVector(uu0 / lapse,
                   uu1 - shift1 * uu0 / lapse,
                   uu2 - shift2 * uu0 / lapse,
                   uu3 - shift3 * uu0 / lapse)
    ucov = gcov * ucon

    b0 = ucov[2] * bb1 + ucov[3] * bb2 + ucov[4] * bb3
    bcon = SVector(b0,
                   (bb1 + b0 * ucon[2]) / ucon[1],
                   (bb2 + b0 * ucon[3]) / ucon[1],
                   (bb3 + b0 * ucon[4]) / ucon[1])
    bcov = gcov * bcon

    b_sq = dot(bcon, bcov)

    return (ucon = ucon, ucov = ucov, bcon = bcon, bcov = bcov, b_sq = b_sq)
end

"""
    state_type(::Type{Tp}, ::Type{Tg}, units::Units, electrons::ElectronModel) -> Type

Number type of the `PlasmaState` decoded from primitives of type `Tp` with a metric of
element type `Tg`: the promotion of both with the unit and electron-model types, so a
`ForwardDiff.Dual` mass scale or temperature-ratio parameter flows through.
"""
@inline state_type(::Type{Tp}, ::Type{Tg}, ::Units{Tu}, ::ElectronModel{Te}) where {Tp,Tg,Tu,Te} =
    promote_type(Tp, Tg, Tu, Te)

"""
    decode(prim::PlasmaPrimitives, g, units::Units, electrons::ElectronModel, cuts::PlasmaCuts)
        -> PlasmaState

The plasma state of the primitives `prim` at a point whose metric pair is
`g = (gcov, gcon)`. The sample is `vacuum` when the density or the ideal-gas
pressure `(γ − 1) e_int` is not positive, when the metric has no normal observer
(`g^{tt} ≥ 0`: the slice is not spacelike there, as it is not deep inside a hole of a
superposed metric), when `b² ≤ 0`, or when any decoded quantity falls outside
`cuts`. Otherwise: `n_e = ρ dens_unit / (μ m_p) / (1 + 1/(n_e/n_i))`,
`B = √b² B_unit`, `Θ_e` from `electron_theta`, `σ = b²/ρ` and `β⁻¹ = b²/(2 p_gas)`
in code units, and the four-vectors of `primitive_four_vectors`.
"""
@inline function decode(prim::PlasmaPrimitives{Tp}, g, units::Units, electrons::ElectronModel,
                        cuts::PlasmaCuts) where {Tp}
    gcov, gcon = g
    T = state_type(Tp, eltype(gcov), units, electrons)
    rho = prim.rho
    pgas = pgas_from_eint(prim.eint, electrons.gamma)
    gcon[1, 1] < 0 || return vacuum(T)
    fv = primitive_four_vectors(prim.uu, prim.bb, gcov, gcon)
    (rho <= 0.0 || pgas <= 0.0) && return vacuum(T)
    b_sq = fv.b_sq
    b_sq <= 0.0 && return vacuum(T)

    rho_cgs = rho * units.dens_unit
    pgas_cgs = pgas * units.e_unit
    n_cgs = rho_cgs / (electrons.mu * MP)
    n_e_cgs = n_cgs / (1.0 + 1.0 / electrons.ne_ni)
    bb_cgs = sqrt(b_sq) * units.B_unit
    sigma = b_sq / rho
    beta_inv = b_sq / (2.0 * pgas)
    theta_e = electron_theta(electrons, rho_cgs, pgas_cgs, beta_inv)
    omitted(cuts, rho_cgs, n_e_cgs, pgas_cgs, theta_e, bb_cgs, sigma, beta_inv) && return vacuum(T)
    return PlasmaState{T}(n_e_cgs, theta_e, bb_cgs, fv.ucon, fv.ucov, fv.bcon, fv.bcov,
                          b_sq, sigma, beta_inv, true)
end
