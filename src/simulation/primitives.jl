"""
primitives.jl — normal-frame primitive → GR four-vector transform (after
blacklight, `simulation_coefficients.cpp`). AthenaK GRMHD dumps store the
normal-frame velocity `uu^i = W v^i` (measured by the Eulerian observer) plus
the cell-centred lab 3-field `B^i`. Decode: W = sqrt(1 + gamma_ij uu^i uu^j),
lapse = 1/sqrt(-g^{00}), shift^i = -g^{0i}/g^{00}, u^0 = W/lapse,
u^i = uu^i - shift^i W/lapse, b^0 = u_i B^i, b^i = (B^i + b^0 u^i)/u^0; then
u^μ u_μ = -1 and b^μ u_μ = 0. Index 1 is the time component; physics is Float64.
"""

"Ideal-gas pressure from internal-energy density: p_gas = (gamma - 1) * eint."
@inline pgas_from_eint(eint, gamma) = (gamma - 1) * eint

"""
    primitive_four_vectors(uu1, uu2, uu3, bb1, bb2, bb3, gcov, gcon) -> NamedTuple

Given the AthenaK normal-frame velocity `(uu1, uu2, uu3)`, the cell-centred lab
3-field `(bb1, bb2, bb3)`, and the covariant / contravariant metric
`gcov`, `gcon :: SMatrix{4,4}` at the sample point (index 1 = time), return
`(ucon, ucov, bcon, bcov, b_sq)`: u^μ, u_μ, b^μ, b_μ as `SVector{4}` and
`b_sq = b^μ b_μ ≥ 0`. Matches blacklight's `PrimitiveToFourVectorsG`.
"""
@inline function primitive_four_vectors(uu1, uu2, uu3, bb1, bb2, bb3,
                                        gcov::SMatrix{4,4}, gcon::SMatrix{4,4})
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

    return (ucon=ucon, ucov=ucov, bcon=bcon, bcov=bcov, b_sq=b_sq)
end
