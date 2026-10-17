# Transfer modes and the parameter set of the marches: units, electron model, cuts, the
# nonthermal populations with their fit constants computed once on the host, and the
# observing frequency.

"""
    TransferMode

The quantity a march accumulates along a ray: [`StokesI`](@ref), the invariant
intensity, or [`IQUV`](@ref), the full Stokes vector in the camera polarization
frame.
"""
abstract type TransferMode end

"""
    StokesI()

Unpolarized transfer: the march accumulates the Stokes I intensity with the exact
segment operators of [`segment`](@ref)`(StokesI(), j, α, ds)`.
"""
struct StokesI <: TransferMode end

"""
    IQUV()

Polarized transfer: the march accumulates `(I, Q, U, V)` with the Mueller operators
of [`segment`](@ref)`(IQUV(), j, α, ρ, ds)`, rotated into the parallel-transported
camera polarization frame.
"""
struct IQUV <: TransferMode end

"""
    PowerLaw{T}
    PowerLaw(frac, p, gamma_min, gamma_max)

Power-law electron population `n(γ) ∝ γ^-p` on `gamma_min ≤ γ ≤ gamma_max`
carrying the fraction `frac` of the electrons, with the `p`-dependent constants of
its synchrotron fits: `jj`, `aa` (emissivity and absorptivity normalisations,
Marszewski et al. 2021), `jj_q`, `jj_v`, `aa_q`, `aa_v` (polarized emissivity and
absorptivity factors), and `rho`, `rho_q`, `rho_v` (rotativity factors; `rho_q` is
`-Inf` at the removable pole `p = 2`, where the fit takes its `ln X` limit). The
constructor computes the constants once (blacklight
`PrecomputePowerLawCoefficients`); `PowerLaw{T}(pl)` converts every field to `T`.
"""
struct PowerLaw{T}
    frac::T
    p::T
    gamma_min::T
    gamma_max::T
    jj::T
    aa::T
    jj_q::T
    jj_v::T
    aa_q::T
    aa_v::T
    rho::T
    rho_q::T
    rho_v::T
end

function PowerLaw(frac, p, gamma_min, gamma_max)
    frac, p, gamma_min, gamma_max = promote(frac, p, gamma_min, gamma_max)
    var_a = 3.0^(p / 2.0) * (p - 1.0)
    var_b = 2.0 * (p + 1.0)
    var_c = gamma_min^(1.0 - p) - gamma_max^(1.0 - p)
    var_d = gamma((3.0 * p - 1.0) / 12.0)
    var_e = gamma((3.0 * p + 19.0) / 12.0)
    var_f = 3.0^((p + 1.0) / 2.0) * (p - 1.0) / 4.0
    var_g = gamma((3.0 * p + 2.0) / 12.0)
    var_h = gamma((3.0 * p + 22.0) / 12.0)
    jj = var_a / var_b / var_c * var_d * var_e
    aa = var_f / var_c * var_g * var_h

    var_i = 2.0 * (p + 2.0) / (p + 1.0)
    var_j = gamma_min^(-(p + 1.0))
    var_k = log(gamma_min)
    jj_q = -(p + 1.0) / (p + 7.0 / 3.0)
    jj_v = 0.684 * p^0.49
    aa_q = -(0.034 * p - 0.0344)^0.086
    aa_v = (0.71 * p + 0.0352)^0.394
    rho = (p - 1.0) / var_c
    rho_q = -gamma_min^(2.0 - p) / (p / 2.0 - 1.0)
    rho_v = var_i * var_j * var_k
    T = typeof(frac)
    return PowerLaw{T}(frac, p, gamma_min, gamma_max, jj, aa, jj_q, jj_v, aa_q, aa_v, rho,
                       rho_q, rho_v)
end

PowerLaw{T}(pl::PowerLaw) where {T} =
    PowerLaw{T}(ntuple(i -> T(getfield(pl, i)), Val(fieldcount(PowerLaw)))...)

"""
    Kappa{T}
    Kappa(frac, kappa, w)

Kappa electron population of index `kappa` and width `w` carrying the fraction
`frac` of the electrons, with the constants of its synchrotron fits (Pandya et al.
2016; Marszewski et al. 2021): the low- and high-frequency emissivity and
absorptivity normalisations `jj_low`, `jj_high`, `aa_low`, `aa_high`,
`aa_high_i` and bridge exponents `jj_x_i`, `aa_x_i`; their Q and V factors
`jj_low_q`, `jj_low_v`, `jj_high_q`, `jj_high_v`, `aa_low_q`, `aa_low_v`,
`aa_high_q`, `aa_high_v` and exponents `jj_x_q`, `jj_x_v`, `aa_x_q`, `aa_x_v`; the
Faraday-rotation factor `rho_v = K₀(1/w)/K₂(1/w)`; and the rotativity fits
bracketing `kappa` (`rho_frac` the blend weight between the bracketing indices,
`rho_q_low_a…e`, `rho_q_high_a…e`, `rho_v_low_a`, `rho_v_low_b`, `rho_v_high_a`,
`rho_v_high_b`), tabulated for `kappa` in [3.5, 5]. The constructor computes the
constants once (blacklight `PrecomputeKappaCoefficients`) and is host only: the
hypergeometric series is `Float64`. `Kappa{T}(k)` converts every field to `T`.
"""
struct Kappa{T}
    frac::T
    kappa::T
    w::T
    jj_low::T
    jj_high::T
    jj_x_i::T
    aa_low::T
    aa_high::T
    aa_high_i::T
    aa_x_i::T
    jj_low_q::T
    jj_low_v::T
    jj_high_q::T
    jj_high_v::T
    jj_x_q::T
    jj_x_v::T
    aa_low_q::T
    aa_low_v::T
    aa_high_q::T
    aa_high_v::T
    aa_x_q::T
    aa_x_v::T
    rho_v::T
    rho_frac::T
    rho_q_low_a::T
    rho_q_low_b::T
    rho_q_low_c::T
    rho_q_low_d::T
    rho_q_low_e::T
    rho_q_high_a::T
    rho_q_high_b::T
    rho_q_high_c::T
    rho_q_high_d::T
    rho_q_high_e::T
    rho_v_low_a::T
    rho_v_low_b::T
    rho_v_high_a::T
    rho_v_high_b::T
end

function Kappa(frac, kappa, w)
    frac, k, w = promote(frac, kappa, w)
    var_a = 4.0 * π * gamma(k - 4.0 / 3.0)
    var_b = 3.0^(7.0 / 3.0) * gamma(k - 2.0)
    var_c = 3.0^((k - 1.0) / 2.0)
    var_d = (k - 2.0) * (k - 1.0) / 4.0
    var_e = gamma(k / 4.0 - 1.0 / 3.0)
    var_f = gamma(k / 4.0 + 4.0 / 3.0)
    var_g = 3.0^(1.0 / 6.0) * 10.0 / 41.0
    var_h = w * k
    var_i = 2.0 * π * var_h^(k - 10.0 / 3.0)
    var_j = (k - 2.0) * (k - 1.0) * k
    var_k = 3.0 * k - 1.0
    var_l = gamma(5.0 / 3.0)
    var_m = hypergeometric_2f1(k - 1.0 / 3.0, k + 1.0, k + 2.0 / 3.0, -var_h)
    var_n = π^1.5 / 3.0
    var_o = var_j / (var_h * var_h * var_h)
    var_p = 2.0 * gamma(2.0 + k / 2.0) / (2.0 + k) - 1.0
    jj_low = var_a / var_b
    jj_high = var_c * var_d * var_e * var_f
    jj_x_i = 3.0 * k^(-1.5)
    aa_low = var_g * var_i * var_j / var_k * var_l * var_m
    aa_high = var_n * var_o * var_p
    aa_x_i = (-1.75 + 1.6 * k)^(-0.86)
    aa_high_i = (3.0 / k)^4.75 + 0.6

    var_q = 14.3 * w^(-0.928)
    var_r = 169.0 * k^(-8.0) + 0.0052 * k - 0.0526 + 47.0 / (200.0 * k)
    jj_low_q = 0.5
    jj_low_v = 0.5625 * k^(-0.528) / w
    jj_high_q = 0.64 + 0.02 * k
    jj_high_v = 0.765625 * k^(-0.44) / w
    jj_x_q = 3.7 * k^(-1.6)
    jj_x_v = jj_x_i
    aa_low_q = 25.0 / 48.0
    aa_low_v = 77.0 / (100.0 * w) * k^(-0.7)
    aa_high_q = 441.0 * k^(-5.76) + 0.55
    aa_high_v = var_q * var_r
    aa_x_q = 1.4 * k^(-1.15)
    aa_x_v = 1.22 * k^(-1.136) + 0.007

    rho_v = besselk(0, 1.0 / w) / besselk(2, 1.0 / w)
    r = _kappa_rotativity(k, w)
    T = typeof(frac)
    return Kappa{T}(frac, k, w, jj_low, jj_high, jj_x_i, aa_low, aa_high, aa_high_i, aa_x_i,
                    jj_low_q, jj_low_v, jj_high_q, jj_high_v, jj_x_q, jj_x_v,
                    aa_low_q, aa_low_v, aa_high_q, aa_high_v, aa_x_q, aa_x_v, rho_v,
                    r.frac, r.q_low_a, r.q_low_b, r.q_low_c, r.q_low_d, r.q_low_e,
                    r.q_high_a, r.q_high_b, r.q_high_c, r.q_high_d, r.q_high_e,
                    r.v_low_a, r.v_low_b, r.v_high_a, r.v_high_b)
end

Kappa{T}(kp::Kappa) where {T} =
    Kappa{T}(ntuple(i -> T(getfield(kp, i)), Val(fieldcount(Kappa)))...)

# The rotativity fits of the two tabulated kappa indices bracketing `k` (3.5, 4, 4.5, 5)
# and the linear blend weight between them.
function _kappa_rotativity(k, w)
    if k < 4.0
        return (frac = (k - 3.5) / (4.0 - 3.5),
                q_low_a = 17.0 * w + sqrt(w) * (-3.0 + 7.0 * exp(-5.0 * w)),
                q_low_b = -1.0 / 30.0, q_low_c = 0.1, q_low_d = -1.5, q_low_e = 0.471,
                q_high_a = 46.0 / 3.0 * w + sqrt(w) * (-5.0 / 3.0 + 17.0 / 3.0 * exp(-5.0 * w)),
                q_high_b = -1.0 / 18.0, q_high_c = 1.0 / 6.0, q_high_d = -1.75, q_high_e = 0.5,
                v_low_a = (w * w + 2.0 * w + 1.0) / (3.125 * w * w + 4.0 * w + 1.0),
                v_low_b = 0.447,
                v_high_a = (w * w + 54.0 * w + 50.0) / (30.0 / 11.0 * w * w + 134.0 * w + 50.0),
                v_high_b = 0.391)
    elseif k < 4.5
        return (frac = (k - 4.0) / (4.5 - 4.0),
                q_low_a = 46.0 / 3.0 * w + sqrt(w) * (-5.0 / 3.0 + 17.0 / 3.0 * exp(-5.0 * w)),
                q_low_b = -1.0 / 18.0, q_low_c = 1.0 / 6.0, q_low_d = -1.75, q_low_e = 0.5,
                q_high_a = 14.0 * w + sqrt(w) * (-1.625 + 4.5 * exp(-5.0 * w)),
                q_high_b = -1.0 / 12.0, q_high_c = 0.25, q_high_d = -2.0, q_high_e = 0.525,
                v_low_a = (w * w + 54.0 * w + 50.0) / (30.0 / 11.0 * w * w + 134.0 * w + 50.0),
                v_low_b = 0.391,
                v_high_a = (w * w + 43.0 * w + 38.0) / (7.0 / 3.0 * w * w + 92.5 * w + 38.0),
                v_high_b = 0.348)
    else
        return (frac = (k - 4.5) / (5.0 - 4.5),
                q_low_a = 14.0 * w + sqrt(w) * (-1.625 + 4.5 * exp(-5.0 * w)),
                q_low_b = -1.0 / 12.0, q_low_c = 0.25, q_low_d = -2.0, q_low_e = 0.525,
                q_high_a = 12.5 * w + sqrt(w) * (-1.0 + 5.0 * exp(-5.0 * w)),
                q_high_b = -0.125, q_high_c = 0.375, q_high_d = -2.25, q_high_e = 0.541,
                v_low_a = (w * w + 43.0 * w + 38.0) / (7.0 / 3.0 * w * w + 92.5 * w + 38.0),
                v_low_b = 0.348,
                v_high_a = (w + 13.0 / 14.0) / (2.0 * w + 13.0 / 14.0),
                v_high_b = 0.313)
    end
end

"""
    TransferParams{T,P,K,E,C,N}
    TransferParams(units, electrons, cuts, nu_obs; powerlaw = nothing, kappa = nothing)

The parameter set a march reads at every sample: the code-unit to cgs conversions
`units::Units{T}` (its length unit `L_unit` turns an affine step into a cgs path
length, `ds = h L_unit / ν_obs` for `k_t = -1` at the camera), the
`electrons::ElectronModel` (its `thermal_frac` is the thermal population fraction),
the plasma `cuts::PlasmaCuts`, the optional nonthermal populations
`powerlaw::Union{Nothing,PowerLaw}` and `kappa::Union{Nothing,Kappa}` (`nothing`
removes a population at compile time) and the observing frequency `nu_obs` (Hz).
The camera polarization vertical is the program's `CameraFrame.up`, not a field
here.

`isbits` whenever its fields are. Each sub-record keeps its own number type, so a
derivative enters only where it is seeded: `Units(m_bh, Dual(mass_scale))` gives
`T = Dual` with the electron model, the populations and `nu_obs` left `Float64`; a
`Dual` `rat_high` makes `E = ElectronModel{Dual}`. A mass-scale ladder is an
`NTuple{K}` of sets differing only in `units`.
"""
struct TransferParams{T,P<:Union{Nothing,PowerLaw},K<:Union{Nothing,Kappa},E<:ElectronModel,
                      C<:PlasmaCuts,N<:Real}
    units::Units{T}
    electrons::E
    cuts::C
    powerlaw::P
    kappa::K
    nu_obs::N
end

TransferParams(units::Units, electrons::ElectronModel, cuts::PlasmaCuts, nu_obs::Real;
               powerlaw = nothing, kappa = nothing) =
    TransferParams(units, electrons, cuts, powerlaw, kappa, nu_obs)

"""
    has_powerlaw(p::TransferParams) -> Bool

Whether the power-law population contributes: false at compile time when
`p.powerlaw` is `nothing`, else `p.powerlaw.frac != 0`.
"""
@inline has_powerlaw(p::TransferParams) = _active(p.powerlaw)

"""
    has_kappa(p::TransferParams) -> Bool

Whether the kappa population contributes: false at compile time when `p.kappa` is
`nothing`, else `p.kappa.frac != 0`.
"""
@inline has_kappa(p::TransferParams) = _active(p.kappa)

@inline _active(::Nothing) = false
@inline _active(pop::Union{PowerLaw,Kappa}) = pop.frac != 0.0

_numtype(::Type{Nothing}) = Union{}
_numtype(::Type{Units{T}}) where {T} = T
_numtype(::Type{ElectronModel{T}}) where {T} = T
_numtype(::Type{PowerLaw{T}}) where {T} = T
_numtype(::Type{Kappa{T}}) where {T} = T

# Number type of the zero a coefficient function returns for a sample that neither
# emits nor absorbs: the promotion of the unit and electron-model types.
@inline _zero_type(::TransferParams{T,P,K,E}) where {T,P,K,E} = promote_type(T, _numtype(E))

"""
    output_type(mode, params) -> Type

The value type of `march(mode, y0, prog)` for `prog.params == params`:

- `StokesI`, one `TransferParams`: the promotion of the number types of its units,
  electron model, populations and frequency (`Float64`, or a `ForwardDiff.Dual`
  carrying parameter derivatives);
- `StokesI`, an `NTuple{K}` of sets: `SVector{K,Float64}`, one intensity per set;
- `IQUV`: `SVector{4,Float64}`, the camera-frame `(I, Q, U, V)`.
"""
output_type(::StokesI, ::TransferParams{T,P,K,E,C,N}) where {T,P,K,E,C,N} =
    promote_type(T, _numtype(E), _numtype(P), _numtype(K), N)
output_type(::StokesI, ::NTuple{K,TransferParams}) where {K} = SVector{K,Float64}
output_type(::IQUV, ::TransferParams) = SVector{4,Float64}
