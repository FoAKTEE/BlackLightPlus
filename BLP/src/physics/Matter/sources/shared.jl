# Kinematics shared by the analytic sources, in the Cartesian Kerr-Schild frame with
# signature (−,+,+,+) and G = c = M = 1: the axisymmetric Killing vectors, circular
# flows, the constant-ℓ potential of Abramowicz, Jaroszyński & Sikora (1978), the
# toroidal field direction, the plasma-β field strength, and the state record every
# analytic source returns. Module-private.

"""
    killing_vectors(x::SVector{4}) -> (xi, eta)

The Killing vectors of a stationary axisymmetric spacetime, contravariant in the CKS
frame at `x`: `ξ = ∂_t = (1, 0, 0, 0)` and `η = ∂_φ = x ∂_y − y ∂_x = (0, −y, x, 0)`.
"""
@inline function killing_vectors(x::SVector{4,T}) where {T}
    xi  = SVector(one(T), zero(T), zero(T), zero(T))
    eta = SVector(zero(T), -x[3], x[2], zero(T))
    return xi, eta
end

"""
    tphi_scalars(gcov, x) -> (tt, tp, pp)

The Killing-projected metric scalars `X_tt = g(ξ, ξ)`, `X_tφ = g(ξ, η)` and
`X_φφ = g(η, η)` at `x` from the covariant metric `gcov` in the frame of `x`. They
equal the Boyer–Lindquist `g_tt`, `g_tφ`, `g_φφ`, the Killing vectors being geometric.
"""
@inline function tphi_scalars(gcov::SMatrix{4,4}, x::SVector{4})
    xi, eta = killing_vectors(x)
    g_xi  = gcov * xi
    g_eta = gcov * eta
    return (tt = dot(xi, g_xi), tp = dot(xi, g_eta), pp = dot(eta, g_eta))
end

"""
    circular_velocity(gcov, x, Omega) -> (ucon, ucov, den)

Four-velocity `u^μ = u^t (ξ + Ω η)` of the circular flow of coordinate angular
velocity `Omega` at `x`, normalised to `u·u = −1`, its covariant form, and
`den = X_tt + 2Ω X_tφ + Ω² X_φφ = −1/(u^t)²`. The flow exists only where `den < 0`;
for `den > 0` the real square root throws a `DomainError`, and `den = 0` gives an
infinite `u^t`.
"""
@inline function circular_velocity(gcov::SMatrix{4,4}, x::SVector{4}, Omega)
    xi, eta = killing_vectors(x)
    ubar = xi + Omega * eta
    den = dot(ubar, gcov * ubar)
    ut = 1 / sqrt(-den)
    ucon = ut * ubar
    ucov = gcov * ucon
    return (ucon = ucon, ucov = ucov, den = den)
end

"""
    constant_l_potential(Xtt, Xtp, Xpp, l0) -> (W, Omega, den)

Effective potential of a flow of constant specific angular momentum
`ℓ = −u_φ/u_t = l0`, in the publication sign of Abramowicz, Jaroszyński & Sikora
(1978): `W = ln|u_t|`, minimal at the pressure maximum and increasing outward. With
`Ω = −(X_tφ + ℓ X_tt)/(X_φφ + ℓ X_tφ)` the flow's angular velocity,
`W = ln|X_tt + Ω X_tφ| − ½ ln|den|` and `den = X_tt + 2Ω X_tφ + Ω² X_φφ`, negative
where the flow is defined.
"""
@inline function constant_l_potential(Xtt, Xtp, Xpp, l0)
    Omega = -(Xtp + l0 * Xtt) / (Xpp + l0 * Xtp)
    num = Xtt + Omega * Xtp
    den = Xtt + 2 * Omega * Xtp + Omega^2 * Xpp
    W = log(abs(num)) - 0.5 * log(abs(den))
    return (W = W, Omega = Omega, den = den)
end

"""
    equatorial_point(a, r) -> SVector{4}

The CKS point at `t = 0` on the equator `z = 0`, azimuth 0, at Boyer–Lindquist radius
`r`: its cylindrical radius is `√(r² + a²)`, from `(x² + y²)/(r² + a²) + z²/r² = 1`.
"""
@inline equatorial_point(a, r) = SVector(zero(r), sqrt(r^2 + a^2), zero(r), zero(r))

"""
    toroidal_direction(gcov, x, ell, ucon, ucov) -> SVector{4}

Unit spacelike field direction (`b·b = +1`, `b·u = 0`) of a toroidal field for a flow
of specific angular momentum `ell`: `b ∝ ℓ ξ + η` projected orthogonal to `u`. On the
rotation axis `η → 0` and the projection degenerates (`b·b ≤ 1e-10`); `∂_x` projected
orthogonal to `u` is used instead, so `b` stays finite. The pitch angle is undefined
there and only the scalar field strength carries weight.
"""
@inline function toroidal_direction(gcov::SMatrix{4,4}, x::SVector{4},
                                    ell, ucon::SVector{4}, ucov::SVector{4})
    xi, eta = killing_vectors(x)
    bdir = ell * xi + eta
    bu = dot(gcov * bdir, ucon)
    bp = bdir + bu * ucon
    n2 = dot(bp, gcov * bp)
    if !(n2 > 1e-10)
        T = eltype(bp)
        v = SVector(zero(T), one(T), zero(T), zero(T))
        bp = v + dot(gcov * v, ucon) * ucon
        n2 = dot(bp, gcov * bp)
    end
    return bp / sqrt(n2)
end

"""
    bisect(f, lo, hi; tol = 1e-12, itmax = 200) -> Float64

Root of `f` on the bracket `[lo, hi]`, across which `f` changes sign, by bisection to
an interval narrower than `tol` or `itmax` halvings: the midpoint of the last
bracket. Host-side source geometry.
"""
function bisect(f, lo::Float64, hi::Float64; tol::Float64 = 1e-12, itmax::Int = 200)
    flo = f(lo)
    for _ in 1:itmax
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0
            hi = mid
        else
            lo = mid
            flo = fm
        end
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

"""
    gas_pressure(n_e, theta_e; f_ion = 1.0)

Gas pressure (erg cm⁻³) of electrons of density `n_e` (cm⁻³) and temperature
`theta_e` plus `f_ion n_e` ions at the same temperature: `p = (1 + f_ion) n_e Θₑ mₑc²`.
"""
@inline gas_pressure(n_e, theta_e; f_ion::Float64 = 1.0) =
    (1.0 + f_ion) * n_e * theta_e * (ME * CL * CL)

"""
    bfield_from_beta(p_gas, beta)

Field strength (G) at plasma beta `beta = p_gas/p_mag` with `p_mag = B²/8π`:
`B = √(8π p_gas/β)`.
"""
@inline bfield_from_beta(p_gas, beta) = sqrt(8.0 * π * p_gas / beta)

"""
    param_type(src) -> Type

Number type of the parameters of an analytic provider.
"""
function param_type end

"""
    analytic_type(src, x, gcov) -> Type

Number type of the `PlasmaState` of `src` at `x` with covariant metric `gcov`: the
promotion of the parameter, position and metric types, so that a `ForwardDiff.Dual`
parameter or position gives one concrete state type on every branch.
"""
@inline analytic_type(src, x::SVector{4,Tx}, gcov::SMatrix{4,4,Tg}) where {Tx,Tg} =
    promote_type(param_type(src), Tx, Tg)

"""
    analytic_state(T, n_e, theta_e, B, ucon, ucov, bcon, gcov) -> PlasmaState{T}

The valid state of an analytic source: density `n_e` (cm⁻³), temperature `theta_e`,
field strength `B` (G), four-velocity `ucon`/`ucov` and unit field direction `bcon`,
with `bcov = gcov bcon` and `b_sq = b^μ b_μ`, unity to rounding. `sigma` and
`beta_inv` are zero: an analytic source has no code-unit rest-mass density.
"""
@inline function analytic_state(::Type{T}, n_e, theta_e, B, ucon::SVector{4}, ucov::SVector{4},
                                bcon::SVector{4}, gcov::SMatrix{4,4}) where {T}
    bcov = gcov * bcon
    b_sq = dot(bcon, bcov)
    o = zero(T)
    return PlasmaState{T}(n_e, theta_e, B, ucon, ucov, bcon, bcov, b_sq, o, o, true)
end

"""
    radius_above(r) -> Float64

The smallest integer strictly above the derivative-free value of `r`: the domain
radius of a source whose outermost emitting radius is `r` up to rounding.
"""
@inline radius_above(r) = floor(Float64(primal(r))) + 1.0
