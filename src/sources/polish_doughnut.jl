# Constant-angular-momentum equilibrium torus ("Polish doughnut"): Abramowicz,
# Jaroszyński & Sikora 1978 (A&A 63, 221) with Komissarov 2006 (MNRAS 368, 993)
# polytropic thermodynamics; Gyoto lib/PolishDoughnut.C pins the conventions
# (equation-first; no code ported). A barotropic flow on constant l = -u_phi/u_t
# has isobars = equipotentials of W = ln|u_t| (publication sign): a minimum at
# the pressure-max centre and a saddle at the cusp, both where l = l_K(r); the
# torus is the closed region W < W_surface between r_cusp and r_out. Polytrope
# p = kappa rho^Gamma (Gamma = 1 + 1/n) with p << rho c^2 gives, for the depth
# w = Delta_W/Delta_W_centre ∈ (0,1]: n_e ∝ w^n, Theta_e ∝ w, |B| from plasma beta.

"""
    PolishDoughnut(a; l0=nothing, lambda=nothing, r_center=nothing,
                   n_e_center=1e7, theta_e_center=50.0, polytrope_n=1.5,
                   beta=10.0, W_surface=nothing, fill_factor=0.9, prograde=true)

Constant-l Abramowicz equilibrium torus around a Kerr hole of spin `a`. The
angular momentum comes from exactly one of `r_center` (l0 = l_K(r_center)), `l0`
directly, or `lambda ∈ (0,1)` (l0 = lambda (l_mb - l_ms) + l_ms), in that
priority order; default `lambda = 0.5`. A closed torus needs l_ms < l0 < l_mb.
`n_e_center`/`theta_e_center` set the central density [cm^-3] and temperature;
`polytrope_n` is the polytropic index n (Gamma = 1+1/n); `beta` sets |B| from
p_gas. `W_surface` overrides the surface potential (default: cusp potential,
Roche-lobe-filling); if the cusp overflows (l0 >= l_mb), `fill_factor ∈ (0,1)`
sets the surface that fraction down the central well so r_out stays finite.
Derived geometry is precomputed; the struct is `isbits`.
"""
struct PolishDoughnut{T} <: AbstractSource
    a::T
    l0::T
    polytrope_n::T
    beta::T
    n_e_center::T
    theta_e_center::T
    r_cusp::T
    r_center::T
    r_out::T
    W_surface::T
    W_center::T
    prograde::Bool
end

function PolishDoughnut(a::Real; l0=nothing, lambda=nothing, r_center=nothing,
                        n_e_center::Real=1e7, theta_e_center::Real=50.0,
                        polytrope_n::Real=1.5, beta::Real=10.0,
                        W_surface=nothing, fill_factor::Real=0.9, prograde::Bool=true)
    af = Float64(a)
    st = KerrSchildCartesian(af)
    r_isco = isco_radius(af; prograde=prograde)
    r_ph = photon_orbit_radius(af; prograde=prograde)
    lK(r) = keplerian_specific_angular_momentum(af, r; prograde=prograde)

    # marginally-stable / marginally-bound Keplerian l (bracket the bound-torus
    # window l_ms < l0 < l_mb; outside it the Roche lobe overflows -> no closed torus).
    l_ms = lK(r_isco)
    l_mb = lK(marginally_bound_radius(af; prograde=prograde))

    # 1) resolve the constant specific angular momentum l0 and the centre
    #    (priority r_center > l0 > lambda > default lambda = 0.5).
    local l0f, rc
    if r_center !== nothing
        rc = Float64(r_center)
        l0f = lK(rc)
    elseif l0 !== nothing
        l0f = Float64(l0)
        rc = _bisect(r -> lK(r) - l0f, r_isco, 1.0e5)          # outer (stable) root
    else
        lam = lambda === nothing ? 0.5 : Float64(lambda)
        l0f = lam * (l_mb - l_ms) + l_ms
        rc = _bisect(r -> lK(r) - l0f, r_isco, 1.0e5)
    end

    # Closed/overflowing-torus window. l_K(r) has its extremum l_ms at the ISCO, so
    # l_K(r) = l0 has a cusp/centre root pair only when |l0| is at or beyond |l_ms|
    # (toward l_mb). For |l0| < |l_ms| there is no equilibrium torus and the _bisect
    # calls below would silently return a bracket endpoint (an unbounded "torus" with
    # density rising outward) — reject it. The overflow side (|l0| >= |l_mb|) is
    # supported via `fill_factor`.
    toward_mb = sign(l_mb - l_ms)                # +1 prograde, -1 retrograde
    (l0f - l_ms) * toward_mb > 0 || throw(ArgumentError(
        "PolishDoughnut: l0=$(l0f) admits no equilibrium torus — need |l0| beyond the " *
        "marginally-stable l_ms=$(l_ms) toward l_mb=$(l_mb) " *
        "($(prograde ? "prograde" : "retrograde") branch)."))

    # 2) inner cusp: the other root of l_K(r) = l0, on (r_ph, r_isco).
    r_cusp = _bisect(r -> lK(r) - l0f, r_ph + 1e-6, r_isco)

    # 3) equatorial potentials at cusp and centre (publication sign W = ln|u_t|).
    Weq(r) = _constant_l_potential(_tphi_scalars(metric(st, _equatorial_point(af, r)),
                                                 _equatorial_point(af, r))..., l0f)[1]
    W_center = Weq(rc)
    W_cusp = Weq(r_cusp)
    # Torus surface. Default = the cusp potential (Roche-lobe-filling) when the
    # cusp is bound (W_cusp < 0, i.e. l0 < l_mb). If l0 >= l_mb the cusp overflows
    # (W_cusp >= 0) and W = W_cusp never closes at finite r; fall back to a
    # bound level a fraction `fill_factor` down the central well so r_out stays
    # finite (a standard free-inner-edge doughnut).
    Wsurf = W_surface !== nothing ? Float64(W_surface) :
            (W_cusp < 0 ? W_cusp : fill_factor * W_center)

    # 4) outer edge: W(r) climbs back to W_surface on the far side of the centre
    #    (W -> 0^- as r -> inf, and W_surface < 0), so the root exists in (rc, inf).
    r_out = _bisect(r -> Weq(r) - Wsurf, rc, 1.0e5)

    T = promote_type(typeof(af), Float64)
    return PolishDoughnut{T}(af, l0f, Float64(polytrope_n), Float64(beta),
                             Float64(n_e_center), Float64(theta_e_center),
                             r_cusp, rc, r_out, Wsurf, W_center, prograde)
end

"""
    torus_potential(src::PolishDoughnut, st, x::SVector{4}) -> W

Relativistic effective potential W = ln|u_t| of the constant-l flow at CKS
position `x` (publication sign; minimum at the pressure-max centre). Returns
`+Inf` where the constant-l flow is undefined (the fluid 4-velocity would be
imaginary).
"""
@inline function torus_potential(src::PolishDoughnut, st, x::SVector{4})
    gcov = metric(st, x)
    Xtt, Xtp, Xpp = _tphi_scalars(gcov, x)
    W, _, den = _constant_l_potential(Xtt, Xtp, Xpp, src.l0)
    return den < 0 ? W : convert(typeof(W), Inf)
end

function source_plasma_at(src::PolishDoughnut, st, x::SVector{4})
    gcov = metric(st, x)
    gcon = inverse_metric(st, x)
    r = ks_radius(st, x)
    r < src.r_cusp && return _vacuum(gcov, gcon)                 # inner funnel

    Xtt, Xtp, Xpp = _tphi_scalars(gcov, x)
    W, Omega, den = _constant_l_potential(Xtt, Xtp, Xpp, src.l0)
    den < 0 || return _vacuum(gcov, gcon)                        # flow undefined

    # potential depth normalized to the centre: w = 1 at centre, ->0 at surface.
    w = (src.W_surface - W) / (src.W_surface - src.W_center)
    w > 0 || return _vacuum(gcov, gcon)                          # outside the torus
    w = min(w, one(w))

    n_e = src.n_e_center * w^src.polytrope_n
    theta_e = src.theta_e_center * w
    p_gas = _gas_pressure(n_e, theta_e)
    B = _bfield_from_beta(p_gas, src.beta)

    ucon, ucov, _ = _circular_velocity(gcov, x, Omega)
    bcon = _toroidal_direction(gcov, x, src.l0, ucon, ucov)
    return _plasma(n_e, theta_e, B, ucon, ucov, bcon, gcov, gcon)
end
