# Monte Carlo — sphere_model.jl
#
# Uniform, optically-thin thermal-synchrotron sphere: the end-to-end MC
# validation model (gpumonty sphere_model). A ball of radius R of static,
# uniform plasma (n_e, Theta_e, B) in flat Minkowski spacetime, its escaping
# spectrum checked against the closed-form thin-sphere emissivity. Reference
# point (n_e = 1e13 cm^-3, Theta_e = 100, B = 1 G, R = 1 cm); B points along x1
# rather than the reference's x3 to avoid the Gram-Schmidt degeneracy of
# `emission_tetrad` when b-hat coincides with x3 (physics identical by isotropy).

# ============================================================ SphereModel

"""
    SphereModel

Uniform optically-thin synchrotron sphere (`isbits`). Carries the plasma state,
radius, the emitting volume `dV = (4/3) pi R^3`, and the (position-independent)
flat-space kinematics: static fluid `ucon`/`ucov`, magnetic-field direction
`bcon` (along x1), and the Minkowski metric `gcov`/`gcon`.
"""
struct SphereModel{ST<:AbstractSpacetime}
    spacetime::ST
    n_e::Float64
    theta_e::Float64
    B::Float64
    R::Float64
    dV::Float64
    ucon::SVector{4,Float64}
    ucov::SVector{4,Float64}
    bcon::SVector{4,Float64}
    gcov::SMatrix{4,4,Float64,16}
    gcon::SMatrix{4,4,Float64,16}
end

"""
    SphereModel(; n_e=1e13, theta_e=100, B=1, R=1) -> SphereModel

Construct the sphere at the gpumonty reference point in flat Minkowski spacetime,
with a static fluid and B along x1.
"""
function SphereModel(; n_e::Real = 1.0e13, theta_e::Real = 100.0,
                     B::Real = 1.0, R::Real = 1.0)
    st = Minkowski()
    x0 = SVector(0.0, 0.0, 0.0, 0.0)
    g = SMatrix{4,4,Float64}(metric(st, x0))          # eta
    gi = SMatrix{4,4,Float64}(inverse_metric(st, x0))  # eta
    ucon = SVector(1.0, 0.0, 0.0, 0.0)                 # static fluid
    ucov = g * ucon                                    # (-1,0,0,0)
    bcon = SVector(0.0, 1.0, 0.0, 0.0)                 # B along x1
    dV = (4.0 / 3.0) * π * Float64(R)^3
    return SphereModel(st, Float64(n_e), Float64(theta_e), Float64(B),
                       Float64(R), dV, ucon, ucov, bcon, g, gi)
end

"""
    sphere_zone(model, x) -> EmissionZone

The emitting zone at coordinate `x`: the sphere's uniform plasma and flat-space
kinematics with the whole-sphere volume `dV`. (For weight-table purposes one such
zone represents the entire sphere; for per-photon generation only `x` varies.)
"""
@inline function sphere_zone(model::SphereModel, x::SVector{4,Float64})
    return EmissionZone(model.n_e, model.theta_e, model.B, model.dV,
                        x, model.ucon, model.ucov, model.bcon,
                        model.gcov, model.gcon)
end

"""
    sample_sphere_photon(model, wgt, dnmax, rng[, gt]) -> (Superphoton, MCRng)

Generate one superphoton: draw a uniform-random position in the ball of radius `R`
(`r = R U^{1/3}`, isotropic direction), build the local `EmissionZone`, and
`sample_zone_photon`. Allocation-free; returns the counter-advanced RNG. `gt`
(with its matching `tabulated_dnmax` as `dnmax`) selects the opt-in tabulated
frequency rejection — see `sample_zone_photon`.
"""
@inline function sample_sphere_photon(model::SphereModel, wgt::WeightTable,
                                      dnmax::Real, rng::MCRng,
                                      gt::Union{Nothing,IntJnuTable} = nothing)
    u, rng = mc_uniform(rng)
    r = model.R * cbrt(u)                         # uniform density in the ball
    dir, rng = sample_direction(rng)
    x = SVector(0.0, r * dir[1], r * dir[2], r * dir[3])
    sp, rng = sample_zone_photon(sphere_zone(model, x), wgt, dnmax, rng, gt)
    return sp, rng
end

"""
    sample_sphere_photon(model, wgt, rng) -> (Superphoton, MCRng)

Convenience form that derives the frequency-rejection envelope `dnmax` from the
sphere zone before sampling. Prefer the 4-argument form in a hot loop.
"""
@inline function sample_sphere_photon(model::SphereModel, wgt::WeightTable, rng::MCRng)
    _, dnmax = zone_nsuperphotons(sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0)), wgt)
    return sample_sphere_photon(model, wgt, dnmax, rng)
end

"""
    analytic_nuLnu(model, nu) -> Float64

Closed-form optically-thin thin-sphere spectrum [erg s^-1]:

    nu L_nu = nu * V * int_jnu_zone(n_e, Theta_e, B, nu),   V = (4/3) pi R^3,

the frequency-times-volume-times-(angle-integrated)-emissivity, in the SAME
angle-integrated (total) convention as `nuLnu_total`. This is the gpumonty python
checker's `Luminosity_analytic * nu * dv` (its `int_jnu` equals `int_jnu_zone`).
"""
@inline analytic_nuLnu(model::SphereModel, nu::Real) =
    nu * model.dV * int_jnu_zone(model.n_e, model.theta_e, model.B, nu)

# ============================================================ ScatteringSphere

"""
    ScatteringSphere(; n_e=1e18, theta_e=3, B=100, R=3e5) -> SphereModel

The idealized Comptonization sphere (arXiv:2602.13198 §3.2 regime): a uniform
thermal plasma that emits thermal-synchrotron seed photons and
Compton-scatters them, sized so `tau_es = n_e sigma_T R ~ 0.2` is moderate
(successive up-scattering bumps visible). Returns a plain `SphereModel`; the
scattering transport is selected by `mc_spectrum_scatter`. Synchrotron
self-absorption is neglected in the scattering transport (it exceeds the
Thomson opacity by ~1e9-1e12 at the emission peak, so a moderate-`tau_es`
sphere has no closed-form seed); `analytic_nuLnu` is then the direct emission
the unscattered `nscatt=0` spectrum reduces to, up to `exp(-tau_scatt)`.
"""
ScatteringSphere(; n_e::Real = 1.0e18, theta_e::Real = 3.0,
                 B::Real = 100.0, R::Real = 3.0e5) =
    SphereModel(; n_e = n_e, theta_e = theta_e, B = B, R = R)

"""
    scattering_tau_es(model) -> Float64

Radial Thomson scattering optical depth `tau_es = n_e sigma_T R` of the sphere (the
soft-photon limit `sigma_h -> sigma_T`; the geometry-averaged escape depth over
uniform isotropic emission is smaller, `<tau> ~ 0.75 tau_es`).
"""
@inline scattering_tau_es(model::SphereModel) = model.n_e * SIGMA_THOMSON * model.R
