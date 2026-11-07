# Run configuration: the declarative user input that `prepare` turns into a plan. The
# spacetime, the matter, the transfer mode and the precision lanes are values passed to
# `prepare`, not configuration fields.

"""
    CameraConfig(; inclination_deg = 60.0, distance = 1000.0, fov = 40.0, resolution = 64,
                 infinity = false)

Camera geometry: observer inclination `inclination_deg` (degrees from the spin axis),
camera distance `distance` and full field of view `fov` (both in M = GM/c²), and
`resolution` pixels per side. With `infinity = true` the observer is at infinity: the
image coordinates are Bardeen's, and each ray is carried in from infinity through the
far zone and started on the plane at `distance` (stationary spacetimes only).
"""
Base.@kwdef struct CameraConfig
    inclination_deg::Float64 = 60.0
    distance::Float64 = 1000.0
    fov::Float64 = 40.0
    resolution::Int = 64
    infinity::Bool = false
end
CameraConfig(inclination_deg, distance, fov, resolution) =
    CameraConfig(inclination_deg, distance, fov, resolution, false)

"""
    PlasmaConfig(; mu = 0.5, ne_ni = 1.0, use_p = true, rat_low = 1.0, rat_high = 10.0,
                 gamma = 13/9, gamma_i = 5/3, gamma_e = 4/3, thermal_frac = 1.0,
                 power_frac = 0.0, kappa_frac = 0.0, power_p = 3.5, gamma_min = 25.0,
                 gamma_max = 1e7, kappa = 4.0, w = 0.5, sigma_cut = 1.0,
                 theta_e_min = -1.0)

Electron thermodynamics and emission model: mean molecular weight `mu` (m_p), electrons
per ion `ne_ni`, whether the fluid pressure is the total pressure (`use_p`), the
`rat_low`/`rat_high` ion-to-electron temperature ratios at β → 0 and β → ∞, the fluid,
ion and electron adiabatic indices `gamma`, `gamma_i`, `gamma_e`; the population
fractions `thermal_frac`, `power_frac`, `kappa_frac`; the power-law index `power_p` and
Lorentz-factor cutoffs `gamma_min`, `gamma_max` (used when `power_frac != 0`); the kappa
index `kappa` and width `w` (used when `kappa_frac != 0`, fits valid for `kappa` in
[3.5, 5]); and the emission cuts `sigma_cut` (largest magnetisation emitted) and
`theta_e_min` (smallest electron temperature emitted). A negative cut is off.
"""
Base.@kwdef struct PlasmaConfig
    mu::Float64 = 0.5
    ne_ni::Float64 = 1.0
    use_p::Bool = true
    rat_low::Float64 = 1.0
    rat_high::Float64 = 10.0
    gamma::Float64 = 13.0 / 9.0
    gamma_i::Float64 = 5.0 / 3.0
    gamma_e::Float64 = 4.0 / 3.0
    thermal_frac::Float64 = 1.0
    power_frac::Float64 = 0.0
    kappa_frac::Float64 = 0.0
    power_p::Float64 = 3.5
    gamma_min::Float64 = 25.0
    gamma_max::Float64 = 1.0e7
    kappa::Float64 = 4.0
    w::Float64 = 0.5
    sigma_cut::Float64 = 1.0
    theta_e_min::Float64 = -1.0
end

"""
    RunConfig(; m_bh_cgs = 6.2e9 × 1.989e33, mass_scale_cgs = 1e24, frequency_hz = 230e9,
              camera = CameraConfig(), plasma = PlasmaConfig(), max_steps = 2500,
              C = 80.0, escape_factor = 1.1, integrator = :rk4, capture_redshift = 100.0,
              horizon_eps = 1e-3, winding = false)

One run: the hole mass `m_bh_cgs` (g), the density mass scale `mass_scale_cgs` (g), the
observing frequency `frequency_hz` (Hz), the camera and the plasma model, and the march
budget: at most `max_steps` steps of affine length (distance to the horizon proxy)/`C`
taken by `integrator` (`:rk4`, the classical fourth-order step, or `:verner9`, the
ninth-order step), a ray escaping beyond `escape_factor × camera.distance` and captured
where its normal-frame energy exceeds `capture_redshift` (camera energy 1) or it comes
within `horizon_eps` of the horizon proxy, and with `winding` the step near a spinning
horizon shortened to resolve the frame-dragged winding (`winding_factor`).
[`precise`](@ref) gives the precise mode of a
configuration. The spacetime, the matter, the transfer mode and the precision lanes are
arguments of [`prepare`](@ref).
"""
Base.@kwdef struct RunConfig
    m_bh_cgs::Float64 = 6.2e9 * 1.989e33
    mass_scale_cgs::Float64 = 1.0e24
    frequency_hz::Float64 = 230.0e9
    camera::CameraConfig = CameraConfig()
    plasma::PlasmaConfig = PlasmaConfig()
    max_steps::Int = 2500
    C::Float64 = 80.0
    escape_factor::Float64 = 1.1
    integrator::Symbol = :rk4
    capture_redshift::Float64 = 100.0
    horizon_eps::Float64 = 1.0e-3
    winding::Bool = false
end

"Step divisor of the precise mode ([`precise`](@ref))."
const PRECISE_C = 20.0
"Redshift cut of the precise mode ([`precise`](@ref))."
const PRECISE_CAPTURE = 1.0e6

"""
    precise(cfg::RunConfig = RunConfig(); infinity = true, C = PRECISE_C) -> RunConfig

The precise mode of `cfg`: the `:verner9` march at step divisor `C` with the step
resolving the winding near a spinning horizon, the redshift cut at `PRECISE_CAPTURE` with
no horizon backstop (`horizon_eps = 0`), a step limit of at least 10⁵, and, with
`infinity = true`, the camera at infinity (on a stationary spacetime; pass
`infinity = false` for a time-dependent one). Every other field is `cfg`'s. The geodesic
of every product follows the ninth-order step; a volumetric render samples its plasma
once per step, so its accuracy is set by `C`, not by the integrator, and it keeps the
production divisor or a larger one.
"""
function precise(cfg::RunConfig = RunConfig(); infinity::Bool = true, C::Real = PRECISE_C)
    c = cfg.camera
    cam = CameraConfig(c.inclination_deg, c.distance, c.fov, c.resolution, infinity || c.infinity)
    return RunConfig(cfg.m_bh_cgs, cfg.mass_scale_cgs, cfg.frequency_hz, cam, cfg.plasma,
                     max(cfg.max_steps, 100_000), Float64(C), cfg.escape_factor, :verner9,
                     PRECISE_CAPTURE, 0.0, true)
end
