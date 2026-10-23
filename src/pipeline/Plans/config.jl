# Run configuration: the declarative user input that `prepare` turns into a plan. The
# spacetime, the matter, the transfer mode and the precision lanes are values passed to
# `prepare`, not configuration fields.

"""
    CameraConfig(; inclination_deg = 60.0, distance = 1000.0, fov = 40.0, resolution = 64)

Camera geometry: observer inclination `inclination_deg` (degrees from the spin axis),
camera distance `distance` and full field of view `fov` (both in M = GM/c²), and
`resolution` pixels per side.
"""
Base.@kwdef struct CameraConfig
    inclination_deg::Float64 = 60.0
    distance::Float64 = 1000.0
    fov::Float64 = 40.0
    resolution::Int = 64
end

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
              C = 80.0, escape_factor = 1.1)

One run: the hole mass `m_bh_cgs` (g), the density mass scale `mass_scale_cgs` (g), the
observing frequency `frequency_hz` (Hz), the camera and the plasma model, and the march
budget: at most `max_steps` steps of affine length (distance to the horizon proxy)/`C`,
a ray escaping beyond `escape_factor × camera.distance`. The spacetime, the matter, the
transfer mode and the precision lanes are arguments of [`prepare`](@ref).
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
end
