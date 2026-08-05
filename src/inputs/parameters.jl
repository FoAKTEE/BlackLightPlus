"""
    Inputs

Run configuration as typed Julia structs (blacklight `inputs/parameters.hpp`
heritage); callers construct these directly, keyword-style, in place of the
C++ key=value input deck.
"""
module Inputs

export CameraConfig, PlasmaConfig, RunConfig

"Camera geometry: observer inclination (degrees from the spin axis), camera distance and full field of view (both `GM/c²`), and pixels per side."
Base.@kwdef struct CameraConfig
    inclination_deg::Float64 = 60.0     # observer inclination from spin axis
    distance::Float64        = 1000.0   # camera distance [GM/c^2]
    fov::Float64             = 40.0     # full field of view [GM/c^2]
    resolution::Int          = 64       # pixels per side
end

"Electron thermodynamics + emission model: the R_low/R_high β-dependent T_i/T_e prescription, composition (`mu`, `ne_ni`), adiabatic indices, the thermal/power/kappa emission fractions, and the σ / Θ_e emission cuts."
Base.@kwdef struct PlasmaConfig
    mu::Float64          = 0.5          # mean molecular weight [m_p]
    ne_ni::Float64       = 1.0          # electrons per ion
    use_p::Bool          = true         # match total pressure (vs internal energy)
    rat_low::Float64     = 1.0          # T_i/T_e at beta -> 0 (R_low)
    rat_high::Float64    = 10.0         # T_i/T_e at beta -> inf (R_high)
    gamma::Float64       = 13.0 / 9.0   # fluid adiabatic index
    gamma_i::Float64     = 5.0 / 3.0
    gamma_e::Float64     = 4.0 / 3.0
    thermal_frac::Float64 = 1.0
    power_frac::Float64  = 0.0
    kappa_frac::Float64  = 0.0
    # Nonthermal power-law tail (active only when power_frac != 0); consumed by
    # `precompute_power_law!`.
    power_p::Float64     = 3.5          # power-law electron index p
    gamma_min::Float64   = 25.0         # low-energy Lorentz cutoff gamma_min
    gamma_max::Float64   = 1.0e7        # high-energy Lorentz cutoff gamma_max
    # Nonthermal kappa distribution (active only when kappa_frac != 0); consumed by
    # `precompute_kappa!` (fits valid for kappa in [3.5, 5]). Defaults follow the
    # blacklight (White 2022) fiducial κ=4, w=1/2.
    kappa::Float64       = 4.0          # kappa-distribution index κ
    w::Float64           = 0.5          # kappa-distribution width w
    sigma_cut::Float64   = 1.0          # sigma_max condition cut (blacklight convention)
    theta_e_min::Float64 = -1.0         # emission floor on Theta_e (-1 disables)
end

"""
Top-level run configuration (the object a caller builds and hands to `render`): black-hole
spin `bh_a` and mass, the density mass-scale, observing frequency, the nested `CameraConfig`
and `PlasmaConfig`, the step budget, the polarization toggle, and the geodesic `tracer`
(`:numerical` ODE or `:analytic` exact-Kerr).
"""
Base.@kwdef struct RunConfig
    bh_a::Float64            = 0.9
    m_bh_cgs::Float64        = 6.2e9 * 1.989e33   # M_BH [g]
    mass_scale_cgs::Float64  = 1.0e24             # density-unit mass scale [g]
    frequency_hz::Float64    = 230.0e9
    camera::CameraConfig     = CameraConfig()
    plasma::PlasmaConfig     = PlasmaConfig()
    max_steps::Int           = 2500
    polarized::Bool          = true
    tracer::Symbol           = :numerical         # :numerical (ODE) | :analytic (AART/Krang exact-Kerr)
end

end # module
