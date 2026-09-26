# Radiation API

The `Radiation` module converts sampled plasma state into invariant transfer
coefficients and propagates Stokes parameters along a ray. A single
plasma-decode chain converts code-unit $(\rho, p_\mathrm{gas}, b^2)$ into cgs
$(n_e, \Theta_e, B, \sigma, \beta^{-1})$, and the same chain feeds both imaging
and Monte Carlo.

```@docs
BlackLightPlus.Radiation
```

## Plasma decode and electron temperature

```@docs
SimCoeffConfig
electron_theta
plasma_cut_omits
BlackLightPlus.Radiation.plasma_decode
```

## Tetrads

```@docs
fluid_tetrad
BlackLightPlus.Radiation.fluid_tetrad_checked
```

## Transfer coefficients

`thermal_synchrotron_polarized` and its siblings return the invariant
emissivities, absorptivities, and rotativities
$(j_{I,Q,V};\ \alpha_{I,Q,V};\ \rho_{Q,V})$ in the fluid frame. Three electron
distribution functions mix additively: thermal
[Dexter2016, Marszewski2021](@citep), power-law, and kappa
[Pandya2016](@citep).

```@docs
thermal_synchrotron
thermal_synchrotron_polarized
power_law_synchrotron_polarized
kappa_synchrotron_polarized
```

### Host-side precomputation

Special functions that are expensive on device — Bessel $K_\nu$, $\Gamma$,
hypergeometric ${}_2F_1$ — are evaluated once on the host so that the per-step
kernel code stays closed-form.

```@docs
precompute_power_law!
precompute_kappa!
```

### Device-safe special functions

```@docs
bessel_k0
bessel_k1
bessel_k2
harmonic_bridge
```

## Stokes coupling

The analytic constant-coefficient segment solution of the polarized transfer
equation [LandiDeglInnocenti1985](@citet), covering emission-only, pure
absorption, pure Faraday rotation, and the full absorption-plus-rotation
eigenstructure.

```@docs
stokes_couple
couple_unpolarized
BlackLightPlus.Radiation.stokes_segment
BlackLightPlus.Radiation.unpolarized_segment
BlackLightPlus.Radiation.stokes_admissible
```

## Marching along a ray

The per-segment drivers that consume the coefficients above. Exported by
`Radiation` but not re-exported at the top level; the `render` drivers call
them for you.

```@docs
BlackLightPlus.Radiation.eight_coefficients
BlackLightPlus.Radiation.unpolarized_coefficients
BlackLightPlus.Radiation.coefficient_params
BlackLightPlus.Radiation.unpolarized_march
BlackLightPlus.Radiation.polarized_march
```

### Reverse marches

One march per transfer mode, shared by the CPU drivers and every fused kernel
([the transfer scheme](@ref "Radiative transfer")). A march control supplies the
geodesic: a fixed-control Kerr march, a generic spacetime march, or the stored
states of a traced path.

```@docs
BlackLightPlus.Radiation.reverse_unpolarized_march
BlackLightPlus.Radiation.reverse_polarized_march
BlackLightPlus.Radiation.KerrMarch
BlackLightPlus.Radiation.SpacetimeMarch
BlackLightPlus.Radiation.StoredPath
```

## Equatorial disk source

```@docs
EquatorialDisk
equatorial_disk_intensity
gfactor_disk
gfactor_circular
gfactor_plunge
johnson_su_profile
```
