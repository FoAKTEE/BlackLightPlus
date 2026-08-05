# Monte Carlo API

The `MonteCarlo` module implements the grmonty scheme [Dolence2009](@citep) for
Compton-scattering spectral synthesis. Superphoton packets sample the
synchrotron emissivity zone by zone, carry covariant momentum along geodesics,
decay by Kirchhoff absorption, and scatter off a relativistic Maxwellian.

## Random numbers

The counter-based Philox4x32-10 generator [Salmon2011](@citep). It is
stateless and keyed per superphoton, so GPU runs replay the CPU result
photon-for-photon at matched seeds.

```@docs
philox4x32
philox_rand
MCRng
mc_uniform
mc_gauss
sample_direction
standard_gamma
chi_square
```

## Superphotons

```@docs
Superphoton
photon_position
photon_momentum
photon_weight
photon_energy
photon_nscatt
```

## Emission

```@docs
j_nu
alpha_nu
jmax
int_jnu_zone
emission_tetrad
```

## Emission zones and weight tables

Frequency and pitch-angle rejection sampling against a log-frequency weight
table, which sets how many superphotons each zone emits.

```@docs
EmissionZone
WeightTable
weight_at
build_weight_table
zone_nsuperphotons
sample_zone_photon
ZoneEmissionTable
build_zone_emission_table
zone_counts
```

## Compton cross sections

The thermally averaged ("hot") Klein–Nishina cross section, tabulated on a
log-log $(\varepsilon, \Theta_e)$ grid. Building the table is the expensive
setup step — build once and reuse.

```@docs
kn_total
boostcross
dNdgamma_e
total_compton_cross_num
HotcrossTable
build_hotcross_table
load_hotcross
hotcross_lookup
kappa_es
alpha_inv_scatt
```

## Scattering

Electrons are drawn from the relativistic Maxwellian by the
[Canfield1987](@citet) sampler, with exact Klein–Nishina and Thomson
rest-frame kinematics.

```@docs
lorentz_boost
klein_nishina
sample_scattered_energy
sample_electron
sample_electron_gamma
sample_mu
compton_scatter
scatter_superphoton
```

## Tracking

The scatter trigger uses the [Kahn1950](@citet) biased exponential transform.
The bias is unbiased by construction and only reduces variance.

```@docs
track_superphoton
track_superphoton_scatter!
bias_func
ScatterWorkspace
push_photon!
pop_photon!
```

## Spectra

```@docs
Spectrum
record!
record_energy!
merge_spectra!
nu_centers
nuLnu
nuLnu_total
nuLnu_solid
total_power
total_weight
mean_energy
```

## Sphere models

An optically thin synchrotron sphere with a closed-form answer, and its
scattering counterpart — the standard validation targets.

```@docs
SphereModel
sphere_zone
sample_sphere_photon
analytic_nuLnu
mc_spectrum
ScatteringSphere
scattering_tau_es
mc_spectrum_scatter
ScatterResult
```

## GRMHD spectra

```@docs
GRMHDModel
plasma_at
emission_zones
track_superphoton_grmhd
track_superphoton_grmhd_scatter!
grmhd_bias
mc_spectrum_grmhd
GRMHDSpectrumResult
incl_nuLnu
```
