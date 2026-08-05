# Geodesics API

The `Geodesics` module owns everything between the camera and the emission
point: the spacetime types, the metric interface, null-geodesic integration,
and the exact-Kerr analytic tracer. Every spacetime supplies only a covariant
metric closure $x \mapsto g_{\mu\nu}(x)$; inverse metrics and Christoffel
symbols come from automatic differentiation, so a new metric never needs
hand-coded derivatives.

## Spacetimes

```@docs
KerrSchildCartesian
Minkowski
UserMetric
SuperposedKerrSchild
ADMGridSpacetime
adm_grid_from_metric
sample_adm
```

## Metric interface

```@docs
metric
inverse_metric
inverse_metric_gradient
ks_radius
horizon_radius
ks_radius_hole
horizon_radius_hole
```

## Geodesic integration

```@docs
geodesic_rhs
trace_geodesic
trace_rays
trace_rays_sampled
rk4_step_fixed
nullify_covariant
```

## Conserved quantities and diagnostics

```@docs
energy
angular_momentum_z
null_error
```

## Camera

```@docs
PlaneParallelCamera
camera_rays
camera_rays_at
```

## Kerr orbits

Characteristic radii and orbital quantities for the Kerr metric, used to place
sources and to set integration bounds.

```@docs
isco_radius
isco_prograde
photon_orbit_radius
marginally_bound_radius
keplerian_omega
keplerian_specific_angular_momentum
```

## Analytic Kerr tracer

An exact closed-form tracer for equatorial sources in Kerr, inverting the
geodesic with Legendre and Jacobi elliptic integrals in the formulation of
[GrallaLupsasca2020NullGeodesics](@citet). No ODE integration is involved.

```@docs
conserved_quantities
radial_roots
angular_turning_points
analytic_angular_integrals
analytic_radial_integrals
analytic_emission
equatorial_intensity
```

## Lensing bands and photon rings

Machinery for the nested order-$n$ photon-ring images, following AART
[CardenasAvendano2023](@citep).

```@docs
critical_curve
apparent_horizon
lensing_band
band_grid
polygon_area
```
