# Geodesics API

The `Geodesics` module owns everything between the camera and the emission
point: the spacetime types, the metric interface, null-geodesic integration,
and the exact-Kerr analytic tracer. Every spacetime supplies only a covariant
metric closure $x \mapsto g_{\mu\nu}(x)$; inverse metrics and Christoffel
symbols come from automatic differentiation, so a new metric never needs
hand-coded derivatives.

```@docs
BlackLightPlus.Geodesics
```

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

### Derived geometric quantities

Exported by `Geodesics` but not re-exported at the top level; reach them as
`BlackLightPlus.Geodesics.<name>`. Christoffel symbols are never materialized
on the hot path — the integrator uses `inverse_metric_gradient` directly — so
these exist for diagnostics and tests.

```@docs
BlackLightPlus.Geodesics.christoffel
BlackLightPlus.Geodesics.metric_gradient
BlackLightPlus.Geodesics.kerr_horizon
BlackLightPlus.Geodesics.kerr_delta
BlackLightPlus.Geodesics.kerr_radial_potential
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

### Shared kernels of the CPU and device marches

The functions every fixed-control march calls, on the CPU and inside the fused
kernels: the metric and its inverse from one evaluation, index raising and
lowering, the geodesic right-hand side extended by parallel-transported frame
vectors, and the step control and termination of the binary and ADM
spacetimes ([March termination](@ref)).

```@docs
BlackLightPlus.Geodesics.metric_and_inverse
BlackLightPlus.Geodesics.lower_index
BlackLightPlus.Geodesics.raise_and_norm
BlackLightPlus.Geodesics.geodesic_transport_rhs
BlackLightPlus.Geodesics.spacetime_control
BlackLightPlus.Geodesics.adm_control
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
