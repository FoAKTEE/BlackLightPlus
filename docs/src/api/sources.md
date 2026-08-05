# Sources API

The `Sources` module provides semi-analytic plasma models — alternatives to a
GRMHD snapshot when you want a controlled, reproducible emission region.

All sources implement the `source_plasma_at` seam, which is the single point
both the imaging pipeline and the Monte Carlo module call to obtain plasma
state at a position.

```@docs
BlackLightPlus.Sources
```

## Interface

```@docs
AbstractSource
source_plasma_at
```

## Models

```@docs
PolishDoughnut
PageThorneDisk
EquatorialHotspot
```

## Model internals

```@docs
torus_potential
page_thorne_flux
page_thorne_temperature
hotspot_center
```
