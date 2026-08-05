# Render drivers API

These are the top-level entry points. They compose the submodules — geodesics,
simulation input, radiation — at root scope, which keeps `Radiation` free of a
hard dependency on `Simulation`.

## Two-phase render

`render` traces and caches geodesics, then integrates radiative transfer along
them. `render_amr` wraps that in adaptive image-plane refinement
[White2022](@citep), tiling the image and subdividing blocks flagged by
relative-gradient and relative-Laplacian indicators.

```@docs
render
render_amr
```

## Analytic render

`render_analytic` uses the exact-Kerr elliptic tracer and needs no simulation
data. `photon_ring_layers` renders the $n = 0, 1, 2$ photon-ring images
separately on their lensing bands.

```@docs
render_analytic
photon_ring_layers
```
