# API Overview

```@docs
BlackLightPlus.BlackLightPlus
```

## What is public

**The public API is exactly what `BlackLightPlus` exports.** Every exported
symbol is documented on the pages listed below, and that coverage is enforced
at build time — the documentation build runs with `checkdocs = :exports`, so a
new export without a docstring fails CI.

Anything not exported is internal. Internal functions may be renamed, changed,
or removed without a version bump, even in a patch release. If you find
yourself reaching for `BlackLightPlus.Something` that is not exported, please
open an issue rather than depending on it.

## Layout

The package is organized as submodules, each owning one stage of the pipeline.
Symbols are re-exported from the top level, so `using BlackLightPlus` is
sufficient — the submodule structure is documentation, not something you need
to navigate at the call site.

| Page | Module | Owns |
|---|---|---|
| [Configuration](@ref "Configuration API") | `Inputs` | run, camera, and plasma configuration |
| [Geodesics](@ref "Geodesics API") | `Geodesics` | spacetimes, metrics, integration, cameras, the analytic Kerr tracer |
| [Simulation input](@ref "Simulation API") | `Simulation` | AthenaK and Z4c readers, snapshot series |
| [Radiation](@ref "Radiation API") | `Radiation` | tetrads, transfer coefficients, Stokes coupling |
| [Sources](@ref "Sources API") | `Sources` | analytic plasma models |
| [Monte Carlo](@ref "Monte Carlo API") | `MonteCarlo` | superphotons, Compton scattering, spectra |
| [Outputs](@ref "Outputs API") | `Outputs` | image writers, interferometric observables |
| [Render drivers](@ref "Render drivers API") | top level | `render`, `render_amr`, `render_analytic` |
| [GPU drivers](@ref "GPU drivers API") | standalone files | fused GPU kernels, included explicitly |

## Conventions

These hold throughout the API and are not repeated in every docstring:

- Metric signature $(-,+,+,+)$; geometrized units $G = c = M = 1$.
- Cartesian Kerr–Schild (CKS) coordinates $x^\mu = (t,x,y,z)$.
- The geodesic state is the 8-vector $y = [x^\mu;\, p_\mu]$, carrying
  **covariant** momentum.
- Transfer coefficients are Lorentz-invariant: $j_\nu/\nu^2$,
  $\alpha_\nu \nu$, $\rho_\nu \nu$, with invariant intensity
  $\mathcal{I} = I_\nu/\nu^3$.
- Physics arithmetic is `Float64`. Plasma quantities are in cgs.

Where a function departs from any of these, its docstring says so.

## Physical constants

`Constants` is module-private: its values are used throughout the package but
are not part of the public API and are not re-exported.

```@docs
BlackLightPlus.Constants
```
