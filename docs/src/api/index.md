# API Overview

```@docs
BlackLightPlus
```

## What is public

The public API is what `BlackLightPlus` exports: the configuration records and `prepare`,
the spacetimes, the matter providers, the transfer modes, the executors and the backend
switch, the products, their result types and the analysis utilities. Every exported
name has a docstring and the documentation build checks that each is on one of the pages
below (`checkdocs = :exports`). The structure gate `test/structure.jl` checks the same
for every submodule's exports and holds the top-level API at its budget.

A name with a leading underscore is private to its module and is never referenced from
another module or from the tests. The remaining non-exported names are the submodule
interfaces: stable within a minor version, documented on the same pages under
"Internal", and reached as `BlackLightPlus.Spacetime.trace_geodesic` or with
`using BlackLightPlus.Spacetime: trace_geodesic`.

## Layout

| Page | Module | Owns |
|---|---|---|
| [Numerics](@ref "Numerics module") | `Numerics` | constants and units, the ray state, stencils, exact sums, the Philox generator, special functions |
| [Spacetime](@ref "Spacetime module") | `Spacetime` | the `Spacetime` protocol, Kerr, user metrics, the static and moving superpositions, ADM grids, geodesics, termination, the camera, exact Kerr |
| [Matter](@ref "Matter module") | `Matter` | the `PlasmaProvider` protocol, the plasma decode and electron model, the analytic sources, emission zones |
| [Transfer](@ref "Transfer module") | `Transfer` | synchrotron coefficients, tetrads, segment operators, the marches |
| [MonteCarlo](@ref "MonteCarlo module") | `MonteCarlo` | superphotons, emission tables, scattering, opacities, sinks |
| [Data](@ref "Data module") | `Data` | AthenaK readers, fluid blocks, uniform grids, snapshot series, the npy writer, tables |
| [Execution](@ref "Execution module") | `Execution` | devices, pools, the kernel, executors, the backend switch, ray sources, schedules, accumulators |
| [Plans](@ref "Plans module") | `Plans` | configuration, lanes, the plan and `prepare` |
| [Analysis](@ref "Analysis module") | `Analysis` | refinement indicators, lensing bands, interferometry |
| [Products](@ref "Products module") | `Products` | `render`, `render_amr`, `render_analytic`, `photon_ring_layers`, `trace`, `shadow`, `sensitivity`, `variants`, `mc_spectrum` and their result types |

## Conventions

These hold throughout the API and are not repeated in every docstring:

- Metric signature $(-,+,+,+)$; geometrized units $G = c = M = 1$.
- Cartesian Kerr–Schild coordinates $x^\mu = (t, x, y, z)$.
- The ray state is the 8-vector $y = [x^\mu;\, p_\mu]$ with covariant momentum, $p = -k$
  for the received wave covector $k$ ($p_t = +1$), integrated to increasing affine
  parameter from the camera into the past.
- Transfer coefficients are Lorentz invariant: $j_\nu/\nu^2$, $\alpha_\nu \nu$,
  $\rho_\nu \nu$; the invariant intensity is $I_\nu/\nu^3$.
- Images are laid out `[iβ, iα]`; Stokes parameters follow IEEE/IAU. The full table is
  in [Method](@ref method-conventions).
