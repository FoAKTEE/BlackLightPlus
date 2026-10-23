# BlackLight+

General-relativistic ray tracing, polarized radiative transfer and Monte Carlo
Comptonization in Julia, on CPU threads or GPUs through one `backend` option. The package
is `BlackLightPlus`.

```@raw html
<img src="assets/torus_IQUV.png" alt="Polarized image of an analytic magnetized torus around a Kerr black hole"
     style="max-width:100%; border-radius:6px; margin:1.5rem 0;">
```

A run is a plan, prepared once from a configuration, a spacetime and a matter source;
every product of a plan runs on either backend with identical results, bitwise per
pixel or per photon whatever the device count and chunking.

## Capabilities

- **Spacetimes** — Kerr in Cartesian Kerr–Schild coordinates; any analytic metric
  through a closure (`UserMetric`), differentiated by forward-mode AD; superposed
  Kerr–Schild binaries at rest (`SuperposedKerrSchild`) and in motion
  (`BoostedSuperposedKerrSchild` on a `CircularOrbit`), each hole's term in the rest
  frame of its velocity; numerical-relativity slices (`ADMGrid`) from AthenaK Z4c dumps
  with a $C^1$ interpolant.
- **Geodesics** — Hamiltonian null geodesics in covariant momentum, $p_t$ conserved
  exactly on stationary metrics; a fixed-control RK4 march with the step
  $(r - r_h)/C$ for imaging and an adaptive Vern7 solve with event termination; an
  exact-Kerr elliptic path with lensing bands and photon-ring layers.
- **Radiative transfer** — Stokes $I$ or full $IQUV$ in a parallel-transported camera
  frame with exact constant-coefficient segment operators; thermal, power-law and kappa
  synchrotron emission and absorption, Faraday rotation and conversion; the
  $R_{\rm high}/R_{\rm low}$ electron temperature; magnetization and temperature cuts.
- **Matter** — AthenaK GRMHD snapshots (`.athdf`, `.bin`) sampled on their mesh
  hierarchy, uniform grids, slow light over a snapshot series, and analytic sources: a
  constant-angular-momentum torus, a Page–Thorne disk, an orbiting hotspot, a uniform
  sphere.
- **Products** — images, adaptive image-plane refinement, trajectories, shadows, exact
  parameter derivatives of the image, mass-scale ladders from one trace, Monte Carlo
  spectra with Compton scattering.
- **Execution** — one kernel for every device computation; Julia threads or a pool of
  CUDA devices chosen by `backend`; `Float32` geodesic lanes and mixed lanes on Kerr.

## Where to go next

| If you want to | Read |
|---|---|
| Render a first image, no data or GPU needed | [Getting Started](@ref) |
| Image a GRMHD snapshot, fast or slow light | [Imaging GRMHD data](@ref) |
| Use a binary, a moving binary or a numerical-relativity slice | [Spacetimes and binaries](@ref) |
| Run on GPUs | [GPU rendering](@ref) |
| Compute Compton spectra | [Monte Carlo spectra](@ref) |
| Understand the physics and numerics | [Method](@ref) |
| See the validation and the figure reproductions | [Validation](@ref) |
| See measured performance | [Benchmarks](@ref) |
| Read the design and its gates | [Architecture](@ref) |
| Look up a function | [API Overview](@ref) |

## Citing

A method paper is in preparation. Until it appears, cite the repository through its
[`CITATION.cff`](https://github.com/FoAKTEE/BlackLightPlus/blob/main/CITATION.cff).

## License

MIT. Third-party provenance is recorded in
[`NOTICE`](https://github.com/FoAKTEE/BlackLightPlus/blob/main/NOTICE).
