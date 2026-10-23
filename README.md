# BlackLight+

BlackLight+ is a Julia package for general-relativistic ray tracing, polarized radiative
transfer and Monte Carlo Comptonization. The package module is `BlackLightPlus`. A run is a
plan: a configuration, a spacetime and a matter source prepared once. Every product of a
plan (an image, an adaptively refined image, trajectories, a shadow map, parameter
sensitivities, a spectrum) runs on CPU threads or on GPUs through one `backend` option.

## What it computes

- Spacetimes: Kerr in Cartesian Kerr–Schild coordinates (`KerrSchildCartesian`), flat
  space (`Minkowski`), any user metric given as a closure `x -> g_μν` (`UserMetric`), the
  superposed Kerr–Schild binary at rest (`SuperposedKerrSchild`) and in orbit
  (`BlackLightPlus.Spacetime.BoostedSuperposedKerrSchild` with `CircularOrbit`), and
  numerical-relativity slices on a Cartesian grid (`ADMGrid`, built from AthenaK Z4c dumps
  by `z4c_grid_spacetime` or from any metric by `adm_grid_from_metric`). Metric
  derivatives come from forward-mode automatic differentiation, so a new metric needs no
  hand-written connection.
- Matter: AthenaK GRMHD snapshots (`AthenaKSnapshot`, `.athdf` and `.bin`), uniform grids,
  snapshot series for slow light (`SnapshotSeries`, `FrameWindow`), and analytic sources
  (`PolishDoughnut`, `PageThorneDisk`, `EquatorialDisk`, `EquatorialHotspot`,
  `UniformSphere`).
- Transfer: Stokes `I` or full `IQUV` with thermal, power-law and kappa synchrotron
  emission and absorption, Faraday rotation and conversion, integrated in the fluid frame
  and transported to the camera frame.
- Monte Carlo: superphoton spectra with Compton scattering, with exact (order-independent)
  accumulators, so a given plan, superphoton count and seed give the same bins on every
  backend.
- Analysis: visibility amplitudes, lensing bands and the critical curve of exact Kerr.

## Install

Julia 1.10 or later.

```julia
using Pkg
Pkg.add(url = "https://github.com/FoAKTEE/BlackLightPlus")
```

From a clone: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. GPUs need
`using CUDA` in the session, which loads the `BlackLightPlusCUDAExt` extension.

## Images

```julia
using BlackLightPlus
cfg  = RunConfig(mass_scale_cgs = 1e24, frequency_hz = 230e9,
                 camera = CameraConfig(inclination_deg = 60, fov = 40, resolution = 128))
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9),
               matter = AthenaKSnapshot("dump.athdf"), mode = IQUV())
img  = render(plan)              # img.I, img.Q, img.U, img.V; img.run records where it ran
```

`render_amr`, `render_analytic`, `sensitivity` and `variants` take the same plan.

## Trajectories and shadows

```julia
paths = trace(plan; backend = :cpu)                  # source → camera states per pixel
img2  = render(plan; path = paths, backend = :cpu)   # the two-phase render
bare  = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = nothing)
fates = shadow(bare).fate                            # 1 captured, 2 escaped, 0 step-limited
```

## Monte Carlo spectra

```julia
ball = prepare(UniformSphere(n_e = 1e13, theta_e = 100.0, B = 1.0, R = 1.0))
thin = mc_spectrum(ball, 10^6)                       # Spectrum
comp = mc_spectrum(ball, 10^6; scattering = true)    # ScatterResult: one Spectrum per order
spec = mc_spectrum(plan, 10^6; scattering = true)    # GRMHDSpectrumResult: inclination bands
```

## CPU or GPU

```julia
render(plan; backend = :cpu)       # Julia threads
render(plan; backend = :gpu)       # the idle GPUs
render(plan; backend = [0, 1])     # GPUs 0 and 1
backend!(:gpu)                     # session default; backend!(nothing) clears it
```

The keyword wins over `backend!`, which wins over the environment variable `BLP_BACKEND`
(`auto | cpu | gpu | gpu:0,1 | host_pool | host_pool:N`), which wins over `:auto`.

## Standing

The measured record is in the documentation (`docs/src/validation.md`,
`docs/src/design/verification.md`); the statements below are its headlines.

- Against the reference implementation it replaced: analytic images, traces and Monte
  Carlo sums agree bitwise on one hardware; `IQUV` images agree to round-off. On an A100
  the reference device drivers and this package agree bitwise, `IQUV` included. A GPU
  against the CPU differs by 1e-14 to 3e-10 of the image maximum.
- Against RAPTOR: the Gold et al. (2020) test problems T1–T5 agree to NMSE 2.5e-13 in
  intensity, and the images of a binary GRMHD snapshot agree in flux to 5.2e-4.
- Against Blacklight (C++): intensity images of one torus snapshot agree to NMSE 7.9e-4 to
  5.7e-3. The released Blacklight disagrees in Stokes V by a factor of 2 to 7 because its
  polarized coupling misplaces two matrix entries; with those corrected, V agrees to
  NMSE 0.04 (validation page).
- Speed, on the machine named in the verification record: one A100 traces 1.24e6 Stokes-I
  rays per second against 1.69e5 on 128 CPU threads; two A100s give results bitwise equal
  to one.

## Tests, figures and documentation

```
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

runs the unit tests and the structure gates (layering, one kernel, the export budget, file
budgets, no attribution prose). The generators of the paper's figures and of the figure
reproductions of three reference papers are under `figures/`, one directory per set with
a table of what each figure is and how far it was reproduced; they read simulation data
that is not part of this repository.

The documentation site is built with `julia --project=docs docs/make.jl`: getting
started, the guides (GRMHD data, spacetimes and binaries, GPU rendering, Monte Carlo
spectra), the method, validation, benchmarks, the API by module, and the design pages
(`docs/src/design/`).

## Branches

`main` holds the package, its tests, its documentation and the figure generators. The
`blp` branch holds the development history, the reference implementation and its oracle
tests, the review records and the figure images.

## Citing and license

Cite the package with `CITATION.cff`; a method paper is in preparation. MIT license, see
`LICENSE`; third-party notices in `NOTICE`.
