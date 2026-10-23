# BlackLight+

General-relativistic ray tracing, polarized radiative transfer and Monte Carlo
Comptonization in Julia, on CPU threads or GPUs through one `backend` option. The package
is `BlackLightPlus`: a run is a plan, prepared once from a configuration, a spacetime and a
matter source; every product of a plan (images, adaptive images, trajectories, shadows,
sensitivities, spectra) runs on either backend with identical results.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/FoAKTEE/BlackLightPlus")
```

or, from a clone, `julia --project=. -e 'using Pkg; Pkg.instantiate()'`. GPUs need
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

The spacetime may be `KerrSchildCartesian`, `SuperposedKerrSchild`, the moving binary
`BlackLightPlus.Spacetime.BoostedSuperposedKerrSchild`, `ADMGrid` (a numerical-relativity
slice) or `UserMetric` (any closure `x -> g_μν`, differentiated by forward-mode AD); the
matter an `AthenaKSnapshot`, a `UniformGrid`, a `FrameWindow` of a `SnapshotSeries` (slow
light) or an analytic source such as `PolishDoughnut(0.9)`. `render_amr`,
`render_analytic`, `sensitivity` and `variants` take the same plan.

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

Spectra are exact sums: the same plan, `Ns` and seed give the same bins on every backend.

## CPU or GPU

```julia
render(plan; backend = :cpu)       # Julia threads
render(plan; backend = :gpu)       # the idle GPUs
render(plan; backend = [0, 1])     # GPUs 0 and 1
backend!(:gpu)                     # session default; backend!(nothing) clears it
```

The keyword wins over `backend!`, which wins over the environment variable `BLP_BACKEND`
(`auto | cpu | gpu | gpu:0,1 | host_pool | host_pool:N`), which wins over `:auto`.

## Tests and figures

```
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

runs the suite, the structure gates included. The generators of the paper's figures and of
the figure reproductions of three reference papers are under `figures/`, each directory
with its standing table; they read simulation data that is not part of this repository.
The comparison against the reference implementation (the bitwise oracle), the review
records and the figure images live on the `blp` branch.

## Documentation

The site (`docs/`): getting started, the guides (GRMHD data, spacetimes, GPUs, Monte
Carlo), the method, the API by module, validation and benchmarks; the design pages
(`docs/src/design/`: architecture, principles, performance, the verification record and
the repository standard) state the design and its measured standing.
