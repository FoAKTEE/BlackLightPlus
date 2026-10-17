# BLP

General-relativistic ray tracing, polarized radiative transfer and Monte Carlo
Comptonization in Julia. A run is a plan, prepared once from a configuration, a spacetime
and a matter source; every product of a plan runs on CPU threads or on GPUs through one
`backend` option.

## Install

From the repository root, `julia --project=BLP -e 'using Pkg; Pkg.instantiate()'`; from
another environment, `] dev ./BLP`. GPUs need `using CUDA` in the session, which loads the
`BLPCUDAExt` extension.

## Images

```julia
using BLP
cfg  = RunConfig(mass_scale_cgs = 1e24, frequency_hz = 230e9,
                 camera = CameraConfig(inclination_deg = 60, fov = 40, resolution = 128))
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9),
               matter = AthenaKSnapshot("dump.athdf"), mode = IQUV())
img  = render(plan)              # img.I, img.Q, img.U, img.V; img.run records where it ran
```

The spacetime may be `SuperposedKerrSchild`, `ADMGrid` or `UserMetric`; the matter a
`UniformGrid`, a `FrameWindow` of a `SnapshotSeries` (slow light) or an analytic source
such as `PolishDoughnut(0.9)`. `render_amr`, `render_analytic`, `sensitivity` and
`variants` take the same plan.

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
backend()                          # the resolved default
```

The keyword wins over `backend!`, which wins over the environment variable `BLP_BACKEND`
(`auto | cpu | gpu | gpu:0,1 | host_pool | host_pool:N`, read once), which wins over
`:auto`. `:auto` takes the GPUs when CUDA is loaded and a device is idle, else threads, and
logs its choice once per session. An explicit choice that cannot be honoured, such as
`:gpu` without `using CUDA`, throws `BackendUnavailable`.

## Tests

From the repository root:

```
julia -t auto --project=BLP -e 'using Pkg; Pkg.test()'
julia -t 8 --project=BLP/test/oracle BLP/test/oracle/runtests.jl
```

The first runs the suite, the structure gates included; the second compares BLP with the
reference implementation in `src/` (instantiate `BLP/test/oracle` once). Point
`BLP_REF_GPUMONTY` at the gpumonty `hotcross.dat` for the hot-cross comparisons, and set
`BLP_TEST_DEVICES=cuda` to run the device comparisons of the oracle on the GPUs. On a
machine whose login shell has no device access, run both commands inside the job
scheduler's allocation, and pin the CUDA runtime in `LocalPreferences.toml`
(`[CUDA_Runtime_jll] version = "13.3"`) so the runtime JLL does not have to see the
driver when it compiles.

## Documentation

- [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md): tiers, modules, protocols, programs,
  products, the public API and the continuity standard.
- [`docs/DESIGN_PRINCIPLES.md`](../docs/DESIGN_PRINCIPLES.md): the principles and their
  gates.
- [`docs/VERIFICATION.md`](../docs/VERIFICATION.md): the measured results of the tests,
  the oracle, the paper's figures and the benchmarks on the CPU and the A100.
