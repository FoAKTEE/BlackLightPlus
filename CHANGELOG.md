# Changelog

All notable changes to BlackLight+ (the package `BlackLightPlus`) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on its exported API.
While the version is `0.x`, a bump of `x` may break that API.

## [Unreleased]

- The package moved to the repository root under the name `BlackLightPlus` (formerly the
  module `BLP` under `BLP/`); the CUDA extension is `BlackLightPlusCUDAExt`; environment
  variables keep the `BLP_` prefix.
- `BoostedSuperposedKerrSchild` and `CircularOrbit`: the superposed Kerr-Schild binary in
  motion, each hole's term in the rest frame of its instantaneous velocity, for
  time-dependent ray tracing of orbiting binaries (reached through
  `BlackLightPlus.Spacetime`). Every product runs on it, on the CPU and on GPUs:
  polarized renders transport the camera frame with the time-dependent connection
  (`geodesic_transport_rhs` for any `Spacetime`), and `CircularOrbit(…; t0)` phases the
  orbit to the observation epoch of a `SnapshotSeries`.
- `Spacetime.proxy_distance(st, x)` and `horizon_floor(st)`: the horizon proxy of a march
  on any spacetime (the per-hole radius at the ray's time on a binary, the lapse on an
  `ADMGrid`, the coordinate radius without a horizon). The Monte Carlo `CurvedMarch` and
  the emission-zone culling read it, so `mc_spectrum` runs on binaries.
- `Matter.decode` returns vacuum where the metric has no normal observer (`g^{tt} ≥ 0`,
  deep inside a hole of a superposed metric) instead of failing on the four-velocity.
- Figure generators under `figures/`: the paper's figures, the dynamic-ray figures and the
  reproductions of the Mahakala, KPolaris and Blacklight papers' figures, each with its
  standing table.

## [0.3.0]

The next-generation package: three tiers (physics, infrastructure, pipeline), one kernel
for every product, CPU threads or GPUs through the `backend` option, the structure gates of
`test/structure.jl` as the standard, validated bitwise against the reference
implementation (kept on the `blp` branch) and on the A100. Its development history is the
`blp` branch of this repository.
