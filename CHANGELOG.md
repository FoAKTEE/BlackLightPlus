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
  `BlackLightPlus.Spacetime`).
- Figure generators under `figures/`: the paper's figures, the dynamic-ray figures and the
  reproductions of the Mahakala, KPolaris and Blacklight papers' figures, each with its
  standing table.

## [0.3.0]

The next-generation package: three tiers (physics, infrastructure, pipeline), one kernel
for every product, CPU threads or GPUs through the `backend` option, the structure gates of
`test/structure.jl` as the standard, validated bitwise against the reference
implementation (kept on the `blp` branch) and on the A100. Its development history is the
`blp` branch of this repository.
