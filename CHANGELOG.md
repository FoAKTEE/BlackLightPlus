# Changelog

All notable changes to BlackLight+ (the package `BlackLightPlus`) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on its exported API.
While the version is `0.x`, a bump of `x` may break that API.

## [Unreleased]

- `segment(::IQUV)` is exact in every case of the constant-coefficient transfer equation.
  At the dichroism-cone boundary |α_P| = α_I without rotation (or with ρ ∥ α) it takes the
  limit of the Landi Degl'Innocenti solution, in which the mode polarized against α̂ is
  unabsorbed and gathers its emission linearly, where it returned the total-absorption
  operator and dropped α_P; within a relative gap of 1e-6 of the poles of the closed form,
  and for a thick segment whose slowest mode is still shallow, the eigenmode integrals are
  evaluated through φ(z) = (1 - e^-z)/z. A segment with Faraday rotation and no absorption
  rotates its emission integral, where it added j ds unrotated (first order in the step);
  nilpotent coupling (|α_P| = |ρ_P|, α ⊥ ρ) takes its cubic exponential; a segment is
  saturated only when its slowest mode is deeper than `dtau_max`. The projection of the
  coefficients onto the physical cone stays the default; `segment(...; project = false)`
  solves the equation for coefficients outside the cone too (the one-zone tests of Prather
  et al. 2023).
- The closed-form Kerr tracer gives no image of order n >= 1 to a ray whose four real radial
  roots all lie inside the outer horizon (Gralla-Lupsasca region II): its Mino-time budget
  ends at the horizon, where it counted a turning point at r4 and returned spurious n = 1, 2
  images inside the shadow at high spin and inclination. The radial roots take the
  cancellation-free branch of the resolvent cubic (w- = -P/(3 w+)), which removes a loss of
  precision where P -> 0, and the case-II antiderivative is exactly zero at the turning point
  for dual numbers too.
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
- The fixed-control march on `SuperposedKerrSchild` and `BoostedSuperposedKerrSchild`
  captures a state without a normal observer (`g^{tt} ≥ 0`, as after a coarse step into a
  hole): `normal_frame_energy` is `+Inf` there, in the lane's number type, so the redshift
  cut ends the ray on the CPU and on GPUs alike, where the CPU raised a `DomainError` and
  a GPU captured on a NaN. The adaptive tracer's redshift event reads the same energy. A
  ray that never reaches such a state is bitwise unchanged.
- `EquatorialHotspot` and `PageThorneDisk` are vacuum where the Keplerian circular flow is
  not timelike (inside the photon orbit, which every captured ray crosses):
  `circular_velocity` gives `u^t = +Inf` there, where the CPU raised a `DomainError` and a
  GPU returned vacuum on a NaN, so a hotspot image with captured rays renders on the CPU.
- Figure generators under `figures/`: the paper's figures, the dynamic-ray figures and the
  reproductions of the Mahakala, KPolaris and Blacklight papers' figures, each with its
  standing table.

## [0.3.0]

The next-generation package: three tiers (physics, infrastructure, pipeline), one kernel
for every product, CPU threads or GPUs through the `backend` option, the structure gates of
`test/structure.jl` as the standard, validated bitwise against the reference
implementation (kept on the `blp` branch) and on the A100. Its development history is the
`blp` branch of this repository.
