# Changelog

All notable changes to BlackLight+ (the package `BlackLightPlus`) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on its exported API.
While the version is `0.x`, a bump of `x` may break that API.

## [Unreleased]

- `mc_spectrum` on a plasma grid emits from every cell centre its block owns by default
  (`stride = 1`), above the horizon proxy; `stride = s` keeps one owned cell per group of
  `s³`, drawn uniformly on a stream of the seed, with the group's owned volume, an
  unbiased quadrature of the same emission. The fixed first cell of each group with the
  whole group's volume moved emission between radii where the emissivity changes within
  a group (on the AthenaK torus at stride 4, 70 % of the emitted power from
  1.6–2.0 M and the escaping millimetre flux 20–27 % low).
- The angle-integrated synchrotron emissivity is `4π n_e e² ν_c/c (√2π/27) F(K)`,
  `K = ν/((2/9) ν_c Θₑ²)`, from the one-variable table `MonteCarlo.EmissivityTable`
  (built when the package compiles; within 3e-9 of the former 512-interval quadrature,
  now `int_jnu_simpson`). The zone tables (`Data.emission_sites`, `Data.zone_tables`)
  are two streaming, threaded passes that never store the zone × frequency matrix: 21 s
  on 32 threads for the 10,303,459 cells of the AthenaK torus snapshot, where the per-call
  table took an hour and 33 GB.
- The per-zone counts are the trapezoid rule over the weight nodes, so the expected
  number of primaries is `Ns`; the counts and the emitted power were high by
  `1/n_esamp` (+0.25 % at the default 400 intervals).
- The scattering events of a step are a Poisson process of mean `b dτ_s` at points drawn
  from the step's depletion, each child of weight `w (1 − e^{−Δτ})/(b Δτ)`: unbiased for
  every bias with absorption on (the one child per step at the biased point was 14 % high
  at `b dτ_s = 1` and 2.2 times high at 10 in steps thick to absorption), and no longer
  capped at one child per step. A child kicks and marches on its own stream, named by the
  ordinal of its event (`widest_layout`, `child_stream`); an event at a valid site spawns
  even when the step leaves the plasma.
- The default importance bias is `MonteCarlo.StateBias`,
  `b = clamp(b0 A(Θₑ) w/w_ref, 1, bias_max)` with `A = 1 + 4Θₑ + 16Θₑ²` and `w_ref` the
  mean primary weight, so a child carries the energy weight of a mean primary over `b0`
  whatever the gas that scattered it; `bias = b0` sets its scale (1 by default),
  `bias = Bias(b0)` is the constant bias `b0 (n + 1)²`, and `bias_max` and `split` are
  new keywords. On the AthenaK torus at stride 1 and the same wall time, the summed
  effective number of photons of the bins above 1e15 Hz is 34 times that of the constant
  bias `b0 = 1e4` (the bins to 1e18 Hz hold 10³ and more each); a child's own bias is near
  1, so the light scattered two and three times is sampled less. A number given as `bias`
  now selects the state bias.
- Every `Spectrum` carries `Σ (w E)²` per bin; `Execution.nuLnu_sigma` gives the standard
  error and the effective number of photons of each bin.
- `GRMHDSpectrumResult` holds the per-order spectra (`orders`, `max_scatt`) and the
  ledger of `ScatterResult`; both carry the weight scattered in the steps that may spawn
  (`scatter_loss`, `Δτ_s (1 − e^{−Δτ})/Δτ` of every step, so `w_children + w_unspawned`
  estimates it with absorption on), in the steps of the last order
  (`scatter_loss_final`), the weight of events that spawned nothing (`w_unspawned`), and
  the emitted, escaped, absorbed, captured and dropped power. The `MonteCarlo.Sink`
  protocol gains `child_stream`; `spawn!(sink, child, rng)` takes the child's stream and
  `tally!(sink, sph, ::MarchLedger)` the march's ledger.
- The tabulated generation (`tabulated`, `IntJnuTable`, `ZoneEmissionTable`,
  `build_zone_emission_table`, `zone_counts`, `build_int_jnu_table`,
  `zone_int_jnu_tables`) is gone: the exact acceptance costs one table lookup.
- A photon that may spawn no more children stops, captured, once a step moves it inward
  inside the innermost spherical photon orbit on Kerr (`Spacetime.plunge_radius`), and
  the hot-cross table is built once per process; the bins are bitwise unchanged, and the
  thin torus tracks 2.2 times as many photons per second.

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
