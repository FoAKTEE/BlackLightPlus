# Changelog

All notable changes to BlackLight+ (the package `BlackLightPlus`) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) on its exported API.
While the version is `0.x`, a bump of `x` may break that API.

## [Unreleased]

- A Monte Carlo run on a pool builds its tables on the devices: the zone tables (the
  weight table and every zone's count and envelope, `Products.grid_tables`) and the
  process's hot-cross table (`Products.hotcross_build`, at the first scattering run), as
  device programs with the arithmetic of the host builders. Their logarithms and
  exponentials are `Numerics.tang_log` and `tang_exp`, Base's table-driven algorithms with
  explicit fused multiply-adds, which a CUDA kernel computes to the host's bits (CUDA's own
  `exp` and `log` differ in the last bit), and the per-node sums are exact limb sums, so the
  tables are bitwise those of the host builders on every executor and every spectrum is
  unchanged bit for bit on the CPU and on the GPU. On the AthenaK torus at stride 1
  (10.3 M zones) on one A100 the zone tables take 0.60 s where the job's four host threads
  took 154 s, the hot-cross table 0.05 s for 3.3 s, and a thin `mc_spectrum` call of 10⁶
  primaries 16 s for 146 s.
- `mc_spectrum` keeps the emission zones and the zone tables of the last grid run and
  reuses them in a run on the same blocks, spacetime, parameters, zone options (and seed,
  above stride 1) and generation settings: a repeated stride-1 call takes 10.5 s for 31 s
  on 32 CPU threads and 4.1 s for 150 s on one A100. The per-zone count reads the weight
  nodes evaluated once (`MonteCarlo.ZoneNodes`): the host zone tables take 16 s for 21 s
  on 32 threads.
- On a device, a cascade lane starts its later chunks with the child slots it grew to,
  and the Monte Carlo primary and layer programs launch in blocks of one warp; with both,
  the torus photon phase on one A100 is 1.16 (thin) and 1.20 (Compton) times faster, bit
  for bit; the Compton phase now runs at the rate of 32 CPU threads.
- `render_crossings` queues the plan's pixels in its cost order, as `render` does: the
  marched disk of the code comparison at 500² is 1.08–1.28 times faster on one A100 and
  1.06–1.09 times on 32 CPU threads, bit for bit.

- `StateBias` takes an order factor `f` (`mc_spectrum(…; bias_order = f)`, 1 by default):
  the bias of a photon scattered `n` times is `clamp(b0 A(Θₑ) (w/w_ref) fⁿ, 1, bias_max)`.
  Without it a child's own bias is near 1, so the light scattered two and three times
  and the high-frequency tail it makes are sampled little. On the AthenaK torus of the
  paper (density unit 1e-21 g cm⁻³, stride 1, `b0 = 1e4`, 3e5 primaries) the factor 100
  leaves the effective photon number per tracked photon of the Compton hump within 5 %
  and raises it 9 and 30 times at 1e21 and 1e22 Hz (50° band), and that of the second
  order 25 times; the factor 300 raises the tail 14 and 80 times for 10–15 % less in
  the hump. `f = 1` gives every bias bit for bit as before. Tests: `event_bias` with and
  without the factor, the one-step expectation at the biases of later generations, and
  the escaped power of every order of a sphere of Thomson depth 0.2 equal with and
  without the factor, absorption off and on.
- The closed-form Kerr tracer takes its radial sn, cn and sc from `Spacetime.jacobi_sncndn`
  and the Jacobi amplitude from `Spacetime.jacobi_amplitude`, which uses the m → 1
  expansion (A&S 16.15.4) only where $(1-m)\cosh^2u<\sqrt\varepsilon$ and the AGM elsewhere.
  JacobiElliptic 0.3 takes the expansion wherever $1-m<\sqrt\varepsilon$, at any $u$; near
  the critical curve, where the radial parameter comes within $10^{-8}$ of 1 and $u$
  reaches $K(m)$, the landing radius was off by up to $4\times10^{-3}$ (relative; order one
  at $80^\circ$ inclination) and the intensity by up to $3\times10^{-3}$ of the maximum, at
  70 and 58 points of the radial cuts of the code comparison and at none of its image
  pixels. The order-0 landing radius at those points now agrees with a 40-digit landing to
  $8\times10^{-14}$ (relative), and the intensity at the 14 of them sampled by the precise
  march agrees with AART and the precise march to $7\times10^{-13}$ of the maximum; every
  other value is unchanged bit for bit. `dn` is formed as $\sqrt{(1-m)+m\,\mathrm{cn}^2}$, free of cancellation as
  $\mathrm{sn}\to1$, and the radial inversion no longer allocates under forced bounds checks.
- The Z4c to ADM resample (`resample_z4c_adm`, `z4c_grid_spacetime`) and `sample_var`
  interpolate across meshblock faces through ghost layers, by the rule of the GRMHD
  blocks: in the outer half cell of a block the stencil blends the outermost cell with the
  ghost beyond the face, the field of the mesh about the ghost centre (`Data.ghost_value`
  on the dump's columns, `Data.BinColumns`; copy, restriction, limited prolongation,
  linear extrapolation), where it held the outermost cell. The nodes are continuous across
  faces between blocks of one level and exact for a linear field everywhere; inside a
  block they are unchanged, bit for bit. On the head-on slice of the capture figure
  (192³ nodes over $|x_i|\le12\,M$, cells of $1/4\,M$, same-level faces at $-8$, $0$,
  $8\,M$) the 9.1 % of nodes within half a cell of a face change by up to
  $2.5\times10^{-3}$ of $\alpha$ and $5\times10^{-3}$ of $\gamma_{ii}$, and the slope
  between the two nodes on either side of each face, 1.99 times the field's before,
  follows the field. 75 of $301^2$ image points of the capture map change class, all on a
  class boundary; the bisected capture boundary moves by $3\times10^{-3}\,M$ (median of
  24 radial scans, at most $1.3\times10^{-2}\,M$) and the outer capture edges of the deck
  row by $2.5\times10^{-3}$ and $1.7\times10^{-3}\,M$. The ghost machinery
  (`Data.mesh_value`) reads any block mesh that provides its cells, widths, cell values
  and locate rule.
- A precise mode for the numerical march, `Plans.precise(cfg)`, at the accuracy of the
  closed-form Kerr tracer: the ninth-order step of Verner's 9(8) pair (`Spacetime.Verner9`,
  beside `ClassicalRK4`) under the fixed control, the step shortened near a spinning
  horizon to resolve the frame-dragged winding (`winding_factor`), the state carried
  compensated, the redshift cut at 10^6 without the horizon backstop, and the camera at
  infinity (`CameraConfig(infinity = true)`: each ray carried in from infinity through the
  far zone of a stationary metric, `Spacetime.infinity_ray`, with the metric offset
  `raise_index_offset` formed without cancellation on Kerr–Schild spacetimes).
  `Products.render_crossings` images the equatorial disk by the march, the crossing found
  to the integrator's order (`Spacetime.surface_crossing`). Against the closed form on the
  two configurations of the code comparison the precise march reaches NMSE 10^-26–10^-22
  per order (median residual 10^-17–10^-13 of the image maximum), where the production
  march stops at 10^-10–10^-8. `MarchBudget` carries the integrator, the redshift cut, the
  horizon backstop and the winding switch; `RunConfig` gains `integrator`,
  `capture_redshift`, `horizon_eps` and `winding`. The production defaults are unchanged:
  images, traces and shadows are those of before, bit for bit on one device.
- The Kerr–Schild radius takes the root of max(r², 0): a device `hypot` may round below
  |x² + y² + z² − a²| inside the ring, where a step that falls through the horizon raised a
  `DomainError` on the GPU.
- `Transfer.segment(IQUV)` is exact to rounding at small absorption and at any
  coefficient scale. The closed form of the eigenmode integrals carries
  $1/(\alpha_I^2-\lambda_1^2)$ and keeps a relative accuracy
  $\epsilon/[(\alpha_I-\lambda_1)\Delta s]$; it is used only where the slow-mode depth
  $(\alpha_I-\lambda_1)\Delta s\ge1$, and the $\varphi$ form, regular as $\alpha\to0$,
  elsewhere. The eigen-decomposition is formed from the polarized coefficients over the
  largest of them, with the cancelling root of $\lambda^2$ as
  $(\boldsymbol\alpha\cdot\boldsymbol\rho)^2$ over the other. The segment returned NaN
  for $0<\alpha_I\lesssim10^{-154}$ (the far gas of the power-law and kappa fits), lost
  up to 6 % of $b$ at $\alpha_I\Delta s=10^{-14}$ and all of the emission of $I$ below
  $\alpha_I\Delta s\approx\epsilon$, and with $\rho_P=10^{10}\alpha_I$ let the slow mode
  decay at $\alpha_I$ instead of $\alpha_I-\lambda_1$; every case now matches the 320-bit
  solution to $10^{-15}$. The thermal production images change by at most $3\times10^{-8}$
  of their maximum.
- AthenaK snapshots are sampled on every meshblock extended by one layer of ghost cells,
  built when the snapshot is loaded from the neighbouring blocks: copied at the same
  level, the mean of the finer cells from a finer block, the minmod-limited linear
  prolongation of AthenaK from a coarser one, edges and corners included, one-sided linear
  extrapolation outside the mesh (`Data.fill_ghosts!`, `Data.mesh_value`). The sample is
  continuous across faces between blocks of one level, a linear field is reproduced
  exactly everywhere and the field is second-order accurate everywhere; the outermost
  cell value was held in the outer half cell of every block. On the AthenaK torus of the
  GRMHD comparison ($i = 163^\circ$, 230 GHz) the seam at the block face $x = 0$ inside
  the shadow (a step of 1.29 times the neighbouring steps) is gone, the net linear
  polarization is 3.349 % (2.647 % before) and the image agrees with Blacklight's
  interblock interpolation, its high-face anchor coordinate corrected, to NMSE
  $9.7\times10^{-6}$ in $I$. `FluidBlocks.data` is `(8, ni + 2, nj + 2, nk + 2, nmb)`
  (20 % more memory for blocks of $32^3$, 10 % for $64^3$; 1 s for 512 blocks of $64^3$
  on 24 threads); `Data.corner(b, i, j, k, m)` reads the interior as `1:n` and the ghost
  layer as `0` and `n + 1`.
- `render_analytic(plan, disk; observables = false)` runs `Products.RadiusProgram`
  (`Spacetime.radius_prep`, `band_radius`): the landing radius without the azimuth and time
  integrals, the intensity, radius and radial sign bit for bit those of the full path, about
  four times faster on the CPU; `photon_ring_layers` uses it. `Spacetime.surface_terms`
  evaluates the source terms of a surface once per screen point, which `SurfaceProgram`
  shares across orders.
- `photon_ring_layers(...; dx = (dx0, dx1, dx2, …))` takes one spacing per band (AART's
  per-order resolution): a band finer than the first is traced on its own grid, enumerated
  within the band by `Analysis.band_points` (the in-band nodes of `band_grid`, row by row from
  the polygon crossings, so the cost follows the band's area), and deposited on the common
  grid by cell-area weights, keeping its ring flux; the result gains `dxs` and the fine
  samples `points`. On the a = 0.94, 17° disk, dx = (0.02, 0.01, 0.005) gives the n = 1 and
  n = 2 ring fluxes of a uniform 0.01 and 0.005 grid with 2.0e6 rays instead of 6.1e6 and
  2.5e7.
- Closed-form movies and light curves of time-dependent equatorial sources:
  `Matter.EquatorialSpot` (a profile f(t, x, y), by default the orbiting `GaussianSpot` of
  AART's flare model, times g^p of the g-disk flow) evaluated at the emission time of every
  landing, and `Products.analytic_movie(plan, spot; times, bands, ro, slow)`, which lands
  every pixel and order once and paints each frame in one pass (fast light with
  `slow = false`). Against AART's `flare_model` at D = 1e4 the frames agree to NMSE 1e-17 (n = 0)
  and 1e-14 (n = 1, 2) away from the pixels where AART takes the radial sign of the turning
  point seen from infinity, and against Krang's slow-light emission coordinates at infinity
  to NMSE 3e-18 (n = 0); light curves over one orbit to 1e-9.
- Photon rings of high order: the case-II time antiderivative takes √R/(r − r3) from the
  factored radial potential, exactly zero at the turning point, so successive orders up to 7
  step by the photon-shell time lapse τ to 1e-5 where the rounding residue of the expanded
  potential had left errors up to 5e-4 at n = 7 (radii, azimuths and intensities are
  unchanged; times change at the rounding level elsewhere). `Spacetime.photon_shell_exponents(a, r̃)`
  (γ, δ, τ of P2 Eqs. 68–72) and `Spacetime.critical_point(a, θo, r̃)`; `lensing_band` finds
  the edges of bands of any order (logarithmic search toward the critical curve, where it
  fell back to fixed scales from n = 5 on).
- The closed-form Kerr tracer images thin surfaces at any polar angle, polarized:
  `Spacetime.surface_prep`, `surface_landing` and `image_crossings` invert ordinary and
  vortical rays (radial cases II, III and IV) to the crossings of the cone θ = θs, ordered
  in Krang's image orders with every crossing in one order and with the sign of p^θ at
  emission (`Landing` gains `θ` and `νθ`); at the equator they reproduce `band_landing` bit
  for bit. `ro = Inf` places the observer at infinity with the regularized emission time
  lim (t + ro + 2 ln ro). `Matter.SynchrotronSurface` (the power-law synchrotron surface of
  Narayan et al. 2021) and `Matter.PolarizedDisk` (AART's polarized g-disk) carry the
  rest-frame electric vector k × b to the screen through its Walker–Penrose constant
  (`Spacetime.penrose_walker`, `screen_polarization`, the ZAMO and rest-frame maps), and
  `render_analytic(plan, materials)` returns Stokes I, Q, U per order for one material or a
  tuple. Against Krang and AART the I, Q, U and κ agree to NMSE 10^-28–10^-24; the EVPA
  agrees with the package's polarized march to its C^-4 error.
- `angular_turning_points` clamps the discriminant at zero, so a vortical ray whose
  observer sits at its polar turning point no longer raises a `DomainError`.
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
