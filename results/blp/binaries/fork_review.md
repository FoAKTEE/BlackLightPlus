# smressle/blacklight_binaries @ 038d46318a81 — review

Scope: the fork of Blacklight (ref-code/c-white-blacklight @ a7c7215) that replaces the Kerr
metric by the superposed, boosted Kerr-Schild (SKS) binary metric of arXiv:2403.13308
(`sources/arxiv-2403.13308.md`), read against the original, the two papers and BLP's
`SuperposedKerrSchild`. Code anchors are to the fork unless marked. Evidence: the vacuum
and ray checks in `checks/` (Julia, BLP environment) and the mock runs in `runs/README.md`.

## 1. What the fork changes

- Input `orbit_file` (input_reader.cpp:134): a table of hole positions, spin vectors,
  dimensionless spin of hole 2 and 3-velocities on a uniform time grid, read and linearly
  interpolated to the snapshot time once in each of `GeodesicIntegrator` and
  `RadiationIntegrator` (geodesic_integrator.cpp:330–381, radiation_integrator.cpp:42).
  `q_ratio` from the header is the mass of hole 2; hole 1 has mass 1 (bh_m); the file's
  spin of hole 2 is multiplied by `q_ratio` (geodesic_integrator.cpp:367–369).
- Metric (geodesic_geometry.cpp:29–592, duplicated in radiation_geometry.cpp:494–1035):
  per hole, `get_prime_coords` applies the spatial Lorentz contraction to x − s_A
  (eq. fermicoord2 of the paper with the time term dropped), `boosted_BH_metric_addition`
  forms f = 2 M r'³/(r'⁴ + (a·x')²), the arbitrary-spin null covector l'_μ and
  l_μ = Λ(v)^ν_μ l'_ν, and `CovariantSimulationMetricFunctionForDerivatives` sums
  η + Σ_A f_A l_μ l_ν. Inside r' < 0.8 r_h the term is evaluated at r' = 0.8 r_h along
  the same (θ, φ) (geodesic_geometry.cpp:379).
- Inverse metric by a hand-written symmetric 4×4 cofactor inverse (`gluInvertMatrix`,
  geodesic_geometry.cpp:236); metric derivatives by central differences of the covariant
  metric, DEL = 1e-7 in the geodesic integrator and 1e-4 in the radiation integrator,
  then ∂g^{μν} = −g^{μα} g^{νβ} ∂g_{αβ} (geodesic_geometry.cpp:1112–1249).
- Geodesic equation extended to dp_t/dλ (geodesics.cpp:1297–1300, 1332–1337), with the
  time column of the derivative set to zero in the geodesic integrator
  (geodesic_geometry.cpp:1177) but finite-differenced from the orbit at t ± DEL in the
  radiation integrator's connection (radiation_geometry.cpp:364).
- Termination per hole at r'_A < r_terminate,A with r_h,A = M_A + √(M_A² − a_A²) and the
  photon-orbit, multiplicative or additive rule (geodesic_integrator.cpp:126–151);
  step size from min_A(r'_A − r_h,A) (geodesics.cpp:633, 1072) or min_A r'_A (dp, :176).
- Momentum renormalization: the original rescales the spatial covector keeping k_t
  (original geodesics.cpp:521, 710); the fork keeps the spatial covector and re-solves k_t
  from the null condition, taking the first root if it is negative, else the second
  (geodesics.cpp:389, 529, 790, 969).

## 2. Verified correct

- The boosted, arbitrarily-spinning single-hole term is the exact Kerr metric: the
  transcription of `get_prime_coords`, `GetBoyerLindquistCoordinates` and
  `boosted_BH_metric_addition` with s(t) = s₀ + v t has |R_μν| ≤ 1e-16 at four points
  with |a| = 0.84 in an arbitrary direction and |v| = 0.31, while the same construction
  without the boost or with Λ(−v) has |R_μν| ~ 1e-2 (`checks/fork_metric_vacuum.log`).
  The round trip x' → (r, θ, φ) → x' of `convert_spherical_to_cartesian_ks` is exact to
  7e-16; the r' = 0.8 r_h clamp is continuous. [SOLID]
- `gluInvertMatrix` inverts random symmetric 4×4 matrices to 1e-11
  (`checks/glu_inverse_check.log`). [SOLID]
- The root selection is consistent with Blacklight's sign convention: the camera stores
  the future-directed covector with k_t < 0 (original camera.cpp:565) and steps to
  negative affine parameter (original geodesics.cpp:467); the fork's first root is the
  negative one when the spatial norm g^{ij}k_ik_j is positive. [SOLID]
- With one hole at rest at the origin the fork reproduces the original: identical NaN
  masks, flux ratio 1.0000, pixel differences ≤ 3.2e-3 of the peak for Schwarzschild and
  ≤ 2.6e-3 for Kerr a = 0.9, geodesic times within 11 s of 1.7e3 s
  (`runs/README.md`). The Kerr a = 0.9 case contains an ergoregion, so the
  renormalization alone does not break a stationary metric. [SOLID]

## 3. Defects

D1. The metric is frozen at one coordinate time while the holes move. The orbit is
interpolated once at the snapshot time (geodesic_integrator.cpp:118) and used at every
point of every ray. A hole's term frozen at t₀ is not the boosted Kerr metric: the surface
r' = 2M, null in the time-dependent metric (normal norm 8e-16), has a normal of norm
−0.52 to +0.35 in the frozen metric, so it is no longer a horizon. Backward-traced parallel
rays past a single Schwarzschild hole of velocity v (`checks/frozen_vs_moving.log`,
exact edge of the capture interval 5.127 M for the camera used, measured 5.12–5.13 M in the
static and in every time-dependent case):

| v | geometry | time-dependent metric | frozen metric |
|---|---|---|---|
| 0.05 | approaching | half-width 5.10, 0 stuck | half-width 4.15, 110 of 281 rays stuck |
| 0.10 | approaching | half-width 5.10, 0 stuck | half-width 3.35, 88 stuck |
| 0.20 | approaching | half-width 5.10, 0 stuck | half-width 1.95, 40 stuck |
| 0.05 | transverse | centre +0.20, half-width 5.10 | centre +1.60, 153 stuck |
| 0.10 | transverse | centre +0.40, half-width 5.10 | centre +2.63, 152 stuck |
| 0.20 | transverse | centre +0.80, half-width 5.10 | 4 captured, 96 stuck |

"Stuck" rays end by the integrator's step collapsing (dt below ε) at a surface outside the
termination radius, which in the fork becomes `ray_max_steps` and a NaN pixel. The
time-dependent metric shifts a transversely moving hole's shadow by 4 v M and leaves its
size unchanged; the frozen metric shrinks the shadow by 19–62% for v = 0.05–0.20. In the
fork's own run, a single boosted hole at v = 0.2 produces 427 NaN pixels of 4096 on one
side of the hole and 11% less flux than the same hole at rest (`runs/README.md`). The
orbital speed of the imaged binaries is 0.10 (sep 25, q = 1). This is the first-order
effect for near-horizon rays, and it is larger than the boost itself; an unboosted static
hole is closer to the truth than a frozen boosted one. [SOLID]

D2. The new renormalization is not continuous through ergoregions. Where
g^{ij}k_ik_j < 0 the two roots for k_t share the sign of g^{0i}k_i; the rule "first root if
negative, else second" then returns a positive k_t and the ray reverses in time. In the
binary mock run (RK4) 5,435 warnings fired on 2,308 of 4,096 pixels; 27.4% of them lie
in an ergoregion and 99.9% of those have k_t > 0, none outside (parsed from
`runs/logs/fork_binary_v02_rk4.log` before trimming). The trajectory the fork dumps for
pixel 1233 starts at k_t = −0.915, changes the sign of k_t twice, ends at k_t = −3.7e4
with |k_i| = 4.0e4 and a time coordinate that is not monotonic. The original's rescale of
the spatial covector with k_t fixed has no such branch: one root is 1 for a null state and
the other exceeds 1 for a future-directed photon. The dp integrator shares the new rule
(geodesics.cpp:389) and gives 2,397 NaN pixels on the same deck. The damage follows D1:
the frozen metric drives rays into regions the exact metric would not, and the
renormalization then fails to bring them back. [SOLID]

D3. Multi-snapshot runs never update the orbit. Both integrators read the orbit in their
constructors at the time of snapshot 0 (blacklight.cpp:115 reads snapshot 0 before the
geodesics; the loop at :186 reads snapshot n but no orbit), so with
`simulation_multiple = true` or `slow_light_on = true` every image after the first uses the
holes of the first snapshot while the fluid has moved. With slow light the time passed to
the orbit reader is `time[0]`, the earliest slice of the chunk, not the snapshot's. [SOLID]

D4. The radiation integrator normalizes the velocity direction without a guard
(radiation_geometry.cpp:540, 877: `nx = vxbh/beta_mag`), unlike the geodesic integrator
(geodesic_geometry.cpp:84, 421: `/(beta_mag + 1e-15)`). A hole at rest gives 0 × NaN = NaN
in every metric component of the radiation integrator: the v = 0 mock run has 4,096 of
4,096 NaN pixels and 2.96 million NaN-emissivity prints, while its geodesics are fine. The
merged remnant of the paper's merger interpolation has v = v_kick, so a remnant at rest,
or any test with static holes, hits this. [SOLID]

D5. `model_type = formula` segfaults: the constructor reads `orbit_quantities(IA1X)`
(geodesic_integrator.cpp:126) and `CovariantGeodesicMetric` calls the binary metric with
the unallocated orbit array for every model type; `input/example_formula.input` exits with
signal 11 (`runs/logs/fork_formula_mode.log`). [SOLID]

D6. The two copies of the metric code disagree: SMALL = 1e-15 (geodesic_integrator.hpp:165)
against 1e-7 (radiation_integrator.hpp:369, and a file-scope duplicate at
radiation_geometry.cpp:12); DEL = 1e-7 against 1e-4; ∂_t g dropped in the geodesic equation
but kept in the connection used for polarization transport, so the transported frame is
not transported along the integrated geodesic. The radiation-integrator SMALL moves
sample points by up to 1e-7 M along the spin axis near each hole's equatorial plane, which
is 0.1% of its finite-difference step. [SOLID]

D7. Inside the ring region R' ≤ a of either hole `ContravariantGeodesicMetricDerivative`
zeroes `dgcov` but never writes `dgcon` (geodesic_geometry.cpp:1136–1146 against
1221–1230), which the substep then reads. Unreachable while rays stop at r' ≥ r_h, since
R' ≤ a lies inside the horizon. [SOLID, latent]

D8. Debugging output is live. Every RK4 step of pixel 1233 appends a record to
`debug_output_1233.txt` in the working directory (geodesics.cpp:858; 2,475 lines per
64² image), renormalization warnings and "Bad geodesic" reports go to stderr
(geodesics.cpp:801, 888), the metric prints at three hard-coded points
(geodesic_geometry.cpp:458, 531, 572) and whenever |g_yy| > 1000
(radiation_geometry.cpp:982, 1017), and every NaN emissivity prints 7 lines
(simulation_coefficients.cpp:510): 4.6 million lines and 88.7 of 92.4 s wall time in the
dp binary run. [SOLID]

D9. Cost. One derivative evaluation calls the metric 9 times (two of them at t ± DEL whose
difference is then discarded), each substep adds a full analytic derivative that is only
compared in commented-out code (`_alt`, geodesics.cpp:1328) and the RK4 loop evaluates the
derivative once more after each step for the dumped record (geodesics.cpp:849); every
metric evaluation heap-allocates two `Array<double>` (geodesic_geometry.cpp:402, 513) and
calls acos and atan2. Geodesics take 6.3 s against 0.5 s for the original on the 64² mock
(12×, 8 threads), before the stderr traffic. [SOLID]

D10. Orbit file and conventions. The reader mixes a text header with binary doubles
through `fscanf("%i %lf \n")` (geodesic_integrator.cpp:337), whose trailing whitespace
directive consumes any leading whitespace bytes of the first double; `fopen` failure is
reported but not fatal; `fread` results are unchecked; the time grid is assumed uniform
(dt from the first two entries, :373). Masses are M₁ = 1 and M₂ = q in code units, whereas
the AthenaK binary runs imaged by BLP use total mass 1 with m₁ = 1/(1+q): an orbit file
for those dumps has to be written in the fork's units or the unit mass changed. The
single-hole spin `simulation_a` still sets the radius used for the camera-sphere cut and
the `sks` fluid metric, which is now a different spacetime from the geodesic one; only
`simulation_coord = cks` dumps are consistent. [SOLID]

## 4. What a near-horizon comparison with BLP needs

The fork as it stands cannot serve as the reference for near-horizon rays of a moving
hole (D1, D2). In order of necessity:

1. Evaluate the metric at the ray's own coordinate time: orbit interpolated at
   t_snap + x⁰ (the camera has x⁰ = 0, original camera.cpp:61, and x⁰ decreases along
   the ray; the snapshot time is the time at the holes, so the offset is the light travel
   time from the camera sphere), with the full ∂_t g in the geodesic equation and the
   connection. The fork already integrates dp_t and already builds the t ± DEL metrics.
2. Restore the original renormalization (rescale the spatial covector, k_t fixed) or
   select the k_t root continuous with the previous step; never the sign test alone.
3. Re-read the orbit per snapshot (D3); guard |v| = 0 (D4); allocate or bypass the orbit
   in formula mode (D5); one SMALL, one DEL, one treatment of ∂_t g (D6); remove the
   debugging I/O (D8).
4. Then the exact benchmark both codes can run is a single boosted Kerr hole: its metric
   is vacuum (`checks/fork_metric_vacuum.jl`) and its shadow is the Lorentz transform of
   the Kerr shadow (capture edge invariant for a collinear boost, shifted by 4 v M for a
   transverse one at the camera of `checks/frozen_vs_moving.jl`). After that, the two-hole
   decks of BLP (`BLP/figures/fig_eht_bbh.jl`, `fig_bbh_i90_pol_amr.jl`: sep 25, q 1,
   a = 0, v = 0.1) with the orbit written in the fork's units.
