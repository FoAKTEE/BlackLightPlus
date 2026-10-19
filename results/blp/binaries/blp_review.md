# BLP `SuperposedKerrSchild` — standing against arXiv:2103.15707 and arXiv:2403.13308

Scope: `BLP/src/physics/Spacetime/superposed.jl` (BLP 0.3.0, branch blp @ 520e5e9) and the
binary products built on it (`BLP/figures/fig_eht_bbh.jl`, `fig_bbh_i90_pol_amr.jl`), read
against the superposed-metric papers (`sources/`) and the fork reviewed in
`fork_review.md`. Evidence: `checks/blp_superposed_check.log`, `checks/frozen_vs_moving.log`.

## 1. What BLP implements

`SuperposedKerrSchild{N}` is g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν with each hole at rest at
`center[A]`, spin along ±z, horizon proxy M_A + √(M_A² − a_A²); inverse by the Woodbury
identity with an N×N solve; gradients by forward-mode AD; capture at
min_A(r_A − r_h,A) < 1e-3 or normal-frame energy > 100 (termination.jl:139–151). It is the
static superposition of Kerr-Schild holes: equation sks of arXiv:2403.13308 with Λ = 1,
s_A constant and a_A ∥ z. The implementation is correct for what it states:

- two massive holes (masses 1/2, spins 0.15 and −0.25, separation 20): |g g⁻¹ − 1| ≤ 2e-16,
  `raise_and_norm`, `lower_index`, `raise_index` agree with the matrices to 2e-16, the AD
  gradient of g^{μν} agrees with central differences to 2.6e-10 (the shipped test
  `test/physics/spacetime_superposed.jl` checks the gradient only with a massless second
  hole); [SOLID]
- a massless second hole reproduces single Kerr bitwise (shipped test). [SOLID]

The documentation states the approximation (`docs/src/guide/spacetimes.md:34–60`,
`docs/src/method.md:63–72`) and `fig_eht_bbh.jl` records "static SuperposedKerrSchild (no
puncture boost)" in every `records.json`; the figure script refuses tilted spins
(`fig_eht_bbh.jl:204`).

## 2. What is missing against the SKS metric

M1. The boost. Each term of the SKS metric is f_A (Λ_A l^A)_μ (Λ_A l^A)_ν with Λ_A the
Lorentz boost of the hole's 3-velocity (arXiv:2403.13308 eq. sks, l.259–263); the paper
notes that dropping it is a first-order error in v even at large separation (l.265). BLP
omits it. For one hole of mass 1/2 at its own radius r' = 2–3, the boosted and unboosted
perturbations h = g − η differ by 20% of max|h| at v = 0.112 (sep 20) and 27% at v = 0.158
(sep 10) (`checks/blp_superposed_check.log`). The holes of the imaged dumps move at
v = 0.10 (sep 25, q = 1; `records.json` `puncture_velocity`). [SOLID]

M2. Spin direction. Spins are restricted to ±z; the SKS term takes any a^i
(eq. covectorBH). The dumps imaged so far have a = 0; `fig_eht_bbh.jl` maps |a| > m_A to
the dimensionless spin and errors on th_a ≠ 0. [SOLID]

M3. Time dependence (addressed 2026-10-19 by `BoostedSuperposedKerrSchild`, see §3).
`SuperposedKerrSchild <: StationarySpacetime`: the holes do not move
during a ray's passage and ∂_t g ≡ 0. A ray crosses the near zone of a sep-25 binary in
~50–100 M of coordinate time, during which each hole moves 5–10 M, more than its horizon
diameter. The slow-light machinery for the fluid exists (`FrameWindow`, `SnapshotSeries`,
emission time t_obs + x⁰) but there is no time-dependent spacetime type other than
`UserMetric`, whose generic `geodesic_rhs` already keeps ∂_t g (metric_ops.jl:22–25). [SOLID]

M4. Merger interpolation and the orbit table of the paper (sections tranmerger and
numerical implementation) have no counterpart. [SOLID]

Consequence for the published binary images. The AthenaK dynbbh runs were evolved on the
SKS metric of their own code (`use_traj_table = 0`, analytic circular orbit, sep 25, q 1,
a = 0 in the dump header). That implementation is `/data/haiyangw/athenak/src/pgen/dynbbh.cpp`
(read 2026-10-19): `SuperposedBBH` (l.1356–) forms each hole's coordinates with the spatial
Lorentz contraction of x − ξ_A along v_A (`x1BH1 = (x − ξ) − (Γ−1)(v·(x−ξ)) v_x/v²`, l.1430–1435)
and `numerical_4metric` (l.1060–1130) differentiates the metric in time along the
trajectory (h = 5e-5), so the simulation metric is boosted and time dependent; it also
excises r' < |a|(1 + buffer) + floor by clamping z' and scales each mass by
`adjust_mass`. BLP's static unboosted superposition therefore differs from the metric the
fluid was evolved on by the boost terms quantified above. [SOLID] BLP reconstructs the fluid
four-velocity from the dump's normal-frame velocity with its own metric (decode.jl) and
traces rays in it. If the simulation metric is boosted, u^μ, b^μ and the frame
transformations carry an O(v f) ≈ 10–20% error near each horizon, and the shadows sit
where a static hole's would: the time-dependent boosted metric displaces a transversely
moving hole's shadow by 4 v M = 0.4 M at v = 0.1 (`checks/frozen_vs_moving.log`), 3 µas at
the M87* scaling of the figure against a 20 µas beam. [PRELIMINARY until the dynbbh metric
is confirmed]

## 3. The extension must be time dependent, not frozen

Adding the boost to the static type (Λ_A l^A with fixed centres) reproduces the fork's
frozen metric, and the measurements in `fork_review.md` D1 apply: the frozen boosted hole's
r' = r_h surface is not null, its shadow shrinks by 19–62% for v = 0.05–0.20 and a third of
the rays stall. The time-dependent metric with the same formulas is exact (vacuum to
1e-16; capture edge equal to the static one for a collinear boost; a 4 v M shift for a
transverse one). The required type is therefore

- `BoostedSuperposedKerrSchild{N} <: Spacetime` (not Stationary): per hole mass, spin
  vector, and s_A(t), v_A(t) from an orbit table (the AthenaK analytic circular orbit for
  the existing dumps; the paper's HDF5 table in general), metric
  η + Σ_A f_A(x'_A) (Λ_A l'^A)(Λ_A l'^A) with x'_A = Λ_A^{(spatial)}(x − s_A(t)) — the
  construction verified in `checks/fork_metric_vacuum.jl`; AD through t gives ∂_t g;
- march control: per-hole r'_A in the boosted frame against r_h,A, the redshift cut as
  now, step from min_A(r'_A − r_h,A); the capture test must use r'_A at the ray's time;
- the fused kernels: the metric is isbits once the orbit table is a device array or the
  circular orbit a closed form; the stationary fast paths (`raise_and_norm` with three
  spatial seeds) are replaced by the four-column gradient for this type;
- the fluid at the emission time through the existing `FrameWindow`, so the metric time
  and the fluid time agree (t_emit = t_obs + x⁰).

Exact tests that exist for it: single boosted Kerr (vacuum; shadow = Lorentz-transformed
Kerr shadow, `checks/frozen_vs_moving.jl` gives the camera and the expected edges);
reduction to `SuperposedKerrSchild` at v = 0 bitwise; reduction to `KerrSchildCartesian`
for one hole at rest.

Status 2026-10-19: `BoostedSuperposedKerrSchild{N} <: Spacetime` exists
(`BLP/src/physics/Spacetime/boosted_superposed.jl`) with per-hole mass, spin vector and a
trajectory callable (`CircularOrbit`, or any closure generic in t), the adaptive tracer
and the march control with per-hole r'_A at the ray's time, and the tests of
`BLP/test/physics/spacetime_boosted.jl` (at rest = single Kerr and the static
superposition to 1e-12; uniform motion: null covector, Ricci < 1e-7, g = Λ g_rest Λ to
1e-12, ∂_t g = −v·∇g to 1e-10; capture by a moving hole; a miss when aimed at the frozen
position). Not done: an orbit table from a file, the fused kernels on the device, and the
`FrameWindow` coupling of the fluid time. Figures: `results/blp/dynrays/`.

## 4. Slow light across the three codes

| code | fluid | metric |
|---|---|---|
| Blacklight a7c7215 | `slow_light_on`, chunked snapshot slices, x⁰ bracketing (simulation_sampling.cpp:296–345) | Kerr, stationary; nothing to add |
| the fork | inherited, untested with the binary metric; the orbit time passed is the chunk's earliest slice | frozen at the first snapshot's time (fork_review.md D1, D3) |
| BLP 0.3.0 | `SnapshotSeries`/`FrameWindow`, t_emit = t_obs + x⁰ | stationary types only; `UserMetric` keeps ∂_t g but has no capture surface |

What "adding slow light" means for the binary in both the fork and BLP is the same item:
the metric at the ray's own time, with ∂_t g retained, and the fluid frame chosen at that
time.
