# Blacklight paper (arXiv:2203.15963, White 2022) — the 17 figures with BlackLightPlus

> The figure images and run records of this set are kept on the `blp` branch under `results/blp/figures_blacklight/`; this branch holds the generators and this standing table.

Generators `figures/blacklight/fig_*.jl` (Julia, `julia -t 32 --project=.`, from the
repository root), plotters `figures/blacklight/plot_*.py`, data under
`figures/paper/out/blacklight/<set>/` (not tracked; each set has a `record.json` with the
setup, timings and measured numbers and the BlackLightPlus commit). The reference rasters are the
paper's `tex/figures/*.pdf`.

## The fiducial system

The paper's fiducial data set, the thick aligned disk of White 2020 (Athena++), is not on
this machine. Every fiducial-deck figure here uses the AthenaK torus dump
`torus.mhd_w_bcc.00999.athdf` of the single-hole series at
`/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/` (spin 0.9, 819 dumps, cadence
30 M, the deck of `figures/paper/fig_pol_hires.jl`): the electron model is that deck's
(R_low = 1, R_high = 10, γ = 13/9, total-pressure split, σ ≤ 1 emits), with the paper's free
scales: hole mass 4.152e6 Msun, source distance 8.178 kpc, and the density mass scale set
so that the fiducial image carries 2.4 Jy at 230 GHz (`fig_calibrate.jl`: 7.709e15 g =
3.342e-20 L³, on a `variants` ladder of 25 scales, confirmed by one 256² render at 2.400 Jy).
The paper's camera is kept where it transfers: plane-parallel, full width 24 M, centre at
r = 100 M and θ = 45°, 256² pixels, 230 GHz at the camera, the march budget C = 80 and at
most 4000 steps. Numbers below are from this substitute system, so they agree with the
paper in kind, not in value.

## Standing

| Figure | The paper | Here | Measured | Standing |
|---|---|---|---|---|
| 1 deflection_values | midplane Kerr deflection of 51 rays, a = 0.9, camera at r = 1000 M, DP at 1e-8, against the finite-distance Iyer & Hansen formula | the same rays with the adaptive Vern7 tracer at reltol = abstol = 1e-8, each ray run to its own starting radius, the formula integrated to that radius | 37 rays outside the photon ring (b₋ = −6.83, b₊ = 2.84), |error| 7e-10 … 1.8e-8 (median 3.9e-9), 48–89 steps | [SOLID] |
| 2 deflection_errors | errors against step count for DP (tolerances 1e-12 … 1e-5), RK4 and RK2 (steps 0.002 … 0.1) | Vern7 over the same tolerances; the fixed-control RK4 march with step (r − r_h)/C over C = 2 … 1000 (BlackLightPlus has no RK2) | Vern7: 24–42 steps at 1e-5 (errors 1e-6 … 1e-5) to 148–292 at 1e-12 (floor 5e-9); RK4: C = 2, 20–36 steps, errors up to 7e-2; C = 50, 500–900 steps, errors below 6e-8; floor 9e-9 from C = 100 | [SOLID] |
| 3 formula | the five parameterized models of Gold et al. 2020 at 230 GHz | BlackLightPlus's fixed-control march (C = 100) traces the rays of the plane-parallel camera at r = 1000 M, inclination 60°, width 30 M, 128²; the model coefficients evaluated along each stored ray from the definitions of Blacklight's `formula_coefficients.cpp` with the parameters of Gold et al. Table 1 and the unpolarized transfer integrated with BlackLightPlus's segment operator | fluxes 1.6429, 1.4388, 0.4423, 0.2708, 0.0255 Jy against the exact 1.6465, 1.4360, 0.4418, 0.2710, 0.0255 Jy (within 0.2%) at the source distance 8.25 kpc that puts the five on the exact values together (raw ratios 0.944–0.948, i.e. the only free factor is the pixel solid angle); the paper reports 1.6602, 1.4494, 0.4464, 0.2730, 0.0257 | [SOLID] |
| 4 stokes | I, Q, U, V of the fiducial image | the same, IQUV render of the fiducial deck | 2.364 Jy in the polarized march (2.400 unpolarized), net LP 52%, net CP 0.24% | [SOLID] (in kind) |
| 5 cuts | σ ≤ 1 fiducial; β⁻¹ ≥ 2 only; near and far half-spaces about the plane through the origin orthogonal to the line of sight | the cuts applied to the primitives through a masked provider (β⁻¹ = b²/2p from the primitives and the Kerr metric) | 2.400, 0.707, 0.922, 1.510 Jy | [SOLID] (in kind) |
| 6 flat | fiducial beside the image traced in flat spacetime | the same deck on `Minkowski()` (rays and plasma four-vectors flat, as Blacklight's `ray_flat`) | Kerr 2.400 Jy, flat 4.318 Jy; the flat image lacks the photon ring and the magnification | [SOLID] (in kind) |
| 7 slow | fast light at t = 10500 M against slow light over 160 dumps 1 M apart, camera at r = 25 M | fast light on dump 00998 (t = 78540 M) against slow light sampling dumps 00993 … 00999 (t = 78390 … 78570 M, cadence 30 M) at each ray's emission time t_obs + x⁰ with t_obs = 78570 M (`SeriesProvider`, bracketing dumps blended linearly); camera at r = 25 M | fast 2.445 Jy, slow 2.441 Jy; the longest ray spans 120.9 M of coordinate time, so the slow-light image draws on five of the seven dumps; 7 s against 50 s | [PRELIMINARY]: the 30 M cadence interpolates the fluid over 30 M where the paper resolved 1 M |
| 8 adaptive_grid | schematic of the block refinement | drawn (no data) | — | [SOLID] |
| 9 adaptive | root 32² in 8² blocks, three levels on the absolute intensity gradient (5e-5 cgs on ≥ 5 pixels): I, τ, V | `render_amr` with `AbsoluteGradient(5e-5; count = 5)`, block 8, three levels, IQUV; τ = ∫α_I dλ from the traced rays at the root and at the effective 256², the refined τ shown at each block's level | 14336 rays for the refined image against 65536 at a uniform 256²; levels of the 16 base blocks 9/2/0/5; root 2.324 Jy, refined 2.361 Jy; root 5.1 s, refined 3.1 s | [SOLID] (in kind) |
| 10 nonthermal | thermal; power law p = 3, γ = 4 … 1000; κ = 4, w = 1/2; equal mix; nonthermal parameters chosen to match the thermal flux | the same distributions through `PlasmaConfig`, the mass scale of each nonthermal case set on a `variants` ladder to the fiducial 2.4 Jy | mass scales 7.13×, 6.31×, 1.88× the fiducial for power law, kappa, mixed; fluxes 2.400, 2.407, 2.411, 2.404 Jy | [SOLID] (in kind) |
| 11 auxiliary_length | elapsed time and proper length per ray | from `trace` (fixed-control march): camera time minus the ray's end time, Σ (g_ij dx^i dx^j)^½ over the stored steps | elapsed time 134–291 M and proper length 103–255 M across the 24 M field (camera at r = 100 M) | [SOLID] (in kind) |
| 12 auxiliary_average | λ, I_j, τ; Θ_e averaged by λ, by emission, by optical depth | the same from the traced rays sampled along their paths with BlackLightPlus's plasma decode and coefficients; ⟨q⟩_τ integrated as d⟨q⟩/dτ = q − ⟨q⟩ source → camera | τ up to 0.59; the three Θ_e averages differ as in the paper: the affine average is dominated by the long cool outer path, the emission average by the hot ring, the optical-depth average by the absorbing foreground | [SOLID] (in kind) |
| 13 render_edge | false-colour renderings at θ = 70°: ρ > 5e-18 g cm⁻³ blue (τ_s = 1/30 M⁻¹), σ > 1 red (1/25), β⁻¹ = 1 crossings green (opacity 0.25); width 60 M with r < 45 M, flat and Kerr; width 24 M with r < 20 M and z < −r_h, flat and Kerr | the same rules on the traced rays, colours mixed in CIE XYZ and projected to sRGB; "flat" traces and decodes on `Minkowski()`; the density threshold is ρ > ρ_max/20 of the dump (the paper's 5e-18 g cm⁻³ belongs to its density scale; the dump's densities are 1e-20 … 3e-20 g cm⁻³ at the calibrated mass scale) | four renderings, 25 s in all; the curved panels show the lensed ring of the dense disk and the magnetized funnel above and below the hole | [SOLID] (in kind) |
| 14 render_face | face-on: ρ > 2e-17 g cm⁻³ (τ_s = 1/5), the 230 GHz image, Θ_e > 20 (τ_s = 1); matter inside r = 10 M | the same with ρ > ρ_max/5 and Θ_e > 50 (every electron inside r = 10 M of the torus is above Θ_e = 20) | the 230 GHz image of the matter inside r = 10 M carries 1.934 Jy; the dense regions trace the ring, the hot electrons fill the funnel | [SOLID] (in kind) |
| 15 true | true-colour images at θ = 0°, 45°, 90° from 12 frequencies 152–324 GHz boosted so that 230 GHz → 550 nm, with the 230 GHz images | 12 Stokes I renders per inclination, I'_λ ∝ ν² I_ν at λ' = 550 nm × 230 GHz/ν, integrated against the CIE 1931 2° matching functions (Wyman et al. 2013 fit), exposure at the 99.7th percentile of Y, sRGB | 230 GHz fluxes 2.400 (45°; the fiducial), face-on and edge-on as recorded; the images are near-white with a blue tint on the approaching side, as the paper finds | [SOLID] (in kind) |
| 16 scaling_strong | rays per second per thread against threads at 256² on four machines | BlackLightPlus on this node (144 logical cores shared with two concurrent workloads), the 256² fiducial image timed warm at 1, 2, 4, 8, 16, 32, 64 threads | 2893, 2813, 2552, 2341, 2120, 1871, 2185 rays s⁻¹ thread⁻¹ at 1, 2, 4, 8, 16, 32, 64 threads (0.47 s for 256² at 64 threads); the paper measures 100–1000 rays s⁻¹ thread⁻¹ | [PRELIMINARY]: the node was shared during the measurement |
| 17 scaling_resolution | time per pixel against resolution at four threads, one image and ten at once | BlackLightPlus at four threads: 64² … 1024², one image and ten images through `variants` (ten mass scales, one trace) | one image 113, 96, 91, 85, 86 μs per pixel at 64², 128², 256², 512², 1024²; ten images at once 71, 33, 32, 30, 31 μs per pixel per image | [PRELIMINARY]: shared node |

## Notes

- Figures 1–2 (`fig_deflection.jl`, `plot_deflection.py`). The formula of the paper is
  evaluated with the camera radius of each ray (r = (1000² + α²)^½), the escape event is
  placed at that radius, and the deflection is the unwrapped azimuth change of the
  backward trace; both ends at one radius cancel the Kerr-Schild/Boyer-Lindquist azimuth
  offset exactly. The fixed-control march overshoots the camera radius by its last step;
  its azimuth is interpolated linearly in 1/r over that step. The error floor of 5e-9 is the
  quadrature of the formula (400 Gauss-Legendre nodes after u = u₀ − w²).
- Figure 3 (`fig_formula.jl`, `plot_formula.py`; the same images draw figure 7 of the
  Mahakala reproduction). The camera frequency is 230 GHz at r = 1000 M (the paper's
  "frequencies of 230 GHz at the location of the camera"); the colour scale is the paper's,
  I_ν × 10⁵ (75/128)² / f with f the exact flux of each test.
- Figures 4, 5, 6 (`fig_fiducial.jl`, `plot_fiducial.py`), 10 (`fig_nonthermal.jl`).
- Figure 7 (`fig_slowlight.jl`). The series cadence (30 M) sets the time resolution of the blend; the ray span of 121 M follows from the camera at r = 25 M and the horizon approach (the paper: 159 M).
- Figure 9 (`fig_amr.jl`). The refined τ is the fine-grid τ averaged over each block's
  pixels at its level, not a march at those pixel centres.
- Figures 11–14 (`fig_auxiliary.jl`). ρ in g cm⁻³ is the code density times the density
  unit; σ and β⁻¹ are the decoded state's; the σ cut is off for the renderings. Crossings
  are counted where β⁻¹ − 1 changes sign between consecutive valid samples; the torus
  crosses β⁻¹ = 1 many times along a ray (its plasma β fluctuates about 1), so the green
  blend saturates where the paper's disk shows 0–3 crossings.
- Figure 15 (`fig_truecolor.jl`).
- Figures 16–17 (`fig_scaling.jl`, `run_scaling.sh`). Each thread count is a separate `julia -t N` process; the image is rendered twice and the second timed. The node ran two other workloads at the time (the per-thread rate at 64 threads above that at 32 is this contention). The ten-image run is `variants` over ten mass scales: one geodesic trace, ten transfer lanes, which is why its cost per image is a third of a single image's.
