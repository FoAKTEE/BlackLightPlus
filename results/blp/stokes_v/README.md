# Stokes V between BlackLight+ and Blacklight: the cause

Scope: the circular-polarization disagreement found in the KPolaris figure reproduction
(`results/blp/figures_kpolaris/README.md`, figure 1): on the torus snapshot 00999 imaged
at 256², i = 60°, d = 100 M, BlackLight+'s Stokes V was 1.4–5× Blacklight's in summed
magnitude with correlation 0.13 (230 GHz) and 0.39 (1 THz), while I agreed to NMSE 6e-3
and Q, U agreed at 1 THz but diverged at 230 GHz (NMSE 0.37, net U/I of opposite sign).

## What was compared

- Coefficients: both codes evaluate the same thermal fits (Marszewski et al. 2021 forms;
  BlackLight+'s `thermal_synchrotron_polarized` is a transcription of Blacklight's
  `ThermalSynchrotronPolarized`), with the same cold-electron threshold Θ_e = 0.01
  (BlackLight+ blends ρ_Q to zero over 0.005–0.01) and the same signed pitch angle
  (`cos θ_B` from k·b in the fluid tetrad, k the received wave vector).
- Transfer matrix: both use the Landi Degl'Innocenti 1985 / ipole convention
  `K = [α_I α_Q 0 α_V; α_Q α_I ρ_V 0; 0 −ρ_V α_I ρ_Q; α_V 0 −ρ_Q α_I]`; Blacklight's
  pure-rotation step reproduces it, and RAPTOR's matrix is the same.
- BlackLight+'s segment operator against the matrix exponential `exp(−K ds)` on 4000
  random admissible coefficient sets (α_P < α_I, |j_P| < j_I, ρ_P up to 10, ds up to 3):
  median deviation 2e-15, maximum 1.6e-10 (`segment` rescales inadmissible inputs onto
  the Stokes cone by design; those were excluded).
- Blacklight's unsplit coupling (`image_rotation_split = false`, the path of the
  comparison deck), `polarized.cpp` lines 672–703 (mirror a7c7215): the coupling matrices
  M₂ and M₃ of the 1985 solution assign their (U,V) entry to the (Q,U) slot:

  ```
  mm_2[1][2] = s * lambda_1 * alpha_s[3] + lambda_2 * rho_s[3];
  mm_2[1][2] = s * lambda_1 * alpha_s[1] + lambda_2 * rho_s[1];   // should be mm_2[2][3]
  ...
  mm_3[1][2] = -(s * lambda_2 * alpha_s[3] - lambda_1 * rho_s[3]);
  mm_3[1][2] = -(s * lambda_2 * alpha_s[1] - lambda_1 * rho_s[1]); // should be mm_3[2][3]
  ```

  so `mm_2[2][3]`, `mm_3[2][3]` (and their antisymmetric partners) stay zero and the (Q,U)
  entries carry the (U,V) values. These entries are the U↔V coupling through which Faraday
  conversion acts whenever absorption and rotation are both present; the error vanishes as
  ρ → 0. Upstream master carries the same lines (`upstream_master_lines.txt`, fetched
  2026-10-23).

## The test

`build_run_fix.sh` builds Blacklight with the two indices corrected
(`blacklight_polarized_m2m3.patch`, nothing else changed) and runs the comparison deck at
230 GHz and 1 THz; `stats_fix.py` compares BlackLight+'s images with the stock and the
corrected build (`stats_fixm23.txt`):

| | stock Blacklight | corrected Blacklight |
|---|---|---|
| 230 GHz, V: NMSE / correlation / Σ\|V\| ratio / sign agreement | 52 / 0.13 / 5.25 / 0.64 | 0.041 / 0.980 / 1.01 / 0.99 |
| 230 GHz, Q, U: NMSE | 0.37, 0.31 | 0.012, 0.011 |
| 230 GHz, net U/I (BlackLight+ +0.0448) | −0.0348 | +0.0461 |
| 230 GHz, net V/I (BlackLight+ +0.0041) | +0.0010 | +0.0039 |
| 1 THz, V: NMSE / correlation / ratio / sign | 4.6 / 0.39 / 1.39 / 0.91 | 0.076 / 0.966 / 1.03 / 0.99 |
| 1 THz, Q, U: NMSE | 0.043, 0.039 | 0.042, 0.039 |
| I: NMSE (230 GHz, 1 THz) | 5.7e-3, 2.9e-2 | 5.5e-3, 2.9e-2 |

The residual after the fix is the sampling and decode difference already measured in I.
BlackLight+'s polarized transfer needs no change. [SOLID]

Also observed, not investigated: Blacklight with `image_rotation_split = true` (the
Strang-split path, `blacklight_split_230.input`) gives I 27–43% high and Q, U
anti-correlated with BlackLight+ (correlation −0.25 and −0.35 at 1 THz), so that path has
a problem of its own. The variant builds with ρ zeroed (`build_run_variants.sh`) ran, but
their BlackLight+ counterparts (`blp_variants.jl`) stopped on a domain error in the
redefined fit and were not needed once the corrected build settled the question.

Files: the patch, the build-and-run scripts, the decks of the corrected and split runs,
the comparison statistics and the upstream line check. The Blacklight source copy, the
converted dumps and the images are in the job's temporary directory and are not kept.
