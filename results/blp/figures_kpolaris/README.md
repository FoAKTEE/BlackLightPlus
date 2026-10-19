# KPolaris (arXiv:2609.26298) figures reproduced with BLP

Generators in `BLP/figures/kpolaris/` (run from the repository root), data under
`BLP/figures/out/kpolaris/` (not tracked), the curated figures and records here. The
paper's data (KHARMA snapshots, an analytic RIAF) and its reference code (ipole) are not
on this machine; every figure is traced on the AthenaK Fishbone–Moncrief torus
`torus.mhd_w_bcc.00999.athdf` (a = 0.9, 344 blocks of 32³, five levels, γ = 13/9; the
819-dump series at Δt = 30 M under `/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/`)
with the fiducial plasma deck of `BLP/figures/fig_pol_hires.jl` (M = 6.2×10⁹ M☉, mass
scale 10⁻²¹ L³, R_low 1, R_high 10, total-pressure split, σ cut 1), and the independent
code is Blacklight (`ref-code/c-white-blacklight` @ a7c7215, built with its Makefile),
which reads the dumps in AthenaK's binary layout (`athdf_to_bin.jl`).

| figure | paper | here | numbers | standing |
|---|---|---|---|---|
| 1 `fig1_code_comparison.png` | KPolaris vs ipole, four models at 230 GHz: reference intensity and signed residuals; NMSE per Stokes at 230 and 345 GHz | BLP vs Blacklight on the torus: fast light at 230 and 345 GHz, slow light over dumps 00990–00999 (t_obs = 78570) at both frequencies; 256², i = 60°, d = 100 M, fov 30 M, C = 80 in both codes | NMSE(I) 5.7×10⁻³, 1.1×10⁻², 5.4×10⁻³, 1.1×10⁻² (fast 230, fast 345, slow 230, slow 345); flux within −0.03%, +0.17%, −0.29%, −0.34%; NMSE(Q, U) 0.37/0.31 and 0.36/0.40 at 230 GHz, 0.083/0.088 and 0.090/0.105 at 345 GHz; NMSE(V) 44–62; m_L,net BLP 5.3, 4.2, 7.5, 6.0% vs Blacklight 4.8, 6.5, 8.7, 8.3% | I [SOLID] in kind: the two codes agree in flux and morphology, and the residual does not change under step refinement (both at C = 320: 6.6×10⁻³; BLP's own C = 80 vs 320: 3.2×10⁻⁵), so it is a sampling or plasma-decode difference between the codes, not integration error. Q, U [PRELIMINARY]: they agree where Faraday effects are weak (1 THz: NMSE 0.043/0.039, correlation 0.98, equal amplitude) and diverge with the Faraday depth. V [HOLE]: BLP's V is 2–7× Blacklight's at every frequency (1 THz: NMSE 4.6, correlation 0.39), a circular-polarization convention or coefficient difference between the two codes that is not reconciled here. RIAF column [HOLE]: no RIAF source in BLP, no independent code for it. |
| 2 `fig2_accuracy_cost.png` | GPU integration time vs largest Stokes NMSE for a geometrical-tolerance ladder 10⁻¹³–10⁻⁸ on three models, the adopted 10⁻¹⁰ ringed | BLP's control is the step divisor C of the fixed-control march (step (r − r_h)/C, limit 40 C): ladder C = 10 … 640 against C = 2560, 256² full Stokes at 230 GHz on the fiducial deck (d = 1000 M); one A100 (Slurm) and 32 CPU threads; medians of three calls after a warm-up | A100: 0.04, 0.08, 0.16, 0.32, 0.64, 1.27, 2.63 s for max NMSE 1.8×10⁻², 6.3×10⁻³, 4.7×10⁻⁴, 1.9×10⁻⁴, 6.3×10⁻⁵, 1.3×10⁻⁵, 3.1×10⁻⁶ (NMSE(I) 2.0×10⁻³ … 5.1×10⁻⁷); CPU 0.38 … 20.4 s for the same errors; reference 10.4 s (A100), 88 s (CPU); adopted C = 80: 0.32 s, max NMSE 1.9×10⁻⁴ | [SOLID] in kind: the same quantity (time against self-convergence error) for BLP's own accuracy control; the control parameter differs from KPolaris', so the curves are not comparable point by point |
| 3 `fig3_performance.png` | complete image time vs resolution 256²–2048² on three GPUs | `render` time (median and range of three calls after a warm-up) at 256², 512², 1024², 2048² full Stokes, 230 GHz, fiducial deck, on one A100, two A100s and 32 CPU threads; the snapshot load (5–6 s) and plan preparation (≤ 1 s) recorded apart | one A100 0.30, 0.97, 3.81, 15.4 s; two A100 0.33, 0.64, 2.20, 7.88 s; CPU 2.74, 11.3, 40.3, 167 s | [SOLID] in kind: the render time scales with the pixel count on every backend; the paper's "complete image" time includes initialization, upload and output, recorded here separately; the CPU times were taken on the shared node with two other workers running |
| 4 `fig4a_image_ray.png`, `fig4b_poincare.png` | selected ray in a SANE snapshot at 86 GHz, i = 85°: image with EVPA ticks, the geodesic with emission and conversion intervals over a density slice, Poincaré spheres with and without conversion on the outgoing segment | the torus at 86 GHz, i = 85°, d = 1000 M, fov 40 M, 256²; the pixel of largest |V| among those brighter than 0.3 max I ((α, β) = (−3.36, −1.17), I = 2.8×10⁻⁴ cgs, LP 0.42, V/I 0.25); the polarized march of `BLP.Transfer.march(::IQUV)` replayed with every segment operator stored in the transported camera frame and composed source → camera | the replay reproduces the pixel's Stokes vector to five digits; 1445 samples, 1083 emitting; outgoing segment of 419 samples (I within 1% of its final value); full transfer Π/I 49.1%, V/I 25.5%; without conversion on the segment Π/I 49.1%, V/I 21.3%; emission interval samples 485–425, conversion interval 458–410 (camera-ordered) | [SOLID] in kind: the same diagnostics from BLP's own operators; the conversion effect on this ray is smaller than on the paper's (its V/I goes from −12.8% to −2.0%) |

## Figure 1: registration test of the intensity residual

The 0.57% residual at 230 GHz is not an image registration difference: shifting the
Blacklight image by −1 … +1 pixel in steps of 0.25 (bilinear) leaves the minimum NMSE at the
unshifted position (5.71×10⁻³), rescaling it by 0.99 … 1.01 raises the NMSE to
1.7×10⁻² … 7.9×10⁻², and every transpose or flip raises it to ≥ 0.46. The residual's
dipole pattern along the ring edges is therefore the sampling and plasma-decode difference
between the codes at the sharpest gradients, as the step-refinement test above also shows.

## Figure 1: settings and convention mapping

Blacklight deck (`blacklight_decks/`): `simulation_format = athenak` on the converted
dumps (the dump's input deck is embedded as the binary header's parameter block, so
Blacklight reads γ from `<mhd> gamma`); `simulation_coord = cks`, a = 0.9,
M = 6.2×10⁹ M☉, `simulation_rho_cgs = 1e-21`; plane camera at Kerr–Schild r = 100,
θ = 60°, φ = 0, `camera_k_r = 1` (photons received along the outward radial direction,
BLP's inward plane normal reversed), width 30 M, rotation 0, 256²; RK4 with
`ray_step = 1/80` and additive termination at r_h + 10⁻³ (BLP: C = 80 with the horizon
backstop 10⁻³ and the redshift cut E_n > 100); `image_normalization = infinity`,
polarized; `plasma_model = ti_te_beta`, `plasma_use_p`, μ = 0.5, n_e/n_i = 1,
R_low 1, R_high 10, γ = 13/9, `cut_sigma_max = 1`; slow light with
`simulation_file = torus_{05d}.bin`, files 990–999, chunk 10, `slow_t_start = 78570`
(BLP's `t_obs`), interpolation on. Blacklight's image array `[row, column]` is BLP's
`[iβ, iα]` as stored: NMSE(I) 5.7×10⁻³ for the identity against ≥ 0.46 for every other
dihedral transform (`records/fig1_nmse.json`). Blacklight marks 46 of 65536 geodesics as
ending unexpectedly and writes NaN for them (287 pixels fast, 381 slow); those pixels are
left out of every sum. BLP: `fig_comparison.jl` (fast: `render` of the plan; slow:
`SeriesProvider(SnapshotSeries(paths, times; t_obs))`, emission time t_obs + x⁰ bracketed
and blended linearly between dumps). The ten dumps cover the emission times of every
sample: with the camera at 100 M the latest-emitting samples sit 110–200 M before t_obs,
and Blacklight refuses extrapolation beyond its loaded chunk.

Diagnostics of the residuals (`blacklight_decks/fast_1000.*`, `fast_230_c320.*`;
BLP images `blp_fast_1000_*`, `blp_fast_230_c320_*`): at 1 THz, where Faraday rotation
and conversion are negligible, NMSE(I) 2.9×10⁻² (flux +0.7%), NMSE(Q, U) 0.043/0.039 with
correlation 0.98 and equal RMS amplitude, NMSE(V) 4.6 with correlation 0.39 and BLP's RMS
2.3× Blacklight's; at 230 GHz with both codes at C = 320, NMSE(I) 6.6×10⁻³ and the
Q, U, V values of the C = 80 pair. So the Q, U differences at 230–345 GHz follow the
Faraday depth, and the V difference is intrinsic to the circular-polarization coefficients
or their sign convention in one of the two codes. BLP's Stokes signs are fixed against
IEEE/IAU by thin-slab tests (`docs/src/validation.md`, R02) and its images against RAPTOR
(R07, R09: L₁(V) ≤ 10⁻⁵ with the matched solver); the corresponding Blacklight check is
not made here.

Run: `julia --project=BLP BLP/figures/kpolaris/athdf_to_bin.jl <out_dir> <athdf…>`;
`<blacklight>/bin/blacklight <deck>`; `julia -t 32 --project=BLP
BLP/figures/kpolaris/fig_comparison.jl`; `python3 BLP/figures/kpolaris/plot_comparison.py
<deck dir>`.

## Figures 2 and 3: runs

`fig_accuracy_cost.jl` and `fig_performance.jl` with `BLP_BACKEND=cpu` (32 threads,
`julia -t 32 --project=BLP`) and inside Slurm GPU jobs
(`sbatch --gres=gpu:N BLP/figures/kpolaris/gpu_timing.sbatch N <script>`, the environment
`results/blp/gpu/env`, 4 Julia threads; logs in `records/`); `plot_timing.py` draws both.
GPU times are `render` on an idle A100 (the second GPU run at 256² had one slow call of
3.3 s among its three, kept as the range bar). Records: `records/fig2_accuracy_*.json`,
`records/fig3_performance_*.json`.

## Figure 4: method

`fig_selected_ray.jl` renders the image, selects the pixel, and replays its march: the
same loop as `BLP.Transfer.march(::IQUV)` (fixed control, RK4 transport of the camera
frame, `coefficients`, `fluid_tetrad_checked`, `segment`, `qu_rotation`), storing each
sample's Mueller operator and emission vector rotated into the transported camera frame,
and the same operators with ρ_Q = 0. The Stokes vector after each segment is the
source → camera composition; the intervals are the 5–95% span of the cumulative
|j_V ds| and |ρ_Q U_fluid ds| contributions weighted by the subsequent scalar
transmission, in forward path order; the displayed segment starts at the first sample
after which the invariant intensity stays within 1% of its final value, and the
no-conversion replay starts from the same incident Stokes vector. `plot_selected_ray.py`
draws the panels (EVPA χ = ½ atan2(U, Q) east of north, north = +β, east = −α;
the density slice is ρ in the z = 0 plane). Record: `records/fig4_record.json`.
