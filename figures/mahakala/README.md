# Mahakala (arXiv:2304.03804) figures with BlackLightPlus

> The figure images and run records of this set are kept on the `blp` branch under `results/blp/figures_mahakala/`; this branch holds the generators and this standing table.

The twelve figures of Sarkar et al. 2023 (Mahakala) traced with BlackLightPlus. The geodesic tests
(figures 1, 3–6) are the paper's setups; the GRMHD figures (2, 8–11) use the AthenaK
single-hole torus series on this machine in place of the paper's MAD and SANE a = 0.9375
runs, which are not available: spin 0.9, dumps
`/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.NNNNN.athdf`
(819 dumps 30 M apart, 344 meshblocks, 5 refinement levels), with the paper's hole mass
6.5e9 Msun, 230 GHz, plane-parallel camera at 1000 M and i = 17°, field of view 64 M,
electron temperature from the internal energy (γ_i = 5/3, γ_e = 4/3, R_low = 1), σ ≤ 1.
The torus's code density is not the paper's: the mass scale 𝓜_ref = 2e23 (ρ_unit = 𝓜/L³)
gives 0.55 Jy at 230 GHz for M87's distance at R_high = 10 and stands in for the paper's
𝓜 = 5e25 (a factor 250); the paper's ladder 5e23, 5e24, 1e25 maps to 𝓜_ref/100, /10, /5.
Figure 7 (the Gold et al. 2020 formula models) is produced with the Blacklight-paper
generator (`figures/blacklight/`).

Generators: `figures/mahakala/fig_geodesic_tests.jl` (figures 1, 3, 4, 5),
`figures/paper/fig_spherical_orbits.jl` (figure 6), `figures/mahakala/fig_grmhd.jl`
(figures 2, 8–11); plotters `plot_geodesic_tests.py`, `plot_grmhd.py`; data under
`figures/paper/out/mahakala/` (not tracked). Run from the repository root with
`julia -t 32 --project=. <generator>` then `python3 <plotter>`; `fig_grmhd.jl` takes the
figures to make from `BLP_MAHAKALA_PARTS` (default `9,8,2,11`) and the Blacklight binary
from `BLACKLIGHT` (built from ref-code/c-white-blacklight with its Makefile).

| Figure | Paper | Here | Standing |
|---|---|---|---|
| 1 `fig01_in_vs_out.png` | Schwarzschild rays forward and backward in time in the ingoing and outgoing Cartesian KS charts | `KerrSchildCartesian(0)` (ingoing) and a `UserMetric` with l_0 = −1 (outgoing); the inward photon integrated forward and the outward photon backward from r = 50, t = 0 with the adaptive Vern7 solve. Ingoing forward and outgoing backward reach r = 0.25 at t = ±49.75 (horizon-penetrating); ingoing backward and outgoing forward approach r = 2 asymptotically (t = ∓100 at r = 2.000). | [SOLID] |
| 2 `fig02_trajectories.png` | 3-d view of photon trajectories from the image plane through a GRMHD snapshot, coloured by synchrotron emission | 81 rays of the central ±16 M of a 48² camera of the torus deck (`trace`, 69,639 march points), the invariant emissivity j_ν/ν² at every point (three decades below its maximum), the central ±16 M of the 512² image of figure 9 on the image plane 30 M from the hole | [PRELIMINARY] torus in place of the MAD run |
| 3 `fig03_null_vs_c.png` | \|k·k\| against coordinate time for one ray at a = 0.9 and step control C = 20 … 100 | the prograde equatorial ray at b = 3.3 (b₊ = 2.844) from a camera at 100 M (the paper's distance), RK4 at the fixed-control step (r − r_h)/C: max \|g^{μν}p_μp_ν\| 3.6e-06 (C = 20), 9.0e-08 (50), 5.6e-09 (100), the rise to its plateau at the periastron \|t\| ≈ 110 M; the paper's integration is in coordinate time with step r/C and reaches 1e-10 … 1e-8 | [SOLID] same dependence on C, errors 10–100× the paper's for the affine-parameter march |
| 4 `fig04_deflection.png` | deflection angle of equatorial rays at a = 0.9 against b with the analytic result, errors ~1e-6 | 76 rays, b = −20 … 20 by 0.4 outside [b₋, b₊] = [−6.832, 2.844], camera at 10⁴ M, Vern7 reltol 1e-12; the angle from the wave-vector direction with the weak-field correction of the legs outside the camera and escape radii; exact value by quadrature of the equatorial Kerr integral: max error 7.5e-7, median 4.6e-7 | [SOLID] |
| 5 `fig05_convergence.png` | 4th-order convergence of max\|k·k\| with step size for orbits A, C, E, F | RK4 at Δλ = 1/4 … 1/1024 over 40 M or until the orbit leaves its sphere by 0.1 M: slopes −4.08, −4.19, −4.21, −4.38 over 1/4 … 1/128, round-off floor 5e-14 … 2e-13 at 1/1024 | [SOLID] |
| 6 `fig06_spherical_orbits.png` | the four unstable spherical photon orbits at a = M | the orbits of `fig_spherical_orbits.jl` (RK4 at Δλ = 1/1024, the part within 0.01 M of the sphere), ring singularity and equatorial circle drawn | [SOLID] |
| 7 `fig07_gold_models.png` | the five Gold et al. 2020 models | from the Blacklight-paper generator (`figures/blacklight/`) | see `results/blp/figures_blacklight/README.md` |
| 8 `fig08_code_comparison.png` | Mahakala against ipole on a MAD snapshot, NMSE 9.4e-3 | BlackLightPlus against Blacklight (ref-code/c-white-blacklight) on the torus snapshot 00999 at 256², R_high = 10, 𝓜_ref: NMSE 7.92e-04, flux ratio BlackLightPlus/Blacklight 1.0020 (the Blacklight array in the same orientation as BlackLightPlus's [i_β, i_α]; the dump rewritten in the Athena++ layout by `athdf_for_blacklight.py`, samples outside the mesh vacuum, Blacklight's Dormand–Prince integrator at tolerance 1e-8) | [SOLID] below the paper's 9.4e-3 and the 0.02 of the code comparison it cites |
| 9 `fig09_regions.png` | images at R_high = 10, 40, 160 and the three-colour decompositions by polar region and by ISCO and photon-orbit shells | the same at 512² on the torus (r_ms± = 2.32, 8.72; r_ph± = 1.56, 3.91 for a = 0.9), each region rendered with the plasma elsewhere set to vacuum | [PRELIMINARY] torus in place of the MAD run |
| 10 `fig10_cross_sections.png` | middle-row cross sections of figure 9 with the per-region flux fractions and the shadow | the same; shadow band from `shadow` (captured pixels of the middle row, α = -4.19 … 5.44 M); fractions R_high = 10: jet 8.1%, disk 91.2%, counter-jet 0.7%; ISCO shells 4.1/83.3/12.7%; photon-orbit shells 0.0/42.5/57.9%; R_high = 40: jet 6.6%, disk 92.8%, counter-jet 0.6%; ISCO shells 6.3/83.1/10.7%; photon-orbit shells 0.0/48.4/51.9%; R_high = 160: jet 5.1%, disk 94.5%, counter-jet 0.4%; ISCO shells 2.5/86.9/10.5%; photon-orbit shells 0.0/36.7/63.3% | [PRELIMINARY] |
| 11 `fig11_profiles.png` | per-shell I_ν/ν³ and ⟨j_ν/ν²⟩ against radius for the R_high, 𝓜 and i ladders, averaged over 50 MAD snapshots | per-shell ΔI_ν/ν³ (the invariant intensity change accumulated inside the shell along the marched rays, absorption of deeper light included, per pixel) and the affine-path-weighted ⟨j_ν/ν²⟩ over the ray samples, shells of 0.5 M, averaged over the 10 dumps 00909 … 00999 (300 M apart), 192² cameras | [PRELIMINARY] torus in place of MAD, 10 dumps |
| 12 | the same for the SANE run | no second single-hole GRMHD run is on this machine (`/data/haiyangw/learn/torus-grmhd` holds derived 64³ cubes of the same kind of run, not AthenaK dumps) | [HOLE] |

## Notes

- Figure 2: the rays are coloured by log j_ν/ν² over four decades below its maximum
  (inferno; black where the emission is below that), the paper's convention.
- Figure 1: the forward integration of the inward photon uses the state p = k (dt/dλ > 0),
  the backward integration of the outward photon p = −k; both charts are traced with the
  general geodesic right-hand side of BlackLightPlus (`geodesic_rhs`) and stop at |t| = 100, r = 0.25
  or r = 60.
- Figures 3 and 5: the paper integrates in coordinate time (its step is Δt, the Mahakala
  option used for these tests) while BlackLightPlus marches in the affine parameter; near the horizon
  and on the spherical orbits dt/dλ = k^t ≈ 5–10, so the paper's nominal step is 5–10× the
  smaller affine step, which for a fourth-order scheme accounts for the 10³–10⁴ lower
  error amplitudes of its figure 5 (and the 10–100× of its figure 3) at equal nominal
  step; the orders of convergence agree.
- Figure 3: the paper's error is measured along an integration in coordinate time with
  the step r/C; BlackLightPlus marches in the affine parameter with the step (r − r_h)/C, the control
  of its imaging products (C = 80 by default). The error rises through the periastron and
  settles afterwards in both.
- Figure 4: at the camera distance 1000 M of the paper the coordinate direction of the
  wave vector differs from the asymptotic one by f·α/r ≈ 4e-5, which exceeds the
  integration error; the camera is therefore at 10⁴ M, with the bending outside the camera
  and escape radii (2 M b/r²) added analytically. The exact angle is
  δ = 2∫_{r₀}^∞ [(b − a) + a(r² + a² − ab)/Δ]/√R dr − π with
  R = (r² + a² − ab)² − Δ(b − a)², integrated after r = r₀/(1 − s²) with the composite
  Simpson rule on 2²⁰ panels.
- Figures 9–11: a region's image is the render with the plasma outside the region
  replaced by vacuum, so light from the region is neither absorbed nor emitted outside it
  (the paper's choice); the three colours are the region images scaled by the full
  image's maximum and raised to the power 0.4. Figure 11's per-shell intensity keeps the
  absorption of light from deeper shells.
