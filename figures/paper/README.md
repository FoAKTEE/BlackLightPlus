# Generators of the paper's figures

The data behind the figures of the BlackLight+ method paper. Each generator is run from
the repository root with `julia -t <threads> --project=. figures/paper/<generator>` and
writes arrays and a record under `figures/paper/out/<set>/` (not tracked); the manuscript's
own plotting scripts draw the PDFs from those arrays. The simulation data they read
(AthenaK GRMHD snapshots and a Z4c slice) are not part of this repository; the paths are
the script defaults or the `BLP_*` environment variables named in each header.

| Paper figure | Generator | What it writes |
|---|---|---|
| fig_analytic (exact-Kerr equatorial disk) and fig_summary (a) | `fig_examples.jl` | the analytic-disk image and a torus demonstration render |
| fig_polarized (polarized GRMHD torus) and fig_summary (b) | `fig_pol_hires.jl` | Stokes I, Q, U, V of the torus snapshot and the run record |
| fig_capture (Z4c binary slice) and fig_summary (c) | `fig_z4c_phasespace.jl`, `fig_z4c_amr.jl` | capture map, bisected boundary, ray deck; the class map and refinement levels of the image-plane AMR |
| fig_binary (binary GRMHD images with EVPA ticks) and fig_summary (d) | `fig_eht_bbh.jl` | 230 GHz Stokes images of three binary snapshots |
| none in the manuscript; the i = 90° deck of `bench/bench.jl` | `fig_bbh_i90_pol_amr.jl` | the adaptive full-Stokes image of snapshot 00455 at i = 90° |
| fig_spherical_orbits and fig_spherical_convergence (Appendix) | `fig_spherical_orbits.jl` | the four unstable spherical photon orbits at a = M and their RK4 step-size sweep |
| fig_raptor_cmp (Appendix) | `fig_raptor_r07.jl` | the polarized images of snapshot 00455 compared with RAPTOR |
| fig_performance | `bench/bench.jl` with `bench/slurm/` | the benchmark records |
| fig_pipeline | none (a drawn schematic) | — |

The printed figures as images, their run records and the comparison records against the
reference implementation are on the `blp` branch of this repository
(`results/blp/figures_blacklightplus/`, `results/blp/`); the plotting scripts are in the
manuscript repository.
