# Generators of the paper's figures

The data behind the figures of the BlackLight+ method paper. Each generator is run from
the repository root with `julia -t <threads> --project=. figures/paper/<generator>` and
writes arrays and a record under `figures/paper/out/<set>/` (not tracked); the manuscript's
own plotting scripts draw the PDFs from those arrays. The simulation data they read
(AthenaK GRMHD snapshots and a Z4c slice) are not part of this repository; the paths are
the script defaults or the `BLP_*` environment variables named in each header.

| Paper figure | Generator | What it writes |
|---|---|---|
| fig_analytic (exact-Kerr equatorial disk) and the analytic panel of fig_summary | `fig_examples.jl` | the analytic-disk image and a torus demonstration render |
| fig_polarized (polarized GRMHD torus) | `fig_pol_hires.jl` | Stokes I, Q, U, V of the torus snapshot and the run record |
| fig_binary (a), (e) and the ray panel of fig_summary | `fig_z4c_phasespace.jl`, `fig_z4c_amr.jl` | capture map, bisected boundary, ray deck; the class map and refinement levels of the image-plane AMR |
| fig_binary (b)–(d) | `fig_eht_bbh.jl` | 230 GHz Stokes images of three binary snapshots |
| fig_binary (f)–(h) | `fig_bbh_i90_pol_amr.jl` | the adaptive full-Stokes image of snapshot 00455 at i = 90° |
| fig_spherical_orbits (Appendix) | `fig_spherical_orbits.jl` | the four unstable spherical photon orbits at a = M |
| fig_raptor_cmp (Appendix) | `fig_raptor_r07.jl` | the polarized images of snapshot 00455 compared with RAPTOR |
| fig_performance | `bench/bench.jl` with `bench/slurm/` | the benchmark records |
| fig_pipeline | none (a drawn schematic) | — |

The arrays of the printed figures, the comparison records against the reference
implementation and the manuscript's plotting scripts are on the `blp` branch of this
repository (`figures/out`, `results/blp`, and the nested manuscript repository).
