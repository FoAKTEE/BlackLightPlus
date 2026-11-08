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
| fig_capture (Z4c slice of a Brill–Lindquist head-on collision) and fig_summary (c) | `fig_z4c_phasespace.jl`, `fig_z4c_amr.jl` | the puncture positions of the slice, capture map, bisected boundary, ray deck; the class map and refinement levels of the image-plane AMR |
| fig_binary (binary GRMHD images with EVPA ticks) and fig_summary (d) | `fig_eht_bbh.jl` | 230 GHz Stokes images of three binary snapshots |
| fig_dynrays (rays through the orbiting binary against the frozen metric) | `fig_dynrays.jl` | the orbital-plane ray fan, the capture-class maps at four observer times and the freeze-time scan |
| fig_dynrays_slowlight (fast and slow light of the GRMHD binary) | `fig_dynrays_slowlight.jl` | fast- and slow-light Stokes I of snapshot 00455 at i = 90°, the sky-plane crossing times, the epoch scan, the light-travel delays and the translation scan |
| fig_dyntests (conserved charges of rays in time-dependent spacetimes) | `fig_dyntests.jl` | the step-constant sweep of a moving Kerr hole and an orbiting binary, and the cross-check against `trace` |
| none; the CPU-against-GPU numbers of the dynamical-spacetime tests | `fig_dyntests_gpu.jl` (inside a Slurm GPU job, CUDA environment named in its header) | trace and shadow of the fig_dyntests spacetimes on the CPU threads and the GPU |
| none; the two limits of the dynamical-spacetime tests | `fig_dyntests_limits.jl` | the binary at zero orbital frequency against the static superposition, and slow light on a static fluid series against fast light |
| none in the manuscript; the i = 90° deck of `bench/bench.jl` | `fig_bbh_i90_pol_amr.jl` | the adaptive full-Stokes image of snapshot 00455 at i = 90° |
| fig_spherical_orbits and fig_spherical_convergence (Appendix) | `fig_spherical_orbits.jl` | the four unstable spherical photon orbits at a = M and their RK4 step-size sweep |
| fig_raptor_cmp (Appendix) | `fig_raptor_r07.jl` | the polarized images of snapshot 00455 compared with RAPTOR |
| fig_mc_dyn_tests and fig_mc_dyn_symmetry (the Monte Carlo over an emission window) | `fig_mc_dyn_tests.jl` | the light-curve damping of a variable sphere against 3 j₁(x)/x, the light echo of a scattering shell, the frozen sphere and Kerr torus against the stationary run, the errors against the photon count, and co-rotating plasmas on an orbiting binary binned by azimuth sector |
| fig_mc_dyn_binary and fig_mc_dyn_symmetry (c) (light curves of the circumbinary flow) | `fig_mc_dyn_binary.jl` | snapshot 00455 carried along the orbit of the orbiting binary metric: window-mean spectra and the time-resolved sums by arrival bin, inclination band, azimuth sector and frequency, with the helical-symmetry statistics |
| fig_mc_dyn_binary_hi (converged azimuth-time maps of the circumbinary flow) | `fig_mc_dyn_binary_hi.jl` | the run of `fig_mc_dyn_binary.jl` and a run without scattering at eight times the primaries, whose unscattered light fills every cell of the azimuth-time map |
| fig_cmp_disk, fig_cmp_disk_b and fig_cmp_disk_cuts (Appendix, analytic thin-disk comparison) | `fig_cmp_disk.jl` (shared definitions in `fig_cmp_disk_common.jl`), then `fig_cmp_disk_krang.jl` in an environment with Krang.jl and this package; AART runs from its own checkout on the `configs.json` that `fig_cmp_disk.jl` writes | per configuration (A, B): the intensity and landing radius of image orders 0, 1, 2 on AART's grid nodes and on radial cuts through the photon rings, from the closed form, the RK4 march, the precise march and Krang; the critical curve; the step-divisor sweep of the RK4 march |
| fig_performance | `bench/bench.jl` with `bench/slurm/` | the benchmark records |
| fig_pipeline | none (a drawn schematic) | — |

The printed figures as images, their run records and the comparison records against the
reference implementation are on the `blp` branch of this repository
(`results/blp/figures_blacklightplus/`, `results/blp/`); the plotting scripts are in the
manuscript repository.
