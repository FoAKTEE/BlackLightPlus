# BlackLight+ method paper (arXiv:2609.26298) figures

The eight figures of the manuscript as printed (manuscript repository commit `b1d2a43`),
rasterized at 200 dpi from its PDFs, and the run records of the data they are drawn from
(`records/`, the layout of `figures/out/`; each record holds the code SHA of its run). The
PDFs, the plotting scripts (`scripts/plot_*.py`, style `scripts/figstyle.py`) and the
per-figure setup and provenance (`figures/CAPTIONS.md`) are in the manuscript repository;
the arrays (`*.npy`) are not tracked. Generators are in `BLP/figures/` on this branch and
in `figures/paper/` on `main` (the same code with the module named `BlackLightPlus`); run
them from the repository root with `julia -t <threads> --project=BLP BLP/figures/<generator>`.

| Figure | Content | Generator | Records | Numbers |
|---|---|---|---|---|
| 1 `fig01_pipeline.png` | code structure | none (drawn schematic) | — | — |
| 2 `fig02_summary.png` | (a) exact-Kerr equatorial disk, (b) polarized GRMHD torus with EVPA ticks, (c) geodesics in the Z4c binary slice, (d) binary GRMHD image | `fig_examples.jl`, `fig_pol_hires.jl`, `fig_z4c_phasespace.jl`, `fig_eht_bbh.jl` | `examples_meta.txt`, `pol_hires/meta.txt`, `z4c_phasespace/deck.json`, `eht_bbh/s455_i90_Rh40/records.json` | (b) 94 ticks, largest cell LP fraction 0.74; (c) 42 deck rays, 5 bending more than 360° |
| 3 `fig03_capture.png` | ray classes and refinement levels of the image-plane AMR on the Z4c slice | `fig_z4c_phasespace.jl`, `fig_z4c_amr.jl` | `z4c_amr/records.json`, `z4c_amr/boundary_agreement.json`, `z4c_amr_l4/records.json`, `z4c_phasespace/deck.json` | 8192² effective from 824,832 rays (1.23%); 7.57% of pixels captured; 288 radial-scan anchors within median 6.7e-4 M, max 1.25e-3 M of the class edge (finest pixel 1.83e-3 M) |
| 4 `fig04_binary.png` | 230 GHz Stokes I of snapshots 00455, 00093, 00122 at i = 90° with EVPA ticks | `fig_eht_bbh.jl` | `eht_bbh/s{455,093,122}_i90_Rh40/records.json` | 2.17, 0.69, 1.18 Jy; 45, 205, 17 ticks with median cell LP fraction 0.056, 0.321, 0.361 |
| 5 `fig05_polarized.png` | Q, U, LP fraction and V of the torus snapshot 00999, 230 GHz, i = 60° | `fig_pol_hires.jl` | `pol_hires/meta.txt` | Σ\|P\|/ΣI = 0.518, Σ\|V\|/ΣI = 0.018, net \|ΣP\|/ΣI = 0.052 |
| 6 `fig06_spherical_orbits.png` | unstable spherical photon orbits A, C, E, F at a = M | `fig_spherical_orbits.jl` | `spherical_orbits/record.json` | \|r − r0\| > 1e-2 M at λ = 34.58, 43.61, 52.16, 67.56 (RK4, Δλ = 1/1024) |
| 7 `fig07_spherical_convergence.png` | RK4 step sweep Δλ = 1/4 … 1/1024 on the same orbits (Mahakala Fig. 5) | `fig_spherical_orbits.jl` | `spherical_orbits/record.json` | slopes of max\|H\| 3.93, 4.11, 4.13, 4.36 over 1/4 … 1/128; round-off floor (2.7–4.5)e-14 for E and F |
| 8 `fig08_raptor_cmp.png` | BlackLight+ and RAPTOR on snapshot 00455 | `r07_blp.jl` | `results/blp/raptor/evidence/R07/metrics.json` | ΔF/F = 8.3e-6, NMSE(I) = 1.5e-5 |
