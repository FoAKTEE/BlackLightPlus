# BLP verification record

The measured state of the package `BLP` (`BLP/`, version 0.2.0) on the host `anta`
(2× Xeon Platinum 8360Y, 144 hardware threads; 2× A100 80 GB; Julia 1.12.6), as of
2026-10-17 on branch `blp`. The standards these measurements are held to are in
`docs/ARCHITECTURE.md` (§6 continuity, §10 tests) and `BLP/test/structure.jl`; the
provenance of the paper's figures is in the manuscript's `figures/CAPTIONS.md`. This file
records the results and the commands that reproduce them.

## 1. Test suite

```
julia -t auto --project=BLP -e 'using Pkg; Pkg.test()'
```

| Where | Files | Tests | Result |
|---|--:|--:|---|
| CPU (threads and host-device pools) | 58 | 14,121 | all pass, 1,377 of them structure gates |
| A100, inside a Slurm allocation | 58 | infra 792; structure and physics 12,784; the 13 pipeline files | all pass |

On the GPU the CUDA extension tests, the backend switch (`backend = :gpu`, `BLP_BACKEND`)
and every product's `:gpu` path run on the device; on the CPU they skip with a report. The
GPU run uses an environment in which `CUDA` is a direct dependency (§5).

## 2. Oracle against the reference implementation

```
julia -t 8 --project=BLP/test/oracle BLP/test/oracle/runtests.jl                     # CPU
BLP_TEST_DEVICES=cuda julia -t 4 --project=BLP/test/oracle -e 'using Test;
    Base.include(Module(:O), "BLP/test/oracle/products.jl")'                       # one script on the GPU
```

| Scripts | Device of the pools | Checks | Result |
|---|---|--:|---|
| all ten | host devices | 1,372 | all pass |
| `products.jl` | A100 | 85 | all pass |
| `spectra.jl` | A100 | 55 | all pass |
| `montecarlo.jl` | A100 | 63 | all pass |

Four classes of agreement hold, each stated in `docs/ARCHITECTURE.md` §6: bitwise on one
hardware (Stokes I on every path, the samplers, `decode`, shadows, the exact Monte Carlo
sums; on the A100 also the reference device drivers against BLP, IQUV included);
round-off natively and bitwise under `julia -C generic` (IQUV, traces, the torus Monte
Carlo on the CPU); round-off only (the two-phase path, the analytic sources' four-vectors);
and across hardware (`Decks.CROSS_HARDWARE`), where a GPU pool against CPU threads differs
by the two targets' floating-point contraction.

| Cross-hardware deviation measured on the A100 | Oracle decks | Production decks |
|---|--:|--:|
| images, of the image maximum | 1e-14 to 1e-9 | 1e-12 to 3e-10 |
| Monte Carlo sums, relative, counts equal | 1e-16 | 1e-16 (thin) to 2e-12 (third scattering order) |

## 3. The paper's figures

Generators under `BLP/figures/`, arrays under `BLP/figures/out/` and `BLP/out/raptor/`
(the layout of the reference `figures/out/`), PDFs under `BLP/figures/paper/`, drawn by the
manuscript's own plotters with their data root and output directory taken from the
environment:

```
julia -t 32 --project=BLP BLP/figures/fig_examples.jl          # also fig_pol_hires, fig_spherical_orbits,
                                                                #   fig_z4c_phasespace, fig_z4c_amr, fig_eht_bbh,
                                                                #   fig_bbh_i90_pol_amr, r07_blp
export BLP_FIG_REPO=$PWD/BLP BLP_FIG_DIR=$PWD/BLP/figures/paper
python3 paper/scripts/plot_<name>.py                           # analytic polarized spherical_orbits binary summary raptor_cmp
python3 BLP/figures/plot_pipeline.py; python3 BLP/bench/plot_performance.py
python3 BLP/figures/compare_arrays.py <set>; python3 BLP/figures/compare_bbh.py; python3 BLP/figures/compare_z4c.py
```

Every product runs with `backend = :cpu`; the wall times below are on the shared host.

| Figure | Generator | Data against the reference arrays | PDF against the published one | Wall |
|---|---|---|---|--:|
| fig_analytic | `fig_examples.jl` | 13 pixels within 8 ulp; equal to a rerun of the reference | identical at 60 dpi | 1 s |
| fig_polarized | `fig_pol_hires.jl` | IQUV within 4e-11 of the maximum | identical | 31 s, 32 threads |
| fig_spherical_orbits | `fig_spherical_orbits.jl` | RK4 trajectories equal to a rerun of the reference | 4 pixels of 103,356 | 8 s |
| fig_raptor_cmp | `r07_blp.jl` | I, Q, U, V within 2e-10; every RAPTOR metric equal in its listed digits | identical | 120 s, 48 threads |
| fig_binary (a, e) | `fig_z4c_amr.jl`, `fig_z4c_phasespace.jl` | class map, level maps, 1201² capture map and 288 anchors bitwise; ring within one bisection step | 1 pixel at 45 dpi | 1710 s + 1597 s, 48 threads |
| fig_binary (b–h) | `fig_eht_bbh.jl`, `fig_bbh_i90_pol_amr.jl` | fluxes 2.17, 0.69, 1.18 Jy and every record number equal to the recorded digits; Stokes I within 8e-10 of the maximum, V within 8e-7 of its own | with the above | 657 s + 136 s, 48 threads |
| fig_summary | the above | — | trajectory panel differs (see below) | — |
| fig_pipeline | `plot_pipeline.py` | schematic of `docs/ARCHITECTURE.md` §1–§2 | redrawn | — |
| fig_performance | `BLP/bench/` | §4 | new measurements | — |

Three of the stored reference data sets no longer reproduce with the reference package at
its current commit, while BLP reproduces that package's rerun bitwise: the analytic disk
and the spherical orbits (rounding of an earlier environment) and the Z4c ring and deck
trajectories (the reference geodesic right-hand side changed at e253c10 after the published
run). The summary figure's trajectory panel differs for that reason. Three products the
figures needed are not entry points of BLP and are assembled in the scripts from the
submodule interfaces: a per-level barrier with a custom per-ray value in `render_amr`
(the Z4c class refinement), ray classes per hole, and a coordinate-list ladder in
`variants`.

## 4. Benchmarks

`BLP/bench/README.md` states the protocol; `BLP/bench/results/anta_2026-10-17.md` holds
every measurement with its conditions (threads, device count, lanes, chunk, host load),
and `BLP/figures/paper/fig_performance.pdf` draws them. Warm best of three after a first
call; the CPU executor at chunk 64; the GPU inside a Slurm allocation of four hardware
threads; both GPUs bitwise equal to one on every leg.

| Leg (BLP deck: a = 0.9 torus, 230 GHz, 60°, fov 30) | 128 CPU threads | 1 A100 | 2 A100 |
|---|--:|--:|--:|
| Stokes I 768² (rays/s) | 1.39e5 | 1.20e6 | 2.07e6 |
| IQUV 384² (rays/s) | 4.63e4 | 2.34e5 | 3.58e5 |
| adaptive IQUV production image, 1536² effective (wall) | 11.2 s | 2.97 s | 1.82 s |
| exact-Kerr disk 1024² (rays/s; CPU 64 threads) | 3.02e6 | 7.24e6 | 6.70e6 |
| Monte Carlo thin sphere, 1e7 (ph/s; CPU 64 threads) | 2.44e6 | 2.14e7 | 3.77e7 |

Two executor settings matter more than the hardware: the threaded executor's default
chunk of 4096 idles threads on small images (192² at 128 threads runs 7× slower than at
chunk 64), and the pool's default chunk of 65,536 is 14 to 36 percent below 2^17; neither
executor sizes chunks from the item and device counts.

### 4.1 Against the reference package

```
BLP/bench/slurm/run_reference.sh "1@reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384/reference=amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384"
BLP/bench/slurm/run_reference.sh 1@reference=bbh455_480,bbh455_pol_480/reference=bbh122_pol_480/reference=bbh_i90_amr_pol
BLP/bench/slurm/run_reference_cpu.sh 128:reference:amr_384,amr_pol_384,amr_driver_pol,analytic_1024,mc_thin,mc_scatter 128:torus:img_192,img_384,img_768,amr_384
julia --project=BLP BLP/bench/bench.jl merge-reference         # results/anta_<date>_reference.{json,md}
```

`BLP/bench/reference.jl` times the reference package (`src/`) on the legs of its own
harness `bench/bench_multigpu.jl`, included unchanged, and `bench.jl` with
`BLP_DECK=reference` times the BLP calls of the same legs. The two halves of a GPU row run
in one Slurm allocation, back to back on the same device, from two environments that hold
the same CUDA.jl, GPUCompiler, LLVM.jl and KernelAbstractions (`results/blp/gpu/env_old`,
`results/blp/gpu/env`); the CPU rows run on the login shell at a load below 16.
`BLP/bench/results/anta_2026-10-17_reference.md` holds every configuration, the CUPTI
phases of one profiled call per leg and the Slurm history; `fig_performance.pdf` draws the
reference's torus-deck series in grey beside BLP's.

On the same hardware the two packages agree: on the A100 every pair of outputs is bitwise
identical (the 15 reference-deck legs and the 4 torus-deck legs); on the CPU the analytic
and Monte Carlo outputs are bitwise identical and the images agree to 1e-13 to 2.6e-11 of
the peak on the reference deck and to 7e-10 on the torus deck (the reference's CPU render
is two-phase, BLP's fused), with equal refinement levels.

The ratios below are BLP's rate over the reference's; above 1, BLP is faster. "Default"
pairs the two defaults (reference: the harness call, pool lanes 2, chunk 65,536, the Monte
Carlo drivers' 2^20 primaries; BLP: lanes 2, chunk 65,536, or threads at chunk 4,096);
"best" pairs the best configuration of each package (reference also at chunk 2^17; BLP at
2^17 and, on the Monte Carlo and binary legs, 2^20; threads at chunk 64).

| Leg, one A100 (reference deck: a = 0.9 torus, fov 40, R_high 40) | reference (s) | BLP default (s) | default | best |
|---|--:|--:|--:|--:|
| fused I 384² / 768², uniform resample 256³ | 0.102 / 0.334 | 0.097 / 0.329 | 1.05 / 1.02 | 1.12 / 1.01 |
| fused I 768², mixed precision | 0.256 | 0.308 | 0.83 | 0.91 |
| fused IQUV 256² / 384² | 0.244 / 0.414 | 0.219 / 0.421 | 1.12 / 0.98 | 1.13 / 1.28 |
| native blocks I / IQUV 384² | 0.138 / 0.622 | 0.140 / 0.710 | 0.98 / 0.88 | 1.05 / 0.83 |
| adaptive IQUV, base 384, block 8 | 5.32 | 5.98 | 0.89 | 0.90 |
| exact-Kerr disk 1024² | 0.098 | 0.321 | 0.31 | 0.60 |
| Monte Carlo thin / scattering sphere, 1e7 | 0.410 / 0.752 | 0.686 / 1.88 | 0.60 / 0.40 | 1.13 / 0.54 |
| binary 00455 I / IQUV 480² | 1.39 / 3.76 | 2.05 / 3.94 | 0.68 / 0.95 | 0.89 / 1.11 |
| binary 00122 IQUV 480²; 00455 at i = 90°, adaptive IQUV | 3.18; 16.0 | 3.63; 17.1 | 0.88; 0.94 | 1.02; 1.01 |
| BLP deck: Stokes I 192² / 384² / 768²; adaptive IQUV | 0.057 / 0.148 / 0.553; 3.29 | 0.056 / 0.151 / 0.569; 3.42 | 1.02 / 0.98 / 0.97; 0.96 | 1.02 / 1.04 / 0.99; 0.91 |

| Leg, 128 CPU threads | reference (s) | BLP best (s) | default | best |
|---|--:|--:|--:|--:|
| native blocks I / IQUV 384² | 4.80 / 6.71 | 1.11 / 2.89 | 1.68 / 1.01 | 4.32 / 2.32 |
| adaptive IQUV, base 384, block 8 | 42.1 | 20.1 | 1.73 | 2.10 |
| exact-Kerr disk 1024² | 0.451 | 0.314 | 1.44 | 1.44 |
| Monte Carlo thin / scattering sphere, 1e7 | 3.55 / 3.81 | 3.39 / 4.25 | 1.00 / 0.90 | 1.05 / 0.90 |
| BLP deck: Stokes I 192² / 384² / 768²; adaptive IQUV | 1.75 / 4.77 / 11.6; 28.8 | 0.41 / 1.30 / 4.25; 11.3 | 0.60 / 1.64 / 1.91; 1.44 | 4.31 / 3.67 / 2.73; 2.54 |

| Geometric mean over the reference-deck legs | default | best |
|---|--:|--:|
| one A100, 15 legs | 0.79 (0.31 to 1.12) | 0.95 (0.54 to 1.28) |
| CPU, 16 threads, 6 legs | 1.16 (0.97 to 1.60) | 1.32 (0.98 to 1.65) |
| CPU, 64 threads, 6 legs | 1.39 (1.01 to 1.98) | 1.79 (1.11 to 3.21) |
| CPU, 128 threads, 6 legs | 1.25 (0.90 to 1.73) | 1.75 (0.90 to 4.32) |

Where the two differ, from the profiled call of each leg:

- Exact-Kerr disk (0.31 at the defaults): BLP's kernels take 46 ms against the reference's
  59 ms, followed by 304 ms on the host. The per-pixel outputs are gathered as one vector
  of tuples and split into the band maps after the last chunk, where the reference copies
  each chunk into its maps while the others run.
- Scattering sphere (0.40): the device idles for 977 ms of a 2,014 ms call. Each
  scattering order completes over all photons before its children return to the host and
  go up again for the next order; the reference runs every order inside each 2^20-photon
  chunk. At the same 2^20 chunk BLP is still 1.85× slower.
- Thin sphere (0.60): the chunk alone, 153 launches of 65,536 against the drivers' 2^20.
  At equal chunks BLP is faster, 1.13× at 2^20.
- Mixed precision (0.83) and native-block IQUV (0.88): BLP computes the Float64 camera ray
  inside the march kernel, which costs the Float32 lane its register saving (226 against
  132 registers per thread, blocks of 256 against 384); both IQUV kernels sit at the
  255-register cap, and BLP's native-block kernel is 16 percent slower at the same launch
  shape.
- Binary 00455 I (0.68): the profiled calls differ by 3 percent (1.69 against 1.64 s); the
  default pool splits the image into four chunks over two lanes, and at 2^20 the call
  takes 1.55 s against 1.39 s.
- CPU imaging (BLP 1.2 to 4.3× faster): BLP marches each ray once, where the reference
  traces, stores the path and integrates, and gains little from 64 to 128 threads. Monte
  Carlo is equal within 10 percent.

First calls are 12 to 16 s for both packages on a leg that compiles its program (BLP 23 s
on the adaptive leg; the reference harness compiles it in a setup run of 32.6 s); on the
CPU the reference's first calls are longer (23 against 6.5 s for native blocks I 384² at
128 threads). The two-device rows of the comparison are not measured; the results file
lists them with the pending Slurm job.

## 5. Running on `anta`

The host is its own Slurm node. Outside an allocation the devices are not accessible
(`nvidia-smi` fails to initialise NVML); inside one they are:

```
srun -p preempt --gres=gpu:1 -c 2 -t 01:00:00 <command>      # at most 1 h and 2 CPUs per GPU
```

Jobs on `preempt` are cancelled without warning by other users' `priority` and `long`
jobs, so long runs are split into short, resumable jobs and resubmitted; this machine's job
scripts, runner and logs are under `results/blp/gpu/` and `BLP/bench/slurm/`.

CUDA.jl's runtime JLL, compiled where no driver is visible, selects no runtime; the
environments that load CUDA therefore pin it in a `LocalPreferences.toml`
(`[CUDA_Runtime_jll] version = "13.3"`, with `CUDA_Runtime_jll` in the project's
`[extras]`): `BLP/`, `BLP/test/oracle/` and the test environment
`results/blp/gpu/env` (BLP developed, `CUDA` a direct dependency).
