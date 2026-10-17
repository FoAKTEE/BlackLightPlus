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
