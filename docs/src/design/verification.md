# Verification record

The measured state of the package `BlackLightPlus` (version 0.3.0) on the host `anta`
(2× Xeon Platinum 8360Y, 144 hardware threads; 2× A100 80 GB; Julia 1.12.6), as of
2026-10-18 on branch `blp`, whose evidence directories (`results/blp/`) this record cites. The standards these measurements are held to are in
[Architecture](@ref) (§6 continuity, §10 tests) and `test/structure.jl`; the
provenance of the paper's figures is in the manuscript's `figures/CAPTIONS.md`. This file
records the results and the commands that reproduce them.

## 1. Test suite

```
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

| Where | Files | Tests | Result |
|---|--:|--:|---|
| CPU (threads and host-device pools) | 58 | 14,327 | all pass, 1,394 of them structure gates |
| A100, inside a Slurm allocation | 58 | infra 913; structure and physics 12,801; pipeline 600 | all pass |

On the GPU the CUDA extension tests, the backend switch (`backend = :gpu`, `BLP_BACKEND`)
and every product's `:gpu` path run on the device; on the CPU they skip with a report. The
GPU run uses an environment in which `CUDA` is a direct dependency (§5).

## 2. Oracle against the reference implementation

```
julia -t 8 --project=test/oracle test/oracle/runtests.jl   # on the blp branch, CPU
BLP_TEST_DEVICES=cuda julia -t 4 --project=test/oracle -e 'using Test;
    Base.include(Module(:O), "test/oracle/products.jl")'                       # one script on the GPU
```

| Scripts | Device of the pools | Checks | Result |
|---|---|--:|---|
| all ten | host devices | 1,372 | all pass |
| `products.jl` | A100 | 85 | all pass |
| `spectra.jl` | A100 | 55 | all pass |
| `montecarlo.jl` | A100 | 63 | all pass |

Four classes of agreement hold, each stated in [Architecture](@ref) §6: bitwise on one
hardware (Stokes I on every path, the samplers, `decode`, shadows, the exact Monte Carlo
sums; on the A100 also the reference device drivers against BlackLightPlus, IQUV included);
round-off natively and bitwise under `julia -C generic` (IQUV, traces, the torus Monte
Carlo on the CPU); round-off only (the two-phase path, the analytic sources' four-vectors);
and across hardware (`Decks.CROSS_HARDWARE`), where a GPU pool against CPU threads differs
by the two targets' floating-point contraction.

| Cross-hardware deviation measured on the A100 | Oracle decks | Production decks |
|---|--:|--:|
| images, of the image maximum | 1e-14 to 1e-9 | 1e-12 to 3e-10 |
| Monte Carlo sums, relative, counts equal | 1e-16 | 1e-16 (thin) to 2e-12 (third scattering order) |

## 3. The paper's figures

Generators under `figures/paper/`, arrays under `figures/paper/out/` and `out/raptor/`
(the layout of the reference `figures/out/`), the PDFs drawn by the
manuscript's own plotters with their data root and output directory taken from the
environment:

```
julia -t 32 --project=. figures/paper/fig_examples.jl          # also fig_pol_hires, fig_spherical_orbits,
                                                                #   fig_z4c_phasespace, fig_z4c_amr, fig_eht_bbh,
                                                                #   fig_bbh_i90_pol_amr, fig_raptor_r07
export BLP_FIG_REPO=$PWD BLP_FIG_DIR=<pdf directory>
python3 paper/scripts/plot_<name>.py                           # analytic polarized spherical_orbits binary summary raptor_cmp
python3 bench/plot_performance.py
```

Every product runs with `backend = :cpu`; the wall times below are on the shared host.

| Figure | Generator | Data against the reference arrays | PDF against the published one | Wall |
|---|---|---|---|--:|
| fig_analytic | `fig_examples.jl` | 13 pixels within 8 ulp; equal to a rerun of the reference | identical at 60 dpi | 1 s |
| fig_polarized | `fig_pol_hires.jl` | IQUV within 4e-11 of the maximum | identical | 31 s, 32 threads |
| fig_spherical_orbits | `fig_spherical_orbits.jl` | RK4 trajectories equal to a rerun of the reference | 4 pixels of 103,356 | 8 s |
| fig_raptor_cmp | `fig_raptor_r07.jl` | I, Q, U, V within 2e-10; every RAPTOR metric equal in its listed digits | identical | 120 s, 48 threads |
| fig_binary (a, e) | `fig_z4c_amr.jl`, `fig_z4c_phasespace.jl` | class map, level maps, 1201² capture map and 288 anchors bitwise; ring within one bisection step | 1 pixel at 45 dpi | 1710 s + 1597 s, 48 threads |
| fig_binary (b–h) | `fig_eht_bbh.jl`, `fig_bbh_i90_pol_amr.jl` | fluxes 2.17, 0.69, 1.18 Jy and every record number equal to the recorded digits; Stokes I within 8e-10 of the maximum, V within 8e-7 of its own | with the above | 657 s + 136 s, 48 threads |
| fig_summary | the above | — | trajectory panel differs (see below) | — |
| fig_pipeline | `plot_pipeline.py` | schematic of [Architecture](@ref) §1–§2 | redrawn | — |
| fig_performance | `bench/` | §4 | new measurements | — |

Three of the stored reference data sets no longer reproduce with the reference package at
its current commit, while BlackLightPlus reproduces that package's rerun bitwise: the analytic disk
and the spherical orbits (rounding of an earlier environment) and the Z4c ring and deck
trajectories (the reference geodesic right-hand side changed at e253c10 after the published
run). The summary figure's trajectory panel differs for that reason. Three products the
figures needed are not entry points of BlackLightPlus and are assembled in the scripts from the
submodule interfaces: a per-level barrier with a custom per-ray value in `render_amr`
(the Z4c class refinement), ray classes per hole, and a coordinate-list ladder in
`variants`.

## 4. Benchmarks

`bench/README.md` states the protocol; `bench/results/anta_2026-10-18.md` holds
every measurement with its conditions (threads, device count, lanes, chunk, host load),
and `bench/plot_performance.py` draws them. Warm best of three after a first
call; both executors at their `auto` chunk (before 2026-10-18, `anta_2026-10-17.md`, the
threads at chunk 64 and the pool at the best chunk of a sweep); the GPU inside a Slurm
allocation of four hardware threads; both GPUs bitwise equal to one on every leg. An arrow
reads before → after the executor and product changes of [Performance](@ref); the
two-A100 column stands at its 2026-10-17 values, no two-device run having been recorded
since.

| Leg (BlackLightPlus deck: a = 0.9 torus, 230 GHz, 60°, fov 30) | 128 CPU threads | 1 A100 | 2 A100 |
|---|--:|--:|--:|
| Stokes I 768² (rays/s) | 1.39e5 → 1.69e5 | 1.20e6 → 1.24e6 | 2.07e6 |
| IQUV 384² (rays/s) | 4.63e4 → 5.57e4 | 2.34e5 → 2.51e5 | 3.58e5 |
| adaptive IQUV production image, 1536² effective (wall) | 11.2 s → 11.1 s | 2.97 s → 2.65 s | 1.82 s |
| exact-Kerr disk 1024² (rays/s; CPU 64 threads) | 3.02e6 → 5.60e6 | 7.24e6 → 7.78e6 | 6.70e6 |
| Monte Carlo thin sphere, 1e7 (ph/s; CPU 64 threads) | 2.44e6 → 2.46e6 | 2.14e7 → 2.76e7 | 3.77e7 |

The executors size their chunks from the item count, the output size and the device and
thread counts ([Performance](@ref) §2), so no leg depends on a chunk setting: on one
A100 the `auto` chunk matches or beats the best chunk of the earlier sweeps on every leg,
and the Monte Carlo scattering sphere, whose generations now run inside each chunk, goes
from 1.65 s to 0.84 s.

### 4.1 Against the reference package

```
G=reference=fused_384,fused_768,fused_mixed_768,fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384,amr_driver_pol
G=$G/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768,amr_384
bench/slurm/run_reference.sh "1@$G" "2@$G" 1@reference=bbh455_480,bbh455_pol_480,bbh122_pol_480,bbh_i90_amr_pol
L=amr_384,amr_pol_384,amr_driver_pol,analytic_1024,mc_thin,mc_scatter
bench/slurm/run_reference_cpu.sh 16:reference:$L 64:reference:$L 128:reference:$L 128:torus:img_192,img_384,img_768,amr_384
julia --project=. bench/bench.jl merge-reference         # results/anta_<date>_reference.{json,md}
```

`bench/reference.jl` times the reference package (`src/`) on the legs of its own
harness `bench/bench_multigpu.jl`, included unchanged, and `bench.jl` with
`BLP_DECK=reference` times the BlackLightPlus calls of the same legs. The two halves of a GPU row run
in one Slurm allocation, back to back on the same device, from two environments that hold
the same CUDA.jl, GPUCompiler, LLVM.jl and KernelAbstractions (`results/blp/gpu/env_old`,
`results/blp/gpu/env`); the CPU rows run on the login shell at a load below 16. Two
records exist: `bench/results/anta_2026-10-17_reference.md`, BlackLightPlus at 65481d9, before
the executor and product changes of [Performance](@ref), and
`anta_2026-10-18_reference.md`, BlackLightPlus at 7be2569, after them; each holds every
configuration, the CUPTI phases of one profiled call per leg and the Slurm history.
`fig_performance.pdf` draws the reference's torus-deck series in grey beside BlackLightPlus's.

On the same hardware the two packages agree, before and after: on the A100 every pair of
outputs is bitwise identical (the 15 reference-deck legs and the 4 torus-deck legs); on
the CPU the analytic and Monte Carlo outputs are bitwise identical and the images agree to
1e-13 to 2.6e-11 of the peak on the reference deck and to 7e-10 on the torus deck (the
reference's CPU render is two-phase, BlackLightPlus's fused), with equal refinement levels.

The ratios below are BlackLightPlus's rate over the reference's; above 1, BlackLightPlus is faster. "Default"
pairs the two defaults: the reference's harness call (pool lanes 2, chunk 65,536, the
Monte Carlo drivers' 2^20 primaries) against BlackLightPlus at lanes 2, chunk 65,536, or threads at
chunk 4,096 before, and against BlackLightPlus's `auto` chunk after. "Best" pairs the best
configuration of each package (reference also at chunk 2^17; BlackLightPlus at `auto`, 2^16 and 2^17,
and before also 2^20 on the Monte Carlo and binary legs; threads at `auto`, 4,096 and 64).
An arrow reads before → after.

| Leg, one A100 (reference deck: a = 0.9 torus, fov 40, R_high 40) | reference (s) | BlackLightPlus default (s) | default | best |
|---|--:|--:|--:|--:|
| fused I 384² / 768², uniform resample 256³ | 0.102 / 0.334 | 0.097 → 0.078 / 0.329 → 0.289 | 1.05 → 1.30 / 1.02 → 1.16 | 1.12 → 1.20 / 1.01 → 1.02 |
| fused I 768², mixed precision | 0.256 | 0.308 → 0.278 | 0.83 → 0.92 | 0.91 → 0.88 |
| fused IQUV 256² / 384² | 0.244 / 0.414 | 0.219 → 0.211 / 0.421 → 0.339 | 1.12 → 1.15 / 0.98 → 1.22 | 1.13 → 1.15 / 1.28 → 1.22 |
| native blocks I / IQUV 384² | 0.138 / 0.622 | 0.140 → 0.125 / 0.710 → 0.568 | 0.98 → 1.11 / 0.88 → 1.10 | 1.05 → 1.07 / 0.83 → 0.90 |
| adaptive IQUV, base 384, block 8 | 5.32 | 5.98 → 4.59 | 0.89 → 1.16 | 0.90 → 0.99 |
| exact-Kerr disk 1024² | 0.098 → 0.087 | 0.321 → 0.065 | 0.31 → 1.34 | 0.60 → 1.34 |
| Monte Carlo thin / scattering sphere, 1e7 | 0.410 / 0.752 | 0.686 → 0.357 / 1.88 → 0.742 | 0.60 → 1.15 / 0.40 → 1.01 | 1.13 → 1.15 / 0.54 → 1.07 |
| binary 00455 I / IQUV 480² | 1.39 / 3.76 | 2.05 → 1.56 / 3.94 → 2.93 | 0.68 → 0.89 / 0.95 → 1.28 | 0.89 → 0.89 / 1.11 → 1.11 |
| binary 00122 IQUV 480²; 00455 at i = 90°, adaptive IQUV | 3.18; 16.0 | 3.63 → 2.77; 17.1 → 13.9 | 0.88 → 1.15; 0.94 → 1.15 | 1.02 → 1.03; 1.01 → 1.01 |
| BlackLightPlus deck: Stokes I 192² / 384² / 768²; adaptive IQUV | 0.057 / 0.148 / 0.553; 3.29 | 0.056 → 0.057 / 0.151 → 0.129 / 0.569 → 0.472; 3.42 → 2.74 | 1.02 → 1.00 / 0.98 → 1.14 / 0.97 → 1.17; 0.96 → 1.20 | 1.02 → 1.01 / 1.04 → 1.10 / 0.99 → 1.04; 0.91 → 0.99 |

| Leg, 128 CPU threads | reference (s) | BlackLightPlus default (s) | default | best |
|---|--:|--:|--:|--:|
| native blocks I / IQUV 384² | 4.80 / 6.71 | 0.842 / 2.53 | 1.68 → 5.70 / 1.01 → 2.65 | 4.32 → 5.81 / 2.32 → 2.65 |
| adaptive IQUV, base 384, block 8 | 42.1 | 18.6 | 1.73 → 2.27 | 2.10 → 2.27 |
| exact-Kerr disk 1024² | 0.451 | 0.229 | 1.44 → 1.97 | 1.44 → 2.38 |
| Monte Carlo thin / scattering sphere, 1e7 | 3.55 / 3.81 | 3.22 / 3.49 | 1.00 → 1.10 / 0.90 → 1.09 | 1.05 → 1.10 / 0.90 → 1.09 |
| BlackLightPlus deck: Stokes I 192² / 384² / 768²; adaptive IQUV | 1.75 / 4.77 / 11.6; 28.8 | 0.266 / 0.902 / 3.42; 11.0 | 0.60 → 6.58 / 1.64 → 5.29 / 1.91 → 3.40; 1.44 → 2.61 | 4.31 → 6.58 / 3.67 → 5.42 / 2.73 → 3.43; 2.54 → 2.62 |

| Geometric mean | default | best |
|---|--:|--:|
| one A100, 19 legs (15 reference deck, 4 BlackLightPlus deck) | 0.83 → 1.13 (0.89 to 1.34) | 0.95 → 1.06 (0.88 to 1.34) |
| CPU, 16 threads, 6 legs | 1.16 → 1.44 (0.97 to 2.03) | 1.32 → 1.47 (0.98 to 2.07) |
| CPU, 64 threads, 6 legs | 1.39 → 2.13 (1.27 to 3.42) | 1.79 → 2.30 (1.36 to 4.18) |
| CPU, 128 threads, 6 legs | 1.25 → 2.08 (1.09 to 5.70) | 1.75 → 2.16 (1.09 to 5.81) |

What moved each leg, from the profiled call of each record ([Performance](@ref) §3
lists the capabilities behind them):

- Exact-Kerr disk: the maps are written by the program in their final layout, so the host
  work after the last launch falls from 304 ms to 5 ms and the profiled call from 389 ms
  to 111 ms; the kernels take 56 ms in both packages, in 8 launches against 16. Its wall
  times vary by a factor two from call to call on both packages, from the collection of
  the previous call's 120 MB of maps (the reference's profiled call spends 193 ms before
  its first launch); its row is the best of seven calls of each package.
- Scattering sphere: every scattering generation of a chunk runs on the lane that took it,
  with the children in two device buffers; the device idles 161 ms of a 805 ms call
  against 916 of 1,944 ms, in 72 launches against 182, and 0.5 ms of device-to-host copies
  against 134 ms.
- Thin sphere, binary and native-block legs: the `auto` chunk is one launch per run of up
  to 2^20 items (10 launches against 153 on the thin sphere; one against four on the
  binary images), which is the reference's Monte Carlo chunk and the best explicit chunk
  of the earlier sweeps.
- Mixed precision (0.92) and the best native-block IQUV (0.90) and binary 00455 I (0.89)
  rows are the remaining kernel-level differences: the Float32 lane runs at 226 registers
  against the reference's 132 (blocks of 256 against 384), and both IQUV kernels sit at the
  255-register cap with BlackLightPlus's 16 percent slower at the same launch shape. A separate
  camera-ray launch, the reference's arrangement, leaves those register counts unchanged
  and is not faster, so the difference lies in the march itself and is tracked by the
  kernel audit (`bench/kernel_audit.jl`).
- CPU imaging: BlackLightPlus marches each ray once, where the reference traces, stores the path and
  integrates; the threaded executor's workers now pull chunks sized per run, so small
  images no longer idle threads.

First calls are 12 to 18 s for both packages on a leg that compiles its program (BlackLightPlus 23 s
on the adaptive leg; the reference harness compiles it in a setup run of 32.6 s); on the
CPU the reference's first calls are longer (23 against 6.5 s for native blocks I 384² at
128 threads). No two-device pair of either package has been recorded; the Slurm history
of the record lists the attempts.

## 5. Running on `anta`

The host is its own Slurm node. Outside an allocation the devices are not accessible
(`nvidia-smi` fails to initialise NVML); inside one they are:

```
srun -p preempt --gres=gpu:1 -c 2 -t 01:00:00 <command>      # at most 1 h and 2 CPUs per GPU
```

Jobs on `preempt` are cancelled without warning by other users' `priority` and `long`
jobs, so long runs are split into short, resumable jobs and resubmitted; this machine's job
scripts, runner and logs are under `results/blp/gpu/` and `bench/slurm/`.

CUDA.jl's runtime JLL, compiled where no driver is visible, selects no runtime; the
environments that load CUDA therefore pin it in a `LocalPreferences.toml`
(`[CUDA_Runtime_jll] version = "13.3"`, with `CUDA_Runtime_jll` in the project's
`[extras]`): `BlackLightPlus/`, `test/oracle/` and the test environment
`results/blp/gpu/env` (BlackLightPlus developed, `CUDA` a direct dependency).
