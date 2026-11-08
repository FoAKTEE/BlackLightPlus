# GPU kernel profile of BlackLightPlus on the A100

Measured facts about the GPU execution of BlackLightPlus at `a5923dd` (worktree
`/data2/haiyangw/scratch/blp_wavefront`, clean in every job: `dirty 0` in each job header),
the evidence base of the wavefront and coherence design. Estimates are marked **(estimate)**;
everything else is a measurement whose method and raw files are named next to it. The
machine-readable form of every number is `profile.json`; scripts, logs and raw profiles are
under `profile/` (`scripts/`, `logs/`, `raw/`, `data/`).

## 1. Setup

| item | value |
|---|---|
| GPU | NVIDIA A100 80GB PCIe, 108 SMs, driver 580.105.08, node `anta`, Slurm partition `preempt` |
| clock during the runs | 1065 MHz SM clock in every sampled range (nsys GPU metrics, 2 kHz; ncu `smsp__cycles_elapsed.avg.per_second` = 1.065 GHz) |
| host | 4 CPU threads per job (`-c 4`), `julia -t 4` |
| software | Julia 1.12.6, CUDA.jl 6.2.2 (CUDACore), CUDA runtime 13.3 (LocalPreferences), CUDA_Driver_jll 13.3.1, KernelAbstractions 0.9.43, GPUCompiler 1.23.0 |
| profilers | Nsight Compute 2026.1.1 (counters permitted: `RmProfilingAdminOnly: 0`, probe job 3794), Nsight Systems 2025.6.3 |
| environment | `/data2/haiyangw/scratch/wavefront/env` (copy of `results/blp/gpu/env_main` developed against the worktree) |
| Slurm | the node exposes 4 CPUs to Slurm in total (`CfgTRES=cpu=4`), so one job runs at a time on the node, whatever its GPU count |

Decks (the decks of `results/blp/mc_audit/speed/gpu/README.md`):

- **MC torus**: AthenaK snapshot 00999, Kerr a = 0.9, stride 1 (10,303,459 zones), Ns 1e6
  (1,000,005 primaries), constant bias `Bias(1e4)`, `max_scatt = 3` for Compton (7,302,016
  children), plunge stop on, `CurvedMarch(C = 80, escape = 1e3)`, seed `0x5eed0700`. Chunks of
  250,002 primaries (`cascade_chunk` on one device: n/4).
- **torus imaging** (coherent baseline): `render`, Stokes I, 768², inclination 17°, fov 30.
- **standard march**: `render_crossings`, comparison B (a = 0.99, 80°, fov 24.02, 500²),
  production march.
- **precise march**: the same with `Plans.precise` (Verner 9).

## 2. Per-kernel metrics (Nsight Compute)

Method: `ncu --set full` plus the divergence and access metrics listed in
`scripts/ncu.sbatch`, application clocks (`--clock-control none`), kernel replay; the deck is
compiled and warmed outside the profiled range, one call runs between `cuProfilerStart` and
`cuProfilerStop` (`scripts/ncu_target.jl`) on a pool of one lane (`DevicePool(...; lanes = 1)`),
and the first launches whose demangled name matches the program are profiled. Reports
`raw/ncu_{torus,cross_prod,cross_prec}_3798.ncu-rep`, `raw/ncu_thin_3799.ncu-rep`,
`raw/ncu_compton_3807.ncu-rep`; extraction `scripts/ncu_extract.py` →
`data/ncu_*.json`. Every kernel is the package's single kernel `_foreach!` specialised on the
program named in the first column. ncu replays a kernel 53–56 times at about 290 times its
duration (the thin pair took 550 s), so the Compton kernels are profiled on the same zones
with Ns = 1.6e5 (160,159 primaries spread over the whole torus, chunks of 40,040): a run at
Ns = 1e6 did not finish its first kernel in 25 minutes. The Compton run uses `capacity = 4,000,000`
child slots so that no generation overflows: each profiled launch is one generation of the
first chunk (an overflow rerun is the same program over the same items; section 4.5).

### 2.1 Launch, resources, occupancy

| kernel (program) | duration (ncu) | regs/thread | stack (local) B/thread | spill instructions / all warp instructions | block | grid | smem | occupancy limit (registers) | theoretical / achieved occupancy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| torus imaging (`ImagingProgram`, 768², whole image) | 0.451 s | 255 | 1952 | 67.3 M / 64.2 G (0.10 %) | 256 | 2304 | 0 | 1 block (8 warps) / SM | 12.5 % / 12.3 % |
| standard march (`CrossingProgram`, first chunk) | 0.086 s | 255 | 2256 | 304 M / 10.1 G (3.0 %) | 256 | 512 | 0 | 1 block / SM | 12.5 % / 12.0 % |
| precise march (`CrossingProgram`, Verner 9, first chunk) | 1.600 s | 255 | 5632 | 976 M / 23.2 G (4.2 %) | 256 | 512 | 0 | 1 block / SM | 12.5 % / 9.4 % |
| MC thin primary (`MCPrimary`, chunk 1) | 0.962 s | 255 | 3184 | 311 M / 73.8 G (0.42 %) | 32 | 7813 | 0 | 8 blocks (8 warps) / SM | 12.5 % / 9.36 % |
| MC thin primary (`MCPrimary`, chunk 2) | 0.970 s | 255 | 3184 | 330 M / 78.3 G (0.42 %) | 32 | 7813 | 0 | 8 blocks / SM | 12.5 % / 9.25 % |
| MC Compton primary (`MCPrimary`, chunk 1, Ns 1.6e5) | 2.008 s | 255 | 6352 | 4.90 G / 104.1 G (4.7 %) | 32 | 1252 | 0 | 8 blocks / SM | 12.5 % / 7.79 % |
| MC Compton layer, generation 1 (`MCLayer`, chunk 1) | 0.808 s | 255 | 6352 (kernel 5448) | 661 M / 14.2 G (4.7 %) | 32 | 466 | 0 | 8 blocks / SM | 12.5 % / 4.01 % |
| MC Compton layer, generation 2 | 0.693 s | 255 | 6352 (kernel 5448) | 647 M / 14.4 G (4.5 %) | 32 | 1292 | 0 | 8 blocks / SM | 12.5 % / 8.73 % |
| MC Compton layer, generation 3 | 0.277 s | 255 | 6352 (kernel 5448) | 750 M / 19.4 G (3.9 %) | 32 | 7662 | 0 | 8 blocks / SM | 12.5 % / 10.5 % |

Compile-time view (`ptxas -arch=sm_80 -O3 -v` of CUDA 13.3, the runtime of the environment,
on the PTX that `CUDA.@device_code` dumps for the same program types, `scripts/devcode.jl`,
job 3801, `raw/devcode/`, `scripts/ptxas_stats.py` → `data/ptxas.json`; ncu's
`launch__stack_size` is the context's per-thread stack, 8 B above the largest frame of the
kernels loaded so far, so the layer launches report the primary's 6352 B):

| kernel | registers | stack frame (cumulative) | spill stores / loads in the kernel body | spill stores / loads in non-inlined callees |
|---|--:|--:|---|---|
| torus imaging | 255 | 1944 B | 12 / 0 B | `camera_state` 908 / 1088 B, `ImagingProgram` 24 / 36 B, `afoldl` 0 / 68 B |
| standard march | 255 | 2248 B | 12 / 0 B | `CrossingProgram` 1008 / 872 B, `camera_state` 944 / 1120 B, `afoldl` 0 / 72 B |
| precise march | 255 | 5624 B | 12 / 0 B | `CrossingProgram` 2304 / 4368 B, `camera_state` 476 / 820 B |
| MC thin primary | 255 | 3176 B | 0 / 0 B | `track!` 144 / 168 B, `MCPrimary` 24 / 0 B |
| MC Compton primary | 255 | 6352 B | 4 / 4 B | `track!` 1704 / 2568 B, `MCPrimary` 8 / 0 B |
| MC Compton layer | 255 | 5448 B | 0 / 0 B | `track!` 1696 / 2536 B |
| `Drain` (cascade) | 32 | 1656 B | 0 / 0 B | none |

The stack frame holds spill slots and local arrays (values indexed at run time); with scattering
compiled in, `track!` spills about 12 times more than without (1704 / 2568 B against
144 / 168 B) and the frame doubles (6352 B against 3176 B).

Every large kernel sits at the 255-register cap (256 allocated), so a streaming
multiprocessor holds at most 8 warps (12.5 % of its 64): one block of 256 threads for the
imaging and crossing programs, eight blocks of 32 for the Monte Carlo programs. The small
programs: `ZoneSums` and `ZoneCounts` 72 registers (block 896), `HotcrossSigma` 62 (block
1024), `Drain` 32 (block 1024) (nsys kernel records, `data/tl_3797.json`). Shared memory is
0 in every kernel. CUPTI's `localMemoryPerThread` in the nsys trace reads 0 for every kernel;
the ptxas frames, ncu's `launch__stack_size` and the SASS spill counters are the local-memory
measure.

### 2.2 Execution efficiency, memory, pipes

| kernel | active threads per warp instruction (of 32) | branch targets uniform | divergent branch targets | global load sectors per request | L1 hit (all / local loads) | L2 hit | DRAM read / write, throughput | local memory traffic | FP64 pipe | issue active | warps active / eligible per scheduler |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| torus imaging | 31.3 | 99.89 % | 0.11 % | 5.77 | 99.1 % / 99.1 % | 74.6 % | 0.98 / 0.60 GB, 0.18 % | 39.4 GB | 71.2 % | 31.9 % | 1.98 / 0.42 |
| standard march | 29.1 | 99.94 % | 0.06 % | 3.0 | 99.2 % / 99.6 % | 100 % | 0.03 / 0.24 GB, 0.16 % | 68.8 GB | 69.6 % | 28.7 % | 1.96 / 0.39 |
| precise march | 26.2 | 99.98 % | 0.02 % | 3.0 | 59.8 % / 79.4 % | 92.3 % | 14.7 / 69.3 GB, 2.7 % | 294 GB | 39.5 % | 18.4 % | 1.77 / 0.23 |
| MC thin primary, chunk 1 | 6.37 | 98.70 % | 1.30 % | 5.40 | 96.6 % / 95.1 % | 72.5 % | 4.96 GB read, 0.31 % | 67.1 GB | 46.0 % | 23.1 % | 1.68 / 0.27 |
| MC thin primary, chunk 2 | 5.97 | 98.74 % | 1.26 % | 4.99 | 96.4 % / 94.9 % | 74.0 % | 4.79 GB read, 0.30 % | 70.5 GB | 45.7 % | 23.0 % | 1.67 / 0.27 |
| MC Compton primary (Ns 1.6e5) | 1.69 | 99.94 % | 0.06 % | 1.46 | 88.0 % / 88.4 % | 98.5 % | 1.98 GB read, 0.07 % | 391 GB | 37.9 % | 22.5 % | 1.61 / 0.26 |
| MC Compton layer, generation 1 | 2.13 | 99.71 % | 0.29 % | 1.89 | 95.9 % / 96.8 % | 97.7 % | 0.70 GB read, 0.05 % | 53.8 GB | 20.6 % | 15.6 % | 1.06 / 0.16 |
| MC Compton layer, generation 2 | 3.50 | 99.35 % | 0.65 % | 3.62 | 73.0 % / 73.4 % | 91.2 % | 6.46 GB read, 0.67 % | 67.7 GB | 20.9 % | 11.6 % | 1.68 / 0.12 |
| MC Compton layer, generation 3 | 10.65 | 96.12 % | 3.88 % | 9.29 | 74.3 % / 68.8 % | 87.9 % | 9.72 GB read, 2.2 % | 137 GB | 39.1 % | 19.4 % | 1.76 / 0.22 |

"Active threads per warp instruction" is `smsp__thread_inst_executed_per_inst_executed.ratio`
(32 = every lane active); "branch targets uniform" is
`smsp__sass_average_branch_targets_threads_uniform.pct`; global load sectors per request is
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`
(8 for a coalesced 8-byte load by 32 lanes, 4 for 4-byte, up to 32 for a fully scattered warp,
fewer when few lanes are active); local memory traffic is 32 B
times the local load and store sectors; the FP64 pipe column is
`sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active`, issue active
`smsp__issue_active.avg.pct_of_peak_sustained_active`.

### 2.3 Warp stall reasons (average warps stalled per issued instruction, largest first)

| kernel | stalls |
|---|---|
| torus imaging | wait 2.84, selected 1.00, math pipe throttle 0.93, branch resolving 0.39, not selected 0.30, no instruction 0.29, short scoreboard 0.27, long scoreboard 0.13 |
| standard march | wait 2.86, math pipe throttle 1.11, selected 1.00, no instruction 0.73, branch resolving 0.41, not selected 0.35, short scoreboard 0.22, long scoreboard 0.07 |
| precise march | wait 2.90, **no instruction 2.70**, **long scoreboard 1.17**, selected 1.00, math pipe throttle 0.89, branch resolving 0.35, not selected 0.25, short scoreboard 0.21 |
| MC thin primary (chunk 1) | wait 2.83, **no instruction 1.84**, selected 1.00, math pipe throttle 0.56, branch resolving 0.37, short scoreboard 0.24, long scoreboard 0.23, not selected 0.17 |
| MC Compton primary | wait 2.90, **no instruction 1.37**, selected 1.00, long scoreboard 0.68, math pipe throttle 0.47, branch resolving 0.35, short scoreboard 0.24, not selected 0.14 |
| MC Compton layer, generation 1 | wait 2.87, **no instruction 1.93**, selected 1.00, branch resolving 0.34, long scoreboard 0.28, short scoreboard 0.24, math pipe throttle 0.09 |
| MC Compton layer, generation 2 | **no instruction 8.75**, wait 2.88, selected 1.00, long scoreboard 0.94, branch resolving 0.35, math pipe throttle 0.28, short scoreboard 0.22 |
| MC Compton layer, generation 3 | wait 2.96, **long scoreboard 2.11**, **no instruction 1.82**, selected 1.00, math pipe throttle 0.42, branch resolving 0.37, short scoreboard 0.25 |

### 2.4 Durations in production (Nsight Systems, two lanes, no profiler replay)

From `raw/tl_3797.nsys-rep` (`scripts/tl.jl`, `scripts/tl_analyze.py` → `data/tl_3797.json`),
warm calls:

| call | launches of the program | kernel durations |
|---|---|---|
| thin `mc_spectrum` | 4 × `MCPrimary` (7813 blocks of 32) | 0.963, 0.956, 1.090, 0.673 s (sum 3.683 s) |
| Compton `mc_spectrum` | 4 × `MCPrimary` | 7.422, 9.024, 10.425, 5.740 s (sum 32.61 s) |
| | 17 × `MCLayer` (12 generations + 5 overflow reruns) | generation 1: 1.38–1.95 s; generation 2: 2.03–2.38 s; generation 3: 1.09–1.78 s (sum 30.83 s, of which the five reruns 9.79 s) |
| | 19 × `Drain` | sum 0.006 s |
| torus imaging `render` | 1 × `ImagingProgram` | 0.452 s |
| standard march `render_crossings` | 2 × `CrossingProgram` (cost-ordered queue in two chunks) | 0.087 + 0.067 s |
| precise march `render_crossings` | 2 × `CrossingProgram` | 1.603 + 0.127 s |

## 3. Device-wide activity (Nsight Systems GPU metrics)

Sampled at 2 kHz over the kernel window of each warm call (`raw/tl_3797.sqlite`, table
`GPU_METRICS`); "warps in flight" is the percentage of the 64 warp slots per SM.

| kernel window | SMs active (mean) | samples with < 50 % / < 25 % of SMs active | warps in flight (mean) |
|---|--:|--:|--:|
| torus imaging, 0.452 s | 97.8 % | 2 % / 1 % | 11.7 % (7.5 warps/SM) |
| standard march, first launch, 0.087 s | 90.2 % | 9 % / 6 % | 10.8 % |
| precise march, first launch, 1.603 s | 20.5 % | 85 % / 84 % | 2.6 % (1.6 warps/SM) |
| thin MC photon phase, 3.71 s | 85.3 % | 16 % / 13 % | 8.0 % (5.1 warps/SM) |
| Compton MC photon phase, 62.5 s | 87.1 % | 13 % / 11 % | 8.9 % (5.7 warps/SM) |

SMs active over the precise launch, 0.1 s bins: 100, 100, 54, 15, 12, 8, 6, 6, 5, 5, 4, 4, 3,
3, 2, 1, 1 %. The launch is a full GPU for 0.2 s and a tail of a few blocks for 1.4 s.

Per-pixel step counts of the two marches of comparison B (`scripts/cross_steps.jl` on 16 CPU
threads, `data/cross_steps.json`; a step count is a property of the ray), with the lane
efficiency E of warps of 32 and of blocks of 256 pixels (the launch's block) in index order,
in the plan's queue order (the order the GPU visits) and sorted by steps:

| march | steps mean / median / p99 / p99.9 / max | share of steps in the slowest 0.1 % / 1 % | E warps of 32: index / queue / sorted | E blocks of 256: index / queue / sorted |
|---|---|---|---|---|
| standard | 927 / 912 / 1419 / 2341 / 2500 | 0.27 % / 1.9 % | 0.961 / 0.979 / 1.000 | 0.802 / 0.954 / 0.999 |
| precise (Verner 9) | 351 / 235 / 2395 / 8665 / 27255 | 3.9 % / 13.9 % | 0.851 / 0.891 / 0.994 | 0.492 / 0.642 / 0.950 |

The slowest precise pixel takes 27,255 steps, 78 times the mean; the precise launch lasts as
long as its slowest blocks (section 2.1: 1.600 s), while a pixel of the standard march is
capped at 2,500 steps.

## 4. Monte Carlo photons: steps, divergence, residency

Method: `scripts/mc_probe.jl` runs the torus deck through `Execution.foreach_cascade!` exactly
as `Products.run_layers` does, with probe programs wrapping `MCPrimary` and `MCLayer` (script
only; the package is unchanged): the march control forwards to `CurvedMarch` and stores the
loop iteration of every `march_control` call (the last is the photon's loop iterations,
"steps" below), the sink forwards to `MCSink` and stores the `MarchLedger` event count (child
streams drawn, "events") and outcome, and every photon stores its start and end
`%globaltimer` (ns), `%smid`, the octant of its spatial wave vector `k^i = g^{iν} p_ν` and its
start radius; layer photons are appended with their slot in the generation buffer and their
stream identity (layer, primary ancestor), which recovers the slot order of every launch,
i.e. the arrival order of the spawns on the GPU. One A100, the production pool (2 lanes),
job 3796 (`logs/mc_probe_3796.log`). Analysis on the host: `scripts/probe_analyze.jl`
(`logs/probe_analyze_s1.log`, `data/probe_s1/analysis.json`), `scripts/stats_extra.jl`,
`scripts/residency.jl`, `scripts/perlaunch.jl`. Records: `data/probe_s1/*.bin`.

Fidelity of the probe: thin, 3.81 s against 3.72 s for the plain `run_layers` run and the
accumulators identical; Compton, the same 7,302,016 children and 62.22 s against 63.06 s (job
3796), and in a second job (3803, `logs/mc_probe_3803.log`) the probe's spectrum (count, Σw,
Σw E of all 4 × 640 cells) and ledger totals equal the plain run's exactly, as two plain runs
do (61.91 s probe, 62.29 and 62.86 s plain); the raw integer limbs of the exact accumulators
differ between any two Compton runs, plain against plain included, with the same exact sums.
The step counts of escaped primaries are identical in the thin and Compton probes (482,418
photons, mean 549.55 steps), as the physics requires.

### 4.1 Steps and events per photon

| photons | count | steps mean / median / p90 / p99 / p99.9 / max | share of steps in the top 1 % | events mean / p90 / max | outcomes (escaped / captured / step limit) |
|---|--:|---|--:|---|---|
| thin primaries | 1,000,005 | 458 / 258 / 937 / 3266 / 7490 / 20000 | 10.9 % | 0 | 482,418 / 517,579 / 8 |
| Compton primaries | 1,000,005 | 1084 / 585 / 2246 / 9703 / 19045 / 20000 | 12.5 % | 0.43 / 2 / 10 | 482,418 / 516,803 / 784 |
| Compton layer 1 (children of primaries, tracked by generation 1) | 359,280 | 413 / 247 / 734 / 3101 / 10356 / 20000 | 14.6 % | 2.77 / 6 / 20 | 313,276 / 45,967 / 37 |
| Compton layer 2 | 991,373 | 226 / 176 / 386 / 948 / 3667 / 20000 | 9.0 % | 6.03 / 11 / 32 | 975,407 / 15,952 / 14 |
| Compton layer 3 | 5,951,363 | 181 / 157 / 315 / 536 / 834 / 11511 | 3.7 % | 0 | 5,943,511 / 7,852 / 0 |

By outcome (primaries): thin captured photons take 372 steps on average (42.1 % of the
steps), escaped 550 (57.9 %); Compton captured photons take 1555 (74.1 % of the steps; a
photon that may still spawn is not stopped by the plunge test), escaped 550 (24.5 %), step
limit 1.4 %. By start radius √(x² + y² + z²) (Kerr–Schild Cartesian, deciles from 1.75 to
862 M): thin mean steps 360, 514, 578, 615, 650, 644, 607, 380, 189, 153; Compton 1817, 1834, 1781, 1625, 1296, 1012, 780, 399, 189,
153. Half of the primaries start inside r = 2.75 M.

### 4.2 Lane efficiency of warps of 32 consecutive items

E = Σ steps / Σ_warps (32 × the warp's maximum steps), computed per launch (a chunk of
primaries, or one generation of one chunk) with the launch's items in the stated order; 1 − E
is the fraction of lane-steps idle, 1/E the ratio max/mean. "warp max/mean" gives the median
and 90th percentile over warps.

| order inside each launch | thin primaries E (warp max/mean median, p90) | Compton primaries E | Compton layers E (all) | layer 1 | layer 2 | layer 3 |
|---|--:|--:|--:|--:|--:|--:|
| as the GPU ran it (current) | **0.201** (3.37, 9.34) | **0.187** (3.83, 7.53) | **0.300** (2.61, 3.96) | 0.167 | 0.225 | 0.367 |
| random permutation | 0.163 (5.20, 9.20) | 0.143 (6.17, 10.5) | 0.291 | 0.147 | 0.219 | 0.366 |
| by zone (cell) | 0.201 (identical to current) | 0.187 | 0.366 | 0.246 | 0.345 | 0.397 |
| by zone, then octant | 0.260 (2.38, 5.83) | 0.246 | 0.366 | 0.248 | 0.345 | 0.397 |
| by meshblock, then octant, then zone | 0.279 (1.82, 5.50) | 0.262 (1.87, 5.19) | 0.500 (1.66, 2.19) | 0.305 | 0.446 | 0.564 |
| by octant, then zone | 0.279 | 0.262 | 0.504 | 0.309 | 0.454 | 0.567 |
| by radial shell (Δln r = 0.05), then octant | 0.277 | 0.259 | 0.478 | 0.269 | 0.412 | 0.555 |
| by start radius alone | 0.201 | 0.187 | | | | |
| by outcome (not known before the march), octant, shell | 0.309 | 0.287 | | | | |
| by steps, longest first (oracle) | 0.997 | 0.999 | 0.997 | 0.992 | 0.993 | 0.999 |

Weighting a step and an event equally (cost = steps + events) changes the thin, Compton
primary and all-layer entries by at most 0.006. Per launch, the current order gives thin
E = 0.209, 0.195, 0.165, 0.256 (6.69, 6.25, 5.27, 8.20 active lanes of 32) and Compton primaries 0.190, 0.184, 0.175, 0.211; ncu measures
6.37 and 5.97 active threads per warp instruction on thin chunks 1 and 2 (section 2.2), so the
step-count tail accounts for nearly all of the SIMT inefficiency of the thin kernel and branch
divergence inside a step for the rest (98.7 % of branch targets uniform).

Composition of a warp in the current order: thin and Compton primaries hold 8.96 distinct
zones, 1.01 meshblocks and 6.36 of the 8 direction octants per warp (3.9 photons per occupied
zone; the primaries of a zone are consecutive items, zones in site order); Compton layer warps
hold 27.3 meshblocks and 7.89 octants (arrival order of the spawns).

### 4.3 Measured warp timing

From the `%globaltimer` stamps (a warp = 32 consecutive items of a launch = one block, resident
from its first lane's start to its last lane's end):

| launch | items | span | warp span mean / max | median over warps of span / the warp's largest step count | resident warps peak / mean | time from the last moment at ≥ half the peak to the end |
|---|--:|--:|---|--:|---|---|
| thin chunk 1 | 250,002 | 0.952 s | 0.065 / 0.529 s | 30.3 µs | 864 / 537 | 0.213 s (22 %) |
| thin chunk 2 | 250,002 | 1.004 s | 0.069 / 0.565 s | 30.2 µs | 864 / 538 | 0.247 s (25 %) |
| thin chunk 3 | 250,002 | 1.110 s | 0.076 / 0.528 s | 29.2 µs | 864 / 534 | 0.196 s (18 %) |
| thin chunk 4 | 249,999 | 0.697 s | 0.050 / 0.566 s | 30.6 µs | 864 / 558 | 0.113 s (16 %) |
| Compton primaries, chunks 1–4 | 250,002 each | 7.34, 8.83, 10.43, 5.64 s | 0.60–0.89 / 2.7–3.2 s | 120–127 µs | 864 / 618–667 | 0.70–1.59 s (10–18 %) |
| Compton layer 1, chunks 1–4 | 65,689–110,830 | 1.47–1.68 s | 0.20–0.32 / 1.1–1.6 s | 115–147 µs | 864 / 414–448 | 0.64–0.72 s (39–44 %) |
| Compton layer 2, chunks 1–4 | 172,044–317,859 | 1.69–2.38 s | 0.15–0.17 / 1.25–1.43 s | 185–222 µs | 864 / 551–642 | 0.28–0.49 s (12–24 %) |
| Compton layer 3, chunks 1–4 | 1,029,813–1,933,401 | 1.09–1.79 s | 0.022 / 0.34–0.43 s | 44.3 µs | 864 / 660–730 | 0.04–0.13 s (3–12 %) |

Instructions per march step: the thin chunks 1 and 2 execute 73.8 G and 78.3 G warp
instructions (ncu) over 17.79 M and 18.89 M warp-steps (Σ over warps of the warp's largest
step count, `scripts/warpsteps.jl`), 4,150 warp instructions per warp-step in both; at the
measured 30 µs per warp-step that is about 32,000 SM cycles per warp-step, 7.7 cycles per
warp instruction. On the Compton deck of the ncu run (Ns 1.6e5, chunk 1; step counts of the
same launches from probe job 3809, `scripts/launch_e.jl`, `logs/probe_analyze_ns1p6e5.log`): the Compton primary
executes 104.1 G warp instructions over 6.64 M warp-steps, 15,665 per warp-step (3.8 times the
thin kernel's); generation 1 14.2 G over 1.32 M (10,735), generation 2 14.4 G over 1.37 M
(10,522), generation 3 19.4 G over 3.76 M (5,154). The ratio of the lanes still marching to the
active threads per warp instruction is 3.6 (primary), 2.2 and 2.0 (generations 1 and 2), 1.1
(generation 3) and 1.05 (thin) (for the layers the lanes alive are those of the probe's
arrival order, which differs from the ncu run's). The measurements are consistent with the
marching lanes of a warp running a step in about two to four separate groups in the launches
whose photons spawn, which the larger instruction count per warp-step agrees with
(inference, not a direct measurement); the lanes of these launches also end their march at
different times (next paragraph).

In a launch whose photons do not spawn (thin primaries, layer 3) a lane's end time equals its
warp's end to within 0.01 % of the warp span: the lanes reconverge after the march loop, so a
finished lane waits for the slowest photon of its warp. In the spawning launches (Compton
primaries, layers 1 and 2) lanes record their end 39–49 % of the warp span before the warp
ends on average; the warp still holds its registers until its slowest lane ends.

The peak of 864 resident warps is the register limit (8 per SM × 108). Between the start of a
launch and its tail the SMs do not stay full: at 0.1, 0.3 and 0.5 s into thin chunk 1 the
resident warps are 750, 684 and 668 while blocks remain to be scheduled, with 52, 24 and 24 of
the 108 SMs at 8 warps and the rest at 2–7 (`scripts/smcheck.jl`, per-SM counts from
`%smid`). The cause is not established; the 1-warp blocks at 255 registers
fill two warp slots per scheduler partition, and a candidate is the partition a new block is
assigned to.

### 4.4 Residency over the whole photon phase

![resident warps](profile/data/fig_residency.png)

`data/fig_residency.png` (`scripts/plot_residency.py`, `data/probe_s1/residency.json`, 10 ms
bins): thin, mean 539 of 864 warp slots (0.62) over 3.76 s, below half of the slots for
0.78 s; Compton, mean 606 (0.70) over 62.2 s, below half for 11.4 s, below 90 % for 54.4 s.
Warp-seconds of the Compton phase: primaries 20,668, layers 11,684, overflow reruns 5,331
(14.1 %). The launches of the two lanes of the device follow each other (section 5.3): each
launch's tail runs alone on the GPU.

### 4.5 Overflow reruns of the Compton cascade

The probe records hold 8,304,106 layer photons for 7,302,016 distinct children: 1,002,090 are
second runs of a generation that overflowed its child buffer (stream identities seen twice).
The reruns are generation 1 and 2 of the first chunk of each lane (chunks 1 and 2) and
generation 2 of chunk 4: a chunk starts with `max(4096, chunk/2 = 125,001, slots grown
earlier)` child slots, and a chunk's generation 1 spawns 172,044–317,859 children and
generation 2 1.03–1.93 M. In the nsys trace of the warm Compton call the five reruns are
`MCLayer` launches of 1.52, 1.68, 2.15, 2.06 and 2.38 s (9.79 s of the 63.2 s call; launches of
the same grid as the generation they repeat). The primary generation does not overflow
(4 `MCPrimary` launches per call).

## 5. Host timeline of `mc_spectrum` on one A100

Method: `scripts/tl.jl` under `nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all
--gpu-metrics-frequency=2000` (job 3797, `logs/tl_3797.log`), an NVTX range per call;
`scripts/tl_analyze.py` → `data/tl_3797.json`, `logs/tl_analyze_3797.log`.
"Host before" is the time from the start of the call to its first kernel; "inner idle" the
GPU idle time between the first and the last kernel of the call.

### 5.1 Calls

| call | wall | GPU busy (union of kernels) | host before first kernel | host after last kernel | inner idle | launches | copies H→D / D→H | stream synchronisations |
|---|--:|--:|--:|--:|--:|--:|---|--:|
| thin, cold (first call of the process) | 50.14 s | 4.12 s (8 %) | 27.30 s | 2.08 s | 16.64 s | 15 | 23, 1267 MB, 0.50 s / 28, 167 MB | 193 |
| thin, warm | 4.27 s | 3.68 s (86 %) | 0.53 s | 0.04 s | 0.02 s | 4 | 7, 84 MB, 0.018 s / 6, 1.5 MB | 187 |
| Compton, cold (kernels of the thin call cached) | 101.5 s | 61.4 s (60 %) | 2.46 s | 0.29 s | 37.4 s | 41 | 64, 92 MB / 58, 9.5 MB | 602 |
| Compton, warm | 63.2 s | 62.4 s (99 %) | 0.29 s | 0.44 s | 0.08 s | 40 | 60, 92 MB / 57, 9.3 MB | 627 |
| torus imaging, cold / warm | 14.95 / 0.49 s | 0.45 s | 14.32 / 0.02 s | 0.17 / 0.02 s | 0 | 1 | 2.4 MB / 4.7 MB | 3–9 |
| standard march, cold / warm | 15.59 / 0.19 s | 0.14–0.15 s | 14.96 / 0.002 s | 0.49 / 0.03–0.06 s | 0–0.003 s | 2 | 1 MB / 32 copies, 32 MB | 68–82 |
| precise march, cold / warm | 37.34 / 1.82 s | 1.62–1.73 s | 35.58 / 0.002–0.06 s | 0.14 / 0.02–0.09 s | 0–0.011 s | 2 | 1 MB / 32 copies, 32 MB | 67–68 |

### 5.2 Phases of the cold thin call (event times from the call's start)

| interval | event |
|---|---|
| 0 – 10.5 s | host: site scan (2.96 s warm, `phase sites`) and first-call host compilation; no CUDA activity |
| 10.5 s | upload of the sites, 659 MB in 0.30 s |
| 10.8 – 26.3 s | compilation of `ZoneSums` (module load at 26.32 s) |
| 27.30 s | `ZoneSums`, 0.425 s |
| 27.7 – 29.95 s | compilation of `ZoneCounts` |
| 30.08 – 30.38 s | 10 × `ZoneCounts`, 0.011–0.015 s each, with 22 small read-backs |
| 30.4 – 33.5 s | host (zone offsets 0.30 s, emission radius, program set-up) |
| 33.52 s | upload of the sites for the photon program (433 MB, 0.17 s) and two 82 MB arrays |
| 33.8 – 44.37 s | compilation of the thin `MCPrimary` |
| 44.50 – 48.06 s | 4 × `MCPrimary`, one after the other (section 5.3) |
| 48.06 – 50.14 s | host: gather and result (0.04 s in the warm call; the rest is first-call compilation) |

The warm thin call: 0.51 s of host work (zone offsets and set-up; the zones and tables come
from the memo), one 82 MB upload (0.018 s), the four launches (3.68 s), 0.04 s of gather.
Phases of a warm grid run timed one by one: site scan 2.96 s, `grid_tables` 0.80 s (GPU busy
0.57 s: `ZoneSums` 0.43 s, 10 × `ZoneCounts` 0.135 s; 660 MB up, 165 MB down), zone offsets
0.30 s (host).

Compilation measured as the interval to the module load (`cuModuleLoadDataEx`): `ZoneSums`
15.5 s, `ZoneCounts` 2.3 s, thin `MCPrimary` 10.6 s, Compton `MCPrimary` 16.5 s, `MCLayer`
up to 21.4 s (both lanes waited from the end of their first primary launch, 33.7 s and
41.3 s, to the module load at 55.18 s), torus `ImagingProgram` 14.3 s, `CrossingProgram`
production 15.0 s, precise 35.6 s (host before the first kernel of the cold call).

### 5.3 The two lanes of one device do not overlap their kernels

Kernel records of the warm calls (`stream`, launch API time, kernel start and end):

- thin warm: lane A (stream 15) launches at 0.527 s, its kernel ends at 1.490 s; lane B
  (stream 13) launches at 1.491 s; A launches again at 2.470 s, B at 3.561 s. `MCPrimary`
  summed 3.683 s = its union 3.683 s.
- Compton warm: `MCPrimary` summed = union = 32.61 s; `MCLayer` summed 30.83 s, union 29.80 s;
  all kernels summed 63.45 s, union 62.41 s. The only overlaps are launches made by both
  lanes within a millisecond of each other (16.760/16.761 s: the second kernel starts 0.75 s
  later, filling the slots the first frees).
- In the 0.96 s of lane A's first thin kernel, lane B's thread makes no CUDA API call after
  0.5270 s and resumes at 1.4907 s, 0.3 ms after the kernel ends; two CUDA.jl
  synchronisation workers wait on lane A's stream 15 during that kernel (0.5289 s, 0.96 s and
  0.5414 s, 0.95 s), none on stream 13.
- Two-GPU job (section 6): on the one-device pool the warm calls' `MCPrimary` kernels sum to
  their union (7.871 s for 8 launches); on each device of the two-device pool the summed kernel
  time exceeds the union by 0–0.34 s per call, only where both lanes of the device launch
  within a millisecond. In the cold calls the first launches of the two lanes overlap.

With CUDA.jl's early collection off (`JULIA_CUDA_GC_EARLY=false`, which removes the
`maybe_collect` of every `synchronize`; job 3802, `raw/tl_3802.nsys-rep`,
`logs/tl_analyze_3802.log`) the kernels still do not overlap: thin warm call 4.135 s, the four
`MCPrimary` launches sum to their union (3.726 s) and lane A (stream 13) runs three of them in
a row (launch at 0.387, 1.331, 2.468 s, each at the previous kernel's end) before lane B
(stream 15) runs the fourth at 3.152 s; Compton warm call 63.36 s, `MCPrimary` summed = union
= 32.66 s, `MCLayer` summed = union = 30.18 s. The early collection is therefore not the cause.
Mechanism, from CUDA.jl's source (CUDACore `src/memory.jl`, `Base.convert(::Type{CuPtr{T}},
managed::Managed)`): converting a device array for a launch on a stream other than the one
that last used it calls `maybe_synchronize`, which synchronises the previous stream while the
array is marked dirty, and every conversion marks it dirty. The two lanes of a device share the
device twin of the program (`ProgramTwins`: one twin per device) and with it every array it
holds (sites, zone tables, snapshot, the accumulators' device twins), so a lane's launch waits
for the other lane's running kernel and takes the arrays over; this matches the second waiter
on stream 15 above and the alternation of the lanes.

A check of the mechanism (job 3808, `scripts/lanes_micro.jl`, `logs/one_3808.log`): two Julia
tasks, each on its own stream, launching a KernelAbstractions spin kernel on half of the GPU
four times run their eight kernels in 1.12 times (KernelAbstractions synchronisation) and 1.01
times (`CUDA.synchronize(blocking = true)`) the time of one task's four, i.e. concurrently, when
each task reads its own input array; when both read one shared array the launches interleave
(0.460 s for eight kernels against 0.438 s for one task's four). Through the package's pool, a
spin program that holds no device arrays (outputs in per-lane buffers) runs 8 chunks in
1.541 s on two lanes against 3.028 s on one lane: the pool's lanes overlap when their programs
share no arrays.

## 6. Two GPUs

Job 3795 (`--gres=gpu:2`, `logs/twogpu_3795.log`, `scripts/twogpu.jl`, under nsys
`raw/twogpu_3795.nsys-rep`): Slurm grants both GPUs (`CUDA_VISIBLE_DEVICES=0,1`, 84.6 GB
free on each); `resolve(:gpu)` selects both
(`PoolExecutor([CUDADevice(0), CUDADevice(1)]; lanes = 2)`), chunk 125,001. Thin torus at
stride 4 (160,997 zones, 1,000,005 primaries), three calls each:

| executor | calls (first includes compilation) | bins |
|---|---|---|
| `gpu:0` | 45.18, 3.99, 4.09 s | reference |
| `gpu:0,1` | 8.63, 2.69, 2.68 s | bitwise `gpu:0` |
| `:gpu` (both) | 2.58, 2.67, 2.53 s | bitwise `gpu:0` |

Each device runs 4 of the 8 chunks per call (kernel records on devices 0 and 1); per device
the kernels are busy 2.27–2.61 s of the 2.35–2.63 s call. A job with two GPUs still has 4
host CPUs and occupies the whole node's CPU allocation.

## 7. The working hypothesis against the measurements

| claim of the hypothesis | measurement |
|---|---|
| a warp holds photons from random cells | **not for primaries**: the primaries of a zone are consecutive items and zones are in site order, so a warp of primaries holds 1.01 meshblocks (8.96 zones); a random order would lower E from 0.201 to 0.163 (thin). **Yes for layer photons**: 27.3 meshblocks per warp in arrival order |
| ... with random directions | yes: 6.36 of 8 octants per warp of primaries, 7.89 per warp of layer photons |
| so memory diverges | the accesses of a warp are scattered, 5.0–5.4 sectors per global load request for 6.0–6.4 active lanes (about one sector per active lane; torus imaging 5.77 for 31.3 lanes), but the thin primary kernel is not limited by them: L1 hit 96 %, L2 hit 73 %, DRAM 0.3 % of peak, long-scoreboard stalls 0.21–0.23 per issue; the layer photons are where memory diverges: generations 2 and 3 load 3.6 and 9.3 sectors per request with L1 hits of 73–74 % and long-scoreboard stalls of 0.94 and 2.11 per issue, against 1.5–1.9 sectors and 88–96 % L1 hits for the Compton primary and generation 1 |
| so control diverges | branches are 98.7 % uniform inside a step; the divergence is in the loop trip count: 6.0–6.4 active threads per warp instruction (thin), matching the step-count lane efficiency 0.195–0.209 of the same launches; in the kernels whose photons spawn, the instruction-level efficiency falls far below the step tail: on the same launches (Ns 1.6e5 deck, probe job 3809) the Compton primary has 6.15 lanes still marching on average (E = 0.192) but 1.69 active threads per warp instruction, generation 1 4.76 against 2.13, generation 2 6.92 against 3.50, while the non-spawning generation 3 has 11.78 against 10.65 and the thin primary 6.69 against 6.37 |
| warps wait for their slowest photon | yes: lane efficiency 0.201 (thin), 0.187 (Compton primaries), 0.300 (layers, 0.167 / 0.225 / 0.367 by generation); finished lanes of a non-spawning launch end with their warp; a block of one warp holds one eighth of the SM's registers until its slowest photon ends |
| coherence sorting fixes it | partly: sorting each launch by meshblock (or octant) then octant (zone) raises E to 0.279 (thin), 0.262 (Compton primaries), 0.500 (layers: 0.305 / 0.446 / 0.564); E by step count (the oracle a regrouping by remaining work approaches) is 0.992–0.999 |
| the megakernel carries the Compton sampler's registers in every thread | every large kernel is at the 255-register cap, the thin primary (no scattering code) and the imaging and crossing programs included; the scattering code shows in the spills and the frame instead: `track!` spills 1704 / 2568 B (stores / loads) in the Compton kernels against 144 / 168 B in the thin one, and the stack frame is 6352 B (Compton primary), 5448 B (layer) against 3176 B (thin); in ncu, 3.9–4.7 % of the Compton kernels' warp instructions are spill loads and stores (thin 0.42 %), 391 GB of local traffic in the 2.0 s Compton primary launch |
| the precise march suffers register spills | it spills most (stack 5632 B, 4.2 % of instructions are spill loads and stores, L1 hit 60 %, 69 GB of DRAM writes, long-scoreboard stalls 1.17 and instruction-fetch stalls 2.70 per issue), and its launch is dominated by its tail: 84 % of the 1.6 s with fewer than a quarter of the SMs active, the slowest pixel 27,255 steps |

Measured in addition: in the launches whose photons spawn (Compton primary, generations 1
and 2) the active threads per warp instruction are 2.0–3.6 times below the lanes still
marching (1.69–3.50 against 4.8–6.9) and the warp issues 10,500–15,700 instructions per
warp-step against 4,150 without scattering (section 4.3); the two lanes of a device do not
overlap their kernels (section 5.3), so every launch's tail idles the GPU (thin: 0.11–0.25 s per launch); 14 % of the Compton
photon phase's warp-seconds and 9.8 s of the 63.2 s call are overflow reruns (section 4.5); the
SMs hold 650–750 of the 864 possible warps in the bulk of a thin launch (section 4.3); cold
calls spend 2–36 s per program in compilation (section 5.2).

Estimates **(estimate)**, derived from the measurements above and not measured: if every
warp-step of the thin primary launches had 32 active lanes at today's cost per warp-step,
their lane-steps would shrink by up to 1/E ≈ 5.0 (Compton primaries 5.3, layers 3.3), before
the cost of regrouping and any change of registers or occupancy; by active threads per warp
instruction the room is 32/6.4 ≈ 5 (thin) and 32/1.69 ≈ 19, 32/2.13 ≈ 15, 32/3.50 ≈ 9 for
the Compton primary and generations 1 and 2, of which the step tail explains 5.2, 6.7 and 4.6
and the ratio of section 4.3 (3.6, 2.2, 2.0) the rest; sorting by meshblock and
octant alone bounds the same factor at 1.39 (thin), 1.40 (Compton primaries) and 1.67
(layers). Removing the overflow reruns would remove at most their 9.8 s of the 63.2 s Compton
call; letting the two lanes overlap could at most fill the launch tails, 0.78 s of the 3.76 s
thin phase spent below half of the warp slots.

## 8. Notes on method

- ncu replays each kernel 53–56 times (`--set full`); the thin profiled call took 550 s for
  two kernels. ncu durations are single-kernel, single-lane runs; nsys durations are the
  production schedule.
- The lane efficiency of section 4.2 counts loop iterations of `track!`; it does not weight
  the extra work of a step that samples a scattering event. ncu's active threads per warp
  instruction (section 2.2) is the instruction-weighted measure of the same kernels.
- Reorderings in section 4.2 are applied inside each launch only, to the items the launch
  actually held; the oracle order needs the step counts, which are known only after the march.
- Spatial keys: zone = the zone index of a primary (site order); for a layer photon the cell of
  its start position (`Data.locate` meshblock, nearest cell centre), computed on the host.
- The probe's `%globaltimer` stamps are taken at the start of the program and after
  `transport!`; residency excludes the block launch and exit (microseconds).
- The compile-time table of section 2.1 runs CUDA 13.3's `ptxas` at `-O3` on the PTX that
  `CUDA.@device_code` dumped; the kernels CUDA.jl loads have the same register counts and,
  within 8 B, the same stacks (ncu `launch__stack_size`).
- The Slurm node grants one job at a time (4 CPUs in total), and other agents' jobs share the
  queue; every measurement here ran alone on its GPU.
