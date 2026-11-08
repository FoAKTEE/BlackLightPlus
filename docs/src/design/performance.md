# Performance

How BlackLightPlus spends its time, the knobs that change it, and every speed capability of the
reference package (`src/`) with its form in BlackLightPlus. Protocols and types are specified in
[Architecture](@ref) §4.2 and §5; the measurements are in [Verification record](@ref) §4
(CPU against the A100) and §4.1 (against the reference package on the same decks). The
results of every product are bitwise independent of everything in this document: the
executor, the chunk, the lane count, the device count, the queue order and the child-buffer
capacity change the wall time and nothing else.

## 1. Execution model

- **Programs and the one kernel.** A product is a per-item program run by an executor:
  `ThreadedExecutor` calls it from Julia threads on the host arrays; `PoolExecutor` runs its
  device twin through the single `_foreach!` kernel on the lanes of a `DevicePool`. The
  program supplies its launch traits: `workgroupsize` (64 for the polarized imaging
  program, 32 for the Monte Carlo primary and layer programs, else the backend's choice), `stack_bytes` (32 KiB for the exact-Kerr program)
  and `chunk_hint` (an upper bound on its chunk under `:auto`).
- **Threads.** The items are cut into chunks in queue order; one worker task per thread
  pulls the next chunk from one queue (`run_workers!`), so the long marches at the head of
  a cost-ordered queue are spread over every thread and no thread idles on a contiguous
  run. A program's exception is rethrown as a `CapturedException` once the queue drains.
- **Pool.** `lanes` persistent worker tasks per device (two by default), each with its own
  stream and pinned staging buffers, pull chunks from one shared queue. A chunk is one
  launch: a range of items goes to the kernel as it is (no index upload); any other item
  list is staged as `Int32` when it fits, `Int` otherwise. The outputs come back through
  the lane's pinned staging and are scattered into the host arrays while the other lanes
  compute. The first failure is rethrown after the queue drains.
- **Residency.** Every host array a program holds is uploaded once per device and found
  again by identity (weakly held); the registry pool of a device list and lane count is
  shared by every executor built on it, whatever its chunk, so a change of chunk re-uploads
  nothing. `backend = :gpu` resolves to that pool.
- **Outputs in the product's layout.** Programs write their final arrays: a Stokes I image
  is the output vector reshaped, the fifteen exact-Kerr maps are the fifteen output vectors
  reshaped (the analytic program visits the pixels in index order, so every chunk writes a
  contiguous run of each map), the IQUV planes are split once per call. The AMR worklist
  paints block values in the visits.
- **Accumulators.** Monte Carlo bins, the weight ledger and the scalar diagnostics are exact
  integer-limb sums (`ExactBins`, `ExactScalars`) written through atomics; each device that
  runs a chunk writes a zeroed twin that `gather!` folds into the host sums.
- **Tables on the run's executor.** The Monte Carlo zone tables (the weight table, every
  zone's count and envelope) and the hot-cross table are built where the run executes: on
  threads by the host builders of `Data`, on a pool by device programs (`ZoneSums`,
  `ZoneCounts`, `HotcrossSigma`) with the same arithmetic. Their logarithms and
  exponentials are `tang_log` and `tang_exp`, Base's table-driven algorithms written with
  explicit fused multiply-adds, which a device computes to the host's bits (the vendor
  library's `exp` and `log` differ in the last bit), the node coordinates and the Jüttner
  normalisations come from the host, and the per-node sums are exact limb sums, so the
  tables are bitwise the same on every executor. The zones and tables of the last grid
  run are kept for the next run on the same blocks, spacetime, parameters, zone options
  and generation settings.
- **The cascade.** Scattering runs as a cascade (`foreach_cascade!`): a chunk of primaries
  and every scattering generation of their children run on the worker that took the chunk.
  On a lane the children live in two device buffers swapped per generation, and the only
  host read per generation is their count; a generation records into lane scratch sums
  that are folded into the device's totals once its children fit, and a generation whose
  children overflow the buffer runs again in full after the buffer grows to the counted
  children and an eighth more; a lane starts its later chunks of the run with the slots
  it grew to. On threads each
  task keeps two growing buffers and reruns only the photon whose children overflowed.
  Child streams are keyed by genealogy and the sums are exact, so the spectrum does not
  depend on the chunking, the lane or the arrival order.

## 2. Defaults and tuning

| knob | default | where | effect |
|---|---|---|---|
| executor | `:auto` | `backend` keyword > `backend!` > `BLP_BACKEND` > `:auto` | GPUs when CUDA is loaded and a device is idle, else threads |
| chunk | `:auto` | `ThreadedExecutor(; chunk)`, `PoolExecutor(devices; chunk)`, `DevicePool(devices; chunk)` | items per launch or per thread task; `describe(exec).chunk` reports the chunk of the last run |
| lanes | 2 | `PoolExecutor(devices; lanes)` | worker tasks and streams per device; pools differ per lane count |
| queue | cost order on Kerr, identity elsewhere | `prepare(cfg; queue = perm)` | the order pixels enter the queue of `render`, `render_crossings`, `shadow`, `sensitivity` and `variants`; for a movie, `march_sort_order(vec(shadow(plan).steps))` of the previous frame |
| cascade chunk | `cascade_chunk(exec, n)` | `foreach_cascade!(…; chunk)` | primaries per cascade chunk |
| child slots | at least 4096, half the chunk, and the slots the lane grew to in the run | `mc_spectrum(…; capacity)` | first size of a generation's child buffer; overflow grows it and reruns |
| hot-cross table | built once per process, on the first run's executor | `mc_spectrum(…; table)` | `hotcross_table(plan)` takes the process's table at the plan's first use; a given table is used as is |
| zones and zone tables | kept from the last grid run | the run's blocks, spacetime, parameters, zone options, generation settings | a run that matches the last one reuses its `emission_sites` and its tables; any change rebuilds them |
| workgroup | backend's choice; 64 for IQUV imaging, 32 for the Monte Carlo primaries and layers | `workgroupsize(prog)` | threads per block of the launch |
| stack | backend's default; 32 KiB for the exact-Kerr program | `stack_bytes(prog)` | device stack per thread |

**The `:auto` chunk rule** (`chunk_for`), applied per run from the item count `n`, the
bytes of output per item and the program's `chunk_hint`:

- a pool on one device: `2^20`, one launch per run of up to `2^20` items;
- a pool on `d > 1` devices with `l` lanes each: `clamp(cld(n, 2 l d), 2^16, 2^20)`, two
  chunks per lane where the run is large enough;
- on a pool, at most `max(2^16, 2^24 ÷ bytes)`, so one launch's outputs (16 MiB at most)
  are copied back and scattered while the other lanes compute;
- threads: `clamp(cld(n, 16 t), 16, 4096)` on `t` Julia threads;
- then at most the program's `chunk_hint`; an explicit chunk is used as given.

**The cascade chunk** (`cascade_chunk`): on a pool `min(cld(n, 4 d), 2^20)`, a quarter of
a device's share of the primaries, at least the executor's chunk when that is explicit; on
threads `cld(n, 64 t)`, at least the executor's chunk when that is explicit. An AMR worklist
chunk holds `max(1, c ÷ block²)` blocks of `c` items.

**What to tune, in order.** The chunk only when a leg is known to prefer another size
(`bench/bench.jl` with `BLP_TUNE=1` sweeps lanes and chunks and records the best per
leg); the lane count for kernels at the register cap; the queue for a movie; `capacity`
only to avoid the overflow rerun on a deck whose scattering multiplicity is known.

**Kernel resources.** `bench/kernel_audit.jl` compiles the device twin of every program
kind for sm_80 without a GPU and reports registers, stack and spills against
`kernel_audit_baseline.json`; `--check` fails on growth, `--write` records a deliberate
change, `--device` compares with a GPU's own compilation (`bench/README.md`).

## 3. Capabilities of the reference package and their form in BlackLightPlus

Status: *same* (the mechanism is the same), *form* (the same effect through a different
mechanism), *beyond* (BlackLightPlus has more than the reference), *not carried* (with the reason in
§4). The reference locations are in `src/`, BlackLightPlus's in `src/` unless given.

### 3.1 Devices, pool and residency

| capability | reference | BlackLightPlus | status |
|---|---|---|---|
| Persistent pool: worker tasks per device, one stream each, shared queue | `gpu_pool.jl` | `DevicePool`, `pool.jl` | same |
| Pinned staging buffers, freed explicitly | `gpu_pool.jl` | `lane_stage`, `BlackLightPlusCUDAExt.jl` | same |
| Dynamic chunk queue, first items first | `gpu_pool.jl` | `foreach_chunk`, `run_workers!` | same, and on threads (beyond) |
| First-error propagation | `gpu_pool.jl` | `Job`, `pool.jl` | same |
| Default pool reused across calls | `default_pool` | registry keyed by (devices, lanes); a changed chunk shares it | beyond |
| Upload once: snapshots, grids, spacetimes, tables | `upload_per_device` | residency by array identity, `residency.jl` | form |
| Concurrent upload to every device | `gpu_pool.jl` | lazy, devices in parallel | form |
| Two lanes per device | `lanes = 2` | `lanes = 2` | same |
| Logical host devices; thread-count warning | `HostDevice` | `HostDevice`, `devices.jl` | same |
| Read-only cache for resident arrays | `@Const` | `readonly`, `ReadOnlyArray` | same |
| Monte Carlo chunk of `2^20` primaries | `mc_gpu.jl` | `:auto` chunk rule and `cascade_chunk` | form |
| Global-index chunks, no index upload | `mc_gpu.jl` | ranges passed unstaged; other lists staged as `Int32` | beyond |
| Chunk sizing | fixed constants | `:auto` per run, `chunk_hint` per program | beyond |

### 3.2 Scheduling and precision

| capability | reference | BlackLightPlus | status |
|---|---|---|---|
| Cost-ordered 16×16 tile queue from the critical curve | `gpu_image.jl` | `cost_order`, once per plan | same |
| Mixed-precision rescue pixels first | `gpu_image.jl` | `mixed_partition`, `rescue_pixels` | same |
| `march_sort_order` and a caller's order | `order` keyword | `prepare(cfg; queue = perm)`, validated by `queue_items` | same |
| Mixed precision (`Float32` geodesics, `Float64` transfer) | `gpu_image.jl` | `MixedLanes` | same |
| Camera ray in the lane's precision from the item index | `_camray_kernel!` | non-inlined `camera_state` inside the program | form |
| Separate camera-ray kernel for the `Float32` lane | `gpu_camera.jl` | fused: the non-inlined `camera_state` in the program | not carried (§4) |
| Domain-radius early exit on outbound rays | `gpu_amr.jl` | `R_em = domain_radius(matter)` for every matter | same |
| One termination rule; step `(r − r_h)/C` | `reverse_march.jl` | `controls.jl`, `termination.jl` | same |
| Slow-light single-read shortcut; window advance reuses the upload | `gpu_image.jl`, `gpu_amr.jl` | `series.jl` | same |
| Mass-scale ladder; in-kernel derivative | `gpu_image.jl`, `gpu_image_ad.jl` | `variants`, `sensitivity` with `Dual` | same |
| Polarized block of 64; 32 KiB stack for the exact-Kerr program | `gpu_pol.jl`, `gpu_analytic.jl` | `workgroupsize`, `stack_bytes` | same |

### 3.3 Matter and imaging

| capability | reference | BlackLightPlus | status |
|---|---|---|---|
| Uniform resample grid | `gpu_image.jl` | `uniform_grid.jl` | same |
| Native block sampling with a dense leaf table | `gpu_amr.jl` | `fluid_blocks.jl` | same |
| Hashed leaf lookup on the host, scans on the device | `athenak_reader.jl` | same split | same |
| Streaming adaptive refinement, no level barrier | `gpu_amr.jl`, `amr.jl` | `stream_blocks!`, `worklist.jl` | same |
| Threaded host painting of block values | `amr.jl` | `worklist.jl` | same |
| Per-chunk copy into the final per-item arrays | `gpu_pool.jl` | `scatter!` per chunk on the lane | same |
| Per-chunk copy into the product layout (exact-Kerr maps) | `gpu_analytic.jl` | fifteen output vectors reshaped, identity queue | same |
| Stokes I image without a copy | `gpu_image.jl` | the output vector reshaped | same |
| Tracer with device-side trajectory logs | `gpu_tracer.jl` | 20-step windows relaunched (`TRACE_WINDOW`) | form (§4) |

### 3.4 Monte Carlo

| capability | reference | BlackLightPlus | status |
|---|---|---|---|
| Every scattering order inside each chunk; children on the device | `mc_gpu.jl` | `foreach_cascade!`, `cascade.jl` | same |
| Overflow rerun at the exact size, per chunk and layer | `mc_gpu.jl` | per chunk and generation; on threads per photon | same |
| Tables built once, uploaded concurrently | `mc_gpu.jl` | `F(K)` emissivity table at compilation; hot-cross table once per process and zone tables per grid run, each on the run's executor (device programs on a pool, bitwise the host builders), the zones and tables of the last run kept; uploads lazy per device | beyond |
| Ancestry-keyed streams; exact accumulation | `rng.jl`, `exact_sum.jl` | `philox.jl`, `exactsum.jl`, `accumulators.jl` | same |
| 64 ledger stripes | `mc_gpu.jl` | `LEDGER_STRIPES` | same |
| Compton sampler behind a non-inlined call | `mc_gpu.jl` | inlined sampler | not carried (§4) |
| CPU: per-thread private accumulators, depth-first scattering | `mc_cpu.jl` | task-local child buffers, shared exact atomics | form |

### 3.5 Diagnostics and gates

| capability | reference | BlackLightPlus | status |
|---|---|---|---|
| Offline register, stack and spill audit with a baseline gate | `bench/kernel_audit.jl` | `bench/kernel_audit.jl`, `--check` | same |
| Multi-GPU scaling | 1.7–1.8× on two devices | [Verification record](@ref) §4.1 | measured |
| Stiffness telemetry; sampler probes | `gpu_pol.jl`, `gpu_amr.jl` | not carried | §4 |

## 4. Not carried, and why

- **The two-phase stored-trajectory render as the CPU default.** BlackLightPlus's fused march is
  1.2–4.3× faster on the CPU ([Verification record](@ref) §4.1); the stored path remains as
  `render(plan, trajectories)` on the host.
- **The GPU two-phase pipeline.** Bound by the PCIe copy of the trajectories; the fused
  march on the device is faster on every leg.
- **Twenty-one specialised kernels.** One kernel specialised on the program type compiles
  to the same device code per program; the audit lists forty-one such specialisations.
- **The separate camera-ray kernel.** Replaced by the non-inlined `camera_state` inside the
  program. A two-launch form (a ray program writing `Float64` rays into a lane buffer, then
  the march reading them by chunk position) is bitwise equal to the fused march but not
  faster on any program kind on the A100, and leaves the register counts of the `Float32`
  lane (226) and of the IQUV programs (255) unchanged; rays stored already narrowed to
  `Float32` change the Stokes I image of that lane. The register footprint of those
  programs is a question for the kernel audit, not for the camera ray.
- **The Compton sampler behind a non-inlined call.** On the A100 the scattering kernel is
  slower with the boundary (743 ms against 643 ms in the scattering-sphere leg) and stays
  at the 255-register cap; on the CPU the boundary changes the last bits of the torus
  scattering spectrum against the reference, which the oracle compares exactly.
- **Diagnostic kernels** (stiffness telemetry, sampler probes, `render_with_powerlaw`):
  measurements, not products; `bench/` and the kernel audit take their place.
- **The level-synchronous adaptive closure.** The streaming worklist queues a child as soon
  as its parent's visit returns; the level barrier only added launches.
- **The `upload_bounds` switch.** Residency uploads every array a program holds once;
  there is nothing to switch.
- **The lazy leaf-table build.** The table is built with the blocks in `Data`, once.
- **The single-device default for the exact-Kerr product.** The product runs on the
  resolved executor like every other; its maps are written per chunk, so several devices
  share the work without a host epilogue.
- **Device-side trajectory logs for `trace`.** The tracer resumes exactly from 20-step
  windows; a ragged per-item column output in `Execution` would replace the relaunches
  (at most 125 for a 2,500-step march) and is the one remaining item of the record.

## 5. Standing against the reference package

Geometric mean of BlackLightPlus's rate over the reference's on the same hardware, before the
capabilities of §3 were in their present form (BlackLightPlus at 65481d9, `anta_2026-10-17_reference.md`)
and after (7be2569, `anta_2026-10-18_reference.md`); the per-leg rows and the profiled
phases are in [Verification record](@ref) §4.1. Every pair of outputs is bitwise identical on
the A100 and agrees to round-off or bitwise on the CPU, before and after.

| hardware | legs | default: before → after | best: before → after |
|---|--:|--:|--:|
| one A100 | 19 | 0.83 → 1.13 | 0.95 → 1.06 |
| CPU, 16 threads | 6 | 1.16 → 1.44 | 1.32 → 1.47 |
| CPU, 64 threads | 6 | 1.39 → 2.13 | 1.79 → 2.30 |
| CPU, 128 threads | 6 | 1.25 → 2.08 | 1.75 → 2.16 |

No two-device pair of either package has been recorded.

The legs that moved most on one A100: the scattering sphere 0.40 → 1.01 (the cascade), the
thin sphere 0.60 → 1.15 and the binary images 0.68 → 0.89 and 0.95 → 1.28 (the `auto`
chunk), the exact-Kerr disk 0.31 → 1.34 (the maps written in place). On the CPU the
imaging legs moved most, native blocks I 384² at 128 threads 1.68 → 5.70 and the BlackLightPlus
deck's 192² image 0.60 → 6.58 (chunks sized per run and pulled by the workers); the Monte
Carlo legs at 128 threads stand at 1.10 and 1.09.
The rows still below 1 at best, mixed precision (0.88), native-block IQUV (0.90) and the
binary Stokes I image (0.89), are kernel-level differences (§4, the camera-ray kernel) that
the kernel audit tracks.

## 6. The Monte Carlo tables and the march on both backends

Measured on 2026-11-07 on `anta`, before (`be1bc9b`, and the trees before each change named
in the rows) and after this record's changes; scripts, logs and the figure in
`results/blp/mc_audit/speed/gpu/`. CPU rows: Julia threads on the shared host (load 25–95,
so CPU times scatter by about 15 %); GPU rows: one A100 with the four host threads of its
Slurm allocation. Every output is bitwise unchanged on each backend, and CPU against GPU
is unchanged.

**Monte Carlo**, AthenaK torus 00999 at stride 1 (10,303,459 zones), 10⁶ primaries; CPU
32 threads; the Compton phase at constant bias 10⁴ (7,302,016 children).

| phase | CPU before | CPU after | A100 before | A100 after |
|---|--:|--:|--:|--:|
| emission sites, on the host for both | 1.07 s | 0.88 s | 1.95 s | 2.04 s |
| zone tables | 20.8 s | 16.4 s | 154 s | 0.60 s |
| hot-cross table, once per process | 0.59 s | 0.54 s | 3.33 s | 0.05 s |
| thin photon phase, before the plunge stop and now | 24.7 s | 11.3 s | 8.84 s | 3.74 s |
| Compton photon phase | 63.0 s | 62.5 s | 74.8 s | 62.5 s |
| thin `mc_spectrum` call | 35.9 s | 31.7 s | 146 s | 16.0 s |
| the same call repeated | 31.0 s | 10.5 s | 150 s | 4.1 s |

The first device zone tables of a process add 15–17 s of compilation. The plunge stop
alone is 2.2× on the CPU and 2.1× on the A100 (24.9 → 11.3 s and 7.94 → 3.74 s in one
tree); the blocks of one warp bring the A100's thin phase from 4.34 to 3.74 s and, with the
child slots carried across chunks, its Compton phase from 74.8 to 62.5 s, the rate of 32
CPU threads. The Compton phase does not stop plunging photons that may still spawn.

**Imaging.** GPU best of five calls; CPU best of four on 16 threads, the trees
interleaved (torus), or best of two on 32 threads (closed form, crossings).

| path | trees | CPU before → after | A100 before → after |
|---|---|--:|--:|
| torus Stokes I 768², ghost layers | `cc4ce14` → `9f436c9` | 21.9 → 21.6 s | 0.465 → 0.457 s |
| torus IQUV 384², degenerate-limit segment | `5515dfa` → `cc4ce14` | 12.4 → 12.6 s | 0.540 → 0.544 s |
| torus IQUV 384², small-absorption segment | `9f436c9` → `be1bc9b` | 12.3 → 13.0 s | 0.538 → 0.562 s |
| Kerr disk 2048², radius-only path, product | observables on → off | 1.70 → 0.52 s | 0.458 → 0.305 s |
| Kerr disk 2048², radius-only path, program | observables on → off | 1.47 → 0.33 s | 0.205 → 0.036 s |
| marched disk B 500², production, cost-ordered queue | `be1bc9b` → this tree | 3.60 → 3.39 s | 0.208 → 0.163 s |
| marched disk B 500², precise, cost-ordered queue | `be1bc9b` → this tree | 5.37 → 4.91 s | 1.753 → 1.647 s |

The ghost layers cost neither backend anything measurable; the small-absorption segment
costs about 5 % of the IQUV march on both; the radius-only program is 4.4 times faster
than the full one on the CPU and 5.7 times on the A100.

What stays on the host of a GPU run: the site scan (`emission_sites`, 2–3 s at stride 1 on
four threads: its plasma decode contracts StaticArrays `muladd` chains, which a device
compiles differently) and the stochastic rounding of the counts (`zone_offsets`, one
stream in zone order); with the zones kept, a repeated run skips both the scan and the
tables. The closed-form radius path is bound by its host work on the GPU: at 2048² its
kernel takes 0.036 s, the product 0.31 s, most of it allocating and first touching the
288 MB of returned maps and filling the 192 MB of NaN observables it skips. The precise
march runs on the A100 at 52–61× four CPU threads on comparison A and 19× on B, where the
production march runs at 69–150×: Verner 9 keeps fifteen stage vectors live at the
255-register cap; a stage loop with one call site of the right-hand side, or a non-inlined
right-hand side, is not faster on B (2.41 and 1.80 s against 1.90 s) and changes the CPU
bits of the step, so the step stays as it is.
