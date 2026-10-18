# BLP performance record

How BLP spends its time, the knobs that change it, and every speed capability of the
reference package (`src/`) with its form in BLP. Protocols and types are specified in
`docs/ARCHITECTURE.md` §4.2 and §5; the measurements are in `docs/VERIFICATION.md` §4
(CPU against the A100) and §4.1 (against the reference package on the same decks). The
results of every product are bitwise independent of everything in this document: the
executor, the chunk, the lane count, the device count, the queue order and the child-buffer
capacity change the wall time and nothing else.

## 1. Execution model

- **Programs and the one kernel.** A product is a per-item program run by an executor:
  `ThreadedExecutor` calls it from Julia threads on the host arrays; `PoolExecutor` runs its
  device twin through the single `_foreach!` kernel on the lanes of a `DevicePool`. The
  program supplies its launch traits: `workgroupsize` (64 for the polarized imaging
  program, else the backend's choice), `stack_bytes` (32 KiB for the exact-Kerr program)
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
- **The cascade.** Scattering runs as a cascade (`foreach_cascade!`): a chunk of primaries
  and every scattering generation of their children run on the worker that took the chunk.
  On a lane the children live in two device buffers swapped per generation, and the only
  host read per generation is their count; a generation records into lane scratch sums
  that are folded into the device's totals once its children fit, and a generation whose
  children overflow the buffer runs again in full after the buffer grows. On threads each
  task keeps two growing buffers and reruns only the photon whose children overflowed.
  Child streams are keyed by genealogy and the sums are exact, so the spectrum does not
  depend on the chunking, the lane or the arrival order.

## 2. Defaults and tuning

| knob | default | where | effect |
|---|---|---|---|
| executor | `:auto` | `backend` keyword > `backend!` > `BLP_BACKEND` > `:auto` | GPUs when CUDA is loaded and a device is idle, else threads |
| chunk | `:auto` | `ThreadedExecutor(; chunk)`, `PoolExecutor(devices; chunk)`, `DevicePool(devices; chunk)` | items per launch or per thread task; `describe(exec).chunk` reports the chunk of the last run |
| lanes | 2 | `PoolExecutor(devices; lanes)` | worker tasks and streams per device; pools differ per lane count |
| queue | cost order on Kerr, identity elsewhere | `prepare(cfg; queue = perm)` | the order pixels enter the queue; for a movie, `march_sort_order(vec(shadow(plan).steps))` of the previous frame |
| cascade chunk | `cascade_chunk(exec, n)` | `foreach_cascade!(…; chunk)` | primaries per cascade chunk |
| child slots | at least 4096, half the chunk | `mc_spectrum(…; capacity)` | first size of a generation's child buffer; overflow grows it and reruns |
| hot-cross table | built once per plan | `mc_spectrum(…; table)` | `hotcross_table(plan)` memoises the table in the plan; a given table is used as is |
| workgroup | backend's choice; 64 for IQUV imaging | `workgroupsize(prog)` | threads per block of the launch |
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
(`BLP/bench/bench.jl` with `BLP_TUNE=1` sweeps lanes and chunks and records the best per
leg); the lane count for kernels at the register cap; the queue for a movie; `capacity`
only to avoid the overflow rerun on a deck whose scattering multiplicity is known.

**Kernel resources.** `BLP/bench/kernel_audit.jl` compiles the device twin of every program
kind for sm_80 without a GPU and reports registers, stack and spills against
`kernel_audit_baseline.json`; `--check` fails on growth, `--write` records a deliberate
change, `--device` compares with a GPU's own compilation (`BLP/bench/README.md`).

## 3. Capabilities of the reference package and their form in BLP

Status: *same* (the mechanism is the same), *form* (the same effect through a different
mechanism), *beyond* (BLP has more than the reference), *not carried* (with the reason in
§4). The reference locations are in `src/`, BLP's in `BLP/src/` unless given.

### 3.1 Devices, pool and residency

| capability | reference | BLP | status |
|---|---|---|---|
| Persistent pool: worker tasks per device, one stream each, shared queue | `gpu_pool.jl` | `DevicePool`, `pool.jl` | same |
| Pinned staging buffers, freed explicitly | `gpu_pool.jl` | `lane_stage`, `BLPCUDAExt.jl` | same |
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

| capability | reference | BLP | status |
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

| capability | reference | BLP | status |
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

| capability | reference | BLP | status |
|---|---|---|---|
| Every scattering order inside each chunk; children on the device | `mc_gpu.jl` | `foreach_cascade!`, `cascade.jl` | same |
| Overflow rerun at the exact size, per chunk and layer | `mc_gpu.jl` | per chunk and generation; on threads per photon | same |
| Tables built once, uploaded concurrently | `mc_gpu.jl` | hot-cross table memoised in the plan; weight tables per call; uploads lazy per device | form |
| Ancestry-keyed streams; exact accumulation | `rng.jl`, `exact_sum.jl` | `philox.jl`, `exactsum.jl`, `accumulators.jl` | same |
| 64 ledger stripes | `mc_gpu.jl` | `LEDGER_STRIPES` | same |
| Compton sampler behind a non-inlined call | `mc_gpu.jl` | inlined sampler | not carried (§4) |
| CPU: per-thread private accumulators, depth-first scattering | `mc_cpu.jl` | task-local child buffers, shared exact atomics | form |

### 3.5 Diagnostics and gates

| capability | reference | BLP | status |
|---|---|---|---|
| Offline register, stack and spill audit with a baseline gate | `bench/kernel_audit.jl` | `BLP/bench/kernel_audit.jl`, `--check` | same |
| Multi-GPU scaling | 1.7–1.8× on two devices | `docs/VERIFICATION.md` §4.1 | measured |
| Stiffness telemetry; sampler probes | `gpu_pol.jl`, `gpu_amr.jl` | not carried | §4 |

## 4. Not carried, and why

- **The two-phase stored-trajectory render as the CPU default.** BLP's fused march is
  1.2–4.3× faster on the CPU (`docs/VERIFICATION.md` §4.1); the stored path remains as
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
  measurements, not products; `BLP/bench/` and the kernel audit take their place.
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

Geometric mean of BLP's rate over the reference's on the same hardware, before the
capabilities of §3 were in their present form (BLP at 65481d9, `anta_2026-10-17_reference.md`)
and after (7be2569, `anta_2026-10-18_reference.md`); the per-leg rows and the profiled
phases are in `docs/VERIFICATION.md` §4.1. Every pair of outputs is bitwise identical on
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
imaging legs moved most, native blocks I 384² at 128 threads 1.68 → 5.70 and the BLP
deck's 192² image 0.60 → 6.58 (chunks sized per run and pulled by the workers); the Monte
Carlo legs at 128 threads stand at 1.10 and 1.09.
The rows still below 1 at best, mixed precision (0.88), native-block IQUV (0.90) and the
binary Stokes I image (0.89), are kernel-level differences (§4, the camera-ray kernel) that
the kernel audit tracks.
