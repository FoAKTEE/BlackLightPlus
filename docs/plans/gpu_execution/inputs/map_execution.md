# Map of the execution layer for GPU work

Tree: worktree `/data2/haiyangw/scratch/blp_wavefront`, branch `gpu-wavefront` at `a5923dd`
(identical to `main`). Paths are relative to the repository root. Toolchain of the CUDA
environment `/data2/haiyangw/scratch/wavefront/env`: Julia 1.12.6, KernelAbstractions
0.9.43, CUDA.jl 6.2.2 (KA backend in `CUDACore/src/CUDAKernels.jl`), Atomix 1.2.1,
GPUCompiler 1.23.0, StaticArrays 1.9.22, Adapt 4.7.1.

## 0. Facts that shape every design

1. There is exactly one kernel, `_foreach!` (`src/infra/Execution/kernel.jl:7-10`), one
   thread per item, no grid-stride loop, no shared memory, no barriers. `test/structure.jl:63`
   fails the suite if `src` holds a second `@kernel`; `test/structure.jl:66-67` fails it if
   `src` names `CUDA.`. Every new device algorithm (sort, scan, compaction, persistent
   loop) is either a *program* run by `_foreach!` or a device-protocol method implemented
   in `ext/BlackLightPlusCUDAExt.jl`.
2. A program is an `isbits`-after-`Adapt` functor called `prog(item, i, outs...)`; it never
   sees the KernelAbstractions context, so it cannot use `@localmem`, `@synchronize`,
   `@groupsize` or a group index (`kernel.jl:1-10`).
3. Every launch synchronizes the lane's stream before returning (`kernel.jl:42`); the
   cascade reads one `Int64` count back per generation with a blocking device-to-host copy
   (`cascade.jl:352-356`). There is no asynchronous launch, event or graph API.
4. Results are schedule-independent because (a) each item's work is a pure function of
   the item, (b) Monte Carlo streams are Philox counters keyed by genealogy, and (c) every
   cross-item sum is an exact integer-limb sum. Per-photon arithmetic is *not*
   order-free: its running `Float64` sums are tallied once per photon (§6, I7).
5. Bits depend on code shape: moving an inlining boundary changes `muladd` contraction
   (the Compton sampler behind a non-inlined call changed CPU bits,
   `docs/src/design/performance.md` §4; the Verner stage loop changed 1,355 of 20,000 CPU
   steps, §6 and `results/blp/mc_audit/speed/gpu/README.md` "Findings"). Register
   allocation, launch shape, visiting order and chunking do not change bits.
6. Measured kernel resources (`bench/kernel_audit_baseline.json`, sm_80, offline):
   `mc_torus_thin_primary` 255 regs, 2264 B stack, 312 B callee spill;
   `mc_torus_scatter_primary` 255 regs, 2960 B, 4/4 B spill, 1908 B callee spill;
   `mc_torus_scatter_layer` 255 regs, 2848 B, 1884 B callee spill; `mc_*_drain` 32 regs.
   At 255 registers an A100 SM holds 8 warps of 64. The precise march
   (`CrossingProgram`, `src/pipeline/Products/crossings.jl:26`) has no kernel-audit case
   and no launch traits.

## 1. Module layout and the layering it obeys

- `src/infra/Execution/Execution.jl:11-58`: include order devices → pool → residency →
  kernel → executor → backend → ray_sources → schedule → worklist → accumulators →
  children → cascade; exports at `:32-52`; `atexit(close_pools)` at `:54-56`.
- Layering (`test/structure.jl:12-13, 44-60`): `Numerics, Spacetime, Matter, Transfer,
  MonteCarlo, Data, Execution, Plans, Analysis, Products`. A module imports only earlier
  ones. Physics modules (`MonteCarlo` among them) may not use KernelAbstractions, CUDA or
  `Base.Threads`. `Execution` imports `Spacetime.camera_ray` (`Execution.jl:16`) and
  `Numerics` (`children.jl:6`, `accumulators.jl:8-9`).
- Other gates that bind a new design: only the eponymous file includes and exports
  (`structure.jl:71-80`); every module export has a docstring (`:121-127`); at most 54
  top-level exports (`:115`); at most 600 lines per source file (`:130-133`; today
  `accumulators.jl` 488, `pool.jl` 366, `cascade.jl` 365); no history or attribution prose
  (`:138-143`); `Threads.@threads` only in `infra/Data` (`:107-111`).
- The CUDA methods of the device protocol: `ext/BlackLightPlusCUDAExt.jl:14-15` imports
  `backend, activate!, functional, stage_alloc, stage_free, free!, prepare_launch!,
  stack_bytes, readonly, select_devices` and defines them for `CUDADevice` (`:17-91`).

## 2. From a program to a device launch

### 2.1 The program contract
- `executor.jl:126-141`: `foreach_item!(exec, prog, items; outs, order)` runs
  `prog(item, i, outs...)` once per item; the program writes item `item`'s outputs at
  `outs[k][i]`, where `i` is the item itself on threads and the position in the chunk on a
  device; a program without outputs writes through atomics. `order` permutes the visit
  (`executor.jl:160-161`, `schedule.jl:16-21`). The multi-part form queues several programs
  in one queue (`executor.jl:148-151, 167-169`), used by mixed fp64/fp32 lanes
  (`schedule.jl:162-167`, `src/pipeline/Products/image.jl:104-109`).
- Launch traits, per program type: `workgroupsize(prog)` (`kernel.jl:19`, default
  `nothing`), `chunk_hint(prog)` (`kernel.jl:28`), `stack_bytes(prog)` (`devices.jl:154`).
  Set today: `ImagingProgram{IQUV}` 64 (`programs.jl:56`), `MCPrimary` 32 (`programs.jl:440`),
  `MCLayer` 32 (`programs.jl:467`), 32 KiB stack for the closed-form Kerr programs
  (`analytic.jl:70, 106`, `analytic_movie.jl:41`, `analytic_surfaces.jl:90`).

### 2.2 Choosing the executor
- `backend.jl:30-44` `resolve`: `:cpu` → `ThreadedExecutor()`, `:gpu` → `PoolExecutor`
  over the selected CUDA devices (`backend.jl:132-149`), `:host_pool` → `PoolExecutor` over
  `HostDevice(1)`, `:auto` → the GPU pool when CUDA is loaded, functional and a device is
  idle (`backend.jl:113-130`), else threads; strings parse the `BLP_BACKEND` grammar
  (`backend.jl:6-7, 99-111`). Session default via `backend!`/`backend()` (`backend.jl:53-72`).
  Idle selection (≥ 8 GB free, ≤ 15 % utilisation): `ext/BlackLightPlusCUDAExt.jl:62-91`.

### 2.3 Chunk sizes
- `executor.jl:62-79` (`auto_chunk`) and `:97-110` (`chunk_for`, `resolve_chunk!`): one
  device → `2^20` items per launch; `d > 1` devices with `l` lanes →
  `clamp(cld(n, 2 l d), 2^16, 2^20)`; at most `max(2^16, 2^24 ÷ bytes_per_item)`; threads →
  `clamp(cld(n, 16 t), 16, 4096)`; then at most the program's `chunk_hint`.
- `chunk_ranges(n, c)` (`executor.jl:164-165`): chunk boundaries depend only on `n` and `c`.
- Cascade chunk (`cascade.jl:24-28`): pool `min(cld(n, 4·ndev), 2^20)`, threads
  `cld(n, 64 t)`, each at least an explicit executor chunk.

### 2.4 The threads path
- `executor.jl:172-187`: `run_workers!` (`pool.jl:217-231`) spawns `min(nthreads, nchunks)`
  tasks that pull `(part, range)` chunks from one `Channel` in queue order and call the host
  program directly on host arrays (`prog(item, item, outs...)`). No `Adapt`, no
  KernelAbstractions.

### 2.5 The pool path: lanes, jobs, streams
- `DevicePool(devices; lanes = 2)` (`pool.jl:86-105`) spawns `lanes` persistent tasks per
  device, stored device-major (`pool.jl:97-102, 123-125`). A lane activates its device once
  (`pool.jl:129-138`) and serves jobs from its `inbox` (`pool.jl:143-157`); each lane task
  has its own CUDA stream (CUDA.jl task-local streams; `performance.md` §1 "Pool"). Fewer
  Julia threads than `devices + 1` draws a warning (`pool.jl:93-96`).
- A job (`pool.jl:57-70`) is one shared `Channel` of items posted to every lane
  (`run_job!`, `pool.jl:204-213`); lanes pull items in queue order (`drain!`,
  `pool.jl:168-179`), so chunks balance dynamically across lanes and devices. The first
  failure is kept and rethrown after the queue drains. `enqueue!` (`pool.jl:187-191`) adds
  items from inside a running item (`stream_chunks`, `pool.jl:250`). Nested jobs from a
  lane are refused (`pool.jl:196-200`). `foreach_device` (`pool.jl:258-266`) runs one
  function per device; `on_device` (`pool.jl:269-276`) runs on a lane of a given device.
- Registry: one pool per `(devices, lanes)` (`pool.jl:293-316`), shared by every
  `PoolExecutor` of that list whatever its chunk (`executor.jl:45-54`).
- Lane-owned storage: `lane_buffer` (device vector, grows by free-and-reallocate,
  `pool.jl:325-332`), `lane_array` (device array of exact dims, `pool.jl:340-348`),
  `lane_stage` (pinned host staging, `pool.jl:357-366`; pinned allocation in the ext,
  `ext:34-38`). Only the owning lane's task touches them (`pool.jl:10-12`).

### 2.6 The device twin of a program
- `pool program → device twin`: `ProgramTwins` builds one twin per device per call, lazily,
  under a per-device lock (`residency.jl:88-105`), with `adapt_program`
  (`residency.jl:72-73`) = `Adapt.adapt(DeviceAdaptor(pool, dev, true), prog)`.
- `DeviceAdaptor` (`residency.jl:14-23`): every `Array` leaf becomes its cached device copy
  (`cached_twin`, uploaded once per device and found again by `objectid`, held weakly,
  `residency.jl:34-45`), wrapped by `readonly` (`devices.jl:157-165`; on CUDA a
  `ReadOnlyArray` that kernel conversion turns into `Base.Experimental.Const`, `ext:49-60`).
  Contiguous `view`s of a `CuArray` are `CuArray`s (GPUArrays `unsafe_contiguous_view`),
  so the cascade's lane views also read through `Const`.
- Types whose twin is not "arrays copied" define `Adapt.adapt_structure(::DeviceAdaptor, x)`:
  `ExactBins` → zeroed device twin registered with the host object
  (`accumulators.jl:148-151`); `ExactScalars` likewise (`accumulators.jl:417-418`);
  `ChildBuffer` → freshly allocated device buffers of the same capacity, registered
  (`children.jl:44-51`).

### 2.7 The launch
- `run_chunk!` (`executor.jl:208-218`): `device_items` passes a range as an `isbits`
  `UnitRange{Int}` (no copy) and stages any other list as `Int32` (or `Int`) through pinned
  memory into a lane buffer (`executor.jl:245-258`); output buffers are lane buffers keyed
  `(:out, k)` (`executor.jl:212`).
- `launch!` (`kernel.jl:33-43`): `prepare_launch!` (raises the CUDA stack limit when
  `stack_bytes(prog)` exceeds it, `ext:42-47`), `_foreach!(lane.backend)` (a dynamically
  sized KA kernel), then `kern(dprog, items, n, outs...; ndrange = n[, workgroupsize =
  min(wg, n)])` and `KernelAbstractions.synchronize(lane.backend)`.
- What KA 0.9.43 + CUDA.jl 6.2.2 do with that call (`CUDACore/src/CUDAKernels.jl:107-150`):
  `@cuda launch=false always_inline=false` compiles (cached per argument types) without
  `maxthreads` (dynamic kernel) and without `maxregs`; with no workgroup size the
  occupancy API (`launch_configuration(...; max_threads = n)`) picks the block size;
  `blocks = cld(n, threads)`; bounds are masked by `__validindex` and again by `i <= n`
  (`kernel.jl:9`). `KA.synchronize(::CUDABackend) = synchronize()` waits on the calling
  task's stream (`CUDAKernels.jl:28`).
- Kernel-argument conversion (`cudaconvert`) turns `CuArray` into `CuDeviceArray` and
  `ReadOnlyArray` into `Const`; the program struct becomes `isbits` (checked in tests by
  `device_isbits`, `test/decks.jl:200-220`).

### 2.8 Outputs back to the host
- `gather!(lane, k, o, d, idx)` (`executor.jl:220-226`): device → pinned stage → `scatter!`
  by item (`executor.jl:229-235`; a contiguous range is one `copyto!`). The lane scatters
  while the other lanes compute; the `2^24`-byte cap on a launch's outputs sizes that overlap
  (`executor.jl:65, 76`).
- Programs without outputs (all Monte Carlo programs) write only through atomics into
  device twins of accumulators, gathered after the run (§4.3).

### 2.9 Other drivers on the same machinery
- `stream_blocks!` (`worklist.jl:56-101`): an AMR worklist where each chunk's visits
  `enqueue!` child block chunks into the running job (`worklist.jl:67`): the package's
  existing host-driven streaming queue across lanes and devices.
- `trace_fixed` (`src/pipeline/Products/trace.jl:66-80`): a resumable march, window by
  window, the live list compacted on the host (`absorb_window!`, `trace.jl:89`), the
  resume states (`TraceState`, `programs.jl:106`; `TraceProgram`, `programs.jl:128`) uploaded as
  a host array per window; successive windows continue the march exactly.

## 3. The cascade

### 3.1 Pieces
- `ChildBuffer{P,R,C,W}(photons, rngs, count, layout, twins)` (`children.jl:30-39`): an
  append queue of `Superphoton` (144 bytes) and `MCRng` (24 bytes) as two arrays, an
  `Int64` counter of spawn attempts, the stream layout. `spawn!` (`children.jl:69-76`):
  `slot = Atomix.@atomic count[1] += 1`, store when `slot <= capacity`, else count only.
  `child_stream` (`children.jl:60-61`) names the child stream `child_rng(rng, event,
  layout)` and whether the ordinal fits.
- `Generation(photons, rngs)` (`children.jl:134-140`): what a layer program reads.
- `Cascade{P,L,S}(primary, layer, slot, depth, capacity)` (`cascade.jl:83-89`); the child
  slot is found by an `Adapt` probe that records every `ChildBuffer` in the program
  (`cascade.jl:92-101`).
- `CascadeSwap(lane, children, parents, scratch)` (`cascade.jl:115-154`): the per-launch
  rebinding adaptor. Every `ChildBuffer` becomes the chunk's child buffer, every
  `Generation` the previous generation, and with `scratch` every `ExactBins`/`ExactScalars`
  writes zeroed scratch arrays of the same types: lane arrays keyed by position
  `(:cascade_scratch, k)` on a pool, fresh host arrays on threads (`cascade.jl:143-154`).
  `pairs` lists (own, scratch); `allocated` flags a fresh lane array so it is zeroed.
- `Drain(totals, scratch, keep)` (`cascade.jl:163-187`): a program over
  `1:maximum(length, scratch)` that adds each nonzero scratch element into its total
  (atomic, when `keep`) and zeroes it.

### 3.2 Threads (`cascade.jl:191-264`)
- `run_workers!` over the chunks of `cascade_chunk` (`cld(n, 64 t)` primaries); each task
  holds two task-local `ChildBuffer`s of `capacity` (default 4096) slots
  (`cascade.jl:194-205`).
- `chunk_generations!(c, r, bufs)` (`cascade.jl:208-235`): for `g = 0:depth`, out/parents
  alternate (`:216-217`), the program is the primary (items `r`) or the layer (items
  `1:m`), rebound by `CascadeSwap(nothing, out, gen, false)` (`:222`); generations run
  depth-first per chunk, breadth-first inside it.
- `run_growing!` (`cascade.jl:248-264`): items run in order into the shared totals; an item
  whose children overflow resets the counter to where its children began, doubles the
  buffer, and runs again as the scratch twin `rerun()` (fresh zeroed host scratch,
  discarded), so its additions count once; the scratch program must have the program's
  type (`:258-259`). Only the overflowing item reruns.

### 3.3 Pool (`cascade.jl:268-365`)
- `run_cascade!` (`cascade.jl:268-282`): one `ProgramTwins` each for the primary and the
  layer, `foreach_chunk` over the chunk ranges, `fresh[k]` and `caps[k]` per lane, the
  spawned counts summed atomically; the program's own child-buffer slot and its twins are
  emptied at the end (`:279`).
- Generation 0 without a child slot (thin runs): one launch over the range `r` into the
  device twins of the accumulators (`cascade.jl:292-295`).
- With a slot, per chunk on lane `k` (`cascade.jl:296-337`):
  - initial child slots `cap = max(4096, cld(length(r), 2), caps[k])` (`:298-299`); one
    device counter `(:cascade, :count)` (`:300`);
  - for each generation `g`: parents are read-only views of side `:a`/`:b` of
    `lane_children` (`:308-310, 341-343`), output the other side with `slots = cap`
    (`0` in the final generation: children counted, not stored, `:311`);
  - the device twin is rebound by `CascadeSwap(lane, out, parents, !final)` (`:315-316`):
    non-final generations record into lane scratch, the final one into the device totals;
  - launch sequence of a non-final generation: [zeroing `Drain` when the lane's scratch is
    fresh or reallocated, `:317-320`] → `set_count!(0)` (pinned host → device, `:321,
    345-350`) → `launch!` + sync (`:322`) → `get_count` (blocking device → host, `:323,
    352-356`) → on overflow: zeroing `Drain`, `slots = cap = n_out + n_out ÷ 8`, `caps[k] =
    cap`, the whole generation runs again (`:324-328`) → `Drain(keep = true)` + sync into
    the device totals (`:330, 359-365`);
  - `m = n_out` parents for the next generation; stop at `m == 0` (`:331-335`).
- Consequence worth measuring: every lane's first chunk starts with `cld(chunk, 2)` slots,
  so in a Compton run with several children per primary the first chunk of each lane
  reruns its whole primary generation once. With `n = 10^6` on one A100 the chunk is
  250,000 (4 chunks, 2 lanes), so 2 of 4 primary generations run twice; with two A100s
  4 of 8. The `capacity` keyword of `mc_spectrum` (`montecarlo.jl:249-262`) presizes it.

### 3.4 What the Monte Carlo product plugs in
- `run_layers` (`src/pipeline/Products/montecarlo.jl:387-397`):
  `foreach_cascade!(exec, MCPrimary(source, transport, sink, seed), MCLayer(Superphoton[],
  MCRng[], transport, sink), n0, depth; capacity)` with `depth = layout.max_layer` (the
  scattering order cap `K`), then `gather!` of every order's bins and of the ledger. The
  cascade `chunk` is not exposed by `mc_spectrum`.
- `MCPrimary` (`programs.jl:420-437`): `primary_rng(seed, g)`, `emit` (`ZoneSource`:
  `searchsortedfirst(offset, g)` over the zone offsets, decode of the zone's site, generation,
  `programs.jl:392-409`), ledger `EMITTED`, `track!`. `MCLayer` (`programs.jl:452-465`):
  `track!` of `generation.photons[j]` on `generation.rngs[j]`.
- `MCSink` (`programs.jl:303-344`): `record!` into `orders[min(nscatt, K) + 1]` and the
  ledger; `spawn!`/`child_stream` delegate to the `ChildBuffer`; `tally!` adds the
  photon's `MarchLedger`; `at_stripe` puts item `g`'s ledger additions in stripe
  `(g − 1) mod 64` (`programs.jl:244, 314-315`).
- `track!` (`src/physics/MonteCarlo/track.jl:276-341`): one loop over steps
  (`:292-336`) with loop-carried `y, w, tau_a, tau_s, na0, ns0, th0, r_prev, nev, rng` and
  the per-photon running sums `loss, loss_final, spawned, unspawned, absorbed`; per step a
  geodesic step, a plasma sample, rates, and with scattering `step_events`
  (`track.jl:221-251`): a Poisson number of events, each a partial geodesic step, a
  plasma sample at the event (`scatter_site`), the Compton kick (`scatter_superphoton`,
  `:237`) and `spawn!` (`:243`). `record!` and `tally!` happen once, after the loop
  (`:338-340`). Physics sees only the `Sink` protocol (`src/physics/MonteCarlo/sinks.jl:1-87`).

## 4. Exact accumulators and atomics

### 4.1 Integer limbs (`src/physics/Numerics/exactsum.jl`)
- A finite `Float64` is an integer multiple of `2^-1074`; a cell holds it in 68 signed
  `Int64` limbs of 31 bits plus one non-finite counter (`XSUM_NSLOT = 69`,
  `exactsum.jl:1-22`). `xsum_parts` splits an addend into one limb index and three
  increments, pure integer arithmetic (`:32-49`). `xsum_atomic_add!` issues at most three
  atomic `Int64` additions (zero increments skipped, `:79-90`); `xsum_add!` is the
  non-atomic form (`:58-69`). A cell takes at least `2^32 − 1` addends before a limb can
  overflow (`:20-21`). Rounding to nearest-even happens once, on the host, through
  `BigInt` (`:100-131`); `xsum_round_all` rounds a whole array (`:147-153`).

### 4.2 Accumulators (`src/infra/Execution/accumulators.jl`)
- `ExactBins` (`:112-151`): per (row, frequency bin) an `Int64` count and limbs of `Σw`,
  `ΣwE`, `Σ(wE)²` (arrays `69 × rows × n_ebins`; 441,600 bytes each for 4 × 200).
  `record!` (`:161-169`) → `record_rows!` (`:171-189`) → `add_cell!` (`:191-198`): one count
  atomic plus up to three limb atomics per sum, i.e. up to 10 atomics per row; an
  `InclinationBands((17, 50, 90), 10)` photon hits the all-sky row and at most one band.
- `ExactScalars` (`:408-447`): `add!` = up to 3 limb atomics + 1 count atomic;
  `add_limbs!` merges a local limb accumulator with one atomic per nonzero limb (`:440-447`).
  The Monte Carlo ledger is `ExactScalars(11 · 64)` (`programs.jl:244-272`): eleven
  quantities in 64 stripes, read back as exact sums over the stripes (`ledger_sum`,
  `ledger_count`, `programs.jl:275-289`).
- Atomics everywhere are `Atomix.@atomic` on `Int64` (native 64-bit integer atomics on
  sm_80; `lock xadd` on the host). No floating-point atomic exists in the layer.

### 4.3 Host and device, and the gather
- Threads: programs write the host arrays directly, shared by every task.
- Pool: each program twin on each device gets zeroed accumulator twins registered on the
  host object (`Twins`, `accumulators.jl:13-33`); in a cascade the primary and the layer
  each register their own (two per device per accumulator). Inside a non-final cascade
  generation the writes go to lane scratch first (§3.3). After the run the caller calls
  `gather!` (`accumulators.jl:226-232, 456-462`; `children.jl:105-125` for child-buffer
  twins of plain `foreach_item!` programs), which adds each twin into the host arrays by
  integer addition (`merge!`, `accumulators.jl:209-216, 450-454`) and frees it.
  `Spectrum`/`scalar` refuse pending twins (`accumulators.jl:234, 273-280, 473-484`).
- Atomic traffic per tracked photon (scattering runs): about 4 ledger adds × 4 atomics in
  `tally!`, 4 more for `ESCAPED`/`EMITTED`, up to 20 for an escape record, one counter
  atomic per spawn. Contended addresses: the bins of the spectral peak (same cell for
  many photons) and the single spawn counter per chunk (`children.jl:70`); the ledger is
  spread by stripe so the 32 threads of a warp hit 32 stripes.

## 5. Several devices

- The pool shares one chunk queue among all lanes of all devices (§2.5), so devices pull
  chunks until the queue is empty; with `:auto` the image chunk is
  `clamp(cld(n, 2 l d), 2^16, 2^20)` (two chunks per lane) and the cascade chunk
  `cld(n, 4 d)` (four per device). Children never leave the lane that made them
  (`cascade.jl:1-7`): no balancing below the chunk.
- Residency is per device: each device uploads its own copy of every array a program holds
  (the plasma grid, tables, zone sites), lazily, on its first lane that needs it
  (`residency.jl:34-45, 88-105`).
- Gathering: per-item outputs are scattered by item into the host arrays by whichever lane
  ran the chunk (§2.8); accumulators come back as one twin per device per program twin,
  folded by exact integer addition (§4.3); the cascade's spawned counts are summed on the
  host (`cascade.jl:271-276`).
- Tests run the device comparisons on 1, 2, 3 and 5 logical `HostDevice`s
  (`test/decks.jl:158-161`) and on real GPUs under `BLP_TEST_DEVICES=cuda`
  (`test/decks.jl:163-197`); the cascade test checks bitwise equality across threads of any
  chunking, pools of every size, fine chunks and capacities 1 and 2
  (`test/infra/execution_cascade.jl:115-135`).

## 6. Invariants that make results independent of scheduling

| # | invariant | where | what a new design must keep |
|---|---|---|---|
| I1 | An item's work and outputs depend only on the item; `i` only addresses outputs | `executor.jl:1-4, 130-141`; `kernel.jl:1-5` | a program never uses `i` (chunk position, queue slot, thread id) in arithmetic |
| I2 | Chunk boundaries are a function of `n` and the chunk | `executor.jl:163-165` | chunking may change freely |
| I3 | Streams are counter-based Philox keyed by identity: `primary_rng(seed, gid)`; `child_rng(parent, event, layout)` puts the event ordinal into disjoint identity fields | `philox.jl:73-77, 96-110, 192-220`; `children.jl:60-61` | a resumable photon carries its `rng` and its event ordinal `nev` in its state |
| I4 | Cross-item sums are exact integer-limb sums with integer atomics, rounded once on the host | `exactsum.jl:1-12, 79-90, 100-131`; `accumulators.jl:191-198, 425-447` | no floating-point atomics, no float partial sums across photons |
| I5 | The set of children is deterministic; their storage order is not | `children.jl:1-4, 17-18, 69-76` | any permutation of a generation is result-neutral |
| I6 | An overflow rerun counts every addition once (scratch twins, `Drain`, same program type) | `cascade.jl:47-61, 115-187, 248-264, 311-334` | a state-mutating launch that can overflow needs its inputs preserved and its additions in scratch |
| I7 | Inside one photon the arithmetic is sequential on one thread, and its running `Float64` sums are tallied once at the end | `track.jl:287-340`; `programs.jl:330-344` | carry `loss, loss_final, spawned, unspawned, absorbed` in the state; tallying per segment changes the ledger bits and its addend counts |
| I8 | The ledger stripe of an addition depends on the item index, but every read sums all stripes | `programs.jl:244, 275-289, 314-315` | reordering photons only moves additions between stripes |
| I9 | Host-side random decisions are serial and keyed: zone rounding from one stream in zone order | `montecarlo.jl:372-382`; `run_rng`, `philox.jl:202` | keep them on the host or reproduce the order |
| I10 | The same compiled program runs on every device of one architecture; CPU and GPU contract differently | `test/decks.jl:168-174`; README "Findings" | one GPU vs two GPUs stays bitwise only while all devices run identical code |
| I11 | The executor is chosen only through `resolve` | `backend.jl:30-44` | no backend in names of plans, configs or functions |

Not invariant under restructuring: the code shape of `track!`, `step_events`,
`scatter_superphoton`, the plasma sample and the Verner step (fact 5). A split of the
photon loop at the loop's back edge (`track.jl:292`) carries only loop-carried values
across the cut and is the lowest-risk cut; a split inside `step_events` (sampler in its own
stage) is the highest-risk one. Every cut is checked bitwise on both backends.

## 7. Where the GPU designs plug in

### 7.1 Coherence sort of the photons
- Generations on a pool: in `chunk_generations!(lane, …)` between building `parents`
  (`cascade.jl:308-310`) and the first launch of the generation (`:322`). Needed: a key
  program over `1:m` writing a key per child into a lane buffer (cell or meshblock of
  `x`, radius shell, direction octant, energy bin; a Products program over a Matter
  lookup), a device sort or counting sort, and a gather program writing the permuted
  photons and streams into a third lane side (e.g. `(:cascade, :c)`); the parents become
  that side. The overflow rerun then replays the sorted parents, which is fine (I5).
- Generations on threads: in `chunk_generations!(c, r, bufs)` before `Generation(...)`
  (`cascade.jl:219`), a host `sortperm` and permutation of the first `m` slots. Possible
  CPU gain through cache locality of the plasma lookups; measure.
- Primaries: `foreach_cascade!` has no `order` (`cascade.jl:67-79`) and launches the range
  `r` (`cascade.jl:293, 307`). A primary order would be staged by `device_items`
  (`executor.jl:246-254`); item stays the global index, so results are unchanged (I1, I3).
  Consecutive primaries already share zones (`ZoneSource`, `programs.jl:404-409`), whose
  order is the site order.
- Sort primitive: either a program-built counting sort (histogram by atomics through
  `_foreach!`, a scan, a scatter with an atomic cursor per bucket: order inside a bucket
  arbitrary, which is allowed) or a device-protocol `sort` with the CUDA.jl implementation
  in the extension and a host fallback in `devices.jl`.

### 7.2 A queue / wavefront primitive
- Existing parts to reuse: the append queue with overflow count (`ChildBuffer`,
  `children.jl:30-76`), lane ping-pong buffers (`lane_children`, `cascade.jl:340-343`), the
  device counter round trip (`set_count!`/`get_count`, `cascade.jl:345-356`), the per-launch
  rebinding of buffers and scratch inside an existing device twin without re-adapting from
  the host (`CascadeSwap`, `cascade.jl:115-154`), scratch folding (`Drain`,
  `cascade.jl:163-187`), and the resumable-window precedent (`trace.jl:66-80`).
- Shape that fits the layer: a driver in a new `Execution` file (the 600-line budget rules
  out growing `cascade.jl` much), parallel to `chunk_generations!(lane, …)`: per chunk a
  state queue `A` (live photons), an output queue `B`, and the child queue; a *segment*
  program advances each live photon of `A` by at most `K` steps, appends it to `B` if still
  live (atomic slot), spawns children, and calls `record!`/`tally!` only when the photon
  retires; the host loop reads the count of `B`, swaps, repeats. Children can enter the
  live queue as new photons at once: correctness needs no generation barrier, because a
  photon's spawning is decided inside `track!` by `nscatt < max_scatt`
  (`track.jl:287`) and its stream by its genealogy.
- Needed from `MonteCarlo` (physics, no KernelAbstractions): `track!` split into begin /
  advance-K-steps / finish over an `isbits` march state holding the loop-carried values,
  the running sums (I7), `nev`, `rng`, the outcome and the entering `Superphoton` (about
  350 bytes per photon; 88 MB per side for a 250,000-photon chunk). The loop body must
  compile to the same arithmetic (§6).
- Overflow: a segment mutates photon state, so a rerun after the child queue overflows
  needs `A` intact (segments read `A`, write `B`) and the segment's additions in scratch,
  drained only when its children fit — `CascadeSwap(scratch = true)` + `Drain` at segment
  granularity. A cheaper alternative removes the full-generation rerun even in today's
  cascade: a parent whose spawn finds no slot records `(item, first lost ordinal)` in an
  overflow list (its later spawns also fail, the counter only grows), and only those
  parents run again into discarded scratch, storing children from that ordinal on; this
  needs `spawn!` to report success and the sink to carry the ordinal threshold.
- Cost per segment on a lane: one `set_count!`, one launch + sync, one blocking count read,
  one `Drain` launch + sync. Segments must be long enough (milliseconds) to hide that; the
  two lanes per device overlap each other's host round trips.

### 7.3 Persistent threads
- As a program, inside the one kernel: launch `W` items (resident threads: SMs × resident
  blocks × workgroup), each looping `k = Atomix.@atomic ctr[1] += 1` until `k > n`. The
  counter is a lane buffer like `(:cascade, :count)` (`cascade.jl:300`), reset with
  `set_count!`. `W` needs an occupancy query: a new device-protocol function
  (`devices.jl` pattern) implemented in the extension with CUDA's occupancy API, with a
  `HostDevice` and thread fallback.
- Effect on the megakernel: removes block tails (a thread whose photon ends takes the next
  one) but keeps intra-warp divergence (warp lanes sit at different loop positions). Pairs
  naturally with §7.2 (a persistent segment kernel pulling from `A`).
- Results: unchanged by I1 (each photon is processed by exactly one thread, whichever).
- The ledger stripe must come from the photon's item, not from the thread (I8 holds
  either way, but keeping `at_stripe(item)` keeps the contention pattern).

### 7.4 Multi-GPU distribution
- Today: dynamic chunk queue over every lane (`pool.jl:204-213`; `cascade.jl:275-277`),
  four cascade chunks per device (`cascade.jl:24-25`), children pinned to their lane,
  per-device accumulator twins folded exactly.
- Result-neutral knobs: a finer cascade chunk (the `chunk` keyword of `foreach_cascade!`,
  `cascade.jl:67-68`, not exposed by `mc_spectrum`, `montecarlo.jl:391-393`); chunk order
  by predicted cost (the queue is index order); a shared initial child capacity so lanes
  do not each overflow their first chunk (`caps` is per lane, `cascade.jl:273, 298-299,
  327`); and child migration: a lane posts surplus children as new items of a streaming job
  (`stream_chunks`/`enqueue!`, `pool.jl:187-191, 250`), carried as host arrays and uploaded
  by the receiving lane (PCIe cost ~170 bytes per child; 7.3e6 children ≈ 1.2 GB), with
  device-to-device copies only through a new extension method.
- Two A100s in a 4-CPU Slurm job: `julia -t 4` serves 4 lane tasks plus the caller
  (`pool.jl:93-96`); whether CUDA.jl's `synchronize` spins on a host thread decides
  whether `lanes = 1` is better on two devices.

### 7.5 Register and occupancy control (MC megakernel, Verner 9 march)
- `launch!` passes only a workgroup size (`kernel.jl:36-41`); KA never passes `maxregs`
  and passes `maxthreads` only for statically sized kernels (`CUDAKernels.jl:118-123`). A
  register cap needs an extension-level launch for `CUDADevice` that compiles
  `obj.f(ctx, args...)` with `@cuda launch=false maxregs=…` using KA's context, selected
  by a new launch trait. Register allocation does not change arithmetic, so caps and
  workgroup sizes are bitwise-safe experiments (verify per deck).
- `CrossingProgram` (`crossings.jl:26`) needs a kernel-audit case in
  `bench/kernel_audit.jl` before register work; it sets no `workgroupsize`.

## 8. Blockers and risks

| # | blocker | where | consequence / way around |
|---|---|---|---|
| B1 | One `@kernel` in `src` | `test/structure.jl:63` | sort, scan, compaction, persistent loops must be programs run by `_foreach!`, or extension methods; a second kernel changes the standard |
| B2 | No `CUDA.` in `src` | `test/structure.jl:66-67` | vendor primitives (sort, occupancy, warp vote/shuffle) only as device-protocol methods or methods on `CuDeviceArray` types in `ext/`, each with a host fallback |
| B3 | Programs get no KA context | `kernel.jl:7-10` | no shared memory, barriers or group index in a program; KA's CPU backend splits `@synchronize` syntactically in the kernel body, so block-cooperative algorithms (shared-memory binning, block queues) need the kernel body itself to change |
| B4 | No warp primitives in KA 0.9.43 | toolchain | warp-aggregated atomics or ballot-based compaction only through extension methods dispatched on device array types |
| B5 | Every launch synchronizes; ndrange is set on the host; counts come back by blocking copies | `kernel.jl:42`; `cascade.jl:345-356` | fine-grained wavefronts pay a host round trip per launch; device-driven continuation would need over-provisioned launches that read their length on the device, or a graph API in the extension |
| B6 | Bits follow code shape (contraction across inlining boundaries) | `performance.md` §4 (sampler), §6 (Verner); speed README "Findings" | any split of `track!`/`step_events`/the sampler/Verner can change CPU and GPU bits; cut at the step-loop back edge first; verify every deck on both backends |
| B7 | Per-photon running sums tallied once | `track.jl:287-340` | the march state carries them; no per-segment tally |
| B8 | Overflow recovery is per generation | `cascade.jl:311-334` | a state-mutating segment needs preserved inputs plus scratch; today's first-chunk rerun per lane is a measurable cost of its own (§3.3) |
| B9 | Layering and budgets | `test/structure.jl:12-13, 44-60, 115, 121-133` | resumable march in `MonteCarlo` (no KA, no Threads), queues in `Execution`, wiring in `Products`; new file for a new driver; docstrings on exports |
| B10 | Lane-owned buffers | `pool.jl:10-12, 322-324` | a queue shared by the two lanes of a device or children moved across devices need an ownership rule |
| B11 | Children pinned to their lane; coarse cascade chunks; `chunk` not exposed | `cascade.jl:1-7, 24-25`; `montecarlo.jl:391-393` | two-device balance of Compton runs rests on four chunks per device |
| B12 | No register cap through KA | `kernel.jl:33-43`; `CUDAKernels.jl:107-150` | extension-level launch path needed for `maxregs` |
| B13 | Per-device residency copies | `residency.jl:34-45` | each GPU uploads the full grid and tables once per process |
| B14 | The emission-site scan runs on the host | speed README; `montecarlo.jl:345` | bounds end-to-end gains of a GPU run (2 s at stride 1); outside `Execution` |

## 9. Verification hooks

- `test/infra/execution_cascade.jl` (bitwise across executors, chunks, capacities, 1–5
  devices), `test/infra/execution_accumulators.jl`, `test/infra/execution_executors.jl`,
  `test/infra/execution_pool.jl`; `BLP_TEST_DEVICES=cuda` runs the device comparisons on
  GPUs (`test/decks.jl:163-197`); `test/structure.jl` for the gates.
- Fixed decks: `results/blp/mc_audit/speed/gpu/scripts/decks.jl` and `compare_decks.jl` in
  the main checkout (13 decks: Monte Carlo torus thin/Compton at stride 8, torus stride 1,
  both spheres, crossings A/B, closed-form disk, torus I/IQUV), baselines
  `logs/decks_cpu.log` and `logs/decks_gpu_3789.log`; outputs serialized under
  `DECKS_OUT` (default `/data2/haiyangw/scratch/gpuspeed/decks`).
- Kernel resources: `bench/kernel_audit.jl` (offline sm_80, `--check` against
  `bench/kernel_audit_baseline.json`, `--device` against a GPU's own compilation).
- Instrumentation points for profiling runs (local, not committed): the overflow branch
  `cascade.jl:324-328` (reruns per lane and chunk), `n_out` per generation (`:323`), and
  launch timing around `launch!` (`kernel.jl:33-43`).
