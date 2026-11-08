# GPU execution

How BlackLightPlus runs its programs on Julia threads and on GPUs: which schedule each
executor uses, the primitives and buffers of the device schedule of the Monte Carlo
photon phase, the invariants that keep every result bitwise independent of the
schedule and why they hold, how a program is added, and how execution is measured. The
program contract, the pool and the residency protocol are specified in
[Architecture](@ref) §4.2; the knobs and the records of measured speed are in
[Performance](@ref).

**Status.** Normative. A component marked *gated* is part of the standard only while
its gate in §9 holds; without it the component is absent and the schedule of §3 serves
its programs.

## 1. The measurements the design rests on

One A100 80 GB PCIe inside a Slurm job with four host threads (SM clock 1,065 MHz in
every sampled range), the AthenaK torus 00999 at stride 1 (10,303,459 zones), 10⁶
primaries, Kerr a = 0.9; Compton at constant bias 10⁴ with three scattering orders
(7,302,016 children). Nsight Compute and Nsight Systems on the history schedule of §3,
and a probe that stamps every photon's steps, events, start, end and SM.

| measurement | value | consequence |
|---|---|---|
| lane efficiency E = Σ steps / Σ over warps of 32 × the warp's largest step count | 0.201 thin primaries, 0.187 Compton primaries, 0.300 layers; 6.4 active threads per warp instruction (thin); 98.7 % of branch targets uniform inside a step | the step-count spread of a warp, not branching inside a step, sets the SIMT efficiency: a warp must not hold photons for their whole lives |
| E after reordering each launch | by meshblock and direction octant 0.279 / 0.262 / 0.500; by remaining steps (oracle) 0.997 / 0.999 / 0.997 | compaction by remaining work is the lever; a coherence sort is secondary (§4.5) |
| warps whose photons spawn | 1.69–3.50 active threads per instruction against 4.8–6.9 lanes still marching; 10,500–15,700 warp instructions per warp-step against 4,150 (thin) and 5,154 (layer 3, which cannot spawn); lanes end 39–49 % of the warp span before the warp | spawning warps run a step in two to four groups and do not reconverge at the end of the march loop: a kernel exit every K steps is the reconvergence point; a loop that regenerates photons in place has none |
| registers | every large kernel at 255 registers, 8 warps per SM (12.5 %); `track!` spill code 1,704 / 2,568 B with the Compton sampler compiled in against 144 / 168 B without | the register cap is a measured knob (§2.5); the sampler leaves the march only when its share of the split is shown (§4.4) |
| residency | 650–750 of 864 warp slots in the bulk of a launch; 539 (thin) and 606 (Compton) on average over the phase | the cohort is refilled continuously, and a call has one tail, not one per chunk |
| overflow reruns | 1,002,090 photons tracked twice, 9.79 s of the 63.2 s Compton call | child slots come from a device budget (§3); a wavefront overflow reruns one launch (§4.2) |
| the two lanes of a device | their kernels never overlap: the lanes share the device twin's arrays and CUDA.jl synchronises a shared array with the stream that last used it; a pool program holding no shared array runs 8 chunks in 1.541 s on two lanes against 3.028 s on one | shared twins are exempt from that synchronisation under the rule of §2.3 |
| emission | 1.2–1.4 × 10⁶ primaries per second on the device, 0.7–0.8 s of the 3.74 s thin phase | emission runs in its own coherent launch (`MCStart`) and is the floor of the thin phase |
| precise march (Verner 9), disk B 500² | 1.6 s launch, 84 % of it with fewer than a quarter of the SMs active; slowest pixel 27,255 steps, 78 times the mean | the precise march is bound by one ray's serial latency; its levers are per-step cost and register allocation, not a schedule |
| first device compilation of a process | 8–13 s whatever the kernel, 1.3–1.7 s without ForwardDiff loaded | the extension precompiles the fixed-type programs (§2.6) |

The literature agrees on the coarse form: truncated histories with consolidation gain up
to 2.5× (Shift, multigroup), step caps 25–35 % on an A100 (Celeritas), a stage split at
material evaluation 36–221 % (Laine et al.); per-operation wavefronts are 2–7.7× slower
than a megakernel (Dr.Jit), and on CPUs history-based transport is 1.8–4× faster than
event-based in every study.

## 2. Execution model

### 2.1 One kernel, programs, executors

There is one kernel, `_foreach!`, one thread per item, no shared memory and no barrier.
Every device algorithm, the queues, the counting sort and the drains included, is a
program run by `_foreach!` or a device-protocol function whose CUDA method lives in the
extension. A program is an `isbits`-after-`Adapt` functor called as
`prog(item, i, outs...)` ([Architecture](@ref) §4.2).

### 2.2 Schedules

| executor | per-item products (`foreach_item!`) | Monte Carlo of a stationary plasma | Monte Carlo over an emission window; any other cascade |
|---|---|---|---|
| `ThreadedExecutor` | chunks pulled by one worker task per thread | cascade (§3), `track!` with the kick inline | cascade |
| `PoolExecutor` of `HostDevice`s | chunks pulled by the lanes | cascade, bitwise the threads | cascade |
| `PoolExecutor` of `CUDADevice`s | chunks pulled by the lanes, lanes overlapping (§2.3) | wavefront (§4), one per device | cascade with device-budget child slots (§3) |

The schedule is a property of the device and of the programs: `foreach_wavefront!` runs
the wavefront when every device of the pool answers `wavefront(dev) = true` (a
`CUDADevice`; the test device `Decks.WaveHost` also answers `true`) and the `Wavefront`
carries stage programs, and calls `foreach_cascade!` with its history programs
otherwise. The CPU or GPU is chosen only by the one `backend` option (keyword, then the
session default, then `BLP_BACKEND`, then `:auto`); no plan, configuration, keyword or
function name names a backend or a schedule. The budget, cohort, refill and sort
settings of §4.3 are keywords of the Execution primitive, set by tests and benchmarks.

Threads keep the history schedule because a CPU thread runs one photon at a time: there
are no SIMT lanes for a segment to recover, and the host's floating-point contraction
depends on the inlining context, so the fused `track!` is both the fast and the
bit-stable form on the host.

### 2.3 Lanes and shared twins

A device runs two lanes, each a task with its own stream. Their launches overlap when no
array of one launch waits on the other lane's stream. `share_lanes!(dev, a)` (device
protocol; identity on a `HostDevice`; on a `CUDADevice` it synchronises the stream that
wrote `a` and calls `CUDA.enable_synchronization!(a, false)`) exempts an array from
CUDA.jl's implicit cross-stream synchronisation. The rule:

1. It applies to residency twins, complete after a synchronised upload and never written;
   to accumulator twins (`ExactBins`, `ExactScalars`), written only by `Int64` atomics,
   whose sums commute; and to the child-buffer twins of plain `foreach_item!` programs,
   whose slots come from one atomic counter.
2. It never applies to lane-owned storage (`lane_buffer`, `lane_array`, `lane_stage`, the
   cascade's generations and the wavefront's queues): one stream uses them.
3. Every lane of the device synchronises before a shared array is freed, evicted from the
   residency cache, zeroed or overwritten, and before `gather!` reads an accumulator twin.

The wavefront uses the first lane of each device (§4.6); the cascade, the imaging and
the crossing programs use both.

### 2.4 Several devices

Lanes of every device pull from one chunk queue (per-item products and the cascade), or
every device draws primary blocks from one atomic block counter (the wavefront).
Children never leave the device that spawned them. Each device holds its own residency
copies and its own accumulator twins, folded into the host sums by exact integer
addition. Every device of one architecture runs the same compiled programs, so one GPU
and two GPUs give bitwise equal results. A two-GPU Slurm job holds the node's four CPUs,
as a one-GPU job does.

### 2.5 Launch traits

Per program type, each with the default that leaves the backend's choice: `workgroupsize`
(64 for IQUV imaging, 32 for the Monte Carlo programs), `chunk_hint`, `stack_bytes`
(32 KiB for the exact-Kerr programs) and `register_cap` (*gated*, §9: the most registers
per thread, honoured by the extension's launch, which compiles `_foreach!` with
`maxregs`). A register cap and a workgroup size reach only `ptxas` and the launch
configuration: the PTX of the program is the same, so no rounding changes.

### 2.6 Compilation

The extension's precompile workload compiles a trivial program and the programs whose
device types do not depend on a deck (`ZoneSums`, `ZoneCounts`, `HotcrossSigma`, `Drain`,
the host-output epilogue programs) for sm_80, so the first device compilation of a
process does not re-infer the GPU compiler; GPUCompiler's disk cache, enabled by the
`[GPUCompiler] disk_cache = "true"` preference of the GPU environment, then loads their
cubins. A cold cascade or wavefront compiles its next programs on a host task while the
first launches run.

### 2.7 Host outputs

A product's host epilogue (the band sums `I = ((0 + I₁) + I₂) + …`, the NaN maps of the
observables a radius-only call skips, the zero planes of a Stokes I image, the split of
the IQUV planes) runs as programs (`BandSum`, `Fill`, `Split`) on `ThreadedExecutor()`
over the host arrays, whatever the run's executor: the closed-form 2048² radius call
spends 0.27 s of its 0.305 s on the A100 allocating, first touching and filling 512 MiB of
host maps on one or two threads. Reductions whose order fixes their bits (the light curve
of `analytic_movie`, the flux of `sensitivity`, the deposits and the weighted composite of
`photon_ring_layers`, `zone_offsets`) stay serial on the host.

## 3. The cascade

`foreach_cascade!(exec, primary, layer, n, depth; capacity, chunk)` runs the primaries in
chunks and, inside each chunk, every scattering generation of their children on the
worker that took the chunk ([Architecture](@ref) §4.2). It is the schedule of threads and
host pools, and of devices for programs without stage programs.

- **Threads.** Each task keeps two growing child buffers and reruns only the item whose
  children overflowed, into discarded scratch.
- **Lanes.** The chunk's children live in two lane buffers swapped per generation; the
  only host read per generation is their count. A generation that can spawn records into
  zeroed lane scratch, drained into the device totals once its children fit.
- **Child slots.** A non-final generation of `m` parents starts with
  `max(MIN_CHILD_SLOTS, cld(m, 2), learned(dev), min(budget(dev), 16 m))` slots, where
  `budget(dev) = free_bytes(dev) ÷ 8 ÷ (2 × lanes × (sizeof(T) + sizeof(MCRng)))` and
  `learned(dev)` is the largest child count any lane of the device has recorded in the
  process (kept by the pool under its lock). On an 80 GB A100 with two lanes the budget is
  about 1.49 × 10⁷ slots per side and lane; at a chunk of 250,002 primaries the floor
  `16 m` is 4.0 × 10⁶ slots, 672 MB per side, against at most 1.93 × 10⁶ children of a
  generation of the torus deck. A generation whose children still overflow runs again in
  full with slots for its counted children and an eighth more; capacity never changes a
  bit.

## 4. The device wavefront

### 4.1 Stages and queues

A photon's life is cut into launches of stage programs that read and write device queues
of the first lane of each device:

| stage | program | reads | writes | accumulators |
|---|---|---|---|---|
| start | `MCStart` | a range of primary indices `g` | one `MarchState` per primary into B | `EMITTED`, one addend per primary |
| advance | `MCAdvance` | cohort A through the order `q` | survivors into B, finished photons into D, children into C | none |
| branch | `MCBranch` (inline kick) or `MCKick` (deferred kick, *gated*) | children C | one `MarchState` per child into B | none |
| retire | `MCRetire` | D | nothing | `record!` of the escapes, `tally!` of every photon |
| key | `MCKey` (*gated*) | A | one `UInt32` key per state | none |

Each stage is a program run by `_foreach!` as `prog(item, i)` without outputs. A launch
rebinds the queue placeholders (`WaveQueues`) inside the program's device twin to the
lane's buffers through the `WaveSwap` adaptor, as `CascadeSwap` rebinds the cascade's.
`Cohort(states, order)` is the read view (member `j` is `states[order[j]]`, or
`states[j]` without an order); `StateBuffer(states, count)` is the append queue,
`admit!(q, s)` taking a slot with one `Int64` atomic.

### 4.2 One iteration on a device

With `nA` live states in A:

1. **Sort** (*gated*, every `sort_every` iterations): `MCKey` writes a key per member and
   counts a histogram by integer atomics; one copy of the histogram to the host, an
   exclusive scan, one copy of the cursors back; `KeyPlace` writes `q[cursor[key]++] = j`.
   Without a sort `q` is the identity.
2. Zero the three counters (live, done, children).
3. **Advance** over `1:nA`: member `j` runs `march_steps` on `A[q[j]]` up to step
   `last = min(it − 1 + min(K, M), M)` (`M = max_steps(ctl)`, computed in `Int64`) and
   admits the state to B if it is still live, to D if it stopped; its children go to C
   through the sink.
4. Read the three counters back (one 24-byte copy). `nB > cap_B` or `nD > cap_D` is an
   error, checked on every iteration. If `nC > cap_C`: zero the counters, grow C to
   `nC + nC ÷ 8` slots (and B with it) and repeat step 3 on the unchanged A and `q`.
5. **Retire** over `1:nD`: `march_close`, i.e. `record!` and `tally!`.
6. **Branch** over `1:nC`: `march_open` of every child (after its kick in the deferred
   form), admitted to B.
7. **Refill**: while `nB < P` and primary blocks remain, take blocks from the shared
   counter and run **start** over their range, admitted to B.
8. Swap A and B. Stop when `nA = 0` and no block remains.

Before step 3, `cap_B ≥ nA + cap_C + R` and `cap_D ≥ nA`: every member yields one state,
the branch adds `nC ≤ cap_C` and the refill at most `cap_B − nB`, which the host chooses.
The advance writes nothing but its queues and counters, and its draws come from
per-photon streams, so a repeated advance yields the same survivors, retirees and
children: an overflow costs one launch (about 5–10 ms at P = 2^18), not a generation.
The child count of the call is the sum of the child counters of the advances that
succeeded, the number the cascade returns. The launches of steps 5–7 depend only on the
order of the lane's stream and return without waiting; the host waits at the counter
reads.

### 4.3 Budget, cohort, refill, end of run

| setting | default | rule |
|---|---|---|
| budget K | `WAVE_BUDGET = 32` steps per advance launch | doubles every iteration once no primary block remains and `nA < P/4`, up to `M` |
| cohort P | `WAVE_COHORT = 2^18` live states per device | about eleven waves of the 750 resident warps of the bulk |
| refill R | `WAVE_REFILL = 2^14` consecutive primaries per block | a device takes `cld(P − nB, R)` blocks at once from one counter shared by every device |
| sort | `sort_every = 0` (off) | *gated*, §4.5 |

Children join B in the iteration they are spawned, before any refill, so the cohort grows
during a burst of spawning and the refill pauses. Within an advance a photon that stops
partway idles for the rest of the launch, about `K / (2 L)` of its lane-steps for a mean
remaining life of `L` steps (3.5 % for thin photons, `L = 458`). Results do not depend on
K, P, R, the block order, the sort, the order inside a bucket or the workgroup size.

### 4.4 The scattering stage

- **Inline kick** (the default). The advance calls the sink's `branch!`, whose default
  runs the kick and `spawn!` inside the march exactly as `track!` does; the child queue
  holds `Superphoton` and `MCRng` (168 B) and `MCBranch` opens each child (its start
  sample and rates). The step body is that of `track!`, kick included.
- **Deferred kick** (*gated*). On a child buffer of `ScatterEvent`s, `branch!` stores the
  kick's inputs: the event point, `momentum(yev)`, the event site's
  `EmissionZone{Float64}`, the child weight, the depths at the step start, `nscatt + 1`,
  the parent's diagnostics and the child stream before the kick. `MCKick` runs the kick
  and the child's open in its own launch, so the four nested rejection loops of the
  Compton sampler leave the march. The decision `ok && fits && W > weight_min` and the
  parent's `spawned` and `unspawned` sums stay in the march.

The form follows the element type of the child buffer; threads always use the inline
kick. The event-site sample stays in the march in both forms: moving it out would rebuild
the parent's per-photon sums out of their step order.

### 4.5 The coherence sort (*gated*)

The key of a state at `x` with covariant momentum `p` is
`photon_key = (s × 8 + octant(x)) × 8 + octant(k)`: `s` the shell
`clamp(⌊(ln r − ln r_min) / 0.05⌋, 0, 127)` of the coordinate radius, then the octant of
the position, then the octant of the spatial wave vector `k^i = g^{iν} p_ν`; 8,192
buckets, computed from the state alone. By shell and direction octant the measured lane
efficiency is 0.277 (thin) and 0.478 (layers), by meshblock and direction octant 0.279 and
0.500. The sort moves indices, never states. Its target is the children, whose warps hold
27.3 meshblocks and load 3.6–9.3 sectors per request at 73–74 % L1 hits; the thin march
(L1 96 %, DRAM 0.3 % of peak) is not memory-bound.

### 4.6 Several devices and lanes

Each device runs one wavefront on its first lane; the device's second lane serves the
per-item programs and the cascade. All devices draw primary blocks from the one shared
counter, so they finish within one block of each other. Children stay on their device.
Each stage program has one device twin per device, with zeroed accumulator twins
registered on the host accumulators and folded by `gather!`.

### 4.7 What the wavefront serves

- `Wavefront(; start, advance, branch, retire, key, primary, layer)` holds the stage
  programs and the history programs of the same transport; the constructor checks that
  the stage programs agree on the state type and that the scattering depth of the run is
  the transport's `max_scatt`, so a photon of the last generation never spawns
  (`spawning = S && nscatt < max_scatt`).
- Runs over an emission window (`TimedPrimary`, `TimedLayer`) and generic cascade
  programs take the cascade on every executor.
- `foreach_wavefront!` returns the number of children spawned, as `foreach_cascade!`.

## 5. Data layout

States are arrays of `isbits` records. A state is loaded and stored once per advance,
688 B per K = 32 steps, against about 350–420 B of plasma per step, so a structure of
arrays is not used.

`MarchState`, 344 B: every value that crosses `track!`'s loop back edge, and the photon the
march started from:

| field | type | bytes | in `track!` |
|---|---|--:|---|
| `y` | `SVector{8,Float64}` | 64 | position and covariant momentum |
| `w`, `tau_a`, `tau_s` | `Float64` | 24 | weight, optical depths |
| `na0`, `ns0`, `th0` | `Float64` | 24 | rates and temperature at the step start; never re-sampled at a resume |
| `r_prev` | `Float64` | 8 | the radius of the previous control |
| `loss`, `loss_final`, `spawned`, `unspawned`, `absorbed` | `Float64` | 40 | the per-photon running sums, tallied once at retirement |
| `rng` | `MCRng` | 24 | only the draw counter advances |
| `nev` | `Int` | 8 | the event ordinal, continued across segments |
| `it`, `outcome` | `Int32` | 8 | the loop iteration (at most 100,001), the stop |
| `entry` | `Superphoton` | 144 | the photon as it entered the march |

Per device at the defaults (P = 2^18, R = 2^14, first child slots P):

| buffer | element | slots | bytes |
|---|---|--:|--:|
| A, B | `MarchState` | `P + cap_C + R` ≈ 5.4 × 10⁵ each | 2 × 186 MB |
| D | `MarchState` | as A, at least `nA` | 186 MB |
| C, inline kick | `Superphoton`, `MCRng` | `cap_C` (grows) | 44 MB |
| C, deferred kick | `ScatterEvent` (552 B), `MCRng` | `cap_C` | 151 MB |
| `q`, keys | `Int32`, `UInt32` | `cap_B` | 4.3 MB |
| histogram, cursors | `Int64` | 8,192 | 128 KB |
| counters | `Int64` | 3 | 24 B |

The cascade's child slots are `Superphoton` (144 B) and `MCRng` (24 B) in two arrays per
side; a plasma sample of the native blocks is 8 corner cells × 8 `Float32` primitives in
four 64-byte runs.

## 6. Invariants and why they hold

| # | invariant | why it holds |
|---|---|---|
| I1 | an item's work depends only on the item; `i` only addresses outputs | no program reads `i`, a thread index, a slot position, a chunk or a device in its arithmetic |
| I2 | chunk boundaries are a function of `n` and the chunk | `chunk_ranges`; the queue order is free |
| I3 | streams are counter-based Philox keyed by identity | `primary_rng(seed, g)`; `child_rng(parent, nev, layout)` puts the event ordinal into disjoint identity fields; a state carries its `rng` and `nev` across segments |
| I4 | cross-item sums are exact | `ExactBins` and `ExactScalars` add integer limbs with `Int64` atomics and round once on the host; no floating-point atomic and no float partial sum across photons exists |
| I5 | the set of children is deterministic, their storage order is not | `spawn!` stores by atomic arrival; every reader is order-free |
| I6 | an overflow counts every addition once | the cascade reruns into discarded scratch of the program's own type; the wavefront's advance writes no accumulator |
| I7 | inside a photon the arithmetic is sequential and its running sums are tallied once | the five sums live in `MarchState` and are tallied by `march_close` |
| I8 | a ledger addend's stripe may move, every read sums all stripes | start uses the primary index, retire the done slot; `ledger_sum` and `ledger_count` sum the 64 stripes exactly |
| I9 | host random decisions are serial and keyed | `zone_offsets` draws one stream in zone order on the host |
| I10 | every device of one architecture runs the same compiled programs | one GPU against two is bitwise; CPU and GPU contract differently |
| I11 | the executor is chosen only through `resolve` | §2.2 |
| I12 | a photon's life is cut only between whole calls and at `track!`'s loop back edge | every loop-carried value crosses as an exact `Float64` or integer; only pure functions of constants (`spawning`, the rate frame, `Cgr`) are recomputed; no carried value ends in a contract-flagged multiply (`y` ends in the plain `y + (h/6) acc`, the weight, depths and sums in plain operations), so no fusion spans the cut |
| I13 | the compiled arithmetic of a photon is that of `track!` on each backend | on the host `track!` is one non-inlined function whose body inlines `march_open`, `march_steps` and `march_close`, the statements of the uncut loop in their order; on the device every multiply of the Monte Carlo PTX is rounded (`.rn`) or fused into an `fma`, so an enclosing kernel can decide only the association of `reassoc` reductions; the stage programs are device functions of their own (no floating-point operation in a kernel entry), and the offline gate of §8.3 compares the floating-point instruction multiset of `MCAdvance`'s loop with `track!`'s |
| I14 | shared twins are only read, or written by integer atomics | §2.3 |
| I15 | a register cap or a workgroup size changes no rounding | it reaches only `ptxas` and the launch; the MC PTX holds no unrounded multiply or fma, and its few unrounded adds have no multiply operand, so `ptxas` has nothing to contract |

What follows for each comparison of the protocol (§8.1):

- **CPU, 1 against 16 threads.** The threaded cascade's chunks depend on `n` and the chunk,
  the totals are exact and an overflow reruns one item: bitwise.
- **CPU against the reference tree.** Threads run `track!` as composed in I13; bitwise is
  checked on every deck, the window decks included, because the host's contraction follows
  the inlining context and an equal source is no proof.
- **GPU against the reference tree.** The cascade's knobs (slots, overlap, cap, workgroup)
  change no compiled arithmetic (I14, I15). The wavefront runs the statements of `track!`
  split at the back edge (I12) in stage programs (I13); its bits are checked at budgets
  K ∈ {1, 7, 32, ∞}, a dependence on K exposing a fusion across the cut.
- **One GPU against two.** I1–I5, I10: bitwise.
- **CPU against GPU.** Equal counts and integer parts in every cell; on each deck the
  largest difference no larger than the reference tree's own CPU–GPU difference, recorded
  per deck with the reference (from 1.35e-16 to 1.18e-13 of the largest magnitude on the
  Monte Carlo decks, 3.1e-13 thin and 6.1e-10 Compton per cell at stride 1, up to 3.15e-13
  on the torus images, 9.03e-11 on the precise march and 2.31e-9 on the closed-form
  disk).
- **Host pools.** A pool of `HostDevice`s runs the cascade with the compiled programs of
  threads and is bitwise the threads; a `WaveHost` pool runs the stage programs on the
  host, whose contexts differ from `track!`'s, and agrees with threads within 1e-11.

Class of a change (decides its verification):

- **A, outside the compiled body**: visiting order, chunking, regrouping, compaction,
  lanes, devices, workgroup size, register allocation, child capacity, where an output
  lives, when a kernel compiles. Bitwise by I1–I11, I14, I15; the decks confirm it.
- **B, inside the body, exact operation for operation**: moving an inlining boundary,
  splitting a loop, a new enclosing program. StaticArrays `dot` and `M*v`, `muladd` and
  ForwardDiff partials carry `reassoc` or `contract`, which LLVM treats by context on the
  host; bitwise only if the decks on both backends say so.
- **C, through a re-baseline**: a change that is known to move bits on a backend (BLP-owned
  ports of the vendor library's elementary functions on the device; explicit `fma`
  reductions on the host). Outside this standard until the reference is re-taken.

## 7. Adding a program

### 7.1 A per-item program

1. A functor whose fields are what the item needs; arrays become device twins by
   `Adapt` (inputs uploaded once per device by identity, behind `readonly`).
2. `prog(item, i, outs...)` computes from `item` alone and writes `outs[k][i]`, or writes
   through exact accumulators when it has no outputs. No floating-point atomics, no read
   of `i` in arithmetic, no partial float sum across items.
3. Launch traits only when measured: `workgroupsize`, `chunk_hint`, `stack_bytes`,
   `register_cap`.
4. Tests: `device_isbits` of the device twin, no allocation on the host, bitwise equality
   of threads, `:host_pool` and `Decks.hosts(1, 2, 3, 5)`; under `BLP_TEST_DEVICES=cuda`
   one and two GPUs bitwise each other.
5. A case in `bench/kernel_audit.jl` for its registers, stack and spills, and its entry in
   the offline device-code gate (§8.3).

### 7.2 A Monte Carlo transport

1. The march description `March` (control, spacetime, provider, opacities, sink, the
   absorption and scattering switches) and the three `@inline` pieces `march_open`,
   `march_steps`, `march_close` over `MarchState`; `track!` is their composition.
2. A sink implementing the `Sink` protocol (`record!`, `tally!`, `child_stream`,
   `spawn!`, `branch!`), whose child slot is a `ChildBuffer` of the child element type.
3. The history programs (`MCPrimary`, `MCLayer` or their equivalents): the CPU path and
   the oracle of the bit tests.
4. The stage programs built from the same transport and sink, and the `Wavefront` that
   holds both sets.
5. Streams: a primary's from `primary_rng(seed, g)`, a child's from `child_stream` with
   the parent's ordinal `nev`; no other source of randomness.

### 7.3 Code that must give the same bits in another context

A function whose bits must not depend on where it is inlined, or that must give the
host's bits on a device, is written in portable arithmetic: `+ - * /`, `sqrt`, explicit
`fma`, `min`, `max`, `abs`, `copysign`, conversions, comparisons, integer arithmetic and
BLP-owned ports of elementary functions (`tang_exp`, `tang_log`); no `muladd`, no
StaticArrays reduction or product, no `^` with a non-literal exponent, no `@fastmath`, no
call into a non-inlined Base or dependency math specialisation. The zone-table and
hot-cross programs are written this way and are bitwise host = device. Converting existing
physics to this form moves bits (class C).

## 8. Measuring

### 8.1 Bits

- **Reference.** A detached worktree at the base commit with its own environment (its
  `BlackLightPlus` path is that worktree, never a moving checkout); its deck outputs on 1
  and 16 CPU threads, on one A100 and on two A100s are recorded once per base.
- **Decks.** The thirteen fixed decks (Monte Carlo torus thin and Compton at stride 8, thin
  at stride 1, the thin and the scattering sphere, the marched disks A and B precise and
  production, the Kerr disk and its radius path, torus Stokes I and IQUV) and the window
  decks (the torus at stride 8 and the scattering sphere over an emission window), each
  output serialised and compared leaf by leaf: bitwise, or the largest difference over the
  deck's largest magnitude with the integer parts compared.
- **Photon records.** For a Monte Carlo deck, every photon's end state (`x`, `p`, `w`,
  outcome, `nev`, steps), ledger values and children, keyed by stream identity and sorted,
  so a difference is located to one photon and one step.
- **Stride-1 bins.** The thin and Compton photon phases of the timing deck, every cell of
  every order, the ledger totals and the child count.

### 8.2 Time

- **GPU.** One Slurm job per comparison on partition `preempt` (one A100, `-c 4`,
  `julia -t 4`, `JULIA_PKG_OFFLINE=true`, the CUDA runtime pinned to 13.3 in
  `LocalPreferences.toml`), the job alone on the node. Processes in the order reference,
  branch, reference; each loads the snapshot, builds the sites and tables, compiles each
  phase once and times three warm calls of each photon phase around `run_layers`. Report
  every call, the best and the median; the bins of every call equal the reference's. The
  SM clock (`nvidia-smi -q -d CLOCK,PERFORMANCE`) is read at the start and the end. A
  preempted job is rerun whole, never merged.
- **CPU.** 32 threads on the shared host, the trees interleaved reference, branch,
  reference, branch, best of three and median, the load average before and after each
  run. "No slower" is best(new) ≤ 1.03 × best(reference); a comparison during which the
  load average exceeds 100 or moves by more than 30 is repeated.

### 8.3 Profiles and device code

- **Nsight Compute** (`ncu --set full`, application clocks, kernel replay, one call
  between `cuProfilerStart` and `cuProfilerStop` on a pool of one lane):
  `smsp__thread_inst_executed_per_inst_executed.ratio` (active threads per instruction),
  `smsp__sass_average_branch_targets_threads_uniform.pct`, sectors per global load
  request, L1 and L2 hit rates, local-memory traffic, the FP64 pipe, issue activity,
  achieved occupancy and the warp stall reasons. The Compton kernels are profiled at
  Ns = 1.6 × 10⁵: replay costs about 290 times a kernel's duration.
- **Nsight Systems** (`--trace=cuda,nvtx --gpu-metrics-devices=all
  --gpu-metrics-frequency=2000`): kernel union against the sum (lane overlap), host gaps
  between launches, SMs active and warps in flight over the call, module-load intervals
  (compilation).
- **Photon probe.** Wrappers of the march control and the sink that record each photon's
  steps, events, outcome, start and end `%globaltimer`, `%smid` and stream identity; the
  lane efficiency E, residency and warp spans follow from them. The probe's bins equal the
  plain run's.
- **Offline device code** (no GPU): `bench/kernel_audit.jl` compiles the device twin of
  every program kind for sm_80 and reports registers, stack and spills against its
  baseline (`--check` fails on growth). The device-code gate checks, from the PTX that
  `CUDA.@device_code` dumps: no floating-point operation in a kernel entry; no unrounded
  `mul.f64` or `fma.f64` in the Monte Carlo programs; per function, the counts of `DFMA`,
  `DMUL` and `DADD` with their rounding and of `contract` and `reassoc` instructions,
  against the reference tree.

## 9. Gated components

| component | gate | adopted when |
|---|---|---|
| wavefront schedule | the advance at K ∈ {1, 7, 32, ∞} bitwise the reference on the A100 (photon records and decks); the CPU decks bitwise with `track!` composed of the pieces | both hold; otherwise the cascade remains the device schedule |
| deferred kick | the kick probe (`weight_min = Inf`, the kick never executed, same draws) brings the Compton primaries below 8,000 warp instructions per warp-step (from 15,665); the Compton decks bitwise the reference on the A100 | both hold |
| coherence sort | the Compton photon phase at least 3 % faster with the sort on, every setting bitwise | it holds |
| register cap | a phase at least 5 % faster at the cap, decks bitwise, PTX unchanged | it holds, per program type |

## 10. Not part of the standard, and why

- **A single persistent loop that regenerates photons.** It has no exit at which a warp
  reconverges, and spawning warps measurably do not reconverge at the end of a march;
  split groups would accumulate over the launch. A warp vote at the loop head would hang
  on partial warps and cannot be exercised on host pools.
- **Per-operation stages and a step split between geodesic and plasma.** 2–7.7× slower
  than a megakernel (Dr.Jit), 72 B of state per step, and a step-internal cut whose bits
  depend on context.
- **The event-site sample outside the march.** It rebuilds the parent's `spawned` and
  `unspawned` sums out of their step order (I7).
- **A wavefront on threads.** No SIMT lanes to recover; event-based transport is 1.8–4×
  slower on CPUs; the host's contraction would move bits.
- **Rerunning only the parents whose children overflowed.** The wavefront's advance
  reruns one side-effect-free launch, and the cascade's device-budget slots make an
  overflow rare.
- **A pilot march that predicts step counts for the order key.** A second march control to
  maintain; the shell of `photon_key` orders the long-lived inner photons first.
- **A Matter protocol function for the sort key.** `photon_key` from the state alone is as
  coherent (0.277 / 0.478 against 0.279 / 0.500 by meshblock).
- **An untracked read-only array type in the extension.** `share_lanes!` uses CUDA.jl's
  public `enable_synchronization!`.
- **Device-resident result arrays.** A result type that depends on the resolved executor,
  for at most 0.2 s on one closed-form call; the threaded host epilogues of §2.7 take most
  of that host work.
- **The site scan on a device, ports of the vendor library's functions, explicit-`fma`
  reductions in the physics, and host continuation of the slowest device rays.** Each
  moves bits on a backend (class C) and waits for a re-baseline.
