# Wavefront execution of the Monte Carlo on devices

Design of an event-based ("wavefront") schedule for the BlackLightPlus Monte Carlo photon
phase on GPUs: each photon life is cut into stage launches fed by device queues (a march
of at most K steps, the scattering children, the retirements), with compaction, an
optional coherence sort and refill until the queues are empty. It is one new
execution-layer primitive, `foreach_wavefront!`, next to `foreach_cascade!`. Tree:
`a5923dd` (worktree `/data2/haiyangw/scratch/blp_wavefront`, branch `gpu-wavefront`);
paths are relative to the repository root. Measurements are quoted from
`plan/profile.md` (§ numbers given as P§) and the maps `plan/map_*.md`; every number not
measured is marked **(estimate)**.

## 0. Summary

- **What it fixes.** On the A100 the thin photon phase runs at a lane efficiency of
  0.201 and the Compton primaries at 0.187 (P§4.2): a warp holds its 255-register slot
  until its slowest photon ends. In the launches whose photons spawn, the warp also issues
  each step in 2–4 groups (1.69–3.50 active threads per instruction against 4.8–6.9 lanes
  still marching, P§4.3, §7), lanes end 39–49 % of the warp span before the warp
  (P§4.3), and 14 % of the Compton warp-seconds are overflow reruns (P§4.4–4.5). A march
  of at most K steps per launch, with survivors compacted into the next launch, bounds the
  step spread of a warp to K steps and forces the warp to reconverge at every launch
  boundary. A child queue whose overflow reruns only that one march launch replaces the
  rerun of whole generations.
- **What it leaves alone.** Each photon's sequence of floating-point operations, its
  Philox stream and every addend of the exact accumulators. The CPU keeps the history
  form, which is today's cascade with today's compiled `track!` (§2.7).
- **Expected effect (estimate, §6).** Thin photon phase 3.74 s → about 1.2 s on one A100
  (3.2×). Compton photon phase 62.5 s → 6.8–12.3 s (5–9×), about 8.8 s with the kick
  inside the march and about 6.8 s with the kick in its own stage. The CPU stays at
  11.3 s and 62.5 s on 32 threads. For thin runs the floor is the critical path, the
  20,000-step photons at about 25 µs per step alone (0.45 s), which no schedule removes.
- **What would falsify it early (§9).** If the `track!` split changes CPU bits (day 2),
  if the resumed march changes device bits or depends on K (day 4–5), if a replay of the
  recorded step counts predicts < 1.5× for thin (day 1, no code), or if the
  kick-isolation probe shows the kick is not the source of the divergence (day 1, one GPU
  job).

## 1. Evidence the design rests on

| fact | measurement | consequence for the design |
|---|---|---|
| step-count tail sets the SIMT efficiency of thin runs | E = 0.201 thin, 6.37 active threads per instruction; branches 98.7 % uniform inside a step (P§2.2, §4.2) | bounding the steps per launch recovers up to 1/E ≈ 5× lane-steps |
| sorting alone is a small lever | by meshblock, then octant: E 0.279 thin, 0.262 Compton primaries, 0.500 layers; the oracle (by steps) gives 0.99 (P§4.2) | compaction by remaining work (segments) is the main tool; the sort is secondary |
| spawning launches issue steps in groups | Compton primary 15,665 instructions per warp-step against 4,150 thin and 5,154 for the non-spawning layer 3; the ratio of lanes still marching to active threads is 3.6, 2.2, 2.0 for the primary and layers 1, 2 against 1.05–1.1 without spawning (P§4.3) | a launch boundary every K steps reconverges the warp; the kick's rejection loops are the prime suspect (W0b) |
| no reconvergence at the loop exit in spawning launches | lanes end 39–49 % of the warp span before the warp ends; in non-spawning launches within 0.01 % (P§4.3) | same |
| overflow reruns | 1,002,090 rerun photons, 9.79 s of the 63.2 s Compton call, 14.1 % of its warp-seconds (P§4.4–4.5) | a child queue whose overflow reruns one march launch, never a generation |
| occupancy is register-capped | 255 registers in every large kernel, 8 warps per SM, residency 650–750 of 864 warps in the bulk of a launch (P§2.1, §4.3) | the design does not change occupancy; it changes what the resident warps do |
| thin march is not memory-bound | L1 hit 96 %, DRAM 0.3 % of peak, long-scoreboard 0.23 per issue (P§2.2) | no sort gain expected for thin |
| layer photons are memory-incoherent | 27.3 meshblocks per warp; layers 2, 3: 3.6 and 9.3 sectors per request, L1 hit 73–74 %, long scoreboard 0.94 and 2.11 (P§2.2, §4.2) | the coherence sort targets the children |
| two lanes of a device do not overlap | shared program-twin arrays serialise the lanes' launches (P§5.3) | one wavefront per device, driven by its first lane |
| launch tails | thin: 0.11–0.25 s per launch below half the warp slots, 0.78 s of 3.76 s (P§4.3, §4.4) | one end-of-run tail instead of one per chunk |
| one A100 thin warp-step | 30 µs per warp-step at 8 warps per SM, 7.7 cycles per warp instruction (P§4.3) | the cost unit of §6 |

Literature (`map_literature.md` §2, §6): event-based execution wins on GPUs only in coarse
form and loses 1.8–4× on CPUs. Truncated histories with consolidation give up to 2.5×
(Shift multigroup) and step caps 25–35 % (Celeritas on an A100). Splitting at whole calls
is the safe cut; per-operation wavefronts (Dr.Jit) are 2–7.7× slower.

## 2. Architecture

### 2.1 Stages and queues

```
             refill blocks of primaries (atomic block counter shared by all devices)
                                  │
                                  ▼
   ┌──────────┐  start    ┌──────────────┐   advance (≤ K steps)   ┌──────────────┐
   │ primaries│ ────────► │ next cohort B │ ◄───── survivors ────── │  cohort A    │
   └──────────┘           └──────────────┘                          └──────┬───────┘
                                  ▲                                         │ advance
                                  │ branch (open, or kick + open)           ├──► done D ──► retire ──► ExactBins, ledger
                                  └──────────── children C ◄────────────────┘
                     [sort: keys of A → counting sort → order q in which advance reads A]
```

| stage | program (Products) | reads | writes | accumulator side effects |
|---|---|---|---|---|
| start | `MCStart` | primary items `g` (a range) | one `MarchState` per item into B | `EMITTED` (one addend per primary) |
| advance | `MCAdvance` | cohort A through the order q | survivors into B, retirees into D, children into C | none |
| branch | `MCBranch` (inline kick) or `MCKick` (deferred kick) | children C | one `MarchState` per child into B | none |
| retire | `MCRetire` | D | nothing | `record!` (escapes), `tally!` (every photon) |
| key (optional) | `MCKey` | A | one `UInt32` key per state | none |

Every stage is a program run by the one kernel `_foreach!` (`src/infra/Execution/kernel.jl:7-10`),
called as `prog(item, i)` without outputs, as the cascade programs are. The queues are
lane buffers of the device's first lane. Each launch rebinds the program's placeholders
to them inside the program's device twin, the mechanism `CascadeSwap` uses
(`src/infra/Execution/cascade.jl:115-154`).

### 2.2 One iteration of a device wavefront

On the first lane of each device, with `nA` live states in A:

1. **Sort** (every `M` iterations, when a key program is given): `MCKey` over A writes
   `key[j]` and counts `hist[key]` (integer atomics). One device-to-host copy of
   `hist`, an exclusive scan on the host, one host-to-device copy of the cursors, then
   `KeyPlace` over A writes `q[cursor[key[j]]++] = j`. Order inside a bucket is free.
   Without a sort, `q` is the identity.
2. Reset the three counters (live, done, children): one host-to-device copy of three
   `Int64`.
3. **advance** over `1:nA`: member `j` reads `A[q[j]]`, runs `march_steps` until it
   stops or has taken `K` steps, and appends the state to B (live) or D (stopped).
   Children go to C through the sink.
4. Read the three counters back (one device-to-host copy of three `Int64`). If the child
   counter exceeds C's slots, discard the launch's outputs, grow C to `n + n ÷ 8` slots
   (and B by the same amount), and go to 2. A is untouched by step 3, so the rerun sees
   the same inputs (§5.4).
5. **retire** over `1:nD`: `march_close` of `D[j]`, i.e. `record!` and `tally!`.
6. **branch** over `1:nC`: `march_open` of every child (after the kick in the deferred
   form), appended to B.
7. **Refill**: while `nB < P` and primary blocks remain, take blocks from the shared
   counter and run **start** over their range, appended to B.
8. Swap A and B; `nA = nB`. Stop when `nA == 0` and no block remains.

Host round trips per iteration: two copies of 24 bytes, plus two of `8 × n_regions`
words in a sorting iteration. Launches: three or four, plus three with a sort. Today
every launch synchronises (`kernel.jl:42`); the launches of steps 5–7 depend only on stream
order, so W8 adds a launch that does not wait.

### 2.3 Budget, cohort size, refill, end of run

- **Budget K** (steps per advance launch): `WAVE_BUDGET = 32` to start, tuned in W8. A
  lane does at most K steps per launch. A photon that stops partway through idles for the
  rest of that launch, so a cohort loses `K/(2 L̄)` of its lane-steps, `L̄` being the mean
  remaining life (3.5 % for thin at `L̄ = 458`) **(estimate)**. In the tail (no primaries
  left and `nA < P/4`) K doubles every iteration up to `max_steps(ctl)`, so the last
  photons run with few relaunches. Results do not depend on K (§5.5).
- **Cohort P** (target live states per device): `WAVE_COHORT = 2^18`, about 11 waves of
  the 750 resident warps × 32 lanes measured in the bulk (P§4.3). That leaves under 5 %
  of a launch in its last partial wave **(estimate)**. A smaller cohort lets the
  long-lived photons advance faster during the bulk, at the price of more launch tail
  (§6.1); tuned in W8.
- **Refill blocks**: `WAVE_REFILL = 2^14` consecutive primaries, taken with
  `Threads.atomic_add!` on one block counter shared by every device of the pool. A device
  takes `cld(P − nB, R)` blocks at once, so the gid range is contiguous and one start
  launch covers it.
- **Children have priority**: all children of an iteration join B in the same
  iteration (B is sized for them in advance), and refill fills only the remaining deficit.
  The pool therefore grows during a burst of spawning, and the refill pauses.

### 2.4 The coherence sort

Key = `region(provider, x) × 8 + octant(k^i)`. Here `region` is a new Matter protocol
function: the meshblock from `locate` for `FluidBlocks`, 0 by default. The octant is that
of the spatial wave vector `k^i = g^{iν} p_ν`. The torus has 344 meshblocks, so 2,752
buckets. Counting sort as in §2.2, with integer atomics through `_foreach!`, and the scan
on the host, because a program has no shared memory (`map_execution.md` B3). The sort
moves indices only; states stay in place, as in Shift and OpenMC. Expected value: the
layer photons (gen 2, 3 memory-incoherent, P§2.2), not thin (§6.2). The sort is optional
and stays only if W7 measures a gain.

### 2.5 The scattering stage in two forms

- **Inline kick** (W4). The advance program calls the sink protocol's `branch!`, whose
  default runs today's kick and `spawn!` inside the march (`track.jl:237-243`). Children
  are `Superphoton` + `MCRng` (168 B), as in the cascade's child buffer. The branch
  program `MCBranch` opens them (start sample and rates, `track.jl:283-286`). The step
  body is identical to today's `track!` loop, including the kick, so this form carries
  the least bit risk.
- **Deferred kick** (W6). `branch!` on a sink whose child buffer holds `ScatterEvent`s
  stores the kick's inputs: the event point, `momentum(yev)`, the event site's
  `EmissionZone{Float64}`, the child weight, the depths at the step start, `nscatt + 1`
  and the diagnostics. The child stream goes in the buffer's stream array. `MCKick` runs
  the kick and the child's open in its own launch, so the rejection loops of
  `sample_electron`, `sample_y`, `_gamma_ge1` and the Klein–Nishina loop
  (`scatter.jl`, `map_mcprograms.md` §4) leave the march. GPUmonty uses the same cut
  (`kernels.cu:928`). The decision `ok && fits && W > weight_min` and the sums
  `spawned`/`unspawned` stay in the march (`track.jl:236, 244-246`), so the parent's
  per-photon sums keep their order (I7).

The form follows the element type of the child buffer; the event-site sample stays in the
march in both forms. Moving it out (cut E of `map_mcprograms.md` §8.3) would rebuild the
parent's `spawned`/`unspawned` sums out of order, and this design does not do it.

### 2.6 Several devices

`foreach_device` (`src/infra/Execution/pool.jl:258-266`) runs one wavefront per device
on its first lane. All devices draw primary blocks from the one shared counter, so they
balance to within one block at the end. Children never leave the device that spawned
them. Each program has one device twin per device (`ProgramTwins`,
`residency.jl:88-105`), with zeroed accumulator twins registered on the host
accumulators, folded by `gather!` (`accumulators.jl:209-232, 450-462`) as today. The
device's second lane stays idle during the photon phase: the two lanes of a device
serialise their launches (P§5.3), and two wavefronts on one device would share its
throughput and double the tail.

### 2.7 The CPU path

The CPU does **not** run the stage structure. `foreach_wavefront!` on a
`ThreadedExecutor` calls `foreach_cascade!(exec, wf.primary, wf.layer, n, depth; capacity)`:
today's depth-first chunk cascade of `MCPrimary`/`MCLayer` (`cascade.jl:191-264`), whose
`track!` keeps the kick inline. Reasons:

1. A CPU thread runs one photon at a time, so it has no SIMT lanes whose efficiency a
   segment could recover.
2. The literature: history-based execution is 1.8–4× faster on CPUs in every study
   (`map_literature.md` §2.1).
3. The host's floating-point contraction depends on the inlining context. The Compton
   sampler behind a non-inlined call changed the CPU bits (`docs/src/design/performance.md`
   §4), the Verner stage loop changed 1,355 of 20,000 steps (speed README, Findings), and
   the host fuses a contract pair only where instruction selection sees it
   (`map_compiler.md` §2–3). A deferred kick on the CPU would almost surely move CPU bits.
4. Pools of `HostDevice`s stay bitwise with threads only because they run the same
   compiled `track!`. That is the bitwise test of
   `test/pipeline/products_montecarlo.jl:259-281`.

The CPU shares the physics. `track!` becomes a composition of the three `@inline`
functions the device stages call (`march_open`, `march_steps`, `march_close`, §3.1),
compiled inside the one non-inlined `track!` as today. The kick goes through the
`branch!` hook whose default is today's code. So the CPU path is today's IR, and W1
checks that bitwise. `HostDevice` pools likewise keep the cascade, through the device
trait `wavefront(dev)` (§3.3). The test device `Decks.WaveHost` runs the wavefront on the
host for coverage without a GPU (§3.6).

An optional CPU gain (W-CPU, measurement first): sorting each chunk generation's children
by `region` before the layer program runs (`cascade.jl:219`). The result is bitwise by I5.
It is adopted only if the Compton phase on 32 threads gets at least 3 % faster.

## 3. Exact changes

### 3.1 MonteCarlo (physics; no KernelAbstractions, no threads)

`src/physics/MonteCarlo/track.jl` (342 lines now). Either it grows to about 440 lines, or
the new pieces go to a new file `march_state.jl` included by `MonteCarlo.jl` before
`track.jl`.

| name | kind | content |
|---|---|---|
| `March{C,S,P,O,K,A,B}` | struct (exists, `track.jl:204-210`) | gains `absorb::A`, `scatter::B` (the `Val`s of `track!`); exported, documented |
| `MarchState` | `isbits` struct, 344 B | the loop-carried state of `track.jl:281-291` (layout in §4) |
| `march_open(m, sph, rng) -> MarchState` | `@inline` | `track.jl:279-291`: start sample at `sph.x`, `na0`, `ns0`, `th0`, `r_prev = Inf`, `nev = 0`, the five sums at 0, `it = 1`, `outcome = 0`, `entry = sph` |
| `march_steps(m, s, last) -> MarchState` | `@inline` | the loop `track.jl:292-334` over `it in Int(s.it):last`, every loop-carried value loaded from `s`; `spawning`, `frame`, `Cgr` recomputed from `m` and `s.entry` (pure functions of constants); a stop or plunge sets `outcome` as now; finishing the loop with `last == max_steps(ctl)` gives `DROPPED_MARCH`, finishing it earlier leaves `outcome = 0` and `it = last + 1` |
| `march_close(m, s) -> (sph, recorded, rng)` | `@inline` | `track.jl:335-341`: the output photon, `record!` on escape, `tally!(sink, s.entry, MarchLedger(...))` |
| `track!(ctl, st, provider, opac, sph, rng, sink; absorb, scatter)` | function (not inlined, as now) | `m = March(...)`; `march_close(m, march_steps(m, march_open(m, sph, rng), max_steps(ctl)))` |
| `ScatterEvent` | `isbits` struct, 552 B | `x`, `p` (= `momentum(yev)`), `zone::EmissionZone{Float64}`, `W`, `tau_a`, `tau_s`, `nscatt::Int32`, `x1i`, `x2i`, `ne0`, `thetae0`, `b0` |
| `kick(ev, crng) -> (Superphoton, MCRng)` | `@inline` | `track.jl:237-242`: `scatter_superphoton(ev.p, ev.zone, crng)` and the child `Superphoton` |
| `branch!(sink, ev, crng)` | sink protocol, `sinks.jl` | default `@inline branch!(s::Sink, ev, crng) = spawn!(s, kick(ev, crng)...)` |
| `step_events` | edit (`track.jl:236-243`) | the three lines of kick, child and `spawn!` become `branch!(m.sink, ScatterEvent(xev, momentum(yev), zone, W, stp.tau_a, stp.tau_s, sph.nscatt + Int32(1), sph.x1i, …), crng)`; the condition and the `spawned`/`unspawned` sums are unchanged |

Exports added to `MonteCarlo.jl:31-58`, each documented: `March`, `MarchState`, `march_open`,
`march_steps`, `march_close`, `ScatterEvent`, `kick`, `branch!`.

### 3.2 Matter and Data (W7 only)

- `src/physics/Matter/protocol.jl`: `region(p::PlasmaProvider, x) -> Int`, default `0`;
  documented and exported from Matter.
- `src/infra/Data/fluid_blocks.jl`: `region(b::FluidBlocks, x::SVector{4}) =
  locate(b, x[2], x[3], x[4])` (`fluid_blocks.jl:240`; one `Int32` read of the leaf
  table). `UniformSphere` and analytic providers keep the default.

### 3.3 Execution

New file `src/infra/Execution/wavefront.jl` (about 380 lines), included after
`cascade.jl` in `Execution.jl:18-29`:

| name | kind | content |
|---|---|---|
| `wavefront(dev::AbstractDevice) -> Bool` | device protocol, in `devices.jl` | `false` for `HostDevice` and the fallback, `true` for `CUDADevice` (both handles live in `src`; no `CUDA.` needed); documented |
| `Cohort{S,O}(states, order)` | `isbits` after Adapt | read view: member `j` is `states[order[j]]`, or `states[j]` when `order === nothing` |
| `StateBuffer{S,C}(states, count)` | `isbits` after Adapt | append queue: `admit!(q, s)`, one `Int64` atomic, store if the slot exists (by construction it always does, §5.4) |
| `WaveQueues{C,L,D}(cohort, live, done, budget::Int)` | placeholder / launch view | what a stage program reads and writes; `WaveQueues(T)` is the empty host placeholder of state type `T` |
| `Wavefront(; start, advance, branch, retire, key = nothing, primary, layer)` | struct | the stage programs and the history-form programs; checks that the placeholders agree on the state type |
| `foreach_wavefront!(exec, wf, n, depth; capacity = nothing, budget = WAVE_BUDGET, cohort = WAVE_COHORT, refill = WAVE_REFILL, sort_every = 0) -> Int` | entry point | the `ThreadedExecutor` method calls `foreach_cascade!(exec, wf.primary, wf.layer, n, depth; capacity)`; a `PoolExecutor` whose devices all answer `wavefront` runs §2.2 per device; any other pool calls `foreach_cascade!`; returns the children received |
| `WaveSwap` | Adapt adaptor | per launch: `WaveQueues` → the lane's device views, `ChildBuffer` → the lane's child queue, `Generation` → the children of the iteration (cf. `CascadeSwap`) |
| `run_wavefront!(lane, wf, twins, blocks, next, caps, opts)` | driver | the loop of §2.2; child slots per device carried across iterations and calls as `caps` (cf. `cascade.jl:273, 298-299, 327`) |
| `KeyCount`, `KeyPlace` | programs (W7) | the histogram and the placement of the counting sort |
| `WAVE_BUDGET = 32`, `WAVE_COHORT = 2^18`, `WAVE_REFILL = 2^14` | constants | documented defaults; results never depend on them |

Other files:
- `kernel.jl:33-43`: `launch!(…; wait = true)`. With `wait = false` the launch returns
  without `KernelAbstractions.synchronize` (W8). Stream order on the lane serialises the
  kernels.
- `Execution.jl:32-52`: export `foreach_wavefront!`, `Wavefront`, `WaveQueues`, `Cohort`,
  `StateBuffer`, `admit!`, `wavefront` (docstrings required, `test/structure.jl:121-127`).
  Nothing new at top level (54 of 54, `structure.jl:115`).
- `cascade.jl`, `children.jl`, `pool.jl`, `accumulators.jl`: unchanged. `ChildBuffer` and
  `Generation` serve as the child queue and its read view; `foreach_device` is the driver's
  host fan-out.

### 3.4 Products

New file `src/pipeline/Products/mc_stages.jl` (about 260 lines; `programs.jl` is at 533
of 600), included after `programs.jl` (`Products.jl:45`):

| program | call `(item, i)` | holds |
|---|---|---|
| `MCStart(source, transport, sink, seed, queues)` | `rng = primary_rng(seed, g)`; `sph, rng = emit(source, g, rng)`; `add_ledger!(at_stripe(sink, g), EMITTED, sph.w * sph.E)`; `admit!(queues.live, march_open(m, sph, rng))` | the run's `MCSink` (ledger for `EMITTED`) |
| `MCAdvance(transport, sink, queues)` | `s = queues.cohort[j]`; `s = march_steps(m, s, min(Int(s.it) + queues.budget − 1, max_steps(ctl)))`; `admit!(s.outcome == 0 ? queues.live : queues.done, s)` | the march sink `MCSink(ctl, (), slot, nothing)`: child stream layout and child buffer only, no accumulators, so no accumulator twins |
| `MCBranch(generation, transport, sink, queues)` | `admit!(queues.live, march_open(m, photons[j], rngs[j]))` | the march sink |
| `MCKick(generation, transport, sink, queues)` | `child, crng = kick(events[j], rngs[j])`; `admit!(queues.live, march_open(m, child, crng))` | the march sink |
| `MCRetire(transport, sink, queues)` | `march_close(m(at_stripe(sink, j)), queues.cohort[j])` | the run's `MCSink` (bins, child slot, ledger), so `tally!` keeps its scattering quantities (`programs.jl:334`) |
| `MCKey(provider, st, queues)` (W7) | `key[j] = region(provider, x) × 8 + octant(inverse_metric(st, x) p)` | provider and spacetime |

- `programs.jl:327-328`: `MCSink` gains `branch!(s::MCSink, ev, crng) = scatter_into!(s.children, ev, crng)`,
  with `scatter_into!(b::ChildBuffer{<:AbstractVector{Superphoton}}, ev, crng) = spawn!(b, kick(ev, crng)...)`
  (today's code) and `scatter_into!(b::ChildBuffer{<:AbstractVector{ScatterEvent}}, ev, crng) = spawn!(b, ev, crng)`.
- `montecarlo.jl:387-397` (`run_layers`): builds `Wavefront(start = MCStart(…),
  advance = MCAdvance(…), branch = scatter ? MCBranch(…) or MCKick(…) : nothing,
  retire = MCRetire(…), key = …, primary = MCPrimary(…), layer = MCLayer(…))` and calls
  `foreach_wavefront!`. The gathers are unchanged. `MCPrimary`/`MCLayer`
  (`programs.jl:420-467`) stay: they are the history form.
- `montecarlo.jl` docstring of `mc_spectrum` (`:230-247`): scattering runs as a
  wavefront on GPUs and as a cascade on threads; `capacity` is the initial child slots of
  a device (wavefront) or of a task (cascade). No keyword is added; the schedule is the
  executor's.
- `workgroupsize` of the stage programs: 32, as the MC programs today
  (`programs.jl:436-440`). Larger blocks become possible because a lane takes at most K
  steps, and W8 tests 64, 128 and 256 (bitwise by construction).

### 3.5 Extension

`ext/BlackLightPlusCUDAExt.jl`: no change is needed. `wavefront(::CUDADevice)` lives in
`devices.jl`.

### 3.6 Tests

| file | content |
|---|---|
| `test/decks.jl` | `WaveHost(k) <: AbstractDevice`: forwards the device protocol to `HostDevice(k)` and answers `wavefront = true` (about 15 lines) |
| `test/infra/execution_wavefront.jl` (new) | toy programs (keyed streams, spawns by stream identity, exact sums, as `test/infra/execution_cascade.jl:23-80`): totals, member checksum and child count are bitwise across `WaveHost` 1, 2, 3, 5 devices, budgets 1, 3, 64, `typemax`, cohorts 1, 7, 2^10, refill blocks 1, 5, capacities 1, 2, `nothing`, sort on and off, and equal to the cascade's for programs that respect `depth`; `ThreadedExecutor` and `HostDevice` pools take the cascade path (checked through a probe of the launched program types) |
| `test/physics/montecarlo_track.jl` | on the host: `march_open` → `march_steps` in pieces of K ∈ {1, 2, 7, 32} → `march_close` equals `track!` bitwise (state, sums, stream, recorded photon, ledger) for 10⁴ photons of the thin torus deck and of the scattering sphere; `branch!` default equals the former inline kick |
| `test/pipeline/products_montecarlo.jl` | MC thin, sphere scattering and torus scattering on `WaveHost` pools of 1, 2, 3 devices and capacities 1, 3: bitwise among themselves; equal counts and sums within 1e-11 of threads (the host compiles the stage programs in other contexts than `track!`, so bitwise equality to threads is recorded, not asserted); `device_isbits` and zero allocation of every stage program; the existing executor test (`:259-281`) unchanged and passing |
| `test/infra/execution_cuda.jl` (GPU job) | the wavefront decks on one and two GPUs bitwise; budgets and capacities as above |
| `test/structure.jl` | no change; must pass (one kernel, no `CUDA.`, budgets, docstrings, exports) |

### 3.7 Benchmarks and documents

- `bench/kernel_audit.jl`: cases `mc_torus_{thin,scatter}_advance`, `mc_torus_scatter_kick`,
  `mc_torus_start`, `mc_torus_retire` (registers, stack, spills, offline sm_80). This
  requires fixing the `mc_torus_*` cases that fail at `a5923dd` (`map_march.md` §10).
- `docs/src/design/performance.md`: §1 (execution model: the wavefront schedule of
  devices, the history schedule of threads), §3.4 (Monte Carlo), §4 (not carried: the CPU
  wavefront, per-step stage splits, cut E), §6 (the new tables).
- `docs/src/design/architecture.md` (Execution section): `foreach_wavefront!`, the queues,
  the device trait. `docs/src/api/execution.md`: the new names.
- Record of the work (implementation stages): `results/blp/mc_audit/speed/wavefront/`
  in the main checkout (README, `scripts/`, `logs/`).

## 4. Data layout and memory

States are AoS of `isbits` records. Per segment, the traffic is one load and one store of
the state per K steps (688 B per 32 steps, 21 B per step) against about 350–420 B per step
of plasma samples (11–13 sectors, `map_march.md` §5.1), about 5 % **(estimate)**. A
structure-of-arrays layout (Laine et al.: +80 %) becomes worth doing only if W5 measures
the state traffic above 10 % of the advance kernel's time.

`MarchState` (344 B, field order fixed for alignment):

| field | type | bytes | source in `track!` |
|---|---|--:|---|
| `y` | `SVector{8,Float64}` | 64 | `track.jl:281, 333` |
| `w`, `tau_a`, `tau_s` | `Float64` | 24 | `:282, 330-332` |
| `na0`, `ns0`, `th0` | `Float64` | 24 | `:284-286, 328, 333` |
| `r_prev` | `Float64` | 8 | `:288, 303` |
| `loss`, `loss_final`, `spawned`, `unspawned`, `absorbed` | `Float64` | 40 | `:291, 313, 318, 323, 325` |
| `rng` | `MCRng` | 24 | `:321` (advanced only through draws) |
| `nev` | `Int` | 8 | `:290, 321` |
| `it`, `outcome` | `Int32` | 8 | loop variable (≤ 100,001), `:289, 295-300` |
| `entry` | `Superphoton` | 144 | `sph` (read: `nscatt`, `w`, diagnostics; `:287, 319, 336-339`) |

Per device, at the defaults (P = 2^18, refill R = 2^14, initial child slots = P):

| buffer | element | slots | bytes |
|---|---|--:|--:|
| A, B (cohort, next) | `MarchState` | `P + cap_C + R` each ≈ 5.4e5 | 2 × 186 MB |
| D (done) | `MarchState` | P | 90 MB |
| C, inline kick | `Superphoton` + `MCRng` | `cap_C` ≈ P (grows) | 44 MB |
| C, deferred kick | `ScatterEvent` + `MCRng` | `cap_C` | 151 MB |
| `q`, `key` | `Int32`, `UInt32` | `cap_B` | 4.3 MB |
| `hist`, cursors | `Int64` | 2,752 | 44 KB |
| counters | `Int64` | 3 | 24 B |

Under 1 GB of the A100's 80 GB, so memory does not constrain P or `cap_C`.

## 5. The invariants and why they hold

### 5.1 Each photon's arithmetic

The stages cut the photon life only between whole calls. Between launches, a photon is
the `MarchState` of §4: every value that crosses `track!`'s loop back edge
(`track.jl:292-334`), stored and reloaded as an IEEE `Float64` or integer. Nothing is
recomputed at a resume. `spawning`, `frame` and `Cgr` are pure functions of `m` and
`s.entry`, and the same expressions give the same bits.

- **CPU.** `track!` remains one non-inlined function whose body inlines `march_open`,
  `march_steps(…, max_steps(ctl))` and `march_close`. The operations and their order are
  those of today's body. The struct packing and unpacking of `MarchState` vanishes under
  SROA, and with `s.it = 1` from `march_open` the loop bounds fold to `1:max_steps(ctl)`.
  The `branch!` default inlines today's kick and `spawn!`. The compiled body is expected
  to be the same IR. This is an expectation, not a proof, so W1 checks it bitwise on all
  13 decks at 1 and 16 threads and on the host-pool tests.
- **Device.** CUDA.jl contracts every `muladd` and every contract-flagged pair, and emits
  every other operation `.rn`, which `ptxas` does not fuse (`map_compiler.md` §2). The
  rounding of an operation therefore does not depend on the kernel that contains it, with
  two exceptions:
  (a) a contract-flagged product whose consumer is in another basic block, which `ptxas`
  may fuse (`map_compiler.md` §2, `xbb`). Across the back edge this needs a loop-carried
  value whose last operation is a contract-flagged `fmul`. None of the carried values is
  one: `y` ends in the plain `y + (h/6) * acc` (`src/physics/Spacetime/geodesic.jl:189-195`),
  `w *= …`, the depths and the five sums are plain operations (`track.jl:309-333`).
  (b) `reassoc` reductions (StaticArrays `dot`) re-associated differently around
  different constants. Every operand is a run-time value in both forms.
  A dependence of the bits on K would expose (a), so the K sweep of W2 detects it.
- **Kick context** (deferred form only). Moving the kick to `MCKick` breaks a fusion only
  if a zone field or `momentum(yev)` ends in a contract-flagged `fmul` consumed by a
  contract-flagged add in the kick. The zone comes from the decode
  (`primitive_four_vectors`: `fma` chains, `map_compiler.md` §2) and from the Kerr–Schild
  metric (plain operations), so no such pair is expected. W6 checks it on the Compton
  decks against main on the GPU. If W6 fails, the inline form stays.
- **Emission and retirement.** The start stage computes `emit` → `EMITTED` →
  `march_open`, today's `MCPrimary` prefix (`programs.jl:428-433`) and `track!` prologue.
  `MCRetire` computes `track!`'s epilogue. Both are cuts between whole calls.

### 5.2 Streams

- A primary's stream is `primary_rng(seed, g)` with the global index `g`
  (`philox.jl:192-193`), whatever block, device or iteration emits it.
- A child's stream is `child_rng(parent.rng, nev, layout)` (`philox.jl:213-220`). It
  depends only on the parent's identity (key and `sid`, carried unchanged in
  `MarchState.rng`) and on the ordinal `nev`, which continues across segments in the
  state.
- Draw order on a photon's own stream: generation, then one uniform per spawning step
  with `dτ_s > 0` and one per event (`map_mcprograms.md` §5). The loop is today's, so the
  order is today's at every K.
- Inline kick: the child is stored with its stream advanced past the kick, as `spawn!`
  does now (`track.jl:237, 243`). Deferred kick: the event record carries the child
  stream before the kick, and `MCKick` draws the same prefix of it. In both forms the
  child's own march continues on the same stream position.
- `fits` (`event < event_capacity(layout)`) and the layout come from the same
  `ChildBuffer` placeholder as today (`children.jl:60-61`).

### 5.3 Exact accumulators

- **Bins** (`ExactBins`, `accumulators.jl:161-198`): an escape adds one count
  (`Int64` atomic) and `w`, `wE`, `(wE)²` through `xsum_atomic_add!`
  (`exactsum.jl:79-90`). `xsum_parts` maps a `Float64` addend to one limb index and three
  integer increments, a pure function of the addend. A cell's limbs after the run are the
  integer sum of its addends' increments: order-free, exact, and free of overflow below
  2³² − 1 addends per cell (`exactsum.jl:20-21`). The cell is rounded once on the host.
  The multiset of addends per cell is one `(E, w, dir)` per escaped photon, taken from
  its final state, which is bitwise today's (§5.1). So every cell is bitwise today's
  (`same`, `products_montecarlo.jl:37`).
- **Ledger** (`ExactScalars(11 · 64)`, `programs.jl:244-289`): one addend per primary for
  `EMITTED` (start) and `PRIMARY_WEIGHT` (retire, from `s.entry.w`); one per photon for
  `ABSORBED`, `SPAWNED`, `SCATTER_LOSS`, `SCATTER_LOSS_FINAL` and `UNSPAWNED`; one per
  captured or dropped photon; one per escape for `ESCAPED`. Each value is a per-photon
  `Float64` sum accumulated in step order inside `march_steps` and tallied once at
  retirement (I7). The stripe of an addend changes: start uses `g`, retire the done slot
  `j`, where today the layers use their slot in the generation (`programs.jl:314-315, 463`).
  But every read sums all 64 stripes exactly (`ledger_sum`, `ledger_count`,
  `programs.jl:282, 289`), so the totals and addend counts, `n_recorded` among them, are
  unchanged. Today two plain Compton runs already differ in raw stripe limbs with equal
  sums (P§4).
- **No double counting.** The advance writes no accumulator. Start, branch and retire
  cannot overflow (§5.4), so each runs once per item.
- **Child count.** The child counter after the advance launch that succeeded is summed
  over iterations and devices. It equals the number of stored children, the size of the
  genealogy set, as `foreach_cascade!` returns it today (the final layer spawns nothing:
  `spawning = S && nscatt < max_scatt`, `track.jl:287`).

### 5.4 Child-queue overflow

- Before each advance launch: `cap_B ≥ nA + cap_C + R` and `cap_D ≥ nA`. Each member
  yields at most one state (B or D). The branch adds exactly `nC ≤ cap_C` states, and
  the refill adds at most `cap_B − nB` (chosen by the host). B and D never overflow, and
  `admit!` finds a slot by construction. A debug check compares the counters with the
  capacities after each launch and throws if one is exceeded.
- C can overflow: `spawn!` counts every attempt and stores only within capacity
  (`children.jl:69-76`). On `nC > cap_C` the driver discards B, D and C (resets their
  counters), grows `cap_C` to `nC + nC ÷ 8` (and `cap_B` with it) and relaunches the
  advance on the unchanged A and q. The advance is a pure function of A apart from its
  output buffers: it has no accumulator effects, and its draws come from per-photon
  streams. The rerun therefore produces the same set of survivors, retirees and children.
  The grown capacity carries over to later iterations and calls (`caps`). The cascade's
  generation-scale rerun (`cascade.jl:324-328`), with its scratch accumulators and
  drains, is not needed.
- Cost of an overflow: one advance launch, 5–10 ms at P = 2^18 **(estimate)**, against
  9.79 s of generation reruns today (P§4.5).

### 5.5 Independence from K, P, R, order, sort and block size

Each of these changes only which thread runs which part of which photon, and when: I1
(work is a function of the item), I3 (keyed streams), I4 (exact sums) and I5 (the set of
children is deterministic). §5.1 makes every part's arithmetic independent of the launch
it runs in. W2–W4 test the claim on the decks with K ∈ {1, 7, 32, 4096}, P ∈ {2^10, 2^18},
capacity ∈ {1, 3, `nothing`}, sort on and off, and blocks of 32 to 256.

### 5.6 One GPU against two

Every device of one architecture runs the same compiled programs (I10). The primary
blocks a device takes change only where a photon runs. Per-device accumulator twins are
folded by integer addition, so one and two A100s give bitwise equal bins, ledger and child
count. Profile job 3795 found the same for the thin cascade (P§6).

### 5.7 CPU: 1 against 16 threads, and against main

The threaded path is `foreach_cascade!` as today: chunk boundaries depend only on `n`
and the chunk, totals are exact, and an overflow reruns only the overflowing item
(`cascade.jl:191-264`). Bitwise across thread counts by the same argument as today, and
bitwise against main if W1 passes (§5.1).

### 5.8 CPU against GPU

If the wavefront is bitwise main on the GPU (W2, W4, W6) and the CPU is bitwise main
(W1), the CPU–GPU comparison is today's: equal counts per cell, sums within 3.1e-13
(thin) and 6.1e-10 (Compton) per cell on the stride-1 runs (speed README) and within the
deck tolerances of `compare_decks.jl`. If a device-side cut moves device bits, the
difference stays at rounding level. It still breaks "bitwise against main" for the GPU
decks, and accepting it is the user's call (§9).

### 5.9 The backend option

The schedule is a property of the device kind (`wavefront(dev)`), read inside Execution.
It is not a keyword of `mc_spectrum`, a field of a plan or config, or part of a function
name. Products calls `foreach_wavefront!` whatever the executor, and the user still
selects CPU or GPU only through `backend` (keyword, then session default, then
`BLP_BACKEND`, then auto). `budget`, `cohort`, `refill` and `sort_every` are keywords of
the Execution primitive, set by tests and benchmarks and never by plans.

## 6. Expected speed-up (estimates from the measured profile)

Cost unit: the measured time per warp-step of a class at full residency (P§4.3), with
E_new = 0.90 for the lane efficiency of the segment launches (§2.3: 3.5 % idle inside a
segment, under 5 % in the last partial wave, rounded down) **(estimate)**, and 750
resident warps (the bulk residency measured in P§4.3; today's means are 539 thin and
606 Compton over the whole phase). Today's warp-seconds come out of the same model within
8 %: thin 4.58e8 / 0.201 / 32 × 30.3 µs = 2,158 against 539 × 3.76 = 2,027 measured;
Compton primaries 1.084e9 / 0.187 / 32 × 123.5 µs = 22,370 against 20,668 measured.

### 6.1 Thin photon phase (torus 00999, stride 1, 1,000,005 primaries; today 3.74 s)

| term | arithmetic | time |
|---|---|--:|
| bulk | 1,000,005 × 458 steps = 4.58e8 photon-steps; / 0.90 / 32 = 1.59e7 warp-steps; × 30.3 µs / 750 warps | 0.64 s |
| tail | the 20,000-step photons (8 of them, P§4.1). During the bulk an iteration takes 2^18 / (750 × 32) = 10.9 waves × 32 steps × 30.3 µs = 10.6 ms, i.e. 3.0 steps/ms, so about 1,900 steps by the end of the bulk. The remaining 18,100 steps run alone at about 25 µs per step (6.5 cycles per instruction × 4,150 instructions at 1.065 GHz: the measured stalls of P§2.3 less "not selected" and most of "math pipe throttle") | 0.45 s |
| stages | about 70 iterations × (4 launches + 2 copies) × 20 µs = 6 ms; emission of 1e6 primaries in coherent blocks (about 1 % of the photon-steps' work) about 0.04 s; retirement of 1e6 records about 0.02 s | 0.07 s |
| **total** | | **≈ 1.16 s (3.2×; range 1.0–1.4 s, 2.7–3.7×)** |

The thin march is not memory-bound (L1 96 %), so the sort is expected to give nothing
here. Two A100s halve the bulk and stages but not the tail: about 0.8 s. Against 32 CPU
threads (11.3 s): about 10×.

### 6.2 Compton photon phase (Ns 1e6, Bias(1e4), max_scatt 3; today 62.5 s)

Photon-steps by class (P§4.1): primaries 1,000,005 × 1,084 = 1.084e9; layer 1
359,280 × 413 = 1.48e8; layer 2 991,373 × 226 = 2.24e8; layer 3 5,951,363 × 181 =
1.077e9; in all 2.53e9. Warp-steps at E = 0.90 (photon-steps / 28.8): 3.76e7, 5.15e6,
7.78e6, 3.74e7. Event rates per photon-step: 4.0e-4, 6.7e-3, 2.7e-2, 0. Events per warp
and segment of 32 steps: 0.41, 6.9, 27.3, 0.

Cost per warp-step in three scenarios:

| class | today (P§4.3) | S1: deferred kick, warps reconverge, an event costs about one step | S2: inline kick; segments bound the primaries' splitting | S3: per-warp-step costs unchanged |
|---|--:|--:|--:|--:|
| primaries | 120–127 µs | 46 µs (44.3 × (1 + 0.03 next-event draw + 0.013 events)) | 54 µs (44.3 × (1 + 0.34 × 0.5 split + 0.41 × 0.12 kick); 0.34 = 1 − e^{−0.41}) | 123.5 µs |
| layer 1 | 115–147 µs | 54 µs (44.3 × (1 + 6.9/32)) | 130 µs | 130 µs |
| layer 2 | 185–222 µs | 82 µs (44.3 × (1 + 27.3/32)) | 200 µs | 200 µs |
| layer 3 | 44.3 µs | 44.3 µs | 44.3 µs | 44.3 µs |

| term | S1 | S2 | S3 |
|---|--:|--:|--:|
| warp-seconds: primaries / L1 / L2 / L3 | 1,731 / 278 / 638 / 1,657 | 2,033 / 670 / 1,556 / 1,657 | 4,648 / 670 / 1,556 / 1,657 |
| bulk (sum / 750 warps) | 5.74 s | 7.89 s | 11.37 s |
| kicks in their own stage (7.3e6 kicks × about 40,000 lane-instructions / 32 at about 1.15e11 warp-instructions/s) | 0.1–0.2 s | in the bulk | in the bulk |
| tail (step-limited primaries, 784 of them, and their late children) | 0.6 s | 0.6 s | 0.6 s |
| stages (about 300 iterations of launches and copies; 7.3e6 opens; 8.3e6 retirements; rare advance reruns) | 0.3 s | 0.3 s | 0.3 s |
| **total** | **≈ 6.8 s (9.2×)** | **≈ 8.8 s (7.1×)** | **≈ 12.3 s (5.1×)** |

S2 is the expected result of W4 (inline kick) and S1 that of W6 (deferred kick). S3 is
the floor if the segment form reduces only the step tail and the reruns. Removing the
reruns alone (a presized `capacity`, bitwise) would bring today's 62.5 s to about 52.7 s,
so measured against that baseline the wavefront gives 4.3–7.8×. The sort (W7) targets the
layer steps (2,573 of the 4,304 warp-seconds in S1). Bringing them from layer 3's
8.6 ns per instruction to thin's 7.3 ns would save about 15 % of them, about 0.5 s
**(estimate)**. Two A100s: S1 about 3.8 s. Against 32 CPU threads (62.5 s): 5–9×.

### 6.3 CPU

Unchanged by construction (§2.7): thin 11.3 s, Compton 62.5 s on 32 threads. W1 must show
no slowdown under the timing protocol (§8.2). W-CPU may gain a few percent on the Compton
phase.

### 6.4 Small decks

The stride-8 decks (2e4 primaries) fit in less than one wave. Their GPU time is the tail
in both schedules, and the wavefront adds about 15–20 launches with K doubling, about
1 ms **(estimate)**. They are reported, not targeted.

## 7. Risks and how they show

| # | risk | detection | response |
|---|---|---|---|
| R1 | the `track!` split changes CPU bits | W1: 13 decks at 1 and 16 threads, `compare_decks.jl`; host-pool tests | try `march_steps` behind `@noinline` (one compiled loop for both forms); if bits still move, stop and ask the user for a one-time CPU re-baseline |
| R2 | the resumed march changes device bits, or they depend on K | W2: GPU decks at K ∈ {1, 7, 32, 4096} against main | locate the operation by comparing per-photon states after N steps at K = 1 and K = ∞ (scratch probe); a carried value fused across the back edge has to be written with an explicit `fma` or plain operations, which is a re-baseline, the user's call |
| R3 | the deferred kick changes device bits | W6: Compton decks against main GPU | keep the inline form (S2) |
| R4 | divergence persists inside segments (S3) | W5: ncu active threads per instruction against lanes stepping (from counters), instructions per warp-step of `MCAdvance` | deferred kick (W6); smaller K for spawning cohorts; if still ≥ 2×, accept S3 |
| R5 | residency below 750 warps in the segment launches | ncu achieved occupancy; the `%smid` probe of P§4.3 | blocks of 64–256 (W8, bitwise) |
| R6 | host gaps between launches | nsys: inner idle time of the call (today 0.02–0.08 s) | launches that do not wait (W8); one combined counter copy; larger K; fewer sort iterations |
| R7 | the tail longer than estimated | nsys GPU metrics: time with under 25 % of SMs active at the end | adaptive K; smaller cohort P; accept the critical path |
| R8 | atomic contention in retire (bins of the spectral peak) and on the queue counters | ncu of `MCRetire`; its share of the call | batch retirements over several iterations; split the queue counter by block range (prefix sums through two launches) |
| R9 | cold-call compilation grows (five programs instead of two) | nsys module-load intervals (today 10.6–21.4 s per MC program, P§5.2) | report; warm calls are the target; the start and branch programs are smaller than `MCPrimary` |
| R10 | the wavefront logic is exercised only on GPUs | the `WaveHost` tests (§3.6) | run every queue, overflow and multi-device test on `WaveHost` in CI |
| R11 | pool growth in child bursts | peak `nB`, `cap_C` per call in the driver's log line | refill pauses (§2.3); `cap_C` hard limit with reruns |
| R12 | the CPU slows through the refactor | timing protocol on 32 threads | `@noinline` variant (R1) or revert |

## 8. Work breakdown

Each item is at most about two days, and each lists its acceptance. Dependencies are in
the last column. Items in the same row of dependency can go to different agents.

| item | content | acceptance | after |
|---|---|---|---|
| **W0a** replay (0.5 d, host, no code in the package) | replay of the wavefront schedule on the recorded per-photon step counts and genealogy (`plan/profile/data/probe_s1/{thin_primary,compton_primary,compton_layer}.bin`: steps, events, layer, stream identity) on a model device of 750 × 32 lanes. Child spawn times uniform over the parent's life, with the parent's start and end as bounds. Cost tables S1–S3 of §6.2. Sweep K ∈ {8 … 512}, P ∈ {2^15 … 2^19} | a report of E_new, bulk, tail and total per (K, P) for thin and Compton; the thin estimate of §6.1 holds within 30 % or the design numbers are revised | — |
| **W0b** kick-isolation probe (0.5 d, 1 GPU job, no package code) | the Compton deck's primary generation (Ns 1.6e5 as P§2.1) through `run_layers` with `Opacities(…; weight_min = Inf)`: the same steps and draws, no kick, no spawn. ncu instructions per warp-step and active threads per instruction against the plain run | if instructions per warp-step fall below 8,000 (from 15,665), the kick causes the splitting (S1 reachable; W6 a priority); if not, the event site does (S2 or S3; W6 deprioritised) | — |
| **W1** march split (1.5 d) | §3.1 complete: `MarchState`, `march_open`/`march_steps`/`march_close`, `track!` composed, `ScatterEvent`, `kick`, `branch!`; host resume tests (§3.6) | 13 decks bitwise main at 16 and 1 threads; full test suite on the host; MC timing on 32 threads within the protocol of §8.2 (best of 3, new ≤ ref × 1.03); one GPU job: the cascade on CUDA (still `MCPrimary`/`MCLayer`) bitwise main on the GPU decks; kernel audit of `MCPrimary`/`MCLayer`: registers and stack unchanged | — |
| **W2** device resume check (1 d, scratch program + 1 GPU job) | a scratch `foreach_item!` program that runs `march_steps` K steps per launch over a host-compacted index list (no queues), thin and Compton primaries of the stride-8 decks | per-photon final states, bins and ledger bitwise main GPU for K ∈ {1, 7, 32, ∞} | W1 |
| **W3** primitive, thin form (2 d) | §3.3 without children and sort: `Wavefront`, `WaveQueues`, `Cohort`, `StateBuffer`, `WaveSwap`, the driver (start, advance, retire, refill, adaptive K), `wavefront(dev)`, `WaveHost`; §3.4 `MCStart`, `MCAdvance`, `MCRetire`; `run_layers` on `foreach_wavefront!` | host: structure gates; toy tests bitwise across `WaveHost` configurations; MC thin on `WaveHost` bitwise among configurations and within 1e-11 of threads; GPU job: `mc_torus_thin`, `mc_torus_s1`, `mc_sphere_thin` bitwise main GPU; one against two GPUs bitwise; **thin photon phase at stride 1, Ns 1e6 ≤ 2.0 s** (estimate 1.16 s) | W1, W2 |
| **W4** children, inline kick (2 d) | `MCBranch`; child queue C, overflow rerun of the advance, `caps`, child count | GPU: `mc_torus_compton`, `mc_sphere_scatt` bitwise main GPU, also with capacity 1 and 3; one against two GPUs bitwise; `WaveHost` scattering tests; **Compton photon phase ≤ 15 s** (S2 estimate 8.8 s) | W3 |
| **W5** profile the stages (1 d, GPU jobs) | ncu of `MCAdvance` thin and Compton (deck of P§2.1), `MCStart`, `MCBranch`, `MCRetire`; nsys of one warm call of each phase; E per iteration from the counters | a report against §6: instructions per warp-step per class, active threads against lanes stepping, residency, inner idle, tail length | W4 |
| **W6** deferred kick (2 d) | `ScatterEvent` children, `MCKick`, `scatter_into!` dispatch | Compton decks bitwise main GPU (else stop, R3); CPU decks unchanged; ncu: spills of `MCAdvance` (scatter) below W4's; **Compton photon phase ≤ 10 s** (S1 estimate 6.8 s) | W5 (W0b decides priority) |
| **W7** coherence sort (2 d) | `region` (§3.2), `MCKey`, `KeyCount`, `KeyPlace`, host scan; `sort_every` | GPU decks bitwise with the sort on (any order); sectors per request and L1 hit of `MCAdvance` on layer-dominated cohorts; kept on by default only if the Compton phase gains at least 3 %, otherwise removed (no dead code) | W4 |
| **W8** launches and tuning (1.5 d) | `launch!(…; wait = false)` for the stream-ordered launches; one counter copy; sweep of K, P, R and block size (32–256) | nsys inner idle ≤ 3 % of the photon phase; defaults set from the sweep; every point of the sweep bitwise | W5 (W6, W7 when done) |
| **W-CPU** child sort on threads (1 d, optional) | sort each chunk generation's children by `region` in `chunk_generations!(c, r, bufs)` | CPU decks bitwise; Compton on 32 threads at least 3 % faster, otherwise not landed | W7 (`region`) |
| **W9** documents (1 d) | §3.7: performance.md, architecture.md, API page, docstrings, kernel-audit cases and baseline | `test/structure.jl` passes; docs build; text written as the standard | W8 |
| **W10** final verification (1 d, 2–3 GPU jobs) | the deck protocol on CPU (1 and 16 threads), one GPU, two GPUs; the timing protocol for thin and Compton at stride 1 on one and two A100s and 32 CPU threads; record in `results/blp/mc_audit/speed/wavefront/README.md` | every invariant of the task statement checked and logged with the job ids | W9 |

Critical path: W1 → W2 → W3 → W4 → W5 → W6/W7 → W8 → W9 → W10, about 15 working days.
W0a and W0b run on day 1 in parallel with W1. The toy tests and `WaveHost` of W3, and the
sort primitive of W7 (toy keys), can be written in parallel with W2.

### 8.1 Bit protocol

- CPU decks: `results/blp/mc_audit/speed/gpu/scripts/decks.jl` (13 decks) with the branch
  environment and with `ref_main` (`/data2/haiyangw/scratch/wavefront/ref_main`, at
  `a5923dd`). Each runs at `julia -t 16` and `-t 1`, `DECKS_OUT` under
  `/data2/haiyangw/scratch/wavefront/decks/`, then `compare_decks.jl`: "all decks bitwise".
- GPU decks: the same, inside one Slurm job (`preempt`, `--gres=gpu:1`, `-c 4`,
  `julia -t 4`) through `results/blp/gpu/run_job.sh '^decks done'`. The job header
  shows the commit and dirty count 0. Main GPU against branch GPU must be bitwise; branch
  GPU against branch CPU must be equal in integer parts and within today's relative
  difference.
- Two GPUs: `--gres=gpu:2`; the MC decks under `:gpu` against `gpu:0`, bitwise.
- Sweeps (W2–W8): the MC decks through `run_layers` with explicit `budget`, `cohort`,
  `refill`, `capacity` and `sort_every`, every output bitwise main GPU.

### 8.2 Timing protocol

- GPU: one Slurm job per comparison (the node runs one job at a time, 4 CPUs).
  Processes in A-B-A order (`a5923dd`, branch, `a5923dd`), each process doing: snapshot
  load, sites, tables, one compile call of each phase, then three warm calls of the
  thin photon phase (stride 1, Ns 1e6, seed `0x5eed0700`) and three of the Compton phase
  (Bias(1e4), `max_scatt = 3`), timed around `run_layers` as `mc_phases.jl` does. Report
  every call, the best and the median. The bins of every call must equal the
  reference's. Record the SM clock (`nvidia-smi --query-gpu=clocks.sm`) at the start and
  end. A preempted job is rerun, not merged.
- CPU: 32 threads, the trees interleaved A-B-A-B, best of three and median, load average
  before and after. "No slower" means best(new) ≤ 1.03 × best(ref); a failure at load
  above 60 is rerun at lower load.

## 9. What falsifies the design early

| when | test | falsified if | consequence |
|---|---|---|---|
| day 1, host only | W0a replay | thin total ≥ 2.5 s (under 1.5×) at every (K, P); or the tail exceeds the bulk for both phases | the segment schedule gains too little over today; redirect the work to the lane overlap (P§5.3) and the reruns (§6.2) |
| day 1, one GPU job | W0b kick probe | instructions per warp-step stay ≥ 12,000 without the kick | the kick does not cause the divergence; S1 is out of reach; plan for S2/S3 and drop W6 |
| day 2 | W1 CPU decks | any of the 13 decks not bitwise, also with `@noinline march_steps` | sharing the step code between CPU and device is incompatible with "bitwise against main"; the user decides between a CPU re-baseline and leaving the device form unbuilt |
| day 3–4 | W2 device resume | device bits differ from main for some K, or across K | the cut at the back edge is not bit-free on the device; same decision |
| day 6 | W3 thin speed | thin photon phase > 2.0 s with E per iteration ≥ 0.8 measured | the per-step cost or residency assumption is wrong; W5 profiling before building children |
| day 8–9 | W4/W5 | Compton photon phase > 15 s, or active threads per instruction below half the lanes stepping in `MCAdvance` | S3 regime; W6 decides; if W6 also fails, the gain is about 5× and the remaining levers are outside this design (registers, cut F) |

## 10. Outside this design

- **The precise march (Verner 9).** Its tail is one pixel's serial latency: 27,255 steps
  (P§3), with the launch a full GPU for 0.2 s and a tail of a few blocks for 1.4 s. A
  segment schedule cannot shorten a serial chain, so the lever there is the per-step cost
  (registers, `map_march.md` §8 B1–B4).
- **Cut F** (geodesic and plasma halves of a step in separate kernels): 72 B of state per
  step and a step-internal cut (bit risk, `map_mcprograms.md` §8.4). The per-operation
  wavefront is 2–7.7× slower in Dr.Jit.
- **Cut E** (the event site out of the march): rebuilds the parent's sums out of order
  (I7).
- **Lane overlap** (P§5.3: shared twins serialise the two lanes of a device). It is
  independent of this design and benefits the cascade and the imaging programs. The
  wavefront does not need it because it uses one lane per device.
- **The host site scan** (2 s at stride 1): `map_compiler.md` §5; a re-baseline decision.
