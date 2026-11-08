# GPU execution of the BLP Monte Carlo: the incremental design

Tree `a5923dd` (branch `gpu-wavefront`, worktree `/data2/haiyangw/scratch/blp_wavefront`).
Paths are relative to the repository root unless absolute. Inputs: `plan/profile.md`
(+ `profile.json`), `plan/map_execution.md`, `plan/map_mcprograms.md`, `plan/map_march.md`,
`plan/map_compiler.md`, `plan/map_outputs.md`, `plan/map_literature.md`. New evidence
gathered for this design (offline, no GPU job) is in `plan/design_aux/` (section 12).
Sections 1 and 12 hold measurements with their sources; every figure in sections 0, 6 and
7.1–7.2 beyond the measured "today" values is an estimate, and its arithmetic is shown.

Scope: the photon phase of `mc_spectrum` (`Products.run_layers` →
`Execution.foreach_cascade!`). The angle: keep the per-photon megakernel and the
generation cascade, and recover GPU throughput with tools that change no compiled
arithmetic: child capacity, lane overlap through read-only and atomic-only sharing,
coherence ordering of primaries and of every generation's children, balanced chunks over
two GPUs, register caps, and bit-portable host code. No new execution primitive (no
second kernel, no queue, no resumable march, no persistent threads). The precise march is
outside this design; the register-cap trait and its bitwise argument (section 4.2) carry
over to `CrossingProgram`, whose PTX has four unrounded multiplies, each feeding an
`fma.rn`, a pair ptxas does not fuse (map_march §3, map_compiler §2).

## 0. Verdict

| photon phase, torus 00999 stride 1, 10⁶ primaries | CPU 32 thr | A100 today | Tier 1, certain part (capacity + lane overlap) | Tier 1 with spatial keys | Tier 1 with step keys (if F3 passes) | two A100s, Tier 1 best case | Tier 2 (deferred kick, if F4 passes), one / two A100s |
|---|--:|--:|--:|--:|--:|--:|--:|
| thin | 11.3 s | 3.74 s | 3.1 s | 3.1 s (no primary key: section 6.2) | 2.1–2.3 s | 1.1–1.2 s | 1.7 s / 0.9 s with the emission split (section 7.2) |
| Compton (bias 10⁴, 7.30 M children) | 62.5 s | 62.5 s | 47.8 s | 32.5–46.4 s (central 38.6) | 19.1–47.0 s | 10–24 s | 16 s / 8.6 s (spatial), 11 s / 6.1 s (step keys) |

All A100 figures beyond "today" are estimates (section 6). This design alone takes the thin phase
from 3.0× to about 5× the rate of 32 CPU threads on one A100 (about 10× on two), and the
Compton phase from 1.0× to 1.3–3.3× on one A100 (2.6–6× on two). The Compton range
is wide because one quantity is not yet measured: how much a launch whose photons
**spawn** speeds up when its warps hold photons of similar lifetime. Experiment F2
(section 9) measures it with oracle ordering in one GPU job before any package change,
and the range collapses to one number.

What the incremental tools cannot reach: in the spawning launches (Compton primaries,
generations 1 and 2: 87 % of the Compton warp-seconds once the reruns are gone) the
lanes of a warp execute each step in 2.0–3.6 separate groups (ncu: 1.69 active threads
per warp instruction against 6.15 lanes still marching for the Compton primary) and
generation 2 stalls on instruction fetch (8.75 "no instruction" stalls per issue).
Ordering, capacity, overlap, devices and register caps do not touch that. The smallest
remedy that keeps the megakernel is to take the Compton kick out of the march (Tier 2,
section 7), gated by experiment F4; if F4 shows the penalty is not the kick, the
remaining gap belongs to a stage-split (wavefront) design.

CPU: the threaded path keeps its code, so its bits and its speed are unchanged by
construction (section 5); this angle has no CPU speed-up of its own beyond an optional,
measured child ordering on threads.

## 1. Measured facts this design uses

| fact | value | source |
|---|---|---|
| warp-seconds, thin phase | 2,033 (mean 539 of 864 warp slots, span 3.76 s) | profile §4.4, `profile.json` `mc_probe.residency` |
| warp-seconds, Compton phase | primaries 20,661; generation 1 2,716; generation 2 4,929; generation 3 4,040; overflow reruns 5,332; total 37,683 (mean 606 resident, 62.2 s) | `profile/logs/probe_analyze_s1.log:29-70` (span × mean resident per launch) |
| time per warp-step | thin 29–31 µs in every chunk (lane efficiency 0.165–0.256 across chunks); Compton primary 120–127 µs; generation 1 115–147, 2 185–222, 3 44 µs | same log |
| lane efficiency E (Σ steps / Σ 32·warp max) now → sorted by (meshblock, octant, zone) → oracle | thin 0.201 → 0.279 → 0.997; Compton primaries 0.187 → 0.262 → 0.999; generation 1 0.167 → 0.305 → 0.992; 2 0.225 → 0.446 → 0.993; 3 0.367 → 0.564 → 0.999 | profile §4.2 |
| active threads per warp instruction against lanes still marching | thin 6.37 / 6.69; Compton primary 1.69 / 6.15; generation 1 2.13 / 4.76; 2 3.50 / 6.92; 3 10.65 / 11.78 | profile §2.2, §7 |
| lane end before warp end | non-spawning launches 0.0001 of the warp span; spawning launches 0.39–0.49 | probe log |
| lanes of one device serialize | every launch converts the shared device twin's arrays, CUDA.jl's `maybe_synchronize` waits on the other lane's stream | profile §5.3; `CUDACore/src/memory.jl:589-633` |
| overflow reruns | 5 of 17 `MCLayer` launches, 9.79 s of the 63.2 s call, 14 % of warp-seconds | profile §4.5 |
| resident warps in the bulk of a launch | 750, 684, 668 of 864 at 0.1, 0.3, 0.5 s into thin chunk 1 | profile §4.3 |
| emission inside the thin phase | device generation 1.2–1.4·10⁶ primaries/s → 0.71–0.83 s of 3.74 s; CPU profile 21 % | `results/blp/mc_audit/speed/PLAN.md:86, 112, 134-135` |
| every global load of the hot data already goes through the read-only path | SASS: 195 `LDG.E.CONSTANT` + 45 `LDG.E.64.CONSTANT` against 7 plain `LDG.E.64` (thin); layer 306 against 24 | `design_aux/thin.sass`, `layer.sass` |
| ptxas has no floating-point freedom in the MC kernels | PTX: 0 unrounded `mul.f64`, 0 unrounded `fma.f64`; 5–9 unrounded `add/sub.f64` with no multiply operand | `profile/raw/devcode/{thin,compton}/*.asm`, section 4.2 |
| register caps, offline (CUDA 13.3 `ptxas -O3`, sm_80), `track!` spill stores / loads | thin 255: 144/168 B, 168: 588/724, 128: 912/976; Compton layer 255: 1696/2536, 168: 2344/3948, 128: 2932/4804; Compton primary 255: 1704/2568, 168: 2360/4244 | `design_aux/*_r168.log`, `*_r128.log` |
| two GPUs today | thin stride 4: 3.99–4.09 s → 2.53–2.69 s, bitwise | profile §6 |

## 2. Architecture

The execution model stays: one kernel (`_foreach!`, `src/infra/Execution/kernel.jl:7-10`),
one thread per photon, the per-photon `track!` megakernel, the generation cascade per chunk
(`src/infra/Execution/cascade.jl:290-338`), exact integer bins, genealogy-keyed Philox
streams. Around it:

```
foreach_cascade!(exec, primary, layer, n, depth)             pool path only
  chunks: strided k:K:n when the primary has an order key      (balanced across lanes/devices)
  per chunk on a lane, per generation g:
      items_g   = r (g = 0) | 1:m (g ≥ 1)                     as today
      keys      = KEY_g over items_g          (new launch)    order_key(prog): UInt32 per item
      perm      = host stable radix sort of keys              (lane stage, ms)
      launch prog over items_g[perm]          (device Int32)  program unchanged, item unchanged
      overflow: rerun prog over the same items_g[perm]        rare: slots from a device budget
  lanes of a device run concurrently: shared twins are read-only after upload or written only by
  integer atomics, and are exempt from CUDA.jl's implicit stream synchronization (share!)
  register cap per program type (register_cap trait → maxregs in the CUDA extension)
```

- **Ordering, not regrouping.** A generation's photons are visited in an order that puts
  photons of similar fate in the same warp. The program and the item it receives are
  unchanged; only the visit order changes. This is the executor contract that
  `foreach_item!(...; order)` already honours bitwise (`executor.jl:126-141, 160-161`).
- **Keys are side programs.** A key program computes a 32-bit key per item from what the
  item will be (a primary's emitted photon, a child's stored state); its arithmetic never
  reaches an accumulator, so its bits are irrelevant.
- **Two key families.** Spatial key (meshblock, direction octant, coarse cell): measured
  E gains. Step key (predicted march length from a plasma-free coarse pilot march, the
  march's trajectory being plasma-independent, `track.jl:292-305`, map_mcprograms §3.2):
  E gain to be measured (F3).
- **The threaded path does not take the new structure** (section 5).

## 3. Exact changes

### 3.1 Execution layer (`src/infra/Execution`)

| file | change | lines (estimate) |
|---|---|--:|
| `devices.jl` (190 lines) | device protocol: `share!(dev, a) -> a` (default identity); `free_bytes(dev) -> Int` (default 0); `launch_kernel!(dev, kern, args; ndrange, workgroupsize, maxregs)` (default: `kern(args...; ndrange, workgroupsize)`, cap ignored) | +35 |
| `residency.jl` (105) | `cached_twin` calls `share!(dev, t)` on the fresh upload (`residency.jl:41`) | +2 |
| `accumulators.jl` (488) | `zeroed` (`:33`) returns `share!(dev, upload(...))` | +1 |
| `children.jl` (140) | the device `ChildBuffer` twin (`:44-51`, used by non-cascade programs) is shared the same way | +3 |
| `kernel.jl` (44) | traits `order_key(prog)` (default `nothing`) and `register_cap(prog)` (default `nothing`); `launch!` (`:33-44`) passes `register_cap(prog)` to `launch_kernel!` | +25 |
| `order.jl` (new, after `kernel.jl`) | `ordered_items!(lane, kprog, dkprog, items, m) -> device Int32 vector`: launch of the key program into lane buffer `(:order, :keys)` (UInt32), D2H through pinned stage `(:order, :keys)`, `radix_order!` on the host, H2D of `items[perm]` through stage `(:order, :items)` into lane buffer `(:order, :items)` (Int32); `radix_order!(perm, keys, scratch)`: stable LSD radix sort of `UInt64(key) << 32 | position`, two 16-bit passes | +120 |
| `cascade.jl` (365) | see below | +60 |
| `Execution.jl` | include `order.jl`; export `order_key`, `register_cap`, `share!`, `free_bytes` with docstrings (module exports, not top-level) | +4 |

`cascade.jl` in detail:

1. `Cascade{P,L,S,PK,LK}` (`:83-89`) gains `pkey = order_key(primary)` and
   `lkey = order_key(layer)`.
2. `run_cascade!(::PoolExecutor, ...)` (`:268-282`) builds `ProgramTwins` for the key
   programs next to the two existing ones.
3. `chunk_generations!(lane, ...)` (`:290-338`), for every generation including the
   thin-only branch (`:292-295`) and the final one: after `dprog = Adapt.adapt(swap, twin)`
   (`:316`), adapt the key twin with the same `CascadeSwap` (so its `Generation` is the
   parents), call `ordered_items!` once per generation, and launch the program over the
   returned items (`:322`). The overflow branch (`:324-328`) relaunches over the same
   items; keys are not recomputed.
4. Child slots of a non-final generation (`:298-299`):
   `cap = max(MIN_CHILD_SLOTS, cld(length(r), 2), caps_dev, slots already allocated in the
   lane's child buffer, min(budget, 16·length(r)))`, where
   `budget = free_bytes(dev) ÷ 8 ÷ (2 · lanes_of(dev) · (sizeof(T) + sizeof(MCRng)))` and
   `caps_dev` is the largest growth of any lane of the same device in this call (today:
   per lane, reset per call). On an 80 GB A100 with two lanes: 14.9 M slots per side per
   lane; the floor is `16·250,002 = 4.0 M` slots, 672 MB per side; generation 2 of the
   deck needs at most 1.93 M. The overflow mechanism is unchanged and keeps every run
   correct when the floor is too small; the lane then remembers its growth across calls
   through its buffer length. With the strided chunks of item 5 (125,001 primaries per
   chunk on one A100) the floor is 2.0 M slots, 336 MB per side, against 0.5–1.0 M children
   from generation 2 of such a chunk.
5. Strided chunks: `cascade_chunk` (`:24-28`) gives `K = 8·ndev` chunks on a pool whose
   primary has an order key, as `k:K:n` (`StepRange`, isbits, passed unstaged to the key
   launch); otherwise contiguous ranges as now. `chunk_generations!` takes an
   `AbstractRange`.
6. Unchanged: `run_cascade!(::ThreadedExecutor, ...)`, `chunk_generations!(c, r, bufs)`,
   `run_growing!`, `CascadeSwap`, `Drain`, `set_count!`/`get_count`.

Data layout per lane (new): `(:order, :keys)` UInt32 × m, `(:order, :items)` Int32 × m on
the device; pinned stages of the same sizes; host scratch UInt64 × 2m for the radix sort.
At the largest generation of the deck (m = 1.93 M): 15 MB device, 15 MB pinned, 31 MB
host per lane. Child slots keep their layout (`Superphoton` 144 B and `MCRng` 24 B in two
arrays, `children.jl:30-39`). Primaries need m ≤ 2³¹ − 1 for Int32 items.

### 3.2 Monte Carlo programs (`src/pipeline/Products`, `src/physics/MonteCarlo`)

New file `src/pipeline/Products/montecarlo_order.jl` (≈150 lines; `programs.jl` is at 533
of the 600-line budget), included after `montecarlo.jl`:

| name | kind | what it does |
|---|---|---|
| `coherence_key(provider, st, x, p, steps) -> UInt32` | function | bits 31–22: `1023 − min(steps ÷ s₀, 1023)` (0 when no step key); 21–9: meshblock `Data.locate(blocks, x)` (13 bits; 344 blocks in the torus); 8–6: octant of `k^i = g^{iν} p_ν` (signs, `inverse_metric(st, x)`); 5–0: coarse cell inside the block (4×4×4). `UniformSphere`: radial shell in place of the block |
| `MCPrimaryKey(source, seed, pilot)` | program `(item, i, keys)` | `sph, _ = emit(source, item, primary_rng(seed, item))`, `keys[i] = coherence_key(...)` |
| `MCLayerKey(generation, ctl, st, provider, pilot, max_scatt)` | program | `keys[i] = coherence_key(...)` of `generation.photons[item]` |
| `pilot_steps(pctl, st, y, spawning, cap) -> Int` | function (F3 decides) | plasma-free march with `CurvedMarch(C = C_pilot)` and the control's own stop rules (plunge stop only when `!spawning`), at most `cap` steps; the pilot step count predicts the march's (both scale with C ln of the distance ratio near the horizon) |
| `order_key(p::MCPrimary)`, `order_key(p::MCLayer)` | trait methods | the two key programs above, built from the program's own fields |
| `register_cap(::MCPrimary)`, `register_cap(::MCLayer)` | trait methods | per program type, from item B7 (default `nothing`) |

`MCPrimary`, `MCLayer`, `MCSink`, `track!`, `step_events`, `scatter_superphoton` and the
sources are not edited. `run_layers` (`montecarlo.jl:387-397`) is unchanged: the keys come
through the trait.

### 3.3 CUDA extension (`ext/BlackLightPlusCUDAExt.jl`)

| method | body |
|---|---|
| `share!(::CUDADevice, a::CuArray)` | `CUDA.synchronize()` (the upload or zeroing on this lane's stream is complete), then `CUDA.enable_synchronization!(a, false)` (public API, `CUDACore/src/array.jl:497-516`): later conversions on other streams neither wait for nor take over the writer's stream |
| `free_bytes(::CUDADevice)` | `CUDA.memory_info()[1]` on the active device |
| `launch_kernel!(::CUDADevice, kern, args; ndrange, workgroupsize, maxregs)` | for `maxregs === nothing` the KA call as today; otherwise the body of KA's CUDA launch (`CUDACore/src/CUDAKernels.jl:111-150`: `KA.launch_config`, `KA.mkcontext`, `@cuda launch=false always_inline=… maxthreads=… maxregs=…`, `kernel(ctx, args...; threads, blocks)`) with the explicit workgroup the MC programs set |

The extension already imports the protocol functions it defines (`ext:14-15`); the
structure gates (`test/structure.jl:63`: one `@kernel` in `src`; `:66-67`: no `CUDA.` in
`src`) stay green because nothing new enters `src` but programs and protocol defaults.

### 3.4 Bit-portable host code

- `Numerics.port_hypot`: Base's `_hypot` with the non-FMA branch and unfused operations,
  read by `_ks_term` (`src/physics/Spacetime/kerr_schild.jl:15`). It is bitwise the
  `hypot` every deck of this depot has run (0 mismatches in 10⁷ Kerr–Schild arguments,
  map_compiler §3.2, §5) and removes the dependence of the host bits on which
  dependency pkgimages the depot loads. On the device it replaces `__nv_hypot`, so the
  GPU decks move toward the CPU's (allowed: "or tighter"); it lands as its own item with
  its own GPU re-record.
- Portability gate (`test/portability.jl`, host only): for designated functions, compile
  the device job with `libraries = false` and assert no `__nv_`, no `llvm.fmuladd`, no
  `contract`/`reassoc`/`afn`/`fast` flag; for the host, the same flags absent from
  `code_llvm`. First designated function: `port_hypot`. It is the tool that later makes
  the decode, the kick and the site scan context-free (section 7.3).

## 4. The invariants

### 4.1 Results bitwise against main on the fixed decks (CPU)

The CPU decks run `backend = :cpu` → `ThreadedExecutor`; its cascade code is not edited
(section 3.1, item 6). `port_hypot` is bitwise the depot's `hypot` (measured over 10⁷
arguments, to be re-checked on the 13 decks). Nothing else reaches the threaded path.

### 4.2 Why the pool path is bitwise, change by change

| change | compiled arithmetic that changes | argument |
|---|---|---|
| child slots from a budget | none | no program changes; fewer overflow branches run; the existing test already proves capacity-independence (`test/infra/execution_cascade.jl:115-135`, capacities 1, 2, 5) |
| `share!` | none | host-side stream bookkeeping only; shared arrays are either read-only after a synchronized upload (residency twins) or written only by `Int64` atomics (`ExactBins`, `ExactScalars`, the `ChildBuffer` counter), whose sums are order-free; every launch synchronizes its own lane (`kernel.jl:42`) before anything reads back |
| order keys and permuted items | the kernel entry only (items `UnitRange` → `CuDeviceVector{Int32}`; one index load) | the floating-point work lives in non-inlined device functions whose method instances are unchanged: `julia_MCPrimary_*` and `julia__track__16_*` are `.func`s in the PTX (`profile/raw/devcode/thin/gpu__foreach!_1.asm:1474, 7274`; `compton/gpu__foreach!_3.asm:15, 2033`; `_4.asm:1339`), and the kernel entries, where `MCLayer`'s body is inlined, hold no `f64` operation at all (thin entry lines 459–1059, layer entry 458–1262: 0 `add/sub/mul/fma/div/sqrt.f64`). Each photon receives the same `item`: a primary its global index (stream `primary_rng(seed, item)`, ledger stripe `(item − 1) mod 64`), a child its buffer slot (stream from the slot, stripe from the slot). Keys and sorts are integer work on separate buffers |
| strided chunks | none | chunk membership is a fixed function of `n` and `K`; results do not depend on the chunking (I2, tested over chunk sizes 7–4096) |
| register caps | none in PTX; ptxas allocation, spills and rematerialization | `.maxnreg` reaches ptxas only. Every multiply in the MC PTX is `.rn` and the few unrounded adds have no multiply operand, so ptxas cannot contract, and PTX forbids reassociation. Offline at caps 255/168/128 the SASS `DFMA` counts are identical (thin 2534, layer 4554); `DMUL`/`DADD` differ by ≤ 1.3 % (rematerialization or moves; confirmed bitwise by the decks, item B7) |
| workgroup size | none | measured bitwise at 4 sizes (`results/blp/mc_audit/speed/gpu/logs/one_3782.log`) |
| `port_hypot` | `_ks_term` on the device | not bitwise against main on the GPU by design; CPU bitwise; lands separately |

### 4.3 The exact accumulators

- `ExactBins` per scattering order (`accumulators.jl:112-198`): per (row, bin) one `Int64`
  count and three 69-limb `Int64` sums (Σw, ΣwE, Σ(wE)²), written by `Int64` atomics
  (`xsum_atomic_add!`, `exactsum.jl:79-90`), rounded once on the host. Addition order and
  device do not matter.
- Ledger `ExactScalars(11 × 64)` (`programs.jl:244-272`): eleven quantities in 64 stripes;
  every read sums all stripes (I8). A permutation of the visit moves no addend between
  photons; in the layer an addend's stripe follows the slot, as today.
- `n_recorded` (the `ESCAPED` addend count) and the spawned count are `Int64` atomics.
- Per-photon running `Float64` sums (`loss`, `loss_final`, `spawned`, `unspawned`,
  `absorbed`) stay inside one `track!` call in step order (I7): nothing here splits a
  march.
- `Drain` of scratch into totals (`cascade.jl:163-187`) is integer addition.

### 4.4 Genealogy-keyed streams

Draw `k` of a photon's stream is Philox of `(k, sid)` under `(seed, id₀)`
(`philox.jl:51-54, 90-110`): a primary's identity is its index, a child's the parent's
identity with the event ordinal in the next field and the layer tag incremented
(`child_rng`, `philox.jl:213-220`), counter 0. No identity involves a thread, a slot
position in the visit order, a chunk or a device. Ordering changes when a photon runs,
never which draws it consumes.

### 4.5 Child-buffer overflow

Unchanged mechanism (I6): a non-final generation records into lane scratch; if its
children exceed the slots, the scratch is zeroed, the slots grow to `n_out·9/8`, and the
generation reruns over the same (permuted) items; only a generation whose children fit is
drained into the totals. The rerun spawns the identical set of children (same streams).
New: the slots start large enough that the deck never overflows (section 3.1, item 4),
and growth persists for the lane. The permuted item list stays in its lane buffer across
the rerun. The final generation stores no child (`slots = 0`) and is ordered as well.

### 4.6 One GPU against two GPUs

Every device runs the same compiled programs (same architecture; I10). Per-device
accumulator twins fold by exact integer addition (`accumulators.jl:209-216`). Children never
leave their lane. Chunk membership (strided or not) does not depend on the device count
beyond the deterministic `K`, and results do not depend on chunking. Per-device child
budgets differ only in which overflow branches run. So two GPUs stay bitwise one GPU,
as measured today (profile §6).

### 4.7 CPU against GPU

No floating-point code changes on either side except `port_hypot`, which brings the device
to the host's `hypot`. Acceptance: on every deck, the branch's CPU–GPU difference is at most
main's CPU–GPU difference (task tolerances: MC equal counts, sums within 1e-11; images
within 1.5e-13; stride-1 bins as recorded, equal counts, 3.1e-13 thin and 6.1e-10 Compton
per cell).

### 4.8 One backend option

No new keyword, plan field, config field or function name names a backend. Keys, register
caps and capacity are traits or automatic rules; `share!`, `free_bytes`, `launch_kernel!`
dispatch on the device. The executor is still chosen only by `resolve` (I11).

## 5. The CPU path

- `ThreadedExecutor` keeps `run_cascade!(::ThreadedExecutor)` and `chunk_generations!(c, r,
  bufs)` as they are: the key traits are not consulted, chunks stay contiguous, no `share!`,
  no register cap. CPU bits and CPU speed are unchanged by construction; CPU 1 against 16
  threads stays bitwise.
- The `HostDevice` pool (`:host_pool`, the test executor) takes the whole pool structure
  (keys, sort, permuted items, strided chunks). It costs that executor the key launches
  (re-emission for primaries); it is a test and debugging executor, and its bitwise
  agreement with threads (`test/pipeline/products_montecarlo.jl:351, 392`) is the CPU-side
  proof that ordering changes no bit.
- One measured option (item B8): the threaded cascade orders each generation's children
  by the spatial key before running them (host `sortperm` of a few thousand keys per task
  buffer), for plasma-cache locality (`primitives` is 20.5 % of the CPU thin profile,
  every lookup a cache miss, `PLAN.md:112-113`). Adopted only if the Compton phase on 32
  threads is at least 3 % faster in three interleaved runs; bitwise either way.
- CPU speed-ups for the same deck belong to arithmetic-level work outside this design
  (for example `march_control` re-evaluating the metric the previous sample evaluated, 37 of
  541 ns per step, `PLAN.md:109, 268`; generation, 21 % of the thin CPU profile); none of the
  items here slows the CPU.

## 6. Expected speed-up

### 6.1 Model

`T = W / R + τ`: `W` the warp-seconds of the phase, `R` the mean resident warps, `τ` the
unfilled tail of the last launches. Measured: thin `W = 2,033`, `R = 539` → 3.77 s (3.76
measured); Compton `W = 37,683`, `R = 606` → 62.2 s (62.2 measured).

Rules (estimates):

- Ordering scales a launch's `W` by `E_now / E_ordered` when the time per warp-step does not
  depend on the active lanes. That holds for the non-spawning launches: thin warp-steps
  take 29–31 µs at lane efficiencies 0.165–0.256, the full-warp imaging march 22 µs per
  warp-step on a lighter step (profile §4.3, map_march §6). For the spawning launches the
  factor lies between `E_now / E_ordered` (groups per warp-step unchanged) and 1 (groups
  growing with the lanes); the central value is the geometric mean.
- Lane overlap raises `R` to the bulk residency of a launch: 0.80 × 864 = 691 (range
  650–750: 668–750 in thin chunk 1, launch means 551–730). `τ`: 0.15 s thin, 1.0 s
  Compton.
- Emission is 0.20 of the thin `W` (410 warp-s; device generation rate 1.2–1.4·10⁶/s and
  the CPU share 21 %). A primary key re-emits: +410 warp-s.
- A pilot march at `C_pilot = C/8` costs `W_march · (1/8) · (16 µs / t_ws)`, with 16 µs per
  plasma-free warp-step (54 % of a thin step on the CPU: `rk4_step` 249 + `march_control` 44 of
  541 ns, `PLAN.md:107-109`): thin 108, Compton primaries 339, generation 1 41, 2 48, 3 184
  warp-s.

### 6.2 Thin phase, one A100

| step | W (warp-s) | T | arithmetic |
|---|--:|--:|---|
| today | 2,033 | 3.74 s | measured |
| lane overlap | 2,033 | 3.09 s (2.86–3.28) | 2,033 / 691 + 0.15 |
| + spatial primary key (re-emission) | 1,989 | 3.03 s | 410 + 410 + 1,623 × 0.201/0.279; gain 2 %: **not adopted** |
| + step key, E_p = 0.6 | 1,472 | 2.28 s | 410 + 410 + 108 + 1,623 × 0.201/0.6 |
| + step key, E_p = 0.8 | 1,336 | 2.08 s | 410 + 410 + 108 + 1,623 × 0.201/0.8 |
| two A100s, E_p = 0.6 / 0.8 | | 1.22 / 1.12 s | W / (2 × 691) + 0.15 |

Against 32 CPU threads (11.3 s): 3.0× today, 3.7× with overlap, 5.0–5.4× with step keys,
9.3–10× on two A100s. Break-even of a re-emitting step key: `E_p > 0.30`.

### 6.3 Compton phase, one A100

| step | W (warp-s) | T | arithmetic |
|---|--:|--:|---|
| today | 37,683 | 62.5 s | measured |
| no overflow reruns | 32,351 | 52.6 s | 62.2 − 9.6 s of rerun spans (the reruns ran alone) |
| + lane overlap | 32,351 | 47.8 s (44.1–50.8) | 32,351 / 691 + 1.0 |
| + spatial keys for the children | 27,258–30,936 | 40.4–45.8 s | 20,661 + 2,716 × [0.546, 1] + 4,929 × [0.504, 1] + 4,040 × 0.651 |
| + spatial key for the primaries | 21,759–31,346 (central 26,005) | 32.5–46.4 s (central 38.6) | primaries 20,661 × [0.714, 1] + 410 |
| step keys (E_p = 0.6) for primaries, generations 1, 2; spatial for 3 | 12,511–31,774 | 19.1–47.0 s | primaries 20,661 × [0.312, 1] + 410 + 339; generation 1 2,716 × [0.278, 1] + 41; 2 4,929 × [0.375, 1] + 48; 3 2,630 |
| two A100s: overlap / spatial / step keys | | 24.4 / 16.7–23.7 / 10.1–24.0 s | W / (2 × 691) + 1.0 |

Against 32 CPU threads (62.5 s): 1.0× today; 1.3× certain; 1.3–1.9× (central 1.6×) with
spatial keys; 1.3–3.3× with step keys; on two A100s 2.6×, 2.6–3.7×, 2.6–6.2×.

### 6.4 What stays out of reach of Tier 1

In the spawning launches the lanes run each step in 2–3.6 groups, generation 2 stalls on
instruction fetch, and their warp-steps cost 120–222 µs against 44 µs for generation 3,
the same compiled kernel on photons that cannot spawn. That factor 2.7–5 multiplies 28,306
of the 32,351 warp-seconds that remain without reruns. Tier 2 targets it (section 7).

## 7. Tier 2: code-shape items, gated

These move compiled arithmetic between contexts. On the device the context does not
decide the rounding (every `muladd` and flagged pair is fused, map_compiler §2), so GPU
bits are expected to hold; on the host the SLP vectorizer decides contraction per
context, so CPU bits can move (the non-inlined sampler moved them before,
`docs/src/design/performance.md` §4). Each item is prototyped, checked on the decks on both
backends, and adopted only when bitwise on the CPU, or by the user's decision to
re-baseline.

### 7.1 Deferred Compton kick (gate: F4)

- The march records an event instead of kicking: `step_events` (`track.jl:221-251`) keeps
  the event loop, the partial step, `scatter_site`, `child_stream`, the
  `ok && fits && W > weight_min` decision and the `spawned`/`unspawned` sums exactly as now,
  and hands `spawn!` a `KickRecord` in place of the kicked `Superphoton`: `xev` (32 B),
  `p_ev` (32), `W` (8), the zone fields the kick reads, `ucon` (32), `bcon` (32), `theta_e`
  (8), `gcov` (128), `nscatt` (4), `tau_a`, `tau_s` (16), the parent's diagnostics (40),
  plus the child stream (24): 356 B against 168 B per slot (2 M slots: 712 MB per side).
- `MCLayer` kicks first: `p_new, _, crng = scatter_superphoton(p_ev, zone, crng)` with
  identical arguments, builds the `Superphoton` exactly as `track.jl:238-242` does, and
  tracks it on the advanced stream. The draw sequence of every stream is unchanged: the
  kick consumes the prefix of the child's stream as before.
- Effect: the march kernel loses the four nested rejection loops of the sampler
  (`scatter.jl:50-278`) and most of the spills scattering adds to `track!` (1,704/2,568 B
  against 144/168 B in the thin kernel); the kicks of a generation run at the head of the
  child's launch, all lanes together. (The layer kernel's SASS holds 17 `YIELD`
  instructions against 5 in the thin kernel.)
- Estimate (if F4 attributes the penalty to the kick): a spawning warp-step costs the 44 µs
  of generation 3 × (1 + 2 step-equivalents per event × events per warp-step) + 6 µs for the
  uniform a spawning step draws: primaries 0.0024 events per warp-step → 50 µs (×0.41 of
  122), generation 1 0.032 → 53 µs (×0.40 of 131), generation 2 0.185 → 66 µs (×0.32 of
  207); ordering then acts as on non-spawning launches. Spatial keys: W = 20,661 × 0.41 ×
  0.714 + 410 + 2,716 × 0.40 × 0.546 + 4,929 × 0.32 × 0.504 + 2,630 = 10,476 → 16.2 s
  (3.9× CPU 32), two A100s 8.6 s (7.3×). Step keys (E_p = 0.6): W = 3,389 + 343 + 639 +
  2,630 = 7,001 → 11.1 s (5.6×), two A100s 6.1 s (10×).

### 7.2 Emission split for primaries (gate: CPU bits)

`MCEmit` (emit, `EMITTED`, store into the generation-0 buffer) followed by the generation-0
march as an `MCLayer` launch removes the re-emission of the primary key (−410 warp-s) and
runs emission at its own register count. Thin with step keys (E_p = 0.6): W = 410 + 108 +
544 = 1,062 → 1.69 s; two A100s 0.9 s. The emission arithmetic (`generate`, `emission_zone`
and its decode) moves into a new function body; the decode is known to be context-sensitive
on the host (map_compiler §3.1).

### 7.3 Decisions for the user (re-baselines)

1. **Portable arithmetic in the Monte Carlo physics** (explicit `fma`, ordered sums, BLP-owned
   elementary functions, checked by the portability gate): one re-baseline of the
   Monte Carlo decks after which every Tier-2 item, and any later stage split, is bitwise by
   construction on both backends, and CPU = GPU for the ported functions.
2. **Device site scan** (map_compiler §5): 2.0–2.96 s per uncached stride-1 call → about
   0.1 s; changes 661 of 104,776 zones (0.63 %) by at most 3 ulp in `theta_e` and `B`. Outside
   the photon phase; a cold thin call is 2.96 (scan) + 0.80 (tables) + 0.30 (offsets) + 3.74 s.

Not planned: trimming the diagnostics (`x1i … b0`, `tau`) from the device photon: 14 of 255
registers, a code-shape change for an expected gain under 5 %.

## 8. Risks and their detection

| risk | detection | response |
|---|---|---|
| spawning launches do not speed up when ordered (groups grow with the lanes) | F2 per-launch times with oracle and spatial order; ncu active threads per instruction after ordering | Compton stops at the certain part (47.8 s); Tier 2 or the stage-split design |
| ordering by steps scatters the memory accesses of a warp | ncu sectors per request, L1/L2 hit, long-scoreboard stalls of ordered launches (generations 2, 3 are memory-sensitive: 3.6 and 9.3 sectors per request, 0.94 and 2.11 long-scoreboard stalls) | meshblock as the minor key (built in); spatial key for generation 3 |
| two different MC kernels on one SM share its instruction cache | nsys/ncu "no instruction" stalls and warp time per step with overlap against without | overlap only launches of the same program (lane pairing) or keep serialized launches for the layer kernels |
| an unsynchronized shared array is freed or written while another lane reads it | `compute-sanitizer --tool memcheck` and `racecheck` on the stride-8 decks; bitwise decks on every run (an exact sum exposes any lost update) | `share!` applies only to residency twins (read-only after a synchronized upload) and atomic-only twins; lane-owned buffers keep CUDA.jl's synchronization |
| resident warps stay at 78–87 % of the slots while blocks remain (one-warp blocks) | per-SM residency probe (`profile/scripts/smcheck.jl`) | workgroup sweep 32/64/128 after ordering (B8): ordered warps finish together, so wider blocks cost little |
| host sort and staging exceed 2 % of a phase | nsys gaps between a lane's launches | device counting sort as programs (histogram by atomics, scan, scatter): a later item |
| the child budget is too small on a smaller or shared GPU | overflow count per call (log line from the cascade) | the rerun path stays correct; the budget only removes reruns |
| a kernel entry with Int32 items compiles the program bodies differently | decks bitwise on CPU (`:host_pool`) and GPU against main | the bodies are separate device functions (section 4.2); a difference would show which body inlined |
| the capped launch path depends on KernelAbstractions internals | `BLP_TEST_DEVICES=cuda` suite and the kernel audit on every toolchain change | the trait defaults to `nothing`; dropping the cap restores KA's launch |
| strided chunks lose emission-zone locality without keys | strided only when the primary has an order key | contiguous otherwise |
| pilot cost on long photons | pilot step cap (2,500 at C/8 for the 20,000-step march) | lower cap or spatial key |

## 9. What falsifies the design early

All four experiments run before any package edit: script-level method overrides in a
scratch environment copied from `/data2/haiyangw/scratch/wavefront/env`, jobs modelled on
`results/blp/gpu/run_job.sh`, bins compared with `compare.jl` against the plain run.

| id | experiment | pass | if it fails |
|---|---|---|---|
| F1 | capacity and overlap: the timing deck (stride 1, 10⁶, bias 10⁴, seed `0x5eed0700`) plain; with `capacity = 4_000_000`; plus `CUDA.enable_synchronization!(a, false)` on every array of the device twins after `program_twin!` | Compton ≤ 54 s with capacity, ≤ 50 s with both; thin ≤ 3.3 s with overlap; nsys shows kernels of both lanes concurrent; bins bitwise | overlap row removed (−0.7 s thin, −5 s Compton); per-lane twins (memory ×2) as the fallback test |
| F2 | ordering: a script `chunk_generations!(lane, …)` that reads each generation's children to the host, orders them (spatial key computed on the host; oracle key: the recorded step count of each photon by stream identity from `profile/data/probe_s1/*.bin`), and launches over the permuted slots; primaries by the same two keys | oracle-ordered generations 1 and 2 at least 1.5× faster than arrival order and spatial order at least 1.2×; non-spawning generation 3 within 15 % of the 1/E prediction | if oracle ordering gives under 1.3× on the spawning launches, ordering is dropped for them (Compton ≈ 46 s on one A100) and the design hands the spawning launches to Tier 2 or the stage split |
| F3 | step predictor, CPU only: record `(x₀, p₀, steps, outcome)` of every photon of the thin and Compton timing decks (the probe's control wrapper and recording sink, `profile/scripts/mc_probe.jl`, on 32 threads); run `pilot_steps` at C/4, C/8, C/16 and evaluate E per launch in key order | E_p ≥ 0.6 on thin primaries and generation 3 at C/8 | step keys dropped; spatial keys only; thin stays at 3.1 s |
| F4 | kick attribution on the Compton primaries (generation 0 only): `weight_min = Inf` (same steps, events and draws, no kick, no spawn) and a script stub `scatter_superphoton(p, zone, rng) = (p, 0.0, rng)` (kick code absent); probe warp time per step, ncu active threads per instruction on chunk 1 at Ns 1.6e5 | warp time per step ≤ 70 µs (from 122) and active threads within 1.3× of lanes marching in either variant | Tier 2.1 dropped: the penalty sits in the per-step scattering work; the spawning march needs the stage-split design |
| F5 | register caps: a script capped launch (the extension method of section 3.3) at 255/192/168/128 for thin `MCPrimary`, Compton `MCPrimary` and `MCLayer` | ≥ 5 % on a phase with bitwise bins | B7 not built |

F1 with F5 fits one GPU job and F4 another (each under 30 minutes after compilation); F2
needs one job of about 40 minutes; F3 needs no GPU.

## 10. Work breakdown

Roles for the agent team: **exec** (execution layer), **mc** (Monte Carlo programs),
**bench** (GPU jobs, profiles, timings), **cpu** (CPU-side analysis), **docs** (portability
and records). Every item lands on `gpu-wavefront` as one commit with the Chandra template,
after its acceptance passes.

| id | item | role | days | depends | acceptance |
|---|---|---|--:|---|---|
| A1 | F1 + F5 experiments | bench | 1 | — | the pass column of F1, F5; logs and nsys files under `plan/exp_incremental/` |
| A2 | F2 experiment | bench + mc | 1.5 | A1 (overlap override reused) | the pass column of F2; per-launch table: items, span, warp time per step, E |
| A3 | F3 experiment | cpu | 1 | — | E per launch for spatial keys reproducing profile §4.2 within 0.01, and for pilot keys at three C values |
| A4 | F4 experiment | bench + mc | 0.5 | — | the pass column of F4 |
| B1 | child slots from a device budget; growth kept per device and across calls | exec | 1 | A1 | Compton timing deck: 12 `MCLayer` launches per call (4 chunks × 3 generations, no rerun) on one A100; test suite incl. capacities 1, 2, 5; decks bitwise CPU and GPU against main; Compton ≤ 54 s |
| B2 | `share!` in the protocol, ext, residency, accumulators, child twins; a CUDA test that two lanes reading one shared array overlap | exec | 1.5 | A1 | nsys: summed kernel time ≥ 1.1 × union on the thin call; thin ≤ 3.3 s, Compton ≤ 50 s (with B1); memcheck and racecheck clean on `mc_torus_thin` and `mc_torus_compton` at stride 8; decks bitwise |
| B3 | `order_key` trait, `order.jl`, the cascade wiring incl. overflow reruns over permuted items | exec | 2 | — | `test/infra/execution_cascade.jl` extended: keys on and off, pools of 1, 2, 3, 5 host devices, chunks 37 and 211, capacities 1 and 2, bitwise against threads; `radix_order!` against `sortperm` (stable) on random keys |
| B4 | `coherence_key`, `MCPrimaryKey`, `MCLayerKey` (spatial), wired by the trait | mc | 1.5 | B3, A2 | `products_montecarlo.jl` host-pool tests bitwise; GPU decks bitwise against main; timing deck per-generation times within 15 % of F2's spatial leg |
| B5 | `pilot_steps` and the composite key | mc | 1.5 | B4, A3 pass | as B4, with F2's projected step-key times within 20 % |
| B6 | strided chunks (`K = 8·ndev`) for keyed pools | exec | 1 | B3 | chunk spans within 10 % of each other on the Compton deck; two-GPU job: bins bitwise one GPU, speed-up ≥ 1.8× |
| B7 | `register_cap` trait and the capped launch in the extension; kernel-audit cases at the adopted caps | exec | 2 | A1 (F5 pass) | decks bitwise; the adopted cap ≥ 5 % on its phase; `bench/kernel_audit.jl --check` updated |
| B8 | workgroup sweep after ordering (32/64/128); threaded child ordering measured on 32 threads | bench + cpu | 1 | B4 | decision recorded with times; bitwise |
| B9 | `port_hypot`, the portability gate | docs | 1.5 | — | CPU decks bitwise against main; GPU decks re-recorded with the CPU–GPU differences no larger than main's; gate test green |
| B10 | documentation as the standard: `docs/src/design/performance.md` (Monte Carlo execution on a pool: ordering, capacity, lane sharing, caps), `docs/src/design/architecture.md` (the traits and protocol functions), `docs/src/guide/gpu.md`; the record README with the measured tables | docs | 1 | B1–B8 | `test/structure.jl` green (exports with docstrings, files ≤ 600 lines, no history prose) |
| B11 | final validation (section 11) | bench | 1 | all adopted | every line of section 11 |
| C1 | deferred kick prototype and bit checks | mc | 2 | A4 pass | decks on CPU and GPU against main; CPU bits decide adoption or a request to the user |
| C2 | deferred kick adoption | mc + exec | 2 | C1 | as B4, Compton timing within 20 % of section 7.1 |
| C3 | emission split prototype and bit checks | mc | 1.5 | B4 | CPU bits decide adoption or a request to the user |

Order: A1–A4 in parallel (A1 and A4 can share one job); B1, B2, B3, B9 in parallel after A1;
B4 → B5; B6, B7, B8 after B4; C1 after A4; B10, B11 last. Critical path A1 → B3 → B4 → B5 →
B8 → B11: about 8 working days with the GPU queue.

## 11. Verification and timing protocol

Bits (every adopted item, and B11 in full):

1. Test suite on the host (`julia -t 4 --project test/runtests.jl`), then under
   `BLP_TEST_DEVICES=cuda` in a GPU job (`results/blp/gpu/pkgtest.sbatch` pattern).
2. The 13 fixed decks (`results/blp/mc_audit/speed/gpu/scripts/decks.jl`, compared by
   `compare_decks.jl`): branch against main on 16 CPU threads; branch on 1 against 16 threads;
   branch against main on one A100 (Tier 1: bitwise; after B9: re-recorded); branch CPU against
   branch GPU, differences no larger than main's; one A100 against two A100s (`--gres=gpu:2`),
   bitwise. Outputs under `DECKS_OUT=/data2/haiyangw/scratch/wavefront/decks`.
3. The stride-1 timing deck bins (`mc_phases.jl` legs, `compare.jl`) against main on the same
   backend, bitwise.

Timings:

- GPU: one A100 in a Slurm job on partition `preempt` with `-c 4`, `julia -t 4`, the job
  alone on the node (the node grants one job at a time); `mc_phases.jl <label> gpu 1 1e6 1e6
  nohost` for the photon phases (`run_layers` on prepared tables); one compiling call, then
  three warm calls, reported as best and median; one further call under `nsys` for the kernel
  union and the lane overlap. Two GPUs: the same with `--gres=gpu:2`.
- CPU: `julia -t 32` on `anta`, the same script with `cpu`, best and median of three, main and
  branch interleaved, load average recorded (the host scatters by about 15 %).
- Every timing table names the tree, the job id and the log; the A100 clock is 1065 MHz in
  every sampled range (profile §1).

## 12. Evidence produced for this design

`/data2/haiyangw/scratch/wavefront/plan/design_aux/` (the CUDA 13.3 `ptxas` of the Julia artifact
`3cd7a8e8…`, `cuobjdump` from `/usr/local/cuda-13`, offline, no GPU):

- `thin.ptx`, `layer.ptx`, `cprim.ptx`: copies of the profile's PTX dumps (thin `MCPrimary`,
  Compton `MCLayer`, Compton `MCPrimary`).
- `thin.cubin`, `layer.cubin` and `.sass`: the uncapped kernels (255 registers; stacks 3176 and
  5448 B, as profile §2.1); global-load and atomic flavours (section 1).
- `*_r168.*`, `*_r128.*`: the same at `--maxrregcount` 168 and 128: register counts, stack
  frames, `track!` spill bytes (section 1) and FP instruction counts (`DFMA` identical, section
  4.2).
- The count of unrounded FP operations in the MC PTX (0 `mul.f64`, 0 `fma.f64`; thin 5, Compton
  primary 9, layer 7 unrounded `add/sub.f64`, none with a multiply operand) is read from
  `profile/raw/devcode/{thin,compton}/*.asm`.
