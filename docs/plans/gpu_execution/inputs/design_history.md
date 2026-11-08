# History-based GPU execution of the Monte Carlo: design

Design of the device execution of the BlackLightPlus Monte Carlo (MC) as optimized
history-based transport: one march kernel per photon generation, run by persistent threads
that regenerate (a lane whose photon ends takes the next one at the next step), over photons
ordered by position and direction, with scattering events written to a deferred buffer that a
separate Compton kernel turns into the next generation, and with the emission and the Compton
sampler moved out of the march kernel. Measured facts come from `profile.md` (cited as P§),
`map_execution.md` (E§), `map_mcprograms.md` (M§), `map_march.md` (R§), `map_compiler.md`
(C§), `map_outputs.md` (O§) and `map_literature.md` (L§) in this directory. Numbers that are
derived rather than measured are marked **(estimate)**.

## 0. Scope and base tree

- Scope: the photon phase of `mc_spectrum` on a `PoolExecutor` (thin and Compton, grid and
  sphere sources). The site scan (host, 2 s at stride 1, C§5), the zone tables, imaging, the
  precise march and the window Monte Carlo (`mc_run_timed`, CPU threads only) are outside it.
  Two pieces of shared infrastructure (lane overlap, A3; register cap, A4) also apply to
  imaging and the precise march.
- Base tree. The profile and the maps are of `a5923dd`. `main` has since moved to `f6decd6`
  with Monte Carlo changes that matter here: `generate` is `@noinline` (b71d7b4: its `@simd`
  dot products round differently when inlined into another program), the emission tetrad
  takes another axis near a degenerate field (15f56a2: last-bit changes of the photons of some
  zones), and the window Monte Carlo (`montecarlo_time.jl`) whose `TimedPrimary` and
  `TimedLayer` call the same `track!`. `track.jl`, `cascade.jl`, `children.jl` and
  `programs.jl` are unchanged. The branch `gpu-wavefront` is rebased onto `main` before any
  work item, and every bitwise reference below is `main` at that base commit, re-recorded in
  item H0 on both backends. The deck records of the earlier speed work (`2c50d86`, `5d8e377`)
  are no longer references for the Monte Carlo decks.

## 1. Summary

Per chunk of primaries on a device lane, generation by generation:

```
Emit(chunk) ─► G0 ─► Order ─► March(G0) ─► E0 ─► Kick(E0) ─► G1 ─► Order ─► March(G1) ─► E1 ─► … ─► March(GK)
                               │ record!, tally! into the device totals (exact integer atomics)
                               └ overflow parents ─► March again into discarded scratch
```

- **Emit** (one thread per primary) runs `generate` and writes the photon and its advanced
  stream into the generation buffer `G0`, with a coherence key.
- **Order** is a counting sort of the generation's keys into a permutation (inner radius
  first, then position octant, then direction octant).
- **March** is one persistent launch per generation: as many threads as the device holds,
  each looping over march steps; a lane whose photon stops records it and claims the next
  photon of the permuted queue in the same loop. The Compton sampler is not in this kernel:
  an event writes a `ScatterEvent` (event-site plasma and metric, momentum, weight, child
  stream, child header) to the deferred buffer `Eg`.
- **Kick** (one thread per event) runs the Compton sampler on each event and writes child `j`
  into slot `j` of the next generation buffer, with its key.

What it removes, per the profile: the lane idling of the history kernels (lane efficiency
E = 0.20 thin, 0.19 Compton primaries, 0.30 layers, P§4.2), the split execution of spawning
warps (active threads per instruction 2.0–3.6 times below the lanes still marching, and
2.0–3.0 times the warp instructions per step of the non-spawning march, P§4.3), the
Compton sampler's spill code in the march (1704/2568 B against 144/168 B thin, P§2.1), the
overflow reruns (9.8 s of the 63.2 s Compton call, P§4.5) and the partly empty SMs (539 of
864 warp slots on average in the thin phase, P§4.4).

Expected photon phases on one A100 **(estimate, §7)**: thin 3.74 s → 0.6–0.8 s (4.8–6.1×);
Compton 62.5 s → 4.5–6.6 s if the deferred kick restores the warps' convergence (9.5–14×),
9.1–13 s if it does not (4.8–6.9×). The CPU keeps today's cascade and runs the same compiled
march at the same speed (§6).

The gains rest on one open question, settled first (§10): whether the device compiles a
restructured march to the same bits as `main`'s `track!`. The device rounds every operation
the source fixes and fuses every `muladd` and contract pair (llc within a block, ptxas across
blocks, C§2), and the
profile's probe programs, which add stores inside `march_control` and the sink inlined into
`track!`, gave spectra and ledgers bitwise equal to the plain run on the A100 (P§4, jobs
3796, 3803); the remaining freedom is the reassociation of `@simd` reductions (R§8, b71d7b4).
If the restructured march does not keep `main`'s bits, the bitwise-by-construction items of
track A remain (§7.4, about 1.5× on Compton) and the user decides on a one-time rebaseline
(§11).

## 2. What the profile requires of the design

| measured (source) | consequence for the design |
|---|---|
| lanes reconverge after the march loop; a finished lane waits for its warp's slowest photon; E = 0.201 / 0.187 / 0.300 (P§4.2, §4.3) | regeneration inside the step loop, not around it: an outer claim loop around `track!` keeps E (a lane leaving `track!` waits at the loop's reconvergence point) |
| E by step count (oracle) 0.99; by meshblock and octant 0.28 / 0.26 / 0.50 (P§4.2) | sorting alone bounds the gain at 1.4–1.7×; the lever is regeneration, sorting serves cache locality and the order of long photons |
| spawning launches issue 15,665 (primary), 10,735, 10,522 warp instructions per warp-step against 5,154 for the non-spawning layer 3 and 4,150 thin; ratios 3.04, 2.08, 2.04 match the measured lanes-marching / active-thread ratios 3.6, 2.2, 2.0 (P§4.3) | the spawning code splits a warp into groups that run the step body separately; the march kernel must hold no spawning code that prevents reconvergence (the deferred kick) and may need an explicit convergence point (B5) |
| the kick is 0.4 % of the Compton CPU profile (M§4; about 1.1 µs·thread per kick **(estimate)**) and rare per step (4.0e-4 events per primary lane-step, P§4.1) | the kick's own instructions are not the cost; its control flow inside the march is. A separate kick kernel costs ≤ 0.1 s **(estimate)** |
| every large kernel at 255 registers, 8 warps per SM; Compton `track!` spill code 12× thin, frame 6352 against 3176 B; 391 GB local traffic in the 2.0 s Compton primary launch (P§2.1–2.2) | the sampler and the emission leave the march kernel; a register cap is a separate, ptxas-only experiment (A4) |
| bulk residency 650–750 of 864 warps while blocks remain; 0.62–0.70 of the slots on average (P§4.3–4.4) | persistent launches fill every slot until the queue empties |
| the slowest photons set each launch's end: the longest thin warps span 0.53–0.57 s; tails 16–25 % (thin) and 39–44 % (layer 1) of a launch (P§4.3) | few, large launches (one chunk per device when memory allows), the longest photons first in the queue |
| 1,002,090 of 8,304,106 layer photons are second runs of an overflowing generation, 9.8 s (P§4.5) | overflow reruns only the parents whose events did not fit, into discarded scratch; capacity learned per device |
| the two lanes of a device do not overlap their kernels: they share device arrays that CUDA.jl synchronises across streams (P§5.3) | chunk policy that does not need two lanes (one chunk per device); lane overlap as separate infrastructure (A3) |
| thin primary: L1 hit 96 %, DRAM 0.3 % of peak; layers 2 and 3: 3.6 and 9.3 sectors per request, L1 73–74 %, long-scoreboard 0.94 and 2.11 per issue (P§2.2) | memory coherence matters for children; the sort key keeps the photons live at one time in a narrow window of the plasma |

## 3. Architecture

### 3.1 One chunk on a device lane

For a chunk `r` of primaries (§3.7) the lane runs, for `g = 0, 1, …, K` (`K = max_scatt`;
`K = 0` thin):

1. `g = 0`: **Emit** over `r` → `G0` (photon, stream, key per primary). `g ≥ 1`: `Gg` is the
   kick's output.
2. **Order**: `KeyCount` over the keys (atomic histogram), exclusive scan on the host
   (≤ 8,192 buckets, 64 KB each way), `KeyPlace` (atomic cursor per bucket) → `order`, a
   permutation of `1:|Gg|`.
3. **March** over the queue `order` of `Gg`: persistent, `W = resident_threads(dev)` threads
   (SMs × the most threads an SM holds, 221,184 on the A100; blocks beyond the residency start
   once the queue is empty and exit at once), claim counter reset to 0; every photon recorded and tallied into the device twins of the
   accumulators; events appended to `Eg` (capacity `ecap`, §3.4); `first_lost` filled with
   `typemax` beforehand.
4. Event count `n_ev` read back (one 8-byte copy, as `get_count` now). If `n_ev > ecap`: grow
   `Eg` to `n_ev` keeping its first `ecap` records, reset the event counter to `ecap`, compact
   the parents with `first_lost < typemax` into a list, and run **March** again over that list
   with `threshold = first_lost` into lane scratch accumulators, zeroed after (§5.4).
5. `n_ev == 0` or `g == K`: stop. Else **Kick** over `1:n_ev` → `G(g+1)`, then `g += 1`.

The children of a chunk never leave its lane, as in the cascade today (E§3, `cascade.jl:1-7`).
The returned child count is `Σ n_ev`, the cascade's return value.

### 3.2 The kernels

All device code is programs run by the one kernel `_foreach!` (`kernel.jl:7-10`; the
structure gate allows no second `@kernel`).

| program (module) | items | reads | writes | notes |
|---|---|---|---|---|
| `MCEmit` (Products) | primaries `g ∈ r` | sites, offsets, tables (read-only twins) | `G0.photons[j]`, `G0.rngs[j]`, `keys[j]` with `j = g − first(r) + 1`; ledger `EMITTED` | calls `emit(source, g, primary_rng(seed, g))`, i.e. the `@noinline generate` of `main` |
| `KeyCount`, `KeyPlace` (Execution) | `1:m` | `keys` | `counts`, `cursor`, `order` | integer only |
| `MCMarch` (Products) | `1:W` threads | `Gg`, `order`, plasma, tables | device totals (bins, ledger); `Eg`, `first_lost` | the regenerating loop of §3.3 |
| `MCKick` (Products) | `1:n_ev` | `Eg` | `G(g+1).photons[j]`, `.rngs[j]`, `keys[j]` | `kick(Eg[j])`; no atomics |
| `Fill` (Execution) | `1:m` | – | `first_lost`, `threshold` | `typemax` / zero |
| `Drain` (existing) | scratch length | scratch | zeroes scratch | after an overflow rerun only, `keep = false` |

### 3.3 The regenerating march

`track!` (`track.jl:276-342`) is a set-up (lines 278–291), a loop body (293–333) and an end
(335–341). The design names these as `@inline` physics functions in `MonteCarlo` over an
`isbits` state carrying exactly the loop-carried values of `track!`:

```julia
struct MarchState            # 200 B: the values track! carries across its loop
    y::SVector{8,Float64}; w::Float64; tau_a::Float64; tau_s::Float64
    na0::Float64; ns0::Float64; th0::Float64; r_prev::Float64
    it::Int; nev::Int; rng::MCRng
    loss::Float64; loss_final::Float64; spawned::Float64; unspawned::Float64; absorbed::Float64
end
march_begin(m::March, sph, rng)          -> MarchState                  # lines 281–291
march_step(m::March, sph, s::MarchState) -> (MarchState, outcome::Int)  # one iteration; LIVE = 0
march_end(m::March, sph, s, outcome)     -> (sph = out, recorded, rng)  # lines 335–341
```

`march_step` evaluates, in `track!`'s order, the step limit (`it > max_steps(ctl)` →
`DROPPED_MARCH`), `march_control` and the stop, the plunge test, the step, the sample, the
rates, the depths, the events and the decay, and returns `LIVE` with `it + 1` or the
outcome. `Cgr` and `rate_frame` move into `March` (the fixed inputs), computed once per photon
as now. The device program:

```julia
function (p::MCMarch)(t, i)                         # t ∈ 1:W, one thread
    live = false; item = 0; sph = NO_PHOTON; s = NO_STATE; m = p.march0
    while true
        if !live
            k = claim!(p.queue)                      # Atomix add on one Int64
            k > p.n && break
            item = Int(@inbounds p.order[k])
            sph = @inbounds p.gen.photons[item]
            m = with_sink(p.march0, at_item(p.sink, item))
            s = march_begin(m, sph, @inbounds p.gen.rngs[item])
            live = true
        end
        s, outcome = march_step(m, sph, s)
        if outcome != LIVE
            march_end(m, sph, s, outcome)            # record!, tally!
            live = false
        end
    end
    return nothing
end
```

One loop: a lane that finishes runs `march_end` and, at the next iteration, `march_begin` of
its next photon while the other lanes step; there is no inner loop whose exit would
reconverge the warp. A warp-iteration costs one step plus, in the lanes that start or end, a
start sample or a record. With a mean of 458 steps per thin photon, `1 − (1 − 1/458)^32 = 6.8 %`
of warp-iterations hold a start **(estimate: E' ≈ 0.9 with the end of the queue)**.

Convergence (item B5, conditional on measurement): if the march kernel still runs warps in
groups, the loop head becomes a warp vote, `mask = lane_vote(p.queue, mask, has_work)`,
which converges the lanes named in `mask` at that instruction and keeps every lane in the loop
until its whole warp has no work, so no lane exits while others still vote. `lane_vote` is a
device-protocol function dispatched on the array type of `p.queue`: `CUDA.vote_ballot_sync`
in the extension, `has_work ? mask : 0` on host arrays. `W` is then a multiple of 32 and the
program runs with `n = W`, so no lane of a warp is masked out by `i <= n`.

### 3.4 Deferred scattering events

`step_events` (`track.jl:221-251`) keeps everything that decides the parent's own state: the
event draws on the parent's stream, the partial RK4 to the event, the site sample
(`scatter_site`, whose validity `ok` decides `spawned` against `unspawned`), the ordinal and
`child_stream`. It replaces the kick and the spawn (lines 237–243) by

```julia
scatter!(m.sink, ScatterEvent(zone, momentum(yev), W, crng, stp.tau_a, stp.tau_s,
                              sph.nscatt + Int32(1), sph.x1i, sph.x2i, sph.ne0,
                              sph.thetae0, sph.b0), ordinal)
```

where `ordinal` is the `nev` that `child_stream` was called with; `spawned += W` follows
unchanged. `ScatterEvent{Z}` (MonteCarlo, 544 B) holds the
`EmissionZone` of the site (416 B, `x` the event point), the momentum, the child weight, the
child stream at counter 0, and the child's depths, order and diagnostics.
`kick(ev) -> (Superphoton, MCRng)` is `scatter_superphoton(ev.p, ev.zone, ev.rng)` and the
child `Superphoton(ev.zone.x, p_new, ev.w; …)`, the expressions of lines 237–242.
`scatter!` is a new method of the `Sink` protocol (`sinks.jl`): its default is
`spawn!(sink, kick(ev)...)`, the inline kick of today, and is what `MCSink` with a
`ChildBuffer`, `VectorSink` and `NullSink` do; `MCSink` whose children slot is an
`EventSlot` (the device staged path) stores `ev` instead:

```julia
@inline function defer!(s::EventSlot, ev, ordinal)       # Execution, events.jl
    b = s.buf
    ordinal < @inbounds(b.threshold[s.item]) && return nothing   # stored by the first pass
    slot = Atomix.@atomic b.count[1] += Int64(1)
    if slot <= length(b.events)
        @inbounds b.events[slot] = ev
    else                                                 # only this photon's lane writes s.item
        @inbounds b.first_lost[s.item] = min(b.first_lost[s.item], Int64(ordinal))
    end
    return nothing
end
```

The kick kernel reads each record once and writes child `j` at slot `j`: no atomics, and the
generation buffer it writes is allocated after `n_ev` is known, so it cannot overflow.

### 3.5 Coherence order

Key of a photon at `x` with covariant momentum `p` (Products, `photon_key`): the shell
`s = clamp(⌊(ln r − ln r_min)/0.05⌋, 0, S − 1)` of the coordinate radius `r = |x⃗|`, then the
octant of `x⃗`, then the octant of the spatial `k^i = g^{iν}p_ν`; key `= (s·8 + oₓ)·8 + o_k`,
`r_min` the horizon floor, `S = 128` (8,192 buckets). `MCEmit` and `MCKick` write it. Properties used:

- spatial and directional coherence: P§4.2 measures this family (radial shell then octant:
  E = 0.277 thin, 0.478 layers, against 0.279 and 0.500 by meshblock), without a provider
  lookup in the key;
- longest photons first: steps per primary fall with the start radius (Compton deciles 1,817
  → 153 steps, thin p99 4,545 → 446, P§4.1, `profile.json` `stats_extra`), so the inner shells
  enter the queue first and the photons that set a launch's end start early;
- a regenerating launch holds as many live photons as resident lanes (27,648 at 255
  registers: 108 SMs × 8 warps × 32), a window of
  consecutive queue entries, so the plasma read at any moment is a narrow range of shells and
  octants: the L2 working set of the layer launches shrinks.

Order inside a bucket is the arrival order of `KeyPlace`'s atomic cursor, not deterministic,
which the results do not see (I5, §5.6).

### 3.6 Register and code-size reduction

| change | effect on the march kernel | how it is measured |
|---|---|---|
| R1. the Compton sampler (tetrad, boosts, four nested rejection loops, ≥ 15 Philox draws) in `MCKick` | removes the code that multiplies `track!`'s spill code 12× and doubles its frame (P§2.1, R§4.6); the loops that likely keep spawning warps split (§2) | `ptxas -v` offline on the march kernel (host only, R§10); ncu active threads per instruction against lanes alive (H3, B4) |
| R2. emission (`emission_zone` decode, `generate`'s rejection loops, the zone search) in `MCEmit` | removes emission from the march kernel and from the regeneration path: a start is two loads and one sample | as R1 |
| R3. register cap per program (`max_registers(prog)`, honoured by an extension-level launch with `@cuda maxregs`, A4) | 168 → 12 warps/SM, 128 → 16 warps/SM, against spills; ptxas only, the PTX is unchanged | sweep on the decks and the timing protocol; keep a cap only where faster |
| R4. one march kernel for every generation | `MCPrimary` and `MCLayer` (two copies of `track!`) become one `MCMarch` per transport type; compile time of the Compton run about the same **(estimate)** | cold-call timing |

The march itself (RK4 with Dual metric, plasma sample and decode) sits at the cap in the thin
kernel (255 registers, P§2.1); R1 and R2 are not expected to lower the register count below
255 by themselves, only the spills, the frame and the instruction footprint. R3 is the
occupancy experiment.

### 3.7 Chunks, lanes, devices

- One device: one chunk of all primaries when the buffers fit (§4.4: about 5.5 GB at 10⁶
  Compton primaries), run by one lane; the other lane idles. A chunk is as large as possible because
  each launch lasts at least as long as its slowest photon (§7.3).
- `D > 1` devices: `D` chunks interleaved by item (`k:D:n`, every chunk samples every zone, as
  GPUmonty spreads each zone over its devices, L§3), so the devices are balanced without a
  fine queue; more chunks per device only when the buffers do not fit. The chunk queue of
  the pool is unchanged (lanes pull chunks).
- With lane overlap (A3) the policy can move to two chunks per device on two lanes; the
  choice is a tuning (B6), never a result.

### 3.8 What stays as it is

`foreach_cascade!`'s signature and contract, the generic pool cascade (`chunk_generations!`
on a lane, used by programs that declare no stages, the cascade tests' programs among them),
the threads cascade, `MCPrimary`, `MCLayer`, `MCSink`, `track!` as a function of its own (the
stationary, window and generic programs share its machine code), the accumulators, the
streams, `mc_spectrum`'s keywords, the site scan and the tables.

## 4. Changes by file

### 4.1 Physics (`MonteCarlo`, no KernelAbstractions, no threads)

| file | change |
|---|---|
| `src/physics/MonteCarlo/track.jl` | `MarchState`; `march_begin`, `march_step`, `march_end` (`@inline`); `track!` as their composition, kept non-inlined; `Cgr` and the rate frame in `March`; `step_events` calls `scatter!(m.sink, ScatterEvent(…), ordinal)` in place of lines 237–243. If item B1 shows that the composition moves the CPU bits, §11 decides |
| `src/physics/MonteCarlo/events.jl` (new, ~80 lines) | `ScatterEvent{Z}`; `kick(ev)` |
| `src/physics/MonteCarlo/sinks.jl` | `scatter!(sink, ev, ordinal)` in the protocol with the default `spawn!(sink, kick(ev)...)` |
| `src/physics/MonteCarlo/MonteCarlo.jl` | include `events.jl`; exports `ScatterEvent`, `kick`, `scatter!`, `MarchState`, `march_begin`, `march_step`, `march_end` with docstrings (module exports, not top-level) |

### 4.2 Execution

| file | change |
|---|---|
| `src/infra/Execution/events.jl` (new, ~150 lines) | `EventBuffer{E,C,F,T}(events, count, first_lost, threshold, layout)`; `EventSlot(buf, item)`; `defer!`; `child_stream(::EventSlot, rng, e)` as `ChildBuffer`'s; lane-buffer constructors; growth that keeps the first `ecap` records (allocate, `copyto!`, free) |
| `src/infra/Execution/coherence.jl` (new, ~120 lines) | `KeyCount`, `KeyPlace`, `Fill` programs; `counting_order!(lane, keys, m, nbuckets) -> order` (two launches and the host scan); a host method on threads (`sortperm`) for item C1 |
| `src/infra/Execution/stages.jl` (new, ≤ 400 lines) | `CascadeStages(emit, march, kick)`; the trait `cascade_stages(primary) = nothing`; `staged_cascade(dev)::Bool` (device protocol: `false` by default, `true` for `CUDADevice` in the extension, overridable by the internal keyword `staged` of `foreach_cascade!` for tests); `run_staged!(lane, stages, r, caps)`: the flow of §3.1, the swap adaptor `StagedSwap` that places the lane buffers (generation, order, events, counters, scratch) into the device twins without re-adapting (the `CascadeSwap` pattern, `cascade.jl:115-154`) |
| `src/infra/Execution/cascade.jl` | `run_cascade!(::PoolExecutor, …)` calls `run_staged!` when `cascade_stages(c.primary) !== nothing` and the pool's devices are `staged_cascade`; `cascade_chunk` gains the staged rule of §3.7 |
| `src/infra/Execution/devices.jl` | `resident_threads(dev)` (host: `32 × Threads.nthreads()`); `lane_vote(a, mask, flag)` (host: `flag ? mask : zero(mask)`) |
| `src/infra/Execution/kernel.jl` | `claim!(q) = Atomix.@atomic q[1] += Int64(1)`; launch trait `max_registers(prog) = nothing`; `launch!` passes it to `launch_capped!(lane, …)` when set (device protocol, A4) |
| `src/infra/Execution/residency.jl` | per-lane twins of a program's mutable accumulators; read-only twins shared per device (A3) |
| `src/infra/Execution/Execution.jl` | include `events.jl`, `coherence.jl`, `stages.jl` after `children.jl`; exports with docstrings |
| `ext/BlackLightPlusCUDAExt.jl` | `resident_threads(::CUDADevice)` = SMs × max threads per SM; `lane_vote(::CuDeviceArray, …)` = `CUDA.vote_ballot_sync`; `staged_cascade(::CUDADevice) = true`; `launch_capped!` (the `_foreach!` launch of `CUDAKernels.jl:107-150` with `maxregs`, A4); untracked conversion of `ReadOnlyArray` (A3) |

### 4.3 Products

| file | change |
|---|---|
| `src/pipeline/Products/programs.jl` → new `programs_mc.jl` (the 600-line budget) | `MCEmit(source, sink, seed, out)`, `MCMarch(transport, sink, gen, order, queue, n)`, `MCKick(events, out, keys)`; `cascade_stages(p::MCPrimary)` builds them from `MCPrimary`'s source, transport, sink and seed; `photon_key`; `at_item(sink, item)` (an `MCSink` whose children are an `EventSlot`; the type of the CPU's `MCSink` does not change); `scatter!(::MCSink{…,<:ChildBuffer}, ev, o) = spawn!(…, kick(ev)...)`, `scatter!(::MCSink{…,<:EventSlot}, ev, o) = defer!(…)`; `workgroupsize`, `stack_bytes` of the three programs |
| `src/pipeline/Products/montecarlo.jl` | none: `run_layers` calls `foreach_cascade!` with `MCPrimary` and `MCLayer` as now |

### 4.4 Data layout (per chunk on a lane)

All lane buffers (`lane_buffer`, device vectors, array of structures as `ChildBuffer` is).
Sizes for one chunk of 10⁶ Compton primaries (P§4.1: largest generation 5,951,363 photons).

| buffer | element | bytes | length | size |
|---|---|--:|---|--:|
| generation `G`, two sides | `Superphoton`, `MCRng` | 144 + 24 | largest generation | 2 × 1.0 GB |
| keys, order | `UInt32`, `Int32` | 4 + 4 | largest generation | 48 MB |
| bucket counts, cursors | `Int32` | 4 | 8,192 | 64 KB |
| events `E` | `ScatterEvent` | 544 | events of a generation (5.95 M for g = 2) | 3.2 GB |
| `first_lost`, `threshold` | `Int64` | 8 + 8 | largest generation (one entry per parent) | 95 MB |
| counters (claim, events, rerun list) | `Int64` | 8 | 3 | – |
| rerun list | `Int32` | 4 | ≤ parents | ≤ 24 MB |

About 5.5 GB per lane at one chunk of 10⁶ (thin: 168 MB). The event capacity is capped at a
budget of the free device memory (default a quarter); beyond it the overflow path of §5.4
runs, slower and with the same result.

### 4.5 Tests, benchmarks, documents

| file | change |
|---|---|
| `test/infra/execution_stages.jl` (new) | toy staged programs (emit, persistent march with regeneration, kick) on `Decks.hosts(1, 2, 3, 5)` with `staged = true`: totals, child sets (checksum of stream identities, as `execution_cascade.jl`) and count bitwise equal to the generic cascade and to threads; capacities 1 and 2 (overflow and rerun every generation); chunks 1, 37, n; `n < W`; `n = 0`; depth 0 |
| `test/infra/execution_coherence.jl` (new) | `counting_order!` is a permutation with non-decreasing keys on every executor |
| `test/physics/montecarlo_track.jl` | per photon, `track!` equals `march_begin`/`march_step`/`march_end` driven by a host loop, bitwise; `scatter!`'s default equals the former inline kick (children and streams bitwise); `kick` of a `ScatterEvent` equals `scatter_superphoton` |
| `test/pipeline/products_montecarlo.jl` | unchanged assertions; a staged leg on a host pool (`staged = true`) whose bins equal threads within 1e-12 relative and whose counts and child count are equal |
| `test/infra/execution_cuda.jl` | staged Monte Carlo on one and two CUDA devices: bitwise each other and the generic pool cascade |
| `bench/kernel_audit.jl` | cases `mc_emit`, `mc_march_thin`, `mc_march_scatter`, `mc_kick` |
| `docs/src/design/performance.md` §1, §6; `architecture.md` §3.5, §4.2, §5.2 | the staged cascade on devices, the march pieces, `ScatterEvent`, the measured phases |

## 5. Invariants

### 5.1 Per photon: the same operations in the same order

A photon's floating-point work is the function of its entering `Superphoton` and stream
alone (I1, E§6). In the staged march it executes `march_begin`, then `march_step` until the
outcome, then `march_end`: the statements of `track!`'s set-up, of each loop iteration in
order, and of its end, on the same values. What differs is (a) which thread runs it and when
(claim order, warp, SM), which no operation reads; (b) values stored and reloaded at the
stage cuts (`Superphoton`, `MCRng`, `ScatterEvent`), which are exact `Float64` copies; (c)
the compile context of the step body (inlined into `MCMarch`'s loop instead of `track!`'s),
which on the device can move bits essentially only through the reassociation of `@simd`
reductions and through whether ptxas fuses a contract pair that LLVM left split across blocks
(C§2, R§8, b71d7b4). (c) is
the one point that needs a test: H2 (thin) and H3 (Compton) compare, on the A100, per-photon
end states keyed by stream identity and the deck bins against `main`. The physics is never
re-sampled across a cut: the kick reads the event site's state stored by the parent, never a
new sample of the site; a photon's start sample is its own `march_begin` sample, as in
`track!` line 283.

The per-photon running sums `loss`, `loss_final`, `spawned`, `unspawned`, `absorbed` live in
`MarchState` and are tallied once in `march_end` (I7, M§3.5). `spawned` and `unspawned`
depend on `ok && fits && W > weight_min`, all known in the march; the kick changes neither.
`nev` counts ordinals in the parent's march as now; the parent's stream is not drawn by the
kick (the kick draws on `crng`).

### 5.2 Exact accumulators

`ExactBins.record!` (`accumulators.jl:161-198`) and `ExactScalars.add!` (`:425-429`) add the
integer limbs of each addend with integer atomics (`exactsum.jl:79-90`). Every escaped photon
gives the same `(E, w, dir)`, so each cell receives the same multiset of addends and its exact
integer value is the same; `Spectrum` rounds it once on the host. The ledger stripe of an
addition is the photon's slot in its generation (`at_item` keeps `(item − 1) mod 64`) instead
of its primary index or layer slot: the additions move between stripes, every read sums all
stripes (I8, `programs.jl:275-289`), and the addend counts (`ESCAPED`: `n_recorded`) are
unchanged. The raw limbs differ between any two runs already (P§4, probe fidelity); every
comparison is of rounded sums. `EMITTED` is added in `MCEmit`, the other ledger quantities in
`march_end`, the same values once each. No floating-point atomic and no float partial sum
across photons is introduced (I4).

### 5.3 Genealogy-keyed streams

`MCEmit` draws `primary_rng(seed, g)` (`philox.jl:192-193`) for primary `g` and stores the
stream advanced by `generate`; the march continues from it, as `MCPrimary` passes it from
`emit` to `transport!` now. An event's child stream is `child_rng(rng, nev, layout)` drawn in
the march (`children.jl:60-61`), stored at counter 0 in the `ScatterEvent`, advanced by the
kick and stored with the child, whose march continues from it, as `spawn!` stores it now
(`track.jl:237, 243`). The layout is `widest_layout(N, K)` as now. No stream depends on a
slot, a thread, a lane or a device (I3).

### 5.4 Event-buffer overflow

- A parent whose append finds no slot completes its march with the same arithmetic (the
  append does not feed back) and records the smallest lost ordinal in `first_lost[item]`;
  only its own lane writes that index, so no atomic is needed.
- The event counter only grows, so once an append of a parent fails, all its later appends
  fail too: its stored events are exactly the ordinals below `first_lost[item]`, its lost
  events exactly those at or above.
- The rerun is the same compiled `MCMarch` (the same type: scratch arrays are lane arrays of
  the accumulators' element types and shapes, as `CascadeSwap` makes them) over the parents
  with `first_lost < typemax`, with `threshold = first_lost`. Each re-marches identically (same
  photon, same stream, same ordinals) and appends exactly its lost events, `n_ev − ecap` in
  all, into the grown buffer. Its record!/tally! additions go to lane scratch, zeroed by
  `Drain(keep = false)` and never added to the totals, so every addition of the call counts
  once (I6).
- The set of stored events is then the set of all events of the generation, the set of
  children the inline kick would have spawned. Their slot order differs, which nothing reads
  (I5).
- The generation buffers are written by the kick at the event's slot after `n_ev` is known
  and never overflow; `first_lost`, `threshold` and the rerun list have one entry per parent.

### 5.5 Devices

- One GPU against two: each chunk's photons, events and children are the same sets whatever
  device runs the chunk; both A100s run the same compiled programs (same architecture, I10);
  the per-device twins are folded by exact integer addition (`merge!`). The chunk layout
  (interleaved, `D` chunks) depends on `D`, and results do not depend on chunking (I2) because
  every item's stream is keyed by its global index.
- `resident_threads`, workgroup sizes, register caps and the chunk policy change scheduling and
  register allocation only; ptxas does not change the rounding of `.rn` operations (C§2).
  Each is verified on the decks anyway (A4, B6).

### 5.6 Order

Sorted order, bucket arrival order, claim order and slot order decide only which thread
processes which photon and when (I1, I5). The counting sort reads integer keys; a key may be
any function of the photon, and a wrong key costs speed, never bits.

### 5.7 The CPU path and the CPU–GPU tolerance

The threads executor runs the cascade of today (§6). CPU 1 against 16 threads stays bitwise
by the existing argument (exact sums, keyed streams, `run_growing!`). CPU against GPU keeps
its present agreement (equal counts, sums within the recorded tolerances) as long as the GPU
bits equal `main`'s GPU bits, which H2, H3 and B4 test.

### 5.8 The backend option

No plan, configuration or function name carries a backend: the staged path is chosen inside
`run_cascade!(::PoolExecutor, …)` by the program's `cascade_stages` and the devices'
`staged_cascade`, both internal. `backend`, the session default, `BLP_BACKEND` and `:auto`
resolve executors exactly as now (I11).

## 6. The CPU path

- Code: `ThreadedExecutor` keeps `run_cascade!(::ThreadedExecutor, …)`
  (`cascade.jl:191-264`): chunks of `cld(n, 64t)` primaries, depth-first per chunk, two
  task-local `ChildBuffer`s, `run_growing!` reruns of single overflowing items, `MCPrimary`
  and `MCLayer` with `track!` and the inline kick (`scatter!`'s default).
- It does not share the staged structure. A CPU thread already regenerates (`run_items!`
  takes the next item when a photon ends), has no lanes to idle, spends 0.4 % of the Compton
  profile in the kick (M§4), and its compiled bits follow the inlining context (C§3.1, the
  host SLP and fusion decisions), so the fused march is both the fast and the bit-stable form
  on the host. The literature finds history-based transport faster on CPUs in every study
  (1.8–4×, L§2.1).
- It shares the physics: `track!` is the composition of the same `march_*` pieces and
  `scatter!` the same protocol, if B1 shows the CPU decks bitwise at 1 and 16 threads and the
  window-MC tests unchanged; the source then has one home for the march.
- Speed: the compiled `track!` is unchanged when the decks are bitwise, so the CPU speed is
  unchanged; B1 times it (protocol §9.1) and accepts at most noise (best of three interleaved
  runs within 5 % of `main`'s best on 32 threads, two rounds).
- Optional C1: a host sort of each generation's children by `photon_key` before the layer
  (`chunk_generations!(c, r, bufs)`, `cascade.jl:219`), for the cache locality of the plasma
  reads (20.5 % of the thin CPU profile in `primitives`, M§7). Bitwise by I5; kept only if
  faster on 32 threads.
- Host pools (`HostDevice`, the KA CPU backend) are test vehicles. They run the generic pool
  cascade (bitwise threads, as tested now) unless `staged = true` is passed; the staged MC
  programs on a host pool agree with threads to rounding, not bitwise, because the step body
  is inlined into another loop on the host.

## 7. Expected speed-up

### 7.1 Model and its calibration

For a launch, `T = W·τ/R` with lane-steps `L` (Σ steps over photons), lane efficiency `E`,
warp-steps `W = L/(32E)`, the time per warp-step `τ` (P§4.3: median over warps of span over
the warp's largest step count) and the mean resident warps `R` (P§4.3–4.4). On today's
launches (measured inputs; `L` from P§4.1 means × counts):

| component | L | E | W = L/(32E) | τ | R | model T | measured |
|---|--:|--:|--:|--:|--:|--:|--:|
| thin primaries | 458 × 1,000,005 = 4.58e8 | 0.201 | 7.12e7 | 30.3 µs | 539 | 4.00 s | 3.68 s (kernels, P§2.4) |
| Compton primaries | 1084 × 1,000,005 = 1.084e9 | 0.187 | 1.81e8 | 124 µs | 640 | 35.1 s | 32.6 s |
| layer 1 | 413 × 359,280 = 1.48e8 | 0.167 | 2.78e7 | 130 µs | 430 | 8.4 s | 5.5–7.8 s |
| layer 2 | 226 × 991,373 = 2.24e8 | 0.225 | 3.11e7 | 200 µs | 600 | 10.4 s | 8.1–9.5 s |
| layer 3 | 181 × 5,951,363 = 1.08e9 | 0.367 | 9.17e7 | 44.3 µs | 695 | 5.85 s | 4.4–7.1 s |

(Measured layer times: four launches per generation at 1.38–1.95, 2.03–2.38 and 1.09–1.78 s,
P§2.4.) The model is 0–25 % high; the estimates below use it uncorrected.

### 7.2 Assumptions **(estimate)**

- `E' = 0.90`: starts in 6.8 % of thin warp-iterations at about one step each, ends at a
  fraction of a step, the end of the queue (§3.3). Intra-step branch divergence stays as
  measured (lanes-alive / active-threads ratio 1.05 thin, 1.1 layer 3).
- `R' = 864`: persistent launch, every warp slot until the queue empties.
- `τ'` thin `= 30.3 µs × f`, `f ∈ [1.0, 1.3]`: a warp-step at 32 active lanes issues more
  sectors per load than at 6.4 (5.4 sectors per request today, P§2.2); thin is not
  memory-bound (L1 96 %, DRAM 0.3 %), so `f` is small.
- `τ'` scattering marches under **H-conv** (the deferred kick lets warps reconverge every
  step): `44.3 µs` (layer 3: the scattering march without spawning code, 5,154 instructions
  per warp-step) `× (1 + 1.1·p_ev) × f`, `f ∈ [0.9, 1.3]` (sorting lowers the memory stalls,
  more lanes raise traffic), `p_ev = 1 − e^{−32ν}` the probability that a warp-iteration holds
  an event, `ν` events per lane-step (P§4.1: 0.43/1084, 2.77/413, 6.03/226), an event path
  (partial RK4, site sample, record store) about 1.1 steps. Without H-conv: today's `τ × f`.
- Extra kernels: kicks ≤ 0.08 s (7.3e6 kicks × 1.1 µs·thread at ≥ 100 thread-equivalents; the
  A100 runs coherent march code at about 760, R§6); emission ≤ 0.05 s; sorts, fills and host
  round trips ≤ 0.02 s; reruns 0.

| launch | `p_ev` | `τ'` (µs), H-conv | `τ'` (µs), no H-conv | `W' = L/(32·0.9)` | bulk `T' = W'τ'/864`, H-conv | no H-conv |
|---|--:|--:|--:|--:|--:|--:|
| thin | – | 30.3–39.4 | – | 1.59e7 | 0.56–0.73 s | – |
| Compton primaries | 0.013 | 40.4–58.4 | 112–161 | 3.76e7 | 1.76–2.54 s | 4.86–7.02 s |
| layer 1 | 0.193 | 48.3–69.8 | 117–169 | 5.15e6 | 0.29–0.42 s | 0.70–1.01 s |
| layer 2 | 0.574 | 65.0–94.0 | 180–260 | 7.78e6 | 0.59–0.85 s | 1.62–2.34 s |
| layer 3 | 0 | 39.9–57.6 | 39.9–57.6 | 3.74e7 | 1.73–2.49 s | 1.73–2.49 s |

### 7.3 The floor of a launch

A launch lasts at least as long as its slowest photon: `L_max × t`, with `L_max = 20,000`
(the step limit, reached by 8 thin, 784 Compton-primary, 37 layer-1 and 14 layer-2 photons;
layer 3 at most 11,511, P§4.1) and `t` between a lone warp's step (about 15 µs **(estimate)**)
and the contended 27–30 µs of a thin warp-step (the longest thin warps span 0.53–0.57 s,
P§4.3, which is 20,000 steps at that rate): 0.3–0.6 s, layer 3 0.17–0.35 s. With the inner shells first (§3.5) the longest
photons start early and their floor overlaps the bulk. Per launch `T ≈ max(bulk, floor)`.

| chunk policy, one A100 | thin | Compton, H-conv | Compton, no H-conv |
|---|--:|--:|--:|
| one chunk of all primaries (proposed) | max(0.56–0.73, 0.3–0.6) + 0.05 = **0.61–0.78 s** | 1.76–2.54 + max(0.29–0.42, 0.3–0.6) + max(0.59–0.85, 0.3–0.6) + 1.73–2.49 + 0.15 = **4.5–6.6 s** | 4.86–7.02 + 0.70–1.01 + 1.62–2.34 + 1.73–2.49 + 0.15 = **9.1–13.0 s** |
| two chunks, two lanes not overlapping (today's lanes) | 2 × max(0.28–0.37, 0.3–0.6) = 0.6–1.2 s | 4.8–7.6 s | 9.0–13.2 s |
| four chunks, not overlapping (today's chunking) | 4 × max(0.14–0.18, 0.3–0.6) = 1.2–2.4 s | 6–10 s | 9.5–14 s |

### 7.4 Results

| photon phase | today, A100 | design, A100 **(estimate)** | ratio | today, 32 CPU threads | design against 32 threads |
|---|--:|--:|--:|--:|--:|
| thin, torus stride 1, 10⁶ | 3.74 s | 0.61–0.78 s | 4.8–6.1× | 11.3 s | 14–19× |
| Compton, H-conv | 62.5 s | 4.5–6.6 s | 9.5–14× | 62.5 s | 9.5–14× |
| Compton, no H-conv | 62.5 s | 9.1–13.0 s | 4.8–6.9× | 62.5 s | 4.8–6.9× |
| thin `mc_spectrum` call repeated (zones kept) | 4.1 s | 1.2–1.4 s (0.53 s host before the kernels, P§5.1) | 3× | 10.5 s | 8× |
| two A100s, Compton, H-conv (interleaved chunks, even split) | – | per device max(bulk/2, floor) per generation: 2.4–3.8 s | 1.7–1.9× over one A100 | | |

Not included: the register cap (A4), whose sign is unknown (Shift gained 1.7–4.9× from
occupancy, GPUmonty lost at a 32-register cap, L§2.2). The variant without H-conv still
removes the reruns, the idle lanes and the empty slots; H-conv decides the factor of about two
between the rows.

Bitwise-by-construction items alone (track A, if H2/H3 refute the restructured march's bits):
removing the reruns (−9.8 s), ordering the layer launches by key (layers ∝ 1/E: 0.167 → 0.305,
0.225 → 0.446, 0.367 → 0.564, P§4.2: 21.0 s → 11.7 s) and persistent warps around the
unchanged `track!` call in `MCLayer` (residency 0.62–0.70 → about 0.9) give Compton
62.5 → about 38–41 s (1.5–1.6×) and thin about 1.1–1.3× **(estimate)**.

## 8. Risks and their detection

| # | risk | detection | response |
|---|---|---|---|
| 1 | the restructured march or the separate kick compiles to other device bits than `main`'s `track!` (reassociated `@simd` reductions) | H2/H3: per-photon end states (keyed by stream identity) and per-child records, sorted by stream identity, against `main` on the A100; deck bins | stop track B; track A; §11 decision |
| 2 | composing `track!` from the pieces moves the CPU bits | B1: CPU decks at 1 and 16 threads, window-MC tests, `montecarlo_track.jl` | keep `track!`'s source; §11 decision |
| 3 | warps still run the step in groups after the deferral (H-conv false) | H1 (kick never executed), H3 and B4: ncu `smsp__thread_inst_executed_per_inst_executed` against lanes alive from the probe; warp instructions per warp-step against layer 3's 5,154 | B5, the voted loop head |
| 4 | launch floors dominate (single long photons) | probe `%globaltimer` stamps: time from the queue's last claim to the launch end; nsys SM-active samples | larger chunks; B8, a step budget per launch with the `MarchState` carried to the next launch |
| 5 | 32 active lanes raise memory stalls (τ' high) | ncu long-scoreboard per issue, sectors per request, L2 hit on the march kernel | sort key with meshblock rank instead of position octant (B6) |
| 6 | regeneration overhead (starts, ends) above 10 % | ncu instructions per step against the probe's step counts | warp-wide starts when half the lanes are idle (voted, B5) |
| 7 | overflow rerun miscounts or loses children | toy tests at capacity 1 and 2; deck child counts; `w_children` | – |
| 8 | voted loop deadlocks or hangs on partial warps | GPU unit test with `n ∈ {1, 31, 33, W − 1}`, `W` multiple of 32 | `n = W` always, lanes never exit before their warp |
| 9 | claim-counter contention | ncu stall on the atomic | warp-aggregated claim (one atomic per warp), in the extension |
| 10 | event buffers exceed the memory budget | counted events against capacity per generation | the overflow path; more chunks |
| 11 | the extension-level launch (A4) or the untracked read-only conversion (A3) breaks with a KernelAbstractions or CUDA.jl update | version-pinned tests in `execution_cuda.jl` | the generic launch stays the fallback |
| 12 | compile time of the cold call grows | cold-call timing in the protocol | none required; reported |
| 13 | the structure gates (one kernel, no `CUDA.` in `src`, 600 lines, exports with docstrings) | `test/structure.jl` in every item | – |

## 9. Work breakdown

Scratch for every item: `/data2/haiyangw/scratch/wavefront/history/{proto,scripts,logs,decks}`;
the environment is a copy of `/data2/haiyangw/scratch/wavefront/env` (CUDA runtime 13.3 in
`LocalPreferences.toml`) developed against the worktree; the reference tree is
`/data2/haiyangw/scratch/wavefront/ref_main` moved to the base commit, with an environment of
its own (not `env_ref`, which points at the main checkout, whose tree has moved). GPU jobs follow `results/blp/gpu/run_job.sh` and the `*.sbatch` files of the main
checkout (partition `preempt`, 1 h, 4 CPUs, `JULIA_PKG_OFFLINE=true julia -t 4 …`).

### 9.1 Protocols

- **Bitwise decks.** `results/blp/mc_audit/speed/gpu/scripts/decks.jl` and `compare_decks.jl`
  (13 decks), run for the reference and the branch: CPU at 16 threads and at 1 thread, the
  A100 in one job; plus the stride-1 thin and Compton runs of `mc_phases.jl` compared with
  `compare.jl` (every cell of every order, ledger totals, child counts). Pass: every deck
  bitwise on each backend; CPU against GPU within the reference's own CPU–GPU differences.
- **GPU timing.** One job per measurement set, reference and branch interleaved in the job
  (reference, branch, reference, branch); per leg one cold call (reported apart) and three
  warm calls, best and median; the photon phase as `mc_phases.jl` times it; one warm call
  under `nsys` for kernel sums and the SM clock (P§1: 1065 MHz); `nvidia-smi` idle check at
  the start.
- **CPU timing.** 32 threads, reference and branch interleaved, best of three, two rounds,
  load average logged (the host scatters about 15 %); "no slower" means the branch's best
  within 5 % of the reference's in both rounds.

### 9.2 Track H: falsification first (no package edits)

| item | days | work | accepted when |
|---|--:|---|---|
| H0 | 1 | rebase `gpu-wavefront` onto `main`; reference decks at the base on CPU (1, 16 threads) and the A100; the per-photon recorder (the probe sink of `profile/scripts/mc_probe.jl` extended to store each photon's end state, ledger fields, `nev` and stream identity; children by stream identity); the timing scripts | two reference runs bitwise on decks and per-photon records |
| H1 | 1 | (a) offline: PTX of `track!` in `MCLayer` over a range, over a staged item list and inside a claim loop, compared function by function; SASS of `track!` (cuobjdump) for the convergence barriers (`BSSY`/`BSYNC`) around the event region. (b) A100: the Compton primary generation with `weight_min = Inf` (no kick executed, same draws) under ncu as P§2.1 (Ns 1.6e5): warp instructions per warp-step and active threads per instruction | report with the decision of §10 |
| H2 | 2 | thin prototype as a script: `MarchState` and the pieces (copies of `track!`'s statements), `MCEmit`, the key and the counting sort, the persistent regenerating `MCMarch`, `resident_threads` from CUDA attributes; on the A100: per-photon records against H0 for `mc_torus_thin` and `mc_sphere_thin`; bins of `mc_torus_s1` and of the stride-1 phase; thin phase timing; claim-to-end stamps | go: bitwise everywhere and thin phase ≤ 1.5 s; stop otherwise (§10) |
| H3 | 2 | Compton prototype: H2 plus `ScatterEvent`, the deferred append with `first_lost`, `MCKick`, the overflow rerun; per-child records and bins against H0 for `mc_torus_compton` and `mc_sphere_scatt`; capacity forced to 1/8 of the events to exercise the rerun; ncu on the march kernel (active threads / lanes alive, instructions per warp-step); offline ptxas of march and kick | go: bitwise and Compton phase ≤ 20 s; B5 joins the critical path if active / alive < 0.6 |

### 9.3 Track A: shared infrastructure, bitwise by construction (in parallel with H)

| item | days | work | accepted when |
|---|--:|---|---|
| A1 | 1 | child capacity learned per device and shared by its lanes, first guess `max(MIN_CHILD_SLOTS, 8 × parents)` within the memory budget, in today's pool cascade | decks bitwise both backends; Compton phase −8 s or more on the A100 |
| A2 | 2 | `coherence.jl` (keys, counting sort), `photon_key`, today's `MCLayer` launched over the sorted item list; H1(a)'s PTX identity | decks bitwise; layer launches faster than today's sum (21.0 s) on the A100; CPU untouched |
| A3 | 2 | lane overlap: per-lane mutable twins, untracked read-only conversion in the extension | `profile/scripts/lanes_micro.jl` and an nsys trace show the two lanes' kernels overlapping; decks bitwise; no regression of thin, Compton, torus imaging, crossings on the A100 |
| A4 | 2 | `max_registers(prog)` and `launch_capped!` in the extension; kernel-audit cases; sweep 255/168/128 for the MC kernels (today's and, after B3, the staged ones) and the precise march | decks bitwise at every cap; a cap kept only where faster |
| A5 | 1 | only if track B stops: persistent warps in today's `MCLayer` (a claim loop around the unchanged `transport!` call; `track!` stays a function of its own) and, verified by decks, in `MCPrimary` | H1(a)-style PTX identity of `track!`; decks bitwise; layer launches faster on the A100 |

### 9.4 Track B: the staged history march (after H2 and H3 pass)

| item | days | work | accepted when |
|---|--:|---|---|
| B1 | 2 | `MonteCarlo`: `MarchState`, the pieces, `ScatterEvent`, `kick`, `scatter!`; `track!` composed of them | CPU decks bitwise at 1 and 16 threads; window-MC tests and the full suite green; CPU timing no slower (§9.1); else §11 |
| B2 | 2 | `Execution`: `events.jl`, `claim!`, `resident_threads`, `stages.jl` with the rerun path, the `cascade_stages` and `staged_cascade` traits | `execution_stages.jl` and `execution_coherence.jl` green on `Decks.hosts(1, 2, 3, 5)`; structure gates |
| B3 | 2 | `Products`: `MCEmit`, `MCMarch`, `MCKick`, `cascade_stages(::MCPrimary)`, `at_item`, `scatter!` methods | `products_montecarlo.jl` green (generic path bitwise, staged host leg to 1e-12); zero allocations of the programs on the host; `device_isbits` of the three programs |
| B4 | 1 | A100 jobs: decks; stride-1 phases; one against two A100s; ncu and nsys of the march | decks and stride-1 runs bitwise against the reference GPU; one = two GPUs bitwise; thin phase ≤ 1.2 s, Compton ≤ 13 s (the upper ends of §7) |
| B5 | 2 | conditional: `lane_vote` in the extension and the voted loop head | active / alive ≥ 0.8 on the march; decks bitwise; Compton faster than B4 |
| B6 | 1 | tuning: chunk policy (one, two per lane), workgroup size, `W`, key variants (position octant against meshblock rank) | decks bitwise for every setting; defaults set by the timing protocol |
| B7 | 1 | documents (performance.md §1, §6; architecture.md §3.5, §4.2, §5.2), kernel-audit baseline, bench legs; commits by the Chandra template | docs build; structure gates; review |
| B8 | 2 | contingency: a step budget per launch, the `MarchState` of photons over budget carried to the next generation's launch | only if B4 shows floors above 25 % of a phase; decks bitwise |
| C1 | 1 | optional: host sort of each generation's children on threads | CPU decks bitwise; kept only if faster at 32 threads |

Order and team: H0 first; H1, H2 and A1, A2 in parallel (one agent on the device prototypes,
one on the execution layer, one on verification and timing); H3 after H2; B1–B3 in parallel
after the go of H3 (physics, execution, products); B4 gates B5–B7. About 28 agent-days in all,
12 on the critical path (H0, H2, H3, B1/B2/B3 in parallel, B4, B5, B6, B7).

## 10. What would falsify the design early

| test | when | outcome that falsifies | consequence |
|---|---|---|---|
| H1(b) kick never executed (`weight_min = Inf`) | day 2 | warp instructions per warp-step of the Compton primary stay above 2.5 × 5,154 | the deferral alone does not restore convergence: B5 (voted loop head) on the critical path, and the event-site path (partial RK4 and sample) becomes the suspect; estimate falls to the no-H-conv row |
| H1(a) PTX identity | day 2 | `track!`'s PTX differs between the range launch and the staged item list | even track A's ordering is not bitwise by construction; A2 verified by decks only |
| H2 bits | day 4 | any photon of the thin decks differs from `main` on the A100 | the restructured march does not keep `main`'s device bits: track B stops; track A proceeds; §11 |
| H2 speed | day 4 | thin phase above 1.5 s with bitwise results | regeneration does not pay (32-lane warp-steps cost more than 1/E saves): the history design's main lever fails; the wavefront designs take over |
| H2 floor | day 4 | more than 30 % of the launch after the last claim | floors dominate: B8 moves into track B's core, i.e. towards a wavefront with carried state |
| H3 bits | day 6 | any child (by stream identity) or bin differs from `main` | the separate kick kernel does not keep the sampler's bits: deferral needs §11 |
| H3 convergence | day 6 | active threads per instruction below 0.6 of lanes alive in the march | B5 required before B4's targets |
| B1 bits | after go | CPU decks or window tests differ | §11 |

## 11. Decisions for the user

1. **If H2 or H3 show other device bits than `main`** (risk 1): either keep `main`'s device bits
   and take track A only (about 1.5× on Compton), or accept one rebaseline of the Monte Carlo
   bits on both backends in which `march_step` and `kick` become functions of their own,
   called by `track!`, the window programs and the staged march alike (the pattern of
   `generate` in b71d7b4). After it, every schedule (staged or not, host pool or device) runs
   the same machine code per backend, and further restructuring cannot move bits. The physics
   is unchanged; the CPU pays one call per step (to be timed; the CPU must stay no slower).
2. **If B1 shows that composing `track!` moves the CPU bits** (risk 2): the same choice for the
   CPU: the compile-once rebaseline, or two homes of the march body (`track!` verbatim and the
   pieces for the device), which the package's "one concept, one home" standard rejects.
3. **Host pools**: the staged MC programs on a `HostDevice` pool agree with threads to rounding
   (§6). The design keeps host pools on the generic cascade so that `mc_spectrum` on
   `:host_pool` stays bitwise threads, as the suite asserts; a staged host leg is a tolerance
   test. A rebaseline under decision 1 would make it bitwise.
