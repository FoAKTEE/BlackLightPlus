# Fast GPU Monte Carlo transport and coherent ray processing: literature map

Scope: GPU algorithms for Monte Carlo (MC) particle and photon transport and for coherent
ray processing, read against the BlackLightPlus (BLP) Monte Carlo: superphotons with
Poisson in-step scattering, variable lifetimes, an AMR plasma in device memory, Float64
arithmetic, and bitwise reproducibility across thread counts, backends and device counts.
Every citation in the reference list (section 8) has been checked against a publisher
record, an arXiv page or the paper's full text; the level of each check is stated there.
Numbers marked *full text* were read in the paper itself, *abstract* only in its abstract
or record. BLP source references are to the worktree `/data2/haiyangw/scratch/blp_wavefront`
at `a5923dd` (identical to main).

## 1. The BLP Monte Carlo in the literature's terms

- **History-based with branching.** One program call carries one superphoton from birth to
  escape, capture or the step cap (`track!`, `src/physics/MonteCarlo/track.jl:276`; RK4
  steps, at most `CURVED_MAX_STEPS = 20_000`, `track.jl:23`). Each step samples the plasma
  at its end (`sample(provider, …)`, `track.jl:306`), so every step is an AMR lookup and a
  plasma decode in Float64.
- **In-step branching with the Compton kick inline.** A step with scattering depth draws a
  Poisson number of events; each event re-integrates a partial RK4 step, samples the event
  site and runs the Compton kick, whose electron and Klein–Nishina rejection loops are
  capped at 10⁵ and 10⁶ iterations (`step_events`, `track.jl:221–252`, the kick at
  `track.jl:237`; caps at `scatter.jl:7,9`). The kick therefore lives inside the parent's
  march loop: its registers and its divergent loops are part of every thread of the
  scattering march.
- **Generations as layers.** `foreach_cascade!` (`src/infra/Execution/cascade.jl:67`) runs
  generation 0 over a chunk of primaries (≤ 2²⁰ per device chunk) and then one layer program
  per generation over that chunk's children, buffers swapped per generation. This is the
  generation-synchronous scheme GPUmonty uses (section 3) and a coarse form of event-based
  scheduling: one event type ("track a photon of generation g") per launch.
- **Reproducibility by construction.** Streams are Philox keyed by ancestry
  (`primary_rng(seed, g)`, `child_stream` per event ordinal) and the bins are exact integer
  sums (`ExactBins`), so the totals do not depend on which thread runs which photon or in
  what order. This is the property that makes reordering, compaction and regrouping free of
  statistical consequences; the literature's codes mostly lack it (section 4).
- **Measured symptom.** On one A100 against 32 CPU threads (`results/blp/mc_audit/speed/gpu/
  README.md` in the main checkout): thin photon phase 11.3 s vs 3.74 s, Compton photon phase
  62.5 s vs 62.5 s. The Compton phase tracks 1e6 primaries plus 7,302,016 children. Taken
  per tracked photon (lengths of children and primaries differ, so this is a consistency
  check, not a measurement), the CPU spends about 0.7× the thin per-photon time in the
  Compton phase and the A100 about 2×: the scattering code path roughly triples the GPU's
  cost relative to the CPU's. That is the signature the GPUmonty and Shift papers attribute
  to register pressure and low occupancy (sections 2.2, 3).

## 2. Techniques

Summary (details and conditions in the subsections):

| technique | fixes | reported effect (conditions) | cost | fit for BLP (section 6) |
|---|---|---|---|---|
| event-based (many small kernels, queues) | divergence, registers | GPU: event 6× faster than history (OpenMC, continuous energy); history 3–3.5× faster (Quicksilver, multigroup); CPU: history 1.8–4× faster | state round-trips through memory, queue upkeep, launches | coarse form only |
| kernel flattening (split the hot kernel) | register-limited occupancy | Shift CE: 168 regs → 46–62 regs, occupancy 12.5 % → 50–62.5 %, tracking ×1.7–4.9 over the cached history kernel (K40, P100) | more launches, more state traffic | high, if profiling confirms registers |
| history-length truncation + consolidation | lifetime spread (warps wait for the slowest) | up to 2.5× (Shift MG, 252 groups, sort every ~10 collisions); Celeritas step cap 25–35 % (A100) | survivor state store, compaction | high |
| persistent threads, atomic work counter | hardware scheduler tail, block-level imbalance | 1.5–2.2× (ray traversal, GTX 285, 2009); ~1 % on NVIDIA (MCX-CL, 2018) | negligible | low gain, no risk |
| regeneration / compaction inside a kernel | idle lanes after termination | 1–5 % (Laine 2013), 12–16 % (Wald 2011) | coherence loss, divergent init code | low |
| wavefront (Laine 2013) | divergence across materials, registers, I-cache | +36 %, +79 %, +221 % on three scenes (K20); thread utilization 23 % → 53 % | 212 B state per path in DRAM, queues | stage split at the kick |
| fine wavefront (Dr.Jit) | — | 2.0–7.7× *slower* than megakernel on GPU, >100× slower on CPU | memory traffic | rejected |
| sorting by key | memory incoherence of lookups | large when a lookup dominates and is memory-bound (OpenMC XS lookup; Eisenacher 5–20×, out of core); net loss when lookups are cheap (Quicksilver MG; Celeritas by action) | sort per iteration | test only after profiling the plasma sample |
| hardware reordering (SER, DWF) | divergence | vendor-claimed 2–3× for some RT workloads (Ada+, ray-tracing pipelines only) | not available on A100 in CUDA compute | none |

### 2.1 History-based versus event-based Monte Carlo

- **Origin.** Brown and Martin (1984) formulated event-based ("vectorized") MC for vector
  supercomputers: particles are banked by the next event they need, and each event type is
  processed for the whole bank as a vector operation. They report at least 20–40× over
  CDC-7600 scalar runs on the CYBER-205 without loss of accuracy (*abstract*). The GPU
  literature inherits the vocabulary: *history-based* (one thread follows one particle to
  termination) and *event-based* (one kernel per event type over the particles that need it).
- **Multigroup: history wins on GPUs.** Hamilton, Slattery and Evans (2018) find history-
  based tracking outperforms event-based tracking for multigroup MC on K40 and P100, and
  that a history loop that occasionally filters dead particles beats both (section 2.3); one
  GPU matched 20–40 CPU cores (K40) and 60–80 (P100) (*full text*). Ma et al. (2021),
  Quicksilver on K80: history-based is ~1.8× faster than event-based on CPUs and 3.1–3.5×
  faster on GPUs; event-based moves more data (140 vs 95 GB/s on the GPU), sorting before the
  cross-section kernel costs more than it saves, and replacing atomic queue pushes by a
  prefix sum doubled event-based speed (*full text*).
- **Continuous energy: event wins on GPUs.** Hamilton and Evans (2019), Shift on K40/P100/
  V100: event-based tracking significantly outperforms history-based for continuous-energy
  transport; one P100 matched ~90 CPU cores and one V100 more than 150 (*full text*). Tramm
  et al. (2024, building on Tramm et al. 2022), OpenMC on Intel, NVIDIA and AMD GPUs:
  event-based is typically ~6× faster than history-based on GPUs, history-based ~4× faster
  on CPUs; periodic sorting of particles by material and energy before the cross-section
  lookup gives large bandwidth gains in that kernel (*full text*). Bergmann and Vujić (2015,
  WARP) built a continuous-energy event-based GPU code; its 2017 benchmark puts WARP at
  0.8–7.6× of production CPU codes (*record and secondary summary*).
- **Thin threads.** Bleile et al. (2019): in Quicksilver, moving from "fat threads" (one
  thread advances a vault of many particles) to one light thread per particle, with
  atomics for shared tallies, gave ~10× on GPUs (*full text*).
- **What decides it.** The literature is consistent: event-based pays on GPUs when one
  event kernel is expensive and memory-bound (continuous-energy cross-section lookup, upwards
  of 80 % of OpenMC's runtime as cited by Bleile et al. from the XSBench work; event-GPU
  beats history-GPU by 1.2–1.5× once Ma et al. add such lookups to Quicksilver), because sorting and specialization
  then concentrate memory traffic; it loses when per-event work is cheap and the extra state
  traffic dominates (multigroup). On CPUs history-based wins in every study above. A design
  that must speed up both backends cannot simply switch to event-based execution.

### 2.2 Register pressure, occupancy and kernel flattening

- **Shift.** Hamilton and Evans (2019) report that the largest gain came not from history →
  event but from *flattening* the event-based "move to collision" kernel into several small
  kernels. Registers (ptxas): history kernel 168, event "move to collision" 127, flattened
  "path length tally" 62 and "distance to collision" 46, giving theoretical occupancy 12.5 %,
  25 %, 50 % and 62.5 %. Tracking rates (10³ n/s, fresh-fuel SMR, inactive / active cycles),
  history with caching → event → flattened: K40 12.5/3.7 → 19.5/5.8 → 34.8/18.0; P100
  57.6/19.2 → 73.7/24.7 → 97.1/49.3, so flattened over history ×1.7–4.9. Capping the flattened kernels at 12.5 % occupancy by
  allocating dummy shared memory drops K40 active tracking from 18.0 to 8.2; the event-based
  kernel at 12.5 % is slower than history-based. Their conclusion: occupancy, not reduced
  thread divergence, is the leading driver (*full text*).
- **GPUmonty** (section 3) sees the same limit in a grmonty port: 12.5 % theoretical
  occupancy in every tracking kernel; capping registers at 32 raises occupancy to ~100 % but
  spills, memory throughput rises to 68.9 % and runtime grows (*full text*).
- **Counterpoint.** Volkov (2010, GTC talk) shows kernels reaching higher throughput at
  lower occupancy through instruction-level parallelism (e.g. SGEMM at 33 % occupancy 1.6×
  faster than at 67 %) (*slides via mirror*). Low occupancy is a symptom only when the
  kernel is latency-bound, which MC tracking is (memory-latency-bound per Hamilton and Evans;
  "complex control flow, instruction latency, and special-function units" per GPUmonty).
- **A100 arithmetic** (NVIDIA A100 whitepaper, *full text*): 65,536 32-bit registers per SM,
  at most 255 per thread, at most 64 warps and 32 thread blocks per SM, 192 KB combined L1 and
  shared memory per SM, FP64 peak 9.7 TFLOPS (19.5 with FP64 tensor cores). A Float64 value
  takes two registers. Warps resident per SM by registers per thread (ignoring allocation
  granularity): ≤32 → 64 (100 %), 64 → 32 (50 %), 128 → 16 (25 %), 168 → 12 (19 %), 255 → 8
  (12.5 %). With one-warp blocks (BLP's Monte Carlo programs, README) the 32-blocks-per-SM
  limit caps occupancy at 50 %, which binds only below 64 registers per thread.

### 2.3 History-length truncation, consolidation and step caps

- **Truncated histories.** Hamilton, Slattery and Evans (2018, Alg. 4): each thread follows
  its particle for at most N collisions; the kernel ends; surviving particles are
  consolidated through an index list (no particle data moved); the next launch restocks the
  warps. On C5G7 the 252-group problem runs 2.5× faster at the best interval (~10) than the
  pure history limit, and the run time is flat for intervals below 100; the 8-group problem is
  insensitive. Caveat in the paper: when divergence occurs *within* one movement (many
  boundary crossings between collisions in optically thin regions), truncating on
  collisions does not help (*full text*). The truncation must therefore count the unit that
  varies, which for BLP is march steps, not scatterings.
- **Celeritas.** Lund et al. (2025, A100 on Perlmutter): capping magnetic-field substeps
  per iteration (paused tracks resume next step) gives up to 35 % (1000 → 10 substeps, CMS
  Run 2 test) and up to 25 % in full CMS geometries; 98 % of steps converge in fewer than 10
  substeps (*full text*).
- **Ray traversal.** Aila and Laine (2009) limit inner traversal loops to N = 4 iterations
  between work fetches when replacing terminated rays (*full text*).
- **Consolidation mechanics.** Shift and Quicksilver compared sorting full particle data
  (prohibitive), scan-and-scatter of indices, and atomic appends to per-event index buffers
  inside the kernels; atomics match scan-and-scatter at large counts and win at small ones
  (Hamilton et al. 2018; Hamilton and Evans 2019 then sort the consolidated *indices* for
  locality) (*full text*). Laine et al. (2013) aggregate queue appends per warp with ballot
  and population count: +40 % overall, +32 % in ray casts from the better ordering alone
  (*full text*).

### 2.4 Persistent threads and dynamic work distribution

- Aila and Laine (2009): launch just enough threads to fill the GPU and let them fetch work
  from a global pool through an atomic counter. On GTX 285 packet traversal became 1.5–2.2×
  faster and reached 10–20 % of a simulated upper bound; the hardware work distributor of the
  time was the bottleneck for rays of very different lengths (*full text*).
- Gupta, Stuart and Owens (2012) define the persistent-threads style and find up to an
  order of magnitude in micro-benchmarks, and losses in many cases (*abstract*).
- GPUmonty fetches photons through `atomicAdd` on a global counter in its sampling, tracking
  and scattering kernels (section 3). MCX-CL (Yu et al. 2018) balances photons dynamically
  within a workgroup: ~1 % on NVIDIA GPUs, 13 % on AMD (*full text*).
- Cost is negligible and, under keyed streams and exact sums, results cannot depend on the
  fetch order. Expected gain on current NVIDIA hardware is small; it removes the tail of a
  launch, not the divergence inside a warp.

### 2.5 Regeneration and compaction of terminated work

- Novák, Havran and Dachsbacher (2010): regenerate a new path in a lane whose path ended
  (*abstract*). van Antwerpen (2011): stream compaction plus regeneration for path tracing,
  BDPT and MLT on the GPU (*abstract*). Wald (2011): compacting partially filled warps into
  full ones raises SIMD utilization substantially, but hardware limits of the time leave net
  speed-ups of 12–16 % (*abstract*).
- Laine et al. (2013): regenerating when more than half a warp idles gives 1–5 %, because
  initialization code runs at low utilization and regeneration lowers the coherence of the
  warp (*full text*). Aila and Laine (2009): replacing terminated rays helps in some cases and
  hurts when SIMD efficiency is already high; no setting was consistently better on GTX 285
  (*full text*).
- For BLP, regeneration means running emission sampling (rejection over frequency) inline
  in a divergent lane, the case Laine et al. found worth moving to a separate "new path"
  kernel (99 % thread utilization there).

### 2.6 Wavefront path tracing, and its limits

- **Laine, Karras and Aila (2013)** (*full text*). A pool of 2²⁰ paths lives in global
  memory (212 B per path, SoA); each iteration a logic kernel advances every path one
  segment, then per-material kernels and ray-cast kernels run over compact queues filled by
  warp-aggregated atomics. On a Tesla K20: 42.99 → 58.38 Mpaths/s (+36 %, one material),
  5.41 → 9.70 (+79 %), 2.71 → 8.71 (+221 %, three expensive materials). Thread utilization in
  the hardest scene rises from 23 % (megakernel) to 53 % (logic 60 %, new path 99 %, materials
  71 %, ray casts 35 %). SoA over AoS: +80 % overall. The gain even with one material comes
  from running ray casts at low register counts and from the instruction cache, which cannot
  hold a megakernel. They name participating media (short steps through a separate data
  structure) as a case for a specialized volume-marching kernel. Enlarging the pool from 1M
  to 8M paths gains ~5 %.
- **Dr.Jit (Jakob et al. 2022)** (*full text*): a JIT that can emit either one megakernel or
  wavefronts of smaller kernels exchanging data through memory. Wavefront-style execution
  reduces GPU performance by 2.0–7.7× and raises CPU cost by more than 100×, attributed to
  the memory traffic (CPU ~95 GiB/s vs GPU ~715 GiB/s on their machines). This is the
  fine-grained end (per-operation kernels); Laine's coarse stages are the other end.
- **Reading.** The benefit of a wavefront comes from isolating a register-heavy or
  divergent stage so that the rest runs at high occupancy, at the price of keeping state in
  memory. The split points must be few and coarse; on CPUs any split that spills state to
  memory per step is a loss.

### 2.7 Sorting and reordering for coherence

- **Spatial queues.** Pharr, Kolb, Gershbein and Hanrahan (1997) queue rays in voxels of the
  scene and schedule voxels by what is in the geometry cache, rendering scenes of millions of
  primitives with a tenth of the scene in memory (*full text*). This is the graphics
  ancestor of queueing photons per AMR meshblock.
- **Ray sorting on GPUs.** Garanzha and Loop (2010): hash, compress, sort and decompress
  rays into coherent packets, then breadth-first frustum traversal; 3× over an optimized
  single-kernel depth-first tracer (*abstract*). Moon et al. (2010): reorder rays along a
  space-filling curve of estimated hit points; more than an order of magnitude on massive,
  out-of-core models (*abstract*). Meister et al. (2020): ray-reordering keys for wavefront
  path tracing on RTX GPUs give 1.3–2.0× trace speed, but the reordering overhead is hard to
  recover when the trace is hardware-accelerated (*abstract*).
- **Sorted deferred shading.** Eisenacher, Nichols, Selle and Burley (2013), Disney
  production renderer on CPUs: sort ray batches of 30–60 M rays for traversal coherence, then
  sort hits by mesh and face for shading; growing batches from 2¹⁵ to 2²⁵ rays makes
  traversal 5× and shading 20× faster; the city scene takes 11.2 min sorted and 819 min
  unsorted, ~70× (*full text*). The gain is out-of-core texture and geometry access.
- **Shader sorting.** Hoberock, Lu, Jia and Hart (2009): sorting shading requests by shader
  costs time but improves shading in every scene except those with simple shading
  (*abstract*).
- **MC particle sorting.** OpenMC sorts by material and energy before the cross-section
  lookup, with vendor sort libraries (Tramm et al. 2024, *full text*). Shift sorts
  consolidated indices for locality (Hamilton and Evans 2019). Celeritas (Lund et al. 2025,
  A100): sorting track indices by along-step action cuts instructions in that kernel by up to
  38 % but raises global memory operations by up to 344 %, a net slowdown in general;
  partitioning track storage by particle type (charged vs neutral) gives up to 30 %, less in
  complex geometry (*full text*). Quicksilver multigroup: sorting before the cross-section
  kernel costs more than it saves (Ma et al. 2021, *full text*).
- **Similar-state grouping for adaptive integrators.** Niemeyer and Sung (2014), one thread
  per ODE system with adaptive Runge–Kutta–Cash–Karp on a GPU: assigning random instead of
  neighbouring (similar) initial conditions to adjacent threads slows integration by up to
  2.3×, the threads of a warp then taking different numbers of sub-steps (*full text*).
- **Reading.** Sorting pays when one lookup dominates the runtime and its memory accesses
  become coherent after sorting; it loses when the sorted kernel is cheap or the sort itself
  moves data. Sorting indices, not state, is the standard.

### 2.8 Hardware divergence mitigation

- Fung, Sham, Yuan and Aamodt (2007): dynamic warp formation regroups threads after
  divergent branches, 20.7 % average in simulation for an estimated 4.7 % area (*abstract*);
  it is a hardware proposal.
- NVIDIA Shader Execution Reordering (2022 whitepaper): `ReorderThread` in ray-tracing
  pipelines (DXR, Vulkan, OptiX) regroups threads by a key on Ada and later GPUs; NVIDIA claims
  2–3× for some ray-tracing workloads (*whitepaper and secondary analysis*). It is neither
  exposed to plain CUDA compute kernels nor accelerated on the A100, so it does not apply here.

### 2.9 GPU photon Monte Carlo codes outside astrophysics

- **MCX** (Fang and Boas 2009): one thread runs photon after photon (time-resolved photon
  migration in voxelized turbid media); above 300× over a CPU with 1792 threads on a
  low-cost card when fluence is accumulated with non-atomic, racy additions, 75× with atomic
  operations (*abstract*). **MCX-CL** (Yu et al. 2018): profiling shows the kernel spends 62 %
  of its time in thread divergence; native math and tuned thread counts give 3–12 % on NVIDIA
  GPUs (more on AMD), while simplifying the control flow to reduce divergence gives 1.9× on
  one NVIDIA GPU and costs 1–7 % on others; CUDA MCX is 2.1–5.4× faster than OpenCL on
  NVIDIA; atomic fluence costs ~5 % on recent NVIDIA GPUs (*full text*). The design accepts
  races and vendor fast math, both excluded by BLP's invariants.
- **GRay** (Chan, Psaltis and Özel 2013): per-ray geodesic integration on GPUs in single
  precision, ~1 ns per photon per step and two orders of magnitude over CPU codes
  (*abstract*): the megakernel-per-ray design for pure ray tracing in Kerr.
- **Ensemble ODEs in Julia.** Utkarsh et al. (2024, DiffEqGPU.jl): generating one GPU kernel
  per whole ODE integration (adaptive steps inside) matches hand-written CUDA and is 20–100×
  faster than vectorized-map (array-per-stage) ensembles in JAX and PyTorch (*abstract*).
  This is the evidence that for an integrator such as BLP's precise march, the
  trajectory-per-thread megakernel is the right base and stage-level wavefronts are not.

## 3. GPUmonty (arXiv:2602.13198; code `ref-code/gpumonty`, HEAD 21a00ce)

Paper: Naethe Motta, Nemmen and Joshi (2026), CUDA/C port of grmonty (thermal synchrotron,
Compton scattering with bias, Verlet geodesic integration with an iterated predictor).

**Design** (paper section 3; code references relative to
`/data/haiyangw/claude/Mahakala/ref-code/gpumonty`):

- Five kernels: generation, sampling, tracking, scattered tracking, recording. Generation is
  stride-parallel over zones; sampling and tracking assign one photon per thread and fetch
  the next photon with `atomicAdd` on a global counter (`src/kernels.cu:706`, `:852`,
  `:916`): persistent threads with dynamic work fetch.
- Scattering is not recursive: a scattering event appends the scattered photon's state to a
  per-layer SoA array through `atomicAdd(&d_num_scat_phs[round_scat], 1)`
  (`src/track.cu:218–240`); after all photons of a layer finish, `track_scat` processes the
  next layer, first running the Compton kick `scatter_super_photon` and then tracking
  (`src/kernels.cu:893–946`, kick at `:928`). Up to `MAX_LAYER_SCA = 5` layers
  (`src/config.h:335`). The kick is thus *deferred* to the child's own kernel, unlike BLP's
  inline kick (section 1).
- SoA photon storage (`struct of_photonSOA`, `src/decs.h:136`); plasma primitives read
  through a texture object (`src/kernels.cu:842`, `:897`); spectrum by double-precision
  `atomicAdd` per bin (`src/track.cu:373–392`); a separate record kernel bins escaped photons.
- Memory batching: photons are split into serial batches that fit device memory; only the
  recorded contributions survive a batch. A bias-tuning loop re-tracks a layer when the
  scattered-to-incoming ratio misses its target (`docs/bias_tuning.rst`).
- Multi-GPU: OpenMP thread per device, each device takes a slice of every zone's photons so
  that inner (scattering-heavy) and outer zones are shared evenly; spectra reduced on device
  0 (`docs/multi_gpus.rst`).
- Build: `--use_fast_math --maxrregcount=255 -dlcm=cg` (`makefile:142`), 256 threads per
  block, `N_BLOCKS` set to the device's resident limit (`src/config.h:42,49`,
  `GetGPUBlocks.mk`).
- Randomness: cuRAND XORWOW, one state per resident thread (`src/defs.h:221`), seeded from
  the wall clock and the thread index (`src/kernels.cu:559–563`). With dynamic photon fetch
  and floating-point atomics, results differ from run to run; reproducibility is not a goal.

**Performance** (A100 40 GB SXM against a 64-core EPYC 7763 node running igrmonty;
SANE iharm3d snapshot 288×128×128; *full text*):

- Speed-up rises with work and plateaus at ~12× for Ns ≳ 10⁶ (Ns ≲ 10⁴: parity; a dip at
  Ns = 10⁷ when two memory batches are needed). At Ns = 5×10⁷: 1.6×10⁹ superphotons in 898 s
  (1.78×10⁶ /s) against 1.4×10⁹ in 10,906 s (1.31×10⁵ /s). Laptop RTX 3050 against a Ryzen 5
  6600H: ~8×.
- Kernel time: tracking 42 %, scattered tracking 57 %. Theoretical occupancy 12.5 % in every
  tracking kernel (achieved ~100 % of that); compute throughput 17–18 %, memory throughput
  25–26 %. Capping at 32 registers: occupancy ~100 %, compute throughput ~10 %, memory
  throughput 68.9 % from spills, longer runtime. The authors conclude that higher occupancy
  "would require substantial refactoring … into multiple smaller kernels" and leave it to
  future work.

**Reading for BLP.** GPUmonty already has the generation-layered structure and the deferred
kick, and still sits at 12.5 % occupancy: deferring the kick alone does not shrink the
tracking kernel enough when the tracker itself (geodesic step, plasma decode, opacities,
the hot-cross table, the bias, the event test) is register-heavy. Its 12× against 64 cores
is measured without any reproducibility constraint and with fast math. BLP's thin phase (3×
against 32 threads) and Compton phase (1×) have the same profile class.

## 4. Reproducibility mechanisms in the literature

- **Per-particle streams.** Brown (1994) gives arbitrary-stride skip-ahead for linear
  congruential generators so that each particle history starts at a fixed offset of one
  sequence, the basis of reproducible parallel MC in production codes (*record*). Salmon,
  Moraes, Dror and Shaw (2011) introduce counter-based generators (Philox, Threefry): a keyed
  bijection of a counter, with at least 2⁶⁴ streams of period ≥ 2¹²⁸, passing BigCrush, and
  faster than cuRAND on a GPU (*abstract*). BLP's ancestry-keyed Philox is this design.
- **Where other codes stand.** OpenMC's GPU port reproduces serial CPU results but notes
  floating-point non-associativity at scale (Tramm et al. 2024). Celeritas ties RNG state to
  track slots, so CPU and GPU agree only for the same slot count (Lund et al. 2025). MCX
  accepts races in its fastest mode. GPUmonty is not reproducible (section 3).
- **Order-independent sums.** Demmel and Nguyen (2013) and Collange et al. (2015) give
  reproducible or correctly rounded parallel floating-point summation (pre-rounding or long
  accumulators, CPUs and GPUs). BLP's exact integer bins solve the same problem more cheaply
  for its tallies.
- **Consequence.** Every technique in section 2 that only changes *which thread runs which
  photon, when, or in which order* (persistent threads, truncation and consolidation,
  sorting, compaction, layering, splitting across devices) leaves BLP's results bitwise
  unchanged, provided each photon's own sequence of floating-point operations is unchanged.
  Techniques that change arithmetic (fast math, float atomics, reduced precision, a different
  integrator stage structure) do not qualify.

## 5. A constraint specific to BLP: compiled arithmetic must not move

The main checkout's speed record found that compacting the Verner 9 step (one call site of
the right-hand side, or the right-hand side behind a non-inlined call) changes the CPU bits
of 1,355 of 20,000 steps by up to 8.3e-15 relative, because the inlined right-hand sides
are contracted differently (`results/blp/mc_audit/speed/gpu/README.md`, "Findings"); the
site scan stays on the host for the same reason. Any kernel split therefore has to cut
between whole calls whose bodies keep their compiled form, with the state crossing the cut
stored as full Float64 values. Cuts *inside* a step (between RK stages, between the plasma
sample and the opacities) risk changed bits; cuts *between* steps or between the parent's
event and the child's kick are the low-risk ones. Each cut must be checked against the fixed
decks on both backends.

## 6. Fit to BLP

The requirements: superphotons with Poisson in-step scattering, lifetimes from a few to
20,000 RK4 steps, an AMR plasma sampled every step from device memory, Float64, bitwise
results, and gains on both the CPU and the GPU.

**Test first, before any restructuring.** The literature disagrees on the cause it was
asked to test. Shift's data say register-limited occupancy dominated, not divergence;
GPUmonty reports 12.5 % occupancy and a latency-bound tracker; Laine et al. and MCX-CL
measured low thread utilization (23 %, 62 % divergent time). The Nsight Compute quantities
that separate these are: registers per thread and the register-limited occupancy; achieved
occupancy; local-memory (spill) traffic; average active threads per executed warp
instruction (thread utilization); global-load sectors per request (coalescing of the plasma
reads); warp stall reasons (long scoreboard = memory latency, branch resolving and
no-instruction = divergence and I-cache); per kernel, for the thin and the Compton programs.
Add the distribution of march steps per photon and per child (the lifetime spread Shift
plots in its Fig. 2) and the time share of the kick. Metric names to confirm with
`ncu --query-metrics` on the node.

**Fits well (in order of expected value per risk):**

1. **Deferred Compton kick (stage split at the event).** The parent records the event (the
   event position and incoming momentum, the decoded plasma at the event site, the child's
   stream key, weight, depths, order) and stops there; the child's layer program begins with
   the kick, then tracks. GPUmonty does exactly this (`kernels.cu:928`); it is Laine's
   material stage and Eisenacher's deferred shading in MC form. It removes the rejection
   loops and the sampler's registers from the march of every scattering photon, and puts the
   kicks of a generation in one dense launch. Bitwise: the kick sees identical inputs and the
   same stream; the cut is between whole calls. CPU: same work, a smaller hot loop (GeantV
   attributes much of its CPU gain to fewer instruction-cache misses). Cost: a larger child
   record (the plasma state is about a dozen Float64; storing it avoids re-sampling the site
   in a different compiled context).
2. **Step-budgeted march with consolidation.** The march runs at most S steps per launch;
   survivors persist their full state (ray state, weight, depths, rates at the step start,
   `r_prev`, step counter `it`, event ordinal, ledger partial sums, stream counter) in SoA and
   are consolidated by an index list before the next launch (Hamilton et al. 2018 Alg. 4,
   Celeritas step cap, Aila and Laine loop cap). It attacks the lifetime spread directly:
   warps no longer wait for the 20,000-step photon. Bitwise: a photon's operation sequence is
   unchanged if every loop-carried value, the ledger sums included, crosses the launch
   boundary. CPU: the executor can run S = ∞ (the current history loop), so the CPU path
   loses nothing. Cost: ~25 Float64 of state per survivor (about 200 B, the size of Laine's
   212 B path state), an index compaction per slice; Shift's data put the best slice where the
   run time is flat (interval ≲ 100 events), so S is a tunable with a broad optimum.
3. **Kernel flattening of the tracker, if profiling shows register-limited occupancy.** With
   (1) and (2) in place, the per-slice tracker can be split once more between the geodesic
   step and the plasma/opacity evaluation only if registers stay above ~128 per thread; this
   is Shift's largest single gain (×1.7–4.9 over history) but sits on the risky side of section 5.
4. **Index sorting of survivors and children by meshblock (and frequency bin).** Cheap
   (32-bit radix sort of indices, no state moved), consistent with OpenMC's and Shift's
   practice and with Pharr's spatial queues; it makes the per-step AMR reads of a warp hit the
   same meshblocks and gives the adaptive step neighbouring states (Niemeyer and Sung's 2.3×).
   Its value depends on how much of the step is the plasma sample: Celeritas and Quicksilver
   show net losses when the sorted kernel is cheap. Adopt only if the profile shows the plasma
   reads memory-bound.
5. **Persistent threads with an atomic photon counter.** Free under keyed streams and exact
   sums, small expected gain on current hardware; useful as the launch form for (2), where
   slices of very different sizes follow one another.

**Fits poorly or not at all:**

- Full event-based execution with many small kernels: loses on CPUs in every study (1.8–4×),
  and the fine-grained form costs 2–7.7× on GPUs (Dr.Jit); multigroup-like cheap events lose
  on GPUs too. Only the two coarse cuts above survive the both-backends requirement.
- Sorting the photon state itself per iteration: prohibitive in Shift, WARP and Celeritas.
- In-kernel regeneration of primaries: 1–16 % in graphics, and for BLP it would run emission
  rejection sampling in divergent lanes; generation belongs in its own launch, which the
  cascade already has.
- Race-accepting tallies (MCX), floating-point atomics and per-thread RNG state (GPUmonty),
  fast math or reduced precision (GPUmonty `--use_fast_math`, MCX native math, GRay single
  precision): excluded by the bitwise and Float64 invariants.
- Hardware reordering (SER, dynamic warp formation): unavailable on the A100 for CUDA compute.

**The precise march (Verner 9, register spills).** The literature supports keeping one
trajectory per thread (DiffEqGPU.jl; GRay) and treating spills as an occupancy and
instruction-level-parallelism trade (Volkov; GPUmonty's 32-register experiment). Stage-level
splitting is the Dr.Jit fine-wavefront case and has already been measured slower and
bit-changing in BLP. Remaining levers: the cost-ordered queue BLP already uses, grouping
rays with similar step-size histories (Niemeyer and Sung), a step budget with consolidation
as in (2) if the per-ray step counts spread widely, and a register cap explored with the
deck checks. `maxregs` is a `@cuda` compiler keyword in CUDA.jl, but the KernelAbstractions
CUDA backend compiles with `@cuda launch=false always_inline=… maxthreads=…` only
(CUDACore 6.2.2, `src/CUDAKernels.jl:125`), so a register cap has to be reached through
BLP's backend layer without appearing in plan or function names; `always_inline` changes
inlining and therefore possibly the compiled arithmetic (section 5).

**Expected range.** Literature effects for comparable changes on latency-bound MC trackers:
1.7–4.9× from occupancy recovered by flattening (Shift), up to 2.5× from truncation and
consolidation (Shift multigroup), 1.36–3.2× from a coarse wavefront split (Laine et al.),
25–35 % from step caps (Celeritas, A100), ~0–2× from persistent threads, and anywhere from a
loss to large gains from sorting. GPUmonty's 12× against 64 cores, from a design of BLP's
class at 12.5 % occupancy (with fast math and no reproducibility constraint), indicates what a
well-ported but unrestructured tracker reaches; the
BLP Compton phase at 1× against 32 threads lies well below it.

## 7. Open questions for the profiling stage

- Registers per thread and spill bytes of the thin and Compton MC programs on the A100, and
  whether the kick's rejection loops are inlined into the march.
- Thread utilization and stall breakdown of the Compton program against the thin program.
- Distribution of steps per primary and per child, per generation; fraction of photons that
  reach the 20,000-step cap.
- Time share of the plasma sample (AMR search + decode) per step, and its global-load
  efficiency.
- Order of the children in the device child buffer (arrival order) and whether the layer
  program's warps start from scattered event sites.
- Whether a deferred kick, compiled in the layer kernel, reproduces the inline kick's bits
  on the CPU and on the A100 (section 5).

## 8. References (verified)

Levels: *full text* = read in the paper; *abstract* = abstract or publisher/arXiv/OSTI
record; *secondary* = number taken from a citing source.

1. F. B. Brown, W. R. Martin (1984). Monte Carlo methods for radiation transport analysis on
   vector computers. *Progress in Nuclear Energy* 14(3), 269–299.
   doi:10.1016/0149-1970(84)90024-6. Abstract (U. Michigan Deep Blue record).
   https://deepblue.lib.umich.edu/handle/2027.42/24996
2. F. B. Brown (1994). Random number generation with arbitrary strides. *Trans. Am. Nucl.
   Soc.* 71, 202–203. Record. https://www.osti.gov/biblio/89100
3. S. P. Hamilton, S. R. Slattery, T. M. Evans (2018). Multigroup Monte Carlo on GPUs:
   Comparison of history- and event-based algorithms. *Annals of Nuclear Energy* 113,
   506–518. doi:10.1016/j.anucene.2017.11.032. Full text (OSTI 1414707).
   https://www.osti.gov/biblio/1414707
4. S. P. Hamilton, T. M. Evans (2019). Continuous-energy Monte Carlo neutron transport on
   GPUs in the Shift code. *Annals of Nuclear Energy* 128, 236–247.
   doi:10.1016/j.anucene.2019.01.012. Full text (OSTI 1492181).
   https://www.osti.gov/biblio/1492181
5. J. E. Sweezy, C. J. Solomon, T. P. Burke, S. P. Hamilton, T. M. Evans (2018). The state of
   Monte Carlo neutron transport on GPU accelerators – a review. RPSD 2018, ANS. Abstract.
   https://impact.ornl.gov/en/publications/the-state-of-monte-carlo-neutron-transport-on-gpu-accelerators-a-/
6. J. R. Tramm, P. K. Romano, J. Doerfert, A. L. Lund, P. C. Shriwise, A. R. Siegel,
   G. Ridley, A. Pastrello (2022). Toward portable GPU acceleration of the OpenMC Monte Carlo
   particle transport code. PHYSOR 2022. Cited via ref. 7 (bibliographic entry checked there).
7. J. Tramm, P. Romano, P. Shriwise, A. Lund, J. Doerfert, P. Steinbrecher, A. Siegel,
   G. Ridley (2024). Performance portable Monte Carlo particle transport on Intel, NVIDIA,
   and AMD GPUs. arXiv:2403.12345. Full text (HTML). https://arxiv.org/abs/2403.12345
8. R. M. Bergmann, J. L. Vujić (2015). Algorithmic choices in WARP – A framework for
   continuous energy Monte Carlo neutron transport in general 3D geometries on GPUs. *Annals
   of Nuclear Energy* 77. doi:10.1016/j.anucene.2014.10.039. Record (OSTI 1247567);
   0.8–7.6× from a secondary summary of the 2017 companion benchmark.
   https://www.osti.gov/pages/biblio/1247567
9. R. Bleile, P. Brantley, D. Richards, S. Dawson, M. S. McKinley, M. O'Brien, H. Childs
   (2019). Thin-Threads: an approach for history-based Monte Carlo on GPUs. HPCS 2019,
   273–280. Full text (LLNL-TR-781441). https://cdux.cs.uoregon.edu/pubs/BleileHPCS.pdf
10. D. Ma, B. Yang, Q. Zhang, J. Liu, T. Li (2021). Evaluation of single-node performance of
    parallel algorithms for multigroup Monte Carlo particle transport methods. *Frontiers in
    Energy Research* 9, 705823. doi:10.3389/fenrg.2021.705823. Full text.
    https://www.frontiersin.org/articles/10.3389/fenrg.2021.705823/full
11. S. C. Tognini, P. Canal, T. M. Evans, G. Lima, A. L. Lund, S. R. Johnson, S. Y. Jun,
    V. R. Pascuzzi, P. K. Romano (2022). Celeritas: GPU-accelerated particle transport for
    detector simulation in High Energy Physics experiments. arXiv:2203.09467 (Snowmass 2021).
    Abstract. https://arxiv.org/abs/2203.09467
12. A. L. Lund, J. Esseiva, S. R. Johnson, E. Biondo, P. Canal, T. Evans, H. Hollenbeck,
    S. Y. Jun, G. Lima, B. Morgan, S. C. Tognini (2025). Accelerating detector simulations with
    Celeritas: profiling and performance optimizations. arXiv:2503.17608 (CHEP 2024). Full
    text (HTML). https://arxiv.org/abs/2503.17608
13. G. Amadio et al. (2020). GeantV: results from the prototype of concurrent vector particle
    transport simulation in HEP. *Computing and Software for Big Science*,
    doi:10.1007/s41781-020-00048-6; arXiv:2005.00949. Full text: CPU speed-ups 1.43–2.12×
    over Geant4, vectorization gains largely lost to basketization overheads (10–25 %),
    gains attributed to code and data locality. https://arxiv.org/abs/2005.00949
14. S. Laine, T. Karras, T. Aila (2013). Megakernels considered harmful: wavefront path
    tracing on GPUs. HPG 2013, 137–144. doi:10.1145/2492045.2492060. Full text.
    https://users.aalto.fi/~laines9/publications/laine2013hpg_paper.pdf
15. W. Jakob, S. Speierer, N. Roussel, D. Vicini (2022). Dr.Jit: a just-in-time compiler for
    differentiable rendering. *ACM Trans. Graph.* 41(4) (SIGGRAPH 2022).
    doi:10.1145/3528223.3530099; arXiv:2202.01284. Full text. https://arxiv.org/abs/2202.01284
16. T. Aila, S. Laine (2009). Understanding the efficiency of ray traversal on GPUs. HPG 2009,
    145–149. Full text.
    https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf
17. K. Gupta, J. A. Stuart, J. D. Owens (2012). A study of persistent threads style GPU
    programming for GPGPU workloads. Innovative Parallel Computing (InPar) 2012.
    doi:10.1109/InPar.2012.6339596. Abstract. https://escholarship.org/uc/item/3j76d3td
18. J. Novák, V. Havran, C. Dachsbacher (2010). Path regeneration for interactive path
    tracing. Eurographics 2010 Short Papers, 61–64. doi:10.2312/egsh.20101048. Abstract.
    https://diglib.eg.org/items/36eaff3e-650b-48e4-a79a-1fd814f00a43
19. D. van Antwerpen (2011). Improving SIMD efficiency for parallel Monte Carlo light
    transport on the GPU. HPG 2011, 41–50. doi:10.1145/2018323.2018330. Abstract.
    https://diglib.eg.org:443/xmlui/handle/10.2312/14907
20. I. Wald (2011). Active thread compaction for GPU path tracing. HPG 2011, 51–58.
    doi:10.1145/2018323.2018331. Abstract. https://diglib.eg.org/handle/10.2312/14908
21. J. Hoberock, V. Lu, Y. Jia, J. C. Hart (2009). Stream compaction for deferred shading.
    HPG 2009, 173–180. doi:10.1145/1572769.1572797. Abstract.
    https://diglib.eg.org/handle/10.2312/EGGH.HPG09.173-180
22. K. Garanzha, C. Loop (2010). Fast ray sorting and breadth-first packet traversal for GPU
    ray tracing. *Computer Graphics Forum* 29(2), 289–298.
    doi:10.1111/j.1467-8659.2009.01598.x. Abstract.
    https://diglib.eg.org/items/86c62fdb-d534-458a-86fc-8c75d37d019c
23. C. Eisenacher, G. Nichols, A. Selle, B. Burley (2013). Sorted deferred shading for
    production path tracing. *Computer Graphics Forum* 32(4), 125–132 (EGSR 2013).
    doi:10.1111/cgf.12158. Full text.
    https://diglib.eg.org/handle/10.1111/v32i4pp125-132
24. M. Pharr, C. Kolb, R. Gershbein, P. Hanrahan (1997). Rendering complex scenes with
    memory-coherent ray tracing. SIGGRAPH 97, 101–108. doi:10.1145/258734.258791. Full text.
    https://graphics.stanford.edu/papers/coherentrt/
25. B. Moon, Y. Byun, T.-J. Kim, P. Claudio, H.-S. Kim, Y.-J. Ban, S. W. Nam, S.-E. Yoon
    (2010). Cache-oblivious ray reordering. *ACM Trans. Graph.* 29(3), 28.
    doi:10.1145/1805964.1805972. Abstract. https://sgvr.kaist.ac.kr/publication/cache-oblivious-ray-reordering/
26. D. Meister, J. Bokšanský, M. Guthe, J. Bittner (2020). On ray reordering techniques for
    faster GPU ray tracing. I3D 2020. doi:10.1145/3384382.3384534; arXiv:2506.11273.
    Abstract. https://arxiv.org/abs/2506.11273
27. A. Lier, M. Stamminger, K. Selgrad (2018). CPU-style SIMD ray traversal on GPUs. HPG 2018.
    doi:10.1145/3231578.3231583. Abstract (one ray's traversal spread over several lanes).
    https://diglib.eg.org/handle/10.1145/3231578-3231583
28. W. W. L. Fung, I. Sham, G. Yuan, T. M. Aamodt (2007). Dynamic warp formation and
    scheduling for efficient GPU control flow. MICRO 2007, 407–420. doi:10.1109/MICRO.2007.12.
    Abstract. https://people.ece.ubc.ca/aamodt/papers/wwlfung.micro2007.pdf
29. NVIDIA (2022). Shader Execution Reordering, whitepaper. Introduction and API sections;
    the 2–3× figure is NVIDIA's own claim.
    https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf
30. NVIDIA (2020). NVIDIA A100 Tensor Core GPU Architecture, whitepaper V1.0. Full text
    (Tables 3 and 5). https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
31. V. Volkov (2010). Better performance at lower occupancy. GPU Technology Conference 2010
    (talk). Slides via a third-party mirror.
    https://www.sambuz.com/doc/better-performance-at-lower-occupancy-ppt-presentation-907440
32. Q. Fang, D. A. Boas (2009). Monte Carlo simulation of photon migration in 3D turbid media
    accelerated by graphics processing units. *Optics Express* 17(22), 20178–20190.
    doi:10.1364/OE.17.020178. Abstract. https://hgpu.org/?p=2128
33. L. Yu, F. Nina-Paravecino, D. Kaeli, Q. Fang (2018). Scalable and massively parallel
    Monte Carlo photon transport simulations for heterogeneous computing platforms.
    *J. Biomed. Opt.* 23(1), 010504. doi:10.1117/1.JBO.23.1.010504; arXiv:1711.03244. Full
    text. https://arxiv.org/abs/1711.03244
34. C.-K. Chan, D. Psaltis, F. Özel (2013). GRay: a massively parallel GPU-based code for ray
    tracing in relativistic spacetimes. *ApJ* 777(1), 13. doi:10.1088/0004-637X/777/1/13;
    arXiv:1303.5057. Abstract. https://arxiv.org/abs/1303.5057
35. U. Utkarsh, V. Churavy, Y. Ma, T. Besard, P. Srisuma, T. Gymnich, A. R. Gerlach,
    A. Edelman, G. Barbastathis, R. D. Braatz, C. Rackauckas (2024). Automated translation and
    accelerated solving of differential equations on multiple GPU platforms. *Computer
    Methods in Applied Mechanics and Engineering* 419. doi:10.1016/j.cma.2023.116591;
    arXiv:2304.06835. Abstract. https://arxiv.org/abs/2304.06835
36. K. E. Niemeyer, C.-J. Sung (2014). Accelerating moderately stiff chemical kinetics in
    reactive-flow simulations using GPUs. *J. Comput. Phys.* 256, 854–871.
    doi:10.1016/j.jcp.2013.09.025; arXiv:1309.2710. Full text. https://arxiv.org/abs/1309.2710
37. J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw (2011). Parallel random numbers: as easy
    as 1, 2, 3. SC11, 1–12. doi:10.1145/2063384.2063405. Abstract.
    https://users.cs.utah.edu/~hari/teaching/bigdata/random123sc11.pdf
38. J. Demmel, H. D. Nguyen (2013). Fast reproducible floating-point summation. ARITH 21,
    163–172. doi:10.1109/ARITH.2013.9. Record. https://researchr.org/publication/DemmelN13
39. S. Collange, D. Defour, S. Graillat, R. Iakymchuk (2015). Numerical reproducibility for
    the parallel reduction on multi- and many-core architectures. *Parallel Computing* 49,
    83–97. doi:10.1016/j.parco.2015.09.001. Record. https://hal-lirmm.ccsd.cnrs.fr/lirmm-01206348
40. P. Naethe Motta, R. Nemmen, A. V. Joshi (2026). GPUmonty: a GPU-accelerated relativistic
    Monte Carlo radiative transfer code. arXiv:2602.13198. Full text
    (`/data/haiyangw/claude/Mahakala/ref-paper/arxiv-2602.13198/main.tex`) and code
    (`/data/haiyangw/claude/Mahakala/ref-code/gpumonty`, GPL v2, reference only).
    https://arxiv.org/abs/2602.13198

Downloaded copies of the full texts read for this map: `/data2/haiyangw/scratch/wavefront/
plan/lit_dl/` (laine2013, aila2009, eisenacher2013, hamilton, bleile, mcx, geantv, drjit,
niemeyer, pharr1997, a100).
