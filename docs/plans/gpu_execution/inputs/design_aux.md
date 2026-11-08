# Design of the GPU items outside the Monte Carlo executor

Items: (1) bit-portable device arithmetic, the device site scan and bitwise CPU = GPU
results; (2) device-resident and caller-supplied outputs; (3) two A100s; (4) register
pressure and the tail of the precise march; (5) compile latency.

Tree `a5923dd` (worktree `/data2/haiyangw/scratch/blp_wavefront`, branch `gpu-wavefront`,
unmodified). Inputs: `profile.md`, `map_execution.md`, `map_mcprograms.md`, `map_march.md`,
`map_compiler.md`, `map_outputs.md`, `map_literature.md` in this directory. New evidence
for this design is in `aux/` (offline, sm_80, no GPU, host `anta` at load 137–143 on 144
cores, so absolute compile times are noisy and ratios are the point); its index is §9.
**Measured** numbers name their source; **(estimate)** marks a number derived from
measurements and not itself measured.

## 0. Decisions for the user

Items 2, 3 and 5, and track A of item 4, change no bit on either backend. Item 1 and
tracks B and C of item 4 cannot be done without moving bits, and need one or both of:

| | decision | what moves | what it buys |
|---|---|---|---|
| **D1** | GPU bits may change where they move onto the host's bits | GPU baselines of every fixed deck (each calls libdevice functions, §3.1); host bits unchanged by construction (each port is bitwise Base on the host) | depot-independent host bits; smaller CPU–GPU gaps; the libdevice half of CPU = GPU |
| **D2** | one host re-baseline: the StaticArrays reductions and `muladd` chains of the physics become explicit `fma` chains in the device's current association | host bits of the decks that read the decode, the geodesic right-hand side or the tetrads, by a few ulp per operation (site scan: 661 of 104,776 zones, ≤ 3 ulp in `theta_e` and `B`, `map_compiler.md` §3.3); GPU bits there unchanged wherever the explicit form reproduces the device's association (verified site by site on the A100) | device site scan with the host's bits; CPU = GPU decks (with D1); code-shape changes that are bitwise by construction on both backends (item 4 B, the Monte Carlo executor's stage splits); host continuation of device marches (item 4 C) |

With D1 and D2 the end state is symmetric: the GPU baseline moves only for elementary
functions, the CPU baseline only for reductions, and afterwards the two agree bit for bit
on every converted path. Without them, item 1 stops at its gate in report mode (1.1) and
item 4 stops at track A.

## 1. Invariants and the bit taxonomy

Gates every work item passes (the bit suite, §8.4): the 13 fixed decks of
`results/blp/mc_audit/speed/gpu/scripts/decks.jl` bitwise against `a5923dd` on 1 and 16
CPU threads and on one A100; one GPU against two GPUs bitwise; CPU against GPU no worse
than today per deck; the test suite with `test/structure.jl` green; speed on both backends
kept or improved. The CPU–GPU gaps of today, deck by deck (`logs/decks_gpu_3789.log` in the
main checkout, largest |a − b| over the deck's largest magnitude; integer parts equal on
every deck):

| deck | gap | deck | gap | deck | gap |
|---|--:|---|--:|---|--:|
| cf_disk | 2.31e-9 | cross_B_precise | 9.03e-11 | mc_torus_compton | 1.18e-13 |
| cf_disk_radius | 6.16e-12 | cross_B_production | 2.04e-12 | mc_torus_s1 | 2.24e-15 |
| cross_A_precise | 1.23e-12 | mc_sphere_scatt | 1.06e-14 | mc_torus_thin | 4.07e-15 |
| cross_A_production | 5.04e-12 | mc_sphere_thin | 1.35e-16 | torus_I / torus_IQUV | 2.51e-13 / 3.15e-13 |

A change is in one of three classes (`map_march.md` §8):

- **A, outside the compiled body**: visiting order, chunking, regrouping, lanes, devices,
  workgroup size, register allocation, where outputs live, when a kernel is compiled.
  Bitwise by construction (executor contract, Philox ancestry streams, exact integer sums;
  `map_execution.md` §6 I1–I11). Verified, never assumed, for register caps (§6.2).
- **B, inside the body, exact operation for operation**: bitwise only if the compile context
  does not re-associate or contract differently. Today it does: StaticArrays `dot` carries
  `reassoc contract`, `M*v` and `muladd` carry `contract`, and LLVM treats them by context
  (V9Sched moved 1,060 and 7,041 `Ib` entries on disks A and B; the non-inlined right-hand
  side 1,355 of 20,000 steps). Every class-B change is checked on both backends.
- **C, through a re-baseline**: D1, D2.

After D2 the converted paths hold no flagged operation, and class B collapses into class A
for them: a function made only of IEEE operations and explicit `fma` has one admissible
rounding per operation on the host and on the device, in every inlining context.

## 2. Summary

| item | design | class | expected gain (basis) | effort | needs |
|---|---|---|---|---|---|
| 1 | portable-arithmetic standard with a host-only IR gate; ports of the libdevice functions; explicit reductions; site scan as a device program; CPU = GPU decks path by path | C (D1, D2) | site scan 2.0 s → ≈ 0.1 s per uncached GPU grid call (estimate); per-snapshot thin call ≈ 6.7 → ≈ 4.8 s (estimate); CPU = GPU bitwise on converted decks | gate 1 d; ports 0.5 d each; reductions 3–5 d (Kerr march) to 2–3 weeks (all physics); scan 2 d | D1, D2 |
| 2 | threaded host epilogues as programs; `into` (caller storage); `resident` outputs in `PoolArray`s that programs read in place | A | closed-form 2048² radius 0.305 s → ≈ 0.20 / 0.05 / 0.02 s on the A100 (default / `into` / `resident`, estimates from host measurements); CPU 32 threads 0.52 → ≈ 0.42 / 0.34 s | 4–6 d | — |
| 3 | lanes of a device that overlap (shared device arrays without implicit synchronisation); cost-ordered chunk queue; two-device chunk rule; two-GPU job template and records | A | thin torus stride 4: two GPUs 2.53–2.69 s → ≈ 1.9 s, one GPU 3.93 → ≈ 3.1 s (estimates from a launch model fitted to job 3795) | 3–4 d | — |
| 4 | A: lane overlap and batched queues, camera-ray launch, workgroup and register-cap experiments; B (after D2): V9Sched, non-inlined right-hand side, deferred crossings; C (after D1 + D2 on the Kerr march): host continuation of the slowest rays | A, then B | single image 1.65 s: A ≈ −8 %, B → ≈ 1.1–1.4 s, C → ≈ 1.0–1.1 s on 4 host threads, ≈ 0.6 s on 16 (estimates); batches of images up to ≈ 4× throughput under A (estimate) | A 3 d; B 4 d; C 4 d | B: D2; C: D1, D2 |
| 5 | precompile workload in the CUDA extension (the first device compile of a process is 7–13 s because ForwardDiff's load invalidates the compiler); GPUCompiler disk cache for fixed-type programs; compile-ahead of the next kernel; smaller kernels from item 4 B | A | first compile 8–13 s → ≈ 1–2 s per process (estimate from 0.9–1.7 s without ForwardDiff, measured); zone and hot-cross tables ≈ 3 s → module load; cold Compton call 101.5 s → ≈ 80 s (estimate) | 3–5 d | — |

## 3. Item 1: bit-portable device arithmetic

### 3.1 Facts

- Device arithmetic is IEEE-exact wherever the source fixes the rounding (`+ - * /`, `sqrt`,
  explicit `fma`, `min`/`max`, conversions); no CUDA.jl, KernelAbstractions, `llc` or
  `ptxas` setting reaches contraction (`map_compiler.md` §2, §4).
- The device departs from the host through (a) libdevice replacements of Base math, (b)
  `muladd` → `llvm.fmuladd` → always `fma.rn`, (c) `@simd` reductions with
  `reassoc contract`, fused by the device and fused or not by the host according to context.
- The host is the side whose rounding nothing controls: the same decode gives different
  bits in two host contexts (10,004 of 104,776 zones differ in `theta_e`), and `hypot` takes
  Base's non-FMA branch because the first image to compile it is a generic-target pkgimage
  (206 of the loaded pkgimages are `generic`), so host bits depend on the depot
  (`map_compiler.md` §3.1–3.2). The "bitwise against main" invariant holds only inside this
  depot.
- The zone-table and hot-cross programs are already portable and bitwise host = device
  (`tang_exp`, `tang_log`, explicit `fma`; census below: no libdevice call, no `fmuladd`,
  no flagged instruction).
- Census of whole kernels (**measured**, `aux/census_kernels.jl`, `aux/census*.log`: device
  IR with `libraries = false`, so every libdevice call stays a declaration; call sites in
  the optimised IR):

| kernel | libdevice functions (call sites) | `fmuladd` | `contract` / `reassoc` instructions |
|---|---|--:|--:|
| `ZoneSums`, `ZoneCounts`, `HotcrossSigma` | none | 0 | 0 / 0 |
| `MCPrimary` thin torus | acos 1, cbrt 4, cos 1, exp 7, expm1 5, hypot 10, log 2, pow 4, sin 1 | 202 | 283 / 144 |
| `MCPrimary` Compton torus | acos 1, cbrt 4, cos 9, exp 17, expm1 7, hypot 15, log 19, log10 12, log1p 1, pow 7, sin 5 | 355 | 527 / 268 |
| `MCPrimary` scattering sphere | acos 1, cbrt 3, cos 10, exp 15, expm1 5, log 19, log10 12, log1p 1, pow 5, sin 6 | 44 | 26 / 14 |
| `ImagingProgram` torus, Stokes I | cbrt 1, exp 2, expm1 2, hypot 50, pow 1 | 31 | 877 / 448 |
| `ImagingProgram` blocks, IQUV | cbrt 1, cos 5, cosh 1, exp 16, expm1 6, hypot 51, log 4, pow 6, sin 6, sinh 1, tanh 1 | 1003 | 1128 / 607 |
| `CrossingProgram` RK4 | asinh 1, atan2 2, cbrt 6, exp 1, hypot 64, pow 3 | 10 | 1149 / 579 |
| `CrossingProgram` Verner 9 | asinh 1, atan2 2, cbrt 6, exp 1, hypot 108, pow 3 | 10 | 2205 / 1107 |
| `AnalyticProgram` (closed form) | acos 6, asin 34, asinh 1, atan 1, atan2 1, atanh 1, cbrt 7, cos 16, cosh 12, cospi 1, exp 1, fmod 10, hypot 2, log 13, pow 48, sin 50, sincos 2, sinpi 1, tanh 6 | 173 | 0 / 0 |

  The flagged instructions come from a few dozen source sites (StaticArrays `dot` in
  `raise_and_norm`, `kerr_schild.jl:97, 99`, and in `far_zone.jl`; the decode's
  `gcov * ucon`, `gcov * bcon`, `dot(bcon, bcov)`, `decode.jl:36, 43, 45`; tetrads and
  transport products), each multiplied by the inlined copies and the three Dual partials.

### 3.2 The standard

A **portable function** uses only `+ - * /`, `sqrt`, `fma`, `min`/`max`, `abs`, `copysign`,
conversions, comparisons, integer arithmetic, and BLP-owned ports of elementary functions;
it uses no `muladd`, no StaticArrays reduction or product (`dot`, `norm`, `sum`, `M*v`,
`M*N`), no `^` with a non-literal exponent, no `@fastmath`, and no call into a non-inlined
Base or dependency math specialization. Such a function has the same bits on the host and
the device in every context, and the same host bits in every depot.

- **Ports** (in `Numerics`, beside `tang_exp` and `tang_log`): Base's algorithm with every
  `muladd` an explicit `fma`, its branches fixed, and bitwise Base on the host over sweeps of
  the arguments the decks reach (the `tang_exp` protocol: 5e8 arguments on the host, 2e7 on
  the A100). `hypot` is ported with the non-FMA split branch, which is the hypot every deck
  of this depot runs (0 mismatches in 1e7 Kerr–Schild arguments, `exp/hypot_env.jl`).
- **Constants of a deck** that the device now computes per item (`2.0^(11.0/12.0)` and
  `3.0^(7.0/3.0)` in the synchrotron fit, the `cbrt` and `acos` of the ISCO and photon-orbit
  radii) are computed once on the host and carried in the program as `Float64` fields:
  the host folds them today with Julia's `pow`, so the host bits stay and the device takes
  them.
- **Explicit reductions** (D2): `dot4(a, b) = fma(a4, b4, fma(a3, b3, fma(a2, b2, fma(a1, b1, 0.0))))`
  and `mv4(M, v)[k] = fma(M[k,4], v4, fma(M[k,3], v3, fma(M[k,2], v2, M[k,1] * v1)))`: the
  association the device compiles today (`map_compiler.md` §2, census bodies), so the GPU
  bits of a converted site do not move. Dual arguments use the same expression on `Dual`s;
  a site whose device bits do move under the explicit form is listed in the re-baseline
  record with both backends' differences.
- **The gate** `test/portability.jl` (host only, no GPU): for each program on the portable
  list, compile its device job with `libraries = false` (as `aux/census_kernels.jl` does)
  and assert no `__nv_` symbol, no `llvm.fmuladd`, no `contract`/`reassoc`/`afn`/`fast`
  flag; for the host, assert the same flags absent from `code_llvm` of the program body and
  no call to a non-inlined Base math function. Before conversion it prints the census
  (report mode); a converted program enters the list and stays enforced.

### 3.3 Tiers

| tier | content | CPU bits | GPU bits | decision |
|---|---|---|---|---|
| P0 | the gate in report mode; the GPU bit check of `exp/gpu_bitcheck.jl` (written, never run); a pinned reference tree for the bit suite | unchanged | unchanged | none |
| P1 | ports of every libdevice function the fixed decks call (union over the census: hypot, exp, expm1, log, log10, log1p, cbrt, pow, sin, cos, acos, asin, atan, atan2, asinh, atanh, sinh, cosh, tanh, sincos, sinpi, cospi, fmod) and host-computed constants, kernel by kernel | unchanged (each port bitwise Base on the host) | move onto the host's elementary functions | D1 |
| P2 | explicit reductions in the decode (`primitive_four_vectors`), the Kerr–Schild routes (`raise_and_norm`, `raise_index`, `lower_index`, `ks_radius_energy` is already plain), the far zone, then tetrads, transport and coefficients | move by a few ulp per converted site | unchanged by construction (device association kept) | D2 |
| P3 | the site scan as a device program (§3.4) | unchanged after P2 | the host's after P1 + P2 | — |
| P4 | deck by deck, CPU = GPU bitwise as each kernel enters the portable list | — | — | — |

Order of P4 by effort and value: the zone tables (already portable), the Kerr crossing
march (production and precise: hypot, cbrt, exp, pow, atan2, asinh plus about ten
reduction sites), the Monte Carlo grid march (adds acos, cos, sin, log, log10, log1p,
expm1 and the decode), Stokes I imaging, IQUV imaging (the transport products, 1003
`fmuladd`), and last the closed form (19 functions, 173 `muladd`, partly inside
JacobiElliptic, which BLP does not own: a port means vendoring the routines it uses or an
upstream change).

### 3.4 The site scan on the device

- Program `SiteScan` over the groups `1:nmb·ngroups` (`Data.emission_sites`,
  `src/infra/Data/tables.jl:103-163`): per group the owned-cell count, the Philox pick
  (integer), `proxy_distance`, `sample` (locate, trilinear, metric, decode), `det` (4×4,
  plain), and the cuts; it writes the `ZoneSite` and a validity flag at its group index. A
  `Compact` pass then writes the valid sites in group order (count per block of groups,
  exclusive scan on the host over at most 11.3 M / 1024 block counts, scatter): the list is
  the host's, in the host's order.
- The arithmetic of one group is one function, `site_of(blocks, st, params, mb, g)`, called
  by the host loop of `emission_sites` (threads, today's per-block lists, no dense array)
  and by `SiteScan` (devices). One code path; the executor decides only the loop.
- With item 2, the sites stay resident on the device: `ZoneSums`, `ZoneCounts` and
  `MCPrimary` read them in place, which removes the 659 MB and 433 MB uploads of the cold
  call (0.30 s and 0.17 s, `profile.md` §5.2). The host still needs `maximum(proxy radius)`
  (`R_em`, a reduction the scan program writes per block) and the counts for
  `zone_offsets` (stays on the host: one stream in zone order, I9).
- Device memory at stride 1: 11.3 M groups × 72 B ≈ 0.8 GB of dense records, freed after the
  compaction.

### 3.5 Can CPU and GPU images be bitwise equal?

Yes, by the construction above and by nothing else (`map_compiler.md` §6): after D1 and D2
every operation of a converted kernel has one admissible rounding on both backends. The
census of §3.1 is the work list: the Kerr crossing march is a few days (6 functions, about
ten reduction sites), the torus Stokes I image a week (adds the decode, the synchrotron fit
and `expm1`), IQUV and the Monte Carlo march one to two weeks more, the closed form last.
Each kernel enters the gate list when its deck is bitwise CPU = GPU on the A100. The value
beyond validation: CPU and GPU become interchangeable within one result, which item 4 C
uses (rays continued on host threads) and which removes the per-depot fragility of today's
host bits.

### 3.6 Expected gains

- Device site scan: 2.0–2.04 s on the four host threads of a GPU job (**measured**, README
  "Monte Carlo" table) → ≈ 0.1 s (estimate: 8 corners × 8 primitives per cell, ≈ 2.6 GB of
  loads at stride 1). Gains per uncached grid call; a run on a new snapshot (a movie, a
  parameter scan over snapshots) pays scan 2.0 + tables 0.6 + offsets 0.3 + thin photons
  3.74 ≈ 6.7 s today and ≈ 4.8 s after (estimate). Repeated calls on the same snapshot hit
  the memo and gain nothing.
- CPU: the site function is the same arithmetic; explicit `fma` costs nothing on the host
  (`llvm.fma` vectorises, `map_compiler.md` §4). Measured before adoption (invariant).
- CPU = GPU decks: no speed; a validation property, and the precondition of item 4 C.

### 3.7 Risks

- A port that is bitwise Base on the swept arguments but not on a deck argument: the decks
  are the final test; sweeps cover each function's reduced-argument ranges and the decks'
  actual argument histograms.
- Base's own bits differ by depot for functions compiled in generic pkgimages (`hypot`
  today): the port fixes the branch, so it equals today's host in this depot; in a native
  depot it equals the bits the old code would *not* have given. This is the intended
  robustness, recorded in the re-baseline note.
- Dual arithmetic under explicit `fma`: ForwardDiff's `fma` and `muladd` for `Dual` compute
  partials with their own association; the explicit reduction is written on the value and
  partials as the device compiles today, and verified by the GPU decks.
- Speed on the device: explicit `fma` chains are what the device runs today; no change
  expected. Ports replace tuned libdevice code by Base's algorithms: measured per kernel
  (`tang_exp` was neutral on the A100).

### 3.8 Work items

| id | work | acceptance |
|---|---|---|
| 1.1 | `test/portability.jl`: census per program (libraries = false device IR; host IR flags and calls), report mode | runs on the host without a GPU in < 5 min; prints the census of §3.1; suite green |
| 1.2 | GPU bit check job (`exp/gpu_bitcheck.sbatch`, one A100, minutes) | stock device ≠ host for `dot`, `muladd`, `hypot`, KS metric, kinematics; portable device = portable host in all 1e6 inputs |
| 1.3 (D1) | `hypot` port (split branch); `_ks_term` reads it | host: 0 mismatches against the environment's `hypot` in 1e7 KS arguments and every CPU deck bitwise; GPU decks change, CPU–GPU gaps no larger than §1 |
| 1.4 (D1) | ports for the Kerr crossing kernels (cbrt, exp via `tang_exp`, pow, atan2, asinh) and host-computed constants | each port bitwise Base on the host over its sweep; CPU decks bitwise; GPU kernel IR free of `__nv_` (gate) |
| 1.5 (D2) | `dot4`, `mv4` and the Dual forms in the Kerr–Schild routes and the far zone | GPU decks bitwise against `a5923dd` GPU; CPU differences recorded; `cross_*` decks CPU = GPU bitwise; CrossingProgram on the enforced list |
| 1.6 (D2) | explicit kinematics in `primitive_four_vectors` | host emulation `fused_h` reproduced (0 mismatches over the 104,776 zones of `amr_deck(nc = 32)`); GPU decks bitwise |
| 1.7 | `site_of`, `SiteScan`, `Compact`, `emission_sites` on the run's executor | device sites bitwise the host's at strides 1 and 8 (11.3 M groups); CPU scan time not above today's (interleaved runs, 4 and 32 threads); GPU scan ≤ 0.2 s |
| 1.8 | P4 deck by deck (Monte Carlo march ports and decode; Stokes I; IQUV; closed form) | each deck bitwise CPU = GPU on the A100 before its kernel enters the enforced list; speed rows on both backends |

## 4. Item 2: outputs that stay where they are computed, or go where the caller says

### 4.1 Facts

- Every per-item product allocates its host outputs; on a pool each chunk writes per-lane
  device scratch, copied through pinned staging and scattered into the host arrays by the
  lane task (`executor.jl:266-293`); no output stays on a device and no product accepts
  caller storage (`map_outputs.md` §1).
- Closed-form 2048² radius call: product 0.305 s, its program into preallocated outputs
  0.036 s on the A100 (**measured**, README). The 0.27 s difference: first touch of 9 fresh
  32 MiB outputs by the two lane tasks (≈ 0.13 s), six NaN maps filled on one thread
  (≈ 0.10 s), a one-thread band sum (≈ 0.035 s) and the collection of the previous call's
  512 MiB (`map_outputs.md` §3; host reproductions 0.179 s fresh against 0.013 s reused,
  NaN fill 0.138 → 0.043 s threaded, band sum 0.050 → 0.015 s threaded, bitwise). The CPU
  pays the same epilogue: 32 threads 0.52 s against 0.33 s for the program.
- `analytic_movie` is the one product chain and makes a full round trip per call.
- Monte Carlo results are not output-bound (≈ 11 MB per device per call).

### 4.2 Design

Taken from `map_outputs.md` §4 with this order:

1. **Default path, no API, both backends.** The band sums (`I .+= Ib[b]`) and the NaN and
   zero fills become `BandSum` and `Fill` programs run on `ThreadedExecutor()` when the
   outputs are host arrays (host data stays on the host whatever the run's executor; the
   pipeline uses `run_workers!`, not `Threads.@threads`). `analytic_movie` keeps its spot
   maps as `output_array`s internally and writes every frame in place. `BandSum` adds in
   band order (`((0.0 + Ib1) + Ib2) + Ib3`, no multiply), bitwise the broadcasts.
2. **`into = r`** on `render_analytic` (both forms), `render_crossings`, `render`,
   `analytic_movie`: the arrays of a previous result of the same product and shape are
   overwritten and returned; no allocation, no first touch, no collection. A host array
   given as an output is evicted from the pool's residency cache before the run (a cached
   input must not be mutated: `residency.jl:3-4`).
3. **`resident = true`**: outputs are `output_array(exec, T, dims)`: an `Array` on
   `ThreadedExecutor`, a `PoolArray` on a `PoolExecutor` (per device that ran a chunk, one
   device vector of all elements; the items its chunks wrote recorded). A program that
   holds a `PoolArray` reads its device part with no upload (`adapt_structure` on
   `DeviceAdaptor`); `convert(Array, a)` assembles the host array, bitwise a host-output
   run; `resident(pool, dev, a)` hands the device vector to user CUDA code. On several
   devices `complete!(a, dev)` copies the parts the other devices wrote (ranges: one
   `copyto!` each; permuted chunks: gather and scatter programs), ordered before the
   consumer launch by a device-protocol copy whose CUDA method synchronises the source
   stream.

Nothing names a backend: placement follows the executor the one `backend` option resolved;
on threads both keywords give `Array`s of today's types and bits. No top-level export and
no product function is added (exports 54/54, family 9/10); `Execution` exports
`PoolArray` and `output_array` with docstrings. `item_bytes` counts host outputs only, so a
resident run takes the one-device `2^20` chunk (no bit depends on the chunk).

### 4.3 Expected gains (estimates from the host measurements; to be measured on the A100)

| call | today | default path | `into` | `resident` |
|---|--:|--:|--:|--:|
| closed form radius 2048², A100 | 0.305 s | ≈ 0.20 s | ≈ 0.05 s | ≈ 0.01–0.02 s (kernel) |
| closed form full 2048², A100 | 0.458 s | ≈ 0.35 s | ≈ 0.21 s | ≈ kernel |
| closed form radius 2048², CPU 32 threads | 0.52 s | ≈ 0.42 s | ≈ 0.34 s | = default |
| `render_crossings` 500², A100 production | 0.163 s | a few % | 10–15 % | 10–15 % |
| `analytic_movie` | round trip | no re-upload, no frame copy | — | frames on device |

### 4.4 Risks

- Stale residency when a caller reuses a host array that a program uploaded earlier:
  eviction before every pool run that writes it (tested).
- `PoolArray` is not an `AbstractArray`: host indexing fails loudly; the assembled array is
  bitwise, the written-item record decides which device holds which element.
- Two devices: `complete!` without peer access goes through the host (`copyto!` in CUDA.jl
  6.2.2 picks the path); ordering is the device-protocol copy's job.
- Host reductions that pin an order stay host-side (`analytic_movie` light curve,
  `sensitivity` flux, `photon_ring_layers` deposits, `zone_offsets`).

### 4.5 Work items

| id | work | acceptance |
|---|---|---|
| 2.1 | `BandSum`, `Fill`, `Split` programs; default epilogues on `ThreadedExecutor()`; `analytic_movie` chain and in-place frames | every deck bitwise on CPU and GPU; `cf_radius` on the A100 ≤ 0.22 s and on 32 CPU threads ≤ 0.45 s, GC time per call reported (`@timed`) |
| 2.2 | `into` on the five products; residency eviction | `into` results bitwise the default on `:cpu`, `:host_pool`, `Decks.hosts(3)` and the A100; zero allocation above 1 MiB per call (`@allocated`); reused host output after an upload gives fresh values |
| 2.3 | `outputs.jl`: `PoolArray`, `output_array`, `complete!`, `Array`/`convert`/`copyto!`; `run_chunk!` in-place writes; `adapt_structure` | `PoolArray` legs bitwise `Array` legs for ranges, permuted queues, `order`, multi-part queues, explicit chunks, 1–3 host devices; structure gates green |
| 2.4 | `resident` on the products; `Image{M,P,R}` | `convert(Array, r.I)` bitwise the default on CPU and GPU, one and two GPUs; `cf_radius_resident` kernel-bound on the A100 |
| 2.5 | docs: architecture §4.2 and §5.3, performance §1, §2, §6, guide `gpu.md` "Keeping results on the devices" | `docs/make.jl` builds; no narrative prose (structure gate) |

## 5. Item 3: both A100s

### 5.1 Facts

- Slurm grants both GPUs to `--gres=gpu:2` with the node's four CPUs; `resolve(:gpu)` then
  builds `PoolExecutor([CUDADevice(0), CUDADevice(1)]; lanes = 2)`; the node runs one job
  at a time whatever its GPU count (`profile.md` §1, §6).
- Thin torus at stride 4, 1e6 primaries (**measured**, job 3795; per-device kernel times
  extracted from `raw/twogpu_3795.sqlite` for this design):

| executor | warm calls | launches per device | launch durations | per-device kernel union | bins |
|---|---|---|---|---|---|
| `gpu:0` | 3.99, 4.09 s | 4 × 250,002 | 0.56–1.26 s (mean 0.98) | 3.93–3.94 s | reference |
| `gpu:0,1` | 2.69, 2.68 s | 4 × 125,001 | 0.32–0.93 s (mean 0.63) | 2.16–2.61 s | bitwise |
| `:gpu` (both) | 2.58, 2.67, 2.53 s | 4 × 125,001 | 0.32–0.99 s | 2.20–2.81 s | bitwise |

  Two GPUs give 1.5×. Halving a launch's primaries cuts its duration from 0.98 to 0.63 s
  on average: duration ≈ 0.28 s + 0.35 s per 125,000 primaries (estimate, two-point fit).
  The fixed 0.28 s is the launch's tail, set by its slowest photons (warp span maximum
  0.53–0.57 s, `profile.md` §4.3), and it is paid once per launch: four times per device.
- The two lanes of a device do not overlap their kernels: a launch converts every array
  of the device twin the two lanes share, and CUDA.jl's `Managed` conversion synchronises
  with the stream that used the array last (`CUDACore/src/memory.jl:589-632`); a spin
  program holding no shared array overlaps (8 chunks in 1.54 s on two lanes against 3.03 s
  on one, `profile.md` §5.3). So every launch's tail idles the device.
- Device-to-device balance is dynamic (one chunk queue over all lanes) and children stay on
  their lane; each device uploads its own copy of every input (B13).
- One GPU against two is bitwise by I10 (the same compiled program on both A100s) and the
  exact sums; the decks were never recorded on two GPUs; `performance.md` §5: "No
  two-device pair of either package has been recorded."

### 5.2 Design

1. **Lanes that overlap.** A device-protocol function `share_lanes!(dev, a)`: complete the
   copy into `a` (synchronise the uploading stream once) and let every lane of `dev` use
   `a` without implicit synchronisation. Fallback: no-op. CUDA method (extension):
   `CUDA.synchronize(a); CUDA.enable_synchronization!(a, false)`. Called for every device
   array a program twin holds: by `cached_twin` after `upload` (read-only inputs, never
   mutated after upload: `residency.jl:3-4`), by `zeroed` (accumulator twins: integer
   atomics from both lanes commute, I4) and by the child-buffer twins of plain
   `foreach_item!` programs (`children.jl:44-51`: distinct slots from one atomic counter);
   `gather!` reads them only after every lane's launch has synchronised its own stream
   (`kernel.jl:42`). Lane-owned buffers keep CUDA.jl's default (one stream uses them). Two kernels of one device then
   run concurrently; blocks of the second fill the SMs that the first's tail frees.
   Results: unchanged (I1, I4, I5); this is the executor change the Monte Carlo executor
   design also relies on, and it is owned here.
2. **Chunk rules on several devices.** With lanes overlapping, a device pays one tail per
   pair of concurrent launches instead of one per launch. The cascade chunk
   (`cascade.jl:24-25`, `cld(n, 4 d)`) and the image chunk (`executor.jl:69-76`,
   `cld(n, 2 l d)`) are re-measured with overlap on; candidates `cld(n, 4 d)`,
   `cld(n, 2 d)`, `cld(n, l d)`. Bitwise (I2).
3. **Cost-ordered chunk queue.** The pool queue is index order; chunks whose primaries come
   from the inner zones run longest (launch durations 0.32–1.26 s for equal counts). The
   cascade posts its chunks longest first by a predicted cost (emitted count weighted by the
   zone-radius decile step means of `profile.md` §4.1); the end-of-call imbalance between
   the devices (0.25 s in the second two-GPU call) shrinks. Bitwise (I2: boundaries depend
   on `n` and the chunk only; the queue order is free).
4. **Shared first capacity.** `caps` is per lane (`cascade.jl:273, 298-299, 327`), so with
   `d` devices and `l` lanes the first chunk of each of the `l d` lanes overflows and reruns
   in a Compton run (9.8 s of the 63.2 s call on one GPU, `profile.md` §4.5); a pool-wide
   capacity read by every lane's first chunk removes the repeats. Bitwise (capacity never
   changes a bit). The Monte Carlo executor design may replace the generation layout
   altogether; this item is the fallback for today's cascade.
5. **Job template.** `two_gpu.sbatch` in `results/blp/gpu/` (main checkout; results only):
   `-p preempt --gres=gpu:2 -c 4 -t 01:00:00`, `run_job.sh` as body, `julia -t 4` (four lane
   tasks and the caller on four threads; the pool's warning fires below `devices + 1`
   threads), `JULIA_PKG_OFFLINE=true`, the environment a copy of `env_main` with
   `[CUDA_Runtime_jll] version = "13.3"`. One process runs every leg on `gpu:0` and on
   `gpu:0,1` (as `twogpu.jl`), so one job yields the speed rows and the one-against-two bit
   comparison; a `nvidia-smi topo -m` line records the peer path for item 2's
   `complete!`. A two-GPU job holds the whole node, as a one-GPU job does (4 CPUs in all).
6. **Records.** The two-device rows of the reference benchmark and of the fixed decks
   (`performance.md` §5, Verification record §4.1).

### 5.3 Expected gains (estimates from the launch model of §5.1)

| run | today | with 1 (overlap) | with 1–3 |
|---|--:|--:|--:|
| thin torus stride 4, one A100 | 3.93 s kernels | ≈ 3.1 s (4 × 0.70 + one 0.28 tail) | ≈ 3.0 s |
| the same, two A100s | 2.16–2.61 s per device | ≈ 1.7 s per device (4 × 0.35 + 0.28) | ≈ 1.6 s |
| thin torus stride 1, one A100 | 3.74 s photon phase | ≤ 0.78 s less: the time below half the warp slots (`profile.md` §4.4) | |
| Compton torus, one A100 | 62.5 s | ≤ 11.4 s less (below half the slots) | 4 also removes ≤ 9.8 s of reruns |
| torus imaging 768², two A100s | — | ≈ 0.25 s kernels (half of 0.457 s plus tails) | |

Two GPUs over today's one GPU: ≈ 2.3× on the thin deck; over one GPU with overlap ≈ 1.8×.
CPU: unchanged (pool-only code).

### 5.4 Risks

- Disabling CUDA.jl's implicit synchronisation is unsafe for an array written on one stream
  and read on another before a synchronisation: the rule is restricted to arrays complete
  at upload (inputs) and to accumulator twins read only after the job; a test runs two lanes
  on a shared accumulator and compares against one lane, and a debug mode re-enables
  synchronisation for comparison.
- Two concurrent 255-register kernels share the eight warp slots of an SM: overlap fills
  tails, it does not add occupancy.
- Four host CPUs serve two devices: host phases (offsets, set-up, gathers) do not shrink
  with the device count; the device site scan (item 1) and resident outputs (item 2)
  matter more on two devices.

### 5.5 Work items

| id | work | acceptance |
|---|---|---|
| 3.1 | `share_lanes!` in the device protocol, CUDA method, calls in `cached_twin` and `zeroed` | nsys: on one A100 the summed `MCPrimary` kernel time exceeds the union (overlap) and the thin stride-1 photon phase ≤ 3.4 s; every deck bitwise on one and two GPUs; CPU decks bitwise |
| 3.2 | chunk-rule sweep with overlap on (cascade and image chunks) | table of thin, Compton and torus imaging on one and two GPUs; the rule adopted keeps both rows at their best; bitwise |
| 3.3 | cost-ordered cascade chunk queue | end-of-call device imbalance ≤ 0.1 s on the thin deck; bitwise |
| 3.4 | pool-wide first capacity | Compton deck: no rerun on any lane's first chunk (counted at `cascade.jl:324-328`); bitwise |
| 3.5 | `two_gpu.sbatch`, the decks and the reference legs on `gpu:0` and `gpu:0,1` | `compare_decks.jl` bitwise one against two; speed rows recorded in the Verification record |
| 3.6 | `BLP_TEST_DEVICES=cuda` legs on two devices (`test/decks.jl:163-197`, `test/infra/execution_cuda.jl`) | green in the two-GPU job |

## 6. Item 4: the precise march

### 6.1 Facts

- Comparison B, 500², `Plans.precise` (Verner 9): A100 1.647 s (best of five; the two
  launches 1.603 + 0.127 s, nsys), CPU 4.91 s on 32 threads (**measured**, README).
- ncu, first launch: 255 registers, 5,632 B stack, 4.2 % of instructions spill loads and
  stores, L1 hit 59.8 %, 69 GB of DRAM writes, issue active 18.4 %; stalls per issued
  instruction: wait 2.90, no instruction 2.70, long scoreboard 1.17, math throttle 0.89
  (`profile.md` §2). ptxas: 2,304 / 4,368 B of spill code in the body, four Verner bodies
  per kernel, a non-inlined `afoldl` per step (`map_march.md` §4.4).
- The launch is a full GPU for 0.2 s and a tail of a few blocks for 1.4 s (SMs active per
  0.1 s: 100, 100, 54, 15, 12, 8, …, 1 %; 84 % of the samples below a quarter of the SMs).
  Steps per pixel: mean 351, median 235, p99 2,395, p99.9 8,665, max 27,255 (78× the mean);
  the slowest 1 % hold 13.9 % of the steps (`profile.md` §3).
- So the launch is bounded below by the serial latency of its slowest ray: 1.6 s / 27,255
  steps ≈ 55–59 µs per Verner step of a lone warp (estimate; upper bound, the slowest pixel
  sits in the first chunk, which the cost order launches first). A host thread does a step
  in ≈ 1.8 µs (estimate: 4.91 s × 32 threads over 8.8e7 steps). The area under the
  SM-active curve is ≈ 0.33 s of full GPU: the work of the image fits in a fifth of its
  launch.
- Consequences: registers, spills and occupancy act on the bulk (0.2–0.4 s); only a shorter
  lone-warp step or moving the slowest rays elsewhere shortens a single image; several
  images in flight hide the tail.

### 6.2 Track A: bitwise by construction, now

- **A1. Overlap and batches.** With lanes overlapping (3.1) the second chunk (0.127 s) runs
  in the first one's tail (≈ −8 % on one image). A product call that queues several images
  in one multi-part `foreach_item!` (the form exists, `executor.jl:148-151`: parts share a
  queue) keeps the device busy across tails; with the persistent-thread loop the Monte
  Carlo executor design provides (one launch pulling items through an atomic counter), a
  batch of B images costs ≈ max(B × 0.33 s, 1.5 s) instead of B × 1.65 s (estimate; 4.4×
  at B = 4). The batch is a method of the product on a vector of plans (or of disks), whose
  results are per element bitwise the single calls.
- **A2. The camera ray in its own launch** (a ray program writing `Float64` states, the
  march reading them): recorded bitwise equal to the fused march (`performance.md` §4);
  for this kernel the stack drops 5,624 → 4,944 B and `camera_state`'s 476 / 820 B of
  spills leave (`map_march.md` §4.4, host-ray variant). Expected small; it also removes one
  of the five Verner bodies from the kernel (compile time, §7).
- **A3. Workgroup size** `workgroupsize(::CrossingProgram)`: today the occupancy API picks
  256, one block per SM, so a slow ray holds a whole SM; blocks of 32 (as the Monte Carlo
  programs, 4.34 → 3.72 s there) let the SM take new blocks around it (lane efficiency in
  queue order 0.642 for blocks of 256 against 0.891 for warps of 32, `profile.md` §3).
  Acts on the bulk; the slowest ray is unchanged. Bitwise (A4 of `map_march.md` §8).
- **A4. Register cap**: a launch trait `register_cap(prog)` (default `nothing`) honoured by
  an extension-level launch that compiles `_foreach!`'s function with `@cuda launch=false
  maxregs=r` and KernelAbstractions' context (KA passes no `maxregs`, `CUDAKernels.jl:107-150`).
  Allocation only, but `ptxas` may fuse the 13 non-`.rn` adds and subtractions and the 4
  plain multiplies of these kernels (`audit/*.ptx`) differently: adoption requires the SASS floating-point instruction
  sequence (DFMA, DMUL, DADD with operands' rounding) identical to the uncapped kernel and
  the decks bitwise. GPUmonty's 32-register experiment lost; expected neutral to negative
  for the tail, worth one sweep (128, 168) for the bulk.
- **A5. Clocks.** The SM clock read 1,065 MHz in every sampled range (`profile.md` §1), the
  base clock of the A100 80 GB PCIe, also in the near-idle tail; `nvidia-smi -q -d
  CLOCK,PERFORMANCE` inside a job tells whether application clocks are pinned. If they are,
  the 1,410 MHz boost (1.32×) is an administrator's setting, for every GPU result.

### 6.3 Track B: code shape, after D2

With the Kerr–Schild routes and the far zone free of flagged operations (1.5), changes to
the step's shape are bitwise by construction on both backends:

- **B1. V9Sched** (`map_march.md` §4.4): stage sums opened after `k10`, peak live vectors
  11 → 7, stack 5,624 → 2,776 B, spill code 2,304 / 4,368 → 1,612 / 1,980 B, no `afoldl`.
- **B2. The right-hand side behind one non-inlined call**: one copy of the ≈ 800-instruction
  body instead of fifteen inlined per step (a step is ≈ 12,000 SASS instructions, ≈ 190 KB,
  streamed per step: the no-instruction stalls); recorded 1.80 s against 1.90 s on B when
  it still moved CPU bits (`V9Call`).
- **B3. Crossing landings stored at the crossing** instead of carried in `maps` (−30
  registers of loop state).
- **B4. Deferred crossings**: the march records `(y, h, k)` at each sign change and
  `surface_crossing` runs after the loop (or as a second launch over the recorded states),
  one Verner body in the loop instead of four.

Expected: the lone-warp step loses part of its long-scoreboard and no-instruction stalls
(together 3.9 of ≈ 9.5 cycles between issues in the bulk); single image 1.65 s → ≈
1.1–1.4 s (estimate); compile 33 s → < 10 s (ptxas is 26 s of it, §7.1); CPU speed measured
(the non-inlined right-hand side must not slow the host).

### 6.4 Track C: host continuation of the slowest rays, after D1 and D2 on the Kerr march

With the crossing kernel bitwise CPU = GPU (1.4, 1.5), a ray marched for `W` steps on the
device and continued on a host thread gives the bits of either. The march becomes resumable
(state `y`, `lo`, `r_prev`, `n`, `k`, `maps`, as `TraceProgram` does for the tracer,
`programs.jl:106-178`); the device runs every ray for at most `W` steps, the survivors
continue on the host threads, the device moves to the next work. Model on the measured per-pixel steps
(`aux/tail_model.jl`, `aux/tail_model.log`; device 57 µs per step of a lone warp, host
1.8 µs per step and thread): in a 4-CPU job `W = 12,000` leaves 129 survivors with
639,588 steps, ≈ 0.68 s on the device and ≈ 0.29 s on the host, ≈ 0.97 s against 1.65 s;
with 16 host threads `W = 5,000–8,000` gives ≈ 0.6 s (estimates). At the 3.3 µs per host
step measured on the loaded shared host during this design, the 4-thread figure is ≈
1.1 s. `W` is an executor tuning (results do not depend on it, by portability). The host share uses
the run's host threads, so it belongs to the pool executor, selected by nothing but the
resolved executor.

Rejected: stage vectors in shared memory (programs get no KernelAbstractions context, B3 of
`map_execution.md` §8, and V9Sched reaches 7 live vectors without it); a kernel per Verner
stage (the fine-wavefront case: 2.0–7.7× slower in Dr.Jit, state traffic of 15 × 64 B per
step, and the one-trajectory-per-thread base is what DiffEqGPU.jl and GRay support,
`map_literature.md` §2.6, §2.9).

### 6.5 Risks

- Track A gains little on one image: the slowest ray bounds it. The batch API is where A
  pays; its value depends on how often images come in sets (spin and inclination scans,
  the comparison figures).
- B2 on the host: a non-inlined right-hand side may cost host speed (call overhead per RHS
  in a 2-µs step); measured before adoption.
- C needs bitwise CPU = GPU on the whole crossing kernel, including the camera at
  infinity, `surface_crossing` and the landing observables (atan2, asinh).

### 6.6 Work items

| id | work | acceptance |
|---|---|---|
| 4.1 | kernel-audit case for `CrossingProgram` (production, precise; camera and host rays) in `bench/kernel_audit.jl`; the baseline refreshed past `infinity_ray` | `--check` passes at `a5923dd` |
| 4.2 | A3 workgroup sweep (32, 64, 128, 256) and A4 register-cap sweep | decks bitwise; SASS FP sequence identical for every adopted cap; timing table A and B, production and precise |
| 4.3 | A2 camera-ray launch for `CrossingProgram` | decks bitwise; stack and spills per audit; timing |
| 4.4 | A1 batched `render_crossings` / `render_analytic` over several plans, through the multi-part queue | per-element results bitwise single calls on CPU and GPU; four-image batch on the A100 ≤ 2.5 s with overlap (3.1), ≤ 1.8 s with the persistent loop |
| 4.5 | A5 clock query in the next GPU job | recorded |
| 4.6 (D2) | B1–B4, one at a time, after 1.5 | decks bitwise on both backends (by construction, verified); single image ≤ 1.4 s; ptxas time per audit; CPU not slower |
| 4.7 (D1, D2) | resumable crossing march; host continuation in the pool | images bitwise the pure-device and the pure-host runs; ≤ 1.1 s on the 4-CPU job |

## 7. Item 5: compile latency

### 7.1 Facts (**measured** offline, `aux/compile_phases.jl`, `aux/phases_{A,B}.log`)

The first device compilation of a process costs 8–13 s whatever the kernel; the rest is
per program. A trivial kernel compiled first takes 10.3 s with BLP loaded; `ZoneSums`
compiled first 12.9 s, compiled second 2.0 s, compiled again 0.1 s. Phases per program
after the first compilation (inference = first minus second `emit_llvm`; LLVM = codegen and
optimisation; `llc` the external NVPTX backend; `ptxas` the step a launch adds):

| program | inference | LLVM | llc | ptxas | total | interval to module load in a job (`profile.md` §5.2) |
|---|--:|--:|--:|--:|--:|--:|
| first kernel of a process (trivial) | 10.3 (all phases) | | | | 10.5 | — |
| `ZoneSums` | 1.9 | 0.1 | 0.1 | 0.3 | 2.4 | 15.5 (first kernel) |
| `ZoneCounts` | 0.1 | 0.1 | 0.1 | 0.1 | 0.4 | 2.3 |
| `HotcrossSigma` | 0.4 | 0.1 | 0.1 | 0.2 | 0.8 | — |
| `MCPrimary` thin | 2.6 | 2.0 | 0.6 | 3.6 | 8.8 | 10.6 |
| `MCPrimary` Compton | 1.8 | 5.7 | 1.0 | 10.5 | 19.0 | 16.5 |
| `MCLayer` | 1.3 | 3.3 | 2.1 | 10.4 | 17.1 | ≤ 21.4 |
| `ImagingProgram` torus I | 4.8 | 1.4 | 0.9 | 6.8 | 13.9 | 14.3 |
| `CrossingProgram` RK4 | ≈ 0 | 2.2 | 0.3 | 10.3 | 12.8 | 15.0 |
| `CrossingProgram` Verner 9 | 1.0 | 5.1 | 1.2 | 25.9 | 33.2 | 35.6 |

(Inference of a program compiled after others is reduced by what they share.) So "the
16 s device compile of the zone tables" is the process's first-compilation cost (≈ 13 s)
plus ≈ 3 s for the three table programs.

The first-compilation cost comes from loading ForwardDiff (`aux/warmup*.jl`,
`aux/warmup*.log`; a trivial kernel compiled to PTX as the first device compilation, after
loading the named packages before CUDA):

| loaded first | first compile | loaded first | first compile |
|---|--:|---|--:|
| nothing (KernelAbstractions kernel) | 1.7 s | ForwardDiff | 7.1 s |
| nothing (plain function) | 1.3 s | StaticArrays | 1.4 s |
| BlackLightPlus (plain / KA) | 8.5 / 8.3 s | SpecialFunctions | 0.9 s |
| OrdinaryDiffEq | 8.3 s | SciMLBase | 1.0 s |
| ForwardDiff, StaticArrays, SpecialFunctions | 6.8 s | HDF5, FFTW, JacobiElliptic | 2.3 s |

The mechanism (inference): ForwardDiff's methods invalidate natively compiled code of the
GPU compiler pipeline that CUDACore's precompile workload cached, so the first compilation
re-infers and re-compiles it on the host.

GPUCompiler 1.23 has a disk cache of compiled kernels (`disk_cache` preference,
`GPUCompiler/src/execution.jl:99-119, 159-270`): it stores CUDACore's compile output (the
cubin after `ptxas`) keyed by the build id of the kernel's `CodeInstance` and the compiler
configuration, and it applies only to `CodeInstance`s that come from a package image,
i.e. inferred during precompilation; a runtime-inferred kernel is never cached.

### 7.2 Design

1. **Precompile workload in `BlackLightPlusCUDAExt`** (PrecompileTools `@compile_workload`,
   no GPU needed, as CUDACore's own workload: `GPUCompiler.compile(:asm, job)` for an
   sm_80 configuration built as `_compiler_config` builds it): compile a trivial program
   through `_foreach!` and the fixed-type programs whose device types do not depend on a
   deck: `ZoneSums`, `ZoneCounts`, `HotcrossSigma`, `Drain` for the Monte Carlo
   accumulators, `BandSum`/`Fill` of item 2. The extension's image is built with
   ForwardDiff and CUDA loaded, so the pipeline code it caches is valid in the world a BLP
   session runs in. Expected: first compilation 8–13 s → ≈ 1–2 s (the no-ForwardDiff
   figure); table programs 3 s → their inference cached.
2. **Disk cache of those kernels.** The environment preference
   `[GPUCompiler] disk_cache = "true"` (in the GPU environment's `LocalPreferences.toml`
   beside the CUDA runtime pin; documented in `guide/gpu.md`): the workload's
   `CodeInstance`s come from the extension image, so their cubins are cached on the first
   launch and loaded thereafter (no inference, LLVM, llc or ptxas). Cache files under
   the depot's scratch space (root file system, 67 GB free; ≈ 0.1–3 MB per kernel).
3. **Opt-in workloads for deck programs** selected by a package preference listing deck
   families (Monte Carlo torus thin and Compton, torus imaging I and IQUV, Kerr crossings
   production and precise, closed form): each adds its compile time (9–33 s, table above)
   to the extension's precompilation once and removes it from every later process. The
   types must equal the runtime twin types exactly; a test launches each family and asserts
   no compilation happened (`GPUCompiler.compile_hook`).
4. **Compile-ahead.** A device-protocol `prepare!(lane, dprog, items, outs)` compiles a
   program's kernel without launching (CUDA method: `cufunction` of `_foreach!`'s function
   with the KernelAbstractions context type, as `bench/kernel_audit.jl` `device_resources`
   does); fallback no-op. The cascade calls it for the layer and the drain on a host task
   when it starts the primary generation (the cold Compton call waited 21.4 s for
   `MCLayer` after its first primary launches, `profile.md` §5.2, behind ≈ 33 s of primary
   kernels), and a grid run for `MCPrimary` while the host builds offsets and set-up.
   The same code is compiled, so the bits are those of a lazy compilation.
5. **Smaller kernels**: item 4 B1, B2, B4 cut the precise kernel's PTX (202,926 lines;
   RK4 95,309) and its 26 s of `ptxas`.

CPU: none of these touches a host path; the extension's precompilation grows by about
10 s for (1) and by the chosen families for (3).

### 7.3 Expected gains

| call | today (**measured**, `profile.md` §5.1) | after 1–2 (estimate) | after 1–4 (estimate) |
|---|--:|--:|--:|
| thin `mc_spectrum`, cold | 50.1 s | ≈ 36 s (−11 s first compile, −3 s tables) | ≈ 25 s with the thin family precompiled |
| Compton, cold after thin | 101.5 s | unchanged | ≈ 80 s (layer compiled ahead) |
| torus imaging, cold | 14.95 s | ≈ 14 s | ≈ 1 s with the imaging family |
| precise crossings, cold | 37.3 s | ≈ 36 s | ≈ 2 s with the crossing family; ≈ 12 s after item 4 B |

### 7.4 Risks

- The workload's configuration must equal the runtime one (`cap`, `ptx`, `always_inline =
  false`, `kernel = true`, no `maxthreads`), or the disk cache misses silently: the test of
  (3) catches it.
- Another package loaded after BLP that invalidates the pipeline again restores the first
  compilation cost; the precompiled program `CodeInstance`s are then invalid and recompiled
  (correct, only slower).
- A cubin from the cache must be the cubin a fresh compilation gives: byte comparison in the
  test, and the decks.
- Precompilation without the NVPTX backend (another machine) skips the workload, as
  CUDACore's does.

### 7.5 Work items

| id | work | acceptance |
|---|---|---|
| 5.1 | confirm the invalidation (SnoopCompileCore `@snoop_invalidations` if available offline; else the load-order bisection of `aux/warmup2.jl` extended to ForwardDiff's method groups); report upstream | the invalidated methods listed |
| 5.2 | extension workload: trivial program and fixed-type programs | in a fresh process with BLP and CUDA loaded, the first device compilation ≤ 2 s offline; cold thin call in a job ≤ 37 s; decks bitwise |
| 5.3 | disk cache preference, documentation, cache-hit test | second process: table programs load without compilation; cubins byte-identical to a fresh compilation |
| 5.4 | `prepare!` and its calls in the cascade and grid runs | cold Compton call ≤ 85 s; bitwise |
| 5.5 | opt-in deck families by preference | per family: a launch triggers no compilation; precompile time recorded |

## 8. Order, team and verification

### 8.1 Dependencies

- 3.1 (lane overlap) first: it is a prerequisite of 4.4 and of the Monte Carlo executor's
  multi-lane behaviour, and it touches `residency.jl` and the extension, which 2.3 and 5.2
  also touch.
- 2.1 and 2.2 are independent; 2.3–2.4 after 3.1 (both edit `residency.jl`).
- 5.1–5.3 independent; 5.4 after the Monte Carlo executor design fixes the cascade's
  structure (it calls into the cascade).
- 1.1–1.2 now; 1.3–1.4 after D1; 1.5–1.6 after D2; 1.7 after 1.3 and 1.6, and benefits
  from 2.3 (resident sites); 4.6 after 1.5; 4.7 after 1.4, 1.5 and 4.6.
- Files edited by several items: `src/infra/Execution/residency.jl` (3.1, 2.3),
  `executor.jl` (2.3, 3.2), `cascade.jl` (3.2–3.4, 5.4, and the Monte Carlo executor),
  `ext/BlackLightPlusCUDAExt.jl` (3.1, 2.3, 4.2, 5.2, 5.4): one owner per file at a time,
  small commits, rebased in the order above.

### 8.2 Work packages for a team

| package | items | first deliverable | GPU jobs |
|---|---|---|---|
| Pool and devices | 3.1–3.6 | lane overlap with the thin deck's nsys check | 3 (one A100 overlap; two-GPU sweep; records) |
| Outputs | 2.1–2.5 | threaded epilogues (2.1), bitwise on both backends | 2 |
| Portability | 1.1–1.8 | the gate in report mode and the GPU bit check | 1 now; one per tier |
| March | 4.1–4.7 | crossing audit case and the workgroup and cap sweeps | 2 now; more after D2 |
| Compile latency | 5.1–5.5 | the extension workload | 1 |

Every package runs the bit suite (§8.4) before each commit and reports speed on both
backends. Figures the packages make are rendered and looked at before they are reported.

### 8.3 GPU job plan

Jobs run on `preempt` (≤ 1 h, 4 CPUs, preemptible without requeue, one job on the node at a
time, another agent in the queue): legs are grouped so that a job carries one package's
speed rows and its deck comparison together; a two-GPU job runs its one-GPU legs too
(`gpu:0` and `gpu:0,1` in one process). Each job uses `run_job.sh` with a pattern line,
writes logs under the package's directory in `/data2/haiyangw/scratch/wavefront/`, and is
waited on with `squeue -j <id>`.

### 8.4 The bit suite

- Reference: the fixed decks of `results/blp/mc_audit/speed/gpu/scripts/decks.jl` run once
  from the pinned tree `/data2/haiyangw/scratch/wavefront/ref_main` (`a5923dd`) on 1 and
  16 CPU threads and on one A100, outputs under `/data2/.../wavefront/decks/ref_*`. The
  environment `env_ref` points at the main checkout's working tree, which moves with the
  user's commits; the reference needs an environment whose BlackLightPlus path is
  `ref_main`.
- Per change: the same decks from the worktree; `compare_decks.jl` bitwise on each backend;
  one GPU against two GPUs bitwise; CPU against GPU no larger than §1 per deck; the test
  suite including `test/structure.jl`, `test/portability.jl` (once it exists) and
  `BLP_TEST_DEVICES=cuda` legs in a GPU job.
- Under D1 or D2 the reference is re-taken once per tier, and the re-baseline record lists
  per deck the differences against `a5923dd` on each backend and the new CPU–GPU gap.

## 9. Evidence index (`/data2/haiyangw/scratch/wavefront/plan/aux/`)

| file | what |
|---|---|
| `compile_phases.jl`, `compile_phases_defs.jl` | offline compile phases per program kind (inference, LLVM, llc, ptxas), kinds captured as `bench/kernel_audit.jl` captures them |
| `phases_A.log`, `phases_B.log` | two kernel orders: `ZoneSums` first, and a trivial kernel first |
| `warmup.jl`, `warmup.log` | first device compilation with and without BLP loaded, KA kernel and plain function |
| `warmup2.jl`, `warmup2.log` | the same after loading single dependencies (ForwardDiff is the one) |
| `census_kernels.jl`, `census.log`, `census2.log` | libdevice calls, `fmuladd`, `contract`/`reassoc` instructions per kernel (libraries = false device IR) |
| `tail_model.jl`, `tail_model.log`, `precise_B_steps.bin` | per-pixel steps of the precise march of comparison B on host threads and the host-continuation model of §6.4 |

Per-device kernel times of the two-GPU job were read from
`profile/raw/twogpu_3795.sqlite` (`CUPTI_ACTIVITY_KIND_KERNEL`, grouped by call) for §5.1.
No repository file was modified and nothing was written under `/data`.
