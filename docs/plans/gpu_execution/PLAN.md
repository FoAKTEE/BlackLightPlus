# GPU infrastructure: work plan

The plan that lands the standard `docs/src/design/gpu_execution.md` (branch
`gpu-wavefront`, commit `46eced3` on top of main `f6decd6`): a device wavefront for the
Monte Carlo photon phase, gated per phase against the cascade it replaces; the pool and
launch items that are bitwise by construction; a bitwise fallback for the Monte Carlo if
the wavefront is falsified; the host-side items that speed both backends, starting from a
CPU profile; and the gates every item passes. Inputs: the profile
(`/data2/haiyangw/scratch/wavefront/plan/profile.md`, `profile.json`, `profile/logs/`),
the maps (`map_execution.md`, `map_mcprograms.md`, `map_march.md`, `map_compiler.md`,
`map_outputs.md`, `map_literature.md`) and the four designs (`design_event.md`,
`design_incremental.md`, `design_history.md`, `design_aux.md`), all in
`/data2/haiyangw/scratch/wavefront/plan/`. Section numbers `S§n` refer to the standard.
Every number below cites its record; a number without a measurement of the current tree
is marked *estimate* and names the item that measures it.

## 0. Decisions taken by this plan

- **MC executor: the segment wavefront of `design_event`, gated per phase.** Built on the
  bitwise-by-construction pool items of `design_incremental` (Tier 1). Its premise, that a
  kernel exit every K steps bounds the two-to-four-group split of spawning warps (87 % of
  the Compton warp-seconds once the reruns are gone, inferred from counters in profile
  §4.3), is tested by W2's timed leg on the exact instrument (a resumable K-step march over
  a compacted list) before W3 and W4 are built, and the go of W3 is F3b's prediction with
  measured inputs. The wavefront serves a phase (thin, scattering) only if it is faster
  than the cascade of the same tree on one A100 and on two (S§9); otherwise that phase
  keeps the cascade. The CPU keeps the history schedule.
- **Window Monte Carlo runs on threads only.** `mc_spectrum` rejects a `window` on any
  executor but `ThreadedExecutor` (`src/pipeline/Products/montecarlo.jl:305-306`, asserted
  by `test/pipeline/products_montecarlo_time.jl:314`). The window decks are CPU decks
  (gates C, C′); the GPU gates D and E use the device decks of the manifest. The earlier
  W9 (timed runs on the wavefront) is not in this plan: lifting the restriction is a new
  feature with its own design.
- **A fallback that needs no `track!` change.** If W1 or W2 moves bits, F3b says no-go, or
  a phase stays on the cascade, B1 (children and primaries of the device cascade ordered by
  `photon_key`, `design_incremental` B3/B4, Compton 40.4–45.8 s with keyed children,
  `design_incremental.md` §6.3) starts at once on top of T1 and T2.
- **CPU line.** A CPU profile at `f6decd6` (P1) sets the CPU targets; the metric reuse
  (C2) is decided before W1, so `MarchState` is fixed once; the child order on threads
  (C1) and a generation item (C3) follow P1. Every CPU-path item passes the paired CPU
  protocol of §8.4.
- **Grafted:** from `design_history` the re-recorded references (G0), the per-photon
  records keyed by stream identity (G1), `photon_key` (no Matter protocol; K1, owned
  independently of the sort) and the inner-shells-first refill order (W8); from
  `design_incremental` the device budget for child slots (T1), the offline device-code
  screen (G2), the ordered-launch experiment (F2), the register-cap launch (T3) and the
  ordered cascade (B1); from `design_aux` lane overlap (`share_lanes!`, narrowed to
  read-only arrays, T2), the threaded host epilogues (A1), caller storage (A2, optional),
  compile latency (A4), the clock check and the two-GPU legs of every gate.
- **Dropped, with the reason in S§10:** the persistent regenerating loop, the claim loop
  inside `MCAdvance`, `design_history` A5's persistent warps around `track!` and the warp
  vote (no reconvergence point); the second lane for
  survivors (`pool.jl:196-199` refuses a pool job from a pool worker; two wavefronts on a
  device double the tail); lanes sharing written arrays; the untracked `ReadOnlyArray`
  conversion; the per-parent `first_lost` overflow; pilot-march step keys; the Matter
  `region` protocol; `PoolArray`/`resident` outputs; the strided and cost-ordered cascade
  chunks; the separate camera launch of the crossing program (a class-B cut for at most
  2.5 % of headroom, profile §3); every item that moves bits on a backend (`port_hypot`,
  explicit reductions, the device site scan, Verner rescheduling, host continuation of
  slow rays): those wait for the user's decisions D1–D3 (§9).
- **Base.** `gpu-wavefront` was fast-forwarded from `a5923dd` to main `f6decd6` before the
  standard was committed; main has not moved since (`git log f6decd6..main` empty on
  2026-11-07). Every bitwise reference is main at `f6decd6`, re-recorded in G0. The
  profile's numbers (taken at `a5923dd`) are the evidence of the design; `b71d7b4`
  (generate `@noinline`), `15f56a2` (emission tetrad) and `9903dee` (window MC) move photon
  bits, so every per-photon input (step counts, genealogy) is re-recorded at `f6decd6` in
  G1 before F2 and F3b use it, and G2's baseline is `f6decd6`'s own dump. Main's Monte
  Carlo moves fast (12 commits touching `src/physics/MonteCarlo`, `montecarlo.jl`,
  `programs.jl` or `Execution` on 2026-11-07): the lead asks the user for a freeze or a
  rebase window on `track.jl`, `sinks.jl`, `programs.jl`, `montecarlo.jl` and `cascade.jl`
  with the Monte Carlo audit team for W1–W4 (D6), and L1 re-records the references at
  main's tip before anything lands.

## 1. Goals, baselines and targets

Baselines: `results/blp/mc_audit/speed/gpu/README.md` and the profile, trees `a5923dd` /
`2c50d86`, one A100 80 GB PCIe in a Slurm job with 4 host threads (SM clock 1,065 MHz),
CPU on `anta`. G0 re-records every row at `f6decd6`. **A target is the ratio to its G0
baseline** (A100 target over the A100 baseline, CPU over CPU); the seconds in brackets are
the ratio applied to today's number, for information. "Kept" is defined in §8.3 (GPU) and
§8.4 (CPU).

| quantity (deck) | CPU baseline | A100 baseline | A100 target (ratio) | CPU target | items |
|---|--:|--:|---|---|---|
| thin photon phase (torus 00999 stride 1, 10⁶ primaries) | 11.3 s, 32 thr | 3.74 s | T2 ≤ 0.88 (3.3 s), standing until W3; W3 gate ≤ 0.53 (2.0 s); W8 ≤ 0.43 (1.6 s) | kept; C3 per P1 | T2, W3, W8, C3 |
| Compton photon phase (bias 10⁴, 3 orders, 7.30 M children) | 62.5 s, 32 thr | 62.5 s | T1 ≤ 0.86 (54 s); T1+T2 ≤ 0.80 (50 s), standing until W4 (permanent if W4 falls); W4 gate ≤ 0.24 (15 s); W8 ≤ 0.19 (12 s) inline kick, ≤ 0.15 (9 s) deferred; fallback T1+T2+B1 ≤ 0.74 (46 s) | kept; C1 per P1 | T1, T2, W4, W6, W8, B1, C1 |
| thin / Compton phase on two A100s | — | G0 (J3) | ratio to one A100 of the same tree: thin ≥ 1.4, Compton ≥ 1.6 (provisional; fixed after J3 and F3b) | — | W4, W8 |
| warm thin `mc_spectrum` call | 10.5 s | 4.1 s | ≤ 0.59 (2.4 s) | kept | W3, W8 |
| cold thin `mc_spectrum` call (first of a process) | — | 50.1 s | A4 in compile terms (§3.6): first device compilation, host time before the first kernel and module-load intervals shorter with A4 than without, same tree; the cold call reported | — | A4 |
| cold Compton call after a thin call | — | 101.5 s | reported; A5 only while a device cascade serves a phase | — | A5 |
| torus Stokes I 768² | 21.6 s, 16 thr | 0.457 s | kept | kept | T2, T3 |
| marched disk B 500², production | 3.39 s, 32 thr | 0.163 s | T2 ≤ 0.90 (its two launches, 0.087 + 0.067 s, overlapping) | kept | T2 |
| marched disk B 500², precise (Verner 9) | 4.91 s, 32 thr | 1.647 s | T2 ≤ 0.94 (1.55 s; the 0.127 s second launch overlapping), credited to T2 alone | kept | T2 (A3 reported only) |
| closed-form Kerr disk 2048², radius path | 0.52 s, 32 thr | 0.305 s | A1 ≤ 0.72 (0.22 s); A2 ≤ 0.26 (0.08 s) | A1 ≤ 0.87 (0.45 s); A2 ≤ 0.73 (0.38 s) | A1, A2 |
| site scan, stride 1 | 0.88 s, 32 thr | 2.0 s (host, 4 thr) | unchanged (needs D1 + D2) | unchanged | — |

Expected after the plan, one A100 (*estimate*, from `design_event` §6 with the emission
rate, launch gaps and full-lane step cost treated as unknown until F4c, F2 and W2 measure
them; F3b replaces these numbers): thin 1.4–1.8 s, Compton 9–15 s with the inline kick,
7.5–9 s if F1 attributes the split to the kick; two A100s halve the bulk but not the tail
(S§4.6). If the wavefront is falsified: Compton 40–46 s (T1 + T2 + B1), thin about 3.3 s
(T2). CPU: the host epilogues, caller storage, and the P1-ranked candidates C1–C3.

## 2. Team, branches, environments

| agent | owns | items |
|---|---|---|
| **lead** | the DAG, merges into `gpu-wavefront`, the tip gates after merges, rebases, raises D1–D6 | L1 (with bench) |
| **bench** | references, gates, the GPU queue (submitter), records | G0, G1, G3, the gate legs of every item, V1 (gates) |
| **bench2** | timing and profiles; second submitter when bench is unavailable | F4, W5, the §8.3–8.4 legs of every item, V1 (timing table) |
| **probe** | device-code and model analysis; never edits the package | G2, F1, F3 |
| **probe2** | experiments on the cascade; analysis of W2's timed leg | F2, U1, W2 (timing analysis) |
| **pool** | `src/infra/Execution` pool, devices, launch, cascade; the extension's protocol methods | T4, T1, T2, T3, W10, the Execution half of B1 |
| **mc** | `src/physics/MonteCarlo` and every edit to the Monte Carlo programs of `src/pipeline/Products` (`mc_stages.jl`, `mc_order.jl`, `programs.jl`, `montecarlo.jl`) | W1, W2, W6, K1, the Products half of W3, W4, W7, B1 |
| **wave** | every edit to `src/infra/Execution/wavefront.jl`, `wave_queues.jl`, `wave_sort.jl`, `test/decks.jl` (`WaveHost`) | the Execution half of W3, W4, W6 (element type), W7; W8 |
| **host** | the host outputs, the CPU profile and the CPU candidates | P1, A1, A2, C1, C2, C3 |
| **march** | the crossing program, compile latency | A3, A4, A5 |
| **docs** | the design documents | V2 |

**Branches and merges.**
- Integration branch `gpu-wavefront`, worktree `/data2/haiyangw/scratch/blp_wavefront`;
  only the lead commits there (fast-forward or rebase of accepted items, never a merge
  commit of a failing item).
- Each agent that edits the package works in `/data2/haiyangw/scratch/wavefront/wt/<agent>`
  on branch `wf-<agent>`, created from `gpu-wavefront` and rebased onto it after every
  merge. W3, W4, W6, W7 and B1 are accepted on the combination of two agents' branches,
  assembled by the lead in `/data2/haiyangw/scratch/wavefront/wt/integrate`.
- **After every merge** the lead runs gate C (A-B, §8.1) on the tip. **Once per wave** the
  tip passes gates D, E and F in the wave job (§5). A tip failure resets `gpu-wavefront` to
  the previous tip (local branch, lead only), bisects among the wave's merges with the
  failing gate, and returns the item to its owner.
- Items edit only docstrings and the `docs/src/api/*.md` entries of their own exports;
  `architecture.md`, `performance.md`, `gpu_execution.md`, `guide/gpu.md` and
  `CHANGELOG.md` are V2's.

**Environments and the depot.**
- `gates/mkenv.sh <worktree> <name>` copies `/data2/haiyangw/scratch/wavefront/env` (CUDA
  runtime 13.3 in `LocalPreferences.toml`) to `env_<name>`, and `env_test` to
  `envtest_<name>`, with the `BlackLightPlus` path set to the worktree; docs builds use a
  copy of `env_docs`. Never point an environment at the main checkout (`env_ref` does; it
  is not used).
- `gates/envcheck.sh <envA> <envB>` diffs the Manifests (but for the `BlackLightPlus` path)
  and the `LocalPreferences.toml`; every gate header runs it against `env_refmain` and a
  difference not declared by the item voids the gate. An item that changes `Project.toml`
  (A4) re-resolves every env copy offline from the same template and declares the diff.
- `gates/precompile.sh <env>` precompiles on the host before any job is submitted (no
  precompilation inside a GPU job) and lists the cache files it wrote. No other `Pkg`
  operation runs in `~/.julia` for the duration; the lead tells the other agents through
  the user. Every script exports `JULIA_MAX_NUM_PRECOMPILE_FILES=40`: the default 10 cache
  pairs of `compiled/v1.12/BlackLightPlus` would be evicted by about 12 trees.
- `gates/fingerprint.jl` prints the host-bit fingerprint (S§8.1): the bits of
  `hypot(56.95171875, -6.4125000000000005)`, the CPU target of every loaded package image,
  Julia, CUDA runtime and driver versions. Every job header and deck file carries it
  (`map_compiler.md` §3.2: host `hypot` bits follow which image compiled
  `Base.Math._hypot`).
- Reference tree: `/data2/haiyangw/scratch/wavefront/ref_main` (now at `a5923dd`), moved
  to `f6decd6` in G0, with `env_refmain`. Moved only by the lead, with a re-record (R13).
- Scratch: `gates/`, `decks/`, `records/<item>/`, `jobs/<item>/`, `exp/<item>/` under
  `/data2/haiyangw/scratch/wavefront/`. Nothing large under `/data` (about 4 GB free); the
  record summary is copied to `results/blp/gpu_infra/RECORD.md` in V1.

**Rules for every agent.**
- Julia: `JULIA_PKG_OFFLINE=true julia ... script.jl` (never `-e 'Pkg.offline(true)'
  script.jl`). Waits: `squeue -h -j <id>`, `kill -0 <pid>`, or `pgrep -f "[j]ulia"`; never
  a `pgrep -f` pattern that matches its own command line.
- Commits on `wf-*` and `gpu-wavefront`: the Chandra template
  (`/data/haiyangw/claude/Chandra/_common/contracts/commit_template.md`), no attribution of
  any kind (no Co-Authored-By, no Claude, Anthropic or AI mention); never stage
  `figures/blacklight/*` or `figures/fig_spherical_orbits.jl`.
- Prose that `test/structure.jl` rejects: in `src` (case-insensitive) "keep in sync",
  "twin of", "byte-for-byte", "ported from"; in `src`, `ext` and tests "Claude",
  "Anthropic", "Co-Authored", "AI assist", "generated by"; write "the program's device
  twin" and "children spawned by". Files in `src` stay at or below 600 lines: the
  wavefront is split from the start into `wavefront.jl` (driver), `wave_queues.jl`
  (queues, views, counter block, `WaveSwap`) and `wave_sort.jl` (W7); `mc_stages.jl`
  gains `mc_kick.jl` in W6 if it passes 500 lines. Documentation states the standard; no
  narrative of the work.
- Figures (residency, timing plots): rendered and looked at by the agent that made them
  before they are reported.

## 3. Work items

Format: **id — title** (owner; effort; GPU jobs of §5; depends). Goal. Files. Acceptance.
Every item also passes the gates of §8 that its class requires (§8.2); the acceptance
lists what is specific to the item. "Kept" and "faster" follow §8.3–8.4.

### 3.1 Phase G: base, oracle and gates

**G0 — Base and references** (bench; 1.5 d; J1, J2, J3; depends —)
Goal: every comparison of the work is against main at `f6decd6` on each backend, through
scripts that cannot pass on missing data.
- Move `ref_main` to `f6decd6` (detached, clean) and build `env_refmain`; precompile it.
- `gates/decks.toml`: the deck manifest (name, backends, expected leaves). `gates/decks.jl
  <label> <backend> [--with-cuda] [--only mc]`: every deck of the manifest for the
  backend, every field of every result but `plan` and `run` serialised by a generic
  walker, the fingerprint stored with the file. Decks (S§8.1):
  - the 13 fixed decks of `results/blp/mc_audit/speed/gpu/scripts/decks.jl` (today they
    drop `w_primary`, `w_first_order`, `scatter_loss_final`, `w_unspawned`,
    `dropped_power`, and for `mc_sphere_scatt` all but four fields);
  - coverage, threads and devices: `mc_binary_thin`, `mc_binary_scatt` (static
    `SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)])` of
    `products_montecarlo_time.jl:288`, about 2 × 10³ primaries);
  - ledger decks through `run_layers` (as `mc_phases.jl` calls it): `rl_torus_thin`,
    `rl_torus_compton` (stride 8, 2 × 10⁴), `rl_sphere_scatt`, each with the bins, the 11
    `ledger_sum` and 11 `ledger_count` values over the stripes (never stripe by stripe,
    S I8) and the child count;
  - step-limit decks: `rl_torus_compton_nmax64` (`CurvedMarch(...; nmax = 64)`) and
    `rl_sphere_scatt_nmax8` (`FlatMarch(R, 0.05, 8)`), so `march_steps`' `last == M` →
    `DROPPED` edge runs on both backends;
  - window decks, threads only: `mc_torus_window` (stride 8, 2 × 10⁴, seed `0x5eed0706`,
    `window = (0.0, 100.0)`, `time_bins = 16`), `mc_sphere_window_scatt` (seed
    `0x5eed0707`, `window = (0.0, 1.0e7)`, `time_bins = 10`), `mc_vsphere_window`
    (`VariableSphere`, `StaticFrame` rates), `mc_corotating_window` (`Corotating` on the
    moving `BoostedSuperposedKerrSchild`), `mc_frames_window` (`FrameSequence`), each
    about 2 × 10³ primaries, from the setups of `products_montecarlo_time.jl`.
  25 CPU decks, 20 device decks.
- `gates/compare_decks.jl a b [--metric]`: strict (S§8.1): a key missing in either file,
  a different leaf count, shape or element type, or a fingerprint mismatch fails; prints
  the decks and leaves compared and requires the manifest's counts; exits non-zero on any
  failure. `gates/compare_phases.jl`: the same for `mc_phases` outputs (no `zip`
  truncation, a missing leg file fails). `gates/mc_phases.jl`: serialises bins, the 22
  ledger values and the child count of every leg. Copies of `deck.jl`, `paths2.jl`,
  `run_job.sh` into `gates/` (the main checkout's copies are read, never edited). Gate
  scripts test exit statuses, never grep for a success line.
- Records: CPU decks at `-t 16` twice and `-t 1`, pinned (`ref_t16a`, `ref_t16b`,
  `ref_t1`), and with CUDA loaded (`ref_t16_cu`); the device decks on `gpu:0` (J1,
  `ref_gpu1`) and on `gpu:0` and `gpu:0,1` in one process (J3, `ref_gpu2`); the stride-1
  thin and Compton phases (bins, ledger, child count, three warm calls each) on one A100
  (J2) and on two (J3), and on 32 pinned CPU threads; the imaging legs of `paths2.jl` on
  the A100 and the CPU; a short nsys of the thin and Compton launches at `f6decd6` (J2);
  `nvidia-smi -q -d CLOCK,PERFORMANCE` at the start and end of every job.
- CPU noise floor: five interleaved A-A pairs of `ref_main` against itself for the thin
  and Compton phases and torus imaging under §8.4 → δ per row (`records/G0/noise.txt`).
- Acceptance: `ref_t16a` = `ref_t16b` = `ref_t1` = `ref_t16_cu` (strict compare, exit 0);
  `ref_gpu1` = `ref_gpu2` (one and two devices in J3); the per-deck CPU–GPU gap table
  under the metrics of S§6 in `records/G0/gaps.txt`; every §1 baseline at `f6decd6` with
  job ids, clocks and fingerprints in `records/G0/README.md`; whether the application
  clocks are pinned below the 1,410 MHz boost recorded (D5).

**G1 — Photon records** (bench; 1 d; J4; depends G0)
Goal: a bit oracle that locates a difference to one photon, and the per-photon inputs of
F2 and F3b at `f6decd6`.
- `gates/recorder/record.jl`: `run_layers` on a deck with a forwarding march control and a
  forwarding sink that forward the whole `Sink` protocol explicitly (`record!`, `tally!`,
  `child_stream`, `spawn!`, and `branch!` once W1 adds it), recording at `march_close`
  whatever the schedule, per photon: stream identity (layer tag, primary index,
  ordinals), entry photon, end `x`, `p`, `w`, outcome, `nev`, step count, the five ledger
  values; children by stream identity; sorted by identity.
  `gates/recorder/compare_records.jl` exits non-zero at the first differing photon and
  field.
- Decks: `mc_torus_thin`, `mc_torus_compton`, `mc_sphere_scatt` (references `rec_ref_gpu`,
  `rec_ref_cpu`); the stride-1 timing deck, thin and Compton (`rec_s1_gpu`, the step
  counts and genealogy for F2 and F3b), with the fraction of photons whose step counts or
  children differ from `plan/profile/data/probe_s1/*.bin` (`a5923dd`) reported; the step
  counts of the Ns 1.6 × 10⁵ Compton deck (F1's denominators).
- Acceptance: two runs bitwise on the A100; on the A100 the recorder's bins equal the
  plain run's. On the CPU the forwarding types are other method instances (S§8.1): the
  recorder's difference from the plain run is recorded, and CPU records serve for
  localisation only. A test that the kick placement recorded by the forwarding sink
  matches production (after W1).
- Maintenance: bench updates the recorder in W1 (`branch!`) and W3 (records at `MCRetire`'s
  `march_close`); both are part of those items' acceptance.

**G2 — Offline device-code screen** (probe; 1 d; 0 jobs; depends T4)
Goal: locate, without a GPU, a change of compiled context in the device programs.
- `gates/devcode_gate.jl <env> <label>` through `AuditExecutor`/`AuditDevice` of the
  repaired `bench/kernel_audit.jl` (or the forwarding of `plan/audit/march_audit2.jl`),
  never a `CUDADevice`: for each program type (thin and Compton `MCPrimary`, `MCLayer`,
  `CrossingProgram` production and precise, the stage programs once they exist) the PTX,
  CUDA 13.3 `ptxas -O3 -v -arch=sm_80` and `cuobjdump -sass`; per function registers,
  stack, spill bytes; floating-point operations of each kernel entry; counts of unrounded
  `mul.f64`, `add.f64`, `fma.f64`; `DFMA`, `DMUL`, `DADD` counts with their rounding and,
  with `--sequence`, their order; `contract`/`reassoc` counts of the LLVM IR.
  `--against <label>` compares two trees.
- Acceptance: the baseline is `f6decd6`'s own dump; its differences from the `a5923dd`
  profile (255 registers, `track!` a `.func`, stacks and spills of profile §2) are
  reported, not failed; the screen passes against itself.

**G3 — Gate runner and job templates** (bench; 1 d; validated in J1–J4, gate F in J11;
depends G0)
- `gates/cpu_gates.sh <worktree> <env> <label>`: gate A (`<worktree>/test/runtests.jl` in
  the item's `envtest_*` copy, never `--project=.`), B (strict docs), C and C′ (§8.1),
  each by exit status.
- `gates/gpu_gates.sbatch <worktree> <env> <label> [legs]` (`-p preempt --gres=gpu:1 -c 4
  -t 01:00:00`, body `gates/run_job.sh` with the fingerprint and `envcheck.sh` in the
  header): gate D, gate R, optional gate G and timing legs in A-B-A order.
- `gates/gpuF.sbatch <worktree> <env>`: gate F alone, about 25 min (22 min 25 s at
  `5d8e377`, `results/blp/mc_audit/speed/gpu/logs/unit_3790.log`, plus the newer test
  files), a copy of `results/blp/gpu/unit_tests_main.jl` whose `ROOT` comes from
  `BLP_TEST_ROOT` (the original derives it from `@__DIR__` and would resolve the scratch
  directory) and whose done-markers live under `jobs/F/`, so a preempted run resumes.
- `gates/gpu2_gates.sbatch` (`--gres=gpu:2`): gate E (§8.1) and the two-GPU timing legs.
- `gates/gap_check.jl`: the per-class metrics of S§6 (per-cell for Monte Carlo bins,
  largest-magnitude-relative for images, integer-valued fields exact, equal NaN and Inf
  masks), the branch's CPU–GPU gap ≤ the reference's per deck (`records/G0/gaps.txt`).
- `gates/submit.sh <sbatch> <priority>`: the only way to submit; logs to `jobs/queue.log`
  and enforces §5's rules.
- `gates/mkenv.sh`, `envcheck.sh`, `precompile.sh`, `fingerprint.jl`,
  `timing_cpu.sh <ref> <refenv> <wt> <env>` (§8.4).

### 3.2 Phase F: falsifiers and measurements (no package edits)

**F1 — Kick isolation** (probe; 1 d; J5; depends G1, G2)
Goal: decide whether the Compton sampler causes the split of spawning warps.
- Offline: SASS of `track!` in the Compton `MCPrimary` at `f6decd6`: reconvergence
  instructions (`BSSY`, `BSYNC`, `WARPSYNC`, `YIELD`) around `step_events` and the
  sampler's loops, against the thin kernel (17 against 5 `YIELD` recorded).
- A100, the Compton deck at Ns = 1.6 × 10⁵ (primaries only), ncu as profile §2, each leg in
  its own process: (a) plain; (b) `Opacities(...; weight_min = Inf)` (no kick, no spawn);
  (c) a script method `scatter_superphoton(p, zone, rng) = (p, 0.0, rng)` (sampler
  absent, `spawn!` kept). Warp instructions per warp-step (over G1's step counts), active
  threads per instruction against lanes still marching, warp time per step.
- Legs (b) and (c) change the physics by design: exempt from the bit check.
- Decision on (c): below 8,000 warp instructions per warp-step (from 15,665), or active
  threads within 1.3× of the lanes marching → the kick causes the split: W6 on the path.
  At or above 12,000 → W6 dropped. In between: W6 after W8. (b) − (c) attributes the
  spawn's counter and store.

**F2 — Ordered launches on the cascade** (probe2; 1.5 d; J6; depends G1, G3)
Goal: the full-lane cost factor f (per-warp-step cost at about 32 live lanes against the
measured partial-lane cost), which every estimate assumes.
- A script override of the pool's `chunk_generations!(lane, ...)` that reads each
  generation's children to the host, permutes them by (i) G1's step count at `f6decd6`,
  longest first, (ii) `photon_key` (a script copy of S§4.5), (iii) a meshblock key,
  (iv) arrival order, and launches the unchanged `MCLayer` over the permuted slots; the
  primaries through `order` by the same keys. Timing deck at stride 1. An ncu leg with the
  memory metrics (sectors per request, L1 hit) on one layer launch under (ii) and (iii),
  for W7's key choice.
- Acceptance: a per-launch table (items, span, warp time per step, E) under `records/F2/`;
  bins, ledger and child count bitwise in every leg (I5).
- Result: f, and the response of the non-spawning generation 3 to 1/E (within 15 %
  confirms the cost model). F2 does not decide the segment regime; W2's timed leg does.

**F3 — Schedule replay** (probe; F3a 1 d on days 0–1, F3b 0.5 d after F2, F4c and W2; 0
jobs)
Goal: the go of W3–W8 on a model fed with measured inputs.
- Host model on the recorded step counts and genealogy: 668–750 resident warps × 32 lanes
  (a range, profile §4.3); the segment schedule at K ∈ {8, 16, 32, 64, 128, 256, 512}, P ∈
  {2^15, …, 2^19}, R ∈ {2^12, 2^14, 2^16}, with the refill rule of S§4.3; per-warp-step
  cost by class; emission rate; host time per launch and per iteration; lone-warp step
  latency (thin and spawning Compton primaries); children spawned along the parent's life
  from the genealogy; the same model for the cascade, with and without ordering (B1).
- F3a (days 0–1): inputs from `a5923dd` records and stand-ins, each labelled: emission
  1.2–1.4 × 10⁶ s⁻¹ (*estimate*, one unlogged line,
  `results/blp/mc_audit/speed/PLAN.md:134-135`); host time per launch from the warm Compton
  call (11 gaps above 1 ms, 0.079 s, 627 synchronisations for 40 kernels,
  `profile/logs/tl_analyze_3797.log:104-110`); lone-warp latency 25 µs per thin step
  (derived, `design_event.md` §6.1) and 135–160 µs per Compton-primary step (2.7–3.2 s for
  20,000 steps in the history form, profile §4.3). F3a informs; it stops nothing.
- F3b (day 5): G1's `f6decd6` records, F2's f, F4c's emission rate, launch overhead,
  lone-warp latency and append cost, W2's segment cost per warp-step.
- Go: a predicted thin phase ≤ 1.8 s and Compton phase ≤ 15 s on one A100 at some
  (K, P, R). No-go → W3–W8 stop; B1 starts; T1 and T2 carry the device gain.
- Acceptance: E per iteration, bulk, tail and total per (K, P, R) for thin and Compton,
  and the predicted two-GPU ratios, in `records/F3/`.

**F4 — Pool knobs and device micro-benchmarks** (bench2; 2 d; J7, J8, J9; depends G0,
G3)
Goal: the Tier-1 gains and the four device costs every wavefront estimate needs.
- F4a (J7), script overrides on the timing deck, one A100: (a) `capacity = 4_000_000`;
  (b) (a) plus `CUDA.enable_synchronization!(a, false)` on the read-only residency arrays
  of the device twins only, accumulator twins per lane (T2's narrowed rule); nsys.
- F4b (J8): a script capped launch (`@cuda launch=false maxregs=r` with
  KernelAbstractions' context) at r ∈ {255, 192, 168, 128} for thin `MCPrimary`, Compton
  `MCPrimary`, `MCLayer` on the Ns 1.6 × 10⁵ deck and `CrossingProgram` precise on disk B
  500², one compiling and one warm call per cap.
- F4c (J9): (f) a scratch emission-only program (emit, `EMITTED`, store of the photon) at
  launch sizes 2^14, 2^16, 2^18, 2^20: primaries per second, ncu active threads per
  instruction (limited metric set), registers; (g) a no-op program whose twin is
  MC-sized (sites, tables, snapshot, accumulators) through `launch!` with a
  `CascadeSwap`-style adapt and a 24-byte counter read: host time per launch and per
  4-launch iteration, with `wait = true` and with a prototype that does not wait;
  (h) one warp of step-limit photons launched alone (thin, and Compton primaries with
  spawning): time per step; (i) 2^18 `admit!`-style appends per launch to one `Int64`
  counter, plain and with a ballot-plus-popc warp-aggregated prototype. U1's GPU bit check
  rides on J9.
- Acceptance: bins bitwise in every leg of (a), (b), (c-legs of F4b); Compton ≤ 54 s with
  (a), ≤ 50 s with (b), thin ≤ 3.3 s with (b), nsys shows both lanes' kernels concurrent;
  a cap at least 5 % faster on some phase → T3 is built, else dropped; records of (f)–(i)
  in `records/F4/`.
- Consequences: (g) above 1 ms per iteration → W8's launches that do not wait and the
  single counter copy move into W3, with W8's acceptance on them; (i) above 3 % of an
  advance (from F3b's advance time) → W10 joins W4.

**U1 — Bit-portability census and GPU bit check** (probe2; 1 d; rides on J9; depends G0)
- At `f6decd6`: libdevice calls, `fmuladd` and `contract`/`reassoc` counts per program
  (`plan/aux/census_kernels.jl`); `plan/exp/gpu_bitcheck.jl` on the A100; a CPU timing of
  `raise_and_norm` and the decode in today's form and in the explicit-`fma` form (D2's CPU
  cost).
- Acceptance: `records/U1/README.md` with, per decision D1 and D2, the decks it would
  re-baseline, the expected CPU–GPU gaps afterwards and D2's CPU cost. No package edit.

### 3.3 Phase P: the CPU profile

**P1 — CPU profile at `f6decd6`** (host; 1.5 d; 0 jobs; depends G0)
Goal: evidence for the CPU half of "speedup on both CPU and GPU", replacing the older-tree
shares (generation 21 %, primitives 20.5 % cache-miss bound, control re-evaluation 37 of
541 ns per step; `results/blp/mc_audit/speed/PLAN.md:107-115`).
- Inclusive Julia profiles, pinned (§8.4): thin and Compton photon phases on 32 threads,
  torus imaging 768² on 16, marched disk B production and precise on 32; `perf stat`
  cache-miss and cycle counters per phase; per-thread busy time and chunk spans of the
  threaded cascade; the cost of the `ExactBins` and ledger atomics (a variant that skips
  recording, a script override in its own process); allocations and GC time.
- Acceptance: `records/P1/README.md` with shares per function and phase; the CPU targets
  of C1 and C3 written into §1 and §3.7 from it; any share ≥ 10 % of a phase that no item
  covers listed for the lead.

### 3.4 Phase T: pool items, bitwise by construction

**T4 — Kernel audit repair** (pool; 1 d; 0 jobs; depends G0)
- Files: `bench/kernel_audit.jl` (the `mc_torus_*` cases dispatch through an executor that
  `grid_tables` and `hotcross_build` accept; cases for `CrossingProgram` production and
  precise, camera and host rays), `bench/kernel_audit_baseline.json` (re-taken at
  `f6decd6`), `bench/README.md`.
- Acceptance: `bench/kernel_audit.jl --check` passes on the base against its own re-taken
  baseline; unblocks G2.

**T1 — Child slots from a device budget** (pool; 1 d; J10, J11; depends G3, F4a)
- Goal: no overflow rerun on the timing deck (9.79 s of the 63.2 s Compton call). The MC
  gain stands until W4 serves the scattering phase, and permanently if it does not.
- Files: `src/infra/Execution/cascade.jl` (the slot rule of S§3; an explicit `capacity`
  verbatim for every generation of every chunk; the rerun count), `pool.jl` (the learned
  slots per device, under the pool's lock), `devices.jl` (`free_bytes(dev)`, default 0,
  which turns the budget off), `executor.jl` (`describe` reports the reruns of the last
  call), `ext/BlackLightPlusCUDAExt.jl` (`free_bytes` of a `CUDADevice`), `Execution.jl`
  (exports, docstrings), `docs/src/api/execution.md`, `test/infra/execution_cascade.jl`
  (capacities 1, 2, 5 with `reruns > 0` asserted and results unchanged; learned slots
  shared by the lanes of a device; budget 0).
- Acceptance: nsys of the warm Compton call shows 12 `MCLayer` launches (4 chunks × 3
  generations) and no rerun; Compton photon phase ≤ 0.86 × G0 on one A100 (A-B-A); thin
  kept; gate F runs `execution_cascade.jl` first in a fresh process, so a capacity leg
  cannot inherit learned slots; the threaded path is not edited (gate C, no CPU timing).

**T2 — Lane overlap on read-only arrays, `share_lanes!`** (pool; 2.5 d; J12, J13, J14;
depends T1)
- Files: `devices.jl` (`share_lanes!`, identity by default; `sync_lanes(pool, dev)`; the
  lane check switch), `ext/BlackLightPlusCUDAExt.jl` (synchronise the writer's stream,
  `CUDA.enable_synchronization!(a, false)`, explicit free after `sync_lanes`; the debug
  assertion that every lane's stream is idle, `CUDA.isdone`), `residency.jl`
  (`cached_twin` shares fresh uploads; eviction synchronises the device's lanes, then
  frees), `accumulators.jl` and `children.jl` (accumulator and `ChildBuffer` twins
  registered per lane, folded by `gather!`), `pool.jl` (program twins hold residency twins
  until `sync_lanes` at the end of a call), `executor.jl`, `docs/src/api/execution.md`,
  `test/infra/execution_pool.jl` (two lanes reading one shared array into per-lane
  accumulators equal one lane; twin counts back to their start), `test/infra/execution_cuda.jl`
  (the same on a GPU; the stress test: lane A in a long kernel reading a shared array
  while lane B zeroes its accumulators, uploads, evicts and frees, checked by exact sums
  and a canary pattern; the assertion fires when a free is attempted under a busy lane).
- Acceptance: gate R with five repetitions of every device Monte Carlo deck in one job,
  bitwise each other and `ref_gpu1`; the stress test passes in the job; `compute-sanitizer
  --tool memcheck` with `JULIA_CUDA_MEMORY_POOL=none` clean on `mc_torus_thin` and
  `torus_I`, after a check that `/usr/local/cuda-13` (13.2.1) runs CUDA.jl's runtime 13.3
  under the sanitizer (no `racecheck`: BLP kernels use no shared memory); nsys: summed
  `MCPrimary` kernel time ≥ 1.1 × their union on the thin call; thin ≤ 0.88 × G0; Compton
  ≤ 0.80 × G0 with T1 (both standing until W3/W4); disk B production ≤ 0.90 × G0 and
  precise ≤ 0.94 × G0; torus Stokes I kept; gate E in the wave job.

**T3 — Register cap** (pool; 2 d; J15, J16; depends A4 (extension file), T4, F4b pass)
- Files: `kernel.jl` (trait `register_cap(prog)`, default `nothing`; `launch!` passes it),
  `devices.jl` (`launch_capped!`, which ignores the cap by default), the extension (the
  launch through `KA.launch_config`, `KA.mkcontext` and `@cuda launch=false maxregs=r`),
  trait methods next to the `workgroupsize` methods of the adopted programs
  (`programs.jl`, `crossings.jl`), `Execution.jl` exports, `docs/src/api/execution.md`,
  `bench/kernel_audit.jl` cases at the adopted caps, `test/infra/execution_cuda.jl`
  (capped and uncapped launches bitwise; a test that fails loudly if the KernelAbstractions
  internals it calls change).
- Acceptance: G2: the PTX of capped and uncapped programs identical once the `.maxnreg`
  line is removed (GPUCompiler emits it, `ptx.jl:243-246`), the `DFMA`, `DMUL`, `DADD`
  sequences equal in order; each adopted cap ≥ 5 % faster on its phase (A-B-A), no other
  row slower; A3's legs ride on J15.

### 3.5 Phase W: the device wavefront

**W1 — March split** (mc; 2 d; J17, J18; depends G1, G2, G3, the C2 decision)
- Goal: `track!` composed of the pieces the stage programs call, with today's bits, and
  the `Sink` protocol's `branch!` with the `ScatterEvent` that W6 reuses.
- Files: `src/physics/MonteCarlo/track.jl` (`March` gains the absorption and scattering
  switches; `track!` = `march_close(m, march_steps(m, march_open(m, sph, rng),
  max_steps(ctl)))`, non-inlined; `step_events` builds the step's `ScatterEvent` and calls
  `branch!(sink, ev, crng)`), new `march_state.jl` (`MarchState`, with C2's field if C2
  landed; the `@inline` pieces), new `events.jl` (`ScatterEvent`, `kick`), `sinks.jl`
  (`branch!`, whose default runs `kick(ev)` and `spawn!` inline), `MonteCarlo.jl`
  (includes, exports with docstrings), `docs/src/api/montecarlo.md` (`MarchState`,
  `march_open`, `march_steps`, `march_close`, `ScatterEvent`, `kick`, `branch!`),
  `test/physics/montecarlo_track.jl`.
- Tests: (i) the pieces in steps of K ∈ {1, 2, 7, 32, ∞} bitwise each other on the host
  for 10⁴ photons of the thin torus deck and of the scattering sphere (one compiled loop,
  K a runtime argument); (ii) the pieces against `track!` within 1e-11, and bitwise in
  addition when `march_steps` is `@noinline`; (iii) a counting sink sees one `branch!` per
  scattering event of `track!`.
- Acceptance: gates A–D, G (GPU records bitwise; CPU record differences recorded); the
  recorder forwards `branch!` (bench); G2 screen: registers of `MCPrimary` and `MCLayer`
  equal, stack within +16 B, spill bytes reported, the multiset of `track!` equal; CPU
  thin and Compton phases kept (§8.4).
- On failure, counting only gates C and G (unit test (ii) cannot fail the item):
  `march_steps` behind `@noinline` (one compiled loop for `track!` and the device), checked
  again; if the bits still move → W1–W8 stop, D3 is raised, B1 starts; phases T, A, C
  continue.

**W2 — Device resume check and the segment premise** (mc, probe2 for the timing
analysis; 1.5 d; J19, J20; depends W1)
- A scratch program (not in the package): `foreach_item!` over a host-compacted index
  list, running `march_steps` up to K steps per launch on a `MarchState` array in a lane
  buffer, children counted into a `ChildBuffer`.
- Bits (J19), K ∈ {1, 7, 32, ∞}: thin primaries of `mc_torus_thin` (bins bitwise
  `ref_gpu1`, records bitwise `rec_ref_gpu`); Compton primaries of `mc_torus_compton`
  (their records and the generation-1 child set, stream identities and photons, bitwise
  `rec_ref_gpu`); generation-1 `MCLayer` children read from `rec_ref_gpu` (entry photon and
  stream of each), their records bitwise; G2 screen of the scratch advance body against
  `track!`'s loop.
- Timing (J20): Compton primaries at Ns 1.6 × 10⁵ and thin at stride 1, K ∈ {8, 32, 128,
  ∞}: warp instructions per warp-step, active threads per instruction against lanes
  stepping, time per warp-step (ncu with a limited metric set; nsys).
- Decision: at K = 32 the Compton primaries at ≤ 8,000 warp instructions per warp-step, or
  active threads within 1.3× of the lanes stepping → segments bound the split (regime S2);
  otherwise S3. The segment cost enters F3b.
- On a bit failure: W3–W8 stop, D3, B1 starts.

**W3 — Wavefront, thin phase** (wave + mc; 3 d; J21, J22, J23, J24 (two GPUs); depends W2,
F3b go)
- Files (wave): new `src/infra/Execution/wavefront.jl` (`Wavefront`, `run_wavefront!`,
  `foreach_wavefront!`, `with_wavefront_settings`; the host preconditions and growth, the
  per-iteration checks, the end-of-call identities and the progress guard of S§4.2; the
  shared block counter with S§4.3's refill rule and the clipped last block; per-device
  admitted and child counts; `finally` emptying every program's child slot), new
  `wave_queues.jl` (`WaveQueues`, `Cohort`, `StateBuffer`, `admit!`, `WaveSwap`, the
  counter block and its zeroing program), `devices.jl` (`wavefront(dev)`), `executor.jl`
  (`describe`: settings, per-device counts, reruns), `Execution.jl`,
  `docs/src/api/execution.md`, `test/decks.jl` (`WaveHost`), new
  `test/infra/execution_wavefront.jl`: toy programs with totals, member checksum and child
  count bitwise across `WaveHost` 1, 2, 3, 5 devices, budgets 1, 3, 64, `typemax` with
  doubling on and off, cohorts 1, 7, 2^10, refill blocks 1, 5, `n` not a multiple of R;
  equal to the cascade's, a toy spawning at the last generation included (equal count,
  children dropped); the conservation identities; a toy state that cannot advance raises
  the progress error; resident and twin counts back to their start after a call; calls
  outside `with_wavefront_settings` use the defaults; threads and `HostDevice` pools take
  the cascade (checked by the launched program types).
- Files (mc): new `src/pipeline/Products/mc_stages.jl` (`MCStart`, `MCAdvance`,
  `MCRetire`), `Products.jl`, `montecarlo.jl` (`run_layers` builds a `Wavefront` for
  stationary runs without scattering; scattering runs keep the cascade until W4),
  `docs/src/api/products.md`, `test/pipeline/products_montecarlo.jl` (thin on `WaveHost`
  pools of 1, 2, 3 devices bitwise among themselves and at K ∈ {1, 7, 32, ∞} with
  doubling off, within 1e-11 of threads; `device_isbits` and no allocation of the stage
  programs; the executor test unchanged).
- Acceptance: gates A–E, G (records at `MCRetire`), R (three repetitions); every thin
  device deck (`mc_torus_thin`, `mc_torus_s1`, `mc_sphere_thin`, `mc_binary_thin`,
  `rl_torus_thin`) at K ∈ {1, 7, 32, `typemax`} with doubling off bitwise `ref_gpu1`; G2
  screen for `MCStart`, `MCAdvance`, `MCRetire`; the stage programs' compile time (nsys
  module loads) reported against the cascade's; thin photon phase ≤ 0.53 × G0 on one A100
  and faster than the same tree's cascade (`wavefront = false`, A-B-A median) on one A100
  and on two; the warm thin call reported.
- Falsified if the thin phase is not faster than the cascade, or above 0.53 × G0 while the
  driver's log shows E ≥ 0.8 per iteration → W5's profile before W4; a thin phase slower
  than the cascade keeps the cascade (S§9).

**W4 — Children, inline kick** (wave + mc; 2.5 d; J25–J29 (J29 two GPUs); depends W3)
- Files (wave): `wavefront.jl`, `wave_queues.jl` (child queue C, the advance rerun, `cap_C`
  carried per device across iterations and calls unless `capacity` is explicit, the depth
  rule through `generation(state)` and the `dropped` count, the child count). Files (mc):
  `mc_stages.jl` (`MCBranch`), `programs.jl` (`MCSink`'s `branch!`), `montecarlo.jl`
  (stationary scattering runs on the wavefront), `docs/src/api/products.md`; tests:
  `WaveHost` scattering at capacities 1 and 3 (`reruns > 0` asserted) and `nothing`, the
  budget sweep on `WaveHost`, `products_montecarlo.jl` scattering legs. W10 joins if F4c
  says so.
- Acceptance: gates A–E, G (every generation), R (three repetitions); every device Monte
  Carlo deck (thin and scattering, coverage, ledger and step-limit decks) at K ∈ {1, 7,
  32, `typemax`} with doubling off bitwise `ref_gpu1`; capacities 1 and 3 with reruns > 0
  bitwise; a run at P = 64, K = 1, capacity 1; Compton photon phase ≤ 0.24 × G0 on one
  A100 and faster than the same tree's cascade on one A100 and on two; two-A100 ratios per
  §1; compile time reported.
- Falsified if Compton > 0.24 × G0, or active threads per instruction of `MCAdvance`
  below half the lanes stepping → regime S3; W6 decides. A scattering phase slower than
  the cascade keeps the cascade, and B1 starts.

**W5 — Stage profile** (bench2; 1 d; J30, J31; depends W4)
- ncu of `MCAdvance` (thin, Compton at Ns 1.6 × 10⁵), `MCStart`, `MCBranch`, `MCRetire`;
  nsys of one warm call of each phase; the driver's per-iteration log (nA, nB, nC, nD, K,
  launch times); `MCStart`'s rate and registers against F4c(f).
- Acceptance: `records/W5/README.md` with instructions per warp-step per class, active
  threads against lanes stepping, residency, host gaps, tail length, emission time,
  against F3b's model; the decision list for W6–W8.

**W6 — Deferred kick (gated by F1(c))** (mc + wave; 2 d; J32, J33, J34; depends W5, F1)
- Files: `programs.jl` (`branch!` on a `ScatterEvent` child buffer), `mc_stages.jl` or
  `mc_kick.jl` (`MCKick`, which calls W1's `kick`), `wavefront.jl` (the element type of C
  per `Wavefront`, wave), `docs/src/api/products.md`, tests (on the host `MCKick`'s child
  equals `branch!`'s inline child), a kernel-audit case. `track.jl` and `events.jl` are
  not edited: the threaded body is W1's.
- Acceptance: gates C (unchanged threaded bits), D, G, R; the scattering decks at K ∈ {1,
  7, 32, ∞} with doubling off bitwise `ref_gpu1` (else stop: the inline kick stays and the
  result is recorded); G2: spill bytes of `MCAdvance` with scattering below W4's; Compton
  ≤ 0.16 × G0 (10 s).

**W7 — Coherence sort (gated)** (wave + mc; 2 d; J35, J36; depends W4, K1, F2's key
measurement)
- The key is the one of F2's ncu leg with fewer sectors per request (S§4.5).
- Files: new `wave_sort.jl` (`KeyCount`, `KeyPlace`, the host scan, `sort_every`),
  `mc_stages.jl` (`MCKey` over K1's `photon_key`), `docs/src/api/*.md`, tests (the order is
  a permutation with non-decreasing keys on every executor; sort on and off bitwise on
  `WaveHost`).
- Acceptance: device decks bitwise at `sort_every` ∈ {1, 4}; ncu sectors per request and
  L1 hit of `MCAdvance` on layer-dominated cohorts; the sort becomes the default only if
  the scattering phase is ≥ 3 % faster; otherwise W7's code is removed (`photon_key`
  stays, K1).

**W8 — Launches and tuning** (wave; 2 d; J37–J41 (J41 two GPUs); depends W5; W6, W7 when
they land; parts move into W3 if F4c(g) says so)
- Launches that do not wait, under the rules of S§4.2 (counters zeroed on the device, two
  pinned counter stages alternating, ranges and settings as kernel arguments, an overflow
  rerun zeroing on the stream), with `launch!(...; wait = false)` in `kernel.jl`; one
  combined counter copy; refill blocks in predicted-cost order (inner zones first, from the
  zone radii).
- Sweeps one factor at a time from the defaults on the stride-1 timing deck: K ∈ {16, 32,
  64, 128}, P ∈ {2^16, 2^17, 2^18, 2^19}, R ∈ {2^12, 2^14, 2^16}, workgroup ∈ {32, 64,
  128, 256}; every point three times, its bins, ledger and child count bitwise G0's
  stride-1 record (J2).
- Files: `wavefront.jl`, `wave_queues.jl`, `kernel.jl`, `test/infra/execution_wavefront.jl`
  and `execution_cuda.jl` (a launch that does not wait followed by the counter read equals
  the waiting form; the stage alternation), `docs/src/api/execution.md`.
- Acceptance: every point bitwise in its three repetitions; the run at P = 64, K = 1,
  capacity 1 (an overflow every iteration) bitwise; nsys host gaps ≤ 3 % of the photon
  phase; defaults set to the best median; thin ≤ 0.43 × G0, Compton ≤ 0.19 × G0 (inline)
  or ≤ 0.15 × G0 (deferred) on one A100; two-A100 ratios per §1.

**W10 — Warp-aggregated appends (contingent: F4c(i) or W5)** (pool; 1.5 d; 1 job; depends
W3)
- `admit!`'s slot claim through an extension method (ballot and popc per warp, one atomic
  per warp), the plain atomic by default; tests on `WaveHost` and CUDA (order-free, so
  bitwise). Acceptance: the advance ≥ 3 % faster; decks bitwise.

### 3.6 Phase K, B: the key and the fallback

**K1 — `photon_key`** (mc; 0.5 d; 0 jobs; depends G0)
- Files: new `src/pipeline/Products/mc_order.jl` (`photon_key(st, ctl, y)`, the total
  function of S§4.5), `Products.jl`, `docs/src/api/products.md`,
  `test/pipeline/products_montecarlo.jl` (r = 0, r ≤ r_min, NaN, Inf, Minkowski, Kerr, a
  static binary; keys in `0:8191`; the same keys on threads and `WaveHost`).
- Acceptance: gates A, B. Consumers: C1, B1, W7; independent of W7's fate.

**B1 — Ordered device cascade (contingent fallback)** (pool + mc; 3 d; J42, J43, J44;
depends T2, K1; starts on a W1 or W2 bit failure, an F3b no-go, or a phase that keeps the
cascade)
- Files (pool): `cascade.jl` (the trait `child_order(prog)`, default `nothing`; a keyed
  generation launches over its children permuted by a counting sort on the lane, the
  overflow rerun over the same permutation; `item_order(prog)` for the primaries),
  `Execution.jl`, `docs/src/api/execution.md`, `test/infra/execution_cascade.jl` (keys on
  and off, pools of 1, 2, 3, 5 host devices, chunks 37 and 211, capacities 1 and 2 with
  reruns > 0, bitwise threads). Files (mc): the `MCLayer` and `MCPrimary` methods over
  K1, `products_montecarlo.jl`.
- Acceptance: gates A–F, R (three repetitions); Compton ≤ 0.74 × G0 with T1 and T2 on one
  A100 (`design_incremental.md` §6.3: 40.4–45.8 s with keyed children, 32.5–46.4 s with
  keyed primaries); thin kept.

### 3.7 Phase A and C: auxiliary and CPU items

**A1 — Threaded host epilogues (gated)** (host; 1.5 d; J45, gate F in the wave job;
depends G3)
- Files: new `src/pipeline/Products/epilogues.jl` (`BandSum`, `Fill`, `Split`, run on
  `ThreadedExecutor()` over host arrays), `analytic.jl`, `analytic_surfaces.jl`,
  `crossings.jl`, `image.jl`, `analytic_movie.jl` (frames written in place, the spot maps
  kept for the frame program), `Products.jl`, `docs/src/api/products.md`, tests in
  `products_analytic.jl`, `products_crossings.jl`, `products_surfaces.jl`,
  `products_movie.jl`, `products_render.jl` (bitwise the broadcast forms).
  `photon_ring_layers`' weighted composite and every order-fixing reduction stay.
- Acceptance: gates A–D; gate F's pipeline files; `cf_radius` on the A100 ≤ 0.72 × G0 and
  on 32 CPU threads ≤ 0.87 × G0, GC time per call reported (`@timed`); torus imaging and
  crossings kept on both backends.

**A2 — Caller storage `into` (optional)** (host; 2 d; J46; depends A1, T2)
- Files: the five image products (`render_analytic` both forms, `render_crossings`,
  `render`, `analytic_movie`), `residency.jl` (eviction of a host array given as an output),
  `docs/src/api/products.md`, tests.
- Acceptance: results with `into` bitwise the default on `:cpu`, `:host_pool`,
  `Decks.hosts(3)` and the A100; no allocation above 1 MiB per call (`@allocated`); a host
  output reused after an upload gives fresh values; `cf_radius` with `into` ≤ 0.26 × G0 on
  the A100 and ≤ 0.73 × G0 on 32 CPU threads.

**A3 — Crossing program legs (measurement)** (march; 0.5 d; rides on J15; depends T3)
- `workgroupsize(::CrossingProgram)` at 32, 64, 128, 256 and the caps of T3, on disk B
  500² production and precise. Expected headroom of the workgroup at most 2.5 %
  (queue-order lane efficiency 0.954 for blocks of 256 against 0.979 for warps, profile
  §3); offline `ptxas` shows caps multiplying spills.
- Acceptance: legs reported; a production setting adopted only at ≥ 5 % (the trait method
  in `crossings.jl`, gate F's `products_crossings.jl`); precise legs reported without a
  target. The threaded path is not touched.

**A4 — Compile latency (gated)** (march; 2.5 d; J47, J48; depends A1 (epilogue program
types), T2 (extension file))
- Confirm the invalidation by ForwardDiff (`SnoopCompileCore` if it is in the depot, else
  the load-order bisection of `plan/aux/warmup2.jl`); the extension's precompile workload
  (a trivial program, `ZoneSums`, `ZoneCounts`, `HotcrossSigma`, `Drain`, the epilogue
  programs) for sm_80; the `[GPUCompiler] disk_cache = "true"` preference documented in
  the GPU environment and in the docstring of the workload, with a cache-hit test; a
  `PrecompileTools` dependency only if needed, with every env copy re-resolved and the
  Manifest diff declared (§2).
- Acceptance, each part with and without on the same tree, fresh processes in one job:
  the first device compilation, the host time before the first kernel of a cold thin
  call, and the nsys module-load intervals per program, each shorter with the part; cached
  cubins byte-identical to fresh compilations; gates C and C′ bitwise (the workload and
  the preference change the package images a session loads); gate D; the extension's
  precompilation time recorded.

**A5 — Compile ahead (contingent)** (march; 1.5 d; 1 job; depends W4 or B1)
- Built only while a device cascade serves a phase (W3 or W4 kept the cascade, or B1
  adopted). `prepare!(lane, prog)` in the device protocol (compile without launching;
  CUDA method in the extension), the cascade compiling its next programs on a host task;
  `devices.jl`, the extension, `cascade.jl`, `Execution.jl`, `docs/src/api/execution.md`,
  `test/infra/execution_cuda.jl` (a prepared program launches without compiling, bitwise).
- Acceptance: that phase's cold call shorter with than without (same tree, fresh
  processes); decks bitwise; nsys module-load intervals overlapping the first launches.

**C2 — The step control reuses the sample's metric** (host; 1.5 d; J49 and CPU timing;
depends G0, G3; decided before W1's gate job)
- The control reads the inverse metric and Kerr–Schild radius that the step's last
  `sample` evaluated, instead of evaluating them again (37 of 541 ns per step on one thread
  at an older tree, `results/blp/mc_audit/speed/PLAN.md:266-269`); vacuum samples compute
  the pair too. Files: `track.jl`, the sample's return in `src/physics/MonteCarlo`,
  `test/physics/montecarlo_track.jl`.
- Acceptance: gates A–D, G (class B on both backends); thin and Compton phases on 32
  threads faster beyond δ (§8.4) and kept on the A100. If it lands, the carried pair (the
  4 × 4 inverse metric and the radius, 136 B) is a `MarchState` field from W1 on (state
  344 → 480 B; S§5 table and buffer sizes updated in W1), W2's and W3's budget sweeps and
  G2's registers and spills of `MCAdvance` cover it. If it has not passed by W1's gate job,
  it is dropped from this plan.

**C1 — Child order on threads** (host; 1 d; 0 jobs; depends K1, T1 (cascade.jl), P1)
- First a host measurement: the Compton phase on 32 threads with each generation's
  children permuted by `photon_key` (script override), against arrival order, with
  `perf stat` cache misses per step. Build only if faster beyond δ.
- Files: `cascade.jl` (the threaded `chunk_generations!` permutes each generation's
  children by `child_order` before the layer runs; the trait is B1's, introduced here if
  C1 lands first), the `MCLayer` method over K1, tests (keys on and off bitwise on
  threads).
- Acceptance: gates A–C; the Compton phase on 32 threads faster beyond δ, by at least the
  target P1 sets; else not landed.

**C3 — Generation on threads (from P1)** (host; 1.5 d; 0 jobs; depends P1)
- If P1 shows emission at ≥ 15 % of the thin phase on 32 threads (21 % at an older tree):
  the zone-constant factors of the frequency proposal (4.2–4.6 proposals per primary)
  computed once per primary outside the rejection loop, operation for operation (class B).
  Files: the emission functions of `src/physics/MonteCarlo` and `src/pipeline/Products`
  that P1 names, after W4 merges (`track.jl` free).
- Acceptance: gates A–D, G; thin phase on 32 threads ≤ 0.95 × G0 and kept on the A100;
  else not landed.

### 3.8 Phase V, L: verification, documents, landing

**V1 — Final verification and record** (bench, bench2; 2 d; J50–J54; depends every
adopted item)
- Gates A–I, C′, R on the `gpu-wavefront` tip; every row of §1 on one A100, two A100s and
  32 CPU threads by §8.3–8.4; the cumulative CPU ratio of every CPU row against G0 (a
  median ratio above `1 + δ` is a regression to fix or report); `records/V1/README.md`,
  summarised in `results/blp/gpu_infra/RECORD.md` of the main checkout (text only, in a
  stage that allows writes under `results/` there).

**V2 — Documents** (docs; 1 d; 0 jobs; depends V1)
- `docs/src/design/gpu_execution.md` to the adopted state (gated components that failed
  their gate removed, with their reason in S§10), `performance.md` §1, §2, §6,
  `architecture.md` §4.2, `guide/gpu.md`, `CHANGELOG.md`.
- Acceptance: gates A and B; no narrative prose; every number traced to a record.

**L1 — Landing** (lead + bench; 1.5 d; J55–J59; depends V2, the user's word)
- Rebase `gpu-wavefront` onto main's tip; move `ref_main` to that tip and re-record the
  references by G0's procedure (CPU decks, `ref_gpu1`, `ref_gpu2`, stride-1 bins,
  records); rerun gates A–I, C′, R of the rebased tip against them; prepare the push for
  the user's approval (nothing is pushed without it); sync the record to the `blp` branch
  through a worktree, as the repository's branch convention requires.

### 3.9 Dependency graph

| item | depends on | unblocks |
|---|---|---|
| G0 | — | everything compared bitwise |
| G1 | G0 | F1, F2, F3b, W1, the G gate |
| T4 | G0 | G2, T3 |
| G2 | T4 | F1, W1, the H screen |
| G3 | G0 | F2, F4, T1, A1, W1, C2 |
| F1 | G1, G2 | the W6 decision |
| F2 | G1, G3 | f for F3b; W7's key |
| F3a / F3b | — / F2, F4c, W2, G1 | — / the go of W3–W8 |
| F4 | G0, G3 | T1, T2, T3 (go), F3b, the W3/W10 scope |
| U1 | G0 (rides on J9) | the evidence of D1, D2 |
| P1 | G0 | C1, C3 targets |
| T1 | G3, F4a | T2, C1 |
| T2 | T1 | A2, A4, B1 |
| A4 | A1, T2 | T3 |
| T3 | A4, T4, F4b pass | A3 |
| C2 | G0, G3 | W1 (the `MarchState` layout) |
| W1 | G1, G2, G3, C2 decided | W2 |
| W2 | W1 | F3b, W3 |
| W3 | W2, F3b go | W4, W10 |
| W4 | W3 | W5, W7, A5 |
| W5 | W4 | W6, W8 |
| W6 | W5, F1 (kick attributed) | W8 |
| W7 | W4, K1, F2 | W8 |
| W8 | W5; W6, W7 when they land | V1 |
| K1 | G0 | C1, B1, W7 |
| B1 | T2, K1, a falsification | A5, V1 |
| A1 | G3 | A2, A4 |
| A2 | A1, T2 | V1 (optional) |
| A3 | T3 | V1 |
| A5 | W4 or B1, a device cascade serving a phase | V1 |
| C1 | K1, T1, P1 | V1 |
| C3 | P1, W4 merged | V1 |
| V1 | every adopted item | V2 |
| V2 | V1 | L1 |
| L1 | V2, the user's word | — |

Effort: about 57 agent-days (G 4.5, F and U 7, P 1.5, T 6.5, W 17.5, K and B 3.5, A 8,
C 4, V and L 4.5), B1, W10 and A5 included.

## 4. Order of execution and parallel work

### 4.1 Waves

The node runs one job at a time (four CPUs per job and in total, profile §1), shared with
another agent; the calendar counts GPU slots, at about five jobs a day and half a day of
queue and analysis latency per sequential GPU gate.

| wave (working days) | bench / bench2 | probe / probe2 | pool | mc | wave | host | march |
|---|---|---|---|---|---|---|---|
| 0 (d0–2) | G0 (J1–J3), G3 | F3a / U1 prep | T4 | W1 draft, K1 | toy driver on `WaveHost`, no package wiring | P1, C2 code, A1 code | A4 invalidation study (offline) |
| 1 (d2–5) | G1 (J4); F4 (J7–J9) | G2 (after T4), F1 (J5) / F2 (J6) | T1 (J10, J11) | W1 (J17, J18), W2 (J19, J20) | queues and `WaveSwap` on `WaveHost` | C2 (J49, CPU timing), A1 (J45) | A4 workload code |
| 2 (d5–7) | wave job after T1, T2, A1, C2, W1 (J60) | F3b (go) / W2 analysis | T2 (J12–J14) | W3 Products half | W3 Execution half | C1 measurement, A2 code | A4 (J47, J48) after T2 |
| 3 (d7–11) | W3 jobs (J21–J24) | — | T3 (J15, J16) if F4b passed | W3, then W4 programs | W3, then W4 driver | A2 (J46) | A3 legs on J15 |
| 4 (d11–14) | W4 jobs (J25–J29); W5 (J30, J31) | — | W10 if needed | W4, W7 Products half | W4, W7 | C1 | — |
| 5 (d14–19) | W6 (J32–J34), W7 (J35, J36), W8 (J37–J41) | — | — | W6 if F1 | W6 element type, W8 | C3 after W4 merges | A5 if a cascade phase remains |
| 6 (d19–22) | V1 (J50–J54) | — | — | — | — | — | docs agent: V2 |
| 7 (d22–24) | L1 (J55–J59) with the lead | — | — | — | — | — | — |

On a falsification (W1 or W2 bits, F3b no-go, a phase kept on the cascade) B1 takes the
W slot of waves 3–4 with J42–J44, and A5 follows it.

Critical path: G0 (J1–J3) → G1 (J4) → W1 (J17, J18) → W2 (J19, J20) → F3b → W3 (J21–J24) →
W4 (J25–J29) → W5 (J30, J31) → W6 (J32–J34) → W8 (J37–J41) → V1 (J50–J54) → V2 → L1
(J55–J59): 30 sequential GPU jobs, about 24 working days. F1, F2 and F4 lie beside it but
feed F3b, so they run before every non-critical job.

### 4.2 File ownership

One owner at a time per file; later owners rebase onto the merged earlier item. Additive
export lines in `Execution.jl`/`Products.jl` and `@docs` entries in `docs/src/api/*.md`
are resolved by the lead at merge.

| file | items, in merge order |
|---|---|
| `src/infra/Execution/cascade.jl` | T1 → C1 or B1 (whichever first introduces `child_order`) → the other |
| `src/infra/Execution/devices.jl` | T1 → T2 → W3 → T3 → A5 |
| `src/infra/Execution/pool.jl`, `executor.jl` | T1 → T2 → W3 |
| `src/infra/Execution/residency.jl` | T2 → A2 |
| `src/infra/Execution/accumulators.jl`, `children.jl` | T2 |
| `src/infra/Execution/kernel.jl` | T3 → W8 |
| `src/infra/Execution/wavefront.jl`, `wave_queues.jl`, `wave_sort.jl` (new) | wave only: W3 → W4 (→ W10 via pool, on wave's branch) → W6 → W7 → W8 |
| `ext/BlackLightPlusCUDAExt.jl` | T1 → T2 → A4 → T3 → W10 → A5 |
| `src/physics/MonteCarlo/track.jl`, `sinks.jl`, `MonteCarlo.jl`, `march_state.jl`, `events.jl` | C2 → W1 → C3 |
| `src/pipeline/Products/mc_stages.jl`, `mc_kick.jl`, `mc_order.jl` (new) | mc only: K1 → W3 → W4 → W6 → W7 |
| `src/pipeline/Products/montecarlo.jl` | W3 → W4 → B1 |
| `src/pipeline/Products/programs.jl` | W4 → W6 → T3 (trait methods) → B1 |
| `src/pipeline/Products/analytic.jl`, `analytic_surfaces.jl`, `image.jl`, `analytic_movie.jl`, `epilogues.jl` | A1 → A2 |
| `src/pipeline/Products/crossings.jl` | A1 → A2 → T3 (trait method) → A3 |
| `test/decks.jl` | W3 |
| `test/infra/execution_cascade.jl` | T1 → C1 / B1 |
| `test/infra/execution_pool.jl`, `execution_cuda.jl` | T2 → W3 → T3 → W4 → W8 |
| `test/infra/execution_wavefront.jl` (new) | W3 → W4 → W7 → W8 |
| `test/physics/montecarlo_track.jl` | C2 → W1 → C3 |
| `test/pipeline/products_montecarlo.jl` | K1 → W3 → W4 → W6 → B1 |
| `bench/kernel_audit.jl`, `kernel_audit_baseline.json` | T4 → T3 → W5/W6 cases → A3 |
| `Project.toml` | A4 (a precompile dependency, if needed) |
| `docs/src/design/*.md`, `guide/gpu.md`, `CHANGELOG.md` | V2 |

## 5. GPU job budget and queue plan

Node `anta`, partition `preempt`: 2 × A100 80 GB, 4 CPUs per job and in total, so one job
at a time; at most 1 h per job; preemptible without requeue; another agent submits too.
Job content is capped at about 45 min. Gate F (about 25 min) always has its own job and
resumes from its done-markers after a preemption.

| job | item | GPUs | content | wall |
|---|---|--:|---|--:|
| J1 | G0 | 1 | reference device decks on `gpu:0`; imaging legs; fingerprint; clocks | 35 min |
| J2 | G0 | 1 | stride-1 thin and Compton phases (bins, ledger, child count, three warm calls each); short nsys of each launch kind | 40 min |
| J3 | G0 | 2 | device decks on `gpu:0` and `gpu:0,1` in one process; stride-1 phases on two A100s | 45 min |
| J4 | G1 | 1 | records of three stride-8 decks; probe of the stride-1 deck at `f6decd6`; step counts at Ns 1.6 × 10⁵ | 45 min |
| J5 | F1 | 1 | ncu (a), (b), (c), each its own process | 45 min |
| J6 | F2 | 1 | ordered launches (four orders), ncu memory leg | 45 min |
| J7 | F4a | 1 | capacity and overlap legs, nsys | 40 min |
| J8 | F4b | 1 | cap legs, one warm call per cap | 45 min |
| J9 | F4c + U1 | 1 | emission at four sizes, launch overhead, lone warps, appends; GPU bit check | 45 min |
| J10, J11 | T1 | 1 | gate D, Compton A-B-A with nsys; gate F | 45 + 30 min |
| J12–J14 | T2 | 1 | gate R ×5, stress, memcheck; A-B-A timing with nsys; gate F | 45 + 40 + 30 min |
| J15, J16 | T3 (if F4b) + A3 | 1 | gate D, timing, A3 legs; gate F | 45 + 30 min |
| J17, J18 | W1 | 1 | gates D, G; gate F | 35 + 30 min |
| J19, J20 | W2 | 1 | bits at four budgets with records; timed and ncu leg | 35 + 45 min |
| J21–J24 | W3 | 1, 1, 1, 2 | gates D, G, R(3), budget sweep; A-B-A against the cascade, module loads; gate F; gate E and two-GPU timing | 45 + 35 + 30 + 45 min |
| J25–J29 | W4 | 1, 1, 1, 1, 2 | budget sweep on every device MC deck; gates D, G, R(3), capacities, stress run; A-B-A against the cascade; gate F; gate E and two-GPU timing | 45 + 40 + 35 + 30 + 45 min |
| J30, J31 | W5 | 1 | ncu of the stage programs; nsys | 2 × 45 min |
| J32–J34 | W6 | 1 | gates D, G, R, budget sweep; ncu and timing; gate F | 45 + 40 + 30 min |
| J35, J36 | W7 | 1 | sort legs bitwise, ncu; timing | 40 + 35 min |
| J37–J41 | W8 | 1, 1, 1, 1, 2 | sweeps ×3 (two jobs); stress run, gates D, R; gate F; gate E and two-GPU timing | 45 + 45 + 40 + 30 + 45 min |
| J42–J44 | B1 (contingent) | 1 | gates D, R; timing; gate F | 45 + 40 + 30 min |
| J45, J46 | A1, A2 | 1 | gates D, timing each | 2 × 40 min |
| J47, J48 | A4 | 1 | compile measurements with and without in fresh processes, gate D; gate F | 45 + 30 min |
| J49 | C2 | 1 | gates D, G; A100 thin and Compton A-B-A | 45 min |
| J50–J54 | V1 | 1, 1, 2, 1, 1 | gates D, G, R; gate F; gate E and two-GPU timing; one-A100 timing of the MC rows; imaging and cold rows | 5 × 45 min |
| J55–J59 | L1 | 1, 2, 1, 1, 2 | re-record at main's tip (one, two A100s); gates D, G, R on the rebased tip; gate F; gate E | 5 × 45 min |
| J60–J62 | wave jobs (after waves 2, 4, 5) | 2 | tip gates D (on `gpu:0`), E, F of the items merged in the wave | 3 × 45 min |
| — | contingent W10, A5 | 1 | gates D, timing | 2 × 40 min |

64 jobs and about 43 job-hours with every contingent item, 59 without them; with a
quarter of reruns for preemptions and failures, 74–80 jobs, 15–16 node-days at five jobs
a day.

Queue rules:
- Submission only through `gates/submit.sh`, by bench, or by bench2 when bench is
  unavailable; other agents hand bench an sbatch file and its `run_job.sh` pattern line.
  At most two team jobs queued or running at a time.
- Priority: critical-path jobs first; then F1, F2, F4 and U1 (they feed F3b); then T, B,
  A, C; a lower-priority job is not submitted while a higher-priority one is ready.
- Job names `wf-<item>-<n>`; logs `/data2/haiyangw/scratch/wavefront/jobs/<item>/%x_%j.log`;
  waits with `squeue -h -j <id>`.
- A job runs from the worktree under test with its own precompiled environment; the header
  must show `dirty 0`, the fingerprint and a clean `envcheck.sh`, or the job does not count
  for a gate.
- Every timing comparison runs inside one job, reference and branch interleaved, so the
  clock cancels in the ratio.
- A preempted or failed job is resubmitted whole; partial logs are never merged (gate F
  resumes by its markers, which is not a merge of logs).

## 6. Risk register

| # | risk | early test (when) | response |
|---|---|---|---|
| R1 | composing `track!` from the pieces moves CPU bits | W1 gates C and G, window and coverage decks (day 3–4) | `@noinline march_steps`; if still moved, D3 and B1 |
| R2 | the resumed march moves device bits, or they depend on K | W2 at K ∈ {1, 7, 32, ∞} with records of primaries and layer children (day 4); budget sweeps with doubling off in W3, W4 | locate by records; a carried value fused across the back edge is a re-baseline: D3; B1 |
| R3 | segments do not bound the split (regime S3) | W2 timed leg (day 5), F1 | deferred kick (W6); F3b may stop W3; B1 |
| R4 | per-warp-step cost rises at 32 live lanes (factor f up to 1.3) | F2 (day 3–4), W5 | the sort (W7); the cohort size; F3b uses the measured f |
| R5 | emission or host time per launch sets a floor above the targets | F4c (day 3–4), F3b | launches that do not wait in W3; register cap for `MCStart`; refill order |
| R6 | the tail is longer than modelled | F4c(h) lone-warp latency, F3b, W5 nsys | budget doubling; inner zones first |
| R7 | atomic contention on the queue counters | F4c(i), W5 ncu | W10 |
| R8 | a shared or asynchronously used buffer is overwritten or freed under a running kernel | T2 stress test, gate R ×5; W8 stress run at capacity 1, gate R ×3 | sharing only read-only arrays; explicit frees after `sync_lanes`; alternating stages; drop the array from the share list |
| R9 | the wavefront loses or duplicates photons silently on a deck no gate covers | always-on checks and identities of S§4.2; toy tests | an error, never a silent loss |
| R10 | a state cannot advance and the job runs into the 1 h limit | the progress guard (error after `M + 2` iterations without a retirement) | an error with the state's index |
| R11 | the two-GPU gate never exercises a second device | gate E's small settings, the refill rule of S§4.3, the per-device admitted counts | the gate fails unless every device admitted primaries |
| R12 | a comparison script passes on missing data | strict scripts with manifest counts and exit statuses (G0) | the gate fails |
| R13 | main moves during the work | the lead checks `git diff --stat f6decd6..main -- src ext test` before each merge; D6 | rebase at a wave boundary; when `src/` changed, re-record G0 and rerun the open items' gates; L1 re-records at the tip |
| R14 | the depot or package images change host bits with no code change | fingerprints in every deck file and job; gate C runs the reference in the same session; `ref_now` = `ref_t16a` check | a fingerprint mismatch voids the comparison; a reference drift is a depot event for the lead, not an item failure |
| R15 | an item touching the extension, `Project.toml` or preferences moves CPU bits only with CUDA loaded | gate C′ | revert the change or declare it to D1/D2 |
| R16 | the photon recorder's wrapper types hide a production-only drift | records locate, decks decide | the decks are the gate |
| R17 | host load spoils CPU timings (load average 25–143 seen) | pinned cores, paired protocol, δ from G0 | repeat; report the interval |
| R18 | the single GPU slot or preemption stalls the critical path | the queue log; priority rule | 45-min jobs; gate F alone and resumable; bench2 submits when bench cannot |
| R19 | an item slows the CPU | §8.4 on every item touching a CPU path; V1's cumulative ratio | `@noinline` variant or revert |
| R20 | the A100 runs at its 1,065 MHz base clock | G0 clock query | the record states the clock; ratios from one job; D5 |
| R21 | a capacity test stops testing overflow once slots are learned | `reruns > 0` asserted; explicit capacity verbatim; a fresh process first in gate F | the test fails |
| R22 | the explicit-capacity, learned and carried slot rules diverge between cascade and wavefront | toy tests at capacities 1, 3 on both schedules | one rule (S§3, S§4.2) |
| R23 | file budget or banned phrases fail gate A late | the split files from W3 on; the phrase list of §2 | split before merge |
| R24 | precompilation inside a GPU job | `precompile.sh` before every submission; 40 cache files | resubmit after precompiling |

## 7. Early falsification calendar

| day | test | falsified if | consequence |
|---|---|---|---|
| 1 | F3a | — (informational, stand-in inputs) | flags the inputs F4c must measure first |
| 3–4 | F4 | Compton > 54 s with presized slots; no lane overlap on read-only sharing; (g) above 1 ms per iteration; (i) above 3 % of an advance | T1/T2 targets revised; W8's launches into W3; W10 into W4 |
| 3 | F1 (c) | ≥ 12,000 warp instructions per warp-step without the sampler | W6 dropped; Compton target ≤ 0.19–0.24 × G0 |
| 3–4 | F2 | f > 1.3 | F3b with f; W7 more likely |
| 4 | W1 | gate C or G not bitwise, also with `@noinline march_steps` | D3; B1; T, A, C continue |
| 4–5 | W2 bits | device bits differ for some K | D3; B1 |
| 5 | W2 timing | segments do not bound the split at K = 32 (S3) | W6 decides the Compton target; F3b with S3 |
| 5 | F3b | predicted thin > 1.8 s or Compton > 15 s at every (K, P, R) | W3–W8 stop; B1; T1, T2 carry the device gain |
| 10 | W3 | thin not faster than the cascade, or > 0.53 × G0 with E ≥ 0.8 | W5 before W4; thin keeps the cascade if slower |
| 13 | W4 | Compton > 0.24 × G0, or active threads < half the lanes stepping; or slower than the cascade | S3: W6 decides; slower: scattering keeps the cascade, B1 |

## 8. Verification gates

### 8.1 The gates

| gate | what | command (from the worktree under test) | pass |
|---|---|---|---|
| A | structure and full CPU suite | `JULIA_PKG_OFFLINE=true julia -t 4 --project=<envtest copy> <wt>/test/runtests.jl` | exit 0; `Test Summary` with 0 fail, 0 error |
| B | strict docs build | `JULIA_PKG_OFFLINE=true julia --project=<env_docs copy> docs/make.jl` (`warnonly = false`, `checkdocs = :exports`) | exit 0 |
| C | CPU decks, A-B | `gates/gate_c.sh <wt> <env> <label>`: the reference's and the item's 25 CPU decks at `-t 16`, pinned, in one session window, then the item's at `-t 1`; `gates/compare_decks.jl` | exit 0 for item `t16` = reference `t16` of the session, item `t1` = item `t16`; reference of the session = `ref_t16a` (a failure there voids the gate and goes to the lead, R14) |
| C′ | CPU decks with CUDA loaded | `gates/decks.jl <label>_cu cpu --with-cuda` (CUDA loaded, backend `:cpu`), against the session's reference run the same way | exit 0; required for A4 and any item touching the extension, `Project.toml` or `LocalPreferences.toml` |
| D | A100 decks | in `gates/gpu_gates.sbatch`: `gates/decks.jl <label> gpu:0` (20 decks); compare with `ref_gpu1`; `gates/gap_check.jl` with the item's `t16` CPU file | exit 0 for both |
| E | two A100s | in `gates/gpu2_gates.sbatch`, one process: the device MC decks on `gpu:0` and on `gpu:0,1`, then on `gpu:0,1` inside `with_wavefront_settings(cohort = 2^12, refill = 64)`; `describe` reports two devices (explicit ordinals bypass the 15 % utilisation filter of `select_devices`, `ext/BlackLightPlusCUDAExt.jl:81`); the stride-1 phases on two devices with bins, ledger sums and counts and the child count | exit 0 for every pair against `ref_gpu1` and `ref_gpu2`; every device admitted > 0 primaries in every MC deck, and spawned children in every scattering deck that spawns at least 100 on one device (from `describe`); stride-1 two-device bins, ledger and count = one-device |
| F | GPU unit tests | `gates/gpuF.sbatch`: `BLP_TEST_ROOT=<wt> BLP_TEST_DEVICES=cuda` copy of `unit_tests_main.jl structure infra pipeline physics` | 0 failing files |
| G | photon records | `gates/recorder/record.jl` on the A100 (and the CPU for localisation); `compare_records.jl` against `rec_ref_*` | exit 0 on the A100 |
| H | device-code screen (diagnostic) | `gates/devcode_gate.jl <env> <label> --against ref` | reported: registers, stacks, spills within the item's stated change; no floating-point operation in a kernel entry; counts of unrounded operations; sequences for capped builds |
| I | timing | §8.3 and §8.4 | the item's targets; every §1 row kept |
| R | repetition | each device MC deck run n times in one job (n = 5 for T2; 3 for W3, W4, W6, W8, B1) | all runs bitwise each other and the reference |

### 8.2 Which gates an item passes

| item class | A | B | C | C′ | D | E | F | G | H | I | R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| experiments (F2, F4, U1) | — | — | — | — | bins, ledger and count bitwise per leg | — | — | — | — | report | — |
| physics-changing legs (F1 (b), (c)) | — | — | — | — | exempt | — | — | — | — | report | — |
| pool items (T1–T4) | ✓ | ✓ | ✓ | T1–T3 (extension) | ✓ | wave job | ✓ | — | T3 | ✓ | T2 |
| march split, wavefront, fallback (W1–W8, W10, B1) | ✓ | ✓ | ✓ | — | ✓ | W3, W4, W8 own; others wave job | ✓ | ✓ | ✓ | ✓ | W3, W4, W6, W8, B1 |
| key (K1) | ✓ | ✓ | — | — | — | — | — | — | — | — | — |
| host outputs (A1, A2) | ✓ | ✓ | ✓ | — | ✓ | wave job | pipeline files (wave job) | — | — | CPU and GPU | — |
| crossing and compile items (A3–A5) | ✓ | ✓ | ✓ | A4, A5 | ✓ | — | ✓ | — | A3 | ✓ | — |
| CPU candidates (C1–C3) | ✓ | ✓ | ✓ | — | C2, C3 | wave job | wave job | C2, C3 | C2, C3 | CPU (and GPU for C2, C3) | — |
| final, landing (V1, L1) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ every row (V1) | ✓ |

### 8.3 GPU timing protocol

One Slurm job per comparison (`-p preempt --gres=gpu:1 -c 4 -t 01:00:00`, `julia -t 4`,
`JULIA_PKG_OFFLINE=true`, CUDA runtime 13.3), the job alone on the node. Three processes
in the order reference, branch, reference (or, for a wavefront item, the branch's cascade,
the branch's wavefront, the branch's cascade, through `with_wavefront_settings(wavefront =
false)`), each: load the snapshot, build the sites and tables, one compiling call of each
phase, then three warm calls of the thin photon phase (stride 1, Ns 10⁶, seed
`0x5eed0700`) and three of the Compton phase (bias 10⁴, `max_scatt = 3`), timed around
`run_layers` as `gates/mc_phases.jl` does; imaging legs through `gates/paths2.jl` best of
five. Report every call, the best and the median; the bins, ledger and child count of every
call equal the reference's; the SM clock at the start and the end. Two GPUs: the same in
`--gres=gpu:2`. Pass: the item's target on the median ratio; "kept" is a median ratio at
most `1 + δ_GPU`, with δ_GPU the spread of the two reference processes of the job (at
least 0.01).

### 8.4 CPU timing protocol

32 threads pinned with `taskset` to 32 physical cores of one NUMA node (node 0: cores
0–35, hyperthreads 72–107; node 1: 36–71, 108–143), the other trees and agents told to
keep off those cores for the run; at least five interleaved reference–branch pairs; per
run the wall time, the per-thread CPU time and the load average before and after. The
statistic is the median of the paired ratios branch / reference with its 90 % interval
(order statistics of the pairs); δ, the noise half-width, is that interval's for the
reference against itself, measured in G0 per row and stated as the minimum detectable
effect. "Kept" is a median ratio ≤ `1 + δ`; "faster by x" is an interval wholly below
`1 − x`; a speed-up target smaller than δ cannot be decided and is reported as such. A
pair during which the load average moves by more than 30 is repeated.

### 8.5 Tolerances

Every comparison of a tree with the reference on the same backend is bitwise. CPU against
GPU follows the per-class metrics of S§6 (per-cell relative error of the Monte Carlo bins
with equal counts, per-value error of the scalars and the ledger, largest-magnitude
relative error of images and crossings with integer-valued fields exact, equal NaN and
Inf masks), each no worse than the reference's own gap on that deck in
`records/G0/gaps.txt`. The headline bounds (Monte Carlo sums within 1e-11, images within
1.5e-13) are already exceeded by the reference: 6.1e-10 per cell on the stride-1 Compton
bins (`results/blp/mc_audit/speed/gpu/README.md`), and on the images 2.51e-13 (torus
Stokes I), 3.15e-13 (IQUV), 9.03e-11 (precise disk B), 2.31e-9 (Kerr disk)
(`logs/decks_gpu_3789.log`, at `2c50d86`). The per-deck reference gap is the bound only
until the user rules on D4.

## 9. Decisions for the user

The team does not take these; the lead raises them with the evidence, and every item they
do not block continues.

| | decision | what it moves | what it buys | evidence |
|---|---|---|---|---|
| D1 | GPU bits may move onto the host's elementary functions (BLP-owned ports of `hypot`, `exp`, `log`, `cbrt`, `pow`, trig) | GPU baselines of every deck that calls them | host bits independent of the depot's package images (the host `hypot` takes Base's non-FMA branch because a generic-target image compiled it first); smaller CPU–GPU gaps; half of CPU = GPU | `map_compiler.md` §3.2; U1 |
| D2 | one host re-baseline: StaticArrays reductions and `muladd` chains of the decode and the Kerr–Schild routes become explicit `fma` chains in the device's association | CPU bits of the decode-reading decks (661 of 104,776 site-scan zones, ≤ 3 ulp in `theta_e` and `B`); the CPU cost U1 measures (dependent `fma` chains replace vectorisable reductions) | the site scan on the device (2.0 s → about 0.1 s per uncached stride-1 call); CPU = GPU bitwise on converted decks (with D1); Verner stage rescheduling bitwise by construction (precise march 1.65 s → about 1.1–1.4 s, *estimate*) | `map_compiler.md` §3.3, §5; `design_aux.md` §3, §6.3; U1 |
| D3 | only if W1 or W2 fails: one Monte Carlo re-baseline in which the step body is one non-inlined function shared by `track!`, the timed programs and `MCAdvance` | the last bits of every Monte Carlo deck on both backends, once | every later schedule change (segments, sort, cohort, kick placement) bitwise by construction on each backend | W1, W2 records |
| D4 | the CPU–GPU tolerance: the per-deck reference gaps of G0 under the metrics of S§6, or a re-baseline to the headline bounds | the bound of gate D's gap check | a written tolerance that the reference meets | `records/G0/gaps.txt`; §8.5 |
| D5 | if G0 shows application clocks pinned at the 1,065 MHz base: ask the administrator to restore the 1,410 MHz boost | every GPU row by about 1.32× (ratios within a job unchanged) | free device speed | G0 clock query |
| D6 | a freeze or rebase window on `track.jl`, `sinks.jl`, `programs.jl`, `montecarlo.jl`, `cascade.jl` with the Monte Carlo audit team during W1–W4 | the audit team's schedule | W1–W4 without conflicting rewrites; references that stay valid | `git log` of main on 2026-11-07 |

## 10. Records of this plan

- Standard: `docs/src/design/gpu_execution.md` on `gpu-wavefront` (`46eced3`, on `4c396ed`
  and main `f6decd6`), strict docs build passing (`plan/docs_build/with_doc3.log`).
- This plan: `results/blp/gpu_infra/PLAN.md` in the main checkout and
  `/data2/haiyangw/scratch/wavefront/plan/PLAN.md`.
- Per item: `records/<item>/README.md` under `/data2/haiyangw/scratch/wavefront/`, with the
  tree, job ids, logs, tables, fingerprints and the gates passed.
