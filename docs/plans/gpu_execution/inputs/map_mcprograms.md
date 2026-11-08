# Map of the Monte Carlo programs

Tree: branch `gpu-wavefront` at `a5923dd` (identical to `main`). Paths are relative to the
repository root; `results/...` paths live only in the main checkout
`/data/haiyangw/claude/BlackLightPlus` (untracked there). Measured numbers are quoted from
`results/blp/mc_audit/speed/gpu/README.md` (the A100 record of 2026-11-07),
`results/blp/mc_audit/speed/PLAN.md` (the CPU profile, tree `899dbe1`, before the plunge
stop and the Poisson events) and `bench/kernel_audit_baseline.json` (register audit at
`7be2569`, a commit outside the history of `a5923dd`; the Monte Carlo code changed since,
so those counts are indicative and need a re-audit).

## 1. Call graph

```
mc_spectrum                       src/pipeline/Products/montecarlo.jl:249-283
 └ mc_run (sphere | grid)         montecarlo.jl:305-326 | 339-368
    ├ grid_zones → emission_sites, grid_tables   montecarlo_tables.jl:134-163, src/infra/Data/tables.jl:103-163
    ├ zone_offsets (stochastic rounding)          montecarlo.jl:372-381
    └ run_layers                                   montecarlo.jl:387-397
       └ Execution.foreach_cascade!               src/infra/Execution/cascade.jl:67-79
          ├ threads: chunk_generations!            cascade.jl:208-235 (run_growing! 248-264)
          └ pool:    chunk_generations!            cascade.jl:290-338 (launch! kernel.jl:33-44)
             generation 0: MCPrimary(item)         src/pipeline/Products/programs.jl:428-434
                ├ emit(ZoneSource|SphereSource)    programs.jl:404-409 | 380
                │   └ generate                     src/physics/MonteCarlo/generate.jl:40-82 | 92-99
                ├ add_ledger!(EMITTED)             programs.jl:431
                └ transport! → track!              programs.jl:363-365 → track.jl:276-342
                    ├ march_control/geodesic_step  track.jl:98-108 (Curved), 83-96 (Flat)
                    ├ sample (plasma)              src/physics/Matter/protocol.jl:64-78
                    ├ absorption/scattering_rate   opacity.jl:42-86
                    ├ step_events                  track.jl:221-251
                    │   ├ scatter_site             track.jl:191-196
                    │   ├ child_stream             programs.jl:327 → children.jl:60-61
                    │   ├ scatter_superphoton      scatter.jl:264-278, 304-305 (the kick)
                    │   └ spawn!                   programs.jl:328 → children.jl:69-76
                    ├ record! (escape)             programs.jl:320-325 → accumulators.jl:161-197
                    └ tally! (ledger)              programs.jl:330-344
             generation g ≥ 1: MCLayer(item)       programs.jl:461-465 (same transport!)
```

One kernel serves every program (`_foreach!`, `src/infra/Execution/kernel.jl:7-10`): one
thread per item, `prog(item, i)`. `MCPrimary` and `MCLayer` launch in blocks of 32
(`programs.jl:440, 467`). `track!` carries no `@inline` (`track.jl:276`); everything it calls
is `@inline`, so the kick, the plasma sample, the rates and the sink are compiled into the
body of `track!`.

## 2. Records and their sizes

Sizes from the field layout (Julia C alignment; checked with plain structs of the same
fields).

| record | bytes | definition | contents |
|---|--:|---|---|
| `Superphoton` | 144 | `superphoton.jl:18-31` | `x` 0–31, `p` (covariant, mₑc²) 32–63, `w` 64, `E = −p_t` 72, `tau_abs` 80, `tau_scatt` 88, `nscatt::Int32` 96 (+4 pad), `x1i` 104, `x2i` 112, `ne0` 120, `thetae0` 128, `b0` 136 |
| `MCRng` | 24 | `src/physics/Numerics/philox.jl:73-77` | `key::NTuple{2,UInt32}`, `ctr::UInt32` (draw index), `sid::NTuple{3,UInt32}` |
| child slot | 168 | `children.jl:30-39` | `photons[j]::Superphoton` and `rngs[j]::MCRng` in two arrays, one `count::Int64` |
| `ZoneSite` | 64 | `src/physics/Matter/zones.jl:62-68` | `x`, `dV`, `n_e`, `theta_e`, `B` |
| `EmissionZone{Float64}` | 416 | `zones.jl:25-36` | 4 scalars, `x`, `ucon`, `ucov`, `bcon`, `gcov`, `gcon` |
| `PlasmaState{Float64}` | 184 | `src/physics/Matter/records.jl:33-45` | `n_e`, `theta_e`, `B`, `ucon`, `ucov`, `bcon`, `bcov`, `b_sq`, `sigma`, `beta_inv`, `valid` |
| `MarchLedger` | 80 | `sinks.jl:69-79` | `outcome`, `scatter_loss`, `scatter_loss_final`, `spawned`, `unspawned`, `absorbed`, `w`, `E`, `events` |
| fluid cell | 32 | `src/infra/Data/athenak_hdf5.jl:78` | `data[f, i, j, k, mb]`, `Float32`, the eight primitives of a cell contiguous, one ghost layer per side |

What the production sink and the march read of a `Superphoton`: `x`, `p`, `w`, `E`, `nscatt`
(`programs.jl:320-344`, `track.jl:281-287`). `tau_abs`, `tau_scatt` are accumulated and
passed to children (`track.jl:239`, `331-332`) and `x1i`, `x2i`, `ne0`, `thetae0`, `b0` are
copied unchanged (`track.jl:240-242`, `335-337`), but `MCSink` never reads them: 56 of the
144 bytes are diagnostics in the device programs (the host `VectorSink` keeps them).

## 3. The life of a photon

### 3.1 Emission

Grid (`ZoneSource`, `programs.jl:392-409`):

1. Zone of primary `g`: `searchsortedfirst(offset, g) − 1` over the `nz + 1` cumulative
   counts (`programs.jl:405`): about 24 dependent loads of an 82 MB `Int64` vector at
   stride 1 (10,303,459 zones).
2. The zone again: `emission_zone(provider, st, site, params)` (`zones.jl:78-82`) re-samples
   the plasma at `site.x` (locate, trilinear, metric, decode) for `ucon`, `ucov`, `bcon`,
   `gcov`, `gcon`; `n_e`, `Θₑ`, `B`, `dV` are the site's.
3. `generate` (`generate.jl:40-82`) on the stream `primary_rng(seed, g)` (`programs.jl:430`):
   frequency by rejection, two draws per trial, `weight_at` and `int_jnu` (the `F(K)` table,
   `emissivity.jl:181-222`) per trial, cap `REJECT_MAX_ITER = 100,000` (`generate.jl:48-55`);
   pitch angle by rejection against `jmax`, two draws and one `thermal_synchrotron` per
   trial (`generate.jl:57-65`); one azimuth draw (`66-67`); the emission tetrad
   (Gram–Schmidt under `gcov`, `emissivity.jl:245-268`), `kcon = Eᵀ k_tet`, `p = gcov kcon`
   (`generate.jl:73-75`). The photon starts with `tau = 0`, `nscatt = 0`, the zone's
   diagnostics.
4. `EMITTED += w E` into the ledger stripe `(g − 1) mod 64` (`programs.jl:429-431`).

Sphere (`SphereSource`, `programs.jl:374-380`; `generate.jl:92-99`): one draw for the
radius, two for the direction, the sphere's zone at that point, then step 3.

### 3.2 The march loop (`track!`, `track.jl:276-342`)

Initialisation (`278-291`): `y = (x, p)`, `w`, `tau_a`, `tau_s`; one plasma sample at the
start point (`283`) for the start rates `na0` (absorption), `ns0` (scattering) and `th0`
(temperature, scattering runs only); `spawning = S && nscatt < max_scatt` (`287`);
`r_prev = Inf`, `nev = 0`, five Float64 accumulators (`loss`, `loss_final`, `spawned`,
`unspawned`, `absorbed`).

Each iteration `it = 1 … max_steps(ctl)` (`292-334`; `CURVED_MAX_STEPS = 20,000`,
`track.jl:23`; flat `SPHERE_MAX_STEPS + 1`):

| order | line | work |
|---|---|---|
| control | 293 | `march_control`: `proxy_distance` (Kerr–Schild radius), `g^{μν}(x) p` for `|k|`, stop test, `h = d/(C(|k| + 1e-300))` (`98-106`) |
| stop | 294-298 | escape (`r > escape`, or `r > R_em` outbound), capture (`d < HORIZON_EPS_SAMPLED = 1e-3` or non-finite), else dropped |
| plunge | 299-302 | only when `!spawning`: inward and `r < plunge_radius(st)` → captured (`140-141`; `src/physics/Spacetime/termination.jl:290-291`) |
| geodesic | 305 | `rk4_step`: four `geodesic_rhs`, each one `raise_and_norm` on a position with three `ForwardDiff` partials (`src/physics/Spacetime/geodesic.jl:80-88, 189-195`) |
| sample | 306 | `sample(provider, st, x_{n+1})`: `locate` (leaf table, one `Int32` load, `src/infra/Data/fluid_blocks.jl:243-250`), trilinear over 8 cells × 8 `Float32` (`314-330`), and where `ρ > 0` `metric_and_inverse` and `decode` (`protocol.jl:64-73`) |
| rates | 307-308 | `absorption_rate` (`ν α_ν`: `thermal_synchrotron` with `exp`, `cbrt`, `expm1`, `opacity.jl:42-52`); with S `scattering_rate` (`hotcross_lookup`: two `log10`, four table loads, `10^x`, `hotcross.jl:153-185`) |
| depths | 309-312 | trapezoid `dτ_a`, `dτ_s`, `lost = (1 − e^{−Δτ})/Δτ` |
| absorbed | 313 | `absorbed += w E dτ_a lost` |
| events | 314-329 | with S and `dτ_s > 0`: if `spawning`, `loss += w dτ_s lost`, the bias `(b, copies)` from `max(th0, th1)` and `w` (`319`, `opacity.jl:158-166`), `step_events` (`321-323`); else `loss_final += …` (`325`) |
| decay | 330-333 | `w *= e^{−(dτ_a + dτ_s)}`, `tau += dτ`, `y = y_{n+1}`, `na0 = na1`, `ns0 = ns1` |

The trajectory does not depend on the plasma: `march_control`, `geodesic_step` and the
stop and plunge tests read only `y`, `r_prev`, `it` and the static `spawning`
(`293-305`); weight never ends a march (no roulette or weight floor). The plasma only
decays `w`, fills the accumulators and drives the event draws.

### 3.3 Poisson scattering events (`step_events`, `track.jl:221-251`)

- Child weight `W = w(1 − e^{−Δτ})/(bΔτ)/copies` (`223`, `163`); mean event count
  `λ = b dτ_s` (`224`).
- Events are cumulated exponential gaps on the CDF coordinate `u` (`next_event`,
  `174-179`): one uniform per gap; a step with no event draws one uniform.
- Per event: `geodesic_step(ctl, y_n, h·frac)` (a fresh RK4 from the step start, `230`); the
  plasma at the event point, `scatter_site` (`232`, `191-196`: a full `sample`, validity
  `ok`).
- Per copy: `child_stream(sink, rng, nev)` (`234`), `nev += 1` (`235`); if `ok && fits &&
  W > weight_min` (`236`; `weight_min = 0`, `opacity.jl:7`): the kick
  `scatter_superphoton(p_ev, zone, crng)` (`237`), the child `Superphoton(xev, p_new, W;
  tau at the step start, nscatt + 1, the parent's diagnostics)` (`238-242`), `spawn!`
  (`243`), `spawned += W`; else `unspawned += W` (`246`).
- `spawn!` takes a slot from one atomic counter per buffer and writes 168 bytes if the slot
  exists (`children.jl:69-76`); arrival order is not deterministic, the set is.

### 3.4 Stop, escape, record

After the loop (`335-341`): the output photon; on `ESCAPED_MARCH` `record!(sink, out)`
(`338`): `MCSink.record!` (`programs.jl:320-325`) bins order `min(nscatt, K) + 1` with
`escape_direction` (`g^{μν}(x) p` at the final point, `track.jl:152-155`), then
`ExactBins.record!` (`accumulators.jl:161-169`): `log ν`, the bin, `acos` of the polar
angle, and for the all-sky row and each inclination band that contains it `add_cell!`
(`191-197`): one `Int64` atomic and three exact sums of up to three limb atomics each
(`src/physics/Numerics/exactsum.jl:79-90`). `ESCAPED += w E` into the ledger.

### 3.5 Ledger

`tally!(sink, sph_in, MarchLedger(...))` (`track.jl:339-340`, `programs.jl:330-344`) adds
`ABSORBED`, `CAPTURED` or `DROPPED` (`w E`), and in scattering runs `SPAWNED`,
`SPAWNED_BY_PRIMARIES` and `PRIMARY_WEIGHT` (primaries), `SCATTER_LOSS`,
`SCATTER_LOSS_FINAL`, `UNSPAWNED`. Every quantity is an `ExactScalars` of 69 `Int64` limbs
(`exactsum.jl:17-19`) in 64 stripes × 11 slots (`programs.jl:244-258`); `add!` is up to
three limb atomics and one count atomic (`accumulators.jl:425-429`). Sums are exact, so an
addend can be added from any stage at any time; the per-photon values (`spawned`, `loss`,
`absorbed`, …) are Float64 sums in step and event order and must keep that order. The
addend count matters for `ESCAPED` only (`n_recorded`, `montecarlo.jl:366`).

### 3.6 Children and generations

`foreach_cascade!` (`cascade.jl:67-79`) cuts the primaries into chunks of consecutive
indices: on a pool `min(cld(n, 4·ndev), 2^20)` (one A100, 10⁶ primaries: 250,002 per chunk,
four chunks on two lanes, `cascade.jl:24-25`, `pool.jl:86`); on threads
`cld(n, 64·nthreads)` (`26-27`). Per chunk, generation `g` tracks the children of `g − 1`,
up to `depth = max_scatt` (`chunk_generations!`, `290-338`). Layer `g` photons all carry
`nscatt = g`; the last layer never spawns (`spawning` false).

Pool details: a non-final generation records into lane scratch accumulators, drained
into the totals after its children fit (`CascadeSwap`, `115-154`; `Drain`, `163-187`;
`330`); the child buffers are two lane arrays swapped per generation (`341-343`); the
count is read back after every launch (`352-356`). A generation that overflows its slots
reruns in full with `n_out·9/8` slots (`324-329`); the start capacity is
`max(4096, chunk/2, the lane's earlier growth)` (`298-299`), so the first chunk of each
lane can run each non-final generation twice. Overflow reruns were never instrumented
(PLAN.md:119); passing a large `capacity` removes them with bitwise-identical results.

## 4. The Compton sampler (`scatter.jl`)

`scatter_superphoton(p, zone, crng)` (`304-305` → `264-278`):

| step | lines | loops and draws |
|---|---|---|
| emission tetrad under `gcov` (reads `ucon`, `bcon`, `gcov`; not `ucov`, `gcon`) | `emissivity.jl:245-274` | none; seven 4×4 mat-vecs, four `sqrt` |
| `k_tet = (−(E p)₀, (E p)_{1:3})` | 269-270 | |
| `sample_electron` | 121-148 | loop ≤ `MAX_ELECTRON_ITER = 1e5`: `sample_electron_gamma` → `sample_y`, one `r` (`sample_mu`), `kn_total(γ(1−βμ)k⁰)` (`tang_log` above 1e-3), one accept draw |
| └ `sample_y` | 50-83 | loop ≤ 1e5: one mixture draw, `chi_square(3|4|5|6)` chosen by branch, one accept draw (acceptance `√(1+Θy²/2)/(1+y√(Θ/2))`) |
| └ `chi_square` → `standard_gamma` → `_gamma_ge1` | `philox.jl:300-303, 260-268, 271-293` | Marsaglia–Tsang loop ≤ 1000: `mc_gauss` (2 draws, `log`, `sqrt`, `cos`), one draw, up to two `log` |
| electron direction: `photon_triad` (2 draws), azimuth (1 draw) | 137-147, 152-159 | |
| boost into the electron frame | 234, `lorentz_boost` 24-41 | |
| `sample_scattered_energy` | 212-220 | `k0 ≤ 1e-4`: Thomson cosine loop ≤ `MAX_KN_ITER = 1e6`, 2 draws per trial (acceptance 2/3, 174-185); else Klein–Nishina energy loop ≤ 1e6, 2 draws per trial (189-202) |
| scattered direction: `photon_triad` (2), azimuth (1); boost back | 239-250 | |
| `Kcon = Eᵀ kp`, `p_new = gcov Kcon` | 274-276 | |

Four nested data-dependent rejection loops (`sample_electron` ⊃ `sample_y` ⊃
`_gamma_ge1`, then the energy loop), at least 15 uniforms per kick, each a Philox4x32-10
evaluation of which two output words are used (`philox.jl:30-54`). The kick reads no
table: `kn_total` is closed form; the hot-cross table is read by the march
(`scattering_rate`), not the kick. On the CPU the kick is 0.4 % of the Compton torus
profile (PLAN.md:113-115); its GPU cost is its registers and the divergence of one lane
running it while the warp waits.

`hotcross_lookup` (`hotcross.jl:153-185`), read every step of a scattering run, falls back
to `total_compton_cross_num` (`93-120`: 40 × 240 = 9,600 iterations with `tang_exp` and
`kn_total`, plus a Bessel normalisation) for `Θₑ ≥ 1e4`, `ε ≤ 1e-12` or `ε ≥ 1e15` with
`εΘₑ ≥ 1e-6`; how often the torus reaches it is not measured.

## 5. Random streams

- Draw `n` of a stream is `philox4x32((n, sid₁, sid₂, sid₃), (seed, id₀))`, two words to one
  `Float64` (`philox.jl:51-54, 90-94`); the counter is 32 bits.
- Identity `I = gid + Σ_j e_j 2^{G+(j−1)E} + k 2^{124}` (`philox.jl:96-109`): `gid` the
  primary index, `e_j` the event ordinal of the generation-`j` ancestor, `k` the layer.
  `widest_layout(N, K)` sets `E = min(32, (124 − G) ÷ K)` (`147-153`); 10⁶ primaries, three
  layers: `G = 20`, `E = 32`, `2^32` ordinals per photon (`event_capacity`, `161`).
- `primary_rng(seed, g)`: identity `g`, counter 0 (`192-193`). `child_rng(parent, e, L)`:
  the parent's identity with `e` in the next field and the layer tag incremented, counter
  0, independent of the parent's draw count (`213-220`). `run_rng` (`202-203`) and
  `MCRng(seed, 0)` (`zone_offsets`, `montecarlo.jl:374`) name host decisions no photon
  carries.
- Draw order on a primary's stream: generation (§3.1), then in the march one uniform per
  spawning step with `dτ_s > 0` plus one per event (`next_event`). Thin runs draw nothing in
  the march. A child's stream starts with its kick's draws and continues with its own
  march (`spawn!` stores the stream advanced past the kick, `track.jl:237, 243`). Copies of
  one event take consecutive ordinals.

Consequences for any split: a photon's draws come only from its own stream in a fixed
order; the kick consumes a prefix of the child's stream that nothing else reads, so it can
run in any kernel at any time; the set of children, every child's stream and every bin
addend are independent of scheduling, grouping and order.

## 6. Order of the work

- Zones are in block order, within a block groups `k`, `j`, `i` outermost to innermost
  (`src/infra/Data/tables.jl:100-101, 123-151`). `zone_offsets` gives each zone a
  contiguous range of primary indices (`montecarlo.jl:372-381`); at stride 1 the mean is
  0.1 primary per zone, so 32 consecutive primaries (one block of `MCPrimary`) come from
  about 320 consecutive cells of one or two meshblocks. Emission is spatially coherent;
  frequencies and directions are independent, so a warp spreads out from its first steps.
- A chunk is a contiguous primary range, so a chunk covers contiguous meshblocks.
- Children sit in atomic arrival order (`children.jl:69-76`): an `MCLayer` warp holds
  children of unrelated parents anywhere in the plasma from its first step. In the Compton
  deck children are 7.3 of the 8.3 million tracked photons.
- The plasma is about 360 MB of `Float32` blocks (PLAN.md:113), beyond the A100's 40 MB L2;
  a sample is 4 runs of 64 contiguous bytes, `(ni+2)·32` and `(ni+2)(nj+2)·32` bytes apart,
  plus the leaf-table and block-geometry loads.

## 7. Register- and branch-heavy parts

| part | where | kind | evidence |
|---|---|---|---|
| RK4 with Dual-number metric (4 RHS, 3 partials each) | `geodesic.jl:80-88, 189-195` | registers | `mc_torus_thin_primary` at 255 registers, 2.26 KB stack, 312 B callee spills, against 117 registers for the flat-march `mc_sphere_thin_primary` (baseline `7be2569`) |
| plasma decode with the metric pair | `protocol.jl:64-73`, `src/physics/Matter/decode.jl` | registers, memory | 188 ns per CPU sample, 20.5 % of the thin profile in `primitives` (cache misses) (PLAN.md:107-114) |
| Compton kick inlined in the march | `track.jl:237`, `scatter.jl` | registers, branches (4 nested rejection loops) | torus scatter kernels 255 registers, 2.85–2.96 KB stack, 1.9 KB callee spills; the sphere goes from 117 to 255 registers with scattering (baseline); a non-inlined sampler was slower on the A100 (743 against 643 ms) and stayed at the cap (`docs/src/design/performance.md:195-200`) |
| event branch per step | `track.jl:316-323`, `221-251` | branches | a lane with an event runs a partial RK4, a full sample and the kick while the warp waits |
| lifetime divergence | `track.jl:292-302` | branches (loop exit) | captured 1,579 steps, escaped 547, step limit 20,000 (thin torus before the plunge stop, PLAN.md:94-101); spawning photons have no plunge stop (`299`; performance.md:266) |
| generation rejection | `generate.jl:48-65` | branches | two loops, data-dependent trip counts |
| hot-cross fallback | `hotcross.jl:185` | branch, 9,600-iteration loop | not measured |
| record rows | `accumulators.jl:180-189` | atomics | 1 + bands per escape, up to 10 atomics per row |
| zone search | `programs.jl:405` | dependent loads | ~24 loads per primary |

The measured ratios point at a GPU-specific penalty in the scattering run: per tracked
photon the Compton phase costs 0.67× a thin photon on 32 CPU threads (62.5 s/8.3 M against
11.3 s/1.0 M) but 2.0× on the A100 (62.5 s/8.3 M against 3.74 s/1.0 M) (gpu/README.md).
At a few hundred to a thousand steps per photon the A100 sustains about 1e8 steps/s
(PLAN.md:131), a few percent of its FP64 rate and far below its DRAM bandwidth, so the
limits are latency at the occupancy of 255-register kernels (8 warps per SM), divergence
and the incoherent sample loads, not throughput.

## 8. Natural stage boundaries

### 8.1 The stages and the state that crosses each cut

| cut | location now | state across it |
|---|---|---|
| A. Emit │ March | `programs.jl:430` → `432` | `Superphoton` + `MCRng`, i.e. one child slot (168 B; 100 B without the diagnostics). Primaries can enter the same queue as children. |
| B. March K steps │ March K steps | after `track.jl:333`, before `293` | the live state below |
| C. March │ Kick | `track.jl:236` → `237` | the event record below |
| D. March │ Finish | `track.jl:335-341` | the done record below |
| E. Event site │ March (optional) | `track.jl:228` → `230` | `y_n`, `h`, `frac`, `W`, ordinal; needs the per-photon `spawned`/`unspawned` sums rebuilt in ordinal order (§8.3) |
| F. Geodesic │ plasma (optional) | `track.jl:305` → `306` | stored `(y_{n+1}, h_n)` per step, 72 B; legal because the trajectory is plasma-independent (§3.2) |

Live state (cut B), every field carried, nothing recomputed at a resume:

| field | bytes | note |
|---|--:|---|
| `y` | 64 | position and covariant momentum |
| `w` | 8 | |
| `na0`, `ns0`, `th0` | 24 | rates and temperature at the segment start; re-sampling at a resume risks other bits (§8.4) |
| `r_prev` | 8 | |
| `it` | 4 | `FlatMarch` reads it (`track.jl:88`); ≤ 100,001 |
| `nev` | 8 | event ordinals continue across segments |
| `loss`, `loss_final`, `spawned`, `unspawned`, `absorbed` | 40 | Float64 per-photon sums; never split into partial sums |
| `rng` | 24 | only `ctr` changes; `key`, `sid` are the identity |
| `nscatt` | 4 | `spawning` follows from it and `max_scatt` |
| `tau_a`, `tau_s` | 16 | diagnostics, passed to children |
| `x1i`, `x2i`, `ne0`, `thetae0`, `b0` | 40 | diagnostics; not read by `MCSink` |
| `w0` | 8 | only if `PRIMARY_WEIGHT` stays at the tally; it can move to Emit (exact sum) |

About 200 B per live photon without the diagnostics, 256 B with them; a chunk of 250,000
photons is 50–64 MB per side of a ping-pong pair.

Event record (cut C), with the site sampled in the march so `ok` and the per-photon
`spawned`/`unspawned` sums stay where they are:

| field | bytes |
|---|--:|
| `xev`, `p_ev = momentum(yev)` | 64 |
| `ucon`, `bcon`, `theta_e` | 72 |
| `gcov` (16; 10 if it is bitwise symmetric) | 128 |
| `W` | 8 |
| child stream `child_rng(rng, nev)` | 24 |
| child `nscatt` | 4 |
| `tau_a`, `tau_s` at the step start, `x1i`…`b0` | 56 (diagnostics) |

About 300 B, 356 B with diagnostics. The kick reads neither `ucov` nor `gcon`. The kick
stage writes the child (`Superphoton` + advanced stream) into the live queue; the
generation layering of the cascade becomes optional, since nothing depends on which
generation shares a launch.

Done record (cut D): `x`, `p` (64), `w` (8), `nscatt` (4), outcome (4), `nev` (8), the five
accumulators (40): 128 B; `E = −p_t` and the escape direction are recomputed from `p` and
`x`, as `track!` does now.

### 8.2 A wavefront schedule

Per chunk on a pool lane:

```
live ← Emit(primaries)                      # cut A
repeat until live is empty
    (live′, done, events) ← MarchK(live)    # cut B; no side effects but its buffers
    Finish(done)                            # cut D: record!, tally!
    live′ ← live′ ∪ Kick(events)            # cut C: children join the live queue
    live ← live′                            # optionally reordered by block or Morton key
```

- `MarchK` writes only its output buffers and counters, so an event-buffer overflow reruns
  it from the unchanged input with a larger buffer, without scratch accumulators or
  drains; the done buffer never overflows (at most one record per live photon).
- Survivors are compacted by the atomic append, which removes the lifetime divergence
  (the slowest photon no longer holds a warp of finished ones), and the queue can be
  refilled with new primaries to keep the device full instead of draining each chunk.
- Reordering the live queue by meshblock (counting sort by `locate`, or by a Morton key of
  `x`) restores coherence of the sample loads; children, which start at random sites,
  gain most.
- `K` is an executor tuning like the chunk: results do not depend on it. The host reads
  three counters per iteration.
- On threads the same stages run with `K = max_steps` (one segment per photon) and the kick
  after the chunk's march, which is the current depth-wise cascade with the kick moved out
  of the march; or the fused `track!` stays, as the executors already schedule the cascade
  differently (`cascade.jl:191-264` against `268-338`). Either way the backend choice stays
  inside the executor.
- Two devices: chunks still go to lanes; streams and exact sums keep one GPU and two GPUs
  bitwise.

### 8.3 Optional finer cuts

- Cut E removes the partial RK4 and the site sample from the march, leaving a march whose
  only branch is the stop. The per-photon `spawned` and `unspawned` are Float64 sums over
  the events in ordinal order, and the deck compares `w_children`; with E the site stage
  writes `(parent, ordinal, W, ok)` and the finish stage sums them per parent in ordinal
  order once all of a parent's events are sited. Worth it only if the event-site branch
  still dominates after C.
- Cut F splits the step into a geodesic kernel (RK4 only) and a plasma kernel (sample,
  rates, depths, events) over stored states; it lowers the registers of each half (the
  thin torus march alone is at the 255-register cap) at 72 B per stored step. The event
  partial step still needs an RK4 in the plasma half, or cut E. The imaging two-phase
  pipeline was dropped for its PCIe copy (performance.md:184-185); here the states stay
  on the device.

### 8.4 Bitwise risks of each cut

1. Contraction and inlining context. Moving the sampler behind a non-inlined call changed
   the CPU bits of the torus scattering spectrum (performance.md:195-200), and a Verner
   stage loop changed the CPU bits of the step (README Findings; performance.md:294-296).
   Every cut puts code into a new program; each must be prototyped and checked on the
   fixed decks on both backends before it is adopted. Cuts that keep the physics calls in
   the same function bodies (B: the loop body unchanged; C: `scatter_superphoton` called
   with the same arguments) carry the least risk; the mechanism of the earlier bit change
   is not established.
2. Carried, never recomputed: `na0`, `ns0`, `th0`, `r_prev` across B; the child's start
   sample stays the child's own (`track.jl:283`) rather than the parent's event-site sample,
   unless the two are shown bitwise equal.
3. Per-photon Float64 sums keep their order (B carries them; C and D move no addend of
   them); ledger addends may move between stages but keep their values; one `ESCAPED`
   addend per escape.
4. `nev` and `it` continue across segments (child streams; `FlatMarch` stop at `it > 1`).

## 9. Measurements that test the incoherence hypothesis

On the existing code, without edits to the package:

1. `capacity = 4_000_000` against the default on the Compton deck: the share of overflow
   reruns in the 62.5 s (bitwise-identical bins by construction).
2. Per-photon steps and events on the CPU through a forwarding control wrapper and a
   recording sink (`track!` is generic in both; the counted runs of PLAN.md:119-120 used
   this): the warp efficiency `Σ steps / Σ_warps 32·max steps` for consecutive primaries and
   for arrival-ordered children, the fraction of warp-iterations with an event, and the
   children per generation.
3. Gen-0 time of the Compton deck with `weight_min = Inf` (every event unspawned: no kick,
   no spawn, same draws) and with `max_scatt = 0` (scattering rates, no events), against
   the thin run: the cost of the kick and spawn, and of the scattering rate, in the march.
4. `ncu` on `MCPrimary` and `MCLayer` (thin and Compton): registers, local-memory traffic,
   achieved occupancy, `smsp__thread_inst_executed_per_inst_executed` (divergence), sectors
   per request of the global loads (sample coherence), stall reasons (long scoreboard
   against branch resolution).
5. A layer with its children sorted by meshblock before launch (host-side sort of a copy
   in a prototype), against arrival order: the cost of sample incoherence alone.
6. The kernel audit (`bench/kernel_audit.jl --only mc_*`) at `a5923dd`: the baseline counts
   are from `7be2569`.

## 10. Adjacent observations

- The emission samples the plasma at the zone twice: in `emission_zone` (`zones.jl:79`) and
  again at the start of `track!` (`track.jl:283`).
- `march_control` re-evaluates the metric and the radius at the point the previous step
  sampled: 37 of its 44 ns per step on the CPU (PLAN.md:107-110).
- The zone search could be a precomputed primary-to-zone map (4 B per primary).
- `MCRng` uses two of the four Philox output words per draw.
