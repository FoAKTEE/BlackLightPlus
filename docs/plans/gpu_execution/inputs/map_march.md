# Map of the geodesic march and its sampling

Tree: `a5923dd` (main = worktree `/data2/haiyangw/scratch/blp_wavefront`). Paths are relative to
the repository root. Offline device numbers are `ptxas --verbose` for sm_80 (ptxas 13.3.33,
Julia 1.12.6, CUDA.jl 6.2, the launch configuration of `bench/kernel_audit.jl`), produced by
the scratch scripts in `/data2/haiyangw/scratch/wavefront/plan/audit/` (`march_audit.jl`,
`march_audit2.jl`, logs beside them, PTX and SASS kept for the crossing kernels). Spill
figures from ptxas are static bytes of spill instructions, not dynamic traffic. "Measured"
marks a number from a run; "estimate" marks a count from the code.

## 1. The imaging item, end to end

| layer | code | what it does |
|---|---|---|
| kernel | `src/infra/Execution/kernel.jl:7-10` | `_foreach!`: one thread per item, `prog(Int(items[i]), i, outs...)` |
| launch | `src/infra/Execution/kernel.jl:33-44` | workgroup from `workgroupsize(prog)` (IQUV 64, `programs.jl:56`; Monte Carlo 32, `programs.jl:440,467`; otherwise the backend's) |
| program body | `src/pipeline/Products/programs.jl:50-54` | `ImagingProgram` call operator, compiled as a device function of its own (no inlining hint, `programs.jl:5-7`) |
| camera ray | `programs.jl:22` → `src/infra/Execution/ray_sources.jl:175-179` → `src/physics/Spacetime/camera.jl:64-71` | `@noinline camera_state`; plane camera: one `metric` and `null_covariant` (`geodesic.jl:41-57`); with `infinity` the far-zone march `far_zone.jl:84-120` |
| lane | `programs.jl:26` | `Float64` ray narrowed to the geodesic lane (`MixedLanes`: `Float32` bulk) |
| march | `src/physics/Transfer/march.jl:81-99` (Stokes I), `:136-195` (IQUV) | per step: `march_control`, plasma `sample`, coefficients, segment, geodesic step |
| control | `src/physics/Transfer/controls.jl:58-59,85-86` → `src/physics/Spacetime/termination.jl:190-195` | Kerr: one closed-form `_ks_term` gives `r`, `E_n`; `h = (r − r_h)/C` (× `winding_factor`, `termination.jl:143-148`) |
| step | `src/physics/Spacetime/integrators.jl:117-119` | `rk4_step` (`geodesic.jl:189-195`) or the Verner 9 `erk_step` (`integrators.jl:92-93`) |

The crossing program of the precise march (`src/pipeline/Products/crossings.jl:67-108`)
replaces the transfer by `march_step_compensated` (`integrators.jl:131-137`, crossings.jl:88)
and, at each equator crossing whose order is wanted, `surface_crossing`
(`integrators.jl:193-215`, crossings.jl:94): Newton on the partial step, up to 33 further
full steps of the integrator.

Precise mode (`src/pipeline/Plans/config.jl:113-119`): `:verner9`, `C = PRECISE_C = 20`
(`config.jl:97`), redshift cut `1e6` (`config.jl:99`), no horizon backstop, `winding = true`,
`max_steps ≥ 1e5`, camera at infinity. Production: RK4, `C = 80`, `max_steps = 2500`
(`config.jl:87-88`).

## 2. Per-ray state

| march | loop-carried state (code) | bytes | f64 registers |
|---|---|--:|--:|
| Stokes I, f64 lane | `y::SVector{8}`, `Iinv`, `trans`, `r_prev`, `it` (`march.jl:87-91`) | 96 | 24 |
| Stokes I, f32 lane | `y::SVector{8,Float32}`, `r_prev::Float32`, f64 `Iinv`, `trans` | 56 | 14 |
| Stokes I ladder of K sets | `Iinv`, `trans` become `SVector{K}` (`march.jl:54-67`) | 80 + 16K | 20 + 4K |
| IQUV | `y`, `eU`, `eUu` (`SVector{4}` each), `S::SVector{4}`, `G::SMatrix{4,4}`, `r_prev` (`march.jl:142-145`) | 296 | 74 |
| crossings, Verner 9 | `y`, `lo` (compensation), `r_prev`, `n`, `k`, `maps` = K × 5 `Float64` (`crossings.jl:72-80`) | 272 (K = 3) | 68 |
| crossings, RK4 | as above without `lo` | 208 | 52 |
| Monte Carlo photon (`track!`) | `y`, `w`, `tau_a`, `tau_s`, `na0`, `ns0`, `th0`, `r_prev`, five ledger sums, `nev`, `it`, `rng`, the parent fields kept for children (`track.jl:281-291`, loop `:292-334`) | ≈ 250 | ≈ 60 |
| Monte Carlo records | `Superphoton` (`src/physics/MonteCarlo/superphoton.jl:18-31`) 144 B; `MCRng` (`src/physics/Numerics/philox.jl:73-77`) 24 B | 168 | |

The ray state itself is `RayState = SVector{8}` = `[x^μ; p_μ]` (`src/physics/Numerics/raystate.jl`),
64 B in f64. `maps` (15 doubles at K = 3) is carried through the whole crossing march and
written once at the end (`crossings.jl:106`).

## 3. Metric evaluation and derivatives

- **Closed form.** Kerr–Schild `g = η + f l lᵀ`, `g⁻¹ = η − f l^μ l^ν` from one primitive
  `_ks_term` (`src/physics/Spacetime/kerr_schild.jl:12-24`: `hypot`, `sqrt`, four divisions).
  `metric_and_inverse` (`:69-74`), `raise_index` (`:81-85`), `raise_and_norm` (`:94-100`) never
  form a 4×4 matrix. On a device `hypot` is libdevice's (the comment at `:16`), one source of
  CPU/GPU last-bit differences.
- **Derivatives by AD, never by hand** (`src/physics/Spacetime/metric_ops.jl:1-2`).
  Stationary spacetimes: the position seeded with three spatial `Dual` partials
  (`metric_ops.jl:7-12`), and `geodesic_rhs` (`geodesic.jl:80-88`) takes `dx/dλ` from the
  values of `raise_and_norm` and `dp/dλ = −½∂ᵢ(2H)` from the partials of its norm: one Dual
  `_ks_term` per right-hand side. General spacetimes: `ForwardDiff.jacobian` of
  `inverse_metric` over four coordinates (`geodesic.jl:91-106`, `metric_ops.jl:22-25`).
  Parallel transport (IQUV): `metric_and_inverse` at the seeded position, both matrices in
  Duals (`geodesic.jl:123-149`).
- **Binary** (`superposed.jl`): per-hole `_ks_term` and a Woodbury `N×N` solve per
  evaluation. **ADM grid** (`adm_grid.jl:62-79`): ten Catmull–Rom fields over a 4×4×4
  stencil (640 node reads per evaluation, times four in a Dual RHS).
- **Evaluations per Kerr step** (estimate): Stokes I RK4: 1 plain (control) + 1 plain (sample,
  emitting points only) + 4 Dual. Verner 9: 1 + 1 + 15 Dual. IQUV RK4: 1 + 1 + 4 Dual metric
  pairs + two tetrads (`march.jl:156-164`). Monte Carlo thin: 2 plain (control: `proxy_distance`
  and `inverse_metric`, `track.jl:98-106`) + 4 Dual + 1 plain (sample at the step's end,
  `track.jl:306`). The far-zone RHS (`far_zone.jl:30-47`) takes a plain `raise_index_offset`
  and a Dual `geodesic_rhs` at the same point.
- **Fast-math freedom inside the RHS.** `raise_and_norm` contracts with StaticArrays `dot`
  (`kerr_schild.jl:97,99`), whose `@simd` loop (`StaticArrays linalg.jl:273`) puts `reassoc`
  and `contract` flags on the reduction, Dual partials included: the CPU IR of
  `geodesic_rhs` on Kerr holds 14 `reassoc` and 26 `contract` flags; `raise_and_norm` on a
  Dual position alone holds 16 and 29, while `_ks_term` on Duals (with `hypot`) holds none
  (`audit/rhs_flags.jl`, `audit/rhs_flags3.jl`). The decode's
  `gcov * ucon`, `gcov * bcon` (`decode.jl:36,43`) are StaticArrays `muladd` chains and
  `dot(bcon, bcov)` (`decode.jl:45`) another `@simd` reduction. On the device these become
  `fma.rn` chains; every other floating-point operation of the march is emitted with `.rn`
  (crossing kernel: 10,415 `mul.rn.f64`, 7,222 `add.rn.f64`, 1,558 `fma.rn.f64`, 4 plain
  `mul.f64`, the latter the leading products of `dot`s in `camera_state`, each feeding an
  `fma.rn`). So ptxas cannot change any rounding; LLVM's treatment of these flagged reductions is the one place where
  the compile context decides bits (§8).

## 4. The step and its live state

### 4.1 RK4 (`geodesic.jl:189-195`)

Low-storage streaming form: `y`, `acc`, `ys`, `d` (4 × 8 doubles = 64 registers) live across
each RHS. The Dual `_ks_term` and `raise_and_norm` hold about 10–12 Duals of 4 doubles at
their peak (estimate 80–100 registers). Measured: the RK4 Kerr march with no plasma and no
camera ray (`trace_kerr_f64_resume`) compiles to 162 registers, no spills.

### 4.2 Plasma phase of a Stokes I step (`march.jl:39-51`)

`sample` (`src/physics/Matter/protocol.jl:64-73`) → `primitives` (§5) → when `rho > 0`:
`metric_and_inverse` (10 distinct entries per matrix after CSE, about 40 registers),
`decode` (`src/physics/Matter/decode.jl:73-96`: `primitive_four_vectors` `:19-48`, four
4-vectors, then the cgs scalars), `coefficients` (`src/physics/Transfer/coefficients.jl:39-63`:
thermal fit with `exp`, `pow`; `__internal_accurate_pow` is a real call), `segment`
(`segments.jl:21-34`). Measured: the Stokes I body on native blocks without the camera ray
reaches 255 registers with 8 B of spills (`img_I_blocks_kerr_f64_hostrays`); on the f32
geodesic lane 246, none.

### 4.3 IQUV step (`march.jl:146-193`)

Loop-carried 37 doubles; the RK4 of `{y, eU, eUu}` (`march.jl:173-188`) keeps `y, eU, eUu`
(16 doubles), the accumulators `ay/ae/au` (16) and `S, G` (20) live across each transport
RHS (estimate ≈ 106 registers), and that RHS holds the metric pair in Duals
(`geodesic.jl:127-140`: two matrices of 10 distinct Dual entries, ≈ 80 doubles) before its
own products: over the cap. Audit baseline: 255 registers, 4.3 KB stack, 3.3 KB callee
spills; the program runs in blocks of 64.

### 4.4 Verner 9 stage storage (`integrators.jl:74-91`)

Fifteen stage vectors `k1 … k15` (`:75-89`), each `SVector{N}` (N = 8 for the geodesic, 16
for the IQUV transport, 8 for the far zone), combined by the 2010 tableau whose sparsity is:
stage 2–7 read `k1 … k6` sparsely, stages 8–15 read `k1, k6, k7, …, k(s−1)`, the final sum
`k1, k8 … k15` (`:90`). Kept as written, the stage vectors live from their stage to their
last reader. Live 8-vectors during each RHS call (y and the compensation `lo` included):

| call | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| as written | 2 | 3 | 3 | 4 | 4 | 5 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **11** | 10 |
| `V9Sched`: stage sums 11–15 and the final sum opened after `k10` | 2 | 3 | 3 | 4 | 4 | 5 | 4 | 5 | 6 | 7 | **7** | 6 | 5 | 4 | 3 |

Peak as written: 9 stage vectors + `y` + `lo` = 88 doubles = 176 registers across the 14th
RHS (estimate), plus the RHS's own 80–100 and about 38 of the crossing program's other loop
state (`maps`, counters): the step cannot fit in 255 registers. A search over every opening
time of every partial sum (`audit/v9_liveness.py`) finds no schedule below 7; `V9Sched`
(`audit/march_audit2.jl`) reaches it by opening the partial sums of stages 11–15 and of the
final sum right after `k10` (from `k1, k6 … k10`), each still a left fold in the tableau's
order, and adding each later stage vector to the open sums as it is produced. Between two
calls the peak rises to 14 vectors (the moment the sums are opened), where no RHS
temporaries are live.

Two compiled details of the step:

- The ten-term sum of stage 15 (`integrators.jl:89`) is Base's n-ary `+` → `afoldl`, which
  is not inlined on either backend: on the device each Verner step calls `julia_afoldl_*`
  with eight pointer arguments (the stage vectors pass through the stack); on the CPU the
  IR of `march_step_compensated` calls `afoldl` too. The final sum (`:90`) is inlined.
- Code size (estimate). One inlined Kerr RHS is about 800 SASS instructions (from the difference of the
  RK4 and Verner 9 crossing kernels: 37,608 and 72,848 instructions for 31 and 75 inlined
  right-hand sides). The Verner 9 crossing kernel holds four Verner bodies (the
  compensated step and three in `surface_crossing`, `integrators.jl:196,203,212`) plus the
  far-zone one in `camera_state`; one step is about 12,000 instructions (≈ 190 KB of SASS)
  against about 3,200 (≈ 51 KB) for RK4. Whether the precise march stalls on instruction
  fetch is a profiler question (§9).

Measured device resources of the crossing program (stack is per thread, cumulative):

| kernel | regs | stack B | body spill st/ld B | `camera_state` spill st/ld B |
|---|--:|--:|--:|--:|
| RK4 crossings | 255 | 2,248 | 1,008 / 872 | 944 / 1,120 |
| RK4 crossings, host rays | 255 | 1,592 | 1,028 / 904 | – |
| Verner 9 crossings (camera at infinity or plane) | 255 | 5,624 | 2,304 / 4,368 | 476 / 820 |
| Verner 9 crossings, host rays | 255 | 4,944 | 2,296 / 4,496 | – |
| `V9Sched` crossings | 255 | 2,776 | 1,612 / 1,980 | 804 / 928 |
| `V9Sched` crossings, host rays | 255 | 2,104 | 1,616 / 2,120 | – |

`V9Sched` halves the stack and removes the `afoldl` call. Its stage arithmetic is the
package's bit for bit (§8), but with the RHS inlined it changes CPU bits in the program.

### 4.5 The camera ray (`camera_state`)

`camera_state` (`programs.jl:22`) is compiled once per kernel and contains `infinity_ray`
whatever the frame says (a run-time branch, `camera.jl:65`): the far-zone start (`far_zone.jl:91-104`,
three corrector passes with `_far_null`'s four Newton iterations, `far_zone.jl:54-68`) and
17 Verner 9 steps from `u0 = 1e-8` to `1/d = 1e-3` at ratio 2 (`far_zone.jl:110-114`,
`FAR_U0`, `FAR_RATIO` at `:25,27`). A kernel's register count is the maximum over its call
tree, so this function sets it for every light kernel:

| kernel | regs at `7be2569` (audit baseline) | regs at `a5923dd` | `camera_state` spill st/ld B |
|---|--:|--:|--:|
| `trace_kerr_f64` (first window, calls `camera_state`) | 172 | 255 | 648 / 1,080 |
| `trace_kerr_f64_resume` (no camera ray) | 168 | 162 | – |
| `shadow_kerr_f64` | 170 | 255 | 892 / 1,080 |
| `img_I_uniform_kerr_f64` | 240 | 255 | 908 / 1,088 |

255 registers allow 8 resident warps per SM on the A100 (65,536 / (256 × 32)); 162 allow 12.
The heavy bodies (plasma imaging, crossings, Monte Carlo) sit at 255 without the camera ray
too, so for them the camera ray costs stack and spill code, not occupancy.

### 4.6 The Monte Carlo step (`src/physics/MonteCarlo/track.jl:276-342`)

Per step: `march_control` (`track.jl:98-106`), `rk4_step` (`:108`), `sample` at the step's
end (`:306`), absorption and scattering rates (`opacity.jl:42-58,80-86`; the hot-cross table
221 × 81 doubles, 143 KB), and with scattering `step_events` (`:221-251`): a
Poisson-many loop with a second partial RK4 (`:230`), a second `sample` (`:232`), the
Compton sampler (`scatter.jl`, rejection loops) and an atomic `spawn!`
(`src/infra/Execution/children.jl:69-76`). `track!` is a device function of its own.

| kernel | regs | stack B | `track!` spill st/ld B |
|---|--:|--:|--:|
| `mc_sphere_thin_primary` (flat march) | 118 | 1,456 | 0 / 0 |
| `mc_torus_thin_primary` | 255 | 3,208 | 144 / 168 |
| `mc_torus_scatter_primary` | 255 | 6,624 | 1,936 / 3,492 |
| `mc_torus_scatter_layer` | 255 | 5,720 | 1,920 / 3,448 |
| `mc_sphere_scatter_primary` | 255 | 3,448 | 444 / 536 |

The thin torus march is at the cap without the Compton code; scattering multiplies its
spill code by 13 and doubles its stack.

## 5. Bytes of one plasma sample

### 5.1 Native blocks (`src/infra/Data/fluid_blocks.jl`)

- **Layout** (`fluid_blocks.jl:101-112`, doc `:80-99`): `data[8, ni+2, nj+2, nk+2, nmb]`
  `Float32`, feature fastest, one ghost layer per face (`ghost_cells.jl:187`), so one cell is
  32 B = one 32-byte sector, aligned. Geometry `x_left`, `dx`, `lo`, `hi` are `(3, nmb)`
  `Float64`.
- **Locate** (`:243-250`, `LeafTable`): three multiplies and one `Int32` read of the
  finest-block grid (4 B, one sector). A mesh without a leaf table is a `LevelHash` on the
  host and becomes `BlockScan` on a device (`:67`), which reads `lo` and `hi` of every block
  per sample (`:266-276`): 48 B × `nmb`.
- **Stencil** (`block_sample`, `:314-330`; `shell_weights`, `:299-302`): `x_left[:, mb]` and
  `dx[:, mb]` (2 × 24 B; 2–4 sectors), three divisions, then 8 corner cells × 8 primitives =
  64 `Float32` loads (`corner`, `:286-287`). The cells `i1, i2` are adjacent (64 B); the
  four pairs sit at strides `32 (ni+2)` and `32 (ni+2)(nj+2)` bytes: four 64-byte runs,
  8 sectors, a pair split across a 128-byte line when `i1 ≡ 3 (mod 4)`. Blend: `lerp8`
  (`src/physics/Numerics/stencils.jl:44-53`), along i, then j, then k, association fixed.
- **Per sample**: ≈ 308 B of useful data (256 fluid + 48 geometry + 4 leaf), 11–13 sectors.
  Every step samples, inside or outside the plasma; the metric and the decode run only
  where `rho > 0` (`protocol.jl:66`). A `FrameWindow` (`series.jl:220-226`) samples two
  frames: twice the bytes.
- **Torus 00999** (measured, `audit/snap_info.jl`): `LeafTable` locate, 344 blocks of 32³ cells, levels 0–5,
  `dx` 0.25–8 M; `data` 433 MB (ghosts add 20 %), leaf grid 128³ `Int32` = 8.4 MB, geometry
  16.5 KB. The leaf grid and the geometry fit the A100's 40 MB L2; the fluid does not.
- **Uniform grid** (`src/infra/Data/uniform_grid.jl:56-71`): the same 8-corner gather from
  `data[8, n, n, n]` without locate or geometry: 256 B, 8 sectors.

### 5.2 Locality across a warp

- **Imaging.** The pixel queue is `cost_order` (`src/infra/Execution/schedule.jl:128-151`,
  `prepare.jl:100`): 16 × 16 tiles, β fastest inside a tile (`schedule.jl:145-147`), so a warp
  of 32 consecutive items is a 2 × 16 pixel patch. At the torus deck's 768², fov 30 M, a
  pixel is 0.039 M against cells of 0.25–8 M: the 32 rays of a warp read the same few cells
  at the same step, a handful of sectors per warp-sample served by L1 broadcast.
- **Monte Carlo.** Primaries `g` of one zone are consecutive (`programs.jl:404-409`; zones in
  block and cell order, `src/infra/Data/tables.jl:103-143`), so a warp starts at one or a few
  cell centres; isotropic emission directions separate the photons by about `d/C` per step
  (`track.jl:105`), and within tens of steps every lane gathers its own 8 cells: up to
  32 × 8 = 256 sectors per warp-sample from a 433 MB array. Children are stored in arrival
  order through an atomic counter (`children.jl:60-76`), so a layer's warp mixes unrelated
  parents from its first step.

## 6. Why imaging is coherent across a warp and Monte Carlo photons are not

| | imaging (camera rays) | Monte Carlo photons |
|---|---|---|
| start | the camera plane, every lane at step 1 together | one zone per run of primaries, random direction and energy (`generate.jl`); children anywhere in the chunk |
| neighbours | impact parameters one pixel apart: same branches (`stop`, `rho > 0`, `valid`), same step sizes `h(r)` | independent directions: different cells, different branches from the first steps |
| lifetime | measured steps per ray in queue order: torus I 768² mean 895, max 2,339, warp step efficiency (Σ steps / 32 max) **0.982** (`audit/torus_steps.jl`); disks A/B 256²: production 0.971 / 0.962, precise 0.938 / 0.822 (`audit/v9_context.jl`) | recorded thin torus (stride 4, 1e6 primaries, before the plunge stop, `results/blp/mc_audit/speed/PLAN.md` §1.1): 55 % captured at 1,579 steps, 45 % escaped at 547, 0.08 % at the 20,000-step limit; a warp of 32 waits for its slowest (blocks of one warp, `programs.jl:436-440`): efficiency ≤ 0.55 from the class means alone (estimate; spread within a class lowers it) |
| control flow per step | identical across the patch except at the shadow edge and near the critical curve (where `cost_order` puts the tiles first) | `rho > 0` per lane; with scattering, `dtau_s > 0`, a Poisson-many event loop, rejection loops in the Compton sampler, atomics |
| memory per warp-sample | a handful of sectors | up to 256 sectors |
| registers | Stokes I body 255 (f32 lane 246) | thin 255, Compton 255 + 1.9/3.5 KB spill code |

The two marches cost the same per step on the CPU, so the gap is the device's: torus
imaging 5.28e8 ray-steps in 21.6 s on 16 threads (654 ns per step per thread, measured
steps over the recorded time) against the thin torus march's 4.6e7 steps/s on 32 threads
(696 ns per step per thread, `PLAN.md` §1.1); on one A100 the imaging march runs at
1.16e9 steps/s (5.28e8 steps in 0.457 s) and the Monte Carlo march at 1.2e8 (`PLAN.md`
§1.1): about 760 against 83 CPU threads' worth for the same arithmetic per step.

The crossing march is coherent in its steps but not in its crossings: the lane whose step
changes the sign of `z` runs `surface_crossing` (up to 33 Verner steps, `integrators.jl:202-212`)
while the other 31 wait, once per crossing time present in the warp.

## 7. Register-pressure hot spots, ranked

1. **Verner 9 stage storage** (`integrators.jl:74-91`): 11 live 8-vectors at the 14th RHS
   (176 registers before the RHS's own); 5.6 KB stack and 2.3/4.4 KB of spill code in the
   crossing program; a non-inlined `afoldl` per step (`:89`); four inlined Verner bodies
   per crossing kernel (≈ 1.2 MB of SASS). Also the precise Stokes I march: 255 registers,
   520/548 B body spills (`img_I_blocks_kerr_f64_v9_hostrays`), and `transport_step` for
   IQUV with 16-component stages (`integrators.jl:156-161`).
2. **The polarized RK4** (`march.jl:173-188` with `geodesic.jl:123-149`): three 16-double
   groups plus the Dual metric pair; 255 registers, 4.3 KB stack, blocks of 64.
3. **The Monte Carlo step body** (`track.jl:292-334` with `step_events`, `:221-251`): two RK4
   bodies, two sample/decode bodies and the Compton sampler in one function; thin at the cap,
   Compton 6.6 KB stack.
4. **`camera_state`** (`programs.jl:22`, `far_zone.jl:84-120`): the far-zone Verner march
   compiled into every camera-ray kernel lifts light kernels from 162–172 to 255 registers
   (§4.5) and adds 1.3–2.0 KB of spill code to every camera-ray kernel.
5. **The 8-corner gather** (`fluid_blocks.jl:325-329`): 64 loads that the scheduler may hoist
   together (64 registers in `Float32`, 128 once widened) before `lerp8`.
6. **`maps` in the crossing program** (`crossings.jl:72,95-97`): 15 doubles (30 registers)
   carried through the march.

## 8. Which changes keep the bits

Every floating-point operation of the march is IEEE with fixed rounding except the
flagged reductions of §3 (StaticArrays `dot` under `@simd`: `reassoc`, `contract`; StaticArrays
`muladd` chains: `contract`). LLVM reassociates and fuses those according to the code
around them, so a rewrite that is exact operation for operation can still move bits when it
changes the IR of the function that inlines the RHS. Measured on the CPU (`audit/`):

| variant | stage arithmetic | CPU bits against the package's Verner 9 |
|---|---|---|
| `V9Fold`: final sum (`:90`) as explicit binary adds | identical | bitwise, disks A and B at 160², precise (5.5e6 and 9.0e6 steps) |
| `V9Sched`: §4.4 schedule, RHS inlined | identical | differ: 1,060 of 76,800 `Ib` entries (A), 7,041 (B) |
| `V9Sched`, unit context: `erk_increment` alone, 22,405 Kerr states, inlined RHS | identical | bitwise |
| `V9SchedNI` against `V9NI`: both with the RHS behind one `@noinline` function, in `render_crossings` | identical | bitwise, A and B at 128² |

So the stage arithmetic of `V9Sched` is exact; the differences come from how the inlined
RHS copies are optimized inside the larger program body. The same mechanism explains the
recorded `V9Loop`/`V9Call` CPU differences (`results/blp/mc_audit/speed/gpu/README.md`,
Findings) and the host-only site scan (decode `muladd` chains).

Classes of candidate reductions:

**A. Outside the program body: bitwise by construction** (the compiled body is unchanged).

- A1. Register cap per kernel (`maxregs` on the launch, or launch bounds): ptxas only; the PTX
  rounding is fixed (`.rn` everywhere but the four leading products noted in §3). Trades
  occupancy against spills; the kernels at 255 hold 8 warps per SM.
- A2. Camera rays in their own launch (a ray program writing `Float64` states, the march
  reading them; recorded bitwise equal to the fused march, `docs/src/design/performance.md`
  §4): drops `camera_state` from the march kernel; for trace and shadow 255 → about 162
  registers (the resume kernel), for every kernel its 1.3–2.0 KB of spill code.
- A3. Item order and regrouping: queue order (`cost_order`), sorting Monte Carlo photons by
  cell and direction between launches of the same body, compaction of live photons; per
  item bitwise by the executor contract (`executor.jl` `foreach_item!` docstring) and the
  Philox ancestry streams with exact integer bins.
- A4. Workgroup size (`kernel.jl:12-19`).

**B. Inside the body, exact operation for operation: verify on the decks on both backends.**

- B1. `V9Sched` (§4.4): peak 11 → 7 live vectors during an RHS; stack 5,624 → 2,776 B,
  spill code 2,304/4,368 → 1,612/1,980 B, no `afoldl` call. Moves CPU bits as written (above).
- B2. The long stage sums written as binary adds (on the device the ten-term sum of stage 15,
  `integrators.jl:89`, the one `afoldl` call per Verner body; on the CPU at least one of
  stages 11–15, since `V9Fold` still calls `afoldl` there and `V9Sched` does not): removes the
  non-inlined call per step on both backends. The final-sum rewrite of the same kind
  (`V9Fold`) kept the CPU bits; the stage-sum rewrite alone is untested.
- B3. Crossing landings stored at their crossing instead of carried in `maps` (−30 registers of
  loop state; no arithmetic changes).
- B4. Deferred crossings: keep `(y, h, k)` at the sign change and solve them after the march or
  in a converged pass; one Verner body in the march loop instead of four, and no divergent
  Newton loop inside it. `surface_crossing` is a function of `(y, h)` alone.
- B5. Stage vectors in shared memory (`@localmem`): 9 × 64 B per thread, 36 KB per block of
  64; loads and stores only.
- B6. Blend the corner pairs as they arrive (the `lerp8` order already starts with
  `c000, c100`), so at most two cells and the partial blends are live.
- B7. A windowed Monte Carlo march (bounded step windows with the photon state in memory,
  re-sorted between windows, the pattern of `TraceProgram`, `programs.jl:128-178`): the per-step
  code is the same, the enclosing function is not.

**C. Through one rebaseline of the bits (needs the user's decision; the invariant pins
main's bits).**

- C1. Explicit-order sums in place of `dot` in the RHS (`kerr_schild.jl:97,99`, the binary's
  `_superposed_raise_and_norm`) and of the decode's StaticArrays products: no flagged
  reductions left, so every class-B change becomes bitwise by construction afterwards, and
  the CPU/GPU difference shrinks to the libdevice functions (`hypot`, `pow`, `exp`).
- C2. The RHS behind `@noinline` (one compiled copy per spacetime): `V9SchedNI ≡ V9NI` shows it
  makes stage-level changes context-free; recorded on the A100 as 1.80 s against 1.90 s on
  disk B (`V9Call`), with a small CPU bit change (`proto/verner_unit.jl`: 1,355 of 20,000 steps,
  ≤ 8.3e-15 relative).

## 9. What the profiler should settle (ncu on the A100)

- Precise march: local-memory traffic per step (`l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}`),
  stalls on instruction fetch (`smsp__pcsamp_warps_issue_stalled_no_instructions`) against
  long scoreboard, before and after B1/B2.
- Imaging against Monte Carlo: warp execution efficiency
  (`smsp__thread_inst_executed_per_inst_executed`), sectors per global load request
  (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / ..._requests`), L2 hit rate on the fluid
  array, achieved occupancy (8 warps per SM expected at 255 registers).
- Crossing divergence: the share of instructions inside `surface_crossing` with fewer than
  32 active lanes.

## 10. Tooling notes

- `bench/kernel_audit.jl` at `a5923dd`: the `mc_torus_*` cases fail (`grid_tables` and
  `hotcross_build` dispatch on `ThreadedExecutor`/`PoolExecutor`, not on its `AuditExecutor`;
  `march_audit2.jl` forwards them), and its baseline (`7be2569`) predates `infinity_ray`, so
  `--check` reports the §4.5 kernels as regressions. The crossing program is not among its cases.
- Scratch: `/data2/haiyangw/scratch/wavefront/plan/audit/` — `march_audit.jl` and
  `march_audit2.jl` (offline resources, Verner variants `V9Fold`, `V9Sched`), `v9_unit.jl`
  and `v9_context.jl` (CPU bit checks), `torus_steps.jl` (imaging step counts),
  `rhs_flags.jl` (flagged IR), `snap_info.jl` (torus mesh), PTX of the crossing and imaging
  kernels, SASS in `sass/`.
