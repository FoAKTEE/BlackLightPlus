# BlackLightPlus next-generation code design: principles

**Status:** draft for review. Nothing here is adopted until approved; the
architecture proposal follows approval.
**Scope:** the structure of `src/`, `ext/` and `test/`. The physics, the
numerical schemes and the validated results are inputs to this document, not
its subject.
**Relation to `docs/STANDARDS.md`:** that document governs the repository and
its documentation and stays in force. The two rules it would need to change
are listed in §6.

Each principle is a rule, the evidence in the current tree that motivates it,
and a check a reviewer can mark pass or fail.

---

## 1. Goal

The current code is correct, validated and fast. Its structure makes four
things expensive: finding where a concept lives, adding a combination of
existing capabilities, loading the GPU path, and reading a driver end to end.
The next generation keeps every physics function, numerical scheme and
determinism contract, and reorganises them so that

1. every concept has one definition in one module, and the module graph has
   one direction;
2. a spacetime, a matter model, a transfer mode, a ray source, a backend and a
   precision lane combine by argument, never by a new function;
3. the GPU path is ordinary package code: namespaced, precompiled, and run in
   CI on the KernelAbstractions CPU backend;
4. a driver reads as a short pipeline.

Non-goals: new physics; changing the reverse-accumulation transfer scheme, the
march termination rule or the fp64 default; any kernel register, stack or
spill growth beyond the `bench/kernel_audit.jl` tolerance.

## 2. What is preserved

These are design decisions of the current code that the next generation keeps
verbatim. They are the reason the rewrite is a reorganisation and not a
reimplementation.

- **The hot-path rule.** Everything reachable from a march is pure,
  allocation-free, `isbits` and `StaticArrays`-typed, so one implementation
  serves CPU threads, KernelAbstractions kernels and ForwardDiff/Enzyme
  (`CONTRIBUTING.md`, `docs/STANDARDS.md` §9).
- **One physics source.** The metric methods and AD-derived derivatives
  (`geodesics/metrics.jl`), the plasma decode and synchrotron fits
  (`radiation/coefficients.jl`), the tetrads, the segment operators
  (`radiation/transfer.jl`) and the two reverse marches
  (`radiation/reverse_march.jl`) are called by the CPU drivers and compiled
  in-kernel by every GPU driver.
- **The march abstraction.** `reverse_unpolarized_march` and
  `reverse_polarized_march`, parameterised by a march control (`KerrMarch`,
  `SpacetimeMarch`, `StoredPath`) and a sampler, with one termination rule
  (`march_stop`).
- **Spacetime as a type.** `metric`, `inverse_metric`, `raise_and_norm`,
  `lower_index`, `spacetime_control` dispatch on the spacetime; a new metric
  defines `metric(st, x)` and inherits the engine. The ADM grid's device twin
  (`DevADMParams`) runs the CPU bodies by duck typing.
- **Determinism.** Per-item outputs independent of chunking and device count;
  exact integer-limb accumulation for every reduction (`montecarlo/exact_sum.jl`);
  Monte Carlo streams keyed by genealogy (`montecarlo/rng.jl`).
- **The device pool and the per-item executor** (`gpu_pool.jl`: `DevicePool`,
  `march!`, `MarchPart`, `CameraRays`, `resident`).
- **The two-phase stored-trajectory path** (`trace_rays_sampled` +
  `StoredPath`) as the validation and adaptive-tracer route.
- **The test culture**: analytic solutions, cross-code references, bitwise
  device-invariance decks, the offline kernel audit.
- **`docs/STANDARDS.md`** in full except the two rules in §6.

## 3. Diagnosis of the current tree

Counts are over `src/` at commit `e364c7d` (13,831 lines).

| # | Finding | Evidence | Cost |
|---|---|---|---|
| D1 | The GPU path is not in the package. Nine files (3,694 lines, 27% of `src/`) are `include`d into the caller's module behind 12 `isdefined(@__MODULE__, …) \|\| include(…)` guards with a documented include order. | `gpu_*.jl`, `geodesics/gpu_tracer.jl`, `montecarlo/mc_gpu.jl`; `docs/src/guide/gpu.md` "Include order matters". | No precompilation, no namespace (prefixes `_gimg_`, `_gpol_`, `_gamr_`, `_gsp_`, `_famr_` stand in for modules), methods added to package functions from `Main` (`import BlackLightPlus: stream_blocks!`), every test and figure script repeats the include dance. |
| D2 | The stated reason for D1 does not hold. `CUDA` and `Enzyme` are hard dependencies in `Project.toml` `[deps]`; the README says CUDA "is not a package dependency". | `Project.toml`; `README.md` Installation. | The deployment rule protects nothing. Package extensions (`[weakdeps]`, Julia ≥ 1.9) would make CUDA optional in fact. |
| D3 | Private symbols cross module boundaries. 29 sites in `src/` import or call `_`-prefixed names of another module; tests, benchmarks and figures reach five more. | `_emission_prep`, `_emission_band`, `_sample_coefficients`, `_normal_observer`, `_prepared_scc`, `_MCM._mc_weight_at`, …; `BlackLightPlus._camera_up` in five test files. | The real internal interface is undeclared; "private" has no meaning; refactors break callers that the convention says cannot exist. |
| D4 | Capability axes are encoded in function names and kernel copies. 29 entry points in the `render`/`march`/`trace`/`mc_spectrum` family and 21 kernels enumerate {CPU, GPU} × {I, IQUV} × {uniform, AMR leaf, AMR scan, stored path} × {fast, slow light} × {Kerr, generic spacetime} × {frame, ray list, AMR block}. | `render_gpu_amr_polarized_slow`, `_gpol_amr_kernel_sl!`, …; `results/blp/multigpu/design.md` §4.1 "the combinatorics are the duplication". | A new combination is a new function with a copied prologue; readers cannot tell which differences are physics and which are plumbing. |
| D5 | Driver prologues are copied. The camera is rebuilt from `cfg` at 13 sites, the horizon radius inlined as `1 + sqrt(1 - a * a)` at 18, the escape radius as `1.1 * cfg.camera.distance` at 14. | `grep` over `src/`. | Derived quantities have no owner; a change to one convention is a 14-file edit. |
| D6 | Four parameter representations for one physics configuration: `PlasmaConfig` → mutable 90-field `SimCoeffConfig` → `coefficient_params` NamedTuple → the march `gp` NamedTuple merged with `nu_obs`, `L_unit`, `cu1..cu4`, `intensity_only`. The march reads 38 distinct fields of `gp` by duck typing; no single place lists them. | `radiation/coefficients.jl`, `radiation/integrator.jl` `_march_params`, `gpu_image.jl` `_gimg_params`, `gpu_pol.jl` `_gpol_params`, `render.jl` `_scc_config`/`_prepared_scc`, `montecarlo/grmhd_model.jl` constructor. | The kernel parameter contract is implicit; `SimCoeffConfig` is mutable only so `precompute_*!` can fill derived constants; `use_p` is an `Int` standing for a `Bool`. |
| D7 | Two matter interfaces. Snapshots return code-unit primitives `(rho, pgas, uu, bb, inside)`; analytic sources return cgs `(n_e, theta_e, B, ucon, …, inside)`; the marches take a third layout, `SVector{8}` `(rho, u1, u2, u3, eint, b1, b2, b3)`. A figure script hand-inverts the plasma decode to drive the march from a torus. | `simulation/athenak_reader.jl`, `sources/sources.jl`, `radiation/integrator.jl` `_stored_sample`, `figures/fig_examples.jl` `source_sampler`. | Analytic sources cannot be imaged through the public drivers; `pgas_from_eint` is applied and then inverted in the adapter. |
| D8 | Samplers are reimplemented per backend and per representation: four trilinear interpolants, two slow-light blends plus the CPU one, two vacuum sentinels. | `athenak_reader.jl`, `athenak_z4c.jl`, `gpu_image.jl` `_gimg_sample`, `gpu_amr.jl` `_gamr_lerp`; `_gimg_sample_t`, `_gamr_sample_t`, `sample_primitives_t`; `Sources._vacuum`, `MonteCarlo._vacuum`. | 37 comments of the form "keep in sync", "device twin", "byte-for-byte" guard the copies by hand. |
| D9 | Module responsibilities leak. `Geodesics` holds the AART image-plane setup (`lensing_bands.jl`) and an intensity seam that takes a source callable to avoid depending on `Radiation`; `Radiation` holds a disk *source* (`equatorial_source.jl`); `MonteCarlo` holds the generic exact-sum and RNG utilities; `Simulation` depends on `Geodesics` only to build an `ADMGridSpacetime`; `Outputs` holds interferometry analysis; the imaging drivers live at root scope because "cross-module composition happens here". | module files; `render.jl` header. | The dependency graph has back edges resolved by callbacks and re-exports (the Kerr orbit primitives are re-exported by `Radiation` and `Sources`). |
| D10 | Positional tuples and magic indices. Seven-tuples returned by position (`_emission_band`, `_mc_march_scatter_core`); the ray state `y[1..4]`/`y[5..8]` unpacked at 17 sites; functions with 10 to 20 positional arguments (`eight_coefficients`, `_mc_grmhd_loop!`). | `geodesics/analytic_kerr.jl`, `montecarlo/mc_common.jl`, `montecarlo/mc_cpu.jl`. | A reader must hold the layout in their head; a reordered field is a silent bug. |
| D11 | Comments narrate provenance and measurements inside the code: heritage ("blacklight `X.cpp`"), deviations, measured errors, history ("inherited weakness, fixed here"). File heads run 10 to 20 lines. | `geodesics/integrator.jl` lines 343–362, `analytic_kerr.jl` lines 378–385, `transfer.jl` docstring of `stokes_segment`. | The contract is buried under the justification; the same facts are repeated in `method.md`, violating STANDARDS §3 "one fact, one home". |
| D12 | Precision and AD lanes are threaded by hand. Every driver converts `(a, r_h, C, escape, R_em)` to the lane type at the call site; the transfer lane carries a second spin `gp.bh_a`. | `gpu_image.jl` `f32()`, `gpu_pol.jl`, `gpu_amr.jl` `_amr_mixed!`, `reverse_march.jl` `transfer_spacetime`. | The two-lane contract is enforced only by "a stray Float64 is a MethodError". |
| D13 | 23 `Threads.@threads` sites hand-chunk CPU work; the CPU drivers have no executor analogue of `march!`. | `render.jl`, `amr.jl`, `mc_cpu.jl`, `emission_tables.jl`, … | CPU and GPU drivers diverge although the per-ray function is the same. |
| D14 | The public API is the union of every test's needs: 183 exported names, including `philox4x32`, `boostcross`, `xsum_add!`. `test/runtests.jl` includes files conditionally on *source* file existence; a dozen newer test files are not included at all. | `src/BlackLightPlus.jl` export block; `test/runtests.jl`; `CHANGELOG.md` [Unreleased] Tests. | Semantic versioning covers internals; the suite's coverage is unknown from its entry point. |
| D15 | Global caches live in physics files: `_CURVE_LAST`, `_COST_ORDER_LAST` (`gpu_image.jl`), `_MC_TABLE_BUILDS`, `_CAMRAYS_ID`; `AthenaKSnapshot` carries a mutable `LeafCache`. | named files. | State and locking policy is scattered across layers. |

## 4. Principles

### A. Structure

**P1. Layered, one-directional module graph.**
*Rule.* The package is a stack of layers; a module imports only from layers
below it. The layers, bottom up: *Core* (numeric utilities: exact sums,
counter-based RNG, interpolation stencils, physical constants, units),
*Spacetime* (metrics, geodesic engine, cameras, the exact-Kerr tracer and its
orbit primitives), *Matter* (the plasma-state protocol and its providers:
AthenaK snapshots, snapshot series, analytic sources, emission zones),
*Transfer* (coefficients, tetrads, segment operators, the imaging marches, the
Monte Carlo transport), *Execution* (executors: native threads, the
KernelAbstractions device pool; residency; ray sources), *Drivers* (run
preparation and the product entry points), *Analysis* (image-plane refinement
indicators, interferometry, writers). No file at root scope composes modules.
*Why.* D9. Back edges are today resolved by callbacks (`equatorial_intensity`'s
`source`), re-exports and root-scope files.
*Check.* `BLP/test/structure.jl`, "layering": every `using ..X` or `import ..X`
of a module names a module before it in `Numerics, Spacetime, Matter, Transfer,
MonteCarlo, Data, Execution, Plans, Analysis, Products`; no physics module
imports `KernelAbstractions`, `HDF5`, `CUDA` or `Base.Threads`; no physics or
pipeline file contains `Threads.@threads`.

**P2. One package; optional backends are extensions.**
*Rule.* All kernels, the device pool and the executors are package code
written against KernelAbstractions, which is already a hard dependency. Code
that needs `CUDA` (device activation, pinned host memory, NVML probing, stack
limits, `unsafe_free!`) lives in `ext/BlackLightPlusCUDAExt.jl`; `CUDA` and
`Enzyme` move to `[weakdeps]`/`[extras]`. There are no `include` guards and no
include-order contract anywhere in `src/`, `test/`, `bench/` or `figures/`.
*Why.* D1, D2. The kernels already run on the KA CPU backend (`HostDevice`),
which is how the device tests run without a GPU; nothing but the CUDA-specific
glue requires CUDA at load time.
*Check.* `structure.jl`, "one kernel, no device files, no include guards, no
CUDA in src": no `isdefined(@__MODULE__` and no `gpu_*.jl` file in `BLP/src`, no
`using CUDA`/`import CUDA` and no `CUDA.` call there, and `BLP/ext` loads
`CUDA`; the device tests run on `HostDevice`s.

**P3. Declared interfaces; private stays private.**
*Rule.* Each module declares its interface by `export`. Another module may use
only exported names. A `_`-prefixed name is never referenced outside its
module. What the top-level package exports is the *public* API (P16); what a
submodule exports is an *internal interface*, documented, stable within a
minor version, and the only thing tests may import from it.
*Why.* D3.
*Check.* `structure.jl`, "private names stay private": no `Module._name` between
the ten modules and no `using ..X: _name` in `BLP/src`, no `BLP.X._name` in
`BLP/test` or `BLP/test/oracle`; "eponymous files own includes and exports": a
module's `include(` and `export` lines are in its eponymous file only; "public
API": every name a submodule exports is defined and has a docstring.

### B. Composition

**P4. Axes are arguments, never names.**
*Rule.* The imaging axes are: spacetime (a `Spacetime` value), matter (a
`PlasmaProvider` value, P5), transfer mode (`StokesI()` or `IQUV()`), ray
source (camera frame, coordinate list, refinement block), executor (P8),
lanes (P7) and time dependence (a property of the provider). There is one
image entry point and one spectrum entry point per product; a new combination
of existing axes requires no new function and no new kernel. Kernels exist
only for genuinely distinct per-item programs: the two transfer marches, the
trajectory tracer, the shadow tracer, the analytic tracer, the Monte Carlo
primary and layer kernels, the camera-ray kernel. The sampler and the
spacetime enter a kernel as arguments.
*Why.* D4, D5. The current marches already take the sampler and the control as
arguments; the drivers and kernels do not.
*Check.* `structure.jl`: exactly one `@kernel` in `BLP/src` ("one kernel, …");
at most 10 exported names in the `render`/`trace`/`shadow`/`mc_spectrum`/
`sensitivity`/`variants`/`photon_ring_layers` family ("public API").

**P5. One matter protocol.**
*Rule.* A `PlasmaProvider` answers `plasma(provider, x)` for a four-position
`x` with one typed record: the code-unit primitives `(rho, uu, eint, bb)` and
a validity flag, or `nothing`-equivalent vacuum with a single representation.
Snapshots, snapshot series (time blending), analytic sources and the Monte
Carlo zone list are providers; the cgs decode (`plasma_decode`) is applied by
the transfer layer, once. A provider's device twin is the same struct with
device arrays (`Adapt.@adapt_structure`), sampling through the same body. One
trilinear interpolant, one time blend, one vacuum sentinel.
*Why.* D7, D8. `DevADMParams` already proves the pattern for spacetimes.
*Check.* `structure.jl`, "one concept, one home": the trilinear blend
(`c100 - c000`) in one file and one `vacuum` definition in `BLP/src`; no "keep in
sync", "twin of", "byte-for-byte" or "ported from" there.

**P6. Prepare once.**
*Rule.* `RunConfig` is declarative user input. `prepare(cfg; spacetime,
matter, …)` returns an immutable plan that owns everything derived: the
spacetime, the camera and its frame, the horizon and escape radii, the
emission-confinement radius, the units, the frequency, the transfer parameters
with their precomputed nonthermal constants, and the lanes. Drivers, kernels
and executors consume the plan. No derived scalar is computed in a driver.
*Why.* D5, D6.
*Check.* `structure.jl`, "one concept, one home": `1 + sqrt(1 - a` only in
`kerr_schild.jl`, `kerr_orbits.jl` and `analytic_kerr.jl`, and no `sqrt(1 - a`
in `BLP/src/pipeline`; `1.1 * ` in the pipeline only in `config.jl`; at most one
`PlaneParallelCamera(` in the pipeline.

**P7. Typed parameter records and explicit lanes.**
*Rule.* The march parameter set is an immutable, documented struct
`TransferParams{T}` (units, electron model, cuts, frequency, length unit,
camera vertical) with the nonthermal populations as optional typed
sub-records (`Nothing` when absent, so selection is at compile time, as
`hasfield` is today). `SimCoeffConfig` and `coefficient_params` disappear; the
host precompute returns the record instead of mutating one. The AD lane is
`TransferParams{Dual}`; the K-variant ladder is an `NTuple{K}` of records. The
geodesic lane type and the transfer lane type are set once in the plan
(`Lanes{Tg,Tt}`) and flow by dispatch; no driver converts scalars by hand.
`Bool` fields are `Bool`.
*Why.* D6, D12.
*Check.* No `NamedTuple` crosses a kernel boundary as a parameter set; no
`Float32(` call in a driver; one definition of the lane rule.

**P8. One executor per backend; the backend is a switch; a driver is a pipeline.**
*Rule.* An executor runs a per-item function over an index set and writes
per-item outputs: `ThreadedExecutor` (native threads, host arrays) and
`PoolExecutor` (the device pool over KA backends, CPU or CUDA). Reductions go
through exact accumulators merged in item order. Where a product runs is one
keyword, `backend`, accepted by every product, with a session default
(`backend!`) and an environment override (`BLP_BACKEND`); a plan never records
a backend; `:auto` logs its choice once, and an explicit choice that cannot be
honoured is an error, never a silent downgrade. Every driver has the same
shape: resolve backend → ray source → march under executor → assemble product.
The Monte Carlo drivers follow the same shape with generate/track/record as the
per-item function.
*Why.* D13, D4. `march!` and `MarchPart` are the executor for one backend
already; the CPU side hand-rolls 23 loops; today the backend is chosen by
calling a differently named function after a differently ordered set of
`include`s.
*Check.* `structure.jl`, "threads only in the executor and the Data loaders
and builders": the files with `Threads.@threads` are `infra/Execution/executor.jl`
and files under `infra/Data/`, at most five; "layering": none in physics or
pipeline. Every product accepts `backend` (`BLP/test/pipeline/products_backend.jl`).

### C. Readability

**P9. One concept, one name, one home.**
*Rule.* Each concept is defined once, in the module that owns it, under one
name: horizon radius, escape radius, vacuum, the trilinear stencil, the
Stokes-plane reshape, the camera vertical, the Kerr orbit radii. Modules
replace prefix pseudo-namespaces. Re-exports are allowed only upward through
the layer stack, never sideways.
*Why.* D5, D8, D9; the Kerr orbit primitives are exported by `Geodesics`, re-exported by `Radiation` and `Sources`.
*Check.* `structure.jl`, "one concept, one home": the greps of P5, P6 and P10.

**P10. Records over positional tuples; accessors over indices.**
*Rule.* A value with three or more fields crossing a function boundary is a
NamedTuple with documented field names or a small struct, never a positional
tuple. The eight-component ray state has `position(y)`, `momentum(y)` and a
constructor; callers never index `y[5]`. Multi-flag results (`stop`, `h`, `r`)
are named.
*Why.* D10.
*Check.* `structure.jl`, "one concept, one home": the
`SVector(y[1], y[2], y[3], y[4])` pattern appears only in `Numerics/raystate.jl`.

**P11. Comments carry contracts; documents carry derivations.**
*Rule.* A docstring states the contract: inputs, units, conventions, outputs,
failure behaviour, and one reference. A comment states an invariant or a
non-obvious reason. Heritage ("after blacklight `X.cpp`"), deviations from
reference codes, measured errors and history live in `docs/src/method.md`,
`NOTICE` and `CHANGELOG.md`, and the code points to them by section. A file
head is at most ten lines: role, layer, counterpart, reference.
*Why.* D11; STANDARDS §3 "one fact, one home".
*Check.* `structure.jl`: no "keep in sync", "twin of", "byte-for-byte" or
"ported from" in `BLP/src` ("one concept, one home"), and no attribution line in
`BLP/src`, `BLP/ext`, `BLP/test`, `BLP/README.md` or `BLP/Project.toml` ("no
attribution or process prose").

**P12. Size and arity budgets.**
*Rule.* A function takes at most six positional arguments; beyond that the
arguments are a record. A file is at most ~500 lines and one concern. A
kernel body is one call to a march plus the output write. A product entry
point is at most ~40 lines.
*Why.* D10, D4; `_mc_grmhd_loop!` takes 20 positional arguments,
`gpu_image.jl` and `gpu_amr.jl` each mix four concerns.
*Check.* `structure.jl`, "file budgets": every file of `BLP/src` is at most 600
lines, and no stray `test_*.jl` file sits in `BLP/test`.

### D. Contracts

**P13. The hot-path rule, enforced.**
*Rule.* Unchanged from today (§2), now tested: each march is compiled with a
`Float64`, a `Float32` geodesic lane and a `Dual` transfer lane on the KA CPU
backend, and `@allocated` of one march call is zero; the offline kernel audit
runs in CI against its baseline.
*Check.* The test exists and passes; the audit gate is a CI step.

**P14. Determinism per entry point.**
*Rule.* Every product entry point states its determinism contract in its
docstring, one of: *bitwise per item* (independent of executor, chunking,
device count and queue order), *bitwise reduction* (exact accumulators merged
in canonical order), or *tolerance* (with the tolerance and the reference).
The existing device-invariance decks become the tests of these contracts.
*Why.* The contracts exist today but are stated in prose in `method.md` and
in scattered comments.
*Check.* Each entry point's docstring has a "Determinism" line; a test
references each.

**P15. Continuity during the migration.**
*Rule.* The migration proceeds module by module with the suite green at every
step. Where the arithmetic order is unchanged the output stays bitwise
identical to the current code on the regression decks; where a unification
changes arithmetic (the uniform-grid interpolant, the sampler layout) the
change is accepted by physics tolerance against the CPU reference, documented
in the changelog, and the pinned outputs are re-baselined once. No kernel's
registers, stack or spill may grow beyond the audit tolerance.
*Why.* `results/blp/multigpu/design.md` §5.6 set this policy for the previous
consolidation; it worked.
*Check.* Each migration commit cites the deck set it was checked against and
states "bitwise" or the tolerance.

### E. Surface, state, tests

**P16. A small, product-shaped public API.**
*Rule.* The top-level package exports: the configuration types, the spacetime
types, the plasma providers and sources, the transfer modes, the executors
and device selection, the product entry points (`render`, `render_amr`,
`render_analytic`, `photon_ring_layers`, `mc_spectrum`), the result types, and
the analysis and writer functions: 54 names, the gate's ceiling. Everything else
is a submodule interface (P3). Semantic versioning applies to the public API.
*Why.* D14.
*Check.* `structure.jl`, "public API": `names(BLP)` holds at most 54 names, each
defined, at most 10 of them in the product family of P4; every name a submodule
exports is defined and documented.

**P17. Global state confined to the execution layer.**
*Rule.* Mutable global state exists only in *Execution*: the pool registry,
device residency, staging buffers. It is keyed by value where a value exists
(a configuration key) and by identity with a weak reference otherwise, has a
documented lifetime, and is behind one lock discipline. Physics and driver
modules hold no caches; what they memoise (the critical curve, the cost order,
leaf tables) is computed in `prepare` and stored in the plan or the provider.
*Why.* D15.
*Check.* `grep -rn "const _[A-Z_]* = Ref\|ReentrantLock()\|Threads.Atomic" src`
hits only the execution module and provider constructors.

**P18. Tests exercise contracts through declared interfaces.**
*Rule.* `test/runtests.jl` includes every test file unconditionally; a test
skips, and reports the skip, only on a missing external resource (device,
data file). Tests import public names or submodule exports, never `_` names.
Each principle's *Check* above that is mechanical (P1, P3, P6, P9, P10, P12,
P16, P17) is a test in `BLP/test/structure.jl`, so the structure cannot
regress silently.
*Why.* D3, D14.
*Check.* `BLP/test/runtests.jl` includes `structure.jl` and every file under
`physics/`, `infra/` and `pipeline/` without condition; `structure.jl` keeps
leading-underscore names out of `BLP/test` and `BLP/test/oracle` and stray
`test_*.jl` files out of `BLP/test`.

## 5. Gates

The numbers a reviewer reads off the tree to confirm the principles hold: the
reference package `src/` at `e364c7d`, the package `BLP/`, and the gate
`BLP/test/structure.jl` holds `BLP/` to.

| Metric | `src/` | `BLP/` | Gate |
|---|---|---|---|
| `_`-prefixed names used across a module boundary | 29 | 0 | none |
| Private names reached from the tests | 5 | 0 | none in `BLP/test` and `BLP/test/oracle` |
| `isdefined(@__MODULE__ …) \|\| include` guards | 12 | 0 | none |
| `include` sites outside the eponymous module files | 11 | 0 | none, nor `export` |
| Kernels (`@kernel`) | 21 | 1 | exactly 1 |
| Entry points in the `render`/`trace`/`shadow`/`mc_spectrum`/`sensitivity`/`variants`/`photon_ring_layers` family | 29 | 9 | ≤ 10 exported |
| Top-level exported names | 183 | 54 | ≤ 54, each defined |
| Files over 600 lines | 5 | 0 (largest 547) | none |
| Camera constructed from the configuration | 13 sites | 1 (`prepare`) | ≤ 1 `PlaneParallelCamera(` in the pipeline |
| Horizon radius inlined | 18 sites | 5, all in `kerr_schild.jl`, `kerr_orbits.jl`, `analytic_kerr.jl`; 0 in the pipeline | `1 + sqrt(1 - a` only in those three files |
| Escape radius inlined | 14 sites | 1 (`prepare`) | no `1.1 * ` in the pipeline outside `config.jl` |
| Trilinear interpolant implementations | 4 | 1 (`lerp8`) | `c100 - c000` in one file |
| Vacuum sentinel definitions | 2 | 1 | one |
| "keep in sync" / "twin of" / "byte-for-byte" comments | 37 | 0 | none, nor "ported from" |
| `Threads.@threads` sites (files) | 23 (10) | 10 (5) | only in `Execution/executor.jl` and `Data`, ≤ 5 files |
| `Threads.@threads` sites outside the executor and table builders | 23 | 4 (the `Data` loaders) | — |
| `SVector(y[1], y[2], y[3], y[4])` sites | 17 | 1 (`raystate.jl`) | only in `raystate.jl` |
| Parameter representations between `PlasmaConfig` and the march | 4 | 1 (`TransferParams`) | — |
| Test files not included by `runtests.jl` | ~13 | 0 | no stray `test_*.jl` |

## 6. Decisions requested

Each changes the architecture proposal materially. A recommendation is given
for each.

1. **GPU code into the package (P2).** Requires amending `STANDARDS.md` §1
   (standalone `gpu_*.jl` drivers) and §5 (the GPU page's include-order note),
   and moving `CUDA` to `[weakdeps]` with an `ext/`. *Recommend yes*: it is the
   single largest readability and modularity gain, and the current rule does
   not deliver the CUDA-free installation it claims.
2. **Public API size (P16).** Shrink to the product-shaped set and have tests
   import submodule interfaces explicitly, or keep exporting what tests use.
   *Recommend shrink*; the `0.x` version contract allows it once.
3. **Default CPU imaging path (P8).** Keep the two-phase stored-trajectory
   path as the default `render` on the CPU, or make the fused reverse march
   under the threaded executor the default and keep the stored path for the
   adaptive tracer and diagnostics. *Recommend fused default*: it removes the
   CPU/GPU driver split; the stored path stays available and tested.
4. **Numerical continuity (P15).** Which outputs must stay bitwise identical
   to the current tree, beyond the device-invariance decks. *Recommend*:
   bitwise for every path whose arithmetic order is unchanged (Kerr AMR
   imaging, polarized marches, Monte Carlo spectra); documented tolerance for
   the uniform-grid resample path and for any sampler-layout unification.
5. **Configuration redesign (P6).** `RunConfig` today mixes a spin (`bh_a`),
   a `spacetime` keyword on `render`, a `tracer` symbol that changes what
   `render` ignores, and a `polarized` flag duplicated as a keyword.
   *Recommend* a redesigned config (spacetime and matter as values, transfer
   mode as a value) with a one-release deprecation shim that maps the current
   fields.
6. **Diagnostics in the package.** `render_with_powerlaw`,
   `polunpol_consistency`, `stiffness_telemetry`, the AMR probe kernels and
   `trace_shadow_gpu` are diagnostics. *Recommend* moving them to `bench/`
   and `test/` as scripts over the public API, or into a `Diagnostics`
   submodule that is not exported.
7. **Module names.** Keep the current submodule names where the contents
   survive (`Geodesics`, `Radiation`, `MonteCarlo`, `Sources`), or rename to
   the layer vocabulary of P1 (`Spacetime`, `Matter`, `Transfer`, `Execution`,
   `Drivers`, `Analysis`). *Recommend* the layer vocabulary: the current names
   no longer describe the contents (`Radiation` holds a source; `Simulation`
   holds a spacetime builder).
8. **Migration mode.** Incremental in place, module by module, with the
   suite green at every commit; or a parallel tree swapped at the end.
   *Recommend incremental*: the gates in §5 are checkable at every step and
   the paper's figure scripts keep working throughout.
9. **Rewrite budget for comments (P11).** Move the heritage and measurement
   prose into `method.md`/`NOTICE` as part of the migration, or leave the
   prose and only enforce the rule on new code. *Recommend* moving it in the
   same commit that moves each file, so the "one fact, one home" rule holds
   at the end.
