# Host–device bit portability in BLP: compiler map

Toolchain as installed for the GPU environment `/data2/haiyangw/scratch/wavefront/env`
(worktree `blp_wavefront` at a5923dd): Julia 1.12.6, CUDA.jl 6.2.2 (CUDACore 6.2.2),
GPUCompiler 1.23.0, LLVM.jl 9.13.2, KernelAbstractions 0.9.43, StaticArrays 1.9.22,
ForwardDiff 1.4.6, NVPTX_LLVM_Backend_jll 22.1.7 (external `llc`), CUDA_Compiler_jll
0.4.4 (`ptxas` 13.3.33). Host CPU: Xeon 8360Y (icelake-server, FMA, AVX-512).

Every number below was measured on the host without a GPU (device code compiled to PTX
and SASS by CUDA.jl's own pipeline, inspected, not executed) unless marked otherwise.
Scripts and outputs: `/data2/haiyangw/scratch/wavefront/plan/exp/` (index in §8).

## 1. Summary

- Device arithmetic is IEEE-exact wherever the source fixes the rounding: `+ - * /`,
  `sqrt`, explicit `fma`, `min`/`max`, conversions, `floor`/`trunc` compile to
  `.rn` instructions that neither LLVM nor `ptxas` contracts. A computation written only
  with these is bitwise portable by construction.
- Device bits leave the host's in three ways: (a) CUDA.jl replaces Base math with
  libdevice (`exp`, `log`, `hypot`, `pow`, trig, `cbrt`, …); (b) CUDA.jl replaces
  `Base.muladd(::Float64,…)` with `llvm.fmuladd`, which NVPTX always fuses, while the
  host fuses a `muladd` only where x86 instruction selection sees the pair; (c) `@simd`
  reductions (StaticArrays `dot`, `sum`, `norm`) carry `reassoc contract`, which the
  device fuses and the host often does not.
- The host side is the uncontrolled one. The same decode gives different bits in two
  host contexts (§3.1), and `Base.hypot` gives different bits depending on the package
  environment, because 206 of the 209–218 loaded dependency pkgimages in this depot
  were compiled for the `generic` CPU target (§3.2).
- No compiler option reproduces the host on the device. CUDA.jl exposes no contraction
  control, KernelAbstractions passes only `always_inline` and `maxthreads`, and `llc` and
  `ptxas` run with fixed flags (§4). Disabling contraction on the device would still not
  match a host that contracts some pairs and not others.
- The clean route is source-level: the code that must agree (site scan, decode, Kerr–Schild
  metric) uses only explicitly rounded operations and BLP-owned ports of the library
  functions it needs, checked by a host-only portability gate that inspects the
  device and host IR (§5). That removes all compiler freedom on both sides. It cannot keep
  main's bits exactly: emulating the device semantics on the host changes 661 of 104,776
  site-scan zones (0.63 %) by at most 3 ulp in `theta_e` and `B` (§3.3). This needs a
  one-time re-baseline, which is the user's decision.
- Bitwise CPU = GPU images are feasible by the same construction, but the image path holds
  about 60 StaticArrays reductions, about 50 matrix products and about 15 libdevice-replaced
  functions, so the cost is weeks (§6).

## 2. Where device code can differ in bits

| source construct | host (x86, native) | device (CUDA.jl 6.2, sm_80) | portable? |
|---|---|---|---|
| `a*b + c`, no flags | `vmulsd`+`vaddsd` | `mul.rn.f64` + `add.rn.f64`; `ptxas` leaves `.rn` alone (SASS `DMUL`, `DADD`) | yes |
| `fma(a,b,c)` | `vfmadd` (software fma without hardware FMA, still exact) | `fma.rn.f64` | yes |
| `/`, `sqrt`, `inv` | IEEE | `div.rn.f64`, `sqrt.rn.f64` (the fast lowerings act only on `afn` ops) | yes |
| `min`/`max` (Float64) | `llvm.minimum`/`maximum` | the same intrinsics (CUDACore `math.jl` overrides, sm_80 `min.NaN`) | yes |
| `muladd(a,b,c)` | `fmul contract`+`fadd contract`; fused only for a single-use scalar or same-width vector pair in one basic block | CUDA override to `llvm.fmuladd` (`device/intrinsics/math.jl:519`) → always `fma.rn` | **no** |
| StaticArrays `M*v`, `M*N` | `muladd` chains (`matrix_multiply_add.jl:428`); fusion depends on SLP vectorization | all fused | **no** (context-dependent) |
| StaticArrays `dot`, `sum`-like | `@simd` loop (`linalg.jl:273`): `fmul contract`, `fadd reassoc contract`; SLP vectorizes the products and leaves the chain unfused | sequential `fma.rn` chain from `0.0` | **no** |
| contract pair split across basic blocks | never fused (x86 fuses within a DAG) | `llc` leaves `mul.f64`/`add.f64` without `.rn`; `ptxas --fmad=true` (default) fuses them into `DFMA` | **no** |
| `exp log log2 log10 log1p expm1 exp2 exp10 sin cos tan sincos sinpi cospi asin acos atan atan(y,x) sinh cosh tanh asinh acosh atanh cbrt hypot rem ldexp` (Float64) | Julia's Base algorithms (sysimage or pkgimage code) | libdevice `__nv_*` (`math.jl:22–509`) | **no** (except `ldexp` away from underflow) |
| `x^y` (Float64^Float64) | Julia's `pow` | `__nv_pow` (`math.jl:368`) | **no** |
| `x^n`, runtime `n::Integer` | Base `power_by_squaring` | `n ∈ {-1,0,1,2,3}` exact, else `__nv_pow` (`math.jl:412`) | **no** for `|n| > 3` |
| literal `x^2`, `x^3` | `literal_pow` → `x*x`, `x*x*x` | same | yes |
| constant expressions with `^`, e.g. `2.0^(11.0/12.0)` (`synchrotron.jl`) | folded at compile time with Julia's `pow` | `__nv_pow` at run time (an `llvmcall` does not fold) | **no** |
| `@fastmath`, `fastmath=true` | host fast-math | `afn` + `contract` + `reassoc` everywhere, approximate f64 div/sqrt/rsqrt (`GPUCompiler ptx.jl:278–305`) | **no** (none in BLP) |
| Float32 denormals | IEEE | IEEE unless `fastmath` (FTZ only via `denormal-fp-math-f32`); f64 never flushes | yes |

How each row was established:

- `llc` row tests (`exp/llc/t.ll`, `k.ll`, `xbb.ll`, built with the bundled `llc -O2 -mcpu=sm_80`
  and `ptxas`, disassembled with `cuobjdump -sass`):

  | IR | default `llc` | `-nvptx-fma-level=0` | `=1`/`=2` or `-fp-contract=fast` | `-fp-contract=off` |
  |---|---|---|---|---|
  | `fmul`+`fadd`, no flags | `mul.rn`+`add.rn` | same | `fma.rn` | `mul.rn`+`add.rn` |
  | `fmul contract`+`fadd contract` | `fma.rn` | `fma.rn` | `fma.rn` | `fma.rn` |
  | the same `fmul contract` with two uses | both fused (NVPTX fuses aggressively) | both fused | both fused | both fused |
  | `llvm.fmuladd` | `fma.rn` | `fma.rn` | `fma.rn` | `mul.rn`+`add.rn` |
  | `fadd reassoc contract` chain over plain `fmul` | `mul.rn` + `add.f64` (no `.rn`) | same | fused chain | same as default |
  | `"unsafe-fp-math"="true"` | `mul.rn`+`add.rn` (LLVM 22 ignores it) | | | |

  In LLVM 22 the `.rn` suffix is per instruction: an op without `contract` always gets
  `.rn`; global fusion requires an explicit `-nvptx-fma-level` or `-fp-contract=fast`, neither
  of which GPUCompiler passes. `ptxas` did not fuse mixed pairs (`mul.f64` with `add.rn.f64`,
  `mul.rn.f64` with `add.f64`). It did fuse a cross-block pair of two non-`.rn` ops (`xbb`:
  two `DFMA`), and `--fmad=false` stopped it.
- The device evidence of a753d7d agrees: `sF`, a Hermite polynomial of plain `*`/`+`, gave 0
  mismatches against the host in 1e7 plasma states on the A100
  (`results/blp/mc_audit/speed/gpu/proto/gpu_check_3777.log`). A device that contracted
  plain ops could not give that.
- Census of the same kernel bodies on both sides (`exp/census.jl`; host `code_native`,
  device PTX from `CUDACore._compiler_config(nothing; arch=v"8.0")`; files in `exp/host/`,
  `exp/ptx/`):

  | body | host FP instructions | device PTX | same rounding? |
  |---|---|---|---|
  | `SMatrix{4,4}*SVector{4}` | 3 `vfmadd231pd ymm` + 1 `vmulpd ymm` (all 12 fused) | 12 `fma.rn` + 4 `mul.rn` | yes |
  | `dot(::SVector{4},::SVector{4})` | 2 `vmulpd xmm` + 4 `vaddsd` (unfused) | 4 `fma.rn` | **no** |
  | `muladd(a,b,c)*muladd(a,b,d)` | 2 `vmulsd` + 2 `vaddsd` (unfused) | 2 `fma.rn` + 1 `mul.rn` | **no** |
  | `primitive_four_vectors` on loaded matrices | 6 `vfmadd213pd ymm` + 4 `vfmadd231sd` (all 28 fused) | 28 `fma.rn` | yes |
  | KS `metric_and_inverse` | no fma, 2 calls to a non-inlined `_hypot` | `__nv_hypot` inlined (its 4 `fma.rn`) | **no** (hypot) |
  | site body: `proxy_distance` + metric + `decode` + `det` | 15 fused lanes (3 `vfmadd213pd xmm`, 9 `sd`) out of about 26 contract pairs in the IR; 2 `_hypot` calls | 29 `fma.rn` (all pairs) + `__nv_hypot` | **no** |

  `primitive_four_vectors` alone fuses everything; the same code inlined into the site body,
  where the metric is computed in scalar registers, leaves about 11 pairs unfused. CUDA.jl's
  comment on its `muladd` override (fusion "robust under vectorization", JuliaGPU/CUDA.jl#3149)
  describes the same effect.

## 3. Host-side non-determinism

### 3.1 Context dependence

The decode inlined in `Data.emission_sites` (threaded closure, `src/infra/Data/tables.jl:103`)
and in the serial `Decks.torus_zones` (`test/decks.jl:514`) differ on the host itself:
`amr_deck(nc = 32)`, torus parameters, 104,776 zones: `dV` and `n_e` equal, `theta_e`
differs in 10,004 zones and `B` in 8,500 (`exp/emulate.jl stock`). The test suite already
allows for this: `test/pipeline/products_montecarlo.jl:173–174` compares the two decodes at
`rtol = 1e-14`, not bitwise. The README note on the Verner 9 step (1,355 of 20,000 steps
differing when the stages are compacted) is the same effect.

### 3.2 Environment dependence (generic-target pkgimages)

In `/data2/haiyangw/scratch/wavefront/env` and in `results/blp/gpu/env_main`, 206 of the
loaded native-code pkgimages carry the CPU target `generic; features_en=(cx16)`, with no
FMA and no AVX. Among them are StaticArrays, ForwardDiff, SpecialFunctions,
KernelAbstractions, GPUCompiler, CUDA, CUDACore, OrdinaryDiffEqCore and LinearSolve. BLP's
own image is icelake-server. `JULIA_CPU_TARGET` is unset and `JLOptions().cpu_target` is
`native`. The depot holds both generic and icelake caches for these packages (for example
`compiled/v1.12/LinearSolve/WR6RC_cXq43.ji` generic, `WR6RC_EaZYq.ji` icelake), and the
generic ones are the ones loaded. Who wrote the generic caches, and why the loader prefers
them, was not established.

Consequence: `Base.Math._hypot(::Float64, ::Float64)` is not in the sysimage. The first
image that compiled it was a generic one in the LinearSolve tree (the bisection reaches
`OrdinaryDiffEqRosenbrock` → `OrdinaryDiffEqDifferentiation` → `LinearSolve`), and that
code takes Base's non-FMA branch:

- `hypot(56.95171875, -6.4125000000000005)` (the Kerr–Schild `hypot` of a site-scan
  zone) is `57.311590667413` (correctly rounded) in a plain process and
  `57.31159066741301` in a process that loads BlackLightPlus before its first `hypot` call
  (`exp/hypot_which3.jl`, `exp/hypot_bisect*.jl`, `exp/one_pkg.jl`);
- over 1e7 Kerr–Schild-like arguments the BLP environment's `hypot` equals Base's
  non-FMA branch with unfused `muladd` in every case, and the FMA branch in none of the
  15,943 cases where the two differ. 1,580 of 1e6 are not correctly rounded
  (`exp/hypot_env.jl`);
- the other Base functions checked (`exp expm1 log log1p log10 cbrt sqrt sin cos tan sincos
  atan atan2 asin acos sinh cosh tanh pow x^7`) are bitwise equal between the plain and the
  BLP environment; only `hypot` differed, in 1,455 of 1e6 (`exp/math_env.jl`).

So the host's Kerr–Schild radius, and every BLP result that reads it, depends on which caches
the depot happens to load. The main-against-branch deck comparisons stay bitwise only
because both sides load the same caches. Recompiling the depot natively (deleting the generic
caches, `Pkg.precompile` after an update, a fresh machine) changes the host bits of every
deck that evaluates the Kerr–Schild metric. Generic dependency images also mean
non-vectorized, unfused host code wherever BLP calls a dependency specialization that is
not inlined.

### 3.3 The site scan under device semantics, emulated on the host

`exp/emulate.jl` redefines, in a separate host process, `Base.muladd(::Float64…)` as `fma`
and the StaticArrays `dot` as the sequential fma chain the PTX shows. That reproduces the
device's contraction on the host. The script then reruns `emission_sites`
(`exp/compare_emulation.jl`):

| mode | `hypot` | zones differing from stock `emission_sites` | fields |
|---|---|---|---|
| all fused | recompiled natively (FMA branch) | 739 of 104,776 | dV 52, θₑ 507, B 408 |
| all fused | the environment's non-FMA split branch | 661 (0.63 %) | dV 0, θₑ 473, B 372; max rel 6.7e-16 |
| mat-vec fused, `dot` unfused | env | 22,427 | |
| nothing fused | env | 32,800 | |

`n_e` never differs, because it involves no contraction. With the environment's `hypot`,
`dV` matches exactly, so the remaining 661 zones come from the `muladd`/`dot` contraction of
`b_sq` and the four-vectors. The stock site scan is "almost fully fused, minus a
context-specific subset" that SLP vectorization decides. No simple explicit form reproduces it.

## 4. Compile options and what can reach them

- `@cuda` / `cufunction` compiler keywords (`CUDACore compiler/execution.jl:67`):
  `kernel, name, always_inline, minthreads, maxthreads, blocks_per_sm, maxregs, fastmath,
  arch (cap), ptx`. None controls contraction. `fastmath = true` adds contraction and
  approximations.
- `GPUCompiler.PTXCompilerTarget` (`ptx.jl:21–46`): `cap, ptx, feature_set, debuginfo,
  minthreads, maxthreads, blocks_per_sm, maxregs, fastmath`. `CompilerConfig` adds
  `always_inline, opt_level, libraries, optimize, validate, …`
  (`interface.jl:111–136`). Nothing for FP contraction.
- `llc` (external, LLVM 22.1.7; `ptx.jl:359–364`): fixed `-mtriple -mcpu -mattr -filetype`.
  It never receives `-nvptx-fma-level` or `-fp-contract`. In-process `LLVM.clopts` set
  options of Julia's own LLVM, which runs only the middle end, so they do not reach this
  process.
- `ptxas` (`CUDACore compiler/compilation.jl:351–436`): fixed `--verbose --gpu-name
  --output-file`, plus line-info or debug flags from `-g`. `--fmad` keeps its default `true`.
- KernelAbstractions `CUDABackend(; prefer_blocks, always_inline)`; a launch compiles with
  `@cuda launch=false always_inline=backend.always_inline maxthreads=…`
  (`CUDACore CUDAKernels.jl:125`). No other compiler keyword reaches it. BLP builds
  `CUDA.CUDABackend()` with the defaults (`ext/BlackLightPlusCUDAExt.jl:17`) and has one
  kernel (`src/infra/Execution/kernel.jl:7`). `maxregs`, `minthreads` and `blocks_per_sm`
  (relevant to the register spills of the precise march) need either a direct `@cuda`
  kernel in the extension in place of the KA kernel, or KA internals. Register and spill
  counts can be read on the host without a GPU: compile the job as `exp/census.jl` does and
  run the bundled `ptxas --verbose` (the site body: 75 registers, 0 spill).
- A disable-contraction mode does not exist. Building one would take all of: an overlay of
  `Base.muladd` in CUDA's method table (piracy, global to every kernel in the session), an
  LLVM pass that strips `contract`/`reassoc` (no hook on the KA or `cufunction` path, so a
  custom `AbstractCUDACompilerParams` and an own `cufunction`), and `ptxas --fmad=false`
  (an override of `CUDACore.compile`). The result would be a device that never contracts,
  which still differs from a host that contracts some pairs. The device can be made to
  contract nothing or everything, but the host's choice is not reproducible, so no compiler
  setting is the lever.

Cost of the options that exist: `maxregs`/`always_inline` affect speed only. Explicit `fma`
in place of `muladd` costs nothing on the device (`fma.rn` either way) and nothing on the
host (`llvm.fma` vectorizes). Plain operations in place of a host-fused pair cost one extra
instruction each.

## 5. Recommendation: the site scan and the plasma decode

**Standard: portable arithmetic.** A function that must give the host's bits on a device
uses only `+ - * /`, `sqrt`, explicit `fma`, `min`/`max`, conversions and comparisons, and
BLP-owned ports of any elementary function, written with explicit operations
(`Numerics.tang_exp`, `tang_log` are the existing examples). It uses no `muladd`, no `@simd`
or StaticArrays reductions (`dot`, `norm`, `sum`, `M*v`), no `^` with a non-literal
exponent, no `@fastmath`, and no call into a non-inlined Base or dependency math
specialization. Then the host and the device each have exactly one admissible rounding per
operation (IEEE, f64 never flushes), and the bits agree by construction, in every host
context and every depot.

Concretely, for the Monte Carlo site scan (`Data.emission_sites`) on `KerrSchildCartesian`:

1. `Numerics.port_hypot`: Base's `_hypot` algorithm with its branch fixed. Two choices:
   - the non-FMA branch with unfused operations: bitwise the hypot that every deck of this
     depot has run (0 mismatches in 1e7 Kerr–Schild arguments, `exp/hypot_env.jl`), so
     `dV`, `r` and the metric keep main's bits;
   - the FMA branch: correctly rounded, and bitwise `Base.hypot` in a natively compiled
     depot (0 mismatches against the FMA port or the split port, `exp/hypot_port.jl`).
     Choosing it changes about 0.16 % of hypot results by 1 ulp.

   `_ks_term` (`src/physics/Spacetime/kerr_schild.jl:15`) reads it. This also removes the
   depot dependence of §3.2 for every host path, and it is worth doing even if nothing moves
   to the device.
2. Explicit kinematics in `primitive_four_vectors` (`src/physics/Matter/decode.jl:36, 43,
   45`): `mv4(g, v)` as StaticArrays' association with every `muladd` an `fma`
   (`fma(g[k,4], v4, fma(g[k,3], v3, fma(g[k,2], v2, g[k,1]*v1)))`), and `dot4` as the
   sequential `fma` chain from `0.0`. These are the device's current semantics, so the
   device bits do not move. The rest of `decode`, `electron_theta`, `lerp8`, `det` (4×4,
   plain) and the metric outer products are already portable.
3. A device site-scan program over groups: Philox pick (integer), `locate`, `proxy_distance`,
   `sample`, `det`, with a count → exclusive scan → write pass so that the list keeps block
   and group order. On the CPU executor it is the same program, so there is one code path.
4. A portability gate (host only, in the style of `test/structure.jl`): for each designated
   portable program, compile its device job with `libraries = false` (`exp/census.jl` shows
   how, with no GPU needed) and assert no `__nv_` symbol, no `llvm.fmuladd`, and no
   `contract`/`reassoc`/`afn`/`fast` flag in the IR. For the host, assert the same flags
   absent from `code_llvm` and no call to a non-inlined Base math function. A regression
   then fails at test time instead of as a tolerance surprise on a GPU.

Baseline: step 2 cannot keep main's site-scan bits. The emulation in §3.3 changes 661 of
104,776 zones (0.63 %, ≤ 3 ulp in `theta_e` and `B`). The torus decks at strides 1 and 8,
and every deck whose decode or march inlines `primitive_four_vectors`, change in their last
bits. The alternatives are worse: reproducing the stock fusion pattern of the
`emission_sites` closure (read from its optimized IR) would need a second decode used only
by the scan, which breaks "one sequence shared by imaging and Monte Carlo" and is rebuilt
silently by any nearby change. Leaving the scan on the host (status quo) costs 2.0 s per
uncached stride-1 call on four threads, and 0 when `grid_zones` keeps the zones.

So: step 1 (with the non-FMA split branch) is safe now and keeps every deck bitwise. Steps
2–4 need the user to accept a one-time re-baseline of the decode-reading decks, after which
CPU and GPU zones, zone tables and counts are bitwise equal for the site scan.

Estimated effort: `port_hypot` and its tests 0.5 day; explicit kinematics plus a host
comparison against the `fused_h` emulation (expected 0 mismatches) 1 day; device scan
program with an order-preserving scan 1–2 days; gate test 0.5–1 day; deck re-baseline and
CPU/GPU comparison 0.5 day plus two GPU jobs. Expected speed: the device scan is compute-
and bandwidth-light (8 corners × 8 primitives per cell, about 2.6 GB at stride 1), so about
0.1 s on the A100 in place of 2.0 s on the host. CPU: same operation count, to be measured,
since the invariant requires no CPU slowdown. Superposed and boosted metrics need the same
treatment (per-hole `_ks_term`, Lorentz mat-vecs, `sincos` of the orbital phase, which can be
taken once per call on the host because the scan runs at fixed t).

## 6. Can images be bitwise equal on CPU and GPU?

Yes, by the same construction, and nothing else achieves it. The blockers on the imaging
and march paths:

- libdevice-replaced functions: `exp`, `expm1`, `cbrt`, `sqrt` (portable already), `log`,
  `^` with float exponents and run-time constants (`2.0^(11.0/12.0)`, `3.0^(7.0/3.0)`),
  `sin`/`cos`/`sincos` (camera, tetrads, analytic surfaces, orbits), `atan`/`atan(y,x)`,
  `asin`/`acos`, `hypot`, possibly `tanh`/`sinh`/`cosh`. Each needs a port of Base's
  algorithm with explicit `fma` (Base's `exp`, `log`, `cbrt`, `atan`, trig kernels and
  `pow` are pure Julia), and each port must be verified bitwise against Base on the host
  over the whole range (like `test/physics/numerics_elementary.jl`) and on the device.
- implicit contraction: about 65 `dot`/`norm`/`sum` calls and about 50 StaticArrays matrix
  products in `src/physics` (metric, geodesic right-hand side, tetrads, coefficients), plus
  `muladd` reached through ForwardDiff duals in StaticArrays products
  (`ForwardDiff dual.jl:667–699`; plain `Dual*Dual` is already portable,
  `partials.jl:222`).
- reductions over pixels or rays must stay exact or ordered (they already are).

Effort: two to four weeks across most physics files, with every imaging deck re-baselined
and CPU speed re-measured. The current agreement (images within 1.5e-13, Monte Carlo sums
within 1e-11) has no physical consequence. Unless the user wants bitwise CPU = GPU images as
a property in its own right, the recommendation is to apply the standard to the site scan,
decode and metric only (§5), and to the elementary functions as they are touched.

## 7. GPU confirmation (written, not run)

`exp/gpu_bitcheck.jl` with `exp/gpu_bitcheck.sbatch` (partition `preempt`, 1 GPU, 4 CPUs,
30 min, log in `exp/`). It runs the census bodies (`dot`, mat-vec, shared-product
`muladd`, `hypot`, `primitive_four_vectors` on the KS metric, the KS metric pair) on 1e6
inputs on an A100 against the host, stock and portable versions. Expected: stock device ≠
stock host for `dot`, `muladd`, `hypot`, the KS metric and the kinematics; portable device =
portable host everywhere. Host-only dry run done: every kernel compiles for sm_80 and the host
side runs (`exp/gpu_bitcheck_dry.jl`). One job of a few minutes suffices; it is not needed
for the recommendation, which rests on the host evidence above.

## 8. Experiment index (`/data2/haiyangw/scratch/wavefront/plan/exp/`)

| file | what |
|---|---|
| `llc/t.ll`, `llc/k.ll`, `llc/xbb.ll` (+ `.ptx`, `.cubin`) | `llc` flag matrix, `ptxas` fusion of mixed and cross-block pairs |
| `census.jl`, `host/*.s`, `host/*.ll`, `ptx/*.ptx` | host native code against device PTX for six bodies; `dump.jl` writes the host files |
| `emulate.jl`, `compare_emulation.jl`, `{stock,fused,fused_h,h1,h0}.bin` | site scan under host and emulated-device contraction; host context comparison |
| `dv_debug.jl`, `r_debug*.jl` | trace of the `dV` differences to `hypot` |
| `hypot_port.jl`, `hypot_port_defs.jl`, `hypot_env.jl`, `hypot_env_defs.jl`, `hypot_branch.jl`, `hypot_which*.jl`, `hypot_bisect*.jl`, `one_pkg.jl` | `hypot` ports, branch identification, package bisection |
| `math_env.jl`, `math_cmp.jl`, `{plain,blp}.bin` | Base math bits, plain against BLP environment |
| `cache_target2.jl`, `env_targets.jl` | CPU targets of cached and loaded pkgimages |
| `gpu_bitcheck.jl`, `gpu_bitcheck.sbatch`, `gpu_bitcheck_dry.jl` | the GPU confirmation (not run) and its host dry run |
