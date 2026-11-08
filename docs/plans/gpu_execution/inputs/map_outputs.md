# How BLP products return results, and a device-resident output API

Tree: worktree `/data2/haiyangw/scratch/blp_wavefront` (branch `gpu-wavefront`, at `a5923dd`, identical to
main). Every `file:line` below is in that tree unless prefixed. Measurements made for this map:
`/data2/haiyangw/scratch/wavefront/plan/scripts/{host_outputs.jl,cf_cpu.jl}` (host `anta`, load
average 108–117 on 144 cores, so absolute times are noisy; ratios are the point).

## 0. Summary

- Every per-item product writes into **host arrays allocated by the product** before the run. On
  `ThreadedExecutor` the program writes them directly; on `PoolExecutor` each chunk writes
  **per-lane device scratch** (`lane_buffer`), which is copied back through **per-lane pinned staging**
  and then scattered into the host arrays by the lane task (`executor.jl:266-293`). No product output
  ever stays on a device, and no product accepts caller storage.
- Inputs are the opposite: every host array a program holds is uploaded once per device and cached by
  object identity (`residency.jl:20-45`). The only built-in product chain, `analytic_movie`, therefore
  makes a full round trip: SpotProgram maps D2H, the same maps H2D for FrameProgram, every frame D2H
  plus a host copy (`analytic_movie.jl:240-251`).
- Monte Carlo results are not output-bound: bins, ledger and children are exact integer sums written
  through atomics into per-call zeroed device twins and folded on the host by `gather!`; about 11 MB
  per device per call (§2.4).
- The closed-form 2048² radius call (0.305 s on the A100, 0.036 s for its program into preallocated
  outputs) spends its host time in four places, all in `render_analytic` itself
  (`analytic.jl:158-167`): first touch of 9 freshly allocated 32 MiB output vectors by the two lane
  tasks that scatter into them, a single-threaded fill of 6 NaN maps, a single-threaded band sum,
  and the garbage collection of the previous call's 512 MiB. The same overhead exists on the CPU
  (4 threads here: product 3.26 s against program 2.59 s, 0.67 s of host work, 0.31 s of it GC).
- Proposal (§4): keyword-only `resident` and `into` on the image products, backed by one new
  Execution type `PoolArray` and one constructor `output_array(exec, T, dims)`. Placement follows the
  executor chosen by the one `backend` option; on `ThreadedExecutor` `output_array` returns a plain
  `Array`, so CPU results keep their types and bits. A pool writes `PoolArray` outputs in place on its
  devices (no staging, no copy), and a program that holds a `PoolArray` reads its device part with no
  upload, which is how products chain on the GPU. Default-path fixes with no API change (threaded band
  sum and NaN fill, the movie chain, frames written in place) speed both backends, bitwise.

## 1. The output contract

| piece | where | what it does with outputs |
|---|---|---|
| kernel | `kernel.jl:7-10` | `prog(Int(items[i]), i, outs...)` for `i ≤ n`; `outs` are whatever arrays the executor passes |
| program protocol | `Execution.jl:7-8`, `executor.jl:185-199` | program writes item `item`'s outputs at `outs[k][i]`; afterwards `outs[k][item]` holds them |
| `store_all!` | `analytic.jl:54-58` | unrolled `outs[k][i] = vals[k]` used by the closed-form, crossing and surface programs |
| `foreach_item!` | `executor.jl:200-209` | `outs::Tuple` of host arrays; returns `outs`; no allocation of its own |
| threads | `executor.jl:230-245` | `run_range!` calls `prog(item, item, outs...)` on the host arrays: `i == item` |
| pool | `executor.jl:247-256`, `266-276` | per chunk: device items, per-lane device outputs, `launch!` (synchronising, `kernel.jl:33-44`), `gather!` per output |
| D2H | `executor.jl:278-284` | `copyto!(stage, 1, d, 1, n)` into the lane's pinned stage (`pool.jl:357-366`; `cuMemHostAlloc`, `ext/BlackLightPlusCUDAExt.jl:34-38`) |
| scatter | `executor.jl:287-293` | `o[idx[q]] = st[q]`, or one `copyto!(o, first(idx), st, 1, n)` when the chunk's items are a range |
| device scratch | `pool.jl:325-332` | `lane_buffer(lane, (:out, k), T, n)`: chunk-sized, per lane, reused across chunks and calls |
| items | `executor.jl:303-312` | ranges pass unstaged; vectors (a permuted `plan.queue`) staged as `Int32` per chunk |
| chunk sizing | `executor.jl:120-135`, `175` | `item_bytes(outs)` caps a one-device launch at `max(2^16, 2^24 ÷ bytes)` items so one launch's outputs (≤16 MiB) are copied and scattered while the other lane computes |
| input residency | `residency.jl:20-45`, `86-105` | `DeviceAdaptor` replaces each `Array` field of a program by its cached device copy (keyed by `objectid`, held weakly); `ProgramTwins` adapts each program once per device and call |
| accumulators | `accumulators.jl:145-151`, `226-232`, `415-418`, `456-462` | `DeviceAdaptor` makes a zeroed device twin per adaptation (`zeroed`, `accumulators.jl:33`: host `zeros` then `upload`); `gather!` D2H-merges and frees them |
| children | `children.jl:409-416`, `470-490`; `cascade.jl:290-365` | outside the cascade a pool allocates a device `ChildBuffer` twin per adaptation and `gather!` appends it on the host; inside the cascade the children live in two lane buffers (`cascade.jl:341-343`) and only the count is read back (`cascade.jl:345-356`) |

Consequences that matter for any redesign:

1. The device never holds a full output: `lane_buffer` is chunk-sized, so a result exists only on the
   host.
2. A cached input must not be mutated in place (`residency.jl:3-4`): `cached_twin` re-uploads only when
   the `objectid` slot holds a different object (`residency.jl:39`). Writing a host array that some
   program already uploaded leaves a stale device copy, which matters for any caller-storage API (§4.4).
3. Results are bitwise per item whatever writes them. Reordering, regrouping or relocating the writes
   changes no bit, so device-resident outputs are compatible with the invariants by construction; only
   host-side reductions (pairwise `sum`, sequential deposits) pin an order.

## 2. Per product: allocation, fill, copy, post-processing

Notation: N = pixels, K = bands/orders, "first touch" = the first write to a fresh page of a
`Vector{Float64}(undef, N)` (32 MiB at 2048², above glibc's 32 MiB mmap ceiling, so each call gets new
pages and frees them with `munmap` when collected).

### 2.1 Closed form and marched disk

| product | outputs allocated (host) | filled by | host post-processing after the run |
|---|---|---|---|
| `render_analytic(plan, disk)`, observables | `analytic.jl:153`: `analytic_outputs(K, N)` = 5K vectors (`analytic.jl:38-39`) | `AnalyticProgram` over `1:N` (`analytic.jl:154-155`) | `band_maps` reshape, no copy (`analytic.jl:121`, `156`); `I = zeros(n,n)` then `K` broadcasts `I .+= Ib[b]`, one thread (`analytic.jl:164-167`) |
| same, `observables = false` | `analytic.jl:158`: 3K vectors | `RadiusProgram` (`analytic.jl:159-160`) | reshape (`161`); **2K NaN maps by `fill(NaN, n, n)`, one thread** (`162`); band sum (`164-167`) |
| `render_analytic(plan, materials)` | `analytic_surfaces.jl:133`: `surface_outputs(K, N)` = 7K (`37-38`) | `SurfaceProgram` over `1:N` (`134`) | reshape (`136`); `I`, `Q`, `U` zero maps and 3K broadcasts, one thread (`137-140`) |
| `render_crossings` | `crossings.jl:143`: `crossing_outputs(K, N)` = 5K+1 (`41-42`) | `CrossingProgram` over `plan.queue` (a permutation: staged items, random scatter) or `1:N` for `coords` (`144`) | comprehension of reshapes (`146`); `I = zero(Ib[1])` and K broadcasts (`148-151`) |
| `photon_ring_layers` | per band `outs` of 3 vectors (`analytic.jl:217`, `224`); `Ib`, `masks` grids (`209-211`) | `RadiusProgram` over the band's points (`219`, `226`) | `paint_band!` / `bin_band!`, sequential host deposits (`244-266`); the weighted composite (`231-234`) |
| `analytic_movie` | `analytic_movie.jl:240`: 4K vectors; `frames` (`246`); `buf` (`247`) | `SpotProgram` (`241-242`), then one `FrameProgram` per time into `buf` (`248-249`) | `frames[:, :, k] .= reshape(buf, n, n)`, one copy per frame (`250`); light curve by host pairwise `sum` (`253`) |

`analytic_movie` on a pool is the one product chain in the package and it round-trips: the 4K SpotProgram
maps come back D2H through staging, `FrameProgram(A, X, Y, TS, …)` then uploads the same 4K vectors H2D
(`cached_twin`, fresh objects every call), and every frame comes back D2H into `buf` and is copied again
into `frames`.

### 2.2 Imaging

| product | outputs | host post-processing |
|---|---|---|
| `render(plan)` | `image.jl:70`: one `output_type(plan)` per pixel (`Float64`, or `SVector{4,Float64}` for IQUV), filled over `plan.queue` (`99-109`) | StokesI: `convert` (no copy) plus three `zeros(n, n)` planes (`25-29`); IQUV: four comprehension passes into new planes (`31-35`) |
| `render(plan, coords)` | `image.jl:92` | `stokes_matrix`, one host pass (`112-121`) |
| `render(plan; path)` | `trace.jl:119` | host only (`trace.jl:111`) |
| `sensitivity` / `variants` | `sensitivity.jl:199`, `239` | `sum(out)` pairwise on the host for `flux` (`202`), `value.`/`partials.`/`getindex.` maps (`203-204`, `242`) |
| `shadow` | `trace.jl:147-148` (`ys`, `ms`) | five host `map`/`reshape` passes (`152-155`) |
| `trace` | 21 window vectors per window (`trace.jl:70`), `logs`, `hlogs` | variable-length host logs window after window (`75-81`, `89-106`): host-bound by design |
| `render_amr` | per chunk `vals` (`worklist.jl:216`, `236`); mosaic planes `eff × eff` × 4 (`amr.jl:18-24`) | `paint_block!` and `needs_refinement` per block on the host (`amr.jl:32-51`, `80-91`): the refinement decision needs the values on the host |

The torus renders are kernel-bound (768² StokesI: 4.7 MB of output against 0.457 s), so outputs are not
their problem. `render_crossings` at 500² writes 16 × 2 MB; on the A100 production march (0.163 s) the
first touch and band sum are plausibly 10–15 % of the call (estimate from the host rates of §3; to be
measured).

### 2.3 Where host reductions pin an order (must stay on the host for bitwise results)

- `analytic_movie` light curve: `sum(view(frames, :, :, k))` (`analytic_movie.jl:253`), Base's
  pairwise, SIMD-blocked sum.
- `sensitivity` flux: `sum(out)` over `Dual`s (`sensitivity.jl:202`).
- `photon_ring_layers`: `bin_band!` adds fine nodes into a cell in sequence (`analytic.jl:244-253`); the
  composite multiplies and adds (`233`), which a device may contract to an fma.
- `zone_offsets`: one stream in zone order (`montecarlo.jl:372-381`).
- `hotcross_build`: `log10` on the host (`montecarlo_tables.jl:200`).

Pure band sums (`I .+= Ib[b]`, `analytic.jl:164-167`, `crossings.jl:148-151`, `analytic_surfaces.jl:137-140`)
are elementwise `((0.0 + Ib1) + Ib2) + Ib3`: no multiply, so no contraction; a program doing the same
adds in band order gives the same bits on any executor (checked on the host: `host_outputs.jl`, "band sum
bitwise: true").

### 2.4 Monte Carlo

- `mc_spectrum` → `mc_run` builds host `ExactBins` per order (`montecarlo.jl:315`, `352`) and the ledger
  `mc_ledger()` (`316`, `353`; `programs.jl:272`). `run_layers` (`montecarlo.jl:387-397`) runs
  `foreach_cascade!` with `MCPrimary`/`MCLayer`, whose `MCSink` holds the bins, the ledger and a
  capacity-0 `ChildBuffer` slot.
- On a pool, `run_cascade!` (`cascade.jl:268-282`) makes `ProgramTwins` for the primary and the layer;
  each device's first adaptation of each registers zeroed twins of every bin and of the ledger
  (`accumulators.jl:148-151`, `417-418`): 4 orders × (count + 3 × 69 limbs × 4 rows × 200 bins) ≈ 5.3 MB
  per program twin, two programs, plus the ledger (704 scalars × 69 limbs, 0.39 MB) ≈ 11 MB of host
  `zeros` + H2D per device per call, and the same D2H in `merge!` (`accumulators.jl:209-216`).
- Non-final generations record into lane scratch (`cascade.jl:143-154`) drained into the twins by the
  `Drain` program through the one kernel (`cascade.jl:163-187`, `359-365`); children stay in lane
  buffers; one 8-byte D2H per generation for the count.
- `Spectrum` rounds each cell once on the host (`accumulators.jl:273-280`); the result types hold host
  matrices only.
- Tables: `grid_tables` on a pool (`montecarlo_tables.jl:86-108`) returns `Nz` and `dnmax` as host
  vectors (`103-106`; 2 × 82 MB D2H at stride 1); `zone_offsets` needs `Nz` on the host;
  `ZoneSource` then uploads `dnmax` and `offset` again by identity (`montecarlo.jl:355`). About 10 ms of
  round trip against 0.60 s of tables: not worth an API.

Verdict: the MC result path costs milliseconds against 3.74 s (thin) and 62.5 s (Compton) on the A100.
Its GPU problem is the transport kernel, not the outputs. The relevant link to the wavefront work is
that the cascade already keeps its intermediate state on the device (`cascade.jl:290-343`); the
`PoolArray` of §4 generalises that from lane-scoped scratch to arrays that outlive a call.

## 3. Why the closed-form 2048² radius call is 0.27 s of host work in 0.305 s

Record: `results/blp/mc_audit/speed/gpu/README.md` (main checkout), legs `cf_radius` 0.305 s and
`cf_radius_prog` 0.036 s, `logs/paths2_chain_3783.log:36-37` (`final_3788.log:120-121`: 0.326 / 0.035 s).
The `_prog` leg (`scripts/paths2.jl:52-54`) runs `foreach_item!` into 9 preallocated vectors reused across
the five timed calls, so it contains the kernel, 288 MiB D2H into pinned staging and the staging-to-host
copies into already resident pages. The difference, 0.27 s, is the product's own host work:

1. **Allocation and first touch of the 9 outputs, ≈0.13 s.** `analytic.jl:158` allocates 9 × 32 MiB
   uninitialised. `foreach_item!` sizes chunks at `2^24 ÷ 72 = 233,016` items (`executor.jl:134`,
   `item_bytes` = 9 × 8 B), 19 chunks. Each chunk on one of the two lanes: launch and synchronise
   (`kernel.jl:33-44`), then for each of the 9 outputs a D2H into the pinned stage and a `copyto!` into
   the fresh host vector (`executor.jl:278-284`, `293`). That copy is the first write to every page, so
   the page faults (and the kernel's page zeroing) run on at most two threads, the two lane tasks.
   Reproduced on the host: 0.179 s fresh against 0.013 s into reused vectors (`host_outputs.jl`, two
   tasks, same chunking).
2. **Six NaN maps, ≈0.10 s.** `analytic.jl:162`: `[fill(NaN, n, n) for _ in 1:K]` twice, 192 MiB
   allocated and first-touched by the caller's thread alone. Reproduced: 0.138 s one task, 0.043 s
   threaded fresh, 0.009 s threaded into existing maps.
3. **Band sum, ≈0.035 s.** `analytic.jl:164-167`: `zeros(n, n)` (one more fresh map) and three
   full-map broadcasts on one thread. Reproduced: 0.050 s as written, 0.015 s threaded single pass fresh,
   0.007 s into an existing map; bitwise identical.
4. **Garbage collection.** Each call allocates 512 MiB (9 + 6 + 1 maps of 32 MiB; measured 512.2 MiB in
   `cf_cpu.jl`), so the next call collects about that much. On the CPU at 4 threads the product spends
   0.31 s in GC of its 0.67 s overhead (`cf_cpu.jl`: program 2.59 s, product 3.26 s, `Ib` bitwise the
   program's). The README's 0.13/0.10/0.035 split was timed op by op; GC is either inside those numbers
   or on top, and should be separated in the GPU profiling job (`@timed(...).gctime`).

The device itself is not the bottleneck: the program leg's 0.036 s is an upper bound on the kernel,
since it also holds 288 MiB of D2H (≈12 ms at PCIe 4 pinned rates) and 288 MiB of host copies (≈13 ms
on two tasks, measured). The kernel alone is probably 0.01–0.02 s; measure it with `nsys` or
`CUDA.@elapsed` around `launch!` before quoting.

The same structure holds for the full closed form (`cf_full`, 15 outputs + I: product 0.458 s against
program 0.205 s) and, scaled down, for `render_crossings` and `render_analytic(plan, materials)`. On 32 CPU
threads the closed-form radius call is 0.52 s against 0.33 s for its program
(`logs/paths2_cpu32.log:6-7`): the CPU pays the same NaN fill, band sum and GC, only its first touch is
spread over all worker threads because they write the outputs directly.

## 4. Proposed API: outputs that stay where the executor computed them, or go where the caller says

### 4.1 Constraints it satisfies

- One `backend` option decides where a product runs; the proposal adds no backend argument, no
  backend in any name, and nothing to plans or configs. Output placement follows the resolved
  executor: on `ThreadedExecutor` there is nothing to place.
- CPU behaviour unchanged: on threads every new path returns the same `Array`/`Matrix{Float64}` types
  and the same bits (`products_analytic.jl:39` asserts `Matrix{Float64}` and keeps passing).
- Gates (`test/structure.jl`): top-level exports are at 54 of 54 (`structure.jl:115`), the product family
  at 9 of 10 (`119-120`): the proposal adds **no top-level export and no product function**, only
  keywords and two `Execution` exports. One kernel (`structure.jl:63`): every new device operation is a
  program through `_foreach!`. No `CUDA.` in `src` (`67`): device allocation through
  `KernelAbstractions.allocate(backend(dev), …)`, copies through `copyto!` (CUDA.jl 6.2.2 copies between
  devices directly when peer access is available and through the host otherwise,
  `~/.julia/packages/CUDA/nBYpB/CUDACore/src/array.jl:628-647`). Files stay under 600 lines.
- Bitwise: values are per item and unchanged; host-order reductions (§2.3) stay on the host.

### 4.2 Execution layer (new file `src/infra/Execution/outputs.jl`, included after `residency.jl`)

```julia
"""
    output_array(exec, T, dims) -> output array
An array of `dims` elements of `T` for the outputs of runs on `exec`: an `Array{T}` (uninitialised)
on a `ThreadedExecutor`, a `PoolArray` on a `PoolExecutor`.
"""
output_array(::ThreadedExecutor, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
output_array(e::PoolExecutor, ::Type{T}, dims::Dims) where {T} = PoolArray{T}(e.pool, dims)

"""
    PoolArray{T,N}
An output array that lives on the devices of a `DevicePool`: per device that ran a chunk, one device
vector of all `prod(dims)` elements, of which that device wrote the items of its chunks. Not an
`AbstractArray` (no host indexing). `size`, `eltype`, `reshape`, `vec` share the storage;
`Array(a)` / `copyto!(dst::Array, a)` / `convert(Array, a)` assemble the host array, bitwise the values a
host-output run returns; `resident(pool, dev, a)` is the complete copy on `dev` (§4.3). Items no chunk
wrote are undefined, as in `Array{T}(undef, dims)`.
"""
struct PoolArray{T,N}
    pool::DevicePool
    dims::NTuple{N,Int}
    parts::Vector{Any}            # per device slot of the pool: device vector or nothing
    written::Vector{Vector{Any}}  # per device slot: item lists (ranges or host index views) of its chunks
    lock::ReentrantLock
end
```

`convert(Array, x::Array) === x` (no copy), so `convert(Array, r.I)` is the one backend-agnostic way
for user code to get host data from either executor.

Changes in the executor (`executor.jl`):

- `run_chunk!` (`266-276`): per output, a host `Array` keeps today's path (`lane_buffer` + `gather!`);
  a `PoolArray` of the executor's pool gets `view(local_part!(o, lane), itd)`, where `itd` is the
  chunk's device item list from `device_items` (`303-312`) and `local_part!` allocates the device's part
  on the lane at its first chunk; the program's `outs[k][i] = v` then lands at `part[item]`. No staging,
  no D2H, no scatter; the chunk's item list is appended to `written[dev]` under the lock. Programs,
  `store_all!` and the kernel are unchanged (Adapt already maps `SubArray` of a device array with a
  range or device-vector index).
- `item_bytes` (`175`) counts host outputs only, so resident runs take the one-device `2^20` chunk
  (4 launches instead of 19 at 2048²); no bit depends on the chunk.
- `run_parts!(::ThreadedExecutor)` (`230-237`) is unchanged: there, outputs are `Array`s. A `PoolArray`
  given to a threaded run, or one of another pool, is an `ArgumentError` naming both.
- Host outputs of a pool run are evicted from the pool's residency caches before the run (`evict!`,
  a delete by `objectid` in `residency.jl`), so a host array a program uploaded earlier can safely be
  reused as an output (the stale-twin hazard of §1.2).

Changes in residency (`residency.jl`):

- `Adapt.adapt_structure(to::DeviceAdaptor, a::PoolArray)` returns `readonly(to.dev, complete!(a, to.dev))`:
  a program that holds a `PoolArray` field reads the device part with no upload. This is the chaining
  mechanism; it runs inside `program_twin!`, on a lane with the device active.
- `resident(pool, dev, a::PoolArray)` (the existing exported name, `residency.jl:57-58`) returns
  `complete!(a, dev)`: the device vector for user kernels (one device: the part itself, no copy).
- `complete!(a, dev)`: on one device a no-op; on several, the items other devices wrote are copied into
  `dev`'s part: contiguous `copyto!` for range chunks; for permuted chunks a gather program on the
  source device into a lane buffer, one copy, a scatter program on `dev` (both through `_foreach!`).
  The copies must be ordered before the consumer launch: a device-protocol method
  (`devices.jl`, fallback `copyto!`; CUDA method in the extension that synchronises the source
  context's stream) is the clean place for that.
- `Array(a)`: one device, one D2H of the part; several, each device's written item lists copied out of
  its part (contiguous for ranges), so one GPU and two GPUs give the same bits.

### 4.3 Products: two keywords

```julia
render_analytic(plan, disk; bands, ro, observables, backend, resident = false, into = nothing)
render_analytic(plan, materials; bands, ro, backend, resident = false, into = nothing)
render_crossings(plan, disk; orders, coords, backend, resident = false, into = nothing)
render(plan; backend, path, resident = false, into = nothing)
analytic_movie(plan, spot; times, bands, ro, slow, backend, resident = false, into = nothing)
```

- `resident = true`: every returned map is `output_array(exec, …)`, reshaped. On threads that is the
  same `Matrix{Float64}` as today; on a pool a `PoolArray{Float64,2}` that never left the devices. The
  host post-processing becomes programs on the run's executor: `BandSum` (`I`, `Q`, `U`: adds in band
  order, bitwise the broadcasts) and `Fill` (the NaN observables of `observables = false`, the zero
  `Q`, `U`, `V` planes of a Stokes I `render`, and `Split` for IQUV planes, a copy of each component).
  `Image` gains a plane-type parameter (`Image{M,P,R}`, `I::M`; `M = Matrix{Float64}` by default; the
  only use of its parameters is its definition, `image.jl:15`).
- `into = r`: `r` is a result of the same product with the same pixel count and band count; its arrays
  are overwritten and the returned result shares them (the old `r` shows the new values). Host arrays
  give host outputs with no allocation, no first touch and no GC; `PoolArray`s of the executor's pool give
  resident outputs with no device allocation. Placement follows `into` when given; a `resident` that
  contradicts it, a size or band-count mismatch, or arrays of another pool are an `ArgumentError`.
- Neither keyword names a backend or exists in a plan. With both left at their defaults, the result is
  the type and the bits of today on every executor.

User view:

```julia
backend!(:gpu)                                      # or BLP_BACKEND=gpu; the one option
r = render_analytic(plan, disk; observables = false, resident = true)
I = convert(Array, r.I)                             # host copy when wanted; bitwise the default result
d = resident(backend().pool, CUDADevice(0), r.rs[1])  # the device vector for CUDA.jl code, no copy
for a in spins                                      # a scan into the same device storage
    r = render_analytic(plan_of(a), EquatorialDisk(a); observables = false, into = r)
end
# a user program over resident maps (Execution.foreach_item!), no host round trip:
out = Execution.output_array(backend(), Float64, (npixels(plan),))
foreach_item!(backend(), MyProgram(r.Ib[1], r.rs[1]), 1:npixels(plan); outs = (out,))
```

The same lines run with `backend!(:cpu)`: `resident` and `output_array` give `Array`s, `convert` is the
identity, `resident(…)` is not needed.

### 4.4 Default-path changes (no API, bitwise, both backends)

1. Band sums and NaN/zero fills as `BandSum`/`Fill` programs on `ThreadedExecutor()` when the outputs are
   host arrays (the pipeline may not use `Threads.@threads`, `structure.jl:57-59`; `run_workers!` is
   allowed). Measured on 4 threads: NaN maps 0.138 → 0.043 s, band sum 0.050 → 0.015 s, so the GPU
   radius call goes from about 0.305 to about 0.20 s and the CPU call saves the same 0.13 s on 4 threads
   (more on 32).
2. `analytic_movie` keeps its SpotProgram maps as `output_array`s internally: on a pool the FrameProgram
   reads them with no upload, and the returned `A`, `x`, `y`, `ts` maps are assembled once at the end
   (the same D2H as today, minus the 4K-map H2D). Each `FrameProgram` writes `vec(view(frames, :, :, k))`
   directly, dropping `buf` and its copy (`analytic_movie.jl:247-250`) on both backends. The light curve
   stays a host sum.
3. Optional, MC: `grid_tables` returns `dnmax` as an `output_array` so `ZoneSource` reads it in place
   (`montecarlo_tables.jl:103-106`, `montecarlo.jl:355`): about 10 ms, low priority.

### 4.5 Expected effect (estimates, to be measured on the A100)

| call | today | default path after §4.4 | `into` (host arrays) | `resident` |
|---|--:|--:|--:|--:|
| closed form radius 2048², A100 | 0.305 s | ≈0.20 s | ≈0.05 s (program 0.036 + refills) | ≈kernel, 0.01–0.02 s |
| closed form full 2048², A100 | 0.458 s | ≈0.35 s | ≈0.21 s | ≈kernel (stack-heavy program) |
| closed form radius 2048², CPU 32 thr | 0.52 s | ≈0.42 s | ≈0.34 s | = default (threads) |
| `render_crossings` 500², A100 production | 0.163 s | −(a few %) | −(10–15 %?) | −(10–15 %?) |
| `analytic_movie` | round trip per call | no H2D re-upload; no frame copy | — | frames on device |
| torus `render`, `mc_spectrum` | kernel-bound | unchanged | unchanged | unchanged |

### 4.6 Not covered, and why

- `photon_ring_layers`, `render_amr`, `trace`, `render(plan; path)`, `sensitivity`, `variants`: their
  assembly is a host reduction with an order (§2.3), a host refinement decision, or variable-length
  host logs. `shadow` could take `resident` later through a small split program; low value.
- MC results: tiny and already device-accumulated (§2.4).
- Interop beyond `resident(pool, dev, a)` (for example `CuArray(a)` in the extension) is optional.

## 5. Call sites that would change

Source (`src/`):

| file:line | change |
|---|---|
| `infra/Execution/Execution.jl:18-29`, `32-52` | include `outputs.jl` after `residency.jl`; export `PoolArray`, `output_array` (docstrings required, `structure.jl:122-126`) |
| `infra/Execution/outputs.jl` (new) | `PoolArray`, `output_array`, `complete!`, `Array`/`copyto!`/`convert`/`reshape`/`vec`/`size`, gather/scatter programs for multi-device completion |
| `infra/Execution/executor.jl:175` | `item_bytes` counts host outputs only |
| `infra/Execution/executor.jl:200-216` | argument check: no `PoolArray` on threads or of another pool; evict host outputs from residency before a pool run |
| `infra/Execution/executor.jl:266-276` | `run_chunk!`: per-output target (lane buffer + `gather!`, or in-place view of the device part) and the written-items record |
| `infra/Execution/residency.jl:20-23`, `57-58` | `adapt_structure(::DeviceAdaptor, ::PoolArray)`; `resident(pool, dev, ::PoolArray)`; `evict!` |
| `infra/Execution/devices.jl:163-221` | ordered cross-device copy in the device protocol (fallback `copyto!`) |
| `ext/BlackLightPlusCUDAExt.jl` | the CUDA method of that copy (stream synchronisation of the source context) |
| `pipeline/Products/analytic.jl:143-170` | `resident`/`into`; outputs from `output_array` or `into` (`153`, `158`); `band_maps` on either type (`121`); `Fill` for the NaN maps (`162`); `BandSum` for `I` (`164-167`) |
| `pipeline/Products/analytic_surfaces.jl:123-144` | the same for `surface_outputs` (`133`), maps (`136`), `I`/`Q`/`U` (`137-140`) |
| `pipeline/Products/crossings.jl:130-155` | the same for `crossing_outputs` (`143`), maps (`146`), `I` (`148-151`), `steps` (`153`) |
| `pipeline/Products/image.jl:15-35`, `66-73` | `Image{M,P,R}`; `render` outputs (`70`); zero planes and IQUV split as programs under `resident` |
| `pipeline/Products/analytic_movie.jl:232-257` | internal `output_array` for the 4K maps (`240`); frames written in place (`246-250`); `resident`/`into` |
| `pipeline/Products/programs.jl` (or `analytic.jl`) | `BandSum`, `Fill`, `Split` programs (`isbits` after `Adapt`) |
| `pipeline/Products/montecarlo_tables.jl:103-106` | optional: `dnmax` as `output_array` |
| unchanged | `kernel.jl`, `pool.jl` (uses `on_device` `269-276` and lane tasks as they are), `cascade.jl`, `children.jl`, `accumulators.jl`, `worklist.jl`, `amr.jl`, `trace.jl`, `sensitivity.jl`, `montecarlo.jl`, every program |

Tests (`test/`):

| file:line | change |
|---|---|
| `infra/execution_executors.jl:118-250` | `PoolArray` outputs on `:host_pool` and `Decks.hosts(2/3)`: bitwise against `Array` outputs for ranges, permuted items, `order`, several parts (`206-223`), explicit chunks; the subset semantics of `168-174` for `PoolArray` (unwritten items undefined, written ones exact); `ArgumentError`s |
| `infra/execution_residency.jl:45-100` | a program holding a `PoolArray` reads it with no new cache entry; multi-device `complete!`; eviction of a reused host output |
| `pipeline/products_analytic.jl:18-56` | `resident`/`into` on `:cpu` (types `Matrix{Float64}`, bits as `r`), on host pools (`Array(...)` bitwise), `observables = false` with `into` refills NaN |
| `pipeline/products_crossings.jl:71-84` | the same for `render_crossings` (permuted queue) |
| `pipeline/products_surfaces.jl:28-51` | the same for materials |
| `pipeline/products_movie.jl:11-46` | the in-place frames and the internal chain stay bitwise (`39-46`) |
| `pipeline/products_render.jl:25-60` | `render(plan; resident = true)` StokesI and IQUV planes |
| `pipeline/products_backend.jl:33` | run records unchanged |
| `infra/execution_cuda.jl` | one-device and two-device `PoolArray` legs on CUDA (GPU job) |
| `decks.jl`; main checkout `results/blp/mc_audit/speed/gpu/scripts/decks.jl` | resident legs whose `Array` must equal the default decks bitwise, CPU and GPU |
| `structure.jl` | no change expected; it must keep passing (exports 54, family 9) |

Docs and benchmarks:

| file:line | change |
|---|---|
| `docs/src/design/architecture.md:329-433` (§4.2) | the `outs` contract (`Array` or `PoolArray`), output residency, `output_array`, `complete!` |
| `docs/src/design/architecture.md:541-602` (§5.3) | `resident` and `into` in the products table and bullets; `Image{M,P,R}` |
| `docs/src/design/performance.md:33-37` (§1 "Outputs in the product's layout") and `64-79` (§2 knobs) | resident outputs and caller storage; the chunk rule for resident runs |
| `docs/src/design/performance.md:288-291` (§6) | the closed-form paragraph after the change, with the new numbers |
| `docs/src/guide/gpu.md` | a "Keeping results on the devices" section; `docs/src/guide/closed_form.md:21` (`observables = false` with `into`) |
| `docs/src/api/execution.md`, `docs/src/api/products.md` | the new names and keywords |
| `bench/bench.jl:317` | analytic leg with `into` and `resident` variants |
| `bench/kernel_audit.jl:240-245` | optional legs for `BandSum`/`Fill`/`Split`; existing baselines unaffected (default path uses the same launches) |
| main checkout `results/blp/mc_audit/speed/gpu/scripts/paths2.jl:41-56` | `cf_radius_into`, `cf_radius_resident`, kernel-only timing |

Figure and paper scripts (`figures/paper/fig_cmp_disk.jl:182`, `189`, `233-234`; `fig_examples.jl:42`)
run on `:cpu` and need no change.

## 6. Verification for the implementation stage

1. CPU first, no GPU: `:cpu`, `:host_pool`, `Decks.hosts(3)` with `resident`/`into`; every leg bitwise the
   default result; `test/structure.jl` green.
2. Fixed decks (`scripts/decks.jl`) on 1 and 16 threads and on the A100, bitwise against `a5923dd`;
   add resident legs compared through `Array`.
3. GPU job (partition `preempt`, one A100, 4 CPUs): `cf_radius` default / `into` / `resident`, kernel
   alone (`CUDA.@elapsed` or `nsys`), GC time per call; `render_crossings` 500² and 2048² with `into`;
   `analytic_movie` at 512², 50 frames. Then two devices for `complete!` and `Array` (bitwise one vs two
   GPUs) and the P2P path between the two A100s.
