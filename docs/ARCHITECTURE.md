# BLP architecture

The package `BLP` (`BLP/`, version 0.2.0, Julia ≥ 1.10) applies the principles of
`docs/DESIGN_PRINCIPLES.md` (P1–P18); `BLP/test/structure.jl` holds their mechanical
gates. `BlackLightPlus`, the package in `src/`, is the reference implementation the
oracle suite compares against (§6).

---

## 1. Three tiers, one direction

```
pipeline/   Plans · Analysis · Products            composition: configuration, plans,
                                                   image-plane analysis, products
    │  uses both tiers below
infra/      Data · Execution  (+ ext/BLPCUDAExt)   machinery: file formats, fluid containers,
                                                   executors, device pool, residency, scheduling
    │  implements the protocols of, and uses, the tier below
physics/    Numerics · Spacetime · Matter ·        equations: pure, isbits after Adapt,
            Transfer · MonteCarlo                  allocation-free on the hot path, no threads
```

Rules, each a gate of `BLP/test/structure.jl`.

- `BLP.jl` includes the modules in the order `Numerics → Spacetime → Matter → Transfer →
  MonteCarlo → Data → Execution → Plans → Analysis → Products`; a module imports
  (`using ..X`, `import ..X`) only modules before it.
- No physics module imports `KernelAbstractions`, `HDF5`, `CUDA` or `Base.Threads`.
  `Threads.@threads` appears only in `Execution/executor.jl` and in the `Data` loaders
  and table builders (`athenak_hdf5.jl`, `athenak_bin.jl`, `uniform_grid.jl`, `tables.jl`).
- There is exactly one `@kernel`, `_foreach!` in `Execution/kernel.jl`. Every device
  computation is a per-item program run through it.
- `src/` neither loads nor calls `CUDA`; `ext/BLPCUDAExt.jl` adds the `CUDADevice` methods
  of the `Execution` device protocol. Nothing imports `ext/`.
- A module's eponymous file holds all its `include`s and `export`s. A name with a leading
  underscore is never referenced outside its module, from `src/` or from the tests.
- No file of `src/` exceeds 600 lines.

The physics tier defines the protocols (`Spacetime`, `PlasmaProvider`, `TransferMode`, the
march-control functions, `Sink`); the infra tier supplies the grid-backed providers and the
machinery that runs per-item programs; the pipeline tier turns a configuration into a plan,
builds programs from it and hands them to an executor.

## 2. Directory layout

Paths relative to `BLP/`; each directory lists its files in include order, the eponymous
file first. §8 gives the role and the main definitions of every file.

```
src/
├── BLP.jl
├── physics/
│   ├── Numerics/      Numerics.jl constants.jl units.jl raystate.jl stencils.jl exactsum.jl
│   │                  philox.jl special.jl
│   ├── Spacetime/     Spacetime.jl protocol.jl kerr_schild.jl user_metric.jl superposed.jl
│   │                  adm_grid.jl metric_ops.jl geodesic.jl termination.jl adaptive.jl
│   │                  camera.jl kerr_orbits.jl analytic_kerr.jl
│   ├── Matter/        Matter.jl records.jl electrons.jl decode.jl protocol.jl
│   │   ├── sources/   shared.jl polish_doughnut.jl page_thorne.jl hotspot.jl gdisk.jl
│   │   └── zones.jl
│   ├── Transfer/      Transfer.jl params.jl synchrotron.jl tetrads.jl coefficients.jl
│   │                  segments.jl controls.jl march.jl
│   └── MonteCarlo/    MonteCarlo.jl superphoton.jl emissivity.jl hotcross.jl tables.jl
│                      generate.jl scatter.jl opacity.jl sinks.jl track.jl
├── infra/
│   ├── Data/          Data.jl fluid_blocks.jl athenak_hdf5.jl athenak_bin.jl uniform_grid.jl
│   │                  series.jl npy.jl tables.jl
│   └── Execution/     Execution.jl devices.jl pool.jl residency.jl kernel.jl executor.jl
│                      backend.jl ray_sources.jl schedule.jl worklist.jl accumulators.jl
└── pipeline/
    ├── Plans/         Plans.jl config.jl lanes.jl plan.jl prepare.jl
    ├── Analysis/      Analysis.jl refinement.jl lensing_bands.jl interferometry.jl
    └── Products/      Products.jl programs.jl image.jl trace.jl sensitivity.jl analytic.jl
                       amr.jl
ext/
└── BLPCUDAExt.jl
```

## 3. Physics tier

Everything here is pure, `isbits` (or `isbits` after `Adapt`), `StaticArrays`-typed and
number-type generic, and allocates nothing on the hot path. `Adapt` only declares
structure. The adaptive tracer `trace_geodesic` is the one host-only solver.

### 3.1 Numerics

| Item | Definitions |
|---|---|
| Constants | `CL`, `EE`, `ME`, `MP`, `KB`, `HPL`, `GNEWT`, `MSUN`, `SQRT2`, `SIGMA_THOMSON` (cgs) |
| Units | `Units{T}` with fields `L_unit`, `T_unit`, `dens_unit`, `Ne_unit`, `B_unit`, `e_unit`; `Units(m_bh_cgs, mass_scale_cgs)`, a `Dual` mass scale promotes every field |
| Ray state | `const RayState{T} = SVector{8,T}`; `position(y)`, `momentum(y)`, `raystate(x, p)` |
| Stencils | `primal(x)`; `corner_weights(f, n) -> (i0, d)`; `lerp8(c::NTuple{8}, d1, d2, d3)`, the one trilinear blend, association order fixed; `cr_weights`, `cr_axis`, `cr_sample` (Catmull-Rom); `loglinear_at(x, table, x0, dx, n)` |
| Exact sum | 68 signed limbs of 31 bits per cell; `xsum_parts`, `xsum_add!(acc, x, I...)`, `xsum_atomic_add!(acc, x, I...)`, `xsum_round`, `xsum_zeros`, `xsum_round_all`, `XSUM_MAX_ADDENDS` |
| RNG | `philox4x32`, `philox_rand`, `MCRng`; samplers `mc_uniform`, `mc_gauss`, `sample_direction`, `standard_gamma`, `chi_square`, each returning `(value, rng)`; `MCStreamLayout(n_primary, track_max, max_layer)`, `primary_rng(seed, gid)`, `child_rng(parent, event, layout)`, `stream_id`, `stream_layer` |
| Special | `bessel_i0`, `bessel_i1`, `bessel_k0`, `bessel_k1`, `bessel_k2` (A&S 9.8), `harmonic_bridge`, host `hypergeometric_2f1` |

### 3.2 Spacetime

```julia
abstract type Spacetime{T} end                            # T: the lane (Float64, Float32, Dual)
abstract type StationarySpacetime{T} <: Spacetime{T} end  # ∂_t g ≡ 0

metric(st, x)::SMatrix{4,4}                               # required
inverse_metric(st, x) = inv(metric(st, x))                # defaults, closed forms override
metric_and_inverse(st, x) -> (gcov, gcon)
lower_index(st, x, v); raise_index(st, x, p); raise_and_norm(st, x, p)
horizon_radius(st)                                        # Kerr r₊; per-hole radii; ADM alpha_min
lane(st, ::Type{T})                                       # scalars converted from the held values

inverse_metric_gradient(st, x); metric_gradient(st, x); christoffel(st, x[, gcon, dg])   # ForwardDiff
geodesic_rhs(st, y); geodesic_transport_rhs(st, y, vs); rk4_step(st, y, h)
null_error(st, y); null_covariant(st, x, p_spatial) -> (p, ok)

struct MarchBudget{T}; C::T; escape::T; R_em::T; nmax::Int; end     # lane(b, T) converts
struct MarchStep{T}; stop::Bool; h::T; R::T; end
march_control(st, y, r_prev, b::MarchBudget) -> MarchStep           # the one termination rule
trace_geodesic(st, y0; λmax, reltol, abstol, escape, solver = Vern7(), save_everystep)
```

| Type | Fields | `lane(st, T)` | Capture in `march_control` |
|---|---|---|---|
| `KerrSchildCartesian{T}` | `a`, `r_h` (computed once from `a`) | narrows both | `r − r_h` on the Kerr-Schild radius |
| `Minkowski{T}` | — | `Minkowski{T}()` | none |
| `UserMetric{F,T}` | `g`, a closure `x -> SMatrix{4,4}` | same closure | none |
| `SuperposedKerrSchild{N,T}` | `mass`, `spin`, `center`, `r_h` per hole | every field | `min_A(r_A − r_h,A)` |
| `ADMGrid{T,A}` | origin, spacing, node counts, `data`, `alpha_min`, `offgrid_eta` | the grid itself | lapse below `alpha_min` |

- A march stops when the distance above the horizon proxy falls below
  `HORIZON_EPS_SAMPLED` (ADM: the lapse margin below zero) or is non-finite, when the
  normal-frame photon energy exceeds `CAPTURE_REDSHIFT = 100`, or when the ray is outbound
  beyond `escape` or `R_em`. The step is that distance over `C` on Kerr and binaries,
  `max(R, 1)/C` on the coordinate radius `R` elsewhere. The rule's arguments share one
  type: a `Float64` scalar in a `Float32` march is a `MethodError`.
- `ADMGrid` is one struct for host and device: `Adapt.adapt` replaces `data`, and every
  method reads it through the same Catmull-Rom body. Its scalars share the data type, so
  it has no `Float32` lane.
- `PlaneParallelCamera(; inc_deg, d, fov, npix)`; `CameraFrame{T}(sin_i, cos_i, d, up)`
  with `up = (0, −cos i, 0, sin i)`, the polarization vertical the IQUV march transports.
  `camera_ray(st, frame, α, β) -> (y, ok)` is the received photon traced backward
  (`p = −k`, `p_t = +1`); `ok` is false without a positive-energy null root at the camera.
- Exact Kerr: `pixel_prep(α, β, θo, a; ro)` once per screen point, then
  `band_landing(prep, n) -> Landing{T}` with fields `r, φ, t, νr, valid, λ, η`;
  `critical_curve(a, θo; n)`. Painting a landing is `Matter.intensity(disk, landing)`.

### 3.3 Matter

```julia
struct PlasmaPrimitives{T}    # code units, AthenaK layout; inside = false ⇒ every field zero
    rho::T; uu::SVector{3,T}; eint::T; bb::SVector{3,T}; inside::Bool
end
struct PlasmaState{T}         # cgs, with the fluid kinematics at the sample
    n_e::T; theta_e::T; B::T
    ucon::SVector{4,T}; ucov::SVector{4,T}; bcon::SVector{4,T}; bcov::SVector{4,T}
    b_sq::T; sigma::T; beta_inv::T; valid::Bool
end
vacuum(::Type{T}) -> PlasmaState{T}                        # the one invalid state

ElectronModel{T}(; mu, ne_ni, use_p::Bool, rat_low, rat_high, gamma, gamma_i, gamma_e, thermal_frac)
Bound{T}(lo, hi)                                           # an infinite end is inactive
legacy_bound(T, lo, hi)                                    # a negative end becomes ∓Inf
PlasmaCuts{T}(; rho, n_e, pgas, theta_e, b, sigma, beta_inv)   # Bounds, inactive by default
decode(prim, g, units, electrons, cuts) -> PlasmaState     # one decode, imaging and Monte Carlo

abstract type PlasmaProvider end
abstract type GridProvider <: PlasmaProvider end           # primitives(p, x) -> PlasmaPrimitives
abstract type AnalyticProvider <: PlasmaProvider end       # plasma_state(p, st, x, g, params)
domain_radius(p) -> Float64                                # vacuum beyond this coordinate radius
sample(p, st, x, params) -> (state, g)                     # g = metric_and_inverse(st, x)
```

- `sample` on a grid provider evaluates the metric only where `prim.rho > 0`; elsewhere
  it returns `vacuum` and a zero pair that must not be read. A NaN quantity never trips a
  cut; `electron_theta` is NaN when `thermal_frac` is zero.
- Analytic providers, which take the metric pair instead of recomputing it:
  `PolishDoughnut`, `PageThorneDisk`, `EquatorialHotspot` and `UniformSphere` (the Monte
  Carlo sphere). The grid providers live in `Data` (§4.1). `EquatorialDisk` is not a
  provider: the exact-Kerr products paint a landing with `intensity(disk, landing)`.
  `EmissionZone{T}` is the input of superphoton generation.

### 3.4 Transfer

```julia
abstract type TransferMode end
struct StokesI <: TransferMode end
struct IQUV    <: TransferMode end

PowerLaw(frac, p, gamma_min, gamma_max); Kappa(frac, kappa, w)     # fit constants computed once
struct TransferParams{T,P,K,E,C,N}
    units::Units{T}; electrons::E; cuts::C; powerlaw::P; kappa::K; nu_obs::N
end
TransferParams(units, electrons, cuts, nu_obs; powerlaw = nothing, kappa = nothing)
has_powerlaw(p); has_kappa(p)          # false at compile time for `nothing`
output_type(mode, params)              # StokesI: the promoted scalar, SVector{K} for K sets; IQUV: SVector{4}

coefficients(::StokesI, p, state, kcov, ν_obs)                -> (j, α)
coefficients(::IQUV, p, state, kcov, kcon, g, ν_obs, renorm)  -> (j, α, ρ, tetrad, valid)
segment(::StokesI, j, α, ds) -> (A, B);  segment(::IQUV, j, α, ρ, ds) -> (M, b)

struct SpacetimeMarch{S,T}; st::S; budget::MarchBudget{T}; end  # fixed control, geodesic lane
struct StoredPath{S,V,D}; st::S; ys::V; dls::D; end             # replay of a trace
march_control(ctl, y, r_prev, it); geodesic_step(ctl, y, h, it); max_steps(ctl)
march(mode, y0, prog)
```

- A march reads five fields of its program: `st` (the `Float64` transfer spacetime of the
  plasma decode, the wave vector and the tetrads), `control`, `provider`, `params` and
  `frame` (a `CameraFrame{Float64}`). The geodesic, and under `IQUV` the transported
  camera frame, run in the lane of `y0`; the transfer runs in `Float64`.
- The derivative lane is a `TransferParams` whose units or electron model are `Dual`; the
  mass-scale ladder is an `NTuple{K,TransferParams}` with an `SVector{K}` accumulator
  (Stokes I, grid providers, primitives sampled once per step).
- `SpacetimeMarch` defines its own `Adapt.adapt_structure` (in `controls.jl`), adapting its
  spacetime, so an ADM grid's data reaches the device with the control. The IQUV march
  also ends where the transported frame admits no spacelike screen leg.

### 3.5 MonteCarlo

```julia
generate(zone::EmissionZone, tables::GenerationTables, rng) -> (Superphoton, rng)
generate(sphere::UniformSphere, tables, rng)                 # uniform point, then the zone method
GenerationTables(weights::WeightTable, dnmax[, int_jnu::IntJnuTable])

rate_frame(provider) -> FluidFrame() | StaticFrame()         # StaticFrame for UniformSphere
absorption_rate(state, p); absorption_rate(::StaticFrame, state, kcon)   # ν α_ν
scattering_rate(table::HotcrossTable, state, p)              # ν n_e σ_h
struct Bias{T}; b0::T; end; bias(b, nscatt)                  # max(1, b0 (n + 1)²)
Opacities(; params, table, bias = Bias(1.0), L_unit = 1.0, weight_min = MC_WEIGHT_MIN, max_scatt = 0)

abstract type Sink end      # record!(sink, sph); spawn!(sink, child, rng, event); tally!(sink, sph, loss, spawned)
may_scatter(provider, state); scatter_site(provider, st, x, params) -> (zone, ok)
FlatMarch(sphere; step_frac = SPHERE_STEP_FRAC, nmax = SPHERE_MAX_STEPS)
CurvedMarch(st; C = 80.0, escape = 1.0e3, R_em = 100.0, nmax = CURVED_MAX_STEPS)
track!(ctl, st, provider, opac, sph, rng, sink; absorb = Val(true), scatter = Val(false))
    -> (sph, recorded, rng)
```

- One tracker serves the sphere and the GRMHD fluid, with and without scattering. The
  control decides the step and the stop (`FlatMarch`: straight lines, `h = step_frac
  R/|k|`; `CurvedMarch`: RK4, `h = (r − r_h)/(C |k|)`) through the march-control functions
  of `Transfer`; the provider supplies the plasma through `Matter.sample`; `absorb` and
  `scatter` are compile-time switches.
- `rate_frame(provider)` selects the absorption rule; the two rules agree in exact
  arithmetic on a null `k` and differ in rounding. Scattering is the biased
  exponential-transform decision `scatter_event`; `may_scatter` (`state.valid`) and
  `scatter_site` (the provider's sample) decide where and off what a photon scatters. The
  sphere scatters on every step that accrues depth, the leaving step included, and always
  kicks with its uniform plasma. A child's stream is `child_rng(rng, event, layout)`.
- `bias` returns 1 unless `b0 (n + 1)²` exceeds 1, a NaN product included. Constants carry
  no leading underscore. `fluid_tetrad` (imaging, e₍₃₎ along k) and `emission_tetrad`
  (Monte Carlo, e₍₁₎ along b) are different physics and stay apart.

## 4. Infrastructure tier

No physics is written here. This tier owns arrays, files, threads, devices, queues and
caches, and implements the physics protocols over its containers.

### 4.1 Data

```julia
struct FluidBlocks{A5,A2,L} <: GridProvider    # one struct for host and device
    data::A5        # (8, ni, nj, nk, nmb) Float32, feature fastest, PRIM_NAMES order
    x_left::A2; dx::A2; lo::A2; hi::A2          # (3, nmb) Float64
    ni::Int; nj::Int; nk::Int; nmb::Int
    locate::L       # LeafTable | LevelHash | BlockScan
end
struct AthenaKSnapshot <: GridProvider; blocks; levels; logical_locations; names; time; end
AthenaKSnapshot(filename; gamma = nothing)
AthenaKSnapshot(data, lo, hi, x_left, dx, levels, logical_locations; names, time)
struct UniformGrid{A4} <: GridProvider; data::A4; n; xo; inv_dx; box_half; end
resample(p::GridProvider, box_half, n) -> UniformGrid
struct FrameWindow{P} <: GridProvider; a::P; b::P; t_a; inv_dt; t_obs; end
SnapshotSeries(paths, times; maxslots = 2, t_obs = times[end])
window(series, ia[, box_half, n]; t_obs); advance(win, series, ia; t_obs)
struct SeriesProvider <: GridProvider; series::SnapshotSeries; end
blend(sa::PlasmaPrimitives, sb::PlasmaPrimitives, w)          # the one time blend
```

- The snapshot constructors build the locate strategy eagerly (`build_locate`): a
  `LeafTable` (dense finest-block grid), a `LevelHash` for meshes beyond 8,000,000
  finest-block cells or not aligned to the finest grid, or a `BlockScan`. A `LevelHash`
  is host only: `Adapt.adapt` turns it into the `BlockScan` every device runs.
- `primitives(blocks, x)` is the clamped trilinear sample in the owning block, in
  `Float64`. `FluidBlocks` (index `(x − x_left)/dx`) and `UniformGrid` (index
  `(x − xo) inv_dx`, vacuum outside the node box) share `lerp8`.
- `FrameWindow{P}` samples both frames and blends the interpolated results at the
  emission time `t_obs + X⁰`; `P` is `FluidBlocks` or `UniformGrid`, the frames share the
  mesh, and the window is `isbits` after `Adapt` when `P` is. `window` builds one from a
  series; `advance` keeps the upper frame as the new lower one. `SeriesProvider` brackets
  any emission time through the series on the host, loading under the series lock.
- `.bin` dumps: `read_athenak_bin`, `sample_var`, `resample_z4c_adm(s, half, n;
  alpha_min)` onto an `ADMGrid`, `z4c_grid_spacetime`, `athenak_bin_snapshot(binfile;
  gamma)`. `write_npy(path, a)` writes `.npy` version 1.0.
- `tables.jl` holds the threaded builders `load_hotcross`, `build_hotcross_table`,
  `build_weight_table`, `build_zone_emission_table`, `zone_counts`, `build_int_jnu_table`
  and `zone_int_jnu_tables`. Each threaded loop writes disjoint columns or merges exact
  partial sums, so every table is independent of the thread count.

### 4.2 Execution

```julia
KernelAbstractions.@kernel function _foreach!(prog, @Const(items), n, outs...)   # the one kernel
    i = @index(Global)
    @inbounds i <= n && prog(items[i], i, outs...)
end

abstract type Executor end
ThreadedExecutor(; chunk = 4096)                       # Julia threads, host arrays
PoolExecutor(devices; lanes = 2, chunk = 65536); PoolExecutor(pool::DevicePool)
foreach_item!(exec, prog, items; outs::Tuple = (), order = nothing) -> outs
foreach_item!(exec, (prog64 => rescue, prog32 => bulk); outs) -> outs   # one queue, several programs
describe(exec) -> (kind, devices, lanes, chunk, threads)

abstract type RaySource end; ray(src, item) -> (y::SVector{8}, ok::Bool)
HostRays(rays); CameraRays(st, frame, gen)     # gen: PixelGrid(fov, npix) | BlockGrid(bs) | CoordList(α, β)
resident(pool_or_exec, dev, obj); device_program(pool_or_exec, dev, prog)
```

- **Programs.** A program is an `isbits`-after-`Adapt` functor called as
  `prog(item, i, outs...)`; it writes the outputs of `item` at `outs[k][i]` and returns
  `nothing`, or writes through atomics when it has no outputs. `i` is the item under
  threads and the position in the chunk on a device; after `foreach_item!`,
  `outs[k][item]` holds the outputs of `item`, bitwise the same whatever the executor,
  chunk size, device count, lane scheduling or `order`.
- **Threads.** `ThreadedExecutor` cuts the items into chunks of `chunk` and hands them to
  threads dynamically (`Threads.@threads :dynamic`); each item calls the host program with
  the host arrays. This is the only `Threads.@threads` over items.
- **Pool.** A `DevicePool` runs `lanes` persistent worker tasks per device pulling chunks
  from one shared queue. A chunk stages its item indices, launches the program's device
  twin once (after `prepare_launch!(dev, prog)`, which raises the CUDA stack to
  `stack_bytes(prog)`, at `workgroupsize(prog)`), synchronizes and copies the outputs back
  through host staging into `outs[k][item]`. Executors with the same devices, `lanes` and
  `chunk` share one pool of the process registry, closed at exit; a `DevicePool` built
  directly is the caller's to `close`.
- **Residency.** `DeviceAdaptor(pool, dev, readonly)` is the `Adapt` adaptor of a device
  twin: each `Array` leaf becomes its device copy, cached per pool and device by the
  identity of the host array and held weakly (a cached host array must not be mutated
  afterwards). A program is adapted once per device and call with `readonly = true`, its
  arrays behind `readonly(dev, a)` (`constify` on a `HostDevice`, the read-only data cache
  on a `CUDADevice`). A type whose twin is not its arrays copied defines
  `Adapt.adapt_structure(::DeviceAdaptor, x)`: the accumulators register zeroed twins; the
  worklist swaps the blocks of a `BlockGrid` per chunk.
- **Ray sources** are program fields: `CameraRays` calls `camera_ray` inside the program,
  so device rays equal host rays. **Scheduling** (`cost_order(curve, fov, npix; tile)`,
  `rescue_pixels(curve, fov, npix; rescue_px)`, `mixed_partition(items, rescue)`,
  `march_sort_order`, `curve_inclination`) is pure and evaluated by `prepare`.
- **Worklist.** `stream_blocks!(exec, prog, work, block, visit; outtype)` runs chunks of at
  most `max(1, chunk ÷ block²)` blocks; `visit(blk, vals)` returns the block's children,
  queued at once, with no level barrier. The visited set and every block's values do not
  depend on the executor or the completion order.
- **Accumulators.** `ExactBins(bands; nu_min = 1e8, nu_max = 1e16, n_ebins = 200)` over
  `PolarBands(n)` or `InclinationBands(incl_deg, half_width_deg)` holds per row and
  frequency bin the count and the limbs of `Σw` and `ΣwE`; `record!(bins, E, w, dir)`
  takes the escape direction and adds through atomics on either executor; `Spectrum(bins)`
  rounds each cell once and answers `band`, `nu_centers`, `nuLnu`, `nuLnu_total`,
  `nuLnu_solid`, `total_power`, `total_weight`, `mean_energy`. `ExactScalars(n)` has
  `add!`, `scalar`, `addends`. `ChildBuffer(T, capacity, layout)` stores children with
  their genealogy streams through `spawn!(buf, child, rng, event)`; a child beyond the
  capacity is counted (`overflow`) and the layer is rerun at a capacity of at least the
  count, which spawns identical children. `gather!(acc)` adds the device twins after
  `foreach_item!`; `Spectrum`, `scalar` and `children` throw while twins are pending. The
  accumulators are not `Sink`s: a Monte Carlo sink forwards `record!(sink, sph)` to
  `record!(bins, sph.E, sph.w, escape_direction(ctl, sph))` and `spawn!` to the buffer.
- **Devices.** `HostDevice(id)` (the KernelAbstractions CPU backend as a logical device)
  and `CUDADevice(ordinal)`, an integer meaning the latter. The protocol is `backend(dev)`,
  `activate!`, `functional`, `stage_alloc`/`stage_free`, `free!`, `prepare_launch!`,
  `readonly`, `upload` and `select_devices`; the fallbacks serve `HostDevice` and throw
  `BackendUnavailable` for other devices. The extension supplies the CUDA methods,
  page-locked staging freed explicitly and the NVML idle probe included.
- **Session state.** The backend session record and the pool registry are the package's
  only session state; both sit behind locks.

## 5. Pipeline tier

### 5.1 Plans

```julia
CameraConfig(; inclination_deg = 60.0, distance = 1000.0, fov = 40.0, resolution = 64)
PlasmaConfig(; mu = 0.5, ne_ni = 1.0, use_p = true, rat_low = 1.0, rat_high = 10.0,
             gamma = 13/9, gamma_i = 5/3, gamma_e = 4/3, thermal_frac = 1.0, power_frac = 0.0,
             kappa_frac = 0.0, power_p = 3.5, gamma_min = 25.0, gamma_max = 1e7, kappa = 4.0,
             w = 0.5, sigma_cut = 1.0, theta_e_min = -1.0)          # a negative cut is off
RunConfig(; m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1e24, frequency_hz = 230e9,
          camera = CameraConfig(), plasma = PlasmaConfig(), max_steps = 2500, C = 80.0,
          escape_factor = 1.1)

struct Lanes{Tg,Tt,A} end; Lanes() = Lanes{Float64,Float64,Float64}()   # Tt = A = Float64
MixedLanes(rescue_px = 2)       # Float32 bulk, Float64 within rescue_px of the critical curve
geodesic_type(lanes)            # Tg; Float32 for MixedLanes

transfer_params(cfg; mass_scale = cfg.mass_scale_cgs, rat_high = cfg.plasma.rat_high)
prepare(cfg::RunConfig; spacetime, matter, mode = IQUV(), lanes = Lanes(), tile = 16) -> Plan

struct Plan{S,Sg,P,M,T,Bg,H,L}
    config::RunConfig
    spacetime::S; geodesic_spacetime::Sg       # Float64 transfer lane; geodesic lane
    provider::P; mode::M; params::T            # matter or nothing; StokesI() | IQUV(); TransferParams
    camera::PlaneParallelCamera; frame::CameraFrame{Float64}; pixels::PixelGrid{Vector{Float64}}
    budget::MarchBudget{Float64}; geodesic_budget::Bg
    horizon::H; escape::Float64; R_em::Float64; theta_o::Float64
    curve::Tuple{Vector{Float64},Vector{Float64}}
    lanes::L; queue::Vector{Int}; rescue::Vector{Int}
end
```

The spacetime, the matter, the transfer mode and the lanes are values passed to `prepare`,
never configuration fields. `prepare` derives everything once: the camera, its frame and
pixel centres; the transfer parameters (units, electron model, the cuts
`sigma ≤ sigma_cut` and `theta_e ≥ theta_e_min` through `legacy_bound`, a population per
nonzero fraction); `escape = escape_factor × distance`; `R_em = domain_radius(matter)`,
`Inf` with `matter = nothing`; the march budget on both lanes; the horizon proxy (`NaN`
when the spacetime has none); `theta_o`; on Kerr, the critical curve at
`curve_inclination` and the cost-ordered queue (identity order elsewhere); the rescue set
of `MixedLanes`. It throws `ArgumentError` for a spacetime that is not `Float64`, a
geodesic lane the spacetime does not have, transfer or accumulator lanes other than
`Float64`, and `MixedLanes` off Kerr. A plan is immutable and backend-free.

### 5.2 Programs

```julia
struct ImagingProgram{M,S,C,P,T,F,R} <: Function
    mode::M; st::S; control::C; provider::P; params::T; frame::F; rays::R
end
Adapt.@adapt_structure ImagingProgram

function (p::ImagingProgram)(item, i, out)
    y0, ok = camera_state(p.rays, item)        # @noinline; the Float64 camera ray
    @inbounds out[i] = ok ? march(p.mode, lane_state(p.control, y0), p) : invalid(eltype(out))
    return nothing
end
workgroupsize(::ImagingProgram{IQUV}) = 64
```

- `imaging_program(plan; mode, params, gen)` sets `st = plan.spacetime`,
  `control = SpacetimeMarch(plan.geodesic_spacetime, plan.geodesic_budget)` (the `Float64`
  pair for the rescue lane) and `rays = CameraRays(plan.spacetime, plan.frame, gen)`:
  camera rays are made on the `Float64` spacetime and narrowed to the control's lane. A
  ray without a positive-energy null root gets `invalid(T)`, NaN in every component.
- The ray is computed in the non-inlined `camera_state`, and no program call operator
  carries `@inline`, so a program body is compiled as a function of its own and the
  threaded executor and the host-device pool run the same floating-point contractions.
- `TraceProgram(st, budget, rays, resume)` marches one window of at most
  `TRACE_WINDOW = 20` steps into per-item `TraceStep{T}(y, h)` outputs and a
  `TraceState{T}(y, r_prev, done, live, ok)` from which the next window resumes exactly;
  `TRACE_WINDOW + 2` output arrays keep the call below the argument count at which Julia
  boxes arguments. `ShadowProgram(st, budget, escape, rays)` records fate, step count,
  largest null error and final radius and state. `AnalyticProgram(gen, theta_o, a, ro,
  disk, bands)` lands every band of one screen point and needs a 32 KiB device stack.
- `StoredPathProgram` (a march along a stored trajectory) and `AdaptiveTraceProgram`
  (`trace_geodesic` with every step saved) hold host vectors and run on threads only.
  Every other program is `isbits` after `Adapt` and allocates nothing.

### 5.3 Products

| Entry point | Returns | Constraints |
|---|---|---|
| `render(plan; backend)` | `Image` | needs matter |
| `render(plan; path::Trajectories, backend)` | `Image` | threads only |
| `render(plan, coords; backend)` | `4 × n` `Matrix{Float64}`, rows I, Q, U, V | no `MixedLanes` |
| `render_amr(plan; block = 16, max_level = 2, refine, backend)` | `(image, levels, nrays)` | no `MixedLanes`; `resolution % block == 0` |
| `render_analytic(plan, disk; bands = (0, 1, 2), ro, backend)` | `(I, Ib, rs, ts, phis, redshift_sign, bands, plan, run)` | Kerr plan with the disk's spin |
| `photon_ring_layers(plan, disk; bands, dx, box, npts, ro, fudge, backend)` | `(I, Ib, masks, grid, N, box, dx, bands, plan, run)` | Kerr plan with the disk's spin |
| `trace(plan; backend, adaptive = false)` | `Trajectories` | no `MixedLanes`; `adaptive` on threads only |
| `shadow(plan; backend)` | `(fate, steps, null_error, radius, final, plan, run)` | no `MixedLanes` |
| `sensitivity(plan, param; backend)` | `(image, derivative, flux, dflux, param)` | `param` is `:mass_scale` or `:rat_high` |
| `variants(plan, mass_scales; backend)` | `(images, mass_scales, plan, run)` | grid providers |
| `mc_spectrum(plan, Ns; scattering, backend, bands, …)` | `Spectrum`, `ScatterResult` or `GRMHDSpectrumResult` | — |

- Determinism: every imaging product is bitwise per item, whatever the executor, device
  count, chunking or queue order; `render_amr`'s visited blocks, mosaic, level map and ray
  count are independent of the completion order; `mc_spectrum` is a bitwise reduction.
- Every product resolves `backend` with `resolve` (`nothing`: the session default) and
  records `run = (executor = describe(exec), lanes, version, seconds)` in its result, or in
  its `image` for `render_amr` and `sensitivity`; `render(plan, coords)` returns the bare
  matrix.
- A camera ray without a positive-energy null root is a NaN pixel in a render and an
  error in `trace` and `shadow`. `render(plan; path)` and `trace(…; adaptive = true)` throw
  `ArgumentError` unless the backend resolves to a `ThreadedExecutor`. The analytic
  products throw unless the plan's spacetime is a `KerrSchildCartesian` whose spin equals
  the disk's.
- `render_amr` paints each block into a mosaic at `resolution × 2^max_level` and builds its
  level map from per-level visited masks, without lock or atomic: blocks of one level own
  distinct cells, and a child is queued only after its parent's visit returns.
- `render` allocates one `output_type(plan)` per pixel and runs `march_frame!`:
  `imaging_program(plan)` over `plan.queue`, or under `MixedLanes` the parts
  `(rescue => q.rescue, bulk => q.bulk)` of `q = mixed_partition(plan.queue, plan.rescue)`
  in one queue. `sensitivity` and `variants` run it in `StokesI` mode with a `Dual`
  parameter set or a mass-scale tuple.

`mc_spectrum(plan, Ns; scattering, backend, bands, …)` is the Monte Carlo product. Its
programs write through the accumulators of `Execution`: escaped photons into `ExactBins`
(`PolarBands` for the sphere, `InclinationBands` for viewing inclinations), scattered
children into a `ChildBuffer` that feeds the next layer. `gather!` follows every
`foreach_item!` and `Spectrum(bins)` rounds once per cell, so the spectrum is identical on
every executor, thread count, device count and chunking.

### 5.4 Choosing CPU or GPU

Where a product runs is one keyword, `backend`, with a session default and an environment
override. A plan never records a backend: the same plan renders on the CPU and on the
GPUs, and device uploads are cached by the pool.

| Precedence | Form | Example |
|---|---|---|
| 1 | keyword on the product call | `render(plan; backend = :gpu)` |
| 2 | session default | `backend!(:gpu)`; `backend!([0, 1])`; `backend!(PoolExecutor([0, 1]; chunk = 2^15))`; `backend!(nothing)` clears it |
| 3 | `BLP_BACKEND`, read once at first use | `auto`, `cpu`, `gpu`, `gpu:0,1`, `host_pool`, `host_pool:2` |
| 4 | `:auto` | the pool over `select_devices()` when CUDA is loaded, functional and a device is idle, else threads |

`resolve` accepts `nothing`, `:auto`, `:cpu` (`ThreadedExecutor()`), `:gpu`, `:host_pool`
(a pool over `HostDevice(1)`), an `Executor`, a `DevicePool`, a device or an ordinal, a
list of devices or ordinals, and a string in the `BLP_BACKEND` grammar.

- `:auto` never downgrades silently: its first resolution in a session logs the choice
  and the reason (`CUDA not loaded`, `no functional device`, `no idle device`, or the
  devices used).
- An explicit choice is strict. `:gpu` without the CUDA extension, a functional device or
  a visible device throws `BackendUnavailable` naming the remedy (`using CUDA`,
  `CUDA_VISIBLE_DEVICES`, or `backend!(:cpu)`); its device selection is kept for the
  session. `backend!` resolves at once, so an unusable choice throws when it is set. A
  malformed `BLP_BACKEND` is an `ArgumentError` and is read again at the next use.
- `backend()` returns the resolved session default; an executor prints its kind, devices,
  lanes per device, chunk and Julia threads. `Execution.reset_backend!()`, a test hook,
  forgets the session default, the environment reading, the cached `:auto` and `:gpu`
  choices and the log; registered pools are kept.
- Device knobs (`lanes`, `chunk`) exist only on `PoolExecutor` and `DevicePool`;
  `ThreadedExecutor` has `chunk`. Precision lanes are a plan property, independent of the
  backend. Julia activates the extension only after the user's own `using CUDA`.

### 5.5 Analysis

`refinement.jl`: `RelativeGradient(cut = 0.15; frac = 0.03)`,
`RelativeLaplacian(cut = 1.0; frac = 0.03)`, `AbsoluteGradient(cut; count = 5)`,
`AnyOf(indicators...)`, `refinement_indicator(; …)` and `needs_refinement(indicator, I)`;
any callable `f(I) -> Bool` serves as an indicator. `lensing_bands.jl`: `polygon_area`,
`in_polygon`, `apparent_horizon`, `lensing_band`, `band_grid` on the Bardeen screen.
`interferometry.jl`: `tukey_window`, `radon_cut`, `visamp`, `visamp_minima`. Host code.

## 6. Continuity with the reference implementation

`BlackLightPlus`, the package in `src/`, is the reference implementation. The oracle suite
`BLP/test/oracle/` loads both packages, evaluates the reference device files into a
scratch module to run them on host devices, and compares BLP with the reference function
by function and product by product on the decks of `BLP/test/decks.jl`.

- **Bitwise.** Every `Numerics` function; every `Spacetime` method on Kerr, binaries, an
  ADM grid, a user metric and flat space (camera rays and exact-Kerr landings with their
  partials included); the `Data` samplers and readers, `decode` and `Analysis`; the
  analytic sources' scalars and validity; the transfer fits, tetrads and segment
  operators; the Monte Carlo emissivities, hot cross section, generation, Compton kick and
  sphere tracker with its children and streams. Stokes I images on every path: uniform
  grid, native blocks, coordinate lists, the superposed binary, slow light on both
  providers, mixed lanes, the mass-scale ladder, the `:mass_scale` sensitivity image, the
  exact-Kerr maps, photon-ring layers, AMR mosaics with their level maps and ray counts,
  and shadow fates and step counts.
- **Round-off natively, bitwise without FMA.** IQUV images, traces, the shadows' null
  errors and final states, the `:rat_high` sensitivity image, the transfer functions on
  `Dual` states and the Monte Carlo torus zones and marches agree to about 1e-12 to 1e-9
  relative. On FMA hardware LLVM fuses `muladd` pairs of the StaticArrays products into
  FMA instructions according to the inlining context, which differs between the two
  packages. The `transfer`, `products` and `montecarlo` scripts rerun themselves under
  `julia -C generic`, a CPU target without FMA (`BLP_ORACLE_NOFMA=1`), where each of these
  comparisons must be bitwise.
- **Round-off.** The analytic sources' four-vectors and hot-spot scalars, for the same
  cause: recompiled with the metric pair passed in, the reference body deviates alike.
- **The two-phase path.** BLP's CPU two-phase IQUV render transports the camera frame
  along the stored states, the fused march with the geodesic; the two differ at about
  2e-11 of the image maximum even without FMA. BLP's renders against the reference's
  two-phase CPU render, and the sensitivity derivatives, must agree within 1e-7 of the
  image maximum; their deviations are reported.
- **Exact sums.** Monte Carlo spectra are exact integer-limb sums: identical under 1, 3 and
  8 threads, several chunkings and pools of 1, 2, 3 and 5 host devices.

Run the suite from the repository root:

```
julia --project=BLP/test/oracle -e 'using Pkg; Pkg.instantiate()'
julia -t 8 --project=BLP/test/oracle BLP/test/oracle/runtests.jl
```

Its environment develops both packages (`BLP/` and the repository root). The comparisons
with the gpumonty hot-cross table read `hotcross.dat` from the path in `BLP_REF_GPUMONTY`,
else from a `Mahakala` checkout beside the repository, and skip when it is absent.

## 7. Public API

Every name of `names(BLP)`. The keyword `backend` of a product defaults to `nothing`.

| Name | Signature | Result |
|---|---|---|
| `CameraConfig` | `CameraConfig(; inclination_deg, distance, fov, resolution)` | camera record (§5.1) |
| `PlasmaConfig` | `PlasmaConfig(; mu, ne_ni, use_p, rat_low, rat_high, …, sigma_cut, theta_e_min)` | electron and emission record (§5.1) |
| `RunConfig` | `RunConfig(; m_bh_cgs, mass_scale_cgs, frequency_hz, camera, plasma, max_steps, C, escape_factor)` | run record (§5.1) |
| `Lanes` | `Lanes{Tg,Tt,A}()`, `Lanes()` | precision lanes, `Tt = A = Float64` |
| `MixedLanes` | `MixedLanes(rescue_px = 2)` | mixed lanes on Kerr |
| `prepare` | `prepare(cfg; spacetime, matter, mode = IQUV(), lanes = Lanes(), tile = 16)` | `Plan` |
| `KerrSchildCartesian` | `KerrSchildCartesian(a)` | `KerrSchildCartesian{T}` |
| `Minkowski` | `Minkowski()` | `Minkowski{Float64}` |
| `UserMetric` | `UserMetric(g)` | `UserMetric{F,Float64}` |
| `SuperposedKerrSchild` | `SuperposedKerrSchild(masses, spins, centers)` | `SuperposedKerrSchild{N,Float64}` |
| `ADMGrid` | `ADMGrid(xs, ys, zs, data; alpha_min = 0.1, offgrid_eta = true)`, `ADMGrid(st; alpha_min)` | `ADMGrid{Float64,Array{Float64,4}}` |
| `adm_grid_from_metric` | `adm_grid_from_metric(metric_fn, xs, ys, zs; alpha_min = 0.1, offgrid_eta = true, excise = nothing)` | `ADMGrid` |
| `z4c_grid_spacetime` | `z4c_grid_spacetime(binfile, half, n; alpha_min = 0.15)` | `(ADMGrid, AthenaKBin)` |
| `AthenaKSnapshot` | `AthenaKSnapshot(filename; gamma = nothing)`, `AthenaKSnapshot(data, lo, hi, x_left, dx, levels, logical_locations; names, time)` | grid provider |
| `UniformGrid` | `UniformGrid(data, box_half)`, `UniformGrid(p::GridProvider, box_half, n)` | grid provider |
| `SnapshotSeries` | `SnapshotSeries(paths, times; maxslots = 2, t_obs = times[end])` | series of snapshots |
| `FrameWindow` | `FrameWindow(a, b, t_a, t_b; t_obs)` | grid provider |
| `SeriesProvider` | `SeriesProvider(series)` | host grid provider |
| `PolishDoughnut` | `PolishDoughnut(a; l0, lambda, r_center, n_e_center, theta_e_center, …)` | analytic provider |
| `PageThorneDisk` | `PageThorneDisk(a; mdot, r_out, n_e_max, theta_e_max, beta, scale_height, prograde)` | analytic provider |
| `EquatorialHotspot` | `EquatorialHotspot(a; r_orbit, t0, phi0, R_spot, n_e0, theta_e0, B0, …)` | analytic provider |
| `EquatorialDisk` | `EquatorialDisk(a; sub_kep, betar, betaphi, gamma_p, mu_p, sigma_p, redshift_power, rin)` | g-disk of the exact-Kerr products |
| `UniformSphere` | `UniformSphere(; n_e = 1e13, theta_e = 100.0, B = 1.0, R = 1.0)` | analytic provider |
| `StokesI`, `IQUV` | `StokesI()`, `IQUV()` | transfer modes |
| `ThreadedExecutor` | `ThreadedExecutor(; chunk = 4096)` | `Executor` |
| `PoolExecutor` | `PoolExecutor(devices; lanes = 2, chunk = 65536)`, `PoolExecutor(pool)` | `Executor` |
| `DevicePool` | `DevicePool(devices; lanes = 2, chunk = 65536)` | pool, closed with `close` |
| `HostDevice` | `HostDevice(id)` | device |
| `CUDADevice` | `CUDADevice(ordinal)` | device |
| `select_devices` | `select_devices(; candidates, util_threshold = 15, min_free_gb = 8, fallback = false)` | `Vector{CUDADevice}` |
| `backend` | `backend()`; `backend(dev)` | the session `Executor`; the device's KernelAbstractions backend |
| `backend!` | `backend!(b)` | the new session `Executor`, or `nothing` |
| `render` | `render(plan; backend, path = nothing)`; `render(plan, coords; backend)` | `Image`; `Matrix{Float64}` |
| `render_amr` | `render_amr(plan; block = 16, max_level = 2, refine = refinement_indicator(), backend)` | `NamedTuple` |
| `render_analytic` | `render_analytic(plan, disk; bands = (0, 1, 2), ro = plan.camera.d, backend)` | `NamedTuple` |
| `photon_ring_layers` | `photon_ring_layers(plan, disk; bands = (0, 1, 2), dx = 0.02, box = 12.0, npts = 100, ro, fudge = 1.0, backend)` | `NamedTuple` |
| `trace` | `trace(plan; backend, adaptive = false)` | `Trajectories` |
| `shadow` | `shadow(plan; backend)` | `NamedTuple` |
| `sensitivity` | `sensitivity(plan, param::Symbol; backend)` | `NamedTuple` |
| `variants` | `variants(plan, mass_scales; backend)` | `NamedTuple` |
| `mc_spectrum` | `mc_spectrum(plan, Ns; scattering, backend, bands, …)` | `Spectrum`, `ScatterResult` or `GRMHDSpectrumResult` |
| `Image` | fields `I`, `Q`, `U`, `V` (`Matrix{Float64}`, `[iβ, iα]`), `plan`, `run` | result type |
| `Trajectories` | fields `states`, `dls` (source → camera, `Float64`), `plan`, `run` | result type |
| `Spectrum` | `Spectrum(bins::ExactBins)` | rounded spectrum (§4.2) |
| `ScatterResult` | — | result of a scattering `mc_spectrum` run |
| `GRMHDSpectrumResult` | — | result of a GRMHD `mc_spectrum` run |
| `write_npy` | `write_npy(path, a::AbstractMatrix{Float64})` | `path` |
| `visamp` | `visamp(img; angles = (0.0, 90.0), fov_muas, pad = 16, maxbaseline = Inf, window = :tukey)` | `Vector` of `(angle, baselines, amplitude)` |
| `visamp_minima` | `visamp_minima(baselines, amplitude; skip = 1, bl_min = 0.0)` | `Vector{Float64}` |
| `radon_cut` | `radon_cut(img, angle_deg; fov_muas)` | `(projection, s_axis)` |
| `tukey_window` | `tukey_window(n, α = 0.5)` | `Vector{Float64}` |
| `lensing_band` | `lensing_band(a, θo, n; npts = 100, box = 25.0)` | `(inner, outer)` |
| `critical_curve` | `critical_curve(a, θo; n = 100)` | `(α, β)` vectors |

## 8. File map

Paths relative to `BLP/src/` (the extension: `BLP/ext/`). The role is the file's header
comment or module docstring.

| File | Role | Main definitions |
|---|---|---|
| `BLP.jl` | The package: the ten modules in dependency order | the 54 exports (§9) |
| `physics/Numerics/Numerics.jl` | Number-type-generic building blocks shared by every tier | includes, exports |
| `physics/Numerics/constants.jl` | CGS physical constants | `CL`, `ME`, `HPL`, `SIGMA_THOMSON`, … |
| `physics/Numerics/units.jl` | Code-unit to cgs conversions of a GRMHD model | `Units` |
| `physics/Numerics/raystate.jl` | The eight-component ray state `[x^μ; p_μ]` | `RayState`, `position`, `momentum`, `raystate` |
| `physics/Numerics/stencils.jl` | Trilinear blend, Catmull-Rom stencil, log-linear lookup | `lerp8`, `corner_weights`, `cr_axis`, `loglinear_at` |
| `physics/Numerics/exactsum.jl` | Order-independent exact summation in integer limbs | `xsum_add!`, `xsum_atomic_add!`, `xsum_round` |
| `physics/Numerics/philox.jl` | Counter-based random numbers and the Monte Carlo samplers | `MCRng`, `MCStreamLayout`, `child_rng` |
| `physics/Numerics/special.jl` | Special functions of the synchrotron fits | `bessel_k2`, `hypergeometric_2f1` |
| `physics/Spacetime/Spacetime.jl` | Spacetimes and null geodesics | includes, exports |
| `physics/Spacetime/protocol.jl` | The spacetime protocol; every method but `metric` defaulted | `Spacetime`, `metric`, `lane`, `horizon_radius` |
| `physics/Spacetime/kerr_schild.jl` | Kerr in Cartesian Kerr-Schild coordinates, and flat space | `KerrSchildCartesian`, `Minkowski`, `ks_radius` |
| `physics/Spacetime/user_metric.jl` | A spacetime from a user closure | `UserMetric` |
| `physics/Spacetime/superposed.jl` | Superposed Kerr-Schild holes, Woodbury inverse | `SuperposedKerrSchild` |
| `physics/Spacetime/adm_grid.jl` | A frozen-slice numerical spacetime on a Cartesian grid | `ADMGrid`, `sample_adm`, `adm_grid_from_metric` |
| `physics/Spacetime/metric_ops.jl` | Metric derivatives by forward-mode AD | `inverse_metric_gradient`, `christoffel` |
| `physics/Spacetime/geodesic.jl` | Hamiltonian null geodesics in covariant-momentum form | `geodesic_rhs`, `rk4_step`, `null_covariant` |
| `physics/Spacetime/termination.jl` | The fixed-control march: step size and the one termination rule | `MarchBudget`, `MarchStep`, `march_control` |
| `physics/Spacetime/adaptive.jl` | Adaptive geodesic solves with event termination (host) | `trace_geodesic` |
| `physics/Spacetime/camera.jl` | Plane-parallel camera | `PlaneParallelCamera`, `CameraFrame`, `camera_ray` |
| `physics/Spacetime/kerr_orbits.jl` | Exact-Kerr orbital primitives | `kerr_horizon`, `isco_radius`, `keplerian_omega` |
| `physics/Spacetime/analytic_kerr.jl` | Exact-Kerr equatorial-emission ray tracing | `pixel_prep`, `band_landing`, `Landing`, `critical_curve` |
| `physics/Matter/Matter.jl` | Plasma as the transfer coefficients see it | includes, exports |
| `physics/Matter/records.jl` | The primitive and state records | `PlasmaPrimitives`, `PlasmaState`, `vacuum` |
| `physics/Matter/electrons.jl` | Electron model and plasma cuts | `ElectronModel`, `Bound`, `PlasmaCuts`, `omitted` |
| `physics/Matter/decode.jl` | Code-unit primitives to the cgs plasma state | `primitive_four_vectors`, `decode` |
| `physics/Matter/protocol.jl` | The plasma provider protocol | `PlasmaProvider`, `GridProvider`, `sample` |
| `physics/Matter/sources/shared.jl` | Kinematics shared by the analytic sources | module-private helpers |
| `physics/Matter/sources/polish_doughnut.jl` | Constant-angular-momentum equilibrium torus | `PolishDoughnut`, `torus_potential` |
| `physics/Matter/sources/page_thorne.jl` | Page & Thorne thin-disk flux | `PageThorneDisk`, `page_thorne_flux` |
| `physics/Matter/sources/hotspot.jl` | Orbiting Gaussian hot spot | `EquatorialHotspot`, `hotspot_center` |
| `physics/Matter/sources/gdisk.jl` | The equatorial g-disk | `EquatorialDisk`, `intensity`, `gfactor_disk` |
| `physics/Matter/zones.jl` | Monte Carlo emission zones and the uniform sphere | `EmissionZone`, `UniformSphere`, `emission_zone` |
| `physics/Transfer/Transfer.jl` | Radiative transfer along a ray | includes, exports |
| `physics/Transfer/params.jl` | Transfer modes and the parameter set of the marches | `StokesI`, `IQUV`, `PowerLaw`, `Kappa`, `TransferParams` |
| `physics/Transfer/synchrotron.jl` | Synchrotron coefficients of three populations | `thermal_synchrotron`, `kappa_synchrotron`, … |
| `physics/Transfer/tetrads.jl` | Orthonormal frames of the polarized transfer | `fluid_tetrad`, `fluid_tetrad_checked`, `qu_rotation` |
| `physics/Transfer/coefficients.jl` | Invariant transfer coefficients of one decoded sample | `coefficients` |
| `physics/Transfer/segments.jl` | Segment operators of constant-coefficient transfer | `segment`, `couple`, `stokes_admissible` |
| `physics/Transfer/controls.jl` | March controls: how a march advances and when it stops | `SpacetimeMarch`, `StoredPath`, `geodesic_step` |
| `physics/Transfer/march.jl` | The reverse (camera → source) transfer marches | `march` |
| `physics/MonteCarlo/MonteCarlo.jl` | Monte Carlo Comptonization | includes, exports |
| `physics/MonteCarlo/superphoton.jl` | The Monte Carlo packet | `Superphoton` |
| `physics/MonteCarlo/emissivity.jl` | Emission-frame physics of superphoton generation | `j_nu`, `alpha_nu`, `emission_tetrad` |
| `physics/MonteCarlo/hotcross.jl` | The thermally averaged Compton cross section | `HotcrossTable`, `alpha_inv_scatt` |
| `physics/MonteCarlo/tables.jl` | Generation table records and lookups | `WeightTable`, `ZoneEmissionTable`, `IntJnuTable` |
| `physics/MonteCarlo/generate.jl` | Superphoton generation from an emission zone | `GenerationTables`, `generate` |
| `physics/MonteCarlo/scatter.jl` | The Compton kick | `scatter_superphoton` |
| `physics/MonteCarlo/opacity.jl` | Opacity rates, the scattering bias, the opacity record | `rate_frame`, `Bias`, `Opacities` |
| `physics/MonteCarlo/sinks.jl` | The sink protocol | `Sink`, `record!`, `spawn!`, `tally!` |
| `physics/MonteCarlo/track.jl` | The one superphoton tracker and its two march controls | `FlatMarch`, `CurvedMarch`, `track!` |
| `infra/Data/Data.jl` | Containers and files | includes, exports |
| `infra/Data/fluid_blocks.jl` | AthenaK meshblocks as a grid provider | `FluidBlocks`, `LeafTable`, `LevelHash`, `BlockScan` |
| `infra/Data/athenak_hdf5.jl` | AthenaK `.athdf` snapshots | `AthenaKSnapshot`, `same_mesh`, `share_mesh` |
| `infra/Data/athenak_bin.jl` | AthenaK `.bin` dumps: reader, Z4c to ADM, GRMHD adapter | `read_athenak_bin`, `resample_z4c_adm` |
| `infra/Data/uniform_grid.jl` | The fluid resampled onto a uniform node grid | `UniformGrid`, `resample` |
| `infra/Data/series.jl` | Slow light: series, two-frame windows, series provider | `SnapshotSeries`, `FrameWindow`, `SeriesProvider` |
| `infra/Data/npy.jl` | NumPy `.npy` writer | `write_npy` |
| `infra/Data/tables.jl` | Host builders of the Monte Carlo tables | `load_hotcross`, `build_weight_table`, … |
| `infra/Execution/Execution.jl` | Where programs run | includes, exports |
| `infra/Execution/devices.jl` | Device handles and the device protocol | `HostDevice`, `CUDADevice`, `BackendUnavailable` |
| `infra/Execution/pool.jl` | The device pool: lanes, shared queue, registry | `DevicePool` |
| `infra/Execution/residency.jl` | A host object as a device sees it | `DeviceAdaptor`, `resident`, `device_program` |
| `infra/Execution/kernel.jl` | The only kernel of the package | `_foreach!`, `workgroupsize` |
| `infra/Execution/executor.jl` | Executors: run a program over items, gather outputs | `ThreadedExecutor`, `PoolExecutor`, `foreach_item!` |
| `infra/Execution/backend.jl` | The CPU/GPU switch | `resolve`, `backend!`, `backend` |
| `infra/Execution/ray_sources.jl` | The initial state of an item's ray, made in the program | `CameraRays`, `PixelGrid`, `BlockGrid`, `CoordList` |
| `infra/Execution/schedule.jl` | Scheduling geometry of full-frame renders | `cost_order`, `rescue_pixels`, `mixed_partition` |
| `infra/Execution/worklist.jl` | The image-block worklist of adaptive refinement | `ImageBlock`, `stream_blocks!` |
| `infra/Execution/accumulators.jl` | Exact bins, exact scalars and the child buffer | `ExactBins`, `Spectrum`, `ChildBuffer`, `gather!` |
| `pipeline/Plans/Plans.jl` | Run configuration, precision lanes and `prepare` | includes, exports |
| `pipeline/Plans/config.jl` | Run configuration | `CameraConfig`, `PlasmaConfig`, `RunConfig` |
| `pipeline/Plans/lanes.jl` | Precision lanes | `Lanes`, `MixedLanes`, `geodesic_type` |
| `pipeline/Plans/plan.jl` | The plan | `Plan` |
| `pipeline/Plans/prepare.jl` | Every derived quantity of a run, computed once | `transfer_params`, `prepare` |
| `pipeline/Analysis/Analysis.jl` | Image-plane analysis | includes, exports |
| `pipeline/Analysis/refinement.jl` | Block refinement indicators | `refinement_indicator`, `needs_refinement` |
| `pipeline/Analysis/lensing_bands.jl` | Screen geometry of photon-ring imaging | `lensing_band`, `band_grid` |
| `pipeline/Analysis/interferometry.jl` | Visibility amplitudes along baselines | `radon_cut`, `visamp` |
| `pipeline/Products/Products.jl` | Programs built from a plan and the entry points | includes, exports |
| `pipeline/Products/programs.jl` | The per-item programs the executors run | `ImagingProgram`, `TraceProgram`, `ShadowProgram` |
| `pipeline/Products/image.jl` | Full-frame and coordinate renders, the run record | `Image`, `render` |
| `pipeline/Products/trace.jl` | Trajectories, ray fates, the two-phase render | `Trajectories`, `trace`, `shadow` |
| `pipeline/Products/sensitivity.jl` | Parameter derivatives and ladders of Stokes I | `sensitivity`, `variants` |
| `pipeline/Products/analytic.jl` | Exact-Kerr images of an equatorial disk | `render_analytic`, `photon_ring_layers` |
| `pipeline/Products/amr.jl` | Adaptive image-plane refinement | `render_amr` |
| `BLPCUDAExt.jl` (`ext/`) | CUDA methods of the device protocol for `CUDADevice` | extension module |

## 9. Exports and internal interfaces

`BLP.jl` exports the 54 names of §7, in its groups: configuration, spacetimes, matter,
transfer modes, execution, products, results, analysis and I/O. They are the public API,
under semantic versioning; the gate allows at most 54 names and at most 10 in the
`render`/`trace`/`shadow`/`mc_spectrum`/`sensitivity`/`variants`/`photon_ring_layers`
family. Each submodule exports its internal interface: the method tables of §3 and §4,
the programs of §5.2, and `Plan`, `transfer_params` and `geodesic_type` of `Plans`. Every
name a submodule exports is defined and has a docstring. Tests import from the
submodules (`using BLP.Transfer: segment`); no test or oracle script reaches a
leading-underscore name.

## 10. Tests

| Path (`BLP/test/`) | Contents |
|---|---|
| `runtests.jl` | includes `structure.jl`, then every file under `physics/`, `infra/` and `pipeline/`, each into its own module |
| `decks.jl` | module `Decks`: the synthetic uniform-grid and block decks, the reference configuration, `bits`, camera rays, sampled trajectories, host-device lists, `device_isbits`, the Monte Carlo sinks, programs and sphere runs |
| `structure.jl` | the gates of §1 and §9 (`DESIGN_PRINCIPLES.md` §5) |
| `physics/` | `numerics_*`, `spacetime_*`, `matter_*`, `transfer_*`, `montecarlo_*`: closed forms, conservation, AD through each layer, the hot-path contract |
| `infra/` | `data_*` (readers on synthetic files, locates, grids, series, tables), `execution_*` (executors bitwise against one another, pool, residency, ray sources, schedule, worklist, accumulators, backend switch, CUDA extension) |
| `pipeline/` | `analysis_*`, `plans_prepare`, `products_*` (render, ray orientation and Stokes convention, trace, AMR, analytic, sensitivity, programs, backend) |
| `oracle/` | its own environment, developing BLP and BlackLightPlus: `runtests.jl`, `helpers.jl` and the comparison scripts (§6) |

Run the main suite with `julia -t auto --project=BLP -e 'using Pkg; Pkg.test()'`. Device
code runs on `HostDevice` pools. A test that needs a missing resource (the CUDA package,
the gpumonty table) skips inside its file and reports the skip.
