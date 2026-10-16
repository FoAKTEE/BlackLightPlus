# BlackLightPlus next-generation architecture

**Status:** proposal for review. Applies the principles of
`docs/DESIGN_PRINCIPLES.md` (P1–P18) to a concrete code structure.
**Assumptions:** the nine decisions of `DESIGN_PRINCIPLES.md` §6 are taken as
recommended: GPU code in the package behind a CUDA extension, a product-shaped
public API, the fused march as the default CPU path, bitwise continuity where
the arithmetic order is unchanged and documented tolerance elsewhere, a
redesigned configuration with a one-release shim, diagnostics out of the
package, layer-vocabulary module names, incremental migration, and provenance
prose moved to `method.md`. Each is reversible; §12 lists what changes if one
is reversed.

---

## 1. Three tiers, one direction

```
pipeline/   Plans · Products · Analysis            composition: configuration, preparation,
                                                   products, image-plane analysis
    │  uses both tiers below
infra/      Data · Execution   (+ ext/CUDA)        machinery: file formats, fluid containers,
                                                   executors, device pool, residency, scheduling
    │  implements the protocols of, and uses, the tier below
physics/    Numerics · Spacetime · Matter ·        equations: pure, isbits, StaticArrays,
            Transfer · MonteCarlo                  no arrays of its own, no threads, no I/O
```

Rules.

- A module imports only from its own tier or a lower tier, and within a tier
  only from the modules included before it: `Numerics → Spacetime → Matter →
  Transfer → MonteCarlo`; `Data → Execution`; `Plans → Analysis → Products`
  (`Products` uses the refinement indicators of `Analysis`).
- The physics tier defines protocols (`Spacetime`, `PlasmaProvider`,
  `TransferMode`, `MarchControl`, `Sink`). The infra tier supplies concrete
  providers (fluid blocks, grids, windows) and the machinery that runs
  per-item programs. The pipeline tier builds programs from a plan and hands
  them to an executor.
- Nothing imports `ext/`. The extension adds methods to `Execution` functions
  for the `CUDADevice` handle.
- There is exactly one `@kernel` in the package (`Execution/kernel.jl`). Every
  device computation is a per-item program run through it.

## 2. Directory layout

```
src/
├── BlackLightPlus.jl                 includes the tiers in order; `export` of the public API (§9)
├── physics/
│   ├── Numerics/
│   │   ├── Numerics.jl
│   │   ├── constants.jl              CGS constants                                  ← constants.jl
│   │   ├── units.jl                  `Units{T}` record and `units(m_bh, mass_scale)` ← athenak_reader.jl
│   │   ├── raystate.jl               `RayState{T}`, `position`, `momentum`, `raystate`
│   │   ├── stencils.jl               trilinear lerp on 8 corners, Catmull-Rom weights,
│   │   │                             log-linear table lookup                        ← athenak_reader, numerical_spacetime, mc_common
│   │   ├── exactsum.jl               limbs, `xsum_parts`, host and atomic add, round ← exact_sum.jl, mc_gpu.jl
│   │   ├── philox.jl                 Philox4x32, `MCRng`, stream identity, samplers  ← rng.jl
│   │   └── special.jl                A&S Bessel fits, host 2F1                       ← coefficients.jl
│   ├── Spacetime/
│   │   ├── Spacetime.jl
│   │   ├── protocol.jl               abstract types, method table (§3.2), lane conversion
│   │   ├── kerr_schild.jl            `_ks_term`, `KerrSchildCartesian`, `Minkowski`  ← metrics.jl
│   │   ├── user_metric.jl            `UserMetric`                                    ← metrics.jl
│   │   ├── superposed.jl             `SuperposedKerrSchild`, Woodbury inverse         ← spacetimes.jl
│   │   ├── adm_grid.jl               `ADMGrid{T,A}` (one struct for host and device), builder ← numerical_spacetime.jl, gpu_spacetime.jl
│   │   ├── metric_ops.jl             AD gradients, Christoffel, index-op defaults    ← metrics.jl
│   │   ├── geodesic.jl               RHS, transport RHS, RK4 step, null root, energies ← integrator.jl
│   │   ├── termination.jl            `MarchBudget`, `march_stop`, `march_control` per type ← integrator.jl, spacetimes.jl, numerical_spacetime.jl
│   │   ├── adaptive.jl               `trace_geodesic` (OrdinaryDiffEq), host only    ← integrator.jl, spacetimes.jl, numerical_spacetime.jl
│   │   ├── camera.jl                 `PlaneParallelCamera`, `CameraFrame` (with `up`), `camera_ray` ← camera.jl, render.jl `_camera_up`
│   │   ├── kerr_orbits.jl            unchanged
│   │   └── analytic_kerr.jl          `pixel_prep`, `band_landing`, `Landing`, `critical_curve` ← analytic_kerr.jl, lensing_bands.jl
│   ├── Matter/
│   │   ├── Matter.jl
│   │   ├── records.jl                `PlasmaPrimitives`, `PlasmaState`, `vacuum`
│   │   ├── electrons.jl              `ElectronModel`, `PlasmaCuts`, `electron_theta`, `omitted` ← coefficients.jl
│   │   ├── decode.jl                 `primitive_four_vectors`, `decode`             ← primitives.jl, coefficients.jl `plasma_decode`
│   │   ├── protocol.jl               `PlasmaProvider`, `plasma_state` default method
│   │   ├── sources/
│   │   │   ├── shared.jl             Killing scalars, circular velocity, toroidal field ← sources.jl
│   │   │   ├── polish_doughnut.jl    unchanged physics, new signature
│   │   │   ├── page_thorne.jl
│   │   │   ├── hotspot.jl
│   │   │   └── gdisk.jl              `EquatorialDisk`, g-factors, Johnson-SU, `intensity(disk, landing)` ← radiation/equatorial_source.jl
│   │   └── zones.jl                  `EmissionZone`, `UniformSphere` provider        ← emission_tables.jl, sphere_model.jl
│   ├── Transfer/
│   │   ├── Transfer.jl
│   │   ├── params.jl                 `Thermal`, `PowerLaw`, `Kappa` (precompute in constructors), `TransferParams` ← coefficients.jl, render.jl
│   │   ├── synchrotron.jl            thermal / power-law / kappa fits, unpolarized and polarized ← coefficients.jl
│   │   ├── tetrads.jl                `fluid_tetrad(_checked)`, `normal_observer`, `qu_rotation` ← tetrads.jl, reverse_march.jl, integrator.jl
│   │   ├── coefficients.jl           `coefficients(::StokesI, …)`, `coefficients(::IQUV, …)` ← integrator.jl
│   │   ├── segments.jl               segment operators, admissibility                ← transfer.jl
│   │   ├── controls.jl               `SpacetimeMarch{S,T}`, `StoredPath`, `march_control` ← reverse_march.jl (KerrMarch folded in)
│   │   └── march.jl                  `march(::StokesI, …)`, `march(::IQUV, …)`        ← reverse_march.jl
│   └── MonteCarlo/
│       ├── MonteCarlo.jl
│       ├── superphoton.jl            unchanged
│       ├── emissivity.jl             `j_nu`, `alpha_nu`, `int_jnu_zone`, `emission_tetrad` ← emissivity.jl
│       ├── tables.jl                 `WeightTable`, `IntJnuTable` records and lookups (builders are infra) ← emission_tables.jl
│       ├── hotcross.jl               KN, boosted cross section, Maxwell–Jüttner, `HotcrossTable` lookup ← hotcross.jl
│       ├── generate.jl               `generate(zone, tables, rng)`                   ← generate.jl, sphere_model.jl, mc_gpu.jl
│       ├── scatter.jl                Compton kick                                    ← scatter.jl
│       ├── opacity.jl                `absorption_rate`, `scattering_rate`, `Bias`     ← track.jl, mc_cpu.jl, mc_gpu.jl (6 copies → 2)
│       ├── sinks.jl                  `Sink` protocol: `record!`, `spawn!`
│       └── track.jl                  `track!` (one march), `FlatMarch` control        ← track.jl, mc_common.jl, mc_cpu.jl, mc_gpu.jl (6 trackers → 1)
├── infra/
│   ├── Data/
│   │   ├── Data.jl
│   │   ├── fluid_blocks.jl           `FluidBlocks` provider, locate strategies, `primitives`, Adapt ← athenak_reader.jl, gpu_amr.jl, gpu_spacetime.jl
│   │   ├── athenak_hdf5.jl           `.athdf` → `AthenaKSnapshot`                    ← athenak_reader.jl
│   │   ├── athenak_bin.jl            `.bin` reader, GRMHD adapter, Z4c → `ADMGrid`    ← athenak_z4c.jl
│   │   ├── uniform_grid.jl           `UniformGrid` provider, `resample`               ← gpu_image.jl
│   │   ├── series.jl                 `SnapshotSeries` (LRU), `FrameWindow`, `SeriesProvider`, `blend` ← time_series.jl, gpu_image.jl, gpu_amr.jl
│   │   ├── tables.jl                 threaded builders of weight / zone / int-jnu tables, hotcross load and build ← emission_tables.jl, hotcross.jl
│   │   └── npy.jl                    `write_npy`                                     ← outputs.jl
│   └── Execution/
│       ├── Execution.jl
│       ├── devices.jl                `HostDevice`, `CUDADevice`, `backend`, `upload`, staging hooks ← gpu_pool.jl
│       ├── executor.jl               `Executor`, `ThreadedExecutor`, `PoolExecutor`, `foreach_item!`
│       ├── backend.jl                the CPU/GPU switch: `resolve`, `backend!`, `backend`, `BLP_BACKEND` (§5.4)
│       ├── kernel.jl                 the one `@kernel`
│       ├── pool.jl                   `DevicePool`, lanes, jobs, `enqueue!`            ← gpu_pool.jl
│       ├── residency.jl              `resident`, per-device adapted twins             ← gpu_pool.jl, gpu_image.jl, gpu_amr.jl, gpu_spacetime.jl
│       ├── ray_sources.jl            `RaySource`: `HostRays`, `CameraRays{PixelGrid|BlockGrid|CoordList}` ← gpu_camera.jl, gpu_pool.jl
│       ├── accumulators.jl           `ExactBins` (atomic), `ChildBuffer`, merges      ← exact_sum.jl, spectrum.jl, mc_gpu.jl, track.jl
│       ├── schedule.jl               `cost_order`, `rescue_pixels`, `mixed_partition`, `march_sort_order` ← gpu_image.jl
│       └── worklist.jl               `stream_blocks!` over any executor              ← amr.jl, gpu_amr.jl
├── pipeline/
│   ├── Plans/
│   │   ├── Plans.jl
│   │   ├── config.jl                 `CameraConfig`, `PlasmaConfig`, `RunConfig`, deprecation shim ← inputs/parameters.jl
│   │   ├── lanes.jl                  `Lanes{Tg,Tt,A}`, `MixedLanes`
│   │   ├── plan.jl                   `Plan` record
│   │   └── prepare.jl                `prepare(cfg; spacetime, matter, mode, lanes)` ← render.jl `_pipeline_setup`/`_scc_config`/`_prepared_scc`, every `_g*_params`
│   ├── Analysis/
│   │   ├── Analysis.jl
│   │   ├── refinement.jl             block indicators                                ← amr.jl `_block_needs_refinement`
│   │   ├── lensing_bands.jl          apparent horizon, band edges, band grids         ← geodesics/lensing_bands.jl
│   │   └── interferometry.jl         unchanged                                       ← outputs/interferometry.jl
│   └── Products/
│       ├── Products.jl
│       ├── programs.jl               `ImagingProgram`, `TraceProgram`, `ShadowProgram`, `AnalyticProgram`, `MCPrimary`, `MCLayer`
│       ├── image.jl                  `Image`, `render(plan)`, `render(plan, coords)`  ← render.jl, gpu_image/pol/amr/spacetime.jl
│       ├── amr.jl                    `render_amr`                                    ← amr.jl, gpu_amr.jl `AMRStokes`
│       ├── analytic.jl               `render_analytic`, `photon_ring_layers`         ← render.jl, gpu_analytic.jl
│       ├── trace.jl                  `trace`, `shadow`, `Trajectories`               ← integrator.jl, gpu_tracer.jl, gpu_spacetime.jl
│       ├── sensitivity.jl            `sensitivity`, `variants`                       ← gpu_image_ad.jl, gpu_image.jl kvariant
│       └── montecarlo.jl             `emission_zones`, `mc_spectrum`, results        ← mc_cpu.jl, mc_gpu.jl, grmhd_model.jl
ext/
└── BlackLightPlusCUDAExt.jl         `CUDADevice` methods: activate, `CUDABackend`, pinned staging,
                                      NVML probe for `select_devices`, stack limit, `unsafe_free!`
```

Estimated size after migration: 9,000–10,000 lines against 13,831 today. The
reduction is the driver prologues, the kernel enumeration, the sampler and
tracker copies, and the provenance prose (§6, §7, §8).

## 3. Physics tier

Everything here is pure, allocation-free, `isbits` (or `isbits` after
`Adapt`), `StaticArrays`-typed and number-type generic. No module here imports
`KernelAbstractions`, `HDF5`, `Threads` or `CUDA`. `Adapt` is used only to
declare structure.

### 3.1 Numerics

| Item | Signature | Replaces |
|---|---|---|
| Ray state | `const RayState{T} = SVector{8,T}`; `position(y)`, `momentum(y)`, `raystate(x, p)` | 17 sites of `SVector(y[1], y[2], y[3], y[4])` |
| Trilinear | `lerp8(c000, …, c111, d1, d2, d3)`; `corner_weights(f) -> (i0, d)` | four interpolants |
| Catmull-Rom | `cr_weights(t)`, `cr_axis(c, c0, dc, n)` | `numerical_spacetime.jl` |
| Log-linear table | `loglinear_at(x, table, x0, dx, n)` | `_mc_weight_at` |
| Exact sum | `xsum_parts`, `xsum_add!(acc, x, I...)`, `xsum_atomic_add!`, `xsum_round` | `exact_sum.jl` + `_xsum_atomic_add!` |
| RNG | `MCRng`, `mc_uniform`, `mc_gauss`, `sample_direction`, `standard_gamma`, `chi_square`, `MCStreamLayout`, `primary_rng`, `child_rng` | `rng.jl` |
| Units | `struct Units{T}; L; T; dens; Ne; B; e; end`; `Units(m_bh, mass_scale)` | `units()` NamedTuple, `_unit_fields` |

### 3.2 Spacetime

Protocol (the method table replaces the implicit one in `metrics.jl`,
`spacetimes.jl`, `numerical_spacetime.jl`, `reverse_march.jl` and
`gpu_spacetime.jl`):

```julia
abstract type Spacetime{T} end
abstract type StationarySpacetime{T} <: Spacetime{T} end

# required
metric(st, x)::SMatrix{4,4}

# defaults, overridden by closed forms
inverse_metric(st, x) = inv(metric(st, x))
metric_and_inverse(st, x)
lower_index(st, x, v); raise_index(st, x, p); raise_and_norm(st, x, p)
inverse_metric_gradient(st, x)            # ForwardDiff; spatial seeding when stationary
geodesic_rhs(st, y); geodesic_transport_rhs(st, y, vs); rk4_step(st, y, h)

# termination (one rule; per-type horizon proxy)
struct MarchBudget{T}; C::T; escape::T; R_em::T; nmax::Int; end
march_control(st, y, r_prev, b::MarchBudget) -> MarchStep{T}   # (stop, h, R)
horizon_radius(st)                         # Kerr: closed form; binary: per hole; ADM: lapse floor

# lanes
lane(st, ::Type{T})                        # KerrSchildCartesian{Float32}(a), ADMGrid data untouched

# host only
trace_geodesic(st, y0; …)                  # adaptive solve with per-type callbacks (unchanged)
```

Concrete types: `KerrSchildCartesian{T}` (fields `a`, `r_h`; the horizon
radius is computed once in `Float64` and narrowed by `lane`, as the drivers
narrow `r_h` today, so the fp32 lane stays bitwise), `Minkowski{T}`,
`UserMetric{F,T}`, `SuperposedKerrSchild{N,T}`, `ADMGrid{T,A<:AbstractArray{T,4}}`.
`ADMGrid`
is one struct: with `A = Array` it is today's `ADMGridSpacetime`, after
`Adapt.adapt(backend, st)` it is today's `DevADMParams`; the sampling body is
written once. `KerrMarch` disappears: a Kerr march is
`SpacetimeMarch{KerrSchildCartesian{Tg},Tg}` whose `march_control` method is
the closed-form `ks_radius_energy` rule, bitwise what `KerrMarch` does today.

`CameraFrame{T}` gains the vertical: `sin_i, cos_i, d, up::SVector{4,T}`. The
polarized march reads `frame.up`; `cu1..cu4` and `_camera_up` disappear.

`analytic_kerr.jl` exports `pixel_prep(α, β, θo, a; ro)` and
`band_landing(prep, n) -> Landing` with named fields `(r, φ, t, νr, valid, λ, η)`
(today's seven-tuple), plus `critical_curve`. The intensity seam leaves this
module: painting is `intensity(source, landing)` in `Matter.sources/gdisk.jl`,
composed in `Products/analytic.jl`. `Geodesics` no longer takes a source
callback.

### 3.3 Matter

```julia
struct PlasmaPrimitives{T}    # code units, AthenaK layout
    rho::T; uu::SVector{3,T}; eint::T; bb::SVector{3,T}; inside::Bool
end
struct PlasmaState{T}         # cgs, with the fluid kinematics at the sample
    n_e::T; theta_e::T; B::T
    ucon::SVector{4,T}; ucov::SVector{4,T}; bcon::SVector{4,T}; bcov::SVector{4,T}
    b_sq::T; sigma::T; beta_inv::T
    valid::Bool
end
vacuum(::Type{T}) -> PlasmaState{T}       # the single sentinel

struct ElectronModel{T}; mu; ne_ni; use_p::Bool; rat_low; rat_high; gamma; gamma_i; gamma_e; end
struct PlasmaCuts{T}; rho; n_e; pgas; theta_e; b; sigma; beta_inv; end   # each a Bound{T}(lo, hi), ±Inf = off

abstract type PlasmaProvider end
plasma_state(p::PlasmaProvider, st, x, g::MetricPair, params) -> PlasmaState
    # default: decode(primitives(p, x), g, params.units, params.electrons, params.cuts)
primitives(p::PlasmaProvider, x) -> PlasmaPrimitives      # grid-backed providers implement this
```

Providers in this tier: `PolishDoughnut`, `PageThorneDisk`,
`EquatorialHotspot` (implement `plasma_state` directly in cgs, taking the
metric pair instead of recomputing it) and `UniformSphere` (the Monte Carlo
sphere). Grid-backed providers live in `Data` (§4.1). The Monte Carlo
`plasma_at` and the imaging `eight_coefficients` decode become one call to
`decode`; `fig_examples.jl`'s inverse-decode adapter is deleted because a
torus is a provider the public `render` accepts.

The cut convention changes from `-1` disables to `±Inf` disables; every
comparison is unchanged in effect, so the images are bitwise identical.

### 3.4 Transfer

```julia
abstract type TransferMode end
struct StokesI <: TransferMode end
struct IQUV    <: TransferMode end

struct Thermal{T};  frac::T; end
struct PowerLaw{T}; frac; p; gamma_min; gamma_max; jj; aa; jj_q; jj_v; aa_q; aa_v; rho; rho_q; rho_v; end
struct Kappa{T};    frac; kappa; w; …30 fitted constants…; end          # constructors do the precompute
struct TransferParams{T, P<:Union{Nothing,PowerLaw{T}}, K<:Union{Nothing,Kappa{T}}}
    units::Units{T}; electrons::ElectronModel{T}; cuts::PlasmaCuts{T}
    thermal::Thermal{T}; powerlaw::P; kappa::K
    nu_obs::T; L_unit::T
end
has_powerlaw(::TransferParams{T,Nothing}) = false   # compile-time, as `hasfield` is today

coefficients(::StokesI, p, state, kcov, ν)                 -> (j, α)
coefficients(::IQUV,    p, state, kcov, kcon, g, ν, renorm) -> (j4, α4, ρ4, tetrad, valid)
segment(::StokesI, j, α, ds)    -> (A, B)
segment(::IQUV,    j, α, ρ, ds) -> (M, b)

struct SpacetimeMarch{S,T}; st::S; budget::MarchBudget{T}; end      # geodesic lane
struct StoredPath{S,V,D};  st::S; ys::V; dls::D; end
march(mode, y0, prog) -> Float64 | Dual | SVector{K} | SVector{4}
```

`SimCoeffConfig` (mutable, 90 flat fields), `coefficient_params`,
`_march_params`, `_gimg_params`, `_gpol_params` and the `gp` NamedTuple are
replaced by `TransferParams`. The AD lane is `TransferParams{Dual}`; the
mass-scale ladder is an `NTuple{K,TransferParams}` with an `SVector{K}`
accumulator, as today. `use_p` is a `Bool`. The march bodies of
`reverse_march.jl` move unchanged except for field access through the
sub-records, so the arithmetic order is unchanged.

### 3.5 MonteCarlo

```julia
generate(zone::EmissionZone, tables, rng) -> (Superphoton, rng)     # one generator; the sphere draws its
                                                                   # emission point through UniformSphere
absorption_rate(state, p)              -> ν α_ν                     # replaces 3 copies
scattering_rate(table, state, p)       -> ν n_e σ_h                 # replaces 3 copies
struct Bias{T}; b0::T; end; bias(b::Bias, nscatt)

abstract type Sink end                  # record!(sink, sph); spawn!(sink, child, rng, event)
struct FlatMarch{T}; R::T; step_frac::T; nmax::Int; end             # the sphere's straight-line control
track!(ctl, st, provider, opacities, sph, rng, sink; absorb, scatter) -> (sph, recorded, rng)
```

One tracker replaces `track_superphoton`, `track_superphoton_scatter!`,
`_mc_march_scatter_core`, `_mc_track_thin_w`, `_mc_march_scatter!`,
`track_superphoton_grmhd` and `track_superphoton_grmhd_scatter!`. The control
decides the step (`FlatMarch`: `step_frac R / |k|`; `SpacetimeMarch`: the
`_grmhd_dlam` rule), the provider supplies the plasma, `absorb`/`scatter`
are compile-time switches (`Val`), and the sink receives escapes and
children. The exponential-transform scatter decision stays one function.

## 4. Infrastructure tier

No physics is written here. This tier owns arrays, files, threads, devices,
queues and caches, and implements the physics protocols over its containers.

### 4.1 Data

```julia
struct FluidBlocks{A5,A2,L} <: PlasmaProvider     # the sampling struct; isbits after Adapt
    data::A5          # (8, ni, nj, nk, nmb) Float32, feature-fastest on host AND device
    x_left::A2; dx::A2; lo::A2; hi::A2            # (3, nmb)
    ni::Int; nj::Int; nk::Int; nmb::Int
    locate::L         # LeafTable | LevelHash | BlockScan
end
primitives(b::FluidBlocks, x) -> PlasmaPrimitives      # the one trilinear body (clamped, block-interior)
Adapt.@adapt_structure FluidBlocks                      # LevelHash adapts to BlockScan (device has no Dict)

struct AthenaKSnapshot             # host metadata + blocks
    blocks::FluidBlocks; levels; logical_locations; names; time
end
struct UniformGrid{A4} <: PlasmaProvider;  data::A4; n; xo; inv_dx; end
struct FrameWindow{P} <: PlasmaProvider;   a::P; b::P; t_a; inv_dt; t_obs; end    # device-able two-frame blend
struct SeriesProvider <: PlasmaProvider;   series::SnapshotSeries; end             # host, any bracket
blend(sa, sb, w)                                        # the one time blend
```

- The host array layout becomes feature-fastest `(8, ni, nj, nk, nmb)`, the
  device layout of today. The CPU sampler reads the same eight floats per
  corner, so its results are bitwise unchanged, and `permutedims` at upload
  disappears.
- `LeafTable`/`LevelHash`/`BlockScan` are built eagerly in the snapshot
  constructor (P17); `LeafCache` and its lock disappear.
- `FluidBlocks` and `UniformGrid` share `lerp8`; each computes its own
  fractional index (`(x - x_left)/dx` versus `(x - xo) * inv_dx`), so both
  stay bitwise what they are today.
- `SnapshotSeries` keeps the LRU loader; `window(series, ia)` returns a
  `FrameWindow`; `advance(window, series)` promotes the upper frame.
- `read_athenak_bin`, `athenak_bin_snapshot`, the `press` layout, and the Z4c
  resample into `ADMGrid` move here unchanged.
- `tables.jl` holds the threaded builders (`build_weight_table`,
  `build_zone_emission_table`, `zone_counts`, `build_int_jnu_table`,
  `build_hotcross_table`, `load_hotcross`); the lookups they feed are physics.

### 4.2 Execution

```julia
abstract type Executor end
struct ThreadedExecutor <: Executor; chunk::Int; end                  # native threads, host arrays
struct PoolExecutor     <: Executor; pool::DevicePool; end            # KA backends: HostDevice or CUDADevice
ThreadedExecutor(; chunk = 4096)
PoolExecutor(devices; lanes = 2, chunk = 65536)                       # ordinals, CUDADevice/HostDevice handles, or a DevicePool

# the CPU/GPU switch (backend.jl, policy in §5.4)
resolve(backend) -> Executor          # :auto | :cpu | :gpu | :host_pool | Executor | device list | DevicePool
backend!(backend); backend() -> Executor                              # session default, printable
describe(exec) -> NamedTuple          # kind, devices, lanes, chunk, threads; stored in every result

# the one per-item contract
foreach_item!(exec, prog, items; outs::Tuple = (), order = nothing)
    # prog(item::Int, outs_view...) -> nothing  or  prog(item) -> value stored at outs[k][item]

abstract type RaySource end
ray(src, item) -> (RayState, ok::Bool)
struct HostRays{V}; rays::V; end
struct CameraRays{S,F,G}; st::S; frame::F; gen::G; end        # G ∈ PixelGrid | BlockGrid | CoordList
```

- `kernel.jl` holds the only `@kernel`:
  `@kernel function _foreach!(prog, @Const(items), n, outs...)`. A program is
  an `isbits`-after-`Adapt` functor; the kernel specialises on its type, so
  the generated code is that of today's hand-written kernels. Workgroup size
  is a program property (`workgroupsize(::ImagingProgram{IQUV}) = 64`).
- `ThreadedExecutor` runs the same functor in a chunked `Threads.@threads`
  loop. This is the only `Threads.@threads` over items in the package.
- `pool.jl` is today's `DevicePool`, lanes, jobs, `enqueue!`, pinned staging,
  with `march!`/`MarchPart` folded into `foreach_item!`. `rays_on_device!`
  disappears: the ray source is part of the program and is adapted once per
  device by `residency.jl`.
- `residency.jl`: `resident(exec, obj)` returns the per-device adapted twin
  of any host object (blocks, grids, ADM data, tables), cached per pool by
  identity with a weak reference; `device_program(lane, prog)` adapts the
  program's fields. `upload_amr`, `upload_grid`, `upload_adm`,
  `upload_amr_scan`, `upload_window` and their `resident_*` wrappers collapse
  into this one function.
- `accumulators.jl`: `ExactBins` (count, Σw limbs, ΣwE limbs) with atomic
  `record!`, used by both executors (Atomix on host arrays, KA atomics in the
  kernel), so a Monte Carlo spectrum is bitwise independent of executor,
  thread count, device count and chunking by construction. `ChildBuffer`
  (SoA with an atomic slot counter, overflow → rerun at exact size) serves
  both executors; `ScatterWorkspace`'s depth-first stack is retired in favour
  of the layered loop everywhere.
- `schedule.jl`: `cost_order`, `rescue_pixels`, `mixed_partition`,
  `march_sort_order`, `queue_items`, computed in `prepare` and stored in the
  plan (no module-level `Ref` caches).
- `worklist.jl`: `stream_blocks!(exec, prog, work, block, visit)` over a
  channel; the pool executor streams children without level barriers as
  `AMRStokes` does today, the threaded executor runs the same loop with
  thread workers. One implementation.
- `devices.jl`: `HostDevice(id)`, `CUDADevice(ordinal)`; `backend(dev)`,
  `activate!(dev)`, `stage_alloc`, `stage_free`, `free!`, `select_devices`.
  The CUDA methods live in the extension; integer ordinals remain accepted
  as `CUDADevice`.
- `backend.jl`: `resolve` turns a symbol, a device list, a pool or an
  executor into an `Executor`; `backend!` and `backend` hold the session
  default; `BLP_BACKEND` seeds it at first use. `:auto` chooses the pool when
  the CUDA extension is loaded, `CUDA.functional()` holds and
  `select_devices()` is non-empty, else threads, and logs the choice and its
  reason once per session. An explicit `:gpu` that cannot be honoured throws.

## 5. Pipeline tier

### 5.1 Plans

```julia
Base.@kwdef struct CameraConfig; inclination_deg = 60.0; distance = 1000.0; fov = 40.0; resolution = 64; end
Base.@kwdef struct PlasmaConfig; …as today…; end            # electron model, populations, cuts
Base.@kwdef struct RunConfig
    m_bh_cgs; mass_scale_cgs; frequency_hz
    camera::CameraConfig; plasma::PlasmaConfig
    max_steps = 2500; C = 80.0; escape_factor = 1.1
end
# removed: bh_a (the spacetime is a value), tracer (a product), polarized (a mode)

struct Lanes{Tg,Tt,A}; end                        # geodesic lane, transfer lane, accumulator
Lanes() = Lanes{Float64,Float64,Float64}()
struct MixedLanes; rescue_px::Int; end            # fp32 bulk, fp64 within rescue_px of the critical curve

prepare(cfg::RunConfig; spacetime, matter, mode = IQUV(), lanes = Lanes(), tile = 16) -> Plan
prepare(model::UniformSphere; …) -> Plan          # Monte Carlo sphere needs no RunConfig

struct Plan{S,Sg,P,M,T,L}
    spacetime::S; geodesic_spacetime::Sg          # transfer lane; geodesic lane
    provider::P; mode::M
    params::T                                     # TransferParams or NTuple of them
    camera::PlaneParallelCamera; frame::CameraFrame
    budget::MarchBudget; horizon::Float64; escape::Float64; R_em::Float64
    lanes::L; queue::Vector{Int}; rescue::Vector{Int}
end
```

Everything a driver derives today (camera, frame, horizon, escape,
emission-confinement radius, units, prepared nonthermal constants, cost order,
rescue set, lane conversions) is computed here once. `R_em` comes from the
provider (`domain_radius(provider)`), replacing the three ad-hoc formulas.

Compatibility: for one minor release `RunConfig(bh_a = …, tracer = …,
polarized = …)` is accepted with a deprecation warning and mapped to
`spacetime = KerrSchildCartesian(bh_a)` and a mode, and the old entry points
(§7) exist as deprecated one-line wrappers so the paper's figure scripts keep
running during the migration.

### 5.2 Programs

```julia
struct ImagingProgram{M,S,C,P,T,F,A,R}  <: Function
    mode::M; st::S; control::C; provider::P; params::T; frame::F; rays::R
end
(p::ImagingProgram)(item) = (y0, ok) = ray(p.rays, item); ok ? march(p.mode, y0, p) : nan_output(p)
Adapt.@adapt_structure ImagingProgram
```

`TraceProgram` (stores trajectories and steps into row views),
`ShadowProgram`, `AnalyticProgram` (per pixel: `pixel_prep`, then
`band_landing` and `intensity` per band), `MCPrimary` and `MCLayer` (generate
or read a child, `track!`, sinks) are built the same way. A program is the
unit the executors run; it is the only place a mode, a spacetime, a provider,
a parameter set and a ray source meet.

### 5.3 Products

| Entry point | Returns | Determinism |
|---|---|---|
| `render(plan; backend)` | `Image` (I, Q, U, V planes, metadata) | bitwise per item |
| `render(plan, coords; backend)` | `4 × n` Stokes matrix for image-plane coordinates | bitwise per item |
| `render_amr(plan; block, max_level, refine, executor)` | `Image` mosaic, level map, ray count | bitwise per item; mosaic independent of completion order |
| `render_analytic(plan; bands, source, backend)` and `photon_ring_layers(plan, disk; …)` | per-band images and landing maps | bitwise per item |
| `trace(plan; backend, store = true)` | `Trajectories` (source→camera states and steps) | bitwise per item |
| `shadow(plan; backend)` | ray fates, step counts, null error | bitwise per item |
| `sensitivity(plan, param; backend)` and `variants(plan, mass_scales; backend)` | image and derivative; K images | bitwise per item |
| `mc_spectrum(plan, Ns; scattering, backend, bands, …)` | `Spectrum`, `ScatterResult` or `GRMHDSpectrumResult` | bitwise reduction (exact bins, genealogy streams) |

`render` for the CPU is the fused march under `ThreadedExecutor`, so the
CPU and device products are the same program; the two-phase path is
`render(plan; path = trace(plan))`, which marches a `StoredPath`. Stokes I is
bitwise identical between the two paths (same RK4 and RHS); IQUV agrees to the
round-off between the transport RHS and the tracer RHS, which is today's
CPU-versus-device agreement. Slow light is a `FrameWindow` or `SeriesProvider` as `matter`.
A binary or numerical spacetime is a `spacetime` argument. Mixed precision is
`lanes = MixedLanes()`. A GRMHD Monte Carlo spectrum on the device, which the
current code cannot do, follows from `backend = :gpu` plus the resident
`FluidBlocks`.

The body of `render` is the budget of P12:

```julia
function render(plan::Plan; backend = :auto)
    exec = resolve(backend)
    prog = ImagingProgram(plan.mode, plan.spacetime, SpacetimeMarch(plan.geodesic_spacetime, plan.budget),
                          plan.provider, plan.params, plan.frame, CameraRays(plan.geodesic_spacetime, plan.frame, PixelGrid(plan.camera)))
    out = Vector{output_type(plan)}(undef, npixels(plan.camera))
    foreach_item!(exec, prog, plan.queue; outs = (out,))
    return Image(plan, out, describe(exec))
end
```

with the mixed-lane variant adding a second program over `plan.rescue`
through `mixed_partition`.

### 5.4 Choosing CPU or GPU

Where a product runs is one keyword, `backend`, with a session default and
an environment override. A plan never records a backend: the same plan
renders on the CPU and on the GPUs, and device uploads are cached by the
pool, so switching back and forth costs one upload per device.

| Precedence | Form | Example |
|---|---|---|
| 1 | keyword on the product call | `render(plan; backend = :gpu)` |
| 2 | session default | `BlackLightPlus.backend!(:gpu)`; `backend!([0, 1])`; `backend!(PoolExecutor([0, 1]; chunk = 2^15))` |
| 3 | environment, read once at first use | `BLP_BACKEND=gpu`, `BLP_BACKEND=gpu:0,1`, `BLP_BACKEND=cpu`, `BLP_BACKEND=host_pool:2` |
| 4 | `:auto` | the pool over `select_devices()` when CUDA is loaded and functional, else threads |

Accepted values everywhere: `:auto`, `:cpu`, `:gpu`, `:host_pool` (the
KernelAbstractions CPU backend through the pool, for tests), an `Executor`,
a list of device ordinals or handles, or a `DevicePool`.

Rules.

- `:auto` never downgrades silently. The first resolution in a session logs
  the choice and the reason: `CUDA not loaded`, `no functional device`,
  `no idle device`, or the device list used.
- An explicit `:cpu` or `:gpu` is strict. `:gpu` without the CUDA extension
  loaded, or without a functional device, throws `BackendUnavailable` naming
  the remedy (`using CUDA`, `CUDA_VISIBLE_DEVICES`, or `backend!(:cpu)`).
- `BlackLightPlus.backend()` returns the resolved session default and prints
  it: kind, devices, lanes per device, chunk, Julia threads.
- Every result carries `result.run`: the executor description, the lanes, the
  package version and the wall time, so a saved image states where and how it
  was made.
- Device knobs (`lanes`, `chunk`, workgroup overrides) exist only on
  `PoolExecutor`; `ThreadedExecutor` has `chunk`. No other function in the
  public API names a device.
- Precision lanes (`Lanes`, `MixedLanes`) are a plan property, independent of
  the backend; mixed lanes run on either.

The switch is `Execution/backend.jl` (about 80 lines); the extension
supplies the `CUDA.functional()` probe and the NVML side of `select_devices`.

### 5.5 Analysis

`refinement.jl` (the relative-gradient, Laplacian and absolute-gradient block
indicators as pure functions of a block array), `lensing_bands.jl` (apparent
horizon, band edges, band grids; host bisection and polygon tests) and
`interferometry.jl` (unchanged). `write_npy` is `Data`.

## 6. Repetition removed

| Today | Sites | Tomorrow |
|---|---|---|
| Driver prologue: camera from cfg, `1 + sqrt(1 - a*a)`, `1.1 * distance`, `R_em`, `gp` | 13 / 18 / 14 / 4 formulas | `prepare` |
| `_scc_config`, `_prepared_scc`, `_gimg_params`, `_gpol_params`, `_march_params`, `GRMHDModel` constructor | 6 | `TransferParams` constructors in `prepare` |
| `SimCoeffConfig` → `coefficient_params` → `gp` | 3 representations | `TransferParams` |
| `KerrMarch`, `SpacetimeMarch` | 2 | `SpacetimeMarch{S,T}` |
| `ADMGridSpacetime`, `DevADMParams` | 2 | `ADMGrid{T,A}` |
| Trilinear: `sample_primitives`, `_sample_stencil`, `_gimg_sample`, `_gamr_lerp` | 4 | `lerp8` + two index maps |
| Corner reads `_corner`, `_gimg_corner`, `_gamr_corner` | 3 | `corner(blocks, i, j, k, mb)` |
| Locate: `_find_meshblock_scan`/`_gamr_locate_scan`, `_leaf_locate`/`_gamr_locate_fast`, `_hash_locate` | 5 | `locate(::BlockScan)`, `locate(::LeafTable)`, `locate(::LevelHash)` |
| Time blend: `sample_primitives_t`, `_gimg_sample_t`, `_gamr_sample_t` | 3 | `blend` in `FrameWindow` |
| Vacuum: `Sources._vacuum`, `MonteCarlo._vacuum`, zero `SVector{8}` | 3 | `vacuum(T)` |
| Plasma decode sequence in `eight_coefficients`, `unpolarized_coefficients`, `plasma_at` | 3 | `decode` |
| Imaging kernels {I, IQUV} × {uniform, AMR leaf, AMR scan} × {fast, slow} + kvariant + stiffness + shadow + tracer + analytic + camray + MC primary/layer + probes | 21 | 1 generic kernel; the axes are program fields |
| `render*`, `march_rays*`, `trace*`, `mc_spectrum*` entry points | 29 | 9 (§7) |
| Uploads: `upload_grid`, `upload_amr`, `upload_amr_scan`, `upload_adm`, `upload_window`, `upload_amr_window`, `resident_grid`, `resident_amr`, `_resident_scan`, `_gsp_per_device` | 10 | `resident` |
| Opacity rates `_nu_alpha`, `_nu_alpha_scatt`, `_grmhd_nu_alpha`, `_grmhd_nu_alpha_scatt`, `_mc_nu_alpha_dev`, `_mc_nu_alpha_scatt_dev` | 6 | `absorption_rate`, `scattering_rate` |
| Trackers (sphere, sphere-scatter, GRMHD, GRMHD-scatter, device thin, device scatter) | 6 | `track!` with a control and a sink |
| Generators `sample_zone_photon`, `sample_sphere_photon`, `_mc_gen_sphere_photon` | 3 | `generate` + `UniformSphere` |
| Spectra: `Spectrum`, `ExactSpectrum`, device bins, per-thread merge | 4 | `ExactBins` (accumulator) + `Spectrum` (product) |
| Child queues: `ScatterWorkspace` stack, device SoA buffer + rerun | 2 | `ChildBuffer`, layered loop on both executors |
| MC drivers `mc_spectrum`, `mc_spectrum_scatter`, `mc_spectrum_grmhd`, `mc_spectrum_gpu` ×2 | 5 | `mc_spectrum(plan, Ns; …)` |
| Stokes planes reshape `_stokes_planes` + 4 inline copies | 5 | `Image(plan, out)` |
| `Threads.@threads` item loops | 23 | `ThreadedExecutor` + 4 table builders |
| AMR streaming: `render_amr` level loop, `AMRStokes`, `stream_blocks!` ×2 | 3 | `worklist.jl` |
| Camera vertical: `_camera_up`, `cu1..cu4`, `_gpol_params` | 3 | `CameraFrame.up` |
| `isdefined … include` guards, include-order notes | 12 | 0 |

Not merged, because they are different physics: `fluid_tetrad` (imaging,
e₍₃₎ along k) and `emission_tetrad` (Monte Carlo, e₍₁₎ along b); the
unpolarized and polarized marches; the adaptive `trace_geodesic` and the
fixed-control march.

## 7. Entry-point map

| Today | Tomorrow |
|---|---|
| `render(cfg, snap; polarized, spacetime)` | `render(prepare(cfg; spacetime, matter = snap, mode))` |
| `render_with_powerlaw` | `prepare(cfg)` with `PlasmaConfig(power_frac = …)`; deleted |
| `render_gpu_fused`, `render_gpu_polarized` | `matter = UniformGrid(snap, box_half, n)`, `backend = :gpu` |
| `render_gpu_amr`, `render_gpu_amr_polarized` | `matter = snap` (native blocks), `backend = :gpu` |
| `render_gpu_spacetime` | `spacetime = SuperposedKerrSchild(…)` or `ADMGrid`; matter as above |
| `*_slow` (4) | `matter = FrameWindow(series, ia)` or `SeriesProvider(series)` |
| `render_gpu_fused_grad` | `sensitivity(plan, :mass_scale)` |
| `render_gpu_fused_kvariant` | `variants(plan, mass_scales)` |
| `march_rays_gpu_amr`, `march_rays_gpu_spacetime` | `render(plan, coords)` |
| `render_amr(cfg, path; stokes = AMRStokes(…))` | `render_amr(plan; backend)` |
| `render_analytic`, `render_analytic_gpu` | `render_analytic(plan; backend)` |
| `photon_ring_layers` | unchanged name, takes a plan |
| `trace_rays`, `trace_rays_sampled`, `trace_rays_gpu` | `trace(plan; backend, adaptive = false)` |
| `trace_shadow_gpu` | `shadow(plan; backend)` |
| `mc_spectrum`, `mc_spectrum_scatter`, `mc_spectrum_gpu` | `mc_spectrum(prepare(UniformSphere(…)), Ns; scattering, executor)` |
| `mc_spectrum_grmhd` | `mc_spectrum(prepare(cfg; spacetime, matter = snap), Ns; scattering, bands, …)` |
| `GRMHDModel`, `SphereModel`, `ScatteringSphere` | a `Plan`; `UniformSphere` provider; a constructor preset |
| `polunpol_consistency`, `stiffness_telemetry`, `amr_sample_probe`, `amr_locate_probe` | scripts in `bench/` and `test/` over the public API |
| `devices = …` keyword, `select_devices`, `DevicePool`, `HostDevice`, `default_pool` | `backend = …` on every product (§5.4); `select_devices`, `DevicePool`, `HostDevice` unchanged in `Execution`; `default_pool()` becomes `backend()` |

## 8. File map

Every current file and where its contents go. "split" means the file's
physics goes to the first destination and its machinery to the second.

| Current | Destination |
|---|---|
| `constants.jl` | `Numerics/constants.jl` |
| `inputs/parameters.jl` | `Plans/config.jl` |
| `geodesics/metrics.jl` | `Spacetime/kerr_schild.jl`, `user_metric.jl`, `metric_ops.jl` |
| `geodesics/camera.jl` | `Spacetime/camera.jl` |
| `geodesics/integrator.jl` | `Spacetime/geodesic.jl`, `termination.jl`, `adaptive.jl`; `trace_rays*` → `Products/trace.jl` |
| `geodesics/spacetimes.jl` | `Spacetime/superposed.jl`, `termination.jl`, `adaptive.jl` |
| `geodesics/numerical_spacetime.jl` | `Spacetime/adm_grid.jl`; stencil → `Numerics/stencils.jl` |
| `geodesics/kerr_orbits.jl` | `Spacetime/kerr_orbits.jl` |
| `geodesics/analytic_kerr.jl` | `Spacetime/analytic_kerr.jl` (seam removed) |
| `geodesics/lensing_bands.jl` | split: `critical_curve` → `Spacetime/analytic_kerr.jl`; bands, grids, polygons → `Analysis/lensing_bands.jl` |
| `geodesics/gpu_tracer.jl` | `Products/trace.jl` (`TraceProgram`) |
| `simulation/primitives.jl` | `Matter/decode.jl` |
| `simulation/athenak_reader.jl` | `Data/athenak_hdf5.jl`, `Data/fluid_blocks.jl`; `units` → `Numerics/units.jl` |
| `simulation/athenak_z4c.jl` | `Data/athenak_bin.jl` |
| `simulation/time_series.jl` | `Data/series.jl` |
| `radiation/tetrads.jl` | `Transfer/tetrads.jl` |
| `radiation/coefficients.jl` | `Transfer/params.jl`, `synchrotron.jl`; Bessel and 2F1 → `Numerics/special.jl`; decode and cuts → `Matter/` |
| `radiation/transfer.jl` | `Transfer/segments.jl` |
| `radiation/integrator.jl` | `Transfer/coefficients.jl`; `polarized_march`/`unpolarized_march` → `StoredPath` use in `Products/image.jl` |
| `radiation/reverse_march.jl` | `Transfer/controls.jl`, `march.jl` |
| `radiation/equatorial_source.jl` | `Matter/sources/gdisk.jl` |
| `montecarlo/rng.jl` | `Numerics/philox.jl` |
| `montecarlo/exact_sum.jl` | `Numerics/exactsum.jl`; `ExactSpectrum` → `Execution/accumulators.jl` |
| `montecarlo/mc_common.jl` | `Numerics/stencils.jl` (lookup), `MonteCarlo/track.jl` (scatter event) |
| `montecarlo/emissivity.jl` | `MonteCarlo/emissivity.jl` |
| `montecarlo/superphoton.jl` | `MonteCarlo/superphoton.jl` |
| `montecarlo/hotcross.jl` | `MonteCarlo/hotcross.jl`; file load and table build → `Data/tables.jl` |
| `montecarlo/scatter.jl` | `MonteCarlo/scatter.jl` |
| `montecarlo/emission_tables.jl` | `Matter/zones.jl` (`EmissionZone`), `MonteCarlo/tables.jl` (records, lookups), `Data/tables.jl` (builders) |
| `montecarlo/generate.jl` | `MonteCarlo/generate.jl` |
| `montecarlo/spectrum.jl` | `Execution/accumulators.jl` (binning into `ExactBins`), `Products/montecarlo.jl` (`Spectrum` product and `nuLnu*`) |
| `montecarlo/track.jl` | `MonteCarlo/opacity.jl`, `track.jl`; workspace → `Execution/accumulators.jl` |
| `montecarlo/sphere_model.jl` | `Matter/zones.jl` (`UniformSphere`), `Products/montecarlo.jl` (`analytic_nuLnu`) |
| `montecarlo/grmhd_model.jl` | `Products/montecarlo.jl` (`emission_zones`); `plasma_at` → `Matter/decode.jl` |
| `montecarlo/mc_cpu.jl` | `Products/montecarlo.jl` (drivers), `MonteCarlo/track.jl` (GRMHD step rule) |
| `montecarlo/mc_gpu.jl` | `Products/montecarlo.jl` (`MCPrimary`, `MCLayer`), `Execution/accumulators.jl` |
| `sources/sources.jl` | `Matter/protocol.jl`, `sources/shared.jl` |
| `sources/polish_doughnut.jl`, `page_thorne.jl`, `hotspot.jl` | `Matter/sources/` |
| `outputs/outputs.jl` | `Data/npy.jl` |
| `outputs/interferometry.jl` | `Analysis/interferometry.jl` |
| `render.jl` | `Plans/prepare.jl`, `Products/image.jl`, `Products/analytic.jl`; `polunpol_consistency` → `bench/` |
| `amr.jl` | `Analysis/refinement.jl`, `Execution/worklist.jl`, `Products/amr.jl` |
| `gpu_metric.jl` | `Spacetime/termination.jl` (already the same rule) |
| `gpu_camera.jl` | `Execution/ray_sources.jl` |
| `gpu_pool.jl` | `Execution/devices.jl`, `pool.jl`, `residency.jl`, `executor.jl`; CUDA bits → `ext/` |
| `gpu_image.jl` | `Data/uniform_grid.jl`, `Data/series.jl` (window), `Execution/schedule.jl`, `Products/image.jl`, `sensitivity.jl` |
| `gpu_pol.jl` | `Products/image.jl`; stiffness telemetry → `bench/` |
| `gpu_amr.jl` | `Data/fluid_blocks.jl`, `Data/series.jl`, `Execution/worklist.jl`, `Products/image.jl`, `amr.jl`; probes → `test/` |
| `gpu_spacetime.jl` | `Spacetime/adm_grid.jl`, `Data/fluid_blocks.jl` (`BlockScan`), `Products/image.jl`, `trace.jl` |
| `gpu_analytic.jl` | `Products/analytic.jl` (`AnalyticProgram`); stack-limit hook → `ext/` |
| `gpu_image_ad.jl` | `Products/sensitivity.jl` |

## 9. Public API and internal interfaces

Top-level exports (54 names):

```
Configuration   CameraConfig PlasmaConfig RunConfig Lanes MixedLanes prepare
Spacetimes      KerrSchildCartesian Minkowski UserMetric SuperposedKerrSchild ADMGrid
                adm_grid_from_metric z4c_grid_spacetime
Matter          AthenaKSnapshot UniformGrid SnapshotSeries FrameWindow SeriesProvider
                PolishDoughnut PageThorneDisk EquatorialHotspot EquatorialDisk UniformSphere
Modes           StokesI IQUV
Execution       ThreadedExecutor PoolExecutor DevicePool HostDevice CUDADevice select_devices backend backend!
Products        render render_amr render_analytic photon_ring_layers trace shadow sensitivity variants mc_spectrum
Results         Image Trajectories Spectrum ScatterResult GRMHDSpectrumResult
Analysis / I-O  write_npy visamp visamp_minima radon_cut tukey_window lensing_band critical_curve
```

Each submodule exports its interface (the method tables of §3 and §4). Tests
import those explicitly (`using BlackLightPlus.Transfer: stokes_segment`).
`Constants` becomes `Numerics` and stays unexported. `STANDARDS.md` §5
"the public API is exactly what `BlackLightPlus.jl` exports" stays true and
now means 54 names. The `DESIGN_PRINCIPLES.md` §5 gate of 50 is met if
`radon_cut`, `tukey_window`, `SeriesProvider` and `CUDADevice` (ordinals
suffice) are judged internal; they are listed here for review.

## 10. Tests

```
test/
├── runtests.jl              includes every file below unconditionally
├── decks.jl                 shared synthetic decks (today's device_decks.jl + tiny_snap)
├── structure.jl             the mechanical gates of DESIGN_PRINCIPLES §5: layering, private reach,
│                            kernel count, arity and file budgets, export count, duplicate greps
├── physics/                 spacetime, geodesics, analytic Kerr, coefficients, segments, marches,
│                            sources, MC generation/scatter/hotcross/rng/exactsum; AD through each
├── infra/                   readers and layouts, locate strategies, series/window, executors
│                            (threaded vs HostDevice pool: bitwise per item), pool scheduling, accumulators,
│                            backend switch (precedence, `BLP_BACKEND` grammar, strict errors, the one-time log)
└── pipeline/                products on the decks: one plan on `:cpu` and `:host_pool` bitwise, AMR mosaic independence,
                             slow light fast-light limit, mixed lanes, MC CPU-vs-device equality,
                             RAPTOR/AART/gpumonty cross-code references
```

Resource skips (device, data file, reference table) are inside the test
files and are reported. The device tests run on `HostDevice`s in CI; CUDA
runs are the documented pre-release procedure, unchanged.

## 11. Migration

Incremental, module by module, suite green at every commit. Each phase ends
with its gate from `DESIGN_PRINCIPLES.md` §5 met and the regression decks
checked. "Bitwise" below means identical bits on the decks against the tree
before the phase; "tolerance" means the documented physics tolerance against
the CPU reference with a one-time re-baseline of pinned outputs.

| Phase | Work | Acceptance |
|---|---|---|
| 0 | Freeze decks and hash their outputs: Kerr native-AMR I and IQUV at 64², binary IQUV, ADM shadow, analytic disk, AMR mosaic, slow-light window, MC sphere / scatter / GRMHD stride, trajectories. | hashes recorded in `test/decks.jl` |
| 1 | `Numerics`: move constants, units, exact sum (+ atomic add), RNG, stencils, Bessel/2F1; add ray-state accessors and replace the 17 sites. | bitwise |
| 2 | `Spacetime`: rename, `ADMGrid{T,A}` absorbs `DevADMParams`, `MarchBudget` + `march_control` absorbs `KerrMarch`, `CameraFrame.up`, `Landing` record, seam removed. | bitwise |
| 3 | `Matter`: records, `decode`, `PlasmaCuts` with `±Inf`, providers protocol, analytic sources on the metric pair, `UniformSphere`. | bitwise |
| 4 | `Transfer`: `TransferParams` replaces `SimCoeffConfig`/`coefficient_params`/`gp`; marches and coefficient functions read sub-records. | bitwise |
| 5 | `Data`: `FluidBlocks` feature-fastest layout, eager locate, `UniformGrid`, `FrameWindow`, `blend`; delete the device sampler copies. | bitwise (native blocks, uniform grid each keep their index map) |
| 6 | `Execution`: `Executor` protocol, generic kernel, ray sources in programs, `resident`, `ExactBins` atomic on both executors, `ChildBuffer`, `worklist`, schedule caches into the plan, `backend.jl` with `BLP_BACKEND`; CUDA glue to `ext/`, `CUDA` to `[weakdeps]`. | bitwise per item against phase-5 outputs on `HostDevice`; MC bitwise by exact sums |
| 7 | `Plans` + `Products`: `prepare`, `Plan`, programs, the nine entry points with the `backend` keyword, `result.run` metadata, deprecated wrappers for the 29 old names, figure scripts switched to the new API. | bitwise against the phase-6 `HostDevice` outputs; the fused CPU default agrees with today's stored-path CPU images bitwise for Stokes I and to the current CPU/device round-off for IQUV, both checked on the decks |
| 8 | `MonteCarlo` + `Products/montecarlo.jl`: one tracker, one generator, layered scatter on both executors, GRMHD MC on the pool. | spectra bitwise (exact sums, genealogy streams); child sets identical |
| 9 | `Analysis`, docs: `method.md` absorbs provenance prose, `STANDARDS.md` §1/§5 amended, `gpu.md` rewritten, API pages regenerated, `CHANGELOG`; `structure.jl` switched on; export list cut to §9. | all gates of §5 met |
| 10 | Remove the deprecated wrappers and the `RunConfig` shim; tag `v0.2.0`. | — |

The kernel audit (`bench/kernel_audit.jl --check`) runs at every phase
boundary; registers, stack and spill must not exceed the baseline.

## 12. Risks and reversals

- **Program functors must be `isbits` after `Adapt`.** Providers that carry
  `Dict`s or `String`s are split into a sampling struct and host metadata
  (`FluidBlocks` / `AthenaKSnapshot`). `LevelHash` adapts to `BlockScan` on
  the device, which is today's scan path; documented.
- **Compile time.** One source kernel specialised per program type produces
  the same number of specialisations as today's hand-written kernels; no
  change expected. If the generic kernel costs registers (the audit will
  say), the fallback is one kernel per transfer mode, still three kernels.
- **Threaded executor and atomics.** Exact integer atomics on host arrays
  are cheap at the Monte Carlo record rate; if contention shows at high
  thread counts, per-thread `ExactBins` merged by integer addition is a
  drop-in with identical bits.
- **Deprecation shims.** They keep the paper's figure scripts running
  through phase 7 and are removed in phase 10. If decision §6.5 is reversed
  (keep `RunConfig` fields), `prepare` reads `bh_a`/`tracer`/`polarized` and
  the shim is permanent.
- **If decision §6.1 is reversed** (GPU code stays outside the package), the
  `Execution` module and the generic kernel still apply; only `pool.jl` and
  `devices.jl` would be an `include`d file again, and the gates on include
  guards and precompilation are dropped.
- **If decision §6.3 is reversed** (stored path stays the CPU default),
  `render` calls `trace` then marches a `StoredPath`; nothing else moves.
- **Silent CPU fallback.** `:auto` on a session without `using CUDA` runs on
  threads. The once-per-session log and `result.run` make it visible;
  `backend!(:gpu)` or `BLP_BACKEND=gpu` turns it into an error.
- **Extension loading.** Julia activates the CUDA extension only after the
  user's own `using CUDA`; the package cannot load it on their behalf, so
  `:gpu` before that throws with the remedy.
- **World age.** Pool workers call programs through `invokelatest` as today.
- **Naming.** `Numerics` rather than `Core` (clash with `Base.Core`); the
  other module names are open.

## 13. Dissolving the `gpu_*.jl` files

The eleven device files (`gpu_metric`, `gpu_camera`, `gpu_analytic`,
`gpu_image_ad`, `gpu_spacetime`, `gpu_pol`, `gpu_pool`, `gpu_image`,
`gpu_amr`, `geodesics/gpu_tracer`, `montecarlo/mc_gpu`; 3,694 lines) contain
almost no device-specific code. Three measurements fix what they are:

- **20 call sites** in all of them touch CUDA directly (`device!`, pinned
  `alloc`/`free`, `limit!`, `ndevices`, `memory_info`, NVML, `unsafe_free!`,
  `CuArray`): 10 in `gpu_pool.jl`, 8 in the two `gpu_amr.jl` validation
  probes, 2 in `gpu_analytic.jl` (stack limit). Everything else is
  KernelAbstractions, which already runs on the CPU backend.
- **The eleven march kernels are one kernel.** Each body is
  `out[i] = reverse_*_march(y0s[i], y -> <sampler>(…, y), …, <control>, nmax)`;
  they differ only in which sampler closure and which control they pass. The
  AD kernel is already an alias (`_fused_kernel_ad! = _fused_kernel!`).
- **The physics they call is CPU code.** `metric_and_inverse`, `rk4_step_fixed`,
  `geodesic_transport_rhs`, `plasma_decode`, the synchrotron fits, the tetrads,
  the segment operators and both marches are imported from the package; the
  files add copies only of the samplers and of the ADM parameter struct.

### 13.1 Line dissection

Approximate lines per category, read off the current files.

| File | CUDA-only | KA-generic executor | Copy of CPU physics or sampler | Driver and kernel enumeration | Container and upload plumbing | Diagnostics | Scheduling geometry |
|---|---|---|---|---|---|---|---|
| `gpu_pool.jl` (481) | 60 | 360 | | | 60 | | |
| `gpu_image.jl` (689) | | 50 | 45 | 235 | 210 | | 150 |
| `gpu_amr.jl` (696) | | 70 | 90 | 260 | 170 | 60 | 10 |
| `gpu_pol.jl` (345) | | | | 215 | | 130 | |
| `gpu_spacetime.jl` (331) | | | 70 | 200 | 60 | | |
| `mc_gpu.jl` (641) | | 30 | 230 | 330 | 50 | | |
| `gpu_tracer.jl` (149) | | | | 140 | | | |
| `gpu_analytic.jl` (130) | 10 | | | 115 | | | |
| `gpu_camera.jl` (118) | | 95 | | 15 | | | |
| `gpu_image_ad.jl` (92) | | | | 85 | | | |
| `gpu_metric.jl` (22) | | | 22 | | | | |
| **Total (≈)** | **70** | **605** | **460** | **1,595** | **550** | **190** | **160** |

### 13.2 Where each category goes

| Category | Lines today | Destination | Lines after |
|---|---|---|---|
| CUDA-only | 70 | `ext/BlackLightPlusCUDAExt.jl` | ≈ 80 |
| KA-generic executor | 605 | `Execution/pool.jl`, `executor.jl`, `kernel.jl`, `ray_sources.jl`, `worklist.jl` | ≈ 500 (`march!`/`march_chunk!`/`rays_on_device!`/`AMRStokes` fold into `foreach_item!` and `stream_blocks!`) |
| Copies of CPU physics and samplers | 460 | deleted; providers sample through the one body after `Adapt`; `ADMGrid{T,A}` replaces `DevADMParams`; `march_control` replaces `_gimg_stop` | 0 |
| Driver and kernel enumeration | 1,595 | `Products/programs.jl` (the program functors, ≈ 120) and the product bodies, which the CPU shares | ≈ 120 attributable to devices; the rest is the single CPU/GPU product layer |
| Container and upload plumbing | 550 | `Data/uniform_grid.jl`, `Data/series.jl` (`FrameWindow`, `advance`), `Execution/residency.jl` | ≈ 150; the uniform and AMR slow-light windows become one generic `FrameWindow{P}` |
| Diagnostics | 190 | `bench/stiffness.jl`, `test/infra/amr_probe.jl` as scripts over `foreach_item!` | moved, not kept in `src/` |
| Scheduling geometry | 160 | `Execution/schedule.jl`, run once in `prepare` | ≈ 150 |

Result: no file named `gpu_*`; about 1,000 lines of backend-neutral execution
infrastructure that the CPU path also uses; about 80 lines that need CUDA.
The user-visible effect is that

```julia
using BlackLightPlus, CUDA
include(joinpath(root, "src", "gpu_image.jl")); include(…"gpu_pol.jl"); include(…"gpu_amr.jl")
S = render_gpu_amr_polarized_slow(cfg, win; devices = [0, 1])
```

becomes

```julia
using BlackLightPlus, CUDA
plan = prepare(cfg; spacetime = KerrSchildCartesian(a), matter = window(series, ia), mode = IQUV())
S    = render(plan; backend = [0, 1])
```

and the same two lines with `backend = :cpu` run on the CPU. With no keyword,
`BlackLightPlus.backend!` or `BLP_BACKEND` decides, and `:auto` takes the GPUs
whenever CUDA is loaded and a device is idle (§5.4).

### 13.3 Mechanics

```julia
# Execution/kernel.jl — the only @kernel in the package
@kernel function _foreach!(prog, @Const(items), n, outs...)
    i = @index(Global)
    @inbounds i <= n && prog(items[i], i, outs...)
end

# Products/programs.jl — what the eleven march kernels become
struct ImagingProgram{M,S,C,P,T,F,R} <: Function
    mode::M; st::S; control::C; provider::P; params::T; frame::F; rays::R
end
Adapt.@adapt_structure ImagingProgram
@inline function (p::ImagingProgram)(item, i, out)
    y0, ok = ray(p.rays, item)
    @inbounds out[i] = ok ? march(p.mode, y0, p) : invalid(eltype(out))
    return nothing
end
```

- A program is built once per product call on the host and adapted once per
  device by `resident`; the kernel is launched per chunk with the chunk's
  item indices. The kernel specialises on `typeof(prog)`, so each combination
  of mode, spacetime, provider, parameter type and lane compiles to the code
  the hand-written kernel compiles to today, and the kernel audit enumerates
  program types instead of kernel names.
- The ray source is a program field: `CameraRays` computes the ray from the
  frame in-kernel (today's `_camray_kernel!` body), `HostRays` reads an
  adapted array. The separate ray kernel and `rays_on_device!` go.
- Workgroup size is a trait of the program (`workgroupsize(::ImagingProgram{IQUV}) = 64`),
  replacing `_GPOL_WG` and the `wg` keywords.
- The Monte Carlo primary and layer kernels are programs with no per-item
  output; they write through `Sink`s (atomic exact bins, child buffer) exactly
  as the kernels do today. Because the threaded executor runs the same
  program with the same atomic sinks, the CPU Monte Carlo drivers in
  `mc_cpu.jl` (the generate/track/record loops, ≈ 300 lines) disappear too.
- Slow light is not a kernel or a driver: `FrameWindow{P}` is a provider that
  blends two providers of type `P` at the emission time. One struct serves
  the uniform grid and the native AMR blocks, host and device.

### 13.4 Risks specific to the dissolution

- **Read-only load hints.** `@Const` applies to top-level kernel arguments
  only; the fluid arrays nested in a program need the same hint, applied when
  the device program is built (wrap the provider's arrays in
  `KernelAbstractions.Const`). The audit's PTX check (`ld.global.nc` on the
  corner reads) is the acceptance test; if the hint cannot be carried through
  a struct field, the provider arrays are hoisted to top-level kernel
  arguments by the executor, which keeps one kernel per provider arity (two).
- **Register pressure.** Program fields live in the kernel parameter space,
  as today's scalar arguments do; no change is expected, and the audit gate
  decides. The fallback is one kernel per transfer mode (three kernels).
- **Trajectory output.** The tracer program writes a `(width, nmax)` log per
  chunk on the device and per-ray vectors on the host; this is one program
  with two trajectory sinks, the only place the two executors differ.
- **Launch count.** Unchanged: one launch per chunk, as today.

If every risk materialises the floor is three kernels and two provider
arities, which is six specialisation families against twenty-one kernel
definitions today; the files, the include order and the enumerated drivers
go in every case.
