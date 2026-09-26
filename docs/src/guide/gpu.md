# GPU rendering

The GPU path collapses geodesic integration, fluid sampling, and radiative
transfer into a single kernel, one ray per thread, storing no trajectory — only
the image returns to the host. [Benchmarks](@ref) lists the measured
throughput, the machine, and which numbers predate the device pool.

## Why the drivers are included, not imported

CUDA is deliberately **not** a dependency of the package. The GPU drivers are
standalone files under `src/` that you `include` explicitly after loading CUDA
yourself. This keeps BlackLightPlus installable and usable on CUDA-free
machines, at the cost of one extra step here.

```julia
using BlackLightPlus, CUDA, KernelAbstractions

root = dirname(dirname(pathof(BlackLightPlus)))
include(joinpath(root, "src", "gpu_image.jl"))   # fused unpolarized imager
include(joinpath(root, "src", "gpu_pol.jl"))     # fused polarized imager
include(joinpath(root, "src", "gpu_amr.jl"))     # native AMR fluid sampling
```

!!! warning "Include order matters"
    `gpu_amr.jl` depends on definitions from `gpu_image.jl` and `gpu_pol.jl`.
    Include them in the order shown above.

## Rendering

```julia
snap = BlackLightPlus.Simulation.AthenaKSnapshot("torus.athdf")

img = render_gpu_amr(cfg, snap)               # Stokes I, native AMR sampling
S   = render_gpu_amr_polarized(cfg, snap)     # full IQUV
```

`render_gpu_amr` samples the AthenaK meshblock hierarchy natively on device and
is bit-for-bit identical to the CPU sampler. The alternative,
`render_gpu_fused`, resamples the AMR fluid once onto a uniform Cartesian box —
faster to set up, but it changes fluxes at the percent level and can rotate the
EVPA on Faraday-thick decks. Prefer `render_gpu_amr` for production.

## Choosing devices

Devices default to `default_pool()`: the idle GPUs that `select_devices()` finds,
probed once per session. To pin them:

```julia
img = render_gpu_amr(cfg, snap; devices = [0, 1])
```

or restrict visibility before starting Julia:

```sh
CUDA_VISIBLE_DEVICES=0,1 julia --project=. -t 4
```

Each device list gets one persistent `DevicePool`, created on first use: two worker
tasks per device, each with its own stream and pinned staging buffers. Start Julia with
at least one more thread than devices (`-t 3` for two GPUs) so the workers of different
devices run in parallel. Rays are queued in chunks and each worker pulls the next chunk
when it finishes the last, so a slower or shared device simply takes fewer. The queue holds
16×16-pixel tiles, those nearest the photon ring (the longest rays) first
(`cost_order`), so the slowest work starts early instead of ending the render; pass
`order` to set another queue order, e.g. `march_sort_order(counts)` from a previous
frame's trip counts. With `precision = :mixed` the near-ring pixels that need fp64 enter
the same queue ahead of the fp32 pixels. Each chunk sends only its pixel indices; the
camera rays are generated on the device by the same `camera_ray` function as the CPU
`camera_rays`. Every ray writes its own pixel, so the
image is bitwise identical for any device count and chunking. The design is in
[Method](@ref "Multi-device execution"); its GPU scaling is not yet measured
([Benchmarks](@ref)).

A snapshot or `UniformFluidGrid` passed to a driver is uploaded once per pool and reused
by later calls with the same object; do not modify it in place between renders (upload
a new object instead). Pre-uploaded grids (`upload_grid`, `upload_amr`) are used as
given. For explicit control, build a pool and pass it as `devices`:

```julia
pool = DevicePool([0, 1]; lanes = 2, chunk = 65536)
img  = render_gpu_amr(cfg, snap; devices = pool)
close(pool)
```

`devices = [HostDevice(1), HostDevice(2)]` runs the same drivers on the
KernelAbstractions CPU backend, one logical device per entry, which is how the
multi-device tests run without a GPU.

## Adaptive refinement on the GPU

`render_amr` refines image blocks where the intensity varies sharply. Pass the native-AMR
device marcher as its `stokes` to march the blocks on the GPUs:

```julia
S = render_amr(cfg, ""; block = 8, max_level = 2, polarized = true,
               stokes = AMRStokes(cfg, snap; polarized = true, devices = [0, 1]))
```

Every block is a work item of the device pool. The device builds a block's rays from
its origin and pixel size, and the worker that receives a block's Stokes values paints
it into the mosaic, tests it and queues its four children at once, so refined blocks
march while the rest of their parent level is still in flight. Painting and the
refinement test run on the host threads. The mosaic, level map and ray count are the
same as those of the level-by-level driver, for any device count.

## Other drivers

| File | Entry point | Purpose |
|---|---|---|
| `gpu_analytic.jl` | `render_analytic_gpu` | exact-Kerr elliptic tracer on device |
| `gpu_image_ad.jl` | `render_gpu_fused_grad` | image **and** its exact parameter derivative |
| `montecarlo/mc_gpu.jl` | `mc_spectrum_gpu` | Monte Carlo spectra on device |

## Gradients in one pass

`render_gpu_fused_grad` seeds a ForwardDiff dual on a physical scalar and
carries it through the transfer accumulation, so a single pass returns both the
image and its exact derivative with respect to that parameter, at about 1.13×
the cost of the plain image:

```julia
include(joinpath(root, "src", "gpu_image_ad.jl"))

img, dimg = render_gpu_fused_grad(cfg, snap; wrt = :mass_scale)
```

This is the differentiable-GRRT construction of Jipole
[NaetheMotta2025](@citep), applied inside the fused kernel. The derivatives
match finite differences to $7\times10^{-19}$ and $6\times10^{-9}$ for the mass
scale and $R_\mathrm{high}$ respectively.

## Verifying against the CPU

CPU and GPU run the same marches and per-sample physics (one function per
transfer mode and per coefficient), so on identical trajectories they agree to
floating-point contraction; a larger mismatch means the trajectories or the fluid
sampling differ, not the transfer:

```julia
cpu = render(cfg, snap; polarized = true)
gpu = render_gpu_amr_polarized(cfg, snap)
maximum(abs, cpu.I .- gpu.I)
```

## Where to go next

- [Benchmarks](@ref) — measured throughput and scaling.
- [Method](@ref "Method") — the reverse affine/Mueller accumulation the kernel uses.
