# GPU drivers API

!!! note "These are not exported"
    CUDA is deliberately not a dependency of BlackLightPlus. The GPU drivers
    are standalone files under `src/` that are **not** included by the package;
    you include them explicitly after loading CUDA yourself. Because they are
    not part of the loaded module, their docstrings cannot be pulled in
    automatically, so this page documents them by hand.

    See [GPU rendering](@ref) for the task-oriented walkthrough.

## Loading

```julia
using BlackLightPlus, CUDA, KernelAbstractions

root = dirname(dirname(pathof(BlackLightPlus)))
include(joinpath(root, "src", "gpu_image.jl"))
include(joinpath(root, "src", "gpu_pol.jl"))
include(joinpath(root, "src", "gpu_amr.jl"))
```

Include order matters: `gpu_amr.jl` depends on definitions from the two files
above it.

## Imaging kernels

### `render_gpu_fused(cfg, snapshot; devices=default_pool())`

Fused single-pass unpolarized imager. Geodesic integration, fluid sampling, and
radiative transfer run in one kernel, one ray per thread, storing no
trajectory. The fluid is resampled once onto a uniform Cartesian box, which
changes fluxes at the percent level. Returns a Stokes $I$ image.

*File:* `src/gpu_image.jl`

### `render_gpu_amr(cfg, snapshot; devices=default_pool())`

As above, but samples the AthenaK meshblock hierarchy natively on device: a
dense finest-level leaf-index grid gives an $O(1)$ block locate, followed by
the same clamped trilinear interpolation as the CPU reference. Bit-for-bit
identical to the host sampler. **Prefer this for production.**

*File:* `src/gpu_amr.jl`

### `render_gpu_amr_polarized(cfg, snapshot; devices=default_pool())`

Full IQUV on device with native AMR sampling. Accumulates the segment couplings
in reverse with a running transmission, so per-ray state is $O(1)$ regardless
of path length. Reproduces the CPU coherency march bit-for-bit on identical
trajectories. Returns a Stokes `NamedTuple`.

*File:* `src/gpu_pol.jl`

### `AMRStokes(cfg, snapshot; polarized=false, devices=default_pool())`

Native-AMR device marcher for the `stokes` argument of `render_amr`: refinement blocks
are streamed through the device pool, rays are made on the device from each block's
origin and pixel size, and children are queued as soon as their parent is tested, with
no barrier between levels. Also callable as `stokes(coords)`.

*File:* `src/gpu_amr.jl`

### `render_analytic_gpu(cfg; source=...)`

The exact-Kerr elliptic tracer on device. Matches the CPU tracer per pixel to
round-off, with a median relative difference of $2.5\times10^{-16}$.

*File:* `src/gpu_analytic.jl`

## Differentiable imaging

### `render_gpu_fused_grad(cfg, snapshot; wrt)`

Returns the image **and** its exact derivative with respect to a physical
scalar, in a single pass at about 1.13× the cost of the plain image. Metric
derivatives come from ForwardDiff dual numbers evaluated inside the kernel; no
Christoffel tensor is ever materialized. This is the differentiable-GRRT
construction of Jipole [NaetheMotta2025](@citep).

`wrt` selects the parameter — the density scale or $R_\mathrm{high}$.

*File:* `src/gpu_image_ad.jl`

## Monte Carlo

### `mc_spectrum_gpu(model, n; devices=select_devices())`

Monte Carlo spectral synthesis on device. Multi-order scattering is layered,
not recursive: the host relaunches once per scattering generation. Devices
run chunks of `chunk` (default 2²⁰) contiguous global photon ids on the device pool;
the host tables are built once per call and uploaded once per device. Streams are
keyed by genealogy and bins are exact integer-limb sums merged in chunk order, so the
output is bitwise independent of the device count and the chunking. `devices` accepts
CUDA ordinals, `HostDevice`s (the KernelAbstractions CPU backend) or a `DevicePool`.

*File:* `src/montecarlo/mc_gpu.jl`

## Devices and the pool

### `select_devices()`

Probes for idle CUDA devices and returns their indices.

### `default_pool()`, `device_pool(devices)`, `DevicePool(devices; lanes=2, chunk=65536)`

Every driver takes `devices` as a list (CUDA ordinals or `HostDevice`s) or a
`DevicePool`. `device_pool(devices)` returns the one persistent pool of a device list;
`default_pool()` is the pool of `select_devices()`, probed once per session. A pool runs
`lanes` worker tasks per device and queues the rays in `chunk`-ray chunks that the
workers pull as they finish, writing each ray's output to its own pixel, so the result
is bitwise independent of the device count. Snapshots and host grids passed to a driver
are uploaded once per pool. `close(pool)` stops its workers.

*File:* `src/gpu_pool.jl` (included by the driver files)
