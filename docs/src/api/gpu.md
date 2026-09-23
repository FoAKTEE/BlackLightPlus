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

### `render_gpu_fused(cfg, snapshot; devices=select_devices())`

Fused single-pass unpolarized imager. Geodesic integration, fluid sampling, and
radiative transfer run in one kernel, one ray per thread, storing no
trajectory. The fluid is resampled once onto a uniform Cartesian box, which
changes fluxes at the percent level. Returns a Stokes $I$ image.

*File:* `src/gpu_image.jl`

### `render_gpu_amr(cfg, snapshot; devices=select_devices())`

As above, but samples the AthenaK meshblock hierarchy natively on device: a
dense finest-level leaf-index grid gives an $O(1)$ block locate, followed by
the same clamped trilinear interpolation as the CPU reference. Bit-for-bit
identical to the host sampler. **Prefer this for production.**

*File:* `src/gpu_amr.jl`

### `render_gpu_amr_polarized(cfg, snapshot; devices=select_devices())`

Full IQUV on device with native AMR sampling. Accumulates the segment couplings
in reverse with a running transmission, so per-ray state is $O(1)$ regardless
of path length. Reproduces the CPU coherency march bit-for-bit on identical
trajectories. Returns a Stokes `NamedTuple`.

*File:* `src/gpu_pol.jl`

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
take contiguous ranges of global photon ids. Streams are keyed by genealogy
and bins are exact integer-limb sums, so the output is bitwise independent of
the device count. `devices` accepts CUDA ordinals or `HostDevice`s (the
KernelAbstractions CPU backend).

*File:* `src/montecarlo/mc_gpu.jl`

## Device selection

### `select_devices()`

Probes for idle CUDA devices and returns their indices. Override by passing
`devices=[0,1]` to any driver, or by setting `CUDA_VISIBLE_DEVICES` before
starting Julia. Rays are split across devices in interleaved order so that
cheap and expensive rays balance.
