# Getting Started

From a clean installation to a rendered black-hole image, with no simulation data and
no GPU.

## Install

Julia 1.10 or later. Either add the package to an environment,

```julia
using Pkg
Pkg.add(url = "https://github.com/FoAKTEE/BlackLightPlus")
```

or work from a clone:

```sh
git clone https://github.com/FoAKTEE/BlackLightPlus.git
cd BlackLightPlus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

CUDA is optional: `using CUDA` in a session loads the extension that adds the GPU
devices ([GPU rendering](@ref)).

## A first image

The run is described by a configuration, a spacetime and a matter source, turned into a
plan once; the products take the plan. An equilibrium torus around a Kerr hole needs no
data:

```julia
using BlackLightPlus

cfg = RunConfig(m_bh_cgs = 6.2e9 * 1.989e33,          # the hole mass, g
                frequency_hz = 230.0e9,
                camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                      fov = 40.0, resolution = 128))
torus = PolishDoughnut(0.94; r_center = 9.0, n_e_center = 5.0e5,
                       theta_e_center = 40.0, beta = 8.0)
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.94), matter = torus, mode = IQUV())
img = render(plan; backend = :cpu)
```

`img.I`, `img.Q`, `img.U` and `img.V` are `128 × 128` matrices of specific intensity in
cgs units, laid out `[iβ, iα]` with the image vertical along the first axis; `img.run`
records the executor, the precision lanes, the package version and the wall time.

Save a plane as a NumPy file and plot it with whatever you use:

```julia
write_npy("torus_I.npy", img.I)
```

```python
import numpy as np, matplotlib.pyplot as plt
plt.imshow(np.load("torus_I.npy"), origin="lower", cmap="afmhot")
plt.savefig("torus_I.png", dpi=150)
```

## The exact-Kerr disk

For an equatorial source in Kerr the closed-form geodesic solution replaces the
integration. A plan without matter carries the camera; `render_analytic` images the disk
order by order:

```julia
cfg_a = RunConfig(camera = CameraConfig(inclination_deg = 17.0, distance = 1.0e4,
                                        fov = 20.0, resolution = 512))
plan_a = prepare(cfg_a; spacetime = KerrSchildCartesian(0.94), matter = nothing)
disk = render_analytic(plan_a, EquatorialDisk(0.94); bands = (0, 1, 2), backend = :cpu)
disk.I                        # the sum over the photon-ring orders 0, 1, 2
disk.Ib[2]                    # the order-1 image alone
```

## The plan and its products

`prepare(cfg; spacetime, matter, mode, lanes)` derives the camera, the pixel grid, the
transfer parameters, the march budget and the queue order once. The products:

| Product | Result |
|---|---|
| `render(plan)` | the image, Stokes $I$ or $IQUV$ by the plan's `mode` |
| `render_amr(plan; block, max_level, refine)` | the image on an adaptively refined pixel grid, the level map and the ray count |
| `render_analytic(plan, disk; bands)`, `photon_ring_layers(plan, disk)` | exact-Kerr images of an equatorial disk and the razor-thin photon rings |
| `trace(plan)` | the trajectories of every camera ray, source to camera |
| `render(plan; path = trace(plan))` | the two-phase render along stored trajectories |
| `shadow(plan)` | the fate of every ray without plasma: captured, escaped, step-limited |
| `sensitivity(plan, :mass_scale)` | the Stokes $I$ image and its exact derivative with respect to the mass scale or `:rat_high` |
| `variants(plan, mass_scales)` | one image per density mass scale from one trace |
| `mc_spectrum(plan, Ns; scattering)` | the Monte Carlo spectrum of the plan's matter |

A plan with `matter = nothing` serves `shadow` and the exact-Kerr products.

## Choosing where to run

Every product takes a `backend` keyword:

```julia
render(plan; backend = :cpu)       # Julia threads
render(plan; backend = :gpu)       # the idle CUDA devices (after `using CUDA`)
render(plan; backend = [0, 1])     # CUDA devices 0 and 1
backend!(:gpu)                     # the session default; backend!(nothing) clears it
backend()                          # the resolved default
```

The keyword wins over `backend!`, which wins over the environment variable
`BLP_BACKEND` (`auto | cpu | gpu | gpu:0,1 | host_pool | host_pool:N`), which wins over
`:auto`. `:auto` takes the GPUs when CUDA is loaded and a device is idle, else threads,
and logs its choice once per session. Start Julia with threads (`julia -t auto`) for the
CPU backend.

## Next

- [Imaging GRMHD data](@ref) for AthenaK snapshots, the plasma model, cuts, slow light
  and adaptive refinement.
- [Spacetimes and binaries](@ref) for user metrics, binaries at rest and in motion, and
  numerical-relativity slices.
- [GPU rendering](@ref) and [Monte Carlo spectra](@ref).
- [Method](@ref) for the physics and the numerical schemes.
