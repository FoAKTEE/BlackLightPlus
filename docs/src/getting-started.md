# Getting Started

This page takes you from a clean checkout to a rendered black-hole image. It
needs no simulation data and no GPU. Follow it top to bottom; the
[Guides](@ref "Imaging GRMHD data") cover options and alternatives once you
have this working.

## Requirements

- Julia 1.10 or later.
- About 2 GB of disk for the dependency set.
- CUDA is **not** required. It is needed only for the GPU drivers described in
  [GPU rendering](@ref).

## Install

```sh
git clone https://github.com/FoAKTEE/BlackLightPlus.git
cd BlackLightPlus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` is committed, so `instantiate` reproduces the exact dependency
set the benchmarks and figures were produced with.

## Render your first image

The exact-Kerr elliptic tracer images an equatorial disk analytically, with no
input data. Start Julia in the project and run:

```julia
using BlackLightPlus

cfg = RunConfig(
    bh_a = 0.94,
    tracer = :analytic,
    camera = CameraConfig(
        inclination_deg = 17.0,
        distance = 1.0e4,
        fov = 20.0,
        resolution = 512,
    ),
)

img = render_analytic(cfg; source = EquatorialDisk(0.94))
```

`img.I` is a 512×512 matrix of Stokes $I$. The spin is $a/M = 0.94$, the
observer sits at $17°$ inclination — close to the viewing angle inferred for
M87\* — and the field of view is 20 gravitational radii.

## Save it

```julia
write_npy("first_image.npy", img.I)
```

The result is a NumPy `.npy` file, readable with `numpy.load`, so you can plot
it with whatever you normally use:

```python
import numpy as np, matplotlib.pyplot as plt
plt.imshow(np.load("first_image.npy"), origin="lower", cmap="afmhot")
plt.savefig("first_image.png", dpi=150)
```

You should see a bright, asymmetric ring: the direct ($n=0$) image of the disk,
Doppler-boosted on the approaching side, wrapped around the black-hole shadow.

## What just happened

`render_analytic` took the closed-form Kerr geodesic solution rather than
integrating an ODE, so the image came back in well under a second. Every ray
was traced backwards from the camera to the equatorial plane using Legendre and
Jacobi elliptic integrals, and the disk emission was evaluated at the landing
point. The [Method](@ref) page explains the formulation.

## Next steps

- **Real simulation data** — [Imaging GRMHD data](@ref) covers the two-phase
  `render` pipeline on AthenaK snapshots, including full IQUV polarization.
- **Other spacetimes** — [Numerical-relativity and binary
  spacetimes](@ref) covers binary black holes and Z4c numerical-relativity
  slices.
- **Speed** — [GPU rendering](@ref) covers the fused single-pass kernels.
- **Spectra** — [Monte Carlo spectra](@ref) covers Compton-scattering spectral
  synthesis.
