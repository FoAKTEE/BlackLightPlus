# Monte Carlo spectra

The Monte Carlo module synthesizes spectra including Compton scattering,
following the grmonty scheme [Dolence2009](@citep). Superphoton packets sample
the synchrotron emissivity zone by zone, carry covariant momentum along
geodesics, decay by Kirchhoff absorption, and scatter off a relativistic
Maxwellian electron population.

## A thin synchrotron sphere

The simplest model is an optically thin synchrotron sphere, which has a closed
form to check against:

```julia
using BlackLightPlus

spec = mc_spectrum(SphereModel(), 1e6)     # 10^6 superphotons
nu, L, _ = nuLnu_total(spec)               # ν L_ν [erg/s] against ν [Hz]
```

`analytic_nuLnu` gives the closed-form answer for the same model; the Monte
Carlo matches it to a few percent per bin.

## Adding scattering

`ScatteringSphere` turns on Compton scattering:

```julia
spec = mc_spectrum_scatter(ScatteringSphere(; tau_es = 0.1), 1e6)
```

Scattering uses the thermally averaged ("hot") Klein–Nishina cross section,
tabulated on a log-log $(\varepsilon, \Theta_e)$ grid. Building that table is
the expensive setup step, so build it once and pass it in:

```julia
table = build_hotcross_table()
```

## GRMHD spectra

```julia
model = GRMHDModel(
    "torus.athdf";
    bh_a = 0.9,
    m_bh_cgs = 6.2e9 * 1.989e33,
    mass_scale_cgs = 1.0e24,
)

res = mc_spectrum_grmhd(model, 1e6; table = build_hotcross_table())
```

`res` carries per-inclination-band spectra in `incl_nuLnu`, so one run gives
spectra at every viewing angle rather than one:

```julia
nu, L = incl_nuLnu(res, 3)     # the 3rd inclination band
```

Expect the gravitationally redshifted synchrotron bump, with a
Compton-scattered tail extending to higher frequency as the optical depth
rises.

## Reproducibility

Randomness comes from the counter-based Philox4x32-10 generator
[Salmon2011](@citep). It is stateless and keyed per superphoton, so runs
replay exactly:

```julia
a = mc_spectrum(SphereModel(), 1e5; seed = 42)
b = mc_spectrum(SphereModel(), 1e5; seed = 42)
nuLnu_total(a) == nuLnu_total(b)     # true
```

Every stream is named by the photon's genealogy: a primary by its global
index, a scattered child by its parent's stream and the ordinal of the
scattering event along the parent's march. No stream depends on the thread,
chunk, device or buffer slot that processes the photon. Spectra and
diagnostics are accumulated exactly, as integer multiples of $2^{-1074}$, and
rounded once at the end. Results with and without scattering are therefore
bitwise identical for any thread count, device count and scheduling. On the
KernelAbstractions CPU backend the device kernels reproduce the CPU drivers
bit for bit. A CUDA run replays the CPU result photon-for-photon at matched
seeds, with a spectrum ratio of 1.000000 and a worst-bin difference of
$1.1\times10^{-15}$.

## Running on GPU

```julia
using BlackLightPlus, CUDA
root = dirname(dirname(pathof(BlackLightPlus)))
include(joinpath(root, "src", "montecarlo", "mc_gpu.jl"))

spec = mc_spectrum_gpu(SphereModel(), 1e8)
```

One A100 reaches $3.8\times10^{7}$ photons/s on the thin sphere and
$2.7\times10^{7}$ with scattering, against $1.4\times10^{6}$ for a 64-thread
CPU run. Multi-order scattering is layered, not recursive: the host
relaunches once per scattering generation, and children are appended to a
device buffer. A launch that overflows the buffer is rerun with the exact
size, so no child is dropped.

## Where to go next

- [Method](@ref "Method") — the sampling, biasing, and normalization scheme.
- [Benchmarks](@ref) — measured throughput.
