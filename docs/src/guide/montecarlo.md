# Monte Carlo spectra

`mc_spectrum` synthesizes spectra in the grmonty scheme [Dolence2009](@citep):
superphoton packets sample the synchrotron emissivity zone by zone, carry covariant
momentum along geodesics, decay by Kirchhoff absorption and, with `scattering = true`,
Compton-scatter off a relativistic Maxwellian electron population. The matter of the
plan decides the run: a uniform sphere in flat spacetime, or a GRMHD grid on a
stationary spacetime.

## A thin synchrotron sphere

A static ball of thermal plasma in flat spacetime has a closed-form optically thin
spectrum to check against:

```julia
using BlackLightPlus
using BlackLightPlus.Execution: nuLnu_total
using BlackLightPlus.Products: analytic_nuLnu

ball = prepare(UniformSphere(n_e = 1e13, theta_e = 100.0, B = 1.0, R = 1.0))  # cgs, R in cm
spec = mc_spectrum(ball, 10^6)                       # about 10⁶ superphotons
nu, L, count = nuLnu_total(spec)                     # ν L_ν [erg/s] at the bin centres nu
analytic_nuLnu(ball, nu[100])                        # the closed form at one frequency
```

The Monte Carlo matches the closed form to a few percent per bin where the bins are
populated.

## Scattering

```julia
comp = mc_spectrum(ball, 10^6; scattering = true)    # ScatterResult
comp.total                                           # the Spectrum summed over orders
comp.orders[2]                                       # the singly scattered photons
comp.n_primary, comp.n_children, comp.scatter_loss
```

Scattering uses the thermally averaged ("hot") Klein–Nishina cross section tabulated on a
log-log grid in photon energy and electron temperature; the table is built at the plan's
first scattering run and kept in the plan (`table` passes one in). Each scattering
generation runs as a layer of a cascade over chunks of primaries, up to `max_scatt`
generations (3 by default); a layer whose child buffer overflows is rerun at the counted
size, so no child is lost or counted twice. The sphere's Thomson depth sets the default
importance bias (`bias`); on the sphere the absorption is off by default (the idealized
Comptonization sphere) and `absorption = true` turns it on.

## GRMHD spectra

On a plan with a snapshot or a uniform grid the primaries are spread over the emission
zones of the plan (`stride`, `r_emit_max`, `max_zones` select them), the geodesic march
uses the plan's step divisor `C`, and the result holds one spectrum per inclination band:

```julia
using BlackLightPlus.Products: incl_nuLnu

plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = snap)
res = mc_spectrum(plan, 10^6; scattering = true)     # GRMHDSpectrumResult
nu, L = incl_nuLnu(res)                              # L[k, :] for band k
res.incl_deg, res.half_width_deg, res.emitted_power, res.escaped_power
```

The default bands are three inclination bands of half-width 10° about 17°, 50° and 90°;
`bands = BlackLightPlus.Execution.InclinationBands((30, 60, 90), 10)` changes them. Expect
the gravitationally redshifted synchrotron bump, with a Compton tail growing with the
optical depth.

## Reproducibility

Randomness is the counter-based Philox4x32-10 generator [Salmon2011](@citep). Every
photon's stream is named by its genealogy, the primary by its global index and a
scattered child by its parent's stream and the march step of the scattering, so no stream
depends on the thread, chunk, device or buffer slot that carries the photon. Spectrum bins
and diagnostics are accumulated exactly, as integer multiples of $2^{-1074}$ in
multi-limb sums, and rounded once at the end. The same plan, `Ns` and `seed` therefore
give the same bins on every backend, for any thread count, device count and schedule:

```julia
a = mc_spectrum(ball, 10^5; seed = 42)
b = mc_spectrum(ball, 10^5; seed = 42, backend = :cpu)
nuLnu_total(a) == nuLnu_total(b)                     # true
```

## On GPUs

```julia
using CUDA
spec = mc_spectrum(ball, 10^8; backend = :gpu)
```

The cascade runs on each device lane with two swapped child buffers (`capacity` sets
their initial size); the measured throughput is on [Benchmarks](@ref). The sampling,
biasing and normalization are on the [Method](@ref) page.
