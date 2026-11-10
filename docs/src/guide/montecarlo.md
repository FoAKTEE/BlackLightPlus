# Monte Carlo spectra

`mc_spectrum` synthesizes spectra in the grmonty scheme [Dolence2009](@citep):
superphoton packets sample the synchrotron emissivity zone by zone, carry covariant
momentum along geodesics, decay by Kirchhoff absorption and, with `scattering = true`,
Compton-scatter off a relativistic Maxwellian electron population. The matter of the
plan decides the run: a uniform sphere in flat spacetime, or a GRMHD grid on any
spacetime, a moving binary included, the grid one snapshot held fixed while the
superphotons travel, or, over an emission window, a plasma that changes in time.

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

The Monte Carlo matches the closed form within the standard error of each bin. Every
spectrum carries `Σ (w E)²` per bin beside `Σ w E`, so its errors come with it:

```julia
using BlackLightPlus.Execution: nuLnu_sigma

nu, sigma, neff = nuLnu_sigma(spec)                  # standard error of nuLnu_total, effective photons
```

`neff = (Σ w E)²/Σ (w E)²` is the number of equal-weight photons that would give the
same error; a bin whose relative error is `1/√neff` of a few percent is converged.

## Scattering

```julia
comp = mc_spectrum(ball, 10^6; scattering = true)    # ScatterResult
comp.total                                           # the Spectrum summed over orders
comp.orders[2]                                       # the singly scattered photons
comp.n_primary, comp.n_children, comp.scatter_loss
```

Scattering uses the thermally averaged ("hot") Klein–Nishina cross section tabulated on a
log-log grid in photon energy and electron temperature; the table is built once per
process, at the first scattering run (`table` passes one in). On CPU threads each
scattering generation runs as a layer of a cascade over chunks of primaries, up to
`max_scatt` generations (3 by default); a layer whose child buffer overflows is rerun at
the counted size, so no child is lost or counted twice. On GPUs every generation shares
one cohort per device (below), and the result is the device cascade's bit for bit. On
the sphere the absorption is off by default (the idealized Comptonization sphere) and
`absorption = true` turns it on.

Every step's scattering events are a Poisson process whose mean is the step's
scattering depth times the importance bias `b`; each event spawns a child that carries
`1/b` of the photons the step scatters, at a point drawn from the step's own depletion,
so the estimate is unbiased for every `b`, with or without absorption. The default bias
follows the gas and the photon, `b = clamp(b0 A(Θₑ) w/w_ref, 1, bias_max)` with
`A = 1 + 4Θₑ + 16Θₑ²` the mean energy gain of a scattering and `w_ref` the mean primary
weight: a child carries the energy of a mean primary over `b0`, whatever the gas that
scattered it. `bias = b0` (1 by default) raises the number of children; the Compton tail
of a hot accretion flow needs `b0` of 10³–10⁴. Children follow the photon weight, so where
the top of a scattered spectrum is fed by seeds of high frequency, whose packets carry few
photons, the state bias gives them few children, and a child's own bias is near 1, so the
light scattered two and three times is sampled less: those bins show a small effective
photon number in `nuLnu_sigma`. `bias_order = f` multiplies the bias of a photon scattered
`n` times by `fⁿ`, which samples the higher orders and the tail they make; a larger `b0`
or the constant bias also samples them. `bias = Bias(b0)` is the constant bias
`b0 (n + 1)²` of order `n`, `bias_max` caps the bias, and `split = m` spawns up to `m`
lighter children for an event whose bias was capped. The result holds the ledger of the
run: `w_children + w_unspawned` estimates `scatter_loss`, the weight scattered out of the
beams that may spawn, and `emitted_power`, `escaped_power`, `absorbed_power`,
`captured_power` and `dropped_power` are its power budget.

## GRMHD spectra

On a plan with a snapshot the primaries are spread over the emission zones of the
plan, one per owned cell above the horizon, the geodesic march uses the plan's step
divisor `C`, and the result holds one spectrum per inclination band:

```julia
using BlackLightPlus.Products: incl_nuLnu

plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = snap)
res = mc_spectrum(plan, 10^6; scattering = true)     # GRMHDSpectrumResult
nu, L = incl_nuLnu(res)                              # L[k, :] for band k
res.incl_deg, res.half_width_deg, res.emitted_power, res.escaped_power
```

The default bands are three inclination bands of half-width 10° about 17°, 50° and 90°;
`bands = BlackLightPlus.Execution.InclinationBands((30, 60, 90), 10)` changes them.
`res.orders[n + 1]` is the spectrum of the photons scattered `n` times, every band and
the all-sky row, and the ledger is that of the sphere's result. Expect the
gravitationally redshifted synchrotron bump, with a Compton tail growing with the
optical depth.

The angle-integrated emissivity of every zone comes from a one-dimensional table of the
synchrotron fit, so the zone tables of a 10⁷-cell snapshot take seconds and the run uses
every cell. For a quick look, `stride = s` keeps one randomly drawn owned cell per group of
`s³` cells with the group's volume: an unbiased but noisier quadrature of the same
emission. `r_emit_max` and `max_zones` limit the zones.

## Light curves and time-resolved spectra

With `window = (t_a, t_b)` the plasma emits over that span of coordinate time, every
superphoton reads plasma and metric at its own time, and the escaped photons are also binned
by their arrival time at a distant observer. A sphere whose density varies in time shows the
light-travel smearing across the ball:

```julia
using BlackLightPlus
using BlackLightPlus.Matter: UniformSphere, VariableSphere
using BlackLightPlus.Products: light_curve, time_spectrum

ω = 1.5; P = 2π / ω
times = collect(range(0.0, 6P; length = 145))                     # 24 nodes per period
vs = VariableSphere(UniformSphere(R = 1.0), times, 1 .+ 0.5 .* sin.(ω .* times))
res = mc_spectrum(prepare(vs), 10^6; window = (0.0, 6P), time_bins = (0.0, 6P, 72))
t, L, sigma, count = light_curve(res)                            # erg/s in each arrival bin
nu, nuLnu, sig, n = time_spectrum(res, 30)                       # the spectrum of bin 30
res.result                                                       # the window-mean spectrum
res.interior                                                     # bins that see the whole ball
```

The light curve's modulation is the emitted one times `3 j₁(ωR)/(ωR)`. `light_curve` takes a
band row (`row`, 0 for the whole sky), a frequency range (`nu_min`, `nu_max`) and a scattering
order (`order`); `time_ebins` sets the frequency bins of the time-resolved spectra.

On a plasma grid the matter may be a snapshot held fixed (on a stationary spacetime), a series
of frames (`BlackLightPlus.Data.FrameSequence`, frames at given coordinate times, on a stationary
spacetime) or a snapshot carried rigidly with an orbiting binary
(`BlackLightPlus.Data.Corotating`, on a spacetime with that helical symmetry); `sectors` splits
every inclination band into azimuth sectors, so each sector is one observer azimuth:

```julia
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit
using BlackLightPlus.Data: Corotating

orbit = CircularOrbit((0.5, 0.5), 25.0)
binary = BoostedSuperposedKerrSchild([0.5, 0.5], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
plan = prepare(cfg; spacetime = binary, matter = Corotating(snap, orbit.omega))
P = 2π / orbit.omega
res = mc_spectrum(plan, 10^6; scattering = true, window = (0.0, 2P), sectors = 16,
                  time_bins = (0.0, 2P, 64))
```

With a snapshot held fixed and no window, or with a window on a stationary spacetime, the
window-mean spectrum `res.result` is that of the stationary run bit for bit. The run over a
window uses the CPU threads.

## Reproducibility

Randomness is the counter-based Philox4x32-10 generator [Salmon2011](@citep). Every
photon's stream is named by its genealogy, the primary by its global index and a
scattered child by its parent's stream and the ordinal of its scattering event along the
parent's march, so no stream depends on the thread, chunk, device or buffer slot that
carries the photon. Spectrum bins
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

A stationary run takes the device wavefront: each device advances a cohort of photons of
every generation a few steps per launch, adds the children a launch spawns to the cohort
and refills it with primaries ([GPU rendering](@ref) explains it; `capacity` sets the
first size of the child queue, and a launch whose children overflow it runs again). A run
over an emission window uses the CPU threads. The measured throughput is on
[Benchmarks](@ref). The sampling,
biasing and normalization are on the [Method](@ref) page.
