# Imaging GRMHD data

Images of AthenaK GRMHD snapshots: the snapshot as the plan's matter, the physical scale,
the electron model and cuts, slow light over a snapshot series, adaptive refinement, and
parameter ladders and derivatives from one trace. [Getting Started](@ref) covers the plan
and the products.

## A snapshot as matter

`AthenaKSnapshot` reads an `.athdf` dump (the `mhd_w_bcc` output with a pressure column
needs the adiabatic index `gamma`; an internal-energy column does not); `athenak_bin_snapshot`
in the `Data` module reads the `.bin` layout. The snapshot is a grid provider: it is
sampled on its meshblock hierarchy with a trilinear stencil, and nodes outside the mesh
are vacuum.

```julia
using BlackLightPlus

snap = AthenaKSnapshot("torus.mhd_w_bcc.00999.athdf"; gamma = 13 / 9)
L = 6.2e9 * 1.989e33 * 6.674e-8 / 2.998e10^2          # GM/c² of the hole, cm
cfg = RunConfig(m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1.0e-21 * L^3,
                frequency_hz = 230.0e9,
                camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                      fov = 30.0, resolution = 256),
                plasma = PlasmaConfig(rat_low = 1.0, rat_high = 10.0, sigma_cut = 1.0))
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = snap, mode = IQUV())
img = render(plan)
```

A loaded snapshot serves as many plans as needed; prepare one per camera:

```julia
for inc in (10.0, 30.0, 60.0, 90.0)
    c = RunConfig(cfg; camera = CameraConfig(cfg.camera; inclination_deg = inc))
    p = prepare(c; spacetime = KerrSchildCartesian(0.9), matter = snap, mode = StokesI())
    write_npy("img_i$(Int(inc)).npy", render(p).I)
end
```

## The physical scale

A snapshot is in code units; three fields of `RunConfig` tie it to cgs:

| Field | Meaning |
|---|---|
| `m_bh_cgs` | the hole mass in grams; sets the length unit $GM/c^2$ and the time unit |
| `mass_scale_cgs` | the density mass scale $\mathcal M$ in grams: the code density times $\mathcal M / (GM/c^2)^3$ is the cgs density |
| `frequency_hz` | the observing frequency at infinity |

`mass_scale_cgs` is the free parameter that sets the accretion rate and is tuned to an
observed flux. `variants` renders a ladder of mass scales from one trace:

```julia
lad = variants(plan, [1e23, 3e23, 1e24, 3e24])
lad.images[3]                      # the Stokes I image at 1e24 g
```

and `sensitivity` returns the exact derivative of the Stokes $I$ image with respect to
the mass scale or to `rat_high`, by forward-mode differentiation of the transfer:

```julia
s = sensitivity(plan, :mass_scale)
s.image.I, s.derivative, s.flux, s.dflux
```

## Electron temperature and cuts

`PlasmaConfig` carries the $R_{\rm high}/R_{\rm low}$ prescription
[Moscibrodzka2016](@citep), $T_i/T_e = (R_{\rm high} \beta^2 + R_{\rm low})/(1 + \beta^2)$,
the split of the fluid pressure (`use_p`, the total pressure by default, else the
internal energy with the adiabatic indices `gamma_i`, `gamma_e`), the population fractions
`thermal_frac`, `power_frac`, `kappa_frac` with the power-law and kappa parameters, and the
emission cuts:

```julia
plasma = PlasmaConfig(rat_high = 40.0, rat_low = 1.0,
                      sigma_cut = 1.0,            # no emission where b²/ρ > 1
                      theta_e_min = -1.0,         # no temperature floor
                      power_frac = 0.1, power_p = 3.0, gamma_min = 25.0, gamma_max = 1e7)
```

The magnetization cut removes the funnel, whose density a GRMHD simulation does not
resolve. Other cuts (a plasma-β threshold, a half-space) are applied at the primitive
level by wrapping the provider: define a `GridProvider` whose `primitives` returns the
snapshot's primitives inside the region and the vacuum record outside, as the Blacklight
figure generator does (`figures/blacklight/common.jl`).

## Slow light

A `SnapshotSeries` holds dumps at ascending code times with a two-frame cache; the
observation epoch `t_obs` is the camera time. On the host, `SeriesProvider` samples the
series at each ray's emission time $t_{\rm obs} + x^0$, loading and evicting frames as
the rays need them; on a device, a `FrameWindow` holds two bracketing frames and blends
them linearly in time:

```julia
paths = ["torus.mhd_w_bcc.$(lpad(k, 5, '0')).athdf" for k in 990:999]
times = [54030.0 + 30.0 * (k - 181) for k in 990:999]
series = SnapshotSeries(paths, times; t_obs = times[end])

plan_slow = prepare(cfg; spacetime = KerrSchildCartesian(0.9),
                    matter = SeriesProvider(series), mode = IQUV())
img_slow = render(plan_slow; backend = :cpu)

win = BlackLightPlus.Data.window(series, 9)          # frames 9 and 10 as one provider
plan_win = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = win, mode = IQUV())
```

The series must cover every emission time the rays reach: for a camera at 1000 M the
rays spend about a light-crossing time of the emitting region in the plasma, so the
frames before `t_obs` have to span it. The method is on the [Method](@ref) page
("Slow light").

## Adaptive refinement

`render_amr` tiles the image into `block × block` pixel blocks and replaces a block by
four children at twice the pixel density, down to `max_level`, when its Stokes $I$ values
satisfy the refinement indicator [White2022](@citep):

```julia
using BlackLightPlus.Analysis: RelativeGradient, AbsoluteGradient

amr = render_amr(plan; block = 16, max_level = 2, refine = RelativeGradient(0.15; frac = 0.03))
amr.image.I                        # the mosaic at resolution × 4 per side
amr.levels                         # the finest level per base block
amr.nrays                          # rays marched
```

`AbsoluteGradient(cut; count)` fires on an absolute pixel-to-pixel gradient on at least
`count` pixels, `RelativeLaplacian` on the relative Laplacian, and any callable
`f(I) -> Bool` serves as an indicator. CPU and GPU backends make identical refinement
decisions.

## Uniform grids and trajectories

`UniformGrid(snap, box_half, n)` resamples a snapshot onto `n³` nodes of a box, a cheaper
provider for repeated renders at modest accuracy. `trace(plan)` returns every camera
ray's states and affine steps, source to camera, and `render(plan; path = paths)` runs the
transfer along stored trajectories, so several renders share one trace:

```julia
paths = trace(plan; backend = :cpu)
img_a = render(plan; path = paths, backend = :cpu)
```

`shadow(prepare(cfg; spacetime = st, matter = nothing)).fate` gives the capture map of the
camera (1 captured, 2 escaped, 0 step-limited) without any plasma.

## Saving and analysing

`write_npy(path, A)` writes a matrix as a NumPy file. For interferometric observables,
`radon_cut(img.I, angle_deg; fov_muas)` projects the image onto a baseline position angle
and `visamp(img.I; angles, fov_muas)` returns the visibility amplitude along baselines,
with `visamp_minima` locating its minima, whose spacing measures the ring diameter.

## Known input issues

These concern the AthenaK binary-black-hole runs imaged by `figures/paper/fig_eht_bbh.jl`,
not the ray tracer.

- **Spin in units of the total mass.** The AthenaK `dynbbh` problem generator sets each
  hole's Kerr–Schild spin to the header value `a1`, `a2` in units of the total mass while
  the hole mass is $1/(1+q)$, so the spinning runs (snapshots 00093 and 00122,
  $a_1 = a_2 = 0.8$, $q = 1$) carry a Kerr–Schild term with $a/m = 1.6$. The generator
  images such a hole at dimensionless spin $\chi = a$ and records the mapping in each run
  record.
- **Boosted terms.** The same generator evaluates each hole's term in the rest frame of
  its velocity and differentiates the metric in time along the trajectory; the static
  `SuperposedKerrSchild` differs from the simulation's metric by the boost terms, which
  reach 20–27% of a hole's metric perturbation at the orbital speeds of these dumps.
  `BoostedSuperposedKerrSchild` ([Spacetimes and binaries](@ref)) carries them.
- **Snapshot 00093** holds a cold, dense ball at $r < 8$ about the origin and the transient
  of its restart; 28–46% of its 230 GHz flux lies outside a 56 M field of view.
- **Orbital phase of snapshot 00455.** The hole positions follow from the analytic
  trajectory, $\phi_0 = {\rm rem2pi}(\Omega t)$ with $\Omega = {\rm sep}^{-3/2}$; the dump
  header stores the time to six digits, so the generator takes $t$ from the run's `tlim`
  when it lies within 0.5 of the header time.

## Where to go next

- [GPU rendering](@ref) for the same images on one or more GPUs.
- [Spacetimes and binaries](@ref) for non-Kerr backgrounds.
- [Method](@ref) for the transfer scheme and the coefficient fits.
