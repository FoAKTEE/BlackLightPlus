# Imaging GRMHD data

This guide renders images from AthenaK GRMHD snapshots. It assumes you have
completed [Getting Started](@ref) and have an AthenaK `.athdf` or `.bin`
snapshot to hand.

## The basic render

`render` runs the two-phase pipeline — trace geodesics, then integrate
radiative transfer along them — and returns a Stokes `NamedTuple` of
`resolution × resolution` images:

```julia
using BlackLightPlus

cfg = RunConfig(
    bh_a = 0.9,
    m_bh_cgs = 6.2e9 * 1.989e33,     # M87*, in grams
    mass_scale_cgs = 1.0e24,          # density unit
    frequency_hz = 230.0e9,           # 230 GHz
    camera = CameraConfig(inclination_deg = 60.0, fov = 40.0, resolution = 256),
    plasma = PlasmaConfig(rat_high = 10.0, sigma_cut = 1.0),
)

img_unpol = render(cfg, "torus.athdf"; polarized = false)   # Stokes I only
img_pol   = render(cfg, "torus.athdf"; polarized = true)    # full IQUV
```

`img_pol` has fields `I`, `Q`, `U`, and `V`.

## Setting the physical scale

A GRMHD snapshot is in code units; three configuration fields tie it to cgs:

| Field | Meaning |
|---|---|
| `m_bh_cgs` | black-hole mass in grams — sets the length and time units |
| `mass_scale_cgs` | density unit, the free scaling that sets the accretion rate |
| `frequency_hz` | observing frequency in the observer's frame |

`mass_scale_cgs` is the parameter you tune to match an observed flux. Doubling
it roughly doubles the optically thin flux.

## Electron temperature and plasma cuts

`PlasmaConfig` carries the $R_\mathrm{high}/R_\mathrm{low}$ prescription
[Moscibrodzka2016](@citep), which sets the ion-to-electron temperature ratio as
a function of plasma $\beta$:

```julia
plasma = PlasmaConfig(
    rat_high = 10.0,      # T_i/T_e in the disk body (high beta)
    rat_low = 1.0,        # T_i/T_e in the jet/funnel (low beta)
    sigma_cut = 1.0,      # omit zones with magnetization above this
    theta_e_min = 0.0,    # omit zones with electron temperature below this
)
```

`sigma_cut` matters: the funnel region of a GRMHD simulation is numerically
unreliable at high magnetization, and the standard practice is to exclude it
rather than trust it.

## Adaptive refinement

`render_amr` adds adaptive image-plane refinement, tiling the image into blocks
and subdividing those with steep gradients [White2022](@citep):

```julia
img = render_amr(cfg, "torus.athdf"; max_level = 2)
```

With a base resolution of 384 and two levels, the effective resolution is
1536², at roughly a quarter of the ray count of a uniform 1536² image. CPU and
GPU backends make identical refinement decisions.

## Reusing a loaded snapshot

Large snapshots are expensive to read. Load once and reuse across camera
angles:

```julia
snap = BlackLightPlus.Simulation.AthenaKSnapshot("torus.athdf")

for inc in (10.0, 30.0, 60.0, 90.0)
    c = RunConfig(cfg; camera = CameraConfig(cfg.camera; inclination_deg = inc))
    write_npy("img_i$(Int(inc)).npy", render(c, snap).I)
end
```

## Saving output

```julia
write_npy("stokes_I.npy", img.I)
```

For interferometric observables, `radon_cut` and `visamp` compute visibility
amplitudes from an image — see [Outputs](@ref "Outputs API").

## Known input issues

These concern the AthenaK binary-black-hole runs imaged by
`figures/fig_eht_bbh.jl` and the [RAPTOR comparison](@ref "Validation against RAPTOR"),
not either ray tracer.

- **Spin in units of the total mass.** The AthenaK `dynbbh` problem generator
  (`src/pgen/dynbbh.cpp` at AthenaK commit `6df4d7e8`, lines 1346–1352) sets each
  hole's Kerr–Schild spin to the header value `a1`, `a2` in units of the total
  mass while the hole mass is $1/(1+q)$. The spinning runs (snapshots 00093 and
  00122: $a_1=a_2=0.8$, $q=1$) therefore evolve a Kerr–Schild term with
  $a/m=1.6$, beyond the extremal limit, which no geodesic code can trace.
  `fig_eht_bbh.jl` images such a hole at dimensionless spin $\chi=a$
  ($a_{\rm used}=\chi\,m_A=0.4$) and records the mapping as `spin_mapping` in
  each run record.
- **Spin cross terms in `dynbbh.cpp`.** Hole 1's $\boldsymbol a\cdot\boldsymbol x$
  uses `o13 = x2BH1*a2x` (line 1485) where $a_{1y}$ belongs, its $a^2$ term uses
  $a_{2x}^2$ (line 1479), and hole 2's spin vector is built from `bbh.a1`
  (lines 1349–1351). None of these changes the runs imaged here, which have both
  spins along $z$ and $a_1=a_2$.
- **Snapshot 00093** carries a cold, dense ball at $r<8$ about the origin and the
  transient of its restart; 28%, 41% and 46% of its 230 GHz flux at
  $i=17^\circ,60^\circ,90^\circ$ lies outside the 56 $M$ field of view. Both codes
  image the data as is.
- **Orbital phase of snapshot 00455.** The hole positions follow from the
  analytic trajectory, $\phi_0={\rm rem2pi}(\Omega t)$ with
  $\Omega={\rm sep}^{-3/2}=0.008$ for separation 25. The dump header stores the
  time to six significant digits (±0.004 rad in $\phi_0$), so
  `fig_eht_bbh.jl` takes $t$ from `tlim` $=453146.89818$ when it lies within 0.5
  of the header time, giving $\phi_0=-0.2227368$ and centres
  $\pm(12.19120680,-2.76124551,0)$. The adaptive-refinement deck of
  `figures/fig_bbh_i90_pol_amr.jl` fixes $\phi_0=-0.2233022$, $5.7\times10^{-4}$
  rad (0.007 $M$ in position) from that value.

## Where to go next

- [GPU rendering](@ref) — the same images on one or more GPUs.
- [Numerical-relativity and binary spacetimes](@ref) — non-Kerr backgrounds.
- [Method](@ref "Method") — the transfer scheme and coefficient fits.
