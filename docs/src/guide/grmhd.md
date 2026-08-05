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

## Where to go next

- [GPU rendering](@ref) — the same images, roughly 100× faster.
- [Numerical-relativity and binary spacetimes](@ref) — non-Kerr backgrounds.
- [Method](@ref "Method") — the transfer scheme and coefficient fits.
