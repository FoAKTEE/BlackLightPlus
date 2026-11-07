# Closed-form Kerr images

For thin sources in Kerr the geodesics are elliptic functions of the screen point, so an
image needs no integration: every pixel inverts its ray in closed form to the points where
it crosses the emitting surfaces (the [Method](@ref "Analytic Kerr fast path and lensing
bands") states the formulas). The products take a plan without matter, whose camera fixes
the pixels and the inclination, and a material.

## The equatorial disk

```julia
using BlackLightPlus
using BlackLightPlus.Plans: CameraConfig, RunConfig, prepare

cfg = RunConfig(camera = CameraConfig(inclination_deg = 17.0, fov = 20.0, resolution = 512))
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.94), matter = nothing)
disk = render_analytic(plan, EquatorialDisk(0.94); bands = (0, 1, 2), backend = :cpu)
disk.Ib[2]          # the n = 1 photon ring alone
```

## Polarized surfaces and cones

`SynchrotronSurface` is a thin surface of constant polar angle emitting power-law
synchrotron radiation, polarized along $\hat k\times b$ in the emitter's rest frame;
`PolarizedDisk` adds AART's polarization to the g-disk. A tuple of materials images
several surfaces at once:

```julia
using BlackLightPlus.Matter: SynchrotronSurface, PolarizedDisk

ι = 0.58
b(η) = (sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))     # rest-frame field (e_r, e_φ, e_z)
upper = SynchrotronSurface(0.94; theta_s = deg2rad(75), b = b(2.64), beta = 0.87,
                           theta_v = π / 2, phi_v = -1.7, sigma = 0.73)
lower = SynchrotronSurface(0.94; theta_s = deg2rad(105), b = b(π - 2.64), beta = 0.87,
                           theta_v = π / 2, phi_v = -1.7, sigma = 0.73)
img = render_analytic(plan, (upper, lower); bands = (0, 1, 2), ro = Inf, backend = :cpu)
img.I, img.Q, img.U              # Q > 0 north (+β), U > 0 north-east (east = −α)
img.Qb[1]                        # Stokes Q of order 0

aart = render_analytic(plan, PolarizedDisk(EquatorialDisk(0.94); b = (1.0, 0.0, 0.0)))
```

A cone closer to a pole than the line of sight (a jet) shows its near and far sides in
the same order; `image_crossings(prep, θs, n)` lists the crossings of order `n`, and
`surface_landing(prep, θs, m)` the landing after `m` polar turning points, with its
radius, azimuth, emission time and the signs of $p^r$ and $p^\theta$ at emission.

## Photon rings of high order

Every order `n` is a band of `bands`; orders above 2 are rings narrower than
$e^{-\gamma n}$ of the screen and are resolved on cuts rather than on a pixel grid:

```julia
using BlackLightPlus.Spacetime: critical_point, photon_shell_exponents, pixel_prep, band_landing

a, θo = 0.94, deg2rad(17.0)
α0, β0 = critical_point(a, θo, 2.5)            # the critical point of shell radius 2.5
γ, δ, τ = photon_shell_exponents(a, 2.5)       # demagnification, rotation, delay per order
l = band_landing(pixel_prep(1.00001α0, 1.00001β0, θo, a; ro = Inf), 4)   # order 4 just outside
```

`lensing_band(a, θo, n)` returns the inner and outer edges of band `n` for any order.

## The observer at infinity

`ro = Inf` places the observer at infinity: radii and azimuths are the limits of a far
observer, and the landing time is the emission time relative to the observer's retarded
time, $\tilde t=\lim(t+r_o+2\ln r_o)$. A finite `ro` (the camera distance by default)
reproduces AART's finite observer.
