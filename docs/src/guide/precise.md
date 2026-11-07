# Precise ray tracing

The imaging products march every ray with a fixed-control step: the step is the
distance to the horizon over a divisor `C`, and the classical fourth-order Runge–Kutta
step (`:rk4`) advances the state. The precise mode replaces the integrator, the camera
and the stopping rule so that the marched geodesics reach the accuracy of the
closed-form Kerr tracer, on any spacetime the march supports.

```julia
using BlackLightPlus
using BlackLightPlus.Plans: precise
using BlackLightPlus.Products: render_crossings

a = 0.94
cfg = precise(RunConfig(camera = CameraConfig(inclination_deg = 17.0, fov = 50.0,
                                              resolution = 512)))
plan = prepare(cfg; spacetime = KerrSchildCartesian(a), matter = nothing)
disk = EquatorialDisk(a)

marched = render_crossings(plan, disk)            # the numerical march, orders 0, 1, 2
exact = render_analytic(plan, disk; ro = 1e15)    # the closed form, observer at infinity
```

The two images agree to rounding error: `marched.Ib[n]` and `exact.Ib[n]` differ by
about 10⁻¹⁶ of the image maximum in the median pixel of order `n = 0`, 10⁻¹⁴ for
`n = 1` and 10⁻¹³ for `n = 2`, and by 10⁻¹² at the 99th percentile of `n = 2`, where the
photon ring amplifies rounding most.

## What the precise mode sets

`precise(cfg)` returns `cfg` with

| field | production | precise |
|---|---|---|
| `integrator` | `:rk4` | `:verner9` |
| `C` | 80 | `PRECISE_C` = 20 |
| `winding` | `false` | `true` |
| `capture_redshift` | 100 | `PRECISE_CAPTURE` = 10⁶ |
| `horizon_eps` | 10⁻³ | 0 |
| `camera.infinity` | `false` | `true` (keyword `infinity`) |
| `max_steps` | 2500 | at least 10⁵ |

- **`:verner9`** is the ninth-order step of Verner's 9(8) pair: fifteen right-hand sides
  per step against four, with an error that falls as `C⁻⁹` instead of `C⁻⁴`. At equal
  cost it is many orders of magnitude more accurate; at `C = 20` it reaches rounding
  error on Kerr.
- **`winding`** shortens the step near a spinning horizon, where a backward-traced ray
  winds around the spin axis faster than its radius changes.
- **`capture_redshift`** is the photon energy in the local normal frame at which a ray
  falling into a black hole is taken as captured; anything it would still meet is
  redshifted by at least that factor. At 10⁶ the omitted emission of a thin disk is
  below 10⁻¹⁵ of the image maximum.
- **`camera.infinity`** makes the image coordinates the Bardeen coordinates of an
  observer at infinity, as in the closed-form tracer: each ray is carried in from
  infinity through the far zone of the metric and started on the plane at
  `camera.distance`. It needs a stationary spacetime (Kerr, a static binary, an ADM
  slice); a time-dependent one keeps the camera at its finite distance,
  `precise(cfg; infinity = false)`. On a binary the redshift cut stays at 100 whatever
  the configuration: the horizons of a superposition are approximate, and a ray
  followed beyond that energy stalls above them and is lost.

## When to use it

The precise mode serves the products whose accuracy is that of the geodesics:
images of emission surfaces (`render_crossings`), trajectories (`trace`), shadows and
capture maps (`shadow`), and the charges of rays on dynamical spacetimes. A
volumetric render (`render` with a plasma) samples its plasma once per step, so its
accuracy is set by the step divisor and not by the integrator (the flux of a torus
image converges as `C⁻²` with either integrator): keep `C` at the production value or
above,
`precise(cfg; C = 80.0)`.

The precise mode takes fewer, more expensive steps than the production march and
spends none between the camera and the black hole, which the far zone covers in about
twenty steps. On the CPU it costs 1.3 to 2 times the production march per ray at
moderate spin and up to six times at spin 0.99, where every captured ray carries the
winding-resolved step down to the redshift cut; `precise(cfg; C = 14.0)` reaches the
same accuracy at about 70 % of the cost on both configurations of the code comparison.

## Integrators and options one at a time

Every field can be set on its own. The production integrator converges at fourth
order once the camera is at infinity:

```julia
cfg = RunConfig(camera = CameraConfig(inclination_deg = 17.0, infinity = true),
                C = 320.0, capture_redshift = 1e6, max_steps = 10^5)
```

With a redshift cut far above the production value near a fast-spinning black hole,
set `winding = true` as well: an unresolved step near the horizon meets the equator at
spurious points, whose intensities come out negative or not finite.

The ninth-order step can march a time-dependent spacetime with the camera at its
finite distance:

```julia
cfg = precise(RunConfig(camera = CameraConfig(distance = 1000.0)); infinity = false)
plan = prepare(cfg; spacetime = BoostedSuperposedKerrSchild(masses, spins, orbit),
               matter = nothing)
tr = trace(plan)
```

The step functions are reachable directly through `BlackLightPlus.Spacetime`:
`march_step(Verner9(), st, y, h)`, `surface_crossing(Verner9(), st, y, h,
EquatorialPlane())`, and `infinity_ray(st, frame, α, β)`.

## A user metric at infinity

The camera at infinity needs the offset `(g^{μν} − η^{μν}) p_ν` of the raised covector
without the cancellation of subtracting `η` from `g^{-1}`. Kerr–Schild spacetimes
provide it. A `UserMetric` is not stationary by declaration and keeps the finite
camera; a stationary user spacetime type gains the camera at infinity by defining
`Spacetime.raise_index_offset(st, x, p)` in closed form.
