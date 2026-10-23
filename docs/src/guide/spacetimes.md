# Spacetimes and binaries

Every spacetime supplies a covariant metric $g_{\mu\nu}(x)$ on Cartesian coordinates
$x^\mu = (t, x, y, z)$; the inverse, the gradients and the Christoffel symbols come from
forward-mode automatic differentiation, so a new metric needs no hand-coded
derivatives. Pass any spacetime to `prepare` as `spacetime`.

## Kerr

`KerrSchildCartesian(a)` is Kerr of unit mass and spin `a` in Cartesian Kerr–Schild
form, $g_{\mu\nu} = \eta_{\mu\nu} + f\, l_\mu l_\nu$, with the closed-form inverse that the
null covector allows. `Minkowski()` is flat space.

## An arbitrary analytic metric

`UserMetric` wraps any closure `x -> g_μν` that is generic in the number type of `x`:

```julia
using BlackLightPlus, StaticArrays

function schwarzschild(x)
    r = sqrt(x[2]^2 + x[3]^2 + x[4]^2)
    f = 2.0 / r
    l = SVector(1.0, x[2] / r, x[3] / r, x[4] / r)
    η = SMatrix{4,4}(-1.0, 0, 0, 0,  0, 1.0, 0, 0,  0, 0, 1.0, 0,  0, 0, 0, 1.0)
    return η + f * (l * l')
end

st = UserMetric(schwarzschild)
```

The closure may depend on `t`; the general geodesic right-hand side keeps $\partial_t g$.
No horizon is known for an arbitrary metric, so the march stops only on the escape sphere
or at the step limit; put a horizon-penetrating chart under the hole or a redshift cut
in the metric if rays must end there.

## Binary black holes at rest

`SuperposedKerrSchild` superposes $N$ Kerr–Schild holes with spins along $\pm z$,
$g_{\mu\nu} = \eta_{\mu\nu} + \sum_A f_A\, l^A_\mu l^A_\nu$, the standard wide-separation
binary ansatz; its inverse is the Woodbury identity with an $N \times N$ solve. Rays end at
either hole's horizon or at the redshift cut ([Method](@ref "March termination")).

```julia
snap = AthenaKSnapshot("torus.mhd_w_bcc.00455.bin"; gamma = 13 / 9)
st = SuperposedKerrSchild([0.5, 0.5],                                 # masses
                          [0.0, 0.0],                                 # spins
                          [(12.19, -2.77, 0.0), (-12.19, 2.77, 0.0)]) # centres
plan = prepare(cfg; spacetime = st, matter = snap, mode = StokesI())
img = render(plan)
```

A second hole of zero mass reproduces single Kerr bitwise, which is the test to run
first if an image looks wrong.

!!! note "This is an approximation"
    The superposition is not a solution of Einstein's equations; the cross terms grow as
    the holes approach. It is a reasonable model at wide separation and a poor one near
    merger, where a Z4c slice ([Numerical relativity](@ref)) is the alternative.

## Binary black holes in motion

`BoostedSuperposedKerrSchild` moves the holes along a prescribed trajectory and takes
each hole's term in the rest frame of its instantaneous velocity (Combi & Ressler 2024,
arXiv:2403.13308),

```math
g_{\mu\nu} = \eta_{\mu\nu} + \sum_A f_A\, (\Lambda_A l^A)_\mu (\Lambda_A l^A)_\nu ,
\qquad x'_A = \Lambda(v_A)\,(x - s_A(t)) ,
```

with the Kerr–Schild term of hole $A$ (any spin direction) evaluated at the
Lorentz-contracted offset $x'_A$ and its null covector boosted back. The term is exact
Kerr for a hole in uniform motion, and the boosted covector stays null, so the Woodbury
inverse of the static superposition serves unchanged. The metric depends on time through
$s_A(t)$ and $v_A(t)$: the spacetime is not stationary, the geodesic right-hand side keeps
$\partial_t g$, and $p_t$ drifts along a ray. These types are reached through the
`Spacetime` module:

```julia
using BlackLightPlus
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit,
                                trace_geodesic, nullify_covariant

orbit = CircularOrbit((0.5, 0.5), 8.0)                 # separation 8 M, ω = √(M/d³)
st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0, 0, 0), (0, 0, 0)], orbit)

x0 = SVector(0.0, 45.0, 0.0, 0.0)                       # the camera, at t = 0
k = nullify_covariant(st, x0, SVector(1.0, 0.0, 0.0))   # received along +x
sol = trace_geodesic(st, vcat(x0, k / k[1]); save_everystep = true)
```

The observer time is $t = 0$ at the camera; a backward-traced ray runs to earlier times
and meets the holes where they were then. Freezing the metric at one time is not an
option for a moving hole: the frozen boosted horizon is not a null surface and rays
stall on it. Polarized renders transport the camera frame with the time-dependent
connection (both metric gradients, the time column included, at every step).
`boost = false` carries the static terms along the trajectory without the boost, for
comparison. `figures/dynrays/` traces an orbiting binary's ray fan and the snapshot
trajectories of Bohn et al. 2015 in this spacetime.

### Products on a moving metric

Every product of a plan accepts the type: `render` (fast light, or slow light over a
`SeriesProvider` or `FrameWindow`), `render_amr`, `trace` and the two-phase render along
its trajectories, `shadow`, `variants`, `sensitivity` and `mc_spectrum`. The exception is
`MixedLanes`, whose rescue set is the neighbourhood of the Kerr critical curve; a
`Lanes{Float32,Float64,Float64}` plan marches the geodesics in `Float32` on any
spacetime.

The two clocks of slow light are one clock. The camera's $t = 0$ is the observation
epoch `t_obs` of the series, and a ray's emission time is $t_{\rm obs} + x^0$; the
holes are read at the same $x^0$. `CircularOrbit(…; t0)` sets the time at which the
orbit has its `phase`, so a binary phased to the fluid's clock takes `t0 = t_obs`:

```julia
series = SnapshotSeries(paths, times; t_obs = times[end])
orbit = CircularOrbit((0.5, 0.5), 8.0; phase = 0.0, t0 = times[end])
st = BoostedSuperposedKerrSchild([0.5, 0.5], [(0, 0, 0), (0, 0, 0)], orbit)
plan = prepare(cfg; spacetime = st, matter = SeriesProvider(series), mode = IQUV())
```

The Monte Carlo march reads the spacetime through `proxy_distance(st, x)`, the
distance above the nearest hole's horizon at the ray's time and the coordinate radius
its escape rules compare against, so a superphoton ends on whichever hole it meets.
Fluid cells deep inside a hole, where the superposed metric has no normal observer
($g^{tt} \ge 0$), decode to vacuum and emit nothing; the cells outside the horizons are
read as on Kerr.

On a GPU the type is a plain bits value (`isbits`), so `backend = :gpu` runs the same
kernels: the metric and its gradients at every step from the trajectory, the per-hole
capture, and the parallel transport. Slow light on a device goes through a `FrameWindow`
([Imaging GRMHD data](@ref)); `SeriesProvider` is a host provider.

## Numerical relativity

`z4c_grid_spacetime` reads an AthenaK Z4c `.bin` dump [Zhu2025](@citep), maps the
conformal Z4c variables to the ADM fields $(\alpha, \beta^i, \gamma_{ij})$ and resamples
them onto a uniform node box as an `ADMGrid`:

```julia
st, dump = z4c_grid_spacetime("bbh.z4c.00001.bin", 12.0, 192; alpha_min = 0.55)
```

The grid reassembles the 4-metric through the 3+1 decomposition and interpolates the ten
fields with a $C^1$ Catmull–Rom stencil, because the geodesic equation differentiates
through the interpolant and a trilinear interpolant would give discontinuous
Christoffel symbols at the cell faces. Rays end where the interpolated lapse falls below
`alpha_min` (the numerical horizon) or on the escape sphere inside the grid; off the grid
the metric is Minkowski.

!!! warning "The slice is frozen"
    An `ADMGrid` sets $\partial_t g = 0$. It is valid while the light-crossing time of the
    imaged volume is short against the evolution time of the spacetime; the builder
    warns once per session.

`adm_grid_from_metric` samples any analytic metric onto an ADM grid, the cleanest test
of the grid machinery against a known answer:

```julia
using BlackLightPlus.Spacetime: metric, ks_radius, horizon_radius

kerr = KerrSchildCartesian(0.9)
xs = -20.0:0.5:20.0
grid = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min = 0.1,
                            excise = x -> ks_radius(kerr, x) < horizon_radius(kerr))
```

Images from `grid` converge to those from `kerr` as the node spacing shrinks.

## Checking a metric

The metric and its inverse at a point, and the null constraint of a camera ray:

```julia
using BlackLightPlus.Spacetime: metric, inverse_metric, null_error, camera_ray, CameraFrame,
                                PlaneParallelCamera
using LinearAlgebra: I

x = SVector(0.0, 5.0, 3.0, 1.0)
maximum(abs, metric(st, x) * inverse_metric(st, x) - I)        # ≈ 1e-16
y, ok = camera_ray(st, CameraFrame(PlaneParallelCamera(inc_deg = 60.0, d = 1000.0)), 2.0, 1.0)
null_error(st, y)                                              # ≈ 0
```

## Where to go next

- [Imaging GRMHD data](@ref) to add a fluid to these backgrounds.
- [Method](@ref) for the 3+1 decomposition, the Woodbury inverse and the termination rule.
