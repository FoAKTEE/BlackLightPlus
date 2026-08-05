# Numerical-relativity and binary spacetimes

BlackLightPlus images any spacetime that can supply a covariant metric
$g_{\mu\nu}(x)$. All derivative machinery — inverse-metric gradients,
Christoffel symbols — comes from automatic differentiation, so a new metric
never needs hand-coded derivatives. This guide covers the three non-Kerr
backgrounds the package ships with.

Pass any of them through the `spacetime` keyword of `render`, or into the
camera and tracer directly.

## An arbitrary analytic metric

`UserMetric` wraps any closure $x \mapsto g_{\mu\nu}$:

```julia
using BlackLightPlus, StaticArrays

function my_metric(x)
    # x = (t, x, y, z) in Cartesian Kerr-Schild; return the 4x4 covariant metric
    r = sqrt(x[2]^2 + x[3]^2 + x[4]^2)
    f = 2.0 / r
    l = SVector(1.0, x[2]/r, x[3]/r, x[4]/r)
    η = SMatrix{4,4}(-1.0, 0, 0, 0,  0, 1.0, 0, 0,  0, 0, 1.0, 0,  0, 0, 0, 1.0)
    return η + f * (l * l')
end

st = UserMetric(my_metric)
```

The inverse metric and its gradients are obtained by `StaticArrays` inversion
and ForwardDiff. `Minkowski()` is flat space, useful as a null test.

## Binary black holes: superposed Kerr–Schild

`SuperposedKerrSchild` superposes $N$ Kerr–Schild holes,

```math
g_{\mu\nu} = \eta_{\mu\nu} + \sum_A f_A\, l^A_\mu l^A_\nu ,
```

giving an analytic approximate binary spacetime. Rays terminate on entering any
hole's horizon.

```julia
snap = athenak_bin_snapshot("torus.mhd_w_bcc.00455.bin"; gamma = 13/9)

st = SuperposedKerrSchild(
    [0.5, 0.5],                                 # masses
    [0.0, 0.0],                                 # spins
    [(12.19, -2.77, 0.0), (-12.19, 2.77, 0.0)], # centers
)

img = render(cfg, snap; spacetime = st)
```

The construction reduces bitwise to single Kerr when a second mass vanishes,
which is the test to run first if results look wrong.

!!! note "This is an approximation"
    Superposed Kerr–Schild is not a solution of Einstein's equations. The
    cross terms break the single-hole inverse identity, and the constraint
    violation grows as the holes approach. It is a reasonable model at wide
    separation and a poor one near merger — for that, use a Z4c slice below.

## Numerical relativity: Z4c slices from AthenaK

`z4c_grid_spacetime` reads an AthenaK Z4c `.bin` dump [Zhu2025](@citep), maps
the conformal Z4c variables to physical ADM fields, and builds an
`ADMGridSpacetime`:

```julia
st = z4c_grid_spacetime("bbh.z4c.00001.bin", 12.0, 192)  # half-width 12 M, 192³
```

The grid stores the ten ADM fields $(\alpha, \beta^i, \gamma_{ij})$ and
reassembles the 4-metric through the standard 3+1 decomposition. Fields are
interpolated with a $C^1$ cubic Catmull–Rom stencil — $C^1$ continuity is
required because AD differentiates *through* the interpolant, and a trilinear
($C^0$) interpolant would give discontinuous Christoffel symbols at cell faces.

Rays terminate where the lapse falls below a threshold (the numerical horizon)
or where they leave the grid.

!!! warning "The slice is frozen"
    `ADMGridSpacetime` sets $\partial_t g = 0$. This is valid only while the
    light-crossing time of the imaged volume is short compared to the
    spacetime's evolution timescale. The builder emits a warning when the
    requested volume is large enough for this to matter.

### Checking a reconstruction

Before trusting an unfamiliar dump, verify the metric round-trips:

```julia
x = SVector(0.0, 5.0, 3.0, 1.0)
g = metric(st, x)
ginv = inverse_metric(st, x)
maximum(abs, g * ginv - I)     # should be ~1e-16
```

A real BBH merger dump reconstructs to $4\times10^{-16}$ and lenses into two
puncture shadows.

## Sampling an analytic metric onto a grid

`adm_grid_from_metric` samples any analytic metric onto an ADM grid, which is
the cleanest way to test the grid machinery against a known answer:

```julia
exact = KerrSchildCartesian(0.9)
grid  = adm_grid_from_metric(exact, 20.0, 128)
# images from `grid` should converge to images from `exact` as resolution rises
```

## Where to go next

- [Imaging GRMHD data](@ref) — adding a fluid to these backgrounds.
- [Method](@ref "Method") — the 3+1 decomposition and interpolation requirements.
