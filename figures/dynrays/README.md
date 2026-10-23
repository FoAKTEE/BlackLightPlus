# Rays through an orbiting binary in the moving superposed Kerr-Schild metric

> The figure images and run records of this set are kept on the `blp` branch under `results/blp/dynrays/`; this branch holds the generators and this standing table.

Reproductions, with BlackLightPlus's time-dependent geodesic tracer, of the two reference figures in
`figures/ref/`: `image (57).png` (figure 8 of arXiv:1410.7775, Bohn et al. 2015: geodesic
snapshots against the horizons of a merging binary, traced through a SpEC spacetime) and
`rays.gif` (a fan of rays in the orbital plane of an orbiting binary, holes moving while
the light crosses). Both are planar views of the orbital plane. Generator
`figures/paper/fig_dyn_rays.jl` (commit recorded in `setup.json`), plotter
`figures/paper/plot_dyn_rays.py`; data under `figures/dynrays/out/dyn_rays/` (not tracked).

## Spacetime

`BoostedSuperposedKerrSchild` on a `CircularOrbit`: two Kerr-Schild holes of mass 1/2,
no spin, separation 8 M on the Keplerian circular orbit (ω = √(1/8³) = 0.0442, period
142 M, hole speed 0.177), each hole's term in the rest frame of its instantaneous
velocity (the SKS metric of arXiv:2403.13308, t̄ = 0 form). The metric is evaluated at
the ray's own coordinate time and ∂_t g enters the geodesic equation. A frozen snapshot
cannot do this: the frozen boosted horizon is not null and rays stall on it
(`results/blp/binaries/fork_review.md`, D1). The unboosted variant (`boost = false`,
static terms carried along the orbit) is traced for comparison.

Rays: adaptive Vern7, reltol 1e-10, abstol 1e-12, dense output sampled every 0.05 in λ;
captured at r'_A < r_h,A + 1e-3 or at normal-frame energy 100 (the redshift cut, which in
the binary ends the approach at r'_A − r_h,A ≈ 0.2), escaped at R > 200 M outbound.

## Figures

- `fan_boosted.gif`, `fan_boosted_last.png` — the ray-fan animation: 61 rays received at
  (45, 0, 0) from directions within ±20° of −x, drawn from the camera to their position at
  coordinate time t = 0, −1, …, −99 M (one frame per M, 100 ms per frame), coloured by
  launch angle, solid where the ray escapes and dashed where it is captured; the horizons
  r'_A = r_h,A (Lorentz-contracted circles) at that time. Fates: 41 escape, 11 end on hole
  1, 9 on hole 2.
- `bohn_boosted.png` — the figure-8 analogue: an edge-on plane-parallel camera at x = 60 M,
  16 rays on the image row β = 0 at α = 4.75–7.65 M and three inside the eyebrow run at
  α = 7.70–7.80 M (chosen from a 321-ray scan of |α| ≤ 8 M: hole-1 shadow 0–3.55 M, escape
  3.6–4.9, hole-2 captures 4.95–7.55, escape, hole-1 captures 7.7–7.8, escape); the orbit's
  phase aligns the holes along the line of sight at t = −69 M, when the rays pass them.
  Panels A–C at the median times the bundle crosses x = 8.2, 4.5 and 1.5 M (t = −61, −66,
  −70 M), D thirty M later; dots mark the current positions, solid lines rays from
  infinity, dotted lines rays from a hole. Fates: 2 escape, 11 hole 2, 3 hole 1. A
  captured ray that has reached the redshift cut is drawn riding on with its hole
  (translated by the hole's displacement since the stop), the asymptotic approach to the
  moving horizon.
- `bohn_unboosted.png` — the same rays in the unboosted moving superposition: 9 escape,
  7 hole 2, 0 hole 1; the eyebrow rays are not captured.
- `fan_compare.png` — the last fan frame, boosted beside unboosted (41/11/9 vs 43/8/10).

## Mass ratio 100:1

The same spacetime with masses 100/101 and 1/101 (separation 8 M, the same period 142 M;
the secondary at radius 7.92 M moves at 0.35 c, the primary's horizon radius is 1.98 M and
the secondary's 0.020 M). Generator runs: `BLP_DYN_Q=100 julia -t 16 --project=.
figures/paper/fig_dyn_rays.jl` (the fan; the Bohn row is skipped: no far-hole captures on
the scan row) and `julia -t 48 --project=. figures/paper/fig_dyn_class.jl`, drawn by
`plot_dyn_rays.py figures/dynrays/out/dyn_rays_q100` and `plot_dyn_class.py`.

- `fan_q100_boosted.gif`, `fan_q100_boosted_last.png`, `fan_q100_compare.png`,
  `setup_q100.json` — the ray fan of the 100:1 binary, as above; the secondary's horizon
  is drawn with a grey locator circle of radius 0.5 M. Fates: 42 escape, 18 end on the
  primary, 1 on the secondary (unboosted: 43/18/0).
- `class_q100.gif`, `class_q100_first.png`, `setup_class_q100.json` — the left column of
  the paper's binary figure (`figures/paper/paper/fig_binary.pdf` (a) and (e)) as an
  animation over one orbital period: 36 observer times t = 0, 3.95, …, 138 M, a
  plane-parallel camera at 60° from the orbital axis and 100 M, image box |α|, |β| ≤ 12 M.
  (a) the ray classes (escaped, captured by the primary, captured by the secondary,
  escaped after bending > 360°), the primary's image position at its passage time
  marked, an inset of ±0.6 M about the secondary's shadow; (e) the finest refinement
  level per tile of the image-plane AMR (base 128², blocks of 8², three levels, finest
  pixel 0.023 M, effective 1024²), refined where the classes of a block or its one-pixel
  border differ. Per frame 67–75 thousand rays in 7–17 s on 48 threads (303 s in all);
  classes per frame: 0.89 M escaped, 151–156 thousand primary, 0–63 secondary (none
  when the secondary passes behind the primary, frames 16–21), 333–456 bent > 360°.
  Along the orbit the secondary's shadow (3–4 pixels across) crosses the primary's
  shadow edge, is eclipsed behind it, and re-emerges; the refinement box follows it.

## Standing against the references

Inspected against `figures/ref/` by eye (frames 0, 12, …, 99 of each animation side by
side; the four panels against figure 8):

- the fan: the same geometry (pinhole fan from the right, holes at 0.018 of the frame in
  horizon radius and 0.13 in separation, the holes rotating through ≈110° during the 100
  frames, the purple-to-yellow order of the rays), the same evolution (an empty frame
  while the rays travel in, the convex wavefront, the shadow cusp as the bundle meets
  the first hole, dashed spirals into both holes, strongly deflected rays crossing the
  fan at large angles in the late frames); [SOLID]
- the snapshots: the bundle starts close together (A, B), sweeps round the near hole (C)
  and disperses (D) into rays that end on the far hole, on the near hole (the eyebrow
  rays) and at infinity, each dotted ray ending on a horizon drawn at the panel's time,
  as in figure 8; the reference binary is closer to merger (its horizons are ≈4 M
  apart) and its camera is at an angle to the orbital plane's axes, so the panels agree
  in kind, not in the exact ray pattern. [SOLID]

The reference gif's binary parameters and camera are not recorded; the separation 8 M and
the fan of ±20° at 45 M were chosen to match its proportions. The superposition is an
approximation at this separation (the holes move at 0.18 c); it is the metric the
AthenaK dynbbh simulations are evolved on (`results/blp/binaries/blp_review.md`, §2).
