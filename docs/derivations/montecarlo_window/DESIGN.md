# Monte Carlo in a dynamical spacetime with an evolving plasma: design and exact tests

Scope. The present `mc_spectrum` tracks superphotons through any spacetime, the metric (and its
time derivative) evaluated at the photon's own coordinate time, but the plasma is one snapshot held
fixed and every superphoton is emitted at x^t = 0; escaped photons are binned in frequency and
inclination only. This design adds an evolving plasma (a snapshot series), emission spread over a
time window, and binning in arrival time at a distant observer, so the module returns light curves
and time-resolved spectra. Everything below is to be derived again and checked by the implementer.

## 1. Emission over a time window

The plasma is a series of snapshots S_k at coordinate times t_k; between t_k and t_{k+1} the
primitive variables are interpolated linearly in time after interpolation in space, exactly as the
slow-light imager does (method.tex, "Slow light"), so the emissivity j_nu(x, t) is defined for
t in [t_1, t_K].

Emission window [t_a, t_b] (inside [t_1, t_K]). The number of photons emitted per unit frequency in
the window is

  dN/dnu = int_{t_a}^{t_b} dt int d^3x sqrt(-g) (u^t) ... j_nu(x,t) / (h nu)   (same zone measure as
           the stationary module; check the sqrt(-g) u^t factor against the stationary estimator)

Sampling: draw the emission time t_e from the distribution proportional to the emitted power in
time, P(t) proportional to sum over zones of the zone emission at t (piecewise linear between
snapshots, so P(t) is piecewise linear and can be sampled exactly), then the zone and frequency from
the stationary tables evaluated at t_e (linear blend of the two bracketing snapshots' zone tables:
the F(K) table makes per-snapshot zone tables cheap). The superphoton weight is the emitted photon
number of the window divided by the number drawn, as in the stationary module. Random numbers for
t_e come from a separate stream index of the primary's Philox stream, so that the frozen limit
(all S_k identical) reproduces the stationary photons bit for bit apart from the time coordinate.

## 2. Transport

The photon's state already carries x^t. Absorption and scattering read the plasma at the photon's
current x^t (two-snapshot blend), and the metric at the same x^t (already so). Children inherit the
parent's x^t at the event. A photon that reaches x^t > t_K (or < t_1) samples the nearest snapshot,
as the imager does; record the fraction of path length spent outside [t_1, t_K] as a diagnostic.

## 3. Arrival time and the estimator

A photon escapes at coordinate time t_esc and position x_esc on the escape sphere with direction n
(from the spatial wave vector, normalized). A distant observer in direction n receives it at

  t_arr = t_esc - n . x_esc   (+ a constant common to all photons: D/c and the far-field Shapiro
                                delay 2M ln(D/R_esc), which do not depend on the photon)

so t_arr is the arrival time relative to light from the origin emitted at t = 0. Binning in
(t_arr, nu, inclination band [, azimuth]) with bin width Delta t gives the luminosity per unit
observer time

  nu L_nu(t, theta) = sum_{photons in bin} w h nu * nu/Delta nu ... / (Delta t * Delta Omega_band / 4 pi)

(the stationary nu L_nu estimator divided by Delta t instead of by the run's notional duration).
Exactness check: summing nu L_nu(t) Delta t over all t_arr gives the stationary-estimator energy;
for a stationary source the mean over interior bins equals the stationary nu L_nu.

Edges: arrival times within the largest light-crossing time of the window edges see a partial
source; exclude them (or mark them) in light curves. Report per-bin standard errors from sum (wE)^2.

## 4. Exact tests (each with a figure)

T1. Frozen limit. A series of identical snapshots: every photon identical to the stationary run
    except its time coordinate; the time-integrated spectrum equals the stationary spectrum times
    (t_b - t_a) bit for bit; interior light-curve bins constant within their errors.

T2. Variable optically thin sphere (flat space). Uniform sphere of radius R, emissivity
    j(t) = j_0 [1 + A sin(omega t)] (a snapshot series sampled densely enough that the linear
    blend error is negligible, or an analytic time-dependent emitter). For an observer along z the
    delay relative to the centre is tau = -z with weight proportional to the slab area
    pi (R^2 - z^2), so the delay distribution is P(tau) = 3 (R^2 - tau^2) / (4 R^3), |tau| <= R.
    The observed light curve is L(t) = L_0 [1 + A |Phat| sin(omega t)] with no phase shift and

      Phat(omega) = 3 j_1(omega R) / (omega R) = 3 (sin x - x cos x) / x^3,  x = omega R.

    Check the damping |Phat| over a range of omega R (including the zeros of j_1).

T3. Light echo from a scattering shell (flat space, Thomson limit: cold electrons, low photon
    energy). A flash at the centre at t = 0, a thin optically thin shell of radius r. A photon
    scattered at angle psi to the observer direction arrives with delay tau = r (1 - cos psi). With
    the Thomson phase function (1 + cos^2 psi) the first-order echo transfer function is

      Psi(tau) = 3 / (8 r) [1 + (1 - tau/r)^2],  0 <= tau <= 2 r,

    normalized to the scattered fraction (tau_shell for a thin shell). Hot electrons broaden it
    only through the angular kernel; keep Theta_e << 1 for the exact form.

T4. Helical symmetry (orbiting binary). On the orbiting boosted superposed metric with a plasma
    that rotates rigidly with the orbit (the snapshot rotated by Omega (t - t_s), as in the
    slow-light imaging subsection), the system is invariant under t -> t + tau, phi -> phi + Omega tau.
    Binned also in observer azimuth, the result must satisfy
      L(t + tau, theta, phi + Omega tau) = L(t, theta, phi)
    to within errors, and the light curve at fixed (theta, phi) is periodic with the orbital period
    (half of it for an equal-mass binary). This tests time dependence of metric and plasma together.

T5. Kerr with a stationary torus series: equals T1 on a curved spacetime (time-integrated spectrum
    vs the stationary module, bitwise).

T6. Convergence: light-curve bin errors fall as N^-1/2; per-bin standard errors agree with the
    scatter of independent seeds.

## 5. Interface (proposal)

`mc_spectrum(plan, Ns; window = (t_a, t_b), time_bins = n_t or edges, azimuth_bins = ...)` where the
plan's matter may be a `SnapshotSeries`; the result gains a time axis (edges, per-bin values and
errors), keeps the time-integrated spectrum, and records the out-of-series path fraction. With a
single snapshot and no window the behaviour and the numbers are those of the stationary module.

## 6. Notes for the implementer

- Derive the zone measure and the emitted-number normalization again from the stationary code
  after the fixes (F(K) table, stride-1 zones, half-weight end nodes) and state it in the docs.
- Keep scheduling independence (counter-based streams keyed by ancestry, exact integer bins) with
  the extra time axis; the exact accumulator memory grows with the number of time bins.
- GPU: the two-snapshot window of the imager's device path is the model for a device series.
