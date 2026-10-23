# Validation

The package is held to three comparisons: the bitwise oracle against the reference
implementation it replaced, the cross-code comparisons against RAPTOR and Blacklight, and
the reproduction of published figures. The measurements, with their commands, are in the
[Verification record](@ref); the evidence files they cite (`results/blp/`) and the
reference implementation live on the `blp` branch of the repository.

## The oracle

The previous implementation is the reference: on identical decks the package reproduces
its Stokes $I$ images on every path, its samplers, plasma decode, shadows and exact Monte
Carlo sums bitwise on one hardware, and its $IQUV$ images, traces and torus Monte Carlo
to round-off (bitwise under `julia -C generic`). On the A100 the reference device drivers
and the package agree bitwise, $IQUV$ included. Across hardware a GPU pool against CPU
threads differs by the two targets' floating-point contraction: $10^{-14}$–$10^{-9}$ of the
image maximum on the oracle decks, $10^{-12}$–$3\times10^{-10}$ on the production decks,
$10^{-16}$ relative on thin Monte Carlo sums. The test suite (14,453 tests, the structure
gates included) passes on the CPU and on the A100.

## Against RAPTOR

The reference implementation was compared with RAPTOR [Bronzwaer2018, Bronzwaer2020](@citep),
an independent C code with its own geodesic integrator, polarized transfer step and
camera, on analytic scenes and on binary-black-hole GRMHD snapshots; the package inherits
that comparison through the oracle. RAPTOR ran from a fork that adds, each behind a
compile-time switch that leaves the upstream build unchanged, the superposed Kerr–Schild
metric, the step control $h = \min_A (r_A - r_{h,A})/C$ and termination, the plane-parallel
camera, the AthenaK reader and plasma decode, the thermal coefficients of
[Marszewski2021](@citet), the IEEE/IAU readout and, for one leg, the exact
constant-coefficient polarized step. Metrics are those of the EHT code comparisons
[Gold2020](@citep).

| Comparison | Result |
|---|---|
| thermal coefficients, 40,959 cells in $(\nu, B, \Theta_e, \theta_B)$ | ported coefficients within $2.3\times10^{-13}$ relative |
| metric and geodesics (Kerr and the binary snapshots) | $\lvert g - g_{\rm BLP} \rvert \le 4.4\times10^{-16}$; identical end classes and sample sets, both RK4 of order 4 |
| camera | states equal to $2.4\times10^{-15}$ at $480^2$ |
| analytic scenes, $480^2$, $C = 160$ | every Stokes sign agrees on the orientation slabs; the test problems T1–T5 of [Gold2020](@citet) at three inclinations: ${\rm NMSE}(I) \le 2.5\times10^{-13}$ |
| AthenaK reader and plasma decode, $10^5$ points per snapshot | primitives, four-vectors, $n_e$, $B$, $\Theta_e$, $\sigma$, $\beta^{-1}$ and cut flags bit-identical |
| binary snapshot images, exact polarized step | $L_1(I) \le 1.9\times10^{-7}$, $L_1(\lvert P \rvert) \le 5.2\times10^{-6}$, $L_1(V) \le 8.7\times10^{-6}$ |
| binary snapshot images, RAPTOR's own step | flux within $5.2\times10^{-4}$, $L_1(I) \le 6.7\times10^{-3}$; the $\lvert P \rvert$ and $V$ differences fall with $C$ at RAPTOR's self-convergence rate |

With the exact step the two codes march the same states and differ only by the $O(h^4)$
truncation difference between RAPTOR's Christoffel form $(x^\mu, k^\mu)$ and the
Hamiltonian form $(x^\mu, p_\mu)$ of the same RK4 march. With RAPTOR's native step the
polarized differences are the discretisation error of that step, whose implicit trapezoid
on absorbing segments converges at second order and loses phase with the Faraday depth.
The thresholds of the binary images ($|\Delta F/F| \le 0.01$, $L_1(I) \le 0.03$,
${\rm NMSE}(I) \le 10^{-3}$, $|\Delta m| \le 0.01$, $|\Delta v_{\rm net}| \le 0.002$) are met on
every snapshot; the net EVPA of the edge-on views crosses Faraday depths of $10^2$–$10^4$
rad and is not converged in $C$ in either code.

## Against Blacklight

The figure reproductions below image the AthenaK single-hole torus with the package and
with Blacklight [White2022](@citep) (its C++ code, with the dumps converted to its reader's
layout) on matched cameras, plasma prescriptions and step controls. Stokes $I$ agrees to
${\rm NMSE}$ $7.9\times10^{-4}$ (the Mahakala setup, $256^2$, $R_{\rm high} = 10$) and
$5.7\times10^{-3}$ (the KPolaris setup, $i = 60^\circ$, 230 GHz), with the flux within 0.35%;
the residual is unchanged under step refinement and under sub-pixel shifts or rescaling
of either image, so it is a sampling and plasma-decode difference between the codes, not
integration or registration. $Q$ and $U$ agree where Faraday effects are weak (NMSE 0.04 at
1 THz) and diverge with the Faraday depth. Stokes $V$ differs by a factor 2–7 at every
frequency; the package's $V$ agrees with RAPTOR, so the circular-polarization coefficient
or convention of one of the two codes is an open item.

## Published figures

The generators under `figures/` reproduce published figures with the package; each
directory holds a README with a standing table (one row per figure: what the paper
shows, what was traced and with which data, the measured numbers, and a tag). The images
and run records are on the `blp` branch under `results/blp/`. The data substitutions are
stated in each table: none of the papers' GRMHD simulations is available on the host, so
every GRMHD figure uses the AthenaK single-hole torus series (spin 0.9, 819 dumps 30 M
apart).

| Set | Paper | Figures | Standing |
|---|---|---|---|
| `figures/paper/` | the method paper of this package | 8 | data figures equal to the reference package's arrays (Stokes I bitwise, $IQUV$ at round-off), every record number equal in its digits ([Verification record](@ref) §3) |
| `figures/dynrays/` | Bohn et al. 2015, figure 8, and a ray-fan animation | 2 (+ the 100:1 binary's fan and class-map animations) | in kind in the moving superposed metric: the fan geometry, the shadow cusp, dashed capture spirals, the bundle sweeping round the near hole and ending on both horizons |
| `figures/mahakala/` | Mahakala (arXiv:2304.03804) | 12 | 6 exact (geodesic tests: deflection angles to $7.5\times10^{-7}$ of the quadrature, fourth-order convergence, the spherical orbits, both Kerr–Schild charts), 4 in kind on the torus, 1 code comparison (NMSE $7.9\times10^{-4}$ against Blacklight), 1 without data (the SANE run) |
| `figures/kpolaris/` | KPolaris (arXiv:2609.26298) | 4 | code comparison (I in kind, $V$ open), accuracy against cost for the $C$ ladder, render time against resolution on one and two A100s and 32 threads, the selected-ray Poincaré diagnostics; no RIAF source |
| `figures/blacklight/` | Blacklight (arXiv:2203.15963) | 17 | the deflection tests and the five Gold et al. 2020 formula models as printed (model fluxes within 0.5% of Mahakala's), 11 fiducial-deck figures in kind on the torus, slow light and the two scaling figures preliminary (30 M dump cadence, shared node) |

The error amplitudes of Mahakala's geodesic tests are $10^3$–$10^4$ below the package's at
equal nominal step because that paper steps in coordinate time ($k^t \approx 5$–$10$ on those
orbits) where the package marches in the affine parameter; the orders of convergence agree.
