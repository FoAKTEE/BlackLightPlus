# Validation

The package is held to three comparisons: the bitwise oracle against the reference
implementation it replaced, the cross-code comparisons against RAPTOR and Blacklight, and
the reproduction of published figures. The measurements, with their commands, are in the
[Verification record](@ref); the evidence files they cite (`results/blp/`) and the
reference implementation live on the `blp` branch of the repository.

## The oracle

The previous implementation is the reference: on identical decks the package reproduces
its Stokes $I$ images on every path, its samplers, plasma decode, shadows and the Monte
Carlo emissivities, hot cross section and Compton kick bitwise on one hardware, and its
$IQUV$ images and traces to round-off (bitwise under `julia -C generic`). The Monte Carlo
estimator departs from the reference's where that one is biased (the zones, the counts,
the events of a step and their child weights; [Method](@ref "Monte Carlo spectral
synthesis")), so the Monte Carlo spectra are validated against closed forms, against ray
tracing and across biases instead. On the A100 the reference device drivers
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
integration or registration. Against the released Blacklight, $Q$ and $U$ agreed only where Faraday effects are weak
and Stokes $V$ differed by a factor 2–7 at every frequency. The cause is in Blacklight: its
unsplit polarized coupling (`polarized.cpp`, the path taken when absorption and rotation
coexist) writes the $(U,V)$ entries of the Landi Degl'Innocenti matrices $M_2$ and $M_3$
into the $(Q,U)$ slot, so the $U \leftrightarrow V$ coupling of Faraday conversion is lost;
upstream master carries the same lines. Both codes evaluate the same coefficient fits,
thresholds, pitch-angle sign and transfer matrix, and the package's segment operator
equals $\exp(-K\,ds)$ to $1.6\times10^{-10}$ on 4000 random admissible coefficient sets.
With the two indices corrected in Blacklight and nothing else, the images agree: at
230 GHz NMSE$(V)$ falls from 52 to 0.041 (correlation 0.13 to 0.98) and NMSE$(Q,U)$ from
0.37/0.31 to 0.012/0.011, at 1 THz NMSE$(V)$ from 4.6 to 0.076, with $I$ unchanged. The
patch, the builds and the statistics are under `results/blp/stokes_v/` on the `blp`
branch.

## Closed-form Kerr against AART and Krang

The closed-form tracer is compared with AART [CardenasAvendano2023](@citep) and Krang
[Chang2024](@citep) at identical screen points (the nodes of a square grid, even so that no
node lies on the $\alpha$ axis), each code with its own elliptic integrals, and with the
package's own numerical march. NMSE is the normalized mean squared error of
[Gold2020](@citet) with the reference code in the denominator.

| Comparison | Result |
|---|---|
| equatorial g-disk intensity, orders 0–2, $2500^2$ and $1200^2$ | NMSE $\sim10^{-22}$ against AART and Krang |
| polarized g-disk, Walker–Penrose constant in AART's gauge ($f_t=1$), $400^2$, two scenes | NMSE($\kappa$) $7\times10^{-25}$, $3\times10^{-24}$; EVPA median $9\times10^{-16}$ rad, maximum $10^{-9}$ rad (on AART's Cardano curve) |
| power-law synchrotron surfaces, $400^2$, four scenes (dual cone, equator at $80^\circ$, jet cones in both hemispheres, polar cones with vortical rays), orders 0–2 | NMSE of $I$, $Q$, $U$ between $7\times10^{-29}$ and $2\times10^{-24}$ against Krang; EVPA median $10^{-15}$ rad |
| cone landings against the traced crossings of the same rays (ordinary and vortical, cases II–IV) | radius within $10^{-4}$, the polar sign of every crossing equal |
| landing radius at orders 0–6 along screen rays through 9 critical points, two scenes | within $\varepsilon\,\lvert\partial\ln r_s/\partial\ln s\rvert$ of Krang ($3\times10^{-10}$ at $n=0$, $10^{-6}$ at $n=6$, where $d_6\sim10^{-7}\,M$) |
| order-0 landing radius on the radial cuts through the critical curve ($1.6\times10^6$ and $1.5\times10^6$ points, $1-k_r$ down to $10^{-15}$) | within $8\times10^{-14}$ of a 40-digit landing at the 70 and 60 points where Krang is off by up to $4\times10^{-3}$ and $1.2$ (relative); NMSE of the cut intensity against AART $4\times10^{-21}$–$10^{-19}$ per order outside AART's own 2 and 4 outlying points ($7\times10^{-12}$ with them) |
| photon-ring exponents against the landings of orders 6 and 7 | $\gamma$, $\delta$, $\tau$ to $10^{-6}$–$10^{-5}$, the differences falling as $e^{-\gamma n}$ |
| orbiting hot spot, 25 frames over one orbit, orders 0–2, $240^2$ | against AART's `flare_model` at $D=10^4$: NMSE $10^{-17}$ ($n=0$), $10^{-14}$ ($n=1,2$) outside the 81 pixels of AART's turning-point sign; against Krang's slow-light coordinates at infinity: NMSE $3\times10^{-18}$, $10^{-16}$, $2\times10^{-15}$; light curves to $10^{-9}$ |
| per-order resolution of the photon-ring layers, $a=0.94$, $17^\circ$ | spacings $(0.02, 0.01, 0.005)$ reproduce the $n=1$, $2$ ring fluxes of uniform $0.01$ and $0.005$ grids with $2.0\times10^6$ rays instead of $6.1\times10^6$ and $2.5\times10^7$ |
| every closed-form product on one A100 against the CPU threads (g-disk, intensity-only g-disk, dual cone, polar cones, polarized g-disk, hot-spot movie) | 53–85 % of the values bitwise, the rest within $4\times10^{-9}$ of the image maximum ($6\times10^{-8}$ on the hot-spot movie), the floating-point contraction of the two targets; $3.5$–$7\times10^6$ rays/s, 7–21 times four CPU threads |
| EVPA against the package's polarized march (transported camera frame) | falls as $C^{-4}$ to the camera floor: $2\times10^{-5}$ rad at $d=10^5\,M$, median $10^{-8}$ rad at $d=10^9\,M$ |

Four reference-code findings shape these comparisons. Krang's polarized synchrotron
material weights its Stokes $I$ by $|\hat k\times B|^{2(1+\sigma)}$ where its intensity material
(and the emissivity of the model) has $|\hat k\times B|^{1+\sigma}$; its fast-light entry points
give the emission polar sign of a cone in the far hemisphere outside the observer's cone the
wrong sign. Both are corrected in the comparison and reported upstream. AART flips the
radial sign of emission at the turning point seen from infinity while its radial integrals
start at the finite observer distance, which tilts the EVPA by up to $4\times10^{-5}$ rad
and the emission time by up to $4\times10^{-2}\,M$ on a thin ring at $D=10^4$–$10^5$; the
polarization comparison uses $D=10^{15}$ and the movie comparison separates those pixels;
the fix is reported upstream. JacobiElliptic.jl 0.3, which Krang uses, takes the first-order
expansion of $\mathrm{am}(u|m)$ about $m=1$ wherever $1-m<\sqrt\varepsilon$, at any $u$; on
rays near the critical curve, with $u\approx K(m)$, it moves the landing radius by up to
$4\times10^{-3}$ (relative) at $17^\circ$ and by order one at $80^\circ$, and the intensity by up
to $3\times10^{-3}$ of the maximum. SciPy's
`ellipj`, which AART uses, does the same for $1-m\le10^{-10}$. The package's
`jacobi_amplitude` keeps the expansion to $(1-m)\cosh^2u<\sqrt\varepsilon$; the JacobiElliptic.jl
fix is reported upstream.

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
