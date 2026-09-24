# Method

BlackLightPlus is a Julia package for general-relativistic ray tracing and fully
polarized radiative transfer in arbitrary spacetimes, on CPUs and GPUs, with
end-to-end automatic differentiation (AD). It images GRMHD simulation data or
semi-analytic sources, and includes a Monte Carlo Compton-scattering module for
spectral synthesis. The design follows two reference lineages — Blacklight
[White2022](@citep) for structure and ipole [MoscibrodzkaGammie2018](@citep) for
the polarized transfer scheme — with the analytic-Kerr machinery of AART
[CardenasAvendano2023](@citep) and Krang.jl [Chang2024](@citep), and a
grmonty-class Monte Carlo [Dolence2009](@citep).

Two design rules hold throughout. Every physical or geometric primitive has a
single implementation shared by all consumers (CPU imaging, GPU kernels, Monte
Carlo, AD). Everything reachable from a march loop is pure, allocation-free,
`isbits`, and StaticArrays-typed, so one code path serves CPU threads,
KernelAbstractions GPU kernels, and ForwardDiff/Enzyme differentiation.

## Conventions

Metric signature $(-,+,+,+)$; geometrized units $G=c=M=1$; Cartesian
Kerr–Schild (CKS) coordinates $x^\mu=(t,x,y,z)$. The geodesic state is the
8-vector $y=[x^\mu;\,p_\mu]$ carrying **covariant** momentum. Transfer
coefficients propagate in Lorentz-invariant form: $j_\nu/\nu^2$,
$\alpha_\nu\,\nu$, $\rho_\nu\,\nu$, and invariant intensity
$\mathcal I=I_\nu/\nu^3$. All physics arithmetic is `Float64`; plasma
quantities are in cgs.

## Spacetimes

Every spacetime supplies only a covariant metric closure
$x\mapsto g_{\mu\nu}(x)$; all derivative machinery (inverse metric gradients,
Christoffel symbols) is derived by forward-mode AD (ForwardDiff), so a new
metric never needs hand-coded derivatives.

- **`KerrSchildCartesian(a)`** — Kerr in CKS form
  $g_{ab}=\eta_{ab}+f\,l_a l_b$, with the closed-form inverse
  $g^{ab}=\eta^{ab}-f\,l^a l^b$ (exact because $l$ is null), so no matrix
  inverse is needed on the hot path.
- **`UserMetric(f)`** — wraps any closure $x\mapsto g_{\mu\nu}$; inverse and
  gradients come from StaticArrays `inv` and AD. **`Minkowski()`** is flat space.
- **`SuperposedKerrSchild(masses, spins, centers)`** — superposes $N$
  Kerr–Schild holes, $g_{\mu\nu}=\eta_{\mu\nu}+\sum_A f_A l^A_\mu l^A_\nu$,
  an analytic approximate binary (or $N$-body) spacetime. The cross terms break
  the single-hole inverse identity, so the full $4\times4$ inverse is taken and
  differentiated by AD. The construction reduces bitwise to single Kerr when a
  second mass vanishes; rays terminate on entering any hole's horizon.
- **`ADMGridSpacetime`** — a frozen-slice numerical spacetime: a Cartesian grid
  of the ten ADM fields $(\alpha,\beta^i,\gamma_{ij})$, reassembled into the
  4-metric via the standard 3+1 decomposition. Fields are interpolated with a
  $C^1$ cubic Catmull–Rom tensor-product stencil — $C^1$ continuity is required
  because AD differentiates *through* the interpolant to obtain
  $\partial g_{\mu\nu}$, and a trilinear ($C^0$) interpolant would produce
  discontinuous Christoffels at cell faces. Rays terminate where the lapse
  falls below a threshold (numerical horizon) or off-grid.
  `adm_grid_from_metric` samples any analytic metric onto a grid for testing.
- **Z4c input** — `read_athenak_bin` parses AthenaK [Stone2026](@citep)
  v1.1 `.bin` AMR dumps, as written by its numerical-relativity solver
  [Zhu2025](@citep); `resample_z4c_adm` / `z4c_grid_spacetime` map the
  conformal Z4c variables to physical ADM
  ($\gamma_{ij}=\tilde g_{ij}/\chi$, $\alpha$, $\beta^i$ passed through) and
  build an `ADMGridSpacetime`. A real binary-black-hole merger dump
  reconstructs with $g\,g^{-1}=\mathbb 1$ to $4\times10^{-16}$ and lenses into
  two puncture shadows. The frozen slice sets $\partial_t g=0$, valid while the
  light-crossing time of the imaged volume is short compared to the spacetime's
  evolution timescale.

## Geodesics

Null geodesics integrate the super-Hamiltonian
$H=\tfrac12 g^{\mu\nu}p_\mu p_\nu$:

$$\frac{dx^\mu}{d\lambda}=g^{\mu\nu}p_\nu,\qquad
\frac{dp_\mu}{d\lambda}=-\tfrac12\,(\partial_\mu g^{\alpha\beta})\,p_\alpha p_\beta .$$

For stationary metrics $\dot p_t\equiv0$, so integrating covariant momentum
conserves the energy $E=-p_t$ exactly (zero drift by construction); only the
three spatial $p_i$ are advanced, with $\partial_i g^{\mu\nu}$ from AD.
`nullify_covariant` closes the initial condition by solving the null quadratic
for the root with $p_t<0$ (positive energy).

Two integration modes are provided. `trace_geodesic` is an adaptive
high-order solve (Vern7, `reltol` $10^{-9}$–$10^{-10}$) with event-terminated
horizon capture and escape. The GPU/AD static-graph mode is fixed-step
classical RK4 (`rk4_step_fixed`); the production sampler `trace_rays_sampled`
marches the control step $h=(r-r_h)/C$ (dense near the hole), which doubles as
the radiation sampling grid, and returns trajectories reordered source-to-camera
with per-segment affine spans. Validation gates: exact $p_t$ conservation,
$L_z$ and null drift below $10^{-8}$, Schwarzschild shadow $b_c=3\sqrt3$ to
better than $10^{-3}$.

`PlaneParallelCamera` places a parallel-ray image plane at distance $d$ and
inclination $i$. Each pixel's ray is the photon received at the camera, with
wave covector $k_\mu$ normalized to $k_t=-1$, so the fluid-frame frequency
$\nu=-k_\mu u^\mu\,\nu_{\rm obs}$ needs no per-pixel factor. The ray state
carries $p_\mu=-k_\mu$ and is integrated to increasing $\lambda$: the received
photon traced backward along its null geodesic, past-directed, with $x^t$
decreasing from 0 at the camera (slow light samples the fluid at
$t_{\rm emit}=t_{\rm obs}+x^t$). Image coordinates $(\alpha,\beta)$ are
Bardeen's screen coordinates: a ray at $\alpha$ has $\lambda=L_z/E=-\alpha\sin i$.

## Analytic Kerr fast path and lensing bands

For equatorial sources in Kerr, an exact closed-form tracer (no ODE) inverts
the geodesic with Legendre/Jacobi elliptic integrals in the formulation of
[GrallaLupsasca2020NullGeodesics](@citet) and its lensing companion
[GrallaLupsasca2020Lensing](@citep), following the conventions of Krang.jl
[Chang2024](@citep) and the finite-observer photon-ring machinery of AART
[CardenasAvendano2023](@citep):
conserved quantities $(\lambda,\eta)$ from Bardeen screen coordinates, quartic
radial roots, angular and radial Mino-time integrals, and per-image-order
$n=0,1,2$ equatorial landing points $(r_s,\phi_s,t_s)$. `critical_curve`
returns the photon-shell boundary; `lensing_band` and `band_grid` build the
nested order-$n$ lensing-band masks used by `photon_ring_layers` to compute
razor-thin photon-ring images and, via the interferometry utilities, visibility
amplitudes. The analytic $r_s$ agrees with the ODE tracer's equatorial crossing
to $6\times10^{-6}$, and ForwardDiff propagates $dr_s/d(a,\theta_o)$ through
the elliptic functions (verified against finite differences).

## Radiative transfer

A single plasma-decode chain converts sampled code-unit
$(\rho,p_{\rm gas},b^2)$ to cgs $(n_e,\Theta_e,B,\sigma,\beta^{-1})$, with the
$R_{\rm high}/R_{\rm low}$ electron-temperature prescription
[Moscibrodzka2016](@citep) and configurable $\sigma$, $\beta$,
$\Theta_e$ emission cuts. The same chain feeds imaging and Monte Carlo.

**Coefficients.** `eight_coefficients` returns the invariant emissivities,
absorptivities, and rotativities
$(j_{I,Q,V};\ \alpha_{I,Q,V};\ \rho_{Q,V})$ in the fluid frame, additively
mixing three electron distribution functions:

- *thermal* — polarized emissivities in the form of
  [Dexter2016](@citet), with the fits of [Marszewski2021](@citet) for the
  Faraday rotation ($\rho_V$) and conversion ($\rho_Q$) coefficients,
  absorption by Kirchhoff's law;
- *power-law* — [Pandya2016](@citet) fits with host-precomputed
  $p$-dependent constants;
- *kappa* — [Pandya2016](@citet) low/high-frequency asymptotes joined by
  harmonic bridges, with precomputed $\Gamma$, Bessel, and ${}_2F_1$ factors.

All special functions expensive on device (Bessel $K_\nu$, $\Gamma$,
hypergeometric) are either evaluated in host-side precomputation or replaced by
device-safe polynomial fits, so the per-step kernel code is closed-form.

**Unpolarized transport** uses the exact constant-coefficient segment solution
of the invariant transfer equation
$d\mathcal I/d\lambda=j_{\rm inv}-\alpha_{\rm inv}\mathcal I$, written as the
affine operator $\mathcal I\mapsto A\mathcal I+B$ with $A=e^{-\Delta\tau}$,
$B=(j/\alpha)\,e^{-\Delta\tau}\,\mathrm{expm1}(\Delta\tau)$, and saturation
to the source function when thick.

**Polarized transport** applies the analytic constant-coefficient Stokes
coupling of [LandiDeglInnocenti1985](@citet) per segment in the fluid tetrad
(Gram–Schmidt frame anchored to $u^\mu$, $k^\mu$, $b^\mu$), as an affine
operator $S\mapsto MS+b$. Instead of transporting the coherency tensor
$N^{\mu\nu}$ [MoscibrodzkaGammie2018](@citep), the camera polarization frame
(its vertical leg and 4-velocity) is parallel-transported along the ray with the
same fourth-order Runge–Kutta step as the geodesic, and each segment operator is
rotated into it by a Q–U rotation (the angle between the fluid $e_{(1)}$ and the
transported screen leg). The Stokes vector is accumulated camera-to-source with a
running Mueller transmission (below); CPU and GPU run this one march. On a
Kerr torus deck the scheme converges at first order in $1/C$ like the coefficient
quadrature, with 30% smaller Stokes errors and five to eight times smaller EVPA
errors than the Strang-split coherency transport at equal $C$.
Stokes parameters are referred to the basis $(e_{(2)},-e_{(1)})$ of each tetrad,
with $e_{(3)}$ along the propagation direction. In the fluid frame $e_{(2)}$ is
the projection of $b^\mu$, the basis of the [Marszewski2021](@citet)
coefficients ($Q>0$ in the $k$–$B$ plane); at the camera $e_{(2)}$ is the image
vertical $+\beta$ (north) and $-e_{(1)}$ is $-\alpha$ (east), so the output
follows IEEE/IAU: $Q>0$ north–south, EVPA $\chi=\tfrac12\arg(Q+iU)$ east of
north, $V>0$ counter-clockwise as seen by the observer. The
coupling handles all branches — emission-only, pure absorption, pure Faraday
rotation, and the full absorption+rotation eigenstructure — and matches
matrix-exponential oracles to $10^{-9}$–$10^{-16}$. Degenerate couplings fall
back safely and results are clamped to physical admissibility
($|P|\le I$).

## Fused GPU imaging

The GPU production path collapses geodesic integration, fluid sampling, and
radiative transfer into a single KernelAbstractions kernel, one ray per thread,
storing no trajectory — only the image returns to the host. Key elements:

- **Reverse affine/Mueller accumulation.** The kernel marches camera-to-source
  while transfer is physically ordered source-to-camera. Each segment couple is
  affine, $S_{\rm out}=M S_{\rm in}+b$, so the observed result is accumulated
  in reverse with a running transmission: $S \mathrel{+}= G\,(Rb)$,
  $G\leftarrow G\,(RMR^{\mathsf T})$, where $(M,b)$ is the analytic segment
  solution in the fluid frame and $R$ the Q–U rotation into a
  parallel-transported camera readout frame (two frame vectors carried as extra
  RK4 states). Per-ray state is $O(1)$ regardless of path length. The kernel
  runs the CPU march itself (one march per transfer mode, fed by a fixed-control
  step or by stored trajectory states), so CPU and GPU agree to floating-point
  contraction on identical trajectories.
- **In-kernel AD.** Metric derivatives come from ForwardDiff dual numbers
  evaluated inside the kernel; no Christoffel tensor is ever materialized
  (a major register saving). The same mechanism seeds a dual on a physical
  scalar (density scale, $R_{\rm high}$) through the transfer accumulation:
  one pass returns the image and its exact parameter derivative
  (the differentiable-GRRT construction of Jipole [NaetheMotta2025](@citep)),
  at $\approx1.13\times$ overhead.
- **Fluid sampling.** A fast baseline resamples the AMR fluid once onto a
  uniform Cartesian box (an approximation that changes fluxes at the percent
  level but can rotate the EVPA on Faraday-thick decks). The production
  sampler instead uploads the AthenaK meshblock hierarchy and samples it
  natively on device: a dense finest-level leaf-index grid gives an $O(1)$
  block locate, followed by the same clamped trilinear interpolation as the
  CPU reference — bit-for-bit identical to the host sampler.
- **Multi-GPU.** Rays are split across devices in interleaved order to balance
  cheap and expensive rays; the Monte Carlo path uses disjoint RNG seed ranges
  per device so multi-GPU runs are reproducible.

## Adaptive image-plane refinement

`render_amr` implements the adaptive ray tracing of [White2022](@citet): the image is
tiled into blocks; relative-gradient and relative-Laplacian indicators with
critical-fraction thresholds flag blocks, which spawn four children at doubled
pixel density up to `max_level`. All levels share one Stokes marcher, and the
fused GPU kernels plug in through a flat ray-list interface, so CPU and GPU
backends make identical refinement decisions. Refinement typically saves
$\sim4\times$ in rays on top of kernel throughput.

## Monte Carlo spectral synthesis

The Monte Carlo module implements the grmonty scheme [Dolence2009](@citep):
superphoton packets sample the synchrotron emissivity zone-by-zone with
frequency and pitch-angle rejection sampling and a log-frequency weight table;
packets carry covariant momentum and decay by Kirchhoff absorption along
geodesics. Compton scattering uses the thermally averaged ("hot") Klein–Nishina
cross section tabulated on a log-log $(\varepsilon,\Theta_e)$ grid, scattering
electrons drawn from the relativistic Maxwellian by the
[Canfield1987](@citet) sampler, exact Klein–Nishina/Thomson rest-frame
kinematics, and the [Kahn1950](@citet) biased exponential-transform scatter
trigger (unbiased by construction; bias only reduces variance). Randomness is
the counter-based Philox4x32-10 generator
[Salmon2011](@citep): stateless, GPU-safe streams keyed
per superphoton, so GPU runs replay the CPU oracle photon-for-photon. Spectra
are binned in (exit angle, $\log\nu$) with grmonty normalization; the
optically thin synchrotron sphere matches its closed form to a few percent
per bin, and GRMHD spectra show the gravitationally redshifted synchrotron
bump. On GPU, multi-order scattering is layered (host relaunch per scattering
generation with atomic child append) rather than recursive.

## Deviations from reference codes

Three deliberate, oracle-verified departures from Blacklight
[White2022](@citep) and ipole [MoscibrodzkaGammie2018](@citep):

1. **The dedicated $\rho=0$ coupling branch is removed.** The references'
   absorption-without-rotation branch disagrees with an exact
   matrix-exponential/RK4 oracle at the $\sim20\%$ level on emitting slabs.
   BlackLightPlus routes $\rho_p=0$ through the full analytic branch, which
   reduces exactly to that limit and matches the oracle to $10^{-9}$.
2. **Coherency transport direction.** $N^{\mu\nu}$ is parallel-transported
   along the physical march direction (source to camera), i.e. along the photon
   wave vector $k^\mu=-p^\mu$ of the traced state; the references' sign
   convention is tied to their sample ordering. Verified by
   polarized/unpolarized consistency.
3. **Degenerate-coupling fallback** is the total-absorption solution
   (admissibility-clamped), not the removed defective branch.

The polarized-transfer stiffness diagnostic re-derives the transfer-matrix
eigenvalue analysis used by RAPTOR [Bronzwaer2018](@citep); that derivation is
independent, and no RAPTOR source was consulted.

## Works cited

Full bibliographic details, DOIs, and arXiv identifiers for everything cited
above are on the [References](@ref "References") page.
