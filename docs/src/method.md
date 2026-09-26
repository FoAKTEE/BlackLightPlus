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

## [Conventions](@id method-conventions)

Metric signature $(-,+,+,+)$; geometrized units $G=c=M=1$ (lengths in
$r_g=GM/c^2$, times in $r_g/c$); Cartesian Kerr–Schild (CKS) coordinates
$x^\mu=(t,x,y,z)$ with $g_{\mu\nu}=\eta_{\mu\nu}+f\,l_\mu l_\nu$. The geodesic
state is the 8-vector $y=[x^\mu;\,p_\mu]$ carrying **covariant** momentum.
Transfer coefficients propagate in Lorentz-invariant form: $j_\nu/\nu^2$,
$\alpha_\nu\,\nu$, $\rho_\nu\,\nu$, and invariant intensity
$\mathcal I=I_\nu/\nu^3$; the camera output is $S_\nu=\nu_{\rm obs}^3\,\mathcal S$.
All physics arithmetic is `Float64` unless a `:mixed` precision lane is
requested; plasma and radiation quantities are in cgs, with the code magnetic
field $b^\mu$ in Heaviside–Lorentz units ($B=\sqrt{b^2}\,c\sqrt{4\pi\rho_{\rm unit}}$,
$\beta=2p_{\rm gas}/b^2$, $\sigma=b^2/\rho$).

| Quantity | Convention |
|---|---|
| Received photon | wave covector $k_\mu$ normalized to $k_t=-1$ at the camera; fluid-frame frequency $\nu=\nu_{\rm obs}\,(-k_\mu u^\mu)$ |
| Traced ray | state $p_\mu=-k_\mu$ ($p_t=+1$), integrated to increasing $\lambda$: the received photon traced backward, past-directed, $x^t$ decreasing from 0 at the camera; every consumer reads $k=-p$ |
| Slow light | emission time $t_{\rm emit}=t_{\rm obs}+x^t$ |
| Camera axes | inclination $i$ from $+\hat z$ (spin axis), observer azimuth 0; $\hat\beta=(-\cos i,0,\sin i)$ is north (image up), $\hat\alpha=+\hat y$; images stored $M[i_\beta,i_\alpha]$ |
| Screen coordinates | Bardeen's: a ray at $\alpha$ has $\lambda=L_z/E=-\alpha\sin i$ |
| Stokes basis | referred to $(e_{(2)},-e_{(1)})$ of each tetrad, $e_{(3)}$ along $k$; at the camera $(e_{(2)},-e_{(1)})=$ (north, east) |
| Stokes signs | IEEE/IAU: $Q>0$ north–south, $U>0$ north-east–south-west, EVPA $\chi=\tfrac12\arg(Q+iU)$ east of north, $V>0$ counter-clockwise as seen by the observer |
| Transfer matrix | $d\mathcal S/d\lambda=j-\mathbf K\mathcal S$ in the form of [LandiDeglInnocenti1985](@citet), evaluated in the fluid tetrad where $j_U=\alpha_U=\rho_U=0$ |

The backward orientation and the camera Stokes frame are pinned by
`test/test_ray_orientation.jl`: $\dot x^t<0$ along the trace,
$L_z/E=-\alpha\sin i$, $\nu/\nu_{\rm obs}=\gamma(1-v)$ for gas approaching the
camera, EVPA perpendicular to the projected field on thin slabs, $V>0$ for a
field pointing toward the observer, and the sense of Faraday rotation.

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
  second mass vanishes; rays end by the shared rule of [March termination](@ref).
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
inclination $i$. Each pixel's ray is the photon received at the camera, traced
backward with the orientation and normalization of [Conventions](@ref method-conventions), so the
fluid-frame frequency needs no per-pixel factor. The pixel-to-ray map
`camera_ray` is one `isbits` function shared by the host `camera_rays` and the
device ray generator, so host and device rays are identical.

### March termination

Every fixed-control march — the CPU sampled tracers (Kerr and binary), the
device tracer, and every fused kernel, unpolarized or polarized, in `Float64`
or in the `Float32` lane — ends through one rule, `Geodesics.march_stop`
(`src/geodesics/integrator.jl`), so CPU and device integrate the same sample
set:

- **Horizon backstop**: $d<\epsilon_h=10^{-3}$ (`HORIZON_EPS_SAMPLED`), where
  $d=r-r_h$ for Kerr, $d=\min_A(r_A-r_{h,A})$ for a superposed binary, and the
  lapse excess over its floor for an ADM grid; a non-finite $d$ also captures.
- **Redshift cut**: the photon energy in the local normal frame,
  $E_n=-k\cdot n=\alpha\,|dx^t/d\lambda|$ (camera energy 1), exceeds
  $E_c=10^2$ (`CAPTURE_REDSHIFT`).
- **Escape**: outbound ($R>R_{\rm prev}$) beyond the escape sphere
  ($1.1\times10^3$ M; smaller for an ADM grid) or beyond the emission radius.

A backward ray falling toward a hole approaches the future horizon
asymptotically: $x^t\to-\infty$ while $E_n\propto1/(r-r_h)$ grows without bound.
The redshift cut ends it at finite affine parameter. The affine length left
beyond the cut is $\Delta\lambda_{\rm rest}\approx\alpha_h/(\kappa E_c)$
($\kappa$ the surface gravity, $\alpha_h\le1/\sqrt2$ the Kerr–Schild lapse on the
horizon; $0.045\,M$ at $a=0.9$), and since the fluid-frame frequency there is
at least $E_c\,\nu_{\rm obs}/(2\gamma)$, the omitted observed intensity obeys

$$\Delta I_\nu\le\nu_{\rm obs}^3\,\frac{L_{\rm unit}}{\nu_{\rm obs}}\,
\Delta\lambda_{\rm rest}\max_{\rm beyond\ cut}\frac{j_\nu}{\nu^2},$$

with transmission bounded by one. `test/test_march_termination.jl` checks this
bound ray by ray and $\Delta\lambda_{\rm rest}$ against $\alpha_h/(\kappa E_c)$
(measured maximum 0.0450 against 0.0466 on a plunging Kerr deck). Up to the cut
the parallel-transported camera frame of the polarized march keeps the
invariant $-k\cdot e_u$ within 1% of its camera value (maximum drift
$5.2\times10^{-3}$ in `Float64`, $6.0\times10^{-3}$ in `Float32`). In a
superposed binary the companion's potential moves each hole's horizon outward
(to $r_A\approx1.04$ for $m_A=0.5$ at separation 25), where $r_A-r_{h,A}$ stalls
above $\epsilon_h$; the redshift cut is what captures those rays
(`test/test_bbh_superposed.jl`).

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
operator $S\mapsto MS+b$ (`stokes_segment`). The coupling handles all
branches — emission-only, pure absorption, pure Faraday rotation, and the full
absorption+rotation eigenstructure — and matches matrix-exponential oracles to
$10^{-9}$–$10^{-16}$. Instead of transporting the coherency tensor
$N^{\mu\nu}$ [MoscibrodzkaGammie2018](@citep), the camera polarization frame
(its vertical leg and 4-velocity) is parallel-transported along the ray with the
same fourth-order Runge–Kutta step as the geodesic, and each segment operator is
rotated into it by a Q–U rotation $R$ (the angle between the fluid $e_{(1)}$ and
the transported screen leg). The Stokes vector is accumulated camera-to-source
with a running Mueller transmission $G$: $S\mathrel{+}=G\,(Rb)$,
$G\leftarrow G\,(RMR^{\mathsf T})$. Per-ray state is $O(1)$ regardless of path
length, and the operator of each segment is exact for constant coefficients, so
the transfer adds no discretisation error beyond the sampling of the
coefficients along the ray.
On a Kerr torus deck ($a=0.9375$, $16^2$ pixels, $C=20$–$320$ against $C=1280$)
the scheme converges at first order in $1/C$ like the coefficient quadrature,
with 30% smaller Stokes errors and five to eight times smaller EVPA errors than
a Strang-split coherency transport at equal $C$ ($C=80$: $3.75\times10^{-3}$ /
$0.008^\circ$ against $5.40\times10^{-3}$ / $0.065^\circ$); its Stokes $I$
departs from the unpolarized march by $1.9\times10^{-7}$ at $C=1280$, against
$1.1\times10^{-4}$ for the coherency transport.
`fluid_tetrad_checked` ends a march whose screen leg has no spacelike solution
($k$ far off the null cone at a coarse step) instead of producing a NaN.

In the fluid frame $e_{(2)}$ is the projection of $b^\mu$, the basis of the
[Marszewski2021](@citet) coefficients ($Q>0$ in the $k$–$B$ plane); the tetrad
rule applied at the camera gives the IEEE/IAU output of [Conventions](@ref method-conventions)
with no relabelling.

**Admissibility.** Every segment operator enforces the physical cone on its
coefficients before it is formed: when $|\alpha_P|=(\alpha_Q^2+\alpha_V^2)^{1/2}$
exceeds $\alpha_I$ the polarized absorptivities are rescaled onto
$|\alpha_P|=\alpha_I$, and when $|j_P|$ exceeds $j_I$ the polarized
emissivities are rescaled onto $|j_P|=j_I$, keeping $j_I$ and the polarization
direction. The thermal fits of [Marszewski2021](@citet) give $|j_P|>j_I$ at small
pitch angles; without the rescale the observed Stokes vector would depend on
whether such an emitter lies in front of or behind brighter gas, because the
accumulated vector is clamped to $|P|\le I$ (`stokes_admissible`) only on
partial sums (`test/test_emission_admissibility.jl`: a thermal slab and a
field-aligned screen give the same $V/I$ in either order).

**One physics source.** The plasma decode, the coefficient functions (fed by the
`isbits` parameter bundle `coefficient_params`), the fluid tetrad, the segment
operators and the two reverse marches (`reverse_unpolarized_march`,
`reverse_polarized_march`) are single functions called by the CPU drivers and
by every fused kernel. A march control supplies the geodesic: `KerrMarch`
(fixed-control Kerr, `Float64` or `Float32` lane), `SpacetimeMarch` (binary,
ADM, or Kerr through the generic control) or `StoredPath` (the states of
`trace_rays` / `trace_rays_sampled`). On identical trajectories the CPU march
and the fused kernel agree to $1.4\times10^{-8}$ (relative $L_2$, $I$) and
$\le3.5\times10^{-9}$ ($Q,U,V$) at $C=1280$.

## Fused GPU imaging

The GPU production path collapses geodesic integration, fluid sampling, and
radiative transfer into a single KernelAbstractions kernel, one ray per thread,
storing no trajectory — only the image returns to the host. Key elements:

- **One march per ray, no stored path.** The kernel runs the reverse march of
  [Radiative transfer](@ref) itself, advancing the geodesic, the two
  transported frame vectors and the Stokes accumulation in the same loop, so
  per-ray state is $O(1)$ regardless of path length. On identical trajectories
  CPU and GPU therefore agree to floating-point contraction.
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

## Multi-device execution

Every GPU driver runs on a persistent `DevicePool`, one per device list,
created on first use: two worker tasks per device, each with its own stream,
device buffers and pinned staging buffers. Work is a queue of chunks; each
worker pulls the next chunk when it finishes the last, so a slower or shared
device takes fewer. The design rules:

- **Ownership, not partition, fixes the result.** Every ray writes only its own
  pixel, every AMR block only its own tile, and every Monte Carlo photon draws
  from a stream named by its genealogy, not by the chunk that carries it. The
  output is therefore bitwise identical for any device count, chunk size and
  schedule (`test/test_device_invariance.jl`, `test/test_device_pool.jl`).
- **Rays are generated on the device.** A chunk uploads only its pixel or block
  indices (4 bytes per item); the device builds each camera ray with the same
  `camera_ray` function as the host `camera_rays` (`test/test_device_rays.jl`:
  device rays `isequal` host rays).
- **Longest work first.** Full-frame Kerr drivers queue $16\times16$-pixel tiles
  in order of predicted cost — distance to the critical curve on a 4-pixel
  grid (`cost_order`) — so the rays that wind near the photon ring start first
  instead of ending the render; `march_sort_order(counts)` orders by measured
  trip counts from a previous frame, longest first. In `:mixed` precision the
  near-ring pixels that need `Float64` enter the same queue ahead of the
  `Float32` pixels (`test/test_device_schedule.jl`: tile rank against measured
  trip counts, Spearman $\rho=0.95$).
- **Streaming adaptive refinement.** Each refinement block is a work item; the
  worker that receives a block's Stokes values paints it, runs the refinement
  test and queues its four children at once, so refined blocks march while the
  rest of their parent level is in flight. Mosaic, level map and ray count equal
  those of the level-synchronous driver (`test/test_device_amr.jl`).
- **Upload once.** A snapshot or uniform grid is uploaded once per pool and
  reused by later calls on the same object; Monte Carlo host tables are built
  once per call and uploaded to all devices concurrently.

The pool needs at least one more Julia thread than devices (`-t 3` for two
GPUs) so that the workers of different devices run concurrently; with fewer it
warns, since the device workers cannot all run at once. On the KernelAbstractions CPU backend,
`HostDevice` entries act as logical devices, which is how the invariance tests
run without a GPU.

**Performance status.** The pool, device ray generation, cost-ordered queue,
streaming AMR driver, single-source kernels and shared marches are verified
for correctness on the KernelAbstractions CPU backend and by the offline
`sm_80` register, stack and spill audit; their GPU throughput has not been
measured. The GPU numbers and the pending acceptance runs are on
[Benchmarks](@ref).

## Adaptive image-plane refinement

`render_amr` implements the adaptive ray tracing of [White2022](@citet): the image is
tiled into blocks; relative-gradient and relative-Laplacian indicators with
critical-fraction thresholds flag blocks, which spawn four children at doubled
pixel density up to `max_level`. All levels share one Stokes marcher; the
device marcher `AMRStokes` streams the blocks through the device pool
([Multi-device execution](@ref)), and CPU and GPU backends make identical
refinement decisions. Refinement typically saves
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
trigger (unbiased by construction; bias only reduces variance). Spectra
are binned in (exit angle, $\log\nu$) with grmonty normalization; the
optically thin synchrotron sphere matches its closed form to a few percent
per bin, and GRMHD spectra show the gravitationally redshifted synchrotron
bump. On GPU, multi-order scattering is layered rather than recursive: the host
relaunches once per scattering generation, and a launch whose child buffer
overflows is rerun at the exact size, so no child is dropped.

**Streams keyed by ancestry.** Randomness is the counter-based
Philox4x32-10 generator [Salmon2011](@citep). Each photon's stream is named by
its genealogy: a 128-bit identity
$I=g+\sum_{j=1}^{k}e_j\,2^{G+(j-1)E}+k\,2^{124}$, with $g<2^G$ the primary's
global index, $e_j<2^E$ the march step at which the layer-$j$ ancestor
scattered, and $k\le15$ the scattering layer, carried in Philox key word 2 and
counter words 2–4 (the draw index in counter word 1, the seed in key word 1).
`MCStreamLayout` sizes $G$ from the primary count and $E$ from the step budget
and rejects any run with $G+kE>124$, so distinct genealogies map to disjoint
bit fields and hence distinct Philox inputs for every draw. A primary with
$g<2^{32}$ has the stream `MCRng(seed, g)`. No stream depends on the thread,
chunk, device or buffer slot that carries the photon.

**Exact accumulation.** Spectrum bins and scattering diagnostics are summed
exactly. Every finite double is an integer multiple of $2^{-1074}$ below
$2^{1024}$; the accumulator stores that integer in 68 signed 64-bit limbs of 31
bits (2108 bits for the 2098-bit range). One addend touches three consecutive
limbs, each by less than $2^{31}$, so a bin holds at least $2^{32}-1$ addends
before a limb can overflow, and the sum is rounded once to nearest-even at the
end. Integer addition is associative, so the result is independent of order,
grouping and atomic interleaving; the cost is three integer adds (atomic on a
device) per addend. A fixed-point window of two or three words would not
suffice: the synchrotron weight table alone spans about $2^{152}$.

With both, spectra, diagnostics and the set of scattered children are bitwise
identical for any thread count, device count and schedule
(`test/test_mc_exact.jl`, `test/test_device_invariance.jl`: 1, 2, 5 devices;
1, 4, 16 threads), and the device kernels on the KernelAbstractions CPU backend
reproduce the CPU drivers bit for bit.

## Deviations from reference codes

Three deliberate, oracle-verified departures from Blacklight
[White2022](@citep) and ipole [MoscibrodzkaGammie2018](@citep):

1. **The dedicated $\rho=0$ coupling branch is removed.** The references'
   absorption-without-rotation branch disagrees with an exact
   matrix-exponential/RK4 oracle at the $\sim20\%$ level on emitting slabs.
   BlackLightPlus routes $\rho_p=0$ through the full analytic branch, which
   reduces exactly to that limit and matches the oracle to $10^{-9}$.
2. **Transported camera frame instead of the coherency tensor.** ipole
   parallel-transports the coherency tensor $N^{\mu\nu}$ between segments;
   BlackLightPlus transports the camera readout frame and applies the exact
   segment operator rotated into it ([Radiative transfer](@ref)). Against a
   Strang-split coherency transport at equal step density the EVPA error is
   five to eight times smaller.
3. **Degenerate-coupling fallback** is the total-absorption solution
   (admissibility-clamped), not the removed defective branch.

The images are cross-validated against RAPTOR [Bronzwaer2018](@citep) on
analytic scenes and binary-black-hole GRMHD snapshots; protocol and results are
on [Validation against RAPTOR](@ref).

The polarized-transfer stiffness diagnostic re-derives the transfer-matrix
eigenvalue analysis used by RAPTOR [Bronzwaer2018](@citep); that derivation is
independent, and no RAPTOR source was consulted.

## Works cited

Full bibliographic details, DOIs, and arXiv identifiers for everything cited
above are on the [References](@ref "References") page.
