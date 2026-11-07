# Method

BlackLight+ (the package `BlackLightPlus`) is a Julia package for general-relativistic ray
tracing and fully polarized radiative transfer in arbitrary spacetimes, on CPUs and GPUs,
with end-to-end automatic differentiation (AD). It images GRMHD simulation data or
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
All physics arithmetic is `Float64` unless a `Float32` geodesic lane or
`MixedLanes` is requested; plasma and radiation quantities are in cgs, with the code magnetic
field $b^\mu$ in Heaviside–Lorentz units ($B=\sqrt{b^2}\,c\sqrt{4\pi\rho_{\rm unit}}$,
$\beta=2p_{\rm gas}/b^2$, $\sigma=b^2/\rho$).

| Quantity | Convention |
|---|---|
| Received photon | wave covector $k_\mu$ normalized to $k_t=-1$ at the camera; fluid-frame frequency $\nu=\nu_{\rm obs}\,(-k_\mu u^\mu)$ |
| Traced ray | state $p_\mu=-k_\mu$ ($p_t=+1$), integrated to increasing $\lambda$: the received photon traced backward, past-directed, $x^t$ decreasing from 0 at the camera; every consumer reads $k=-p$ |
| Slow light | emission time $t_{\rm emit}=t_{\rm obs}+x^t$, sampled from a `SnapshotSeries` through `SeriesProvider` (host) or a `FrameWindow` of two bracketing frames (any backend), blended linearly in time |
| Camera axes | inclination $i$ from $+\hat z$ (spin axis), observer azimuth 0; $\hat\beta=(-\cos i,0,\sin i)$ is north (image up), $\hat\alpha=+\hat y$; images stored $M[i_\beta,i_\alpha]$ |
| Screen coordinates | Bardeen's: a ray at $\alpha$ has $\lambda=L_z/E=-\alpha\sin i$ |
| Stokes basis | referred to $(e_{(2)},-e_{(1)})$ of each tetrad, $e_{(3)}$ along $k$; at the camera $(e_{(2)},-e_{(1)})=$ (north, east) |
| Stokes signs | IEEE/IAU: $Q>0$ north–south, $U>0$ north-east–south-west, EVPA $\chi=\tfrac12\arg(Q+iU)$ east of north, $V>0$ counter-clockwise as seen by the observer |
| Transfer matrix | $d\mathcal S/d\lambda=j-\mathbf K\mathcal S$ in the form of [LandiDeglInnocenti1985](@citet), evaluated in the fluid tetrad where $j_U=\alpha_U=\rho_U=0$ |

The backward orientation and the camera Stokes frame are pinned by
`test/pipeline/products_orientation.jl` and `test/physics/spacetime_geodesics.jl`: $\dot x^t<0$ along the trace,
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
  the single-hole inverse identity; the inverse follows from the Woodbury
  identity $g^{-1}=\eta^{-1}-U(\mathbf 1_N+FU^{\mathsf T}L)^{-1}FU^{\mathsf T}$, with
  $L$ the $4\times N$ matrix of the $l^A_\mu$, $U=\eta^{-1}L$ and
  $F={\rm diag}(f_A)$, so a binary needs one $2\times2$ solve, and AD
  differentiates through it. The construction reduces bitwise to single Kerr when a
  second mass vanishes; rays end by the shared rule of [March termination](@ref).
- **`BoostedSuperposedKerrSchild`** — the superposition in motion: each
  hole's term in the rest frame of its instantaneous velocity along a
  prescribed trajectory (`CircularOrbit` or a closure), the covector boosted
  back, the Woodbury inverse unchanged; a non-stationary spacetime whose
  geodesic flow keeps ``\partial_t g``. Capture is per hole at the rest-frame
  radius ``r'_A < r_{h,A} + \epsilon`` or by the redshift cut.
- **`ADMGrid`** — a frozen-slice numerical spacetime: a Cartesian grid
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
  build an `ADMGrid`. A real binary-black-hole merger dump
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
horizon capture and escape. The imaging products march with fixed-step
classical RK4 (`rk4_step`) at the control step $h=(r-r_h)/C$ of `march_control`
(dense near the hole), which doubles as the radiation sampling grid; `trace`
returns these trajectories ordered source to camera with their affine steps. Validation gates: exact $p_t$ conservation,
$L_z$ and null drift below $10^{-8}$, Schwarzschild shadow $b_c=3\sqrt3$ to
better than $10^{-3}$.

`PlaneParallelCamera` places a parallel-ray image plane at distance $d$ and
inclination $i$. Each pixel's ray is the photon received at the camera, traced
backward with the orientation and normalization of [Conventions](@ref method-conventions), so the
fluid-frame frequency needs no per-pixel factor. The pixel-to-ray map
`camera_ray` is one `isbits` function shared by the host `camera_rays` and the
device ray generator, so host and device rays are identical.

### March termination

Every fixed-control march, on threads or on a device, unpolarized or polarized,
in `Float64` or in the `Float32` lane, ends through one rule, `Spacetime.march_stop`
(`src/physics/Spacetime/termination.jl`), so CPU and device integrate the same
sample set:

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

with transmission bounded by one. `test/physics/spacetime_termination.jl` checks this
bound ray by ray and $\Delta\lambda_{\rm rest}$ against $\alpha_h/(\kappa E_c)$
(measured maximum 0.0450 against 0.0466 on a plunging Kerr deck). Up to the cut
the parallel-transported camera frame of the polarized march keeps the
invariant $-k\cdot e_u$ within 1% of its camera value (maximum drift
$5.2\times10^{-3}$ in `Float64`, $6.0\times10^{-3}$ in `Float32`). In a
superposed binary the companion's potential moves each hole's horizon outward
(to $r_A\approx1.04$ for $m_A=0.5$ at separation 25), where $r_A-r_{h,A}$ stalls
above $\epsilon_h$; the redshift cut is what captures those rays
(`test/physics/spacetime_superposed.jl`). The moving binary
`BoostedSuperposedKerrSchild` applies the same rule to each hole's rest-frame
radius at the ray's time.

### The precise march

The precise mode (`Plans.precise`, guide [Precise ray tracing](@ref)) keeps the
fixed-control structure of the march, whose step $h$ is a function of the state
alone, and replaces the parts whose errors set the floor of the production march
against the closed form.

**Ninth-order step.** `Verner9` is the ninth-order solution of Verner's 9(8) pair
[Verner2010](@citep): fifteen stages, local error $O(h^{10})$, so at the control step
$h=d/C$ the image error falls as $C^{-9}$. The embedded eighth-order weights are not
used: an accept/reject decision would make the step sequence a discontinuous function
of the initial state, while with $h=h(y)$ two backends whose arithmetic differs in the
last bit take the same steps and end within that rounding times the ray's condition
number. On a long march the rounding of the state update $y\leftarrow y+\Delta y$
accumulates once per step; `march_step_compensated` carries the state as the
unevaluated sum $y+y_{\rm lo}$, with $y_{\rm lo}$ the two-sum error of each update, so
the update rounds once per ray.

**Winding near a spinning horizon.** A ray traced backward toward a black hole of spin
$a$ approaches the past horizon, where its Kerr–Schild azimuth turns by
$d\phi=2a\,dr/\Delta$, $\Delta=(r-r_+)(r-r_-)$. A step of radial extent
$(r-r_+)/C$ turns it by $2|a|/((r-r_-)C)$, up to $7/C$ radians at $a=0.99$, and the
ninth-order step loses its accuracy there. With `winding` the step distance is
$d\,w$, $w=(1+q^4)^{-1/4}$, $q=2|a|/(r-r_-)$ (`winding_factor`, $r_-=a^2/r_+$), which
bounds the turn per step by about $1/C$ and leaves the step unchanged where
$r-r_-\gg2|a|$; a binary takes the smallest $d_A w_A$ over its holes.

**Surface crossings at full order.** An emission surface $s(x)=0$ (the equator $z=0$
for a thin disk, `EquatorialPlane`) is met inside the step whose end has the other
sign of $s$. The crossing is the root in $c\in(0,h)$ of $s(\Phi_c(y))$, with
$\Phi_c$ the integrator's own step of length $c$, found by Newton's iteration with
$ds/dc=\nabla s\cdot dx/d\lambda$ at $\Phi_c(y)$ and a bisection fallback inside the
bracket (`surface_crossing`). Since $\Phi_c$ is the method applied over $c\le h$, the
crossing state carries the local error of the step, $O(h^{p+1})$; the root of the
linear interpolant of $s$ between the step's ends would carry $O(h^2)$. The iteration
converges in about one step per crossing from the secant start, so a crossing costs one
or two partial steps.

**Camera at infinity.** A plane-parallel ray started at distance $d$ with no
transverse momentum is not the ray an observer at infinity receives at the same screen
point: the far zone bends the latter by $\delta p_\perp\sim Mb/d^2$ and displaces it by
$\delta w\sim Mb/d$ at the plane, so its conserved quantities carry an error
$O(M/d)$; at large $d$ the transverse offset $b$ is rounded against $d$, which leaves
$\epsilon d/b$. The camera at infinity (`infinity_ray`) integrates the far zone in
$u=1/s$, $s=x\cdot\hat n$, with the position $x=\hat n s+w$ and the momentum
$p=(1,-\hat n+\delta p)$:

$$\frac{dw}{du}=-\frac{v_\perp}{u^2v_s},\qquad
\frac{d\,\delta p}{du}=-\frac{F}{u^2v_s},\qquad
\frac{d\tau}{du}=-\frac{v^t-v_s}{u^2v_s},$$

$v=g^{\mu\nu}p_\nu$, $v_s=v\cdot\hat n$, $v_\perp=v-\hat n v_s$, $F=dp/d\lambda$,
$\tau=t-s$. A metric with $g-\eta=O(M/r)$ gives $v_\perp=O(u^2)$, $F_\perp=O(u^3)$ and
$F_\parallel=O(u^2)$, so $w$ and $\delta p$ are regular at $u=0$, where the ray from
infinity has $w=(-\beta\cos i,\alpha,\beta\sin i)$ and $\delta p=0$; the Shapiro
term $A/u$ of $d\tau/du$ is common to all screen points and is integrated apart. The
offset $v_c=(g^{\mu\nu}-\eta^{\mu\nu})p_\nu$ (`raise_index_offset`) enters
$v_\perp=(\delta p+v_c)_\perp$; Kerr–Schild spacetimes form it from their terms,
$v_c=-f\,(l\cdot p)\,l$ for one black hole and the Woodbury correction for a
superposition, so every term carries rounding of itself and never of $s$. The general
solution also holds the mode $w\propto1/u$ of a ray at another angle, which an error
of $\delta p_\perp$ excites with weight $1/u$ at the plane; the march therefore starts
at $u_0=10^{-8}$ from the leading behaviour, $\delta p_\perp\propto u^2$ (trapezoid)
and $w-w(0)\propto u$ (midpoint, with corrector passes), an error $O(u_0^2)$, and
takes `Verner9` steps $u\to2u$ to $u=1/d$. Each step $[u,2u]$ has the same relative
structure, so its truncation is a fixed fraction of the bending in that step, below
$10^{-15}\,M$ in total. On Kerr the started ray carries Bardeen's
$\lambda=-\alpha\sin i$ and $\eta=(\alpha^2-a^2)\cos^2i+\beta^2$ to $3\times10^{-14}$
relative (`test/physics/spacetime_integrators.jl`).

**Capture.** In Kerr–Schild coordinates every ray that falls in approaches the past
horizon with $E_n\propto1/(r-r_+)$ (the [March termination](@ref) rule), so the
redshift cut alone captures and the precise mode sets $\epsilon_h=0$. A thin disk's
observed intensity is $f(r)\,g^3$ with $g\le2\gamma/E_c$ beyond the cut, $\gamma$ the
Lorentz factor of the gas relative to the normal observer, so the omitted intensity
relative to the image maximum is at most $(2\gamma/E_c)^3 f_{\max}/I_{\max}$; at
$E_c=10^6$ this is below $10^{-15}$. A higher cut costs $C\ln10/w$ steps per decade on
captured rays only. On a superposed binary the horizons are approximate: $d$ stalls above
zero while $E_n$ grows, and a march continued beyond $E_n\approx10^2$ loses the ray, so
`march_control` keeps the cut there at most at `CAPTURE_REDSHIFT` whatever the budget.

A volumetric render samples its plasma once per step, so its accuracy is set by $C$ and
not by the integrator (a torus image's flux converges as $C^{-2}$ with either); the
precise mode serves
the products whose accuracy is that of the geodesics: surface images
(`render_crossings`), trajectories, shadows and capture maps.

## Analytic Kerr fast path and lensing bands

For sources on surfaces in Kerr, an exact closed-form tracer (no ODE) inverts
the geodesic with Legendre/Jacobi elliptic integrals in the formulation of
[GrallaLupsasca2020NullGeodesics](@citet) and its lensing companion
[GrallaLupsasca2020Lensing](@citep), following the conventions of Krang.jl
[Chang2024](@citep) and the finite-observer photon-ring machinery of AART
[CardenasAvendano2023](@citep):
conserved quantities $(\lambda,\eta)$ from Bardeen screen coordinates, quartic
radial roots, angular and radial Mino-time integrals, and per-image-order
landing points $(r_s,\phi_s,t_s)$. `critical_curve`
returns the photon-shell boundary; `lensing_band` and `band_grid` build the
nested order-$n$ lensing-band masks used by `photon_ring_layers` to compute
razor-thin photon-ring images and, via the interferometry utilities, visibility
amplitudes. The analytic $r_s$ agrees with the ODE tracer's equatorial crossing
to $6\times10^{-6}$, and ForwardDiff propagates $dr_s/d(a,\theta_o)$ through
the elliptic functions (verified against finite differences).

### Surfaces at any polar angle

With $u=\cos^2\theta$ the polar potential factors as
$\Theta\sin^2\theta=a^2(u_+-u)(u-u_-)$. For an ordinary ray ($\eta>0$,
$u_-<0<u_+$) the antiderivatives
$$\Psi_\theta=\frac{F(\varphi|k)}{a\sqrt{-u_-}},\quad
\Psi_\phi=\frac{\Pi(u_+;\varphi|k)}{a\sqrt{-u_-}},\quad
\Psi_t=-\frac{2u_+}{a\sqrt{-u_-}}\,\frac{E(\varphi|k)-F(\varphi|k)}{2k},$$
with $\sin\varphi=\cos\theta/\sqrt{u_+}$ and $k=u_+/u_-$, satisfy
$d\Psi_\theta/d\theta=-1/\sqrt\Theta$, $d\Psi_\phi/d\theta=-1/(\sin^2\theta\sqrt\Theta)$
and $d\Psi_t/d\theta=-\cos^2\theta/\sqrt\Theta$, and $\Psi$ runs monotonically
between the turning values $\pm\hat G/2$, $\hat G=2K(k)/(a\sqrt{-u_-})$. A ray traced
back from the observer starts toward the pole when $\beta>0$ ($\theta_o<\pi/2$), so
with $s_\beta={\rm sign}\,\beta$ (${\rm sign}\,0=+1$) it meets the surface
$\theta=\theta_s$ after $m$ polar turning points at
$$G_i(m)=m\,\hat G_i-s_\beta\Psi_i(\theta_o)+\nu_\theta\Psi_i(\theta_s),\qquad
\nu_\theta=s_\beta(-1)^m,\qquad i\in\{\theta,\phi,t\},$$
where $\nu_\theta$ is the sign of $p^\theta$ of the photon at emission. A crossing exists
when $\cos^2\theta_s<u_+$ and $G_\theta(m)>0$; at the equator $\Psi(\pi/2)=0$ and
$m=n+H(\beta)$. A vortical ray ($\eta<0$, $0<u_-<u_+$) stays in one hemisphere; with
$\sin^2\psi=(\cos^2\theta-u_-)/(u_+-u_-)$ and $k_v=1-u_+/u_-$,
$\Psi_\theta=F(\psi|k_v)/(a\sqrt{u_-})$,
$\Psi_\phi=\Pi(n_v;\psi|k_v)/((1-u_-)a\sqrt{u_-})$ with $n_v=(u_+-u_-)/(1-u_-)$,
$\Psi_t=\sqrt{u_-}E(\psi|k_v)/a$, and the same formula holds with $\hat G$ the half
oscillation $K(k_v)/(a\sqrt{u_-})$ and $\Psi$ centred on it. The radial inversion of
$G_\theta$ is the sn, cn or sc function of cases II, III and IV
([GrallaLupsasca2020NullGeodesics](@citet) App. B); case IV (no real root) is met by
vortical rays only, since $R(0)=-a^2\eta<0$ forces two real roots when $\eta>0$.

Image orders follow Krang's labelling with every crossing in exactly one order. Let
$m_0=0$ when the first leg of the traced ray reaches the surface and $m_0=1$ otherwise.
A surface at most as inclined as the line of sight ($|\cos\theta_s|\le|\cos\theta_o|$,
the equator among them) has the single crossing $m=n+m_0$ in order $n$; a surface
closer to a pole has the pair $\{n,n+1\}$ when $n\equiv m_0\pmod 2$ (the near and far
sides of a jet cone) and none otherwise; a vortical ray has the pair
$\{2n+m_0,2n+m_0+1\}$. `surface_prep` holds the per-pixel state, `surface_landing`
the crossing after $m$ turning points and `image_crossings` the crossings of order $n$.
At the equator the surface path reproduces `band_landing` bit for bit.

### Observer at infinity

`pixel_prep` and `surface_prep` take `ro = Inf` for an observer at infinity:
$I_0$, $I_\pm$ and the azimuth converge, while $dt/dr=1+2/r+O(r^{-2})$ makes
$I_1\sim\ln r$ and $I_2\sim r$ diverge. The landing time is then the regularized
$\tilde t=\lim_{r_o\to\infty}(t_s+r_o+2\ln r_o)$, the emission time relative to the
observer's retarded time, evaluated in closed form by the connection formula
$\Pi(n;\varphi|k)+\Pi(k/n;\varphi|k)=F(\varphi|k)+\sqrt c\,R_C\big((c-1)(c-k),(c-n)(c-k/n)\big)$,
$c=\csc^2\varphi$, whose $R_C$ term carries the logarithm. The finite-observer
landings approach it as $1/r_o$ in all three radial cases.

### Photon-ring orders

A ray that lands near the critical curve circles the photon shell before leaving: at screen
distance $d$ from the critical point of shell radius $\tilde r$, successive images of one
source radius satisfy $d_{m+1}/d_m\to e^{-\gamma}$, with azimuth lapse $\delta$ and time lapse
$\tau$ per half orbit ([GrallaLupsasca2020Lensing](@citet) Eqs. 68, 70, 72;
[Johnson2020](@citet)),
$$\gamma=\frac{4\tilde r\sqrt{\tilde\chi}}{a\sqrt{-\tilde u_-}}K,\quad
\delta=\frac{2}{\sqrt{-\tilde u_-}}\left[\frac{\tilde r+1}{\tilde r-1}K+\frac{\tilde\lambda}{a}\Pi\right]+2\pi H(\tilde r-\tilde r_0),\quad
\tau=\frac{2}{a\sqrt{-\tilde u_-}}\left[\tilde r^2\frac{\tilde r+3}{\tilde r-1}K-2a^2\tilde u_+\frac{E-K}{2k}\right],$$
$\tilde\chi=1-\Delta(\tilde r)/(\tilde r(\tilde r-1)^2)$, complete integrals of parameter
$k=\tilde u_+/\tilde u_-$ (`photon_shell_exponents`; `critical_point` gives the screen point of
$\tilde r$). The landings of orders 0–7 reproduce them: along the screen ray through a critical
point, $\ln(d_{n-1}/d_n)-\gamma$, the azimuth step minus $2\pi-\delta$ and the time step minus
$\tau$ fall as $e^{-\gamma n}$ to $10^{-6}$–$10^{-5}$ at $n=7$. The case-II time keeps that accuracy
because $\sqrt{R}/(r-r_3)$ is evaluated from the factored potential, exactly zero at the
turning point (the expanded polynomial leaves a residue there that $\sqrt{\cdot}$ lifts to
$\sim\sqrt\varepsilon$ and $1/(r_4-r_3)$ amplifies near the critical curve); every term is
evaluated at the inverted radius, so time and azimuth belong to the point of the geodesic at
that radius. What remains is the conditioning of the near-double roots $r_3\approx r_4$; the landing
radius agrees with Krang's to within $\varepsilon\,|\partial\ln r_s/\partial\ln s|$ along the
screen ray at every order. `lensing_band` finds the band edges of any order by a logarithmic
search toward the critical curve.

### Time-dependent sources

The landing time $t_s$ (zero at the observation, or the retarded time for an observer at
infinity) makes a source $f(t,x,y)$ on the equator observable at time $t_{\rm obs}$ as
$I=f(t_s+t_{\rm obs},r_s\cos\phi_s,r_s\sin\phi_s)\,g^p$. `analytic_movie` computes the
landings of every pixel and order once and each frame as one pass over them; the default
`EquatorialSpot` profile is the Gaussian of AART's flare model on a circular orbit,
$f=\exp(-|x-x_c(t)|^2/w^2)$, $x_c=r_c(\cos(\Omega t+\phi_0),\sin(\Omega t+\phi_0))$.

### Polarization by the Walker–Penrose constant

At a landing the photon momentum is $p_\mu=(-1,\nu_r\sqrt R/\Delta,\nu_\theta\sqrt{\Theta},\lambda)$.
The emitter's rest frame is the zero-angular-momentum frame
$e_{(t)}=(\partial_t+\omega\partial_\phi)/\alpha_L$, $e_{(r)}=\sqrt{\Delta/\Sigma}\,\partial_r$,
$e_{(\phi)}=\sqrt{\Sigma/\mathcal A}\,\partial_\phi/\sin\theta$, $e_{(z)}=-\partial_\theta/\sqrt\Sigma$
($\alpha_L=\sqrt{\Sigma\Delta/\mathcal A}$, $\omega=2ar/\mathcal A$) boosted by the emitter's
velocity; there $p=\omega_{\rm em}(1,\hat k)$, the redshift is $g=1/\omega_{\rm em}$, and the
synchrotron electric vector is $f=(0,\hat k\times b)$ for the rest-frame field $b$
[Narayan2021](@citep). For a field given as a spacetime vector $B^\mu$ the three conditions
$f\cdot p=f\cdot u=f\cdot B=0$ of [CardenasAvendano2023](@citet) select the same direction.
The Walker–Penrose constant
$$\kappa=(r-ia\cos\theta)(\mathcal A_\kappa-i\mathcal B_\kappa),\quad
\mathcal A_\kappa=(p^tf^r-p^rf^t)+a\sin^2\theta(p^rf^\phi-p^\phi f^r),\quad
\mathcal B_\kappa=[(r^2+a^2)(p^\phi f^\theta-p^\theta f^\phi)-a(p^tf^\theta-p^\theta f^t)]\sin\theta$$
is conserved along the ray, invariant under $f\to f+cp$, and
$|\kappa|^2=(f\cdot f)(\eta+(\lambda-a)^2)$ [Himwich2020](@citep). On the screen, with
$\nu=-(\alpha+a\sin\theta_o)$ and $\nu^2+\beta^2=\eta+(\lambda-a)^2$,
$E_\alpha=(\beta\kappa_2-\nu\kappa_1)/(\nu^2+\beta^2)$ and
$E_\beta=(\beta\kappa_1+\nu\kappa_2)/(\nu^2+\beta^2)$ [Gelles2021](@citep), so
$E_\alpha^2+E_\beta^2=f\cdot f$; with north $+\beta$ and east $-\alpha$,
$Q=E_\beta^2-E_\alpha^2$ and $U=-2E_\alpha E_\beta$ scaled to the polarized intensity.
`SynchrotronSurface` is the power-law synchrotron surface of [Narayan2021](@citet)
(observed $I=|\hat k\times b|^{1+\sigma}\min(1/|\cos\zeta|,100)\,\mathcal P(r)\,g^{3+\sigma}$,
fully polarized); `PolarizedDisk` is the AART g-disk with $|P|=I$ along the field direction
fixed by a ZAMO-frame field.

The closed-form EVPA agrees with the numerical polarized march of this package, whose
camera frame is parallel-transported along each RK4 step: the difference falls as
$C^{-4}$ to a floor set by the plane-parallel camera, $2\times10^{-5}$ rad at a camera
distance of $10^5\,M$ and $10^{-8}$ rad (median) at $10^9\,M$. It agrees with Krang's
polarized synchrotron material and AART's polarization model to rounding error (NMSE
of $I,Q,U$ between $10^{-28}$ and $10^{-24}$, EVPA medians of $10^{-15}$ rad), after two
corrections to Krang (the Stokes $I$ weight of its polarized material and the emission
polar sign on far-hemisphere cones) described in the validation record.

## Sampling the meshblock hierarchy

An AthenaK snapshot holds cell averages on blocks of $n_1\times n_2\times n_3$ cells
whose refinement levels differ by at most one between blocks that share a face, an edge
or a corner. The sampled field in a block is the trilinear interpolant of its cells
extended by one layer of ghost cells, built once when the snapshot is loaded
(`Data.fill_ghosts!`), so a point between the outermost cell centre and the block face
blends the outermost cell with the ghost cell beyond the face. The ghost at centre
$\mathbf p$, with the block's cell widths $\mathbf h$, takes the value
$V(\mathbf p,\mathbf h)$ of the mesh on that cell (`Data.mesh_value`):

- a block of the same level covers $\mathbf p$: its cell (copy);
- a finer block covers $\mathbf p$: the mean of $V$ over the children of the cell, the
  volume average of the finer cells inside it (restriction);
- a coarser block, of cell widths $\mathbf H$, covers $\mathbf p$: from the coarse cell
  $c$ that contains $\mathbf p$,
  ```math
  V=u_c+\sum_a\operatorname{minmod}\big(u_c-V(\mathbf x_c-H_a\hat{\mathbf e}_a),\;
  V(\mathbf x_c+H_a\hat{\mathbf e}_a)-u_c\big)\,\frac{p_a-x_{c,a}}{H_a},
  ```
  the limited piecewise-linear prolongation of AthenaK (`ProlongCC`), its neighbours
  taken at the coarse widths by the same rules (where the finer block covers one, its
  restriction);
- $\mathbf p$ lies outside the mesh along axis $a$:
  $V=2V(\mathbf p-sh_a\hat{\mathbf e}_a)-V(\mathbf p-2sh_a\hat{\mathbf e}_a)$, $s$ the
  outward sign, along each such axis in turn (one-sided linear extrapolation; the value
  $V(\mathbf p-sh_a\hat{\mathbf e}_a)$ is held where the mesh is one cell thick).

*Continuity.* A ghost value depends on its centre and widths alone, not on the block
that asks. Two blocks of one level that share a face therefore blend the same eight
nodes at every point of the face, and the sampled field is continuous across it to
rounding, where the face's edges and corners border coarser or finer blocks too. Across a
face between two levels the sides interpolate on different nodes and differ at second
order: continuity there would need the fine side's trace on the face, bilinear between
fine tangential nodes, to equal the coarse side's, bilinear between coarse ones, and
their kinks do not coincide.

*Order.* Copy, restriction (the mean of a linear field over a cell is its centre value),
the limited slope of linear data (the two one-sided differences are equal) and linear
extrapolation are exact for a linear field, so a linear field is sampled exactly
everywhere: the faces, edges and corners between levels and the outer boundary included.
For a smooth field restriction and extrapolation err by $O(h^2)$. The limited slope
departs from the derivative by $O(H)$ only where the one-sided differences change sign,
within a cell of an extremum, where the derivative is itself $O(H)$; times the offset
$|p_a-x_{c,a}|=H_a/4$ the prolonged value errs by $O(H^2)$. Trilinear interpolation of
second-order nodes is second order, so the sampled field is second-order accurate
everywhere (`test/infra/data_ghosts.jl`: a quadratic field on a three-level mesh
converges at second order, the coarse–fine shells included).

*Why the limiter.* The unlimited slope $(u_{c+1}-u_{c-1})/2H$ is second order too and
also a function of position, but at the offset $H/4$ it overshoots wherever the field
changes by a large factor across a coarse cell (the torus surface, a funnel wall):
$u=(1,1,100)$ prolongs to $-11.4$. The minmod slope keeps every prolonged value between
$u_c$ and its neighbours; copy and restriction are averages; the trilinear blend is a
convex combination of its nodes. Inside the mesh the sampled field therefore stays within
the range of the cells around it: no new extrema, and $\rho>0$, $e>0$ wherever the cells
have them. Only the extrapolated ghosts at the outer boundary can leave that range, and a
non-positive density or pressure decodes as vacuum.

A block of $n^3$ cells is stored with $(n+2)^3$ (20% more memory for $32^3$, 10% for
$64^3$); the ghost layers travel with the block data to every device, so threads and
kernels run the same stencil. Imaging, the slow-light windows (each frame carries its own
ghost layers), the uniform resample and the Monte Carlo march all sample this field; the
Monte Carlo emission zones sit at cell centres, where the sample is the cell value.

## Radiative transfer

A single plasma-decode chain converts sampled code-unit
$(\rho,p_{\rm gas},b^2)$ to cgs $(n_e,\Theta_e,B,\sigma,\beta^{-1})$, with the
$R_{\rm high}/R_{\rm low}$ electron-temperature prescription
[Moscibrodzka2016](@citep) and configurable $\sigma$, $\beta$,
$\Theta_e$ emission cuts. The same chain feeds imaging and Monte Carlo.

**Coefficients.** `coefficients` returns the invariant emissivities,
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

**Polarized transport** applies the exact solution of the transfer equation
with constant coefficients per segment in the fluid tetrad (Gram–Schmidt frame
anchored to $u^\mu$, $k^\mu$, $b^\mu$), as the affine operator
$S\mapsto MS+b$, $M=e^{-\mathbf K\Delta s}$, $b=\int_0^{\Delta s}e^{-\mathbf K s}\,ds\;j$
(`segment`). With $\mathbf K=\alpha_I\mathbf 1+\mathbf N$, the eigenvalues of
$\mathbf N$ are $\pm\lambda_1$ and $\pm i\lambda_2$,
$\lambda_{1,2}^2=\big[(\alpha_P^2-\rho_P^2)^2/4+(\boldsymbol\alpha\cdot\boldsymbol\rho)^2\big]^{1/2}\pm(\alpha_P^2-\rho_P^2)/2$,
and [LandiDeglInnocenti1985](@citet)

```math
e^{-\mathbf K s}=e^{-\alpha_I s}\Big[\tfrac12(\cosh\lambda_1 s+\cos\lambda_2 s)\,\mathbf M_1
-\sin\lambda_2 s\,\mathbf M_2-\sinh\lambda_1 s\,\mathbf M_3
+\tfrac12(\cosh\lambda_1 s-\cos\lambda_2 s)\,\mathbf M_4\Big],
```

so $b$ needs the four integrals $C_1\pm S_1=\Delta s\,\varphi\big((\alpha_I\mp\lambda_1)\Delta s\big)$
of $e^{-\alpha_I s}(\cosh,\sinh)\lambda_1 s$ and
$C_2+iS_2=\Delta s\,\varphi\big((\alpha_I-i\lambda_2)\Delta s\big)$ of
$e^{-\alpha_I s}(\cos,\sin)\lambda_2 s$, with $\varphi(z)=(1-e^{-z})/z$:
$b=\big[\tfrac12(C_1+C_2)\mathbf M_1-S_2\mathbf M_2-S_1\mathbf M_3+\tfrac12(C_1-C_2)\mathbf M_4\big]j$.
Their usual closed forms carry $1/(\alpha_I^2-\lambda_1^2)$ and
$1/(\alpha_I^2+\lambda_2^2)$ and are differences of terms of order
$1/(\alpha_I\mp\lambda_1)$ whose difference is of order $\Delta s$, so they keep only a
relative accuracy $\epsilon/[(\alpha_I-\lambda_1)\Delta s]$ of their value ($\epsilon$ the
rounding unit; the cosine pair fares better, $|\alpha_I-i\lambda_2|\ge\alpha_I-\lambda_1$).
The segment uses them only where the slow-mode depth $(\alpha_I-\lambda_1)\Delta s\ge1$
and the relative gap to either pole exceeds $10^{-6}$, and evaluates $\varphi$, regular at
$z=0$, everywhere else. That is the small-absorption limit: as $\alpha\to0$ at fixed
$\rho$, $\lambda_1\to|\boldsymbol\alpha\cdot\boldsymbol\rho|/\rho_P$, $\lambda_2\to\rho_P$,
$C_1\to\Delta s$, $S_1\to0$ and $C_2+iS_2\to[\sin\theta+i(1-\cos\theta)]/\rho_P$
($\theta=\rho_P\Delta s$), the rotation segment below, whereas the closed forms become
$0/0$ (once $\alpha_I\lesssim10^{-154}$ their denominators underflow, as for the far
gas of a power-law or kappa fit). The eigen-decomposition is formed scale-free:
$\lambda_{1,2}$ are homogeneous of degree 1 in $(\boldsymbol\alpha_P,\boldsymbol\rho_P)$ and
$\mathbf M_{2,3,4}$ of degree 0, so both come from the polarized coefficients divided by
the largest of them, and of $\lambda_{1,2}^2$ the root that the formula gives as a
difference is taken as $(\boldsymbol\alpha\cdot\boldsymbol\rho)^2$ over the other
($\lambda_1^2\lambda_2^2=(\boldsymbol\alpha\cdot\boldsymbol\rho)^2$). No square under- or
overflows and neither root cancels; with $\rho_P=10^{10}\alpha_I$ the difference form
loses $\lambda_1$ altogether and the slow mode decays at $\alpha_I$ instead of
$\alpha_I-\lambda_1$. The cases:

- *emission only* ($\alpha_I=\alpha_P=\rho_P=0$): $M=\mathbf 1$, $b=j\,\Delta s$;
- *total absorption* ($\alpha_P=\rho_P=0$): $M=e^{-\alpha_I\Delta s}\mathbf 1$,
  $b=(1-e^{-\alpha_I\Delta s})\,j/\alpha_I$;
- *Faraday rotation and conversion without absorption* ($\alpha_I=\alpha_P=0$):
  $e^{-\mathbf N s}$ rotates $(Q,U,V)$ about $\hat{\boldsymbol\rho}$ through
  $\theta=\rho_P s$, $e^{-\mathbf N s}=\cos\theta\,\mathbf 1+(1-\cos\theta)\,\hat{\boldsymbol\rho}\hat{\boldsymbol\rho}^{\mathsf T}+\sin\theta\,[\hat{\boldsymbol\rho}\times]$,
  and integrating each term over the segment ($\theta=\rho_P\Delta s$) gives
  $b_I=j_I\Delta s$ and
  $\boldsymbol b_P=\frac{\sin\theta}{\rho_P}\boldsymbol j_P+\frac{\theta-\sin\theta}{\rho_P}\hat{\boldsymbol\rho}(\hat{\boldsymbol\rho}\cdot\boldsymbol j_P)+\frac{1-\cos\theta}{\rho_P}\hat{\boldsymbol\rho}\times\boldsymbol j_P$;
- *the dichroism-cone boundary* $|\alpha_P|=\alpha_I$ with $\rho=0$ (or
  $\boldsymbol\rho\parallel\boldsymbol\alpha$): $\lambda_1=\alpha_I$, the mode polarized
  against $\hat{\boldsymbol\alpha}$ is unabsorbed and gathers its emission linearly,
  $C_1-S_1=\Delta s$; the $\varphi$ form is this limit of the general solution;
- *nilpotent coupling* ($\lambda_1=\lambda_2=0$, i.e. $|\alpha_P|=|\rho_P|$ and
  $\boldsymbol\alpha\perp\boldsymbol\rho$): $\mathbf N^4=0$, so
  $e^{-\mathbf K s}=e^{-\alpha_I s}(\mathbf 1-s\mathbf N+s^2\mathbf N^2/2-s^3\mathbf N^3/6)$
  and $b=\sum_k(-\mathbf N)^k g_k\,j$, $g_k=\int_0^{\Delta s}s^k e^{-\alpha_I s}\,ds/k!$.

A segment is saturated ($M=0$, $b=\mathbf K^{-1}j$) when its slowest mode, at the
rate $\alpha_I-\lambda_1$, is deeper than 100; a deeper segment with a shallower
slow mode takes the $\varphi$ form, whose decay factors stay finite. Every case
matches the 256-bit solution $\exp\!\big(\Delta s\,[\,-\mathbf K\ \ j;\ 0\ \ 0\,]\big)$
to rounding error (`test/physics/transfer_segments.jl`). Instead of transporting the coherency tensor
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

**Admissibility.** In production every segment operator projects its
coefficients onto the physical cone before it is formed: when
$|\alpha_P|=(\alpha_Q^2+\alpha_V^2)^{1/2}$ exceeds $\alpha_I$ the polarized
absorptivities are rescaled onto $|\alpha_P|=\alpha_I$ (onto $\alpha_P=0$ when
$\alpha_I=0$), and when $|j_P|$ exceeds $j_I$ the polarized emissivities are
rescaled onto $|j_P|=j_I$, keeping $j_I$ and the polarization direction.
`segment(...; project = false)` solves the transfer equation for the coefficients as
given, outside the cone too; the one-zone tests of [Prather2023](@citet), whose
coefficients lie outside it, run through that path. The thermal fits of [Marszewski2021](@citet) give $|j_P|>j_I$ at small
pitch angles; without the rescale the observed Stokes vector would depend on
whether such an emitter lies in front of or behind brighter gas, because the
accumulated vector is clamped to $|P|\le I$ (`stokes_admissible`) only on
partial sums (`test/physics/transfer_segments.jl`: a thermal slab and a
field-aligned screen give the same $V/I$ in either order).

**One physics source.** The plasma decode, the coefficient functions (fed by the
`isbits` parameter bundle `TransferParams`), the fluid tetrad, the segment
operators and the reverse march (`march`, in the mode `StokesI` or `IQUV`) are
single functions called by the threaded executor and by the device kernel. A march
control supplies the geodesic: `SpacetimeMarch` (the fixed-control march on any
spacetime, `Float64` or `Float32` lane) or `StoredPath` (the states of `trace`).
The two backends run the same program, so on identical trajectories they agree
to the floating-point contraction of the two targets.

## One kernel, two backends

Every product is a per-item program, one item per pixel, block or primary photon,
run by an executor: `ThreadedExecutor` calls it from Julia threads, `PoolExecutor`
runs it through the single kernel of `Execution` on the lanes of a `DevicePool`.
The program is the same code on both: the reverse march of
[Radiative transfer](@ref) advances the geodesic, the two transported frame vectors
and the Stokes accumulation in one loop, storing no trajectory, so per-ray state is
$O(1)$ regardless of path length. Metric derivatives come from ForwardDiff dual
numbers evaluated inside the march; no Christoffel tensor is materialized. The same
mechanism seeds a dual on a physical scalar (the density mass scale, $R_{\rm high}$)
through the transfer accumulation, so `sensitivity` returns the image and its exact
parameter derivative in one pass (the differentiable-GRRT construction of Jipole
[NaetheMotta2025](@citep)). The fluid is sampled on the AthenaK meshblock hierarchy on
both backends with the same trilinear stencil on ghost-extended blocks
([Sampling the meshblock hierarchy](@ref)); a `UniformGrid` resample is the cheaper
alternative, at the percent level in flux and with EVPA changes on Faraday-thick decks.

## Multi-device execution

Every device run uses a persistent `DevicePool`, one per device list,
created on first use: two worker tasks per device, each with its own stream,
device buffers and pinned staging buffers. Work is a queue of chunks; each
worker pulls the next chunk when it finishes the last, so a slower or shared
device takes fewer. The design rules:

- **Ownership, not partition, fixes the result.** Every ray writes only its own
  pixel, every AMR block only its own tile, and every Monte Carlo photon draws
  from a stream named by its genealogy, not by the chunk that carries it. The
  output is therefore bitwise identical for any device count, chunk size and
  schedule (`test/pipeline/products_backend.jl`, `test/infra/execution_pool.jl`).
- **Rays are generated on the device.** A chunk uploads only its pixel or block
  indices (4 bytes per item); the device builds each camera ray with the same
  `camera_ray` function as the host (`test/infra/execution_ray_sources.jl`:
  device rays `isequal` host rays).
- **Longest work first.** On Kerr, `prepare` queues $16\times16$-pixel tiles
  in order of predicted cost — distance to the critical curve (`cost_order`) — so the
  rays that wind near the photon ring start first instead of ending the render;
  `march_sort_order(counts)` orders by measured trip counts from a previous frame,
  longest first, through the `queue` keyword of `prepare`. With `MixedLanes` the
  near-ring pixels that need `Float64` enter the same queue ahead of the `Float32`
  pixels (`test/infra/execution_schedule.jl`).
- **Streaming adaptive refinement.** Each refinement block is a work item; the
  worker that receives a block's Stokes values paints it, runs the refinement
  test and queues its four children at once, so refined blocks march while the
  rest of their parent level is in flight. Mosaic, level map and ray count are
  independent of the executor and the completion order (`test/pipeline/products_amr.jl`).
- **Upload once.** A snapshot or uniform grid is uploaded once per pool and
  reused by later calls on the same object; Monte Carlo host tables are built
  once per call and uploaded to all devices concurrently.

The pool needs at least one more Julia thread than devices (`-t 3` for two
GPUs) so that the workers of different devices run concurrently. On the
KernelAbstractions CPU backend, `HostDevice` entries act as logical devices, which is
how the invariance tests run without a GPU. The measured throughput on one and two
A100s is on [Benchmarks](@ref); the execution model and its knobs are in
[Performance](@ref).

## Adaptive image-plane refinement

`render_amr` implements the adaptive ray tracing of [White2022](@citet): the image is
tiled into blocks; the indicators of `Analysis` (relative gradient, relative
Laplacian, absolute gradient, or any callable) flag blocks, which spawn four children
at doubled pixel density up to `max_level`. All levels run the plan's imaging program;
the blocks stream through the executor's worklist ([Multi-device execution](@ref)),
and both backends make identical refinement decisions. Refinement typically saves
$\sim4\times$ in rays over a uniform image at the finest pixel.

## Monte Carlo spectral synthesis

The Monte Carlo module implements the grmonty scheme [Dolence2009](@citep):
superphoton packets sample the synchrotron emissivity zone-by-zone with
frequency and pitch-angle rejection sampling and a log-frequency weight table;
packets carry covariant momentum and decay by Kirchhoff absorption along
geodesics. Compton scattering uses the thermally averaged ("hot") Klein–Nishina
cross section tabulated on a log-log $(\varepsilon,\Theta_e)$ grid, scattering
electrons drawn from the relativistic Maxwellian by the
[Canfield1987](@citet) sampler and exact Klein–Nishina/Thomson rest-frame
kinematics; the scattering events of a step follow the biased
exponential transform [Kahn1950](@citet) in the exact form below. Spectra
are binned in (exit angle, $\log\nu$) with grmonty normalization; the
optically thin synchrotron sphere matches its closed form within its per-bin
standard errors, and GRMHD spectra show the gravitationally redshifted
synchrotron bump. On GPU, multi-order scattering is layered rather than
recursive: the host relaunches once per scattering generation, and a launch
whose child buffer overflows is rerun at the exact size, so no child is dropped.
The march reads any spacetime through `proxy_distance`: the distance above the
horizon proxy sets the step (grmonty's `stepsize`) and the capture, the radius
the escape, so the same tracker runs on Kerr, on a binary at rest or in motion
(the nearest hole at the photon's time) and on an ADM grid.

**The angle-integrated emissivity.** With $X=\nu/\nu_s$,
$\nu_s=\tfrac29\nu_c\Theta_e^2\sin\theta$, $\nu_c=eB/2\pi m_ec$, the thermal fit of
`thermal_synchrotron` is
$j_\nu=(n_ee^2\nu_c/c)(\sqrt2\pi/27)\sin\theta\,G(X)$ with
$G(X)=(X^{1/2}+2^{11/12}X^{1/6})^2e^{-X^{1/3}}$. It depends on $\nu$ and $\Theta_e$
only through $K=\nu/(\tfrac29\nu_c\Theta_e^2)=X\sin\theta$, so

```math
\int j_\nu\,d\Omega=4\pi\!\int_0^{\pi/2}\! j_\nu\sin\theta\,d\theta
=\frac{4\pi n_e e^2\nu_c}{c}\,\frac{\sqrt2\,\pi}{27}\,F(K),\qquad
F(K)=\int_0^{\pi/2}\sin^2\theta\;G\!\left(\frac{K}{\sin\theta}\right)d\theta ,
```

exactly. `EmissivityTable` holds $\ln F$ and $d\ln F/d\ln K$ (the derivative under the
integral) on 6,324 nodes, $\Delta\ln K\le0.02$ from $K=10^{-30}$ to $10^3$ and
$\le0.005$ from $10^3$ to $3\times10^8$ (where the curvature of $\ln F$ grows as
$K^{1/3}$), from a 2,048-interval Simpson rule; `int_jnu` interpolates by cubic Hermite
polynomials, continues $\ln F$ along the first slope below the table ($F\propto K^{1/3}$)
and returns 0 above it ($F<10^{-280}$). The table agrees with the 512-interval rule of
`int_jnu_simpson` to $3\times10^{-9}$ and with an 8,192-interval rule to $7\times10^{-11}$
over the whole range, and costs one logarithm and one exponential per evaluation. It is
built once, when the package compiles, and generation reads the same function as the
zone tables, so the frequency acceptance and its envelope stay consistent.

**Zones, weights and counts.** A zone of proper volume $dV$ emits
$d\dot N/d\ln\nu=(dV/h)\int j_\nu d\Omega$ photons per second per unit $\ln\nu$. The
weight table $w_m=\sum_z\int j_\nu d\Omega(\nu_m)\,dV_z/(hN_s')$ on $n+1$ nodes over
$[\nu_{\min},\nu_{\max}]$, $N_s'=N_s/\ln(\nu_{\max}/\nu_{\min})$, makes the generated
density uniform in $\ln\nu$; zone $z$ receives
$N_z=(dV_z/h)\,\Delta\ln\nu\sum_m c_m\int j_\nu d\Omega(\nu_m)/w_m$ primaries (rounded
stochastically), with the trapezoid weights $c_0=c_n=\tfrac12$, $c_m=1$ inside. Then
$\sum_zN_z=N_s'\Delta\ln\nu\sum_mc_m=N_s$: the expected number of primaries is the
target, and the emitted power carries no excess of one end node. Both passes over the
zones (the exact node sums of the weight table, then the counts) stream: the
zone × frequency matrix is never stored. On a block mesh every cell centre that its
block owns is a zone. A zone below the horizon proxy is dropped, since a photon emitted
there is captured before its first step; this leaves every escaping estimate unchanged.
For quick runs `stride = s` cuts each block into groups of $s^3$ cells and draws one of
a group's $n_\mathrm{own}$ owned cells uniformly, on a stream of the seed, with the
volume $n_\mathrm{own}\,\Delta V$ of the drawn cell: for any $f$,
$\mathrm E\big[n_\mathrm{own}f(c)\Delta V(c)\big]=\sum_{c\in\text{owned}}f(c)\Delta V(c)$,
so the strided zones are an unbiased quadrature of the stride-1 sum (a fixed cell per
group is not: where the emissivity changes by decades within a group it moves the
emission between radii).

**Scattering within a step.** Over a step of absorption depth $a=\Delta\tau_a$ and
scattering depth $s=\Delta\tau_s$ (trapezoid, rates uniform in the fraction $f$ of the
step), $\Delta\tau=a+s$, a photon of weight $w$ scatters at $f$ with expected weight
density $p(f)=w\,s\,e^{-\Delta\tau f}$ and leaves with $w\,e^{-\Delta\tau}$. The events of
the step are a Poisson process of mean $bs$ for a bias $b\ge1$, at fractions drawn
independently from $q(f)=\Delta\tau e^{-\Delta\tau f}/(1-e^{-\Delta\tau})$, each spawning a
child of weight $W=w(1-e^{-\Delta\tau})/(b\Delta\tau)$. The expected child weight density
is

```math
b s\cdot W\cdot q(f)=w\,s\,e^{-\Delta\tau f}=p(f)
```

for every $b$, $a$ and $s$, and $W\le w/b$. The parent decays deterministically by
$e^{-\Delta\tau}$. The events are drawn as the points of a rate-$bs$ Poisson process on
the CDF coordinate $u=(1-e^{-\Delta\tau f})/(1-e^{-\Delta\tau})\in[0,1)$, cumulated
exponential gaps on the photon's stream, so a step without an event costs one uniform.
The ledger of a march records, per step, the scattered weight
$w s(1-e^{-\Delta\tau})/\Delta\tau$ (the expectation of the spawned weight) and the
absorbed power $wEa(1-e^{-\Delta\tau})/\Delta\tau$; their sum with the final weight closes
the photon's budget exactly. An event whose site is outside the plasma spawns nothing
and its weight is counted (the trapezoid depth of a step that leaves the plasma), as is
the scattered weight of the order `max_scatt`, which may not spawn.

**The importance bias.** Let $b$ be any positive function of what is known before the
step's draws: the photon (weight, energy, order, history) and the step's samples.
Conditioned on that state, the expected child density is $p(f)$ whatever $b$, and the
parent's weight does not depend on $b$ at all; by the tower property every expectation
linear in the escaping photons, each bin of $\nu L_\nu$ included, is independent of the
rule. The default `StateBias` is
$b=\mathrm{clamp}(b_0A(\Theta_e)\,w/w_\mathrm{ref},1,b_\mathrm{max})$, with
$A(\Theta_e)=1+4\Theta_e+16\Theta_e^2$ (the cold and hot limits of the mean energy gain
$1+4\Theta_eK_3/K_2$ of one Thomson scattering, at most 15 % above it), $\Theta_e$ the
larger end temperature of the step and $w_\mathrm{ref}$ the mean primary weight. A child
then carries about $w_\mathrm{ref}/(b_0A)$ photons and the energy weight
$w_\mathrm{ref}E/b_0$ whatever the gas that scattered it, so the rare scatterings of
heavy seed photons off hot electrons, which feed the highest frequencies, are split into
many light children rather than a few heavy ones; $b_\mathrm{max}$ bounds the events per
step. With `split` $>1$, an event whose uncapped bias exceeds $b_\mathrm{max}$ spawns
$m$ children of weight $W/m$ with independent kicks, which keeps the expected child
density by linearity.

**The estimator and its variance.** A bin collects
$\nu L_\nu=(m_ec^2/\Delta\ln\nu)(4\pi/\Delta\Omega)\sum_jw_jE_j$. Its variance is estimated
by $(m_ec^2/\Delta\ln\nu)^2(4\pi/\Delta\Omega)^2\sum_j(w_jE_j)^2$, the variance of a
compound Poisson sum, accumulated exactly beside $\sum wE$ (`nuLnu_sigma`); the
effective number of photons $N_\mathrm{eff}=(\sum wE)^2/\sum(wE)^2$ sets the relative
error $N_\mathrm{eff}^{-1/2}$. Children of one parent that land in one bin are
correlated, which this photon-level estimate does not see; seed-to-seed scatter checks
it.

**Photons certain to be captured.** On Kerr, the radial potential of a photon moving
inward inside every spherical photon orbit has no root above the horizon
[Teo2003, GrallaLupsasca2020NullGeodesics](@citep), so a photon that may spawn no more children stops, captured,
once a step moves it inward below the prograde equatorial photon orbit less 0.02 M. No
photon that would escape is stopped, and the bins are bitwise those of the full march.

**Streams keyed by ancestry.** Randomness is the counter-based
Philox4x32-10 generator [Salmon2011](@citep). Each photon's stream is named by
its genealogy: a 128-bit identity
$I=g+\sum_{j=1}^{k}e_j\,2^{G+(j-1)E}+k\,2^{124}$, with $g<2^G$ the primary's
global index, $e_j<2^E$ the ordinal of the scattering event of the layer-$j$ ancestor
along its parent's march, and $k\le15$ the scattering layer, carried in Philox key word
2 and counter words 2–4 (the draw index in counter word 1, the seed in key word 1).
`widest_layout` sizes $G$ from the primary count and gives the event field the widest
width that fits, $E=\min(32,\lfloor(124-G)/k_{\max}\rfloor)$; an event beyond it spawns
nothing and is counted. Distinct genealogies map to disjoint bit fields and hence
distinct Philox inputs for every draw; identities with bit 123 set and $k=0$ name the
host streams of a run (the cell drawn in a strided zone). A child's kick and its march
draw from its own stream, so the parent's stream carries only its own events. A primary
with $g<2^{32}$ has the stream `MCRng(seed, g)`. No stream depends on the thread,
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
(`test/physics/numerics_exactsum.jl`, `test/pipeline/products_montecarlo.jl`,
`test/pipeline/products_backend.jl`), and the device kernel on the
KernelAbstractions CPU backend reproduces the threaded run bit for bit.

## Deviations from reference codes

Four deliberate, oracle-verified departures from Blacklight
[White2022](@citep) and ipole [MoscibrodzkaGammie2018](@citep):

1. **No dedicated $\rho=0$ coupling branch.** Blacklight's
   absorption-without-rotation step, which it also applies to both halves of its
   split step, damps the emission terms of $I$ at the polarized rate and carries a
   sign error in one polarized term (reported to the developers together with the
   misplaced $(U,V)$ entries of its $\mathbf M_2$, $\mathbf M_3$). BlackLightPlus
   routes $\rho_P=0$ through the general solution above, whose limit it is.
2. **Transported camera frame instead of the coherency tensor.** ipole
   parallel-transports the coherency tensor $N^{\mu\nu}$ between segments;
   BlackLightPlus transports the camera readout frame and applies the exact
   segment operator rotated into it ([Radiative transfer](@ref)). Against a
   Strang-split coherency transport at equal step density the EVPA error is
   five to eight times smaller.
3. **Degenerate eigenstructures take their exact limits** (the dichroism-cone
   boundary, nilpotent coupling, rotation without absorption) in place of an
   approximate fallback.
4. **Interpolation across meshblock faces.** Blacklight interpolates within a block and
   extrapolates linearly from the two outermost cells in the outer half cell (its
   default); with `simulation_block_interp` it takes the anchors beyond a face from the
   neighbouring blocks, but on the high side of each axis it forms the blend fraction
   from `x1v(b, i + 1)` one past the block, the first cell centre of the next block in
   storage order. BlackLightPlus interpolates across every face through its ghost layers
   ([Sampling the meshblock hierarchy](@ref)). On the AthenaK torus deck of the
   GRMHD-snapshot comparison ($i=163^\circ$, 230 GHz) Blacklight with that coordinate
   taken at the face, $2x^1_f(b,i+1)-x^1_v(b,i)$, agrees with BlackLightPlus to NMSE
   $9.7\times10^{-6}$, $2.1\times10^{-5}$, $2.1\times10^{-5}$, $5.5\times10^{-5}$ in
   $I$, $Q$, $U$, $V$ (net linear polarization 3.352% and 3.349%); its in-block
   extrapolation differs by $3.8\times10^{-4}$ in $I$ (3.246%).

The images are cross-validated against RAPTOR [Bronzwaer2018](@citep) on
analytic scenes and binary-black-hole GRMHD snapshots, and against Blacklight
[White2022](@citep) on a single-hole torus; protocol and results are on
[Validation](@ref).

## Works cited

Full bibliographic details, DOIs, and arXiv identifiers for everything cited
above are on the [References](@ref "References") page.
