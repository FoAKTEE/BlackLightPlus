# Code-description appendix material — BlackLightPlus

Material for a code appendix of the method paper (arXiv id of the structural model:
2609.26298). `src/` is the ground truth; every statement below is tied to `file:line` at the
`paper` branch head. Notation follows `results/blp/paper_arxiv-2609.26298/conventions.md`:
signature $(-,+,+,+)$, $G=c=M=1$, CKS coordinates, received wave covector $k_\mu$ with
$k_t=-1$ at the camera, ray state $p_\mu=-k_\mu$, invariant Stokes vector
$\Svec=(I_\nu,Q_\nu,U_\nu,V_\nu)/\nu^3$, invariant coefficients $j_A=j_{\nu,A}/\nu^2$,
$\alpha_A=\nu\alpha_{\nu,A}$, $\rho_A=\nu\rho_{\nu,A}$, Stokes in $(e_{(2)},-e_{(1)})$, IEEE/IAU.
Citations are bib keys of `paper/bibliography.bib`; keys not yet there are marked
**[new]** and listed in §6.

Reference texts compared: Blacklight paper (ref-paper/arxiv-2203.15963) and code
(ref-code/c-white-blacklight @ a7c7215); ipole (arxiv-1712.03057); grtrans (arxiv-1602.03184);
grmonty (arxiv-0909.0708); GPUmonty (arxiv-2602.13198); RAPTOR I/II (arxiv-1801.10452,
arxiv-2007.03045) and code (ref-code/jordydavelaar-raptor @ b91e937, fork `blp-bbh`);
Mahakala (arxiv-2304.03804); AART (arxiv-2211.07469); Krang (doi-10.21105_joss.07273);
Gralla & Lupsasca (arxiv-1910.12881, arxiv-1910.12873); Jipole (arxiv-2509.07065);
Pandya+2016 (arxiv-1602.08749); Marszewski+2021 (arxiv-2108.10359).

---

## 1. Proposed appendix structure

| Section | Content | Components (§4) |
|---|---|---|
| A.1 Code organisation | module table, two design rules (one implementation per primitive; pure, allocation-free, `isbits` march bodies), data flow of the three pipelines | §2, §3 |
| A.2 Spacetimes and metric derivatives | CKS Kerr closed form, superposed binary, user metric, ADM grid with $C^1$ Catmull–Rom, spatial dual seeding | C1–C3 |
| A.3 Geodesic integration and termination | Hamiltonian RHS, fixed-control RK4 $h=(r-r_{\rm h})/C$, adaptive Vern7 path, ray reversal, `march_stop` | C4–C5 |
| A.4 Camera | plane-parallel screen, received covector, pixel centres, device ray generation | C6 |
| A.5 Analytic Kerr tracer | $(\lambda,\eta)$, radial roots, Mino-time inversion, image orders, lensing bands, $g$-disk source, visibility amplitudes | C7–C9 |
| A.6 Fluid input and plasma decode | AthenaK readers, leaf locate, trilinear sample, four-vectors, units, $R(\beta)$, cuts | C10–C11 |
| A.7 Transfer coefficients | thermal / power-law / kappa, basis and sign relative to grtrans, admissibility rescale | C12 |
| A.8 Fluid tetrad and polarized segment operator | Gram–Schmidt tetrad and its fallback, closed-form $\exp(-\Kmat\,\Delta\aff)$ branches, saturation | C13–C14 |
| A.9 Reverse march and transported readout frame | $G_n$ chain, $R_n$, frame transport RHS, stored-path vs fixed-control drivers, unpolarized march | C15–C16 |
| A.10 Time-dependent sources | slow light, snapshot windows | C17 |
| A.11 Image-plane adaptive refinement | indicators, block tree, streaming driver | C18 |
| A.12 GPU kernels | fused kernels, native AMR sampling, generic spacetime kernels, analytic kernel, dual-number derivative, mixed precision | C19–C20 |
| A.13 Multi-device runtime | pool, lanes, queue, cost order, determinism | C21 |
| A.14 Monte Carlo | superphoton generation, tracking, scattering with bias, hot cross section, Philox stream keys, exact sums, estimator, validation sphere | C22–C27 |
| A.15 Cross-code comparison build | RAPTOR fork switches and upstream defects | C28 |

KPolaris (the structural model) puts its algorithm in the main text and uses appendices for
coefficient formulae; a code appendix here carries the module table (§2), one pseudocode block
per pipeline (§3), and one subsection per row above, each ending with a "relation to prior
codes" paragraph (the "Heritage and changes" item of every component below).

---

## 2. Module table

Line counts from `wc -l` at the branch head. "Device" = compiled into KernelAbstractions
kernels (CUDA ordinals or `HostDevice` logical devices on the KA CPU backend). Files marked
"GPU driver" are not included from `src/BlackLightPlus.jl` and are loaded explicitly with CUDA.

| File | Role | Lines | Device |
|---|---|--:|---|
| `src/BlackLightPlus.jl` | module root, public API | 111 | — |
| `src/constants.jl` | cgs constants (values as Blacklight `constants.hpp`/ipole) | 21 | yes (constants) |
| `src/inputs/parameters.jl` | `CameraConfig`, `PlasmaConfig`, `RunConfig` | 65 | — |
| `src/geodesics/metrics.jl` | CKS Kerr closed form `_ks_term`, $g$, $g^{-1}$, index ops, AD gradients, Christoffels | 239 | yes |
| `src/geodesics/spacetimes.jl` | `SuperposedKerrSchild`, binary march control | 257 | yes |
| `src/geodesics/numerical_spacetime.jl` | `ADMGridSpacetime`, Catmull–Rom, ADM assembly | 358 | yes |
| `src/geodesics/integrator.jl` | Hamiltonian RHS, frame-transport RHS, RK4, Vern7 driver, `march_stop`, sampled tracer | 410 | yes (RHS, RK4, stop) |
| `src/geodesics/camera.jl` | plane-parallel camera, `camera_ray` | 85 | yes (`camera_ray`) |
| `src/geodesics/kerr_orbits.jl` | ISCO, photon orbit, Keplerian $\Omega$, radial potential | 89 | yes |
| `src/geodesics/analytic_kerr.jl` | elliptic-integral equatorial tracer | 466 | yes |
| `src/geodesics/lensing_bands.jl` | critical curve, apparent horizon, lensing bands, band grids | 350 | host |
| `src/geodesics/gpu_tracer.jl` | device geodesic tracer with stored samples | 149 | yes |
| `src/simulation/athenak_reader.jl` | `.athdf` reader, leaf table, level hash, trilinear sample, units | 416 | host (sampler ported to device) |
| `src/simulation/athenak_z4c.jl` | `.bin` reader, Z4c→ADM, resample to ADM grid | 308 | host |
| `src/simulation/primitives.jl` | $u^\mu$, $b^\mu$ from AthenaK primitives | 50 | yes |
| `src/simulation/time_series.jl` | `SnapshotSeries`, slow-light blend | 217 | host |
| `src/radiation/coefficients.jl` | plasma decode, $R(\beta)$, cuts, thermal/power-law/kappa coefficients, host precompute | 777 | yes (except precompute) |
| `src/radiation/tetrads.jl` | fluid tetrad | 86 | yes |
| `src/radiation/transfer.jl` | polarized and unpolarized segment operators, admissibility | 188 | yes |
| `src/radiation/reverse_march.jl` | `KerrMarch`/`SpacetimeMarch`/`StoredPath`, reverse marches | 294 | yes |
| `src/radiation/integrator.jl` | `eight_coefficients`, CPU march wrappers | 179 | yes (coefficients) |
| `src/radiation/equatorial_source.jl` | AART $g$-disk redshift and Johnson-SU profile | 206 | yes |
| `src/sources/*.jl` | Polish doughnut, Page–Thorne disk, hot spot (analytic plasma) | 591 | host |
| `src/render.jl` | two-phase CPU driver, analytic render, photon-ring layers | 280 | host |
| `src/amr.jl` | image-plane adaptive refinement | 268 | host (device marcher pluggable) |
| `src/outputs/*.jl` | `.npy` writer, Radon cut, visibility amplitude | 178 | host |
| `src/gpu_metric.jl` | shared device metric helpers | 22 | GPU driver |
| `src/gpu_pool.jl` | `DevicePool`, lanes, queue, staging | 481 | GPU driver |
| `src/gpu_camera.jl` | device ray generation | 118 | GPU driver |
| `src/gpu_image.jl` | fused unpolarized kernel (uniform grid), cost order, mixed precision, slow light | 689 | GPU driver |
| `src/gpu_pol.jl` | fused polarized kernel, stiffness telemetry | 345 | GPU driver |
| `src/gpu_amr.jl` | native-AMR device sampler and kernels, streaming AMR marcher | 696 | GPU driver |
| `src/gpu_spacetime.jl` | kernels for binary / ADM spacetimes, shadow tracer | 331 | GPU driver |
| `src/gpu_analytic.jl` | analytic-tracer kernel | 130 | GPU driver |
| `src/gpu_image_ad.jl` | dual-number parameter derivative of the image | 92 | GPU driver |
| `src/montecarlo/*.jl` (13 files) | Philox RNG, exact sums, emission tables, generation, tracking, scattering, hot cross section, spectrum, sphere and GRMHD models, CPU and device drivers | 3,652 | sphere model on device; GRMHD model CPU |

---

## 3. Data flow

### 3.1 CPU imaging (`render`, src/render.jl:86)

```
RunConfig ─► _pipeline_setup (render.jl:26)
   camera_rays(st, cam)            # C6: one state y=[x^μ; p_μ] per pixel, p_t = +1
   trace_rays_sampled(st, rays; C) # C4: RK4, h=(r-r_h)/C, march_stop; states stored source→camera
   sampler = sample_primitives(snap, ·)   # C10: leaf locate + trilinear
 _prepared_scc (render.jl:65)      # C11/C12: units, R(β), host precompute of nonthermal constants
 for each ray (Threads.@threads):
   polarized:   reverse_polarized_march(ys[end], sample, gp, StoredPath(st, ys, dls), n)   # C15
   unpolarized: reverse_unpolarized_march(...)                                             # C16
 → (I, Q, U, V) as npix×npix, rows β, columns α
```

### 3.2 Fused GPU imaging (`render_gpu_fused` gpu_image.jl:381, `render_gpu_polarized` gpu_pol.jl:146, `render_gpu_amr` gpu_amr.jl:362)

```
host: upload fluid once per pool (uniform grid or AMR blocks + leaf grid)      # C19
      items = cost_order(...)  (or caller order)                              # C21
pool lanes pull chunks:  rays_on_device!  (camera_ray on device, 4 B/pixel)   # C6, C21
kernel, one thread per ray:
   reverse_*_march(y0, sample_on_device, gp, KerrMarch(a, r_h, C, escape, R_em), nmax)
   # geodesic step, frame transport, coefficient sample, segment operator,
   # accumulation — same Julia functions as the CPU march
write out[ray index]  →  copy back per chunk
```

### 3.3 Monte Carlo (`mc_spectrum_grmhd` mc_cpu.jl:535; `mc_spectrum`/`mc_spectrum_scatter` mc_cpu.jl:25/102; `mc_spectrum_gpu` mc_gpu.jl:304/516)

```
zones = emission_zones(model)                       # one EmissionZone per emitting leaf cell
wgt   = build_weight_table(zones, N_s)              # w(ν) on a log-ν grid
for zone: (N_z, dnmax) = zone_nsuperphotons(zone, wgt)
for primary gid (stream mc_primary_rng(seed, gid)):
   sph = sample_zone_photon(zone, wgt, dnmax, rng)  # ν, pitch angle, tetrad → p_μ
   track (absorption; optional biased scattering spawning children with mc_child_rng)
   record into ExactSpectrum (xsum_add!: 68 limbs per bin)
round once: Spectrum(ExactSpectrum) → νL_ν
```

---

## 4. Components

Each component: **Implementation** (file:function:line) · **Algorithm** · **Data** ·
**CPU/GPU** · **Heritage and changes** · **Cite**.

### C1. Kerr in CKS and metric contractions

- **Implementation.** `_ks_term` metrics.jl:53; `metric`/`inverse_metric` :104/:110;
  `lower_index`/`raise_index`/`raise_and_norm` :176/:180/:196; `horizon_radius` :102.
- **Algorithm.** $r^2=\tfrac12[\varrho^2-a^2+\sqrt{(\varrho^2-a^2)^2+4a^2z^2}]$ (computed with
  `hypot`), $f=2r^3/(r^4+a^2z^2)$, $l_\mu=(1,\,(rx+ay)/(r^2+a^2),\,(ry-ax)/(r^2+a^2),\,z/r)$,
  $g_{\mu\nu}=\eta_{\mu\nu}+f\,l_\mu l_\nu$, $g^{\mu\nu}=\eta^{\mu\nu}-f\,l^\mu l^\nu$ (exact because
  $l$ is null for both metrics). On the open disk $x^2+y^2<a^2$, $z=0$ the $0/0$ is removed
  ($f=0$); the ring $x^2+y^2=a^2$ keeps its NaN. `raise_and_norm` returns $g^{\mu\nu}p_\nu$ and
  $g^{\mu\nu}p_\mu p_\nu$ without forming a matrix.
- **CPU/GPU.** Same functions on both; the device re-seeds the same closed form
  (gpu_metric.jl).
- **Heritage and changes.** Blacklight hand-codes metric derivatives per spacetime
  (geodesic_geometry.cpp); here derivatives come from AD (C2). RAPTOR's CKS metric carries
  regularisers $1/(\Sigma+10^{-3})$ and $l_z=z/(r+10^{-3})$ in $g_{\mu\nu}$ but not in
  $g^{\mu\nu}$ (defect D1, BLP_CHANGES.md), giving $g\,g^{-1}\neq\mathbf 1$ at $10^{-4}$; no
  regulariser is used here.
- **Cite.** White2022, Bronzwaer2018 (for D1), Zhang2026KPolaris (CKS in KPolaris).

### C2. Metric derivatives by forward-mode AD; other analytic spacetimes

- **Implementation.** `_seed_spatial` integrator.jl:213; `geodesic_rhs` (stationary)
  :157; `inverse_metric_gradient` metrics.jl:142/:150; `metric_gradient` :210;
  `christoffel` :226 (diagnostic, not on the march path); `UserMetric` metrics.jl:31;
  `SuperposedKerrSchild` spacetimes.jl:27 with `_superposed_gcov`/`_superposed_gcon` :61/:68.
- **Algorithm.** The position is promoted to dual numbers with three spatial partials
  ($\partial_t g\equiv0$ for stationary metrics, so no time partial). One call of
  `raise_and_norm` at the dual position returns $g^{\mu\nu}p_\nu$ (value lane) and
  $\partial_i(g^{\alpha\beta}p_\alpha p_\beta)$ (partials), from which $\dot p_i=-\tfrac12\partial_i
  g^{\alpha\beta}p_\alpha p_\beta$. Neither $\partial g$ nor $\Gamma$ is stored. The binary uses the
  $4\times4$ inverse of $\eta+\sum_A f_A l^A l^A$ and differentiates through it; with one hole of
  zero mass it reduces to single Kerr bitwise. `UserMetric` takes any closure and keeps the
  $\dot p_t$ term (time-dependent closures allowed, integrator.jl:226).
- **CPU/GPU.** Identical; dual numbers live in registers inside kernels.
- **Heritage and changes.** Blacklight, ipole, RAPTOR and grmonty evaluate Christoffel symbols
  analytically or by finite differences (RAPTOR D2: relative FD step, six digits lost near
  coordinate planes). KPolaris integrates contravariant $k^\mu$ with $\Gamma$. The covariant
  Hamiltonian form is Blacklight's (conventions.md, W22 l.262–266); the AD derivative is new.
- **Cite.** White2022; RAPTOR fork analytic connection agrees with AD to $4.5\times10^{-15}$,
  FD to $6.1\times10^{-11}$ (docs/src/validation.md R04).

### C3. Numerical-relativity slice (ADM grid)

- **Implementation.** `ADMGridSpacetime` numerical_spacetime.jl:36; `_cr_weights` :67;
  `_interp` :89; `sample_adm` :117; `_adm_gcov`/`_adm_gcon` :149/:183; `adm_control` :319;
  Z4c reader `read_athenak_bin` athenak_z4c.jl:35, `resample_z4c_adm` :188,
  `z4c_grid_spacetime` :236; device `DevADMParams` gpu_spacetime.jl:43.
- **Algorithm.** Ten ADM fields $(\alpha,\beta^i,\gamma_{ij})$ on a uniform box; tensor-product
  cubic Catmull–Rom weights (centred-difference tangents, $C^1$ across nodes); the stencil index
  uses the dual's value, the weights keep the dual, so AD returns the exact derivative of the
  interpolant. $g_{00}=-\alpha^2+\beta_i\beta^i$, $g_{0i}=\gamma_{ij}\beta^j$, $g_{ij}=\gamma_{ij}$.
  Z4c→ADM: $\gamma_{ij}=\tilde\gamma_{ij}/\chi$ ($\chi\ge10^{-12}$), $\alpha,\beta^i$ passed through.
  Step $h=\max(R,1)/C$; capture when $\alpha<\alpha_{\min}$ (default 0.1); escape radius 18
  (grid-bounded). Off-grid returns $\eta_{\mu\nu}$.
- **CPU/GPU.** CPU and generic-spacetime kernels (gpu_spacetime.jl `trace_shadow_gpu` :317,
  `render_gpu_spacetime` :293).
- **Heritage and changes.** No reference imaging code ray-traces an interpolated ADM slice
  with AD derivatives. Trilinear ($C^0$) interpolation, used for fluid data everywhere, is
  rejected for the metric because it makes $\Gamma$ discontinuous at cell faces. Frozen slice:
  $\partial_t g=0$.
- **Cite.** Zhu2025 (Z4c in AthenaK), Stone2026, AthenaKCode.

### C4. Geodesic integration

- **Implementation.** `geodesic_rhs` integrator.jl:157; `rk4_step_fixed` :293 (low-storage
  stage accumulation); `trace_rays_sampled` :364 (production CPU tracer); `trace_rays` :313
  with `trace_geodesic` :251 (adaptive Vern7, `reltol` $10^{-10}$ default, $10^{-9}$ in
  `trace_rays`); `null_covariant` :129; device tracer gpu_tracer.jl:35.
- **Algorithm.** State $y=[x^\mu;p_\mu]$, $\dot x^\mu=g^{\mu\nu}p_\nu$, $\dot p_t=0$,
  $\dot p_i=-\tfrac12\partial_i g^{\alpha\beta}p_\alpha p_\beta$. Fixed-control classical RK4 with
  $h=(r-r_{\rm h})/C$ (default $C=80$ in `render`, render.jl:87); the accepted states are also
  the radiation samples. Optional caps: `max_step`, `cell_size_fn` (never stride over an AMR
  cell, off by default). Output reordered source→camera with per-segment affine spans. The
  adaptive path (Vern7 with continuous callbacks for horizon $r_{\rm h}+10^{-4}$ and outbound
  escape) is used for capture maps and oracle comparisons.
- **CPU/GPU.** Fixed-control RK4 on both; Vern7 CPU only.
- **Heritage and changes.** Step rule $h\propto(r-r_{\rm h})$ = Blacklight `IntegrateGeodesicsRK4`
  (geodesics.cpp:416–418: "product of ray_step with the radial displacement above the horizon")
  and Mahakala; Blacklight's default is Dormand–Prince RK5(4) with dense output
  (geodesics.cpp:26–38) and a proper-length cap. RAPTOR integrates $k^\mu$ with RK4 and its own
  step control referenced to one hole at the origin; the fork adds this step rule under
  `STEP_BBH` (M3). Differences here: covariant state (exact $p_t$), no re-nulling of $k$
  during the march (RAPTOR re-nulls when $|k\cdot k|>10^{-6}$), and radiation sampled on the
  geodesic states themselves (no interpolation onto a separate sample grid).
- **Cite.** White2022, Sharma2023Mahakala, Bronzwaer2018.

### C5. Ray time-orientation and the termination rule

- **Implementation.** `camera_ray` camera.jl:45 (stores $p=-k/E$); `march_captured`
  integrator.jl:46, `march_escaped` :56, `march_stop` :66 (constants :11–35);
  `ks_radius_energy` :85; `normal_frame_energy` :70; `superposed_control`
  spacetimes.jl:143; `spacetime_control` :165–172; `adm_control` numerical_spacetime.jl:319.
- **Algorithm.** Integrate $p_\mu=-k_\mu$ to increasing $\aff$: the flow
  $(\aff,p)\to(-\aff,-p)$ leaves Hamilton's equations invariant, so the state traces the received
  photon backward with $x^t$ decreasing from 0. Stop when (i) $d<\epsilon_{\rm h}=10^{-3}$ ($d=r-r_{\rm h}$;
  $\min_A(r_A-r_{{\rm h},A})$ for the binary; $\alpha-\alpha_{\min}$ with $\epsilon=0$ for ADM) or
  $d$ non-finite; (ii) $E_n=-k\cdot n=|\dot x^t|/\sqrt{-g^{tt}}>E_{\rm c}=10^2$; (iii) outbound
  beyond the escape sphere ($1.1\times10^3$) or emission radius. For CKS Kerr,
  $g^{tt}=-(1+f)$, $g^{ti}=f l_i$, so $E_n$ costs one `_ks_term`.
- **CPU/GPU.** One function on every march (sampled CPU tracer, device tracer, every kernel,
  fp64 and fp32 lanes; matched scalar types enforced).
- **Heritage and changes.** Blacklight, ipole, Mahakala, KPolaris integrate the physical
  received $k$ to decreasing $\aff$ (conventions.md OPEN-1); equivalent. The redshift cut is new:
  the horizon backstop alone stalls in the superposed binary, where the companion moves the
  null surface outward. Omitted-flux bound $\Delta I_\nu\le\nuobs^3(L/\nuobs)\,\Delta\aff_{\rm rest}
  \max j_\nu/\nu^2$ with $\Delta\aff_{\rm rest}\approx\alpha_{\rm h}/(\kappa E_{\rm c})$
  (integrator.jl:19–33; test/test_march_termination.jl).
- **Cite.** — (new material).

### C6. Camera and device ray generation

- **Implementation.** `PlaneParallelCamera` camera.jl:15; `CameraFrame` :28; `camera_ray`
  :45; `pixel_centres` :54; `camera_rays` :68; `camera_rays_at` :74; `_camera_up`
  render.jl:10; device: `PixelGrid`/`BlockGrid` gpu_camera.jl:20/:36, `_camray_kernel!` :56,
  `rays_on_device!` :100.
- **Algorithm.** Pixel $(\alpha,\beta)$ at
  $x^i=(-\beta\cos\incl+d\sin\incl,\ \alpha,\ \beta\sin\incl+d\cos\incl)$; provisional
  $k^\mu=(1,\sin\incl,0,\cos\incl)$ lowered with $g_{\mu\nu}(x)$; $k_t$ re-solved from the null
  quadratic keeping the root $k_t<0$ (`null_covariant`, fails inside an ergosphere); state
  $p=k/k_t$ so $p_t=+1$. Pixel centres at $-\tfrac{\rm fov}{2}+(j-\tfrac12)\Delta$. Camera vertical
  $(0,-\cos\incl,0,\sin\incl)$ enters the polarized readout (C15). Device rays: a chunk uploads
  only 4-byte pixel or block indices; the kernel calls the same `camera_ray`.
- **CPU/GPU.** Identical function; device rays are `isequal` to host rays
  (test/test_device_rays.jl).
- **Heritage and changes.** Image-plane map = Mahakala Eqs. 5–7. Blacklight has plane-parallel
  and pinhole cameras with an up vector and roll (camera.cpp:387–489); only plane-parallel is
  implemented here, roll 0, observer azimuth 0. Image coordinates coincide with Bardeen's
  ($L_z/E=-\alpha\sin\incl$); RAPTOR's fork camera (M4) reproduces these states to
  $2.4\times10^{-15}$.
- **Cite.** Sharma2023Mahakala, White2022.

### C7. Analytic Kerr equatorial tracer

- **Implementation.** analytic_kerr.jl: `conserved_quantities` :39, `radial_roots` :53,
  `angular_turning_points` :81, `_angular_parts` :111, `_radial_case2_antideriv` :172,
  `_radial_case3_antideriv` :211, `analytic_emission` :315, `_emission_prep` :339,
  `_emission_band` :413; kerr_orbits.jl (radial potential :29, ISCO :40); drivers
  `render_analytic` render.jl:165, `_paint_layer` :145; kernel `_ganalytic_kernel!`
  gpu_analytic.jl:26.
- **Algorithm.** $(\alpha,\beta)\mapsto(\lambda,\eta)$; roots $r_{1..4}$ of the quartic $R(r)$ in
  closed form; angular Mino-time integrals in Legendre form; the source-side equatorial crossing
  of order $n$ from the Mino-time budget $G_\theta^{(n)}$, inverted with Jacobi sn/cn for the two
  radial root cases (II: turning point then escape; III: horizon plunge); outputs
  $(r_s,\phi_s,t_s,\nu_r,{\rm valid})$. Per pixel the $n$-independent part (`_emission_prep`) is
  computed once, each order $n$ finishes with one inversion. Domain normalisation:
  $\theta_o>\pi/2$ mirrored, $a<0$ mirrored, $|a|<10^{-6}$ clamped. ForwardDiff-generic in
  $(\alpha,\beta,\theta_o,a)$.
- **CPU/GPU.** `JacobiElliptic` is device-compatible; the kernel evaluates the same Float64
  closed form (agreement to round-off).
- **Heritage and changes.** Template: Krang.jl (Julia, GPU-capable, MIT) and AART (Python, MIT),
  physics from Gralla & Lupsasca. Changes: split prep/band evaluation; rejection of aliased sn/cn
  "ghost" landings when the required Mino time exceeds the ray's budget; Heaviside $H(0)=1$ for
  $\beta=0$; AD through the elliptic functions (verified against finite differences).
- **Cite.** GrallaLupsasca2020NullGeodesics, GrallaLupsasca2020Lensing, CardenasAvendano2023,
  AARTCode, Chang2024, KrangCode.

### C8. Critical curve, lensing bands, photon-ring layers

- **Implementation.** lensing_bands.jl: `_shell_lambda_eta` :129, `critical_curve` :144,
  `apparent_horizon` :204, `lensing_band` :274, `band_grid` :322, `_in_polygon`/winding
  number :26/:66; `photon_ring_layers` render.jl:220; `cost_order`/`_rescue_pixels` reuse
  `critical_curve` (gpu_image.jl:237, :322).
- **Algorithm.** Photon-shell parametrisation $(\lambda(r),\eta(r))$ gives the critical curve;
  band edges by bisection on AART's residuals (`ApparentBH`, `nlayers`); per-band adaptive
  Cartesian grids and winding-number masks.
- **Heritage and changes.** Port of AART `lensingbands` (MIT notice kept). The critical curve
  is also used for GPU scheduling (C21) and fp64 rescue (C20), which AART does not do.
- **Cite.** CardenasAvendano2023, AARTCode, GrallaLupsasca2020Lensing.

### C9. Equatorial $g$-disk source and interferometry

- **Implementation.** equatorial_source.jl: `gfactor_circular` :80, `gfactor_plunge` :96,
  `gfactor_disk` :116, `johnson_su_profile` :133, `equatorial_disk_intensity` :156/:201;
  interferometry.jl: `radon_cut` :65, `tukey_window` :26, `visamp` :106, `visamp_minima` :138.
- **Algorithm.** $g=\nuobs/\nu_{\rm emit}$ for sub-Keplerian circular flow outside the ISCO and
  radial plunge inside (AART parameters `sub_kep`, $\beta_r$, $\beta_\phi$), intensity
  $g^3\times$ Johnson-SU radial profile. Visibility amplitude: Radon projection at a baseline
  angle, Tukey window ($\alpha=0.5$), zero-padded 1-D FFT (Fourier-slice theorem).
- **Heritage and changes.** Port of AART `intensity_f.py`, `rprofs_f.py`, `visamp_f.py`
  (MIT). No physics change.
- **Cite.** CardenasAvendano2023, AARTCode, GrallaLupsasca2020NullGeodesics,
  Johnson2020 **[new]**, Gralla2020Shape **[new]** (Johnson-SU profile reference in the code
  comment, arXiv:2008.03879).

### C10. AthenaK input and fluid sampling

- **Implementation.** athenak_reader.jl: `AthenaKSnapshot` :208 (`.athdf`), `LeafTable` :29,
  `_build_leaf_table` :85, `_leaf_locate` :118, `LevelHash` :53, `_hash_locate` :156,
  `_find_meshblock_scan` :273, `meshblock_index` :333, `sample_primitives` :360;
  `athenak_bin_snapshot` athenak_z4c.jl:272 (`.bin`); `primitive_four_vectors`
  primitives.jl:23; device: `upload_amr` gpu_amr.jl:66, `_gamr_sample` :149; uniform resample
  `resample_fluid_uniform` gpu_image.jl:66, `_gimg_sample` :126.
- **Algorithm.** Owning meshblock by a dense finest-block index grid (every AthenaK block face
  is a multiple of the finest block width, so each index cell has one leaf; $O(1)$), a per-level
  hash (when the dense grid would exceed $8\times10^6$ cells), or a linear scan (oracle). Clamped
  cell-centred trilinear interpolation of the 8 primitives
  $(\rho,u^1,u^2,u^3,e,B^1,B^2,B^3)$. Four-vectors: $u^0$ from normalisation of the
  normal-frame velocity, $u^\mu$ via lapse and shift, $b^0=u_iB^i$, $b^i=(B^i+b^0u^i)/u^0$.
- **Data.** Float32 block array `(8, ni, nj, nk, nmb)` on device (feature-fastest), per-block
  left cell centre and spacing, Int32 leaf grid.
- **CPU/GPU.** Device AMR sampler is bit-identical to `sample_primitives` (gpu_amr.jl:1–8); the
  uniform-grid path is an approximation (percent-level flux changes; can rotate EVPA on
  Faraday-thick decks).
- **Heritage and changes.** Four-vector construction = Blacklight `PrimitiveToFourVectorsG`.
  Blacklight reads `.athdf` with a block search; the dense leaf grid and hash are new. RAPTOR
  upstream samples with weights measured from the lower cell face while data sit at cell
  centres (D4); the fork reader (M5) uses this code's cell-centred scheme.
- **Cite.** White2022, Stone2026, AthenaKCode.

### C11. Units, electron temperature, cuts

- **Implementation.** `units` athenak_reader.jl:408; `plasma_decode` coefficients.jl:292;
  `electron_theta` :266; `plasma_cut_omits` :240; config coefficients.jl:186–228,
  parameters.jl:21–45.
- **Algorithm.** $L=GM/c^2$, $\rho_{\rm unit}=\Mscale/L^3$, $B_{\rm unit}=c\sqrt{4\pi\rho_{\rm unit}}$;
  $n=\rho/(\mu m_p)$, $n_e=n/(1+n_i/n_e)$; $\beta^{-1}=b^2/(2p_{\rm gas})$, $\sigma=b^2/\rho$;
  $R=(R_{\rm high}\beta^2+R_{\rm low})/(1+\beta^2)$;
  total-pressure split $k_BT_e=\frac{1+n_e/n_i}{R+n_e/n_i}\,\mu m_p p_{\rm gas}/\rho$ (default), or
  internal-energy split with $\gamma_i=5/3$, $\gamma_e=4/3$; 14 optional cuts (defaults: $\sigma>1$
  omitted). Same decode for imaging and Monte Carlo (`plasma_at` grmhd_model.jl:104).
- **Heritage and changes.** $R(\beta)$ of Mościbrodzka+2016 with $\beta_{\rm crit}=1$; decode and
  cut logic = Blacklight `PlasmaCutOmits`/`ti_te_beta`. RAPTOR upstream uses
  $(\gamma-1)=1/3$ regardless of the dump's $\gamma$ (D3), $0.75\times$ this $\Theta_e$ at
  $\gamma=13/9$.
- **Cite.** Moscibrodzka2016, White2022, Wong2022 (internal-energy split used by KPolaris).

### C12. Transfer coefficients

- **Implementation.** coefficients.jl: `thermal_synchrotron` :316 (unpolarized),
  `thermal_synchrotron_polarized` :365, `power_law_synchrotron_polarized` :485,
  `kappa_synchrotron_polarized` :563, `harmonic_bridge` :453, `precompute_power_law!` :645,
  `precompute_kappa!` :683, `_hypergeometric_2f1` :618, Bessel fits :22–69; mixing
  `eight_coefficients` integrator.jl:25; `coefficient_params` coefficients.jl:217.
- **Algorithm.** Fluid-frame $\nu=\nuobs(-k_\mu u^\mu)$; signed $\cos\thB$ from tetrad components
  of $k$ and $b$ (integrator.jl:44–53). Thermal: $j_I,j_Q,j_V$ = the Marszewski+2021 thermal fits
  (their $J_S$, with $J_V$ from Dexter 2016 recast), $j_Q<0$; $\alpha_S=j_S/B_\nu$ with $B_\nu$ via
  `expm1`; $\rho_Q=-\frac{n_ee^2\nu_c^2\sin^2\thB}{m_ec\nu^3}f_m(X)[K_1/K_2+6\Thetae]$,
  $\rho_V=\frac{2n_ee^2\nu_c}{m_ec\nu^2}\frac{K_0-\Delta J_5}{K_2}\cos\thB$ (Shcherbakov 2008 as
  modified by Dexter 2016, in the Marszewski sign convention), converted to invariants.
  Power law and kappa: Pandya+2016 fits (kappa: low/high-frequency asymptotes joined by
  harmonic bridges), with $p$-, $\kappa$-dependent $\Gamma$, Bessel and ${}_2F_1$ factors precomputed
  on the host; the populations add with their fractions. Parameter sets carry a population's
  fields only when its fraction is non-zero, so a thermal deck compiles to the thermal path.
- **Changes relative to Blacklight** (whose coefficient code this follows function by
  function): (i) $\rho_Q$ cold-plasma switch at $\Thetae=0.01$ replaced by a $C^1$ smoothstep over
  $[0.005,0.01]$ (coefficients.jl:343–358, :408–411); (ii) $\rho_V$ factor clamped to its cold
  limit when outside $[0,1]$; (iii) power-law $\rho_Q$ removable pole at $p=2$ evaluated as its
  $\ln X$ limit (:513–517). Basis: the fits' $Q>0$ lies in the $k$–$B$ plane (Marszewski basis),
  which is the tetrad's $e_{(2)}$ direction (C13); grtrans/Huang+2011 put $k$ in the $y$–$z$ plane,
  which reverses every $Q$ and $U$ coefficient (M21 l.89). RAPTOR's native fits differ in
  $\alpha_Q$ by more than 10% on 19% of test cells at $\Thetae\ge32$ and give NaN $\rho_Q$ for
  $\Thetae<1.26\times10^{-3}$ (D10).
- **CPU/GPU.** Same functions; Bessel and special functions are polynomial fits or host
  precomputation, so the kernel code is closed form.
- **Cite.** Marszewski2021, Pandya2016, Dexter2016, Shcherbakov2008 **[new]**, Leung2011
  **[new]** (the $j_I$ thermal fit; MC emissivity cites it, emissivity.jl:6), White2022.

### C13. Fluid tetrad

- **Implementation.** `fluid_tetrad_checked` tetrads.jl:34; `fluid_tetrad` :20;
  `_normal_observer` integrator.jl:122.
- **Algorithm.** $\omega=-k_\mu u^\mu$; $e_{(0)}=u$; $e_{(3)}=k/\omega-u$;
  $e_{(2)}\propto b-\frac{k\cdot b}{\omega}e_{(3)}+\frac{u\cdot b}{\omega}k$ normalised with $g_{\mu\nu}$;
  $e_{(1)}$ from the metric cross product $\epsilon_{\mu\alpha\beta\gamma}e_{(0)}^\alpha e_{(2)}^\beta
  e_{(3)}^\gamma$ (the $\sqrt{-g}$ factor is 1 in CKS Kerr; elsewhere $e_{(1)}$ is renormalised,
  `renorm_e1`). Rows are $e_{(a)}^{\ \mu}$. Validity flag false when no spacelike $e_{(2)}$ exists.
- **Heritage and changes.** Blacklight `Tetrad` (radiation_geometry.hpp). Change: when $b$ lies
  in span$(u,k)$ ($\sin\thB=0$) the projected vector vanishes and Blacklight divides by zero; here
  the coordinate axis most orthogonal to $e_{(3)}$ is projected instead (ipole
  `set_Econ_from_trial` fallback). The same function builds the camera tetrad and, at every
  sample, the readout screen from the transported frame (C15).
- **Cite.** White2022, MoscibrodzkaGammie2018.

### C14. Polarized and unpolarized segment operators

- **Implementation.** `stokes_segment` transfer.jl:36; `_segment_total_absorption` :119;
  `stokes_admissible` :133; `stokes_couple` :150; `unpolarized_segment` :165;
  `couple_unpolarized` :185; `POL_DELTA_TAU_MAX=100` :15.
- **Algorithm.** For constant $(\boldsymbol j,\Kmat)$ over $\Delta\aff_{\rm c}$, the exact solution
  $\Svec\mapsto M\Svec+b$ with $M=e^{-\Kmat\Delta\aff_{\rm c}}$, $b=(\mathbf 1-M)\Kmat^{-1}\boldsymbol j$, written
  with $\lambda_{1,2}=\big[\sqrt{\tfrac14(\alpha_P^2-\rho_P^2)^2+(\boldsymbol\alpha\cdot\boldsymbol\rho)^2}
  \pm\tfrac12(\alpha_P^2-\rho_P^2)\big]^{1/2}$ and the matrices $M_2,M_3,M_4$ of Landi
  Degl'Innocenti & Landi Degl'Innocenti (1985) as used by ipole/Blacklight. Branches: emission
  only; total absorption ($\alpha_P=\rho_P=0$); pure rotation ($\alpha_I=0$; closed-form rotation
  about $\hat\rho$); full. Saturation at $\alpha_I\Delta\aff_{\rm c}>100$: $M=0$, $b=\Kmat^{-1}\boldsymbol j$.
  Admissibility: $|\alpha_P|>\alpha_I$ and $|j_P|>j_I$ rescaled onto the cone before forming the
  operator; the accumulated vector is projected onto $|P|\le I$. Unpolarized: $A=e^{-\Delta\tau}$,
  $B=(j/\alpha)e^{-\Delta\tau}{\rm expm1}(\Delta\tau)$, $B=j\Delta\aff_{\rm c}$ for $\alpha\le0$.
- **Changes relative to Blacklight `StokesCoupleNoSplit` / ipole.** (i) the dedicated
  absorption-without-rotation branch is removed ($\rho_P=0$ goes through the full solution with
  $\lambda_2=0$; the removed branch disagreed with a matrix-exponential oracle at ~20%);
  (ii) degenerate eigenstructure falls back to total absorption instead of dividing by zero;
  (iii) Blacklight `mm_2`/`mm_3` index typos and `mm_4` signs corrected; (iv) cone rescales
  (order-independence of the observed Stokes vector, test/test_emission_admissibility.jl).
  Blacklight and ipole Strang-split coupling from transport (polarized.cpp:48–50); RAPTOR
  selects RK4 or implicit trapezoid by a stiffness test that uses the wrong sign (D9, all
  absorbing segments go to the trapezoid, second order; D12 negative transmission for
  $\tau>2$).
- **Cite.** LandiDeglInnocenti1985, MoscibrodzkaGammie2018, White2022, Bronzwaer2020.

### C15. Reverse polarized march with a transported readout frame

- **Implementation.** `reverse_polarized_march` reverse_march.jl:223; `_qu_rotation` :185;
  `_rotate_qu` :196/:198; `geodesic_transport_rhs` integrator.jl:181; march controls
  `KerrMarch` :26, `SpacetimeMarch` :41, `StoredPath` :57, `march_control` :65–77.
- **Algorithm.**
  1. At the camera: normal observer $e_{(0)}=n$, camera up vector projected, camera tetrad from
     `fluid_tetrad`; keep $E_{(2)}$ (screen up) and $E_{(0)}$.
  2. Each step: RK4 of the 16-component system $\{y,E_{(2)},E_{(0)}\}$, with
     $\dot V^\mu=-\Gamma^\mu_{\alpha\beta}k^\alpha V^\beta$ evaluated as
     $-\tfrac12 g^{\mu\lambda}[(k\cdot\partial)g_{\lambda\beta}V^\beta+(V\cdot\partial)g_{\lambda\alpha}k^\alpha
     -k^\alpha V^\beta\partial_\lambda g_{\alpha\beta}]$ from one dual-seeded `metric_and_inverse`
     (no $\Gamma$ formed).
  3. At a sample with plasma: fluid tetrad and coefficients; readout tetrad rebuilt from the
     transported $(E_{(0)},E_{(2)})$ and the current $k$; $(\cos2\chi,\sin2\chi)$ from
     $E_{(1)}\cdot e^{\rm fl}_{(1)}$, $E_{(1)}\cdot e^{\rm fl}_{(2)}$; $(M_n,b_n)$ from C14;
     $\Svec\leftarrow\mathcal P(\Svec+G_nR_nb_n)$, $G_{n+1}=G_nR_nM_nR_n^{\mathsf T}$ ($\mathcal P$ = cone projection).
  4. Stop by `march_stop` or when the readout screen has no spacelike leg.
  5. Return $\nuobs^3\Svec$ in the camera frame; no final rotation.
- **Data.** Per ray: $y$ (8), two frame vectors (8), $\Svec$ (4), $G$ (16) — independent of
  path length.
- **CPU/GPU.** CPU wraps stored trajectories in `StoredPath` (the RK4 result then only carries
  the frame, `next_state` :81); kernels use `KerrMarch`/`SpacetimeMarch`. Geodesic and frame in
  the lane type (fp32 or fp64); coefficients and Mueller algebra in Float64.
- **Heritage and changes.** ipole (and Blacklight, polarized.cpp:48) parallel-transport the
  coherency tensor $N^{\mu\nu}$ in coordinates with a second-order split scheme, converting to the
  fluid-frame Stokes vector and back at each step (ipole l.387, l.472, l.519). Here the
  radiation is never in coordinate form: only two vectors are transported with the geodesic's
  RK4, and the fluid-frame operator is rotated into the transported frame. Measured on a Kerr
  torus: 30% smaller Stokes errors and 5–8$\times$ smaller EVPA errors than Strang-split coherency
  transport at equal $C$ (docs/src/method.md). Accumulation order: camera→source with a running
  transmission, so the kernel needs no stored path (KPolaris runs a separate pass for the
  geodesic and one for transfer).
- **Cite.** MoscibrodzkaGammie2018, White2022, Zhang2026KPolaris, LandiDeglInnocenti1985.

### C16. Reverse unpolarized march and parameter lanes

- **Implementation.** `reverse_unpolarized_march` reverse_march.jl:142; `_segment_ab`
  :110/:116; `unpolarized_march` integrator.jl:174.
- **Algorithm.** $I_{\rm inv}\mathrel{+}=\mathcal T B_n$, $\mathcal T\leftarrow\mathcal T A_n$; accumulator type
  generic: Float64, a ForwardDiff dual (parameter derivative, C20), or an $K$-vector with one lane
  per mass scale (`render_gpu_fused_kvariant` gpu_image.jl:470).
- **Heritage and changes.** Segment operator = Blacklight `UnpolarizedTransferStep`; the
  reverse accumulation and the multi-lane accumulator are new. RAPTOR drops emission on
  absorption-free segments (D8).
- **Cite.** White2022.

### C17. Slow light

- **Implementation.** `SnapshotSeries` time_series.jl:22; `bracketing_pair` :112;
  `get_frame` :152 (LRU of `maxslots` frames); `sample_primitives_t` :203; device windows
  `upload_series_window` gpu_image.jl:576, `_fused_kernel_sl!` :631, `render_gpu_fused_slow`
  :668, AMR: `_gamr_sample_t` gpu_amr.jl:585, `render_gpu_amr_slow` :652.
- **Algorithm.** Emission time $t_{\rm obs}+x^t$ ($x^t\le0$ on the backward ray); the two
  bracketing dumps are sampled spatially and primitives blended linearly in $t$; ends clamped;
  $\Delta t_{\rm dump}\to\infty$ recovers fast light bit for bit.
- **Heritage and changes.** Scheme of RAPTOR I; window and blend conventions checked against
  ipole. Device: two-frame window uploaded and advanced per frame.
- **Cite.** Bronzwaer2018, MoscibrodzkaGammie2018.

### C18. Image-plane adaptive refinement

- **Implementation.** `render_amr` amr.jl:120; `_block_needs_refinement` :12;
  `stream_blocks!` :238; `_paint_block!` :208; device marcher `AMRStokes` gpu_amr.jl:442,
  `stream_blocks!(::AMRStokes)` :465.
- **Algorithm.** Image tiled into blocks of `block` pixels (default 8); a block refines into four
  children at twice the pixel density when the fraction of pixels whose relative gradient
  exceeds 0.15 is above 0.03, or whose relative 5-point Laplacian exceeds 1.0 is above 0.03;
  optionally the absolute-gradient test alone (refine at $\ge5$ pixels above the cut); up to
  `max_level` (default 2); output mosaic at base$\cdot2^{\rm max\_level}$ with a level map. Device
  driver: each block is a work item; the lane that receives a block's Stokes values tests it and
  enqueues its children immediately (no per-level barrier); mosaic, level map and ray count equal
  the level-synchronous driver (test/test_device_amr.jl).
- **Heritage and changes.** White (2022) adaptive ray tracing; Blacklight
  `radiation_adaptive.cpp` evaluates value, absolute-gradient, relative-gradient,
  absolute-Laplacian and relative-Laplacian tests with fraction thresholds (:170–226). Here the
  relative pair is the default; the absolute gradient is a count threshold and, when set,
  replaces the relative pair; value and absolute-Laplacian tests are not implemented. Blacklight
  refines level by level and re-traces each level's geodesics; here levels stream.
- **Cite.** White2022.

### C19. Fused GPU kernels and fluid representations

- **Implementation.** `_fused_kernel!` gpu_image.jl:157; `_gpol_kernel!` gpu_pol.jl:95;
  `_famr_kernel!`/`_gpol_amr_kernel!` gpu_amr.jl:219/:232; `_gsp_amr_kernel!`/
  `_gsp_pol_amr_kernel!` gpu_spacetime.jl:144/:158 (binary and ADM spacetimes, scan locate);
  `upload_grid` gpu_image.jl:96; `upload_amr` gpu_amr.jl:66; `upload_amr_scan`
  gpu_spacetime.jl:127.
- **Algorithm.** One ray per thread; the kernel body is one call of the reverse march (C15/C16)
  with the sampler closure swapped: uniform-grid trilinear, native AMR (leaf locate + the CPU
  trilinear), or scan locate for non-Kerr spacetimes. No trajectory or $\Gamma$ array is stored; only
  the per-pixel result leaves the device. Kernels differ only in the sampler.
- **Heritage and changes.** KPolaris: CUDA with a geodesic pass and a transfer pass. GPUmonty:
  CUDA kernels for Monte Carlo. Here the kernels are KernelAbstractions compilations of the CPU
  Julia functions, so CPU and device run the same source; on identical trajectories CPU march
  and kernel agree to $1.4\times10^{-8}$ ($I$) and $\le3.5\times10^{-9}$ ($Q,U,V$) (docs/src/method.md).
- **Cite.** Zhang2026KPolaris, Motta2026GPUmonty.

### C20. Mixed precision and parameter derivatives

- **Implementation.** `render_gpu_fused(...; precision=:mixed)` gpu_image.jl:381;
  `_rescue_pixels` :261; `_walk_critical_curve` :230; `_mixed_parts` :358; AMR `_amr_mixed!`
  gpu_amr.jl:329; `render_gpu_fused_grad` gpu_image_ad.jl:66, `_giad_params` :36.
- **Algorithm.** Mixed: the geodesic and frame run in Float32, sampling and transfer in Float64;
  pixels within 2 pixels (Chebyshev) of the Kerr critical curve are marched in Float64 and queued
  first. Derivative: one scalar of the parameter tuple ($\Mscale$, which scales $\rho_{\rm unit}$,
  $e_{\rm unit}\propto\Mscale$ and $B_{\rm unit}\propto\sqrt{\Mscale}$, or $R_{\rm high}$) is a dual with one
  partial; because each segment is affine in $I$, the dual accumulator returns $I$ and
  $\partial I/\partial\theta$ exactly in one pass (the geodesic stays Float64).
- **Heritage and changes.** Jipole differentiates ipole's transfer with forward-mode AD; here
  the same identity is used inside the GPU kernel, for unpolarized images and two parameters.
  Enzyme reverse mode is verified on the transfer operator (method.tex:6), not on the kernel.
- **Cite.** NaetheMotta2025.

### C21. Multi-device runtime

- **Implementation.** gpu_pool.jl: `HostDevice` :22, `PoolLane` :91, `DevicePool` :113/:136,
  `_lane_loop` :162, `_work!` :189, `enqueue!` :208, `_run!` :220, `foreach_device` :255,
  `resident` :344, `lane_stage` :381, `march!` :438, `march_chunk!` :454, `queue_items` :476;
  `cost_order` gpu_image.jl:322; `march_sort_order` :173.
- **Algorithm.** Per device, `lanes` (default 2) persistent worker tasks, each with its own stream,
  device buffers and pinned staging buffers (freed explicitly after the last job). A job is a
  channel of items; lanes pull the next item when done, so slower devices take fewer; streaming
  jobs may `enqueue!` new items (AMR children). Inputs uploaded once per pool and cached.
  Queue order: $16\times16$ tiles in descending predicted cost
  $\sum_{\rm px}\log[n/(d+\tfrac14)]$, $d$ the distance to the critical curve on a 4-pixel grid;
  or measured trip counts of a previous frame, longest first. Determinism: every item writes only
  its own output index, so the image is bitwise independent of device count, chunking and
  queue order.
- **Heritage and changes.** No reference imaging code has a multi-device scheduler; GPUmonty
  splits photons into batches on one device.
- **Cite.** — (new material); test/test_device_invariance.jl, test/test_device_pool.jl,
  test/test_device_schedule.jl.

### C22. Monte Carlo: emission tables and superphoton generation

- **Implementation.** `EmissionZone` emission_tables.jl:40; `build_weight_table` :96/:206;
  `zone_nsuperphotons` :126; `zone_counts` :245; `IntJnuTable` :305 (opt-in);
  `int_jnu_zone` emissivity.jl:64; `emission_tetrad` :94; `sample_zone_photon`
  generate.jl:35/:89; `emission_zones` grmhd_model.jl:150; `Superphoton` superphoton.jl:27.
- **Algorithm.** Weight $w(\nu)=\sum_{\rm zones}\big[\int j_\nu\,d\Omega\big]\,dV/(h\Nsp)$ on 401 log-spaced
  nodes over $[10^8,10^{16}]$ Hz (log-linear interpolation, floor $10^{-100}$), $dV=\sqrt{-g}\,d^3x\,L^3$.
  Per zone: expected count $N_z=(dV/h)\int d\ln\nu\,\big[\int j_\nu d\Omega\big]/w$ and envelope
  $\max_\nu[\int j_\nu d\Omega]/w$. Generation: $\nu$ log-uniform accepted with probability
  $(\int j_\nu d\Omega/w)/{\rm dnmax}$; $\cos\theta$ about $\hat b$ accepted with
  $j_\nu(\theta)/j_\nu(\pi/2)$; tetrad-frame null vector mapped through the $(u,\hat b)$ tetrad and
  lowered to $p_\mu$. Emissivity: thermal $j_I$ of C12 (Leung+2011 form) without the invariant
  factor.
- **Heritage and changes.** grmonty `init_weight_table`/`init_zone`/`sample_zone_photon`,
  GPUmonty weights.cu/kernels.cu. Changes: covariant momentum; rejection loops capped
  (never reached); the emissivity is the imaging pipeline's function.
- **Cite.** Dolence2009, Motta2026GPUmonty, Leung2011 **[new]**.

### C23. Monte Carlo: tracking and absorption

- **Implementation.** `track_superphoton` track.jl:50 (flat sphere); `track_superphoton_grmhd`
  mc_cpu.jl:337; `_grmhd_dlam` :320; `_grmhd_nu_alpha` :264; `_grmhd_escape_theta` :308.
- **Algorithm.** Curved-space tracking by RK4 with affine step $(r-r_{\rm h})/(C|k^i|)$ (so the
  coordinate advance is $(r-r_{\rm h})/C$ although $|k|\sim h\nu/m_ec^2$); weight
  $w\leftarrow w\,e^{-\Delta\tau_a}$ with the trapezoid
  $\Delta\tau_a=\tfrac12[(\nu\alpha_\nu)_n+(\nu\alpha_\nu)_{n+1}]\,(L\,h/m_ec^2)\Delta\aff$; capture at
  $r-r_{\rm h}<10^{-3}$; step cap $2\times10^4$; record by the unfolded polar angle of the asymptotic
  direction.
- **Heritage and changes.** grmonty `track_super_photon`, `dtauK`, `stepsize`,
  `get_bk_angle`. The GRMHD tracker shares `plasma_at` with imaging, so both see identical
  plasma. The GRMHD Monte Carlo runs on the CPU; the device path covers the sphere model.
- **Cite.** Dolence2009.

### C24. Monte Carlo: scattering, bias, hot cross section

- **Implementation.** `_mc_scatter_event` mc_common.jl:44; `_mc_march_scatter_core` :68;
  `track_superphoton_scatter!` track.jl:213; `track_superphoton_grmhd_scatter!`
  mc_cpu.jl:389; `bias_func` track.jl:119; `grmhd_bias` mc_cpu.jl:301; scatter.jl:
  `lorentz_boost` :32, `sample_y` :55, `sample_electron` :135, `klein_nishina` :185,
  `sample_scattered_energy` :230, `compton_scatter` :252, `scatter_superphoton` :289/:317;
  hotcross.jl: `kn_total` :43, `boostcross` :62, `build_hotcross_table` :163, `load_hotcross`
  :140, `hotcross_lookup` :192, `alpha_inv_scatt` :245.
- **Algorithm.** Per step, scattering depth $\Delta\tau_s$ (trapezoid of $\nu n_e\sigh$); draw
  $x_1=-\ln U$; spawn a child when $b\,\Delta\tau_s>x_1$ at fraction $x_1/(b\Delta\tau_s)$ into the step,
  with weight $w(1-e^{-\Delta\tau_s})/(1-e^{-b\Delta\tau_s})$; the parent is depleted by the physical
  $e^{-\Delta\tau_s}$ every step. Hence $\mathbb E[\text{child weight}]=w(1-e^{-\Delta\tau_s})$ for any
  $b\ge1$. Bias: sphere $b=\max(1,n_e\sigma_TR(n_{\rm sc}+1)^2)$; GRMHD
  $b=\max(1,b_0(n_{\rm sc}+1)^2)$. Kinematics: electron from the relativistic Maxwellian by the
  Canfield $\chi^2$-mixture sampler, flux-weighted by $(1-\beta\mu)$ and Klein–Nishina accepted;
  boost to the electron frame, Klein–Nishina (or Thomson) energy and angle, boost back.
  $\sigh(\varepsilon,\Thetae)$: Maxwell–Jüttner average of $\sigma_{\rm KN}$ by midpoint quadrature
  (40 $\mu_e$ × 240 $\gamma_e$ nodes, $\gamma_e\le1+12\Thetae$), tabulated on $221\times81$ log–log nodes over
  $\varepsilon\in[10^{-12},10^{15}]$, $\Thetae\in[10^{-4},10^4]$.
- **Heritage and changes.** grmonty/GPUmonty scattering (GPUmonty §3.2 and its sphere bias
  $(n_{\rm sc}+1)^2$). GPUmonty's split is $w\to w(1-1/b)$ for the parent and $w/b$ for the child
  (GPUmonty l.196); here the parent loses the physical fraction and the child carries the ratio
  above, which reduces to $w/b$ for $b\Delta\tau_s\to0$. Device scattering is layered: the host
  relaunches once per generation and reruns a launch whose child buffer overflows at the exact
  size, so no child is dropped. Direction sampling by inversion (no rejection); samplers
  reimplemented from the algorithms, no GPL source copied (rng.jl:1–9).
- **Cite.** Dolence2009, Motta2026GPUmonty, Canfield1987, Kahn1950, Wienke1985 **[new,
  optional]** (hot cross section, cited in hotcross.jl:7).

### C25. Monte Carlo: counter-based streams keyed by ancestry

- **Implementation.** rng.jl: `philox4x32` :38, `philox_rand` :57/:64, `MCRng` :83,
  `mc_uniform` :101, `MCStreamLayout` :134, `mc_stream_id` :154, `mc_primary_rng` :172,
  `mc_child_rng` :183, `mc_gauss`, `sample_direction`, `standard_gamma`.
- **Algorithm.** Philox4x32-10 (10 rounds, multipliers `0xD2511F53`, `0xCD9E8D57`, Weyl key
  increments). Draw $n$ of a stream: ${\rm Philox}((n,s_1,s_2,s_3),(\text{seed},s_0))$; uniform from two
  output words ($[0,1-2^{-64}]$). Stream identity
  $I=g+\sum_{j=1}^{k}e_j2^{G+(j-1)E}+k\,2^{124}$ ($g$ primary index, $e_j$ spawn ordinal of the
  layer-$j$ ancestor along its parent's march, $k\le15$ layer), with $G$ from $N$ and $E$ from the
  step budget; runs with $G+kE>124$ are rejected, so distinct genealogies give disjoint Philox
  inputs. The child stream depends only on the parent identity and the event ordinal.
- **Heritage and changes.** grmonty uses a stateful per-thread generator; GPUmonty uses
  cuRAND states; neither result is independent of scheduling. Philox from Salmon+2011; the
  genealogy key is new.
- **Cite.** Salmon2011.

### C26. Monte Carlo: exact accumulation and estimator

- **Implementation.** exact_sum.jl: `xsum_parts` :38, `xsum_add!` :64, `xsum_round` :83,
  `ExactSpectrum` :146, `merge_spectra!` :200, `Spectrum(::ExactSpectrum)` :208,
  `ExactScalars` :221; device `_xsum_atomic_add!` mc_gpu.jl:65; spectrum.jl: `record!`
  :74 (exact: exact_sum.jl:175), `record_energy!` :103 (exact_sum.jl:190), `nuLnu_solid` :126, `nuLnu` :163, `nuLnu_total` :184.
- **Algorithm.** A double is an integer multiple of $2^{-1074}$ below $2^{1024}$; it is split into
  three consecutive 31-bit limbs of a 68-limb Int64 accumulator (2108 bits); integer addition is
  associative, so the sum is order independent; one round to nearest-even at the end. Capacity
  $2^{32}-1$ addends per bin; non-finite values counted separately. Estimator per angle bin:
  $\nu L_\nu=m_ec^2\,(4\pi/\Delta\Omega)\,(\Delta\ln\nu)^{-1}\sum_jw_jE_j$, $E=-p_t$ in $m_ec^2$,
  $\Delta\Omega=2\cdot2\pi(\cos\theta_i-\cos\theta_f)$ for equator-folded bins.
- **Heritage and changes.** Estimator = grmonty `report_spectrum`. Exact sums are new (grmonty
  and GPUmonty reduce in floating point); they make spectra, diagnostics and child sets bitwise
  identical for any thread and device count (test/test_mc_exact.jl).
- **Cite.** Dolence2009.

### C27. Monte Carlo validation sphere

- **Implementation.** `SphereModel` sphere_model.jl:21/:41; `sphere_zone` :62;
  `analytic_nuLnu`; `mc_spectrum_gpu` mc_gpu.jl:304 (thin), :516 (scattering).
- **Algorithm.** Uniform static sphere in Minkowski, reference point $n_e=10^{13}$ cm$^{-3}$,
  $\Thetae=100$, $B=1$ G, $R=1$ cm; field along $x^1$ (instead of $x^3$) to avoid the Gram–Schmidt
  degeneracy of the emission tetrad; compared with the closed-form thin-sphere spectrum.
- **Heritage and changes.** GPUmonty `sphere_model` test.
- **Cite.** Motta2026GPUmonty.

### C28. RAPTOR comparison build

- **Implementation.** ref-code/jordydavelaar-raptor, branch `blp-bbh` from `b91e937`;
  patches and recipe in results/blp/raptor/; switch table and defect table in
  ref-code/jordydavelaar-raptor/BLP_CHANGES.md; protocol and results docs/src/validation.md.
- **Content for the appendix.** Switches (defaults keep upstream bitwise):
  `metric==SKS` (M1 superposed binary with Woodbury inverse), `SKS_CONNECTION` (M2a FD with
  absolute step / M2b analytic), `STEP_BBH` (M3 this code's march, stop rule and sample pairing),
  `CAMERA_PP` (M4), AthenaK reader and decode (M5/M6), `EMIS_BLP` (M7 coefficients and cone
  rescales), `STOKES_IAU` (M8), `POL_SOLVER_LD85` (M11 exact segment step, agrees with
  `stokes_couple` to $4.2\times10^{-14}$), `POL_KEEP_I`; M12 one dynamic OpenMP loop over pixels.
  Upstream defects D1–D16 (metric regularisers, FD step, $\Theta_e$ with fixed $\gamma-1$, trilinear
  offset, ASCII writer, Planck `exp-1`, camera basis sign $Q,U\to-Q,-U$, dropped emission,
  stiffness sign, $\rho_Q$ NaN, zeroed faint pixels, trapezoid negative transmission, sample
  pairing, OpenMP blocking, setup flag, uninitialised variable). The stiffness telemetry in
  gpu_pol.jl:20–60 re-derives the eigenvalue criterion ($\max|\lambda|\Delta s>-\ln0.99$) without
  RAPTOR source (GPLv3).
- **Cite.** Bronzwaer2018, Bronzwaer2020, Gold2020 **[new]** (metrics and test problems T1–T5).

### C29. Analytic plasma sources (optional appendix paragraph)

- `PolishDoughnut` polish_doughnut.jl:27 (constant-$\ell$ torus, polytrope; conventions pinned
  by Gyoto, equations only), `PageThorneDisk` page_thorne.jl:24 (Page–Thorne flux, blackbody
  $T_{\rm eff}$), `EquatorialHotspot` hotspot.jl:24 (Gaussian in proper separation on a Keplerian
  orbit), all through `source_plasma_at` sources.jl:53. Candidate new keys: Abramowicz1978,
  Komissarov2006, PageThorne1974, SchnittmanBertschinger2004 (not verified here; only needed if
  the appendix lists these sources).

---

## 5. Statements to align between paper text, docs and code

| Item | Text | Code | Note |
|---|---|---|---|
| Blacklight parallelism | docs/src/method.md and BlackLightPlus.jl:5 say "C++/Kokkos" | ref-code blacklight.cpp:10 includes `omp.h`, `omp_set_num_threads` | Blacklight is C++ with OpenMP. |
| Thermal emissivity source | docs/src/method.md: "polarized emissivities in the form of Dexter2016" | coefficients.jl:365 cites Marszewski+2021 Eqs. 28/30/31 | M21 tabulates $J_I,J_Q$ (Leung/Pandya fits) and adopts the Dexter 2016 $J_V$; cite Marszewski2021 (+Dexter2016 for $J_V$ and $\rho$). |
| Rotativity origin | method.tex:60 cites Pandya2016, Marszewski2021 | $\rho_Q,\rho_V$ are Shcherbakov 2008 as modified by Dexter 2016 (M21 l.344) | Shcherbakov2008 absent from the bib. |
| GL "P2" | analytic_kerr.jl:8, lensing_bands.jl:8 cite PRD 101 044033 | bib `GrallaLupsasca2020Lensing` is PRD 101 044031 (arXiv:1910.12873, verified) | Code comment eid is 044031. |
| Monte Carlo on device | method.tex:79–80 describes GRMHD MC with device-count independence | `mc_spectrum_gpu` takes a `SphereModel` only (mc_gpu.jl:304, :516); GRMHD MC is threaded CPU | Device invariance holds for the sphere; thread invariance for GRMHD. |
| Adaptive refinement tests | method.tex silent | amr.jl implements relative gradient + relative Laplacian (default) or absolute gradient | Blacklight also has value and absolute-Laplacian tests. |
| Vern7 use | method.tex:30 "capture maps use Vern7" | `trace_geodesic` Vern7 is the adaptive CPU path; `trace_shadow_gpu` is RK4 | Capture-map provenance per paper/figures/CAPTIONS.md. |
| Superposed inverse | method.tex:21 | CPU uses the full $4\times4$ inverse; RAPTOR fork uses Woodbury | Both exact. |

---

## 6. Citations

### 6.1 Existing keys used above (paper/bibliography.bib)

White2022, BlacklightCode, MoscibrodzkaGammie2018, Dexter2016, Bronzwaer2018, Bronzwaer2020,
Dolence2009, Motta2026GPUmonty, NaetheMotta2025, Sharma2023Mahakala, Zhang2026KPolaris,
GrallaLupsasca2020NullGeodesics, GrallaLupsasca2020Lensing, CardenasAvendano2023, AARTCode,
Chang2024, KrangCode, Pandya2016, Marszewski2021, Moscibrodzka2016, Wong2022,
LandiDeglInnocenti1985, Canfield1987, Kahn1950, Salmon2011, Stone2026, AthenaKCode, Zhu2025.

### 6.2 New entries needed

| Key | Reference | Identifier (verified) | Where |
|---|---|---|---|
| Gold2020 | Gold et al. 2020, ApJ 897, 148, "Verification of Radiative Transfer Schemes for the EHT" | doi:10.3847/1538-4357/ab96c6; no arXiv e-print (ref-paper/doi-10.3847_1538-4357_ab96c6/PROVENANCE.md); entry already in docs/references.bib:395 | C28, validation metrics |
| Shcherbakov2008 | Shcherbakov 2008, ApJ 688, 695, "Propagation Effects in Magnetized Transrelativistic Plasmas" | arXiv:0809.0012 (title checked on arxiv.org/abs); doi:10.1086/592326 | C12 ($\rho_Q,\rho_V$) |
| Leung2011 | Leung, Gammie & Noble 2011, ApJ 737, 21, "Numerical Calculation of Magnetobremsstrahlung Emission and Absorption Coefficients" | doi:10.1088/0004-637X/737/1/21 (from the M21 bib, ref-paper/arxiv-2108.10359/tex/transfer_coefficients.bib); arXiv id not verified | C12, C22 |
| Johnson2020 | Johnson et al. 2020, Sci. Adv. 6, eaaz1310, "Universal Interferometric Signatures of a Black Hole's Photon Ring" | arXiv:1907.04329; doi:10.1126/sciadv.aaz1310 (checked on arxiv.org/abs) | C9 |
| Gralla2020Shape | Gralla, Lupsasca & Marrone 2020, PRD 102, 124004, "The Shape of the Black Hole Photon Ring: A Precise Test of Strong-Field General Relativity" | arXiv:2008.03879; doi:10.1103/PhysRevD.102.124004 (checked) | C9 (Johnson-SU profile) — optional |
| Huang2011 (optional) | Huang & Shcherbakov 2011 (basis with $k$ in the $y$–$z$ plane) | not verified | C12 basis remark |
| Wienke1985 (optional) | Wienke 1985 (thermally averaged Compton cross section) | not verified | C24 |
| Prather2023 | already present in the bib (EHT code comparison, arXiv:2303.12004, doi:10.3847/1538-4357/acc586) | — | C28 context |
