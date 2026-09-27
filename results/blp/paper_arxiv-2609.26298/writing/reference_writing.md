# Reference writing patterns: method sections and appendices of GR ray-tracing / transfer code papers

Scope: how ten code papers organise their method and appendix material, introduce notation,
decide derivation depth, and describe software; followed by patterns for the BlackLightPlus
(BLP) method-paper revision. Line numbers refer to the mirrored tex in `ref-paper/<id>/tex/`.
Lengths are tex line counts (a proxy; ~6–8 tex lines ≈ one journal column-paragraph block).

Papers: KP = KPolaris (arxiv-2609.26298, `ms.tex`, 634 lines, structural model);
W22 = Blacklight (arxiv-2203.15963, `blacklight_method.tex`, 819); MG18 = ipole
(arxiv-1712.03057, `ms.tex`, 1391 incl. bbl); R1 = RAPTOR I (arxiv-1801.10452, 1665);
R2 = RAPTOR II (arxiv-2007.03045, 861); S23 = Mahakala (arxiv-2304.03804, 470);
GM26 = GPUmonty (arxiv-2602.13198, 455); D16 = grtrans (arxiv-1602.03184, 2079);
J25 = Jipole (arxiv-2509.07065, 426); A23 = AART (arxiv-2211.07469, 2574).

Rough counts of displayed-math environments / figures / tables:
KP 15/4/8, W22 54/17/5, MG18 48/7/0, R1 37/15/5, R2 63/14/2, S23 29/12/0, GM26 30/6/1,
D16 71/12/0, J25 28/11/0, A23 225/16/3. KP is by far the most equation-sparse; it pushes all
numbers into tables.

---

## 1. Per-paper structure

### 1.1 KPolaris (KP) — structural model

Outline (tex lines):
- §1 Introduction (43–67): ends with a one-paragraph roadmap (l.66).
- §2 Numerical Method (68–181, ~115 lines), five subsections:
  Spacetime and observer (73–79); Ray tracing and parallel transport (81–96);
  Invariant polarized transfer (98–154); Multi-frequency and slow-light transfer (156–163);
  Radiative diagnostics and parameter sensitivities (165–181).
- §3 Models and Inputs (183–206): analytic models, GRMHD readers, units, $R(\beta)$,
  $\Theta_e$, coefficient fits — plasma physics separated from numerics.
- §4 Numerical Accuracy (208–273): convergence and NMSE metric; independent-code (ipole)
  comparison with an NMSE table; accuracy at performance settings.
- §5 Performance (275–354); §6 Selected-Ray Analysis (356–383, one showcase figure);
  §7 Conclusions (385–394); Data and Software (396–410).
- Appendices (412–634, ~220 lines, i.e. about as long as §2–3 together), all
  table- or definition-driven:
  A Numerical Controls (414–476: integration-settings table, memory table);
  B Simulation Samples and Cameras (478–502: inputs table);
  C Performance Test Configurations (504–554: hardware table, timing/accuracy table);
  D Auxiliary Quantities and Emission Contributions (556–590: definitions + closure check);
  E Ray Transfer and Parameter Sensitivities (592–634: display rules, derivative checks).

Symbol introduction:
- Global conventions stated once in the section lead (l.70: signature, $G=c=1$ for rays,
  physical units for coefficients).
- Symbols introduced inline in the defining sentence ("Let $\lambda$ be an affine parameter
  and $k^\mu=dx^\mu/d\lambda$ ...", l.84), or by a trailing "where" clause immediately after
  the display (l.114–115 for $u^\mu$, $u^\mu_{\rm obs}$; l.137 for $\ell_\lambda$).
- Coefficient families are named collectively in one sentence after the matrix
  (l.133: emission / absorption / rotation / conversion) rather than one by one.
- No notation table. Code name through a macro.

Derivation depth:
- Nothing is derived in the text. Equations are stated with citations: geodesic and
  parallel-transport equations (86–95), invariant Stokes vector, local frequency
  (eq:local-frequency), transfer equation and propagation matrix, per-affine-parameter
  coefficient map (eq:affine-coefficients), Strang composition (eq:strang-transfer) with
  its order claim cited to Strang (1968), tangent equation (eq:tangent) obtained by
  "differentiating Eq. (transfer)". The exact constant-coefficient updates are only named
  ($\mathsf F_h$, $\mathsf A_h$, l.151), not written.
- Unit subtleties handled in one clause (l.137: $\ell_\lambda$ "including the conversion
  to the length units"; appendix D l.562–564 restates $\mathcal J_I=j_{\nu,I}/\nu^2$,
  $d\ell=\nu\,d\lambda$ in code normalisation).

Code description:
- No module table, no pseudocode, no file names. The algorithm is described as passes
  (Pass A backward geometry, Pass B forward transfer, l.94) with a storage-cost argument
  (memory ∝ pixels). Implementation facts (Kokkos backends, snapshot streaming window,
  compressed format) are one sentence each (l.94, 160–163).
- Every numerical control lives in tables with units stated in `\tablecomments`
  (e.g. step limits "refer to increments in the affine parameter", l.440–446).

Validation structure:
- Three layers: analytic constant-coefficient tests (one paragraph, l.214);
  image metric defined once (NMSE, eq:image-nmse, with citation to Prather+2023);
  cross-code table (tab:ipole-nmse) with all four Stokes components and a residual figure;
  accuracy–cost figure at the adopted settings.
- Closure/consistency checks go to the appendix with a single quantitative number each
  (l.586–589, l.614–617, l.627–633).

What works: very short method prose; conventions up front; a clean split between
"method" (what equation), "models" (what physics), "controls" (what numbers, in
appendix tables). Diagnostics are defined by an equation plus one sentence of meaning.
What it omits: the actual segment solution, frame-rotation formulas, and camera formulas —
a reader cannot re-implement the code from the paper alone.

### 1.2 Blacklight (W22)

Outline: Introduction (107–121); Integration Algorithms (123–446, ~320 lines, the core):
coordinate definitions (CKS metric, $f$, $l_\alpha$, SKS map, normal frame, ~50 lines before
the first subsection), Camera Definition (177–256), Geodesics (258–335), Unpolarized
Radiation (337–360), Polarized Radiation (362–394), Sampling (396–404), Coefficients
(406–446); Test Problems (447–513); Code Features (515–755: data cuts, flat spacetime, slow
light, adaptive ray tracing, nonthermal electrons, auxiliary images, false/true colour);
Performance (757–793); Code Philosophy (795–819). No appendices.

Symbol introduction:
- An explicit index-convention paragraph after the metric (α,β,γ,δ spacetime and a,b space
  for CKS; μ,ν,i,j for SKS; n,p,q for the normal frame; l.~146–174).
- Every derived quantity is defined in the display that introduces it, often as a
  `subequations` block listing auxiliaries ($f$, $l_\alpha$, $r$, $R$ in eq:cks_aux).
- Physical variables collected in one sentence in Sampling (l.~402) with their definitions
  ($n_e$, $\Theta_e$, $B=(b_\alpha b^\alpha)^{1/2}$, $\sigma$, $\beta^{-1}$).

Derivation depth:
- Implementable formulas in full: camera frame metric, line-of-sight/vertical/horizontal
  vectors by Gram–Schmidt with the Levi-Civita cross product and roll angle ψ, pixel
  offsets $a,b=(\{i,j\}-N/2+1/2)w/N$, both camera types, the null-completion rule for
  $k^t$ including the ergosphere root choice, and both frequency normalisations
  ($-nk_\alpha u^\alpha=\nu$ vs $-nk_t=\nu$) (l.177–256).
- Integrator Butcher tableaux printed as tables (tab:rk2, tab:rk4, tab:dp incl. Shampine
  midpoint weights). Step rule $\Delta\lambda=f_{\rm step}(r-r_{\rm h})$.
- Unpolarized segment solution written with its three-case form and the `expm1` guard and
  $\Delta\tau_{\max}=100$ (l.350–360).
- Polarized transfer: only summarised (ipole scheme, Strang split, predictor–corrector
  table); the full constant-coefficient solution is cited, with a footnote reporting a
  typo in MG18 eq. 24 (l.~393). Coefficients: distribution functions and $T_e$ prescriptions
  written; fits cited to M21.

Code description: prose plus the "Code Philosophy" section (five main classes, no
external dependencies, public-domain, one version). Adaptive ray tracing described with a
schematic of blocks (fig:adaptive_grid) and all four refinement indicators as equations
(l.581–628). Performance gives a time breakdown by component (75% transfer, 13% sampling,
12% geodesics) and a pinned commit hash in a footnote (l.~760).

What works: a reader can reproduce the camera, integrator, and unpolarized solver exactly;
feature sections are self-contained; the percentage cost breakdown is informative.

### 1.3 ipole (MG18)

Outline: Introduction (150–238); Governing Equations (239–381, ~140 lines); Numerical
Methods (383–488) with an explicit numbered integration scheme (466–488); Tests (490–756:
transport-only tests in nontrivial geometry, source+transport, comparison with grtrans in
Minkowski and in a GRMHD flow); Summary; Appendix A special solutions (1233–~1330: Faraday
rotation alone; emission+absorption alone with eigenbasis).

Symbol introduction: "≡" definitions inline ("$I_\nu\equiv$ specific intensity, $\nu\equiv$
frequency", l.~243); Stokes sign conventions stated in prose with IEEE definition and a
citation (l.~290–295); coefficient count stated ("11 transfer coefficients") and positivity
constraints written as an inequality.

Derivation depth: builds up from the non-relativistic scalar equation → covariant scalar
equation → polarized flat → coherency tensor $N^{\alpha\beta}$ with the four Pauli-like
matrices $m_A$ written out and the $S\leftrightarrow N$ maps both ways (eq:StoN, eq:NtoS) →
covariant tensor equation expanded in a coordinate basis (eq:covtrans2). The
constant-coefficient solution is given in the $O_{AB}$/$P_{AB}$ form (eq:fullsol and the
$P_{AB}$ multline, l.450–460) with explicit auxiliary definitions ($f_1$, $f_2$) and
typo notes on LDI85, Peraiah, and D16 (l.427, l.439). Special cases derived in the
appendix with intermediate eigenvectors "rather than simply stating the result".

Code description: a 7-step numbered list (l.472–484) is the entire algorithm description;
it is precise enough to implement. No file names.

What works: the numbered step list; the typo notes (increase trust); the appendix special
cases double as test solutions.

### 1.4 RAPTOR I (R1) and RAPTOR II (R2)

R1 outline: Equations of motion for light rays (239–493: geodesic equation from basics,
RK4 with all 8 update coefficients written, Verlet, adaptive step rule, Bardeen camera
initial conditions, coordinate systems); Radiative transfer (495–735); Code verification
(737–1137); Time-dependent RT in GRMHD (1139–1348); appendices: Coordinate
Transformations BL–KS and KS–MKS (1443–1554), Code performance with hardware and setup
tables (1556–1665).

R2 outline: Polarized RT in curved spacetime (142–360: Stokes parameters, polarization
four-vector, spacetime propagation, tetrad construction, plasma interaction);
Implementation (362–423: integration strategy, plasma scheme with explicit/implicit
switch, vacuum scheme); Code Verification (425–597); appendices: Stiffness analysis
(642–692), Implicit trapezoid integrator (694–776), Thermal coefficients (778–).

Symbol introduction: textbook build-up (R1 l.249–307 starts from Newtonian free fall);
intermediate scalars for the tetrad defined in a `subequations` list before use
($d^2,\beta,\omega,\mathcal C,\mathcal N$; R2 l.~296–305); Levi-Civita sign convention
stated where used (R2 l.322). "Underlined index" notation explained once (R2 l.420).

Derivation depth: high and explicit — every RK stage written out; the tetrad written in
closed form; the stiffness criterion derived from the characteristic polynomial of the
transfer matrix (biquadratic in $\alpha_I-\Lambda$) and the RK4 stability polynomial
(R2 l.642–692); the implicit step reduced to a 4×4 linear solve in an appendix.

Code description: parallelisation named (OpenMP/OpenACC), speed-up defined as an equation
(R1 l.1576), hardware and setup tables in the appendix. No module table.

What works: the stiffness appendix is a good model for a short, fully derived
"why this solver" appendix. What to avoid: the textbook build-up and writing every RK
stage cost space without informing a specialist reader.

### 1.5 Mahakala (S23)

Outline: Introduction; Algorithm (86–178, one section, no subsections) with a 3D data-flow
figure (fig2, l.96–101: image plane → trajectories → emission along rays) and an
ingoing/outgoing-coordinate comparison figure; Synthetic images (180–272: RT, emissivity);
Tests from the literature (274–371); Applications; Summary.

Symbols: conventions stated after the first metric display (l.~121: signature, $c=G=1$,
Greek indices 0–3); image-plane map given as three explicit equations
($x=-y'\cos i+d\sin i$, …; l.128–136) — the source of BLP's camera map.

Derivation depth: moderate; rewrites the geodesic equation in metric-derivative form
following GRay2 and in coordinate-time form, each as one equation with a one-line reason
(operation count, horizon blow-up of $dt/d\lambda$). Step rule $\mathcal S=r/\mathcal C$
with a figure justifying the default $\mathcal C$ (l.160–170).

What works: every modelling choice has a one-sentence reason and, where numeric, a figure
justifying the default.

### 1.6 GPUmonty (GM26)

Outline: Introduction; Governing Equations and Numerical Methods (67–235: superphoton
creation, emissivity, sampling procedure, geodesic equation, covariant RT, absorption,
scattering); Implementation of the GPU-Accelerated Algorithms (237–255, ~20 lines of
prose); Tests (257–359: thin synchrotron sphere, optically thick scattering sphere, GRMHD
comparison with igrmonty); GPU benchmark (361–430: runtime and speed-up table, kernel
times, Nsight metrics); Conclusions.

Symbols: each defined in a trailing "where"/"Here" clause after its display (weights
l.~73–79; $j_\nu$ and $\nu_s$ l.~86–99 with every constant listed).

Derivation depth: equations stated, not derived (weights, sampling densities, rejection
tests, hot cross section, KN cross section with small-ε limit); the sampling algorithm is
written as a sequence of inequalities on uniform random numbers — implementable.

Code description: kernel-level prose — five kernels (creation, sampling, tracking,
scattering, recording; l.244), the parallel pattern of each (stride-based over zones vs
dynamic one-photon-per-thread), the replacement of grmonty's recursion by an array of
scattered photons processed after the primaries (l.252), memory-driven batching (l.254),
RNG named with its period (l.242). Benchmark section reports per-kernel time shares and
profiler metrics.

What works: one paragraph per GPU design decision, each tied to a GPU constraint
(stack size, atomics, warp divergence). BLP's Monte Carlo description should follow this.

### 1.7 grtrans (D16)

Outline: Methods (164–848: ray tracing via geokerr; parallel transport of the polarisation
basis via the Walker–Penrose constant; transformation to the fluid frame; transfer
equation; coefficients; fluid models; integration — LSODA and quadrature solutions);
Tests and examples (850–1188); Code structure and performance (1190–1385: convergence,
scaling, Code organisation with one subsubsection per module — geodesics, fluid models,
transfer coefficients, other modules, driver, Python interface); Discussion;
Appendices A–D: coefficient fits (1440–1697), Faraday coefficients (1699–1870), analytic
test solutions (1872–1983), closed form of $\mathbf O(s,s')$ (1985–).

Symbols: long "where" blocks after each display (Walker–Penrose, l.~240–260, lists the full
$k^\mu$ in BL form).

Derivation depth: very high; coefficients re-derived in appendices; analytic test solutions
supplied for every integration test.

Code description: the only paper with an explicit per-module "Code organisation"
subsection (l.1268–1385): what each module owns, how to add a model (template files), the
driver's global objects, the Python/f2py interface. Good template for a BLP "software
structure" appendix, but without Fortran-specific detail.

### 1.8 Jipole (J25)

Outline: Governing Equations (59–112) incl. Derivative Evolution (84–112); Numerical
Methods (114–173: geodesic integration, intensity integration, conjugate gradient);
Validation (175–301: thin disk, analytic models, AD vs finite differences); Data analysis
application (303–402).

Derivation depth: the tangent (derivative) equations are derived by the chain rule, with
the partials of the right-hand sides named $f_1,f_2$ and their AD evaluation stated
(l.84–112). Numerical update formulas written with sign explanations ("minus signs arise
because we integrate backwards", l.~126). Unit conversion of the step written explicitly
($d\lambda_{\rm cgs}=d\lambda\,L_{\rm unit}h/(m_ec^2)$).

What works: an AD section that writes exactly which Jacobian products are formed; BLP's
differentiation paragraph should be of this form.

### 1.9 AART (A23)

Outline: Introduction with an explicit "Summary" subsection mapping sections to figures
(194–221); Theoretical framework (223–743); Equatorial emission model (745–1402);
Applications; Ray tracing requirements (resolution, field of view); Photon ring shape;
Future outlook; Appendices: explicit geodesic integrals (1823–1965), accretion-flow
four-velocities by regime (1967–2367), Matérn covariance (2369–).

Symbols/conventions: stated at the end of the summary (l.220–221) including a pointer to an
external convention list for elliptic integrals.

Derivation depth: main text states results; the appendix lists "all formulas needed" to
reproduce the images, explicitly deferring derivations to Gralla & Lupsasca (l.1827) —
a formula-reference appendix, not a derivation appendix. Four-velocity appendix is split by
physical regime (inside/outside ISCO; geodesic/non-geodesic).

What works: the "formulas needed to reproduce" appendix model is appropriate for BLP's
analytic tracer and equatorial source (ported from AART), with derivations cited.

---

## 2. Cross-paper comparison

| Aspect | KP | W22 | MG18 | R1/R2 | S23 | GM26 | D16 | J25 | A23 |
|---|---|---|---|---|---|---|---|---|---|
| Conventions paragraph | method lead | after metric + index paragraph | inline in prose | per section | after metric | none | per section | none | end of summary |
| Notation table | no | no | no | no | no | no | no | no | no |
| Symbol intro style | inline + "where" | subequations lists | "≡" inline | subequation lists | "where" | "where"/"Here" | long "where" | "where" | inline |
| Segment solution | named only | unpolarized full, polarized cited | full ($P_{AB}$) + special cases | RK4/implicit, stiffness derived | — | — | full closed form (app.) | 2-point averaged exponential | — |
| Camera formulas | prose | full | prose (pinhole) | Bardeen $k_\mu$ | full map | — | BL pixels | — | full |
| Algorithm listing | passes in prose | prose + Butcher tables | 7 numbered steps | RK stages | prose + 3D figure | kernel paragraphs | per-module subsections | update equations | prose |
| Parameter/controls tables | appendix tables | none | none | appendix | none | benchmark table | none | none | tables |
| Cross-code validation | ipole NMSE table | test problems | grtrans | ipole/BHOSS etc. | literature tests | igrmonty | analytic | ipole thin disk | — |

Observations:
1. No paper uses a notation table; all rely on a conventions paragraph plus inline or
   "where" definitions. KP and MG18 handle convention mapping by prose statements of the
   camera basis, EVPA definition, and $V$ handedness (KP l.78, MG18 l.~290–295).
2. Derivations, when given, go either in the main text (MG18 coherency tensor, J25 tangent
   equations) or in a single-purpose appendix (MG18 special solutions, R2 stiffness,
   D16 analytic tests, A23 formula list). None interleaves long derivations with results.
3. Algorithms are best communicated by either a numbered step list (MG18) or a
   pass/kernel decomposition with a one-sentence reason per design choice (KP, GM26).
4. Controls and hardware belong in appendix tables with units and meaning in the table
   comments (KP A–C, R1 app.), keeping the main text free of parameter values.
5. Honesty devices that raise trust: pinned commit hashes (W22), typo notes on prior
   literature (MG18, W22), explicit sign-convention maps against other codes (MG18
   footnote on $\rho_U$ vs D16).

---

## 3. Patterns to follow for the BLP revision

### 3.1 Main-text method section (keep KP's compactness)

- Keep the current conventions paragraph (paper/sections/method.tex l.11–16) as the single
  place for signature, units, $k$ orientation, invariants, and Stokes/EVPA conventions
  (KP l.70, S23 l.121). Add one sentence stating the index conventions (Greek spacetime,
  Latin space, parenthesised tetrad, $A,B\in\{I,Q,U,V\}$ Stokes), as W22 does.
- Every display gets its symbols either in the sentence before ("Let …") or a trailing
  "where" clause (KP l.84, l.114, l.137). Each coefficient family named collectively once
  (KP l.133).
- State equations in the main text; move every derivation, closed-form segment solution,
  frame formula, and numerical control to appendices, referenced by label.
- For each design choice give a one-sentence reason (S23 step-size default; GM26 GPU
  constraints). BLP already does this in places (AD for Christoffels, fixed-size arrays).

### 3.2 Proposed appendix layout (derivation + code-description appendices)

Map BLP components to appendix subsections; each subsection has (i) the equations needed to
re-implement, (ii) a short derivation or a citation to where it is derived, (iii) the
source modules, (iv) what differs from the code it follows.

| Appendix | Content | Model paper for form | BLP sources |
|---|---|---|---|
| A. Notation and conventions | table of conventions vs ipole/Blacklight/grtrans/M21/KP, incl. $p=-k$ storage and Stokes basis; symbol list only if the journal allows | MG18 footnote style, conventions.md | conventions.md |
| B. Spacetimes and metric derivatives | CKS Kerr ($f$, $l_\mu$, $r(x,y,z)$), exact inverse, superposed metric, ADM reconstruction $g_{\mu\nu}(\alpha,\beta^i,\gamma_{ij})$, $\gamma_{ij}=\tilde\gamma_{ij}/\chi$, Catmull–Rom $C^1$ stencil; AD of $g^{\mu\nu}$ | W22 eq:cks_aux block | src/geodesics/{metrics,spacetimes,numerical_spacetime}.jl, src/simulation/athenak_z4c.jl |
| C. Geodesic integration | Hamiltonian derivation; $k_t$ conservation; orientation $(\lambda,p)\to(-\lambda,-p)$; RK4 tableau reference and step rule $h=(r-r_h)/C$; Vern7 for capture maps; termination rule and omitted-length bound | W22 Geodesics + tables; S23 step rule | src/geodesics/integrator.jl, src/gpu_metric.jl |
| D. Camera, tetrads, and Stokes basis | plane-parallel map, $k_\mu$ at camera, Bardeen check $L_z=-\alpha\sin i$; camera tetrad; fluid tetrad (Gram–Schmidt, degenerate fallback); Q–U rotation $R_n$ | W22 Camera Definition (full formulas); R2 tetrad list | src/geodesics/camera.jl, src/radiation/tetrads.jl |
| E. Polarized segment solution | constant-coefficient solution written explicitly in the $U$-free frame; limiting cases (Faraday only, emission–absorption only) as tests; typo/sign notes vs MG18/D16 | MG18 eq:fullsol + App. A; R2 stiffness app. | src/radiation/transfer.jl |
| F. Transported-frame march | derivation of Eq. (reverse) (affine composition, $G_n$), relation to ipole's $N^{\mu\nu}$ transport, why coefficients at the RK step start keep per-ray state constant | MG18 7-step list | src/radiation/{reverse_march,integrator}.jl, src/gpu_pol.jl |
| G. Plasma model and coefficients | units ($\rho_{\rm unit}$, $B$), $R(\beta)$, $T_e$ splits, distributions, M21/P16 fits with basis statement and $j_Q<0$; D16 sign map | W22 Coefficients; KP §3 | src/radiation/coefficients.jl, src/simulation/athenak_reader.jl |
| H. Analytic Kerr tracer and equatorial source | formula-reference appendix: $(\alpha,\beta)\to(\lambda,\eta)$, radial/angular Mino integrals, image order $n$, redshift $g$ and flow four-velocities, with derivations cited to GL20/A23 | A23 App. A–B | src/geodesics/{analytic_kerr,lensing_bands,kerr_orbits}.jl, src/radiation/equatorial_source.jl, src/gpu_analytic.jl |
| I. GPU kernels and multi-GPU pool | one paragraph per kernel (unpolarized fused, polarized fused, AMR, mixed precision, AD), each with the GPU constraint it answers; pool/queue design and the bitwise-determinism argument; table: kernel → file → state per thread | GM26 §3 prose; D16 code organisation | src/gpu_*.jl |
| J. Differentiation | forward-mode dual seeding, which Jacobian products are formed, Enzyme check; compare Jipole's tangent equations | J25 Derivative Evolution | src/gpu_image_ad.jl |
| K. Monte Carlo | superphoton weights, zone sampling, absorption weight decay, hot cross section, Canfield sampler, Kahn bias, Philox key layout (ancestry), fixed-point exact sums and why they give bitwise invariance | GM26 §2–3 | src/montecarlo/*.jl |
| L. Numerical controls and test configurations | tables: step constant $C$, horizon backstop, $E_c$, pixel counts, AMR block/thresholds, MC photon counts; hardware table | KP App. A–C | src/render.jl, src/inputs/parameters.jl, docs/src/benchmarks.md |

Ordering rule (KP, A23): appendices follow the order in which the main text references
them; each appendix opens with one sentence saying which main-text equation it supports.

### 3.3 Derivation depth per item

- Derive in full (short, in appendix): Hamiltonian geodesic form from $H=\frac12 g^{\mu\nu}k_\mu k_\nu$;
  $k_t$ conservation; invariance of $I_\nu/\nu^3$ and coefficient scalings; frequency ratio;
  the affine composition behind Eq. (reverse); the camera map and Bardeen identity;
  the omitted-affine-length bound at termination; MC estimator normalisation.
  These are 3–10 lines each (J25/R2 depth).
- Write in full but cite the derivation: the constant-coefficient polarized solution
  (LDI85/MG18), coefficient fits (M21/P16), Kerr elliptic-integral solutions (GL20/A23),
  hot cross section and electron sampler (Canfield87/D09/GM26).
- State only: RK4 and Vern7 tableaux (cite), Philox algorithm (cite Salmon11), NMSE metric.

### 3.4 Code description devices

- A kernel/module table (component → file:function → CPU/GPU → per-thread state → follows
  which code, what changed). D16's module subsections and GM26's kernel paragraphs are the
  models; BLP's figure fig:pipeline already provides the data-flow view (S23 fig2 analogue).
- A numbered step list for the fused polarized march (MG18 l.472–484 style), 6–8 steps.
- Pin the code version (commit hash) where performance or bitwise claims are made (W22
  footnote l.~760).
- Keep file names out of the main text; put them in the appendix table only (none of the
  reference papers puts file paths in the method section).

### 3.5 Validation presentation

- Keep KP's three layers: analytic/limit tests (one paragraph + appendix solutions),
  cross-code image metric table (all four Stokes, NMSE defined once with citation),
  and consistency/closure checks with one number each in an appendix.
- CPU–GPU agreement and device-count bitwise invariance are BLP-specific checks; present
  them as closure checks in the KP appendix style (definition, single number, pointer to
  test file).

### 3.6 Things to avoid

- Textbook build-ups (R1 geodesic section from Newtonian free fall).
- Writing every RK stage (R1 l.332–377, R2 l.404–423) — cite the tableau instead.
- Naming a solver without writing it when the paper claims it as a contribution (KP l.151
  names $\mathsf F_h$, $\mathsf A_h$ only); BLP claims the exact segment + transported-frame
  march, so the appendix must write both.
- Mixing code-feature showcases into the method section (W22 Code Features is long and
  figure-heavy; BLP should keep features to the results section).
- Unstated sign conventions for $Q$ basis and $V$ (MG18 and KP both spend explicit
  sentences on this; BLP's $j_Q<0$ basis statement at method.tex l.59–60 must stay).
