# Derivations for every methodology equation in `paper/sections/method.tex`

Scope: each displayed or inline equation of `paper/sections/method.tex` (state of branch
`paper`, 80 lines), identified by its label or, for unlabelled inline relations, by an id
`Mk` and its line. Every derivation is written in the conventions of
`results/blp/paper_arxiv-2609.26298/conventions.md`: signature $(-,+,+,+)$, $G=c=M=1$ for
geometry, cgs for plasma and radiation, CKS coordinates $x^\mu=(t,x,y,z)$, received wave
covector $k_\mu$ with $k_t=-1$ at the camera, ray state $p_\mu=-k_\mu$ integrated to
increasing $\lambda$, invariant Stokes vector $\boldsymbol{\mathcal S}=(I_\nu,Q_\nu,U_\nu,V_\nu)/\nu^3$,
invariant coefficients $j_A=j_{\nu,A}/\nu^2$, $\alpha_A=\nu\alpha_{\nu,A}$, $\rho_A=\nu\rho_{\nu,A}$,
Stokes referred to $(e_{(2)},-e_{(1)})$ of each tetrad, IEEE/IAU signs.

Source keys (mirrored tex under `ref-paper/`; equation numbers are the journal numbering
recovered from the tex; where a source uses `eqnarray` or `cases` blocks the number is
given together with the tex line so it can be checked):
MG18 = ipole (1712.03057, `ms.tex`), W22 = Blacklight (2203.15963), D09 = grmonty
(0909.0708), GPM = GPUmonty (2602.13198), S23 = Mahakala (2304.03804), KP = KPolaris
(2609.26298), B20 = RAPTOR II (2007.03045), D16 = grtrans (1602.03184), M21 = Marszewski+
2021 (2108.10359, `Main.tex`), P16 = Pandya+2016 (1602.08749), GL20a = Gralla & Lupsasca
"Null geodesics of the Kerr exterior" (1910.12881, `KerrGeo.tex`), GL20b = "Lensing by Kerr
black holes" (1910.12873, `KerrLensing.tex`), CA23 = AART (2211.07469), NM25 = Jipole
(2509.07065), LD85 = Landi Degl'Innocenti & Landi Degl'Innocenti 1985 (no arXiv version;
its segment solution is reproduced as MG18 Eq. 24, which is the version checked here).
Code references are `file:line` in `src/` at HEAD of branch `paper`.

## 0. Index of equations in method.tex

| id | line | relation | section below |
|---|---|---|---|
| M1 | 12 | signature, units, $\rg=GM/c^2$ | §1 |
| M2 | 13 | $\gcov=\eta_{\mu\nu}+f\,l_\mu l_\nu$ (CKS Kerr) | §2.1 |
| M3 | 14 | $k^\mu=\dd x^\mu/\dd\lambda$, $k_t=-1$, $\nu=\nuobs(-k_\mu u^\mu)$ | §4 |
| M4 | 15 | $\boldsymbol{\mathcal S}=(I_\nu,Q_\nu,U_\nu,V_\nu)/\nu^3$, $j_A=j_{\nu,A}/\nu^2$, $\alpha_A=\nu\alpha_{\nu,A}$, $\rho_A=\nu\rho_{\nu,A}$ | §5.1 |
| M5 | 16 | IEEE/IAU Stokes, $\chi=\tfrac12\arg(Q+iU)$ | §6.3 |
| M6 | 21 | $g^{\mu\nu}=\eta^{\mu\nu}-f\,l^\mu l^\nu$ | §2.1 |
| M7 | 21 | superposed metric $\gcov=\eta_{\mu\nu}+\sum_A f_A l^A_\mu l^A_\nu$ | §2.2 |
| M8 | 22 | ADM fields, $\gamma_{ij}=\tilde\gamma_{ij}/\chi$, $C^1$ Catmull–Rom | §2.3 |
| M9 | 24 | $H=\tfrac12 g^{\mu\nu}k_\mu k_\nu=0$ | §3.1 |
| **eq:geodesic** | 25–29 | Hamilton's equations | §3.1 |
| M10 | 30 | $k_t$ conserved | §3.2 |
| M11 | 30 | step $h=(r-\rhor)/C$, $C=80$ | §3.4 |
| M12 | 32 | camera pixel map $x^i(\alpha,\beta)$, $\hat\beta$, $\hat\alpha$ | §6.1 |
| M13 | 33 | $\hat n$, right-handed $(\hat\alpha,\hat\beta,\hat n)$, state $-k_\mu$, $\lambda'=-\lambda$ | §3.3, §6.1 |
| M14 | 34 | $L_z=xk_y-yk_x=-\alpha\sin i$ (Bardeen) | §6.2 |
| M15 | 35 | camera tetrad, Stokes referred to $(e_{(2)},-e_{(1)})$ | §6.3 |
| M16 | 37 | $(\alpha,\beta)\mapsto(\lambda,\eta)$, Mino-time inversion, $(r_s,\phi_s,t_s)$ | §3.5 |
| **eq:transfer** | 45–55 | $\dd\boldsymbol{\mathcal S}/\dd\lambda_{\rm c}=\boldsymbol j-\mathbf K\boldsymbol{\mathcal S}$ | §5.2 |
| M17 | 56 | $\lambda_{\rm c}=\lambda\rg/\nuobs$, $\dd\ell=\nu\,\dd\lambda_{\rm c}$ | §5.1 |
| M18 | 56 | $\rho_V$ Faraday rotation, $\rho_Q$ Faraday conversion | §5.5 |
| M19 | 56 | $S_\nu=\nuobs^3\boldsymbol{\mathcal S}$ | §5.1 |
| M20 | 57 | exact constant-coefficient segment solution | §5.3 |
| M21 | 58 | transported vertical vector and four-velocity; rotation $R_n$ by $2\times$ angle | §7.1–7.2 |
| M22 | 59 | fluid tetrad; $j_U=\alpha_U=\rho_U=0$; $Q>0$ in the $k$–$B$ plane | §6.4 |
| M23 | 60 | $j_Q<0$ in the P16/M21 basis; grtrans basis flips all $Q,U$ coefficients | §5.6 |
| M24 | 66 | $\boldsymbol{\mathcal S}_{\rm out}=M_n\boldsymbol{\mathcal S}_{\rm in}+b_n$ | §5.3, §7.3 |
| **eq:reverse** | 67–71 | reverse accumulation $G_n$ | §7.3 |
| M25 | 72 | coefficients at the start of each step | §7.4 |
| M26 | 74 | stop at $r-\rhor<10^{-3}$ or $-k_\mu n^\mu>E_{\rm c}=10^2$; $\Delta\lambda\lesssim\alpha_{\rm h}/(\kappa E_{\rm c})$ | §8 |
| M27 | 75 | dual-number derivative of the unpolarized image | §9 |
| M28 | 79 | superphoton $(x^\mu,k_\mu,w)$, $E=-k_t$ in $m_ec^2$, Kirchhoff weight loss | §10.1–10.3 |
| M29 | 80 | $\sigma_{\rm h}(\varepsilon,\Theta_e)$, Canfield sampler, Kahn exponential transform | §10.4 |
| M30 | 80 | Philox streams keyed by ancestry; exact bin sums | §10.5–10.6 |

Count: 3 labelled displays and 30 inline relations, 33 in total. Each has a derivation below.
Items flagged **[paper]** are statements in method.tex that the derivation shows to be
imprecise or incomplete. They are collected in §11.

---

## 1. Units and conventions (M1)

- Geometric units: $G=c=M=1$. A code length $x$ is $x\,\rg$ in cm, with $\rg=GM/c^2$, and a
  code time $t$ is $t\,\rg/c$ in s. The code constant is `L_unit` $=\rg$
  (`src/radiation/integrator.jl:8`).
- Plasma and radiation are cgs. The code field $b^\mu$ is Heaviside–Lorentz, with
  $B_{\rm cgs}=\sqrt{b^2}\,c\sqrt{4\pi\rho_{\rm unit}}$, $\rho_{\rm unit}=\mathcal M/\rg^3$ and
  $\beta=2p_{\rm gas}/b^2$ (conventions.md, Units row).
- Signature $(-,+,+,+)$. It matches MG18, W22, D09, S23 and KP, so no signs change on import.

---

## 2. Spacetimes

### 2.1 Kerr in CKS form and its exact inverse (M2, M6)

**Metric.** Following S23 Eqs. (1)–(4) (`ref-paper/arxiv-2304.03804/tex/main.tex:104–119`),
$$g_{\mu\nu}=\eta_{\mu\nu}+f\,l_\mu l_\nu,\quad
f=\frac{2Mr^3}{r^4+a^2z^2},\quad
l_\mu=\Big(1,\ \frac{rx+ay}{r^2+a^2},\ \frac{ry-ax}{r^2+a^2},\ \frac{z}{r}\Big),$$
with the Kerr–Schild radius $r$ as the positive root of
$x^2+y^2+z^2=r^2+a^2(1-z^2/r^2)$. Solving that quadratic in $r^2$ gives
$$r^2=\tfrac12\Big[R^2-a^2+\sqrt{(R^2-a^2)^2+4a^2z^2}\Big],\qquad R^2=x^2+y^2+z^2.$$
The code evaluates the root with `hypot` to avoid cancellation (`src/geodesics/metrics.jl:52–74`, `_ks_term`).
- *Differences from S23:* S23 writes $l_0=\pm1$. The code fixes $l_0=+1$, which is ingoing
  Kerr–Schild, regular on the future horizon. This is also the reason a backward-traced
  ray approaches the future horizon only asymptotically (§8).
- *Symbols to define in the paper:* $\eta_{\mu\nu}={\rm diag}(-1,1,1,1)$, $f$, $l_\mu$, $a$, and $r$.
  Note that $r$ is not the Euclidean radius (symbols.md §8).

**$l$ is null.** Raise with $\eta$: $l^\mu=\eta^{\mu\nu}l_\nu=(-1,l_x,l_y,l_z)$. Then
$$\eta^{\mu\nu}l_\mu l_\nu=-1+\frac{(rx+ay)^2+(ry-ax)^2}{(r^2+a^2)^2}+\frac{z^2}{r^2}
=-1+\frac{x^2+y^2}{r^2+a^2}+\frac{z^2}{r^2}=0.$$
The middle step uses $(rx+ay)^2+(ry-ax)^2=(r^2+a^2)(x^2+y^2)$, and the last uses the
defining relation for $r$ divided by $r^2+a^2$.

**Inverse.** Try $g^{\mu\nu}=\eta^{\mu\nu}-f\,l^\mu l^\nu$:
$$g_{\mu\lambda}g^{\lambda\nu}=(\eta_{\mu\lambda}+f l_\mu l_\lambda)(\eta^{\lambda\nu}-f l^\lambda l^\nu)
=\delta_\mu^\nu+f l_\mu l^\nu-f l_\mu l^\nu-f^2 l_\mu (l_\lambda l^\lambda) l^\nu=\delta_\mu^\nu .$$
The last term vanishes because $l$ is null. No matrix inverse is needed
(`metrics.jl:3–7`; W22 uses the same form).

**Determinant.** $\det(\eta+f\,l\,l^{\sf T})=\det\eta\,(1+f\,l^\mu l_\mu)=-1$ by the
matrix-determinant lemma, so $\sqrt{-g}=1$. The fluid tetrad's Levi-Civita cross product
therefore needs no $\sqrt{-g}$ factor in single CKS. Off CKS the code renormalises
$e_{(1)}$ (`src/radiation/tetrads.jl:7–9`, `renorm_e1`).

**Lapse on the horizon (used in §8).** From $g^{tt}=\eta^{tt}-f(l^t)^2=-(1+f)$, the lapse is
$\alpha=(-g^{tt})^{-1/2}=(1+f)^{-1/2}$. On the horizon $r_+^2+a^2=2r_+$, so
$f=2r_+^3/(r_+^4+a^2z^2)\ge 2r_+/(r_+^2+a^2)=1$, and hence $\alpha_{\rm h}\le 1/\sqrt2$. Equality holds on the poles.

### 2.2 Superposed Kerr–Schild binary (M7)

$g_{\mu\nu}=\eta_{\mu\nu}+\sum_{A=1}^{N}f_A l^A_\mu l^A_\nu$. Each hole's $(f_A,l^A)$ is §2.1
evaluated at the offset from its centre, with mass $m_A$ and spin $a_A$. The product
$(\eta+\sum_A f_Al^Al^A)(\eta^{-1}-\sum_B f_B l^B l^B)$ leaves the cross terms
$-f_Af_B\,l^A_\mu(l^A\!\cdot l^B)l^{B\nu}$ with $A\ne B$. Since $l^A\!\cdot l^B\ne0$, the
single-hole inverse no longer holds.

- **Code, exact form.** The Woodbury identity is applied with
  $L=[l^1\cdots l^N]$ (covariant, $4\times N$), $U=\eta^{-1}L$ (raised), and $F={\rm diag}(f_A)$:
  $$g^{-1}=\eta^{-1}-U\,(1_N+F\,U^{\sf T}L)^{-1}F\,U^{\sf T}.$$
  Only an $N\times N$ solve is needed. The diagonal of $U^{\sf T}L$ vanishes because each
  $l^A$ is null, and $N=1$ gives back M6 exactly (`src/geodesics/spacetimes.jl:63–95`).
- **[paper/docs]** The documentation (`docs/src/method.md`, Spacetimes) still says that "the
  full $4\times4$ inverse is taken". The code does a $2\times2$ Woodbury solve for a binary.
- *Validity:* the superposition is not a vacuum solution. The error is of order
  $m_Am_B/(d\,r)$ in the overlap region, so it approximates a wide binary. The paper already
  says "approximates".

### 2.3 ADM slice and $C^1$ interpolation (M8)

- **3+1 assembly.** $g_{tt}=-\alpha^2+\beta_i\beta^i$, $g_{ti}=\beta_i=\gamma_{ij}\beta^j$,
  $g_{ij}=\gamma_{ij}$, with inverse $g^{tt}=-1/\alpha^2$, $g^{ti}=\beta^i/\alpha^2$ and
  $g^{ij}=\gamma^{ij}-\beta^i\beta^j/\alpha^2$ (standard ADM).
  - *Symbol clash:* $\alpha$ is also a screen coordinate and an absorptivity. symbols.md §7
    proposes $N,N^i$ for the lapse and shift.
- **Z4c.** AthenaK stores the conformal metric $\tilde\gamma_{ij}$ and
  $\chi=(\det\gamma)^{-1/3}$, so $\gamma_{ij}=\tilde\gamma_{ij}/\chi$
  (`src/simulation/athenak_z4c.jl:8`, with $\chi$ floored at $10^{-12}$).
- **Why $C^1$.** The geodesic force is $-\tfrac12\partial_i g^{\alpha\beta}k_\alpha k_\beta$
  (§3.1).
  - With a trilinear (only $C^0$) interpolant, $\partial_i g$ jumps at every cell face. The RK4
    local error then drops from $O(h^5)$ to $O(h^2)$ on steps that straddle a face, because
    the RHS is discontinuous inside the step.
  - The Catmull–Rom weights for nodes $x_{-1},x_0,x_1,x_2$ at fraction $t\in[0,1]$ are
    $$w=\tfrac12\big(-t^3+2t^2-t,\ 3t^3-5t^2+2,\ -3t^3+4t^2+t,\ t^3-t^2\big).$$
    These weights sum to 1. At $t=0$ they give $w=(0,1,0,0)$, and the derivative there is
    $(f_1-f_{-1})/2$, the centred difference; the same holds at $t=1$. Neighbouring cells
    therefore share value and slope at every node, which makes the interpolant $C^1$ and
    the Christoffel symbols continuous (`src/geodesics/numerical_spacetime.jl:64–74`).
  - The second derivative still jumps, so the force is continuous but its gradient is not.
    RK4 on such a field converges at second order across faces.
  - **[paper]** The sentence "Christoffel symbols obtained by AD are continuous" is correct
    and should stay. The paper should not claim fourth-order convergence for ADM runs.

---

## 3. Geodesics

### 3.1 Hamiltonian form (M9, eq:geodesic)

Take the super-Hamiltonian $H(x,k)=\tfrac12 g^{\mu\nu}(x)k_\mu k_\nu$ on phase space
$(x^\mu,k_\mu)$. Hamilton's equations are
$$\frac{\dd x^\mu}{\dd\lambda}=\frac{\partial H}{\partial k_\mu}=g^{\mu\nu}k_\nu,\qquad
\frac{\dd k_\mu}{\dd\lambda}=-\frac{\partial H}{\partial x^\mu}=-\tfrac12\,\partial_\mu g^{\alpha\beta}\,k_\alpha k_\beta .$$
The first uses the symmetry of $g^{\mu\nu}$.

**Equivalence to the geodesic equation.** Differentiating $g^{\alpha\gamma}g_{\gamma\beta}=\delta^\alpha_\beta$ gives
$\partial_\mu g^{\alpha\beta}=-g^{\alpha\gamma}g^{\beta\delta}\partial_\mu g_{\gamma\delta}$, so
$\dot k_\mu=\tfrac12\partial_\mu g_{\gamma\delta}k^\gamma k^\delta$. Then
$$\frac{\dd k^\mu}{\dd\lambda}=\frac{\dd(g^{\mu\nu}k_\nu)}{\dd\lambda}
=g^{\mu\nu}\Big(\tfrac12\partial_\nu g_{\gamma\delta}-\partial_\gamma g_{\nu\delta}\Big)k^\gamma k^\delta
=-\Gamma^\mu_{\gamma\delta}k^\gamma k^\delta .$$
Here $\dd g^{\mu\nu}/\dd\lambda\,k_\nu=-g^{\mu\alpha}(\partial_\gamma g_{\alpha\delta})k^\gamma k^\delta$,
and the symmetrisation $\partial_\gamma g_{\nu\delta}k^\gamma k^\delta=\tfrac12(\partial_\gamma g_{\nu\delta}+\partial_\delta g_{\nu\gamma})k^\gamma k^\delta$
was used. This is MG18 Eqs. (3)–(4), D09 Eqs. (9)–(11), GPM Eq. (10) and KP Eqs. (1)–(2).
The covariant form itself is the one W22 uses (conventions.md: `ref-paper/arxiv-2203.15963`
around l.262–266).

**Null constraint.** $\dd H/\dd\lambda=\{H,H\}=0$, so $H=0$ holds exactly on the flow. The
initial data make $H=0$ exactly because `null_covariant` solves for $k_t$ given $k_i$
(`src/geodesics/integrator.jl:128–145`). RK4 preserves $H$ only to truncation error. The code
reports $g^{\mu\nu}k_\mu k_\nu$ as a diagnostic (`null_error`, `integrator.jl:95–100`).

**No Christoffel array.** The code gets $\partial_i g^{\alpha\beta}k_\alpha k_\beta=\partial_i(2H)$
by evaluating the contraction $2H$ at a position that carries three dual partials
(`geodesic_rhs`, `integrator.jl:157–166`). One pass therefore gives both $g^{\mu\nu}k_\nu$ (the
value) and $\nabla_x(2H)$ (the partials). The dual number is forward-mode AD on the metric
closure, and the result is exact to rounding.
- *Difference from MG18, D09 and KP:* they integrate $k^\mu$ with $\Gamma^\mu_{\alpha\beta}$,
  which are either hand-coded or finite-differenced (D09 uses numerical derivatives of the
  metric).

**Dimension check.** $x$ has units of $\rg$ and $k_\mu$ is dimensionless ($k_t=-1$), so
$\lambda$ has units of $\rg$. $\partial_\mu g^{\alpha\beta}$ has units of $\rg^{-1}$, so
$\dot k_\mu$ has units of $\rg^{-1}$. Consistent.

### 3.2 Exact conservation of $k_t$ (M10)

If $\partial_t g^{\alpha\beta}=0$, then $\dot k_t=-\tfrac12\partial_tg^{\alpha\beta}k_\alpha k_\beta=0$ identically.

The code sets that RHS component to zero rather than computing it
(`integrator.jl:163`, the `zero(T)` in slot 5; `_seed_spatial` seeds only $x,y,z$). The
stored $k_t$ therefore never changes, even under round-off. This is exact conservation of
the energy-at-infinity $E=-k_t$ **in the discrete scheme**, which is what the paper claims.

By contrast, $L_z=xk_y-yk_x$ (the Killing vector $\partial_\phi=x\partial_y-y\partial_x$ in
CKS) is conserved only to truncation error. The documentation gates quote $L_z$ drift
below $10^{-8}$.

### 3.3 Backward orientation (M13)

- Define $p_\mu=-k_\mu$ and $\lambda'=-\lambda$. Then
  $$\frac{\dd x^\mu}{\dd\lambda'}=-g^{\mu\nu}k_\nu=g^{\mu\nu}p_\nu,\qquad
  \frac{\dd p_\mu}{\dd\lambda'}=\frac{\dd k_\mu}{\dd\lambda}=-\tfrac12\partial_\mu g^{\alpha\beta}k_\alpha k_\beta=-\tfrac12\partial_\mu g^{\alpha\beta}p_\alpha p_\beta .$$
  The Hamiltonian is quadratic, so the flow is invariant under $(\lambda,k)\to(-\lambda,-k)$.
  The same RHS function therefore integrates the reversed ray.
- Along the traced ray $\dd x^t/\dd\lambda'=g^{t\nu}p_\nu$. Outside the ergoregion this has
  the sign of $-g^{tt}k_t<0$, so $x^t$ decreases from 0 at the camera.
- Emission time for slow light: $t_{\rm emit}=t_{\rm obs}+x^t$ (conventions.md, Ray time-orientation).
- *Difference from W22, S23, MG18 and KP:* those codes integrate the physical $k$ to
  decreasing $\lambda$. The two formulations are equivalent under this symmetry. The code is
  tested in `test/test_ray_orientation.jl`.

### 3.4 Step rule (M11)

$h=(r-\rhor)/C$ is an affine step (`reverse_march.jl:67`, `KerrMarch`). Why this rule:
- At camera energy $E=1$, $|\dot x^i|\sim1$ far from the hole, so the coordinate step is
  $\approx h\approx r/C$. The relative step $h/r\approx 1/C$ is scale-free, matching S23
  Eq. (11) (step $\propto r/\mathcal C$) and W22 Eq. (7) ($\Delta\lambda=f_{\rm step}(r-r_{\rm h})$).
- Near the horizon the geometric scale is $r-\rhor$, so the step shrinks with it.
- The same step is used as the radiation sample spacing (§7.4). The first-order-in-$1/C$
  quadrature error in the docs (Kerr torus deck, `docs/src/method.md`) comes from that
  sampling, not from RK4.
- *Difference from W22:* W22 also offers an adaptive RK5(4) path (W22 Eq. 8). The production
  path here is fixed-control RK4 so that the CPU and device sample sets are identical.

### 3.5 Analytic equatorial tracer (M16)

This follows GL20a/GL20b and CA23. The code is in `src/geodesics/analytic_kerr.jl`, with
lensing bands in `src/geodesics/lensing_bands.jl`.

1. **Conserved quantities.** $\lambda=L/E=-k_\phi/k_t$ and $\eta=Q/E^2$, with Carter
   constant $Q=k_\theta^2-\cos^2\theta\,(a^2k_t^2-k_\phi^2\csc^2\theta)$ (GL20a Eqs. 3–5,
   `KerrGeo.tex:72–79`). These are Boyer–Lindquist components. The KS-to-BL map changes only
   $t$ and $\phi$ by functions of $r$, so $k_t$, $k_\phi$ and $k_\theta$ are the same in both.
2. **Screen map.** A distant observer at inclination $\theta_o$ sees the ray at
   $\alpha=-\lambda/\sin\theta_o$ and $\beta=\pm_o\sqrt{\Theta(\theta_o)}$, where
   $\Theta(\theta)=\eta+a^2\cos^2\theta-\lambda^2\cot^2\theta$ and $\pm_o={\rm sign}\,\beta$
   (GL20b `eq:ScreenCoordinates`, `KerrLensing.tex:396`; CA23 Eq. 4). Inverting gives
   $$\lambda=-\alpha\sin\theta_o,\qquad \eta=(\alpha^2-a^2)\cos^2\theta_o+\beta^2$$
   (GL20b `eq:lambdaScreen`/`eq:etaScreen`, `KerrLensing.tex:421`). The second follows from
   $\beta^2=\Theta(\theta_o)$ with $\lambda^2\cot^2\theta_o=\alpha^2\cos^2\theta_o$.
3. **Consistency with the numerical camera.** §6.2 shows $L_z=-\alpha\sin i$, which is the
   first relation with $\theta_o=i$. The code's pixel $(\alpha,\beta)$ is therefore Bardeen's
   $(\alpha,\beta)$ with no sign flip (conventions.md, Camera row).
4. **Mino time.** With $\dd x^\mu/\dd\tau=(\Sigma/E)k^\mu$, the radial and polar equations
   separate as $(\dd r/\dd\tau)^2=\mathcal R(r)$ and $(\dd\theta/\dd\tau)^2=\Theta(\theta)$, where
   $\mathcal R(r)=(r^2+a^2-a\lambda)^2-\Delta(r)[\eta+(\lambda-a)^2]$ (GL20a Eqs. 6–7, 24).
   The equality $I_r=G_\theta$ (GL20a Eq. 9) fixes the Mino time at which the ray reaches
   $\theta=\pi/2$ for the $m$-th time:
   $$\tau_m=G_\theta^{(m)}=m\hat G_\theta+\nu_\theta\big[(-1)^m\mathcal G_\theta^o-\mathcal G_\theta^s\big]$$
   (GL20a Eq. 51, `eq:UnpackedTau`).
5. **Source radius.** Inverting $I_r(r_s)=\tau_m$ with the Jacobi ${\rm sn}$ function gives
   $r_s$ in closed form for each root type (GL20a appendix B, `KerrGeo.tex:848+`). $\phi_s$
   and $t_s$ follow from GL20a Eqs. (10)–(11), with the $G_\phi$ and $G_t$ integrals in
   Legendre form (GL20a Eqs. 28–30, 46–47).
6. **Image order.** $n=m-1$ for the $m$-th equatorial crossing. $n=0,1,2$ are the direct
   image and the first two photon-ring images.
7. **Derivatives.** ${\rm sn}$, $F$, $\Pi$ and $E'$ are evaluated in closed form in generic
   Julia. ForwardDiff therefore propagates $\partial r_s/\partial(a,\theta_o)$ by the chain
   rule through each special-function evaluation. The documentation checks this against
   finite differences.

- *Symbol clash:* the conserved $(\lambda,\eta)$ collide with the affine $\lambda$ and the
  Minkowski $\eta$. symbols.md §7 proposes $\hat\lambda,\hat\eta$.

---

## 4. Frequency and redshift factor (M3)

1. **Physical wave vector.** Let $K^\mu_{\rm phys}$ be the physical wave four-vector in units
   where $-K_\mu u^\mu$ equals the frequency (Hz) measured by an observer with four-velocity
   $u$. The frequency measured by a static observer at infinity is $\nu_\infty=-K_t$, which
   is conserved (§3.2).
2. **Code normalisation.** The code stores $k_\mu=K_\mu/\nu_\infty$, so $k_t=-1$.
   Contracting with $u$ gives $\nu=-K_\mu u^\mu=\nu_\infty(-k_\mu u^\mu)$.
   Identifying $\nu_\infty\equiv\nuobs$ gives the paper's relation
   $$\nu=\nuobs(-k_\mu u^\mu),\qquad g\equiv\frac{\nuobs}{\nu}=\frac{1}{-k_\mu u^\mu}.$$
   Code: `src/radiation/integrator.jl:56–60`, `nu_cgs = minus_ku * ν_obs`.
3. **Which observer defines $\nuobs$.** The camera's normal observer $n^\mu$ at distance $d$
   measures $-k\cdot n=1/\alpha_{\rm cam}$, not 1. In CKS, $\alpha_{\rm cam}=(1+f)^{-1/2}\approx1-M/d$
   (§2.1). $\nuobs$ is therefore the frequency at infinity, and the camera observer sees
   $\nuobs(1+M/d)$: 0.1% more at $d=10^3$, which is 0.3% in $\nu^3$.
   - **[paper]** method.tex:14 says that "an observer with four-velocity $u^\mu$ measures
     $\nu=\nuobs(-k_\mu u^\mu)$", which leaves the reference observer implicit. One clause
     fixes this: "$\nuobs$ is the frequency at infinity, $E=-k_t=1$". KP Eq. (5) instead
     normalises to the finite observer, with a ratio
     $(-u_\mu k^\mu)/(-u_{{\rm obs}\,\mu}k^\mu_{\rm obs})$. The two agree to $O(M/d)$.
4. **Signs.** For gas moving toward the camera with speed $v$ relative to the normal
   observer, $-k\cdot u=\gamma E_n(1-v\cos\psi)$, which is less than $E_n$ when $\cos\psi>0$
   (blueshift, $\nu<\nuobs$ at the source). The code checks
   $\nu/\nuobs=\gamma(1-v)$ for head-on approach in flat space in
   `test/test_ray_orientation.jl`. This fixes the sign of $k$: $k$ is the received
   (future-directed) covector, not the traced $p$.
5. **Invariant pitch angle** (used in `unpolarized_coefficients`, `integrator.jl:102–105`).
   In the fluid frame, $k^{(i)}$ has norm $\omega=-k\cdot u$ because $k$ is null. With
   $b\cdot u=0$, the component of $b$ along $\hat k$ is $k\cdot b/\omega$. Therefore
   $$\cos^2\theta_B=\frac{(k\cdot b)^2}{(k\cdot u)^2\,b^2}.$$
   This is frame-free, so the Stokes-$I$ path builds no tetrad.

---

## 5. Invariant polarized transfer

### 5.1 Invariants, affine normalisation, units (M4, M17, M19)

- **Flat-space equation.** In a local inertial frame comoving with the fluid,
  $\dd I_\nu/\dd\ell=j_\nu-\alpha_\nu I_\nu$, where $\ell$ is proper path length (MG18 Eq. 1).
- **Invariants.** $I_\nu/\nu^3$ is Lorentz invariant (photon occupation number), and so are
  $j_\nu/\nu^2$ and $\nu\alpha_\nu$ (MG18 Eq. 2; Mihalas & Mihalas 1984). The polarized
  generalisation, with the invariants
  $\boldsymbol{\mathcal S}$, $j_A$, $\alpha_A$ and $\rho_A$ for $A\in\{I,Q,U,V\}$, is MG18
  Eqs. (19)–(20). The code stores these invariants (`src/radiation/coefficients.jl:5–7`).
- **Affine parameter in physical units.** Let the physical wave vector be $K^\mu=\nuobs k^\mu$
  (§4) and let positions be in cm, $X^\mu=\rg x^\mu$. A physical affine parameter
  $\lambda_{\rm c}$ with $\dd X^\mu/\dd\lambda_{\rm c}=K^\mu$ gives
  $\rg\,\dd x^\mu=\nuobs k^\mu\dd\lambda_{\rm c}=\nuobs\,\dd x^\mu\,\dd\lambda_{\rm c}/\dd\lambda$, so
  $$\lambda_{\rm c}=\lambda\,\rg/\nuobs\qquad[\text{cm Hz}^{-1}].$$
  Code: `ds = h * L_unit / nu_obs` (`reverse_march.jl:160, 270`).
- **Fluid-frame path length.** The fluid-frame time elapsed along the ray is
  $-u_\mu\dd X^\mu=-u_\mu K^\mu\dd\lambda_{\rm c}=\nu\,\dd\lambda_{\rm c}$ (cm, since $c=1$ in
  $X^0=ct$). For a null ray, proper spatial distance equals elapsed light time, so
  $$\dd\ell=\nu\,\dd\lambda_{\rm c}.$$
- **Invariant equation.** Over an infinitesimal step, $\nu$ in the local frame changes only
  at second order, so
  $$\frac{\dd(I_\nu/\nu^3)}{\dd\lambda_{\rm c}}=\frac{\nu}{\nu^3}(j_\nu-\alpha_\nu I_\nu)=\frac{j_\nu}{\nu^2}-(\nu\alpha_\nu)\frac{I_\nu}{\nu^3}.$$
  This is MG18 Eq. (2) and D09 Eq. (12) with $\mathcal C=1$ in these units.
- **Dimension check.**
  - $I_\nu/\nu^3$ has units erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Hz$^{-4}$.
  - $j_\nu/\nu^2\times\dd\lambda_{\rm c}$ has units erg s$^{-1}$ cm$^{-3}$ sr$^{-1}$ Hz$^{-3}\times$ cm Hz$^{-1}$, which match.
  - $\nu\alpha_\nu\,\dd\lambda_{\rm c}=\alpha_\nu\,\dd\ell$ is dimensionless (the optical depth). Consistent.
- **Relation to KP.** KP Eq. (8) writes $\mathcal J=\ell_\lambda j_\nu/\nu^3$ per unit
  dimensionless affine parameter, with $\ell_\lambda=-u_\mu k^\mu$ times a length unit. The
  map is $\mathcal J\,\dd\lambda_{\rm KP}=j_A\,\dd\lambda_{\rm c}$ (conventions.md, Invariant row).
- **Camera output (M19).** At the camera, $\nu=\nuobs$ by the normalisation of §4. Hence
  $I_\nu=\nuobs^3\mathcal I$ and $S_\nu=\nuobs^3\boldsymbol{\mathcal S}$ (`reverse_march.jl:167, 293`).
- **[paper]** method.tex:56 defines $\lambda_{\rm c}$ but should state the units of
  $\boldsymbol{\mathcal S}$ and $\lambda_{\rm c}$ once. The name "affine parameter of the momentum in frequency
  units" is hard to parse. Suggested wording: "the affine parameter of the wave vector
  $\nuobs k^\mu$ (cm Hz$^{-1}$)".

### 5.2 Transfer matrix form (eq:transfer)

- **The equation.** $\dd\boldsymbol{\mathcal S}/\dd\lambda_{\rm c}=\boldsymbol j-\mathbf K\boldsymbol{\mathcal S}$, with $\mathbf K$ as displayed. This
  is the form of LD85, reproduced in MG18 Eqs. (6) and (19)–(20), KP Eqs. (6)–(7) and M21
  Eqs. (1)–(2). It holds in an orthonormal frame comoving with the fluid, with $e_{(3)}$
  along the photon. It covers emission, dichroism, and birefringence.
- **Structure.** Split $\mathbf K=\alpha_I\mathbf 1+\mathbf K'$, with
  $$\mathbf K'=\begin{pmatrix}0&\boldsymbol\alpha^{\sf T}\\ \boldsymbol\alpha&[\boldsymbol\rho]_\times^{\sf T}\end{pmatrix},\qquad
  \boldsymbol\alpha=(\alpha_Q,\alpha_U,\alpha_V),\ \boldsymbol\rho=(\rho_Q,\rho_U,\rho_V),$$
  where the lower-right block satisfies $K'_{ij}=\epsilon_{ijk}\rho_k$. Check against the display:
  $K'_{QU}=\epsilon_{QUV}\rho_V=\rho_V$ ✓, $K'_{QV}=\epsilon_{QVU}\rho_U=-\rho_U$ ✓,
  $K'_{UV}=\epsilon_{UVQ}\rho_Q=\rho_Q$ ✓.
- **Per-component form.** With $\boldsymbol p=(Q,U,V)$,
  $$\frac{\dd\mathcal I}{\dd\lambda_{\rm c}}=j_I-\alpha_I\mathcal I-\boldsymbol\alpha\cdot\boldsymbol p,\qquad
  \frac{\dd\boldsymbol p}{\dd\lambda_{\rm c}}=\boldsymbol j_p-\alpha_I\boldsymbol p-\boldsymbol\alpha\,\mathcal I+\boldsymbol\rho\times\boldsymbol p .$$
- **Geometric meaning.** The $\boldsymbol\rho$ term is a rotation of the Poincaré vector
  $\boldsymbol p$ about $\boldsymbol\rho$ at angular rate $|\boldsymbol\rho|$ per unit $\lambda_{\rm c}$.
- **Lorentz-generator property (used in §5.3 and §5.4).** For $\boldsymbol j=0$ and $\alpha_I=0$,
  $$\frac{\dd}{\dd\lambda_{\rm c}}(\mathcal I^2-|\boldsymbol p|^2)=2\mathcal I(-\boldsymbol\alpha\cdot\boldsymbol p)-2\boldsymbol p\cdot(-\boldsymbol\alpha\mathcal I+\boldsymbol\rho\times\boldsymbol p)=0.$$
  So $\mathbf K'$ generates a proper Lorentz transformation of the Stokes 4-vector: a boost
  of rapidity rate $\boldsymbol\alpha$ and a rotation of rate $\boldsymbol\rho$.
- **Signs versus D16.** D16's matrix (`ref-paper/arxiv-1602.03184`, transfer matrix near
  tex l.396–408) has the opposite sign on the $\rho_U$ entries (MG18 footnote). This has no
  effect here, since $\rho_U=0$ in the fluid tetrad (§6.4).
- *Symbols to define in the paper:* the Stokes index $A$ and its range; the fluid-frame
  $j_{\nu,A},\alpha_{\nu,A},\rho_{\nu,A}$; and bold $\boldsymbol j=(j_I,j_Q,j_U,j_V)$ (symbols.md §8).

### 5.3 Exact constant-coefficient segment solution (M20, M24)

For constant $\boldsymbol j$ and $\mathbf K$ over a segment of length $s=\Delta\lambda_{\rm c}$:
$$\boldsymbol{\mathcal S}(s)=O(s)\,\boldsymbol{\mathcal S}_0+P(s)\,\boldsymbol j,\qquad O(s)=e^{-\mathbf Ks},\qquad P(s)=\int_0^s O(s')\,\dd s' .$$
This is the variation-of-constants formula: $\frac{\dd}{\dd s}(e^{\mathbf Ks}\boldsymbol{\mathcal S})=e^{\mathbf Ks}\boldsymbol j$,
integrated from 0 to $s$. It is MG18 Eqs. (21)–(23). When $\mathbf K$ is invertible,
$P=\mathbf K^{-1}(\mathbf 1-O)$. The segment operator is affine,
$\boldsymbol{\mathcal S}_{\rm out}=M\boldsymbol{\mathcal S}_{\rm in}+b$ with $M=O(s)$ and $b=P(s)\boldsymbol j$. This is M24.

**Eigenstructure of $\mathbf K'$.**
- $\mathbf K'$ has the algebra of an electromagnetic field tensor $F^\mu{}_\nu$, with
  "electric" part $\boldsymbol\alpha$ and "magnetic" part $\boldsymbol\rho$. Its characteristic polynomial is
  $$\det(\mathbf K'-\Lambda\mathbf 1)=\Lambda^4-(\alpha^2-\rho^2)\Lambda^2-(\boldsymbol\alpha\cdot\boldsymbol\rho)^2 ,$$
  with $\alpha\equiv|\boldsymbol\alpha|$ and $\rho\equiv|\boldsymbol\rho|$ (the invariants $E^2-B^2$ and $(E\cdot B)^2$).
- The roots are $\Lambda=\pm\Lambda_1$ and $\pm i\Lambda_2$, with
  $$\Lambda_{1,2}=\Big[\sqrt{\tfrac14(\alpha^2-\rho^2)^2+(\boldsymbol\alpha\cdot\boldsymbol\rho)^2}\ \pm\ \tfrac12(\alpha^2-\rho^2)\Big]^{1/2}.$$
  Code: `la`, `lb`, `l1`, `l2` (`src/radiation/transfer.jl:74–79`).
- Two identities follow: $\Theta\equiv\Lambda_1^2+\Lambda_2^2=2\sqrt{\cdot}$ and $\Lambda_1\Lambda_2=|\boldsymbol\alpha\cdot\boldsymbol\rho|$.
- Define the dual $\tilde{\mathbf K}'$ by $(\boldsymbol\alpha,\boldsymbol\rho)\to(\boldsymbol\rho,-\boldsymbol\alpha)$ and
  $\sigma={\rm sgn}(\boldsymbol\alpha\cdot\boldsymbol\rho)$. Then the four matrices
  $$M_2=\frac{\Lambda_2\mathbf K'-\sigma\Lambda_1\tilde{\mathbf K}'}{\Theta},\quad
  M_3=\frac{\Lambda_1\mathbf K'+\sigma\Lambda_2\tilde{\mathbf K}'}{\Theta},\quad
  M_4=\frac{2}{\Theta}\Big[\mathbf K'^2-\tfrac12(\alpha^2-\rho^2)\mathbf 1\Big]$$
  satisfy $\mathbf K'=\Lambda_1M_3+\Lambda_2M_2$. Proof: $\Lambda_1^2\mathbf K'+\sigma\Lambda_1\Lambda_2\tilde{\mathbf K}'+\Lambda_2^2\mathbf K'-\sigma\Lambda_1\Lambda_2\tilde{\mathbf K}'=\Theta\mathbf K'$.
- $\tfrac12(\mathbf 1\pm M_4)$ project onto the real ($\pm\Lambda_1$) and imaginary
  ($\pm i\Lambda_2$) eigenspace pairs.
- Entry-by-entry check against the code (`transfer.jl:87–99`):
  - $(M_2)_{IQ}=(\Lambda_2\alpha_Q-\sigma\Lambda_1\rho_Q)/\Theta$ matches `m2[1,2]`.
  - $(M_2)_{QU}=(\Lambda_2\rho_V+\sigma\Lambda_1\alpha_V)/\Theta$ matches `m2[2,3]`.
  - $(M_4)_{II}=(\alpha^2+\rho^2)/\Theta$ matches `2*it*d0`.
  - $(M_4)_{IU}=2(\alpha_Q\rho_V-\alpha_V\rho_Q)/\Theta$ matches `m4[1,3]`.

**Propagator** (LD85; MG18 Eq. 24 and its $O_{AB}$):
$$O(s)=e^{-\alpha_Is}\Big[\tfrac12(\mathbf 1+M_4)\cosh\Lambda_1s-M_3\sinh\Lambda_1s+\tfrac12(\mathbf 1-M_4)\cos\Lambda_2s-M_2\sin\Lambda_2s\Big].$$
- Check at $s=0$: $O(0)=\mathbf 1$.
- Check the derivative: $O'(0)=-\alpha_I\mathbf 1-\Lambda_1M_3-\Lambda_2M_2=-\alpha_I\mathbf 1-\mathbf K'=-\mathbf K$ ✓.
- Code: `oo` (`transfer.jl:113`).

**Integrated operator.** $P(s)=\int_0^sO$ follows from four elementary integrals, with $a\equiv\alpha_I$:
$$\int_0^s e^{-as'}\cosh\Lambda s'\,\dd s'=\frac{a-e^{-as}(a\cosh\Lambda s+\Lambda\sinh\Lambda s)}{a^2-\Lambda^2},\qquad
\int_0^s e^{-as'}\sinh\Lambda s'\,\dd s'=\frac{\Lambda-e^{-as}(a\sinh\Lambda s+\Lambda\cosh\Lambda s)}{a^2-\Lambda^2},$$
$$\int_0^s e^{-as'}\cos\Lambda s'\,\dd s'=\frac{a-e^{-as}(a\cos\Lambda s-\Lambda\sin\Lambda s)}{a^2+\Lambda^2},\qquad
\int_0^s e^{-as'}\sin\Lambda s'\,\dd s'=\frac{\Lambda-e^{-as}(a\sin\Lambda s+\Lambda\cos\Lambda s)}{a^2+\Lambda^2}.$$
Write $f_1=1/(a^2-\Lambda_1^2)$ and $f_2=1/(a^2+\Lambda_2^2)$, and collect the constant,
$\cosh$, $\sinh$, $\cos$ and $\sin$ terms:
- constant: $C_h=\tfrac12af_1(\mathbf 1+M_4)-\Lambda_1f_1M_3$ and $C_c=\tfrac12af_2(\mathbf 1-M_4)-\Lambda_2f_2M_2$ (code `cosh_t`, `cos_t`);
- $\sinh$: $S_h=\tfrac12\Lambda_1f_1(\mathbf 1+M_4)-af_1M_3$ (code `sinh_t`);
- $\sin$: $S_c=-\tfrac12\Lambda_2f_2(\mathbf 1-M_4)-af_2M_2$ (code `sin_t`);

so that
$$P(s)=C_h+C_c-e^{-as}\big[\cosh(\Lambda_1s)C_h+\cos(\Lambda_2s)C_c+\sin(\Lambda_2s)S_c+\sinh(\Lambda_1s)S_h\big],$$
which is `pp` (`transfer.jl:104–112`) and MG18 Eq. (24). ✓

**Special branches** (`transfer.jl:59–71, 117–125`). Each follows from the general formula
in the limits below.
1. **Emission only** ($\mathbf K=0$): $O=\mathbf 1$ and $P=s\mathbf 1$.
2. **Total absorption only** ($\boldsymbol\alpha=\boldsymbol\rho=0$): $O=e^{-\alpha_Is}\mathbf 1$ and
   $P=(1-e^{-\alpha_Is})/\alpha_I$. The code uses $-{\rm expm1}(-\Delta\tau)$ for small $\Delta\tau$.
3. **Pure rotation** ($\alpha_I=0$, which forces $\boldsymbol\alpha=0$ by admissibility): $O$ is the rotation by
   angle $\rho s$ about $\hat{\boldsymbol\rho}$,
   $$O_{pp}=\cos\rho s\,\mathbf 1+(1-\cos\rho s)\hat{\boldsymbol\rho}\hat{\boldsymbol\rho}^{\sf T}+\sin\rho s\,[\hat{\boldsymbol\rho}]_\times ,$$
   (Rodrigues), so that $\dot{\boldsymbol p}=\boldsymbol\rho\times\boldsymbol p$. This is MG18 Eqs. (A2)–(A5). The
   identity $1-\cos\rho s=2\sin^2(\rho s/2)$ gives the code's `ss` terms.
4. **Saturation** ($\Delta\tau=\alpha_Is>100$): $O\to0$ and $P\to\mathbf K^{-1}$ (the source
   function). The code returns $P$ evaluated at $s\to\infty$, which is `pp` before the
   exponential terms.

**Deviations from W22 and MG18** (`transfer.jl:26–34`):
1. The case $\rho=0$ with $\alpha\neq0$ uses the general formula. There $\Lambda_1=\alpha$,
   $\Lambda_2=0$, $M_2$ drops out and the formula stays regular. The references' dedicated
   "absorption without rotation" branch is dropped because it disagrees with a
   matrix-exponential oracle (docs, Deviations 1).
2. **Degenerate guard.** $f_1$ diverges when $\alpha_I^2=\Lambda_1^2$. Since
   $\Lambda_1^2\le\alpha^2$ (with equality iff $\rho\parallel\alpha$ or $\rho=0$) and admissibility
   enforces $\alpha\le\alpha_I$ (§5.4), this happens only on the cone boundary
   $\alpha=\alpha_I$ with $\boldsymbol\rho\parallel\boldsymbol\alpha$. The code then falls back to branch 2.
   - That fallback is an approximation: it drops the polarized part of the exact operator on
     a set of measure zero.
   - **[paper]** The paper should call it a guard for a degenerate case, not a solution.

**Limit and sign checks.**
- As $s\to0$: $b=P\boldsymbol j=s\boldsymbol j+O(s^2)$ and $M=\mathbf 1-s\mathbf K+O(s^2)$, which is the forward-Euler step of eq:transfer.
- Thick limit: $b\to\mathbf K^{-1}\boldsymbol j$. For an unpolarized source this is $j_I/\alpha_I$, and with Kirchhoff's law it gives $B_\nu/\nu^3$.
- Unpolarized reduction: $A=e^{-\Delta\tau}$ and $B=(j/\alpha)(1-e^{-\Delta\tau})$. This is W22 Eq. (10), with code `unpolarized_segment` (`transfer.jl:165–177`).

### 5.4 Admissibility (cone preservation)

- **Claim.** If $j_I\ge|\boldsymbol j_p|$, $\alpha_I\ge|\boldsymbol\alpha|$ and $\boldsymbol{\mathcal S}_0$ lies in the forward cone
  $\mathcal I\ge|\boldsymbol p|$, then $\boldsymbol{\mathcal S}(s)$ stays in the cone.
- **Proof.**
  1. $e^{-\mathbf K's}$ is a proper orthochronous Lorentz transformation (§5.2), so it maps
     the forward cone onto itself.
  2. The prefactor $e^{-\alpha_Is}>0$ preserves the cone.
  3. $P=\int_0^sO$ is an integral of cone-preserving maps, so $P\boldsymbol j$ lies in the cone when $\boldsymbol j$ does.
  4. The sum of two cone vectors lies in the cone.
- **Why $\alpha_I\ge|\boldsymbol\alpha|$ matters.** The boost can raise $\mathcal I$ by up to $e^{\alpha s}$. With
  $\alpha>\alpha_I$ some polarization state would grow as $e^{(\alpha-\alpha_I)s}$, which is
  unphysical gain.
- **Code.** It rescales the fits onto $|\boldsymbol\alpha|\le\alpha_I$ and $|\boldsymbol j_p|\le j_I$ before forming
  the operator (`transfer.jl:41–53`). The M21 thermal fits violate $|\boldsymbol j_p|\le j_I$ at small
  pitch angle (docs, Admissibility). MG18 Eq. (7) states the emissivity condition.
- **The partial-sum clamp** `stokes_admissible` (`transfer.jl:133–141`) is applied after
  every accumulation (`reverse_march.jl:273`). For admissible coefficients it never fires
  except through round-off.

### 5.5 Faraday rotation and conversion (M18)

**Faraday rotation.** Set $\alpha=0$ and $j=0$. Then $\dd Q/\dd\lambda_{\rm c}=-\rho_VU$ and
$\dd U/\dd\lambda_{\rm c}=\rho_VQ$ (from rows 2–3 of $\mathbf K$), so
$$\frac{\dd(Q+iU)}{\dd\lambda_{\rm c}}=i\rho_V(Q+iU)\ \Rightarrow\ \frac{\dd\chi}{\dd\lambda_{\rm c}}=\frac{\rho_V}{2},\qquad \chi=\tfrac12\arg(Q+iU).$$
The linear polarization rotates at half the rotativity. With $\rho_V>0$, $\chi$ increases,
i.e. it rotates from north toward east in the $(e_{(2)},-e_{(1)})$ basis.
- **Cold-plasma limit of the M21 fit** (M21 `Main.tex:358`): $\rho_V=\frac{2n_ee^2\nu_c}{m_ec\nu^2}\frac{K_0-\Delta J_5}{K_2}\cos\theta_B$, with Bessel argument $1/\Theta_e$.
  - As $\Theta_e\to0$, $K_0/K_2\to1$ and $\Delta J_5\to0$ for $X\gg1$.
  - With $\nu_c=eB/(2\pi m_ec)$ this gives $\rho_{\nu,V}=\frac{e^3n_eB\cos\theta_B}{\pi m_e^2c^2\nu^2}$.
  - Standard Faraday rotation is $\dd\chi/\dd\ell={\rm RM}$-rate $=\frac{e^3n_eB_\parallel}{2\pi m_e^2c^2\nu^2}$ (from ${\rm RM}=\frac{e^3}{2\pi m_e^2c^4}\int n_eB_\parallel\dd\ell$ and $\lambda^2=c^2/\nu^2$), and $\dd\chi/\dd\ell=\rho_{\nu,V}/2$ ✓.
- **Sign.** $\cos\theta_B>0$ for $\boldsymbol B$ along $\hat{\boldsymbol k}$, i.e. toward the observer. Rotation is then
  toward east, which matches the thin-slab Faraday test in `test/test_ray_orientation.jl`
  (conventions.md, Stokes row).
- **Parity.** $\rho_V$ is odd in $B$ ($\cos\theta_B$).

**Faraday conversion.** Rows 3–4 give $\dd U/\dd\lambda_{\rm c}=-\rho_QV$ and
$\dd V/\dd\lambda_{\rm c}=\rho_QU$. This is a rotation about the $Q$ axis of the Poincaré
sphere, exchanging $U$ and $V$.
- **Parity.** $\rho_Q$ is even in $B$ ($\propto\sin^2\theta_B$).
- **Cold-plasma limit of M21** (`Main.tex:345`): $\rho_Q\to-\frac{n_ee^2\nu_c^2\sin^2\theta_B}{m_ec\nu^3}$ as $K_1/K_2\to1$, $6\Theta_e\to0$ and $f_m\to1$.
  - The magnitude equals $\frac{\omega_p^2\omega_B^2\sin^2\theta_B}{2c\,\omega^3}$ with
    $\omega_p^2=4\pi n_ee^2/m_e$, $\omega_B=2\pi\nu_c$ and $\omega=2\pi\nu$. This is the
    cold-plasma birefringence $\frac{\omega}{c}|n_\perp-n_\parallel|$ ✓.
  - The negative sign belongs to the M21 basis ($Q>0$ in the $k$–$B$ plane).
- **Code** (thermal): `src/radiation/coefficients.jl:356–388`. The power-law and kappa forms
  are at `:469` and `:547–557` (conventions.md).

### 5.6 Coefficient basis and the grtrans sign (M23)

Rotate the linear-polarization basis by $90^\circ$ about $\hat k$. Then $2\chi\to2\chi+\pi$,
and the Stokes rotation is $R={\rm diag}(1,-1,-1,1)$.
- Coefficient vectors transform as $\boldsymbol j\to R\boldsymbol j$, and the matrix as $\mathbf K\to R\mathbf KR^{\sf T}$.
- Entry by entry: $(\mathbf K)_{IQ}=\alpha_Q\to-\alpha_Q$, $(\mathbf K)_{IU}=\alpha_U\to-\alpha_U$ and $(\mathbf K)_{UV}=\rho_Q\to-\rho_Q$.
  $(\mathbf K)_{QV}=-\rho_U$ also changes sign, so $\rho_U\to-\rho_U$. Meanwhile $(\mathbf K)_{QU}=\rho_V$ and
  $(\mathbf K)_{IV}=\alpha_V$ are unchanged.
- So every $Q$ and $U$ coefficient changes sign, and $V$ coefficients and $\mathbf K$'s form are
  unchanged. This is the relation between the P16/M21 basis ($Q>0$ in the $k$–$B$ plane,
  $j_Q<0$) and the D16 basis ($Q>0$ perpendicular, $j_Q>0$). The claim at method.tex:60 is verified.
- Synchrotron emission is polarized perpendicular to the projected field, so $j_Q<0$ in a
  basis whose $Q>0$ axis is the projected field ✓.

---

## 6. Camera and tetrads

### 6.1 Pixel map and screen triad (M12, M13)

The screen centre is at $d\hat n$, with $\hat n=(\sin i,0,\cos i)$ the unit vector toward the
camera in the $x$–$z$ plane (observer azimuth 0). The screen axes are $\hat\alpha=\hat y$ and
$\hat\beta=(-\cos i,0,\sin i)$. $\hat\beta$ is the projection of $\hat z$ onto the plane
perpendicular to $\hat n$: $\hat z-(\hat z\cdot\hat n)\hat n=(-\sin i\cos i,0,\sin^2 i)$, which
normalised is $\hat\beta$.

The pixel at $(\alpha,\beta)$ sits at
$$\boldsymbol x=d\hat n+\alpha\hat\alpha+\beta\hat\beta=(d\sin i-\beta\cos i,\ \alpha,\ d\cos i+\beta\sin i).$$
This is S23 Eqs. (5)–(7) (`main.tex:129`) with $x'=\alpha$ and $y'=\beta$, and code
`camera_ray` (`src/geodesics/camera.jl:45–51`).

**Handedness.** $\hat y\times(v_x,0,v_z)=(v_z,0,-v_x)$, so $\hat\alpha\times\hat\beta=(\sin i,0,\cos i)=\hat n$.
The triad $(\hat\alpha,\hat\beta,\hat n)$ is therefore right-handed ✓.

**Initial covector.**
1. Take the provisional received direction $k^\mu\propto(1,\hat n)$ (outgoing, toward the camera).
2. Lower the index with $g_{\mu\nu}(x)$ to get the spatial $k_i$.
3. Solve $g^{\mu\nu}k_\mu k_\nu=0$ for $k_t$ with $k_t<0$ (positive energy). Outside the
   ergoregion the two roots of the quadratic have opposite signs, so the root is unique
   (`integrator.jl:103–145`).
4. Store $p=k/k_t$, so that $p_t=+1$ and $p=-k/E$ ✓.
- *Difference from S23:* S23 launches the contravariant tangent directly. W22 uses a
  pinhole-free parallel-ray camera with a roll angle, and the code corresponds to roll $\psi=0$.

### 6.2 Bardeen coordinates (M14)

$L_z=k_\phi$ is the $\partial_\phi$ component. In CKS, $\partial_\phi=x\partial_y-y\partial_x$,
so $L_z=xk_y-yk_x$.

As $d\to\infty$ the metric becomes flat at the camera, so $k_i\to E\hat n_i$ with $E=1$,
giving $k_x=\sin i$ and $k_y=0$. Then
$$L_z=(d\sin i-\beta\cos i)\cdot0-\alpha\sin i=-\alpha\sin i .$$
With $\lambda=L_z/E$ this is the first Bardeen relation of §3.5 ✓.
- At finite $d$ the lowered $k_y$ picks up $f\,l_y(l\cdot k)=O(M/d)$, so the identity holds to $O(M/d)$.
- The test `test/test_ray_orientation.jl` checks $L_z/E=-\alpha\sin i$.

### 6.3 Camera tetrad and the Stokes basis (M5, M15)

**The tetrad.**
- $e_{(0)}=n^\mu$ is the normal observer, with $n^\mu=(1/\alpha,-\beta^i/\alpha)$ from
  $n_\mu=(-\alpha,0,0,0)$ (`src/radiation/integrator.jl:122–126`). Then $n\cdot n=-1$.
- $e_{(3)}=k/\omega-n$, with $\omega=-k\cdot n$. This is a unit vector along the received
  photon (derivation in §6.4).
- $e_{(2)}$ is the up vector $\hat\beta$ lifted to four dimensions and projected (§6.4).
- $e_{(1)}=e_{(2)}\times e_{(3)}$.

At large $d$ the spatial parts are $e_{(3)}=\hat n$ and $e_{(2)}=\hat\beta$. Since
$\hat\beta\times\hat n=\hat\alpha$ (cyclic permutation of $\hat\alpha\times\hat\beta=\hat n$),
$e_{(1)}=\hat\alpha$.

**IAU orientation.** The observer faces the source along $-\hat n$, with up $=\hat\beta$. Their
right-hand direction is $(-\hat n)\times\hat\beta=\hat\beta\times\hat n=\hat\alpha$. On the sky
seen from Earth, east is to the left when north is up, so east $=-\hat\alpha=-e_{(1)}$ and
north $=\hat\beta=e_{(2)}$. The pair (north, east) $=(e_{(2)},-e_{(1)})$.

**Right-handedness about propagation.** $e_{(2)}\times(-e_{(1)})=\hat\alpha\times\hat\beta=\hat n$,
the propagation direction. This is the IEEE requirement for the sign of $V$.

**Stokes definitions.** With the transverse field
$\boldsymbol E=E_N\hat e_N+E_E\hat e_E$, the Stokes parameters are
- $Q=\langle|E_N|^2-|E_E|^2\rangle$, positive north–south;
- $U=2{\rm Re}\langle E_NE_E^*\rangle$, positive NE–SW;
- $V=2{\rm Im}\langle E_N^*E_E\rangle$ for a field $\boldsymbol E={\rm Re}[(E_N\hat e_N+E_E\hat e_E)e^{-i\omega t}]$.
  For $E_E=iE_N$ the field is $\hat e_N\cos\omega t+\hat e_E\sin\omega t$. It rotates from N
  toward E, which is counter-clockwise for a receiver looking at the source with north up
  and east left. It is right-handed about $\hat e_N\times\hat e_E=\hat n$, and $V=2|E_N|^2>0$ ✓.

$\chi=\tfrac12\arg(Q+iU)$ is then measured from N toward E. These are the IAU (1974) and
IEEE conventions. The tests fix the absolute EVPA on thin slabs: EVPA is perpendicular to
the projected $B$, $B$ north gives $Q<0$, and $B$ toward the observer gives $V>0$
(conventions.md).
- *Difference from MG18:* MG18 refers Stokes to $(e_{(1)},e_{(2)})$ with the coherency map
  of MG18 Eqs. (13)–(14). The OPEN-2 note in conventions.md proves the two readouts equivalent.
- *Symbols to define in the paper:* $n^\mu$, which appears first at method.tex:74 (symbols.md §8).

### 6.4 Fluid tetrad (M22)

Given $u$, the received $k$ and the field $b$ (with $u\cdot b=0$), the code builds the tetrad
as follows (`src/radiation/tetrads.jl:34–86`, after W22 `radiation_geometry.hpp`).

1. $e_{(0)}=u$.
2. $e_{(3)}=k/\omega-u$ with $\omega=-k\cdot u$.
   - Norm: $e_{(3)}\cdot e_{(3)}=0-2(k\cdot u)/\omega+u\cdot u=2-1=1$.
   - Orthogonality: $e_{(3)}\cdot u=-1+1=0$.
   - In the fluid frame, $k=\omega(e_{(0)}+e_{(3)})$, so $e_{(3)}$ is the unit propagation direction.
3. $\tilde e_{(2)}=b-\frac{k\cdot b}{\omega}e_{(3)}+\frac{u\cdot b}{\omega}k$, then normalised.
   - Orthogonal to $u$: $b\cdot u-0-\frac{u\cdot b}{\omega}\omega=0$.
   - Orthogonal to $e_{(3)}$: using $b\cdot e_{(3)}=k\cdot b/\omega-u\cdot b$ and $k\cdot e_{(3)}=\omega$,
     $$\big(\tfrac{k\cdot b}{\omega}-u\cdot b\big)-\tfrac{k\cdot b}{\omega}+\tfrac{u\cdot b}{\omega}\omega=0.$$
   - When $u\cdot b=0$, $\tilde e_{(2)}$ is the part of $b$ perpendicular to $\hat k$ in the fluid frame.
4. $e_{(1)\mu}=\epsilon_{\mu\nu\rho\sigma}e_{(0)}^\nu e_{(2)}^\rho e_{(3)}^\sigma$, raised with $g^{\mu\nu}$.
   - In the fluid rest frame this is $e_{(1)}=e_{(2)}\times e_{(3)}$.
   - It is unit when $\sqrt{-g}=1$ (single CKS). Otherwise the code renormalises it.

**Zero $U$ coefficients.** The plasma is symmetric under rotations about $\hat k$ except
through the transverse field direction $\hat b_\perp$, which is a line: synchrotron
coefficients are even in $\hat b_\perp\to-\hat b_\perp$ for the linear part. Any
linear-polarization coefficient $c\in\{j,\alpha,\rho\}$ therefore has the spin-2 form
$(c_Q,c_U)=c_{\rm lin}(\theta_B)(\cos2\psi,\sin2\psi)$, where $\psi$ is the angle of
$\hat b_\perp$ from the Stokes reference axis. The circular parts $c_V$ are spin-0 and odd in
$\cos\theta_B$. The tetrad puts $e_{(2)}\parallel\hat b_\perp$, so $\psi=0$ and
$c_U=0$. The same statement appears in M21 (susceptibility form, Eqs. 5–6, with the
field along $\hat z$ and $k$ in the $x$–$z$ plane).
- A mirror argument alone does not suffice: reflecting through the $k$–$b$ plane flips the
  axial vector $b$ as well as $U$ and $V$.

**$Q>0$ lies in the photon–field plane.** With Stokes referred to $(e_{(2)},-e_{(1)})$, $Q>0$
means $E\parallel e_{(2)}$, the projected field, which lies in the plane of $k$ and $b$ ✓.

**Pitch angle.** $\cos\theta_B=(k\cdot b)/(\omega|b|)$, carrying the sign of $\hat k\cdot\hat b$
(`integrator.jl:43–54`, computed from tetrad components). The unpolarized path uses the
invariant form of §4.

**Degenerate case.** If $b\parallel\hat k$ in the fluid frame, $\tilde e_{(2)}=0$. Then
$\sin\theta_B=0$ and all synchrotron coefficients vanish except $\rho_V$ and $j_V$-type terms,
which are rotation-invariant about $\hat k$. The code substitutes a coordinate axis for $b$
(`tetrads.jl:46–58`, following ipole's `set_Econ_from_trial`).

---

## 7. Transported readout frame and reverse accumulation

### 7.1 Parallel transport of the camera vertical vector and four-velocity (M21, part 1)

- **Transport equation.** $\dd V^\mu/\dd\lambda=-\Gamma^\mu_{\alpha\beta}k^\alpha V^\beta$ (KP Eq. 3; B20 Eq. 4
  for the polarization vector). Contracting $\Gamma^\mu_{\alpha\beta}=\tfrac12g^{\mu\lambda}(\partial_\alpha g_{\lambda\beta}+\partial_\beta g_{\lambda\alpha}-\partial_\lambda g_{\alpha\beta})$
  with $k^\alpha V^\beta$ gives
  $$2\Gamma^\mu_{\alpha\beta}k^\alpha V^\beta=g^{\mu\lambda}\big[(k\cdot\partial)g_{\lambda\beta}V^\beta+(V\cdot\partial)g_{\lambda\alpha}k^\alpha-k^\alpha V^\beta\partial_\lambda g_{\alpha\beta}\big].$$
  Only $\partial_i g_{\mu\nu}$ is needed, and it comes from the same dual-number evaluation as
  §3.1 (`geodesic_transport_rhs`, `integrator.jl:180–211`). The $\Gamma$ array is never formed.
- **Reparametrisation.** The code transports along the traced ray, whose tangent is
  $\dd x/\dd\lambda'=g^{\mu\nu}p_\nu=-k^\mu$. Parallel transport is independent of the
  parametrisation and orientation of the curve, so this equals transport along the physical ray.
- **Integrator.** The vectors are integrated as extra RK4 states with the geodesic, in the
  same four stages (`reverse_march.jl:278–291`).
- **Invariants.** Parallel transport preserves all inner products among $u_{\rm cam}$,
  $e_{(2),\rm cam}$ and $k$. The last is itself parallel-transported, since the curve is a
  geodesic. The tetrad rebuilt from the transported $(u,e_{(2)})$ and the local $k$ by the
  §6.4 construction therefore equals the transported camera tetrad
  (`reverse_march.jl:261–266`). The rebuild also re-orthonormalises it against the local
  $k$, which removes the RK4 drift of the inner products.
- **In vacuum.** Geometric optics transports the polarization vector $f^\mu$ parallel, with
  $f\perp k$ (B20 Eq. 4). Its components in a parallel-transported screen basis are
  therefore constant. So the Stokes vector read in the transported frame does not change
  between plasma segments, and no final rotation is needed at the camera.
- *Difference from MG18:* MG18 transports the coherency tensor $N^{\alpha\beta}$ with
  $\dd N^{\alpha\beta}/\dd\lambda=-\Gamma^\alpha_{\mu\nu}k^\mu N^{\nu\beta}-\Gamma^\beta_{\mu\nu}k^\mu N^{\alpha\nu}$
  (MG18 Eq. 16), split from the plasma step. At the camera it maps $N\to S$ by MG18 Eq. (14).
  Here two vectors are transported instead of a $4\times4$ Hermitian tensor. The plasma
  operator is exact per segment and never split from transport.
- *Difference from KP:* KP transports screen vectors $e_A$ (KP Eq. 3) together with the
  Stokes vector and uses a Strang split (KP Eq. 9).

### 7.2 Stokes rotation $R_n$ (M21, part 2)

- **Angle.** Let $E$ be the transported readout tetrad at segment $n$ and $e^{\rm fl}$ the
  fluid tetrad. Both have $e_{(3)}$ along the same $k$ (in their own frames). Define
  $$c=E_{(1)}\cdot e^{\rm fl}_{(1)},\quad s=E_{(1)}\cdot e^{\rm fl}_{(2)},\quad \cos\chi_n=c/\sqrt{c^2+s^2},\ \sin\chi_n=s/\sqrt{c^2+s^2}$$
  (`_qu_rotation`, `reverse_march.jl:185–192`).
- **Gauge invariance.** The two screens belong to different observers. Their transverse
  vectors differ by multiples of $k$, $E_{(1)}=\cos\chi\,e^{\rm fl}_{(1)}+\sin\chi\,e^{\rm fl}_{(2)}+a\,k+b\,u_{\rm fl}$.
  Contracting with $e^{\rm fl}_{(1,2)}$ removes the $k$ and $u$ parts, since
  $e^{\rm fl}_{(1,2)}\cdot k=e^{\rm fl}_{(1,2)}\cdot u_{\rm fl}=0$. Stokes parameters are
  invariant under $f\to f+ck$ (MG18, B20), so $\chi_n$ is the physical angle between the two screens.
- **Stokes rotation.** A field with components $(E_1,E_2)$ in the fluid screen has components
  $E'_1=\cos\chi E_1+\sin\chi E_2$ and $E'_2=-\sin\chi E_1+\cos\chi E_2$ in the readout
  screen. Substituting into $Q=|E_1|^2-|E_2|^2$ and $U=2{\rm Re}\,E_1E_2^*$ gives
  $$R_n=\begin{pmatrix}1&0&0&0\\0&\cos2\chi_n&\sin2\chi_n&0\\0&-\sin2\chi_n&\cos2\chi_n&0\\0&0&0&1\end{pmatrix},$$
  with $I$ and $V$ unchanged. $V$ is unchanged because both bases are right-handed about the
  same propagation direction. This matches `_rotate_qu` (`reverse_march.jl:196–207`).
- **Basis choice.** The code's Stokes are referred to $(e_{(2)},-e_{(1)})$, not $(e_{(1)},e_{(2)})$.
  That changes $(Q,U)\to(-Q,-U)$ in both frames, and $-\mathbf 1$ commutes with $R_n$, so the
  same $R_n$ applies ✓. This is the paper's "twice the angle between the fluid $e_{(1)}$ and
  the transported screen vector".
- **[paper]** method.tex:58 speaks of "the transported screen vector". The code transports
  the camera **vertical** $e_{(2)}$ and measures the angle from the rebuilt $E_{(1)}$. The
  wording should say "the $e_{(1)}$ of the readout tetrad rebuilt from the transported
  vertical vector and four-velocity".

### 7.3 Reverse accumulation (eq:reverse)

1. **Segments.** Number the plasma segments $n=0,1,\dots,N-1$ in integration order,
   camera-first. Segment $n$ maps the Stokes vector entering from the source side to the
   vector leaving toward the camera: $\boldsymbol{\mathcal S}^{\rm fl}_{{\rm out},n}=M_n\boldsymbol{\mathcal S}^{\rm fl}_{{\rm in},n}+b_n$ in the fluid basis (§5.3).
2. **Readout basis.** Express both sides in the readout basis with $\boldsymbol{\mathcal S}=R_n\boldsymbol{\mathcal S}^{\rm fl}$
   and $R_n^{-1}=R_n^{\sf T}$:
   $$\boldsymbol{\mathcal S}_{{\rm out},n}=T_n\boldsymbol{\mathcal S}_{{\rm in},n}+R_nb_n,\qquad T_n\equiv R_nM_nR_n^{\sf T}.$$
3. **Chaining.** Between segments the readout-frame Stokes vector is constant (§7.1). So
   $\boldsymbol{\mathcal S}_{{\rm in},n}=\boldsymbol{\mathcal S}_{{\rm out},n+1}$ and $\boldsymbol{\mathcal S}_{\rm obs}=\boldsymbol{\mathcal S}_{{\rm out},0}$.
4. **Unrolling** gives
   $$\boldsymbol{\mathcal S}_{\rm obs}=\sum_{n=0}^{N-1}G_nR_nb_n+G_N\boldsymbol{\mathcal S}_{\rm bg},\qquad G_n=T_0T_1\cdots T_{n-1},\ G_0=\mathbf 1,$$
   that is, $G_{n+1}=G_nR_nM_nR_n^{\sf T}$, which is eq:reverse. $\boldsymbol{\mathcal S}_{\rm bg}=0$ (no background source).
5. **Meaning of $G_n$.** $G_n$ is the Mueller transmission from the camera side of segment $n$ to the camera.
6. **Order of the product.** New factors multiply on the **right**, because a segment farther
   from the camera acts first on light that later passes through the nearer segments.
7. **Why this order is used.** It lets the geodesic (integrated outward) and the transfer
   (physically inward) share one loop. The per-ray state is only $(\boldsymbol{\mathcal S},G)$: 4 + 16 numbers.
   - *Difference from W22 and MG18:* they store the path and integrate source→camera.
8. **Unpolarized case.** With $G$ a scalar transmission,
   $\mathcal I_{\rm obs}=\sum_nB_n\prod_{k<n}A_k$ (`reverse_march.jl:6, 162–163`).
9. **Stopping criterion.** $G_n$ can serve as one: $\|G_n\|<\epsilon$. The current code runs
   until the geodesic stops.
10. **Admissibility clamp.** Applied to the partial sums (`reverse_march.jl:273`). The
    partial sums are the observed contribution of the camera-side part of the ray. For
    admissible coefficients each partial sum is admissible (§5.4), so the clamp acts only on
    round-off.

**Equivalence check.** Forward accumulation from the source
($\boldsymbol{\mathcal S}\leftarrow T_n\boldsymbol{\mathcal S}+R_nb_n$ for $n=N-1,\dots,0$) gives the same sum by
associativity of the matrix product. The docs report agreement between the CPU march and
the fused kernel, which run the same reverse form, to $1.4\times10^{-8}$ (relative $L_2$)
at $C=1280$.

### 7.4 Sampling and quadrature order (M25)

- **Pairing.** Segment $n$ uses the coefficients at the state $y_n$ at the start of RK4 step
  $n$, over the affine length $h_n$ of that step (`reverse_march.jl:151–165, 246–275`). This
  is left-endpoint quadrature of the coefficients along the ray, with local error
  $O(h^2\,\partial_\lambda\mathbf K)$. It is first order globally in $h\propto1/C$, matching
  the first-order convergence in the docs.
  - *Difference from W22:* W22 also samples at the step points. MG18 uses midpoint
    coefficients in its split scheme, which is second order.
- **[paper]** "so the per-ray state is independent of path length" (method.tex:72) follows
  from the reverse accumulation, not from where the coefficients are taken. The two clauses
  should be separated.

---

## 8. Termination rule and bound (M26)

### 8.1 The rule

The code stops a march when any of these hold (`march_stop`, `integrator.jl:37–67`):
- $d=r-\rhor<\epsilon_{\rm h}=10^{-3}$ (horizon backstop);
- $E_n=-k\cdot n>E_{\rm c}=10^2$ (redshift cut);
- the ray is outbound past the escape radius or the emission radius.

The polarized march additionally stops when the rebuilt readout tetrad has no spacelike
$e_{(2)}$ (`reverse_march.jl:265–269`).
- **[paper]** method.tex:74 reads "ends at $r-\rhor<10^{-3}$, where no spacelike screen vector
  exists". These are two separate conditions in the code: the backstop, and the screen-leg
  failure that occurs when $k$ drifts off the null cone on a coarse step. They should be
  stated separately.

**Normal-frame energy.** $n_\mu=(-\alpha,0,0,0)$, so $-k\cdot n=\alpha k^t=\alpha\,g^{t\nu}k_\nu$.
In CKS, $g^{tt}=-(1+f)$ and $g^{ti}=f\,l_i$ (from $l^t=-1$), giving
$$E_n=\frac{|f\,l_ik_i-(1+f)k_t|}{\sqrt{1+f}}$$
(`ks_radius_energy`, `integrator.jl:86–90`, with $p=-k$ inside the absolute value).

### 8.2 Omitted affine length $\Delta\lambda\lesssim\alpha_{\rm h}/(\kappa E_{\rm c})$

1. **Setup.** The traced ray is the time reverse of a photon that climbed out from just
   outside the future horizon. Near the horizon, use ingoing coordinates $(v,r)$ with the
   Kerr–Schild time $t=v-r+F(r)$, where $F$ is regular. Take Schwarzschild first for
   clarity; Kerr follows with $\kappa$ and $\alpha_{\rm h}$ replaced by their Kerr values.
2. **Near-horizon kinematics.** For the outgoing photon, $k^r=\dd r/\dd\lambda\to E$, which
   is finite. Also $k^v=g^{vr}k_r=2E/F_{\rm s}(r)$, with $F_{\rm s}=1-2M/r\simeq2\kappa(r-\rhor)$ and $\kappa=1/(4M)$.
3. **Coordinate-time rate.** $\dd t/\dd\lambda\simeq k^v\simeq E/[\kappa(r-\rhor)]$.
4. **Normal-frame energy.** $E_n=\alpha|\dd t/\dd\lambda|\simeq\alpha_{\rm h}E/[\kappa(r-\rhor)]$,
   with $\alpha_{\rm h}$ the KS lapse on the horizon (§2.1).
5. **Where the cut fires.** At $r_{\rm c}-\rhor=\alpha_{\rm h}E/(\kappa E_{\rm c})$.
6. **Remaining affine length.** $\Delta\lambda_{\rm rest}=(r_{\rm c}-\rhor)/|\dd r/\dd\lambda|\simeq\alpha_{\rm h}/(\kappa E_{\rm c})$, with $E=1$.

**Numbers.**
- Schwarzschild: $\alpha_{\rm h}=1/\sqrt2$ and $\kappa=1/4$ give $\Delta\lambda_{\rm rest}=0.0283$.
- Kerr: $\kappa=(r_+-r_-)/[2(r_+^2+a^2)]$. At $a=0.9$, $r_\pm=1\pm0.4359$ and $\kappa=0.1518$.
  - $\alpha_{\rm h}$ ranges from $(1+2/r_+)^{-1/2}=0.6465$ on the equator to $1/\sqrt2$ on the pole (§2.1).
  - So $\Delta\lambda_{\rm rest}\in[0.0426,0.0466]$. The docs quote a measured maximum of
    0.0450 against the polar bound 0.0466 ✓.
- In Kerr, $|\dd r/\dd\lambda|$ at the horizon depends on $(\lambda,\eta)$ and is at most of
  order $E$. The relation is therefore "$\lesssim$" up to that factor.
- **[paper]** The paper should state that $\alpha_{\rm h}$ is the lapse **in Kerr–Schild
  slicing** (finite on the horizon; the BL lapse vanishes there). symbols.md §8 records the same point.

**Omitted intensity.**
- Beyond the cut, the fluid-frame frequency obeys
  $\nu=\nuobs(-k\cdot u)\ge\nuobs E_n/[\gamma(1+v)]\ge\nuobs E_{\rm c}/(2\gamma)$. Here $\gamma=-u\cdot n$
  and $v$ is the fluid speed relative to $n$. The first inequality uses $-k\cdot u=\gamma E_n(1-\boldsymbol v\cdot\hat{\boldsymbol k})\ge\gamma E_n(1-v)=E_n/[\gamma(1+v)]$.
- Transmission is at most 1, so
  $$\Delta I_\nu\le\nuobs^3\,\frac{\rg}{\nuobs}\,\Delta\lambda_{\rm rest}\ \max_{\rm beyond\ cut}\frac{j_\nu}{\nu^2}$$
  (`integrator.jl:23–33`, checked ray by ray in `test/test_march_termination.jl`).
- Synchrotron $j_\nu/\nu^2$ falls steeply for $\nu\gg\nu_s$, and the frequency beyond the cut
  is at least $50\nuobs/\gamma$. This makes the bound small. The paper states a flux change
  below $10^{-8}$.

---

## 9. Dual-number parameter derivative (M27)

- **Seeding.** Let a scalar parameter $q$ ($\mathcal M$ or $R_{\rm high}$) enter only through
  the coefficients. Then $A_n(q)$ and $B_n(q)$ are smooth functions of $q$, and the geodesic
  is independent of $q$.
- **Image.** $I_{\rm obs}(q)=\nuobs^3\sum_nB_n(q)\prod_{k<n}A_k(q)$.
- **Exactness.** Forward-mode AD evaluates this expression in the dual algebra
  $\mathbb R[\epsilon]/\epsilon^2$. The result is $I_{\rm obs}+\epsilon\,\partial_qI_{\rm obs}$,
  exact to rounding for the **discrete** estimator. It follows from the product rule on the
  recurrence
  $$\mathcal I\leftarrow\mathcal I+\Pi B_n,\qquad \Pi\leftarrow\Pi A_n,$$
  where each quantity carries its dual part (`gpu_image_ad.jl:1–8`; NM25 makes the same
  argument for ipole).
- **Branches.** The branches (vacuum, cuts, saturation $\Delta\tau>100$) are piecewise. At a
  cut boundary the derivative is that of the active branch, and the true derivative has a
  jump there.
- **[paper]** The paper says "exact derivative". The precise wording is "the exact
  derivative of the discretised image, away from the emission-cut boundaries".
- **Mass scale.** For $q=\mathcal M$: $n_e\propto\mathcal M$, $B\propto\mathcal M^{1/2}$, and
  $\Theta_e$ is independent of $\mathcal M$. The dual propagates through the unit tuple
  (`gpu_image_ad.jl:7`).

---

## 10. Monte Carlo

### 10.1 Superphoton state and energy (M28)

- **State.** $(x^\mu,k_\mu,w)$. Here $k_\mu$ is the **physical** covector in units of
  $m_ec^2$ (with $c=1$ in geometric units), and $w$ is the number of physical photons carried.
- **Energy.** $E=-k_t=h\nu_\infty/(m_ec^2)$, conserved in a stationary metric (§3.2).
- **Normalisation difference.** This differs from the imaging covector ($k_t=-1$), and
  symbols.md flags the double use. The MC tracer integrates the **physical** photon forward
  in time: emission, then escape. There is no reversal.
- **Local frequency.** $\nu=(-k\cdot u)\,m_ec^2/h$.
- *Difference from D09 and GPM:* they carry contravariant $k^\mu$. The map is $k_\mu=g_{\mu\nu}k^\nu$ (`src/montecarlo/generate.jl:73`).

### 10.2 Weights and generation

1. **Photon emission rate.** In each zone of coordinate volume $\Delta V$ (with $\sqrt{-g}$
   included), the rate per unit $\ln\nu$ is
   $\frac{\dd\dot N}{\dd\ln\nu}=\frac{\Delta V}{h}\int j_\nu\,\dd\Omega$. This is D09 Eqs. (1)–(3)
   and GPM Eqs. (1), (5) with $\Delta t=1$ s.
2. **Weight choice.** $w(\nu)$ is chosen so that the superphoton density per $\ln\nu$ is flat.
   Summed over zones, the expected number per unit $\ln\nu$ is then
   $N_{\rm s}/\Delta\ln\nu_{\rm band}$:
   $$w(\nu)=\frac{\Delta\ln\nu_{\rm band}}{hN_{\rm s}}\sum_{\rm zones}\Delta V\!\int\! j_\nu\,\dd\Omega .$$
   This is GPM Eq. (2) and D09 Eq. (34). The code builds it with
   `build_weight_table(zones, N_s/band)` (`src/montecarlo/emission_tables.jl:96–113`;
   `mc_cpu.jl:33–34`, `mc_gpu.jl:228–230`).
3. **Per-zone count.** It follows as
   $N_{{\rm s},i}=\frac{\Delta V_i}{h}\int\dd\ln\nu\,\frac{\int j_\nu\dd\Omega}{w(\nu)}$
   (`zone_nsuperphotons`, `emission_tables.jl:127+`), and $\sum_iN_{{\rm s},i}=N_{\rm s}$ ✓.
4. **Unbiasedness.** A superphoton drawn from
   $\dd N_{\rm s}\propto\frac{1}{w}\frac{\dd N}{\dd\ln\nu}\dd\ln\nu$ and carrying weight $w$ gives
   $E[\sum w]=\dd N$ in every $\ln\nu$ bin. This holds for any smooth positive $w(\nu)$.
   Log-linear interpolation of $w$ in $\ln\nu$ therefore biases nothing, as long as the same
   interpolated $w$ is used for both the count and the weight.
   - **Code check.** The frequency rejection accepts with probability
     $(\int j_\nu\dd\Omega/w)/\max(\cdot)$ using the same interpolated $w$ that the packet
     carries (`src/montecarlo/generate.jl:40–48`), so the spectral shape is unbiased. The
     per-zone count $N_{{\rm s},i}$ and the envelope $\max(\cdot)$ are evaluated on the grid
     nodes (`emission_tables.jl:124`, as gpumonty `init_zone`). This leaves a normalisation
     error of the order of the $\ln\nu$ quadrature error of the node sum, and an envelope
     clip wherever the continuous maximum between nodes exceeds the node maximum. Both vanish
     as the table is refined; they are properties of the grmonty scheme, not new.

### 10.3 Kirchhoff absorption and the $\nu L_\nu$ estimator

- **Weight decay.** Along a step, $\dd w/\dd\tau_a=-w$ (D09 Eq. 14, GPM Eq. 18), with
  $\dd\tau_a=(\nu\alpha_\nu)\,\mathcal C\,\dd\lambda$ and $\mathcal C=\rg h/(m_ec^2)$ (D09 Eq. 13).
  - $\mathcal C$ converts the code affine parameter to the physical $\lambda_{\rm c}$ of §5.1.
    The wave vector in Hz is $K=(m_ec^2/h)\,k$, and $\rg\,\dd x=K\,\dd\lambda_{\rm c}$ with
    $\dd x=k\,\dd\lambda$. So $\dd\lambda_{\rm c}=\rg h/(m_ec^2)\,\dd\lambda=\mathcal C\,\dd\lambda$,
    and $\dd\tau_a=\nu\alpha_\nu\,\dd\lambda_{\rm c}$ as in §5.1.
  - The code applies the trapezoid rule $\Delta\tau_a=\tfrac12[(\nu\alpha)_n+(\nu\alpha)_{n+1}]\mathcal C h$
    (D09 Eq. 16) and sets $w\leftarrow we^{-\Delta\tau_a}$ (`mc_cpu.jl:329–372`).
  - $\alpha_\nu=j_\nu/B_\nu(T_e)$ (Kirchhoff) for the thermal part.
- **Step length.** $h$ is chosen so that the **coordinate** step is $(r-\rhor)/C$, i.e.
  $h=(r-\rhor)/(C|k^i|)$ (`_grmhd_dlam`, `mc_cpu.jl:317–324`, after grmonty `stepsize`).
  $|k|$ is tiny in these units, and this choice keeps the geometry independent of it.
- **Estimator.** Collect the escaping packets in a bin of solid angle $\Delta\Omega$ and log
  width $\Delta\ln\nu$. Energy per unit time into the bin is $m_ec^2\sum_jw_jE_j$. The flux at
  distance $R$ is that divided by $R^2\Delta\Omega\,\nu\,\Delta\ln\nu$. Isotropic-equivalent
  $\nu L_\nu=4\pi R^2\nu F_\nu$ then gives
  $$\nu L_\nu=m_ec^2\,\frac{4\pi}{\Delta\Omega}\,\frac{1}{\Delta\ln\nu}\sum_jw_jE_j$$
  (D09 Eq. 32 with $\Delta t=1$; `src/montecarlo/spectrum.jl:4–8, 158–170`).
  - Folded bins: $\Delta\Omega_j=2\cdot2\pi(\cos\theta_{j-1}-\cos\theta_j)$, with
    $\sum_j\Delta\Omega_j=4\pi$ ✓ (`spectrum.jl:52–60`).
- **Unbiasedness.** $E[\sum w_jE_j]$ equals the emitted energy times $e^{-\tau}$ along each
  path. This is the Monte Carlo form of the formal solution.

### 10.4 Scattering: hot cross section and exponential transform (M29)

- **Hot cross section** (D09 Eq. 24; GPM Eq. 23):
  $$\sigma_{\rm h}(\varepsilon,\Theta_e)=\frac{1}{n_e}\int\dd^3p\,\frac{\dd n_e}{\dd^3p}(1-\mu_e\beta_e)\,\sigma_{\rm KN}(\varepsilon_e),\qquad \varepsilon_e=\varepsilon\gamma_e(1-\mu_e\beta_e),$$
  with $\varepsilon=h\nu/(m_ec^2)$ and $\sigma_{\rm KN}$ from D09 Eq. (26).
  - The factor $1-\mu_e\beta_e$ is the flux factor $(-k\cdot p_e)/(k^tp_e^t)$ in the fluid
    frame (D09 Eq. 21, GPM Eq. 22).
  - Then $\alpha_{\nu,\rm s}=n_e\sigma_{\rm h}$ (D09 Eq. 25), and $\nu\alpha_{\nu,\rm s}$ is invariant.
  - Code: a 221×81 log–log table built by midpoint quadrature (`src/montecarlo/hotcross.jl:1–30`).
- **Exponential transform, as implemented here** (`_mc_scatter_event`, `src/montecarlo/mc_common.jl:33–50`).
  In each step of scattering depth $\Delta\tau_s$, with bias $b\ge1$:
  1. Draw $x_1=-\ln U$.
  2. Spawn a child if $b\Delta\tau_s>x_1$. The probability of this is
     $p_{\rm sp}=1-e^{-b\Delta\tau_s}$.
  3. Give the child weight
     $w_{\rm ch}=w\,\frac{1-e^{-\Delta\tau_s}}{1-e^{-b\Delta\tau_s}}$, attenuated by absorption
     up to the event point, which lies at fraction $x_1/(b\Delta\tau_s)$ of the step.
  4. Deplete the parent deterministically: $w\leftarrow we^{-\Delta\tau_s}$.
- **Unbiasedness.** $E[w_{\rm ch}]=p_{\rm sp}w_{\rm ch}=w(1-e^{-\Delta\tau_s})$, the physical
  scattered fraction. Adding the depleted parent $we^{-\Delta\tau_s}$ conserves the weight in
  expectation for every $b\ge1$ ✓. As $b\Delta\tau_s\to0$, $w_{\rm ch}\to w/b$.
- **Difference from D09** (Sect. 5.1, `ms.tex:536–555`) **and GPM** (`main.tex:193–196`):
  - They use $p=1-e^{-b\tau_s}$ with the scattered weight $w/b$ and the parent reset to $w(1-1/b)$
    at each event. That conserves weight **per event**, not in expectation per step.
  - They draw the scattering depth once per flight. Here a fresh draw is made in each step.
  - Both schemes are unbiased. This one keeps the parent's decay deterministic, and it makes
    the event count per step a Bernoulli variable, which suits the layered GPU relaunch
    (docs, Monte Carlo).
  - **[paper]** "Exponential transform of Ref. Kahn1950" should add that the weights are
    $w(1-e^{-\Delta\tau})/(1-e^{-b\Delta\tau})$ per step. This is not grmonty's $w/b$.
- **Event position.** Given a spawn, $x_1/(b\Delta\tau_s)$ is exponentially distributed with
  rate $b\Delta\tau_s$, truncated to $[0,1]$. For $b\Delta\tau_s\ll1$ this is uniform, which
  is the physical position distribution at first order.
- **Bias function.** $b=\max\big(1,\ b_0(n_{\rm sc}+1)^2\big)$ (`grmhd_bias`, `mc_cpu.jl:301–302`).
  The sphere test uses $b_0=n_e\sigma_TR$ (GPM Eq. 35; `track.jl:112–120`).
- **Electron sampling.** The scattering electron is drawn from the Maxwell–Jüttner
  distribution weighted by the flux factor, with the rejection scheme of Canfield et al.
  (1987) as in D09 Sect. 5.3. This uses $\chi^2$ and $\Gamma$ deviates from the Philox stream
  (`src/montecarlo/rng.jl:230–285`; Marsaglia–Tsang for $\Gamma$).
- **Rest-frame kinematics.** Exact Klein–Nishina (D09 Eqs. 29–30).

### 10.5 Stream keys by ancestry (M30, part 1)

**Identity.** With $g<2^G$ the global primary index, $e_j<2^E$ the step ordinal at which the
generation-$j$ ancestor spawned, and $k\le K\le15$ the scattering layer:
$$I=g+\sum_{j=1}^{k}e_j\,2^{G+(j-1)E}+k\,2^{124}$$
(`src/montecarlo/rng.jl:116–175`).

**Injectivity.** The fields occupy disjoint bit ranges:
- $[0,G)$ for $g$;
- $[G+(j-1)E,\,G+jE)$ for $e_j$;
- $[124,128)$ for $k$.

This requires $G+KE\le124$, which `MCStreamLayout` enforces.
- The widths are $G={\rm bitwidth}(N_{\rm primary})$, which ensures $g\le N<2^G$, and
  $E={\rm bitwidth}(t_{\max}-1)$, which ensures $e_j\le t_{\max}-1<2^E$.
- Hence $I$ determines $(g,k,e_1,\dots,e_k)$ uniquely, and distinct genealogies give distinct $I$.

**Philox inputs.**
- The key is $({\rm seed},I_{0:31})$ and the counter is $({\rm draw},I_{32:63},I_{64:95},I_{96:127})$.
- For a fixed key, Philox4x32-10 is a bijection of the 128-bit counter. Draws with different
  counters therefore never collide within one key.
- Different keys give streams that are independent in the design sense of Salmon et al. (2011).
- Every (seed, genealogy, draw) triple is a distinct Philox input.

**Child stream.** `mc_child_rng(parent, e, L)` writes $e$ into field $k+1$ and increments the
layer tag. It depends only on the parent's identity and the spawn ordinal. It does not
depend on which thread, chunk or device runs the parent.

**Consequence.** The random numbers seen by each photon depend only on its genealogy. So
the set of spawned children, and each escaping packet's $(x,k,w)$, is independent of
scheduling.

### 10.6 Exact bin sums (M30, part 2)

1. **Doubles as integers.** A finite double is $x=m\,2^q$ with $|m|<2^{53}$ and $q\ge-1074$,
   so $x/2^{-1074}$ is an integer. Its magnitude is below $2^{1024+1074}=2^{2098}$.
2. **Accumulator.** The accumulator stores that integer in 68 signed 64-bit limbs of 31 bits
   each ($68\times31=2108\ge2098$ ✓).
3. **One addend.** It is $m$ shifted by $r\le30$ bits, so at most 83 bits. It touches three
   consecutive limbs with increments $|\delta|<2^{31}$ (`xsum_parts`, `src/montecarlo/exact_sum.jl:37–57`).
4. **Capacity.** A signed 64-bit limb absorbs at least $\lfloor(2^{63}-1)/(2^{31}-1)\rfloor\ge2^{32}-1$
   such increments without carries.
5. **Order independence.** Integer addition is associative and commutative. The limb vector
   after any permutation or grouping of the addends is therefore identical, including atomic
   interleavings on a device.
6. **Result.** The final value is the exact sum, rounded once to nearest-even, so it does
   not depend on order.
7. **Consequence.** Together with §10.5, $\nu L_\nu$ is bitwise independent of device count,
   thread count and schedule (method.tex:80).
- **[paper]** "Exact fixed-point bin sums" is accurate: the sums are fixed point at
  quantum $2^{-1074}$. The paper could say "a 2108-bit integer (long) accumulator", which is
  more informative.

---

## 11. Issues found in method.tex by these derivations

1. **l.14.** The reference observer for $\nuobs$ is left implicit. $\nuobs$ is the frequency
   at infinity. The camera normal observer sees $\nuobs(1+M/d)$ (§4).
2. **l.56.** The units of $\lambda_{\rm c}$ and $\boldsymbol{\mathcal S}$ are unstated, and the phrase "affine
   parameter of the momentum in frequency units" is hard to parse (§5.1).
3. **l.58.** "Transported screen vector" should be "the $e_{(1)}$ of the tetrad rebuilt from the
   transported vertical vector and four-velocity". The index $n$ of $R_n$ is also used before
   it is defined at l.72 (§7.2; symbols.md).
4. **l.59.** The reason $j_U=\alpha_U=\rho_U=0$ is not given. It holds because the field has no
   $e_{(1)}$ component, so $e_{(2)}$ is the only transverse axis (§6.4).
5. **l.72.** Coefficients at the step start make the quadrature first order. The "per-ray state
   independent of path length" follows from the reverse form, not from the sampling (§7.4).
6. **l.74.** The horizon backstop and the missing-screen-vector stop are different conditions.
   $\alpha_{\rm h}$ must be the KS lapse. The bound is $\alpha_{\rm h}/(\kappa E_{\rm c})$ up to the
   factor $|\dd r/\dd\lambda|^{-1}$, which is of order 1 (§8).
7. **l.75.** "Exact derivative" should be "exact derivative of the discrete image, away from
   the emission-cut boundaries" (§9).
8. **l.80.** The exponential-transform weights differ from grmonty. The per-step child weight is
   $w(1-e^{-\Delta\tau})/(1-e^{-b\Delta\tau})$ and the parent decays deterministically (§10.4).
9. **Docs, not the paper.** `docs/src/method.md` still says the superposed metric uses a full
   $4\times4$ inverse. The code uses the $N\times N$ Woodbury form (§2.2).
10. **§6.4 zero-$U$ argument.** If the appendix gives a reason for $j_U=\alpha_U=\rho_U=0$, it
    should use the spin-2 form of the linear coefficients about $\hat k$. A mirror
    argument does not work, because $b$ is axial.
11. **l.80.** Device-count independence of Monte Carlo spectra holds for the device kernels.
    On a GPU those kernels run only the sphere model (`src/montecarlo/mc_gpu.jl:304, 516`).
    The GRMHD Monte Carlo runs on CPU threads, where the invariance is over thread count
    (code_appendix_material.md §5).
