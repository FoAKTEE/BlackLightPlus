# Symbol table — BlackLightPlus method paper (arxiv-2609.26298)

Scope: every mathematical symbol in the manuscript, in the order `paper/main.tex` inputs the
sections (abstract, introduction, related_work, method, experiments, results, discussion,
conclusion, acknowledgments, appendix), figure captions (`\paperfig` in the section files) and
tables included. `paper/figures/CAPTIONS.md` is not input by `main.tex` and is excluded.
Line numbers refer to `paper/sections/<file>.tex` at the working-tree state of branch `paper`.
Conventions follow `results/blp/paper_arxiv-2609.26298/conventions.md` (signature $(-,+,+,+)$,
$G=c=M=1$ for geometry, cgs for plasma and radiation, ray state $p_\mu=-k_\mu$ in the code, IEEE/IAU
Stokes).

Status flags:
- **OK** — defined in words or by an equation at or before first use (same sentence counts).
- **DEFINED-IMPLICITLY** — meaning follows only from a formula or a name, with no explicit statement.
- **USED-BEFORE-DEFINITION** — defined later in the text than its first use.
- **NEVER-DEFINED** — no definition anywhere in the manuscript.
- **STANDARD** — physical constant or universal notation; a one-time definition is still customary
  in a code paper (ApJ/MNRAS style) and is listed for completeness.

Units: "geom." = geometric units with lengths in $r_{\rm g}$; "dimensionless" as stated; cgs otherwise.

## 1. Geometry, coordinates, spacetimes

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $x$ (closure argument), $x^\mu$ | spacetime point; coordinates $(t,x,y,z)$ | geom. | introduction.tex:15 | method.tex:13 | USED-BEFORE-DEFINITION (intro uses $x\mapsto g_{\mu\nu}(x)$; method.tex:13 fixes $x^\mu=(t,x,y,z)$) |
| $g_{\mu\nu}$ | spacetime metric | dimensionless | introduction.tex:15 | introduction.tex:15 ("metric closure") | DEFINED-IMPLICITLY |
| $g^{\mu\nu}$ | inverse metric | dimensionless | method.tex:21 | method.tex:21 | OK |
| $\mu,\nu,\alpha,\beta$ (indices) | spacetime indices $0..3$ | — | method.tex:13 | not defined | STANDARD (index $\alpha,\beta$ collide with lapse, screen coordinates, absorptivity, plasma $\beta$; see §7) |
| $i,j$ (indices) | spatial indices $1..3$ | — | method.tex:22 | not defined | STANDARD (collides with inclination $i$; see §7) |
| $G,c$ | gravitational constant, speed of light | cgs | method.tex:12 | method.tex:12 (set to 1) | STANDARD |
| $M$ | black-hole mass | g; length unit $r_{\rm g}$ in geom. | method.tex:12 ("$G=c=M=1$") | experiments.tex:6 ("black-hole mass $M$") | USED-BEFORE-DEFINITION |
| $r_{\rm g}$ | gravitational radius $GM/c^2$ | cm | method.tex:12 | method.tex:12 | OK |
| $(-,+,+,+)$ | metric signature | — | method.tex:12 | method.tex:12 | OK |
| $t,x,y,z$ | Cartesian Kerr–Schild coordinates | geom. | method.tex:13 | method.tex:13 | OK |
| $\eta_{\mu\nu}$, $\eta^{\mu\nu}$ | Minkowski metric ${\rm diag}(-1,1,1,1)$ | dimensionless | method.tex:13 | not defined | NEVER-DEFINED (also collides with Carter constant $\eta$, method.tex:37) |
| $f$ | Kerr–Schild scalar $f=2r^3/(r^4+a^2z^2)$ | dimensionless | method.tex:13 | not defined | NEVER-DEFINED |
| $l_\mu$, $l^\mu$ | Kerr–Schild null covector, $l_\mu=\big(1,\tfrac{rx+ay}{r^2+a^2},\tfrac{ry-ax}{r^2+a^2},\tfrac{z}{r}\big)$, null w.r.t. both $\eta$ and $g$ | dimensionless | method.tex:13 | not defined | NEVER-DEFINED |
| $\hat z$ | spin direction | — | method.tex:13 | method.tex:13 | OK |
| $a$ | dimensionless spin $J/(GM^2/c)$ | dimensionless | method.tex:40 (fig. analytic caption, $a=0.94$) | not defined | NEVER-DEFINED (also needed in $f$, $l_\mu$, $r$) |
| $f_A$, $l^{A}_{\mu}$ | Kerr–Schild scalar and null covector of hole $A$ (boosted, displaced) in the superposed metric | dimensionless | method.tex:21 | not defined | NEVER-DEFINED (index $A$ collides with Stokes index $A$, method.tex:15) |
| $\alpha$ (lapse) | ADM lapse | dimensionless | method.tex:22 | method.tex:22 ("ADM fields") | DEFINED-IMPLICITLY |
| $\beta^i$ | ADM shift vector | dimensionless | method.tex:22 | method.tex:22 ("ADM fields") | DEFINED-IMPLICITLY |
| $\gamma_{ij}$ | ADM spatial metric | dimensionless | method.tex:22 | method.tex:22 | DEFINED-IMPLICITLY |
| $\tilde\gamma_{ij}$ | conformal spatial metric of the Z4c formulation | dimensionless | method.tex:22 | not defined | NEVER-DEFINED |
| $\chi$ (Z4c) | Z4c conformal factor, $\gamma_{ij}=\tilde\gamma_{ij}/\chi$ | dimensionless | method.tex:22 | not defined | NEVER-DEFINED (collides with EVPA $\chi$, method.tex:16) |
| $C^1$ | once continuously differentiable | — | method.tex:22 | not defined | STANDARD (collides with step constant $C$) |
| $r$ | Kerr–Schild radius, root of $r^4-(x^2+y^2+z^2-a^2)r^2-a^2z^2=0$ | geom. | method.tex:30 | not defined | NEVER-DEFINED (not the Euclidean radius in CKS) |
| $r_{\rm h}$ | outer horizon radius $1+\sqrt{1-a^2}$ | geom. | method.tex:30 | method.tex:30 | OK |
| $b_{\rm c}$ | critical impact parameter ($3\sqrt3$ for $a=0$) | geom. | results.tex:16 | not defined | NEVER-DEFINED (collides with $b^\mu$, $b_n$) |
| $M_\odot$ | solar mass | g | results.tex:32 | not defined | STANDARD |
| $x_i$ (fig. z4c) | Cartesian coordinates of the grid | geom. | results.tex:35 | method.tex:13 (as $x,y,z$) | OK |

## 2. Geodesics and camera

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $k^\mu$ | wave vector $\dd x^\mu/\dd\lambda$ | dimensionless ($k_t=-1$ at camera) | method.tex:14 | method.tex:14 | OK |
| $k_\mu$ | covariant wave vector, integrated state (code stores $p_\mu=-k_\mu$) | dimensionless | method.tex:14 | method.tex:14 ("covariant form") | DEFINED-IMPLICITLY |
| $k_t$ | time component of $k_\mu$; $-k_t$ = energy at infinity | dimensionless | method.tex:14 | method.tex:14 | OK |
| $\lambda$ (affine) | affine parameter | geom. ($r_{\rm g}$) | method.tex:14 | method.tex:14 (only through $k^\mu=\dd x^\mu/\dd\lambda$) | DEFINED-IMPLICITLY (collides with conserved $\lambda$, method.tex:37) |
| $\lambda'$ | $-\lambda$, backward integration parameter | geom. | method.tex:33 | method.tex:33 | OK |
| $u^\mu$ | fluid (or observer) four-velocity | dimensionless | method.tex:14 | method.tex:14 | OK |
| $H$ | Hamiltonian $\tfrac12 g^{\mu\nu}k_\mu k_\nu$ | dimensionless | method.tex:24 | method.tex:24 | OK |
| $\partial_\mu$ | coordinate partial derivative | geom.$^{-1}$ | method.tex:27 | not defined | STANDARD |
| $h$ | RK4 step in $\lambda$, $(r-r_{\rm h})/C$ | geom. | method.tex:30 | method.tex:30 | OK |
| $C$ | step-control constant (80) | dimensionless | method.tex:30 | method.tex:30 | OK |
| $d$ | camera distance | geom. | method.tex:32 | method.tex:32 | OK |
| $i$ | inclination of the line of sight from $+\hat z$ | rad | method.tex:32 | method.tex:32 | OK (collides with $\sqrt{-1}$ at method.tex:16 and ion subscript $i$; see §7) |
| $(\alpha,\beta)$ (screen) | screen coordinates of a pixel; equal to Bardeen coordinates | geom. | method.tex:32 | method.tex:32 ("pixel $(\alpha,\beta)$"), method.tex:34 (Bardeen) | OK (collide with lapse and plasma $\beta$) |
| $x^i$ (pixel) | pixel launch position | geom. | method.tex:32 | method.tex:32 | OK |
| $\hat\alpha,\hat\beta$ | screen unit vectors; $\hat\beta$ projected spin axis (north), $\hat\alpha=\hat y$ | — | method.tex:32 | method.tex:32 | OK |
| $\hat n$ | outward line-of-sight unit vector | — | method.tex:33 | method.tex:33 | OK (collides with $n$, $n^\mu$) |
| $L_z$ | axial angular momentum $xk_y-yk_x$ | geom. | method.tex:34 | method.tex:34 | OK |
| $k_x,k_y$ | spatial components of $k_\mu$ | dimensionless | method.tex:34 | method.tex:14 (by index) | OK |
| $e_{(a)}$, $e_{(0)},\dots,e_{(3)}$ | orthonormal tetrad legs (camera, then fluid) | dimensionless | method.tex:35 | method.tex:35 (camera), method.tex:59 (fluid) | OK; tetrad index $(a)$ not named as frame index |
| normal observer | observer with $u_\mu=-\alpha\,\delta^t_\mu$ | — | method.tex:35 | not defined | DEFINED-IMPLICITLY (its four-velocity $n^\mu$ first appears unnamed at method.tex:74) |
| $\lambda$ (conserved) | energy-scaled angular momentum $-k_\phi/k_t$ ($=L_z/E$) | geom. | method.tex:37 | method.tex:37 ("the conserved $(\lambda,\eta)$") | NEVER-DEFINED (meaning not given; overloads affine $\lambda$) |
| $\eta$ (Carter) | energy-scaled Carter constant $Q_{\rm C}/E^2$ | geom.$^2$ | method.tex:37 | method.tex:37 | NEVER-DEFINED (overloads $\eta_{\mu\nu}$) |
| $(r_s,\phi_s,t_s)$ | Boyer–Lindquist radius, azimuth, time of the equatorial crossing | geom. | method.tex:37 | method.tex:37 | OK for "crossing"; $\phi$ (azimuth) NEVER-DEFINED, BL vs KS coordinate not stated |
| $n$ (image order) | number of half-orbits about the photon shell before the crossing ($0,1,2$) | integer | method.tex:37 | method.tex:37 | OK (collides with segment index $n$) |
| $I$, $I_{\max}$ | image intensity and its peak | arbitrary / cgs | method.tex:40 (fig. analytic caption) | not defined | DEFINED-IMPLICITLY (relation to $I_\nu$ and $S_\nu$ not stated) |
| $\beta=0$ (caption) | screen row | geom. | method.tex:40 | method.tex:32 | OK |
| $n^\mu$ | normal-observer four-velocity | dimensionless | method.tex:74 | not defined | NEVER-DEFINED |
| $E_{\rm c}$ | cut on normal-observer energy $-k_\mu n^\mu$ ($10^2$) | dimensionless | method.tex:74 | method.tex:74 | OK |
| $\kappa$ | horizon surface gravity | geom.$^{-1}$ | method.tex:74 | method.tex:74 | OK (collides with $\kappa$ distribution, experiments.tex:16) |
| $\alpha_{\rm h}$ | Kerr–Schild lapse $(1+f)^{-1/2}$ on the horizon | dimensionless | method.tex:74 | method.tex:74 | OK (slicing unstated; the BL lapse vanishes there) |

## 3. Radiation and polarized transfer

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $\nu$ | fluid-frame frequency $\nu_{\rm obs}(-k_\mu u^\mu)$ | Hz | method.tex:14 | method.tex:14 | OK |
| $\nu_{\rm obs}$ | observing frequency at the camera | Hz | method.tex:14 | not defined | DEFINED-IMPLICITLY |
| $I_\nu,Q_\nu,U_\nu,V_\nu$ | fluid-frame Stokes specific intensities | erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$ | method.tex:15 | method.tex:15–16 | DEFINED-IMPLICITLY (named "Stokes parameters" at :16) |
| $\boldsymbol{\mathcal S}$ | invariant Stokes vector $(I_\nu,Q_\nu,U_\nu,V_\nu)/\nu^3$ | erg s$^{-1}$ cm$^{-2}$ Hz$^{-4}$ sr$^{-1}$ | method.tex:15 | method.tex:15 | OK |
| $A$ (Stokes index) | $A\in\{I,Q,U,V\}$ | — | method.tex:15 | not defined | NEVER-DEFINED (range not stated; collides with hole label $A$) |
| $j_{\nu,A}$ | fluid-frame emissivity | erg s$^{-1}$ cm$^{-3}$ Hz$^{-1}$ sr$^{-1}$ | method.tex:15 | not defined | NEVER-DEFINED (word "emissivity" first at experiments.tex:16) |
| $\alpha_{\nu,A}$ | fluid-frame absorptivity | cm$^{-1}$ | method.tex:15 | not defined | NEVER-DEFINED ("absorptivities" at experiments.tex:16) |
| $\rho_{\nu,A}$ | fluid-frame rotativity | cm$^{-1}$ | method.tex:15 | not defined | NEVER-DEFINED ("rotativities" at experiments.tex:17) |
| $j_A$ | invariant emissivity $j_{\nu,A}/\nu^2$ | erg s$^{-1}$ cm$^{-3}$ Hz$^{-3}$ sr$^{-1}$ | method.tex:15 | method.tex:15 | OK |
| $\alpha_A$ | invariant absorptivity $\nu\alpha_{\nu,A}$ | Hz cm$^{-1}$ | method.tex:15 | method.tex:15 | OK |
| $\rho_A$ | invariant rotativity $\nu\rho_{\nu,A}$ | Hz cm$^{-1}$ | method.tex:15 | method.tex:15 | OK |
| $Q,U,V$ (unsubscripted) | Stokes parameters in IEEE/IAU sense | as $S_\nu$ | method.tex:16 | method.tex:16 | OK |
| $\chi$ (EVPA) | electric-vector position angle $\tfrac12\arg(Q+iU)$, east of north | rad | method.tex:16 | method.tex:16 | OK (collides with Z4c $\chi$) |
| $i$ (imaginary) | $\sqrt{-1}$ in $\arg(Q+iU)$ | — | method.tex:16 | not defined | STANDARD (collides with inclination $i$ defined later at :32) |
| $\boldsymbol j$ | invariant emission vector $(j_I,j_Q,j_U,j_V)^{\mathsf T}$ | as $j_A$ | method.tex:47 | not defined | DEFINED-IMPLICITLY |
| $\lambda_{\rm c}$ | $\lambda\,r_{\rm g}/\nu_{\rm obs}$, affine parameter of the momentum in frequency units | cm Hz$^{-1}$ | method.tex:47 | method.tex:56 ("where") | OK |
| $\mathbf K$ | invariant transfer (propagation) matrix | Hz cm$^{-1}$ | method.tex:47 | method.tex:48–53 | OK |
| $\alpha_I,\alpha_Q,\alpha_U,\alpha_V$ | invariant absorption coefficients | Hz cm$^{-1}$ | method.tex:49 | method.tex:15 | OK (collide with lapse $\alpha$ and screen $\alpha$ only through the letter) |
| $\rho_Q,\rho_U,\rho_V$ | invariant Faraday conversion ($Q$, $U$) and rotation ($V$) | Hz cm$^{-1}$ | method.tex:50 | method.tex:56 | OK; $\rho_U$ meaning (conversion in the $U$ axis) not stated |
| $\ell$, $\dd\ell$ | fluid-frame path length, $\dd\ell=\nu\,\dd\lambda_{\rm c}$ | cm | method.tex:56 | method.tex:56 | OK |
| $S_\nu$ | camera Stokes vector $\nu_{\rm obs}^3\boldsymbol{\mathcal S}$ | erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$ | method.tex:56 | method.tex:56 | OK (collides with $\boldsymbol{\mathcal S}$ notation only typographically) |
| $N^{\mu\nu}$ | coherency (polarization) tensor | as $S_\nu$ | method.tex:58 | method.tex:58 (named) | DEFINED-IMPLICITLY (no formula; not used elsewhere) |
| $R_n$ | $4\times4$ rotation of $(Q,U)$ by twice the angle between fluid $e_{(1)}$ and transported screen vector, segment $n$ | dimensionless | method.tex:58 | method.tex:58 | OK; index $n$ USED-BEFORE-DEFINITION (segment index defined at method.tex:72) |
| $u$ (unindexed) | fluid four-velocity as a vector | — | method.tex:59 | method.tex:14 | OK |
| projected field | $b^\mu$ projected orthogonal to $u$ and $k$ | — | method.tex:59 | not defined | USED-BEFORE-DEFINITION ($b^\mu$ enters at experiments.tex:6) |
| $j_Q<0$ | sign of linear emissivity in the photon–field basis | — | method.tex:60 | method.tex:59–60 | OK |
| $\Delta\lambda_{\rm c}$ | step of $\lambda_{\rm c}$ | cm Hz$^{-1}$ | results.tex:14 | not defined | DEFINED-IMPLICITLY |
| $\exp(-\mathbf K\Delta\lambda_{\rm c})$ | matrix exponential | dimensionless | results.tex:14 | not defined | STANDARD |

## 4. Fused kernel, reverse composition, termination

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $\boldsymbol{\mathcal S}_{\rm in},\boldsymbol{\mathcal S}_{\rm out}$ | invariant Stokes vector entering/leaving a segment (transfer direction, toward camera) | as $\boldsymbol{\mathcal S}$ | method.tex:66 | method.tex:66 | OK |
| $M_n$ | $4\times4$ homogeneous (Mueller) part of segment $n$, $\exp(-\mathbf K_n\Delta\lambda_{\rm c})$ | dimensionless | method.tex:66 | method.tex:66 | DEFINED-IMPLICITLY (only via the affine map; no formula) |
| $b_n$ | inhomogeneous (emission) part of segment $n$ | as $\boldsymbol{\mathcal S}$ | method.tex:66 | method.tex:66 | DEFINED-IMPLICITLY (collides with $b^\mu$, $b_{\rm c}$) |
| $\boldsymbol{\mathcal S}_{\rm obs}$ | accumulated camera Stokes vector (invariant) | as $\boldsymbol{\mathcal S}$ | method.tex:68 | method.tex:68 | DEFINED-IMPLICITLY |
| $G_n$ | transmission matrix from segment $n$ to the camera | dimensionless | method.tex:68 | method.tex:72 | OK (same sentence; collides with Newton's $G$, method.tex:12) |
| $\mathbf 1$ | $4\times4$ identity | — | method.tex:69 | not defined | STANDARD |
| $R_n^{\mathsf T}$ | transpose (inverse) of $R_n$ | — | method.tex:69 | not defined | STANDARD |
| $n$ (segment) | segment / RK-step index along the ray, $n=0$ at camera | integer | method.tex:58 | method.tex:72 | USED-BEFORE-DEFINITION |
| $10^{-3}$ backstop, $r-r_{\rm h}$ | horizon stop | geom. | method.tex:74 | method.tex:74 | OK |

## 5. Plasma and models

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $\Rhigh$ ($R_{\rm high}$) | ion-to-electron temperature ratio at high $\beta$ | dimensionless | method.tex:75 | experiments.tex:9 | USED-BEFORE-DEFINITION |
| "density scale" | $\mathcal M$ | g | method.tex:75 (in words) | experiments.tex:6 | USED-BEFORE-DEFINITION |
| $\mathcal M$ | density (mass) scale | g | experiments.tex:6 | experiments.tex:6 | OK ("density scale" of mass units; $\rho_{\rm unit}=\mathcal M/r_{\rm g}^3$) |
| $\rho_{\rm unit}$ | code-to-cgs density unit | g cm$^{-3}$ | experiments.tex:6 | experiments.tex:6 | OK |
| $B$ | fluid-frame magnetic-field strength | G | experiments.tex:6 | experiments.tex:6 | OK |
| $b^\mu$ | Heaviside–Lorentz magnetic four-vector (code units) | code | experiments.tex:6 | experiments.tex:6 | OK; "projected field" used at method.tex:59 |
| $\beta$ (plasma) | $p_{\rm gas}/p_{\rm mag}$ | dimensionless | experiments.tex:6 | experiments.tex:6 | OK (collides with screen $\beta$ and shift $\beta^i$) |
| $p_{\rm gas}$ | gas pressure | code / erg cm$^{-3}$ | experiments.tex:6 | not defined | DEFINED-IMPLICITLY |
| $p_{\rm mag}$ | magnetic pressure $b^\mu b_\mu/2$ | code | experiments.tex:6 | experiments.tex:6 | OK |
| $\sigma$ | magnetization $b^\mu b_\mu/\rho$ | dimensionless | experiments.tex:6 | experiments.tex:6 | OK (collides with $\sigma_{\rm h}$ only in letter) |
| $\rho$ | rest-mass density (code units) | code | experiments.tex:6 | not defined | NEVER-DEFINED (collides with $\rho_A$, $\rho_{\nu,A}$, $\rho_{\rm unit}$) |
| $R$ | $T_i/T_e$ | dimensionless | experiments.tex:9 | experiments.tex:9 | OK (collides with $R_n$) |
| $T_i,T_e$ | ion and electron temperatures | K | experiments.tex:9 | experiments.tex:9 | OK |
| $\Rlow$ ($R_{\rm low}$) | ratio at low $\beta$ | dimensionless | experiments.tex:9 | experiments.tex:9 | OK |
| $\beta_{\rm crit}$ | transition plasma $\beta$ (=1, absorbed in eq:rhigh) | dimensionless | not used | not defined | macro `\betacrit` unused; equation hard-codes $\beta_{\rm crit}=1$ without saying so |
| $k_{\rm B}$ | Boltzmann constant | erg K$^{-1}$ | experiments.tex:10 | not defined | STANDARD |
| $n_{\rm e},n_i$ | electron and ion number densities | cm$^{-3}$ | experiments.tex:10 | not defined | NEVER-DEFINED |
| $\mu$ (molecular weight) | mean molecular weight | dimensionless | experiments.tex:10 | not defined | NEVER-DEFINED (collides with index $\mu$) |
| $m_p$ | proton mass | g | experiments.tex:10 | not defined | STANDARD |
| $\gamma_i,\gamma_e$ | ion and electron adiabatic indices | dimensionless | experiments.tex:13 | experiments.tex:13 | OK (collide with $\gamma_{ij}$) |
| $\sigma_{\rm cut}$ | magnetization above which samples are vacuum | dimensionless | experiments.tex:14 | experiments.tex:14 | OK |
| $\kappa$ (distribution) | kappa-distribution index | dimensionless | experiments.tex:16 ("$\kappa$ electrons") | not defined | NEVER-DEFINED (collides with surface gravity $\kappa$; width $w$ of P16 not introduced) |
| power-law parameters $p,\gamma_{\min},\gamma_{\max}$ | power-law index and Lorentz-factor bounds | dimensionless | not used | not defined | referred to only as "power-law" (experiments.tex:16) |
| $B_\nu(T_e)$ | Planck function | erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$ | experiments.tex:16 | not defined | DEFINED-IMPLICITLY ("Kirchhoff's law"; collides with field $B$) |
| $\theta_B$ | fluid-frame angle between photon direction and field | rad | method.tex:59 (in words, "photon–field plane") / experiments.tex:17 (symbol) | experiments.tex:17 | OK |
| $\Theta_{\rm e}$ | dimensionless electron temperature $k_{\rm B}T_e/m_ec^2$ | dimensionless | method.tex:80 | not defined | NEVER-DEFINED (and used before $T_e$, experiments.tex:9) |
| $K_{0,1,2}$ | modified Bessel functions of the second kind | dimensionless | results.tex:15 | not defined | NEVER-DEFINED |

## 6. Monte Carlo

| Symbol | Meaning | Units | First used | Defined | Status |
|---|---|---|---|---|---|
| $(x^\mu,k_\mu,w)$ | superphoton state | geom. / $m_ec^2$ / dimensionless | method.tex:79 | method.tex:79 | OK |
| $w$ | superphoton weight: number of physical photons per superphoton | dimensionless | method.tex:79 | method.tex:79 | OK |
| $k_\mu$ (MC) | covariant momentum in units of $m_ec$ (not the $k_t=-1$ imaging normalization) | $m_ec^2$ | method.tex:79 | method.tex:79 ("in $m_ec^2$") | DEFINED-IMPLICITLY (normalization differs from method.tex:14; not stated) |
| $E$ | superphoton energy at infinity $-k_t$ | $m_ec^2$ | method.tex:79 | method.tex:79 | OK (collides with $E_{\rm c}$ normalization) |
| $m_e$ | electron mass | g | method.tex:79 | not defined | STANDARD |
| $\sigma_{\rm h}(\varepsilon,\Theta_e)$ | thermal-averaged (hot) Klein–Nishina cross section | cm$^2$ | method.tex:80 | method.tex:80 (in words) | OK for $\sigma_{\rm h}$ |
| $\varepsilon$ | photon energy in the fluid frame, $h\nu/m_ec^2$ | dimensionless | method.tex:80 | not defined | NEVER-DEFINED |
| $N_{\rm s}$, $\nu L_\nu$, $h$ (Planck) | superphoton count, spectrum, Planck constant | — | not used | not defined | macros `\Nsp`, `\nuLnu` unused; estimator not written |

## 7. Overloaded symbols and proposed disambiguation

Each proposal keeps the conventions.md choice for the quantity used most and renames the rarer one.
Macros listed exist in `paper/macros.tex` unless marked "new".

| Letter | Uses (file:line) | Proposal |
|---|---|---|
| $\alpha$ | ADM lapse (method.tex:22); screen coordinate (method.tex:32); invariant absorptivity $\alpha_A$ (method.tex:15); horizon lapse $\alpha_{\rm h}$ (method.tex:74); index $\alpha$ (method.tex:27) | Keep $\alpha,\beta$ for Bardeen screen coordinates (conventions.md; GL20, AART use them) and $\alpha_A$ for absorptivity (MG18, M21). Write the lapse as $N$ with shift $N^i$ (common in NR) or $\alpha_{\rm ADM}$; write $\alpha_{\rm h}$ as $N_{\rm h}$. Use Greek indices $\mu,\nu,\sigma,\tau$ in Eq. (geodesic) instead of $\alpha,\beta$: $\dot k_\mu=-\tfrac12\partial_\mu g^{\sigma\tau}k_\sigma k_\tau$. |
| $\beta$ | ADM shift $\beta^i$ (method.tex:22); screen coordinate (method.tex:32); plasma $\beta=p_{\rm gas}/p_{\rm mag}$ (experiments.tex:6, eq:rhigh) | Screen: $\beta$. Shift: $N^i$ (paired with lapse $N$). Plasma: $\beta_{\rm p}$, with $\beta_{\rm crit}$ (`\betacrit`) written explicitly in eq:rhigh. |
| $\lambda$ | affine parameter (method.tex:14); conserved $\lambda=L_z/E$ (method.tex:37); $\lambda'$, $\lambda_{\rm c}$ | Keep affine $\lambda$ (`\aff`). Conserved quantities of GL20/AART: $\hat\lambda\equiv L_z/E$, $\hat\eta\equiv Q_{\rm C}/E^2$, defined once. |
| $\eta$ | Minkowski $\eta_{\mu\nu}$ (method.tex:13); Carter constant (method.tex:37) | Keep $\eta_{\mu\nu}$; Carter as $\hat\eta$ (see above). |
| $\chi$ | EVPA (method.tex:16, `\EVPA`); Z4c conformal factor (method.tex:22) | Keep EVPA $\chi$ (conventions.md). Z4c factor: $\chi_{\rm Z4c}$ or write $\gamma_{ij}=\psi^4\tilde\gamma_{ij}$ with $\chi\equiv\psi^{-4}$ named once as "conformal factor". |
| $i$ | inclination (method.tex:32, `\incl`); $\sqrt{-1}$ (method.tex:16); spatial index (method.tex:22,32); ion subscript $T_i,n_i,\gamma_i$ (experiments.tex:9–13) | Inclination: $\theta_{\rm o}$ (GL20/AART/Krang notation), by redefining `\incl` once. Imaginary unit: $\mathrm{i}$ (upright). Ion subscript: roman $\rm i$ ($T_{\rm i}$, $n_{\rm i}$, $\gamma_{\rm i}$), matching $n_{\rm e}$. |
| $n$ | image order (method.tex:37); segment index (method.tex:58, 66–72); $\hat n$ line of sight (method.tex:33); normal observer $n^\mu$ (method.tex:74); $n_{\rm e},n_i$ densities | Image order $n$ (Johnson+2020, AART, Krang); segment index $m$ ($s$, $k$, $j$ are taken by path length, wave vector, emission); line of sight $\hat e_{\rm los}$; normal observer $n^\mu$, defined once as $n_\mu=-N\delta^t_\mu$. |
| $\kappa$ | surface gravity (method.tex:74); kappa distribution (experiments.tex:16) | Surface gravity: $\kappa_{\rm h}$ or $g_{\rm H}$; distribution index keeps $\kappa$ (P16, M21). |
| $G$ | Newton's constant (method.tex:12); transmission $G_n$ (method.tex:68) | Transmission: $\mathbf T_m$ (bold matrix, like $\mathbf K$), with $\mathbf M_m$, $\mathbf R_m$ bold for all $4\times4$ matrices. |
| $b$ | magnetic four-vector $b^\mu$ (experiments.tex:6); segment source $b_n$ (method.tex:66); critical impact parameter $b_{\rm c}$ (results.tex:16) | Segment source: $\boldsymbol{\mathcal E}_m$ or $\boldsymbol s_m$; critical impact parameter: $b_{\rm c}$ acceptable once defined, or $\tilde r_{\rm c}$. |
| $A$ | Stokes index (method.tex:15); hole label (method.tex:21) | Stokes index $A\in\{I,Q,U,V\}$ (as in the macros; conventions.md writes $S$); hole label $h\in\{1,2\}$ ($f_h$, $l^{(h)}_\mu$), since $(a)$ is the tetrad index and $I$ a Stokes label. |
| $\rho$ | rest-mass density (experiments.tex:6); rotativities $\rho_A,\rho_{\nu,A}$ (method.tex:15); $\rho_{\rm unit}$ | Keep $\rho_A$ (MG18, M21). Density: $\rho$ stays but must be defined as "rest-mass density in code units"; alternatively $\rho_0$. |
| $\mu$ | spacetime index; mean molecular weight (experiments.tex:10) | Mean molecular weight: $\bar\mu$ or $\mu_{\rm w}$. |
| $B$ | field strength (experiments.tex:6); Planck function $B_\nu$ (experiments.tex:16) | Planck function: $B_\nu$ is standard; define it in words at first use. |
| $R$ | temperature ratio (experiments.tex:9); rotation $R_n$ (method.tex:58) | Rotation: $\mathbf R_m$ bold; ratio stays $R$. |
| $h$ | RK4 step (method.tex:30); Planck constant (needed if $\varepsilon=h\nu/m_ec^2$ is written) | Step: $\Delta\lambda$ (then $\Delta\lambda_{\rm c}=\Delta\lambda\,r_{\rm g}/\nu_{\rm obs}$, consistent with results.tex:14). |
| $C$ | step constant (method.tex:30); $C^1$ (method.tex:22) | Harmless; optionally $C_{\rm step}$. |
| $E$ | superphoton energy $-k_t$ (method.tex:79, in $m_ec^2$); $E_{\rm c}$ with $k_t=-1$ imaging normalization (method.tex:74) | State the two normalizations; write the cut as $E_n>E_{\rm c}$ with $E_n\equiv-k_\mu n^\mu/(-k_t)$. |
| $\sigma$ | magnetization (experiments.tex:6); $\sigma_{\rm h}$ (method.tex:80) | Harmless with subscript; keep. |
| $M$ | black-hole mass (method.tex:12); Mueller matrix $M_n$ (method.tex:66); $\mathcal M$ density scale | Mueller: bold $\mathbf M_m$. |
| $S$ | $S_\nu$ camera Stokes (method.tex:56); $\boldsymbol{\mathcal S}$ invariant | Keep; write $\boldsymbol S_\nu$ bold for the vector. |

Disambiguation constraint from conventions.md: $(\alpha,\beta)$ screen, $\chi$ EVPA, $\lambda$ affine,
$j_A,\alpha_A,\rho_A$ invariant coefficients, $w$ weight, $\sigma_{\rm h}$, $\Theta_{\rm e}$, $\mathcal M$
are fixed; renaming falls on the lapse/shift, plasma $\beta$, conserved $(\lambda,\eta)$, Z4c $\chi$,
inclination, segment index and the $4\times4$ matrices.

## 8. Undefined or late-defined symbols

Suggested sentences are in our conventions (conventions.md) and can be inserted at the cited line.

| Symbol | File:line of first use | Status | Suggested definition |
|---|---|---|---|
| $M$ | method.tex:12 | USED-BEFORE-DEFINITION (experiments.tex:6) | "…with $M$ the black-hole mass, geometric quantities use $G=c=M=1$…" |
| $x$ in $x\mapsto g_{\mu\nu}(x)$ | introduction.tex:15 | USED-BEFORE-DEFINITION | "a metric closure $x^\mu\mapsto g_{\mu\nu}(x^\mu)$ on spacetime points $x^\mu$". |
| $\eta_{\mu\nu}$ | method.tex:13 | NEVER-DEFINED | "$\eta_{\mu\nu}={\rm diag}(-1,1,1,1)$ is the Minkowski metric". |
| $f$, $l_\mu$ | method.tex:13 | NEVER-DEFINED | "with $f=2r^3/(r^4+a^2z^2)$ and the null covector $l_\mu=\big(1,(rx+ay)/(r^2+a^2),(ry-ax)/(r^2+a^2),z/r\big)$, where $r$ solves $r^4-(\rho^2-a^2)r^2-a^2z^2=0$, $\rho^2=x^2+y^2+z^2$, and $a$ is the dimensionless spin". (Use $R^2$ for $x^2+y^2+z^2$ if $\rho$ is reserved.) |
| $a$ | method.tex:40 | NEVER-DEFINED | define with $f$ at method.tex:13 as "dimensionless spin $a=J/M$ (in $G=c=1$)". |
| $r$ | method.tex:30 | NEVER-DEFINED | define with $f$ (above); note it is the Kerr–Schild (Boyer–Lindquist) radius, not $\sqrt{x^2+y^2+z^2}$. |
| $f_A$, $l^A_\mu$ | method.tex:21 | NEVER-DEFINED | "where $f_h$ and $l^{(h)}_\mu$ are the Kerr–Schild scalar and null covector of hole $h$, boosted to its velocity and centred on its position". |
| $\alpha,\beta^i,\gamma_{ij}$ | method.tex:22 | DEFINED-IMPLICITLY | "the lapse $N$, shift $N^i$ and spatial metric $\gamma_{ij}$, $\dd s^2=-N^2\dd t^2+\gamma_{ij}(\dd x^i+N^i\dd t)(\dd x^j+N^j\dd t)$". |
| $\tilde\gamma_{ij}$, $\chi$ (Z4c) | method.tex:22 | NEVER-DEFINED | "with the conformal metric $\tilde\gamma_{ij}$ and conformal factor $\chi_{\rm Z4c}$ evolved by the Z4c system". |
| $\lambda$ (affine) | method.tex:14 | DEFINED-IMPLICITLY | "The wave vector is $k^\mu=\dd x^\mu/\dd\lambda$ with $\lambda$ an affine parameter along the null geodesic". |
| $\nu_{\rm obs}$ | method.tex:14 | DEFINED-IMPLICITLY | "…so that the observing frequency at the camera is $\nu_{\rm obs}$ and an observer with four-velocity $u^\mu$ measures $\nu=\nu_{\rm obs}(-k_\mu u^\mu)$". |
| $A$ (Stokes index), $j_{\nu,A},\alpha_{\nu,A},\rho_{\nu,A}$ | method.tex:15 | NEVER-DEFINED | "with fluid-frame emissivities $j_{\nu,A}$, absorptivities $\alpha_{\nu,A}$ and rotativities $\rho_{\nu,A}$, $A\in\{I,Q,U,V\}$, and invariant coefficients $j_A=j_{\nu,A}/\nu^2$, …". |
| $\boldsymbol j$ | method.tex:47 | DEFINED-IMPLICITLY | "where $\boldsymbol j=(j_I,j_Q,j_U,j_V)^{\mathsf T}$". |
| $i$ (imaginary unit) before inclination | method.tex:16 | STANDARD, collides | write $\mathrm{i}$; define inclination as $\theta_{\rm o}$ at method.tex:32. |
| $(\lambda,\eta)$ conserved | method.tex:37 | NEVER-DEFINED | "maps $(\alpha,\beta)$ to the energy-scaled angular momentum $\hat\lambda=L_z/E=-\alpha\sin\theta_{\rm o}$ and Carter constant $\hat\eta=(\beta^2+\alpha^2-a^2)\cos^2\theta_{\rm o}$ (at $d\to\infty$)". |
| $\phi_s$, $r_s$, $t_s$ coordinates | method.tex:37 | DEFINED-IMPLICITLY | "the Boyer–Lindquist radius, azimuth and time $(r_s,\phi_s,t_s)$ of the $n$-th equatorial crossing". |
| $I$, $I_{\max}$ | method.tex:40 | DEFINED-IMPLICITLY | caption: "$I$ is the camera Stokes $I$ ($S_\nu$ component), $I_{\max}$ its maximum over the image". |
| $R_n$ index $n$ | method.tex:58 | USED-BEFORE-DEFINITION | introduce the segment index when segments are introduced at method.tex:57: "Each step $m$ (a segment of length $\Delta\lambda_{\rm c,m}$) applies…". |
| projected field / $b^\mu$ | method.tex:59 | USED-BEFORE-DEFINITION | "$e_{(2)}$ along $b^\mu$ projected orthogonal to $u^\mu$ and $k^\mu$, with $b^\mu$ the magnetic four-vector (Sec. models)". |
| $\theta_B$ (concept) | method.tex:59 | USED-BEFORE-DEFINITION (symbol at experiments.tex:17) | move the definition of $\theta_B$ to Sec. transfer. |
| $M_n$, $b_n$ | method.tex:66 | DEFINED-IMPLICITLY | "with $\mathbf M_m=\exp(-\mathbf K_m\Delta\lambda_{{\rm c},m})$ and $\boldsymbol s_m=\int_0^{\Delta\lambda_{{\rm c},m}}\exp(-\mathbf K_m s)\,\dd s\,\boldsymbol j_m$". |
| $\boldsymbol{\mathcal S}_{\rm obs}$ | method.tex:68 | DEFINED-IMPLICITLY | "$\boldsymbol{\mathcal S}_{\rm obs}$, initialised to zero, is the camera-frame invariant Stokes vector". |
| $n^\mu$ | method.tex:74 | NEVER-DEFINED | "the normal (Eulerian) observer $n_\mu=-N\,\delta^t_\mu$, the camera's $e_{(0)}$". |
| $\alpha_{\rm h}$ | method.tex:74 | OK, slicing unstated | "the Kerr–Schild lapse $N=(1+f)^{-1/2}$ evaluated on the horizon" (finite in KS slicing, zero in Boyer–Lindquist slicing; the bound needs the KS value). |
| $\Rhigh$ | method.tex:75 | USED-BEFORE-DEFINITION (experiments.tex:9) | "such as the density scale $\mathcal M$ or the temperature-ratio parameter $R_{\rm high}$ (Eq. rhigh)". |
| $\Theta_{\rm e}$ | method.tex:80 | NEVER-DEFINED | "dimensionless electron temperature $\Theta_{\rm e}=k_{\rm B}T_e/m_ec^2$" (at experiments.tex:10, and forward-reference from method.tex:80). |
| $\varepsilon$ | method.tex:80 | NEVER-DEFINED | "fluid-frame photon energy $\varepsilon=h\nu/m_ec^2=-k_\mu u^\mu$ (with $k_\mu$ in $m_ec$ units)". |
| $k_\mu$ normalization (MC) | method.tex:79 | DEFINED-IMPLICITLY | "In the Monte Carlo module $k_\mu$ is in units of $m_ec$, so $E=-k_t$ is the energy at infinity in $m_ec^2$". |
| $\rho$ (density) | experiments.tex:6 | NEVER-DEFINED | "rest-mass density $\rho$ (code units)". |
| $p_{\rm gas}$ | experiments.tex:6 | DEFINED-IMPLICITLY | "gas pressure $p_{\rm gas}$". |
| $\beta_{\rm crit}$ | experiments.tex:9 (implicit, value 1) | NEVER-DEFINED | write $R=(R_{\rm high}b^2+R_{\rm low})/(1+b^2)$ with $b=\beta_{\rm p}/\beta_{\rm crit}$, $\beta_{\rm crit}=1$. |
| $n_{\rm e},n_i$ | experiments.tex:10 | NEVER-DEFINED | "electron and ion number densities $n_{\rm e}$, $n_{\rm i}$". |
| $\mu$ (mean molecular weight) | experiments.tex:10 | NEVER-DEFINED | "mean molecular weight $\bar\mu$". |
| $\kappa$ (distribution), width $w$ | experiments.tex:16 | NEVER-DEFINED | "$\kappa$-distributed electrons with index $\kappa$ and width $w_\kappa$" (rename P16's $w$ to avoid the weight $w$). |
| $B_\nu(T_e)$ | experiments.tex:16 | DEFINED-IMPLICITLY | "the Planck function $B_\nu(T_e)$". |
| $\Delta\lambda_{\rm c}$ | results.tex:14 | DEFINED-IMPLICITLY | define with $\lambda_{\rm c}$ at method.tex:56 or in the segment sentence. |
| $K_{0,1,2}$ | results.tex:15 | NEVER-DEFINED | "modified Bessel functions of the second kind $K_n$ (in the M21 thermal fits)". |
| $b_{\rm c}$ | results.tex:16 | NEVER-DEFINED | "critical impact parameter $b_{\rm c}$ (the Schwarzschild shadow radius $3\sqrt3$)". |
| $m_e,k_{\rm B},m_p,M_\odot$ | method.tex:79; experiments.tex:10; results.tex:32 | STANDARD | one sentence "$m_e$, $m_p$: electron and proton masses; $k_{\rm B}$: Boltzmann constant". |

## 9. Macros defined in `paper/macros.tex` but unused in the sections

`\aimg`, `\bimg`, `\aB`, `\bB`, `\mnet`, `\nuc`, `\nus`, `\Iinv`, `\Nsp`, `\nuLnu`, `\betacrit`,
`\tetrad`, `\kcov`, `\kcon`, `\ucon`, `\bcon`, `\Kerr`, `\GMc`, `\sgn`.
`\nuc`, `\nus`, `\Nsp`, `\nuLnu`, `\betacrit` are needed once the coefficient fits
(M21 thermal: $\nu_{\rm c}=eB/2\pi m_ec$, $\nu_{\rm s}=\tfrac29\nu_{\rm c}\Theta_{\rm e}^2\sin\theta_B$) and
the Monte Carlo estimator ($N_{\rm s}$, $\nu L_\nu$) are written out in an appendix.
