# arxiv-2203.15963 — White 2022, Blacklight (ApJS 262, 28)
- Mirror: ref-paper/arxiv-2203.15963/; code ref-code/c-white-blacklight @ a7c7215 (PROVENANCE.md). Bib key: White2022.
- Role: reference (structure, camera, unpolarized and polarized march that BlackLightPlus ports).
- Declarations (literature-grounded):
  - Covariant-momentum geodesic equations, $dk_t/d\lambda=0$ in CKS (blacklight_method.tex l.262–266); integration toward negative $\lambda$ from the camera (l.272; code geodesics.cpp:467 `h = -ray_step*(r-r_horizon)`).
  - Camera: line of sight $K$, up $U=\hat z$ Gram–Schmidt to $v$, $h=v\times K$, roll $\psi$ (l.184–220); normalization $-n\,k_t=\nu$ for plane-parallel cameras (l.254–258).
  - Invariants $I=I_\nu\nu^{-3}$, $j_I=j_\nu\nu^{-2}$, $\alpha_I=\alpha_\nu\nu$ (l.332–337).
  - Code: tetrad `Tetrad` puts `up` (= $b^\mu$ in the fluid) on $e_{(2)}$, $e_{(1)}$ completes a right-handed triad (radiation_geometry.cpp:590–640); Stokes $Q=\tfrac12(N_{11}-N_{22})$, $V=\tfrac12\,{\rm Im}(N_{21}-N_{12})$ (polarized.cpp:934–938); thermal $j_Q<0$ (simulation_coefficients.cpp:493).
- [OPEN] none for this node beyond conventions.md OPEN-1/OPEN-2.
