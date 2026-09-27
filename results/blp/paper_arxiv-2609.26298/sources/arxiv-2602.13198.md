# arxiv-2602.13198 — Naethe Motta, Nemmen & Joshi 2026, GPUmonty (ApJ 1001, 148)
- Mirror: ref-paper/arxiv-2602.13198/ (tex/main.tex; PROVENANCE.md). Bib key: Motta2026GPUmonty.
- Role: reference for the Monte Carlo weight table, generation, absorption, biased scattering and the sphere test (src/montecarlo/).
- Declarations (literature-grounded):
  - Weight $w_\nu=\frac{\Delta t\,\Delta\ln\nu}{hN_s}\int\sqrt{-g}\,d^3x\int j_\nu d\Omega$ (main.tex:76, Eq. 2).
  - Absorption $dw/d\tau_a=-w$, trapezoid $\tau_a$ (main.tex:175–186, Eqs. 18–20).
  - Biased scattering $p=1-e^{-b\tau_s}$, parent $w(1-1/b)$, child $w/b$ (main.tex:193–196); sphere bias $b=n_e\sigma_TR(n_{sc}+1)^2$ (main.tex:321).
- Difference recorded in writing/derivations.md §10.4: BlackLightPlus spawns per step with child weight $w(1-e^{-\Delta\tau_s})/(1-e^{-b\Delta\tau_s})$ and depletes the parent by $e^{-\Delta\tau_s}$.
- [OPEN] none.
