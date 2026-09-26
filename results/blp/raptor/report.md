# RAPTOR and BlackLightPlus on the binary-black-hole snapshots

Nodes R07 (s455 image comparison), R08 (CPU timings), R09 (spinning snapshots) and R10 (i = 90° AMR-figure deck) of `dag.md`; design and attribution rule in `design.md` (§3 conventions, §4 modifications, §5 attribution). Every number below is read from `evidence/R07/metrics.json`, `evidence/R07/polstep.json`, `evidence/R07/native_cold.txt` and `evidence/R08/timing.json` by `tools/r08_report.py`; the R09 and R10 sections (spinning snapshots and the i = 90° AMR-figure deck) are read from `evidence/R09` and `evidence/R10` by `tools/r09_report.py`.

## Setup

| item | value (both codes) |
|---|---|
| snapshot | s455, `SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin` (AthenaK `.bin`, 512 blocks of 64³) |
| binary | static superposed Kerr–Schild, m_A = 0.5, a_A = 0, sep 25, centres ±(12.19120680, −2.76124551, 0) from the dump time |
| scaling | M = 1.3e10 M_sun (6.5e9 per hole), D = 16.8 Mpc, 230 GHz; θ_g = 7.6382 µas/M |
| plasma | thermal, R_low 1, R_high 40, σ > 1 cut, γ = 13/9 from the dump, n_e = ρ/m_p, mass scale 2.274019497e28 g (the calibrated value of `figures/out/eht_bbh/s455_i17_Rh40`, used by both codes) |
| camera | plane-parallel at d = 1000 M, φ_o = 0, FOV 56 M, 480²; i = 17°, 60°, 90° |
| march | RK4, h = min_A(r_A − r_h,A)/C, nmax = 50 C, capture d < 1e−3 or E_n > 100, escape R > 1100 |
| transfer | polarized along the whole ray; BlackLightPlus coefficients (M21, `expm1` Kirchhoff, j_P ≤ j_I and α_P ≤ α_I rescales) |
| codes | BlackLightPlus `c02af6c` CPU path (`tools/r07_blp.jl`); RAPTOR fork `blp-bbh` (`tools/r07_build.sh`, `tools/r07_raptor.sh`), legs a/b built at `e06fd46`, legs c/stock at `502d876` (the two differ only in `rho_Q_thermal`, which legs a/b do not call) |

RAPTOR legs:

- **(a) matched**: M1–M9 and M12 with RAPTOR's own polarized step (`check_stiffness` → RK4 or implicit trapezoid, f^μ transport between samples).
- **(b) matched solver**: (a) plus M11, the exact constant-coefficient step of BlackLightPlus `stokes_segment` (`POL_SOLVER_LD85`).
- **(c) native fits** (report only): (a) with the upstream emission fits (`EMIS_BLP 0`).
- **stock** (report only): the upstream choices that can run on this data: upstream fits, upstream constants (G 6.674e−8, M_sun 1.989e33, kpc 3.086e21, m_e, m_p, e, h), the upstream plasma decode of `model/bhac` (n_e = ρ/(m_p + m_e) + 1e−6, b² floor, Θ_e = (u/ρ)(1/3)(m_p/m_e)/(1 + R), Θ_e ∈ [1e−3, 100], r < 1 and r > 50 cuts) and the polarized radius 50 M, on the matched geometry, camera, march and IEEE/IAU readout. Upstream RAPTOR has no binary metric, no AthenaK reader and a single-hole step control whose inner cutoff is the horizon of a hole at the origin, so the geometry legs cannot run unmodified; the upstream Bardeen camera at r = 1e4 differs from the plane-parallel camera by a sub-pixel projection and is not used.

## Modifications to RAPTOR

Branch `blp-bbh` from upstream `b91e937`; the patch series is `patches/` (`fork.md`).

| commit | title |
|---|---|
| `720a111` | test(m7-coeffs): add the R03 thermal coefficient table driver |
| `ed64b3a` | feat(m9-constants): add the blp_analytic model headers with BlackLightPlus constants |
| `066295b` | feat(m7-coeffs): port the BlackLightPlus thermal coefficients as EMIS_BLP |
| `49448b2` | test(m1-metric): add the metric and connection probe for R04 |
| `7833602` | feat(m1-metric): add the superposed Kerr-Schild binary metric SKS |
| `87883ff` | feat(m2a-fd-connection): use an absolute finite-difference step for SKS |
| `7e3b422` | feat(m2b-analytic-connection): add analytic Christoffel symbols for SKS |
| `3e3f887` | test(m3-march): add the ray-dump interface and the R04 test builds |
| `445ddb1` | feat(m3-march): adopt the BlackLightPlus march control, termination and sample set |
| `1a3714a` | test(m4-camera): expose RAPTOR's camera image states to the R05 camera check |
| `4965d55` | feat(m4-camera): add the BlackLightPlus plane-parallel camera (CAMERA_PP) |
| `840784d` | test(m8-stokes): add the R02 analytic scenes and the single-pixel transfer library |
| `82b8ba2` | fix(m8-stokes): reset the analytic scene to its defaults on every read |
| `b9d7443` | feat(m8-stokes): read the camera Stokes out in the IEEE/IAU convention (STOKES_IAU) |
| `82c904f` | fix(m7-coeffs): mirror the BlackLightPlus emission admissibility rescale |
| `aaa32ce` | fix(rte): add the emission of absorption-free segments in the unpolarized transfer |
| `d1bc730` | test(m11-ld85): add the single-segment polarized transfer library for R07 |
| `59adab6` | feat(m11-ld85): add the exact constant-coefficient polarized step (POL_SOLVER_LD85) |
| `bffead8` | perf(m12-openmp): trace every pixel in one OpenMP loop with phase timers |
| `e06fd46` | feat(m6-plasma): add the upstream RAPTOR plasma model as a switch (ATHK_UPSTREAM_PLASMA) |
| `d01fc9a` | test(native-coeffs): check the upstream thermal fits for finite values at cold electrons |
| `502d876` | fix(native-coeffs): take K1/K2 -> 1 in rho_Q_thermal when K2 underflows |
| `e6359a6` | fix(pol-readout): keep the transported Stokes I when the polarized invariant stays below the activation threshold |

The R07/R08 modifications:

- **M11** `pol_rte_ld85_step`: port of `stokes_segment` + `stokes_admissible`; 20000 random segments (δτ 1e−4–30, Faraday depth 1e−3–300 per segment) agree with BlackLightPlus `stokes_couple` to 4.2e-14 (relative).
- **M12** one `parallel for schedule(dynamic, 64)` over every pixel, `RAPTOR_TIMERS` phase timers, per-pixel end class and stored-state count in the HDF5 image; the R01 synthetic image (switches off) stays bitwise identical to upstream `b91e937`.
- **native-coefficient fix** `rho_Q_thermal`: K₁/K₂ → 1 when K₂(1/Θ_e) underflows (the guard `rho_V_thermal` already applies to K₀/K₂). Before: 7353 of 110808 values non-finite; largest Theta_e with a non-finite value 1.2589e-03; after: 0 of 110808 values non-finite; largest Theta_e with a non-finite value 0.0000e+00.
- **upstream plasma switch** `ATHK_UPSTREAM_PLASMA` for the stock leg.

## Native polarized step

`check_stiffness` evaluates the RK4 amplification 1 + τ + τ²/2 + τ³/6 + τ⁴/24 with τ = +λ dl on `dl_current` without the `C_CONST` factor of the step, and selects the implicit trapezoid when any |amplification| > 0.99. For any absorbing segment the amplification exceeds 1, so the trapezoid is taken on 100.0% of the test segments; the step converges at order 2.00 (`polstep.json`). Its phase error per segment grows with the Faraday depth per segment:

| Faraday depth per segment | median relative error | max |
|---|---|---|
| 0.001-0.1 | 2.68e-05 | 1.55e+00 |
| 0.1-1 | 4.50e-03 | 1.49e+00 |
| 1-10 | 1.98e-01 | 1.86e+00 |
| 10-300 | 3.22e-01 | 1.85e+00 |

## Results

Thresholds (`dag.md` R07): |ΔF/F| ≤ 0.01 (≤ 0.005 after Richardson); L1(I) ≤ 0.03; NMSE(I) ≤ 0.001; |Δm_net|, |Δ⟨|m|⟩| ≤ 0.01; |ΔEVPA_net| ≤ 2°; |Δv_net| ≤ 0.002; ring diameters within 0.5 µas (native and 20 µas blur); |Δf_C| ≤ 0.02; i = 90°: the pair unresolved in both codes. m_net, ⟨|m|⟩, v_net and EVPA_net are the 20 µas-blurred EHT definitions of `figures/plot_eht_bbh.py`; L1 and NMSE are on the native 480² images. Differences are RAPTOR − BlackLightPlus.

### Leg (a), native polarized step

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | -1.75e-05 | 2.61e-03 | 1.09e-04 | 9.49e-02 | 3.78e-01 | +0.0018 | -0.0003 | +4.34 | +0.00091 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 17 | 80 | -3.07e-04 | 2.31e-03 | 8.86e-05 | 7.84e-02 | 3.07e-01 | -0.0074 | +0.0050 | +1.08 | +0.00023 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | -4.15e-05 | 1.08e-03 | 1.67e-05 | 4.08e-02 | 1.53e-01 | -0.0005 | +0.0019 | -0.67 | +0.00019 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | 1.69e-05 | 3.43e-04 | 1.51e-06 | 1.66e-02 | 5.18e-02 | +0.0006 | +0.0001 | +0.33 | +0.00005 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 640 | 2.97e-06 | 9.25e-05 | 9.72e-08 | 5.48e-03 | 1.47e-02 | +0.0007 | +0.0001 | +0.03 | +0.00002 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | -4.85e-05 | 6.65e-03 | 4.69e-04 | 8.10e-02 | 4.34e-01 | -0.0001 | +0.0029 | -33.25 | -0.00068 | 0.00 | 0.001 | 1.000000 | miss evpa_net_deg (discretisation) |
| 60 | 80 | 3.78e-04 | 4.75e-03 | 2.80e-04 | 5.87e-02 | 2.94e-01 | +0.0084 | -0.0032 | +75.59 | -0.00051 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 60 | 160 | 1.07e-04 | 2.63e-03 | 1.14e-04 | 3.17e-02 | 2.01e-01 | -0.0000 | -0.0051 | +59.64 | -0.00001 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 60 | 320 | -6.02e-05 | 1.03e-03 | 1.96e-05 | 1.35e-02 | 1.00e-01 | +0.0072 | +0.0006 | -7.19 | -0.00045 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 90 | 40 | 2.19e-04 | 5.43e-03 | 5.56e-04 | 5.92e-02 | 2.53e-01 | +0.0079 | -0.0009 | -22.10 | -0.00012 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 90 | 80 | 5.11e-04 | 3.58e-03 | 3.67e-04 | 4.10e-02 | 1.46e-01 | -0.0002 | -0.0066 | -52.36 | -0.00009 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |
| 90 | 160 | 2.92e-05 | 1.62e-03 | 1.20e-04 | 2.05e-02 | 6.94e-02 | +0.0028 | +0.0075 | -0.74 | -0.00002 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | 8.27e-06 | 5.57e-04 | 1.51e-05 | 8.65e-03 | 2.55e-02 | +0.0038 | +0.0026 | -75.80 | +0.00012 | 0.00 | 0.000 | 1.000000 | miss evpa_net_deg (discretisation) |

Flux after Richardson extrapolation of each code's C ladder (three finest levels):

| i | RAPTOR limit [Jy] | BlackLightPlus limit [Jy] | relative difference | gate |
|---|---|---|---|---|
| 17 | 0.981443 (order 2.16) | 0.981460 (order 1.65) | -1.72e-05 | pass |
| 60 | 1.463943 (non-monotone: finest level) | 1.464031 (non-monotone: finest level) | -6.02e-05 | pass |
| 90 | 2.165833 (non-monotone: finest level) | 2.165815 (non-monotone: finest level) | 8.27e-06 | pass |

Evaluated levels (C = 160 and the finest level):

- i = 17°, C = 160: all thresholds met
- i = 17°, C = 640: all thresholds met
- i = 60°, C = 160: misses evpa_net_deg; every miss attributed to discretisation
- i = 60°, C = 320: misses evpa_net_deg; every miss attributed to discretisation
- i = 90°, C = 160: all thresholds met
- i = 90°, C = 320: misses evpa_net_deg; every miss attributed to discretisation

### Leg (b), LD85 step (M11)

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | -3.15e-08 | 8.74e-08 | 1.41e-13 | 5.17e-06 | 8.69e-06 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 80 | -2.04e-09 | 5.76e-09 | 6.83e-16 | 8.31e-07 | 1.13e-06 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | -1.57e-10 | 4.45e-10 | 5.09e-18 | 9.78e-08 | 1.29e-07 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | 3.57e-12 | 1.35e-10 | 1.04e-18 | 1.37e-08 | 1.94e-08 | -0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 640 | 2.92e-12 | 8.61e-11 | 2.46e-19 | 3.76e-09 | 5.44e-09 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | 3.40e-09 | 1.03e-07 | 8.62e-13 | 2.87e-06 | 4.84e-06 | +0.0000 | +0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 80 | 7.23e-10 | 7.70e-09 | 5.19e-15 | 3.99e-07 | 5.42e-07 | -0.0000 | +0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 160 | 8.65e-11 | 6.90e-10 | 4.43e-17 | 4.87e-08 | 6.87e-08 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 320 | 3.73e-11 | 2.17e-10 | 5.79e-18 | 1.49e-08 | 1.15e-08 | -0.0000 | +0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 40 | -4.91e-09 | 1.91e-07 | 2.22e-12 | 4.35e-06 | 7.95e-06 | +0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 80 | 3.52e-10 | 1.26e-08 | 8.19e-15 | 5.24e-07 | 8.17e-07 | +0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 160 | 9.83e-10 | 2.02e-09 | 2.71e-15 | 8.10e-08 | 1.19e-07 | +0.0000 | +0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | -1.72e-11 | 2.06e-10 | 1.39e-18 | 8.14e-09 | 1.41e-08 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |

Flux after Richardson extrapolation of each code's C ladder (three finest levels):

| i | RAPTOR limit [Jy] | BlackLightPlus limit [Jy] | relative difference | gate |
|---|---|---|---|---|
| 17 | 0.981460 (order 1.65) | 0.981460 (order 1.65) | -3.26e-11 | pass |
| 60 | 1.464031 (non-monotone: finest level) | 1.464031 (non-monotone: finest level) | 3.73e-11 | pass |
| 90 | 2.165815 (non-monotone: finest level) | 2.165815 (non-monotone: finest level) | -1.72e-11 | pass |

Evaluated levels (C = 160 and the finest level):

- i = 17°, C = 160: all thresholds met
- i = 17°, C = 640: all thresholds met
- i = 60°, C = 160: all thresholds met
- i = 60°, C = 320: all thresholds met
- i = 90°, C = 160: all thresholds met
- i = 90°, C = 320: all thresholds met

### Attribution (design.md §5)

**Leg (a) − BlackLightPlus is RAPTOR's polarized step.** The two codes march the same states (stored samples per image identical up to the terminal state of each ray: e.g. i = 17°, C = 80: 192400511 RAPTOR vs 192630911 BlackLightPlus, a difference of one per pixel), share every end class, and with the LD85 step (leg b) the images agree to L1(I) ≤ 1.9e-07, L1(|P|) ≤ 5.2e-06 and L1(V) ≤ 8.7e-06 at every level, falling with C: the O(h⁴) truncation difference between the Christoffel form (x^μ, k^μ) and the Hamiltonian form (x^μ, p_μ) of the same RK4 march. The leg (a) differences in |P| and V fall with C at the rate of RAPTOR's own self-convergence (i = 17°: L1(|P|) 9.5e-02, 7.8e-02, 4.1e-02, 1.7e-02, 5.5e-03 and L1(V) 3.8e-01, 3.1e-01, 1.5e-01, 5.2e-02, 1.5e-02 for C = 40, 80, 160, 320, 640), so they are the discretisation error of the trapezoid step, not a model difference.

**Net EVPA at i = 60° and 90°.** The flux-weighted per-pixel EVPA change of each code between C = 160 and its finest level, and the Faraday depth through the bright pixels (RAPTOR `tauF`, 10/50/90th percentiles), are:

| i | Faraday depth p10/p50/p90 [rad] | m_net (BLP, C = 160) | BLP per-pixel EVPA change C=160→finest [°] | RAPTOR (a) same [°] | (a) − BLP per-pixel [°] at C = 160 |
|---|---|---|---|---|---|
| 17 | 13 / 35 / 92 | 0.0446 | 1.7 | 11.6 | 11.5 |
| 60 | 191 / 376 / 737 | 0.0077 | 18.7 | 51.6 | 47.3 |
| 90 | 9266 / 10219 / 12772 | 0.0043 | 45.4 | 45.0 | 44.5 |

At i = 60° and 90° the rays cross the cold, dense orbital plane; Faraday depths of hundreds to ten thousand radians scramble the EVPA below the pixel scale, the net polarization m_net is a residual of order 0.01 of cancelling contributions, and the net EVPA of either code moves by tens of degrees between C levels. The EVPA_net misses of leg (a) are within 2 max(ε_R, ε_B) and are classed as discretisation; the 2° threshold is not resolvable for these two inclinations by either code at C ≤ 320. At i = 17° (median Faraday depth 35 rad) both codes converge and leg (a) meets the EVPA threshold from C = 80.

**Resolution.** BlackLightPlus alone, i = 17°, C = 80 (`metrics.json` `resolution`):

| npix | F [Jy] | m_net | leg (a): ΔF/F | L1(I) | L1(\|P\|) | Δm_net | leg (b): L1(I) | L1(\|P\|) |
|---|---|---|---|---|---|---|---|---|
| 240 | 0.98654 | 0.0618 | -7.46e-04 | 2.70e-03 | 8.34e-02 | -0.0095 | 8.23e-09 | 7.04e-07 |
| 480 | 0.98136 | 0.0453 | -3.07e-04 | 2.31e-03 | 7.84e-02 | -0.0074 | 5.76e-09 | 8.31e-07 |
| 960 | 0.98258 | 0.0532 | -2.74e-04 | 2.31e-03 | 7.80e-02 | -0.0078 | 7.94e-09 | 8.40e-07 |

The net linear polarization of s455 at i = 17° changes by more than 0.01 between 240², 480² and 960² in both codes: the Faraday-rotated EVPA pattern is not resolved at the 480² pixel. The code-to-code comparison holds at every resolution.

### Report-only legs (C = 80)

| leg | i | ΔF/F | L1(I) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net |
|---|---|---|---|---|---|---|---|
| (c) upstream fits | 17 | -0.0119 | 0.0124 | -0.0102 | -0.0012 | +1.1 | -0.00036 |
| (c) upstream fits | 60 | -0.0072 | 0.0121 | +0.0093 | +0.0005 | +1.7 | -0.00125 |
| (c) upstream fits | 90 | -0.0053 | 0.0119 | -0.0014 | -0.0017 | +29.9 | +0.00005 |
| (c) upstream fits, before the ρ_Q fix | 17 | -0.2010 | 0.2015 | -0.0061 | +0.0034 | +6.2 | +0.00013 |
| (c) upstream fits, before the ρ_Q fix | 60 | -0.3015 | 0.3046 | +0.0149 | +0.0056 | -11.4 | -0.00562 |
| (c) upstream fits, before the ρ_Q fix | 90 | -1.0000 | 1.0000 | — | — | +73.9 | — |
| stock | 17 | -0.4742 | 0.4742 | -0.0210 | -0.0218 | -11.9 | -0.00257 |
| stock | 60 | -0.4360 | 0.4360 | +0.0136 | +0.0131 | +4.7 | -0.00245 |
| stock | 90 | -0.4251 | 0.4251 | -0.0020 | +0.0006 | +13.8 | -0.00199 |
| stock, before the ρ_Q fix | 17 | -0.4742 | 0.4742 | -0.0210 | -0.0218 | -11.9 | -0.00257 |
| stock, before the ρ_Q fix | 60 | -0.4360 | 0.4360 | +0.0136 | +0.0131 | +4.7 | -0.00245 |
| stock, before the ρ_Q fix | 90 | -0.4251 | 0.4251 | -0.0020 | +0.0006 | +13.8 | -0.00199 |

- The upstream fits change the flux by -1.2% to -0.5% and m_net by up to 0.010 (compare the R03 ratio table, `evidence/R03/summary.json`).
- Before the ρ_Q fix, the upstream fits blank every pixel whose ray meets a sample with Θ_e below the K₂ underflow (1/Θ_e ≳ 705; the largest non-finite value on the test grid is at Θ_e = 1.2589e-03): a NaN ρ_Q makes I_pol NaN, `POLARIZATION_ACTIVE` is cleared and the pixel reads out zero. Leg (c) has no Θ_e bound and loses 20% (i = 17°), 30% (i = 60°), 100% (i = 90°) of the flux. The stock images are the same before and after the fix: with its Θ_e ≥ 1e−3 cut no s455 sample reaches the NaN range.
- The stock leg is 43%–47% fainter. The upstream decode takes Θ_e = (u/ρ)(1/3)(m_p/m_e)/(1 + R), and u/(3ρ) = p/(3(γ − 1)ρ) = 0.75 p/ρ for γ = 13/9, i.e. 0.75 × the pressure-based Θ_e; it adds the Θ_e ∈ [1e−3, 100] and r ≤ 50 cuts, n_e = ρ/(m_p + m_e) + 1e−6 cm⁻³, the 1e−6 b² floor and the upstream constants, at the same mass scale; the flux change is not decomposed into these terms.

## CPU timings (R08)

Image: s455, i = 17 deg, 480^2, polarized, C = 80, nmax 4000, 230 GHz. Host: 2x Xeon Platinum 8360Y (72 cores, 144 hardware threads), shared. Protocol: best of 3 warm runs per leg (page cache warm; BlackLightPlus compiled in a 24^2 warm-up render, excluded); RAPTOR OMP_PROC_BIND=close OMP_PLACES=cores; BlackLightPlus julia -t T; phases: load (snapshot read), setup (camera), trace (geodesics), transfer (polarized RT), output.

| run | load | setup | trace | transfer | output | end-to-end [s] | rays/s (trace + transfer) | samples/s | peak RSS [GB] |
|---|---|---|---|---|---|---|---|---|---|
| BlackLightPlus (CPU), 16 thr | 3.39 | 0.025 | 9.62 | 33.85 | 0.02 | 46.90 | 5300 | 4.43e+06 | 24.4 |
| RAPTOR (a, native step), 16 thr | 0.27 | 0.007 | 66.90 | 109.03 | 0.97 | 177.18 | 1310 | 1.09e+06 | 4.0 |
| RAPTOR (b, LD85 step), 16 thr | 0.27 | 0.007 | 66.97 | 99.72 | 0.98 | 167.95 | 1382 | 1.15e+06 | 4.0 |
| BlackLightPlus (CPU), 64 thr | 3.42 | 0.022 | 4.47 | 10.13 | 0.02 | 18.05 | 15787 | 1.32e+07 | 24.2 |
| RAPTOR (a, native step), 64 thr | 0.27 | 0.008 | 19.00 | 31.03 | 0.99 | 51.31 | 4605 | 3.85e+06 | 4.0 |
| RAPTOR (b, LD85 step), 64 thr | 0.28 | 0.008 | 18.97 | 28.34 | 0.98 | 48.59 | 4869 | 4.07e+06 | 4.0 |

Strong scaling of the ray phases (trace + transfer) from 16 to 64 threads: BlackLightPlus 2.98× (74%), RAPTOR (a) 3.52× (88%), RAPTOR (b) 3.52× (88%). At 64 threads BlackLightPlus is 3.43× RAPTOR (a) in rays/s and 2.84× faster end to end.

BlackLightPlus compilation (24² warm-up render, excluded above): 16 thr 5.8 s, 64 thr 7.0 s.

Phase definitions. RAPTOR (`RAPTOR_TIMERS`, M12): load = `read_model` + `init_model` (snapshot read); setup = constants and camera blocks; trace and transfer = the wall of the pixel loop split in proportion to the thread-summed geodesic (camera ray, march) and transfer times; output = spectrum, HDF5 and ASCII image writes. BlackLightPlus (`r07_blp.jl`): load = header, binary setup and `athenak_bin_snapshot`; setup = `camera_rays`; trace = `trace_rays_sampled`; transfer = `polarized_march` over the rays; output = five `.npy` writes. RAPTOR's peak RSS is the process maximum; BlackLightPlus reports `Sys.maxrss()` of the Julia process, which holds the image states of every ray (`trace_rays_sampled` stores the path before the transfer) and the compiled code.

Load average at the start and end of each leg (1/5/15 min; the 1-min value during a 64-thread leg includes the leg's own threads): a 64 before 54.36 42.18 43.44; a 64 after 63.20 50.87 46.55; b 64 before 63.20 50.87 46.55; b 64 after 58.83 54.96 48.81; blp 64 before 58.83 54.96 48.81; blp 64 after 48.32 52.64 48.61; a 16 before 48.32 52.64 48.61; a 16 after 17.65 24.44 35.64; b 16 before 17.65 24.44 35.64; b 16 after 21.75 21.27 29.03; blp 16 before 21.75 21.27 29.03; blp 16 after 14.17 18.38 26.75.

**GPU legs (deferred, `dag.md` R11).** No usable device on this host (NVML error). The legs on a host with A100s:

- BlackLightPlus: the s455 image is `march_rays_gpu_spacetime(cfg, st, snap, camera_rays(st, cam); polarized = true, C = 80, nmax = 4000, devices = [0])` and `devices = [0, 1]` (`src/gpu_spacetime.jl`: the CPU march in-kernel on the superposed Kerr–Schild metric, Float64, one snapshot upload per device), with `cfg`, `st`, `snap`, `cam` of `tools/r07_blp.jl`; the timed phases are the snapshot upload, the fused march and the device-to-host copy, best of 3 warm runs, and the image is checked against the CPU image of `r07_blp.jl` (L1 ≤ 1e−10). The Kerr-deck scaling legs of `bench/bench_multigpu.jl` (`julia --project=. -t 8 bench/bench_multigpu.jl`) give the pool's 1→2 device speed-up.
- RAPTOR: M13 (direct calls instead of the `rk4_step` function pointer, iterative `determ`, preallocated per-ray buffers, device Bessels) and `#pragma acc parallel loop gang vector` over the M12 pixel loop; built with `nvc -acc=gpu -gpu=cc80` in an `r07_build.sh` run directory and run with `r07_raptor.sh` on the same inputs (one device; RAPTOR has no multi-device path). The port is not built; `nvc -mp` builds of the CPU sources exist (`fork.md`).

## Spinning snapshots (R09)

s122 (`torus.mhd_w_bcc.00122.athdf`, sep 16) and s093 (`torus.mhd_w_bcc.00093.bin`, 38.7 GB, sep 25) carry the header spin a = 0.8 per hole in units of the total mass, i.e. a > m_A = 0.5; both codes image them, as `figures/fig_eht_bbh.jl` does, as χ = 0.8 (a_used = 0.4 per hole, along +z: th_a = 0 in the header). Everything else is the R07 protocol: M = 1.3e10 M_sun, D = 16.8 Mpc, 230 GHz, R_high 40, FOV 56 M, 480², C ladder with nmax = 50 C, the calibrated mass scale of `figures/out/eht_bbh/<snap>_i17_Rh40/records.json` passed to both codes, legs (a) and (b).

| snapshot | a_used | centres | mass scale [g] | BLP load [s] | RAPTOR load [s] | peak RSS BLP / RAPTOR [GB] |
|---|---|---|---|---|---|---|
| s122 | 0.4, 0.4 | ±(0.32159304, 7.99353351, 0) | 1.000498515e+21 | 5.2 | 0.4 | 63.8 / 4.1 |
| s093 | 0.4, 0.4 | ±(-10.42584453, 6.89577884, 0) | 2.252873787e+20 | 88.3 | 2.4 | 99.8 / 32.1 |

BlackLightPlus peak RSS is the Julia process maximum over the whole run and includes the stored ray states of the largest image (C = 320); RAPTOR's is the largest per-image process maximum. Load times are single measurements with the page cache in whatever state the preceding legs left it (RAPTOR ran after BlackLightPlus; the s093 figure record has a BlackLightPlus load of 25.9 s). The dag.md bound on the s093 reader, peak memory ≤ 64 GB, holds for RAPTOR (32.1 GB).

### Geometry of the spinning binary

Metric level (1e4 points per leg, `tools/r04_metric.jl`; s122r is s122 with both spins along −z):

| leg | max \|g − g_BLP\| | max \|g g⁻¹ − I\| | Γ analytic (M2b) vs AD | Γ FD (M2a) vs AD | gate |
|---|---|---|---|---|---|
| s093 | 4.44e-16 | 1.22e-15 | 4.68e-15 | 7.05e-11 | pass |
| s122 | 2.22e-16 | 1.30e-15 | 4.13e-15 | 5.99e-11 | pass |
| s122r | 4.44e-16 | 8.24e-16 | 4.04e-15 | 5.45e-11 | pass |

Geodesic level (1050 rays per leg, i = 60°, `tools/r04_rays.jl`, finite-difference build):

| leg | class agreement | max \|Δx\| escaped, C = 80 (/ 2 max ε) | C = 160 | escape angle C = 80 / 160 [rad] | order BLP / RAPTOR | gate (R04 rule) |
|---|---|---|---|---|---|---|
| s093 | 1.0000 | 2.32e-03 (0.79) | 4.40e-05 | 2.11e-06 / 3.98e-08 | 4.02 / 3.57 | pass |
| s122r | 1.0000 | 2.24e-03 (0.72) | 4.80e-05 | 2.01e-06 / 4.36e-08 | 4.02 / 3.63 | pass |

Spin orientation and frame dragging (`tools/r09_geometry.jl`, i = 90°, C = 160, bisected shadow edges on the β = 0 line, test drivers built with max_steps 4e5 so that no bisection ray ends on the step budget; budget-limited rays met: 0):

| hole | photon-orbit edges α = −ξ [M] | BLP edges, d = 1000 | RAPTOR − BLP | 1/d-extrapolated (d = 1000, 4000) − photon orbit | shadow shift | gate |
|---|---|---|---|---|---|---|
| Kerr a=-0.9 | [-6.832319, 2.844421] | [-6.828665, 2.847999] | 0.00e+00 | 1.37e-05 | -1.9903 | pass |
| Kerr a=0.9 | [-2.844421, 6.832319] | [-2.847999, 6.828665] | 0.00e+00 | 1.37e-05 | +1.9903 | pass |

With the observer on +x and α = +ŷ, a hole spinning along +z has its prograde edge at α = −ξ_pro < 0 and its shadow displaced toward +α (west); both codes give the same displacement with the same sign, and reverse it for a < 0. At d = 1000 the plane-parallel camera offsets the edges by O(M b/d) (3.65e-03 M); the 1/d extrapolation removes it.

| binary | hole 1 shadow centre α [M] BLP / RAPTOR | hole 2 | max \|edge_R − edge_B\| [M] | gate |
|---|---|---|---|---|
| s093 | +10.4076 / +10.4076 | -6.0592 / -6.0592 | 0.00e+00 | pass |
| s122 | +9.1611 / +9.1611 | -7.6124 / -7.6124 | 0.00e+00 | pass |
| s122r | +7.5342 / +7.5342 | -9.2388 / -9.2388 | 0.00e+00 | pass |

Reversing both spins (s122 → s122r) moves each shadow centre by 1.6269 M, 1.6264 M (holes 1, 2) in both codes.

**Horizon and capture rule.** Both codes stop a ray at d = min_A(r_A − r_h,A) < 1e−3 with r_h,A = m_A + √(m_A² − a_A²) = 0.8 of the isolated hole, or when the normal-frame energy exceeds 100 (the BlackLightPlus redshift cut and horizon backstop, M3). In the superposed metric the surface g^{μν} ∂_μ r_A ∂_ν r_A = 0 around each hole lies outside r_h (2000 directions per hole; RAPTOR's g^{μν} vanishes on it to the same rounding):

| binary | hole | r_A of the null surface min / median / max | RAPTOR \|g^{rr}\| there (relative) | g^tt ≥ 0 region: max r_A BLP / RAPTOR | gate |
|---|---|---|---|---|---|
| s093 | 1 | 0.8207 / 0.8571 / 0.9193 | 2.07e-15 | 0.1469 / 0.1469 | pass |
| s093 | 2 | 0.8207 / 0.8586 / 0.9191 | 2.04e-15 | 0.1511 / 0.1511 | pass |
| s122 | 1 | 0.8325 / 0.8890 / 0.9762 | 1.56e-15 | 0.2528 / 0.2528 | pass |
| s122 | 2 | 0.8325 / 0.8904 / 0.9761 | 1.67e-15 | 0.2442 / 0.2442 | pass |

Rays that enter the perturbed horizon (between the null surface and r_h + 1e−3) reach the energy cut first: over every stored state of every BlackLightPlus image the smallest r_A − r_h,A is 0.063 (s122), 0.043 (s093), so no sample lies within r_A < 0.843. The inverse metric has g^tt ≥ 0 only deep inside each hole (the table), where `primitive_four_vectors` has no lapse and throws; the number of stored states there is 0 (s122), 0 (s093). The rays never sample that region, so BlackLightPlus needs no change.

### s122 (sep 16), leg (b), LD85 step

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | -3.79e-06 | 4.30e-06 | 7.57e-08 | 1.84e-04 | 3.38e-04 | +0.0000 | -0.0000 | -0.02 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 80 | 1.46e-07 | 1.65e-07 | 8.75e-11 | 1.76e-05 | 2.96e-05 | +0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | 1.13e-08 | 1.27e-08 | 4.96e-13 | 1.60e-06 | 2.94e-06 | +0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | 2.11e-10 | 4.63e-10 | 5.76e-17 | 1.51e-07 | 3.06e-07 | +0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | -2.70e-06 | 6.14e-06 | 4.41e-08 | 1.65e-04 | 1.18e-03 | -0.0000 | -0.0000 | +0.00 | -0.00001 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 80 | -4.03e-07 | 5.32e-07 | 4.55e-10 | 1.56e-05 | 1.29e-04 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 160 | -2.73e-08 | 3.79e-08 | 2.09e-12 | 1.49e-06 | 1.12e-05 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 320 | -3.00e-09 | 4.11e-09 | 2.39e-14 | 1.57e-07 | 1.23e-06 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 40 | 1.28e-06 | 3.55e-06 | 9.81e-10 | 4.42e-04 | 1.04e-03 | -0.0001 | -0.0001 | -0.01 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 80 | 1.24e-07 | 3.80e-07 | 1.26e-11 | 4.78e-05 | 1.06e-04 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 160 | 1.29e-08 | 2.96e-08 | 7.18e-14 | 4.18e-06 | 8.00e-06 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | 1.01e-09 | 3.54e-09 | 1.01e-15 | 4.32e-07 | 7.86e-07 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |

Flux after Richardson extrapolation (three finest levels): i = 17°: +1.12e-07 (pass); i = 60°: -5.02e-06 (pass); i = 90°: +8.07e-07 (pass).

Evaluated levels: i = 17°, C = 160: all thresholds met; i = 17°, C = 320: all thresholds met; i = 60°, C = 160: all thresholds met; i = 60°, C = 320: all thresholds met; i = 90°, C = 160: all thresholds met; i = 90°, C = 320: all thresholds met.

### s122 (sep 16), leg (a), native polarized step

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | 1.81e-06 | 1.10e-05 | 7.60e-08 | 3.09e-04 | 7.36e-04 | -0.0000 | +0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 80 | 1.53e-06 | 1.90e-06 | 1.48e-10 | 5.00e-05 | 1.30e-04 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | 3.58e-07 | 4.49e-07 | 4.06e-12 | 9.81e-06 | 2.81e-05 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | 8.69e-08 | 1.10e-07 | 2.16e-13 | 2.22e-06 | 6.60e-06 | -0.0000 | +0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | 1.08e-05 | 2.89e-05 | 5.85e-08 | 3.14e-04 | 2.39e-03 | -0.0000 | +0.0004 | -0.20 | -0.00001 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 80 | 3.03e-06 | 6.25e-06 | 1.39e-09 | 5.23e-05 | 4.35e-04 | -0.0000 | +0.0001 | -0.05 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 160 | 8.22e-07 | 1.47e-06 | 6.09e-11 | 1.07e-05 | 8.80e-05 | -0.0000 | +0.0000 | -0.01 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 320 | 2.10e-07 | 3.63e-07 | 3.78e-12 | 2.50e-06 | 2.05e-05 | -0.0000 | +0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 40 | 1.04e-05 | 3.37e-05 | 2.75e-08 | 6.49e-04 | 2.34e-03 | +0.0002 | +0.0000 | -0.12 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 80 | 2.36e-06 | 7.98e-06 | 1.61e-09 | 1.00e-04 | 4.35e-04 | +0.0001 | +0.0000 | -0.03 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 160 | 5.76e-07 | 1.95e-06 | 1.03e-10 | 1.74e-05 | 9.07e-05 | +0.0000 | +0.0000 | -0.01 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | 1.43e-07 | 4.85e-07 | 6.41e-12 | 3.75e-06 | 2.14e-05 | +0.0000 | +0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |

Flux after Richardson extrapolation (three finest levels): i = 17°: +5.11e-07 (pass); i = 60°: +1.95e-05 (pass); i = 90°: +9.73e-06 (pass).

Evaluated levels: i = 17°, C = 160: all thresholds met; i = 17°, C = 320: all thresholds met; i = 60°, C = 160: all thresholds met; i = 60°, C = 320: all thresholds met; i = 90°, C = 160: all thresholds met; i = 90°, C = 320: all thresholds met.

s122 (sep 16), resolution at i = 17°, C = 80 (leg b − BlackLightPlus): 240²: L1(I) 7.34e-08, L1(\|P\|) 1.49e-05, m_net BLP 0.0790; 480²: L1(I) 1.65e-07, L1(\|P\|) 1.76e-05, m_net BLP 0.0775; 960²: L1(I) 6.36e-08, L1(\|P\|) 2.12e-05, m_net BLP 0.0780.

### s093 (sep 25), leg (b), LD85 step

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | 3.02e-09 | 4.23e-08 | 2.46e-12 | 1.93e-05 | 1.09e-04 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 80 | 4.47e-10 | 2.63e-09 | 2.62e-14 | 2.05e-06 | 1.38e-05 | -0.0000 | -0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | 3.74e-11 | 2.13e-10 | 1.94e-16 | 2.17e-07 | 1.15e-06 | -0.0000 | -0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | 8.18e-12 | 5.40e-11 | 4.89e-19 | 2.60e-08 | 1.12e-07 | -0.0000 | -0.0000 | +0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | -9.01e-09 | 5.70e-08 | 8.91e-13 | 4.36e-05 | 2.54e-04 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 80 | -5.37e-10 | 2.97e-09 | 3.32e-15 | 4.07e-06 | 2.02e-05 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 160 | -3.73e-11 | 2.28e-10 | 2.00e-17 | 4.20e-07 | 1.80e-06 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 320 | 1.27e-11 | 7.41e-11 | 1.28e-18 | 5.24e-08 | 1.95e-07 | -0.0000 | +0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 40 | 1.34e-07 | 2.21e-07 | 5.16e-10 | 4.71e-05 | 3.02e-04 | -0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 80 | 5.89e-09 | 9.73e-09 | 1.63e-12 | 4.94e-06 | 2.58e-05 | -0.0000 | -0.0000 | +0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 160 | 3.23e-10 | 6.01e-10 | 4.87e-15 | 5.56e-07 | 3.12e-06 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | 2.06e-10 | 3.34e-10 | 2.53e-15 | 6.67e-08 | 3.36e-07 | -0.0000 | -0.0000 | -0.00 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |

Flux after Richardson extrapolation (three finest levels): i = 17°: +4.60e-10 (pass); i = 60°: -3.76e-09 (pass); i = 90°: +9.00e-09 (pass).

Evaluated levels: i = 17°, C = 160: all thresholds met; i = 17°, C = 320: all thresholds met; i = 60°, C = 160: all thresholds met; i = 60°, C = 320: all thresholds met; i = 90°, C = 160: all thresholds met; i = 90°, C = 320: all thresholds met.

### s093 (sep 25), leg (a), native polarized step

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 17 | 40 | 2.56e-08 | 6.22e-07 | 2.38e-10 | 5.13e-03 | 6.20e-04 | -0.0003 | +0.0001 | -0.03 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 80 | 6.20e-08 | 3.61e-07 | 1.14e-10 | 3.63e-03 | 3.70e-04 | +0.0000 | -0.0001 | -0.10 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 160 | 1.69e-08 | 1.94e-07 | 5.45e-11 | 1.84e-03 | 1.95e-04 | +0.0001 | +0.0000 | +0.02 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 17 | 320 | -9.23e-09 | 6.96e-08 | 1.57e-11 | 5.66e-04 | 7.50e-05 | +0.0000 | +0.0000 | +0.03 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 40 | -1.16e-07 | 1.67e-06 | 1.43e-09 | 2.68e-03 | 2.10e-03 | +0.0005 | +0.0011 | +0.90 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 80 | -1.52e-08 | 1.13e-06 | 6.41e-10 | 1.99e-03 | 1.45e-03 | -0.0003 | +0.0002 | +0.11 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 160 | 6.65e-08 | 7.60e-07 | 3.22e-10 | 1.43e-03 | 9.61e-04 | -0.0001 | -0.0001 | -0.14 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 60 | 320 | -5.79e-08 | 3.58e-07 | 9.77e-11 | 6.46e-04 | 6.25e-04 | +0.0001 | +0.0001 | -0.01 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 40 | 2.34e-07 | 1.17e-06 | 1.08e-09 | 2.05e-03 | 1.91e-03 | +0.0013 | +0.0003 | -1.12 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 80 | -1.72e-07 | 6.82e-07 | 5.06e-10 | 1.40e-03 | 1.04e-03 | -0.0005 | -0.0006 | -0.39 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 160 | 1.01e-08 | 3.89e-07 | 1.60e-10 | 8.53e-04 | 4.68e-04 | -0.0001 | +0.0002 | +0.31 | +0.00000 | 0.00 | 0.000 | 1.000000 | pass |
| 90 | 320 | -2.84e-08 | 2.80e-07 | 3.92e-10 | 4.30e-04 | 2.00e-04 | -0.0000 | -0.0000 | -0.00 | -0.00000 | 0.00 | 0.000 | 1.000000 | pass |

Flux after Richardson extrapolation (three finest levels): i = 17°: +6.35e-08 (pass); i = 60°: -2.67e-06 (pass); i = 90°: -4.96e-07 (pass).

Evaluated levels: i = 17°, C = 160: all thresholds met; i = 17°, C = 320: all thresholds met; i = 60°, C = 160: all thresholds met; i = 60°, C = 320: all thresholds met; i = 90°, C = 160: all thresholds met; i = 90°, C = 320: all thresholds met.

s093 (sep 25), resolution at i = 17°, C = 80 (leg b − BlackLightPlus): 240²: L1(I) 2.09e-09, L1(\|P\|) 2.29e-06, m_net BLP 0.0442; 480²: L1(I) 2.63e-09, L1(\|P\|) 2.05e-06, m_net BLP 0.0455; 960²: L1(I) 3.34e-09, L1(\|P\|) 2.16e-06, m_net BLP 0.0452.

### R09 attribution

With the LD85 step the two codes agree on both spinning snapshots to L1 ≤ 1.12e-05 in I, |P| and V at C ≥ 160 (≤ 1.29e-04 at C = 80, ≤ 1.18e-03 at C = 40) at every inclination, with identical end classes: the spinning superposed metric, the frame dragging of each hole, the capture rule, the `.athdf` and 4096-block `.bin` readers and the plasma model are the same in both codes. Leg (b)'s residual in |P| and V falls by about an order of magnitude per doubling of C (the O(h⁴) truncation difference of the Christoffel and Hamiltonian forms of the same RK4 march, concentrated on the photon rings). Leg (a) adds RAPTOR's second-order polarized step; unlike s455 its difference stays small because the Faraday depth through the bright pixels (RAPTOR `tauF`, 10/50/90th percentiles at C = 160) is low: s122 i = 17°: 1.6/4.3/8.7, i = 60°: 5.5/9.4/17, i = 90°: 6/9.5/18; s093 i = 17°: 0.16/0.36/7.5, i = 60°: 0.18/0.34/0.66, i = 90°: 0.16/0.3/0.8 rad, against 13–12772 rad for s455.

**s093 artefacts** (present identically in both codes' input, not corrected): a cold dense ball at r < 8 about the origin, the restart transient of the dump, and 28% (i = 17°), 41% (i = 60°), 46% (i = 90°) of the flux outside the 56 M field of view (`figures/out/eht_bbh` records, 224 M wide-field check).

## i = 90° AMR-figure deck (R10)

Deck of `figures/fig_bbh_i90_pol_amr.jl`: s455 (γ = 13/9), static superposed Kerr–Schild with a = 0 and centres ±12.5 (cos φ₀, sin φ₀, 0), φ₀ = −0.2233022; 230 GHz; i = 90°; FOV 40 M; d = 1000; C = 80; nmax 2500; M = 6.5e9 × 1.989e33 g (total); mass scale 1e24 g; R_low 1, R_high 10, σ > 1 cut. RAPTOR has no image-plane AMR: it renders uniform images (camera blocks of 8² pixels for 1024²; `NPIX1D=8 tools/r07_build.sh`), compared with BlackLightPlus uniform images (`tools/r10_blp.jl`, `af1b2d1`) and with the BlackLightPlus AMR mosaic re-rendered at HEAD (`af1b2d1`; base 256², block 8, max level 4, effective 4096²) block-averaged to 1024². RAPTOR reads γ from the dump header (1.44444444444, 3e−12 from 13/9). The blurred metrics take D = 16.8 Mpc (θ_g = 3.820 µas/M) and compare the codes with each other only.

### Uniform 1024²

| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | Δm_net | Δ⟨\|m\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 90 | 80 | -2.69e-08 | 1.12e-07 | 4.11e-13 | 3.03e-06 | 1.61e-05 | -0.0000 | -0.0000 | -0.00 | -0.00000 | 3.06 | 0.000 | 1.000000 | pass |
| 90 | 80 | -1.66e-07 | 4.73e-07 | 4.17e-13 | 3.18e-06 | 6.85e-05 | -0.0000 | -0.0000 | -0.00 | -0.00000 | 5.37 | 0.000 | 1.000000 | pass |
| 90 | 160 | -2.58e-07 | 6.27e-07 | 1.06e-14 | 5.21e-07 | 8.62e-05 | -0.0000 | +0.0000 | -0.00 | +0.00000 | 5.71 | 0.000 | 1.000000 | pass |

Rows: leg (b) at C = 80, then leg (a) at C = 80 and 160 (nmax 5000; BlackLightPlus at the same C).
The ring column includes hole2's native diameter, which is not gated: at i = 90° its ring cuts cross emission-free sky (smallest half-cut maximum 3.5e-88 of the image peak), where the peak finder lands on values many decades below the image; its blurred ring agrees.

Emission-free tail. Where BlackLightPlus is below 1e−60 of the peak (692205 pixels), leg (b) exceeds 1e−30 of the peak in 15 isolated pixels (at most 3.8e-10 of the peak), and the converse tail reaches 6.4e-10; these single pixels move with C in each code (thin features met or missed by the samples). Leg (a) leaves C = 80: 568886 pixels, up to 1.9e-09 of the peak, 8.0e-09 of the flux, C = 160: 597219 pixels, up to 2.6e-09 of the peak, 1.2e-08 of the flux there. RAPTOR's native step does not extinguish radiation behind optically thick gas: `check_stiffness` selects the implicit trapezoid, whose transmission over one absorbing segment is (1 − τ/2)/(1 + τ/2) → −1 (`tools/r10_trapezoid.jl`, `evidence/R10/trapezoid.json`: τ = 3: -0.200 against e^−τ = 5.0e-02, τ = 10: -0.667 against e^−τ = 4.5e-05, τ = 100: -0.961 against e^−τ = 3.7e-44; the LD85 step returns e^−τ), so a sign-alternating remnant of the radiation behind each thick segment reaches the camera.

**Fork fix (`e6359a6`, `POL_KEEP_I`).** Upstream RAPTOR reads a pixel out as I = Q = U = V = 0 unless its polarized invariant exceeded 1e−100 at the last sample; on this deck that blanked most emission-free pixels and unpolarized emission in general (`tools/r10_polkeep.jl`: Gold T3 through the polarized build, I = 0 on 64 of 64 pixels before, |I_pol/I_unpol − 1| = 4.4e−16 after). The matched build now reads out the transported I.
- leg (b): pass
- leg (a): pass

### AMR mosaic against RAPTOR

A uniform 1024² image samples each pixel at its centre; the block-averaged mosaic averages 16 finest sub-pixels wherever the AMR refined. On this edge-on image the two sampling rules differ by more than the threshold in either code (first rows), so the gate compares like with like: the central 20 M × 20 M zone, which holds every refined block (yes), rendered uniformly at the finest AMR pixel (2048², 40/4096 M) and block-averaged into the uniform 1024² frame (composite).

| pair (full frame, block-averaged to the uniform grid) | L1(I) | NMSE(I) | L1(\|P\|) | L1(V) | LP pair | Δm_net | gate |
|---|---|---|---|---|---|---|---|
| RAPTOR (b) uniform (point-sampled) vs AMR | 1.52e-01 | 7.35e-02 | 1.48e-01 | 2.01e-01 | 0.8419 / 0.8369 | -0.0040 | report |
| AMR vs BlackLightPlus uniform (point-sampled) | 1.52e-01 | 6.35e-02 | 1.46e-01 | 1.91e-01 | 0.8369 / 0.8419 | +0.0040 | report |
| RAPTOR (b) composite vs AMR | 4.34e-03 | 2.26e-04 | 4.50e-03 | 6.97e-03 | 0.8369 / 0.8369 | -0.0003 | pass |
| BlackLightPlus composite vs AMR | 4.34e-03 | 2.26e-04 | 4.50e-03 | 6.96e-03 | 0.8369 / 0.8369 | -0.0003 | report |

LP of the full 4096² mosaic: 0.8430 (record at bb233b2: 0.843); per-level rays [65536, 16640, 41216, 116992, 333312], 6.3% of the base blocks refined, carrying 99.4% of the flux.

**Zone at the finest pixel** (2048², 7.9% of its pixels at the finest AMR level), pixel by pixel:

| pair | L1(I) | L1(\|P\|) | L1(V) | LP pair |
|---|---|---|---|---|
| RAPTOR (b) vs BlackLightPlus uniform | 1.18e-07 | 1.41e-06 | 1.30e-05 | 0.8430 / 0.8430 |
| RAPTOR (b) vs AMR crop | 5.39e-03 | 5.61e-03 | 8.39e-03 | 0.8430 / 0.8430 |
| BlackLightPlus uniform vs AMR crop | 5.39e-03 | 5.61e-03 | 8.38e-03 | 0.8430 / 0.8430 |

Over the finest-level pixels alone RAPTOR (b) and the AMR crop differ by L1(I) 1.13e-07; the rest of the zone difference is the coarser painting of the AMR's lower levels.

### Timing (64 threads, same host)

| run | rays | wall [s] | rays/s |
|---|---|---|---|
| RAPTOR (b) uniform 1024² | 1048576 | 238.4 | 4398 |
| RAPTOR (a) uniform 1024² | 1048576 | 285.3 | 3676 |
| BlackLightPlus uniform 1024² | 1048576 | 102.0 | 10277 |
| BlackLightPlus uniform 2048² | 4194304 | 357.6 | 11728 |
| RAPTOR (b) zone 2048² (fov 20) | 4194304 | 1874.0 | 2238 |
| BlackLightPlus zone 2048² (fov 20) | 4194304 | 438.4 | 9567 |
| BlackLightPlus AMR (effective 4096²) | 573696 | 83.4 | 6880 |

BlackLightPlus walls are image walls (snapshot load 3.4 s excluded); RAPTOR's are process walls including its load. The AMR wall is `render_amr` over all levels, polarized; the unpolarized level-0 probe that sets the refinement cut (10.0 s, compilation included) is not counted. The zone runs are slower per ray than the full frame: their rays concentrate near the holes, where h = d/C is small.

At 2048²: AMR block-averaged vs BlackLightPlus uniform L1(I) 9.69e-02, L1(\|P\|) 9.40e-02; BlackLightPlus 1024² vs its 2048² block-averaged L1(I) 1.60e-01.

## Figures

- `figures/images_i17.png`, `images_i60.png`, `images_i90.png`: I, |P|, EVPA and V from BlackLightPlus and RAPTOR (a) at C = 320, with the per-pixel differences of legs (a) and (b).
- `figures/convergence.png`: normalized L1 of I, |P| and V against C: each code's self-convergence and the code-to-code differences of legs (a) and (b).
- `figures/timing.png`: phase walls of the R08 runs.
- `figures/r09_s122_images_i17.png` … `r09_s093_images_i90.png`: the R09 panels (as `images_i*.png`) for s122 and s093 at C = 320; `figures/r09_s122_convergence.png`, `r09_s093_convergence.png`: their convergence curves.
- `figures/r10_i90.png`: the R10 deck: BlackLightPlus uniform, RAPTOR (b) uniform and the block-averaged AMR mosaic (I, |P|, EVPA, V), with the differences RAPTOR − BlackLightPlus and RAPTOR composite − AMR; `figures/r10_zone.png`: the central 20 M zone at the finest AMR pixel.

## Caveats

- The thresholds are evaluated on the fiducial mass scale of the BlackLightPlus figure record; neither code is re-calibrated.
- m_net, ⟨|m|⟩ and EVPA_net of s455 at i = 60° and 90° are not converged in C or in resolution in either code (Faraday depth 10²–10⁴ rad); they are compared code to code at equal discretisation only.
- RAPTOR's upstream Bardeen camera, native step size and single-hole termination are not exercised; the stock leg runs on the matched geometry.
- Leg (d) of `dag.md` (RAPTOR's native `stepsize()`) is not run: the native step and termination are single-hole (r from the origin, horizon of a hole at the origin) and do not terminate at the two SKS horizons.
- Timings cover 16 and 64 threads (the host is shared; `dag.md` lists 1, 16, 64, 128).
- Timings are on a shared host; the load average is recorded next to each leg. RAPTOR's rays/s include the pixel-loop scheduling of 230,400 rays in chunks of 64; BlackLightPlus threads over rays in the same way (`Threads.@threads :dynamic`).

