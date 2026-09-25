# RAPTOR and BlackLightPlus on the s455 binary-black-hole snapshot

Nodes R07 (image comparison) and R08 (CPU timings) of `dag.md`; design and attribution rule in `design.md` (§3 conventions, §4 modifications, §5 attribution). Every number below is read from `evidence/R07/metrics.json`, `evidence/R07/polstep.json`, `evidence/R07/native_cold.txt` and `evidence/R08/timing.json` by `tools/r08_report.py`.

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

## Figures

- `figures/images_i17.png`, `images_i60.png`, `images_i90.png`: I, |P|, EVPA and V from BlackLightPlus and RAPTOR (a) at C = 320, with the per-pixel differences of legs (a) and (b).
- `figures/convergence.png`: normalized L1 of I, |P| and V against C: each code's self-convergence and the code-to-code differences of legs (a) and (b).
- `figures/timing.png`: phase walls of the R08 runs.

## Caveats

- The thresholds are evaluated on the fiducial mass scale of the BlackLightPlus figure record; neither code is re-calibrated.
- m_net, ⟨|m|⟩ and EVPA_net of s455 at i = 60° and 90° are not converged in C or in resolution in either code (Faraday depth 10²–10⁴ rad); they are compared code to code at equal discretisation only.
- RAPTOR's upstream Bardeen camera, native step size and single-hole termination are not exercised; the stock leg runs on the matched geometry.
- Leg (d) of `dag.md` (RAPTOR's native `stepsize()`) is not run: the native step and termination are single-hole (r from the origin, horizon of a hole at the origin) and do not terminate at the two SKS horizons.
- Timings cover 16 and 64 threads (the host is shared; `dag.md` lists 1, 16, 64, 128).
- Timings are on a shared host; the load average is recorded next to each leg. RAPTOR's rays/s include the pixel-loop scheduling of 230,400 rays in chunks of 64; BlackLightPlus threads over rays in the same way (`Threads.@threads :dynamic`).

