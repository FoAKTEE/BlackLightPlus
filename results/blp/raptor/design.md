# RAPTOR vs BlackLightPlus on the binary-black-hole snapshots: design

Scope: ray-trace the AthenaK binary-black-hole GRMHD snapshots used for the
EHT-style mocks (`figures/fig_eht_bbh.jl`, `figures/fig_bbh_i90_pol_amr.jl`)
with a modified RAPTOR and with BlackLightPlus (BLP). Every model choice is
matched, so the residual differences are discretisation and solver
differences. Those residuals are measured by convergence studies and by
unit-level comparisons. The nodes that implement this design are listed in
`dag.md` (R01–R11). The Chandra ledger paper id is `arxiv-1801.10452`.

Code references:
- RAPTOR: `ref-code/jordydavelaar-raptor` at upstream `b91e937`; paths are
  relative to that directory.
- BLP: branch `paper` at `ee88885`.
- Paper references: the source notes in `sources/`.

## 0. Evidence base

1. **Code reading** of the whole RAPTOR `src/`, `model/`, `run/`,
   `setup.sh`, `python/plotting/rapplot.py` and README, with `file:line`
   references. BLP references come from `src/` and the two figure drivers.
2. **Measured here.** Two RAPTOR builds, run on a synthetic BHAC-format
   snapshot (§1.1):
   - gcc 11.4 through `h5cc`;
   - NVHPC `nvc` 25.9 with `-mp`.

   Timings are taken on the 2× Xeon Platinum 8360Y host (144 hardware
   threads, 503 GB RAM).
3. **Literature.** Eight arXiv mirrors and one journal PDF, with notes in
   `sources/`:
   - RAPTOR I (1801.10452) and RAPTOR II (2007.03045);
   - Davelaar+ 1712.02266, 1906.10065, 1811.08369, 2112.05828, 2310.19766;
   - the EHT code comparisons: Gold+2020 (`doi-10.3847_1538-4357_ab96c6`)
     and Prather+2023 (2303.12004).
4. **Recorded BLP runs.** `figures/out/eht_bbh/*/records.json` hold the
   calibrated mass scales, fluxes, ring metrics and timings. They were
   produced by the ee88885 driver; `src/` is identical to 7c388c5.
- **No GPU is usable on this host** (`nvaccelinfo`: "No accelerators found";
  NVML error). All legs in this design run on the CPU. The GPU legs are
  specified in R11 and deferred.

## 1. RAPTOR audit

### 1.1 Code map and build

| unit | role | notes |
|---|---|---|
| `setup.sh` | copies `run/*`, `model/<code>/*` into the run dir, `sed`-edits `definitions.h` (metric, integrator, POL, SFC, BPOL, `num_frequencies`), `make clean && make all` | `-g` flag has no effect: the script compares with `"SMR"`/`"AMR"` (`setup.sh:118-129`) but the default and documented values are lower case, so `AMR 0 / SMR 1` from `definitions.h:49-50` always holds |
| `run/makefile` | `h5cc -fopenmp -std=c99 -O2 -lgsl`, 13 sources | needs `$RAPTOR`; libraries hdf5, gsl, OpenMP only |
| `run/model.in` | 14 runtime parameters read positionally (`core.c:226-242`): MBH [Msun], DISTANCE [kpc], M_UNIT [g], R_LOW, R_HIGH, INCLINATION [deg], IMG_WIDTH, IMG_HEIGHT, CAM_SIZE_X/Y [GM/c²], FREQS_PER_DEC, FREQ_MIN [Hz], STEPSIZE, MAX_LEVEL | compile-time: `num_frequencies`, `num_pixels_1d=10`, `rcam=1e4`, `max_steps=1e4`, `RT_OUTER_CUTOFF=50`, `SIGMA_CUT=1`, `THETAE_MAX/MIN` (`definitions.h:38-142`) |
| `src/main.c` | read model, `init_model`, `set_constants`, camera blocks, per-block trace + RT, spectrum, output | image computed in 10×10-pixel blocks (`main.c:134-150`) |
| `src/core.c` | `read_model`, `calculate_image_block` (OpenMP), `compute_spec` | parallel region per block of 100 pixels, `schedule(static,1)` (`core.c:286-323`) |
| `src/camera.c` | block camera, SMR/AMR image refinement, pixel lookup | pixel centres `(k+0.5)·px − fov/2` (`camera.c:61-77`) |
| `src/metric.c` | `metric_dd/uu` for CAR, BL, MBL, KS, MKS, MKSHARM, MKSBHAC, MKSN, CKS; analytic Γ for BL/KS/MKS*; numerical Γ; Bardeen camera | `metric.c:18-398`, `:435-608`, `:612-1123`, `:1131-1223` |
| `src/gr_integrator.c` | RK2, RK4, velocity Verlet, RK45; step size; `integrate_geodesic` | `:20-148`, `:151-181`, `:251-302`, `:334-427` |
| `src/GRmath.c` | `get_r`, index gymnastics, BL↔KS↔CKS, `normalize_null`, plasma-frame frequency, pitch angle | `:448-461`, `:554-598`, `:801-835` |
| `src/rte_integrator.c` | unpolarized transfer | `:18-145` |
| `src/pol_rte_integrator.c` | polarized transfer: f-vector transport, RK4/trapezoid Stokes step, camera readout | `:19-101`, `:409-544`, `:563-707`, `:709-829` |
| `src/pol_emission.c` | thermal, κ and power-law j/α/ρ | thermal `:122-135`, `:210-235`, `:288-304`, `:357-372`, `:431-447`, `:529-532`, `:590-593`, `:671-674` |
| `src/tetrad.c` | random and oriented tetrads, `tetrad_d` | `:54-159`, `:163-255`, `:302-324` |
| `src/io.c` | HDF5 image `[block][pixel]` per Stokes and τ, spectrum, uniform ASCII image | `:284-545`, `:584-622` |
| `model/bhac` | BHAC AMR `.dat` reader (con2prim inside RAPTOR), units, fluid parameters | `model.c:43-848`, `:850-859`, `:999-1199` |
| `model/harm3d`, `model/HAMR` | HARM-family readers (MKS, uniform grids) | constant `T_p/T_e=3` (`harm3d/model.c:171-173`) |

**CPU and GPU.** The public code is CPU-only. It contains two OpenMP
pragmas (`core.c:286`, `bhac/model.c:805`) and no OpenACC or CUDA; this was
checked by grep on all 11 upstream branches. RAPTOR I describes an OpenACC
version: 104,900 geodesics/s on one GTX 1080 against 10,707/s on 10 CPU
cores (`sources/arxiv-1801.10452.md`), but that version is not public.

Obstacles to an OpenACC port:
- the geodesic step takes a function pointer (`rk4_step(double*, void (*f)(...), ...)`, `gr_integrator.c:151`);
- `determ` is recursive (`tetrad.c:20-48`), and so is `rk45_step` (`gr_integrator.c:142`);
- there is a per-pixel `malloc` in the hot loop (`core.c:291`);
- the Bessel functions come from GSL (`pol_emission.c:48`).

NVHPC 24.1–26.3 is installed (`/opt/nvidia/hpc_sdk`). `nvc -mp` builds the
unmodified sources (`fork.md`).

**Build status (R01).** Both builds succeed:
- gcc/`h5cc`, configured with `-c=bhac -m=cks -r=pol -i=rk4`;
- `nvc -mp`.

The tutorial snapshot host `astro.ru.nl` is unreachable from this machine
(HTTP and HTTPS time out), so the default example cannot be run. The
baseline instead runs on a synthetic BHAC-format snapshot. It is written by
`tools/make_bhac_synthetic.py` with these parameters:
- 64 level-1 leaves of 16³ cells over [−40, 40]³ M, CKS, a = 0.9375;
- a Gaussian ring with R = 12 and width 4, at rest in the normal frame;
- a vertical field with β = 10; γ = 4/3.

The image is 100², FOV 40 M, i = 60°, 230 GHz. Results:

| build | threads | wall [s] | CPU [s] | F_ν [Jy] (M_UNIT = 1e25 g) |
|---|---|---|---|---|
| gcc -O2 | 1 | 184.0 | 183.9 | 1.37395e-06 |
| gcc -O2 | 16 | 15.73 | 220.1 | 1.37395e-06 |
| gcc -O2 | 64 | 9.55 | 410.1 | 1.37395e-06 |
| nvc -mp -O2 | 16 | 6.67 | 102.4 | 1.37405e-06 |

From these runs:
- The single-thread cost is 18.4 ms per pixel. Most of it is the numerical Christoffel symbols: 16 metric evaluations per RHS call, 4 calls per RK4 step.
- The thread speed-ups are 11.7× at 16 threads and 19.3× at 64. The cause is the 100-pixel parallel regions.
- The nvc build is 2.4× faster than gcc and agrees on the flux to 7.3e-5.
- `tools/raptor_h5_to_array.py` reassembles the HDF5 image. Its sum reproduces the spectrum-file flux to 1e-6.

### 1.2 Geodesics, metric, camera

- **State and equation.** The state is (x^μ, k^μ), contravariant. The acceleration is −Γ^μ_αβ k^α k^β (`gr_integrator.c:306-331`).
- **Christoffel symbols.**
  - For CKS they are numerical (`connection_num_udd`, `metric.c:435-608`). The stencil is 4th-order central, (−g₊₊ + 8g₊ − 8g₋ + g₋₋)/(12h), with a relative step h = `delta_num`·max(|x^i|, 1e−6) = 1e−4·max(|x^i|, 1e−6), so h drops to 1e−10 at x^i ≈ 0 (`metric.c:461-485`, `definitions.h:133`). Γ = ½ g^{αk}(∂g) comes from the analytic g^{μν}.
  - The analytic Γ exist only for BL, KS, MKS, MKSHARM and MKSBHAC (`metric.c:612-1123`).
  - Papers: Γ by finite differences (1906.10065 l.520–522); Prather+2023 states "analytic or 4th-order FD".
- **CKS metric** (`metric.c:170-207`, `:358-395`). It is single-hole Kerr, g = η + f l l, with two regularisers:
  - 1/(Σ + 1e−3) for r < 1;
  - l_z = z/(r + 1e−3) in g_μν, but z/r in g^μν.

  As a result, g_μν g^νλ ≠ δ. Measured for a = 0.9375:

  | x | max \|g g⁻¹ − I\| |
  |---|---|
  | (3, 0, 3) | 1.0e−4 |
  | (6, 2, 5) | 2.3e−5 |
  | (10, 0, 10) | 9.7e−6 |
  | (20, 5, 15) | 2.1e−6 |

  The error falls off roughly as r⁻². `get_r` inverts the single-hole KS radius (`GRmath.c:448-461`).
- **Integrators.** The default is classical RK4 (`definitions.h:142`, `gr_integrator.c:151-181`). RK2, velocity Verlet and RK45 are also available. RK45 retries by recursion and compares the error with `tol` (`:20-148`).
- **Step size** for CKS: dλ = −STEPSIZE · r / (|k^x| + |k^y| + |k^z|) (`gr_integrator.c:251-274`). The default STEPSIZE is 0.01 (`run/model.in`). For MKSBHAC it is additionally capped at a quarter of a cell crossing (`:277-294`).
- **Termination** (`gr_integrator.c:366-369`, `constants.c:72-73`, `definitions.h:124-137`): the ray stops when
  - r ≤ r_h(1 + 0.01) with r_h = 1 + √(1 − a²); or
  - r ≥ 1.1 r_cam = 1.1e4; or
  - 1e4 steps are reached; or
  - the "θ-turn" order cut fires. That cut uses `photon_u[6]`, which is k^y in CKS, and the uninitialised `thetadot_prev` (`:340`, `:386-391`). With `max_order = 100` it never fires.
- **Camera** (`metric.c:1131-1223`): Bardeen–Cunningham impact parameters at the BL point (r_cam = 1e4, θ = i, φ = 0) with E = 1:
  - L = −α sin i;
  - Q = β² + cos²i (α² − 1);
  - k_θ = sign(β) √|…|;
  - k^r > 0 from the null condition.

  The ray is then transformed BL → KS → CKS (`GRmath.c:634-798`) and integrated with dλ < 0. This means k^μ is the received, future-directed wave vector, traced backward. Pixel centres are at (k + 0.5) px − fov/2 (`camera.c:74-77`). The field of view is given in units of GM/c² of MBH (`core.c:236-237`, `camera.c:53-59`). Images must be square and multiples of 10 pixels (`camera.c:28-33`).
- **Output layout.** HDF5 datasets are `[block][pixel]`. Block b has ind = (b / n_b, b mod n_b); pixel p has (x, y) = (p / 10, p mod 10); α comes from x and β from y (`camera.c:39-45`, `:69-77`). The files also hold per-pixel `alpha` and `beta` (`io.c:510-542`).

  The ASCII `uniform_img` writer finds pixels by truncating `(x − corner)/dx − 0.5` (`camera.c:261-262`). This truncation duplicates pixels: on the R01 image only 8278 of 10000 rows are unique, and the ASCII flux is 1.1% high. The comparison therefore reads the HDF5 file only.

### 1.3 Radiative transfer

- **Unpolarized** (`rte_integrator.c:49-140`). The march runs from state n = steps−1 down to state 1. Each state is weighted by the preceding step dl[n−1], as a Riemann sum. The segment is solved exactly with constant S = j/α, using a series for dτ < 1e−5.

  Invariants:
  - j/ν² and να;
  - dl_s = dl · m_e c²/(hν), with dτ = α dl_s · C and C = R_g h/(m_e c²);
  - the output is I · ν³.

  This is the same per-state weighting as BLP (`reverse_march.jl:73-76`).
- **Polarized** (`pol_rte_integrator.c:727-829`). The march is source → camera. The polarization state is carried as a complex vector f^μ with I_inv and I_pol, parallel-transported by RK4 along the stored samples (`:55-101`, `:794-804`).

  Per sample:
  1. Build the fluid tetrad with e₀ = u, e₂ = b_⊥ ("para"), e₁ = ε(u, k, b)/(ωN) ("perp") and e₃ = k/ω − u (`tetrad.c:163-255`).
  2. Convert f^μ to Stokes as Q = I_p(|f¹|² − |f²|²), U = 2 I_p Re(f¹* f²), V = −2 I_p Im(f¹* f²) (`:103-112`).
  3. Take one constant-coefficient step. It is RK4 unless the 4th-order stability test on the eigenvalues fails (`check_stiffness`, `:360-407`, threshold 0.99); then an implicit trapezoid LU solve is used (`:493-544`).
  4. Renormalise if |P| > I (`:653-667`).

  At the camera the Stokes basis is built from the Eulerian observer and `cam_up = (0, 0, 0, −1)`, which is −ẑ in CKS (`:709-725`).

  Other properties of the polarized march:
  - It is only active for r < `RT_OUTER_CUTOFF` = 50 (`:780`).
  - k is renormalised to null when |k·k| > 1e−6 and r > 2 (`:771-772`).
  - The papers state IEEE/IAU Stokes, EVPA east of north and V > 0 right-handed (2007.03045 l.159).
- **Thermal coefficients** (`pol_emission.c`, `DF=TH`):
  - j_I, j_Q, j_V use Dexter-2016-type fits I_I, I_Q, I_V of x = ν/ν_c with ν_c = 3eB sinθ Θ²/(4π m_e c) (`:288-304`, `:357-372`, `:431-447`). **j_Q > 0.**
  - Absorption follows Kirchhoff with the full Planck function, computed as `exp(hν/kT) − 1` (`:53-60`, `:529-532`, `:590-593`, `:671-674`). At 230 GHz hν/kT ≈ 1.9e−9/Θ_e, so this form loses about half of the double precision. BLP uses `expm1` (`coefficients.jl:336`, `:396`).
  - ρ_Q = … f_m(X) (K₁/K₂ + 6Θ) sin²θ, with Shcherbakov's f_m (`:122-135`).
  - ρ_V uses K₀/K₂ × (1 − 0.11 ln(1 + 0.035 X)) cos θ (`DEXTER=0`, `:210-235`).
  - The Bessel functions are GSL K_n, switched to small-argument asymptotics for 1/Θ < 0.2 (`:30-51`).
  - If √(j_Q² + j_V²) > j_I, j_Q and j_V are rescaled (`pol_rte_integrator.c:345-350`).
  - Pitch angle: θ_B = acos(k·b / (|k·u| |b|)) with the received k (`GRmath.c:815-835`).
- **Stokes basis.** In the fluid frame Q > 0 lies along e₁ ⟂ B, so the physical j_Q is positive in RAPTOR's basis, while BLP has j_Q < 0 in the M21 basis. Both describe emission polarized ⟂ B. At the camera, e₂ ∥ ∓(projected ẑ), so RAPTOR's Q > 0 means east–west. We therefore expect Q_R = −Q_IAU. The signs of U and V in RAPTOR's (e₁, e₂, k) handedness are fixed empirically in R02.

### 1.4 Plasma, units, data (BHAC model)

- **Units** (`bhac/model.c:850-859`):
  - L = GM/c² with the MBH of `model.in`;
  - ρ_unit = M_UNIT/L³;
  - B_unit = c √(4π ρ_unit);
  - n_e = ρ ρ_unit/(m_p + m_e) + 1e−6 cm⁻³ (`:1070`).
  - Images are in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹ × 1e23. Flux = Σ I · (px R_g/D)² (`core.c:328-361`).
- **Fluid four-vectors** (`:1080-1131`). The inputs are the BHAC primitives Γv^i and the Eulerian B^i. They are converted with
  - u^t = Γ/α and u^i = Γv^i − Γβ^i/α;
  - b^t = Γ B^i v_i/α and b^i = (B^i + α b^t u^i)/Γ.

  This is the BHAC convention, B^i_BHAC = α · B^i_AthenaK. BLP's AthenaK decode uses b⁰ = u_i B^i (`primitives.jl:23-49`). **Applying the BHAC formula to AthenaK `bcc` would mis-scale b by the lapse**, which is 0.5–0.8 near the holes.
- **Field strength.** B = √(|b²| + 1e−6) · B_unit (`:1134-1139`). The 1e−6 floor is in code units, so it is not negligible in low-density regions.
- **Electron temperature** (`:1168-1186`). With β = u(γ − 1)/(½b²) and R = (R_high β² + R_low)/(1 + β²):

  Θ_e = (u/ρ) · (1/3) · (m_p/m_e)/(1 + R).

  The factor 1/3 is fixed and does not follow the dump's γ. For γ = 13/9 this gives 0.75 × the p-based BLP Θ_e.
- **Cuts** (`:1022`, `:1191-1196`): no emission if r < 1, σ = b²/ρ > 1, r > 50, Θ_e > 100 or Θ_e < 1e−3.
- **Data layout.**
  - BHAC `.dat` holds conserved variables. RAPTOR runs its own con2prim, with an entropy backup (`:322-461`).
  - Primitives are stored in `double ****p`, with one `malloc` per cell (`:862-876`). That is about 8.6e9 allocations for s093, so the layout cannot hold the AthenaK snapshots.
  - Leaves are found by a linear scan (`find_igrid`, `:56-90`), cached per ray.
- **Interpolation** (`:878-996`). Trilinear inside the owning block, clamped at the block edges. The weights are measured from the cell's lower face, (x − (lb + i dx))/dx, while the data sit at cell centres lb + (i + ½)dx (`:312`). **The samples are therefore shifted by half a cell.** The paper describes nearest-neighbour sampling (1906.10065 l.557–562), which is neither of these.

## 2. BLP reference configuration (the target)

The figure drivers set the following (records from ee88885):

**Units and constants** (`fig_eht_bbh.jl:73-95`):
- M = 1.3e10 M_sun, the total binary mass, with M_sun = 1.98847e33 g;
- D = 16.8 Mpc;
- G = 6.6743e−8 (`constants.jl`);
- L_unit = 1.919670598e15 cm, θ_g = 7.638 µas/M.

**Camera** (`camera.jl:40-58`, `render.jl:10-13`):
- plane-parallel, d = 1000 M, φ_o = 0;
- pixel position x = (0, d sin i − β cos i, α, d cos i + β sin i);
- the provisional received k^μ = (1, sin i, 0, cos i) is lowered with the superposed metric, the spatial k_i are kept and k_t < 0 is re-solved, normalised to k_t = −1;
- α = +ŷ is west and β = north; images are M[iβ, iα];
- FOV 56 M, 480² (px = 0.1167 M = 0.891 µas); i = 17°, 60°, 90°.

**Metric** (`spacetimes.jl:19-123`):
- `SuperposedKerrSchild`, g = η + Σ_A f_A l^A l^A with m_A = 0.5; static, with no boosts;
- centres at ±sep/2 (cos φ₀, sin φ₀, 0), with φ₀ from the dump time;
- the inverse by Woodbury; ∂g by forward-mode AD of H = ½ g^{μν} p_μ p_ν.

**Integrator** (`integrator.jl:293-299`, `spacetimes.jl:143-155`, `integrator.jl:14-57`):
- RK4 on (x^μ, p_μ) with h = min_A(r_A − r_h,A)/C, C = 80, nmax = 4000;
- capture if d < 1e−3, if d is non-finite, or if the normal-frame energy exceeds 100;
- escape if R > 1100 and increasing.

**Plasma** (`coefficients.jl:240-304`, `primitives.jl`, `athenak_reader.jl:360-416`, `athenak_z4c.jl:272-308`):
- n_e = ρ/m_p;
- Θ_e = (m_p/m_e)(p/ρ)/(1 + R) with p = (γ − 1) · Float32(p_dump/(γ − 1)) and γ = 13/9 from the dump;
- R_low = 1, R_high = 40 fiducial;
- cut σ > 1 only, with no Θ_e cap, no floor and no excision mask;
- trilinear interpolation on cell centres inside the owning block, clamped at the edges. The owning block comes from a dense finest-block table (≤ 8e6 cells) or, above that, a hash on (level, lx1, lx2, lx3), both owning lo ≤ x < hi; the linear scan (lo < x ≤ hi, last block wins) is the fallback for meshes neither can index. All three snapshots use the hash.

**Transfer** (`reverse_march.jl`, `transfer.jl:34-170`):
- a Riemann sum over the stored states, each weighted by the preceding step;
- the exact constant-coefficient LD85 solution; α_P > α_I is rescaled and |P| ≤ I is clamped;
- M21 thermal fits with Kirchhoff (Planck with `expm1`), A&S Bessels, and ramps below Θ_e < 0.01;
- IEEE/IAU Stokes (`conventions.md`).

**Snapshots** (`records.json`):

| id | file | sep | a_used per hole | φ₀ [rad] | C₁ = −C₂ | ms at R_high = 40 [g] | F at i = 17/60/90 [Jy] |
|---|---|---|---|---|---|---|---|
| s455 | `.../SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin` (4.83 GB, 512 blocks of 64³, levels 0–8) | 25 | 0 | −0.2227368026 | (12.19120680, −2.76124551, 0) | 2.274019497e28 | 0.9814 / 1.4633 / 2.1660 |
| s093 | `/data2/haiyangw/aurora/cbd_rel/torus.mhd_w_bcc.00093.bin` (38.7 GB, 4096 blocks, levels 1–9) | 25 | 0.4 (χ = 0.8) | 2.557236714 | (−10.42584453, 6.89577884, 0) | 2.252873787e20 | 0.9999 / 0.7472 / 0.6950 |
| s122 | `/data2/haiyangw/aurora/cbd_rel/torus.mhd_w_bcc.00122.athdf` (4.83 GB) | 16 | 0.4 (χ = 0.8) | 1.530586362 | (0.32159304, 7.99353351, 0) | 1.000498515e21 | 0.9863 / 1.1681 / 1.1813 |

**AMR i = 90° figure** (`fig_bbh_i90_pol_amr.jl`) differs from the EHT driver:
- s455 only, with φ₀ = −0.2233022 and centres ±12.5 (cos φ₀, sin φ₀);
- M = 6.5e9 × 1.989e33 g as the total mass;
- ms = 1e24 g, uncalibrated;
- R_high = 10, γ = 13/9, FOV 40, nmax 2500;
- AMR base 256², max level 4 (effective 4096²).

Its record was produced at bb233b2, which differs from HEAD in 34 `src/` files. The comparison therefore re-runs BLP at HEAD.

## 3. Convention matching

In the "common setting" column below, "BLP" means that the RAPTOR fork
reproduces the BLP choice exactly. "Native" means that each code keeps its
own algorithm, and the resulting difference is measured separately by the
node named.

| # | item | RAPTOR (upstream) | BLP | common setting | residual measured in |
|---|---|---|---|---|---|
| C1 | spacetime | single Kerr in CKS with 1e−3 regularisers (`metric.c:170-207`) | static superposed KS, 2 holes, Σ f_A l l (`spacetimes.jl:45-61`) | BLP (M1) | R04, R05 |
| C2 | g^μν | analytic single-KS form (`metric.c:358-395`) | Woodbury 2×2 exact (`spacetimes.jl:68-96`) | BLP (M1) | R05 (‖g g⁻¹ − I‖ ≤ 1e−13) |
| C3 | ∂g, Γ | 4th-order FD, relative step 1e−4 \|x^i\| (`metric.c:435-608`) | forward AD of the Hamiltonian (`integrator.jl:157-218`) | native FD (M2a) plus analytic ∂g (M2b) | R04 (FD vs analytic vs AD) |
| C4 | geodesic variables | (x^μ, k^μ), Christoffel form | (x^μ, p_μ), Hamiltonian form | native (same ODE) | R04 (O(h⁴) truncation difference) |
| C5 | integrator | RK4 (`gr_integrator.c:151-181`) | RK4 (`integrator.jl:293-299`) | RK4 in both | — |
| C6 | step size | −STEPSIZE · r/Σ\|k^i\| (`gr_integrator.c:251-274`) | min_A(r_A − r_h,A)/C, C = 80 | BLP rule (M3); native rule as a convergence leg | R04, R07 |
| C7 | termination | r < 1.01 r_h, r > 1.1e4, 1e4 steps | d < 1e−3, E_n > 100, R > 1100 outbound, 4000 steps | BLP (M3) | R04 (termination class per ray) |
| C8 | camera | Bardeen BL at r = 1e4 then BL→KS→CKS (`metric.c:1131-1223`) | plane-parallel SKS at d = 1000 (`camera.jl:40-50`) | BLP (M4) | R02, R04 |
| C9 | image axes | α (L = −α sin i), β = sign(k_θ): Bardeen | Bardeen; α = +ŷ west, β north | identical by construction; verified in R02 | R02 |
| C10 | pixel centres, FOV | (k + ½)px − fov/2; FOV in GM/c² of MBH | same; FOV 56 M, 480² | same (MBH = total mass) | — |
| C11 | output layout | HDF5 `[block][pixel]` | `M[iβ, iα]` | converter `tools/raptor_h5_to_array.py` | R01 (flux identity 1e−6) |
| C12 | constants | G 6.674e−8, M_sun 1.989e33, kpc 3.086e21, m_e 9.1093829e−28, m_p 1.6726219e−24 (`definitions.h:148-157`) | G 6.6743e−8, M_sun 1.98847e33 (driver), pc 3.0856775814913673e18, m_e 9.1094e−28, m_p 1.6726e−24 | BLP (M9) | R03 (the unmatched set changes L_unit by 2.2e−4) |
| C13 | mass scale | M_UNIT free | ms calibrated to 1 Jy at i = 17° | BLP's calibrated ms per snapshot passed to both | — |
| C14 | n_e | ρ/(m_p + m_e) + 1e−6 cm⁻³ | ρ/m_p (μ = 0.5, n_e/n_i = 1) | BLP (M6) | — |
| C15 | \|B\| | √(\|b²\| + 1e−6) · B_unit | √(b²) · B_unit | BLP (M6) | — |
| C16 | u^μ, b^μ from dump | BHAC convention (Eulerian B^i) | AthenaK convention b⁰ = u_i B^i (`primitives.jl:23-49`) | BLP (M5) | R06 |
| C17 | Θ_e | (u/ρ)(1/3)(m_p/m_e)/(1 + R) | (m_p/m_e)(p/ρ)/(1 + R), p via Float32 eint | BLP (M6) | R06 |
| C18 | R(β) | R_high β²/(1 + β²) + R_low/(1 + β²), β = p/(b²/2) | same | same | — |
| C19 | emission cuts | σ > 1, Θ_e ∉ [1e−3, 100], r > 50, r < 1 | σ > 1; vacuum if B = 0, ρ ≤ 0 or p ≤ 0; nothing beyond R = 1100 | BLP (M6) | R07 (cut attribution) |
| C20 | excision gas | no mask | no mask | no mask | — |
| C21 | data reader | BHAC `.dat`, conserved variables | AthenaK `.bin` v1.1 (Float32) and `.athdf` | BLP (M5) | R06 |
| C22 | interpolation | trilinear, half-cell-shifted weights, clamp at block edge | trilinear on cell centres, clamp, owning block only | BLP (M5) | R06 (point samples) |
| C23 | block ownership | first leaf with lb ≤ x + 1e−9 < ub (linear scan) | lo ≤ x < hi (dense table or per-level hash); lo < x ≤ hi, last wins (scan fallback) | BLP (M5) | R06 |
| C24 | j, α, ρ fits | Dexter16-type I_I/I_Q/I_V, Shcherbakov ρ_Q/ρ_V, GSL Bessel, Planck `exp−1` | M21, A&S Bessels, `expm1`, Θ_e < 0.01 ramps | BLP port (M7); native fits as a separate leg | R03 (coefficient ratios), R07 (native-fit image) |
| C25 | fluid Q basis | Q ∥ e₁ ⟂ B, j_Q > 0 | code (e₂, −e₁), j_Q < 0 (M21) | physically identical; M7 flips the sign of the Q-type coefficients into RAPTOR's basis | R03 |
| C26 | camera Stokes basis | Eulerian observer, cam_up = −ẑ; expected Q_R = −Q_IAU | IEEE/IAU (north, east) | IEEE/IAU; RAPTOR output mapped by a fixed sign table fixed in R02 (M8) | R02 |
| C27 | polarized solver | RK4 or implicit trapezoid (stiff), f^μ RK4 transport, \|P\| renormalisation | exact LD85 per segment, transported camera frame, clamp | native; exact-LD85 option (M11) for a diagnostic leg | R07 |
| C28 | unpolarized solver | exact exponential, per-state dl[n−1] | exact exponential, per-state dls[n−1] | same (sample inclusion checked) | R04 |
| C29 | RT radial range | polarized r < 50 | whole ray up to R = 1100 | BLP (M6: RT_OUTER_CUTOFF ≥ 1100) | R07 |
| C30 | k renormalisation in RT | re-nulled when \|k·k\| > 1e−6, r > 2 | none | none in matched mode (M3) | R04 |
| C31 | time | fast light | fast light | fast light | — |
| C32 | spins | global `a` from dump | per hole χ m_A, along ±z | per hole a_A = 0 (s455), 0.4 (s093, s122) (M1) | R09 |
| C33 | orbital phase | — | φ₀ from tlim or header time | centres read from `records.json`, passed through a `bbh.in` file (M1) | — |
| C34 | precision | double; data Float32→double | Float64 math; Float32 data; eint rounded to Float32 | BLP storage path (M5) | R06 |
| C35 | parallelism | OpenMP per 10×10 block | Julia threads per ray | same thread count for timing; M12 parallelises over all pixels (bitwise-identical output) | R08 |

## 4. RAPTOR modifications (branch `blp-bbh`)

Each item is one or more commits. Compile-time switches keep upstream
behaviour available where a native leg is needed.

- **M1 superposed Kerr–Schild metric** (`metric == SKS`): `metric_dd`
  = η + Σ_{A=1,2} f_A l^A_μ l^A_ν at X − c_A, with exactly the BLP
  `_ks_term` (`metrics.jl:53-71`) and no regularisers. `metric_uu` is the
  exact inverse through the 2×2 Woodbury solve.

  New helpers:
  - `ks_radius_hole(X, A)` and `horizon_radius_hole(A)`;
  - `horizon_distance(X)` = min_A(r_A − r_h,A);
  - `get_r` → global R for the outer tests.

  A runtime file `bbh.in` supplies m_A, a_A (along ±z), c_A and the control
  constants. `a` in `constants.c` is not used by this metric.

  **Derivatives, the RAPTOR way:**
  - M2a (the default): `connection_num_udd` on the new `metric_dd`, with an
    absolute step h = 1e−4 · max(1, min_A r_A). This replaces the step
    h ∝ |x^i|, which collapses to 1e−10 at x^i ≈ 0 and loses 6 digits there.
  - M2b (the reference): analytic ∂_i g_μν = Σ_A (∂_i f_A l l + f_A ∂_i(l l)).
    The derivatives are hand-derived from r_A(X), with
    ∂_i r = (X_i r² + δ_iz a² Z)/(r(2r² − R² + a²)) for the KS radius,
    and Γ is formed with the exact g^μν.
- **M3 step and termination** (`STEP_BBH`):
  - h = −horizon_distance/C with C = 80;
  - capture if d < 1e−3, if d is non-finite, or if the normal-frame energy |g^{tμ}k_μ|/√(−g^{tt}) > 100;
  - escape if R > 1100 and increasing; nmax = 4000;
  - the θ-turn cut and the RT k-renormalisation are removed.

  The native `stepsize()` is kept for the STEPSIZE convergence leg.
- **M4 plane-parallel camera** (`CAMERA_PP`): the BLP `camera_ray`
  (`camera.jl:45-51`), with pixel position (0, d sin i − β cos i, α, d cos i + β sin i).
  The provisional received k^μ = (1, sin i, 0, cos i) is lowered with the SKS metric,
  the spatial k_i are kept and k_t is re-solved on the null cone, normalised
  to k_t = −1. Then k^μ = g^{μν}k_ν, which is the received future-directed
  vector integrated with dλ < 0, as RAPTOR expects.
- **M5 AthenaK reader** (`model/athenak/`):
  - reads `.bin` v1.1 (the Int32[10] header, 6 bounds and Float32 data per block) and `.athdf` (the `uov`/`B` datasets, `Levels`, `LogicalLocations`), using the variable names from the file header;
  - stores the data as contiguous Float32 arrays [block][var][k][j][i];
  - forms eint = Float32(p/(γ − 1)) in double, as BLP does;
  - finds blocks through the BlackLightPlus locate paths: the dense finest-block table, else an open-addressing hash on (level, lx1, lx2, lx3) probed coarse to fine (ownership lo ≤ x < hi), else the linear scan;
  - interpolates trilinearly on cell centres inside the owning block, clamped;
  - builds u^μ and b^μ with the AthenaK convention (`primitives.jl`), evaluated with the SKS metric at the sample point.

  Memory: 0.89 × the dump size (s455 and s122 4.0 GB, s093 32.0 GB peak RSS).
- **M6 plasma model** (model-level `get_fluid_params`):
  - n_e = ρ ρ_unit/m_p;
  - B = √(b²) B_unit;
  - Θ_e = (m_p/m_e)(p/ρ)/(1 + R(β)) with β = p/(b²/2);
  - σ > 1 cut only;
  - vacuum if B = 0, ρ ≤ 0 or p ≤ 0;
  - no r < 1 mask, no Θ_e bounds, and `RT_OUTER_CUTOFF` = 1100 (the escape radius).
- **M7 BLP coefficient set** (`EMIS_BLP`): a line-by-line port of the BLP
  thermal M21 fits:
  - j_I, j_Q, j_V;
  - Kirchhoff with `expm1`;
  - ρ_Q and ρ_V from M21 eqs. 33–37;
  - A&S Bessel polynomials;
  - the ramps below Θ_e < 0.01;
  - the α_P > α_I rescale.

  The Q-type coefficients (j_Q, α_Q, ρ_Q) are sign-flipped into RAPTOR's
  e₁ ⟂ B basis. The upstream Dexter16 set stays available as `EMIS_NATIVE`.
- **M8 Stokes readout.** The camera tetrad takes up = +projected ẑ, i.e.
  north (e₂). The readout is referred to (e₂, −e₁) through the sign table
  fixed in R02, so that the RAPTOR HDF5 output is IEEE/IAU. The mapping is
  applied in the code, not in post-processing.
- **M9 constants.** `definitions.h` takes BLP's G, M_sun, pc, m_e, m_p, e and h.
  DISTANCE and MBH are passed in the same units as the driver.
- **M10 output.** No code change. `tools/raptor_h5_to_array.py` produces
  `M[iβ, iα]` and replaces the truncating `uniform_img` writer.
- **M11 exact LD85 step** (`POL_SOLVER_LD85`, diagnostic): the BLP
  constant-coefficient matrix exponential (`transfer.jl:34-108`), in place of
  the RK4/trapezoid step. It separates solver error from model differences.
- **M12 OpenMP over all pixels plus phase timers** (performance only): one
  `parallel for schedule(dynamic, 64)` over every pixel of the image. The
  load, trace and RT phases are timed with `omp_get_wtime`. The output is
  bitwise identical to the per-block loop, which R08 checks.
- **M13 (deferred, R11) OpenACC port:**
  - replace the function pointers with direct calls;
  - make `determ` and `rk45` iterative;
  - preallocate the per-ray buffers;
  - move the M7 A&S Bessels to the device, so GSL is no longer used on the device;
  - `#pragma acc parallel loop gang vector` over the pixels, with the snapshot in device memory (s455 4.8 GB, s093 about 35 GB: within one A100 80 GB).

## 5. Acceptance metrics

The metrics are those used in the EHT code comparisons, plus the ring
metrics of `plot_eht_bbh.py`:

- **Flux:** relative total-flux difference.
- **Image error:**
  - normalized L1 = Σ|I_R − I_B| / Σ I_B;
  - NMSE = Σ(I_R − I_B)² / Σ I_B², as in Gold+2020 and Prather+2023. There RAPTOR's NMSE against the exact solution is 7e−5 to 1.5e−4, and the worst GRMHD NMSE is 0.02 (I), 0.045 (LP) and 0.13 (V).
- **Polarization:** net m, ⟨|m|⟩, net v, EVPA_net = ½ arg ΣP, β₂.
- **Rings:** ring diameters and f_C at native resolution and after 20 µas blur.
- **Per-pixel error maps** of I, Q, U and V.

**Attribution rule.** For each observable X, measure each code's
self-convergence error ε_R and ε_B with C ∈ {40, 80, 160} and
resolution ∈ {240, 480, 960}, using Richardson estimates. A code-to-code
difference |X_R − X_B| ≤ 2 max(ε_R, ε_B) is discretisation. A larger
difference is a model or implementation difference: it is opened as an
error-ledger row and bisected with the unit legs (R03, R04, R06).

Quantitative thresholds per node are given in `dag.md`.

## 6. Blockers and risks

- **GPU.** No usable device (NVML error). The GPU timing legs (R11) are
  deferred. NVHPC is present, so the OpenACC port can be compiled and
  CPU-tested with `-acc=multicore`.
- **Tutorial data unreachable.** The R01 baseline therefore uses the
  synthetic BHAC snapshot. Running the upstream tutorial (data2000.dat,
  expected 3.63027 Jy) remains an [OPEN] check for when `astro.ru.nl` is
  reachable.
- **Cost.**
  - Upstream RAPTOR spends 18.4 ms of CPU per pixel on the synthetic model
    (single hole, polarized, STEPSIZE 0.01).
  - A 480² polarized image is 230,400 pixels, i.e. about 1.2 CPU-hours at
    that rate. The two-hole metric doubles the cost of each metric
    evaluation, so the estimate is about 2–3 CPU-hours, or a few minutes on
    64 threads.
  - M2b (analytic ∂g) removes the 16 metric evaluations per RHS call.
    Resolution-convergence legs at 960² cost 4× the 480² run.
- **Memory.** s093 needs about 35 GB with the M5 layout; the host has 503 GB.
- **Upstream quirks that the fork does not fix** (they are not on the matched
  path, and are recorded here for the report):
  - `uniform_img` pixel duplication;
  - the BHAC half-cell interpolation offset;
  - the Planck `exp − 1` cancellation;
  - the no-op `-g` flag in `setup.sh`;
  - the uninitialised `thetadot_prev`;
  - the CKS regularisers.

## 7. Node map

| node | modifications | convention rows closed |
|---|---|---|
| R01 | — (upstream build, synthetic baseline) | C11 |
| R03 | M7, M9 | C12, C24, C25 |
| R04 | test harness only | C3, C4, C7, C28, C30 (measured) |
| R05 | M1, M2a, M2b, M3, M4 | C1, C2, C3, C5–C8, C10, C30, C32, C33 |
| R02 | M8 | C9, C26 |
| R06 | M5, M6 | C14–C23, C34 |
| R07 | M11 (diagnostic leg) | C13, C27, C29, C31 |
| R08 | M12 | C35 |
| R09 | — | C32 (spins) |
| R10 | — | — |
| R11 | M13 (deferred) | — |
