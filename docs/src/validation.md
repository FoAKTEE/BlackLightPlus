# Validation against RAPTOR

BlackLightPlus images are cross-validated against RAPTOR
[Bronzwaer2018, Bronzwaer2020](@citep), an independent C code with its own
geodesic integrator, polarized transfer step and camera, on analytic scenes and
on binary-black-hole GRMHD snapshots. This page states the protocol, the
settings the two codes share, and the results. Every number is read from a
JSON file under `results/blp/raptor/evidence/`, written by the scripts in
`results/blp/raptor/tools/`; the full tables are in
`results/blp/raptor/report.md`.

## The RAPTOR build

RAPTOR is run from a fork (branch `blp-bbh` of upstream commit `b91e937`),
reproducible from the patch series in `results/blp/raptor/patches/`
(`results/blp/raptor/fork.md` gives the rebuild and build recipes). Upstream
RAPTOR has no binary metric, no AthenaK reader and a single-hole step control,
so the matched runs add those, each behind a compile-time switch that leaves the
upstream build unchanged (the upstream synthetic image stays bitwise identical,
`evidence/R05/r01_bitwise.json`). The fork's `BLP_CHANGES.md` lists every
modification with its switch, and every upstream defect found, fixed or
reported. In short:

| Change | Switch | Role |
|---|---|---|
| superposed Kerr–Schild binary metric, finite-difference or analytic connection | `metric == SKS`, `SKS_CONNECTION` | geometry |
| BlackLightPlus step control $h=\min_A(r_A-r_{h,A})/C$, termination and sample pairing | `STEP_BBH` | march |
| plane-parallel camera | `CAMERA_PP` | camera |
| AthenaK `.bin` / `.athdf` reader and the BlackLightPlus plasma decode | `athenak` model | plasma |
| BlackLightPlus thermal coefficients | `EMIS_BLP` | coefficients |
| IEEE/IAU camera readout | `STOKES_IAU` | Stokes convention |
| exact constant-coefficient polarized step | `POL_SOLVER_LD85` | leg (b) only |
| keep the transported Stokes $I$ of unpolarized pixels | `POL_KEEP_I` | readout |

Images are compared as HDF5 arrays converted to the BlackLightPlus layout
$M[i_\beta,i_\alpha]$ (`tools/raptor_h5_to_array.py`), never through RAPTOR's
ASCII writer.

## Matched settings

| Item | Both codes |
|---|---|
| Snapshots | AthenaK binary runs s455 (`.bin`, 512 blocks of $64^3$, non-spinning), s122 (`.athdf`, separation 16) and s093 (38.7 GB `.bin`, separation 25) |
| Spacetime | static superposed Kerr–Schild, $m_A=0.5$, hole centres from the dump time (s455: $\pm(12.19120680,-2.76124551,0)$) |
| Scaling | $M=1.3\times10^{10}\,M_\odot$ (each hole M87-like), $D=16.8$ Mpc, 230 GHz |
| Plasma | thermal, $R_{\rm low}=1$, $R_{\rm high}=40$, $\sigma>1$ cut, $\gamma$ from the dump, $n_e=\rho/m_p$, the calibrated mass scale of the BlackLightPlus figure record |
| Camera | plane-parallel at $d=1000\,M$, azimuth 0, FOV 56 $M$, $480^2$; $i=17^\circ,60^\circ,90^\circ$ |
| March | RK4, $h=\min_A(r_A-r_{h,A})/C$, $n_{\max}=50C$, capture at $d<10^{-3}$ or $E_n>100$, escape at $R>1100$ ([March termination](@ref)) |
| Transfer | polarized along the whole ray; thermal fits of [Marszewski2021](@citet) with `expm1` Kirchhoff and the $j_P\le j_I$, $\alpha_P\le\alpha_I$ rescales |

RAPTOR runs in two matched legs and two report-only legs:

- **(a) matched**: everything above, with RAPTOR's own polarized step;
- **(b) matched solver**: (a) with the exact constant-coefficient step of
  BlackLightPlus's segment operator (`POL_SOLVER_LD85`);
- **(c) native fits** (report only): (a) with RAPTOR's upstream emission fits;
- **stock** (report only): upstream fits, constants and plasma decode on the
  matched geometry, camera, march and readout.

## Metrics and the attribution rule

Metrics are those of the EHT code comparisons [Gold2020](@citep): relative total
flux $\Delta F/F$; normalized $L_1=\sum|I_R-I_B|/\sum I_B$ (also for $|P|$ and
$V$); ${\rm NMSE}=\sum(I_R-I_B)^2/\sum I_B^2$; net and mean linear polarization
$m_{\rm net}$, $\langle|m|\rangle$, net circular polarization $v_{\rm net}$ and
net EVPA on 20 µas-blurred images; ring diameters and central-depression
fractions. The thresholds for the binary images are $|\Delta F/F|\le0.01$,
$L_1(I)\le0.03$, ${\rm NMSE}(I)\le10^{-3}$, $|\Delta m|\le0.01$,
$|\Delta{\rm EVPA}_{\rm net}|\le2^\circ$, $|\Delta v_{\rm net}|\le0.002$, ring
diameters within 0.5 µas.

Each image is rendered by both codes on a ladder of step divisors
$C\in\{40,80,160,320\}$ (and 640 at $i=17^\circ$) and pixel counts
$\{240,480,960\}^2$. For an observable $X$ with self-convergence errors
$\varepsilon_R$, $\varepsilon_B$ (Richardson estimates from each code's own
ladder), a code-to-code difference $|X_R-X_B|\le2\max(\varepsilon_R,\varepsilon_B)$
is discretisation. A larger difference is a model or implementation difference
and is bisected with the unit comparisons (coefficients, geodesics, reader).

## Results

| Node | Comparison | Result | Evidence |
|---|---|---|---|
| R01 | upstream RAPTOR baseline | synthetic image flux $1.373950533\times10^{-6}$ Jy reproduced; HDF5 reassembly matches the spectrum to $1.8\times10^{-15}$ | `evidence/R01/baseline.json` |
| R03 | thermal coefficients, 40959 cells in $(\nu,B,\Theta_e,\theta_B)$ | ported coefficients within $2.3\times10^{-13}$ relative (libm round-off); upstream fits: $\alpha_I$ median ratio 0.993, $\alpha_Q$ outside $\pm10\%$ on 19% of cells ($\Theta_e\ge32$) | `evidence/R03/summary.json` |
| R04 | metric and geodesics (Kerr, s455, s122) | $\lvert g-g_{\rm BLP}\rvert\le4.4\times10^{-16}$; analytic connection $4.5\times10^{-15}$, finite-difference $6.1\times10^{-11}$ relative to AD; identical end classes and sample sets, both RK4 order 4 | `evidence/R04/rays_kerr_s455_s122.json` |
| R05 | camera | camera states equal `camera_rays` to $2.4\times10^{-15}$ at $480^2$ | `evidence/R05/camera_diff.json` |
| R02 | analytic scenes, $480^2$, $C=160$ | orientation slabs: every Stokes sign agrees ($B$ north: $Q<0$; north-west: $U>0$; toward the observer: $V>0$; Faraday rotation east); orientation ring within $7\times10^{-8}$; test problems T1–T5 of [Gold2020](@citet) at three inclinations: ${\rm NMSE}(I)\le2.5\times10^{-13}$, $L_1(I)\le1.8\times10^{-9}$ | `evidence/R02/analytic.json`, `evidence/R02/orientation.json` |
| R06 | AthenaK reader and plasma decode, $10^5$ points per snapshot | owning block, primitives, $u^\mu$, $b^\mu$, $n_e$, $B$, $\Theta_e$, $\sigma$, $\beta^{-1}$ and cut flags bit-identical on s455, s122 and s093 | `evidence/R06/s455_samples.json` |
| R07 | s455 images, leg (b) | every threshold met at every $C$ and inclination: $L_1(I)\le1.9\times10^{-7}$, $L_1(\lvert P\rvert)\le5.2\times10^{-6}$, $L_1(V)\le8.7\times10^{-6}$, identical end classes | `evidence/R07/metrics.json` |
| R07 | s455 images, leg (a) | flux within $5.2\times10^{-4}$, $L_1(I)\le6.7\times10^{-3}$; $\lvert P\rvert$ and $V$ differences fall with $C$ at RAPTOR's own self-convergence rate; net EVPA misses at $i=60^\circ,90^\circ$ attributed to discretisation | `evidence/R07/metrics.json`, `evidence/R07/polstep.json` |
| R09 | spinning snapshots s122, s093, leg (b) | $L_1$ of $I$, $\lvert P\rvert$, $V$ $\le1.1\times10^{-5}$ at $C\ge160$ ($\le1.3\times10^{-4}$ at $C=80$); leg (a) within every threshold; shadow edges and frame-dragging displacement identical | `evidence/R09/geometry.json`, `evidence/R09/s122_metrics.json`, `evidence/R09/s093_metrics.json` |
| R10 | $i=90^\circ$ adaptive-refinement deck | uniform $1024^2$, leg (b): $L_1(I)=1.1\times10^{-7}$, $L_1(\lvert P\rvert)=3.0\times10^{-6}$, $L_1(V)=1.6\times10^{-5}$; against the adaptive mosaic in its refined zone: $L_1(I)=4.3\times10^{-3}$, linear polarization 0.8369 in both | `evidence/R10/metrics.json` |

**Attribution.** With the exact step (leg b) the two codes march the same
states and differ only by the $O(h^4)$ truncation difference between RAPTOR's
Christoffel form $(x^\mu,k^\mu)$ and BlackLightPlus's Hamiltonian form
$(x^\mu,p_\mu)$ of the same RK4 march; the residual falls by about an order of
magnitude per doubling of $C$. With RAPTOR's native step (leg a) the $|P|$ and
$V$ differences are the discretisation error of that step: its stiffness test
selects the implicit trapezoid on every absorbing segment, which converges at
second order and whose per-segment phase error grows with Faraday depth (median
relative error 0.20 at Faraday depth 1–10 per segment). On s455 at
$i=60^\circ$ and $90^\circ$ the rays cross the cold, dense orbital plane with
Faraday depths of $10^2$–$10^4$ rad; $m_{\rm net}$ is a residual of cancelling
contributions and the net EVPA of either code moves by tens of degrees between
$C$ levels, so the 2° EVPA threshold is not resolvable there by either code at
$C\le320$.

**Report-only legs** ($C=80$). RAPTOR's upstream emission fits change the flux
by $-1.2\%$ to $-0.5\%$ and $m_{\rm net}$ by up to 0.010. The stock leg is
43–47% fainter, mostly because its decode takes
$\Theta_e=(u/\rho)\tfrac13(m_p/m_e)/(1+R)$, i.e. $(\gamma-1)=1/3$ regardless of the
dump's $\gamma$ ($0.75\times$ the pressure-based $\Theta_e$ for $\gamma=13/9$), plus
its $\Theta_e$ and radius cuts; the split among these terms is not measured.

## Timings

CPU only; best of 3 warm runs; host 2× Intel Xeon Platinum 8360Y (72 cores,
144 hardware threads), shared, with the load average recorded per leg in
`evidence/R08/timing.json`. RAPTOR with `OMP_PROC_BIND=close
OMP_PLACES=cores`; BlackLightPlus with `julia -t T`, compilation excluded. s455,
$i=17^\circ$, $480^2$, polarized, $C=80$:

| Run | Threads | Trace [s] | Transfer [s] | End to end [s] | Rays/s | Peak RSS [GB] |
|---|--:|--:|--:|--:|--:|--:|
| BlackLightPlus | 16 | 9.62 | 33.85 | 46.90 | 5300 | 24.4 |
| RAPTOR (a) | 16 | 66.90 | 109.03 | 177.18 | 1310 | 4.0 |
| RAPTOR (b) | 16 | 66.97 | 99.72 | 167.95 | 1382 | 4.0 |
| BlackLightPlus | 64 | 4.47 | 10.13 | 18.05 | 15787 | 24.2 |
| RAPTOR (a) | 64 | 19.00 | 31.03 | 51.31 | 4605 | 4.0 |
| RAPTOR (b) | 64 | 18.97 | 28.34 | 48.59 | 4869 | 4.0 |

At 64 threads BlackLightPlus traces 3.4× the rays per second of RAPTOR (a) and
is 2.8× faster end to end; RAPTOR scales better from 16 to 64 threads (88%
against 74% parallel efficiency). BlackLightPlus holds the image states of
every ray before transfer, hence its larger memory. On the $i=90^\circ$
adaptive-refinement deck at 64 threads, a uniform $1024^2$ image takes 238 s in
RAPTOR (b) and 102 s in BlackLightPlus, and the BlackLightPlus adaptive image
(effective $4096^2$, 573,696 rays) 83 s (`evidence/R10/metrics.json`).

## Caveats

- Neither code is re-calibrated: thresholds are evaluated at the mass scale of
  the BlackLightPlus figure record.
- $m_{\rm net}$, $\langle|m|\rangle$ and the net EVPA of s455 at $i=60^\circ$ and
  $90^\circ$ are not converged in $C$ or in resolution in either code; they are
  compared at equal discretisation only. The net linear polarization of s455 at
  $i=17^\circ$ changes by more than 0.01 between $240^2$, $480^2$ and $960^2$ in
  both codes.
- RAPTOR's upstream camera, step size and single-hole termination are not
  exercised; its native step control terminates only at a hole at the origin.
- RAPTOR has no image-plane refinement; the adaptive mosaic is compared in its
  refined zone rendered uniformly at the finest pixel.
- Timings are on a shared host at 16 and 64 threads. GPU legs are not run: no
  usable device on the host, and RAPTOR has no GPU build here.
- The spinning snapshots carry a spin above the extremal limit in their
  headers; both codes image them at $\chi=0.8$ ([Known input issues](@ref)).
