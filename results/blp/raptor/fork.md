# RAPTOR fork `blp-bbh`: layout and reproduction

## Repository

| item | value |
|---|---|
| upstream | https://github.com/jordydavelaar/raptor |
| pinned upstream commit | `b91e937b7b746eb24992b6c1dd8831c5dea0cef1` (main, 2024-08-12) |
| working copy | `ref-code/jordydavelaar-raptor` (full upstream history, remote `origin`) |
| pristine mirror | branch `main` at `b91e937`, never edited |
| fork branch | `blp-bbh`, branched from `b91e937`; every modification is one commit on it |
| provenance | `ref-code/jordydavelaar-raptor/PROVENANCE.md` (listed in `.git/info/exclude`, so it never enters the fork history) |
| patch series | `results/blp/raptor/patches/NNNN-*.patch` plus `patches/SERIES` (base, head, one line per commit, sha256 per patch) |

`ref-code/` is ignored by the BlackLightPlus repository. The patch series is
therefore the tracked record of the fork. After every commit on `blp-bbh`, run
`results/blp/raptor/tools/export_patches.sh` from the BlackLightPlus root, then
commit `results/blp/raptor/patches/` together with the ledger row of the node.

Commits on `blp-bbh` use the same title grammar as BlackLightPlus
(`<type>(<scope>): <summary>`). Scopes follow the modification ids of
`design.md` §4 (`m1-metric`, `m5-athenak`, ...). Each commit names its
DAG node in the body (`- next:`/`- change:` objects).

## Rebuilding the fork from the series

```bash
git clone https://github.com/jordydavelaar/raptor raptor-blp
cd raptor-blp
git checkout -b blp-bbh b91e937b7b746eb24992b6c1dd8831c5dea0cef1
P=/path/to/BlackLightPlus/results/blp/raptor/patches
# --keep-cr: src/GRmath.c has CRLF endings; the blp-reader series goes after 0016, where
# blp-bbh merges it (0017+ edit model/athenak)
git am -3 --keep-cr $(ls $P/00*.patch | head -16) $P/blp-reader/*.patch $(ls $P/00*.patch | tail -n +17)
git rev-parse 'HEAD^{tree}'   # equals `tree` in patches/SERIES
```

`format-patch --zero-commit` keeps the patches free of local commit hashes.
`git am` recreates the commits with the original author, date and message, so
the rebuilt tree is identical to `blp-bbh`. The commit hashes differ only when
the committer identity differs.

## Build recipes

The run directory holds the compile-time configuration. RAPTOR copies
`run/*` and `model/<code>/*` into it and edits `definitions.h` there
(`setup.sh:42-160`). Run directories live under `out/raptor/<node>/`, which
BlackLightPlus ignores.

```bash
export RAPTOR=/data/haiyangw/claude/BlackLightPlus/ref-code/jordydavelaar-raptor
mkdir -p out/raptor/R01_build && cd out/raptor/R01_build
bash $RAPTOR/setup.sh -c=bhac -m=cks -r=pol -i=rk4      # gcc 11.4 via h5cc (HDF5 1.10.7), GSL 2.7.1, OpenMP
python3 ../../../results/blp/raptor/tools/make_bhac_synthetic.py synthetic.dat 0.9375   # also writes grid_cks.in
OMP_NUM_THREADS=16 OMP_STACKSIZE=20m ./RAPTOR model.in synthetic.dat 0
python3 ../../../results/blp/raptor/tools/raptor_h5_to_array.py output/img_data_0.h5 40 100 6.5e9 16800
```

The same sources also build with NVHPC `nvc` 25.9 (`-mp`, CPU OpenMP). This
build is the starting point of the deferred OpenACC leg (`dag.md` R11):

```bash
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin
SRC=$RAPTOR/src
$NV/nvc -mp -O2 -c99 -I$SRC -I. -I/usr/include/hdf5/serial \
  $SRC/{main,core,io,GRmath,gr_integrator,rte_integrator,pol_rte_integrator,metric,pol_emission,tetrad,constants,camera}.c model.c \
  -o RAPTOR_nvc -lgsl -lgslcblas -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lm
```

## Compile-time switches and the analytic model

The modifications are guarded by switches whose defaults (`src/functions.h`)
keep upstream behaviour; `model/blp_analytic/definitions.h` turns them on.

| switch | default | blp_analytic | modification |
|---|---|---|---|
| `metric == SKS` | — | on | M1 superposed Kerr–Schild binary from `bbh.in` (`$RAPTOR_BBH`) |
| `SKS_CONNECTION` | `SKS_CONN_FD` | FD | M2a finite differences, `SKS_CONN_ANALYTIC` = M2b |
| `STEP_BBH` | 0 | 1 | M3 march control, termination and transfer sample set |
| `CAMERA_PP` | 0 | 1 | M4 plane-parallel camera at `cam_dist` |
| `EMIS_BLP` | 0 | 1 | M7 BlackLightPlus thermal coefficients, with the j_P ≤ j_I and α_P ≤ α_I rescales of `stokes_segment` |
| `STOKES_IAU` | 0 | 1 | M8 camera Stokes readout: normal observer, up = image north, basis (north, east) |
| `EMIS_MODEL` | 0 | 1 | coefficients supplied by the model (`model_evaluate_coeffs`, the Gold et al. 2020 emitter) |
| `POL_SOLVER_LD85` | 0 | 0 | M11 exact constant-coefficient polarized step (`pol_rte_ld85_step`, the BlackLightPlus `stokes_segment` + `stokes_admissible`) in place of `check_stiffness` → RK4/trapezoid |
| `ATHK_UPSTREAM_PLASMA` | 0 | — | AthenaK model only (`model_definitions.h`): the `model/bhac` plasma decode (R07 stock leg) |

`model/blp_analytic/definitions.h` also carries the BlackLightPlus constants (M9).
`model/blp_analytic/model.c` holds the analytic scenes of `$RAPTOR_SCENE`
(default `scene.in`): `slab` (thin slab and Faraday screen of
`test/test_ray_orientation.jl`), `ring` (the orientation-check ring of
`fig_eht_bbh.jl`) and `gold` (Gold et al. 2020 T1–T5); thermal scenes decode
BlackLightPlus primitives exactly as `primitive_four_vectors` and `plasma_decode`.

`bbh.in` holds `nholes`, `mass`, `spin`, `center1`, `center2`, `C`, `nmax`,
`escape`, `horizon_eps`, `capture_redshift` and `cam_dist`.

Upstream fix on the matched path: the unpolarized transfer adds `j ds` on
absorption-free segments (upstream dropped them).

M12 (always on without AMR): `calculate_image_all` runs every pixel of every block in one
OpenMP loop (`schedule(dynamic, 64)`); `main` prints `RAPTOR_TIMERS` (load, setup, loop,
trace, transfer, output walls; thread-summed geodesic and transfer times; rays; stored
states); under `STEP_BBH` the HDF5 image carries `endclass` and `nsamples` per pixel.

Upstream fix on the native path: `rho_Q_thermal` takes K₁/K₂ → 1 when K₂(1/Θ_e)
underflows (Θ_e < 1.4e−3), as `rho_V_thermal` does for K₀/K₂; before, the NaN blanked the
pixel.

## Unit-test drivers

`tests/build_tests.sh <dir>` builds the drivers against `model/blp_analytic`:
`coeff_table` (R03), `metric_dump` and `metric_dump_cks` (R04 metric level),
`ray_dump` and the `ccall` libraries `libraptor_sks_fd.so`,
`libraptor_sks_an.so`, `libraptor_cks.so` (R04 rays, R05 camera), and
`libraptor_rt_pol.so`, `libraptor_rt_unpol.so` (one image point through the
camera, march and transfer; R02), `libraptor_pol_step.so` (one polarized segment, native
and M11 step; R07) and `native_cold` (finiteness of the upstream fits for cold electrons). The
BlackLightPlus side lives in `results/blp/raptor/tools/`: `r03_verify.sh`,
`r04_verify.sh`, `r05_verify.sh`, `r02_verify.sh` and `r01_verify.sh` export the fork branch with
`git archive`, build it and run the comparisons.

## Branch `blp-reader`: AthenaK model (M5, M6, R06)

`blp-reader` branches from `b91e937` and holds the AthenaK data layer; its
series is `patches/blp-reader/` (`BRANCH=blp-reader
OUT=results/blp/raptor/patches/blp-reader tools/export_patches.sh`). It merges
onto `blp-bbh` without conflicts.

| file | role |
|---|---|
| `model/athenak/athenak_mesh.[ch]` | M5 reader: `.bin` v1.1 and `.athdf` (`mhd_w_bcc`) into Float32 `[block][prim][k][j][i]` (dens, velx, vely, velz, eint, bcc1–3); eint = Float32(press/(γ − 1)) with the dump's `<mhd>/gamma`; owning block and clamped cell-centre trilinear sample by the BlackLightPlus `meshblock_index` / `sample_primitives` rules |
| `model/athenak/athenak_plasma.[ch]` | M6: `athk_four_vectors` (AthenaK b⁰ = u_i B^i) and `athk_plasma_decode` (n_e, Θ_e, \|B\|, σ, β⁻¹, σ > 1 cut, vacuum gates), the ports of `primitive_four_vectors` and `plasma_decode` |
| `model/athenak/model.c` | RAPTOR model: `init_model`, `set_units`, `get_fluid_params` on the compiled metric (CKS or SKS) |
| `model/athenak/tests/athenak_probe.c` | data-layer probe on a points file (R06) |
| `model/athenak/tests/fluid_probe.c` | `init_model` + `get_fluid_params` on the SKS metric (R06 end-to-end leg; needs M1) |
| `setup.sh -c=athenak`, `run/makefile` | the run directory takes `model/blp_analytic/definitions.h` when present (else bhac's), sets `RT_OUTER_CUTOFF` 1100 and `EMIS_MODEL` 0; `-m=sks`; `model.mk` adds the model sources |

The emission side consumes the data layer through three calls per sample:

```c
int b = athk_sample(&athk, x, y, z, prim);              // owning block, 8 primitives
athk_four_vectors(prim, g_dd, g_uu, &fv);               // u^mu, u_mu, b^mu, b_mu, b^2
int emit = athk_plasma_decode(&athk_par, prim[ATHK_DENS],
                              athk_pgas(prim[ATHK_EINT], athk.gamma), fv.b_sq, &pl);
// pl.n_e_cgs [cm^-3], pl.theta_e, pl.bb_cgs [G]; emit == 0 is vacuum
```

Run directory and image (after the merge onto `blp-bbh`, with `bbh.in` from the
snapshot's binary geometry):

```bash
export RAPTOR=/data/haiyangw/claude/BlackLightPlus/ref-code/jordydavelaar-raptor
bash $RAPTOR/setup.sh -c=athenak -m=sks -r=pol -i=rk4
OMP_NUM_THREADS=64 ./RAPTOR model.in /data2/.../torus.mhd_w_bcc.00455.bin 0
python3 results/blp/raptor/tools/raptor_h5_to_array.py output/img_data_0.h5 56 480 1.3e10 16800 img.npz --constants blp
```

R06 reproduction (`dag.md` R06): `results/blp/raptor/tools/r06_run.sh TAG
[threads]` writes the BlackLightPlus reference (`r06_reference.jl`), runs the
probe and compares (`r06_compare.py`) into `evidence/R06/TAG_samples.json`;
`RAPTOR_SKS=<blp-reader merged onto blp-bbh>` adds the end-to-end leg.
`r06_verify.sh` re-runs s455 and s122 on the committed `blp-reader`.

## R07/R08 image runs

`tools/r07_build.sh <dir> [a,b,c,stock]` exports the fork and builds the R07 legs (athenak,
SKS, pol, RK4; `max_steps` 4e4): (a) matched, (b) + `POL_SOLVER_LD85`, (c) `EMIS_BLP 0`,
stock (upstream fits, constants, plasma decode, `RT_OUTER_CUTOFF` 50).
`tools/r07_blp.jl` renders the BlackLightPlus images and writes `bbh.in`;
`tools/r07_raptor.sh` runs a leg over inclinations, C and resolutions (nmax = 50 C);
`tools/r07_compare.py` writes `evidence/R07/metrics.json`; `tools/r07_figures.py` the
figures; `tools/r08_timing.sh` and `tools/r08_collect.py` the timings
(`evidence/R08/timing.json`); `tools/r08_report.py` renders `report.md`.
`tools/r07_verify.sh` and `tools/r08_verify.sh` are the ledger re-checks.
