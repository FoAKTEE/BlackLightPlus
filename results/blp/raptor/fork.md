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
git am /path/to/BlackLightPlus/results/blp/raptor/patches/*.patch
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
| `EMIS_BLP` | 0 | 1 | M7 BlackLightPlus thermal coefficients |

`model/blp_analytic/definitions.h` also carries the BlackLightPlus constants (M9).
`bbh.in` holds `nholes`, `mass`, `spin`, `center1`, `center2`, `C`, `nmax`,
`escape`, `horizon_eps`, `capture_redshift` and `cam_dist`.

## Unit-test drivers

`tests/build_tests.sh <dir>` builds the drivers against `model/blp_analytic`:
`coeff_table` (R03), `metric_dump` and `metric_dump_cks` (R04 metric level),
`ray_dump` and the `ccall` libraries `libraptor_sks_fd.so`,
`libraptor_sks_an.so`, `libraptor_cks.so` (R04 rays, R05 camera). The
BlackLightPlus side lives in `results/blp/raptor/tools/`: `r03_verify.sh`,
`r04_verify.sh`, `r05_verify.sh` and `r01_verify.sh` export the fork branch with
`git archive`, build it and run the comparisons.
