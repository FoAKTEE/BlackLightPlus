#!/bin/bash
# r07_raptor.sh — RAPTOR leg of the R07/R08 comparison (dag.md R07, R08): run a built
# RAPTOR (setup.sh -c=athenak -m=sks -r=pol -i=rk4) on s455 with the BlackLightPlus
# parameters of r07_blp.jl (M = 1.3e10 Msun, D = 16.8 Mpc, 230 GHz, R_low 1, R_high 40,
# the calibrated mass scale, FOV 56 M) and a bbh.in of the dump's binary geometry, for
# every (inclination, C, npix); nmax = 50 C. Each run directory
# <out>/<tag>_i<inc>_C<C>_n<npix>/ holds model.in, bbh.in, the RAPTOR output, the
# reassembled img.npz (BlackLightPlus constants) and timing.json (wall, max RSS from
# /usr/bin/time, and the RAPTOR_TIMERS line of the M12 phase timers when present).
# Run from the BlackLightPlus root.
# Usage: r07_raptor.sh <raptor_binary> <bbh.in template> <out> <incs> <Cs> <npixs> [threads] [tag] [repeats]
set -euo pipefail
BIN=$(readlink -f "${1:?RAPTOR binary}")
BBH=$(readlink -f "${2:?bbh.in template}")
OUT=${3:?output directory}
INCS=${4:?inclinations}
CS=${5:?step constants}
NPIXS=${6:?resolutions}
THREADS=${7:-64}
TAG=${8:-raptor}
REPEATS=${9:-1}
ROOT=$PWD
SNAP=${R07_SNAP:-/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin}
MS=${R07_MASS_SCALE:-2.274019497e28}
ulimit -s unlimited
for NPIX in ${NPIXS//,/ }; do
for INC in ${INCS//,/ }; do
for C in ${CS//,/ }; do
  D=$(printf "%s/%s_i%02d_C%d_n%d" "$OUT" "$TAG" "$INC" "$C" "$NPIX")
  rm -rf "$D"; mkdir -p "$D"
  cat > "$D/model.in" <<EOF
MBH		(Msun)	1.3e10
DISTANCE		(kpc) 16800
M_UNIT		(g)	$MS
R_LOW		(-)	1
R_HIGH		(-)	40
INCLINATION	(deg)           $INC
IMG_WIDTH	(pixels)	$NPIX
IMG_HEIGHT	(pixels)	$NPIX
CAM_SIZE_X	(Rg)		56
CAM_SIZE_Y	(Rg)		56
FREQS_PER_DEC	(-)		1
FREQ_MIN	(Hz)		230.e9
STEPSIZE	(-)		0.01
MAX_LEVEL	(-)		1
EOF
  sed -e "s/^C .*/C $C.0/" -e "s/^nmax .*/nmax $((50 * C))/" "$BBH" > "$D/bbh.in"
  WALLS=()
  for REP in $(seq 1 "$REPEATS"); do
    ( cd "$D" && RAPTOR_BBH=$PWD/bbh.in OMP_NUM_THREADS=$THREADS OMP_STACKSIZE=64m \
        /usr/bin/time -f "%e %M" -o time_$REP.txt "$BIN" model.in "$SNAP" 0 > run.log 2> run.err \
      && grep RAPTOR_TIMERS run.err > timers_$REP.txt || true )
    WALLS+=("$(cut -d' ' -f1 "$D/time_$REP.txt")")
  done
  python3 "$ROOT/results/blp/raptor/tools/raptor_h5_to_array.py" "$D/output/img_data_0.h5" 56 "$NPIX" 1.3e10 16800 \
      "$D/img.npz" --constants blp > "$D/reassemble.log"
  python3 - "$D" "$THREADS" "$REPEATS" <<'EOF'
import json, sys, os, re
d, threads, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
runs = []
for r in range(1, reps + 1):
    wall, rss = open(os.path.join(d, f"time_{r}.txt")).read().split()[-2:]
    runs.append({"wall_s": float(wall), "maxrss_GB": int(rss) / 2**20})
rec = {"threads": threads, "runs": runs, "wall_best_s": min(x["wall_s"] for x in runs),
       "maxrss_GB": max(x["maxrss_GB"] for x in runs), "loadavg_after": os.getloadavg()}
phases = []
for r in range(1, reps + 1):
    p = os.path.join(d, f"timers_{r}.txt")
    if os.path.isfile(p) and os.path.getsize(p):
        phases.append({k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", open(p).read())})
if phases:
    rec["phases_runs"] = phases
    rec["phases_best"] = {k: min(p[k] for p in phases) for k in phases[0]}
json.dump(rec, open(os.path.join(d, "timing.json"), "w"), indent=1)
EOF
  rm -f "$D/output/uniform_img_"*.dat
  printf "%s i=%s C=%s n=%s: %s | walls %s s\n" "$TAG" "$INC" "$C" "$NPIX" "$(head -1 "$D/reassemble.log")" "${WALLS[*]}"
done; done; done
