#!/bin/bash
# r08_timing.sh — R08 CPU timings (dag.md R08): RAPTOR (fork blp-bbh, legs a and b of
# r07_build.sh) and BlackLightPlus on the same host, same thread count, same image
# (s455, i = 17, 480^2, polarized, C = 80, nmax 4000), best of `repeats` warm runs
# (the snapshot is in the page cache after the first run; BlackLightPlus compiles in a
# 24^2 warm-up render that is excluded and reported separately). OpenMP threads are
# pinned (OMP_PROC_BIND=close, OMP_PLACES=cores); Julia uses `-t T`. The load average
# is recorded before and after each leg (the host is shared).
# Run from the BlackLightPlus root.
# Usage: r08_timing.sh <r07 build dir> <bbh.in> <outdir> [threads list] [repeats] [npix]
set -euo pipefail
BLD=${1:?r07_build.sh directory}
BBH=${2:?bbh.in}
OUT=${3:?output directory}
TLIST=${4:-16,64}
REP=${5:-3}
NPIX=${6:-480}
T=$(dirname "$0")
mkdir -p "$OUT"
export OMP_PROC_BIND=close OMP_PLACES=cores
for TH in ${TLIST//,/ }; do
  for LEG in a b; do
    echo "$LEG $TH before $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/loadavg.log"
    bash "$T/r07_raptor.sh" "$BLD/$LEG/RAPTOR" "$BBH" "$OUT" 17 80 "$NPIX" "$TH" "t${LEG}${TH}" "$REP"
    echo "$LEG $TH after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/loadavg.log"
  done
  echo "blp $TH before $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/loadavg.log"
  julia --project=. -t "$TH" "$T/r07_blp.jl" "$OUT/blp_t$TH" 17 80 "$NPIX" "$REP" 2>&1 | grep -v "Warning\|@ Black"
  echo "blp $TH after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/loadavg.log"
done
