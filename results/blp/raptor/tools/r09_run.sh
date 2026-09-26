#!/bin/bash
# r09_run.sh — R09 image legs on one spinning snapshot (dag.md R09): BlackLightPlus
# (r07_blp.jl) and RAPTOR legs (a) native polarized step and (b) LD85 step (r07_raptor.sh),
# i = 17, 60, 90 over the step constants CS at 480^2, and i = 17, C = 80 at 240^2 and 960^2;
# then r07_compare.py into <out>/metrics.json. The two codes run one after the other (s093
# needs ~35 GB per code for the snapshot alone).
# Run from the BlackLightPlus root.
# Usage: r09_run.sh <tag> <snapshot> <mass_scale> <build_dir> <out> [threads] [CS]
set -euo pipefail
TAG=${1:?tag}
export R07_SNAP=${2:?snapshot}
export R07_MASS_SCALE=${3:?mass scale}
BLD=${4:?build dir with a/RAPTOR, b/RAPTOR}
OUT=${5:?output directory}
N=${6:-64}
CS=${7:-40,80,160,320}
T=$(dirname "$0")
mkdir -p "$OUT"
echo "[$TAG] BlackLightPlus $(date -Is)"
julia --project=. -t "$N" "$T/r07_blp.jl" "$OUT" 17,60,90 "$CS" 480 1 2>&1 | tee "$OUT/blp.log" | grep -E "^(BLP|i=)"
julia --project=. -t "$N" "$T/r07_blp.jl" "$OUT" 17 80 240,960 1 2>&1 | tee "$OUT/blp_res.log" | grep -E "^i="
for LEG in a b; do
  echo "[$TAG] RAPTOR leg $LEG $(date -Is)"
  bash "$T/r07_raptor.sh" "$BLD/$LEG/RAPTOR" "$OUT/bbh.in" "$OUT" 17,60,90 "$CS" 480 "$N" "r$LEG"
  bash "$T/r07_raptor.sh" "$BLD/$LEG/RAPTOR" "$OUT/bbh.in" "$OUT" 17 80 240,960 "$N" "r$LEG"
done
python3 "$T/r07_compare.py" "$OUT" "$OUT/metrics.json" --report-legs "" | tee "$OUT/compare.log"
