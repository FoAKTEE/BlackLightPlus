#!/bin/bash
# R05 verification on the fork head: camera states vs BlackLightPlus (480^2, i = 17/60/90),
# the R04 SKS legs, and the R01 synthetic image of the fork head bitwise identical to
# upstream b91e937 (new switches off). Run from the BlackLightPlus root.
# Usage: r05_verify.sh [outdir] [threads]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OUT=${1:-$TMP/ev}
N=${2:-32}
bash "$T/build_fork_tests.sh" "$TMP"
mkdir -p "$OUT"
julia --project=. "$T/r05_camera.jl" "$TMP/bin/libraptor_sks_fd.so" "$OUT/camera_diff.json"
julia --project=. -t "$N" "$T/r04_rays.jl" "$TMP/bin" "$OUT" s455,s122
bash "$T/r01_run.sh" b91e937 "$TMP/r01_main" "$N" > /dev/null
bash "$T/r01_run.sh" "${REF:-blp-bbh}" "$TMP/r01_head" "$N" > /dev/null
python3 "$T/r01_verify.py" "$TMP/r01_main" "$TMP/r01_head" > "$OUT/r01_bitwise.json"
cp "$TMP/fork_commit" "$OUT/fork_commit"
cat "$OUT/r01_bitwise.json" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d['pass'] else 1)"
