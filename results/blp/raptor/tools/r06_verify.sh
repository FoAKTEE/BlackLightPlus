#!/bin/bash
# R06 verification on the committed fork branch (default blp-reader): export it with
# git archive, rebuild the AthenaK probe, write the BlackLightPlus reference at 1e5
# points on s455 (.bin) and s122 (.athdf) and require every r06_compare.py check to
# pass (owning block, primitives, u^mu/b^mu and plasma decode, flags, memory).
# Run from the BlackLightPlus root.
# Usage: r06_verify.sh [outdir] [threads]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OUT=${1:-$TMP/ev}
[ -n "$OUT" ] || OUT=$TMP/ev
N=${2:-16}
REF=${REF:-blp-reader}
mkdir -p "$TMP/fork" "$OUT"
git -C ref-code/jordydavelaar-raptor archive "$REF" | tar -x -C "$TMP/fork"
git -C ref-code/jordydavelaar-raptor rev-parse "$REF" > "$OUT/fork_commit"
for tag in s455 s122; do
    RAPTOR_READER=$TMP/fork R06_OUT=$TMP/run R06_EV=$OUT bash "$T/r06_run.sh" "$tag" "$N"
done
