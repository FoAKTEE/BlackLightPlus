#!/bin/bash
# R04 verification: build the fork's test drivers (blp-bbh via git archive) and run the
# metric-level and ray-level comparisons. Exit 0 iff both pass (rays: acceptance_rule in
# the JSON). Run from the BlackLightPlus root. Usage: r04_verify.sh [outdir] [threads]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OUT=${1:-$TMP/ev}
bash "$T/build_fork_tests.sh" "$TMP"
mkdir -p "$OUT"
julia --project=. "$T/r04_metric.jl" "$TMP/bin" "$OUT/metric.json"
julia --project=. -t "${2:-32}" "$T/r04_rays.jl" "$TMP/bin" "$OUT" kerr,s455,s122
cp "$TMP/fork_commit" "$OUT/fork_commit"
