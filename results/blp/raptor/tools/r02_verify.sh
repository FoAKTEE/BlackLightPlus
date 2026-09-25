#!/bin/bash
# R02 verification: build the fork (blp-bbh via git archive), run the orientation tests
# (slabs + orientation-check ring), the RAPTOR executable-vs-library pipeline check, and
# the analytic image comparison at <npix> (default 120) and C = 160.
# Run from the BlackLightPlus root. Usage: r02_verify.sh [outdir] [threads] [npix]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
OUT=${1:-$TMP/ev}
N=${2:-32}
NPIX=${3:-120}
bash "$T/build_fork_tests.sh" "$TMP"
mkdir -p "$OUT"
julia --project=. -t "$N" "$T/r02_orientation.jl" "$TMP/bin/libraptor_rt_pol.so" "$OUT/orientation.json" | grep -v "^Using model"
julia --project=. -t "$N" "$T/r02_main_check.jl" "$TMP/src" "$TMP/bin" "$OUT/pipeline.json" | grep -v "^Using model"
julia --project=. -t "$N" "$T/r02_images.jl" "$TMP/bin" "$OUT/analytic_${NPIX}.json" "$NPIX" 160
cp "$TMP/fork_commit" "$OUT/fork_commit"
