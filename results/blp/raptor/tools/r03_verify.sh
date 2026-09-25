#!/bin/bash
# R03 verification: build tests/coeff_table from the fork branch blp-bbh (exported with
# git archive, independent of the working-tree checkout), print the R03 table and
# compare it with BlackLightPlus. Exit 0 iff EMIS_BLP matches to <= 1e-10.
# Run from the BlackLightPlus root. Usage: r03_verify.sh [outdir]  (default: a temp dir)
set -euo pipefail
RAPTOR=${RAPTOR:-ref-code/jordydavelaar-raptor}
REF=${REF:-blp-bbh}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git -C "$RAPTOR" archive "$REF" | tar -x -C "$TMP"
bash "$TMP/tests/build_tests.sh" "$TMP/bin" >/dev/null
"$TMP/bin/coeff_table" > "$TMP/table.csv"
julia --project=. results/blp/raptor/tools/r03_coeffs.jl "$TMP/table.csv" "${1:-$TMP/ev}"
