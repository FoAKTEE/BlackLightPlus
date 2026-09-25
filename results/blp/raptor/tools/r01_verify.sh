#!/bin/bash
# R01 verification: build upstream RAPTOR b91e937 (bhac, CKS, pol, RK4; gcc/h5cc), run the
# synthetic BHAC baseline and check the flux (1.37395e-06 Jy) and the HDF5 reassembly.
# Run from the BlackLightPlus root. Usage: r01_verify.sh [out.json] [threads]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
bash "$T/r01_run.sh" b91e937 "$TMP/r01" "${2:-16}" > /dev/null
python3 "$T/r01_verify.py" "$TMP/r01" > "$TMP/r01.json" || { cat "$TMP/r01.json"; exit 1; }
cat "$TMP/r01.json"
[ -n "${1:-}" ] && cp "$TMP/r01.json" "$1" || true
