#!/bin/bash
# R07 verification (reduced re-check of evidence/R07/metrics.json within the ledger CPU
# budget): build the fork legs a and b (r07_build.sh), re-run the segment test of the M11
# step (r07_polstep.jl), render s455 at <npix>^2 (default 60) with both codes at
# i = 17 (C = 80, 160) and i = 60, 90 (C = 80), and check with r07_verify.py:
#   leg (b) against BlackLightPlus: L1 and NMSE of I, |P|, V <= 1e-5 (the 480^2 evidence
#   reaches 8.7e-6 in L1(V) at C = 40: RK4 truncation, Christoffel vs Hamiltonian form), class agreement 1;
#   leg (a): its net Q, U move toward leg (b) from C = 80 to C = 160 (the native-step
#   discretisation attributed in metrics.json), I within 1% of BlackLightPlus;
#   the committed evidence/R07/metrics.json passes (leg b at 480^2, attribution of leg a).
# Run from the BlackLightPlus root. Usage: r07_verify.sh [threads] [npix]
set -euo pipefail
T=$(dirname "$0")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
N=${1:-8}
NPIX=${2:-60}
bash "$T/r07_build.sh" "$TMP/bld" a,b > /dev/null
bash "$TMP/bld/src/tests/build_tests.sh" "$TMP/tb" > "$TMP/tb.log" 2>&1
julia --project=. "$T/r07_polstep.jl" "$TMP/tb/libraptor_pol_step.so" "$TMP/polstep.json" 4000
julia --project=. -t "$N" "$T/r07_blp.jl" "$TMP/run" 17 80,160 "$NPIX" 1 2>&1 | grep "^i="
julia --project=. -t "$N" "$T/r07_blp.jl" "$TMP/run" 60,90 80 "$NPIX" 1 2>&1 | grep "^i="
bash "$T/r07_raptor.sh" "$TMP/bld/a/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 17 80,160 "$NPIX" "$N" ra
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 17 80,160 "$NPIX" "$N" rb
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 60,90 80 "$NPIX" "$N" rb
python3 "$T/r07_verify.py" "$TMP/run" "$NPIX" results/blp/raptor/evidence/R07/metrics.json
