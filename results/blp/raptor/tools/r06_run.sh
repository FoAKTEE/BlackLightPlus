#!/bin/bash
# R06 reader acceptance for one snapshot: build the RAPTOR AthenaK probe, write the
# BlackLightPlus reference at 1e5 points, run the probe at the same points and compare.
# Run from the BlackLightPlus root:
#   results/blp/raptor/tools/r06_run.sh s455|s122|s093 [THREADS]
# Scratch goes to out/raptor/R06/ (R06_OUT); the comparison JSON to
# results/blp/raptor/evidence/R06/<tag>_samples.json (R06_EV). RAPTOR_READER is the
# fork tree with model/athenak (default the blp-reader worktree).
set -euo pipefail
TAG=$1
NT=${2:-16}
RAPTOR=${RAPTOR_READER:-ref-code/raptor-reader}
OUT=${R06_OUT:-out/raptor/R06}
EV=${R06_EV:-results/blp/raptor/evidence/R06}
mkdir -p "$OUT" "$EV"
SRC=$(cd "$RAPTOR/model/athenak" && pwd)
(cd "$OUT" && h5cc -std=c99 -O2 -fopenmp -Wall -Wextra -I"$SRC" "$SRC/tests/athenak_probe.c" \
    "$SRC/athenak_mesh.c" "$SRC/athenak_plasma.c" -o athenak_probe -lm)
julia --project=. -t "$NT" results/blp/raptor/tools/r06_reference.jl "$TAG" "$OUT"
read -r SNAP MBH MS RH < <(python3 -c "
import json; m = json.load(open('$OUT/${TAG}_meta.json'))
print(m['file'], repr(m['m_bh_g']), repr(m['mass_scale_g']), repr(m['R_high']))")
OMP_NUM_THREADS=$NT "$OUT/athenak_probe" "$SNAP" "$OUT/${TAG}_points.bin" \
    "$OUT/${TAG}_raptor.bin" --mbh "$MBH" --ms "$MS" --rhigh "$RH" > "$OUT/${TAG}_probe.json"
# end-to-end leg: init_model + get_fluid_params with RAPTOR's SKS metric (M1), when
# RAPTOR_SKS names a fork tree that has it (src/bbh_metric.c)
if [ -n "${RAPTOR_SKS:-}" ] && [ -f "$RAPTOR_SKS/src/bbh_metric.c" ]; then
    B=$(cd "$OUT" && pwd)/sks_build
    rm -rf "$B"; mkdir -p "$B"
    (cd "$B" && RAPTOR=$RAPTOR_SKS bash "$RAPTOR_SKS/setup.sh" -c=athenak -m=sks > setup.log 2>&1)
    R=$RAPTOR_SKS/src
    (cd "$B" && h5cc -std=gnu99 -O2 -fopenmp -I"$R" -I. "$RAPTOR_SKS/model/athenak/tests/fluid_probe.c" \
        "$R"/{core,io,GRmath,gr_integrator,rte_integrator,pol_rte_integrator,metric,pol_emission,tetrad,constants,camera,blp_coeffs,bbh_metric}.c \
        model.c athenak_mesh.c athenak_plasma.c -o fluid_probe -lgsl -lgslcblas -lm 2> build.log)
    python3 -c "
import json; m = json.load(open('$OUT/${TAG}_meta.json'))
c1, c2 = m['centers']; a1, a2 = m['a_used']
print('nholes 2\nmass 0.5 0.5\nspin %r %r\ncenter1 %r %r %r\ncenter2 %r %r %r' % (a1, a2, *c1, *c2))
" > "$OUT/${TAG}_bbh.in"
    OMP_NUM_THREADS=$NT "$B/fluid_probe" "$SNAP" "$OUT/${TAG}_points.bin" "$OUT/${TAG}_fluid.bin" \
        "$OUT/${TAG}_bbh.in" "$MBH" "$MS" "$RH" 2> "$OUT/${TAG}_fluid.log"
    # the tree's commit and its parents (a trial merge of blp-reader onto blp-bbh)
    git -C "$RAPTOR_SKS" log -1 --format='%H parents %P' > "$OUT/${TAG}_fluid.sha"
fi
python3 results/blp/raptor/tools/r06_compare.py "$OUT" "$TAG" "$EV/${TAG}_samples.json"
