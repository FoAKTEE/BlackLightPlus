#!/bin/bash
# R09 verification (reduced re-check within the ledger CPU budget):
#  1. geometry: the fork's test drivers (git archive blp-bbh, max_steps 4e5) against
#     BlackLightPlus: r04_metric.jl on s093, s122, s122r (reversed spins); r04_rays.jl on
#     s093; r09_geometry.jl (Kerr spin orientation against the photon orbits, binary shadow
#     edges, horizon null surface, g^tt >= 0 region);
#  2. s122 images at <npix>^2 (default 60) with both codes (r07_verify.py checks: leg b
#     L1/NMSE of I, |P|, V <= 2e-4 and identical classes at i = 17 (C = 80, 160), 60, 90
#     (C = 80), its L1(|P|) falling >= 4x from C = 80 to 160; leg a flux within 1% and its
#     polarization converging toward leg b) and the committed evidence/R09/s122_metrics.json.
#     2e-4 is above the largest leg (b) norm of the 480^2 evidence at C = 80 (1.3e-4, L1(V),
#     s122 i = 60): the spinning snapshots put more emission on the photon rings, where the
#     O(h^4) difference of the two geodesic forms is largest;
#  3. s093 at 40^2, i = 17, C = 80, leg b against BlackLightPlus (<= 2e-4), and the
#     committed evidence/R09/s093_metrics.json; no stored state of the re-run images has
#     g^tt >= 0; the committed evidence/R09/runs.json passes.
# Run from the BlackLightPlus root. Usage: r09_verify.sh [threads] [npix]
set -euo pipefail
T=$(dirname "$0")
EV=$(dirname "$T")/evidence/R09
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
N=${1:-16}
NPIX=${2:-60}
DATA=/data2/haiyangw/aurora/cbd_rel
mkdir -p "$TMP/tb/src"
git -C ref-code/jordydavelaar-raptor archive blp-bbh | tar -x -C "$TMP/tb/src"
sed -i 's/^#define max_steps (1e4)/#define max_steps (4e5)/' "$TMP/tb/src/model/blp_analytic/definitions.h"
bash "$TMP/tb/src/tests/build_tests.sh" "$TMP/tb/bin" > "$TMP/tb.log" 2>&1
julia --project=. "$T/r04_metric.jl" "$TMP/tb/bin" "$TMP/metric.json" s093,s122,s122r | tail -4
julia --project=. -t "$N" "$T/r04_rays.jl" "$TMP/tb/bin" "$TMP" s093 | grep -E "^\[s093\|fd\]|R04 rays"
julia --project=. -t "$N" "$T/r09_geometry.jl" "$TMP/tb/bin" "$TMP/geometry.json"
bash "$T/r07_build.sh" "$TMP/bld" a,b > /dev/null
export R07_SNAP=$DATA/torus.mhd_w_bcc.00122.athdf R07_MASS_SCALE=1.000498515e21
julia --project=. -t "$N" "$T/r07_blp.jl" "$TMP/run" 17 80,160 "$NPIX" 1 2>&1 | grep "^i="
julia --project=. -t "$N" "$T/r07_blp.jl" "$TMP/run" 60,90 80 "$NPIX" 1 2>&1 | grep "^i="
bash "$T/r07_raptor.sh" "$TMP/bld/a/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 17 80,160 "$NPIX" "$N" ra
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 17 80,160 "$NPIX" "$N" rb
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 60,90 80 "$NPIX" "$N" rb
python3 "$T/r07_verify.py" "$TMP/run" "$NPIX" "$EV/s122_metrics.json" 2e-4
export R07_SNAP=$DATA/torus.mhd_w_bcc.00093.bin R07_MASS_SCALE=2.252873787e20
julia --project=. -t "$N" "$T/r07_blp.jl" "$TMP/run93" 17 80 40 1 2>&1 | grep "^i="
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run93/bbh.in" "$TMP/run93" 17 80 40 "$N" rb
python3 - "$TMP/run" "$TMP/run93" "$NPIX" "$EV" <<'EOF'
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "results/blp/raptor/tools"))
import r07_compare as rc
run, run93, npix, ev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
B = rc.load_blp(run93, 17, 80, 40); rc.CENTERS = B["record"]["centers"]
R = rc.load_raptor(run93, "rb", 17, 80, 40)
m, _, _ = rc.pair_metrics(R, B, 17, 40)
w = max(m["l1_I"], m["l1_P"], m["l1_V"], m["nmse_I"], m["nmse_P"], m["nmse_V"])
check("s093 leg b i=17 C=80 n=40", w <= 2e-4 and m["class_agreement"] == 1.0, f"max norm {w:.2e}")
n = sum(json.load(open(p))["sample_safety"]["states_gtt_nonneg"]
        for p in glob.glob(os.path.join(run, "blp_*", "record.json")) + glob.glob(os.path.join(run93, "blp_*", "record.json")))
check("re-run images: no stored state with g^tt >= 0", n == 0, f"({n})")
met = json.load(open(os.path.join(ev, "s093_metrics.json")))
for inc, e in met["legs"]["rb"].items():
    check(f"evidence s093 leg b i={inc}", all(s["pass"] for s in e["evaluated"].values()))
for inc, e in met["legs"]["ra"].items():
    check(f"evidence s093 leg a i={inc} no model-attributed miss",
          all(not s["model_attributed"] for s in e["evaluated"].values()))
for f in ("geometry.json", "runs.json"):
    check(f"evidence {f} pass", json.load(open(os.path.join(ev, f)))["pass"] is True)
print("R09 verify", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
