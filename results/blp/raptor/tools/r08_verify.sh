#!/bin/bash
# R08 verification: (1) report.md is exactly the rendering of the committed evidence
# (every number traces to evidence/R07, evidence/R08 and patches/SERIES); (2) the timing
# evidence covers RAPTOR and BlackLightPlus at every recorded thread count with all five
# phases; (3) M12: the R01 synthetic image of the fork head is bitwise identical to
# upstream b91e937; (4) the figures exist.
# Run from the BlackLightPlus root. Usage: r08_verify.sh [threads]
set -euo pipefail
T=$(dirname "$0")
R=$(dirname "$T")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
N=${1:-16}
python3 "$T/r08_report.py" "$TMP/report.md" > /dev/null
diff -q "$TMP/report.md" "$R/report.md"
echo "PASS report.md renders from the evidence"
python3 - "$R/evidence/R08/timing.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["runs"]
th = sorted({v["threads"] for v in t.values()})
ok = True
for n in th:
    codes = {v["code"] for v in t.values() if v["threads"] == n}
    ok &= codes == {"RAPTOR", "BlackLightPlus"}
for v in t.values():
    ok &= all(v["phases_s"][p] >= 0 for p in ("load", "setup", "trace", "transfer", "output"))
    ok &= v["rays"] == 230400 and v["rays_per_s"] > 0
print(("PASS" if ok else "FAIL"), "timing covers both codes at threads", th)
sys.exit(0 if ok else 1)
EOF
bash "$T/r01_run.sh" b91e937 "$TMP/r01_main" "$N" > /dev/null
bash "$T/r01_run.sh" "${REF:-blp-bbh}" "$TMP/r01_head" "$N" > /dev/null
python3 "$T/r01_verify.py" "$TMP/r01_main" "$TMP/r01_head" > "$TMP/r01.json"
python3 -c "import json,sys; d=json.load(open('$TMP/r01.json')); print('PASS' if d['pass'] else 'FAIL', 'R01 bitwise with M12'); sys.exit(0 if d['pass'] else 1)"
for f in images_i17 images_i60 images_i90 convergence timing; do test -s "$R/figures/$f.png"; done
echo "PASS figures present"
