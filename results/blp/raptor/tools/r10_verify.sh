#!/bin/bash
# R10 verification (reduced re-check within the ledger CPU budget) of the i = 90
# AMR-figure deck: build RAPTOR legs a, b with 8-pixel camera blocks (r07_build.sh
# NPIX1D=8); render the deck with BlackLightPlus at <npix>^2 (default 64, r10_blp.jl) and
# with image-plane AMR at HEAD (figures/fig_bbh_i90_pol_amr.jl, base <npix>/2, max level 2,
# i.e. effective 2 <npix>); run RAPTOR legs a and b at <npix>^2 and leg b on the central
# 20 M zone at the finest AMR pixel (<npix>^2, fov 20); the polarized-readout test
# r10_polkeep.jl and the thick-segment transmission r10_trapezoid.jl (LD85 = exp(-tau)) on
# the fork's test drivers; r10_compare.py on the reduced run; check:
#   leg b against BlackLightPlus: L1 and NMSE of I, |P|, V <= 1e-5, identical classes;
#   leg a: flux within 1%;
#   RAPTOR (b) composite against the AMR mosaic block-averaged to <npix>^2: L1(I) <= 3%,
#   |dLP| <= 0.01;
#   the committed evidence/R10/metrics.json passes.
# Run from the BlackLightPlus root. Usage: r10_verify.sh [threads] [npix]
set -euo pipefail
T=$(dirname "$0")
EV=$(dirname "$T")/evidence/R10
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
N=${1:-16}
NPIX=${2:-64}
NPIX1D=8 bash "$T/r07_build.sh" "$TMP/bld" a,b > /dev/null
bash "$TMP/bld/src/tests/build_tests.sh" "$TMP/tb" > "$TMP/tb.log" 2>&1
julia --project=. "$T/r10_polkeep.jl" "$TMP/tb" 2>/dev/null | grep "pixels lit"
julia --project=. "$T/r10_trapezoid.jl" "$TMP/tb/libraptor_pol_step.so" | tail -2
julia --project=. -t "$N" "$T/r10_blp.jl" "$TMP/run" "$NPIX" 40 | grep "^n="
BLP_OUT=$TMP/run/amr BLP_BASE_RES=$((NPIX / 2)) BLP_MAXLEVEL=2 \
  julia --project=. -t "$N" figures/fig_bbh_i90_pol_amr.jl > "$TMP/amr.log" 2>&1
grep "^AMR" "$TMP/amr.log"
export R07_MASS_SCALE=1e24 R07_MBH=$(python3 -c "print(repr(6.5e9 * 1.989e33 / 1.98847e33))") \
       R07_RHIGH=10 R07_FREQ=2.3e11 R07_FOV=40 R07_NMAX=2500
bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 90 80 "$NPIX" "$N" rb
bash "$T/r07_raptor.sh" "$TMP/bld/a/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 90 80 "$NPIX" "$N" ra
R07_FOV=20 bash "$T/r07_raptor.sh" "$TMP/bld/b/RAPTOR" "$TMP/run/bbh.in" "$TMP/run" 90 80 "$NPIX" "$N" rbz
python3 "$T/r10_compare.py" "$TMP/run" "$TMP/metrics.json" --npix "$NPIX" --extra-npix "" > "$TMP/compare.log" || true
python3 - "$TMP/run" "$NPIX" "$EV/metrics.json" <<'EOF'
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "results/blp/raptor/tools"))
import r10_compare as r
rc = r.rc
run, n, mpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
B = r.load_blp_dir(os.path.join(run, f"blp_i90_C80_n{n}"))
rc.CENTERS = B["record"]["centers"]
Rb = rc.load_raptor(run, "rb", 90, 80, n)
Ra = rc.load_raptor(run, "ra", 90, 80, n)
m = r.simple(Rb, B)
worst = max(m["l1_I"], m["nmse_I"], m["l1_P"], m["l1_V"])
cls = float((Rb["class"] == B["class"]).mean())
check("leg b vs BlackLightPlus", worst <= 1e-5 and cls == 1.0, f"max norm {worst:.2e}, class agreement {cls}")
check("leg a flux within 1%", abs(r.simple(Ra, B)["flux_rel"]) <= 0.01, f"{r.simple(Ra, B)['flux_rel']:+.2e}")
red = json.load(open(os.path.join(run, "..", "metrics.json")))
ma = red["amr"]["rb_composite_vs_amr"]
check("RAPTOR (b) composite vs AMR block-averaged", ma["l1_I"] <= 0.03 and abs(ma["d_LP"]) <= 0.01,
      f"L1(I) {ma['l1_I']:.3e}, LP {ma['x']['LP']:.4f} vs {ma['ref']['LP']:.4f}")
met = json.load(open(mpath))
check("evidence R10 metrics.json pass", met["pass"] is True and met["amr"]["pass"] is True
      and met["legs"]["rb"]["pass"] and met["legs"]["ra"]["pass"])
print("R10 verify", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
