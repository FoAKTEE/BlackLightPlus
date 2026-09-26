"""r07_verify.py — checks of the reduced R07 re-run (r07_verify.sh) and of the committed
evidence/R07/metrics.json. Usage: python3 r07_verify.py <rundir> <npix> <metrics.json>
Optional 4th argument: the leg (b) tolerance (default 1e-5, the s455 value); R09 passes the
spinning-snapshot value and then also requires leg (b)'s L1(|P|) at i = 17 to fall by >= 4x
from C = 80 to C = 160 (the RK4 truncation difference of the two geodesic forms).
Exit 0 iff every check passes; prints one line per check."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_compare as rc  # noqa: E402

run, npix, mpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
TOL = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


rc.CENTERS = rc.load_blp(run, 17, 80, npix)["record"]["centers"]
l1p_b = {}
for inc, Cs in ((17, (80, 160)), (60, (80,)), (90, (80,))):
    for C in Cs:
        B = rc.load_blp(run, inc, C, npix)
        R = rc.load_raptor(run, "rb", inc, C, npix)
        m, _, _ = rc.pair_metrics(R, B, inc, npix)
        worst = max(m["l1_I"], m["l1_P"], m["l1_V"], m["nmse_I"], m["nmse_P"], m["nmse_V"])
        if inc == 17:
            l1p_b[C] = m["l1_P"]
        check(f"leg b i={inc} C={C}", worst <= TOL and m["class_agreement"] == 1.0,
              f"max(L1, NMSE of I, |P|, V) {worst:.2e}, class agreement {m['class_agreement']}")
if len(sys.argv) > 4:
    check("leg b L1(|P|) falls with C (i=17)", l1p_b[160] <= l1p_b[80] / 4, f"C=80 {l1p_b[80]:.2e}, C=160 {l1p_b[160]:.2e}")
# leg a: native-step discretisation shrinks toward leg b with C
d = []
for C in (80, 160):
    A = rc.load_raptor(run, "ra", 17, C, npix)
    Bb = rc.load_raptor(run, "rb", 17, C, npix)
    d.append(np.abs((A["Q"] + 1j * A["U"]) - (Bb["Q"] + 1j * Bb["U"])).sum() / np.abs(Bb["Q"] + 1j * Bb["U"]).sum())
    if C == 80:
        B = rc.load_blp(run, 17, C, npix)
        check("leg a flux within 1% of BlackLightPlus (i=17, C=80)", abs(A["I"].sum() / B["I"].sum() - 1) <= 0.01,
              f"{A['I'].sum() / B['I'].sum() - 1:+.2e}")
check("leg a polarization converges toward the LD85 step", d[1] < d[0], f"L1(P_a - P_b): C=80 {d[0]:.3f}, C=160 {d[1]:.3f}")
met = json.load(open(mpath))
check("evidence metrics.json pass", met.get("pass") is True)
for inc, e in met["legs"]["rb"].items():
    check(f"evidence leg b i={inc}", all(s["pass"] for s in e["evaluated"].values()))
for inc, e in met["legs"]["ra"].items():
    check(f"evidence leg a i={inc} no model-attributed miss",
          all(not s["model_attributed"] for s in e["evaluated"].values()))
print("R07 verify", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
