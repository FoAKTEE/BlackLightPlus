"""R01/R05 baseline check on runs made by r01_run.sh.

usage: python3 r01_verify.py <rundir_ref> [<rundir_other>]

<rundir_ref>: the upstream build (b91e937). Checks the spectrum-file flux against the
recorded R01 value 1.37395e-06 Jy (to 1e-5 relative) and the reassembled image sum
against the spectrum flux (<= 1e-6 relative).
<rundir_other>: optional second run (e.g. the blp-bbh head with the new switches off);
checks that every HDF5 dataset is bitwise identical to <rundir_ref>.
Exit 0 iff every check passes; prints one JSON object.
"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from raptor_h5_to_array import load  # noqa: E402

R01_FLUX = 1.37395e-06


def spectrum_flux(run):
    with open(f"{run}/run/output/spectrum_0_60.00.dat") as f:
        return float(f.readline().split()[1])


def check_ref(run):
    F = spectrum_flux(run)
    img = load(f"{run}/run/output/img_data_0.h5", 40.0, 100, 6.5e9, 16800.0)
    s = img["I0"].sum() if "I0" in img else next(v for k, v in img.items() if k.startswith("I")).sum()
    return {"flux_Jy": F, "flux_rel_to_R01": F / R01_FLUX - 1,
            "reassembled_sum_Jy": float(s), "reassembly_rel": float(s / F - 1),
            "pass": bool(abs(F / R01_FLUX - 1) <= 1e-5 and abs(s / F - 1) <= 1e-6)}


def bitwise(a, b):
    out = {}
    with h5py.File(f"{a}/run/output/img_data_0.h5") as fa, h5py.File(f"{b}/run/output/img_data_0.h5") as fb:
        assert sorted(fa.keys()) == sorted(fb.keys())
        for k in fa.keys():
            x, y = fa[k][...], fb[k][...]
            out[k] = bool(x.shape == y.shape and x.tobytes() == y.tobytes())
    return {"datasets_bitwise_identical": out, "pass": all(out.values()),
            "spectrum_identical": open(f"{a}/run/output/spectrum_0_60.00.dat").read()
            == open(f"{b}/run/output/spectrum_0_60.00.dat").read()}


res = {"ref": check_ref(sys.argv[1])}
ok = res["ref"]["pass"]
if len(sys.argv) > 2:
    res["bitwise"] = bitwise(sys.argv[1], sys.argv[2])
    ok = ok and res["bitwise"]["pass"] and res["bitwise"]["spectrum_identical"]
res["pass"] = ok
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
