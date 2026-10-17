#!/usr/bin/env python3
"""Compare the arrays of a figure data set with those of the reference run.

For every array the reference set holds (figures/out/...), the array of the same name
under BLP/figures/out/... is compared: shapes, bitwise equality (NaNs equal in place),
the number of differing entries, max|new - old| / max|old| and the relative L2 norm
||new - old|| / ||old|| over the finite entries. Arrays with a few columns (trajectory
and diagnostic tables) also get max|new - old| per column; tables of unequal length
are compared over their common leading rows. Prints a table and writes
BLP/figures/out/comparison_<name>.json.

Usage, from the repository root:
    python3 BLP/figures/compare_arrays.py examples | pol_hires | spherical_orbits
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NEW = os.path.join(HERE, "out")
OLD = os.path.join(REPO, "figures", "out")

SETS = {
    "examples": ["analytic_disk_I.npy", "torus_I.npy", "torus_Q.npy", "torus_U.npy",
                 "torus_V.npy"],
    "pol_hires": [os.path.join("pol_hires", f"{k}.npy") for k in "IQUV"],
    "spherical_orbits": None,          # every .npy of the reference directory
}


def compare(new, old):
    rec = {"shape_new": list(new.shape), "shape_old": list(old.shape)}
    rec["bitwise"] = bool(new.shape == old.shape and new.tobytes() == old.tobytes())
    if new.ndim == 2 and new.shape[1:] == old.shape[1:] and new.shape[0] != old.shape[0]:
        n = min(new.shape[0], old.shape[0])
        new, old = new[:n], old[:n]
        rec["rows_compared"] = n
    elif new.shape != old.shape:
        rec["comparable"] = False
        return rec
    fin = np.isfinite(old) & np.isfinite(new)
    rec["nan_mismatch"] = int(np.sum(np.isnan(old) != np.isnan(new)))
    d = np.where(fin, new - old, 0.0)
    o = np.where(fin, old, 0.0)
    rec["n_differ"] = int(np.sum((new != old) & fin))
    rec["n_entries"] = int(old.size)
    amax = np.abs(o).max()
    rec["max_abs_diff"] = float(np.abs(d).max())
    rec["max_abs_diff_rel"] = float(np.abs(d).max() / amax) if amax > 0 else float(np.abs(d).max())
    l2 = np.sqrt(np.sum(o * o))
    rec["rel_l2"] = float(np.sqrt(np.sum(d * d)) / l2) if l2 > 0 else float(np.sqrt(np.sum(d * d)))
    if new.ndim == 2 and new.shape[1] <= 8:
        rec["max_abs_diff_per_column"] = [float(v) for v in np.abs(d).max(axis=0)]
        if not rec["bitwise"]:
            row = np.flatnonzero(np.any((new != old) & fin, axis=1))
            rec["first_differing_row"] = int(row[0]) if row.size else None
    return rec


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in SETS:
        sys.exit(f"usage: compare_arrays.py {{{' | '.join(SETS)}}}")
    files = SETS[name]
    if files is None:
        files = sorted(os.path.relpath(p, OLD) for p in
                       glob.glob(os.path.join(OLD, name, "*.npy")))
    out = {"set": name, "new_root": NEW, "old_root": OLD, "arrays": {}}
    print(f"{'array':34s} {'shape new':>12s} {'shape old':>12s} {'bitwise':>7s} "
          f"{'n differ':>9s} {'max|d|/max|old|':>16s} {'rel L2':>10s}")
    for f in files:
        pn, po = os.path.join(NEW, f), os.path.join(OLD, f)
        if not os.path.exists(pn):
            out["arrays"][f] = {"missing_new": True}
            print(f"{f:34s} missing in the new set")
            continue
        r = compare(np.load(pn), np.load(po))
        out["arrays"][f] = r
        sn = "x".join(map(str, r["shape_new"]))
        so = "x".join(map(str, r["shape_old"]))
        if r.get("comparable", True) is False:
            print(f"{f:34s} {sn:>12s} {so:>12s} {'no':>7s}   shapes differ")
            continue
        print(f"{f:34s} {sn:>12s} {so:>12s} {('yes' if r['bitwise'] else 'no'):>7s} "
              f"{r['n_differ']:9d} {r['max_abs_diff_rel']:16.3e} {r['rel_l2']:10.3e}")
    path = os.path.join(NEW, f"comparison_{name}.json")
    with open(path, "w") as io:
        json.dump(out, io, indent=1)
        io.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
