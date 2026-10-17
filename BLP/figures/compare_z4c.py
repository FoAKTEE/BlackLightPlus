#!/usr/bin/env python3
"""Compare two runs of the Z4c binary-black-hole data sets: the reference outputs under
figures/out/ and the BLP outputs under BLP/figures/out/.

z4c_amr: the class, bend, level and root-class arrays (fraction of equal pixels, counts per
class), the per-ray cache chunks (coordinates, class, fate, bending angle), the record
numbers (rays, classes and wall time per level, level histograms, pixel class counts) and
the boundary agreement of the capture edge with the ring and anchor points (the distance
of plot_binary.py: to the nearest midpoint between a captured and a non-captured
4-neighbour pixel).
z4c_phasespace: the capture map (equal cells, components), the ring and anchor points
(row-matched and nearest-point distances), and per deck ray the launch point, fate,
bending-angle class, bending angle, step count, final state and escape direction.

Prints the tables and writes BLP/figures/out/comparison_z4c.json.
Run from anywhere: python3 BLP/figures/compare_z4c.py [--no-boundary]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(REPO, "figures", "out")
NEW = os.path.join(REPO, "BLP", "figures", "out")
BEND_CUT = 360.0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def boundary_distance(cls, lim, pts):
    """Distance from each point to the nearest midpoint between two 4-neighbour pixels of
    which exactly one is captured (plot_binary.boundary_distance)."""
    n = cls.shape[0]
    px = 2 * lim / n
    c = -lim + (np.arange(n) + 0.5) * px
    cap = cls == 1
    mids = []
    jb, ia = np.nonzero(cap[1:, :] != cap[:-1, :])
    mids.append(np.column_stack([c[ia], 0.5 * (c[jb] + c[jb + 1])]))
    jb, ia = np.nonzero(cap[:, 1:] != cap[:, :-1])
    mids.append(np.column_stack([0.5 * (c[ia] + c[ia + 1]), c[jb]]))
    d, _ = cKDTree(np.vstack(mids)).query(pts)
    return d


def dist_stats(d):
    return {"n": int(len(d)), "median": float(np.median(d)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max())}


def same(a, b):
    return a == b


# ------------------------------------------------------------------------- z4c_amr
def compare_amr(boundary=True):
    o, n = os.path.join(OLD, "z4c_amr"), os.path.join(NEW, "z4c_amr")
    ro, rn = load_json(os.path.join(o, "records.json")), load_json(os.path.join(n, "records.json"))
    out = {"old_dir": o, "new_dir": n, "arrays": {}, "records": {}}

    for name in ("class", "levels", "levels_fine", "root_class"):
        A, B = np.load(os.path.join(o, name + ".npy")), np.load(os.path.join(n, name + ".npy"))
        row = {"shape_old": list(A.shape), "shape_new": list(B.shape)}
        if A.shape == B.shape:
            eq = A == B
            row.update(equal_fraction=float(eq.mean()), n_different=int((~eq).sum()),
                       bitwise_equal=bool(eq.all()))
        if name in ("class", "root_class"):
            row["counts_old"] = [int((A == q).sum()) for q in range(4)]
            row["counts_new"] = [int((B == q).sum()) for q in range(4)]
        if name in ("levels", "levels_fine"):
            ml = int(max(A.max(), B.max()))
            row["hist_old"] = [int((A == q).sum()) for q in range(ml + 1)]
            row["hist_new"] = [int((B == q).sum()) for q in range(ml + 1)]
        out["arrays"][name] = row
        del A, B
    A, B = np.load(os.path.join(o, "bend.npy")), np.load(os.path.join(n, "bend.npy"))
    both = np.isfinite(A) & np.isfinite(B)
    out["arrays"]["bend"] = {
        "bitwise_equal": bool(np.array_equal(A, B, equal_nan=True)),
        "nan_pattern_equal": bool(np.array_equal(np.isnan(A), np.isnan(B))),
        "max_abs_diff_deg": float(np.abs(A[both] - B[both]).max()) if both.any() else None}
    del A, B

    # per-ray cache chunks: every ray of every level
    levels = {}
    for f in sorted(glob.glob(os.path.join(n, "cache", "L*_c*.npy"))):
        base = os.path.basename(f)
        lv = int(base[1:base.index("_")])
        g = os.path.join(o, "cache", base)
        L = levels.setdefault(lv, dict(chunks=0, rays=0, missing_old=0, coords_equal=0,
                                       class_equal=0, fate_equal=0, bend_bitwise=0,
                                       bend_max_abs_diff=0.0))
        Mn = np.load(f)
        L["chunks"] += 1
        L["rays"] += Mn.shape[1]
        if not os.path.exists(g):
            L["missing_old"] += Mn.shape[1]
            continue
        Mo = np.load(g)
        if Mo.shape != Mn.shape:
            L["missing_old"] += Mn.shape[1]
            continue
        ceq = (Mo[0] == Mn[0]) & (Mo[1] == Mn[1])
        L["coords_equal"] += int(ceq.sum())
        L["class_equal"] += int((Mo[2] == Mn[2]).sum())
        L["fate_equal"] += int((Mo[4] == Mn[4]).sum())
        L["bend_bitwise"] += int(((Mo[3] == Mn[3]) | (np.isnan(Mo[3]) & np.isnan(Mn[3]))).sum())
        fin = np.isfinite(Mo[3]) & np.isfinite(Mn[3])
        if fin.any():
            L["bend_max_abs_diff"] = max(L["bend_max_abs_diff"],
                                         float(np.abs(Mo[3][fin] - Mn[3][fin]).max()))
    out["cache"] = {str(k): v for k, v in sorted(levels.items())}

    for key in ("nrays", "per_level_rays", "per_level_classes_esc_cap_bent_other",
                "levels_hist", "pixel_class_counts_esc_cap_bent_other",
                "ray_fraction_vs_uniform", "per_level_wall_s", "wall_s", "threads",
                "rays_per_s", "git_sha"):
        out["records"][key] = {"old": ro.get(key), "new": rn.get(key),
                               "equal": same(ro.get(key), rn.get(key))}
    for key in ("tile_hist", "consistent_with_base_map"):
        a, b = ro["levels_fine"].get(key), rn["levels_fine"].get(key)
        out["records"]["levels_fine." + key] = {"old": a, "new": b, "equal": same(a, b)}
    for key in ("base_res", "block", "max_level", "eff_res", "finest_px"):
        a, b = ro["amr"].get(key), rn["amr"].get(key)
        out["records"]["amr." + key] = {"old": a, "new": b, "equal": same(a, b)}

    if boundary:
        # capture edge of each class map against the ring and anchors of each phase-space run
        lim = rn["lim"]
        ps_o, ps_n = os.path.join(OLD, "z4c_phasespace"), os.path.join(NEW, "z4c_phasespace")
        pts = {"anchors_old": np.load(os.path.join(ps_o, "anchors.npy"))[:, 2:4],
               "ring_old": np.load(os.path.join(ps_o, "ring.npy"))[:, 2:4]}
        if os.path.exists(os.path.join(ps_n, "anchors.npy")):
            pts["anchors_new"] = np.load(os.path.join(ps_n, "anchors.npy"))[:, 2:4]
            pts["ring_new"] = np.load(os.path.join(ps_n, "ring.npy"))[:, 2:4]
        ba = {}
        for tag, d in (("class_old", o), ("class_new", n)):
            C = np.load(os.path.join(d, "class.npy"))
            ba[tag] = {k: dist_stats(boundary_distance(C, lim, p)) for k, p in pts.items()}
            ba[tag]["captured_fraction"] = float((C == 1).mean())
            del C
        out["boundary_agreement"] = ba
    return out


# ------------------------------------------------------------------ z4c_phasespace
def match_points(Po, Pn):
    """Row-matched (component, transition) comparison where the row keys agree, and the
    nearest-point distance of every new point to the old set (and back)."""
    res = {"n_old": int(len(Po)), "n_new": int(len(Pn))}
    if len(Po) == len(Pn) and np.array_equal(Po[:, :2], Pn[:, :2]):
        d = np.hypot(Po[:, 2] - Pn[:, 2], Po[:, 3] - Pn[:, 3])
        res["row_matched"] = dist_stats(d)
    else:
        res["row_matched"] = None
    dn, _ = cKDTree(Po[:, 2:4]).query(Pn[:, 2:4])
    do, _ = cKDTree(Pn[:, 2:4]).query(Po[:, 2:4])
    res["nearest_new_to_old"] = dist_stats(dn)
    res["nearest_old_to_new"] = dist_stats(do)
    return res


def ray_class(r):
    if r["bend_deg"] > BEND_CUT:
        return "bent>360"
    return r["fate"]


def compare_phasespace():
    o, n = os.path.join(OLD, "z4c_phasespace"), os.path.join(NEW, "z4c_phasespace")
    do_, dn = load_json(os.path.join(o, "deck.json")), load_json(os.path.join(n, "deck.json"))
    out = {"old_dir": o, "new_dir": n}
    Mo, Mn = np.load(os.path.join(o, "map.npy")), np.load(os.path.join(n, "map.npy"))
    if Mo.shape != Mn.shape:
        raise SystemExit(f"map shapes differ: {Mo.shape} vs {Mn.shape}")
    eq = Mo == Mn
    out["map"] = {"shape_old": list(Mo.shape), "shape_new": list(Mn.shape),
                  "equal_fraction": float(eq.mean()), "n_different": int((~eq).sum()),
                  "captured_old_only": int(((Mo == 1) & (Mn == 0)).sum()),
                  "captured_new_only": int(((Mo == 0) & (Mn == 1)).sum()),
                  "n_captured_old": int((Mo == 1).sum()), "n_captured_new": int((Mn == 1).sum())}
    if (~eq).any():
        lim, nm = dn["lim"], Mn.shape[0]
        ax = np.linspace(-lim, lim, nm)
        ia, jb = np.nonzero(~eq)
        cap = Mo == 1
        edge = np.zeros_like(cap)
        edge[1:, :] |= cap[1:, :] != cap[:-1, :]
        edge[:-1, :] |= cap[1:, :] != cap[:-1, :]
        edge[:, 1:] |= cap[:, 1:] != cap[:, :-1]
        edge[:, :-1] |= cap[:, 1:] != cap[:, :-1]
        out["map"]["different_cells_on_old_edge"] = int(edge[~eq].sum())
        out["map"]["different_cells_ab"] = [[float(ax[i]), float(ax[j])]
                                            for i, j in zip(ia[:200], jb[:200])]
    for key in ("n_captured_cells", "n_components", "component_sizes", "kept_components",
                "n_anchor_points", "n_ring_points", "naz_anchor", "naz_ring", "edge_b_lo",
                "edge_b_hi", "nmap", "git_sha", "backend"):
        out.setdefault("records", {})[key] = {"old": do_.get(key), "new": dn.get(key),
                                              "equal": same(do_.get(key), dn.get(key))}
    out["records"]["edge_b_lo"]["diff"] = dn["edge_b_lo"] - do_["edge_b_lo"]
    out["records"]["edge_b_hi"]["diff"] = dn["edge_b_hi"] - do_["edge_b_hi"]
    out["anchors"] = match_points(np.load(os.path.join(o, "anchors.npy")),
                                  np.load(os.path.join(n, "anchors.npy")))
    out["ring"] = match_points(np.load(os.path.join(o, "ring.npy")),
                               np.load(os.path.join(n, "ring.npy")))

    rays = []
    ro = {r["name"]: r for r in do_["rays"]}
    for r in dn["rays"]:
        s = ro.get(r["name"])
        if s is None:
            rays.append({"name": r["name"], "missing_old": True})
            continue
        To = np.load(os.path.join(o, f"traj_{r['name']}.npy"))
        Tn = np.load(os.path.join(n, f"traj_{r['name']}.npy"))
        row = {"name": r["name"], "da": r["a"] - s["a"], "db": r["b"] - s["b"],
               "fate_old": s["fate"], "fate_new": r["fate"],
               "class_old": ray_class(s), "class_new": ray_class(r),
               "bend_old": s["bend_deg"], "bend_new": r["bend_deg"],
               "dbend": r["bend_deg"] - s["bend_deg"],
               "npts_old": s["n_points"], "npts_new": r["n_points"],
               "bitwise": bool(To.shape == Tn.shape and np.array_equal(To, Tn)),
               "dx_end": float(np.linalg.norm(Tn[-1, 2:5] - To[-1, 2:5])),
               "dp_end": float(np.linalg.norm(Tn[-1, 5:9] - To[-1, 5:9])),
               "dx_start": float(np.linalg.norm(Tn[0, 2:5] - To[0, 2:5])),
               "dp_start": float(np.linalg.norm(Tn[0, 5:9] - To[0, 5:9])),
               "end_r_old": s["end_r"], "end_r_new": r["end_r"]}
        if r["fate"] == "escape" and s["fate"] == "escape":
            vo = np.array([s["p_ea"], s["p_eb"]])
            vn = np.array([r["p_ea"], r["p_eb"]])
            row["d_escape_dir"] = float(np.linalg.norm(vn - vo))
        rays.append(row)
    out["deck"] = rays
    cnt = lambda deck: {"bent>360": sum(ray_class(x) == "bent>360" for x in deck["rays"]),
                        "bent>360_captured": sum(ray_class(x) == "bent>360" and
                                                 x["fate"] == "capture" for x in deck["rays"]),
                        "captured": sum(x["fate"] == "capture" for x in deck["rays"]),
                        "escaped": sum(x["fate"] == "escape" for x in deck["rays"])}
    out["deck_counts"] = {"old": cnt(do_), "new": cnt(dn)}
    out["wall_s_new"] = dn.get("wall_s")
    return out


# ---------------------------------------------------------------------------- print
def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def show(cmp):
    if "amr" in cmp:
        a = cmp["amr"]
        print("== z4c_amr: arrays")
        for k, v in a["arrays"].items():
            print(f"  {k:12s} " + "  ".join(f"{kk}={fmt(vv)}" for kk, vv in v.items()))
        print("== z4c_amr: per-ray cache (old vs new, every ray)")
        print(f"  {'level':>5} {'rays':>8} {'coords=':>8} {'class=':>8} {'fate=':>8} "
              f"{'bend bitwise':>12} {'max|dbend|':>10}")
        for k, v in a["cache"].items():
            print(f"  {k:>5} {v['rays']:8d} {v['coords_equal']:8d} {v['class_equal']:8d} "
                  f"{v['fate_equal']:8d} {v['bend_bitwise']:12d} {v['bend_max_abs_diff']:10.3g}")
        print("== z4c_amr: records")
        for k, v in a["records"].items():
            print(f"  {k:38s} equal={v['equal']!s:5s} old={fmt(v['old'])}  new={fmt(v['new'])}")
        if "boundary_agreement" in a:
            print("== z4c_amr: boundary agreement (distance to the class-1 edge, M)")
            for tag, d in a["boundary_agreement"].items():
                for k, s in d.items():
                    if k == "captured_fraction":
                        print(f"  {tag:10s} captured fraction {s:.8f}")
                    else:
                        print(f"  {tag:10s} {k:12s} n={s['n']:5d} median={s['median']:.3e} "
                              f"p99={s['p99']:.3e} max={s['max']:.3e}")
    if "phasespace" in cmp:
        p = cmp["phasespace"]
        print("== z4c_phasespace: map")
        print("  " + "  ".join(f"{k}={fmt(v)}" for k, v in p["map"].items()
                               if k != "different_cells_ab"))
        print("== z4c_phasespace: records")
        for k, v in p["records"].items():
            print(f"  {k:20s} equal={v['equal']!s:5s} old={fmt(v['old'])}  new={fmt(v['new'])}"
                  + (f"  diff={v['diff']:.3e}" if "diff" in v else ""))
        for k in ("anchors", "ring"):
            v = p[k]
            print(f"== z4c_phasespace: {k} n_old={v['n_old']} n_new={v['n_new']}")
            for kk in ("row_matched", "nearest_new_to_old", "nearest_old_to_new"):
                s = v[kk]
                print(f"  {kk:20s} " + ("rows differ" if s is None else
                      f"median={s['median']:.3e} p99={s['p99']:.3e} max={s['max']:.3e}"))
        print("== z4c_phasespace: deck")
        print(f"  {'ray':18s} {'fate o/n':>15s} {'class o/n':>19s} {'bend old':>9s} "
              f"{'bend new':>9s} {'dbend':>9s} {'npts o/n':>10s} {'|dx_end|':>9s} "
              f"{'|dp_end|':>9s} {'|d esc dir|':>10s}")
        for r in p["deck"]:
            print(f"  {r['name'][4:]:18s} {r['fate_old']:>7s}/{r['fate_new']:<7s} "
                  f"{r['class_old']:>9s}/{r['class_new']:<9s} {r['bend_old']:9.3f} "
                  f"{r['bend_new']:9.3f} {r['dbend']:9.2e} {r['npts_old']:5d}/{r['npts_new']:<4d} "
                  f"{r['dx_end']:9.2e} {r['dp_end']:9.2e} "
                  f"{r.get('d_escape_dir', float('nan')):10.2e}")
        print(f"  counts old {p['deck_counts']['old']}  new {p['deck_counts']['new']}")


def main():
    boundary = "--no-boundary" not in sys.argv
    cmp = {}
    if os.path.exists(os.path.join(NEW, "z4c_amr", "records.json")):
        cmp["amr"] = compare_amr(boundary)
    if os.path.exists(os.path.join(NEW, "z4c_phasespace", "deck.json")):
        cmp["phasespace"] = compare_phasespace()
    show(cmp)
    path = os.path.join(NEW, "comparison_z4c.json")
    with open(path, "w") as f:
        json.dump(cmp, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
