"""R06 reader acceptance: compare the RAPTOR AthenaK probe with the BlackLightPlus
reference at the same points (dag.md R06).

Inputs (from r06_reference.jl and model/athenak/tests/athenak_probe.c):
  DIR/TAG_ref.bin, DIR/TAG_raptor.bin   int64 n, int64 ncol, n x ncol float64
  DIR/TAG_class.bin                     n int8 point classes
  DIR/TAG_meta.json, DIR/TAG_probe.json run metadata (BLP and RAPTOR sides)
  DIR/TAG_fluid.bin (optional)          init_model + get_fluid_params on RAPTOR's
                                        SKS metric (model/athenak/tests/fluid_probe.c)

Acceptance (dag.md R06):
  owning block identical at every point; inside flag identical;
  interpolated primitives <= 1e-6 relative;
  u^mu, b^mu, n_e, B, Theta_e, sigma, beta_inv <= 1e-6 relative;
  u.u = -1 and u.b = 0 to 1e-10 (RAPTOR side);
  emit and sigma-cut flags identical;
  load memory <= 1.1 x the file size.
Relative differences are |a - b| / |b| for |b| > 0; a pair with b = 0 must have a = 0.

usage: python3 r06_compare.py DIR TAG OUT.json
"""
import json
import os
import sys

import numpy as np

COLS = (["block", "rho", "u1", "u2", "u3", "eint", "b1", "b2", "b3", "pgas"]
        + [f"ucon{i}" for i in range(4)] + [f"ucov{i}" for i in range(4)]
        + [f"bcon{i}" for i in range(4)] + [f"bcov{i}" for i in range(4)]
        + ["b_sq", "n_e", "B", "theta_e", "sigma", "beta_inv", "rho_cgs", "pgas_cgs",
           "emit", "omit", "u_dot_u", "u_dot_b"])
CLASSES = ["ball_R60", "block_faces", "level_jumps", "edges_corners", "excision",
           "domain_boundary_outside", "cell_centres"]
PRIMS = COLS[1:10]
DERIVED = COLS[10:34]
TOL = 1e-6


def load(path):
    with open(path, "rb") as f:
        n, ncol = np.fromfile(f, dtype=np.int64, count=2)
        a = np.fromfile(f, dtype=np.float64, count=n * ncol).reshape(n, ncol)
    return a


def reldiff(a, b):
    """Per-element relative difference; b == 0 requires a == 0 (else inf). NaN pairs match."""
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b != 0, d / np.abs(b), np.where(a == 0, 0.0, np.inf))
    r = np.where(both_nan, 0.0, r)
    return np.where(np.isnan(r), np.inf, r)


def main():
    d, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ref = load(os.path.join(d, f"{tag}_ref.bin"))
    rap = load(os.path.join(d, f"{tag}_raptor.bin"))
    cls = np.fromfile(os.path.join(d, f"{tag}_class.bin"), dtype=np.int8)
    meta = json.load(open(os.path.join(d, f"{tag}_meta.json")))
    probe = json.load(open(os.path.join(d, f"{tag}_probe.json")))
    assert ref.shape == rap.shape and ref.shape[1] == len(COLS) and len(cls) == ref.shape[0]
    n = ref.shape[0]
    col = {c: i for i, c in enumerate(COLS)}

    blk_r, blk_b = rap[:, 0], ref[:, 0]
    inside_b = blk_b >= 0
    inside_r = blk_r >= 0
    res = {"tag": tag, "file": meta["file"], "npoints": int(n)}
    res["points_per_class"] = {CLASSES[k]: int((cls == k).sum()) for k in range(len(CLASSES))}
    res["inside_points"] = int(inside_b.sum())
    res["block_mismatch"] = int((blk_r != blk_b).sum())
    res["inside_flag_mismatch"] = int((inside_r != inside_b).sum())
    res["block_mismatch_per_class"] = {CLASSES[k]: int(((blk_r != blk_b) & (cls == k)).sum())
                                       for k in range(len(CLASSES))}

    per = {}
    for c in PRIMS + DERIVED:
        i = col[c]
        r = reldiff(rap[inside_b, i], ref[inside_b, i])
        a, b = rap[inside_b, i], ref[inside_b, i]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        per[c] = {"max_rel": float(r.max()) if r.size else 0.0,
                  "bit_identical_fraction": float(same.mean()) if same.size else 1.0,
                  "n_over_tol": int((r > TOL).sum())}
    res["columns"] = per
    res["max_rel_primitives"] = max(per[c]["max_rel"] for c in PRIMS)
    res["max_rel_derived"] = max(per[c]["max_rel"] for c in DERIVED)
    for flag in ("emit", "omit"):
        res[f"{flag}_mismatch"] = int((rap[:, col[flag]] != ref[:, col[flag]]).sum())
    res["emitting_points"] = int((ref[:, col["emit"]] == 1).sum())
    res["sigma_cut_points"] = int((ref[:, col["omit"]] == 1).sum())

    ok = inside_r & np.all(np.isfinite(rap[:, col["ucon0"]:col["b_sq"] + 1]), axis=1)
    uu = np.abs(rap[ok, col["u_dot_u"]] + 1.0)
    bmag = np.sqrt(np.abs(rap[ok, col["b_sq"]]))
    ub = np.abs(rap[ok, col["u_dot_b"]])
    ub_rel = np.where(bmag > 0, ub / np.where(bmag > 0, bmag, 1.0), ub)
    res["u_dot_u_plus_1_max"] = float(uu.max()) if uu.size else 0.0
    res["u_dot_b_over_absb_max"] = float(ub_rel.max()) if ub_rel.size else 0.0
    res["u_dot_b_abs_max"] = float(ub.max()) if ub.size else 0.0
    res["nonfinite_fourvector_points"] = int((inside_r & ~ok).sum())
    res["nonfinite_fourvector_points_blp"] = int((inside_b & np.isnan(ref[:, col["ucon0"]])).sum())
    res["nonfinite_fourvector_per_class"] = {
        CLASSES[k]: int((inside_r & ~ok & (cls == k)).sum()) for k in range(len(CLASSES))}

    size_gb = os.path.getsize(meta["file"]) / 2 ** 30
    res["timing_memory"] = {
        "file_gb": size_gb,
        "raptor_load_s": probe["load_s"], "raptor_peak_rss_gb": probe["peak_rss_gb_load"],
        "raptor_data_gb": probe["data_bytes"] / 2 ** 30,
        "raptor_rss_over_file": probe["peak_rss_gb_load"] / size_gb,
        "raptor_sample_s": probe["sample_s"],
        "blp_load_s": meta["blp_load_s"], "blp_peak_rss_gb": meta["blp_peak_rss_gb"],
        "blp_sample_s": meta["blp_sample_s"], "blp_threads": meta["blp_threads"],
    }
    res["locate"] = {"raptor": probe["locate"], "blp": meta["locate"]}
    res["gamma"] = {"raptor": probe["gamma"], "blp": meta["gamma"]}
    res["level_jump_faces"] = meta["level_jump_faces"]

    fluid_path = os.path.join(d, f"{tag}_fluid.bin")
    if os.path.exists(fluid_path):
        fl = load(fluid_path)
        sha = open(os.path.join(d, f"{tag}_fluid.sha")).read().strip()
        emit_b = ref[:, col["emit"]] == 1
        fres = {"fork_tree": sha,
                "block_mismatch": int((fl[:, 0] != blk_b).sum()),
                "emit_mismatch": int((fl[:, 1] != ref[:, col["emit"]]).sum()),
                "metric_gdd_max_rel": float(np.nanmax(fl[inside_b, 22])),
                "metric_guu_max_rel": float(np.nanmax(fl[inside_b, 23]))}
        # four-vectors: max_mu |a_mu - b_mu| / max_mu |b_mu| (components near
        # cancellation, e.g. u_i ~ 1e-18 of a fluid at rest, make a per-component
        # ratio ill-conditioned: the metrics themselves differ at 1e-16)
        fcols = {}
        for name, j in (("ucon", 2), ("ucov", 6), ("bcon", 10), ("bcov", 14)):
            a = fl[inside_b, j:j + 4]
            b = ref[inside_b, col[name + "0"]:col[name + "0"] + 4]
            both_nan = np.all(np.isnan(a) & np.isnan(b), axis=1)
            nrm = np.max(np.abs(b), axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(nrm > 0, np.max(np.abs(a - b), axis=1) / nrm,
                             np.where(np.all(a == 0, axis=1), 0.0, np.inf))
            r = np.where(both_nan, 0.0, np.where(np.isnan(r), np.inf, r))
            fcols[name] = float(r.max()) if r.size else 0.0
        for name, j in (("B", 19), ("theta_e", 20), ("sigma", 21)):
            r = reldiff(fl[inside_b, j], ref[inside_b, col[name]])
            fcols[name] = float(r.max()) if r.size else 0.0
        r = reldiff(fl[emit_b, 18], ref[emit_b, col["n_e"]])
        fcols["n_e"] = float(r.max()) if r.size else 0.0
        fres["columns_max_rel"] = fcols
        fres["max_rel"] = max(fcols.values())
        res["get_fluid_params_sks"] = fres

    checks = {
        "block_identical": res["block_mismatch"] == 0,
        "inside_identical": res["inside_flag_mismatch"] == 0,
        "primitives_le_1e-6": res["max_rel_primitives"] <= TOL,
        "derived_le_1e-6": res["max_rel_derived"] <= TOL,
        "u_dot_u_le_1e-10": res["u_dot_u_plus_1_max"] <= 1e-10,
        "u_dot_b_le_1e-10": res["u_dot_b_over_absb_max"] <= 1e-10,
        "emit_flag_identical": res["emit_mismatch"] == 0,
        "sigma_cut_flag_identical": res["omit_mismatch"] == 0,
        "memory_le_1.1_file": res["timing_memory"]["raptor_rss_over_file"] <= 1.1,
        "same_gamma": probe["gamma"] == meta["gamma"],
        "same_locate_path": probe["locate"] == meta["locate"],
    }
    if "get_fluid_params_sks" in res:
        f = res["get_fluid_params_sks"]
        checks["sks_block_identical"] = f["block_mismatch"] == 0
        checks["sks_emit_identical"] = f["emit_mismatch"] == 0
        checks["sks_fluid_le_1e-6"] = f["max_rel"] <= TOL
    res["checks"] = checks
    res["pass"] = all(checks.values())
    with open(out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("tag", "pass", "block_mismatch", "max_rel_primitives",
                                          "max_rel_derived", "u_dot_u_plus_1_max",
                                          "u_dot_b_over_absb_max", "emit_mismatch",
                                          "omit_mismatch")}, sort_keys=True))
    print(json.dumps(checks))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
