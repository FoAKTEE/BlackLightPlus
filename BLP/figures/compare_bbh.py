#!/usr/bin/env python3
"""Compare the binary-black-hole figure data of BLP with the reference runs, and compute
the RAPTOR comparison metrics of the BLP images of snapshot 00455.

Data sets (new under BLP/, reference under the repository root):
  eht_bbh    figures/out/eht_bbh/<snap>_i<inc>_Rh40/{I,Q,U,V,class}.npy and records.json,
             snapshots 00455, 00122, 00093 at i = 17 and 90 deg
  amr        figures/out/bbh_i90_pol_amr/{amr4096_*, root256_*, probe256_*}.npy and
             records.json
  r07        out/raptor/R07/rerun_head/blp_i90_C<C>_n480/{I,Q,U,V,class}.npy and
             record.json (C = 320; C = 40, 80, 160 against out/raptor/R07/prod)

Per array: bitwise equality (NaNs equal in place), entries that differ, max|new - old| /
max|old|, the relative L2 norm ||new - old|| / ||old||, and the sums of both with their
relative difference (the flux up to the pixel solid angle; in Jy for the 480^2 images).
Per record: every numeric leaf present in both records except wall times, thread counts
and provenance strings, with old, new and |new - old| / |old|; lists are compared
element by element (largest relative difference). For the eht_bbh images, the image
metrics of figures/plot_eht_bbh.py (20 uas blurred polarization, beta_2, ring diameters
and f_C per hole) are computed from the new arrays and compared with the `metrics`
block of the reference records. Prints a summary and writes
BLP/figures/out/comparison_bbh.json.

RAPTOR metrics: the definitions of results/blp/raptor/tools/r07_compare.py (imported,
unchanged) between the BLP arrays of BLP/out/raptor/R07/rerun_head and the RAPTOR legs
of BLP/out/raptor/R07/prod at i = 90 deg, over the step constants C present in both
(the self-convergence errors and the Richardson flux need C = 40 ... 320): legs ra and rb
per level, the report legs at C = 80, and the BLP observables. Writes
BLP/results/blp/raptor/evidence/R07/metrics.json (the keys of the reference file; the
i = 17 and 60 deg entries and the i = 17 deg resolution legs are not part of it).

Usage, from the repository root:
    python3 BLP/figures/compare_bbh.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLP_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(BLP_ROOT)
OUT_JSON = os.path.join(HERE, "out", "comparison_bbh.json")
MET_JSON = os.path.join(BLP_ROOT, "results", "blp", "raptor", "evidence", "R07", "metrics.json")

C_CGS = 2.99792458e10
G_CGS = 6.6743e-8
MSUN = 1.98847e33
L_UNIT = G_CGS * 2 * 6.5e9 * MSUN / C_CGS ** 2
D_CM = 16.8e6 * 3.0856775814913673e18

SNAPS = ("s455", "s122", "s093")
INCS = (17, 90)
SKIP_KEYS = {"when", "dir", "git_sha", "threads", "host_cpus", "julia", "backend",
             "blp_version", "what", "code", "walls_s", "maxrss_GB", "rays_per_s",
             "samples_per_s", "rate_rays_per_s", "est_wall_s", "timing_best_s",
             "diagnostics_wall_s", "load_s", "wall_s_jit_incl", "repeats", "dataset",
             "snapshot_file", "file", "rate_source", "source"}


def pixsr(fov, npix):
    return (fov / npix * L_UNIT / D_CM) ** 2


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- arrays
def compare_array(new, old, jy=None):
    r = {"shape_new": list(new.shape), "shape_old": list(old.shape)}
    r["bitwise"] = bool(new.shape == old.shape and new.tobytes() == old.tobytes())
    if new.shape != old.shape:
        r["comparable"] = False
        return r
    fin = np.isfinite(old) & np.isfinite(new)
    r["nan_mismatch"] = int(np.sum(np.isnan(old) != np.isnan(new)))
    d = np.where(fin, new - old, 0.0)
    o = np.where(fin, old, 0.0)
    r["n_differ"] = int(np.sum((new != old) & fin))
    r["n_entries"] = int(old.size)
    amax = np.abs(o).max()
    r["max_abs_diff"] = float(np.abs(d).max())
    r["max_abs_diff_rel"] = float(np.abs(d).max() / amax) if amax > 0 else float(np.abs(d).max())
    l2 = np.sqrt(np.sum(o * o))
    r["rel_l2"] = float(np.sqrt(np.sum(d * d)) / l2) if l2 > 0 else float(np.sqrt(np.sum(d * d)))
    sn, so = float(np.sum(np.where(fin, new, 0.0))), float(np.sum(o))
    r["sum_new"], r["sum_old"] = sn, so
    r["sum_rel_diff"] = (sn - so) / abs(so) if so != 0 else sn - so
    if jy is not None:
        r["flux_new_Jy"], r["flux_old_Jy"] = sn * jy, so * jy
    return r


def array_set(pairs, jy=None):
    out = {}
    for label, pn, po in pairs:
        if not os.path.exists(pn) or not os.path.exists(po):
            out[label] = {"missing_new": not os.path.exists(pn), "missing_old": not os.path.exists(po)}
            continue
        out[label] = compare_array(np.load(pn), np.load(po), jy)
    return out


# ---------------------------------------------------------------- records
def leaves(x, path=""):
    """Numeric, boolean and string leaves of a JSON tree as {path: value}; lists of numbers
    stay whole."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP_KEYS or k.endswith("_wall_s") or k == "wall_s":
                continue
            out.update(leaves(v, f"{path}.{k}" if path else k))
    elif isinstance(x, list) and x and all(isinstance(v, (dict, list)) for v in x) and \
            not all(isinstance(v, list) and all(isinstance(w, (int, float)) for w in v) for v in x):
        for i, v in enumerate(x):
            out.update(leaves(v, f"{path}[{i}]"))
    else:
        out[path] = x
    return out


def _flat(v):
    if isinstance(v, list):
        r = []
        for w in v:
            r.extend(_flat(w))
        return r
    return [v]


def compare_value(new, old):
    if isinstance(old, (bool, str)) or old is None or isinstance(new, (bool, str)) or new is None:
        return {"old": old, "new": new, "equal": new == old}
    fo, fn = _flat(old), _flat(new)
    if len(fo) != len(fn):
        return {"old": old, "new": new, "equal": False, "note": "lengths differ"}
    rel = 0.0
    for a, b in zip(fo, fn):
        if a is None or b is None:
            if a != b:
                rel = float("inf")
            continue
        if a != b:
            rel = max(rel, abs(b - a) / abs(a) if a != 0 else float("inf"))
    return {"old": old, "new": new, "equal": fo == fn, "max_rel_diff": rel}


def compare_records(new, old):
    ln, lo = leaves(new), leaves(old)
    out = {k: compare_value(ln[k], lo[k]) for k in sorted(set(ln) & set(lo))}
    return out, sorted(set(lo) - set(ln)), sorted(set(ln) - set(lo))


# ---------------------------------------------------------------- data sets
def eht_set(peb):
    res = {}
    new_root = os.path.join(HERE, "out", "eht_bbh")
    old_root = os.path.join(REPO, "figures", "out", "eht_bbh")
    jy = pixsr(56.0, 480) / 1e-23
    for s in SNAPS:
        for inc in INCS:
            name = f"{s}_i{inc:02d}_Rh40"
            dn, do = os.path.join(new_root, name), os.path.join(old_root, name)
            if not os.path.exists(os.path.join(dn, "records.json")):
                continue
            arr = array_set([(c, os.path.join(dn, f"{c}.npy"), os.path.join(do, f"{c}.npy"))
                             for c in ("I", "Q", "U", "V", "class")], jy)
            rn = json.load(open(os.path.join(dn, "records.json")))
            ro = json.load(open(os.path.join(do, "records.json")))
            rec, only_old, only_new = compare_records(rn, {k: v for k, v in ro.items() if k != "metrics"})
            entry = {"arrays": arr, "record": rec, "record_keys_only_old": only_old,
                     "record_keys_only_new": only_new}
            if "metrics" in ro:
                img = {c: np.load(os.path.join(dn, f"{c}.npy")) for c in ("I", "Q", "U", "V", "class")}
                met, _ = peb.metrics(rn, img)
                m, mo, mn = compare_records(met, ro["metrics"])
                entry["image_metrics"] = m
            res[name] = entry
    return res


def amr_set():
    dn = os.path.join(HERE, "out", "bbh_i90_pol_amr")
    do = os.path.join(REPO, "figures", "out", "bbh_i90_pol_amr")
    if not os.path.exists(os.path.join(dn, "records.json")):
        return {}
    names = [f"amr4096_{c}" for c in ("I", "Q", "U", "V", "levels", "levels_fine")] + \
            [f"root256_{c}" for c in "IQUV"] + [f"probe256_{c}" for c in ("I", "grad", "class")]
    arr = array_set([(n, os.path.join(dn, f"{n}.npy"), os.path.join(do, f"{n}.npy")) for n in names])
    rn = json.load(open(os.path.join(dn, "records.json")))
    ro = json.load(open(os.path.join(do, "records.json")))
    rec, only_old, only_new = compare_records(rn, ro)
    return {"arrays": arr, "record": rec, "record_keys_only_old": only_old,
            "record_keys_only_new": only_new}


def r07_set():
    run = os.path.join(BLP_ROOT, "out", "raptor", "R07", "rerun_head")
    jy = pixsr(56.0, 480) / 1e-23
    res = {}
    if not os.path.isdir(run):
        return res
    for d in sorted(os.listdir(run)):
        if not (d.startswith("blp_i") and os.path.isdir(os.path.join(run, d))):
            continue
        C = int(d.split("_C")[1].split("_")[0])
        ref = os.path.join(REPO, "out", "raptor", "R07", "rerun_head" if C == 320 else "prod", d)
        if not os.path.isdir(ref):
            continue
        dn = os.path.join(run, d)
        arr = array_set([(c, os.path.join(dn, f"{c}.npy"), os.path.join(ref, f"{c}.npy"))
                         for c in ("I", "Q", "U", "V", "class")], jy)
        rn = json.load(open(os.path.join(dn, "record.json")))
        ro = json.load(open(os.path.join(ref, "record.json")))
        rec, only_old, only_new = compare_records(rn, ro)
        res[d] = {"reference": os.path.relpath(ref, REPO), "arrays": arr, "record": rec,
                  "record_keys_only_old": only_old, "record_keys_only_new": only_new}
    return res


# ---------------------------------------------------------------- RAPTOR metrics
def r07_metrics(rc, inc=90, npix=480):
    run = os.path.join(BLP_ROOT, "out", "raptor", "R07", "rerun_head")
    prod = os.path.join(BLP_ROOT, "out", "raptor", "R07", "prod")
    Cs = sorted(int(d.split("_C")[1].split("_")[0]) for d in os.listdir(run)
                if d.startswith(f"blp_i{inc:02d}_") and d.endswith(f"_n{npix}"))
    blp = {C: rc.load_blp(run, inc, C, npix) for C in Cs}
    rc.CENTERS = blp[Cs[0]]["record"]["centers"]
    res = {"thresholds": rc.THRESH, "npix": npix, "fov_M": rc.FOV, "uas_per_M": rc.UAS_PER_M,
           "legs": {}, "report_legs": {}, "resolution": {},
           "blp_git_sha": blp[Cs[-1]]["record"]["git_sha"]}
    allpass = True
    oB = {C: rc.observables(blp[C], inc, npix) for C in Cs}
    res["blp"] = {str(inc): {"observables": {str(C): oB[C] for C in Cs},
                             "flux_richardson": rc.richardson(Cs, [oB[C]["flux_I_Jy"] for C in Cs])}}
    for leg in ("ra", "rb"):
        ims = {C: rc.load_raptor(prod, leg, inc, C, npix) for C in Cs}
        CL = [C for C in Cs if ims[C] is not None]
        if not CL:
            continue
        oR = {C: rc.observables(ims[C], inc, npix) for C in CL}
        epsR = rc.self_eps(oR, ims, CL) if len(CL) >= 2 else {C: {} for C in CL}
        epsB = rc.self_eps(oB, blp, CL) if len(CL) >= 2 else {C: {} for C in CL}
        rich_R = rc.richardson(CL, [oR[C]["flux_I_Jy"] for C in CL])
        rich_Bl = rc.richardson(CL, [oB[C]["flux_I_Jy"] for C in CL])
        levels = {}
        for C in CL:
            m, _, _ = rc.pair_metrics(ims[C], blp[C], inc, npix, oR[C], oB[C])
            g = rc.gate(m, inc)
            m["gate"] = g
            m["pass"] = all(g.values())
            m["attribution"] = rc.attribute(m, g, epsR[C], epsB[C])
            m["eps_raptor"] = epsR[C]
            m["eps_blp"] = epsB[C]
            m["observables_raptor"] = oR[C]
            if ims[C]["timing"]:
                m["raptor_wall_s"] = ims[C]["timing"]["wall_best_s"]
            m["blp_wall_s"] = blp[C]["record"]["timing_best_s"]["image_wall"]
            if ims[C]["nsamples"] is not None:
                m["samples_raptor"] = int(ims[C]["nsamples"].sum())
            m["samples_blp"] = blp[C]["record"]["samples"]
            if ims[C]["tauF"] is not None:
                br = blp[C]["I"] > 0.01 * blp[C]["I"].max()
                m["faraday_depth_raptor_bright_pct_10_50_90"] = [
                    float(x) for x in np.percentile(ims[C]["tauF"][br], [10, 50, 90])]
            levels[str(C)] = m
        fr = rich_R["limit"] / rich_Bl["limit"] - 1 if rich_R and rich_Bl else None
        evalC = sorted(C for C in {min(160, CL[-1]), CL[-1]} if str(C) in levels)
        summary = {}
        for C in evalC:
            m = levels[str(C)]
            model_miss = [k for k, v in m["attribution"].items() if v["class"] == "model"]
            summary[str(C)] = {"pass": m["pass"], "failed": [k for k, v in m["gate"].items() if not v],
                               "model_attributed": model_miss}
        res["legs"].setdefault(leg, {})[str(inc)] = {
            "C_levels": CL, "levels": levels, "flux_richardson_raptor": rich_R,
            "flux_richardson_blp": rich_Bl, "flux_rel_richardson": fr,
            "flux_richardson_pass": fr is not None and abs(fr) <= rc.THRESH["flux_rel_richardson"],
            "evaluated": summary}
        if leg == "rb":
            allpass &= all(s["pass"] for s in summary.values())
        if leg == "ra":
            allpass &= all(not s["model_attributed"] for s in summary.values())
    if 80 in blp:
        for leg in ("rc", "rstock", "rc0", "rstock0"):
            R = rc.load_raptor(prod, leg, inc, 80, npix)
            if R is None:
                continue
            m, oRr, _ = rc.pair_metrics(R, blp[80], inc, npix, None, oB[80])
            m["observables_raptor"] = oRr
            res["report_legs"].setdefault(leg, {})[str(inc)] = {"C": 80, "vs_blp": m}
    res["pass"] = bool(allpass)
    return res


# ---------------------------------------------------------------- summary
def fmt_arrays(title, arrs):
    for k, r in arrs.items():
        if "n_differ" not in r:
            print(f"  {title:24s} {k:18s} {r}")
            continue
        print(f"  {title:24s} {k:18s} {'bitwise' if r['bitwise'] else 'differs':8s} "
              f"{r['n_differ']:9d} {r['max_abs_diff_rel']:11.3e} {r['rel_l2']:11.3e} "
              f"{r['sum_rel_diff']:+11.3e}")


def show(rec, keys, title):
    for k in keys:
        if k in rec:
            v = rec[k]
            if v.get("equal"):
                print(f"  {title:24s} {k:58s} equal  {v['new']}")
            else:
                print(f"  {title:24s} {k:58s} old {v['old']} new {v['new']} "
                      f"rel {v.get('max_rel_diff', '-')}")


def main():
    sys.path.insert(0, os.path.join(REPO, "figures"))
    peb = load_module("plot_eht_bbh", os.path.join(REPO, "figures", "plot_eht_bbh.py"))
    rc = load_module("r07_compare", os.path.join(REPO, "results", "blp", "raptor", "tools",
                                                 "r07_compare.py"))
    out = {"new_root": BLP_ROOT, "old_root": REPO,
           "eht_bbh": eht_set(peb), "amr": amr_set(), "r07": r07_set()}
    print(f"  {'set':24s} {'array':18s} {'':8s} {'n differ':>9s} {'max|d|/max':>11s} "
          f"{'rel L2':>11s} {'d sum/sum':>11s}")
    for name, e in out["eht_bbh"].items():
        fmt_arrays(name, e["arrays"])
    if out["amr"]:
        fmt_arrays("bbh_i90_pol_amr", out["amr"]["arrays"])
    for name, e in out["r07"].items():
        fmt_arrays(name, e["arrays"])
    print()
    for name, e in out["eht_bbh"].items():
        show(e["record"], ["flux_Jy", "mass_scale_g", "calibration.slope_dlnF_dlnms",
                           "calibration.polarized_secant", "ray_classes_cap1_cap2_esc_nmax_nonfinite",
                           "mean_samples", "stats_native.m_net_native",
                           "stats_native.mabs_avg_native", "stats_native.v_net_native",
                           "diagnostics.frac_flux_within_r2", "diagnostics.frac_flux_outside_fov",
                           "mdot_proxy.code"], name)
        if "image_metrics" in e:
            show(e["image_metrics"], ["m_net", "mabs_avg", "evpa_net_deg",
                                      "rings.hole1.native.diameter_uas",
                                      "rings.hole2.native.diameter_uas"], name + " metrics")
    if out["amr"]:
        show(out["amr"]["record"], ["amr_leg.nrays", "amr_leg.per_level_rays", "amr_leg.levels_hist",
                                    "levels_fine.tile_hist", "amr_leg.LP", "amr_leg.flux_I",
                                    "amr_leg.absV_over_I", "amr_leg.netV_over_I",
                                    "indicator.abs_grad_cut", "probe.classes_cap1_cap2_esc_nmax_nonfinite",
                                    "amr_leg.per_level_classes_cap1_cap2_esc_nmax_nonfinite",
                                    "amr_leg.per_level_mean_samples", "root_leg.LP"], "amr")
    for name, e in out["r07"].items():
        show(e["record"], ["flux_IQUV_Jy", "classes_cap1_cap2_esc_nmax_nonfinite", "samples",
                           "sample_safety.min_horizon_gap"], name)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1, default=float)
        f.write("\n")
    print("wrote", OUT_JSON)

    met = r07_metrics(rc)
    os.makedirs(os.path.dirname(MET_JSON), exist_ok=True)
    with open(MET_JSON, "w") as f:
        json.dump(met, f, indent=1, default=float)
    old = json.load(open(os.path.join(REPO, "results", "blp", "raptor", "evidence", "R07", "metrics.json")))
    print()
    for leg in ("ra", "rb"):
        for C, m in met["legs"].get(leg, {}).get("90", {}).get("levels", {}).items():
            o = old["legs"][leg]["90"]["levels"].get(C, {})
            print(f"  R07 {leg} i=90 C={C:>3}: dF/F {m['flux_rel']:+.3e} (old {o.get('flux_rel', float('nan')):+.3e}) "
                  f"L1(I) {m['l1_I']:.4e} (old {o.get('l1_I', float('nan')):.4e}) "
                  f"NMSE(I) {m['nmse_I']:.3e} (old {o.get('nmse_I', float('nan')):.3e}) "
                  f"L1(P) {m['l1_P']:.3e} (old {o.get('l1_P', float('nan')):.3e}) "
                  f"L1(V) {m['l1_V']:.3e} (old {o.get('l1_V', float('nan')):.3e}) "
                  f"<|dchi|> {m['evpa_pixel']['flux_weighted_mean_abs_deg']:.4g} "
                  f"(old {o.get('evpa_pixel', {}).get('flux_weighted_mean_abs_deg', float('nan')):.4g}) "
                  f"pass {m['pass']}")
    print("R07 metrics", "PASS" if met["pass"] else "FAIL", "| wrote", MET_JSON)


if __name__ == "__main__":
    main()
