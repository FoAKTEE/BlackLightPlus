"""r07_compare.py — R07 image comparison of RAPTOR (fork blp-bbh) and BlackLightPlus on
s455 (dag.md R07; design.md §5).

Inputs (one run directory, written by r07_blp.jl and r07_raptor.sh):
  blp_i<inc>_C<C>_n<npix>/{I,Q,U,V,class}.npy, record.json   BlackLightPlus
  <leg>_i<inc>_C<C>_n<npix>/img.npz, timing.json              RAPTOR legs
     ra = (a) matched, native polarized step; rb = (b) matched + LD85 step (M11);
     rc = (c) upstream emission fits; rstock = upstream fits, constants, plasma decode,
     polarized radius 50 (report only)

Per image pair: total flux I/Q/U/V; flux difference; normalized L1 and NMSE of I;
L1 and NMSE of |P| = |Q + iU| and of V; EVPA difference (net, and per pixel where
|P|/I > 1% and I > 1% of its peak: flux-weighted mean and percentiles); m_net, <|m|>,
v_net, beta_2 (20 uas blur, EHT definitions, and native) with figures/plot_eht_bbh.py;
per-hole ring diameter and f_C (native and 20 uas blur) about each code's own shadow
centroid; ray termination class agreement.

Per code and observable X, the self-convergence error at step constant C is
eps(C) = |X(C) - X(C_f)| against the finest level C_f, and eps(C_f) = |X(C_f/2) - X(C_f)|
(for image norms, the same with L1/NMSE of the images). A code-to-code difference with
|X_R - X_B| <= 2 max(eps_R, eps_B) is discretisation (design.md §5), else it is a
model/implementation difference. The flux is also compared after Richardson
extrapolation of each code's C-ladder (the three finest levels).

Acceptance (dag.md R07), evaluated at C = 160 and at the finest level of each leg:
  |dF/F| <= 1% (<= 0.5% after Richardson); L1(I) <= 3%; NMSE(I) <= 1e-3;
  |d m_net| <= 0.01; |d <|m|>| <= 0.01; |d EVPA_net| <= 2 deg; |d v_net| <= 0.002;
  per-hole ring diameter within 0.5 uas (native and 20 uas blur); |d f_C| <= 0.02;
  i = 90: the pair is flagged unresolved in both codes.
A threshold miss that the attribution rule classes as discretisation is reported as such.

Usage: python3 r07_compare.py <rundir> <out.json> [--incs 17,60,90] [--npix 480]
       [--legs ra,rb] [--report-legs rc,rstock]
Exit 0 iff leg rb passes at C = 160 and at the finest level for every inclination and
every leg (a) threshold miss is attributed to discretisation.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "figures"))
import plot_eht_bbh as peb  # noqa: E402

C_CGS = 2.99792458e10
MSUN = 1.98847e33
G = 6.6743e-8
M_BH = 2 * 6.5e9 * MSUN
L_UNIT = G * M_BH / C_CGS ** 2
D_CM = 16.8e6 * 3.0856775814913673e18
UAS_PER_M = L_UNIT / D_CM * (180 / np.pi) * 3600e6
FOV = 56.0
NU = 230e9
CENTERS = None  # from the BlackLightPlus record

THRESH = {"flux_rel": 0.01, "flux_rel_richardson": 0.005, "l1_I": 0.03, "nmse_I": 1e-3,
          "m_net": 0.01, "mabs_avg": 0.01, "evpa_net_deg": 2.0, "v_net": 0.002,
          "ring_diameter_uas": 0.5, "f_C": 0.02}


def pixsr(npix):
    return (FOV / npix * L_UNIT / D_CM) ** 2


def load_blp(run, inc, C, npix):
    d = os.path.join(run, f"blp_i{inc:02d}_C{C}_n{npix}")
    if not os.path.isdir(d):
        return None
    img = {c: np.load(os.path.join(d, f"{c}.npy")) for c in "IQUV"}
    img["class"] = np.load(os.path.join(d, "class.npy"))
    with open(os.path.join(d, "record.json")) as f:
        img["record"] = json.load(f)
    return img


def load_raptor(run, leg, inc, C, npix):
    d = os.path.join(run, f"{leg}_i{inc:02d}_C{C}_n{npix}")
    p = os.path.join(d, "img.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    s = pixsr(npix) / 1e-23                     # Jy per pixel -> cgs intensity
    img = {c: z["IQUV"][0, k] / s for k, c in enumerate("IQUV")}
    img["class"] = z["endclass"] if "endclass" in z.files else np.zeros_like(img["I"])
    img["nsamples"] = z["nsamples"] if "nsamples" in z.files else None
    img["tauF"] = next((z[k] for k in z.files if k.startswith("tauF")), None)
    tj = os.path.join(d, "timing.json")
    img["timing"] = json.load(open(tj)) if os.path.isfile(tj) else None
    return img


def project(c, inc):
    return [c[1], -c[0] * np.cos(np.radians(inc)) + c[2] * np.sin(np.radians(inc))]


def pseudo_rec(img, inc, npix):
    """The record fields figures/plot_eht_bbh.metrics reads, from one code's image."""
    ps = FOV / npix
    ax = -FOV / 2 + (np.arange(npix) + 0.5) * ps
    A, B = np.meshgrid(ax, ax)
    sh = {}
    for k in (1, 2):
        m = img["class"] == k
        pr = project(CENTERS[k - 1], inc)
        sh[f"hole{k}"] = ({"npix": int(m.sum()), "centroid_M": [float(A[m].mean()), float(B[m].mean())],
                           "projected_M": pr} if m.any() else {"npix": 0, "projected_M": pr})
    return {"camera": {"res": npix, "fov_M": FOV}, "scaling": {"theta_g_uas": UAS_PER_M},
            "frequency_hz": NU, "flux_Jy": float(img["I"].sum() * pixsr(npix) / 1e-23),
            "projected_holes_M": [project(c, inc) for c in CENTERS], "shadow_check": sh}


def observables(img, inc, npix):
    """Scalar observables of one image (the plot_eht_bbh metrics plus fluxes)."""
    rec = pseudo_rec(img, inc, npix)
    met, _ = peb.metrics(rec, img)
    f = pixsr(npix) / 1e-23
    I, Q, U, V = (img[c] for c in "IQUV")
    P = Q + 1j * U
    o = {"flux_I_Jy": float(I.sum() * f), "flux_Q_Jy": float(Q.sum() * f),
         "flux_U_Jy": float(U.sum() * f), "flux_V_Jy": float(V.sum() * f),
         "m_net": met["m_net"], "mabs_avg": met["mabs_avg"], "v_net": met["v_net"],
         "evpa_net_deg": met["evpa_net_deg"], "beta2_amp": met["beta2_amp"],
         "beta2_arg_deg": met["beta2_arg_deg"],
         "native": dict(met["native"], evpa_net_deg=float(np.degrees(0.5 * np.angle(P.sum())))),
         "resolved_pair": met["rings"]["resolved_pair"],
         "ring_centre_separation_M": met["rings"]["centre_separation_M"]}
    for k in (1, 2):
        h = met["rings"][f"hole{k}"]
        for lab in ("native", "blur20"):
            o[f"hole{k}_{lab}_diameter_uas"] = h[lab]["diameter_uas"]
            o[f"hole{k}_{lab}_f_C"] = h[lab]["f_C"]
    return o


def wrap90(d):
    return (np.asarray(d) + 90.0) % 180.0 - 90.0


def evpa_pixel(R, B):
    """Per-pixel EVPA difference where |P|/I > 1% and I > 1% of the peak (both images)."""
    IR, IB = R["I"], B["I"]
    PR, PB = np.hypot(R["Q"], R["U"]), np.hypot(B["Q"], B["U"])
    sel = (IB > 0.01 * IB.max()) & (PB > 0.01 * IB) & (PR > 0.01 * IR)
    dchi = wrap90(np.degrees(0.5 * (np.angle(R["Q"] + 1j * R["U"]) - np.angle(B["Q"] + 1j * B["U"]))))
    w = IB[sel]
    if not sel.any():
        return {"n": 0, "flux_weighted_mean_abs_deg": None, "median_abs_deg": None, "p90_abs_deg": None}
    return {"n": int(sel.sum()),
            "flux_weighted_mean_abs_deg": float((np.abs(dchi[sel]) * w).sum() / w.sum()),
            "median_abs_deg": float(np.median(np.abs(dchi[sel]))),
            "p90_abs_deg": float(np.percentile(np.abs(dchi[sel]), 90))}


def pair_metrics(R, B, inc, npix, oR=None, oB=None):
    """Image-level comparison R vs B (B is the reference) and observable differences."""
    oR = oR or observables(R, inc, npix)
    oB = oB or observables(B, inc, npix)
    IR, IB = R["I"], B["I"]
    PR, PB = np.hypot(R["Q"], R["U"]), np.hypot(B["Q"], B["U"])
    m = {"flux_rel": oR["flux_I_Jy"] / oB["flux_I_Jy"] - 1,
         "l1_I": float(np.abs(IR - IB).sum() / IB.sum()),
         "nmse_I": float(((IR - IB) ** 2).sum() / (IB ** 2).sum()),
         "l1_P": float(np.abs(PR - PB).sum() / PB.sum()),
         "nmse_P": float(((PR - PB) ** 2).sum() / (PB ** 2).sum()),
         "l1_V": float(np.abs(R["V"] - B["V"]).sum() / np.abs(B["V"]).sum()),
         "nmse_V": float(((R["V"] - B["V"]) ** 2).sum() / (B["V"] ** 2).sum())}
    for c in "QUV":
        m[f"flux_{c}_diff_Jy"] = oR[f"flux_{c}_Jy"] - oB[f"flux_{c}_Jy"]
    m["evpa_pixel"] = evpa_pixel(R, B)
    for k in ("m_net", "mabs_avg", "v_net", "beta2_amp"):
        m[f"d_{k}"] = oR[k] - oB[k]
    m["d_evpa_net_deg"] = float(wrap90(oR["evpa_net_deg"] - oB["evpa_net_deg"]))
    m["d_beta2_arg_deg"] = float((oR["beta2_arg_deg"] - oB["beta2_arg_deg"] + 180) % 360 - 180)
    m["native"] = {k: oR["native"][k] - oB["native"][k] for k in ("m_net", "mabs_avg", "v_net", "beta2_amp")}
    m["native"]["evpa_net_deg"] = float(wrap90(oR["native"]["evpa_net_deg"] - oB["native"]["evpa_net_deg"]))
    for k in (1, 2):
        for lab in ("native", "blur20"):
            m[f"d_hole{k}_{lab}_diameter_uas"] = oR[f"hole{k}_{lab}_diameter_uas"] - oB[f"hole{k}_{lab}_diameter_uas"]
            m[f"d_hole{k}_{lab}_f_C"] = oR[f"hole{k}_{lab}_f_C"] - oB[f"hole{k}_{lab}_f_C"]
    cR, cB = R["class"], B["class"]
    m["class_agreement"] = float((cR == cB).mean())
    m["class_mismatch_pixels"] = int((cR != cB).sum())
    m["resolved_pair"] = [oR["resolved_pair"], oB["resolved_pair"]]
    return m, oR, oB


def gate(m, inc):
    """dag.md R07 thresholds on one pair; returns {check: pass}."""
    g = {"flux_rel": abs(m["flux_rel"]) <= THRESH["flux_rel"],
         "l1_I": m["l1_I"] <= THRESH["l1_I"], "nmse_I": m["nmse_I"] <= THRESH["nmse_I"],
         "m_net": abs(m["d_m_net"]) <= THRESH["m_net"],
         "mabs_avg": abs(m["d_mabs_avg"]) <= THRESH["mabs_avg"],
         "evpa_net_deg": abs(m["d_evpa_net_deg"]) <= THRESH["evpa_net_deg"],
         "v_net": abs(m["d_v_net"]) <= THRESH["v_net"]}
    for k in (1, 2):
        for lab in ("native", "blur20"):
            g[f"hole{k}_{lab}_diameter"] = abs(m[f"d_hole{k}_{lab}_diameter_uas"]) <= THRESH["ring_diameter_uas"]
            g[f"hole{k}_{lab}_f_C"] = abs(m[f"d_hole{k}_{lab}_f_C"]) <= THRESH["f_C"]
    if inc == 90:
        g["unresolved_both"] = (not m["resolved_pair"][0]) and (not m["resolved_pair"][1])
    return g


# observable key in pair_metrics -> key in observables (for self-convergence)
OBS_OF = {"flux_rel": "flux_I_Jy", "m_net": "m_net", "mabs_avg": "mabs_avg", "v_net": "v_net",
          "evpa_net_deg": "evpa_net_deg"}
for _k in (1, 2):
    for _lab in ("native", "blur20"):
        OBS_OF[f"hole{_k}_{_lab}_diameter"] = f"hole{_k}_{_lab}_diameter_uas"
        OBS_OF[f"hole{_k}_{_lab}_f_C"] = f"hole{_k}_{_lab}_f_C"


def obs_diff(key, a, b):
    if key == "evpa_net_deg":
        return abs(float(wrap90(a - b)))
    return abs(a - b)


def self_eps(obs_by_C, imgs_by_C, Cs):
    """eps per level for every observable and image norm of one code (see module doc)."""
    Cf = Cs[-1]
    out = {}
    for C in Cs:
        ref = Cs[-2] if C == Cf else Cf
        e = {k: obs_diff(k, obs_by_C[C][v], obs_by_C[ref][v]) for k, v in OBS_OF.items()}
        e["flux_rel"] = e["flux_rel"] / obs_by_C[Cf]["flux_I_Jy"]
        a, b = imgs_by_C[C], imgs_by_C[ref]
        e["l1_I"] = float(np.abs(a["I"] - b["I"]).sum() / b["I"].sum())
        e["nmse_I"] = float(((a["I"] - b["I"]) ** 2).sum() / (b["I"] ** 2).sum())
        pa, pb = np.hypot(a["Q"], a["U"]), np.hypot(b["Q"], b["U"])
        e["l1_P"] = float(np.abs(pa - pb).sum() / pb.sum())
        e["l1_V"] = float(np.abs(a["V"] - b["V"]).sum() / np.abs(b["V"]).sum())
        e["evpa_pixel_fw_deg"] = evpa_pixel(a, b)["flux_weighted_mean_abs_deg"]
        out[C] = e
    return out


def richardson(Cs, F):
    """Aitken/Richardson limit and observed order from the three finest levels (C doubling)."""
    if len(Cs) < 3:
        return None
    f1, f2, f3 = F[-3], F[-2], F[-1]
    d1, d2 = f2 - f1, f3 - f2
    if d1 == 0 or d2 == 0 or d1 * d2 <= 0:
        return {"limit": f3, "order": None, "note": "non-monotone: finest level used"}
    p = np.log2(d1 / d2)
    return {"limit": f3 + d2 / (2 ** p - 1), "order": float(p)}


def attribute(m, g, epsR, epsB):
    """design.md §5 per failed check: 'discretisation' if |X_R - X_B| <= 2 max(eps_R, eps_B)."""
    out = {}
    val = {"flux_rel": abs(m["flux_rel"]), "l1_I": m["l1_I"], "nmse_I": m["nmse_I"],
           "m_net": abs(m["d_m_net"]), "mabs_avg": abs(m["d_mabs_avg"]),
           "evpa_net_deg": abs(m["d_evpa_net_deg"]), "v_net": abs(m["d_v_net"])}
    for k in (1, 2):
        for lab in ("native", "blur20"):
            val[f"hole{k}_{lab}_diameter"] = abs(m[f"d_hole{k}_{lab}_diameter_uas"])
            val[f"hole{k}_{lab}_f_C"] = abs(m[f"d_hole{k}_{lab}_f_C"])
    for k, ok in g.items():
        if ok or k == "unresolved_both":
            continue
        tol = 2 * max(epsR.get(k, 0.0), epsB.get(k, 0.0))
        out[k] = {"difference": val[k], "eps_raptor": epsR.get(k), "eps_blp": epsB.get(k),
                  "two_max_eps": tol, "class": "discretisation" if val[k] <= tol else "model"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--incs", default="17,60,90")
    ap.add_argument("--npix", type=int, default=480)
    ap.add_argument("--legs", default="ra,rb")
    ap.add_argument("--report-legs", default="rc,rstock")
    ap.add_argument("--report-C", type=int, default=80)
    ap.add_argument("--res-C", type=int, default=80)
    ap.add_argument("--res", default="240,480,960")
    args = ap.parse_args()
    incs = [int(x) for x in args.incs.split(",")]
    legs = [x for x in args.legs.split(",") if x]
    global CENTERS
    res = {"thresholds": THRESH, "npix": args.npix, "fov_M": FOV, "uas_per_M": UAS_PER_M,
           "legs": {}, "report_legs": {}, "resolution": {}}
    allpass = True
    for inc in incs:
        Cs = sorted(int(d.split("_C")[1].split("_")[0]) for d in os.listdir(args.run)
                    if d.startswith(f"blp_i{inc:02d}_") and d.endswith(f"_n{args.npix}"))
        blp = {C: load_blp(args.run, inc, C, args.npix) for C in Cs}
        if CENTERS is None:
            CENTERS = blp[Cs[0]]["record"]["centers"]
            res["blp_git_sha"] = blp[Cs[0]]["record"]["git_sha"]
        oB = {C: observables(blp[C], inc, args.npix) for C in Cs}
        rich_B = richardson(Cs, [oB[C]["flux_I_Jy"] for C in Cs])
        res.setdefault("blp", {})[str(inc)] = {"observables": {str(C): oB[C] for C in Cs},
                                                "flux_richardson": rich_B}
        for leg in legs:
            ims = {C: load_raptor(args.run, leg, inc, C, args.npix) for C in Cs}
            CL = [C for C in Cs if ims[C] is not None]
            if len(CL) < 2:
                continue
            oR = {C: observables(ims[C], inc, args.npix) for C in CL}
            epsR = self_eps(oR, ims, CL)
            epsB = self_eps(oB, blp, CL)
            rich_R = richardson(CL, [oR[C]["flux_I_Jy"] for C in CL])
            rich_Bl = richardson(CL, [oB[C]["flux_I_Jy"] for C in CL])
            levels = {}
            for C in CL:
                m, _, _ = pair_metrics(ims[C], blp[C], inc, args.npix, oR[C], oB[C])
                g = gate(m, inc)
                m["gate"] = g
                m["pass"] = all(g.values())
                m["attribution"] = attribute(m, g, epsR[C], epsB[C])
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
            fr = None
            if rich_R and rich_Bl:
                fr = rich_R["limit"] / rich_Bl["limit"] - 1
            evalC = sorted({min(160, CL[-1]), CL[-1]})
            summary = {}
            for C in evalC:
                m = levels[str(C)]
                model_miss = [k for k, v in m["attribution"].items() if v["class"] == "model"]
                summary[str(C)] = {"pass": m["pass"], "failed": [k for k, v in m["gate"].items() if not v],
                                   "model_attributed": model_miss}
            res["legs"].setdefault(leg, {})[str(inc)] = {
                "C_levels": CL, "levels": levels, "flux_richardson_raptor": rich_R,
                "flux_richardson_blp": rich_Bl, "flux_rel_richardson": fr,
                "flux_richardson_pass": fr is not None and abs(fr) <= THRESH["flux_rel_richardson"],
                "evaluated": summary}
            if leg == "rb":
                allpass &= all(s["pass"] for s in summary.values())
            if leg == "ra":
                allpass &= all(not s["model_attributed"] for s in summary.values())
        for leg in [x for x in args.report_legs.split(",") if x]:
            C = args.report_C
            R = load_raptor(args.run, leg, inc, C, args.npix)
            if R is None or C not in blp:
                continue
            m, oRr, _ = pair_metrics(R, blp[C], inc, args.npix, None, oB[C])
            m["observables_raptor"] = oRr
            res["report_legs"].setdefault(leg, {})[str(inc)] = {"C": C, "vs_blp": m}
    # resolution legs at i = 17
    for leg in legs:
        for npix in [int(x) for x in args.res.split(",") if x]:
            B = load_blp(args.run, 17, args.res_C, npix)
            R = load_raptor(args.run, leg, 17, args.res_C, npix)
            if B is None or R is None:
                continue
            m, oRr, oBb = pair_metrics(R, B, 17, npix)
            m["gate"] = gate(m, 17)
            m["observables_raptor"] = oRr
            m["observables_blp"] = oBb
            res["resolution"].setdefault(leg, {})[str(npix)] = m
    res["pass"] = bool(allpass)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    for leg, d in res["legs"].items():
        for inc, e in d.items():
            for C, s in e["evaluated"].items():
                m = e["levels"][C]
                print(f"{leg} i={inc:>2} C={C:>3}: dF {m['flux_rel']:+.2e} L1 {m['l1_I']:.2e} NMSE {m['nmse_I']:.2e} "
                      f"dm {m['d_m_net']:+.4f} d<|m|> {m['d_mabs_avg']:+.4f} dEVPA {m['d_evpa_net_deg']:+.2f} "
                      f"dv {m['d_v_net']:+.5f} cls {m['class_agreement']:.5f} | "
                      f"{'PASS' if s['pass'] else 'miss ' + ','.join(s['failed'])} "
                      f"{'model: ' + ','.join(s['model_attributed']) if s['model_attributed'] else ''}")
            print(f"{leg} i={inc:>2} Richardson flux diff {e['flux_rel_richardson']}")
    print("R07 compare", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
