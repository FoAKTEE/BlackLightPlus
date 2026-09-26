"""r10_compare.py — R10 comparison (dag.md R10) of the i = 90 AMR-figure deck
(figures/fig_bbh_i90_pol_amr.jl): RAPTOR uniform images against BlackLightPlus uniform
images at the same resolution, and against the BlackLightPlus AMR mosaic (re-rendered
at HEAD) block-averaged to that resolution; the refined zone at the finest AMR pixel.

Inputs (one run directory):
  blp_i90_C<C>_n<npix>/{I,Q,U,V,class}.npy, record.json     r10_blp.jl (fov 40)
  blp_i90_C80_n<nz>_fov20/...                                r10_blp.jl (fov 20 zone)
  r<leg>_i90_C<C>_n<npix>/img.npz, timing.json               r07_raptor.sh (fov 40)
  rbz_i90_C80_n<nz>/img.npz                                  r07_raptor.sh (fov 20 zone, leg b)
  amr/amr<eff>_{I,Q,U,V,levels_fine}.npy, amr/records.json   fig_bbh_i90_pol_amr.jl at HEAD
Legs: ra = RAPTOR's native polarized step, rb = the LD85 step (M11).

Uniform pairs use the r07_compare metrics (flux, L1/NMSE of I, |P|, V, m_net, <|m|>,
EVPA_net, v_net, ring diameters and f_C, class agreement); a hole's native ring diameter
and f_C are gated only where every ring cut has emission on both halves (ring_defined). The deck has no distance; the
blurred metrics take D = 16.8 Mpc (theta_g = 3.82 uas/M for M = 6.5e9 x 1.989e33 g) and
are code-to-code comparisons only. Leg (a) misses are attributed with each code's C = 80
-> 160 self-convergence (design.md §5), as in R07.

AMR mosaic (<eff>^2, unrefined blocks painted from their level) block-averaged to
<npix>^2 against the RAPTOR and BlackLightPlus uniform images: L1(I), NMSE(I), L1(|P|),
L1(V), LP = sum|P|/sum I, m_net, v_net. A uniform <npix>^2 image samples each pixel at its
centre, the block-averaged mosaic averages (eff/npix)^2 sub-pixels wherever the AMR
refined; on this edge-on image the two sampling rules differ by ~15% in L1(I) in either
code. The like-for-like comparison is the composite frame: the central 20 M x 20 M zone
(which contains every refined block) rendered uniformly at the finest AMR pixel
(nz = eff/2, pixel 40/eff M) and block-averaged, the uniform <npix>^2 image outside it;
the zone itself is also compared pixel by pixel with the mosaic crop (whole zone and the
finest-level pixels) and RAPTOR with BlackLightPlus.

Acceptance (dag.md R10): uniform leg (b): the R07 thresholds; leg (a): R07 thresholds or
every miss attributed to discretisation; RAPTOR (b) composite against the block-averaged
AMR mosaic: L1(I) <= 3% over the full frame, LP within 0.01, every refined block inside
the zone.
Usage: python3 r10_compare.py <rundir> <out.json> [--npix 1024]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r07_compare as rc  # noqa: E402

M_BH = 6.5e9 * 1.989e33
rc.L_UNIT = rc.G * M_BH / rc.C_CGS ** 2
rc.UAS_PER_M = rc.L_UNIT / rc.D_CM * (180 / np.pi) * 3600e6
rc.NU = 2.3e11
rc.FOV = 40.0
MBH_MSUN_RAPTOR = M_BH / rc.MSUN     # model.in MBH with the BlackLightPlus M_sun (M9)


def load_blp_dir(d):
    img = {c: np.load(os.path.join(d, f"{c}.npy")) for c in "IQUV"}
    img["class"] = np.load(os.path.join(d, "class.npy"))
    img["record"] = json.load(open(os.path.join(d, "record.json")))
    return img


def simple(X, Y):
    """X against the reference Y: flux, L1/NMSE of I, L1 of |P| and V, LP, m_net, v_net."""
    IX, IY = X["I"], Y["I"]
    PX, PY = np.hypot(X["Q"], X["U"]), np.hypot(Y["Q"], Y["U"])

    def pol(Z, P):
        s = Z["I"].sum()
        return {"LP": float(P.sum() / s), "m_net": float(abs((Z["Q"] + 1j * Z["U"]).sum()) / s),
                "v_net": float(Z["V"].sum() / s),
                "evpa_net_deg": float(np.degrees(0.5 * np.angle((Z["Q"] + 1j * Z["U"]).sum())))}
    oX, oY = pol(X, PX), pol(Y, PY)
    return {"flux_rel": float(IX.sum() / IY.sum() - 1),
            "l1_I": float(np.abs(IX - IY).sum() / IY.sum()),
            "nmse_I": float(((IX - IY) ** 2).sum() / (IY ** 2).sum()),
            "l1_P": float(np.abs(PX - PY).sum() / PY.sum()),
            "l1_V": float(np.abs(X["V"] - Y["V"]).sum() / np.abs(Y["V"]).sum()),
            "x": oX, "ref": oY,
            "d_LP": oX["LP"] - oY["LP"], "d_m_net": oX["m_net"] - oY["m_net"],
            "d_v_net": oX["v_net"] - oY["v_net"],
            "d_evpa_net_deg": float(rc.wrap90(oX["evpa_net_deg"] - oY["evpa_net_deg"]))}


RING_FLOOR = 1e-6


def ring_defined(img, npix):
    """Per hole: every ring cut of figures/plot_eht_bbh.ring_metrics has emission on both
    halves (half-cut maximum >= RING_FLOOR x the image peak). Where a half-cut crosses
    emission-free sky its argmax lands on values many decades below the peak and the
    native ring diameter is undefined (the blurred rings are unaffected)."""
    peb = rc.peb
    rec = rc.pseudo_rec(img, 90, npix)
    c = peb.hole_centres(rec)
    _, _, _, px, _ = peb.geometry(rec)
    rmax = min(10.0, 0.5 * float(np.hypot(*(c[0] - c[1]))))
    r = np.arange(-rmax, rmax + 1e-9, px / 2)
    pk = img["I"].max()
    out = {}
    for k in range(2):
        lo = []
        for pa in np.radians(peb.RING_CUTS_DEG):
            prof = peb._sample(img["I"], rec, c[k][0] - r * np.sin(pa), c[k][1] + r * np.cos(pa))
            lo.append(min(prof[r < 0].max(), prof[r > 0].max()) / pk)
        out[f"hole{k + 1}"] = {"defined": bool(min(lo) >= RING_FLOOR), "min_half_cut_peak_rel": float(min(lo))}
    return out


def block_mean(a, f):
    n = a.shape[0] // f
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--npix", type=int, default=1024)
    ap.add_argument("--extra-npix", default="2048")
    args = ap.parse_args()
    run, npix = args.run, args.npix
    res = {"npix": npix, "fov_M": 40.0, "uas_per_M_at_16.8Mpc": rc.UAS_PER_M, "thresholds": rc.THRESH,
           "legs": {}, "amr": {}, "zone": {}, "extra": {}}
    ok = True
    # ------------------------------------------------ uniform: RAPTOR vs BlackLightPlus
    Cs = sorted(int(d.split("_C")[1].split("_")[0]) for d in os.listdir(run)
                if d.startswith("blp_i90_C") and d.endswith(f"_n{npix}"))
    blp = {C: load_blp_dir(os.path.join(run, f"blp_i90_C{C}_n{npix}")) for C in Cs}
    rc.CENTERS = blp[80]["record"]["centers"]
    res["blp_git_sha"] = blp[80]["record"]["git_sha"]
    oB = {C: rc.observables(blp[C], 90, npix) for C in Cs}
    for leg in ("rb", "ra"):
        ims = {C: rc.load_raptor(run, leg, 90, C, npix) for C in Cs}
        CL = [C for C in Cs if ims[C] is not None]
        if 80 not in CL:
            continue
        oR = {C: rc.observables(ims[C], 90, npix) for C in CL}
        epsR = rc.self_eps(oR, ims, CL) if len(CL) > 1 else {C: {} for C in CL}
        epsB = rc.self_eps(oB, blp, CL) if len(CL) > 1 else {C: {} for C in CL}
        levels = {}
        for C in CL:
            m, _, _ = rc.pair_metrics(ims[C], blp[C], 90, npix, oR[C], oB[C])
            g = rc.gate(m, 90)
            m["ring_defined_blp"] = ring_defined(blp[C], npix)
            for k in (1, 2):
                if not m["ring_defined_blp"][f"hole{k}"]["defined"]:
                    for key in (f"hole{k}_native_diameter", f"hole{k}_native_f_C"):
                        m.setdefault("gate_not_applicable", {})[key] = g.pop(key)
            m["gate"] = g
            m["pass"] = all(g.values())
            m["attribution"] = rc.attribute(m, g, epsR[C], epsB[C])
            m["eps_raptor"], m["eps_blp"] = epsR[C], epsB[C]
            m["observables_raptor"], m["observables_blp"] = oR[C], oB[C]
            t = ims[C]["timing"]
            m["raptor_wall_s"] = t["wall_best_s"] if t else None
            m["raptor_phases_s"] = t.get("phases_best") if t else None
            m["raptor_maxrss_GB"] = t["maxrss_GB"] if t else None
            m["blp_wall_s"] = blp[C]["record"]["image_wall_s"]
            m["blp_record"] = {k: blp[C]["record"][k] for k in ("trace_s", "transfer_s", "rays_per_s", "samples",
                                                                "classes_cap1_cap2_esc_nmax_nonfinite", "maxrss_GB",
                                                                "load_s", "threads", "LP")}
            if ims[C]["nsamples"] is not None:
                m["samples_raptor"] = int(ims[C]["nsamples"].sum())
            # the emission-free tail: pixels where one code is below 1e-60 of the peak
            IR, IBc = ims[C]["I"], blp[C]["I"]
            pk = IBc.max()
            zb, zr = IBc < 1e-60 * pk, IR < 1e-60 * pk
            m["tail"] = {"blp_empty_pixels": int(zb.sum()),
                         "raptor_above_1e-30_there": int((IR[zb] > 1e-30 * pk).sum()),
                         "raptor_max_there_rel_peak": float(IR[zb].max() / pk) if zb.any() else 0.0,
                         "raptor_flux_there_rel": float(IR[zb].sum() / IBc.sum()),
                         "raptor_empty_pixels": int(zr.sum()),
                         "blp_max_there_rel_peak": float(IBc[zr].max() / pk) if zr.any() else 0.0}
            levels[str(C)] = m
        m80 = levels["80"]
        model = [k for k, v in m80["attribution"].items() if v["class"] == "model"]
        passed = m80["pass"] if leg == "rb" else (m80["pass"] or not model)
        res["legs"][leg] = {"C_levels": CL, "levels": levels, "pass": bool(passed),
                            "failed_C80": [k for k, v in m80["gate"].items() if not v],
                            "model_attributed_C80": model}
        ok &= passed
    # ------------------------------------------------ AMR mosaic vs uniform
    ad = os.path.join(run, "amr")
    arec = json.load(open(os.path.join(ad, "records.json")))
    eff = arec["amr"]["eff_res"]
    A = {c: np.load(os.path.join(ad, f"amr{eff}_{c}.npy")) for c in "IQUV"}
    lev = np.load(os.path.join(ad, f"amr{eff}_levels_fine.npy"))
    f = eff // npix
    Ab = {c: block_mean(A[c], f) for c in "IQUV"}
    res["amr"]["record"] = {k: arec[k] for k in ("git_sha", "threads", "amr", "indicator")}
    res["amr"]["record"]["amr_leg"] = {k: arec["amr_leg"][k] for k in (
        "wall_s", "nrays", "rays_per_s", "per_level_rays", "per_level_wall_s", "LP", "levels_hist",
        "refined_block_share", "refined_flux_share", "area_flux_I", "per_level_nonfinite_rays")}
    res["amr"]["record"]["snapshot_load_s"] = arec["snapshot"]["load_s"]
    res["amr"]["record"]["probe_wall_s_jit_incl"] = arec["probe"]["wall_s_jit_incl"]
    res["amr"]["record"]["level0_repeat"] = arec["level0_repeat"]
    res["amr"]["LP_full_mosaic"] = float(np.hypot(A["Q"], A["U"]).sum() / A["I"].sum())
    Bu = blp[80]
    res["amr"]["vs_blp_uniform"] = simple(Ab, Bu)
    for leg in ("rb", "ra"):
        R = rc.load_raptor(run, leg, 90, 80, npix)
        if R is not None:
            res["amr"][f"{leg}_vs_amr"] = simple(R, Ab)
    # ------------------------------------------------ refined zone (fov 20) at the finest pixel
    nz = eff * 20 // 40
    lo = (eff - nz) // 2
    t = eff // lev.shape[0]                      # pixels per level-map tile
    L = np.kron(lev, np.ones((t, t)))
    outside = np.ones_like(L, dtype=bool)
    outside[lo:lo + nz, lo:lo + nz] = False
    res["amr"]["refined_inside_zone"] = bool((L[outside] == 0).all())
    crop = {c: A[c][lo:lo + nz, lo:lo + nz] for c in "IQUV"}
    fine = L[lo:lo + nz, lo:lo + nz] == arec["amr"]["max_level"]
    zd = os.path.join(run, f"blp_i90_C80_n{nz}_fov20")
    rc.FOV = 20.0
    Rz = rc.load_raptor(run, "rbz", 90, 80, nz)
    rc.FOV = 40.0
    Wz = load_blp_dir(zd) if os.path.isdir(zd) else None
    z = res["zone"]
    z["fov_M"] = 20.0
    z["n"] = nz
    z["finest_level_pixel_share"] = float(fine.mean())
    if Wz is not None:
        z["blp_vs_amr_crop"] = simple(Wz, crop)
        z["blp_wall_s"] = Wz["record"]["image_wall_s"]
    if Rz is not None:
        z["rbz_vs_amr_crop"] = simple(Rz, crop)
        z["rbz_vs_amr_crop_finest_only"] = {
            "l1_I": float(np.abs(Rz["I"][fine] - crop["I"][fine]).sum() / crop["I"][fine].sum())}
        if Wz is not None:
            z["rbz_vs_blp"] = simple(Rz, Wz)
        if Rz["timing"]:
            z["rbz_wall_s"] = Rz["timing"]["wall_best_s"]
    # composite frames: the zone block-averaged into the central npix/2 pixels of the
    # uniform npix^2 image
    f2 = nz // (npix // 2)
    q = npix // 4

    def composite(U, Z):
        out = {c: U[c].copy() for c in "IQUV"}
        for c in "IQUV":
            out[c][q:q + npix // 2, q:q + npix // 2] = block_mean(Z[c], f2)
        return out
    rb = None
    Ru = rc.load_raptor(run, "rb", 90, 80, npix)
    if Rz is not None and Ru is not None:
        rb = simple(composite(Ru, Rz), Ab)
        res["amr"]["rb_composite_vs_amr"] = rb
    if Wz is not None:
        res["amr"]["blp_composite_vs_amr"] = simple(composite(Bu, Wz), Ab)
    amr_ok = (rb is not None and res["amr"]["refined_inside_zone"] and rb["l1_I"] <= 0.03
              and abs(rb["d_LP"]) <= 0.01)
    res["amr"]["pass"] = bool(amr_ok)
    res["amr"]["recorded_LP_bb233b2"] = 0.843
    ok &= amr_ok
    # ------------------------------------------------ extra resolution (report)
    for n2 in [int(x) for x in args.extra_npix.split(",") if x]:
        d2 = os.path.join(run, f"blp_i90_C80_n{n2}")
        if not os.path.isdir(d2):
            continue
        B2 = load_blp_dir(d2)
        e = {"blp_wall_s": B2["record"]["image_wall_s"], "blp_rays_per_s": B2["record"]["rays_per_s"]}
        for leg in ("rb", "ra"):
            R2 = rc.load_raptor(run, leg, 90, 80, n2)
            if R2 is None:
                continue
            e[f"{leg}_vs_blp"] = simple(R2, B2)
            e[f"{leg}_wall_s"] = R2["timing"]["wall_best_s"]
        if eff % n2 == 0:
            A2 = {c: block_mean(A[c], eff // n2) for c in "IQUV"}
            e["amr_vs_blp"] = simple(A2, B2)
            e["blp_vs_blp1024_upsampled_note"] = "resolution change of BlackLightPlus alone below"
            e["blp1024_vs_blp_blockavg"] = simple(blp[80], {c: block_mean(B2[c], n2 // npix) for c in "IQUV"}) \
                if n2 % npix == 0 else None
        res["extra"][str(n2)] = e
    res["pass"] = bool(ok)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    for leg, d in res["legs"].items():
        for C, m in d["levels"].items():
            print(f"{leg} C={C}: dF {m['flux_rel']:+.2e} L1 {m['l1_I']:.2e} NMSE {m['nmse_I']:.2e} L1P {m['l1_P']:.2e} "
                  f"L1V {m['l1_V']:.2e} dm {m['d_m_net']:+.4f} dEVPA {m['d_evpa_net_deg']:+.2f} cls {m['class_agreement']:.6f} "
                  f"| {'PASS' if m['pass'] else 'miss ' + ','.join(k for k, v in m['gate'].items() if not v)}"
                  f" | RAPTOR {m['raptor_wall_s']} s BLP {m['blp_wall_s']:.1f} s")
        print(f"{leg}: {'PASS' if d['pass'] else 'FAIL'} (model-attributed at C=80: {d['model_attributed_C80']})")
    for k, v in res["amr"].items():
        if isinstance(v, dict) and "l1_I" in v:
            print(f"AMR {k}: L1 {v['l1_I']:.3e} NMSE {v['nmse_I']:.2e} L1P {v['l1_P']:.2e} L1V {v['l1_V']:.2e} "
                  f"LP {v['x']['LP']:.4f} vs {v['ref']['LP']:.4f}")
    for k, v in res["zone"].items():
        if isinstance(v, dict) and "flux_rel" in v:
            print(f"zone {k}: L1 {v['l1_I']:.3e} L1P {v['l1_P']:.2e} LP {v['x']['LP']:.4f} vs {v['ref']['LP']:.4f}")
    print("R10 compare", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
