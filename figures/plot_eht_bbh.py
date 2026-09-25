"""EHT-style figures for the binary-black-hole 230 GHz mocks (output of fig_eht_bbh.jl).

Sky orientation: north (+Dec) up is the projected orbital angular momentum (image +beta);
east (+RA) is image -alpha and is drawn on the left, so the RA axis increases leftward.
EVPA chi = 0.5 arg(Q + iU) is east of north (IAU), the camera Stokes convention.

Blurring: circular Gaussian, FWHM 20 uas (EHT nominal resolution), applied to I, Q, U, V
(zero padding at the FOV edge; the flux lost off the edge is recorded).
Image metrics (blurred images, the EHT M87* polarimetry definitions):
  m_net = |sum(Q + iU)| / sum(I);  <|m|> = sum(|Q + iU|) / sum(I);  v_net = sum(V) / sum(I);
  beta_m = sum(P exp(-i m phi)) / sum(I), P = Q + iU, phi = position angle east of north
  about the image centre (the binary centre of mass), over the whole FOV (Palumbo+2020
  beta_2 with an unrestricted annulus). Per-hole beta_2 about each projected puncture
  within 8 M is added where the two 8 M discs do not overlap.

Rings: per hole, cuts at position angles 0-150 deg through the shadow centroid (captured
rays; the projected puncture if none); diameter = median peak-to-peak separation; f_C =
mean I within 5 uas of the centre over the mean I on the circle of that diameter, on the
native, 10 uas and 20 uas images. Pairs closer than 8 M in projection are flagged.

Writes figures/examples/eht_bbh/fig{1..4}_*.png and .pdf, adds a "metrics" block to
each image's records.json, and writes figures/out/eht_bbh/metrics_summary.json.
Usage: python3 figures/plot_eht_bbh.py   [BLP_OUT=<run dir>] [BLP_FIG=<figure dir>]
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter, map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLP_OUT", os.path.join(HERE, "out", "eht_bbh"))
FIG = os.environ.get("BLP_FIG", os.path.join(HERE, "examples", "eht_bbh"))

BEAM_UAS = 20.0
C_CGS = 2.99792458e10
K_B = 1.380649e-16
TICK_SPACING_UAS = 7.0
TICK_ICUT = 0.10          # EVPA ticks only where blurred I > 10% of its peak
PER_HOLE_R_M = 8.0
LIM1_UAS = 170.0          # fig 1 / 3 / 4 panel half-width (the render FOV is +-214 uas)
LIM2_UAS = 160.0          # fig 2 / 3 / 4 panel half-width
NG_BEAM_UAS = 10.0        # ngEHT-like resolution, the second blur in fig 2
RING_CUTS_DEG = np.arange(0, 180, 30)   # cut position angles through each hole
FC_DISC_UAS = 5.0         # f_C: mean inside this radius over the mean on the ring

SNAP_LABEL = {
    "s455": "00455  sep 25, a=0 (zoom-in SANE)",
    "s093": "00093  sep 25, a=0.8",
    "s122": "00122  sep 16, a=0.8",
}


def load(name):
    d = os.path.join(OUT, name)
    with open(os.path.join(d, "records.json")) as f:
        rec = json.load(f)
    img = {c: np.load(os.path.join(d, f"{c}.npy")) for c in ("I", "Q", "U", "V", "class")}
    return rec, img


def geometry(rec):
    cam = rec["camera"]
    res, fov = cam["res"], cam["fov_M"]
    uas = rec["scaling"]["theta_g_uas"]
    px_m = fov / res
    ax_m = -fov / 2 + (np.arange(res) + 0.5) * px_m          # pixel centres (alpha or beta)
    return res, fov, uas, px_m, ax_m


def blur(a, rec, fwhm=BEAM_UAS):
    _, _, uas, px_m, _ = geometry(rec)
    sig_px = fwhm / (2 * np.sqrt(2 * np.log(2))) / (px_m * uas)
    return gaussian_filter(a, sig_px, mode="constant", cval=0.0, truncate=5.0)


def tb(I_cgs, rec):
    nu = rec["frequency_hz"]
    return I_cgs * C_CGS ** 2 / (2 * K_B * nu ** 2)


def beta_m(P, I, phi, mask, m):
    return np.sum(P[mask] * np.exp(-1j * m * phi[mask])) / np.sum(I[mask])


def metrics(rec, img):
    res, fov, uas, px_m, ax_m = geometry(rec)
    A, B = np.meshgrid(ax_m, ax_m)                            # A = alpha, B = beta ([ib, ia])
    phi = np.arctan2(-A, B)                                   # position angle east of north
    Ib, Qb, Ub, Vb = (blur(img[c], rec) for c in "IQUV")
    Pb = Qb + 1j * Ub
    I, Q, U, V = (img[c] for c in "IQUV")
    P = Q + 1j * U
    full = np.ones_like(I, dtype=bool)
    b2 = beta_m(Pb, Ib, phi, full, 2)
    b2n = beta_m(P, I, phi, full, 2)
    out = {
        "blur_fwhm_uas": BEAM_UAS,
        "flux_native_Jy": rec["flux_Jy"],
        "flux_blurred_Jy": rec["flux_Jy"] * Ib.sum() / I.sum(),
        "m_net": float(abs(Pb.sum()) / Ib.sum()),
        "mabs_avg": float(np.abs(Pb).sum() / Ib.sum()),
        "v_net": float(Vb.sum() / Ib.sum()),
        "beta2_amp": float(abs(b2)),
        "beta2_arg_deg": float(np.degrees(np.angle(b2))),
        "evpa_net_deg": float(np.degrees(0.5 * np.angle(Pb.sum()))),
        "native": {
            "m_net": float(abs(P.sum()) / I.sum()),
            "mabs_avg": float(np.abs(P).sum() / I.sum()),
            "v_net": float(V.sum() / I.sum()),
            "beta2_amp": float(abs(b2n)),
            "beta2_arg_deg": float(np.degrees(np.angle(b2n))),
        },
        "Tb_peak_native_K": float(tb(I.max(), rec)),
        "Tb_peak_blurred_K": float(tb(Ib.max(), rec)),
        "definition": "blurred 20 uas; beta_2 about the image centre over the full FOV, phi east of north",
    }
    holes = rec["projected_holes_M"]
    dproj = np.hypot(holes[0][0] - holes[1][0], holes[0][1] - holes[1][1])
    if dproj > 2 * PER_HOLE_R_M:
        ph = {}
        for k, (ha, hb) in enumerate(holes):
            m = np.hypot(A - ha, B - hb) < PER_HOLE_R_M
            phk = np.arctan2(-(A - ha), B - hb)
            bk = beta_m(Pb, Ib, phk, m, 2)
            ph[f"hole{k + 1}"] = {"beta2_amp": float(abs(bk)),
                                  "beta2_arg_deg": float(np.degrees(np.angle(bk))),
                                  "m_net": float(abs(Pb[m].sum()) / Ib[m].sum()),
                                  "flux_share": float(Ib[m].sum() / Ib.sum())}
        out["per_hole_r8M"] = ph
    I10 = blur(I, rec, NG_BEAM_UAS)
    out["rings"] = rings(rec, I, I10, Ib)
    return out, (Ib, Qb, Ub, Vb, I10)


def _sample(img, rec, a_m, b_m):
    """Bilinear I at image coordinates (alpha, beta) [M]."""
    res, fov, _, px_m, _ = geometry(rec)
    ia = (np.asarray(a_m) + fov / 2) / px_m - 0.5
    ib = (np.asarray(b_m) + fov / 2) / px_m - 0.5
    return map_coordinates(img, [ib, ia], order=1, mode="constant", cval=0.0)


def hole_centres(rec):
    """Shadow centroid of each hole (captured-ray mask), else the projected puncture."""
    out = []
    for k in (1, 2):
        h = rec["shadow_check"][f"hole{k}"]
        out.append(h["centroid_M"] if h.get("npix", 0) > 0 else h["projected_M"])
    return np.array(out, dtype=float)


def ring_metrics(img, rec, centre, rmax_m):
    """Ring diameter from peak-to-peak separations on cuts through `centre` and the
    fractional central brightness f_C = <I>(r < 5 uas) / <I>(ring at r = d/2)."""
    _, _, uas, px_m, _ = geometry(rec)
    r = np.arange(-rmax_m, rmax_m + 1e-9, px_m / 2)
    diams, offc = [], []
    for pa in np.radians(RING_CUTS_DEG):
        # position angle east of north: east = -alpha
        a = centre[0] - r * np.sin(pa)
        b = centre[1] + r * np.cos(pa)
        prof = _sample(img, rec, a, b)
        neg, pos = r < 0, r > 0
        rl = r[neg][np.argmax(prof[neg])]
        rr = r[pos][np.argmax(prof[pos])]
        diams.append(rr - rl)
        offc.append(min(-rl, rr))
    d = float(np.median(diams))
    th = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    ring = _sample(img, rec, centre[0] + d / 2 * np.cos(th), centre[1] + d / 2 * np.sin(th)).mean()
    rd = FC_DISC_UAS / uas
    gr = np.arange(-rd, rd + 1e-9, px_m / 2)
    ga, gb = np.meshgrid(gr, gr)
    m = ga ** 2 + gb ** 2 <= rd ** 2
    cen = _sample(img, rec, centre[0] + ga[m], centre[1] + gb[m]).mean()
    fc = float(cen / ring) if ring > 0 else float("nan")
    return {"diameter_M": d, "diameter_uas": d * uas,
            "diameter_cut_spread_uas": float(np.std(diams) * uas),
            "f_C": fc, "min_peak_offset_M": float(np.min(offc)),
            "depression": bool(fc < 1.0 and np.median(offc) > 0.25 * d / 2)}


def rings(rec, img_native, img_10, img_20):
    """Per-hole ring diameter and f_C on the native, 10 uas and 20 uas images."""
    c = hole_centres(rec)
    dsep = float(np.hypot(*(c[0] - c[1])))
    rmax = min(10.0, 0.5 * dsep)
    out = {"centres_M": c.tolist(), "centre_separation_M": dsep, "cut_half_length_M": rmax,
           "cuts_pa_deg": RING_CUTS_DEG.tolist(), "fc_disc_uas": FC_DISC_UAS,
           "resolved_pair": bool(dsep > 8.0)}
    for k in range(2):
        out[f"hole{k + 1}"] = {lab: ring_metrics(im, rec, c[k], rmax) for lab, im in
                               (("native", img_native), ("blur10", img_10), ("blur20", img_20))}
    return out


def extent_uas(rec):
    _, fov, uas, _, _ = geometry(rec)
    h = fov / 2 * uas
    return [h, -h, -h, h]                   # RA (= -alpha) increases to the left


def draw_beam(ax, rec, color="w", fwhm=BEAM_UAS, lim=None):
    _, fov, uas, _, _ = geometry(rec)
    h = fov / 2 * uas if lim is None else lim
    ax.add_patch(Circle((h - 0.1 * h, -h + 0.1 * h), fwhm / 2, fill=False, ec=color, lw=1.0))


def set_lim(ax, lim):
    ax.set_xlim(lim, -lim)
    ax.set_ylim(-lim, lim)


def mark_holes(ax, rec, color="c"):
    uas = rec["scaling"]["theta_g_uas"]
    for ha, hb in rec["projected_holes_M"]:
        ax.plot(-ha * uas, hb * uas, "+", color=color, ms=6, mew=0.8, alpha=0.8)


def style_axes(ax, left=True, bottom=True):
    ax.set_xlabel("Relative RA (µas)" if bottom else "")
    ax.set_ylabel("Relative Dec (µas)" if left else "")
    if not bottom:
        ax.set_xticklabels([])
    if not left:
        ax.set_yticklabels([])
    ax.tick_params(direction="in", color="0.7", length=3)


def evpa_ticks(ax, rec, Ib, Qb, Ub, cmap, mnorm, lw=1.3):
    res, fov, uas, px_m, ax_m = geometry(rec)
    step = max(1, int(round(TICK_SPACING_UAS / (px_m * uas))))
    P = np.hypot(Qb, Ub)
    Pmax = P[Ib > TICK_ICUT * Ib.max()].max()
    segs, cols = [], []
    for ib in range(step // 2, res, step):
        for ia in range(step // 2, res, step):
            if Ib[ib, ia] <= TICK_ICUT * Ib.max():
                continue
            chi = 0.5 * np.arctan2(Ub[ib, ia], Qb[ib, ia])
            L = 0.95 * step * px_m * uas * P[ib, ia] / Pmax
            ra, dec = -ax_m[ia] * uas, ax_m[ib] * uas
            dra, ddec = 0.5 * L * np.sin(chi), 0.5 * L * np.cos(chi)   # east of north
            segs.append([(ra - dra, dec - ddec), (ra + dra, dec + ddec)])
            cols.append(P[ib, ia] / Ib[ib, ia])
    lc = LineCollection(segs, colors=cmap(mnorm(np.array(cols))), linewidths=lw,
                        capstyle="round")
    ax.add_collection(lc)


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG, stem + ".png"))


def main():
    with open(os.path.join(OUT, "index.json")) as f:
        index = json.load(f)
    names = [e["name"] for e in index["images"]]
    recs, imgs, mets, blurred = {}, {}, {}, {}
    summary = {}
    for n in names:
        rec, img = load(n)
        met, bl = metrics(rec, img)
        rec["metrics"] = met
        with open(os.path.join(OUT, n, "records.json"), "w") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
        recs[n], imgs[n], mets[n], blurred[n] = rec, img, met, bl
        summary[n] = {"flux_Jy": rec["flux_Jy"], "mass_scale_g": rec["mass_scale_g"],
                      "inc": rec["camera"]["inclination_deg"], "R_high": rec["plasma"]["R_high"],
                      "time_M": rec["snapshot"]["time_M"], **{k: met[k] for k in (
                          "m_net", "mabs_avg", "v_net", "beta2_amp", "beta2_arg_deg",
                          "flux_blurred_Jy", "Tb_peak_blurred_K")}, "rings": met["rings"]}
        print(f"{n:18s} F={rec['flux_Jy']:.4f} ms={rec['mass_scale_g']:.3e} "
              f"m_net={met['m_net']:.4f} <|m|>={met['mabs_avg']:.4f} "
              f"|b2|={met['beta2_amp']:.4f} arg={met['beta2_arg_deg']:7.1f} V/I={met['v_net']:+.2e}")
        rg = met["rings"]
        for h in ("hole1", "hole2"):
            print("     " + h + " " + "  ".join(
                f"{lab}: d={rg[h][lab]['diameter_uas']:5.1f}uas fC={rg[h][lab]['f_C']:.2f}"
                f"{'' if rg[h][lab]['depression'] else ' (no dip)'}"
                for lab in ("native", "blur10", "blur20")) + ("" if rg["resolved_pair"] else "  [pair overlaps]"))
    with open(os.path.join(OUT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})
    snaps = [s for s in ("s455", "s093", "s122") if any(n.startswith(s + "_") for n in names)]
    incs = sorted({int(round(recs[n]["camera"]["inclination_deg"])) for n in names})
    rh_fid = 40

    # -------------------------------------------------- fig 1: snapshot x inclination grid
    fig, axs = plt.subplots(len(snaps), len(incs), figsize=(3.3 * len(incs) + 0.9, 3.25 * len(snaps)),
                            squeeze=False, facecolor="black")
    for r, s in enumerate(snaps):
        row = [f"{s}_i{i:02d}_Rh{rh_fid}" for i in incs]
        vmax = max(tb(blurred[n][0].max(), recs[n]) for n in row if n in recs) / 1e9
        for c, n in enumerate(row):
            ax = axs[r, c]
            ax.set_facecolor("black")
            if n not in recs:
                ax.axis("off"); continue
            rec = recs[n]
            im = ax.imshow(tb(blurred[n][0], rec) / 1e9, origin="lower", extent=extent_uas(rec),
                           cmap="afmhot", vmin=0, vmax=vmax, interpolation="bicubic")
            set_lim(ax, LIM1_UAS)
            draw_beam(ax, rec, lim=LIM1_UAS)
            mark_holes(ax, rec)
            ax.text(0.04, 0.95, f"i = {incs[c]}°", transform=ax.transAxes, color="w", va="top")
            ax.text(0.96, 0.95, f"{rec['flux_Jy']:.2f} Jy", transform=ax.transAxes, color="w",
                    va="top", ha="right")
            rg = mets[n]["rings"]
            if rg["resolved_pair"]:
                ax.text(0.96, 0.04, "$f_C$ " + " / ".join(f"{rg[h]['blur20']['f_C']:.2f}"
                                                         for h in ("hole1", "hole2")),
                        transform=ax.transAxes, color="w", ha="right", fontsize=8)
            style_axes(ax, left=c == 0, bottom=r == len(snaps) - 1)
            for lab in (ax.xaxis.label, ax.yaxis.label):
                lab.set_color("w")
            ax.tick_params(colors="w")
            if c == 0:
                t = rec["snapshot"]["time_M"]
                ax.set_title(f"{SNAP_LABEL.get(s, s)}   t = {t:.0f} M", color="w", loc="left")
        cb = fig.colorbar(im, ax=axs[r, :].tolist(), fraction=0.025, pad=0.01)
        cb.set_label(r"$T_b$ ($10^9$ K), 20 µas blur", color="w")
        cb.ax.tick_params(colors="w")
    save(fig, "fig1_snapshot_inclination_grid")

    # -------------------------------------------------- fig 2: native vs blurred
    fid = "s455"
    pairs = [f"{fid}_i17_Rh{rh_fid}", f"{fid}_i90_Rh{rh_fid}"]
    pairs = [p for p in pairs if p in recs]
    if pairs:
        rows = (("native", "native (sqrt stretch)"), ("blur10", f"{NG_BEAM_UAS:g} µas blur (ngEHT-like)"),
                ("blur20", f"{BEAM_UAS:g} µas blur (EHT)"))
        fig, axs = plt.subplots(3, len(pairs), figsize=(3.9 * len(pairs) + 0.6, 10.6),
                                squeeze=False, facecolor="black")
        for c, n in enumerate(pairs):
            rec = recs[n]
            rg = mets[n]["rings"]
            data_by = {"native": imgs[n]["I"], "blur10": blurred[n][4], "blur20": blurred[n][0]}
            for r, (key, lab) in enumerate(rows):
                ax = axs[r, c]
                T = tb(data_by[key], rec) / 1e9
                norm = PowerNorm(0.5, vmin=0, vmax=T.max()) if key == "native" else Normalize(0, T.max())
                im = ax.imshow(T, origin="lower", extent=extent_uas(rec), cmap="afmhot", norm=norm,
                               interpolation="nearest" if key == "native" else "bicubic")
                set_lim(ax, LIM2_UAS)
                if key == "native":
                    mark_holes(ax, rec)
                else:
                    draw_beam(ax, rec, fwhm=NG_BEAM_UAS if key == "blur10" else BEAM_UAS,
                              lim=LIM2_UAS)
                ax.set_title(f"i = {rec['camera']['inclination_deg']:.0f}°, {lab}, "
                             f"{rec['flux_Jy']:.2f} Jy", color="w")
                if rg["resolved_pair"]:
                    ftxt = "\n".join(f"hole {k}: d = {rg[h][key]['diameter_uas']:.0f} µas, "
                                     f"$f_C$ = {rg[h][key]['f_C']:.2f}"
                                     for k, h in ((1, "hole1"), (2, "hole2")))
                    ax.text(0.97, 0.03, ftxt, transform=ax.transAxes, color="w", fontsize=7,
                            ha="right")
                style_axes(ax, left=c == 0, bottom=r == 2)
                for l_ in (ax.xaxis.label, ax.yaxis.label):
                    l_.set_color("w")
                ax.tick_params(colors="w")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label(r"$T_b$ ($10^9$ K)", color="w")
                cb.ax.tick_params(colors="w")
        fig.suptitle(f"{SNAP_LABEL[fid]}, t = {recs[pairs[0]]['snapshot']['time_M']:.0f} M, "
                     f"230 GHz, R_high = {rh_fid}", color="w", y=1.0)
        save(fig, "fig2_native_vs_blurred")

    # -------------------------------------------------- fig 3: polarimetric panel
    if pairs:
        mnorm = Normalize(0.0, 0.4)
        cmap = LinearSegmentedColormap.from_list("plasma_hi", plt.get_cmap("plasma")(np.linspace(0.25, 1.0, 256)))
        fig, axs = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs) + 0.8, 4.4),
                                squeeze=False, facecolor="white")
        for c, n in enumerate(pairs):
            rec, met = recs[n], mets[n]
            Ib, Qb, Ub, _, _ = blurred[n]
            ax = axs[0, c]
            ax.set_facecolor("black")
            ax.imshow(tb(Ib, rec) / 1e9, origin="lower", extent=extent_uas(rec), cmap="gray",
                      vmin=0, interpolation="bicubic")
            evpa_ticks(ax, rec, Ib, Qb, Ub, cmap, mnorm)
            set_lim(ax, LIM2_UAS)
            draw_beam(ax, rec, lim=LIM2_UAS)
            txt = (f"$m_{{net}}$ = {met['m_net']:.3f}   "
                   f"$\\langle|m|\\rangle$ = {met['mabs_avg']:.3f}   "
                   f"$v_{{net}}$ = {met['v_net']:+.1e}\n"
                   f"$|\\beta_2|$ = {met['beta2_amp']:.3f}   "
                   f"$\\angle\\beta_2$ = {met['beta2_arg_deg']:.0f}°")
            ax.set_title(txt, fontsize=8.5)
            ax.text(0.97, 0.03, f"i = {rec['camera']['inclination_deg']:.0f}°", transform=ax.transAxes,
                    color="w", ha="right")
            style_axes(ax, left=c == 0)
        sm = plt.cm.ScalarMappable(norm=mnorm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axs[0, :].tolist(), fraction=0.03, pad=0.02)
        cb.set_label("fractional linear polarization |m|")
        fig.suptitle(f"{SNAP_LABEL[fid]}: 20 µas-blurred I (gray) and EVPA "
                     f"(ticks where I > {int(TICK_ICUT * 100)}% of peak)", y=1.0)
        save(fig, "fig3_polarimetric")

    # -------------------------------------------------- fig 4: R_high survey
    rh_names = sorted([n for n in names if n.startswith(f"{fid}_i17_")],
                      key=lambda n: recs[n]["plasma"]["R_high"])
    if len(rh_names) > 1:
        vmax = max(tb(blurred[n][0].max(), recs[n]) for n in rh_names) / 1e9
        fig, axs = plt.subplots(1, len(rh_names), figsize=(2.9 * len(rh_names) + 0.9, 3.2),
                                squeeze=False, facecolor="black")
        for c, n in enumerate(rh_names):
            rec, met = recs[n], mets[n]
            ax = axs[0, c]
            im = ax.imshow(tb(blurred[n][0], rec) / 1e9, origin="lower", extent=extent_uas(rec),
                           cmap="afmhot", vmin=0, vmax=vmax, interpolation="bicubic")
            set_lim(ax, LIM2_UAS)
            draw_beam(ax, rec, lim=LIM2_UAS)
            rg = met["rings"]
            ax.text(0.04, 0.95, f"$R_{{high}}$ = {rec['plasma']['R_high']:g}", transform=ax.transAxes,
                    color="w", va="top")
            ax.text(0.97, 0.03, "$f_C$ " + " / ".join(f"{rg[h]['blur20']['f_C']:.2f}"
                                                     for h in ("hole1", "hole2")),
                    transform=ax.transAxes, color="w", ha="right", fontsize=7.5)
            ax.set_title(f"{rec['flux_Jy']:.2f} Jy, $\\mathcal{{M}}$ = {rec['mass_scale_g']:.2e} g\n"
                         f"$m_{{net}}$ {met['m_net']:.3f}, $|\\beta_2|$ {met['beta2_amp']:.3f}",
                         color="w", fontsize=8)
            style_axes(ax, left=c == 0)
            for l_ in (ax.xaxis.label, ax.yaxis.label):
                l_.set_color("w")
            ax.tick_params(colors="w")
        cb = fig.colorbar(im, ax=axs[0, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label(r"$T_b$ ($10^9$ K), 20 µas blur", color="w")
        cb.ax.tick_params(colors="w")
        fig.suptitle(f"{SNAP_LABEL[fid]}, i = 17°, each calibrated to 1.0 Jy", color="w", y=1.06)
        save(fig, "fig4_rhigh_survey")


if __name__ == "__main__":
    main()
