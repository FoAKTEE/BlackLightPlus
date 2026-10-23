#!/usr/bin/env python3
"""Figure 1 of the KPolaris reproduction: BlackLightPlus against Blacklight on the torus dump.

Reads figures/paper/out/kpolaris/comparison/blp_<fast|slow>_<230|345>_{I,Q,U,V}.npy ([iβ, iα])
and the Blacklight npz files <bl_dir>/<fast|slow>_<230|345>.npz (I_nu, Q_nu, U_nu, V_nu; the
Blacklight image array is read as [row, column] with row 0 the top of the image when
camera_rotation = 0, column 0 the left; it is mapped to BlackLightPlus's [iβ, iα] layout by the
dihedral transform that minimises NMSE(I) in the 230 GHz fast-light pair, and the same
transform is applied to every pair; pixels NaN in Blacklight, rays it ends unexpectedly,
are left out of every sum). Writes fig1_code_comparison.png (top: Blacklight I
normalised by its peak; bottom: (I_BLP − I_BL)/max I_BL) and nmse.json (per-component NMSE,
flux and net linear polarization of both codes, the transform used).

Run: python3 figures/kpolaris/plot_comparison.py <bl_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "kpolaris", "comparison")
BL = sys.argv[1]
CASES = [("fast_230", "fast light, 230 GHz"), ("fast_345", "fast light, 345 GHz"),
         ("slow_230", "slow light, 230 GHz"), ("slow_345", "slow light, 345 GHz")]
STOKES = "IQUV"
FOV = 30.0

TRANSFORMS = {
    "identity": lambda a: a, "T": lambda a: a.T, "flipud": lambda a: a[::-1, :],
    "fliplr": lambda a: a[:, ::-1], "rot180": lambda a: a[::-1, ::-1],
    "T_flipud": lambda a: a.T[::-1, :], "T_fliplr": lambda a: a.T[:, ::-1],
    "T_rot180": lambda a: a.T[::-1, ::-1]}


def nmse(a, ref):
    """NMSE over the pixels finite in both images (Blacklight marks rays that end
    unexpectedly as NaN)."""
    m = np.isfinite(a) & np.isfinite(ref)
    return float(np.sum((a[m] - ref[m]) ** 2) / np.sum(ref[m] ** 2))


def load_blp(case):
    return {s: np.load(os.path.join(OUT, f"blp_{case}_{s}.npy")) for s in STOKES}


def load_bl(case):
    z = np.load(os.path.join(BL, case + ".npz"))
    return {s: np.array(z[f"{s}_nu"], dtype=float) for s in STOKES}


def main():
    pairs = {c: (load_blp(c), load_bl(c)) for c, _ in CASES if
             os.path.exists(os.path.join(BL, c + ".npz")) and
             os.path.exists(os.path.join(OUT, f"blp_{c}_I.npy"))}
    blp0, bl0 = pairs["fast_230"]
    scores = {k: nmse(blp0["I"], f(bl0["I"])) for k, f in TRANSFORMS.items()}
    best = min(scores, key=scores.get)
    tf = TRANSFORMS[best]
    rec = {"transform": best, "transform_nmse_I": scores, "cases": {}}
    ncase = len(pairs)
    fig, axs = plt.subplots(2, ncase, figsize=(2.9 * ncase + 1.2, 5.6), dpi=150)
    axs = np.atleast_2d(axs).reshape(2, ncase)
    ext = [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]
    for k, (c, label) in enumerate([cl for cl in CASES if cl[0] in pairs]):
        blp, bl = pairs[c]
        bl = {s: tf(bl[s]) for s in STOKES}
        r = {s: nmse(blp[s], bl[s]) for s in STOKES}
        m = np.isfinite(bl["I"])
        flux = {"blp": float(blp["I"][m].sum()), "bl": float(bl["I"][m].sum())}
        ml = {code: float(np.hypot(d["Q"][m].sum(), d["U"][m].sum()) / d["I"][m].sum())
              for code, d in (("blp", blp), ("bl", bl))}
        rec["cases"][c] = {"nmse": r, "sum_I": flux, "mL_net": ml, "nan_pixels_bl": int((~m).sum()),
                           "flux_diff_percent": 100 * (flux["blp"] - flux["bl"]) / flux["bl"],
                           "mL_net_diff_points": 100 * (ml["blp"] - ml["bl"])}
        imax = np.nanmax(bl["I"])
        ax = axs[0, k]
        im0 = ax.imshow(bl["I"] / imax, origin="lower", extent=ext, cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax = axs[1, k]
        res = (blp["I"] - bl["I"]) / imax
        lim = 0.01
        im1 = ax.imshow(res, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.text(0.03, 0.03, f"NMSE(I) = {r['I']:.2e}\nmax NMSE = {max(r.values()):.2e}",
                transform=ax.transAxes, fontsize=7, color="k")
        for ax in axs[:, k]:
            ax.set_xlabel(r"$\alpha$ [$GM/c^2$]", fontsize=8)
            ax.tick_params(labelsize=7)
        axs[0, k].set_xlabel("")
    axs[0, 0].set_ylabel(r"$\beta$ [$GM/c^2$]", fontsize=8)
    axs[1, 0].set_ylabel(r"$\beta$ [$GM/c^2$]", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.93, bottom=0.09, wspace=0.25, hspace=0.25)
    cax0 = fig.add_axes([0.90, 0.56, 0.015, 0.35])
    fig.colorbar(im0, cax=cax0).set_label(r"$I_{\rm BL}/I_{\rm BL,max}$", fontsize=8)
    cax1 = fig.add_axes([0.90, 0.10, 0.015, 0.35])
    fig.colorbar(im1, cax=cax1).set_label(r"$(I_{\rm BlackLightPlus}-I_{\rm BL})/I_{\rm BL,max}$", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig1_code_comparison.png"))
    with open(os.path.join(OUT, "nmse.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("transform", best, {k: f"{v:.2e}" for k, v in scores.items()})
    for c, v in rec["cases"].items():
        print(c, {s: f"{x:.2e}" for s, x in v["nmse"].items()}, "flux %", f"{v['flux_diff_percent']:.3f}",
              "mL", v["mL_net"])


if __name__ == "__main__":
    main()
