#!/usr/bin/env python3
"""Review images of the binary-black-hole figure data: the right-hand 2 x 3 grid of
fig_binary with the stretches of paper/scripts/plot_binary.py.

Top row, (b)-(d): native 230 GHz Stokes I of snapshots 00455, 00093, 00122 at
theta_o = 90 deg (<root>/figures/out/eht_bbh/s<snap>_i90_Rh40/{I.npy, records.json}) as
Rayleigh-Jeans brightness temperature T_b = I c^2 / (2 k nu^2) on one LogNorm from
1e7 K to the largest peak, +-170 uas, RA increasing to the left, projected holes marked,
flux at the upper right. Bottom row, (f)-(h): the adaptive full-Stokes image of 00455
(<root>/figures/out/bbh_i90_pol_amr/{amr4096_{I,Q,U,V}.npy, amr4096_levels_fine.npy,
records.json}): sqrt(Q^2 + U^2)/I on [0, 1] and V/I symmetric at the 99th percentile of
|V/I|, both on pixels with I > 1e-4 max(I), and the finest refinement level per 8 x 8
tile, +-13.5 M.

Writes BLP/figures/out/eht_bbh/review/fig_binary_right.png from the BLP data (root BLP/)
and fig_binary_right_ref.png from the reference data (the repository root), drawn
alike. A snapshot whose data are missing is left blank.

Usage, from the repository root:
    python3 BLP/figures/check_bbh.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLP_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(BLP_ROOT)
REVIEW = os.path.join(HERE, "out", "eht_bbh", "review")
SNAPS = ("s455", "s093", "s122")
C_CGS = 2.99792458e10
K_B = 1.380649e-16
LIM_UAS = 170.0
TB_MIN = 1.0e7
CROP_M = 13.5
EMIT = 1e-4
SKY = "#56B4E9"


def level_cmap(max_level):
    return ListedColormap(plt.get_cmap("plasma")(np.linspace(0.0, 1.0, max_level + 1)))


def eht_data(root):
    out = []
    for s in SNAPS:
        d = os.path.join(root, "figures", "out", "eht_bbh", f"{s}_i90_Rh40")
        if not os.path.exists(os.path.join(d, "records.json")):
            out.append((s, None, None))
            continue
        rec = json.load(open(os.path.join(d, "records.json")))
        I = np.load(os.path.join(d, "I.npy"))
        out.append((s, rec, I * C_CGS ** 2 / (2 * K_B * rec["frequency_hz"] ** 2)))
    return out


def amr_data(root):
    d = os.path.join(root, "figures", "out", "bbh_i90_pol_amr")
    if not os.path.exists(os.path.join(d, "records.json")):
        return None
    rec = json.load(open(os.path.join(d, "records.json")))
    eff = rec["amr"]["eff_res"]
    I, Q, U, V = (np.load(os.path.join(d, f"amr{eff}_{c}.npy")) for c in "IQUV")
    lvf = np.load(os.path.join(d, f"amr{eff}_levels_fine.npy"))
    emit = I > EMIT * I.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(emit, np.hypot(Q, U) / np.where(emit, I, 1.0), np.nan)
        vi = np.where(emit, V / np.where(emit, I, 1.0), np.nan)
    return rec, lp, vi, lvf


def draw(root, path, label):
    eht = eht_data(root)
    amr = amr_data(root)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.6))
    tbs = [tb for _, _, tb in eht if tb is not None]
    if tbs:
        normT = LogNorm(vmin=TB_MIN, vmax=max(tb.max() for tb in tbs))
    for k, (s, r, tb) in enumerate(eht):
        ax = axs[0, k]
        if r is None:
            ax.set_axis_off()
            continue
        uas = r["scaling"]["theta_g_uas"]
        h = r["camera"]["fov_M"] / 2 * uas
        im = ax.imshow(np.clip(tb, TB_MIN, None), origin="lower", extent=[h, -h, -h, h],
                       cmap="inferno", norm=normT, interpolation="nearest")
        for ha, hb in r["projected_holes_M"]:
            ax.plot(-ha * uas, hb * uas, "+", color=SKY, ms=7, mew=1.0)
        ax.set_xlim(LIM_UAS, -LIM_UAS)
        ax.set_ylim(-LIM_UAS, LIM_UAS)
        ax.set_xlabel("relative RA [uas]")
        ax.set_ylabel("relative Dec [uas]" if k == 0 else "")
        ax.text(0.04, 0.955, f"({'bcd'[k]}) {s[1:].rjust(5, '0')}", transform=ax.transAxes,
                color="w", va="top")
        ax.text(0.96, 0.955, f"{r['flux_Jy']:.2f} Jy", transform=ax.transAxes, color="w",
                va="top", ha="right")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13,
                     label="T_b [K]")
    if amr is None:
        for ax in axs[1]:
            ax.set_axis_off()
    else:
        rec, lp, vi, lvf = amr
        ml = rec["amr"]["max_level"]
        half = rec["fov_M"] / 2
        vmax = float(np.nanpercentile(np.abs(vi), 99.0))
        specs = [(lp, "cividis", Normalize(0.0, 1.0), "sqrt(Q^2+U^2)/I", "antialiased"),
                 (vi, "RdBu_r", Normalize(-vmax, vmax), f"V/I (+-{vmax:.2e})", "antialiased"),
                 (lvf, level_cmap(ml), BoundaryNorm(np.arange(-0.5, ml + 1), ml + 1),
                  "refinement level", "nearest")]
        for k, (data, cmap, norm, lab, interp) in enumerate(specs):
            ax = axs[1, k]
            cm = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
            cm.set_bad("#d9d9d9")
            im = ax.imshow(data, origin="lower", extent=[-half, half, -half, half], cmap=cm,
                           norm=norm, interpolation=interp)
            ax.set_xlim(-CROP_M, CROP_M)
            ax.set_ylim(-CROP_M, CROP_M)
            ax.set_xlabel("alpha [GM/c^2]")
            ax.set_ylabel("beta [GM/c^2]" if k == 0 else "")
            ax.text(0.04, 0.955, f"({'fgh'[k]})", transform=ax.transAxes, va="top",
                    color="k" if k < 2 else "w")
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13, label=lab)
        lv = rec["amr_leg"]["levels_hist"]
        axs[1, 2].set_title(f"levels hist {lv}", fontsize=8)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    os.makedirs(REVIEW, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def main():
    draw(BLP_ROOT, os.path.join(REVIEW, "fig_binary_right.png"), "BLP data (BLP/figures/out)")
    draw(REPO, os.path.join(REVIEW, "fig_binary_right_ref.png"), "reference data (figures/out)")


if __name__ == "__main__":
    main()
