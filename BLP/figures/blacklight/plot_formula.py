#!/usr/bin/env python3
"""Figure 3 of the Blacklight-paper reproduction and figure 7 of the Mahakala-paper
reproduction from out/blacklight/formula/: the five Gold et al. 2020 model images.
fig03_formula.png shows I_nu scaled by 1e5 (75/128)^2 / f with f the exact flux of each
test (the paper's colour scale, 0-2); results/blp/figures_mahakala/fig07_gold_models.png
shows the per-pixel flux divided by the exact total flux, S/S_tot,exact (0-2e-4), with the
total flux printed in each panel. Axes in microarcseconds at 1 M = 5 muas.
Run: python3 BLP/figures/blacklight/plot_formula.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "BLP", "figures", "out", "blacklight", "formula")
RES_BL = os.path.join(ROOT, "results", "blp", "figures_blacklight")
RES_MH = os.path.join(ROOT, "results", "blp", "figures_mahakala")
os.makedirs(RES_BL, exist_ok=True); os.makedirs(RES_MH, exist_ok=True)
rec = json.load(open(os.path.join(OUT, "record.json")))
EXACT = [rec["tests"][f"test{k}"]["exact_jy"] for k in range(1, 6)]
# the distance that matches the exact fluxes (the five ratios agree to 0.4%)
ratio = np.mean([rec["tests"][f"test{k}"]["ratio"] for k in range(1, 6)])
pix_sr = (rec["fov"] / rec["res"] * rec["m_cm"] / rec["d_cm"]) ** 2 / ratio
lim = rec["fov"] / 2 * 5.0                          # muas
ext = [-lim, lim, -lim, lim]

fig, axs = plt.subplots(1, 5, figsize=(10.5, 2.5), dpi=170)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.8, wspace=0.08)
for k, ax in enumerate(axs, 1):
    I = np.load(os.path.join(OUT, f"test{k}_I.npy"))
    scaled = I * 1e5 * (75 / 128) ** 2 / EXACT[k - 1]
    im = ax.imshow(scaled, origin="lower", extent=ext, cmap="cubehelix", vmin=0, vmax=2, interpolation="nearest")
    ax.text(0.04, 0.93, f"Test {k}", transform=ax.transAxes, color="w", fontsize=8, va="top")
    ax.set_xlabel(r"$a$ [$\mu$as]", fontsize=8); ax.tick_params(labelsize=7)
    if k == 1:
        ax.set_ylabel(r"$b$ [$\mu$as]", fontsize=8)
    else:
        ax.set_yticklabels([])
cax = fig.add_axes([0.25, 0.9, 0.5, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label(r"$I_\nu$ (scaled)", fontsize=8, labelpad=-30)
cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(RES_BL, "fig03_formula.png"))
plt.close(fig)

fig, axs = plt.subplots(1, 5, figsize=(10.5, 2.5), dpi=170)
fig.subplots_adjust(left=0.05, right=0.93, bottom=0.2, top=0.92, wspace=0.08)
for k, ax in enumerate(axs, 1):
    I = np.load(os.path.join(OUT, f"test{k}_I.npy"))
    S = I * pix_sr * 1e23                            # Jy per pixel
    im = ax.imshow(S / EXACT[k - 1], origin="lower", extent=ext, cmap="cubehelix", vmin=0, vmax=2e-4, interpolation="nearest")
    ax.text(0.04, 0.93, r"$S_\mathrm{tot} = %.4f$ Jy" % S.sum(), transform=ax.transAxes, color="w", fontsize=7, va="top")
    ax.set_xlabel("R.A. [$\\mu$as]", fontsize=8); ax.tick_params(labelsize=7)
    if k == 1:
        ax.set_ylabel("DEC [$\\mu$as]", fontsize=8)
    else:
        ax.set_yticklabels([])
cax = fig.add_axes([0.94, 0.2, 0.012, 0.72])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7)
cb.set_ticks([0, 1e-4, 2e-4])
fig.savefig(os.path.join(RES_MH, "fig07_gold_models.png"))
print("fluxes", [round(rec["tests"][f"test{k}"]["flux_jy"] / ratio, 4) for k in range(1, 6)], "distance kpc", rec["d_cm"] / np.sqrt(ratio) / 3.0857e21)
