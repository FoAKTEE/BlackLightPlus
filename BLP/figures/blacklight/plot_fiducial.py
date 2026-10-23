#!/usr/bin/env python3
"""Figures 4, 5, 6 and 10 of the Blacklight-paper reproduction from out/blacklight/:
fig04_stokes.png (I, Q, U, V of the fiducial image), fig05_cuts.png (fiducial, beta^-1 >= 2,
near half-space, far half-space), fig06_flat.png (Kerr beside flat), fig10_nonthermal.png
(thermal, power law, kappa, mixed). Intensities in erg s^-1 cm^-2 sr^-1 Hz^-1; the axes
in M of the 24 M camera. Run: python3 BLP/figures/blacklight/plot_fiducial.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "BLP", "figures", "out", "blacklight")
RES = os.path.join(ROOT, "results", "blp", "figures_blacklight")
os.makedirs(RES, exist_ok=True)
FOV = 24.0
EXT = [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]
load = lambda d, n: np.load(os.path.join(OUT, d, n + ".npy"))


def panel(ax, A, title, cmap="inferno", vmin=None, vmax=None, sym=False):
    if sym:
        m = np.nanmax(np.abs(A)); vmin, vmax = -m, m
    im = ax.imshow(A, origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10]); ax.tick_params(labelsize=7)
    return im


def cbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.formatter.set_powerlimits((-2, 2)); cb.update_ticks()


# figure 4
I, Q, U, V = (load("stokes", "fiducial_" + s) for s in "IQUV")
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0), dpi=160)
cbar(fig, panel(axs[0, 0], I, "$I$"), axs[0, 0])
cbar(fig, panel(axs[0, 1], Q, "$Q$", cmap="RdBu_r", sym=True), axs[0, 1])
cbar(fig, panel(axs[1, 0], U, "$U$", cmap="RdBu_r", sym=True), axs[1, 0])
cbar(fig, panel(axs[1, 1], V, "$V$", cmap="RdBu_r", sym=True), axs[1, 1])
for ax in axs[1]: ax.set_xlabel(r"$\alpha$ [$GM/c^2$]", fontsize=8)
for ax in axs[:, 0]: ax.set_ylabel(r"$\beta$ [$GM/c^2$]", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig04_stokes.png")); plt.close(fig)

# figure 5
rec = json.load(open(os.path.join(OUT, "cuts", "record.json")))["cuts"]
tags = [("fiducial", r"fiducial ($\sigma \leq 1$)"), ("beta_inv_ge_2", r"$\beta^{-1} \geq 2$ only"),
        ("near_half", "near half-space only"), ("far_half", "far half-space only")]
vmax = load("cuts", "fiducial_I").max()
fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0), dpi=160)
for ax, (tag, title) in zip(axs.ravel(), tags):
    im = panel(ax, load("cuts", tag + "_I"), f"{title}: {rec[tag]['flux_jy']:.2f} Jy", vmin=0, vmax=vmax)
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label=r"$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Hz$^{-1}$]")
fig.savefig(os.path.join(RES, "fig05_cuts.png"), bbox_inches="tight"); plt.close(fig)

# figure 6
rec = json.load(open(os.path.join(OUT, "flat", "record.json")))["runs"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2), dpi=160)
vmax = max(load("flat", "kerr_I").max(), load("flat", "flat_I").max())
for ax, tag, title in zip(axs, ("kerr", "flat"), ("Kerr", "flat spacetime")):
    im = panel(ax, load("flat", tag + "_I"), f"{title}: {rec[tag]['flux_jy']:.2f} Jy", vmin=0, vmax=vmax)
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
fig.savefig(os.path.join(RES, "fig06_flat.png"), bbox_inches="tight"); plt.close(fig)

# figure 10
if os.path.exists(os.path.join(OUT, "nonthermal", "record.json")):
    rec = json.load(open(os.path.join(OUT, "nonthermal", "record.json")))["runs"]
    tags = [("thermal", "thermal"), ("powerlaw", "power law"), ("kappa", r"$\kappa$ distribution"), ("mixed", "mixed")]
    vmax = max(load("nonthermal", t + "_I").max() for t, _ in tags)
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.0), dpi=160)
    for ax, (tag, title) in zip(axs.ravel(), tags):
        im = panel(ax, load("nonthermal", tag + "_I"), f"{title}: {rec[tag]['flux_jy']:.2f} Jy", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    fig.savefig(os.path.join(RES, "fig10_nonthermal.png"), bbox_inches="tight"); plt.close(fig)
print("wrote figures 4-6, 10 to", RES)
