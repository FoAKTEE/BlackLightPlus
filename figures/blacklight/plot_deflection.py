#!/usr/bin/env python3
"""Figures 1 and 2 of the Blacklight-paper reproduction from out/blacklight/deflection/:
fig01_deflection_values.png (deflection minus the flat-space sgn(b) pi against the impact
parameter, the formula as a line, the differences below) and fig02_deflection_errors.png
(absolute error against the number of steps for the Vern7 tolerance ladder and the RK4
step-divisor ladder). Run: python3 figures/blacklight/plot_deflection.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "figures", "out", "blacklight", "deflection")
RES = os.path.join(ROOT, "results", "blp", "figures_blacklight")
os.makedirs(RES, exist_ok=True)

v = json.load(open(os.path.join(OUT, "values.json")))
rays = sorted(v["rays"], key=lambda r: r["b"])
b = np.array([r["b"] for r in rays]); d = np.array([r["dphi"] for r in rays])
f = np.array([r["formula"] for r in rays]); fl = np.array([r["flat"] for r in rays])
fig, axs = plt.subplots(2, 1, figsize=(4.2, 4.6), dpi=170, sharex=True,
                        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08})
bb = np.linspace(b.min(), b.max(), 400)
for side in (bb < v["b_minus"], bb > v["b_plus"]):
    pass
axs[0].plot(b, f - fl, "-", color="0.4", lw=1.0, label="formula")
axs[0].plot(b, d - fl, "o", color="C0", ms=3.5, label="BlackLightPlus, Vern7 $10^{-8}$")
axs[0].axvspan(v["b_minus"], v["b_plus"], color="0.9", zorder=0)
axs[0].set_ylabel(r"$\Delta\phi - \mathrm{sgn}(b)\,\pi$")
axs[0].legend(frameon=False, fontsize=8)
axs[1].plot(b, d - f, "o", color="C0", ms=3.5)
axs[1].axhline(0, color="0.5", lw=0.6)
axs[1].axvspan(v["b_minus"], v["b_plus"], color="0.9", zorder=0)
axs[1].set_xlabel(r"$b$ [$GM/c^2$]"); axs[1].set_ylabel("difference")
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.savefig(os.path.join(RES, "fig01_deflection_values.png"), bbox_inches="tight")
plt.close(fig)

e = json.load(open(os.path.join(OUT, "errors.json")))
fig, axs = plt.subplots(2, 1, figsize=(4.2, 5.4), dpi=170, sharex=True, gridspec_kw={"hspace": 0.1})
for ax, (key, title, lab) in zip(axs, (("vern7", "Vern7 (adaptive)", "tolerance"), ("rk4", "RK4, step $(r - r_\\mathrm{h})/C$", "$C$"))):
    lad = e["ladders"][key]
    keys = sorted(lad.keys(), key=float)
    cm = plt.get_cmap("viridis")
    for i, k in enumerate(keys):
        rows = lad[k]
        ax.loglog([r["steps"] for r in rows], [max(r["err"], 1e-16) for r in rows], ".", ms=4,
                  color=cm(i / max(len(keys) - 1, 1)), label=f"{lab} = {float(k):g}")
    ax.set_ylabel(r"$|\Delta\phi - \Delta\phi_\mathrm{formula}|$")
    ax.text(0.03, 0.05, title, transform=ax.transAxes, fontsize=8)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
axs[1].set_xlabel("number of steps")
fig.savefig(os.path.join(RES, "fig02_deflection_errors.png"), bbox_inches="tight")
print("wrote", RES)
