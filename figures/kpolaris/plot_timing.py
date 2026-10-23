#!/usr/bin/env python3
"""Figures 2 and 3 of the KPolaris reproduction from the timing records of
fig_accuracy_cost.jl and fig_performance.jl (figures/paper/out/kpolaris/{accuracy_cost,
performance}/<backend>.json):

  fig2_accuracy_cost.png   render time (median of three) against the largest Stokes NMSE
                           relative to the C = 2560 image, one curve per backend, the
                           adopted C = 80 ringed;
  fig3_performance.png     render time against resolution, medians with the range of the
                           three calls as bars, one curve per backend.

Run: python3 figures/kpolaris/plot_timing.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "kpolaris")
LABEL = {"cpu": "CPU, 32 threads", "gpu1": "1 A100", "gpu2": "2 A100"}
STYLE = {"cpu": ("tab:blue", "s"), "gpu1": ("tab:orange", "o"), "gpu2": ("tab:purple", "^")}


def load(sub):
    recs = {}
    for f in sorted(glob.glob(os.path.join(OUT, sub, "*.json"))):
        r = json.load(open(f))
        recs[os.path.splitext(os.path.basename(f))[0]] = r
    return recs


def accuracy_cost():
    recs = load("accuracy_cost")
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=150)
    for key, r in recs.items():
        rows = sorted(r["rows"], key=lambda q: q["C"])
        x = [q["max_nmse"] for q in rows]
        y = [q["median"] for q in rows]
        c, m = STYLE.get(key, ("k", "x"))
        ax.plot(x, y, marker=m, color=c, lw=1.2, ms=4, label=LABEL.get(key, key))
        for q in rows:
            if q["C"] == r["adopted_C"]:
                ax.plot([q["max_nmse"]], [q["median"]], marker="o", mfc="none", mec=c, ms=11, mew=1.4)
            ax.annotate(f"C={q['C']:.0f}", (q["max_nmse"], q["median"]), fontsize=5.5,
                        textcoords="offset points", xytext=(3, 3), color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\max_{I,Q,U,V}$ NMSE (against $C = 2560$)", fontsize=8)
    ax.set_ylabel("render time (s), median of 3", fontsize=8)
    ax.set_title("torus, 256², 230 GHz, full Stokes", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, frameon=False, title="open ring: adopted C = 80", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "accuracy_cost", "fig2_accuracy_cost.png"))


def performance():
    recs = load("performance")
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=150)
    for key, r in recs.items():
        rows = sorted(r["rows"], key=lambda q: q["res"])
        x = [q["res"] for q in rows]
        y = np.array([q["median"] for q in rows])
        lo = y - np.array([q["min"] for q in rows]); hi = np.array([q["max"] for q in rows]) - y
        c, m = STYLE.get(key, ("k", "x"))
        ax.errorbar(x, y, yerr=[lo, hi], marker=m, color=c, lw=1.2, ms=4, capsize=2,
                    label=LABEL.get(key, key))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([256, 512, 1024, 2048]); ax.set_xticklabels(["256", "512", "1024", "2048"])
    ax.set_xlabel(r"$N$ ($N \times N$ pixels)", fontsize=8)
    ax.set_ylabel("render time (s), median of 3", fontsize=8)
    ax.set_title("torus, 230 GHz, full Stokes", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "performance", "fig3_performance.png"))


if __name__ == "__main__":
    accuracy_cost()
    performance()
    print("drawn")
