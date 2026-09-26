"""r07_figures.py — figures of the RAPTOR vs BlackLightPlus s455 comparison (R07, R08).

  images_i<inc>.png   rows: BlackLightPlus, RAPTOR (a) matched with its native polarized
                      step, (a) - BlackLightPlus, (b) (LD85 step, M11) - BlackLightPlus;
                      columns: I, |P|, EVPA, V at step constant C (default the finest
                      common level). Sky orientation of figures/plot_eht_bbh.py (east
                      left). Differences are per-pixel relative errors: dI / max I_B,
                      d|P| / max |P|_B, dV / V_B,99.9, and the EVPA difference in degrees
                      where |P|/I > 1% and I > 1% of its peak.
  convergence.png     normalized L1 of I, |P| and V against the step constant C: each
                      code's self-convergence (against its finest level) and the
                      code-to-code difference of legs (a) and (b), one row per inclination.
  timing.png          R08 phase walls per code (from evidence/R08/timing.json).

Usage: python3 r07_figures.py <rundir> <metrics.json> <figdir> [--C 320] [--timing t.json]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r07_compare as rc  # noqa: E402

COL = {"blp": "#2a78d6", "ra": "#eb6834", "rb": "#1baf7a"}
LAB = {"blp": "BlackLightPlus", "ra": "RAPTOR (a) native step", "rb": "RAPTOR (b) LD85 step"}
INK2 = "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": "#0b0b0b",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False})


def extent(npix):
    h = rc.FOV / 2 * rc.UAS_PER_M
    return [h, -h, -h, h]


def img_panels(run, inc, C, npix, out, met, label="s455"):
    B = rc.load_blp(run, inc, C, npix)
    A = rc.load_raptor(run, "ra", inc, C, npix)
    Bb = rc.load_raptor(run, "rb", inc, C, npix)
    ext = extent(npix)
    lim = 150.0

    def comps(X):
        I = X["I"]
        P = np.hypot(X["Q"], X["U"])
        chi = np.degrees(0.5 * np.arctan2(X["U"], X["Q"]))
        mask = (I < 0.01 * I.max()) | (P < 0.01 * I)
        return I, P, np.ma.array(chi, mask=mask), X["V"]

    IB, PB, CB, VB = comps(B)
    rows = [("BlackLightPlus", comps(B)), ("RAPTOR (a)", comps(A))]
    fig, axs = plt.subplots(4, 4, figsize=(12.4, 12.6), constrained_layout=True)
    vmI, vmP = IB.max(), PB.max()
    vmV = np.percentile(np.abs(VB), 99.9)       # V peaks in a few pixels: clip at the 99.9th percentile
    for r, (name, (I, P, chi, V)) in enumerate(rows):
        ims = [axs[r, 0].imshow(I / vmI, origin="lower", extent=ext, cmap="afmhot", norm=PowerNorm(0.5, 0, 1)),
               axs[r, 1].imshow(P / vmP, origin="lower", extent=ext, cmap="afmhot", norm=PowerNorm(0.5, 0, 1)),
               axs[r, 2].imshow(chi, origin="lower", extent=ext, cmap="twilight", vmin=-90, vmax=90),
               axs[r, 3].imshow(V / vmV, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)]
        for c, (im, lab) in enumerate(zip(ims, ["I / max I_B", "|P| / max |P|_B", "EVPA [deg E of N]", "V / V_B,99.9"])):
            fig.colorbar(im, ax=axs[r, c], shrink=0.8, label=lab)
        axs[r, 0].set_ylabel(f"{name}\nDec [µas]")
    for r, (leg, X) in enumerate((("ra", A), ("rb", Bb)), start=2):
        I, P, chi, V = comps(X)
        m = met["legs"][leg][str(inc)]["levels"][str(C)]
        dI = (I - IB) / vmI
        dP = (P - PB) / vmP
        dchi = np.ma.array(rc.wrap90(chi.data - CB.data), mask=chi.mask | CB.mask)
        dV = (V - VB) / vmV
        s = [max(np.percentile(np.abs(a), 99.9), 1e-12) for a in (dI, dP, dV)]
        ims = [axs[r, 0].imshow(dI, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[0], vmax=s[0]),
               axs[r, 1].imshow(dP, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[1], vmax=s[1]),
               axs[r, 2].imshow(dchi, origin="lower", extent=ext, cmap="RdBu_r",
                                vmin=-max(float(np.abs(dchi).max()), 1e-9), vmax=max(float(np.abs(dchi).max()), 1e-9)),
               axs[r, 3].imshow(dV, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[2], vmax=s[2])]
        labs = ["ΔI / max I_B", "Δ|P| / max |P|_B", "ΔEVPA [deg]", "ΔV / V_B,99.9"]
        for c, im in enumerate(ims):
            fig.colorbar(im, ax=axs[r, c], shrink=0.8, label=labs[c])
        name = "(a) − BLP" if leg == "ra" else "(b) − BLP"
        axs[r, 0].set_ylabel(f"{name}\nDec [µas]")
        ev = m["evpa_pixel"]["flux_weighted_mean_abs_deg"]
        axs[r, 0].set_title(f"L1 {m['l1_I']:.1e}, NMSE {m['nmse_I']:.1e}", fontsize=8)
        axs[r, 1].set_title(f"L1(|P|) {m['l1_P']:.1e}", fontsize=8)
        axs[r, 2].set_title(f"flux-weighted |ΔEVPA| {ev:.2g}°", fontsize=8)
        axs[r, 3].set_title(f"L1(V) {m['l1_V']:.1e}", fontsize=8)
    for ax in axs.ravel():
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
    for ax in axs[-1]:
        ax.set_xlabel("RA [µas]")
    for c, t in enumerate(["I", "|P|", "EVPA", "V"]):
        axs[0, c].set_title(t)
    fig.suptitle(f"{label}, i = {inc}°, 230 GHz, {npix}², C = {C}: BlackLightPlus and RAPTOR (fork blp-bbh)")
    fig.savefig(out, dpi=90)
    plt.close(fig)


def convergence(met, out, label="s455"):
    incs = sorted(met["blp"].keys(), key=int)
    fig, axs = plt.subplots(len(incs), 3, figsize=(11, 3.1 * len(incs)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for r, inc in enumerate(incs):
        for c, (key, name) in enumerate((("l1_I", "I"), ("l1_P", "|P|"), ("l1_V", "V"))):
            ax = axs[r, c]
            ra = met["legs"]["ra"][inc]
            Cs = ra["C_levels"]
            lv = ra["levels"]
            Cself = Cs[:-1]
            ax.plot(Cself, [lv[str(C)]["eps_blp"][key] for C in Cself], "o:", color=COL["blp"], lw=1.5, ms=8,
                    mfc="white", mew=2, label="BlackLightPlus self-convergence")
            ax.plot(Cself, [lv[str(C)]["eps_raptor"][key] for C in Cself], "s:", color=COL["ra"], lw=1.5, ms=8,
                    mfc="white", mew=2, label="RAPTOR (a) self-convergence")
            ax.plot(Cs, [lv[str(C)][key] for C in Cs], "s-", color=COL["ra"], lw=2, ms=8,
                    label="(a) − BlackLightPlus", mec="white")
            if "rb" in met["legs"] and inc in met["legs"]["rb"]:
                rb = met["legs"]["rb"][inc]
                ax.plot(rb["C_levels"], [max(rb["levels"][str(C)][key], 1e-16) for C in rb["C_levels"]], "D-",
                        color=COL["rb"], lw=2, ms=8, label="(b) − BlackLightPlus", mec="white")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(Cs)
            ax.set_xticklabels([str(C) for C in Cs])
            ax.grid(True, which="major", color="#e4e3df", lw=0.6)
            ax.set_title(f"i = {inc}°: normalized L1 of {name}", fontsize=9)
            if r == len(incs) - 1:
                ax.set_xlabel("step constant C (h = d_horizon / C)")
            if c == 0:
                ax.set_ylabel("normalized L1")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{label} 480²: self-convergence (against each code's finest level) and code-to-code difference")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def timing(tj, out):
    t = json.load(open(tj))
    phases = ["load", "setup", "trace", "transfer", "output"]
    shades = ["#cde2fb", "#9ec5f4", "#5598e7", "#184f95", "#0d366b"]
    rows = [(k, v) for k, v in t["runs"].items()]
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.55 * len(rows)), constrained_layout=True)
    for y, (name, v) in enumerate(rows):
        left = 0.0
        for p, colr in zip(phases, shades):
            w = v["phases_s"].get(p, 0.0)
            ax.barh(y, w, left=left, color=colr, edgecolor="white", linewidth=2, height=0.6,
                    label=p if y == 0 else None)
            left += w
        ax.text(left, y, f"  {left:.1f} s, {v['rays_per_s']:.0f} rays/s", va="center", fontsize=8, color="#0b0b0b")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([v["label"] for _, v in rows])
    ax.invert_yaxis()
    ax.set_xlabel("wall time [s] (best of 3, warm)")
    ax.set_xlim(0, max(sum(v["phases_s"].get(p, 0) for p in phases) for _, v in rows) * 1.35)
    ax.legend(loc="lower right", ncol=5, frameon=False, fontsize=8)
    ax.set_title(t.get("title", "CPU timings"), fontsize=9)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("metrics")
    ap.add_argument("figdir")
    ap.add_argument("--C", type=int, default=320)
    ap.add_argument("--npix", type=int, default=480)
    ap.add_argument("--timing")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    met = json.load(open(a.metrics))
    rc.CENTERS = None
    for inc in sorted(met["blp"].keys(), key=int):
        img_panels(a.run, int(inc), a.C, a.npix, os.path.join(a.figdir, f"images_i{int(inc):02d}.png"), met)
    convergence(met, os.path.join(a.figdir, "convergence.png"))
    if a.timing:
        timing(a.timing, os.path.join(a.figdir, "timing.png"))


if __name__ == "__main__":
    main()
