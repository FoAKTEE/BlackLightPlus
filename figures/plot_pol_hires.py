#!/usr/bin/env python3
# plot_pol_hires.py — figure for the high-resolution polarized render: Stokes I
# (afmhot) with white EVPA polarization ticks, plus a Q/U/V column (RdBu_r,
# symmetric limits, colorbars). Reads the I/Q/U/V.npy arrays written by
# fig_pol_hires.jl and saves pol_hires.png. Dependencies: numpy and matplotlib.
# Env: BLP_FIG_IN (input dir, default figures/out/pol_hires), BLP_FIG_OUT
# (output dir, default = input dir).

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.environ.get("BLP_FIG_IN", os.path.join(HERE, "out", "pol_hires"))
OUTDIR = os.environ.get("BLP_FIG_OUT", IN)
FOV = 30.0          # full field of view (CameraConfig.fov), image spans +-FOV/2
# fig_pol_hires.jl arrays are indexed [beta, alpha] (axis 0 = beta, axis 1 = alpha)


def main():
    I = np.load(os.path.join(IN, "I.npy"))
    Q = np.load(os.path.join(IN, "Q.npy"))
    U = np.load(os.path.join(IN, "U.npy"))
    V = np.load(os.path.join(IN, "V.npy"))
    res = I.shape[0]
    half = FOV / 2
    ext = [-half, half, -half, half]
    plt.style.use("default")
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="black")
    gs = GridSpec(3, 2, width_ratios=[2.0, 1.0], height_ratios=[1, 1, 1],
                  wspace=0.12, hspace=0.18)

    # main panel: Stokes I + EVPA ticks
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(I, origin="lower", extent=ext, cmap="afmhot", vmin=0,
              vmax=np.nanmax(I))
    m = np.hypot(Q, U) / np.where(I > 0, I, np.inf)
    chi = 0.5 * np.arctan2(U, Q)
    step = max(res // 40, 1)
    Imax = np.nanmax(I)
    for ib in range(0, res, step):
        for ia in range(0, res, step):
            if I[ib, ia] <= 0.03 * Imax:
                continue
            mm = m[ib, ia]
            xc = (ia + 0.5) / res * FOV - half
            yc = (ib + 0.5) / res * FOV - half
            if mm < 0.02:
                continue
            # EVPA chi from north (+beta) toward east (-alpha)
            c = chi[ib, ia]
            dx, dy = -np.sin(c), np.cos(c)
            Lh = np.clip(mm, 0, 0.6) * (FOV / res) * step * 0.95
            ax.plot([xc - Lh * dx / 2, xc + Lh * dx / 2],
                    [yc - Lh * dy / 2, yc + Lh * dy / 2],
                    "-", color="w", lw=0.8, alpha=0.85)
    ax.set_xlabel(r"$\alpha$ [$GM/c^2$]", color="0.8")
    ax.set_ylabel(r"$\beta$ [$GM/c^2$]", color="0.8")
    ax.tick_params(colors="0.7")

    # side column: Q, U, V on symmetric diverging limits
    for row, (S, name) in enumerate(((Q, "Q"), (U, "U"), (V, "V"))):
        axi = fig.add_subplot(gs[row, 1])
        lim = np.nanmax(np.abs(S)) or 1.0
        im = axi.imshow(S, origin="lower", extent=ext, cmap="RdBu_r",
                        vmin=-lim, vmax=lim)
        axi.set_title(name, color="w", fontsize=9, pad=2)
        axi.tick_params(colors="0.6", labelsize=7)
        cb = fig.colorbar(im, ax=axi, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors="0.6", labelsize=6)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "pol_hires.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="black")
    print("wrote", out)


if __name__ == "__main__":
    main()
