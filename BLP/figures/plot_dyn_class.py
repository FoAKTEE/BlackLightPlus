#!/usr/bin/env python3
"""Draw the animated class and refinement-level maps of fig_dyn_class.jl (the left column
of the paper's binary figure, one frame per observer time):

  class_<q>.gif / class_<q>.png   two panels per frame — (a) the ray classes (escaped
                                  black, captured by hole 1 orange, by hole 2 blue, escaped
                                  after bending > 360° yellow, undetermined grey) with the
                                  primary's image position at its passage time marked and an
                                  inset of ±0.6 M about the secondary's shadow (its class-2
                                  pixels' median, else its flat-light-travel position); (e) the
                                  finest refinement level per finest tile — and the first
                                  frame as a PNG.

Run: python3 BLP/figures/plot_dyn_class.py [out_dir]   (default BLP/figures/out/dyn_class_q100)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "dyn_class_q100")
S = json.load(open(os.path.join(OUT, "setup.json")))
CLASS_COLORS = ["black", "#f98e09", "#4ea3dd", "#fcffa4", "0.5"]
TICKS = [-12, -6, 0, 6, 12] if S["lim"] >= 10 else [-6, -3, 0, 3, 6]
SI, CI = np.sin(np.radians(S["inc_deg"])), np.cos(np.radians(S["inc_deg"]))
DHAT = np.array([SI, 0.0, CI])
ZOOM = 0.6                 # half-width of the inset about the secondary's shadow


def hole_pos(A, t):
    m1, m2 = S["masses"]
    r = (m2 / (m1 + m2) * S["separation"], -m1 / (m1 + m2) * S["separation"])[A]
    ph = S["omega"] * t
    return np.array([r * np.cos(ph), r * np.sin(ph), 0.0])


def hole_image(A, t_obs):
    """Image position (α, β) = (y, -x cos i + z sin i) of hole A at the time a ray received
    at t_obs passes it: t_p = t_obs - (DCAM - d̂·s_A(t_p)), flat light travel, iterated."""
    tp = t_obs - S["dcam"]
    for _ in range(4):
        tp = t_obs - (S["dcam"] - DHAT @ hole_pos(A, tp))
    s = hole_pos(A, tp)
    return s[1], -s[0] * CI + s[2] * SI


def frame_figure(k):
    lim = S["lim"]
    ext = [-lim, lim, -lim, lim]
    cls = np.load(os.path.join(OUT, f"class_{k:03d}.npy"))
    lvf = np.load(os.path.join(OUT, f"levels_fine_{k:03d}.npy"))
    ml = S["maxlevel"]
    fig, axs = plt.subplots(2, 1, figsize=(3.6, 7.6), dpi=140)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.07, hspace=0.28)
    ax = axs[0]
    ax.imshow(cls, origin="lower", extent=ext, cmap=ListedColormap(CLASS_COLORS),
              norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), interpolation="nearest")
    t = S["times"][k]
    ha, hb = hole_image(0, t)                   # the primary; the secondary is located by its shadow
    ax.plot([ha], [hb], "+", color="w", ms=6, mew=0.8)
    # inset: the secondary's shadow and eyebrows at the finest resolution
    n = cls.shape[0]
    cc = -lim + (np.arange(n) + 0.5) * (2 * lim / n)
    jb, ia = np.nonzero(cls == 2)
    zx, zy = (float(np.median(cc[ia])), float(np.median(cc[jb]))) if len(ia) else hole_image(1, t)
    axi = ax.inset_axes([0.60, 0.60, 0.36, 0.36])
    axi.imshow(cls, origin="lower", extent=ext, cmap=ListedColormap(CLASS_COLORS),
               norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), interpolation="nearest")
    axi.set_xlim(zx - ZOOM, zx + ZOOM); axi.set_ylim(zy - ZOOM, zy + ZOOM)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("w"); sp.set_linewidth(0.8)
    ax.add_patch(plt.Rectangle((zx - ZOOM, zy - ZOOM), 2 * ZOOM, 2 * ZOOM, fill=False,
                               ec="w", lw=0.6))
    ax.text(0.04, 0.955, "(a)", transform=ax.transAxes, va="top", color="w", fontsize=9)
    ax.text(0.96, 0.555, f"t = {t:.0f} M", transform=ax.transAxes, va="top",
            ha="right", color="w", fontsize=8)
    handles = [Patch(fc=CLASS_COLORS[1], ec="0.3", lw=0.3, label="captured, hole 1"),
               Patch(fc=CLASS_COLORS[2], ec="0.3", lw=0.3, label="captured, hole 2"),
               Patch(fc=CLASS_COLORS[0], ec="0.3", lw=0.3, label="escaped"),
               Patch(fc=CLASS_COLORS[3], ec="0.3", lw=0.3, label=r"bend $>360^\circ$")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False, ncol=2,
              labelcolor="w", handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0.0, -0.005))
    ax = axs[1]
    im = ax.imshow(lvf, origin="lower", extent=ext, cmap=plt.get_cmap("plasma", ml + 1),
                   norm=BoundaryNorm(np.arange(-0.5, ml + 1), ml + 1), interpolation="nearest")
    ax.text(0.04, 0.955, "(e)", transform=ax.transAxes, va="top", color="w", fontsize=9)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", ticks=np.arange(ml + 1), pad=0.16,
                      fraction=0.05, aspect=25)
    cb.set_label("refinement level", labelpad=2, fontsize=8)
    cb.ax.tick_params(labelsize=7, direction="out", length=2)
    for ax in axs:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xticks(TICKS); ax.set_yticks(TICKS)
        ax.tick_params(color="w", labelsize=7.5)
        ax.set_xlabel(r"$\alpha$ [$GM/c^2$]", labelpad=1.5, fontsize=8)
        ax.set_ylabel(r"$\beta$ [$GM/c^2$]", labelpad=1, fontsize=8)
    return fig


if __name__ == "__main__":
    q = int(round(S["q"]))
    frames = []
    n = len(S["times"])
    for k in range(n):
        if not os.path.exists(os.path.join(OUT, f"class_{k:03d}.npy")):
            break
        fig = frame_figure(k)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        if k == 0:
            fig.savefig(os.path.join(OUT, f"class_q{q}.png"))
        plt.close(fig)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for f in frames]
    pal[0].save(os.path.join(OUT, f"class_q{q}.gif"), save_all=True, append_images=pal[1:],
                duration=150, loop=0, optimize=True)
    print("wrote", len(frames), "frames to", OUT)
