#!/usr/bin/env python3
"""Review images of the Z4c binary-black-hole data sets under BLP/figures/out/, drawn with
the paper's plotting code (paper/scripts/plot_binary.py and z4c_rays.py, imported by path
with BLP_FIG_REPO pointing at BLP/) for comparison with fig_binary (a), (e) and
fig_summary (c):

  class.png     ray-class map of z4c_amr/class.npy, rolled as in fig_binary (a), with the
                bisected ring (blue) and every third radial-scan anchor (white) of
                z4c_phasespace/, and the zoom inset on the median class-2 pixel with
                alpha_0 > 0.3, beta_0 > 4;
  levels.png    finest refinement level per finest block (z4c_amr/levels_fine.npy), as (e);
  rays.png      the deck trajectories (z4c_phasespace/deck.json, traj_*.npy), as (c);
  map_diff.png  the z4c_phasespace capture map with the cells that differ from
                figures/out/z4c_phasespace/map.npy marked.

Writes PNGs to BLP/figures/out/z4c_amr/review/.
Run: python3 BLP/figures/check_z4c.py
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("BLP_FIG_REPO", os.path.join(REPO, "BLP"))
SCRIPTS = os.path.join(REPO, "paper", "scripts")
sys.path.insert(0, SCRIPTS)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fs = load("figstyle")
z4c_rays = load("z4c_rays")
pb = load("plot_binary")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

AMR = pb.Z4C_AMR
PS = pb.PS
OUT = os.path.join(AMR, "review")
OLD_MAP = os.path.join(REPO, "figures", "out", "z4c_phasespace", "map.npy")
DPI = 250


def class_panel(cls, lim, ring, anchors):
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.22)
    ext = [-lim, lim, -lim, lim]
    cmC = ListedColormap(pb.CLASS_COLORS)
    nC = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    shown = pb.roll(np.where(cls > 2, np.nan, cls))
    rx, ry = pb.roll_pts(ring[:, 2], ring[:, 3])
    ax.imshow(shown, origin="lower", extent=ext, cmap=cmC, norm=nC, interpolation="nearest")
    ax.scatter(rx, ry, s=0.6, color=pb.C_BOUND, linewidths=0, zorder=4, rasterized=True)
    A = anchors[::3]
    ax.scatter(*pb.roll_pts(A[:, 2], A[:, 3]), s=8.0, color="white", edgecolors="k",
               linewidths=0.3, zorder=5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([-6, -3, 0, 3, 6])
    ax.tick_params(color="0.7")
    ax.set_xlabel(r"$\alpha$ [$GM/c^2$]")
    ax.set_ylabel(r"$\beta$ [$GM/c^2$]")
    ax.text(0.04, 0.955, "(a) BLP", transform=ax.transAxes, va="top", color="w")
    n = cls.shape[0]
    cc = -lim + (np.arange(n) + 0.5) * (2 * lim / n)
    jb, ia = np.nonzero(cls == 2)
    sel = (cc[ia] > 0.3) & (cc[jb] > 4.0)
    zc = (float(np.median(cc[ia[sel]])), float(np.median(cc[jb[sel]])))
    zw = 0.15
    zx, zy = pb.roll_pts(*zc)
    axi = ax.inset_axes([0.30, 0.64, 0.34, 0.34])
    axi.imshow(shown, origin="lower", extent=ext, cmap=cmC, norm=nC, interpolation="nearest")
    axi.scatter(rx, ry, s=4.0, color=pb.C_BOUND, linewidths=0, zorder=4)
    axi.set_xlim(zx - zw, zx + zw)
    axi.set_ylim(zy - zw, zy + zw)
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color("white")
    ax.indicate_inset_zoom(axi, edgecolor="white", alpha=0.9, linewidth=0.6)
    handles = [Patch(fc=pb.CLASS_COLORS[1], ec="0.3", lw=0.3, label="captured"),
               Patch(fc=pb.CLASS_COLORS[0], ec="0.3", lw=0.3, label="escaped"),
               Patch(fc=pb.CLASS_COLORS[2], ec="0.3", lw=0.3, label=r"bend $>360^\circ$"),
               Line2D([], [], marker="o", ls="", color=pb.C_BOUND, ms=3, label="bisected edge"),
               Line2D([], [], marker="o", ls="", color="white", mec="k", mew=0.4, ms=4,
                      label="radial scans")]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False,
               bbox_to_anchor=(0.08, 0.0))
    return fig, zc


def levels_panel(lvf, lim, ml):
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.22)
    ext = [-lim, lim, -lim, lim]
    im = ax.imshow(pb.roll(lvf), origin="lower", extent=ext, cmap=fs.level_cmap(ml),
                   norm=BoundaryNorm(np.arange(-0.5, ml + 1), ml + 1), interpolation="nearest")
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([-6, -3, 0, 3, 6])
    ax.set_xlabel(r"$\alpha$ [$GM/c^2$]")
    ax.set_ylabel(r"$\beta$ [$GM/c^2$]")
    ax.text(0.04, 0.955, "(e) BLP", transform=ax.transAxes, va="top", color="w")
    cax = fig.add_axes([0.25, 0.07, 0.6, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=np.arange(ml + 1))
    cb.set_label("refinement level")
    return fig


def rays_panel():
    fig = plt.figure(figsize=(4.2, 4.2))
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    deck = z4c_rays.load_deck()
    rs = z4c_rays.draw_rays(ax, deck, legend=True, label_obs=False, zoom=1.25)
    return fig, rs


def map_diff_panel(lim):
    Mn = np.load(os.path.join(PS, "map.npy"))
    Mo = np.load(OLD_MAP)
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.2)
    ext = [-lim, lim, -lim, lim]
    # map rows are the a axis: transpose to [b, a] for origin="lower", then roll
    ax.imshow(pb.roll(Mn.T), origin="lower", extent=ext, cmap=ListedColormap(
        ["black", pb.CLASS_COLORS[1]]), interpolation="nearest")
    nd = 0
    if Mo.shape == Mn.shape:
        nm = Mn.shape[0]
        axv = np.linspace(-lim, lim, nm)
        ia, jb = np.nonzero(Mo != Mn)
        nd = len(ia)
        x, y = pb.roll_pts(axv[ia], axv[jb])
        ax.scatter(x, y, s=14, facecolors="none", edgecolors="cyan", linewidths=0.6,
                   label=f"{nd} cells differ from figures/out")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.27), frameon=False)
    ax.set_xlabel(r"$\alpha$ [$GM/c^2$]")
    ax.set_ylabel(r"$\beta$ [$GM/c^2$]")
    ax.text(0.04, 0.955, f"capture map {Mn.shape[0]}$^2$ (BLP)", transform=ax.transAxes,
            va="top", color="w")
    return fig, nd


def main():
    try:
        fs.setup()
    except Exception as err:                       # no LaTeX: matplotlib's own text
        print("figstyle.setup skipped:", err)
    os.makedirs(OUT, exist_ok=True)
    rec = json.load(open(os.path.join(AMR, "records.json")))
    lim, ml = rec["lim"], rec["amr"]["max_level"]
    cls = np.load(os.path.join(AMR, "class.npy"))
    ring = np.load(os.path.join(PS, "ring.npy"))
    anchors = np.load(os.path.join(PS, "anchors.npy"))
    fig, zc = class_panel(cls, lim, ring, anchors)
    fig.savefig(os.path.join(OUT, "class.png"), dpi=DPI)
    plt.close(fig)
    del cls
    fig = levels_panel(np.load(os.path.join(AMR, "levels_fine.npy")), lim, ml)
    fig.savefig(os.path.join(OUT, "levels.png"), dpi=DPI)
    plt.close(fig)
    fig, rs = rays_panel()
    fig.savefig(os.path.join(OUT, "rays.png"), dpi=DPI, facecolor="black")
    plt.close(fig)
    fig, nd = map_diff_panel(lim)
    fig.savefig(os.path.join(OUT, "map_diff.png"), dpi=DPI)
    plt.close(fig)
    print(f"zoom centre {zc}; rays drawn {rs}; map cells differing {nd}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
