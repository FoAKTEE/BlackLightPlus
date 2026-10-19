#!/usr/bin/env python3
"""Draw the dynamic-ray figures from BLP/figures/out/dyn_rays/ (fig_dyn_rays.jl):

  fan_<variant>.gif    the pinhole fan in the orbital plane, one frame per coordinate time in
                       setup `fan_frames`: each ray drawn from the camera to its position at
                       that time (solid: escapes, dashed: captured), coloured by launch angle,
                       the holes' horizons (r'_A = r_h,A, Lorentz-contracted) at that time, a
                       hole with r_h < 0.1 M also marked by a grey circle of radius 0.5 M;
  fan_<variant>.png    the last frame;
  bohn_<variant>.png   four snapshots of the Bohn-row rays (at the median times the bundle
                       crosses x = 8.2, 4.5 and 1.5 M, and 30 M later): lines
                       from the camera to the current position, dots at the current position,
                       solid for rays from infinity, dashed for rays from a hole;
  fan_compare.png      the last fan frame, boosted beside unboosted.

Run: python3 BLP/figures/plot_dyn_rays.py [out_dir]   (default BLP/figures/out/dyn_rays)
The Bohn panels are drawn when the directory holds them.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "dyn_rays")
S = json.load(open(os.path.join(OUT, "setup.json")))
FAN_LIM = 30.0
BOHN_LIM = 9.0
DPI = 150


def rays_of(name):
    a = np.load(os.path.join(OUT, name + ".npy"))
    idx = a[:, 0].astype(int)
    return [a[idx == i, 1:] for i in range(idx.max() + 1)]     # columns λ, t, x, y, z


def hole_state(A, t):
    r = S["orbit_radii"][A]
    ph = S["omega"] * t + S["phase"]
    s = np.array([r * np.cos(ph), r * np.sin(ph)])
    v = np.array([-r * S["omega"] * np.sin(ph), r * S["omega"] * np.cos(ph)])
    return s, v


def horizon(A, t, boosted=True, n=120):
    """Points of r'_A = r_h,A at time t: the rest-frame circle mapped by the inverse
    spatial boost, x = s + p + (1/Γ - 1)(n·p) n."""
    s, v = hole_state(A, t)
    rh = S["r_h"][A]
    phi = np.linspace(0, 2 * np.pi, n)
    p = rh * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    if boosted:
        b = np.linalg.norm(v)
        if b > 0:
            nv = v / b
            g = 1.0 / np.sqrt(1 - b * b)
            p = p + (1 / g - 1) * (p @ nv)[:, None] * nv[None, :]
    return s[None, :] + p


def upto(path, t, fate=0):
    """The path rows from the camera to coordinate time t (t decreases along the ray), the
    last point interpolated to t; None when the ray starts later than t. A captured ray
    (fate 1 or 2) that stopped before t rides on with its hole: its stop point translated
    along the hole's displacement since the stop (sampled every 0.5 M), the asymptotic
    approach to the moving horizon. Returns (xy, gone) with gone for an escaped ray that
    has left."""
    tt = path[:, 1]
    if tt[0] < t:
        return None, False
    k = np.searchsorted(-tt, -t)            # first index with tt < t
    if k >= len(tt):
        if fate in (1, 2):
            A = fate - 1
            s_end = hole_state(A, tt[-1])[0]
            taus = np.arange(tt[-1], t, -0.5)[1:]
            tail = [path[-1, 2:4] + hole_state(A, tau)[0] - s_end for tau in list(taus) + [t]]
            return np.vstack([path[:, 2:4], np.array(tail)]), False
        return path[:, 2:4], True
    k = max(k, 1)
    f = (tt[k - 1] - t) / (tt[k - 1] - tt[k])
    last = path[k - 1, 2:4] + f * (path[k, 2:4] - path[k - 1, 2:4])
    return np.vstack([path[:k, 2:4], last]), False


def entry_time(path, x_thr):
    """Coordinate time at which the ray first has x < x_thr, or its last time."""
    k = np.nonzero(path[:, 2] < x_thr)[0]
    return path[k[0], 1] if len(k) else path[-1, 1]


def draw_holes(ax, t, boosted, lw=1.0):
    """The horizons; a hole whose horizon is under 0.1 M also gets a thin grey locator
    circle of radius 0.5 M."""
    for A in range(2):
        h = horizon(A, t, boosted)
        ax.plot(h[:, 0], h[:, 1], color="k", lw=lw, zorder=5)
        if S["r_h"][A] < 0.1:
            s = hole_state(A, t)[0]
            phi = np.linspace(0, 2 * np.pi, 60)
            ax.plot(s[0] + 0.5 * np.cos(phi), s[1] + 0.5 * np.sin(phi), color="0.55", lw=0.5,
                    zorder=5)


def fan_frame(ax, paths, fates, t, boosted, cmap):
    n = len(paths)
    for i, (p, f) in enumerate(zip(paths, fates)):
        xy, _ = upto(p, t, f)
        if xy is None:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(1 - i / (n - 1)), lw=0.9,
                ls="-" if f == 0 else (0, (2.5, 2.0)), zorder=3)
    draw_holes(ax, t, boosted)
    ax.set_xlim(-FAN_LIM, FAN_LIM)
    ax.set_ylim(-FAN_LIM, FAN_LIM)
    ax.set_aspect("equal")
    ax.axis("off")


def fan_animation(variant):
    paths = rays_of("fan_" + variant)
    fates = S["fan_fates_" + variant]
    boosted = variant == "boosted"
    cmap = plt.get_cmap("viridis")
    frames = []
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    fig.subplots_adjust(0, 0, 1, 1)
    for t in S["fan_frames"]:
        ax.clear()
        fan_frame(ax, paths, fates, t, boosted, cmap)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
    plt.close(fig)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for f in frames]
    pal[0].save(os.path.join(OUT, f"fan_{variant}.gif"), save_all=True, append_images=pal[1:],
                duration=100, loop=0, optimize=True)
    frames[-1].save(os.path.join(OUT, f"fan_{variant}.png"))


def fan_compare():
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
    cmap = plt.get_cmap("viridis")
    t = S["fan_frames"][-1]
    for ax, variant, title in zip(axs, ("boosted", "unboosted"),
                                  ("boosted moving superposition", "unboosted moving superposition")):
        fan_frame(ax, rays_of("fan_" + variant), S["fan_fates_" + variant], t, variant == "boosted", cmap)
        ax.set_title(f"{title}, t = {t:.0f} M", fontsize=10)
    fig.savefig(os.path.join(OUT, "fan_compare.png"), bbox_inches="tight")
    plt.close(fig)


def bohn_panels(variant):
    paths = rays_of("bohn_" + variant)
    fates = S["bohn_fates_" + variant]
    boosted = variant == "boosted"
    # snapshots at the median times the bundle crosses x = 8.2, 4.5 and 1.5 M, then 14 M later
    snaps = [float(np.median([entry_time(p, x) for p in paths])) for x in (8.2, 4.5, 1.5)]
    snaps.append(snaps[-1] - 30.0)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.2), dpi=DPI)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.99, wspace=0.03, hspace=0.03)
    for k, (ax, label) in enumerate(zip(axs.ravel(), "ABCD")):
        t = snaps[k]
        th = t
        for p, f in zip(paths, fates):
            xy, gone = upto(p, t, f)
            if xy is None:
                continue
            ax.plot(xy[:, 0], xy[:, 1], color="k", lw=0.7,
                    ls="-" if f == 0 else (0, (1.2, 1.4)), zorder=3)
            if not gone:
                ax.plot(xy[-1, 0], xy[-1, 1], "o", color="k", ms=4.5, zorder=6)
        draw_holes(ax, th, boosted, lw=1.1)
        ax.set_xlim(-BOHN_LIM, BOHN_LIM)
        ax.set_ylim(-BOHN_LIM, BOHN_LIM)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=13)
        ax.text(0.97, 0.05, f"t = {th:.0f} M", transform=ax.transAxes, fontsize=8, ha="right")
    fig.savefig(os.path.join(OUT, f"bohn_{variant}.png"))
    plt.close(fig)


if __name__ == "__main__":
    for variant in ("boosted", "unboosted"):
        fan_animation(variant)
        if os.path.exists(os.path.join(OUT, f"bohn_{variant}.npy")):
            bohn_panels(variant)
    fan_compare()
    print("wrote", OUT)
