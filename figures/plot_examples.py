#!/usr/bin/env python3
# plot_examples.py — PNG panels for the BlackLightPlus demonstration renders.
# Loads the .npy images written by fig_examples.jl from figures/out/ and saves
# PNGs alongside them: the exact-Kerr equatorial-disk intensity map and, when
# the polarized torus outputs are present, Stokes I, the linear-polarization
# fraction |P|/I, and Q/U panels. Dependencies: numpy and matplotlib only.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(name):
    path = os.path.join(OUT, name)
    return np.load(path) if os.path.exists(path) else None


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# (a) analytic equatorial disk: sqrt stretch brings out the photon rings
I = load("analytic_disk_I.npy")
if I is not None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(np.sqrt(np.maximum(I, 0.0)), origin="lower", cmap="afmhot")
    ax.set_title("Exact-Kerr equatorial disk, a = 0.94, i = 17°, n = 0–2")
    ax.set_xlabel("α [px]")
    ax.set_ylabel("β [px]")
    fig.colorbar(im, ax=ax, label=r"$\sqrt{I}$ [arb.]")
    save(fig, "analytic_disk_I.png")

# (b) polarized torus: Stokes I, |P|/I, Q, U
S = {k: load("torus_%s.npy" % k) for k in "IQUV"}
if all(v is not None for v in S.values()):
    I, Q, U = S["I"], S["Q"], S["U"]
    P = np.hypot(Q, U)
    lit = I > 1e-3 * I.max()
    frac = np.where(lit, P / np.where(lit, I, 1.0), np.nan)
    qu = max(np.abs(Q).max(), np.abs(U).max())
    panels = [
        (I, "Stokes I", "afmhot", {}),
        (frac, "|P|/I", "viridis", dict(vmin=0.0, vmax=1.0)),
        (Q, "Stokes Q", "RdBu_r", dict(vmin=-qu, vmax=qu)),
        (U, "Stokes U", "RdBu_r", dict(vmin=-qu, vmax=qu)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (img, title, cmap, kw) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xlabel("α [px]")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("β [px]")
    fig.suptitle("Polarized torus (PolishDoughnut), a = 0.94, i = 60°, 230 GHz")
    save(fig, "torus_IQUV.png")
