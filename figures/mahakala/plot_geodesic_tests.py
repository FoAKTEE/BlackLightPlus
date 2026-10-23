#!/usr/bin/env python3
"""Draw Mahakala figures 1, 3, 4, 5 and 6 from figures/paper/out/mahakala/geodesic_tests/
(fig_geodesic_tests.jl) and figures/paper/out/spherical_orbits/ (fig_spherical_orbits.jl):

  fig01_in_vs_out.png     t against r of the radial Schwarzschild rays in the ingoing and
                          outgoing Cartesian Kerr-Schild charts, forward (green) and backward
                          (red) in time, the horizon r = 2 as the vertical line;
  fig03_null_vs_c.png     |g^{μν}p_μp_ν| against |t| along the a = 0.9 camera ray for each C,
                          inset: the ray's path in the orbital plane;
  fig04_deflection.png    Δφ/π against b with the exact curve and the capture band; below,
                          the absolute error;
  fig05_convergence.png   max|g^{μν}p_μp_ν| against 1/Δλ for the four spherical orbits with
                          the Δλ⁴ reference line;
  fig06_spherical_orbits.png  the four spherical photon orbits at a = 1 in 3-d, the ring
                          singularity (black) and the orbit's equatorial circle (red dashed).

Writes to results/blp/figures_mahakala/. Run: python3 figures/mahakala/plot_geodesic_tests.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "figures", "out", "mahakala", "geodesic_tests")
ORB = os.path.join(REPO, "figures", "out", "spherical_orbits")
RES = os.path.join(REPO, "results", "blp", "figures_mahakala")
os.makedirs(RES, exist_ok=True)
S = json.load(open(os.path.join(OUT, "setup.json")))
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})


def fig1():
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 3.2), sharey=True, dpi=160)
    for ax, chart in zip(axs, ("ingoing", "outgoing")):
        for d, c, lab in (("backward", "r", "Backward"), ("forward", "g", "Forward")):
            M = np.load(os.path.join(OUT, f"in_vs_out_{chart}_{d}.npy"))
            ax.plot(M[:, 1], M[:, 0], color=c, lw=1.3, label=lab)
        ax.axvline(2.0, color="b", lw=0.8)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xlim(0, 52); ax.set_ylim(-100, 100)
        ax.set_xlabel(r"$r\ (GM c^{-2})$")
        ax.text(0.97, 0.04, chart.capitalize(), transform=ax.transAxes, ha="right")
        ax.legend(loc="upper right", frameon=False)
    axs[0].set_ylabel(r"$t\ (GM c^{-3})$")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig01_in_vs_out.png"))
    plt.close(fig)


def fig3():
    Cs = S["fig3"]["C"]
    fig, ax = plt.subplots(figsize=(4.4, 3.6), dpi=160)
    cmap = plt.get_cmap("viridis")
    for k, C in enumerate(Cs):
        M = np.load(os.path.join(OUT, f"null_vs_c_C{C}.npy"))
        m = M[:, 0] > 0
        ax.plot(M[m, 0], np.maximum(M[m, 1], 1e-17), lw=1.0, color=cmap(k / (len(Cs) - 1)), label=str(C))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$|t|\ (GM c^{-3})$"); ax.set_ylabel(r"$|k_\mu k^\mu|$")
    ax.legend(title=r"$\mathcal{C}$", loc="upper right", frameon=False, ncol=1, bbox_to_anchor=(1.02, 1.0))
    axi = ax.inset_axes([0.45, 0.08, 0.3, 0.3])
    M = np.load(os.path.join(OUT, f"null_vs_c_C{Cs[-1]}.npy"))
    axi.plot(M[:, 2], M[:, 3], color="b", lw=0.8)
    a = S["fig3"]["spin"]
    rh = 1 + np.sqrt(1 - a * a)
    th = np.linspace(0, 2 * np.pi, 100)
    axi.fill(np.sqrt(rh**2 + a**2) * np.cos(th), np.sqrt(rh**2 + a**2) * np.sin(th), color="0.5")
    axi.set_xlim(-5, 5); axi.set_ylim(-5, 5); axi.set_aspect("equal")
    axi.set_xlabel(r"$X$", fontsize=7, labelpad=0); axi.set_ylabel(r"$Y$", fontsize=7, labelpad=0)
    axi.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig03_null_vs_c.png"))
    plt.close(fig)


def fig4():
    M = np.load(os.path.join(OUT, "deflection.npy"))
    bm, bp = S["fig4"]["b_minus"], S["fig4"]["b_plus"]
    fig, axs = plt.subplots(2, 1, figsize=(4.4, 4.4), dpi=160, sharex=True,
                            gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axs[0]
    ax.axvspan(bm, bp, color="0.75", label="Shadow")
    gap = np.nonzero(np.diff(M[:, 0]) > 0.5)[0]            # the capture band
    ex = M[:, 2].copy()
    xs = M[:, 0].copy()
    if len(gap):
        xs = np.insert(xs, gap[0] + 1, np.nan); ex = np.insert(ex, gap[0] + 1, np.nan)
    ax.plot(xs, ex / np.pi, color="r", lw=1.2, label="exact")
    ax.plot(M[:, 0], M[:, 1] / np.pi, "o", color="b", ms=2.5, label="BlackLightPlus")
    ax.set_ylim(0, 2); ax.set_ylabel(r"$\Delta\phi/\pi$")
    ax.legend(loc="upper right", frameon=False)
    ax = axs[1]
    ax.plot(M[:, 0], np.abs(M[:, 1] - M[:, 2]), "o", color="b", ms=2.5)
    ax.axvspan(bm, bp, color="0.75")
    ax.set_yscale("log")
    ax.set_xlabel(r"$b\ (GM c^{-2})$"); ax.set_ylabel(r"abs(Error in $\Delta\phi$)")
    ax.set_xlim(-20, 20)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig04_deflection.png"))
    plt.close(fig)


def fig5():
    M = np.load(os.path.join(OUT, "convergence.npy"))
    labs = S["fig5"]["orbits"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=160)
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for io, lab in enumerate(labs):
        m = M[:, 0] == io + 1
        ax.plot(1 / M[m, 1], M[m, 2], "o-", color=cols[io], ms=4, label=f"Test {lab}")
    x = np.array([8.0, 1024.0])
    ax.plot(x, 1e-4 * (x / 8.0) ** -4, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/\Delta\lambda\ (c^3 G^{-1} M^{-1})$"); ax.set_ylabel(r"max$(|k^\mu k_\mu|)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig05_convergence.png"))
    plt.close(fig)


def fig6():
    rec = json.load(open(os.path.join(ORB, "record.json")))
    a = rec["spin"]
    fig = plt.figure(figsize=(7.6, 7.6), dpi=130)
    cols = {"A": "b", "C": "orange", "E": "g", "F": "purple"}
    for k, o in enumerate(rec["orbits"]):
        lab, r0 = o["label"], o["r0"]
        T = np.load(os.path.join(ORB, f"traj_{lab}.npy"))
        on = np.abs(T[:, 4]) < 1e-2                 # the part still on the sphere
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.plot(T[on, 1], T[on, 2], T[on, 3], color=cols[lab], lw=1.0, label=f"Test {lab}")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(a * np.cos(th), a * np.sin(th), 0 * th, color="k", lw=1.2)
        R = np.sqrt(r0**2 + a**2)
        ax.plot(R * np.cos(th), R * np.sin(th), 0 * th, "r--", lw=1.0)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
        ax.set_xlabel(r"$X (GMc^{-2})$"); ax.set_ylabel(r"$Y (GMc^{-2})$"); ax.set_zlabel(r"$Z (GMc^{-2})$")
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig06_spherical_orbits.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig3(); fig4(); fig5(); fig6()
    print("wrote", RES)
