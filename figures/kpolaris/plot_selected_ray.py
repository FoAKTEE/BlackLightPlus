#!/usr/bin/env python3
"""Figure 4 of the KPolaris reproduction from figures/paper/out/kpolaris/selected_ray/
(fig_selected_ray.jl): fig4a_image_ray.png — (a) Stokes I at 86 GHz and i = 85° with EVPA
ticks and the selected pixel, (b) the selected ray in Cartesian Kerr–Schild coordinates over
the density slice of the orbital plane, with the intrinsic-circular-emission (magenta) and
Faraday-conversion (teal) intervals and the start of the displayed segment (orange);
fig4b_poincare.png — (c) the full transfer and (d) the transfer without conversion on that
segment as (Q, U, V)/Π on Poincaré spheres, colours from purple to yellow along the path.

Run: python3 figures/kpolaris/plot_selected_ray.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "kpolaris", "selected_ray")
R = json.load(open(os.path.join(OUT, "record.json")))
I = np.load(os.path.join(OUT, "image_I.npy")); Q = np.load(os.path.join(OUT, "image_Q.npy"))
U = np.load(os.path.join(OUT, "image_U.npy")); V = np.load(os.path.join(OUT, "image_V.npy"))
P = np.load(os.path.join(OUT, "path.npy"))          # camera → source rows
RP = np.load(os.path.join(OUT, "replay.npy"))       # segment rows, source → camera order
rho = np.load(os.path.join(OUT, "slice_rho.npy"))
fov = R["fov"]; n = I.shape[0]; lim = fov / 2
ext = [-lim, lim, -lim, lim]


def panel_ab():
    fig = plt.figure(figsize=(9.2, 4.6), dpi=150)
    ax = fig.add_axes([0.07, 0.22, 0.36, 0.70])
    im = ax.imshow(I / I.max(), origin="lower", extent=ext, cmap="afmhot", vmin=0, vmax=1)
    # EVPA ticks on a coarse grid: χ = ½ atan2(U, Q) east of north, north = +β, east = −α
    step = n // 20
    cc = -lim + (np.arange(n) + 0.5) * fov / n
    for j in range(step // 2, n, step):
        for i in range(step // 2, n, step):
            if I[j, i] < 0.05 * I.max():
                continue
            chi = 0.5 * np.arctan2(U[j, i], Q[j, i])
            m = np.hypot(Q[j, i], U[j, i]) / I[j, i]
            L = 0.55 * step * fov / n * min(m / 0.5, 1.0) + 0.1
            da, db = -np.sin(chi) * L, np.cos(chi) * L
            ax.plot([cc[i] - da, cc[i] + da], [cc[j] - db, cc[j] + db], color="w", lw=0.7)
    a, b = R["pixel_ab"]
    ax.plot([a], [b], "o", mfc="none", mec="cyan", ms=9, mew=1.4)
    ax.annotate("selected ray", (a, b), xytext=(a - 2, b - 9), color="cyan", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="cyan", lw=0.8))
    ax.set_xlabel(r"$x_{\rm screen}$ [$M$]", fontsize=9); ax.set_ylabel(r"$y_{\rm screen}$ [$M$]", fontsize=9)
    ax.set_title("a   The observed image", fontsize=10, loc="left")
    ax.text(0.03, 0.95, "torus · 86 GHz · 85°", transform=ax.transAxes, color="w", fontsize=8, va="top")
    cax = fig.add_axes([0.07, 0.06, 0.36, 0.025])
    fig.colorbar(im, cax=cax, orientation="horizontal").set_label(r"$I/I_{\max}$", fontsize=8)
    cax.tick_params(labelsize=7)
    # (b) the ray in 3D over the density slice
    ax3 = fig.add_axes([0.47, 0.02, 0.52, 0.90], projection="3d")
    half = R["slice_half"]; ns = R["slice_n"]
    g = np.linspace(-half, half, ns); X, Y = np.meshgrid(g, g)
    lr = np.log10(np.maximum(rho, 1e-8))
    vmax = np.percentile(lr, 99.5); vmin = vmax - 3.0
    ax3.contourf(X, Y, np.clip(lr, vmin, vmax), zdir="z", offset=0.0,
                 levels=np.linspace(vmin, vmax, 16), cmap="Greys", alpha=0.35, antialiased=True)
    x, y, z = P[:, 1], P[:, 2], P[:, 3]
    inside = (np.abs(x) < half) & (np.abs(y) < half) & (np.abs(z) < 0.6 * half)
    ax3.plot(x[inside], y[inside], z[inside], color="0.3", lw=1.3, zorder=10)
    k5e, k95e = R["emission_interval"]; k5c, k95c = R["conversion_interval"]
    def span(k5, k95, color, label):
        if k5 <= 0 or k95 <= 0:
            return
        lo, hi = sorted((k5 - 1, k95 - 1))
        ax3.plot(x[lo:hi + 1], y[lo:hi + 1], z[lo:hi + 1], color=color, lw=3.2, label=label, zorder=11)
    span(k5c, k95c, "teal", "Faraday conversion")
    ax3.lines[-1].set_linewidth(5.5)
    span(k5e, k95e, "magenta", "Intrinsic circular emission")
    ax3.lines[-1].set_linewidth(2.4)
    ks = R["segment_start_sample"] - 1
    ax3.plot([x[ks]], [y[ks]], [z[ks]], "o", color="orange", ms=6, label="Start of panels c, d", zorder=12)
    # arrow toward the observer: the last in-box samples point along the outgoing ray
    kin = np.nonzero(inside)[0]
    k1, k2 = kin[0], kin[min(len(kin) - 1, 40)]
    d = np.array([x[k1] - x[k2], y[k1] - y[k2], z[k1] - z[k2]]); d *= 6.0 / np.linalg.norm(d)
    ax3.quiver(x[k1], y[k1], z[k1], d[0], d[1], d[2], color="0.3", arrow_length_ratio=0.35, lw=1.2)
    u = np.linspace(0, 2 * np.pi, 60); rh = 1 + np.sqrt(1 - 0.81)
    ax3.plot(rh * np.cos(u), rh * np.sin(u), 0 * u, color="k", lw=2)
    ax3.set_xlim(-half, half); ax3.set_ylim(-half, half); ax3.set_zlim(-half * 0.6, half * 0.6)
    ax3.set_xlabel("x [M]", fontsize=8); ax3.set_ylabel("y [M]", fontsize=8); ax3.set_zlabel("z [M]", fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.view_init(elev=28, azim=-55)
    ax3.set_box_aspect((1, 1, 0.6))
    ax3.set_title("b   A ray through the emitting flow", fontsize=10, loc="left")
    ax3.legend(fontsize=7, loc="upper left", frameon=False)
    fig.savefig(os.path.join(OUT, "fig4a_image_ray.png"))


def sphere(ax, S, title, pi_frac, v_frac):
    uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color="0.8",
                      lw=0.3, alpha=0.6)
    pol = np.sqrt(S[:, 1] ** 2 + S[:, 2] ** 2 + S[:, 3] ** 2)
    ok = pol > 0
    q, u, v = (S[ok, 1] / pol[ok], S[ok, 2] / pol[ok], S[ok, 3] / pol[ok])
    cols = plt.get_cmap("viridis")(np.linspace(0, 1, len(q)))
    for k in range(len(q) - 1):
        ax.plot(q[k:k + 2], u[k:k + 2], v[k:k + 2], color=cols[k], lw=1.1)
    ax.plot([q[0]], [u[0]], [v[0]], "o", color="orange", ms=6, label="Segment start")
    ax.plot([q[-1]], [u[-1]], [v[-1]], "*", color="tab:blue", ms=9, label="Observed state")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_xlabel(r"$Q/\Pi$", fontsize=8); ax.set_ylabel(r"$U/\Pi$", fontsize=8); ax.set_zlabel(r"$V/\Pi$", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text2D(0.5, -0.04, f"Total polarization: {100 * pi_frac:.1f}%; final V/I = {100 * v_frac:.1f}%",
              transform=ax.transAxes, ha="center", fontsize=8)


def panel_cd():
    fig = plt.figure(figsize=(9.2, 4.6), dpi=150)
    ax1 = fig.add_axes([0.02, 0.08, 0.46, 0.86], projection="3d")
    ax2 = fig.add_axes([0.52, 0.08, 0.46, 0.86], projection="3d")
    sphere(ax1, RP[:, 0:4], "c   Full polarized transfer", R["full_pi_over_i"], R["full_v_over_i"])
    sphere(ax2, RP[:, 4:8], "d   No Faraday conversion in this segment", R["noconv_pi_over_i"],
           R["noconv_v_over_i"])
    ax1.legend(fontsize=7, loc="upper left", frameon=False)
    fig.text(0.5, 0.01, "Curves progress from purple to yellow in recorded path order", ha="center",
             fontsize=7, color="0.4")
    fig.savefig(os.path.join(OUT, "fig4b_poincare.png"))


if __name__ == "__main__":
    panel_ab()
    panel_cd()
    print("drawn", OUT)
