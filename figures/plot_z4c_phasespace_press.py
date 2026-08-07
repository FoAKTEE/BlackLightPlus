#!/usr/bin/env python3
"""Press-quality alpha-beta phase-space figure for the Z4c BBH deck.

Same content as plot_z4c_phasespace.py -- capture map, light-ring boundaries,
CPU anchors, launch fates, escape-momentum quiver, and the 3D deck trajectories
-- rendered for publication and press use: titled, annotated, legended, with
the panels sized to the data instead of floating in empty canvas.

Env:
  BLP_FIG_IN   input dir (default figures/out/z4c_phasespace)
  BLP_FIG_OUT  output png path or dir
  BLP_FIG_DPI  raster density (default 300)
numpy + matplotlib only.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------- palette ---
BG = "#05070d"          # page background
PANEL = "#0a0e17"       # escape region
CAP_LO = "#3b0d16"      # capture region, near the boundary
CAP_HI = "#7d1a24"      # capture region, deep interior
C_CAP = "#ff5c5c"       # captured launches
C_ESC = "#5cb3ff"       # escaping launches
C_RING = "#ffc857"      # light-ring boundary
C_TXT = "#e8ecf4"
C_DIM = "#8a93a6"

CAPMAP = LinearSegmentedColormap.from_list("cap", [PANEL, PANEL, CAP_LO, CAP_HI])


def ring_polylines(R):
    """Group refined boundary rows [comp, k, a, b, br] into closed polylines,
    each sorted by azimuth about its own centroid."""
    out = []
    for comp in np.unique(R[:, 0]):
        for k in np.unique(R[R[:, 0] == comp, 1]):
            P = R[(R[:, 0] == comp) & (R[:, 1] == k)][:, 2:4]
            if len(P) < 6:
                continue
            c = P.mean(axis=0)
            ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
            P = P[np.argsort(ang)]
            out.append(np.vstack([P, P[:1]]))
    return out


def load_inputs():
    here = os.path.dirname(os.path.abspath(__file__))
    ind = os.environ.get("BLP_FIG_IN", os.path.join(here, "out", "z4c_phasespace"))
    with open(os.path.join(ind, "deck.json")) as f:
        deck = json.load(f)
    return dict(
        ind=ind, deck=deck, tag=deck["tag"], lim=deck["lim"],
        d_hat=deck["d_hat"], rays=deck["rays"], xp=deck.get("xp", 3.3298),
        mask=np.load(os.path.join(ind, "map.npy")),
        ring=np.load(os.path.join(ind, "ring.npy")),
        anchors=np.load(os.path.join(ind, "anchors.npy")),
    )


def glow(ax, x, y, color, lw, n=4, base_alpha=0.16, zorder=5, **kw):
    """Draw a line with a soft outer glow: wide translucent passes underneath a
    crisp core. Reads as luminous rather than flat on a dark background."""
    for i in range(n, 0, -1):
        ax.plot(x, y, color=color, lw=lw + 2.6 * i, alpha=base_alpha / i,
                solid_capstyle="round", zorder=zorder - 1, **kw)
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=zorder, **kw)


def panel_a(ax, D):
    lim = D["lim"]
    ax.imshow(D["mask"].T, origin="lower", extent=[-lim, lim, -lim, lim],
              cmap=CAPMAP, vmin=0, vmax=1, interpolation="bilinear", zorder=0)

    for P in ring_polylines(D["ring"]):
        glow(ax, P[:, 0], P[:, 1], C_RING, 1.9, zorder=6)

    # Independent CPU bisection of the same boundary. Drawn above the ring glow,
    # otherwise they disappear into it.
    A0 = D["anchors"][:: max(1, len(D["anchors"]) // 320)]
    ax.scatter(A0[:, 2], A0[:, 3], s=5.5, color="white", alpha=0.92,
               linewidths=0, zorder=8)

    qa, qb, qu, qv = [], [], [], []
    for r in D["rays"]:
        cap = r["fate"] == "capture"
        ax.scatter([r["a"]], [r["b"]], s=42, zorder=9,
                   color=C_CAP if cap else C_ESC,
                   edgecolors=BG, linewidths=0.9)
        if not cap and r["p_ea"] is not None:
            qa.append(r["a"]); qb.append(r["b"])
            qu.append(r["p_ea"]); qv.append(r["p_eb"])
    ax.quiver(qa, qb, qu, qv, color=C_ESC, scale=13.0, width=0.005,
              headwidth=3.6, headlength=4.0, alpha=0.85, zorder=8)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha\ [M]$", fontsize=20, color=C_TXT, labelpad=8)
    ax.set_ylabel(r"$\beta\ [M]$", fontsize=20, color=C_TXT, labelpad=6)
    ax.tick_params(colors=C_DIM, labelsize=14, length=5, width=1.0)
    for s in ax.spines.values():
        s.set_color("#243044")
        s.set_linewidth(1.0)
    ax.grid(True, color="#ffffff", alpha=0.045, lw=0.6)
    ax.set_axisbelow(False)

    ax.text(0.035, 0.958, "Image plane", transform=ax.transAxes, fontsize=19,
            color=C_TXT, va="top", ha="left", fontweight="medium")

    handles = [
        Line2D([], [], color=C_RING, lw=3.0, label="light ring"),
        Line2D([], [], marker="o", ls="", color=C_CAP, ms=11,
               markeredgecolor=BG, label="captured"),
        Line2D([], [], marker="o", ls="", color=C_ESC, ms=11,
               markeredgecolor=BG, label="escaped"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=15,
                    frameon=True, facecolor="#0d1220", edgecolor="#243044",
                    labelcolor=C_TXT, borderpad=0.8, handletextpad=1.0,
                    labelspacing=0.6, framealpha=0.92)
    leg.set_zorder(12)


def panel_b(ax, D):
    """3D deck trajectories. Lines are split into segments and depth-shaded so
    the bundle reads as a volume rather than a flat tangle."""
    xp = D["xp"]
    view_elev, view_azim = 19.0, -60.0

    ce, sa = np.cos(np.deg2rad(view_elev)), np.sin(np.deg2rad(view_azim))
    ca = np.cos(np.deg2rad(view_azim))

    def depth(x, y, z):
        """Signed distance along the viewing direction; larger = nearer."""
        return -(x * ca + y * sa) * ce + z * np.sin(np.deg2rad(view_elev))

    # Clip on radius, not on a box: a box keeps long straight tails that shoot
    # into the corners and make the bundle read as scribble.
    lim3 = 8.6
    tracks = []
    for r in D["rays"]:
        T = np.load(os.path.join(D["ind"], f"traj_{r['name']}.npy"))
        x, y, z = T[:, 2], T[:, 3], T[:, 4]
        keep = np.sqrt(x * x + y * y + z * z) < lim3
        x, y, z = x[keep], y[keep], z[keep]
        if len(x) >= 3:
            tracks.append((r, x, y, z))

    # Frame on the bundle, not on the origin: the geodesics all arrive from the
    # observer side, so an origin-centred box leaves the action off to one side.
    allp = np.concatenate([np.column_stack(t[1:]) for t in tracks], axis=0)
    ctr = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    ctr = 0.55 * ctr  # ease back toward the holes so they stay in frame
    half3 = 6.5

    for r, x, y, z in tracks:
        orbiting = r["bend_deg"] > 360.0
        base = C_RING if orbiting else (C_CAP if r["fate"] == "capture" else C_ESC)
        lw = 2.3 if orbiting else 0.85

        pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        d = depth(x, y, z)
        dn = (d - d.min()) / max(float(np.ptp(d)), 1e-9)
        dn = 0.5 * (dn[:-1] + dn[1:])
        rgba = np.tile(np.array(to_rgba(base)), (len(segs), 1))
        rgba[:, 3] = (0.26 + 0.70 * dn) * (1.0 if orbiting else 0.55)

        lc = Line3DCollection(segs, colors=rgba,
                              linewidths=lw * (0.55 + 0.75 * dn))
        lc.set_zorder(4 if orbiting else 3)
        ax.add_collection3d(lc)

    # Horizons: an opaque core plus a rim drawn in the plane facing the camera,
    # so each hole reads as a disc from any view angle rather than a foreshortened
    # ellipse. A halo separates them from the trajectory bundle behind.
    er, ea = np.deg2rad(view_elev), np.deg2rad(view_azim)
    dv = np.array([np.cos(er) * np.cos(ea), np.cos(er) * np.sin(ea), np.sin(er)])
    e1 = np.array([-np.sin(ea), np.cos(ea), 0.0])
    e2 = np.cross(dv, e1)
    th = np.linspace(0, 2 * np.pi, 200)

    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 72), np.linspace(0, np.pi, 40))
    R = 0.47
    for xc in (-xp, xp):
        c = np.array([xc, 0.0, 0.0])
        for k, a, lw in ((3.4, 0.10, 7.0), (2.4, 0.13, 5.0), (1.7, 0.17, 3.4)):
            ring = c[:, None] + k * R * (np.cos(th) * e1[:, None]
                                         + np.sin(th) * e2[:, None])
            ax.plot(ring[0], ring[1], ring[2], color="#6f93d8", lw=lw,
                    alpha=a, zorder=6, solid_capstyle="round")
        ax.plot_surface(xc + R * np.cos(u) * np.sin(v), R * np.sin(u) * np.sin(v),
                        R * np.cos(v), color="#04060b", shade=False,
                        linewidth=0, antialiased=True, zorder=7)
        rim = c[:, None] + R * (np.cos(th) * e1[:, None] + np.sin(th) * e2[:, None])
        ax.plot(rim[0], rim[1], rim[2], color="#bcd4ff", lw=1.5, alpha=0.95,
                zorder=8)

    # Merger axis.
    ax.plot([-xp, xp], [0, 0], [0, 0], color=C_DIM, lw=0.9, alpha=0.5,
            ls=(0, (4, 3)), zorder=4)

    # Line of sight to the observer.
    d = np.array(D["d_hat"])
    L = np.outer(np.linspace(0, 8.2, 2), d)
    ax.plot(L[:, 0], L[:, 1], L[:, 2], ls=(0, (6, 5)), color="#dbe4f2",
            lw=1.4, alpha=0.9, zorder=7)
    ax.text(L[-1, 0] * 1.04, L[-1, 1] * 1.04, L[-1, 2] * 1.04 + 0.45,
            "to observer", color=C_TXT, fontsize=15, zorder=8)

    ax.set_xlim(ctr[0] - half3, ctr[0] + half3)
    ax.set_ylim(ctr[1] - half3, ctr[1] + half3)
    ax.set_zlim(ctr[2] - half3, ctr[2] + half3)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=view_elev, azim=view_azim)
    try:
        ax.set_proj_type("persp", focal_length=0.72)
    except (TypeError, ValueError):
        ax.set_proj_type("persp")

    ax.text2D(0.085, 0.945, "Light paths", transform=ax.transAxes, fontsize=19,
              color=C_TXT, va="top", ha="left", fontweight="medium")

    handles = [
        Line2D([], [], color=C_RING, lw=3.4, label="orbiting"),
        Line2D([], [], color=C_CAP, lw=2.4, label="captured"),
        Line2D([], [], color=C_ESC, lw=2.4, label="escaping"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=15,
                    frameon=True, facecolor="#0d1220", edgecolor="#243044",
                    labelcolor=C_TXT, borderpad=0.8, handletextpad=1.0,
                    labelspacing=0.6, framealpha=0.92,
                    bbox_to_anchor=(1.02, 0.02))
    leg.set_zorder(12)


def main():
    global Line3DCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F811

    D = load_inputs()
    deck = D["deck"]
    dpi = int(os.environ.get("BLP_FIG_DPI", "300"))

    fig = plt.figure(figsize=(15.5, 8.2), facecolor=BG)

    axa = fig.add_axes([0.055, 0.165, 0.375, 0.665])
    axa.set_facecolor(PANEL)
    panel_a(axa, D)

    axb = fig.add_axes([0.475, 0.075, 0.505, 0.815], projection="3d")
    axb.set_facecolor(BG)
    try:
        axb.patch.set_alpha(0.0)
    except AttributeError:
        pass
    panel_b(axb, D)

    fig.text(0.055, 0.962, "Photon capture by two black holes",
             fontsize=31, color=C_TXT, va="top", ha="left", fontweight="semibold")

    nmap = deck.get("nmap", "?")
    inc = deck.get("iota_deg", 30.0)
    fig.text(
        0.055, 0.022,
        f"Numerical-relativity merger  ·  ${nmap}^2$ rays  ·  "
        f"$\\iota = {inc:.0f}^\\circ$  ·  BlackLightPlus",
        fontsize=15, color="#7c869a", va="bottom", ha="left")

    out = os.environ.get("BLP_FIG_OUT", "")
    if not out.endswith(".png"):
        outdir = out if out else D["ind"]
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"z4c_phasespace_{D['tag']}_press.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=BG)
    plt.close(fig)
    print("WROTE", out, f"({dpi} dpi)")


if __name__ == "__main__":
    main()
