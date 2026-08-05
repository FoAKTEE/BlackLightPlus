#!/usr/bin/env python3
# plot_z4c_phasespace.py — α-β phase-space figure for the Z4c BBH deck computed
# by fig_z4c_phasespace.jl. Left: dense capture map, gold light-ring polylines,
# white CPU anchors, deck launch fates + escape-momentum quiver. Right: 3D deck
# trajectories (gold when bend > 360°). numpy + matplotlib only.
# Env: BLP_FIG_IN (input dir, default figures/out/z4c_phasespace; also accepts
#      an evidence dir with map80k_iXX/ring80k_iXX/ring_iXX + JSON files),
#      BLP_FIG_MA (multiangle dir for the evidence layout),
#      BLP_FIG_OUT (output png or dir), BLP_FIG_IOTA (30, evidence layout only).
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.style.use("dark_background")

C_CAP = "#ff5252"
C_ESC = "#63b3ff"
C_RING = "#ffd166"


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


def _pick(d, *names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def load_inputs():
    """Loader shim: the fig_z4c_phasespace.jl layout (deck.json + map/ring/
    anchors.npy) or the evidence layout (per-tag npy + separate multiangle)."""
    here = os.path.dirname(os.path.abspath(__file__))
    ind = os.environ.get("BLP_FIG_IN", os.path.join(here, "out", "z4c_phasespace"))
    if os.path.exists(os.path.join(ind, "deck.json")):
        with open(os.path.join(ind, "deck.json")) as f:
            deck = json.load(f)
        tag = deck["tag"]
        mpath = _pick(ind, "map.npy")
        rpath = _pick(ind, "ring.npy")
        apath = _pick(ind, "anchors.npy")
        madir = ind
    else:
        tag = "i%02d" % round(float(os.environ.get("BLP_FIG_IOTA", "30")))
        with open(os.path.join(ind, "z4c_phasespace.json")) as f:
            meta = json.load(f)
        madir = os.environ.get(
            "BLP_FIG_MA", os.path.join(os.path.dirname(ind), "z4c_multiangle"))
        with open(os.path.join(madir, "z4c_multiangle.json")) as f:
            ma = json.load(f)
        deck = dict(ma["decks"][tag], tag=tag, lim=meta["lim"])
        mpath = _pick(ind, f"map80k_{tag}.npy", f"map_{tag}.npy")
        rpath = _pick(ind, f"ring80k_{tag}.npy", f"ring2k_{tag}.npy",
                      f"ring_{tag}.npy")
        apath = _pick(ind, f"ring_{tag}.npy")
    return dict(ind=ind, tag=tag, lim=deck["lim"], d_hat=deck["d_hat"],
                rays=deck["rays"], madir=madir, mask=np.load(mpath),
                ring=np.load(rpath),
                anchors=np.load(apath) if apath else None)


def main():
    D = load_inputs()
    lim, tag = D["lim"], D["tag"]

    fig = plt.figure(figsize=(19.2, 8.8))
    ax = fig.add_axes([0.05, 0.10, 0.44, 0.78])
    cmap = ListedColormap(["#0b0e14", "#4a1418"])
    ax.imshow(D["mask"].T, origin="lower", extent=[-lim, lim, -lim, lim],
              cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    for P in ring_polylines(D["ring"]):
        ax.plot(P[:, 0], P[:, 1], color=C_RING, lw=1.7, zorder=6)
    if D["anchors"] is not None:
        A0 = D["anchors"][:: max(1, len(D["anchors"]) // 130)]  # keep dots sparse
        ax.scatter(A0[:, 2], A0[:, 3], s=5, color="w", zorder=7)
    A, Bc, U, V = [], [], [], []
    for r in D["rays"]:
        col = C_CAP if r["fate"] == "capture" else C_ESC
        ax.scatter([r["a"]], [r["b"]], color=col, s=14, zorder=7,
                   edgecolors="k", linewidths=0.3)
        if r["fate"] == "escape" and r["p_ea"] is not None:
            A.append(r["a"])
            Bc.append(r["b"])
            U.append(r["p_ea"])
            V.append(r["p_eb"])
    ax.quiver(A, Bc, U, V, color=C_ESC, scale=13.0, width=0.0045,
              headwidth=3.4, alpha=0.75, zorder=5)
    ax.set_xlabel(r"$a = \alpha\;[M]$", fontsize=11)
    ax.set_ylabel(r"$b = \beta\;[M]$", fontsize=11)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)

    ax3 = fig.add_axes([0.52, 0.10, 0.45, 0.78], projection="3d")
    for r in D["rays"]:
        T = np.load(os.path.join(D["madir"], f"traj_{r['name']}.npy"))
        x, y, z = T[:, 2], T[:, 3], T[:, 4]
        keep = (np.abs(x) < 13) & (np.abs(y) < 13) & (np.abs(z) < 13)
        col = C_CAP if r["fate"] == "capture" else C_ESC
        orb = r["bend_deg"] > 360.0
        ax3.plot(x[keep], y[keep], z[keep], color=C_RING if orb else col,
                 lw=1.1 if orb else 0.6, alpha=0.95 if orb else 0.6)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, np.pi, 20))
    for xc in (-3.3298, 3.3298):
        ax3.plot_surface(xc + 0.47 * np.cos(u) * np.sin(v),
                         0.47 * np.sin(u) * np.sin(v), 0.47 * np.cos(v),
                         color="#4a4f5a", linewidth=0)
    d = np.array(D["d_hat"])
    L = np.outer(np.linspace(0, 15, 8), d)
    ax3.plot(L[:, 0], L[:, 1], L[:, 2], "--", color="0.7", lw=0.8)
    ax3.set_xlim(-12, 12)
    ax3.set_ylim(-12, 12)
    ax3.set_zlim(-12, 12)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_axis_off()
    ax3.view_init(elev=18, azim=-55)

    out = os.environ.get("BLP_FIG_OUT", "")
    if not out.endswith(".png"):
        outdir = out if out else D["ind"]
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f"z4c_phasespace_{tag}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="black")
    plt.close(fig)
    print("WROTE", out)


if __name__ == "__main__":
    main()
