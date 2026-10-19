#!/usr/bin/env python3
"""Draw Mahakala figures 2 and 8–11 from BLP/figures/out/mahakala/grmhd/ (fig_grmhd.jl):

  fig02_trajectories.png  the image plane (log intensity) and selected rays in 3-d coloured
                          by the invariant emissivity along them;
  fig08_code_comparison.png  BLP and Blacklight images of the same snapshot and deck, the
                          difference, and the NMSE Σ|A−B|²/Σ|A|² (A = Blacklight);
  fig09_regions.png       rows R_high = 10, 40, 160: the image, the three-colour image by
                          polar region (jet green, disk red, counter-jet blue) and by the
                          ISCO and photon-orbit shells (inner green, middle red, outer blue);
  fig10_cross_sections.png  the middle-row cross sections of fig 9 with the per-region flux
                          fractions and the capture band (grey);
  fig11_profiles.png      per-shell ΔI_ν/ν³ (top) and ⟨j_ν/ν²⟩ (bottom) against radius for
                          the R_high, mass-scale and inclination ladders, averaged over the
                          ten dumps; dashed lines at the retrograde ISCO and the prograde
                          photon orbit, the band r < r_ms,- shaded.

Writes to results/blp/figures_mahakala/. Run: python3 BLP/figures/mahakala/plot_grmhd.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(REPO, "BLP", "figures", "out", "mahakala", "grmhd")
RES = os.path.join(REPO, "results", "blp", "figures_mahakala")
os.makedirs(RES, exist_ok=True)
S = {}
for f in sorted(os.listdir(OUT)):                       # one setup file per generator run
    if f.startswith("setup") and f.endswith(".json"):
        S.update(json.load(open(os.path.join(OUT, f))))
FOV = S["fov"]
GAMMA = 0.4
plt.rcParams.update({"font.size": 8})


def ext():
    return [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]


def gamma_scale(a, norm):
    return np.clip(a / norm, 0, 1) ** GAMMA


def fig9():
    rhs = S["rhigh_fig9"]
    fig, axs = plt.subplots(3, 4, figsize=(8.0, 6.2), dpi=150)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
    groups = (("disk", "jet", "counterjet"), ("ms_mid", "ms_in", "ms_out"), ("ph_mid", "ph_in", "ph_out"))
    labels = ((r"$\theta<\pi/4$", r"$\pi/4\leq\theta\leq3\pi/4$", r"$3\pi/4<\theta$"),
              (r"$r<r_{ms+}$", r"$r_{ms+}\leq r\leq r_{ms-}$", r"$r_{ms-}<r$"),
              (r"$r<r_{ph+}$", r"$r_{ph+}\leq r\leq r_{ph-}$", r"$r_{ph-}<r$"))
    for i, rh in enumerate(rhs):
        full = np.load(os.path.join(OUT, f"fig9_R{int(rh)}_full.npy"))
        norm = full.max()
        ax = axs[i, 0]
        ax.imshow(gamma_scale(full, norm), origin="lower", extent=ext(), cmap="afmhot", vmin=0, vmax=1)
        ax.text(0.04, 0.95, rf"$R_{{\rm high}}={int(rh)}$", transform=ax.transAxes, color="w", va="top", fontsize=9)
        for j, (grp, labs) in enumerate(zip(groups, labels)):
            rgb = np.zeros(full.shape + (3,))
            for c, name in enumerate(grp):                    # R, G, B
                rgb[:, :, c] = gamma_scale(np.load(os.path.join(OUT, f"fig9_R{int(rh)}_{name}.npy")), norm)
            ax = axs[i, j + 1]
            ax.imshow(rgb, origin="lower", extent=ext())
            if i == 0:
                for c, (col, lab) in enumerate(zip(("g", "r", "b"), labs)):
                    ax.text(0.04, 0.95 - 0.07 * c, lab, transform=ax.transAxes, color=col, va="top", fontsize=7)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(RES, "fig09_regions.png"))
    plt.close(fig)


def fig10():
    rhs = S["rhigh_fig9"]
    sh = np.load(os.path.join(OUT, "fig9_shadow.npy"))
    n = sh.shape[0]
    cc = -FOV / 2 + (np.arange(n) + 0.5) * FOV / n
    row = n // 2
    cap = cc[sh[row, :] == 1]
    band = (cap.min(), cap.max()) if len(cap) else (0, 0)
    groups = (("disk", "jet", "counterjet"), ("ms_mid", "ms_in", "ms_out"), ("ph_mid", "ph_in", "ph_out"))
    cols = ("r", "g", "b")
    fig, axs = plt.subplots(3, 4, figsize=(8.0, 5.4), dpi=150, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.09, wspace=0.08, hspace=0.08)
    for i, rh in enumerate(rhs):
        full = np.load(os.path.join(OUT, f"fig9_R{int(rh)}_full.npy"))
        norm = full.max(); tot = full.sum()
        ax = axs[i, 0]
        ax.plot(cc, full[row, :] / norm, "k", lw=0.9)
        ax.axvspan(*band, color="0.75")
        for j, grp in enumerate(groups):
            ax = axs[i, j + 1]
            ax.axvspan(*band, color="0.75")
            for c, name in enumerate(grp):
                im = np.load(os.path.join(OUT, f"fig9_R{int(rh)}_{name}.npy"))
                ax.plot(cc, im[row, :] / norm, color=cols[c], lw=0.9)
                ax.text(0.03, 0.93 - 0.11 * ["g", "r", "b"].index(cols[c]), f"{100 * im.sum() / tot:.1f}%",
                        transform=ax.transAxes, color=cols[c], va="top", fontsize=7)
    for ax in axs[-1, :]:
        ax.set_xlabel(r"$X\ (GMc^{-2})$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$I_\nu$")
    axs[0, 0].set_xlim(-25, 25); axs[0, 0].set_ylim(0, 1.05)
    fig.savefig(os.path.join(RES, "fig10_cross_sections.png"))
    plt.close(fig)
    return band


def nmse(A, B):
    return float(np.sum((A - B) ** 2) / np.sum(A ** 2))


def fig8():
    B = np.load(os.path.join(OUT, "fig8_blp.npy"))
    path = os.path.join(OUT, "fig8_blacklight.npz")
    if not os.path.exists(path):
        print("fig 8: no Blacklight output"); return None
    z = np.load(path)
    key = "I_nu" if "I_nu" in z.files else z.files[0]
    A = np.squeeze(z[key]).astype(float)
    # the orientation of Blacklight's array against BLP's [i_beta, i_alpha]: the dihedral
    # transform with the smallest NMSE, recorded with the result
    cands = {}
    for name, T in (("id", lambda a: a), ("T", lambda a: a.T), ("flipud", np.flipud), ("fliplr", np.fliplr),
                    ("rot90", lambda a: np.rot90(a, 1)), ("rot180", lambda a: np.rot90(a, 2)),
                    ("rot270", lambda a: np.rot90(a, 3)), ("T.flipud", lambda a: np.flipud(a.T))):
        cands[name] = nmse(T(A), B)
    best = min(cands, key=cands.get)
    Ab = {"id": A, "T": A.T, "flipud": np.flipud(A), "fliplr": np.fliplr(A), "rot90": np.rot90(A, 1),
          "rot180": np.rot90(A, 2), "rot270": np.rot90(A, 3), "T.flipud": np.flipud(A.T)}[best]
    v = cands[best]
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9), dpi=150)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.25)
    vmax = max(B.max(), Ab.max())
    for ax, im, lab in zip(axs[:2], (B, Ab), ("BLP", "Blacklight")):
        m = ax.imshow(im, origin="lower", extent=ext(), cmap="afmhot", vmin=0, vmax=vmax)
        ax.text(0.04, 0.05, lab, transform=ax.transAxes, color="w")
        fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
    d = B - Ab
    lim = np.abs(d).max()
    m = axs[2].imshow(d, origin="lower", extent=ext(), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axs[2].text(0.04, 0.05, f"NMSE = {v:.2e}", transform=axs[2].transAxes, color="k")
    fig.colorbar(m, ax=axs[2], fraction=0.046, pad=0.02)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(RES, "fig08_code_comparison.png"))
    plt.close(fig)
    return {"nmse": v, "orientation": best, "candidates": cands,
            "flux_ratio_blp_over_blacklight": float(B.sum() / Ab.sum())}


def fig2():
    R = np.load(os.path.join(OUT, "fig2_rays.npy"))
    img = np.load(os.path.join(OUT, f"fig9_R10_full.npy"))
    inc = np.radians(S["inclination_deg"])
    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    BOX = 30.0
    jmax = np.nanmax(R[:, 4]); jmin = jmax * 1e-5
    cmap = plt.get_cmap("inferno")
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for k in np.unique(R[:, 0]):
        m = R[:, 0] == k
        x, y, z, j = R[m, 1], R[m, 2], R[m, 3], np.maximum(R[m, 4], jmin)
        inside = (np.abs(x) < BOX) & (np.abs(y) < BOX) & (np.abs(z) < BOX)
        if inside.sum() < 2:
            continue
        pts = np.stack([x[inside], y[inside], z[inside]], 1)
        segs = np.stack([pts[:-1], pts[1:]], 1)
        cols = cmap(np.clip((np.log10(j[inside][:-1]) - np.log10(jmin) - 1) / 4, 0, 1))
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=1.1))
    # the image plane at BOX along the camera direction, spanning ±16 M of the field of view
    dhat = np.array([np.sin(inc), 0, np.cos(inc)]); ea = np.array([0, 1.0, 0]); eb = np.array([-np.cos(inc), 0, np.sin(inc)])
    n = img.shape[0]
    half = 16.0
    i0, i1 = int(n * (0.5 - half / FOV)), int(n * (0.5 + half / FOV))
    sub = img[i0:i1:4, i0:i1:4]
    a = -half + (np.arange(sub.shape[1]) + 0.5) * 2 * half / sub.shape[1]
    b = -half + (np.arange(sub.shape[0]) + 0.5) * 2 * half / sub.shape[0]
    A, Bm = np.meshgrid(a, b)
    P = BOX * dhat[None, None, :] + A[:, :, None] * ea[None, None, :] + Bm[:, :, None] * eb[None, None, :]
    lg = np.log10(np.maximum(sub, sub.max() * 1e-3))
    fc = plt.get_cmap("afmhot")((lg - lg.min()) / (lg.max() - lg.min()))
    ax.plot_surface(P[:, :, 0], P[:, :, 1], P[:, :, 2], facecolors=fc, rstride=1, cstride=1, shade=False,
                    linewidth=0, antialiased=False, alpha=0.9)
    ax.set_xlim(-BOX, BOX); ax.set_ylim(-BOX, BOX); ax.set_zlim(-BOX, BOX)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-55)
    fig.savefig(os.path.join(RES, "fig02_trajectories.png"))
    plt.close(fig)


def fig11():
    titles = (r"$i=17^\circ$, $\mathcal{M}=\mathcal{M}_{\rm ref}/10$", r"$i=17^\circ$, $R_{\rm high}=10$",
              r"$R_{\rm high}=80$, $\mathcal{M}=\mathcal{M}_{\rm ref}/10$")
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.2), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.09, wspace=0.25, hspace=0.08)
    rms, rph = S["r_ms"][1], S["r_ph"][0]
    for j, (ladder, title) in enumerate(zip(S["fig11_ladders"], titles)):
        for it in ladder["items"]:
            path = os.path.join(OUT, f"fig11_{it['tag']}.npy")
            if not os.path.exists(path):
                continue
            M = np.load(path)
            lab = it["label"].replace("R_high", r"$R_{\rm high}$").replace("M_ref", r"$\mathcal{M}_{\rm ref}$").replace("deg", r"$^\circ$")
            axs[0, j].plot(M[:, 0], np.maximum(M[:, 1], 1e-60), lw=1.0, label=lab)
            axs[1, j].plot(M[:, 0], np.maximum(M[:, 2], 1e-60), lw=1.0, label=lab)
        axs[0, j].set_title(title, fontsize=8)
        for ax in axs[:, j]:
            ax.axvspan(0, rms, color="0.85", zorder=0)
            ax.axvline(rms, color="k", ls="--", lw=0.8); ax.axvline(rph, color="k", ls="--", lw=0.8)
            ax.set_yscale("log"); ax.set_xlim(0, 40)
        axs[0, j].legend(frameon=False, fontsize=7)
        axs[1, j].set_xlabel(r"$r\ (GMc^{-2})$")
    axs[0, 0].set_ylabel(r"$\Delta I_\nu/\nu^3$ (emission $-$ absorption)")
    axs[1, 0].set_ylabel(r"$j_\nu/\nu^2$ (emissivity)")
    fig.savefig(os.path.join(RES, "fig11_profiles.png"))
    plt.close(fig)


if __name__ == "__main__":
    rec = {}
    if os.path.exists(os.path.join(OUT, "fig9_R10_full.npy")):
        fig9(); rec["shadow_band"] = fig10()
    if os.path.exists(os.path.join(OUT, "fig8_blp.npy")):
        rec["fig8"] = fig8()
    if os.path.exists(os.path.join(OUT, "fig2_rays.npy")):
        fig2()
    if any(f.startswith("fig11_") for f in os.listdir(OUT)):
        fig11()
    json.dump(rec, open(os.path.join(OUT, "plot_record.json"), "w"), indent=1)
    print("wrote", RES, rec)
