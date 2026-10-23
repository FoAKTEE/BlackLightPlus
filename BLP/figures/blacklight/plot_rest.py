#!/usr/bin/env python3
"""Figures 7, 9, 11-17 of the Blacklight-paper reproduction from out/blacklight/:
fig07_slow.png, fig09_adaptive.png, fig11_auxiliary_length.png, fig12_auxiliary_average.png,
fig13_render_edge.png, fig14_render_face.png, fig15_true.png, fig16_scaling_strong.png,
fig17_scaling_resolution.png. Each is drawn when its data directory exists.
Run: python3 BLP/figures/blacklight/plot_rest.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "BLP", "figures", "out", "blacklight")
RES = os.path.join(ROOT, "results", "blp", "figures_blacklight")
os.makedirs(RES, exist_ok=True)
FOV = 24.0
EXT = [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]
MSUN = 1.989e33; GN = 6.674e-8; CL = 2.998e10
L_CM = GN * 4.152e6 * MSUN / CL ** 2
load = lambda d, n: np.load(os.path.join(OUT, d, n + ".npy"))
have = lambda d, n="record.json": os.path.exists(os.path.join(OUT, d, n))


def panel(ax, A, title=None, cmap="inferno", vmin=None, vmax=None, ext=EXT, log=False):
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.imshow(A, origin="lower", extent=ext, cmap=cmap, vmin=None if log else vmin,
                   vmax=None if log else vmax, norm=norm, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def xyz_to_srgb(X, Y, Z):
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.einsum("ij,...j->...i", M, np.stack([X, Y, Z], axis=-1))
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


# figure 7: slow light
if have("slowlight"):
    rec = json.load(open(os.path.join(OUT, "slowlight", "record.json")))["result"]
    fast, slow = load("slowlight", "fast_I"), load("slowlight", "slow_I")
    vmax = max(fast.max(), slow.max())
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2), dpi=160)
    im = panel(axs[0], fast, f"fast light, t = {rec['fast']['time']:.0f} M: {rec['fast']['flux_jy']:.2f} Jy", vmin=0, vmax=vmax)
    im = panel(axs[1], slow, f"slow light, t_obs = {rec['slow']['t_obs']:.0f} M: {rec['slow']['flux_jy']:.2f} Jy", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    fig.savefig(os.path.join(RES, "fig07_slow.png"), bbox_inches="tight"); plt.close(fig)

# figure 9: adaptive refinement
if have("amr"):
    rec = json.load(open(os.path.join(OUT, "amr", "record.json")))
    rI, rV, rT = load("amr", "root_I"), load("amr", "root_V"), load("amr", "root_tau")
    aI, aV, fT = load("amr", "amr_I"), load("amr", "amr_V"), load("amr", "fine_tau")
    lv = load("amr", "levels").astype(int)                      # [beta block, alpha block], 4x4
    # refined tau: each base block of level L shown at its pixel size 2^(3-L) fine pixels
    aT = np.zeros_like(fT)
    nb = lv.shape[0]; bpx = fT.shape[0] // nb
    for bj in range(nb):
        for bi in range(nb):
            s = 2 ** (3 - lv[bj, bi])
            blk = fT[bj * bpx:(bj + 1) * bpx, bi * bpx:(bi + 1) * bpx]
            coarse = blk.reshape(bpx // s, s, bpx // s, s).mean(axis=(1, 3))
            aT[bj * bpx:(bj + 1) * bpx, bi * bpx:(bi + 1) * bpx] = np.kron(coarse, np.ones((s, s)))
    fig, axs = plt.subplots(2, 4, figsize=(12.0, 6.0), dpi=160)
    vI = max(rI.max(), aI.max()); vV = max(np.abs(rV).max(), np.abs(aV).max()); vT = max(rT.max(), aT.max())
    lcm = matplotlib.colors.ListedColormap(["#1f1f6e", "#7a2a8f", "#d9684a", "#f6e55a"])
    for row, (I, T, V, LV) in enumerate(((rI, rT, rV, np.zeros_like(lv)), (aI, aT, aV, lv))):
        fig.colorbar(panel(axs[row, 0], I, "$I_\\nu$" if row == 0 else None, vmin=0, vmax=vI), ax=axs[row, 0], fraction=0.046)
        fig.colorbar(panel(axs[row, 1], T, r"$\tau$" if row == 0 else None, cmap="viridis", vmin=0, vmax=vT), ax=axs[row, 1], fraction=0.046)
        fig.colorbar(panel(axs[row, 2], V, "$V$" if row == 0 else None, cmap="RdBu_r", vmin=-vV, vmax=vV), ax=axs[row, 2], fraction=0.046)
        im = axs[row, 3].imshow(LV, origin="lower", extent=EXT, cmap=lcm, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[row, 3].tick_params(labelsize=7)
        fig.colorbar(im, ax=axs[row, 3], fraction=0.046, ticks=[0, 1, 2, 3])
    axs[0, 3].set_title("refinement level", fontsize=8)
    axs[0, 0].set_ylabel("root $32^2$", fontsize=8); axs[1, 0].set_ylabel(f"refined, {rec['amr_rays']} rays", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig09_adaptive.png")); plt.close(fig)

# figures 11-14
if have("auxiliary"):
    t, s = load("auxiliary", "aux_time"), load("auxiliary", "aux_length")
    vmax = max(t.max(), s.max())
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2), dpi=160)
    panel(axs[0], t, "elapsed time [$GM/c^3$]", cmap="viridis", vmin=0, vmax=vmax)
    im = panel(axs[1], s, "proper length [$GM/c^2$]", cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    fig.savefig(os.path.join(RES, "fig11_auxiliary_length.png"), bbox_inches="tight"); plt.close(fig)

    lam, Ij, tau = load("auxiliary", "aux_lambda"), load("auxiliary", "aux_Ij"), load("auxiliary", "aux_tau")
    tl, tj, tt = load("auxiliary", "aux_theta_lambda"), load("auxiliary", "aux_theta_j"), load("auxiliary", "aux_theta_tau")
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 6.0), dpi=160)
    fig.colorbar(panel(axs[0, 0], lam, r"$\lambda$ [$GM/c^2$]", cmap="viridis"), ax=axs[0, 0], fraction=0.046)
    fig.colorbar(panel(axs[0, 1], Ij, r"$I_j$ [erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Hz$^{-1}$]", cmap="viridis"), ax=axs[0, 1], fraction=0.046)
    fig.colorbar(panel(axs[0, 2], tau, r"$\tau$", cmap="viridis"), ax=axs[0, 2], fraction=0.046)
    th = np.concatenate([tl.ravel(), tj.ravel(), tt.ravel()]); th = th[th > 0]
    lo, hi = np.percentile(th, 0.5), np.percentile(th, 99.5)
    for ax, A, title in zip(axs[1], (tl, tj, tt), (r"$\langle\Theta_e\rangle_\lambda$", r"$\langle\Theta_e\rangle_{I_j}$", r"$\langle\Theta_e\rangle_\tau$")):
        im = panel(ax, np.where(A > 0, A, np.nan), title, cmap="plasma", vmin=lo, vmax=hi, log=True)
    fig.colorbar(im, ax=axs[1], fraction=0.015, pad=0.02, label=r"$\langle\Theta_e\rangle$")
    fig.savefig(os.path.join(RES, "fig12_auxiliary_average.png"), bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(11, 3.0), dpi=160)
    for ax, (tag, fov, title) in zip(axs, (("wide_flat", 60.0, "flat"), ("wide_kerr", 60.0, "curved"),
                                           ("zoom_flat", 24.0, "flat, $z < -r_\\mathrm{h}$"), ("zoom_kerr", 24.0, "curved, $z < -r_\\mathrm{h}$"))):
        rgb = xyz_to_srgb(*(load("auxiliary", f"edge_{tag}_{c}") for c in "XYZ"))
        ax.imshow(rgb, origin="lower", extent=[-fov / 2, fov / 2, -fov / 2, fov / 2], interpolation="nearest")
        ax.set_title(title, fontsize=8); ax.tick_params(labelsize=7)
    arec = json.load(open(os.path.join(OUT, "auxiliary", "record.json")))["result"]
    fig.suptitle(r"blue: $\rho > %.1\times10^{%d}$ g cm$^{-3}$ ($\tau_s = 1/30\,M$); red: $\sigma > 1$ ($\tau_s = 1/25\,M$); green: $\beta^{-1} = 1$ crossings (opacity 0.25)" % (arec["rho_thresholds_cgs"]["edge"] / 10 ** np.floor(np.log10(arec["rho_thresholds_cgs"]["edge"])), np.floor(np.log10(arec["rho_thresholds_cgs"]["edge"]))), fontsize=8)
    fig.savefig(os.path.join(RES, "fig13_render_edge.png"), bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.1), dpi=160)
    axs[0].imshow(xyz_to_srgb(*(load("auxiliary", f"face_rho_{c}") for c in "XYZ")), origin="lower", extent=EXT, interpolation="nearest")
    axs[0].set_title(r"$\rho > \rho_\mathrm{max}/5$ ($\tau_s = 1/5\,M$)", fontsize=8)
    im = panel(axs[1], load("auxiliary", "face_I"), "$I_\\nu$ at 230 GHz", vmin=0)
    axs[2].imshow(xyz_to_srgb(*(load("auxiliary", f"face_theta_{c}") for c in "XYZ")), origin="lower", extent=EXT, interpolation="nearest")
    axs[2].set_title(r"$\Theta_e > %g$ ($\tau_s = 1/M$)" % arec["theta_hot"], fontsize=8)
    for ax in axs: ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(RES, "fig14_render_face.png"), bbox_inches="tight"); plt.close(fig)

# figure 15: true colour (CIE 1931 2-degree observer, Wyman, Sloan & Shirley 2013 fit)
def cmf(lam_nm):
    g = lambda x, mu, s1, s2: np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)
    x = 1.056 * g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * g(lam_nm, 459.0, 26.0, 13.8)
    return x, y, z

if have("truecolor"):
    rec = json.load(open(os.path.join(OUT, "truecolor", "record.json")))["result"]
    nus = np.array(rec["frequencies_hz"])
    lam_p = 550.0 * 2.3e11 / nus                              # boosted wavelengths in nm
    order = np.argsort(lam_p)
    fig, axs = plt.subplots(2, 3, figsize=(9.0, 6.0), dpi=160)
    for col, (inc, title) in enumerate(((0, "face-on"), (45, r"$\theta = 45^\circ$"), (90, "edge-on"))):
        imgs = [np.load(os.path.join(OUT, "truecolor", f"i{inc:02d}_f{k + 1:02d}_I.npy")) for k in range(12)]
        Il = np.array([nus[k] ** 2 * imgs[k] for k in range(12)])[order]     # I'_lambda up to constants
        lp = lam_p[order]
        X = np.zeros_like(imgs[0]); Y = np.zeros_like(X); Z = np.zeros_like(X)
        xb, yb, zb = cmf(lp)
        for k in range(11):
            dl = lp[k + 1] - lp[k]
            X += 0.5 * (Il[k] * xb[k] + Il[k + 1] * xb[k + 1]) * dl
            Y += 0.5 * (Il[k] * yb[k] + Il[k + 1] * yb[k + 1]) * dl
            Z += 0.5 * (Il[k] * zb[k] + Il[k + 1] * zb[k + 1]) * dl
        expo = np.percentile(Y, 99.7)
        rgb = xyz_to_srgb(X / expo, Y / expo, Z / expo)
        axs[0, col].imshow(rgb, origin="lower", extent=EXT, interpolation="nearest")
        axs[0, col].set_title(title, fontsize=8); axs[0, col].tick_params(labelsize=7)
        mono = np.load(os.path.join(OUT, "truecolor", f"i{inc:02d}_230_I.npy"))
        im = panel(axs[1, col], mono, None, vmin=0)
    fig.colorbar(im, ax=axs[1], fraction=0.015, pad=0.02, label=r"$I_\nu$ at 230 GHz")
    fig.savefig(os.path.join(RES, "fig15_true.png"), bbox_inches="tight"); plt.close(fig)

# figures 16-17: scaling
if have("scaling", "results.jsonl"):
    txt = open(os.path.join(OUT, "scaling", "results.jsonl")).read()
    rows, pos, dec = [], 0, json.JSONDecoder()
    while True:
        while pos < len(txt) and txt[pos].isspace():
            pos += 1
        if pos >= len(txt):
            break
        obj, pos = dec.raw_decode(txt, pos)
        rows.append(obj)
    strong = sorted([r for r in rows if r["mode"] == "strong"], key=lambda r: r["threads"])
    if strong:
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
        th = [r["threads"] for r in strong]; rs = [r["rays_per_s"] / r["threads"] for r in strong]
        ax.plot(th, rs, "o-", color="C0", label=f"BLP, {strong[0]['host_cpus']}-thread node")
        ax.set_xscale("log", base=2); ax.set_xlabel("threads"); ax.set_ylabel("rays s$^{-1}$ thread$^{-1}$")
        ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(RES, "fig16_scaling_strong.png"), bbox_inches="tight"); plt.close(fig)
    resr = sorted([r for r in rows if r["mode"] == "resolution"], key=lambda r: r["res"])
    if resr:
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=160)
        px = [r["res"] for r in resr]
        ax.loglog(px, [r["seconds_one"] / r["res"] ** 2 * 1e6 for r in resr], "o-", color="C0", label="one image")
        ax.loglog(px, [r["seconds_ten"] / 10 / r["res"] ** 2 * 1e6 for r in resr], "s--", color="C1", label="ten images at once, per image")
        ax.set_xlabel("pixels per side"); ax.set_ylabel(r"$\mu$s per pixel (4 threads)"); ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(RES, "fig17_scaling_resolution.png"), bbox_inches="tight"); plt.close(fig)
print("wrote the available figures to", RES)
