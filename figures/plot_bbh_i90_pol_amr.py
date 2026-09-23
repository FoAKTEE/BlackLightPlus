"""BBH GRMHD superposed-metric full-IQUV image at i = 90°, value-based image-plane AMR
(output of fig_bbh_i90_pol_amr.jl), in the White 2022 adaptive-figure composition:
two rows x four columns with shared per-column colorbars on top.

  row 0 "root level":       the level-0 (base²) batch of the AMR render, blocky
  row 1 "root + N levels":  the AMR mosaic at full effective resolution
  cols: Stokes I (PowerNorm gamma = 0.5, inferno) | LP fraction sqrt(Q²+U²)/I
        (viridis) | V/I (symmetric RdBu_r) | per-pixel refinement level
        (deepest paint wins; the root row is all level 0)

plus one zoom inset on the bottom-row I panel: the far mini-disk ring (brightest
pixel at alpha > +4 M, the primary lensed image of the far hole's mini-disk) at
native finest resolution.

Reads figures/out/bbh_i90_pol_amr/{records.json, amr<EFF>_*.npy, root<BASE>_*.npy};
writes figures/examples/bbh_i90_pol_amr.png.
Usage: python3 figures/plot_bbh_i90_pol_amr.py   [BLP_OUT=<run dir>] [BLP_PNG=<path>]
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, PowerNorm
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLP_OUT", os.path.join(HERE, "out", "bbh_i90_pol_amr"))
PNG = os.environ.get("BLP_PNG", os.path.join(HERE, "examples", "bbh_i90_pol_amr.png"))

LEVEL_COLORS = ["#0d0887", "#7e03a8", "#cc4778", "#f89441", "#f0f921"]  # plasma, discrete


def frac_map(num, I, emit):
    """num/I on emitting pixels, NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(emit, num / np.where(emit, I, 1.0), np.nan)


def main():
    with open(os.path.join(OUT, "records.json")) as f:
        rec = json.load(f)
    amr = rec["amr"]
    base, ml, eff, block = amr["base_res"], amr["max_level"], amr["eff_res"], amr["block"]
    ld = lambda name: np.load(os.path.join(OUT, name))
    I, Q, U, V = (ld(f"amr{eff}_{c}.npy") for c in "IQUV")
    lvf = ld(f"amr{eff}_levels_fine.npy")
    rI, rQ, rU, rV = (ld(f"root{base}_{c}.npy") for c in "IQUV")

    fov = float(rec["fov_M"])
    half = fov / 2
    ext = [-half, half, -half, half]
    leg = rec["amr_leg"]
    ind = rec["indicator"]
    uni = rec["uniform_est"]

    emit = I > 1e-4 * I.max()
    remit = rI > 1e-4 * rI.max()
    lpm = frac_map(np.hypot(Q, U), I, emit)
    rlpm = frac_map(np.hypot(rQ, rU), rI, remit)
    vim = frac_map(V, I, emit)
    rvim = frac_map(rV, rI, remit)

    normI = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.percentile(I, 99.8)))
    normLP = Normalize(0.0, 1.0)
    vmax = float(np.nanpercentile(np.abs(vim), 99.0))
    normVI = Normalize(-vmax, vmax)
    cmapLV = ListedColormap(LEVEL_COLORS[: ml + 1])
    normLV = BoundaryNorm(np.arange(-0.5, ml + 1), cmapLV.N)

    # far mini-disk ring: brightest pixel at alpha > +4 M, 6 M zoom box
    n = I.shape[0]
    axc = np.linspace(-half, half, n, endpoint=False) + fov / n / 2
    jj, ii = np.unravel_index(np.argmax(np.where(axc[None, :] > 4.0, I, 0.0)), I.shape)
    ring_a, ring_b = axc[ii], axc[jj]
    zw = 3.0

    crop = 13.5                     # panels frame the emission; the render box is ±20 M

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17.6, 9.9))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.045, 1, 1],
                          left=0.048, right=0.992, top=0.862, bottom=0.105,
                          wspace=0.07, hspace=0.055)
    axs = np.array([[fig.add_subplot(gs[r + 1, c]) for c in range(4)] for r in range(2)])
    caxs = [fig.add_subplot(gs[0, c]) for c in range(4)]

    rows = [(rI, rlpm, rvim, np.zeros_like(lvf)), (I, lpm, vim, lvf)]
    specs = [("inferno", normI), ("viridis", normLP), ("RdBu_r", normVI), (cmapLV, normLV)]
    ims = [None] * 4
    for r, row in enumerate(rows):
        for c, (data, (cmap, norm)) in enumerate(zip(row, specs)):
            ax = axs[r, c]
            interp = "nearest" if (r == 0 or c == 3) else "antialiased"
            cm = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
            cm.set_bad("#101010")
            ims[c] = ax.imshow(data, origin="lower", extent=ext, cmap=cm, norm=norm,
                               interpolation=interp)
            ax.set_xlim(-crop, crop)
            ax.set_ylim(-crop, crop)
            ax.tick_params(labelsize=8.5)
            if r == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$\alpha\ (GM/c^2)$", fontsize=10.5)
            if c == 0:
                ax.set_ylabel(r"$\beta\ (GM/c^2)$", fontsize=10.5)
            else:
                ax.set_yticklabels([])

    labels = [r"$I$  (code units, $\gamma=0.5$ stretch)",
              r"LP fraction  $\sqrt{Q^2+U^2}/I$",
              r"$V/I$  (99th-percentile $|V/I|$ range)",
              "refinement level"]
    vtk = MaxNLocator(nbins=4, symmetric=True).tick_values(-vmax, vmax)
    tickss = [None, [0.0, 0.25, 0.5, 0.75, 1.0], vtk[np.abs(vtk) <= vmax * 1.0001],
              np.arange(ml + 1)]
    for c, (cax, lbl, tk) in enumerate(zip(caxs, labels, tickss)):
        cb = fig.colorbar(ims[c], cax=cax, orientation="horizontal", ticks=tk)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cb.set_label(lbl, fontsize=10)
        cb.ax.tick_params(labelsize=7.5)

    axs[0, 0].text(0.03, 0.965, "root level", transform=axs[0, 0].transAxes,
                   ha="left", va="top", fontsize=12, color="white")
    axs[1, 0].text(0.03, 0.965, f"root + {ml} levels", transform=axs[1, 0].transAxes,
                   ha="left", va="top", fontsize=12, color="white")

    axi = axs[1, 0].inset_axes([0.02, 0.02, 0.40, 0.40])
    axi.imshow(I, origin="lower", extent=ext, cmap="inferno", norm=normI,
               interpolation="nearest")
    axi.set_xlim(ring_a - zw, ring_a + zw)
    axi.set_ylim(ring_b - zw, ring_b + zw)
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_color("white")
        s.set_linewidth(0.9)
    axs[1, 0].indicate_inset_zoom(axi, edgecolor="white", alpha=0.75, linewidth=0.9)
    axi.text(0.03, 0.965, f"far mini-disk ring\n{2*zw:.0f} M box, native "
             f"{fov/eff:.4f} M px", transform=axi.transAxes, ha="left", va="top",
             fontsize=7.0, color="white",
             bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2))

    thr = rec["threads"]
    plr = leg["per_level_rays"]
    rep = rec["full_repeat"]
    det = rep["bitwise_identical"]
    det_txt = (f"; repeat {rep['wall_s']:.0f} s, bitwise identical" if det else
               "; repeat NOT bitwise identical") if det is not None else ""
    fig.text(0.048, 0.034,
             f"AMR render {leg['wall_s']:.1f} s on {thr} CPU threads ({leg['nrays']} rays, "
             f"{leg['rays_per_s']:.2e} rays/s; per-level {plr}{det_txt})   |   "
             f"snapshot load {rec['snapshot']['load_s']:.1f} s, level-0 probe "
             f"{rec['probe']['wall_s_jit_incl']:.1f} s   |   panels show "
             f"$\\pm${crop:.1f} M of the $\\pm$20 M render box",
             ha="left", va="bottom", fontsize=8.6, color="white", alpha=0.95)
    fig.text(0.048, 0.008,
             f"uniform {eff}$^2$ polarized NOT run ({uni['nrays']/1e6:.1f}M rays) — ESTIMATE "
             f"{uni['est_wall_s']:.0f} s from the measured uniform {base}$^2$ polarized frame "
             f"rate {uni['rate_rays_per_s']:.2e} rays/s $\\Rightarrow$ est saving "
             f"{uni['est_wall_s']/leg['wall_s']:.0f}$\\times$ wall / "
             f"{leg['nrays']/uni['nrays']:.4f}$\\times$ rays",
             ha="left", va="bottom", fontsize=8.6, color="white", alpha=0.95)

    hist = leg["levels_hist"]
    fig.suptitle(
        "BlackLightPlus BBH GRMHD superposed-metric full IQUV, $i=90^\\circ$, fov=40 M — "
        "CPU value-based AMR, backward-traced photons, White 2022 adaptive-figure composition\n"
        "aurora torus.mhd_w_bcc.00455 (t = 453147 M)  |  SuperposedKerrSchild "
        f"(m$_1$=m$_2$=0.5, a=0, sep=25 M)  |  230 GHz, f64, CPU ({thr} threads)  |  "
        f"git {rec['git_sha'][:7]}\n"
        f"base {base} / block {block} / max_level {ml} $\\to$ {eff}$^2$ eff "
        f"(finest px {fov/eff:.4f} M)  |  indicator: abs_grad_cut = "
        "$10^{-3}\\times$max$|\\nabla_{\\rm abs}I|_{\\rm level-0}$"
        f" = {ind['abs_grad_cut']:.2e}, count$\\geq$5 (White 2022)  |  "
        f"levels hist {hist}  |  AMR {leg['wall_s']:.1f} s",
        fontsize=11.5, y=0.985)

    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.savefig(PNG, dpi=150)
    plt.close(fig)
    print("saved", PNG)
    print(f"far ring @ alpha={ring_a:+.2f} beta={ring_b:+.2f} M | V/I range ±{vmax:.2e} | "
          f"LP {leg['LP']:.4f} (root {rec['root_leg']['LP']:.4f})")


if __name__ == "__main__":
    main()
