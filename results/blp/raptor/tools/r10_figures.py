"""r10_figures.py — figures of the R10 comparison (i = 90 AMR-figure deck):
  r10_i90.png     rows: BlackLightPlus uniform, RAPTOR (b) uniform, AMR mosaic block-averaged
                  to the uniform grid, RAPTOR (b) - BlackLightPlus, RAPTOR (b) composite (the
                  finest-pixel zone block-averaged into the uniform image) - AMR;
                  columns: I, |P|, EVPA, V (image axes in M, east left as plot_eht_bbh.py)
  r10_zone.png    the central 20 M zone at the finest AMR pixel: BlackLightPlus uniform,
                  RAPTOR (b), the AMR mosaic crop, and the differences
Usage: python3 r10_figures.py <rundir> <metrics.json> <figdir> [--npix 1024]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r10_compare as r  # noqa: E402

rc = r.rc
plt.rcParams.update({"font.size": 9})


def comps(X):
    I = X["I"]
    P = np.hypot(X["Q"], X["U"])
    chi = np.degrees(0.5 * np.arctan2(X["U"], X["Q"]))
    return I, P, np.ma.array(chi, mask=(I < 0.01 * I.max()) | (P < 0.01 * I)), X["V"]


def panel(fig, axs, rows, diffs, ext, ref, title, out, lim=None):
    IB, PB, CB, VB = comps(ref)
    # the edge-on image peaks in a few pixels: scale I and |P| to their 99.9th percentile
    # over the emitting pixels (brighter pixels saturate)
    vmI = np.percentile(IB[IB > 1e-6 * IB.max()], 99.9)
    vmP = np.percentile(PB[PB > 1e-6 * PB.max()], 99.9)
    vmV = max(np.percentile(np.abs(VB), 99.9), 1e-300)
    for k, (name, X) in enumerate(rows):
        I, P, chi, V = comps(X)
        ims = [axs[k, 0].imshow(I / vmI, origin="lower", extent=ext, cmap="afmhot", norm=PowerNorm(0.5, 0, 1)),
               axs[k, 1].imshow(P / vmP, origin="lower", extent=ext, cmap="afmhot", norm=PowerNorm(0.5, 0, 1)),
               axs[k, 2].imshow(chi, origin="lower", extent=ext, cmap="twilight", vmin=-90, vmax=90),
               axs[k, 3].imshow(V / vmV, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)]
        for c, (im, lab) in enumerate(zip(ims, ["I / I_B,99.9", "|P| / |P|_B,99.9", "EVPA [deg]", "V / V_B,99.9"])):
            fig.colorbar(im, ax=axs[k, c], shrink=0.8, label=lab)
        axs[k, 0].set_ylabel(f"{name}\nβ [M]")
    for j, (name, X, Y) in enumerate(diffs, start=len(rows)):
        I, P, chi, V = comps(X)
        I2, P2, chi2, V2 = comps(Y)
        dI, dP, dV = (I - I2) / vmI, (P - P2) / vmP, (V - V2) / vmV
        dchi = np.ma.array(rc.wrap90(chi.data - chi2.data), mask=chi.mask | chi2.mask)
        s = [max(np.percentile(np.abs(a), 99.9), 1e-12) for a in (dI, dP, dV)]
        sc = max(float(np.abs(dchi).max()), 1e-9)
        ims = [axs[j, 0].imshow(dI, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[0], vmax=s[0]),
               axs[j, 1].imshow(dP, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[1], vmax=s[1]),
               axs[j, 2].imshow(dchi, origin="lower", extent=ext, cmap="RdBu_r", vmin=-sc, vmax=sc),
               axs[j, 3].imshow(dV, origin="lower", extent=ext, cmap="RdBu_r", vmin=-s[2], vmax=s[2])]
        for c, (im, lab) in enumerate(zip(ims, ["ΔI / I_B,99.9", "Δ|P| / |P|_B,99.9", "ΔEVPA [deg]", "ΔV / V_B,99.9"])):
            fig.colorbar(im, ax=axs[j, c], shrink=0.8, label=lab)
        m = r.simple(X, Y)
        axs[j, 0].set_title(f"L1 {m['l1_I']:.1e}", fontsize=8)
        axs[j, 1].set_title(f"L1(|P|) {m['l1_P']:.1e}", fontsize=8)
        axs[j, 3].set_title(f"L1(V) {m['l1_V']:.1e}", fontsize=8)
        axs[j, 0].set_ylabel(f"{name}\nβ [M]")
    lim = lim or ext[0]
    for ax in axs.ravel():
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
    for ax in axs[-1]:
        ax.set_xlabel("α [M] (east left)")
    for c, t in enumerate(["I", "|P|", "EVPA", "V"]):
        axs[0, c].set_title(t)
    fig.suptitle(title)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("metrics")
    ap.add_argument("figdir")
    ap.add_argument("--npix", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    n = a.npix
    met = json.load(open(a.metrics))
    B = r.load_blp_dir(os.path.join(a.run, f"blp_i90_C80_n{n}"))
    rc.CENTERS = B["record"]["centers"]
    R = rc.load_raptor(a.run, "rb", 90, 80, n)
    eff = met["amr"]["record"]["amr"]["eff_res"]
    A = {c: np.load(os.path.join(a.run, "amr", f"amr{eff}_{c}.npy")) for c in "IQUV"}
    Ab = {c: r.block_mean(A[c], eff // n) for c in "IQUV"}
    ext = [20, -20, -20, 20]
    nz = 2 * n
    rc.FOV = 20.0
    Rz = rc.load_raptor(a.run, "rbz", 90, 80, nz)
    rc.FOV = 40.0
    Rc = {c: R[c].copy() for c in "IQUV"}          # composite: the zone block-averaged in
    q = n // 4
    for c in "IQUV":
        Rc[c][q:q + n // 2, q:q + n // 2] = r.block_mean(Rz[c], nz // (n // 2))
    fig, axs = plt.subplots(5, 4, figsize=(12.4, 15.5), constrained_layout=True)
    panel(fig, axs, [("BlackLightPlus uniform", B), ("RAPTOR (b) uniform", R), (f"AMR {eff}² → {n}²", Ab)],
          [("RAPTOR (b) − BLP", R, B), ("RAPTOR (b) composite − AMR", Rc, Ab)], ext, B,
          f"s455 i = 90° AMR-figure deck, 230 GHz, {n}², C = 80: BlackLightPlus {met['blp_git_sha'][:7]}, "
          f"RAPTOR fork blp-bbh, AMR at {met['amr']['record']['git_sha'][:7]}", os.path.join(a.figdir, "r10_i90.png"),
          lim=12)
    wd = os.path.join(a.run, f"blp_i90_C80_n{nz}_fov20")
    if os.path.isdir(wd):
        W = r.load_blp_dir(wd)
        lo = (eff - nz) // 2
        crop = {c: A[c][lo:lo + nz, lo:lo + nz] for c in "IQUV"}
        fig, axs = plt.subplots(5, 4, figsize=(12.4, 15.5), constrained_layout=True)
        panel(fig, axs, [("BlackLightPlus uniform", W), ("RAPTOR (b)", Rz), ("AMR crop", crop)],
              [("RAPTOR (b) − BLP", Rz, W), ("RAPTOR (b) − AMR crop", Rz, crop)], [10, -10, -10, 10], W,
              f"central 20 M zone at the finest AMR pixel (40/{eff} M), {nz}²", os.path.join(a.figdir, "r10_zone.png"), lim=8)

if __name__ == "__main__":
    main()
