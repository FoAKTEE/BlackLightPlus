#!/usr/bin/env python3
"""fig_pipeline.pdf: schematic of the BLP package.

Every box restates docs/ARCHITECTURE.md: the three tiers and the direction of their
dependencies (section 1), the ten modules of BLP/src and the CUDA extension (section 2,
roles from sections 3 and 4), the one kernel through which every device computation runs
(sections 1 and 4.2), the products (section 5.3) and the backend switch (section 5.4).
Modules are numbered in include order; a module imports only modules with a lower number.
Schematic, no data. The style is paper/scripts/figstyle.py (LaTeX text in Latin Modern,
Okabe-Ito colours), imported from there by path.

Writes BLP/figures/paper/fig_pipeline.pdf.
Run from the repository root: python3 BLP/figures/plot_pipeline.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "paper", "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import figstyle as fs  # noqa: E402

OUTDIR = os.path.join(HERE, "paper")
W = fs.WIDTH
LINE = 0.138          # line pitch [in] for 7.5 pt text
TITLE = 0.19          # title-to-first-line pitch [in]
SIZE = 7.5


def tt(s):
    """Typewriter text with the LaTeX specials of Julia names escaped."""
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def mod(k, name):
    return r"\textbf{%d\,%s}" % (k, name)


IND = 0.42            # indent [in] of a continuation line

# (title, role, colour, lines); a line is (indent [in], text)
TIERS = [
    ("Pipeline tier", "composition: configuration, plans, image-plane analysis, products",
     fs.OI["blue"], [
         (0, mod(8, "Plans") + ": " + tt("RunConfig") + r", then "
          + tt("prepare(cfg; spacetime, matter, mode, lanes)") + r" $\to$ " + tt("Plan")),
         (IND, "camera, transfer parameters, march budget and queue derived once;"
          " immutable"),
         (0, mod(9, "Analysis") + ": refinement indicators, lensing bands, interferometry"),
         (0, mod(10, "Products") + ": " + ", ".join(tt(s) for s in (
             "render", "render_amr", "render_analytic", "photon_ring_layers")) + ","),
         (IND, ", ".join(tt(s) for s in (
             "trace", "shadow", "sensitivity", "variants", "mc_spectrum"))),
     ]),
    ("Infrastructure tier",
     "machinery: file formats, fluid containers, executors, device pool, residency, scheduling",
     fs.OI["green"], [
         (0, mod(6, "Data") + ": AthenaK " + tt(".athdf") + " and " + tt(".bin") + " readers, "
          + tt("FluidBlocks") + ", " + tt("UniformGrid") + ","),
         (IND, tt("SnapshotSeries") + " and " + tt("FrameWindow") + " (slow light), "
          + tt(".npy") + " writer, Monte Carlo tables"),
         (0, mod(7, "Execution") + ": " + tt("ThreadedExecutor") + ", " + tt("PoolExecutor")
          + " over a " + tt("DevicePool") + ", residency,"),
         (IND, "ray sources, scheduling, worklist, exact accumulators"),
         (0, tt("ext/BLPCUDAExt") + ": the " + tt("CUDADevice") + " methods of the device"
          " protocol, active after " + tt("using CUDA")),
     ]),
    ("Physics tier",
     "equations: pure, isbits after Adapt, allocation-free on the hot path, no threads",
     fs.OI["verm"], [
         (0, mod(1, "Numerics") + ": constants, units, ray state, stencils, exact sums,"
          " Philox RNG, special functions"),
         (0, mod(2, "Spacetime") + ": Kerr--Schild, user metric, superposed binary, ADM grid;"),
         (IND, "geodesics, march control, adaptive Vern7, camera, exact Kerr"),
         (0, mod(3, "Matter") + r": primitives $\to$ " + tt("decode") + r" $\to$ plasma state;"
          " the provider protocol, analytic sources"),
         (0, mod(4, "Transfer") + ": " + tt("StokesI") + ", " + tt("IQUV")
          + "; synchrotron coefficients, tetrads, segments, the march"),
         (0, mod(5, "MonteCarlo") + ": superphoton generation, opacities, Compton scattering, "
          + tt("track!")),
     ]),
]

KERNEL = ("One kernel", fs.OI["purple"], [
    (0, tt("_foreach!") + " in " + tt("Execution/kernel.jl") + " runs"),
    (0, "every device computation as a per-item"),
    (0, "program " + tt("prog(item, i, outs...)") + ":"),
    (0, "imaging, trace, shadow, exact Kerr and"),
    (0, "Monte Carlo; bitwise per item"),
])
SWITCH = ("Backend switch", "0.30", [
    (0, "keyword " + tt("backend") + r" $>$ " + tt("backend!") + r" $>$ "
     + tt("BLP_BACKEND") + r" $>$ " + tt(":auto")),
    (0, tt(":cpu") + ": " + tt("ThreadedExecutor") + " (Julia threads)"),
    (0, tt(":gpu") + ": " + tt("PoolExecutor") + " over the idle GPUs"),
    (0, "a plan never records a backend: the same"),
    (0, "plan renders on the CPU and on the GPUs"),
])
NOTE = "Modules numbered in include order; each imports only modules with a lower number."

X0, X1 = 0.36, W - 0.04          # tier boxes
XM = 0.5 * (X0 + X1)


def box(ax, x0, y0, x1, y1, edge, face="white", lw=1.0, ls="-"):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0,rounding_size=0.06",
                       fc=face, ec=edge, lw=lw, ls=ls)
    ax.add_patch(p)


def arrow(ax, xy0, xy1, color="0.25", lw=1.0, cs="arc3"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, connectionstyle=cs,
                                 shrinkA=0, shrinkB=0))


def text_block(ax, x, ytop, title, lines, color, size=SIZE, role=None):
    ax.text(x, ytop, title, fontsize=9, color=color, va="top", ha="left")
    if role is not None:
        ax.text(x, ytop - TITLE, r"\textit{%s}" % role, fontsize=7.2, color="0.30",
                va="top", ha="left")
        ytop -= LINE
    for k, (dx, s) in enumerate(lines):
        ax.text(x + dx, ytop - TITLE - k * LINE, s, fontsize=size, va="top", ha="left")


def block_height(lines, role=False):
    return TITLE + (len(lines) + role) * LINE + 0.07


def main():
    fs.setup()
    gap, top_pad, note_gap, bot_pad = 0.24, 0.04, 0.30, 0.04
    heights = [block_height(lines, role=True) for _, _, _, lines in TIERS]
    hb = max(block_height(KERNEL[2]), block_height(SWITCH[2]))
    H = top_pad + sum(heights) + gap * (len(TIERS) - 1) + note_gap + hb + bot_pad
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    y = H - top_pad
    spans = []
    for (title, role, col, lines), h in zip(TIERS, heights):
        box(ax, X0, y - h, X1, y, col, lw=1.1)
        text_block(ax, X0 + 0.07, y - 0.05, title, lines, col, role=role)
        spans.append((y - h, y))
        y -= h + gap

    # pipeline -> infrastructure: uses; infrastructure -> physics: implements and uses
    labels = ("uses", "implements the protocols of, and uses")
    for ((lo, _), (_, hi)), lab in zip(zip(spans[:-1], spans[1:]), labels):
        arrow(ax, (XM, lo), (XM, hi))
        ax.text(XM + 0.07, 0.5 * (lo + hi), lab, fontsize=7.2, color="0.25",
                va="center", ha="left")
    # pipeline -> physics: uses
    xl = 0.20
    top, bot = spans[0][0] + 0.12, spans[2][1] - 0.12
    ax.plot([X0, xl, xl], [top, top, bot], color="0.25", lw=1.0)
    arrow(ax, (xl, bot), (X0, bot))
    ax.text(xl - 0.07, 0.5 * (top + bot), "uses", rotation=90, fontsize=7.2, color="0.25",
            va="center", ha="center")
    ax.text(X0, spans[2][0] - 0.5 * note_gap, NOTE, fontsize=7.2, color="0.30",
            va="center", ha="left")

    # the kernel and the backend switch
    yb = spans[2][0] - note_gap
    xs = X0 + 2.20
    for (title, col, lines), (x0, x1) in zip((KERNEL, SWITCH), ((X0, xs - 0.10), (xs, X1))):
        box(ax, x0, yb - hb, x1, yb, col, face="#f4f4f4", lw=0.9)
        text_block(ax, x0 + 0.07, yb - 0.05, title, lines, col, size=7.2)

    fs.save(fig, "fig_pipeline.pdf", outdir=OUTDIR)


if __name__ == "__main__":
    main()
