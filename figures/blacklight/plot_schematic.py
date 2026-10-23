#!/usr/bin/env python3
"""Figure 8 of the Blacklight-paper reproduction: the block-refinement schematic (no data).
A root grid of 8^2 pixels (gray squares) sampled at their centres (black points) forms four
blocks of 4^2 pixels (black squares); the upper-right block is refined into four blocks
(blue squares) of 4^2 pixels (pale blue squares) sampled at their centres (blue points).
Run: python3 figures/blacklight/plot_schematic.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results", "blp", "figures_blacklight")
os.makedirs(RES, exist_ok=True)

fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=200)
n = 8
for i in range(n):
    for j in range(n):
        ax.add_patch(Rectangle((i, j), 1, 1, fc="none", ec="0.75", lw=0.5))
        if not (i >= 4 and j >= 4):
            ax.plot(i + 0.5, j + 0.5, "o", color="k", ms=2.2)
for bi in range(2):
    for bj in range(2):
        ax.add_patch(Rectangle((4 * bi, 4 * bj), 4, 4, fc="none", ec="k", lw=1.2))
h = 0.5
for ci in range(2):
    for cj in range(2):
        x0, y0 = 4 + 2 * ci, 4 + 2 * cj
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((x0 + i * h, y0 + j * h), h, h, fc="none", ec="#9ecae1", lw=0.5))
                ax.plot(x0 + (i + 0.5) * h, y0 + (j + 0.5) * h, "o", color="#2171b5", ms=1.6)
        ax.add_patch(Rectangle((x0, y0), 2, 2, fc="none", ec="#2171b5", lw=1.2))
ax.set_xlim(-0.2, 8.2); ax.set_ylim(-0.2, 8.2)
ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(RES, "fig08_adaptive_grid.png"), bbox_inches="tight")
print("wrote fig08")
