"""Plot the R04 geodesic convergence (max position error before closest approach against
the Vern7 reference vs march control C) per leg, from rays_*.json.
usage: python3 r04_plot.py <rays.json> <out.png>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
C = [20, 40, 80, 160]
series = [("blp", "BlackLightPlus", "#2a78d6", "o"), ("fd", "RAPTOR, FD connection (M2a)", "#eb6834", "s"),
          ("an", "RAPTOR, analytic connection (M2b)", "#1baf7a", "^")]
legs = [k for k in ("kerr", "s455", "s122") if k in d]
fig, axs = plt.subplots(1, len(legs), figsize=(4 * len(legs), 3.4), sharey=True)
for ax, leg in zip(axs, legs):
    conv = d[leg]["convergence"]
    for key, name, col, mk in series:
        ax.loglog(C, conv[key]["max_err"], color=col, lw=2, marker=mk, ms=8, label=name)
    e0 = conv["blp"]["max_err"][0]
    ax.loglog(C, [e0 * (c / C[0]) ** -4 for c in C], color="#888888", lw=1, ls="--", label="C$^{-4}$")
    ax.set_xticks(C, [str(c) for c in C]); ax.minorticks_off()
    ax.set_title(leg)
    ax.set_xlabel("march control C")
    ax.grid(True, which="major", color="#dddddd", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("max |x - x_ref| before closest approach [M]")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
