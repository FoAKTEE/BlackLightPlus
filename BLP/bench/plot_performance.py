#!/usr/bin/env python3
"""fig_performance.pdf from the BLP benchmark records.

(a) Unpolarized imaging throughput of the AthenaK torus (rays per wall second) against
    image resolution: the CPU at each measured thread count and the GPU at one and two
    devices.
(b) Wall time of the adaptive polarized production image (render_amr, base 384, two
    levels, 1536^2 effective) on the CPU and on the GPUs, with the speed-up over the CPU
    row with the most threads.

Input: the `final` records of BLP/bench/results/anta_<date>.json (the newest such file,
or the path given as the first argument), written by `bench.jl merge`. CPU rows are those
run with the benchmark's CPU chunk (64 items); cases without a record are left out.
Style: paper/scripts/figstyle.py. Output: BLP/figures/paper/fig_performance.pdf.

Run from the repository root: python3 BLP/bench/plot_performance.py [records.json]
"""
import glob
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ["BLP_FIG_DIR"] = os.path.join(REPO, "BLP", "figures", "paper")

_spec = importlib.util.spec_from_file_location(
    "figstyle", os.path.join(REPO, "paper", "scripts", "figstyle.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

import matplotlib.pyplot as plt  # noqa: E402  (figstyle selects the backend)
from matplotlib.ticker import FuncFormatter  # noqa: E402

CPU_CHUNK = 64
CPU_STYLE = {1: (":", "v", "grey"), 8: ("--", "s", "sky"), 16: ("-", "o", "blue"),
             64: ("-.", "^", "green"), 128: ("-", "P", "purple")}


def load(path):
    with open(path) as f:
        doc = json.load(f)
    return [r for r in doc["records"] if r.get("final")]


def cpu_rows(recs, leg):
    return [r for r in recs if r["leg"] == leg and r["backend"] == "cpu"
            and r["chunk"] == CPU_CHUNK]


def gpu_rows(recs, leg):
    return [r for r in recs if r["leg"] == leg and r["backend"] == "gpu"]


def series(rows, by):
    out = {}
    for r in rows:
        out.setdefault(r[by], []).append((r["res"], r["rate"]))
    return {k: sorted(v) for k, v in out.items()}


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "anta_*.json")))
    if not paths:
        sys.exit("no BLP/bench/results/anta_*.json; run `bench.jl merge` first")
    recs = load(paths[-1])

    fs.setup()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.WIDTH, 2.45),
                                 gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.9, wspace=0.08)

    # (a) throughput against resolution, each series labelled at its right end
    res_all = set()
    ends = []
    for th, pts in sorted(series(cpu_rows(recs, "img"), "threads").items()):
        ls, mk, col = CPU_STYLE.get(th, ("-", "x", "black"))
        x, y = zip(*pts)
        res_all.update(x)
        ax.plot(x, y, ls=ls, marker=mk, color=fs.OI[col], ms=3.5, lw=1.1)
        ends.append((x[-1], y[-1], f"{th} thr", fs.OI[col]))
    for nd, pts in sorted(series(gpu_rows(recs, "img"), "ndev").items()):
        x, y = zip(*pts)
        res_all.update(x)
        col = fs.OI["verm" if nd == 1 else "orange"]
        ax.plot(x, y, "D-" if nd == 1 else "*-", color=col, ms=3.5 if nd == 1 else 6, lw=1.1)
        ends.append((x[-1], y[-1], f"{nd} GPU" + ("s" if nd > 1 else ""), col))
    ends.sort(key=lambda e: e[1])
    ylab = []
    for _x, yv, _t, _c in ends:            # keep labels at least a factor 1.45 apart
        ylab.append(max(yv, ylab[-1] * 1.45) if ylab else yv)
    for (xv, _yv, txt, col), yl in zip(ends, ylab):
        ax.text(xv * 1.1, yl, txt, color=col, va="center", ha="left", fontsize=7)
    ticks = sorted(res_all)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([rf"${r}^2$" for r in ticks])
    ax.minorticks_off()
    if ticks:
        ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 2.1)
    ax.set_xlabel("image resolution")
    ax.set_ylabel(r"throughput [rays s$^{-1}$]")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 1.5, hi * 3)
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.03, 0.88, r"$a = 0.9$ torus, Stokes $I$", transform=ax.transAxes, va="top",
            fontsize=7)

    # (b) the production image
    cpu = sorted(cpu_rows(recs, "amr"), key=lambda r: r["threads"])
    gpu = sorted(gpu_rows(recs, "amr"), key=lambda r: r["ndev"])
    ref = cpu[-1] if cpu else None
    bars = [(f"CPU, {r['threads']} threads", r["wall_s"],
             fs.OI[CPU_STYLE.get(r["threads"], ("", "", "blue"))[2]]) for r in cpu]
    bars += [(f"{r['ndev']} GPU" + ("s" if r["ndev"] > 1 else ""), r["wall_s"],
              fs.OI["verm" if r["ndev"] == 1 else "orange"]) for r in gpu]
    y = list(range(len(bars)))[::-1]
    t = [b[1] for b in bars]
    bx.barh(y, t, color=[b[2] for b in bars], height=0.62)
    tmin = min(t) if t else 1.0
    tmax = max(t) if t else 10.0
    x0 = tmin / 2.5
    for yi, (lab, ti, _c) in zip(y, bars):
        sp = rf"  ({ref['wall_s'] / ti:.3g}$\times$)" if ref else ""
        bx.text(ti * 1.12, yi, rf"{ti:.3g} s{sp}", va="center", fontsize=7.5)
        bx.text(x0 * 1.08, yi + 0.40, lab, va="bottom", ha="left", fontsize=7.5)
    bx.set_xscale("log")
    bx.set_xlim(x0, tmax * 12)
    bx.set_ylim(-1.25, len(bars) + 0.1)
    bx.set_yticks([])
    bx.set_xlabel("wall time [s]")
    bx.tick_params(axis="y", which="both", left=False, right=False)
    nrays = (cpu + gpu)[0]["nrays"] if cpu + gpu else 0
    bx.text(0.97, 0.97, "(b)", transform=bx.transAxes, va="top", ha="right", fontsize=9)
    bx.text(0.97, 0.03, rf"adaptive IQUV, $1536^2$ effective, {nrays:,} rays".replace(",", "{,}"),
            transform=bx.transAxes, va="bottom", ha="right", fontsize=7)
    bx.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    fs.save(fig, "fig_performance.pdf")


if __name__ == "__main__":
    main()
