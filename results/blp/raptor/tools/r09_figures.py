"""r09_figures.py — figures of the R09 spinning-snapshot comparison (r07_figures.py panels):
  r09_<tag>_images_i<inc>.png   BlackLightPlus, RAPTOR (a), (a) - BLP, (b) - BLP at step
                                constant C (I, |P|, EVPA, V)
  r09_<tag>_convergence.png     self-convergence and code-to-code L1 against C
Usage: python3 r09_figures.py <figdir> <tag>=<rundir>:<metrics.json> [...] [--C 320]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r07_compare as rc  # noqa: E402
import r07_figures as rf  # noqa: E402

LABEL = {"s122": "s122 (sep 16, χ = 0.8)", "s093": "s093 (sep 25, χ = 0.8)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figdir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--C", type=int, default=320)
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    for spec in a.runs:
        tag, rest = spec.split("=", 1)
        run, mpath = rest.split(":", 1)
        met = json.load(open(mpath))
        rc.CENTERS = None
        for inc in sorted(met["blp"].keys(), key=int):
            rf.img_panels(run, int(inc), a.C, 480, os.path.join(a.figdir, f"r09_{tag}_images_i{int(inc):02d}.png"),
                          met, label=LABEL.get(tag, tag))
        rf.convergence(met, os.path.join(a.figdir, f"r09_{tag}_convergence.png"), label=LABEL.get(tag, tag))


if __name__ == "__main__":
    main()
