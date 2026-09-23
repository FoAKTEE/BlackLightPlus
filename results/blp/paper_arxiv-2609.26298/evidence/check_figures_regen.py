#!/usr/bin/env python3
"""Check that paper figures 2-4 are built from runs after the ray-orientation fix.

1. paper/scripts/check_figures.py passes (five PDFs, benchmark literals).
2. fig_analytic.pdf, fig_polarized.pdf, fig_z4c.pdf and figures/examples/*.png are
   newer than the commit time of 706b55f (camera ray orientation fix).
3. The run metadata of every data set the figures read records a code SHA that
   contains 706b55f and has a clean src/:
   figures/out/examples_meta.txt, figures/out/pol_hires/meta.txt,
   figures/out/z4c_phasespace/deck.json.
4. paper/scripts/plot_analytic.py reads the .npy data (no PNG decoding).
5. No `PENDING sign-fix` marker remains in paper/sections.
Exit status 0 = pass. Run from the repository root.
"""
import glob
import json
import os
import re
import subprocess
import sys

FIX = "706b55f"
ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                      text=True, check=True).stdout.strip()


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)


def main():
    bad = []
    r = subprocess.run([sys.executable, os.path.join(ROOT, "paper", "scripts",
                                                     "check_figures.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        bad.append("paper/scripts/check_figures.py failed")

    t_fix = int(git("show", "-s", "--format=%ct", FIX).stdout.strip())
    outs = [os.path.join(ROOT, "paper", "figures", f"fig_{n}.pdf")
            for n in ("analytic", "polarized", "z4c")]
    outs += sorted(glob.glob(os.path.join(ROOT, "figures", "examples", "*.png")))
    for p in outs:
        ok = os.path.getmtime(p) > t_fix
        print("ok  " if ok else "FAIL", "newer than", FIX, os.path.relpath(p, ROOT))
        if not ok:
            bad.append(p)

    shas = {}
    for name in ("figures/out/examples_meta.txt", "figures/out/pol_hires/meta.txt"):
        m = re.search(r"^git=(\S+)$", open(os.path.join(ROOT, name)).read(), re.M)
        shas[name] = m.group(1) if m else None
    deck = json.load(open(os.path.join(ROOT, "figures/out/z4c_phasespace/deck.json")))
    shas["figures/out/z4c_phasespace/deck.json"] = deck.get("git_sha")
    for name, sha in shas.items():
        ok = (sha is not None and not sha.endswith("-dirty")
              and git("merge-base", "--is-ancestor", FIX, sha).returncode == 0)
        print("ok  " if ok else "FAIL", f"{name} git={sha}")
        if not ok:
            bad.append(name)

    src = open(os.path.join(ROOT, "paper", "scripts", "plot_analytic.py")).read()
    if "analytic_disk_I.npy" not in src or "PIL" in src:
        bad.append("plot_analytic.py does not read analytic_disk_I.npy")
    pend = [p for p in glob.glob(os.path.join(ROOT, "paper", "sections", "*.tex"))
            if "PENDING sign-fix" in open(p).read()]
    if pend:
        bad.append("PENDING sign-fix markers: " + ", ".join(pend))
    if bad:
        print("FAIL:", bad)
        return 1
    print("OK: figures 2-4 built from post-%s runs" % FIX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
