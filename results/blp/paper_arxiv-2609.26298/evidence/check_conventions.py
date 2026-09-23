#!/usr/bin/env python3
"""Verify conventions.md (all rows, non-empty chosen/code cells) and macros_conventions.tex."""
import pathlib, re, shutil, subprocess, sys, tempfile

D = pathlib.Path(__file__).resolve().parent.parent
ROWS = ["Metric signature", "Units", "Geodesic equations", "Affine-parameter normalization",
        "Ray time-orientation", "Camera, screen, pixel axes", "Stokes parameters", "Invariant quantities",
        "Transfer-matrix form", "Faraday rotation", "Emissivity / absorptivity symbols",
        "Electron temperature", "Monte Carlo weights"]
table = [l for l in (D / "conventions.md").read_text().splitlines() if l.startswith("| ")]
fail = []
for key in ROWS:
    hit = [l for l in table if l[2:].startswith(key)]
    if not hit:
        fail.append(f"missing row: {key}"); continue
    cells = [c.strip() for c in hit[0].strip().strip("|").split(" | ")]
    if len(cells) < 4 or not cells[1] or not cells[2] or not re.search(r"src/\S+:\d+", cells[2]):
        fail.append(f"empty chosen/code cell: {key}")
mac = (D / "macros_conventions.tex").read_text()
n = len(re.findall(r"\\newcommand\{\\[A-Za-z]+\}", mac))
if n == 0:
    fail.append("no \\newcommand in macros_conventions.tex")
names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", mac)
if len(names) != len(set(names)):
    fail.append("duplicate macro names")
if shutil.which("pdflatex"):
    with tempfile.TemporaryDirectory() as t:
        tp = pathlib.Path(t)
        (tp / "m.tex").write_text(mac)
        body = " ".join(f"${{\\{x}}}$" if x not in ("tetrad", "jnu", "anu", "rnu", "jinv", "ainv", "rinv")
                        else "" for x in names)
        body += r" $\tetrad{a}{\mu}\jnu{Q}\anu{V}\rnu{V}\jinv{I}\ainv{I}\rinv{Q}$"
        (tp / "t.tex").write_text("\\documentclass[aps,prd]{revtex4-2}\\usepackage{amsmath,amssymb}"
                                  "\\input{m.tex}\\begin{document}" + body + "\\end{document}\n")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                           cwd=t, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            fail.append("revtex compile failed: " + r.stdout[-400:])
print(f"rows={len(ROWS)} newcommands={n} failures={len(fail)}")
for f in fail:
    print("FAIL", f)
sys.exit(1 if fail else 0)
