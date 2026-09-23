#!/usr/bin/env python3
"""Verify the method-paper manuscript in paper/ (run from the repository root).

Checks, each a hard failure:
  1. paper/main.tex builds with latexmk (pdflatex + bibtex);
  2. main.log reports no undefined references or citations;
  3. the bibliography starts on page <= 6 of main.pdf (or at the top of page 7), so the
     body, acknowledgments and any appendix fit in six pages;
  4. exactly five figure environments in sections/*.tex, each of the five PDFs in
     paper/figures/ used once, and five "Figure N." captions in the PDF;
  5. every number-bearing source line carries a provenance comment: an evidence id
     (% E12) from evidence/claims.md, or % SRC:, % FIG:, % CONV: for code parameters,
     figure setup (paper/figures/CAPTIONS.md), and convention constants; [OPEN] ids
     (E32-E34) appear only in sections/discussion.tex, the limitations section;
  6. the introduction has exactly three paragraphs.
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
PAPER = ROOT / "paper"
CLAIMS = ROOT / "results/blp/paper_arxiv-2609.26298/evidence/claims.md"
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))
fail = []

# 1. build
r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                   cwd=PAPER, capture_output=True, text=True)
if r.returncode != 0 or not (PAPER / "main.pdf").exists():
    print(r.stdout[-3000:], r.stderr[-2000:])
    sys.exit("FAIL: latexmk build failed")

# 2. undefined references / citations
log = (PAPER / "main.log").read_text(errors="replace")
bad = re.findall(r".*(?:undefined references|Citation .* undefined|Reference .* undefined).*", log)
if bad:
    fail.append(f"undefined refs/cites: {bad[:5]}")

# 3. page on which the bibliography starts
def page_text(p):
    return subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout", "main.pdf", "-"],
                          cwd=PAPER, capture_output=True, text=True).stdout
npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", "main.pdf"], cwd=PAPER,
             capture_output=True, text=True).stdout).group(1))
bib_page, before = None, ""
for p in range(1, npages + 1):
    t = page_text(p)
    m = re.search(r"^\s*\[1\]\s", t, re.M)
    if m:
        bib_page, before = p, t[:m.start()]
        break
if bib_page is None:
    fail.append("bibliography not found in PDF")
elif not (bib_page <= 6 or (bib_page == 7 and re.fullmatch(r"[\s\d]*", before))):
    fail.append(f"bibliography starts on page {bib_page}: body exceeds six pages")

# 4. figures
def strip_comments(line):
    return re.sub(r"(?<!\\)%.*", "", line)
src = {f.name: f.read_text() for f in SECTIONS}
body = "\n".join(strip_comments(l) for s in src.values() for l in s.splitlines())
n_env = len(re.findall(r"\\paperfig\b", body)) + len(re.findall(r"\\begin\{figure\*?\}", body))
if n_env != 5:
    fail.append(f"{n_env} figure environments, expected 5")
figs = sorted(p.name for p in (PAPER / "figures").glob("*.pdf"))
for f in figs:
    k = body.count("{" + f + "}")
    if k != 1:
        fail.append(f"figures/{f} used {k} times")
full = "\n".join(page_text(p) for p in range(1, npages + 1))
caps = sorted(set(re.findall(r"Figure (\d+)\.", full)))
if caps != ["1", "2", "3", "4", "5"]:
    fail.append(f"PDF figure captions {caps}, expected 1-5")

# 5. provenance of every number-bearing line
ids = set(re.findall(r"^\| (E\d+) \|", CLAIMS.read_text(), re.M))
open_ids = {i for i in ids if re.search(rf"^\| {i} \| \[OPEN\]", CLAIMS.read_text(), re.M)}
NUM = re.compile(r"\d+\.\d+|\d\s*\\times\s*10|10\^\{?-?\d|\d+\s*\\?%|\d{2,}")
NAMES = re.compile(r"A100|8360Y|M87|Z4c|revtex4-2|apsrev4-2")
TAG = re.compile(r"%.*?(\bE\d+\b|SRC:|FIG:|CONV:)")
MATH_ENV = re.compile(r"\\(begin|end)\{(equation|align|gather)\*?\}")
used = set()
for name, text in src.items():
    in_math = False
    for n, line in enumerate(text.splitlines(), 1):
        if MATH_ENV.search(line):
            in_math = line.lstrip().startswith("\\begin") and "\\end" not in line
            continue
        code = strip_comments(line)
        ref_ids = set(re.findall(r"\bE(\d+)\b", line[len(code):]))
        for rng in re.findall(r"\bE(\d+)--?E(\d+)", line[len(code):]):
            ref_ids |= {str(i) for i in range(int(rng[0]), int(rng[1]) + 1)}
        for i in ref_ids:
            eid = "E" + i
            used.add(eid)
            if eid not in ids:
                fail.append(f"{name}:{n}: unknown evidence id {eid}")
            elif eid in open_ids and name != "discussion.tex":
                fail.append(f"{name}:{n}: [OPEN] claim {eid} used outside the limitations section")
        if in_math or not code.strip():
            continue
        c = re.sub(r"\\(cite|ref|eqref|label|input)\{[^}]*\}", "", code)
        c = re.sub(r"\\paperfig(\[[^]]*\])?\{[^}]*\}", r"\\paperfig", c)
        c = re.sub(r"\\begin\{tabular\}\{[^\n]*?\}(?=\s*$)", "", c)
        c = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^]]*\])?", "", c)
        c = re.sub(r"p\{0\.\d+\\columnwidth\}", "", c)
        c = NAMES.sub("", c)
        c = re.sub(r"_\{\d+\}|_\d", "", c)  # tensor indices such as N_{11}
        if NUM.search(c) and not TAG.search(line):
            fail.append(f"{name}:{n}: number without provenance comment: {code.strip()[:90]}")

# 6. introduction: exactly three paragraphs
intro = re.sub(r"(?m)^[ \t]*%.*\n", "", src["introduction.tex"])
intro = re.sub(r"\\section\{[^}]*\}\s*\\label\{[^}]*\}", "", intro).strip()
paras = [p for p in re.split(r"\n\s*\n", intro) if p.strip()]
if len(paras) != 3:
    fail.append(f"introduction has {len(paras)} paragraphs, expected 3")

if fail:
    print("\n".join("FAIL: " + f for f in fail))
    sys.exit(1)
print(f"OK: builds; bibliography starts on page {bib_page}; 5 figures; "
      f"no undefined refs; intro 3 paragraphs; evidence ids used: "
      f"{', '.join(sorted(used, key=lambda s: int(s[1:])))}")
