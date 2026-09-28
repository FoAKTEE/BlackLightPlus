#!/usr/bin/env python3
"""Verify the method-paper manuscript in paper/ (run from the repository root).

Checks, each a hard failure:
  1. paper/main.tex builds with latexmk (pdflatex + bibtex);
  2. main.log reports no undefined references or citations, and no overfull \hbox or
     \vbox wider than 1 pt;
  3. main-text length. The layout is: abstract ... conclusions, acknowledgments, then
     sections/appendix.tex, which issues \clearpage before \appendix, then the
     bibliography. The appendices therefore come before the bibliography, so the main text
     is measured at the appendix start: the first page carrying the heading
     "Appendix A:" is page P, the main text (abstract through conclusions, plus the
     acknowledgments) occupies pages 1..P-1, and P-1 must be <= 10. Appendices and the
     bibliography are not counted. The check also requires the \clearpage, and that the
     bibliography starts after the appendix;
  4. exactly six figure environments in the main-text section files (every sections/*.tex
     except appendix*.tex), each of the six PDFs in paper/figures/ used once, and
     "Figure N." captions 1-6 in the PDF; figures in appendix*.tex are counted separately
     and reported;
  5. every number-bearing source line carries a provenance comment: an evidence id
     (% E12) from evidence/claims.md, or % SRC:, % FIG:, % CONV: for code parameters,
     figure setup (paper/figures/CAPTIONS.md), and convention constants; [OPEN] ids
     (E32-E34) appear only in sections/discussion.tex, the limitations section;
     sections/acknowledgments.tex is author-controlled (funding, allocation and grant
     identifiers, tool versions) and carries no results, so it is exempt from this rule;
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
for kind, pts in re.findall(r"Overfull \\(hbox|vbox) \((\d+(?:\.\d+)?)pt too", log):
    if float(pts) > 1.0:
        fail.append(f"overfull \\{kind} {pts} pt > 1 pt")

# 3. main-text length, measured at the appendix start
def page_text(p):
    return subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), "-layout", "main.pdf", "-"],
                          cwd=PAPER, capture_output=True, text=True).stdout
npages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", "main.pdf"], cwd=PAPER,
             capture_output=True, text=True).stdout).group(1))
texts = {p: page_text(p) for p in range(1, npages + 1)}
app_page = next((p for p in texts if re.search(r"Appendix A:", texts[p])), None)
bib_page = next((p for p in texts if re.search(r"^\s*\[1\]\s", texts[p], re.M)), None)
appsrc = (PAPER / "sections" / "appendix.tex").read_text()
if not re.search(r"\\clearpage\s*\n\s*\\appendix", appsrc):
    fail.append("sections/appendix.tex must issue \\clearpage before \\appendix")
if app_page is None:
    fail.append("appendix heading 'Appendix A:' not found in PDF")
elif app_page - 1 > 10:
    fail.append(f"main text occupies {app_page - 1} pages (> 10)")
if bib_page is None:
    fail.append("bibliography not found in PDF")
elif app_page is not None and bib_page < app_page:
    fail.append(f"bibliography (page {bib_page}) precedes the appendix (page {app_page})")
main_pages = None if app_page is None else app_page - 1

# 4. figures: six in the main text, appendix figures counted separately
def strip_comments(line):
    return re.sub(r"(?<!\\)%.*", "", line)
src = {f.name: f.read_text() for f in SECTIONS}
def body_of(names):
    return "\n".join(strip_comments(l) for n in names for l in src[n].splitlines())
main_names = [n for n in src if not n.startswith("appendix")]
app_names = [n for n in src if n.startswith("appendix")]
def n_fig(text):
    return len(re.findall(r"\\paperfig\b", text)) + len(re.findall(r"\\begin\{figure\*?\}", text))
body = body_of(src)
n_main, n_app = n_fig(body_of(main_names)), n_fig(body_of(app_names))
if n_main != 6:
    fail.append(f"{n_main} figure environments in the main text, expected 6")
figs = sorted(p.name for p in (PAPER / "figures").glob("*.pdf"))
if len(figs) != 6 + n_app:
    fail.append(f"{len(figs)} PDFs in paper/figures, expected {6 + n_app}")
for f in figs:
    k = body.count("{" + f + "}")
    if k != 1:
        fail.append(f"figures/{f} used {k} times")
full = "\n".join(texts[p] for p in range(1, (main_pages or npages) + 1))
caps = sorted(set(re.findall(r"Figure (\d+)\.", full)), key=int)
if caps != [str(i) for i in range(1, 7)]:
    fail.append(f"main-text PDF figure captions {caps}, expected 1-6")

# 5. provenance of every number-bearing line
ids = set(re.findall(r"^\| (E\d+) \|", CLAIMS.read_text(), re.M))
open_ids = {i for i in ids if re.search(rf"^\| {i} \| \[OPEN\]", CLAIMS.read_text(), re.M)}
NUM = re.compile(r"\d+\.\d+|\d\s*\\times\s*10|10\^\{?-?\d|\d+\s*\\?%|\d{2,}")
NAMES = re.compile(r"A100|8360Y|M87|Z4c|revtex4-2|apsrev4-2")
TAG = re.compile(r"%.*?(\bE\d+\b|SRC:|FIG:|CONV:)")
MATH_ENV = re.compile(r"\\(begin|end)\{(equation|align|gather)\*?\}")
used = set()
for name, text in src.items():
    if name == "acknowledgments.tex":
        continue
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
print(f"OK: builds; main text {main_pages} pages (appendix starts on page {app_page}, "
      f"bibliography on page {bib_page}, {npages} pages total); {n_main} main-text figures, "
      f"{n_app} appendix figures; no undefined refs; no overfull box > 1 pt; "
      f"intro 3 paragraphs; evidence ids used: "
      f"{', '.join(sorted(used, key=lambda s: int(s[1:])))}")
