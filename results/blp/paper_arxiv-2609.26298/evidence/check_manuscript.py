#!/usr/bin/env python3
r"""Verify the method-paper manuscript in paper/ (run from the repository root).

The manuscript uses the AASTeX v7.0.2 class (paper/aastex702.cls) with the AAS bibliography
style (paper/aasjournalv7.1.bst), both bundled in paper/, for The Astrophysical Journal
Supplement Series. Checks, each a hard failure:
  1. paper/main.tex builds with latexmk (pdflatex + bibtex); main.tex loads aastex702 and
     the bundled aasjournalv7.1 style, and the front matter carries \keywords with at most
     six \uat{}{} concepts and a \software statement;
  2. main.log reports no undefined references or citations, and no overfull \hbox or
     \vbox wider than 1 pt;
  3. document order. The layout is two-column: abstract ... conclusions, acknowledgments,
     software, then sections/appendix.tex, which issues \clearpage before \appendix (one
     column), then the bibliography. The main text ends where the "APPENDIX" heading page
     P begins, so it occupies pages 1..P-1; the check requires the \clearpage, the heading,
     and that the "REFERENCES" heading follows the appendix start. The main-text page
     count is reported but not limited: the former 10-page bound was the Physical Review D
     target, and ApJS sets no page limit;
  4. exactly seven figure environments in the main-text section files (every
     sections/*.tex except appendix*.tex), each written out explicitly as
     \begin{figure*}[!t] \centering \includegraphics[width=...]{<pdf>} \caption{...}
     \label{fig:...} \end{figure*}; no \paperfig macro is defined (macros.tex) or used;
     each of the seven PDFs in paper/figures/ used once, and "Figure N." captions 1-7 in
     the PDF; figures in appendix*.tex are counted separately and reported;
  5. every number-bearing source line carries a provenance comment: an evidence id
     (% E12) from evidence/claims.md, or % SRC:, % FIG:, % CONV: for code parameters,
     figure setup (paper/figures/CAPTIONS.md), and convention constants; [OPEN] ids
     (E32-E34) do not appear, since the manuscript has no limitations section;
     language-model version identifiers (e.g. Opus 4.8) are names, not results;
     sections/acknowledgments.tex is author-controlled (funding, allocation and grant
     identifiers, tool versions) and carries no results, so it is exempt from this rule;
  6. the introduction, up to its first \subsection, has exactly three paragraphs.
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

# 1b. AASTeX class, bundled AAS bibliography style, UAT keywords, software statement
main_tex = (PAPER / "main.tex").read_text()
main_code = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in main_tex.splitlines())
if not re.search(r"\\documentclass(\[[^]]*\])?\{aastex702\}", main_code):
    fail.append("main.tex does not load the aastex702 class")
if not re.search(r"\\bibliographystyle\{aasjournalv7\.1\}", main_code):
    fail.append("main.tex does not use the aasjournalv7.1 bibliography style")
for f in ("aastex702.cls", "aasjournalv7.1.bst"):
    if not (PAPER / f).exists():
        fail.append(f"paper/{f} is not bundled")
kw = re.search(r"\\keywords\{(.*)\}", main_code)
n_uat = len(re.findall(r"\\uat\{[^}]+\}\{\d+\}", kw.group(1))) if kw else 0
if not 1 <= n_uat <= 6:
    fail.append(f"\\keywords carries {n_uat} UAT concepts, expected 1-6")
if not re.search(r"\\software\{", main_code):
    fail.append("main.tex has no \\software statement")

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
app_page = next((p for p in texts if re.search(r"^\s*(\d+\s+)?APPENDIX\s*$", texts[p], re.M)), None)
bib_page = next((p for p in texts if re.search(r"^\s*(\d+\s+)?REFERENCES\s*$", texts[p], re.M)), None)
appsrc = (PAPER / "sections" / "appendix.tex").read_text()
if not re.search(r"\\clearpage\s*\n\s*\\appendix", appsrc):
    fail.append("sections/appendix.tex must issue \\clearpage before \\appendix")
if app_page is None:
    fail.append("appendix heading 'APPENDIX' not found in PDF")
if bib_page is None:
    fail.append("bibliography heading 'REFERENCES' not found in PDF")
elif app_page is not None and bib_page < app_page:
    fail.append(f"bibliography (page {bib_page}) precedes the appendix (page {app_page})")
main_pages = None if app_page is None else app_page - 1

# 4. figures: seven explicit figure environments in the main text, appendix figures
#    counted separately
def strip_comments(line):
    return re.sub(r"(?<!\\)%.*", "", line)
src = {f.name: f.read_text() for f in SECTIONS}
def body_of(names):
    return "\n".join(strip_comments(l) for n in names for l in src[n].splitlines())
main_names = [n for n in src if not n.startswith("appendix")]
app_names = [n for n in src if n.startswith("appendix")]
N_FIG = 7
FIG_ENV = re.compile(r"\\begin\{figure\*\}\[!t\]\s*\\centering\s*"
                     r"\\includegraphics\[width=[^]]+\]\{[^}]+\.pdf\}\s*\\caption\{.*?"
                     r"\\label\{fig:[^}]+\}\s*\\end\{figure\*\}", re.S)
def n_fig(text):
    return len(re.findall(r"\\begin\{figure\*?\}", text))
body = body_of(src)
main_body = body_of(main_names)
n_main, n_app = n_fig(main_body), n_fig(body_of(app_names))
if n_main != N_FIG:
    fail.append(f"{n_main} figure environments in the main text, expected {N_FIG}")
n_explicit = len(FIG_ENV.findall(main_body))
if n_explicit != n_main:
    fail.append(f"{n_main - n_explicit} main-text figures are not explicit figure* "
                "environments with centering, includegraphics, caption and label")
macros = "\n".join(strip_comments(l) for l in (PAPER / "macros.tex").read_text().splitlines())
if re.search(r"\\paperfig\b", body + macros):
    fail.append("\\paperfig is still defined or used")
figs = sorted(p.name for p in (PAPER / "figures").glob("*.pdf"))
if len(figs) != N_FIG + n_app:
    fail.append(f"{len(figs)} PDFs in paper/figures, expected {N_FIG + n_app}")
for f in figs:
    k = body.count("{" + f + "}")
    if k != 1:
        fail.append(f"figures/{f} used {k} times")
full = "\n".join(texts[p] for p in range(1, (main_pages or npages) + 1))
caps = sorted(set(re.findall(r"Figure (\d+)\.", full)), key=int)
if caps != [str(i) for i in range(1, N_FIG + 1)]:
    fail.append(f"main-text PDF figure captions {caps}, expected 1-{N_FIG}")

# 5. provenance of every number-bearing line
ids = set(re.findall(r"^\| (E\d+) \|", CLAIMS.read_text(), re.M))
open_ids = {i for i in ids if re.search(rf"^\| {i} \| \[OPEN\]", CLAIMS.read_text(), re.M)}
NUM = re.compile(r"\d+\.\d+|\d\s*\\times\s*10|10\^\{?-?\d|\d+\s*\\?%|\d{2,}")
NAMES = re.compile(r"A100|8360Y|M87|Z4c|aastex702|aasjournalv7|(?:Opus|ChatGPT)\s+\d+(?:\.\d+)*")
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
                fail.append(f"{name}:{n}: [OPEN] claim {eid} used, but the manuscript has no limitations section")
        if in_math or not code.strip():
            continue
        c = re.sub(r"\\(cite[a-z]*|ref|eqref|label|input)\{[^}]*\}", "", code)
        c = re.sub(r"\\includegraphics(\[[^]]*\])?\{[^}]*\}", "", c)
        c = re.sub(r"\\begin\{tabular\}\{[^\n]*?\}(?=\s*$)", "", c)
        c = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^]]*\])?", "", c)
        c = re.sub(r"p\{0\.\d+\\columnwidth\}", "", c)
        c = NAMES.sub("", c)
        c = re.sub(r"_\{\d+\}|_\d", "", c)  # tensor indices such as N_{11}
        if NUM.search(c) and not TAG.search(line):
            fail.append(f"{name}:{n}: number without provenance comment: {code.strip()[:90]}")

# 6. introduction: exactly three paragraphs before its first subsection
intro = re.sub(r"(?m)^[ \t]*%.*\n", "", src["introduction.tex"])
intro = re.split(r"\\subsection\*?\{", intro)[0]
intro = re.sub(r"\\section\{[^}]*\}\s*\\label\{[^}]*\}", "", intro).strip()
paras = [p for p in re.split(r"\n\s*\n", intro) if p.strip()]
if len(paras) != 3:
    fail.append(f"introduction has {len(paras)} paragraphs, expected 3")

if fail:
    print("\n".join("FAIL: " + f for f in fail))
    sys.exit(1)
print(f"OK: builds (AASTeX v7.0.2, {n_uat} UAT keywords); main text {main_pages} pages "
      f"(appendix starts on page {app_page}, references on page {bib_page}, {npages} pages "
      f"total); {n_main} main-text figures, "
      f"{n_app} appendix figures; no undefined refs; no overfull box > 1 pt; "
      f"intro 3 paragraphs; evidence ids used: "
      f"{', '.join(sorted(used, key=lambda s: int(s[1:])))}")
