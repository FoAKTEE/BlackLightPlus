#!/usr/bin/env python3
"""Check the citations of the final documentation.

1. Every commit cited in backticks exists: CHANGELOG.md in this repository,
   ref-code/jordydavelaar-raptor/BLP_CHANGES.md in the RAPTOR fork,
   docs/src/validation.md in either.
2. Every `<sha>^:<path>:<line>` location exists at that parent commit.
3. Every cited test location `test/<file>.jl:<line>` exists and its line holds a
   @test or @testset.
4. Every upstream RAPTOR location in BLP_CHANGES.md (`path:line`, and the bare
   `:line` continuations that follow it) exists at the fork base b91e937.
5. Every cited repository path exists: figures/, docs/, test/, bench/, src/, results/
   and evidence/ (results/blp/raptor/evidence) in this repository; src/, model/, tests/,
   run/ of BLP_CHANGES.md in the fork.
6. docs/make.jl lists every page under docs/src (validation.md included).
Run from the repository root; exit status 0 = pass.
"""
import glob
import itertools
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FORK = os.path.join(ROOT, "ref-code", "jordydavelaar-raptor")
RAPTOR_BASE = "b91e937"
RAPTOR_RES = os.path.join(ROOT, "results", "blp", "raptor")
bad = []
n_ok = {"sha": 0, "parent_loc": 0, "test_loc": 0, "raptor_loc": 0, "path": 0, "nav": 0}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def is_commit(repo, sha):
    r = git(repo, "cat-file", "-t", sha)
    return r.returncode == 0 and r.stdout.strip() == "commit"


def nlines(repo, rev, path):
    r = git(repo, "show", f"{rev}:{path}")
    return None if r.returncode != 0 else r.stdout.count("\n") + 1


def expand(p):
    m = re.search(r"\{([^{}]+)\}", p)
    if not m:
        return [p]
    return list(itertools.chain.from_iterable(
        expand(p[:m.start()] + alt + p[m.end():]) for alt in m.group(1).split(",")))


SHA = re.compile(r"`([0-9a-f]{7,40})\^?`")
PARENT_LOC = re.compile(r"`([0-9a-f]{7,40})\^:([\w./-]+):(\d+)`")
TEST_LOC = re.compile(r"`(test/[\w./-]+\.jl):(\d+)`")
PATH = re.compile(r"`((?:figures|docs|test|bench|src|results|evidence)/[\w./{},*-]+)`")


def check_shas(text, repos, where):
    for sha in sorted(set(SHA.findall(text))):
        if any(is_commit(r, sha) for r in repos):
            n_ok["sha"] += 1
        else:
            bad.append(f"{where}: commit {sha} not found")


def check_paths(text, base, where, skip=()):
    for p in sorted(set(PATH.findall(text))):
        if "*" in p or p in skip:
            continue
        p = re.sub(r":\d+(-\d+)?$", "", p)
        for q in expand(p):
            root = RAPTOR_RES if q.startswith("evidence/") else base
            if os.path.exists(os.path.join(root, q)):
                n_ok["path"] += 1
            else:
                bad.append(f"{where}: path {q} missing")


# CHANGELOG.md
cl = open(os.path.join(ROOT, "CHANGELOG.md")).read()
check_shas(cl, [ROOT], "CHANGELOG.md")
for sha, path, line in PARENT_LOC.findall(cl):
    n = nlines(ROOT, sha + "^", path)
    if n is not None and int(line) <= n:
        n_ok["parent_loc"] += 1
    else:
        bad.append(f"CHANGELOG.md: {sha}^:{path}:{line} missing")
for path, line in TEST_LOC.findall(cl):
    fp = os.path.join(ROOT, path)
    lines = open(fp).read().splitlines() if os.path.exists(fp) else []
    if int(line) <= len(lines) and "@test" in lines[int(line) - 1]:
        n_ok["test_loc"] += 1
    else:
        bad.append(f"CHANGELOG.md: {path}:{line} is not a @test/@testset line")
check_paths(PARENT_LOC.sub("", cl), ROOT, "CHANGELOG.md")

# BLP_CHANGES.md (fork)
bc_path = os.path.join(FORK, "BLP_CHANGES.md")
bc = open(bc_path).read() if os.path.exists(bc_path) else ""
if not bc:
    bad.append("BLP_CHANGES.md missing in the fork")
check_shas(bc, [FORK], "BLP_CHANGES.md")
for row in bc.splitlines():
    if not row.startswith("| D"):
        continue
    loc_cell = row.split("|")[3]
    last = None
    for tok in re.findall(r"`([^`]+)`", loc_cell):
        m = re.fullmatch(r"([\w./-]+\.(?:c|h|sh)):(\d+)(?:-\d+)?", tok)
        c = re.fullmatch(r":(\d+)(?:-\d+)?", tok)
        if m:
            last, line = m.group(1), int(m.group(2))
        elif c and last:
            line = int(c.group(1))
        else:
            continue
        n = nlines(FORK, RAPTOR_BASE, last)
        if n is not None and line <= n:
            n_ok["raptor_loc"] += 1
        else:
            bad.append(f"BLP_CHANGES.md: {last}:{line} missing at {RAPTOR_BASE}")
for p in sorted(set(re.findall(r"`((?:src|model|tests|run)/[\w./{},-]+|setup\.sh)(?::\d+)?`", bc))):
    for q in expand(p):
        if os.path.exists(os.path.join(FORK, q)):
            n_ok["path"] += 1
        else:
            bad.append(f"BLP_CHANGES.md: fork path {q} missing")
for p in sorted(set(re.findall(r"`((?:evidence|results|docs)/[\w./-]+)`", bc))):
    root = RAPTOR_RES if p.startswith("evidence/") else ROOT
    if os.path.exists(os.path.join(root, p)):
        n_ok["path"] += 1
    else:
        bad.append(f"BLP_CHANGES.md: path {p} missing")

# validation.md
va = open(os.path.join(ROOT, "docs", "src", "validation.md")).read()
check_shas(va, [ROOT, FORK], "validation.md")
check_paths(va, ROOT, "validation.md")
for p in re.findall(r"`(tools/[\w./-]+)`", va):
    if os.path.exists(os.path.join(RAPTOR_RES, p)):
        n_ok["path"] += 1
    else:
        bad.append(f"validation.md: results/blp/raptor/{p} missing")

# docs nav
mk = open(os.path.join(ROOT, "docs", "make.jl")).read()
pages = [os.path.relpath(p, os.path.join(ROOT, "docs", "src"))
         for p in glob.glob(os.path.join(ROOT, "docs", "src", "**", "*.md"), recursive=True)]
for p in sorted(pages):
    if f'"{p}"' in mk:
        n_ok["nav"] += 1
    else:
        bad.append(f"docs/make.jl: page {p} not in the navigation")

print("checked:", ", ".join(f"{k} {v}" for k, v in n_ok.items()))
if bad:
    print("FAIL")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("OK")
