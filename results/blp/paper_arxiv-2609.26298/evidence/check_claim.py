#!/usr/bin/env python3
"""Assert that a source file still carries the exact text an evidence claim cites.

Usage: check_claim.py <file> <spec> [<spec> ...]
  file  a working-tree path, or <rev>:<path> (e.g. HEAD:test/x.jl) read with `git show`
  spec  "N=text"  line N (1-based) contains `text` verbatim
        "*=text"  some line of the file contains `text` verbatim
        "!=text"  no line of the file contains `text` (absence of evidence)
Exit 0 iff every spec holds. Run from the repository root.
"""
import subprocess
import sys
from pathlib import Path


def read_source(spec):
    """Working-tree file, or committed content when spec is <rev>:<path>."""
    if ":" in spec and not Path(spec).exists():
        return subprocess.run(["git", "show", spec], check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout
    return Path(spec).read_text(encoding="utf-8")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    path = argv[1]
    lines = read_source(path).splitlines()
    bad = []
    for spec in argv[2:]:
        key, sep, text = spec.partition("=")
        if not sep or not text:
            bad.append(f"malformed spec {spec!r}")
        elif key == "*":
            if not any(text in ln for ln in lines):
                bad.append(f"{path}: text not found: {text!r}")
        elif key == "!":
            hits = [i + 1 for i, ln in enumerate(lines) if text in ln]
            if hits:
                bad.append(f"{path}: text present at lines {hits}: {text!r}")
        else:
            n = int(key)
            got = lines[n - 1] if 0 < n <= len(lines) else ""
            if text not in got:
                bad.append(f"{path}:{n}: expected {text!r}, found {got.strip()!r}")
    for b in bad:
        print("FAIL", b)
    if not bad:
        print(f"OK {path} ({len(argv) - 2} specs)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
