#!/bin/bash
# Export a fork branch of RAPTOR (default blp-bbh) as a patch series against the
# pinned upstream commit, and record the series manifest. Run from the
# BlackLightPlus repo root. BRANCH and OUT select another branch and directory
# (BRANCH=blp-reader OUT=results/blp/raptor/patches/blp-reader).
set -euo pipefail
RAPTOR=${RAPTOR:-ref-code/jordydavelaar-raptor}
BASE=b91e937b7b746eb24992b6c1dd8831c5dea0cef1
BRANCH=${BRANCH:-blp-bbh}
OUT=${OUT:-results/blp/raptor/patches}
mkdir -p "$OUT"
rm -f "$OUT"/*.patch
git -C "$RAPTOR" format-patch --no-signature --zero-commit -o "$PWD/$OUT" "$BASE..$BRANCH" >/dev/null
{
  echo "base $BASE"
  echo "head $(git -C "$RAPTOR" rev-parse "$BRANCH")"
  echo "tree $(git -C "$RAPTOR" rev-parse "$BRANCH^{tree}")"
  git -C "$RAPTOR" log --reverse --format='%h %s' "$BASE..$BRANCH"
} > "$OUT/SERIES"
(cd "$OUT" && sha256sum *.patch 2>/dev/null >> SERIES || true)
cat "$OUT/SERIES"
