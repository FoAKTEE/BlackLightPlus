#!/bin/bash
# Export the blp-bbh fork of RAPTOR as a patch series against the pinned upstream
# commit, and record the series manifest. Run from the BlackLightPlus repo root.
set -euo pipefail
RAPTOR=${RAPTOR:-ref-code/jordydavelaar-raptor}
BASE=b91e937b7b746eb24992b6c1dd8831c5dea0cef1
OUT=results/blp/raptor/patches
rm -f "$OUT"/*.patch
git -C "$RAPTOR" format-patch --no-signature --zero-commit -o "$PWD/$OUT" "$BASE..blp-bbh" >/dev/null
{
  echo "base $BASE"
  echo "head $(git -C "$RAPTOR" rev-parse blp-bbh)"
  echo "tree $(git -C "$RAPTOR" rev-parse "blp-bbh^{tree}")"
  git -C "$RAPTOR" log --reverse --format='%h %s' "$BASE..blp-bbh"
} > "$OUT/SERIES"
(cd "$OUT" && sha256sum *.patch 2>/dev/null >> SERIES || true)
cat "$OUT/SERIES"
