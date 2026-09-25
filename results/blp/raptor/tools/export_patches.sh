#!/bin/bash
# Export a fork branch of RAPTOR (default blp-bbh) as a patch series against the
# pinned upstream commit, and record the series manifest. Run from the
# BlackLightPlus repo root. BRANCH and OUT select another branch and directory
# (BRANCH=blp-reader OUT=results/blp/raptor/patches/blp-reader). Commits reachable
# from EXCLUDE (default blp-reader when exporting blp-bbh) are left out, so the blp-bbh
# series holds its own commits and the merged branch's series lives in its own
# directory; `git am -3 --keep-cr` of both series rebuilds the merged tree.
set -euo pipefail
RAPTOR=${RAPTOR:-ref-code/jordydavelaar-raptor}
BASE=b91e937b7b746eb24992b6c1dd8831c5dea0cef1
BRANCH=${BRANCH:-blp-bbh}
OUT=${OUT:-results/blp/raptor/patches}
if [ "$BRANCH" = blp-bbh ]; then EXCLUDE=${EXCLUDE-blp-reader}; else EXCLUDE=${EXCLUDE-}; fi
RANGE=("$BASE..$BRANCH")
[ -n "$EXCLUDE" ] && RANGE+=("^$EXCLUDE")
mkdir -p "$OUT"
rm -f "$OUT"/*.patch
git -C "$RAPTOR" format-patch --no-signature --zero-commit -o "$PWD/$OUT" "${RANGE[@]}" >/dev/null
{
  echo "base $BASE"
  echo "head $(git -C "$RAPTOR" rev-parse "$BRANCH")"
  echo "tree $(git -C "$RAPTOR" rev-parse "$BRANCH^{tree}")"
  [ -n "$EXCLUDE" ] && echo "excluded $EXCLUDE $(git -C "$RAPTOR" rev-parse "$EXCLUDE")"
  git -C "$RAPTOR" log --reverse --no-merges --format='%h %s' "${RANGE[@]}"
} > "$OUT/SERIES"
(cd "$OUT" && sha256sum *.patch 2>/dev/null >> SERIES || true)
cat "$OUT/SERIES"
