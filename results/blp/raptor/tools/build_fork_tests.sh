#!/bin/bash
# Export the fork branch (default blp-bbh) with git archive and build its unit-test
# drivers into <dir>/bin. Run from the BlackLightPlus root. Usage: build_fork_tests.sh <dir>
set -euo pipefail
DIR=${1:?directory}
RAPTOR=${RAPTOR:-ref-code/jordydavelaar-raptor}
REF=${REF:-blp-bbh}
mkdir -p "$DIR/src"
git -C "$RAPTOR" archive "$REF" | tar -x -C "$DIR/src"
git -C "$RAPTOR" rev-parse "$REF" > "$DIR/fork_commit"
bash "$DIR/src/tests/build_tests.sh" "$DIR/bin" > "$DIR/build.log" 2>&1
