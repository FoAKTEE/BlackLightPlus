#!/bin/bash
# r07_build.sh — build the RAPTOR variants of the R07 legs from a fork ref (git archive of
# ref-code/jordydavelaar-raptor, default blp-bbh) with setup.sh -c=athenak -m=sks -r=pol
# -i=rk4, then edit the run-directory definitions.h per leg and rebuild:
#   a      matched (M1-M9, M12): native polarized step (check_stiffness -> RK4/trapezoid)
#   b      a + POL_SOLVER_LD85 (M11): exact constant-coefficient polarized step
#   c      a with the upstream emission fits (EMIS_BLP 0)
#   stock  c + upstream constants (model/bhac/definitions.h), the upstream plasma decode
#          (ATHK_UPSTREAM_PLASMA) and RT_OUTER_CUTOFF 50, on the matched geometry, camera,
#          march and IEEE/IAU readout
# Every variant sets max_steps 4e4 (bbh.in nmax = 50 C up to C = 640). NPIX1D (env, default
# 10) sets RAPTOR's camera block side num_pixels_1d (image sides must be multiples of it;
# R10 uses 8 for 1024^2).
# Run from the BlackLightPlus root. Usage: r07_build.sh <dir> [legs]
# Writes <dir>/src (the exported fork), <dir>/fork_commit and <dir>/<leg>/RAPTOR.
set -euo pipefail
DIR=$(mkdir -p "${1:?directory}" && cd "$1" && pwd)
LEGS=${2:-a,b,c,stock}
REPO=${RAPTOR_REPO:-ref-code/jordydavelaar-raptor}
REF=${REF:-blp-bbh}
rm -rf "$DIR/src"; mkdir -p "$DIR/src"
git -C "$REPO" archive "$REF" | tar -x -C "$DIR/src"
git -C "$REPO" rev-parse "$REF" > "$DIR/fork_commit"
export RAPTOR="$DIR/src"
setdef() { sed -i "/#define $2 (/s/.*/#define $2 ($3)/" "$1"; grep -q "^#define $2 ($3)" "$1"; }
for LEG in ${LEGS//,/ }; do
  B="$DIR/$LEG"
  rm -rf "$B"; mkdir -p "$B"
  ( cd "$B" && bash "$RAPTOR/setup.sh" -c=athenak -m=sks -r=pol -i=rk4 > setup.log 2>&1 )
  D="$B/definitions.h"
  setdef "$D" max_steps 4e4
  if [ -n "${NPIX1D:-}" ]; then
    sed -i -e "s/^#define num_pixels_1d .*/#define num_pixels_1d $NPIX1D/" \
           -e "s/^#define tot_pixels .*/#define tot_pixels $((NPIX1D * NPIX1D))/" "$D"
    grep -q "^#define num_pixels_1d $NPIX1D\$" "$D"
  fi
  case $LEG in
    a) ;;
    b) sed -i 's/^#define EMIS_MODEL (0)$/#define EMIS_MODEL (0)\n#define POL_SOLVER_LD85 (1)/' "$D"
       grep -q '^#define POL_SOLVER_LD85 (1)' "$D" ;;
    c) setdef "$D" EMIS_BLP 0 ;;
    stock)
       setdef "$D" EMIS_BLP 0
       setdef "$D" RT_OUTER_CUTOFF 50.
       for k in ELECTRON_CHARGE ELECTRON_MASS PROTON_MASS BOLTZMANN_CONSTANT PLANCK_CONSTANT GGRAV MSUN KPCTOCM; do
         v=$(sed -n "s/^#define $k (\(.*\))\$/\1/p" "$RAPTOR/model/bhac/definitions.h")
         [ -n "$v" ] || { echo "no upstream $k"; exit 1; }
         setdef "$D" "$k" "$v"
       done
       sed -i 's/^#define ATHK_UPSTREAM_PLASMA (0)$/#define ATHK_UPSTREAM_PLASMA (1)/' "$B/model_definitions.h"
       grep -q '^#define ATHK_UPSTREAM_PLASMA (1)' "$B/model_definitions.h" ;;
    *) echo "unknown leg $LEG"; exit 1 ;;
  esac
  ( cd "$B" && make clean > /dev/null && make all > make.log 2>&1 )
  test -x "$B/RAPTOR"
  echo "built $B/RAPTOR"
done
