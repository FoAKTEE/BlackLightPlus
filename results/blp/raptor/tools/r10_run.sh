#!/bin/bash
# r10_run.sh — R10 legs (dag.md R10) of the i = 90 AMR-figure deck
# (figures/fig_bbh_i90_pol_amr.jl): the BlackLightPlus AMR render at HEAD, BlackLightPlus
# uniform images (r10_blp.jl) and RAPTOR legs (a) native and (b) LD85 step (r07_raptor.sh
# with the deck's MBH, R_high 10, fov 40, 230 GHz, mass scale 1e24 g, nmax 2500):
#   <npix>^2 at C = 80 (both legs) and C = 160 / nmax 5000 (BlackLightPlus and leg a, for
#   the attribution of leg a); the central 20 M zone at 2 <npix>^2 (the finest AMR pixel for
#   npix = 1024, leg b and BlackLightPlus); BlackLightPlus <npix2>^2 (resolution, report);
#   then r10_compare.py. Stages: all, or "raptor" (RAPTOR legs and the comparison only).
# Codes run one after the other. Run from the BlackLightPlus root.
# Usage: r10_run.sh <build_dir (NPIX1D=8)> <out> [threads] [npix] [npix2] [stages]
set -euo pipefail
BLD=${1:?build dir}
OUT=${2:?out}
N=${3:-64}
NPIX=${4:-1024}
NPIX2=${5:-2048}
STAGES=${6:-all}
NZ=$((2 * NPIX))
T=$(dirname "$0")
mkdir -p "$OUT"
if [ "$STAGES" = all ]; then
  echo "[R10] AMR at HEAD $(date -Is)"
  BLP_OUT=$OUT/amr julia --project=. -t "$N" figures/fig_bbh_i90_pol_amr.jl > "$OUT/amr.log" 2>&1
  grep -E "^AMR|flux_I|levels hist" "$OUT/amr.log"
  echo "[R10] BlackLightPlus uniform $(date -Is)"
  julia --project=. -t "$N" "$T/r10_blp.jl" "$OUT" "$NPIX" 40 | tee "$OUT/blp.log"
  R10_C=160 R10_NMAX=5000 julia --project=. -t "$N" "$T/r10_blp.jl" "$OUT" "$NPIX" 40 | tee -a "$OUT/blp.log"
  julia --project=. -t "$N" "$T/r10_blp.jl" "$OUT" "$NPIX2" 40 | tee -a "$OUT/blp.log"
fi
[ -d "$OUT/blp_i90_C80_n${NZ}_fov20" ] || julia --project=. -t "$N" "$T/r10_blp.jl" "$OUT" "$NZ" 20 | tee -a "$OUT/blp.log"
export R07_MASS_SCALE=1e24 R07_MBH=$(python3 -c "print(repr(6.5e9 * 1.989e33 / 1.98847e33))") \
       R07_RHIGH=10 R07_FREQ=2.3e11 R07_FOV=40 R07_NMAX=2500
echo "[R10] RAPTOR $(date -Is)"
bash "$T/r07_raptor.sh" "$BLD/b/RAPTOR" "$OUT/bbh.in" "$OUT" 90 80 "$NPIX" "$N" rb
bash "$T/r07_raptor.sh" "$BLD/a/RAPTOR" "$OUT/bbh.in" "$OUT" 90 80 "$NPIX" "$N" ra
R07_NMAX=5000 bash "$T/r07_raptor.sh" "$BLD/a/RAPTOR" "$OUT/bbh.in" "$OUT" 90 160 "$NPIX" "$N" ra
R07_FOV=20 bash "$T/r07_raptor.sh" "$BLD/b/RAPTOR" "$OUT/bbh.in" "$OUT" 90 80 "$NZ" "$N" rbz
python3 "$T/r10_compare.py" "$OUT" "$OUT/metrics.json" --npix "$NPIX" --extra-npix "$NPIX2" | tee "$OUT/compare.log"
