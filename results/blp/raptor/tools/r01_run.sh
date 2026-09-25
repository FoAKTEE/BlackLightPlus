#!/bin/bash
# Build RAPTOR at a git ref of ref-code/jordydavelaar-raptor (exported with git archive)
# with the R01 configuration (bhac, CKS, polarized, RK4, gcc/h5cc), run it on the
# synthetic BHAC snapshot and reassemble the HDF5 image. Run from the BlackLightPlus root.
# Usage: r01_run.sh <ref> <rundir> [threads]
set -euo pipefail
REF=${1:?git ref}
RUN=${2:?run directory}
THREADS=${3:-16}
ROOT=$PWD
REPO=${RAPTOR_REPO:-ref-code/jordydavelaar-raptor}
rm -rf "$RUN"
mkdir -p "$RUN/src" "$RUN/run"
git -C "$REPO" archive "$REF" | tar -x -C "$RUN/src"
cd "$RUN/run"
export RAPTOR=$(cd ../src && pwd)
bash "$RAPTOR/setup.sh" -c=bhac -m=cks -r=pol -i=rk4 > setup.log 2>&1
test -x RAPTOR || { tail -20 setup.log; exit 1; }
python3 "$ROOT/results/blp/raptor/tools/make_bhac_synthetic.py" synthetic.dat 0.9375 > synth.log
cp "$ROOT/out/raptor/R01_build/model.in" model.in 2>/dev/null || cat > model.in <<'EOF'
MBH		(Msun)	6.5e9
DISTANCE		(kpc) 16800
M_UNIT		(g)	1e+25
R_LOW		(-)	1
R_HIGH		(-)	1

INCLINATION	(deg)           60
IMG_WIDTH	(pixels)	100
IMG_HEIGHT	(pixels)	100
CAM_SIZE_X	(Rg)		40
CAM_SIZE_Y	(Rg)		40
FREQS_PER_DEC	(-)		1
FREQ_MIN	(Hz)		230.e9
STEPSIZE	(-)		0.01
MAX_LEVEL	(-)		1
EOF
OMP_NUM_THREADS=$THREADS OMP_STACKSIZE=20m ./RAPTOR model.in synthetic.dat 0 > run.log 2> run.err
python3 "$ROOT/results/blp/raptor/tools/raptor_h5_to_array.py" output/img_data_0.h5 40 100 6.5e9 16800 > reassemble.log
cat output/spectrum_0_60.00.dat
cat reassemble.log
