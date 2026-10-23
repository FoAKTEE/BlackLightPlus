#!/bin/bash
set -u
T=/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris; D=$T/diag_v; SRC=$T/blacklight/src_copy
rm -rf $D/bl_fixm23 && cp -r $SRC $D/bl_fixm23
f=$D/bl_fixm23/src/radiation_integrator/polarized.cpp
python3 - "$f" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old2 = "              mm_2[1][2] = s * lambda_1 * alpha_s[1] + lambda_2 * rho_s[1];"
new2 = "              mm_2[2][3] = s * lambda_1 * alpha_s[1] + lambda_2 * rho_s[1];"
old3 = "              mm_3[1][2] = -(s * lambda_2 * alpha_s[1] - lambda_1 * rho_s[1]);"
new3 = "              mm_3[2][3] = -(s * lambda_2 * alpha_s[1] - lambda_1 * rho_s[1]);"
assert s.count(old2) == 1 and s.count(old3) == 1, (s.count(old2), s.count(old3))
s = s.replace(old2, new2).replace(old3, new3)
open(p, "w").write(s); print("patched M2/M3 (U,V) entries")
PY
grep -n "mm_2\[2\]\[3\] = \|mm_3\[2\]\[3\] = " $f
(cd $D/bl_fixm23 && make -j 16 > build.log 2>&1 && echo "built fixm23" || { echo "BUILD FAILED fixm23"; tail -5 build.log; })
for nu in 1000 230; do
  sed "s#output_file   = .*#output_file   = $D/bl_fixm23_${nu}.npz#" $T/bl_runs/fast_${nu}.input > $D/run_fixm23_${nu}.input
  (cd $D && $D/bl_fixm23/bin/blacklight $D/run_fixm23_${nu}.input > $D/run_fixm23_${nu}.log 2>&1 && echo "ran fixm23 $nu" || { echo "RUN FAILED fixm23 $nu"; tail -3 $D/run_fixm23_${nu}.log; })
  sed "s#output_file   = .*#output_file   = $D/bl_split_${nu}.npz#; s#image_rotation_split  = false#image_rotation_split  = true#" $T/bl_runs/fast_${nu}.input > $D/run_split_${nu}.input
  (cd $D && $SRC/bin/blacklight $D/run_split_${nu}.input > $D/run_split_${nu}.log 2>&1 && echo "ran split $nu" || { echo "RUN FAILED split $nu"; tail -3 $D/run_split_${nu}.log; })
done
echo "fix runs done"
