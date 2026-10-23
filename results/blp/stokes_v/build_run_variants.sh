#!/bin/bash
# Build Blacklight variants with rotativities zeroed and run them on the comparison deck at 230 GHz and 1 THz.
set -u
T=/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris; D=$T/diag_v; SRC=$T/blacklight/src_copy
for v in norho norhoq norhov; do
  rm -rf $D/bl_$v && cp -r $SRC $D/bl_$v
  f=$D/bl_$v/src/radiation_integrator/polarized.cpp
  case $v in
    norho)  sed -i 's#^\(\s*\)rho_s\[1\] = rho_q\[adaptive_level\](l,m,n);#\1rho_s[1] = 0.0;#; s#^\(\s*\)rho_s\[3\] = rho_v\[adaptive_level\](l,m,n);#\1rho_s[3] = 0.0;#' $f ;;
    norhoq) sed -i 's#^\(\s*\)rho_s\[1\] = rho_q\[adaptive_level\](l,m,n);#\1rho_s[1] = 0.0;#' $f ;;
    norhov) sed -i 's#^\(\s*\)rho_s\[3\] = rho_v\[adaptive_level\](l,m,n);#\1rho_s[3] = 0.0;#' $f ;;
  esac
  grep -n "rho_s\[1\] = \|rho_s\[3\] = " $f
  (cd $D/bl_$v && make -j 16 > build.log 2>&1 && echo "built $v" || { echo "BUILD FAILED $v"; tail -5 build.log; })
done
for v in norho norhoq norhov; do
  for nu in 1000 230; do
    sed "s#output_file   = .*#output_file   = $D/bl_${v}_${nu}.npz#" $T/bl_runs/fast_${nu}.input > $D/run_${v}_${nu}.input
    (cd $D && $D/bl_$v/bin/blacklight $D/run_${v}_${nu}.input > $D/run_${v}_${nu}.log 2>&1 && echo "ran $v $nu" || { echo "RUN FAILED $v $nu"; tail -3 $D/run_${v}_${nu}.log; })
  done
done
echo "variants done"
