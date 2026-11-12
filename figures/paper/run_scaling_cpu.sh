#!/bin/bash
# run_scaling_cpu.sh — driver of fig_scaling_cpu.jl: the Monte Carlo tables once, then the
# thread ladder 1, 2, 4, 8, 16, 32, 64, 72, 128, 144 in rounds, one fresh process per thread
# count and round, each pinned with taskset to its CPU set (Xeon Platinum 8360Y, two
# sockets of 36 cores, NUMA node 0 = CPUs 0-35 and their hyperthreads 72-107, node 1 =
# 36-71 and 108-143):
#   1-32 threads   one per core on socket 1 (CPUs 36, 36-37, 36-39, 36-43, 36-51, 36-67)
#   64             one per core on 64 cores, 8-71 (the cores of both sockets but 0-7)
#   72             one per core on every core, 0-71
#   128            both hyperthreads of the 64 cores of 64 threads, 8-71 and 80-143
#   144            every hardware thread, 0-143
# Rounds alternate the order of the ladder (ascending, descending, ascending, ...). Beside
# every process a sampler (on CPUs 4-7) writes the mean, smallest and largest current
# frequency of its CPUs every two seconds (freq/<label>.txt), and the log names the running
# Slurm jobs. The Slurm jobs of the host's GPUs hold CPUs 0-3 and 72-75: with
# BLP_SCALING_GATE=1 a process whose CPU set holds them (72 and 144 threads) starts only
# when no Slurm job runs on the host (checked every 20 s, the wait logged).
#
# Usage: run_scaling_cpu.sh <env> <outdir> [rounds] [first round]   (rounds default 3)
#        run_scaling_cpu.sh <env> <outdir> tables
# BLP_SCALING_LADDER (default "1 2 4 8 16 32 64 72 128 144") sets the thread counts.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GEN=$HERE/fig_scaling_cpu.jl
ENVP=$1
OUTD=$2
export JULIA_PKG_OFFLINE=true JULIA_MAX_NUM_PRECOMPILE_FILES=100
export BLP_SCALING_OUT=$OUTD/records.jsonl BLP_SCALING_TABLES=$OUTD/tables
LADDER=${BLP_SCALING_LADDER:-"1 2 4 8 16 32 64 72 128 144"}
mkdir -p "$OUTD/logs" "$OUTD/freq"

cpuset() {
  case $1 in
    1) echo 36 ;; 2) echo 36-37 ;; 4) echo 36-39 ;; 8) echo 36-43 ;; 16) echo 36-51 ;;
    32) echo 36-67 ;; 64) echo 8-71 ;; 72) echo 0-71 ;; 128) echo 8-71,80-143 ;; 144) echo 0-143 ;;
  esac
}
legs() {
  local w=wiquv:3,wi:3,mcthin:3,mccompton:3
  case $1 in
    1|2) echo amr:1,i768:1,$w ;; 4) echo amr:1,i768:2,$w ;; 8) echo amr:2,i768:3,$w ;;
    *) echo amr:3,i768:3,$w ;;
  esac
}
expand() {  # 8-71,80-143 -> 8 9 ... 143
  local out=() r a b
  IFS=, read -ra parts <<< "$1"
  for r in "${parts[@]}"; do
    if [[ $r == *-* ]]; then a=${r%-*}; b=${r#*-}; out+=($(seq "$a" "$b")); else out+=("$r"); fi
  done
  echo "${out[@]}"
}
sampler() {  # cpus outfile: every 2 s, epoch mean min max of scaling_cur_freq (kHz), one awk per sample
  local files=() c
  for c in $(expand "$1"); do files+=("/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq"); done
  while :; do
    awk -v t="$(date +%s.%N)" '{s += $1; if (NR == 1 || $1 < mn) mn = $1; if ($1 > mx) mx = $1}
      END {printf "%s %d %d %d\n", t, s / NR, mn, mx}' "${files[@]}" >> "$2"
    sleep 2
  done
}
header() {
  echo "  $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg); julia processes $(pgrep -c -x julia); slurm running: $(squeue -h -t R -w "$(hostname -s)" -o '%i/%u/%C' 2>/dev/null | tr '\n' ' ')"
}

echo "run_scaling_cpu: $(date -Is) generator $(sha256sum "$GEN" | cut -c1-16) env $ENVP out $OUTD"
echo "  tree $(grep -A3 "deps.BlackLightPlus\]" "$ENVP/Manifest.toml" | grep path | cut -d\" -f2) at $(git -C "$(grep -A3 "deps.BlackLightPlus\]" "$ENVP/Manifest.toml" | grep path | cut -d\" -f2)" rev-parse --short HEAD 2>/dev/null)"

if [ "${3:-}" = tables ]; then
  header
  taskset -c "$(cpuset 128)" julia -t 128 --project="$ENVP" "$GEN" tables "${LADDER// /,}" \
      > "$OUTD/logs/tables.log" 2>&1 || echo "  tables failed"
  cat "$OUTD/logs/tables.log"
  exit 0
fi

ROUNDS=${3:-3}
FIRST=${4:-1}
for ((r = FIRST; r < FIRST + ROUNDS; r++)); do
  if ((r % 2 == 1)); then order=$LADDER; else order=$(echo $LADDER | tr ' ' '\n' | sort -rn | tr '\n' ' '); fi
  echo "round $r: $order"
  for t in $order; do
    lab=r${r}_t${t}
    cs=$(cpuset "$t")
    echo "  $lab: cpus $cs, $t threads, legs $(legs "$t")"
    if [ "${BLP_SCALING_GATE:-0}" = 1 ] && { [ "$t" = 72 ] || [ "$t" = 144 ]; }; then
      w=0
      while (($(squeue -h -t R -w "$(hostname -s)" 2>/dev/null | wc -l) > 0)); do sleep 20; w=$((w + 20)); done
      echo "  waited $w s for the host's Slurm jobs to end"
    fi
    header
    sampler "$cs" "$OUTD/freq/$lab.txt" &   # on CPUs 4-7 (outside every set but 72 and 144)
    sp=$!
    taskset -pc 4-7 "$sp" > /dev/null
    BLP_SCALING_ROUND=$r BLP_SCALING_CPUSET=$cs \
      taskset -c "$cs" julia -t "$t" --project="$ENVP" "$GEN" run "$lab" "$(legs "$t")" \
      > "$OUTD/logs/$lab.log" 2>&1 || echo "  $lab failed (exit $?)"
    kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
    grep -E "^(amr|i768|wiquv|wi|mcthin|mccompton) " "$OUTD/logs/$lab.log" | sed 's/^/    /'
  done
done
echo "run_scaling_cpu done $(date -Is)"
