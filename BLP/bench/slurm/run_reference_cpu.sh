#!/bin/bash
# The CPU legs of the comparison with the reference package, on the login shell:
#   BLP/bench/slurm/run_reference_cpu.sh <threads>:<deck>:<leg>[,<leg>...][:ref|:blp] ...
# e.g. 16:reference:amr_384,mc_thin 128:torus:img_384,amr_384. For each item and leg, the
# reference package's leg (BLP/bench/reference.jl in results/blp/gpu/env_old) and then
# BLP's (BLP/bench/bench.jl in results/blp/gpu/env with BLP_COMPARE=1: ThreadedExecutor
# chunk 4096, then 64), one Julia process each with <threads> threads; both environments
# hold the same versions of the shared packages. A process starts only when the 1-minute
# load average is below BLP_LOAD_MAX (16) and no Slurm job of this user named blp-ref-* is
# running; the pending jobs of that name are held while it runs (scontrol uhold, then
# release), so the GPU legs and the CPU legs never share the host. Waits, starts and ends
# go to BLP/bench/results/reference/logs/cpu_runner.log, the output of each process to
# logs/cpu_<deck>_t<threads>_<leg>_<ref|blp>.log, the records to
# raw/{ref,blp}_cpu.jsonl; legs already recorded there are skipped unless BLP_SKIP_DONE=0
# (a new record then supersedes the old one in the merge). A trailing `:ref` or `:blp`
# runs that package only; BLP_CPU_CHUNKS passes through to bench.jl.
cd "$(dirname "$0")/../../.."     # the repository root
R=BLP/bench/results/reference
logs=$R/logs
mkdir -p $logs $R/raw
max=${BLP_LOAD_MAX:-16}
export BLP_REF_GPUMONTY=${BLP_REF_GPUMONTY:-$PWD/../Mahakala/ref-code/gpumonty/table/hotcross.dat}
say() { echo "$(date '+%F %T') $*" | tee -a $logs/cpu_runner.log; }
load() { cut -d' ' -f1 /proc/loadavg; }
running() { squeue -h -u $USER -t R -o %j | grep -c '^blp-ref-'; }
pending() { squeue -h -u $USER -t PD -o '%i %j' | awk '$2 ~ /^blp-ref-/ {print $1}'; }
held=""
release() { [ -n "$held" ] && scontrol release $held 2>/dev/null; held=""; }
trap 'release; say "interrupted"; exit 1' INT TERM
# Wait for a quiet host, then hold the pending GPU jobs.
acquire() {
  local n=0
  while true; do
    l=$(load); r=$(running)
    if awk -v l="$l" -v m="$max" 'BEGIN {exit !(l < m)}' && [ "$r" = 0 ]; then
      held=$(pending | tr '\n' ' ')
      [ -n "$held" ] && scontrol uhold $held 2>/dev/null
      if [ "$(running)" = 0 ]; then
        [ $n -gt 0 ] && say "waited $((n * 30)) s for $1"
        return
      fi
      release
    fi
    [ $((n % 10)) = 0 ] && say "wait for $1: load1=$l running_gpu_jobs=$r others>1core: $(ps -eo user:20,pcpu --no-headers | awk '$2 > 100 {u[$1] += $2} END {for (k in u) printf "%s:%.0f%% ", k, u[k]}')"
    n=$((n + 1)); sleep 30
  done
}
run() {   # run <tag> <command...>
  local tag=$1; shift
  acquire "$tag"
  say "start $tag load1=$(load) held=[$held]"
  "$@" >> $logs/$tag.log 2>&1
  local st=$?
  release
  say "end $tag status=$st load1=$(load)"
}
skip=${BLP_SKIP_DONE:-1}
for item in "$@"; do
  IFS=: read -r t deck legs pkg <<< "$item"
  for leg in ${legs//,/ }; do
    [ "$pkg" = blp ] ||
    run "cpu_${deck}_t${t}_${leg}_ref" env BLP_BACKEND=cpu BLP_DECK=$deck BLP_LEGS=$leg \
        BLP_SKIP_DONE=$skip BLP_OUT=$R/raw/ref_cpu.jsonl \
        julia -t $t --project=results/blp/gpu/env_old BLP/bench/reference.jl
    [ "$pkg" = ref ] ||
    run "cpu_${deck}_t${t}_${leg}_blp" env BLP_BACKEND=cpu BLP_DECK=$deck BLP_COMPARE=1 \
        BLP_LEGS=$leg BLP_SKIP_DONE=$skip BLP_OUT=$R/raw/blp_cpu.jsonl \
        julia -t $t --project=results/blp/gpu/env BLP/bench/bench.jl
  done
done
say "all done: $*"
