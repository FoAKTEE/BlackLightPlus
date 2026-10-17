#!/bin/bash
# Run GPU jobs of BLP/bench/slurm/gpu.sbatch in turn, each until its log ends with `exit 0`:
#   BLP/bench/slurm/run_gpu.sh tune:1 bench:1 tune:2 bench:2 [bench:1:img,pol ...]
# An item is phase:ndev[:legs]. A job that ends without `exit` in its log (preempted,
# cancelled, timed out) is resubmitted, at most 60 times per item; a job whose log ends
# with a nonzero exit is not retried. Submissions, final states and times are appended to
# BLP/bench/results/logs/runner.log.
cd "$(dirname "$0")/../../.."     # the repository root
logs=BLP/bench/results/logs
mkdir -p $logs
say() { echo "$(date '+%F %T') $*" | tee -a $logs/runner.log; }
for item in "$@"; do
  IFS=: read -r phase ndev legs <<< "$item"
  tag="${phase}${ndev}${legs:+_${legs//,/-}}"
  jn="blp-bench-$tag"
  for attempt in $(seq 1 60); do
    while [ -n "$(squeue -h -u $USER -n "$jn" -o %i)" ]; do sleep 30; done
    last=$(ls -t $logs/${tag}_*.log 2>/dev/null | head -1)
    if [ -n "$last" ] && grep -qs '^exit 0' "$last"; then say "$tag done: $last"; break; fi
    if [ -n "$last" ] && grep -qs '^exit [1-9]' "$last"; then say "$tag FAILED: $last"; break; fi
    jid=$(sbatch --parsable --gres=gpu:$ndev -J "$jn" -o "$logs/${tag}_%j.log" \
          BLP/bench/slurm/gpu.sbatch "$phase" "$ndev" "$legs") || { say "$tag: submit failed"; break; }
    say "$tag attempt $attempt job $jid submitted"
    started=""
    while true; do
      st=$(sacct -j "$jid" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
      [ -z "$started" ] && [ "$st" = RUNNING ] && { started=1; say "$tag job $jid running"; }
      case "$st" in COMPLETED|FAILED|TIMEOUT|PREEMPTED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY) break;; esac
      sleep 30
    done
    say "$tag job $jid $st"
  done
done
say "all done: $*"
