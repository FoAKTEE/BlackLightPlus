#!/bin/bash
# Run GPU jobs of BLP/bench/slurm/reference.sbatch in turn, each until its log ends with
# `exit 0`:
#   BLP/bench/slurm/run_reference.sh <N>@<deck>=<legs>[/<deck>=<legs>...] ...
# e.g. 1@reference=fused_384,fused_768/torus=img_192,img_384 runs those groups on one
# device. An item asks for N devices and 2N CPUs. A job that ends without `exit` in its log
# (preempted, cancelled, timed out) is resubmitted, at most 60 times per item; a job whose
# log ends with a nonzero exit is not retried; a job of the item already in the queue
# (from an earlier runner) is followed instead of submitting another; a refused submission
# is retried every minute. Submissions, starts
# and final Slurm states are appended to BLP/bench/results/reference/logs/runner.log; the
# job logs are logs/<tag>_<job>.log with tag gpu<N>_<first 8 hex digits of the md5 of the
# groups>.
cd "$(dirname "$0")/../../.."     # the repository root
logs=BLP/bench/results/reference/logs
mkdir -p $logs
say() { echo "$(date '+%F %T') $*" | tee -a $logs/runner.log; }
# Slurm state of job $1: from sacct, else from scontrol, else GONE once Slurm no longer
# knows it (sacct answers nothing while the accounting database is unavailable).
job_state() {
  local st
  st=$(sacct -j "$1" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$st" ] && st=$(scontrol show job "$1" 2>/dev/null | grep -o 'JobState=[A-Z_]*' | cut -d= -f2)
  [ -z "$st" ] && st=$(squeue -h -j "$1" -o %T 2>/dev/null | head -1)
  echo "${st:-GONE}"
}
for item in "$@"; do
  ndev=${item%%@*}; groups=${item#*@}
  tag="gpu${ndev}_$(printf %s "$groups" | md5sum | cut -c1-8)"
  jn="blp-ref-$tag"
  say "$tag is $item"
  for attempt in $(seq 1 60); do
    jid=$(squeue -h -u $USER -n "$jn" -o %i | head -1)
    if [ -n "$jid" ]; then            # a job of this item already queued: follow it
      say "$tag attempt $attempt job $jid adopted"
    else
      last=$(ls -t $logs/${tag}_*.log 2>/dev/null | head -1)
      if [ -n "$last" ] && grep -qs '^exit 0' "$last"; then say "$tag done: $last"; break; fi
      if [ -n "$last" ] && grep -qs '^exit [1-9]' "$last"; then say "$tag FAILED: $last"; break; fi
      nfail=0
      until jid=$(sbatch --parsable --gres=gpu:$ndev -c $((2 * ndev)) -J "$jn" \
                  -o "$logs/${tag}_%j.log" BLP/bench/slurm/reference.sbatch "$ndev" "$groups" \
                  2>> $logs/sbatch_errors.log); do
        [ $((nfail % 30)) = 0 ] &&
          say "$tag: submit failed ($(tail -1 $logs/sbatch_errors.log)); retrying every 60 s"
        nfail=$((nfail + 1)); sleep 60
      done
      say "$tag attempt $attempt job $jid submitted"
    fi
    started=""
    while true; do
      st=$(job_state "$jid")
      [ -z "$started" ] && [ "$st" = RUNNING ] && { started=1; say "$tag job $jid running"; }
      case "$st" in COMPLETED|FAILED|TIMEOUT|PREEMPTED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|GONE) break;; esac
      sleep 30
    done
    say "$tag job $jid $st"
  done
done
say "all done: $*"
