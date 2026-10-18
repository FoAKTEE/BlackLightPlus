#!/bin/bash
# One GPU job of the comparison with the reference package:
#   sbatch --gres=gpu:N -c 2N reference.sbatch <N> <deck>=<leg>,<leg>...[/<deck>=<leg>,...]
# For each group (a deck, `reference` or `torus`, and a comma list of legs), the reference
# package's legs (BLP/bench/reference.jl in results/blp/gpu/env_old) and then BLP's legs of
# the same deck (BLP/bench/bench.jl in results/blp/gpu/env, BLP_COMPARE=1), one Julia
# process each with 4 threads on devices 0..N-1, back to back. Records are appended to
# BLP/bench/results/reference/raw/{ref,blp}_gpu.jsonl; configurations already recorded
# there are skipped (BLP_SKIP_DONE=1, unless set to 0 at submission), so a resubmitted job
# continues where a preempted one stopped; BLP_POOLS and BLP_REF_CHUNKS pass through.
# Before each process the log gets the time, `uptime` and the nvidia-smi state. TMPDIR is a
# directory of the job under results/reference/tmp. The log ends with `exit <status>`. run_reference.sh submits these jobs from the
# repository root and resubmits after preemption.
#SBATCH -p preempt
#SBATCH -c 2
#SBATCH -t 01:00:00
cd "$SLURM_SUBMIT_DIR"           # the repository root (run_reference.sh submits from there)
ndev=$1; groups=$2
R=BLP/bench/results/reference
export BLP_BACKEND=gpu BLP_SKIP_DONE=${BLP_SKIP_DONE:-1} BLP_COMPARE=1
export BLP_DEVICES=$(seq -s, 0 $((ndev - 1)))
export BLP_REF_GPUMONTY=${BLP_REF_GPUMONTY:-$PWD/../Mahakala/ref-code/gpumonty/table/hotcross.dat}
# temporaries (ptxas and the like) beside the results, removed at the end of the job
export TMPDIR=$PWD/$R/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
state() {
  echo "# $(date '+%F %T') $*: $(uptime)"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
}
echo "host $(hostname) job $SLURM_JOB_ID ndev $ndev groups $groups commit $(git rev-parse --short HEAD) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(date '+%F %T')"
status=0
IFS=/ read -ra gs <<< "$groups"
for g in "${gs[@]}"; do
  deck=${g%%=*}; legs=${g#*=}
  state "reference $deck $legs"
  BLP_DECK=$deck BLP_LEGS=$legs BLP_OUT=$R/raw/ref_gpu.jsonl \
    julia -t 4 --project=results/blp/gpu/env_old BLP/bench/reference.jl || status=$?
  state "BLP $deck $legs"
  BLP_DECK=$deck BLP_LEGS=$legs BLP_OUT=$R/raw/blp_gpu.jsonl \
    julia -t 4 --project=results/blp/gpu/env BLP/bench/bench.jl || status=$?
done
state end
rm -rf "$TMPDIR"
echo "exit $status"
