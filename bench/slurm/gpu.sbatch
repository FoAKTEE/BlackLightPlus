#!/bin/bash
# One GPU job of bench/bench.jl: sbatch [--gres=gpu:N] gpu.sbatch <tune|bench> <N> [legs]
# runs the legs (comma list; default all) on devices 0..N-1 with 4 Julia threads, appending
# to BLP_OUT (default bench/results/raw/gpu.jsonl) and skipping the legs and tuning
# configurations already recorded there, so a resubmitted job continues where a preempted
# one stopped. Other bench.jl variables (BLP_TUNE_FROM, BLP_TUNE_CHUNKS, BLP_RESES, ...)
# pass through from the submitting environment. The log ends with `exit <status>`.
# run_gpu.sh submits these jobs from the repository root and resubmits after preemption.
#SBATCH -p preempt
#SBATCH -c 2
#SBATCH -t 01:00:00
cd "$SLURM_SUBMIT_DIR"           # the repository root (run_gpu.sh submits from there)
phase=$1; ndev=$2; legs=${3:-}
export BLP_BACKEND=gpu BLP_SKIP_DONE=1
export BLP_DEVICES=$(seq -s, 0 $((ndev - 1)))
export BLP_OUT=${BLP_OUT:-bench/results/raw/gpu.jsonl}
[ -n "$legs" ] && export BLP_LEGS=$legs
[ "$phase" = tune ] && export BLP_TUNE=1
echo "host $(hostname) job $SLURM_JOB_ID phase $phase ndev $ndev legs ${legs:-all} commit $(git rev-parse --short HEAD) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
julia -t 4 --project=results/blp/gpu/env bench/bench.jl
echo "exit $?"
