#!/bin/bash
# One GPU job of the KPolaris timing figures: sbatch --gres=gpu:N gpu_timing.sbatch <N> <script>
# runs BLP/figures/kpolaris/<script>.jl on N devices with 4 Julia threads in the GPU
# environment results/blp/gpu/env. Submit from the repository root.
#SBATCH -p preempt
#SBATCH -c 4
#SBATCH -t 01:00:00
cd "$SLURM_SUBMIT_DIR"
ndev=$1; script=$2
export BLP_BACKEND=gpu BLP_NDEV=$ndev
echo "host $(hostname) job $SLURM_JOB_ID ndev $ndev script $script commit $(git rev-parse --short HEAD) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $(date '+%F %T')"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
julia -t 4 --project=results/blp/gpu/env BLP/figures/kpolaris/$script.jl
echo "exit $?"
