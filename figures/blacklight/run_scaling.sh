#!/bin/bash
# Driver of fig_scaling.jl: the strong-scaling ladder over thread counts, then the
# resolution ladder at four threads pinned to four physical cores of socket 0 (CPUs
# 20-23). Run from the repository root.
set -e
rm -f figures/out/blacklight/scaling/results.jsonl
for n in 1 2 4 8 16 32 64; do
    julia -t $n --project=. figures/blacklight/fig_scaling.jl strong
done
taskset -c 20-23 julia -t 4 --project=. figures/blacklight/fig_scaling.jl resolution
