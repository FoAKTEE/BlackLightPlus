#!/bin/bash
# Driver of fig_scaling.jl: the strong-scaling ladder over thread counts, then the
# resolution ladder at four threads. Run from the repository root.
set -e
rm -f BLP/figures/out/blacklight/scaling/results.jsonl
for n in 1 2 4 8 16 32 64; do
    julia -t $n --project=BLP BLP/figures/blacklight/fig_scaling.jl strong
done
julia -t 4 --project=BLP BLP/figures/blacklight/fig_scaling.jl resolution
