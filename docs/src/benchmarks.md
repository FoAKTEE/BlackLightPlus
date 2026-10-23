# Benchmarks

Measured performance of the package on one host, `anta`: 2× Intel Xeon Platinum 8360Y
(72 cores, 144 hardware threads) and 2× NVIDIA A100 80 GB, CUDA 13, Julia 1.12.6. The
protocol is in `bench/README.md`: a first call (the JIT compilation and, on a GPU, the
uploads to a fresh device pool) followed by three calls whose best wall time is the
result; imaging throughput is pixels per wall second (the ray count for `render_amr`),
Monte Carlo throughput superphotons per wall second, primaries plus scattered children.
The GPU legs ran inside a scheduler allocation of four hardware threads; both GPUs are
bitwise equal to one on every leg. The full tables with their conditions are in the
[Verification record](@ref) (§4) and in `bench/results/` on the `blp` branch.

## One deck, three backends

The deck: the AthenaK torus on `KerrSchildCartesian(0.9)`, 230 GHz, inclination 60°,
field of view 30 M, the plasma model of the paper's figures.

| Leg | 128 CPU threads | 1 A100 | 2 A100 |
|---|--:|--:|--:|
| Stokes I 768² (rays/s) | 1.69×10⁵ | 1.24×10⁶ | 2.07×10⁶ |
| IQUV 384² (rays/s) | 5.57×10⁴ | 2.51×10⁵ | 3.58×10⁵ |
| adaptive IQUV production image, 1536² effective (wall) | 11.1 s | 2.65 s | 1.82 s |
| exact-Kerr disk 1024² (rays/s; CPU at 64 threads) | 5.60×10⁶ | 7.78×10⁶ | 6.70×10⁶ |
| Monte Carlo thin sphere, 10⁷ superphotons (ph/s; CPU at 64 threads) | 2.46×10⁶ | 2.76×10⁷ | 3.77×10⁷ |

The executors size their chunks from the item count, the output size and the device and
thread counts, so no leg depends on a chunk setting; on one A100 the automatic chunk
matches or beats the best chunk of a sweep on every leg.

## Against the reference implementation

On identical decks the package and the reference implementation it replaced (the `blp`
branch) are bitwise equal on the same A100; the one-A100 geometric-mean speed ratio of the
package over the reference is 1.13 at the defaults and 1.06 at each code's best setting,
and on the CPU the package is 1.4–2.3× faster. The remaining gaps on the device are kernel
register pressure, recorded by `bench/kernel_audit.jl` against its baseline.

## Running the benchmarks

```sh
R=bench/results
BLP_OUT=$R/raw/cpu_t16.jsonl julia -t 16 --project=. bench/bench.jl       # CPU legs
sbatch bench/slurm/run_gpu.sh                                               # GPU legs
julia --project=. bench/bench.jl merge                                      # dated JSON and table
python3 bench/plot_performance.py                                           # the figure
```

`bench/bench.jl` and `bench/reference.jl` list the legs, the decks, the protocol and
every environment variable in their headers (`BLP_LEGS`, `BLP_OUT`, `BLP_TEST_GRMHD` for
the snapshot path). The benchmark results are not part of this branch.
