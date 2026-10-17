# BLP benchmarks

`bench.jl` times the BLP products on CPU threads or CUDA devices and writes one JSON
record per leg; `bench.jl merge` joins the records into a dated JSON file and a markdown
table; `plot_performance.py` draws `fig_performance.pdf` from that file. The script header
of `bench.jl` lists the legs, the decks, the protocol and every environment variable.

| leg | product | sizes |
|---|---|---|
| `img` | `render`, `StokesI()`, AthenaK torus on `KerrSchildCartesian(0.9)` | 192², 384², 768² |
| `pol` | `render`, `IQUV()`, same deck | 256², 384² |
| `amr` | `render_amr(plan; block = 16, max_level = 2)`, `IQUV()`, same deck | base 384², 1536² effective |
| `analytic` | `render_analytic(plan, EquatorialDisk(0.94); bands = (0, 1, 2))` | 1024² |
| `mc_thin` | `mc_spectrum(prepare(UniformSphere()), 1e7)` | 10⁷ superphotons |
| `mc_scatter` | `mc_spectrum(…; scattering = true)` on the sphere of Thomson depth 0.2 | 10⁷ primaries |
| `mc_scatter0` | `mc_spectrum(prepare(UniformSphere()), 1e7; scattering = true)` | 10⁷ primaries |

A timed leg is a first call (the JIT compilation, and on a GPU the uploads to a fresh
device pool) followed by three calls whose best wall time is the result. Imaging
throughput is pixels (the ray count for `render_amr`) per wall second; Monte Carlo
throughput is superphotons, primaries plus scattered children, per wall second.

## CPU legs

From the repository root, one process per thread count, each writing its own record file:

```sh
R=BLP/bench/results
BLP_OUT=$R/raw/cpu_t16.jsonl julia -t 16 --project=BLP BLP/bench/bench.jl
BLP_LEGS=img BLP_OUT=$R/raw/cpu_t8.jsonl julia -t 8 --project=BLP BLP/bench/bench.jl
BLP_LEGS=img,analytic BLP_OUT=$R/raw/cpu_t1.jsonl julia -t 1 --project=BLP BLP/bench/bench.jl
BLP_LEGS=img,analytic,mc_thin,mc_scatter,mc_scatter0 BLP_OUT=$R/raw/cpu_t64.jsonl \
    julia -t 64 --project=BLP BLP/bench/bench.jl
BLP_LEGS=img,pol,amr BLP_OUT=$R/raw/cpu_t128.jsonl julia -t 128 --project=BLP BLP/bench/bench.jl
```

The CPU executor is `ThreadedExecutor(chunk = 64)`; `BLP_CPU_CHUNK=4096` gives the chunk
of `backend = :cpu`. `Threads.@threads` hands each thread one contiguous run of chunks,
so at 4096 items per chunk a 192² image has nine chunks and keeps at most nine threads
busy. On a shared host, run the 64- and 128-thread legs when the one-minute load average
is low; every record carries the load before and after its timed calls.

## GPU legs

The devices are reachable only inside Slurm jobs (partition `preempt`, at most one hour,
cancelled without notice when a higher-priority job needs the node). The environment
`results/blp/gpu/env` holds BLP, CUDA and the pinned CUDA runtime.

```sh
BLP/bench/slurm/run_gpu.sh tune:1 bench:1 tune:2 bench:2
```

runs, in turn, the pool sweep on one device (lanes 1–4 × chunk 2¹⁴–2¹⁷ per leg, one
fresh pool per configuration), the timed legs on one device with each leg's best
configuration, and the same on two devices. Each item is one `slurm/gpu.sbatch` job
(`sbatch --gres=gpu:N slurm/gpu.sbatch <tune|bench> <N> [legs]`), resubmitted until its
log in `results/logs/` ends with `exit 0`; records already in `results/raw/gpu.jsonl`
are skipped, so a preempted job loses only the leg in flight. `results/logs/runner.log`
keeps every submission and its final Slurm state. Interactively, inside an allocation:

```sh
BLP_BACKEND=gpu BLP_DEVICES=0 BLP_OUT=BLP/bench/results/raw/gpu.jsonl \
    julia -t 4 --project=results/blp/gpu/env BLP/bench/bench.jl
```

`BLP_LANES` and `BLP_CHUNK` fix the pool configuration instead of the sweep's choice.
Other grids are recorded under another kind, which the timed legs ignore; the chunks
above the sweep, for example:

```sh
BLP_TUNE_KIND=tune_ext BLP_TUNE_CHUNKS=262144,524288,1048576 BLP_TUNE_LANES=1,2 \
BLP_RESES=768 BLP/bench/slurm/run_gpu.sh tune:1:img,amr,analytic,mc_thin,mc_scatter
```

## Table and figure

```sh
julia --project=BLP BLP/bench/bench.jl merge     # results/anta_<date>.json and .md
python3 BLP/bench/plot_performance.py            # BLP/figures/paper/fig_performance.pdf
```

The merge keeps every record and marks as `final` the latest one per leg, resolution,
backend, thread or device count and CPU chunk; the table and the figure use those. It
compares each GPU leg's saved output arrays (`results/out/`) with the CPU output of the
same leg (bitwise, or the flux ratio and the largest differences; for Monte Carlo the
bin counts and the weight sums separately) and a two-device output with the one-device
output. The plotter takes its style from `paper/scripts/figstyle.py` and needs LaTeX and
Ghostscript.
