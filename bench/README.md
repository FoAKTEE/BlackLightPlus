# BlackLightPlus benchmarks

`bench.jl` times the BlackLightPlus products on CPU threads or CUDA devices and writes one JSON
record per leg; `bench.jl merge` joins the records into a dated JSON file and a markdown
table; `plot_performance.py` draws `fig_performance.pdf` from that file. `reference.jl`
times the reference package on the same decks and `bench.jl merge-reference` pairs the
two. The script headers of `bench.jl` and `reference.jl` list the legs, the decks, the
protocol and every environment variable; `common.jl` holds the record, host-state and
CUPTI helpers both include. `kernel_audit.jl` holds the registers, stack and spills of
every device kernel against a recorded baseline.

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
R=bench/results
BLP_OUT=$R/raw/cpu_t16.jsonl julia -t 16 --project=. bench/bench.jl
BLP_LEGS=img BLP_OUT=$R/raw/cpu_t8.jsonl julia -t 8 --project=. bench/bench.jl
BLP_LEGS=img,analytic BLP_OUT=$R/raw/cpu_t1.jsonl julia -t 1 --project=. bench/bench.jl
BLP_LEGS=img,analytic,mc_thin,mc_scatter,mc_scatter0 BLP_OUT=$R/raw/cpu_t64.jsonl \
    julia -t 64 --project=. bench/bench.jl
BLP_LEGS=img,pol,amr BLP_OUT=$R/raw/cpu_t128.jsonl julia -t 128 --project=. bench/bench.jl
```

The CPU executor is `ThreadedExecutor()` at its `auto` chunk (`clamp(cld(n, 16 t), 16,
4096)` items on `t` threads); `BLP_CPU_CHUNK=64` or `4096` fixes the chunk instead. One
worker per thread pulls the chunks from a queue, so a chunk count below the thread count
idles threads. On a shared host, run the 64- and 128-thread legs when the one-minute load
average is low; every record carries the load before and after its timed calls.

## GPU legs

The devices are reachable only inside Slurm jobs (partition `preempt`, at most one hour,
cancelled without notice when a higher-priority job needs the node). The environment
`results/blp/gpu/env` on the `blp` branch holds BlackLightPlus, CUDA and the pinned CUDA runtime.

```sh
bench/slurm/run_gpu.sh tune:1 bench:1 tune:2 bench:2
```

runs, in turn, the pool sweep on one device (lanes 1–4 × chunk 2¹⁴–2¹⁷ per leg, one
fresh pool per configuration), the timed legs on one device with each leg's best
configuration, and the same on two devices. Each item is one `slurm/gpu.sbatch` job
(`sbatch --gres=gpu:N slurm/gpu.sbatch <tune|bench> <N> [legs]`), resubmitted until its
log in `results/logs/` ends with `exit 0`; records already in `results/raw/gpu.jsonl`
are skipped, so a preempted job loses only the leg in flight. `results/logs/runner.log`
keeps every submission and its final Slurm state. Interactively, inside an allocation:

```sh
BLP_BACKEND=gpu BLP_DEVICES=0 BLP_OUT=bench/results/raw/gpu.jsonl \
    julia -t 4 --project=results/blp/gpu/env bench/bench.jl
```

`BLP_LANES` and `BLP_CHUNK` fix the pool configuration instead of the sweep's choice.
Other grids are recorded under another kind, which the timed legs ignore; the chunks
above the sweep, for example:

```sh
BLP_TUNE_KIND=tune_ext BLP_TUNE_CHUNKS=262144,524288,1048576 BLP_TUNE_LANES=1,2 \
BLP_RESES=768 bench/slurm/run_gpu.sh tune:1:img,amr,analytic,mc_thin,mc_scatter
```

## Comparison with the reference package

`reference.jl` times the reference package `BlackLightPlus` (the repository's `src/`) on
the legs of `bench/bench_multigpu.jl`, which it includes unchanged, and `bench.jl` with
`BLP_DECK=reference` times the BlackLightPlus calls of the same legs: the fused unpolarized image at
384² and 768² on the uniform resample (40 M half-width, 256³), the same at 768² in mixed
precision, the fused polarized image at 256² and 384², the native-block I and IQUV images
at 384², the adaptive polarized image (base 384, block 8, two levels), the exact-Kerr
g-disk at 1024², and the thin and scattering Monte Carlo spheres at 10⁷ superphotons. The
deck is the harness's: a = 0.9, mass scale 10²⁴ g, fov 40, R_high 40, `use_p = false`,
σ_cut 100, Θe_min 0.3. `BLP_DECK=torus` runs the reference on `bench.jl`'s own deck
(`img_192`, `img_384`, `img_768`, `amr_384`) for the figure. Both scripts write records
of the same format to `results/reference/raw/` and output arrays to
`results/reference/out/<deck>/` (tags `ref-…` and `blp-…`).

The reference runs in its own environment, built once from the repository root:

```sh
E=results/blp/gpu/env_old; mkdir -p $E
cp results/blp/gpu/env/LocalPreferences.toml results/blp/gpu/env/Manifest.toml $E/
cat > $E/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[extras]
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
TOML
julia --project=$E -e 'using Pkg; Pkg.offline(true); Pkg.develop(path = ".");
    Pkg.add(["JacobiElliptic", "ForwardDiff", "Adapt", "LinearAlgebra"]); Pkg.precompile()'
ln -s ../../../../src $E/src
```

Starting from BlackLightPlus's GPU manifest keeps CUDA.jl, GPUCompiler, LLVM.jl, KernelAbstractions,
StaticArrays and the CUDA runtime at the same versions in both environments. The link
`src` exists because `bench_multigpu.jl` loads its device files from the `src/` beside the
active project.

GPU legs, one Slurm job per item, both packages back to back in each job:

```sh
G=reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384
G=$G/reference=amr_384,amr_pol_384/reference=amr_driver_pol
G=$G/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384
bench/slurm/run_reference.sh "1@$G" "2@$G"
```

The binary legs (`bbh455_480`, `bbh455_pol_480`, `bbh122_pol_480`, `bbh_i90_amr_pol`) are
an item of their own:

```sh
bench/slurm/run_reference.sh 1@reference=bbh455_480,bbh455_pol_480,bbh122_pol_480,bbh_i90_amr_pol
```

BlackLightPlus runs every leg at the pool's `auto` chunk (on one device one launch of up to 2²⁰
items, the chunk of the reference's Monte Carlo drivers; at most 16 MiB of outputs per
launch) and at the explicit chunks 65536 and 131072 (`BLP_POOLS`); on the CPU at the
`auto` chunk and at 4096 and 64 (`BLP_CPU_CHUNKS`).

An item `N@<deck>=<legs>/…` is one `slurm/reference.sbatch` job on N devices with 2N CPUs;
for each group the reference process (`reference.jl`, `results/blp/gpu/env_old`) runs and
then the BlackLightPlus process (`bench.jl`, `results/blp/gpu/env`), each with `-t 4`. The reference
runs at its harness default (the device list, whose cached pool has lanes 2 and chunk
65536; the Monte Carlo drivers cut 2²⁰-primary chunks) and on a pool of chunk 2¹⁷ (Monte
Carlo: the drivers' `chunk` keyword at 2¹⁶ and 2¹⁷); BlackLightPlus at its default (lanes 2, chunk
`auto`) and at chunks 2¹⁶ and 2¹⁷. One more call of each default runs under the CUPTI profiler
(`phase` records: host prologue and epilogue, kernel, copy and idle time per device, the
dominant kernel's registers and local memory). The job logs and `runner.log` (every
submission and its final state) are in `results/reference/logs/`; a preempted job is
resubmitted and skips the configurations already recorded (`BLP_SKIP_DONE=0` at submission
measures them again; the merge keeps the latest record of each configuration). The job's
`TMPDIR` is a directory under `results/reference/tmp/`.

CPU legs, on the login shell:

```sh
L=amr_384,amr_pol_384,amr_driver_pol,analytic_1024,mc_thin,mc_scatter
bench/slurm/run_reference_cpu.sh 16:reference:$L 64:reference:$L 128:reference:$L \
    128:torus:img_192,img_384,img_768,amr_384
```

Each leg runs as a reference process and then a BlackLightPlus process (ThreadedExecutor chunk `auto`,
then 4096 and 64), both in the GPU environments above, so the shared packages have the same
versions. A process starts when the one-minute load is below 16 and no `blp-ref-*` job is
running; pending ones are held while it runs. The reference's CPU legs are its CPU entry
points: the two-phase `render` given the loaded snapshot, `render_amr` with its two-phase
closure (which reads the snapshot in each call), `render_analytic`, `mc_spectrum` and
`mc_spectrum_scatter`. `logs/cpu_runner.log` keeps the waits.

## Kernel audit

`kernel_audit.jl` compiles the device twin of every program kind for sm_80 without a GPU
and reports per kernel the registers, the per-thread stack, the spill stores and loads,
and the spills of the non-inlined functions it calls (`ptxas --verbose`). Each case calls a
product on the synthetic decks of `test/decks.jl` through a recording executor, so
the audited programs are the ones the products launch: the imaging program over the
uniform grid, the native blocks, both frame windows and the Polish doughnut on Kerr, the
blocks on the binary and the ADM grid, in `StokesI` and `IQUV`, on the `Float64` and
`Float32` geodesic lanes; its block-grid (`render_amr`) and coordinate-list forms; the
`sensitivity` derivatives and a `variants` ladder; the analytic program and its
photon-ring layer; the trace windows; the shadows; and the Monte Carlo cascades of the
sphere and the torus, their primary and, when they scatter, the layer and the drain of
the lane scratch, placed in a lane's child buffers and scratch as `foreach_cascade!`
places them. Each kernel is compiled for the item list its lane passes: a range as
`UnitRange{Int}` (the analytic pixels, the block chunks, the trace windows, the cascade's
chunks, generations and drain), any other list (the pixel queues, the photon-ring band
points) as a device vector of `Int32` when every item fits, of `Int` otherwise. From the
repository root, in the CUDA environment:

```sh
E=results/blp/gpu/env
julia -t 4 --project=$E bench/kernel_audit.jl            # the table
julia -t 4 --project=$E bench/kernel_audit.jl --check    # against the baseline
julia -t 4 --project=$E bench/kernel_audit.jl --write    # regenerate the baseline
```

`--check` fails (exit 1) when a kernel's registers or stack exceed
`kernel_audit_baseline.json` by more than `--tol-regs` registers or `--tol-stack` bytes
(both 0 by default), when any spill count grows, when a kernel no longer compiles, or
when a baseline kernel is no longer audited; a kernel absent from the baseline is
reported as `NEW`. `--write` records the baseline after a deliberate change of a kernel,
with the toolchain (Julia, ptxas, PTX ISA) and the commit. `--device`, on a GPU, compiles
each kernel for device 0 and checks that its register count equals the offline one.

## Table and figure

```sh
julia --project=. bench/bench.jl merge     # results/anta_<date>.json and .md
julia --project=. bench/bench.jl merge-reference   # results/anta_<date>_reference.json and .md
python3 bench/plot_performance.py            # figures/paper/paper/fig_performance.pdf
```

`merge-reference` pairs each reference record with the BlackLightPlus records of the same deck, leg,
backend and device or thread count: the wall times and rates at both packages' defaults
and best configurations, the ratio BlackLightPlus/reference, the first calls, the loads, the device
phases, the agreement of the two outputs on the same hardware (bitwise, else the flux
ratio and the largest difference over the peak; Monte Carlo counts and sums), and the
Slurm jobs with their preemption history.

The merge keeps every record and marks as `final` the latest one per leg, resolution,
backend, thread or device count and CPU chunk; the table and the figure use those. It
compares each GPU leg's saved output arrays (`results/out/`) with the CPU output of the
same leg (bitwise, or the flux ratio and the largest differences; for Monte Carlo the
bin counts and the weight sums separately) and a two-device output with the one-device
output. The plotter takes its style from `paper/scripts/figstyle.py` and needs LaTeX and
Ghostscript.

The measured records (`bench/results/`) are kept on the `blp` branch; this branch holds the runner and the Slurm scripts.
