# Benchmarks

Each script prints one machine-parseable line per timed leg (`grep '^BENCH'`).

- **bench_cpu.jl** — CPU legs in one process: GRMHD imaging (`render`, rays/s),
  Monte Carlo scattering (`mc_spectrum_scatter`, superphotons/s), and analytic
  exact-Kerr g-disk imaging (`render_analytic`, rays/s).
- **bench_gpu.jl** — GPU legs on 1–2 CUDA devices: fused GRMHD imaging
  (`render_gpu_fused`) and analytic g-disk (`render_analytic_gpu`, validated
  against the CPU path before timing).
- **kernel_audit.jl** — offline resource audit, no GPU needed: compiles every
  KernelAbstractions kernel in `src/` for sm_80 as a CUDA launch would and prints
  registers, stack and spill bytes from `ptxas --verbose` (`grep '^KRES'`). `--check`
  compares against `kernel_audit_baseline.json` and exits 1 if registers or stack grow
  beyond `--tol-regs` / `--tol-stack` (default 0) or any spill grows; `--write` records a
  new baseline; `--only a,b` restricts the kernel set.

Invocation (from the repo root):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. -t <N> bench/bench_cpu.jl
    CUDA_VISIBLE_DEVICES=0 BLP_NGPU=1 julia --project=. bench/bench_gpu.jl
    julia --project=. bench/kernel_audit.jl --check

Environment knobs:

- `BLP_RESES` (default `192 384 768`) — imaging resolutions; `BLP_AN_RES` (default `1024`) — analytic resolution.
- `BLP_MC_NS` (default `1e6`) — Monte Carlo superphoton count (CPU only).
- `BLP_NGPU` (`1` or `2`) — GPU count (GPU only; set `CUDA_VISIBLE_DEVICES` to match).
- `BLP_TEST_GRMHD` — AthenaK `.athdf` snapshot; `BLP_REF_GPUMONTY` — gpumonty
  `hotcross.dat` (CPU MC leg). A leg is skipped with a note if its file is missing.

Rough runtime at defaults: tens of minutes for bench_cpu.jl (thread-count dependent), a few minutes for bench_gpu.jl.
