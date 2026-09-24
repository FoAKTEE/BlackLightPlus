# Benchmarks

Measured performance of BlackLightPlus. All numbers below were measured on one
machine: 2× NVIDIA A100 80 GB, 2× Intel Xeon Platinum 8360Y (144 cores),
CUDA 13.2, Julia 1.12.6. Timings are warm (JIT excluded), best-of-N, taken with
each leg running sequentially and exclusively. Imaging throughput is rays/s
(= pixels/wall); Monte Carlo throughput counts tracked superphotons/s
(primaries plus all scattered children).

## Headline results

| path | measurement |
|---|---|
| Fused GPU imager (unpolarized, 384²) | 7.3×10⁵ rays/s = **129×** an 8-thread CPU render; flux match 1.04%, pixel correlation 0.999 |
| Fused GPU polarized (IQUV, 256²) | **23.8×** the 16-thread CPU march; on identical trajectories the kernel runs the CPU march (transported readout frame, reverse Mueller chain) |
| GPU native AMR sampling (256²) | 3.5×10⁵ rays/s = **64.4×** the CPU AMR path, 1.03× the uniform-resample GPU kernel; device sampler bit-exact vs CPU (49200/49200 probes) |
| In-kernel autodiff (parameter gradients) | **1.13×** overhead; dF/d(mass scale) and dF/dR_high match finite differences to 7×10⁻¹⁹ and 6×10⁻⁹ |
| Analytic exact-Kerr kernel (1024²) | 9.1×10⁶ rays/s on one A100 (CPU: 6.7×10⁴ at 1 thread, 1.7×10⁶ at 64) |
| Monte Carlo GPU (one A100) | 3.8×10⁷ ph/s thin sphere, 2.7×10⁷ with Compton scattering; CPU reaches 1.4×10⁶ ph/s at 64 threads |

## End-to-end production image

Adaptive polarized image of an AthenaK GRMHD snapshot (a/M = 0.9, 230 GHz,
i = 60°, base 384, two refinement levels → 1536² effective, 543,488 rays):

| configuration | wall time | vs CPU |
|---|--:|--:|
| two-phase, 128-thread CPU march | 31.8 s | 1.0× |
| two-phase, geodesics on 2× A100 | 40.8 s | 0.78× |
| fused single-pass, 1× A100 | 2.78 s | 11.4× |
| fused single-pass, 2× A100 | 1.69 s | 18.8× |

The stored-trajectory two-phase GPU pipeline is PCIe-bound and slower than a
well-threaded CPU; the fused kernel removes that ceiling. Two-GPU scaling on
full frames is 1.74–1.82× (1.64–1.67× end-to-end through the AMR driver).

## CPU scaling (unpolarized imaging, AthenaK torus)

| threads | 1 | 8 | 16 | 
|---|--:|--:|--:|
| 384² rays/s | 965 | 6,254 | 8,470 | 
| 768² rays/s | 993 | 6,263 | 11,405 | 

The two-phase CPU pipeline is memory-bandwidth-bound and saturates near 16–32
threads at high resolution. Single-GPU fused throughput at the same deck:
3.9×10⁵ (192²), 5.9×10⁵ (384²), 6.3×10⁵ (768²) rays/s; two A100s reach
1.0–1.1×10⁶ rays/s at 768² on an uncontended machine.

## Monte Carlo

| configuration | thin sphere | scattering sphere |
|---|--:|--:|
| CPU, 64 threads | — | 1.44×10⁶ ph/s |
| 1× A100 | 3.81×10⁷ | 2.67×10⁷ ph/s |
| 2× A100 | 6.89×10⁷ (1.81×) | 4.63×10⁷ (1.73×) ph/s |

## Accuracy cross-checks

- Fused GPU vs CPU render (uniform-resample fluid): total flux within ~1%
  (inside the cross-code ensemble spread used as the acceptance class), pixel
  correlation > 0.999. With native device AMR sampling, polarized agreement is
  corr(I,Q,U,V) = (1.0, 0.99999, 0.99998, 0.99999).
- GPU vs CPU parity is bit-for-bit wherever the same algorithm runs on both
  sides: the device AMR fluid sampler and the Monte Carlo replay of
  matched RNG streams
  (spectrum ratio 1.000000, worst bin 1.1×10⁻¹⁵).
- The analytic GPU kernel matches the CPU elliptic tracer per pixel to
  round-off (median relative difference ≤ 2.5×10⁻¹⁶).
- Compton scattering unbiasedness (spawned weight / incident scattering loss):
  0.9954 on GPU, 0.9943 on CPU.

## Running the benchmarks

Two self-contained benchmark drivers are retained under `bench/`:

```sh
# CPU legs (imaging, Monte Carlo, analytic) at N threads:
OMP_NUM_THREADS=1 julia --project=. -t <N> bench/bench_cpu.jl

# GPU legs (fused imaging + analytic kernel) on 1 or 2 devices:
CUDA_VISIBLE_DEVICES=0   BLP_NGPU=1 julia --project=. bench/bench_gpu.jl
CUDA_VISIBLE_DEVICES=0,1 BLP_NGPU=2 julia --project=. bench/bench_gpu.jl
```

Both scripts print one machine-parseable `BENCH ...` line per timed leg.
Environment variables: `BLP_TEST_GRMHD` (path to the AthenaK `.athdf` snapshot),
`BLP_RESES` (imaging resolutions, default `192 384 768`), `BLP_MC_NS`
(Monte Carlo superphoton target, default `1e6`), `BLP_AN_RES` (analytic
resolution, default `1024`). GPU timings exclude the one-time fluid
resample/upload, which is amortized across renders. Cross-code comparisons
require externally built reference codes and are not shipped here.
