# BLP against the reference package on anta, 2026-10-17

Host: 2× Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz (72 cores, 144 hardware threads), 2× NVIDIA A100 80 GB PCIe; Julia 1.12.6; commits b40fdb1; reference `src/` trees e76ab3c; BLP `BLP/src`+`BLP/ext` trees ced90bb+c8ef624. Both GPU environments (`results/blp/gpu/env_old` for the reference, `results/blp/gpu/env` for BLP) hold the same CUDA.jl, GPUCompiler, LLVM.jl and KernelAbstractions versions and the CUDA runtime 13.3.

Every GPU row ran inside a Slurm job on partition `preempt` with 4 Julia threads; the reference leg and the BLP leg of a row ran in the same job, one process each, back to back, on the same devices. CPU rows ran on the login shell, started only when the 1-minute load was below 16. `wall` is the best of 3 calls after a first call; `first` is that first call (`*`: it compiled the leg's program; `setup`: the reference harness runs the AMR driver once on one device in its setup to count the rays, which then holds the compilation). The uploads of the reference harness's setup are outside `first`; BLP uploads inside its first call. `load` is the 1-minute load average before/after the timed calls. Ratio: BLP rate over reference rate (above 1: BLP faster); `default` pairs the defaults (reference: the harness's call; BLP: PoolExecutor lanes 2, chunk 65536, or ThreadedExecutor chunk 4096), `best` the best configuration of each package. Agreement: the BLP output against the reference output of the same hardware.

The reference's GPU configurations are its harness default (the cached DevicePool of lanes 2, chunk 65536; the Monte Carlo drivers cut their own chunks of 2^20 primaries) and a DevicePool of lanes 2, chunk 131072 (Monte Carlo: the drivers' `chunk` keyword at 65536 and 131072). BLP's: lanes 2 at chunk 65536 (default) and 131072 (tuned), and where recorded 1048576 (the Monte Carlo and binary legs, which then count in BLP's best). On the CPU the reference runs its CPU entry points: the two-phase render (trace, then march along the stored states), `render_amr` with its two-phase closure (which reads the snapshot in each call), `render_analytic`, `mc_spectrum`, `mc_spectrum_scatter`; BLP runs the same products as on the GPU with ThreadedExecutor chunk 4096 (default) and 64 (tuned).


## GPU

| leg | devices | reference wall [s] | reference rate | reference best (config) | BLP wall [s] default | BLP rate default | BLP wall [s] tuned | BLP rate tuned | BLP/ref default | BLP/ref best | agreement | first [s] ref / BLP | load ref / BLP | jobs |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| fused I 384², uniform resample | 1 | 0.102 | 1.45e+06 | 1.57e+06 (chunk 131072) | 0.097 | 1.52e+06 | 0.0841 | 1.75e+06 | 1.05 | 1.12 | bitwise | 13.1* / 12.2* | 6/7 / 7/7 | 2998 |
| fused I 768², uniform resample | 1 | 0.334 | 1.76e+06 | 2.02e+06 (chunk 131072) | 0.329 | 1.79e+06 | 0.291 | 2.03e+06 | 1.02 | 1.01 | bitwise | 0.344 / 0.402 | 7/7 / 7/7 | 2998 |
| fused I 768², mixed precision | 1 | 0.256 | 2.30e+06 | 2.40e+06 (chunk 131072) | 0.308 | 1.92e+06 | 0.272 | 2.17e+06 | 0.83 | 0.91 | bitwise | 2.58* / 1.98* | 7/7 / 7/7 | 2998 |
| fused IQUV 256², uniform resample | 1 | 0.244 | 2.68e+05 | 2.70e+05 (chunk 131072) | 0.219 | 3.00e+05 | 0.214 | 3.06e+05 | 1.12 | 1.13 | bitwise | 16.2* / 16.5* | 7/7 / 7/7 | 2998 |
| fused IQUV 384², uniform resample | 1 | 0.414 | 3.56e+05 | 3.56e+05 (default) | 0.421 | 3.50e+05 | 0.323 | 4.56e+05 | 0.98 | 1.28 | bitwise | 0.441 / 0.545 | 7/7 / 7/7 | 2998 |
| native blocks I 384² | 1 | 0.138 | 1.07e+06 | 1.11e+06 (chunk 131072) | 0.14 | 1.05e+06 | 0.127 | 1.16e+06 | 0.98 | 1.05 | bitwise | 12.9* / 13.3* | 7/7 / 17/14 | 2998 |
| native blocks IQUV 384² | 1 | 0.622 | 2.37e+05 | 2.87e+05 (chunk 131072) | 0.71 | 2.08e+05 | 0.616 | 2.39e+05 | 0.88 | 0.83 | bitwise | 5.62* / 6.05* | 7/7 / 14/13 | 2998 |
| adaptive IQUV, base 384, block 8 | 1 | 5.32 | 2.14e+05 | 2.51e+05 (chunk 131072) | 5.98 | 1.91e+05 | 5.06 | 2.26e+05 | 0.89 | 0.90 | bitwise | setup 32.6, 5.39 / 23.0* | 10/9 / 8/8 | 2998 |
| exact-Kerr g-disk 1024² | 1 | 0.0981 | 1.07e+07 | 1.07e+07 (default) | 0.321 | 3.26e+06 | 0.163 | 6.42e+06 | 0.31 | 0.60 | bitwise | 16.0* / 16.3* | 8/8 / 8/9 | 2998 |
| MC thin sphere, 10⁷ | 1 | 0.41 | 2.45e+07 | 2.45e+07 (default) | 0.686 | 1.46e+07 | 0.482 | 2.08e+07 | 0.60 | 1.13 | bitwise | 12.7* / 16.8* | 11/10 / 9/8 | 2999 |
| MC scattering sphere, 10⁷ | 1 | 0.752 | 1.58e+07 | 1.58e+07 (default) | 1.88 | 6.31e+06 | 1.65 | 7.17e+06 | 0.40 | 0.54 | bitwise | 9.44* / 10.5* | 10/10 / 8/8 | 2999 |
| binary 00455 I 480² | 1 | 1.39 | 1.66e+05 | 1.66e+05 (default) | 2.05 | 1.13e+05 | 1.79 | 1.29e+05 | 0.68 | 0.89 | bitwise | 14.9* / 15.8* | 7/7 / 8/7 | 3000 |
| binary 00455 IQUV 480² | 1 | 3.76 | 6.13e+04 | 7.07e+04 (chunk 131072) | 3.94 | 5.85e+04 | 2.93 | 7.85e+04 | 0.95 | 1.11 | bitwise | 10.3* / 9.87* | 7/7 / 7/7 | 3000 |
| binary 00122 IQUV 480² | 1 | 3.18 | 7.23e+04 | 8.10e+04 (chunk 131072) | 3.63 | 6.35e+04 | 3.08 | 7.48e+04 | 0.88 | 1.02 | bitwise | 19.9* / 20.5* | 14/11 / 9/8 | 3000 |
| binary 00455 i = 90° adaptive IQUV | 1 | 16.0 | 3.59e+04 | 4.11e+04 (chunk 131072) | 17.1 | 3.36e+04 | 15.1 | 3.80e+04 | 0.94 | 1.01 | bitwise | setup 59.9, 16.1 / 35.5* | 7/7 / 7/7 | 3000 |
| torus: blocks I 192² | 1 | 0.0568 | 6.49e+05 | 6.49e+05 (default) | 0.0559 | 6.59e+05 | 0.056 | 6.58e+05 | 1.02 | 1.02 | bitwise | 13.3* / 14.0* | 10/10 / 8/8 | 2998 |
| torus: blocks I 384² | 1 | 0.148 | 9.96e+05 | 1.03e+06 (chunk 131072) | 0.151 | 9.75e+05 | 0.137 | 1.08e+06 | 0.98 | 1.04 | bitwise | 0.178 / 0.187 | 9/9 / 9/9 | 2998 |
| torus: blocks I 768² | 1 | 0.553 | 1.07e+06 | 1.20e+06 (chunk 131072) | 0.569 | 1.04e+06 | 0.494 | 1.19e+06 | 0.97 | 0.99 | bitwise | 0.587 / 0.607 | 9/9 / 9/8 | 2998 |
| torus: adaptive IQUV, base 384, block 16 | 1 | 3.29 | 1.91e+05 | 2.33e+05 (chunk 131072) | 3.42 | 1.84e+05 | 2.99 | 2.11e+05 | 0.96 | 0.91 | bitwise | 19.3* / 20.8* | 8/8 / 7/7 | 2998 |

## CPU

| leg | threads | reference wall [s] | reference rate | reference best (config) | BLP wall [s] default | BLP rate default | BLP wall [s] tuned | BLP rate tuned | BLP/ref default | BLP/ref best | agreement | first [s] ref / BLP | load ref / BLP | jobs |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| native blocks I 384² | 16 | 8.68 | 1.70e+04 | 1.70e+04 (threads) | 7.15 | 2.06e+04 | 5.46 | 2.70e+04 | 1.21 | 1.59 | flux 1.000000000, max ΔI/peak 1.0e-13 | 15.4* / 10.7* | 6/9 / 8/9 | none |
| native blocks IQUV 384² | 16 | 17.0 | 8677 | 8677 (threads) | 16.9 | 8735 | 12.9 | 1.14e+04 | 1.01 | 1.32 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 31.9* / 20.7* | 13/15 / 12/13 | none |
| adaptive IQUV, base 384, block 8 | 16 | 123 | 9276 | 9276 (threads) | 77.1 | 1.48e+04 | 74.7 | 1.53e+04 | 1.60 | 1.65 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 142* / 80.1* | 14/16 / 13/19 | none |
| exact-Kerr g-disk 1024² | 16 | 1.28 | 8.18e+05 | 8.18e+05 (threads) | 0.968 | 1.08e+06 | 0.797 | 1.32e+06 | 1.33 | 1.61 | bitwise | 2.5* / 1.96* | 11/12 / 10/11 | none |
| MC thin sphere, 10⁷ | 16 | 14.3 | 7.00e+05 | 7.00e+05 (threads) | 14.7 | 6.83e+05 | 14.5 | 6.92e+05 | 0.98 | 0.99 | bitwise | 16.0* / 18.8* | 10/15 / 14/17 | none |
| MC scattering sphere, 10⁷ | 16 | 15.1 | 7.85e+05 | 7.85e+05 (threads) | 15.5 | 7.65e+05 | 15.4 | 7.69e+05 | 0.97 | 0.98 | bitwise | 17.7* / 20.3* | 10/15 / 13/16 | none |
| native blocks I 384² | 64 | 5.0 | 2.95e+04 | 2.95e+04 (threads) | 2.52 | 5.86e+04 | 1.56 | 9.47e+04 | 1.98 | 3.21 | flux 1.000000000, max ΔI/peak 1.0e-13 | 24.3* / 6.36* | 9/25 / 13/16 | none |
| native blocks IQUV 384² | 64 | 8.17 | 1.80e+04 | 1.80e+04 (threads) | 6.15 | 2.40e+04 | 3.65 | 4.04e+04 | 1.33 | 2.24 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 20.7* / 9.86* | 10/38 / 13/22 | none |
| adaptive IQUV, base 384, block 8 | 64 | 45.8 | 2.49e+04 | 2.49e+04 (threads) | 26.5 | 4.30e+04 | 22.1 | 5.17e+04 | 1.73 | 2.07 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 150* / 30.6* | 13/56 / 13/48 | none |
| exact-Kerr g-disk 1024² | 64 | 0.743 | 1.41e+06 | 1.41e+06 (threads) | 0.508 | 2.06e+06 | 0.484 | 2.17e+06 | 1.46 | 1.54 | bitwise | 1.63* / 1.68* | 11/13 / 11/11 | none |
| MC thin sphere, 10⁷ | 64 | 5.63 | 1.78e+06 | 1.78e+06 (threads) | 5.58 | 1.80e+06 | 4.42 | 2.27e+06 | 1.01 | 1.27 | bitwise | 7.94* / 9.6* | 9/26 / 14/29 | none |
| MC scattering sphere, 10⁷ | 64 | 5.96 | 1.99e+06 | 1.99e+06 (threads) | 5.56 | 2.13e+06 | 5.36 | 2.21e+06 | 1.07 | 1.11 | bitwise | 8.44* / 11.9* | 10/26 / 10/26 | none |
| native blocks I 384² | 128 | 4.8 | 3.07e+04 | 3.07e+04 (threads) | 2.86 | 5.15e+04 | 1.11 | 1.33e+05 | 1.68 | 4.32 | flux 1.000000000, max ΔI/peak 1.0e-13 | 22.8* / 6.51* | 12/63 / 9/20 | none |
| native blocks IQUV 384² | 128 | 6.71 | 2.20e+04 | 2.20e+04 (threads) | 6.63 | 2.23e+04 | 2.89 | 5.10e+04 | 1.01 | 2.32 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 27.2* / 10.0* | 10/51 / 13/26 | none |
| adaptive IQUV, base 384, block 8 | 128 | 42.1 | 2.71e+04 | 2.71e+04 (threads) | 24.4 | 4.68e+04 | 20.1 | 5.69e+04 | 1.73 | 2.10 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 112* / 29.4* | 11/86 / 11/69 | none |
| exact-Kerr g-disk 1024² | 128 | 0.451 | 2.33e+06 | 2.33e+06 (threads) | 0.314 | 3.34e+06 | 0.348 | 3.01e+06 | 1.44 | 1.44 | bitwise | 1.17* / 1.33* | 19/19 / 11/10 | none |
| MC thin sphere, 10⁷ | 128 | 3.55 | 2.82e+06 | 2.82e+06 (threads) | 3.54 | 2.83e+06 | 3.39 | 2.96e+06 | 1.00 | 1.05 | bitwise | 4.99* / 7.16* | 18/42 / 11/34 | none |
| MC scattering sphere, 10⁷ | 128 | 3.81 | 3.11e+06 | 3.11e+06 (threads) | 4.25 | 2.79e+06 | 4.66 | 2.54e+06 | 0.90 | 0.90 | bitwise | 6.11* / 8.71* | 13/34 / 13/39 | none |
| torus: blocks I 192² | 128 | 1.75 | 2.10e+04 | 2.10e+04 (threads) | 2.92 | 1.26e+04 | 0.407 | 9.06e+04 | 0.60 | 4.31 | flux 1.000000000, max ΔI/peak 1.9e-13 | 9.6* / 5.84* | 9/16 / 9/10 | none |
| torus: blocks I 384² | 128 | 4.77 | 3.09e+04 | 3.09e+04 (threads) | 2.91 | 5.07e+04 | 1.3 | 1.14e+05 | 1.64 | 3.67 | flux 1.000000000, max ΔI/peak 1.4e-11 | 18.8* / 5.9* | 12/58 / 11/20 | none |
| torus: blocks I 768² | 128 | 11.6 | 5.08e+04 | 5.08e+04 (threads) | 6.08 | 9.70e+04 | 4.25 | 1.39e+05 | 1.91 | 2.73 | flux 1.000000000, max ΔI/peak 2.0e-12 | 74.5* / 9.98* | 10/99 / 22/36 | none |
| torus: adaptive IQUV, base 384, block 16 | 128 | 28.8 | 2.19e+04 | 2.19e+04 (threads) | 20.0 | 3.15e+04 | 11.3 | 5.55e+04 | 1.44 | 2.54 | flux 1.000000000, max ΔI/peak 7.3e-10, QUV 4.4e-10, same levels | 72.9* / 26.2* | 11/82 / 12/50 | none |

## Not measured

On two GPUs (no record of either package): fused I 384², uniform resample; fused I 768², uniform resample; fused I 768², mixed precision; fused IQUV 256², uniform resample; fused IQUV 384², uniform resample; native blocks I 384²; native blocks IQUV 384²; adaptive IQUV, base 384, block 8; exact-Kerr g-disk 1024²; MC thin sphere, 10⁷; MC scattering sphere, 10⁷; binary 00455 I 480²; binary 00455 IQUV 480²; binary 00122 IQUV 480²; binary 00455 i = 90° adaptive IQUV; torus: blocks I 192²; torus: blocks I 384²; torus: blocks I 768²; torus: adaptive IQUV, base 384, block 16. No two-device run of these legs had completed when this table was written; the Slurm jobs below show the attempts.

## BLP/reference ratio per hardware (reference deck)

Geometric mean over the legs (smallest–largest), rate ratio BLP/reference.

| hardware | default vs default | best vs best |
|---|---|---|
| 1 GPU | 0.79 (0.31–1.12, 15 legs) | 0.95 (0.54–1.28, 15 legs) |
| CPU, 16 threads | 1.16 (0.97–1.60, 6 legs) | 1.32 (0.98–1.65, 6 legs) |
| CPU, 64 threads | 1.39 (1.01–1.98, 6 legs) | 1.79 (1.11–3.21, 6 legs) |
| CPU, 128 threads | 1.25 (0.90–1.73, 6 legs) | 1.75 (0.90–4.32, 6 legs) |

## Device phases of one profiled call (default configuration)

CUPTI trace of one warm call: the call's window, the host time before the first device activity (prologue) and after the last (epilogue), and per device the union of kernel intervals, the kernel launches and the idle time.

| leg | devices | package | window [ms] | prologue [ms] | epilogue [ms] | kernel [ms] per device | launches per device | idle [ms] per device |
|---|--:|---|--:|--:|--:|---|---|---|
| fused I 384², uniform resample | 1 | reference | 102.3 | 8.6 | 1.3 | 91.7 | 6 | 10.6 |
| fused I 384², uniform resample | 1 | BLP | 80.5 | 4.0 | 1.8 | 74.3 | 3 | 6.1 |
| fused I 768², uniform resample | 1 | reference | 339.8 | 9.4 | 2.5 | 324.8 | 18 | 14.8 |
| fused I 768², uniform resample | 1 | BLP | 356.8 | 3.3 | 18.1 | 331.3 | 9 | 25.3 |
| fused I 768², mixed precision | 1 | reference | 259.0 | 4.2 | 2.9 | 242.7 | 20 | 16.1 |
| fused I 768², mixed precision | 1 | BLP | 326.2 | 7.6 | 22.0 | 292.2 | 10 | 33.7 |
| fused IQUV 256², uniform resample | 1 | reference | 259.2 | 17.7 | 12.9 | 227.5 | 2 | 31.5 |
| fused IQUV 256², uniform resample | 1 | BLP | 216.9 | 3.2 | 2.2 | 211.0 | 1 | 5.8 |
| fused IQUV 384², uniform resample | 1 | reference | 434.5 | 6.8 | 8.5 | 417.6 | 6 | 16.7 |
| fused IQUV 384², uniform resample | 1 | BLP | 433.2 | 9.2 | 7.1 | 415.1 | 3 | 17.9 |
| native blocks I 384² | 1 | reference | 150.2 | 14.3 | 0.8 | 134.1 | 6 | 16.1 |
| native blocks I 384² | 1 | BLP | 134.4 | 4.8 | 0.6 | 128.1 | 3 | 6.3 |
| native blocks IQUV 384² | 1 | reference | 619.4 | 3.4 | 5.8 | 608.4 | 6 | 10.8 |
| native blocks IQUV 384² | 1 | BLP | 714.5 | 3.8 | 2.3 | 707.7 | 3 | 6.7 |
| adaptive IQUV, base 384, block 8 | 1 | reference | 5510.0 | 197.3 | 1.7 | 5298.4 | 46 | 210.8 |
| adaptive IQUV, base 384, block 8 | 1 | BLP | 5990.1 | 22.9 | 3.7 | 5950.2 | 23 | 38.6 |
| exact-Kerr g-disk 1024² | 1 | reference | 115.1 | 9.3 | 7.4 | 58.8 | 16 | 53.4 |
| exact-Kerr g-disk 1024² | 1 | BLP | 388.5 | 7.1 | 304.2 | 46.3 | 16 | 337.9 |
| MC thin sphere, 10⁷ | 1 | reference | 410.0 | 33.5 | 10.5 | 363.4 | 40 | 46.5 |
| MC thin sphere, 10⁷ | 1 | BLP | 763.6 | 29.9 | 3.1 | 528.1 | 153 | 232.7 |
| MC scattering sphere, 10⁷ | 1 | reference | 687.6 | 22.5 | 23.2 | 617.1 | 410 | 68.8 |
| MC scattering sphere, 10⁷ | 1 | BLP | 2014.2 | 102.1 | 13.2 | 854.2 | 182 | 977.4 |
| binary 00455 I 480² | 1 | reference | 1643.0 | 9.1 | 3.1 | 1629.3 | 8 | 13.6 |
| binary 00455 I 480² | 1 | BLP | 1688.4 | 4.1 | 9.9 | 1673.6 | 4 | 14.6 |
| binary 00455 IQUV 480² | 1 | reference | 3772.8 | 6.3 | 14.2 | 3748.6 | 8 | 24.0 |
| binary 00455 IQUV 480² | 1 | BLP | 3979.1 | 3.4 | 12.1 | 3962.3 | 4 | 16.7 |
| binary 00122 IQUV 480² | 1 | reference | 3498.7 | 10.1 | 15.8 | 3471.8 | 8 | 26.8 |
| binary 00122 IQUV 480² | 1 | BLP | 3653.1 | 14.5 | 13.2 | 3621.6 | 4 | 31.2 |
| binary 00455 i = 90° adaptive IQUV | 1 | reference | 16031.0 | 29.1 | 20.1 | 15729.7 | 11 | 300.3 |
| binary 00455 i = 90° adaptive IQUV | 1 | BLP | 17375.0 | 35.3 | 4.4 | 17058.8 | 11 | 315.5 |
| torus: blocks I 192² | 1 | reference | 59.7 | 4.1 | 1.4 | 53.7 | 2 | 6.0 |
| torus: blocks I 192² | 1 | BLP | 66.6 | 9.5 | 1.2 | 54.9 | 1 | 11.7 |
| torus: blocks I 384² | 1 | reference | 158.4 | 11.9 | 1.7 | 143.7 | 6 | 14.6 |
| torus: blocks I 384² | 1 | BLP | 158.6 | 9.6 | 2.9 | 145.0 | 3 | 13.5 |
| torus: blocks I 768² | 1 | reference | 560.2 | 5.2 | 2.7 | 544.8 | 18 | 15.3 |
| torus: blocks I 768² | 1 | BLP | 583.3 | 4.8 | 18.8 | 555.5 | 9 | 27.6 |
| torus: adaptive IQUV, base 384, block 16 | 1 | reference | 3254.4 | 2.7 | 2.0 | 3240.6 | 28 | 13.1 |
| torus: adaptive IQUV, base 384, block 16 | 1 | BLP | 3561.7 | 10.3 | 6.5 | 3529.5 | 14 | 31.5 |

## Every configuration

| leg | backend | devices / threads | package | config | wall [s] | rate | first [s] | load | other cores | same as first config | job |
|---|---|--:|---|---|--:|--:|--:|---|--:|---|---|
| fused I 384², uniform resample | gpu | 1 | reference | chunk131072 | 0.094 | 1.57e+06 | 0.0883 | 7/7 | 5.2 | bitwise | 2998 |
| fused I 384², uniform resample | gpu | 1 | reference | default | 0.102 | 1.45e+06 | 13.1* | 6/7 | 5.6 | first | 2998 |
| fused I 384², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.0841 | 1.75e+06 | 0.245 | 7/7 | 5.6 | bitwise | 2998 |
| fused I 384², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.097 | 1.52e+06 | 12.2* | 7/7 | 5.3 | bitwise | 2998 |
| fused I 768², uniform resample | gpu | 1 | reference | chunk131072 | 0.293 | 2.02e+06 | 0.293 | 7/7 | 5.5 | bitwise | 2998 |
| fused I 768², uniform resample | gpu | 1 | reference | default | 0.334 | 1.76e+06 | 0.344 | 7/7 | 5.8 | first | 2998 |
| fused I 768², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.291 | 2.03e+06 | 0.341 | 7/7 | 5.4 | bitwise | 2998 |
| fused I 768², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.329 | 1.79e+06 | 0.402 | 7/7 | 5.4 | bitwise | 2998 |
| fused I 768², mixed precision | gpu | 1 | reference | chunk131072 | 0.246 | 2.40e+06 | 0.324 | 7/7 | 5.8 | bitwise | 2998 |
| fused I 768², mixed precision | gpu | 1 | reference | default | 0.256 | 2.30e+06 | 2.58* | 7/7 | 5.8 | first | 2998 |
| fused I 768², mixed precision | gpu | 1 | BLP | lanes2_chunk131072 | 0.272 | 2.17e+06 | 0.329 | 7/7 | 5.3 | bitwise | 2998 |
| fused I 768², mixed precision | gpu | 1 | BLP | lanes2_chunk65536 | 0.308 | 1.92e+06 | 1.98* | 7/7 | 5.4 | bitwise | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | reference | chunk131072 | 0.243 | 2.70e+05 | 0.244 | 7/7 | 6.1 | bitwise | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | reference | default | 0.244 | 2.68e+05 | 16.2* | 7/7 | 5.4 | first | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.214 | 3.06e+05 | 0.411 | 7/7 | 5.4 | bitwise | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.219 | 3.00e+05 | 16.5* | 7/7 | 5.9 | bitwise | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | reference | chunk131072 | 0.422 | 3.50e+05 | 0.428 | 7/7 | 5.4 | bitwise | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | reference | default | 0.414 | 3.56e+05 | 0.441 | 7/7 | 5.4 | first | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.323 | 4.56e+05 | 0.391 | 7/7 | 5.4 | bitwise | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.421 | 3.50e+05 | 0.545 | 7/7 | 5.7 | bitwise | 2998 |
| native blocks I 384² | gpu | 1 | reference | chunk131072 | 0.133 | 1.11e+06 | 0.13 | 7/7 | 5.2 | bitwise | 2998 |
| native blocks I 384² | gpu | 1 | reference | default | 0.138 | 1.07e+06 | 12.9* | 7/7 | 5.4 | first | 2998 |
| native blocks I 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.127 | 1.16e+06 | 0.266 | 14/14 | 6.5 | bitwise | 2998 |
| native blocks I 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.14 | 1.05e+06 | 13.3* | 17/14 | 5.3 | bitwise | 2998 |
| native blocks IQUV 384² | gpu | 1 | reference | chunk131072 | 0.513 | 2.87e+05 | 0.543 | 7/7 | 5.4 | bitwise | 2998 |
| native blocks IQUV 384² | gpu | 1 | reference | default | 0.622 | 2.37e+05 | 5.62* | 7/7 | 5.4 | first | 2998 |
| native blocks IQUV 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.616 | 2.39e+05 | 0.658 | 13/13 | 6.4 | bitwise | 2998 |
| native blocks IQUV 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.71 | 2.08e+05 | 6.05* | 14/13 | 32.0 | bitwise | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | reference | chunk131072 | 4.55 | 2.51e+05 | setup 32.6, 4.51 | 9/9 | 5.3 | bitwise | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | reference | default | 5.32 | 2.14e+05 | setup 32.6, 5.39 | 10/9 | 5.3 | first | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | BLP | lanes2_chunk131072 | 5.06 | 2.26e+05 | 5.24 | 8/8 | 7.8 | bitwise | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | BLP | lanes2_chunk65536 | 5.98 | 1.91e+05 | 23.0* | 8/8 | 5.4 | bitwise | 2998 |
| exact-Kerr g-disk 1024² | gpu | 1 | reference | chunk131072 | 0.113 | 9.31e+06 | 0.257 | 8/8 | 5.5 | bitwise | 2998 |
| exact-Kerr g-disk 1024² | gpu | 1 | reference | default | 0.0981 | 1.07e+07 | 16.0* | 8/8 | 5.3 | first | 2998 |
| exact-Kerr g-disk 1024² | gpu | 1 | BLP | lanes2_chunk131072 | 0.163 | 6.42e+06 | 0.162 | 14/14 | 5.2 | bitwise | 2998 |
| exact-Kerr g-disk 1024² | gpu | 1 | BLP | lanes2_chunk65536 | 0.321 | 3.26e+06 | 16.3* | 8/9 | 9.8 | bitwise | 2998 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | chunk131072 | 0.613 | 1.64e+07 | 0.585 | 10/10 | 5.2 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | chunk65536 | 0.841 | 1.19e+07 | 1.01 | 10/10 | 6.6 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | default | 0.41 | 2.45e+07 | 12.7* | 11/10 | 7.3 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk1048576 | 0.362 | 2.77e+07 | 0.374 | 8/8 | 5.2 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk131072 | 0.482 | 2.08e+07 | 0.459 | 8/8 | 6.1 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk65536 | 0.686 | 1.46e+07 | 16.8* | 9/8 | 5.5 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | chunk131072 | 1.17 | 1.01e+07 | 1.13 | 10/9 | 5.3 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | chunk65536 | 2.21 | 5.37e+06 | 2.57 | 10/10 | 5.3 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | default | 0.752 | 1.58e+07 | 9.44* | 10/10 | 5.5 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk1048576 | 1.39 | 8.55e+06 | 1.42 | 8/8 | 5.4 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk131072 | 1.65 | 7.17e+06 | 1.66 | 8/8 | 5.4 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk65536 | 1.88 | 6.31e+06 | 10.5* | 8/8 | 5.3 | bitwise | 2999 |
| binary 00455 I 480² | gpu | 1 | reference | chunk131072 | 1.45 | 1.59e+05 | 1.47 | 7/7 | 5.3 | bitwise | 3000 |
| binary 00455 I 480² | gpu | 1 | reference | default | 1.39 | 1.66e+05 | 14.9* | 7/7 | 5.3 | first | 3000 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 1.55 | 1.48e+05 | 2.02 | 7/7 | 5.3 | bitwise | 3000 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk131072 | 1.79 | 1.29e+05 | 2.88 | 7/7 | 5.4 | bitwise | 3000 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk65536 | 2.05 | 1.13e+05 | 15.8* | 8/7 | 5.4 | first | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | reference | chunk131072 | 3.26 | 7.07e+04 | 3.3 | 7/7 | 18.0 | bitwise | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | reference | default | 3.76 | 6.13e+04 | 10.3* | 7/7 | 5.4 | first | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 2.97 | 7.76e+04 | 3.64 | 19/17 | 15.0 | bitwise | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk131072 | 2.93 | 7.85e+04 | 4.35 | 7/19 | 26.0 | bitwise | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk65536 | 3.94 | 5.85e+04 | 9.87* | 7/7 | 6.6 | first | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | reference | chunk131072 | 2.84 | 8.10e+04 | 2.92 | 10/10 | 5.5 | bitwise | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | reference | default | 3.18 | 7.23e+04 | 19.9* | 14/11 | 5.6 | first | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 2.79 | 8.27e+04 | 3.22 | 7/7 | 5.5 | bitwise | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk131072 | 3.08 | 7.48e+04 | 4.25 | 8/7 | 5.5 | bitwise | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk65536 | 3.63 | 6.35e+04 | 20.5* | 9/8 | 5.4 | first | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | reference | chunk131072 | 14.0 | 4.11e+04 | setup 59.9, 14.1 | 7/7 | 6.1 | bitwise | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | reference | default | 16.0 | 3.59e+04 | setup 59.9, 16.1 | 7/7 | 6.0 | first | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk1048576 | 13.9 | 4.13e+04 | 15.0 | 7/7 | 6.3 | bitwise | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk131072 | 15.1 | 3.80e+04 | 16.1 | 7/7 | 6.5 | bitwise | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk65536 | 17.1 | 3.36e+04 | 35.5* | 7/7 | 5.8 | first | 3000 |
| torus: blocks I 192² | gpu | 1 | reference | chunk131072 | 0.0582 | 6.34e+05 | 0.0578 | 10/10 | 5.2 | bitwise | 2998 |
| torus: blocks I 192² | gpu | 1 | reference | default | 0.0568 | 6.49e+05 | 13.3* | 10/10 | 6.3 | first | 2998 |
| torus: blocks I 192² | gpu | 1 | BLP | lanes2_chunk131072 | 0.056 | 6.58e+05 | 0.182 | 9/9 | 8.6 | bitwise | 2998 |
| torus: blocks I 192² | gpu | 1 | BLP | lanes2_chunk65536 | 0.0559 | 6.59e+05 | 14.0* | 8/8 | 9.1 | bitwise | 2998 |
| torus: blocks I 384² | gpu | 1 | reference | chunk131072 | 0.143 | 1.03e+06 | 0.148 | 9/9 | 5.3 | bitwise | 2998 |
| torus: blocks I 384² | gpu | 1 | reference | default | 0.148 | 9.96e+05 | 0.178 | 9/9 | 5.4 | first | 2998 |
| torus: blocks I 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.137 | 1.08e+06 | 0.174 | 9/9 | 5.6 | bitwise | 2998 |
| torus: blocks I 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.151 | 9.75e+05 | 0.187 | 9/9 | 6.3 | bitwise | 2998 |
| torus: blocks I 768² | gpu | 1 | reference | chunk131072 | 0.491 | 1.20e+06 | 0.501 | 9/9 | 6.1 | bitwise | 2998 |
| torus: blocks I 768² | gpu | 1 | reference | default | 0.553 | 1.07e+06 | 0.587 | 9/9 | 5.3 | first | 2998 |
| torus: blocks I 768² | gpu | 1 | BLP | lanes2_chunk131072 | 0.494 | 1.19e+06 | 0.626 | 8/8 | 5.5 | bitwise | 2998 |
| torus: blocks I 768² | gpu | 1 | BLP | lanes2_chunk65536 | 0.569 | 1.04e+06 | 0.607 | 9/8 | 5.3 | bitwise | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | reference | chunk131072 | 2.71 | 2.33e+05 | 2.74 | 8/7 | 5.4 | bitwise | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | reference | default | 3.29 | 1.91e+05 | 19.3* | 8/8 | 15.0 | first | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | BLP | lanes2_chunk131072 | 2.99 | 2.11e+05 | 3.14 | 7/8 | 12.0 | bitwise | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | BLP | lanes2_chunk65536 | 3.42 | 1.84e+05 | 20.8* | 7/7 | 9.4 | bitwise | 2998 |
| native blocks I 384² | cpu | 16 | reference | threads | 8.68 | 1.70e+04 | 15.4* | 6/9 | 3.8 | first | none |
| native blocks I 384² | cpu | 16 | BLP | chunk4096 | 7.15 | 2.06e+04 | 10.7* | 8/9 | 5.8 | first | none |
| native blocks I 384² | cpu | 16 | BLP | chunk64 | 5.46 | 2.70e+04 | 5.61 | 9/10 | 3.6 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | reference | threads | 17.0 | 8677 | 31.9* | 13/15 | 4.8 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | BLP | chunk4096 | 16.9 | 8735 | 20.7* | 12/13 | 3.9 | first | none |
| native blocks IQUV 384² | cpu | 16 | BLP | chunk64 | 12.9 | 1.14e+04 | 13.2 | 12/15 | 4.6 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | reference | threads | 123 | 9276 | 142* | 14/16 | 2.8 | first | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | BLP | chunk4096 | 77.1 | 1.48e+04 | 80.1* | 13/19 | 5.8 | first | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | BLP | chunk64 | 74.7 | 1.53e+04 | 75.4 | 19/19 | 3.5 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 16 | reference | threads | 1.28 | 8.18e+05 | 2.5* | 11/12 | 2.5 | first | none |
| exact-Kerr g-disk 1024² | cpu | 16 | BLP | chunk4096 | 0.968 | 1.08e+06 | 1.96* | 10/11 | 2.6 | first | none |
| exact-Kerr g-disk 1024² | cpu | 16 | BLP | chunk64 | 0.797 | 1.32e+06 | 0.792 | 11/12 | 2.5 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 16 | reference | threads | 14.3 | 7.00e+05 | 16.0* | 10/15 | 2.6 | first | none |
| MC thin sphere, 10⁷ | cpu | 16 | BLP | chunk4096 | 14.7 | 6.83e+05 | 18.8* | 14/17 | 2.4 | first | none |
| MC thin sphere, 10⁷ | cpu | 16 | BLP | chunk64 | 14.5 | 6.92e+05 | 18.9* | 14/16 | 2.5 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 16 | reference | threads | 15.1 | 7.85e+05 | 17.7* | 10/15 | 2.5 | first | none |
| MC scattering sphere, 10⁷ | cpu | 16 | BLP | chunk4096 | 15.5 | 7.65e+05 | 20.3* | 13/16 | 2.6 | first | none |
| MC scattering sphere, 10⁷ | cpu | 16 | BLP | chunk64 | 15.4 | 7.69e+05 | 16.6 | 15/17 | 2.4 | bitwise | none |
| native blocks I 384² | cpu | 64 | reference | threads | 5.0 | 2.95e+04 | 24.3* | 9/25 | 3.2 | bitwise | none |
| native blocks I 384² | cpu | 64 | BLP | chunk4096 | 2.52 | 5.86e+04 | 6.36* | 13/16 | 2.4 | bitwise | none |
| native blocks I 384² | cpu | 64 | BLP | chunk64 | 1.56 | 9.47e+04 | 2.11 | 15/19 | 2.3 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | reference | threads | 8.17 | 1.80e+04 | 20.7* | 10/38 | 5.5 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | BLP | chunk4096 | 6.15 | 2.40e+04 | 9.86* | 13/22 | 2.4 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | BLP | chunk64 | 3.65 | 4.04e+04 | 4.29 | 21/29 | 2.3 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | reference | threads | 45.8 | 2.49e+04 | 150* | 13/56 | 3.6 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | BLP | chunk4096 | 26.5 | 4.30e+04 | 30.6* | 13/48 | 2.4 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | BLP | chunk64 | 22.1 | 5.17e+04 | 22.7 | 45/62 | 3.1 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | reference | threads | 0.743 | 1.41e+06 | 1.63* | 11/13 | 2.7 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | BLP | chunk4096 | 0.508 | 2.06e+06 | 1.68* | 11/11 | 2.0 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | BLP | chunk64 | 0.484 | 2.17e+06 | 0.645 | 11/11 | 2.5 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | reference | threads | 5.63 | 1.78e+06 | 7.94* | 9/26 | 2.3 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | BLP | chunk4096 | 5.58 | 1.80e+06 | 9.6* | 14/29 | 2.4 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | BLP | chunk64 | 4.42 | 2.27e+06 | 5.5 | 29/40 | 2.4 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | reference | threads | 5.96 | 1.99e+06 | 8.44* | 10/26 | 2.6 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | BLP | chunk4096 | 5.56 | 2.13e+06 | 11.9* | 10/26 | 2.4 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | BLP | chunk64 | 5.36 | 2.21e+06 | 5.77 | 26/38 | 2.4 | bitwise | none |
| native blocks I 384² | cpu | 128 | reference | threads | 4.8 | 3.07e+04 | 22.8* | 12/63 | 4.0 | bitwise | none |
| native blocks I 384² | cpu | 128 | BLP | chunk4096 | 2.86 | 5.15e+04 | 6.51* | 9/20 | 3.4 | bitwise | none |
| native blocks I 384² | cpu | 128 | BLP | chunk64 | 1.11 | 1.33e+05 | 1.28 | 19/27 | 2.8 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | reference | threads | 6.71 | 2.20e+04 | 27.2* | 10/51 | 3.1 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | BLP | chunk4096 | 6.63 | 2.23e+04 | 10.0* | 13/26 | 2.4 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | BLP | chunk64 | 2.89 | 5.10e+04 | 3.11 | 26/47 | 2.4 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | reference | threads | 42.1 | 2.71e+04 | 112* | 11/86 | 3.4 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | BLP | chunk4096 | 24.4 | 4.68e+04 | 29.4* | 11/69 | 2.5 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | BLP | chunk64 | 20.1 | 5.69e+04 | 20.5 | 69/108 | 2.3 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | reference | threads | 0.451 | 2.33e+06 | 1.17* | 19/19 | 2.4 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | BLP | chunk4096 | 0.314 | 3.34e+06 | 1.33* | 11/10 | 2.4 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | BLP | chunk64 | 0.348 | 3.01e+06 | 0.369 | 10/10 | 2.8 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | reference | threads | 3.55 | 2.82e+06 | 4.99* | 18/42 | 2.2 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | BLP | chunk4096 | 3.54 | 2.83e+06 | 7.16* | 11/34 | 2.3 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | BLP | chunk64 | 3.39 | 2.96e+06 | 3.46 | 34/55 | 2.2 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | reference | threads | 3.81 | 3.11e+06 | 6.11* | 13/34 | 2.4 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | BLP | chunk4096 | 4.25 | 2.79e+06 | 8.71* | 13/39 | 2.3 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | BLP | chunk64 | 4.66 | 2.54e+06 | 4.91 | 39/65 | 2.4 | bitwise | none |
| torus: blocks I 192² | cpu | 128 | reference | threads | 1.75 | 2.10e+04 | 9.6* | 9/16 | 3.4 | first | none |
| torus: blocks I 192² | cpu | 128 | BLP | chunk4096 | 2.92 | 1.26e+04 | 5.84* | 9/10 | 3.5 | first | none |
| torus: blocks I 192² | cpu | 128 | BLP | chunk64 | 0.407 | 9.06e+04 | 0.435 | 10/10 | 3.4 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | reference | threads | 4.77 | 3.09e+04 | 18.8* | 12/58 | 5.9 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | BLP | chunk4096 | 2.91 | 5.07e+04 | 5.9* | 11/20 | 4.8 | first | none |
| torus: blocks I 384² | cpu | 128 | BLP | chunk64 | 1.3 | 1.14e+05 | 1.27 | 20/27 | 3.1 | bitwise | none |
| torus: blocks I 768² | cpu | 128 | reference | threads | 11.6 | 5.08e+04 | 74.5* | 10/99 | 4.9 | first | none |
| torus: blocks I 768² | cpu | 128 | BLP | chunk4096 | 6.08 | 9.70e+04 | 9.98* | 22/36 | 2.6 | first | none |
| torus: blocks I 768² | cpu | 128 | BLP | chunk64 | 4.25 | 1.39e+05 | 4.24 | 36/49 | 4.6 | bitwise | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | reference | threads | 28.8 | 2.19e+04 | 72.9* | 11/82 | 4.2 | first | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | BLP | chunk4096 | 20.0 | 3.15e+04 | 26.2* | 12/50 | 4.3 | first | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | BLP | chunk64 | 11.3 | 5.55e+04 | 11.5 | 47/84 | 5.6 | bitwise | none |

## Slurm jobs

Every submission of `slurm/run_reference.sh` (partition `preempt`): a job that ended PREEMPTED or CANCELLED was resubmitted and its successor skipped the legs already recorded; `exit` is the status at the end of the job's log (`none`: the job was cut off). A job whose log ends in a nonzero exit is not retried; its log is under `logs/superseded/`, and the BLP records of its legs whose reference half did not run are under `raw/superseded/`, outside the tables.

| job | item | attempt | submitted | started | ended | final state | exit |
|--:|---|--:|---|---|---|---|---|
| 2994 | gpu1_f87f96d2 | 1 | 2026-10-17 16:37:43 | 2026-10-17 21:12:12 | 2026-10-17 21:22:42 | COMPLETED | 1 |
| 2995 | gpu2_f87f96d2 | 1 | 2026-10-17 18:50:11 | 2026-10-17 21:22:45 | 2026-10-17 21:26:15 | PREEMPTED | none |
| 2997 | gpu2_f87f96d2 | 2 | 2026-10-17 21:26:15 | – | 2026-10-18 00:08:50 | CANCELLED | – |
| 2998 | gpu1_f87f96d2 | 1 | 2026-10-17 21:40:11 | 2026-10-17 21:40:41 | 2026-10-17 21:55:11 | COMPLETED | 0 |
| 2999 | gpu1_efa03062 | 1 | 2026-10-17 21:57:10 | 2026-10-17 21:57:40 | 2026-10-17 22:00:10 | COMPLETED | 0 |
| 3000 | gpu1_e41829b9 | 1 | 2026-10-17 22:00:10 | 2026-10-17 22:00:40 | 2026-10-17 22:15:11 | COMPLETED | 0 |
| 3007 | gpu2_f87f96d2 | 1 | 2026-10-18 00:13:22 | – | – | PENDING | – |

Items (`tag`: `<devices>@<deck>=<legs>/…`, see `slurm/run_reference.sh`):

- `gpu1_f87f96d2`: `1@reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384/reference=amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384`
- `gpu2_f87f96d2`: `2@reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384/reference=amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384`
- `gpu1_efa03062`: `1@reference=mc_thin,mc_scatter`
- `gpu1_e41829b9`: `1@reference=bbh455_480,bbh455_pol_480/reference=bbh122_pol_480/reference=bbh_i90_amr_pol`
