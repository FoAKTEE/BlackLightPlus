# BLP against the reference package on anta, 2026-10-18

Host: 2× Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz (72 cores, 144 hardware threads), 2× NVIDIA A100 80 GB PCIe; Julia 1.12.6; commits 8b9a3ba, 97b23ac, b40fdb1, bf486b4; reference `src/` trees e76ab3c; BLP `BLP/src`+`BLP/ext` trees 2cf2367+c8ef624, 7913a3e+c8ef624, cc870b9+c8ef624, ced90bb+c8ef624. Both GPU environments (`results/blp/gpu/env_old` for the reference, `results/blp/gpu/env` for BLP) hold the same CUDA.jl, GPUCompiler, LLVM.jl and KernelAbstractions versions and the CUDA runtime 13.3.

Every GPU row ran inside a Slurm job on partition `preempt` with 4 Julia threads; the reference leg and the BLP leg of a row ran in the same job, one process each, back to back, on the same devices. CPU rows ran on the login shell, started only when the 1-minute load was below 16. `wall` is the best of 3 calls after a first call; `first` is that first call (`*`: it compiled the leg's program; `setup`: the reference harness runs the AMR driver once on one device in its setup to count the rays, which then holds the compilation). The uploads of the reference harness's setup are outside `first`; BLP uploads inside its first call. `load` is the 1-minute load average before/after the timed calls. Ratio: BLP rate over reference rate (above 1: BLP faster); `default` pairs the defaults (reference: the harness's call; BLP: PoolExecutor lanes 2 or ThreadedExecutor, both at chunk `auto`), `best` the best configuration of each package. Agreement: the BLP output against the reference output of the same hardware.

The reference's GPU configurations are its harness default (the cached DevicePool of lanes 2, chunk 65536; the Monte Carlo drivers cut their own chunks of 2^20 primaries) and a DevicePool of lanes 2, chunk 131072 (Monte Carlo: the drivers' `chunk` keyword at 65536 and 131072). BLP's: lanes 2 at chunk `auto` (default: one launch of up to 2^20 items on one device, at most 16 MiB of outputs per launch, the Monte Carlo cascade at 2^20 primaries), 131072 (tuned) and 65536, and where recorded 1048576 (the Monte Carlo and binary legs, which then count in BLP's best). On the CPU the reference runs its CPU entry points: the two-phase render (trace, then march along the stored states), `render_amr` with its two-phase closure (which reads the snapshot in each call), `render_analytic`, `mc_spectrum`, `mc_spectrum_scatter`; BLP runs the same products as on the GPU with ThreadedExecutor chunk `auto` (default), 64 (tuned) and 4096.


## GPU

| leg | devices | reference wall [s] | reference rate | reference best (config) | BLP wall [s] default | BLP rate default | BLP wall [s] tuned | BLP rate tuned | BLP/ref default | BLP/ref best | agreement | first [s] ref / BLP | load ref / BLP | jobs |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| fused I 384², uniform resample | 1 | 0.102 | 1.45e+06 | 1.57e+06 (chunk 131072) | 0.0781 | 1.89e+06 | 0.0842 | 1.75e+06 | 1.30 | 1.20 | bitwise | 13.1* / 14.0* | 6/7 / 8/8 | 2998, 3023 |
| fused I 768², uniform resample | 1 | 0.334 | 1.76e+06 | 2.02e+06 (chunk 131072) | 0.289 | 2.04e+06 | 0.287 | 2.05e+06 | 1.16 | 1.02 | bitwise | 0.344 / 0.334 | 7/7 / 9/9 | 2998, 3023 |
| fused I 768², mixed precision | 1 | 0.256 | 2.30e+06 | 2.40e+06 (chunk 131072) | 0.278 | 2.12e+06 | 0.281 | 2.10e+06 | 0.92 | 0.88 | bitwise | 2.58* / 2.66* | 7/7 / 9/9 | 2998, 3023 |
| fused IQUV 256², uniform resample | 1 | 0.244 | 2.68e+05 | 2.70e+05 (chunk 131072) | 0.211 | 3.10e+05 | 0.211 | 3.11e+05 | 1.15 | 1.15 | bitwise | 16.2* / 6.89* | 7/7 / 9/14 | 2998, 3023 |
| fused IQUV 384², uniform resample | 1 | 0.414 | 3.56e+05 | 3.56e+05 (default) | 0.339 | 4.35e+05 | 0.366 | 4.03e+05 | 1.22 | 1.22 | bitwise | 0.441 / 0.389 | 7/7 / 13/13 | 2998, 3023 |
| native blocks I 384² | 1 | 0.138 | 1.07e+06 | 1.11e+06 (chunk 131072) | 0.125 | 1.18e+06 | 0.129 | 1.15e+06 | 1.11 | 1.07 | bitwise | 12.9* / 14.2* | 7/7 / 11/9 | 2998, 3023 |
| native blocks IQUV 384² | 1 | 0.622 | 2.37e+05 | 2.87e+05 (chunk 131072) | 0.568 | 2.60e+05 | 0.62 | 2.38e+05 | 1.10 | 0.90 | bitwise | 5.62* / 5.83* | 7/7 / 9/8 | 2998, 3023 |
| adaptive IQUV, base 384, block 8 | 1 | 5.32 | 2.14e+05 | 2.51e+05 (chunk 131072) | 4.59 | 2.49e+05 | 5.07 | 2.25e+05 | 1.16 | 0.99 | bitwise | setup 32.6, 5.39 / 9.64* | 10/9 / 9/8 | 2998, 3023 |
| exact-Kerr g-disk 1024² | 1 | 0.0868 | 1.21e+07 | 1.21e+07 (default) | 0.065 | 1.61e+07 | 0.0832 | 1.26e+07 | 1.34 | 1.34 | bitwise | 15.6* / 17.1* | 17/13 / 11/9 | 3029 |
| MC thin sphere, 10⁷ | 1 | 0.41 | 2.45e+07 | 2.45e+07 (default) | 0.357 | 2.81e+07 | 0.366 | 2.74e+07 | 1.15 | 1.15 | bitwise | 12.7* / 5.27* | 11/10 / 7/7 | 2999, 3023 |
| MC scattering sphere, 10⁷ | 1 | 0.752 | 1.58e+07 | 1.58e+07 (default) | 0.742 | 1.60e+07 | 0.719 | 1.65e+07 | 1.01 | 1.07 | bitwise | 9.44* / 9.91* | 10/10 / 7/7 | 2999, 3023 |
| binary 00455 I 480² | 1 | 1.39 | 1.66e+05 | 1.66e+05 (default) | 1.55 | 1.48e+05 | 1.78 | 1.29e+05 | 0.89 | 0.89 | bitwise | 14.9* / 16.3* | 7/7 / 6/5 | 3000, 3026 |
| binary 00455 IQUV 480² | 1 | 3.76 | 6.13e+04 | 7.07e+04 (chunk 131072) | 2.93 | 7.87e+04 | 3.44 | 6.71e+04 | 1.28 | 1.11 | bitwise | 10.3* / 9.45* | 7/7 / 5/4 | 3000, 3026 |
| binary 00122 IQUV 480² | 1 | 3.18 | 7.23e+04 | 8.10e+04 (chunk 131072) | 2.77 | 8.32e+04 | 2.88 | 7.99e+04 | 1.15 | 1.03 | bitwise | 19.9* / 3.23 | 14/11 / 4/4 | 3000, 3026 |
| binary 00455 i = 90° adaptive IQUV | 1 | 16.0 | 3.59e+04 | 4.11e+04 (chunk 131072) | 13.9 | 4.13e+04 | 14.3 | 4.02e+04 | 1.15 | 1.01 | bitwise | setup 59.9, 16.1 / 19.8* | 7/7 / 3/3 | 3000, 3026 |
| torus: blocks I 192² | 1 | 0.0568 | 6.49e+05 | 6.49e+05 (default) | 0.0567 | 6.50e+05 | 0.0563 | 6.55e+05 | 1.00 | 1.01 | bitwise | 13.3* / 13.5* | 10/10 / 6/6 | 2998, 3023 |
| torus: blocks I 384² | 1 | 0.148 | 9.96e+05 | 1.03e+06 (chunk 131072) | 0.129 | 1.14e+06 | 0.136 | 1.08e+06 | 1.14 | 1.10 | bitwise | 0.178 / 0.205 | 9/9 / 6/6 | 2998, 3023 |
| torus: blocks I 768² | 1 | 0.553 | 1.07e+06 | 1.20e+06 (chunk 131072) | 0.472 | 1.25e+06 | 0.486 | 1.21e+06 | 1.17 | 1.04 | bitwise | 0.587 / 0.519 | 9/9 / 6/6 | 2998, 3023 |
| torus: adaptive IQUV, base 384, block 16 | 1 | 3.29 | 1.91e+05 | 2.33e+05 (chunk 131072) | 2.74 | 2.30e+05 | 3.05 | 2.06e+05 | 1.20 | 0.99 | bitwise | 19.3* / 9.74* | 8/8 / 6/23 | 2998, 3023 |

## CPU

| leg | threads | reference wall [s] | reference rate | reference best (config) | BLP wall [s] default | BLP rate default | BLP wall [s] tuned | BLP rate tuned | BLP/ref default | BLP/ref best | agreement | first [s] ref / BLP | load ref / BLP | jobs |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|--:|---|---|---|---|
| native blocks I 384² | 16 | 8.68 | 1.70e+04 | 1.70e+04 (threads) | 4.27 | 3.45e+04 | 4.2 | 3.51e+04 | 2.03 | 2.07 | flux 1.000000000, max ΔI/peak 1.0e-13 | 15.4* / 7.63* | 6/9 / 3/8 | none |
| native blocks IQUV 384² | 16 | 17.0 | 8677 | 8677 (threads) | 10.7 | 1.37e+04 | 10.7 | 1.38e+04 | 1.58 | 1.59 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 31.9* / 14.3* | 13/15 / 13/17 | none |
| adaptive IQUV, base 384, block 8 | 16 | 123 | 9276 | 9276 (threads) | 76.4 | 1.49e+04 | 77.1 | 1.48e+04 | 1.61 | 1.61 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 142* / 81.4* | 14/16 / 10/20 | none |
| exact-Kerr g-disk 1024² | 16 | 1.28 | 8.18e+05 | 8.18e+05 (threads) | 0.731 | 1.43e+06 | 0.659 | 1.59e+06 | 1.75 | 1.95 | bitwise | 2.5* / 1.96* | 11/12 / 14/14 | none |
| MC thin sphere, 10⁷ | 16 | 14.3 | 7.00e+05 | 7.00e+05 (threads) | 14.7 | 6.82e+05 | 14.6 | 6.89e+05 | 0.97 | 0.98 | bitwise | 16.0* / 19.7* | 10/15 / 12/24 | none |
| MC scattering sphere, 10⁷ | 16 | 15.1 | 7.85e+05 | 7.85e+05 (threads) | 15.1 | 7.86e+05 | 14.9 | 7.96e+05 | 1.00 | 1.01 | bitwise | 17.7* / 20.9* | 10/15 / 13/17 | none |
| native blocks I 384² | 64 | 5.0 | 2.95e+04 | 2.95e+04 (threads) | 1.46 | 1.01e+05 | 1.19 | 1.23e+05 | 3.42 | 4.18 | flux 1.000000000, max ΔI/peak 1.0e-13 | 24.3* / 5.26* | 9/25 / 11/20 | none |
| native blocks IQUV 384² | 64 | 8.17 | 1.80e+04 | 1.80e+04 (threads) | 3.0 | 4.91e+04 | 2.95 | 5.00e+04 | 2.72 | 2.77 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 20.7* / 7.53* | 10/38 / 14/29 | none |
| adaptive IQUV, base 384, block 8 | 64 | 45.8 | 2.49e+04 | 2.49e+04 (threads) | 21.7 | 5.26e+04 | 21.8 | 5.23e+04 | 2.11 | 2.11 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 150* / 26.3* | 13/56 / 7/53 | none |
| exact-Kerr g-disk 1024² | 64 | 0.743 | 1.41e+06 | 1.41e+06 (threads) | 0.271 | 3.88e+06 | 0.229 | 4.57e+06 | 2.75 | 3.24 | bitwise | 1.63* / 1.96* | 11/13 / 11/11 | none |
| MC thin sphere, 10⁷ | 64 | 5.63 | 1.78e+06 | 1.78e+06 (threads) | 4.45 | 2.25e+06 | 4.14 | 2.42e+06 | 1.27 | 1.36 | bitwise | 7.94* / 8.92* | 9/26 / 10/27 | none |
| MC scattering sphere, 10⁷ | 64 | 5.96 | 1.99e+06 | 1.99e+06 (threads) | 4.35 | 2.73e+06 | 4.36 | 2.72e+06 | 1.37 | 1.37 | bitwise | 8.44* / 10.9* | 10/26 / 11/21 | none |
| native blocks I 384² | 128 | 4.8 | 3.07e+04 | 3.07e+04 (threads) | 0.842 | 1.75e+05 | 0.827 | 1.78e+05 | 5.70 | 5.81 | flux 1.000000000, max ΔI/peak 1.0e-13 | 22.8* / 4.34* | 12/63 / 10/18 | none |
| native blocks IQUV 384² | 128 | 6.71 | 2.20e+04 | 2.20e+04 (threads) | 2.53 | 5.82e+04 | 2.54 | 5.80e+04 | 2.65 | 2.65 | flux 1.000000000, max ΔI/peak 1.7e-12, QUV 1.3e-12 | 27.2* / 7.37* | 10/51 / 10/36 | none |
| adaptive IQUV, base 384, block 8 | 128 | 42.1 | 2.71e+04 | 2.71e+04 (threads) | 18.6 | 6.14e+04 | 18.7 | 6.10e+04 | 2.27 | 2.27 | flux 1.000000000, max ΔI/peak 2.6e-11, QUV 2.0e-11, same levels | 112* / 23.6* | 11/86 / 10/96 | none |
| exact-Kerr g-disk 1024² | 128 | 0.451 | 2.33e+06 | 2.33e+06 (threads) | 0.229 | 4.59e+06 | 0.189 | 5.54e+06 | 1.97 | 2.38 | bitwise | 1.17* / 2.31* | 19/19 / 10/10 | none |
| MC thin sphere, 10⁷ | 128 | 3.55 | 2.82e+06 | 2.82e+06 (threads) | 3.22 | 3.12e+06 | 3.22 | 3.11e+06 | 1.10 | 1.10 | bitwise | 4.99* / 7.02* | 18/42 / 12/38 | none |
| MC scattering sphere, 10⁷ | 128 | 3.81 | 3.11e+06 | 3.11e+06 (threads) | 3.49 | 3.40e+06 | 3.51 | 3.38e+06 | 1.09 | 1.09 | bitwise | 6.11* / 10.2* | 13/34 / 10/43 | none |
| torus: blocks I 192² | 128 | 1.75 | 2.10e+04 | 2.10e+04 (threads) | 0.266 | 1.38e+05 | 0.28 | 1.32e+05 | 6.58 | 6.58 | flux 1.000000000, max ΔI/peak 1.9e-13 | 9.6* / 3.94* | 9/16 / 11/21 | none |
| torus: blocks I 384² | 128 | 4.77 | 3.09e+04 | 3.09e+04 (threads) | 0.902 | 1.63e+05 | 0.88 | 1.68e+05 | 5.29 | 5.42 | flux 1.000000000, max ΔI/peak 1.4e-11 | 18.8* / 4.44* | 12/58 / 9/19 | none |
| torus: blocks I 768² | 128 | 11.6 | 5.08e+04 | 5.08e+04 (threads) | 3.42 | 1.73e+05 | 3.38 | 1.74e+05 | 3.40 | 3.43 | flux 1.000000000, max ΔI/peak 2.0e-12 | 74.5* / 7.26* | 10/99 / 10/42 | none |
| torus: adaptive IQUV, base 384, block 16 | 128 | 28.8 | 2.19e+04 | 2.19e+04 (threads) | 11.0 | 5.72e+04 | 11.0 | 5.74e+04 | 2.61 | 2.62 | flux 1.000000000, max ΔI/peak 7.3e-10, QUV 4.4e-10, same levels | 72.9* / 15.6* | 11/82 / 18/73 | none |

## Not measured

On two GPUs (no record of either package): fused I 384², uniform resample; fused I 768², uniform resample; fused I 768², mixed precision; fused IQUV 256², uniform resample; fused IQUV 384², uniform resample; native blocks I 384²; native blocks IQUV 384²; adaptive IQUV, base 384, block 8; exact-Kerr g-disk 1024²; MC thin sphere, 10⁷; MC scattering sphere, 10⁷; binary 00455 I 480²; binary 00455 IQUV 480²; binary 00122 IQUV 480²; binary 00455 i = 90° adaptive IQUV; torus: blocks I 192²; torus: blocks I 384²; torus: blocks I 768²; torus: adaptive IQUV, base 384, block 16. No two-device run of these legs had completed when this table was written; the Slurm jobs below show the attempts.

## BLP/reference ratio per hardware (reference deck)

Geometric mean over the legs (smallest–largest), rate ratio BLP/reference.

| hardware | default vs default | best vs best |
|---|---|---|
| 1 GPU | 1.13 (0.89–1.34, 15 legs) | 1.06 (0.88–1.34, 15 legs) |
| CPU, 16 threads | 1.44 (0.97–2.03, 6 legs) | 1.47 (0.98–2.07, 6 legs) |
| CPU, 64 threads | 2.13 (1.27–3.42, 6 legs) | 2.30 (1.36–4.18, 6 legs) |
| CPU, 128 threads | 2.08 (1.09–5.70, 6 legs) | 2.16 (1.09–5.81, 6 legs) |

## Device phases of one profiled call (default configuration)

CUPTI trace of one warm call: the call's window, the host time before the first device activity (prologue) and after the last (epilogue), and per device the union of kernel intervals, the kernel launches and the idle time.

| leg | devices | package | window [ms] | prologue [ms] | epilogue [ms] | kernel [ms] per device | launches per device | idle [ms] per device |
|---|--:|---|--:|--:|--:|---|---|---|
| fused I 384², uniform resample | 1 | reference | 102.3 | 8.6 | 1.3 | 91.7 | 6 | 10.6 |
| fused I 384², uniform resample | 1 | BLP | 86.9 | 11.8 | 1.0 | 73.3 | 1 | 13.5 |
| fused I 768², uniform resample | 1 | reference | 339.8 | 9.4 | 2.5 | 324.8 | 18 | 14.8 |
| fused I 768², uniform resample | 1 | BLP | 291.0 | 5.8 | 14.1 | 270.2 | 1 | 20.5 |
| fused I 768², mixed precision | 1 | reference | 259.0 | 4.2 | 2.9 | 242.7 | 20 | 16.1 |
| fused I 768², mixed precision | 1 | BLP | 279.6 | 10.3 | 18.0 | 249.9 | 2 | 29.6 |
| fused IQUV 256², uniform resample | 1 | reference | 259.2 | 17.7 | 12.9 | 227.5 | 2 | 31.5 |
| fused IQUV 256², uniform resample | 1 | BLP | 290.8 | 1.6 | 5.9 | 212.5 | 1 | 78.1 |
| fused IQUV 384², uniform resample | 1 | reference | 434.5 | 6.8 | 8.5 | 417.6 | 6 | 16.7 |
| fused IQUV 384², uniform resample | 1 | BLP | 353.8 | 12.6 | 12.6 | 327.6 | 1 | 26.0 |
| native blocks I 384² | 1 | reference | 150.2 | 14.3 | 0.8 | 134.1 | 6 | 16.1 |
| native blocks I 384² | 1 | BLP | 123.2 | 4.6 | 1.1 | 116.9 | 1 | 6.2 |
| native blocks IQUV 384² | 1 | reference | 619.4 | 3.4 | 5.8 | 608.4 | 6 | 10.8 |
| native blocks IQUV 384² | 1 | BLP | 579.6 | 8.8 | 3.1 | 566.9 | 1 | 12.5 |
| adaptive IQUV, base 384, block 8 | 1 | reference | 5510.0 | 197.3 | 1.7 | 5298.4 | 46 | 210.8 |
| adaptive IQUV, base 384, block 8 | 1 | BLP | 4496.2 | 7.1 | 7.2 | 4423.2 | 4 | 72.0 |
| exact-Kerr g-disk 1024² | 1 | reference | 305.1 | 192.5 | 21.0 | 56.6 | 16 | 244.5 |
| exact-Kerr g-disk 1024² | 1 | BLP | 111.0 | 46.2 | 4.8 | 56.4 | 8 | 53.7 |
| MC thin sphere, 10⁷ | 1 | reference | 410.0 | 33.5 | 10.5 | 363.4 | 40 | 46.5 |
| MC thin sphere, 10⁷ | 1 | BLP | 353.8 | 18.2 | 7.8 | 324.0 | 10 | 29.6 |
| MC scattering sphere, 10⁷ | 1 | reference | 687.6 | 22.5 | 23.2 | 617.1 | 410 | 68.8 |
| MC scattering sphere, 10⁷ | 1 | BLP | 744.0 | 28.3 | 26.0 | 646.2 | 72 | 96.8 |
| binary 00455 I 480² | 1 | reference | 1643.0 | 9.1 | 3.1 | 1629.3 | 8 | 13.6 |
| binary 00455 I 480² | 1 | BLP | 1553.1 | 3.8 | 5.1 | 1543.7 | 1 | 9.3 |
| binary 00455 IQUV 480² | 1 | reference | 3772.8 | 6.3 | 14.2 | 3748.6 | 8 | 24.0 |
| binary 00455 IQUV 480² | 1 | BLP | 2958.2 | 10.6 | 8.3 | 2938.2 | 1 | 19.6 |
| binary 00122 IQUV 480² | 1 | reference | 3498.7 | 10.1 | 15.8 | 3471.8 | 8 | 26.8 |
| binary 00122 IQUV 480² | 1 | BLP | 2838.8 | 11.8 | 7.6 | 2818.7 | 1 | 19.8 |
| binary 00455 i = 90° adaptive IQUV | 1 | reference | 16031.0 | 29.1 | 20.1 | 15729.7 | 11 | 300.3 |
| binary 00455 i = 90° adaptive IQUV | 1 | BLP | 13916.6 | 10.2 | 19.3 | 13547.0 | 5 | 368.7 |
| torus: blocks I 192² | 1 | reference | 59.7 | 4.1 | 1.4 | 53.7 | 2 | 6.0 |
| torus: blocks I 192² | 1 | BLP | 63.9 | 6.7 | 1.6 | 54.8 | 1 | 9.1 |
| torus: blocks I 384² | 1 | reference | 158.4 | 11.9 | 1.7 | 143.7 | 6 | 14.6 |
| torus: blocks I 384² | 1 | BLP | 137.7 | 5.6 | 5.7 | 125.9 | 1 | 11.8 |
| torus: blocks I 768² | 1 | reference | 560.2 | 5.2 | 2.7 | 544.8 | 18 | 15.3 |
| torus: blocks I 768² | 1 | BLP | 474.2 | 5.3 | 15.5 | 452.5 | 1 | 21.4 |
| torus: adaptive IQUV, base 384, block 16 | 1 | reference | 3254.4 | 2.7 | 2.0 | 3240.6 | 28 | 13.1 |
| torus: adaptive IQUV, base 384, block 16 | 1 | BLP | 2683.4 | 6.2 | 15.4 | 2612.6 | 3 | 69.9 |

## Every configuration

| leg | backend | devices / threads | package | config | wall [s] | rate | first [s] | load | other cores | same as first config | job |
|---|---|--:|---|---|--:|--:|--:|---|--:|---|---|
| fused I 384², uniform resample | gpu | 1 | reference | chunk131072 | 0.094 | 1.57e+06 | 0.0883 | 7/7 | 5.2 | bitwise | 2998 |
| fused I 384², uniform resample | gpu | 1 | reference | default | 0.102 | 1.45e+06 | 13.1* | 6/7 | 5.6 | first | 2998 |
| fused I 384², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.0842 | 1.75e+06 | 0.14 | 9/9 | 63.0 | bitwise | 3023 |
| fused I 384², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.098 | 1.51e+06 | 0.276 | 9/9 | 9.1 | bitwise | 3023 |
| fused I 384², uniform resample | gpu | 1 | BLP | lanes2_chunkauto | 0.0781 | 1.89e+06 | 14.0* | 8/8 | 4.9 | bitwise | 3023 |
| fused I 768², uniform resample | gpu | 1 | reference | chunk131072 | 0.293 | 2.02e+06 | 0.293 | 7/7 | 5.5 | bitwise | 2998 |
| fused I 768², uniform resample | gpu | 1 | reference | default | 0.334 | 1.76e+06 | 0.344 | 7/7 | 5.8 | first | 2998 |
| fused I 768², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.287 | 2.05e+06 | 0.4 | 9/9 | 26.0 | bitwise | 3023 |
| fused I 768², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.341 | 1.73e+06 | 0.431 | 9/9 | 5.9 | bitwise | 3023 |
| fused I 768², uniform resample | gpu | 1 | BLP | lanes2_chunkauto | 0.289 | 2.04e+06 | 0.334 | 9/9 | 29.0 | bitwise | 3023 |
| fused I 768², mixed precision | gpu | 1 | reference | chunk131072 | 0.246 | 2.40e+06 | 0.324 | 7/7 | 5.8 | bitwise | 2998 |
| fused I 768², mixed precision | gpu | 1 | reference | default | 0.256 | 2.30e+06 | 2.58* | 7/7 | 5.8 | first | 2998 |
| fused I 768², mixed precision | gpu | 1 | BLP | lanes2_chunk131072 | 0.281 | 2.10e+06 | 0.342 | 9/9 | 6.0 | bitwise | 3023 |
| fused I 768², mixed precision | gpu | 1 | BLP | lanes2_chunk65536 | 0.303 | 1.94e+06 | 0.411 | 9/9 | 4.7 | bitwise | 3023 |
| fused I 768², mixed precision | gpu | 1 | BLP | lanes2_chunkauto | 0.278 | 2.12e+06 | 2.66* | 9/9 | 6.3 | bitwise | 3023 |
| fused IQUV 256², uniform resample | gpu | 1 | reference | chunk131072 | 0.243 | 2.70e+05 | 0.244 | 7/7 | 6.1 | bitwise | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | reference | default | 0.244 | 2.68e+05 | 16.2* | 7/7 | 5.4 | first | 2998 |
| fused IQUV 256², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.211 | 3.11e+05 | 0.27 | 14/14 | 24.0 | bitwise | 3023 |
| fused IQUV 256², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.215 | 3.06e+05 | 0.275 | 14/14 | 39.0 | bitwise | 3023 |
| fused IQUV 256², uniform resample | gpu | 1 | BLP | lanes2_chunkauto | 0.211 | 3.10e+05 | 6.89* | 9/14 | 15.0 | bitwise | 3023 |
| fused IQUV 384², uniform resample | gpu | 1 | reference | chunk131072 | 0.422 | 3.50e+05 | 0.428 | 7/7 | 5.4 | bitwise | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | reference | default | 0.414 | 3.56e+05 | 0.441 | 7/7 | 5.4 | first | 2998 |
| fused IQUV 384², uniform resample | gpu | 1 | BLP | lanes2_chunk131072 | 0.366 | 4.03e+05 | 0.414 | 13/13 | 9.5 | bitwise | 3023 |
| fused IQUV 384², uniform resample | gpu | 1 | BLP | lanes2_chunk65536 | 0.357 | 4.13e+05 | 0.48 | 13/13 | 4.4 | bitwise | 3023 |
| fused IQUV 384², uniform resample | gpu | 1 | BLP | lanes2_chunkauto | 0.339 | 4.35e+05 | 0.389 | 13/13 | 9.3 | bitwise | 3023 |
| native blocks I 384² | gpu | 1 | reference | chunk131072 | 0.133 | 1.11e+06 | 0.13 | 7/7 | 5.2 | bitwise | 2998 |
| native blocks I 384² | gpu | 1 | reference | default | 0.138 | 1.07e+06 | 12.9* | 7/7 | 5.4 | first | 2998 |
| native blocks I 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.129 | 1.15e+06 | 0.163 | 9/9 | 2.3 | bitwise | 3023 |
| native blocks I 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.139 | 1.06e+06 | 0.234 | 9/9 | 3.5 | bitwise | 3023 |
| native blocks I 384² | gpu | 1 | BLP | lanes2_chunkauto | 0.125 | 1.18e+06 | 14.2* | 11/9 | 2.4 | bitwise | 3023 |
| native blocks IQUV 384² | gpu | 1 | reference | chunk131072 | 0.513 | 2.87e+05 | 0.543 | 7/7 | 5.4 | bitwise | 2998 |
| native blocks IQUV 384² | gpu | 1 | reference | default | 0.622 | 2.37e+05 | 5.62* | 7/7 | 5.4 | first | 2998 |
| native blocks IQUV 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.62 | 2.38e+05 | 0.66 | 9/9 | 3.6 | bitwise | 3023 |
| native blocks IQUV 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.698 | 2.11e+05 | 0.739 | 8/9 | 6.2 | bitwise | 3023 |
| native blocks IQUV 384² | gpu | 1 | BLP | lanes2_chunkauto | 0.568 | 2.60e+05 | 5.83* | 9/8 | 3.1 | bitwise | 3023 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | reference | chunk131072 | 4.55 | 2.51e+05 | setup 32.6, 4.51 | 9/9 | 5.3 | bitwise | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | reference | default | 5.32 | 2.14e+05 | setup 32.6, 5.39 | 10/9 | 5.3 | first | 2998 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | BLP | lanes2_chunk131072 | 5.07 | 2.25e+05 | 5.13 | 8/8 | 7.2 | bitwise | 3023 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | BLP | lanes2_chunk65536 | 5.94 | 1.92e+05 | 6.1 | 8/8 | 6.8 | bitwise | 3023 |
| adaptive IQUV, base 384, block 8 | gpu | 1 | BLP | lanes2_chunkauto | 4.59 | 2.49e+05 | 9.64* | 9/8 | 4.9 | bitwise | 3023 |
| exact-Kerr g-disk 1024² | gpu | 1 | reference | chunk131072 | 0.116 | 9.05e+06 | 0.306 | 13/12 | 1.7 | bitwise | 3029 |
| exact-Kerr g-disk 1024² | gpu | 1 | reference | default | 0.0868 | 1.21e+07 | 15.6* | 17/13 | 2.4 | bitwise | 3029 |
| exact-Kerr g-disk 1024² | gpu | 1 | BLP | lanes2_chunk131072 | 0.0832 | 1.26e+07 | 0.0904 | 9/8 | 2.5 | bitwise | 3029 |
| exact-Kerr g-disk 1024² | gpu | 1 | BLP | lanes2_chunk65536 | 0.106 | 9.90e+06 | 0.103 | 9/9 | 2.5 | bitwise | 3029 |
| exact-Kerr g-disk 1024² | gpu | 1 | BLP | lanes2_chunkauto | 0.065 | 1.61e+07 | 17.1* | 11/9 | 17.0 | bitwise | 3029 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | chunk131072 | 0.613 | 1.64e+07 | 0.585 | 10/10 | 5.2 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | chunk65536 | 0.841 | 1.19e+07 | 1.01 | 10/10 | 6.6 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | reference | default | 0.41 | 2.45e+07 | 12.7* | 11/10 | 7.3 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk1048576 | 0.362 | 2.77e+07 | 0.374 | 8/8 | 5.2 | bitwise | 2999 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk131072 | 0.366 | 2.74e+07 | 0.352 | 7/7 | 3.4 | bitwise | 3023 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk65536 | 0.363 | 2.76e+07 | 0.363 | 7/7 | 4.0 | bitwise | 3023 |
| MC thin sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunkauto | 0.357 | 2.81e+07 | 5.27* | 7/7 | 3.4 | bitwise | 3023 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | chunk131072 | 1.17 | 1.01e+07 | 1.13 | 10/9 | 5.3 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | chunk65536 | 2.21 | 5.37e+06 | 2.57 | 10/10 | 5.3 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | reference | default | 0.752 | 1.58e+07 | 9.44* | 10/10 | 5.5 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk1048576 | 1.39 | 8.55e+06 | 1.42 | 8/8 | 5.4 | bitwise | 2999 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk131072 | 0.719 | 1.65e+07 | 0.721 | 7/7 | 3.5 | bitwise | 3023 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunk65536 | 0.704 | 1.68e+07 | 0.74 | 7/7 | 3.4 | bitwise | 3023 |
| MC scattering sphere, 10⁷ | gpu | 1 | BLP | lanes2_chunkauto | 0.742 | 1.60e+07 | 9.91* | 7/7 | 3.6 | bitwise | 3023 |
| binary 00455 I 480² | gpu | 1 | reference | chunk131072 | 1.45 | 1.59e+05 | 1.47 | 7/7 | 5.3 | bitwise | 3000 |
| binary 00455 I 480² | gpu | 1 | reference | default | 1.39 | 1.66e+05 | 14.9* | 7/7 | 5.3 | first | 3000 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 1.55 | 1.48e+05 | 2.02 | 7/7 | 5.3 | bitwise | 3000 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk131072 | 1.78 | 1.29e+05 | 2.45 | 5/5 | 1.6 | bitwise | 3026 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunk65536 | 1.91 | 1.20e+05 | 3.23 | 4/5 | 1.8 | bitwise | 3026 |
| binary 00455 I 480² | gpu | 1 | BLP | lanes2_chunkauto | 1.55 | 1.48e+05 | 16.3* | 6/5 | 1.9 | bitwise | 3026 |
| binary 00455 IQUV 480² | gpu | 1 | reference | chunk131072 | 3.26 | 7.07e+04 | 3.3 | 7/7 | 18.0 | bitwise | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | reference | default | 3.76 | 6.13e+04 | 10.3* | 7/7 | 5.4 | first | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 2.97 | 7.76e+04 | 3.64 | 19/17 | 15.0 | bitwise | 3000 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk131072 | 3.44 | 6.71e+04 | 4.06 | 4/4 | 2.3 | bitwise | 3026 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunk65536 | 3.08 | 7.49e+04 | 4.88 | 4/4 | 1.6 | bitwise | 3026 |
| binary 00455 IQUV 480² | gpu | 1 | BLP | lanes2_chunkauto | 2.93 | 7.87e+04 | 9.45* | 5/4 | 1.7 | bitwise | 3026 |
| binary 00122 IQUV 480² | gpu | 1 | reference | chunk131072 | 2.84 | 8.10e+04 | 2.92 | 10/10 | 5.5 | bitwise | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | reference | default | 3.18 | 7.23e+04 | 19.9* | 14/11 | 5.6 | first | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk1048576 | 2.79 | 8.27e+04 | 3.22 | 7/7 | 5.5 | bitwise | 3000 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk131072 | 2.88 | 7.99e+04 | 3.56 | 3/3 | 1.8 | bitwise | 3026 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunk65536 | 3.31 | 6.96e+04 | 4.11 | 3/3 | 1.9 | bitwise | 3026 |
| binary 00122 IQUV 480² | gpu | 1 | BLP | lanes2_chunkauto | 2.77 | 8.32e+04 | 3.23 | 4/4 | 1.9 | bitwise | 3026 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | reference | chunk131072 | 14.0 | 4.11e+04 | setup 59.9, 14.1 | 7/7 | 6.1 | bitwise | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | reference | default | 16.0 | 3.59e+04 | setup 59.9, 16.1 | 7/7 | 6.0 | first | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk1048576 | 13.9 | 4.13e+04 | 15.0 | 7/7 | 6.3 | bitwise | 3000 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk131072 | 14.3 | 4.02e+04 | 16.2 | 4/3 | 2.0 | bitwise | 3026 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunk65536 | 16.2 | 3.54e+04 | 18.5 | 3/4 | 2.6 | bitwise | 3026 |
| binary 00455 i = 90° adaptive IQUV | gpu | 1 | BLP | lanes2_chunkauto | 13.9 | 4.13e+04 | 19.8* | 3/3 | 1.9 | bitwise | 3026 |
| torus: blocks I 192² | gpu | 1 | reference | chunk131072 | 0.0582 | 6.34e+05 | 0.0578 | 10/10 | 5.2 | bitwise | 2998 |
| torus: blocks I 192² | gpu | 1 | reference | default | 0.0568 | 6.49e+05 | 13.3* | 10/10 | 6.3 | first | 2998 |
| torus: blocks I 192² | gpu | 1 | BLP | lanes2_chunk131072 | 0.0563 | 6.55e+05 | 0.0932 | 6/6 | 3.6 | bitwise | 3023 |
| torus: blocks I 192² | gpu | 1 | BLP | lanes2_chunk65536 | 0.0562 | 6.56e+05 | 0.156 | 6/6 | 3.1 | bitwise | 3023 |
| torus: blocks I 192² | gpu | 1 | BLP | lanes2_chunkauto | 0.0567 | 6.50e+05 | 13.5* | 6/6 | 4.2 | bitwise | 3023 |
| torus: blocks I 384² | gpu | 1 | reference | chunk131072 | 0.143 | 1.03e+06 | 0.148 | 9/9 | 5.3 | bitwise | 2998 |
| torus: blocks I 384² | gpu | 1 | reference | default | 0.148 | 9.96e+05 | 0.178 | 9/9 | 5.4 | first | 2998 |
| torus: blocks I 384² | gpu | 1 | BLP | lanes2_chunk131072 | 0.136 | 1.08e+06 | 0.175 | 6/6 | 3.3 | bitwise | 3023 |
| torus: blocks I 384² | gpu | 1 | BLP | lanes2_chunk65536 | 0.148 | 9.99e+05 | 0.186 | 6/6 | 3.2 | bitwise | 3023 |
| torus: blocks I 384² | gpu | 1 | BLP | lanes2_chunkauto | 0.129 | 1.14e+06 | 0.205 | 6/6 | 3.2 | bitwise | 3023 |
| torus: blocks I 768² | gpu | 1 | reference | chunk131072 | 0.491 | 1.20e+06 | 0.501 | 9/9 | 6.1 | bitwise | 2998 |
| torus: blocks I 768² | gpu | 1 | reference | default | 0.553 | 1.07e+06 | 0.587 | 9/9 | 5.3 | first | 2998 |
| torus: blocks I 768² | gpu | 1 | BLP | lanes2_chunk131072 | 0.486 | 1.21e+06 | 0.527 | 6/6 | 3.8 | bitwise | 3023 |
| torus: blocks I 768² | gpu | 1 | BLP | lanes2_chunk65536 | 0.547 | 1.08e+06 | 0.623 | 6/6 | 3.5 | bitwise | 3023 |
| torus: blocks I 768² | gpu | 1 | BLP | lanes2_chunkauto | 0.472 | 1.25e+06 | 0.519 | 6/6 | 3.6 | bitwise | 3023 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | reference | chunk131072 | 2.71 | 2.33e+05 | 2.74 | 8/7 | 5.4 | bitwise | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | reference | default | 3.29 | 1.91e+05 | 19.3* | 8/8 | 15.0 | first | 2998 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | BLP | lanes2_chunk131072 | 3.05 | 2.06e+05 | 3.19 | 18/15 | 5.6 | bitwise | 3023 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | BLP | lanes2_chunk65536 | 3.64 | 1.73e+05 | 3.82 | 21/18 | 4.3 | bitwise | 3023 |
| torus: adaptive IQUV, base 384, block 16 | gpu | 1 | BLP | lanes2_chunkauto | 2.74 | 2.30e+05 | 9.74* | 6/23 | 85.0 | bitwise | 3023 |
| native blocks I 384² | cpu | 16 | reference | threads | 8.68 | 1.70e+04 | 15.4* | 6/9 | 3.8 | first | none |
| native blocks I 384² | cpu | 16 | BLP | chunk4096 | 5.28 | 2.79e+04 | 5.33 | 8/10 | 3.1 | bitwise | none |
| native blocks I 384² | cpu | 16 | BLP | chunk64 | 4.2 | 3.51e+04 | 4.23 | 10/13 | 2.7 | bitwise | none |
| native blocks I 384² | cpu | 16 | BLP | chunkauto | 4.27 | 3.45e+04 | 7.63* | 3/8 | 2.4 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | reference | threads | 17.0 | 8677 | 31.9* | 13/15 | 4.8 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | BLP | chunk4096 | 12.9 | 1.15e+04 | 14.1 | 27/25 | 5.0 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | BLP | chunk64 | 10.7 | 1.38e+04 | 10.3 | 25/38 | 20.0 | bitwise | none |
| native blocks IQUV 384² | cpu | 16 | BLP | chunkauto | 10.7 | 1.37e+04 | 14.3* | 13/17 | 3.6 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | reference | threads | 123 | 9276 | 142* | 14/16 | 2.8 | first | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | BLP | chunk4096 | 77.7 | 1.47e+04 | 78.2 | 20/23 | 5.4 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | BLP | chunk64 | 77.1 | 1.48e+04 | 78.5 | 23/23 | 9.7 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 16 | BLP | chunkauto | 76.4 | 1.49e+04 | 81.4* | 10/20 | 3.4 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 16 | reference | threads | 1.28 | 8.18e+05 | 2.5* | 11/12 | 2.5 | first | none |
| exact-Kerr g-disk 1024² | cpu | 16 | BLP | chunk4096 | 0.968 | 1.08e+06 | 1.96* | 10/11 | 2.6 | first | none |
| exact-Kerr g-disk 1024² | cpu | 16 | BLP | chunk64 | 0.659 | 1.59e+06 | 0.765 | 13/13 | 4.4 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 16 | BLP | chunkauto | 0.731 | 1.43e+06 | 2.53* | 14/14 | 3.0 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 16 | reference | threads | 14.3 | 7.00e+05 | 16.0* | 10/15 | 2.6 | first | none |
| MC thin sphere, 10⁷ | cpu | 16 | BLP | chunk4096 | 14.5 | 6.89e+05 | 14.8 | 24/38 | 27.0 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 16 | BLP | chunk64 | 14.6 | 6.89e+05 | 15.3 | 38/31 | 7.4 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 16 | BLP | chunkauto | 14.7 | 6.82e+05 | 19.7* | 12/24 | 11.0 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 16 | reference | threads | 15.1 | 7.85e+05 | 17.7* | 10/15 | 2.5 | first | none |
| MC scattering sphere, 10⁷ | cpu | 16 | BLP | chunk4096 | 15.1 | 7.83e+05 | 15.1 | 17/19 | 3.9 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 16 | BLP | chunk64 | 14.9 | 7.96e+05 | 15.1 | 19/19 | 3.0 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 16 | BLP | chunkauto | 15.1 | 7.86e+05 | 20.9* | 13/17 | 3.8 | bitwise | none |
| native blocks I 384² | cpu | 64 | reference | threads | 5.0 | 2.95e+04 | 24.3* | 9/25 | 3.2 | bitwise | none |
| native blocks I 384² | cpu | 64 | BLP | chunk4096 | 2.76 | 5.35e+04 | 2.64 | 19/20 | 3.0 | bitwise | none |
| native blocks I 384² | cpu | 64 | BLP | chunk64 | 1.19 | 1.23e+05 | 1.31 | 20/24 | 2.6 | bitwise | none |
| native blocks I 384² | cpu | 64 | BLP | chunkauto | 1.46 | 1.01e+05 | 5.26* | 11/20 | 4.1 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | reference | threads | 8.17 | 1.80e+04 | 20.7* | 10/38 | 5.5 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | BLP | chunk4096 | 5.95 | 2.48e+04 | 6.27 | 29/31 | 2.7 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | BLP | chunk64 | 2.95 | 5.00e+04 | 3.04 | 31/36 | 3.0 | bitwise | none |
| native blocks IQUV 384² | cpu | 64 | BLP | chunkauto | 3.0 | 4.91e+04 | 7.53* | 14/29 | 2.9 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | reference | threads | 45.8 | 2.49e+04 | 150* | 13/56 | 3.6 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | BLP | chunk4096 | 26.9 | 4.25e+04 | 26.7 | 49/55 | 5.7 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | BLP | chunk64 | 21.8 | 5.23e+04 | 22.4 | 55/64 | 4.5 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 64 | BLP | chunkauto | 21.7 | 5.26e+04 | 26.3* | 7/53 | 2.3 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | reference | threads | 0.743 | 1.41e+06 | 1.63* | 11/13 | 2.7 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | BLP | chunk4096 | 0.244 | 4.30e+06 | 0.267 | 11/11 | 4.8 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | BLP | chunk64 | 0.229 | 4.57e+06 | 0.229 | 11/11 | 4.3 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 64 | BLP | chunkauto | 0.271 | 3.88e+06 | 1.96* | 11/11 | 4.2 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | reference | threads | 5.63 | 1.78e+06 | 7.94* | 9/26 | 2.3 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | BLP | chunk4096 | 4.14 | 2.42e+06 | 4.4 | 27/38 | 3.9 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | BLP | chunk64 | 4.14 | 2.42e+06 | 4.12 | 38/45 | 3.8 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 64 | BLP | chunkauto | 4.45 | 2.25e+06 | 8.92* | 10/27 | 5.1 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | reference | threads | 5.96 | 1.99e+06 | 8.44* | 10/26 | 2.6 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | BLP | chunk4096 | 4.4 | 2.69e+06 | 4.42 | 21/31 | 2.4 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | BLP | chunk64 | 4.36 | 2.72e+06 | 4.36 | 31/41 | 6.0 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 64 | BLP | chunkauto | 4.35 | 2.73e+06 | 10.9* | 11/21 | 2.7 | bitwise | none |
| native blocks I 384² | cpu | 128 | reference | threads | 4.8 | 3.07e+04 | 22.8* | 12/63 | 4.0 | bitwise | none |
| native blocks I 384² | cpu | 128 | BLP | chunk4096 | 2.72 | 5.41e+04 | 2.72 | 18/18 | 1.6 | bitwise | none |
| native blocks I 384² | cpu | 128 | BLP | chunk64 | 0.827 | 1.78e+05 | 0.972 | 18/18 | 1.6 | bitwise | none |
| native blocks I 384² | cpu | 128 | BLP | chunkauto | 0.842 | 1.75e+05 | 4.34* | 10/18 | 1.4 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | reference | threads | 6.71 | 2.20e+04 | 27.2* | 10/51 | 3.1 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | BLP | chunk4096 | 6.12 | 2.41e+04 | 6.23 | 36/34 | 1.7 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | BLP | chunk64 | 2.54 | 5.80e+04 | 2.57 | 34/49 | 2.5 | bitwise | none |
| native blocks IQUV 384² | cpu | 128 | BLP | chunkauto | 2.53 | 5.82e+04 | 7.37* | 10/36 | 1.3 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | reference | threads | 42.1 | 2.71e+04 | 112* | 11/86 | 3.4 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | BLP | chunk4096 | 23.6 | 4.83e+04 | 25.5 | 96/82 | 1.9 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | BLP | chunk64 | 18.7 | 6.10e+04 | 18.9 | 82/116 | 1.5 | bitwise | none |
| adaptive IQUV, base 384, block 8 | cpu | 128 | BLP | chunkauto | 18.6 | 6.14e+04 | 23.6* | 10/96 | 2.1 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | reference | threads | 0.451 | 2.33e+06 | 1.17* | 19/19 | 2.4 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | BLP | chunk4096 | 0.266 | 3.94e+06 | 0.242 | 10/19 | 3.7 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | BLP | chunk64 | 0.189 | 5.54e+06 | 0.257 | 19/19 | 2.0 | bitwise | none |
| exact-Kerr g-disk 1024² | cpu | 128 | BLP | chunkauto | 0.229 | 4.59e+06 | 2.31* | 10/10 | 2.4 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | reference | threads | 3.55 | 2.82e+06 | 4.99* | 18/42 | 2.2 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | BLP | chunk4096 | 3.26 | 3.08e+06 | 3.23 | 38/58 | 3.4 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | BLP | chunk64 | 3.22 | 3.11e+06 | 3.23 | 58/69 | 1.2 | bitwise | none |
| MC thin sphere, 10⁷ | cpu | 128 | BLP | chunkauto | 3.22 | 3.12e+06 | 7.02* | 12/38 | 1.2 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | reference | threads | 3.81 | 3.11e+06 | 6.11* | 13/34 | 2.4 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | BLP | chunk4096 | 3.54 | 3.34e+06 | 3.44 | 43/62 | 1.7 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | BLP | chunk64 | 3.51 | 3.38e+06 | 3.59 | 62/77 | 1.3 | bitwise | none |
| MC scattering sphere, 10⁷ | cpu | 128 | BLP | chunkauto | 3.49 | 3.40e+06 | 10.2* | 10/43 | 1.3 | bitwise | none |
| torus: blocks I 192² | cpu | 128 | reference | threads | 1.75 | 2.10e+04 | 9.6* | 9/16 | 3.4 | first | none |
| torus: blocks I 192² | cpu | 128 | BLP | chunk4096 | 2.71 | 1.36e+04 | 2.67 | 19/18 | 1.6 | bitwise | none |
| torus: blocks I 192² | cpu | 128 | BLP | chunk64 | 0.28 | 1.32e+05 | 0.34 | 18/18 | 1.0 | bitwise | none |
| torus: blocks I 192² | cpu | 128 | BLP | chunkauto | 0.266 | 1.38e+05 | 3.94* | 11/21 | 1.6 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | reference | threads | 4.77 | 3.09e+04 | 18.8* | 12/58 | 5.9 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | BLP | chunk4096 | 2.95 | 4.99e+04 | 2.77 | 19/21 | 1.4 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | BLP | chunk64 | 0.88 | 1.68e+05 | 1.01 | 21/29 | 1.3 | bitwise | none |
| torus: blocks I 384² | cpu | 128 | BLP | chunkauto | 0.902 | 1.63e+05 | 4.44* | 9/19 | 1.2 | bitwise | none |
| torus: blocks I 768² | cpu | 128 | reference | threads | 11.6 | 5.08e+04 | 74.5* | 10/99 | 4.9 | first | none |
| torus: blocks I 768² | cpu | 128 | BLP | chunk4096 | 4.56 | 1.29e+05 | 4.7 | 42/37 | 1.5 | bitwise | none |
| torus: blocks I 768² | cpu | 128 | BLP | chunk64 | 3.38 | 1.74e+05 | 3.38 | 37/51 | 1.7 | bitwise | none |
| torus: blocks I 768² | cpu | 128 | BLP | chunkauto | 3.42 | 1.73e+05 | 7.26* | 10/42 | 1.4 | bitwise | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | reference | threads | 28.8 | 2.19e+04 | 72.9* | 11/82 | 4.2 | first | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | BLP | chunk4096 | 19.4 | 3.24e+04 | 19.3 | 73/49 | 2.7 | bitwise | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | BLP | chunk64 | 11.0 | 5.74e+04 | 11.2 | 49/92 | 3.1 | bitwise | none |
| torus: adaptive IQUV, base 384, block 16 | cpu | 128 | BLP | chunkauto | 11.0 | 5.72e+04 | 15.6* | 18/73 | 1.8 | bitwise | none |

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
| 3023 | gpu1_765ddd02 | 1 | 2026-10-18 02:55:32 | 2026-10-18 02:58:02 | 2026-10-18 03:05:02 | COMPLETED | 0 |
| 3026 | gpu1_dc5503fc | 1 | 2026-10-18 03:05:02 | 2026-10-18 03:10:33 | 2026-10-18 03:17:33 | COMPLETED | 0 |

Items (`tag`: `<devices>@<deck>=<legs>/…`, see `slurm/run_reference.sh`):

- `gpu1_f87f96d2`: `1@reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384/reference=amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384`
- `gpu2_f87f96d2`: `2@reference=fused_384,fused_768,fused_mixed_768/reference=fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384/reference=amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768/torus=amr_384`
- `gpu1_efa03062`: `1@reference=mc_thin,mc_scatter`
- `gpu1_e41829b9`: `1@reference=bbh455_480,bbh455_pol_480/reference=bbh122_pol_480/reference=bbh_i90_amr_pol`
- `gpu1_765ddd02`: `1@reference=fused_384,fused_768,fused_mixed_768,fused_pol_256,fused_pol_384/reference=amr_384,amr_pol_384,amr_driver_pol/reference=analytic_1024,mc_thin,mc_scatter/torus=img_192,img_384,img_768,amr_384`
- `gpu1_dc5503fc`: `1@reference=bbh455_480,bbh455_pol_480,bbh122_pol_480,bbh_i90_amr_pol`
