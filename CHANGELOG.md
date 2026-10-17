# Changelog

All notable changes to BlackLightPlus are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on its exported API. While the version is `0.x`, a bump of `x` may break that
API. Anything not exported is internal and may change in any release.

Locations written `<sha>^:<path>:<line>` are at the parent of the fixing commit,
i.e. the code as it was when the defect was live; plain `<path>:<line>` is at the
current tree. `results/blp/docs_check.py` checks that every cited commit, file
and line exists.

## [Unreleased]

GPU throughput of the code in this section has not been measured: every
change is verified on the KernelAbstractions CPU backend and by the offline
kernel audit, and the GPU numbers in `docs/src/benchmarks.md` describe v0.1.0.
The pending acceptance runs are listed there.

### Added

- **Device pool.** Every GPU driver runs on a persistent `DevicePool` (two
  worker tasks per device, each with its own stream and pinned staging buffers,
  a dynamic chunk queue, first-error propagation); `device_pool`,
  `default_pool`, `HostDevice` (KernelAbstractions CPU backend as a logical
  device), `foreach_device`, `foreach_chunk`, `enqueue!`, `march!`,
  `upload_per_device`. A snapshot or grid is uploaded once per pool and reused
  (`bd26470`, `fd1e9a0`).
- **Device camera rays.** `camera_ray(st, CameraFrame, α, β)` is the single
  `isbits` pixel-to-ray map of the host `camera_rays` and the device ray
  kernel; a chunk uploads only its pixel or block indices (`CameraRays`,
  `PixelGrid`, `BlockGrid`, `pixel_centres`, `null_covariant`; `0610b31`).
- **Cost-ordered queue.** `cost_order(a, inc, fov, npix; tile = 16)` queues
  16×16 tiles by predicted cost (distance to the critical curve), the default
  for full-frame Kerr drivers; `:mixed` precision queues its `Float64` rescue
  pixels first in the same queue (`41743a1`).
- **Streaming adaptive refinement.** `AMRStokes` marches `render_amr` blocks on
  the device pool; `stream_blocks!` paints, tests and queues children as each
  block returns, with host painting and refinement tests threaded
  (`march_chunk!`; `4f5f1e3`).
- **Monte Carlo on the pool.** `mc_spectrum_gpu(...; chunk = 2^20)` runs
  global-index chunks as pool work items; host tables are built once per call
  and uploaded concurrently (`c4895d2`).
- **Ancestry-keyed Monte Carlo streams and exact accumulation.**
  `MCStreamLayout`, `mc_primary_rng`, `mc_child_rng`, `mc_stream_id`,
  `mc_stream_layer`; `ExactSpectrum`, `xsum_add!`, `xsum_round` (`b15367f`).
- **Shared reverse marches.** `reverse_unpolarized_march` and
  `reverse_polarized_march` under the march controls `KerrMarch`,
  `SpacetimeMarch` and `StoredPath`, used by the CPU drivers and every fused
  kernel; `fluid_tetrad_checked`; `Geodesics.spacetime_control`,
  `Geodesics.adm_control` (`6031897`).
- **Single physics source.** `coefficient_params`, `has_power_law`,
  `has_kappa`, `unpolarized_coefficients`, `stokes_segment`,
  `unpolarized_segment`, `stokes_admissible` (`35ebb96`);
  `Geodesics.metric_and_inverse`, `Geodesics.raise_and_norm`,
  `Geodesics.geodesic_transport_rhs` and closed-form binary and ADM inverses
  (`e253c10`).
- **One march termination rule.** `Geodesics.march_stop`,
  `march_captured`, `march_escaped`, `ks_radius_energy`,
  `normal_frame_energy`, `CAPTURE_REDSHIFT` (`bb233b2`, `47527a4`).
- **AthenaK `.athdf` dumps with `press` (and `temperature`) primitives**, read
  with `AthenaKSnapshot(path; gamma)` (`c2f615b`).
- **Hashed leaf lookup** for nested meshes whose finest-level table exceeds
  the dense-table cap (`ce50e4c`).
- **Benchmarks and audits.** `bench/kernel_audit.jl` (offline `sm_80`
  register, stack and spill audit with a `--check` regression gate;
  `3b09c6f`), `bench/bench_multigpu.jl` and `bench/bench_lines.jl`
  (phase-timed one- and two-device legs; `43bffa1`), `bench/bench_amr_host.jl`.
- **Figure scripts.** `figures/fig_bbh_i90_pol_amr.jl` (edge-on polarized
  adaptive image of a superposed binary; `3736b32`) and
  `figures/fig_eht_bbh.jl` with `plot_eht_bbh.py` (230 GHz EHT-style mocks of
  binary GRMHD snapshots; `7c388c5`, `ee88885`).
- **Documentation.** [Validation against RAPTOR](docs/src/validation.md);
  method sections on conventions, march termination, admissibility, the single
  physics source and multi-device execution; known input issues of the AthenaK
  binary runs (`docs/src/guide/grmhd.md`).
- **Tests.** `test/test_ray_orientation.jl`, `test/test_bbh_superposed.jl`,
  `test/test_march_termination.jl`, `test/test_emission_admissibility.jl`,
  `test/test_athdf_press_layout.jl`, `test/test_leaf_hash.jl`,
  `test/test_device_pool.jl`, `test/test_device_rays.jl`,
  `test/test_device_schedule.jl`, `test/test_device_amr.jl`,
  `test/test_device_invariance.jl`, `test/test_mc_exact.jl`,
  `test/test_bench_lines.jl`. They are not yet included by `test/runtests.jl`;
  the include lines are in `docs/src/contributing.md`.

### Changed

- **Polarized transfer scheme on the CPU.** The CPU polarized march is the
  transported-camera-frame reverse Mueller chain that the kernels run, in place
  of the Strang-split coherency transport. Both are first order in the step
  density; on a Kerr torus deck the frame scheme's Stokes $L_2$ error is 30%
  smaller and its EVPA error five to eight times smaller ($C=80$:
  $3.75\times10^{-3}$ / 0.008° against $5.40\times10^{-3}$ / 0.065°), and its
  $I$ departs from the unpolarized march by $1.9\times10^{-7}$ instead of
  $1.1\times10^{-4}$ at $C=1280$. CPU polarized images change at the percent
  level near the photon ring (pol_hires: $\sum I$ −0.30%, max $|\Delta I|/I_{\max}$
  1.8%, EVPA p99 1.5°) (`6031897`).
- **GPU kernels call the CPU physics.** Metric, geodesic step, frame transport,
  plasma decode, coefficients, fluid tetrad and segment operators are the CPU
  functions; kernel images move by at most $2.6\times10^{-12}$ of the maximum
  (Stokes $I$) and $4\times10^{-11}$ (IQUV, binary); CPU Stokes $I$ by at most
  $1.1\times10^{-7}$ relative (pitch angle from the invariant instead of the
  tetrad) (`35ebb96`). Kernel registers: fused 240 → 172, stack 2208 → 32 B
  (`e253c10`, `35ebb96`).
- **Coefficient functions** take a `SimCoeffConfig` or its `isbits`
  `coefficient_params(scc)` bundle (`35ebb96`).
- **`devices` keyword** of every GPU driver: default `default_pool()` instead
  of `select_devices()`; accepts device ordinals, `HostDevice`s or a
  `DevicePool`; `order` is the queue order (`bd26470`, `fd1e9a0`).
- **`march_sort_order(counts)`** returns the longest trip count first; it was
  ascending, which put the slowest rays at the end of the render (`41743a1`).
- **`mc_spectrum_gpu`**: `n_gen_override` removed; the scattering method takes
  `min_capacity = 4096` (initial child-buffer size; a layer that overflows is
  rerun at the exact size); both methods take `chunk` and a pool as
  `devices` (`b15367f`, `c4895d2`).
- **`ScatterWorkspace(layout::MCStreamLayout)`** (or `ScatterWorkspace()`)
  replaces `ScatterWorkspace(child_seed)`; the child weight and first-order
  weight are read with `child_weight(ws)` and `first_order_weight(ws)`.
  `MCRng` carries the stream identity in a new `sid` field (`b15367f`).
- **Monte Carlo results** are bitwise independent of thread count, device
  count and scheduling, and differ in the last bits and in the scattered
  children from v0.1.0 runs at the same seed (`b15367f`).
- **Binary and Kerr marches** stop at normal-frame photon energy
  $E_n>10^2$ in addition to the horizon backstop; the omitted observed
  intensity is bounded (method page, March termination) and the flux change on
  a plunging Kerr deck is 0.0 (`bb233b2`, `47527a4`).
- **`AthenaKSnapshot(filename; gamma = nothing)`**: `gamma` is required for
  press-layout dumps; `render` passes the configured $\gamma$ to the `.athdf`
  loader (`c2f615b`).

### Fixed

- **Device-pool staging buffers could leave freed host memory registered with
  CUDA.** A lane's pinned staging buffer was a `CUDA.pin`ned `Vector`, released
  by a GC finalizer that takes CUDA's pin lock; a buffer replaced by a larger one
  or dropped by `close(pool)` whose finalizer ran while another lane held that
  lock failed to unregister ("task switch not allowed from inside gc
  finalizer"), and a later pin of the reused memory threw
  `ERROR_HOST_MEMORY_ALREADY_REGISTERED`, aborting the render. Staging buffers
  are now page-locked allocations freed explicitly when replaced (after the
  lane's stream drains) and when the lane stops.
  Cause: `ac31867^:src/gpu_pool.jl:37` (`CUDA.pin`),
  `ac31867^:src/gpu_pool.jl:367` (replacement) and
  `ac31867^:src/gpu_pool.jl:255` (`close`). Test: `test/test_pool_pinned.jl`
  (CUDA only) (`ac31867`).
- **Camera rays were the time reverse of the received photon.** The camera
  launched the covector $(k_t,k_i)=(-1,\hat n_{\rm in})$ and integrated it
  forward: a future-directed ingoing ray, the $(t,\phi)\to(-t,-\phi)$ image of the
  physical one. Images were mirrored ($L_z/E=+\alpha\sin i$), gas approaching the
  camera was redshifted ($\nu/\nu_{\rm obs}=\gamma(1+v)$), $U$ and $V$ were
  mirrored, and slow light sampled $t_{\rm obs}-x^t$. Rays are now the received
  photon traced backward, state $p=-k$ with $p_t=+1$, and every consumer reads
  $k=-p$; the camera-to-Bardeen map is the identity.
  Cause: `706b55f^:src/geodesics/camera.jl:52` and
  `706b55f^:src/geodesics/camera.jl:55` (launch and normalization),
  `706b55f^:src/render.jl:186` (sign-flipped Bardeen map),
  `706b55f^:src/simulation/time_series.jl:204` (slow-light time).
  Fix: `706b55f`; the figure script `figures/fig_z4c_phasespace.jl` in
  `4e95701`. Test: `test/test_ray_orientation.jl:28` (17 of 64 checks passed
  before, 64 of 64 after), `test/test_ray_orientation.jl:141` (Doppler sign).
  Outputs: `figures/examples/{analytic_disk_I,torus_IQUV,pol_hires,z4c_phasespace_i30}.png`
  regenerated in `0ad8e19` (pol_hires net LP 0.10 → 0.05, net $V/I$ −0.0055 →
  +0.0042); paper figures 2–4 regenerated; `docs/src/assets/torus_IQUV.png`
  regenerated in `3a6595a`.
- **Backward rays into a superposed binary escaped instead of being
  captured.** The companion's potential moves each hole's horizon outward
  ($r_A\approx1.04$ for $m_A=0.5$ at separation 25), so a backward ray stalled at
  $r_A-r_{h,A}\approx0.04$, never met the $10^{-3}$ capture test, and diverged
  under the frozen step to $R\sim10^4$–$10^8$, where it was classed as escaped:
  every shadow ray of the $i=90^\circ$ binary deck.
  Cause: `bb233b2^:src/geodesics/spacetimes.jl:134` (only capture test) and
  `bb233b2^:src/gpu_spacetime.jl:89`. Fix: `bb233b2` (redshift capture at
  $E_n>10^2$). Test: `test/test_bbh_superposed.jl:55`,
  `test/test_bbh_superposed.jl:78` (27 tracer and 18 kernel checks failed
  before). Outputs: the $i=90^\circ$ binary figure was first rendered after the
  fix (`3736b32`).
- **CPU coefficient decode threw on round-off negative $b^2$.** The CPU
  rejected only $b^2=0$ while the device decoders rejected $b^2\le0$; a
  round-off negative $b^2$ reached `sqrt` and threw a `DomainError`.
  Cause: `80e8971^:src/radiation/integrator.jl:29`. Fix: `80e8971`.
  Test: `test/test_march_termination.jl:142`. Outputs: none (the CPU run aborted).
- **GPU and CPU polarized marches stopped on different samples.** A frame-drift
  stop added to the fused polarized kernels only (`1e3707c`, ending the march
  when $-k\cdot u_{\rm cam}$ drifts by 1%, to avoid a timelike screen leg and a
  `DomainError` in the `Float32` lane near the future horizon) ended 152 of 180
  plunging rays early on a Kerr deck ($a=0.9$, $E_n$ 124–5140 at the stop), so
  the CPU and device polarized marches integrated different sample sets (stop
  step equal on 104 of 256 rays), and polarized and unpolarized kernels
  disagreed; the horizon threshold was also duplicated as literals. All marches
  now end on one rule (horizon backstop, redshift cut $E_n>10^2$, outbound
  escape); stop steps agree on 256 of 256 rays and the transported frame keeps
  $-k\cdot e_u$ within $6\times10^{-3}$ up to the cut.
  Cause: `47527a4^:src/gpu_pol.jl:586` (`_GPOL_FRAME_TOL`, kernel-only),
  `47527a4^:src/gpu_image.jl:343` and `47527a4^:src/geodesics/gpu_tracer.jl:82`
  (literal `1e-3`). Fix: `47527a4`. Test: `test/test_march_termination.jl:104`,
  `test/test_march_termination.jl:157`, `test/test_march_termination.jl:281`,
  `test/test_march_termination.jl:296` (emission beyond the cut within its
  bound), `test/test_march_termination.jl:368` (binary). Outputs:
  `figures/out/pol_hires` re-rendered in `6a48f0f` (at `1cd4a77`) and paper
  figure 3 redrawn; `figures/examples/{torus_IQUV,bbh_i90_pol_amr}.png`
  regenerated in `3a6595a`.
- **Monte Carlo results depended on device count, thread count and
  scheduling.** Device $k$ was seeded with $\text{seed}+(k-1)2^{30}$ masked to
  32 bits, so a fifth device replayed the first device's primaries exactly
  (correlated samples, underestimated variance); scattered children were keyed
  by an atomic buffer slot on the GPU and by the chunk (thread) index on the
  CPU, so scattering spectra were not reproducible even on one GPU with a fixed
  seed; floating-point bin sums depended on summation order; an overflowing
  child buffer dropped a timing-dependent set of children.
  Cause: `b15367f^:src/montecarlo/mc_gpu.jl:32` (seed stride),
  `b15367f^:src/montecarlo/mc_gpu.jl:484` (child stream from the slot),
  `b15367f^:src/montecarlo/mc_cpu.jl:168` (child seed from the chunk),
  `b15367f^:src/montecarlo/spectrum.jl:89` (float bins).
  Fix: `b15367f` (tests flipped from broken in `b89de13`), pool chunks in
  `c4895d2`. Test: `test/test_mc_exact.jl:48`, `test/test_mc_exact.jl:87`,
  `test/test_device_invariance.jl:136`, `test/test_device_invariance.jl:160`,
  `test/test_device_invariance.jl:212`, `test/test_device_invariance.jl:186`.
  Outputs: no saved spectra; the v0.1.0 CUDA replay figure in
  `docs/src/benchmarks.md` is marked as a v0.1.0 measurement.
- **Device copies of the physics had drifted from the CPU.** The fused
  kernels applied 2 of the 14 plasma cuts (only $\sigma_{\max}$ and
  $\Theta_{e,\min}$), their polarized segment operator lacked the
  $\alpha_P\le\alpha_I$ rescale (so a non-admissible sample could amplify
  polarization as $e^{(\alpha_P-\alpha_I)\,ds}$), and the stiffness telemetry
  kernel built its metric from the march spin instead of the plasma spin. The
  device copies are deleted; kernels call the CPU functions.
  Cause: `35ebb96^:src/gpu_image.jl:273` (cuts),
  `35ebb96^:src/gpu_pol.jl:425` (segment operator without rescale),
  `35ebb96^:src/gpu_pol.jl:922` (telemetry spin). Fix: `35ebb96`.
  Test: `test/test_transfer.jl:490`, `test/test_transfer.jl:336`,
  `test/test_transfer.jl:409` (kernel against CPU march).
  Outputs: GPU images using cuts other than $\sigma$ and $\Theta_e$, or with
  $\alpha_P>\alpha_I$ samples; no saved GPU image is affected (the figures are
  CPU renders).
- **Coarse-step polarized marches ended in a `DomainError` (CPU) or NaN
  (CUDA).** At coarse steps $k$ drifts off the null cone and the fluid tetrad's
  screen leg has no spacelike solution; the march now stops there
  (`fluid_tetrad_checked`).
  Cause: `6031897^:src/radiation/tetrads.jl:45`. Fix: `6031897`.
  Test: `test/test_transfer.jl:409`.
- **Non-admissible polarized emission made images depend on the order of the
  emitters.** The thermal fits of Marszewski et al. (2021) give $|j_P|>j_I$ at
  small pitch angles (17% of a coefficient grid, up to 8×). The reverse Mueller
  chain clamps $|P|\le I$ on partial sums, so a thermal slab with a
  field-aligned screen gave $V/I=0.019$ with the screen in front and 0.197
  behind. The segment operator now rescales $j_P$ onto $j_I$ (keeping $j_I$ and
  the polarization direction), as it does $\alpha_P$ onto $\alpha_I$.
  Cause: `2fab069^:src/radiation/transfer.jl:34` (`stokes_segment` without the
  emission rescale). Fix: `2fab069`. Test:
  `test/test_emission_admissibility.jl:34`,
  `test/test_emission_admissibility.jl:43`.
  Outputs regenerated in `3a6595a` (runs at `bb85b21`):
  `figures/examples/torus_IQUV.png` ($\sum I$ −0.08%, net LP 0.0365 → 0.0366,
  EVPA change median 0.04°, p99 1.5°; this run also postdates `6031897`) and
  `figures/examples/bbh_i90_pol_amr.png` (flux −0.31%, LP 0.8434 → 0.8430; also
  postdates `47527a4`, `35ebb96`, `6031897`). `figures/out/pol_hires`, paper
  figure 3 and the `figures/out/eht_bbh` runs changed by at most
  $2.3\times10^{-10}$ of the image maximum and their figures are
  pixel-identical.
- **`.athdf` dumps storing `press` failed to load**, with "unexpected AthenaK
  primitive order", and `render` never passed $\gamma$ to the `.athdf` loader.
  Cause: `c2f615b^:src/simulation/athenak_reader.jl:153`,
  `c2f615b^:src/render.jl:17`. Fix: `c2f615b`.
  Test: `test/test_athdf_press_layout.jl:49`.
- **Leaf lookup on deep nested meshes fell back to a linear scan** over all
  meshblocks once the dense finest-level table exceeded its 8,000,000-cell cap
  (correct, slow). Cause: `ce50e4c^:src/simulation/athenak_reader.jl:78`,
  `ce50e4c^:src/simulation/athenak_reader.jl:269`. Fix: `ce50e4c`.
  Test: `test/test_leaf_hash.jl:41` (hashed locate equals the scan on 40,000
  points).
- **`figures/plot_pol_hires.py` plotted the image transposed with doubled
  axes** (±30 $M$ for a 30 $M$ field of view, arrays indexed $[\beta,\alpha]$
  shown transposed). Cause: `59f7bd8^:figures/plot_pol_hires.py:28`,
  `59f7bd8^:figures/plot_pol_hires.py:36`. Fix: `59f7bd8`.
- **`figures/fig_z4c_phasespace.jl` copied the anchors as the light ring on
  the CPU path** instead of bisecting it. Cause:
  `5c30761^:figures/fig_z4c_phasespace.jl:411`. Fix: `5c30761`.
- **Figure runs recorded no code version.** `fig_examples.jl`,
  `fig_pol_hires.jl` and `fig_z4c_phasespace.jl` write the HEAD SHA (with
  `-dirty` for uncommitted `src/` edits) into their metadata (`e6beecd`).
- **`figures/fig_eht_bbh.jl` scaled the binary to M87's mass in total**, so
  each hole had half the M87 mass and its ring matched the 20 µas beam. Each
  hole is now M87-like ($M=1.3\times10^{10}\,M_\odot$ total, $\theta_g=7.64$ µas)
  and the plots report ring diameter and central depression.
  Cause: `ee88885^:figures/fig_eht_bbh.jl:72`. Fix: `ee88885`.
- **The README hero caption described an AthenaK GRMHD torus at $a=0.9$**; the
  image is the analytic `PolishDoughnut` at $a=0.94$, $128^2$
  (`2c85701^:README.md:12`; fix `2c85701`). README and method text that still
  described coherency-tensor transport after `6031897` now describe the
  transported-frame march.
- **`docs/src/benchmarks.md` claimed bit-for-bit GPU–CPU polarized parity**
  against a CPU march that used a different scheme; it now states the measured
  correlation (`f865bf2^:docs/src/benchmarks.md:15`; fix `f865bf2`).
- **The kernel audit compiled stale Monte Carlo kernel signatures** (float bins,
  child buffers without streams) after `b15367f`
  (`8ca80ac^:bench/kernel_audit.jl:106`; fix `8ca80ac`).
- **`Pkg.test` lacked test dependencies** (`Printf`, `Random`, `Statistics`,
  `LinearAlgebra` compat) and a slow-light bit-equality test failed under
  `--check-bounds=yes` (`66706a6`).
- **The documentation build failed** on 26 exported names without a manual
  entry (reverse marches, stream identities, exact accumulation, shared
  geodesic kernels); they are now on the API pages.

### Removed

- The Strang-split coherency-tensor transport of the CPU polarized march and
  its helpers (`6031897`).
- The device copies of the physics (`_gimg_*`, `_gpol_*`, `_gsp_*` coefficient,
  Bessel, tetrad, segment and metric functions and the device CGS constants;
  `35ebb96`) and the per-kernel marches (`_gimg_march`, `_gpol_march`,
  `_gsp_march`, `_gsp_pol_march`, `_gsp_ctl`; `6031897`).
- The kernel-only frame-drift stop `_GPOL_FRAME_TOL` (`47527a4`).
- `mc_spectrum_gpu(...; n_gen_override)` and `ScatterWorkspace(child_seed)`
  (`b15367f`).

## [0.2.0] - 2026-10-16

The package `BLP` (`BLP/`, module `BLP`). BlackLightPlus, the package in `src/`, is
unchanged and remains the reference implementation the oracle suite compares against.
Structure and contracts: `docs/ARCHITECTURE.md`; principles and gates:
`docs/DESIGN_PRINCIPLES.md`.

### Added

- **Ten modules in three tiers.** Physics (`Numerics`, `Spacetime`, `Matter`,
  `Transfer`, `MonteCarlo`), infrastructure (`Data`, `Execution`) and pipeline
  (`Plans`, `Analysis`, `Products`), each importing only the modules before it. The
  CUDA methods live in the extension `BLPCUDAExt`; `CUDA` is a weak dependency.
- **Plans.** `prepare(cfg; spacetime, matter, mode, lanes)` derives every quantity of a
  run once into a backend-free `Plan`. The spacetime (`KerrSchildCartesian`,
  `SuperposedKerrSchild`, `ADMGrid`, `UserMetric`, `Minkowski`), the matter (AthenaK
  snapshots, uniform grids, slow-light windows and series, analytic sources), the
  transfer mode (`StokesI`, `IQUV`) and the precision lanes (`Lanes`, `MixedLanes`) are
  values, not configuration fields.
- **Products.** `render` (full frame, coordinate lists, along stored trajectories),
  `render_amr`, `render_analytic`, `photon_ring_layers`, `trace`, `shadow`,
  `sensitivity`, `variants` and `mc_spectrum`. Every result records its executor,
  lanes, package version and wall time in `run`.
- **One kernel.** Every device computation is a per-item program run through the single
  `@kernel` by a `ThreadedExecutor`, or by a `PoolExecutor` over `HostDevice`s and
  `CUDADevice`s; threads and host-device pools give bitwise identical per-item results.
  Monte Carlo spectra are exact integer-limb sums, independent of the executor, thread
  count, device count and chunking.
- **The backend switch.** Every product takes `backend` (`:auto`, `:cpu`, `:gpu`,
  `:host_pool`, an executor, a device list or a pool); `backend!` sets the session
  default and `BLP_BACKEND` (`auto | cpu | gpu | gpu:0,1 | host_pool | host_pool:N`)
  seeds it. `:auto` takes the GPUs when CUDA is loaded and a device is idle, else
  threads, and logs its choice once per session; an explicit choice that cannot be
  honoured throws `BackendUnavailable`.
- **Structure gates.** `BLP/test/structure.jl` tests the layering, the single kernel,
  private names, the export count and the submodule docstrings, file budgets and the
  absence of history prose.
- **Oracle suite.** `BLP/test/oracle/` compares BLP with BlackLightPlus. `Numerics`,
  `Spacetime`, the `Data` samplers, `decode`, `Analysis` and Stokes I images on every
  path are bitwise; IQUV images, traces and the Monte Carlo torus marches agree to
  round-off natively, from floating-point contraction on FMA hardware, and bitwise
  under `julia -C generic`; Monte Carlo spectra are exact sums.

## [0.1.0] - 2026-08-05

Initial release.

[Unreleased]: https://github.com/FoAKTEE/BlackLightPlus/compare/v0.1.0...HEAD
[0.2.0]: https://github.com/FoAKTEE/BlackLightPlus/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/FoAKTEE/BlackLightPlus/releases/tag/v0.1.0
