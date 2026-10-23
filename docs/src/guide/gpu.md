# GPU rendering

Every product runs on Julia threads or on a pool of CUDA devices; the `backend` option
decides, and the result is the same to the last bit per pixel or per photon whatever the
device count and chunking (the two targets differ only by their floating-point
contraction, at the $10^{-14}$–$10^{-9}$ level of the image maximum).

## Loading the extension

CUDA is a weak dependency. Load it in the session and the extension
`BlackLightPlusCUDAExt` adds the `CUDADevice` methods of the device protocol:

```julia
using BlackLightPlus, CUDA
img = render(plan; backend = :gpu)
```

Without `using CUDA`, `backend = :gpu` throws `BackendUnavailable`; `:auto` falls back to
threads with a logged reason. On a machine whose login shell has no device access, run
the GPU work inside the scheduler's allocation and pin the CUDA runtime in
`LocalPreferences.toml` (`[CUDA_Runtime_jll] version = "13.3"`) so the runtime does not
have to see the driver when it compiles.

## Choosing devices

```julia
select_devices()                       # the idle CUDA devices: free memory ≥ 8 GB, utilisation ≤ 15%
render(plan; backend = :gpu)           # a pool over those devices
render(plan; backend = [0, 1])         # CUDA ordinals 0 and 1
render(plan; backend = CUDADevice(1))  # one device
backend!("gpu:0,1")                    # the BLP_BACKEND grammar works here too
```

Each device list gets one persistent `DevicePool`, created on first use and kept in the
process registry: `lanes` worker tasks per device (two by default), each with its own
stream and pinned staging buffers, pulling chunks from one shared queue, so a slower or
shared device takes fewer. Run Julia with more threads than devices (`-t 3` for two
GPUs) so the workers of different devices run concurrently. For explicit control build
the pool yourself and close it when done:

```julia
pool = DevicePool([0, 1]; lanes = 2, chunk = 65536)
img = render(plan; backend = PoolExecutor(pool))
close(pool)
```

`backend = [HostDevice(1), HostDevice(2)]` runs the same programs on the
KernelAbstractions CPU backend, one logical device per entry; that is how the
multi-device tests run without a GPU.

## What the device does

A product is a per-item program run by one kernel. On a pool a chunk of pixel indices
goes to the device, the device builds each camera ray with the same `camera_ray` as the
host, marches the geodesic, samples the fluid on the AthenaK mesh hierarchy with the same
trilinear stencil as the host, runs the transfer and writes its pixel; no trajectory is
stored. Host arrays a program holds (the snapshot, the tables) are uploaded once per device
and found again by identity, so a change of chunk or lane count re-uploads nothing. The
image tiles are queued longest-marches-first on Kerr (distance to the critical curve), and
`prepare(cfg; queue = ...)` takes another order, for example the measured trip counts of
the previous frame of a movie.

## Precision lanes

The geodesic lane is a plan property:

```julia
prepare(cfg; spacetime, matter, lanes = Lanes{Float32,Float64,Float64}())   # Float32 geodesics
prepare(cfg; spacetime, matter, lanes = MixedLanes(2))                       # Kerr only
```

`MixedLanes(rescue_px)` marches the pixels within `rescue_px` of the Kerr critical curve
in `Float64`, where a `Float32` march loses the ray, and everything else in `Float32`; the
transfer and the accumulator are always `Float64`. `render_amr`, `trace`, `shadow` and
`render(plan, coords)` take uniform lanes only.

## Verifying against the CPU

The two backends run the same programs, so a difference beyond round-off means the
plan differs, not the physics:

```julia
cpu = render(plan; backend = :cpu)
gpu = render(plan; backend = :gpu)
maximum(abs, cpu.I .- gpu.I) / maximum(cpu.I)          # 1e-14 … 1e-9
```

Measured throughput and the two-device scaling are on [Benchmarks](@ref); the
execution model and the tuning knobs (`chunk`, `lanes`, the queue) are in
[Performance](@ref).
