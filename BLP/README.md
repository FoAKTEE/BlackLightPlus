# BLP

The second version of BlackLightPlus: general-relativistic ray tracing,
polarized radiative transfer and Monte Carlo Comptonization in Julia, as one
package whose products run on the CPU or on GPUs through a single `backend`
option.

```julia
using BLP
cfg  = RunConfig(m_bh_cgs = 6.2e9 * 1.989e33, mass_scale_cgs = 1e24, frequency_hz = 230e9)
plan = prepare(cfg; spacetime = KerrSchildCartesian(0.9), matter = AthenaKSnapshot("dump.athdf"), mode = IQUV())
img  = render(plan)                 # :auto — GPUs when `using CUDA` finds one, else threads
img  = render(plan; backend = :cpu)
img  = render(plan; backend = [0, 1])
```

The session default is `backend!(:gpu)`; the environment variable
`BLP_BACKEND` (`auto`, `cpu`, `gpu`, `gpu:0,1`, `host_pool:2`) seeds it.
`backend()` prints the resolved default and every result records the executor
it ran on in `result.run`.

Structure and contracts: `../docs/ARCHITECTURE.md`; principles and gates:
`../docs/DESIGN_PRINCIPLES.md`.

Install from the repository root with `] dev ./BLP`; test with
`julia -t auto --project=BLP -e 'using Pkg; Pkg.test()'`.
