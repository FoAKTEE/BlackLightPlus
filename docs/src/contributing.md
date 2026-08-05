# Contributing

## Development setup

```sh
git clone https://github.com/FoAKTEE/BlackLightPlus.git
cd BlackLightPlus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running the tests

```sh
julia --project=. -t 8 test/runtests.jl
```

The suite validates geodesics, transfer coefficients, polarized coupling,
automatic differentiation, and Monte Carlo scattering against analytic
solutions and published reference codes.

Test blocks that need a GPU or external simulation data skip automatically when
the device or file is absent, and report the skip — a skip is never a silent
pass.

### Environment variables

| Variable | Effect |
|---|---|
| `BLP_SKIP_GPU=1` | force-skip GPU test blocks even when CUDA is available |
| `BLP_TEST_GRMHD` | path to an AthenaK GRMHD snapshot, for data-dependent tests and benchmarks |
| `BLP_REF_AART` | reference data from AART, for cross-code tests |
| `BLP_REF_GPUMONTY` | reference data from gpumonty, for cross-code tests |
| `BLP_NGPU`, `BLP_RESES`, `BLP_MC_NS`, `BLP_AN_RES` | benchmark knobs, see [Benchmarks](@ref) |

### Before a release

CI runs without a GPU and without external data, so the GPU and data-dependent
paths must be exercised manually before tagging:

```sh
BLP_TEST_GRMHD=/path/to/torus.athdf julia --project=. -t 16 test/runtests.jl
CUDA_VISIBLE_DEVICES=0,1 BLP_NGPU=2 julia --project=. bench/bench_gpu.jl
```

Record the result in the changelog entry for that release. Expect tens of
minutes for the full CPU suite and a few minutes for the GPU benchmarks.

## Building the documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs with `warnonly = false`, `checkdocs = :exports`, and doctests
enabled. A broken cross-reference, an exported symbol without a docstring, or a
failing doctest fails the build, and a documentation failure is a CI failure.

## Code style

- Formatting is mechanical: `JuliaFormatter` with the committed
  `.JuliaFormatter.toml`, 92-column margin. Run it before opening a pull
  request; CI checks it.
- Types are `CamelCase`, functions are `snake_case`, mutating functions end in
  `!`. Unicode is used for physical symbols where it matches the literature
  (`ν`, `α`, `Θ_e`) and avoided elsewhere.
- Comments explain *why*, not *what*. File-head comments state the file's role
  in the pipeline and its counterpart in the reference codes.

### The hot-path rule

Anything reachable from a march loop must be pure, allocation-free, `isbits`,
and `StaticArrays`-typed. This is what lets one implementation serve CPU
threads, KernelAbstractions GPU kernels, and ForwardDiff/Enzyme
differentiation. A change that breaks this property must say so explicitly in
the pull request and in the changelog.

## Documentation expectations

Every exported symbol needs a docstring with an indented signature line, a
one-sentence summary, and stated units and coordinate conventions wherever a
physical quantity crosses the boundary. Examples should be `jldoctest` blocks
so CI verifies them; if an example needs external data, mark it as a plain
`julia` block and name the test file that exercises it.

Citations use the shared bibliography. Add the entry to `docs/references.bib`
with a verified DOI — never a guessed one — and cite it inline as
`[Key](@citet)` or `[Key](@citep)`.

The full contract is in
[`docs/STANDARDS.md`](https://github.com/FoAKTEE/BlackLightPlus/blob/main/docs/STANDARDS.md).

## Pull requests

- Branch from `main`, keep the change focused on one concern.
- Tests and docs build must pass.
- Update `CHANGELOG.md` under `[Unreleased]`, describing the change in terms of
  behaviour a user would notice.
- If the change touches provenance — a new dependency on or derivation from
  third-party work — update `NOTICE` in the same commit.
