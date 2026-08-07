# Contributing to BlackLightPlus

Thanks for your interest in the code. This file covers the mechanics; the full
normative standard is [`docs/STANDARDS.md`](docs/STANDARDS.md).

## Development setup

```sh
git clone https://github.com/FoAKTEE/BlackLightPlus.git
cd BlackLightPlus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Requires Julia ≥ 1.10. `Manifest.toml` is committed, so `instantiate`
reproduces the exact dependency set used for the benchmarks and figures.

## Running the tests

```sh
julia --project=. -t 8 test/runtests.jl
```

Test blocks that need a GPU or external simulation data skip automatically when
the device or file is absent, and report the skip — a skip is never a silent
pass.

| Variable | Effect |
|---|---|
| `BLP_SKIP_GPU=1` | force-skip GPU test blocks even when CUDA is available |
| `BLP_TEST_GRMHD` | AthenaK GRMHD snapshot path, for data-dependent tests |
| `BLP_REF_AART`, `BLP_REF_GPUMONTY` | reference data for cross-code tests |
| `BLP_NGPU`, `BLP_RESES`, `BLP_MC_NS`, `BLP_AN_RES` | benchmark knobs |

CI runs without a GPU and without external data, so those paths must be
exercised manually before a release:

```sh
BLP_TEST_GRMHD=/path/to/torus.athdf julia --project=. -t 16 test/runtests.jl
CUDA_VISIBLE_DEVICES=0,1 BLP_NGPU=2 julia --project=. bench/bench_gpu.jl
```

Record the outcome in the changelog entry for that release. Budget tens of
minutes for the full CPU suite and a few minutes for the GPU benchmarks.

## Building the documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs with `warnonly = false`, `checkdocs = :exports`, and doctests
enabled. A broken cross-reference, an exported symbol without a docstring, or a
failing doctest fails the build, and a documentation failure is a CI failure.

## Formatting

92-column margin, four-space indent, matching the file you are editing. This is
a convention rather than a tool gate — parts of the source are hand-aligned
where alignment carries meaning (metric components, coefficient tables, kernel
argument lists), and an automatic formatter destroys that.

## Code conventions

- Types `CamelCase`, functions `snake_case`, mutating functions end in `!`.
- Unicode for physical symbols where it matches the literature (`ν`, `α`,
  `Θ_e`); ASCII everywhere else.
- Comments explain *why*, not *what*. File-head comments state the file's role
  in the pipeline and its counterpart in the reference codes.

**The hot-path rule.** Anything reachable from a march loop must be pure,
allocation-free, `isbits`, and `StaticArrays`-typed, so one implementation
serves CPU threads, KernelAbstractions GPU kernels, and ForwardDiff/Enzyme
differentiation. A change that breaks this must say so
explicitly in the pull request and the changelog.

## Documentation and citations

Every exported symbol needs a docstring with an indented signature line, a
one-sentence summary, and stated units and coordinate conventions wherever a
physical quantity crosses the boundary. Prefer `jldoctest` examples so CI
verifies them.

To cite a paper, add it to [`docs/references.bib`](docs/references.bib) with a
verified DOI, omitting the field rather than guessing if none exists, and cite
it inline with `@citet` (narrative) or `@citep` (parenthetical).

## Pull requests

1. Branch from `main`; keep the change focused on one concern.
2. Tests, docs build, and format check must pass.
3. Add a `CHANGELOG.md` entry under `[Unreleased]`, written in terms of
   behaviour a user would notice.
4. If the change adds a dependency on or derivation from third-party work,
   update [`NOTICE`](NOTICE) in the same commit.

## Reporting problems

Open an issue using one of the templates. For a rendering discrepancy, please
include the `RunConfig`, the Julia and CUDA versions, whether it reproduces on
CPU, and the snapshot (or a description of it) if the data can be shared.
