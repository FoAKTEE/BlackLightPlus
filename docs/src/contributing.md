# Contributing

## Development setup

```sh
git clone https://github.com/FoAKTEE/BlackLightPlus.git
cd BlackLightPlus
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running the tests

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

`test/runtests.jl` includes every file under `test/physics`, `test/infra` and
`test/pipeline`, each in its own module, after `test/structure.jl`, the mechanical gates
of the design ([Design principles](@ref), §4): layering, one kernel, no private reach
across modules, one home per concept, the size of the public API, documented exports, file
budgets and the absence of attribution or process prose. A new source file joins the
suite by existing; a stray `test_*.jl` fails the gate.

Test blocks that need a CUDA device or external simulation data skip with a report when
the device or file is absent. On a machine where GPUs are reached through a scheduler,
run the suite inside the allocation; `BLP_TEST_DEVICES=cuda` makes the device comparisons
run on the GPUs.

The comparison against the reference implementation (the bitwise oracle, `test/oracle/`)
lives on the `blp` branch with that implementation; its results are in the
[Verification record](@ref).

## Building the documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The build runs with `warnonly = false` and `checkdocs = :exports`: a broken
cross-reference or an exported name without a docstring fails the build, and a
documentation failure is a CI failure.

## Code style

- 92-column margin, four-space indent, matching the file you are editing. Parts of the
  source are hand-aligned where alignment carries meaning (metric components, coefficient
  tables), so no automatic formatter is applied.
- Types are `CamelCase`, functions `snake_case`, mutating functions end in `!`. Unicode
  is used for physical symbols where it matches the literature (`ν`, `α`, `Θ_e`) and
  avoided elsewhere.
- A name with a leading underscore is private to its module and is never referenced from
  another module or from the tests.
- Comments state the contract and the physics; provenance and measurements belong to the
  [Method](@ref) page and the [Verification record](@ref), not to the code.

### The hot-path rule

Anything reachable from a march loop is pure, allocation-free, `isbits` after
`Adapt.adapt`, and `StaticArrays`-typed, so one implementation serves CPU threads, the
device kernel and forward-mode differentiation. A change that breaks this property must
say so in the pull request and in the changelog.

## Documentation expectations

Every exported name needs a docstring with an indented signature line, a one-sentence
summary, and stated units and coordinate conventions wherever a physical quantity crosses
the boundary. Examples that need external data are plain `julia` blocks naming the file
they expect.

Citations use the shared bibliography: add the entry to `docs/references.bib` with a
verified DOI, then cite it inline, narrative form for "Author (Year) showed…",
parenthetical otherwise:

````markdown
the scheme of [White2022](@citet)        renders as   White (2022)
the adaptive scheme [White2022](@citep)  renders as   (White, 2022)
````

The full contract is the [Repository standard](@ref).

## Pull requests

- Branch from `main`; keep the change focused on one concern.
- Tests and the docs build must pass.
- Update `CHANGELOG.md` under `[Unreleased]`, describing the change in terms of behaviour
  a user would notice.
- If the change touches provenance (a new dependency on or derivation from third-party
  work) update `NOTICE` in the same commit.
