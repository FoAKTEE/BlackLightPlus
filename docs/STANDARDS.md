# BlackLightPlus repository and documentation standard

**Status:** normative. **Applies to:** everything in this repository.
**Target:** companion code for a method paper, published as a public GitHub
repository — citable, reproducible, and reviewable by an astrophysicist who has
never seen the code.

Every rule below is stated so that a reviewer can mark it pass or fail.

The three governing principles:

1. **A reader can get from zero to a rendered image without reading source.**
2. **Every physics claim, number, and algorithm is traceable** — to a citation,
   to a test, or to a benchmark with a stated machine.
3. **Nothing is documented in two places.** One fact, one home, links elsewhere.

---

## 1. Repository layout

```
BlackLightPlus/
├── README.md               entry point; §3.1
├── LICENSE                 MIT
├── NOTICE                  third-party provenance; §11
├── CITATION.cff            machine-readable citation; §6.4
├── CHANGELOG.md            Keep a Changelog format; §7.2
├── CONTRIBUTING.md         how to build, test, and submit; §3.1
├── CODE_OF_CONDUCT.md      Contributor Covenant 2.1
├── Project.toml            name, uuid, version, deps, compat; §7.1
├── Manifest.toml           committed — this is an application, not a library
├── .gitignore
├── .JuliaFormatter.toml    style config; §9
├── .github/
│   ├── workflows/          CI: test, docs, format; §8
│   └── ISSUE_TEMPLATE/     bug report, feature request
├── src/                    package source; one submodule per directory
├── test/                   runtests.jl + test_*.jl, one per subsystem
├── bench/                  benchmark drivers + bench/README.md
├── docs/                   Documenter.jl site; §3
│   ├── Project.toml
│   ├── make.jl
│   ├── references.bib      §6.2
│   └── src/                site pages
└── figures/                paper-figure scripts; §10.3
```

**Rules**

- `src/` mirrors the module tree. A directory is a submodule; its
  eponymous file (`geodesics/geodesics.jl`) is the only file that `include`s
  its siblings and the only one that declares exports.
- Files that require an optional dependency (CUDA) live at `src/` top level as
  standalone drivers (`gpu_*.jl`), are **not** `include`d by the package, and
  are documented as explicit-include drivers in the GPU guide. This keeps the
  package installable on CUDA-free machines, and must be stated in `README.md`
  and in the GPU page so it does not read as an oversight.
- No file exceeds ~1000 lines. Above that, split by physical concern.
- No generated artifact is committed: no `.npy`, `.png`, `.pdf`, no `out/`.
  Figures in `figures/examples/` are the sole exception (§10.3) and each must
  be regenerable by a named script in the same directory.

## 2. Required root files

Each root file has one job. A file that does two jobs is split.

| File | Answers | Must contain |
|---|---|---|
| `README.md` | "What is this and should I use it?" | §3.1 |
| `CONTRIBUTING.md` | "How do I build, test, and submit a change?" | dev setup, test invocation, style rule, PR expectations, where docs live |
| `CHANGELOG.md` | "What changed between versions?" | §7.2 |
| `CITATION.cff` | "How do I cite this?" | §6.4 |
| `NOTICE` | "What is derived from elsewhere?" | §11 |
| `LICENSE` | "What may I do with it?" | MIT, one copyright line, current year |
| `CODE_OF_CONDUCT.md` | "What are the community norms?" | Contributor Covenant 2.1, real contact address |

## 3. Documentation architecture

Documentation is split by **reader intent**, not by subsystem. The four modes
are distinct and must not be mixed within one page:

| Mode | Page(s) | Reader | Rule |
|---|---|---|---|
| **Orientation** | `README.md` | evaluating in 60 s | no API detail; links out |
| **Tutorial** | `docs/src/getting-started.md` | first successful run | one linear path, no branches, no options |
| **How-to** | `docs/src/guide/*.md` | has a specific goal | task-titled ("Image a GRMHD snapshot"), assumes tutorial done |
| **Explanation** | `docs/src/method.md` | wants to know *why* | physics and numerics; no installation, no API listings |
| **Reference** | `docs/src/api/*.md` | knows what they want | generated from docstrings; §5 |

**Rules**

- A page states its mode in the first paragraph, implicitly, by what it does.
  Installation instructions never appear in an explanation page; physics
  derivations never appear in a how-to.
- Cross-reference by link, never by restatement. If two pages need the same
  fact, one owns it and the other links (`[the transfer scheme](@ref)`).
- Every code block in a tutorial or how-to is **runnable as written** against
  either no external data or a named, described input file. No pseudo-code, no
  `...` elisions in a block the reader is meant to execute.
- Every claim of the form "X is N× faster" or "matches to 10⁻⁹" links to the
  benchmark table (§10) or the test that establishes it.

### 3.1 README contract

The README is an abstract, not a manual. Required sections, in order:

1. **Title + one-sentence description** — what it does, in what language, for whom.
2. **Badges** — CI status, docs (stable/dev), license, DOI. Nothing else.
3. **Hero figure** — one rendered image with a caption naming the physics and
   the script that produced it.
4. **Capabilities** — bulleted, each ≤ 3 lines, each naming the concrete
   entry point or type. No paragraphs.
5. **Installation** — exact commands, real clone URL (no placeholders), stated
   Julia lower bound, explicit statement of what is optional (CUDA).
6. **Quickstart** — one block that runs with no external data, and one that
   shows the GRMHD path. Both must be copy-pasteable.
7. **Documentation** — a table of links to the doc site pages, one line each.
8. **Testing** — invocation and what the suite covers in two sentences.
9. **Citing** — §6.5.
10. **Heritage / related codes** — §6.3.
11. **License** — one line plus a pointer to `NOTICE`.

Hard limits:

- Total ≤ 150 lines.
- No **prose** section longer than 12 lines. Fenced code blocks, table rows,
  and the capability bullets do not count toward this — they are bounded by
  their own rules above (two quickstart blocks; capability bullets ≤ 3 lines
  each). A README whose prose stays under 12 lines per section but whose
  quickstart is 20 lines of runnable code is correct.
- No API surface enumerated; no benchmark table (link to it).

### 3.2 Doc site

- Built with **Documenter.jl**, deployed to GitHub Pages on tag and on `main`.
- `docs/Project.toml` pins `Documenter` and declares `BlackLightPlus` via
  `Pkg.develop(path="..")` in the workflow — never a duplicated dep list.
- `makedocs(; warnonly = false)` — a broken cross-reference or a missing
  docstring **fails the build**. Doc-site failure is a CI failure.
- Math renders through KaTeX; every `$…$` in a page must render (checked by
  the build, not by eye).
- The site's navigation order is the reader's order: Home → Getting Started →
  Guides → Method → Benchmarks → API → Contributing → References.

## 4. Docstring standard

Julia's official convention, applied without exception to every **exported**
symbol. Internal helpers (leading `_`) may use the one-line form.

````julia
"""
    trace_rays_sampled(spacetime, rays; C=80.0, nmax=..., escape=...)

One-sentence summary in the imperative mood, ending in a period.

Extended description: what the function computes and the convention it assumes
(coordinates, signature, units). State the *why* only if it is not obvious.

# Arguments
- `spacetime`: any `AbstractSpacetime`; supplies `g_{μν}(x)`.
- `rays`: initial states `[x^μ; p_μ]` from [`camera_rays`](@ref).

# Keywords
- `C`: control-step divisor; step is `h = (r - r_h) / C`.

# Returns
`(states, dls)` — trajectories reordered source-to-camera, and per-segment
affine spans.

# Examples
```jldoctest
julia> st = KerrSchildCartesian(0.9);

julia> size(camera_rays(st, PlaneParallelCamera(60.0, 1e3, 20.0, 8)))
(8, 8)
```

# References
Following the sampling scheme of White (2022) §3.2.

See also [`trace_geodesic`](@ref), [`camera_rays`](@ref).
"""
````

**Rules**

- Signature line first, indented four spaces, in a bare code block. This is what
  makes `?trace_rays_sampled` render correctly and is non-negotiable for
  exported symbols.
- Summary is one sentence. If it needs two, the second belongs in the extended
  description.
- Units and coordinate conventions are stated wherever a physical quantity
  enters or leaves. "cgs", "geometrized `G = c = M = 1`", "CKS" — say which.
- Every `# Examples` block is a `jldoctest` and is executed in CI (§8). An
  example that cannot run without external data is marked
  ```` ```julia ```` and is **not** a doctest — but must then be exercised by a
  test file, named in the docstring.
- Types get docstrings on the type, not on the constructor, and document every
  field.
- A published algorithm gets a `# References` section naming the source in
  author-year form matching §6.1.

## 5. API reference standard

- `docs/src/api/` has one page per submodule: `geodesics.md`, `radiation.md`,
  `simulation.md`, `montecarlo.md`, `sources.md`, `outputs.md`, `inputs.md`,
  plus `driver.md` for `render`/`render_amr`/`render_analytic` and `gpu.md`
  for the optional drivers.
- Each page opens with a 3–5 line orientation paragraph (what this module owns
  and where it sits in the pipeline), then `@docs` blocks grouped by concern
  with a subheading per group — never one undifferentiated alphabetical dump.
- **Coverage is enforced**: `makedocs` runs with `checkdocs = :exports`, so
  every exported symbol must appear in an `@docs` block and must have a
  docstring. A new export without documentation fails CI.
- The public API is exactly what `BlackLightPlus.jl` exports. Anything not
  exported is internal and may change without a version bump — this must be
  stated verbatim on the API index page.
- GPU driver functions are documented on `gpu.md` with an explicit note that
  they require `include`ing the driver after loading CUDA, and the required
  include order.

## 6. Citation and bibliography standard

This repository is scientific work; its citations are held to the standard of
the journals it cites. **AAS journals style** throughout.

### 6.1 Inline citation form

- Parenthetical: `(White 2022)`, `(Mościbrodzka & Gammie 2018)`,
  `(Dolence et al. 2009)`.
- Narrative: `White (2022) showed …`.
- Two authors use `&`; three or more use `et al.` from the first occurrence.
- Diacritics are always correct: **Mościbrodzka**, **Cárdenas-Avendaño**.
  A stripped-accent spelling is a defect, not a formatting preference.
- Never cite a code by name alone. `ipole` is a name; `ipole (Mościbrodzka &
  Gammie 2018)` is a citation. First mention in each document carries the
  citation; later mentions may use the bare name.
- Every author-year that appears in prose **must** resolve to an entry in
  `docs/references.bib`. No orphans in either direction.

### 6.2 `docs/references.bib`

Single source of truth for the bibliography. Every entry:

- Is derived from **NASA ADS / the publisher of record**, not from memory.
- Uses a **human-readable citation key** in `AuthorYear` form (`White2022`,
  `MoscibrodzkaGammie2018`), disambiguated by a title word when one author
  published twice in a year (`GrallaLupsasca2020NullGeodesics`). Keys are
  written by hand into prose, so they must be readable at the call site; the
  ADS bibcode is preserved in a `bibcode` field rather than used as the key.
- Carries, at minimum: `author`, `title`, `journal`, `year`, `volume`, `pages`,
  `doi`, `bibcode`, `adsurl`, `eprint` (arXiv id) where one exists.
- Has every `doi` and `eprint` checked against Crossref, the arXiv API, or the
  publisher's page before commit. This applies to the *whole* entry: an arXiv
  id recalled from memory is as untrustworthy as a guessed DOI.
- Preserves non-ASCII author names in proper LaTeX-safe form.
- Software with no paper cites the software record (Zenodo DOI) or the
  repository, using **`@misc`** with `url` and `year`. Not `@software`: that is
  a BibLaTeX entry type, and the classic-BibTeX parser behind
  DocumenterCitations rejects it, failing the docs build.

A DOI that is not verified against the publisher or ADS is not committed.
Guessing a DOI is worse than omitting the field.

### 6.3 Reference sections in prose documents

- `docs/src/references.md` is the rendered bibliography, generated from
  `references.bib` via `DocumenterCitations.jl`. It is the only place the full
  entries appear.
- `method.md` and other pages cite inline (§6.1) and **do not** carry their own
  reference lists. The present run-on `·`-separated list at the end of
  `method.md` is replaced by links into the bibliography page.
- The README's **Heritage** section names the codes BlackLightPlus builds on,
  each with an author-year citation and a repository link where one exists, and
  states in one sentence what was taken from each (algorithm, convention, or
  validation target). It is a provenance statement, not a bibliography — full
  entries live in the bibliography page.

### 6.4 `CITATION.cff`

Must validate against the CFF 1.2.0 schema and must include, beyond the current
fields: `abstract`, `repository-code`, `url`, `license`, `keywords`, ORCID
identifiers for every author, and `doi` (the Zenodo concept DOI) once the first
release is archived. When the method paper appears, a `preferred-citation`
block of type `article` is added and the top-level entry remains as the
software citation.

### 6.5 "Citing" section

States plainly: which paper to cite when it exists, what to cite in the
interim, and gives one copy-pasteable BibTeX block. No prose beyond three
lines. It is updated in the same commit that creates a release.

## 7. Versioning and releases

### 7.1 Version numbers

- Semantic versioning on the **exported** API (§5). `0.x` while the API is
  still moving; the `0.x` contract is that `x` bumps on breaking change.
- `Project.toml` `[compat]` carries an entry for **every** dependency and for
  `julia`. Missing compat entries are a defect.
- `Manifest.toml` is committed. This repository is a reproducible application:
  a reader must be able to reproduce a figure with the exact dependency set.

### 7.2 `CHANGELOG.md`

[Keep a Changelog](https://keepachangelog.com) 1.1.0 format: an `[Unreleased]`
section at top, then one section per released version with an ISO date, and
entries grouped under `Added` / `Changed` / `Fixed` / `Removed`. Entries are
written for a *user of the code*, describing behaviour, not commits.

### 7.3 Releases

A release is: version bump in `Project.toml` and `CITATION.cff` → changelog
section dated → git tag `vX.Y.Z` → GitHub release → Zenodo archive → DOI badge
and `CITATION.cff` `doi` updated.

## 8. Continuous integration and quality gates

Three workflows, all required to pass on `main` and on every PR:

| Workflow | Runs | Gate |
|---|---|---|
| `CI.yml` | `Pkg.test()` on Julia 1.10 (LTS lower bound) and 1.12, on Linux | all non-skipped tests pass |
| `Documentation.yml` | `docs/make.jl` with `warnonly = false`, `checkdocs = :exports`, doctests on | build clean, no missing docstrings, doctests pass |
| `Format.yml` | `JuliaFormatter` in check mode | no diff |

**Rules**

- CI runs **without** a GPU and **without** external simulation data. Tests
  that need either skip themselves and report the skip; a skip is never a
  silent pass. The suite must print a summary of what was skipped and why.
- The GPU and data-dependent paths are covered by a documented manual
  procedure in `CONTRIBUTING.md`, with the environment variables required
  (`BLP_TEST_GRMHD`, `BLP_REF_AART`, `BLP_REF_GPUMONTY`, `BLP_SKIP_GPU`) and
  the expected runtime. Before a release, that procedure is run and the result
  recorded in the changelog entry.
- A test asserting agreement with a published result names the source and the
  tolerance in a comment at the assertion.

## 9. Code style

- `JuliaFormatter` with a committed `.JuliaFormatter.toml`; 92-column margin
  to match the existing source. Formatting is mechanical and never discussed
  in review.
- Naming follows Julia convention: `TypesInCamelCase`, `functions_in_snake_case`,
  `mutating!` functions marked, no Hungarian prefixes. Unicode is permitted for
  physical symbols where it matches the literature (`ν`, `α`, `Θ_e`, `γ`) and
  avoided everywhere else.
- Comments explain *why*, not *what*. A comment restating the line below it is
  deleted. File-head comments state the file's role in the pipeline and its
  counterpart in the reference codes where one exists — this convention is
  already established in `src/` and is retained.
- Hot-path code (anything reachable from a march loop) is pure, allocation-free,
  `isbits`, and `StaticArrays`-typed, so one implementation serves CPU threads,
  KernelAbstractions kernels, and AD. A change that breaks this must say so in
  the PR and in the changelog.

## 10. Reproducibility

### 10.1 Numbers

Any performance or accuracy number that appears in the README, the doc site,
or the paper must be reproducible from a committed script. A number without a
script is deleted.

### 10.2 Benchmarks

`docs/src/benchmarks.md` states, at the top: the machine (CPU model and count,
GPU model and count, memory), the software stack (CUDA version, Julia version),
the measurement protocol (warm/cold, best-of-N, exclusivity), and the metric
definitions. Every table is regenerable by a named `bench/` script with its
environment variables listed. Numbers carry the resolution and configuration
they were measured at.

### 10.3 Figures

Each figure in `figures/examples/` has: a generator script in `figures/`, a
caption in the doc page that uses it naming the physical configuration, and a
stated runtime and data requirement. Generated intermediates
(`figures/out/`) stay gitignored.

### 10.4 Data

External inputs (AthenaK snapshots, reference-code outputs) are never
committed. Each is described where it is used: what it is, roughly how large,
how to obtain or generate it, and which environment variable points at it.

## 11. Provenance and licensing

`NOTICE` distinguishes three tiers, and every file in the first two carries a
matching header comment:

1. **Ported code** — the upstream license, copyright holder, and URL, per file.
2. **Clean-room reimplementation of a published algorithm** — the paper cited,
   an explicit statement that no source was copied, and the license of the
   codebase being avoided (GPL) so the reason for the clean-room is legible.
3. **Convention and validation targets** — codes whose conventions are followed
   or whose outputs are compared against, with no code relationship.

A new dependency on, or derivation from, third-party work updates `NOTICE` in
the same commit.
