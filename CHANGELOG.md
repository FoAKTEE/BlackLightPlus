# Changelog

All notable changes to BlackLightPlus are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
on its exported API. While the version is `0.x`, a bump of `x` may break that
API. Anything not exported is internal and may change in any release.

## [Unreleased]

### Added

- `docs/STANDARDS.md`, the normative repository and documentation standard this
  project is held to.
- Documenter.jl documentation site with a tutorial, four task-oriented guides,
  and a generated API reference covering all 182 exported symbols. The build
  enforces `checkdocs = :exports`, `warnonly = false`, and doctests, so an
  undocumented export or a broken cross-reference fails CI.
- `docs/references.bib`: a bibliography with DOIs and arXiv identifiers
  verified against Crossref, the arXiv API, and publishers of record.
- CI workflows for tests (Julia 1.10 and latest), documentation, and formatting.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, issue templates, and
  `.JuliaFormatter.toml`.

### Changed

- Citations throughout now follow AAS journals style and resolve to
  `docs/references.bib`. The run-on reference list at the end of the method
  page is replaced by a rendered bibliography.
- `NOTICE` restructured into the three provenance tiers of STANDARDS.md §11,
  and now distinguishes AART's verbatim upstream copyright line from the
  citation of its method paper, whose author order and spelling differ.
- `CITATION.cff` gained `abstract`, `repository-code`, `url`, and expanded
  keywords.
- README rewritten to the STANDARDS.md §3.1 contract: badges, hero figure,
  runnable quickstart, and a documentation index, with the placeholder clone
  URL replaced by the real one.

### Fixed

- AART was cited with the wrong author order. The method paper is
  Cárdenas-Avendaño, Lupsasca & Zhu (2023); the repository previously had
  Cárdenas-Avendaño, Zhu & Lupsasca, which is the order of the upstream
  *copyright line*, not the paper.
- AthenaK was cited as an unpublished 2024 preprint. It is published:
  Stone et al. 2026, ApJS 283, 27. The numerical-relativity solver paper
  (Zhu et al. 2025, ApJS 278, 50) is now cited where the Z4c reader is
  described.
- The Jipole differentiable-GRRT construction was invoked in the method
  documentation without a citation. It is now cited as Naethe Motta, Prather &
  Cárdenas-Avendaño (2025), ApJ 995, 56.
- Gralla & Lupsasca (2020) is now cited unambiguously. The elliptic geodesic
  formulation is "Null geodesics of the Kerr exterior" (PRD 101, 044032), which
  has a companion paper at PRD 101, 044031.
- `.gitignore`'s blanket `*.png` rule excluded the curated example figures and
  every documentation-site asset. Both directories are now negated explicitly.
- Diacritics corrected in prose citations of Mościbrodzka.

## [0.1.0] - 2026-08-05

Initial release.

[Unreleased]: https://github.com/FoAKTEE/BlackLightPlus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FoAKTEE/BlackLightPlus/releases/tag/v0.1.0
