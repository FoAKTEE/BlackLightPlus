# Build locally, from the repository root:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl

using Documenter
using DocumenterCitations
using BlackLightPlus
# Docstrings of one module that cite a name of another by `@ref` resolve through `Main`.
using BlackLightPlus.Execution: foreach_cascade!

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style = :authoryear)

makedocs(;
    modules = [BlackLightPlus],
    authors = "Haiyang Wang and contributors",
    sitename = "BlackLight+",
    repo = Documenter.Remotes.GitHub("FoAKTEE", "BlackLightPlus"),
    format = Documenter.HTML(;
        canonical = "https://foaktee.github.io/BlackLightPlus",
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        assets = String["assets/citations.css"],
        mathengine = Documenter.KaTeX(),
        sidebar_sitename = true,
        size_threshold = 900_000,
        size_threshold_warn = 700_000,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Guides" => [
            "Imaging GRMHD data" => "guide/grmhd.md",
            "Spacetimes and binaries" => "guide/spacetimes.md",
            "GPU rendering" => "guide/gpu.md",
            "Monte Carlo spectra" => "guide/montecarlo.md",
        ],
        "Method" => "method.md",
        "Validation" => "validation.md",
        "Benchmarks" => "benchmarks.md",
        "Design" => [
            "Architecture" => "design/architecture.md",
            "Design principles" => "design/principles.md",
            "Performance" => "design/performance.md",
            "Verification record" => "design/verification.md",
            "Repository standard" => "design/standards.md",
        ],
        "API Reference" => [
            "Overview" => "api/index.md",
            "Numerics" => "api/numerics.md",
            "Spacetime" => "api/spacetime.md",
            "Matter" => "api/matter.md",
            "Transfer" => "api/transfer.md",
            "MonteCarlo" => "api/montecarlo.md",
            "Data" => "api/data.md",
            "Execution" => "api/execution.md",
            "Plans" => "api/plans.md",
            "Analysis" => "api/analysis.md",
            "Products" => "api/products.md",
        ],
        "Contributing" => "contributing.md",
        "References" => "references.md",
    ],
    plugins = [bib],
    warnonly = false,
    checkdocs = :exports,
    doctest = false,
)

deploydocs(;
    repo = "github.com/FoAKTEE/BlackLightPlus",
    devbranch = "main",
    push_preview = true,
)
