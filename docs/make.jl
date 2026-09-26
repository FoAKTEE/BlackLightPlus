# Build locally with:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl

using Documenter
using DocumenterCitations
using BlackLightPlus

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"); style = :authoryear)

makedocs(;
    modules = [BlackLightPlus],
    authors = "Haiyang Wang and contributors",
    sitename = "BlackLightPlus",
    repo = Documenter.Remotes.GitHub("FoAKTEE", "BlackLightPlus"),
    format = Documenter.HTML(;
        canonical = "https://foaktee.github.io/BlackLightPlus",
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        assets = String["assets/citations.css"],
        mathengine = Documenter.KaTeX(),
        sidebar_sitename = false,
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Guides" => [
            "Imaging GRMHD data" => "guide/grmhd.md",
            "Numerical-relativity and binary spacetimes" => "guide/spacetimes.md",
            "GPU rendering" => "guide/gpu.md",
            "Monte Carlo spectra" => "guide/montecarlo.md",
        ],
        "Method" => "method.md",
        "Benchmarks" => "benchmarks.md",
        "Validation" => "validation.md",
        "API Reference" => [
            "Overview" => "api/index.md",
            "Configuration" => "api/inputs.md",
            "Geodesics" => "api/geodesics.md",
            "Simulation input" => "api/simulation.md",
            "Radiation" => "api/radiation.md",
            "Sources" => "api/sources.md",
            "Monte Carlo" => "api/montecarlo.md",
            "Outputs" => "api/outputs.md",
            "Render drivers" => "api/driver.md",
            "GPU drivers" => "api/gpu.md",
        ],
        "Contributing" => "contributing.md",
        "References" => "references.md",
    ],
    plugins = [bib],
    warnonly = false,
    checkdocs = :exports,
    doctest = true,
)

deploydocs(;
    repo = "github.com/FoAKTEE/BlackLightPlus",
    devbranch = "main",
    push_preview = true,
)
