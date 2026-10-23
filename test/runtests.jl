# Every test file under physics/, infra/ and pipeline/ is included, each into
# its own module, so file-scope constants never clash. Shared decks live in
# `decks.jl` as the module `Decks`.
using Test
using BlackLightPlus

include("decks.jl")

@testset "BlackLightPlus" begin
    include("structure.jl")
    for dir in ("physics", "infra", "pipeline")
        isdir(joinpath(@__DIR__, dir)) || continue
        files = sort(filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, dir); join = true)))
        for f in files
            name = Symbol("Test_", dir, "_", splitext(basename(f))[1])
            Base.include(Module(name), f)
        end
    end
end
