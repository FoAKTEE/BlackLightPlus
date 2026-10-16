# Every test file under physics/, infra/ and pipeline/ is included, each into
# its own module, so file-scope constants never clash. Shared decks live in
# `decks.jl` as the module `Decks`.
using Test
using BLP

include("decks.jl")

@testset "BLP" begin
    for dir in ("physics", "infra", "pipeline")
        files = sort(filter(f -> endswith(f, ".jl"), readdir(joinpath(@__DIR__, dir); join = true)))
        for f in files
            name = Symbol("Test_", dir, "_", splitext(basename(f))[1])
            Base.include(Module(name), f)
        end
    end
end
