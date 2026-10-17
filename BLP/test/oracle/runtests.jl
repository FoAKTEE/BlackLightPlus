# Bitwise comparisons of BLP against the BlackLightPlus reference implementation. Run
# from the repository root with
#     julia -t 8 --project=BLP/test/oracle BLP/test/oracle/runtests.jl
# Each script is included into its own module, so their file-scope names never clash.
using Test

const ORACLE_SCRIPTS = ("numerics.jl", "spacetime.jl", "camera.jl", "analytic.jl", "matter.jl",
                        "data.jl", "analysis.jl", "transfer.jl", "montecarlo.jl", "spectra.jl",
                        "products.jl")

@testset "BLP oracle" begin
    for f in ORACLE_SCRIPTS
        Base.include(Module(Symbol("Oracle_", splitext(f)[1])), joinpath(@__DIR__, f))
    end
end
