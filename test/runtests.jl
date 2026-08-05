# BlackLightPlus test suite entry point: each component test file is included
# only when its corresponding source file exists.
using Test
using BlackLightPlus
@testset "BlackLightPlus" begin
    include("test_spacetime.jl")
    include("test_spacetimes.jl")
    isfile(joinpath(@__DIR__, "..", "src", "geodesics", "numerical_spacetime.jl")) &&
        include("test_numerical_spacetime.jl")
    isfile(joinpath(@__DIR__, "..", "src", "simulation", "primitives.jl")) &&
        include("test_simulation.jl")
    isfile(joinpath(@__DIR__, "..", "src", "radiation", "transfer.jl")) &&
        include("test_transfer.jl")
    isfile(joinpath(@__DIR__, "..", "src", "radiation", "coefficients.jl")) &&
        include("test_coefficients.jl")
    isfile(joinpath(@__DIR__, "..", "src", "radiation", "integrator.jl")) &&
        include("test_autodiff.jl")
    # Mixed-precision kernel tests (KA CPU backend only; no CUDA device touched)
    isfile(joinpath(@__DIR__, "..", "src", "gpu_image.jl")) &&
        include("test_mixed_precision.jl")
    isfile(joinpath(@__DIR__, "..", "src", "geodesics", "analytic_kerr.jl")) &&
        include("test_analytic_kerr.jl")
    isfile(joinpath(@__DIR__, "..", "src", "radiation", "equatorial_source.jl")) &&
        include("test_ring_visamp.jl")
    isfile(joinpath(@__DIR__, "..", "src", "sources", "sources.jl")) &&
        include("test_sources.jl")
    # Analytic-disk polarized regression (CPU-only in the suite; the GPU
    # stiffness-telemetry block runs in its standalone invocation)
    isfile(joinpath(@__DIR__, "test_analytic_disk.jl")) &&
        withenv("BLP_SKIP_GPU" => "1") do
            include("test_analytic_disk.jl")
        end
    # Slow-light CPU tests: synthetic tiny_snap, no external data and no CUDA
    # (GPU tests auto-skip under BLP_SKIP_GPU). Included into its own module so
    # its file-scope consts (DATA, RES, …) don't clash with the suite.
    let slowpath = joinpath(@__DIR__, "test_slow_light.jl")
        isfile(slowpath) && withenv("BLP_SKIP_GPU" => "1") do
            Base.include(Module(:TestSlowLight), slowpath)
        end
    end
    if isfile(joinpath(@__DIR__, "..", "src", "montecarlo", "montecarlo.jl"))
        include("test_mc_rng.jl")
        include("test_mc_emissivity.jl")
        include("test_mc_hotcross.jl")
        include("test_mc_scatter.jl")
        include("test_mc_generate.jl")
        include("test_mc_sphere.jl")
        include("test_mc_scatter_sphere.jl")
        isfile(joinpath(@__DIR__, "..", "src", "montecarlo", "grmhd_model.jl")) &&
            include("test_mc_grmhd.jl")
    end
end
