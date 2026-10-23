# The CUDA extension: it loads with `CUDA` and supplies the `CUDADevice` methods; without
# a functional device an explicit `:gpu` throws and `:auto` falls back to threads with the
# reason logged. Skipped when the CUDA package is not installed in the environment.
using Test, BlackLightPlus
using BlackLightPlus.Execution: resolve, ThreadedExecutor, PoolExecutor, CUDADevice,
                     BackendUnavailable, cuda_loaded, functional, select_devices,
                     stage_alloc, backend
const Ex = BlackLightPlus.Execution

const CUDA_ID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const HAS_CUDA = Base.locate_package(CUDA_ID) !== nothing
HAS_CUDA && Base.require(CUDA_ID)

@testset "CUDA extension" begin
    if !HAS_CUDA
        @test_skip "the CUDA package is not installed in this environment"
    else
        ext = Base.get_extension(BlackLightPlus, :BlackLightPlusCUDAExt)
        @test ext !== nothing && cuda_loaded()
        @test which(stage_alloc, (CUDADevice, Type{Float64}, Int)).module === ext
        @test which(backend, (CUDADevice,)).module === ext
        if functional(CUDADevice)
            @test resolve(:gpu) isa PoolExecutor
            @test !isempty(select_devices(; fallback = true))
        else
            @test_throws BackendUnavailable resolve(:gpu)
            err = try resolve(:gpu); nothing catch e; e end
            @test occursin("CUDA_VISIBLE_DEVICES", sprint(showerror, err))
            @test_throws BackendUnavailable PoolExecutor([0])
            @test !functional(CUDADevice(0)) && isempty(select_devices())
            withenv("BLP_BACKEND" => nothing) do
                Ex.reset_backend!()
                a = @test_logs (:info, r"no functional device") resolve(:auto)
                @test a == ThreadedExecutor()
            end
        end
        Ex.reset_backend!()
    end
end
