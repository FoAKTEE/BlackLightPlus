# The CUDA extension's precompile workload. Its program writes each item as a `Float64` at
# the item's place, on threads and, on a functional device, on the GPU. In a session that
# loads the package before CUDA, the extension's package image holds natively compiled
# methods of the GPU compiler (GPUCompiler, the abstract interpreter, CUDACore) that are
# valid in the world where CUDA compiles device code, so the session's first device
# compilation does not compile them. Skipped when the CUDA package is not installed in the
# environment, and the image check when the session instruments code (coverage, allocation
# logs), which never uses image code.
using Test, BlackLightPlus
using BlackLightPlus.Execution: ThreadedExecutor, CUDADevice, foreach_item!, functional, resolve

const CUDA_ID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
const HAS_CUDA = Base.locate_package(CUDA_ID) !== nothing
const CU = HAS_CUDA ? Base.require(CUDA_ID) : nothing

# Native code instances of the methods of the modules `mods` that come from the package
# image with build id `bid` and are valid in the world `w`.
function image_code(mods, bid, w)
    n = 0
    for mod in mods, nm in names(mod; all = true)
        isdefined(mod, nm) || continue
        f = getfield(mod, nm)
        f isa Function || continue
        for m in methods(f), mi in Base.specializations(m)
            m.module === mod || continue
            ci = isdefined(mi, :cache) ? mi.cache : nothing
            while ci !== nothing
                ci.owner === nothing && ci.min_world <= w <= ci.max_world &&
                    Base.object_build_id(ci) == bid && (n += 1)
                ci = isdefined(ci, :next) ? ci.next : nothing
            end
        end
    end
    return n
end

@testset "CUDA extension precompile workload" begin
    if !HAS_CUDA
        @test_skip "the CUDA package is not installed in this environment"
    else
        ext = Base.get_extension(BlackLightPlus, :BlackLightPlusCUDAExt)
        @test ext !== nothing
        out = zeros(5)
        foreach_item!(ThreadedExecutor(), ext.Iota(), [4, 1, 3]; outs = (out,))
        @test out == [1.0, 0.0, 3.0, 4.0, 0.0]
        if functional(CUDADevice)
            g = zeros(5)
            foreach_item!(resolve(:gpu), ext.Iota(), [4, 1, 3]; outs = (g,))
            @test g == out
        end
        instrumented = Base.JLOptions().code_coverage != 0 || Base.JLOptions().malloc_log != 0
        if instrumented
            @test_skip "the session instruments code"
        else
            cc = CU.CUDACore
            w = min(cc._initialization_world[], Base.get_world_counter())
            n = image_code((cc.GPUCompiler, cc.GPUCompiler.CC, cc), Base.module_build_id(ext), w)
            @test n > 0
        end
    end
end
