# Pinned staging of the DevicePool on CUDA (src/gpu_pool.jl, lane_stage / close): a
# staging buffer that a lane replaces (a larger chunk) or that `close(pool)` drops must be
# released deterministically. A buffer pinned by `CUDA.pin` is unregistered by a GC
# finalizer that takes CUDA's pin lock; when that finalizer runs while another lane holds
# the lock, the unregistration fails ("task switch in finalizer"), the host range stays
# registered after the Vector is freed, and a later pin of reused memory throws
# ERROR_HOST_MEMORY_ALREADY_REGISTERED. The stress below grows and drops staging buffers
# on 8 lanes per device with collections in between, over repeated pool lifetimes, then
# renders through a fresh pool. GPU only (skipped without a functional CUDA device).
using Test, KernelAbstractions, CUDA
using BlackLightPlus
@isdefined(DevicePool) || include(joinpath(@__DIR__, "..", "src", "gpu_pool.jl"))

@testset "DevicePool pinned staging (CUDA)" begin
    if !CUDA.functional()
        @test_skip "CUDA not functional"
    else
        devs = collect(0:min(length(CUDA.devices()), 2) - 1)
        err = nothing
        try
            for trial in 1:12
                pool = DevicePool(devs; lanes = 8, chunk = 1)
                foreach_chunk(pool, 1:64) do lane, i
                    for n in (1 << 10, 1 << 14, 1 << 17) .+ i   # grow: drop the smaller buffer
                        st = lane_stage(lane, :stress, Float64, n)
                        d = lane_buffer(lane, :stress, Float64, n)
                        st[1:n] .= i
                        copyto!(d, 1, st, 1, n)
                        copyto!(st, 1, d, 1, n)
                        @assert st[n] == i
                        i % 4 == 0 && GC.gc(false)
                    end
                end
                close(pool)
                GC.gc()
            end
        catch e
            err = e
        end
        @test err === nothing
        # every staging buffer pinned by the pool is released when it is dropped
        pool = DevicePool(devs; lanes = 2, chunk = 4)
        foreach_chunk(pool, 1:8) do lane, i
            lane_stage(lane, :a, Float64, 100)
            lane_stage(lane, :a, Float64, 10_000)         # replaces the 100-element buffer
        end
        close(pool)
        @test all(lane -> isempty(lane.stage), pool.lanes)
    end
end
