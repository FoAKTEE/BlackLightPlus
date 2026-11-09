# The only kernel of the package. A program is an `isbits`-after-`Adapt` functor called
# per item as `prog(item, i, outs...)`; it writes its outputs for `item` at `outs[k][i]`
# itself (or nothing, when it writes through atomics). The kernel specialises on the
# program type, so each program compiles to its own device code. `items` is a range or a
# device vector of `Int32` or `Int` (`device_items`); the program gets each item as `Int`.

KernelAbstractions.@kernel function _foreach!(prog, @Const(items), n, outs...)
    i = @index(Global)
    @inbounds i <= n && prog(Int(items[i]), i, outs...)
end

"""
    workgroupsize(prog) -> Union{Int,Nothing}

Workgroup (CUDA block) size for launches of the program `prog`; `nothing` (the default)
lets the backend choose. Programs override it for occupancy: the polarized march, whose
register footprint limits residency, runs at 64. Results never depend on it.
"""
workgroupsize(prog) = nothing

"""
    register_cap(prog) -> Union{Int,Nothing}

The most registers per thread that a device compiles the program `prog` with; `nothing`
(the default) leaves the compiler's limit. A cap reaches only register allocation and the
launch configuration: the PTX of a capped program is the uncapped PTX with a `.maxnreg`
directive, so results never depend on it. A launch passes it to
[`launch_capped!`](@ref), which the CUDA extension honours and every other device
ignores.
"""
register_cap(prog) = nothing

"""
    chunk_hint(prog) -> Union{Int,Nothing}

The most items a chunk of the program `prog` holds when the executor sizes its chunks
itself (`chunk = :auto`); `nothing` (the default) leaves the executor's rule
([`chunk_for`](@ref)). An explicit chunk ignores it. Results never depend on it.
"""
chunk_hint(prog) = nothing

"""
    launch_capped!(dev, kernel, args::Tuple, n, wg, cap)

Launch the KernelAbstractions `kernel` over `n` items with the arguments `args` and the
workgroup size `wg` (`nothing`: the backend's choice), compiled with at most `cap`
registers per thread ([`register_cap`](@ref)). On a `CUDADevice` (CUDA extension) the
launch is the backend's own with the cap added to the compilation; every other device
ignores the cap. Returns before the device has finished.
"""
function launch_capped!(::AbstractDevice, kernel, args::Tuple, n, wg, cap)
    if wg === nothing
        kernel(args...; ndrange = n)
    else
        kernel(args...; ndrange = n, workgroupsize = wg)
    end
    return nothing
end

# One launch of the device program `dprog`, made from `prog`, over the first `n` entries
# of the device item buffer `items` into the device outputs `outs`, on `lane`; returns
# after the device has finished.
function launch!(lane::PoolLane, prog, dprog, items, n::Integer, outs::Tuple)
    prepare_launch!(lane.dev, prog)
    kern = _foreach!(lane.backend)
    wg = workgroupsize(prog)
    cap = register_cap(prog)
    if cap !== nothing
        launch_capped!(lane.dev, kern, (dprog, items, n, outs...), n,
                       wg === nothing ? nothing : min(wg, n), cap)
    elseif wg === nothing
        kern(dprog, items, n, outs...; ndrange = n)
    else
        kern(dprog, items, n, outs...; ndrange = n, workgroupsize = min(wg, n))
    end
    KernelAbstractions.synchronize(lane.backend)
    return nothing
end
