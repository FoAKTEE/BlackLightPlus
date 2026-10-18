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
    chunk_hint(prog) -> Union{Int,Nothing}

The most items a chunk of the program `prog` holds when the executor sizes its chunks
itself (`chunk = :auto`); `nothing` (the default) leaves the executor's rule
([`chunk_for`](@ref)). An explicit chunk ignores it. Results never depend on it.
"""
chunk_hint(prog) = nothing

# One launch of the device program `dprog`, made from `prog`, over the first `n` entries
# of the device item buffer `items` into the device outputs `outs`, on `lane`; returns
# after the device has finished.
function launch!(lane::PoolLane, prog, dprog, items, n::Integer, outs::Tuple)
    prepare_launch!(lane.dev, prog)
    kern = _foreach!(lane.backend)
    wg = workgroupsize(prog)
    if wg === nothing
        kern(dprog, items, n, outs...; ndrange = n)
    else
        kern(dprog, items, n, outs...; ndrange = n, workgroupsize = min(wg, n))
    end
    KernelAbstractions.synchronize(lane.backend)
    return nothing
end
