# The children of a cascade: where a program spawns them (`ChildBuffer`, fixed slots
# taken by an atomic counter, on host and device arrays) and how the next layer reads
# them (`Generation`). A child's stream is keyed by its genealogy, so the children are
# stored in arrival order, which is not deterministic, while their set is.

using ..Numerics: MCRng, MCStreamLayout, child_rng

"""
    ChildBuffer(T, capacity, layout::MCStreamLayout)

Scattered children of one layer of the layered scattering loop, as a structure of
arrays: `photons::Vector{T}` and their streams `rngs::Vector{MCRng}`, `capacity` slots,
and `count[1]`, the atomic slot counter of spawn attempts. [`spawn!`](@ref) takes the
next slot from the counter and writes the child and its genealogy stream
`child_rng(rng, event, layout)` if the slot exists; a child beyond the capacity is
counted and dropped, and [`overflow`](@ref) reports how many. Children are therefore
stored in arrival order, which is not deterministic; their set is.

Policy (the driver's): a layer whose buffer overflowed is rerun from the same inputs
with a buffer of capacity at least `count[1]` (the rerun spawns the identical children,
their streams being genealogy-keyed), so no child is ever lost. Read the children with
[`children`](@ref) (copies, safe to hand to the next layer's program) and reset the
buffer with `empty!`. Inside a pool program each device writes a twin buffer of the same
capacity; [`gather!`](@ref) appends the stored children of every twin and adds the
attempt counts. In a program run by [`foreach_cascade!`](@ref) the buffer is the slot
that the cascade fills per chunk and generation: only its element type and its layout
are read.
"""
struct ChildBuffer{P,R,C,W}
    photons::P
    rngs::R
    count::C
    layout::MCStreamLayout
    twins::W
end
ChildBuffer(::Type{T}, capacity::Integer, layout::MCStreamLayout) where {T} =
    ChildBuffer(Vector{T}(undef, capacity), Vector{MCRng}(undef, capacity), zeros(Int64, 1),
                layout, Twins())

Adapt.adapt_structure(to, b::ChildBuffer) =
    ChildBuffer(Adapt.adapt(to, b.photons), Adapt.adapt(to, b.rngs), Adapt.adapt(to, b.count),
                b.layout, nothing)
function Adapt.adapt_structure(to::DeviceAdaptor, b::ChildBuffer)
    bk = backend(to.dev)
    cap = length(b.photons)
    t = ChildBuffer(KernelAbstractions.allocate(bk, eltype(b.photons), cap),
                    KernelAbstractions.allocate(bk, MCRng, cap), zeroed(to.dev, b.count),
                    b.layout, nothing)
    return register!(b.twins, to.dev, t)
end

"""
    spawn!(buf::ChildBuffer, child, rng::MCRng, event) -> nothing

Store `child` with the stream `child_rng(rng, event, buf.layout)` (`rng` the parent's
stream, `event` the 0-based spawn ordinal along its march) in the next free slot, or
count it as overflow. Atomic; device-safe.
"""
@inline function spawn!(b::ChildBuffer, child, rng::MCRng, event::Integer)
    slot = Atomix.@atomic b.count[1] += Int64(1)
    if slot <= length(b.photons)
        @inbounds b.photons[slot] = child
        @inbounds b.rngs[slot] = child_rng(rng, event, b.layout)
    end
    return nothing
end

"Slots of the buffer."
capacity(b::ChildBuffer) = length(b.photons)

"Children stored in the buffer."
Base.length(b::ChildBuffer) = min(Int(b.count[1]), length(b.photons))

"Children counted but not stored for lack of slots."
overflow(b::ChildBuffer) = Int(b.count[1]) - length(b)

"""
    children(buf::ChildBuffer) -> (photons::Vector, rngs::Vector)

Copies of the stored children and their streams, in slot order.
"""
function children(b::ChildBuffer)
    pending(b) && error("children: device twins pending; call gather!(buffer) first")
    n = length(b)
    return b.photons[1:n], b.rngs[1:n]
end

"Forget every child and twin: the counter returns to zero."
function Base.empty!(b::ChildBuffer)
    take_twins!(b.twins)
    b.count[1] = 0
    return b
end

function gather!(b::ChildBuffer)
    n_att = Int(b.count[1])
    stored = length(b)
    for (dev, t) in take_twins!(b.twins)
        nd = Int(Array(t.count)[1])
        sd = min(nd, length(t.photons))
        if stored + sd > length(b.photons)
            resize!(b.photons, stored + sd)
            resize!(b.rngs, stored + sd)
        end
        if sd > 0
            copyto!(b.photons, stored + 1, Array(view(t.photons, 1:sd)), 1, sd)
            copyto!(b.rngs, stored + 1, Array(view(t.rngs, 1:sd)), 1, sd)
        end
        stored += sd
        n_att += nd
        foreach(a -> free!(dev, a), (t.photons, t.rngs, t.count))
    end
    b.count[1] = n_att
    return b
end

"""
    Generation(photons, rngs)

One generation of a cascade as a layer program reads it: member `j` is `photons[j]` on
its stream `rngs[j]`. [`foreach_cascade!`](@ref) replaces the `Generation` of its layer
program by the children the previous generation of the chunk stored.
"""
struct Generation{P,R}
    photons::P
    rngs::R
end
Adapt.@adapt_structure Generation

Base.length(g::Generation) = length(g.photons)
