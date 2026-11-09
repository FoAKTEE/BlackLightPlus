# Images: the full-frame render, the coordinate-list render, and the record every product
# result carries about where and how it was made.

"""
    Image

A rendered image: the Stokes planes `I`, `Q`, `U`, `V` (`Matrix{Float64}`, cgs
specific intensity `ν_obs³` times the invariant, laid out `[iβ, iα]` with β along the
first axis; `Q = U = V = 0` for `StokesI`), the `plan` it was rendered from and `run`,
the record of the run: `executor` (`describe` of the executor), `lanes` (the plan's
precision lanes), `version` (the package version) and `seconds` (wall time). A pixel
whose camera ray has no positive-energy null root (a camera inside an ergoregion) is
`NaN`.
"""
struct Image{P,R}
    I::Matrix{Float64}
    Q::Matrix{Float64}
    U::Matrix{Float64}
    V::Matrix{Float64}
    plan::P
    run::R
end

# A Stokes I output vector becomes the I plane itself when it is a `Vector{Float64}`.
# The zero planes and the split of a Stokes vector into planes run on Julia threads.
function Image(plan::Plan, out::AbstractVector{<:Real}, run)
    n = plan.camera.npix
    I = reshape(convert(Vector{Float64}, out), n, n)
    Q, U, V = filled_maps(0.0, 3, (n, n))
    return Image(I, Q, U, V, plan, run)
end

function Image(plan::Plan, out::AbstractVector{<:SVector{4}}, run)
    n = plan.camera.npix
    length(out) == n * n ||
        throw(DimensionMismatch("Image: $(length(out)) pixels for a $n × $n camera"))
    planes = ntuple(_ -> Matrix{Float64}(undef, n, n), 4)
    host_epilogue!(Split(out), planes, length(out))
    return Image(planes..., plan, run)
end

# The image of a run whose output `out` came from `image_output` with `into`, an earlier
# `Image`: an IQUV march wrote the four planes through a `StokesPlanes`; a Stokes I march
# wrote `into.I`, and the zero planes are filled again.
image_result(plan::Plan, out, run, ::Nothing) = Image(plan, out, run)
function image_result(plan::Plan, out, run, into)
    out isa StokesPlanes || fill_maps!(0.0, [into.Q, into.U, into.V])
    return Image(into.I, into.Q, into.U, into.V, plan, run)
end

"""
    StokesPlanes(I, Q, U, V)

The planes of a Stokes-vector image as one vector of `SVector{4,Float64}` over their
items: element `i` reads `(I[i], Q[i], U[i], V[i])`, and storing a Stokes vector there
writes its components to the four planes. An `IQUV` run stores its Stokes vectors through
it into caller-supplied planes: the lanes of a pool as they copy a chunk back, Julia threads
block by block ([`staged_planes!`](@ref)).
"""
struct StokesPlanes <: AbstractVector{SVector{4,Float64}}
    I::Vector{Float64}
    Q::Vector{Float64}
    U::Vector{Float64}
    V::Vector{Float64}
    function StokesPlanes(I, Q, U, V)
        length(I) == length(Q) == length(U) == length(V) ||
            throw(DimensionMismatch("StokesPlanes: planes of lengths $(map(length, (I, Q, U, V)))"))
        return new(I, Q, U, V)
    end
end
Base.size(p::StokesPlanes) = size(p.I)
Base.IndexStyle(::Type{StokesPlanes}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(p::StokesPlanes, i::Int) =
    SVector(p.I[i], p.Q[i], p.U[i], p.V[i])
Base.@propagate_inbounds function Base.setindex!(p::StokesPlanes, v, i::Int)
    s = convert(SVector{4,Float64}, v)
    p.I[i] = s[1]; p.Q[i] = s[2]; p.U[i] = s[3]; p.V[i] = s[4]
    return p
end

# The per-pixel output of an image run: a new vector, or with `into` (an `Image` of as many
# pixels) its I plane for Stokes I and its four planes as a `StokesPlanes` for IQUV.
image_output(exec, plan::Plan, ::Nothing) = Vector{output_type(plan)}(undef, npixels(plan))
function image_output(exec, plan::Plan, into)
    n = plan.camera.npix
    T = output_type(plan)
    (T === Float64 || T === SVector{4,Float64}) ||
        throw(ArgumentError("render: `into` takes Float64 or SVector{4,Float64} pixels, " *
                            "not the plan's $T"))
    m = claim_into(into, :render, (:I, :Q, :U, :V), (), 0, (n, n))
    T === Float64 && return vec(m.I)
    return StokesPlanes(vec(m.I), vec(m.Q), vec(m.U), vec(m.V))
end

# The parts `(prog => items, …)` of an image run into `out`, as `foreach_item!` runs them;
# Stokes vectors into a `StokesPlanes` on Julia threads through [`staged_planes!`](@ref).
run_image!(exec, parts::Tuple, out) = foreach_item!(exec, parts; outs = (out,))
run_image!(exec::ThreadedExecutor, parts::Tuple, out::StokesPlanes) =
    staged_planes!(exec, parts, out)

# Items of a block that a worker marches into its own Stokes vectors before storing them
# in the planes: 8 KiB per worker.
const STAGE_ITEMS = 256

"""
    staged_planes!(exec::ThreadedExecutor, parts, P::StokesPlanes) -> P

The parts `(prog => items, …)` of an `IQUV` image on Julia threads into the planes `P`:
the chunks that `foreach_item!` cuts, each marched in blocks of at most `STAGE_ITEMS`
items into the worker's own vector of Stokes vectors, whose components are then stored in
the planes. On threads a program writes its output itself and is compiled for the
output's type, and a program compiled for `StokesPlanes` contracts its arithmetic
otherwise; the worker's vector has the type of the output of the call without `into`, so
the same compiled program runs and the planes hold that call's bits.
"""
function staged_planes!(exec::ThreadedExecutor, parts::Tuple, P::StokesPlanes)
    ps = map(p -> (p.first, p.second::AbstractVector{<:Integer}), parts)
    n = sum(p -> length(p[2]), ps; init = 0)
    c = Execution.resolve_chunk!(exec, n; hint = Execution.parts_hint(ps),
                                 bytes = Execution.item_bytes((P,)))
    chunks = Execution.part_chunks(ps, c)
    nw = min(Threads.nthreads(), length(chunks))
    bufs = Channel{Vector{SVector{4,Float64}}}(nw)
    foreach(_ -> put!(bufs, Vector{SVector{4,Float64}}(undef, min(c, STAGE_ITEMS))), 1:nw)
    Execution.run_workers!(nw, chunks) do _, (p, r), _
        prog, q = ps[p]
        buf = take!(bufs)
        try
            stage_range!(prog, q, r, buf, P)
        finally
            put!(bufs, buf)
        end
    end
    return P
end

# The items `q[r]` of `prog` in blocks of `length(buf)`: a block marched into `buf` (the
# block's `s`-th item at slot `s`, an integer of the item's type), then stored in `P`.
function stage_range!(prog, q, r::UnitRange{Int}, buf::Vector{SVector{4,Float64}},
                      P::StokesPlanes)
    for b0 in first(r):length(buf):last(r)
        b = b0:min(b0 + length(buf) - 1, last(r))
        for j in b
            item = @inbounds q[j]
            prog(item, oftype(item, j - b0 + 1), buf)
        end
        for j in b
            @inbounds P[q[j]] = buf[j - b0 + 1]
        end
    end
    return nothing
end

Base.show(io::IO, img::Image) =
    print(io, "Image(", size(img.I, 1), "×", size(img.I, 2), ", ", nameof(typeof(img.plan.mode)),
          ", ", img.run.executor.kind, ", ", round(img.run.seconds; digits = 3), " s)")

"""
    run_record(exec, plan, t0) -> NamedTuple

The `run` field of a result: `(executor = describe(exec), lanes = plan.lanes, version,
seconds)`, the wall time counted from `t0` (`time_ns()`).
"""
run_record(exec, plan::Plan, t0::Integer) =
    (executor = describe(exec), lanes = plan.lanes, version = PACKAGE_VERSION,
     seconds = (time_ns() - t0) / 1e9)

"""
    render(plan; backend = nothing, into = nothing) -> Image
    render(plan; path::Trajectories, backend = nothing, into = nothing) -> Image

The image of `plan`: every pixel's camera ray marched with the fused fixed-control
march of `plan.mode` (geodesic, plasma sample and transfer in one pass), queued in
`plan.queue` order on the executor `resolve(backend)` (`nothing`: the session default).
Under `MixedLanes` the rescue pixels are marched on the `Float64` lane, queued first,
and the rest on the `Float32` lane, in one queue. With `path`, a trace of the same plan
([`trace`](@ref)), each pixel is marched along its stored trajectory instead (the
two-phase path; host only: `backend` must resolve to a `ThreadedExecutor`). `into`, an
`Image` of as many pixels (`Float64` or `IQUV` pixels), is overwritten: the march writes its
planes (the zero planes of a Stokes I image filled again; an `IQUV` march on Julia threads
through blocks of Stokes vectors per worker, [`staged_planes!`](@ref)) and the returned
image shares them, bitwise the image without `into`; its arrays are dropped from the
device pools' residency caches first ([`Execution.invalidate!`](@ref)).

Determinism: bitwise per item, whatever the executor, device count, chunking or queue
order.
"""
function render(plan::Plan; backend = nothing, path = nothing, into = nothing)
    path === nothing || return render_stored(plan, path, backend, into)
    t0 = time_ns()
    exec = resolve(backend)
    out = image_output(exec, plan, into)
    march_frame!(exec, plan, out)
    return image_result(plan, out, run_record(exec, plan, t0), into)
end

"""
    render(plan, coords; backend = nothing) -> Matrix{Float64}

The Stokes vectors of the camera rays at image-plane coordinates `coords` (a vector of
`(α, β)` tuples or a `CoordList`, in M): a `4 × n` matrix, column `j` for
`coords[j]`, rows `I, Q, U, V` (`Q = U = V = 0` for `StokesI`), marched on the plan's
geodesic lane. The rescue set of `MixedLanes` is tied to the pixel grid, so a
`MixedLanes` plan is an `ArgumentError` here.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function render(plan::Plan, coords; backend = nothing)
    plan.lanes isa MixedLanes &&
        throw(ArgumentError("render(plan, coords): MixedLanes rescues pixels of the camera " *
                            "grid; prepare the plan with Lanes() for a coordinate list"))
    exec = resolve(backend)
    gen = coords isa CoordList ? coords : CoordList(coords)
    out = Vector{output_type(plan)}(undef, length(gen.α))
    foreach_item!(exec, imaging_program(plan; gen), eachindex(out); outs = (out,))
    return stokes_matrix(out)
end

# The full frame of `plan` into `out`, on the plan's lanes; `kw` override the mode and
# the parameter set of the programs.
march_frame!(exec, plan::Plan, out; kw...) = march_frame!(exec, plan, plan.lanes, out; kw...)

march_frame!(exec, plan::Plan, ::Lanes, out; kw...) =
    run_image!(exec, (imaging_program(plan; kw...) => plan.queue,), out)

function march_frame!(exec, plan::Plan, ::MixedLanes, out; kw...)
    q = mixed_partition(plan.queue, plan.rescue)
    rescue = imaging_program(plan, plan.spacetime, plan.budget; kw...)
    bulk = imaging_program(plan; kw...)
    return run_image!(exec, (rescue => q.rescue, bulk => q.bulk), out)
end

# Per-item Stokes values as a 4 × n matrix (rows Q, U, V zero for a scalar intensity).
function stokes_matrix(out::AbstractVector)
    M = zeros(Float64, 4, length(out))
    @inbounds for j in eachindex(out)
        o = out[j]
        for k in eachindex(o)
            M[k, j] = o[k]
        end
    end
    return M
end
