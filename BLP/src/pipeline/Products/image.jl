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

function Image(plan::Plan, out::AbstractVector{<:Real}, run)
    n = plan.camera.npix
    return Image(reshape(Vector{Float64}(out), n, n), zeros(n, n), zeros(n, n), zeros(n, n),
                 plan, run)
end

function Image(plan::Plan, out::AbstractVector{<:SVector{4}}, run)
    n = plan.camera.npix
    I, Q, U, V = ntuple(k -> reshape([o[k] for o in out], n, n), 4)
    return Image(I, Q, U, V, plan, run)
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
    render(plan; backend = nothing) -> Image
    render(plan; path::Trajectories, backend = nothing) -> Image

The image of `plan`: every pixel's camera ray marched with the fused fixed-control
march of `plan.mode` (geodesic, plasma sample and transfer in one pass), queued in
`plan.queue` order on the executor `resolve(backend)` (`nothing`: the session default).
Under `MixedLanes` the rescue pixels are marched on the `Float64` lane, queued first,
and the rest on the `Float32` lane, in one queue. With `path`, a trace of the same plan
([`trace`](@ref)), each pixel is marched along its stored trajectory instead (the
two-phase path; host only: `backend` must resolve to a `ThreadedExecutor`).

Determinism: bitwise per item, whatever the executor, device count, chunking or queue
order.
"""
function render(plan::Plan; backend = nothing, path = nothing)
    path === nothing || return render_stored(plan, path, backend)
    t0 = time_ns()
    exec = resolve(backend)
    out = Vector{output_type(plan)}(undef, npixels(plan))
    march_frame!(exec, plan, out)
    return Image(plan, out, run_record(exec, plan, t0))
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
    foreach_item!(exec, imaging_program(plan; kw...), plan.queue; outs = (out,))

function march_frame!(exec, plan::Plan, ::MixedLanes, out; kw...)
    q = mixed_partition(plan.queue, plan.rescue)
    rescue = imaging_program(plan, plan.spacetime, plan.budget; kw...)
    bulk = imaging_program(plan; kw...)
    return foreach_item!(exec, (rescue => q.rescue, bulk => q.bulk); outs = (out,))
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
