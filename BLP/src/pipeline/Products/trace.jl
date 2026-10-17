# Trajectories and ray fates: the fixed-control trace (window by window on any executor),
# the adaptive trace (host), the two-phase render along stored trajectories (host), and
# the plasma-free shadow tracer.

"""
    Trajectories

Traced camera rays: `states[k]`, the states `[x^μ; p_μ]` of pixel `k`'s ray ordered
source → camera (the camera ray last), and `dls[k]`, the affine step from state `j` to
`j + 1` (`0` after the last), both `Float64`; pixels in camera order (β fastest).
`plan` and `run` as in [`Image`](@ref).
"""
struct Trajectories{P,R}
    states::Vector{Vector{SVector{8,Float64}}}
    dls::Vector{Vector{Float64}}
    plan::P
    run::R
end

Base.show(io::IO, t::Trajectories) =
    print(io, "Trajectories(", length(t.states), " rays, ", sum(length, t.states; init = 0),
          " states)")

"""
    trace(plan; backend = nothing, adaptive = false) -> Trajectories

The trajectories of the plan's camera rays. By default the fixed-control march of the
imaging products on the plan's geodesic lane (step `(distance to the horizon proxy)/C`,
stop on capture, on outbound escape past `escape` or past `R_em`, at most `max_steps`
steps), run window by window ([`TraceProgram`](@ref)) on the executor
`resolve(backend)`; states and steps of a `Float32` lane are widened to `Float64`. With
`adaptive = true` the host-only adaptive solve `trace_geodesic` (`escape =
plan.escape`, `reltol = 1e-9`, `abstol = 1e-11`), which needs a `ThreadedExecutor`.
`MixedLanes` plans are an `ArgumentError`: a trace runs on one lane. A camera ray
without a positive-energy null root is an error.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function trace(plan::Plan; backend = nothing, adaptive::Bool = false)
    t0 = time_ns()
    exec = resolve(backend)
    plan.lanes isa MixedLanes &&
        throw(ArgumentError("trace: a trace runs on one geodesic lane; prepare the plan " *
                            "with Lanes{Float64,Float64,Float64}() or Lanes{Float32,Float64,Float64}()"))
    states, dls = adaptive ? trace_adaptive(host_only(exec, "trace(…; adaptive = true)"), plan) :
                  trace_fixed(exec, plan)
    return Trajectories(states, dls, plan, run_record(exec, plan, t0))
end

# The executor of a host-only product: a ThreadedExecutor, else an ArgumentError.
host_only(exec::ThreadedExecutor, what) = exec
host_only(exec, what) =
    throw(ArgumentError("$what runs on the host only: pass `backend = :cpu`"))

function trace_adaptive(exec, plan::Plan)
    n = npixels(plan)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    prog = AdaptiveTraceProgram(plan.spacetime, CameraRays(plan.spacetime, plan.frame, plan.pixels),
                                plan.escape)
    foreach_item!(exec, prog, 1:n; outs = (states, dls))
    return states, dls
end

# Window after window over the rays still marching, until none is left.
function trace_fixed(exec, plan::Plan)
    T = geodesic_type(plan.lanes)
    n = npixels(plan)
    rays = CameraRays(plan.spacetime, plan.frame, plan.pixels)
    outs = trace_outputs(T, n)
    logs = [SVector{8,T}[] for _ in 1:n]
    hlogs = [T[] for _ in 1:n]
    live = collect(1:n)
    resume = nothing
    while !isempty(live)
        prog = TraceProgram(plan.geodesic_spacetime, plan.geodesic_budget, rays, resume)
        foreach_item!(exec, prog, eachindex(live); outs)
        live, resume = absorb_window!(logs, hlogs, live, window_views(outs)..., resume === nothing)
    end
    return [SVector{8,Float64}.(reverse!(l)) for l in logs],
           [push!(Float64.(reverse!(h)), 0.0) for h in hlogs]
end

# The step vectors and the TraceState vector of a window.
window_views(outs) = (ntuple(k -> outs[k], Val(TRACE_WINDOW + 1)), outs[end])

# Append the window of every queued ray `live[j]` to its logs (camera → source); the rays
# still marching and their states for the next window.
function absorb_window!(logs, hlogs, live, ws::NTuple, state::Vector{TraceState{T}},
                        first::Bool) where {T}
    next = Int[]
    resume = TraceState{T}[]
    for j in eachindex(live)
        r = live[j]
        m = state[j]
        m.ok || error(NO_ROOT)
        first && push!(logs[r], ws[1][j].y)
        for s in 1:(m.done - length(hlogs[r]))
            w = ws[s + 1][j]
            push!(logs[r], w.y)
            push!(hlogs[r], w.h)
        end
        m.live && (push!(next, r); push!(resume, m))
    end
    return next, resume
end

# The two-phase render: every pixel marched along its stored trajectory, on the host.
function render_stored(plan::Plan, path::Trajectories, backend)
    t0 = time_ns()
    exec = host_only(resolve(backend), "render(plan; path)")
    length(path.states) == npixels(plan) ||
        throw(ArgumentError("render(plan; path): $(length(path.states)) trajectories for " *
                            "$(npixels(plan)) pixels"))
    plan.provider === nothing &&
        throw(ArgumentError("imaging needs matter: prepare the plan with `matter = …`"))
    prog = StoredPathProgram(plan.mode, plan.spacetime, plan.provider, plan.params, plan.frame,
                             path.states, path.dls)
    out = Vector{output_type(plan)}(undef, npixels(plan))
    foreach_item!(exec, prog, plan.queue; outs = (out,))
    return Image(plan, out, run_record(exec, plan, t0))
end

"""
    shadow(plan; backend = nothing) -> NamedTuple

The fate of every camera ray of `plan` without plasma ([`ShadowProgram`](@ref)) on the
plan's geodesic lane: the step of the fixed-control march, capture by the spacetime's
rule, escape once the march radius exceeds `plan.escape`, at most `max_steps` steps.
Returns `(fate, steps, null_error, radius, final, plan, run)`, each map
`npix × npix` in the image layout: `fate` 1 captured, 2 escaped, 0 step-limited;
`steps` taken; `null_error` the largest finite `|g^{μν} p_μ p_ν|` along the ray;
`radius` and `final` the coordinate radius and state where the ray ended. `MixedLanes`
plans are an `ArgumentError`. A camera ray without a positive-energy null root is an
error.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function shadow(plan::Plan; backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    plan.lanes isa MixedLanes &&
        throw(ArgumentError("shadow: a shadow runs on one geodesic lane; prepare the plan " *
                            "with Lanes()"))
    T = geodesic_type(plan.lanes)
    n = npixels(plan)
    ys = Vector{SVector{8,T}}(undef, n)
    ms = Vector{SVector{4,Float64}}(undef, n)
    foreach_item!(exec, shadow_program(plan), plan.queue; outs = (ys, ms))
    any(m -> m[1] < 0, ms) && error(NO_ROOT)
    npx = plan.camera.npix
    map2(f) = reshape(map(f, ms), npx, npx)
    return (fate = map2(m -> Int(m[1])), steps = map2(m -> Int(m[2])), null_error = map2(m -> m[3]),
            radius = map2(m -> m[4]), final = reshape(SVector{8,Float64}.(ys), npx, npx),
            plan = plan, run = run_record(exec, plan, t0))
end

"""
    shadow_program(plan) -> ShadowProgram

The [`ShadowProgram`](@ref) of `plan`: its geodesic-lane spacetime, the step divisor and
step limit of its budget with infinite escape and emission radii (capture decides the
stop), the escape radius `plan.escape` on the lane, and its camera rays.
"""
function shadow_program(plan::Plan)
    b = plan.geodesic_budget
    T = typeof(b.C)
    return ShadowProgram(plan.geodesic_spacetime, MarchBudget{T}(b.C, T(Inf), T(Inf), b.nmax),
                         T(plan.escape), CameraRays(plan.spacetime, plan.frame, plan.pixels))
end
