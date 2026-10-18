# The per-item programs the executors run. A program is the one place a mode, a
# spacetime, a provider, a parameter set and a ray source meet; it is called as
# `prog(item, i, outs...)` and writes the outputs of `item` at `outs[k][i]`. Every
# program except the two host-only ones at the end is `isbits` after `Adapt.adapt` and
# allocates nothing. The call operators carry no inlining hint: a program body is
# compiled as a function of its own, never into the executor's item loop, so the
# threaded executor and the host-device pool run the same floating-point contractions.

"""
    invalid(T) -> T

The value written for an item whose ray has no positive-energy null root at the camera:
`NaN` in every component (value and partials of a `Dual`).
"""
@inline invalid(::Type{T}) where {T<:Real} = T(NaN)
@inline invalid(::Type{SVector{K,T}}) where {K,T} = SVector{K,T}(ntuple(_ -> invalid(T), Val(K)))

# The initial state of `item`'s ray, compiled as a function of its own: the ray's
# arithmetic (and the floating-point contractions the compiler chooses in it) does not
# depend on the program it feeds, so a pixel's ray is the one `camera_ray` makes on the
# host, whichever program, executor or chunk marches it.
@noinline camera_state(rays, item) = ray(rays, item)

# The camera state on the lane of a fixed-control march: rays are made in Float64 and
# narrowed element by element.
@inline lane_state(::SpacetimeMarch{S,T}, y::SVector{8}) where {S,T} = SVector{8,T}(y)

"""
    ImagingProgram(mode, st, control, provider, params, frame, rays)

One pixel of an image: item `item` gets its camera ray from `rays` (a
[`CameraRays`](@ref) source, made in `Float64` on the transfer spacetime), narrowed to
the lane of `control` (a `SpacetimeMarch` on the geodesic lane), and
`out[i] = march(mode, y0, prog)` reads `st` (the `Float64` spacetime of the transfer),
`control`, `provider`, `params` and `frame`. A ray without a positive-energy null root
at the camera gets [`invalid`](@ref). Workgroup size 64 for `IQUV` (register-limited
occupancy), the backend's choice for `StokesI`.
"""
struct ImagingProgram{M,S,C,P,T,F,R} <: Function
    mode::M
    st::S
    control::C
    provider::P
    params::T
    frame::F
    rays::R
end
Adapt.@adapt_structure ImagingProgram

function (p::ImagingProgram)(item, i, out)
    y0, ok = camera_state(p.rays, item)
    @inbounds out[i] = ok ? march(p.mode, lane_state(p.control, y0), p) : invalid(eltype(out))
    return nothing
end

workgroupsize(::ImagingProgram{IQUV}) = 64

"""
    imaging_program(plan; mode = plan.mode, params = plan.params, gen = plan.pixels)
    imaging_program(plan, st_g, budget_g; mode, params, gen)

The [`ImagingProgram`](@ref) of `plan` over the image-coordinate generator `gen` (the
plan's pixel grid, a `BlockGrid` or a `CoordList`), marching the geodesic on
`plan.geodesic_spacetime` with `plan.geodesic_budget`, or on `st_g` with `budget_g`
(the `Float64` rescue lane of `MixedLanes`: `plan.spacetime`, `plan.budget`).
"""
imaging_program(plan::Plan; kw...) =
    imaging_program(plan, plan.geodesic_spacetime, plan.geodesic_budget; kw...)

function imaging_program(plan::Plan, st_g, budget_g; mode = plan.mode, params = plan.params,
                         gen = plan.pixels)
    plan.provider === nothing &&
        throw(ArgumentError("imaging needs matter: prepare the plan with `matter = …`"))
    params === nothing &&
        throw(ArgumentError("imaging needs a parameter set: prepare the plan from a RunConfig"))
    return ImagingProgram(mode, plan.spacetime, SpacetimeMarch(st_g, budget_g), plan.provider,
                          params, plan.frame, CameraRays(plan.spacetime, plan.frame, gen))
end

# ---------------------------------------------------------------- trajectories

"""
Steps a trace window holds. A window's outputs are `TRACE_WINDOW + 2` arrays, which keeps
the program call under the argument count at which Julia boxes its arguments.
"""
const TRACE_WINDOW = 20

"""
    TraceStep{T}(y, h)

One logged step of a trace window: the state `y` reached and the affine step `h` that
reached it (`0` for the window's start state).
"""
struct TraceStep{T}
    y::SVector{8,T}
    h::T
end

"""
    TraceState{T}(y, r_prev, done, live, ok)

Where a traced ray stands after a window: state `y`, the radius `r_prev` of the last
march decision, `done` steps taken in all, `live` (neither stopped nor at the step
limit) and `ok` (the camera ray had a positive-energy null root).
"""
struct TraceState{T}
    y::SVector{8,T}
    r_prev::T
    done::Int
    live::Bool
    ok::Bool
end

"""
    TraceProgram(st, budget, rays, resume)

One window of a fixed-control trace on the geodesic lane: `march_control(st, …,
budget)` decides each step and `rk4_step` takes it, for at most `TRACE_WINDOW` steps and
at most `budget.nmax` in all. The window starts at the camera ray of `rays` (narrowed
to the lane) when `resume === nothing`, else at `resume[item]`, a [`TraceState`](@ref).
Called as `prog(item, i, w_0, …, w_K, state)` with `K = TRACE_WINDOW`: `w_0[i]` is the
start state, `w_s[i]` the [`TraceStep`](@ref) `s` of the window, `state[i]` the ray's
`TraceState` after the window (slots past the window's last step are not written).
Successive windows continue the march exactly, so the trajectory is the one a single
march takes.
"""
struct TraceProgram{S,T,R,Q} <: Function
    st::S
    budget::MarchBudget{T}
    rays::R
    resume::Q
end
Adapt.@adapt_structure TraceProgram

@inline function trace_start(p::TraceProgram{S,T,R,Nothing}, item) where {S,T,R}
    y, ok = camera_state(p.rays, item)
    return SVector{8,T}(y), T(Inf), 0, ok
end
@inline function trace_start(p::TraceProgram, item)
    r = @inbounds p.resume[item]
    return r.y, r.r_prev, r.done, true
end

# `outs[s][i] = v` for a run-time slot `s`, by one branch per slot with a constant tuple
# index: indexing a tuple of host arrays at run time would put the tuple on the heap.
@generated function store_slot!(outs::NTuple{K,Any}, s, i, v) where {K}
    stores = [:(s == $k && (@inbounds outs[$k][i] = v; return nothing)) for k in 1:K]
    return Expr(:block, stores..., :(return nothing))
end

function (p::TraceProgram{S,T})(item, i, outs::Vararg{Any,N}) where {S,T,N}
    ws = ntuple(k -> outs[k], Val(TRACE_WINDOW + 1))
    y, r_prev, done, ok = trace_start(p, item)
    @inbounds first(ws)[i] = TraceStep{T}(y, zero(T))
    n = 0
    stop = !ok
    nmax = p.budget.nmax
    while !stop && n < TRACE_WINDOW && done + n < nmax
        m = march_control(p.st, y, r_prev, p.budget)
        if m.stop
            stop = true
        else
            r_prev = m.R
            y = rk4_step(p.st, y, m.h)
            n += 1
            store_slot!(ws, n + 1, i, TraceStep{T}(y, m.h))
        end
    end
    @inbounds outs[N][i] = TraceState{T}(y, r_prev, done + n, !stop && done + n < nmax, ok)
    return nothing
end

"""
    trace_outputs(T, n) -> Tuple

The output arrays of a [`TraceProgram`](@ref) window over `n` items on the lane `T`:
`TRACE_WINDOW + 1` `TraceStep{T}` vectors and the `TraceState{T}` vector.
"""
trace_outputs(::Type{T}, n::Integer) where {T} =
    (ntuple(_ -> Vector{TraceStep{T}}(undef, n), TRACE_WINDOW + 1)...,
     Vector{TraceState{T}}(undef, n))

# ---------------------------------------------------------------- shadow

"""
    ShadowProgram(st, budget, escape, rays)

The fate of one camera ray without plasma, on the geodesic lane: at every step the null
error `|g^{μν} p_μ p_ν|` is recorded (its finite maximum kept), then the ray is
captured when `march_control(st, y, …, budget)` stops (`budget` has infinite escape and
emission radii, so it stops on capture only), escapes when the decision's radius
exceeds `escape` (no outbound test), and is otherwise stepped by `rk4_step`; a
non-finite state after a step counts as captured. `out_y[i]` is the final state and
`out_m[i] = (fate, steps, max null error, final coordinate radius)` with fate 1
captured, 2 escaped, 0 step-limited, -1 no positive-energy null root at the camera.
"""
struct ShadowProgram{S,T,R} <: Function
    st::S
    budget::MarchBudget{T}
    escape::T
    rays::R
end
Adapt.@adapt_structure ShadowProgram

function (p::ShadowProgram{S,T})(item, i, out_y, out_m) where {S,T}
    y0, ok = camera_state(p.rays, item)
    y = SVector{8,T}(y0)
    status = ok ? 0.0 : -1.0
    maxnull = 0.0
    n = 0
    r_prev = T(Inf)
    for _ in 1:(ok ? p.budget.nmax : 0)
        m = march_control(p.st, y, r_prev, p.budget)
        k = momentum(y)
        nl = abs(dot(k, inverse_metric(p.st, position(y)) * k))
        isfinite(nl) && (maxnull = max(maxnull, nl))
        if m.stop
            status = 1.0
            break
        end
        if m.R > p.escape
            status = 2.0
            break
        end
        r_prev = m.R
        y = rk4_step(p.st, y, m.h)
        n += 1
        if !all(isfinite, y)
            status = 1.0
            break
        end
    end
    @inbounds out_y[i] = y
    @inbounds out_m[i] = SVector(status, Float64(n), Float64(maxnull),
                                 Float64(sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])))
    return nothing
end

# ---------------------------------------------------------------- analytic Kerr

"""
    AnalyticProgram(gen, theta_o, a, ro, disk, bands)

One screen point of the exact-Kerr equatorial tracer: `(α, β) = image_coords(gen,
item)` (Bardeen coordinates), `prep = pixel_prep(α, β, theta_o, a; ro)` once, then for
each photon-ring order `n` of `bands` (an `NTuple{K,Int}`) the landing
`band_landing(prep, n)` and the intensity `intensity(disk, landing)`. Called as
`prog(item, i, Ib, rs, ts, phis, rsgn)`, each output an `SVector{K,Float64}` per item:
the band intensity and the landing radius, time of flight, azimuth and emitted
radial-velocity sign; an invalid landing gives intensity 0 and NaN observables. Needs a
32 KiB device stack.
"""
struct AnalyticProgram{G,D,B} <: Function
    gen::G
    theta_o::Float64
    a::Float64
    ro::Float64
    disk::D
    bands::B
end
Adapt.@adapt_structure AnalyticProgram

@inline function paint(prep, n, disk)
    l = band_landing(prep, n)
    l.valid || return (0.0, NaN, NaN, NaN, NaN)
    return (Float64(intensity(disk, l)), Float64(l.r), Float64(l.t), Float64(l.φ), Float64(l.νr))
end

function (p::AnalyticProgram)(item, i, Ib, rs, ts, phis, rsgn)
    α, β = image_coords(p.gen, item)
    prep = pixel_prep(α, β, p.theta_o, p.a; ro = p.ro)
    maps = map(n -> paint(prep, n, p.disk), p.bands)
    @inbounds begin
        Ib[i] = SVector(map(m -> m[1], maps))
        rs[i] = SVector(map(m -> m[2], maps))
        ts[i] = SVector(map(m -> m[3], maps))
        phis[i] = SVector(map(m -> m[4], maps))
        rsgn[i] = SVector(map(m -> m[5], maps))
    end
    return nothing
end

stack_bytes(::AnalyticProgram) = 32 * 1024

# ---------------------------------------------------------------- Monte Carlo

"Stripes of the Monte Carlo ledger: the program of item `g` adds to stripe `(g − 1) mod LEDGER_STRIPES`, so the atomics of one quantity spread over that many cells."
const LEDGER_STRIPES = 64

# The quantities of the Monte Carlo ledger, scalar `LEDGER_SLOTS s + k` of stripe `s`.
const LEDGER_SLOTS = 6
const SPAWNED = 1                  # weight spawned into children
const SPAWNED_BY_PRIMARIES = 2     # weight spawned by primaries
const SCATTER_LOSS = 3             # incident weight lost to scattering, Σ w (1 − e^{−dτ_s})
const PRIMARY_WEIGHT = 4           # weight of the primaries (scattering runs)
const ESCAPED = 5                  # Σ w E of the escaped photons; addends: their number
const EMITTED = 6                  # Σ w E of the primaries at emission

"""
    mc_ledger() -> ExactScalars

The exact scalars of a Monte Carlo run: `LEDGER_STRIPES` stripes of six quantities, the
weight spawned into children, the weight spawned by primaries, the incident weight lost
to scattering, the weight of the primaries (these four from `tally!`, so with
scattering only), `Σ w E` of the escaped photons and `Σ w E` of the primaries at
emission. Read a quantity with [`ledger_sum`](@ref) and [`ledger_count`](@ref).
"""
mc_ledger() = ExactScalars(LEDGER_SLOTS * LEDGER_STRIPES)

# The scalars of quantity `k` in every stripe.
ledger_scalars(k::Integer) = k:LEDGER_SLOTS:(LEDGER_SLOTS * LEDGER_STRIPES)

"""
    ledger_sum(ledger, k) -> Float64

The exact total of quantity `k` of a [`mc_ledger`](@ref) over its stripes, rounded once.
"""
ledger_sum(l::ExactScalars, k::Integer) = scalar(l, ledger_scalars(k))

"""
    ledger_count(ledger, k) -> Int

The number of additions to quantity `k` of a [`mc_ledger`](@ref) over its stripes.
"""
ledger_count(l::ExactScalars, k::Integer) = addends(l, ledger_scalars(k))

"""
    MCSink(ctl, orders, children, ledger[, stripe = 0])

The sink of the Monte Carlo programs over the exact accumulators: an escaped photon of
scattering order `n` is recorded into `orders[min(n, K) + 1]` (`orders` a tuple of
`K + 1` `ExactBins`) with the direction `escape_direction(ctl, sph)` and its `w E` into
the escaped power of `ledger` (a [`mc_ledger`](@ref)); a child goes to the `ChildBuffer`
`children` (`nothing` without scattering); the scattering ledger of `tally!` goes to the
first four quantities of `ledger`. Every addition goes to stripe `stripe`. `isbits` after
`Adapt.adapt`; writes through atomics only.
"""
struct MCSink{C,O,K,L} <: Sink
    ctl::C
    orders::O
    children::K
    ledger::L
    stripe::Int
end
MCSink(ctl, orders, children, ledger) = MCSink(ctl, orders, children, ledger, 0)
Adapt.@adapt_structure MCSink

# The sink of item `item`: its ledger stripe.
@inline at_stripe(s::MCSink, item::Integer) =
    MCSink(s.ctl, s.orders, s.children, s.ledger, (Int(item) - 1) % LEDGER_STRIPES)

@inline add_ledger!(s::MCSink, k::Int, x) =
    Execution.add!(s.ledger, s.stripe * LEDGER_SLOTS + k, Float64(x))

@inline function record!(s::MCSink, sph::Superphoton)
    k = min(Int(sph.nscatt), length(s.orders) - 1) + 1
    Execution.record!(s.orders[k], sph.E, sph.w, escape_direction(s.ctl, sph))
    add_ledger!(s, ESCAPED, sph.w * sph.E)
    return nothing
end

@inline spawn!(s::MCSink, child, rng, event) = Execution.spawn!(s.children, child, rng, event)

@inline function tally!(s::MCSink, sph, loss, spawned)
    add_ledger!(s, SPAWNED, spawned)
    if sph.nscatt == 0
        add_ledger!(s, SPAWNED_BY_PRIMARIES, spawned)
        add_ledger!(s, PRIMARY_WEIGHT, sph.w)
    end
    add_ledger!(s, SCATTER_LOSS, loss)
    return nothing
end

"""
    MCTransport(ctl, st, provider, opac, absorb, scatter)

The fixed inputs of every march of a Monte Carlo run: the control `ctl` (`FlatMarch` or
`CurvedMarch`), the spacetime `st`, the plasma `provider`, the `Opacities` `opac` and the
compile-time switches `absorb` and `scatter` (`Val`s) of `track!`.
"""
struct MCTransport{C,S,P,O,A,B}
    ctl::C
    st::S
    provider::P
    opac::O
    absorb::A
    scatter::B
end
Adapt.@adapt_structure MCTransport

@inline transport!(t::MCTransport, sph, rng, sink) =
    track!(t.ctl, t.st, t.provider, t.opac, sph, rng, sink; absorb = t.absorb,
           scatter = t.scatter)

"""
    SphereSource(sphere, tables)

The primaries of the uniform sphere: every item is `generate(sphere, tables, rng)`, an
emission point uniform in the ball and the photon of the sphere's zone there, drawn
against the `GenerationTables` `tables`.
"""
struct SphereSource{S,T}
    sphere::S
    tables::T
end
Adapt.@adapt_structure SphereSource

@inline emit(s::SphereSource, item, rng) = generate(s.sphere, s.tables, rng)

"""
    ZoneSource(zones, weights, dnmax, offset, lint)

The primaries of a zone list: primary `g` belongs to the zone `z` with
`offset[z] < g ≤ offset[z + 1]` and is generated from `zones[z]` with the shared weight
table `weights` and the zone's envelope `dnmax[z]`; `lint` is `nothing` (exact
generation) or the matrix whose column `z` is the zone's tabulated `log ∫ j_ν dΩ` on the
weight nodes (tabulated generation).
"""
struct ZoneSource{Z,W,D,F,L}
    zones::Z
    weights::W
    dnmax::D
    offset::F
    lint::L
end
Adapt.@adapt_structure ZoneSource

@inline zone_int_jnu(::Nothing, w, z) = nothing
@inline zone_int_jnu(lint::AbstractMatrix, w, z) = IntJnuTable(w.lnu_min, w.dlnu, view(lint, :, z))

@inline function emit(s::ZoneSource, item, rng)
    z = searchsortedfirst(s.offset, item) - 1
    tables = GenerationTables(s.weights, @inbounds(s.dnmax[z]), zone_int_jnu(s.lint, s.weights, z))
    return generate(@inbounds(s.zones[z]), tables, rng)
end

"""
    MCPrimary(source, transport, sink, seed)

One primary superphoton of a Monte Carlo run, called as `prog(item, i)` without outputs:
item `g` draws the stream `primary_rng(seed, g)`, is emitted by `source`
([`SphereSource`](@ref), [`ZoneSource`](@ref)), adds its `w E` to the emitted power of the
ledger and is tracked by the [`MCTransport`](@ref) `transport` into the
[`MCSink`](@ref) `sink` (ledger stripe of item `g`).
"""
struct MCPrimary{G,T,K} <: Function
    source::G
    transport::T
    sink::K
    seed::UInt32
end
Adapt.@adapt_structure MCPrimary

function (p::MCPrimary)(item, i)
    sink = at_stripe(p.sink, item)
    sph, rng = emit(p.source, item, primary_rng(p.seed, item))
    add_ledger!(sink, EMITTED, sph.w * sph.E)
    transport!(p.transport, sph, rng, sink)
    return nothing
end

"""
    MCLayer(photons, rngs, transport, sink)
    MCLayer(generation::Generation, transport, sink)

One scattered child of a layer of the layered scattering loop, called as `prog(item, i)`
without outputs: item `j` tracks `photons[j]` on its genealogy stream `rngs[j]` (the
members of the `Execution.Generation` `generation`, which `foreach_cascade!` fills) by
the [`MCTransport`](@ref) `transport` into the [`MCSink`](@ref) `sink` (ledger stripe of
item `j`).
"""
struct MCLayer{G,T,K} <: Function
    generation::G
    transport::T
    sink::K
end
MCLayer(photons, rngs, transport, sink) =
    MCLayer(Execution.Generation(photons, rngs), transport, sink)
Adapt.@adapt_structure MCLayer

function (p::MCLayer)(item, i)
    g = p.generation
    @inbounds transport!(p.transport, g.photons[item], g.rngs[item], at_stripe(p.sink, item))
    return nothing
end

# ---------------------------------------------------------------- host-only programs

"""
    StoredPathProgram(mode, st, provider, params, frame, states, dls)

Host only. One pixel marched along its stored trajectory: `out[i] = march(mode,
states[item][end], prog)` under the control `StoredPath(st, states[item], dls[item])`
(states source → camera, as [`Trajectories`](@ref) holds them). Holds the trajectory
vectors, so it is not a device program.
"""
struct StoredPathProgram{M,S,P,T,F,V,D} <: Function
    mode::M
    st::S
    provider::P
    params::T
    frame::F
    states::V
    dls::D
end

function (p::StoredPathProgram)(item, i, out)
    ys = @inbounds p.states[item]
    prog = (st = p.st, control = StoredPath(p.st, ys, @inbounds(p.dls[item])),
            provider = p.provider, params = p.params, frame = p.frame)
    @inbounds out[i] = march(p.mode, ys[end], prog)
    return nothing
end

"""
    AdaptiveTraceProgram(st, rays, escape)

Host only. The adaptive trajectory of one camera ray: `trace_geodesic(st, y0; escape,
reltol = 1e-9, abstol = 1e-11, λmax = 1e5)` with every step saved, stored source →
camera in `states[i]` with `dls[i][k]` the affine span from state `k` to `k + 1` (last
0). Throws when the camera ray has no positive-energy null root.
"""
struct AdaptiveTraceProgram{S,R} <: Function
    st::S
    rays::R
    escape::Float64
end

function (p::AdaptiveTraceProgram)(item, i, states, dls)
    y0, ok = camera_state(p.rays, item)
    ok || error(NO_ROOT)
    sol = trace_geodesic(p.st, y0; escape = p.escape, reltol = 1e-9, abstol = 1e-11,
                         λmax = 1.0e5, save_everystep = true)
    m = length(sol.u)
    ys = Vector{SVector{8,Float64}}(undef, m)
    dl = Vector{Float64}(undef, m)
    for k in 1:m
        ys[k] = sol.u[m + 1 - k]
    end
    for k in 1:m-1
        dl[k] = sol.t[m + 1 - k] - sol.t[m - k]
    end
    dl[m] = 0.0
    states[i] = ys
    dls[i] = dl
    return nothing
end

"Message of a camera ray without a positive-energy null root."
const NO_ROOT = "camera rays: no past-oriented (p_t<0 ⇒ E>0) null root at the camera " *
                "position (inside the ergosphere)"
