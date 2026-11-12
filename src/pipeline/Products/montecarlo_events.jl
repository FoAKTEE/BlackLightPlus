# The Monte Carlo over a window on a metric without a helical symmetry: no map carries a
# photon generated at one time onto another, so each primary of a frame stack held in place
# is generated at its own event (`EventSource`); and the last slice of a metric series, after
# which its metric is that slice (`last_slice`).

using ..Spacetime: ADMSeries

"""
    EventSource(sources, first, profiles)

The primaries of a run over a window on a spacetime without a helical symmetry
(`Spacetime.helical_symmetry(st)` false), the frames of a `Data.FrameStack` held in place
(`omega = 0`): primary `g` of node `m`, `first[m] < g ≤ first[m + 1]`, draws its emission
time `t_e` from `profiles[m]` on its [`time_stream`](@ref) and is generated, as by
`emit(sources[m], g, rng)` ([`ZoneSource`](@ref)), from its zone moved to the event
`(t_e, x)`: the zone's `n_e`, `Θₑ`, `B` and volume are those of the node's sites (decoded at
the frame's time), its kinematics, tetrad and metric those of the frame's plasma in the
metric of `t_e`. On a stationary spacetime the photon is [`TimedSource`](@ref)'s placed
photon bit for bit.
"""
struct EventSource{S,P}
    sources::Vector{S}
    first::Vector{Int}
    profiles::Vector{P}
end
Adapt.@adapt_structure EventSource

@inline function emit(s::EventSource, item, rng)
    m = searchsortedlast(s.first, Int(item) - 1)
    t_e, _ = sample_time(@inbounds(s.profiles[m]), time_stream(rng))
    zs = @inbounds s.sources[m]
    z = searchsortedfirst(zs.offset, item) - 1
    c = @inbounds zs.sites[z]
    site = ZoneSite{Float64}(SVector(t_e, c.x[2], c.x[3], c.x[4]), c.dV, c.n_e, c.theta_e, c.B)
    zone = emission_zone(zs.provider, zs.st, site, zs.params)
    return generate(zone, GenerationTables(zs.weights, @inbounds(zs.dnmax[z]), zs.emissivity), rng)
end

# The last slice time of a metric series (Inf for any other spacetime).
last_slice(::Any) = Inf
last_slice(st::ADMSeries) = st.times[end]
