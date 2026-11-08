# The scattering event of a march step: everything the Compton kick of one child reads,
# fixed by the step before the kick draws. `step_events` builds one per child and hands it
# to the sink's `branch!`, whose default kicks it at once.

"""
    ScatterEvent{Z}

The inputs of the Compton kick of one child: the event point `x` and the parent's
covariant momentum `p` there, the plasma and metric of the event site `zone` (an
`EmissionZone`, from [`scatter_site`](@ref)), the child weight `w`
([`event_weight`](@ref) split in copies), the parent's optical depths `tau_abs` and
`tau_scatt` at the start of the step, the child's scattering count `nscatt` (the
parent's plus one), and the parent's emission diagnostics `x1i`, `x2i`, `ne0`,
`thetae0`, `b0`. `isbits`; 552 bytes with an `EmissionZone{Float64}`.
"""
struct ScatterEvent{Z}
    x::SVector{4,Float64}
    p::SVector{4,Float64}
    zone::Z
    w::Float64
    tau_abs::Float64
    tau_scatt::Float64
    nscatt::Int32
    x1i::Float64
    x2i::Float64
    ne0::Float64
    thetae0::Float64
    b0::Float64
end

"""
    ScatterEvent(x, p, zone, w, tau_abs, tau_scatt, parent::Superphoton)

The event at `x` of a step of the photon `parent`, with its momentum `p` there, the site
`zone`, the child weight `w` and the parent's depths at the step start: the child's
`nscatt` is `parent.nscatt + 1` and its diagnostics are the parent's.
"""
@inline ScatterEvent(x::SVector{4}, p::SVector{4}, zone, w, tau_abs, tau_scatt,
                     parent::Superphoton) =
    ScatterEvent(SVector{4,Float64}(x), SVector{4,Float64}(p), zone, Float64(w),
                 Float64(tau_abs), Float64(tau_scatt), parent.nscatt + Int32(1), parent.x1i,
                 parent.x2i, parent.ne0, parent.thetae0, parent.b0)

"""
    kick(ev::ScatterEvent, rng::MCRng) -> (child::Superphoton, rng)

The child of the event `ev`: the Compton kick `scatter_superphoton(ev.p, ev.zone, rng)`
of the parent's momentum off the event site's plasma, drawn on the child's stream `rng`,
at `ev.x` with the weight, depths, scattering count and diagnostics of `ev`; `rng` is
returned advanced past the kick's draws.
"""
@inline function kick(ev::ScatterEvent, rng::MCRng)
    p_new, _, rng = scatter_superphoton(ev.p, ev.zone, rng)
    child = Superphoton(ev.x, p_new, ev.w;
                        tau_abs = ev.tau_abs, tau_scatt = ev.tau_scatt,
                        nscatt = ev.nscatt,
                        x1i = ev.x1i, x2i = ev.x2i,
                        ne0 = ev.ne0, thetae0 = ev.thetae0, b0 = ev.b0)
    return child, rng
end
