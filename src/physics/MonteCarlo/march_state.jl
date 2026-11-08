# The march of one superphoton cut at the back edge of its step loop: `march_open` builds
# the state before the first step, `march_steps` runs the steps of a range of loop
# iterations, `march_close` records and tallies the stopped photon. Every value the loop
# carries from one iteration to the next is a field of `MarchState`; the switches, the
# rate frame and whether the photon may spawn are recomputed at every call, and the depth
# unit is a field of the `March`. `track!` is their composition over every iteration.

"""
    MarchState

The march of one superphoton between two iterations of its step loop: every value the
loop carries from one iteration to the next, and the photon the march started from.
Fields: `y` (position and covariant momentum), `w`, `tau_a`, `tau_s` (weight and
absorption and scattering depths), `na0`, `ns0`, `th0` (absorption and scattering rates
and electron temperature at the start of the next step; never sampled again), `r_prev`
(the decision radius of the last control, `Inf` before the first), `loss`, `loss_final`,
`spawned`, `unspawned`, `absorbed` (the photon's running sums of its
[`MarchLedger`](@ref)), `rng` (the photon's stream), `nev` (the next event ordinal),
`it` (the loop iteration the march runs next; once stopped, the iteration whose control
decision or plunge stopped it, or `max_steps(ctl) + 1` after the last iteration ran),
`outcome` ([`LIVE_MARCH`](@ref) until the march stops, then its [`ESCAPED_MARCH`](@ref),
`CAPTURED_MARCH` or `DROPPED_MARCH`) and `entry` (the photon as it entered the march).
`isbits`, 344 bytes.
"""
struct MarchState
    y::SVector{8,Float64}
    w::Float64
    tau_a::Float64
    tau_s::Float64
    na0::Float64
    ns0::Float64
    th0::Float64
    r_prev::Float64
    loss::Float64
    loss_final::Float64
    spawned::Float64
    unspawned::Float64
    absorbed::Float64
    rng::MCRng
    nev::Int
    it::Int32
    outcome::Int32
    entry::Superphoton
end

"""
    march_open(m::March, sph::Superphoton, rng::MCRng) -> MarchState

The state of the march `m` of the photon `sph` on its stream `rng` before its first
step: the ray state of `sph`, its weight and depths, the absorption and scattering rates
and the temperature of the plasma sample at its position (zero for a switch that is
off), `r_prev = Inf`, zero running sums, event ordinal 0, loop iteration 1, outcome
[`LIVE_MARCH`](@ref).
"""
@inline function march_open(m::March, sph::Superphoton, rng::MCRng)
    A = switch(m.absorb); S = switch(m.scatter)
    frame = rate_frame(m.provider)
    y = raystate(sph.x, sph.p)
    w = sph.w; tau_a = sph.tau_abs; tau_s = sph.tau_scatt
    s0, g = sample(m.provider, m.st, sph.x, m.opac.params)
    na0 = A ? absorption(frame, s0, g, sph.p) : 0.0
    ns0 = S ? scattering_rate(m.opac.table, s0, sph.p) : 0.0
    th0 = S ? state_theta(s0) : 0.0
    return MarchState(y, w, tau_a, tau_s, na0, ns0, th0, Inf, 0.0, 0.0, 0.0, 0.0, 0.0, rng,
                      0, Int32(1), Int32(LIVE_MARCH), sph)
end

"""
    march_steps(m::March, s::MarchState, last::Integer) -> MarchState

Run the loop iterations `s.it` through `min(last, max_steps(m.ctl))` of the march `m`
from the state `s`, the statements of one iteration of [`track!`](@ref)'s loop each:
the control's decision ([`march_control`](@ref)), the stop or the plunge, the step
([`geodesic_step`](@ref)), the plasma sample at its end, the trapezoid depths, the
running sums, the scattering events (handed to `branch!` of `m.sink`) and the weight
decay. The returned state either stopped (its `outcome` set, `it` the iteration that
stopped it) or holds iteration `last + 1` next; a march whose last iteration
`max_steps(m.ctl)` ran without a stop ends `DROPPED_MARCH` with `it = max_steps(m.ctl) + 1`.
A state that has stopped is returned unchanged. Any cut of the iterations into
consecutive ranges gives the same march.
"""
@inline function march_steps(m::March, s::MarchState, last::Integer)
    s.outcome == LIVE_MARCH || return s
    rng = s.rng
    A = switch(m.absorb); S = switch(m.scatter)
    ctl = m.ctl; st = m.st; provider = m.provider; opac = m.opac
    frame = rate_frame(provider)
    Cgr = m.Cgr
    sph = s.entry
    spawning = S && Int(sph.nscatt) < opac.max_scatt
    M = max_steps(ctl)
    first = Int(s.it)
    stop_at = min(Int(last), M)
    y = s.y
    w = s.w; tau_a = s.tau_a; tau_s = s.tau_s
    na0 = s.na0; ns0 = s.ns0; th0 = s.th0
    r_prev = s.r_prev
    outcome = ifelse(stop_at >= M, DROPPED_MARCH, LIVE_MARCH)
    nev = s.nev
    loss = s.loss; loss_final = s.loss_final; spawned = s.spawned; unspawned = s.unspawned
    absorbed = s.absorbed
    next = max(first, stop_at + 1)
    for it in first:stop_at
        d = march_control(ctl, y, r_prev, it)
        if d.stop
            outcome = escaped(ctl, d) ? ESCAPED_MARCH :
                      (captured(ctl, d) ? CAPTURED_MARCH : DROPPED_MARCH)
            next = it
            break
        end
        if !spawning && plunging(ctl, d.R, r_prev)
            outcome = CAPTURED_MARCH
            next = it
            break
        end
        r_prev = d.R
        h = d.h
        yn = geodesic_step(ctl, y, h, it)
        s1, g = sample(provider, st, position(yn), opac.params)
        na1 = A ? absorption(frame, s1, g, momentum(yn)) : 0.0
        ns1 = S ? scattering_rate(opac.table, s1, momentum(yn)) : 0.0
        dtau_a = 0.5 * (na0 + na1) * Cgr * h
        dtau_s = 0.5 * (ns0 + ns1) * Cgr * h
        dtau = (A ? dtau_a : 0.0) + (S ? dtau_s : 0.0)
        lost = dtau > 0.0 ? (-expm1(-dtau)) / dtau : 1.0
        A && (absorbed += w * (-y[5]) * dtau_a * lost)
        if S
            th1 = state_theta(s1)
            if dtau_s > 0.0
                if spawning
                    loss += w * dtau_s * lost
                    b, copies = event_bias(opac.bias, sph.nscatt, max(th0, th1), w)
                    stp = (it = it, y = y, h = h, tau_a = tau_a, tau_s = tau_s)
                    sp, us, nev, rng = step_events(m, sph, stp, w, dtau_s, dtau, b, copies,
                                                   nev, rng)
                    spawned += sp; unspawned += us
                else
                    loss_final += w * dtau_s * lost
                end
            end
            th0 = th1
        end
        w *= attenuation(m.absorb, m.scatter, dtau_a, dtau_s)
        A && (tau_a += dtau_a)
        S && (tau_s += dtau_s)
        y = yn; na0 = na1; ns0 = ns1
    end
    return MarchState(y, w, tau_a, tau_s, na0, ns0, th0, r_prev, loss, loss_final, spawned,
                      unspawned, absorbed, rng, nev, next % Int32, outcome % Int32, sph)
end

"""
    march_close(m::March, s::MarchState) -> (sph, recorded, rng)

The end of the stopped march `s` of `m`: the photon at the state's position, momentum,
weight and depths with the entry photon's scattering count and diagnostics, handed to
`record!(m.sink, sph)` if the march escaped; the [`MarchLedger`](@ref) of the state's
outcome, running sums, final weight and energy and event count handed to
`tally!(m.sink, entry, ledger)`. Returns what [`track!`](@ref) returns.
"""
@inline function march_close(m::March, s::MarchState)
    sph = s.entry
    outcome = Int(s.outcome)
    out = Superphoton(position(s.y), momentum(s.y), s.w; tau_abs = s.tau_a,
                      tau_scatt = s.tau_s, nscatt = sph.nscatt, x1i = sph.x1i,
                      x2i = sph.x2i, ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    outcome == ESCAPED_MARCH && record!(m.sink, out)
    tally!(m.sink, sph, MarchLedger(outcome, s.loss, s.loss_final, s.spawned, s.unspawned,
                                    s.absorbed, out.w, out.E, s.nev))
    return (sph = out, recorded = outcome == ESCAPED_MARCH, rng = s.rng)
end
