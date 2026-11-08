# The one superphoton tracker and its two march controls. Each step samples the plasma at
# the step's end and accumulates the trapezoid optical depths
#   dτ = ½ (rate_n + rate_{n+1}) (L_unit h/mₑc²) dλ
# of absorption and scattering, with rates uniform over the step in the fraction f of the
# step. A photon of weight w entering a step of depths a = Δτ_a, s = Δτ_s, Δτ = a + s,
# scatters at f with expected weight density w s e^{−Δτ f} df and leaves it with
# w e^{−Δτ}, deterministically. The scattering events of the step are a Poisson process
# of mean b s for a bias b ≥ 1 chosen before the step's draws, at fractions drawn
# independently from q(f) = Δτ e^{−Δτ f}/(1 − e^{−Δτ}), each spawning a child of weight
# W = w (1 − e^{−Δτ})/(b Δτ): the expected child weight density is
#   b s · W · q(f) = w s e^{−Δτ f},
# the physical one, for every b, a and s, and W ≤ w/b. The events are the points of a
# rate-b s Poisson process on the CDF coordinate u = (1 − e^{−Δτ f})/(1 − e^{−Δτ}) ∈ [0, 1),
# drawn as cumulated exponential gaps on the photon's stream, so a step with no event
# draws one uniform. Each child kicks off the plasma at its event and carries its own
# stream, named by the event ordinal (`child_stream`), into a later layer.

"Step of the straight-line march as a fraction of the sphere radius per unit spatial |k|."
const SPHERE_STEP_FRAC = 0.05
"Step limit of the straight-line march (a ray leaves a convex ball in about 1/SPHERE_STEP_FRAC steps)."
const SPHERE_MAX_STEPS = 100_000
"Step limit of the curved march; a photon on a near-horizon orbit is dropped there (grmonty `MAXNSTEP`)."
const CURVED_MAX_STEPS = 20_000

"""
    FlatMarch{T}(R, step_frac, nmax)
    FlatMarch(sphere::UniformSphere; step_frac = SPHERE_STEP_FRAC, nmax = SPHERE_MAX_STEPS)

Straight-line march in flat coordinates for a ball of radius `R`: the wave vector
`k^μ = η^{μν} p_ν` is constant, a step advances `x += k h` with
`h = step_frac R/|k|` (`h = R` for a vanishing spatial `k`), and the photon escapes
after a step that leaves it at `|x| > R`. The emission point itself is not tested.
Without escape the march ends unrecorded after `nmax` steps. Answers the `Transfer`
control protocol (`march_control`, `geodesic_step`, `max_steps`); the `R` of its
decisions is the squared radius `|x|²`.
"""
struct FlatMarch{T}
    R::T
    step_frac::T
    nmax::Int
end
FlatMarch(s::UniformSphere; step_frac::Real = SPHERE_STEP_FRAC, nmax::Integer = SPHERE_MAX_STEPS) =
    FlatMarch(Float64(s.R), Float64(step_frac), Int(nmax))

"""
    CurvedMarch{S,T}(st, r_h, C, escape, R_em, nmax, r_plunge)
    CurvedMarch(st; C = 80.0, escape = 1.0e3, R_em = 100.0, nmax = CURVED_MAX_STEPS,
                plunge = true)

Geodesic march for Monte Carlo transport on the spacetime `st`, whose horizon proxy
`r_h` is `horizon_floor(st)` in the keyword form: with `(d, r) = proxy_distance(st, x)`
(the distance above the horizon proxy and the escape radius: Kerr–Schild radius and
`r - r_h` on Kerr, the coordinate radius and the nearest hole's `r_A - r_h,A` on a
binary, at the ray's time when the metric moves) the affine step is
`h = d/(C (|k| + 10⁻³⁰⁰))`, `|k|` the spatial norm of `k^μ = g^{μν} p_ν`, so the
coordinate advance is about `d/C` whatever the photon energy (grmonty `stepsize`); the
state advances by `rk4_step`. The photon is captured at `d < HORIZON_EPS_SAMPLED` (or a
non-finite `d`) and escapes at `r > escape`, or at `r > R_em` while outbound: beyond the
emission-confinement radius `R_em` it meets no more plasma. At most `nmax` steps; the
`R` of its decisions is `r`. A photon that may spawn no more children is captured once
a step moves it inward below `r_plunge` ([`plunging`](@ref)): `plunge_radius(st)` with
`plunge = true`, `-Inf` (never) otherwise.
"""
struct CurvedMarch{S,T}
    st::S
    r_h::T
    C::T
    escape::T
    R_em::T
    nmax::Int
    r_plunge::T
end
CurvedMarch(st; C::Real = 80.0, escape::Real = 1.0e3, R_em::Real = 100.0,
            nmax::Integer = CURVED_MAX_STEPS, plunge::Bool = true) =
    CurvedMarch(st, Float64(horizon_floor(st)), Float64(C), Float64(escape),
                Float64(R_em), Int(nmax), plunge ? plunge_radius(st) : -Inf)

@inline max_steps(c::FlatMarch) = c.nmax + 1
@inline max_steps(c::CurvedMarch) = c.nmax

@inline flat_kcon(x::SVector{4}, p::SVector{4}) = inverse_metric(Minkowski(), x) * p

@inline function march_control(c::FlatMarch, y::SVector{8}, r_prev, it)
    x = position(y)
    kcon = flat_kcon(x, momentum(y))
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    stop = it > 1 && (r2 > c.R * c.R || it > c.nmax)
    return MarchStep(stop, ks > 0.0 ? c.step_frac * c.R / ks : c.R, r2)
end

@inline function geodesic_step(c::FlatMarch, y::SVector{8}, h, it)
    x = position(y)
    p = momentum(y)
    return raystate(x .+ flat_kcon(x, p) .* h, p)
end

@inline function march_control(c::CurvedMarch, y::SVector{8}, r_prev, it)
    x = position(y)
    d, r = proxy_distance(c.st, x)
    captured = !(d >= HORIZON_EPS_SAMPLED)
    stop = captured || r > c.escape || (r > c.R_em && r > r_prev)
    kcon = inverse_metric(c.st, x) * momentum(y)
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    return MarchStep(stop, d / (c.C * (ks + 1.0e-300)), r)
end

@inline geodesic_step(c::CurvedMarch, y::SVector{8}, h, it) = rk4_step(c.st, y, h)

"""
    escaped(ctl, step::MarchStep) -> Bool

Whether the stopping decision `step` of the control `ctl` is an escape (the photon is
recorded) rather than a capture or the step limit.
"""
@inline escaped(c::FlatMarch, s::MarchStep) = s.R > c.R * c.R
# A curved march stops captured, past `escape`, or outbound past `R_em`; a capture happens
# inside the emission radius, so the stop is an escape when the radius has passed the
# smaller of the two.
@inline escaped(c::CurvedMarch, s::MarchStep) = s.R > min(c.R_em, c.escape)

"""
    captured(ctl, step::MarchStep) -> Bool

Whether the stopping decision `step` of `ctl` is a capture by the horizon: never for a
[`FlatMarch`](@ref) (its other stop is the step limit), every stop that is not an escape
for a [`CurvedMarch`](@ref).
"""
@inline captured(c::FlatMarch, s::MarchStep) = false
@inline captured(c::CurvedMarch, s::MarchStep) = !escaped(c, s)

"""
    plunging(ctl, r, r_prev) -> Bool

Whether a photon at the decision radius `r` after a step from `r_prev` is moving inward
below the control's plunge radius, hence certain to be captured: a finite `r_prev`,
`r < r_prev` and `r < ctl.r_plunge` on a [`CurvedMarch`](@ref); never on a
[`FlatMarch`](@ref). The emission point alone (`r_prev = Inf`) never plunges.
"""
@inline plunging(c::FlatMarch, r, r_prev) = false
@inline plunging(c::CurvedMarch, r, r_prev) = isfinite(r_prev) && r < r_prev && r < c.r_plunge

"""
    escape_direction(ctl, sph::Superphoton) -> SVector{3}

Spatial travel direction of the escaped `sph` (not normalised), the input of the
angular binning: the covariant spatial momentum for a [`FlatMarch`](@ref) (`k^i = p_i`
in flat coordinates), the spatial part of `k^μ = g^{μν}(x) p_ν` for a
[`CurvedMarch`](@ref) (asymptotically flat at the escape radius).
"""
@inline escape_direction(::FlatMarch, sph::Superphoton) = SVector(sph.p[2], sph.p[3], sph.p[4])
@inline function escape_direction(c::CurvedMarch, sph::Superphoton)
    kcon = inverse_metric(c.st, sph.x) * sph.p
    return SVector(kcon[2], kcon[3], kcon[4])
end

"""
    event_weight(w, b, dtau) -> Float64

Weight of every child of a step of total depth `dtau = Δτ_a + Δτ_s > 0` under bias `b`
for a photon of weight `w` at the step start: `w (1 − e^{−Δτ})/(b Δτ)`.
"""
@inline event_weight(w, b, dtau) = w * (-expm1(-dtau)) / (b * dtau)

"""
    next_event(u, lambda, dtau, rng) -> (u, frac, rng)

The next scattering event of a step after the one at CDF coordinate `u` (0 before the
first): `u + E/lambda` with `E = −log U` one exponential draw, `lambda = b Δτ_s` the
expected number of events, and its fraction of the step
`frac = −log(1 + u (e^{−Δτ} − 1))/Δτ`, the inverse of `u = (1 − e^{−Δτ f})/(1 − e^{−Δτ})`.
The step has no further event when the returned `u ≥ 1` (then `frac = 1`).
"""
@inline function next_event(u, lambda, dtau, rng::MCRng)
    x, rng = mc_uniform(rng)
    u += -log(x) / lambda
    u < 1.0 || return u, 1.0, rng
    return u, -log1p(u * expm1(-dtau)) / dtau, rng
end

# Absorption rate of the provider's frame from a sample (state, metric pair g).
@inline absorption(::FluidFrame, s, g, p) = absorption_rate(s, p)
@inline absorption(f::StaticFrame, s, g, p) = absorption_rate(f, s, last(g) * p)

"""
    scatter_site(provider, st, x, params) -> (zone::EmissionZone, ok::Bool)

The plasma and metric a scattering at `x` kicks off: the provider's sample there, `ok`
its validity; for a `UniformSphere` the sphere's uniform plasma, valid everywhere.
"""
@inline function scatter_site(p::PlasmaProvider, st, x::SVector{4}, params)
    s, g = sample(p, st, x, params)
    return EmissionZone(s, 0.0, x, g), s.valid
end
@inline scatter_site(s::UniformSphere, st, x::SVector{4}, params) =
    (emission_zone(s, x, metric_and_inverse(st, x)), true)

@inline attenuation(::Val{true}, ::Val{true}, a, s) = exp(-(a + s))
@inline attenuation(::Val{true}, ::Val{false}, a, s) = exp(-a)
@inline attenuation(::Val{false}, ::Val{true}, a, s) = exp(-s)
@inline attenuation(::Val{false}, ::Val{false}, a, s) = 1.0

"""
    March(ctl, st, provider, opac::Opacities, sink, absorb::Val, scatter::Val)

The fixed inputs of the march of one superphoton, read by [`march_open`](@ref),
[`march_steps`](@ref) and [`march_close`](@ref): the control `ctl`, the spacetime `st`,
the plasma `provider`, the [`Opacities`](@ref) `opac`, the [`Sink`](@ref) `sink`, the
compile-time switches of absorption and scattering (`Val(true)` or `Val(false)`) and the
depth unit `Cgr = L_unit h/(mₑc²)` of `opac`, which turns a rate times the affine step
into an optical depth.
"""
struct March{C,S,P,O,K,A,B}
    ctl::C
    st::S
    provider::P
    opac::O
    sink::K
    absorb::A
    scatter::B
    Cgr::Float64
end
@inline March(ctl, st, provider, opac::Opacities, sink, absorb::Val, scatter::Val) =
    March(ctl, st, provider, opac, sink, absorb, scatter, opac.L_unit * HPL / (ME * CL * CL))

@inline switch(::Val{X}) where {X} = X

# Temperature of a sample for the bias, 0 for vacuum.
@inline state_theta(s) = s.valid ? Float64(s.theta_e) : 0.0

# The events of one step (`stp`: it, y, h, the depths at its start) of the photon `sph`,
# of weight `w` at the step start, with the scattering depth `dtau_s`, the total depth
# `dtau` and the bias decision `(b, copies)`. Each event advances to its fraction of the
# step and samples the event site; each child (`copies` per event, ordinals `nev, nev + 1,
# …`) that spawns goes to the sink's `branch!` as a `ScatterEvent` with its stream.
# Returns the spawned and unspawned weight, the next ordinal and the photon's stream.
@inline function step_events(m::March, sph::Superphoton, stp, w, dtau_s, dtau, b, copies,
                             nev::Int, rng::MCRng)
    W = event_weight(w, b, dtau) / copies
    lambda = b * dtau_s
    spawned = 0.0; unspawned = 0.0
    u = 0.0
    while true
        u, frac, rng = next_event(u, lambda, dtau, rng)
        u < 1.0 || break
        yev = geodesic_step(m.ctl, stp.y, stp.h * frac, stp.it)
        xev = position(yev)
        zone, ok = scatter_site(m.provider, m.st, xev, m.opac.params)
        for _ in 1:copies
            crng, fits = child_stream(m.sink, rng, nev)
            nev += 1
            if ok && fits && W > m.opac.weight_min
                branch!(m.sink, ScatterEvent(xev, momentum(yev), zone, W, stp.tau_a,
                                             stp.tau_s, sph), crng)
                spawned += W
            else
                unspawned += W
            end
        end
    end
    return spawned, unspawned, nev, rng
end

"""
    track!(ctl, st, provider, opacities::Opacities, sph::Superphoton, rng::MCRng, sink;
           absorb = Val(true), scatter = Val(false)) -> (sph, recorded, rng)

March the superphoton `sph` under the control `ctl` ([`FlatMarch`](@ref),
[`CurvedMarch`](@ref)) on the spacetime `st` through the plasma of `provider`
(`Matter.sample` with `opacities.params`) until the control stops it. Every step
accrues the trapezoid absorption depth (`absorb = Val(true)`, rate
[`absorption_rate`](@ref) in the provider's [`rate_frame`](@ref)) and scattering depth
(`scatter = Val(true)`, [`scattering_rate`](@ref) with `opacities.table`), and decays
the weight by `e^{−(dτ_a + dτ_s)}`. With scattering, while `nscatt < opacities.max_scatt`
every step with `dτ_s > 0` draws its Poisson-many events ([`next_event`](@ref)) under the
bias [`event_bias`](@ref) of `opacities.bias` (the step's larger end temperature, the
weight at the step start), each spawning children of weight [`event_weight`](@ref) (split
in `copies`) at its event point, each on the [`child_stream`](@ref) of its ordinal: a
child whose event site ([`scatter_site`](@ref)) is valid, whose ordinal fits the stream
layout and whose weight exceeds `opacities.weight_min` goes to `branch!(sink, ev, rng)`
as a [`ScatterEvent`](@ref) with `nscatt + 1` and the depths at the step start (the
default branch kicks it by `scatter_superphoton` and hands it to `spawn!`); any other
spawns nothing. A photon that may spawn no more stops captured when it plunges
([`plunging`](@ref)). The returned photon has the final state and depths; if the control
reports an escape it is also handed to `record!(sink, sph)`, and its [`MarchLedger`](@ref)
goes to `tally!`. `rng` is the advanced stream; without scattering it is not drawn from.
Allocation-free for `isbits` arguments and sink.

`track!` is the composition
`march_close(m, march_steps(m, march_open(m, sph, rng), max_steps(ctl)))` of the
[`March`](@ref) `m` of its arguments; its body inlines the three pieces.
"""
function track!(ctl, st, provider, opac::Opacities, sph::Superphoton, rng::MCRng, sink;
                absorb::Val{A} = Val(true), scatter::Val{S} = Val(false)) where {A,S}
    m = March(ctl, st, provider, opac, sink, absorb, scatter)
    return march_close(m, march_steps(m, march_open(m, sph, rng), max_steps(ctl)))
end
