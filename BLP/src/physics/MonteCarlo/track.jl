# The one superphoton tracker and its two march controls. Each step samples the plasma at
# the step's end, accumulates the trapezoid optical depths
#   dτ = ½ (rate_n + rate_{n+1}) (L_unit h/mₑc²) dλ
# of absorption (deterministic weight decay) and scattering, and draws the biased
# exponential-transform scattering decision (Kahn 1950; gpumonty `track.cu`): a child
# carrying w (1 − e^{−dτ_s})/(1 − e^{−b dτ_s}) is spawned with probability 1 − e^{−b dτ_s}
# at the fraction x₁/(b dτ_s) of the step, so the expected spawned weight equals the
# physical loss w (1 − e^{−dτ_s}) for every bias b ≥ 1, and the incident beam is
# depleted by e^{−dτ_s}. Children go to the sink and are tracked in a later layer.

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
    CurvedMarch{S,T}(st, r_h, C, escape, R_em, nmax)
    CurvedMarch(st; C = 80.0, escape = 1.0e3, R_em = 100.0, nmax = CURVED_MAX_STEPS)

Geodesic march for Monte Carlo transport on the stationary spacetime `st` with horizon
radius `r_h` (`horizon_radius(st)` in the keyword form): at Kerr–Schild radius `r` the
affine step is `h = (r − r_h)/(C (|k| + 10⁻³⁰⁰))`, `|k|` the spatial norm of
`k^μ = g^{μν} p_ν`, so the coordinate advance is about `(r − r_h)/C` whatever the photon
energy (grmonty `stepsize`); the state advances by `rk4_step`. The photon is captured
at `r < r_h + HORIZON_EPS_SAMPLED` and escapes at `r > escape`, or at `r > R_em` while
outbound: beyond the emission-confinement radius `R_em` it meets no more plasma. At
most `nmax` steps; the `R` of its decisions is `r`.
"""
struct CurvedMarch{S,T}
    st::S
    r_h::T
    C::T
    escape::T
    R_em::T
    nmax::Int
end
CurvedMarch(st; C::Real = 80.0, escape::Real = 1.0e3, R_em::Real = 100.0,
            nmax::Integer = CURVED_MAX_STEPS) =
    CurvedMarch(st, Float64(horizon_radius(st)), Float64(C), Float64(escape),
                Float64(R_em), Int(nmax))

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
    r = ks_radius(c.st, x)
    stop = r < c.r_h + HORIZON_EPS_SAMPLED || r > c.escape || (r > c.R_em && r > r_prev)
    kcon = inverse_metric(c.st, x) * momentum(y)
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    return MarchStep(stop, (r - c.r_h) / (c.C * (ks + 1.0e-300)), r)
end

@inline geodesic_step(c::CurvedMarch, y::SVector{8}, h, it) = rk4_step(c.st, y, h)

"""
    escaped(ctl, step::MarchStep) -> Bool

Whether the stopping decision `step` of the control `ctl` is an escape (the photon is
recorded) rather than a capture or the step limit.
"""
@inline escaped(c::FlatMarch, s::MarchStep) = s.R > c.R * c.R
@inline escaped(c::CurvedMarch, s::MarchStep) = !(s.R < c.r_h + HORIZON_EPS_SAMPLED)

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
    scatter_event(w, b, dtau_s, weight_min, rng) -> (spawn, weight, frac, rng)

The exponential-transform scattering decision of a step of scattering depth `dtau_s`
for a photon of weight `w` under bias `b ≥ 1`: one uniform `U` gives `x₁ = −log U`;
a child of weight `w (1 − e^{−dτ})/(1 − e^{−b dτ})` is spawned when that weight
exceeds `weight_min` and `b dτ > x₁`, at the fraction `frac = x₁/(b dτ)` of the step
(0 without a spawn).
"""
@inline function scatter_event(w, b, dtau_s, weight_min, rng::MCRng)
    weight_scat = w * (-expm1(-dtau_s)) / (-expm1(-b * dtau_s))
    u, rng = mc_uniform(rng)
    x1 = -log(u > 0.0 ? u : 1.0e-300)
    spawn = weight_scat > weight_min && b * dtau_s > x1
    return (spawn = spawn, weight = weight_scat, frac = spawn ? x1 / (b * dtau_s) : 0.0,
            rng = rng)
end

# Absorption rate of the provider's frame from a sample (state, metric pair g).
@inline absorption(::FluidFrame, s, g, p) = absorption_rate(s, p)
@inline absorption(f::StaticFrame, s, g, p) = absorption_rate(f, s, last(g) * p)

"""
    may_scatter(provider, state) -> Bool

Whether a step whose end sample is `state` may scatter: `state.valid`; always for a
`UniformSphere`, whose every step that accrues scattering depth scatters off its
uniform plasma.
"""
@inline may_scatter(::PlasmaProvider, s::PlasmaState) = s.valid
@inline may_scatter(::UniformSphere, s::PlasmaState) = true

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

# The fixed inputs of one march.
struct March{C,S,P,O,K}
    ctl::C
    st::S
    provider::P
    opac::O
    sink::K
end

# Spawn the child of the decision `ev` on step `stp = (it, y, h, dtau_a, tau_a, tau_s)`
# as spawn `event` of `sph`: kick off the plasma at the event point and hand the child
# to the sink. Returns the child's weight (0 when the site is not valid) and the stream.
@inline function spawn_child(m::March, sph::Superphoton, stp, ev, event::Int, ::Val{A}) where {A}
    yev = geodesic_step(m.ctl, stp.y, stp.h * ev.frac, stp.it)
    xev = position(yev)
    zone, ok = scatter_site(m.provider, m.st, xev, m.opac.params)
    rng = ev.rng
    ok || return 0.0, rng
    p_new, _, rng = scatter_superphoton(momentum(yev), zone, rng)
    wc = A ? ev.weight * exp(-stp.dtau_a * ev.frac) : ev.weight
    child = Superphoton(xev, p_new, wc;
                        tau_abs = stp.tau_a, tau_scatt = stp.tau_s,
                        nscatt = sph.nscatt + Int32(1),
                        x1i = sph.x1i, x2i = sph.x2i,
                        ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    spawn!(m.sink, child, rng, event)
    return wc, rng
end

"""
    track!(ctl, st, provider, opacities::Opacities, sph::Superphoton, rng::MCRng, sink;
           absorb = Val(true), scatter = Val(false)) -> (sph, recorded, rng)

March the superphoton `sph` under the control `ctl` ([`FlatMarch`](@ref),
[`CurvedMarch`](@ref)) on the spacetime `st` through the plasma of `provider`
(`Matter.sample` with `opacities.params`) until the control stops it. Every step
accrues the trapezoid absorption depth (`absorb = Val(true)`, rate
[`absorption_rate`](@ref) in the provider's [`rate_frame`](@ref)) and scattering
depth (`scatter = Val(true)`, [`scattering_rate`](@ref) with `opacities.table`), and
decays the weight by `e^{−(dτ_a + dτ_s)}`. With scattering, a step with `dτ_s > 0`
that [`may_scatter`](@ref) while `nscatt < opacities.max_scatt` draws one
[`scatter_event`](@ref) with bias `bias(opacities.bias, nscatt)`; a spawn advances to
the event point, kicks the momentum there off the plasma of [`scatter_site`](@ref) by
`scatter_superphoton` with this photon's stream, and hands the child (weight
additionally absorbed up to the event, `nscatt + 1`, the depths at the step start) to
`spawn!(sink, child, rng, event)`, `event` counting the spawn decisions from 0. The
returned photon has the final state and depths; if the control reports an escape it
is also handed to `record!(sink, sph)`, and with scattering the ledger goes to
`tally!`. `rng` is the advanced stream; without scattering it is not drawn from.
Allocation-free for `isbits` arguments and sink.
"""
function track!(ctl, st, provider, opac::Opacities, sph::Superphoton, rng::MCRng, sink;
                absorb::Val{A} = Val(true), scatter::Val{S} = Val(false)) where {A,S}
    m = March(ctl, st, provider, opac, sink)
    frame = rate_frame(provider)
    Cgr = opac.L_unit * HPL / (ME * CL * CL)
    y = raystate(sph.x, sph.p)
    w = sph.w; tau_a = sph.tau_abs; tau_s = sph.tau_scatt
    s0, g0 = sample(provider, st, sph.x, opac.params)
    na0 = A ? absorption(frame, s0, g0, sph.p) : 0.0
    ns0 = S ? scattering_rate(opac.table, s0, sph.p) : 0.0
    r_prev = Inf
    recorded = false
    nspawn = 0; loss = 0.0; spawned = 0.0
    for it in 1:max_steps(ctl)
        d = march_control(ctl, y, r_prev, it)
        if d.stop
            recorded = escaped(ctl, d)
            break
        end
        r_prev = d.R
        h = d.h
        yn = geodesic_step(ctl, y, h, it)
        s1, g1 = sample(provider, st, position(yn), opac.params)
        na1 = A ? absorption(frame, s1, g1, momentum(yn)) : 0.0
        ns1 = S ? scattering_rate(opac.table, s1, momentum(yn)) : 0.0
        dtau_a = 0.5 * (na0 + na1) * Cgr * h
        dtau_s = 0.5 * (ns0 + ns1) * Cgr * h
        if S && dtau_s > 0.0 && may_scatter(provider, s1) && Int(sph.nscatt) < opac.max_scatt
            ev = scatter_event(w, bias(opac.bias, sph.nscatt), dtau_s, opac.weight_min, rng)
            rng = ev.rng
            if ev.spawn
                stp = (it = it, y = y, h = h, dtau_a = dtau_a, tau_a = tau_a, tau_s = tau_s)
                wc, rng = spawn_child(m, sph, stp, ev, nspawn, absorb)
                nspawn += 1
                spawned += wc
            end
        end
        S && (loss += w * (-expm1(-dtau_s)))
        w *= attenuation(absorb, scatter, dtau_a, dtau_s)
        A && (tau_a += dtau_a)
        S && (tau_s += dtau_s)
        y = yn; na0 = na1; ns0 = ns1
    end
    S && tally!(sink, sph, loss, spawned)
    out = Superphoton(position(y), momentum(y), w; tau_abs = tau_a, tau_scatt = tau_s,
                      nscatt = sph.nscatt, x1i = sph.x1i, x2i = sph.x2i,
                      ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    recorded && record!(sink, out)
    return (sph = out, recorded = recorded, rng = rng)
end
