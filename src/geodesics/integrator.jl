# Geodesics — integrator.jl
#
# Hamiltonian null-geodesic engine in covariant-momentum form (as in blacklight):
#   y = [x^μ; p_μ],   dx^μ/dλ = g^{μν} p_ν,   dp_i/dλ = -½ (∂_i g^{μν}) p_μ p_ν.
# p_t is conserved BY CONSTRUCTION for stationary metrics (dp_t/dλ ≡ 0) — the
# design reason for integrating covariant momentum. Production mode is an adaptive
# high-order solve with event-terminated horizon/escape; `rk4_step_fixed` is the
# fixed-step AD/GPU mode.

# ---- shared geodesic termination conventions ------------------------------------------
# Named constants so the several trace_geodesic/trace_rays* drivers — kept separate
# because their termination physics legitimately differs (KS-radius vs coordinate-radius
# escape, adaptive vs fixed-step horizon guards) — cannot drift to inconsistent values.
const ESCAPE_RADIUS = 1.1e3       # default escape sphere; ADMGridSpacetime overrides it with
                                  # the smaller grid-bounded ADM_ESCAPE_RADIUS (numerical_spacetime.jl)
const HORIZON_EPS = 1e-4          # adaptive Kerr horizon offset: terminate at ks_radius = r_h + this
const HORIZON_EPS_SAMPLED = 1e-3  # fixed-control sampler + binary capture — looser than HORIZON_EPS for
                                  # the coarse steps; gpu_tracer.jl hard-codes the same 1e-3, keep in sync

"Null constraint g^{μν} p_μ p_ν (≡ 0 on exact null geodesics; error diagnostic)."
@inline function null_error(st, y::SVector{8})
    x = SVector(y[1], y[2], y[3], y[4])
    p = SVector(y[5], y[6], y[7], y[8])
    return dot(p, inverse_metric(st, x) * p)
end

"Conserved energy E = -p_t (exact for stationary metrics in this formulation)."
@inline energy(y::SVector{8}) = -y[5]

"Conserved L_z for CKS axisymmetry (Killing vector ∂_φ = x∂_y − y∂_x): x p_y − y p_x."
@inline angular_momentum_z(y::SVector{8}) = y[2] * y[7] - y[3] * y[6]

"""
    nullify_covariant(st, x, p_spatial) -> SVector{4}

Solve g^{μν} p_μ p_ν = 0 for p_t given spatial covariant momentum, taking the
past-oriented root (p_t < 0 ⇒ E > 0). Outside the ergosphere g^{tt} < 0 and
C = g^{ij}p_i p_j > 0, so the two roots have OPPOSITE signs and exactly one is
negative. Inside the ergosphere (f > 1 ⇒ C can be < 0) both roots may be positive:
no E > 0 root exists and the contract is unsatisfiable — this errors rather than
silently returning a negative-energy (E < 0) covector that a downstream p/(−p_t)
normalization would flip.
"""
function nullify_covariant(st, x::SVector{4,T}, ps::SVector{3,T}) where {T}
    gi = inverse_metric(st, x)
    A = gi[1, 1]
    B = 2 * (gi[1, 2] * ps[1] + gi[1, 3] * ps[2] + gi[1, 4] * ps[3])
    C = zero(T)
    @inbounds for i in 1:3, j in 1:3
        C += gi[i+1, j+1] * ps[i] * ps[j]
    end
    disc = B * B - 4 * A * C
    pt = (-B - sqrt(disc)) / (2 * A)
    if !(pt < 0)                                    # first root non-negative (or NaN)
        pt = (-B + sqrt(disc)) / (2 * A)
        pt < 0 || error("nullify_covariant: no past-oriented (p_t<0 ⇒ E>0) null root " *
                        "at x=$x — inside the ergosphere the positive-energy contract is unsatisfiable")
    end
    return SVector(pt, ps[1], ps[2], ps[3])
end

"Hamiltonian geodesic RHS for stationary metrics (the exact-zero dp_t/dλ is dropped)."
@inline function geodesic_rhs(st::StationarySpacetime, y::SVector{8,T}) where {T}
    x = SVector(y[1], y[2], y[3], y[4])
    p = SVector(y[5], y[6], y[7], y[8])
    gi = inverse_metric(st, x)
    dg = inverse_metric_gradient(st, x)                 # dg[a,b,c] = ∂_c g^{ab}
    dx = gi * p
    dp1 = zero(T); dp2 = zero(T); dp3 = zero(T)
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        dp1 -= pp * dg[μ, ν, 2]
        dp2 -= pp * dg[μ, ν, 3]
        dp3 -= pp * dg[μ, ν, 4]
    end
    return SVector(dx[1], dx[2], dx[3], dx[4], zero(T), dp1 / 2, dp2 / 2, dp3 / 2)
end

"""
General geodesic RHS for an arbitrary (possibly TIME-DEPENDENT) analytic metric —
the metric-agnostic path serving the exported `UserMetric`. Identical to the
stationary form but RETAINS the time term `dp_t/dλ = -½ (∂_t g^{μν}) p_μ p_ν`. For
a stationary closure ∂_t g^{μν} ≡ 0, so this reduces exactly to the stationary RHS
(which the more specific `StationarySpacetime` method drops BY CONSTRUCTION)."""
@inline function geodesic_rhs(st::AnalyticSpacetime, y::SVector{8,T}) where {T}
    x = SVector(y[1], y[2], y[3], y[4])
    p = SVector(y[5], y[6], y[7], y[8])
    gi = inverse_metric(st, x)
    dg = inverse_metric_gradient(st, x)                 # dg[a,b,c] = ∂_c g^{ab}
    dx = gi * p
    dp0 = zero(T); dp1 = zero(T); dp2 = zero(T); dp3 = zero(T)
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]
        dp0 -= pp * dg[μ, ν, 1]
        dp1 -= pp * dg[μ, ν, 2]
        dp2 -= pp * dg[μ, ν, 3]
        dp3 -= pp * dg[μ, ν, 4]
    end
    return SVector(dx[1], dx[2], dx[3], dx[4], dp0 / 2, dp1 / 2, dp2 / 2, dp3 / 2)
end

_rhs_ode(y, st, λ) = geodesic_rhs(st, y)

"""
    trace_geodesic(st, y0; kwargs...) -> ODE solution

Production adaptive solve with event-terminated horizon capture / escape.
`save_everystep=true` retains the trajectory (needed by the radiation stage).
"""
function trace_geodesic(st::KerrSchildCartesian, y0::SVector{8}; λmax=1.0e5,
                        reltol=1e-10, abstol=1e-12, escape=ESCAPE_RADIUS, horizon_eps=HORIZON_EPS,
                        solver=Vern7(), save_everystep=false)
    r_h = horizon_radius(st)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    pos(u) = SVector(u[1], u[2], u[3], u[4])
    cb = CallbackSet(
        ContinuousCallback((u, λ, i) -> ks_radius(st, pos(u)) - (r_h + horizon_eps), terminate!),
        # escape fires only on the OUTBOUND (negative→positive) upcrossing (affect_neg!
        # = nothing) — a ray started beyond `escape` is not killed on its way IN.
        ContinuousCallback((u, λ, i) -> ks_radius(st, pos(u)) - escape, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback=cb,
                 save_everystep, save_start=save_everystep)
end

"""
    trace_geodesic(st::AnalyticSpacetime, y0; escape, λmax, ...) -> ODE solution

Metric-agnostic driver for an arbitrary analytic metric (the exported `UserMetric`).
Uses the general `geodesic_rhs` above, so a time-dependent closure integrates
correctly. The engine knows no horizon for an arbitrary metric, so integration
terminates only on the `escape` sphere (OUTBOUND crossing) or at `λmax` — set both
so an interior plunge cannot run away.
"""
function trace_geodesic(st::AnalyticSpacetime, y0::SVector{8}; λmax=1.0e5,
                        reltol=1e-10, abstol=1e-12, escape=ESCAPE_RADIUS,
                        solver=Vern7(), save_everystep=false)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = ContinuousCallback(esc, terminate!, nothing)   # outbound escape only
    return solve(prob, solver; reltol, abstol, callback=cb,
                 save_everystep, save_start=save_everystep)
end

"One classical fixed-step RK4 step (AD/GPU mode: static computation graph)."
@inline function rk4_step_fixed(st, y::SVector{8,T}, h::T) where {T}
    k1 = geodesic_rhs(st, y)
    k2 = geodesic_rhs(st, y + (h / 2) * k1)
    k3 = geodesic_rhs(st, y + (h / 2) * k2)
    k4 = geodesic_rhs(st, y + h * k3)
    return y + (h / 6) * (k1 + 2k2 + 2k3 + k4)
end

"""
    trace_rays(st, rays; escape, reltol, abstol) -> (states, dls)

Batch tracer: every camera ray solved with the adaptive `trace_geodesic`,
THREADED across rays, returning per-ray sample trajectories ORDERED SOURCE →
CAMERA (reversed from integration order), ready for the radiation stage:
  states[i] :: Vector{SVector{8}}  — samples along ray i
  dls[i]    :: Vector{Float64}     — affine step to the NEXT sample (same length;
                                      last entry 0)
Works for ANY `AbstractSpacetime`: the per-spacetime `trace_geodesic` methods own
the termination policy (Kerr horizon, binary per-hole capture, ADM lapse).
"""
function trace_rays(st::AbstractSpacetime, rays::Vector{SVector{8,Float64}};
                    escape=ESCAPE_RADIUS, reltol=1e-9, abstol=1e-11, λmax=1.0e5)
    n = length(rays)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    Threads.@threads for i in 1:n
        sol = trace_geodesic(st, rays[i]; escape, reltol, abstol, λmax,
                             save_everystep=true)
        m = length(sol.u)
        ys = Vector{SVector{8,Float64}}(undef, m)
        dl = Vector{Float64}(undef, m)
        # integration order is camera -> source; store source -> camera
        for k in 1:m
            ys[k] = sol.u[m+1-k]
        end
        for k in 1:m-1
            dl[k] = sol.t[m+1-k] - sol.t[m-k]     # positive affine spans
        end
        dl[m] = 0.0
        states[i] = ys
        dls[i] = dl
    end
    return states, dls
end

"""
    trace_rays_sampled(st, rays; C=40.0, escape=1.1e3, nmax=6000, R_em=Inf,
                       max_step=Inf, cell_size_fn=nothing, cell_frac=1.0)

Fixed-control RK4 tracer with step h = (r - r_h)/C (the blacklight/arXiv:2304.03804
step rule). Unlike the error-adaptive `trace_rays`, the sample spacing here is also
the RADIATION sampling grid — dense where the fluid varies (near the hole), which
the error-adaptive tracer under-samples in smooth-geodesic regions. Output contract
identical to `trace_rays` (source→camera order). `R_em` (default `Inf`, disabled)
adds a far-field early-out — once a ray is outbound past `R_em` it stops sampling
the vacuum return leg; only pass it when the emission is confined within `R_em`.

OPT-IN AMR-cell step cap (as in RAPTOR — never stride over a fluid cell, which
matters for AMR sampling fidelity and slow light): `max_step` hard-caps h, and
`cell_size_fn(x::SVector{4})::Real` supplies the local fluid cell size, capping h
by `cell_frac * cell_size_fn(x)`. Both default to disabled, leaving the sampling
BIT-IDENTICAL — the step is only ever reduced, and `min(h, Inf) === h`.

NOTE (step-density accuracy, measured at a=0.9375; gpu_tracer.jl uses the same
rule): C=80 meets image-level budgets (NMSE(I) 1.5e-4 vs C=160, integrated flux
3.8e-4), but the null-constraint drift max|k·k| on plunging rays (r<3) reaches
~1e-7 (median 1.1e-10; the excess is confined to ~2.4% of captured rays' tails),
and per-pixel transfer convergence is first-order in 1/C. Use C≥160 for
convergence studies (≈2× march cost; max|k·k| ~6e-9).
"""
function trace_rays_sampled(st::KerrSchildCartesian, rays::Vector{SVector{8,Float64}};
                            C::Float64=40.0, escape::Float64=ESCAPE_RADIUS, nmax::Int=6000,
                            R_em::Float64=Inf, max_step::Float64=Inf,
                            cell_size_fn=nothing, cell_frac::Float64=1.0)
    n = length(rays)
    r_h = horizon_radius(st)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    Threads.@threads for i in 1:n
        ys = SVector{8,Float64}[rays[i]]
        hs = Float64[]
        # sizehint!: hundreds of push!es per ray otherwise dominate allocations;
        # changes only the allocation strategy, states/steps stay bit-identical.
        sizehint!(ys, 1024)
        sizehint!(hs, 1024)
        y = rays[i]
        r_prev = Inf
        for _ in 1:nmax
            x = SVector(y[1], y[2], y[3], y[4])
            r = ks_radius(st, x)
            (r < r_h + HORIZON_EPS_SAMPLED || !isfinite(r)) && break
            # escape/emission stop only when OUTBOUND (r > r_prev) — a ray started
            # beyond `escape` is not killed at step 0 on its way in (R_em=Inf disables).
            ((r > escape || r > R_em) && r > r_prev) && break
            r_prev = r
            h = (r - r_h) / C
            # OPT-IN cap: never step over an AMR cell (disabled defaults keep h bit-identical).
            h = min(h, max_step)
            if cell_size_fn !== nothing
                h = min(h, cell_frac * cell_size_fn(x))
            end
            y = rk4_step_fixed(st, y, h)
            push!(ys, y)
            push!(hs, h)
        end
        m = length(ys)
        rev = Vector{SVector{8,Float64}}(undef, m)
        dl = Vector{Float64}(undef, m)
        for k in 1:m
            rev[k] = ys[m+1-k]
        end
        for k in 1:m-1
            dl[k] = hs[m-k]
        end
        dl[m] = 0.0
        states[i] = rev
        dls[i] = dl
    end
    return states, dls
end
