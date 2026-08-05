# Geodesics — spacetimes.jl
#
# GENERAL analytic spacetimes beyond single Kerr. The payoff of the AD design
# (contrast hand-differentiated per-metric providers as in blacklight): a new
# spacetime only defines `metric(st, x)`; the geodesic engine's `inverse_metric`
# (StaticArrays `inv`) and `inverse_metric_gradient` (ForwardDiff through
# `inverse_metric`) then work UNCHANGED, and `geodesic_rhs`, `trace_geodesic`,
# `render`, the GPU tracer, and AD all compose automatically.

using StaticArrays

# ---------------------------------------------------------------- superposed KS binary
"""
    SuperposedKerrSchild(masses, spins, centers)

Analytic superposed Kerr-Schild binary: two (or one) static Kerr-Schild black
holes with spins along ±z,

    g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν      (covariant l^A_0 = 1),

exact for one BH, the standard large-separation binary initial-data ansatz for
two. The single-KS inverse shortcut is invalid for the sum (cross term
f_A f_B l^A·l^B), so `inverse_metric` uses the full 4×4 inverse (StaticArrays)
and `inverse_metric_gradient` differentiates THAT with ForwardDiff — reducing
exactly to single Kerr when a second mass is zero.
"""
struct SuperposedKerrSchild{N,T} <: StationarySpacetime{T}
    mass::SVector{N,T}
    spin::SVector{N,T}
    center::SVector{N,SVector{3,T}}
end

function SuperposedKerrSchild(masses::AbstractVector, spins::AbstractVector,
                              centers::AbstractVector)
    n = length(masses)
    @assert length(spins) == n == length(centers)
    ctrs = SVector{n}(SVector{3,Float64}(c...) for c in centers)
    return SuperposedKerrSchild(SVector{n,Float64}(masses...),
                                SVector{n,Float64}(spins...), ctrs)
end

# One Kerr-Schild hole's (r, f, l_a) comes from the shared `_ks_term` primitive
# (metrics.jl) — the same closed form single Kerr uses, evaluated at the per-hole
# offset. Superposition then sums f_A l^A_μ l^A_ν over the holes.
@inline function metric(st::SuperposedKerrSchild{N}, x::SVector{4,T}) where {N,T}
    g = ETA
    @inbounds for A in 1:N
        c = st.center[A]
        _, f, l = _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])
        g = g + f * (l * l')
    end
    return g
end

@inline inverse_metric(st::SuperposedKerrSchild, x::SVector{4}) = inv(metric(st, x))

"Local KS radius of hole A (for termination / diagnostics)."
@inline function ks_radius_hole(st::SuperposedKerrSchild, x::SVector{4}, A::Int)
    c = st.center[A]
    return _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])[1]
end

"Per-hole horizon radius M_A + sqrt(M_A^2 - a_A^2)."
@inline function horizon_radius_hole(st::SuperposedKerrSchild, A::Int)
    m, a = st.mass[A], st.spin[A]
    d = m * m - a * a
    return m + (d > 0 ? sqrt(d) : zero(m))
end

# trace_geodesic for the binary: terminate when any hole is entered, or on escape.
function trace_geodesic(st::SuperposedKerrSchild{N}, y0::SVector{8}; λmax=1.0e5,
                        reltol=1e-10, abstol=1e-12, escape=ESCAPE_RADIUS,
                        horizon_eps=HORIZON_EPS_SAMPLED,
                        solver=Vern7(), save_everystep=false) where {N}
    rterm = SVector{N}(horizon_radius_hole(st, A) + horizon_eps for A in 1:N)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    function captured(u, λ, i)
        x = SVector(u[1], u[2], u[3], u[4])
        m = Inf
        @inbounds for A in 1:N
            m = min(m, ks_radius_hole(st, x, A) - rterm[A])
        end
        return m
    end
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = CallbackSet(ContinuousCallback(captured, terminate!),
                     # escape only on the OUTBOUND upcrossing (see integrator.jl)
                     ContinuousCallback(esc, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback=cb,
                 save_everystep, save_start=save_everystep)
end

"""
    trace_rays_sampled(st::SuperposedKerrSchild, rays; C=40.0, escape=ESCAPE_RADIUS,
                       nmax=6000, R_em=Inf, max_step=Inf, cell_size_fn=nothing,
                       cell_frac=1.0) -> (states, dls)

Binary sibling of the Kerr fixed-control sampled tracer (integrator.jl): the
same step rule and source→camera output contract, generalized to N punctures.
Two changes vs the Kerr method — (i) the step uses the distance to the NEAREST
horizon, `h = min_A(r_A - r_h_A)/C`, so sampling densifies toward whichever hole
a ray approaches; (ii) capture (at horizon + `HORIZON_EPS_SAMPLED`) terminates
on EITHER hole, matching `trace_geodesic(::SuperposedKerrSchild)`. Escape uses
the global coordinate radius. `R_em` and the opt-in `max_step`/`cell_size_fn`
AMR caps behave exactly as in the Kerr method, and its step-density accuracy
NOTE applies unchanged.
"""
function trace_rays_sampled(st::SuperposedKerrSchild{N}, rays::Vector{SVector{8,Float64}};
                            C::Float64=40.0, escape::Float64=ESCAPE_RADIUS, nmax::Int=6000,
                            R_em::Float64=Inf, max_step::Float64=Inf,
                            cell_size_fn=nothing, cell_frac::Float64=1.0) where {N}
    n = length(rays)
    rh = SVector{N}(horizon_radius_hole(st, A) for A in 1:N)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    Threads.@threads for i in 1:n
        ys = SVector{8,Float64}[rays[i]]
        hs = Float64[]
        sizehint!(ys, 1024)                       # see the Kerr method's sizehint! NOTE
        sizehint!(hs, 1024)
        y = rays[i]
        r_prev = Inf
        for _ in 1:nmax
            x = SVector(y[1], y[2], y[3], y[4])
            # distance to the nearest HORIZON; capture (either hole) at horizon + eps,
            # exactly the Kerr split — stepping by horizon distance while terminating
            # eps outside it keeps h >= eps/C, so capture lands in finitely many steps
            dmin = Inf
            @inbounds for A in 1:N
                dA = ks_radius_hole(st, x, A) - rh[A]
                isfinite(dA) || (dmin = -Inf; break)
                dmin = min(dmin, dA)
            end
            dmin < HORIZON_EPS_SAMPLED && break
            R = sqrt(x[2]^2 + x[3]^2 + x[4]^2)
            ((R > escape || R > R_em) && R > r_prev) && break   # outbound escape only
            r_prev = R
            h = min(dmin / C, max_step)
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
            rev[k] = ys[m+1-k]                    # integration order → source→camera
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
