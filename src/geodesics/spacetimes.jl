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

# ---------------------------------------------------------------- binary capture
# A backward-traced ray (p = -k, x^t decreasing) that falls toward a hole approaches
# the future horizon asymptotically: x^t -> -inf while its energy in the local
# normal frame, E_n = -k·n = alpha |dx^t/dλ| (camera E = 1), grows without bound. For
# an isolated hole that horizon is r_A = r_h,A and the per-hole test r_A - r_h,A < eps
# ends the ray. In the superposition the companion's potential (f_B ≈ 2 m_B / d_AB)
# moves the null surface outward — to r_A ≈ 1.04 for m = 0.5 at d = 25 — so r_A
# stalls above r_h,A + eps while p grows exponentially under a step frozen at
# (r_A - r_h,A)/C, until RK4 diverges and throws the state across the grid. The
# redshift test catches every such approach: past E_n = CAPTURE_REDSHIFT the fluid
# frequency is ≳ 10² ν_obs, where the emission is negligible, and the fixed-control
# step is still accurate (h |p| |∂g| ≲ 0.05).
const CAPTURE_REDSHIFT = 1.0e2

"Photon energy in the local normal (Eulerian) frame, |k·n| = alpha |g^{tμ} p_μ|, in units of the camera energy."
@inline function normal_frame_energy(st, y::SVector{8})
    gi = inverse_metric(st, SVector(y[1], y[2], y[3], y[4]))
    tdot = gi[1, 1] * y[5] + gi[1, 2] * y[6] + gi[1, 3] * y[7] + gi[1, 4] * y[8]
    return abs(tdot) / sqrt(-gi[1, 1])
end

"""
    superposed_control(st::SuperposedKerrSchild, y, C) -> (captured, h, R)

Fixed-control march step for the binary, shared by the CPU sampled tracer and the
fused kernels: step h = min_A(r_A - r_h,A)/C by the NEAREST horizon; captured on
either hole's r_A < r_h,A + `HORIZON_EPS_SAMPLED`, on a non-finite r_A, or once the
normal-frame energy exceeds `CAPTURE_REDSHIFT`; R is the global coordinate radius
for the escape test.
"""
@inline function superposed_control(st::SuperposedKerrSchild{N}, y::SVector{8},
                                    C) where {N}
    x = SVector(y[1], y[2], y[3], y[4])
    dmin = Inf
    @inbounds for A in 1:N
        dA = ks_radius_hole(st, x, A) - horizon_radius_hole(st, A)
        isfinite(dA) || (dmin = -Inf; break)
        dmin = min(dmin, dA)
    end
    R = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    cap = dmin < HORIZON_EPS_SAMPLED || !(normal_frame_energy(st, y) <= CAPTURE_REDSHIFT)
    return cap, dmin / C, R
end

# trace_geodesic for the binary: terminate when any hole is entered (per-hole radius or
# normal-frame redshift, see above), or on escape.
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
    redshift(u, λ, i) = normal_frame_energy(st, SVector{8}(u)) - CAPTURE_REDSHIFT
    cb = CallbackSet(ContinuousCallback(captured, terminate!),
                     ContinuousCallback(redshift, terminate!),
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
Step and capture come from `superposed_control` — (i) the step uses the distance
to the NEAREST horizon, `h = min_A(r_A - r_h_A)/C`, so sampling densifies toward
whichever hole a ray approaches; (ii) capture terminates on EITHER hole (horizon +
`HORIZON_EPS_SAMPLED`, or normal-frame energy above `CAPTURE_REDSHIFT` on the
asymptotic approach to the displaced horizon), matching
`trace_geodesic(::SuperposedKerrSchild)`. Escape uses
the global coordinate radius. `R_em` and the opt-in `max_step`/`cell_size_fn`
AMR caps behave exactly as in the Kerr method, and its step-density accuracy
NOTE applies unchanged.
"""
function trace_rays_sampled(st::SuperposedKerrSchild{N}, rays::Vector{SVector{8,Float64}};
                            C::Float64=40.0, escape::Float64=ESCAPE_RADIUS, nmax::Int=6000,
                            R_em::Float64=Inf, max_step::Float64=Inf,
                            cell_size_fn=nothing, cell_frac::Float64=1.0) where {N}
    n = length(rays)
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
            cap, h, R = superposed_control(st, y, C)
            cap && break
            ((R > escape || R > R_em) && R > r_prev) && break   # outbound escape only
            r_prev = R
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
