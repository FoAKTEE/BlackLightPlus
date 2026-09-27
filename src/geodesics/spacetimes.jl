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
f_A f_B l^A·l^B), so `inverse_metric` uses the Woodbury identity (an N×N solve,
2×2 for a binary; see `_superposed_gcon`) and `inverse_metric_gradient`
differentiates THAT with ForwardDiff — reducing exactly to single Kerr when a
second mass is zero.
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
@inline function _hole_term(st::SuperposedKerrSchild, x::SVector{4}, A::Int)
    c = st.center[A]
    _, f, l = _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])
    return (f, l)
end
# (the `@inline` inside the closure keeps the per-hole body out of a device call)
@inline _hole_terms(st::SuperposedKerrSchild{N}, x::SVector{4}) where {N} =
    ntuple(A -> (@inline; _hole_term(st, x, A)), Val(N))

# Left fold over a tuple, unrolled at compile time (type-stable when the accumulator's
# type changes on the first step, e.g. a Float64 η plus a dual term).
@inline _tfold(op, acc, ::Tuple{}) = acc
@inline _tfold(op, acc, t::Tuple) = _tfold(op, op(acc, first(t)), Base.tail(t))

# g = η + Σ_A f_A l_A l_Aᵀ, summed hole by hole.
@inline _add_hole(g, t) = g + t[1] * (t[2] * t[2]')
@inline _superposed_gcov(terms::Tuple) = _tfold(_add_hole, _eta(typeof(terms[1][1])), terms)

# Contravariant metric by the Woodbury identity: g⁻¹ = η − U K⁻¹ F Uᵀ with U = η L (the
# raised l^A) and the N×N K = I + F P, P = Lᵀ η L (P_AA = l_A·η·l_A vanishes up to
# round-off: each l_A is null). One hole gives the Kerr-Schild closed form η − f l^a l^b;
# two holes need a 2×2 solve in place of a 4×4 inverse, which keeps the in-kernel AD
# derivative cheap.
@inline function _superposed_gcon(terms::NTuple{N}) where {N}
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    return _eta(eltype(F)) - U * ((inv(K) * F) * U')
end

# g^{-1}p by the same Woodbury form, contracted without the 4×4 matrix.
@inline function _superposed_raise(terms::NTuple{N}, p::SVector{4}) where {N}
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    return _eta_apply(p) - U * ((inv(K) * F) * (U' * p))
end

@inline function raise_and_norm(st::SuperposedKerrSchild{N}, x::SVector{4},
                                p::SVector{4}) where {N}
    terms = _hole_terms(st, x)
    L = hcat(map(t -> t[2], terms)...)
    U = hcat(map(t -> (l = t[2]; SVector(-l[1], l[2], l[3], l[4])), terms)...)
    F = Diagonal(SVector(map(first, terms)))
    K = one(SMatrix{N,N,eltype(F)}) + F * (U' * L)
    w = U' * p                                            # l^A p (N components)
    Mw = (inv(K) * F) * w
    ηp = _eta_apply(p)
    return ηp - U * Mw, dot(p, ηp) - dot(w, Mw)
end

@inline lower_index(st::SuperposedKerrSchild, x::SVector{4}, v::SVector{4}) =
    _tfold((g, t) -> (@inline; g + (t[1] * dot(t[2], v)) * t[2]), _eta_apply(v),
           _hole_terms(st, x))
@inline raise_index(st::SuperposedKerrSchild, x::SVector{4}, p::SVector{4}) =
    _superposed_raise(_hole_terms(st, x), p)

@inline metric(st::SuperposedKerrSchild, x::SVector{4}) = _superposed_gcov(_hole_terms(st, x))
@inline inverse_metric(st::SuperposedKerrSchild, x::SVector{4}) =
    _superposed_gcon(_hole_terms(st, x))
@inline function metric_and_inverse(st::SuperposedKerrSchild, x::SVector{4})
    terms = _hole_terms(st, x)
    return _superposed_gcov(terms), _superposed_gcon(terms)
end

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
# A backward ray falling toward hole A approaches the future horizon asymptotically.
# For an isolated hole that horizon is r_A = r_h,A; in the superposition the companion's
# potential (f_B ≈ 2 m_B / d_AB) moves the null surface outward — to r_A ≈ 1.04 for
# m = 0.5 at d = 25 — so r_A stalls above r_h,A + eps while p grows exponentially under a
# step frozen at (r_A - r_h,A)/C. The redshift half of `march_captured` (E_n above
# CAPTURE_REDSHIFT, integrator.jl) ends every such approach while the fixed-control step
# is still accurate (h |p| |∂g| ≲ 0.05).

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
    cap = march_captured(dmin, normal_frame_energy(st, y))
    return cap, dmin / C, R
end

"""
    spacetime_control(st, y, C) -> (captured, h, R)

Fixed-control step, capture and escape radius of a stationary spacetime for the
reverse transfer marches: `superposed_control` for the binary; Kerr-Schild steps by
(r - r_h)/C with the `march_captured` rule on the KS radius r (escape on r); an ADM grid
steps by max(R, 1)/C and captures on the lapse (`adm_control`).
"""
@inline spacetime_control(st::SuperposedKerrSchild, y::SVector{8}, C) =
    superposed_control(st, y, C)
@inline function spacetime_control(st::KerrSchildCartesian, y::SVector{8}, C)
    r, En = ks_radius_energy(y, st.a)
    d = r - horizon_radius(st)
    return march_captured(d, En), d / C, r
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
            (cap || march_escaped(R, r_prev, escape, R_em)) && break
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
