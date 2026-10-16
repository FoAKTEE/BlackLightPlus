# Adaptive high-order geodesic solves with event termination (host only). Every escape
# event fires on the outbound crossing only, so a ray started beyond the escape sphere
# is not stopped on its way in.

_rhs_ode(y, st, λ) = geodesic_rhs(st, y)

"""
    trace_geodesic(st, y0::RayState; λmax = 1e5, reltol = 1e-10, abstol = 1e-12,
                   escape, solver = Vern7(), save_everystep = false, …) -> ODE solution

Adaptive solve of the geodesic flow from `y0` over `λ ∈ [0, λmax]`, terminated by
the spacetime's events:

- `KerrSchildCartesian`: horizon at `ks_radius = r_h + horizon_eps`
  (default `HORIZON_EPS`); escape on the Kerr-Schild radius past `escape`
  (default `ESCAPE_RADIUS`);
- `SuperposedKerrSchild`: any hole at `r_A = r_h,A + horizon_eps` (default
  `HORIZON_EPS_SAMPLED`), the normal-frame energy at `CAPTURE_REDSHIFT`, or
  escape on the coordinate radius;
- `ADMGrid`: interpolated lapse at `alpha_min` (off the grid the lapse reads 1),
  or escape on the coordinate radius (default `ADM_ESCAPE_RADIUS`, which should
  lie inside the grid);
- any other spacetime: escape on the coordinate radius or `λmax` only.

`save_everystep = true` keeps the trajectory (and its start).
"""
function trace_geodesic(st::KerrSchildCartesian, y0::SVector{8}; λmax = 1.0e5,
                        reltol = 1e-10, abstol = 1e-12, escape = ESCAPE_RADIUS,
                        horizon_eps = HORIZON_EPS, solver = Vern7(), save_everystep = false)
    r_h = horizon_radius(st)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    pos(u) = position(SVector{8}(u))
    cb = CallbackSet(
        ContinuousCallback((u, λ, i) -> ks_radius(st, pos(u)) - (r_h + horizon_eps), terminate!),
        ContinuousCallback((u, λ, i) -> ks_radius(st, pos(u)) - escape, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback = cb,
                 save_everystep, save_start = save_everystep)
end

function trace_geodesic(st::SuperposedKerrSchild{N}, y0::SVector{8}; λmax = 1.0e5,
                        reltol = 1e-10, abstol = 1e-12, escape = ESCAPE_RADIUS,
                        horizon_eps = HORIZON_EPS_SAMPLED,
                        solver = Vern7(), save_everystep = false) where {N}
    rterm = SVector{N}(horizon_radius(st, A) + horizon_eps for A in 1:N)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    function captured(u, λ, i)
        x = position(SVector{8}(u))
        m = Inf
        @inbounds for A in 1:N
            m = min(m, ks_radius(st, x, A) - rterm[A])
        end
        return m
    end
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    redshift(u, λ, i) = normal_frame_energy(st, SVector{8}(u)) - CAPTURE_REDSHIFT
    cb = CallbackSet(ContinuousCallback(captured, terminate!),
                     ContinuousCallback(redshift, terminate!),
                     ContinuousCallback(esc, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback = cb,
                 save_everystep, save_start = save_everystep)
end

# Interpolated lapse for the termination event; off the grid it reads 1, so the event
# stays quiet there (the escape sphere ends the ray).
@inline function _lapse_cb(st::ADMGrid, X::SVector{4})
    on_grid(st, X) || return one(eltype(X))
    return sample_adm(st, X[2], X[3], X[4])[1]
end

function trace_geodesic(st::ADMGrid, y0::SVector{8}; λmax = 1.0e5,
                        reltol = 1e-10, abstol = 1e-12, escape = ADM_ESCAPE_RADIUS,
                        solver = Vern7(), save_everystep = false)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    pos(u) = position(SVector{8}(u))
    lapse(u, λ, i) = _lapse_cb(st, pos(u)) - st.alpha_min
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = CallbackSet(ContinuousCallback(lapse, terminate!),
                     ContinuousCallback(esc, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback = cb,
                 save_everystep, save_start = save_everystep)
end

function trace_geodesic(st::Spacetime, y0::SVector{8}; λmax = 1.0e5,
                        reltol = 1e-10, abstol = 1e-12, escape = ESCAPE_RADIUS,
                        solver = Vern7(), save_everystep = false)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = ContinuousCallback(esc, terminate!, nothing)
    return solve(prob, solver; reltol, abstol, callback = cb,
                 save_everystep, save_start = save_everystep)
end
