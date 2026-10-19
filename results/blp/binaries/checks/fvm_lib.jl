# A boosted Schwarzschild hole (exact), traced (a) in the time-dependent metric with the hole
# at s(t) = s0 + v t, and (b) in that metric frozen at one time (the hole fixed at s0), which
# is what a snapshot ray tracer does. Tests: is the termination surface r' = 2M null in each
# metric, and where do backward-traced camera rays end (captured / escaped / stuck)?
using ForwardDiff, StaticArrays, LinearAlgebra, Printf
using OrdinaryDiffEq: Vern7
using SciMLBase: ODEProblem, solve, CallbackSet, ContinuousCallback, terminate!
using BLP.Spacetime: UserMetric, geodesic_rhs, nullify_covariant, metric, inverse_metric
include(joinpath(@__DIR__, "fork_metric_formulas.jl"))   # prime_coords, boosted_term, η

const M = 1.0
const A0 = SVector(0.0, 0.0, 0.0)          # no spin: the exact answer is the Schwarzschild shadow

# metric closures (generic in the number type for AD)
moving(s0, v) = X -> (sbh = s0 + v * X[1]; η + boosted_term(M, prime_coords(X[2], X[3], X[4], sbh, v), v, A0; clamp_h = false))
frozen(s0, v) = X -> (η + boosted_term(M, prime_coords(X[2], X[3], X[4], s0, v), v, A0; clamp_h = false))
rprime(X, s0, v, t_dep) = (sbh = t_dep ? s0 + v * X[1] : s0; bl_coords(prime_coords(X[2], X[3], X[4], sbh, v), A0)[1])

# --- 2. backward-traced parallel camera rays ----------------------------------------------
# camera on the +y side at distance d, receiving photons that travel along +y; pixel offset a
# along x (b = 0). Time-dependent case: the photon passes the hole near t = -d, so place the
# frozen hole where the moving hole is then: s0_frozen = s(-d).
function trace_rays(gfun, s0, v, t_dep, d, as; λmax = 4000.0)
    st = UserMetric(gfun)
    rhs(y, p, λ) = geodesic_rhs(st, y)
    fates = Symbol[]; ends = Float64[]
    for a in as
        x0 = SVector(0.0, a, d, 0.0)
        k = nullify_covariant(st, x0, SVector(0.0, 1.0, 0.0))     # received k, k_t < 0
        y0 = vcat(x0, k / k[1])                                     # p = k/k_t: p_t = +1, x^t decreasing
        cap(u, λ, i) = rprime(SVector(u[1], u[2], u[3], u[4]), s0, v, t_dep) - 2.001M
        esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - 1.5d
        cb = CallbackSet(ContinuousCallback(cap, terminate!), ContinuousCallback(esc, terminate!, nothing))
        sol = solve(ODEProblem(rhs, y0, (0.0, λmax), nothing), Vern7(); reltol = 1e-10, abstol = 1e-12, callback = cb, save_everystep = false)
        u = sol.u[end]
        rp = rprime(SVector(u[1], u[2], u[3], u[4]), s0, v, t_dep)
        R = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
        push!(fates, rp < 2.01M ? :captured : (R > 1.4d ? :escaped : :stuck))
        push!(ends, rp)
    end
    return fates, ends
end

function edge(as, fates)
    capt = [a for (a, f) in zip(as, fates) if f == :captured]
    isempty(capt) && return (NaN, NaN)
    return (minimum(capt), maximum(capt))
end
