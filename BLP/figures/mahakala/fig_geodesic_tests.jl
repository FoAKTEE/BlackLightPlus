# fig_geodesic_tests.jl — the geodesic tests of Mahakala (arXiv:2304.03804) figures 1, 3, 4
# and 5 with BLP's geodesic machinery:
#   in_vs_out    radial null rays of Schwarzschild (M = 1) in the ingoing
#                (`KerrSchildCartesian(0)`) and outgoing (`UserMetric`, l_0 = -1) Cartesian
#                Kerr-Schild charts, the inward photon integrated forward in time and the
#                outward photon backward, from r = 50, t = 0 (figure 1);
#   null_vs_c    |g^{μν} p_μ p_ν| along one equatorial camera ray at a = 0.9 (plane-parallel
#                camera at 100 M, impact parameter b = 3.3 M prograde) marched with RK4 at the fixed-control
#                step (r - r_h)/C for C = 20 … 100 (figure 3);
#   deflection   the deflection angle δ = |Δφ| - π of equatorial rays at a = 0.9 from a
#                plane-parallel camera at 10⁴ M, impact parameter b = L_z/E = -20 … 20 in
#                steps of 0.4 outside the capture band [b₋, b₊], traced with the adaptive
#                Vern7 solve (reltol 1e-12, abstol 1e-14; δ from the direction of the wave
#                vector, corrected for the bending outside the camera and escape radii),
#                against the exact quadrature 2∫_{r₀}^∞ dφ/dr dr of the equatorial Kerr null
#                geodesic (figure 4);
#   convergence  max |g^{μν} p_μ p_ν| along the spherical photon orbits A, C, E, F of GRay2 at
#                a = 1 (initial data as BLP/figures/fig_spherical_orbits.jl) marched with RK4
#                at fixed steps Δλ = 1/4 … 1/1024 M over 40 M of affine parameter or until the
#                orbit leaves its sphere by 0.1 M (figure 5).
# Writes BLP/figures/out/mahakala/geodesic_tests/: in_vs_out_<chart>_<dir>.npy (rows t, r),
# null_vs_c_C<C>.npy (rows |t|, |g^{μν}p_μp_ν|, x, y), deflection.npy (rows b, δ numeric,
# δ exact), convergence.npy (rows orbit index, Δλ, max null), setup.json.
# Run from the repository root:  julia -t 32 --project=BLP BLP/figures/mahakala/fig_geodesic_tests.jl
using Dates, LinearAlgebra, Printf, StaticArrays
using OrdinaryDiffEq: Vern7
using SciMLBase: ODEProblem, solve, CallbackSet, ContinuousCallback, terminate!
using BLP
using BLP.Spacetime: KerrSchildCartesian, UserMetric, ETA, metric, inverse_metric, geodesic_rhs,
                     rk4_step, ks_radius, horizon_radius, null_error, trace_geodesic, angular_momentum_z,
                     PlaneParallelCamera, CameraFrame, camera_ray, kerr_delta
using BLP.Data: write_npy

const OUT = joinpath(dirname(@__DIR__), "out", "mahakala", "geodesic_tests")
const ROOT = dirname(dirname(dirname(@__DIR__)))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))
pos(y) = SVector(y[1], y[2], y[3], y[4])
setup = Dict{String,Any}("code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()))

# ---------------------------------------------------------------------------- figure 1
# Schwarzschild in the two charts; the outgoing chart as a closure (generic in the number type)
function outgoing_ks(X)
    r = sqrt(X[2]^2 + X[3]^2 + X[4]^2)
    f = 2 / r
    l = SVector(-one(r), X[2] / r, X[3] / r, X[4] / r)
    return ETA + f * (l * l')
end
# the future-directed null vector with the spatial contravariant direction `dir` at x
function null_vector(st, x, dir)
    g = metric(st, x)
    A = g[1, 1]; B = 2 * (g[1, 2] * dir[1] + g[1, 3] * dir[2] + g[1, 4] * dir[3])
    C = zero(A)
    for i in 1:3, j in 1:3
        C += g[i+1, j+1] * dir[i] * dir[j]
    end
    kt = (-B + sqrt(B * B - 4A * C)) / (2A)          # k^t > 0 (A < 0 outside the horizon)
    kt > 0 || (kt = (-B - sqrt(B * B - 4A * C)) / (2A))
    kcon = SVector(kt, dir[1], dir[2], dir[3])
    return g * kcon                                   # covariant k_μ
end
function radial_ray(st, forward::Bool)
    x0 = SVector(0.0, 50.0, 0.0, 0.0)
    k = null_vector(st, x0, forward ? SVector(-1.0, 0.0, 0.0) : SVector(1.0, 0.0, 0.0))
    p = forward ? k : -k                              # dt/dλ > 0 forward, < 0 backward
    y0 = vcat(x0, p)
    rhs(y, _, λ) = geodesic_rhs(st, y)
    cb = CallbackSet(ContinuousCallback((u, λ, i) -> abs(u[1]) - 100.0, terminate!),
                     ContinuousCallback((u, λ, i) -> sqrt(u[2]^2 + u[3]^2 + u[4]^2) - 0.25, terminate!),
                     ContinuousCallback((u, λ, i) -> sqrt(u[2]^2 + u[3]^2 + u[4]^2) - 60.0, terminate!))
    sol = solve(ODEProblem(rhs, y0, (0.0, 4000.0), nothing), Vern7(); reltol = 1e-10, abstol = 1e-12,
                callback = cb, save_everystep = true)
    M = Matrix{Float64}(undef, length(sol.u), 2)
    for (k, u) in enumerate(sol.u)
        M[k, 1] = u[1]; M[k, 2] = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
    end
    return M
end
let
    charts = ("ingoing" => KerrSchildCartesian(0.0), "outgoing" => UserMetric(outgoing_ks))
    for (name, st) in charts, (dir, fwd) in (("forward", true), ("backward", false))
        M = radial_ray(st, fwd)
        write_npy(joinpath(OUT, "in_vs_out_$(name)_$(dir).npy"), M)
        logf("fig 1 %-8s %-8s: %4d points, t %8.2f … %8.2f, r %6.3f … %6.3f\n", name, dir, size(M, 1),
             M[1, 1], M[end, 1], minimum(M[:, 2]), maximum(M[:, 2]))
    end
end

# ---------------------------------------------------------------------------- figure 3
const SPIN3 = 0.9
const DCAM3 = 100.0
const ALPHA3 = -3.3                  # α = -b: the prograde ray at b = 3.3 (b₊ = 2.85)
const CS = (20, 30, 40, 50, 60, 70, 80, 90, 100)
let
    st = KerrSchildCartesian(SPIN3)
    r_h = horizon_radius(st)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = DCAM3, fov = 10.0, npix = 2)
    y0, ok = camera_ray(st, CameraFrame(cam), ALPHA3, 0.0)
    ok || error("camera ray")
    for C in CS
        y = y0
        rows = [(0.0, abs(null_error(st, y)), y[2], y[3])]
        R_prev = Inf
        nstep = 0
        while nstep < 200_000
            r = ks_radius(st, pos(y))
            R = sqrt(y[2]^2 + y[3]^2 + y[4]^2)
            d = r - r_h
            (d < 1e-3 || (R > 1.02DCAM3 && R > R_prev)) && break
            R_prev = R
            y = rk4_step(st, y, d / C)
            nstep += 1
            push!(rows, (abs(y[1]), abs(null_error(st, y)), y[2], y[3]))
        end
        M = reduce(vcat, [collect(t)' for t in rows])
        write_npy(joinpath(OUT, "null_vs_c_C$(C).npy"), M)
        logf("fig 3 C = %3d: %6d steps, max |k.k| %.2e, final |t| %.1f\n", C, nstep, maximum(M[:, 2]), M[end, 1])
    end
    setup["fig3"] = Dict("spin" => SPIN3, "camera_distance" => DCAM3, "image_offset" => ALPHA3,
                         "C" => collect(CS))
end

# ---------------------------------------------------------------------------- figure 4
const SPIN4 = 0.9
const DCAM4 = 1.0e4                    # far enough that the coordinate direction of k is asymptotic to 1e-6
# exact Δφ of the equatorial null geodesic with impact parameter b (E = 1, L = b, M = 1):
# 2 ∫_{r₀}^∞ [(b - a) + a (r² + a² - a b)/Δ] / √R dr, R = (r² + a² - a b)² - Δ (b - a)²,
# r₀ the largest root of R; the substitution r = r₀/(1 - s²) makes the integrand smooth
# on [0, 1) and the composite Simpson rule on 2^20 panels resolves it to round-off.
function exact_dphi(a, b)
    Rf(r) = (r^2 + a^2 - a * b)^2 - kerr_delta(r, a) * (b - a)^2
    # largest root: scan down from 200 until R changes sign, then bisect
    r_hi = 200.0
    r_lo = r_hi
    while Rf(r_lo) > 0 && r_lo > 1.0
        r_lo -= 0.01
    end
    Rf(r_lo) > 0 && return NaN                  # no turning point: captured
    r_hi = r_lo + 0.01
    for _ in 1:200
        rm = (r_lo + r_hi) / 2
        Rf(rm) > 0 ? (r_hi = rm) : (r_lo = rm)
    end
    r0 = r_hi
    # the s = 0 end: R(r₀) = 0 exactly; the integrand tends to 2 N(r₀) √(r₀/R'(r₀))
    Rp = 2 * (r0^2 + a^2 - a * b) * 2 * r0 - (2 * r0 - 2) * (b - a)^2
    f0 = 2 * ((b - a) + a * (r0^2 + a^2 - a * b) / kerr_delta(r0, a)) * sqrt(r0 / Rp)
    integrand(s) = begin
        s == 0 && return f0
        om = 1 - s * s
        r = r0 / om
        N = (b - a) + a * (r^2 + a^2 - a * b) / kerr_delta(r, a)
        Rv = max(Rf(r), 0.0)
        drds = 2 * r0 * s / (om * om)
        return Rv > 0 ? N / sqrt(Rv) * drds : f0
    end
    n = 2^20
    h = (1.0 - 1e-9) / n
    acc = f0 + integrand(n * h)
    for k in 1:n-1
        acc += (isodd(k) ? 4 : 2) * integrand(k * h)
    end
    return 2 * acc * h / 3
end
# the deflection of the camera ray at image offset α: the unwrapped change of the direction
# angle of the spatial wave vector between the camera and the escape radius, plus the
# weak-field bending 2 M b / r² of the two legs outside those radii; the impact parameter is
# the conserved b = L_z/E = -(x p_y - y p_x) of the state (p = -k/E, E = 1)
function numeric_deflection(st, α)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = DCAM4, fov = 10.0, npix = 2)
    y0, ok = camera_ray(st, CameraFrame(cam), α, 0.0)
    ok || error("camera ray")
    b = -angular_momentum_z(y0)
    sol = trace_geodesic(st, y0; λmax = 1.0e5, escape = 1.1DCAM4, reltol = 1e-12, abstol = 1e-14,
                         save_everystep = true)
    dir(u) = (x = pos(u); k = inverse_metric(st, x) * SVector(u[5], u[6], u[7], u[8]); atan(k[3], k[2]))
    tot = 0.0
    ψp = dir(sol.u[1])
    for u in sol.u[2:end]
        ψ = dir(u)
        d = ψ - ψp
        d > π && (d -= 2π); d < -π && (d += 2π)
        tot += d; ψp = ψ
    end
    R = sqrt(sol.u[end][2]^2 + sol.u[end][3]^2 + sol.u[end][4]^2)
    return b, abs(tot) + 2 * abs(b) * (1 / DCAM4^2 + 1 / R^2), R
end
let
    st = KerrSchildCartesian(SPIN4)
    bp = -SPIN4 + 6 * cos(acos(-SPIN4) / 3)
    bm = -SPIN4 - 6 * cos(acos(SPIN4) / 3)
    bs = [b for b in -20.0:0.4:20.0 if !(bm <= b <= bp)]
    rows = Vector{NTuple{3,Float64}}(undef, length(bs))
    Threads.@threads for i in eachindex(bs)
        b, dn, R = numeric_deflection(st, -bs[i])             # α = -b for the camera at +x
        R > 1.0DCAM4 || @warn "ray at b = $b ended at R = $R"
        rows[i] = (b, dn, abs(exact_dphi(SPIN4, b)) - π)
    end
    M = reduce(vcat, [collect(t)' for t in rows])
    write_npy(joinpath(OUT, "deflection.npy"), M)
    err = abs.(M[:, 2] .- M[:, 3])
    logf("fig 4: %d rays, b_- = %.4f, b_+ = %.4f, |Δφ_num - Δφ_exact| max %.2e median %.2e\n",
         length(bs), bm, bp, maximum(err), sort(err)[end ÷ 2])
    setup["fig4"] = Dict("spin" => SPIN4, "camera_distance" => DCAM4, "b_minus" => bm, "b_plus" => bp,
                         "nrays" => length(bs), "max_abs_error" => maximum(err),
                         "median_abs_error" => sort(err)[end ÷ 2])
end

# ---------------------------------------------------------------------------- figure 5
const SPIN5 = 1.0
const ORBITS = (("A", 1.8), ("C", 1 + sqrt(2)), ("E", 3.0), ("F", 1 + 2 * sqrt(2)))
const SPAN5 = 40.0
phi_sph(r, a) = -(r^3 - 3r^2 + a^2 * r + a^2) / (a * (r - 1))
q_sph(r, a) = -r^3 * (r^3 - 6r^2 + 9r - 4a^2) / (a^2 * (r - 1)^2)
function orbit_state(st, r0)
    a = st.a
    Φ, Q = phi_sph(r0, a), q_sph(r0, a)
    x = SVector(0.0, sqrt(r0^2 + a^2), 0.0, 0.0)
    pt = -1.0
    py = Φ / x[2]
    pz = sqrt(Q) / r0
    gi = inverse_metric(st, x)
    px = -(gi[2, 1] * pt + gi[2, 3] * py + gi[2, 4] * pz) / gi[2, 2]
    return SVector(x[1], x[2], x[3], x[4], pt, px, py, pz)
end
let
    st = KerrSchildCartesian(SPIN5)
    r_h = horizon_radius(st)
    steps = [2.0^-k for k in 2:10]
    rows = Vector{NTuple{3,Float64}}()
    for (io, (lab, r0)) in enumerate(ORBITS)
        y0 = orbit_state(st, r0)
        res = Vector{Float64}(undef, length(steps))
        Threads.@threads for k in eachindex(steps)
            h = steps[k]
            y = y0
            m = abs(null_error(st, y))
            n = round(Int, SPAN5 / h)
            for _ in 1:n
                y = rk4_step(st, y, h)
                r = ks_radius(st, pos(y))
                abs(r - r0) > 0.1 && break                 # departed the sphere
                m = max(m, abs(null_error(st, y)))
            end
            res[k] = m
        end
        for (k, h) in enumerate(steps)
            push!(rows, (Float64(io), h, res[k]))
        end
        logf("fig 5 orbit %s (r0 = %.4f): max|k.k| at Δλ = 1/4 … 1/1024: %s\n", lab, r0,
             join([@sprintf("%.1e", v) for v in res], " "))
    end
    M = reduce(vcat, [collect(t)' for t in rows])
    write_npy(joinpath(OUT, "convergence.npy"), M)
    setup["fig5"] = Dict("spin" => SPIN5, "orbits" => [l for (l, _) in ORBITS],
                         "r0" => [r for (_, r) in ORBITS], "span" => SPAN5, "steps" => steps)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"
open(io -> println(io, json_str(setup)), joinpath(OUT, "setup.json"), "w")
logf("wrote %s\n", OUT)
