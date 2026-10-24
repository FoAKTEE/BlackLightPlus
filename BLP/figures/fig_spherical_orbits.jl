# fig_spherical_orbits.jl — unstable spherical photon orbits around an extremal Kerr hole
# (a = M), the orbits A, C, E, F of GRay2 (arXiv:1706.07062, Table 1) as imaged in
# Mahakala (arXiv:2304.03804, Fig. 6). Each photon starts on the equator at the
# Cartesian Kerr-Schild point (sqrt(r0^2 + a^2), 0, 0), moving upward (theta decreasing),
# with E = -p_t = 1, L_z = Phi(r0) and Carter constant Q(r0) of the spherical orbit
# (Teo 2003); p_x is fixed by dx/dlambda = 0, i.e. dr/dlambda = 0 at the start.
# The orbit is marched with the fixed-step RK4 step of BLP.Spacetime (`rk4_step`) at the
# reference step dlambda = 1/1024 M for a fixed step count per orbit (A, C: 2^16; E, F:
# 2^17), stopping early on capture (r < r_h + 1e-3) or escape (r > 50), and cross-checked
# with the adaptive Vern7 solve (`trace_geodesic`, reltol 1e-12, abstol 1e-14) over the same
# span. The orbits are unstable: |r - r0| grows by e every 1.5-2.2 M from the truncation and
# round-off seed, and the record holds the lambda at which it first exceeds each of DEPART.
# The convergence sweep repeats the RK4 march from the same initial state at the steps
# CONV_STEPS = 1/4 ... 1/1024 M over the same span, up to the first |r - r0| > 1e-2 M, and
# records max |g^{mu nu} p_mu p_nu| on the sphere and the departure lambda; its 1/1024 row
# reproduces the reference march.
# Writes to BLP/figures/out/spherical_orbits/:
#   traj_<L>.npy       [lambda x y z r-r0] every SUB-th RK4 step (start and final state included)
#   diag_<L>.npy       [lambda, r - r0, g^{mu nu} p_mu p_nu, L_z - Phi] every DSUB-th RK4 step
#   vern7_<L>.npy      [lambda, r - r0, g^{mu nu} p_mu p_nu] at every accepted Vern7 step
#   convergence.npy    [orbit index 1-4, dlambda, max |g^{mu nu} p_mu p_nu| on the sphere,
#                      departure lambda] for every orbit and step of CONV_STEPS
#   record.json        orbit parameters, initial-data residuals, departure lambdas, the
#                      convergence sweep, code SHA, Julia threads and wall time
# Run from the repository root: julia -t 32 --project=BLP BLP/figures/fig_spherical_orbits.jl
# (plot: paper/scripts/plot_spherical_orbits.py and paper/scripts/plot_spherical_convergence.py)
using LinearAlgebra
using Printf
using StaticArrays
using BLP
using BLP.Spacetime: KerrSchildCartesian, inverse_metric, rk4_step, ks_radius,
    horizon_radius, null_error, angular_momentum_z, trace_geodesic

const OUT = joinpath(@__DIR__, "out", "spherical_orbits")
# code version recorded in record.json: HEAD SHA, "-dirty" if BLP/src has uncommitted edits
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const SPIN = 1.0
const H = 1 / 1024
const SUB = 8
const DSUB = 16
const CAPTURE_EPS = 1e-3
const ESCAPE_R = 50.0
const DEPART = (1e-6, 1e-3, 1e-2, 1e-1)
const ON_SPHERE = 1e-2
const CONV_STEPS = [2.0^-k for k in 2:10]
const ORBITS = (("A", 1.8, "1.8", 2^16), ("C", 1 + sqrt(2), "1+sqrt(2)", 2^16),
                ("E", 3.0, "3", 2^17), ("F", 1 + 2 * sqrt(2), "1+2sqrt(2)", 2^17))

# spherical-orbit angular momentum and Carter constant (M = E = 1; Teo 2003)
phi_sph(r, a) = -(r^3 - 3r^2 + a^2 * r + a^2) / (a * (r - 1))
q_sph(r, a) = -r^3 * (r^3 - 6r^2 + 9r - 4a^2) / (a^2 * (r - 1)^2)

function initial_state(st, r0)
    a = st.a
    Φ, Q = phi_sph(r0, a), q_sph(r0, a)
    x = SVector(0.0, sqrt(r0^2 + a^2), 0.0, 0.0)
    pt = -1.0
    py = Φ / x[2]                      # L_z = x p_y - y p_x at y = 0
    pz = sqrt(Q) / r0                  # p_theta = -r p_z on the equator, p_theta = -sqrt(Q) upward
    gi = inverse_metric(st, x)
    px = -(gi[2, 1] * pt + gi[2, 3] * py + gi[2, 4] * pz) / gi[2, 2]   # dx/dlambda = 0
    return SVector(x[1], x[2], x[3], x[4], pt, px, py, pz), Φ, Q
end

pos(y) = SVector(y[1], y[2], y[3], y[4])

first_exceed(λs, drs, t) = (k = findfirst(d -> abs(d) > t, drs); k === nothing ? Inf : λs[k])

function march_rk4(st, y0, r0, Φ, nstep)
    r_h = horizon_radius(st)
    row(y, λ) = (λ, ks_radius(st, pos(y)) - r0, null_error(st, y), angular_momentum_z(y) - Φ)
    traj = [(0.0, y0[2], y0[3], y0[4], 0.0)]
    diag = [row(y0, 0.0)]
    λs = Float64[0.0]; drs = Float64[ks_radius(st, pos(y0)) - r0]; kks = Float64[null_error(st, y0)]
    y = y0; λ = 0.0; stop = "steps"; n = 0
    for outer n in 1:nstep
        y = rk4_step(st, y, H)
        λ = n * H
        _, dr, kk, dl = row(y, λ)
        push!(λs, λ); push!(drs, dr); push!(kks, kk)
        r = r0 + dr
        ended = r < r_h + CAPTURE_EPS || r > ESCAPE_R
        (n % SUB == 0 || ended) && push!(traj, (λ, y[2], y[3], y[4], dr))
        (n % DSUB == 0 || ended) && push!(diag, (λ, dr, kk, dl))
        if ended
            stop = r > ESCAPE_R ? "escaped" : "captured"
            break
        end
    end
    on = λs .< first_exceed(λs, drs, ON_SPHERE)     # still on the sphere (|r - r0| <= 1e-2 M)
    stats = Dict{String,Any}("steps" => n, "lambda_end" => λ, "end" => stop,
        "depart_lambda" => Dict(string(t) => first_exceed(λs, drs, t) for t in DEPART),
        "max_abs_null_on_sphere" => maximum(abs, kks[on]))
    return reduce(vcat, [collect(t)' for t in traj]), reduce(vcat, [collect(t)' for t in diag]), stats
end

# max |g^{mu nu} p_mu p_nu| over the RK4 states at step h before the first |r - r0| > ON_SPHERE,
# and the lambda of that first state (Inf if the orbit holds its sphere over the span)
function sweep_rk4(st, y0, r0, h, span)
    y = y0
    m = abs(null_error(st, y0))
    for n in 1:round(Int, span / h)
        y = rk4_step(st, y, h)
        abs(ks_radius(st, pos(y)) - r0) > ON_SPHERE && return m, n * h
        m = max(m, abs(null_error(st, y)))
    end
    return m, Inf
end

function march_vern7(st, y0, r0, nstep)
    sol = trace_geodesic(st, y0; λmax=nstep * H, reltol=1e-12, abstol=1e-14,
                         save_everystep=true)
    D = zeros(length(sol.u), 3)
    for (k, (λ, y)) in enumerate(zip(sol.t, sol.u))
        D[k, :] .= (λ, ks_radius(st, pos(y)) - r0, null_error(st, y))
    end
    rend = r0 + D[end, 2]
    stop = rend < horizon_radius(st) + 1e-3 ? "captured" : (sol.t[end] < nstep * H ? "escaped" : "lambda_max")
    return D, Dict{String,Any}("steps" => length(sol.t) - 1, "lambda_end" => sol.t[end],
        "end" => stop, "r_end" => rend, "retcode" => string(sol.retcode),
        "depart_lambda" => Dict(string(t) => first_exceed(D[:, 1], D[:, 2], t) for t in DEPART))
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
json_str(x::Bool) = x ? "true" : "false"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ", ") * "]"
json_str(x::NamedTuple) = json_str(Dict(string(k) => v for (k, v) in pairs(x)))
function json_str(d::AbstractDict)
    items = [json_str(string(k)) * ": " * json_str(d[k]) for k in sort!(collect(keys(d)))]
    return "{" * join(items, ", ") * "}"
end

function main()
    t0 = time()
    mkpath(OUT)
    st = KerrSchildCartesian(SPIN)
    orbits = Dict{String,Any}[]
    conv = Matrix{Float64}(undef, 0, 4)
    for (io, (lab, r0, r0_str, nstep)) in enumerate(ORBITS)
        y0, Φ, Q = initial_state(st, r0)
        k0 = inverse_metric(st, pos(y0)) * SVector(y0[5], y0[6], y0[7], y0[8])
        traj, diag, rk = march_rk4(st, y0, r0, Φ, nstep)
        vd, vr = march_vern7(st, y0, r0, nstep)
        sw = [sweep_rk4(st, y0, r0, h, nstep * H) for h in CONV_STEPS]
        conv = vcat(conv, hcat(fill(Float64(io), length(sw)), CONV_STEPS, first.(sw), last.(sw)))
        kref = findfirst(==(H), CONV_STEPS)
        sw[kref] == (rk["max_abs_null_on_sphere"], rk["depart_lambda"][string(ON_SPHERE)]) ||
            error("orbit $lab: the sweep at dlambda = $H does not reproduce the reference march")
        write_npy(joinpath(OUT, "traj_$(lab).npy"), traj)
        write_npy(joinpath(OUT, "diag_$(lab).npy"), diag)
        write_npy(joinpath(OUT, "vern7_$(lab).npy"), vd)
        dep(d) = join([@sprintf("%.2f", d["depart_lambda"][string(t)]) for t in DEPART], "/")
        @printf("%s r0 = %.6f  Phi = %+.5f  Q = %.5f | ic: kk %.1e, dx/dl %.1e | %d steps requested\n",
                lab, r0, Φ, Q, null_error(st, y0), k0[2], nstep)
        @printf("   RK4:   |dr| > 1e-6/1e-3/1e-2/1e-1 at lambda %s; %s at lambda %.2f (%d steps); max|kk| on sphere %.1e\n",
                dep(rk), rk["end"], rk["lambda_end"], rk["steps"], rk["max_abs_null_on_sphere"])
        @printf("   Vern7: |dr| > 1e-6/1e-3/1e-2/1e-1 at lambda %s; %s at lambda %.2f (%d steps)\n",
                dep(vr), vr["end"], vr["lambda_end"], vr["steps"])
        @printf("   sweep dlambda = 1/4 ... 1/1024: max|kk| on sphere %s; |dr| > 1e-2 at lambda %s\n",
                join([@sprintf("%.1e", m) for (m, _) in sw], " "),
                join([@sprintf("%.2f", λd) for (_, λd) in sw], " "))
        push!(orbits, Dict{String,Any}(
            "label" => lab, "r0" => r0, "r0_exact" => r0_str, "Phi" => Φ, "Q" => Q,
            "nstep" => nstep, "equatorial_radius_cks" => sqrt(r0^2 + SPIN^2),
            "initial_state" => collect(y0), "initial_null_error" => null_error(st, y0),
            "initial_dxdlambda" => k0[2], "rk4" => rk, "vern7" => vr,
            "convergence" => Dict{String,Any}("step" => CONV_STEPS,
                "max_abs_null_on_sphere" => first.(sw), "depart_lambda" => last.(sw))))
    end
    write_npy(joinpath(OUT, "convergence.npy"), conv)
    wall = time() - t0
    rec = Dict{String,Any}(
        "spin" => SPIN, "step" => H, "capture_eps" => CAPTURE_EPS, "escape_r" => ESCAPE_R,
        "traj_every" => SUB, "diag_every" => DSUB, "on_sphere" => ON_SPHERE,
        "integrator" => "BLP.Spacetime.rk4_step (fixed step); cross-check trace_geodesic Vern7 reltol 1e-12 abstol 1e-14",
        "reference" => "GRay2 arXiv:1706.07062 Table 1 (A, C, E, F); Mahakala arXiv:2304.03804 Fig. 6",
        "git_sha" => GIT_SHA, "threads" => Threads.nthreads(), "wall_seconds" => wall,
        "orbits" => orbits)
    open(joinpath(OUT, "record.json"), "w") do io
        write(io, json_str(rec), "\n")
    end
    @printf("wrote %s (%.1f s on %d threads)\n", OUT, wall, Threads.nthreads())
end

main()
