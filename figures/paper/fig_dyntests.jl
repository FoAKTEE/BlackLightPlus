# fig_dyntests.jl — exact invariants of null geodesics in two time-dependent spacetimes,
# marched by the production fixed-control RK4 (`march_control` + `rk4_step`, the loop of
# `TraceProgram`) at step constants C = 20 ... 1280.
#
# (1) "kerr": one Kerr hole (M = 1, spin a = 0.9 along z) in uniform motion at
#     v = (0, 0.4, 0.3), |v| = 0.5, through the origin at t = -d: the boosted Kerr-Schild
#     term is exact Kerr, so the state carried into the hole's rest frame,
#     x' = Λ(v)(x - s0) and p' = Λ(-v) p (s(t) = s0 + v t), conserves the rest-frame
#     energy E' = p'_t, the axial angular momentum L' = x' p'_y - y' p'_x and the Carter
#     constant Q' = p_θ² + cos²θ (L'²/sin²θ - a² E'²) (Kerr-Schild r, θ of x';
#     p_θ = cot θ (x' p'_x + y' p'_y) - r sin θ p'_z), and where sin²θ ≤ 0.1 in the
#     axis-regular form Q' = cos²θ [(r² + a²)(p'_x² + p'_y²) - a² E'²]
#     - 2 z' (x' p'_x + y' p'_y) p'_z + (r² - z'²) p'_z² (x'² + y'² = (r² + a²) sin²θ),
#     which keeps the round-off of rays near the spin axis small. E' is the Killing charge of
#     ξ = ∂_t + v·∇, linear in the state; RK4 preserves linear invariants, so E' holds to
#     round-off at every C when AD returns ∂_t g = -v·∇g.
# (2) "binary": two holes of mass 1/2 with spins χ = (0.6, -0.3) along z on the Keplerian
#     circular orbit of separation 8 (ω = 8^{-3/2}), boosted terms, phase such that the
#     holes lie on ±y at t = -d. The orbit is invariant under t → t + τ, φ → φ + ωτ, so
#     the helical Killing vector ξ = ∂_t + ω ∂_φ gives the exact invariant
#     J = p_t + ω (x p_y - y p_x) along every geodesic; J is quadratic in the state and
#     drifts at the integrator's order.
# Camera: plane-parallel, d = 1000, inclination 60°, field of view FOV, NPIX² pixels; escape
# at 1.1 d (production), no step limit in practice (NMAX). Per ray and C the record holds
# the fate, the step count, the final lab energy p_t (camera p_t = 1), the largest
# |ΔE'|, |ΔL'|, |ΔQ'| (kerr) or |ΔJ| (binary) along the ray, and the largest relative null
# error |H|/E_n² (H = g^{μν}p_μp_ν/2, E_n the normal-frame energy). At C = 80 the binary is
# also marched with the time component of the geodesic force set to zero (dp_t/dλ = 0,
# a frozen-time force; at most NMAX_FROZEN steps, and stopped where g^tt ≥ 0) to calibrate
# the size of the J violation a missing ∂_t g makes.
# The sweep starts at C = 20: at C = 10 a step of the binary march can land inside a
# horizon, where g^tt > 0 and the ray is captured with an infinite normal-frame energy.
# The march is checked against the production `trace` (CPU) on a 16² camera at C = 80:
# equal step counts, and states equal to round-off (the two compiled loops may contract
# floating-point operations differently).
# Drawn by paper/scripts/plot_dyntests.py into fig_dyntests.pdf (Fig. dyntests).
# Writes to figures/paper/out/dyntests/ (BLP_DYNTESTS_OUT):
#   sweep_<case>.npy   rows [C, pixel k, α, β, fate (1 captured, 2 escaped), steps,
#                      p_t end, min p_t, max p_t, d1, d2, d3, |H|/E_n² max]
#                      kerr: d1..d3 = max |ΔE'|, |ΔL'|, |ΔQ'|; binary: d1 = max |ΔJ|,
#                      d2 = max |ΔJ| of the frozen-time force (C = 80 only, else NaN), d3 NaN
#   record.json        setups, per-C summaries, trace cross-check, code SHA, threads, wall time
# Run from the repository root: julia -t 24 --project=. figures/paper/fig_dyntests.jl
# Env: BLP_DYNTESTS_NPIX (128), BLP_DYNTESTS_OUT (also the output directory of
#      fig_dyntests_gpu.jl and fig_dyntests_limits.jl, which include this file).
using LinearAlgebra
using Printf
using StaticArrays
using BlackLightPlus
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, CircularOrbit, KerrSchildCartesian,
    MarchBudget, march_control, rk4_step, geodesic_rhs, camera_ray, CameraFrame,
    PlaneParallelCamera, pixel_centres, pixel_coords, inverse_metric, normal_frame_energy,
    ks_radius, horizon_radius

const OUT = get(ENV, "BLP_DYNTESTS_OUT", joinpath(@__DIR__, "out", "dyntests"))
const GIT_SHA = try
    root = dirname(dirname(@__DIR__))
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const DCAM = 1000.0
const INC = 60.0
const FOV = 24.0
const NPIX = parse(Int, get(ENV, "BLP_DYNTESTS_NPIX", "128"))
const ESCAPE = 1.1 * DCAM
const NMAX = 10^6
const NMAX_FROZEN = 20_000                      # step limit of the calibration march
const C_PROD = 80.0
const C_SWEEP = [20.0 * 2.0^k for k in 0:6]

# ---- one Kerr hole in uniform motion
const A_KERR = 0.9
const V_KERR = SVector(0.0, 0.4, 0.3)
const S0_KERR = V_KERR * DCAM                    # s(t) = s0 + v t: at the origin at t = -d
uniform(s0, v) = t -> ((s0 + v * t, v),)
moving_kerr() = BoostedSuperposedKerrSchild([1.0], [(0.0, 0.0, A_KERR)], uniform(S0_KERR, V_KERR))

# Λ(v): Λ⁰₀ = Γ, Λ⁰ᵢ = Λⁱ₀ = -Γ vᵢ, Λⁱⱼ = δᵢⱼ + (Γ-1) nᵢ nⱼ
function boost_matrix(v)
    β = norm(v); Γ = 1 / sqrt(1 - β^2); n = v / β
    S = SMatrix{3,3}(I) + (Γ - 1) * (n * n')
    return SMatrix{4,4}(Γ, -Γ*v[1], -Γ*v[2], -Γ*v[3],
                        -Γ*v[1], S[1,1], S[2,1], S[3,1],
                        -Γ*v[2], S[1,2], S[2,2], S[3,2],
                        -Γ*v[3], S[1,3], S[2,3], S[3,3])
end
const LAM = boost_matrix(V_KERR)
const LAM_INV = boost_matrix(-V_KERR)
const KERR_REST = KerrSchildCartesian(A_KERR)

# rest-frame (E', L', Q') of the lab state y
function kerr_invariants(y)
    X = SVector(y[1], y[2] - S0_KERR[1], y[3] - S0_KERR[2], y[4] - S0_KERR[3])
    xp = LAM * X
    pp = LAM_INV * SVector(y[5], y[6], y[7], y[8])
    E = pp[1]
    L = xp[2] * pp[3] - xp[3] * pp[2]
    r = ks_radius(KERR_REST, SVector(0.0, xp[2], xp[3], xp[4]))
    z = xp[4]
    c2 = z * z / (r * r)                         # cos²θ
    A = xp[2] * pp[2] + xp[3] * pp[3]
    s2 = 1 - c2
    if s2 > 0.1                                  # p_θ form
        pth = sqrt(c2 / s2) * sign(z) * A - r * sqrt(s2) * pp[4]
        Q = pth * pth + c2 * (L * L / s2 - A_KERR^2 * E * E)
    else                                         # axis-regular form
        Q = c2 * ((r * r + A_KERR^2) * (pp[2]^2 + pp[3]^2) - A_KERR^2 * E * E) -
            2 * z * A * pp[4] + (r * r - z * z) * pp[4]^2
    end
    return SVector(E, L, Q)
end

# ---- the orbiting binary
const M_BIN = (0.5, 0.5)
const SEP = 8.0
const CHI = (0.6, -0.3)
const OMEGA = sqrt(sum(M_BIN) / SEP^3)
const PHASE = π / 2 + OMEGA * DCAM              # holes on ±y at t = -d
binary_orbit() = CircularOrbit(M_BIN, SEP; phase = PHASE)
binary() = BoostedSuperposedKerrSchild(collect(M_BIN), [(0.0, 0.0, CHI[1] * M_BIN[1]),
                                                        (0.0, 0.0, CHI[2] * M_BIN[2])], binary_orbit())
binary_invariants(y) = SVector(y[5] + OMEGA * (y[2] * y[7] - y[3] * y[6]))

# ---- the march
camera() = PlaneParallelCamera(inc_deg = INC, d = DCAM, fov = FOV, npix = NPIX)

# RK4 step with the time component of the force removed (the calibration march)
@inline function rk4_frozen(st, y, h)
    f(z) = (d = geodesic_rhs(st, z); SVector(d[1], d[2], d[3], d[4], 0.0, d[6], d[7], d[8]))
    d = f(y);  acc = d;         ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + (h / 2) * d
    d = f(ys); acc = acc + 2*d; ys = y + h * d
    d = f(ys); acc = acc + d
    return y + (h / 6) * acc
end

# |H|/E_n² with E_n the normal-frame energy; NaN where g^tt ≥ 0 (no normal observer)
function rel_null(st, y)
    gi = inverse_metric(st, SVector(y[1], y[2], y[3], y[4]))
    gi[1, 1] < 0 || return NaN
    p = SVector(y[5], y[6], y[7], y[8])
    return abs(dot(p, gi * p)) / (2 * normal_frame_energy(gi, y)^2)
end

# The fixed-control march of `TraceProgram` from y0 at step constant C, with the
# invariant drifts and the relative null error taken at every state.
function march_ray(st, inv, y0, C; frozen = false, keep = false, nmax = NMAX)
    b = MarchBudget(C, ESCAPE, Inf, nmax)
    I0 = inv(y0)
    dmax = zero(I0)
    hmax = rel_null(st, y0)
    ptmin = ptmax = y0[5]
    states = keep ? [y0] : nothing
    y = y0; r_prev = Inf; n = 0
    while n < nmax
        m = march_control(st, y, r_prev, b)
        m.stop && break
        r_prev = m.R
        y = frozen ? rk4_frozen(st, y, m.h) : rk4_step(st, y, m.h)
        n += 1
        dmax = max.(dmax, abs.(inv(y) - I0))
        h = rel_null(st, y)
        isfinite(h) && (hmax = max(hmax, h))
        frozen && !isfinite(h) && break       # the calibration march left the region g^tt < 0
        ptmin = min(ptmin, y[5]); ptmax = max(ptmax, y[5])
        keep && push!(states, y)
    end
    fate = sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESCAPE ? 2.0 : 1.0
    return (; fate, n, y, dmax, hmax, ptmin, ptmax, states)
end

function sweep(st, inv, name)
    cam = camera(); fr = CameraFrame(cam); cen = pixel_centres(cam)
    npx = NPIX^2
    rows = Matrix{Float64}(undef, 0, 13)
    summ = Dict{String,Any}[]
    for C in C_SWEEP
        t0 = time()
        R = Matrix{Float64}(undef, npx, 13)
        Threads.@threads :dynamic for k in 1:npx
            α, β = pixel_coords(cen, NPIX, k)
            y0, ok = camera_ray(st, fr, α, β)
            ok || error("no null root at pixel $k")
            r = march_ray(st, inv, y0, C)
            d = SVector(ntuple(i -> i <= length(r.dmax) ? r.dmax[i] : NaN, 3))
            if name == "binary" && C == C_PROD
                d = SVector(d[1], march_ray(st, inv, y0, C; frozen = true, nmax = NMAX_FROZEN).dmax[1], NaN)
            end
            R[k, :] .= (C, k, α, β, r.fate, r.n, r.y[5], r.ptmin, r.ptmax, d[1], d[2], d[3], r.hmax)
        end
        rows = vcat(rows, R)
        esc = R[:, 5] .== 2; cap = .!esc
        mx(v, m) = any(m) ? maximum(v[m]) : NaN
        s = Dict{String,Any}("C" => C, "escaped" => count(esc), "captured" => count(cap),
            "steps_max" => maximum(R[:, 6]), "steps_median_escaped" => sort(R[esc, 6])[cld(count(esc), 2)],
            "wall_seconds" => time() - t0)
        for (j, lab) in ((10, "d1"), (11, "d2"), (12, "d3"), (13, "rel_null"))
            s[lab * "_max_escaped"] = mx(R[:, j], esc)
            s[lab * "_max_captured"] = mx(R[:, j], cap)
        end
        s["pt_end_escaped_min"] = mx(-R[:, 7], esc) * -1
        s["pt_end_escaped_max"] = mx(R[:, 7], esc)
        s["pt_range_max"] = maximum(R[:, 9] .- R[:, 8])
        push!(summ, s)
        @printf("%-6s C %6.0f  esc %5d cap %5d | d1 %.1e/%.1e  d2 %.1e/%.1e  d3 %.1e/%.1e  |H|/En^2 %.1e/%.1e (escaped/captured) | %.1f s\n",
                name, C, count(esc), count(cap), s["d1_max_escaped"], s["d1_max_captured"],
                s["d2_max_escaped"], s["d2_max_captured"], s["d3_max_escaped"], s["d3_max_captured"],
                s["rel_null_max_escaped"], s["rel_null_max_captured"], s["wall_seconds"])
    end
    write_npy(joinpath(OUT, "sweep_$(name).npy"), rows)
    return summ
end

# my march against the production `trace` on a 16² camera at C = 80: step counts, bitwise
# equal rays, and the largest state difference relative to max(1, |y|) componentwise
function trace_check(st, inv)
    cfg = RunConfig(camera = CameraConfig(inclination_deg = INC, distance = DCAM, fov = FOV, resolution = 16),
                    C = C_PROD, max_steps = NMAX, escape_factor = ESCAPE / DCAM)
    tr = trace(prepare(cfg; spacetime = st, matter = nothing); backend = :cpu)
    cam = PlaneParallelCamera(inc_deg = INC, d = DCAM, fov = FOV, npix = 16)
    fr = CameraFrame(cam); cen = pixel_centres(cam)
    steps = 0; bitwise = 0; gap = 0.0
    for k in 1:256
        α, β = pixel_coords(cen, 16, k)
        y0, _ = camera_ray(st, fr, α, β)
        r = march_ray(st, inv, y0, C_PROD; keep = true)
        s = reverse(tr.states[k])
        steps += length(s) == length(r.states)
        bitwise += s == r.states
        for j in 1:min(length(s), length(r.states))
            gap = max(gap, maximum(abs.(s[j] - r.states[j]) ./ max.(1.0, abs.(s[j]))))
        end
    end
    return Dict{String,Any}("rays" => 256, "equal_step_counts" => steps, "bitwise_equal" => bitwise,
                            "max_state_gap" => gap)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? repr(Float64(x)) : "null"
json_str(x::Bool) = x ? "true" : "false"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ", ") * "]"
json_str(x::Tuple) = json_str(collect(x))
function json_str(d::AbstractDict)
    items = [json_str(string(k)) * ": " * json_str(d[k]) for k in sort!(collect(keys(d)))]
    return "{" * join(items, ", ") * "}"
end

function main()
    t0 = time()
    mkpath(OUT)
    cases = (("kerr", moving_kerr(), kerr_invariants), ("binary", binary(), binary_invariants))
    rec = Dict{String,Any}(
        "camera" => Dict("distance" => DCAM, "inclination_deg" => INC, "fov" => FOV, "npix" => NPIX),
        "escape" => ESCAPE, "C_sweep" => C_SWEEP, "C_production" => C_PROD,
        "kerr" => Dict{String,Any}("mass" => 1.0, "spin" => A_KERR, "velocity" => collect(V_KERR),
                                   "s0" => collect(S0_KERR), "invariants" => "d1 |dE'|, d2 |dL'|, d3 |dQ'|"),
        "binary" => Dict{String,Any}("masses" => collect(M_BIN), "separation" => SEP, "chi" => collect(CHI),
                                     "omega" => OMEGA, "phase" => PHASE, "hole_speed" => OMEGA * SEP / 2,
                                     "invariants" => "d1 |dJ|, d2 |dJ| frozen-time force at C_production"),
        "integrator" => "BlackLightPlus.Spacetime march_control + rk4_step (the TraceProgram loop)",
        "git_sha" => GIT_SHA, "threads" => Threads.nthreads())
    for (name, st, inv) in cases
        tc = trace_check(st, inv)
        @printf("%-6s trace cross-check at C = %.0f: %d/256 equal step counts, %d bitwise equal, max state gap %.1e\n",
                name, C_PROD, tc["equal_step_counts"], tc["bitwise_equal"], tc["max_state_gap"])
        rec[name]["trace_check"] = tc
        rec[name]["sweep"] = sweep(st, inv, name)
        if name == "kerr"
            rec[name]["horizon_radius"] = horizon_radius(st, 1)
        else
            rec[name]["horizon_radii"] = collect(horizon_radius(st))
        end
    end
    wall = time() - t0
    rec["wall_seconds"] = wall
    open(joinpath(OUT, "record.json"), "w") do io
        write(io, json_str(rec), "\n")
    end
    @printf("wrote %s (%.1f s on %d threads)\n", OUT, wall, Threads.nthreads())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
