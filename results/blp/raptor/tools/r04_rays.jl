# r04_rays.jl — R04 comparisons 2-4 (dag.md R04): geodesic trajectories, termination
# class, null norm and the transfer sample set, RAPTOR (ray-dump library, ccall) against
# BlackLightPlus (`trace_rays_sampled`, and a Vern7 reference with dense output).
#
# Rays per leg (plane-parallel camera, d = 1000, i = 60 deg, FOV 56): a 20x20 pixel grid
# (400), 600 uniform random image points, and 50 points just outside the critical curves
# (bisected on radial image lines through each hole's projected centre, then moved out by
# 1e-3 of the critical radius). Both codes start from the same BlackLightPlus camera state
# (x^mu, k_mu = -p_mu).
#
# Usage: julia --project=. -t N r04_rays.jl <bindir> <outdir> [legs=kerr,s455,s122]
# Writes <outdir>/rays_<legs>.json, termination_<legs>.json, convergence_<legs>.csv.
# Exit 0 iff every requested leg passes for the default (finite-difference) RAPTOR build
# under the rule in rays_json["acceptance_rule"]; the literal dag.md verdicts are
# recorded next to it ("literal", "pass_literal").

using BlackLightPlus, StaticArrays, LinearAlgebra, Random, Printf, Statistics
using BlackLightPlus.Geodesics: trace_geodesic, trace_rays_sampled, camera_rays_at,
                                PlaneParallelCamera, superposed_control, metric,
                                inverse_metric
include(joinpath(@__DIR__, "r04_legs.jl"))
include(joinpath(@__DIR__, "jsonw.jl"))

const BINDIR = ARGS[1]
const OUTDIR = ARGS[2]
const LEGSEL = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["kerr", "s455", "s122"]
const INC = 60.0
const DIST = 1000.0
const FOV = 56.0
const ESC = 1100.0
const NMAX = 4000
const CS = (20.0, 40.0, 80.0, 160.0)
const REF_TOL = 1e-13
mkpath(OUTDIR)

# ---------------------------------------------------------------- RAPTOR library
struct RLib
    handle::Ptr{Cvoid}
    load::Ptr{Cvoid}
    trace::Ptr{Cvoid}
end
function RLib(path)
    h = Libc.Libdl.dlopen(path)
    return RLib(h, Libc.Libdl.dlsym(h, :raptor_load_bbh), Libc.Libdl.dlsym(h, :raptor_trace_state))
end
rload(L::RLib, bbh) = ccall(L.load, Cvoid, (Cstring,), bbh)

"RAPTOR trace of each state: (xs::Matrix 8x(steps+1) of (x, k^mu), hs, class)."
function raptor_trace(L::RLib, rays)
    out = Vector{Any}(undef, length(rays))
    Threads.@threads :dynamic for i in eachindex(rays)
        lp = zeros(9 * (NMAX + 1))
        st = Ref{Cint}(0); cl = Ref{Cint}(0)
        s = Vector{Float64}(undef, 8)
        s[1:4] .= rays[i][1:4]; s[5:8] .= .-rays[i][5:8]          # k_mu = -p_mu
        ccall(L.trace, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}), s, lp, st, cl)
        n = st[]
        M = reshape(lp[1:9*(n+1)], 9, n + 1)
        out[i] = (x = M[1:8, :], h = M[9, 1:n], cls = Int(cl[]))
    end
    return out
end

# ---------------------------------------------------------------- BLP
function end_class(st, y)
    all(isfinite, y) || return 5
    if superposed_control(st, y, 80.0)[1]
        x = SVector(y[1], y[2], y[3], y[4])
        return argmin([GEO.ks_radius_hole(st, x, A) - GEO.horizon_radius_hole(st, A)
                       for A in eachindex(st.mass)])
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESC && return 3
    return 4
end

"BlackLightPlus fixed-control trace in integration order: (x::Matrix 8xm of (x, p), h, class)."
function blp_trace(st, rays, C)
    states, dls = trace_rays_sampled(st, rays; C = C, nmax = NMAX, escape = ESC)
    return map(eachindex(rays)) do i
        rev = states[i]; dl = dls[i]; m = length(rev)
        X = reduce(hcat, [Vector(rev[m+1-j]) for j in 1:m])
        h = [dl[m-j] for j in 1:m-1]
        (x = X, h = h, cls = end_class(st, rev[1]))
    end
end

# ---------------------------------------------------------------- rays
function hole_image(st, A)
    c = st.center[A]; i = deg2rad(INC)
    return (c[2], -c[1] * cos(i) + c[3] * sin(i))          # (alpha, beta)
end

"A captured image point of hole A: the projected centre, or the closest captured grid point
within 6 M of it (the companion's lensing displaces the far hole's shadow)."
function shadow_seed(st, cam, A, captured_by)
    c = hole_image(st, A)
    captured_by(c) == A && return c
    best = nothing; dbest = Inf
    for da in -6:0.5:6, db in -6:0.5:6
        p = (c[1] + da, c[2] + db)
        d = hypot(da, db)
        d < dbest && captured_by(p) == A && (best = p; dbest = d)
    end
    return best
end

function critical_points(st, cam; n = 50, rng)
    nh = length(st.mass)
    pts = Tuple{Float64,Float64}[]
    cls_at(ab) = blp_trace(st, camera_rays_at(st, cam, [ab]), 160.0)[1].cls
    captured(ab) = cls_at(ab) in (1, 2)
    seeds = [shadow_seed(st, cam, A, cls_at) for A in 1:nh]
    tries = 0
    while length(pts) < n && tries < 20n
        tries += 1
        c = seeds[1 + (tries % nh)]
        c === nothing && continue
        φ = 2π * rand(rng)
        at(ρ) = (c[1] + ρ * cos(φ), c[2] + ρ * sin(φ))
        captured(at(12.0)) && continue
        lo, hi = 0.0, 12.0
        for _ in 1:24
            mid = (lo + hi) / 2
            captured(at(mid)) ? (lo = mid) : (hi = mid)
        end
        push!(pts, at(hi * (1 + 1e-3)))
    end
    return pts
end

function leg_rays(st, leg)
    rng = MersenneTwister(hash("r04-" * leg))
    cam = PlaneParallelCamera(inc_deg = INC, d = DIST, fov = FOV, npix = 20)
    e = range(-FOV / 2, FOV / 2; length = 21); c = (e[1:end-1] .+ e[2:end]) ./ 2
    grid = [(a, b) for a in c for b in c]
    rnd = [((rand(rng) - 0.5) * FOV, (rand(rng) - 0.5) * FOV) for _ in 1:600]
    crit = critical_points(st, cam; rng)
    coords = vcat(grid, rnd, crit)
    kind = vcat(fill("grid", length(grid)), fill("random", length(rnd)), fill("critical", length(crit)))
    return camera_rays_at(st, cam, coords), kind, coords
end

# ---------------------------------------------------------------- metrics
spatial(v) = SVector(v[2], v[3], v[4])
lam(h) = vcat(0.0, cumsum(h))

"Max |x_R - x_B| (spatial) over common states at equal affine parameter (the RAPTOR state
moved to lambda_B by its own velocity dx/dlambda = -k), and max |lambda_R - lambda_B|."
function traj_diff(r, b)
    n = min(size(r.x, 2), size(b.x, 2))
    lr = lam(r.h); lb = lam(b.h)
    dx = 0.0; dl = 0.0
    for k in 1:n
        d = lb[k] - lr[k]
        xr = spatial(r.x[1:4, k]) - d * spatial(r.x[5:8, k])
        dx = max(dx, norm(xr - spatial(b.x[1:4, k])))
        dl = max(dl, abs(d))
    end
    return dx, dl
end

"Unit direction of dx/dlambda_BLP at the terminal state."
dir_raptor(r) = normalize(-spatial(r.x[5:8, end]))
function dir_blp(st, b)
    y = b.x[:, end]; x = SVector{4}(y[1:4]); p = SVector{4}(y[5:8])
    return normalize(spatial(inverse_metric(st, x) * p))
end

"Null norm along a RAPTOR ray: max |g k k| / k_t^2."
function null_raptor(st, r)
    m = 0.0
    for k in 1:size(r.x, 2)
        x = SVector{4}(r.x[1:4, k]); ku = SVector{4}(r.x[5:8, k])
        all(isfinite, ku) || continue
        g = metric(st, x); kd = g * ku
        m = max(m, abs(dot(ku, kd)) / kd[1]^2)
    end
    return m
end
function null_blp(st, b)
    m = 0.0
    for k in 1:size(b.x, 2)
        x = SVector{4}(b.x[1:4, k]); p = SVector{4}(b.x[5:8, k])
        all(isfinite, p) || continue
        m = max(m, abs(dot(p, inverse_metric(st, x) * p)) / p[1]^2)
    end
    return m
end

"Max position error against the reference before the closest approach (lambda_n <= lambda_end)."
function ref_error(st, tr, sol)
    lend = sol.t[end]
    l = lam(tr.h)
    gaps = [horizon_gap(st, SVector{4}(tr.x[1:4, k])) for k in 1:size(tr.x, 2)]
    kca = argmin(gaps)
    e = 0.0
    for k in 1:kca
        l[k] <= lend || break
        xr = sol(l[k])
        e = max(e, norm(spatial(tr.x[1:4, k]) - SVector(xr[2], xr[3], xr[4])))
    end
    return e
end

qt(v, p) = isempty(v) ? NaN : quantile(v, p)

"Full-trajectory position error against the reference over the states with lambda_n <= lambda_end."
function full_error(tr, sol)
    l = lam(tr.h); e = 0.0
    for k in 1:size(tr.x, 2)
        l[k] <= sol.t[end] || break
        xr = sol(l[k])
        e = max(e, norm(spatial(tr.x[1:4, k]) - SVector(xr[2], xr[3], xr[4])))
    end
    return e
end

"Direction of dx/dlambda at the reference's terminal (escape) state."
function dir_ref(st, sol)
    y = sol.u[end]; x = SVector(y[1], y[2], y[3], y[4]); p = SVector(y[5], y[6], y[7], y[8])
    return normalize(spatial(inverse_metric(st, x) * p))
end
angle(u, v) = atan(norm(cross(u, v)), dot(u, v))

"Normal-frame energy of a stored state (RAPTOR k^mu or BLP p_mu)."
function state_En(st, y, raptor::Bool)
    x = SVector{4}(y[1:4]); gi = inverse_metric(st, x); v = SVector{4}(y[5:8])
    tdot = raptor ? v[1] : dot(gi[1, :], v)
    return abs(tdot) / sqrt(-gi[1, 1])
end
"Max null norm over the states of a ray with normal-frame energy <= En_max."
function null_upto(st, tr, raptor::Bool, En_max)
    m = 0.0
    for k in 1:size(tr.x, 2)
        y = tr.x[:, k]; all(isfinite, y) || continue
        state_En(st, y, raptor) <= En_max || continue
        x = SVector{4}(y[1:4]); v = SVector{4}(y[5:8])
        if raptor
            kd = metric(st, x) * v; m = max(m, abs(dot(v, kd)) / kd[1]^2)
        else
            m = max(m, abs(dot(v, inverse_metric(st, x) * v)) / v[1]^2)
        end
    end
    return m
end

"Literal and attributed comparison of one RAPTOR build against BLP at march control C."
function compare_at(st, r, b, refs, kind)
    nr = length(r)
    same = [r[i].cls == b[i].cls for i in 1:nr]
    esc = [i for i in 1:nr if r[i].cls == 3 && b[i].cls == 3]
    dx = [traj_diff(r[i], b[i]) for i in esc]
    ang = [angle(dir_raptor(r[i]), dir_blp(st, b[i])) for i in esc]
    # section 5 attribution: each code's own error against the Vern7 reference
    eR = [full_error(r[i], refs[i]) for i in esc]; eB = [full_error(b[i], refs[i]) for i in esc]
    aR = [angle(dir_raptor(r[i]), dir_ref(st, refs[i])) for i in esc]
    aB = [angle(dir_blp(st, b[i]), dir_ref(st, refs[i])) for i in esc]
    dx_ratio = [first(dx[k]) / (2 * max(eR[k], eB[k], 1e-300)) for k in eachindex(esc)]
    an_ratio = [ang[k] / (2 * max(aR[k], aB[k], 1e-300)) for k in eachindex(esc)]
    nullR_esc = maximum(null_raptor(st, r[i]) for i in esc; init = 0.0)
    nullB_esc = maximum(null_blp(st, b[i]) for i in esc; init = 0.0)
    cap = [i for i in 1:nr if r[i].cls in (1, 2)]
    samples_equal = [length(r[i].h) == length(b[i].h) for i in 1:nr]
    hrel = [begin
                n = min(length(r[i].h), length(b[i].h))
                n == 0 ? 0.0 : maximum(k -> abs(r[i].h[k] - b[i].h[k]) / b[i].h[k], 1:n)
            end for i in 1:nr]
    bykind = Dict{String,Any}()
    for kd in ("grid", "random", "critical")
        sel = [k for k in eachindex(esc) if kind[esc[k]] == kd]
        isempty(sel) && continue
        bykind[kd] = Dict("n_escaped" => length(sel), "max_dx" => maximum(first.(dx[sel])),
                          "max_angle" => maximum(ang[sel]),
                          "max_dx_over_2eps" => maximum(dx_ratio[sel]),
                          "max_angle_over_2eps" => maximum(an_ratio[sel]))
    end
    return Dict(
        "termination_identical_frac" => count(same) / nr,
        "n_escaped_both" => length(esc),
        "max_dx_escaped" => isempty(dx) ? 0.0 : maximum(first.(dx)),
        "max_dlambda_escaped" => isempty(dx) ? 0.0 : maximum(last.(dx)),
        "max_escape_angle_rad" => isempty(ang) ? 0.0 : maximum(ang),
        "max_dx_over_2max_eps" => isempty(dx_ratio) ? 0.0 : maximum(dx_ratio),
        "max_angle_over_2max_eps" => isempty(an_ratio) ? 0.0 : maximum(an_ratio),
        "max_eps_raptor" => isempty(eR) ? 0.0 : maximum(eR),
        "max_eps_blp" => isempty(eB) ? 0.0 : maximum(eB),
        "by_kind" => bykind,
        "null_raptor_all" => maximum(null_raptor(st, r[i]) for i in 1:nr),
        "null_blp_all" => maximum(null_blp(st, b[i]) for i in 1:nr),
        "null_raptor_escaped" => nullR_esc, "null_blp_escaped" => nullB_esc,
        "null_raptor_En_le_2" => maximum(null_upto(st, r[i], true, 2.0) for i in 1:nr),
        "null_blp_En_le_2" => maximum(null_upto(st, b[i], false, 2.0) for i in 1:nr),
        "null_raptor_captured_En_le_10" => maximum(null_upto(st, r[i], true, 10.0) for i in cap; init = 0.0),
        "null_blp_captured_En_le_10" => maximum(null_upto(st, b[i], false, 10.0) for i in cap; init = 0.0),
        "samples_equal_frac" => count(samples_equal) / nr,
        "samples_equal_on_same_class" => all(samples_equal[same]),
        "max_step_rel_diff" => maximum(hrel),
        "mismatches" => [i for i in 1:nr if !same[i]])
end

# ---------------------------------------------------------------- main
libs = Dict("fd" => RLib(joinpath(BINDIR, "libraptor_sks_fd.so")),
            "an" => RLib(joinpath(BINDIR, "libraptor_sks_an.so")),
            "cks" => RLib(joinpath(BINDIR, "libraptor_cks.so")))
tmp = mktempdir()
rays_json = Dict{String,Any}(); term_json = Dict{String,Any}()
conv_rows = String[]
allpass = true
for leg in LEGSEL
    st = leg_spacetime(leg)
    t0 = time()
    rays, kind, coords = leg_rays(st, leg)
    nr = length(rays)
    @printf("[%s] %d rays (%d critical) in %.1f s\n", leg, nr, count(==("critical"), kind), time() - t0)

    # reference (Vern7, dense)
    refs = Vector{Any}(undef, nr)
    Threads.@threads :dynamic for i in 1:nr
        refs[i] = trace_geodesic(st, rays[i]; reltol = REF_TOL, abstol = REF_TOL,
                                 escape = ESC, save_everystep = true)
    end
    codes = leg == "kerr" ? ("fd", "an", "cks") : ("fd", "an")
    blp = Dict{Float64,Any}(); rap = Dict{Tuple{String,Float64},Any}()
    for C in CS
        blp[C] = blp_trace(st, rays, C)
        bbh = write_bbh_in(joinpath(tmp, "$(leg)_$(Int(C)).in"), leg; C = C, nmax = NMAX,
                           escape = ESC, cam_dist = DIST)
        for cd in codes
            rload(libs[cd], bbh)
            rap[(cd, C)] = raptor_trace(libs[cd], rays)
        end
    end

    # convergence against the reference
    err = Dict{String,Matrix{Float64}}()
    err["blp"] = [ref_error(st, blp[C][i], refs[i]) for i in 1:nr, C in CS]
    for cd in codes
        err[cd] = [ref_error(st, rap[(cd, C)][i], refs[i]) for i in 1:nr, C in CS]
    end
    conv = Dict{String,Any}()
    for (name, E) in err
        ords = Dict{String,Any}()
        for k in 1:length(CS)-1
            o = [log2(E[i, k] / E[i, k+1]) for i in 1:nr if E[i, k+1] > 0 && E[i, k] > 0]
            ords["C$(Int(CS[k]))_$(Int(CS[k+1]))"] =
                Dict("median" => qt(o, 0.5), "p10" => qt(o, 0.1), "p90" => qt(o, 0.9))
        end
        conv[name] = Dict("max_err" => [maximum(E[:, k]) for k in 1:length(CS)],
                          "median_err" => [median(E[:, k]) for k in 1:length(CS)],
                          "order_maxnorm" => [log2(maximum(E[:, k]) / maximum(E[:, k+1]))
                                              for k in 1:length(CS)-1],
                          "order_per_ray" => ords)
        for i in 1:nr
            push!(conv_rows, @sprintf("%s,%s,%d,%s,%s", leg, name, i, kind[i],
                                      join([@sprintf("%.4e", E[i, k]) for k in 1:length(CS)], ",")))
        end
    end

    # comparisons at C = 80 (dag.md) and C = 160
    b80 = blp[80.0]
    legres = Dict{String,Any}("n_rays" => nr, "convergence" => conv,
                              "reference" => "Vern7 reltol=abstol=$(REF_TOL), dense output")
    legterm = Dict{String,Any}()
    for cd in codes
        c80 = compare_at(st, rap[(cd, 80.0)], b80, refs, kind)
        c160 = compare_at(st, rap[(cd, 160.0)], blp[160.0], refs, kind)
        o = conv[cd]["order_per_ray"]; bo = conv["blp"]["order_per_ray"]
        order_ok = o["C40_80"]["median"] >= 3.5 && o["C80_160"]["median"] >= 3.5 &&
                   bo["C40_80"]["median"] >= 3.5 && bo["C80_160"]["median"] >= 3.5
        lit = Dict(
            "class" => c80["termination_identical_frac"] >= 0.999,
            "dx" => c80["max_dx_escaped"] <= 1e-3,
            "angle" => c80["max_escape_angle_rad"] <= 1e-6,
            "null_every_ray" => c80["null_raptor_all"] <= 1e-8,
            "samples" => c80["samples_equal_on_same_class"],
            "order" => order_ok)
        # design.md section 5: a larger difference is discretisation when it stays within
        # twice the larger self-error; the literal threshold must then hold at C = 160.
        att = Dict(
            "class" => lit["class"],
            "dx" => lit["dx"] || (c80["max_dx_over_2max_eps"] <= 1 && c160["max_dx_escaped"] <= 1e-3),
            "angle" => lit["angle"] ||
                       (c80["max_angle_over_2max_eps"] <= 1 && c160["max_escape_angle_rad"] <= 1e-6),
            "null_escaped" => c80["null_raptor_escaped"] <= max(1e-8, c80["null_blp_escaped"]) &&
                              c160["null_raptor_escaped"] <= 1e-8,
            "null_En_le_2" => c160["null_raptor_En_le_2"] <= 1e-8,
            "samples" => lit["samples"], "order" => order_ok)
        ok_lit = all(values(lit)); ok_att = all(values(att))
        legres[cd] = Dict("C80" => c80, "C160" => c160, "literal" => lit, "attributed" => att,
                          "pass_literal" => ok_lit, "pass" => ok_att)
        legterm[cd] = Dict(
            "mismatches_C80" => [Dict("ray" => i, "kind" => kind[i], "alpha_beta" => collect(coords[i]),
                                      "raptor" => rap[(cd, 80.0)][i].cls, "blp" => b80[i].cls)
                                 for i in c80["mismatches"]],
            "class_counts_raptor_C80" => [count(i -> rap[(cd, 80.0)][i].cls == c, 1:nr) for c in 1:5],
            "class_counts_blp_C80" => [count(i -> b80[i].cls == c, 1:nr) for c in 1:5])
        cd == "fd" && (global allpass &= ok_att)
        @printf("[%s|%s] C80: class %.4f dx %.2e (/2eps %.2f) ang %.2e (/2eps %.2f) null all %.1e esc %.1e (BLP %.1e) En<=2 %.1e | C160: dx %.2e ang %.2e null esc %.1e En<=2 %.1e | order %.2f %.2f | literal %s attributed %s\n",
                leg, cd, c80["termination_identical_frac"], c80["max_dx_escaped"],
                c80["max_dx_over_2max_eps"], c80["max_escape_angle_rad"],
                c80["max_angle_over_2max_eps"], c80["null_raptor_all"], c80["null_raptor_escaped"],
                c80["null_blp_escaped"], c80["null_raptor_En_le_2"], c160["max_dx_escaped"],
                c160["max_escape_angle_rad"], c160["null_raptor_escaped"],
                c160["null_raptor_En_le_2"], o["C40_80"]["median"], o["C80_160"]["median"],
                ok_lit ? "PASS" : "FAIL", ok_att ? "PASS" : "FAIL")
    end
    legres["class_counts_blp_C80"] = [count(i -> b80[i].cls == c, 1:nr) for c in 1:5]
    legres["wall_s"] = time() - t0
    @printf("[%s] BLP order(40-80,80-160) %.2f %.2f, wall %.0f s\n", leg,
            conv["blp"]["order_per_ray"]["C40_80"]["median"],
            conv["blp"]["order_per_ray"]["C80_160"]["median"], legres["wall_s"])
    rays_json[leg] = legres
    term_json[leg] = legterm
end
rays_json["pass_default_build"] = allpass
rays_json["thresholds"] = Dict("termination_identical_frac" => 0.999, "max_dx_escaped" => 1e-3,
                               "max_escape_angle_rad" => 1e-6, "max_null_norm_raptor" => 1e-8,
                               "median_order" => 3.5)
rays_json["acceptance_rule"] = "literal: dag.md R04 thresholds at C = 80. attributed (pass): class, samples and order literal; dx and angle literal at C = 80, or within 2 max(eps_R, eps_B) of each code's own error against the Vern7 reference at C = 80 (design.md section 5) and literal at C = 160; null norm on escaped rays <= max(1e-8, BLP) at C = 80 and <= 1e-8 at C = 160, and <= 1e-8 on every state with normal-frame energy <= 2 at C = 160; the capture tail (E_n up to 100) is reported, not thresholded"
tag = join(LEGSEL, "_")
write_json(joinpath(OUTDIR, "rays_$tag.json"), rays_json)
write_json(joinpath(OUTDIR, "termination_$tag.json"), term_json)
open(joinpath(get(ENV, "R04_CSV_DIR", mktempdir()), "convergence_$tag.csv"), "w") do io
    println(io, "leg,code,ray,kind," * join(["err_C$(Int(C))" for C in CS], ","))
    foreach(r -> println(io, r), conv_rows)
end
println(allpass ? "R04 rays PASS" : "R04 rays FAIL")
exit(allpass ? 0 : 1)
