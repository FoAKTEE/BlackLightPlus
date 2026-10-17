# r07_blp.jl — polarized 230 GHz images of the binary GRMHD snapshot 00455 for the
# image comparison with RAPTOR (figure fig_raptor_cmp).
#
# Physics: AthenaK dump torus.mhd_w_bcc.00455.bin (gamma from the dump) on the static
# SuperposedKerrSchild binary of the dump time (fig_eht_bbh.jl: holes of mass
# 1/(1 + q) and q/(1 + q) at +q/(1 + q) sep (cos φ0, sin φ0, 0) and -sep/(1 + q) (cos φ0,
# sin φ0, 0), φ0 = Ω t wrapped to (-π, π], Ω = sep^-3/2; header spins a_A along +z
# (th_a = 0) or -z (th_a = π), |a_A| > m_A imaged as χ = min(|a_A|, 1), a = χ m_A).
# Each hole 6.5e9 M_sun (code mass 1.3e10 M_sun), D = 16.8 Mpc, 230 GHz; thermal
# electrons, R_low = 1, R_high = 40, μ = 0.5, σ_cut = 1; density mass scale
# 2.274019497e28 g (the 17° calibration of fig_eht_bbh.jl for 00455, R07_MASS_SCALE
# overrides). Plane-parallel camera at d = 1000 M, φ_o = 0, FOV 56 M; fused polarized
# march with h = min_A(r_A - r_h,A)/C, at most 50 C steps, escape beyond 1100 M.
#
# Per (inclination, C, npix) it writes <out>/blp_i<inc>_C<C>_n<npix>/:
#   I.npy, Q.npy, U.npy, V.npy   cgs intensity, [iβ, iα]
#   class.npy                     ray end class: 1/2 captured by hole 1/2, 3 escaped,
#                                 4 step budget, 5 non-finite (from `shadow`)
#   record.json                   setup, fluxes, ray classes, stored-state count, the
#                                 sampled-state safety statistics (smallest r_A - r_h,A
#                                 and r_A over every march state, states with g^tt >= 0),
#                                 phase wall times (load, setup = prepare, render = the
#                                 fused trace and transfer, output), BLP commit, threads
# and <out>/bbh.in, the binary geometry in RAPTOR's bbh.in format.
#
# Usage, from the repository root:
#   julia -t 48 --project=BLP BLP/figures/r07_blp.jl [out] [incs] [Cs] [npixs] [repeats]
# defaults: out = BLP/out/raptor/R07/rerun_head, incs 90, Cs 320, npixs 480, repeats 1.
# `repeats` > 1 repeats every render and keeps the best wall per phase; a 24² render
# runs first so that the timed phases exclude compilation. Every product runs with
# backend = :cpu. R07_SNAP selects the dump (.bin or .athdf).

using StaticArrays, Printf, Dates, LinearAlgebra
using HDF5
using BLP
using BLP.Numerics: Units
using BLP.Data: athenak_bin_snapshot
using BLP.Spacetime: ks_radius, horizon_radius, inverse_metric, march_control, rk4_step,
                     camera_ray, MarchBudget

const ROOT = dirname(dirname(@__DIR__))
const SNAP = get(ENV, "R07_SNAP",
    "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "BLP", "out", "raptor", "R07", "rerun_head")
const INCS = parse.(Float64, _list(length(ARGS) >= 2 ? ARGS[2] : "90"))
const CS = parse.(Float64, _list(length(ARGS) >= 3 ? ARGS[3] : "320"))
const NPIXS = parse.(Int, _list(length(ARGS) >= 4 ? ARGS[4] : "480"))
const REPEATS = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1

const MSUN = 1.98847e33
const M_BH = 2 * 6.5e9 * MSUN
const D_CM = 16.8e6 * 3.0856775814913673e18
const NU = 230.0e9
const DIST = 1000.0
const ESCAPE = 1.1 * DIST
const FOV = 56.0
const R_LOW = 1.0
const R_HIGH = 40.0
const MU = 0.5
const SIGMA_CUT = 1.0
const MASS_SCALE = parse(Float64, get(ENV, "R07_MASS_SCALE", "2.274019497e28"))
const L_UNIT = Units(M_BH, 1.0).L_unit
nmax_for(C) = round(Int, 50 * C)

const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- BLP/src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# ---------------------------------------------------------------- minimal JSON writer
function _js(x, ind = "")
    x === nothing && return "null"
    x isa Bool && return x ? "true" : "false"
    x isa Integer && return string(x)
    x isa AbstractFloat && return isfinite(x) ? @sprintf("%.10g", x) : "null"
    x isa Symbol && return _js(string(x))
    x isa AbstractString && return "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
    x isa NamedTuple && return _js(Dict(string(k) => v for (k, v) in pairs(x)), ind)
    x isa Union{AbstractVector,Tuple} && return "[" * join([_js(v, ind) for v in x], ", ") * "]"
    in2 = ind * "  "
    return "{\n" * join([in2 * _js(string(k)) * ": " * _js(v, in2)
                         for (k, v) in sort(collect(x); by = p -> string(first(p)))], ",\n") *
           "\n" * ind * "}"
end
write_json(path, x) = open(io -> println(io, _js(x)), path, "w")

# ---------------------------------------------------------------- binary of the dump
"Dump time and parameter blocks (`par[block][key]`, values as strings, comments cut)."
function read_header(path)
    tt = NaN; lines = String[]
    if endswith(path, ".bin")
        text = ""
        open(path, "r") do io
            H = -1
            while H < 0
                l = readline(io)
                startswith(l, "  time=") && (tt = parse(Float64, strip(l[8:end])))
                startswith(l, "  header offset=") && (H = parse(Int, strip(l[17:end])))
            end
            text = String(read(io, H))
        end
        lines = split(text, '\n')
    else
        tt, lines = h5open(path, "r") do h
            Float64(read_attribute(h, "Time")), String.(read_attribute(h, "Header"))
        end
    end
    par = Dict{String,Dict{String,String}}(); blk = ""
    for l in lines
        s = strip(l)
        if startswith(s, "<") && endswith(s, ">")
            blk = s[2:end-1]; par[blk] = get(par, blk, Dict{String,String}())
        elseif occursin('=', s) && !isempty(blk)
            k, v = split(s, '='; limit = 2)
            par[blk][strip(k)] = strip(split(v, '#')[1])
        end
    end
    return tt, par
end
_num(par, b, k, d) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : d

"The static binary at the dump time (the dump's own `sep`, `q`, `a1`, `a2`, `th_a*`)."
function binary_setup(path)
    t_hdr, par = read_header(path)
    sep = _num(par, "problem", "sep", NaN); q = _num(par, "problem", "q", 1.0)
    a1 = _num(par, "problem", "a1", 0.0); a2 = _num(par, "problem", "a2", 0.0)
    gam = _num(par, "mhd", "gamma", 13 / 9)
    tlim = _num(par, "time", "tlim", NaN)
    t = isfinite(tlim) && abs(tlim - t_hdr) <= 0.5 ? tlim : t_hdr
    Om = sep^-1.5
    phi = rem2pi(Om * t, RoundNearest)
    m1 = 1 / (1 + q); m2 = q / (1 + q)
    r1 = q / (1 + q) * sep; r2 = -sep / (1 + q)
    C1 = (r1 * cos(phi), r1 * sin(phi), 0.0); C2 = (r2 * cos(phi), r2 * sin(phi), 0.0)
    th1 = _num(par, "problem", "th_a1", 0.0); th2 = _num(par, "problem", "th_a2", 0.0)
    function spin_used(a, th, m)
        s = abs(th) < 1e-12 ? 1.0 : abs(th - pi) < 1e-12 ? -1.0 : error("tilted spin th_a = $th")
        a == 0 && return 0.0
        abs(a) <= m && return s * abs(a)
        return s * min(abs(a), 1.0) * m
    end
    au = (spin_used(a1, th1, m1), spin_used(a2, th2, m2))
    st = SuperposedKerrSchild([m1, m2], [au...], [C1, C2])
    return (st = st, t = t, phi0 = phi, masses = (m1, m2), centers = (C1, C2), gamma = gam,
            sep = sep, a_header = (a1, a2), a_used = au)
end

function write_bbh_in(path, bs; C = 80.0)
    c1, c2 = bs.centers
    open(path, "w") do io
        println(io, "nholes 2")
        println(io, "mass $(repr(bs.masses[1])) $(repr(bs.masses[2]))")
        println(io, "spin $(repr(bs.a_used[1])) $(repr(bs.a_used[2]))")
        println(io, "center1 $(repr(c1[1])) $(repr(c1[2])) $(repr(c1[3]))")
        println(io, "center2 $(repr(c2[1])) $(repr(c2[2])) $(repr(c2[3]))")
        println(io, "C $(repr(C))")
        println(io, "nmax $(nmax_for(C))")
        println(io, "escape $(repr(ESCAPE))")
        println(io, "horizon_eps 0.001")
        println(io, "capture_redshift 100.0")
        println(io, "cam_dist $(repr(DIST))")
    end
end

# ---------------------------------------------------------------- plan and diagnostics
function run_config(bs, inc, C, npix)
    return RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = NU,
                     camera = CameraConfig(inclination_deg = inc, distance = DIST, fov = FOV,
                                           resolution = npix),
                     plasma = PlasmaConfig(gamma = bs.gamma, rat_low = R_LOW, rat_high = R_HIGH,
                                           mu = MU, sigma_cut = SIGMA_CUT),
                     max_steps = nmax_for(C), C = C, escape_factor = 1.1)
end

"""
Ray end classes from `shadow`: the final state of a captured ray is assigned to the hole
with the smallest r_A - r_h,A; a step-limited ray whose final state meets the capture
rule is captured too, else escaped beyond `escape` or budget-limited; a non-finite final
state is class 5. Returns (class matrix [iβ, iα], stored states per ray = steps + 1).
"""
function end_classes(plan)
    sh = shadow(plan; backend = :cpu)
    st = plan.spacetime
    cap_only = MarchBudget{Float64}(plan.budget.C, Inf, Inf, plan.budget.nmax)
    cls = similar(sh.fate, Float64)
    for k in eachindex(sh.fate)
        y = sh.final[k]
        if !all(isfinite, y)
            cls[k] = 5
            continue
        end
        f = sh.fate[k]
        cap = f == 1 || (f == 0 && march_control(st, y, Inf, cap_only).stop)
        if cap
            x = SVector(y[1], y[2], y[3], y[4])
            d = [ks_radius(st, x, A) - horizon_radius(st, A) for A in eachindex(st.mass)]
            cls[k] = argmin(d)
        elseif sqrt(y[2]^2 + y[3]^2 + y[4]^2) > plan.escape
            cls[k] = 3
        else
            cls[k] = 4
        end
    end
    return cls, sh.steps .+ 1, sh.run.seconds
end

"""
Smallest r_A - r_h,A and r_A over every state of the fixed-control march of each camera
ray (escape beyond `escape`, no emission early-out), and the number of states with
g^tt >= 0. The march is the rule of the imaging products (`march_control`, `rk4_step`).
"""
function sample_safety(plan)
    st = plan.spacetime
    b = MarchBudget{Float64}(plan.budget.C, plan.escape, Inf, plan.budget.nmax)
    c = plan.pixels.centres
    n = plan.camera.npix
    N = n * n
    gap = fill(Inf, N); rmin = fill(Inf, N); npos = zeros(Int, N)
    Threads.@threads :dynamic for k in 1:N
        α, β = c[(k - 1) ÷ n + 1], c[(k - 1) % n + 1]
        y, ok = camera_ray(st, plan.frame, α, β)
        ok || continue
        r_prev = Inf
        g = Inf; rm = Inf; np = 0
        for it in 0:b.nmax
            if all(isfinite, y)
                x = SVector(y[1], y[2], y[3], y[4])
                for A in eachindex(st.mass)
                    r = ks_radius(st, x, A)
                    rm = min(rm, r)
                    g = min(g, r - horizon_radius(st, A))
                end
                inverse_metric(st, x)[1, 1] >= 0 && (np += 1)
            end
            it == b.nmax && break
            m = march_control(st, y, r_prev, b)
            m.stop && break
            r_prev = m.R
            y = rk4_step(st, y, m.h)
        end
        gap[k] = g; rmin[k] = rm; npos[k] = np
    end
    return Dict("min_horizon_gap" => minimum(gap), "min_rA" => minimum(rmin),
                "states_gtt_nonneg" => sum(npos))
end

"One image: Stokes planes, class map, phase walls, stored states, safety statistics."
function render_once(bs, snap, inc, C, npix; diagnostics = true)
    t0 = time()
    plan = prepare(run_config(bs, inc, C, npix); spacetime = bs.st, matter = snap, mode = IQUV())
    t1 = time()
    img = render(plan; backend = :cpu)
    t2 = time()
    tim = Dict{String,Any}("setup" => t1 - t0, "render" => t2 - t1, "trace" => nothing,
                           "transfer" => nothing)
    diagnostics || return img, nothing, tim, 0, nothing
    cls, ns, tsh = end_classes(plan)
    tim["shadow"] = tsh
    ts = time()
    safety = sample_safety(plan)
    tim["safety"] = time() - ts
    return img, cls, tim, sum(ns), safety
end

function main()
    mkpath(OUT)
    t0 = time()
    bs = binary_setup(SNAP)
    snap = endswith(SNAP, ".bin") ? athenak_bin_snapshot(SNAP; gamma = bs.gamma) :
           AthenaKSnapshot(SNAP; gamma = bs.gamma)
    load_s = time() - t0
    write_bbh_in(joinpath(OUT, "bbh.in"), bs)
    @printf("BLP %s | threads %d | load %.2f s | phi0 %.10f | centres %s %s | a_used %s | gamma %.12g | maxrss %.1f GB\n",
            GIT_SHA, Threads.nthreads(), load_s, bs.phi0, string(bs.centers[1]),
            string(bs.centers[2]), string(bs.a_used), bs.gamma, Sys.maxrss() / 2^30)
    tw = time()
    render_once(bs, snap, 17.0, 80.0, 24)                    # compilation
    jit_s = time() - tw
    @printf("warm-up (24^2, includes compilation) %.1f s\n", jit_s)
    for npix in NPIXS, inc in INCS, C in CS
        dir = joinpath(OUT, @sprintf("blp_i%02d_C%d_n%d", round(Int, inc), round(Int, C), npix))
        best = Dict{String,Any}("setup" => Inf, "render" => Inf, "output" => Inf,
                                "trace" => nothing, "transfer" => nothing)
        walls = Float64[]
        local img, cls, nsamp, safety, tdiag
        for rep in 1:REPEATS
            GC.gc()
            tr = time()
            img, cls, tim, nsamp, safety = render_once(bs, snap, inc, C, npix)
            to = time()
            mkpath(dir)
            for (nm, a) in zip(("I", "Q", "U", "V"), (img.I, img.Q, img.U, img.V))
                write_npy(joinpath(dir, "$(nm).npy"), a)
            end
            write_npy(joinpath(dir, "class.npy"), cls)
            tim["output"] = time() - to
            push!(walls, tim["setup"] + tim["render"] + tim["output"])
            for k in ("setup", "render", "output")
                best[k] = min(best[k], tim[k])
            end
            tdiag = Dict("shadow" => tim["shadow"], "safety" => tim["safety"])
        end
        pixsr = (FOV / npix * L_UNIT / D_CM)^2
        flux = [sum(a) * pixsr / 1e-23 for a in (img.I, img.Q, img.U, img.V)]
        nrays = npix^2
        wall = minimum(walls)
        rec = Dict("code" => "BLP", "git_sha" => GIT_SHA, "snapshot" => SNAP,
                   "inclination_deg" => inc, "C" => C, "nmax" => nmax_for(C), "npix" => npix,
                   "fov_M" => FOV, "distance_M" => DIST, "mass_scale_g" => MASS_SCALE,
                   "M_bh_g" => M_BH, "D_cm" => D_CM, "nu_Hz" => NU, "R_low" => R_LOW,
                   "R_high" => R_HIGH, "sigma_cut" => SIGMA_CUT, "gamma" => bs.gamma,
                   "phi0_rad" => bs.phi0, "centers" => [collect(c) for c in bs.centers],
                   "a_header" => collect(bs.a_header), "a_used" => collect(bs.a_used),
                   "sample_safety" => safety,
                   "flux_IQUV_Jy" => flux, "threads" => Threads.nthreads(),
                   "backend" => "cpu: render(plan; backend = :cpu), fused polarized march " *
                                "over $(img.run.executor.kind) executor",
                   "julia" => string(VERSION), "blp_version" => string(img.run.version),
                   "repeats" => REPEATS, "walls_s" => walls,
                   "timing_best_s" => merge(best, Dict("load" => load_s, "jit_warmup" => jit_s,
                                                       "image_wall" => wall)),
                   "diagnostics_wall_s" => tdiag,
                   "rays" => nrays, "samples" => nsamp, "rays_per_s" => nrays / wall,
                   "samples_per_s" => nsamp / best["render"],
                   "maxrss_GB" => Sys.maxrss() / 2^30,
                   "classes_cap1_cap2_esc_nmax_nonfinite" => [count(==(k), cls) for k in 1:5],
                   "when" => string(Dates.now()))
        write_json(joinpath(dir, "record.json"), rec)
        @printf("i=%2d C=%3d n=%d: F %.5f Jy (Q %+.5f U %+.5f V %+.5f) | setup %.2f render %.2f output %.2f s | %.0f rays/s | classes %s | min r_A-r_h %.2e, g^tt>=0 states %d\n",
                inc, C, npix, flux..., best["setup"], best["render"], best["output"],
                nrays / wall, string(rec["classes_cap1_cap2_esc_nmax_nonfinite"]),
                safety["min_horizon_gap"], safety["states_gtt_nonneg"])
        flush(stdout)
    end
end
main()
