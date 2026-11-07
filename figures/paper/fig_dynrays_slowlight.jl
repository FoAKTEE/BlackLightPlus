# fig_dynrays_slowlight.jl — fast and slow light for the GRMHD binary of fig_binary (a):
# AthenaK snapshot 00455 (q = 1, no spin, separation 25 M), 230 GHz at θ_o = 90° from the
# simulation +x axis, with the scaling, plasma model, camera and polarized march (IQUV) of
# figures/paper/fig_eht_bbh.jl; Stokes I is kept. Drawn by
# paper/scripts/plot_dynrays_slowlight.py into fig_dynrays_slowlight.pdf.
#
# fast — the paper's image: the static SuperposedKerrSchild with the holes at the dump
#        phase φ0 and the dump held fixed (every sample at the dump time t_s).
# slow — the moving binary: BoostedSuperposedKerrSchild on the CircularOrbit of the run
#        (ω = sep^-3/2), metric at the ray's coordinate time x^t, and the plasma sampled
#        at the emission time. The run's dumps are 785 M apart (its mhd_w_bcc output
#        dt, one orbital period) and one dump is on disk, so the evolving source is the
#        dump carried along the helical Killing vector ∂_t + ω ∂_φ of the orbit: the
#        primitives at (t, x) are those of the dump at R_z(−θ) x with velocity and field
#        rotated by R_z(θ), θ = ω (t − t_s). The gas then corotates with the holes and the
#        metric is invariant under the same map.
# Epoch: the camera time x^t = t_ref is identified with t_s (orbit phase φ0 there,
#        t0 = t_ref): t_ref is the intensity-weighted median, over the fast image, of the
#        time at which each pixel's ray crosses the sky plane through the centre of mass
#        (d̂·x = 0), from adaptive Vern7 traces of a stride-STRIDE sub-lattice in the
#        static metric (rays that do not cross are left out).
# Scan: at RES_SCAN², the slow image against fast images at epochs t_s + δ (holes at phase
#        φ0 + ωδ, dump rotated by ωδ), δ = −SCANSPAN:SCANSTEP:SCANSPAN; the L1 norm of the
#        difference over the fast flux per δ, for the whole image and its α < 0 and α > 0
#        halves (the sides of the front hole, hole 1, and of the rear hole).
#
# Writes figures/paper/out/dynrays/slowlight/ (BLP_SL_OUT): I_fast.npy, I_slow.npy
# ([iβ, iα], cgs intensity), tcross.npy (the sub-lattice crossing times, NaN where the
# ray does not cross) and records.json (setup, t_ref, hole positions, fluxes, walls, the
# agreement of I_fast with the saved fig_binary image, code SHA, threads), scan.json and
# delay.json (the light-travel time between each hole's line-of-sight plane and the sky
# plane, intensity-weighted median over the sub-lattice rays crossing both) and
# shift_scan.json (the slow image against the translated fast image at RES_SCAN, and the
# slow image without the boost).
# Run from the repository root:
#   julia -t 24 --project=. figures/paper/fig_dynrays_slowlight.jl
# Env: BLP_SL_SNAP (the 00455 dump), BLP_SL_RES (960), BLP_SL_FOV (56), BLP_SL_INC (90),
#      BLP_SL_MS (mass scale, g; default the 17° calibration of fig_eht_bbh.jl for 00455),
#      BLP_SL_STRIDE (4), BLP_SL_RES_SCAN (240), BLP_SL_SCANSPAN (30), BLP_SL_SCANSTEP (2),
#      BLP_SL_SHIFTMAX (12), BLP_SL_PARTS (images,scan,delay,shift; the last three read
#      records.json and I_fast.npy), BLP_SL_OUT.
using Dates, LinearAlgebra, Printf, Statistics, StaticArrays
using BlackLightPlus
using BlackLightPlus.Numerics: Units
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, SuperposedKerrSchild, CircularOrbit,
                                trace_geodesic, PlaneParallelCamera, CameraFrame, camera_ray,
                                ks_radius, horizon_radius
using BlackLightPlus.Matter: GridProvider, PlasmaPrimitives
import BlackLightPlus.Matter: primitives, domain_radius
using BlackLightPlus.Data: athenak_bin_snapshot, write_npy

const ROOT = dirname(dirname(@__DIR__))
const SNAP = get(ENV, "BLP_SL_SNAP",
                 "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const RES = parse(Int, get(ENV, "BLP_SL_RES", "960"))
const FOV = parse(Float64, get(ENV, "BLP_SL_FOV", "56"))
const INC = parse(Float64, get(ENV, "BLP_SL_INC", "90"))
const MS = parse(Float64, get(ENV, "BLP_SL_MS", "2.272378166e28"))   # eht_bbh/s455_i90_Rh40/records.json
const STRIDE = parse(Int, get(ENV, "BLP_SL_STRIDE", "4"))
const RES_SCAN = parse(Int, get(ENV, "BLP_SL_RES_SCAN", "240"))
const SCANSPAN = parse(Float64, get(ENV, "BLP_SL_SCANSPAN", "30"))
const SCANSTEP = parse(Float64, get(ENV, "BLP_SL_SCANSTEP", "2"))
const SHIFTMAX = parse(Int, get(ENV, "BLP_SL_SHIFTMAX", "12"))
const FIG_BINARY_I = joinpath(@__DIR__, "out", "eht_bbh", "s455_i90_Rh40", "I.npy")
const OUT = get(ENV, "BLP_SL_OUT", joinpath(@__DIR__, "out", "dynrays", "slowlight"))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# scaling, plasma and march of fig_eht_bbh.jl
const MSUN = 1.98847e33
const M_BH = 2 * 6.5e9 * MSUN
const D_CM = 16.8e6 * 3.0856775814913673e18
const NU = 230.0e9
const DIST = 1000.0
const CVAL = 80.0
const MAXSTEPS = 4000
const R_LOW = 1.0
const R_HIGH = 40.0
const MU = 0.5
const SIGMA_CUT = 1.0
const UNITS1 = Units(M_BH, 1.0)
const UAS_PER_M = UNITS1.L_unit / D_CM * (180 / pi) * 3600e6
const SI = sin(deg2rad(INC)); const CI = cos(deg2rad(INC))
const DHAT = SVector(SI, 0.0, CI)

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

# ---------------------------------------------------------------- dump header (as fig_eht_bbh.jl)
function read_header(path)
    tt = NaN; text = ""
    open(path, "r") do io
        H = -1
        while H < 0
            l = readline(io)
            startswith(l, "  time=") && (tt = parse(Float64, strip(l[8:end])))
            startswith(l, "  header offset=") && (H = parse(Int, strip(l[17:end])))
        end
        text = String(read(io, H))
    end
    par = Dict{String,Dict{String,String}}()
    blk = ""
    for l in split(text, '\n')
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
_num(par, b, k, dflt) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : dflt

# ---------------------------------------------------------------- the helical source
"""The dump carried along ∂_t + ω ∂_φ: the state at (t, x) is the dump's at R_z(−θ) x with
vectors rotated by R_z(θ), θ = θ0 + ω (t − t_ref), where camera time t_ref is the dump
time. ω = 0 gives the dump frozen at the rotation θ0 (a fast-light epoch t_s + θ0/Ω)."""
struct HelicalSource{P} <: GridProvider
    snap::P
    omega::Float64
    t_ref::Float64
    theta0::Float64
end
domain_radius(h::HelicalSource) = domain_radius(h.snap)
@inline function primitives(h::HelicalSource, x::SVector{4})
    θ = h.theta0 + h.omega * (Float64(x[1]) - h.t_ref)
    c = cos(θ); s = sin(θ)
    px = Float64(x[2]); py = Float64(x[3])
    p = primitives(h.snap, SVector(Float64(x[1]), c * px + s * py, -s * px + c * py, Float64(x[4])))
    p.inside || return p
    u = p.uu; b = p.bb
    return PlasmaPrimitives{Float64}(p.rho, SVector(c * u[1] - s * u[2], s * u[1] + c * u[2], u[3]),
                                     p.eint, SVector(c * b[1] - s * b[2], s * b[1] + c * b[2], b[3]), true)
end

# ---------------------------------------------------------------- run
config(gam; res = RES) = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
                        camera = CameraConfig(inclination_deg = INC, distance = DIST, fov = FOV,
                                              resolution = res),
                        plasma = PlasmaConfig(gamma = gam, rat_low = R_LOW, rat_high = R_HIGH,
                                              mu = MU, sigma_cut = SIGMA_CUT),
                        max_steps = MAXSTEPS, C = CVAL, escape_factor = 1.1)
flux_Jy(I) = sum(I) * ((FOV / size(I, 1)) * UNITS1.L_unit / D_CM)^2 / 1e-23
"Stokes I (cgs, [iβ, iα]) of the polarized render of `st` and `matter`, and the wall."
function stokes_i(st, matter, gam; res = RES)
    t0 = time()
    im = render(prepare(config(gam; res); spacetime = st, matter = matter, mode = IQUV()); backend = :cpu)
    return Matrix{Float64}(im.I), time() - t0
end
"the static superposition with the holes at phase φ"
static_at(masses, sep, φ) = (o = CircularOrbit(masses, sep; phase = φ);
                             SuperposedKerrSchild(collect(masses), [0.0, 0.0],
                                                  [Tuple(o(0.0)[A][1]) for A in 1:2]))
"weighted median of x with weights w"
function wmedian(x, w)
    p = sortperm(x); c = cumsum(w[p])
    return x[p[searchsortedfirst(c, c[end] / 2)]]
end

"coordinate time at which the saved path first crosses d̂·x = 0 going away from the camera"
function plane_time(us)
    s(u) = DHAT[1] * u[2] + DHAT[2] * u[3] + DHAT[3] * u[4]
    for k in 1:length(us)-1
        s1 = s(us[k]); s2 = s(us[k+1])
        if s1 > 0 && s2 <= 0
            w = s1 / (s1 - s2)
            return us[k][1] + w * (us[k+1][1] - us[k][1])
        end
    end
    return NaN
end

json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::Tuple) = json(collect(x))
json(d::AbstractDict) = "{" * join(("\"$k\": " * json(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"

"setup shared by both parts: (record, snapshot, γ, sep, Ω, φ0, masses, static metric at φ0)"
function setup()
    rec = Dict{String,Any}("code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()),
                           "snapshot" => SNAP, "res" => RES, "fov_M" => FOV, "inc_deg" => INC,
                           "distance_M" => DIST, "C" => CVAL, "max_steps" => MAXSTEPS,
                           "mass_scale_g" => MS, "frequency_hz" => NU, "R_low" => R_LOW,
                           "R_high" => R_HIGH, "sigma_cut" => SIGMA_CUT, "uas_per_M" => UAS_PER_M,
                           "stride" => STRIDE)
    t_hdr, par = read_header(SNAP)
    sep = _num(par, "problem", "sep", NaN); q = _num(par, "problem", "q", 1.0)
    a1 = _num(par, "problem", "a1", 0.0); a2 = _num(par, "problem", "a2", 0.0)
    gam = _num(par, "mhd", "gamma", 13 / 9)
    tlim = _num(par, "time", "tlim", NaN)
    q == 1 && a1 == 0 && a2 == 0 || error("this generator assumes q = 1 and no spin")
    t_s = (isfinite(tlim) && abs(tlim - t_hdr) <= 0.5) ? tlim : t_hdr
    Om = sep^-1.5
    phi0 = rem2pi(Om * t_s, RoundNearest)
    masses = (0.5, 0.5)
    orbit_s = CircularOrbit(masses, sep; phase = phi0)          # holes at the dump phase at t = 0
    centres = [Tuple(orbit_s(0.0)[A][1]) for A in 1:2]
    st_fast = SuperposedKerrSchild(collect(masses), [0.0, 0.0], centres)
    rec["t_s"] = t_s; rec["sep"] = sep; rec["omega"] = Om; rec["phi0"] = phi0
    rec["period"] = 2pi / Om; rec["gamma"] = gam; rec["hole_speed"] = Om * sep / 2
    rec["centres"] = [collect(c) for c in centres]
    rec["velocities"] = [collect(orbit_s(0.0)[A][2]) for A in 1:2]
    logf("snapshot t_s = %.4f, sep %.3g, ω %.6g, φ0 %.6f, γ %.4g; %d threads\n", t_s, sep, Om, phi0, gam,
         Threads.nthreads())

    t0 = time()
    snap = athenak_bin_snapshot(SNAP; gamma = gam)
    rec["load_s"] = time() - t0
    logf("loaded in %.1f s\n", rec["load_s"])
    return rec, snap, gam, sep, Om, phi0, masses, st_fast
end

function run_images()
    rec, snap, gam, sep, Om, phi0, masses, st_fast = setup()

    # fast light
    I_fast, rec["wall_fast_s"] = stokes_i(st_fast, snap, gam)
    write_npy(joinpath(OUT, "I_fast.npy"), I_fast)
    rec["flux_fast_Jy"] = flux_Jy(I_fast)
    logf("fast light: %.4f Jy in %.1f s\n", rec["flux_fast_Jy"], rec["wall_fast_s"])
    if isfile(FIG_BINARY_I) && RES == 960 && FOV == 56 && INC == 90
        ref = open(FIG_BINARY_I) do io           # npy v1, little-endian f8, C order
            magic = read(io, 8); hl = Int(read(io, UInt16)); hdr = String(read(io, hl))
            n = parse(Int, match(r"\((\d+),", hdr).captures[1])
            permutedims(reshape(reinterpret(Float64, read(io, 8 * n * n)), n, n))
        end
        rec["fig_binary_I_max_rel_diff"] = maximum(abs.(I_fast .- ref)) / maximum(ref)
        logf("against the saved fig_binary (a) image: max |ΔI| / max I = %.3e\n",
             rec["fig_binary_I_max_rel_diff"])
    end

    # epoch: intensity-weighted sky-plane crossing time on the stride sub-lattice
    cam = PlaneParallelCamera(inc_deg = INC, d = DIST, fov = FOV, npix = RES)
    fr = CameraFrame(cam)
    edges = range(-FOV / 2, FOV / 2; length = RES + 1)
    cen = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    idx = collect(1 + STRIDE ÷ 2:STRIDE:RES)
    n = length(idx)
    tc = fill(NaN, n, n)
    t0 = time()
    Threads.@threads for ia in 1:n
        for ib in 1:n
            y0, ok = camera_ray(st_fast, fr, cen[idx[ia]], cen[idx[ib]])
            ok || continue
            sol = trace_geodesic(st_fast, y0; λmax = 4000.0, escape = 1.1 * DIST, reltol = 1e-9,
                                 abstol = 1e-11, save_everystep = true)
            tc[ib, ia] = plane_time(sol.u)
        end
    end
    rec["wall_tcross_s"] = time() - t0
    write_npy(joinpath(OUT, "tcross.npy"), tc)
    w = [I_fast[idx[ib], idx[ia]] for ib in 1:n, ia in 1:n]
    ok = isfinite.(tc) .& (w .> 0)
    t_ref = wmedian(tc[ok], w[ok])
    rec["t_ref"] = t_ref
    rec["t_ref_flux_fraction"] = sum(w[ok]) / sum(w)
    rec["t_ref_weighted_quartiles"] = [wmedian(tc[ok], w[ok] .* (tc[ok] .<= t_ref)),
                                       wmedian(tc[ok], w[ok] .* (tc[ok] .>= t_ref))]
    rec["tcross_weighted_mean"] = sum(w[ok] .* tc[ok]) / sum(w[ok])
    rec["tcross_median_all"] = median(tc[isfinite.(tc)])
    logf("t_ref = %.3f (weighted quartiles %s, weighted mean %.3f, flux fraction %.3f; median over crossing rays %.3f) in %.1f s\n",
         t_ref, string(rec["t_ref_weighted_quartiles"]), rec["tcross_weighted_mean"],
         rec["t_ref_flux_fraction"], rec["tcross_median_all"], rec["wall_tcross_s"])

    # slow light on the moving binary
    orbit = CircularOrbit(masses, sep; phase = phi0, t0 = t_ref)
    st_slow = BoostedSuperposedKerrSchild(collect(masses), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
    I_slow, rec["wall_slow_s"] = stokes_i(st_slow, HelicalSource(snap, Om, t_ref, 0.0), gam)
    write_npy(joinpath(OUT, "I_slow.npy"), I_slow)
    rec["flux_slow_Jy"] = flux_Jy(I_slow)
    rec["nonfinite_slow"] = count(!isfinite, I_slow)
    logf("slow light: %.4f Jy in %.1f s (%d non-finite pixels)\n", rec["flux_slow_Jy"],
         rec["wall_slow_s"], rec["nonfinite_slow"])

    # image centroids (M) and the normalised difference
    a = [cen[ia] for ib in 1:RES, ia in 1:RES]; b = [cen[ib] for ib in 1:RES, ia in 1:RES]
    for (name, I) in (("fast", I_fast), ("slow", I_slow))
        rec["centroid_$name"] = [sum(a .* I) / sum(I), sum(b .* I) / sum(I)]
        for (h, sgn) in (("left", -1), ("right", 1))   # α < 0 and α > 0 halves
            m = sgn .* a .> 0
            rec["centroid_$(name)_$h"] = [sum((a .* I)[m]) / sum(I[m]), sum((b .* I)[m]) / sum(I[m])]
            rec["flux_$(name)_$(h)_Jy"] = flux_Jy(I[m])
        end
    end
    rec["l1_diff_over_flux"] = sum(abs.(I_slow .- I_fast)) / sum(I_fast)
    rec["max_abs_diff_over_peak"] = maximum(abs.(I_slow .- I_fast)) / maximum(I_fast)
    logf("centroid fast %s slow %s; |slow - fast|_1 / flux %.3f\n", string(rec["centroid_fast"]),
         string(rec["centroid_slow"]), rec["l1_diff_over_flux"])

    open(io -> println(io, json(rec)), joinpath(OUT, "records.json"), "w")
end

# the slow image at RES_SCAN against fast images at epochs t_s + δ; L1 of the difference
# over the fast flux, for the whole image and its α < 0 and α > 0 halves
function run_scan()
    rec, snap, gam, sep, Om, phi0, masses, _ = setup()
    m = match(r"\"t_ref\": ([-0-9.e]+)", read(joinpath(OUT, "records.json"), String))
    m === nothing && error("run the images part first: no t_ref in records.json")
    t_ref = parse(Float64, m.captures[1])
    st_slow = BoostedSuperposedKerrSchild(collect(masses), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                                          CircularOrbit(masses, sep; phase = phi0, t0 = t_ref))
    δs = collect(-SCANSPAN:SCANSTEP:SCANSPAN)
    Is, w_s = stokes_i(st_slow, HelicalSource(snap, Om, t_ref, 0.0), gam; res = RES_SCAN)
    c = -FOV / 2 .+ (collect(1:RES_SCAN) .- 0.5) .* (FOV / RES_SCAN)
    A = [c[ia] for ib in 1:RES_SCAN, ia in 1:RES_SCAN]
    masks = Dict("all" => trues(RES_SCAN, RES_SCAN), "alpha_neg" => A .< 0, "alpha_pos" => A .> 0)
    l1 = Dict(k => Float64[] for k in keys(masks))
    t0 = time()
    for δ in δs
        If, _ = stokes_i(static_at(masses, sep, phi0 + Om * δ), HelicalSource(snap, 0.0, 0.0, Om * δ),
                         gam; res = RES_SCAN)
        for (k, mk) in masks
            push!(l1[k], sum(abs.(Is[mk] .- If[mk])) / sum(If[mk]))
        end
        logf("  scan δ = %+5.1f: L1 all %.4f, α < 0 %.4f, α > 0 %.4f\n", δ, l1["all"][end],
             l1["alpha_neg"][end], l1["alpha_pos"][end])
    end
    out = Dict{String,Any}("res" => RES_SCAN, "offsets" => δs, "t_ref" => t_ref,
                           "wall_s" => time() - t0 + w_s, "code" => GIT_SHA,
                           "threads" => Threads.nthreads(), "date" => string(now()))
    for (k, v) in l1
        j = argmin(v)
        out["l1_$k"] = v; out["best_offset_$k"] = δs[j]; out["best_l1_$k"] = v[j]
        logf("scan %-9s: minimum %.4f at δ = %+.1f (δ = 0: %.4f)\n", k, v[j], δs[j],
             v[findfirst(==(0.0), δs)])
    end
    open(io -> println(io, json(out)), joinpath(OUT, "scan.json"), "w")
end

# the light-travel time from each hole's line-of-sight plane d̂·x = ℓ_A (ℓ_A = d̂·s_A at
# the dump) to the sky plane, along the rays of the stride sub-lattice in the static metric
# that cross both planes, as the median weighted by the fast-light intensity; with it, the
# displacement ω ℓ_A Δt_A of each hole's image across the line of sight
function run_delay()
    rec, snap, gam, sep, Om, phi0, masses, st_fast = setup()
    I_fast = let io = open(joinpath(OUT, "I_fast.npy"))
        read(io, 8); hl = Int(read(io, UInt16)); hdr = String(read(io, hl))
        n = parse(Int, match(r"\((\d+),", hdr).captures[1])
        M = permutedims(reshape(reinterpret(Float64, read(io, 8 * n * n)), n, n)); close(io); M
    end
    cam = PlaneParallelCamera(inc_deg = INC, d = DIST, fov = FOV, npix = RES)
    fr = CameraFrame(cam)
    edges = range(-FOV / 2, FOV / 2; length = RES + 1)
    cen = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    idx = collect(1 + STRIDE ÷ 2:STRIDE:RES)
    n = length(idx)
    ells = [DHAT[1] * c[1] + DHAT[2] * c[2] + DHAT[3] * c[3] for c in rec["centres"]]
    lag = [fill(NaN, n, n) for _ in 1:2]
    function tplane(us, ℓ)
        for k in 1:length(us)-1
            s1 = DHAT[1] * us[k][2] + DHAT[2] * us[k][3] + DHAT[3] * us[k][4] - ℓ
            s2 = DHAT[1] * us[k+1][2] + DHAT[2] * us[k+1][3] + DHAT[3] * us[k+1][4] - ℓ
            s1 > 0 && s2 <= 0 && return us[k][1] + s1 / (s1 - s2) * (us[k+1][1] - us[k][1])
        end
        return NaN
    end
    t0 = time()
    Threads.@threads for ia in 1:n
        for ib in 1:n
            y0, ok = camera_ray(st_fast, fr, cen[idx[ia]], cen[idx[ib]])
            ok || continue
            sol = trace_geodesic(st_fast, y0; λmax = 4000.0, escape = 1.1 * DIST, reltol = 1e-9,
                                 abstol = 1e-11, save_everystep = true)
            t_sky = tplane(sol.u, 0.0)
            for A in 1:2
                tA = tplane(sol.u, ells[A])
                lag[A][ib, ia] = ells[A] > 0 ? tA - t_sky : t_sky - tA
            end
        end
    end
    w = [I_fast[idx[ib], idx[ia]] for ib in 1:n, ia in 1:n]
    out = Dict{String,Any}("ell" => ells, "omega" => Om, "stride" => STRIDE, "wall_s" => time() - t0,
                           "code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()))
    for A in 1:2
        ok = isfinite.(lag[A]) .& (w .> 0)
        Δt = wmedian(lag[A][ok], w[ok])
        out["lag_$A"] = Δt
        out["lag_quartiles_$A"] = [wmedian(lag[A][ok], w[ok] .* (lag[A][ok] .<= Δt)),
                                   wmedian(lag[A][ok], w[ok] .* (lag[A][ok] .>= Δt))]
        out["lag_flux_fraction_$A"] = sum(w[ok]) / sum(w)
        out["shift_$A"] = Om * ells[A] * Δt * sign(ells[A])
        out["shift_flat_$A"] = Om * ells[A]^2
        logf("hole %d: ℓ = %+.3f, weighted median lag %.3f (quartiles %s, flux fraction %.3f): ω ℓ Δt = %.3f M (flat ω ℓ² = %.3f M)\n",
             A, ells[A], Δt, string(out["lag_quartiles_$A"]), out["lag_flux_fraction_$A"],
             out["shift_$A"], out["shift_flat_$A"])
    end
    open(io -> println(io, json(out)), joinpath(OUT, "delay.json"), "w")
end

# at RES_SCAN: the slow image against the δ = 0 fast image translated along α by k
# pixels (k = 0:SHIFTMAX), L1 over the fast flux per k; and the slow image with the holes'
# terms carried along the orbit without the boost (boost = false), its flux and L1 against
# the boosted slow image
function run_shift()
    rec, snap, gam, sep, Om, phi0, masses, st_fast = setup()
    m = match(r"\"t_ref\": ([-0-9.e]+)", read(joinpath(OUT, "records.json"), String))
    m === nothing && error("run the images part first: no t_ref in records.json")
    t_ref = parse(Float64, m.captures[1])
    orbit = CircularOrbit(masses, sep; phase = phi0, t0 = t_ref)
    zs = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    Is, _ = stokes_i(BoostedSuperposedKerrSchild(collect(masses), zs, orbit), HelicalSource(snap, Om, t_ref, 0.0),
                     gam; res = RES_SCAN)
    Iu, _ = stokes_i(BoostedSuperposedKerrSchild(collect(masses), zs, orbit; boost = false),
                     HelicalSource(snap, Om, t_ref, 0.0), gam; res = RES_SCAN)
    If, _ = stokes_i(st_fast, HelicalSource(snap, 0.0, 0.0, 0.0), gam; res = RES_SCAN)
    px = FOV / RES_SCAN
    ks = collect(0:SHIFTMAX)
    l1 = Float64[]
    for k in ks                                   # fast translated by +k pixels in α (columns)
        Ish = zeros(size(If)); Ish[:, 1+k:end] .= If[:, 1:end-k]
        push!(l1, sum(abs.(Is .- Ish)) / sum(If))
    end
    j = argmin(l1)
    out = Dict{String,Any}("res" => RES_SCAN, "px_M" => px, "shift_px" => ks, "l1_shifted" => l1,
                           "best_shift_px" => ks[j], "best_shift_M" => ks[j] * px, "best_l1" => l1[j],
                           "flux_fast_Jy" => flux_Jy(If), "flux_slow_Jy" => flux_Jy(Is),
                           "flux_slow_unboosted_Jy" => flux_Jy(Iu),
                           "l1_unboosted_vs_boosted" => sum(abs.(Iu .- Is)) / sum(Is),
                           "l1_slow_vs_fast" => l1[1], "t_ref" => t_ref, "code" => GIT_SHA,
                           "threads" => Threads.nthreads(), "date" => string(now()))
    logf("shift: L1 %.4f unshifted, minimum %.4f at %d px = %.3f M; flux fast %.4f slow %.4f slow unboosted %.4f Jy; L1(unboosted, boosted) %.4f\n",
         l1[1], l1[j], ks[j], ks[j] * px, out["flux_fast_Jy"], out["flux_slow_Jy"],
         out["flux_slow_unboosted_Jy"], out["l1_unboosted_vs_boosted"])
    open(io -> println(io, json(out)), joinpath(OUT, "shift_scan.json"), "w")
end

const PARTS = split(get(ENV, "BLP_SL_PARTS", "images,scan,delay,shift"), ",")
"images" in PARTS && run_images()
"scan" in PARTS && run_scan()
"delay" in PARTS && run_delay()
"shift" in PARTS && run_shift()
logf("wrote %s\n", OUT)
