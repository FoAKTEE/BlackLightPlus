# r07_blp.jl — BlackLightPlus leg of the R07/R08 comparison (dag.md R07, R08): s455
# polarized 230 GHz images with the fig_eht_bbh.jl physics (M87-per-hole scaling,
# R_low 1, R_high 40, sigma cut 1, gamma from the dump, calibrated mass scale of the
# figure record), plane-parallel camera at d = 1000, FOV 56 M, over a ladder of step
# constants C (h = min_A(r_A - r_h,A)/C) and resolutions. The step budget scales with
# C (nmax = 50 C, i.e. 4000 at C = 80), so every level resolves the same rays.
#
# Per (inclination, C, npix) it writes <out>/blp_i<inc>_C<C>_n<npix>/{I,Q,U,V,class}.npy
# (M[iβ, iα], cgs intensity; class 1/2 captured by hole 1/2, 3 escaped, 4 budget,
# 5 non-finite) and record.json with the phase timings (load, setup = camera rays,
# trace, transfer, output), rays/s, samples and peak RSS. Also writes <out>/bbh.in, the
# binary geometry of the dump in RAPTOR's bbh.in format (full precision).
#
# Usage: julia --project=. -t N r07_blp.jl <out> <incs> <Cs> <npixs> [repeats]
#   e.g.  r07_blp.jl out/raptor/R07/blp 17,60,90 40,80,160,320 480 1
# `repeats` > 1 repeats every render and keeps the best wall per phase (R08 timing);
# a 24^2 warm-up render runs first so that the timed phases exclude compilation.

using StaticArrays, Printf, Dates, Statistics, LinearAlgebra
using BlackLightPlus
const BLP = BlackLightPlus
const RAD = BLP.Radiation
const SIM = BLP.Simulation
const GEO = BLP.Geodesics
include(joinpath(@__DIR__, "jsonw.jl"))

const SNAP = get(ENV, "R07_SNAP",
    "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const OUT = ARGS[1]
_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
const INCS = parse.(Float64, _list(ARGS[2]))
const CS = parse.(Float64, _list(ARGS[3]))
const NPIXS = parse.(Int, _list(ARGS[4]))
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
const MASS_SCALE = parse(Float64, get(ENV, "R07_MASS_SCALE", "2.274019497e28"))  # figures/out/eht_bbh/s455_i17_Rh40
const L_UNIT = SIM.units(M_BH, 1.0).L_unit
nmax_for(C) = round(Int, 50 * C)

# ---------------------------------------------------------------- binary (fig_eht_bbh.jl)
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
    par = Dict{String,Dict{String,String}}(); blk = ""
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
_num(par, b, k, d) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : d

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
    (a1 == 0 && a2 == 0) || error("R07 expects the zero-spin s455 dump")
    st = SuperposedKerrSchild([m1, m2], [0.0, 0.0], [C1, C2])
    return (st = st, t = t, phi0 = phi, masses = (m1, m2), centers = (C1, C2), gamma = gam, sep = sep)
end

function write_bbh_in(path, bs; C = 80.0)
    c1, c2 = bs.centers
    open(path, "w") do io
        println(io, "nholes 2")
        println(io, "mass $(repr(bs.masses[1])) $(repr(bs.masses[2]))")
        println(io, "spin 0.0 0.0")
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

# ---------------------------------------------------------------- render
function end_class(st, y::SVector{8,Float64}, C)
    all(isfinite, y) || return 5
    x = SVector(y[1], y[2], y[3], y[4])
    if GEO.superposed_control(st, y, C)[1]
        d = [GEO.ks_radius_hole(st, x, A) - GEO.horizon_radius_hole(st, A) for A in eachindex(st.mass)]
        return argmin(d)
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESCAPE && return 3
    return 4
end

function scc_for(gam)
    cfg = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = NU,
                    polarized = true,
                    plasma = PlasmaConfig(gamma = gam, rat_low = R_LOW, rat_high = R_HIGH, mu = MU,
                                          sigma_cut = SIGMA_CUT))
    return BLP._prepared_scc(cfg, SIM.units(M_BH, MASS_SCALE))
end

make_sampler(snap, gam) = x -> begin
    s = SIM.sample_primitives(snap, x)
    (rho = s.rho, pgas = SIM.pgas_from_eint(s.eint, gam), uu = s.uu, bb = s.bb, inside = s.inside)
end

"One image: returns (img[4, npix, npix], class matrix, timings, samples)."
function render_once(bs, sampler, scc, inc, C, npix)
    t0 = time()
    cam = PlaneParallelCamera(inc, DIST, FOV, npix)
    rays = camera_rays(bs.st, cam)
    up = BLP._camera_up(cam)
    t1 = time()
    states, dls = trace_rays_sampled(bs.st, rays; C = C, nmax = nmax_for(C), escape = ESCAPE)
    t2 = time()
    n = length(states)
    S = zeros(4, n)
    Threads.@threads :dynamic for i in 1:n
        s4 = RAD.polarized_march(bs.st, scc, sampler, states[i], dls[i], NU, L_UNIT; cam_up = up)
        @inbounds for c in 1:4
            S[c, i] = s4[c]
        end
    end
    t3 = time()
    img = zeros(4, npix, npix)
    for c in 1:4
        img[c, :, :] .= reshape(S[c, :], npix, npix)
    end
    cls = [end_class(bs.st, s[1], C) for s in states]   # states[1] is the terminal state
    nsamp = sum(length, states)
    return img, reshape(Float64.(cls), npix, npix),
           Dict("setup" => t1 - t0, "trace" => t2 - t1, "transfer" => t3 - t2), nsamp
end

function main()
    mkpath(OUT)
    t0 = time()
    bs = binary_setup(SNAP)
    snap = athenak_bin_snapshot(SNAP; gamma = bs.gamma)
    load_s = time() - t0
    write_bbh_in(joinpath(OUT, "bbh.in"), bs)
    sampler = make_sampler(snap, bs.gamma)
    scc = scc_for(bs.gamma)
    @printf("BLP | threads %d | load %.2f s | phi0 %.10f | centres %s %s | gamma %.12g\n",
            Threads.nthreads(), load_s, bs.phi0, string(bs.centers[1]), string(bs.centers[2]), bs.gamma)
    tw = time()
    render_once(bs, sampler, scc, 17.0, 80.0, 24)            # compilation
    jit_s = time() - tw
    @printf("warm-up (24^2, includes compilation) %.1f s\n", jit_s)
    sha = try readchomp(`git -C $(dirname(dirname(dirname(dirname(@__DIR__))))) rev-parse HEAD`) catch; "unknown" end
    for npix in NPIXS, inc in INCS, C in CS
        best = Dict("setup" => Inf, "trace" => Inf, "transfer" => Inf, "output" => Inf)
        walls = Float64[]
        local img, cls, nsamp
        for rep in 1:REPEATS
            GC.gc()
            tr = time()
            img, cls, tim, nsamp = render_once(bs, sampler, scc, inc, C, npix)
            dir = joinpath(OUT, @sprintf("blp_i%02d_C%d_n%d", round(Int, inc), round(Int, C), npix))
            to = time()
            mkpath(dir)
            for (c, nm) in enumerate(("I", "Q", "U", "V"))
                write_npy(joinpath(dir, "$(nm).npy"), img[c, :, :])
            end
            write_npy(joinpath(dir, "class.npy"), cls)
            tim["output"] = time() - to
            push!(walls, time() - tr)
            for k in keys(best)
                best[k] = min(best[k], tim[k])
            end
        end
        dir = joinpath(OUT, @sprintf("blp_i%02d_C%d_n%d", round(Int, inc), round(Int, C), npix))
        pixsr = (FOV / npix * L_UNIT / D_CM)^2
        flux = [sum(img[c, :, :]) * pixsr / 1e-23 for c in 1:4]
        nrays = npix^2
        wall = minimum(walls)
        rec = Dict("code" => "BlackLightPlus", "git_sha" => sha, "snapshot" => SNAP,
                   "inclination_deg" => inc, "C" => C, "nmax" => nmax_for(C), "npix" => npix,
                   "fov_M" => FOV, "distance_M" => DIST, "mass_scale_g" => MASS_SCALE,
                   "M_bh_g" => M_BH, "D_cm" => D_CM, "nu_Hz" => NU, "R_low" => R_LOW,
                   "R_high" => R_HIGH, "sigma_cut" => SIGMA_CUT, "gamma" => bs.gamma,
                   "phi0_rad" => bs.phi0, "centers" => [collect(c) for c in bs.centers],
                   "flux_IQUV_Jy" => flux, "threads" => Threads.nthreads(),
                   "repeats" => REPEATS, "walls_s" => walls,
                   "timing_best_s" => merge(best, Dict("load" => load_s, "jit_warmup" => jit_s,
                                                       "image_wall" => wall)),
                   "rays" => nrays, "samples" => nsamp, "rays_per_s" => nrays / wall,
                   "samples_per_s" => nsamp / (best["trace"] + best["transfer"]),
                   "maxrss_GB" => Sys.maxrss() / 2^30,
                   "classes_cap1_cap2_esc_nmax_nonfinite" => [count(==(k), cls) for k in 1:5],
                   "when" => string(Dates.now()))
        write_json(joinpath(dir, "record.json"), rec)
        @printf("i=%2d C=%3d n=%d: F %.5f Jy (Q %+.5f U %+.5f V %+.5f) | setup %.2f trace %.2f transfer %.2f output %.2f s | %.0f rays/s | classes %s\n",
                inc, C, npix, flux..., best["setup"], best["trace"], best["transfer"], best["output"],
                nrays / wall, string(rec["classes_cap1_cap2_esc_nmax_nonfinite"]))
        flush(stdout)
    end
end
main()
