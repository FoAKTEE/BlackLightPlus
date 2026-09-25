# fig_eht_bbh.jl — EHT-style 230 GHz mock images (full IQUV) of binary-black-hole
# GRMHD snapshots (AthenaK dynbbh runs), CPU path.
#
# Scaling: each hole is M87*-like, m_hole = 6.5e9 Msun, so the code mass unit (the total
# binary mass) is M = 1.3e10 Msun; D = 16.8 Mpc (theta_g = GM/(c^2 D) = 7.64 uas, t_g =
# GM/c^3 = 17.8 h): a hole's ring (~10.4 m_hole = 5.2 M) spans ~40 uas, twice the 20 uas
# beam. nu = 230 GHz. Plane-parallel camera at distance 1000 M,
# observer azimuth along the simulation +x axis, inclination i from the orbital
# angular momentum (+z); FOV and pixels from BLP_FOV / BLP_RES.
#
# Binary: read from each dump's own parameter block (bin par dump or athdf Header):
# sep, q, a1, a2, th_a*, and the dump time. AthenaK dynbbh (analytic trajectory,
# use_traj_table = false) places hole 1 (mass 1/(1+q)) at +q/(1+q) sep (cos Om t,
# sin Om t, 0) and hole 2 (mass q/(1+q)) at -sep/(1+q) (cos Om t, sin Om t, 0) with
# Om = sep^-3/2; the spin vector is a (sin th cos ph, sin th sin ph, cos th) with `a` the
# Kerr parameter in units of the total mass. The image metric is the static
# SuperposedKerrSchild at that phase (no puncture boost). A header spin with
# |a| > m_A (a superextremal Kerr-Schild term in the simulation) cannot be traced; it is
# imaged as the dimensionless chi = a (a_used = chi m_A), recorded as `spin_mapping`.
#
# Plasma: thermal eDF, R_low = 1, R_high from BLP_RHIGH (fiducial 40) and the survey,
# T_i/T_e = (R_high b^2 + R_low)/(1 + b^2), b = beta, total-pressure split (use_p), mu =
# 0.5, n_e = n_i; sigma_cut = 1 (emission omitted where b^2/rho > 1); gamma from the dump.
#
# Flux calibration: mass_scale (density unit mass) is root-found per snapshot at the
# fiducial inclination so the 230 GHz flux in the FOV is 1.0 Jy (~0.5 Jy per hole).
# Unpolarized ladders of mass scales (one reverse march per ray carries every lane)
# bracket the root on a stride-4 then a full pixel set; the polarized render at the interpolated root is
# checked and, if off by more than BLP_FTOL, moved by one secant step with the ladder's
# local slope. Other inclinations reuse that mass_scale.
#
# Legs per snapshot: load; binary setup; accretion-rate proxy (rest-mass flux through
# r = 5 spheres co-moving with each puncture); per inclination: geodesics once, ray end
# classes (capture per hole, escape, budget), calibration (fiducial inclination),
# polarized render, diagnostics (flux inside r < 2 of the punctures, wide-FOV flux);
# the R_high survey on the fiducial snapshot at the fiducial inclination.
#
# Env: BLP_SNAPS (s455,s093,s122), BLP_INCS (17,60,90), BLP_FID (s455), BLP_FID_INC (17),
# BLP_RHIGH (40), BLP_RH_SURVEY (1,10,160), BLP_RES (480), BLP_FOV (56), BLP_FTOL (0.02),
# BLP_WIDE (1: wide-FOV flux check), BLP_OUT (figures/out/eht_bbh).
# Threads come from `julia -t N`.
# Run:  julia --project=. -t 64 figures/fig_eht_bbh.jl
# Plot: python3 figures/plot_eht_bbh.py

using StaticArrays, Printf, Dates, Statistics, LinearAlgebra
using HDF5
using BlackLightPlus
const BLP = BlackLightPlus
const RAD = BLP.Radiation
const SIM = BLP.Simulation
const GEO = BLP.Geodesics

const REPO = dirname(@__DIR__)
const DATA = "/data2/haiyangw/aurora/cbd_rel"
const SNAP_PATHS = Dict(
    "s455" => joinpath(DATA, "zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin"),
    "s093" => joinpath(DATA, "torus.mhd_w_bcc.00093.bin"),
    "s122" => joinpath(DATA, "torus.mhd_w_bcc.00122.athdf"),
)
_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
const SNAPS = _list(get(ENV, "BLP_SNAPS", "s455,s093,s122"))
const INCS = parse.(Float64, _list(get(ENV, "BLP_INCS", "17,60,90")))
const FID = get(ENV, "BLP_FID", "s455")
const FID_INC = parse(Float64, get(ENV, "BLP_FID_INC", "17"))
const RH_FID = parse(Float64, get(ENV, "BLP_RHIGH", "40"))
const RH_SURVEY = parse.(Float64, _list(get(ENV, "BLP_RH_SURVEY", "1,10,160")))
const RES = parse(Int, get(ENV, "BLP_RES", "480"))
const FOV = parse(Float64, get(ENV, "BLP_FOV", "56"))
const FTOL = parse(Float64, get(ENV, "BLP_FTOL", "0.02"))
const WIDE = get(ENV, "BLP_WIDE", "1") == "1"
const OUT = get(ENV, "BLP_OUT", joinpath(REPO, "figures", "out", "eht_bbh"))

const MSUN = 1.98847e33
const M_HOLE = 6.5e9 * MSUN               # each hole (q = 1)
const M_BH = 2 * M_HOLE                    # code mass unit: total binary mass
const D_CM = 16.8e6 * 3.0856775814913673e18
const NU = 230.0e9
const F_TARGET = 1.0                     # Jy (total, ~0.5 Jy per hole)
const DIST = 1000.0
const ESCAPE = 1.1 * DIST
const CVAL = 80.0
const MAXSTEPS = 4000
const R_LOW = 1.0
const MU = 0.5
const SIGMA_CUT = 1.0
const R_ACC = 5.0                        # accretion-rate proxy sphere radius [M]
const R_INNER_DIAG = 2.0                 # "near-puncture" diagnostic radius [M]
const WIDE_FOV = 4 * FOV
const WIDE_RES = 96

const UNITS1 = SIM.units(M_BH, 1.0)
const L_UNIT = UNITS1.L_unit
const T_UNIT = UNITS1.T_unit
const UAS_PER_M = L_UNIT / D_CM * (180 / pi) * 3600e6
const TG_DAYS = T_UNIT / 86400

const GIT_SHA = try
    sha = readchomp(`git -C $REPO rev-parse HEAD`)
    isempty(readchomp(`git -C $REPO status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

# ---------------------------------------------------------------- minimal JSON writer
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
_json(x::Bool) = x ? "true" : "false"
_json(x::Integer) = string(x)
_json(x::AbstractFloat) = isfinite(x) ? @sprintf("%.10g", x) : "null"
_json(x::Nothing) = "null"
_json(x::Symbol) = _json(string(x))
_json(x::Union{AbstractVector,Tuple}) = "[" * join(map(_json, collect(x)), ", ") * "]"
_json(x::NamedTuple) = _json(Dict(string(k) => v for (k, v) in pairs(x)))
function _json(d::AbstractDict, ind::String = "")
    in2 = ind * "  "
    items = [in2 * _json(string(k)) * ": " *
             (v isa AbstractDict ? _json(v, in2) : v isa NamedTuple ?
              _json(Dict(string(a) => b for (a, b) in pairs(v)), in2) : _json(v))
             for (k, v) in sort(collect(d); by = first)]
    return "{\n" * join(items, ",\n") * "\n" * ind * "}"
end
writejson(path, d) = open(io -> println(io, _json(d)), path, "w")

# ---------------------------------------------------------------- dump headers
"Dump time and parameter blocks (`par[block][key]`, values as strings, comments cut)."
function read_header(path)
    if endswith(path, ".bin")
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
        lines = split(text, '\n')
    else
        tt, lines = h5open(path, "r") do h
            Float64(read_attribute(h, "Time")), String.(read_attribute(h, "Header"))
        end
    end
    par = Dict{String,Dict{String,String}}()
    blk = ""
    for l in lines
        s = strip(l)
        if startswith(s, "<") && endswith(s, ">")
            blk = s[2:end-1]; par[blk] = get(par, blk, Dict{String,String}())
        elseif occursin('=', s) && !isempty(blk)
            k, v = split(s, '='; limit = 2)
            v = strip(split(v, '#')[1])
            par[blk][strip(k)] = v
        end
    end
    return tt, par
end
_num(par, b, k, dflt) = haskey(par, b) && haskey(par[b], k) ? parse(Float64, par[b][k]) : dflt

"Binary geometry at the dump time from the dump's own parameters (see file header)."
function binary_setup(path)
    t_hdr, par = read_header(path)
    sep = _num(par, "problem", "sep", NaN); q = _num(par, "problem", "q", 1.0)
    a1 = _num(par, "problem", "a1", 0.0); a2 = _num(par, "problem", "a2", 0.0)
    th1 = _num(par, "problem", "th_a1", 0.0); th2 = _num(par, "problem", "th_a2", 0.0)
    ph1 = _num(par, "problem", "ph_a1", 0.0); ph2 = _num(par, "problem", "ph_a2", 0.0)
    gam = _num(par, "mhd", "gamma", 13 / 9)
    traj = haskey(par, "problem") ? get(par["problem"], "use_traj_table", "0") : "0"
    traj in ("0", "false") || error("$path: use_traj_table = $traj (tabulated orbit) unsupported")
    tlim = _num(par, "time", "tlim", NaN)
    if isfinite(tlim) && abs(tlim - t_hdr) <= 0.5
        t = tlim; tsrc = "tlim (dump written at the run stop time; header time rounds it)"
    else
        t = t_hdr; tsrc = "header time (6 significant digits: +-0.5 M)"
    end
    Om = sep^-1.5
    phi = rem2pi(Om * t, RoundNearest)
    m1 = 1 / (1 + q); m2 = q / (1 + q)
    r1 = q / (1 + q) * sep; r2 = -sep / (1 + q)
    C1 = (r1 * cos(phi), r1 * sin(phi), 0.0)
    C2 = (r2 * cos(phi), r2 * sin(phi), 0.0)
    V1 = (-Om * C1[2], Om * C1[1], 0.0)
    V2 = (-Om * C2[2], Om * C2[1], 0.0)
    function spin_used(a, th, m)
        s = abs(th) < 1e-12 ? 1.0 : abs(th - pi) < 1e-12 ? -1.0 :
            error("$path: tilted spin th_a = $th unsupported (SuperposedKerrSchild spins lie on z)")
        a == 0 && return 0.0, "a = 0"
        abs(a) <= m && return s * abs(a), @sprintf("a = %.4g <= m = %.4g: used as is", a, m)
        chi = min(abs(a), 1.0)
        return s * chi * m, @sprintf("a = %.4g > m = %.4g (superextremal in the simulation metric): imaged as chi = %.4g, a_used = %.4g",
                                     a, m, chi, chi * m)
    end
    au1, map1 = spin_used(a1, th1, m1)
    au2, map2 = spin_used(a2, th2, m2)
    st = SuperposedKerrSchild([m1, m2], [au1, au2], [C1, C2])
    return (st = st, t = t, t_hdr = t_hdr, t_source = tsrc, sep = sep, q = q, Om = Om,
            phi0 = phi, masses = (m1, m2), centers = (C1, C2), velocities = (V1, V2),
            a_header = (a1, a2), spin_dirs = ((th1, ph1), (th2, ph2)), a_used = (au1, au2),
            spin_mapping = [map1, map2], gamma = gam,
            dfloor = _num(par, "mhd", "dfloor", NaN),
            excise_rad = (_num(par, "coord", "excise_1_rad", NaN),
                          _num(par, "coord", "excise_2_rad", NaN)),
            period = 2pi / Om)
end

# ---------------------------------------------------------------- physics config
function scc_for(ms, rh, gam)
    cfg = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = ms, frequency_hz = NU,
                    polarized = true,
                    plasma = PlasmaConfig(gamma = gam, rat_low = R_LOW, rat_high = rh, mu = MU,
                                          sigma_cut = SIGMA_CUT),
                    max_steps = MAXSTEPS)
    return BLP._prepared_scc(cfg, SIM.units(M_BH, ms))
end

make_sampler(snap, gam; rmask = 0.0, st = nothing) = x -> begin
    s = SIM.sample_primitives(snap, x)
    if rmask > 0 && s.inside
        for A in 1:2
            if GEO.ks_radius_hole(st, x, A) < rmask
                z3 = zero(SVector{3,Float64})
                return (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
            end
        end
    end
    (rho = s.rho, pgas = SIM.pgas_from_eint(s.eint, gam), uu = s.uu, bb = s.bb, inside = s.inside)
end

# ---------------------------------------------------------------- geodesics
# 1 = captured by hole 1, 2 = hole 2 (nearest horizon), 3 = escaped, 4 = step budget, 5 = non-finite
function end_class(st, y::SVector{8,Float64})
    all(isfinite, y) || return 5
    x = SVector(y[1], y[2], y[3], y[4])
    if GEO.superposed_control(st, y, CVAL)[1]
        d = [GEO.ks_radius_hole(st, x, A) - GEO.horizon_radius_hole(st, A)
             for A in eachindex(st.mass)]
        return argmin(d)
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESCAPE && return 3
    return 4
end

function trace(st, inc, fov, res)
    cam = PlaneParallelCamera(inc, DIST, fov, res)
    t0 = time()
    rays = camera_rays(st, cam)                         # α outer, β inner: reshape -> [iβ, iα]
    states, dls = trace_rays_sampled(st, rays; C = CVAL, nmax = MAXSTEPS, escape = ESCAPE)
    wall = time() - t0
    cls = Int8[end_class(st, s[1]) for s in states]
    ns = Int32[length(s) for s in states]
    return (cam = cam, states = states, dls = dls, cls = cls, nsamp = ns, wall = wall,
            res = res, fov = fov, up = BLP._camera_up(cam))
end
classhist(cls) = [count(==(c), cls) for c in 1:5]
pix_sr(fov, res) = (fov / res * L_UNIT / D_CM)^2
"Pixel index subset on a stride-s sub-lattice (flux estimator weight s^2)."
function stride_idx(res, s)
    o = (s + 1) ÷ 2
    return [(ia - 1) * res + ib for ia in o:s:res for ib in o:s:res]
end

# ---------------------------------------------------------------- transfer
"Unpolarized flux [Jy] for every mass scale in `mss` in one march per ray."
function flux_ladder(tr, st, sampler, mss, rh, gam; stride = 1)
    K = length(mss)
    gps = ntuple(k -> RAD._march_params(scc_for(mss[k], rh, gam), NU, L_UNIT,
                                        SVector(0.0, 0.0, 0.0, 0.0)), K)
    return _flux_ladder(tr, st, sampler, gps, Val(K), gam, stride)
end
function _flux_ladder(tr, st, sampler, gps, ::Val{K}, gam, stride) where {K}
    idx = stride_idx(tr.res, stride)
    acc = zeros(K, length(idx))
    smp = RAD._stored_sample(sampler, gam)
    Threads.@threads :dynamic for n in eachindex(idx)
        i = idx[n]
        ys = tr.states[i]
        v = RAD.reverse_unpolarized_march(ys[end], smp, SVector{K,Float64}, gps,
                                          RAD.StoredPath(st, ys, tr.dls[i]), length(ys))
        @inbounds for k in 1:K
            acc[k, n] = v[k]
        end
    end
    return vec(sum(acc; dims = 2)) .* (stride^2 * pix_sr(tr.fov, tr.res) / 1e-23)
end

"Full IQUV image (4 × res × res, [s, iβ, iα]) in cgs intensity."
function render_pol(tr, st, sampler, ms, rh, gam)
    scc = scc_for(ms, rh, gam)
    n = length(tr.states)
    S = zeros(4, n)
    t0 = time()
    Threads.@threads :dynamic for i in 1:n
        s4 = RAD.polarized_march(st, scc, sampler, tr.states[i], tr.dls[i], NU, L_UNIT;
                                 cam_up = tr.up)
        @inbounds for c in 1:4
            S[c, i] = s4[c]
        end
    end
    wall = time() - t0
    img = zeros(4, tr.res, tr.res)
    for c in 1:4
        img[c, :, :] .= reshape(S[c, :], tr.res, tr.res)
    end
    return img, wall
end
flux_Jy(I, tr) = sum(I) * pix_sr(tr.fov, tr.res) / 1e-23

# ---------------------------------------------------------------- calibration
"Interpolated log-log root of F(ms) = F_TARGET on a ladder, or nothing if unbracketed."
function ladder_root(mss, F)
    lm = log.(mss); lf = log.(max.(F, 1e-300)); lt = log(F_TARGET)
    for k in 1:length(mss)-1
        if (lf[k] - lt) * (lf[k+1] - lt) <= 0 && F[k] > 0 && F[k+1] > 0
            w = (lt - lf[k]) / (lf[k+1] - lf[k])
            return exp(lm[k] + w * (lm[k+1] - lm[k])), (lf[k+1] - lf[k]) / (lm[k+1] - lm[k])
        end
    end
    return nothing
end

function calibrate(tr, st, sampler, rh, gam, ms_guess)
    t0 = time()
    ladders = Any[]
    # coarse: stride 4, 1 dex spacing, widened until bracketed
    center = ms_guess
    root = nothing
    for pass in 1:4
        mss = center .* 10.0 .^ (-4.0:1.0:4.0)
        F = flux_ladder(tr, st, sampler, mss, rh, gam; stride = 4)
        push!(ladders, Dict("stride" => 4, "mass_scale" => mss, "flux_Jy" => F))
        @printf("    ladder s4: %s\n", join([@sprintf("%.2e:%.3g", m, f) for (m, f) in zip(mss, F)], " "))
        root = ladder_root(mss, F)
        root !== nothing && break
        center = F[end] < F_TARGET ? mss[end] * 1e4 : mss[1] * 1e-4
    end
    root === nothing && error("flux calibration: $(F_TARGET) Jy not bracketed")
    # fine: full pixel set, +-0.2 dex
    mss = root[1] .* 10.0 .^ (-0.2:0.1:0.2)
    F = flux_ladder(tr, st, sampler, mss, rh, gam; stride = 1)
    push!(ladders, Dict("stride" => 1, "mass_scale" => mss, "flux_Jy" => F))
    @printf("    ladder s1: %s\n", join([@sprintf("%.3e:%.4g", m, f) for (m, f) in zip(mss, F)], " "))
    root2 = ladder_root(mss, F)
    root2 === nothing && (root2 = root)
    # monotonic on the calibrated branch: strictly increasing on the fine ladder and on
    # the coarse ladder from its lowest lane through the bracketing pair (the coarse
    # ladder may turn over far above the root, where absorption by the outer flow wins)
    Fc = ladders[end-1]["flux_Jy"]
    kb = findfirst(>=(F_TARGET), Fc)
    mono = all(diff(F) .> 0) && kb !== nothing && all(diff(Fc[1:kb]) .> 0)
    return (ms = root2[1], slope = root2[2], ladders = ladders, monotonic = mono,
            wall = time() - t0)
end

# ---------------------------------------------------------------- diagnostics
project(c, inc) = (c[2], -c[1] * cosd(inc) + c[3] * sind(inc))   # (α, β) of a 3-point

function shadow_check(tr, st, inc, cen)
    ps = tr.fov / tr.res
    ax = [-tr.fov / 2 + (k - 0.5) * ps for k in 1:tr.res]
    C = reshape(tr.cls, tr.res, tr.res)                  # [iβ, iα]
    out = Dict{String,Any}()
    for A in 1:2
        pix = findall(==(Int8(A)), C)
        pa = project(cen[A], inc)
        if isempty(pix)
            out["hole$A"] = Dict("npix" => 0, "projected_M" => collect(pa))
            continue
        end
        ca = mean(ax[p[2]] for p in pix); cb = mean(ax[p[1]] for p in pix)
        out["hole$A"] = Dict("npix" => length(pix), "area_M2" => length(pix) * ps^2,
                             "centroid_M" => [ca, cb], "projected_M" => collect(pa),
                             "offset_M" => hypot(ca - pa[1], cb - pa[2]))
    end
    return out
end

"Rest-mass accretion rate [code] through a sphere of radius r co-moving with each puncture."
function mdot_proxy(snap, bs, r; nth = 64, nph = 128)
    st = bs.st
    res = Float64[]
    for A in 1:2
        c = bs.centers[A]; v = bs.velocities[A]
        acc = zeros(nth)
        Threads.@threads for it in 1:nth
            th = (it - 0.5) * pi / nth
            s = 0.0
            for ip in 1:nph
                ph = (ip - 0.5) * 2pi / nph
                n = SVector(sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
                x = SVector(0.0, c[1] + r * n[1], c[2] + r * n[2], c[3] + r * n[3])
                p = SIM.sample_primitives(snap, x)
                (p.inside && p.rho > 0) || continue
                gcov, gcon = GEO.metric_and_inverse(st, x)
                fv = SIM.primitive_four_vectors(p.uu[1], p.uu[2], p.uu[3], p.bb[1], p.bb[2],
                                                p.bb[3], gcov, gcon)
                u = fv.ucon
                un = (u[2] - u[1] * v[1]) * n[1] + (u[3] - u[1] * v[2]) * n[2] +
                     (u[4] - u[1] * v[3]) * n[3]
                s += p.rho * un * sqrt(-det(gcov)) * sin(th)
            end
            acc[it] = s
        end
        push!(res, -sum(acc) * r^2 * (pi / nth) * (2pi / nph))
    end
    return res
end

function save_image(dir, img, tr)
    mkpath(dir)
    for (c, nm) in enumerate(("I", "Q", "U", "V"))
        write_npy(joinpath(dir, "$(nm).npy"), img[c, :, :])
    end
    write_npy(joinpath(dir, "class.npy"), Float64.(reshape(tr.cls, tr.res, tr.res)))
end

function image_stats(img, tr)
    I = img[1, :, :]
    F = flux_Jy(I, tr)
    fin = all(isfinite, img)
    return Dict("flux_Jy" => F, "flux_Q_Jy" => flux_Jy(img[2, :, :], tr),
                "flux_U_Jy" => flux_Jy(img[3, :, :], tr), "flux_V_Jy" => flux_Jy(img[4, :, :], tr),
                "m_net_native" => hypot(sum(img[2, :, :]), sum(img[3, :, :])) / sum(I),
                "mabs_avg_native" => sum(hypot.(img[2, :, :], img[3, :, :])) / sum(I),
                "v_net_native" => sum(img[4, :, :]) / sum(I),
                "Imax_cgs" => maximum(I), "finite" => fin,
                "nonfinite_pixels" => count(k -> !all(isfinite, @view img[:, k[1], k[2]]),
                                            CartesianIndices(I)))
end

# ---------------------------------------------------------------- orientation check
# Single Kerr hole (m = 1, a = 0.9, spin +z) through the same camera / tracer / polarized
# march, fed an analytic prograde Keplerian ring (6 < R < 12, |z| < 1, rotating
# counter-clockwise about +z) threaded by a vertical field. With the observer on the
# +x axis the gas at y < 0 moves toward the camera, so the approaching (brighter) side
# must be image alpha < 0 = east = left on the sky; the EVPA of an optically thin
# emitter lies perpendicular to the projected field (north), i.e. east-west: net Q < 0.
# The plasma (p/rho = 0.1, R_high = R_low = 1, mass scale 1e24 g) is hot and optically
# thin at 230 GHz with little Faraday rotation.
function orientation_check(; inc = 60.0, res = 96, fov = 30.0, ms = 1.0e24)
    st1 = SuperposedKerrSchild([1.0], [0.9], [(0.0, 0.0, 0.0)])
    z3 = zero(SVector{3,Float64})
    samp = x -> begin
        R = hypot(x[2], x[3])
        if 6.0 < R < 12.0 && abs(x[4]) < 1.0
            v = 1 / sqrt(R); W = 1 / sqrt(1 - v^2)
            (rho = 1.0, pgas = 0.1, uu = SVector(-W * v * x[3] / R, W * v * x[2] / R, 0.0),
             bb = SVector(0.0, 0.0, 0.3), inside = true)
        else
            (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
        end
    end
    tr = trace(st1, inc, fov, res)
    img, _ = render_pol(tr, st1, samp, ms, 1.0, 13 / 9)     # R = 1: hot, optically thin
    I = img[1, :, :]
    h = res ÷ 2
    east = sum(I[:, 1:h]); west = sum(I[:, h+1:end])
    Q = sum(img[2, :, :]); U = sum(img[3, :, :])
    chi = 0.5 * atand(U, Q)
    ok = east > west && Q < 0
    @printf("orientation check (Kerr a=0.9, prograde ring, i=%g): I(alpha<0, east)/I(alpha>0, west) = %.3f | net EVPA %.1f deg E of N (Q/I %.3f, U/I %.3f) | %s\n",
            inc, east / west, chi, Q / sum(I), U / sum(I), ok ? "PASS" : "FAIL")
    return Dict("inclination_deg" => inc, "east_over_west_I" => east / west,
                "net_evpa_deg" => chi, "Q_over_I" => Q / sum(I), "U_over_I" => U / sum(I),
                "expected" => "east (alpha<0, approaching) brighter; EVPA ~ +-90 (perpendicular to the vertical field)",
                "mass_scale_g" => ms, "R_high" => 1.0,
                "pass" => ok)
end

# ---------------------------------------------------------------- main
println("BlackLightPlus ", GIT_SHA, " | threads=", Threads.nthreads(), " | res ", RES,
        " fov ", FOV, " M (", round(FOV * UAS_PER_M; digits = 1), " uas) | snaps ", SNAPS,
        " | incs ", INCS)
@printf("theta_g = %.4f uas, t_g = %.4f h\n", UAS_PER_M, T_UNIT / 3600)
mkpath(OUT)
index = Dict{String,Any}("images" => Any[], "git_sha" => GIT_SHA, "when" => string(Dates.now()),
                         "threads" => Threads.nthreads(), "res" => RES, "fov_M" => FOV,
                         "uas_per_M" => UAS_PER_M, "tg_days" => TG_DAYS)
ms_guess = Dict("s455" => 3e27, "s093" => 1e21, "s122" => 1e21)
index["orientation_check"] = orientation_check()

for tag in SNAPS
    path = SNAP_PATHS[tag]
    println("== ", tag, "  ", path)
    bs = binary_setup(path)
    @printf("  t = %.6f M (%s) | sep %.4g q %.4g | Om %.6g P %.2f M | phi0 %.6f rad\n", bs.t,
            bs.t_source, bs.sep, bs.q, bs.Om, bs.period, bs.phi0)
    @printf("  centres %s %s | spins %s\n", string(bs.centers[1]), string(bs.centers[2]),
            join(bs.spin_mapping, "; "))
    t0 = time()
    snap = endswith(path, ".bin") ? athenak_bin_snapshot(path; gamma = bs.gamma) :
           SIM.AthenaKSnapshot(path; gamma = bs.gamma)
    load_s = time() - t0
    @printf("  loaded in %.1f s: nmb %d, hash locate %s, maxrss %.1f GB\n", load_s, snap.nmb,
            SIM._leaf_hash(snap) !== nothing, Sys.maxrss() / 2^30)
    sampler = make_sampler(snap, bs.gamma)
    mdot_code = mdot_proxy(snap, bs, R_ACC)
    @printf("  Mdot proxy (code, r=%.0f): %s\n", R_ACC, string(mdot_code))
    snapinfo = Dict("tag" => tag, "file" => path, "time_M" => bs.t, "time_header_M" => bs.t_hdr,
                    "time_source" => bs.t_source, "time_days" => bs.t * TG_DAYS,
                    "sep_M" => bs.sep, "q" => bs.q, "Omega" => bs.Om, "period_M" => bs.period,
                    "phi0_rad" => bs.phi0, "masses" => collect(bs.masses),
                    "centers" => [collect(c) for c in bs.centers],
                    "puncture_velocity" => [collect(v) for v in bs.velocities],
                    "a_header" => collect(bs.a_header), "a_used" => collect(bs.a_used),
                    "spin_dirs_th_ph" => [collect(d) for d in bs.spin_dirs],
                    "spin_mapping" => bs.spin_mapping, "gamma" => bs.gamma,
                    "dfloor" => bs.dfloor, "excise_rad" => collect(bs.excise_rad),
                    "nmb" => snap.nmb, "load_s" => load_s,
                    "mdot_code_r5" => mdot_code)
    ms_cal = nothing; cal = nothing
    incs = sort(INCS; by = i -> i == FID_INC ? 0 : 1)       # fiducial inclination first
    for inc in incs
        tr = trace(bs.st, inc, FOV, RES)
        ch = classhist(tr.cls)
        @printf("  i=%g: traced %d rays in %.1f s | classes [cap1 cap2 esc nmax nonfinite] %s | mean samples %.0f\n",
                inc, length(tr.states), tr.wall, string(ch), mean(tr.nsamp))
        if ms_cal === nothing
            inc == FID_INC || error("fiducial inclination $FID_INC not in BLP_INCS")
            cal = calibrate(tr, bs.st, sampler, RH_FID, bs.gamma, ms_guess[tag])
            ms_cal = cal.ms
            @printf("  calibrated mass_scale %.6e g (slope dlnF/dlnms %.3f, monotonic %s) in %.1f s\n",
                    ms_cal, cal.slope, cal.monotonic, cal.wall)
        end
        rhs = (tag == FID && inc == FID_INC) ? vcat(RH_FID, RH_SURVEY) : [RH_FID]
        for rh in rhs
            ms = ms_cal; calrec = cal
            if rh != RH_FID
                calrec = calibrate(tr, bs.st, sampler, rh, bs.gamma, ms_cal)
                ms = calrec.ms
                @printf("  R_high=%g: mass_scale %.6e (slope %.3f)\n", rh, ms, calrec.slope)
            end
            img, wpol = render_pol(tr, bs.st, sampler, ms, rh, bs.gamma)
            F = flux_Jy(img[1, :, :], tr)
            secant = nothing
            if inc == FID_INC && abs(F / F_TARGET - 1) > FTOL
                ms2 = ms * exp(log(F_TARGET / F) / calrec.slope)
                img2, w2 = render_pol(tr, bs.st, sampler, ms2, rh, bs.gamma)
                F2 = flux_Jy(img2[1, :, :], tr)
                secant = Dict("ms_before" => ms, "flux_before" => F, "ms_after" => ms2,
                              "flux_after" => F2)
                @printf("    polarized flux %.4f Jy off target: secant ms %.4e -> %.4e, flux %.4f\n",
                        F, ms, ms2, F2)
                img, ms, F, wpol = img2, ms2, F2, wpol + w2
                rh == RH_FID && (ms_cal = ms)
            end
            st_ = image_stats(img, tr)
            @printf("  i=%g R_high=%g: pol render %.1f s (%.0f rays/s) | F=%.4f Jy | m_net %.4f <|m|> %.4f V/I %.2e | finite %s\n",
                    inc, rh, wpol, length(tr.states) / wpol, F, st_["m_net_native"],
                    st_["mabs_avg_native"], st_["v_net_native"], st_["finite"])
            # diagnostics on the fiducial R_high only
            diag = Dict{String,Any}()
            if rh == RH_FID
                sm_in = make_sampler(snap, bs.gamma; rmask = R_INNER_DIAG, st = bs.st)
                Fm = flux_ladder(tr, bs.st, sm_in, [ms], rh, bs.gamma; stride = 2)[1]
                Fu = flux_ladder(tr, bs.st, sampler, [ms], rh, bs.gamma; stride = 2)[1]
                diag["unpol_flux_stride2_Jy"] = Fu
                diag["unpol_flux_stride2_mask_r$(Int(R_INNER_DIAG))_Jy"] = Fm
                diag["frac_flux_within_r$(Int(R_INNER_DIAG))"] = 1 - Fm / Fu
                @printf("    flux from gas within r_KS < %.0f of a puncture: %.1f%% (unpol stride-2 %.4f vs masked %.4f Jy)\n",
                        R_INNER_DIAG, 100 * (1 - Fm / Fu), Fu, Fm)
                if WIDE
                    trw = trace(bs.st, inc, WIDE_FOV, WIDE_RES)
                    Fw = flux_ladder(trw, bs.st, sampler, [ms], rh, bs.gamma)[1]
                    trn = trace(bs.st, inc, FOV, WIDE_RES ÷ 4)   # same pixel size, inner FOV
                    Fn = flux_ladder(trn, bs.st, sampler, [ms], rh, bs.gamma)[1]
                    diag["wide_fov_M"] = WIDE_FOV; diag["wide_res"] = WIDE_RES
                    diag["wide_flux_Jy"] = Fw; diag["wide_inner_flux_Jy"] = Fn
                    diag["frac_flux_outside_fov"] = 1 - Fn / Fw
                    @printf("    wide FOV %.0f M: %.4f Jy, inner %.0f M at the same pixel: %.4f Jy (outside FOV %.1f%%)\n",
                            WIDE_FOV, Fw, FOV, Fn, 100 * (1 - Fn / Fw))
                    trw = trn = nothing
                end
            end
            sh = shadow_check(tr, bs.st, inc, bs.centers)
            name = @sprintf("%s_i%02d_Rh%g", tag, round(Int, inc), rh)
            dir = joinpath(OUT, name)
            save_image(dir, img, tr)
            units = SIM.units(M_BH, ms)
            mdot_cgs = mdot_code .* (ms / T_UNIT)
            ledd = 1.26e38 * (M_BH / MSUN)
            mdot_edd = ledd / (0.1 * 2.99792458e10^2)
            rec = Dict(
                "name" => name, "dir" => dir, "snapshot" => snapinfo,
                "git_sha" => GIT_SHA, "julia" => string(VERSION), "when" => string(Dates.now()),
                "threads" => Threads.nthreads(), "host_cpus" => Sys.CPU_THREADS,
                "backend" => "CPU: camera_rays -> trace_rays_sampled(::SuperposedKerrSchild) -> Radiation.polarized_march (threaded over rays)",
                "metric" => "static SuperposedKerrSchild (no puncture boost), masses $(collect(bs.masses)), a_used $(collect(bs.a_used)), centres at phi0",
                "camera" => Dict("inclination_deg" => inc, "fov_M" => FOV, "fov_uas" => FOV * UAS_PER_M,
                                 "res" => RES, "px_M" => FOV / RES, "px_uas" => FOV / RES * UAS_PER_M,
                                 "distance_M" => DIST, "observer_azimuth" => "simulation +x axis (phi_o = 0)",
                                 "orientation" => "beta (north, up) = projected +z (orbital L); alpha = +y = west (right); east = -alpha",
                                 "C" => CVAL, "max_steps" => MAXSTEPS),
                "scaling" => Dict("M_sun" => M_BH / MSUN, "m_hole_Msun" => M_HOLE / MSUN, "D_Mpc" => 16.8, "theta_g_uas" => UAS_PER_M,
                                  "t_g_hours" => T_UNIT / 3600, "L_unit_cm" => L_UNIT),
                "plasma" => Dict("edf" => "thermal", "R_low" => R_LOW, "R_high" => rh,
                                 "beta_crit" => 1.0, "Te_split" => "total pressure (use_p)", "mu" => MU,
                                 "ne_ni" => 1.0, "sigma_cut" => SIGMA_CUT, "gamma" => bs.gamma),
                "frequency_hz" => NU,
                "mass_scale_g" => ms, "dens_unit_gcc" => units.dens_unit, "Ne_unit_cc" => units.Ne_unit,
                "B_unit_G" => units.B_unit,
                "mdot_proxy" => Dict("radius_M" => R_ACC, "code" => mdot_code, "g_per_s" => mdot_cgs,
                                     "msun_per_yr" => mdot_cgs .* (3.15576e7 / MSUN),
                                     "eddington_ratio_eta0.1" => mdot_cgs ./ mdot_edd),
                "calibration" => Dict("target_Jy" => F_TARGET, "inclination_deg" => FID_INC,
                                      "own" => inc == FID_INC,
                                      "slope_dlnF_dlnms" => calrec.slope, "monotonic" => calrec.monotonic,
                                      "wall_s" => calrec.wall, "ladders" => calrec.ladders,
                                      "polarized_secant" => secant),
                "flux_Jy" => F, "stats_native" => st_,
                "ray_classes_cap1_cap2_esc_nmax_nonfinite" => classhist(tr.cls),
                "mean_samples" => mean(tr.nsamp), "trace_wall_s" => tr.wall, "pol_wall_s" => wpol,
                "shadow_check" => sh, "diagnostics" => diag,
                "projected_holes_M" => [collect(project(c, inc)) for c in bs.centers],
            )
            writejson(joinpath(dir, "records.json"), rec)
            push!(index["images"], Dict("name" => name, "tag" => tag, "inc" => inc, "R_high" => rh,
                                        "flux_Jy" => F, "mass_scale_g" => ms))
            writejson(joinpath(OUT, "index.json"), index)
            flush(stdout)
        end
        tr = nothing; GC.gc()
    end
    snap = nothing; sampler = nothing; GC.gc()
end
println("wrote ", OUT)
