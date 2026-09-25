# r06_reference.jl — BlackLightPlus half of the R06 reader acceptance (dag.md R06).
#
# Loads one AthenaK snapshot the way figures/fig_eht_bbh.jl does, draws a fixed set of
# probe points, and writes for each point the BlackLightPlus owning block, the
# interpolated primitives (`sample_primitives`), u^mu / b^mu (`primitive_four_vectors`
# with the SuperposedKerrSchild metric of the dump) and the plasma decode
# (`plasma_decode` with the driver's plasma, R_high = 40) together with the gates of
# `eight_coefficients`. The points file carries g_{mu nu} and g^{mu nu} per point, so the
# RAPTOR probe (model/athenak/tests/athenak_probe.c) evaluates the same four-vector
# transform on the same metric.
#
# Point classes (label file, one Int8 per point):
#   0 uniform in the ball R < 60
#   1 on and near block faces (offsets 0, +-1 ulp, +-1e-9 dx, +-dx/4, +-dx/2, +-dx) of
#     blocks that reach R < 100
#   2 on and near faces across which the refinement level changes
#   3 on block edges and corners
#   4 inside each hole's excision radius (1000 per hole)
#   5 on the domain boundary and outside the mesh
#   6 on cell centres and on midpoints between centres
#
# usage: julia --project=. -t 16 results/blp/raptor/tools/r06_reference.jl TAG OUTDIR [NPTS]
#   TAG in s455, s122, s093. Writes OUTDIR/TAG_{points,ref}.bin, TAG_class.bin,
#   TAG_meta.json.

using StaticArrays, Printf, Random, LinearAlgebra
using HDF5
using BlackLightPlus
const BLP = BlackLightPlus
const RAD = BLP.Radiation
const SIM = BLP.Simulation
const GEO = BLP.Geodesics

const DATA = "/data2/haiyangw/aurora/cbd_rel"
const SNAP_PATHS = Dict(
    "s455" => joinpath(DATA, "zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin"),
    "s093" => joinpath(DATA, "torus.mhd_w_bcc.00093.bin"),
    "s122" => joinpath(DATA, "torus.mhd_w_bcc.00122.athdf"),
)
# calibrated mass scales at R_high = 40 (figures/out/eht_bbh/*/records.json)
const MASS_SCALE = Dict("s455" => 2.274019497e28, "s093" => 2.252873787e20,
                        "s122" => 1.000498515e21)
const MSUN = 1.98847e33
const M_BH = 2 * 6.5e9 * MSUN
const R_HIGH = 40.0
const NCOL = 38

# ------------------------------------------------ dump header (fig_eht_bbh.jl copy)
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

function binary_setup(path)
    t_hdr, par = read_header(path)
    sep = _num(par, "problem", "sep", NaN); q = _num(par, "problem", "q", 1.0)
    a1 = _num(par, "problem", "a1", 0.0); a2 = _num(par, "problem", "a2", 0.0)
    th1 = _num(par, "problem", "th_a1", 0.0); th2 = _num(par, "problem", "th_a2", 0.0)
    gam = _num(par, "mhd", "gamma", 13 / 9)
    tlim = _num(par, "time", "tlim", NaN)
    t = (isfinite(tlim) && abs(tlim - t_hdr) <= 0.5) ? tlim : t_hdr
    Om = sep^-1.5
    phi = rem2pi(Om * t, RoundNearest)
    m1 = 1 / (1 + q); m2 = q / (1 + q)
    r1 = q / (1 + q) * sep; r2 = -sep / (1 + q)
    C1 = (r1 * cos(phi), r1 * sin(phi), 0.0)
    C2 = (r2 * cos(phi), r2 * sin(phi), 0.0)
    function spin_used(a, th, m)
        s = abs(th) < 1e-12 ? 1.0 : -1.0
        a == 0 && return 0.0
        abs(a) <= m && return s * abs(a)
        return s * min(abs(a), 1.0) * m
    end
    st = SuperposedKerrSchild([m1, m2], [spin_used(a1, th1, m1), spin_used(a2, th2, m2)], [C1, C2])
    return (st = st, gamma = gam, centers = (C1, C2), phi0 = phi,
            excise = (_num(par, "coord", "excise_1_rad", 1.0), _num(par, "coord", "excise_2_rad", 1.0)))
end

function scc_for(ms, rh, gam)
    cfg = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = ms, frequency_hz = 230e9,
                    polarized = true,
                    plasma = PlasmaConfig(gamma = gam, rat_low = 1.0, rat_high = rh, mu = 0.5,
                                          sigma_cut = 1.0),
                    max_steps = 4000)
    return BLP._prepared_scc(cfg, SIM.units(M_BH, ms))
end

# ------------------------------------------------------------------ probe points
function draw_points(snap, bs, rng, npts)
    lo = snap.ext_lo; hi = snap.ext_hi; nmb = snap.nmb
    dmin = SVector(minimum(e[1] for e in lo), minimum(e[2] for e in lo), minimum(e[3] for e in lo))
    dmax = SVector(maximum(e[1] for e in hi), maximum(e[2] for e in hi), maximum(e[3] for e in hi))
    blockdx(m) = (hi[m] - lo[m]) ./ SVector(snap.ni, snap.nj, snap.nk)
    near = [m for m in 1:nmb if norm(clamp.(zero(SVector{3,Float64}), lo[m], hi[m])) < 100.0]
    pts = SVector{3,Float64}[]; cls = Int8[]
    push1!(p, c) = (push!(pts, p); push!(cls, Int8(c)))
    n0 = round(Int, 0.40npts); n1 = round(Int, 0.25npts); n2 = round(Int, 0.20npts)
    n3 = round(Int, 0.05npts); n6 = round(Int, 0.07npts)
    # 0: uniform ball
    while count(==(0), cls) < n0
        p = SVector(ntuple(_ -> 120rand(rng) - 60, 3))
        norm(p) < 60 && push1!(p, 0)
    end
    offs(dx) = (0.0, dx * 1e-9, -dx * 1e-9, dx / 4, -dx / 4, dx / 2, -dx / 2, dx, -dx)
    function face_point(m, ax, side, scale)
        f = side == 0 ? lo[m][ax] : hi[m][ax]
        k = rand(rng, 1:11)
        c = k <= 9 ? f + offs(scale)[k] : (k == 10 ? nextfloat(f) : prevfloat(f))
        v = [lo[m][a] + rand(rng) * (hi[m][a] - lo[m][a]) for a in 1:3]
        v[ax] = c
        return SVector{3,Float64}(v)
    end
    # 1: faces of blocks near the holes
    for _ in 1:n1
        m = rand(rng, near); ax = rand(rng, 1:3)
        push1!(face_point(m, ax, rand(rng, 0:1), blockdx(m)[ax]), 1)
    end
    # 2: faces with a level change
    jumps = Tuple{Int,Int,Int}[]
    for m in near, ax in 1:3, side in 0:1
        c = (lo[m] + hi[m]) / 2
        d = blockdx(m)[ax] * 1e-3
        v = collect(c); v[ax] = side == 0 ? lo[m][ax] - d : hi[m][ax] + d
        n = SIM.meshblock_index(snap, v[1], v[2], v[3])
        n != 0 && snap.levels[n] != snap.levels[m] && push!(jumps, (m, ax, side))
    end
    for _ in 1:n2
        m, ax, side = rand(rng, jumps)
        push1!(face_point(m, ax, side, 2 * blockdx(m)[ax]), 2)
    end
    # 3: edges and corners
    for _ in 1:n3
        m = rand(rng, near)
        v = [rand(rng, Bool) ? lo[m][a] : hi[m][a] for a in 1:3]
        rand(rng, Bool) && (a = rand(rng, 1:3); v[a] = lo[m][a] + rand(rng) * (hi[m][a] - lo[m][a]))
        push1!(SVector{3,Float64}(v), 3)
    end
    # 4: excision balls
    for A in 1:2
        C = SVector{3,Float64}(bs.centers[A]); R = bs.excise[A]
        k = 0
        while k < 1000
            p = SVector(ntuple(_ -> 2rand(rng) - 1, 3))
            norm(p) < 1 && (push1!(C + R * p, 4); k += 1)
        end
    end
    # 6: cell centres and centre midpoints
    for _ in 1:n6
        m = rand(rng, near); d = blockdx(m); xl = snap.x_left[m]
        n = SVector(snap.ni, snap.nj, snap.nk)
        half = rand(rng, Bool) ? 0.5 : 0.0
        v = [xl[a] + (rand(rng, 0:n[a]-1) + (half > 0 && rand(rng, Bool) ? 0.5 : 0.0)) * d[a] for a in 1:3]
        push1!(SVector{3,Float64}(v), 6)
    end
    # 5: domain boundary and outside
    while length(pts) < npts
        v = [dmin[a] + rand(rng) * (dmax[a] - dmin[a]) for a in 1:3]
        ax = rand(rng, 1:3)
        k = rand(rng, 1:4)
        v[ax] = k == 1 ? dmin[ax] : k == 2 ? dmax[ax] : k == 3 ? dmax[ax] + 1.0 : dmin[ax] - 1.0
        push1!(SVector{3,Float64}(v), 5)
    end
    return pts, cls, length(jumps)
end

# ------------------------------------------------------------------ reference
function reference(snap, bs, scc, gam, pts)
    n = length(pts)
    P = zeros(35, n); R = zeros(NCOL, n)
    Threads.@threads for q in 1:n
        x, y, z = pts[q]
        x4 = SVector(0.0, x, y, z)
        gcov, gcon = GEO.metric_and_inverse(bs.st, x4)
        P[1:3, q] .= (x, y, z)
        for i in 1:4, j in 1:4
            P[3 + 4(i-1) + j, q] = gcov[i, j]
            P[19 + 4(i-1) + j, q] = gcon[i, j]
        end
        mb = SIM.meshblock_index(snap, x, y, z)
        s = SIM.sample_primitives(snap, x4)
        R[1, q] = mb - 1
        R[2:9, q] .= (s.rho, s.uu[1], s.uu[2], s.uu[3], s.eint, s.bb[1], s.bb[2], s.bb[3])
        pgas = SIM.pgas_from_eint(s.eint, gam)
        R[10, q] = pgas
        mb == 0 && continue
        # no real u^mu where g^00 >= 0 (superposed metric, deep inside a spinning
        # hole's horizon near its ring): BlackLightPlus throws there and never samples
        # it (the march stops at the horizon); recorded as NaN four-vectors, no emission
        if !(-gcon[1, 1] > 0.0)
            R[11:27, q] .= NaN
            R[37:38, q] .= NaN
            continue
        end
        fv = SIM.primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3], s.bb[1], s.bb[2], s.bb[3],
                                        gcov, gcon)
        R[11:14, q] .= fv.ucon; R[15:18, q] .= fv.ucov
        R[19:22, q] .= fv.bcon; R[23:26, q] .= fv.bcov
        R[27, q] = fv.b_sq
        emit = 0.0
        if s.rho > 0.0 && pgas > 0.0 && fv.b_sq > 0.0
            pd = RAD.plasma_decode(scc, s.rho, pgas, fv.b_sq)
            R[28:34, q] .= (pd.n_e_cgs, pd.bb_cgs, pd.theta_e, pd.sigma, pd.beta_inv,
                            pd.rho_cgs, pd.pgas_cgs)
            R[36, q] = pd.omit
            emit = pd.omit ? 0.0 : 1.0
        end
        R[35, q] = emit
        R[37, q] = dot(fv.ucon, fv.ucov)
        R[38, q] = dot(fv.bcon, fv.ucov)
    end
    return P, R
end

function main()
    tag = ARGS[1]; outdir = ARGS[2]
    npts = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100_000
    path = SNAP_PATHS[tag]
    mkpath(outdir)
    bs = binary_setup(path)
    gam = bs.gamma
    GC.gc()
    t_load = @elapsed snap = endswith(path, ".bin") ? SIM.athenak_bin_snapshot(path; gamma = gam) :
                                                      SIM.AthenaKSnapshot(path; gamma = gam)
    GC.gc()
    rss_load = Sys.maxrss() / 2^30
    t_tbl = @elapsed SIM.leaf_table(snap)
    mode = SIM.leaf_table(snap) !== nothing ? "table" :
           snap.leafcache.hash !== nothing ? "hash" : "scan"
    rng = MersenneTwister(20260925)
    pts, cls, njump = draw_points(snap, bs, rng, npts)
    scc = scc_for(MASS_SCALE[tag], R_HIGH, gam)
    reference(snap, bs, scc, gam, pts[1:10])               # compile
    t_ref = @elapsed P, R = reference(snap, bs, scc, gam, pts)
    open(joinpath(outdir, "$(tag)_points.bin"), "w") do io
        write(io, Int64(length(pts)), Int64(1)); write(io, P)
    end
    open(joinpath(outdir, "$(tag)_ref.bin"), "w") do io
        write(io, Int64(length(pts)), Int64(NCOL)); write(io, R)
    end
    open(io -> write(io, cls), joinpath(outdir, "$(tag)_class.bin"), "w")
    open(joinpath(outdir, "$(tag)_meta.json"), "w") do io
        @printf(io, """{"tag": "%s", "file": "%s", "gamma": %.17g, "m_bh_g": %.17g, "mass_scale_g": %.17g,
 "R_high": %.1f, "centers": [[%.17g, %.17g, %.17g], [%.17g, %.17g, %.17g]], "a_used": [%.17g, %.17g],
 "excise_rad": [%.17g, %.17g], "nmb": %d, "locate": "%s", "level_jump_faces": %d, "npoints": %d,
 "blp_load_s": %.4f, "blp_locate_build_s": %.4f, "blp_peak_rss_gb": %.4f, "blp_sample_s": %.4f,
 "blp_threads": %d, "julia": "%s"}\n""",
                tag, path, gam, M_BH, MASS_SCALE[tag], R_HIGH, bs.centers[1]..., bs.centers[2]...,
                bs.st.spin..., bs.excise..., snap.nmb, mode, njump, length(pts),
                t_load, t_tbl, rss_load, t_ref, Threads.nthreads(), string(VERSION))
    end
    @printf("%s: %d points, %d level-jump faces, BLP load %.2f s (peak RSS %.2f GB), locate %s\n",
            tag, length(pts), njump, t_load, rss_load, mode)
end

main()
