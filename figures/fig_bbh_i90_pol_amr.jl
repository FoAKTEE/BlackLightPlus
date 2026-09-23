# fig_bbh_i90_pol_amr.jl — binary-black-hole GRMHD image, full IQUV, at i = 90° via
# value-based image-plane AMR on the CPU.
#
# Deck: AthenaK torus.mhd_w_bcc.00455.bin (gamma = 13/9) on a static
# SuperposedKerrSchild metric, m1 = m2 = 0.5, a1 = a2 = 0, centres
# ±12.5 (cos φ0, sin φ0, 0) with φ0 = -0.2233022; 230 GHz; i = 90°; fov = 40 M;
# distance 1000 M; C = 80; max_steps = 2500; m_bh = 6.5e9 Msun; mass_scale = 1e24 g
# (uncalibrated placeholder).
#
# AMR: render_amr(cfg, BIN; block, max_level, polarized = true, stokes = closure,
# abs_grad_cut, abs_grad_count = 5). The closure traces each coordinate batch with
# the binary fixed-control tracer (trace_rays_sampled(::SuperposedKerrSchild), step
# min_A(r_A - r_h,A)/C, capture on either hole) in chunks and runs the CPU polarized
# march (Radiation.polarized_march) threaded over rays.
#
# Indicator: abs_grad_cut = 1e-3 · max|∇I| on the level-0 (uniform base²) Stokes-I
# image of this deck, rendered unpolarized; abs_grad_count = 5 (White 2022).
#
# Legs: snapshot load; level-0 unpolarized probe and cut; the AMR render (per-level
# rays, walls and termination classes from the instrumented closure); a repeat of the
# level-0 polarized batch (bitwise determinism check and the measured uniform-frame
# rate); optionally a full AMR repeat (BLP_REPEAT=1). The figure's top row is the
# level-0 batch of the AMR render itself.
#
# Env: BLP_BBH_BIN, BLP_BASE_RES (256), BLP_BLOCK (8), BLP_MAXLEVEL (4),
# BLP_CHUNK (rays per trace chunk, 16384), BLP_REPEAT (0), BLP_OUT
# (figures/out/bbh_i90_pol_amr). Threads come from `julia -t N`.
# Run:  julia --project=. -t 112 figures/fig_bbh_i90_pol_amr.jl
# Plot: python3 figures/plot_bbh_i90_pol_amr.py

using StaticArrays, Printf, Dates, Statistics
using BlackLightPlus
const BLP = BlackLightPlus
const RAD = BLP.Radiation
const SIM = BLP.Simulation

const REPO = dirname(@__DIR__)
const BIN = get(ENV, "BLP_BBH_BIN",
    "/data2/haiyangw/aurora/cbd_rel/zoom_in_mad/SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin")
const PHI0 = -0.2233022
const GAMMA = 13.0 / 9.0
const CVAL = 80.0
const FOV = 40.0
const INC = 90.0
const DIST = 1000.0
const NU = 2.3e11
const MAXSTEPS = 2500
const M_BH = 6.5e9 * 1.989e33
const MASS_SCALE = 1.0e24
const BASE_RES = parse(Int, get(ENV, "BLP_BASE_RES", "256"))
const BLOCK = parse(Int, get(ENV, "BLP_BLOCK", "8"))
const MAXLEVEL = parse(Int, get(ENV, "BLP_MAXLEVEL", "4"))
const CHUNK = parse(Int, get(ENV, "BLP_CHUNK", "16384"))
const REPEAT = get(ENV, "BLP_REPEAT", "0") == "1"
const EFF = BASE_RES * 2^MAXLEVEL
const OUT = get(ENV, "BLP_OUT", joinpath(REPO, "figures", "out", "bbh_i90_pol_amr"))
const ESCAPE = 1.1 * DIST

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

# ---------------------------------------------------------------- deck
const C1 = 12.5 .* (cos(PHI0), sin(PHI0), 0.0)
const C2 = (-C1[1], -C1[2], 0.0)
const ST = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [C1, C2])

const CFG = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE,
                      frequency_hz = NU, polarized = true,
                      camera = CameraConfig(inclination_deg = INC, distance = DIST,
                                            fov = FOV, resolution = BASE_RES),
                      plasma = PlasmaConfig(gamma = GAMMA), max_steps = MAXSTEPS)
const CAM = PlaneParallelCamera(INC, DIST, FOV, BASE_RES)
const UNITS = SIM.units(M_BH, MASS_SCALE)
const SCC = BLP._prepared_scc(CFG, UNITS)
const CAM_UP = BLP._camera_up(CAM)

println("BlackLightPlus ", GIT_SHA, " | threads=", Threads.nthreads(),
        " | base ", BASE_RES, " block ", BLOCK, " max_level ", MAXLEVEL, " -> ", EFF, "^2 eff")
mkpath(OUT)
@printf("loading %s\n", BIN)
t0 = time()
const SNAP = athenak_bin_snapshot(BIN; gamma = GAMMA)
const LOAD_S = time() - t0
@printf("snapshot loaded in %.2f s (nmb=%d, leaf table %s)\n", LOAD_S, SNAP.nmb,
        leaf_table(SNAP) === nothing ? "over cap: scan locate" : "O(1)")

const SAMPLER = x -> begin
    s = SIM.sample_primitives(SNAP, x)
    (rho = s.rho, pgas = SIM.pgas_from_eint(s.eint, GAMMA), uu = s.uu, bb = s.bb,
     inside = s.inside)
end

# ---------------------------------------------------------------- ray end classes
# Classify the source end of a traced ray (states[1]) with the tracer's own capture
# rule: 1 = captured by hole 1, 2 = captured by hole 2 (nearest horizon), 3 = escaped,
# 4 = step budget exhausted, 5 = non-finite.
function _end_class(y::SVector{8,Float64})
    all(isfinite, y) || return 5
    x = SVector(y[1], y[2], y[3], y[4])
    if BLP.Geodesics.superposed_control(ST, y, CVAL)[1]
        d1 = ks_radius_hole(ST, x, 1) - horizon_radius_hole(ST, 1)
        d2 = ks_radius_hole(ST, x, 2) - horizon_radius_hole(ST, 2)
        return d1 <= d2 ? 1 : 2
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > ESCAPE && return 3
    return 4
end

"""
Stokes (4 × n) at image coordinates, CPU: chunked binary tracer + threaded march.
Returns (S, cls, nsamp): per-ray end class and sample count.
"""
function cpu_stokes(coords::Vector{Tuple{Float64,Float64}}; polarized::Bool)
    n = length(coords)
    S = zeros(4, n)
    cls = zeros(Int8, n)
    nsamp = zeros(Int32, n)
    for lo in 1:CHUNK:n
        hi = min(n, lo + CHUNK - 1)
        rays = camera_rays_at(ST, CAM, coords[lo:hi])
        states, dls = trace_rays_sampled(ST, rays; C = CVAL, nmax = MAXSTEPS,
                                         escape = ESCAPE)
        Threads.@threads :dynamic for k in eachindex(rays)
            i = lo + k - 1
            ys = states[k]
            cls[i] = _end_class(ys[1])
            nsamp[i] = length(ys)
            if polarized
                s4 = RAD.polarized_march(ST, SCC, SAMPLER, ys, dls[k], NU, UNITS.L_unit;
                                         cam_up = CAM_UP)
                S[1, i] = s4[1]; S[2, i] = s4[2]; S[3, i] = s4[3]; S[4, i] = s4[4]
            else
                S[1, i] = RAD.unpolarized_march(ST, SCC, SAMPLER, ys, dls[k], NU,
                                                UNITS.L_unit)
            end
        end
    end
    return S, cls, nsamp
end

classhist(cls) = [count(==(c), cls) for c in 1:5]

# ---------------------------------------------------------------- level-0 probe
ps0 = FOV / BASE_RES
const COORDS0 = [(-FOV / 2 + (i - 0.5) * ps0, -FOV / 2 + (j - 0.5) * ps0)
                 for i in 1:BASE_RES for j in 1:BASE_RES]           # α-major
t0 = time()
S0u, cls0, ns0 = cpu_stokes(COORDS0; polarized = false)
const PROBE_S = time() - t0
I0 = [S0u[1, (i - 1) * BASE_RES + j] for i in 1:BASE_RES, j in 1:BASE_RES]   # [iα, iβ]
g0 = zeros(BASE_RES, BASE_RES)
for i in 1:BASE_RES, j in 1:BASE_RES
    di = i == 1 ? (I0[2, j] - I0[1, j]) :
         i == BASE_RES ? (I0[BASE_RES, j] - I0[BASE_RES-1, j]) :
         (I0[i+1, j] - I0[i-1, j]) / 2
    dj = j == 1 ? (I0[i, 2] - I0[i, 1]) :
         j == BASE_RES ? (I0[i, BASE_RES] - I0[i, BASE_RES-1]) :
         (I0[i, j+1] - I0[i, j-1]) / 2
    g0[i, j] = hypot(di, dj)
end
const CUT = 1.0e-3 * maximum(g0)
const QS = [0.5, 0.9, 0.95, 0.99, 0.999]
gq = [quantile(vec(g0), q) for q in QS]
@printf("probe level-0 %d^2 unpol: %.2f s (%.0f rays/s, JIT-incl) | classes [cap1 cap2 esc nmax nonfinite]=%s | mean samples %.1f\n",
        BASE_RES, PROBE_S, BASE_RES^2 / PROBE_S, string(classhist(cls0)), mean(ns0))
@printf("  max|grad I|=%.4e -> abs_grad_cut=%.6e | quantiles %s\n", maximum(g0), CUT,
        join([@sprintf("%.3e", x) for x in gq], " "))
write_npy(joinpath(OUT, "probe$(BASE_RES)_I.npy"), permutedims(I0))    # [iβ, iα]
write_npy(joinpath(OUT, "probe$(BASE_RES)_grad.npy"), permutedims(g0))
write_npy(joinpath(OUT, "probe$(BASE_RES)_class.npy"),
          Float64.(permutedims(reshape(cls0, BASE_RES, BASE_RES))))

# ---------------------------------------------------------------- instrumented closure
const LEVEL_RAYS = Int[]
const LEVEL_WALL = Float64[]
const LEVEL_COORDS = Vector{Vector{Tuple{Float64,Float64}}}()
const LEVEL_CLASS = Vector{Vector{Int}}()
const LEVEL_NSAMP = Float64[]
const LEVEL_NONFINITE = Int[]
const LEVEL_S = Matrix{Float64}[]
function stokes_closure(coords)
    t = time()
    S, cls, ns = cpu_stokes(coords; polarized = true)
    push!(LEVEL_WALL, time() - t)
    push!(LEVEL_RAYS, length(coords))
    push!(LEVEL_COORDS, copy(coords))
    push!(LEVEL_CLASS, classhist(cls))
    push!(LEVEL_NSAMP, mean(ns))
    push!(LEVEL_NONFINITE, count(k -> !all(isfinite, @view S[:, k]), axes(S, 2)))
    push!(LEVEL_S, S)
    @printf("  level %d: %d rays in %.2f s (%.0f rays/s) | classes %s | mean samples %.1f | non-finite %d\n",
            length(LEVEL_RAYS) - 1, length(coords), LEVEL_WALL[end],
            length(coords) / LEVEL_WALL[end], string(LEVEL_CLASS[end]), LEVEL_NSAMP[end],
            LEVEL_NONFINITE[end])
    flush(stdout)
    return S
end
function reset_levels!()
    foreach(empty!, (LEVEL_RAYS, LEVEL_WALL, LEVEL_COORDS, LEVEL_CLASS, LEVEL_NSAMP,
                     LEVEL_NONFINITE, LEVEL_S))
end
run_amr() = render_amr(CFG, BIN; block = BLOCK, max_level = MAXLEVEL, polarized = true,
                       stokes = stokes_closure, abs_grad_cut = CUT, abs_grad_count = 5)

println("AMR render (polarized)")
reset_levels!()
t0 = time()
R = run_amr()
const AMR_S = time() - t0
lvl_rays = copy(LEVEL_RAYS); lvl_wall = copy(LEVEL_WALL); lvl_class = copy(LEVEL_CLASS)
lvl_nsamp = copy(LEVEL_NSAMP); lvl_nonfinite = copy(LEVEL_NONFINITE)
lvl_coords = copy(LEVEL_COORDS)
S_level0 = LEVEL_S[1]
@printf("AMR %d-eff: %.2f s | %d rays (%.0f rays/s) | per-level %s\n", EFF, AMR_S, R.nrays,
        R.nrays / AMR_S, string(lvl_rays))

# per-level ray-count consistency: level l+1 renders 4 children per refined level-l block
nb_refined = [0; [lvl_rays[l+1] ÷ (4 * BLOCK^2) for l in 1:length(lvl_rays)-1]]
counts_ok = all(lvl_rays[l] % BLOCK^2 == 0 for l in eachindex(lvl_rays)) &&
            lvl_rays[1] == BASE_RES^2
@printf("per-level blocks rendered %s | refined parents per level %s | counts consistent=%s\n",
        string(lvl_rays .÷ BLOCK^2), string(nb_refined[2:end]), counts_ok)

# ---------------------------------------------------------------- root row + determinism
# Top row: the AMR's own level-0 batch repainted onto the base grid.
# render_amr builds its level-0 worklist as [(bi, bj) for bi, bj][:] — bi fastest.
function level0_image_amr(S)
    img = zeros(4, BASE_RES, BASE_RES)
    nb = BASE_RES ÷ BLOCK
    w = 0
    for bj in 1:nb, bi in 1:nb
        for i in 1:BLOCK, j in 1:BLOCK
            img[:, (bj - 1) * BLOCK + j, (bi - 1) * BLOCK + i] .= S[:, w * BLOCK^2 + (i - 1) * BLOCK + j]
        end
        w += 1
    end
    return img
end
root = level0_image_amr(S_level0)
# consistency: the level-0 paint of the mosaic, sampled at each base pixel's
# unrefined blocks, equals the root image there
cellsz = 2^MAXLEVEL
unref = [R.levels[(jb - 1) ÷ BLOCK + 1, (ib - 1) ÷ BLOCK + 1] == 0
         for jb in 1:BASE_RES, ib in 1:BASE_RES]
root_paint_ok = all(root[1, jb, ib] == R.I[(jb - 1) * cellsz + 1, (ib - 1) * cellsz + 1]
                    for jb in 1:BASE_RES, ib in 1:BASE_RES if unref[jb, ib])

println("repeat of the level-0 polarized batch (determinism + uniform-frame rate)")
t0 = time()
S_rep, cls_rep, _ = cpu_stokes(lvl_coords[1]; polarized = true)
const REP0_S = time() - t0
det_level0 = all(S_rep .=== S_level0)
const UNI_RATE = BASE_RES^2 / REP0_S
@printf("  %.2f s (%.0f rays/s) | bitwise identical to the AMR level-0 batch: %s\n",
        REP0_S, UNI_RATE, det_level0)

det_full = nothing
rep_s = nothing
if REPEAT
    println("full AMR repeat")
    reset_levels!()
    t0 = time()
    R2 = run_amr()
    rep_s = time() - t0
    det_full = all(R.I .=== R2.I) && all(R.Q .=== R2.Q) && all(R.U .=== R2.U) &&
               all(R.V .=== R2.V) && R.levels == R2.levels && LEVEL_RAYS == lvl_rays
    @printf("  %.2f s | bitwise identical: %s\n", rep_s, det_full)
    R2 = nothing
end
empty!(LEVEL_S)

# ---------------------------------------------------------------- per-pixel level map
nt = EFF ÷ BLOCK
levels_fine = zeros(Int, nt, nt)
ts = FOV / nt
for (lidx, lc) in enumerate(lvl_coords)
    l = lidx - 1
    ps = FOV / (BASE_RES * 2^l)
    ntb = 2^(MAXLEVEL - l)
    for k in 1:(length(lc) ÷ BLOCK^2)
        blk = @view lc[(k-1)*BLOCK^2+1:k*BLOCK^2]
        ta = round(Int, (minimum(first, blk) - ps / 2 + FOV / 2) / ts)
        tb = round(Int, (minimum(last, blk) - ps / 2 + FOV / 2) / ts)
        levels_fine[tb+1:tb+ntb, ta+1:ta+ntb] .= l
    end
end
fine_hist = [count(==(l), levels_fine) for l in 0:MAXLEVEL]
tpb = nt ÷ (BASE_RES ÷ BLOCK)
nbb = BASE_RES ÷ BLOCK
fine_consistent = all(maximum(levels_fine[(bj-1)*tpb+1:bj*tpb, (bi-1)*tpb+1:bi*tpb]) ==
                      R.levels[bj, bi] for bj in 1:nbb, bi in 1:nbb)
@printf("per-pixel level map (%dx%d tiles): hist %s | consistent with base map=%s | root paint consistent=%s\n",
        nt, nt, string(fine_hist), fine_consistent, root_paint_ok)

# ---------------------------------------------------------------- diagnostics
finite = all(all(isfinite, getfield(R, k)) for k in (:I, :Q, :U, :V)) && all(isfinite, root)
fluxI = sum(R.I); fa = fluxI * (FOV / EFF)^2
lp = sum(sqrt.(R.Q .^ 2 .+ R.U .^ 2)) / fluxI
vfrac = sum(abs.(R.V)) / fluxI
vnet = sum(R.V) / fluxI
root_flux = sum(root[1, :, :]); root_fa = root_flux * (FOV / BASE_RES)^2
root_lp = sum(sqrt.(root[2, :, :] .^ 2 .+ root[3, :, :] .^ 2)) / root_flux
probe_fa = sum(I0) * (FOV / BASE_RES)^2
hist = [count(==(l), R.levels) for l in 0:MAXLEVEL]
effb = EFF ÷ nbb
fluxblk = [sum(R.I[(bj-1)*effb+1:bj*effb, (bi-1)*effb+1:bi*effb]) for bj in 1:nbb, bi in 1:nbb]
ref = R.levels .> 0
blk_share = count(ref) / length(ref)
flux_share = sum(fluxblk[ref]) / sum(fluxblk)
@printf("flux_I=%.6e area-flux %.6e (root %.6e, unpol probe %.6e) | LP %.4f (root %.4f) | |V|/I %.4e net V/I %.4e | finite=%s\n",
        fluxI, fa, root_fa, probe_fa, lp, root_lp, vfrac, vnet, finite)
@printf("levels hist %s | refined %d/%d blocks (%.1f%%) carrying %.2f%% of flux\n",
        string(hist), count(ref), length(ref), 100 * blk_share, 100 * flux_share)
uni_rays = EFF^2
uni_est_s = uni_rays / UNI_RATE
@printf("uniform %d^2 polarized: NOT RUN — estimate %.0f s from the measured level-0 frame rate %.0f rays/s\n",
        EFF, uni_est_s, UNI_RATE)

# ---------------------------------------------------------------- artifacts
pfx = "amr$(EFF)"
write_npy(joinpath(OUT, "$(pfx)_I.npy"), R.I)
write_npy(joinpath(OUT, "$(pfx)_Q.npy"), R.Q)
write_npy(joinpath(OUT, "$(pfx)_U.npy"), R.U)
write_npy(joinpath(OUT, "$(pfx)_V.npy"), R.V)
write_npy(joinpath(OUT, "$(pfx)_levels.npy"), Float64.(R.levels))
write_npy(joinpath(OUT, "$(pfx)_levels_fine.npy"), Float64.(levels_fine))
for (k, c) in enumerate(("I", "Q", "U", "V"))
    write_npy(joinpath(OUT, "root$(BASE_RES)_$(c).npy"), root[k, :, :])
end

rec = Dict(
    "what" => "BBH GRMHD superposed-metric full-IQUV image, i=90, value-based image-plane AMR, CPU",
    "when" => string(Dates.now()),
    "git_sha" => GIT_SHA,
    "julia" => string(VERSION),
    "threads" => Threads.nthreads(),
    "host_cpus" => Sys.CPU_THREADS,
    "backend" => "CPU: render_amr stokes closure -> camera_rays_at -> trace_rays_sampled(::SuperposedKerrSchild) in chunks of $(CHUNK) -> Radiation.polarized_march (threaded)",
    "dataset" => BIN,
    "metric" => "SuperposedKerrSchild m1=m2=0.5, a1=a2=0, centers ±12.5(cos φ0, sin φ0, 0), φ0=$(PHI0)",
    "centers" => [collect(C1), collect(C2)],
    "inclination_deg" => INC, "fov_M" => FOV, "distance_M" => DIST, "C" => CVAL,
    "max_steps" => MAXSTEPS, "frequency_hz" => NU, "gamma" => GAMMA,
    "m_bh_cgs" => M_BH, "mass_scale_cgs" => MASS_SCALE,
    "amr" => Dict("base_res" => BASE_RES, "block" => BLOCK, "max_level" => MAXLEVEL,
                  "eff_res" => EFF, "finest_px_M" => FOV / EFF),
    "indicator" => Dict("abs_grad_cut" => CUT, "abs_grad_count" => 5,
                        "probe_grad_max" => maximum(g0), "probe_Imax" => maximum(I0),
                        "probe_grad_quantiles" => Dict(string(q) => v for (q, v) in zip(QS, gq))),
    "snapshot" => Dict("load_s" => LOAD_S, "nmb" => SNAP.nmb,
                       "leaf_table" => leaf_table(SNAP) !== nothing),
    "probe" => Dict("wall_s_jit_incl" => PROBE_S, "rays" => BASE_RES^2,
                    "classes_cap1_cap2_esc_nmax_nonfinite" => classhist(cls0),
                    "mean_samples" => mean(ns0), "area_flux_I" => probe_fa),
    "amr_leg" => Dict("wall_s" => AMR_S, "nrays" => R.nrays, "rays_per_s" => R.nrays / AMR_S,
                      "per_level_rays" => lvl_rays, "per_level_wall_s" => lvl_wall,
                      "per_level_classes_cap1_cap2_esc_nmax_nonfinite" => lvl_class,
                      "per_level_mean_samples" => lvl_nsamp,
                      "per_level_nonfinite_rays" => lvl_nonfinite,
                      "ray_counts_consistent" => counts_ok,
                      "flux_I" => fluxI, "flux_Q" => sum(R.Q), "flux_U" => sum(R.U),
                      "flux_V" => sum(R.V), "area_flux_I" => fa,
                      "max_I" => maximum(R.I), "LP" => lp, "absV_over_I" => vfrac,
                      "netV_over_I" => vnet,
                      "levels_hist" => hist, "refined_block_share" => blk_share,
                      "refined_flux_share" => flux_share),
    "levels_fine" => Dict("tile_grid" => nt, "tile_px" => BLOCK, "tile_hist" => fine_hist,
                          "consistent_with_base_map" => fine_consistent),
    "root_leg" => Dict("source" => "level-0 batch of the AMR render", "res" => BASE_RES,
                       "flux_I" => root_flux, "area_flux_I" => root_fa, "LP" => root_lp,
                       "paint_consistent" => root_paint_ok),
    "level0_repeat" => Dict("wall_s" => REP0_S, "rays_per_s" => UNI_RATE,
                            "bitwise_identical" => det_level0),
    "full_repeat" => Dict("wall_s" => rep_s, "bitwise_identical" => det_full),
    "uniform_est" => Dict("res" => EFF, "nrays" => uni_rays, "est_wall_s" => uni_est_s,
                          "rate_source" => "measured level-0 repeat (a uniform base² frame)",
                          "rate_rays_per_s" => UNI_RATE),
    "finite" => finite,
)
open(joinpath(OUT, "records.json"), "w") do io
    println(io, _json(rec))
end
println("wrote ", OUT)
