# fig_bbh_i90_pol_amr.jl — binary-black-hole GRMHD image, full IQUV, at i = 90° by
# value-based image-plane adaptive refinement on CPU threads.
#
# Deck: AthenaK torus.mhd_w_bcc.00455.bin (gamma = 13/9) on a static
# SuperposedKerrSchild metric, m1 = m2 = 0.5, a1 = a2 = 0, centres
# ±12.5 (cos φ0, sin φ0, 0) with φ0 = -0.2233022; 230 GHz; i = 90°; fov = 40 M;
# distance 1000 M; C = 80; max_steps = 2500; m_bh = 6.5e9 Msun; mass_scale = 1e24 g
# (uncalibrated placeholder); PlasmaConfig(gamma = 13/9) defaults otherwise (R_low = 1,
# R_high = 10, sigma_cut = 1).
#
# AMR: render_amr(plan; block, max_level, refine, backend = :cpu) on the polarized plan
# with refine = AbsoluteGradient(cut; count = 5), cut = 1e-3 · max|∇I| of the level-0
# (uniform base²) Stokes I probe of this deck (White 2022). |∇I| on the probe is the
# central difference along each image axis (one-sided at the frame edges), per pixel.
#
# Legs: snapshot load; level-0 Stokes I probe (render of the base² plan) and its ray
# classes (shadow); the AMR render; the uniform base² polarized render (the level-0
# blocks of the AMR, bitwise: the figure's root row, a determinism check against the
# level-0 values the indicator saw, and the uniform-frame rate); optionally a full AMR
# repeat (BLP_REPEAT=1). The visited-block tree is rebuilt from the mosaic and from the
# block intensities the indicator was called with (a block whose mosaic samples equal
# a matrix the indicator declined is a leaf; level max_level is a leaf); it gives the
# per-level ray counts and the finest level per finest 8×8 tile (`levels_fine`), and
# must reproduce render_amr's level map and ray count. Ray classes and march lengths of
# every level come from the plasma-free shadow march over the visited blocks' pixels.
#
# Writes to BLP_OUT (figures/paper/out/bbh_i90_pol_amr):
#   amr4096_{I,Q,U,V}.npy      mosaic at base·2^max_level per side, [iβ, iα], cgs
#   amr4096_levels.npy         finest level per base block, [β block, α block]
#   amr4096_levels_fine.npy    finest level per 8×8 tile of the mosaic, [β tile, α tile]
#   root256_{I,Q,U,V}.npy      the level-0 frame, [iβ, iα]
#   probe256_{I,grad,class}.npy  the Stokes I probe, its |∇I|, its ray classes, [iβ, iα]
#   records.json               setup, indicator, per-level counts, fluxes, LP, V/I,
#                              level histograms, wall times, BlackLightPlus commit, threads
#
# Env: BLP_BBH_BIN, BLP_BASE_RES (256), BLP_BLOCK (8), BLP_MAXLEVEL (4), BLP_REPEAT (0),
# BLP_OUT. Threads come from `julia -t N`.
# Run, from the repository root:
#   julia -t 48 --project=. figures/paper/fig_bbh_i90_pol_amr.jl

using StaticArrays, Printf, Dates, Statistics
using BlackLightPlus
using BlackLightPlus.Data: athenak_bin_snapshot, LeafTable
using BlackLightPlus.Spacetime: ks_radius, horizon_radius, march_control, MarchBudget
using BlackLightPlus.Analysis: AbsoluteGradient, needs_refinement
using BlackLightPlus.Execution: resolve, foreach_item!, CameraRays, BlockGrid
using BlackLightPlus.Products: ShadowProgram

const ROOT = dirname(dirname(@__DIR__))
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
const REPEAT = get(ENV, "BLP_REPEAT", "0") == "1"
const EFF = BASE_RES * 2^MAXLEVEL
const OUT = get(ENV, "BLP_OUT", joinpath(ROOT, "figures", "paper", "out", "bbh_i90_pol_amr"))
const BACKEND = :cpu

const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
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

const CFG = RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MASS_SCALE, frequency_hz = NU,
                      camera = CameraConfig(inclination_deg = INC, distance = DIST,
                                            fov = FOV, resolution = BASE_RES),
                      plasma = PlasmaConfig(gamma = GAMMA), max_steps = MAXSTEPS, C = CVAL,
                      escape_factor = 1.1)

println("BlackLightPlus ", GIT_SHA, " | threads=", Threads.nthreads(),
        " | base ", BASE_RES, " block ", BLOCK, " max_level ", MAXLEVEL, " -> ", EFF, "^2 eff")
mkpath(OUT)
@printf("loading %s\n", BIN)
t0 = time()
const SNAP = athenak_bin_snapshot(BIN; gamma = GAMMA)
const LOAD_S = time() - t0
const LEAF = SNAP.blocks.locate isa LeafTable
@printf("snapshot loaded in %.2f s (nmb=%d, %s)\n", LOAD_S, SNAP.blocks.nmb,
        LEAF ? "leaf table: O(1) locate" : "no leaf table: $(nameof(typeof(SNAP.blocks.locate))) locate")

const PLAN_I = prepare(CFG; spacetime = ST, matter = SNAP, mode = StokesI())
const PLAN_Q = prepare(CFG; spacetime = ST, matter = SNAP, mode = IQUV())

# ---------------------------------------------------------------- ray end classes
# 1 = captured by hole 1, 2 = captured by hole 2 (nearest horizon), 3 = escaped,
# 4 = step budget exhausted, 5 = non-finite: the final state of the plasma-free march
# (a step-limited ray whose final state meets the capture rule counts as captured).
const CAP_ONLY = MarchBudget{Float64}(CVAL, Inf, Inf, MAXSTEPS)
function end_class(y, fate)
    all(isfinite, y) || return 5
    if fate == 1 || (fate == 0 && march_control(ST, y, Inf, CAP_ONLY).stop)
        x = SVector(y[1], y[2], y[3], y[4])
        d1 = ks_radius(ST, x, 1) - horizon_radius(ST, 1)
        d2 = ks_radius(ST, x, 2) - horizon_radius(ST, 2)
        return d1 <= d2 ? 1 : 2
    end
    sqrt(y[2]^2 + y[3]^2 + y[4]^2) > PLAN_I.escape && return 3
    return 4
end
classhist(cls) = [count(==(c), cls) for c in 1:5]

"Classes and stored-state counts (steps + 1) of the rays of `gen` (a BlockGrid with blocks)."
function block_classes(gen, n)
    prog = ShadowProgram(PLAN_I.geodesic_spacetime, CAP_ONLY, PLAN_I.escape,
                         CameraRays(PLAN_I.spacetime, PLAN_I.frame, gen))
    ys = Vector{SVector{8,Float64}}(undef, n)
    ms = Vector{SVector{4,Float64}}(undef, n)
    foreach_item!(resolve(BACKEND), prog, 1:n; outs = (ys, ms))
    return [end_class(ys[k], Int(ms[k][1])) for k in 1:n], [Int(ms[k][2]) + 1 for k in 1:n]
end

# ---------------------------------------------------------------- level-0 probe
t0 = time()
probe = render(PLAN_I; backend = BACKEND)
const PROBE_S = time() - t0
I0 = permutedims(probe.I)                                   # [iα, iβ]
sh0 = shadow(PLAN_I; backend = BACKEND)
cls0 = [end_class(sh0.final[k], sh0.fate[k]) for k in CartesianIndices(sh0.fate)]   # [iβ, iα]
ns0 = vec(sh0.steps) .+ 1
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
@printf("probe level-0 %d^2 Stokes I: %.2f s (%.0f rays/s, JIT-incl) | classes [cap1 cap2 esc nmax nonfinite]=%s | mean samples %.1f\n",
        BASE_RES, PROBE_S, BASE_RES^2 / PROBE_S, string(classhist(cls0)), mean(ns0))
@printf("  max|grad I|=%.4e -> abs_grad_cut=%.6e | quantiles %s\n", maximum(g0), CUT,
        join([@sprintf("%.3e", x) for x in gq], " "))
write_npy(joinpath(OUT, "probe$(BASE_RES)_I.npy"), permutedims(I0))    # [iβ, iα]
write_npy(joinpath(OUT, "probe$(BASE_RES)_grad.npy"), permutedims(g0))
write_npy(joinpath(OUT, "probe$(BASE_RES)_class.npy"), Float64.(cls0))

# ---------------------------------------------------------------- recording indicator
"Bit pattern of a block intensity matrix (the key of the indicator's record)."
bitkey(M) = collect(reinterpret(UInt64, vec(collect(Float64, M))))

"""
The absolute-gradient indicator, recording the bit pattern of every block intensity
matrix it is called with, the patterns of the blocks it declined, and the number of
blocks it refined. Blocks are tested concurrently; the record is locked.
"""
struct RecordingIndicator
    ind::AbsoluteGradient
    lock::ReentrantLock
    declined::Set{Vector{UInt64}}
    seen::Vector{Vector{UInt64}}
    nrefined::Base.RefValue{Int}
end
RecordingIndicator(ind) = RecordingIndicator(ind, ReentrantLock(), Set{Vector{UInt64}}(),
                                             Vector{UInt64}[], Ref(0))
function (r::RecordingIndicator)(I::AbstractMatrix)
    need = needs_refinement(r.ind, I)
    key = bitkey(I)
    lock(r.lock) do
        push!(r.seen, key)
        need ? (r.nrefined[] += 1) : push!(r.declined, key)
    end
    return need
end
function reset!(r::RecordingIndicator)
    empty!(r.declined); empty!(r.seen); r.nrefined[] = 0
    return r
end

const IND = AbsoluteGradient(CUT; count = 5)
const REC = RecordingIndicator(IND)
run_amr() = render_amr(PLAN_Q; block = BLOCK, max_level = MAXLEVEL, refine = REC,
                       backend = BACKEND)

println("AMR render (polarized)")
t0 = time()
R = run_amr()
const AMR_S = time() - t0
@printf("AMR %d-eff: %.2f s | %d rays (%.0f rays/s) | indicator calls %d (%d refined)\n",
        EFF, AMR_S, R.nrays, R.nrays / AMR_S, length(REC.seen), REC.nrefined[])

# ---------------------------------------------------------------- visited-block tree
# Blocks are rebuilt with render_amr's geometry: base block (bi, bj) at
# (-fov/2 + (bi - 1) block px0, -fov/2 + (bj - 1) block px0) with span block px0, a child
# at the parent's corner plus (0 or span/2) per axis with span/2.
struct Blk
    ia::Int        # α index of the block in its level grid (1-based)
    ib::Int        # β index
    level::Int
    a0::Float64
    b0::Float64
    span::Float64
end

"Block pixel intensities read off the mosaic: pixel (i, j) at its cell's first canvas cell."
function mosaic_block(Imos, b::Blk)
    cell = 2^(MAXLEVEL - b.level)
    ca = (b.ia - 1) * BLOCK * cell
    cb = (b.ib - 1) * BLOCK * cell
    return [Imos[cb + (j - 1) * cell + 1, ca + (i - 1) * cell + 1] for i in 1:BLOCK, j in 1:BLOCK]
end

function rebuild_tree(Imos, rec::RecordingIndicator)
    nb = BASE_RES ÷ BLOCK
    px0 = FOV / BASE_RES
    level_blocks = [Blk[] for _ in 0:MAXLEVEL]
    leaves = Blk[]
    work = [Blk(bi, bj, 0, -FOV / 2 + (bi - 1) * BLOCK * px0, -FOV / 2 + (bj - 1) * BLOCK * px0,
                BLOCK * px0) for bi in 1:nb, bj in 1:nb][:]
    while !isempty(work)
        next = Blk[]
        for b in work
            push!(level_blocks[b.level + 1], b)
            leaf = b.level == MAXLEVEL
            if !leaf
                M = mosaic_block(Imos, b)
                leaf = !needs_refinement(rec.ind, M) && bitkey(M) in rec.declined
            end
            if leaf
                push!(leaves, b)
            else
                h = b.span / 2
                for (da, db, oa, ob) in ((0.0, 0.0, 1, 1), (h, 0.0, 2, 1), (0.0, h, 1, 2), (h, h, 2, 2))
                    push!(next, Blk(2 * b.ia - 2 + oa, 2 * b.ib - 2 + ob, b.level + 1,
                                    b.a0 + da, b.b0 + db, h))
                end
            end
        end
        work = next
    end
    return level_blocks, leaves
end

level_blocks, leaves = rebuild_tree(R.image.I, REC)
lvl_rays = [length(bl) * BLOCK^2 for bl in level_blocks]
nb = BASE_RES ÷ BLOCK
levels_rebuilt = zeros(Int, nb, nb)
for b in leaves
    bi = (b.ia - 1) ÷ 2^b.level + 1
    bj = (b.ib - 1) ÷ 2^b.level + 1
    levels_rebuilt[bj, bi] = max(levels_rebuilt[bj, bi], b.level)
end
ntested = sum(length(level_blocks[l + 1]) for l in 0:MAXLEVEL-1)
nleaf_tested = count(b -> b.level < MAXLEVEL, leaves)
tree_ok = sum(lvl_rays) == R.nrays && levels_rebuilt == R.levels &&
          ntested == length(REC.seen) && nleaf_tested == length(REC.seen) - REC.nrefined[]
nb_refined = [0; [lvl_rays[l+1] ÷ (4 * BLOCK^2) for l in 1:length(lvl_rays)-1]]
counts_ok = all(lvl_rays[l] % BLOCK^2 == 0 for l in eachindex(lvl_rays)) &&
            lvl_rays[1] == BASE_RES^2
@printf("per-level rays %s | blocks %s | refined parents per level %s | tree consistent with render_amr (rays, level map, indicator calls)=%s\n",
        string(lvl_rays), string(lvl_rays .÷ BLOCK^2), string(nb_refined[2:end]), tree_ok)
tree_ok || @warn "visited-block tree does not reproduce render_amr's level map and ray count"

# per-level ray classes and march lengths: the shadow march over the visited blocks
lvl_class = Vector{Vector{Int}}(); lvl_nsamp = Float64[]
t0 = time()
for l in 0:MAXLEVEL
    bl = level_blocks[l + 1]
    gen = BlockGrid([SVector(b.a0, b.b0, b.span / BLOCK) for b in bl], BLOCK)
    cls, ns = block_classes(gen, length(bl) * BLOCK^2)
    push!(lvl_class, classhist(cls)); push!(lvl_nsamp, mean(ns))
    @printf("  level %d: %d rays | classes %s | mean samples %.1f\n", l, length(cls),
            string(lvl_class[end]), lvl_nsamp[end])
end
const CLASS_S = time() - t0

# ---------------------------------------------------------------- root row + determinism
# The uniform base² polarized frame: its pixels are the level-0 block pixels of the AMR
# (same image coordinates, same program), so it is the figure's root row.
println("uniform level-0 polarized frame (root row, determinism, uniform-frame rate)")
t0 = time()
root_img = render(PLAN_Q; backend = BACKEND)
const REP0_S = time() - t0
const UNI_RATE = BASE_RES^2 / REP0_S
root = zeros(4, BASE_RES, BASE_RES)
for (c, a) in enumerate((root_img.I, root_img.Q, root_img.U, root_img.V))
    root[c, :, :] .= a
end
# the level-0 intensities the indicator saw, block by block, against the uniform frame
seen0 = Set(REC.seen)
det_level0 = all(level_blocks[1]) do b
    M = [root_img.I[(b.ib - 1) * BLOCK + j, (b.ia - 1) * BLOCK + i] for i in 1:BLOCK, j in 1:BLOCK]
    bitkey(M) in seen0
end
@printf("  %.2f s (%.0f rays/s) | every level-0 block bitwise among the indicator's inputs: %s\n",
        REP0_S, UNI_RATE, det_level0)
# the mosaic at each unrefined base block's cells equals the root image there
cellsz = 2^MAXLEVEL
unref = [R.levels[(jb - 1) ÷ BLOCK + 1, (ib - 1) ÷ BLOCK + 1] == 0
         for jb in 1:BASE_RES, ib in 1:BASE_RES]
root_paint_ok = all(root[c, jb, ib] == getfield(R.image, k)[(jb - 1) * cellsz + 1, (ib - 1) * cellsz + 1]
                    for (c, k) in enumerate((:I, :Q, :U, :V))
                    for jb in 1:BASE_RES, ib in 1:BASE_RES if unref[jb, ib])

det_full = nothing
rep_s = nothing
if REPEAT
    println("full AMR repeat")
    reset!(REC)
    t0 = time()
    R2 = run_amr()
    rep_s = time() - t0
    det_full = all(all(getfield(R.image, k) .=== getfield(R2.image, k)) for k in (:I, :Q, :U, :V)) &&
               R.levels == R2.levels && R.nrays == R2.nrays
    @printf("  %.2f s | bitwise identical: %s\n", rep_s, det_full)
    R2 = nothing
end

# ---------------------------------------------------------------- per-tile level map
nt = EFF ÷ BLOCK
levels_fine = zeros(Int, nt, nt)
for b in leaves
    ntb = 2^(MAXLEVEL - b.level)
    levels_fine[(b.ib - 1) * ntb + 1:b.ib * ntb, (b.ia - 1) * ntb + 1:b.ia * ntb] .= b.level
end
fine_hist = [count(==(l), levels_fine) for l in 0:MAXLEVEL]
tpb = nt ÷ (BASE_RES ÷ BLOCK)
nbb = BASE_RES ÷ BLOCK
fine_consistent = all(maximum(levels_fine[(bj-1)*tpb+1:bj*tpb, (bi-1)*tpb+1:bi*tpb]) ==
                      R.levels[bj, bi] for bj in 1:nbb, bi in 1:nbb)
@printf("per-pixel level map (%dx%d tiles): hist %s | consistent with base map=%s | root paint consistent=%s\n",
        nt, nt, string(fine_hist), fine_consistent, root_paint_ok)

# ---------------------------------------------------------------- diagnostics
RI, RQ, RU, RV = R.image.I, R.image.Q, R.image.U, R.image.V
finite = all(all(isfinite, a) for a in (RI, RQ, RU, RV)) && all(isfinite, root)
fluxI = sum(RI); fa = fluxI * (FOV / EFF)^2
lp = sum(sqrt.(RQ .^ 2 .+ RU .^ 2)) / fluxI
vfrac = sum(abs.(RV)) / fluxI
vnet = sum(RV) / fluxI
root_flux = sum(root[1, :, :]); root_fa = root_flux * (FOV / BASE_RES)^2
root_lp = sum(sqrt.(root[2, :, :] .^ 2 .+ root[3, :, :] .^ 2)) / root_flux
probe_fa = sum(I0) * (FOV / BASE_RES)^2
hist = [count(==(l), R.levels) for l in 0:MAXLEVEL]
effb = EFF ÷ nbb
fluxblk = [sum(RI[(bj-1)*effb+1:bj*effb, (bi-1)*effb+1:bi*effb]) for bj in 1:nbb, bi in 1:nbb]
ref = R.levels .> 0
blk_share = count(ref) / length(ref)
flux_share = sum(fluxblk[ref]) / sum(fluxblk)
@printf("flux_I=%.6e area-flux %.6e (root %.6e, Stokes I probe %.6e) | LP %.4f (root %.4f) | |V|/I %.4e net V/I %.4e | finite=%s\n",
        fluxI, fa, root_fa, probe_fa, lp, root_lp, vfrac, vnet, finite)
@printf("levels hist %s | refined %d/%d blocks (%.1f%%) carrying %.2f%% of flux\n",
        string(hist), count(ref), length(ref), 100 * blk_share, 100 * flux_share)
uni_rays = EFF^2
uni_est_s = uni_rays / UNI_RATE
@printf("uniform %d^2 polarized: NOT RUN — estimate %.0f s from the measured level-0 frame rate %.0f rays/s\n",
        EFF, uni_est_s, UNI_RATE)

# ---------------------------------------------------------------- artifacts
pfx = "amr$(EFF)"
write_npy(joinpath(OUT, "$(pfx)_I.npy"), RI)
write_npy(joinpath(OUT, "$(pfx)_Q.npy"), RQ)
write_npy(joinpath(OUT, "$(pfx)_U.npy"), RU)
write_npy(joinpath(OUT, "$(pfx)_V.npy"), RV)
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
    "blp_version" => string(R.image.run.version),
    "threads" => Threads.nthreads(),
    "host_cpus" => Sys.CPU_THREADS,
    "backend" => "cpu: render_amr(plan; block, max_level, refine, backend = :cpu), fused polarized march over image blocks streamed on Julia threads; probe render(plan) in Stokes I; ray classes from the shadow march",
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
    "snapshot" => Dict("load_s" => LOAD_S, "nmb" => SNAP.blocks.nmb, "leaf_table" => LEAF),
    "probe" => Dict("wall_s_jit_incl" => PROBE_S, "rays" => BASE_RES^2,
                    "classes_cap1_cap2_esc_nmax_nonfinite" => classhist(cls0),
                    "mean_samples" => mean(ns0), "area_flux_I" => probe_fa),
    "amr_leg" => Dict("wall_s" => AMR_S, "nrays" => R.nrays, "rays_per_s" => R.nrays / AMR_S,
                      "per_level_rays" => lvl_rays,
                      "per_level_classes_cap1_cap2_esc_nmax_nonfinite" => lvl_class,
                      "per_level_mean_samples" => lvl_nsamp, "classes_wall_s" => CLASS_S,
                      "ray_counts_consistent" => counts_ok, "tree_consistent" => tree_ok,
                      "flux_I" => fluxI, "flux_Q" => sum(RQ), "flux_U" => sum(RU),
                      "flux_V" => sum(RV), "area_flux_I" => fa,
                      "max_I" => maximum(RI), "LP" => lp, "absV_over_I" => vfrac,
                      "netV_over_I" => vnet,
                      "levels_hist" => hist, "refined_block_share" => blk_share,
                      "refined_flux_share" => flux_share),
    "levels_fine" => Dict("tile_grid" => nt, "tile_px" => BLOCK, "tile_hist" => fine_hist,
                          "consistent_with_base_map" => fine_consistent),
    "root_leg" => Dict("source" => "uniform base² render(plan) of the AMR plan (its level-0 blocks)",
                       "res" => BASE_RES,
                       "flux_I" => root_flux, "area_flux_I" => root_fa, "LP" => root_lp,
                       "paint_consistent" => root_paint_ok),
    "level0_repeat" => Dict("wall_s" => REP0_S, "rays_per_s" => UNI_RATE,
                            "bitwise_identical" => det_level0),
    "full_repeat" => Dict("wall_s" => rep_s, "bitwise_identical" => det_full),
    "uniform_est" => Dict("res" => EFF, "nrays" => uni_rays, "est_wall_s" => uni_est_s,
                          "rate_source" => "measured uniform base² polarized frame",
                          "rate_rays_per_s" => UNI_RATE),
    "finite" => finite,
)
open(joinpath(OUT, "records.json"), "w") do io
    println(io, _json(rec))
end
println("wrote ", OUT)
