# fig_dynrays.jl — null geodesics and capture maps of an orbiting equal-mass binary in the
# boosted superposed Kerr–Schild metric (`BoostedSuperposedKerrSchild` on a
# `CircularOrbit`, metric and ∂_t g at the ray's own coordinate time), each beside the same
# rays in the frozen metric: the static superposition (`SuperposedKerrSchild`) with the
# holes held at the positions they occupy when the camera's rays cross the sky plane
# through the centre of mass (the median crossing time of the escaped rays of the moving
# run, `t_pass`). Drawn by paper/scripts/plot_dynrays.py into fig_dynrays.pdf.
#
# Binary: masses 1/2 and 1/2 (total M = 1), no spin, separation SEP, Keplerian circular
# orbit prograde about +z (ω = SEP^-3/2, period T = 2π/ω, hole speed ω SEP/2).
#
# rays — a pinhole fan of NFAN rays in the orbital plane, received at (XCAM, 0, 0) at
#        t = 0 from directions within ±HALF of −x, traced backward; the orbit phase puts
#        the holes on the x axis at t = −TALIGN (the generator figures/dynrays/fig_dyn_rays.jl
#        and its 2026-10-18 run use the same fan and phase). Adaptive Vern7, reltol 1e-10,
#        abstol 1e-12, λ ≤ LMAX, dense output sampled every DLAM in λ. Frozen plane x = 0.
# maps — ray classes on the image plane of a plane-parallel camera at inclination INC
#        from the orbital axis and distance DCAM, box |α|, |β| ≤ LIM, at NFRAME observer
#        times t_k = k (T/2)/NFRAME (the equal-mass image repeats after T/2). The orbit
#        phase puts the holes on the x axis (hole 1 at +x, toward the camera) at t = −DCAM.
#        Classes: 0 escaped with bending ≤ 360°, 1 captured by hole 1, 2 by hole 2, 3 escaped
#        after bending > 360°, 4 undetermined (affine budget); bending is the summed
#        turning angle of the saved path. Vern7, reltol 1e-9, abstol 1e-11, λ ≤ 600,
#        escape at R > 250 M. Image-plane refinement as figures/dynrays/fig_dyn_class.jl and
#        fig_z4c_amr.jl: BASE² base grid in BLOCK² blocks, a block split into four at twice
#        the pixel density down to MAXLEVEL when the classes of its pixels and of its
#        one-pixel border differ. Frozen sky plane d̂·x = 0, d̂ = (sin i, 0, cos i).
# scan — the freeze time: per frame, a uniform NSCAN² grid of the box traced in the moving
#        metric and in frozen metrics at t_pass + δ, δ = −SCANSPAN:1:SCANSPAN; the
#        fraction of pixels whose capture state (escaped, hole 1, hole 2) differs, per δ.
# Capture (all parts): r'_A < r_h,A + 2e-3 (rest-frame radius of hole A at the ray's
# time) or normal-frame energy above 0.99 of the redshift cut, by the nearer hole.
#
# Writes figures/paper/out/dynrays/ (BLP_DYN_OUT):
#   fan_<moving|frozen>.npy   rows (ray, λ, t, x, y, z)
#   map_<moving|frozen>_<k>.npy   class plane, EFF × EFF, [i_β, i_α]
#   records_rays.json, records_maps.json, records_scan.json   setup, per-ray fates and
#                  crossing times, t_pass, hole positions, per frame rays/wall/classes per
#                  variant, changed pixels, the scan, code SHA, threads.
# Run from the repository root:
#   julia -t 24 --project=. figures/paper/fig_dynrays.jl
# Env: BLP_DYN_PARTS (rays,maps,scan; scan reads records_maps.json), BLP_DYN_NSCAN (160),
#      BLP_DYN_SCANSPAN (12), BLP_DYN_SEP (8), BLP_DYN_XCAM (45), BLP_DYN_HALF (20),
#      BLP_DYN_TALIGN (69), BLP_DYN_INC (60), BLP_DYN_DCAM (100), BLP_DYN_LIM (8),
#      BLP_DYN_NFRAME (4), BLP_DYN_BASE (128), BLP_DYN_BLOCK (8), BLP_DYN_MAXLEVEL (3),
#      BLP_DYN_OUT.
using Dates, LinearAlgebra, Printf, Statistics, StaticArrays
using Adapt
using BlackLightPlus
using BlackLightPlus.Spacetime: BoostedSuperposedKerrSchild, SuperposedKerrSchild, CircularOrbit,
                                trace_geodesic, nullify_covariant, null_covariant, metric,
                                ks_radius, horizon_radius, normal_frame_energy, CAPTURE_REDSHIFT
using BlackLightPlus.Execution: ThreadedExecutor, foreach_item!, image_coords, BlockGrid,
                                ImageBlock, stream_blocks!
using BlackLightPlus.Products: Mosaic, paint_block!, mark_visited!
using BlackLightPlus.Data: write_npy

const ROOT = dirname(dirname(@__DIR__))
const OUT = get(ENV, "BLP_DYN_OUT", joinpath(@__DIR__, "out", "dynrays"))
const PARTS = split(get(ENV, "BLP_DYN_PARTS", "rays,maps,scan"), ",")
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
envf(k, d) = parse(Float64, get(ENV, k, string(d)))
envi(k, d) = parse(Int, get(ENV, k, string(d)))

const MASSES = (0.5, 0.5)
const SEP = envf("BLP_DYN_SEP", 8)
const OMEGA = sqrt(sum(MASSES) / SEP^3)
const PERIOD = 2π / OMEGA
# rays
const XCAM = envf("BLP_DYN_XCAM", 45)
const HALF = envf("BLP_DYN_HALF", 20)
const TALIGN = envf("BLP_DYN_TALIGN", 69)
const NFAN = 61
const FAN_LMAX = 400.0
const FAN_RESC = 200.0
const DLAM = 0.05
# maps
const INC = envf("BLP_DYN_INC", 60)
const DCAM = envf("BLP_DYN_DCAM", 100)
const LIM = envf("BLP_DYN_LIM", 8)
const NFRAME = envi("BLP_DYN_NFRAME", 4)
const BASE = envi("BLP_DYN_BASE", 128)
const BLOCK = envi("BLP_DYN_BLOCK", 8)
const MAXLEVEL = envi("BLP_DYN_MAXLEVEL", 3)
const EFF = BASE * 2^MAXLEVEL
const FOV = 2 * LIM
const MAP_LMAX = 600.0
const MAP_RESC = 250.0
const BEND_CUT = 360.0
const AMR_EXEC = ThreadedExecutor(chunk = BLOCK^2)
const NSCAN = envi("BLP_DYN_NSCAN", 160)
const SCANSPAN = envi("BLP_DYN_SCANSPAN", 12)
const SI = sin(deg2rad(INC)); const CI = cos(deg2rad(INC))
const DHAT = SVector(SI, 0.0, CI)

mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

moving(orbit) = BoostedSuperposedKerrSchild(collect(MASSES), [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], orbit)
"the static superposition with the holes where `orbit` has them at time t"
function frozen(orbit, t)
    tr = orbit(t)
    return SuperposedKerrSchild(collect(MASSES), [0.0, 0.0], [Tuple(tr[1][1]), Tuple(tr[2][1])])
end
holes_at(orbit, t) = [collect(orbit(t)[A][1]) for A in 1:2]
vel_at(orbit, t) = [collect(orbit(t)[A][2]) for A in 1:2]

# fate of an end state: 1/2 captured by the nearer hole, 0 escaped, 3 budget
function fate(st, u, resc)
    x = SVector(u[1], u[2], u[3], u[4])
    R = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
    dA = (ks_radius(st, x, 1) - horizon_radius(st, 1), ks_radius(st, x, 2) - horizon_radius(st, 2))
    if minimum(dA) < 2e-3 || normal_frame_energy(st, u) > 0.99CAPTURE_REDSHIFT
        return dA[1] <= dA[2] ? 1 : 2
    end
    R > 0.99resc && return 0
    return 3
end

"coordinate time at which the saved path first crosses n̂·x = 0 going away from the camera"
function plane_time(us, n̂)
    s(u) = n̂[1] * u[2] + n̂[2] * u[3] + n̂[3] * u[4]
    for k in 1:length(us)-1
        s1 = s(us[k]); s2 = s(us[k+1])
        if s1 > 0 && s2 <= 0
            w = s1 / (s1 - s2)
            return us[k][1] + w * (us[k+1][1] - us[k][1])
        end
    end
    return NaN
end

"summed turning angle (degrees) of the saved coordinate path"
function bend_deg(us)
    tot = 0.0
    prev = nothing
    for k in 1:length(us)-1
        d = SVector(us[k+1][2] - us[k][2], us[k+1][3] - us[k][3], us[k+1][4] - us[k][4])
        n = norm(d)
        n < 1e-12 && continue
        tt = d / n
        prev !== nothing && (tot += acos(clamp(dot(prev, tt), -1.0, 1.0)))
        prev = tt
    end
    return rad2deg(tot)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(x::Tuple) = json_str(collect(x))
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"

common_record() = Dict{String,Any}(
    "masses" => collect(MASSES), "separation" => SEP, "omega" => OMEGA, "period" => PERIOD,
    "hole_speed" => OMEGA * SEP / 2, "r_h" => [horizon_radius(moving(CircularOrbit(MASSES, SEP)), A) for A in 1:2],
    "code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()))
save_record(name, rec) = open(io -> println(io, json_str(rec)), joinpath(OUT, "records_$name.json"), "w")

# ---------------------------------------------------------------- rays: the fan
function run_rays()
    orbit = CircularOrbit(MASSES, SEP; phase = OMEGA * TALIGN)
    θs = range(-deg2rad(HALF), deg2rad(HALF); length = NFAN)
    x0 = SVector(0.0, XCAM, 0.0, 0.0)
    xhat = SVector(1.0, 0.0, 0.0)
    rec = merge(common_record(), Dict{String,Any}("xcam" => XCAM, "half_deg" => HALF, "nfan" => NFAN, "talign" => TALIGN,
                           "phase" => OMEGA * TALIGN, "lmax" => FAN_LMAX, "resc" => FAN_RESC,
                           "dlam" => DLAM, "reltol" => 1e-10, "abstol" => 1e-12,
                           "theta_deg" => collect(rad2deg.(θs))))
    t_pass = NaN
    for variant in ("moving", "frozen")
        st = variant == "moving" ? moving(orbit) : frozen(orbit, t_pass)
        rays = [(k = nullify_covariant(st, x0, SVector(cos(θ), -sin(θ), 0.0)); vcat(x0, k / k[1])) for θ in θs]
        paths = Vector{Matrix{Float64}}(undef, NFAN)
        fates = Vector{Int}(undef, NFAN)
        tcross = Vector{Float64}(undef, NFAN)
        bends = Vector{Float64}(undef, NFAN)
        t0 = time()
        Threads.@threads for i in 1:NFAN
            sol = trace_geodesic(st, rays[i]; λmax = FAN_LMAX, escape = FAN_RESC, reltol = 1e-10,
                                 abstol = 1e-12, save_everystep = true)
            λs = collect(0.0:DLAM:sol.t[end])
            λs[end] == sol.t[end] || push!(λs, sol.t[end])
            p = Matrix{Float64}(undef, length(λs), 6)
            for (j, λ) in enumerate(λs)
                u = sol(λ)
                p[j, :] .= (i - 1, λ, u[1], u[2], u[3], u[4])
            end
            paths[i] = p
            fates[i] = fate(st, sol.u[end], FAN_RESC)
            tcross[i] = plane_time(sol.u, xhat)
            bends[i] = bend_deg(sol.u)
        end
        wall = time() - t0
        write_npy(joinpath(OUT, "fan_$variant.npy"), reduce(vcat, paths))
        if variant == "moving"
            esc = [tcross[i] for i in 1:NFAN if fates[i] == 0 && isfinite(tcross[i])]
            t_pass = median(esc)
            rec["t_pass"] = t_pass
            rec["t_pass_spread"] = [minimum(esc), maximum(esc)]
            rec["holes_t_pass"] = holes_at(orbit, t_pass)
            rec["velocities_t_pass"] = vel_at(orbit, t_pass)
        end
        rec["fates_$variant"] = fates
        rec["tcross_$variant"] = tcross
        rec["bend_deg_$variant"] = bends
        rec["t_end_$variant"] = [p[end, 3] for p in paths]
        rec["wall_s_$variant"] = wall
        logf("fan %-7s fates 0/1/2/3 = %d/%d/%d/%d  %.1f s%s\n", variant, count(==(0), fates),
             count(==(1), fates), count(==(2), fates), count(==(3), fates), wall,
             variant == "moving" ? @sprintf("  t_pass = %.3f", t_pass) : "")
    end
    rec["fate_changed"] = [i - 1 for i in 1:NFAN if rec["fates_moving"][i] != rec["fates_frozen"][i]]
    save_record("rays", rec)
end

# ---------------------------------------------------------------- maps: capture classes
# backward ray of image point (a, b) received at observer time t along the inward plane
# normal d̂; α along y, β along (-cos i, 0, sin i)
function map_ray(st, t, a, b)
    x0 = SVector(t, -b * CI + DCAM * SI, a, b * SI + DCAM * CI)
    k = metric(st, x0) * SVector(1.0, SI, 0.0, CI)
    knull, ok = null_covariant(st, x0, SVector(k[2], k[3], k[4]))
    ok || error("no positive-energy null root at the camera")
    return vcat(x0, knull / knull[1])
end

function classify(st, t, a, b)
    sol = trace_geodesic(st, map_ray(st, t, a, b); λmax = MAP_LMAX, escape = MAP_RESC,
                         reltol = 1e-9, abstol = 1e-11, save_everystep = true)
    f = fate(st, sol.u[end], MAP_RESC)
    bend = bend_deg(sol.u)
    tc = plane_time(sol.u, DHAT)
    f in (1, 2) && return SVector(Float64(f), bend, tc, 0.0)
    f == 0 && return SVector(bend > BEND_CUT ? 3.0 : 0.0, bend, tc, 0.0)
    return SVector(4.0, bend, tc, 0.0)
end

struct ClassProgram{S,G} <: Function
    st::S
    t::Float64
    gen::G
end
Adapt.@adapt_structure ClassProgram
function (p::ClassProgram)(item, i, out)
    a, b = image_coords(p.gen, item)
    @inbounds out[i] = classify(p.st, p.t, a, b)
    return nothing
end

base_work() = (px0 = FOV / BASE; nb = BASE ÷ BLOCK;
               vec([ImageBlock(bi, bj, 0, -FOV / 2 + (bi - 1) * BLOCK * px0,
                               -FOV / 2 + (bj - 1) * BLOCK * px0, BLOCK * px0)
                    for bi in 1:nb, bj in 1:nb]))

function level_coords(work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    coords = Vector{Tuple{Float64,Float64}}(undef, length(work) * bb)
    for (w, blk) in enumerate(work)
        ps = blk.span / BLOCK
        for i in 1:BLOCK, j in 1:BLOCK
            coords[(w - 1) * bb + (i - 1) * BLOCK + j] = (blk.a0 + (i - 0.5) * ps, blk.b0 + (j - 0.5) * ps)
        end
    end
    return coords
end

function trace_level(st, t, work::Vector{ImageBlock})
    bb = BLOCK * BLOCK
    S = Vector{SVector{4,Float64}}(undef, length(work) * bb)
    index = Dict(blk => w for (w, blk) in enumerate(work))
    visit = (blk, vals) -> begin
        w = index[blk]
        for q in 1:bb
            S[(w - 1) * bb + q] = vals[q]
        end
        return ImageBlock[]
    end
    stream_blocks!(AMR_EXEC, ClassProgram(st, t, BlockGrid(BLOCK)), work, BLOCK, visit;
                   outtype = SVector{4,Float64})
    return S, level_coords(work)
end

"one frame: the class plane (EFF × EFF), the level-0 values and the per-level record"
function frame(st, t)
    bb = BLOCK * BLOCK
    D = Dict{Tuple{Int,Int},Int8}[]
    function cls(l, ix, iy)
        while l >= 0
            v = get(D[l+1], (ix, iy), Int8(-1))
            v >= 0 && return v
            l -= 1; ix = fld(ix, 2); iy = fld(iy, 2)
        end
        return Int8(-1)
    end
    function refine(l, blk, S, w)
        ps = blk.span / BLOCK
        n = BASE * 2^l
        ix0 = round(Int, (blk.a0 + FOV / 2) / ps)
        iy0 = round(Int, (blk.b0 + FOV / 2) / ps)
        o = (w - 1) * bb
        c0 = Int8(S[o + 1][1])
        for ix in ix0-1:ix0+BLOCK, iy in iy0-1:iy0+BLOCK
            (0 <= ix < n && 0 <= iy < n) || continue
            c = (ix0 <= ix < ix0 + BLOCK && iy0 <= iy < iy0 + BLOCK) ?
                Int8(S[o + (ix - ix0) * BLOCK + (iy - iy0) + 1][1]) : cls(l, ix, iy)
            c >= 0 && c != c0 && return true
        end
        return false
    end
    m = Mosaic(BASE, BLOCK, MAXLEVEL, FOV)
    work = base_work()
    lvl_rays = Int[]; lvl_wall = Float64[]
    S0 = SVector{4,Float64}[]
    l = 0
    while !isempty(work)
        t0 = time()
        S, coords = trace_level(st, t, work)
        l == 0 && (S0 = S)
        Threads.@threads for w in eachindex(work)
            paint_block!(m, work[w], view(S, (w - 1) * bb + 1:w * bb))
            mark_visited!(m, work[w])
        end
        ps = FOV / (BASE * 2^l)
        d = Dict{Tuple{Int,Int},Int8}()
        sizehint!(d, length(coords))
        for (k, (a, b)) in enumerate(coords)
            d[(floor(Int, (a + FOV / 2) / ps), floor(Int, (b + FOV / 2) / ps))] = Int8(S[k][1])
        end
        push!(D, d)
        kids = ImageBlock[]
        if l < MAXLEVEL
            need = Vector{Bool}(undef, length(work))
            foreach_item!(ThreadedExecutor(chunk = 64),
                          (w, i, out) -> (out[i] = refine(l, work[w], S, w); nothing),
                          eachindex(work); outs = (need,))
            for (w, blk) in enumerate(work)
                need[w] || continue
                half = blk.span / 2
                for (da, db) in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half))
                    push!(kids, ImageBlock(blk.bi, blk.bj, l + 1, blk.a0 + da, blk.b0 + db, half))
                end
            end
        end
        push!(lvl_rays, length(coords)); push!(lvl_wall, time() - t0)
        work = kids
        l += 1
    end
    return copy(m.planes[1]), S0, (rays = lvl_rays, wall = lvl_wall)
end

"image-plane position (α, β) of a point s"
img(s) = [s[2], -s[1] * CI + s[3] * SI]

function run_maps()
    orbit = CircularOrbit(MASSES, SEP; phase = OMEGA * DCAM)
    rec = merge(common_record(), Dict{String,Any}("inc_deg" => INC, "dcam" => DCAM, "lim" => LIM, "fov" => FOV,
                           "nframe" => NFRAME, "base" => BASE, "block" => BLOCK,
                           "maxlevel" => MAXLEVEL, "eff" => EFF, "finest_px" => FOV / EFF,
                           "phase" => OMEGA * DCAM, "lmax" => MAP_LMAX, "resc" => MAP_RESC,
                           "reltol" => 1e-9, "abstol" => 1e-11, "bend_cut_deg" => BEND_CUT,
                           "classes" => ["escaped, bend <= 360 deg", "captured by hole 1",
                                         "captured by hole 2", "escaped, bend > 360 deg",
                                         "undetermined"],
                           "frames" => Dict{String,Any}[]))
    for k in 0:NFRAME-1
        t = k * (PERIOD / 2) / NFRAME
        fr = Dict{String,Any}("t_obs" => t)
        t0 = time()
        Cm, S0, rm = frame(moving(orbit), t)
        wm = time() - t0
        tc = [s[3] for s in S0 if s[1] == 0.0 && isfinite(s[3])]
        t_pass = median(tc)
        fr["t_pass"] = t_pass
        fr["t_pass_quartiles"] = [quantile(tc, 0.25), quantile(tc, 0.75)]
        fr["holes_t_pass"] = holes_at(orbit, t_pass)
        fr["holes_image_t_pass"] = [img(s) for s in holes_at(orbit, t_pass)]
        fr["velocities_t_pass"] = vel_at(orbit, t_pass)
        t0 = time()
        Cf, _, rf = frame(frozen(orbit, t_pass), t)
        wf = time() - t0
        write_npy(joinpath(OUT, @sprintf("map_moving_%d.npy", k)), Cm)
        write_npy(joinpath(OUT, @sprintf("map_frozen_%d.npy", k)), Cf)
        capm = (Cm .== 1.0) .| (Cm .== 2.0)
        capf = (Cf .== 1.0) .| (Cf .== 2.0)
        fr["rays_moving"] = rm.rays; fr["wall_s_moving"] = wm
        fr["rays_frozen"] = rf.rays; fr["wall_s_frozen"] = wf
        fr["classes_moving"] = [count(==(Float64(q)), Cm) for q in 0:4]
        fr["classes_frozen"] = [count(==(Float64(q)), Cf) for q in 0:4]
        fr["class_changed_px"] = count(Cm .!= Cf)
        fr["captured_moving_only_px"] = count(capm .& .!capf)
        fr["captured_frozen_only_px"] = count(capf .& .!capm)
        push!(rec["frames"], fr)
        logf("frame %d  t = %6.2f  t_pass = %8.3f  moving %6d rays %5.0f s  frozen %6d rays %5.0f s  classes %s / %s  changed %d px\n",
             k, t, t_pass, sum(rm.rays), wm, sum(rf.rays), wf, string(fr["classes_moving"]),
             string(fr["classes_frozen"]), fr["class_changed_px"])
        save_record("maps", rec)
    end
end

# ---------------------------------------------------------------- scan: the freeze time
"capture state (0 escaped, 1 or 2 the hole) of a uniform NSCAN² grid at observer time t"
function capture_grid(st, t)
    c = collect(range(-LIM + LIM / NSCAN, LIM - LIM / NSCAN; length = NSCAN))
    out = zeros(Int8, NSCAN, NSCAN)                     # [i_β, i_α]
    Threads.@threads for ia in 1:NSCAN
        for ib in 1:NSCAN
            v = classify(st, t, c[ia], c[ib])[1]
            out[ib, ia] = v in (1.0, 2.0) ? Int8(v) : Int8(0)
        end
    end
    return out
end

function run_scan()
    maps = read(joinpath(OUT, "records_maps.json"), String)
    tps = [parse(Float64, m.captures[1]) for m in eachmatch(r"\"t_pass\": ([-0-9.e]+)", maps)]
    length(tps) == NFRAME || error("records_maps.json has $(length(tps)) frames, NFRAME = $NFRAME")
    orbit = CircularOrbit(MASSES, SEP; phase = OMEGA * DCAM)
    δs = collect(-SCANSPAN:SCANSPAN)
    rec = merge(common_record(), Dict{String,Any}("nscan" => NSCAN, "offsets" => δs,
                                                  "inc_deg" => INC, "dcam" => DCAM, "lim" => LIM,
                                                  "frames" => Dict{String,Any}[]))
    for k in 0:NFRAME-1
        t = k * (PERIOD / 2) / NFRAME
        t0 = time()
        Cm = capture_grid(moving(orbit), t)
        frac = Float64[]
        for δ in δs
            Cf = capture_grid(frozen(orbit, tps[k+1] + δ), t)
            push!(frac, count(Cm .!= Cf) / NSCAN^2)
        end
        j = argmin(frac)
        push!(rec["frames"], Dict{String,Any}("t_obs" => t, "t_pass" => tps[k+1],
                                              "changed_fraction" => frac,
                                              "captured_fraction_moving" => count(Cm .> 0) / NSCAN^2,
                                              "best_offset" => δs[j], "best_fraction" => frac[j],
                                              "wall_s" => time() - t0))
        logf("scan frame %d: changed fraction at δ = 0 %.4f, minimum %.4f at δ = %d (captured %.4f)  %.0f s\n",
             k, frac[SCANSPAN + 1], frac[j], δs[j], count(Cm .> 0) / NSCAN^2, time() - t0)
        save_record("scan", rec)
    end
end

logf("fig_dynrays: parts %s, %d threads, code %s\n", join(PARTS, ","), Threads.nthreads(), GIT_SHA)
"rays" in PARTS && run_rays()
"maps" in PARTS && run_maps()
"scan" in PARTS && run_scan()
logf("wrote %s\n", OUT)
