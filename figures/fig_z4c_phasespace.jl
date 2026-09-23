# fig_z4c_phasespace.jl — α-β phase-space deck for the AthenaK Z4c BBH slice at
# one camera angle ι: (a) dense capture/escape map, (b) light-ring boundary
# (adaptive CPU anchors + bisection-refined polylines), (c) multiangle ray deck
# (launch fates, bend angles, momentum-subspace projections, 3D trajectories).
# GPU (CUDA) map/ring bisection when a device is usable, threaded CPU (Vern7) otherwise.
# Writes map.npy, ring.npy, anchors.npy, deck.json, traj_*.npy to
# figures/out/z4c_phasespace/ (plot with plot_z4c_phasespace.py).
# Env: BLP_Z4C, BLP_FIG_IOTA (30), BLP_FIG_NMAP (283 GPU / 101 CPU),
#      BLP_FIG_NAZ (36), BLP_FIG_NPSI (8).  Run: julia --project=. -t 16 ...
using LinearAlgebra
using Printf
using StaticArrays
using BlackLightPlus
using BlackLightPlus: ADMGridSpacetime, z4c_grid_spacetime, sample_adm,
    trace_geodesic, nullify_covariant, null_error, metric, write_npy

const HAS_GPU = let ok = false
    try
        @eval using CUDA
        ok = @eval CUDA.functional() ? (CUDA.zeros(Float64, 1); true) : false
    catch err
        @printf("[ps] CUDA unusable (%s) — CPU fallback\n", sprint(showerror, err))
        ok = false
    end
    ok
end
HAS_GPU && include(joinpath(dirname(@__DIR__), "src", "gpu_spacetime.jl"))

const BIN = get(ENV, "BLP_Z4C", "/data2/haiyangw/aurora/nr_bbh_bl/bbh.z4c.00001.bin")
const OUT = joinpath(@__DIR__, "out", "z4c_phasespace")
# code version recorded in deck.json: HEAD SHA, "-dirty" if src/ has uncommitted edits
const GIT_SHA = try
    root = dirname(@__DIR__)
    sha = readchomp(`git -C $root rev-parse HEAD`)
    isempty(readchomp(`git -C $root status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end

const HALF = 12.0
const NGRID = 192
const ALPHA_TERM = 0.55            # orbit-limited absorber: terminate at lapse α < 0.55
const ESCAPE = 95.0
const LIM = 7.5
const A0 = 0.041015625             # off-symmetry-plane deck row
const LADDER = (1e-2, 1e-4)
const RHOS = (1.5, 3.0, 4.5, 6.5)
const XP = 3.3298                  # puncture |x|
const BTOL = 1e-4                  # CPU anchor bisection tolerance
const BTOL_GPU = 1e-5              # GPU batched-bisection tolerance
const GPU_C = 60.0
const GPU_NMAX = 20000
const NAZ_MAX = 2000
const RAY_BUDGET = 12000
const NCOMP_MAX = 12

const IOTA = parse(Float64, get(ENV, "BLP_FIG_IOTA", "30"))
const NMAP = parse(Int, get(ENV, "BLP_FIG_NMAP", HAS_GPU ? "283" : "101"))
const NAZ = parse(Int, get(ENV, "BLP_FIG_NAZ", "36"))
const NPSI = parse(Int, get(ENV, "BLP_FIG_NPSI", "8"))
const TAG = @sprintf("i%02d", round(Int, IOTA))

basis(ιdeg) = begin
    ι = deg2rad(ιdeg)
    (d=SVector(cos(ι), 0.0, sin(ι)), ea=SVector(0.0, 1.0, 0.0),
     eb=SVector(-sin(ι), 0.0, cos(ι)))
end

"entry point of the camera line for image coords (a, b): ray-box intersection"
function entry_point(B, a, b; edge=11.9)
    P0 = 90.0 * B.d + a * B.ea + b * B.eb
    n = -B.d
    t_enter, t_exit = -Inf, Inf
    for k in 1:3
        if abs(n[k]) < 1e-15
            abs(P0[k]) > edge && return nothing
        else
            t1 = (-edge - P0[k]) / n[k]
            t2 = (edge - P0[k]) / n[k]
            lo, hi = minmax(t1, t2)
            t_enter = max(t_enter, lo)
            t_exit = min(t_exit, hi)
        end
    end
    t_enter >= t_exit && return nothing
    return P0 + (t_enter + 1e-3) * n
end

"backward ray state p = -k of the photon received along +d̂ (k_t = -1), nullified against the interpolated metric"
function ray_state(st, B, a, b)
    P = entry_point(B, a, b)
    P === nothing && return nothing
    x0 = SVector(0.0, P[1], P[2], P[3])
    k = metric(st, x0) * SVector(1.0, B.d[1], B.d[2], B.d[3])
    k0 = nullify_covariant(st, x0, SVector(k[2], k[3], k[4]))
    return vcat(x0, k0 / k0[1])
end

function trace(st, B, a, b; save=false)
    y0 = ray_state(st, B, a, b)
    y0 === nothing && return nothing
    return trace_geodesic(st, y0; λmax=6000.0, reltol=1e-10, abstol=1e-12,
                          escape=ESCAPE, save_everystep=save)
end

function fate(st, sol)
    sol === nothing && return :offgrid
    u = sol.u[end]
    r = sqrt(u[2]^2 + u[3]^2 + u[4]^2)
    r >= ESCAPE - 0.5 && return :escape
    α = sample_adm(st, u[2], u[3], u[4])[1]
    α <= ALPHA_TERM + 0.01 && return :capture
    return :other
end

captured(st, B, a, b) = fate(st, trace(st, B, a, b)) == :capture

"batched device classification of image points (true = captured)"
function gpu_captured(st, B, pts)
    rays = SVector{8,Float64}[]
    idx = Int[]
    for (k, (a, b)) in enumerate(pts)
        y = ray_state(st, B, a, b)
        y === nothing && continue
        push!(rays, y)
        push!(idx, k)
    end
    out = falses(length(pts))
    if !isempty(rays)
        status, _, _, _, _ = trace_shadow_gpu(st, rays; C=GPU_C, escape=ESCAPE,
                                              nmax=GPU_NMAX)
        for (j, k) in enumerate(idx)
            out[k] = status[j] == 1.0
        end
    end
    return out
end

"4-connected component labeling of the captured mask"
function label_components(mask::Matrix{Bool})
    lab = zeros(Int, size(mask))
    ncomp = 0
    for j in axes(mask, 2), i in axes(mask, 1)
        (mask[i, j] && lab[i, j] == 0) || continue
        ncomp += 1
        stack = [(i, j)]
        lab[i, j] = ncomp
        while !isempty(stack)
            ci, cj = pop!(stack)
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ni, nj = ci + di, cj + dj
                (1 <= ni <= size(mask, 1) && 1 <= nj <= size(mask, 2)) || continue
                if mask[ni, nj] && lab[ni, nj] == 0
                    lab[ni, nj] = ncomp
                    push!(stack, (ni, nj))
                end
            end
        end
    end
    return lab, ncomp
end

component_geometry(lab, keep, axesv) = map(keep) do c
    cells = findall(==(c), lab)
    ca = sum(axesv[I[1]] for I in cells) / length(cells)
    cb = sum(axesv[I[2]] for I in cells) / length(cells)
    (c=c, ca=ca, cb=cb,
     rmax=maximum(hypot(axesv[I[1]] - ca, axesv[I[2]] - cb) for I in cells))
end

"CPU anchors: per-component radial scans, every capture transition bisected with
real traces to BTOL; transitions attributed by the captured side's map label"
function cpu_anchors(st, B, comps, labof, astep, pstep)
    rows = Vector{NTuple{5,Float64}}()
    lk = ReentrantLock()
    for g in comps
        rmax = g.rmax + 2 * astep
        Threads.@threads for m in 1:NAZ
            ψ = 2π * (m - 1) / NAZ
            ux, uy = cos(ψ), sin(ψ)
            prev = captured(st, B, g.ca, g.cb)
            s_prev = 0.0
            ktr = 0
            s = astep
            while s <= rmax
                a, b = g.ca + s * ux, g.cb + s * uy
                cur = (abs(a) <= LIM + 2.9 && abs(b) <= LIM + 2.9) ?
                      captured(st, B, a, b) : false
                if cur != prev
                    lo, hi = s_prev, s
                    while hi - lo > BTOL
                        mid = (lo + hi) / 2
                        cm = captured(st, B, g.ca + mid * ux, g.cb + mid * uy)
                        (cm == prev) ? (lo = mid) : (hi = mid)
                    end
                    se = (lo + hi) / 2
                    scap = prev ? max(lo - 0.45 * pstep, 0.0) : hi + 0.45 * pstep
                    if labof(g.ca + scap * ux, g.cb + scap * uy) == g.c
                        ktr += 1
                        lock(lk) do
                            push!(rows, (Float64(g.c), Float64(ktr),
                                         g.ca + se * ux, g.cb + se * uy, hi - lo))
                        end
                    end
                    prev = cur
                end
                s_prev = s
                s += astep
            end
        end
    end
    sort!(rows; by=r -> (r[1], r[2], atan(r[4], r[3])))
    M = zeros(length(rows), 5)
    for (k, r) in enumerate(rows)
        M[k, :] .= r
    end
    return M
end

"batched classification of image points: device tracer with a GPU, threaded Vern7 otherwise"
function batch_captured(st, B, pts)
    HAS_GPU && return gpu_captured(st, B, pts)
    out = falses(length(pts))
    Threads.@threads for k in eachindex(pts)
        out[k] = captured(st, B, pts[k]...)
    end
    return out
end

"rings: bracket transitions on the dense map, then batched bisection (GPU or CPU)"
function gpu_rings(st, B, gmask, comps, labof, step)
    naz = min(NAZ_MAX, max(90, RAY_BUDGET ÷ max(length(comps), 1)))
    gmaskof = (a, b) -> begin
        i = clamp(round(Int, (a + LIM) / step) + 1, 1, NMAP)
        j = clamp(round(Int, (b + LIM) / step) + 1, 1, NMAP)
        gmask[i, j]
    end
    br = NamedTuple[]
    for g in comps
        rmax = g.rmax + 2 * step
        for m in 1:naz
            ψ = 2π * (m - 1) / naz
            ux, uy = cos(ψ), sin(ψ)
            prev = gmaskof(g.ca, g.cb)
            s_prev = 0.0
            ktr = 0
            s = step
            while s <= rmax
                cur = gmaskof(g.ca + s * ux, g.cb + s * uy) &&
                      abs(g.ca + s * ux) <= LIM && abs(g.cb + s * uy) <= LIM
                if cur != prev
                    scap = prev ? max(s_prev - 0.45 * step, 0.0) : s + 0.45 * step
                    if labof(g.ca + scap * ux, g.cb + scap * uy) == g.c
                        ktr += 1
                        push!(br, (c=g.c, k=ktr, ca=g.ca, cb=g.cb, ux=ux, uy=uy,
                                   lo=s_prev, hi=s, prev=prev))
                    end
                    prev = cur
                end
                s_prev = s
                s += step
            end
        end
    end
    lo = [b.lo for b in br]
    hi = [b.hi for b in br]
    niter = ceil(Int, log2(step / BTOL_GPU))
    for it in 1:niter
        mids = [(br[q].ca + (lo[q] + hi[q]) / 2 * br[q].ux,
                 br[q].cb + (lo[q] + hi[q]) / 2 * br[q].uy) for q in eachindex(br)]
        capm = batch_captured(st, B, mids)
        for q in eachindex(br)
            mid = (lo[q] + hi[q]) / 2
            (capm[q] == br[q].prev) ? (lo[q] = mid) : (hi[q] = mid)
        end
    end
    M = zeros(length(br), 5)
    for q in eachindex(br)
        se = (lo[q] + hi[q]) / 2
        M[q, :] .= (Float64(br[q].c), Float64(br[q].k),
                    br[q].ca + se * br[q].ux, br[q].cb + se * br[q].uy,
                    hi[q] - lo[q])
    end
    return M, naz, niter
end

function edge_bisect_b(st, B, b_cap, b_esc; tol=1e-8)
    @assert captured(st, B, A0, b_cap) "seed b=$b_cap not captured"
    @assert !captured(st, B, A0, b_esc) "seed b=$b_esc not escaped"
    lo, hi = b_cap, b_esc
    while abs(hi - lo) > tol
        mid = (lo + hi) / 2
        captured(st, B, A0, mid) ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"classified b-scan along a=A0 (plunger targets come from the scan, not from
coordinate projections, which the foreground hole's deflection can shift)"
function b_scan(st, B; step=0.05, bmax=8.5)
    bs = collect(-bmax:step:bmax)
    capm = falses(length(bs))
    Threads.@threads for i in eachindex(bs)
        capm[i] = captured(st, B, A0, bs[i])
    end
    return bs, capm
end

function cumulative_bend_deg(M)
    tot = 0.0
    prev = nothing
    for k in 1:size(M, 1)-1
        d = SVector(M[k+1, 3] - M[k, 3], M[k+1, 4] - M[k, 4], M[k+1, 5] - M[k, 5])
        n = norm(d)
        n < 1e-12 && continue
        t = d / n
        prev !== nothing && (tot += acos(clamp(dot(prev, t), -1.0, 1.0)))
        prev = t
    end
    return rad2deg(tot)
end

function ray_bundle(st, B, name, a, b)
    sol = trace(st, B, a, b; save=true)
    sol === nothing && return nothing
    m = length(sol.u)
    M = zeros(m, 11)
    for k in 1:m
        u = sol.u[k]
        M[k, 1] = sol.t[k]
        M[k, 2:9] .= u
        ongrid = maximum(abs.((u[2], u[3], u[4]))) <= HALF
        M[k, 10] = ongrid ? sample_adm(st, u[2], u[3], u[4])[1] : 1.0
        M[k, 11] = null_error(st, u)
    end
    f = fate(st, sol)
    Es = -M[:, 6]
    u_end = sol.u[end]
    pa = pb = NaN
    if f == :escape                 # final transverse momentum in the (e_a, e_b) subspace
        v = SVector(u_end[6], u_end[7], u_end[8])
        v = v / norm(v)
        pa = dot(v, B.ea)
        pb = dot(v, B.eb)
    end
    return (name=name, a=a, b=b, fate=f, M=M,
            e_drift=maximum(abs.(Es .- Es[1])) / abs(Es[1]),
            bend_deg=cumulative_bend_deg(M), pa=pa, pb=pb,
            end_r=sqrt(u_end[2]^2 + u_end[3]^2 + u_end[4]^2),
            end_alpha=M[end, 10], npts=m)
end

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Symbol) = json_str(string(x))
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::Nothing) = "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
function json_str(d::AbstractDict)
    items = [json_str(string(k)) * ":" * json_str(v)
             for (k, v) in sort(collect(d); by=p -> string(p.first))]
    return "{" * join(items, ",") * "}"
end

function main()
    mkpath(OUT)
    t0 = time()
    st, snap = z4c_grid_spacetime(BIN, HALF, NGRID; alpha_min=ALPHA_TERM)
    @printf("[ps %s] grid spacetime built in %.1fs (t=%.2f)  backend=%s\n",
            TAG, time() - t0, snap.time, HAS_GPU ? "gpu" : "cpu")
    B = basis(IOTA)
    axesv = collect(range(-LIM, LIM; length=NMAP))
    step = 2 * LIM / (NMAP - 1)

    # (a) dense capture map (row index = a axis, column index = b axis)
    t1 = time()
    mask = falses(NMAP, NMAP)
    if HAS_GPU
        pts = [(axesv[i], axesv[j]) for i in 1:NMAP for j in 1:NMAP]
        capv = gpu_captured(st, B, pts)
        k = 0
        for i in 1:NMAP, j in 1:NMAP
            k += 1
            mask[i, j] = capv[k]
        end
    else
        idx = [(i, j) for i in 1:NMAP for j in 1:NMAP]
        Threads.@threads for k in eachindex(idx)
            i, j = idx[k]
            mask[i, j] = captured(st, B, axesv[i], axesv[j])
        end
    end
    ncap = count(mask)
    @printf("[ps %s] map %dx%d in %.0fs — %d captured (%.1f%%)\n",
            TAG, NMAP, NMAP, time() - t1, ncap, 100ncap / NMAP^2)
    write_npy(joinpath(OUT, "map.npy"), Float64.(Matrix(mask)))

    lab, ncomp = label_components(Matrix(mask))
    sizes = [count(==(c), lab) for c in 1:ncomp]
    kept_all = findall(s -> s >= 4, sizes)
    order = sort(kept_all; by=c -> -sizes[c])
    keep = sort(order[1:min(NCOMP_MAX, length(order))])
    comps = component_geometry(lab, keep, axesv)
    labof = (a, b) -> begin
        i = clamp(round(Int, (a + LIM) / step) + 1, 1, NMAP)
        j = clamp(round(Int, (b + LIM) / step) + 1, 1, NMAP)
        lab[i, j]
    end
    @printf("[ps %s] %d components (%d kept; sizes %s)\n", TAG, ncomp,
            length(keep), sizes)

    # (b) light-ring boundary: CPU anchors + refined polylines
    t2 = time()
    astep = 2 * LIM / 100           # anchor scan step, fixed at the reference pitch
    A = cpu_anchors(st, B, comps, labof, astep, max(astep, step))
    write_npy(joinpath(OUT, "anchors.npy"), A)
    @printf("[ps %s] %d CPU anchors at %d az (tol %.0e) in %.0fs\n",
            TAG, size(A, 1), NAZ, BTOL, time() - t2)
    t3 = time()
    R, naz_ring, niter = gpu_rings(st, B, mask, comps, labof, step)
    btol_ring = BTOL_GPU
    @printf("[ps %s] %d ring points (%s, %d az/comp, %d rounds) in %.0fs\n",
            TAG, size(R, 1), HAS_GPU ? "GPU" : "CPU", naz_ring, niter, time() - t3)
    write_npy(joinpath(OUT, "ring.npy"), R)

    # (c) multiangle deck at this angle
    t4 = time()
    bp = XP * (-B.eb[1])
    bs, capm = b_scan(st, B)
    icap = findall(capm)
    isempty(icap) && error("deck $TAG: no captured samples on the b-scan")
    ihi, ilo = maximum(icap), minimum(icap)
    b_hi = edge_bisect_b(st, B, bs[ihi], bs[min(ihi + 1, length(bs))])
    b_lo = edge_bisect_b(st, B, bs[ilo], bs[max(ilo - 1, 1)])
    @printf("[ps %s] outer b-edges [%.6f, %.6f] in %.0fs\n",
            TAG, b_lo, b_hi, time() - t4)

    deck = Tuple{String,Float64,Float64,String}[]
    if abs(2 * bp) < 0.05                          # aligned: projections coincide
        push!(deck, ("$(TAG)_plunge_front", 0.0, 0.0, "capture"))
        push!(deck, ("$(TAG)_plunge_axis2", 0.0, 0.02, "capture"))
    else
        for (k, btgt) in enumerate((-abs(bp), abs(bp)))
            ib = icap[argmin(abs.(bs[icap] .- btgt))]
            push!(deck, ("$(TAG)_plunge_b$(k)", A0, bs[ib], "capture"))
        end
    end
    for (k, δ) in enumerate(LADDER)
        push!(deck, ("$(TAG)_lad_hi_cap$(k)", A0, b_hi - δ, "capture"))
        push!(deck, ("$(TAG)_lad_hi_esc$(k)", A0, b_hi + δ, "escape"))
        push!(deck, ("$(TAG)_lad_lo_cap$(k)", A0, b_lo + δ, "capture"))
        push!(deck, ("$(TAG)_lad_lo_esc$(k)", A0, b_lo - δ, "escape"))
    end
    for ρ in RHOS, j in 1:NPSI
        ψ = 2π * (j - 1 + 0.5) / NPSI
        push!(deck, (@sprintf("%s_grid_r%.1f_p%d", TAG, ρ, j),
                     ρ * cos(ψ), ρ * sin(ψ), "any"))
    end

    rays = Dict{String,Any}[]
    t5 = time()
    for (name, a, b, expect) in deck
        rb = ray_bundle(st, B, name, a, b)
        rb === nothing && error("ray $name misses the grid box")
        write_npy(joinpath(OUT, "traj_$(name).npy"), rb.M)
        push!(rays, Dict("name" => name, "a" => a, "b" => b,
                         "fate" => string(rb.fate), "expected" => expect,
                         "E_drift_rel" => rb.e_drift, "bend_deg" => rb.bend_deg,
                         "p_ea" => rb.pa, "p_eb" => rb.pb,
                         "end_r" => rb.end_r, "end_alpha" => rb.end_alpha,
                         "n_points" => rb.npts))
    end
    @printf("[ps %s] %d deck rays in %.0fs — captured %d, max bend %.0f°\n",
            TAG, length(deck), time() - t5,
            count(r -> r["fate"] == "capture", rays),
            maximum(r["bend_deg"] for r in rays))

    meta = Dict(
        "deck" => "AthenaK Z4c BBH head-on merger, frozen slice",
        "dump_time" => snap.time, "backend" => HAS_GPU ? "gpu" : "cpu",
        "iota_deg" => IOTA, "tag" => TAG, "git_sha" => GIT_SHA,
        "d_hat" => collect(B.d), "e_a" => collect(B.ea), "e_b" => collect(B.eb),
        "half" => HALF, "ngrid" => NGRID, "alpha_term" => ALPHA_TERM,
        "escape" => ESCAPE, "lim" => LIM, "nmap" => NMAP,
        "naz_anchor" => NAZ, "naz_ring" => naz_ring,
        "btol_anchor" => BTOL, "btol_ring" => btol_ring,
        "a0_row" => A0, "xp" => XP, "ladder" => collect(LADDER),
        "rhos" => collect(RHOS), "npsi" => NPSI,
        "proj_b_abs" => abs(bp), "edge_b_lo" => b_lo, "edge_b_hi" => b_hi,
        "n_captured_cells" => ncap, "n_components" => ncomp,
        "kept_components" => collect(keep),
        "component_sizes" => sizes,
        "n_anchor_points" => size(A, 1), "n_ring_points" => size(R, 1),
        "integrator" => Dict("solver" => "Vern7", "reltol" => 1e-10,
                             "abstol" => 1e-12),
        "traj_columns" => ["lambda", "t", "x", "y", "z", "p_t", "p_x", "p_y",
                           "p_z", "alpha_interp(offgrid=1)", "k_dot_k"],
        "rays" => rays)
    open(joinpath(OUT, "deck.json"), "w") do io
        write(io, json_str(meta))
    end
    println("Z4C-PHASESPACE DONE -> ", OUT)
end

main()
