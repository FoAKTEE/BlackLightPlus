# fig_z4c_phasespace.jl — capture/escape phase space of the AthenaK Z4c binary-black-hole
# slice seen from one camera direction ι above the puncture axis:
#   (a) capture map on NMAP × NMAP image points over |a|, |b| ≤ 7.5 M;
#   (b) capture boundary: 4-connected captured components of the map (the 12 largest of at
#       least 4 cells), radial scans from each component centroid at NAZ azimuths with every
#       capture transition bisected by traced rays to 1e-4 M (anchors), and the map
#       transitions on up to 2000 azimuths per component bisected to 1e-5 M (ring);
#   (c) the ray deck: two plungers on the row a = A0 (b nearest the projected punctures),
#       ladders 1e-2 and 1e-4 M inside and outside the outer capture edges of that row
#       (edges bisected to 1e-8 M) and a polar grid of radii 1.5, 3, 4.5, 6.5 M at NPSI
#       angles, each traced with its trajectory saved.
#
# Spacetime: the dump resampled onto a 192³ ADMGrid over |x_i| ≤ 12 M (Catmull-Rom), frozen
# slice. A ray at image point (a, b) is the photon received along +d̂ (d̂ = (cos ι, 0, sin ι),
# e_a = ŷ, e_b = (-sin ι, 0, cos ι)) traced backward from the point where the line
# 90 d̂ + a e_a + b e_b - s d̂ enters the box |x_i| ≤ 11.9, its covector nullified in the
# interpolated metric there (p_t = +1). Geodesics: adaptive Vern7, reltol 1e-10,
# abstol 1e-12, λ ≤ 6000. Fates: captured where the lapse falls to 0.55 (event at
# alpha_min = 0.55, accepted at α ≤ 0.56), escaped at r ≥ 95 M (accepted at r ≥ 94.5 M).
# Bending angle: the summed turning angle of the saved coordinate path. Every batch of rays
# runs as a per-item program on the threaded executor.
#
# Writes figures/paper/out/z4c_phasespace/: map.npy (NMAP × NMAP, row = a, 1 = captured),
# anchors.npy and ring.npy (rows: component, transition index, a, b, bracket width),
# traj_<name>.npy (columns in deck.json `traj_columns`), deck.json (setup, counts, per-ray
# fate, bending angle, final escape direction, code SHA, threads, wall times).
#
# Run from the repository root:
#   julia -t 48 --project=. figures/paper/fig_z4c_phasespace.jl
# Env: BLP_Z4C (dump path), BLP_FIG_IOTA (30), BLP_FIG_NMAP (1201), BLP_FIG_NAZ (72),
#      BLP_FIG_NPSI (8).
using Dates
using LinearAlgebra
using Printf
using StaticArrays
using BlackLightPlus
using BlackLightPlus.Spacetime: ADMGrid, trace_geodesic, nullify_covariant, null_error, metric, sample_adm
using BlackLightPlus.Execution: ThreadedExecutor, foreach_item!, image_coords, PixelGrid, CoordList

const BIN = get(ENV, "BLP_Z4C", "/data2/haiyangw/aurora/nr_bbh_bl/bbh.z4c.00001.bin")
const OUT = joinpath(@__DIR__, "out", "z4c_phasespace")
const ROOT = dirname(dirname(@__DIR__))
# code version recorded in the metadata: HEAD SHA, "-dirty" if src has uncommitted edits
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
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
const BTOL = 1e-4                  # anchor bisection tolerance
const BTOL_RING = 1e-5             # ring bisection tolerance
const NAZ_MAX = 2000
const RAY_BUDGET = 12000
const NCOMP_MAX = 12
const LAMBDA_MAX = 6000.0
const RELTOL = 1e-10
const ABSTOL = 1e-12

const IOTA = parse(Float64, get(ENV, "BLP_FIG_IOTA", "30"))
const NMAP = parse(Int, get(ENV, "BLP_FIG_NMAP", "1201"))
const NAZ = parse(Int, get(ENV, "BLP_FIG_NAZ", "72"))
const NPSI = parse(Int, get(ENV, "BLP_FIG_NPSI", "8"))
const TAG = @sprintf("i%02d", round(Int, IOTA))

"formatted progress line, flushed at once"
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

# one ray per item, handed to threads dynamically in chunks of 16
const EXEC = ThreadedExecutor(chunk = 16)

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
    return trace_geodesic(st, y0; λmax=LAMBDA_MAX, reltol=RELTOL, abstol=ABSTOL,
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

"""
    CaptureProgram(st, basis, gen)

Per-item program: whether the ray of image point `image_coords(gen, item)` is captured.
"""
struct CaptureProgram{S,Bs,G} <: Function
    st::S
    basis::Bs
    gen::G
end

function (p::CaptureProgram)(item, i, out)
    a, b = image_coords(p.gen, item)
    @inbounds out[i] = captured(p.st, p.basis, a, b)
    return nothing
end

"capture flags of a list of image points (a, b)"
function batch_captured(st, B, pts)
    out = Vector{Bool}(undef, length(pts))
    isempty(pts) && return out
    foreach_item!(EXEC, CaptureProgram(st, B, CoordList(pts)), eachindex(pts); outs=(out,))
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

"""
    AnchorScan(st, basis, comps, labof, astep, pstep)

Per-item program over (component, azimuth) pairs: the radial scan from the component's
centroid at azimuth 2π(m - 1)/NAZ, each capture transition bisected with traced rays to
BTOL and kept when the captured side lies in the component on the map. Writes the rows
(component, transition index, a, b, bracket width) of the item.
"""
struct AnchorScan{S,Bs,C,L} <: Function
    st::S
    basis::Bs
    comps::C
    labof::L
    astep::Float64
    pstep::Float64
end

function (p::AnchorScan)(item, i, out)
    st, B, astep, pstep = p.st, p.basis, p.astep, p.pstep
    g = p.comps[(item - 1) ÷ NAZ + 1]
    m = (item - 1) % NAZ + 1
    rows = NTuple{5,Float64}[]
    rmax = g.rmax + 2 * astep
    ψ = 2π * (m - 1) / NAZ
    ux, uy = cos(ψ), sin(ψ)
    prev = captured(st, B, g.ca, g.cb)
    s_prev = 0.0
    ktr = 0
    s = astep
    while s <= rmax
        a, b = g.ca + s * ux, g.cb + s * uy
        cur = (abs(a) <= LIM + 2.9 && abs(b) <= LIM + 2.9) ? captured(st, B, a, b) : false
        if cur != prev
            lo, hi = s_prev, s
            while hi - lo > BTOL
                mid = (lo + hi) / 2
                cm = captured(st, B, g.ca + mid * ux, g.cb + mid * uy)
                (cm == prev) ? (lo = mid) : (hi = mid)
            end
            se = (lo + hi) / 2
            scap = prev ? max(lo - 0.45 * pstep, 0.0) : hi + 0.45 * pstep
            if p.labof(g.ca + scap * ux, g.cb + scap * uy) == g.c
                ktr += 1
                push!(rows, (Float64(g.c), Float64(ktr), g.ca + se * ux, g.cb + se * uy,
                             hi - lo))
            end
            prev = cur
        end
        s_prev = s
        s += astep
    end
    @inbounds out[i] = rows
    return nothing
end

"anchors: per-component radial scans at NAZ azimuths, transitions bisected to BTOL"
function cpu_anchors(st, B, comps, labof, astep, pstep)
    n = length(comps) * NAZ
    per = Vector{Vector{NTuple{5,Float64}}}(undef, n)
    foreach_item!(ThreadedExecutor(chunk = 1), AnchorScan(st, B, comps, labof, astep, pstep),
                  1:n; outs=(per,))
    rows = reduce(vcat, per; init=NTuple{5,Float64}[])
    sort!(rows; by=r -> (r[1], r[2], atan(r[4], r[3])))
    M = zeros(length(rows), 5)
    for (k, r) in enumerate(rows)
        M[k, :] .= r
    end
    return M
end

"ring: bracket transitions on the dense map, then batched bisection with traced rays"
function map_rings(st, B, gmask, comps, labof, step)
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
    niter = ceil(Int, log2(step / BTOL_RING))
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
    return bs, batch_captured(st, B, [(A0, b) for b in bs])
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

"""
    DeckProgram(st, basis, deck)

Per-item program: the saved trajectory and diagnostics of deck ray `item`.
"""
struct DeckProgram{S,Bs,D} <: Function
    st::S
    basis::Bs
    deck::D
end

function (p::DeckProgram)(item, i, out)
    name, a, b, _ = p.deck[item]
    @inbounds out[i] = ray_bundle(p.st, p.basis, name, a, b)
    return nothing
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
    load_s = time() - t0
    logf("[ps %s] grid spacetime built in %.1fs (t=%.2f), %d threads\n",
            TAG, load_s, snap.time, Threads.nthreads())
    B = basis(IOTA)
    axesv = collect(range(-LIM, LIM; length=NMAP))
    step = 2 * LIM / (NMAP - 1)

    # (a) dense capture map (row index = a axis, column index = b axis)
    t1 = time()
    flags = Vector{Bool}(undef, NMAP * NMAP)            # item (i - 1) NMAP + j: (a_i, b_j)
    foreach_item!(EXEC, CaptureProgram(st, B, PixelGrid(axesv, NMAP)), 1:NMAP^2;
                  outs=(flags,))
    mask = permutedims(reshape(flags, NMAP, NMAP))
    ncap = count(mask)
    map_s = time() - t1
    logf("[ps %s] map %dx%d in %.0fs — %d captured (%.1f%%)\n",
            TAG, NMAP, NMAP, map_s, ncap, 100ncap / NMAP^2)
    write_npy(joinpath(OUT, "map.npy"), Float64.(mask))

    lab, ncomp = label_components(mask)
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
    logf("[ps %s] %d components (%d kept; sizes %s)\n", TAG, ncomp,
            length(keep), sizes)

    # (b) capture boundary: anchors from radial scans, ring from the map transitions
    t2 = time()
    astep = 2 * LIM / 100           # anchor scan step, fixed at the reference pitch
    A = cpu_anchors(st, B, comps, labof, astep, max(astep, step))
    write_npy(joinpath(OUT, "anchors.npy"), A)
    anchors_s = time() - t2
    logf("[ps %s] %d anchors at %d az (tol %.0e) in %.0fs\n",
            TAG, size(A, 1), NAZ, BTOL, anchors_s)
    t3 = time()
    R, naz_ring, niter = map_rings(st, B, mask, comps, labof, step)
    ring_s = time() - t3
    logf("[ps %s] %d ring points (%d az/comp, %d rounds) in %.0fs\n",
            TAG, size(R, 1), naz_ring, niter, ring_s)
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
    edges_s = time() - t4
    logf("[ps %s] outer b-edges [%.6f, %.6f] in %.0fs\n", TAG, b_lo, b_hi, edges_s)

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

    t5 = time()
    bundles = Vector{Any}(undef, length(deck))
    foreach_item!(ThreadedExecutor(chunk = 1), DeckProgram(st, B, deck), eachindex(deck);
                  outs=(bundles,))
    rays = Dict{String,Any}[]
    for (k, (name, a, b, expect)) in enumerate(deck)
        rb = bundles[k]
        rb === nothing && error("ray $name misses the grid box")
        write_npy(joinpath(OUT, "traj_$(name).npy"), rb.M)
        push!(rays, Dict("name" => name, "a" => a, "b" => b,
                         "fate" => string(rb.fate), "expected" => expect,
                         "E_drift_rel" => rb.e_drift, "bend_deg" => rb.bend_deg,
                         "p_ea" => rb.pa, "p_eb" => rb.pb,
                         "end_r" => rb.end_r, "end_alpha" => rb.end_alpha,
                         "n_points" => rb.npts))
    end
    deck_s = time() - t5
    logf("[ps %s] %d deck rays in %.0fs — captured %d, max bend %.0f°\n",
            TAG, length(deck), deck_s,
            count(r -> r["fate"] == "capture", rays),
            maximum(r["bend_deg"] for r in rays))

    meta = Dict(
        "deck" => "AthenaK Z4c BBH head-on merger, frozen slice",
        "package" => "BlackLightPlus", "when" => string(now()), "julia" => string(VERSION),
        "threads" => Threads.nthreads(), "host" => gethostname(),
        "dataset" => BIN, "dump_time" => snap.time, "backend" => "cpu",
        "iota_deg" => IOTA, "tag" => TAG, "git_sha" => GIT_SHA,
        "d_hat" => collect(B.d), "e_a" => collect(B.ea), "e_b" => collect(B.eb),
        "half" => HALF, "ngrid" => NGRID, "alpha_term" => ALPHA_TERM,
        "escape" => ESCAPE, "lim" => LIM, "nmap" => NMAP,
        "naz_anchor" => NAZ, "naz_ring" => naz_ring,
        "btol_anchor" => BTOL, "btol_ring" => BTOL_RING, "ring_rounds" => niter,
        "a0_row" => A0, "xp" => XP, "ladder" => collect(LADDER),
        "rhos" => collect(RHOS), "npsi" => NPSI,
        "proj_b_abs" => abs(bp), "edge_b_lo" => b_lo, "edge_b_hi" => b_hi,
        "n_captured_cells" => ncap, "n_components" => ncomp,
        "kept_components" => collect(keep),
        "component_sizes" => sizes,
        "n_anchor_points" => size(A, 1), "n_ring_points" => size(R, 1),
        "integrator" => Dict("solver" => "Vern7", "reltol" => RELTOL,
                             "abstol" => ABSTOL, "lambda_max" => LAMBDA_MAX),
        "wall_s" => Dict("spacetime_load" => load_s, "map" => map_s,
                         "anchors" => anchors_s, "ring" => ring_s, "edges" => edges_s,
                         "deck" => deck_s, "total" => time() - t0),
        "traj_columns" => ["lambda", "t", "x", "y", "z", "p_t", "p_x", "p_y",
                           "p_z", "alpha_interp(offgrid=1)", "k_dot_k"],
        "rays" => rays)
    open(joinpath(OUT, "deck.json"), "w") do io
        write(io, json_str(meta))
    end
    logf("[ps %s] total %.0fs\n", TAG, time() - t0)
    println("Z4C-PHASESPACE DONE -> ", OUT)
    flush(stdout)
end

# run only as a program, so fig_z4c_amr.jl can include the ray functions
abspath(PROGRAM_FILE) == (@__FILE__) && main()
