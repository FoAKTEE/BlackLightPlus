# r09_geometry.jl — R09 geometry checks for the spinning binaries (dag.md R09), on top of
# the R04 metric and ray legs (r04_metric.jl, r04_rays.jl with legs s093, s122, s122r):
#
#  1. spin orientation: a single Kerr hole (m = 1, a = +-0.9 along z) at i = 90 deg; the
#     shadow edges on the beta = 0 line, bisected in both codes (RAPTOR ray library and
#     BlackLightPlus trace_rays_sampled, C = 160), against the equatorial photon-orbit
#     impact parameters alpha = -xi(r_ph) (observer on +x, alpha = +y): for a > 0 the
#     prograde edge is at alpha = -xi_pro < 0 and the shadow is displaced toward +alpha;
#  2. the same edge bisection on the binaries s122/s122r and s093 at i = 90 through each
#     hole's image (frame dragging of each hole in the superposed metric), both codes;
#  3. horizon location: the surface g^{mu nu} d_mu r_A d_nu r_A = 0 around each hole of the
#     s122 and s093 binaries (2000 directions per hole), in BlackLightPlus and in RAPTOR
#     (g^{mu nu} from metric_dump at the same points), against r_h = m + sqrt(m^2 - a^2)
#     = 0.8 and the capture surface r_h + 1e-3 used by both codes;
#  4. the g^tt >= 0 region of the superposed inverse metric inside a spinning hole (where
#     BlackLightPlus primitive_four_vectors has no lapse): its largest r_A, in both codes.
#
# Usage: julia --project=. -t N r09_geometry.jl <bindir> <out.json>
#   <bindir>: the fork's test drivers built with max_steps 4e5 (r09_verify.sh), so that
#   the bisections (nmax 3e5, C = 160) never meet a budget-limited ray near the critical
#   curves (at the prograde a = 0.9 photon orbit one revolution costs ~1.3e4 steps)
# Exit 0 iff: the Kerr edges of both codes agree to <= 1e-6 M at camera distances 1000 and
# 4000 M, their 1/d extrapolation lies within 2e-4 M of the photon-orbit value (the
# plane-parallel camera at finite d displaces the edges by O(M b / d)), and the shadow is
# displaced toward +alpha for a > 0 (and -alpha for a < 0) in both codes; binary edges
# agree between the codes to <= 1e-5 M; the horizon
# null-surface radius agrees between the codes to <= 1e-9; the g^tt >= 0 region lies
# inside r_A < 0.5 r_h in both codes; no bisection ray ends on the step budget.

using BlackLightPlus, StaticArrays, LinearAlgebra, Random, Printf, Statistics
using ForwardDiff
using BlackLightPlus.Geodesics: trace_rays_sampled, camera_rays_at, PlaneParallelCamera,
                                superposed_control, inverse_metric
const GEO = BlackLightPlus.Geodesics
include(joinpath(@__DIR__, "r04_legs.jl"))
include(joinpath(@__DIR__, "jsonw.jl"))

const BINDIR = ARGS[1]
const OUTJSON = ARGS[2]
DIST = 1000.0
const KERR_DISTS = (1000.0, 4000.0)
const CGEO = 160.0
const NMAX = 300_000
tmp = mktempdir()
const BUDGET_HITS = Ref(0)

# ---------------------------------------------------------------- RAPTOR ray library
const LIB = Libc.Libdl.dlopen(joinpath(BINDIR, "libraptor_sks_an.so"))
const LOAD = Libc.Libdl.dlsym(LIB, :raptor_load_bbh)
const TRACE = Libc.Libdl.dlsym(LIB, :raptor_trace_state)

function write_bbh(path, m, a, c; C = CGEO)
    open(path, "w") do io
        println(io, "nholes ", length(m))
        println(io, "mass ", join(repr.(m), " "))
        println(io, "spin ", join(repr.(a), " "))
        for (A, cc) in enumerate(c)
            println(io, "center$A ", join(repr.(cc), " "))
        end
        println(io, "C ", repr(C)); println(io, "nmax ", NMAX)
        println(io, "escape ", repr(1.1 * DIST)); println(io, "horizon_eps 0.001")
        println(io, "capture_redshift 100.0"); println(io, "cam_dist ", repr(DIST))
    end
    return path
end

function raptor_class(y)
    lp = zeros(9 * (NMAX + 1)); st = Ref{Cint}(0); cl = Ref{Cint}(0)
    s = Vector{Float64}(undef, 8)
    s[1:4] .= y[1:4]; s[5:8] .= .-y[5:8]
    ccall(TRACE, Cint, (Ptr{Float64}, Ptr{Float64}, Ptr{Cint}, Ptr{Cint}), s, lp, st, cl)
    return Int(cl[])
end

function blp_class(st, y)
    states, _ = trace_rays_sampled(st, [y]; C = CGEO, nmax = NMAX, escape = 1.1 * DIST)
    z = states[1][1]
    all(isfinite, z) || return 5
    if superposed_control(st, z, CGEO)[1]
        x = SVector(z[1], z[2], z[3], z[4])
        return argmin([GEO.ks_radius_hole(st, x, A) - GEO.horizon_radius_hole(st, A)
                       for A in eachindex(st.mass)])
    end
    return sqrt(z[2]^2 + z[3]^2 + z[4]^2) > 1.1 * DIST ? 3 : 4
end

"Edge of the shadow of hole `hole` on the image line p(s) = p0 + s u (s >= 0), bisected
between a captured s = 0 and an escaped s = smax, in code `code` (:blp or :raptor)."
function edge(st, cam, p0, u, smax, hole, code; iters = 32)
    cls(s) = begin
        y = camera_rays_at(st, cam, [(p0[1] + s * u[1], p0[2] + s * u[2])])[1]
        c = code === :blp ? blp_class(st, y) : raptor_class(y)
        c == 4 && (BUDGET_HITS[] += 1)        # a budget-limited ray would bias the edge
        c
    end
    cls(0.0) == hole || return NaN
    cls(smax) == 3 || return NaN
    lo, hi = 0.0, smax
    for _ in 1:iters
        mid = (lo + hi) / 2
        cls(mid) == hole ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# ---------------------------------------------------------------- 1. Kerr spin orientation
"Equatorial circular photon orbit radius and xi = L/E (spin a along +z, m = 1)."
function photon_xi(a, prograde::Bool)
    r = 2 * (1 + cos(2 / 3 * acos(prograde ? -abs(a) : abs(a))))
    return -(r^3 - 3r^2 + a^2 * r + a^2) / (a * (r - 1))
end

res = Dict{String,Any}()
pass = true
kerr = Dict{String,Any}()
for a in (0.9, -0.9)
    st = SuperposedKerrSchild([1.0], [a], [(0.0, 0.0, 0.0)])
    xis = (photon_xi(a, true), photon_xi(a, false))       # prograde, retrograde
    ana = sort([-xis[1], -xis[2]])                         # alpha = -xi, (minus, plus) side
    d = Dict{String,Any}("xi_prograde" => xis[1], "xi_retrograde" => xis[2],
                         "alpha_edges_photon_orbit" => ana)
    for dist in KERR_DISTS
        global DIST = dist
        cam = PlaneParallelCamera(inc_deg = 90.0, d = dist, fov = 20.0, npix = 20)
        ccall(LOAD, Cvoid, (Cstring,), write_bbh(joinpath(tmp, "kerr_$(a)_$(dist).in"), [1.0], [a], [(0.0, 0.0, 0.0)]))
        for code in (:blp, :raptor)
            em = -edge(st, cam, (0.0, 0.0), (-1.0, 0.0), 9.0, 1, code)
            ep = edge(st, cam, (0.0, 0.0), (1.0, 0.0), 9.0, 1, code)
            d["d$(Int(dist))_alpha_edges_$(code)"] = [em, ep]
            d["d$(Int(dist))_shift_$(code)"] = (em + ep) / 2
        end
        d["d$(Int(dist))_blp_minus_raptor"] =
            maximum(abs.(d["d$(Int(dist))_alpha_edges_blp"] .- d["d$(Int(dist))_alpha_edges_raptor"]))
    end
    global DIST = 1000.0
    # the plane-parallel camera launches parallel rays at a finite distance d: the edges
    # differ from the asymptotic (photon-orbit) impact parameters by O(M b / d); the 1/d
    # extrapolation of the two distances removes that term
    d1, d2 = KERR_DISTS
    for code in (:blp, :raptor)
        e1 = d["d$(Int(d1))_alpha_edges_$(code)"]; e2 = d["d$(Int(d2))_alpha_edges_$(code)"]
        einf = e2 .+ (e2 .- e1) .* (d1 / (d2 - d1))
        d["extrapolated_alpha_edges_$(code)"] = einf
        d["d1000_minus_photon_orbit_$(code)"] = maximum(abs.(e1 .- ana))
        d["extrapolated_minus_photon_orbit_$(code)"] = maximum(abs.(einf .- ana))
    end
    d["blp_minus_raptor"] = max(d["d$(Int(d1))_blp_minus_raptor"], d["d$(Int(d2))_blp_minus_raptor"])
    ok = d["blp_minus_raptor"] <= 1e-6 &&
         d["extrapolated_minus_photon_orbit_blp"] <= 2e-4 &&
         d["extrapolated_minus_photon_orbit_raptor"] <= 2e-4 &&
         sign(d["d1000_shift_blp"]) == sign(a) && sign(d["d1000_shift_raptor"]) == sign(a)
    d["pass"] = ok
    global pass &= ok
    kerr["a=$(a)"] = d
    @printf("Kerr a=%+.1f i=90: photon-orbit edges [%.6f, %.6f] | d=1000 BLP [%.6f, %.6f] RAPTOR [%.6f, %.6f] | 1/d-extrapolated BLP [%.6f, %.6f] | |BLP-RAPTOR| %.1e | shift %+.4f / %+.4f | %s\n",
            a, ana..., d["d1000_alpha_edges_blp"]..., d["d1000_alpha_edges_raptor"]...,
            d["extrapolated_alpha_edges_blp"]..., d["blp_minus_raptor"], d["d1000_shift_blp"],
            d["d1000_shift_raptor"], ok ? "PASS" : "FAIL")
end
res["kerr_spin_orientation"] = kerr

# ---------------------------------------------------------------- 2. binary shadow edges
bin = Dict{String,Any}()
for leg in ("s122", "s122r", "s093")
    L = LEGS[leg]
    st = leg_spacetime(leg)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = DIST, fov = 56.0, npix = 20)
    ccall(LOAD, Cvoid, (Cstring,), write_bbh(joinpath(tmp, "$leg.in"), L.m, L.a, L.c))
    d = Dict{String,Any}()
    for A in 1:2
        c = L.c[A]
        # i = 90: alpha = y, beta = z. The companion lenses the shadow off the projected
        # centre: seed at the middle of the captured run of the scan alpha = c_y +- 8
        scan = collect(c[2]-8:0.05:c[2]+8)
        cap = [blp_class(st, camera_rays_at(st, cam, [(s, c[3])])[1]) == A for s in scan]
        p0 = (median(scan[cap]), c[3])
        d["hole$(A)_seed_alpha"] = p0[1]
        for (lab, u) in (("minus_alpha", (-1.0, 0.0)), ("plus_alpha", (1.0, 0.0)))
            eb = edge(st, cam, p0, u, 6.0, A, :blp)
            er = edge(st, cam, p0, u, 6.0, A, :raptor)
            d["hole$(A)_$(lab)"] = Dict("blp" => eb, "raptor" => er, "diff" => abs(eb - er))
        end
        # shadow centre on the alpha line (edges at p0 - e_minus and p0 + e_plus)
        sB = p0[1] + (d["hole$(A)_plus_alpha"]["blp"] - d["hole$(A)_minus_alpha"]["blp"]) / 2
        sR = p0[1] + (d["hole$(A)_plus_alpha"]["raptor"] - d["hole$(A)_minus_alpha"]["raptor"]) / 2
        d["hole$(A)_centre_alpha_blp"] = sB; d["hole$(A)_centre_alpha_raptor"] = sR
    end
    worst = maximum(v["diff"] for (k, v) in d if v isa Dict)
    d["max_diff"] = worst
    d["pass"] = isfinite(worst) && worst <= 1e-5
    global pass &= d["pass"]
    bin[leg] = d
    @printf("%-5s i=90 edges: max |BLP - RAPTOR| %.2e | shadow centre alpha: hole1 BLP %+.4f RAPTOR %+.4f, hole2 BLP %+.4f RAPTOR %+.4f | %s\n",
            leg, worst, d["hole1_centre_alpha_blp"], d["hole1_centre_alpha_raptor"], d["hole2_centre_alpha_blp"],
            d["hole2_centre_alpha_raptor"], d["pass"] ? "PASS" : "FAIL")
end
res["binary_shadow_edges_i90"] = bin

# ---------------------------------------------------------------- 3./4. horizon, g^tt region
function run_dump(bbh, pts)
    inp = join([@sprintf("%.17g %.17g %.17g %.17g", p...) for p in pts], "\n") * "\n"
    out = withenv("RAPTOR_BBH" => bbh) do
        read(pipeline(`$(joinpath(BINDIR, "metric_dump"))`; stdin = IOBuffer(inp)), String)
    end
    return [parse.(Float64, split(l, ",")) for l in split(strip(out), "\n")]
end
ginv_raptor(v) = SMatrix{4,4}(reshape(v[21:36], 4, 4)')

grad_r(st, A, x3) = ForwardDiff.gradient(z -> GEO.ks_radius_hole(st, SVector(0.0, z[1], z[2], z[3]), A), x3)
"g^{ij} d_i r_A d_j r_A at the spatial point x3 (gi: the inverse metric there)."
nrm(gi, g) = sum(gi[i+1, j+1] * g[i] * g[j] for i in 1:3, j in 1:3)

hor = Dict{String,Any}()
rng = MersenneTwister(9)
dirs = [normalize(randn(rng, 3)) for _ in 1:2000]
for leg in ("s122", "s093")
    L = LEGS[leg]; st = leg_spacetime(leg)
    bbh = write_bbh(joinpath(tmp, "$leg.in"), L.m, L.a, L.c)
    d = Dict{String,Any}()
    for A in 1:2
        c = SVector(L.c[A]...)
        rh = GEO.horizon_radius_hole(st, A)
        N(x3) = nrm(inverse_metric(st, SVector(0.0, x3...)), grad_r(st, A, x3))
        rA(x3) = GEO.ks_radius_hole(st, SVector(0.0, x3...), A)
        roots = SVector{3,Float64}[]
        for n in dirs
            # bracket on rho (Cartesian distance from the centre): N < 0 inside, > 0 outside
            lo, hi = 0.5, 1.2
            (N(c + lo * n) < 0 && N(c + hi * n) > 0) || continue
            for _ in 1:60
                mid = (lo + hi) / 2
                N(c + mid * n) < 0 ? (lo = mid) : (hi = mid)
            end
            push!(roots, c + (lo + hi) / 2 * n)
        end
        rr = [rA(x) for x in roots]
        # RAPTOR's inverse metric at the BlackLightPlus null-surface points
        rows = run_dump(bbh, [SVector(0.0, x...) for x in roots])
        nR = [nrm(ginv_raptor(v), grad_r(st, A, x)) for (v, x) in zip(rows, roots)]
        scale = [norm(grad_r(st, A, x))^2 for x in roots]
        # g^tt >= 0 region: dense samples in r_A < r_h
        pts = SVector{3,Float64}[]
        while length(pts) < 200_000
            x = c + (2 .* rand(rng, 3) .- 1) .* 0.8
            rA(x) < rh && push!(pts, x)
        end
        gttB = [inverse_metric(st, SVector(0.0, x...))[1, 1] for x in pts]
        rowsg = run_dump(bbh, [SVector(0.0, x...) for x in pts])
        gttR = [ginv_raptor(v)[1, 1] for v in rowsg]
        posB = [rA(pts[k]) for k in eachindex(pts) if gttB[k] >= 0]
        posR = [rA(pts[k]) for k in eachindex(pts) if gttR[k] >= 0]
        d["hole$A"] = Dict("r_h" => rh, "n_directions" => length(roots),
                           "null_surface_rA_min" => minimum(rr), "null_surface_rA_max" => maximum(rr),
                           "null_surface_rA_median" => median(rr),
                           "raptor_norm_at_blp_surface_max_rel" => maximum(abs.(nR) ./ scale),
                           "gtt_nonneg_samples_blp" => length(posB),
                           "gtt_nonneg_samples_raptor" => length(posR),
                           "gtt_nonneg_max_rA_blp" => isempty(posB) ? 0.0 : maximum(posB),
                           "gtt_nonneg_max_rA_raptor" => isempty(posR) ? 0.0 : maximum(posR),
                           "gtt_sign_agreement" => count(k -> (gttB[k] >= 0) == (gttR[k] >= 0), eachindex(pts)) / length(pts),
                           "n_interior_samples" => length(pts))
        h = d["hole$A"]
        ok = length(roots) == length(dirs) && h["raptor_norm_at_blp_surface_max_rel"] <= 1e-9 &&
             h["gtt_nonneg_max_rA_blp"] < 0.5 * rh && h["gtt_nonneg_max_rA_raptor"] < 0.5 * rh
        h["pass"] = ok
        global pass &= ok
        @printf("%-5s hole %d: null surface r_A in [%.6f, %.6f] (median %.6f; r_h %.4f, capture at %.4f) | RAPTOR |g^rr|/|dr|^2 there <= %.1e | g^tt >= 0 at %d/%d (BLP) %d (RAPTOR) interior samples, max r_A %.4f / %.4f | %s\n",
                leg, A, minimum(rr), maximum(rr), median(rr), rh, rh + 1e-3,
                h["raptor_norm_at_blp_surface_max_rel"], length(posB), length(pts), length(posR),
                h["gtt_nonneg_max_rA_blp"], h["gtt_nonneg_max_rA_raptor"], ok ? "PASS" : "FAIL")
    end
    hor[leg] = d
end
res["horizon"] = hor
res["budget_limited_rays_in_bisections"] = BUDGET_HITS[]
pass &= BUDGET_HITS[] == 0
res["pass"] = pass
res["nmax"] = NMAX
res["thresholds"] = Dict("kerr_edge_code_diff_M" => 1e-6, "kerr_edge_vs_photon_orbit_M" => 2e-3,
                         "binary_edge_code_diff_M" => 1e-5, "raptor_null_norm_rel" => 1e-9,
                         "gtt_region_max_rA_over_rh" => 0.5)
res["C"] = CGEO
write_json(OUTJSON, res)
println(pass ? "R09 geometry PASS" : "R09 geometry FAIL")
exit(pass ? 0 : 1)
