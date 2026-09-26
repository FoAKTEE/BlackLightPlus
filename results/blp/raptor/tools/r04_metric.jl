# r04_metric.jl — R04 comparison 1 (metric level): RAPTOR SKS g, g^-1 and Christoffels
# (M2a finite differences, M2b analytic) against BlackLightPlus g, Woodbury g^-1 and the
# forward-AD Christoffels, at 1e4 points per leg; the upstream CKS Kerr metric
# (regularised, 4th-order FD with h ~ |x^i|) is reported against the same AD reference.
#
# Usage: julia --project=. r04_metric.jl <bindir> <outjson> [legs=kerr,s455,s122]
#   <bindir> holds metric_dump (SKS) and metric_dump_cks (upstream CKS, a = 0.9375)
# Exit 0 iff every SKS leg meets: max|g - g_BLP| <= 1e-14, max|g g^-1 - I| <= 1e-13,
# Gamma(M2b) vs AD <= 1e-12 and Gamma(M2a) vs AD <= 1e-7 (per point, max|dGamma|/max|Gamma|).

using BlackLightPlus, StaticArrays, LinearAlgebra, Random, Printf
include(joinpath(@__DIR__, "r04_legs.jl"))
include(joinpath(@__DIR__, "jsonw.jl"))

bindir, outjson = ARGS[1], ARGS[2]
const NPTS = 10_000

"Points: half uniform in the box |x_i| < Lbox, half within 6 M of a hole; all with r_A - r_h,A > 0.01."
function sample_points(st, rng; Lbox)
    pts = SVector{4,Float64}[]
    nh = length(st.mass)
    while length(pts) < NPTS
        if length(pts) < NPTS ÷ 2
            x = SVector(0.0, (2 .* rand(rng, 3) .- 1) .* Lbox...)
        else
            A = rand(rng, 1:nh)
            d = normalize(randn(rng, 3)) * (0.5 + 5.5 * rand(rng))
            x = SVector(0.0, (SVector(st.center[A]...) + d)...)
        end
        horizon_gap(st, x) > 0.01 && push!(pts, x)
    end
    return pts
end

function run_dump(exe, bbh, pts)
    inp = join([@sprintf("%.17g %.17g %.17g %.17g", p...) for p in pts], "\n") * "\n"
    out = withenv("RAPTOR_BBH" => bbh) do
        read(pipeline(`$exe`; stdin = IOBuffer(inp)), String)
    end
    return [parse.(Float64, split(l, ",")) for l in split(strip(out), "\n")]
end

function compare(rows, st, pts)
    gerr = 0.0; gierr = 0.0; iderr = 0.0; iderr_blp = 0.0; an = 0.0; fd = 0.0
    wan = 0; wfd = 0
    for (k, (v, x)) in enumerate(zip(rows, pts))
        gR = SMatrix{4,4}(reshape(v[5:20], 4, 4)')          # row-major -> matrix
        giR = SMatrix{4,4}(reshape(v[21:36], 4, 4)')
        gB = metric(st, x); giB = inverse_metric(st, x)
        ΓB = christoffel(st, x)                               # Γ[μ,α,β] (1-based)
        gerr = max(gerr, maximum(abs.(gR - gB)))
        gierr = max(gierr, maximum(abs.(giR - giB)))
        iderr = max(iderr, maximum(abs.(gR * giR - I)))
        iderr_blp = max(iderr_blp, maximum(abs.(gB * giB - I)))
        sc = maximum(abs.(ΓB))
        dfd = 0.0; dan = 0.0
        for a in 1:4, b in 1:4, c in 1:4
            idx = 36 + (a - 1) * 16 + (b - 1) * 4 + c
            dfd = max(dfd, abs(v[idx] - ΓB[a, b, c]))
            dan = max(dan, abs(v[idx+64] - ΓB[a, b, c]))
        end
        dfd / sc > fd && (fd = dfd / sc; wfd = k)
        dan / sc > an && (an = dan / sc; wan = k)
    end
    return (g_maxabs = gerr, ginv_maxabs = gierr, g_ginv_minus_I = iderr,
            blp_g_ginv_minus_I = iderr_blp, gamma_analytic_rel = an, gamma_fd_rel = fd,
            worst_analytic_point = collect(pts[wan]), worst_fd_point = collect(pts[wfd]))
end

res = Dict{String,Any}()
pass = true
tmp = mktempdir()
const LEGSEL = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["kerr", "s455", "s122"]
for leg in LEGSEL
    st = leg_spacetime(leg)
    rng = MersenneTwister(hash(leg))
    pts = sample_points(st, rng; Lbox = leg == "kerr" ? 20.0 : 40.0)
    bbh = write_bbh_in(joinpath(tmp, "$leg.in"), leg)
    c = compare(run_dump(joinpath(bindir, "metric_dump"), bbh, pts), st, pts)
    ok = c.g_maxabs <= 1e-14 && c.g_ginv_minus_I <= 1e-13 &&
         c.gamma_analytic_rel <= 1e-12 && c.gamma_fd_rel <= 1e-7
    global pass &= ok
    res[leg] = merge(Dict(string(k) => v for (k, v) in pairs(c)), Dict("pass" => ok, "npoints" => NPTS))
    @printf("%-5s |g-gB| %.2e |gi-giB| %.2e |g gi - I| %.2e (BLP %.2e) Γ_an %.2e Γ_fd %.2e %s\n",
            leg, c.g_maxabs, c.ginv_maxabs, c.g_ginv_minus_I, c.blp_g_ginv_minus_I,
            c.gamma_analytic_rel, c.gamma_fd_rel, ok ? "PASS" : "FAIL")
    if leg == "kerr"
        cks = compare(run_dump(joinpath(bindir, "metric_dump_cks"), bbh, pts), st, pts)
        # the regulariser error vs KS radius
        byr = Dict{String,Any}()
        for (lo, hi) in ((1.3, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 40.0))
            sel = [k for k in eachindex(pts) if lo <= ks_radius(KerrSchildCartesian(0.9375), pts[k]) < hi]
            isempty(sel) && continue
            cs = compare(run_dump(joinpath(bindir, "metric_dump_cks"), bbh, pts[sel]), st, pts[sel])
            byr["r_$(lo)_$(hi)"] = Dict("g_ginv_minus_I" => cs.g_ginv_minus_I,
                                         "g_maxabs" => cs.g_maxabs,
                                         "gamma_fd_rel" => cs.gamma_fd_rel, "n" => length(sel))
        end
        res["kerr_upstream_cks"] = merge(Dict(string(k) => v for (k, v) in pairs(cks)),
                                         Dict("by_ks_radius" => byr,
                                              "note" => "upstream CKS metric (1e-3 regularisers, FD step 1e-4 |x^i|); report only"))
        @printf("CKS   |g-gB| %.2e |g gi - I| %.2e Γ_fd %.2e (report)\n",
                cks.g_maxabs, cks.g_ginv_minus_I, cks.gamma_fd_rel)
    end
end
res["pass"] = pass
res["thresholds"] = Dict("g_maxabs" => 1e-14, "g_ginv_minus_I" => 1e-13,
                         "gamma_analytic_rel" => 1e-12, "gamma_fd_rel" => 1e-7)
mkpath(dirname(outjson))
write_json(outjson, res)
println(pass ? "R04 metric PASS" : "R04 metric FAIL")
exit(pass ? 0 : 1)
