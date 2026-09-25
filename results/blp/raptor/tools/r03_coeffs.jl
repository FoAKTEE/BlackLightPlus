# r03_coeffs.jl — R03 coefficient-level comparison (dag.md R03).
#
# Reads the table printed by the fork's tests/coeff_table (EMIS_BLP port and native
# RAPTOR thermal fits on the R03 grid), evaluates BlackLightPlus
# `thermal_synchrotron_polarized` on the same cells, maps it into RAPTOR's fluid basis
# (Q-type coefficients change sign, design.md C25), and writes
#   evidence/R03/summary.json   acceptance numbers (EMIS_BLP), native ratio report, constants
#   evidence/R03/coeff_table.csv per (nu, B, theta_e): max rel. diff over theta_B (EMIS_BLP)
#                                and native/BLP ratios at theta_B = 13 pi/38
# Exit status 0 iff EMIS_BLP agrees to <= 1e-10 relative on every coefficient with
# |value| > 1e-300, vanishes where BLP vanishes, and every sign agrees.
#
# Usage: julia --project=. results/blp/raptor/tools/r03_coeffs.jl <table.csv> [outdir]

using BlackLightPlus, Printf, Statistics
const RAD = BlackLightPlus.Radiation
const CONST = BlackLightPlus.Constants

const NAMES = ("jI", "jQ", "jV", "aI", "aQ", "aV", "rQ", "rV")
const QTYPE = (false, true, false, false, true, false, true, false)
const TOL = 1e-10
const TINY = 1e-300
const N_E = 1.0e5

table = ARGS[1]
outdir = length(ARGS) >= 2 ? ARGS[2] :
         joinpath(@__DIR__, "..", "evidence", "R03")
mkpath(outdir)

lines = readlines(table)
hdr = split(lines[1], ",")
rows = [parse.(Float64, split(l, ",")) for l in lines[2:end] if !isempty(l)]
n = length(rows)

nu = [r[1] for r in rows]; B = [r[2] for r in rows]
te = [r[3] for r in rows]; th = [r[4] for r in rows]
R = [r[4+c] for r in rows, c in 1:8]           # EMIS_BLP port, RAPTOR basis
N = [r[12+c] for r in rows, c in 1:8]          # native RAPTOR fits
Bl = zeros(n, 8)                               # BlackLightPlus, RAPTOR basis
Xs = zeros(n)                                  # x = nu / nu_s (M21)
for i in 1:n
    v = RAD.thermal_synchrotron_polarized(1.0, N_E, te[i], B[i], nu[i], sin(th[i]), cos(th[i]))
    for c in 1:8
        Bl[i, c] = QTYPE[c] ? -v[c] : v[c]
    end
    nu_c = CONST.EE * B[i] / (2π * CONST.ME * CONST.CL)
    Xs[i] = nu[i] / (2 / 9 * nu_c * te[i]^2 * sin(th[i]))
end

# ---------------------------------------------------------------- EMIS_BLP vs BLP
blp = Dict{String,Any}()
pass = true
for c in 1:8
    rel = 0.0; worst = 0; ncmp = 0; zero_mismatch = 0; sign_mismatch = 0; attr = 0.0
    for i in 1:n
        b = Bl[i, c]; r = R[i, c]
        if abs(b) > TINY
            ncmp += 1
            d = abs(r - b) / abs(b)
            attr = max(attr, d / ((1 + cbrt(Xs[i])) * eps()))
            if d > rel
                rel = d; worst = i
            end
            sign(r) == sign(b) || (sign_mismatch += 1)
        elseif abs(r) > TINY
            zero_mismatch += 1
        end
    end
    ok = rel <= TOL && zero_mismatch == 0 && sign_mismatch == 0
    global pass &= ok
    blp[NAMES[c]] = Dict("max_rel_diff" => rel, "n_compared" => ncmp,
                         "n_zero_mismatch" => zero_mismatch,
                         "n_sign_mismatch" => sign_mismatch, "pass" => ok,
                         "max_rel_over_eps_1_plus_cbrt_x" => attr,
                         "worst_cell" => worst == 0 ? nothing :
                             Dict("nu" => nu[worst], "B" => B[worst], "theta_e" => te[worst],
                                  "theta_B" => th[worst], "x" => Xs[worst],
                                  "blp" => Bl[worst, c], "raptor" => R[worst, c]))
end

# ---------------------------------------------------------------- EMIS_NATIVE report
qtile(v, p) = isempty(v) ? NaN : quantile(v, p)
native = Dict{String,Any}()
for c in 1:8
    idx = [i for i in 1:n if abs(Bl[i, c]) > TINY && abs(N[i, c]) > TINY && isfinite(N[i, c])]
    ratio = [N[i, c] / Bl[i, c] for i in idx]
    out = [idx[k] for k in eachindex(idx) if !(0.9 <= ratio[k] <= 1.1)]
    wk = isempty(idx) ? 0 : idx[argmax(abs.(log.(abs.(ratio))) .+ (ratio .< 0) .* 1e3)]
    nneg = count(<(0), ratio)
    native[NAMES[c]] = Dict(
        "n_compared" => length(idx),
        "n_blp_nonzero_native_zero_or_nonfinite" =>
            count(i -> abs(Bl[i, c]) > TINY && !(abs(N[i, c]) > TINY && isfinite(N[i, c])), 1:n),
        "median" => qtile(ratio, 0.5), "p05" => qtile(ratio, 0.05), "p95" => qtile(ratio, 0.95),
        "n_negative_ratio" => nneg,
        "frac_outside_0.9_1.1" => isempty(idx) ? NaN : length(out) / length(idx),
        "theta_e_range_outside" => isempty(out) ? nothing : [minimum(te[out]), maximum(te[out])],
        "x_range_outside" => isempty(out) ? nothing : [minimum(Xs[out]), maximum(Xs[out])],
        "worst_cell" => wk == 0 ? nothing :
            Dict("nu" => nu[wk], "B" => B[wk], "theta_e" => te[wk], "theta_B" => th[wk],
                 "x" => Xs[wk], "ratio" => N[wk, c] / Bl[wk, c]))
end

# ---------------------------------------------------------------- constants (C12, M9)
G_R, MSUN_R, KPC_R = 6.674e-8, 1.989e33, 3.086e21
G_B, MSUN_B, KPC_B = 6.6743e-8, 1.98847e33, 3.0856775814913673e21
lratio = (G_R * MSUN_R) / (G_B * MSUN_B)
dratio = KPC_R / KPC_B
consts = Dict("L_unit_raptor_over_blp" => lratio, "L_unit_rel_change" => lratio - 1,
              "distance_raptor_over_blp" => dratio,
              "pixel_solid_angle_raptor_over_blp" => (lratio / dratio)^2,
              "pixel_solid_angle_rel_change" => (lratio / dratio)^2 - 1,
              "note" => "flux = sum I (px L_unit / D)^2: the unmatched set rescales the flux of a fixed intensity map by (L ratio / D ratio)^2; L_unit also enters every optical depth (ds = dλ L_unit / ν)")

# ---------------------------------------------------------------- JSON + CSV
function js(x, ind = "")
    x === nothing && return "null"
    x isa Bool && return x ? "true" : "false"
    x isa Integer && return string(x)
    x isa AbstractFloat && return isfinite(x) ? @sprintf("%.10g", x) : "null"
    x isa AbstractString && return "\"" * x * "\""
    x isa AbstractVector && return "[" * join(js.(x), ", ") * "]"
    in2 = ind * "  "
    return "{\n" * join([in2 * "\"$k\": " * js(v, in2) for (k, v) in sort(collect(x); by = first)], ",\n") *
           "\n" * ind * "}"
end
summary = Dict("node" => "R03",
               "attribution" => "EMIS_BLP residuals are libm round-off (Julia native exp/cbrt/pow vs glibc): j and alpha carry exp(-x^(1/3)), whose relative error is ~x^(1/3) eps, and max_rel_over_eps_1_plus_cbrt_x stays O(1); rho_Q carries the cancellation in f_m = va - vb - vc", "tolerance" => TOL, "tiny" => TINY, "n_cells" => n,
               "grid" => "nu {86,230,345} GHz x {0.3,1,3}; n_e 1e5; B {1,10,100} G; theta_e 10^[-2.5,2.5] (41); theta_B pi k/38, k=1..37",
               "emis_blp_vs_blp" => blp, "emis_blp_pass" => pass,
               "emis_native_over_blp" => native, "constants" => consts)
open(joinpath(outdir, "summary.json"), "w") do io
    println(io, js(summary))
end

# compact per-(nu, B, theta_e) table
keyf(i) = (nu[i], B[i], te[i])
groups = Dict{NTuple{3,Float64},Vector{Int}}()
for i in 1:n
    push!(get!(groups, keyf(i), Int[]), i)
end
open(joinpath(outdir, "coeff_table.csv"), "w") do io
    println(io, "nu_Hz,B_G,theta_e,x_at_thetaB13," *
                join(["maxrel_blp_" * s for s in NAMES], ",") * "," *
                join(["native_over_blp_" * s for s in NAMES], ","))
    for k in sort(collect(keys(groups)))
        ids = groups[k]
        i13 = ids[13]                                   # theta_B = 13 pi / 38
        mr = [maximum(i -> abs(Bl[i, c]) > TINY ? abs(R[i, c] - Bl[i, c]) / abs(Bl[i, c]) : 0.0, ids)
              for c in 1:8]
        nr = [abs(Bl[i13, c]) > TINY ? N[i13, c] / Bl[i13, c] : NaN for c in 1:8]
        @printf(io, "%.6g,%.6g,%.6g,%.6g", k[1], k[2], k[3], Xs[i13])
        for v in mr; @printf(io, ",%.3e", v); end
        for v in nr; @printf(io, ",%.6g", v); end
        println(io)
    end
end

for c in 1:8
    b = blp[NAMES[c]]; nt = native[NAMES[c]]
    @printf("%-3s EMIS_BLP max rel %.3e (n=%d, zero-mismatch %d, sign-mismatch %d) | native/BLP median %.4f [p05 %.4f, p95 %.4f] outside[0.9,1.1] %.3f\n",
            NAMES[c], b["max_rel_diff"], b["n_compared"], b["n_zero_mismatch"],
            b["n_sign_mismatch"], nt["median"], nt["p05"], nt["p95"], nt["frac_outside_0.9_1.1"])
end
@printf("L_unit raptor/blp - 1 = %.4e ; pixel solid angle ratio - 1 = %.4e\n",
        consts["L_unit_rel_change"], consts["pixel_solid_angle_rel_change"])
println(pass ? "R03 EMIS_BLP PASS" : "R03 EMIS_BLP FAIL")
exit(pass ? 0 : 1)
