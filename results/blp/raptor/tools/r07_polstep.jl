# r07_polstep.jl — segment-level test of RAPTOR's polarized transfer step against the
# BlackLightPlus constant-coefficient operator (dag.md R07 leg (b), M11).
#
# For random invariant coefficient sets (j_I > 0, |j_P| <= j_I; α_I > 0, α_P <= α_I;
# ρ_Q, ρ_V of either sign; U components zero, as both codes' tetrads give) and random
# admissible input Stokes vectors, one segment of invariant length ds = dl * C spanning
# δτ = α_I ds ∈ [1e-4, 30] and Faraday depth ρ_V ds ∈ [1e-3, 300] is taken by
#   - RAPTOR solver 1 (pol_rte_ld85_step, M11), compared with BlackLightPlus
#     `stokes_couple` (stokes_segment + stokes_admissible); acceptance: max relative
#     difference ≤ 1e-10 of max(|S_B|) per case;
#   - RAPTOR solver 0 (the native check_stiffness -> RK4 / implicit trapezoid step),
#     reported: fraction of steps routed to the trapezoid, error against the exact
#     segment versus the Faraday depth per step, and the observed order in ds.
# C = C_CONST of the M = 1.3e10 Msun run (R_g h / (m_e c^2)), so the stiffness test sees
# the production scaling (it is evaluated on dl, the step on dl * C).
#
# Usage: julia --project=. r07_polstep.jl <libraptor_pol_step.so> <out.json> [ncase]
using BlackLightPlus, StaticArrays, Random, Printf, LinearAlgebra
const RAD = BlackLightPlus.Radiation
include(joinpath(@__DIR__, "jsonw.jl"))

const LIB = ARGS[1]
const OUT = ARGS[2]
const NCASE = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20000
const h = Libc.Libdl.dlopen(LIB)
const FSTEP = Libc.Libdl.dlsym(h, :raptor_pol_step)
const RG = 6.6743e-8 * 1.3e10 * 1.98847e33 / 2.99792458e10^2
const CC = RG * 6.6261e-27 / (9.1094e-28 * 2.99792458e10^2)

function rstep(solver, coef, dl, S)
    s = collect(Float64, S)
    stiff = ccall(FSTEP, Cint, (Cint, Ptr{Float64}, Cdouble, Cdouble, Ptr{Float64}),
                  solver, collect(Float64, coef), dl, CC, s)
    return SVector{4}(s), stiff
end

function case(rng)
    ds = 1.0
    aI = 10.0^(rand(rng) * log10(30 / 1e-4) + log10(1e-4))
    fa = rand(rng); ta = rand(rng) * 2pi
    aQ = aI * fa * cos(ta); aV = aI * fa * sin(ta)
    rV = (rand(rng) < 0.5 ? -1 : 1) * 10.0^(rand(rng) * 5.5 - 3)
    rQ = (rand(rng) < 0.5 ? -1 : 1) * abs(rV) * 10.0^(rand(rng) * 3 - 2)
    jI = aI * 10.0^(rand(rng) * 2 - 1)
    fj = rand(rng); tj = rand(rng) * 2pi
    jQ = jI * fj * cos(tj); jV = jI * fj * sin(tj)
    I0 = jI / aI * rand(rng) * 2
    f0 = rand(rng); u = normalize(randn(rng, 3))
    S0 = SVector(I0, I0 * f0 * u[1], I0 * f0 * u[2], I0 * f0 * u[3])
    return (jI, jQ, 0.0, jV, rQ, 0.0, rV, aI, aQ, 0.0, aV), ds, S0
end

rng = MersenneTwister(20260925)
worst = 0.0; nstiff = 0
nat = Dict{String,Vector{Float64}}()
bins = [(1e-3, 1e-1), (1e-1, 1.0), (1.0, 10.0), (10.0, 300.0)]
for k in 1:NCASE
    coef, ds, S0 = case(rng)
    jI, jQ, _, jV, rQ, _, rV, aI, aQ, _, aV = coef
    SB = RAD.stokes_couple(S0, SVector(jI, jQ, 0.0, jV), SVector(aI, aQ, 0.0, aV),
                           SVector(0.0, rQ, 0.0, rV), ds)
    dl = ds / CC
    S1, _ = rstep(1, coef, dl, S0)
    d = maximum(abs.(S1 .- SB)) / max(maximum(abs.(SB)), 1e-300)
    global worst = max(worst, d)
    S0n, st = rstep(0, coef, dl, S0)
    global nstiff += st
    rho = hypot(rQ, rV) * ds
    for (lo, hi) in bins
        if lo <= rho < hi
            push!(get!(nat, @sprintf("%g-%g", lo, hi), Float64[]),
                  maximum(abs.(S0n .- SB)) / max(maximum(abs.(SB)), 1e-300))
        end
    end
end
# observed order of the native step on a smooth rotation + absorption segment
coef = (1.0, 0.3, 0.0, 0.1, 0.5, 0.0, 2.0, 1.0, 0.2, 0.0, 0.1)
S0 = SVector(0.3, 0.1, -0.05, 0.02)
L = 1.0
errs = Float64[]; ns = [4, 8, 16, 32, 64]
for n in ns
    Sn = S0; Sl = S0
    for _ in 1:n
        Sn, _ = rstep(0, coef, (L / n) / CC, Sn)
    end
    jI, jQ, _, jV, rQ, _, rV, aI, aQ, _, aV = coef
    SB = RAD.stokes_couple(S0, SVector(jI, jQ, 0.0, jV), SVector(aI, aQ, 0.0, aV),
                           SVector(0.0, rQ, 0.0, rV), L)
    push!(errs, maximum(abs.(Sn .- SB)))
end
order = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
res = Dict("ncase" => NCASE, "C_CONST" => CC, "ld85_max_rel_diff" => worst,
           "ld85_pass" => worst <= 1e-10,
           "native_trapezoid_fraction" => nstiff / NCASE,
           "native_rel_err_by_faraday_depth_per_step" =>
               Dict(k => Dict("median" => sort(v)[cld(length(v), 2)], "max" => maximum(v), "n" => length(v))
                    for (k, v) in nat),
           "native_convergence" => Dict("nsteps" => ns, "max_abs_err" => errs, "observed_order" => order),
           "pass" => worst <= 1e-10)
write_json(OUT, res)
@printf("LD85 port vs stokes_couple: max rel diff %.3e over %d cases (%s)\n", worst, NCASE,
        worst <= 1e-10 ? "PASS" : "FAIL")
@printf("native step: trapezoid on %.4f of the cases; order %s\n", nstiff / NCASE, string(round.(order; digits = 2)))
for (k, v) in sort(collect(nat))
    @printf("  native rel err, Faraday depth per step %-10s median %.2e max %.2e (n %d)\n", k,
            sort(v)[cld(length(v), 2)], maximum(v), length(v))
end
exit(worst <= 1e-10 ? 0 : 1)
