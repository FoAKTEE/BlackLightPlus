# fig_cmp_pol (one-zone part): constant-coefficient polarized transfer in BlackLightPlus and
# the reference codes against the exact solution.
#
# Problems (results/blp/rt_compare/polarized/onezone/onezone_common.jl): the two analytic
# tests of Prather et al. (2023, Table 1) and four more (all coefficients, Faraday thick,
# and the two Prather tests moved inside the admissible Stokes cone). Every code solves
#   dS/dλ = j - K S, S(0) = 0, λ ∈ [0, 3],
# with K of Landi Degl'Innocenti (1985), on N uniform steps (Coport: DP5 at a tolerance).
#
# Codes: BlackLightPlus (`segment(...; project = false)`, the exact solver on the coefficients
# as given, composed by the camera-side recurrence of `march(::IQUV)`; `blp_production`: the
# production path, coefficients projected onto the physical cone and the accumulated
# vector kept admissible by `stokes_admissible`), Blacklight (upstream a7c7215 and the build
# with both coupling fixes; unsplit and Strang-split), ipole (split and full Stokes steps),
# RAPTOR (default step choice, RK4, implicit trapezoid), Coport (DP5), Gyoto (Omatrix and
# Pmatrix with and without the source guard). The drivers and their build scripts are in
# results/blp/rt_compare/polarized/onezone/; binaries in results/blp/rt_compare/build/.
#
# Writes figures/paper/out/cmp_pol/onezone/<problem>/<code>.npy ((N+1) × 5: λ, I, Q, U, V)
# at the nominal N = 300, exact.npy, conv_<code>.npy (rows: steps, max|ΔI|, max|ΔQ|,
# max|ΔU|, max|ΔV|, relative error) and summary.tsv.
#
# Run: julia --project=. -t 8 figures/paper/fig_cmp_pol_onezone.jl
using BlackLightPlus, StaticArrays, Printf
const TR = BlackLightPlus.Transfer

const ROOT = dirname(dirname(@__DIR__))
const RC = joinpath(ROOT, "results", "blp", "rt_compare")
const OZ = joinpath(RC, "polarized", "onezone")
const BUILD = joinpath(RC, "build")
const OUT = joinpath(ROOT, "figures", "paper", "out", "cmp_pol", "onezone")
include(joinpath(OZ, "onezone_common.jl"))
using .OneZone

const NOMINAL = 300
const NSWEEP = [3 * 2^k for k in 0:12]
const TOLS = [10.0^-k for k in 3:13]
const COPORT_MODEL = joinpath(BUILD, "coport", "model", "Analytical thick disk")
const COPORT_ENV = joinpath(BUILD, "coport_env")
const REFCODE = "/data/haiyangw/claude/Mahakala/ref-code"            # ipole, Coport, Gyoto
const BL_UPSTREAM = joinpath(ROOT, "ref-code", "c-white-blacklight")  # Blacklight a7c7215
const RAPTOR_UPSTREAM = joinpath(ROOT, "ref-code", "jordydavelaar-raptor")
const BL_FIXES = joinpath(RC, "prs", "repos", "blacklight-polarized")
const GYOTO_SRC = joinpath(REFCODE, "Gyoto")
const EIGEN = joinpath(BUILD, "third_party", "eigen", "eigen-3.4.0")   # Eigen 3.4.0 headers

# ---- source copies, environments and drivers, created when missing ------------------
function ensure_sources()
    b(x) = joinpath(BUILD, x)
    mkpath(BUILD)
    if !isdir(b("blacklight"))
        mkpath(b("blacklight"))
        run(pipeline(`git -C $BL_UPSTREAM archive a7c7215`, `tar -x -C $(b("blacklight"))`))
    end
    if !isdir(b("blacklight-fixed"))   # upstream with both fix branches applied
        mkpath(b("blacklight-fixed"))
        run(pipeline(`git -C $BL_FIXES archive fix/polarized-absorption-step`,
                     `tar -x -C $(b("blacklight-fixed"))`))
        run(pipeline(`git -C $BL_FIXES diff master fix/coupling-matrix-indices`,
                     `patch -p1 -d $(b("blacklight-fixed"))`))
    end
    if !isdir(b("raptor"))
        mkpath(b("raptor"))
        run(pipeline(`git -C $RAPTOR_UPSTREAM archive main`, `tar -x -C $(b("raptor"))`))
    end
    if !isdir(b("ipole"))
        mkpath(b("ipole"))
        for f in readdir(joinpath(REFCODE, "ipole"))
            (endswith(f, ".c") || endswith(f, ".h") || f == "makefile") &&
                cp(joinpath(REFCODE, "ipole", f), b(joinpath("ipole", f)))
        end
    end
    isdir(b("coport")) || cp(joinpath(REFCODE, "Coport"), b("coport"))
    isfile(joinpath(COPORT_ENV, "Project.toml")) ||
        run(`julia --project=$COPORT_ENV -e 'using Pkg; Pkg.add(["OrdinaryDiffEqLowOrderRK", "SciMLBase"])'`)
    isdir(EIGEN) || error("Eigen 3.4.0 headers expected at $EIGEN")
end

function ensure_builds()
    ensure_sources()
    b(x) = joinpath(BUILD, x)
    isfile(b("blacklight_onezone_stock")) ||
        run(`$(joinpath(OZ, "build_blacklight.sh")) $(b("blacklight")) $(b("blacklight_onezone_stock"))`)
    isfile(b("blacklight_onezone_fixed")) ||
        run(`$(joinpath(OZ, "build_blacklight.sh")) $(b("blacklight-fixed")) $(b("blacklight_onezone_fixed"))`)
    (isfile(b("ipole_onezone_split")) && isfile(b("ipole_onezone_full"))) ||
        run(`$(joinpath(OZ, "build_ipole.sh")) $(b("ipole")) $BUILD`)
    isfile(b("raptor_onezone")) ||
        run(`$(joinpath(OZ, "build_raptor.sh")) $(b("raptor")) $(b("raptor_onezone"))`)
    isfile(b("gyoto_onezone")) ||
        run(`$(joinpath(OZ, "build_gyoto.sh")) $GYOTO_SRC $EIGEN $(b("gyoto_onezone"))`)
end

parse_rows(txt) = begin
    rows = [parse.(Float64, split(l)) for l in split(strip(txt), '\n') if !startswith(l, "#")]
    permutedims(reduce(hcat, rows))
end

coefs(p) = string.(coeff_vector(p))

# ---- BlackLightPlus: the camera-side recurrence of march(::IQUV) with R = 1 -----------
function run_blp(p::Problem, N::Int; production::Bool = false)
    j = SVector(p.j); a = SVector(p.a); r = SVector(0.0, p.r[1], p.r[2], p.r[3])
    h = p.L / N
    M, b = TR.segment(TR.IQUV(), j, a, r, h; project = production)
    keep = production ? TR.stokes_admissible : identity
    out = zeros(N + 1, 5)
    out[:, 1] = (0:N) .* h
    # S(λ_n) is the image of n segments seen from a camera at λ_n
    Mpow = one(SMatrix{4,4,Float64})
    pows = Vector{SMatrix{4,4,Float64,16}}(undef, N)
    for m in 1:N
        pows[m] = Mpow
        Mpow = Mpow * M
    end
    Threads.@threads for n in 1:N
        S = SVector(0.0, 0.0, 0.0, 0.0)
        for m in 1:n
            S = keep(S + pows[m] * b)
        end
        out[n + 1, 2:5] = S
    end
    return out
end

# An external driver; a run that stops early (Gyoto's |O_ij| <= 1 check) is padded with NaN.
function ext(bin, N, args...)
    txt = read(pipeline(ignorestatus(Cmd([joinpath(BUILD, bin), string.(args)...])); stderr = devnull), String)
    rows = isempty(strip(txt)) ? zeros(0, 5) : parse_rows(txt)
    size(rows, 1) == N + 1 && return rows
    pad = fill(NaN, N + 1 - size(rows, 1), 5)
    return vcat(rows, pad)
end

const CODES = [
    ("blp", (p, N) -> run_blp(p, N)),
    ("blp_production", (p, N) -> run_blp(p, N; production = true)),
    ("blacklight", (p, N) -> ext("blacklight_onezone_stock", N, 0, p.L, N, coefs(p)...)),
    ("blacklight_split", (p, N) -> ext("blacklight_onezone_stock", N, 1, p.L, N, coefs(p)...)),
    ("blacklight_fixed", (p, N) -> ext("blacklight_onezone_fixed", N, 0, p.L, N, coefs(p)...)),
    ("blacklight_fixed_split", (p, N) -> ext("blacklight_onezone_fixed", N, 1, p.L, N, coefs(p)...)),
    ("ipole_split", (p, N) -> ext("ipole_onezone_split", N, p.L, N, coefs(p)...)),
    ("ipole_full", (p, N) -> ext("ipole_onezone_full", N, p.L, N, coefs(p)...)),
    ("raptor", (p, N) -> ext("raptor_onezone", N, 0, p.L, N, coefs(p)...)),
    ("raptor_rk4", (p, N) -> ext("raptor_onezone", N, 1, p.L, N, coefs(p)...)),
    ("raptor_ie", (p, N) -> ext("raptor_onezone", N, 2, p.L, N, coefs(p)...)),
    ("gyoto", (p, N) -> ext("gyoto_onezone", N, 1, p.L, N, coefs(p)...)),
    ("gyoto_noguard", (p, N) -> ext("gyoto_onezone", N, 0, p.L, N, coefs(p)...)),
]

# Coport: one Julia process per problem runs every tolerance; returns (steps, rows) per
# tolerance, in the order given
function run_coport(p::Problem, N::Int, tols)
    script = joinpath(OZ, "coport_onezone.jl")
    tl = join(string.(tols), ",")
    txt = read(pipeline(`julia --project=$COPORT_ENV $script $COPORT_MODEL $(p.L) $N $tl $(coefs(p))`;
                        stderr = devnull), String)
    blocks = Tuple{Int,Matrix{Float64}}[]
    cur = nothing; rows = String[]
    flush!() = cur === nothing || push!(blocks, (cur[2], parse_rows(join(rows, '\n'))))
    for l in split(txt, '\n')
        if startswith(l, "# tol")
            flush!(); f = split(l)
            cur = (parse(Float64, f[3]), parse(Int, f[5])); rows = String[]
        elseif !isempty(strip(l))
            push!(rows, l)
        end
    end
    flush!()
    return blocks
end

maxerr(S, E) = [maximum(abs.(S[:, 1 + k] .- E[:, k])) for k in 1:4]
relerr(S, E) = maximum(abs.(S[:, 2:5] .- E)) / maximum(abs.(E))

function main()
    ensure_builds()
    mkpath(OUT)
    summary = open(joinpath(OUT, "summary.tsv"), "w")
    println(summary, "problem\tcode\tsteps\tmax_dI\tmax_dQ\tmax_dU\tmax_dV\trel")
    for p in PROBLEMS
        d = joinpath(OUT, p.name); mkpath(d)
        lam = collect((0:NOMINAL) .* (p.L / NOMINAL))
        E = exact(p, lam)
        write_npy(joinpath(d, "exact.npy"), hcat(lam, E))
        exacts = Dict(N => exact(p, collect((0:N) .* (p.L / N))) for N in NSWEEP)
        for (name, f) in CODES
            S = f(p, NOMINAL)
            @assert size(S) == (NOMINAL + 1, 5) "$name on $(p.name): output $(size(S))"
            write_npy(joinpath(d, "$name.npy"), S)
            conv = zeros(length(NSWEEP), 6)
            for (i, N) in enumerate(NSWEEP)
                SN = f(p, N)
                conv[i, :] = [N; maxerr(SN, exacts[N]); relerr(SN, exacts[N])]
            end
            write_npy(joinpath(d, "conv_$name.npy"), conv)
            m = maxerr(S, E)
            @printf(summary, "%s\t%s\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", p.name, name, NOMINAL,
                    m..., relerr(S, E))
            @printf("%-10s %-24s rel %.3e\n", p.name, name, relerr(S, E))
        end
        # Coport: the nominal run at its own tolerance, and the tolerance sweep
        blocks = run_coport(p, NOMINAL, vcat(1e-7, TOLS))
        @assert length(blocks) == length(TOLS) + 1
        steps7, S = blocks[1]
        write_npy(joinpath(d, "coport.npy"), S)
        conv = zeros(length(TOLS), 6)
        for (i, tol) in enumerate(TOLS)
            st, St = blocks[i + 1]
            conv[i, :] = [st; maxerr(St, E); relerr(St, E)]
        end
        write_npy(joinpath(d, "conv_coport.npy"), conv)
        m = maxerr(S, E)
        @printf(summary, "%s\t%s\t%d\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", p.name, "coport", steps7,
                m..., relerr(S, E))
        @printf("%-10s %-24s rel %.3e (DP5, %d steps)\n", p.name, "coport", relerr(S, E), steps7)
    end
    close(summary)
    println("wrote ", OUT)
end

main()
