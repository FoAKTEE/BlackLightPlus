# Shared setup of the oracle scripts: both packages, the old source tree, and loaders
# that evaluate single definitions of the old device files (which `using CUDA` at the top
# and cannot be included without it) into a scratch module.
using Test, Random, StaticArrays, LinearAlgebra
import BLP, BlackLightPlus

const NS = BLP.Spacetime                 # new Spacetime
const NN = BLP.Numerics                  # new Numerics
const OG = BlackLightPlus.Geodesics      # old Geodesics
const OR = BlackLightPlus.Radiation
const OM = BlackLightPlus.MonteCarlo
const ForwardDiff = NS.ForwardDiff
const OLD_SRC = normpath(joinpath(@__DIR__, "..", "..", "..", "src"))

const KA = Base.require(Base.PkgId(Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"),
                                   "KernelAbstractions"))

# Name defined by a top-level expression (function, short function, struct, const).
function defname(ex)
    ex isa Expr || return nothing
    if ex.head === :macrocall
        return defname(ex.args[end])
    elseif ex.head in (:function, :(=))
        sig = ex.args[1]
        while sig isa Expr && sig.head in (:where, :(::))
            sig = sig.args[1]
        end
        sig isa Expr && sig.head === :call && return sig.args[1]
        return sig isa Symbol && ex.head === :(=) ? sig : nothing
    elseif ex.head === :struct
        n = ex.args[2]
        n isa Expr && n.head === :(<:) && (n = n.args[1])
        n isa Expr && n.head === :curly && (n = n.args[1])
        return n
    elseif ex.head === :const
        return defname(ex.args[1])
    end
    return nothing
end

"Evaluate the top-level definitions of `names` from the old file `relpath` into `mod`."
function load_old!(mod::Module, relpath::AbstractString, names)
    src = read(joinpath(OLD_SRC, relpath), String)
    top = Meta.parseall(src)
    found = Set{Symbol}()
    for ex in top.args
        n = defname(ex)
        if n in names
            Core.eval(mod, ex)
            push!(found, n)
        end
    end
    missing_ = setdiff(Set(names), found)
    isempty(missing_) || error("old definitions not found in $relpath: $missing_")
    return mod
end

"A scratch module with StaticArrays, KernelAbstractions and both packages bound."
function scratch_module(name::Symbol)
    m = Module(name)
    Core.eval(m, :(using StaticArrays))
    Core.eval(m, :(const KernelAbstractions = $KA))
    Core.eval(m, :(using .KernelAbstractions))
    Core.eval(m, :(import BlackLightPlus))
    return m
end

# Bitwise equality of isbits values, tuples and NamedTuples of them.
same(a, b) = a === b
same(a::Tuple, b::Tuple) = length(a) == length(b) && all(same(x, y) for (x, y) in zip(a, b))

# Random spatial point at Euclidean radius in [rlo, rhi].
function randpoint(rng, rlo, rhi)
    v = randn(rng, SVector{3,Float64})
    v = v / norm(v)
    return (rlo + (rhi - rlo) * rand(rng)) * v
end

# Random ray states around the centres `ctrs`: half with a null covector (p_t from the
# null root, normalised to p_t = 1), half with a random covector.
function random_states(rng, st_old, ctrs, n; rlo = 1.2, rhi = 40.0)
    ys = SVector{8,Float64}[]
    while length(ys) < n
        c = ctrs[rand(rng, 1:length(ctrs))]
        s = c + randpoint(rng, rlo, rhi)
        x = SVector(10 * randn(rng), s[1], s[2], s[3])
        ps = randn(rng, SVector{3,Float64})
        if isodd(length(ys))
            # a negative discriminant throws a DomainError (real sqrt) in both packages
            p, ok = try
                OG.null_covariant(st_old, x, ps)
            catch err
                err isa DomainError || rethrow()
                (zero(SVector{4,Float64}), false)
            end
            ok || continue
            p = p / p[1]
        else
            p = SVector(1.0 + 0.2randn(rng), ps[1], ps[2], ps[3])
        end
        push!(ys, vcat(x, p))
    end
    return ys
end

# ------------------------------------------------------------------ old device files
# The old device files `using CUDA` (and KernelAbstractions, Adapt, ForwardDiff) at the
# top. Evaluated into a scratch module, those become `using .CUDA` against module bindings
# made by identity (the oracle environment does not list these packages), and nested
# includes get the same rewrite.
const REBOUND = (:CUDA, :KernelAbstractions, :Adapt, :ForwardDiff)
pkg(uuid, name) = Base.require(Base.PkgId(Base.UUID(uuid), name))

rebind_path(e) = e isa Expr && e.head === :(:) ? Expr(:(:), rebind_path(e.args[1]), e.args[2:end]...) :
                 e isa Expr && e.head === :. && !isempty(e.args) && e.args[1] in REBOUND ?
                 Expr(:., :., e.args...) : e
function rebind(ex)
    ex isa Expr || return ex
    ex.head in (:using, :import) && return Expr(ex.head, map(rebind_path, ex.args)...)
    if ex.head === :call && (ex.args[1] === :include || ex.args[1] == :(Base.include))
        args = ex.args[1] === :include ?
               (Expr(:macrocall, Symbol("@__MODULE__"), LineNumberNode(0)), ex.args[2:end]...) :
               ex.args[2:end]
        return Expr(:call, GlobalRef(Base, :include), rebind, args...)
    end
    return Expr(ex.head, map(rebind, ex.args)...)
end

"""
The module `name` with CUDA, KernelAbstractions, Adapt and ForwardDiff bound by identity,
StaticArrays, LinearAlgebra and BlackLightPlus in scope, and the old device files `files`
(paths under the old `src`) included into it in order.
"""
function old_device_module(name::Symbol, files)
    m = Module(name)
    Core.eval(m, :(const CUDA = $(pkg("052768ef-5323-5732-b1bb-66c8b64840ba", "CUDA"))))
    Core.eval(m, :(const KernelAbstractions = $KA))
    Core.eval(m, :(const Adapt = $(pkg("79e6a3ab-5dfb-504d-930d-738a2a938a0e", "Adapt"))))
    Core.eval(m, :(const ForwardDiff = $ForwardDiff))
    Core.eval(m, :(using StaticArrays, LinearAlgebra, BlackLightPlus))
    for f in files
        Base.include(rebind, m, joinpath(OLD_SRC, f))
    end
    return m
end

# ------------------------------------------------------------------ the torus deck, old package
# Both need the module `Decks` (`test/decks.jl`) loaded in the calling script.

"The block deck of `Decks.amr_deck(; nc, half)` as an old `AthenaKSnapshot` (reader layout)."
function old_amr(; nc = 6, half = 12.0)
    w = half; dx = w / nc; nmb = 8
    data = zeros(Float32, nc, nc, nc, nmb, 8)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; xl = SVector{3,Float64}[]
    dxs = SVector{3,Float64}[]; lloc = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(-half + bi * w, -half + bj * w, -half + bk * w)
        push!(lo, l); push!(hi, l .+ w); push!(xl, l .+ dx / 2); push!(dxs, SVector(dx, dx, dx))
        lloc[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            p = Decks.torus_primitives(xl[m][1] + (i - 1) * dx, xl[m][2] + (j - 1) * dx,
                                       xl[m][3] + (k - 1) * dx)
            for q in 1:8
                data[i, j, k, m, q] = p[q]
            end
        end
    end
    return BlackLightPlus.Simulation.AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                                                     collect(String, BLP.Data.PRIM_NAMES), nc, nc,
                                                     nc, nmb)
end

"The old GRMHD model on the torus deck with the plasma of `Decks.torus_params` and `R_em = 100`."
function old_model()
    st = OG.KerrSchildCartesian(Decks.TORUS_A)
    L = NN.GNEWT * Decks.TORUS_MBH / NN.CL^2
    u = BlackLightPlus.Simulation.units(Decks.TORUS_MBH, Decks.TORUS_DENS * L^3)
    cfg = OR.SimCoeffConfig(bh_a = Decks.TORUS_A, d_unit = u.dens_unit,
                            e_unit = u.dens_unit * NN.CL^2, b_unit = u.B_unit, plasma_mu = 0.5,
                            plasma_ne_ni = 1.0, plasma_use_p = 1, plasma_rat_low = 1.0,
                            plasma_rat_high = 10.0, plasma_gamma = 13.0 / 9.0,
                            plasma_gamma_i = 5.0 / 3.0, plasma_gamma_e = 4.0 / 3.0,
                            plasma_thermal_frac = 1.0, cut_sigma_max = 1.0, cut_theta_e_min = -1.0)
    return OM.GRMHDModel(st, old_amr(), cfg, 13.0 / 9.0, u.L_unit, OG.horizon_radius(st), 100.0)
end
