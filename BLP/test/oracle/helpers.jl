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
