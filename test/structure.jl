# The mechanical gates of the design: layering, one kernel, no private reach across
# modules, one home per concept, the size of the public API, documented exports, file
# budgets and the absence of history prose. Every number here is one a reviewer can read
# off the tree; the test fails when the tree drifts from the standard.
using Test
using BlackLightPlus

const SRC = normpath(joinpath(@__DIR__, "..", "src"))
const EXT = normpath(joinpath(@__DIR__, "..", "ext"))
const TESTDIR = @__DIR__

const MODULE_ORDER = ["Numerics", "Spacetime", "Matter", "Transfer", "MonteCarlo",
                      "Data", "Execution", "Plans", "Analysis", "Products"]
const MODULE_DIRS = Dict(
    "Numerics" => "physics/Numerics", "Spacetime" => "physics/Spacetime",
    "Matter" => "physics/Matter", "Transfer" => "physics/Transfer",
    "MonteCarlo" => "physics/MonteCarlo", "Data" => "infra/Data",
    "Execution" => "infra/Execution", "Plans" => "pipeline/Plans",
    "Analysis" => "pipeline/Analysis", "Products" => "pipeline/Products")
const PHYSICS = MODULE_ORDER[1:5]
const PIPELINE = MODULE_ORDER[8:10]

julia_files(dir) = [joinpath(r, f) for (r, _, fs) in walkdir(dir) for f in fs if endswith(f, ".jl")]
src_files() = julia_files(SRC)
module_files(m) = julia_files(joinpath(SRC, MODULE_DIRS[m]))
readlines_of(f) = readlines(f)

"Lines of `f` matching `re`, as `(lineno, line)`."
matches(f, re) = [(i, l) for (i, l) in enumerate(readlines_of(f)) if occursin(re, l)]

"The line without its backquoted spans and its trailing comment: what is code, not prose."
code_only(l) = replace(replace(l, r"`[^`]*`" => ""), r"#.*$" => "")

"Every `(file, lineno, line)` under `files` matching `re`."
function grep(files, re)
    out = Tuple{String,Int,String}[]
    for f in files, (i, l) in matches(f, re)
        push!(out, (relpath(f, SRC), i, l))
    end
    return out
end

@testset "structure" begin
    @testset "layering: a module imports only itself, its tier and the modules before it" begin
        for (k, m) in enumerate(MODULE_ORDER)
            allowed = Set(MODULE_ORDER[1:k-1])
            for f in module_files(m), (i, l) in matches(f, r"^\s*(using|import)\s+\.\.")
                for dep in eachmatch(r"\.\.([A-Z][A-Za-z]*)", l)
                    @test (dep[1] in allowed) || error("$(relpath(f, SRC)):$i imports $(dep[1]) from $m")
                end
            end
        end
        for m in PHYSICS, f in module_files(m)
            @test isempty(matches(f, r"^\s*(using|import)\s+(KernelAbstractions|HDF5|CUDA|Base\.Threads)\b"))
            @test isempty(matches(f, r"Threads\.@threads"))
        end
        for m in PIPELINE, f in module_files(m)
            @test isempty(matches(f, r"Threads\.@threads"))
        end
    end

    @testset "one kernel, no device files, no include guards, no CUDA in src" begin
        @test length(grep(src_files(), r"@kernel\b")) == 1
        @test isempty(filter(f -> startswith(basename(f), "gpu_"), src_files()))
        @test isempty(grep(src_files(), r"isdefined\(@__MODULE__"))
        @test isempty(grep(src_files(), r"^\s*(using|import)\s+CUDA\b"))
        @test isempty(filter(t -> occursin(r"\bCUDA\.[a-z]", code_only(t[3])), grep(src_files(), r"\bCUDA\.[a-z]")))
        @test !isempty(grep(julia_files(EXT), r"^\s*using .*CUDA"))
    end

    @testset "eponymous files own includes and exports" begin
        for m in MODULE_ORDER
            own = joinpath(SRC, MODULE_DIRS[m], m * ".jl")
            for f in module_files(m)
                f == own && continue
                @test isempty(matches(f, r"^\s*include\(")) || error("$(relpath(f, SRC)) includes")
                @test isempty(matches(f, r"^\s*export\s")) || error("$(relpath(f, SRC)) exports")
            end
        end
    end

    @testset "private names stay private" begin
        mods = join(MODULE_ORDER, "|")
        @test isempty(grep(src_files(), Regex("\\b($mods)\\._[A-Za-z]")))
        @test isempty(grep(src_files(), r"^\s*(using|import)\s+\.\.[A-Za-z]+:.*\b_[a-z]"))
        tests = filter(f -> !occursin("oracle", f), julia_files(TESTDIR))
        @test isempty(grep(tests, r"BlackLightPlus\.[A-Za-z]+\._[a-z]"))
        oracle = joinpath(TESTDIR, "oracle")                 # the reference comparison, kept on the blp branch
        isdir(oracle) && @test isempty(grep(julia_files(oracle), r"BlackLightPlus\.[A-Za-z]+\._[a-z]"))
    end

    @testset "one concept, one home" begin
        @test length(unique(first.(grep(src_files(), r"c100 - c000")))) == 1   # the trilinear blend
        @test length(grep(src_files(), r"^@inline function vacuum\(|^function vacuum\(")) == 1
        @test isempty(grep(src_files(), r"SVector\(y\[1\], y\[2\], y\[3\], y\[4\]\)") |>
                      v -> filter(t -> !endswith(t[1], "raystate.jl"), v))
        horizon = grep(src_files(), r"1 \+ sqrt\(1 - a")
        @test filter(t -> !occursin(r"kerr_orbits\.jl|analytic_kerr\.jl|kerr_schild\.jl", t[1]), horizon) == []
        @test isempty(grep(julia_files(joinpath(SRC, "pipeline")), r"sqrt\(1 - a|sqrt\(1\.0 - a"))
        @test isempty(grep(julia_files(joinpath(SRC, "pipeline")), r"1\.1 \* ") |>
                      v -> filter(t -> !endswith(t[1], "config.jl"), v))
        camera_sites = grep(julia_files(joinpath(SRC, "pipeline")), r"PlaneParallelCamera\(")
        @test length(camera_sites) <= 1
        @test isempty(grep(src_files(), r"keep in sync|twin of|byte-for-byte|ported from"i))
    end

    @testset "threads only in the Data loaders and builders" begin
        threaded = unique(first.(grep(src_files(), r"Threads\.@threads")))
        @test all(f -> startswith(f, "infra/Data/"), threaded)
        @test length(threaded) <= 4
    end

    @testset "public API" begin
        exported = setdiff(names(BlackLightPlus), [:BlackLightPlus])
        @test length(exported) <= 54
        for n in exported
            @test isdefined(BlackLightPlus, n) || error("BlackLightPlus exports undefined $n")
        end
        family = filter(n -> occursin(r"^(render|trace|shadow|mc_spectrum|sensitivity|variants|photon_ring_layers)", String(n)), exported)
        @test length(family) <= 10
        for m in MODULE_ORDER
            mod = getfield(BlackLightPlus, Symbol(m))
            for n in setdiff(names(mod), [Symbol(m)])
                @test isdefined(mod, n) || error("$m exports undefined $n")
                @test Base.Docs.hasdoc(mod, n) || error("$m.$n has no docstring")
            end
        end
    end

    @testset "file budgets" begin
        for f in src_files()
            @test length(readlines_of(f)) <= 600 || error("$(relpath(f, SRC)) exceeds 600 lines")
        end
        stray = filter(f -> startswith(basename(f), "test_"), readdir(TESTDIR))
        @test isempty(stray)
    end

    @testset "no attribution or process prose" begin
        files = vcat(src_files(), julia_files(EXT),
                     filter(f -> basename(f) != "structure.jl", julia_files(TESTDIR)),
                     [joinpath(@__DIR__, "..", f) for f in ("README.md", "Project.toml")])
        @test isempty(grep(files, r"Claude|Anthropic|Co-Authored|AI assist|generated by"i))
    end
end
