# bench_lines.jl — the line format of bench_multigpu.jl and its parser.
#
# One record per line, a keyword followed by `key=value` tokens (values carry no spaces,
# except the free-text `reason` of a SKIP, which runs to the end of the line):
#   BENCH leg=<leg> backend=<cuda|cpu> ndev=<n> items=<n> wall_s=<t> first_s=<t> rate=<r>
#         unit=<rays/s|photons/s> [speedup=<T1/Tn>] [equal=<true|false>] [variant=<v>]
#   PHASE leg=<leg> backend=<b> ndev=<n> device=<id|all> phase=<name> seconds=<t>
#   SKIP  leg=<leg> reason=<free text>
# Phases: host_prologue, host_epilogue (all devices), and per device kernel, h2d, d2h,
# idle. Integers parse as Int, other numbers as Float64, `equal` as Bool, and the name-like
# keys (leg, backend, unit, variant, phase, device, reason, commit) stay Strings.

const BENCH_KINDS = ("BENCH", "PHASE", "SKIP")

"""
    bench_line(kind, pairs...) -> String

Format one record, e.g. `bench_line("PHASE", :leg => "fused_384", :seconds => 0.25)`.
"""
function bench_line(kind::AbstractString, pairs::Pair...)
    kind in BENCH_KINDS || error("bench_line: unknown record kind $kind")
    toks = String[kind]
    for (k, v) in pairs
        s = v isa AbstractFloat ? string(round(v; sigdigits = 6)) : string(v)
        (k === :reason || !occursin(' ', s)) ||
            error("bench_line: value of $k contains a space: $(repr(s))")
        push!(toks, string(k, "=", s))
    end
    return join(toks, ' ')
end

_bench_value(k, s) = k === :equal ? parse(Bool, s) :
    (k in (:leg, :backend, :unit, :variant, :phase, :device, :reason, :commit) ? String(s) :
     something(tryparse(Int, s), tryparse(Float64, s), String(s)))

"""
    parse_bench_line(line) -> NamedTuple or nothing

Parse one record into `(kind = "BENCH", leg = ..., ...)`; `nothing` for any other line.
"""
function parse_bench_line(line::AbstractString)
    kind, rest = let m = match(r"^(BENCH|PHASE|SKIP)\s+(.*)$", strip(line))
        m === nothing && return nothing
        String(m[1]), m[2]
    end
    keys_ = Symbol[:kind]; vals = Any[kind]
    while !isempty(rest)
        m = match(r"^(\w+)=(\S*)\s*", rest)
        m === nothing && error("parse_bench_line: malformed token in $(repr(line))")
        k = Symbol(m[1])
        if k === :reason                         # free text to the end of the line
            push!(keys_, k); push!(vals, String(rest[length("reason=")+1:end]))
            break
        end
        push!(keys_, k); push!(vals, _bench_value(k, m[2]))
        rest = rest[length(m.match)+1:end]
    end
    return NamedTuple{Tuple(keys_)}(Tuple(vals))
end

"""
    parse_bench(io_or_text) -> Vector{NamedTuple}

Parse every BENCH / PHASE / SKIP record, ignoring other lines.
"""
parse_bench(io::IO) = [r for r in (parse_bench_line(l) for l in eachline(io)) if r !== nothing]
parse_bench(text::AbstractString) = parse_bench(IOBuffer(text))
