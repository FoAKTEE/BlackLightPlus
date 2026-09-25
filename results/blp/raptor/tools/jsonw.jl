# jsonw.jl — minimal JSON writer for the RAPTOR comparison evidence files.
using Printf

function _js(x, ind = "")
    x === nothing && return "null"
    x isa Bool && return x ? "true" : "false"
    x isa Integer && return string(x)
    x isa AbstractFloat && return isfinite(x) ? @sprintf("%.10g", x) : "null"
    x isa Symbol && return _js(string(x))
    x isa AbstractString && return "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
    x isa NamedTuple && return _js(Dict(string(k) => v for (k, v) in pairs(x)), ind)
    x isa Union{AbstractVector,Tuple} && return "[" * join([_js(v, ind) for v in x], ", ") * "]"
    in2 = ind * "  "
    return "{\n" * join([in2 * _js(string(k)) * ": " * _js(v, in2)
                         for (k, v) in sort(collect(x); by = p -> string(first(p)))], ",\n") *
           "\n" * ind * "}"
end

write_json(path, x) = open(io -> println(io, _js(x)), path, "w")
