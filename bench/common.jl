# common.jl — helpers of bench.jl and reference.jl, included into Main by both:
#   flat JSON records, one object per line (`json`, `parse_json`, `read_records`,
#   `append_record`); `envlist`; output arrays saved as raw Float64 files or compared
#   bitwise with the saved ones (`save_or_compare`); the host state (`load1`,
#   `cpu_sample`/`other_cores`, `smi_state`); and `cuda_phases(f)`, the device phases of
#   one call of `f` from a CUPTI trace by the method of bench/bench_multigpu.jl. The phase
#   functions need CUDA loaded in Main.

envlist(k, d) = [parse(Int, s) for s in split(get(ENV, k, d), ',') if !isempty(strip(s))]

# ---------------------------------------------------------------- JSON (flat records)
json(io, v::AbstractString) = print(io, '"', escape_string(v), '"')
json(io, v::Symbol) = json(io, String(v))
json(io, v::Bool) = print(io, v ? "true" : "false")
json(io, v::Integer) = print(io, v)
json(io, v::AbstractFloat) = isfinite(v) ? print(io, repr(Float64(v))) : print(io, "null")
json(io, ::Nothing) = print(io, "null")
function json(io, v::Union{AbstractVector,Tuple})
    print(io, '[')
    for (i, x) in enumerate(v)
        i > 1 && print(io, ", ")
        json(io, x)
    end
    print(io, ']')
end
function json(io, d::AbstractDict)
    print(io, '{')
    for (i, k) in enumerate(sort!(collect(keys(d)); by = string))
        i > 1 && print(io, ", ")
        json(io, string(k)); print(io, ": "); json(io, d[k])
    end
    print(io, '}')
end
json(io, nt::NamedTuple) = json(io, Dict(pairs(nt)))

# A JSON reader for the records written above: objects, arrays, strings, numbers,
# true, false, null.
function parse_json(s::AbstractString)
    v, i = _pj(s, _ws(s, firstindex(s)))
    return v
end
_ws(s, i) = (while i <= lastindex(s) && isspace(s[i]); i = nextind(s, i); end; i)
function _pj(s, i)
    c = s[i]
    if c == '{'
        d = Dict{String,Any}(); i = _ws(s, i + 1)
        s[i] == '}' && return d, i + 1
        while true
            k, i = _pj(s, _ws(s, i)); i = _ws(s, i)
            s[i] == ':' || error("JSON: expected ':' at $i")
            v, i = _pj(s, _ws(s, i + 1)); d[k] = v; i = _ws(s, i)
            s[i] == '}' && return d, i + 1
            s[i] == ',' || error("JSON: expected ',' at $i"); i += 1
        end
    elseif c == '['
        a = Any[]; i = _ws(s, i + 1)
        s[i] == ']' && return a, i + 1
        while true
            v, i = _pj(s, _ws(s, i)); push!(a, v); i = _ws(s, i)
            s[i] == ']' && return a, i + 1
            s[i] == ',' || error("JSON: expected ',' at $i"); i += 1
        end
    elseif c == '"'
        j = i + 1; buf = IOBuffer()
        while s[j] != '"'
            if s[j] == '\\'
                e = s[j + 1]
                if e == 'u'
                    print(buf, Char(parse(UInt16, s[j+2:j+5]; base = 16))); j += 6
                else
                    print(buf, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e); j += 2
                end
            else
                print(buf, s[j]); j = nextind(s, j)
            end
        end
        return String(take!(buf)), j + 1
    elseif startswith(SubString(s, i), "true")
        return true, i + 4
    elseif startswith(SubString(s, i), "false")
        return false, i + 5
    elseif startswith(SubString(s, i), "null")
        return nothing, i + 4
    else
        m = match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?", SubString(s, i))
        m === nothing && error("JSON: unexpected character at $i")
        v = m[1] === nothing && m[2] === nothing ? parse(Int, m.match) : parse(Float64, m.match)
        return v, i + ncodeunits(m.match)
    end
end

read_records(path) = isfile(path) ?
    [parse_json(l) for l in eachline(path) if !isempty(strip(l))] : Dict{String,Any}[]

function append_record(path::AbstractString, rec::AbstractDict)
    mkpath(dirname(path))
    line = sprint(json, rec)
    open(path, "a") do io
        println(io, line)
    end
    return rec
end

# ---------------------------------------------------------------- saved outputs
"""
    save_or_compare(dir, key, tag, fields) -> Union{Bool,Nothing}

Write each `name => array` of `fields` to `dir/<key>.<tag>.<name>.<n1>x<n2>.f64` (raw
Float64, column-major), or, where that file exists, compare it bitwise. `nothing` when
nothing was there before, else whether every existing file matched.
"""
function save_or_compare(dir, key, tag, fields)
    mkpath(dir)
    same = true; any_old = false
    for (name, a) in fields
        f = joinpath(dir, "$key.$tag.$name.$(join(size(a), 'x')).f64")
        v = Vector{Float64}(vec(a))
        if isfile(f)
            any_old = true
            old = reinterpret(Float64, read(f))
            same &= length(old) == length(v) &&
                    reinterpret(UInt64, old) == reinterpret(UInt64, v)
        else
            write(f, v)
        end
    end
    return any_old ? same : nothing
end

# ---------------------------------------------------------------- host state
load1() = try parse(Float64, split(read("/proc/loadavg", String))[1]) catch; NaN end

# (busy jiffies of the host, CPU jiffies of this process, time): the cores that other
# processes kept busy between two samples a, b are `other_cores(a, b)`.
function cpu_sample()
    v = parse.(Int, split(readline("/proc/stat"))[2:end])
    s = split(read("/proc/self/stat", String))
    return (v[1] + v[2] + v[3] + v[6] + v[7] + v[8], parse(Int, s[14]) + parse(Int, s[15]), time())
end
const CLK_TCK = try parse(Int, readchomp(`getconf CLK_TCK`)) catch; 100 end
other_cores(a, b) = ((b[1] - a[1]) - (b[2] - a[2])) / (CLK_TCK * (b[3] - a[3]))

# nvidia-smi rows `index, memory.used [MiB], utilization.gpu [%], clocks.sm [MHz],
# temperature [C]` and the compute processes `pid, used memory [MiB]` of every visible
# device; empty where nvidia-smi does not answer.
function smi_state()
    q(args) = try
        filter(!isempty, strip.(readlines(pipeline(`nvidia-smi $args`; stderr = devnull))))
    catch
        String[]
    end
    return Dict{String,Any}(
        "smi_gpus" => q(`--query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader,nounits`),
        "smi_apps" => q(`--query-compute-apps=pid,used_memory --format=csv,noheader,nounits`))
end

# ---------------------------------------------------------------- CUPTI phases
# CUPTI.jl's activity-buffer callbacks push to shared vectors without a lock, and the pool
# lanes issue CUDA calls from several threads; the same bodies, serialized, as in
# bench/bench_multigpu.jl (which installs them itself when it is loaded).
function install_cupti_lock!()
    isdefined(Main, :_CUPTI_LOCK) && return
    lk = Threads.SpinLock()
    Core.eval(Main, :(const _CUPTI_LOCK = $lk))
    Core.eval(getfield(Main, :CUDA).CUPTI, quote
        function request_buffer(dest_ptr, sz_ptr, max_num_records_ptr)
            dest = Base.unsafe_wrap(Array, dest_ptr, 1)
            sz = Base.unsafe_wrap(Array, sz_ptr, 1)
            max_num_records = Base.unsafe_wrap(Array, max_num_records_ptr, 1)
            Base.@lock $lk begin
                cfg = activity_config[]
                if cfg !== nothing
                    buf = isempty(cfg.available_buffers) ? allocate_buffer() :
                          pop!(cfg.available_buffers)
                    push!(cfg.active_buffers, buf)
                    sizehint!(cfg.results, length(cfg.active_buffers))
                    dest[] = pointer(buf); sz[] = sizeof(buf)
                else
                    dest[] = C_NULL; sz[] = 0
                end
                max_num_records[] = 0
            end
            return
        end
        function complete_buffer(ctx_handle, stream_id, buf_ptr, sz, valid_sz)
            Base.@lock $lk begin
                cfg = activity_config[]
                cfg === nothing || push!(cfg.results, (ctx_handle, stream_id, buf_ptr, sz, valid_sz))
            end
            return
        end
    end)
    return
end

# Union length of [start, stop] intervals.
function phase_union(iv)
    isempty(iv) && return 0.0
    iv = sort(iv; by = first); tot = 0.0; s, e = iv[1]
    for (a, b) in iv[2:end]
        if a > e
            tot += e - s; s, e = a, b
        else
            e = max(e, b)
        end
    end
    return tot + (e - s)
end

# Activity class of a CUPTI device-trace row name.
function phase_class(name)
    m = match(r"^\[copy (\S+) to (\S+) memory\]", name)
    dev(k) = occursin("device", lowercase(k))
    m === nothing || return dev(m[1]) ? (dev(m[2]) ? :d2d : :d2h) : (dev(m[2]) ? :h2d : :h2h)
    return startswith(name, "[") ? :other : :kernel
end

cupti_clock() = (t = Ref{UInt64}(0); getfield(Main, :CUDA).CUPTI.cuptiGetTimestamp(t); t[] / 1e9)

"""
    cuda_phases(f) -> Dict

One call of `f` under the CUPTI profiler: `window_s` (the call), `host_prologue_s` (call
start to the first device activity), `host_epilogue_s` (last device activity to the call's
end) and per device `d<k>_kernel_s`, `d<k>_h2d_s`, `d<k>_d2h_s` (unions of the intervals,
clipped to the call), `d<k>_idle_s` (the window less the union of all activity),
`d<k>_launches` (kernel rows), and the kernel of the largest summed time
(`d<k>_top_kernel`, its time, launches, registers per thread, local memory per thread,
threads per block and blocks per launch).
"""
function cuda_phases(f)
    install_cupti_lock!()
    t0 = Ref(0.0); t1 = Ref(0.0)
    profile = Core.eval(Main, :(g -> CUDA.@profile g()))
    prof = Base.invokelatest(profile, () -> (t0[] = cupti_clock(); f(); t1[] = cupti_clock()))
    d = prof.device
    rows = [(start = max(d.start[i], t0[]), stop = min(d.stop[i], t1[]), dev = d.device[i],
             cls = phase_class(d.name[i]), i = i)
            for i in eachindex(d.start) if d.stop[i] > t0[] && d.start[i] < t1[]]
    out = Dict{String,Any}("window_s" => t1[] - t0[])
    isempty(rows) && return out
    out["host_prologue_s"] = minimum(r.start for r in rows) - t0[]
    out["host_epilogue_s"] = t1[] - maximum(r.stop for r in rows)
    for dev in sort!(unique(r.dev for r in rows))
        mine = filter(r -> r.dev == dev, rows)
        iv(c) = [(r.start, r.stop) for r in mine if r.cls === c]
        for (ph, c) in (("kernel", :kernel), ("h2d", :h2d), ("d2h", :d2h))
            out["d$(dev)_$(ph)_s"] = phase_union(iv(c))
        end
        out["d$(dev)_launches"] = count(r -> r.cls === :kernel, mine)
        out["d$(dev)_idle_s"] = (t1[] - t0[]) - phase_union([(r.start, r.stop) for r in mine])
        # the kernel with the largest summed time, and its launch resources
        ks = filter(r -> r.cls === :kernel, mine)
        isempty(ks) && continue
        tot = Dict{String,Float64}()
        for r in ks
            tot[d.name[r.i]] = get(tot, d.name[r.i], 0.0) + (r.stop - r.start)
        end
        top = argmax(tot)
        i = first(r.i for r in ks if d.name[r.i] == top)
        val(col) = (hasproperty(d, col) && !ismissing(getproperty(d, col)[i])) ? getproperty(d, col)[i] : nothing
        out["d$(dev)_top_kernel"] = first(top, 120)
        out["d$(dev)_top_kernel_s"] = tot[top]
        out["d$(dev)_top_kernel_launches"] = count(r -> d.name[r.i] == top, ks)
        reg = val(:registers); reg === nothing || (out["d$(dev)_top_kernel_registers"] = reg)
        lm = val(:local_mem); lm === nothing || (out["d$(dev)_top_kernel_local_bytes"] = lm.thread)
        bl = val(:block); bl === nothing || (out["d$(dev)_top_kernel_block"] = Int(bl.x) * Int(bl.y) * Int(bl.z))
        gr = val(:grid); gr === nothing || (out["d$(dev)_top_kernel_grid"] = Int(gr.x) * Int(gr.y) * Int(gr.z))
    end
    return out
end
