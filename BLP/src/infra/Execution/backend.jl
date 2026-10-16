# The CPU/GPU switch. A product resolves its `backend` keyword with `resolve`: an explicit
# value wins; `nothing` means the session default (`backend!`), else the environment
# variable `BLP_BACKEND` (read once, at first use), else `:auto`. The session record below
# and the pool registry are the package's only session state; both sit behind locks.

const BACKEND_GRAMMAR =
    "auto | cpu | gpu | gpu:<ordinal>[,<ordinal>...] | host_pool | host_pool:<count>"

mutable struct Session
    chosen::Union{Nothing,Executor}             # session default (backend!, env or auto)
    env_read::Bool
    env::Any                                    # parsed BLP_BACKEND, or nothing when unset
    auto::Union{Nothing,Executor}               # the :auto choice, logged when made
    gpu::Union{Nothing,Vector{CUDADevice}}      # the :gpu device selection
end
const SESSION = Session(nothing, false, nothing, nothing, nothing)
const SESSION_LOCK = ReentrantLock()

"""
    resolve(backend) -> Executor

The executor named by `backend`: `:cpu` → `ThreadedExecutor()`; `:gpu` → a
`PoolExecutor` over the idle CUDA devices (`BackendUnavailable` without the CUDA extension
or a functional device); `:host_pool` → a `PoolExecutor` over `HostDevice(1)`; `:auto` →
the GPU pool when CUDA is loaded, functional and a device is idle, else threads, logged
once per session with its reason; an `Executor` as given; a `DevicePool`, a device or a
device list (handles or CUDA ordinals) → a `PoolExecutor` over it; a string in the
`BLP_BACKEND` grammar (`$BACKEND_GRAMMAR`); `nothing` → [`backend`](@ref)`()`.
"""
resolve(::Nothing) = backend()
resolve(e::Executor) = e
resolve(p::DevicePool) = PoolExecutor(p)
resolve(d::Union{AbstractDevice,Integer}) = PoolExecutor([d])
resolve(devs::AbstractVector) =
    isempty(devs) ? throw(ArgumentError("backend: empty device list")) : PoolExecutor(devs)
resolve(s::AbstractString) = resolve(parse_backend(s))
function resolve(s::Symbol)
    s === :auto && return auto_executor()
    s === :cpu && return ThreadedExecutor()
    s === :gpu && return gpu_executor()
    s === :host_pool && return PoolExecutor([HostDevice(1)])
    throw(ArgumentError("unknown backend :$s; expected :auto, :cpu, :gpu, :host_pool, an " *
                        "Executor, a device list or a DevicePool"))
end

"""
    backend!(backend) -> Union{Executor,Nothing}

Set the session default executor to [`resolve`](@ref)`(backend)` (so an unusable choice
throws now) and return it; `backend!(nothing)` clears it, so the environment or `:auto`
decides again.
"""
function backend!(b)
    exec = b === nothing ? nothing : resolve(b)
    @lock SESSION_LOCK (SESSION.chosen = exec)
    return exec
end

"""
    backend() -> Executor

The resolved session default: the executor set by [`backend!`](@ref), else the one named
by `BLP_BACKEND` (read at the first call), else the `:auto` choice. Printing it shows the
kind, devices, lanes per device, chunk and Julia threads.
"""
function backend()
    @lock SESSION_LOCK begin
        SESSION.chosen === nothing || return SESSION.chosen
        spec = env_backend()
        SESSION.chosen = spec === nothing ? resolve(:auto) : resolve(spec)
    end
end

"""
    reset_backend!()

Forget the session default, the reading of `BLP_BACKEND`, the cached `:auto` and `:gpu`
choices and the one-time `:auto` log, as at the start of a session. Registered pools and
their residency caches are kept. A test hook.
"""
function reset_backend!()
    @lock SESSION_LOCK begin
        SESSION.chosen = nothing; SESSION.env_read = false; SESSION.env = nothing
        SESSION.auto = nothing; SESSION.gpu = nothing
    end
    return nothing
end

# BLP_BACKEND, parsed once; a malformed value throws and is read again at the next use.
function env_backend()
    if !SESSION.env_read
        v = strip(get(ENV, "BLP_BACKEND", ""))
        SESSION.env = isempty(v) ? nothing : parse_backend(v)
        SESSION.env_read = true
    end
    return SESSION.env
end

function parse_backend(s::AbstractString)
    t = replace(lowercase(strip(s)), " " => "")
    t in ("auto", "cpu", "gpu", "host_pool") && return Symbol(t)
    m = match(r"^gpu:(\d+(?:,\d+)*)$", t)
    if m !== nothing
        ords = tryparse.(Int, split(m[1], ','))
        all(!isnothing, ords) && allunique(ords) && return CUDADevice.(ords)
    end
    m = match(r"^host_pool:(\d+)$", t)
    k = m === nothing ? nothing : tryparse(Int, m[1])
    k !== nothing && k >= 1 && return [HostDevice(i) for i in 1:k]
    throw(ArgumentError("malformed backend \"$s\"; expected $BACKEND_GRAMMAR"))
end

function auto_executor()
    @lock SESSION_LOCK begin
        if SESSION.auto === nothing
            exec, reason = auto_choice()
            @info "BLP backend :auto chose $(exec): $(reason)"
            SESSION.auto = exec
        end
        SESSION.auto
    end
end

function auto_choice()
    cuda_loaded() || return (ThreadedExecutor(), "CUDA not loaded")
    functional(CUDADevice) || return (ThreadedExecutor(), "no functional device")
    devs = select_devices()
    isempty(devs) && return (ThreadedExecutor(), "no idle device")
    return (PoolExecutor(devs), "devices " * join(string.(devs), ", "))
end

function gpu_executor()
    @lock SESSION_LOCK begin
        if SESSION.gpu === nothing
            cuda_loaded() || throw(BackendUnavailable(
                "backend :gpu needs the CUDA extension: run `using CUDA` first, or " *
                "choose the CPU with `backend!(:cpu)`"))
            functional(CUDADevice) || throw(BackendUnavailable(
                "backend :gpu found no functional CUDA device: check the driver and " *
                "`CUDA_VISIBLE_DEVICES`, or choose the CPU with `backend!(:cpu)`"))
            devs = select_devices(; fallback = true)
            isempty(devs) && throw(BackendUnavailable(
                "backend :gpu found no visible CUDA device: check " *
                "`CUDA_VISIBLE_DEVICES`, or choose the CPU with `backend!(:cpu)`"))
            SESSION.gpu = devs
        end
        PoolExecutor(SESSION.gpu)
    end
end
