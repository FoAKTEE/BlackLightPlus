# Geodesics — gpu_tracer.jl  (OPTIONAL heavy dep: CUDA + KernelAbstractions)
# Multi-GPU batch geodesic tracer, output contract identical to
# `trace_rays_sampled` (integrator.jl): one ray per thread, fixed-control RK4 of
# the Hamiltonian covariant-momentum RHS, step h = (r - r_h)/C, termination by the
# shared `march_stop` rule (`_gimg_stop`, gpu_metric.jl: horizon backstop, redshift
# cut, outbound-only escape/R_em) — the CPU tracer's own predicate. ∂_c g^{ab} is
# ForwardDiff.Dual seeded in-kernel — same closed form as the CPU RHS, agreeing to
# fp roundoff (~1 ulp/RHS, ~1e-8 max pixel rel), not bit-for-bit. NOT included from
# geodesics.jl (CUDA optional). Step-density accuracy: NOTE on `trace_rays_sampled`.

using StaticArrays
using ForwardDiff: Dual, partials
using KernelAbstractions
using CUDA

# ---------------------------------------------------------------- device selection
# Identical guarded copy of gpu_image.jl :: select_devices (self-contained) so the tracer
# can auto-pick idle GPUs whether or not gpu_image.jl was co-included. First include wins.
if !@isdefined(select_devices)
"""
    select_devices(candidates=[0,1]; util_threshold=15, min_free_gb=8) -> Vector{Int}

Return the subset of VISIBLE CUDA devices (index < `CUDA.ndevices()`, correct under
`CUDA_VISIBLE_DEVICES` masking) in `candidates` that are idle enough (free ≥ `min_free_gb`
GB and, if NVML util is available, ≤ `util_threshold` %). Always returns ≥1 device; any
probe failure falls back to the static list.
"""
function select_devices(candidates = [0, 1]; util_threshold::Real = 15, min_free_gb::Real = 8)
    vis = try
        filter(d -> 0 <= d < CUDA.ndevices(), candidates)
    catch
        return collect(candidates)
    end
    isempty(vis) && return collect(candidates)
    idle = Int[]
    for d in vis
        ok = try
            CUDA.device!(d)
            free, _ = CUDA.memory_info()
            u = try
                Float64(CUDA.NVML.utilization_rates(CUDA.NVML.Device(CUDA.uuid(CUDA.CuDevice(d)))).compute) * 100.0
            catch
                -1.0
            end
            (free / 1024^3 >= min_free_gb) && (u < 0 || u <= util_threshold)
        catch
            false
        end
        ok && push!(idle, d)
    end
    return isempty(idle) ? vis[1:1] : idle
end
end # @isdefined select_devices

# ---------------------------------------------------------------- device kernels
# The KS inverse metric, 3-lane-AD Hamiltonian RHS and KS radius come from the one
# shared device-metric layer (../gpu_metric.jl); the local names are aliases of the
# shared bodies (see the gpu_metric.jl header).
isdefined(@__MODULE__, :_gimg_gi) || include(joinpath(@__DIR__, "..", "gpu_metric.jl"))
const _gi_cks = _gimg_gi
const _rhs_gpu = _gimg_rhs
const _r_ks = _gimg_rks

# One ray per thread. Marches until termination or nmax steps, logging every
# state (traj[i, 1] = y0, traj[i, s+1] = state after step s) and step (hlog[i, s]);
# cnt[i] = number of steps taken. Layout is ray-major (dim 1) so writes across
# threads at fixed step are coalesced. A terminated thread simply stops (divergent
# trip count) — matching the CPU `break`, hence the sample counts too.
# Everything runs at T = eltype(eltype(y0s)): the host passes every scalar
# (a, C, escape, R_em, r_h) as T (the `a::T` guards make a stray Float64 a
# MethodError), and eltype keeps the in-kernel Inf literal type-pure (identity when
# T == Float64); `march_stop` converts its constants to T.
@kernel function _trace_kernel!(traj, hlog, cnt, @Const(y0s), a, C, escape, R_em, nmax, r_h, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        y = y0s[i]
        traj[i, 1] = y
        n = 0
        r_prev = eltype(y)(Inf)
        for _ in 1:nmax
            stop, r = _gimg_stop(y, r_prev, r_h, escape, R_em, a)
            stop && break
            r_prev = r
            h = (r - r_h) / C
            # low-storage streaming RK4 (bit-identical ops to k1+2k2+2k3+k4)
            d = _rhs_gpu(y, a);  acc = d;         ys = y + (h / 2) * d
            d = _rhs_gpu(ys, a); acc = acc + 2*d; ys = y + (h / 2) * d
            d = _rhs_gpu(ys, a); acc = acc + 2*d; ys = y + h * d
            d = _rhs_gpu(ys, a); acc = acc + d
            y = y + (h / 6) * acc
            n += 1
            traj[i, n + 1] = y
            hlog[i, n] = h
        end
        cnt[i] = Int32(n)
    end
end

# ---------------------------------------------------------------- host driver
# Trace a contiguous ray range `rng` on the CURRENTLY active CUDA device, chunked.
# Buffers are allocated once and reused across chunks; only the used step-depth is
# copied back (max count, typically ≪ nmax). Fills states/dls in place, per the
# `trace_rays_sampled` contract (source→camera; dl[k] = step behind, dl[m] = 0).
# Device containers carry T = the ray element type (Float32 traj/hlog halve the dominant
# footprint); the DOWNLOAD converts back to Float64 (the states/dls element types), so
# the CPU radiation stage consumes exactly the same types. The caller passes the kernel
# scalars already converted to T.
function _trace_part!(states, dls, rays::Vector{SVector{8,T}}, rng, a, r_h, C, escape,
                      R_em, nmax, chunk) where {T}
    backend = CUDABackend()
    kern = _trace_kernel!(backend)
    pl = length(rng)
    width = min(chunk, pl)
    traj = CuArray{SVector{8,T}}(undef, width, nmax + 1)
    hlog = CuArray{T}(undef, width, nmax)
    cnt = CuArray{Int32}(undef, width)
    y0d = CuArray{SVector{8,T}}(undef, width)
    lo = first(rng)
    for cstart in 1:chunk:pl
        cur = min(cstart + chunk - 1, pl) - cstart + 1
        g0 = lo + cstart - 1
        copyto!(y0d, 1, rays, g0, cur)                   # bulk H2D (no scalar indexing)
        kern(traj, hlog, cnt, y0d, a, C, escape, R_em, nmax, r_h, cur; ndrange=cur)
        KernelAbstractions.synchronize(backend)
        hcnt = Array(view(cnt, 1:cur))
        maxn = Int(maximum(hcnt))
        htraj = Array(traj[1:cur, 1:maxn + 1])
        hhlog = maxn >= 1 ? Array(hlog[1:cur, 1:maxn]) : Array{T}(undef, cur, 0)
        @inbounds for lj in 1:cur
            m = Int(hcnt[lj]) + 1                       # stored states (y0 + steps)
            rev = Vector{SVector{8,Float64}}(undef, m)
            dl = Vector{Float64}(undef, m)
            for k in 1:m
                rev[k] = htraj[lj, m + 1 - k]           # integration is camera→source
            end
            for k in 1:m-1
                dl[k] = hhlog[lj, m - k]                 # step behind (rev[k]→rev[k+1])
            end
            dl[m] = 0.0
            states[g0 + lj - 1] = rev
            dls[g0 + lj - 1] = dl
        end
    end
    return nothing
end

"""
    trace_rays_gpu(st, rays; C=80.0, escape=1.1e3, R_em=Inf, nmax=3000,
                   devices=select_devices(), chunk=8192, precision=:f64)
        -> (states, dls)

Multi-GPU drop-in for `trace_rays_sampled`: the same fixed-control RK4 sampling
(step h = (r - r_h)/C, same termination rule) and identical output contract —

  states[i] :: Vector{SVector{8}} — samples ORDERED source→camera
  dls[i]    :: Vector{Float64}    — affine step behind each sample (dl[m] = 0).

States/steps agree with `trace_rays_sampled` to fp roundoff (~1 ulp per RHS
evaluation, ~1e-8 max pixel rel end-to-end; see the header), not bit-for-bit.

The ray list is split into `length(devices)` contiguous parts (kept contiguous for the
bulk H2D `copyto!`); each part runs on its own CUDA device (`CUDA.device!`) inside an
`@async` task so both GPUs execute concurrently. `devices` defaults to `select_devices()`.
`R_em` (default `Inf`, disabled) adds a far-field early-out: once a ray is outbound past
`R_em` (which must exceed the emission support of the consuming radiation stage) it stops
logging the vacuum return leg — pass it only when the fluid is known to be confined.

The march runs at T = the ray element type: Float64 rays reproduce the CPU-matching
path bitwise; `precision = :mixed` narrows Float64 rays to Float32 at upload (fp32
march + fp32 trajectory logs — half the dominant traj footprint), and the output
contract is UNCHANGED (states/dls are converted back to Float64 on download). The
fp32 march is round-off-floored near the critical curve — verify image-level
acceptance before adopting :mixed for a given configuration.
"""
function trace_rays_gpu(st::KerrSchildCartesian, rays::Vector{SVector{8,T}};
                        C::Real=80.0, escape::Real=1.1e3, R_em::Real=Inf,
                        nmax::Int=3000, devices=select_devices(), chunk::Int=8192,
                        precision::Symbol=:f64) where {T<:AbstractFloat}
    precision in (:f64, :mixed) ||
        error("trace_rays_gpu: precision must be :f64 or :mixed, got :$precision")
    if precision === :mixed && T !== Float32
        return trace_rays_gpu(st, [SVector{8,Float32}(r) for r in rays];
                              C=C, escape=escape, R_em=R_em, nmax=nmax,
                              devices=devices, chunk=chunk)
    end
    n = length(rays)
    a = st.a
    r_h = 1 + sqrt(1 - a * a)                            # == horizon_radius(st)
    # kernel scalars at the ray element type (identity when T == Float64;
    # r_h is computed in Float64 above either way, then narrowed once)
    aT = T(a); r_hT = T(r_h); CT = T(C); escapeT = T(escape); R_emT = T(R_em)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    nd = length(devices)
    bounds = [round(Int, n * k / nd) for k in 0:nd]      # balanced contiguous split
    @sync for di in 1:nd
        lo = bounds[di] + 1
        hi = bounds[di+1]
        dev = devices[di]
        lo <= hi || continue
        @async begin
            CUDA.device!(dev)
            _trace_part!(states, dls, rays, lo:hi, aT, r_hT, CT, escapeT, R_emT, nmax, chunk)
        end
    end
    return states, dls
end
