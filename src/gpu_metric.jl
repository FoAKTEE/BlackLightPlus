# gpu_metric.jl — Kerr march control of the fused GPU kernels and the device tracer.
# The metric, its inverse and the geodesic RHS are the Geodesics module functions
# (`metric_and_inverse`, `geodesic_rhs`, `rk4_step_fixed` on `KerrSchildCartesian`),
# compiled in-kernel; this file adds only the stop rule. Loaded solely by the
# CUDA-including GPU driver files (not from BlackLightPlus.jl), each via
#   isdefined(@__MODULE__, :_gimg_stop) || include(<this file>)
# so any include order of the GPU files yields exactly one copy.

using StaticArrays
using BlackLightPlus.Geodesics: ks_radius_energy, march_stop

# ---------------------------------------------------------------- march termination
# The Kerr march control of every fused kernel and the device tracer: the shared
# `march_stop` rule (Geodesics, integrator.jl) on the KS radius and normal-frame energy
# from `ks_radius_energy` — the same functions the CPU sampled tracer calls, so CPU and
# device stop on the same sample. Returns (stop, r); r feeds the step h = (r - r_h)/C.
# Runs in T = eltype(y); the matched ::T scalars make a stray Float64 a MethodError.
@inline function _gimg_stop(y::SVector{8,T}, r_prev::T, r_h::T, escape::T, R_em::T,
                            a::T) where {T}
    r, En = ks_radius_energy(y, a)
    return march_stop(r - r_h, En, r, r_prev, escape, R_em), r
end
