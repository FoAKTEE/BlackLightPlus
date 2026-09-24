# Shared inputs of the multi-device tests (test_device_invariance.jl, test_device_rays.jl,
# test_device_amr.jl): the GPU driver includes, HostDevice lists and pools, a torus deck
# on a uniform grid and on a one-level AMR mesh, and the bit view of driver outputs.
using Test, StaticArrays, KernelAbstractions, Random
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, SuperposedKerrSchild, PlaneParallelCamera, camera_rays
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot

const _DI_SRC = joinpath(@__DIR__, "..", "src")
@isdefined(_fused_kernel!)  || include(joinpath(_DI_SRC, "gpu_image.jl"))
@isdefined(_gpol_kernel!)  || include(joinpath(_DI_SRC, "gpu_pol.jl"))
@isdefined(_gamr_sample)   || include(joinpath(_DI_SRC, "gpu_amr.jl"))
@isdefined(_gsp_amr_kernel!)       || include(joinpath(_DI_SRC, "gpu_spacetime.jl"))
@isdefined(_fused_kernel_ad!) || include(joinpath(_DI_SRC, "gpu_image_ad.jl"))
@isdefined(_mc_shares) || include(joinpath(_DI_SRC, "montecarlo", "mc_gpu.jl"))
@isdefined(_ganalytic_kernel!) || include(joinpath(_DI_SRC, "gpu_analytic.jl"))
@isdefined(_trace_kernel!) || include(joinpath(_DI_SRC, "geodesics", "gpu_tracer.jl"))

const DI_NDEV = (1, 2, 3, 5)
di_hosts(nd) = [HostDevice(k) for k in 1:nd]
const DI_POOLS = Dict(nd => DevicePool(di_hosts(nd); lanes = 2, chunk = 7) for nd in DI_NDEV[2:end])
di_devices(nd) = nd == 1 ? di_hosts(1) : DI_POOLS[nd]

# Bit pattern of every Float64 in an image (Matrix, Vector{SVector}, or NamedTuple of
# matrices): equal bit patterns <=> bitwise-identical output, NaN and -0.0 included.
di_bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
di_bits(x::AbstractArray{<:SVector}) = reduce(vcat, (di_bits(collect(v)) for v in x))
di_bits(x::NamedTuple) = reduce(vcat, (di_bits(v) for v in values(x)))
di_bits(x::Tuple) = reduce(vcat, (di_bits(v) for v in x))
di_bits(x::Real) = [reinterpret(UInt64, Float64(x))]
di_bits(x::AbstractVector{<:AbstractArray}) = reduce(vcat, (di_bits(m) for m in x))

# ------------------------------------------------------------------ imaging decks
const DI_M_BH = 6.2e9 * 1.989e33
const DI_L = BlackLightPlus.Constants.GNEWT * DI_M_BH / BlackLightPlus.Constants.CL^2
di_cfg(; res = 10) = RunConfig(bh_a = 0.9, m_bh_cgs = DI_M_BH, mass_scale_cgs = 1.0e-21 * DI_L^3,
    frequency_hz = 2.3e11,
    camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 24.0, resolution = res),
    plasma = PlasmaConfig(rat_high = 10.0, rat_low = 1.0, gamma = 13.0 / 9.0, use_p = true,
                          sigma_cut = 1.0, theta_e_min = -1.0))

# Torus-like primitives (rho, v1..3, eint, B1..3) at (x, y, z).
function di_prim(x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < 2.2 || r > 11.0) && return ntuple(_ -> 0.0f0, 8)
    rho = exp(-((r - 5.5) / 2.5)^2)
    return Float32.((rho, -0.2y / r, 0.2x / r, 0.01, 0.08rho,
                     0.3sqrt(rho), 0.1sqrt(rho) * x / r, 0.2sqrt(rho)))
end

# Uniform (8, n, n, n) grid on [-bh, bh]^3.
function di_uniform(; n = 24, bh = 12.0)
    host = zeros(Float32, 8, n, n, n)
    dx = 2bh / (n - 1)
    for k in 1:n, j in 1:n, i in 1:n
        host[:, i, j, k] .= di_prim(-bh + (i - 1) * dx, -bh + (j - 1) * dx, -bh + (k - 1) * dx)
    end
    UniformFluidGrid(host, n, bh, -bh, (n - 1) / (2bh))
end

# 2×2×2 meshblocks of 6³ cells on [-12, 12]^3 (one level, finest-aligned); `scale`
# multiplies the density (a second slow-light frame on the same mesh).
function di_amr(; nc = 6, half = 12.0, scale = 1.0f0)
    lo_all = -half; w = half                     # block width
    dx = w / nc
    nmb = 8
    data = zeros(Float32, nc, nc, nc, nmb, 8)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; xl = SVector{3,Float64}[]
    dxs = SVector{3,Float64}[]; lloc = zeros(Int, 3, nmb)
    m = 0
    for bk in 0:1, bj in 0:1, bi in 0:1
        m += 1
        l = SVector(lo_all + bi * w, lo_all + bj * w, lo_all + bk * w)
        push!(lo, l); push!(hi, l .+ w); push!(xl, l .+ dx / 2); push!(dxs, SVector(dx, dx, dx))
        lloc[:, m] .= (bi, bj, bk)
        for k in 1:nc, j in 1:nc, i in 1:nc
            p = di_prim(xl[m][1] + (i - 1) * dx, xl[m][2] + (j - 1) * dx, xl[m][3] + (k - 1) * dx)
            for q in 1:8
                data[i, j, k, m, q] = q == 1 ? scale * p[q] : p[q]
            end
        end
    end
    AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                    ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                    nc, nc, nc, nmb)
end
