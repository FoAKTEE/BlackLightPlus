# Device-count invariance of the multi-device drivers (results/blp/multigpu/dag.md,
# MG03). Every imaging driver renders with `devices = [HostDevice(1), …, HostDevice(nd)]`
# for nd = 1, 2, 3: one logical device per entry on the KernelAbstractions CPU backend,
# through the production interleaved split, per-device upload and per-part launch. Each
# ray is one thread with no cross-ray reduction, so every image is bitwise identical
# across nd. The ray count (10² = 100) is not a multiple of 2 or 3, so the parts are
# uneven.
#
# Monte Carlo: the superphoton streams and the reduced spectrum are independent of the
# device (partition) count, and a scattering run is reproducible run to run (findings
# C1-C3 of results/blp/multigpu/design.md §3, resolved by MG05/MG06):
#   C1  a child's stream is keyed by its genealogy (`mc_child_rng`), not by the atomic
#       buffer slot that stores it;
#   C2  device shares are contiguous global-id ranges of one seed: the stream set is the
#       same for any device count, with no key reuse at five devices;
#   C3  spectra and diagnostics are exact integer-limb sums (exact_sum.jl), and CPU
#       child keys do not depend on the chunk: bitwise independent of the chunk count.
# The device scatter driver runs on HostDevices (KA CPU backend) always, and on CUDA
# devices when CUDA is functional.
using Test, StaticArrays, KernelAbstractions
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, SuperposedKerrSchild
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
using BlackLightPlus.Simulation: AthenaKSnapshot
using BlackLightPlus: SphereModel, ScatteringSphere, mc_spectrum, mc_spectrum_scatter
using BlackLightPlus: HotcrossTable
using BlackLightPlus.Constants: SIGMA_THOMSON
using BlackLightPlus.MonteCarlo: NW, NT

const _DI_SRC = joinpath(@__DIR__, "..", "src")
@isdefined(_gimg_thermal)  || include(joinpath(_DI_SRC, "gpu_image.jl"))
@isdefined(_gpol_thermal)  || include(joinpath(_DI_SRC, "gpu_pol.jl"))
@isdefined(_gamr_sample)   || include(joinpath(_DI_SRC, "gpu_amr.jl"))
@isdefined(_gsp_ctl)       || include(joinpath(_DI_SRC, "gpu_spacetime.jl"))
@isdefined(_fused_kernel_ad!) || include(joinpath(_DI_SRC, "gpu_image_ad.jl"))
@isdefined(_mc_device_shares) || include(joinpath(_DI_SRC, "montecarlo", "mc_gpu.jl"))

const DI_NDEV = (1, 2, 3)
di_devices(nd) = [HostDevice(k) for k in 1:nd]

# Bit pattern of every Float64 in an image (Matrix, Vector{SVector}, or NamedTuple of
# matrices): equal bit patterns <=> bitwise-identical output, NaN and -0.0 included.
di_bits(x::AbstractArray{<:Real}) = reinterpret(UInt64, Float64.(vec(x)))
di_bits(x::AbstractArray{<:SVector}) = reduce(vcat, (di_bits(collect(v)) for v in x))
di_bits(x::NamedTuple) = reduce(vcat, (di_bits(v) for v in values(x)))
di_bits(x::Tuple) = reduce(vcat, (di_bits(v) for v in x))
di_bits(x::Real) = [reinterpret(UInt64, Float64(x))]
di_bits(x::AbstractVector{<:AbstractMatrix}) = reduce(vcat, (di_bits(m) for m in x))

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

# 2×2×2 meshblocks of 6³ cells on [-12, 12]^3 (one level, finest-aligned).
function di_amr(; nc = 6, half = 12.0)
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
                data[i, j, k, m, q] = p[q]
            end
        end
    end
    AthenaKSnapshot(data, lo, hi, xl, dxs, zeros(Int, nmb), lloc,
                    ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                    nc, nc, nc, nmb)
end

# Render with nd = 1, 2, 3 logical devices; assert nonzero finite output and bitwise
# equality of every nd > 1 image with the nd = 1 image.
function di_check(render)
    ref = render(di_devices(1))
    b1 = di_bits(ref)
    fl = reinterpret(Float64, b1)
    @test all(isfinite, fl)
    @test count(!iszero, fl) > 0
    for nd in DI_NDEV[2:end]
        @test di_bits(render(di_devices(nd))) == b1
    end
    return ref
end

@testset "device-count invariance (KA CPU backend, HostDevice)" begin

@testset "imaging: bitwise identical for 1, 2, 3 devices" begin
    cfg = di_cfg()
    fg = di_uniform()
    snap = di_amr()

    @testset "fused unpolarized (:f64)" begin
        di_check(d -> render_gpu_fused(cfg, fg; devices = d))
    end
    @testset "fused unpolarized (:mixed, rescue pool split)" begin
        fr = Ref(0.0)
        di_check(d -> render_gpu_fused(cfg, fg; devices = d, precision = :mixed,
                                       rescued_frac = fr))
        @test fr[] > 0                             # the rescue sweep ran
    end
    @testset "fused K-variant" begin
        ms = [cfg.mass_scale_cgs, 2.0 * cfg.mass_scale_cgs]
        di_check(d -> render_gpu_fused_kvariant(cfg, fg, ms; devices = d))
    end
    @testset "fused forward-mode AD" begin
        di_check(d -> render_gpu_fused_grad(cfg, fg, :mass_scale; devices = d))
    end
    @testset "fused polarized" begin
        di_check(d -> render_gpu_polarized(cfg, fg; devices = d))
    end
    @testset "native AMR unpolarized" begin
        di_check(d -> render_gpu_amr(cfg, snap; devices = d))
    end
    @testset "native AMR polarized" begin
        di_check(d -> render_gpu_amr_polarized(cfg, snap; devices = d))
    end
    @testset "generic spacetime (superposed binary), unpolarized + polarized" begin
        st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        di_check(d -> render_gpu_spacetime(cfg, st, snap; devices = d))
        di_check(d -> render_gpu_spacetime(cfg, st, snap; devices = d, polarized = true))
    end
end

# ------------------------------------------------------------------ Monte Carlo
@testset "MC: GPU driver stream assignment (_mc_device_shares)" begin
    seed = 0x5ca77e5
    N = 1001
    # the Philox (key, counter-word) identity of every launched primary stream
    streams(nd) = Set((r.key, r.sid) for s in _mc_device_shares(N, nd)
                      for r in (BlackLightPlus.MonteCarlo.mc_primary_rng(seed, g)
                                for g in s.first:(s.first + s.n - 1)))
    for nd in 1:5
        sh = _mc_device_shares(N, nd)
        @test sum(s.n for s in sh) == N           # every superphoton launched once
        @test length(streams(nd)) <= N
    end
    @test _mc_device_shares(N, 1)[1] == (first = 1, n = N)   # devices=[d] == single device
    # the primary stream set does not depend on the device count
    @test streams(2) == streams(1)                # C2
    @test streams(3) == streams(1)                # C2
    # no two superphotons share a stream, for any device count
    for nd in 2:4
        @test length(streams(nd)) == N
    end
    @test length(streams(5)) == N                 # C2: no key reuse at five devices
end

# CPU MC: `nthreads` sets the chunk count, the CPU analogue of the device count.
@testset "MC: CPU chunk-count invariance and reproducibility" begin
    sph = SphereModel()
    thin(k) = mc_spectrum(sph, 20_000; nthreads = k, seed = 0xC0FFEE, n_ebins = 60)
    t1 = thin(1)
    for k in (2, 3)
        tk = thin(k)
        @test tk.count == t1.count                # same superphotons, integer bins
        # bitwise spectrum: exact per-chunk sums merged by integer addition
        @test di_bits(tk.sumw) == di_bits(t1.sumw)            # C3
    end

    tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))   # Thomson table
    ssph = ScatteringSphere()
    scat(k) = mc_spectrum_scatter(ssph, tbl, 5_000; nthreads = k, seed = 0xBEEF,
                                  n_ebins = 60, max_scatt = 2)
    s2a = scat(2); s2b = scat(2)
    @test di_bits(s2a.total.sumw) == di_bits(s2b.total.sumw)  # run-to-run, fixed chunks
    @test s2a.n_children == s2b.n_children
    s1 = scat(1)
    @test s1.n_children > 0                       # scattering is exercised
    for k in (2, 3)
        # identical children and spectrum for any chunk count
        @test di_bits(scat(k).total.sumw) == di_bits(s1.total.sumw)   # C3
    end
end

@testset "MC: GPU scatter reproducibility and device split" begin
    tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
    # the device kernels on the KA CPU backend (HostDevice), always run
    hrun(nd) = mc_spectrum_gpu(ScatteringSphere(), tbl, 5_000; devices = di_devices(nd),
                               seed = 0xBEEF, n_ebins = 60, max_scatt = 2)
    a = hrun(1); b = hrun(1)
    @test a.n_children > 0
    @test di_bits(a.total.sumw) == di_bits(b.total.sumw)                  # C1: run to run
    @test di_bits(hrun(2).total.sumw) == di_bits(a.total.sumw)            # C1, C2: 1 vs 2
    if CUDA.functional()
        run(devs) = mc_spectrum_gpu(ScatteringSphere(), tbl, 200_000; devices = devs,
                                    seed = 0xBEEF, max_scatt = 2)
        g = run([0])
        @test di_bits(g.total.sumw) == di_bits(run([0]).total.sumw)       # C1 on CUDA
        if length(CUDA.devices()) >= 2
            @test di_bits(run([0, 1]).total.sumw) == di_bits(g.total.sumw)  # C1, C2 on CUDA
        else
            @test_skip false                      # needs two CUDA devices
        end
    else
        @test_skip false                          # CUDA run to run: needs a CUDA device
        @test_skip false                          # CUDA 1 vs 2: needs two CUDA devices
    end
end

end
