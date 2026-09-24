# Simulation tests: primitive→four-vector identities (exact GR), the AthenaK
# .athdf reader/sampler on a real GRMHD snapshot (isfile-guarded; override the
# path via ENV["BLP_TEST_GRMHD"]), synthetic AthenaK .bin adapters, leaf-table
# handling, and GPU helper parity on the KernelAbstractions CPU backend.
using Test, StaticArrays, LinearAlgebra
using BlackLightPlus.Simulation
using BlackLightPlus.Geodesics

const DATA = get(ENV, "BLP_TEST_GRMHD",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf")

# The four-vector identities hold for ANY normal-frame state, so use a fixed,
# diverse set of (uu^i, B^i) samples (reproducible, no RNG dependency).
const _STATES = (
    (SVector(0.20, -0.07, 0.15),  SVector(0.30, -0.40, 0.10)),
    (SVector(-0.35, 0.42, -0.11), SVector(-0.44, -0.34, -0.37)),
    (SVector(0.02, 0.56, 0.23),   SVector(4.38, -1.19, 1.81)),
    (SVector(0.90, -0.60, 0.30),  SVector(-2.10, 0.75, -1.30)),
    (SVector(0.0, 0.0, 0.0),      SVector(1.0, -2.0, 0.5)),
)

@testset "primitives: four-vector identities" begin
    for a in (0.0, 0.5, 0.9, 0.998), xyz in ((5.0, 2.0, 1.0), (8.0, -3.0, 2.5), (-4.0, 3.0, -2.0))
        st = KerrSchildCartesian(a)
        x = SVector(0.0, xyz...)
        gcov = metric(st, x)
        gcon = inverse_metric(st, x)
        for (uu, bb) in _STATES
            fv = primitive_four_vectors(uu[1], uu[2], uu[3], bb[1], bb[2], bb[3], gcov, gcon)
            # u·u = g_μν u^μ u^ν = -1, b·u = g_μν b^μ u^ν = 0, b_sq = b·b ≥ 0
            @test abs(dot(fv.ucon, fv.ucov) + 1) < 1e-12
            @test abs(dot(fv.bcon, fv.ucov)) < 1e-12
            @test fv.b_sq ≥ 0
            # ucov / bcov are consistent lowerings of ucon / bcon
            @test maximum(abs.(fv.ucov - gcov * fv.ucon)) < 1e-13
            @test maximum(abs.(fv.bcov - gcov * fv.bcon)) < 1e-13
        end
    end
    # pgas_from_eint is the ideal-gas closure
    @test pgas_from_eint(3.0, 5 / 3) == (5 / 3 - 1) * 3.0
end

@testset "units: code → CGS conversions" begin
    u = units(10.0 * 1.989e33, 1.0e18)        # ~10 Msun, arbitrary mass scale
    @test u.T_unit ≈ u.L_unit / 2.99792458e10
    @test u.dens_unit > 0 && u.Ne_unit > 0 && u.B_unit > 0
    @test u.Ne_unit ≈ u.dens_unit / (1.6726e-24 + 9.1094e-28)
end

if isfile(DATA)
    @testset "athenak reader: real snapshot" begin
        snap = AthenaKSnapshot(DATA)
        @test snap.nmb == 344
        @test (snap.ni, snap.nj, snap.nk) == (32, 32, 32)

        for (x, y, z) in ((6.0, 0.0, 0.0), (0.0, 8.0, 1.0), (3.0, 3.0, 2.0))
            s = sample_primitives(snap, SVector(0.0, x, y, z))
            @test s.inside
            @test s.rho > 0
            @test isfinite(s.eint)
            @test all(isfinite, s.uu)
            @test all(isfinite, s.bb)
        end

        # domain is [-512, 512]^3; a point well outside owns no block
        s_out = sample_primitives(snap, SVector(0.0, 600.0, 0.0, 0.0))
        @test !s_out.inside
        @test s_out.rho == 0.0
        @test s_out.uu == zero(SVector{3,Float64})
        @test s_out.bb == zero(SVector{3,Float64})
    end
else
    @info "test_simulation: skipping AthenaK reader tests (data file not present)" DATA
end

# ---------------------------------------------------------------------------------------
# athenak_bin_snapshot: the GRMHD-fluid .bin -> AthenaKSnapshot adapter contract, on a
# synthetic mhd_w_bcc v1.1 dump (no external data): bit passthrough of the 7 direct
# primitives, press -> eint = press/(gamma-1) rounded once from Float64, temperature
# dropped, per-block geometry + DERIVED logical locations, and non-finite zeroing.
@testset "athenak_bin_snapshot: fluid .bin adapter" begin
    γ = 13.0 / 9.0
    fvars = ("dens", "velx", "vely", "velz", "press", "bcc1", "bcc2", "bcc3", "temperature")
    nx = 4; cells = nx^3
    cellval(b, v, c) = Float32(1000 * b + 100 * (v - 1) + c)   # encodes (block, var, cell)
    mktempdir() do dir
        path = joinpath(dir, "fluid.bin")
        open(path, "w") do io
            params = "<par_end>\n"
            print(io, "Athena binary output version=1.1\n  size of preheader=5\n  time=2.0\n",
                  "  cycle=1\n  size of location=8\n  size of variable=4\n",
                  "  number of variables=", length(fvars), "\n",
                  "  variables:  ", join(fvars, "  "), "  \n",
                  "  header offset=", length(params), "\n", params)
            for b in 0:1                                       # two level-0 blocks tiling x in [0,2]
                write(io, Int32[0, nx - 1, 0, nx - 1, 0, nx - 1, b, 0, 0, 0])
                write(io, Float64[b, b + 1, 0, 1, 0, 1])
                for v in 1:length(fvars), c in 0:cells-1       # i fastest, then j,k,var
                    # one poisoned press cell in block 2 exercises the zeroing
                    write(io, (b == 1 && v == 5 && c == 7) ? NaN32 : cellval(b + 1, v, c))
                end
            end
        end
        snap = athenak_bin_snapshot(path; gamma=γ)
        @test (snap.ni, snap.nj, snap.nk, snap.nmb) == (nx, nx, nx, 2)
        @test snap.variable_names == ["dens", "velx", "vely", "velz", "eint",
                                      "bcc1", "bcc2", "bcc3"]
        # geometry: face bounds, spacing, first cell centre, derived logical locations
        @test snap.ext_lo[2] == SVector(1.0, 0.0, 0.0) && snap.ext_hi[2] == SVector(2.0, 1.0, 1.0)
        @test snap.dx[1] == SVector(0.25, 0.25, 0.25)
        @test snap.x_left[2] == SVector(1.125, 0.125, 0.125)
        @test snap.logical_locations == [0 1; 0 0; 0 0]
        # bit passthrough of the direct primitives; press slot holds eint
        binvar_of_prim = (1, 2, 3, 4, 5, 6, 7, 8)              # fvars index per prim slot
        for b in 1:2, (prim, v) in enumerate(binvar_of_prim), c in (0, 7, 63)
            i = c % nx + 1; j = (c ÷ nx) % nx + 1; k = c ÷ nx^2 + 1
            raw = (b == 2 && v == 5 && c == 7) ? 0.0f0 : cellval(b, v, c)   # NaN zeroed
            want = prim == 5 ? Float32(Float64(raw) / (γ - 1.0)) : raw
            @test snap.data[i, j, k, b, prim] === want
        end
        # eint is invertible back to press to one Float32 ulp
        @test pgas_from_eint(Float64(snap.data[2, 1, 1, 1, 5]), γ) ≈ Float64(cellval(1, 5, 1)) rtol = 2e-7
        # the sampler consumes it directly (interior point of block 1)
        s = sample_primitives(snap, SVector(0.0, 0.5, 0.5, 0.5))
        @test s.inside && s.rho > 0 && isfinite(s.eint)
    end
end

# ---------------------------------------------------------------------------------------
# gpu_amr._build_leaf_table alignment guard (regression). The device leaf builder has NO
# scan fallback (unlike its host twin athenak_reader.jl:_build_leaf_table), so a mesh that
# is not origin/finest-aligned previously wrote past `leaf` under @inbounds (silent heap
# corruption + an all-vacuum table -> all-black render). It must now error() loudly.
# Guarded, idempotent includes: gpu_* `using CUDA` (loads without a device) and
# _build_leaf_table is a pure host function (no kernel launched here).
@isdefined(_fused_kernel!)     || include(joinpath(@__DIR__, "..", "src", "gpu_image.jl"))
@isdefined(_gpol_kernel!)     || include(joinpath(@__DIR__, "..", "src", "gpu_pol.jl"))
@isdefined(_build_leaf_table) || include(joinpath(@__DIR__, "..", "src", "gpu_amr.jl"))

@testset "gpu_amr _build_leaf_table: reject non-finest-aligned dumps (was OOB write)" begin
    mk(ext_lo, ext_hi, levels, ll, nmb; ni = 4, nj = 4, nk = 4) = AthenaKSnapshot(
        zeros(Float32, ni, nj, nk, nmb, 8), ext_lo, ext_hi,
        [SVector(0.0, 0.0, 0.0) for _ in 1:nmb], [SVector(1.0, 1.0, 1.0) for _ in 1:nmb],
        levels, ll, String[], ni, nj, nk, nmb)
    # origin/finest-aligned mesh (lx1 in {0,1}) builds + locates correctly (positive control)
    good = mk([SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)],
              [SVector(1.0, 1.0, 1.0), SVector(2.0, 1.0, 1.0)],
              [0, 0], [0 1; 0 0; 0 0], 2)
    leaf, amrp = _build_leaf_table(good)
    @test (amrp.N1, amrp.N2, amrp.N3) == (2, 1, 1)
    @test vec(leaf) == Int32[1, 2]
    # partial-domain dump (lx1 in {2,3}): x1min maps to logical 2 so N1=2, but the blocks
    # index leaf[3]/leaf[4] -> previously a silent OOB write + all-vacuum table.
    bad = mk([SVector(2.0, 0.0, 0.0), SVector(3.0, 0.0, 0.0)],
             [SVector(3.0, 1.0, 1.0), SVector(4.0, 1.0, 1.0)],
             [0, 0], [2 3; 0 0; 0 0], 2)
    @test_throws ErrorException _build_leaf_table(bad)
end

# ------------------------------------------------------------------------------------
# Leaf-table sharing across SnapshotSeries frames (regression): two dumps with the
# SAME block count and global bounding box but a permuted block layout (a
# regrid-shaped change) must NOT share one leaf table — the adopted table would map
# points to the other frame's meshblock indices and silently sample the wrong block.
using BlackLightPlus
using BlackLightPlus.Simulation: synthetic_series, _adopt_shared_leaf!, _mesh_matches,
    leaf_table, meshblock_index, _find_meshblock_scan

# Two level-0 blocks tiling [0,2]x[0,1]x[0,1]; `swap` permutes the block FILE ORDER
# only (identical global bbox and nmb — exactly the case the old guard passed).
function _two_block_snap(swap::Bool)
    ni = nj = nk = 4
    ord = swap ? [2, 1] : [1, 2]
    lo = [SVector(0.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)][ord]
    hi = [SVector(1.0, 1.0, 1.0), SVector(2.0, 1.0, 1.0)][ord]
    ll = [0 1; 0 0; 0 0][:, ord]
    data = zeros(Float32, ni, nj, nk, 2, 8)
    for (slot, blk) in enumerate(ord)
        data[:, :, :, slot, 1] .= 10.0f0 * blk       # region density: 10 (x<1), 20 (x>1)
        data[:, :, :, slot, 5] .= 1.0f0
    end
    AthenaKSnapshot(data, lo, hi, [l .+ 0.125 for l in lo],
                    [SVector(0.25, 0.25, 0.25), SVector(0.25, 0.25, 0.25)],
                    [0, 0], ll,
                    ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                    ni, nj, nk, 2)
end

@testset "leaf-table sharing rejects a permuted (regridded) mesh" begin
    snapA = _two_block_snap(false)
    snapB = _two_block_snap(true)
    tA = leaf_table(snapA)
    @test tA !== nothing
    ser = synthetic_series(snapA; times = [0.0])
    @test !_mesh_matches(snapA, snapB)               # regrid-shaped change: no match
    _adopt_shared_leaf!(ser, snapB)
    @test snapB.leafcache.tbl === nothing            # must NOT adopt A's table
    # locate agrees with the scan oracle in BOTH frames at interior points
    for s in (snapA, snapB), (x, y, z) in ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5))
        @test meshblock_index(s, x, y, z) == _find_meshblock_scan(s, x, y, z)
    end
    @test sample_primitives(snapB, SVector(0.0, 1.5, 0.5, 0.5)).rho == 20.0
    # an identical mesh still shares ONE table (the shared-table fast path is preserved)
    snapA2 = _two_block_snap(false)
    @test _mesh_matches(snapA, snapA2)
    _adopt_shared_leaf!(ser, snapA2)
    @test leaf_table(snapA2) === tA
end

# ------------------------------------------------------------------------------------
# render_amr (regressions): (1) a resolution that is not a multiple of `block` must
# error loudly instead of silently returning zero strips on two image edges; (2) at
# max_level=0 with identical pixel centres the mosaic must match render()'s
# I[iβ, iα] layout pixelwise, NOT its transpose.
@testset "render_amr: resolution guard + render() orientation parity" begin
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    # fov/resolution dyadic -> render() and render_amr() pixel centres coincide EXACTLY
    mkcfg(res) = RunConfig(bh_a = 0.9, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                           frequency_hz = 2.3e11,
                           camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0,
                                                 fov = 32.0, resolution = res))
    # guard fires before the snapshot is touched
    @test_throws "not divisible" render_amr(mkcfg(12), "/nonexistent.athdf")
    if isfile(DATA)
        c = mkcfg(16)
        u = render(c, DATA; polarized = false)
        r = render_amr(c, DATA; block = 8, max_level = 0)
        @test size(r.I) == size(u.I)
        @test isapprox(r.I, u.I; rtol = 1e-10)                # same [iβ, iα] layout
        @test !isapprox(r.I, transpose(u.I); rtol = 1e-3)     # and NOT the transpose
    else
        @info "render_amr parity: skipping (data file not present)" DATA
    end
end

# resample_z4c_adm now locates the owning block + builds the trilinear stencil ONCE
# per grid point and gathers all 11 ADM fields from it (was 11 independent O(n_mb)
# locates/stencils per point). This pins the hoisted path bit-identical to the
# original per-variable `sample_var` composition, on a synthetic AMR .bin.
@testset "z4c resample: shared-stencil == per-variable sample_var (bit-identical)" begin
    AthenaKBin = BlackLightPlus.Simulation.AthenaKBin
    zvars = ["z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
             "z4c_gzz", "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz"]
    nx = 5; nv = length(zvars)
    # coarse root [-4,4]^3 (level 0) + a finer nested block [-1,1]^3 (level 1), so
    # finest-wins location, inter-level sampling and the off-mesh corner all fire.
    x1min = [-4.0, -1.0]; x1max = [4.0, 1.0]
    x2min = [-4.0, -1.0]; x2max = [4.0, 1.0]
    x3min = [-4.0, -1.0]; x3max = [4.0, 1.0]
    level = [0, 1]; n_mb = 2
    data = Array{Float32,5}(undef, nx, nx, nx, nv, n_mb)
    for b in 1:n_mb, v in 1:nv, k in 1:nx, j in 1:nx, i in 1:nx
        cx = x1min[b] + (i - 0.5) * (x1max[b] - x1min[b]) / nx
        cy = x2min[b] + (j - 0.5) * (x2max[b] - x2min[b]) / nx
        cz = x3min[b] + (k - 0.5) * (x3max[b] - x3min[b]) / nx
        val = v == 1 ? 0.4 + 0.3 * sin(0.3cx) * cos(0.2cy) :   # chi in (0,1)
                       0.2 * sin(0.17cx * v) + 0.1 * cos(cy + v) + 0.03cz * v
        data[i, j, k, v, b] = Float32(val)
    end
    s = AthenaKBin(n_mb, nx, nx, nx, zvars, level,
                   x1min, x1max, x2min, x2max, x3min, x3max, data, 0.0)
    half = 6.0; n = 20                       # box extends past the root -> off-mesh corners
    axis, alpha, beta, gamma = resample_z4c_adm(s, half, n)

    col = Dict(name => findfirst(==(name), zvars) for name in zvars)
    ax = range(-half, half; length=n)
    saw_offmesh = false
    for k in 1:n, j in 1:n, i in 1:n
        x = ax[i]; y = ax[j]; z = ax[k]
        chi = sample_var(s, col["z4c_chi"], x, y, z)          # pre-hoist reference path
        if isnan(chi)
            saw_offmesh = true                                # off-mesh -> Minkowski row
            @test alpha[i, j, k] == 1.0
            @test beta[1, i, j, k] == 0.0 && beta[2, i, j, k] == 0.0 && beta[3, i, j, k] == 0.0
            @test gamma[1, i, j, k] == 1.0 && gamma[4, i, j, k] == 1.0 && gamma[6, i, j, k] == 1.0
            @test gamma[2, i, j, k] == 0.0 && gamma[3, i, j, k] == 0.0 && gamma[5, i, j, k] == 0.0
        else
            psi4 = max(chi, 1.0e-12)^(4.0 / -4.0)             # = 1/chi (matches _Z4C_CHI_PSI_POWER)
            @test alpha[i, j, k] == sample_var(s, col["z4c_alpha"], x, y, z)
            @test beta[1, i, j, k] == sample_var(s, col["z4c_betax"], x, y, z)
            @test beta[2, i, j, k] == sample_var(s, col["z4c_betay"], x, y, z)
            @test beta[3, i, j, k] == sample_var(s, col["z4c_betaz"], x, y, z)
            @test gamma[1, i, j, k] == psi4 * sample_var(s, col["z4c_gxx"], x, y, z)
            @test gamma[4, i, j, k] == psi4 * sample_var(s, col["z4c_gyy"], x, y, z)
            @test gamma[6, i, j, k] == psi4 * sample_var(s, col["z4c_gzz"], x, y, z)
        end
    end
    @test saw_offmesh                                          # the Minkowski branch really ran
    @test isnan(sample_var(s, 1, 100.0, 0.0, 0.0))            # off-mesh sample_var -> NaN
    @test !isnan(sample_var(s, 1, 0.0, 0.0, 0.0))            # finest block covers the centre
end

# write_npy now emits the C-order payload in one bulk write of `permutedims(a)`.
# Pin the exact on-disk .npy bytes (header + row-major Float64 stream) so the
# faster path can't silently change the layout numpy.load expects.
using BlackLightPlus: write_npy
@testset "write_npy: bulk write is byte-correct C-order .npy" begin
    a = reshape(collect(1.0:15.0), 5, 3) .+ 0.25             # non-square, distinct values
    path = tempname() * ".npy"
    write_npy(path, a)
    bytes = read(path)
    @test bytes[1:6] == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]   # \x93NUMPY
    @test bytes[7] == 0x01 && bytes[8] == 0x00                       # version 1.0
    hlen = Int(reinterpret(UInt16, bytes[9:10])[1])
    hdr = String(copy(bytes[11:10 + hlen]))
    @test occursin("'shape': (5, 3)", hdr)
    @test occursin("'fortran_order': False", hdr)
    payload = reinterpret(Float64, bytes[11 + hlen:end])
    @test length(payload) == 15
    @test payload == [a[i, j] for i in 1:5 for j in 1:3]            # C order: dim-1 outer
    rm(path; force=true)
end

# ------------------------------------------------------------------------------------
# gpu_tracer._trace_kernel! directional escape (regression, KA CPU backend). The CPU
# escape stops were made outbound-only (pre-fix a ray STARTED beyond `escape` was
# killed at step 0 on its way in); the device tracer carried the old
# unconditional `r > escape -> break`. Mirror check: a camera placed outside the
# escape sphere must still trace inbound, and step counts must track the (fixed)
# CPU trace_rays_sampled on the same rays.
@isdefined(_trace_kernel!) || include(joinpath(@__DIR__, "..", "src", "geodesics", "gpu_tracer.jl"))
using KernelAbstractions
using BlackLightPlus: trace_rays_sampled, PlaneParallelCamera, camera_rays

@testset "gpu tracer: inbound ray beyond `escape` is not killed at step 0" begin
    a = 0.9375
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(30.0, 1200.0, 10.0, 2)   # camera OUTSIDE escape = 1100
    rays = camera_rays(st, cam)
    escape = 1100.0
    r_h = 1 + sqrt(1 - a * a); nmax = 3000; C = 80.0
    nr = length(rays)
    backend = CPU()
    traj = Matrix{SVector{8,Float64}}(undef, nr, nmax + 1)
    hlog = Matrix{Float64}(undef, nr, nmax)
    cnt = Vector{Int32}(undef, nr)
    _trace_kernel!(backend)(traj, hlog, cnt, rays, a, C, escape, Inf, nmax, r_h, nr;
                            ndrange = nr)
    KernelAbstractions.synchronize(backend)
    @test all(>(100), cnt)                              # pre-fix: cnt == 0 for every ray
    # the march really went inside the escape sphere (not just lingered at the camera)
    for i in 1:nr
        rmin = minimum(ks_radius(st, SVector(traj[i, s][1], traj[i, s][2], traj[i, s][3], traj[i, s][4])) for s in 1:Int(cnt[i]) + 1)
        @test rmin < 100.0
    end
    # step-count parity with the (already fixed) CPU tracer on the identical rays;
    # the RHS agree to ~1 ulp (not bitwise), so allow a marginal-break slop of 2.
    states, _ = trace_rays_sampled(st, rays; C = C, escape = escape, nmax = nmax)
    for i in 1:nr
        @test abs(Int(cnt[i]) + 1 - length(states[i])) <= 2
    end
end

# ---------------------------------------------------------------------------------------
# Leaf-table ONE-SOURCE consolidation pin: the device builder now consumes the host
# leaf_table(snap) (athenak_reader.jl) instead of maintaining a drifted twin — SAME
# array object (host cache reused), same locate scalars; the host guard cases (nothing)
# error loudly on the device path, which has no scan fallback (covered above).
@testset "gpu_amr _build_leaf_table: consumes the host leaf_table (one source)" begin
    snap = _two_block_snap(false)
    leaf, amrp = _build_leaf_table(snap)
    t = leaf_table(snap)
    @test t !== nothing
    @test leaf === t.leaf                          # SAME object: no rebuilt twin
    @test (amrp.N1, amrp.N2, amrp.N3) == (t.N1, t.N2, t.N3)
    @test (amrp.x1min, amrp.x2min, amrp.x3min) == (t.x1min, t.x2min, t.x3min)
    @test (amrp.inv_fdx1, amrp.inv_fdx2, amrp.inv_fdx3) == (t.inv_fdx1, t.inv_fdx2, t.inv_fdx3)
    @test amrp.nmb == t.nmb == snap.nmb
    @test (amrp.ni, amrp.nj, amrp.nk) == (snap.ni, snap.nj, snap.nk)
    # locate parity: the device fast locate over (leaf, amrp) == host meshblock_index
    for (x, y, z) in ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (2.5, 0.5, 0.5), (-0.1, 0.2, 0.3))
        @test Int(_gamr_locate_fast(leaf, amrp, x, y, z)) == meshblock_index(snap, x, y, z)
    end
end

# ---------------------------------------------------------------------------------------
# Geodesic RHS regression. Every CPU tracer, device tracer and fused kernel steps with
# the one Geodesics `geodesic_rhs` (Hamiltonian contraction form: dp_i = -½ ∂_i(g^{μν}p_μp_ν)
# from `raise_and_norm` on a dual-seeded position). Checked (i) against the textbook form
# -½ (∂_i g^{μν}) p_μ p_ν with ∂g^{μν} from a ForwardDiff jacobian of the closed-form
# inverse metric, and (ii) against reference values of the pre-consolidation device RHS.
using Random, ForwardDiff
using BlackLightPlus.Geodesics: geodesic_rhs, geodesic_transport_rhs
@testset "geodesic RHS: Hamiltonian contraction == textbook ∂g form; pinned values" begin
    a = 0.9375; st = KerrSchildCartesian(a); r_h = 1 + sqrt(1 - a * a)
    rng = MersenneTwister(0x5EA3)
    ntest = 0
    for _ in 1:3000
        X = (rand(rng) - 0.5) * 2.4e3; Y = (rand(rng) - 0.5) * 2.4e3; Z = (rand(rng) - 0.5) * 2.4e3
        x = SVector(0.0, X, Y, Z)
        ks_radius(st, x) > r_h || continue
        p = SVector(randn(rng), randn(rng), randn(rng), randn(rng))
        y = SVector(0.0, X, Y, Z, p[1], p[2], p[3], p[4])
        J = ForwardDiff.jacobian(xx -> inverse_metric(st, xx), x)      # (16, 4)
        dg(c) = reshape(J[:, c], 4, 4)
        ref = vcat(inverse_metric(st, x) * p, 0.0,
                   [-0.5 * dot(p, dg(c) * p) for c in 2:4])
        d = geodesic_rhs(st, y)
        scale = maximum(abs, ref)
        @test maximum(abs.(d .- ref)) <= 1e-13 * scale
        ntest += 1
    end
    @test ntest > 2500                                          # domain filter kept the sample set
    # reference values (pre-consolidation device RHS and frame transport, a = 0.9375)
    y = SVector(0.0, 3.0, 2.0, 1.5, -1.0, 0.4, -0.2, 0.1)
    @test geodesic_rhs(st, y) ≈ SVector(1.6880859188282549, -0.19485518050083636,
        -0.414892472296037, -0.17098112715009844, 0.0, -0.0661933978494909,
        -0.11243827357252122, -0.06326190184633673) rtol = 1e-14
    y2 = SVector(-12.0, 25.0, -40.0, 7.0, -1.0, -0.3, 0.6, 0.05)
    @test geodesic_rhs(st, y2) ≈ SVector(1.014500231516396, -0.3073612385561768,
        0.6123100190633601, 0.047871077359824474, 0.0, -1.7889358723359663e-5,
        5.7572470545296375e-5, 3.674142047710762e-5) rtol = 1e-13
    dy, (du, dv) = geodesic_transport_rhs(st, y, (SVector(0.1, 0.9, -0.3, 0.2),
                                                  SVector(1.2, 0.1, 0.2, -0.1)))
    @test dy ≈ geodesic_rhs(st, y) rtol = 1e-14
    @test du ≈ SVector(-0.043775113046461755, -0.01634815409749614,
                       -0.021421836958690668, -0.030252624371486254) rtol = 1e-13
    @test dv ≈ SVector(-0.03984434603449974, -0.05443581131101774,
                       -0.043394329516830164, -0.03968083098638997) rtol = 1e-13
end
