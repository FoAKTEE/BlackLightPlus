# The AthenaK `.bin` reader on synthetic v1.1 dumps: declared header sizes, the GRMHD
# fluid adapter (bit passthrough, eint from press, temperature dropped, geometry and
# derived logical locations, non-finite zeroing), and the Z4c resample onto an ADM grid
# (one stencil per node, identical to per-variable `sample_var`; Minkowski off the mesh).
using Test, BLP, StaticArrays
using BLP.Data: AthenaKBin, read_athenak_bin, sample_var, resample_z4c_adm,
                z4c_grid_spacetime, athenak_bin_snapshot, PRIM_NAMES
using BLP.Spacetime: ADMGrid
using BLP.Matter: primitives, pgas_from_eint

# A v1.1 dump of `nvars` variables on blocks tiling x ∈ [0, nmb]; `value(b, v, c)` is the
# value of block `b`, variable `v`, cell `c` (0-based, i fastest).
function write_bin(path, vars; LocT = Float64, var_size = 4, nx = 4, nmb = 2,
                   value = (b, v, c) -> Float32(1000 * b + 100 * (v - 1) + c), time = 3.5)
    params = "<par_end>\n"
    open(path, "w") do io
        print(io, "Athena binary output version=1.1\n  size of preheader=5\n  time=$time\n",
              "  cycle=7\n  size of location=", sizeof(LocT), "\n",
              "  size of variable=", var_size, "\n",
              "  number of variables=", length(vars), "\n",
              "  variables:  ", join(vars, "  "), "  \n",
              "  header offset=", length(params), "\n", params)
        for b in 0:nmb-1
            write(io, Int32[0, nx - 1, 0, nx - 1, 0, nx - 1, b, 0, 0, 0])
            write(io, LocT[b, b + 1, 0, 1, 0, 1])
            for v in 1:length(vars), c in 0:nx^3-1
                write(io, value(b + 1, v, c))
            end
        end
    end
    return path
end

@testset "athenak .bin" begin

@testset "reader honours declared header sizes" begin
    mktempdir() do dir
        for LocT in (Float64, Float32)
            s = read_athenak_bin(write_bin(joinpath(dir, "loc$(sizeof(LocT)).bin"),
                                           ["z4c_alpha", "z4c_chi"]; LocT))
            @test s.n_mb == 2 && s.time == 3.5 && (s.nx1, s.nx2, s.nx3) == (4, 4, 4)
            @test s.x1min == [0.0, 1.0] && s.x1max == [1.0, 2.0]
            @test s.x2min == [0.0, 0.0] && s.x3max == [1.0, 1.0]
            @test s.var_names == ["z4c_alpha", "z4c_chi"]
            @test s.data[1, 1, 1, 1, 1] == 1000.0f0 && s.data[2, 1, 1, 2, 2] == 2101.0f0
        end
        @test_throws ErrorException read_athenak_bin(write_bin(joinpath(dir, "var8.bin"),
                                                               ["a", "b"]; var_size = 8))
        p = write_bin(joinpath(dir, "trunc.bin"), ["a", "b"])
        open(p, "a") do io; write(io, 0x00); end
        @test_throws ErrorException read_athenak_bin(p)
        q = joinpath(dir, "notbin.bin")
        write(q, "Athena binary output version=1.0\n")
        @test_throws ErrorException read_athenak_bin(q)
    end
end

@testset "GRMHD fluid adapter" begin
    γ = 13.0 / 9.0
    fvars = ["dens", "velx", "vely", "velz", "press", "bcc1", "bcc2", "bcc3", "temperature"]
    nx = 4
    cellval(b, v, c) = Float32(1000 * b + 100 * (v - 1) + c)
    poisoned(b, v, c) = (b == 2 && v == 5 && c == 7) ? NaN32 : cellval(b, v, c)
    mktempdir() do dir
        path = write_bin(joinpath(dir, "fluid.bin"), fvars; nx, value = poisoned, time = 2.0)
        snap = (@test_logs (:warn, r"zeroed 1 non-finite") athenak_bin_snapshot(path; gamma = γ))
        b = snap.blocks
        @test (b.ni, b.nj, b.nk, b.nmb) == (nx, nx, nx, 2) && snap.time == 2.0
        @test snap.names == collect(PRIM_NAMES)
        @test b.lo[:, 2] == [1.0, 0.0, 0.0] && b.hi[:, 2] == [2.0, 1.0, 1.0]
        @test b.dx[:, 1] == [0.25, 0.25, 0.25] && b.x_left[:, 2] == [1.125, 0.125, 0.125]
        @test snap.logical_locations == [0 1; 0 0; 0 0]
        for m in 1:2, prim in 1:8, c in (0, 7, 63)
            i = c % nx + 1; j = (c ÷ nx) % nx + 1; k = c ÷ nx^2 + 1
            raw = (m == 2 && prim == 5 && c == 7) ? 0.0f0 : cellval(m, prim, c)
            want = prim == 5 ? Float32(Float64(raw) / (γ - 1.0)) : raw
            @test b.data[prim, i, j, k, m] === want
        end
        @test pgas_from_eint(Float64(b.data[5, 2, 1, 1, 1]), γ) ≈ Float64(cellval(1, 5, 1)) rtol = 2e-7
        s = primitives(snap, SVector(0.0, 0.5, 0.5, 0.5))
        @test s.inside && s.rho > 0 && isfinite(s.eint)
        nogas = write_bin(joinpath(dir, "nogas.bin"), ["dens", "velx"])
        @test_throws "missing GRMHD var" athenak_bin_snapshot(nogas; gamma = γ)
    end
end

@testset "Z4c resample: one stencil per node == per-variable sample_var" begin
    zvars = ["z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
             "z4c_gzz", "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz"]
    nx = 5; nv = length(zvars)
    # a coarse root [-4,4]³ (level 0) and a finer nested block [-1,1]³ (level 1)
    x1min = [-4.0, -1.0]; x1max = [4.0, 1.0]
    level = [0, 1]
    data = Array{Float32,5}(undef, nx, nx, nx, nv, 2)
    for b in 1:2, v in 1:nv, k in 1:nx, j in 1:nx, i in 1:nx
        w = x1max[b] - x1min[b]
        cx = x1min[b] + (i - 0.5) * w / nx
        cy = x1min[b] + (j - 0.5) * w / nx
        cz = x1min[b] + (k - 0.5) * w / nx
        val = v == 1 ? 0.4 + 0.3 * sin(0.3cx) * cos(0.2cy) :
                       0.2 * sin(0.17cx * v) + 0.1 * cos(cy + v) + 0.03cz * v
        data[i, j, k, v, b] = Float32(val)
    end
    s = AthenaKBin(2, nx, nx, nx, zvars, level, x1min, x1max, copy(x1min), copy(x1max),
                   copy(x1min), copy(x1max), data, 0.0)
    half = 6.0; n = 20
    grid = resample_z4c_adm(s, half, n)
    @test grid isa ADMGrid{Float64,Array{Float64,4}} && grid.alpha_min == 0.15
    ax = range(-half, half; length = n)
    @test (grid.x0, grid.dx, grid.nx) == (first(ax), step(ax), n)
    col = Dict(name => findfirst(==(name), zvars) for name in zvars)
    saw_offmesh = false
    nbad = 0
    for k in 1:n, j in 1:n, i in 1:n
        x = ax[i]; y = ax[j]; z = ax[k]
        d = grid.data[:, i, j, k]
        chi = sample_var(s, col["z4c_chi"], x, y, z)
        if isnan(chi)
            saw_offmesh = true
            nbad += d != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        else
            psi4 = max(chi, 1.0e-12)^(4.0 / -4.0)
            want = [sample_var(s, col[v], x, y, z) for v in ("z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz")]
            append!(want, [psi4 * sample_var(s, col[v], x, y, z)
                           for v in ("z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz")])
            nbad += d != want
        end
    end
    @test nbad == 0 && saw_offmesh
    @test isnan(sample_var(s, 1, 100.0, 0.0, 0.0))
    @test !isnan(sample_var(s, 1, 0.0, 0.0, 0.0))
    # the finest block owns the centre: its cell values, not the root's
    c = 1 - 0.5 * 2.0 / nx
    @test sample_var(s, 8, -c, -c, -c) == Float64(data[1, 1, 1, 8, 2])
end

@testset "z4c_grid_spacetime returns the grid and the dump" begin
    zvars = ["z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
             "z4c_gzz", "z4c_alpha", "z4c_betax", "z4c_betay", "z4c_betaz"]
    value(b, v, c) = Float32(v == 1 ? 0.5 : v in (2, 5, 7) ? 1.0 : v == 8 ? 0.75 : 0.0)
    mktempdir() do dir
        path = write_bin(joinpath(dir, "z4c.bin"), zvars; value)
        grid, s = z4c_grid_spacetime(path, 1.0, 9; alpha_min = 0.2)
        @test grid isa ADMGrid && s isa AthenaKBin && grid.alpha_min == 0.2
        @test grid.data == resample_z4c_adm(s, 1.0, 9; alpha_min = 0.2).data
        on = grid.data[:, 6, 7, 7]                  # node (0.25, 0.5, 0.5), on block 1
        @test on[1] == 0.75 && on[5] == on[8] == on[10] == 2.0 && on[2] == on[6] == 0.0
        @test grid.data[:, 1, 1, 1] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    end
end

end
