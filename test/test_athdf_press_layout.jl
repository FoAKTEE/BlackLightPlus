# AthenaK .athdf reader on the mhd_w_bcc primitive layout
# (dens velx vely velz press temperature | bcc1 bcc2 bcc3): the GRMHD dumps written with
# `use_e = true` carry gas pressure plus a temperature column in `uov`, not eint. The
# reader maps press -> eint = press/(gamma-1) (Float64, rounded once to Float32), drops
# temperature, zeroes non-finite cells, and needs gamma for that layout; the render
# loader passes the run's gamma for .athdf paths as it does for .bin.
using Test, StaticArrays, HDF5
using BlackLightPlus
using BlackLightPlus.Simulation

# Synthetic two-block athdf in AthenaK's on-disk layout (C order, as HDF5.jl reverses it):
# uov (nvar, nmb, nk, nj, ni), B (3, nmb, nk, nj, ni), x?f (nmb, n+1), x?v (nmb, n),
# Levels (nmb), LogicalLocations (nmb, 3).
function _write_athdf(path, uovnames; nx = 4, poison = false)
    nmb = 2
    nu = length(uovnames)
    cellval(b, v, i, j, k) = Float32(1000 * b + 100 * v + i + 4 * (j - 1) + 16 * (k - 1))
    uov = Array{Float32,5}(undef, nx, nx, nx, nmb, nu)
    B = Array{Float32,5}(undef, nx, nx, nx, nmb, 3)
    for v in 1:nu, b in 1:nmb, k in 1:nx, j in 1:nx, i in 1:nx
        uov[i, j, k, b, v] = cellval(b, v, i, j, k)
    end
    for v in 1:3, b in 1:nmb, k in 1:nx, j in 1:nx, i in 1:nx
        B[i, j, k, b, v] = cellval(b, 10 + v, i, j, k)
    end
    poison && (uov[2, 1, 1, 2, 5] = NaN32)
    x1f = zeros(nx + 1, nmb); x2f = similar(x1f); x3f = similar(x1f)
    x1v = zeros(nx, nmb); x2v = similar(x1v); x3v = similar(x1v)
    for b in 1:nmb
        x1f[:, b] .= range(b - 1, b; length = nx + 1)
        x2f[:, b] .= range(0, 1; length = nx + 1)
        x3f[:, b] .= range(0, 1; length = nx + 1)
        x1v[:, b] .= (x1f[1:end-1, b] .+ x1f[2:end, b]) ./ 2
        x2v[:, b] .= (x2f[1:end-1, b] .+ x2f[2:end, b]) ./ 2
        x3v[:, b] .= (x3f[1:end-1, b] .+ x3f[2:end, b]) ./ 2
    end
    h5open(path, "w") do h
        h["uov"] = uov; h["B"] = B
        h["x1f"] = x1f; h["x2f"] = x2f; h["x3f"] = x3f
        h["x1v"] = x1v; h["x2v"] = x2v; h["x3v"] = x3v
        h["Levels"] = Int32[0, 0]
        h["LogicalLocations"] = Int64[0 1; 0 0; 0 0]
        write_attribute(h, "VariableNames", [uovnames...; "bcc1"; "bcc2"; "bcc3"])
        write_attribute(h, "Time", 2.0)
    end
    return cellval
end

@testset "athdf reader: press + temperature layout" begin
    γ = 13.0 / 9.0
    mktempdir() do dir
        path = joinpath(dir, "press.athdf")
        cellval = _write_athdf(path, ("dens", "velx", "vely", "velz", "press", "temperature");
                               poison = true)
        # the press layout needs the adiabatic index
        @test_throws "gamma" AthenaKSnapshot(path)
        snap = AthenaKSnapshot(path; gamma = γ)
        @test (snap.ni, snap.nj, snap.nk, snap.nmb) == (4, 4, 4, 2)
        @test snap.variable_names == ["dens", "velx", "vely", "velz", "eint",
                                      "bcc1", "bcc2", "bcc3"]
        # dens/vel pass through; press slot holds eint; temperature dropped; B follows
        for b in 1:2, (i, j, k) in ((1, 1, 1), (3, 2, 4), (4, 4, 4))
            for p in 1:4
                @test snap.data[i, j, k, b, p] === cellval(b, p, i, j, k)
            end
            @test snap.data[i, j, k, b, 5] === Float32(Float64(cellval(b, 5, i, j, k)) / (γ - 1))
            for p in 6:8
                @test snap.data[i, j, k, b, p] === cellval(b, 10 + p - 5, i, j, k)
            end
        end
        @test snap.data[2, 1, 1, 2, 5] === 0.0f0                # non-finite press zeroed
        @test all(isfinite, snap.data)
        s = sample_primitives(snap, SVector(0.0, 0.5, 0.5, 0.5))
        @test s.inside && s.rho > 0 && isfinite(s.eint)
        # the render loader threads gamma through for .athdf paths
        snap2 = BlackLightPlus._load_athenak(path, γ)
        @test snap2.data == snap.data
    end
end

@testset "athdf reader: eint layout unchanged" begin
    mktempdir() do dir
        path = joinpath(dir, "eint.athdf")
        cellval = _write_athdf(path, ("dens", "velx", "vely", "velz", "eint"))
        snap = AthenaKSnapshot(path)
        @test snap.data[3, 2, 1, 2, 5] === cellval(2, 5, 3, 2, 1)
        @test snap.data[3, 2, 1, 2, 7] === cellval(2, 12, 3, 2, 1)
        # gamma is accepted and ignored when eint is stored directly
        @test AthenaKSnapshot(path; gamma = 5 / 3).data == snap.data
    end
end
