# The `.athdf` reader on synthetic files in AthenaK's on-disk layout: the `eint` layout,
# the `mhd_w_bcc` press layout (`eint = press/(γ-1)` rounded once, temperature dropped,
# non-finite cells zeroed, γ required), the dump time, and a path-backed snapshot series
# (lazy loads, least-recently-used eviction, mesh sharing between frames).
using Test, BLP, StaticArrays
using HDF5: h5open, write_attribute
using BLP.Data: AthenaKSnapshot, SnapshotSeries, SeriesProvider, get_frame, window, advance,
                FrameWindow, LeafTable, PRIM_NAMES
using BLP.Matter: primitives, domain_radius
const Decks = Main.Decks

# Two blocks tiling [0,2]×[0,1]×[0,1] in AthenaK's on-disk layout (C order, as HDF5.jl
# reverses it): uov (nvar, nmb, nk, nj, ni), B (3, nmb, nk, nj, ni), x?f (nmb, n+1),
# x?v (nmb, n), Levels (nmb), LogicalLocations (nmb, 3).
cellval(b, v, i, j, k) = Float32(1000 * b + 100 * v + i + 4 * (j - 1) + 16 * (k - 1))

function write_athdf(path, uovnames; nx = 4, poison = false, time = 2.0, scale = 1.0f0)
    nmb = 2
    nu = length(uovnames)
    uov = Array{Float32,5}(undef, nx, nx, nx, nmb, nu)
    B = Array{Float32,5}(undef, nx, nx, nx, nmb, 3)
    for v in 1:nu, b in 1:nmb, k in 1:nx, j in 1:nx, i in 1:nx
        uov[i, j, k, b, v] = (v == 1 ? scale : 1.0f0) * cellval(b, v, i, j, k)
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
        time === nothing || write_attribute(h, "Time", time)
    end
    return path
end

const EINT = ("dens", "velx", "vely", "velz", "eint")

@testset "athdf reader" begin

@testset "press + temperature layout" begin
    γ = 13.0 / 9.0
    mktempdir() do dir
        path = write_athdf(joinpath(dir, "press.athdf"),
                           ("dens", "velx", "vely", "velz", "press", "temperature"); poison = true)
        @test_throws "gamma" AthenaKSnapshot(path)
        snap = (@test_logs (:warn, r"zeroed 1 non-finite") AthenaKSnapshot(path; gamma = γ))
        b = snap.blocks
        @test (b.ni, b.nj, b.nk, b.nmb) == (4, 4, 4, 2)
        @test snap.names == collect(PRIM_NAMES) && snap.time == 2.0
        for m in 1:2, (i, j, k) in ((1, 1, 1), (3, 2, 4), (4, 4, 4))
            for p in 1:4
                @test b.data[p, i, j, k, m] === cellval(m, p, i, j, k)
            end
            @test b.data[5, i, j, k, m] === Float32(Float64(cellval(m, 5, i, j, k)) / (γ - 1))
            for p in 6:8
                @test b.data[p, i, j, k, m] === cellval(m, 10 + p - 5, i, j, k)
            end
        end
        @test b.data[5, 2, 1, 1, 2] === 0.0f0                  # non-finite press zeroed
        @test all(isfinite, b.data)
        s = primitives(snap, SVector(0.0, 0.5, 0.5, 0.5))
        @test s.inside && s.rho > 0 && isfinite(s.eint)
    end
end

@testset "eint layout, geometry and time" begin
    mktempdir() do dir
        path = write_athdf(joinpath(dir, "eint.athdf"), EINT)
        snap = AthenaKSnapshot(path)
        b = snap.blocks
        @test b.data[5, 3, 2, 1, 2] === cellval(2, 5, 3, 2, 1)
        @test b.data[7, 3, 2, 1, 2] === cellval(2, 12, 3, 2, 1)
        @test AthenaKSnapshot(path; gamma = 5 / 3).blocks.data == b.data   # gamma ignored
        @test b.lo[:, 2] == [1.0, 0.0, 0.0] && b.hi[:, 2] == [2.0, 1.0, 1.0]
        @test b.x_left[:, 2] == [1.125, 0.125, 0.125] && b.dx[:, 1] == [0.25, 0.25, 0.25]
        @test snap.levels == [0, 0] && snap.logical_locations == [0 1; 0 0; 0 0]
        @test b.locate isa LeafTable
        @test isnan(AthenaKSnapshot(write_athdf(joinpath(dir, "notime.athdf"), EINT;
                                                time = nothing)).time)
        bad = write_athdf(joinpath(dir, "bad.athdf"), ("dens", "velx", "vely", "velz", "entropy"))
        @test_throws "unexpected AthenaK primitive order" AthenaKSnapshot(bad; gamma = 5 / 3)
    end
end

@testset "path-backed series" begin
    mktempdir() do dir
        paths = [write_athdf(joinpath(dir, "f$k.athdf"), EINT; scale = Float32(k), time = 10.0k)
                 for k in 1:3]
        ser = SnapshotSeries(paths, [10.0, 20.0, 30.0])
        @test ser.t_obs == 30.0 && isempty(ser.frames)
        @test_throws ErrorException SnapshotSeries(paths, [10.0, 20.0, 30.0]; t_obs = 0.0)
        @test_throws ErrorException SnapshotSeries(paths, [10.0, 30.0, 20.0])
        @test_throws ErrorException SnapshotSeries(paths, [10.0, 20.0, 30.0]; maxslots = 1)
        @test_throws ErrorException SnapshotSeries(paths[1:2], [10.0, 20.0, 30.0])
        f1 = get_frame(ser, 1)
        f2 = get_frame(ser, 2)
        @test f1.time == 10.0 && f2.time == 20.0
        # the second frame shares the first frame's mesh objects
        @test f2.blocks.locate === f1.blocks.locate && f2.blocks.lo === f1.blocks.lo &&
              f2.blocks.x_left === f1.blocks.x_left
        @test get_frame(ser, 1) === f1 && ser.order == [2, 1]
        f3 = get_frame(ser, 3)                                    # evicts frame 2
        @test sort(collect(keys(ser.frames))) == [1, 3] && ser.order == [1, 3]
        @test f3.blocks.locate === f1.blocks.locate
        @test_throws ErrorException get_frame(ser, 4)
        x = SVector(-5.0, 0.6, 0.4, 0.3)                          # emitted at t = 25
        s = primitives(SeriesProvider(ser), x)
        r2 = primitives(get_frame(ser, 2), x).rho; r3 = primitives(f3, x).rho
        @test s.rho == 0.5 * r2 + 0.5 * r3
        @test domain_radius(SeriesProvider(ser)) == Inf
        # a window over loaded frames, advanced by one frame
        win = window(ser, 1)
        @test win.t_a == 10.0 && win.inv_dt == 0.1 && win.t_obs == 20.0
        win2 = advance(win, ser, 2)
        @test win2.a === win.b && win2.t_a == 20.0 && win2.t_obs == 30.0
        @test advance(win2, ser, 3).inv_dt == 0.0
    end
end

end
