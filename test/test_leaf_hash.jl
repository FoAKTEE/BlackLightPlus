# Leaf-block locate on meshes too deep for the dense leaf table. AthenaK nested-box
# dumps (a refined box per level around the binary) reach a finest-block grid of
# 1024³–2048³, past the dense table's cell cap, and fell back to the linear scan over
# every meshblock per sample. The fallback is a per-level hash of logical locations;
# it must return the scan's block at every point off a block face, and 0 off the mesh.
using Test, StaticArrays, Random
using BlackLightPlus, BlackLightPlus.Simulation
using BlackLightPlus.Simulation: _find_meshblock_scan

# Leaf-only nested mesh in the AthenaK layout of the binary runs: domain ±4096, four
# root blocks per axis, and per level L a box ±4096/2^L tiled by 4³ blocks of width
# 2048/2^L whose central 2³ are refined to level L+1 (the finest level keeps all 64).
function _nested_snap(nlev)
    lo = SVector{3,Float64}[]; hi = SVector{3,Float64}[]; lv = Int[]
    dom = -4096.0
    for L in 0:nlev
        w = 2048.0 / 2^L
        b0 = -4096.0 / 2^L
        for k in 0:3, j in 0:3, i in 0:3
            L < nlev && all(in((1, 2)), (i, j, k)) && continue      # covered by level L+1
            l = SVector(b0 + i * w, b0 + j * w, b0 + k * w)
            push!(lo, l); push!(hi, l .+ w); push!(lv, L)
        end
    end
    nmb = length(lv)
    ll = Matrix{Int}(undef, 3, nmb)
    for m in 1:nmb, ax in 1:3
        ll[ax, m] = round(Int, (lo[m][ax] - dom) / (hi[m][ax] - lo[m][ax]))
    end
    n = 2
    data = zeros(Float32, n, n, n, nmb, 8)
    for m in 1:nmb
        data[:, :, :, m, 1] .= m
    end
    dx = [(hi[m] - lo[m]) ./ n for m in 1:nmb]
    AthenaKSnapshot(data, lo, hi, [lo[m] + dx[m] / 2 for m in 1:nmb], dx, lv, ll,
                    ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                    n, n, n, nmb)
end

@testset "deep nested mesh: hashed locate == scan" begin
    snap = _nested_snap(9)                       # finest block width 4: 2048³ finest grid
    @test leaf_table(snap) === nothing           # past the dense-table cap
    @test BlackLightPlus.Simulation._leaf_hash(snap) !== nothing   # hashed fallback built
    rng = Xoshiro(20260924)
    npt = 40_000
    agree = 0
    for _ in 1:npt
        r = exp(rand(rng) * log(6000.0))         # log-uniform radius: every level is hit
        v = randn(rng, 3); v = v / sqrt(sum(abs2, v))
        x, y, z = r .* v
        agree += meshblock_index(snap, x, y, z) == _find_meshblock_scan(snap, x, y, z)
    end
    @test agree == npt
    # every block is found at its own centre, and the sampler reads it
    for m in 1:snap.nmb
        c = (snap.ext_lo[m] + snap.ext_hi[m]) / 2
        @test meshblock_index(snap, c...) == m
    end
    s = sample_primitives(snap, SVector(0.0, 0.3, -0.2, 0.1))
    @test s.inside && s.rho == Float64(meshblock_index(snap, 0.3, -0.2, 0.1))
    # off the mesh
    @test meshblock_index(snap, 5000.0, 0.0, 0.0) == 0
    @test meshblock_index(snap, 0.0, -4097.0, 0.0) == 0
    @test !sample_primitives(snap, SVector(0.0, 0.0, 0.0, 9000.0)).inside
end
